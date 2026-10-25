//! # `ragdoll` — Modern SIMD-First Multimedia CLI
//!
//! Subcommands (`0022` / `ADR-0008`):
//! - `ragdoll probe <FILE> [--gop-map] [--json]`
//! - `ragdoll remux <IN> <OUT> [--faststart] [--json]`
//! - `ragdoll transcode [<IN> <OUT>] [--video h264|rgdi] [--scale WxH] [--filter lanczos3|mitchell|bilinear|spline36] [--qp N] [--threads N] [--recipe file.toml] [--json]`
//! - `ragdoll extract <IN> [-o <OUT> | <OUT>] [--at <SEC>] [--thumbnails <N>] [--grid <CxR>] [--json]`
//! - `ragdoll metrics <REF> <DIST> [--json]`
//! - `ragdoll synth <OUT> [--resolution WxH] [--fps N] [--duration SEC] [--json]`
//! - `ragdoll bench [--compare-ffmpeg] [--json]`
//! - `ragdoll self-test`

use std::env;
use std::hint::black_box;
use std::io::BufWriter;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use ragdoll_codec::{
    FlacDecoder, FlacEncoder, H264Decoder, H264Encoder, H264Profile, RateControlMode,
};
use ragdoll_core::{
    AudioFrame, BitReader, BitWriter, CodecId, ColorRange, Colorimetry, CpuTopology, FramePool,
    HardwareMetrics, MatrixCoefficients, PacketFlags, PacketSlice, PixelFormat, Plane, PowerMode,
    ProcessRusageSnapshot, RagdollError, Rational64, Result, SampleFormat, Timestamp, TrackInfo,
    TrackType, VideoFrame, VideoTrackParams, crc16_flac, crc32_mpeg2, escape_rbsp,
    measure_hardware_metrics, optimal_plane_stride, parse_macos_time_l_stderr,
    set_global_power_mode, unescape_rbsp, unescape_rbsp_scalar, with_power_mode,
};
use ragdoll_format::{
    MkvDemuxer, Mp4Demuxer, Mp4Muxer, Y4mMuxer, build_avcc_from_sps_pps, is_stdout_path,
    transmux_mp4_to_mkv, transmux_mp4_to_mpegts,
};
use ragdoll_pipeline::{
    CutOptions, PackageOptions, PipelineRecipe, PipelineTelemetry, ProbeReport, RotationMode,
    TranscodeOptions, concat_media, cut_media, evaluate_media_metrics, execute_recipe,
    extract_media, package_abr, parallel_gop_map, parallel_scale_frames, parse_scale_filter,
    probe_file_detailed, remux_file, remux_file_with_format, synth_media, synth_video_frame,
    transcode_file, transcode_file_with_options,
};
use ragdoll_pix::{
    PolyphaseFilterBank, ScaleFilter, YuvRgbMatrix, dither_plane_u16le_to_u8_bayer,
    dither_plane_u16le_to_u8_bayer_scalar, evaluate_frame_quality, hscale_row_8tap_q14,
    hscale_row_8tap_q14_scalar, nv12_to_rgb24, nv12_to_rgb24_scalar, rgb24_to_yuv420p,
    rgb24_to_yuv420p_scalar, scale_plane, yuv420p_to_rgb24, yuv420p_to_rgb24_scalar,
    yuv420p_to_rgba8, yuv420p_to_rgba8_scalar,
};
use ragdoll_simd::{
    SimdLevel, find_annex_b_start_code, find_annex_b_start_code_scalar, flac_eval_fixed_orders_l1,
    flac_eval_fixed_orders_l1_scalar, forward_dct4x4, forward_dct4x4_scalar, inverse_idct4x4,
    inverse_idct4x4_scalar, sad_16x16_u8, sad_16x16_u8_scalar, sad_plane_u8, sad_plane_u8_scalar,
    satd_8x8_u8, sse_plane_u8, sse_plane_u8_scalar, ssim_4x4xn_u8, ssim_4x4xn_u8_scalar,
    ssim_plane_8bit, ssim_to_db,
};

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if let Err(err) = run_cli(&args) {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

pub fn run_cli(args: &[String]) -> Result<()> {
    let mut filtered = Vec::with_capacity(args.len());
    let mut power_mode = PowerMode::Performance;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--power-mode" | "-P" if i + 1 < args.len() => {
                i += 1;
                power_mode = PowerMode::parse(&args[i]);
            }
            "--battery-saver" | "--efficiency" => {
                power_mode = PowerMode::Efficiency;
            }
            _ => filtered.push(args[i].clone()),
        }
        i += 1;
    }
    set_global_power_mode(power_mode);

    if filtered.is_empty() || matches!(filtered[0].as_str(), "-h" | "--help" | "help") {
        print_help();
        return Ok(());
    }
    if matches!(filtered[0].as_str(), "-V" | "--version" | "version") {
        let topo = CpuTopology::detect();
        println!(
            "ragdoll 0.1.0 ({} | {} P-cores + {} E-cores)",
            SimdLevel::detect().name(),
            topo.p_cores,
            topo.e_cores
        );
        return Ok(());
    }

    let sub = filtered[0].as_str();
    let rest = &filtered[1..];
    match sub {
        "probe" => cmd_probe(rest),
        "remux" => cmd_remux(rest),
        "transcode" => cmd_transcode(rest),
        "cut" => cmd_cut(rest),
        "concat" => cmd_concat(rest),
        "package" => cmd_package(rest),
        "extract" => cmd_extract(rest),
        "metrics" => cmd_metrics(rest),
        "synth" => cmd_synth(rest),
        "bench" => cmd_bench(rest),
        "self-test" | "selftest" => {
            run_self_test();
            Ok(())
        }
        other => Err(RagdollError::InvalidSyntax("unknown subcommand").with_context(other)),
    }
}

trait ErrorContext {
    fn with_context(self, detail: &str) -> Self;
}

impl ErrorContext for RagdollError {
    fn with_context(self, detail: &str) -> Self {
        RagdollError::InvalidData(format!("{self}: {detail}"))
    }
}

fn print_help() {
    let simd = SimdLevel::detect();
    let topo = CpuTopology::detect();
    println!("================================================================================");
    println!("  RAGDOLL v0.1.0 — 100% Safe Rust, Explicit SIMD, Zero-Copy Media Engine");
    println!("  Hardware SIMD : {}", simd.name());
    println!(
        "  CPU Topology  : {} logical ({} P-Cores + {} E-Cores)",
        topo.total_cores, topo.p_cores, topo.e_cores
    );
    println!("================================================================================\n");
    println!("USAGE:");
    println!(
        "  ragdoll <SUBCOMMAND> [OPTIONS] [--power-mode performance|efficiency|balanced | --battery-saver]\n"
    );
    println!("SUBCOMMANDS:");
    println!("  probe <FILE> [--gop-map] [--json]");
    println!("      Sub-millisecond container/codec inspection, CICP colorimetry,");
    println!("      bitrate sparkline, and ASCII GOP waterfall visualization.");
    println!();
    println!("  remux <IN> <OUT> [--faststart] [--json]");
    println!(
        "      Zero-copy mmap -> writev remuxing across .mp4, .mkv, .ts, .flac, .ogg, .y4m, and .wav."
    );
    println!();
    println!(
        "  transcode <IN> <OUT> [--video h264|rgdi] [--audio aac|opus|flac|copy|none] [--sample-rate HZ] [--channels N] [--loudness LUFS] [--true-peak DBTP] [--scale WxH] [--filter lanczos3|mitchell|bilinear|spline36] [--crop W:H:X:Y] [--auto-crop] [--pad W:H:X:Y] [--rotate 90|180|270|hflip|vflip] [--fps NUM[/DEN]] [--overlay FILE] [--tonemap] [--qp N] [--threads N] [--power-mode MODE] [--recipe file.toml] [--json]"
    );
    println!(
        "      Work-stealing GOP-parallel video & audio transcoding, spatial/HDR filters, Kaiser polyphase resampling, EBU R128 loudness normalization, or TOML recipe execution."
    );
    println!();
    println!("  cut <IN> <OUT> [--from <TIME>] [--to <TIME>] [--keyframes-only] [--qp N] [--json]");
    println!(
        "      Smart-GOP frame-accurate non-linear editing: re-encodes only boundary GOPs and zero-copy copies interior GOPs."
    );
    println!();
    println!("  concat <IN1> <IN2> ... -o <OUT> [--json]");
    println!(
        "      Zero-copy multi-segment splicing with monotonic 128-bit PTS/DTS offset normalization."
    );
    println!();
    println!("  package <IN> -o <OUT_DIR> [--ladder SPEC] [--segment-duration SEC] [--json]");
    println!(
        "      One-command CMAF fMP4 HLS (master.m3u8) + DASH (manifest.mpd) + WebVTT trick-play ABR packager."
    );
    println!();
    println!("  extract <IN> [-o <OUT>] [--at <SEC>] [--thumbnails <N> --grid <CxR>] [--json]");
    println!(
        "      Fast keyframe seek and single-frame or contact-sheet (.png/.jpg/.ppm/.y4m) extraction."
    );
    println!();
    println!("  metrics <REF> <DIST> [--json]");
    println!(
        "      SIMD perceptual video quality analyzer (PSNR, 8x8 SSIM, 5-Scale MS-SSIM, VMAF-Lite)."
    );
    println!();
    println!("  synth <OUT> [--resolution WxH] [--fps N] [--duration SEC] [--json]");
    println!(
        "      Procedural SIMD test media generator (SMPTE bars, zone-plates, stereo harmonies)."
    );
    println!();
    println!("  bench [--compare-ffmpeg] [--json]");
    println!(
        "      Automated speed + hardware PMU instruction/cycle/energy benchmark suite across 7 workloads."
    );
    println!();
    println!("  self-test");
    println!("      Run SIMD vs. scalar bit-exact differential kernel verification.");
}

/// Renders an 8-level Unicode block sparkline (`▁▂▃▄▅▆▇█`) over packet byte sizes.
pub fn render_sparkline(sizes: &[(bool, u32)], max_cols: usize) -> String {
    if sizes.is_empty() {
        return String::new();
    }
    const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let cols = sizes.len().min(max_cols.max(1));
    let mut bucket_vals = Vec::with_capacity(cols);
    for c in 0..cols {
        let start = c * sizes.len() / cols;
        let end = ((c + 1) * sizes.len() / cols)
            .max(start + 1)
            .min(sizes.len());
        let avg = sizes[start..end]
            .iter()
            .map(|&(_, sz)| sz as u64)
            .sum::<u64>()
            / ((end - start) as u64);
        bucket_vals.push(avg);
    }
    let max_v = bucket_vals.iter().copied().max().unwrap_or(1).max(1);
    let min_v = bucket_vals.iter().copied().min().unwrap_or(0);
    let span = (max_v - min_v).max(1);

    bucket_vals
        .into_iter()
        .map(|v| {
            let idx = (((v.saturating_sub(min_v)) * 7) / span).min(7) as usize;
            BARS[idx]
        })
        .collect()
}

/// Renders an ASCII GOP Waterfall diagram showing `[I]` keyframe and `[P]` inter-frame byte distributions.
pub fn render_gop_waterfall(sizes: &[(bool, u32)], max_frames: usize) -> String {
    if sizes.is_empty() {
        return String::new();
    }
    let limit = sizes.len().min(max_frames);
    let max_sz = sizes[..limit]
        .iter()
        .map(|&(_, s)| s)
        .max()
        .unwrap_or(1)
        .max(1);
    let mut out = String::new();
    let mut gop_idx = 0usize;

    for (f_idx, &(is_kf, sz)) in sizes[..limit].iter().enumerate() {
        if is_kf {
            gop_idx += 1;
        }
        let bar_len = ((sz as u64 * 32) / (max_sz as u64)).max(1) as usize;
        let ch = if is_kf { '█' } else { '▒' };
        let bar: String = std::iter::repeat_n(ch, bar_len).collect();
        let tag = if is_kf { "I-KEY" } else { "P-INT" };
        out.push_str(&format!(
            "  #{f_idx:03} [GOP {gop_idx:02} {tag}] {:>7} B |{bar}\n",
            sz
        ));
    }
    if sizes.len() > limit {
        out.push_str(&format!(
            "  ... ({} additional packets omitted)\n",
            sizes.len() - limit
        ));
    }
    out
}

fn cmd_probe(args: &[String]) -> Result<()> {
    let mut file_path = None;
    let mut show_gop = false;
    let mut json = false;

    for arg in args {
        match arg.as_str() {
            "--gop-map" | "-g" => show_gop = true,
            "--json" | "-j" => json = true,
            s if (s == "-" || !s.starts_with('-')) && file_path.is_none() => {
                file_path = Some(s.to_string())
            }
            _ => {}
        }
    }
    let path = file_path.ok_or(RagdollError::InvalidSyntax("probe requires <FILE>"))?;
    let rep: ProbeReport = probe_file_detailed(&path)?;

    if json {
        print_probe_json(&rep);
        return Ok(());
    }

    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ RAGDOLL PROBE : {:<60} │", rep.path);
    println!("├──────────────────────────────────────────────────────────────────────────────┤");
    println!(
        "│ Container     : {:<18} File Size : {:>12} B ({:.2} MB)      │",
        rep.container,
        rep.file_size,
        (rep.file_size as f64) / (1024.0 * 1024.0)
    );
    println!(
        "│ Probe Latency : {:>8.2} µs         GOP Count : {:>12}                 │",
        rep.probe_time_us, rep.gop_count
    );
    if let Some(prof) = &rep.h264_profile {
        println!("│ H.264 Profile : {:<60} │", prof);
    }
    println!("├──────────────────────────────────────────────────────────────────────────────┤");
    for t in &rep.tracks {
        if let Some(v) = &t.video {
            println!(
                "│ Track #{:<2} [Video] codec={} {}x{} @ {:.3} fps ({}/{}) fmt={}        │",
                t.index,
                t.codec_id,
                v.width,
                v.height,
                v.frame_rate.as_f64(),
                v.frame_rate.num,
                v.frame_rate.den,
                v.pixel_format
            );
            println!(
                "│   Colorimetry : primaries={:?}, transfer={:?}, matrix={:?}, range={:?}",
                v.colorimetry.primaries,
                v.colorimetry.transfer,
                v.colorimetry.matrix,
                v.colorimetry.range
            );
            println!(
                "│   Samples     : {} packets ({} bytes)",
                t.sample_count, t.total_bytes
            );
        }
        if let Some(a) = &t.audio {
            println!(
                "│ Track #{:<2} [Audio] codec={} {} Hz, {} ch ({:?}), fmt={} ({}-bit) │",
                t.index,
                t.codec_id,
                a.sample_rate,
                a.channels,
                a.channel_layout,
                a.sample_format,
                a.bits_per_coded_sample
            );
            println!(
                "│   Samples     : {} frames ({} bytes)",
                t.sample_count, t.total_bytes
            );
        }
        if t.track_type == TrackType::Subtitle {
            println!(
                "│ Track #{:<2} [Subtitle] codec={} cues={} ({} bytes)                            │",
                t.index, t.codec_id, t.sample_count, t.total_bytes
            );
        }
    }

    if !rep.packet_sizes.is_empty() {
        let spark = render_sparkline(&rep.packet_sizes, 56);
        println!(
            "├──────────────────────────────────────────────────────────────────────────────┤"
        );
        println!("│ Bitrate Spark : {spark:<60} │");
    }
    println!("└──────────────────────────────────────────────────────────────────────────────┘");

    if show_gop && !rep.packet_sizes.is_empty() {
        println!("\nASCII GOP Waterfall (I/P Packet Size Distribution):");
        print!("{}", render_gop_waterfall(&rep.packet_sizes, 32));
    }
    Ok(())
}

fn print_probe_json(rep: &ProbeReport) {
    println!("{{");
    println!("  \"path\": \"{}\",", escape_json(&rep.path));
    println!("  \"container\": \"{}\",", escape_json(&rep.container));
    println!("  \"file_size\": {},", rep.file_size);
    println!("  \"probe_time_us\": {:.3},", rep.probe_time_us);
    println!("  \"gop_count\": {},", rep.gop_count);
    if let Some(p) = &rep.h264_profile {
        println!("  \"h264_profile\": \"{}\",", escape_json(p));
    } else {
        println!("  \"h264_profile\": null,");
    }
    println!("  \"tracks\": [");
    for (i, t) in rep.tracks.iter().enumerate() {
        let comma = if i + 1 < rep.tracks.len() { "," } else { "" };
        if let Some(v) = &t.video {
            println!(
                "    {{\"index\": {}, \"type\": \"video\", \"codec\": \"{}\", \"width\": {}, \"height\": {}, \"fps\": {:.4}, \"pixel_format\": \"{}\", \"samples\": {}}}{comma}",
                t.index,
                t.codec_id,
                v.width,
                v.height,
                v.frame_rate.as_f64(),
                v.pixel_format,
                t.sample_count
            );
        } else if let Some(a) = &t.audio {
            println!(
                "    {{\"index\": {}, \"type\": \"audio\", \"codec\": \"{}\", \"sample_rate\": {}, \"channels\": {}, \"sample_format\": \"{}\", \"samples\": {}}}{comma}",
                t.index, t.codec_id, a.sample_rate, a.channels, a.sample_format, t.sample_count
            );
        } else if t.track_type == TrackType::Subtitle {
            println!(
                "    {{\"index\": {}, \"type\": \"subtitle\", \"codec\": \"{}\", \"samples\": {}}}{comma}",
                t.index, t.codec_id, t.sample_count
            );
        }
    }
    println!("  ]");
    println!("}}");
}

fn cmd_remux(args: &[String]) -> Result<()> {
    let mut positional = Vec::new();
    let mut format_override: Option<String> = None;
    let mut json = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json" | "-j" => json = true,
            "--faststart" => {}
            "--format" | "-f" if i + 1 < args.len() => {
                i += 1;
                format_override = Some(args[i].clone());
            }
            s if s == "-" || !s.starts_with('-') => positional.push(s.to_string()),
            _ => {}
        }
        i += 1;
    }
    if positional.len() < 2 {
        return Err(RagdollError::InvalidSyntax("remux requires <IN> <OUT>"));
    }
    let tel = remux_file_with_format(&positional[0], &positional[1], format_override.as_deref())?;
    if !is_stdout_path(&positional[1]) {
        print_telemetry(&tel, json);
    }
    Ok(())
}

fn cmd_transcode(args: &[String]) -> Result<()> {
    let mut positional = Vec::new();
    let mut opts = TranscodeOptions::default();
    let mut recipe_path: Option<String> = None;
    let mut crf_opt: Option<f32> = None;
    let mut bitrate_kbps: Option<u32> = None;
    let mut maxrate_kbps: Option<u32> = None;
    let mut bufsize_kb: Option<u32> = None;
    let mut json = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json" | "-j" => json = true,
            "--video" | "-c:v" if i + 1 < args.len() => {
                i += 1;
                opts.video_codec = args[i].clone();
            }
            "--profile" | "-profile:v" if i + 1 < args.len() => {
                i += 1;
                opts.h264_profile = H264Profile::parse(&args[i]);
            }
            "--crf" if i + 1 < args.len() => {
                i += 1;
                crf_opt = args[i].parse::<f32>().ok().map(|v| v.clamp(0.0, 51.0));
            }
            "--bitrate" | "-b:v" if i + 1 < args.len() => {
                i += 1;
                bitrate_kbps = parse_bitrate_kbps(&args[i]);
            }
            "--maxrate" if i + 1 < args.len() => {
                i += 1;
                maxrate_kbps = parse_bitrate_kbps(&args[i]);
            }
            "--bufsize" if i + 1 < args.len() => {
                i += 1;
                bufsize_kb = parse_bitrate_kbps(&args[i]);
            }
            "--audio" | "-c:a" if i + 1 < args.len() => {
                i += 1;
                opts.audio_codec = args[i].clone();
            }
            "--sample-rate" | "-ar" if i + 1 < args.len() => {
                i += 1;
                opts.sample_rate = args[i].parse::<u32>().ok();
            }
            "--channels" | "-ac" if i + 1 < args.len() => {
                i += 1;
                opts.channels = args[i].parse::<u8>().ok();
            }
            "--loudness" | "--lufs" if i + 1 < args.len() => {
                i += 1;
                opts.loudness_lufs = args[i].parse::<f64>().ok();
            }
            "--true-peak" | "--dbtp" if i + 1 < args.len() => {
                i += 1;
                opts.true_peak_dbtp = args[i].parse::<f64>().ok();
            }
            "--subs" | "--subtitles" | "-c:s" if i + 1 < args.len() => {
                i += 1;
                opts.subtitle_codec = args[i].clone();
            }
            "--an" => {
                opts.audio_codec = "none".into();
            }
            "--sn" => {
                opts.subtitle_codec = "none".into();
            }
            "--format" | "-f" if i + 1 < args.len() => {
                i += 1;
                opts.format_override = Some(args[i].clone());
            }
            "--scale" | "-s" if i + 1 < args.len() => {
                i += 1;
                opts.scale_dims = parse_dimensions(&args[i]);
            }
            "--filter" if i + 1 < args.len() => {
                i += 1;
                opts.scale_filter = parse_scale_filter(&args[i]);
            }
            "--crop" if i + 1 < args.len() => {
                i += 1;
                opts.crop_spec = Some(args[i].clone());
            }
            "--auto-crop" => {
                opts.auto_crop = true;
            }
            "--pad" if i + 1 < args.len() => {
                i += 1;
                opts.pad_spec = Some(args[i].clone());
            }
            "--rotate" if i + 1 < args.len() => {
                i += 1;
                opts.rotate_mode = Some(RotationMode::parse_spec(&args[i])?);
            }
            "--fps" | "-r:v" if i + 1 < args.len() => {
                i += 1;
                if let Some((n, d)) = args[i].split_once('/') {
                    if let (Ok(num), Ok(den)) = (n.parse::<i64>(), d.parse::<i64>()) {
                        opts.target_fps = Some(Rational64::new(num, den.max(1)));
                    }
                } else if let Ok(num) = args[i].parse::<i64>() {
                    opts.target_fps = Some(Rational64::new(num.max(1), 1));
                }
            }
            "--overlay" if i + 1 < args.len() => {
                i += 1;
                opts.overlay_path = Some(std::path::PathBuf::from(&args[i]));
            }
            "--overlay-xy" if i + 1 < args.len() => {
                i += 1;
                if let Some((xs, ys)) = args[i].split_once(':').or_else(|| args[i].split_once(','))
                    && let (Ok(x), Ok(y)) = (xs.parse::<usize>(), ys.parse::<usize>())
                {
                    opts.overlay_xy = (x, y);
                }
            }
            "--tonemap" => {
                opts.tonemap = true;
            }
            "--qp" if i + 1 < args.len() => {
                i += 1;
                opts.qp = args[i].parse::<u8>().unwrap_or(20).min(51);
            }
            "--threads" | "-t" if i + 1 < args.len() => {
                i += 1;
                opts.threads = args[i].parse::<usize>().unwrap_or(0);
            }
            "--recipe" | "-r" if i + 1 < args.len() => {
                i += 1;
                recipe_path = Some(args[i].clone());
            }
            "--color" if i + 1 < args.len() => {
                i += 1;
            }
            s if s == "-" || !s.starts_with('-') => positional.push(s.to_string()),
            _ => {}
        }
        i += 1;
    }

    if let Some(br) = bitrate_kbps {
        opts.rate_control = Some(RateControlMode::Abr {
            bitrate_kbps: br,
            maxrate_kbps,
            bufsize_kb,
        });
    } else if let Some(crf) = crf_opt {
        opts.rate_control = Some(RateControlMode::Crf {
            crf,
            maxrate_kbps,
            bufsize_kb,
        });
    }

    let tel = if let Some(rp) = recipe_path {
        let toml_text = std::fs::read_to_string(&rp)?;
        let recipe = PipelineRecipe::from_toml(&toml_text)?;
        execute_recipe(&recipe)?
    } else {
        if positional.len() < 2 {
            return Err(RagdollError::InvalidSyntax(
                "transcode requires <IN> <OUT> or --recipe <FILE.toml>",
            ));
        }
        transcode_file_with_options(&positional[0], &positional[1], &opts)?
    };

    if positional.get(1).is_none_or(|p| !is_stdout_path(p)) {
        print_telemetry(&tel, json);
    }
    Ok(())
}

fn cmd_cut(args: &[String]) -> Result<()> {
    let mut positional = Vec::new();
    let mut opts = CutOptions {
        qp: 20,
        ..Default::default()
    };
    let mut json = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json" | "-j" => json = true,
            "--from" | "-ss" if i + 1 < args.len() => {
                i += 1;
                opts.from_sec = parse_time_seconds(&args[i]);
            }
            "--to" | "-to" if i + 1 < args.len() => {
                i += 1;
                opts.to_sec = parse_time_seconds(&args[i]);
            }
            "--keyframes-only" => {
                opts.keyframes_only = true;
            }
            "--qp" if i + 1 < args.len() => {
                i += 1;
                opts.qp = args[i].parse::<u8>().unwrap_or(20).min(51);
            }
            s if !s.starts_with('-') => positional.push(s.to_string()),
            _ => {}
        }
        i += 1;
    }
    if positional.len() < 2 {
        return Err(RagdollError::InvalidSyntax(
            "cut requires <IN> <OUT> [--from <TIME>] [--to <TIME>]",
        ));
    }
    let rep = cut_media(&positional[0], &positional[1], &opts)?;
    if json {
        println!(
            "{{\"operation\":\"cut\",\"output_bytes\":{},\"total_video_packets\":{},\"copied_video_packets\":{},\"reencoded_video_frames\":{},\"audio_packets\":{},\"elapsed_ms\":{:.3}}}",
            rep.output_bytes,
            rep.total_video_packets,
            rep.copied_video_packets,
            rep.reencoded_video_frames,
            rep.audio_packets,
            rep.elapsed_ms
        );
    } else {
        println!(
            "RAGDOLL CUT: {} -> {} | {} video pkts ({} zero-copy copied, {} boundary frames re-encoded), {} audio pkts ({} B in {:.2} ms)",
            positional[0],
            positional[1],
            rep.total_video_packets,
            rep.copied_video_packets,
            rep.reencoded_video_frames,
            rep.audio_packets,
            rep.output_bytes,
            rep.elapsed_ms
        );
    }
    Ok(())
}

fn cmd_concat(args: &[String]) -> Result<()> {
    let mut inputs = Vec::new();
    let mut out_opt: Option<String> = None;
    let mut json = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json" | "-j" => json = true,
            "-o" | "--output" if i + 1 < args.len() => {
                i += 1;
                out_opt = Some(args[i].clone());
            }
            s if !s.starts_with('-') => inputs.push(s.to_string()),
            _ => {}
        }
        i += 1;
    }
    let out_path = if let Some(o) = out_opt {
        o
    } else if inputs.len() >= 2 {
        inputs.pop().unwrap()
    } else {
        return Err(RagdollError::InvalidSyntax(
            "concat requires <IN1> <IN2> ... -o <OUT>",
        ));
    };
    let rep = concat_media(&inputs, &out_path)?;
    if json {
        println!(
            "{{\"operation\":\"concat\",\"segments_merged\":{},\"video_packets\":{},\"audio_packets\":{},\"reencoded\":{},\"output_bytes\":{},\"elapsed_ms\":{:.3}}}",
            rep.segments_merged,
            rep.video_packets,
            rep.audio_packets,
            rep.reencoded,
            rep.output_bytes,
            rep.elapsed_ms
        );
    } else {
        println!(
            "RAGDOLL CONCAT: {} segments -> {} | {} video pkts, {} audio pkts (reencoded={}, {} B in {:.2} ms)",
            rep.segments_merged,
            out_path,
            rep.video_packets,
            rep.audio_packets,
            rep.reencoded,
            rep.output_bytes,
            rep.elapsed_ms
        );
    }
    Ok(())
}

fn cmd_package(args: &[String]) -> Result<()> {
    let mut positional = Vec::new();
    let mut out_dir_opt: Option<String> = None;
    let mut opts = PackageOptions::default();
    let mut json = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json" | "-j" => json = true,
            "-o" | "--output" if i + 1 < args.len() => {
                i += 1;
                out_dir_opt = Some(args[i].clone());
            }
            "--ladder" if i + 1 < args.len() => {
                i += 1;
                opts.ladder_spec = Some(args[i].clone());
            }
            "--segment-duration" | "--seg-dur" if i + 1 < args.len() => {
                i += 1;
                if let Ok(sec) = args[i].parse::<f64>() {
                    opts.segment_duration_sec = sec.max(0.2);
                }
            }
            "--no-thumbnails" => {
                opts.generate_thumbnails = false;
            }
            s if !s.starts_with('-') => positional.push(s.to_string()),
            _ => {}
        }
        i += 1;
    }
    let in_path = positional.first().ok_or(RagdollError::InvalidSyntax(
        "package requires <IN> -o <OUT_DIR>",
    ))?;
    let out_dir = out_dir_opt
        .or_else(|| positional.get(1).cloned())
        .ok_or(RagdollError::InvalidSyntax("package requires -o <OUT_DIR>"))?;
    let rep = package_abr(in_path, &out_dir, &opts)?;
    if json {
        println!(
            "{{\"operation\":\"package\",\"renditions\":{},\"segments_per_rendition\":{},\"master_m3u8\":\"{}\",\"mpd\":\"{}\",\"total_bytes\":{},\"elapsed_ms\":{:.3}}}",
            rep.renditions,
            rep.segments_per_rendition,
            rep.master_m3u8_path.display(),
            rep.mpd_path.display(),
            rep.total_bytes,
            rep.elapsed_ms
        );
    } else {
        println!(
            "RAGDOLL PACKAGE: {} -> {} | {} CMAF renditions x {} segments | HLS={} | DASH={} ({} B in {:.2} ms)",
            in_path,
            out_dir,
            rep.renditions,
            rep.segments_per_rendition,
            rep.master_m3u8_path.display(),
            rep.mpd_path.display(),
            rep.total_bytes,
            rep.elapsed_ms
        );
    }
    Ok(())
}

fn cmd_extract(args: &[String]) -> Result<()> {
    let mut positional = Vec::new();
    let mut out_opt: Option<String> = None;
    let mut at_sec: Option<f64> = None;
    let mut thumb_count: Option<usize> = None;
    let mut grid: (usize, usize) = (4, 4);
    let mut json = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json" | "-j" => json = true,
            "-o" | "--output" if i + 1 < args.len() => {
                i += 1;
                out_opt = Some(args[i].clone());
            }
            "--at" if i + 1 < args.len() => {
                i += 1;
                at_sec = parse_time_seconds(&args[i]);
            }
            "--thumbnails" if i + 1 < args.len() => {
                i += 1;
                thumb_count = args[i].parse::<usize>().ok();
            }
            "--grid" if i + 1 < args.len() => {
                i += 1;
                if let Some((c, r)) = parse_dimensions(&args[i]) {
                    grid = (c, r);
                }
            }
            s if !s.starts_with('-') => positional.push(s.to_string()),
            _ => {}
        }
        i += 1;
    }

    let in_path = positional
        .first()
        .ok_or(RagdollError::InvalidSyntax("extract requires <IN>"))?;
    let out_path =
        out_opt
            .or_else(|| positional.get(1).cloned())
            .ok_or(RagdollError::InvalidSyntax(
                "extract requires -o <OUT> or <OUT>",
            ))?;

    let thumbs = thumb_count.map(|n| (n, grid.0, grid.1));
    let tel = extract_media(in_path, &out_path, at_sec, thumbs)?;
    print_telemetry(&tel, json);
    Ok(())
}

fn cmd_metrics(args: &[String]) -> Result<()> {
    let mut positional = Vec::new();
    let mut json = false;
    for arg in args {
        match arg.as_str() {
            "--json" | "-j" => json = true,
            s if !s.starts_with('-') => positional.push(s.to_string()),
            _ => {}
        }
    }
    if positional.len() < 2 {
        return Err(RagdollError::InvalidSyntax("metrics requires <REF> <DIST>"));
    }
    let (rep, frames, elapsed_ms) = evaluate_media_metrics(&positional[0], &positional[1])?;
    let fps = (frames as f64) / (elapsed_ms / 1000.0).max(1e-9);

    if json {
        println!("{{");
        println!("  \"frames\": {frames},");
        println!("  \"elapsed_ms\": {elapsed_ms:.3},");
        println!("  \"fps\": {fps:.2},");
        println!("  \"psnr_y\": {:.4},", format_finite(rep.psnr_y));
        println!("  \"psnr_u\": {:.4},", format_finite(rep.psnr_u));
        println!("  \"psnr_v\": {:.4},", format_finite(rep.psnr_v));
        println!("  \"psnr_avg\": {:.4},", format_finite(rep.psnr_avg));
        println!(
            "  \"psnr_yuv_611\": {:.4},",
            format_finite(rep.psnr_yuv_611)
        );
        println!("  \"ssim_y\": {:.6},", rep.ssim_y);
        println!("  \"ssim_avg\": {:.6},", rep.ssim_avg);
        println!("  \"ssim_db\": {:.4},", format_finite(rep.ssim_db));
        println!("  \"ms_ssim\": {:.6},", rep.ms_ssim);
        println!("  \"vmaf_lite\": {:.4}", rep.vmaf_lite);
        println!("}}");
    } else {
        println!(
            "┌──────────────────────────────────────────────────────────────────────────────┐"
        );
        println!(
            "│ RAGDOLL PERCEPTUAL QUALITY METRICS ({frames} frames in {elapsed_ms:.2} ms / {fps:.1} fps)"
        );
        println!(
            "├──────────────────────────────────────────────────────────────────────────────┤"
        );
        println!(
            "│ PSNR (dB)     : Y={:.2} dB | U={:.2} dB | V={:.2} dB | Avg={:.2} dB (6:1:1={:.2} dB)",
            rep.psnr_y, rep.psnr_u, rep.psnr_v, rep.psnr_avg, rep.psnr_yuv_611
        );
        println!(
            "│ SSIM (8x8)    : Y={:.5} ({:.2} dB) | Avg={:.5} | 5-Scale MS-SSIM={:.5}",
            rep.ssim_y, rep.ssim_db, rep.ssim_avg, rep.ms_ssim
        );
        println!("│ VMAF-Lite     : {:.2} / 100.0", rep.vmaf_lite);
        println!(
            "└──────────────────────────────────────────────────────────────────────────────┘"
        );
    }
    Ok(())
}

fn cmd_synth(args: &[String]) -> Result<()> {
    let mut out_path: Option<String> = None;
    let mut dims = (1280usize, 720usize);
    let mut fps = 30i64;
    let mut duration = 2.0f64;
    let mut json = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json" | "-j" => json = true,
            "--resolution" | "-s" if i + 1 < args.len() => {
                i += 1;
                if let Some(d) = parse_dimensions(&args[i]) {
                    dims = d;
                }
            }
            "--fps" | "-r" if i + 1 < args.len() => {
                i += 1;
                fps = args[i].parse::<i64>().unwrap_or(30).max(1);
            }
            "--duration" | "-d" if i + 1 < args.len() => {
                i += 1;
                duration = args[i].parse::<f64>().unwrap_or(2.0).max(0.05);
            }
            s if !s.starts_with('-') && out_path.is_none() => out_path = Some(s.to_string()),
            _ => {}
        }
        i += 1;
    }

    let out = out_path.ok_or(RagdollError::InvalidSyntax("synth requires <OUT>"))?;
    let tel = synth_media(&out, dims.0, dims.1, fps, duration)?;
    print_telemetry(&tel, json);
    Ok(())
}

#[derive(Clone, Debug)]
struct BenchRow {
    workload: &'static str,
    unit: &'static str,
    ragdoll_perf_hw: HardwareMetrics,
    ragdoll_perf_rate: f64,
    ragdoll_eff_hw: HardwareMetrics,
    ragdoll_eff_rate: f64,
    ffmpeg_hw: Option<HardwareMetrics>,
    ffmpeg_rate: Option<f64>,
    speedup_perf: Option<f64>,
    speedup_eff: Option<f64>,
}

fn scale_hardware_metrics(mut hw: HardwareMetrics, iters: usize) -> HardwareMetrics {
    let div = (iters.max(1)) as f64;
    if iters > 1 {
        hw.wall_ms /= div;
        hw.user_ms /= div;
        hw.sys_ms /= div;
        hw.cpu_ms /= div;
        hw.instructions = ((hw.instructions as f64) / div).round() as u64;
        hw.cycles = ((hw.cycles as f64) / div).round() as u64;
        hw.energy_mj /= div;
    }
    hw
}

fn run_external_batch_with_telemetry(
    cmds: &[(&str, Vec<String>)],
    iters: usize,
) -> Result<HardwareMetrics> {
    let use_macos_time = cfg!(target_os = "macos") && Path::new("/usr/bin/time").exists();
    let mut total = HardwareMetrics::default();
    let total_iters = iters.max(1);

    for _ in 0..total_iters {
        for (prog, cmd_args) in cmds {
            let t0 = Instant::now();
            if use_macos_time {
                let out = Command::new("/usr/bin/time")
                    .arg("-l")
                    .arg(prog)
                    .args(cmd_args)
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::piped())
                    .output()?;
                assert!(
                    out.status.success(),
                    "{prog} failed: {}",
                    String::from_utf8_lossy(&out.stderr)
                );
                let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let stderr_text = String::from_utf8_lossy(&out.stderr);
                let hw = parse_macos_time_l_stderr(&stderr_text, wall_ms);
                total.wall_ms += hw.wall_ms;
                total.user_ms += hw.user_ms;
                total.sys_ms += hw.sys_ms;
                total.cpu_ms += hw.cpu_ms;
                total.instructions += hw.instructions;
                total.cycles += hw.cycles;
                total.energy_mj += hw.energy_mj;
                if hw.peak_rss_mib > total.peak_rss_mib {
                    total.peak_rss_mib = hw.peak_rss_mib;
                }
                total.pageins += hw.pageins;
                total.idle_wakeups += hw.idle_wakeups;
            } else {
                let status = Command::new(prog)
                    .args(cmd_args)
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status()?;
                assert!(status.success());
                let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
                total.wall_ms += wall_ms;
            }
        }
    }

    let mut scaled = scale_hardware_metrics(total, total_iters);
    if scaled.cycles > 0 {
        scaled.ipc = (scaled.instructions as f64) / (scaled.cycles as f64);
    }
    if scaled.cpu_ms > 0.01 && scaled.cycles > 0 {
        scaled.effective_ghz = (scaled.cycles as f64) / ((scaled.cpu_ms / 1000.0) * 1e9);
    }
    if scaled.wall_ms > 0.001 {
        scaled.avg_power_watts = scaled.energy_mj / scaled.wall_ms;
    }
    Ok(scaled)
}

fn cmd_bench(args: &[String]) -> Result<()> {
    let compare_ffmpeg = args.iter().any(|a| a == "--compare-ffmpeg" || a == "-c");
    let json = args.iter().any(|a| a == "--json" || a == "-j");

    let tmp_dir = env::temp_dir().join(format!("ragdoll_bench_{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir)?;

    let mut rows = Vec::new();

    // -------------------------------------------------------------------------
    // Workload 1: Container Probe & GOP Indexing (300-frame 4K MP4)
    // -------------------------------------------------------------------------
    let probe_mp4 = tmp_dir.join("probe_4k_300f.mp4");
    create_benchmark_mp4(&probe_mp4, 3840, 2160, 300, 32_768)?;

    let probe_iters = 50usize;
    let (_, probe_perf_raw) = with_power_mode(PowerMode::Performance, || {
        measure_hardware_metrics(|| {
            for _ in 0..probe_iters {
                let rep = black_box(probe_file_detailed(&probe_mp4).unwrap());
                assert_eq!(rep.tracks[0].sample_count, 300);
            }
        })
    });
    let probe_perf_hw = scale_hardware_metrics(probe_perf_raw, probe_iters);
    let ragdoll_probe_rate = 1000.0 / probe_perf_hw.wall_ms;

    let (_, probe_eff_raw) = with_power_mode(PowerMode::Efficiency, || {
        measure_hardware_metrics(|| {
            for _ in 0..probe_iters {
                let rep = black_box(probe_file_detailed(&probe_mp4).unwrap());
                assert_eq!(rep.tracks[0].sample_count, 300);
            }
        })
    });
    let probe_eff_hw = scale_hardware_metrics(probe_eff_raw, probe_iters);
    let ragdoll_probe_eff_rate = 1000.0 / probe_eff_hw.wall_ms;

    let (ff_probe_hw, ff_probe_rate, probe_speedup_perf, probe_speedup_eff) = if compare_ffmpeg {
        let hw = run_external_batch_with_telemetry(
            &[(
                "ffprobe",
                vec![
                    "-v".into(),
                    "error".into(),
                    "-show_streams".into(),
                    "-show_packets".into(),
                    "-of".into(),
                    "json".into(),
                    probe_mp4.to_string_lossy().into_owned(),
                ],
            )],
            5,
        )?;
        let rate = 1000.0 / hw.wall_ms;
        (
            Some(hw),
            Some(rate),
            Some(hw.wall_ms / probe_perf_hw.wall_ms),
            Some(hw.wall_ms / probe_eff_hw.wall_ms),
        )
    } else {
        (None, None, None, None)
    };
    rows.push(BenchRow {
        workload: "1. Container Probe & GOP Index (300f 4K MP4)",
        unit: "files/s",
        ragdoll_perf_hw: probe_perf_hw,
        ragdoll_perf_rate: ragdoll_probe_rate,
        ragdoll_eff_hw: probe_eff_hw,
        ragdoll_eff_rate: ragdoll_probe_eff_rate,
        ffmpeg_hw: ff_probe_hw,
        ffmpeg_rate: ff_probe_rate,
        speedup_perf: probe_speedup_perf,
        speedup_eff: probe_speedup_eff,
    });
    let _ = std::fs::remove_file(&probe_mp4);

    // -------------------------------------------------------------------------
    // Workload 2: Zero-Copy Remuxing (MP4 -> MKV, ~16 MB 4K stream)
    // -------------------------------------------------------------------------
    let remux_in = tmp_dir.join("remux_in.mp4");
    create_benchmark_mp4(&remux_in, 3840, 2160, 240, 65_536)?;
    let remux_bytes = std::fs::metadata(&remux_in)?.len() as f64;

    let remux_iters = 5usize;
    let (_, remux_perf_raw) = with_power_mode(PowerMode::Performance, || {
        measure_hardware_metrics(|| {
            for it in 0..remux_iters {
                let out_p = tmp_dir.join(format!("remux_rag_{it}.mkv"));
                let _ = black_box(remux_file(&remux_in, &out_p).unwrap());
                let _ = std::fs::remove_file(&out_p);
            }
        })
    });
    let remux_perf_hw = scale_hardware_metrics(remux_perf_raw, remux_iters);
    let rag_remux_gbps = (remux_bytes / 1e9) / (remux_perf_hw.wall_ms / 1000.0);

    let (_, remux_eff_raw) = with_power_mode(PowerMode::Efficiency, || {
        measure_hardware_metrics(|| {
            for it in 0..remux_iters {
                let out_p = tmp_dir.join(format!("remux_rag_eff_{it}.mkv"));
                let _ = black_box(remux_file(&remux_in, &out_p).unwrap());
                let _ = std::fs::remove_file(&out_p);
            }
        })
    });
    let remux_eff_hw = scale_hardware_metrics(remux_eff_raw, remux_iters);
    let rag_remux_eff_gbps = (remux_bytes / 1e9) / (remux_eff_hw.wall_ms / 1000.0);

    let (ff_remux_hw, ff_remux_gbps, remux_speedup_perf, remux_speedup_eff) = if compare_ffmpeg {
        let out_p = tmp_dir.join("remux_ff.mkv");
        let hw = run_external_batch_with_telemetry(
            &[(
                "ffmpeg",
                vec![
                    "-y".into(),
                    "-v".into(),
                    "error".into(),
                    "-i".into(),
                    remux_in.to_string_lossy().into_owned(),
                    "-c".into(),
                    "copy".into(),
                    out_p.to_string_lossy().into_owned(),
                ],
            )],
            5,
        )?;
        let _ = std::fs::remove_file(&out_p);
        let gbps = (remux_bytes / 1e9) / (hw.wall_ms / 1000.0);
        (
            Some(hw),
            Some(gbps),
            Some(hw.wall_ms / remux_perf_hw.wall_ms),
            Some(hw.wall_ms / remux_eff_hw.wall_ms),
        )
    } else {
        (None, None, None, None)
    };
    let _ = std::fs::remove_file(&remux_in);
    let row2 = BenchRow {
        workload: "2. Zero-Copy Remux (MP4 -> MKV 16 MB 4K)",
        unit: "GB/s",
        ragdoll_perf_hw: remux_perf_hw,
        ragdoll_perf_rate: rag_remux_gbps,
        ragdoll_eff_hw: remux_eff_hw,
        ragdoll_eff_rate: rag_remux_eff_gbps,
        ffmpeg_hw: ff_remux_hw,
        ffmpeg_rate: ff_remux_gbps,
        speedup_perf: remux_speedup_perf,
        speedup_eff: remux_speedup_eff,
    };

    // -------------------------------------------------------------------------
    // Workload 4: Lossless Audio Coding (WAV -> FLAC + FLAC -> WAV, 30s 48kHz Stereo)
    // -------------------------------------------------------------------------
    let row4 = {
        let wav_in = tmp_dir.join("audio_30s.wav");
        let flac_rag = tmp_dir.join("audio_rag.flac");
        let flac_ff = tmp_dir.join("audio_ff.flac");
        synth_media(&wav_in, 0, 0, 30, 30.0)?;
        let audio_samples = 48_000 * 30;
        let mmap_wav = ragdoll_format::MmapSource::open(&wav_in)?;
        let wav_demux = ragdoll_format::WavDemuxer::open(mmap_wav.as_slice())?;
        let mut pcm_frame = AudioFrame::new(48_000, 2, SampleFormat::S16le, audio_samples);
        pcm_frame
            .data
            .as_mut_slice()
            .copy_from_slice(wav_demux.pcm_data());
        drop(mmap_wav);

        let (flac_bytes, audio_perf_hw) = with_power_mode(PowerMode::Performance, || {
            measure_hardware_metrics(|| {
                let flac_enc = FlacEncoder::new(48_000, 2, 16);
                let encoded = flac_enc.encode_stream(&pcm_frame).unwrap();
                let (_info, decoded_pcm) = FlacDecoder::decode_stream(&encoded).unwrap();
                assert_eq!(decoded_pcm.num_samples, audio_samples);
                encoded
            })
        });
        std::fs::write(&flac_rag, &flac_bytes)?;
        drop(flac_bytes);
        let rag_audio_xrt = 30.0 / (audio_perf_hw.wall_ms / 1000.0);

        let (_, audio_eff_hw) = with_power_mode(PowerMode::Efficiency, || {
            measure_hardware_metrics(|| {
                let flac_enc = FlacEncoder::new(48_000, 2, 16);
                let encoded = flac_enc.encode_stream(&pcm_frame).unwrap();
                let (_info, decoded_pcm) = FlacDecoder::decode_stream(&encoded).unwrap();
                assert_eq!(decoded_pcm.num_samples, audio_samples);
            })
        });
        drop(pcm_frame);
        let rag_audio_eff_xrt = 30.0 / (audio_eff_hw.wall_ms / 1000.0);

        let (ff_audio_hw, ff_audio_xrt, audio_speedup_perf, audio_speedup_eff) = if compare_ffmpeg {
            let wav_ff_dec = tmp_dir.join("audio_ff_dec.wav");
            let hw = run_external_batch_with_telemetry(
                &[
                    (
                        "ffmpeg",
                        vec![
                            "-y".into(),
                            "-v".into(),
                            "error".into(),
                            "-i".into(),
                            wav_in.to_string_lossy().into_owned(),
                            "-c:a".into(),
                            "flac".into(),
                            flac_ff.to_string_lossy().into_owned(),
                        ],
                    ),
                    (
                        "ffmpeg",
                        vec![
                            "-y".into(),
                            "-v".into(),
                            "error".into(),
                            "-i".into(),
                            flac_ff.to_string_lossy().into_owned(),
                            wav_ff_dec.to_string_lossy().into_owned(),
                        ],
                    ),
                ],
                1,
            )?;
            let xrt = 30.0 / (hw.wall_ms / 1000.0);
            (
                Some(hw),
                Some(xrt),
                Some(hw.wall_ms / audio_perf_hw.wall_ms),
                Some(hw.wall_ms / audio_eff_hw.wall_ms),
            )
        } else {
            (None, None, None, None)
        };
        BenchRow {
            workload: "4. Lossless FLAC Encode+Decode (30s 48kHz 2ch)",
            unit: "x RT",
            ragdoll_perf_hw: audio_perf_hw,
            ragdoll_perf_rate: rag_audio_xrt,
            ragdoll_eff_hw: audio_eff_hw,
            ragdoll_eff_rate: rag_audio_eff_xrt,
            ffmpeg_hw: ff_audio_hw,
            ffmpeg_rate: ff_audio_xrt,
            speedup_perf: audio_speedup_perf,
            speedup_eff: audio_speedup_eff,
        }
    };

    // -------------------------------------------------------------------------
    // Workload 5: Fast Frame Seek & 4x4 Contact Sheet Extraction
    // -------------------------------------------------------------------------
    let row5 = {
        let sheet_mp4 = tmp_dir.join("sheet_src.mp4");
        let sheet_rag_ppm = tmp_dir.join("sheet_rag.ppm");
        let sheet_ff_ppm = tmp_dir.join("sheet_ff.ppm");
        synth_media(&sheet_mp4, 640, 360, 30, 1.0)?;

        let (_, sheet_perf_hw) = with_power_mode(PowerMode::Performance, || {
            measure_hardware_metrics(|| {
                let _ = extract_media(&sheet_mp4, &sheet_rag_ppm, None, Some((16, 4, 4))).unwrap();
            })
        });
        let rag_sheet_rate = 16.0 / (sheet_perf_hw.wall_ms / 1000.0);

        let (_, sheet_eff_hw) = with_power_mode(PowerMode::Efficiency, || {
            measure_hardware_metrics(|| {
                let _ = extract_media(&sheet_mp4, &sheet_rag_ppm, None, Some((16, 4, 4))).unwrap();
            })
        });
        let rag_sheet_eff_rate = 16.0 / (sheet_eff_hw.wall_ms / 1000.0);

        let (ff_sheet_hw, ff_sheet_rate, sheet_speedup_perf, sheet_speedup_eff) = if compare_ffmpeg
        {
            let hw = run_external_batch_with_telemetry(
                &[(
                    "ffmpeg",
                    vec![
                        "-y".into(),
                        "-v".into(),
                        "error".into(),
                        "-i".into(),
                        sheet_mp4.to_string_lossy().into_owned(),
                        "-vf".into(),
                        "scale=160:90:flags=lanczos,tile=4x4".into(),
                        "-frames:v".into(),
                        "1".into(),
                        sheet_ff_ppm.to_string_lossy().into_owned(),
                    ],
                )],
                1,
            )?;
            let rate = 16.0 / (hw.wall_ms / 1000.0);
            (
                Some(hw),
                Some(rate),
                Some(hw.wall_ms / sheet_perf_hw.wall_ms),
                Some(hw.wall_ms / sheet_eff_hw.wall_ms),
            )
        } else {
            (None, None, None, None)
        };
        BenchRow {
            workload: "5. 4x4 Contact Sheet Extraction (16 Thumbs)",
            unit: "thumbs/s",
            ragdoll_perf_hw: sheet_perf_hw,
            ragdoll_perf_rate: rag_sheet_rate,
            ragdoll_eff_hw: sheet_eff_hw,
            ragdoll_eff_rate: rag_sheet_eff_rate,
            ffmpeg_hw: ff_sheet_hw,
            ffmpeg_rate: ff_sheet_rate,
            speedup_perf: sheet_speedup_perf,
            speedup_eff: sheet_speedup_eff,
        }
    };

    // -------------------------------------------------------------------------
    // Workload 7: 1080p H.264 Inter-Frame Video Transcode (60f Y4M -> MP4)
    // -------------------------------------------------------------------------
    let row7 = {
        let h264_frames_count = 60usize;
        let h264_in_y4m = tmp_dir.join("h264_1080p_60f.y4m");
        let h264_rag_mp4 = tmp_dir.join("h264_rag.mp4");
        let h264_ff_mp4 = tmp_dir.join("h264_ff.mp4");
        {
            let seed_indices: Vec<usize> = (0..15).collect();
            let seed_1080p: Vec<VideoFrame> =
                parallel_gop_map(&seed_indices, 0, |_, &i| synth_video_frame(1920, 1080, i));
            let f = std::fs::File::create(&h264_in_y4m)?;
            let mut mux = Y4mMuxer::new(
                BufWriter::with_capacity(2 * 1024 * 1024, f),
                1920,
                1080,
                PixelFormat::Yuv420p,
                Rational64::FPS_30,
                Colorimetry::bt709_limited(),
            )?;
            for i in 0..h264_frames_count {
                mux.write_frame(&seed_1080p[i % seed_1080p.len()])?;
            }
            mux.finish()?;
        }

        let (_, h264_perf_hw) = with_power_mode(PowerMode::Performance, || {
            measure_hardware_metrics(|| {
                let _ = transcode_file(
                    &h264_in_y4m,
                    &h264_rag_mp4,
                    None,
                    ScaleFilter::Lanczos3,
                    "h264",
                    24,
                    0,
                )
                .unwrap();
            })
        });
        let rag_h264_fps = (h264_frames_count as f64) / (h264_perf_hw.wall_ms / 1000.0);

        let (_, h264_eff_hw) = with_power_mode(PowerMode::Efficiency, || {
            measure_hardware_metrics(|| {
                let _ = transcode_file(
                    &h264_in_y4m,
                    &h264_rag_mp4,
                    None,
                    ScaleFilter::Lanczos3,
                    "h264",
                    24,
                    0,
                )
                .unwrap();
            })
        });
        let rag_h264_eff_fps = (h264_frames_count as f64) / (h264_eff_hw.wall_ms / 1000.0);

        let (ff_h264_hw, ff_h264_fps, h264_speedup_perf, h264_speedup_eff) = if compare_ffmpeg {
            let hw = run_external_batch_with_telemetry(
                &[(
                    "ffmpeg",
                    vec![
                        "-y".into(),
                        "-v".into(),
                        "error".into(),
                        "-i".into(),
                        h264_in_y4m.to_string_lossy().into_owned(),
                        "-c:v".into(),
                        "libx264".into(),
                        "-preset".into(),
                        "ultrafast".into(),
                        "-qp".into(),
                        "24".into(),
                        h264_ff_mp4.to_string_lossy().into_owned(),
                    ],
                )],
                1,
            )?;
            let fps = (h264_frames_count as f64) / (hw.wall_ms / 1000.0);
            (
                Some(hw),
                Some(fps),
                Some(hw.wall_ms / h264_perf_hw.wall_ms),
                Some(hw.wall_ms / h264_eff_hw.wall_ms),
            )
        } else {
            (None, None, None, None)
        };
        let _ = std::fs::remove_file(&h264_in_y4m);
        let _ = std::fs::remove_file(&h264_rag_mp4);
        let _ = std::fs::remove_file(&h264_ff_mp4);
        BenchRow {
            workload: "7. 1080p H.264 Inter Transcode (60f Y4M->MP4)",
            unit: "fps",
            ragdoll_perf_hw: h264_perf_hw,
            ragdoll_perf_rate: rag_h264_fps,
            ragdoll_eff_hw: h264_eff_hw,
            ragdoll_eff_rate: rag_h264_eff_fps,
            ffmpeg_hw: ff_h264_hw,
            ffmpeg_rate: ff_h264_fps,
            speedup_perf: h264_speedup_perf,
            speedup_eff: h264_speedup_eff,
        }
    };

    // -------------------------------------------------------------------------
    // Workload 3: 4K -> 1080p Fused Lanczos-3 Polyphase Scaling (30 frames)
    // -------------------------------------------------------------------------
    let scale_frames_count = 30usize;
    let indices: Vec<usize> = (0..scale_frames_count).collect();
    let frames_4k: Vec<VideoFrame> =
        parallel_gop_map(&indices, 0, |_, &i| synth_video_frame(3840, 2160, i));

    let (scaled_1080p, scale_perf_hw) = with_power_mode(PowerMode::Performance, || {
        measure_hardware_metrics(|| {
            parallel_scale_frames(&frames_4k, 1920, 1080, ScaleFilter::Lanczos3, 0).unwrap()
        })
    });
    assert_eq!(scaled_1080p.len(), scale_frames_count);
    drop(scaled_1080p);
    let rag_scale_fps = (scale_frames_count as f64) / (scale_perf_hw.wall_ms / 1000.0);

    let (scaled_1080p_eff, scale_eff_hw) = with_power_mode(PowerMode::Efficiency, || {
        measure_hardware_metrics(|| {
            parallel_scale_frames(&frames_4k, 1920, 1080, ScaleFilter::Lanczos3, 0).unwrap()
        })
    });
    assert_eq!(scaled_1080p_eff.len(), scale_frames_count);
    drop(scaled_1080p_eff);
    let rag_scale_eff_fps = (scale_frames_count as f64) / (scale_eff_hw.wall_ms / 1000.0);

    let (ff_scale_hw, ff_scale_fps, scale_speedup_perf, scale_speedup_eff) = if compare_ffmpeg {
        let y4m_4k = tmp_dir.join("scale_4k.y4m");
        {
            let f = std::fs::File::create(&y4m_4k)?;
            let mut mux = Y4mMuxer::new(
                BufWriter::with_capacity(1024 * 1024, f),
                3840,
                2160,
                PixelFormat::Yuv420p,
                Rational64::FPS_30,
                Colorimetry::bt709_limited(),
            )?;
            for fr in &frames_4k {
                mux.write_frame(fr)?;
            }
            mux.finish()?;
        }
        let hw = run_external_batch_with_telemetry(
            &[(
                "ffmpeg",
                vec![
                    "-y".into(),
                    "-v".into(),
                    "error".into(),
                    "-i".into(),
                    y4m_4k.to_string_lossy().into_owned(),
                    "-vf".into(),
                    "scale=1920:1080:flags=lanczos".into(),
                    "-f".into(),
                    "null".into(),
                    "-".into(),
                ],
            )],
            1,
        )?;
        let fps = (scale_frames_count as f64) / (hw.wall_ms / 1000.0);
        let _ = std::fs::remove_file(&y4m_4k);
        (
            Some(hw),
            Some(fps),
            Some(hw.wall_ms / scale_perf_hw.wall_ms),
            Some(hw.wall_ms / scale_eff_hw.wall_ms),
        )
    } else {
        (None, None, None, None)
    };
    let row3 = BenchRow {
        workload: "3. 4K -> 1080p Lanczos-3 Scaling (30 frames)",
        unit: "fps",
        ragdoll_perf_hw: scale_perf_hw,
        ragdoll_perf_rate: rag_scale_fps,
        ragdoll_eff_hw: scale_eff_hw,
        ragdoll_eff_rate: rag_scale_eff_fps,
        ffmpeg_hw: ff_scale_hw,
        ffmpeg_rate: ff_scale_fps,
        speedup_perf: scale_speedup_perf,
        speedup_eff: scale_speedup_eff,
    };

    // -------------------------------------------------------------------------
    // Workload 6: 4K Perceptual Quality Metrics (PSNR + 8x8 Windowed SSIM, 20 frames)
    // -------------------------------------------------------------------------
    let metric_frames = 20usize;
    let ref_4k = &frames_4k[..metric_frames];
    let dist_4k = &frames_4k[1..=metric_frames];

    let metric_indices: Vec<usize> = (0..metric_frames).collect();
    let (_, metrics_perf_hw) = with_power_mode(PowerMode::Performance, || {
        measure_hardware_metrics(|| {
            let _metric_results: Vec<_> = parallel_gop_map(&metric_indices, 0, |_, &i| {
                let r = &ref_4k[i];
                let d = &dist_4k[i];
                let psnr = ragdoll_pix::psnr_plane_u8(
                    r.plane(0).data(),
                    r.plane(0).stride(),
                    d.plane(0).data(),
                    d.plane(0).stride(),
                    r.width,
                    r.height,
                );
                let ssim = ssim_plane_8bit(
                    r.plane(0).data(),
                    r.plane(0).stride(),
                    d.plane(0).data(),
                    d.plane(0).stride(),
                    r.width,
                    r.height,
                );
                (psnr, ssim)
            });
        })
    });
    let rag_metrics_fps = (metric_frames as f64) / (metrics_perf_hw.wall_ms / 1000.0);

    let (_, metrics_eff_hw) = with_power_mode(PowerMode::Efficiency, || {
        measure_hardware_metrics(|| {
            let _metric_results: Vec<_> = parallel_gop_map(&metric_indices, 0, |_, &i| {
                let r = &ref_4k[i];
                let d = &dist_4k[i];
                let psnr = ragdoll_pix::psnr_plane_u8(
                    r.plane(0).data(),
                    r.plane(0).stride(),
                    d.plane(0).data(),
                    d.plane(0).stride(),
                    r.width,
                    r.height,
                );
                let ssim = ssim_plane_8bit(
                    r.plane(0).data(),
                    r.plane(0).stride(),
                    d.plane(0).data(),
                    d.plane(0).stride(),
                    r.width,
                    r.height,
                );
                (psnr, ssim)
            });
        })
    });
    let rag_metrics_eff_fps = (metric_frames as f64) / (metrics_eff_hw.wall_ms / 1000.0);

    let (ff_metrics_hw, ff_metrics_fps, metrics_speedup_perf, metrics_speedup_eff) =
        if compare_ffmpeg {
            let ref_y4m = tmp_dir.join("ref_4k.y4m");
            let dist_y4m = tmp_dir.join("dist_4k.y4m");
            for (p, slice) in [(&ref_y4m, ref_4k), (&dist_y4m, dist_4k)] {
                let f = std::fs::File::create(p)?;
                let mut mux = Y4mMuxer::new(
                    BufWriter::with_capacity(1024 * 1024, f),
                    3840,
                    2160,
                    PixelFormat::Yuv420p,
                    Rational64::FPS_30,
                    Colorimetry::bt709_limited(),
                )?;
                for fr in slice {
                    mux.write_frame(fr)?;
                }
                mux.finish()?;
            }
            let hw = run_external_batch_with_telemetry(
                &[(
                    "ffmpeg",
                    vec![
                        "-v".into(),
                        "error".into(),
                        "-i".into(),
                        ref_y4m.to_string_lossy().into_owned(),
                        "-i".into(),
                        dist_y4m.to_string_lossy().into_owned(),
                        "-lavfi".into(),
                        "[0:v][1:v]psnr;[0:v][1:v]ssim".into(),
                        "-f".into(),
                        "null".into(),
                        "-".into(),
                    ],
                )],
                1,
            )?;
            let fps = (metric_frames as f64) / (hw.wall_ms / 1000.0);
            let _ = std::fs::remove_file(&ref_y4m);
            let _ = std::fs::remove_file(&dist_y4m);
            (
                Some(hw),
                Some(fps),
                Some(hw.wall_ms / metrics_perf_hw.wall_ms),
                Some(hw.wall_ms / metrics_eff_hw.wall_ms),
            )
        } else {
            (None, None, None, None)
        };
    drop(frames_4k);
    let row6 = BenchRow {
        workload: "6. 4K Perceptual Metrics (PSNR + 8x8 SSIM 20f)",
        unit: "fps",
        ragdoll_perf_hw: metrics_perf_hw,
        ragdoll_perf_rate: rag_metrics_fps,
        ragdoll_eff_hw: metrics_eff_hw,
        ragdoll_eff_rate: rag_metrics_eff_fps,
        ffmpeg_hw: ff_metrics_hw,
        ffmpeg_rate: ff_metrics_fps,
        speedup_perf: metrics_speedup_perf,
        speedup_eff: metrics_speedup_eff,
    };

    rows.extend([row2, row3, row4, row5, row6, row7]);

    let _ = std::fs::remove_dir_all(&tmp_dir);

    let topo = CpuTopology::detect();
    if json {
        println!("{{");
        println!("  \"simd\": \"{}\",", SimdLevel::detect().name());
        println!(
            "  \"cpu_topology\": {{\"logical_cores\": {}, \"p_cores\": {}, \"e_cores\": {}}},",
            topo.total_cores, topo.p_cores, topo.e_cores
        );
        println!("  \"benchmarks\": [");
        for (i, r) in rows.iter().enumerate() {
            let comma = if i + 1 < rows.len() { "," } else { "" };
            println!(
                "    {{\"workload\": \"{}\", \"unit\": \"{}\", \"ragdoll_perf_ms\": {:.3}, \"ragdoll_perf_rate\": {:.2}, \"ragdoll_perf_instructions\": {}, \"ragdoll_perf_cycles\": {}, \"ragdoll_perf_energy_mj\": {:.3}, \"ragdoll_eff_ms\": {:.3}, \"ragdoll_eff_rate\": {:.2}, \"ragdoll_eff_instructions\": {}, \"ragdoll_eff_cycles\": {}, \"ragdoll_eff_ecore_pct\": {:.1}, \"ragdoll_eff_energy_mj\": {:.3}, \"ffmpeg_ms\": {}, \"ffmpeg_rate\": {}, \"ffmpeg_instructions\": {}, \"ffmpeg_cycles\": {}, \"ffmpeg_energy_mj\": {}, \"speedup_perf\": {}, \"speedup_eff\": {}}}{comma}",
                r.workload,
                r.unit,
                r.ragdoll_perf_hw.wall_ms,
                r.ragdoll_perf_rate,
                r.ragdoll_perf_hw.instructions,
                r.ragdoll_perf_hw.cycles,
                r.ragdoll_perf_hw.energy_mj,
                r.ragdoll_eff_hw.wall_ms,
                r.ragdoll_eff_rate,
                r.ragdoll_eff_hw.instructions,
                r.ragdoll_eff_hw.cycles,
                r.ragdoll_eff_hw.ecore_residency_pct,
                r.ragdoll_eff_hw.energy_mj,
                r.ffmpeg_hw
                    .map_or("null".into(), |h| format!("{:.3}", h.wall_ms)),
                r.ffmpeg_rate.map_or("null".into(), |v| format!("{v:.2}")),
                r.ffmpeg_hw
                    .map_or("null".into(), |h| format!("{}", h.instructions)),
                r.ffmpeg_hw
                    .map_or("null".into(), |h| format!("{}", h.cycles)),
                r.ffmpeg_hw
                    .map_or("null".into(), |h| format!("{:.3}", h.energy_mj)),
                r.speedup_perf.map_or("null".into(), |v| format!("{v:.2}")),
                r.speedup_eff.map_or("null".into(), |v| format!("{v:.2}"))
            );
        }
        println!("  ]");
        println!("}}");
    } else {
        println!(
            "=================================================================================================================================="
        );
        println!(
            "  RAGDOLL v0.1.0 SPEED & BATTERY EFFICIENCY BENCHMARK SUITE — SIMD: {} | CPU: {} P-Cores + {} E-Cores",
            SimdLevel::detect().name(),
            topo.p_cores,
            topo.e_cores
        );
        println!(
            "=================================================================================================================================="
        );
        println!(
            "{:<47} {:>11} {:>14} {:>11} {:>11} {:>10} {:>10}",
            "Workload",
            "P-Core (ms)",
            "P-Core Rate",
            "E-Core (ms)",
            "FFmpeg (ms)",
            "Speedup(P)",
            "Speedup(E)"
        );
        println!(
            "----------------------------------------------------------------------------------------------------------------------------------"
        );
        for r in &rows {
            let ff_ms_str = r
                .ffmpeg_hw
                .map_or_else(|| "N/A".into(), |h| format!("{:.2} ms", h.wall_ms));
            let sp_p_str = r
                .speedup_perf
                .map_or_else(|| "N/A".into(), |v| format!("{v:.2}x"));
            let sp_e_str = r
                .speedup_eff
                .map_or_else(|| "N/A".into(), |v| format!("{v:.2}x"));
            println!(
                "{:<47} {:>8.2} ms {:>8.2} {:<5} {:>8.2} ms {:>11} {:>10} {:>10}",
                r.workload,
                r.ragdoll_perf_hw.wall_ms,
                r.ragdoll_perf_rate,
                r.unit,
                r.ragdoll_eff_hw.wall_ms,
                ff_ms_str,
                sp_p_str,
                sp_e_str
            );
        }
        println!(
            "=================================================================================================================================="
        );
        println!(
            "\nHARDWARE PMU TELEMETRY & BATTERY EFFICIENCY (proc_pid_rusage RUSAGE_INFO_V4 / /usr/bin/time -l):"
        );
        println!(
            "----------------------------------------------------------------------------------------------------------------------------------"
        );
        println!(
            "{:<35} {:<16} {:>10} {:>10} {:>6} {:>8} {:>8} {:>10} {:>8} {:>9} {:>10}",
            "Workload",
            "Engine / Mode",
            "Instr(M)",
            "Cycles(M)",
            "IPC",
            "GHz",
            "E-Core%",
            "Energy",
            "Power",
            "Mem(MiB)",
            "EnergySav"
        );
        println!(
            "----------------------------------------------------------------------------------------------------------------------------------"
        );
        for r in &rows {
            let short_wl = if r.workload.len() > 34 {
                &r.workload[..34]
            } else {
                r.workload
            };
            let ff_energy = r.ffmpeg_hw.map(|h| h.energy_mj).unwrap_or(0.0);
            let p_sav = if ff_energy > 0.0 && r.ragdoll_perf_hw.energy_mj > 0.0 {
                format!("{:.1}x", ff_energy / r.ragdoll_perf_hw.energy_mj)
            } else {
                "N/A".into()
            };
            let e_sav = if ff_energy > 0.0 && r.ragdoll_eff_hw.energy_mj > 0.0 {
                format!("{:.1}x", ff_energy / r.ragdoll_eff_hw.energy_mj)
            } else if r.ragdoll_perf_hw.energy_mj > 0.0 && r.ragdoll_eff_hw.energy_mj > 0.0 {
                format!(
                    "{:.1}x vs P",
                    r.ragdoll_perf_hw.energy_mj / r.ragdoll_eff_hw.energy_mj
                )
            } else {
                "N/A".into()
            };

            println!(
                "{:<35} {:<16} {:>10.2} {:>10.2} {:>6.2} {:>8.2} {:>7.1}% {:>7.2} mJ {:>6.2} W {:>9.1} {:>10}",
                short_wl,
                "Ragdoll P-Core",
                (r.ragdoll_perf_hw.instructions as f64) / 1e6,
                (r.ragdoll_perf_hw.cycles as f64) / 1e6,
                r.ragdoll_perf_hw.ipc,
                r.ragdoll_perf_hw.effective_ghz,
                r.ragdoll_perf_hw.ecore_residency_pct,
                r.ragdoll_perf_hw.energy_mj,
                r.ragdoll_perf_hw.avg_power_watts,
                r.ragdoll_perf_hw.peak_rss_mib,
                p_sav
            );
            println!(
                "{:<35} {:<16} {:>10.2} {:>10.2} {:>6.2} {:>8.2} {:>7.1}% {:>7.2} mJ {:>6.2} W {:>9.1} {:>10}",
                "",
                "Ragdoll E-Core",
                (r.ragdoll_eff_hw.instructions as f64) / 1e6,
                (r.ragdoll_eff_hw.cycles as f64) / 1e6,
                r.ragdoll_eff_hw.ipc,
                r.ragdoll_eff_hw.effective_ghz,
                r.ragdoll_eff_hw.ecore_residency_pct,
                r.ragdoll_eff_hw.energy_mj,
                r.ragdoll_eff_hw.avg_power_watts,
                r.ragdoll_eff_hw.peak_rss_mib,
                e_sav
            );
            if let Some(ff) = r.ffmpeg_hw {
                println!(
                    "{:<35} {:<16} {:>10.2} {:>10.2} {:>6.2} {:>8.2} {:>7.1}% {:>7.2} mJ {:>6.2} W {:>9.1} {:>10}",
                    "",
                    "FFmpeg 9.0.2",
                    (ff.instructions as f64) / 1e6,
                    (ff.cycles as f64) / 1e6,
                    ff.ipc,
                    ff.effective_ghz,
                    ff.ecore_residency_pct,
                    ff.energy_mj,
                    ff.avg_power_watts,
                    ff.peak_rss_mib,
                    "1.0x (ref)"
                );
            }
            println!(
                "----------------------------------------------------------------------------------------------------------------------------------"
            );
        }
    }
    Ok(())
}

fn create_benchmark_mp4(
    path: &Path,
    width: u32,
    height: u32,
    num_packets: usize,
    pkt_size: usize,
) -> Result<()> {
    let sps = ragdoll_codec::Sps::new_baseline(64, 64, Rational64::FPS_30).to_nal_unit();
    let pps = ragdoll_codec::Pps::new_baseline(22).to_nal_unit();
    let avcc = build_avcc_from_sps_pps(&sps, &pps);

    // Encode a small real H.264 IDR slice header + macroblock prefix so ffmpeg's h264_parse BSF
    // validates the slice header cleanly without warnings, followed by RBSP filler NAL unit (0x0C)
    let mut enc = ragdoll_codec::H264Encoder::new(64, 64, Rational64::FPS_30, 22);
    let tiny_frame = VideoFrame::new(64, 64, PixelFormat::Yuv420p);
    let annex_b = enc.encode_frame(&tiny_frame)?;
    let avcc_idr = ragdoll_format::h264_annex_b_to_avcc_au(&annex_b);

    let mut payload = Vec::with_capacity(pkt_size.max(avcc_idr.len() + 8));
    payload.extend_from_slice(&avcc_idr);
    if payload.len() + 8 < pkt_size {
        let filler_len = pkt_size - payload.len() - 4;
        payload.extend_from_slice(&(filler_len as u32).to_be_bytes());
        payload.push(0x0C); // NAL unit type 12: Filler data
        payload.resize(pkt_size - 1, 0xFF);
        payload.push(0x80); // RBSP stop bit
    }

    let tb = Rational64::TIMEBASE_MPEG_90K;
    let mut mux = Mp4Muxer::new();
    mux.add_track(TrackInfo {
        index: 0,
        track_type: TrackType::Video,
        codec_id: CodecId::H264,
        timebase: tb,
        duration_ticks: Some((num_packets as i64) * 3000),
        extra_data: avcc,
        video: Some(VideoTrackParams {
            width: 64,
            height: 64,
            pixel_format: PixelFormat::Yuv420p,
            colorimetry: Colorimetry::bt709_limited(),
            frame_rate: Rational64::FPS_30,
            bit_depth: 8,
            profile: None,
            level: None,
        }),
        audio: None,
        sample_count: num_packets,
        total_bytes: (num_packets * payload.len()) as u64,
    });

    for i in 0..num_packets {
        let ts = Timestamp::new((i as i64) * 3000, tb);
        let flags = if i.is_multiple_of(30) {
            PacketFlags::KEYFRAME
        } else {
            PacketFlags::NONE
        };
        mux.write_packet(PacketSlice::from_mmap(&payload, 0, ts, ts, 3000, flags, 0))?;
    }
    let f = std::fs::File::create(path)?;
    let _ = (width, height);
    mux.finish(BufWriter::with_capacity(4 * 1024 * 1024, f))?;
    Ok(())
}

fn print_telemetry(tel: &PipelineTelemetry, json: bool) {
    if json {
        println!("{{");
        println!("  \"operation\": \"{}\",", escape_json(&tel.operation));
        println!("  \"input\": \"{}\",", escape_json(&tel.input_path));
        println!("  \"outputs\": [");
        for (i, o) in tel.output_paths.iter().enumerate() {
            let comma = if i + 1 < tel.output_paths.len() {
                ","
            } else {
                ""
            };
            println!("    \"{}\"{comma}", escape_json(o));
        }
        println!("  ],");
        println!("  \"frames_processed\": {},", tel.frames_processed);
        println!(
            "  \"audio_samples_processed\": {},",
            tel.audio_samples_processed
        );
        println!("  \"gops_processed\": {},", tel.gops_processed);
        println!("  \"threads_used\": {},", tel.threads_used);
        println!("  \"bytes_in\": {},", tel.bytes_in);
        println!("  \"bytes_out\": {},", tel.bytes_out);
        println!("  \"elapsed_ms\": {:.3},", tel.elapsed_ms);
        println!("  \"fps\": {:.2},", tel.fps);
        println!("  \"bandwidth_gbps\": {:.4},", tel.bandwidth_gbps);
        println!("  \"compression_ratio\": {:.3}", tel.compression_ratio);
        println!("}}");
    } else {
        println!(
            "[ragdoll {}] {} -> {} | frames={} gops={} threads={} | in={} B out={} B ({:.2}:1) | {:.2} ms ({:.1} fps, {:.3} GB/s)",
            tel.operation,
            tel.input_path,
            tel.output_paths.join(", "),
            tel.frames_processed,
            tel.gops_processed,
            tel.threads_used,
            tel.bytes_in,
            tel.bytes_out,
            tel.compression_ratio,
            tel.elapsed_ms,
            tel.fps,
            tel.bandwidth_gbps
        );
    }
}

fn parse_dimensions(s: &str) -> Option<(usize, usize)> {
    let (w, h) = s.split_once('x').or_else(|| s.split_once(':'))?;
    Some((w.parse().ok()?, h.parse().ok()?))
}

fn parse_bitrate_kbps(s: &str) -> Option<u32> {
    let trimmed = s.trim().to_ascii_lowercase();
    if let Some(m) = trimmed
        .strip_suffix("mbps")
        .or_else(|| trimmed.strip_suffix('m'))
    {
        return Some((m.trim().parse::<f64>().ok()? * 1000.0).round() as u32);
    }
    if let Some(k) = trimmed
        .strip_suffix("kbps")
        .or_else(|| trimmed.strip_suffix('k'))
    {
        return Some(k.trim().parse::<f64>().ok()?.round() as u32);
    }
    let val = trimmed.parse::<f64>().ok()?;
    if val >= 100_000.0 {
        Some((val / 1000.0).round() as u32)
    } else {
        Some(val.round() as u32)
    }
}

fn parse_time_seconds(s: &str) -> Option<f64> {
    if let Ok(sec) = s.parse::<f64>() {
        return Some(sec);
    }
    let ts = Timestamp::parse_hms(s).ok()?;
    ts.to_seconds_f64()
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn format_finite(v: f64) -> f64 {
    if v.is_finite() { v } else { 99.99 }
}

pub fn run_self_test() {
    let simd = SimdLevel::detect();
    println!("================================================================================");
    println!("  RAGDOLL v0.1.0 — SIMD Differential Self-Test");
    println!("  Detected Hardware SIMD : {}", simd.name());
    println!("================================================================================\n");

    let pool = FramePool::with_memory_budget(8, 128 * 1024 * 1024);
    let mut frame_a = pool.acquire_scoped(3840, 2160, PixelFormat::Yuv420p);
    let mut frame_b = pool.acquire_scoped(3840, 2160, PixelFormat::Yuv420p);
    frame_a.extend_edges();
    frame_b.extend_edges();

    println!(
        "[Arena] 4K UHD (3840x2160) YUV420p Frame: Y stride = {}B (64B aligned = {}), 2048px de-aliased stride = {}B",
        frame_a.plane(0).stride(),
        (frame_a.plane(0).as_ptr() as usize).is_multiple_of(64),
        optimal_plane_stride(2048),
    );

    let width = 3840usize;
    let height = 2160usize;
    let stride = frame_a.plane(0).stride();
    {
        let ya = frame_a.plane_mut(0);
        for y in 0..height {
            let row = ya.row_mut(y);
            for (x, p) in row.iter_mut().enumerate() {
                *p = ((x * 19 + y * 31 + 17) & 0xFF) as u8;
            }
        }
        let yb = frame_b.plane_mut(0);
        for y in 0..height {
            let row = yb.row_mut(y);
            for (x, p) in row.iter_mut().enumerate() {
                *p = ((x * 19 + y * 31 + 20) & 0xFF) as u8;
            }
        }
    }

    let ya_slice = frame_a.plane(0).data();
    let yb_slice = frame_b.plane(0).data();
    let pixels_per_frame = (width * height) as f64;

    println!(
        "\nKernel (4K 3840x2160 Luma / 16 MB Stream)      Scalar         NEON SIMD      Speedup   Check"
    );
    println!(
        "----------------------------------------------------------------------------------------------"
    );

    let iters = 20;
    let t0 = Instant::now();
    let mut sad_ref = 0u64;
    for _ in 0..iters {
        sad_ref = black_box(sad_plane_u8_scalar(
            black_box(ya_slice),
            stride,
            black_box(yb_slice),
            stride,
            width,
            height,
        ));
    }
    let dt_sad_scalar = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let mut sad_simd = 0u64;
    for _ in 0..iters {
        sad_simd = black_box(sad_plane_u8(
            black_box(ya_slice),
            stride,
            black_box(yb_slice),
            stride,
            width,
            height,
        ));
    }
    let dt_sad_simd = t1.elapsed().as_secs_f64();
    assert_eq!(sad_ref, sad_simd);
    println!(
        "4K Plane SAD (sum |a - b|)                  {:>8.2} Gpx/s   {:>8.2} Gpx/s   {:>5.2}x    EXACT ({sad_simd})",
        (pixels_per_frame * iters as f64) / dt_sad_scalar / 1e9,
        (pixels_per_frame * iters as f64) / dt_sad_simd / 1e9,
        dt_sad_scalar / dt_sad_simd
    );

    let t0 = Instant::now();
    let mut sse_ref = 0u64;
    for _ in 0..iters {
        sse_ref = black_box(sse_plane_u8_scalar(
            black_box(ya_slice),
            stride,
            black_box(yb_slice),
            stride,
            width,
            height,
        ));
    }
    let dt_sse_scalar = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let mut sse_simd = 0u64;
    for _ in 0..iters {
        sse_simd = black_box(sse_plane_u8(
            black_box(ya_slice),
            stride,
            black_box(yb_slice),
            stride,
            width,
            height,
        ));
    }
    let dt_sse_simd = t1.elapsed().as_secs_f64();
    assert_eq!(sse_ref, sse_simd);
    println!(
        "4K Plane SSE (sum (a - b)^2)                {:>8.2} Gpx/s   {:>8.2} Gpx/s   {:>5.2}x    EXACT ({sse_simd})",
        (pixels_per_frame * iters as f64) / dt_sse_scalar / 1e9,
        (pixels_per_frame * iters as f64) / dt_sse_simd / 1e9,
        dt_sse_scalar / dt_sse_simd
    );

    let mut sums_ref = vec![[0i32; 4]; width / 4];
    let mut sums_simd = vec![[0i32; 4]; width / 4];
    let t0 = Instant::now();
    for _ in 0..iters {
        for by in 0..(height / 4) {
            let off = by * 4 * stride;
            ssim_4x4xn_u8_scalar(
                black_box(&ya_slice[off..]),
                stride,
                black_box(&yb_slice[off..]),
                stride,
                &mut sums_ref,
            );
        }
    }
    let dt_ssim_scalar = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    for _ in 0..iters {
        for by in 0..(height / 4) {
            let off = by * 4 * stride;
            ssim_4x4xn_u8(
                black_box(&ya_slice[off..]),
                stride,
                black_box(&yb_slice[off..]),
                stride,
                &mut sums_simd,
            );
        }
    }
    let dt_ssim_simd = t1.elapsed().as_secs_f64();
    assert_eq!(sums_ref, sums_simd);
    let ssim_val = ssim_plane_8bit(ya_slice, stride, yb_slice, stride, width, height);
    println!(
        "4K Plane SSIM 4x4-Reuse (vf_ssim)           {:>8.2} Gpx/s   {:>8.2} Gpx/s   {:>5.2}x    EXACT ({:.5} / {:.2} dB)",
        (pixels_per_frame * iters as f64) / dt_ssim_scalar / 1e9,
        (pixels_per_frame * iters as f64) / dt_ssim_simd / 1e9,
        dt_ssim_scalar / dt_ssim_simd,
        ssim_val,
        ssim_to_db(ssim_val)
    );

    let block = [
        14i16, -22, 7, 3, -9, 31, -15, 4, 18, -6, -11, 2, -3, 8, -5, 1,
    ];
    let dct_iters = 500_000usize;
    let t0 = Instant::now();
    let mut out_scalar = [0i16; 16];
    for _ in 0..dct_iters {
        let c = forward_dct4x4_scalar(black_box(&block));
        out_scalar = black_box(inverse_idct4x4_scalar(black_box(&c)));
    }
    let dt_dct_scalar = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let mut out_simd = [0i16; 16];
    for _ in 0..dct_iters {
        let c = forward_dct4x4(black_box(&block));
        out_simd = black_box(inverse_idct4x4(black_box(&c)));
    }
    let dt_dct_simd = t1.elapsed().as_secs_f64();
    assert_eq!(out_scalar, out_simd);
    println!(
        "H.264 4x4 DCT+IDCT (vtrn1/vtrn2 + vrshr)    {:>8.2} Mblk/s  {:>8.2} Mblk/s  {:>5.2}x    EXACT",
        (dct_iters as f64) / dt_dct_scalar / 1e6,
        (dct_iters as f64) / dt_dct_simd / 1e6,
        dt_dct_scalar / dt_dct_simd
    );

    let stream_len = 8 * 1024 * 1024;
    let mut annex_b = vec![0x5Au8; stream_len];
    for i in (19..stream_len - 16).step_by(23) {
        annex_b[i] = 0x00;
    }
    annex_b[stream_len - 8] = 0x00;
    annex_b[stream_len - 7] = 0x00;
    annex_b[stream_len - 6] = 0x00;
    annex_b[stream_len - 5] = 0x01;

    let scan_iters = 20usize;
    let t0 = Instant::now();
    let mut pos_ref = None;
    for _ in 0..scan_iters {
        pos_ref = black_box(find_annex_b_start_code_scalar(black_box(&annex_b)));
    }
    let dt_scan_scalar = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let mut pos_simd = None;
    for _ in 0..scan_iters {
        pos_simd = black_box(find_annex_b_start_code(black_box(&annex_b)));
    }
    let dt_scan_simd = t1.elapsed().as_secs_f64();
    assert_eq!(pos_ref, pos_simd);
    println!(
        "Annex-B Start-Code Scan (isolated 0x00s)    {:>8.2} GB/s    {:>8.2} GB/s    {:>5.2}x    EXACT",
        (stream_len * scan_iters) as f64 / dt_scan_scalar / 1e9,
        (stream_len * scan_iters) as f64 / dt_scan_simd / 1e9,
        dt_scan_scalar / dt_scan_simd
    );

    let mut rbsp_raw = vec![0x3Cu8; 2 * 1024 * 1024];
    for i in (0..rbsp_raw.len() - 4).step_by(512) {
        rbsp_raw[i] = 0x00;
        rbsp_raw[i + 1] = 0x00;
        rbsp_raw[i + 2] = 0x01;
    }
    let escaped = escape_rbsp(&rbsp_raw);
    let rbsp_iters = 20usize;
    let t0 = Instant::now();
    let mut u_ref = Vec::new();
    for _ in 0..rbsp_iters {
        u_ref = black_box(unescape_rbsp_scalar(black_box(&escaped)));
    }
    let dt_rbsp_scalar = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let mut u_simd = Vec::new();
    for _ in 0..rbsp_iters {
        u_simd = black_box(unescape_rbsp(black_box(&escaped)));
    }
    let dt_rbsp_simd = t1.elapsed().as_secs_f64();
    assert_eq!(u_ref, u_simd);
    assert_eq!(u_simd, rbsp_raw);
    println!(
        "H.264/HEVC RBSP Unescape (vextq_u8 pair)    {:>8.2} GB/s    {:>8.2} GB/s    {:>5.2}x    EXACT",
        (escaped.len() * rbsp_iters) as f64 / dt_rbsp_scalar / 1e9,
        (escaped.len() * rbsp_iters) as f64 / dt_rbsp_simd / 1e9,
        dt_rbsp_scalar / dt_rbsp_simd
    );

    let w1080 = 1920usize;
    let h1080 = 1080usize;
    let px1080 = (w1080 * h1080) as f64;
    let mat709 = YuvRgbMatrix::from_colorimetry(MatrixCoefficients::BT709, ColorRange::Limited);
    let mut rgb_ref = vec![0u8; w1080 * 3 * h1080];
    let mut rgb_simd = vec![0u8; w1080 * 3 * h1080];
    let u_slice = frame_a.plane(1).data();
    let v_slice = frame_a.plane(2).data();
    let c_stride = frame_a.plane(1).stride();

    let pix_iters = 15usize;
    let t0 = Instant::now();
    for _ in 0..pix_iters {
        yuv420p_to_rgb24_scalar(
            black_box(ya_slice),
            stride,
            black_box(u_slice),
            c_stride,
            black_box(v_slice),
            c_stride,
            black_box(&mut rgb_ref),
            w1080 * 3,
            w1080,
            h1080,
            &mat709,
        );
    }
    let dt_y2r_scalar = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    for _ in 0..pix_iters {
        yuv420p_to_rgb24(
            black_box(ya_slice),
            stride,
            black_box(u_slice),
            c_stride,
            black_box(v_slice),
            c_stride,
            black_box(&mut rgb_simd),
            w1080 * 3,
            w1080,
            h1080,
            &mat709,
        );
    }
    let dt_y2r_simd = t1.elapsed().as_secs_f64();
    assert_eq!(rgb_ref, rgb_simd);
    println!(
        "1080p YUV420p -> RGB24 (BT.709 vst3q_u8)    {:>8.2} Gpx/s   {:>8.2} Gpx/s   {:>5.2}x    EXACT",
        (px1080 * pix_iters as f64) / dt_y2r_scalar / 1e9,
        (px1080 * pix_iters as f64) / dt_y2r_simd / 1e9,
        dt_y2r_scalar / dt_y2r_simd
    );

    let mut rgba_ref = vec![0u8; w1080 * 4 * h1080];
    let mut rgba_simd = vec![0u8; w1080 * 4 * h1080];
    let t0 = Instant::now();
    for _ in 0..pix_iters {
        yuv420p_to_rgba8_scalar(
            black_box(ya_slice),
            stride,
            black_box(u_slice),
            c_stride,
            black_box(v_slice),
            c_stride,
            black_box(&mut rgba_ref),
            w1080 * 4,
            w1080,
            h1080,
            &mat709,
        );
    }
    let dt_rgba_scalar = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    for _ in 0..pix_iters {
        yuv420p_to_rgba8(
            black_box(ya_slice),
            stride,
            black_box(u_slice),
            c_stride,
            black_box(v_slice),
            c_stride,
            black_box(&mut rgba_simd),
            w1080 * 4,
            w1080,
            h1080,
            &mat709,
        );
    }
    let dt_rgba_simd = t1.elapsed().as_secs_f64();
    assert_eq!(rgba_ref, rgba_simd);
    println!(
        "1080p YUV420p -> RGBA8 (BT.709 vst4q_u8)    {:>8.2} Gpx/s   {:>8.2} Gpx/s   {:>5.2}x    EXACT",
        (px1080 * pix_iters as f64) / dt_rgba_scalar / 1e9,
        (px1080 * pix_iters as f64) / dt_rgba_simd / 1e9,
        dt_rgba_scalar / dt_rgba_simd
    );

    let mut nv12_uv = vec![128u8; w1080 * (h1080 / 2)];
    for (i, b) in nv12_uv.iter_mut().enumerate() {
        *b = ((i * 23 + 40) % 200 + 20) as u8;
    }
    let mut nv12_ref = vec![0u8; w1080 * 3 * h1080];
    let mut nv12_simd = vec![0u8; w1080 * 3 * h1080];
    let t0 = Instant::now();
    for _ in 0..pix_iters {
        nv12_to_rgb24_scalar(
            black_box(ya_slice),
            stride,
            black_box(&nv12_uv),
            w1080,
            black_box(&mut nv12_ref),
            w1080 * 3,
            w1080,
            h1080,
            &mat709,
        );
    }
    let dt_nv12_scalar = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    for _ in 0..pix_iters {
        nv12_to_rgb24(
            black_box(ya_slice),
            stride,
            black_box(&nv12_uv),
            w1080,
            black_box(&mut nv12_simd),
            w1080 * 3,
            w1080,
            h1080,
            &mat709,
        );
    }
    let dt_nv12_simd = t1.elapsed().as_secs_f64();
    assert_eq!(nv12_ref, nv12_simd);
    println!(
        "1080p NV12 -> RGB24 (BT.709 vld2_u8 UV)     {:>8.2} Gpx/s   {:>8.2} Gpx/s   {:>5.2}x    EXACT",
        (px1080 * pix_iters as f64) / dt_nv12_scalar / 1e9,
        (px1080 * pix_iters as f64) / dt_nv12_simd / 1e9,
        dt_nv12_scalar / dt_nv12_simd
    );

    let mut y10_le = vec![0u8; w1080 * 2 * h1080];
    for (i, chunk) in y10_le.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        *chunk = (((i * 19 + 300) & 1023) as u16).to_le_bytes();
    }
    let mut dither_ref = vec![0u8; w1080 * h1080];
    let mut dither_simd = vec![0u8; w1080 * h1080];
    let t0 = Instant::now();
    for _ in 0..pix_iters {
        dither_plane_u16le_to_u8_bayer_scalar(
            black_box(&y10_le),
            w1080 * 2,
            black_box(&mut dither_ref),
            w1080,
            w1080,
            h1080,
            10,
        );
    }
    let dt_dith_scalar = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    for _ in 0..pix_iters {
        dither_plane_u16le_to_u8_bayer(
            black_box(&y10_le),
            w1080 * 2,
            black_box(&mut dither_simd),
            w1080,
            w1080,
            h1080,
            10,
        );
    }
    let dt_dith_simd = t1.elapsed().as_secs_f64();
    assert_eq!(dither_ref, dither_simd);
    println!(
        "1080p 10-bit -> 8-bit Bayer 16x16 Dither    {:>8.2} Gpx/s   {:>8.2} Gpx/s   {:>5.2}x    EXACT",
        (px1080 * pix_iters as f64) / dt_dith_scalar / 1e9,
        (px1080 * pix_iters as f64) / dt_dith_simd / 1e9,
        dt_dith_scalar / dt_dith_simd
    );

    let mut y_out_ref = vec![0u8; w1080 * h1080];
    let mut u_out_ref = vec![0u8; (w1080 / 2) * (h1080 / 2)];
    let mut v_out_ref = vec![0u8; (w1080 / 2) * (h1080 / 2)];
    let mut y_out_simd = vec![0u8; w1080 * h1080];
    let mut u_out_simd = vec![0u8; (w1080 / 2) * (h1080 / 2)];
    let mut v_out_simd = vec![0u8; (w1080 / 2) * (h1080 / 2)];

    let t0 = Instant::now();
    for _ in 0..pix_iters {
        rgb24_to_yuv420p_scalar(
            black_box(&rgb_simd),
            w1080 * 3,
            &mut y_out_ref,
            w1080,
            &mut u_out_ref,
            w1080 / 2,
            &mut v_out_ref,
            w1080 / 2,
            w1080,
            h1080,
            &mat709,
        );
    }
    let dt_r2y_scalar = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    for _ in 0..pix_iters {
        rgb24_to_yuv420p(
            black_box(&rgb_simd),
            w1080 * 3,
            &mut y_out_simd,
            w1080,
            &mut u_out_simd,
            w1080 / 2,
            &mut v_out_simd,
            w1080 / 2,
            w1080,
            h1080,
            &mat709,
        );
    }
    let dt_r2y_simd = t1.elapsed().as_secs_f64();
    assert_eq!(y_out_ref, y_out_simd);
    assert_eq!(u_out_ref, u_out_simd);
    assert_eq!(v_out_ref, v_out_simd);
    println!(
        "1080p RGB24 -> YUV420p (BT.709 vld3q_u8)    {:>8.2} Gpx/s   {:>8.2} Gpx/s   {:>5.2}x    EXACT",
        (px1080 * pix_iters as f64) / dt_r2y_scalar / 1e9,
        (px1080 * pix_iters as f64) / dt_r2y_simd / 1e9,
        dt_r2y_scalar / dt_r2y_simd
    );

    let hbank = PolyphaseFilterBank::new(3840, 1920, ScaleFilter::Lanczos3);
    let mut scaled_ref = vec![0u8; 1920];
    let mut scaled_simd = vec![0u8; 1920];
    let scale_rows = 2000usize;
    let t0 = Instant::now();
    for r in 0..scale_rows {
        let row = frame_a.plane(0).row_padded(r % 2160);
        hscale_row_8tap_q14_scalar(black_box(row), black_box(&mut scaled_ref), &hbank);
    }
    let dt_hsc_scalar = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    for r in 0..scale_rows {
        let row = frame_a.plane(0).row_padded(r % 2160);
        hscale_row_8tap_q14(black_box(row), black_box(&mut scaled_simd), &hbank);
    }
    let dt_hsc_simd = t1.elapsed().as_secs_f64();
    assert_eq!(scaled_ref, scaled_simd);
    println!(
        "Lanczos-3 8-Tap H-Scale (4K->1080p vpaddq)  {:>8.2} Gpx/s   {:>8.2} Gpx/s   {:>5.2}x    EXACT",
        (1920.0 * scale_rows as f64) / dt_hsc_scalar / 1e9,
        (1920.0 * scale_rows as f64) / dt_hsc_simd / 1e9,
        dt_hsc_scalar / dt_hsc_simd
    );

    // 16x16 Macroblock SAD (Motion Estimation inner kernel)
    let sad16_iters = 500_000usize;
    let t0 = Instant::now();
    let mut sad16_ref = 0u32;
    for it in 0..sad16_iters {
        let off = (it & 63) * 16;
        sad16_ref ^= black_box(sad_16x16_u8_scalar(
            black_box(&ya_slice[off..]),
            stride,
            black_box(&yb_slice[off..]),
            stride,
        ));
    }
    let dt_sad16_scalar = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let mut sad16_simd = 0u32;
    for it in 0..sad16_iters {
        let off = (it & 63) * 16;
        sad16_simd ^= black_box(sad_16x16_u8(
            black_box(&ya_slice[off..]),
            stride,
            black_box(&yb_slice[off..]),
            stride,
        ));
    }
    let dt_sad16_simd = t1.elapsed().as_secs_f64();
    assert_eq!(sad16_ref, sad16_simd);
    println!(
        "H.264 16x16 Macroblock SAD (vabal_u8 x16)   {:>8.2} Mmb/s   {:>8.2} Mmb/s   {:>5.2}x    EXACT",
        (sad16_iters as f64) / dt_sad16_scalar / 1e6,
        (sad16_iters as f64) / dt_sad16_simd / 1e6,
        dt_sad16_scalar / dt_sad16_simd
    );

    // Single-Pass 5-Order FLAC Fixed Predictor L1 Evaluator
    let flac_block_len = 4096usize;
    let flac_samples: Vec<i32> = (0..flac_block_len)
        .map(|i| (((i * 37) as f64 * 0.05).sin() * 18000.0) as i32)
        .collect();
    let flac_iters = 10_000usize;
    let t0 = Instant::now();
    let mut flac_l1_ref = [0u64; 5];
    for _ in 0..flac_iters {
        flac_l1_ref = black_box(flac_eval_fixed_orders_l1_scalar(black_box(&flac_samples)));
    }
    let dt_flac_scalar = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let mut flac_l1_simd = [0u64; 5];
    for _ in 0..flac_iters {
        flac_l1_simd = black_box(flac_eval_fixed_orders_l1(black_box(&flac_samples)));
    }
    let dt_flac_simd = t1.elapsed().as_secs_f64();
    assert_eq!(flac_l1_ref, flac_l1_simd);
    println!(
        "FLAC 5-Order Fixed L1 Eval (1-pass NEON)    {:>8.2} Gsmp/s  {:>8.2} Gsmp/s  {:>5.2}x    EXACT",
        (flac_block_len * flac_iters) as f64 / dt_flac_scalar / 1e9,
        (flac_block_len * flac_iters) as f64 / dt_flac_simd / 1e9,
        dt_flac_scalar / dt_flac_simd
    );

    let mut dst_1080p_plane = Plane::new(1920, 1080, 1);
    let t_l1 = Instant::now();
    scale_plane(
        frame_a.plane(0),
        &mut dst_1080p_plane,
        ScaleFilter::Lanczos3,
    );
    let l1_fps = 1.0 / t_l1.elapsed().as_secs_f64();
    let q_rep = evaluate_frame_quality(&frame_a, &frame_b, None);

    let t_crc16 = Instant::now();
    let mut c16 = 0u16;
    for _ in 0..20 {
        c16 ^= black_box(crc16_flac(black_box(&annex_b)));
    }
    let crc16_gbps = (annex_b.len() * 20) as f64 / t_crc16.elapsed().as_secs_f64() / 1e9;

    let t_crc32 = Instant::now();
    let mut c32 = 0u32;
    for _ in 0..20 {
        c32 ^= black_box(crc32_mpeg2(black_box(&annex_b)));
    }
    let crc32_gbps = (annex_b.len() * 20) as f64 / t_crc32.elapsed().as_secs_f64() / 1e9;

    let mut bw = BitWriter::with_capacity(1024);
    for v in 0..256u64 {
        bw.write_ue(v);
        bw.write_rice_signed((v as i64) - 128, 4);
    }
    bw.write_rbsp_trailing_bits();
    let bits_buf = bw.finish();
    let mut br = BitReader::new(&bits_buf);
    for v in 0..256u64 {
        assert_eq!(br.read_ue().unwrap(), v);
        assert_eq!(br.read_rice_signed(4).unwrap(), (v as i64) - 128);
    }
    assert!(!br.more_rbsp_data());

    let ts_48h = Timestamp::new(48 * 3600 * 90_000, Rational64::TIMEBASE_MPEG_90K);
    let ts_ns = ts_48h.rescale_to(Rational64::TIMEBASE_NANOS);
    let satd8 = satd_8x8_u8(ya_slice, stride, yb_slice, stride);

    let sps: &[u8] = &[
        0x67, 0x42, 0xC0, 0x0A, 0xD9, 0x01, 0x00, 0x4E, 0xFF, 0x00, 0x01, 0x00, 0x01, 0xE2, 0x44,
        0x54,
    ];
    let pps: &[u8] = &[0x68, 0xCE, 0x3C, 0x80];
    let avcc = build_avcc_from_sps_pps(sps, pps);
    let mut sample_payload = vec![0u8; 65_536];
    sample_payload[0..4].copy_from_slice(&65_532u32.to_be_bytes());
    sample_payload[4] = 0x65;
    for (i, b) in sample_payload[5..].iter_mut().enumerate() {
        *b = ((i * 41 + 7) % 251 + 1) as u8;
    }

    let mut mp4_mux = Mp4Muxer::new();
    let tb_90k = Rational64::TIMEBASE_MPEG_90K;
    mp4_mux.add_track(TrackInfo {
        index: 0,
        track_type: TrackType::Video,
        codec_id: CodecId::H264,
        timebase: tb_90k,
        duration_ticks: Some(256 * 3000),
        extra_data: avcc,
        video: Some(VideoTrackParams {
            width: 3840,
            height: 2160,
            pixel_format: PixelFormat::Yuv420p,
            colorimetry: Colorimetry::bt709_limited(),
            frame_rate: Rational64::FPS_30,
            bit_depth: 8,
            profile: None,
            level: None,
        }),
        audio: None,
        sample_count: 256,
        total_bytes: (256 * sample_payload.len()) as u64,
    });
    for i in 0..256usize {
        let ts = Timestamp::new((i * 3000) as i64, tb_90k);
        let flags = if i.is_multiple_of(30) {
            PacketFlags::KEYFRAME
        } else {
            PacketFlags::NONE
        };
        mp4_mux
            .write_packet(PacketSlice::from_mmap(
                &sample_payload,
                0,
                ts,
                ts,
                3000,
                flags,
                0,
            ))
            .unwrap();
    }
    let mut mp4_container = Vec::with_capacity(17 * 1024 * 1024);
    mp4_mux.finish(&mut mp4_container).unwrap();

    let t_demux = Instant::now();
    let mut zero_copy_bytes = 0usize;
    for _ in 0..25 {
        let mut demux = Mp4Demuxer::open(black_box(&mp4_container)).unwrap();
        let (_kf_idx, _) = demux
            .seek_to_timestamp(0, Timestamp::new(135 * 3000, tb_90k))
            .unwrap();
        let mut demux_all = Mp4Demuxer::open(black_box(&mp4_container)).unwrap();
        while let Some(pkt) = demux_all.next_packet().unwrap() {
            zero_copy_bytes += black_box(pkt.len());
        }
    }
    let mp4_demux_gbps = (zero_copy_bytes as f64) / t_demux.elapsed().as_secs_f64() / 1e9;

    let mut mkv_sink = Vec::with_capacity(mp4_container.len() + 65536);
    let t_mkv = Instant::now();
    for _ in 0..5 {
        mkv_sink.clear();
        transmux_mp4_to_mkv(black_box(&mp4_container), &mut mkv_sink).unwrap();
    }
    let mkv_remux_gbps = (mp4_container.len() * 5) as f64 / t_mkv.elapsed().as_secs_f64() / 1e9;

    let mut ts_sink = Vec::with_capacity(mp4_container.len() + 1024 * 1024);
    let t_ts = Instant::now();
    for _ in 0..5 {
        ts_sink.clear();
        transmux_mp4_to_mpegts(black_box(&mp4_container), &mut ts_sink).unwrap();
    }
    let ts_remux_gbps = (mp4_container.len() * 5) as f64 / t_ts.elapsed().as_secs_f64() / 1e9;

    let mkv_check = MkvDemuxer::open(&mkv_sink).unwrap();
    assert!(mkv_check.has_seek_head());

    // M4/M5 H.264 P_L0_16x16 Motion Estimation + Hardware PMU check
    let mut h264_enc = H264Encoder::new(128, 96, Rational64::FPS_30, 22);
    let mut h264_dec = H264Decoder::new();
    let f0 = synth_video_frame(128, 96, 0);
    let f1 = synth_video_frame(128, 96, 2);
    let au0 = h264_enc.encode_frame(&f0).unwrap();
    let au1 = h264_enc.encode_frame(&f1).unwrap();
    let _ = h264_dec.decode_annex_b_packet(&au0).unwrap();
    let d1 = h264_dec.decode_annex_b_packet(&au1).unwrap().unwrap();
    assert_eq!((d1.width, d1.height), (128, 96));
    let snap = ProcessRusageSnapshot::capture();

    println!(
        "----------------------------------------------------------------------------------------------"
    );
    println!(
        "[CRC Slicing-by-8] FLAC CRC-16: {:.2} GB/s | MPEG-TS CRC-32: {:.2} GB/s (crc16=0x{:04X}, crc32=0x{:08X})",
        crc16_gbps, crc32_gbps, c16, c32
    );
    println!(
        "[CLZ Bitstream]    512 Exp-Golomb ue(v) + Signed Rice(k=4) + RBSP stop-bit verified in {} bytes",
        bits_buf.len()
    );
    println!(
        "[128-bit Time]     48h MPEG-TS 90kHz -> Nanoseconds exact: {} ({} ns), 8x8 SATD = {}",
        ts_ns.format_hms(),
        ts_ns.ticks,
        satd8
    );
    println!(
        "[Pixels & Quality] 4K->1080p Lanczos-3 Q6 L1 Ring: {:.1} fps | 4K Quality: PSNR={:.2} dB, SSIM={:.5}, MS-SSIM={:.5}, VMAF-Lite={:.2}",
        l1_fps, q_rep.psnr_y, q_rep.ssim_y, q_rep.ms_ssim, q_rep.vmaf_lite
    );
    println!(
        "[Containers]       Zero-Copy MP4 Demux: {:.2} GB/s | MP4->MKV: {:.2} GB/s | MP4->MPEG-TS: {:.2} GB/s (faststart=true, seekhead={})",
        mp4_demux_gbps,
        mkv_remux_gbps,
        ts_remux_gbps,
        mkv_check.has_seek_head()
    );
    println!(
        "[Codecs & PMU]     H.264 P_L0_16x16 ME: IDR={} B -> P={} B ({:.1}x) | PMU instr={:.1}M, cyc={:.1}M, footprint={:.1} MiB",
        au0.len(),
        au1.len(),
        (au0.len() as f64) / (au1.len().max(1) as f64),
        (snap.instructions as f64) / 1e6,
        (snap.cycles as f64) / 1e6,
        (snap.phys_footprint_bytes as f64) / (1024.0 * 1024.0)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_args(strs: &[&str]) -> Vec<String> {
        strs.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn cli_end_to_end_all_subcommands_and_containers() {
        let dir = env::temp_dir().join(format!("ragdoll_cli_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let mp4_path = dir.join("test.mp4");
        let mkv_path = dir.join("test.mkv");
        let ts_path = dir.join("test.ts");
        let y4m_path = dir.join("test.y4m");
        let ppm_path = dir.join("frame.ppm");
        let sheet_path = dir.join("sheet.ppm");
        let wav_path = dir.join("audio.wav");
        let flac_path = dir.join("audio.flac");
        let wav_rt_path = dir.join("audio_rt.wav");

        // 1. synth video (.mp4) and audio (.wav)
        run_cli(&vec_args(&[
            "synth",
            mp4_path.to_str().unwrap(),
            "--resolution",
            "160x120",
            "--fps",
            "30",
            "--duration",
            "0.5",
            "--json",
        ]))
        .unwrap();
        assert!(mp4_path.exists());

        run_cli(&vec_args(&[
            "synth",
            wav_path.to_str().unwrap(),
            "--duration",
            "0.25",
        ]))
        .unwrap();
        assert!(wav_path.exists());

        // 2. probe (--gop-map & --json)
        run_cli(&vec_args(&[
            "probe",
            mp4_path.to_str().unwrap(),
            "--gop-map",
        ]))
        .unwrap();
        run_cli(&vec_args(&["probe", mp4_path.to_str().unwrap(), "--json"])).unwrap();

        // 3. remux MP4 -> MKV and MP4 -> MPEG-TS
        run_cli(&vec_args(&[
            "remux",
            mp4_path.to_str().unwrap(),
            mkv_path.to_str().unwrap(),
            "--faststart",
        ]))
        .unwrap();
        assert!(mkv_path.exists());

        run_cli(&vec_args(&[
            "remux",
            mp4_path.to_str().unwrap(),
            ts_path.to_str().unwrap(),
            "--json",
        ]))
        .unwrap();
        assert!(ts_path.exists());

        // 4. transcode MP4 -> scaled Y4M, High Profile CABAC CRF MP4, & WAV <-> FLAC
        let high_mp4_path = dir.join("test_high_crf.mp4");
        run_cli(&vec_args(&[
            "transcode",
            mp4_path.to_str().unwrap(),
            high_mp4_path.to_str().unwrap(),
            "--profile",
            "high",
            "--crf",
            "23",
            "--maxrate",
            "3M",
        ]))
        .unwrap();
        assert!(high_mp4_path.exists());

        run_cli(&vec_args(&[
            "transcode",
            high_mp4_path.to_str().unwrap(),
            y4m_path.to_str().unwrap(),
            "--scale",
            "160x120",
            "--filter",
            "lanczos3",
        ]))
        .unwrap();
        assert!(y4m_path.exists());

        run_cli(&vec_args(&[
            "transcode",
            wav_path.to_str().unwrap(),
            flac_path.to_str().unwrap(),
        ]))
        .unwrap();
        assert!(flac_path.exists());

        run_cli(&vec_args(&[
            "transcode",
            flac_path.to_str().unwrap(),
            wav_rt_path.to_str().unwrap(),
            "--sample-rate",
            "44100",
            "--loudness",
            "-16.0",
            "--true-peak",
            "-1.0",
        ]))
        .unwrap();
        assert!(wav_rt_path.exists());

        let ogg_path = dir.join("audio.ogg");
        run_cli(&vec_args(&[
            "transcode",
            flac_path.to_str().unwrap(),
            ogg_path.to_str().unwrap(),
            "--audio",
            "opus",
        ]))
        .unwrap();
        assert!(ogg_path.exists());

        // 5. extract single frame and 2x2 contact sheet
        run_cli(&vec_args(&[
            "extract",
            mp4_path.to_str().unwrap(),
            "-o",
            ppm_path.to_str().unwrap(),
            "--at",
            "0.1",
        ]))
        .unwrap();
        assert!(ppm_path.exists());

        run_cli(&vec_args(&[
            "extract",
            mp4_path.to_str().unwrap(),
            "-o",
            sheet_path.to_str().unwrap(),
            "--thumbnails",
            "4",
            "--grid",
            "2x2",
            "--json",
        ]))
        .unwrap();
        assert!(sheet_path.exists());

        // 6. metrics between MP4 and Y4M
        run_cli(&vec_args(&[
            "metrics",
            mp4_path.to_str().unwrap(),
            y4m_path.to_str().unwrap(),
            "--json",
        ]))
        .unwrap();

        let _ = std::fs::remove_dir_all(&dir);
    }
}
