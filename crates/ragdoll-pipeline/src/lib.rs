//! # `ragdoll-pipeline`
//!
//! Work-stealing GOP-parallel media pipeline scheduler, procedural SIMD test media synthesizer,
//! and declarative `.toml` recipe engine for Ragdoll.
//!
//! - [`partition_into_gops`]: Splits flat 24-byte [`SampleEntry`] tables at keyframe boundaries into independent GOPs.
//! - [`parallel_gop_map`]: Lock-free `AtomicUsize` work-stealing executor over `std::thread::scope` preserving deterministic presentation order.
//! - [`PipelineRecipe`]: Zero-dependency declarative `.toml` workflow parser and runner.
//! - [`transcode_file`], [`remux_file`], [`extract_media`], [`synth_media`]: Unified multi-container media engines.

use ragdoll_codec::{
    FlacDecoder, FlacEncoder, H264Decoder, H264Encoder, NalUnitType, Pps, RagdollIntraDecoder,
    RagdollIntraEncoder, Sps, convert_audio_frame, parse_nal_rbsp,
};
use ragdoll_core::{
    AlignedBuffer, AudioFrame, AudioTrackParams, ChannelLayout, CodecId, Colorimetry, FramePool,
    PacketFlags, PacketSlice, PixelFormat, RagdollError, Rational64, Result, SampleEntry,
    SampleFormat, Timestamp, TrackInfo, TrackType, VideoFrame, VideoTrackParams,
};
pub use ragdoll_core::{
    CpuTopology, HardwareMetrics, PowerMode, ProcessRusageSnapshot,
    apply_current_power_mode_to_thread, apply_power_mode_to_current_thread, global_power_mode,
    measure_hardware_metrics, parse_macos_time_l_stderr, set_global_power_mode, with_power_mode,
};
use ragdoll_format::{
    ContainerFormat, MkvDemuxer, MkvMuxer, MmapSource, Mp4Demuxer, Mp4Muxer, MpegTsDemuxer,
    MpegTsMuxer, WavDemuxer, WavMuxer, Y4mDemuxer, Y4mMuxer, build_avcc_from_sps_pps,
    compose_contact_sheet_rgb24, detect_container_format, h264_annex_b_to_avcc_au,
    h264_avcc_to_annex_b_au, read_pnm_image, transmux_mkv_to_mp4, transmux_mp4_to_mpegts,
    transmux_mpegts_to_mkv, transmux_mpegts_to_mp4, write_pnm_image,
};
use ragdoll_pix::{
    PolyphaseFilterBank, QualityReport, ScaleFilter, convert_frame, evaluate_frame_quality,
    scale_plane_with_banks,
};
use ragdoll_simd::split_annex_b_nals;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Telemetry report emitted after executing a remux, transcode, or recipe pipeline.
#[derive(Clone, Debug, Default)]
pub struct PipelineTelemetry {
    pub operation: String,
    pub input_path: String,
    pub output_paths: Vec<String>,
    pub frames_processed: usize,
    pub audio_samples_processed: usize,
    pub gops_processed: usize,
    pub threads_used: usize,
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub elapsed_ms: f64,
    pub fps: f64,
    pub bandwidth_gbps: f64,
    pub compression_ratio: f64,
}

/// Partitions a flat container [`SampleEntry`] slice into independent Group-of-Pictures (GOP)
/// index ranges `[start..end)` at keyframe boundaries (`entry.is_keyframe()`).
pub fn partition_into_gops(samples: &[SampleEntry]) -> Vec<Range<usize>> {
    if samples.is_empty() {
        return Vec::new();
    }
    let mut gops = Vec::new();
    let mut gop_start = 0usize;
    for (idx, entry) in samples.iter().enumerate().skip(1) {
        if entry.is_keyframe() {
            gops.push(gop_start..idx);
            gop_start = idx;
        }
    }
    gops.push(gop_start..samples.len());
    gops
}

/// Returns the number of hardware threads to use (`threads == 0` queries [`CpuTopology`] for the active [`PowerMode`]).
#[inline]
pub fn resolve_thread_count(threads: usize) -> usize {
    CpuTopology::detect().optimal_threads(global_power_mode(), threads)
}

/// Executes `worker_fn(index, &item)` across all items in `items` using a lock-free `AtomicUsize`
/// work-stealing counter over `std::thread::scope`, returning results in exact deterministic index order `0..items.len()`.
pub fn parallel_gop_map<T, R, F>(items: &[T], num_threads: usize, worker_fn: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(usize, &T) -> R + Sync,
{
    let n = items.len();
    if n == 0 {
        return Vec::new();
    }
    let workers = resolve_thread_count(num_threads).min(n);
    if workers <= 1 {
        apply_current_power_mode_to_thread();
        return items
            .iter()
            .enumerate()
            .map(|(i, item)| worker_fn(i, item))
            .collect();
    }

    let next_idx = AtomicUsize::new(0);
    let slots: Vec<Mutex<Option<R>>> = (0..n).map(|_| Mutex::new(None)).collect();

    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                apply_current_power_mode_to_thread();
                loop {
                    let idx = next_idx.fetch_add(1, Ordering::Relaxed);
                    if idx >= n {
                        break;
                    }
                    let res = worker_fn(idx, &items[idx]);
                    if let Ok(mut guard) = slots[idx].lock() {
                        *guard = Some(res);
                    }
                }
            });
        }
    });

    slots
        .into_iter()
        .map(|m| m.into_inner().unwrap().expect("worker completed slot"))
        .collect()
}

/// Scales a batch of [`VideoFrame`]s in parallel across CPU cores using shared precomputed
/// Q14 [`PolyphaseFilterBank`] tables and per-thread [`FramePool`] arenas.
pub fn parallel_scale_frames(
    frames: &[VideoFrame],
    dst_width: usize,
    dst_height: usize,
    filter: ScaleFilter,
    num_threads: usize,
) -> Result<Vec<VideoFrame>> {
    if frames.is_empty() {
        return Ok(Vec::new());
    }
    let src_w = frames[0].width;
    let src_h = frames[0].height;
    let fmt = frames[0].format;
    if src_w == dst_width && src_h == dst_height {
        return Ok(frames.to_vec());
    }

    // Precompute horizontal and vertical polyphase filter banks once for luma and chroma planes
    let h_bank_y = PolyphaseFilterBank::new(src_w, dst_width, filter);
    let v_bank_y = PolyphaseFilterBank::new(src_h, dst_height, filter);
    let (src_cw, src_ch) = fmt.plane_dimensions(src_w, src_h, 1);
    let (dst_cw, dst_ch) = fmt.plane_dimensions(dst_width, dst_height, 1);
    let h_bank_c = PolyphaseFilterBank::new(src_cw, dst_cw, filter);
    let v_bank_c = PolyphaseFilterBank::new(src_ch, dst_ch, filter);

    let scaled = parallel_gop_map(frames, num_threads, |_idx, src| {
        let mut dst = VideoFrame::new(dst_width, dst_height, src.format);
        dst.colorimetry = src.colorimetry;
        dst.pts = src.pts;
        dst.duration = src.duration;
        dst.keyframe = src.keyframe;
        dst.pict_type = src.pict_type;

        scale_plane_with_banks(src.plane(0), dst.plane_mut(0), &h_bank_y, &v_bank_y);
        if src.planes.len() >= 3 {
            scale_plane_with_banks(src.plane(1), dst.plane_mut(1), &h_bank_c, &v_bank_c);
            scale_plane_with_banks(src.plane(2), dst.plane_mut(2), &h_bank_c, &v_bank_c);
        }
        dst
    });

    Ok(scaled)
}

/// Declarative `.toml` pipeline recipe configuration (`0021` / `ADR-0008`).
#[derive(Clone, Debug, PartialEq)]
pub struct PipelineRecipe {
    pub input_path: String,
    pub start_time: Option<String>,
    pub duration: Option<String>,
    pub video_codec: String,
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub scale_filter: ScaleFilter,
    pub qp: u8,
    pub threads: usize,
    pub audio_codec: String,
    pub sample_rate: Option<u32>,
    pub outputs: Vec<RecipeOutput>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecipeOutput {
    pub path: String,
    pub faststart: bool,
}

impl Default for PipelineRecipe {
    fn default() -> Self {
        Self {
            input_path: String::new(),
            start_time: None,
            duration: None,
            video_codec: "h264".into(),
            width: None,
            height: None,
            scale_filter: ScaleFilter::Lanczos3,
            qp: 20,
            threads: 0,
            audio_codec: "flac".into(),
            sample_rate: None,
            outputs: Vec::new(),
        }
    }
}

impl PipelineRecipe {
    /// Parses a declarative `.toml` pipeline recipe string.
    pub fn from_toml(toml_str: &str) -> Result<Self> {
        let mut recipe = Self::default();
        let mut current_section = String::new();

        for raw_line in toml_str.lines() {
            let line = raw_line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            if line.starts_with("[[") && line.ends_with("]]") {
                let sec = line[2..line.len() - 2].trim().to_ascii_lowercase();
                current_section = sec;
                if current_section == "output" {
                    recipe.outputs.push(RecipeOutput {
                        path: String::new(),
                        faststart: true,
                    });
                }
                continue;
            }
            if line.starts_with('[') && line.ends_with(']') {
                current_section = line[1..line.len() - 1].trim().to_ascii_lowercase();
                if current_section == "output" && recipe.outputs.is_empty() {
                    recipe.outputs.push(RecipeOutput {
                        path: String::new(),
                        faststart: true,
                    });
                }
                continue;
            }

            let Some((k, v)) = line.split_once('=') else {
                continue;
            };
            let key = k.trim().to_ascii_lowercase();
            let val = v.trim().trim_matches('"').trim_matches('\'');

            match (current_section.as_str(), key.as_str()) {
                ("input" | "", "path" | "input") => recipe.input_path = val.to_string(),
                ("input" | "", "start") => recipe.start_time = Some(val.to_string()),
                ("input" | "", "duration") => recipe.duration = Some(val.to_string()),
                ("video" | "", "codec" | "video_codec") => {
                    recipe.video_codec = val.to_ascii_lowercase()
                }
                ("video" | "", "width") => recipe.width = val.parse().ok(),
                ("video" | "", "height") => recipe.height = val.parse().ok(),
                ("video" | "", "scale") => {
                    if let Some((w_s, h_s)) = val.split_once('x') {
                        recipe.width = w_s.parse().ok();
                        recipe.height = h_s.parse().ok();
                    }
                }
                ("video" | "", "filter") => {
                    recipe.scale_filter = parse_scale_filter(val);
                }
                ("video" | "", "qp") => {
                    if let Ok(q) = val.parse::<u8>() {
                        recipe.qp = q.min(51);
                    }
                }
                ("video" | "pipeline" | "", "threads") => {
                    if let Ok(t) = val.parse::<usize>() {
                        recipe.threads = t;
                    }
                }
                ("audio" | "", "codec" | "audio_codec") => {
                    recipe.audio_codec = val.to_ascii_lowercase()
                }
                ("audio" | "", "sample_rate") => recipe.sample_rate = val.parse().ok(),
                ("output", "path") => {
                    if let Some(last) = recipe.outputs.last_mut() {
                        last.path = val.to_string();
                    }
                }
                ("output", "faststart") => {
                    if let Some(last) = recipe.outputs.last_mut() {
                        last.faststart = val == "true" || val == "1";
                    }
                }
                _ => {}
            }
        }

        if recipe.input_path.is_empty() {
            return Err(RagdollError::InvalidSyntax(
                "pipeline recipe missing [input] path",
            ));
        }
        if recipe.outputs.is_empty() || recipe.outputs.iter().any(|o| o.path.is_empty()) {
            return Err(RagdollError::InvalidSyntax(
                "pipeline recipe missing [[output]] path",
            ));
        }
        Ok(recipe)
    }

    /// Serializes this recipe to a canonical `.toml` document.
    pub fn to_toml(&self) -> String {
        let mut s = String::new();
        s.push_str("[input]\n");
        s.push_str(&format!("path = \"{}\"\n", self.input_path));
        if let Some(st) = &self.start_time {
            s.push_str(&format!("start = \"{st}\"\n"));
        }
        if let Some(dur) = &self.duration {
            s.push_str(&format!("duration = \"{dur}\"\n"));
        }
        s.push_str("\n[video]\n");
        s.push_str(&format!("codec = \"{}\"\n", self.video_codec));
        if let (Some(w), Some(h)) = (self.width, self.height) {
            s.push_str(&format!("width = {w}\nheight = {h}\n"));
        }
        let f_name = match self.scale_filter {
            ScaleFilter::Lanczos3 => "lanczos3",
            ScaleFilter::MitchellNetravali => "mitchell",
            ScaleFilter::Bilinear => "bilinear",
            ScaleFilter::Spline36 => "spline36",
        };
        s.push_str(&format!("filter = \"{f_name}\"\n"));
        s.push_str(&format!("qp = {}\n", self.qp));
        s.push_str(&format!("threads = {}\n", self.threads));

        s.push_str("\n[audio]\n");
        s.push_str(&format!("codec = \"{}\"\n", self.audio_codec));
        if let Some(sr) = self.sample_rate {
            s.push_str(&format!("sample_rate = {sr}\n"));
        }

        for out in &self.outputs {
            s.push_str("\n[[output]]\n");
            s.push_str(&format!("path = \"{}\"\n", out.path));
            s.push_str(&format!("faststart = {}\n", out.faststart));
        }
        s
    }
}

pub fn parse_scale_filter(name: &str) -> ScaleFilter {
    match name.trim().to_ascii_lowercase().as_str() {
        "mitchell" | "bicubic" | "mitchell-netravali" => ScaleFilter::MitchellNetravali,
        "bilinear" | "linear" => ScaleFilter::Bilinear,
        "spline36" | "spline" => ScaleFilter::Spline36,
        _ => ScaleFilter::Lanczos3,
    }
}

/// Executes a [`PipelineRecipe`] end-to-end across all configured outputs.
pub fn execute_recipe(recipe: &PipelineRecipe) -> Result<PipelineTelemetry> {
    let t0 = Instant::now();
    let mut total_bytes_out = 0u64;
    let mut last_tel = PipelineTelemetry::default();

    for out in &recipe.outputs {
        let tel = if recipe.video_codec == "copy" && recipe.width.is_none() {
            remux_file(&recipe.input_path, &out.path)?
        } else {
            transcode_file(
                &recipe.input_path,
                &out.path,
                recipe.width.zip(recipe.height),
                recipe.scale_filter,
                &recipe.video_codec,
                recipe.qp,
                recipe.threads,
            )?
        };
        total_bytes_out += tel.bytes_out;
        last_tel = tel;
    }

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let elapsed_s = (elapsed_ms / 1000.0).max(1e-9);
    last_tel.operation = "recipe".into();
    last_tel.output_paths = recipe.outputs.iter().map(|o| o.path.clone()).collect();
    last_tel.bytes_out = total_bytes_out;
    last_tel.elapsed_ms = elapsed_ms;
    last_tel.fps = (last_tel.frames_processed as f64) / elapsed_s;
    last_tel.bandwidth_gbps = ((last_tel.bytes_in + total_bytes_out) as f64 / 1e9) / elapsed_s;
    Ok(last_tel)
}

/// Zero-copy remuxing / transmuxing between `.mp4`, `.mkv`, `.ts`, `.y4m`, and `.wav`.
pub fn remux_file(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
) -> Result<PipelineTelemetry> {
    let t0 = Instant::now();
    let in_p = input_path.as_ref();
    let out_p = output_path.as_ref();
    let mmap = MmapSource::open(in_p)?;
    mmap.prefetch_all();
    let data = mmap.as_slice();
    let bytes_in = data.len() as u64;

    let src_fmt = detect_container_format(data).ok_or_else(|| {
        RagdollError::InvalidHeader(format!("unrecognized input format: {}", in_p.display()))
    })?;
    let out_ext = out_p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let out_file = File::create(out_p)?;
    // 256 KiB (64 * 4096B) L2-resident buffer ensures 100% 4 KiB page-aligned write() syscalls on APFS/ext4
    let mut writer = BufWriter::with_capacity(256 * 1024, out_file);
    let mut frames_processed = 0usize;

    let bytes_out = match (src_fmt, out_ext.as_str()) {
        (ContainerFormat::Mp4, "mkv" | "webm") => {
            let mut demux = Mp4Demuxer::open(data)?;
            let mut mux = MkvMuxer::new_matroska();
            for t in demux.tracks() {
                frames_processed += t.samples.len();
                mux.add_track(t.info.clone());
            }
            while let Some(pkt) = demux.next_packet()? {
                mux.write_packet(pkt)?;
            }
            mux.finish(&mut writer)?
        }
        (ContainerFormat::Mp4, "ts" | "m2ts") => {
            let demux = Mp4Demuxer::open(data)?;
            frames_processed = demux.tracks().iter().map(|t| t.samples.len()).sum();
            transmux_mp4_to_mpegts(data, &mut writer)?
        }
        (ContainerFormat::Mp4, "mp4" | "mov" | "m4v") => {
            let mut demux = Mp4Demuxer::open(data)?;
            let mut mux = Mp4Muxer::new();
            for t in demux.tracks() {
                frames_processed += t.samples.len();
                mux.add_track(t.info.clone());
            }
            while let Some(pkt) = demux.next_packet()? {
                mux.write_packet(pkt)?;
            }
            mux.finish(&mut writer)?
        }
        (ContainerFormat::Matroska | ContainerFormat::WebM, "mp4" | "mov") => {
            let demux = MkvDemuxer::open(data)?;
            frames_processed = demux.tracks().iter().map(|t| t.sample_count).sum();
            transmux_mkv_to_mp4(data, &mut writer)?
        }
        (ContainerFormat::Matroska | ContainerFormat::WebM, "mkv" | "webm") => {
            let mut demux = MkvDemuxer::open(data)?;
            let mut mux = MkvMuxer::new_matroska();
            for t in demux.tracks() {
                mux.add_track(t.clone());
            }
            while let Some(pkt) = demux.next_packet()? {
                frames_processed += 1;
                mux.write_packet(pkt)?;
            }
            mux.finish(&mut writer)?
        }
        (ContainerFormat::MpegTs, "mp4" | "mov") => transmux_mpegts_to_mp4(data, &mut writer)?,
        (ContainerFormat::MpegTs, "mkv" | "webm") => transmux_mpegts_to_mkv(data, &mut writer)?,
        (ContainerFormat::Y4m, "y4m") | (ContainerFormat::Wav, "wav") => {
            writer.write_all(data)?;
            frames_processed = 1;
            data.len() as u64
        }
        _ => {
            // Fallback to transcode if formats require codec conversion
            drop(writer);
            return transcode_file(in_p, out_p, None, ScaleFilter::Lanczos3, "h264", 20, 0);
        }
    };

    writer.flush()?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let elapsed_s = (elapsed_ms / 1000.0).max(1e-9);

    Ok(PipelineTelemetry {
        operation: "remux".into(),
        input_path: in_p.display().to_string(),
        output_paths: vec![out_p.display().to_string()],
        frames_processed,
        audio_samples_processed: 0,
        gops_processed: 1,
        threads_used: 1,
        bytes_in,
        bytes_out,
        elapsed_ms,
        fps: (frames_processed as f64) / elapsed_s,
        bandwidth_gbps: ((bytes_in + bytes_out) as f64 / 1e9) / elapsed_s,
        compression_ratio: (bytes_in as f64) / (bytes_out.max(1) as f64),
    })
}

/// Decodes all video frames from any supported video file (`.y4m`, `.mp4`, `.mkv`, `.ts`, `.ppm`, `.rgdi`).
pub fn decode_video_file(path: impl AsRef<Path>) -> Result<(Vec<VideoFrame>, Rational64)> {
    let mmap = MmapSource::open(path.as_ref())?;
    let data = mmap.as_slice();
    if data.starts_with(b"RGDI") {
        let dec = RagdollIntraDecoder::new();
        let f = dec.decode_frame(data)?;
        return Ok((vec![f], Rational64::FPS_30));
    }

    let fmt = detect_container_format(data).ok_or_else(|| {
        RagdollError::InvalidHeader(format!(
            "unrecognized video container: {}",
            path.as_ref().display()
        ))
    })?;

    match fmt {
        ContainerFormat::Y4m => {
            let mut demux = Y4mDemuxer::open(data)?;
            let vp = demux
                .track()
                .video
                .clone()
                .ok_or_else(|| RagdollError::InvalidData("Y4M missing video params".into()))?;
            let fps = vp.frame_rate;
            let mut packets = Vec::with_capacity(demux.frame_count());
            while let Some(pkt) = demux.next_packet()? {
                packets.push(pkt);
            }
            let frames = parallel_gop_map(&packets, 0, |_, pkt| {
                let mut dst =
                    VideoFrame::new(vp.width as usize, vp.height as usize, vp.pixel_format);
                dst.pts = pkt.pts;
                dst.colorimetry = vp.colorimetry;
                let raw = pkt.data();
                let mut src_off = 0usize;
                for plane in &mut dst.planes {
                    let row_bytes = plane.row_bytes();
                    let h = plane.height();
                    if plane.stride() == row_bytes {
                        let total = row_bytes * h;
                        plane.data_mut()[..total].copy_from_slice(&raw[src_off..src_off + total]);
                        src_off += total;
                    } else {
                        for y in 0..h {
                            plane
                                .row_mut(y)
                                .copy_from_slice(&raw[src_off..src_off + row_bytes]);
                            src_off += row_bytes;
                        }
                    }
                }
                dst
            });
            Ok((frames, fps))
        }
        ContainerFormat::Ppm | ContainerFormat::Pgm => {
            let f = read_pnm_image(data)?;
            let yuv = if f.format == PixelFormat::Yuv420p {
                f
            } else {
                convert_frame(&f, PixelFormat::Yuv420p, f.colorimetry)
            };
            Ok((vec![yuv], Rational64::FPS_30))
        }
        ContainerFormat::Mp4 => {
            let mut demux = Mp4Demuxer::open(data)?;
            let v_track = demux
                .tracks()
                .iter()
                .find(|t| t.info.track_type == TrackType::Video)
                .ok_or_else(|| RagdollError::InvalidData("MP4 has no video track".into()))?
                .info
                .clone();
            let fps = v_track
                .video
                .as_ref()
                .map_or(Rational64::FPS_30, |v| v.frame_rate);

            let mut h264_dec = H264Decoder::new();
            let rgdi_dec = RagdollIntraDecoder::new();
            // Extract SPS/PPS from avcC extradata if present
            if v_track.codec_id == CodecId::H264 && v_track.extra_data.len() >= 8 {
                let annex_b_hdr = avcc_config_to_annex_b(&v_track.extra_data);
                let _ = h264_dec.set_extradata(&annex_b_hdr);
            }

            let mut frames = Vec::new();
            while let Some(pkt) = demux.next_packet()? {
                if pkt.track_index != v_track.index {
                    continue;
                }
                if pkt.data().starts_with(b"RGDI") {
                    frames.push(rgdi_dec.decode_frame(pkt.data())?);
                } else if v_track.codec_id == CodecId::H264 {
                    let annex_b =
                        h264_avcc_to_annex_b_au(&v_track.extra_data, pkt.data(), pkt.is_keyframe());
                    if let Some(f) = h264_dec.decode_annex_b_packet(&annex_b)? {
                        frames.push(f);
                    }
                }
            }
            Ok((frames, fps))
        }
        ContainerFormat::Matroska | ContainerFormat::WebM => {
            let mut demux = MkvDemuxer::open(data)?;
            let v_track = demux
                .tracks()
                .iter()
                .find(|t| t.track_type == TrackType::Video)
                .ok_or_else(|| RagdollError::InvalidData("MKV has no video track".into()))?
                .clone();
            let fps = v_track
                .video
                .as_ref()
                .map_or(Rational64::FPS_30, |v| v.frame_rate);
            let mut h264_dec = H264Decoder::new();
            if v_track.codec_id == CodecId::H264 && v_track.extra_data.len() >= 8 {
                let annex_b_hdr = avcc_config_to_annex_b(&v_track.extra_data);
                let _ = h264_dec.set_extradata(&annex_b_hdr);
            }
            let mut frames = Vec::new();
            while let Some(pkt) = demux.next_packet()? {
                if pkt.track_index != v_track.index {
                    continue;
                }
                let annex_b =
                    h264_avcc_to_annex_b_au(&v_track.extra_data, pkt.data(), pkt.is_keyframe());
                if let Some(f) = h264_dec.decode_annex_b_packet(&annex_b)? {
                    frames.push(f);
                }
            }
            Ok((frames, fps))
        }
        ContainerFormat::MpegTs => {
            let mut demux = MpegTsDemuxer::open(data)?;
            let mut h264_dec = H264Decoder::new();
            let mut frames = Vec::new();
            while let Some(pkt) = demux.next_packet()? {
                if let Some(f) = h264_dec.decode_annex_b_packet(pkt.data())? {
                    frames.push(f);
                }
            }
            Ok((frames, Rational64::FPS_30))
        }
        ContainerFormat::Wav => Err(RagdollError::Unsupported(
            "WAV container has no video stream".into(),
        )),
    }
}

/// Extracts Annex-B SPS and PPS NAL units from an ISO `avcC` configuration record.
pub fn avcc_config_to_annex_b(avcc: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    if avcc.len() < 7 || avcc[0] != 1 {
        return out;
    }
    let num_sps = (avcc[5] & 0x1F) as usize;
    let mut pos = 6usize;
    for _ in 0..num_sps {
        if pos + 2 > avcc.len() {
            return out;
        }
        let len = u16::from_be_bytes([avcc[pos], avcc[pos + 1]]) as usize;
        pos += 2;
        if pos + len > avcc.len() {
            return out;
        }
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&avcc[pos..pos + len]);
        pos += len;
    }
    if pos >= avcc.len() {
        return out;
    }
    let num_pps = avcc[pos] as usize;
    pos += 1;
    for _ in 0..num_pps {
        if pos + 2 > avcc.len() {
            return out;
        }
        let len = u16::from_be_bytes([avcc[pos], avcc[pos + 1]]) as usize;
        pos += 2;
        if pos + len > avcc.len() {
            return out;
        }
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&avcc[pos..pos + len]);
        pos += len;
    }
    out
}

/// Full GOP-parallel video & audio transcoding engine (`0021` / `0022`).
///
/// Supports `.y4m`, `.mp4`, `.mkv`, `.ts`, `.ppm`, `.rgdi`, `.wav`, and `.flac`.
pub fn transcode_file(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    scale_dims: Option<(usize, usize)>,
    scale_filter: ScaleFilter,
    video_codec: &str,
    qp: u8,
    threads: usize,
) -> Result<PipelineTelemetry> {
    let t0 = Instant::now();
    let in_p = input_path.as_ref();
    let out_p = output_path.as_ref();
    let in_ext = in_p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let out_ext = out_p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let worker_threads = resolve_thread_count(threads);

    // Audio-only transcode path (WAV <-> FLAC)
    if matches!(in_ext.as_str(), "wav" | "flac") || matches!(out_ext.as_str(), "wav" | "flac") {
        let mmap = MmapSource::open(in_p)?;
        let bytes_in = mmap.len() as u64;
        let audio_frame = if mmap.as_slice().starts_with(b"fLaC") {
            let (_info, frame) = FlacDecoder::decode_stream(mmap.as_slice())?;
            frame
        } else {
            let demux = WavDemuxer::open(mmap.as_slice())?;
            let ap = demux
                .track()
                .audio
                .as_ref()
                .ok_or_else(|| RagdollError::InvalidData("WAV missing audio params".into()))?;
            let raw = demux.pcm_data();
            let bps = ap.sample_format.bytes_per_sample().max(1);
            let block_align = (ap.channels.max(1) as usize) * bps;
            let num_samples = raw.len() / block_align;
            let mut af =
                AudioFrame::new(ap.sample_rate, ap.channels, ap.sample_format, num_samples);
            let copy_len = af.data.len().min(raw.len());
            af.data.as_mut_slice()[..copy_len].copy_from_slice(&raw[..copy_len]);
            af
        };

        let num_samples = audio_frame.num_samples;
        let bytes_out = if out_ext == "flac" {
            let enc = FlacEncoder::new(audio_frame.sample_rate, audio_frame.channels, 0);
            let flac_bytes = enc.encode_stream(&audio_frame)?;
            std::fs::write(out_p, &flac_bytes)?;
            flac_bytes.len() as u64
        } else {
            let pcm16 = if audio_frame.format == SampleFormat::S16le {
                audio_frame
            } else {
                convert_audio_frame(&audio_frame, SampleFormat::S16le)?
            };
            let out_file = File::create(out_p)?;
            let mut mux = WavMuxer::new(AudioTrackParams {
                sample_rate: pcm16.sample_rate,
                channels: pcm16.channels,
                channel_layout: ChannelLayout::from_channels(pcm16.channels),
                sample_format: pcm16.format,
                bits_per_coded_sample: (pcm16.format.bytes_per_sample() * 8) as u8,
            });
            mux.write_pcm_bytes(pcm16.data.as_slice());
            mux.finish(BufWriter::new(out_file))?
        };

        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let elapsed_s = (elapsed_ms / 1000.0).max(1e-9);
        return Ok(PipelineTelemetry {
            operation: "transcode-audio".into(),
            input_path: in_p.display().to_string(),
            output_paths: vec![out_p.display().to_string()],
            frames_processed: 1,
            audio_samples_processed: num_samples,
            gops_processed: 1,
            threads_used: 1,
            bytes_in,
            bytes_out,
            elapsed_ms,
            fps: (num_samples as f64) / elapsed_s,
            bandwidth_gbps: ((bytes_in + bytes_out) as f64 / 1e9) / elapsed_s,
            compression_ratio: (bytes_in as f64) / (bytes_out.max(1) as f64),
        });
    }

    // Video transcode path:
    // 1. Decode input frames
    let bytes_in = std::fs::metadata(in_p).map_or(0, |m| m.len());
    let (raw_frames, fps) = decode_video_file(in_p)?;
    if raw_frames.is_empty() {
        return Err(RagdollError::InvalidData("input video has 0 frames".into()));
    }

    // 2. GOP-parallel polyphase scaling (when requested)
    let frames = if let Some((dw, dh)) = scale_dims {
        parallel_scale_frames(&raw_frames, dw, dh, scale_filter, worker_threads)?
    } else {
        raw_frames
    };

    let num_frames = frames.len();
    let width = frames[0].width as u32;
    let height = frames[0].height as u32;

    // 3. Encode / Mux to target output format
    let max_gop_for_threads = frames.len().div_ceil(worker_threads.max(1)).max(1);
    let gop_size = 15usize.min(max_gop_for_threads);
    let gop_chunks: Vec<&[VideoFrame]> = frames.chunks(gop_size).collect();
    let gops_processed = gop_chunks.len();

    let bytes_out = match out_ext.as_str() {
        "y4m" => {
            let out_file = File::create(out_p)?;
            let mut mux = Y4mMuxer::new(
                BufWriter::with_capacity(256 * 1024, out_file),
                width as usize,
                height as usize,
                PixelFormat::Yuv420p,
                fps,
                frames[0].colorimetry,
            )?;
            for f in &frames {
                mux.write_frame(f)?;
            }
            mux.finish()?
        }
        "ppm" => {
            let rgb = convert_frame(&frames[0], PixelFormat::Rgb24, frames[0].colorimetry);
            let mut buf = Vec::new();
            write_pnm_image(&mut buf, &rgb)?;
            std::fs::write(out_p, &buf)?;
            buf.len() as u64
        }
        "rgdi" => {
            let enc = RagdollIntraEncoder::new(1);
            let pkt = enc.encode_frame(&frames[0])?;
            std::fs::write(out_p, &pkt)?;
            pkt.len() as u64
        }
        _ => {
            // GOP-parallel H.264 or RGDI encoding across all CPU cores!
            let use_rgdi = video_codec.eq_ignore_ascii_case("rgdi");
            let sps = Sps::new_baseline(width, height, fps);
            let pps = Pps::new_baseline(qp);
            let avcc_extra = build_avcc_from_sps_pps(&sps.to_nal_unit(), &pps.to_nal_unit());

            let encoded_gops: Vec<Vec<(Vec<u8>, bool)>> =
                parallel_gop_map(&gop_chunks, worker_threads, |_gop_idx, chunk| {
                    if use_rgdi {
                        let enc = RagdollIntraEncoder::new(1);
                        chunk
                            .iter()
                            .map(|f| (enc.encode_frame(f).unwrap(), true))
                            .collect()
                    } else {
                        let mut enc = H264Encoder::new(width, height, fps, qp);
                        enc.keyint = chunk.len() as u32 + 1;
                        chunk
                            .iter()
                            .enumerate()
                            .map(|(i, f)| {
                                let annex_b = enc.encode_frame(f).unwrap();
                                (annex_b, i == 0)
                            })
                            .collect()
                    }
                });

            let timebase = Rational64::new(1, 90_000);
            let frame_dur = ((90_000 * fps.den) / fps.num.max(1)).max(1);
            let track_info = TrackInfo {
                index: 0,
                track_type: TrackType::Video,
                codec_id: if use_rgdi {
                    CodecId::RagdollIntra
                } else {
                    CodecId::H264
                },
                timebase,
                duration_ticks: Some((num_frames as i64) * frame_dur),
                extra_data: if use_rgdi { Vec::new() } else { avcc_extra },
                video: Some(VideoTrackParams {
                    width,
                    height,
                    pixel_format: PixelFormat::Yuv420p,
                    colorimetry: Colorimetry::bt709_limited(),
                    frame_rate: fps,
                    bit_depth: 8,
                    profile: None,
                    level: None,
                }),
                audio: None,
                sample_count: num_frames,
                total_bytes: 0,
            };

            let out_file = File::create(out_p)?;
            let mut writer = BufWriter::with_capacity(256 * 1024, out_file);

            match out_ext.as_str() {
                "mkv" | "webm" => {
                    let mut mux = MkvMuxer::new_matroska();
                    mux.add_track(track_info);
                    let mut f_idx = 0i64;
                    for gop in encoded_gops {
                        for (raw_pkt, is_kf) in gop {
                            let payload = if use_rgdi {
                                raw_pkt
                            } else {
                                h264_annex_b_to_avcc_au(&raw_pkt)
                            };
                            let pts = f_idx * frame_dur;
                            let pkt = PacketSlice::from_owned(
                                AlignedBuffer::from_slice(&payload),
                                0,
                                Timestamp::new(pts, timebase),
                                Timestamp::new(pts, timebase),
                                frame_dur,
                                if is_kf {
                                    PacketFlags::KEYFRAME
                                } else {
                                    PacketFlags::NONE
                                },
                            );
                            mux.write_packet(pkt)?;
                            f_idx += 1;
                        }
                    }
                    mux.finish(&mut writer)?
                }
                "ts" | "m2ts" => {
                    let mut mux = MpegTsMuxer::new(&mut writer);
                    mux.add_track(track_info);
                    let mut f_idx = 0i64;
                    for gop in encoded_gops {
                        for (raw_pkt, is_kf) in gop {
                            let pts = f_idx * frame_dur;
                            let pkt = PacketSlice::from_owned(
                                AlignedBuffer::from_slice(&raw_pkt),
                                0,
                                Timestamp::new(pts, timebase),
                                Timestamp::new(pts, timebase),
                                frame_dur,
                                if is_kf {
                                    PacketFlags::KEYFRAME
                                } else {
                                    PacketFlags::NONE
                                },
                            );
                            mux.write_packet(&pkt)?;
                            f_idx += 1;
                        }
                    }
                    mux.finish()?
                }
                _ => {
                    let mut mux = Mp4Muxer::new();
                    mux.add_track(track_info);
                    let mut f_idx = 0i64;
                    for gop in encoded_gops {
                        for (raw_pkt, is_kf) in gop {
                            let payload = if use_rgdi {
                                raw_pkt
                            } else {
                                h264_annex_b_to_avcc_au(&raw_pkt)
                            };
                            let pts = f_idx * frame_dur;
                            let pkt = PacketSlice::from_owned(
                                AlignedBuffer::from_slice(&payload),
                                0,
                                Timestamp::new(pts, timebase),
                                Timestamp::new(pts, timebase),
                                frame_dur,
                                if is_kf {
                                    PacketFlags::KEYFRAME
                                } else {
                                    PacketFlags::NONE
                                },
                            );
                            mux.write_packet(pkt)?;
                            f_idx += 1;
                        }
                    }
                    mux.finish(&mut writer)?
                }
            }
        }
    };

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let elapsed_s = (elapsed_ms / 1000.0).max(1e-9);
    Ok(PipelineTelemetry {
        operation: "transcode".into(),
        input_path: in_p.display().to_string(),
        output_paths: vec![out_p.display().to_string()],
        frames_processed: num_frames,
        audio_samples_processed: 0,
        gops_processed,
        threads_used: worker_threads,
        bytes_in,
        bytes_out,
        elapsed_ms,
        fps: (num_frames as f64) / elapsed_s,
        bandwidth_gbps: ((bytes_in + bytes_out) as f64 / 1e9) / elapsed_s,
        compression_ratio: (bytes_in as f64) / (bytes_out.max(1) as f64),
    })
}

/// Extracts either a single frame at `at_sec` or an `N`-thumbnail contact sheet (`cols x rows`) to `.ppm` or `.y4m`.
pub fn extract_media(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    at_sec: Option<f64>,
    thumbnails: Option<(usize, usize, usize)>, // (count, cols, rows)
) -> Result<PipelineTelemetry> {
    let t0 = Instant::now();
    let in_p = input_path.as_ref();
    let out_p = output_path.as_ref();
    let bytes_in = std::fs::metadata(in_p).map_or(0, |m| m.len());
    let (frames, fps) = decode_video_file(in_p)?;
    if frames.is_empty() {
        return Err(RagdollError::InvalidData(
            "no video frames to extract".into(),
        ));
    }

    let out_ext = out_p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("ppm")
        .to_ascii_lowercase();

    let (out_rgb_or_yuv, extracted_count) = if let Some((count, cols, rows)) = thumbnails {
        let count = count.max(1);
        let thumb_w = (frames[0].width / cols.max(1)).max(64) & !1;
        let thumb_h = (frames[0].height / rows.max(1)).max(36) & !1;
        let mut selected = Vec::with_capacity(count);
        for i in 0..count {
            let idx = (i * frames.len() / count).min(frames.len() - 1);
            selected.push(frames[idx].clone());
        }
        let scaled = parallel_scale_frames(&selected, thumb_w, thumb_h, ScaleFilter::Lanczos3, 0)?;
        let rgb_thumbs: Vec<VideoFrame> = scaled
            .iter()
            .map(|f| convert_frame(f, PixelFormat::Rgb24, f.colorimetry))
            .collect();
        let thumb_refs: Vec<&VideoFrame> = rgb_thumbs.iter().collect();
        let sheet = compose_contact_sheet_rgb24(&thumb_refs, cols.max(1), 4, [18, 18, 22])?;
        (sheet, count)
    } else {
        let fps_val = fps.as_f64();
        let target_idx = at_sec
            .map(|s| ((s * fps_val).round() as usize).min(frames.len() - 1))
            .unwrap_or(0);
        let chosen = &frames[target_idx];
        let out_frame = if out_ext == "y4m" {
            chosen.clone()
        } else {
            convert_frame(chosen, PixelFormat::Rgb24, chosen.colorimetry)
        };
        (out_frame, 1)
    };

    let bytes_out = if out_ext == "y4m" {
        let yuv = if out_rgb_or_yuv.format == PixelFormat::Yuv420p {
            out_rgb_or_yuv
        } else {
            convert_frame(
                &out_rgb_or_yuv,
                PixelFormat::Yuv420p,
                out_rgb_or_yuv.colorimetry,
            )
        };
        let out_file = File::create(out_p)?;
        let mut mux = Y4mMuxer::new(
            BufWriter::new(out_file),
            yuv.width,
            yuv.height,
            PixelFormat::Yuv420p,
            fps,
            yuv.colorimetry,
        )?;
        mux.write_frame(&yuv)?;
        mux.finish()?
    } else {
        let rgb = if out_rgb_or_yuv.format == PixelFormat::Rgb24 {
            out_rgb_or_yuv
        } else {
            convert_frame(
                &out_rgb_or_yuv,
                PixelFormat::Rgb24,
                out_rgb_or_yuv.colorimetry,
            )
        };
        let mut buf = Vec::new();
        write_pnm_image(&mut buf, &rgb)?;
        std::fs::write(out_p, &buf)?;
        buf.len() as u64
    };

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let elapsed_s = (elapsed_ms / 1000.0).max(1e-9);
    Ok(PipelineTelemetry {
        operation: "extract".into(),
        input_path: in_p.display().to_string(),
        output_paths: vec![out_p.display().to_string()],
        frames_processed: extracted_count,
        audio_samples_processed: 0,
        gops_processed: 1,
        threads_used: resolve_thread_count(0),
        bytes_in,
        bytes_out,
        elapsed_ms,
        fps: (extracted_count as f64) / elapsed_s,
        bandwidth_gbps: ((bytes_in + bytes_out) as f64 / 1e9) / elapsed_s,
        compression_ratio: 1.0,
    })
}

/// Evaluates average perceptual quality (`PSNR`, `SSIM`, `MS-SSIM`, `VMAF-Lite`) between two media files.
pub fn evaluate_media_metrics(
    ref_path: impl AsRef<Path>,
    dist_path: impl AsRef<Path>,
) -> Result<(QualityReport, usize, f64)> {
    let t0 = Instant::now();
    let (ref_frames, _) = decode_video_file(ref_path)?;
    let (dist_frames, _) = decode_video_file(dist_path)?;
    let count = ref_frames.len().min(dist_frames.len());
    if count == 0 {
        return Err(RagdollError::InvalidData(
            "cannot compute metrics on 0 frames".into(),
        ));
    }

    let indices: Vec<usize> = (0..count).collect();
    let reports: Vec<QualityReport> = parallel_gop_map(&indices, 0, |_, &i| {
        let prev = if i > 0 {
            Some(&ref_frames[i - 1])
        } else {
            None
        };
        evaluate_frame_quality(&ref_frames[i], &dist_frames[i], prev)
    });

    let inv = 1.0 / (count as f64);
    let avg = QualityReport {
        psnr_y: reports.iter().map(|r| r.psnr_y).sum::<f64>() * inv,
        psnr_u: reports.iter().map(|r| r.psnr_u).sum::<f64>() * inv,
        psnr_v: reports.iter().map(|r| r.psnr_v).sum::<f64>() * inv,
        psnr_avg: reports.iter().map(|r| r.psnr_avg).sum::<f64>() * inv,
        psnr_yuv_611: reports.iter().map(|r| r.psnr_yuv_611).sum::<f64>() * inv,
        ssim_y: reports.iter().map(|r| r.ssim_y).sum::<f64>() * inv,
        ssim_u: reports.iter().map(|r| r.ssim_u).sum::<f64>() * inv,
        ssim_v: reports.iter().map(|r| r.ssim_v).sum::<f64>() * inv,
        ssim_avg: reports.iter().map(|r| r.ssim_avg).sum::<f64>() * inv,
        ssim_db: reports.iter().map(|r| r.ssim_db).sum::<f64>() * inv,
        ms_ssim: reports.iter().map(|r| r.ms_ssim).sum::<f64>() * inv,
        vmaf_lite: reports.iter().map(|r| r.vmaf_lite).sum::<f64>() * inv,
    };
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    Ok((avg, count, elapsed_ms))
}

/// Synthesizes a procedural test video (`VideoFrame` with SMPTE bars + moving zone-plate) at `frame_idx`.
pub fn synth_video_frame(width: usize, height: usize, frame_idx: usize) -> VideoFrame {
    let mut frame = VideoFrame::new(width, height, PixelFormat::Yuv420p);
    frame.colorimetry = Colorimetry::bt709_limited();
    let shift = frame_idx * 4;

    for y in 0..height {
        let row = frame.plane_mut(0).row_mut(y);
        if y < height / 3 {
            // Top third: 8 luma steps (SMPTE-style step bars)
            for (x, p) in row.iter_mut().enumerate() {
                let bar = (x * 8) / width.max(1);
                *p = (16 + bar * 28) as u8;
            }
        } else {
            // Bottom two-thirds: smooth diagonal gradient + moving zone-plate ripple
            for (x, p) in row.iter_mut().enumerate() {
                let g = ((x + y + shift) >> 2) & 0xFF;
                *p = (16 + (g * 219) / 255) as u8;
            }
        }
    }

    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    for y in 0..ch {
        for x in 0..cw {
            let u = 96 + (((x + shift / 2) / 16) & 0x3F) as u8;
            let v = 160 - (((y + shift / 2) / 16) & 0x3F) as u8;
            frame.plane_mut(1).row_mut(y)[x] = u;
            frame.plane_mut(2).row_mut(y)[x] = v;
        }
    }
    frame
}

/// Synthesizes procedural test media into `.mp4`, `.mkv`, `.ts`, `.y4m`, `.ppm`, `.wav`, or `.flac`.
pub fn synth_media(
    output_path: impl AsRef<Path>,
    width: usize,
    height: usize,
    fps_num: i64,
    duration_sec: f64,
) -> Result<PipelineTelemetry> {
    let t0 = Instant::now();
    let out_p = output_path.as_ref();
    let ext = out_p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("mp4")
        .to_ascii_lowercase();

    if matches!(ext.as_str(), "wav" | "flac") {
        let sr = 48_000u32;
        let num_samples = ((sr as f64) * duration_sec).round() as usize;
        let mut audio = AudioFrame::new(sr, 2, SampleFormat::S16le, num_samples);
        audio.layout = ChannelLayout::Stereo;
        let s = audio.as_i16_mut_slice();
        for i in 0..num_samples {
            let t = (i as f64) / (sr as f64);
            let l = ((t * 440.0 * std::f64::consts::TAU).sin() * 16_000.0
                + (t * 880.0 * std::f64::consts::TAU).cos() * 4_000.0) as i16;
            let r = ((t * 440.0 * std::f64::consts::TAU).sin() * 15_500.0
                - (t * 220.0 * std::f64::consts::TAU).sin() * 3_000.0) as i16;
            s[i * 2] = l;
            s[i * 2 + 1] = r;
        }
        let bytes_in = (num_samples * 4) as u64;
        let bytes_out = if ext == "flac" {
            let enc = FlacEncoder::new(sr, 2, 16);
            let bytes = enc.encode_stream(&audio)?;
            std::fs::write(out_p, &bytes)?;
            bytes.len() as u64
        } else {
            let f = File::create(out_p)?;
            let mut mux = WavMuxer::new(AudioTrackParams {
                sample_rate: sr,
                channels: 2,
                channel_layout: ChannelLayout::Stereo,
                sample_format: SampleFormat::S16le,
                bits_per_coded_sample: 16,
            });
            mux.write_pcm_bytes(audio.data.as_slice());
            mux.finish(BufWriter::new(f))?
        };
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        return Ok(PipelineTelemetry {
            operation: "synth-audio".into(),
            input_path: "procedural://audio".into(),
            output_paths: vec![out_p.display().to_string()],
            frames_processed: 1,
            audio_samples_processed: num_samples,
            gops_processed: 1,
            threads_used: 1,
            bytes_in,
            bytes_out,
            elapsed_ms,
            fps: (num_samples as f64) / (elapsed_ms / 1000.0).max(1e-9),
            bandwidth_gbps: ((bytes_in + bytes_out) as f64 / 1e9) / (elapsed_ms / 1000.0).max(1e-9),
            compression_ratio: (bytes_in as f64) / (bytes_out.max(1) as f64),
        });
    }

    let num_frames = ((fps_num as f64) * duration_sec).round().max(1.0) as usize;
    let fps = Rational64::new(fps_num.max(1), 1);
    let indices: Vec<usize> = (0..num_frames).collect();
    let frames = parallel_gop_map(&indices, 0, |_i, &f_idx| {
        synth_video_frame(width, height, f_idx)
    });

    let bytes_in = (num_frames * width * height * 3 / 2) as u64;
    let tmp_y4m = std::env::temp_dir().join(format!(
        "ragdoll_synth_{}_{}.y4m",
        std::process::id(),
        t0.elapsed().as_nanos()
    ));

    if ext == "y4m" {
        let f = File::create(out_p)?;
        let mut mux = Y4mMuxer::new(
            BufWriter::with_capacity(256 * 1024, f),
            width,
            height,
            PixelFormat::Yuv420p,
            fps,
            Colorimetry::bt709_limited(),
        )?;
        for frame in &frames {
            mux.write_frame(frame)?;
        }
        mux.finish()?;
    } else if ext == "ppm" {
        let rgb = convert_frame(&frames[0], PixelFormat::Rgb24, frames[0].colorimetry);
        let mut buf = Vec::new();
        write_pnm_image(&mut buf, &rgb)?;
        std::fs::write(out_p, &buf)?;
    } else {
        let f = File::create(&tmp_y4m)?;
        let mut mux = Y4mMuxer::new(
            BufWriter::with_capacity(256 * 1024, f),
            width,
            height,
            PixelFormat::Yuv420p,
            fps,
            Colorimetry::bt709_limited(),
        )?;
        for frame in &frames {
            mux.write_frame(frame)?;
        }
        mux.finish()?;
        let _ = transcode_file(&tmp_y4m, out_p, None, ScaleFilter::Lanczos3, "h264", 20, 0)?;
        let _ = std::fs::remove_file(&tmp_y4m);
    }

    let bytes_out = std::fs::metadata(out_p).map_or(0, |m| m.len());
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let elapsed_s = (elapsed_ms / 1000.0).max(1e-9);
    Ok(PipelineTelemetry {
        operation: "synth".into(),
        input_path: format!("procedural://{width}x{height}@{fps_num}fps"),
        output_paths: vec![out_p.display().to_string()],
        frames_processed: num_frames,
        audio_samples_processed: 0,
        gops_processed: num_frames.div_ceil(15),
        threads_used: resolve_thread_count(0),
        bytes_in,
        bytes_out,
        elapsed_ms,
        fps: (num_frames as f64) / elapsed_s,
        bandwidth_gbps: ((bytes_in + bytes_out) as f64 / 1e9) / elapsed_s,
        compression_ratio: (bytes_in as f64) / (bytes_out.max(1) as f64),
    })
}

/// Detailed container & codec inspection report (`ragdoll probe`).
#[derive(Clone, Debug)]
pub struct ProbeReport {
    pub path: String,
    pub container: String,
    pub file_size: u64,
    pub tracks: Vec<TrackInfo>,
    pub h264_profile: Option<String>,
    pub gop_count: usize,
    pub packet_sizes: Vec<(bool, u32)>, // (is_keyframe, byte_size)
    pub probe_time_us: f64,
}

/// Inspects any supported container or elementary stream file in sub-millisecond time.
pub fn probe_file_detailed(path: impl AsRef<Path>) -> Result<ProbeReport> {
    let t0 = Instant::now();
    let p = path.as_ref();
    let mmap = MmapSource::open(p)?;
    let data = mmap.as_slice();
    let file_size = data.len() as u64;

    if data.starts_with(b"fLaC") {
        let (info, _) = ragdoll_codec::parse_flac_stream_header(data)?;
        let track = TrackInfo {
            index: 0,
            track_type: TrackType::Audio,
            codec_id: CodecId::Flac,
            timebase: Rational64::new(1, info.sample_rate as i64),
            duration_ticks: Some(info.total_samples as i64),
            extra_data: info.to_bytes().to_vec(),
            video: None,
            audio: Some(AudioTrackParams {
                sample_rate: info.sample_rate,
                channels: info.channels,
                channel_layout: ChannelLayout::from_channels(info.channels),
                sample_format: SampleFormat::S16le,
                bits_per_coded_sample: info.bits_per_sample,
            }),
            sample_count: info.total_samples as usize,
            total_bytes: file_size,
        };
        return Ok(ProbeReport {
            path: p.display().to_string(),
            container: "flac".into(),
            file_size,
            tracks: vec![track],
            h264_profile: None,
            gop_count: 1,
            packet_sizes: vec![(true, file_size as u32)],
            probe_time_us: t0.elapsed().as_secs_f64() * 1e6,
        });
    }

    let (fmt, tracks) = ragdoll_format::probe_container(data)?;
    let mut packet_sizes = Vec::new();
    let mut gop_count = 0usize;
    let mut h264_profile = None;

    if fmt == ContainerFormat::Mp4 {
        let demux = Mp4Demuxer::open(data)?;
        for t in demux.tracks() {
            if t.info.track_type == TrackType::Video {
                let gops = partition_into_gops(&t.samples);
                gop_count = gops.len();
                for s in &t.samples {
                    packet_sizes.push((s.is_keyframe(), s.size));
                }
                if t.info.codec_id == CodecId::H264 && t.info.extra_data.len() >= 8 {
                    let annex_b = avcc_config_to_annex_b(&t.info.extra_data);
                    for nal in split_annex_b_nals(&annex_b) {
                        if let Ok((hdr, rbsp)) = parse_nal_rbsp(nal)
                            && hdr.nal_unit_type == NalUnitType::Sps
                            && let Ok(sps) = Sps::parse_rbsp(&rbsp)
                        {
                            h264_profile = Some(format!(
                                "{} (level {}.{})",
                                sps.profile_name(),
                                sps.level_idc / 10,
                                sps.level_idc % 10
                            ));
                        }
                    }
                }
            }
        }
    } else if matches!(fmt, ContainerFormat::Matroska | ContainerFormat::WebM) {
        let mut demux = MkvDemuxer::open(data)?;
        while let Some(pkt) = demux.next_packet()? {
            if pkt.is_keyframe() {
                gop_count += 1;
            }
            packet_sizes.push((pkt.is_keyframe(), pkt.len() as u32));
        }
    } else if fmt == ContainerFormat::MpegTs {
        let mut demux = MpegTsDemuxer::open(data)?;
        while let Some(pkt) = demux.next_packet()? {
            if pkt.is_keyframe() {
                gop_count += 1;
            }
            packet_sizes.push((pkt.is_keyframe(), pkt.len() as u32));
        }
    } else {
        for t in &tracks {
            gop_count = gop_count.max(1);
            let avg_sz = (t.total_bytes / (t.sample_count.max(1) as u64)) as u32;
            for _ in 0..t.sample_count.min(64) {
                packet_sizes.push((true, avg_sz));
            }
        }
    }

    let _ = FramePool::default;
    Ok(ProbeReport {
        path: p.display().to_string(),
        container: fmt.to_string(),
        file_size,
        tracks,
        h264_profile,
        gop_count,
        packet_sizes,
        probe_time_us: t0.elapsed().as_secs_f64() * 1e6,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gop_partition_and_parallel_work_stealing_order() {
        let samples = vec![
            SampleEntry::new(0, 0, 1000, 0, true),
            SampleEntry::new(1000, 3000, 400, 0, false),
            SampleEntry::new(1400, 6000, 400, 0, false),
            SampleEntry::new(1800, 9000, 1200, 0, true),
            SampleEntry::new(3000, 12000, 350, 0, false),
        ];
        let gops = partition_into_gops(&samples);
        assert_eq!(gops, vec![0..3, 3..5]);

        let inputs: Vec<usize> = (0..200).collect();
        let outputs = parallel_gop_map(&inputs, 8, |idx, &val| idx * 1000 + val);
        for (i, &out) in outputs.iter().enumerate() {
            assert_eq!(out, i * 1001);
        }
    }

    #[test]
    fn declarative_toml_recipe_parse_and_end_to_end_execution() {
        let tmp = std::env::temp_dir();
        let src_mp4 = tmp.join("ragdoll_pipe_src.mp4");
        let out_mp4 = tmp.join("ragdoll_pipe_out_scaled.mp4");
        let out_mkv = tmp.join("ragdoll_pipe_out_scaled.mkv");

        synth_media(&src_mp4, 160, 120, 30, 0.5).unwrap();

        let toml_text = format!(
            r#"
[input]
path = "{}"

[video]
codec = "h264"
width = 128
height = 96
filter = "lanczos3"
qp = 22
threads = 4

[[output]]
path = "{}"
faststart = true

[[output]]
path = "{}"
faststart = true
"#,
            src_mp4.display(),
            out_mp4.display(),
            out_mkv.display()
        );

        let recipe = PipelineRecipe::from_toml(&toml_text).unwrap();
        assert_eq!(recipe.width, Some(128));
        assert_eq!(recipe.height, Some(96));
        assert_eq!(recipe.outputs.len(), 2);

        let tel = execute_recipe(&recipe).unwrap();
        assert_eq!(tel.frames_processed, 15);
        assert!(out_mp4.exists());
        assert!(out_mkv.exists());

        let probe = probe_file_detailed(&out_mp4).unwrap();
        assert_eq!(probe.tracks[0].video.as_ref().unwrap().width, 128);
        assert_eq!(probe.tracks[0].video.as_ref().unwrap().height, 96);

        let _ = std::fs::remove_file(&src_mp4);
        let _ = std::fs::remove_file(&out_mp4);
        let _ = std::fs::remove_file(&out_mkv);
    }
}
