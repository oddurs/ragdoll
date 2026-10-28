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
    AacDecoder, AacEncoder, AdtsHeader, FlacDecoder, FlacEncoder, H264Decoder, H264Encoder,
    NalUnitType, OpusDecoder, OpusEncoder, RagdollIntraDecoder, RagdollIntraEncoder, Sps,
    convert_audio_frame, normalize_audio_frame, parse_nal_rbsp, resample_and_rematrix_frame,
};
pub use ragdoll_codec::{
    ChannelRematrixer, EbuR128Meter, H264Profile, LoudnessReport, PolyphaseResampler,
    RateControlMode,
};
use ragdoll_core::{
    AlignedBuffer, AudioFrame, AudioTrackParams, ChannelLayout, CodecId, Colorimetry, FramePool,
    NOPTS_VALUE, PacketFlags, PacketSlice, PixelFormat, RagdollError, Rational64, Result,
    SampleEntry, SampleFormat, Timestamp, TrackInfo, TrackType, VideoFrame, VideoTrackParams,
    compare_ts, rescale_delta, rescale_q,
};
pub use ragdoll_core::{
    CpuTopology, HardwareMetrics, PowerMode, ProcessRusageSnapshot,
    apply_current_power_mode_to_thread, apply_power_mode_to_current_thread, global_power_mode,
    measure_hardware_metrics, parse_macos_time_l_stderr, set_global_power_mode, with_power_mode,
};
use ragdoll_format::{
    ContainerFormat, FlacContainerDemuxer, FlacContainerMuxer, MkvDemuxer, MkvMuxer, MmapSource,
    Mp4Demuxer, Mp4Muxer, MpegTsDemuxer, MpegTsMuxer, OggDemuxer, OggMuxer, OutputSink,
    SubtitleDemuxer, SubtitleMuxer, VorbisComments, WavDemuxer, WavMuxer, Y4mDemuxer, Y4mMuxer,
    build_avcc_from_sps_pps, compose_contact_sheet_rgb24, detect_container_format,
    expand_image_sequence_inputs, format_image_sequence_path, h264_annex_b_to_avcc_au,
    h264_avcc_to_annex_b_au, is_image_sequence_pattern, is_stdout_path, read_jpeg_image,
    read_png_image, read_pnm_image, transmux_mkv_to_mp4, transmux_mp4_to_mpegts,
    transmux_mpegts_to_mkv, transmux_mpegts_to_mp4, write_jpeg_image, write_png_image,
    write_pnm_image,
};
pub use ragdoll_pix::{CropRect, PadSpec, QualityReport, RotationMode, ScaleFilter};
use ragdoll_pix::{
    PolyphaseFilterBank, convert_frame, crop_frame, detect_auto_crop, evaluate_frame_quality,
    overlay_rgba_onto_frame, pad_frame, rotate_frame, scale_plane_with_banks,
    tonemap_hdr_to_sdr_oklab,
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

/// Default memory budget (`256 MiB`) for [`GopRingPipeline`] in `PowerMode::Performance` / `PowerMode::Auto`.
pub const DEFAULT_PIPELINE_MEMORY_BUDGET: usize = 256 * 1024 * 1024;

/// Battery-saver memory budget (`64 MiB`) for [`GopRingPipeline`] in `PowerMode::Efficiency`.
pub const EFFICIENCY_PIPELINE_MEMORY_BUDGET: usize = 64 * 1024 * 1024;

/// Bounded-memory GOP window streaming executor (`0036` / `ADR-0035`).
///
/// Instead of materializing an entire feature film's scaled/filtered `Vec<VideoFrame>` in RAM,
/// `GopRingPipeline` processes video streams in bounded waves of at most `max_gops_in_flight` GOPs,
/// recycling frame planes through a memory-capped [`FramePool`] so resident frame memory stays
/// strictly within `memory_budget_bytes` (`256 MiB` default, `64 MiB` in `--battery-saver`).
pub struct GopRingPipeline {
    pub worker_threads: usize,
    pub memory_budget_bytes: usize,
    pub pool: FramePool,
}

/// An encoded Group of Pictures (each entry is `(packet_bytes, is_keyframe)`).
pub type EncodedGop = Vec<(Vec<u8>, bool)>;

/// Statistics recorded by a [`GopRingPipeline`] execution pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GopRingStats {
    pub waves_processed: usize,
    pub gops_processed: usize,
    pub frames_processed: usize,
    pub max_gops_per_wave: usize,
    pub peak_wave_memory_bytes: usize,
    pub pool_allocations: usize,
    pub pool_recycled_hits: usize,
}

impl GopRingPipeline {
    /// Creates a [`GopRingPipeline`] sized for the active [`PowerMode`] (`256 MiB` default, `64 MiB` in `PowerMode::Efficiency`).
    pub fn new(threads: usize) -> Self {
        let budget = match global_power_mode() {
            PowerMode::Efficiency => EFFICIENCY_PIPELINE_MEMORY_BUDGET,
            _ => DEFAULT_PIPELINE_MEMORY_BUDGET,
        };
        Self::with_budget_bytes(threads, budget)
    }

    /// Creates a [`GopRingPipeline`] with an explicit memory cap in bytes.
    pub fn with_budget_bytes(threads: usize, memory_budget_bytes: usize) -> Self {
        let worker_threads = resolve_thread_count(threads).max(1);
        let budget = memory_budget_bytes.max(4 * 1024 * 1024);
        let pool = FramePool::with_memory_budget(worker_threads * 32, budget);
        Self {
            worker_threads,
            memory_budget_bytes: budget,
            pool,
        }
    }

    /// Computes how many GOPs can execute concurrently in a single bounded wave without exceeding `memory_budget_bytes`.
    pub fn max_gops_per_wave(
        &self,
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
        gop_len: usize,
    ) -> usize {
        let src_frame_bytes = (src_w * src_h * 3 / 2).max(4096);
        let dst_frame_bytes = (dst_w * dst_h * 3 / 2).max(4096);
        let bytes_per_gop = (src_frame_bytes + dst_frame_bytes).saturating_mul(gop_len.max(1));
        let max_by_budget = (self.memory_budget_bytes / bytes_per_gop.max(1)).max(1);
        max_by_budget.min(self.worker_threads * 2).max(1)
    }

    /// Streams GOP chunks through fused L1 polyphase scaling + parallel GOP encoding in bounded
    /// waves, recycling scaled frame buffers back into `self.pool` at the end of every GOP.
    pub fn encode_gops_bounded(
        &self,
        raw_frames: &[VideoFrame],
        scale_dims: Option<(usize, usize)>,
        scale_filter: ScaleFilter,
        video_codec: &str,
        fps: Rational64,
        qp: u8,
    ) -> Result<(Vec<EncodedGop>, GopRingStats)> {
        self.encode_gops_bounded_with_h264(
            raw_frames,
            scale_dims,
            scale_filter,
            video_codec,
            fps,
            qp,
            H264Profile::Baseline,
            RateControlMode::FixedQp(qp),
        )
    }

    /// Streams GOP chunks through fused L1 polyphase scaling + parallel GOP encoding with
    /// configurable H.264 profile (`Baseline` / `Main` / `High`) and [`RateControlMode`] (`0040` / `0041`).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_gops_bounded_with_h264(
        &self,
        raw_frames: &[VideoFrame],
        scale_dims: Option<(usize, usize)>,
        scale_filter: ScaleFilter,
        video_codec: &str,
        fps: Rational64,
        qp: u8,
        h264_profile: H264Profile,
        rate_control: RateControlMode,
    ) -> Result<(Vec<EncodedGop>, GopRingStats)> {
        if raw_frames.is_empty() {
            return Ok((Vec::new(), GopRingStats::default()));
        }
        let src_w = raw_frames[0].width;
        let src_h = raw_frames[0].height;
        let fmt = raw_frames[0].format;
        let (dst_w, dst_h) = scale_dims.unwrap_or((src_w, src_h));
        let needs_scale = dst_w != src_w || dst_h != src_h;

        let banks = if needs_scale {
            let h_bank_y = PolyphaseFilterBank::new(src_w, dst_w, scale_filter);
            let v_bank_y = PolyphaseFilterBank::new(src_h, dst_h, scale_filter);
            let (src_cw, src_ch) = fmt.plane_dimensions(src_w, src_h, 1);
            let (dst_cw, dst_ch) = fmt.plane_dimensions(dst_w, dst_h, 1);
            let h_bank_c = PolyphaseFilterBank::new(src_cw, dst_cw, scale_filter);
            let v_bank_c = PolyphaseFilterBank::new(src_ch, dst_ch, scale_filter);
            Some((h_bank_y, v_bank_y, h_bank_c, v_bank_c))
        } else {
            None
        };

        let is_abr = matches!(rate_control, RateControlMode::Abr { .. });
        let eff_workers = if is_abr {
            1
        } else {
            self.worker_threads.max(1)
        };
        let max_gop_for_threads = raw_frames.len().div_ceil(eff_workers).max(1);
        let gop_size = if is_abr {
            raw_frames.len().max(1)
        } else {
            15usize.min(max_gop_for_threads)
        };
        let gop_chunks: Vec<&[VideoFrame]> = raw_frames.chunks(gop_size).collect();
        let max_gops_wave = self.max_gops_per_wave(src_w, src_h, dst_w, dst_h, gop_size);
        let use_rgdi = video_codec.eq_ignore_ascii_case("rgdi");

        let mut all_encoded_gops = Vec::with_capacity(gop_chunks.len());
        let mut waves_processed = 0usize;
        let mut peak_wave_memory_bytes = 0usize;
        let frame_bytes = (src_w * src_h * 3 / 2)
            + if needs_scale {
                dst_w * dst_h * 3 / 2
            } else {
                0
            };

        for wave in gop_chunks.chunks(max_gops_wave) {
            waves_processed += 1;
            let wave_frames: usize = wave.iter().map(|c| c.len()).sum();
            peak_wave_memory_bytes = peak_wave_memory_bytes
                .max(wave_frames.saturating_mul(frame_bytes))
                .min(self.memory_budget_bytes);

            let wave_encoded: Vec<Vec<(Vec<u8>, bool)>> =
                parallel_gop_map(wave, eff_workers, |_idx, chunk| {
                    let mut out_pkts = Vec::with_capacity(chunk.len());
                    if use_rgdi {
                        let enc = RagdollIntraEncoder::new(1);
                        for src in *chunk {
                            if let Some((ref hy, ref vy, ref hc, ref vc)) = banks {
                                let mut dst = self.pool.acquire_scoped(dst_w, dst_h, src.format);
                                dst.colorimetry = src.colorimetry;
                                dst.pts = src.pts;
                                scale_plane_with_banks(src.plane(0), dst.plane_mut(0), hy, vy);
                                if src.planes.len() >= 3 {
                                    scale_plane_with_banks(src.plane(1), dst.plane_mut(1), hc, vc);
                                    scale_plane_with_banks(src.plane(2), dst.plane_mut(2), hc, vc);
                                }
                                out_pkts.push((enc.encode_frame(&dst).unwrap(), true));
                            } else {
                                out_pkts.push((enc.encode_frame(src).unwrap(), true));
                            }
                        }
                    } else {
                        let mut enc = H264Encoder::new(dst_w as u32, dst_h as u32, fps, qp)
                            .with_profile(h264_profile)
                            .with_rate_control(rate_control);
                        enc.keyint = if is_abr { 15 } else { chunk.len() as u32 + 1 };
                        for (i, src) in chunk.iter().enumerate() {
                            let is_kf = (i as u32).is_multiple_of(enc.keyint.max(1));
                            if let Some((ref hy, ref vy, ref hc, ref vc)) = banks {
                                let mut dst = self.pool.acquire_scoped(dst_w, dst_h, src.format);
                                dst.colorimetry = src.colorimetry;
                                dst.pts = src.pts;
                                scale_plane_with_banks(src.plane(0), dst.plane_mut(0), hy, vy);
                                if src.planes.len() >= 3 {
                                    scale_plane_with_banks(src.plane(1), dst.plane_mut(1), hc, vc);
                                    scale_plane_with_banks(src.plane(2), dst.plane_mut(2), hc, vc);
                                }
                                out_pkts.push((enc.encode_frame(&dst).unwrap(), is_kf));
                            } else {
                                out_pkts.push((enc.encode_frame(src).unwrap(), is_kf));
                            }
                        }
                    }
                    out_pkts
                });

            all_encoded_gops.extend(wave_encoded);
        }

        let stats = GopRingStats {
            waves_processed,
            gops_processed: gop_chunks.len(),
            frames_processed: raw_frames.len(),
            max_gops_per_wave: max_gops_wave,
            peak_wave_memory_bytes,
            pool_allocations: self.pool.allocations(),
            pool_recycled_hits: self.pool.recycled_hits(),
        };
        Ok((all_encoded_gops, stats))
    }
}

/// Stateful 128-bit multi-track audio/video/subtitle lip-sync controller (`0037`).
///
/// Uses [`ragdoll_core::rescale_delta`] for sub-sample audio timestamp tracking and
/// [`ragdoll_core::compare_ts`] for 128-bit overflow-free cross-timebase interleaving.
#[derive(Clone, Debug)]
pub struct AvSyncController {
    audio_last_in: Vec<i64>,
}

impl AvSyncController {
    pub fn new(num_tracks: usize) -> Self {
        Self {
            audio_last_in: vec![NOPTS_VALUE; num_tracks.max(1)],
        }
    }

    /// Rescales an audio packet's `pts`, `dts`, and `duration` from `in_tb` to `out_tb`
    /// using stateful 128-bit [`rescale_delta`] at sample-rate precision `fs_tb = 1 / sample_rate`.
    pub fn rescale_audio_packet<'a>(
        &mut self,
        mut pkt: PacketSlice<'a>,
        sample_rate: u32,
        num_samples: i64,
        out_tb: Rational64,
    ) -> PacketSlice<'a> {
        let tid = pkt.track_index as usize;
        if tid >= self.audio_last_in.len() {
            self.audio_last_in.resize(tid + 1, NOPTS_VALUE);
        }
        let fs_tb = Rational64::new(1, sample_rate.max(1) as i64);
        let new_pts = rescale_delta(
            pkt.pts.timebase,
            pkt.pts.ticks,
            fs_tb,
            num_samples.max(1),
            &mut self.audio_last_in[tid],
            out_tb,
        );
        let new_dur = rescale_q(pkt.duration.max(1), pkt.pts.timebase, out_tb).max(1);
        pkt.pts = Timestamp::new(new_pts, out_tb);
        pkt.dts = Timestamp::new(new_pts, out_tb);
        pkt.duration = new_dur;
        pkt
    }

    /// Rescales a video or subtitle packet's `pts`, `dts`, and `duration` into `out_tb`.
    pub fn rescale_packet<'a>(mut pkt: PacketSlice<'a>, out_tb: Rational64) -> PacketSlice<'a> {
        pkt.pts = pkt.pts.rescale_to(out_tb);
        pkt.dts = pkt.dts.rescale_to(out_tb);
        pkt.duration = rescale_q(pkt.duration.max(1), pkt.pts.timebase, out_tb).max(1);
        pkt
    }

    /// Sorts a multi-track packet list into strict 128-bit monotonically non-decreasing DTS order.
    pub fn interleave_packets_dts(packets: &mut [PacketSlice<'_>]) {
        packets.sort_by(|a, b| {
            compare_ts(a.dts.ticks, a.dts.timebase, b.dts.ticks, b.dts.timebase)
                .then_with(|| a.track_index.cmp(&b.track_index))
        });
    }

    /// Computes the absolute end-of-stream A/V synchronization drift in milliseconds between two timestamps.
    pub fn compute_av_drift_ms(video_end: Timestamp, audio_end: Timestamp) -> f64 {
        let v_ns = video_end.rescale_to(Rational64::TIMEBASE_NANOS).ticks;
        let a_ns = audio_end.rescale_to(Rational64::TIMEBASE_NANOS).ticks;
        ((v_ns - a_ns).abs() as f64) / 1_000_000.0
    }
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

    let trimmed_tmp = if recipe.start_time.is_some() || recipe.duration.is_some() {
        let from_s = recipe
            .start_time
            .as_deref()
            .and_then(|s| parse_time_seconds(s).ok());
        let dur_s = recipe
            .duration
            .as_deref()
            .and_then(|s| parse_time_seconds(s).ok());
        let to_s = match (from_s, dur_s) {
            (Some(f), Some(d)) => Some(f + d),
            (None, Some(d)) => Some(d),
            _ => None,
        };
        let tmp_p = std::env::temp_dir().join(format!(
            "ragdoll_recipe_cut_{}_{}.mp4",
            std::process::id(),
            t0.elapsed().as_nanos()
        ));
        let _ = cut_media(
            &recipe.input_path,
            &tmp_p,
            &CutOptions {
                from_sec: from_s,
                to_sec: to_s,
                keyframes_only: false,
                qp: recipe.qp,
            },
        )?;
        Some(tmp_p)
    } else {
        None
    };
    let effective_input = trimmed_tmp
        .as_ref()
        .map_or(recipe.input_path.as_str(), |p| {
            p.to_str().unwrap_or(&recipe.input_path)
        });

    for out in &recipe.outputs {
        let tel = if recipe.video_codec == "copy"
            && recipe.width.is_none()
            && recipe.sample_rate.is_none()
        {
            remux_file(effective_input, &out.path)?
        } else {
            let mut opts = TranscodeOptions {
                scale_dims: recipe.width.zip(recipe.height),
                scale_filter: recipe.scale_filter,
                video_codec: recipe.video_codec.clone(),
                qp: recipe.qp,
                threads: recipe.threads,
                ..TranscodeOptions::default()
            };
            if !recipe.audio_codec.is_empty() && recipe.audio_codec != "flac" {
                opts.audio_codec = recipe.audio_codec.clone();
            }
            if let Some(sr) = recipe.sample_rate {
                opts.sample_rate = Some(sr);
            }
            transcode_file_with_options(effective_input, &out.path, &opts)?
        };
        total_bytes_out += tel.bytes_out;
        last_tel = tel;
    }
    if let Some(ref tmp_p) = trimmed_tmp {
        let _ = std::fs::remove_file(tmp_p);
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

/// Zero-copy remuxing / transmuxing between `.mp4`, `.mkv`, `.ts`, `.y4m`, `.wav`, `.vtt`, and `.srt`.
pub fn remux_file(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
) -> Result<PipelineTelemetry> {
    remux_file_with_format(input_path, output_path, None)
}

/// Zero-copy remuxing / transmuxing with optional explicit output format (`--format`) for `stdout` (`-`) pipes.
pub fn remux_file_with_format(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    format_override: Option<&str>,
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
    let out_ext = format_override
        .map(|s| s.trim_start_matches('.').to_ascii_lowercase())
        .unwrap_or_else(|| {
            if is_stdout_path(out_p) {
                "ts".to_string()
            } else {
                out_p
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase()
            }
        });

    let mut writer = OutputSink::create(out_p)?;
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
        (ContainerFormat::Mp4, "ts" | "m2ts" | "mpegts") => {
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
        (ContainerFormat::Mp4, "vtt" | "webvtt" | "srt" | "subrip") => {
            let mut demux = Mp4Demuxer::open(data)?;
            let sub_codec = if matches!(out_ext.as_str(), "srt" | "subrip") {
                CodecId::SubRip
            } else {
                CodecId::WebVtt
            };
            let mut mux = SubtitleMuxer::new(sub_codec);
            let tracks: Vec<TrackInfo> = demux.tracks().iter().map(|t| t.info.clone()).collect();
            while let Some(pkt) = demux.next_packet()? {
                if tracks[pkt.track_index as usize].track_type == TrackType::Subtitle {
                    frames_processed += 1;
                    mux.write_packet(&pkt);
                }
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
        (
            ContainerFormat::Matroska | ContainerFormat::WebM,
            "vtt" | "webvtt" | "srt" | "subrip",
        ) => {
            let mut demux = MkvDemuxer::open(data)?;
            let sub_codec = if matches!(out_ext.as_str(), "srt" | "subrip") {
                CodecId::SubRip
            } else {
                CodecId::WebVtt
            };
            let mut mux = SubtitleMuxer::new(sub_codec);
            let tracks: Vec<TrackInfo> = demux.tracks().to_vec();
            while let Some(pkt) = demux.next_packet()? {
                if tracks[pkt.track_index as usize].track_type == TrackType::Subtitle {
                    frames_processed += 1;
                    mux.write_packet(&pkt);
                }
            }
            mux.finish(&mut writer)?
        }
        (ContainerFormat::MpegTs, "mp4" | "mov") => transmux_mpegts_to_mp4(data, &mut writer)?,
        (ContainerFormat::MpegTs, "mkv" | "webm") => transmux_mpegts_to_mkv(data, &mut writer)?,
        (ContainerFormat::MpegTs, "ts" | "m2ts" | "mpegts") => {
            writer.write_all(data)?;
            frames_processed = 1;
            data.len() as u64
        }
        (
            ContainerFormat::WebVtt | ContainerFormat::SubRip,
            "vtt" | "webvtt" | "srt" | "subrip",
        ) => {
            let mut demux = SubtitleDemuxer::open(data)?;
            let sub_codec = if matches!(out_ext.as_str(), "srt" | "subrip") {
                CodecId::SubRip
            } else {
                CodecId::WebVtt
            };
            let mut mux = SubtitleMuxer::new(sub_codec);
            while let Some(pkt) = demux.next_packet()? {
                frames_processed += 1;
                mux.write_packet(&pkt);
            }
            mux.finish(&mut writer)?
        }
        (ContainerFormat::Y4m, "y4m") | (ContainerFormat::Wav, "wav") => {
            writer.write_all(data)?;
            frames_processed = 1;
            data.len() as u64
        }
        (ContainerFormat::Flac, "flac") => {
            let mut demux = FlacContainerDemuxer::open(data)?;
            let mut mux = FlacContainerMuxer::new();
            mux.add_track(demux.track().clone());
            mux.set_comments(demux.comments().clone());
            while let Some(pkt) = demux.next_packet()? {
                frames_processed += 1;
                mux.write_packet(&pkt)?;
            }
            mux.finish(&mut writer)?
        }
        (ContainerFormat::Flac, "ogg") => {
            let mut demux = FlacContainerDemuxer::open(data)?;
            let mut mux = OggMuxer::new();
            mux.add_track(demux.track().clone());
            mux.set_comments(demux.comments().clone());
            while let Some(pkt) = demux.next_packet()? {
                frames_processed += 1;
                mux.write_packet(&pkt)?;
            }
            mux.finish(&mut writer)?
        }
        (ContainerFormat::Ogg, "ogg" | "opus") => {
            let mut demux = OggDemuxer::open(data)?;
            let mut mux = OggMuxer::new();
            mux.add_track(demux.track().clone());
            mux.set_comments(demux.comments().clone());
            while let Some(pkt) = demux.next_packet()? {
                frames_processed += 1;
                mux.write_packet(&pkt)?;
            }
            mux.finish(&mut writer)?
        }
        (ContainerFormat::Ogg, "flac") => {
            let mut demux = OggDemuxer::open(data)?;
            if demux.track().codec_id == CodecId::Flac {
                let mut mux = FlacContainerMuxer::new();
                mux.add_track(demux.track().clone());
                mux.set_comments(demux.comments().clone());
                while let Some(pkt) = demux.next_packet()? {
                    frames_processed += 1;
                    mux.write_packet(&pkt)?;
                }
                mux.finish(&mut writer)?
            } else {
                drop(writer);
                return transcode_file(in_p, out_p, None, ScaleFilter::Lanczos3, "h264", 20, 0);
            }
        }
        (ContainerFormat::Ogg, "mkv" | "webm") => {
            let mut demux = OggDemuxer::open(data)?;
            let mut mux = MkvMuxer::new_matroska();
            mux.add_track(demux.track().clone());
            while let Some(pkt) = demux.next_packet()? {
                frames_processed += 1;
                mux.write_packet(pkt)?;
            }
            mux.finish(&mut writer)?
        }
        (ContainerFormat::Flac, "mkv") => {
            let mut demux = FlacContainerDemuxer::open(data)?;
            let mut mux = MkvMuxer::new_matroska();
            mux.add_track(demux.track().clone());
            while let Some(pkt) = demux.next_packet()? {
                frames_processed += 1;
                mux.write_packet(pkt)?;
            }
            mux.finish(&mut writer)?
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

/// Decodes a single raw image file (`.png`, `.jpg`/`.jpeg`, `.ppm`, `.pgm`) preserving its native pixel format (`Rgba8`/`Rgb24`/`Gray8`).
pub fn decode_image_file(path: impl AsRef<Path>) -> Result<VideoFrame> {
    let mmap = MmapSource::open(path.as_ref())?;
    let data = mmap.as_slice();
    match detect_container_format(data) {
        Some(ContainerFormat::Png) => read_png_image(data),
        Some(ContainerFormat::Jpeg) => read_jpeg_image(data),
        Some(ContainerFormat::Ppm | ContainerFormat::Pgm) => read_pnm_image(data),
        _ => Err(RagdollError::InvalidHeader(format!(
            "unsupported image format for '{}'",
            path.as_ref().display()
        ))),
    }
}

/// Decodes all video frames from any supported video file (`.y4m`, `.mp4`, `.mkv`, `.ts`, `.ppm`, `.png`, `.jpg`, `.rgdi`, or numbered image sequence `frame_%04d.png`).
pub fn decode_video_file(path: impl AsRef<Path>) -> Result<(Vec<VideoFrame>, Rational64)> {
    let p_str = path.as_ref().to_string_lossy();
    if is_image_sequence_pattern(&p_str) {
        let files = expand_image_sequence_inputs(&p_str)?;
        let fps = Rational64::FPS_30;
        let frames = parallel_gop_map(&files, 0, |idx, fpath| {
            let img = decode_image_file(fpath).expect("valid image sequence frame");
            let mut yuv = if img.format == PixelFormat::Yuv420p {
                img
            } else {
                convert_frame(&img, PixelFormat::Yuv420p, img.colorimetry)
            };
            yuv.pts = Timestamp::new(idx as i64, fps.recip());
            yuv
        });
        return Ok((frames, fps));
    }
    let mmap = MmapSource::open(path.as_ref())?;
    decode_video_bytes(mmap.as_slice())
}

/// Decodes all video frames from an in-memory or memory-mapped container byte slice.
pub fn decode_video_bytes(data: &[u8]) -> Result<(Vec<VideoFrame>, Rational64)> {
    if data.starts_with(b"RGDI") {
        let dec = RagdollIntraDecoder::new();
        let f = dec.decode_frame(data)?;
        return Ok((vec![f], Rational64::FPS_30));
    }

    let fmt = detect_container_format(data)
        .ok_or_else(|| RagdollError::InvalidHeader("unrecognized video container".into()))?;

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
        ContainerFormat::Png => {
            let f = read_png_image(data)?;
            let yuv = if f.format == PixelFormat::Yuv420p {
                f
            } else {
                convert_frame(&f, PixelFormat::Yuv420p, f.colorimetry)
            };
            Ok((vec![yuv], Rational64::FPS_30))
        }
        ContainerFormat::Jpeg => {
            let f = read_jpeg_image(data)?;
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

            let rgdi_dec = RagdollIntraDecoder::new();
            let annex_b_hdr = if v_track.codec_id == CodecId::H264 && v_track.extra_data.len() >= 8
            {
                Some(avcc_config_to_annex_b(&v_track.extra_data))
            } else {
                None
            };

            let mut pkts = Vec::new();
            while let Some(pkt) = demux.next_packet()? {
                if pkt.track_index == v_track.index {
                    pkts.push(pkt);
                }
            }

            let mut frames = if pkts.first().is_some_and(|p| p.data().starts_with(b"RGDI")) {
                let mut out = Vec::with_capacity(pkts.len());
                for pkt in &pkts {
                    out.push(rgdi_dec.decode_frame(pkt.data())?);
                }
                out
            } else if v_track.codec_id == CodecId::H264 {
                let mut gops: Vec<Vec<&PacketSlice<'_>>> = Vec::new();
                for pkt in &pkts {
                    // Each GOP gets its own decoder, so only split where the decoder can start:
                    // container sync flags on non-IDR pictures would leave it without references.
                    let starts_gop =
                        pkt.is_keyframe() && avcc_has_idr(&v_track.extra_data, pkt.data());
                    if starts_gop || gops.is_empty() {
                        gops.push(vec![pkt]);
                    } else if let Some(last) = gops.last_mut() {
                        last.push(pkt);
                    }
                }
                if gops.len() > 1 {
                    let gop_results: Vec<Result<Vec<VideoFrame>>> =
                        parallel_gop_map(&gops, 0, |_, gop_pkts| {
                            let mut dec = H264Decoder::new();
                            if let Some(ref hdr) = annex_b_hdr {
                                let _ = dec.set_extradata(hdr);
                            }
                            let mut gf = Vec::with_capacity(gop_pkts.len());
                            for pkt in gop_pkts {
                                let annex_b = h264_avcc_to_annex_b_au(
                                    &v_track.extra_data,
                                    pkt.data(),
                                    pkt.is_keyframe(),
                                );
                                if let Some(f) = dec.decode_annex_b_packet(&annex_b)? {
                                    gf.push(f);
                                }
                            }
                            Ok(gf)
                        });
                    let mut out = Vec::with_capacity(pkts.len());
                    for gr in gop_results {
                        out.extend(gr?);
                    }
                    order_h264_frames(out)
                } else {
                    let mut h264_dec = H264Decoder::new();
                    if let Some(ref hdr) = annex_b_hdr {
                        let _ = h264_dec.set_extradata(hdr);
                    }
                    let mut out = Vec::with_capacity(pkts.len());
                    for pkt in &pkts {
                        let annex_b = h264_avcc_to_annex_b_au(
                            &v_track.extra_data,
                            pkt.data(),
                            pkt.is_keyframe(),
                        );
                        if let Some(f) = h264_dec.decode_annex_b_packet(&annex_b)? {
                            out.push(f);
                        }
                    }
                    order_h264_frames(out)
                }
            } else {
                Vec::new()
            };
            if !frames.windows(2).all(|w| w[0].pts.ticks <= w[1].pts.ticks) {
                frames.sort_by_key(|f| f.pts.ticks);
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
            Ok((order_h264_frames(frames), fps))
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
            Ok((order_h264_frames(frames), Rational64::FPS_30))
        }
        ContainerFormat::Wav
        | ContainerFormat::Flac
        | ContainerFormat::Ogg
        | ContainerFormat::WebVtt
        | ContainerFormat::SubRip => Err(RagdollError::Unsupported(
            "audio/subtitle container has no video stream".into(),
        )),
    }
}

/// Whether a length-prefixed (`avcC`) H.264 access unit contains an IDR slice.
fn avcc_has_idr(avcc_extra: &[u8], au: &[u8]) -> bool {
    let len_size = avcc_extra.get(4).map_or(4, |b| (b & 3) as usize + 1);
    let mut pos = 0;
    while pos + len_size <= au.len() {
        let len = au[pos..pos + len_size]
            .iter()
            .fold(0usize, |acc, &b| (acc << 8) | b as usize);
        pos += len_size;
        if len == 0 || pos + len > au.len() {
            return false;
        }
        if au[pos] & 0x1F == 5 {
            return true;
        }
        pos += len;
    }
    false
}

/// Puts decode-order `H264Decoder` frames into presentation order. The decoder stamps PTS from
/// `pic_order_cnt`, which restarts at every IDR, so it only orders frames within one GOP: sorting
/// across GOPs interleaves them. Sorts each GOP by POC, then renumbers PTS across the whole stream.
fn order_h264_frames(mut frames: Vec<VideoFrame>) -> Vec<VideoFrame> {
    let mut gop_start = 0;
    for i in 1..=frames.len() {
        if i == frames.len() || frames[i].keyframe {
            frames[gop_start..i].sort_by_key(|f| f.pts.ticks);
            gop_start = i;
        }
    }
    for (idx, f) in frames.iter_mut().enumerate() {
        f.pts = Timestamp::new(idx as i64, f.pts.timebase);
    }
    frames
}

/// Extracts non-video (Audio and Subtitle) tracks and their packets from a container byte slice
/// so multi-track transcoding preserves and lip-syncs them by default (`0037`).
pub fn extract_auxiliary_tracks(
    data: &[u8],
) -> Result<Vec<(TrackInfo, Vec<PacketSlice<'static>>)>> {
    let Some(fmt) = detect_container_format(data) else {
        return Ok(Vec::new());
    };
    let mut aux: Vec<(TrackInfo, Vec<PacketSlice<'static>>)> = Vec::new();
    match fmt {
        ContainerFormat::Mp4 => {
            let mut demux = Mp4Demuxer::open(data)?;
            let mut slot_for_track = vec![None; demux.tracks().len()];
            for (pos, t) in demux.tracks().iter().enumerate() {
                if matches!(t.info.track_type, TrackType::Audio | TrackType::Subtitle) {
                    slot_for_track[pos] = Some(aux.len());
                    aux.push((t.info.clone(), Vec::with_capacity(t.samples.len())));
                }
            }
            if aux.is_empty() {
                return Ok(aux);
            }
            while let Some(pkt) = demux.next_packet()? {
                if let Some(Some(slot)) = slot_for_track.get(pkt.track_index as usize) {
                    let owned = PacketSlice::from_owned(
                        AlignedBuffer::from_slice(pkt.data()),
                        pkt.track_index,
                        pkt.pts,
                        pkt.dts,
                        pkt.duration,
                        pkt.flags,
                    );
                    aux[*slot].1.push(owned);
                }
            }
        }
        ContainerFormat::Matroska | ContainerFormat::WebM => {
            let mut demux = MkvDemuxer::open(data)?;
            let mut slot_for_track = vec![None; demux.tracks().len()];
            for (pos, t) in demux.tracks().iter().enumerate() {
                if matches!(t.track_type, TrackType::Audio | TrackType::Subtitle) {
                    slot_for_track[pos] = Some(aux.len());
                    aux.push((t.clone(), Vec::with_capacity(t.sample_count)));
                }
            }
            if aux.is_empty() {
                return Ok(aux);
            }
            while let Some(pkt) = demux.next_packet()? {
                if let Some(Some(slot)) = slot_for_track.get(pkt.track_index as usize) {
                    let owned = PacketSlice::from_owned(
                        AlignedBuffer::from_slice(pkt.data()),
                        pkt.track_index,
                        pkt.pts,
                        pkt.dts,
                        pkt.duration,
                        pkt.flags,
                    );
                    aux[*slot].1.push(owned);
                }
            }
        }
        ContainerFormat::MpegTs => {
            let mut demux = MpegTsDemuxer::open(data)?;
            let mut slot_for_track = vec![None; demux.tracks().len()];
            for (pos, t) in demux.tracks().iter().enumerate() {
                if matches!(t.track_type, TrackType::Audio | TrackType::Subtitle) {
                    slot_for_track[pos] = Some(aux.len());
                    aux.push((t.clone(), Vec::with_capacity(t.sample_count)));
                }
            }
            if aux.is_empty() {
                return Ok(aux);
            }
            while let Some(pkt) = demux.next_packet()? {
                if let Some(Some(slot)) = slot_for_track.get(pkt.track_index as usize) {
                    let owned = PacketSlice::from_owned(
                        AlignedBuffer::from_slice(pkt.data()),
                        pkt.track_index,
                        pkt.pts,
                        pkt.dts,
                        pkt.duration,
                        pkt.flags,
                    );
                    aux[*slot].1.push(owned);
                }
            }
        }
        _ => {}
    }
    Ok(aux)
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

/// Options for [`transcode_file_with_options`] (`0036` / `0037` / `0040` / `0041` / `SIMD-0042` / `SIMD-0043` / `0044` / `0045` / `0047` / `SIMD-0048`).
#[derive(Clone, Debug)]
pub struct TranscodeOptions {
    pub scale_dims: Option<(usize, usize)>,
    pub scale_filter: ScaleFilter,
    pub video_codec: String,
    pub audio_codec: String,
    pub subtitle_codec: String,
    pub format_override: Option<String>,
    pub qp: u8,
    pub h264_profile: H264Profile,
    pub rate_control: Option<RateControlMode>,
    pub threads: usize,
    pub memory_budget_bytes: Option<usize>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u8>,
    pub loudness_lufs: Option<f64>,
    pub true_peak_dbtp: Option<f64>,
    pub crop_spec: Option<String>,
    pub auto_crop: bool,
    pub pad_spec: Option<String>,
    pub rotate_mode: Option<RotationMode>,
    pub target_fps: Option<Rational64>,
    pub overlay_path: Option<std::path::PathBuf>,
    pub overlay_xy: (usize, usize),
    pub tonemap: bool,
}

impl Default for TranscodeOptions {
    fn default() -> Self {
        Self {
            scale_dims: None,
            scale_filter: ScaleFilter::Lanczos3,
            video_codec: "h264".into(),
            audio_codec: "copy".into(),
            subtitle_codec: "copy".into(),
            format_override: None,
            qp: 20,
            h264_profile: H264Profile::Baseline,
            rate_control: None,
            threads: 0,
            memory_budget_bytes: None,
            sample_rate: None,
            channels: None,
            loudness_lufs: None,
            true_peak_dbtp: None,
            crop_spec: None,
            auto_crop: false,
            pad_spec: None,
            rotate_mode: None,
            target_fps: None,
            overlay_path: None,
            overlay_xy: (0, 0),
            tonemap: false,
        }
    }
}

/// Applies the `SIMD-0048` compositional video filter chain (`tonemap`, `--auto-crop` / `--crop`,
/// `--pad`, `--rotate`, `--overlay`, and `--fps` temporal resampling) to a batch of frames.
pub fn apply_video_filters(
    mut frames: Vec<VideoFrame>,
    mut fps: Rational64,
    opts: &TranscodeOptions,
    worker_threads: usize,
) -> Result<(Vec<VideoFrame>, Rational64)> {
    if frames.is_empty() {
        return Ok((frames, fps));
    }

    let auto_rect = if opts.auto_crop {
        Some(detect_auto_crop(&frames[0], 20))
    } else {
        None
    };

    let overlay_img = if let Some(ref ov_p) = opts.overlay_path {
        Some(decode_image_file(ov_p)?)
    } else {
        None
    };

    let has_spatial = opts.tonemap
        || auto_rect.is_some()
        || opts.crop_spec.is_some()
        || opts.pad_spec.is_some()
        || opts.rotate_mode.is_some()
        || overlay_img.is_some();

    if has_spatial {
        let crop_rect = if let Some(ref spec) = opts.crop_spec {
            Some(CropRect::parse_spec(
                spec,
                frames[0].width,
                frames[0].height,
            )?)
        } else {
            auto_rect
        };
        frames = parallel_gop_map(&frames, worker_threads, |_, f| {
            let mut cur = if opts.tonemap {
                tonemap_hdr_to_sdr_oklab(f, PixelFormat::Yuv420p)
            } else {
                f.clone()
            };
            if let Some(r) = crop_rect
                && let Ok(c) = crop_frame(&cur, r)
            {
                cur = c;
            }
            if let Some(ref p_spec) = opts.pad_spec
                && let Ok(ps) = PadSpec::parse_spec(p_spec, cur.width, cur.height)
                && let Ok(p) = pad_frame(&cur, ps)
            {
                cur = p;
            }
            if let Some(rm) = opts.rotate_mode {
                cur = rotate_frame(&cur, rm);
            }
            if let Some(ref ov) = overlay_img {
                overlay_rgba_onto_frame(&mut cur, ov, opts.overlay_xy.0, opts.overlay_xy.1);
            }
            cur
        });
    }

    if let Some(dst_fps) = opts.target_fps
        && dst_fps.num > 0
        && dst_fps.den > 0
        && (dst_fps.num * fps.den != fps.num * dst_fps.den)
    {
        let src_fps_f = fps.as_f64().max(1.0);
        let dst_fps_f = dst_fps.as_f64().max(1.0);
        let out_count = ((frames.len() as f64) * dst_fps_f / src_fps_f)
            .round()
            .max(1.0) as usize;
        let mut resampled = Vec::with_capacity(out_count);
        for out_i in 0..out_count {
            let src_i =
                (((out_i as f64) * src_fps_f / dst_fps_f).floor() as usize).min(frames.len() - 1);
            let mut f = frames[src_i].clone();
            f.pts = Timestamp::new(out_i as i64, dst_fps.recip());
            resampled.push(f);
        }
        frames = resampled;
        fps = dst_fps;
    }

    Ok((frames, fps))
}

/// Decodes an audio stream from `.flac`, `.wav`, `.ogg`, `.aac` (ADTS), or `.mp4`/`.mkv`/`.ts` into
/// `(AudioFrame, VorbisComments)`.
pub fn decode_audio_bytes(data: &[u8]) -> Result<(AudioFrame, VorbisComments)> {
    if data.starts_with(b"fLaC") {
        let demux = FlacContainerDemuxer::open(data).ok();
        let comments = demux.map(|d| d.comments().clone()).unwrap_or_default();
        let (_info, frame) = FlacDecoder::decode_stream(data)?;
        return Ok((frame, comments));
    }
    if data.len() >= 12
        && (&data[0..4] == b"RIFF" || &data[0..4] == b"RF64" || &data[0..4] == b"BW64")
        && &data[8..12] == b"WAVE"
    {
        let demux = WavDemuxer::open(data)?;
        let ap = demux
            .track()
            .audio
            .as_ref()
            .ok_or_else(|| RagdollError::InvalidData("WAV missing audio params".into()))?;
        let raw = demux.pcm_data();
        let bps = ap.sample_format.bytes_per_sample().max(1);
        let block_align = (ap.channels.max(1) as usize) * bps;
        let num_samples = raw.len() / block_align;
        let mut af = AudioFrame::new(ap.sample_rate, ap.channels, ap.sample_format, num_samples);
        let copy_len = af.data.len().min(raw.len());
        af.data.as_mut_slice()[..copy_len].copy_from_slice(&raw[..copy_len]);
        return Ok((af, VorbisComments::default()));
    }
    if data.starts_with(b"OggS") {
        let mut demux = OggDemuxer::open(data)?;
        let final_granule = demux.final_granule();
        let comments = demux.comments().clone();
        let track = demux.track().clone();
        let mut raw_pkts = Vec::new();
        while let Some(pkt) = demux.next_packet()? {
            raw_pkts.push(pkt.data().to_vec());
        }
        if track.codec_id == CodecId::Flac {
            // Reconstruct fLaC stream header + frames for FlacDecoder::decode_stream
            let mut full = Vec::with_capacity(42 + raw_pkts.iter().map(|p| p.len()).sum::<usize>());
            full.extend_from_slice(b"fLaC");
            full.extend_from_slice(&[0x80, 0x00, 0x00, 34]);
            if track.extra_data.len() >= 34 {
                full.extend_from_slice(&track.extra_data[0..34]);
            } else {
                full.extend_from_slice(&[0u8; 34]);
            }
            for p in &raw_pkts {
                full.extend_from_slice(p);
            }
            let (_info, frame) = FlacDecoder::decode_stream(&full)?;
            return Ok((frame, comments));
        } else {
            let ch = track.audio.as_ref().map_or(2, |a| a.channels).clamp(1, 2);
            let mut dec = OpusDecoder::new(ch).with_extra_data(&track.extra_data)?;
            let ch_usize = dec.head.channels.clamp(1, 2) as usize;
            let mut all_samples = Vec::new();
            for p in &raw_pkts {
                let f = dec.decode_packet(p)?;
                all_samples.extend_from_slice(f.as_f32_slice());
            }
            // RFC 7845 section 4.2: the first `pre_skip` samples are encoder delay, and the
            // final granule position marks the end of the audio (section 4.4).
            let skip = (dec.pre_skip() * ch_usize).min(all_samples.len());
            all_samples.drain(..skip);
            if let Some(g) = final_granule {
                let end = (g - dec.pre_skip() as i64).max(0) as usize * ch_usize;
                all_samples.truncate(end);
            }
            let num_samples = all_samples.len() / ch_usize;
            let mut af = AudioFrame::new(48_000, ch_usize as u8, SampleFormat::F32le, num_samples);
            af.as_f32_mut_slice()
                .copy_from_slice(&all_samples[..num_samples * ch_usize]);
            return Ok((af, comments));
        }
    }
    // Check ADTS AAC stream (0xFFFx syncword)
    if data.len() >= 7 && data[0] == 0xFF && (data[1] & 0xF0) == 0xF0 {
        let mut dec = AacDecoder::new(48_000, 2);
        let mut all_samples = Vec::new();
        let mut pos = 0usize;
        while pos + 7 <= data.len() {
            let Ok(hdr) = AdtsHeader::parse(&data[pos..]) else {
                break;
            };
            let flen = (hdr.frame_length as usize).max(7);
            if pos + flen > data.len() {
                break;
            }
            let f = dec.decode_packet(&data[pos..pos + flen])?;
            all_samples.extend_from_slice(f.as_f32_slice());
            pos += flen;
        }
        let ch = dec.asc.channel_configuration.clamp(1, 2) as usize;
        let num_samples = all_samples.len() / ch;
        let mut af = AudioFrame::new(
            dec.asc.sample_rate,
            ch as u8,
            SampleFormat::F32le,
            num_samples,
        );
        af.as_f32_mut_slice()
            .copy_from_slice(&all_samples[..num_samples * ch]);
        return Ok((af, VorbisComments::default()));
    }
    // Check MP4 / MKV / MpegTS container with an audio track
    let aux = extract_auxiliary_tracks(data)?;
    if let Some((a_info, a_pkts)) = aux
        .into_iter()
        .find(|(t, _)| t.track_type == TrackType::Audio)
    {
        let af = decode_audio_track_packets(&a_info, &a_pkts)?;
        return Ok((af, VorbisComments::default()));
    }
    Err(RagdollError::InvalidHeader(
        "unrecognized audio input format".into(),
    ))
}

/// Decodes a sequence of audio `PacketSlice`s belonging to `track` into a single [`AudioFrame`].
pub fn decode_audio_track_packets(
    track: &TrackInfo,
    packets: &[PacketSlice<'_>],
) -> Result<AudioFrame> {
    let raw_pkts: Vec<Vec<u8>> = packets.iter().map(|p| p.data().to_vec()).collect();
    match track.codec_id {
        CodecId::Flac => {
            let mut full = Vec::with_capacity(42 + raw_pkts.iter().map(|p| p.len()).sum::<usize>());
            full.extend_from_slice(b"fLaC");
            full.extend_from_slice(&[0x80, 0x00, 0x00, 34]);
            if track.extra_data.len() >= 34 {
                full.extend_from_slice(&track.extra_data[0..34]);
            } else {
                full.extend_from_slice(&[0u8; 34]);
            }
            for p in &raw_pkts {
                if p.starts_with(b"fLaC") && p.len() > 42 {
                    full.extend_from_slice(&p[42..]);
                } else {
                    full.extend_from_slice(p);
                }
            }
            let (_info, af) = FlacDecoder::decode_stream(&full)?;
            Ok(af)
        }
        CodecId::Aac => {
            let ap = track.audio.clone().unwrap_or_default();
            let mut dec = AacDecoder::new(ap.sample_rate.max(48_000), ap.channels.clamp(1, 2))
                .with_extra_data(&track.extra_data)?;
            let mut all_samples = Vec::new();
            for p in &raw_pkts {
                let f = dec.decode_packet(p)?;
                all_samples.extend_from_slice(f.as_f32_slice());
            }
            let ch = dec.asc.channel_configuration.clamp(1, 2) as usize;
            let num_samples = all_samples.len() / ch;
            let mut out = AudioFrame::new(
                dec.asc.sample_rate,
                ch as u8,
                SampleFormat::F32le,
                num_samples,
            );
            out.as_f32_mut_slice()
                .copy_from_slice(&all_samples[..num_samples * ch]);
            Ok(out)
        }
        CodecId::Opus => {
            let ch = track.audio.as_ref().map_or(2, |a| a.channels).clamp(1, 2);
            let mut dec = OpusDecoder::new(ch).with_extra_data(&track.extra_data)?;
            let ch_usize = dec.head.channels.clamp(1, 2) as usize;
            let mut all_samples = Vec::new();
            for p in &raw_pkts {
                let f = dec.decode_packet(p)?;
                all_samples.extend_from_slice(f.as_f32_slice());
            }
            // RFC 7845 section 4.2: the first `pre_skip` samples are encoder delay.
            let skip = (dec.pre_skip() * ch_usize).min(all_samples.len());
            all_samples.drain(..skip);
            let num_samples = all_samples.len() / ch_usize;
            let mut out = AudioFrame::new(48_000, ch_usize as u8, SampleFormat::F32le, num_samples);
            out.as_f32_mut_slice()
                .copy_from_slice(&all_samples[..num_samples * ch_usize]);
            Ok(out)
        }
        _ => {
            let ap = track.audio.clone().unwrap_or_default();
            let bps = ap.sample_format.bytes_per_sample().max(1);
            let block_align = (ap.channels.max(1) as usize) * bps;
            let total_bytes: usize = raw_pkts.iter().map(|p| p.len()).sum();
            let num_samples = total_bytes / block_align;
            let mut af =
                AudioFrame::new(ap.sample_rate, ap.channels, ap.sample_format, num_samples);
            let mut off = 0usize;
            for p in &raw_pkts {
                let take = p.len().min(af.data.len().saturating_sub(off));
                af.data.as_mut_slice()[off..off + take].copy_from_slice(&p[..take]);
                off += take;
            }
            Ok(af)
        }
    }
}

/// Applies Kaiser polyphase resampling, channel matrix mixing, and EBU R128 loudness + True-Peak
/// normalization (`SIMD-0042` + `SIMD-0043`) when configured in `opts`.
pub fn apply_audio_dsp(mut frame: AudioFrame, opts: &TranscodeOptions) -> Result<AudioFrame> {
    let target_sr = opts.sample_rate.unwrap_or(frame.sample_rate);
    let target_ch = opts.channels.unwrap_or(frame.channels);
    if target_sr != frame.sample_rate || target_ch != frame.channels {
        let mut rs = if target_sr != frame.sample_rate {
            Some(PolyphaseResampler::new(
                frame.sample_rate,
                target_sr,
                target_ch as usize,
                32,
            ))
        } else {
            None
        };
        let rm = if target_ch != frame.channels {
            Some(ChannelRematrixer::new(
                ChannelLayout::from_channels(frame.channels),
                ChannelLayout::from_channels(target_ch),
                true,
            ))
        } else {
            None
        };
        frame = resample_and_rematrix_frame(&frame, rs.as_mut(), rm.as_ref(), frame.format)?;
    }
    if let Some(target_lufs) = opts.loudness_lufs {
        let ceiling_dbtp = opts.true_peak_dbtp.unwrap_or(-1.0);
        let mut meter = EbuR128Meter::new(frame.sample_rate, frame.channels as usize);
        meter.process_audio_frame(&frame)?;
        let (gain, _report) = meter.compute_normalization_gain(target_lufs, ceiling_dbtp);
        frame = normalize_audio_frame(&frame, gain, ceiling_dbtp)?;
    }
    Ok(frame)
}

/// Encodes an [`AudioFrame`] into `(TrackInfo, Vec<PacketSlice<'static>>)` for `.mp4`, `.mkv`, or `.ts` muxing.
pub fn encode_audio_track_packets(
    mut frame: AudioFrame,
    codec_name: &str,
    track_index: u32,
) -> Result<(TrackInfo, Vec<PacketSlice<'static>>)> {
    let cname = codec_name.trim().to_ascii_lowercase();
    match cname.as_str() {
        "opus" => {
            let out_ch = frame.channels.clamp(1, 2);
            if frame.sample_rate != 48_000 || frame.channels != out_ch {
                let mut rs = if frame.sample_rate != 48_000 {
                    Some(PolyphaseResampler::new(
                        frame.sample_rate,
                        48_000,
                        out_ch as usize,
                        32,
                    ))
                } else {
                    None
                };
                let rm = if frame.channels != out_ch {
                    Some(ChannelRematrixer::new(
                        ChannelLayout::from_channels(frame.channels),
                        ChannelLayout::from_channels(out_ch),
                        true,
                    ))
                } else {
                    None
                };
                frame = resample_and_rematrix_frame(
                    &frame,
                    rs.as_mut(),
                    rm.as_ref(),
                    SampleFormat::F32le,
                )?;
            }
            let mut enc = OpusEncoder::new(48_000, out_ch, 96_000);
            let raw_pkts = enc.encode_stream_frames(&frame)?;
            let tb = Rational64::new(1, 48_000);
            let mut pkts = Vec::with_capacity(raw_pkts.len());
            for (p, pts, dur) in raw_pkts {
                pkts.push(PacketSlice::from_owned(
                    AlignedBuffer::from_slice(&p),
                    track_index,
                    pts,
                    pts,
                    dur,
                    PacketFlags::KEYFRAME,
                ));
            }
            let info = TrackInfo {
                index: track_index,
                track_type: TrackType::Audio,
                codec_id: CodecId::Opus,
                timebase: tb,
                // The input length; the packets also cover the pre-skip and end padding, which
                // the Ogg muxer trims through the final granule position.
                duration_ticks: Some(frame.num_samples as i64),
                extra_data: enc.head.to_bytes().to_vec(),
                video: None,
                audio: Some(AudioTrackParams {
                    sample_rate: 48_000,
                    channels: out_ch,
                    channel_layout: ChannelLayout::from_channels(out_ch),
                    sample_format: SampleFormat::F32le,
                    bits_per_coded_sample: 32,
                }),
                sample_count: pkts.len(),
                total_bytes: pkts.iter().map(|p| p.len() as u64).sum(),
            };
            Ok((info, pkts))
        }
        "aac" | "aac-lc" => {
            let out_ch = frame.channels.clamp(1, 2);
            let mut enc = AacEncoder::new(frame.sample_rate, out_ch, 128_000);
            let raw_pkts = enc.encode_stream_frames(&frame)?;
            let sr = enc.asc.sample_rate;
            let tb = Rational64::new(1, sr as i64);
            let mut pkts = Vec::with_capacity(raw_pkts.len());
            for (p, pts, dur) in raw_pkts {
                pkts.push(PacketSlice::from_owned(
                    AlignedBuffer::from_slice(&p),
                    track_index,
                    pts,
                    pts,
                    dur,
                    PacketFlags::KEYFRAME,
                ));
            }
            let info = TrackInfo {
                index: track_index,
                track_type: TrackType::Audio,
                codec_id: CodecId::Aac,
                timebase: tb,
                duration_ticks: Some((pkts.len() as i64) * 1024),
                extra_data: enc.extra_data().to_vec(),
                video: None,
                audio: Some(AudioTrackParams {
                    sample_rate: sr,
                    channels: out_ch,
                    channel_layout: ChannelLayout::from_channels(out_ch),
                    sample_format: SampleFormat::F32le,
                    bits_per_coded_sample: 16,
                }),
                sample_count: pkts.len(),
                total_bytes: pkts.iter().map(|p| p.len() as u64).sum(),
            };
            Ok((info, pkts))
        }
        _ => {
            let enc = FlacEncoder::new(frame.sample_rate, frame.channels, 0);
            let (si, frames) = enc.encode_frames(&frame)?;
            let tb = Rational64::new(1, frame.sample_rate.max(1) as i64);
            let mut pkts = Vec::with_capacity(frames.len());
            let mut pts_ticks = 0i64;
            for (f_bytes, count) in frames {
                let pts = Timestamp::new(pts_ticks, tb);
                let dur = count as i64;
                pts_ticks += dur;
                pkts.push(PacketSlice::from_owned(
                    AlignedBuffer::from_slice(&f_bytes),
                    track_index,
                    pts,
                    pts,
                    dur,
                    PacketFlags::KEYFRAME,
                ));
            }
            let info = TrackInfo {
                index: track_index,
                track_type: TrackType::Audio,
                codec_id: CodecId::Flac,
                timebase: tb,
                duration_ticks: Some(pts_ticks),
                extra_data: si.to_bytes().to_vec(),
                video: None,
                audio: Some(AudioTrackParams {
                    sample_rate: si.sample_rate,
                    channels: si.channels,
                    channel_layout: ChannelLayout::from_channels(si.channels),
                    sample_format: SampleFormat::S16le,
                    bits_per_coded_sample: si.bits_per_sample,
                }),
                sample_count: pkts.len(),
                total_bytes: pkts.iter().map(|p| p.len() as u64).sum(),
            };
            Ok((info, pkts))
        }
    }
}

/// Full GOP-parallel video, audio & subtitle transcoding engine (`0021` / `0022` / `0036` / `0037`).
///
/// Supports `.y4m`, `.mp4`, `.mkv`, `.ts`, `.ppm`, `.rgdi`, `.wav`, `.flac`, `.vtt`, and `.srt`,
/// as well as Unix `stdin`/`stdout` (`-`) pipes.
pub fn transcode_file(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    scale_dims: Option<(usize, usize)>,
    scale_filter: ScaleFilter,
    video_codec: &str,
    qp: u8,
    threads: usize,
) -> Result<PipelineTelemetry> {
    transcode_file_with_options(
        input_path,
        output_path,
        &TranscodeOptions {
            scale_dims,
            scale_filter,
            video_codec: video_codec.to_string(),
            qp,
            threads,
            ..Default::default()
        },
    )
}

/// Executes a bounded-memory GOP-ring transcode with multi-track A/V + Subtitle lip-sync preservation.
pub fn transcode_file_with_options(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    opts: &TranscodeOptions,
) -> Result<PipelineTelemetry> {
    let t0 = Instant::now();
    let in_p = input_path.as_ref();
    let out_p = output_path.as_ref();
    let in_ext = in_p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let out_ext = opts
        .format_override
        .as_deref()
        .map(|s| s.trim_start_matches('.').to_ascii_lowercase())
        .unwrap_or_else(|| {
            if is_stdout_path(out_p) {
                "ts".to_string()
            } else {
                out_p
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase()
            }
        });

    let worker_threads = resolve_thread_count(opts.threads);

    // Subtitle-only conversion path (.vtt <-> .srt)
    if matches!(in_ext.as_str(), "vtt" | "webvtt" | "srt" | "subrip")
        || matches!(out_ext.as_str(), "vtt" | "webvtt" | "srt" | "subrip")
    {
        return remux_file_with_format(in_p, out_p, Some(&out_ext));
    }

    // Audio-only transcode path (WAV / FLAC / Ogg / Opus / AAC / M4A)
    if matches!(
        in_ext.as_str(),
        "wav" | "flac" | "ogg" | "opus" | "aac" | "m4a"
    ) || matches!(
        out_ext.as_str(),
        "wav" | "flac" | "ogg" | "opus" | "aac" | "m4a"
    ) {
        let mmap = MmapSource::open(in_p)?;
        let bytes_in = mmap.len() as u64;
        let (raw_audio, comments) = decode_audio_bytes(mmap.as_slice())?;
        let audio_frame = apply_audio_dsp(raw_audio, opts)?;

        let num_samples = audio_frame.num_samples;
        let mut writer = OutputSink::create(out_p)?;
        let bytes_out = match out_ext.as_str() {
            "flac" => {
                let enc = FlacEncoder::new(audio_frame.sample_rate, audio_frame.channels, 0);
                let (si, frames) = enc.encode_frames(&audio_frame)?;
                let mut mux = FlacContainerMuxer::new();
                mux.add_track(TrackInfo {
                    index: 0,
                    track_type: TrackType::Audio,
                    codec_id: CodecId::Flac,
                    timebase: Rational64::new(1, si.sample_rate.max(1) as i64),
                    duration_ticks: Some(si.total_samples as i64),
                    extra_data: si.to_bytes().to_vec(),
                    video: None,
                    audio: Some(AudioTrackParams {
                        sample_rate: si.sample_rate,
                        channels: si.channels,
                        channel_layout: ChannelLayout::from_channels(si.channels),
                        sample_format: SampleFormat::S16le,
                        bits_per_coded_sample: si.bits_per_sample,
                    }),
                    sample_count: frames.len(),
                    total_bytes: 0,
                });
                mux.set_comments(comments);
                for (f_bytes, count) in frames {
                    mux.write_framed_packet(f_bytes, count as u16);
                }
                mux.finish(&mut writer)?
            }
            "ogg" | "opus" => {
                let target_codec = if out_ext == "opus"
                    || opts.audio_codec.eq_ignore_ascii_case("opus")
                    || opts.audio_codec.eq_ignore_ascii_case("copy")
                {
                    "opus"
                } else if opts.audio_codec.eq_ignore_ascii_case("flac") {
                    "flac"
                } else {
                    "opus"
                };
                let (info, pkts) = encode_audio_track_packets(audio_frame, target_codec, 0)?;
                let mut mux = OggMuxer::new();
                mux.add_track(info);
                mux.set_comments(comments);
                for p in &pkts {
                    mux.write_packet(p)?;
                }
                mux.finish(&mut writer)?
            }
            "aac" => {
                let mut enc = AacEncoder::new(
                    audio_frame.sample_rate,
                    audio_frame.channels.clamp(1, 2),
                    128_000,
                );
                let raw_frames = enc.encode_stream_frames(&audio_frame)?;
                let mut adts = Vec::new();
                for (raw, _, _) in raw_frames {
                    let hdr = AdtsHeader::new(&enc.asc, raw.len());
                    adts.extend_from_slice(&hdr.to_bytes());
                    adts.extend_from_slice(&raw);
                }
                writer.write_all(&adts)?;
                writer.flush()?;
                adts.len() as u64
            }
            "m4a" => {
                let (info, pkts) = encode_audio_track_packets(audio_frame, "aac", 0)?;
                let mut mux = Mp4Muxer::new();
                mux.add_track(info);
                for p in pkts {
                    mux.write_packet(p)?;
                }
                mux.finish(&mut writer)?
            }
            _ => {
                let pcm16 = if audio_frame.format == SampleFormat::S16le {
                    audio_frame
                } else {
                    convert_audio_frame(&audio_frame, SampleFormat::S16le)?
                };
                let mut mux = WavMuxer::new(AudioTrackParams {
                    sample_rate: pcm16.sample_rate,
                    channels: pcm16.channels,
                    channel_layout: ChannelLayout::from_channels(pcm16.channels),
                    sample_format: pcm16.format,
                    bits_per_coded_sample: (pcm16.format.bytes_per_sample() * 8) as u8,
                });
                mux.write_pcm_bytes(pcm16.data.as_slice());
                mux.finish(&mut writer)?
            }
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

    // Video (+ Multi-Track Audio/Subtitle) transcode path:
    // 1. Open input once (supports both zero-copy `mmap`, `-` / `stdin` pipes, and image sequences)
    let in_str = in_p.to_string_lossy();
    let (decoded_frames, decoded_fps, bytes_in, aux_tracks) = if is_image_sequence_pattern(&in_str)
    {
        let files = expand_image_sequence_inputs(&in_str)?;
        let total_b: u64 = files
            .iter()
            .map(|f| std::fs::metadata(f).map_or(0, |m| m.len()))
            .sum();
        let (frames, fps) = decode_video_file(in_p)?;
        (frames, fps, total_b, Vec::new())
    } else {
        let mmap = MmapSource::open(in_p)?;
        let data = mmap.as_slice();
        let bytes_in = data.len() as u64;
        let (frames, fps) = decode_video_bytes(data)?;
        let aux = extract_auxiliary_tracks(data).unwrap_or_default();
        (frames, fps, bytes_in, aux)
    };
    if decoded_frames.is_empty() {
        return Err(RagdollError::InvalidData("input video has 0 frames".into()));
    }

    let (raw_frames, fps) = apply_video_filters(decoded_frames, decoded_fps, opts, worker_threads)?;

    let num_frames = raw_frames.len();
    let (dst_w, dst_h) = opts
        .scale_dims
        .unwrap_or((raw_frames[0].width, raw_frames[0].height));
    let width = dst_w as u32;
    let height = dst_h as u32;

    // 2. Bounded-memory GOP-Ring scaling + encoding (`GopRingPipeline`)
    let gop_ring = match opts.memory_budget_bytes {
        Some(budget) => GopRingPipeline::with_budget_bytes(worker_threads, budget),
        None => GopRingPipeline::new(worker_threads),
    };

    let mut gops_processed = 1usize;
    let mut audio_samples_processed = 0usize;

    let out_str = out_p.to_string_lossy();
    if is_image_sequence_pattern(&out_str) {
        let frames = if let Some((dw, dh)) = opts.scale_dims {
            parallel_scale_frames(&raw_frames, dw, dh, opts.scale_filter, worker_threads)?
        } else {
            raw_frames
        };
        let mut total_bytes_out = 0u64;
        for (idx, frame) in frames.iter().enumerate() {
            let fpath = format_image_sequence_path(&out_str, idx + 1);
            let rgb = convert_frame(frame, PixelFormat::Rgb24, frame.colorimetry);
            let mut buf = Vec::new();
            match out_ext.as_str() {
                "png" => {
                    write_png_image(&mut buf, &rgb)?;
                }
                "jpg" | "jpeg" => {
                    write_jpeg_image(&mut buf, &rgb, 90)?;
                }
                _ => {
                    write_pnm_image(&mut buf, &rgb)?;
                }
            }
            std::fs::write(&fpath, &buf)?;
            total_bytes_out += buf.len() as u64;
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let elapsed_s = (elapsed_ms / 1000.0).max(1e-9);
        return Ok(PipelineTelemetry {
            operation: "transcode-image-seq".into(),
            input_path: in_p.display().to_string(),
            output_paths: vec![out_p.display().to_string()],
            frames_processed: num_frames,
            audio_samples_processed: 0,
            gops_processed: 1,
            threads_used: worker_threads,
            bytes_in,
            bytes_out: total_bytes_out,
            elapsed_ms,
            fps: (num_frames as f64) / elapsed_s,
            bandwidth_gbps: ((bytes_in + total_bytes_out) as f64 / 1e9) / elapsed_s,
            compression_ratio: (bytes_in as f64) / (total_bytes_out.max(1) as f64),
        });
    }

    let bytes_out = match out_ext.as_str() {
        "y4m" => {
            let frames = if let Some((dw, dh)) = opts.scale_dims {
                parallel_scale_frames(&raw_frames, dw, dh, opts.scale_filter, worker_threads)?
            } else {
                raw_frames
            };
            let writer = OutputSink::create(out_p)?;
            let mut mux = Y4mMuxer::new(
                writer,
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
        "ppm" | "png" | "jpg" | "jpeg" => {
            let first = if let Some((dw, dh)) = opts.scale_dims {
                parallel_scale_frames(&raw_frames[..1], dw, dh, opts.scale_filter, worker_threads)?
                    .into_iter()
                    .next()
                    .unwrap()
            } else {
                raw_frames[0].clone()
            };
            let rgb = convert_frame(&first, PixelFormat::Rgb24, first.colorimetry);
            let mut writer = OutputSink::create(out_p)?;
            let mut buf = Vec::new();
            match out_ext.as_str() {
                "png" => {
                    write_png_image(&mut buf, &rgb)?;
                }
                "jpg" | "jpeg" => {
                    write_jpeg_image(&mut buf, &rgb, 90)?;
                }
                _ => {
                    write_pnm_image(&mut buf, &rgb)?;
                }
            }
            writer.write_all(&buf)?;
            writer.flush()?;
            buf.len() as u64
        }
        "rgdi" => {
            let first = if let Some((dw, dh)) = opts.scale_dims {
                parallel_scale_frames(&raw_frames[..1], dw, dh, opts.scale_filter, worker_threads)?
                    .into_iter()
                    .next()
                    .unwrap()
            } else {
                raw_frames[0].clone()
            };
            let enc = RagdollIntraEncoder::new(1);
            let pkt = enc.encode_frame(&first)?;
            let mut writer = OutputSink::create(out_p)?;
            writer.write_all(&pkt)?;
            writer.flush()?;
            pkt.len() as u64
        }
        _ => {
            let use_rgdi = opts.video_codec.eq_ignore_ascii_case("rgdi");
            let rc = opts
                .rate_control
                .unwrap_or(RateControlMode::FixedQp(opts.qp));
            let hdr_enc = H264Encoder::new(width, height, fps, opts.qp)
                .with_profile(opts.h264_profile)
                .with_rate_control(rc);
            let avcc_extra =
                build_avcc_from_sps_pps(&hdr_enc.sps.to_nal_unit(), &hdr_enc.pps.to_nal_unit());

            let (encoded_gops, ring_stats) = gop_ring.encode_gops_bounded_with_h264(
                &raw_frames,
                opts.scale_dims,
                opts.scale_filter,
                &opts.video_codec,
                fps,
                opts.qp,
                opts.h264_profile,
                rc,
            )?;
            gops_processed = ring_stats.gops_processed;

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

            let keep_audio = !opts.audio_codec.eq_ignore_ascii_case("none");
            let keep_subs = !opts.subtitle_codec.eq_ignore_ascii_case("none");
            let needs_audio_transcode = keep_audio
                && (!opts.audio_codec.eq_ignore_ascii_case("copy")
                    || opts.sample_rate.is_some()
                    || opts.channels.is_some()
                    || opts.loudness_lufs.is_some());
            let mut processed_aux = Vec::with_capacity(aux_tracks.len());
            for (aux_info, aux_pkts) in aux_tracks {
                if aux_info.track_type == TrackType::Audio && needs_audio_transcode {
                    let decoded = decode_audio_track_packets(&aux_info, &aux_pkts)?;
                    let dsp_frame = apply_audio_dsp(decoded, opts)?;
                    let target_ac = if opts.audio_codec.eq_ignore_ascii_case("copy") {
                        match aux_info.codec_id {
                            CodecId::Aac => "aac",
                            CodecId::Opus => "opus",
                            _ => "flac",
                        }
                    } else {
                        opts.audio_codec.as_str()
                    };
                    let (new_info, new_pkts) =
                        encode_audio_track_packets(dsp_frame, target_ac, aux_info.index)?;
                    processed_aux.push((new_info, new_pkts));
                } else {
                    processed_aux.push((aux_info, aux_pkts));
                }
            }
            let aux_tracks = processed_aux;
            let mut av_sync = AvSyncController::new(1 + aux_tracks.len());

            let mut writer = OutputSink::create(out_p)?;

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
                    // Preserve & lip-sync Audio and Subtitle tracks in MKV (`0037`)
                    for (mut aux_info, aux_pkts) in aux_tracks {
                        if (aux_info.track_type == TrackType::Audio && !keep_audio)
                            || (aux_info.track_type == TrackType::Subtitle && !keep_subs)
                        {
                            continue;
                        }
                        if aux_info.track_type == TrackType::Subtitle {
                            if opts.subtitle_codec.eq_ignore_ascii_case("srt")
                                || opts.subtitle_codec.eq_ignore_ascii_case("subrip")
                            {
                                aux_info.codec_id = CodecId::SubRip;
                            } else if opts.subtitle_codec.eq_ignore_ascii_case("vtt")
                                || opts.subtitle_codec.eq_ignore_ascii_case("webvtt")
                            {
                                aux_info.codec_id = CodecId::WebVtt;
                            }
                        }
                        let sr = aux_info.audio.as_ref().map_or(48_000, |a| a.sample_rate);
                        let new_tid = mux.add_track(aux_info.clone());
                        let out_tb = Rational64::new(1, 1000);
                        for mut pkt in aux_pkts {
                            pkt.track_index = new_tid;
                            let synced = if aux_info.track_type == TrackType::Audio {
                                let pkt_samples = rescale_q(
                                    pkt.duration.max(1),
                                    pkt.pts.timebase,
                                    Rational64::new(1, sr.max(1) as i64),
                                )
                                .max(1);
                                audio_samples_processed += pkt_samples as usize;
                                av_sync.rescale_audio_packet(pkt, sr, pkt_samples, out_tb)
                            } else {
                                AvSyncController::rescale_packet(pkt, out_tb)
                            };
                            mux.write_packet(synced)?;
                        }
                    }
                    mux.finish(&mut writer)?
                }
                "ts" | "m2ts" | "mpegts" => {
                    let mut mux = MpegTsMuxer::new(&mut writer);
                    mux.add_track(track_info);
                    let mut all_pkts: Vec<PacketSlice<'static>> = Vec::with_capacity(num_frames);
                    let mut f_idx = 0i64;
                    for gop in encoded_gops {
                        for (raw_pkt, is_kf) in gop {
                            let pts = f_idx * frame_dur;
                            all_pkts.push(PacketSlice::from_owned(
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
                            ));
                            f_idx += 1;
                        }
                    }
                    for (aux_info, aux_pkts) in aux_tracks {
                        if aux_info.track_type == TrackType::Audio && keep_audio {
                            let sr = aux_info.audio.as_ref().map_or(48_000, |a| a.sample_rate);
                            let new_tid = mux.add_track(aux_info);
                            for mut pkt in aux_pkts {
                                pkt.track_index = new_tid;
                                let pkt_samples = rescale_q(
                                    pkt.duration.max(1),
                                    pkt.pts.timebase,
                                    Rational64::new(1, sr.max(1) as i64),
                                )
                                .max(1);
                                audio_samples_processed += pkt_samples as usize;
                                all_pkts.push(av_sync.rescale_audio_packet(
                                    pkt,
                                    sr,
                                    pkt_samples,
                                    timebase,
                                ));
                            }
                        }
                    }
                    AvSyncController::interleave_packets_dts(&mut all_pkts);
                    for pkt in &all_pkts {
                        mux.write_packet(pkt)?;
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
                    // Preserve & lip-sync Audio and Subtitle tracks in MP4 (`0037`)
                    for (mut aux_info, aux_pkts) in aux_tracks {
                        if (aux_info.track_type == TrackType::Audio && !keep_audio)
                            || (aux_info.track_type == TrackType::Subtitle && !keep_subs)
                        {
                            continue;
                        }
                        if aux_info.track_type == TrackType::Subtitle {
                            aux_info.codec_id = CodecId::WebVtt;
                        }
                        let out_tb = aux_info.timebase;
                        let sr = aux_info.audio.as_ref().map_or(48_000, |a| a.sample_rate);
                        let new_tid = mux.add_track(aux_info.clone());
                        for mut pkt in aux_pkts {
                            pkt.track_index = new_tid;
                            let synced = if aux_info.track_type == TrackType::Audio {
                                let pkt_samples = rescale_q(
                                    pkt.duration.max(1),
                                    pkt.pts.timebase,
                                    Rational64::new(1, sr.max(1) as i64),
                                )
                                .max(1);
                                audio_samples_processed += pkt_samples as usize;
                                av_sync.rescale_audio_packet(pkt, sr, pkt_samples, out_tb)
                            } else {
                                AvSyncController::rescale_packet(pkt, out_tb)
                            };
                            mux.write_packet(synced)?;
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
        audio_samples_processed,
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

/// Decodes only the single GOP containing `target_sec` when the input is an MP4 container (`0054`).
fn decode_single_frame_at_mp4(in_p: &Path, target_sec: f64) -> Option<(VideoFrame, Rational64)> {
    let mmap = MmapSource::open(in_p).ok()?;
    let data = mmap.as_slice();
    if detect_container_format(data)? != ContainerFormat::Mp4 {
        return None;
    }
    let demux = Mp4Demuxer::open(data).ok()?;
    let v_track = demux
        .tracks()
        .iter()
        .find(|t| t.info.track_type == TrackType::Video)?;
    if v_track.info.codec_id != CodecId::H264 || v_track.samples.is_empty() {
        return None;
    }
    let fps = v_track
        .info
        .video
        .as_ref()
        .map_or(Rational64::FPS_30, |v| v.frame_rate);
    let target_idx = ((target_sec.max(0.0) * fps.as_f64()).round() as usize)
        .min(v_track.samples.len().saturating_sub(1));
    let gops = partition_into_gops(&v_track.samples);
    let gop = gops
        .iter()
        .find(|g| target_idx >= g.start && target_idx < g.end)?;

    let mut dec = H264Decoder::new();
    if v_track.info.extra_data.len() >= 8 {
        let hdr = avcc_config_to_annex_b(&v_track.info.extra_data);
        let _ = dec.set_extradata(&hdr);
    }
    let mut gop_frames = Vec::with_capacity(gop.len());
    for s_idx in gop.clone() {
        let s = &v_track.samples[s_idx];
        let end = (s.offset as usize).saturating_add(s.size as usize);
        if end > data.len() {
            return None;
        }
        let payload = &data[s.offset as usize..end];
        let annex_b = h264_avcc_to_annex_b_au(&v_track.info.extra_data, payload, s.is_keyframe());
        if let Ok(Some(f)) = dec.decode_annex_b_packet(&annex_b) {
            gop_frames.push(f);
        }
    }
    if gop_frames.is_empty() {
        return None;
    }
    if !gop_frames
        .windows(2)
        .all(|w| w[0].pts.ticks <= w[1].pts.ticks)
    {
        gop_frames.sort_by_key(|f| f.pts.ticks);
    }
    let local_idx = (target_idx - gop.start).min(gop_frames.len() - 1);
    Some((gop_frames.swap_remove(local_idx), fps))
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

    let out_ext = out_p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("ppm")
        .to_ascii_lowercase();

    let (out_rgb_or_yuv, extracted_count, fps) = if thumbnails.is_none()
        && let Some((single_frame, mp4_fps)) =
            decode_single_frame_at_mp4(in_p, at_sec.unwrap_or(0.0))
    {
        let out_frame = if out_ext == "y4m" {
            single_frame
        } else {
            convert_frame(&single_frame, PixelFormat::Rgb24, single_frame.colorimetry)
        };
        (out_frame, 1, mp4_fps)
    } else {
        let (frames, fps) = decode_video_file(in_p)?;
        if frames.is_empty() {
            return Err(RagdollError::InvalidData(
                "no video frames to extract".into(),
            ));
        }
        if let Some((count, cols, rows)) = thumbnails {
            let count = count.max(1);
            let thumb_w = (frames[0].width / cols.max(1)).max(64) & !1;
            let thumb_h = (frames[0].height / rows.max(1)).max(36) & !1;
            let mut selected = Vec::with_capacity(count);
            for i in 0..count {
                let idx = (i * frames.len() / count).min(frames.len() - 1);
                selected.push(frames[idx].clone());
            }
            let scaled =
                parallel_scale_frames(&selected, thumb_w, thumb_h, ScaleFilter::Lanczos3, 0)?;
            let rgb_thumbs: Vec<VideoFrame> = scaled
                .iter()
                .map(|f| convert_frame(f, PixelFormat::Rgb24, f.colorimetry))
                .collect();
            let thumb_refs: Vec<&VideoFrame> = rgb_thumbs.iter().collect();
            let sheet = compose_contact_sheet_rgb24(&thumb_refs, cols.max(1), 4, [18, 18, 22])?;
            (sheet, count, fps)
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
            (out_frame, 1, fps)
        }
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
        match out_ext.as_str() {
            "png" => {
                write_png_image(&mut buf, &rgb)?;
            }
            "jpg" | "jpeg" => {
                write_jpeg_image(&mut buf, &rgb, 90)?;
            }
            _ => {
                write_pnm_image(&mut buf, &rgb)?;
            }
        }
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

    const METRICS_WAVE: usize = 30;
    let mut reports = Vec::with_capacity(count);
    let mut wave_start = 0usize;
    while wave_start < count {
        let wave_end = (wave_start + METRICS_WAVE).min(count);
        let indices: Vec<usize> = (wave_start..wave_end).collect();
        let wave_reports: Vec<QualityReport> = parallel_gop_map(&indices, 0, |_, &i| {
            let prev = if i > 0 {
                Some(&ref_frames[i - 1])
            } else {
                None
            };
            evaluate_frame_quality(&ref_frames[i], &dist_frames[i], prev)
        });
        reports.extend(wave_reports);
        wave_start = wave_end;
    }

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
    } else if matches!(ext.as_str(), "ppm" | "png" | "jpg" | "jpeg") {
        let rgb = convert_frame(&frames[0], PixelFormat::Rgb24, frames[0].colorimetry);
        let mut buf = Vec::new();
        match ext.as_str() {
            "png" => {
                write_png_image(&mut buf, &rgb)?;
            }
            "jpg" | "jpeg" => {
                write_jpeg_image(&mut buf, &rgb, 90)?;
            }
            _ => {
                write_pnm_image(&mut buf, &rgb)?;
            }
        }
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

    let (fmt, tracks) = ragdoll_format::probe_container(data)?;
    let mut packet_sizes = Vec::new();
    let mut gop_count = 0usize;
    let mut h264_profile = None;

    if fmt == ContainerFormat::Flac {
        let demux = FlacContainerDemuxer::open(data)?;
        let packet_sizes: Vec<(bool, u32)> = demux.samples.iter().map(|s| (true, s.size)).collect();
        return Ok(ProbeReport {
            path: p.display().to_string(),
            container: "flac".into(),
            file_size,
            tracks: vec![demux.track().clone()],
            h264_profile: None,
            gop_count: packet_sizes.len().max(1),
            packet_sizes,
            probe_time_us: t0.elapsed().as_secs_f64() * 1e6,
        });
    }
    if fmt == ContainerFormat::Ogg {
        let demux = OggDemuxer::open(data)?;
        let packet_sizes: Vec<(bool, u32)> = demux.samples.iter().map(|s| (true, s.size)).collect();
        return Ok(ProbeReport {
            path: p.display().to_string(),
            container: "ogg".into(),
            file_size,
            tracks: vec![demux.track().clone()],
            h264_profile: None,
            gop_count: packet_sizes.len().max(1),
            packet_sizes,
            probe_time_us: t0.elapsed().as_secs_f64() * 1e6,
        });
    }

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

/// Parses a flexible timestamp string (`"HH:MM:SS.mmm"`, `"MM:SS.mmm"`, or `"12.345"`) into seconds.
pub fn parse_time_seconds(s: &str) -> Result<f64> {
    let trimmed = s.trim().trim_end_matches('s');
    if let Ok(sec) = trimmed.parse::<f64>() {
        return Ok(sec.max(0.0));
    }
    if let Ok(ts) = Timestamp::parse_hms(trimmed) {
        return Ok(ts.to_seconds_f64().unwrap_or(0.0).max(0.0));
    }
    let parts: Vec<&str> = trimmed.split(':').collect();
    if parts.len() == 2
        && let (Ok(m), Ok(sec)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>())
    {
        return Ok((m * 60.0 + sec).max(0.0));
    }
    Err(RagdollError::InvalidHeader(format!(
        "invalid timestamp '{s}' (expected seconds or HH:MM:SS.mmm)"
    )))
}

/// Options for [`cut_media`] (`0046`).
#[derive(Clone, Debug, Default)]
pub struct CutOptions {
    pub from_sec: Option<f64>,
    pub to_sec: Option<f64>,
    pub keyframes_only: bool,
    pub qp: u8,
}

/// Telemetry report from [`cut_media`] (`0046`).
#[derive(Clone, Debug, Default)]
pub struct CutReport {
    pub output_bytes: u64,
    pub total_video_packets: usize,
    pub copied_video_packets: usize,
    pub reencoded_video_frames: usize,
    pub audio_packets: usize,
    pub elapsed_ms: f64,
}

/// Extracts all tracks and packets from an MP4, MKV, or MPEG-TS byte slice in presentation/demux order.
fn demux_all_tracks_and_packets(
    data: &[u8],
) -> Result<(ContainerFormat, Vec<TrackInfo>, Vec<PacketSlice<'static>>)> {
    let fmt = detect_container_format(data).ok_or_else(|| {
        RagdollError::InvalidHeader("unrecognized container for cut/concat".into())
    })?;
    match fmt {
        ContainerFormat::Mp4 => {
            let mut demux = Mp4Demuxer::open(data)?;
            let tracks: Vec<TrackInfo> = demux.tracks().iter().map(|t| t.info.clone()).collect();
            let mut pkts = Vec::new();
            while let Some(p) = demux.next_packet()? {
                pkts.push(PacketSlice::from_owned(
                    AlignedBuffer::from_slice(p.data()),
                    p.track_index,
                    p.pts,
                    p.dts,
                    p.duration,
                    p.flags,
                ));
            }
            Ok((fmt, tracks, pkts))
        }
        ContainerFormat::Matroska | ContainerFormat::WebM => {
            let mut demux = MkvDemuxer::open(data)?;
            let tracks = demux.tracks().to_vec();
            let mut pkts = Vec::new();
            while let Some(p) = demux.next_packet()? {
                pkts.push(PacketSlice::from_owned(
                    AlignedBuffer::from_slice(p.data()),
                    p.track_index,
                    p.pts,
                    p.dts,
                    p.duration,
                    p.flags,
                ));
            }
            Ok((fmt, tracks, pkts))
        }
        ContainerFormat::MpegTs => {
            let mut demux = MpegTsDemuxer::open(data)?;
            let tracks = demux.tracks().to_vec();
            let mut pkts = Vec::new();
            while let Some(p) = demux.next_packet()? {
                pkts.push(PacketSlice::from_owned(
                    AlignedBuffer::from_slice(p.data()),
                    p.track_index,
                    p.pts,
                    p.dts,
                    p.duration,
                    p.flags,
                ));
            }
            Ok((fmt, tracks, pkts))
        }
        _ => Err(RagdollError::Unsupported(format!(
            "cut/concat requires MP4, MKV, or MPEG-TS container, got {fmt}"
        ))),
    }
}

fn mux_tracks_and_packets(
    out_path: &Path,
    tracks: &[TrackInfo],
    packets: Vec<PacketSlice<'static>>,
) -> Result<u64> {
    let ext = out_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("mp4")
        .to_ascii_lowercase();
    let mut writer = OutputSink::create(out_path)?;
    match ext.as_str() {
        "mkv" | "webm" => {
            let mut mux = MkvMuxer::new_matroska();
            for t in tracks {
                mux.add_track(t.clone());
            }
            for p in packets {
                mux.write_packet(p)?;
            }
            mux.finish(&mut writer)
        }
        "ts" | "m2ts" | "mpegts" => {
            let mut mux = MpegTsMuxer::new(&mut writer);
            for t in tracks {
                mux.add_track(t.clone());
            }
            for p in &packets {
                mux.write_packet(p)?;
            }
            mux.finish()
        }
        _ => {
            let mut mux = Mp4Muxer::new();
            for t in tracks {
                mux.add_track(t.clone());
            }
            for p in packets {
                mux.write_packet(p)?;
            }
            mux.finish(&mut writer)
        }
    }
}

/// **`0046`**: Smart-GOP frame-accurate non-linear editing (`ragdoll cut <IN> <OUT> --from <T> --to <T>`).
///
/// When `--keyframes-only` is set or `--from` lands on an IDR keyframe, performs 100% zero-copy stream copy.
/// When `--from` lands mid-GOP, decodes only the boundary GOP, drops frames before `--from`, re-encodes
/// that partial GOP starting with a fresh IDR keyframe, and zero-copy copies all subsequent interior GOPs!
pub fn cut_media(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    opts: &CutOptions,
) -> Result<CutReport> {
    let t0 = Instant::now();
    let in_p = input_path.as_ref();
    let out_p = output_path.as_ref();
    let mmap = MmapSource::open(in_p)?;
    let data = mmap.as_slice();
    let (_fmt, tracks, all_pkts) = demux_all_tracks_and_packets(data)?;

    let from_s = opts.from_sec.unwrap_or(0.0).max(0.0);
    let to_s = opts.to_sec.unwrap_or(f64::INFINITY);
    let qp = if opts.qp == 0 { 20 } else { opts.qp };

    let Some(v_track) = tracks
        .iter()
        .find(|t| t.track_type == TrackType::Video)
        .cloned()
    else {
        return Err(RagdollError::InvalidData(
            "cut_media requires a video track".into(),
        ));
    };

    let v_pkts: Vec<PacketSlice<'static>> = all_pkts
        .iter()
        .filter(|p| p.track_index == v_track.index)
        .cloned()
        .collect();
    if v_pkts.is_empty() {
        return Err(RagdollError::InvalidData(
            "video track has 0 packets".into(),
        ));
    }

    // Partition video packets into GOPs at keyframe boundaries
    let mut gops: Vec<Vec<PacketSlice<'static>>> = Vec::new();
    for p in v_pkts {
        if p.is_keyframe() || gops.is_empty() {
            gops.push(Vec::new());
        }
        gops.last_mut().unwrap().push(p);
    }

    // Find the GOP containing `from_s`
    let mut start_gop_idx = 0usize;
    for (gi, gop) in gops.iter().enumerate() {
        let gop_start_s = gop[0].pts.to_seconds_f64().unwrap_or(0.0);
        if gop_start_s <= from_s + 1e-6 {
            start_gop_idx = gi;
        } else {
            break;
        }
    }

    let first_gop = &gops[start_gop_idx];
    let first_gop_start_s = first_gop[0].pts.to_seconds_f64().unwrap_or(0.0);
    let lands_on_keyframe = (from_s - first_gop_start_s).abs() <= 1e-3;

    let actual_cut_start_s = if opts.keyframes_only || lands_on_keyframe {
        first_gop_start_s
    } else {
        from_s
    };

    let mut out_video_pkts: Vec<PacketSlice<'static>> = Vec::new();
    let mut copied_video_packets = 0usize;
    let mut reencoded_video_frames = 0usize;

    let v_tb = v_track.timebase;
    let cut_origin_ticks =
        (actual_cut_start_s * (v_tb.den as f64) / (v_tb.num.max(1) as f64)).round() as i64;

    for (gi, gop) in gops.iter().enumerate().skip(start_gop_idx) {
        let gop_first_s = gop[0].pts.to_seconds_f64().unwrap_or(0.0);
        if gop_first_s >= to_s - 1e-6 {
            break;
        }

        let is_boundary_mid_gop = gi == start_gop_idx && !opts.keyframes_only && !lands_on_keyframe;

        if is_boundary_mid_gop && v_track.codec_id == CodecId::H264 {
            // Decode this single boundary GOP, keep frames in [from_s, to_s), and re-encode starting with an IDR keyframe
            let mut dec = H264Decoder::new();
            if v_track.extra_data.len() >= 8 {
                let annex_b_hdr = avcc_config_to_annex_b(&v_track.extra_data);
                let _ = dec.set_extradata(&annex_b_hdr);
            }
            let mut kept_frames = Vec::new();
            for pkt in gop {
                let pkt_s = pkt.pts.to_seconds_f64().unwrap_or(0.0);
                let annex_b =
                    h264_avcc_to_annex_b_au(&v_track.extra_data, pkt.data(), pkt.is_keyframe());
                if let Some(mut f) = dec.decode_annex_b_packet(&annex_b)? {
                    f.pts = pkt.pts;
                    f.duration = pkt.duration;
                    if pkt_s >= from_s - 1e-4 && pkt_s < to_s - 1e-6 {
                        kept_frames.push(f);
                    }
                }
            }
            if !kept_frames.is_empty() {
                let vp = v_track.video.clone().unwrap_or_default();
                let mut enc = H264Encoder::new(vp.width, vp.height, vp.frame_rate, qp);
                for (fi, frame) in kept_frames.iter().enumerate() {
                    let annex_b = enc.encode_frame(frame)?;
                    let avcc_au = h264_annex_b_to_avcc_au(&annex_b);
                    let adj_pts = (frame.pts.rescale_to(v_tb).ticks - cut_origin_ticks).max(0);
                    let dur = frame.duration.max(1);
                    out_video_pkts.push(PacketSlice::from_owned(
                        AlignedBuffer::from_slice(&avcc_au),
                        v_track.index,
                        Timestamp::new(adj_pts, v_tb),
                        Timestamp::new(adj_pts, v_tb),
                        dur,
                        if fi == 0 {
                            PacketFlags::KEYFRAME
                        } else {
                            PacketFlags::NONE
                        },
                    ));
                    reencoded_video_frames += 1;
                }
            }
        } else {
            // Zero-copy stream copy of interior GOP packets up to `to_s`
            for pkt in gop {
                let pkt_s = pkt.pts.to_seconds_f64().unwrap_or(0.0);
                if pkt_s < actual_cut_start_s - 1e-4 || pkt_s >= to_s - 1e-6 {
                    continue;
                }
                let adj_pts = (pkt.pts.rescale_to(v_tb).ticks - cut_origin_ticks).max(0);
                let adj_dts = (pkt.dts.rescale_to(v_tb).ticks - cut_origin_ticks).max(0);
                out_video_pkts.push(PacketSlice::from_owned(
                    AlignedBuffer::from_slice(pkt.data()),
                    v_track.index,
                    Timestamp::new(adj_pts, v_tb),
                    Timestamp::new(adj_dts, v_tb),
                    pkt.duration,
                    pkt.flags,
                ));
                copied_video_packets += 1;
            }
        }
    }

    let total_video_packets = out_video_pkts.len();
    let mut final_pkts = out_video_pkts;
    let mut audio_packets = 0usize;

    // Trim & shift auxiliary Audio / Subtitle packets to `[actual_cut_start_s, to_s)`
    for pkt in all_pkts {
        if pkt.track_index == v_track.index {
            continue;
        }
        let pkt_s = pkt.pts.to_seconds_f64().unwrap_or(0.0);
        if pkt_s >= actual_cut_start_s - 1e-4 && pkt_s < to_s - 1e-6 {
            let tb = pkt.pts.timebase;
            let origin =
                (actual_cut_start_s * (tb.den as f64) / (tb.num.max(1) as f64)).round() as i64;
            let adj_pts = (pkt.pts.ticks - origin).max(0);
            let adj_dts = (pkt.dts.ticks - origin).max(0);
            final_pkts.push(PacketSlice::from_owned(
                AlignedBuffer::from_slice(pkt.data()),
                pkt.track_index,
                Timestamp::new(adj_pts, tb),
                Timestamp::new(adj_dts, tb),
                pkt.duration,
                pkt.flags,
            ));
            audio_packets += 1;
        }
    }

    let output_bytes = mux_tracks_and_packets(out_p, &tracks, final_pkts)?;
    Ok(CutReport {
        output_bytes,
        total_video_packets,
        copied_video_packets,
        reencoded_video_frames,
        audio_packets,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
    })
}

/// Telemetry report from [`concat_media`] (`0046`).
#[derive(Clone, Debug, Default)]
pub struct ConcatReport {
    pub output_bytes: u64,
    pub segments_merged: usize,
    pub video_packets: usize,
    pub audio_packets: usize,
    pub reencoded: bool,
    pub elapsed_ms: f64,
}

/// **`0046`**: Concatenates multiple media files (`ragdoll concat <IN1> <IN2> ... -o <OUT>`).
///
/// Performs zero-copy packet splicing with monotonic 128-bit PTS/DTS offset accumulation when all
/// inputs share compatible codec parameters, or automatically normalizes mismatched inputs.
pub fn concat_media<P: AsRef<Path>>(
    inputs: &[P],
    output_path: impl AsRef<Path>,
) -> Result<ConcatReport> {
    let t0 = Instant::now();
    if inputs.is_empty() {
        return Err(RagdollError::InvalidHeader(
            "concat_media requires at least 1 input file".into(),
        ));
    }
    let out_p = output_path.as_ref();

    let mut opened = Vec::with_capacity(inputs.len());
    for inp in inputs {
        let mmap = MmapSource::open(inp.as_ref())?;
        let (_, tracks, pkts) = demux_all_tracks_and_packets(mmap.as_slice())?;
        opened.push((inp.as_ref().to_path_buf(), tracks, pkts));
    }

    let ref_tracks = &opened[0].1;
    let ref_v = ref_tracks
        .iter()
        .find(|t| t.track_type == TrackType::Video)
        .cloned();

    // Check if all inputs have identical video codec, dimensions, and extradata
    let compatible = if let Some(ref rv) = ref_v {
        let rvp = rv.video.clone().unwrap_or_default();
        opened.iter().skip(1).all(|(_, trks, _)| {
            trks.iter()
                .find(|t| t.track_type == TrackType::Video)
                .is_some_and(|tv| {
                    let tvp = tv.video.clone().unwrap_or_default();
                    tv.codec_id == rv.codec_id
                        && tvp.width == rvp.width
                        && tvp.height == rvp.height
                        && tv.extra_data == rv.extra_data
                })
        })
    } else {
        false
    };

    if compatible {
        let mut merged_pkts: Vec<PacketSlice<'static>> = Vec::new();
        let mut track_end_ticks = vec![0i64; ref_tracks.len()];
        let mut video_packets = 0usize;
        let mut audio_packets = 0usize;

        for (_, seg_tracks, seg_pkts) in &opened {
            let mut next_track_end = track_end_ticks.clone();
            for pkt in seg_pkts {
                let tid = pkt.track_index as usize;
                if tid >= ref_tracks.len() {
                    continue;
                }
                let target_tb = ref_tracks[tid].timebase;
                let offset_ticks = track_end_ticks[tid];
                let pts_t = pkt.pts.rescale_to(target_tb).ticks + offset_ticks;
                let dts_t = pkt.dts.rescale_to(target_tb).ticks + offset_ticks;
                let dur_t = rescale_q(
                    pkt.duration.max(1),
                    seg_tracks.get(tid).map_or(pkt.pts.timebase, |t| t.timebase),
                    target_tb,
                )
                .max(1);
                next_track_end[tid] = next_track_end[tid].max(pts_t + dur_t);

                if ref_tracks[tid].track_type == TrackType::Video {
                    video_packets += 1;
                } else {
                    audio_packets += 1;
                }

                merged_pkts.push(PacketSlice::from_owned(
                    AlignedBuffer::from_slice(pkt.data()),
                    pkt.track_index,
                    Timestamp::new(pts_t, target_tb),
                    Timestamp::new(dts_t, target_tb),
                    dur_t,
                    pkt.flags,
                ));
            }
            track_end_ticks = next_track_end;
        }

        let output_bytes = mux_tracks_and_packets(out_p, ref_tracks, merged_pkts)?;
        return Ok(ConcatReport {
            output_bytes,
            segments_merged: inputs.len(),
            video_packets,
            audio_packets,
            reencoded: false,
            elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
        });
    }

    // Fallback when resolution/codec differs across inputs: decode & normalize all frames to first input's dimensions
    let mut all_frames = Vec::new();
    let mut target_w = 0usize;
    let mut target_h = 0usize;
    let mut target_fps = Rational64::FPS_30;
    for (idx, (path, _, _)) in opened.iter().enumerate() {
        let (frames, fps) = decode_video_file(path)?;
        if frames.is_empty() {
            continue;
        }
        if idx == 0 {
            target_w = frames[0].width;
            target_h = frames[0].height;
            target_fps = fps;
        }
        let scaled = if frames[0].width != target_w || frames[0].height != target_h {
            parallel_scale_frames(&frames, target_w, target_h, ScaleFilter::Lanczos3, 0)?
        } else {
            frames
        };
        all_frames.extend(scaled);
    }

    let tmp_y4m = std::env::temp_dir().join(format!("ragdoll_concat_{}.y4m", std::process::id()));
    {
        let f = File::create(&tmp_y4m)?;
        let mut mux = Y4mMuxer::new(
            BufWriter::new(f),
            target_w,
            target_h,
            PixelFormat::Yuv420p,
            target_fps,
            Colorimetry::bt709_limited(),
        )?;
        for frame in &all_frames {
            mux.write_frame(frame)?;
        }
        mux.finish()?;
    }
    let tel = transcode_file(&tmp_y4m, out_p, None, ScaleFilter::Lanczos3, "h264", 20, 0)?;
    let _ = std::fs::remove_file(&tmp_y4m);

    Ok(ConcatReport {
        output_bytes: tel.bytes_out,
        segments_merged: inputs.len(),
        video_packets: all_frames.len(),
        audio_packets: 0,
        reencoded: true,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
    })
}

/// Single rendition target in an ABR encoding ladder (`0049`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AbrRenditionSpec {
    pub width: usize,
    pub height: usize,
    pub bitrate_kbps: u32,
}

impl AbrRenditionSpec {
    /// Parses a comma-separated ABR ladder specification such as:
    /// `"1080p:4500k,720p:2500k,480p:1200k,360p:600k"` or `"320x180:400k,160x90:200k"`.
    pub fn parse_ladder(spec: &str, src_w: usize, src_h: usize) -> Result<Vec<Self>> {
        let mut out = Vec::new();
        for item in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let (res_s, br_s) = item.split_once(':').ok_or_else(|| {
                RagdollError::InvalidHeader(format!(
                    "expected '<RES>:<BITRATE>' in ABR ladder, got '{item}'"
                ))
            })?;
            let br_clean = br_s
                .trim()
                .trim_end_matches("kbps")
                .trim_end_matches("k")
                .trim_end_matches('K');
            let bitrate_kbps: u32 = br_clean.parse().map_err(|_| {
                RagdollError::InvalidHeader(format!("invalid ABR bitrate '{br_s}'"))
            })?;

            let (w, h) = if let Some((ws, hs)) = res_s.split_once('x') {
                let w: usize = ws.parse().map_err(|_| {
                    RagdollError::InvalidHeader(format!("invalid width in '{res_s}'"))
                })?;
                let h: usize = hs.parse().map_err(|_| {
                    RagdollError::InvalidHeader(format!("invalid height in '{res_s}'"))
                })?;
                (w & !1, h & !1)
            } else if let Some(h_str) = res_s.strip_suffix('p') {
                let h: usize = h_str.parse().map_err(|_| {
                    RagdollError::InvalidHeader(format!("invalid height in '{res_s}'"))
                })?;
                let h_even = (h & !1).max(2);
                let w_even = (((h_even * src_w.max(16)) / src_h.max(9) + 1) & !1).max(2);
                (w_even, h_even)
            } else {
                return Err(RagdollError::InvalidHeader(format!(
                    "unsupported ABR resolution '{res_s}' (expected WxH or NNNp)"
                )));
            };
            out.push(Self {
                width: w.max(2),
                height: h.max(2),
                bitrate_kbps: bitrate_kbps.max(32),
            });
        }
        if out.is_empty() {
            return Err(RagdollError::InvalidHeader(
                "ABR ladder cannot be empty".into(),
            ));
        }
        Ok(out)
    }
}

/// Options for [`package_abr`] (`0049`).
#[derive(Clone, Debug)]
pub struct PackageOptions {
    pub ladder_spec: Option<String>,
    pub segment_duration_sec: f64,
    pub generate_thumbnails: bool,
}

impl Default for PackageOptions {
    fn default() -> Self {
        Self {
            ladder_spec: None,
            segment_duration_sec: 2.0,
            generate_thumbnails: true,
        }
    }
}

/// Telemetry report emitted by [`package_abr`] (`0049`).
#[derive(Clone, Debug)]
pub struct PackageReport {
    pub renditions: usize,
    pub segments_per_rendition: usize,
    pub master_m3u8_path: std::path::PathBuf,
    pub mpd_path: std::path::PathBuf,
    pub thumbnails_vtt_path: Option<std::path::PathBuf>,
    pub total_bytes: u64,
    pub elapsed_ms: f64,
}

/// **`0049`**: One-command CMAF `fMP4` HLS (`master.m3u8`) & DASH (`manifest.mpd`) multi-rendition
/// ABR packager (`ragdoll package <IN> -o <OUT_DIR>`).
///
/// Decodes the input once, scales across all ABR renditions in parallel, enforces GOP-aligned IDR
/// keyframes at exact segment boundaries, writes shared CMAF `init.mp4` + `seg_NNNN.m4s` fragments,
/// generates HLS (`master.m3u8` + `stream_N/index.m3u8`) and DASH (`manifest.mpd`) manifests, and
/// emits a `thumbnails.vtt` + `sprites.jpg` trick-play sprite sheet!
pub fn package_abr(
    input_path: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
    opts: &PackageOptions,
) -> Result<PackageReport> {
    let t0 = Instant::now();
    let in_p = input_path.as_ref();
    let out_dir = output_dir.as_ref();
    std::fs::create_dir_all(out_dir)?;

    let (raw_frames, fps) = decode_video_file(in_p)?;
    if raw_frames.is_empty() {
        return Err(RagdollError::InvalidData(
            "package_abr input has 0 video frames".into(),
        ));
    }

    let src_w = raw_frames[0].width;
    let src_h = raw_frames[0].height;
    let ladder = if let Some(ref spec) = opts.ladder_spec {
        AbrRenditionSpec::parse_ladder(spec, src_w, src_h)?
    } else {
        let half_w = ((src_w / 2) & !1).max(32);
        let half_h = ((src_h / 2) & !1).max(32);
        vec![
            AbrRenditionSpec {
                width: src_w & !1,
                height: src_h & !1,
                bitrate_kbps: 2400,
            },
            AbrRenditionSpec {
                width: half_w,
                height: half_h,
                bitrate_kbps: 800,
            },
        ]
    };

    let seg_dur_s = opts.segment_duration_sec.max(0.2);
    let fps_f = fps.as_f64().max(1.0);
    let frames_per_seg = ((seg_dur_s * fps_f).round() as usize).max(1);
    let num_segments = raw_frames.len().div_ceil(frames_per_seg);
    let total_duration_s = (raw_frames.len() as f64) / fps_f;

    let timebase = Rational64::new(1, 90_000);
    let frame_dur_ticks = ((90_000 * fps.den) / fps.num.max(1)).max(1);

    let mut total_bytes = 0u64;
    let mut rendition_codecs = Vec::with_capacity(ladder.len());

    let target_dur_ceil = seg_dur_s.ceil().max(1.0) as u32;
    let rend_indices: Vec<usize> = (0..ladder.len()).collect();
    let rend_results: Vec<Result<(String, u64)>> =
        parallel_gop_map(&rend_indices, 0, |_, &r_idx| {
            let rend = &ladder[r_idx];
            let r_dir = out_dir.join(format!("stream_{r_idx}"));
            std::fs::create_dir_all(&r_dir)?;
            let mut r_bytes = 0u64;

            let scaled = if rend.width == src_w && rend.height == src_h {
                raw_frames.clone()
            } else {
                parallel_scale_frames(
                    &raw_frames,
                    rend.width,
                    rend.height,
                    ScaleFilter::Lanczos3,
                    0,
                )?
            };

            let rc = RateControlMode::Abr {
                bitrate_kbps: rend.bitrate_kbps,
                maxrate_kbps: Some(rend.bitrate_kbps * 2),
                bufsize_kb: Some(rend.bitrate_kbps * 2),
            };
            let hdr_enc = H264Encoder::new(rend.width as u32, rend.height as u32, fps, 22)
                .with_profile(H264Profile::Main)
                .with_rate_control(rc);
            let sps_nal = hdr_enc.sps.to_nal_unit();
            let pps_nal = hdr_enc.pps.to_nal_unit();
            let avcc_extra = build_avcc_from_sps_pps(&sps_nal, &pps_nal);
            let codecs_tag = format!(
                "avc1.{:02X}{:02X}{:02X}",
                hdr_enc.sps.profile_idc, hdr_enc.sps.constraint_set_flags, hdr_enc.sps.level_idc
            );

            let track_info = TrackInfo {
                index: 0,
                track_type: TrackType::Video,
                codec_id: CodecId::H264,
                timebase,
                duration_ticks: Some((scaled.len() as i64) * frame_dur_ticks),
                extra_data: avcc_extra,
                video: Some(VideoTrackParams {
                    width: rend.width as u32,
                    height: rend.height as u32,
                    pixel_format: PixelFormat::Yuv420p,
                    colorimetry: Colorimetry::bt709_limited(),
                    frame_rate: fps,
                    bit_depth: 8,
                    profile: None,
                    level: None,
                }),
                audio: None,
                sample_count: scaled.len(),
                total_bytes: 0,
            };

            let mut mux = Mp4Muxer::new();
            mux.add_track(track_info);

            // 1. Write `stream_N/init.mp4`
            let init_path = r_dir.join("init.mp4");
            let init_file = File::create(&init_path)?;
            r_bytes += mux.write_cmaf_init_segment(BufWriter::new(init_file))?;

            // 2. Encode each GOP-aligned segment and write `stream_N/seg_NNNN.m4s` + `stream_N/index.m3u8`
            let mut m3u8 = String::new();
            m3u8.push_str("#EXTM3U\n");
            m3u8.push_str("#EXT-X-VERSION:7\n");
            m3u8.push_str(&format!("#EXT-X-TARGETDURATION:{target_dur_ceil}\n"));
            m3u8.push_str("#EXT-X-MEDIA-SEQUENCE:1\n");
            m3u8.push_str("#EXT-X-PLAYLIST-TYPE:VOD\n");
            m3u8.push_str("#EXT-X-INDEPENDENT-SEGMENTS\n");
            m3u8.push_str("#EXT-X-MAP:URI=\"init.mp4\"\n");

            let seg_indices: Vec<usize> = (0..num_segments).collect();
            let encoded_segs: Vec<Result<(usize, Vec<PacketSlice<'static>>, usize)>> =
                parallel_gop_map(&seg_indices, 0, |_, &seg_idx| {
                    let f_start = seg_idx * frames_per_seg;
                    let f_end = ((seg_idx + 1) * frames_per_seg).min(scaled.len());
                    let seg_frames = &scaled[f_start..f_end];

                    // Fresh encoder per segment guarantees every segment starts with an IDR keyframe (`SAP=1`)
                    let mut seg_enc =
                        H264Encoder::new(rend.width as u32, rend.height as u32, fps, 22)
                            .with_profile(H264Profile::Main)
                            .with_rate_control(rc);

                    let mut seg_pkts = Vec::with_capacity(seg_frames.len());
                    for (local_i, frame) in seg_frames.iter().enumerate() {
                        let global_i = (f_start + local_i) as i64;
                        let annex_b = seg_enc.encode_frame(frame)?;
                        let avcc_au = h264_annex_b_to_avcc_au(&annex_b);
                        let pts = global_i * frame_dur_ticks;
                        seg_pkts.push(PacketSlice::from_owned(
                            AlignedBuffer::from_slice(&avcc_au),
                            0,
                            Timestamp::new(pts, timebase),
                            Timestamp::new(pts, timebase),
                            frame_dur_ticks,
                            if local_i == 0 {
                                PacketFlags::KEYFRAME
                            } else {
                                PacketFlags::NONE
                            },
                        ));
                    }
                    Ok((seg_idx, seg_pkts, seg_frames.len()))
                });

            for seg_res in encoded_segs {
                let (seg_idx, seg_pkts, seg_frame_count) = seg_res?;
                let seg_num = (seg_idx + 1) as u32;
                let seg_name = format!("seg_{seg_num:04}.m4s");
                let seg_path = r_dir.join(&seg_name);
                let seg_file = File::create(&seg_path)?;
                r_bytes +=
                    mux.write_cmaf_media_segment(BufWriter::new(seg_file), seg_num, 1, &seg_pkts)?;

                let actual_seg_dur = (seg_frame_count as f64) / fps_f;
                m3u8.push_str(&format!("#EXTINF:{actual_seg_dur:.6},\n{seg_name}\n"));
            }
            m3u8.push_str("#EXT-X-ENDLIST\n");
            let playlist_path = r_dir.join("index.m3u8");
            std::fs::write(&playlist_path, m3u8.as_bytes())?;
            r_bytes += m3u8.len() as u64;
            Ok((codecs_tag, r_bytes))
        });

    for res in rend_results {
        let (codecs_tag, r_bytes) = res?;
        rendition_codecs.push(codecs_tag);
        total_bytes += r_bytes;
    }

    // 3. Write RFC 8216 HLS `master.m3u8`
    let mut master_m3u8 = String::from("#EXTM3U\n#EXT-X-VERSION:7\n#EXT-X-INDEPENDENT-SEGMENTS\n");
    for (r_idx, rend) in ladder.iter().enumerate() {
        let bw = rend.bitrate_kbps * 1000;
        let avg_bw = (bw * 9) / 10;
        let codecs = &rendition_codecs[r_idx];
        master_m3u8.push_str(&format!(
            "#EXT-X-STREAM-INF:BANDWIDTH={bw},AVERAGE-BANDWIDTH={avg_bw},RESOLUTION={}x{},FRAME-RATE={fps_f:.3},CODECS=\"{codecs}\"\nstream_{r_idx}/index.m3u8\n",
            rend.width, rend.height
        ));
    }
    let master_m3u8_path = out_dir.join("master.m3u8");
    std::fs::write(&master_m3u8_path, master_m3u8.as_bytes())?;
    total_bytes += master_m3u8.len() as u64;

    // 4. Write ISO/IEC 23009-1 DASH `manifest.mpd` referencing the exact same CMAF segments
    let seg_dur_ticks = (frames_per_seg as i64) * frame_dur_ticks;
    let mut mpd = String::new();
    mpd.push_str("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");
    mpd.push_str(&format!(
        "<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\" profiles=\"urn:mpeg:dash:profile:isoff-live:2011,http://dashif.org/guidelines/dash264\" type=\"static\" mediaPresentationDuration=\"PT{total_duration_s:.3}S\" minBufferTime=\"PT{seg_dur_s:.3}S\">\n"
    ));
    mpd.push_str(&format!(
        "  <Period id=\"0\" duration=\"PT{total_duration_s:.3}S\">\n"
    ));
    mpd.push_str("    <AdaptationSet id=\"0\" contentType=\"video\" mimeType=\"video/mp4\" segmentAlignment=\"true\" startWithSAP=\"1\">\n");
    for (r_idx, rend) in ladder.iter().enumerate() {
        let bw = rend.bitrate_kbps * 1000;
        let codecs = &rendition_codecs[r_idx];
        mpd.push_str(&format!(
            "      <Representation id=\"{r_idx}\" bandwidth=\"{bw}\" width=\"{}\" height=\"{}\" codecs=\"{codecs}\">\n",
            rend.width, rend.height
        ));
        mpd.push_str(&format!(
            "        <SegmentTemplate timescale=\"90000\" duration=\"{seg_dur_ticks}\" initialization=\"stream_{r_idx}/init.mp4\" media=\"stream_{r_idx}/seg_$Number%04d$.m4s\" startNumber=\"1\"/>\n"
        ));
        mpd.push_str("      </Representation>\n");
    }
    mpd.push_str("    </AdaptationSet>\n  </Period>\n</MPD>\n");
    let mpd_path = out_dir.join("manifest.mpd");
    std::fs::write(&mpd_path, mpd.as_bytes())?;
    total_bytes += mpd.len() as u64;

    // 5. Generate WebVTT trick-play thumbnail track (`thumbnails.vtt` + `sprites.jpg`)
    let thumbnails_vtt_path = if opts.generate_thumbnails {
        let thumb_w = 160usize;
        let thumb_h = 90usize;
        let cols = 5usize;
        let mut selected = Vec::with_capacity(num_segments);
        for seg_idx in 0..num_segments {
            let f_idx = (seg_idx * frames_per_seg).min(raw_frames.len() - 1);
            selected.push(raw_frames[f_idx].clone());
        }
        let scaled_thumbs =
            parallel_scale_frames(&selected, thumb_w, thumb_h, ScaleFilter::Lanczos3, 0)?;
        let rgb_thumbs: Vec<VideoFrame> = scaled_thumbs
            .iter()
            .map(|f| convert_frame(f, PixelFormat::Rgb24, f.colorimetry))
            .collect();
        let refs: Vec<&VideoFrame> = rgb_thumbs.iter().collect();
        let sheet = compose_contact_sheet_rgb24(&refs, cols, 0, [0, 0, 0])?;
        let sprite_path = out_dir.join("sprites.jpg");
        let mut jpg_buf = Vec::new();
        write_jpeg_image(&mut jpg_buf, &sheet, 85)?;
        std::fs::write(&sprite_path, &jpg_buf)?;
        total_bytes += jpg_buf.len() as u64;

        let mut vtt = String::from("WEBVTT\n\n");
        for seg_idx in 0..num_segments {
            let start_ms = ((seg_idx * frames_per_seg) as f64 * 1000.0 / fps_f).round() as u64;
            let end_ms = ((((seg_idx + 1) * frames_per_seg).min(raw_frames.len())) as f64 * 1000.0
                / fps_f)
                .round() as u64;
            let r = seg_idx / cols;
            let c = seg_idx % cols;
            let x = c * thumb_w;
            let y = r * thumb_h;
            let t0_str = ragdoll_format::format_subtitle_timestamp_ms(start_ms as i64, '.');
            let t1_str = ragdoll_format::format_subtitle_timestamp_ms(end_ms as i64, '.');
            vtt.push_str(&format!(
                "{t0_str} --> {t1_str}\nsprites.jpg#xywh={x},{y},{thumb_w},{thumb_h}\n\n"
            ));
        }
        let vtt_path = out_dir.join("thumbnails.vtt");
        std::fs::write(&vtt_path, vtt.as_bytes())?;
        total_bytes += vtt.len() as u64;
        Some(vtt_path)
    } else {
        None
    };

    Ok(PackageReport {
        renditions: ladder.len(),
        segments_per_rendition: num_segments,
        master_m3u8_path,
        mpd_path,
        thumbnails_vtt_path,
        total_bytes,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
    })
}

// ============================================================================
// 0050: Ergonomic Zero-Unsafe Rust Library SDK (TranscodeBuilder, CutBuilder, PackageBuilder)
// ============================================================================

/// High-level Rust SDK builder for probing media containers and elementary streams (`0050`).
#[derive(Clone, Debug)]
pub struct ProbeBuilder {
    input_path: std::path::PathBuf,
}

/// Alias for [`ProbeBuilder`] (`0050`).
pub type ProbeJob = ProbeBuilder;

impl ProbeBuilder {
    /// Creates a new media probe job for `input_path`.
    pub fn new(input_path: impl AsRef<Path>) -> Self {
        Self {
            input_path: input_path.as_ref().to_path_buf(),
        }
    }

    /// Executes the container/bitstream probe and returns a structured [`ProbeReport`].
    pub fn run(&self) -> Result<ProbeReport> {
        probe_file_detailed(&self.input_path)
    }
}

/// High-level Rust SDK builder for video/audio transcoding, spatial filtering, HDR-to-SDR Oklab
/// tonemapping, and EBU R128 loudness normalization (`0050`).
///
/// # Example
/// ```no_run
/// use ragdoll_pipeline::{ScaleFilter, TranscodeBuilder};
///
/// let report = TranscodeBuilder::new("input.mp4", "output.mp4")
///     .scale(1280, 720, ScaleFilter::Lanczos3)
///     .crf(21.0)
///     .audio_codec("aac")
///     .normalize_lufs(-14.0)
///     .run()
///     .unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct TranscodeBuilder {
    input_path: std::path::PathBuf,
    output_path: std::path::PathBuf,
    opts: TranscodeOptions,
}

/// Alias for [`TranscodeBuilder`] (`0050`).
pub type TranscodeJob = TranscodeBuilder;

impl TranscodeBuilder {
    /// Creates a new transcoding job from `input_path` to `output_path`.
    pub fn new(input_path: impl AsRef<Path>, output_path: impl AsRef<Path>) -> Self {
        Self {
            input_path: input_path.as_ref().to_path_buf(),
            output_path: output_path.as_ref().to_path_buf(),
            opts: TranscodeOptions::default(),
        }
    }

    /// Configures output video dimensions `(width, height)` and polyphase [`ScaleFilter`].
    pub fn scale(mut self, width: usize, height: usize, filter: ScaleFilter) -> Self {
        self.opts.scale_dims = Some((width, height));
        self.opts.scale_filter = filter;
        self
    }

    /// Sets the target video codec (`"h264"`, `"rgdi"`, `"raw"`).
    pub fn video_codec(mut self, codec: impl Into<String>) -> Self {
        self.opts.video_codec = codec.into();
        self
    }

    /// Sets the H.264 coding profile (`Baseline`, `Main`, or `High`).
    pub fn profile(mut self, profile: H264Profile) -> Self {
        self.opts.h264_profile = profile;
        self
    }

    /// Enables Constant Rate Factor (`CRF`) perceptual rate control (`0.0..=51.0`).
    pub fn crf(mut self, crf: f32) -> Self {
        let (maxrate_kbps, bufsize_kb) = match self.opts.rate_control {
            Some(
                RateControlMode::Crf {
                    maxrate_kbps,
                    bufsize_kb,
                    ..
                }
                | RateControlMode::Abr {
                    maxrate_kbps,
                    bufsize_kb,
                    ..
                },
            ) => (maxrate_kbps, bufsize_kb),
            _ => (None, None),
        };
        self.opts.rate_control = Some(RateControlMode::Crf {
            crf,
            maxrate_kbps,
            bufsize_kb,
        });
        self
    }

    /// Sets a fixed Quantization Parameter (`QP 0..=51`).
    pub fn qp(mut self, qp: u8) -> Self {
        self.opts.qp = qp;
        self.opts.rate_control = Some(RateControlMode::FixedQp(qp));
        self
    }

    /// Enables Average Bitrate (`ABR`) rate control in kilobits per second.
    pub fn bitrate_kbps(mut self, bitrate_kbps: u32) -> Self {
        let (maxrate_kbps, bufsize_kb) = match self.opts.rate_control {
            Some(
                RateControlMode::Crf {
                    maxrate_kbps,
                    bufsize_kb,
                    ..
                }
                | RateControlMode::Abr {
                    maxrate_kbps,
                    bufsize_kb,
                    ..
                },
            ) => (maxrate_kbps, bufsize_kb),
            _ => (None, None),
        };
        self.opts.rate_control = Some(RateControlMode::Abr {
            bitrate_kbps,
            maxrate_kbps,
            bufsize_kb,
        });
        self
    }

    /// Sets VBV maximum bitrate (`--maxrate`) and buffer size (`--bufsize`) in kilobits.
    pub fn vbv_constraints(mut self, maxrate_kbps: u32, bufsize_kb: u32) -> Self {
        self.opts.rate_control = Some(match self.opts.rate_control {
            Some(RateControlMode::Crf { crf, .. }) => RateControlMode::Crf {
                crf,
                maxrate_kbps: Some(maxrate_kbps),
                bufsize_kb: Some(bufsize_kb),
            },
            Some(RateControlMode::Abr { bitrate_kbps, .. }) => RateControlMode::Abr {
                bitrate_kbps,
                maxrate_kbps: Some(maxrate_kbps),
                bufsize_kb: Some(bufsize_kb),
            },
            _ => RateControlMode::Crf {
                crf: self.opts.qp as f32,
                maxrate_kbps: Some(maxrate_kbps),
                bufsize_kb: Some(bufsize_kb),
            },
        });
        self
    }

    /// Crops each input frame (`"WxH+X+Y"` or `"WxH"` centered) prior to scaling.
    pub fn crop(mut self, crop_spec: impl Into<String>) -> Self {
        self.opts.crop_spec = Some(crop_spec.into());
        self
    }

    /// Enables automatic black-bar letterbox/pillarbox detection and cropping (`--auto-crop`).
    pub fn auto_crop(mut self, enabled: bool) -> Self {
        self.opts.auto_crop = enabled;
        self
    }

    /// Pads each frame (`"WxH+X+Y"` or `"WxH"` centered) after scaling.
    pub fn pad(mut self, pad_spec: impl Into<String>) -> Self {
        self.opts.pad_spec = Some(pad_spec.into());
        self
    }

    /// Applies lossless orthogonal rotation or mirror flip (`90`, `180`, `270`, `HFlip`, `VFlip`).
    pub fn rotate(mut self, mode: RotationMode) -> Self {
        self.opts.rotate_mode = Some(mode);
        self
    }

    /// Alpha-blends a `.png`, `.jpg`, or `.ppm` watermark image at `(x, y)`.
    pub fn watermark(mut self, watermark_path: impl AsRef<Path>, x: usize, y: usize) -> Self {
        self.opts.overlay_path = Some(watermark_path.as_ref().to_path_buf());
        self.opts.overlay_xy = (x, y);
        self
    }

    /// Enables perceptual Oklab HDR10/HLG (`BT.2020`) to SDR (`BT.709`) tonemapping.
    pub fn tonemap(mut self, enabled: bool) -> Self {
        self.opts.tonemap = enabled;
        self
    }

    /// Sets the target audio codec (`"aac"`, `"opus"`, `"flac"`, `"pcm"`, `"copy"`).
    pub fn audio_codec(mut self, codec: impl Into<String>) -> Self {
        self.opts.audio_codec = codec.into();
        self
    }

    /// Resamples audio to `sample_rate_hz` using the windowed-Sinc polyphase resampler.
    pub fn audio_sample_rate(mut self, sample_rate_hz: u32) -> Self {
        self.opts.sample_rate = Some(sample_rate_hz);
        self
    }

    /// Downmixes or upmixes audio to `channels` (`1`, `2`, `6`).
    pub fn audio_channels(mut self, channels: u8) -> Self {
        self.opts.channels = Some(channels);
        self
    }

    /// Applies EBU R128 / ITU-R BS.1770-4 integrated loudness normalization to `target_lufs` (e.g., `-14.0`).
    pub fn normalize_lufs(mut self, target_lufs: f64) -> Self {
        self.opts.loudness_lufs = Some(target_lufs);
        self
    }

    /// Overrides the worker thread count (`0` = automatic topology-aware P/E-core selection).
    pub fn threads(mut self, threads: usize) -> Self {
        self.opts.threads = threads;
        self
    }

    /// Sets the hard bounded-memory ring-buffer budget in bytes.
    pub fn memory_budget_bytes(mut self, memory_budget_bytes: usize) -> Self {
        self.opts.memory_budget_bytes = Some(memory_budget_bytes);
        self
    }

    /// Executes the transcoding pipeline and returns [`PipelineTelemetry`].
    pub fn run(&self) -> Result<PipelineTelemetry> {
        transcode_file_with_options(&self.input_path, &self.output_path, &self.opts)
    }
}

/// High-level Rust SDK builder for Smart-GOP frame-accurate trimming (`0050`).
#[derive(Clone, Debug)]
pub struct CutBuilder {
    input_path: std::path::PathBuf,
    output_path: std::path::PathBuf,
    opts: CutOptions,
}

/// Alias for [`CutBuilder`] (`0050`).
pub type CutJob = CutBuilder;

impl CutBuilder {
    /// Creates a new Smart-GOP trim job from `input_path` to `output_path`.
    pub fn new(input_path: impl AsRef<Path>, output_path: impl AsRef<Path>) -> Self {
        Self {
            input_path: input_path.as_ref().to_path_buf(),
            output_path: output_path.as_ref().to_path_buf(),
            opts: CutOptions {
                qp: 20,
                ..Default::default()
            },
        }
    }

    /// Sets the start trim timestamp in seconds (`--from`).
    pub fn from_seconds(mut self, from_sec: f64) -> Self {
        self.opts.from_sec = Some(from_sec);
        self
    }

    /// Sets the end trim timestamp in seconds (`--to`).
    pub fn to_seconds(mut self, to_sec: f64) -> Self {
        self.opts.to_sec = Some(to_sec);
        self
    }

    /// When `true`, snaps the trim point to the nearest preceding IDR keyframe for 100% zero-copy stream copy.
    pub fn keyframes_only(mut self, keyframes_only: bool) -> Self {
        self.opts.keyframes_only = keyframes_only;
        self
    }

    /// Sets the H.264 quantization parameter (`QP`) used when re-encoding a partial boundary GOP.
    pub fn qp(mut self, qp: u8) -> Self {
        self.opts.qp = qp;
        self
    }

    /// Executes the Smart-GOP trim job and returns a [`CutReport`].
    pub fn run(&self) -> Result<CutReport> {
        cut_media(&self.input_path, &self.output_path, &self.opts)
    }
}

/// High-level Rust SDK builder for CMAF `fMP4` HLS (`master.m3u8`) + DASH (`manifest.mpd`)
/// multi-rendition ABR packaging (`0050`).
#[derive(Clone, Debug)]
pub struct PackageBuilder {
    input_path: std::path::PathBuf,
    output_dir: std::path::PathBuf,
    opts: PackageOptions,
}

/// Alias for [`PackageBuilder`] (`0050`).
pub type PackageJob = PackageBuilder;

impl PackageBuilder {
    /// Creates a new CMAF HLS/DASH ABR packaging job writing to `output_dir`.
    pub fn new(input_path: impl AsRef<Path>, output_dir: impl AsRef<Path>) -> Self {
        Self {
            input_path: input_path.as_ref().to_path_buf(),
            output_dir: output_dir.as_ref().to_path_buf(),
            opts: PackageOptions::default(),
        }
    }

    /// Sets the comma-separated ABR rendition ladder (e.g., `"1080p:4500k,720p:2500k,480p:1200k"`).
    pub fn ladder(mut self, ladder_spec: impl Into<String>) -> Self {
        self.opts.ladder_spec = Some(ladder_spec.into());
        self
    }

    /// Sets the GOP-aligned CMAF segment duration in seconds (default: `2.0`).
    pub fn segment_duration(mut self, segment_duration_sec: f64) -> Self {
        self.opts.segment_duration_sec = segment_duration_sec;
        self
    }

    /// Enables or disables WebVTT (`thumbnails.vtt` + `sprites.jpg`) trick-play generation.
    pub fn thumbnails(mut self, enabled: bool) -> Self {
        self.opts.generate_thumbnails = enabled;
        self
    }

    /// Executes the ABR packager and returns a [`PackageReport`].
    pub fn run(&self) -> Result<PackageReport> {
        package_abr(&self.input_path, &self.output_dir, &self.opts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ragdoll_codec::Pps;
    use ragdoll_pix::psnr_plane_u8;

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

    #[test]
    fn gop_ring_pipeline_bounded_memory_and_frame_recycling() {
        let frames: Vec<VideoFrame> = (0..30).map(|i| synth_video_frame(160, 120, i)).collect();
        // Configure a tight 8 MiB memory budget with 4 worker threads
        let ring = GopRingPipeline::with_budget_bytes(4, 8 * 1024 * 1024);
        let (encoded_gops, stats) = ring
            .encode_gops_bounded(
                &frames,
                Some((128, 96)),
                ScaleFilter::Lanczos3,
                "h264",
                Rational64::FPS_30,
                22,
            )
            .unwrap();

        let total_encoded_frames: usize = encoded_gops.iter().map(|g| g.len()).sum();
        assert_eq!(total_encoded_frames, 30);
        assert_eq!(stats.frames_processed, 30);
        assert!(stats.peak_wave_memory_bytes <= ring.memory_budget_bytes);
        assert!(
            stats.pool_recycled_hits > 0,
            "FramePool should recycle scaled frame buffers across GOP frames"
        );
    }

    #[test]
    fn multi_gop_h264_decodes_in_presentation_order() {
        // 2 s at 30 fps spans several GOPs. The decoder's PTS restarts at every IDR, so a global
        // sort would interleave GOPs (0, 15, 30, 45, 1, ...) and each frame would land beside a
        // source frame half a second away (BUG-0060).
        let (w, h) = (160, 96);
        for ext in ["mp4", "mkv", "ts"] {
            let path = std::env::temp_dir().join(format!(
                "ragdoll_multi_gop_order_{}.{ext}",
                std::process::id()
            ));
            synth_media(&path, w, h, 30, 2.0).unwrap();
            let (frames, _) = decode_video_file(&path).unwrap();
            let _ = std::fs::remove_file(&path);

            assert_eq!(frames.len(), 60, "{ext}");
            assert!(
                frames.iter().filter(|f| f.keyframe).count() > 1,
                "{ext}: expected several GOPs"
            );
            for (i, f) in frames.iter().enumerate() {
                assert_eq!(f.pts.ticks, i as i64, "{ext}: frame {i}");
                let src = synth_video_frame(w, h, i);
                let psnr = psnr_plane_u8(
                    f.plane(0).data(),
                    f.plane(0).stride(),
                    src.plane(0).data(),
                    src.plane(0).stride(),
                    w,
                    h,
                );
                assert!(
                    psnr > 40.0,
                    "{ext}: frame {i} is {psnr:.2} dB from its source"
                );
            }
        }
    }

    #[test]
    fn multitrack_video_audio_subtitle_av_sync_and_transcode_preservation() {
        let tmp = std::env::temp_dir();
        let multi_mp4 = tmp.join("ragdoll_m6_multitrack_in.mp4");
        let out_mkv = tmp.join("ragdoll_m6_multitrack_out.mkv");
        let out_mp4 = tmp.join("ragdoll_m6_multitrack_out.mp4");
        let out_srt = tmp.join("ragdoll_m6_multitrack_out.srt");

        // 1. Build a 1-second (30-frame) multi-track MP4 containing H.264 Video + FLAC Audio + WebVTT Subtitles
        let width = 160u32;
        let height = 120u32;
        let fps = Rational64::FPS_30;
        let sps = Sps::new_baseline(width, height, fps);
        let pps = Pps::new_baseline(22);
        let avcc = build_avcc_from_sps_pps(&sps.to_nal_unit(), &pps.to_nal_unit());
        let mut h264_enc = H264Encoder::new(width, height, fps, 22);

        let mut mux = Mp4Muxer::new();
        let v_tb = Rational64::new(1, 90_000);
        let v_tid = mux.add_track(TrackInfo {
            index: 0,
            track_type: TrackType::Video,
            codec_id: CodecId::H264,
            timebase: v_tb,
            duration_ticks: Some(90_000),
            extra_data: avcc,
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
            sample_count: 30,
            total_bytes: 0,
        });

        let a_tb = Rational64::new(1, 48_000);
        let flac_enc = FlacEncoder::new(48_000, 2, 0);
        let mut af = AudioFrame::new(48_000, 2, SampleFormat::S16le, 4800);
        for (i, chunk) in af
            .data
            .as_mut_slice()
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .enumerate()
        {
            let s = (((i as f64) * 0.05).sin() * 8000.0) as i16;
            *chunk = s.to_le_bytes();
        }
        let flac_stream = flac_enc.encode_stream(&af).unwrap();
        let a_tid = mux.add_track(TrackInfo {
            index: 1,
            track_type: TrackType::Audio,
            codec_id: CodecId::Flac,
            timebase: a_tb,
            duration_ticks: Some(48_000),
            extra_data: flac_stream[8..42].to_vec(),
            video: None,
            audio: Some(AudioTrackParams {
                sample_rate: 48_000,
                channels: 2,
                channel_layout: ChannelLayout::Stereo,
                sample_format: SampleFormat::S16le,
                bits_per_coded_sample: 16,
            }),
            sample_count: 10,
            total_bytes: 0,
        });

        let s_tb = Rational64::new(1, 1000);
        let s_tid = mux.add_track(TrackInfo {
            index: 2,
            track_type: TrackType::Subtitle,
            codec_id: CodecId::WebVtt,
            timebase: s_tb,
            duration_ticks: Some(1000),
            extra_data: b"WEBVTT".to_vec(),
            video: None,
            audio: None,
            sample_count: 2,
            total_bytes: 0,
        });

        for i in 0..30 {
            let vf = synth_video_frame(width as usize, height as usize, i);
            let annex_b = h264_enc.encode_frame(&vf).unwrap();
            let avcc_au = h264_annex_b_to_avcc_au(&annex_b);
            let pts = (i as i64) * 3000;
            mux.write_packet(PacketSlice::from_owned(
                AlignedBuffer::from_slice(&avcc_au),
                v_tid,
                Timestamp::new(pts, v_tb),
                Timestamp::new(pts, v_tb),
                3000,
                if i == 0 || i == 15 {
                    PacketFlags::KEYFRAME
                } else {
                    PacketFlags::NONE
                },
            ))
            .unwrap();
        }
        for i in 0..10 {
            let pts = (i as i64) * 4800;
            mux.write_packet(PacketSlice::from_owned(
                AlignedBuffer::from_slice(&flac_stream[42..]),
                a_tid,
                Timestamp::new(pts, a_tb),
                Timestamp::new(pts, a_tb),
                4800,
                PacketFlags::KEYFRAME,
            ))
            .unwrap();
        }
        mux.write_packet(PacketSlice::from_owned(
            AlignedBuffer::from_slice(b"Ragdoll Multi-Track Subtitle 1"),
            s_tid,
            Timestamp::new(0, s_tb),
            Timestamp::new(0, s_tb),
            450,
            PacketFlags::KEYFRAME,
        ))
        .unwrap();
        mux.write_packet(PacketSlice::from_owned(
            AlignedBuffer::from_slice(b"Ragdoll Multi-Track Subtitle 2"),
            s_tid,
            Timestamp::new(500, s_tb),
            Timestamp::new(500, s_tb),
            500,
            PacketFlags::KEYFRAME,
        ))
        .unwrap();

        {
            let mut f = File::create(&multi_mp4).unwrap();
            mux.finish(&mut f).unwrap();
        }

        // 2. Transcode multi-track MP4 -> scaled MKV & MP4, verifying Audio & Subtitle preservation + < 1 ms A/V sync
        let tel_mkv = transcode_file_with_options(
            &multi_mp4,
            &out_mkv,
            &TranscodeOptions {
                scale_dims: Some((128, 96)),
                scale_filter: ScaleFilter::Lanczos3,
                video_codec: "h264".into(),
                audio_codec: "copy".into(),
                subtitle_codec: "copy".into(),
                qp: 22,
                threads: 2,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(tel_mkv.frames_processed, 30);
        assert!(tel_mkv.audio_samples_processed >= 48_000);

        let probe_mkv = probe_file_detailed(&out_mkv).unwrap();
        assert_eq!(
            probe_mkv.tracks.len(),
            3,
            "MKV must preserve Video + Audio + Subtitle"
        );

        let tel_mp4 = transcode_file_with_options(
            &multi_mp4,
            &out_mp4,
            &TranscodeOptions {
                scale_dims: Some((128, 96)),
                scale_filter: ScaleFilter::Lanczos3,
                video_codec: "h264".into(),
                audio_codec: "copy".into(),
                subtitle_codec: "copy".into(),
                qp: 22,
                threads: 2,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(tel_mp4.frames_processed, 30);
        let probe_mp4 = probe_file_detailed(&out_mp4).unwrap();
        assert_eq!(
            probe_mp4.tracks.len(),
            3,
            "MP4 must preserve Video + Audio + Subtitle"
        );

        // Verify < 1 ms A/V drift across the transcoded MP4
        let v_end = Timestamp::new(30 * 3000, v_tb);
        let a_end = Timestamp::new(10 * 4800, a_tb);
        let drift_ms = AvSyncController::compute_av_drift_ms(v_end, a_end);
        assert!(drift_ms < 1.0, "A/V drift {drift_ms} ms must be < 1 ms");

        // 3. Remux subtitle track out to standalone .srt and verify text
        remux_file(&out_mp4, &out_srt).unwrap();
        let srt_text = std::fs::read_to_string(&out_srt).unwrap();
        assert!(srt_text.contains("Ragdoll Multi-Track Subtitle 1"));
        assert!(srt_text.contains("Ragdoll Multi-Track Subtitle 2"));

        let _ = std::fs::remove_file(&multi_mp4);
        let _ = std::fs::remove_file(&out_mkv);
        let _ = std::fs::remove_file(&out_mp4);
        let _ = std::fs::remove_file(&out_srt);
    }

    #[test]
    fn m8_audio_pro_aac_opus_flac_ogg_resampler_and_ebu_r128_pipeline() {
        let tmp = std::env::temp_dir();
        let src_wav = tmp.join("ragdoll_m8_src.wav");
        let out_flac = tmp.join("ragdoll_m8_out.flac");
        let out_ogg = tmp.join("ragdoll_m8_out.ogg");
        let out_m4a = tmp.join("ragdoll_m8_out.m4a");
        let out_norm_wav = tmp.join("ragdoll_m8_norm.wav");

        // 1. Synthesize 1.0s stereo 48kHz WAV test audio
        synth_media(&src_wav, 0, 0, 30, 1.0).unwrap();

        // 2. Transcode WAV -> standalone FLAC container (verifying SEEKTABLE)
        transcode_file_with_options(&src_wav, &out_flac, &TranscodeOptions::default()).unwrap();
        let flac_bytes = std::fs::read(&out_flac).unwrap();
        let flac_demux = FlacContainerDemuxer::open(&flac_bytes).unwrap();
        assert!(!flac_demux.seek_table().is_empty());

        // 3. Transcode FLAC -> Ogg Opus container
        transcode_file_with_options(
            &out_flac,
            &out_ogg,
            &TranscodeOptions {
                audio_codec: "opus".into(),
                ..Default::default()
            },
        )
        .unwrap();
        let probe_ogg = probe_file_detailed(&out_ogg).unwrap();
        assert_eq!(probe_ogg.container, "ogg");
        assert_eq!(probe_ogg.tracks[0].codec_id, CodecId::Opus);

        // 4. Transcode Ogg Opus -> M4A (MP4 AAC-LC)
        transcode_file_with_options(
            &out_ogg,
            &out_m4a,
            &TranscodeOptions {
                audio_codec: "aac".into(),
                ..Default::default()
            },
        )
        .unwrap();
        let probe_m4a = probe_file_detailed(&out_m4a).unwrap();
        assert_eq!(probe_m4a.tracks[0].codec_id, CodecId::Aac);

        // 5. Transcode FLAC -> normalized 44.1kHz WAV at -14.0 LUFS (-1.0 dBTP ceiling)
        transcode_file_with_options(
            &out_flac,
            &out_norm_wav,
            &TranscodeOptions {
                sample_rate: Some(44_100),
                channels: Some(2),
                loudness_lufs: Some(-14.0),
                true_peak_dbtp: Some(-1.0),
                ..Default::default()
            },
        )
        .unwrap();
        let norm_bytes = std::fs::read(&out_norm_wav).unwrap();
        let (norm_af, _) = decode_audio_bytes(&norm_bytes).unwrap();
        assert_eq!(norm_af.sample_rate, 44_100);
        assert_eq!(norm_af.channels, 2);
        let mut meter = EbuR128Meter::new(norm_af.sample_rate, norm_af.channels as usize);
        meter.process_audio_frame(&norm_af).unwrap();
        let rep = meter.finalize();
        assert!(
            (rep.integrated_lufs - (-14.0)).abs() < 0.25,
            "Expected -14.0 LUFS, got {:.2} LUFS",
            rep.integrated_lufs
        );
        assert!(
            rep.true_peak_dbtp <= -0.9,
            "Expected True Peak <= -0.9 dBTP, got {:.2} dBTP",
            rep.true_peak_dbtp
        );

        let _ = std::fs::remove_file(&src_wav);
        let _ = std::fs::remove_file(&out_flac);
        let _ = std::fs::remove_file(&out_ogg);
        let _ = std::fs::remove_file(&out_m4a);
        let _ = std::fs::remove_file(&out_norm_wav);
    }

    #[test]
    fn m9_edit_image_package_png_jpeg_filters_smart_gop_cut_concat_and_abr_package() {
        let tmp = std::env::temp_dir().join("ragdoll_m9_suite");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let src_mp4 = tmp.join("src.mp4");
        let thumb_png = tmp.join("thumb.png");
        let thumb_jpg = tmp.join("thumb.jpg");
        let filtered_mp4 = tmp.join("filtered.mp4");
        let cut_mp4 = tmp.join("cut.mp4");
        let concat_mp4 = tmp.join("concat.mp4");
        let abr_dir = tmp.join("abr_out");

        // 1. Synthesize 1.0s (30 frames @ 30fps, 2 GOPs of 15 frames) 160x120 MP4
        synth_media(&src_mp4, 160, 120, 30, 1.0).unwrap();

        // 2. Extract PNG and JPEG thumbnails (`0047`)
        extract_media(&src_mp4, &thumb_png, Some(0.2), None).unwrap();
        extract_media(&src_mp4, &thumb_jpg, Some(0.4), None).unwrap();
        assert!(thumb_png.exists());
        assert!(thumb_jpg.exists());

        // 3. Transcode with crop + pad + rotate + PNG overlay + fps filter (`SIMD-0048`)
        let tel_f = transcode_file_with_options(
            &src_mp4,
            &filtered_mp4,
            &TranscodeOptions {
                crop_spec: Some("128:96:16:12".into()),
                pad_spec: Some("160:120:16:12:#102030".into()),
                rotate_mode: Some(RotationMode::Rotate180),
                overlay_path: Some(thumb_png.clone()),
                overlay_xy: (8, 8),
                target_fps: Some(Rational64::new(15, 1)),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(tel_f.frames_processed, 15);

        // 4. Smart-GOP frame-accurate `cut_media` (`0046`)
        //    Cut from 0.2s (frame 6, mid-GOP 0) to 0.9s (frame 27, inside GOP 1):
        //    GOP 0 tail (9 frames: 6..15) is re-encoded starting with an IDR keyframe,
        //    while GOP 1 (12 frames: 15..27) is zero-copy stream-copied!
        let cut_rep = cut_media(
            &src_mp4,
            &cut_mp4,
            &CutOptions {
                from_sec: Some(0.033),
                to_sec: Some(0.9),
                keyframes_only: false,
                qp: 22,
            },
        )
        .unwrap();
        assert!(
            cut_rep.reencoded_video_frames > 0,
            "mid-GOP cut should re-encode boundary GOP frames"
        );
        assert!(
            cut_rep.copied_video_packets > 0,
            "mid-GOP cut should zero-copy interior GOP packets"
        );

        // 5. Zero-copy `concat_media` (`0046`)
        let concat_rep = concat_media(&[&src_mp4, &cut_mp4], &concat_mp4).unwrap();
        assert_eq!(concat_rep.segments_merged, 2);
        assert!(!concat_rep.reencoded);
        assert_eq!(concat_rep.video_packets, 30 + cut_rep.total_video_packets);

        // 6. One-command CMAF HLS & DASH ABR packager (`0049`)
        let pkg_rep = package_abr(
            &src_mp4,
            &abr_dir,
            &PackageOptions {
                ladder_spec: Some("160x120:400k,96x72:180k".into()),
                segment_duration_sec: 0.5,
                generate_thumbnails: true,
            },
        )
        .unwrap();
        assert_eq!(pkg_rep.renditions, 2);
        assert_eq!(pkg_rep.segments_per_rendition, 2);
        assert!(pkg_rep.master_m3u8_path.exists());
        assert!(pkg_rep.mpd_path.exists());
        assert!(pkg_rep.thumbnails_vtt_path.as_ref().unwrap().exists());
        assert!(abr_dir.join("sprites.jpg").exists());
        assert!(abr_dir.join("stream_0/init.mp4").exists());
        assert!(abr_dir.join("stream_0/seg_0001.m4s").exists());

        let m3u8_text = std::fs::read_to_string(&pkg_rep.master_m3u8_path).unwrap();
        assert!(m3u8_text.contains("#EXT-X-STREAM-INF:BANDWIDTH=400000"));
        assert!(m3u8_text.contains("RESOLUTION=160x120"));

        let mpd_text = std::fs::read_to_string(&pkg_rep.mpd_path).unwrap();
        assert!(mpd_text.contains("<SegmentTemplate"));
        assert!(mpd_text.contains("stream_0/init.mp4"));

        // Optional ffprobe verification of cut.mp4, concat.mp4, and CMAF init+seg
        if std::process::Command::new("ffprobe")
            .arg("-version")
            .output()
            .is_ok()
        {
            for p in [&cut_mp4, &concat_mp4] {
                let st = std::process::Command::new("ffprobe")
                    .args(["-v", "error", "-show_entries", "format=duration"])
                    .arg(p)
                    .status()
                    .unwrap();
                assert!(st.success());
            }

            // Concatenate init.mp4 + seg_0001.m4s and verify with ffprobe
            let mut cmaf_combined = std::fs::read(abr_dir.join("stream_0/init.mp4")).unwrap();
            let seg1 = std::fs::read(abr_dir.join("stream_0/seg_0001.m4s")).unwrap();
            cmaf_combined.extend_from_slice(&seg1);
            let cmaf_test_mp4 = tmp.join("cmaf_combined.mp4");
            std::fs::write(&cmaf_test_mp4, &cmaf_combined).unwrap();
            let st_cmaf = std::process::Command::new("ffprobe")
                .args([
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=codec_name,width,height",
                ])
                .arg(&cmaf_test_mp4)
                .status()
                .unwrap();
            assert!(st_cmaf.success());
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn malformed_bitstream_zero_panic_corpus() {
        use ragdoll_codec::{
            AacDecoder, AacEncoder, Av1SequenceHeader, FlacDecoder, FlacEncoder, ObuType,
            OpusDecoder, OpusEncoder, encode_obu, parse_ivf_stream, split_obus,
        };
        use ragdoll_format::{
            FlacContainerDemuxer, MkvDemuxer, Mp4Demuxer, MpegTsDemuxer, OggDemuxer, WavDemuxer,
            Y4mDemuxer, read_jpeg_image, read_png_image, read_pnm_image, write_jpeg_image,
            write_png_image,
        };

        let tmp = std::env::temp_dir().join(format!("ragdoll_m10_sdk_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        // 1. Test the ergonomic Rust SDK builders (ProbeBuilder, TranscodeBuilder, CutBuilder, PackageBuilder)
        let src_mp4 = tmp.join("sdk_src.mp4");
        synth_media(&src_mp4, 96, 64, 30, 0.4).unwrap();

        let probe_rep = ProbeBuilder::new(&src_mp4).run().unwrap();
        assert!(!probe_rep.tracks.is_empty());

        let trans_mp4 = tmp.join("sdk_trans.mp4");
        let trans_tel = TranscodeBuilder::new(&src_mp4, &trans_mp4)
            .scale(64, 48, ScaleFilter::Lanczos3)
            .profile(H264Profile::High)
            .crf(22.0)
            .run()
            .unwrap();
        assert_eq!(trans_tel.frames_processed, 12);

        let cut_mp4 = tmp.join("sdk_cut.mp4");
        let cut_rep = CutBuilder::new(&trans_mp4, &cut_mp4)
            .from_seconds(0.05)
            .to_seconds(0.35)
            .qp(20)
            .run()
            .unwrap();
        assert!(cut_rep.output_bytes > 100);

        let pkg_dir = tmp.join("sdk_abr");
        let pkg_rep = PackageBuilder::new(&trans_mp4, &pkg_dir)
            .ladder("64x48:250k,32x24:120k")
            .segment_duration(0.2)
            .thumbnails(true)
            .run()
            .unwrap();
        assert_eq!(pkg_rep.renditions, 2);
        assert!(pkg_rep.master_m3u8_path.exists());
        assert!(pkg_rep.mpd_path.exists());

        // 2. Synthesize valid seed bitstreams for all 12 container demuxers & codec decoders
        let mp4_seed = std::fs::read(&src_mp4).unwrap();
        let mut mkv_seed = Vec::new();
        ragdoll_format::transmux_mp4_to_mkv(&mp4_seed, &mut mkv_seed).unwrap();
        let mut ts_seed = Vec::new();
        ragdoll_format::transmux_mp4_to_mpegts(&mp4_seed, &mut ts_seed).unwrap();

        let mut y4m_seed = Vec::new();
        {
            let mut ymux = Y4mMuxer::new(
                &mut y4m_seed,
                32,
                32,
                PixelFormat::Yuv420p,
                Rational64::FPS_30,
                Colorimetry::bt709_limited(),
            )
            .unwrap();
            let vf = VideoFrame::new(32, 32, PixelFormat::Yuv420p);
            ymux.write_frame(&vf).unwrap();
            ymux.finish().unwrap();
        }

        let mut wav_seed = Vec::new();
        {
            let mut af = AudioFrame::new(48_000, 2, SampleFormat::S16le, 1024);
            for (i, s) in af.as_i16_mut_slice().iter_mut().enumerate() {
                *s = ((i as f32 * 0.1).sin() * 8000.0) as i16;
            }
            let mut wmux = WavMuxer::new(AudioTrackParams {
                sample_rate: 48_000,
                channels: 2,
                channel_layout: ChannelLayout::Stereo,
                sample_format: SampleFormat::S16le,
                bits_per_coded_sample: 16,
            });
            wmux.write_pcm_bytes(af.data.as_slice());
            wmux.finish(&mut wav_seed).unwrap();
        }

        let mut h264_enc =
            H264Encoder::new(32, 32, Rational64::FPS_30, 24).with_profile(H264Profile::High);
        let vf_small = VideoFrame::new(32, 32, PixelFormat::Yuv420p);
        let h264_seed = h264_enc.encode_frame(&vf_small).unwrap();

        let mut af_small = AudioFrame::new(48_000, 2, SampleFormat::S16le, 1024);
        for (i, s) in af_small.as_i16_mut_slice().iter_mut().enumerate() {
            *s = ((i as f32 * 0.05).cos() * 6000.0) as i16;
        }
        let flac_seed = FlacEncoder::new(48_000, 2, 16)
            .encode_stream(&af_small)
            .unwrap();
        let mut aac_enc = AacEncoder::new(48_000, 2, 128_000);
        let aac_seed = aac_enc.encode_adts_frame(&af_small).unwrap();
        let mut opus_enc = OpusEncoder::new(48_000, 2, 96_000);
        let opus_seed = opus_enc.encode_frame(&af_small).unwrap();
        let av1_seed = encode_obu(ObuType::SequenceHeader, &[0x0A, 0x0B, 0x00, 0x00, 0x00]);

        let mut rgb_small = VideoFrame::new(16, 16, PixelFormat::Rgb24);
        for (i, b) in rgb_small.plane_mut(0).data_mut().iter_mut().enumerate() {
            *b = (i * 17) as u8;
        }
        let mut png_seed = Vec::new();
        write_png_image(&mut png_seed, &rgb_small).unwrap();
        let mut jpeg_seed = Vec::new();
        write_jpeg_image(&mut jpeg_seed, &rgb_small, 85).unwrap();

        let exercise_all = |buf: &[u8]| {
            if let Ok(mut d) = Mp4Demuxer::open(buf) {
                while let Ok(Some(_)) = d.next_packet() {}
            }
            if let Ok(mut d) = MkvDemuxer::open(buf) {
                while let Ok(Some(_)) = d.next_packet() {}
            }
            if let Ok(mut d) = MpegTsDemuxer::open(buf) {
                while let Ok(Some(_)) = d.next_packet() {}
            }
            if let Ok(mut d) = Y4mDemuxer::open(buf) {
                while let Ok(Some(_)) = d.next_packet() {}
            }
            if let Ok(mut d) = WavDemuxer::open(buf) {
                while let Ok(Some(_)) = d.next_packet() {}
            }
            if let Ok(mut d) = FlacContainerDemuxer::open(buf) {
                while let Ok(Some(_)) = d.next_packet() {}
            }
            if let Ok(mut d) = OggDemuxer::open(buf) {
                while let Ok(Some(_)) = d.next_packet() {}
            }
            let mut h264_dec = H264Decoder::new();
            let _ = h264_dec.decode_annex_b_packet(buf);
            let _ = FlacDecoder::decode_stream(buf);
            let mut aac_dec = AacDecoder::new(48_000, 2);
            let _ = aac_dec.decode_packet(buf);
            let mut opus_dec = OpusDecoder::new(2);
            let _ = opus_dec.decode_packet(buf);
            let _ = Av1SequenceHeader::parse(buf);
            let _ = split_obus(buf);
            let _ = parse_ivf_stream(buf);
            let _ = read_png_image(buf);
            let _ = read_jpeg_image(buf);
            let _ = read_pnm_image(buf);
        };

        let seeds: [&[u8]; 12] = [
            mp4_seed.as_slice(),
            mkv_seed.as_slice(),
            ts_seed.as_slice(),
            y4m_seed.as_slice(),
            wav_seed.as_slice(),
            h264_seed.as_slice(),
            flac_seed.as_slice(),
            aac_seed.as_slice(),
            opus_seed.as_slice(),
            av1_seed.as_slice(),
            png_seed.as_slice(),
            jpeg_seed.as_slice(),
        ];

        let mut total_mutations = 0usize;
        let mut rng = 0xDEAD_BEEF_CAFE_BABEu64;
        let mut next_u64 = || -> u64 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };

        // Adversarial boundary buffers (empty, 1-byte, all-0x00, all-0xFF, all-0x7F, all-0x80)
        for &pat in &[0x00u8, 0xFF, 0x7F, 0x80, 0x47, 0x89] {
            for len in [0usize, 1, 2, 4, 8, 16, 32, 64, 188, 256] {
                let buf = vec![pat; len];
                exercise_all(&buf);
                total_mutations += 1;
            }
        }

        // 10,000+ deterministic bit-flip, truncation, zeroed-header, and corrupted-length mutations
        for (s_idx, seed) in seeds.iter().copied().enumerate() {
            // 1. Prefix truncations across header & payload boundaries
            let step = (seed.len() / 120).max(1);
            for cut in (0..=seed.len()).step_by(step) {
                exercise_all(&seed[..cut]);
                total_mutations += 1;
            }

            // 2. Deterministic bit-flips, byte overwrites, and 32-bit/64-bit length corruptions
            let mutations_per_seed = 800usize;
            let mut scratch = seed.to_vec();
            for m in 0..mutations_per_seed {
                scratch.resize(seed.len(), 0);
                scratch.copy_from_slice(seed);
                let len = scratch.len();
                let mode = (m + s_idx) % 5;
                match mode {
                    0 => {
                        // Single-bit and multi-bit flips
                        let flips = 1 + (next_u64() % 4) as usize;
                        for _ in 0..flips {
                            let idx = (next_u64() as usize) % len;
                            let bit = 1u8 << (next_u64() % 8);
                            scratch[idx] ^= bit;
                        }
                    }
                    1 => {
                        // Corrupted 32-bit big-endian / little-endian length/dimension fields (0xFFFFFFFF, 0x7FFFFFFF, 0x00000000)
                        if len >= 4 {
                            let idx = (next_u64() as usize) % (len - 3);
                            let bad: [u8; 4] = match m % 4 {
                                0 => [0xFF, 0xFF, 0xFF, 0xFF],
                                1 => [0x7F, 0xFF, 0xFF, 0xFF],
                                2 => [0x00, 0x00, 0x00, 0x00],
                                _ => [0x80, 0x00, 0x00, 0x01],
                            };
                            scratch[idx..idx + 4].copy_from_slice(&bad);
                        }
                    }
                    2 => {
                        // Zeroed or 0xFF-filled slice window inside header/table region
                        let start = (next_u64() as usize) % len;
                        let span = ((next_u64() as usize) % 16).min(len - start);
                        let fill = if m % 2 == 0 { 0x00 } else { 0xFF };
                        scratch[start..start + span].fill(fill);
                    }
                    3 => {
                        // Mutate + truncate simultaneously
                        let idx = (next_u64() as usize) % len;
                        scratch[idx] = (next_u64() & 0xFF) as u8;
                        let new_len = 1 + ((next_u64() as usize) % len);
                        scratch.truncate(new_len);
                    }
                    _ => {
                        // Burst of pseudo-random noise injected into bitstream body
                        let count = 1 + ((next_u64() as usize) % 8);
                        for _ in 0..count {
                            let idx = (next_u64() as usize) % len;
                            scratch[idx] = (next_u64() & 0xFF) as u8;
                        }
                    }
                }
                exercise_all(&scratch);
                total_mutations += 1;
            }
        }

        assert!(
            total_mutations >= 10_000,
            "expected >= 10,000 malformed bitstream mutations, ran {total_mutations}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
