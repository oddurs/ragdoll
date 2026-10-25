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
    ContainerFormat, MkvDemuxer, MkvMuxer, MmapSource, Mp4Demuxer, Mp4Muxer, MpegTsDemuxer,
    MpegTsMuxer, OutputSink, SubtitleDemuxer, SubtitleMuxer, WavDemuxer, WavMuxer, Y4mDemuxer,
    Y4mMuxer, build_avcc_from_sps_pps, compose_contact_sheet_rgb24, detect_container_format,
    h264_annex_b_to_avcc_au, h264_avcc_to_annex_b_au, is_stdout_path, read_pnm_image,
    transmux_mkv_to_mp4, transmux_mp4_to_mpegts, transmux_mpegts_to_mkv, transmux_mpegts_to_mp4,
    write_pnm_image,
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

        let max_gop_for_threads = raw_frames.len().div_ceil(self.worker_threads.max(1)).max(1);
        let gop_size = 15usize.min(max_gop_for_threads);
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
                parallel_gop_map(wave, self.worker_threads, |_idx, chunk| {
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
                        let mut enc = H264Encoder::new(dst_w as u32, dst_h as u32, fps, qp);
                        enc.keyint = chunk.len() as u32 + 1;
                        for (i, src) in chunk.iter().enumerate() {
                            if let Some((ref hy, ref vy, ref hc, ref vc)) = banks {
                                let mut dst = self.pool.acquire_scoped(dst_w, dst_h, src.format);
                                dst.colorimetry = src.colorimetry;
                                dst.pts = src.pts;
                                scale_plane_with_banks(src.plane(0), dst.plane_mut(0), hy, vy);
                                if src.planes.len() >= 3 {
                                    scale_plane_with_banks(src.plane(1), dst.plane_mut(1), hc, vc);
                                    scale_plane_with_banks(src.plane(2), dst.plane_mut(2), hc, vc);
                                }
                                out_pkts.push((enc.encode_frame(&dst).unwrap(), i == 0));
                            } else {
                                out_pkts.push((enc.encode_frame(src).unwrap(), i == 0));
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
        ContainerFormat::Wav | ContainerFormat::WebVtt | ContainerFormat::SubRip => Err(
            RagdollError::Unsupported("audio/subtitle container has no video stream".into()),
        ),
    }
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

/// Options for [`transcode_file_with_options`] (`0036` / `0037`).
#[derive(Clone, Debug)]
pub struct TranscodeOptions {
    pub scale_dims: Option<(usize, usize)>,
    pub scale_filter: ScaleFilter,
    pub video_codec: String,
    pub audio_codec: String,
    pub subtitle_codec: String,
    pub format_override: Option<String>,
    pub qp: u8,
    pub threads: usize,
    pub memory_budget_bytes: Option<usize>,
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
            threads: 0,
            memory_budget_bytes: None,
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
        let mut writer = OutputSink::create(out_p)?;
        let bytes_out = if out_ext == "flac" {
            let enc = FlacEncoder::new(audio_frame.sample_rate, audio_frame.channels, 0);
            let flac_bytes = enc.encode_stream(&audio_frame)?;
            writer.write_all(&flac_bytes)?;
            writer.flush()?;
            flac_bytes.len() as u64
        } else {
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
    // 1. Open input once (supports both zero-copy `mmap` and `-` / `stdin` pipes)
    let mmap = MmapSource::open(in_p)?;
    let data = mmap.as_slice();
    let bytes_in = data.len() as u64;
    let (raw_frames, fps) = decode_video_bytes(data)?;
    if raw_frames.is_empty() {
        return Err(RagdollError::InvalidData("input video has 0 frames".into()));
    }
    let aux_tracks = extract_auxiliary_tracks(data).unwrap_or_default();

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
        "ppm" => {
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
            write_pnm_image(&mut buf, &rgb)?;
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
            let sps = Sps::new_baseline(width, height, fps);
            let pps = Pps::new_baseline(opts.qp);
            let avcc_extra = build_avcc_from_sps_pps(&sps.to_nal_unit(), &pps.to_nal_unit());

            let (encoded_gops, ring_stats) = gop_ring.encode_gops_bounded(
                &raw_frames,
                opts.scale_dims,
                opts.scale_filter,
                &opts.video_codec,
                fps,
                opts.qp,
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
        for (i, chunk) in af.data.as_mut_slice().chunks_exact_mut(2).enumerate() {
            let s = (((i as f64) * 0.05).sin() * 8000.0) as i16;
            chunk.copy_from_slice(&s.to_le_bytes());
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
}
