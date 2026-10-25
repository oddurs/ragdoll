---
id: 35
uid: b8754572-6f08-4151-aee0-75daf74292bb
title: 'Opinionated Production Philosophy: Bounded GOP Ring Streaming, Multi-Track A/V Sync & Smart-GOP Verbs'
type: decision
status: done
milestone: m6-streaming-av-sync
assignee: Oddur Sigurdsson
created: 2026-10-25
updated: 2026-10-25
closed_at: 2026-10-25
priority: p0
area: arch
effort: s
---

## Context

Ragdoll `v0.1.0` proved that 100% pure Safe + Explicit SIMD Rust with zero C FFI outperforms FFmpeg 9.0.2 by `1.4x–577x` in speed and `2.1x–1,346x` in battery efficiency across isolated kernels and single-track clips. However, to serve as a true production replacement for `ffmpeg` and `ffprobe`, Ragdoll must handle multi-hour feature films without unbounded RAM growth, preserve synchronized multi-track audio/video/subtitles by default, and provide human-centric editing and packaging workflows without cloning FFmpeg's 1990s flag soup.

## Options & Trade-offs

1. **Clone FFmpeg's CLI flags and unbounded filtergraph model**: Familiar to legacy scripts, but inherits FFmpeg's positional flag ambiguity, silent track dropping, and broken keyframe-misaligned stream-copy cutting (`-ss ... -c copy`).
2. **Opinionated Ragdoll `v1.0` Architecture**:
   - **Bounded-Memory GOP Ring Streaming (`GopRingPipeline`)**: Never buffer a whole video in `Vec<VideoFrame>`. Stream GOP windows through a `FramePool` capped at a strict memory budget (`256 MiB` default, `64 MiB` in `--battery-saver` mode).
   - **Multi-Track A/V + Subtitle Preservation by Default**: Unless `--an` or `--vn` is passed, `remux`, `transcode`, `cut`, and `concat` preserve and synchronize all video, audio, and subtitle (`WebVTT`/`SRT`) tracks using 128-bit `rescale_delta` lip-sync alignment.
   - **Smart-GOP Editing (`ragdoll cut` & `ragdoll concat`)**: Automatically re-encode only the boundary GOPs straddling non-keyframe cut points while zero-copy `mmap` stream-copying the interior GOPs.
   - **One-Command ABR Packaging (`ragdoll package`)**: Decode each GOP once, fan out fused L1 multi-resolution scaling in memory, encode all ladder rungs in parallel, and emit CMAF HLS (`master.m3u8`) + DASH (`manifest.mpd`) + `fMP4` segments in a single pass.

## Decision

Adopt Option 2 across Milestones `m6` through `m10`. Move full-file in-memory helpers behind a streaming `GopRingPipeline`, elevate `cut`, `concat`, and `package` to first-class CLI verbs, and reject legacy formats (AVI, FLV, RealMedia, WMV, interlaced field coding) in favor of deep mastery over modern containers (`MP4`/`fMP4`/`CMAF`, `MKV`/`WebM`, `MPEG-TS`, `FLAC`, `Ogg`, `WAV`, `Y4M`, `PNG`, `JPEG`) and codecs (`H.264`, `AV1`, `RGDI`, `FLAC`, `AAC-LC`, `Opus`, `PCM`).

## Acceptance criteria

- [x] The architectural invariant, its rationale, and its consequences are recorded.
- [x] Milestones `m6-streaming-av-sync` through `m10-production-v1` are linked to this decision in Cairn.

## 2026-10-25

Recorded Ragdoll v1.0 opinionated production architecture: bounded GopRingPipeline (<256 MiB / <64 MiB battery-saver), default multi-track A/V+Subtitle preservation with 128-bit rescale_delta lip-sync, Smart-GOP cut/concat, and single-decode CMAF ABR package.

## Result

Adopted Option 2 (Opinionated Ragdoll v1.0 Architecture) across Milestones m6 through m10.
