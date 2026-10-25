---
id: 23
uid: 94d7e96c-af40-4fea-b8f4-22ba36717e5c
title: 'Head-to-Head Benchmark Suite: Ragdoll vs. FFmpeg 9.0.2 on Apple M4 across 6 Flagship Workloads'
type: benchmark
status: planned
milestone: m5-cli-bench
depends_on:
- 22
created: 2026-10-24
updated: 2026-10-24
priority: p0
area: bench
effort: l
---

## Workload & Fair Comparison Protocol

Execute live, reproducible, apples-to-apples benchmarks comparing `ragdoll` (release build, `aarch64` NEON) against the host's installed `ffmpeg 9.0.2` / `ffprobe 9.0.2` on Apple M4 across 6 core multimedia workloads:
1. **Container Probe & GOP Indexing**: Deep stream & sample-table inspection of a multi-hundred-frame MP4 file (`ragdoll probe` vs `ffprobe -show_streams -show_packets`).
2. **Zero-Copy Remuxing / Transmuxing**: High-bitrate MP4 -> MKV and MP4 -> Faststart MP4 (`ragdoll remux` vs `ffmpeg -c copy`).
3. **4K -> 1080p Fused Lanczos-3 Scaling & Color Pipeline**: 3840x2160 -> 1920x1080 BT.709 YUV420p Lanczos-3 downscaling (`ragdoll transcode --scale 1920x1080 --filter lanczos3` vs `ffmpeg -vf scale=1920:1080:flags=lanczos`).
4. **Lossless Audio Coding (WAV <-> FLAC)**: Multi-channel 96kHz/24-bit & 48kHz/16-bit PCM to FLAC encoding and decoding (`ragdoll transcode` vs `ffmpeg -c:a flac`).
5. **Fast Frame Seeking & Contact Sheet Extraction**: Extracting a 4x4 thumbnail grid across a video stream (`ragdoll extract --thumbnails 16 --grid 4x4` vs `ffmpeg -vf fps=...,tile=4x4`).
6. **4K Perceptual Quality Metrics (PSNR + SSIM)**: Full-plane PSNR and 8x8 windowed SSIM calculation (`ragdoll metrics` vs `ffmpeg -lavfi psnr,ssim -f null -`).

## Target Metrics vs FFmpeg

- Measure median wall-clock time (ms), throughput (fps / GB/s), speedup multiplier (`Ragdoll vs FFmpeg`), and peak resident memory (RSS MB).
- Export a comprehensive Markdown & JSON benchmark report and record the exact Apple M4 numbers in `## Result`.

## Acceptance criteria

- [ ] Deterministic test media synthesis and head-to-head execution in `ragdoll bench --compare-ffmpeg`.
- [ ] Measured wall-clock time, throughput (GB/s or fps), and peak RSS recorded in `## Result`.
