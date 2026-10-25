---
id: 51
uid: 827cc274-fdac-42f2-836f-71abafe04c94
title: '12-Workload Production v1.0 Benchmark Suite: Ragdoll vs. FFmpeg 9.0.2 (Speed, Quality & Battery Efficiency)'
type: benchmark
status: planned
milestone: m10-production-v1
depends_on:
- 41
- 43
- 48
- 49
- 50
created: 2026-10-25
updated: 2026-10-25
priority: p0
area: bench
effort: m
---

## Workload & Fair Comparison Protocol

Expand `ragdoll bench --compare-ffmpeg` from 7 workloads to a comprehensive **12-Workload Production `v1.0` Benchmark Suite** measured across **Ragdoll P-Core Mode**, **Ragdoll E-Core Battery-Saver Mode**, and **FFmpeg 9.0.2**:
1. Container Probe & GOP Index (`300f` 4K MP4)
2. Zero-Copy Multi-Track Remux (`MP4` -> `MKV`)
3. 4K -> 1080p Lanczos-3 Polyphase Scaling (`30f`)
4. Lossless FLAC Encode + Decode (`30s` 48 kHz Stereo)
5. $4\times4$ Contact Sheet Extraction to `.png`/`.jpg` (`16` Thumbs)
6. 4K Perceptual Quality Metrics (`PSNR + SSIM + MS-SSIM + VMAF-Lite`)
7. 1080p H.264 High/Main Profile CRF Video Transcode (`60f`)
8. Smart-GOP Frame-Accurate Trim (`ragdoll cut` vs `ffmpeg` re-encode)
9. Multi-Track A/V + Resampled Audio Transcode (`44.1 kHz` -> `48 kHz` + EBU R128 `-14 LUFS`)
10. Fused HDR10 (`BT.2020 PQ`) -> SDR (`BT.709`) Oklab Tonemap + Watermark Overlay
11. Multi-Rendition CMAF HLS/DASH ABR Ladder Packaging (`1080p + 720p + 480p`)
12. Bounded-Memory Streaming Transcode Footprint & E-Core Battery Efficiency Verification

## Target Metrics vs FFmpeg

- Outperform FFmpeg 9.0.2 in wall-clock speed on P-cores and achieve `5x–1,000x` lower energy consumption (`mJ`) in E-Core Battery-Saver mode (`QOS_CLASS_BACKGROUND`) across the production suite.

## Acceptance criteria

- [ ] Deterministic test media synthesis and head-to-head execution in `ragdoll bench --compare-ffmpeg`.
- [ ] Measured wall-clock time, throughput (GB/s or fps), hardware PMU instructions/cycles/energy (mJ), and peak RSS recorded in `## Result` and `README.md`.
