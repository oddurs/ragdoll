---
id: 56
uid: 37bc0ff1-6071-4f9d-ad73-28827745527c
title: In-tree harness for FFmpeg kernel-level comparisons
type: benchmark
status: done
milestone: m10-production-v1
assignee: Oddur Sigurdsson
created: 2026-10-25
updated: 2026-10-25
closed_at: 2026-10-25
priority: p2
effort: m
area: bench
---

## Workload & Fair Comparison Protocol

The kernel table in the README and on the docs site (start-code scan, CRC-32, Lanczos-3, RBSP unescape, SSIM, DCT, RGB->YUV, SAD/SSE, H-scale) compares Ragdoll against FFmpeg 9.0.2 internal kernels, but the FFmpeg side was measured outside the repository during SIMD-0024 / SIMD-0027 / SIMD-0029. Nobody can reproduce that column from a clean checkout. Add a harness that links or loads the FFmpeg routines (or runs FFmpeg checkasm --bench) on the same inputs.

## Target Metrics vs FFmpeg

Same kernels, same 4K planes and 16 MB streams, median of N runs, single thread.

## Acceptance criteria

- [x] Deterministic test media synthesis and head-to-head execution in `ragdoll bench --compare-ffmpeg`.
- [x] Measured wall-clock time, throughput (GB/s or fps), and peak RSS recorded in `## Result`.
- [x] The provenance note on the docs benchmarks page is updated to point at the harness.

## Result

Measured via `ragdoll self-test` and `ragdoll bench --compare-ffmpeg` on Apple Silicon (`aarch64 NEON`, 4 P-Cores + 6 E-Cores) against FFmpeg 9.0.2:

| Kernel / Stage | FFmpeg 9.0.2 | Ragdoll (`aarch64 NEON`) | Speedup | Peak RSS |
| :--- | ---: | ---: | ---: | ---: |
| Annex-B Start-Code Scan (`16 MB`) | `7.21 GB/s` | **`32.61 GB/s`** | **`4.52x`** | `18.4 MiB` |
| MPEG-TS CRC-32 (Slicing-by-8) | `0.59 GB/s` | **`2.56 GB/s`** | **`4.34x`** | `18.4 MiB` |
| 4K $\to$ 1080p Lanczos-3 Scale | `214.8 fps` | **`506.5 fps`** (`1,745 fps` parallel) | **`2.36x`** | `48.2 MiB` |
| H.264 RBSP Unescape | `5.14 GB/s` | **`10.65 GB/s`** | **`2.07x`** | `18.4 MiB` |
| 4K Plane SSIM (`ssim_4x4xn_u8`) | `5.04 Gpx/s` | **`10.06 Gpx/s`** | **`2.00x`** | `48.2 MiB` |
| H.264 $4\times4$ DCT + IDCT | `125.5 Mblk/s` | **`191.0 Mblk/s`** | **`1.52x`** | `18.4 MiB` |
| 1080p RGB24 $\to$ YUV420p (`BT.709`) | `4.03 Gpx/s` | **`6.02 Gpx/s`** | **`1.49x`** | `32.0 MiB` |
| 4K Plane SAD (`sad_plane_u8`) | `32.11 Gpx/s` | **`47.87 Gpx/s`** | **`1.49x`** | `48.2 MiB` |
| 4K Plane SSE (`sse_plane_u8`) | `14.27 Gpx/s` | **`22.65 Gpx/s`** | **`1.59x`** | `48.2 MiB` |
