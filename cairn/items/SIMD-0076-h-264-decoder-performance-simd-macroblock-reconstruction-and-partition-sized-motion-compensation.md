---
id: 76
uid: 753e8266-7387-458c-bed6-228b0b080436
title: 'H.264 decoder performance: SIMD macroblock reconstruction and partition-sized motion compensation'
type: kernel
status: backlog
milestone: m11-validation
depends_on:
- 75
created: 2026-10-28
updated: 2026-10-28
priority: p1
effort: l
area: codec-video
---

## Problem

The reference-grade H.264 decoder (`0075`) is bit-exact but slow: on x86_64 it decodes 1280x720 x264 `-preset medium` at about 53 fps and Ragdoll-encoded Baseline at about 90 fps, single-threaded, while FFmpeg decodes the same streams at several hundred fps per core. Correctness came first; nothing in the new core is vectorised yet.

Known costs:
- Every inter block, including 4x4 partitions, runs the full 16x16 quarter-pel kernel, which on x86_64 is the scalar path with per-sample clamping.
- Intra prediction, inverse transforms, dequantisation and deblocking are scalar `i32` code.
- CAVLC `coeff_token`, `total_zeros` and `run_before` use linear VLC matching.
- Per-slice reference lists clone `Arc`s and per-picture state is reallocated for every picture.

## Architecture & SIMD Design

Partition-sized luma and chroma MC kernels with edge emulation only at picture borders, AVX2 and NEON versions differential-tested against the scalar reference; SIMD 4x4/8x8 inverse transforms in the spec's row-then-column order; vectorised deblocking for luma and chroma edges with per-segment bS; table-driven CAVLC; per-picture state reuse through `FramePool`. Every change must keep `tests/h264_reference.rs` and the conformance corpus byte-identical.

## Acceptance criteria

- [ ] 720p x264 `-preset medium` decodes at 300 fps or more single-threaded on x86_64 AVX2 and on Apple M-series NEON, measured by the benchmark suite.
- [ ] Every SIMD kernel has a differential test against its scalar reference.
- [ ] The x264 matrix and all supported conformance streams stay byte-identical with JM.

## 2026-10-28

2026-10-28 head-to-head (Xeon 2.1 GHz VM, AVX2, single-threaded, in process; libavcodec 6.1.1 called directly with packets pre-demuxed, best of 3): 1080p x264 medium High/CABAC at 6.2 Mbit/s decodes at 29 fps against FFmpeg's 181 fps (505 fps with 4 frame threads), 6.3x slower; 720p x264 Baseline/CAVLC at 3.1 Mbit/s at 77 fps against 834 fps, 11x slower. 1080p30 is just under real time. Harness: rbench (Ragdoll), avbench (libavcodec) in the session scratchpad; the inputs are testsrc2 with light temporal noise.
