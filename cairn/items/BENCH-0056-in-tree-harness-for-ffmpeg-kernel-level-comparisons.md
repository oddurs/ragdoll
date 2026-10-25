---
id: 56
uid: 37bc0ff1-6071-4f9d-ad73-28827745527c
title: In-tree harness for FFmpeg kernel-level comparisons
type: benchmark
status: backlog
milestone: m10-production-v1
created: 2026-10-25
updated: 2026-10-25
priority: p2
effort: m
area: bench
---

## Workload & Fair Comparison Protocol

The kernel table in the README and on the docs site (start-code scan, CRC-32, Lanczos-3, RBSP unescape, SSIM, DCT, RGB->YUV, SAD/SSE, H-scale) compares Ragdoll against FFmpeg 9.0.2 internal kernels, but the FFmpeg side was measured outside the repository during SIMD-0024 / SIMD-0027 / SIMD-0029. Nobody can reproduce that column from a clean checkout. Add a harness that links or loads the FFmpeg routines (or runs FFmpeg checkasm --bench) on the same inputs.

## Target Metrics vs FFmpeg

Same kernels, same 4K planes and 16 MB streams, median of N runs, single thread.

## Acceptance criteria

- [ ] Deterministic test media synthesis and head-to-head execution in `ragdoll bench --compare-ffmpeg`.
- [ ] Measured wall-clock time, throughput (GB/s or fps), and peak RSS recorded in `## Result`.
- [ ] The provenance note on the docs benchmarks page is updated to point at the harness.
