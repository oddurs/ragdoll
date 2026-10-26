---
id: 67
uid: 7018d558-4948-4b16-9d54-baf32f6384eb
title: Rate-distortion regression tracker with BD-rate and libvmaf calibration
type: benchmark
status: backlog
milestone: m11-validation
depends_on:
- 62
- 64
created: 2026-10-25
updated: 2026-10-25
priority: p2
effort: m
area: bench
---

## Workload & Fair Comparison Protocol

A fixed clip set (synthetic and real-world, several resolutions and content types) encoded at a ladder of QPs or CRF values. Quality is measured with libvmaf (VMAF, PSNR, SSIM) on frames decoded by FFmpeg, not by our own decoder.

## Target Metrics vs FFmpeg

BD-rate against the previous commit and against x264 at matched preset and profile. VMAF-Lite is calibrated against libvmaf and its correlation is reported.

## Acceptance criteria

- [ ] Deterministic test media synthesis and head-to-head execution in `ragdoll bench --compare-ffmpeg`.
- [ ] Measured wall-clock time, throughput (GB/s or fps), and peak RSS recorded in `## Result`.
- [ ] A BD-rate regression beyond a configured tolerance fails the nightly run.
