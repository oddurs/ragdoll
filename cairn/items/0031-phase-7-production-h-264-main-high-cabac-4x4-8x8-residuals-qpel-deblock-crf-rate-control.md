---
id: 31
uid: 8ad195ce-0f1f-4db3-b4a2-84b8c8668742
key: m7-video-production
title: Phase 7 — Production H.264 (Main/High CABAC, 4x4/8x8 Residuals, Qpel, Deblock & CRF Rate Control)
type: milestone
status: done
assignee: Oddur Sigurdsson
created: 2026-10-25
updated: 2026-10-25
closed_at: 2026-10-25
priority: p0
area: codec-video
---

Upgrade Ragdoll's pure-Rust H.264/AVC engine from Constrained Baseline skip/DC coding to full production Main and High Profile parity: quantized $4\times4$ and $8\times8$ AC/DC DCT residual coding (`I_4x4` all 9 spatial modes, `I_16x16`, `P_L0_16x16`, `P_8x8`), 6-tap FIR quarter-pel (`qpel`) motion interpolation, ITU-T H.264 8.7 adaptive in-loop deblocking filter, CABAC binary arithmetic coding, B-slice decoding, and perceptual CRF (`--crf`) + VBV bitrate control.

## Release gate

Real-world Main/High Profile CABAC + B-frame H.264 streams (iPhone, YouTube, OBS, x264) decode bit-accurately, and Ragdoll's H.264 encoder achieves competitive BD-Rate / PSNR / SSIM / VMAF-Lite under `--crf` and `--bitrate` without `I_PCM` fallback on natural video.

## 2026-10-25

Completed Milestone 7 (m7-video-production): SIMD-0038, 0039, 0040, and 0041 verified bit-exact and closed.

## Result

Shipped Milestone 7 (m7-video-production): SIMD-0038, 0039, 0040, and 0041 complete and verified.
