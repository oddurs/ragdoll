---
id: 41
uid: adef72af-42f0-4501-973b-8695a9b94f9c
title: Perceptual CRF (--crf), Variance-Adaptive Quantization (VAQ) & VBV Bitrate Control
type: feature
status: planned
milestone: m7-video-production
depends_on:
- 39
created: 2026-10-25
updated: 2026-10-25
priority: p1
area: codec-video
effort: l
---

## Problem

Fixed-QP encoding (`--qp`) wastes bits on high-motion or complex-texture macroblocks where the human eye masks distortion and starves flat/dark gradients where banding is obvious. Production users expect `--crf <18..28>` (Constant Rate Factor) and `--bitrate <kbps> --maxrate <kbps>` (VBV-constrained ABR).

## Architecture & SIMD Design

1. Implement `x264`-inspired **Variance-Adaptive Quantization (VAQ)** using our SIMD plane variance kernel (`ssim_4x4xn_u8` / AC energy): compute per-macroblock log-variance $\sigma^2_{\text{MB}}$ and modulate `mb_qp_delta` by $\Delta \text{QP} = \text{round}(k \cdot \log_2(\sigma^2_{\text{MB}} / \bar{\sigma}^2))$.
2. Implement frame-level **CRF & VBV Rate Control** using SATD (`satd_8x8_u8`) lookahead frame complexity estimation to set `slice_qp` dynamically per frame while respecting a Virtual Buffer Verifier (VBV) fullness constraint.
3. Expose `--crf <f32>`, `--bitrate <RATE>`, `--maxrate <RATE>`, and `--preset ultrafast|fast|medium|quality` in `ragdoll transcode`.

## Acceptance criteria

- [ ] `--crf 23` achieves `>= 1.0 dB` higher 5-Scale MS-SSIM / VMAF-Lite at the same average bitrate compared to fixed `--qp`.
- [ ] `--bitrate 2M --maxrate 3M` constrains output bitrate within `+-5%` of target on 1080p video streams.
