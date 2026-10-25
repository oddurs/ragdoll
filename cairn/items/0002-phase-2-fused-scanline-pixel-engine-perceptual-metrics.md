---
id: 2
uid: 0249ce19-bd82-4837-a25f-cf1b111916d9
key: m2-pix
title: Phase 2 — Fused Scanline Pixel Engine & Perceptual Metrics
type: milestone
status: planned
created: 2026-10-24
updated: 2026-10-24
priority: p0
area: pix
---

Replace libswscale's multi-pass full-frame memory round-trips with a fused L1-cache-resident scanline pixel engine supporting BT.601/BT.709/BT.2020/Oklab color science, separable polyphase Lanczos-3 and Mitchell-Netravali scaling, and SIMD-accelerated PSNR, SSIM, and VMAF-lite perceptual quality analyzers.

## Release gate

Fused 4K-to-1080p YUV420p Lanczos-3 downscaling + RGB conversion and SIMD SSIM/PSNR pass differential accuracy checks against scalar references and outperform FFmpeg's swscale/lavfi in memory bandwidth.
