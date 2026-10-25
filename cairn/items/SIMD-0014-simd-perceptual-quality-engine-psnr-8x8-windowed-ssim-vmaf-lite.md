---
id: 14
uid: 0bd3db99-ca51-4051-b97a-58dd631db729
title: 'SIMD Perceptual Quality Engine: PSNR, 8x8 Windowed SSIM & VMAF-Lite'
type: kernel
status: planned
milestone: m2-pix
depends_on:
- 11
- 12
created: 2026-10-24
updated: 2026-10-24
priority: p1
area: pix
effort: m
---

## Mathematical Specification

Compute objective and perceptual video quality metrics across luma and chroma planes:
1. **PSNR**:  \log_{10}(\text{MAX}^2 / \text{MSE})$ from exact 64-bit accumulated SSE.
2. **SSIM**: Structural Similarity over 8x8 blocks with constants  = (0.01 L)^2, C_2 = (0.03 L)^2$:
   96256\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}96256
3. **VMAF-Lite**: Fuses multi-scale Visual Information Fidelity (VIF), Detail Loss Metric (DLM), and temporal motion difference into a 0..100 perceptual score.

## Register & Cache Layout (NEON / AVX2)

1. Compute $\sum x, \sum y, \sum x^2, \sum y^2, \sum xy$ simultaneously in a single pass over 8x8 blocks using NEON widening instructions (`vaddl_u8`, `vmull_u8`, `vmlal_u8`, `vaddvq_u32`).
2. Achieve > 1,000 fps on 1080p and > 300 fps on 4K streams on Apple M4.

## Acceptance criteria

- [ ] Scalar reference and NEON/AVX2 SIMD implementations pass differential bit-exactness tests across odd dimensions.
- [ ] Zero heap allocations in the hot loop.
- [ ] Identical frames report PSNR = inf and SSIM = 1.000000; synthetic degraded frames match analytical reference values.
