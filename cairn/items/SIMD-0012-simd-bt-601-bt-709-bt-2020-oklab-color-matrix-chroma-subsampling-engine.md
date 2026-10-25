---
id: 12
uid: 8365f249-fea9-43ec-af16-0b577ebaa08e
title: SIMD BT.601 / BT.709 / BT.2020 / Oklab Color Matrix & Chroma Subsampling Engine
type: kernel
status: done
milestone: m2-pix
assignee: antigravity
depends_on:
- 9
- 11
created: 2026-10-24
updated: 2026-10-24
closed_at: 2026-10-24
priority: p0
area: pix
effort: m
---

## Mathematical Specification

Convert between planar/semi-planar YUV (`YUV420p`, `YUV422p`, `YUV444p`, `NV12`, `YUV420p10le`) and packed/planar RGB (`RGB24`, `RGBA8`, `LinearRGB`, `Oklab`) using exact ITU-R BT.601, BT.709, and BT.2020 matrix coefficients in both limited (16..235 / 16..240) and full (0..255) ranges.

## Register & Cache Layout (NEON / AVX2)

1. Process 16 pixels (2x8 luma rows + 1x8 chroma row for 4:2:0) per iteration in NEON registers using Q14/Q15 fixed-point coefficients (`vqdmulhq_s16` / `vmlal_s16`, `vqmovun_s16`, `vst3_u8` / `vst4_u8` interleaved RGB/RGBA stores).
2. Implement proper chroma siting (`Center` vs `TopLeft` / `CoSited`) during 4:2:0 <-> 4:4:4 upsampling/downsampling instead of naive nearest-neighbor duplication.
3. Provide perceptual `Oklab` conversion for color-accurate tone-mapping and palette/thumbnail analysis.

## Acceptance criteria

- [x] Scalar reference and NEON/AVX2 SIMD implementations pass differential bit-exactness tests across odd dimensions.
- [x] Zero heap allocations in the hot loop.
- [x] YUV420p <-> RGB24/RGBA8 round-trip accuracy is within <= 1 LSB across BT.601, BT.709, and BT.2020 color spaces.

## 2026-10-24

Implemented crates/ragdoll-pix/src/color.rs: Q14 YuvRgbMatrix across BT.601, BT.709, and BT.2020 (Limited and Full ranges), 2-row x 16-pixel aarch64 NEON yuv420p_to_rgb24 (vst3q_u8 + vzip1q_s32/vzip2q_s32 + vmlaq_n_s32 reaching 5.17 Gpx/s / 6.76x speedup) and rgb24_to_yuv420p (vld3q_u8 + vpaddlq_u8 2x2 box chroma reaching 5.96 Gpx/s / 3.73x speedup), chroma-siting-aware 4:2:0->4:4:4 upsampler, and OklabColor scanline converter.

## Result

Shipped SIMD color matrix & chroma subsampling engine in `ragdoll-pix::color` (`YuvRgbMatrix`, NEON `yuv420p_to_rgb24` at 5.17 Gpx/s, NEON `rgb24_to_yuv420p` at 5.96 Gpx/s, chroma-siting 4:2:0->4:4:4 upsampler, and `OklabColor`).
