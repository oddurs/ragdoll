---
id: 13
uid: 5686bba4-61f2-4f06-a40c-d43ffb2d71f6
title: Fused Scanline Separable Polyphase Lanczos-3 & Mitchell-Netravali Resampler
type: kernel
status: planned
milestone: m2-pix
depends_on:
- 12
created: 2026-10-24
updated: 2026-10-24
priority: p0
area: pix
effort: l
---

## Mathematical Specification

Separable polyphase resampling with windowed sinc **Lanczos-3** (=3$), bicubic **Mitchell-Netravali** (=1/3, C=1/3$), and **Bilinear** kernels:
96256L(x) = \text{sinc}(x)\,\text{sinc}(x/3) \quad (|x| < 3)96256
Weights are normalized per output phase to sum to exactly ^{14} = 16384$ in signed 16-bit fixed-point (`i16`), eliminating DC brightness ripple.

## Register & Cache Layout (NEON / AVX2)

1. Precompute horizontal and vertical `PolyphaseFilterBank` tables once per resolution pair.
2. Execute a **fused L1-cache-resident rolling scanline ring buffer** (8-tap vertical window): each input row is horizontally filtered once into the ring buffer, and vertical NEON dot products (`vmlal_s16` + `vqrshrun_n_s32`) immediately emit finished output rows—eliminating full-frame intermediate allocations.
3. Support fused crop + scale + color conversion in a single pipeline pass.

## Acceptance criteria

- [ ] Scalar reference and NEON/AVX2 SIMD implementations pass differential bit-exactness tests across odd dimensions.
- [ ] Zero heap allocations in the hot loop.
- [ ] 4K (3840x2160) to 1080p (1920x1080) Lanczos-3 scaling preserves flat luma fields with zero DC drift and sharp frequency response.
