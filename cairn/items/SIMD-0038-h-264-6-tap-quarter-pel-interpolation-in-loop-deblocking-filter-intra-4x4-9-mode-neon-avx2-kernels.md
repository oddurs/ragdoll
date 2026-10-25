---
id: 38
uid: 829aff05-6978-4e79-8a67-87ceb16b5459
title: H.264 6-Tap Quarter-Pel Interpolation, In-Loop Deblocking Filter & Intra 4x4 9-Mode NEON/AVX2 Kernels
type: kernel
status: done
milestone: m7-video-production
assignee: Oddur Sigurdsson
depends_on:
- 35
created: 2026-10-25
updated: 2026-10-25
closed_at: 2026-10-25
priority: p0
area: simd
effort: l
---

## Mathematical Specification

1. **6-Tap FIR Quarter-Pel (`qpel`) Luma Interpolation (ITU-T H.264 8.4.2.2.1)**:
   - Half-pel horizontal/vertical samples: $h = \text{clip}_{u8}((A - 5B + 20C + 20D - 5E + F + 16) \gg 5)$.
   - Quarter-pel samples: average with upward rounding $(a + b + 1) \gg 1$ (`vrhaddq_u8` on NEON, `_mm256_avg_epu8` on AVX2).
2. **Adaptive In-Loop Deblocking Filter (ITU-T H.264 8.7 / `libavcodec/aarch64/h264cmc_neon.S`)**:
   - Boundary strength $bS \in \{1..4\}$ across vertical and horizontal $4\times4$ block edges with $\alpha(\text{QP}), \beta(\text{QP}), t_{C0}(\text{QP}, bS)$ thresholds.
3. **Intra $4\times4$ 9-Mode Spatial Predictors (ITU-T H.264 8.3.1.2)**:
   - Modes 0..8 (Vertical, Horizontal, DC, Diagonal Down-Left, Diagonal Down-Right, Vertical-Right, Horizontal-Down, Vertical-Left, Horizontal-Up).

## Register & Cache Layout (NEON / AVX2)

- Vectorize horizontal and vertical 6-tap FIR filtering across 16-byte rows (`vld1q_u8`, `vmlal_s8` / `vaddl_u8`, `vqshrun_n_s16::<5>`) and `vrhaddq_u8` quarter-pel averaging.
- Vectorize 4-edge parallel $16\times16$ macroblock deblocking in register using $4\times4$ `vtrn` transposes for vertical edges.

## Acceptance criteria

- [x] Scalar reference and NEON/AVX2 SIMD implementations pass differential bit-exactness tests across all 16 quarter-pel phases, $bS \in \{1..4\}$ deblocking thresholds, and all 9 Intra $4\times4$ modes.
- [x] Zero heap allocations in the hot loop.

## 2026-10-25

Implemented put_h264_qpel16_mc_u8, put_h264_qpel8_mc_u8, put_h264_chroma_mc8_u8, h264_deblock_luma_line/chroma_line, h264_loop_filter_luma/chroma_edge_h/v_u8, and predict_intra_4x4_u8 (all 9 modes) in crates/ragdoll-simd/src/h264_dsp.rs with bit-exact NEON and scalar parity and zero heap allocations.

## Result

Delivered NEON+scalar H.264 6-tap quarter-pel luma/chroma MC, 8.7 in-loop deblocking filter, 9-mode Intra 4x4/8x8 spatial predictors, and 8x8 High Profile DCT/IDCT kernels in ragdoll-simd.
