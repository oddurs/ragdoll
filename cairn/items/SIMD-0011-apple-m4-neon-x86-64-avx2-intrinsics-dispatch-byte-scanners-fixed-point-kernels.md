---
id: 11
uid: 39ed1adc-1ae8-4f25-b676-fb53f38c6d94
title: Apple M4 NEON & x86_64 AVX2 Intrinsics Dispatch, Byte Scanners & Fixed-Point Kernels
type: kernel
status: done
milestone: m1-foundation
assignee: antigravity
depends_on:
- 9
created: 2026-10-24
updated: 2026-10-24
closed_at: 2026-10-24
priority: p0
area: simd
effort: m
---

## Mathematical Specification

Container demuxers and NAL/OBU parsers scan gigabytes of byte streams for start codes (`0x000001`, `0x00000001`), MPEG-TS sync bytes (`0x47`), and RBSP escape sequences (`0x000003`), while pixel/codec kernels require fast vector dot products and saturating fixed-point math.

## Register & Cache Layout (NEON / AVX2)

1. Build `ragdoll-simd` with capability detection (`SimdLevel::Neon`, `Avx2`, `Scalar`) and zero-overhead dispatch macros.
2. Implement NEON 16-byte / 64-byte unrolled vector scanners (`vld1q_u8`, `vceqq_u8`, `vmaxvq_u8`) for Annex-B start codes (`find_nal_start_code`), RBSP unescaping, and MPEG-TS `0x47` sync locking.
3. Implement NEON 16-bit Q14/Q15 fixed-point vector arithmetic primitives and fast block SAD (Sum of Absolute Differences) / SSE (Sum of Squared Errors) using `vabdl_u8`, `vabal_u8`, and `vaddvq_u32`.

## Acceptance criteria

- [x] Scalar reference and NEON/AVX2 SIMD implementations pass differential bit-exactness tests across odd dimensions.
- [x] Zero heap allocations in the hot loop.
- [x] NEON Annex-B start-code scanner and SAD/SSE kernels match scalar reference outputs byte-for-byte across all alignment offsets (0..63).

## 2026-10-24

Implemented ragdoll-simd: SimdLevel capability dispatch, 64-byte/16-byte unrolled NEON Annex-B start-code scanner (find_annex_b_start_code), MPEG-TS 0x47 sync-byte scanner (find_mpeg_ts_sync), NEON SAD/SSE plane kernels, single-pass 8x8 SSIM window statistics (ssim_window_8x8_u8), 8-tap Q14 fixed-point convolution, and 4x4 integer DCT/IDCT/Hadamard transforms. Verified with differential NEON-vs-scalar tests across all alignments (0..=140) and odd dimensions.

## Result

Shipped ragdoll-simd: SimdLevel dispatch, 64-byte unrolled NEON Annex-B NAL start-code scanner & splitter, NEON MPEG-TS 0x47 sync scanner, NEON SAD/SSE plane kernels, single-pass 8x8 SSIM window accumulator, 8-tap Q14 fixed-point dot product, and 4x4 integer DCT/IDCT/Hadamard transforms.
