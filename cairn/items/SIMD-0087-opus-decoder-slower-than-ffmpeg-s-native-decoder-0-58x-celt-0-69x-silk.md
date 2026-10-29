---
id: 87
uid: 10ab7cba-0730-40c7-bd33-df18e93bb141
title: Opus decoder slower than FFmpeg's native decoder (0.58x CELT, 0.69x SILK)
type: kernel
status: done
milestone: m11-validation
assignee: Claude
created: 2026-10-28
updated: 2026-10-29
closed_at: 2026-10-29
priority: p1
---

## Mathematical Specification

## Register & Cache Layout (NEON / AVX2)

## Acceptance criteria

- [x] Scalar reference and NEON/AVX2 SIMD implementations pass differential bit-exactness tests across odd dimensions.
- [x] Zero heap allocations in the hot loop.

## 2026-10-28

Round 1: AVX2/NEON kernels in opus/simd.rs (FFT, MDCT rotations/TDAC, comb filter, deemphasis, band ops, bits2pulses, SILK resampler and LPC), persistent CELT scratch, shorter exp_rotation chain. 60 s decode, wall clock best-of-N vs references measured back to back: CELT 96k 138-162 -> 79.4-81.5 ms (FFmpeg 79.8-86.4: a tie), SILK 50.0 -> 28.4 ms (FFmpeg 37.2), opus_rld 125.9 -> 76.0 ms (fixed libopus 107.7). Bit-exact with fixed-point libopus (RFC 8251 vectors, 667-configuration golden hashes). Encoder output unchanged, 17% fewer instructions. Open: CELT needs a clear margin (cwrsi mispredicts, FFT multiplies).

## 2026-10-29

Opus decode is allocation-free per packet: the range decoder reuses one packet buffer per decoder (Ec::decoder_with), packet parsing uses a fixed frame table, and the SILK/redundancy/transition/PLC scratch (CELT pitch search, LPC, LTP, CNG) lives in bounded stack arrays. tests/opus_no_alloc.rs decodes every RFC 8251 vector twice at 48k/2ch and 16k/1ch with every 7th packet lost and asserts zero heap allocations on the second pass.
