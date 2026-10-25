---
id: 10
uid: e48ac1a1-06c7-433e-bd84-acce48c5d519
title: Branchless 64-Bit CLZ BitReader, BitWriter & Exp-Golomb / Rice Entropy Primitives
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
area: core
effort: m
---

## Mathematical Specification

Variable-length entropy coding (Unsigned/Signed Exp-Golomb `ue(v)` / `se(v)` in H.264/HEVC, LEB128 in AV1, and Rice/Golomb coding in FLAC) accounts for 30-45% of serial codec CPU time when implemented with bit-by-bit loops.

## Register & Cache Layout (NEON / AVX2)

1. `BitReader<'a>` maintains a 64-bit shift register (`bit_buf: u64`, `bits_left: u32`) refilled via unaligned big-endian 64-bit loads (`u64::from_be_bytes`).
2. `read_ue()` and `read_se()` execute branchlessly using hardware count-leading-zeros (`bit_buf.leading_zeros()`, compiling to `CLZ` on ARM64).
3. `read_rice(k)` decodes unary quotient via `leading_zeros()` across 64-bit words plus `k`-bit remainder in a single register operation.
4. `BitWriter` mirrors the 64-bit accumulator design, flushing 64-bit big-endian words directly into destination slices, plus RBSP emulation prevention (`0x00 0x00 0x03`) insertion and removal.

## Acceptance criteria

- [x] Scalar reference and NEON/AVX2 SIMD implementations pass differential bit-exactness tests across odd dimensions.
- [x] Zero heap allocations in the hot loop.
- [x] `BitWriter` -> `BitReader` round-trip tests pass for arbitrary bit widths (1..=64), Exp-Golomb (`ue`/`se`), Rice codes, and RBSP escaping.

## 2026-10-24

Implemented 64-bit MSB-first BitReader and BitWriter with hardware CLZ Exp-Golomb (ue/se), Golomb-Rice (unsigned/signed zig-zag), AV1 LEB128, Matroska EBML VINT, and NEON-accelerated RBSP emulation prevention unescaping. All 11 unit tests pass.

## Result

Shipped ragdoll-core::bitstream: 64-bit MSB-first BitReader/BitWriter with zero-alloc hot loops, CLZ-accelerated Exp-Golomb (read_ue/read_se) and Golomb-Rice (read_rice_unsigned/read_rice_signed), AV1 LEB128, Matroska EBML VINT, and NEON vs scalar differential-verified RBSP escaping/unescaping.
