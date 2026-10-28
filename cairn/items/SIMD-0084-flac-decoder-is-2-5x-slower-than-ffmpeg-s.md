---
id: 84
uid: abddebad-4386-4961-bad6-f96b3ca95391
title: FLAC decoder is 2.5x slower than FFmpeg's
type: kernel
status: done
milestone: m11-validation
assignee: Claude
created: 2026-10-28
updated: 2026-10-28
closed_at: 2026-10-28
priority: p2
effort: s
area: codec-audio
---

## Problem

On 2026-10-28 (same setup as SIMD-0083), 60 s of 16-bit stereo 44.1 kHz FLAC (FFmpeg level 5) decodes in 140 ms (428x real time) against 57 ms for libavcodec's FLAC decoder (1055x): 2.5x slower.

A callgrind profile:
- `reconstruct_lpc_subframe`: about 27 % (scalar, bounds-checked inner loop).
- `BitReader::read_bits` and `read_unary_zeros` in the Rice residual loop: about 25 %.
- `md5_digest` over the decoded PCM: about 17 %. FFmpeg does not verify the STREAMINFO MD5 by default, so part of the gap is work FFmpeg skips.

## Acceptance criteria

- [x] MD5 verification is optional (on by default for `ragdoll` integrity checks, off in the decode hot path or computed incrementally).
- [x] Order-specialised or SIMD LPC reconstruction and a word-at-a-time Rice decoder, with differential tests against the scalar path.
- [x] Decode speed within 1.3x of libavcodec on the same input, with MD5 off; output stays bit-exact.

## 2026-10-28

Fused Rice + predictor decode: residuals are restored as soon as they are read, with order-specialised (1..=12) loops unrolled over a compile-time ring of the last N samples so history stays in registers; Fixed orders reuse the same loop as LPC with constant coefficients. Narrow 32-bit sums when sum(|c|) * 2^(bps-1) < 2^31, checked afterwards by a vectorised range pass with an exact 64-bit redo for non-conforming streams, so output matches reconstruct_lpc_subframe on every input. The subframe decoder is multiversioned with BMI2/LZCNT at the AVX2 dispatch level.

## 2026-10-28

MD5: decode_stream still verifies (the lib.rs integrity test requires it, and the pipeline relies on it), now incrementally per frame straight from the S16/S24/S32 output bytes with an unrolled MD5 (about 18 ms per 60 s, OpenSSL asm needs about 15.5 ms). decode_stream_unverified skips it, matching FFmpeg's default. Frame CRC-16 uses PCLMULQDQ/PMULL folding (crc16_flac_update_scalar kept as reference).

## 2026-10-28

Result on the 60 s level-5 file, best of 8 rounds x 7 runs back to back: decode_stream 36.6 ms, decode_stream_unverified 22.2 ms, FFmpeg 45.0-46.0 ms (was 139 ms). Callgrind: 466M / 319M instructions vs 1585M before and 589M for avbench. Left on the table: the order-8 loop is bound by the single scalar imul port (8 multiplies per sample) plus spills of the ring history; MD5 is a serial chain and now the largest cost in the verified path.

## Result

FLAC decode 139 ms -> 36.6 ms with MD5 (22.2 ms without, decode_stream_unverified) per 60 s 16-bit stereo vs FFmpeg 45 ms: 64-bit Rice cache, fused order-specialised LPC restore with 32-bit sums when safe, BMI2 multiversioning, AVX2/NEON decorrelate+interleave, faster MD5, CLMUL CRC-16.
