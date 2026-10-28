---
id: 84
uid: abddebad-4386-4961-bad6-f96b3ca95391
title: FLAC decoder is 2.5x slower than FFmpeg's
type: kernel
status: doing
milestone: m11-validation
assignee: Claude
claimed: 2026-10-28
created: 2026-10-28
updated: 2026-10-28
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

- [ ] MD5 verification is optional (on by default for `ragdoll` integrity checks, off in the decode hot path or computed incrementally).
- [ ] Order-specialised or SIMD LPC reconstruction and a word-at-a-time Rice decoder, with differential tests against the scalar path.
- [ ] Decode speed within 1.3x of libavcodec on the same input, with MD5 off; output stays bit-exact.
