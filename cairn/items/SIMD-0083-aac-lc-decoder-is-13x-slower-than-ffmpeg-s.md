---
id: 83
uid: 347f6db3-6bca-4b44-afc1-1a39a914e752
title: AAC-LC decoder is 13x slower than FFmpeg's
type: kernel
status: backlog
milestone: m11-validation
created: 2026-10-28
updated: 2026-10-28
priority: p2
effort: m
area: codec-audio
---

## Problem

The AAC-LC decoder (BUG-0079) is conformant but slow. On 2026-10-28 (Xeon 2.1 GHz VM, AVX2, single-threaded, in process; libavcodec 6.1.1 called directly on pre-demuxed packets, best of 5), 60 s of 128 kbit/s stereo 48 kHz decodes in 341 ms (176x real time) against 25 ms for FFmpeg's float decoder (2367x) and 68 ms for `aac_fixed`: 13x slower than FFmpeg.

A callgrind profile puts the cost in:
- `imdct`: twiddle factors are recomputed with `sin`/`cos` on every call (about 27 % of all instructions in `sincos`, plus the IMDCT itself and its zeroing).
- Huffman decoding reads one bit at a time and looks each prefix up in a SipHash `HashMap`.
- Dequantisation calls `powf(4/3)` per coefficient and `powf` per band.

## Acceptance criteria

- [ ] Precomputed IMDCT twiddles and windows, table-driven Huffman decoding (multi-bit lookup), and a `x^(4/3)` table for |x| < 8192.
- [ ] 60 s of 128 kbit/s stereo decodes at least 1000x real time single-threaded on x86_64 AVX2.
- [ ] `tests/aac_reference.rs` passes unchanged and the FFmpeg/fdk-aac conformance sweep stays within the ISO/IEC 14496-26 bounds.
