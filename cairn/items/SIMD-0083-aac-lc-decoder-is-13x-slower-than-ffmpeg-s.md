---
id: 83
uid: 347f6db3-6bca-4b44-afc1-1a39a914e752
title: AAC-LC decoder is 13x slower than FFmpeg's
type: kernel
status: doing
milestone: m11-validation
assignee: Claude
claimed: 2026-10-28
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

- [x] Precomputed IMDCT twiddles and windows, table-driven Huffman decoding (multi-bit lookup), and a `x^(4/3)` table for |x| < 8192.
- [x] 60 s of 128 kbit/s stereo decodes at least 1000x real time single-threaded on x86_64 AVX2.
- [x] `tests/aac_reference.rs` passes unchanged and the FFmpeg/fdk-aac conformance sweep stays within the ISO/IEC 14496-26 bounds.

## 2026-10-28

Rewrote the decoder's hot paths in f32 (FFmpeg's float decoder is f32 too). IMDCT: new ragdoll_simd::dct4 (Dct4Plan: DCT-IV over an M/2-point complex FFT, input gathered straight into bit-reversed order, radix-8 first pass, radix-2 stages fused in pairs on SIMD, every twiddle precomputed with 1/(M*32768) folded into the post-twiddle) and imdct_window (TDAC windowing + overlap-add from the DCT-IV output, used for long and the eight chained short blocks); AVX2 and NEON paths do the scalar reference's operations in the same order (no FMA), bit-exact, with differential tests. Windows precomputed per shape as whole 1024-sample halves so ONLY_LONG/LONG_START/LONG_STOP are one multiply. Huffman: two-level tables (9-bit first level, complete codes asserted) whose unsigned-codebook codewords include their sign bits, payload = row of a dequantised [f32;4]/[f32;2] table, so a codeword costs one lookup and one vector multiply by 2^((sf-100)/4) from a table; escapes via x^(4/3) table (8256 entries). The spectral loop reads through a register bit cache with 32-bit refills from codeword-independent addresses, which shortens the per-codeword dependency chain (worth ~25% wall clock on its own). Payload copied into a 16 KiB zero-padded buffer read with masked indices, overruns checked per element. Pulses (no encoder at hand writes them) are applied to the dequantised value by recovering q exactly; unit-tested over every reachable q and a synthetic pulse stream matches the old decoder within 1e-2x of the bound. TNS LPC now derived in f32 as FFmpeg does, which brings fdk-aac streams to 7e-3x of the max bound (old f64 path: 0.11x). Results, Xeon VM, 60 s 128k stereo 48 kHz, best of 30 back to back: 15.5 ms vs avbench 19.2 ms (was 341 ms, 3870x real time); callgrind 190M instructions whole process vs avbench 295M (decode alone ~189M vs avcodec_send_packet+receive_frame 212M), was 5079M. Conformance sweep (204 streams: FFmpeg- and fdk-aac-encoded, mono/stereo/5.1, 8-96 kHz, 32-976 kbit/s, forced M/S, intensity, TNS, PNS, fast/twoloop coders) against FFmpeg 6.1's float decoder: no frame errors, worst 4.6e-3x RMS and 7.3e-3x max of the ISO/IEC 14496-26 bounds; the only two streams outside are FFmpeg-encoded with PNS at 88.2 kHz, identical with the old decoder and within bounds with -aac_pns 0. tests/aac_reference.rs unchanged: 0.04-0.12 % of samples differ by one LSB (was 0.03-0.07 %, bound 1 %); 60 s file vs FFmpeg 0.096 % (was 0.070 %), max 0.009 LSB. Left: AudioFrame::new zeroes its 8 KiB buffer (13 % of instructions, little time); the stereo interleave and M/S are scalar; FMA and fusing the post-twiddle into the windowing would trim the IMDCT further.
