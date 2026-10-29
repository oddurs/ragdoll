---
id: 89
uid: 2eed7656-dbba-4877-9bc2-fd321e8cb690
title: H.264 encoder slower than x264 at equal compression (0.12x ultrafast)
type: kernel
status: doing
milestone: m11-validation
assignee: Claude
claimed: 2026-10-28
created: 2026-10-28
updated: 2026-10-29
priority: p1
---

## Mathematical Specification

## Register & Cache Layout (NEON / AVX2)

## Acceptance criteria

- [x] Scalar reference and NEON/AVX2 SIMD implementations pass differential bit-exactness tests across odd dimensions.
- [x] Zero heap allocations in the hot loop.

## 2026-10-29

Round 1 (branch worktree-agent-a35061fd031e6cd15). New frame coder in crates/ragdoll-codec/src/h264/enc.rs and encoder kernels in crates/ragdoll-simd/src/h264_enc.rs (AVX2 + NEON + scalar refs, differential tests).

Measured single-thread, rbench h264-enc vs avbench libx264 (threads=1), 720p testsrc2 150 frames (perf/in/src720.y4m), keyint 30 (ours default), QP 26:
- before: 2.56-2.93 s, 3,256,020 B, Y 46.75 dB; 30-frame callgrind 8.31G instructions.
- after: ~0.27-0.28 s best-of-5 (load ~2), 2,829,687 B, Y 47.19 / avg 46.13 dB; encoder 2.67G instructions for the full clip (30 frames: 514M).
- x264 ultrafast QP 26: ~0.24-0.25 s, 2,846,544 B, Y 46.28 dB; 2.32G instructions incl. harness (~2.0-2.1G encoder). QP 25: 3,119,671 B, Y 47.48.
- So at equal size we are +0.9 dB Y (about 8% fewer bits at equal Y PSNR) but still ~10% slower wall clock and ~1.3x instructions. Not yet faster: do not close.
- Natural content (Baseline, QP 30): foreman CIF 260,714 B Y 35.16 in ~77 ms vs ultrafast QP 28 524,230 B Y 35.11 in ~74 ms; park 720p50 3,343,576 B Y 33.18 in ~386 ms vs ultrafast QP 29 4,940,988 B Y 33.36 in ~245 ms (4.36G vs 1.90G instructions). Against superfast/veryfast (B-frames, CABAC) we are 1.4-3x faster but ~16% larger at equal PSNR.

Where the time goes now (30-frame src720, 514M): deblocking ~78M (the decoder's h264_luma/chroma_deblock kernels are ~46M of it and were left alone), CAVLC ~70M, sub-pel ~60M, IDR frames ~69M each (x264 ultrafast's IDR is ~15M: I16 only, no deblock). On park nearly every MB is coded: per P frame ~65M (sub-pel 13M, deblock 11M, CAVLC 8M).

Left to try: a faster vertical-edge deblock (16x16 transpose per MB) once the decoder kernels settle; x264-style CAVLC runlevel extraction; frame-level half-pel planes when most MBs are coded; cheaper chroma intra analysis; an explicit speed preset (no qpel second iteration / no I4 in IDR) if a faster operating point is wanted. Disabling deblocking would beat ultrafast on src720 (2,844,728 B Y 46.90) but costs 3-8% BD-rate on natural clips, so it stays on.

ragdoll-pipeline malformed_bitstream_zero_panic_corpus fails with the new encoder's seed stream because of a decoder overflow (BUG-0097); the decoder was not touched here.
