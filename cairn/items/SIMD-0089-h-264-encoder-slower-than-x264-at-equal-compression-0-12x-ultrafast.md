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
updated: 2026-10-30
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

## 2026-10-29

Round 2 (branch worktree-agent-a35061fd031e6cd15, commits ca169fb..0ce5e16 on top of eb274a0). New opt-in preset H264Encoder::with_speed(H264Speed::Fast): half-pel only (the more promising of two half-sample candidates, estimated by an average SAD first), intra 16x16 only (no plane, intra screened by SAD and asked to win by 16 lambda in P slices, chroma mode from the luma mode in P slices), no deblocking, P_Skip probed only next to a skipped macroblock. Balanced (default) output is bit-identical to round 1; Fast output changes were each measured (park and src720 within +-0.3% size at equal PSNR per step).

Bit-exact speedups shared by both presets: x86-64-v3 copy of the frame coder (runtime dispatch), lean SIMD dispatch wrappers, compile-time run_before table, packed CAVLC tables, macroblock nC cache, motion vector cost table, P_Skip predictor from the median predictor, 16-bit inverse transform, two-plane chroma MC kernel, 32-bit DC quantizers, in-place quantization, and saturating packs emitted with asm! (LLVM otherwise clamps both inputs before vpackssdw/vpackuswb: up to four extra instructions per pack). New kernels each have AVX2, NEON and scalar versions with differential tests: intra4x4_pred_dir, hpel_hv_16x16, hpel_16x16, chroma_mc_8x8_x2, intra_sad_x3_16x16.

Single thread, best-of-5 wall clock over 5 interleaved rounds (machine shared, load ~2.5; runs vary 15-25%): src720 QP26 (150 frames): Fast 198-245 ms, Balanced 251-313 ms, x264 ultrafast 238-315 ms. park 720p50 QP30 (60 frames): Fast 201-250 ms, Balanced 326-423 ms, x264 ultrafast 221-289 ms. Encoder instructions (callgrind): src720 Fast 1.38G, Balanced 2.14G (round 1: 2.67G), x264 2.22G; park Fast 1.60G, Balanced 3.43G (round 1: 4.36G), x264 1.75G.

Compression (Y PSNR, BD-rate against x264 ultrafast over 5 QPs): src720 Fast -1.8%, Balanced -7.6%; park Fast -17.2%, Balanced -31.4%. At QP30 on park Fast is 3,840,431 B / Y 32.87 dB vs ultrafast 4,415,543 B / 32.56 dB; on src720 Fast at QP26 is 2,936,895 B / 46.88 dB vs ultrafast 2,846,544 B / 46.28 dB (+0.30 dB at ultrafast's size by interpolation).

Left: per P frame on park we are now at x264's instruction count (~26M); remaining cost is CAVLC (~20%, mostly per-level work and the bit writer), forward transform + quantizer (~13%), integer motion search (~10%) and half-pel (~7%). I-cache misses are ~20% above x264's per P frame (the frame coder is one ~180 KB function); splitting cold paths out of the v3 copy is the next lever. The pipeline test malformed_bitstream_zero_panic_corpus still fails (BUG-0097, decoder).

## 2026-10-30

Round 3 (branch worktree-agent-a8d288051031e9368, commits 2b8e292 and bed6bf0 on top of 555b0e9): target was an operating point beating x264 --preset veryfast on BD-rate and on single-thread wall clock. New opt-in preset H264Speed::Compact; the default (Balanced) and Fast/Baseline output are byte-identical to before (qa corpus synth: 5 of 5 verified, nothing re-blessed).

Compact: P partitions 16x8/8x16/8x8, hexagon search range 16 seeded by the predictor, neighbours' and previous anchor's 16x16 results and the lookahead vectors, sub-pel from frame-level half-sample planes (predictor at sub-pel, half-pel diamond by SAD, quarter-pel diamonds by SATD), SA8D transform decision, intra 4x4 in P when intra 16x16 comes close; with CABAC up to three B pictures (middle one a reference: list modification + MMCO, max_num_reorder_frames in VUI), spatial direct, B_Skip, B partitions down to 8x8; half-resolution lookahead placing B pictures by path costs (slicetype_path_cost); QP offsets I -3 / Bref +2 / B +3. New kernels (AVX2 + NEON + scalar, differential tests): pixel_{sad,sad_avg,satd,avg,sad_x4}_wxh, pixel_sa8d_16x16, hpel_planes, downscale_2x_row.

BD-rate (piecewise cubic on log-rate vs PSNR, QP 20/24/28/32/36, keyint 250, FFmpeg psnr filter), Y / average PSNR:
- vs x264 veryfast: src720 -0.94% / -1.56%, park -1.04% / -2.18%, foreman -3.51% / -4.95%. (x264 with psy=0:aq-mode=0, i.e. --tune psnr, gives identical streams under -qp.)
- vs x264 ultrafast: src720 -32.0% / -31.5%, park -45.4% / -45.5%, foreman -69.8% / -70.0%.

Speed, QP 28, taskset one core, interleaved runs (machine noisy, +-10%): src720 at parity (x264/ours median ratio 0.999 and 1.016 in two runs of 9 and 7 rounds); park 1.24-1.33x slower than veryfast; foreman 1.17-1.39x slower. Instructions (callgrind, whole clip incl. input parsing): src720 9.886G vs 9.882G, park 11.74G vs 9.85G (1.19x), foreman 2.49G vs 2.13G (1.17x). x264 ultrafast is ~5x faster than Compact, so Compact stays an opt-in preset. Veryfast speed is NOT beaten on park and foreman.

Speed trade-offs taken (BD cost measured on all three clips): lookahead keeps zero vectors below SATD 16 (+0.07% src720 only, -8% instructions there; x264's 64 cost +0.94%); B 8x8 searches without quarter-pel (+0.28..0.56%, -7.7% on park); B 8x8 searched only in the list cheaper at 16x16 (+0.5..0.8%, -7.4%); 8x8 analysis skipped below 80 (B) / 40 (P) lambdas (+0.21% src720, -7%). Bit-exact speedups: leaner lookahead search, SWAR then SIMD downscale, frame clone without memset, sad_x4 loads/reduction, SA8D kernel, deblock skips internal edges of uniform inter MBs, table-driven 8x8 quant, generic qpel refine skipping the previous centre, full-plane half-sample filter.

Tried and dropped: x264 b-adapt 1 fast decision (+2.6..5.8%), lookahead early exit after candidates (+0.7..1.25% src720), P 16x8/8x16 early termination (+0.27% foreman for -1.2%), only the better-estimated B rect shape (+0.25..0.76% for -3..4.6%), quarter-pel gate on small SATD (no effect), final qpel refine 1 iteration, fixed-4 point batching in PartSearch (no gain).

Left on the table: per search op we still spend ~1.3x x264's instructions (bounds checks, mv cost lookups and dispatch around 8x8 SAD/SATD kernels); CABAC residual is ~7% on park; a SIMD lowres window search (mpsadbw) for the lookahead; a SIMD 8x8 forward DCT. Not closing: the speed half of the target is met only on src720.

## 2026-10-30

Round 4 (branch worktree-agent-a8d288051031e9368, commits 2fe5270 and 6fdd7db on top of f2f92b6): target was Compact faster than x264 veryfast on all three clips with BD-rate <= 0 on each (Y and average PSNR). Met on src720 only.

Finding: at equal instructions the gap was instruction-cache misses and mispredicted branches. Cachegrind on park16 (QP 28): I1 misses 48.2M -> 33.1M (x264 21.3M), mispredicts 21.4M -> 17.1M (x264 12.6M); instructions are unchanged (full clips: src720 9.85G vs x264 9.88G, park60 11.75G vs 9.85G, foreman150 2.45G vs 2.13G).

Changes: partition search compiled once for all sizes, calling per-size tables of x86-64-v3 kernel shims (four inlined copies made a 47 KB search); full-pel search context (precomputed block offset, bounds, motion vector cost rows); new pixel_sad_x4_wxh_at (AVX2/NEON/scalar, differential test); P/B macroblock coding and its intra and residual paths out of line through x86-64-v3 copies, list/quadrant loops kept rolled; branchless CABAC decision; lookahead on the offset SAD kernel. Output of all presets unchanged by these.

Trades (measured on all three clips): chroma_qp_index_offset 1 for Compact (shifts BD from average to luma: src720 Y -0.94 -> -5.38, avg -1.56 -> -1.42; park Y -1.04 -> -2.63, avg -2.18 -> -1.98); no quarter-pel in B 16x8/8x16 searches plus P 16x8/8x16 early termination (x264_mb_analyse_inter_p16x8): +0.13..0.64% for -1.4..2.7% instructions.

Final BD-rate (piecewise cubic, QP 20-36, keyint 250) vs veryfast, Y / avg: src720 -5.17 / -1.14, park -2.50 / -1.85, foreman -3.68 / -4.20. vs ultrafast: -34.9 / -31.5, -46.2 / -45.6, -69.8 / -69.8. All streams decode identically in Ragdoll and FFmpeg.

Speed, QP 28, pinned, 9 interleaved rounds, ours / veryfast median: src720 0.97, foreman 1.06, park 1.20 (runs vary up to 10%). ultrafast is 4.5-6x faster.

Dropped (measured): branchless significance/level coding (+4% instructions, -0.5M mispredicts: net worse), deblocking offsets +-1 (mixed, foreman +1.3%), no P 16x8/8x16 (src720 +1.2%, foreman +2.5%), no intra 4x4 in P (park +1.5..2%, foreman +2.4%), tighter intra 4x4 gate, no quarter-pel in B 16x16 search, sub-partition range 4, lookahead range 8, kernels not inlined (slower), non-inlined MB functions without v3 copies.

Left: per search point ~30 instructions of bookkeeping around the kernels; indirect-call mispredicts from the kernel tables (2.4M on park16); CABAC residual ~11% of park; the lookahead's path search (13 searches per 4 frames, 11-13%). Park needs ~17% more; foreman ~6%.

## 2026-10-30

Round 5 (branch worktree-agent-a8d288051031e9368, no code change kept on top of 5883343): goal was Compact faster than veryfast on park (needs ~17%) and foreman (~6%) with BD <= 0. Not reached; every measured attempt was neutral or worse, so nothing was committed.

Cycle model from cachegrind (Ir/3 + 12 I1 misses + 10 D1 misses + 16 mispredicts + 100 LL misses; it tracked the measured ratios within a few percent) on park16 QP 28: ours 1.890G vs x264 1.608G. The gap is +0.126G from instructions (2.818G vs 2.439G), +0.139G from I1 misses (32.9M vs 21.3M) and +0.071G from mispredicts (16.9M vs 12.6M); D1/LL misses favour us by 0.05G. Even at x264's I1 misses and mispredicts we would stay about 8% slower on park from instructions alone; closing it needs ~15% fewer instructions with no compression loss, and the measured speed/compression trades cost more BD than park's and src720's average-PSNR margins (-1.85%, -1.14%) allow. foreman16..40 model: 0.417G vs 0.412G (I1 7.9M vs 5.4M, mispredicts 3.6M vs 2.9M); measured 1.06x.

Attempts (all output-identical unless noted):
- Search generic over the block size (one out-of-line instance per size, kernels still out of line) to remove the 2.3M indirect mispredicts of the kernel tables: I1 misses 32.9M -> 41.7M, mispredicts 16.9M -> 17.7M, model +6% on park. Dropped.
- Branchless point selection (costs packed with their index, x264 COPY1_IF_LT): +66M instructions, mispredicts unchanged (the compare branch predicts well). Dropped.
- Branchless bypass bins: no change (already a conditional move). Dropped.
- Earlier this session: branchless significance/level coding (+4% instructions for -0.5M mispredicts), x264 b-adapt 1 decision (+2.6..5.8% BD for halving the lookahead), lookahead early exits (+0.7..1.3% BD on src720), P rect removal (+1.2% src720, +2.5% foreman), no P intra 4x4 (+1.5..2.6%).
- Lookahead early-out on path costs: analysed, not implemented: B wins most decisions on park, so the side computed second rarely gets cut, and an exact early-out must still run the searches whose vectors seed the full-resolution search.

Remaining gap on park, by size: CABAC residual (sig/levels ~11% of instructions, 3.3M of the mispredicts are its significance and loop branches, 0.8M the byte-output test), the partition search (~13% of instructions; ~30 instructions of bookkeeping per point, 2.3M indirect mispredicts), B/P macroblock analysis code (6.3M + 2.1M I1 misses), lookahead (~9%).
