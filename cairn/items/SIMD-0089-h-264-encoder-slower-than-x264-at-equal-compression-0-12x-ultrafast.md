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

## 2026-10-30

Round 6 (branch worktree-agent-a8d288051031e9368): earn compression cheaply, spend it on speed. Park and foreman still not under 1.0.

Earned: x264 veryfast uses the inter deadzone f = 1/6 (trellis 0); rounding inter residuals with f = 1/3 like intra ones (luma 4x4/8x8, chroma AC, skip probes and bounds; Compact only) gains on every clip vs veryfast, Y / avg: src720 -5.17/-1.14 -> -5.46/-2.59, park -2.50/-1.85 -> -7.34/-6.02, foreman -3.68/-4.20 -> -6.50/-5.98. Measured alternatives: inter f = 1/7 (+0.4..1.0% worse), 1/5, 1/4, 2/7 (in between), intra f = 0.4 or 0.29 (+0.5..0.8% worse), chroma at 1/6 (src720 avg +1.4% worse), chroma above 1/3 (worse avg), deblocking offsets -1/+1 (mixed). The finer rounding codes ~0.9 dB more at a given QP (park QP 28 35.08 dB vs x264's 34.18), so at the timing protocol's fixed QP it is ~7% slower; every Compact picture is now coded one QP step coarser than asked, so that its PSNR at a QP matches veryfast's (park QP 28: 3,012,029 B / 34.25 dB vs 3,141,391 B / 34.18 dB; foreman 211,142 B / 36.54 vs 208,030 / 36.48; src720 1,727,472 B / 45.23 vs 1,764,455 / 45.14).

Spent (measured on all clips): no P 16x8/8x16 and no intra 4x4 analysis in P macroblocks (park -7.5%, foreman -7.5% instructions). Not taken: x264 b-adapt 1 (only -1.7% instructions on park and +0.8% on foreman once more P frames are coded, and +3..6% BD), lookahead early exits (src720 only).

Final, BD vs veryfast Y / avg: src720 -2.54 / -0.35, park -5.99 / -4.92, foreman -2.12 / -1.59 (vs ultrafast -32.8/-30.5, -47.1/-46.2, -69.2/-68.8). Pinned interleaved 9 rounds, ours / veryfast at QP 28: src720 0.87, foreman 1.04, park 1.14. Instructions, full clips: src720 9.56G vs 9.88G, park60 11.03G vs 9.85G, foreman150 2.33G vs 2.13G.

Remaining gap: park ~12%, foreman ~4%, with the binding BD margin now src720's average PSNR (-0.35%) and foreman's (-1.59%); further trades measured so far each cost more than that. A stronger quantizer (RD-optimized zero-out of trailing +-1 levels, per-position rounding) is the next lever for margin; per-position rounding needs a quant kernel with a bias per coefficient.

## 2026-10-30

Round 7 (branch worktree-agent-a8d288051031e9368, on top of 9946f79): the one-step QP offset is gone (a requested QP is the QP coded, I -3 / P 0 / Bref +2 / B +3 as before); speed is now reported at matched quality: per clip the Compact QP whose Y PSNR is closest to and not below x264 veryfast's at QP 28. That is QP 29 on all three clips, before and after this round. Without the offset the round-6 trades still hold, BD vs veryfast Y / avg: src720 -3.58/-0.94, park -6.05/-4.64, foreman -2.19/-1.53.

Quantizer: trailing +-1 levels are dropped when their CABAC rate (new CabacContext::cost, x264_cabac_entropy table) weighted by x264's lambda2 over the squared step (35/256 at every QP) exceeds the distortion they remove. Measured on luma 4x4, luma 8x8 and chroma AC: -1.1..-1.7% BD on every clip, but with luma it codes 0.42-0.45 dB less Y PSNR at a QP, which moves every clip's matched QP from 29 to 28 (src720 44.81 vs 45.14, park 33.80 vs 34.18, foreman 36.12 vs 36.48 dB at QP 29), about a QP step of speed. Kept for chroma AC only (src720 -2.09/-0.62, park -0.83/+0.27, foreman -0.15/+0.30; Y PSNR at a QP unchanged). Also measured: lambda 25/50 (worse than 35), interior +-1 levels (mixed, dropped), luma rounding 0.39/0.44 with RDO (better Y, worse avg), per-position rounding in a scalar experiment (DC 0.42 to 0.25 at the highest frequency: -0.3..-0.6% Y, avg +-0.1; steeper raises PSNR at a QP but costs foreman +0.4/+0.9), so no per-coefficient-bias kernel was written. decimate_score was already used for inter blocks (4x4 per 8x8 < 4, MB < 6, 8x8, chroma AC < 7).

Other gains: Compact lambda 4/5 of x264's (src720 -1.01/-1.01, park +0.11/+0.04, foreman -0.14/-0.16; 0.65 and 1.25 worse); intra 4x4 back in P macroblocks with x264's 1.5x gate (src720 -0.20/-0.39, park -0.83/-1.16, foreman -2.03/-2.26 for +1.2..1.8% instructions). Not gains: intra 4x4 in B (none), x264's B offsets +1/+2 (+1..2% worse).

Spent (each measured on all clips): B 8x8 gate 80 -> 300 lambdas (park -6.4%, foreman -14.3% instructions); no bi-predicted B 8x8 quadrants (-2.5/-2.1%); only the better-estimated B 16x8/8x16 shape (-1.3/-1.2%); B searches without the predictor's sub-pel probe and with a 4-point diamond instead of the 8-point square after the hexagon (-3.8% park); one final quarter-pel diamond in B (-0.6%). Bit-exact: FullPel::take out of line (search_qpel 17 KB -> 7.5 KB, park I1 misses 30.4M -> 28.9M).

Final BD-rate (piecewise cubic, QP 20-36, keyint 250) vs veryfast Y / avg: src720 -5.91/-1.67, park -6.63/-4.34, foreman -1.53/-0.58; vs ultrafast -34.8/-31.0, -46.3/-44.5, -68.9/-68.4. All streams decode identically in Ragdoll and FFmpeg; Balanced/Fast/Baseline output byte-identical.

Matched quality (Compact QP 29 vs veryfast QP 28): src720 1,702,033 B / 45.27 dB vs 1,764,455 B / 45.14; park 2,996,738 / 34.26 vs 3,141,391 / 34.18; foreman 213,210 / 36.56 vs 208,030 / 36.48 (QP 30 is below on all: 44.37, 33.52, 35.96). Pinned, 15 interleaved rounds, ours / veryfast median: src720 0.85-0.87, foreman 0.95, park 1.06. Instructions, full clips: src720 9.26G vs 9.88G, park60 10.25G vs 9.85G, foreman150 2.02G vs 2.13G. Cycle model (cachegrind): park16 1.666G vs 1.608G (instructions now equal, 2.45G vs 2.44G; I1 misses 28.9M vs 21.3M, mispredicts 14.6M vs 12.6M), src30 0.92x, fore40 0.87x.

Not closed: park stays ~4-6% slower than veryfast at matched quality. The binding margin is foreman's (avg -0.58%); every further park trade measured costs foreman about 1% BD per 1-2% of park instructions (B 16x16 without quarter-pel in the search: park -1.7% for +1.5%; P intra 4x4 gate 1.2x: -1.2% for +1.0%). Luma RDO would give each clip ~1.2% more margin, but under the matched-QP protocol it moves all three to QP 28. A lookahead with vectors chained through a middle frame as extra candidates gains src720 -2.5% BD (park/foreman neutral) but lowers src720's PSNR at QP 29 by 0.24 dB (more B frames), so it was left out too. What is left on park is I-cache misses (search_qpel, B/P macroblock analysis, CABAC residual) and the lookahead (12% of park, 13 searches per 4 frames).

## 2026-10-30

Round 8 (branch worktree-agent-a8d288051031e9368, on top of f064c8b): output-identical speedups only; every preset's output is byte-identical to f064c8b (Compact on all clips and QPs, Balanced/Fast/Baseline by same.sh), so BD-rate is unchanged: vs veryfast Y / avg src720 -5.91/-1.67, park -6.63/-4.34, foreman -1.53/-0.58.

Measurement note: park16 (the first 16 frames) understated the gap. On 28 mid-sequence frames and on the full clip, instructions are already at x264's and the gap is instruction-cache misses and mispredicts. Full clips, cachegrind cycle model (Ir/3 + 12 I1 + 10 D1 + 16 mispredicts + 100 LLd), x264 veryfast QP 28 | f064c8b QP 29 | now QP 29:
- park60: 6.193G | 6.897G | 6.462G (ours/x264 1.114 -> 1.043). Ir 9.85G | 10.25G | 9.51G, I1 misses 92.7M | 127.3M | 117.6M, mispredicts 51.3M | 62.8M | 57.9M.
- src720: 6.477G | 6.166G | 5.952G (0.952 -> 0.919). Ir 9.89G | 9.27G | 8.81G, I1 58.6M | 67.8M | 64.7M.
- foreman150: 1.395G | 1.356G | 1.279G (0.972 -> 0.917). Ir 2.13G | 2.02G | 1.91G, I1 22.1M | 26.9M | 24.9M.
Wall clock on this VM varies by up to 15% between runs of one binary; 15-25 pinned interleaved rounds (order alternating) gave park 0.95-1.06, src720 0.84-0.90, foreman 0.78-0.95, and f064c8b's park 1.06-1.12 in the same runs.

Changes (each output-identical; model / Ir on park):
- The partition search's v3 kernel shims call the AVX2 kernels unchecked (new ragdoll_simd::h264_me::avx2_unchecked, bounds asserted in debug builds; test against the checked API): the bounds asserts and dispatch checks were ~2% of instructions. The lookahead's search and its bi-prediction costs (now a v3 copy) use them too.
- New kernels (AVX2 + NEON + scalar, differential tests): sub_dct8x8_x2 (two 8x8 forward transforms in the two lanes, replacing the scalar 32-bit transform), quant_8x8 (16-bit multiplier per position), intra_satd_x3_8x8 (the lookahead's three intra SATDs from one Hadamard transform). The 8x8 reconstruction uses the decoder's idct8 kernel; a test checks it against add8x8_idct_u8.
- CABAC residual coder: significance map and levels with the coder's registers in locals (Arith; put_byte out of line, taking and returning them by value), one table for rangeTabLPS and transitions, no saturating shift; the encoder's AVX2 quant/dequant/zigzag/mask/DCT/intra-SATD/chroma-MC kernels out of line rather than inlined at each call site.
Instructions on the first 16 park frames: 2.453G -> 2.287G (x264 2.439G).

Measured and dropped (model worse or neutral): residual coder compiled as a v3 copy; bounds-free 4x4 significance loop (more mispredicts); level loop over a bitmask of non-zero positions; residual/probe/intra helpers out of line (I1 up); loops of the residual coding kept rolled (I1 up); direct per-size kernel calls instead of the pointer table (cachegrind counts more mispredicts; wall clock not better); deblocking as a v3 copy (neutral).

Left: per macroblock our encoder touches ~9 KB more distinct code than x264 (park mid-sequence 575 vs 441 I1-missed lines per macroblock); code_b_mb_v3 alone ~150 lines per B macroblock. Shrinking that needs a restructuring of the macroblock analysis (fewer inlined helper copies per macroblock), not local outlining, which measured worse. Also deblocking (4.5% of park, bS in scalar) and the lookahead (now ~9%).
