---
id: 76
uid: 753e8266-7387-458c-bed6-228b0b080436
title: 'H.264 decoder performance: SIMD macroblock reconstruction and partition-sized motion compensation'
type: kernel
status: doing
milestone: m11-validation
assignee: Claude
claimed: 2026-10-28
depends_on:
- 75
created: 2026-10-28
updated: 2026-10-29
priority: p1
effort: l
area: codec-video
---

## Problem

The reference-grade H.264 decoder (`0075`) is bit-exact but slow: on x86_64 it decodes 1280x720 x264 `-preset medium` at about 53 fps and Ragdoll-encoded Baseline at about 90 fps, single-threaded, while FFmpeg decodes the same streams at several hundred fps per core. Correctness came first; nothing in the new core is vectorised yet.

Known costs:
- Every inter block, including 4x4 partitions, runs the full 16x16 quarter-pel kernel, which on x86_64 is the scalar path with per-sample clamping.
- Intra prediction, inverse transforms, dequantisation and deblocking are scalar `i32` code.
- CAVLC `coeff_token`, `total_zeros` and `run_before` use linear VLC matching.
- Per-slice reference lists clone `Arc`s and per-picture state is reallocated for every picture.

## Architecture & SIMD Design

Partition-sized luma and chroma MC kernels with edge emulation only at picture borders, AVX2 and NEON versions differential-tested against the scalar reference; SIMD 4x4/8x8 inverse transforms in the spec's row-then-column order; vectorised deblocking for luma and chroma edges with per-segment bS; table-driven CAVLC; per-picture state reuse through `FramePool`. Every change must keep `tests/h264_reference.rs` and the conformance corpus byte-identical.

## Acceptance criteria

- [ ] 720p x264 `-preset medium` decodes at 300 fps or more single-threaded on x86_64 AVX2 and on Apple M-series NEON, measured by the benchmark suite.
- [x] Every SIMD kernel has a differential test against its scalar reference.
- [x] The x264 matrix and all supported conformance streams stay byte-identical with JM.

## 2026-10-28

2026-10-28 head-to-head (Xeon 2.1 GHz VM, AVX2, single-threaded, in process; libavcodec 6.1.1 called directly with packets pre-demuxed, best of 3): 1080p x264 medium High/CABAC at 6.2 Mbit/s decodes at 29 fps against FFmpeg's 181 fps (505 fps with 4 frame threads), 6.3x slower; 720p x264 Baseline/CAVLC at 3.1 Mbit/s at 77 fps against 834 fps, 11x slower. 1080p30 is just under real time. Harness: rbench (Ragdoll), avbench (libavcodec) in the session scratchpad; the inputs are testsrc2 with light temporal noise.

## 2026-10-28

Round 1 (this PR): partition-sized AVX2/NEON MC + weighting + avg kernels (ragdoll_simd::h264_mc), 16-line SIMD deblocking edges (h264_deblock), per-MB bS shortcuts, picture-buffer reuse with per-MB state reset, bit-cache + branch-free CABAC, CAVLC lookup tables, division-free neighbour location, direct-mode grouping, VideoFrame::from_rows. Per access unit like avbench (rbench h264-dec-au): 1080p High 29 -> 74 fps (FFmpeg 202), 720p Baseline 77 -> 288 fps (FFmpeg 804); callgrind 10.45 G -> 2.76 G instr on 30 frames 1080p (FFmpeg 1.12 G). All 166 JVT streams byte-identical to before. Remaining gap ~2.7x: MC glue (~550 M vs ~210 M), deblock (~380 M vs ~220 M), direct (~300 M vs ~25 M), CABAC residual, intra pred and IDCT still scalar.

## 2026-10-28

Round 2: CABAC engine packed (2*state+mps contexts, 64-bit offset window, state in registers across a residual block); residual blocks decoded as (scan position, level) lists dequantised into a persistent buffer that the IDCTs clear; MB-level direct/P_Skip neighbours and per-MB co-located resolution; rectangle motion stores; inner-edge motion mask and bitmask edge walk in deblocking; whole-sample copies instead of edge emulation. Callgrind on 30 frames of 1080p High: 2.58 G -> 2.06 G instructions (FFmpeg 1.22 G), mispredicts 12.5 M -> 10.4 M (FFmpeg 4.8 M), I1 misses 5.9 M (FFmpeg 1.4 M); data-cache misses are level (12.1 M vs 10.8 M), a fifth of ours from the output copy in make_output. Wall clock on a loaded 4-core VM, per access unit: 1080p 3.88 s -> 3.47 s per 300 frames (FFmpeg ~1.65 s); 720p Baseline 726 ms (FFmpeg ~330 ms), CAVLC path not yet reworked. All 166 JVT streams byte-identical, also under debug assertions.

## 2026-10-29

Round 3: CAVLC from 32-bit peeks with sparse coefficient lists; MC kernels that average into the destination (AVX2/NEON, differential-tested) and direct calls for default-weighted blocks; whole-macroblock spatial direct when the co-located MB has one motion (reference motion records it per MB); fixed-size CABAC significance/last context arrays and a renormalisation table; per-4x4 state (both lists' mv/ref_idx/ref_pic/mvd, nz, direct) as one Vec<BlockInfo>; division-free MB positions; fast left/top neighbours without MBAFF; no eager error values. Callgrind: 30 frames of 1080p High 2.06 G -> 1.67 G instructions (FFmpeg 1.22 G including its demux), 10.4 M -> 9.2 M mispredicts; 60 frames of 720p Baseline 1.28 G -> 1.04 G (FFmpeg 0.74 G including demux). The MC kernels now cost about what FFmpeg's assembly does; the remaining gap is per-call glue around them, CABAC residual (~320 M vs ~250 M) and direct/neighbour bookkeeping. All 166 JVT streams byte-identical.

## 2026-10-29

Round 4: CABAC engine with one-base tables, a refill marker bit and node-state level contexts, constant-folded for 4x4 luma and chroma AC; read_residual per entropy coder; progressive MC path; macroblock motion written once (no per-MB block reset); neighbour availability once per macroblock; Option-free vector prediction; SIMD idct8 and intra predictors (AVX2/NEON) with a per-MB intra mode grid; decoded frames share the decoder's picture planes (no output copy). Callgrind: 30 frames of 1080p High 1.658 G -> 1.389 G instructions (FFmpeg 1.12 G), D1 misses 12.8 M -> ~10.3 M; 60 frames of 720p Baseline 1.024 G -> 0.918 G (FFmpeg 0.68 G). All 166 JVT streams byte-identical.

## 2026-10-29

Round 4, merged: CAVLC residual decoding from compile-time tables with a register-resident reader, paired level/run_before lookups and an inlinable read_ue (720p Baseline CAVLC residual 151 M -> 57 M instructions). Totals: 1080p High 1.385 G, 720p Baseline 0.825 G instructions (FFmpeg 1.12 G and 0.68 G).

## 2026-10-29

Round 5: deblocking kernels take edge direction and strong/normal filter as const generics (h264_luma_deblock_edge::<HORIZONTAL,STRONG>); bS intra/motion conditions computed without short-circuit branches; BlockInfo shrunk to 16 B by dropping ref_pic in favour of per-slice RefIds tables (mb_table index per MB; deblock compares (list,idx) when one table without duplicates covers both MBs). The diet costs ~+25M Ir in strengths but cuts D1 misses ~15%: wall 1080p 152.4 -> 148.4 ms (FFmpeg 120), 720p 76.4 -> 77.3 ms (FFmpeg 54.5). Conformance: 166 JVT streams identical.

## 2026-10-29

Round 6 (instruction counts, callgrind; wall clock not measured: the machine was shared with two benchmark agents). Deblocking visits only the edges it filters (edge 0, then set inner-edge bits) and reads bS blocks with one bounds check per edge; tc0 from a table with -1 for bS 0. Per-MB reset limited to coefficient counts; Intra4x4/8x8 modes never reset. MC entry (mc_luma, mc_chroma_pair) inlined with edge paths cold. CAVLC nC and CABAC cbf contexts from block coordinates without MBAFF. 1080p CABAC 1402 -> 1312M (FFmpeg 1122M), 720p CAVLC 842 -> 761M (FFmpeg 682M). Tried and dropped: branch-free bS for all segments (+116M), fused CABAC dequant (spills, +4.5M), chroma pair kernel (+10M), chunks_mut row fills (+19M). Row-lagged deblocking cut simulated L2 misses 19% at 1080p but its wall-clock effect could not be measured on the loaded machine; parked.

## 2026-10-29

Round 7 (callgrind instruction counts; wall clock still not measurable on the shared machine). Neighbour lookups by block coordinates without MBAFF for CABAC mvd/ref_idx contexts (once per neighbour, not per component), CBP contexts and partition MV predictors; one-list P_Skip neighbours; MC kernels entered through h264_luma_mc_in_plane / h264_chroma_mc_in_plane (one inside-the-picture test covers the kernel bounds); explicitly weighted P slices (x264 weightp) take the frame fast path with a per-slice bitmask of default-weighted references; spatial direct reads the co-located block without resolving its picture; byte scan maps for 4x4 dequantisation. 1080p CABAC 1312 -> 1233M (FFmpeg 1122M, +9.9%), 720p CAVLC 761 -> 735M (FFmpeg 682M, +7.8%). Remaining 1080p gap is mostly CABAC residual decoding (~290M vs FFmpeg's ~190M, which uses an assembly significance-map loop).

## 2026-10-29

Round 8: wall-clock work guided by a SIGPROF sampler (100 us CLOCK_MONOTONIC timer, ours and FFmpeg via LD_PRELOAD). Findings: MC kernels ran at half the IPC of the rest (reference-frame misses); fill_blocks took 11% of 720p time with 3% of instructions (the motion grid started 16 bytes into a cache line, so every macroblock row of records straddled two lines, behind pending stores to the cold destination picture). Changes: FFmpeg-style prefetch of reference samples 4 MBs ahead (2-5%), AlignedVec motion grid on 64-byte boundaries (~2%), CABAC skip flag inline, chroma QPs cached per QP change. Wall clock (best of 13, pinned): 1080p 147 ms vs FFmpeg 122 (1.20x), 720p 71 vs 56 (1.27x). Tried and dropped: PGO (slower), opt-level s (slower), huge pages for planes (~1%, within noise), CABAC sbb-mask decision (+21M instructions), destination prefetch (moves the stall), most-recent free store.

## 2026-10-29

Round 9: inter prediction of progressive frames goes through H264McKernels, raw-pointer luma and chroma-pair kernels chosen once per CPU, with one reach check per plane and the reference planes' sizes checked once per slice; macroblocks of one motion go straight from inter_predict_mb to the kernels; weighted prediction (explicit and implicit) predicts each list into scratch with the same kernels and the chroma pair kernel; whole-sample blocks are copied inline. Callgrind instructions, 60-frame 720p Baseline / 30-frame 1080p High: 747M -> 697M and 1225M -> 1173M. A SIMD bS for list-0 edges was tried and dropped: the scalar early exit for identical motion is cheaper. Batching deblocking 4-8 rows behind was dropped too: fewer I-cache misses in simulation but 10-17% slower on the machine, as the rows fall out of L1. madvise(MADV_HUGEPAGE) picture buffers: no gain at 720p, 10% slower at 1080p.

## 2026-10-29

Round 10 (2026-10-29), single-threaded vs libavcodec 6.1.1 (avbench, same VM, interleaved runs): 720p Baseline/CAVLC now 0.94x FFmpeg's time (15 rounds pinned to one core, median ratio; was 1.04x), 1080p High/CABAC about 0.9x (was 0.98x). The two causes were found with a SIGPROF time profile, not instruction counts: (1) deblocking built each edge's four bS bytes in memory and read them back as one u32, a store-forwarding stall that was the hottest instruction pair in the filter; bS is now built in a register. (2) The destination-row prefetches were issued 32 at a time every 4th macroblock and stalled on the line fill buffers; the same lines are now prefetched a share per macroblock. Also: one record per macroblock (MbInfo) instead of 11 arrays, block records stored whole from a register, both reference lists prefetched. Tried and dropped: branch-free P-only bS (slower: the motion branch predicts well), per-MB SIMD bS (90% of macroblocks only need their two outer edges).

## 2026-10-29

Round 11 (2026-10-29), all measured against main by interleaved rounds and dropped as no gain (within this VM's +-5% session noise): branch-free CABAC bypass bins and coefficient signs (0.964x in one session, 1.027x in the next); branch-free significance map (the last flag decoded in effect only; 1.011x); a ring of four macroblock rows for the per-4x4 arrays of pictures whose motion no later picture reads (all conformance streams identical; 1.010x 720p, 1.006x 1080p, 1.021x 720p 4T, so the 0.9 MB per picture of block records is not a bottleneck). The diff of the ring is in the session scratchpad (perf/ring.diff). Single-thread standing after rounds 10-11: level with FFmpeg (0.94-1.07x across sessions) on 720p Baseline and 1080p High; with four frame threads 1.18x FFmpeg on 1080p High and level on 720p Baseline. Per-stage time profiles against FFmpeg's: our MC and deblocking take fewer samples than FFmpeg's, CABAC residual decoding more (32% of 1080p time), CAVLC residual about the same.
