---
id: 88
uid: 710fa766-054b-445c-9255-77f43e949829
title: 'H.264 decoder: multithreaded decoding to beat FFmpeg''s frame-threaded throughput'
type: feature
status: done
milestone: m11-validation
created: 2026-10-28
updated: 2026-10-29
closed_at: 2026-10-29
priority: p1
---

## Problem

The benchmark suite compares the H.264 decoder with FFmpeg at 1 and at 4 threads (frame
threading). The decoder was single-threaded, so against FFmpeg's 4 threads it lost by 2.5x to
3.5x (300-frame 1080p High CABAC and 720p Baseline CAVLC streams on the 4-core x86_64 VM).

## Architecture & SIMD Design

Frame threading, as FFmpeg does it, with the decoder's output unchanged for any thread count:

- The caller's thread parses parameter sets and slice headers, builds reference lists and runs
  the DPB in decode order. Each progressive 4:2:0 frame picture is then decoded (all its slices
  and the deblocking filter) on one of `threads` workers; fields, MBAFF and monochrome pictures
  stay on the caller's thread once earlier pictures are complete.
- Deblocking runs in the loop, a row behind decoding (`deblock_rows`), so a picture publishes
  luma lines as final while it is decoded (`PicSync`). Motion compensation from a reference still
  being decoded waits for the lines its vectors (plus the 6-tap margin) reach.
- A reference picture's motion is copied row by row into a `LiveMotion` its slices' reference
  tables were written to up front; direct prediction waits for the co-located macroblock row.
- Samples are shared through the frame store (`PicPlanes`): the decoding picture is the only
  writer and readers on other threads read only published rows. Each worker keeps its own
  per-picture state (`PicState`) so that it stays in the worker's caches; sample planes and
  motion buffers are pooled.
- `H264Decoder::with_threads(n)` turns it on; `decode_annex_b_packet` then returns pictures a few
  access units late and `flush` returns the rest.

## Acceptance criteria

- [x] Decoding with `with_threads(n)` gives byte-identical output to single-threaded decoding for every supported stream (x264 reference set, 166 JVT conformance streams) at 2 and 4 threads.
- [x] At 4 threads the decoder is faster than FFmpeg's 4-thread frame threading on the benchmark suite's 1080p High CABAC and 720p Baseline CAVLC streams.
- [x] The CLI and pipeline decode H.264 with frame threads by default.

## 2026-10-29

Workers take reference pictures before non-reference ones (earliest first). Each reference picture waits only on earlier ones, so this cannot deadlock, and P pictures no longer queue behind the B pictures between them: worker line waits at 4 threads on 1080p High fell from ~1000 to ~670 ms of worker time (wall ~0.97x FFmpeg's 4 threads in paired runs). Running twice as many workers as cores, handing cores off around waits, was tried and dropped: the extra wake-ups and cache churn cost more than the waits.

## 2026-10-29

Round 10 (2026-10-29): a picture's sleepers are woken only once the progress they wait for is published (was: every row woke every sleeper, ~4.5 wakeups per wait; now ~1.5); futex time fell from 9.4% to 6.3% of CPU at 720p 4T. With the single-thread gains, 4 threads vs FFmpeg 4 frame threads (11 interleaved rounds, median ratio): 1080p High 0.89x, 720p Baseline 0.98x. Output identical to one thread on all 166 conformance streams at 1, 2 and 4 threads.

## 2026-10-29

Baseline streams (no B slices) keep no per-picture motion under frame threads: the worker's block buffer stays its own, hot in its core's cache, instead of being copied row by row into a shared buffer that nothing reads. 720p Baseline 4T 7% faster (15 interleaved rounds, median ratio 0.929); now 0.94x FFmpeg's 4-thread time. A B slice in a Baseline-labelled (non-conforming) stream predicts direct blocks from a co-located picture without motion, at any thread count, so one and four threads still agree. Tried first: moving the motion over once at the end instead of copying rows; no gain, because the worker then writes into a cold buffer every picture. CLI/pipeline: the sequential H.264 path (Matroska, MPEG-TS, and MP4 GOPs longer than a batch) and single-frame extraction decode with frame threads (up to 8, flushing at the end); MP4 GOP batches already decode GOPs in parallel, one thread each; the cut path's boundary GOP stays on one thread, pairing each packet with its picture. Test: tests/frame_threads.rs (fails without the flush).
