---
id: 72
uid: 1bf378e8-4089-4c8b-b64d-6f74f01d68ba
title: transcode and metrics decode the whole input into memory, so peak RSS grows with input length
type: bug
status: done
milestone: m11-validation
assignee: Claude
created: 2026-10-28
updated: 2026-10-28
closed_at: 2026-10-28
priority: p0
effort: m
area: pipeline
---

## Problem

`0054` is closed with the criterion "peak RSS of `transcode` … independent of input length" ticked, but on `main` at `49dc39f` peak RSS is proportional to input length:

| Input (`ragdoll synth`, 1280x720, 30 fps) | `ragdoll transcode --qp 22` peak RSS |
|---|---|
| 2 s | 113 MB |
| 8 s | 405 MB |

That is about 49 MB per second of 720p, so a one-hour 4K input needs hundreds of GB and is OOM-killed.

Cause: `transcode_file_with_options` calls `decode_video_bytes`, which decodes every frame into a `Vec<VideoFrame>` and passes the whole vector through `apply_video_filters` before `GopRingPipeline` sees it. The ring bounds scaling and encoding only. The same pattern affects:

- `evaluate_media_metrics` (`ragdoll metrics`), which decodes both inputs in full.
- `extract_media` for thumbnails and for any non-MP4 input. `extract --at` on MP4 already decodes only the target GOP.
- `concat_media` and `package_abr`, which call `decode_video_file`.

The test behind `0054`'s first criterion either does not exist or does not measure RSS across two lengths.

## Architecture & SIMD Design

Decode one GOP wave at a time: demux the packet index (already zero-copy), partition it into GOPs, and feed `GopRingPipeline` waves of decoded GOPs whose frame memory counts against the budget. Filters run per wave. `decode_video_bytes` stays as a convenience for callers that want every frame, but no CLI verb uses it on unbounded input. `order_h264_frames` (BUG-0060) applies per wave with a running PTS offset.

## Acceptance criteria

- [x] A test transcodes synthesized inputs of two lengths (at least 4x apart) and asserts peak RSS differs by less than a fixed overhead; it fails on the current code.
- [x] `transcode`, `metrics`, `extract` (thumbnails and non-MP4) and `concat` decode in GOP waves bounded by the memory budget.
- [x] Output stays byte-identical to the current output for the BUG-0060 regression corpus.
- [x] `0054` is corrected to reflect what it delivered.

## 2026-10-28

Design: VideoFrameStream (crates/ragdoll-pipeline/src/stream.rs) decodes a batch at a time in presentation order. MP4/H.264 decodes a wave of whole IDR-started GOPs in parallel (as before, one decoder per GOP), sized by the batch budget (budget/4); a GOP larger than the budget alone goes through one decoder with a 17-picture reorder window. Matroska and MPEG-TS use one decoder with the same window: a conformant stream outputs no picture after more than 16 later-decoded ones, so releasing the lowest POC of 17 equals sorting the keyframe run (the decoder's PTS carries the IDR/MMCO5 period in its high bits). Y4M frames are copied out of the mapping and those pages released (new MmapSource::release_range, MADV_DONTNEED on the read-only private file mapping; Y4mDemuxer::open_mapped also releases while it indexes, since touching each FRAME header fault-maps most of the file). VideoFilterChain streams the transcode filters, including --fps resampling (monotonic source index, frame count from the container index). GopRingPipeline::streaming_encoder/GopRingEncoder encodes each wave as it fills, with the same GOP chunking (gop_size from the planned frame count) and wave grouping as encode_gops_bounded_with_h264, which now runs on it; ABR keeps one encoder and encodes frames as they arrive. transcode, metrics (new evaluate_media_metrics_with_budget), extract (thumbnails and non-MP4 --at, which also stops decoding after the last wanted frame), concat (re-encode path streams into the temporary Y4M), package_abr (waves of segments; every rendition encodes them in parallel) and synth all stream. decode_video_bytes now collects the stream. order_h264_frames is replaced by the stream's ordering.

## 2026-10-28

Evidence. Byte-identical: a 67-case corpus (synth, x264 B-frame MP4/MKV/TS with AAC, Y4M and PNG-sequence inputs; MP4/MKV/TS/Y4M/PNG-sequence/RGDI outputs; scale, fps, crop+pad+rotate, overlay, auto-crop, Baseline/Main/High, ABR, AAC audio; metrics, extract thumbnails and --at on MP4/MKV/TS, concat copy and re-encode, package, synth) hashes 170 output files identically to main at the default budget and with --power-mode efficiency (64 MiB, which drives the large-GOP and multi-wave paths), --threads 4. Peak RSS, 1280x720 synth, default budget, main -> branch: transcode MP4 8/16/32 s 364/686/1336 -> 178/176/191 MB; metrics 8/32 s 685/2626 -> 225/230; extract --thumbnails 380/1349 -> 105/109; package 682/2604 -> 165/170; concat (re-encode) 1331/4200 -> 236/290; transcode Y4M 639/2545 -> 206/259 (257 at 64 s: at the default budget the waves fill after ~10 s of 720p). tests/bounded_memory.rs (320x180, 4 MiB budget, 2 s vs 8 s, child processes reading VmHWM because ru_maxrss survives exec): main fails transcode MP4 16 -> 32 MiB; branch transcode MP4 9 -> 9, metrics 13 -> 13, transcode Y4M 10 -> 11 MiB. Audio is still decoded whole: filed BUG-0086.

## Result

transcode, metrics, extract, concat, package and synth decode video a batch at a time through VideoFrameStream and encode it through the streaming GopRingEncoder, so peak RSS follows the memory budget instead of the input length (720p transcode: 191 MB at 32 s against 1336 MB before), with output byte-identical to before on a 170-file corpus. tests/bounded_memory.rs guards it. Audio is still decoded whole (BUG-0086).
