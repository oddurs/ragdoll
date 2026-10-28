---
id: 72
uid: 1bf378e8-4089-4c8b-b64d-6f74f01d68ba
title: transcode and metrics decode the whole input into memory, so peak RSS grows with input length
type: bug
status: planned
milestone: m11-validation
created: 2026-10-28
updated: 2026-10-28
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

- [ ] A test transcodes synthesized inputs of two lengths (at least 4x apart) and asserts peak RSS differs by less than a fixed overhead; it fails on the current code.
- [ ] `transcode`, `metrics`, `extract` (thumbnails and non-MP4) and `concat` decode in GOP waves bounded by the memory budget.
- [ ] Output stays byte-identical to the current output for the BUG-0060 regression corpus.
- [ ] `0054` is corrected to reflect what it delivered.
