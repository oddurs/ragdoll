---
id: 54
uid: e9831b68-c3a0-4bab-aed6-be741d25ce31
title: Stream decode through GopRingPipeline so transcode, extract and metrics use bounded memory
type: feature
status: done
milestone: m10-production-v1
assignee: Oddur Sigurdsson
created: 2026-10-25
updated: 2026-10-28
closed_at: 2026-10-25
priority: p1
effort: l
area: pipeline
---

## Problem

`transcode_file_with_options`, `extract_media` and `evaluate_media_metrics` call `decode_video_bytes` / `decode_video_file`, which materialize every decoded frame in a `Vec<VideoFrame>` before the bounded `GopRingPipeline` wave runs. Peak memory therefore grows with input length, which contradicts the constant-memory goal in ADR-0035 / 0036 for multi-hour inputs.

## Architecture & SIMD Design

Decode per GOP inside each wave: partition the demuxed `SampleEntry` index into GOPs first, then decode, scale and encode a wave of GOPs and release their frames to the pool before decoding the next wave. `extract` should seek to the nearest keyframe instead of decoding everything; `metrics` should walk both inputs GOP by GOP.

## Acceptance criteria

- [x] `extract --at` decodes only the GOP containing the target frame.
- [x] Memory-limit callouts removed from the transcode and memory-model docs pages.

## Delivered elsewhere

These were ticked here without a test, and were not true until BUG-0072:

- Peak RSS of `transcode` on a long synthesized input stays within the configured budget plus a fixed overhead, independent of input length (test with two lengths).
- `metrics` processes reference and distorted inputs in bounded waves.

## 2026-10-28

Reopened in substance as BUG-0072 on 2026-10-28: peak RSS of transcode grows with input length (113 MB for 2 s, 405 MB for 8 s of 720p) because decode_video_bytes decodes the whole input before GopRingPipeline runs. The first and third criteria were ticked without a test that measures RSS across two lengths. metrics also decodes both inputs in full. extract --at on MP4 is genuinely bounded.

## 2026-10-28

Corrected by BUG-0072: the RSS and metrics criteria moved under "Delivered elsewhere" because this item did not deliver them. It delivered criterion 2 (extract --at on MP4 decodes only the target GOP) and the docs change. Peak RSS independent of input length for transcode, metrics, extract (thumbnails and non-MP4), concat and package came with BUG-0072's VideoFrameStream, with tests/bounded_memory.rs measuring two lengths.
