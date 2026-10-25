---
id: 54
uid: e9831b68-c3a0-4bab-aed6-be741d25ce31
title: Stream decode through GopRingPipeline so transcode, extract and metrics use bounded memory
type: feature
status: backlog
milestone: m10-production-v1
created: 2026-10-25
updated: 2026-10-25
priority: p1
effort: l
area: pipeline
---

## Problem

`transcode_file_with_options`, `extract_media` and `evaluate_media_metrics` call `decode_video_bytes` / `decode_video_file`, which materialize every decoded frame in a `Vec<VideoFrame>` before the bounded `GopRingPipeline` wave runs. Peak memory therefore grows with input length, which contradicts the constant-memory goal in ADR-0035 / 0036 for multi-hour inputs.

## Architecture & SIMD Design

Decode per GOP inside each wave: partition the demuxed `SampleEntry` index into GOPs first, then decode, scale and encode a wave of GOPs and release their frames to the pool before decoding the next wave. `extract` should seek to the nearest keyframe instead of decoding everything; `metrics` should walk both inputs GOP by GOP.

## Acceptance criteria

- [ ] Peak RSS of `transcode` on a long synthesized input stays within the configured budget plus a fixed overhead, independent of input length (test with two lengths).
- [ ] `extract --at` decodes only the GOP containing the target frame.
- [ ] `metrics` processes reference and distorted inputs in bounded waves.
- [ ] Memory-limit callouts removed from the transcode and memory-model docs pages.
