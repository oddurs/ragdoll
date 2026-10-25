---
id: 36
uid: 070cf7c0-4620-46ee-947d-56194afc65f3
title: Bounded-Memory GopRingPipeline & Streaming stdin/stdout Pipe I/O
type: feature
status: done
milestone: m6-streaming-av-sync
assignee: Oddur Sigurdsson
depends_on:
- 35
created: 2026-10-25
updated: 2026-10-25
closed_at: 2026-10-25
priority: p0
area: pipeline
effort: l
---

## Problem

In `v0.1.0`, `decode_video_file` and `transcode_file` materialize all decoded `VideoFrame`s into a `Vec<VideoFrame>` before encoding. While fast on short clips, a 2-hour 1080p or 4K video would exceed system RAM, and Unix pipe workflows (`cat input.ts | ragdoll transcode - out.mp4`) cannot rely on random-access `mmap` alone.

## Architecture & SIMD Design

1. Introduce `GopRingPipeline` in `ragdoll-pipeline`: a bounded-window streaming executor that pulls demuxed GOP batches (e.g. `W = worker_threads` GOPs at a time), recycles plane buffers through `FramePool::with_memory_budget` (`256 MiB` default, `64 MiB` in `PowerMode::Efficiency`), dispatches decode -> fused L1 scale/filter -> encode across worker threads, and flushes encoded packets to the muxer in presentation/decode order.
2. Add `StreamSource` in `ragdoll-format::io` abstracting zero-copy `MmapSource` (for seekable local files) and a 4 MiB ring-buffered `Read` adapter (for `-` / `stdin` pipes and sequential streams).
3. Support `-` / `stdout` streaming output for `MPEG-TS` (`.ts`), fragmented `fMP4`, `MKV`, `Y4M`, and `WAV`.

## Acceptance criteria

- [x] `transcode_file` processes arbitrarily long video streams within a bounded `< 256 MiB` (`< 64 MiB` in `--battery-saver`) memory footprint.
- [x] `ragdoll transcode - - --format ts` and `ragdoll probe -` work over Unix `stdin`/`stdout` pipes.
- [x] Zero throughput regression on multi-core GOP-parallel transcoding compared to `v0.1.0`.

## 2026-10-25

Implemented GopRingPipeline with bounded GOP-wave execution (256 MiB default / 64 MiB efficiency budget), FramePool plane recycling, StreamSource / MmapSource::from_reader, and OutputSink stdin/stdout (-) pipe streaming.
