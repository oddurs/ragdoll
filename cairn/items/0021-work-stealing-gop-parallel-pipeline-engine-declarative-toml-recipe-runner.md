---
id: 21
uid: d25fb4cd-7a02-4dbd-814d-2ab7ad504e17
title: Work-Stealing GOP-Parallel Pipeline Engine & Declarative TOML Recipe Runner
type: feature
status: done
milestone: m5-cli-bench
assignee: Oddur Sigurdsson
depends_on:
- 13
- 15
- 18
- 19
created: 2026-10-24
updated: 2026-10-25
closed_at: 2026-10-25
priority: p0
area: pipeline
effort: m
---

## Problem

Traditional frame-by-frame transcoding pipelines serialize on entropy decoding/encoding or require complex thread queues. Independent GOPs (Groups of Pictures) can be decoded, filtered through L1-resident SIMD scanline strips, and re-encoded in parallel across all Apple M4 performance cores.

## Architecture & SIMD Design

1. Build `ragdoll-pipeline` with a work-stealing GOP-parallel executor that partitions source streams at keyframe boundaries (`stss` / `Cues`) and processes GOP segments concurrently while preserving exact output presentation order.
2. Implement a declarative `.toml` pipeline recipe parser supporting multi-step workflows (input source, trim/seek window, fused pixel filter chain, video/audio codec settings, and multiple output targets).
3. Track real-time telemetry (frames/sec, effective memory bandwidth GB/s, compression ratio, peak RSS).

## Acceptance criteria

- [x] GOP-parallel transcoding scales across available CPU cores with deterministic frame ordering.
- [x] Declarative `.toml` pipeline recipes execute end-to-end transcode and remux jobs accurately.

## 2026-10-25

Implemented work-stealing GOP-parallel pipeline scheduler (partition_into_gops, parallel_gop_map over std::thread::scope + AtomicUsize, parallel_scale_frames with shared Q14 PolyphaseFilterBank tables), declarative TOML recipe parser/runner (PipelineRecipe), and unified multi-container remux/transcode/extract/metrics/synth engines in ragdoll-pipeline.

## Result

Implemented ragdoll-pipeline with lock-free GOP-parallel work-stealing scheduler, declarative TOML recipe runner, and unified multi-container media engines.
