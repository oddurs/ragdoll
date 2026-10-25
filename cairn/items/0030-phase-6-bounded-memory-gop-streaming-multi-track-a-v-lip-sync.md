---
id: 30
uid: afd0db62-a2b4-4de3-8bc6-f6fad4c63abf
key: m6-streaming-av-sync
title: Phase 6 — Bounded-Memory GOP Streaming & Multi-Track A/V Lip-Sync
type: milestone
status: planned
created: 2026-10-25
updated: 2026-10-25
priority: p0
area: pipeline
---

Replace full-file in-memory frame buffering with a bounded-memory GOP ring streaming engine (`GopRingPipeline`) supporting arbitrary-length feature films, Unix `stdin`/`stdout` pipes, and synchronized multi-track video + audio + subtitle (`WebVTT`/`SRT`) demuxing/muxing with 128-bit `rescale_delta` lip-sync alignment.

## Release gate

2-hour 4K multi-track media files and `stdin`/`stdout` pipe streams transcode and remux within a hard `< 256 MiB` memory ceiling (`< 64 MiB` in `--battery-saver` mode) while preserving all video, audio, and subtitle tracks with zero PTS/DTS drift.
