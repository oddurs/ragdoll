---
id: 7
uid: 31c8c132-2f3c-463a-b201-d63841c7d884
title: Zero-Copy Memory-Mapped Demuxing & Fused L1 Scanline Strip Execution
type: decision
status: done
milestone: m1-foundation
assignee: antigravity
created: 2026-10-24
updated: 2026-10-24
closed_at: 2026-10-24
priority: p0
area: arch
effort: s
---

## Context

Two memory bottlenecks dominate traditional media pipelines:
1. **Demuxer Heap Churn**: Allocating and copying `AVPacket` payloads from file I/O buffers into heap memory on every frame.
2. **Filtergraph RAM Round-Trips**: `libswscale` executes color conversion, scaling, and cropping as separate full-frame passes, writing and re-reading 12–48 MB per 4K frame through L3/DRAM.

## Options & Trade-offs

1. **Traditional Frame-at-a-Time Pipeline**: Simple modular pass composition, but saturates DRAM bandwidth on 4K/8K video.
2. **Zero-Copy `PacketSlice<'mmap>` + Fused L1 Scanline Strips + 64B Arena `FramePool` (Selected)**: Demuxers slice directly into kernel `mmap` pages with a flat 24-byte sample index; muxers flush via vectored I/O (`writev`); pixel filters process rolling scanline windows inside L1 cache (32 KB) using 64-byte cache-line-aligned planes.

## Decision

Adopt zero-copy `PacketSlice` views over memory-mapped I/O, 64-byte aligned `FramePool` arenas, and fused L1-cache-resident scanline strip execution across the entire engine.

## Acceptance criteria

- [x] The architectural invariant, its rationale, and its consequences are recorded.

## 2026-10-24

Adopted zero-copy PacketSlice<'mmap> demuxing, flat 24-byte sample index tables, 64-byte cache-line-aligned FramePool arenas, and L1-resident scanline strip execution.

## Result

Input containers memory-map files and emit zero-copy PacketSlice views backed by flat 24-byte sample tables; video planes are 64-byte cache-line-aligned in a reusable FramePool arena; pixel filters fuse color conversion and polyphase scaling into L1-resident scanline strips.
