---
id: 3
uid: b1dea117-0584-4527-a89b-e06037fa97a1
key: m3-format
title: Phase 3 — Zero-Copy Memory-Mapped Containers & Vectored Muxers
type: milestone
status: done
assignee: antigravity
created: 2026-10-24
updated: 2026-10-24
closed_at: 2026-10-24
priority: p0
area: format
---

Implement zero-copy demuxing over memory-mapped files (`PacketSlice<'mmap>`) and vectored I/O (`writev`) muxing for MP4/ISOBMFF (with flat 24-byte sample tables and O(log N) keyframe seeking), MKV/WebM (SIMD EBML varints), MPEG-TS (16-byte NEON 0x47 sync-byte scanner), Y4M, PPM, and WAV.

## Release gate

Remuxing and probing MP4, MKV, MPEG-TS, Y4M, and WAV execute with zero payload heap allocations per packet and produce spec-compliant files verified by ffprobe/ffmpeg interop.

## Result

Completed m3-format milestone: zero-copy MmapSource demuxers and VectoredWriter muxers for MP4/ISOBMFF, MKV/WebM, MPEG-TS, Y4M, PPM/PGM, and WAV.
