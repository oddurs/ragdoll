---
id: 15
uid: 339fe131-83d0-45fc-a3eb-bea510c17c8e
title: Zero-Copy MP4 / ISOBMFF Demuxer, Flat 24B Sample Index, O(log N) Seeker & Faststart Muxer
type: feature
status: planned
milestone: m3-format
depends_on:
- 9
- 10
created: 2026-10-24
updated: 2026-10-24
priority: p0
area: format
effort: l
---

## Problem

FFmpeg's MP4 demuxer walks nested box structures and allocates heap `AVPacket` buffers on every sample read, while non-faststart MP4 muxers write `moov` at the end of the file, breaking web streaming.

## Architecture & SIMD Design

1. Memory-map input MP4/MOV files via `MmapSource` and parse the ISOBMFF box hierarchy (`ftyp`, `moov`, `mvhd`, `trak`, `tkhd`, `mdia`, `mdhd`, `hdlr`, `minf`, `stbl`, `stsd`, `avc1`/`avcC`, `hvc1`, `av01`, `mp4a`, `fLaC`, `stts`, `ctts`, `stsc`, `stsz`, `stco`, `co64`, `stss`, `mdat`).
2. Compile the fragmented sample tables (`stts`, `ctts`, `stsc`, `stsz`, `stco`/`co64`, `stss`) into a single cache-packed `Vec<SampleEntry>` (24 bytes per sample: `offset: u64`, `pts: i64`, `size: u32`, `cts_offset_and_flags: i32`).
3. Implement (\log N)$ binary-search keyframe seeking (`seek_to_timestamp`) and zero-copy `PacketSlice<'mmap>` emission.
4. Implement an ISOBMFF MP4 muxer with two-pass or reserved-header **fast-start (`moov` before `mdat`)** and vectored I/O (`writev`).

## Acceptance criteria

- [ ] Demuxes H.264/AV1/AAC/FLAC MP4 files with zero heap allocations per emitted `PacketSlice`.
- [ ] Keyframe seeking resolves any timestamp in (\log N)$ via the flat sample index.
- [ ] Fast-start MP4 files written by Ragdoll pass `ffprobe` validation with `moov` placed before `mdat`.
