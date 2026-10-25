---
id: 16
uid: 6d591eb3-dec7-41ed-9f36-7eeacbc9eed7
title: Zero-Copy Matroska (MKV / WebM) & MPEG-TS Demuxers and Muxers
type: feature
status: done
milestone: m3-format
assignee: antigravity
depends_on:
- 11
- 15
created: 2026-10-24
updated: 2026-10-24
closed_at: 2026-10-24
priority: p1
area: format
effort: l
---

## Problem

Transmuxing between MP4, MKV/WebM, and MPEG-TS broadcast streams is one of the highest-volume media workloads, yet existing tools copy packet payloads and parse EBML varints and TS sync bytes serially.

## Architecture & SIMD Design

1. **MKV / WebM Engine**:
   - Decode and encode EBML variable-length integers (VINTs) branchlessly using hardware `leading_zeros()` (`CLZ`).
   - Parse `EBML`, `Segment`, `Info`, `Tracks`, `Cues`, and `Cluster` elements, slicing `SimpleBlock` payloads directly from the source `mmap`.
   - Emit spec-compliant Matroska/WebM files with `Tracks`, `Cluster`, `SimpleBlock`, and `Cues` seek tables.
2. **MPEG-TS Engine**:
   - Lock onto 188-byte transport stream packets using `ragdoll-simd`'s NEON `0x47` sync-byte scanner.
   - Parse and emit `PAT` (PID 0), `PMT`, PCR timestamps, and `PES` (Packetized Elementary Stream) headers with automatic AVCC <-> Annex-B bitstream framing.

## Acceptance criteria

- [x] Zero-copy transmuxing `MP4 -> MKV` and `MP4 -> MPEG-TS` completes at gigabyte-per-second throughput.
- [x] Output `.mkv` and `.ts` files play and probe cleanly in `ffprobe` / `ffmpeg`.

## 2026-10-24

Implemented MkvDemuxer & MkvMuxer (mkv.rs) with CLZ EBML VINT parsing, SimpleBlock/BlockGroup (including Xiph/EBML/fixed lacing), Colour CICP metadata, and Cues seek table generation (35.85 GB/s MP4->MKV transmux), plus MpegTsDemuxer & MpegTsMuxer (mpegts.rs) with NEON 0x47 sync-byte lock, Slicing-by-8 crc32_mpeg2 PAT/PMT verification, 33-bit 90kHz PES timestamps, PCR adaptation fields, and automatic H.264 AVCC<->Annex-B framing (11.82 GB/s MP4->TS transmux).

## Result

Shipped zero-copy MkvDemuxer/MkvMuxer and SIMD-synced MpegTsDemuxer/MpegTsMuxer verified with ffprobe/ffmpeg.
