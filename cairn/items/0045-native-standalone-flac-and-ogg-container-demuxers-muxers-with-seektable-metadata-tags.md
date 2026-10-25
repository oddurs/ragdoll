---
id: 45
uid: f3a2d589-3f59-4553-a294-b6b67ec22978
title: Native Standalone .flac and .ogg Container Demuxers/Muxers with SEEKTABLE & Metadata Tags
type: feature
status: planned
milestone: m8-audio-pro
depends_on:
- 37
created: 2026-10-25
updated: 2026-10-25
priority: p1
area: format
effort: m
---

## Problem

In `v0.1.0`, `ragdoll transcode` reads/writes raw `.flac` streams via `FlacEncoder`/`FlacDecoder`, but `ragdoll probe` and `ragdoll remux` do not treat `.flac` or `.ogg` (Ogg FLAC / Ogg Opus) as first-class indexed containers with $O(\log N)$ sample seeking (`SEEKTABLE`) and Vorbis Comment metadata tags (`TITLE`, `ARTIST`, `ALBUM`, `REPLAYGAIN_*`).

## Architecture & SIMD Design

1. Add `FlacContainerDemuxer` and `FlacContainerMuxer` in `ragdoll-format::flac` supporting RFC 9639 metadata blocks (`STREAMINFO`, `SEEKTABLE` with 18-byte seekpoints, `VORBIS_COMMENT`, `PICTURE`, `PADDING`) and `SampleEntry` frame indexing.
2. Add `OggDemuxer` and `OggMuxer` in `ragdoll-format::ogg` using `ragdoll-core`'s Slicing-by-8 CRC-32 (`0x04C11DB7`) for Ogg page framing (`OggS`) encapsulating Opus (`OpusHead`/`OpusTags`) and FLAC packets.

## Acceptance criteria

- [ ] `ragdoll probe`, `ragdoll remux`, and `ragdoll transcode` inspect and convert `.flac` and `.ogg` files with `SEEKTABLE` and Vorbis Comment metadata preservation.
- [ ] `ffprobe` and `ffmpeg` validate all generated `.flac` and `.ogg` files with zero warnings.
