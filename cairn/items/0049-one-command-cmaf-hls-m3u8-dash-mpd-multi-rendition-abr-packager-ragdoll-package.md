---
id: 49
uid: 40de7a77-ff91-4098-a25c-00480f4927cb
title: One-Command CMAF HLS (m3u8) & DASH (mpd) Multi-Rendition ABR Packager (ragdoll package)
type: feature
status: done
milestone: m9-edit-image-package
assignee: Oddur Sigurdsson
depends_on:
- 37
- 39
- 44
created: 2026-10-25
updated: 2026-10-25
closed_at: 2026-10-25
priority: p1
area: pipeline
effort: l
---

## Problem

Generating an Adaptive Bitrate (ABR) streaming package in `ffmpeg` requires complex multi-output `-filter_complex` graphs, duplicate decoding or intermediate files, and manual playlist coordination. Ragdoll should provide a single-command `ragdoll package` verb that decodes each GOP once, scales to all ABR renditions in memory, and writes ready-to-serve CMAF HLS + DASH directories.

## Architecture & SIMD Design

1. Add `ragdoll package <IN> -o <OUT_DIR> [--ladder 1080p@4M,720p@2.5M,480p@1M,360p@600k] [--segment-duration 4.0]`:
   - Decode each input GOP once in `GopRingPipeline`.
   - Fan out the decoded GOP across the requested resolution ladder (`1080p`, `720p`, `480p`, `360p`) using shared `PolyphaseFilterBank` tables and encode all renditions in parallel with aligned IDR keyframe boundaries.
2. Emit **CMAF Fragmented MP4 (`fMP4`)** `init.mp4` + `seg_NNNN.m4s` segments (shared simultaneously by both HLS and DASH!), RFC 8216 `master.m3u8` + per-rendition `stream_N.m3u8` playlists, ISO/IEC 23009-1 `manifest.mpd` DASH manifest, and an optional `thumbnails.vtt` + `.jpg`/`.png` trick-play sprite sheet.

## Acceptance criteria

- [x] `ragdoll package input.mp4 -o ./abr_out/` generates `master.m3u8`, `manifest.mpd`, CMAF `init.mp4` + `.m4s` segments, and `thumbnails.vtt` in a single pass.
- [x] Generated HLS (`master.m3u8`) and DASH (`manifest.mpd`) streams demux and decode cleanly in `ffmpeg -v error -i master.m3u8 -f null -`.

## 2026-10-25

Implemented one-command CMAF fMP4 HLS (master.m3u8) + DASH (manifest.mpd) + WebVTT trick-play sprite sheet ABR packager (ragdoll package / package_abr).
