---
id: 37
uid: 369795ba-f9d7-4431-933a-5377f4b5b441
title: Multi-Track A/V + Subtitle (WebVTT/SRT) Demuxing, Muxing & 128-Bit Lip-Sync Engine
type: feature
status: done
milestone: m6-streaming-av-sync
assignee: Oddur Sigurdsson
depends_on:
- 35
- 36
created: 2026-10-25
updated: 2026-10-25
closed_at: 2026-10-25
priority: p0
area: format
effort: l
---

## Problem

Real-world media files contain interleaved video, multiple audio tracks (e.g. stereo + surround or multi-language), and subtitle tracks (`WebVTT` / `SRT`). In `v0.1.0`, `transcode_file` handles either video-only or audio-only files, dropping audio tracks when transcoding video.

## Architecture & SIMD Design

1. Upgrade `Mp4Demuxer`, `Mp4Muxer`, `MkvDemuxer`, `MkvMuxer`, `MpegTsDemuxer`, and `MpegTsMuxer` to demux and mux arbitrary combinations of Video, Audio, and Subtitle (`CodecId::WebVtt`, `CodecId::SubRip`) tracks in a single interleaved DTS-ordered pass.
2. Implement `AvSyncController` in `ragdoll-pipeline` using `Timestamp::rescale_delta` and `Timestamp::compare_ts` (`ragdoll-core`) to maintain sub-sample audio/video lip-sync across container timebase changes (`90 kHz` MPEG-TS <-> `48 kHz` audio <-> `1000 Hz` MKV).
3. Add CLI track-mapping ergonomics (`--audio copy|aac|opus|flac|none`, `--subs copy|vtt|srt|none`, `--track <SPEC>`) to `ragdoll transcode` and `ragdoll remux`.

## Acceptance criteria

- [x] Transcoding or remuxing a multi-track `.mp4`, `.mkv`, or `.ts` preserves both video and audio tracks with `< 1 ms` A/V PTS drift.
- [x] `WebVTT` (`.vtt`) and `SubRip` (`.srt`) subtitle tracks can be muxed, demuxed, and converted between `.mp4` (`wvtt`), `.mkv` (`S_TEXT/UTF8` / `S_TEXT/WEBVTT`), and standalone `.vtt`/`.srt` files.
- [x] Interleaved packet ordering satisfies `ffmpeg -v error -i out.mp4 -f null -` with zero DTS/PTS monotonicity warnings.

## 2026-10-25

Implemented AvSyncController (128-bit rescale_delta stateful sub-sample audio rescaling and compare_ts DTS interleaving with <1 ms drift), SubtitleDemuxer/SubtitleMuxer for WebVTT (.vtt) and SubRip (.srt), ISO 14496-30 wvtt + vtte MP4 sample framing, Matroska S_TEXT/WEBVTT + S_TEXT/UTF8 BlockGroup+BlockDuration muxing, and default multi-track preservation.
