---
id: 28
uid: 6a35023b-e5a4-4a3d-953a-de8c9b2d7c1a
title: Reference-Driven M3 Container Hardening (libavformat, libwebm & ISOBMFF/Matroska/MPEG-TS/RF64 Audit)
type: feature
status: done
milestone: m3-format
assignee: antigravity
depends_on:
- 15
- 16
- 17
created: 2026-10-24
updated: 2026-10-24
closed_at: 2026-10-24
priority: p0
effort: l
area: format
part_of:
- 3
---

## Problem

Production container workflows encounter fragmented streaming MP4s (`fMP4` / CMAF `moof`+`traf`+`trun`), compact `stz2` sample tables, Matroska files requiring `SeekHead` (`0x114D9B74`) meta-indexing and $O(\log N)$ `Cues` seeking, 192-byte Blu-ray `M2TS` / 204-byte DVB FEC transport streams, ADTS AAC framing when transmuxing between `MP4`/`MKV` and `MPEG-TS`, and 64-bit `RF64`/`BW64` (`ds64`) + IEEE-Float `fact` chunks in studio WAVE files.

## Architecture & SIMD Design

1. **ISOBMFF / MP4 & Fragmented `fMP4` (`mp4.rs` — `libavformat/mov.c`, `movenc.c` audit)**:
   - Parse `mvex`/`trex` and `moof` -> `traf` -> `tfhd`, `tfdt` (v0/v1), and `trun` (including ISO 14496-12 `sample_flags` keyframe detection) into flat 24-byte `SampleEntry` tables.
   - Add `Mp4Muxer::finish_fragmented` to emit CMAF/DASH/MSE-compliant Fragmented MP4 (`ftyp` + init `moov` with `mvex`/`trex` + per-GOP `moof`/`mdat` fragments).
   - Parse compact 4-bit, 8-bit, and 16-bit `stz2` sample size tables.
2. **Matroska / WebM (`mkv.rs` — `libavformat/matroskadec.c`, `matroskaenc.c`, `libwebm` audit)**:
   - Emit deterministic `SeekHead` (`0x114D9B74`) at the start of `Segment` indexing `Info`, `Tracks`, and `Cues` so demuxers locate `Cues` in $O(1)$ without scanning clusters.
   - Parse `Cues` and implement $O(\log N)$ `MkvDemuxer::seek_to_timestamp`, plus `CodecDelay` (`0x56AA`) and `SeekPreRoll` (`0x56BB`) for Opus/AAC.
3. **MPEG-TS / M2TS (`mpegts.rs` — `libavformat/mpegts.c`, `mpegtsenc.c`, `adtsenc.c` audit)**:
   - Auto-detect 188-byte broadcast TS, 192-byte `M2TS` (`TP_extra_header`), and 204-byte DVB FEC packet strides.
   - Emit `SDT` (`PID 0x0011`, `table_id = 0x42`) and PMT registration descriptors (`Opus`, `fLaC`, `AV01`, `HEVC`).
   - Implement bidirectional ADTS AAC framing (`0xFFF1...` 7-byte header $\leftrightarrow$ raw AAC + `AudioSpecificConfig`) and H.264 Annex-B $\to$ AVCC conversion (`transmux_mpegts_to_mp4`, `transmux_mpegts_to_mkv`).
4. **RF64 / BW64 WAVE & Y4M (`wav.rs`, `y4m.rs` — `libavformat/wavdec.c`, `wavenc.c`, `yuv4mpegdec.c` audit)**:
   - Support 64-bit `RF64`/`BW64` (`ds64` chunk) demuxing and muxing (`WavMuxer::finish_rf64`), plus mandatory `fact` chunk emission on `F32le` IEEE Float WAVE files and `XYSCSS=` legacy Y4M tags.

## Acceptance criteria

- [x] Fragmented MP4 (`fMP4` `moof`/`traf`/`tfhd`/`tfdt`/`trun`) and `stz2` demuxing + `Mp4Muxer::finish_fragmented` implemented and verified with `ffprobe`
- [x] Matroska `SeekHead` (`0x114D9B74`) meta-seek emission, `CodecDelay`/`SeekPreRoll`, and $O(\log N)$ `MkvDemuxer::seek_to_timestamp` implemented and verified with `ffprobe`
- [x] 188/192/204-byte MPEG-TS/M2TS stride auto-detection, `SDT` (`PID 0x0011`), registration descriptors, ADTS AAC framing, and bidirectional `MPEG-TS -> MP4 / MKV` transmuxing implemented and verified with `ffprobe`
- [x] 64-bit `RF64`/`BW64` (`ds64`) WAVE demuxing/muxing, IEEE-Float `fact` chunk emission, and `XYSCSS=` Y4M fallback implemented and verified with `ffprobe`

## 2026-10-24

Completed reference-driven container hardening against FFmpeg 9.0.2 (libavformat/mov.c, movenc.c, matroskadec.c, matroskaenc.c, mpegts.c, mpegtsenc.c, adtsenc.c, wavdec.c, wavenc.c, yuv4mpegdec.c) and libwebm (mkvmuxer/mkvparser): added fMP4 (moof/traf/tfhd/tfdt/trun + mvex/trex) demuxing and finish_fragmented muxing, stz2 compact sample size tables, deterministic Matroska SeekHead (0x114D9B74) + O(log N) MkvDemuxer::seek_to_timestamp + Opus CodecDelay/SeekPreRoll, 188B/192B Blu-ray M2TS/204B FEC stride auto-detection + DVB SDT (0x0011) + PMT 0x05 registration descriptors + bidirectional ADTS AAC & H.264 Annex-B <-> AVCC transmuxing (transmux_mpegts_to_mp4, transmux_mpegts_to_mkv), and 64-bit RF64/BW64 WAVE (ds64) + IEEE Float fact chunk + XYSCSS= Y4M tags.
