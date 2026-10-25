---
id: 32
uid: 76edf7d3-947b-47ef-804f-ebd0239d3dd5
key: m8-audio-pro
title: Phase 8 — Lossy Audio (AAC-LC & Opus), Polyphase Resampler & EBU R128 Loudness
type: milestone
status: done
assignee: Oddur Sigurdsson
created: 2026-10-25
updated: 2026-10-25
closed_at: 2026-10-25
priority: p0
area: codec-audio
---

Deliver a complete pure-Rust production audio engine: SIMD windowed-sinc Kaiser (`bessel_i0`) polyphase sample-rate conversion (`44.1/48/88.2/96 kHz`), channel matrix mixing (`5.1/7.1 <-> Stereo/Mono`), EBU R128 (`LUFS`) + True-Peak loudness measurement and normalization, pure-Rust AAC-LC and Opus MDCT/IMDCT lossy audio codecs, and standalone `.flac` / `.ogg` container support with `SEEKTABLE` and metadata tags.

## Release gate

Standard `.mp4` (H.264 + AAC-LC) and `.webm`/`.mkv` (Video + Opus/FLAC) files transcode with synchronized audio resampling and optional `--loudness -14lufs` EBU R128 normalization, verified against `ffmpeg` and `ffprobe`.

## 2026-10-25

Completed Milestone 8 (m8-audio-pro): SIMD Kaiser polyphase resampler & channel rematrixer (SIMD-0042), SIMD EBU R128 K-weighting & True-Peak normalizer (SIMD-0043), pure-Rust AAC-LC & Opus MDCT/IMDCT codecs (0044), and native .flac & .ogg container demuxers/muxers (0045).
