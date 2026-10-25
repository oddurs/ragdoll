---
id: 32
uid: 76edf7d3-947b-47ef-804f-ebd0239d3dd5
key: m8-audio-pro
title: Phase 8 — Lossy Audio (AAC-LC & Opus), Polyphase Resampler & EBU R128 Loudness
type: milestone
status: planned
created: 2026-10-25
updated: 2026-10-25
priority: p0
area: codec-audio
---

Deliver a complete pure-Rust production audio engine: SIMD windowed-sinc Kaiser (`bessel_i0`) polyphase sample-rate conversion (`44.1/48/88.2/96 kHz`), channel matrix mixing (`5.1/7.1 <-> Stereo/Mono`), EBU R128 (`LUFS`) + True-Peak loudness measurement and normalization, pure-Rust AAC-LC and Opus MDCT/IMDCT lossy audio codecs, and standalone `.flac` / `.ogg` container support with `SEEKTABLE` and metadata tags.

## Release gate

Standard `.mp4` (H.264 + AAC-LC) and `.webm`/`.mkv` (Video + Opus/FLAC) files transcode with synchronized audio resampling and optional `--loudness -14lufs` EBU R128 normalization, verified against `ffmpeg` and `ffprobe`.
