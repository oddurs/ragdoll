---
id: 44
uid: 9611eeb5-8e85-41d1-87ef-6e45c90003c6
title: Pure-Rust AAC-LC & Opus MDCT/IMDCT Audio Decoder and Encoder
type: feature
status: done
milestone: m8-audio-pro
assignee: Oddur Sigurdsson
depends_on:
- 42
created: 2026-10-25
updated: 2026-10-25
closed_at: 2026-10-25
priority: p0
area: codec-audio
effort: xl
---

## Problem

Standard `.mp4` and `.ts` streams carry **AAC-LC (`mp4a.40.2`)** audio, while `.webm` and `.mkv` streams carry **Opus** (or Vorbis/FLAC). In `v0.1.0`, Ragdoll packetizes AAC ADTS headers for stream copy but cannot decode or encode AAC-LC or Opus audio in pure Rust.

## Architecture & SIMD Design

1. Implement a shared, SIMD-accelerated **Modified Discrete Cosine Transform (MDCT / IMDCT)** and Sine / Kaiser-Bessel-Derived (KBD) windowing engine in `ragdoll-simd` (`N = 256, 2048` for AAC-LC; `N = 120, 240, 480, 960` for Opus CELT) using Split-Radix / Cooley-Tukey pre/post-twiddle FFT kernels.
2. **Pure-Rust AAC-LC Decoder & Encoder (`crates/ragdoll-codec/src/aac.rs`)**:
   - Parse `AudioSpecificConfig`, `ICS` (Individual Channel Stream), scale factor bands (`sfb`), Noiseless Huffman spectral codebooks (`1..11`), M/S and Intensity stereo, TNS (Temporal Noise Shaping), and IMDCT overlap-add synthesis (`libavcodec/aacdec_template.c` & `aacenc.c` reference).
   - Psychoacoustic scalefactor + dead-zone quantizer and two-loop rate control for AAC-LC encoding (`128–256 kbps` stereo).
3. **Pure-Rust Opus CELT/Hybrid Packet Decoder & Encoder (`crates/ragdoll-codec/src/opus.rs`)**:
   - RFC 6716 range coder, PVQ (Pyramid Vector Quantization) algebraic pulse search, band energy quantization, and 48 kHz MDCT synthesis/analysis.

## Acceptance criteria

- [x] `AacDecoder` and `AacEncoder` round-trip 48 kHz stereo audio and interoperate bidirectionally with `ffmpeg -c:a aac` in `.mp4` and `.ts` containers.
- [x] `OpusDecoder` and `OpusEncoder` round-trip 48 kHz stereo audio in `.mkv`/`.webm` and `.ogg` containers and interoperate with `ffmpeg -c:a libopus`.

## 2026-10-25

Implemented pure-Rust ISO/IEC 14496-3 AAC-LC decoder and encoder (AacDecoder, AacEncoder, AudioSpecificConfig, AdtsHeader, 12 canonical Huffman codebooks HCB_1..HCB_11 + HCB_SF, PNS, Mid-Side/Intensity Stereo, KBD/Sine MDCT/IMDCT) and RFC 6716 Opus decoder and encoder (OpusDecoder, OpusEncoder, OpusHead, OpusRangeEncoder/Decoder, CELT 21-band PVQ algebraic pulse search and normalization) in crates/ragdoll-codec/src/aac.rs and opus.rs.
