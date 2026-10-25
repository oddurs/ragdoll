---
id: 19
uid: 9c8e910e-2e5f-4f71-b478-989e7bb140c6
title: Pure-Rust H.264 / AVC NAL Engine, SPS/PPS/Slice Parser, 4x4 Integer DCT/Hadamard & Frame Codec
type: feature
status: planned
milestone: m4-codec
depends_on:
- 10
- 11
- 12
created: 2026-10-24
updated: 2026-10-24
priority: p0
area: codec-video
effort: xl
---

## Problem

H.264 (ITU-T H.264 / ISO/IEC 14496-10 AVC) is the most ubiquitous video codec on earth. Supporting pure-Rust H.264 bitstream inspection, Annex-B <-> AVCC conversion, and frame decoding/encoding without `libx264` or `openh264` is a centerpiece of Ragdoll's craftsmanship.

## Architecture & SIMD Design

1. **NAL Framing Engine**: Zero-copy AVCC (4-byte length prefix) <-> Annex-B (`0x00000001` start code) conversion using `ragdoll-simd`'s NEON start-code scanner and RBSP unescaping.
2. **Bitstream Parameter Parsers**: Full parsing and serialization of `SPS` (Sequence Parameter Set: profile, level, macroblock dimensions, VUI colorimetry, frame cropping), `PPS` (Picture Parameter Set), `SEI`, and `SliceHeader`.
3. **Transforms & Quantization (`SIMD`)**: Exact H.264 $4\times4$ integer forward/inverse DCT, $4\times4$ Luma DC Hadamard, and $2\times2$ Chroma DC Hadamard transforms plus QP dequantization/quantization scaling tables.
4. **Prediction & Entropy Coding**: Intra $16\times16$ (Vertical, Horizontal, DC, Plane), Intra $4\times4$, Inter P-macroblock motion compensation, and Context-Adaptive Variable-Length Coding (CAVLC) + PCM macroblock fast-path for valid H.264 Baseline/Main bitstream encoding and decoding.

## Acceptance criteria

- [ ] Parses H.264 SPS, PPS, and Slice headers from real-world and FFmpeg-generated MP4/MKV/TS streams with exact resolution, FPS, profile, and colorimetry extraction.
- [ ] Encodes and decodes valid H.264/AVC video frames in 100% pure Rust that play cleanly in `ffmpeg` / `ffprobe` and decode back in Ragdoll.
