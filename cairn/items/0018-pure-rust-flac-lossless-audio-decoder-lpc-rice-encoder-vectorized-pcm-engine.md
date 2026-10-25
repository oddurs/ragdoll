---
id: 18
uid: 08461f63-4769-46c7-b707-d7a2f7c82a7b
title: Pure-Rust FLAC Lossless Audio Decoder, LPC / Rice Encoder & Vectorized PCM Engine
type: feature
status: planned
milestone: m4-codec
depends_on:
- 10
- 11
created: 2026-10-24
updated: 2026-10-24
priority: p0
area: codec-audio
effort: l
---

## Problem

Audio transcoding between uncompressed PCM (`WAV`) and lossless `FLAC` requires fast linear predictive coding (LPC), stereo decorrelation, and Golomb-Rice entropy coding without relying on `libFLAC`.

## Architecture & SIMD Design

1. **Vectorized PCM Engine**: SIMD conversion between `u8`, `s16le`, `s24le`, `s32le`, and `f32le` plus stereo/multi-channel matrixing.
2. **Pure-Rust FLAC Decoder**:
   - Parse `fLaC` streammarker, `STREAMINFO` (MD5, sample rate, channels, bit depth), `SEEKTABLE`, and frame headers (UTF-8 coded sample/frame numbers, CRC-8).
   - Decode Constant, Verbatim, Fixed Linear Predictor (orders 0..=4), and FIR Linear Predictive Coding (LPC orders 1..=32) subframes with Rice/Rice2 residual partitions.
   - Reconstruct Independent, Left-Side, Right-Side, and Mid-Side stereo channels and verify frame CRC-16.
3. **Pure-Rust FLAC Encoder**:
   - Compute Mid-Side vs Independent channel energy via SIMD.
   - Evaluate Fixed Predictors (orders 0..=4) and windowed autocorrelation + Levinson-Durbin LPC recursion to pick the optimal predictor order and Rice parameter $k$.
   - Emit spec-compliant `.flac` streams verified by both Ragdoll and `ffmpeg`.

## Acceptance criteria

- [ ] Pure-Rust FLAC encoder and decoder round-trip 16-bit and 24-bit audio bit-identically (0 LSB error) with valid CRC-8 and CRC-16 checksums.
- [ ] `.flac` files encoded by Ragdoll decode bit-identically in `ffmpeg`, and `.flac` files encoded by `ffmpeg` decode bit-identically in Ragdoll.
