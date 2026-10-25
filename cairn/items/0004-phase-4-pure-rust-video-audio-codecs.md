---
id: 4
uid: b887f676-9c47-47df-8f40-f5d3d825fc90
key: m4-codec
title: Phase 4 — Pure-Rust Video & Audio Codecs
type: milestone
status: planned
created: 2026-10-24
updated: 2026-10-24
priority: p0
area: codec-video
---

Build 100% pure-Rust, zero-FFI video and audio codecs from scratch: H.264/AVC (NAL Annex-B/AVCC SIMD framing, SPS/PPS/Slice headers, CAVLC entropy coding, 4x4 integer DCT/Hadamard transforms, intra/inter prediction), AV1 OBU parser/packetizer, Ragdoll Fast-Intra studio codec, pure-Rust FLAC encoder/decoder (LPC + Levinson-Durbin + SIMD Rice coding), and vectorized PCM.

## Release gate

Video and audio codecs encode and decode bitstreams in pure Rust with zero unsafe memory bugs, full FLAC CRC verification, and clean FFmpeg cross-compatibility.
