---
id: 64
uid: 4c404081-ecc1-4472-ba9e-8f4631d30181
title: Bidirectional interop runner against reference implementations in a pinned container
type: feature
status: backlog
milestone: m11-validation
depends_on:
- 62
created: 2026-10-25
updated: 2026-10-25
priority: p1
effort: l
area: arch
---

## Problem

Interop is currently spot-checked by a few tests that skip when tools are absent. Encoders need an independent decoder as an oracle, and demuxers need an independent parser.

## Architecture & SIMD Design

A pinned container image with FFmpeg, the JM H.264 reference decoder, libFLAC, libopus, dav1d, mkvalidator, MP4Box and tsduck. `qa interop` checks both directions: our decoders on their output (bit-exact where the codec is deterministic), and their decoders and validators on our output, with the invariant that our encoder's reconstruction equals the reference decoder's output. Container checks compare packet tables (offset, size, PTS, DTS, flags) between our demuxers and ffprobe.

## Acceptance criteria

- [ ] For every encoder: our reconstruction is byte-identical to FFmpeg's and JM's decode (H.264), or within the codec's conformance tolerance (AAC, Opus).
- [ ] For every muxer: mkvalidator, MP4Box and tsduck report no errors on our output.
- [ ] For every demuxer: packet tables match ffprobe on the corpus.
- [ ] The image is pinned by digest and rebuilt only deliberately.
