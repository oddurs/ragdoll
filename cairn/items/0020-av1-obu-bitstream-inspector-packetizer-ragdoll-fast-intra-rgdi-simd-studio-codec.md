---
id: 20
uid: 1ade4601-36b5-4827-aed2-4f6060b14d34
title: AV1 OBU Bitstream Inspector/Packetizer & Ragdoll Fast-Intra (RGDI) SIMD Studio Codec
type: feature
status: planned
milestone: m4-codec
depends_on:
- 10
- 11
created: 2026-10-24
updated: 2026-10-24
priority: p1
area: codec-video
effort: m
---

## Problem

Modern streaming relies on AV1 (Open Bitstream Units) for delivery and ultra-fast intra-frame codecs (like ProRes / QOI-video) for multi-gigabyte-per-second intermediate editing and high-speed transcoding.

## Architecture & SIMD Design

1. **AV1 OBU Engine**:
   - Parse LEB128 sizes, `OBU_SEQUENCE_HEADER` (operating points, profile, max dimensions, bit depth, CICP color description), `OBU_TEMPORAL_DELIMITER`, `OBU_FRAME_header`, and `OBU_FRAME` / `OBU_TILE_GROUP`.
   - Support zero-copy AV1 `av1C` box generation and MP4/MKV/IVF remuxing.
2. **Ragdoll Fast-Intra (`rgdi`) Studio Codec**:
   - 8x8 / 4x4 SIMD integer DCT + dead-zone quantization + branchless Rice/Run-Length entropy coder operating independently on slices across all CPU cores.
   - Sustains > 1,000 fps 1080p and > 350 fps 4K encoding/decoding on Apple M4.

## Acceptance criteria

- [ ] AV1 OBU parser extracts exact sequence header parameters and remuxes AV1 streams cleanly across containers.
- [ ] Ragdoll Fast-Intra encodes and decodes 4K video at > 300 fps on Apple M4 with high PSNR (> 42 dB at default quality).
