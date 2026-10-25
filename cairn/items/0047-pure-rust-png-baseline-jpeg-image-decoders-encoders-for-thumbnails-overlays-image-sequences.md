---
id: 47
uid: aed76b8c-f721-4c2b-9f5b-0118ff74bb8f
title: Pure-Rust PNG & Baseline JPEG Image Decoders/Encoders for Thumbnails, Overlays & Image Sequences
type: feature
status: done
milestone: m9-edit-image-package
assignee: Oddur Sigurdsson
depends_on:
- 35
created: 2026-10-25
updated: 2026-10-25
closed_at: 2026-10-25
priority: p0
area: format
effort: l
---

## Problem

In `v0.1.0`, `ragdoll extract` writes uncompressed `.ppm` / `.pgm` images. Production web, mobile, and studio pipelines require compressed **`.png`** (lossless RGB24/RGBA8 with alpha channel for watermarks and contact sheets) and **`.jpg`** (baseline YUV420/YUV444 DCT for fast web thumbnails) with zero external C (`libpng` / `libjpeg-turbo`) or crate dependencies.

## Architecture & SIMD Design

1. **Pure-Rust PNG Decoder & Encoder (`crates/ragdoll-format/src/png.rs`)**:
   - Implement RFC 1950/1951 Zlib/DEFLATE bitstream compression and decompression (LZ77 sliding window + canonical Huffman + Adler-32 + `ragdoll-core`'s Slicing-by-8 CRC-32) and SIMD-accelerated PNG scanline filters (`None`, `Sub`, `Up`, `Average`, `Paeth` predictor in NEON/AVX2).
2. **Pure-Rust Baseline JPEG (`SOF0`) Decoder & Encoder (`crates/ragdoll-format/src/jpeg.rs`)**:
   - Reuse `ragdoll-simd`'s $8\times8$ integer DCT/IDCT + BT.601 full-range (`JFIF`) color matrix + canonical Huffman tables + zig-zag run-length coding (`FF 00` byte stuffing).
3. Support `.png` and `.jpg` in `ragdoll extract`, `ragdoll probe`, `ragdoll transcode` (including image sequences `frame_%04d.png <-> video.mp4`), and watermark `overlay`.

## Acceptance criteria

- [x] `ragdoll extract input.mp4 -o thumb.png` and `-o sheet.jpg --thumbnails 16 --grid 4x4` produce spec-compliant `.png` and `.jpg` files verified by `ffmpeg` / `ffprobe`.
- [x] Image-sequence to video (`ragdoll transcode frame_%04d.png out.mp4`) and video to image-sequence roundtrips pass bit-exact / PSNR checks.

## 2026-10-25

Implemented pure-Rust PNG (RFC 2083 with Zlib/DEFLATE RFC 1950/1951, crc32_png, adler32, adaptive Paeth filtering) and Baseline DCT (SOF0) JPEG (.jpg/.jpeg) decoders and encoders plus image sequence (frame_%04d.png/.jpg) support in ragdoll-format and ragdoll-pipeline.
