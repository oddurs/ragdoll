---
id: 40
uid: 19523f68-ccf0-42f9-9c0d-60d7b1b9e159
title: H.264 Main & High Profile CABAC Arithmetic Engine, 8x8 Transform & B-Frame Decoding
type: feature
status: planned
milestone: m7-video-production
depends_on:
- 39
created: 2026-10-25
updated: 2026-10-25
priority: p0
area: codec-video
effort: xl
---

## Problem

Over 90% of H.264 MP4/MKV/TS files in the wild (iPhone recordings, YouTube/Twitch streams, OBS captures, Blu-ray) use **Main Profile (`profile_idc = 77`)** or **High Profile (`profile_idc = 100`)** with **CABAC (`entropy_coding_mode_flag = 1`)**, $8\times8$ integer DCT (`transform_8x8_mode_flag = 1`), and B-slices (`B_Skip`, `B_Direct_16x16`, `B_L0_L1_16x16`). Without CABAC and High Profile support, Ragdoll cannot decode standard user-supplied MP4 files.

## Architecture & SIMD Design

1. Implement ITU-T H.264 9.3 **CABAC Arithmetic Decoder & Encoder** (`CabacReader` / `CabacWriter` with the 460-entry context state initialization table `cabac_context_init_I` / `cabac_context_init_PB`, range renormalization, `decode_decision`, `decode_bypass`, `decode_terminate`) modeled on `libavcodec/h264_cabac.c`.
2. Implement High Profile (`profile_idc = 100`) `SPS`/`PPS` extensions (`chroma_format_idc`, `bit_depth_luma_minus8`, `transform_8x8_mode_flag`, `pic_scaling_matrix_present_flag`) and $8\times8$ integer forward/inverse DCT (`forward_dct8x8` / `inverse_idct8x8` in `ragdoll-simd`).
3. Implement multi-reference DPB (Decoded Picture Buffer) reordering and B-slice (`slice_type = 1 | 6`) bi-predictive motion compensation (`L0` + `L1` weighted/averaged prediction).

## Acceptance criteria

- [ ] `H264Decoder` decodes Main Profile and High Profile CABAC H.264 streams (including B-frames and $8\times8$ transforms) produced by `ffmpeg -c:v libx264 -profile:v high`.
- [ ] `H264Encoder` supports `--profile baseline|main|high` with CABAC entropy coding reducing bitrate by `10%–18%` at identical PSNR/SSIM compared to CAVLC.
- [ ] DPB presentation-order sorting emits frames in monotonic PTS order across B-frame pyramids.
