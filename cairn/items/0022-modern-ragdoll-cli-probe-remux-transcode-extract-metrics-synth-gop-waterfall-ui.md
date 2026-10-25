---
id: 22
uid: 27ed39a6-73e2-4c6e-9e34-a9c0a273a551
title: 'Modern Ragdoll CLI: probe, remux, transcode, extract, metrics, synth & GOP Waterfall UI'
type: feature
status: planned
milestone: m5-cli-bench
depends_on:
- 8
- 14
- 16
- 17
- 20
- 21
created: 2026-10-24
updated: 2026-10-24
priority: p0
area: cli
effort: l
---

## Problem

Users and automation scripts need a cohesive, fast, self-documenting CLI (`ragdoll`) that exposes every capability of the engine with rich terminal visualizations and zero legacy flag traps.

## Architecture & SIMD Design

Implement `ragdoll-cli` (`ragdoll` binary) with 7 flagship subcommands:
1. `ragdoll probe <FILE> [--gop-map] [--json]`: Sub-millisecond container/codec inspection, colorimetry report, bitrate sparkline, and ASCII GOP waterfall diagram (`I`/`P`/`B` frame size distribution).
2. `ragdoll remux <IN> <OUT> [--faststart]`: Zero-copy `mmap` -> `writev` remuxing across `.mp4`, `.mkv`, `.ts`, `.y4m`, and `.wav`.
3. `ragdoll transcode <IN> <OUT> [--video ...] [--scale ...] [--filter lanczos3|mitchell|bilinear] [--color ...] [--audio ...] [--recipe pipeline.toml]`: Full GOP-parallel video/audio transcoding.
4. `ragdoll extract <IN> [--at <TIME>] [--thumbnails <N> --grid <CxR>] -o <OUT>`: Sub-microsecond keyframe seek + frame or contact-sheet extraction to `.ppm`/`.y4m`.
5. `ragdoll metrics <REF> <DIST> [--json]`: SIMD-accelerated PSNR, SSIM, and VMAF-lite perceptual quality evaluation.
6. `ragdoll synth <OUT> [--resolution 3840x2160] [--fps 60] [--duration 5.0]`: Procedural SIMD test pattern generator (moving zone-plates, SMPTE color bars, audio sine/chirp harmonies).
7. `ragdoll bench [--compare-ffmpeg] [--json]`: Automated benchmark suite runner.

## Acceptance criteria

- [ ] All 7 CLI subcommands (`probe`, `remux`, `transcode`, `extract`, `metrics`, `synth`, `bench`) run cleanly with rich terminal output and `--json` support.
- [ ] End-to-end CLI integration tests pass across MP4, MKV, MPEG-TS, Y4M, PPM, WAV, and FLAC formats.
