---
id: 25
uid: 50ef48e1-a6c7-4a86-9621-8cd00b887b34
title: Public Repository Setup, DCT Zig-Zag Logo Identity & FFmpeg-Aligned Documentation
type: docs
status: done
milestone: m1-foundation
assignee: antigravity
created: 2026-10-24
updated: 2026-10-24
closed_at: 2026-10-24
priority: p0
area: arch
effort: s
---

Design Ragdoll's visual identity (`assets/logo.svg` and `assets/banner.svg`, paying homage to FFmpeg's iconic 8x8 DCT zig-zag scan pattern combined with a geometric Ragdoll emblem), write an FFmpeg-aligned yet unmistakably modern `README.md` (with `libav*` -> `ragdoll-*` architectural mapping, mathematical deep-dives, and live Apple Silicon M4 benchmarks vs. FFmpeg 9.0.2), add `LICENSE-MIT` / `LICENSE-APACHE` and GitHub Actions CI, and publish the repository publicly to `oddurs/ragdoll`.

## Acceptance criteria

- [x] Vector SVG logo (`assets/logo.svg`) and hero banner (`assets/banner.svg`) created, honoring the 8x8 DCT zig-zag scanline motif in a sleek Ragdoll geometric mark.
- [x] `README.md` documents the `libav*` to `ragdoll-*` architecture mapping, mathematical/SIMD design principles, and head-to-head Apple Silicon M4 benchmark results against FFmpeg 9.0.2.
- [x] Dual `LICENSE-MIT` / `LICENSE-APACHE`, `.gitignore`, and GitHub Actions CI workflow (`.github/workflows/ci.yml`) configured.
- [x] Public GitHub repository `oddurs/ragdoll` created, pushed, and verified.

## 2026-10-24

Created assets/logo.svg and assets/banner.svg featuring the 4x4 DCT zig-zag entropy scan path woven through a geometric Ragdoll cat mark with sapphire diamond eyes. Wrote README.md with libav* -> ragdoll-* mapping and Apple M4 vs FFmpeg 9.0.2 benchmarks, added LICENSE-MIT, LICENSE-APACHE, and .github/workflows/ci.yml.

## Result

Created DCT zig-zag Ragdoll vector identity (`assets/logo.svg`, `assets/banner.svg`), FFmpeg-aligned `README.md` with Apple M4 vs. FFmpeg 9.0.2 benchmarks, dual `LICENSE-MIT` / `LICENSE-APACHE`, `.github/workflows/ci.yml`, and published `oddurs/ragdoll` on GitHub.
