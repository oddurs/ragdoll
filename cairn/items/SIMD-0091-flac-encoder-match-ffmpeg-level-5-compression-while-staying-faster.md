---
id: 91
uid: 0b3d405c-48b7-4f73-a354-491304c8de38
title: 'FLAC encoder: match FFmpeg level 5 compression while staying faster'
type: kernel
status: backlog
milestone: m11-validation
created: 2026-10-28
updated: 2026-10-28
priority: p1
---

## Mathematical Specification

## Register & Cache Layout (NEON / AVX2)

## Acceptance criteria

- [ ] Scalar reference and NEON/AVX2 SIMD implementations pass differential bit-exactness tests across odd dimensions.
- [ ] Zero heap allocations in the hot loop.
