---
id: 87
uid: 10ab7cba-0730-40c7-bd33-df18e93bb141
title: Opus decoder slower than FFmpeg's native decoder (0.58x CELT, 0.69x SILK)
type: kernel
status: doing
milestone: m11-validation
assignee: Claude
claimed: 2026-10-28
created: 2026-10-28
updated: 2026-10-28
priority: p1
---

## Mathematical Specification

## Register & Cache Layout (NEON / AVX2)

## Acceptance criteria

- [ ] Scalar reference and NEON/AVX2 SIMD implementations pass differential bit-exactness tests across odd dimensions.
- [ ] Zero heap allocations in the hot loop.
