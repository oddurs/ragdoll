---
id: 89
uid: 2eed7656-dbba-4877-9bc2-fd321e8cb690
title: H.264 encoder slower than x264 at equal compression (0.12x ultrafast)
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
