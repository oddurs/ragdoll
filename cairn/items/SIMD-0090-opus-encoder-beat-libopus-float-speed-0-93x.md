---
id: 90
uid: 2daf4b98-4751-4006-9057-6e961986664a
title: 'Opus encoder: beat libopus float speed (0.93x)'
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
