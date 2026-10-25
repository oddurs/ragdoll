---
id: 1
uid: db4edd32-bfab-4478-a2f6-33400578419d
key: m1-foundation
title: Phase 1 — Foundation, Timebases, Bitstreams & SIMD Primitives
type: milestone
status: done
created: 2026-10-24
updated: 2026-10-24
closed_at: 2026-10-24
priority: p0
area: core
---

Establish the Cargo workspace, 128-bit overflow-safe Rational64 timebases, first-class colorimetry types, 64-byte cache-line-aligned frame arenas, branchless 64-bit CLZ bitstream readers/writers, and the Apple Silicon NEON / x86_64 AVX2 SIMD dispatch layer.

## Release gate

All foundational primitives in `ragdoll-core` and `ragdoll-simd` compile cleanly with zero C dependencies, pass bit-exact property tests, and sustain multi-gigabit bitstream throughput on Apple M4.

## Result

Completed Milestone 1 (M1 — Foundation, Memory & SIMD Runtime), production-caliber reference hardening (`SIMD-0024`), and public repository & brand identity setup (`0025`) across all 7 workspace crates.
