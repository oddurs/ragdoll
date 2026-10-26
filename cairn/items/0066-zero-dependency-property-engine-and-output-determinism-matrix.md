---
id: 66
uid: 3be04830-521c-49b5-8498-76f0d033c9b6
title: Zero-dependency property engine and output-determinism matrix
type: feature
status: backlog
milestone: m11-validation
created: 2026-10-25
updated: 2026-10-25
priority: p1
effort: m
area: arch
---

## Problem

Round-trip and invariant tests exist but use hand-picked inputs, and nothing checks that output is identical across thread counts, power modes, instruction sets and operating systems.

## Architecture & SIMD Design

A small seeded generator (xoshiro-based, inside the workspace, so there are no new dependencies) with shrinking, used for round trips (bit I/O, RBSP, mux to demux, FLAC) and metamorphic relations (SSIM symmetric, PSNR of identical frames is infinite, timestamp rescale composes, scale then metric is monotonic in filter quality). A determinism matrix hashes outputs across `--threads 1..N`, power modes, NEON / AVX2 / scalar, macOS / Linux, and GOP-parallel vs serial execution.

## Acceptance criteria

- [ ] Property tests run with a fixed seed on PRs and a random seed nightly, and failures print a reproducible seed.
- [ ] The determinism matrix runs on merge to `main` and fails on any hash difference.
