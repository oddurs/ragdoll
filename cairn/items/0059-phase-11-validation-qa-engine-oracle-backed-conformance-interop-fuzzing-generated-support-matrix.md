---
id: 59
uid: 47cf4ac9-30bb-41d1-91b4-dc83a3cd9945
key: m11-validation
title: 'Phase 11 — Validation & QA Engine: Oracle-Backed Conformance, Interop, Fuzzing & Generated Support Matrix'
type: milestone
status: planned
created: 2026-10-25
updated: 2026-10-25
priority: p0
area: arch
---

Make every published claim about Ragdoll traceable to an independent oracle. Scalar references gate SIMD; spec conformance streams gate parsers and decoders; reference implementations (FFmpeg, JM, libFLAC, libopus, dav1d, mkvalidator, MP4Box, tsduck) gate interop in both directions; coverage-guided fuzzing gates robustness; rate-distortion, instruction-count and memory gates catch regressions; and the documentation support matrix is generated from QA results instead of written by hand, so the docs cannot overclaim.

## Release gate

- `qa report` produces a JSON scorecard covering conformance, interop, fuzzing, determinism, rate-distortion and performance, and the website renders the Formats & codecs matrix and a public scorecard from it.
- Every feature marked supported has passing conformance and bidirectional interop evidence on `main`.
- PR, merge, nightly and release CI tiers run as specified, with no test that silently passes when a tool is missing.
