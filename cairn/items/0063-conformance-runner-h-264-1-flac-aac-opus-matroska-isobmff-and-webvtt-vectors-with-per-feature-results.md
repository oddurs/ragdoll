---
id: 63
uid: 98962c0c-db60-429c-af7f-f3688a6ffcf9
title: 'Conformance runner: H.264.1, FLAC, AAC, Opus, Matroska, ISOBMFF and WebVTT vectors with per-feature results'
type: feature
status: backlog
milestone: m11-validation
depends_on:
- 62
created: 2026-10-25
updated: 2026-10-25
priority: p1
effort: l
area: arch
---

## Problem

No official conformance bitstream is decoded anywhere in the test suite, so spec compliance is unmeasured.

## Architecture & SIMD Design

`qa conformance` decodes each vector and compares against the reference output hash shipped with the suite: ITU-T H.264.1 (JVT) bitstreams with reference YUV MD5s; the FLAC test suite; ISO/IEC 14496-26 AAC vectors; RFC 8251 Opus test vectors (using the RFC's quality-comparison tool); the Matroska test files; GPAC ISOBMFF conformance files; and W3C WebVTT tests. Each vector is tagged with the spec features it exercises (for example `h264.baseline.cavlc`, `h264.high.cabac.8x8`, `mkv.lacing.ebml`), so results roll up per feature.

## Acceptance criteria

- [ ] Every vector in each suite runs and reports pass, fail or unsupported, with a reason.
- [ ] Results are emitted as JSON keyed by feature tag.
- [ ] A conformance smoke subset runs on PRs in under two minutes; the full set runs on merge to `main`.
