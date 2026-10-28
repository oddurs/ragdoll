---
id: 97
uid: 77c63439-3605-48f0-a053-44e2863b05a4
title: H.264 decoder panics on out-of-range slice deblocking offsets
type: bug
status: backlog
milestone: m11-validation
created: 2026-10-28
updated: 2026-10-28
priority: p1
area: codec-video
---

## Problem

`parse_slice_header` in crates/ragdoll-codec/src/h264/dec/slice.rs computes `2 * r.read_se()? as i32` for `slice_alpha_c0_offset_div2` / `slice_beta_offset_div2`. The cast binds before the multiply, so a malformed `se(v)` larger than 2^30 overflows the `i32` multiply and panics in debug builds (`attempt to multiply with overflow`) instead of returning `InvalidSyntax`.

`ragdoll-pipeline` `tests::malformed_bitstream_zero_panic_corpus` hits it once the seed H.264 stream comes from the faster encoder of SIMD-0089 (its mutations then land on this field).

## Fix

Range-check the `i64` before narrowing, e.g. `let a = r.read_se()?; if !(-6..=6).contains(&a) { return Err(..) }` and only then `2 * a as i32`.

## Acceptance criteria

- [ ] Out-of-range deblocking offsets return `InvalidSyntax` without arithmetic overflow.
- [ ] `cargo test -p ragdoll-pipeline --lib malformed_bitstream_zero_panic_corpus` passes with the SIMD-0089 encoder.
