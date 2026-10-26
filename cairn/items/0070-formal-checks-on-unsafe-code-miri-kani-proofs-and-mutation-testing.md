---
id: 70
uid: 417c9003-89d3-4e16-8cc8-a7b91fb97007
title: 'Formal checks on unsafe code: Miri, Kani proofs and mutation testing'
type: chore
status: backlog
milestone: m11-validation
created: 2026-10-25
updated: 2026-10-25
priority: p3
effort: m
area: arch
---

## Problem

`unsafe` is confined to intrinsics, `AlignedBuffer` and OS calls, but nothing mechanically checks that it is sound, and nothing checks that the tests would catch a wrong change.

## Acceptance criteria

- [ ] Miri runs on `AlignedBuffer`, `FramePool` and the scalar reference paths in the merge tier.
- [ ] Kani proofs show `BitReader` never reads out of bounds and kernel row-slicing never exceeds plane bounds, for bounded sizes.
- [ ] `cargo-mutants` runs on `ragdoll-core` time and bitstream modules at release, and surviving mutants are triaged.
