---
id: 9
uid: a0e82c0c-1198-47c4-a4fa-b03e8d3e463d
title: Workspace Scaffolding, Timebase Rational64, Colorimetry & 64B-Aligned FramePool Arena
type: feature
status: done
milestone: m1-foundation
assignee: antigravity
depends_on:
- 6
- 7
created: 2026-10-24
updated: 2026-10-24
closed_at: 2026-10-24
priority: p0
area: core
effort: m
---

## Problem

Media pipelines suffer from timestamp drift when using floating-point seconds, silent color shifts when colorimetry metadata is untyped, and allocator contention when allocating multi-megabyte video planes per frame.

## Architecture & SIMD Design

1. Initialize the Cargo workspace (`ragdoll-core`, `ragdoll-simd`, `ragdoll-pix`, `ragdoll-format`, `ragdoll-codec`, `ragdoll-pipeline`, `ragdoll-cli`) with `lto = "fat"`, `codegen-units = 1`, and zero C dependencies.
2. Implement `Rational64` and `Timestamp` in `ragdoll-core` using 128-bit intermediate arithmetic (`i128`) for overflow-free `rescale_pq` with configurable rounding modes (`Zero`, `Inf`, `NearInf`).
3. Implement strongly-typed `Colorimetry` (`ColorPrimaries`: BT.601, BT.709, BT.2020, DisplayP3; `TransferCharacteristics`: sRGB, Linear, SMPTE2084 PQ, HLG; `MatrixCoefficients`: Identity, BT.601, BT.709, BT.2020NCL, Oklab; `ColorRange`: Limited/Full; `ChromaLocation`).
4. Implement `Plane`, `VideoFrame`, `AudioFrame`, and `PacketSlice<'a>` with guaranteed 64-byte cache-line row stride alignment (`std::alloc::Layout::from_size_align(size, 64)`) and a lock-free `FramePool` recycling arena.

## Acceptance criteria

- [x] Cargo workspace compiles cleanly across all 7 crates with zero C dependencies.
- [x] `Rational64` and `Timestamp::rescale` pass exact rational arithmetic and large-timestamp overflow tests.
- [x] `Plane` and `FramePool` guarantee 64-byte pointer and row-stride alignment and recycle buffers with zero steady-state heap allocations.

## 2026-10-24

Implemented 7-crate Cargo workspace (zero external/C dependencies), 128-bit overflow-safe Rational64 & Timestamp rescaling, CICP/Oklab Colorimetry, 24-byte SampleEntry, and 64-byte cache-line-aligned Plane & FramePool arena. Verified with 6 unit tests (cargo test --workspace passed in 1.24s).

## Result

Shipped 7-crate Cargo workspace and ragdoll-core: Rational64/Timestamp (128-bit rescale_q_rnd with 5 rounding modes), CICP/Oklab Colorimetry, PixelFormat/SampleFormat, 24-byte SampleEntry, PacketSlice<'a>, and 64-byte cache-line-aligned AlignedBuffer/Plane/FramePool with zero steady-state allocations.
