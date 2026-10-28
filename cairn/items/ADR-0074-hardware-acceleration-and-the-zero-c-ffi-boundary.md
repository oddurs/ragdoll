---
id: 74
uid: cd1c1476-23b1-420c-a12f-88c72f7c6479
title: Hardware acceleration and the zero-C-FFI boundary
type: decision
status: backlog
depends_on:
- 69
- 72
created: 2026-10-28
updated: 2026-10-28
priority: p3
effort: s
area: arch
---

## Context

ADR-0006 commits Ragdoll to pure safe Rust with explicit SIMD and zero C FFI. The main routes to large further gains on video all cross that line:

- hardware decode and encode (Vulkan Video, VideoToolbox, NVDEC/NVENC, VA-API);
- GPU filtering (Vulkan compute, Metal) and unified-memory sharing (IOSurface, DMA-BUF);
- kernel I/O paths such as io_uring with registered buffers;
- delegating codecs Ragdoll does not implement (dav1d for AV1, libavcodec as a fallback).

An external architecture proposal argues for all of these, with projected speedups of 1.4x to 7x and 40% to 85% lower energy per frame. Those projections are not measured, and some assume codecs Ragdoll does not have (HEVC, AV1 and VP9 decoding).

The decision should not be made on projections. It should follow measured CPU-only numbers from the QA scorecard (`0069`) and a genuinely bounded-memory pipeline (BUG-0072), so the gap a hardware path would close is known.

## Options & Trade-offs

1. **Keep ADR-0006 unchanged.** CPU only. Simplest to audit, fuzz and keep deterministic, but caps throughput and energy on GPU- and ASIC-equipped hosts.
2. **Optional hardware crate.** A feature-gated `ragdoll-hw` crate is allowed to call platform APIs. The default build stays pure Rust, and CI tests every hardware path against the bit-exact software decoder. This keeps the core's guarantees and makes FFI opt-in and isolated.
3. **Revise ADR-0006 broadly.** Allow FFI anywhere, including delegating codecs to C libraries. Fastest route to coverage, but gives up the memory-safety and determinism story that distinguishes Ragdoll from FFmpeg.

## Decision

Open. Decide after `0069` publishes CPU-only throughput, first-frame latency, peak RSS and energy per frame, and after BUG-0072 lands.

## Acceptance criteria

- [ ] The decision records which option was chosen and why, citing measured scorecard numbers rather than projections.
- [ ] If option 2 or 3 is chosen, ADR-0006 is amended and the FFI boundary (crates, features, CI requirements) is stated precisely.
