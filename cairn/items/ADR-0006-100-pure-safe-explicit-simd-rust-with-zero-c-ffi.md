---
id: 6
uid: 5be2a263-be1d-4fcc-a386-97ba2aadce92
title: 100% Pure Safe + Explicit SIMD Rust with Zero C FFI
type: decision
status: done
milestone: m1-foundation
assignee: antigravity
created: 2026-10-24
updated: 2026-10-24
closed_at: 2026-10-24
priority: p0
area: arch
effort: s
---

## Context

FFmpeg relies on millones of lines of C/assembly accumulated over 25 years, exposing complex pointer-heavy APIs (`AVFormatContext`, `AVCodecContext`, `AVPacket`) and requiring external C libraries (`libx264`, `libdav1d`, `libswscale`). When reimagining FFmpeg in Rust, we must choose between wrapping C libraries via FFI or writing 100% pure Rust with explicit architecture SIMD intrinsics.

## Options & Trade-offs

1. **C FFI Wrapper**: Fast initial codec coverage, but inherits C memory unsafety, build complexity, and FFmpeg's buffer-copying architectural bottlenecks.
2. **100% Pure Safe + Explicit SIMD Rust (Selected)**: Zero C dependencies; every bitstream parser, entropy coder, DCT/Hadamard transform, Lanczos-3 polyphase filter, and container engine is written from first principles in Rust using `std::arch::aarch64::*` (Apple M4 NEON) and `std::arch::x86_64::*` with scalar fallbacks.

## Decision

Ragdoll is 100% Pure Rust with zero C FFI. All hot loops use explicit SIMD intrinsics (`aarch64` NEON and `x86_64` AVX2) backed by bit-exact scalar reference kernels.

## Acceptance criteria

- [x] The architectural invariant, its rationale, and its consequences are recorded.

## 2026-10-24

Established 100% pure Rust + explicit SIMD (aarch64 NEON / x86_64 AVX2) invariant with zero C FFI dependencies across all 7 workspace crates.

## Result

Ragdoll is strictly 100% pure Rust (zero C FFI). All codecs, containers, bitstream readers, and pixel scalers use safe Rust + explicit std::arch SIMD intrinsics (Apple M4 aarch64 NEON and x86_64 AVX2) verified against bit-exact scalar fallbacks.
