---
id: 55
uid: 0c0c490c-6432-4ab9-949d-288c070eb76b
title: AVX2 kernels for x86_64
type: kernel
status: backlog
milestone: m10-production-v1
created: 2026-10-25
updated: 2026-10-25
priority: p1
effort: xl
area: simd
---

## Mathematical Specification

`SimdLevel::detect()` reports AVX2 on capable x86_64 hosts, but no `#[target_feature(enable = "avx2")]` kernels exist: every public kernel falls back to its `_scalar` reference on x86_64. The README previously claimed AVX2 support. Each NEON kernel already has a scalar reference and differential tests, so each port has an exact target.

## Register & Cache Layout (NEON / AVX2)

Start with the highest-traffic kernels: `find_annex_b_start_code`, `unescape_rbsp_into`, `sad_plane_u8` / `sse_plane_u8`, `ssim_4x4xn_u8`, the YUV/RGB matrices, and the Q14/Q6 polyphase scaler passes. Dispatch at runtime with `is_x86_feature_detected!("avx2")`, cached.

## Acceptance criteria

- [ ] Scalar reference and NEON/AVX2 SIMD implementations pass differential bit-exactness tests across odd dimensions.
- [ ] Zero heap allocations in the hot loop.
- [ ] CI gains an x86_64 job that runs the differential tests with AVX2 enabled.
- [ ] `self-test` reports AVX2 throughput for every ported kernel.
