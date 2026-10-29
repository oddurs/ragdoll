---
id: 90
uid: 2daf4b98-4751-4006-9057-6e961986664a
title: 'Opus encoder: beat libopus float speed (0.93x)'
type: kernel
status: done
milestone: m11-validation
assignee: Claude
created: 2026-10-28
updated: 2026-10-29
closed_at: 2026-10-29
priority: p1
---

## Mathematical Specification

## Register & Cache Layout (NEON / AVX2)

## Acceptance criteria

- [x] Scalar reference and NEON/AVX2 SIMD implementations pass differential bit-exactness tests across odd dimensions.
- [x] Zero heap allocations in the hot loop.

## 2026-10-29

Opus encoder (96k stereo restricted-lowdelay, complexity 9, 60 s music48, best of 9 alternating, loaded 4-vCPU VM): before 320 ms vs libopus 1.4 float 315 ms (0.98x) and 1.3.1 fixed 367 ms; after 221 ms (float/new 1.42x, fixed/new 1.66x; 1.45x faster than before). On a quieter run: 195 ms vs float 310 ms (1.59x). Callgrind on a 10 s clip: 765M -> 433M instructions. Bit-exact: the 60 s bitstream is byte-identical to the baseline, encoder goldens unchanged. What changed: AVX2 PVQ search (per-lane best fraction, lowest index wins ties; scalar finish if an energy wraps), pmaddwd pitch correlations (NEON too), AVX2 forward MDCT folding/rotations, pre-emphasis, pitch decimation and 5-tap FIR, compile-time PVQ U table, branchless icwrs, no per-frame heap allocation (encoder scratch, EcSave instead of Ec clones; tests/opus_no_alloc.rs checks the encoder). Left: exp_rotation's serial chains (~5%), the PVQ search's per-pulse lane reduction latency, compute_theta, transient analysis recursions.

## 2026-10-29

Criterion 2: tests/opus_no_alloc.rs encoding_allocates_nothing_per_frame encodes mono and stereo, 2.5-20 ms frames, VBR/CBR, complexity 5-10 with a counting allocator and asserts zero allocations after the first frame. Criterion 1 is true for AVX2 (simd_enc tests: odd sizes, tails, extreme values, wrap-around); the NEON kernels (xcorr, dot, shifted dot) only compile-check here and run in ARM CI.

## 2026-10-29

Criterion 1: the NEON kernels' differential tests passed in ARM CI on PR #38 (aarch64 NEON job). Merged as a99cad7: 1.42-1.59x libopus 1.4 float, 1.66x libopus 1.3.1 fixed, byte-identical output.
