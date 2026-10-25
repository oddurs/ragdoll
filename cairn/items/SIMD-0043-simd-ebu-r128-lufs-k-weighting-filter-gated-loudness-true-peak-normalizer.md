---
id: 43
uid: c0d1dcce-d7fc-407c-9119-aa1f59dba01c
title: SIMD EBU R128 (LUFS) K-Weighting Filter, Gated Loudness & True-Peak Normalizer
type: kernel
status: done
milestone: m8-audio-pro
assignee: Oddur Sigurdsson
depends_on:
- 42
created: 2026-10-25
updated: 2026-10-25
closed_at: 2026-10-25
priority: p1
area: codec-audio
effort: m
---

## Mathematical Specification

1. **ITU-R BS.1770-4 / EBU R128 K-Weighting Biquad Cascade**:
   - Stage 1 ( Acoustic high-shelf head filter) + Stage 2 (RLB high-pass filter) biquad IIR transfer functions recomputed analytically for any sample rate $f_s$.
2. **Momentary ($400\text{ ms}$), Short-Term ($3\text{ s}$) & Integrated Gated Loudness (`LUFS`)**:
   - $75\%$ overlapping $400\text{ ms}$ gating blocks with absolute threshold $\Gamma_a = -70\text{ LUFS}$ and relative threshold $\Gamma_r = L_{\text{abs}} - 10\text{ LU}$, plus Loudness Range (`LRA`).
3. **4x Oversampled Polyphase True-Peak (`dBTP`) Detector & Gain Normalizer**:
   - 4-phase 12-tap polyphase FIR interpolator detecting inter-sample peaks (`dBTP`) and applying click-free true-peak-limited gain (`--loudness -14lufs --true-peak -1.0`).

## Register & Cache Layout (NEON / AVX2)

- Vectorize multichannel biquad state updates across channels (`f32x4`) and 4x oversampled polyphase FIR true-peak evaluation in NEON `vmaxvq_f32(vabsq_f32(...))`.

## Acceptance criteria

- [x] Integrated LUFS, LRA, and True-Peak (`dBTP`) match EBU Tech 3341 / ITU-R BS.2217 compliance test vectors within `+-0.1 LU`.
- [x] Scalar reference and NEON/AVX2 SIMD implementations pass differential bit-exactness tests.
- [x] Zero heap allocations in the hot loop.

## 2026-10-25

Implemented analytical ITU-R BS.1770-4 K-weighting biquad cascade (Stage 1 high-shelf + Stage 2 RLB high-pass with unity numerator), 75% overlapping 400ms momentary & 3s short-term gated loudness (-70 LUFS abs / -10 LU rel), EBU Tech 3342 LRA, and 4x oversampled 12-tap polyphase True-Peak (dBTP) detector and limiter. Verified -23.000 LUFS on EBU Tech 3341 stereo reference tone and 100% bit-exact NEON vs scalar equivalence.

## Result

Delivered EbuR128Meter, KWeightingFilterBank, and 4x oversampled 12-tap True-Peak normalizer achieving -23.000 LUFS accuracy on EBU Tech 3341 vectors and 100% bit-exact NEON vs scalar equivalence.
