---
id: 42
uid: f405719e-3713-445c-b234-2117e2f3fdee
title: SIMD Windowed-Sinc Kaiser (bessel_i0) Polyphase Audio Resampler & Channel Matrix Mixer
type: kernel
status: planned
milestone: m8-audio-pro
depends_on:
- 37
created: 2026-10-25
updated: 2026-10-25
priority: p0
area: codec-audio
effort: l
---

## Mathematical Specification

1. **Windowed-Sinc Kaiser Polyphase Audio Resampler (`libswresample/resample.c` reference)**:
   - Construct a $P$-phase, $T$-tap ($T \in \{16, 32, 64\}$) polyphase filter bank $h[p, k] = \text{sinc}\!\left(f_c \cdot (k - T/2 + p/P)\right) \cdot \frac{I_0(\beta \sqrt{1 - u^2})}{I_0(\beta)}$ using `ragdoll_core::bessel_i0` ($\beta = 9.0$, $>90\text{ dB}$ stopband attenuation) with DC-unity normalization per phase.
   - Advance exact rational phase $(n \cdot F_{\text{in}}) \pmod{F_{\text{out}}}$ without float drift across arbitrary sample rates (`44.1 kHz <-> 48 kHz <-> 88.2 kHz <-> 96 kHz`).
2. **Channel Matrix Mixer (`ITU-R BS.775` / `libswresample/rematrix.c`)**:
   - Stereo <-> Mono (`0.7071` equal-power or `0.5` linear), `5.1` (`L, R, C, LFE, Ls, Rs`) -> Stereo downmix ($L' = L + 0.7071 C + 0.7071 L_s$, $R' = R + 0.7071 C + 0.7071 R_s$ with unity headroom normalization), and `7.1` -> `5.1` / Stereo.

## Register & Cache Layout (NEON / AVX2)

- Vectorize 16-tap and 32-tap polyphase convolution in `f32x4` (`vfmaq_f32` + `vaddvq_f32`) and Q15 `i16` (`vmull_s16` + `vpadalq_s32`) across planar and interleaved audio channels.
- Vectorize $N \times M$ channel rematrixing over 16-sample blocks in L1 cache.

## Acceptance criteria

- [ ] Scalar reference and NEON/AVX2 SIMD implementations pass differential tests within `1e-6` (`f32`) / `0 LSB` (`Q15`).
- [ ] Zero heap allocations in the per-frame resampling hot loop, sustaining `>1,500x` real-time on `44.1 kHz -> 48 kHz` stereo audio.
