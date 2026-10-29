---
id: 91
uid: 0b3d405c-48b7-4f73-a354-491304c8de38
title: 'FLAC encoder: match FFmpeg level 5 compression while staying faster'
type: kernel
status: done
milestone: m11-validation
created: 2026-10-28
updated: 2026-10-29
closed_at: 2026-10-29
priority: p1
area: codec-audio
effort: l
---

## Mathematical Specification

Per 4096-sample block and channel (left, right, side, mid for stereo):

- **Constant / wasted bits / Verbatim** detection in one pass (OR, XOR with the first sample, magnitude bound).
- **Fixed predictors 0-4**: L1 of every order in one pass (`flac_eval_fixed_orders_l1`, AVX2 added).
- **Stereo**: each channel's estimate is the cheapest fixed order's Rice estimate `rice(2 · L1)`, capped at its Verbatim size; only the best pair of `L+R`, `L+S`, `S+R`, `M+S` is planned exactly (`STEREO_TRY_MARGIN`).
- **LPC**: Tukey(0.5) window, autocorrelation `r[0..=12]` in f64 with fused multiply-adds in a fixed lane order (`flac_autocorr_f64`), Levinson-Durbin to order 12. The order is the one of least expected size `(n - m) · rice_bits(sqrt(2 · err_m / (n · mean(w²)))) + m · (15 + bps)`, with `rice_bits(μ) = min_k k + 1 + μ / 2^k` (`libFLAC`'s log formula clamps at 0 bits/sample and picked order 1-3 on quiet tones: 19 % larger). Coefficients: 15-bit precision, largest shift `<= 15` that fits, error feedback between taps (`libavcodec/lpc.c`).
- **Residual**: exact `i32` when `(Σ|c| + 1) · max|x| < 2^31` (`flac_lpc_residual_i32`), else 64-bit with an `i32` range check.
- **Rice**: partition orders 0-8; folded sums and ORs of the leaves (`flac_rice_partition_sums`), the tree up to order 0, and every partition costed once (`flac_rice_partition_costs`): `k = floor(log2((2 Σu + cnt) / (2 cnt)))`, which minimises `cnt (k + 1) + (Σu - cnt (2^k - 1) / 2) >> k` exactly, or an escape to `bitlen(OR)` raw bits when cheaper (all-zero partitions cost 9 bits).
- **Fixed vs LPC**: the fixed predictor is coded exactly when its single-partition estimate is within 1.3x of LPC's single-partition size (fixed residuals gain more from partitioning on speech).
- Exact Rice costs (`Σ (u >> k)` for `k ± 1` on the chosen order) bought 0.02 %, so the estimate stays.

## Register & Cache Layout (NEON / AVX2)

- `flac_autocorr_f64`: lags in groups of 8, one `x[i..i+4]` load feeding 8 `vfmadd231pd` accumulators (2 `fmla` pairs on NEON); scalar reference uses `f64::mul_add` in the same lane order, so all three are bit-identical.
- `flac_lpc_residual_i32`: 16 samples per step (8 on NEON), `vpmulld` per tap from unaligned history loads, `vpsrad` by the shift.
- `flac_rice_partition_sums`: eight partitions at once, 32-bit lane sums (guarded by `bitlen(OR) + log2(len) <= 32`, else 64-bit), reduced with `vshufps`-pair trees.
- `flac_rice_partition_costs`: four partitions per step in 64-bit lanes; `floor(log2)` from the exponent of the exact `f64` (`v | 2^52` trick), `vpsllvq`/`vpsrlvq`, `vpmuludq`.
- `BitWriter::write_rice_signed_i32`: four codes joined into one field of up to 64 bits per buffer-full test (BMI2 build on AVX2 CPUs).
- The whole frame search runs in an `avx2,bmi1,bmi2,lzcnt` build on x86_64 (plain loops vectorize 8 wide); all buffers live in a per-thread `EncoderWorkspace`, and 16-bit input is deinterleaved block by block inside the workers.

## Acceptance criteria

- [x] Scalar reference and NEON/AVX2 SIMD implementations pass differential bit-exactness tests across odd dimensions.
- [x] Zero heap allocations in the hot loop.

## 2026-10-28

Results (x86_64 AVX2 VM, FFmpeg 6.1.1 flac -compression_level 5 via avbench vs rbench flac-enc with MD5 on; frame bytes, ratio = Ragdoll / FFmpeg): music48 (60 s, 48 kHz) 4,930,486 vs 6,248,327 = 0.789; music44 0.794; speech-like lavfi 44.1 kHz 0.984; tonal lavfi 44.1 kHz 0.812 (before: 1.108 / 1.103 / 1.041 / 1.181). Interleaved single-core wall clock, min of 21 (both include MD5; Ragdoll also its harness copy): music48 65.7 vs 76.1 ms (1.16x), music44 1.15x, speech 1.26x, tonal 1.26x; multithreaded encode_stream 2-4x faster than FFmpeg. Callgrind on 10 s of music48: 185 M -> 93 M instructions (MD5 is 27 M of that, at the hardware limit: OpenSSL's MD5 takes 18 ms on the 11.5 MB input). Every stream decodes bit-exactly in FFmpeg and Ragdoll with a valid STREAMINFO MD5 (test encoder_round_trips_and_beats_ffmpeg_level_5).

## 2026-10-28

Levers measured (music48 / speech / tonal ratios): exhaustive order search 0.788/0.977/0.810 at 2.7x the time, estimate-based single order now within 0.2 % of it; 12-bit precision 0.843 on music48 (15-bit kept); Welch window slightly worse than Tukey(0.5); 4608-sample blocks better on music, worse on speech (4096 kept); exact Rice costs 0.02 %; escape codes 0.6 % on speech; full stereo search another 1 % on music for 40 % more time, a 1.01 stereo margin 0.5 % for 10 % (left at 1.0, STEREO_TRY_MARGIN). Left on the table: madd_epi16 LPC residuals for 16-bit channels, fusing the four stereo L1 passes, NEON versions of the costs kernel (NEON uses the scalar reference there).
