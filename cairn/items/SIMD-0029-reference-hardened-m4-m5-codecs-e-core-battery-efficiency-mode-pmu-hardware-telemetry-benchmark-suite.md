---
id: 29
uid: d0170672-31b3-4f53-a315-cc049805ead5
title: Reference-Hardened M4/M5 Codecs, E-Core Battery-Efficiency Mode & PMU Hardware Telemetry Benchmark Suite
type: kernel
status: done
milestone: m5-cli-bench
assignee: Oddur Sigurdsson
created: 2026-10-25
updated: 2026-10-25
closed_at: 2026-10-25
priority: p0
effort: l
area: arch
part_of:
- 5
---

## Mathematical Specification

Reference-hardening pass across `ragdoll-codec`, `ragdoll-simd`, `ragdoll-pipeline`, and `ragdoll-cli` benchmarking against `FFmpeg 9.0.2` (`libavcodec`), `xiph/flac` (`libFLAC`), and `videolan/dav1d` (`src/arm/64`), optimizing both raw throughput and battery/energy efficiency (instructions retired per sample/pixel, L1/L2 cache working set locality, and Apple Silicon E-core QoS scheduling):

1. **FLAC Single-Pass Cascaded Fixed Predictor & Multi-Partition Rice Coding (`libFLAC/fixed.c:244-250`, `libavcodec/flacenc.c`)**:
   - Compute all 5 Fixed predictor orders $k \in \{0, 1, 2, 3, 4\}$ via cascaded recurrence $\Delta^{(k)} x[n] = \Delta^{(k-1)} x[n] - \Delta^{(k-1)} x[n-1]$ and accumulate their $L_1$ error sums in a **single fused SIMD NEON / AVX2 memory pass**, reducing sample buffer memory traffic by $5\times$.
   - Implement adaptive multi-partition Rice entropy coding (`partition_order = 0..=4`, up to 16 partitions per block) using pre-accumulated partition $L_1$ sums (`find_optimal_param` from `libavcodec/flacenc.c`), plus RFC 9639 MD5 audio payload signature computation (`FlacStreamInfo::md5_signature`).
2. **H.264 NEON Spatial Intra Prediction & Motion-Compensated `P_L0_16x16` Inter Coding (`libavcodec/aarch64/h264pred_neon.S`, `libavcodec/h264_mb.c`)**:
   - Add NEON-accelerated $16\times16$ and $8\times8$ Intra predictors (DC, Vertical, Horizontal, Plane) and $16\times16$ SAD/SSD macroblock distortion kernels.
   - Add `P_L0_16x16` inter-frame macroblock encoding and decoding (`mb_type = 0` in P-slices with integer-pel diamond motion estimation and Exp-Golomb `mvd_l0` motion vector differences + $16\times16$ AC/DC CAVLC residual coding) for high-efficiency inter-frame compression on moving video.
3. **Battery-Saving Efficiency Mode & Hardware PMU Energy/Instruction Telemetry (`pthread_set_qos_class_self_np`, `proc_pid_rusage` `RUSAGE_INFO_V4`)**:
   - Expose `PowerMode` (`Performance`, `Balanced`, `Efficiency`) in `ragdoll-pipeline` and `--power-mode` in `ragdoll-cli`, pinning worker threads to Apple Silicon Efficiency cores (`QOS_CLASS_BACKGROUND` / `QOS_CLASS_UTILITY`, 64 KiB L1D / 4 MiB L2) with E-core-sized concurrency in `Efficiency` mode to slash power draw on battery-constrained devices.
   - Collect hardware PMU counters (`ri_instructions`, `ri_cycles`, `ri_phys_footprint`, user/sys CPU nanoseconds, IPC, instructions/pixel, instructions/sample, and estimated energy efficiency ratio) via `proc_pid_rusage(RUSAGE_INFO_V4)` on macOS and `getrusage` on Linux.

## Register & Cache Layout (NEON / AVX2)

- **FLAC Fused Fixed Predictor**: Keeps 4-lane running history registers (`vextq_s32`) in registers across loop iterations, writing residuals once and accumulating `vabsq_s32` into 64-bit accumulators (`vpaddlq_u32`).
- **H.264 Diamond Motion Search**: Uses 16-byte aligned NEON `vabdq_u8` + `vaddvq_u8` $16\times16$ SAD within a $\pm 16$ search window that stays resident inside the 64 KiB E-core L1D cache.
- **E-Core Battery Mode**: Limits active worker threads to Efficiency core count (`hw.perflevel1.logicalcpu` = 6 on M4) with `QOS_CLASS_BACKGROUND` so P-cores remain power-gated.

## Acceptance criteria

- [x] Scalar reference and NEON/AVX2 SIMD implementations pass differential bit-exactness tests across odd dimensions.
- [x] Zero heap allocations in the hot loop.
- [x] Single-pass cascaded FLAC Fixed predictor and adaptive multi-partition Rice coding (`partition_order = 0..=4`) + RFC 9639 MD5 verification pass `ffmpeg` and internal bit-exact roundtrip tests.
- [x] H.264 `P_L0_16x16` motion-compensated inter-frame encoder and decoder pass roundtrip and `ffmpeg` decode interoperability tests.
- [x] `PowerMode` (`Performance`, `Balanced`, `Efficiency`) and hardware PMU telemetry (`instructions`, `cycles`, `IPC`, `CPU-ms`, `RSS`) are integrated into `ragdoll-pipeline` and `ragdoll bench --compare-ffmpeg`.

## 2026-10-25

Completed reference-based hardening across ragdoll-simd, ragdoll-codec (FLAC single-pass NEON 5-order Fixed L1 evaluator, 4-mode stereo competition, adaptive porder 0..4 Rice partitions, RFC 1321 MD5; H.264 NEON Intra 16x16/8x8 predictors, ITU-T H.264 8.4.1.3 median MV predictor, P_Skip + P_L0_16x16 diamond motion estimation), ragdoll-format (is_valid_avcc_packet disambiguating 256..511B AVCC NALs from 3-byte Annex-B start codes), ragdoll-core/ragdoll-pipeline (CpuTopology, PowerMode::Efficiency QOS_CLASS_BACKGROUND E-core battery-saver scheduling, proc_pid_rusage RUSAGE_INFO_V4 hardware PMU telemetry + /usr/bin/time -l parser), and ragdoll-cli (7-workload speed + PMU energy/battery benchmark suite).
