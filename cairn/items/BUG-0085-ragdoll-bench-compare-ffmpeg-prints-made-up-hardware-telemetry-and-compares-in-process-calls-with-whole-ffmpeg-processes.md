---
id: 85
uid: 9f7e9912-5158-486d-8bdc-fe0028320ea2
title: ragdoll bench --compare-ffmpeg prints made-up hardware telemetry and compares in-process calls with whole FFmpeg processes
type: bug
status: backlog
milestone: m11-validation
created: 2026-10-28
updated: 2026-10-28
priority: p1
effort: m
area: bench
---

## What happens

`ragdoll bench --compare-ffmpeg` (`cmd_bench` in `ragdoll-cli`) presents numbers that do not hold up:

- The "HARDWARE PMU TELEMETRY & BATTERY EFFICIENCY" table is not measured on Linux: every instruction and cycle count is 0.00, energy and power are modelled, the FFmpeg row is labelled "FFmpeg 9.0.2" when 6.1.1 is installed, and the "P-Core"/"E-Core" split is reported on a 4-vCPU VM with no such cores.
- Each Ragdoll workload is a library call on data already in memory, while each FFmpeg workload is a separate `ffmpeg`/`ffprobe` process that starts up, probes, reads and writes files (and for probe, prints every packet as JSON). The speedups are mostly FFmpeg process overhead: "Container Probe 1178x" compares 0.05 ms with ffprobe's ~60 ms start-up.

On 2026-10-28 a like-for-like comparison (both sides in process, packets pre-demuxed, only codec calls timed, single-threaded) gave very different results from the suite's: AAC-LC decode 13x slower than FFmpeg (SIMD-0083), FLAC decode 2.5x slower (SIMD-0084), H.264 decode 6.3x slower at 1080p High (SIMD-0076), Opus within 0.8-1.1x of libopus, FLAC encode 1.5x faster than FFmpeg level 5 with 11 % larger files.

## What should happen

The benchmark reports only what it measures, and compares equal work:
- Hardware counters come from `perf_event_open` (Linux) or the macOS PMU, and are omitted, not zeroed or modelled, where unavailable. The FFmpeg version is read from `ffmpeg -version`. Core-type rows appear only on hybrid CPUs.
- The FFmpeg side runs in process through libavcodec/libavformat (as an optional feature), or the CLI timing subtracts measured start-up and both sides do the same I/O.
- Every row states threads, input and what is timed.

## Reproduction

1. `ragdoll bench --compare-ffmpeg` on Linux x86_64.
2. The PMU table shows 0.00 instructions and cycles, "FFmpeg 9.0.2", and P/E-core rows; "1. Container Probe" reports a speedup of over 1000x.

## Acceptance criteria

- [ ] No modelled, placeholder or hard-coded telemetry is printed; unavailable metrics are omitted with a note.
- [ ] Each FFmpeg comparison times the same work as the Ragdoll side (in process, or start-up subtracted and I/O matched), documented per workload.
- [ ] A regression test checks the JSON output for zeroed counters and a hard-coded FFmpeg version.
