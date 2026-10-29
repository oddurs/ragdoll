---
id: 99
uid: abc2ecef-b301-485a-a147-d4496c9f0914
title: 'Site and README: refresh codec benchmark numbers and limitations after the speed rounds'
type: docs
status: done
milestone: m11-validation
created: 2026-10-29
updated: 2026-10-29
closed_at: 2026-10-29
priority: p2
area: arch
---

## Reader & Question

Someone deciding whether Ragdoll's codecs are fast enough to use: how do they compare with FFmpeg, libopus and x264 today, and what is still missing?

## Content

The home page's benchmark table, status lists and footnote; the Benchmarks page's codec table and notes; the H.264 and AAC rows of Formats; the ragdoll-codec crate page; the From FFmpeg and Introduction limitations; the Roadmap's speed and audio rows and known limitations; the README's performance table and roadmap.

## Acceptance criteria

- [x] The documentation accurately reflects the current CLI and benchmark numbers.

## 2026-10-29

The site, docs and README still described the codecs as several times slower than FFmpeg (H.264 decode 6x, AAC 13x, FLAC 2.5x), the H.264 decoder as single-threaded and the AAC encoder as without rate control. Refreshed from one benchmark session (scratchpad perf/bench2.py: in-process, codec calls only; H.264 decoding as the median of 11 interleaved rounds, 1 and 4 threads) and PSNR measured with FFmpeg's psnr filter for the encoder comparison. Single-thread H.264 decoding is reported as a tie (0.94-1.07x across sessions).
