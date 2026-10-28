---
id: 79
uid: 1953c6d3-24c2-482e-8a60-5bda8a27141e
title: AAC-LC decoder diverges from FFmpeg (26 dB SNR) and loses sync at 256 kbit/s
type: bug
status: backlog
milestone: m11-validation
created: 2026-10-28
updated: 2026-10-28
priority: p1
effort: m
area: codec-audio
---

## Problem

On 2026-10-28, ADTS streams encoded by FFmpeg's native AAC encoder (8 s, a chirp plus tones and noise) were decoded with `AacDecoder::decode_packet` and compared with FFmpeg 6.1's decoder:

| Stream | Ragdoll | SNR vs FFmpeg |
|---|---|---|
| 128 kbit/s stereo 44.1 kHz | 346/346 frames | 25.7 dB |
| 48 kbit/s mono 44.1 kHz | 346/346 frames | 26.7 dB |
| 256 kbit/s stereo 48 kHz | 9 frame errors ("unexpected end of bitstream", "SSR gain_control_data not supported") | -2.6 dB |

Two decoders of the same AAC-LC stream should agree to float rounding (well above 90 dB), so a coding tool is decoded wrongly. The 256 kbit/s errors point to a mis-parsed element: an SCE/CPE field desync leading to `gain_control_data` being read.

## Acceptance criteria

- [ ] FFmpeg-, fdk-aac- and Ragdoll-encoded AAC-LC (mono, stereo, 5.1; 32-320 kbit/s; 8-96 kHz) decode within the ISO/IEC 14496-26 conformance tolerance of FFmpeg's decoder.
- [ ] No frame errors on valid streams.
