---
id: 79
uid: 1953c6d3-24c2-482e-8a60-5bda8a27141e
title: AAC-LC decoder diverges from FFmpeg (26 dB SNR) and loses sync at 256 kbit/s
type: bug
status: done
milestone: m11-validation
assignee: Claude
created: 2026-10-28
updated: 2026-10-28
closed_at: 2026-10-28
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

- [x] FFmpeg-, fdk-aac- and Ragdoll-encoded AAC-LC (mono, stereo, 5.1; 32-320 kbit/s; 8-96 kHz) decode within the ISO/IEC 14496-26 conformance tolerance of FFmpeg's decoder.
- [x] No frame errors on valid streams.

## 2026-10-28

Replaced the decoder rather than patching it: aac/decoder.rs is a fresh ISO/IEC 14496-3 AAC-LC decoder in f64 (SCE/CPE/LFE/DSE/PCE/FIL; all window sequences with sine/KBD shapes; scale factor bands for all 13 rates; pulse data on the quantised integers; M/S; intensity; PNS; TNS with the tns_max_bands tables and order 12/7; IMDCT as DCT-IV over an N/4 complex FFT), and 5.1 and 7.1 output reordered from bitstream order to FL FR FC LFE BL BR (SL SR). The encoder stays in aac/mod.rs. Intensity stereo follows FFmpeg, not the letter of the spec: FFmpeg inverts an intensity band wherever M/S is signalled, ms_mask_present 2 included, and its own encoder writes that mode over intensity bands (-aac_ms 1); 14496-3 invert_intensity() reads ms_used only for ms_mask_present 1, which put the right channel 1000x past the tolerance on such streams. Evidence, each against FFmpeg 6.1's float decoder with the 14496-26 bounds (RMS 2^-15/sqrt(12), max 2^-14), PNS off where the encoder allows (its noise is implementation-defined): FFmpeg-encoded mono/stereo/5.1 at 8-96 kHz and 16-320 kbit/s, forced M/S included, at most 4.3e-3 x RMS and 3.9e-3 x max; fdk-aac-encoded (built locally from mstorsjo/fdk-aac as a tool only) mono/stereo/5.1 at 8-96 kHz and 32-320 kbit/s at most 1.5e-2 x RMS and 0.11 x max; Ragdoll-encoded 1.0e-3 x. No frame errors on any stream, the old 256 kbit/s desync included. The larger fdk-aac residual is FFmpeg's single-precision TNS: fdk uses TNS on most frames, the excess sits in TNS frames at the transient's frequency, and deriving our TNS coefficients and filter in f32 as FFmpeg does cuts the max from 0.11 to 0.02 x, so the f64 path is kept. Regression test tests/aac_reference.rs: 7 FFmpeg-encoded ADTS fixtures (tools/aac_fixtures/gen.sh, 264 KB) covering every window sequence, codebooks 1-11, ms_mask_present 0/1/2, intensity in both phases, TNS, 8-96 kHz and 5.1, compared with FFmpeg's s16 output: every sample within one LSB, about 0.06 % differ (bound 1 %). FFmpeg's encoder never writes pulse data, so pulse decoding is covered only by the fdk-aac sweep. Filed BUG-0082: the Ragdoll AAC encoder has no rate control (35 kbit/s for a 128k request).

## Result

AAC-LC decodes within 0.11x of the ISO/IEC 14496-26 tolerance of FFmpeg's float decoder (typically 3e-3x) for FFmpeg-, fdk-aac- and Ragdoll-encoded mono, stereo and 5.1 at 8-96 kHz and 16-320 kbit/s, with no frame errors; intensity stereo follows FFmpeg's ms_mask_present 2 inversion. tests/aac_reference.rs pins it against FFmpeg's output on 7 fixtures.
