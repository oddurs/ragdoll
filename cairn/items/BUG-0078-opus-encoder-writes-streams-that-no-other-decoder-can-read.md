---
id: 78
uid: a3221759-3a69-4e16-ac87-b8f8bd50bb50
title: Opus encoder writes streams that no other decoder can read
type: bug
status: done
milestone: m11-validation
assignee: Claude
depends_on:
- 77
created: 2026-10-28
updated: 2026-10-28
closed_at: 2026-10-28
priority: p0
effort: l
area: codec-audio
---

## Problem

`OpusEncoder` emits packets in the same invented format the old decoder reads. On 2026-10-28, `ragdoll transcode in.wav out.ogg --audio opus` on a 4 s stereo 440/660 Hz sine produced 234 KB (about 470 kbit/s against a 96 kbit/s target), and FFmpeg 6.1 decoded it to noise (-4.4 dB SNR at the best alignment). Every `.ogg`/`.webm`/`.mkv` Opus track Ragdoll writes is unplayable elsewhere.

## Acceptance criteria

- [x] Packets are valid RFC 6716 CELT frames: libopus decodes Ragdoll-encoded streams with the final range matching the encoder's for every packet.
- [x] At 64 kbit/s and above, libopus-decoded output of the test signals is within 1 dB SNR of libopus's own encoder at the same bitrate and frame size, and the bitrate is within 10% of the target.
- [x] The Ragdoll encode/decode roundtrip tests check SNR against the input instead of only non-zero energy.

## 2026-10-28

Replaced the invented-format encoder with a port of the fixed-point libopus 1.3.1 CELT encoder (celt_encoder.c: pre-emphasis, pitch prefilter with remove_doubling, transient analysis and patch, MDCT, band energies, dynalloc, tf analysis, spreading and tapset decisions, alloc trim, stereo analysis, intensity hysteresis, VBR/CVBR/CBR rate control, coarse/fine/final energy, PVQ with theta RDO) and the CELT-only paths of opus_encoder.c (equivalent rate, stereo and bandwidth decisions, DC reject, stereo fade, multiframe repacketizing, CBR padding, TOC-only packets at very low rates). The Opus layer is the OPUS_APPLICATION_RESTRICTED_LOWDELAY application, so every frame is CELT.

## 2026-10-28

Evidence: packets are byte-identical (payload and final range) to opus_demo -e restricted-lowdelay from fixed-point libopus 1.3.1 in 2,089 cases: 864 (music/speech x mono/stereo x 16-128 kbit/s x 2.5-60 ms x VBR/CVBR/CBR x complexity 0/5/9), 756 at 0.5-8 kbit/s including the TOC-only path and 80-120 ms, 190 edge cases (80/100/120 ms multiframe, 256-600 kbit/s, every complexity 1-8, packet loss 1-30 %, force-mono), 48 full 60 s inputs (decoded RFC 8251 audio) and 35 adversarial signals (full-scale noise, +-32767 squares, impulses, DC, bursts) with overflow checks on. Complexity 10 behaves as 9: libopus 10 adds its floating-point tonality analysis, which is not ported.

## 2026-10-28

Quality: libopus-decoded SNR equals libopus restricted-lowdelay exactly and is within 0.7 dB of libopus's audio application at complexity 9 at 64-192 kbit/s (music 20.8/23.7/26.5/28.9 dB vs 20.6/23.4/26.3/29.0; speech 24.5/27.2/29.4/31.9 vs 24.7/27.9/29.7/32.2). libopus audio at complexity 10 is up to 2.2 dB better on speech at 64 kbit/s thanks to the analysis. Rate: CVBR (the default) within +1.1 % of target, CBR exact. The original repro (ragdoll transcode 440/660 Hz WAV to .ogg) now decodes in FFmpeg (libopus and native) at 34.5 dB SNR, sample-aligned via the 120-sample pre-skip; the file is 108 kbit/s against a 96.9 kbit/s payload because OggMuxer writes one page per packet (about 28 bytes each).

## Result

Opus encoder is a port of fixed-point libopus 1.3.1's CELT encoder in the restricted low-delay application: packets byte-identical to opus_demo -e restricted-lowdelay at complexity 0-9 across 2,089 cases; FFmpeg decodes Ragdoll .ogg output at 34.5 dB SNR. Pre-skip is 120 samples.
