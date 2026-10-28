---
id: 78
uid: a3221759-3a69-4e16-ac87-b8f8bd50bb50
title: Opus encoder writes streams that no other decoder can read
type: bug
status: planned
milestone: m11-validation
depends_on:
- 77
created: 2026-10-28
updated: 2026-10-28
priority: p0
effort: l
area: codec-audio
---

## Problem

`OpusEncoder` emits packets in the same invented format the old decoder reads. On 2026-10-28, `ragdoll transcode in.wav out.ogg --audio opus` on a 4 s stereo 440/660 Hz sine produced 234 KB (about 470 kbit/s against a 96 kbit/s target), and FFmpeg 6.1 decoded it to noise (-4.4 dB SNR at the best alignment). Every `.ogg`/`.webm`/`.mkv` Opus track Ragdoll writes is unplayable elsewhere.

## Acceptance criteria

- [ ] Packets are valid RFC 6716 CELT frames: libopus decodes Ragdoll-encoded streams with the final range matching the encoder's for every packet.
- [ ] At 64 kbit/s and above, libopus-decoded output of the test signals is within 1 dB SNR of libopus's own encoder at the same bitrate and frame size, and the bitrate is within 10% of the target.
- [ ] The Ragdoll encode/decode roundtrip tests check SNR against the input instead of only non-zero energy.
