---
id: 82
uid: 18c32a99-2a85-4ec6-b621-783738fdbbd3
title: AAC encoder ignores its target bitrate (no rate control; 35 kbit/s at 128k)
type: bug
status: backlog
milestone: m11-validation
created: 2026-10-28
updated: 2026-10-28
priority: p2
effort: m
area: codec-audio
---

## What happens

`AacEncoder` has no rate control. `bitrate_bps` only picks a fixed offset for `global_gain` (0, -6 or -10 steps for under 128k, 128k and 192k and up), every band shares that one scale factor, and nothing measures the bits spent. On 2026-10-28, `ragdoll transcode` of a 4 s, 48 kHz stereo WAV (tones and a chirp) with `--audio aac`, which asks for 128 kbit/s, wrote 17,690 bytes: 35 kbit/s. The output bitrate follows the signal, not the setting, and nothing in the encoder bounds a frame's size, so a loud, broadband signal can also overshoot.

The encoder is otherwise conformant: FFmpeg decodes its streams, and the Ragdoll and FFmpeg decoders agree on them within 1e-3 of the ISO/IEC 14496-26 bound (BUG-0079).

## What should happen

The average bitrate tracks `bitrate_bps` (as FFmpeg's and fdk-aac's encoders do), with a bit reservoir bounding each frame by the ADTS buffer fullness, and per-band scale factors that spend the bits where the signal needs them.

## Reproduction

1. `ffmpeg -f lavfi -i "aevalsrc=0.3*sin(2*PI*440*t)+0.2*sin(2*PI*(200+1500*t)*t)+0.05*sin(2*PI*9000*t)|0.3*sin(2*PI*660*t):s=48000:d=4" -c:a pcm_s16le in.wav`
2. `ragdoll transcode in.wav out.aac --audio aac`
3. `out.aac` is 17,690 bytes (35 kbit/s) for a 128 kbit/s request.

## Acceptance criteria

- [ ] On speech, music and synthetic signals the average bitrate is within 10 % of the target from 32 to 320 kbit/s.
- [ ] No frame exceeds the bit reservoir (6144 bits per channel).
- [ ] A regression test reproduces the failure and passes with the fix.
