---
id: 86
uid: 5d1b334c-3681-41a3-bc2b-ab0b09287d9f
title: Audio transcode decodes the whole track into one AudioFrame, so peak RSS grows with duration
type: bug
status: backlog
milestone: m11-validation
created: 2026-10-28
updated: 2026-10-28
priority: p2
effort: m
area: pipeline
---

## What happens

`decode_audio_bytes` and `decode_audio_track_packets` decode a whole audio track into one `AudioFrame`, and `apply_audio_dsp` and the encoders take that frame whole. This affects the audio-only `transcode` path, and the video path when it re-encodes audio (`--audio aac|opus|flac`, `--sample-rate`, `--channels`, `--loudness`). Peak RSS therefore grows with duration: 48 kHz stereo `f32` is 384 KB per second, or 1.4 GB per hour, before resampling and encoder copies. Video frames stream in bounded batches since BUG-0072.

## What should happen

Audio is decoded, resampled, measured and encoded in bounded chunks. EBU R128 normalisation needs the integrated loudness first, so it takes a measuring pass and then a streaming gain pass.

## Reproduction

1. `ragdoll synth a.wav --duration 150` and `--duration 600` (48 kHz stereo, 29 and 115 MB).
2. `ragdoll transcode a.wav a.ogg --audio opus`: peak RSS 89 MB for 150 s and 342 MB for 600 s (2026-10-28, x86_64).

## Acceptance criteria

- [ ] A test transcodes two audio durations at least 4x apart and asserts that peak RSS differs by less than a fixed overhead.
- [ ] Output stays byte-identical to the whole-track path for FLAC, Opus and AAC, with and without resampling and loudness normalisation.
