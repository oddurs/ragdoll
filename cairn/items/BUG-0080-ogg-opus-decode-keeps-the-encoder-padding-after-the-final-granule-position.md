---
id: 80
uid: d4c2e0cf-5635-4191-8d1d-127aa4e7c028
title: Ogg Opus decode keeps the encoder padding after the final granule position
type: bug
status: done
milestone: m11-validation
assignee: Claude
created: 2026-10-28
updated: 2026-10-28
closed_at: 2026-10-28
priority: p2
effort: s
area: format
---

## What happens

`ragdoll_pipeline::decode_audio_bytes` decodes every packet of an Ogg Opus stream and drops the `OpusHead` pre-skip, but does not trim the end. The last packet is decoded in full, so the encoder's padding stays in the output. `crates/ragdoll-pipeline/tests/data/opus/voip_mono_60ms.opus` (1.5 s from FFmpeg's libopus) has a final granule position of 72312, i.e. 72000 samples after the 312-sample pre-skip, but decodes to 74568 samples.

## What should happen

RFC 7845 section 4.4: the granule position of the last page gives the end of the stream, and the decoder drops the samples after it. The 1.5 s file decodes to exactly 72000 samples. The same applies to the Matroska path, whose end comes from the block durations or `DiscardPadding`.

## Reproduction

1. `cargo test -p ragdoll-pipeline --test opus_ogg` and look at the sample counts in `CASES` (74568 and 48168 for 1.5 s and 1.0 s files).

## Acceptance criteria

- [x] `OggDemuxer` exposes the final granule position, and `decode_audio_bytes` trims Ogg Opus output to `final_granule - pre_skip` samples.
- [x] The `opus_ogg` test expects 72000 and 48000 samples, still sample-exact with libopus over the kept range.
- [x] A regression test reproduces the failure and passes with the fix.

## 2026-10-28

OggDemuxer records the last page granule position (final_granule()); for Opus the track duration is final_granule - pre_skip. decode_audio_bytes drops the pre-skip and truncates at that end. OggMuxer writes the final page with granule min(sum of packet durations, pre_skip + track.duration_ticks) for Opus, and the pipeline's Opus track duration is now the input length instead of packets x 960, so Ragdoll's own files carry the trim. Evidence: the FFmpeg-libopus test files decode to 72000 and 48000 samples (1.5 s and 1.0 s), still sample-exact with libopus over the kept range; a 3.7 s Ragdoll transcode decodes in FFmpeg to exactly 177600 samples (the input), 34.5 dB SNR; ogg->ogg remux keeps granule 72312 and FFmpeg's decode identical.

## Result

Ogg Opus decode honours the final granule position: decode_audio_bytes output is exactly the encoded length after pre-skip, and OggMuxer writes the trimmed end granule from the track duration.
