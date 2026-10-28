---
id: 81
uid: b594a11f-16de-42f7-9a76-0c8ae97e09d1
title: OggMuxer writes one page per packet, adding about 12 % overhead to Opus at 96 kbit/s
type: bug
status: backlog
milestone: m11-validation
created: 2026-10-28
updated: 2026-10-28
priority: p2
effort: s
area: format
---

## What happens

`OggMuxer` closes a page after every packet. Each page costs a 27-byte header plus its segment table, so 20 ms Opus packets at 96 kbit/s carry about 28 bytes of framing each: `ragdoll transcode` of a 4 s stereo WAV to `.ogg` wrote 54,415 bytes for 48,680 bytes of Opus payload (108 kbit/s on disk for a 96.9 kbit/s stream). libopus's tools and FFmpeg fill pages to about 4 KB or one second.

## What should happen

Pages hold several packets, flushed when the next packet would push the page past about 4 KB (or 255 segments), or when a page would span more than about one second. RFC 7845 section 3 also asks that the first audio data page begins a new page, and that the end-of-stream page carries the final granule position.

## Reproduction

1. `ragdoll transcode in.wav out.ogg --audio opus` on 4 s of stereo audio, then compare the file size with the sum of packet sizes (`ffprobe -show_packets`).

## Acceptance criteria

- [ ] Ogg Opus and Ogg FLAC pages carry multiple packets; overhead below 1 % at 96 kbit/s Opus.
- [ ] Granule positions stay correct (FFmpeg reads the same duration and samples), and seeking still works.
- [ ] A regression test reproduces the failure and passes with the fix.
