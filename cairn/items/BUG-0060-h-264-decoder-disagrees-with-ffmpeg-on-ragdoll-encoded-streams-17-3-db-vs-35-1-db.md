---
id: 60
uid: 63253b72-1939-490e-ac7d-15a8fa6c93fc
title: H.264 decoder disagrees with FFmpeg on Ragdoll-encoded streams (17.3 dB vs 35.1 dB)
type: bug
status: planned
milestone: m11-validation
created: 2026-10-25
updated: 2026-10-25
priority: p0
effort: m
area: codec-video
---

## Problem

On `main` at `efb82d2`, `ragdoll synth s.mp4 --resolution 1280x720 --duration 2` produces a 205,329-byte Constrained Baseline stream. FFmpeg decodes it at 35.14 dB average PSNR against the source frames (`ragdoll synth src.y4m` with the same arguments). Ragdoll's own `H264Decoder` decodes the same file at 17.32 dB, and the two decoders' outputs differ (17.4 dB between them; MD5 `215c8187…` vs `f734030d…`). H.264 decoding is bit-exact by specification, so Ragdoll's decoder is wrong. Since FFmpeg agrees with the encoder's intent, the encoder is likely fine.

Impact: `transcode`, `extract` and `metrics` decode with `H264Decoder`, so they produce corrupted frames and misleading scores on Ragdoll-encoded H.264. The 71 workspace tests pass because no test cross-decodes Ragdoll output with an independent decoder.

Repro:

```sh
ragdoll synth s.mp4 --resolution 1280x720 --duration 2
ragdoll synth src.y4m --resolution 1280x720 --duration 2
ragdoll transcode s.mp4 rd.y4m
ffmpeg -i s.mp4 -i src.y4m -lavfi "[0][1]psnr" -f null -   # 35.14 dB
ffmpeg -i rd.y4m -i src.y4m -lavfi "[0][1]psnr" -f null -  # 17.32 dB
```

Suspects: decoder-side residual / intra-4x4 prediction, deblocking, or reconstruction order added in M7, where the encoder and decoder share no reference check.

## Acceptance criteria

- [ ] Ragdoll and FFmpeg decode Ragdoll-encoded H.264 to byte-identical YUV across QP 10, 22, 30 and 40 and multi-GOP motion.
- [ ] A regression test asserts the encoder's reconstruction equals `H264Decoder` output for every frame.
- [ ] The interop runner covers this case so it cannot regress silently.
