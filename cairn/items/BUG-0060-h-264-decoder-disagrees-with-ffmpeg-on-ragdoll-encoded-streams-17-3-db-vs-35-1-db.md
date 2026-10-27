---
id: 60
uid: 63253b72-1939-490e-ac7d-15a8fa6c93fc
title: H.264 decoder disagrees with FFmpeg on Ragdoll-encoded streams (17.3 dB vs 35.1 dB)
type: bug
status: planned
milestone: m11-validation
depends_on:
- 64
created: 2026-10-25
updated: 2026-10-27
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

- [x] Ragdoll and FFmpeg decode Ragdoll-encoded H.264 to byte-identical YUV across QP 10, 22, 30 and 40 and multi-GOP motion.
- [x] A regression test asserts the encoder's reconstruction equals `H264Decoder` output for every frame.
- [ ] The interop runner covers this case so it cannot regress silently.

## Findings

The decoder's bitstream parsing was correct; two other defects produced the mismatch.

1. **Frame order, not pixels (the 17.3 dB).** Every frame `ragdoll transcode` wrote was byte-identical to one of FFmpeg's, but in the order 0, 15, 30, 45, 1, 16, … `H264Decoder` stamps PTS from `pic_order_cnt`, which restarts at each IDR, and `decode_video_bytes` sorted the whole stream by that PTS, interleaving GOPs. This hit every multi-GOP H.264 input in MP4 (parallel and serial paths), Matroska and MPEG-TS, so `transcode`, `extract`, `metrics`, `concat` and `package` all saw it. Fixed by `order_h264_frames` in `ragdoll-pipeline`: sort by POC within each GOP, then renumber PTS across the stream.
2. **8x8 inverse transform sign (High profile, ±1–2 levels).** `inverse_idct8x8` computed `b5 = a5 - (a3 >> 2)`; ITU-T H.264 eq. 8-345 is `f5 = (e3 >> 2) - e5`. Outputs 1 and 6 of each pass were sign-flipped. The encoder reconstructs with the same function, so Ragdoll agreed with itself and drifted from every conformant decoder. The old test only checked the output length; it now checks a block against FFmpeg's `h264_idct8_add` arithmetic.

Verified on 2026-10-27: Baseline, Main and High at QP 10, 22, 30 and 40, in MP4, MKV and TS (60 frames, four GOPs, 1280x720), decode byte-identically in Ragdoll and FFmpeg. Regression tests: `h264::tests::encoder_reconstruction_matches_decoder_every_frame` and `tests::multi_gop_h264_decodes_in_presentation_order`.

The last criterion waits on the interop runner (`0064`), which already specifies the reconstruction-equals-reference-decoder invariant.

FFmpeg's `psnr` filter reports about 34 dB for Ragdoll MKV output even though the decoded frames are identical to the MP4's. It pairs frames by timestamp, and MKV's millisecond timestamps do not line up with the y4m's 1/30 s. Compare MKV output as raw YUV, not with `psnr` against a y4m.
