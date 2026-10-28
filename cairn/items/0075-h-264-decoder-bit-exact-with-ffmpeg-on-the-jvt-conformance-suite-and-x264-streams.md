---
id: 75
uid: 35f37170-c325-4fc2-a3b2-eed905c2ffe2
title: 'H.264 decoder: bit-exact with FFmpeg on the JVT conformance suite and x264 streams'
type: feature
status: doing
milestone: m11-validation
assignee: Claude
claimed: 2026-10-28
created: 2026-10-28
updated: 2026-10-28
priority: p0
effort: xl
area: codec-video
---

## Problem

`H264Decoder` decodes Ragdoll's own encoder output correctly (BUG-0060) but not third-party H.264. On 2026-10-28, decoding with a harness that writes `decode_stream_pts_ordered` output and comparing byte-for-byte with FFmpeg 6.1:

| Corpus | Pass | Fail | Skipped |
|---|---|---|---|
| libx264 feature matrix (25 streams, 352x288) | 5 | 20 | 0 |
| JVT conformance, ITU-T H.264.1 via FFmpeg FATE (195 streams) | 15 | 160 | 20 |

The decoder was written against the narrow subset Ragdoll's encoder emits, so the failures are structural rather than isolated:

- every slice NAL becomes a separate frame, so multi-slice pictures cannot decode;
- reference list modification, prediction weight tables and memory management control operations are parsed and discarded;
- there is no decoded picture buffer: no POC derivation (types 0, 1, 2), no frame_num- or POC-ordered reference list initialisation, no long-term references, no output bumping;
- the SPS and PPS parsers discard scaling matrices, POC type 1 parameters and `direct_8x8_inference_flag`, and only one SPS and PPS are kept;
- deblocking uses one QP offset for both alpha and beta, and one chroma QP offset for both Cb and Cr.

## Architecture & SIMD Design

A new decoder core under `ragdoll-codec/src/h264/`, structured after the spec (and FFmpeg's `h264dec`): parameter-set tables by id; complete slice header; picture assembly across slices with per-slice neighbour availability; a DPB with POC types 0-2, sliding window and MMCO, long-term references, list initialisation and modification, frame_num gaps, and output bumping; CAVLC and CABAC macroblock layers for every I, P and B type including direct spatial and temporal prediction; explicit and implicit weighted prediction; scaling matrices; and spec-exact deblocking. Verified kernels are reused: quarter-pel and chroma MC, 4x4 and 8x8 IDCT, and the CABAC engine. `H264Decoder`'s public API is unchanged.

Scope, as tiers of the FFmpeg FATE conformance list (FFmpeg is the oracle only where FATE verifies it):

1. 95 progressive 8-bit 4:2:0 streams (Baseline, Main, High, including multiple slices and arbitrary slice order).
2. 69 interlaced 8-bit 4:2:0 streams (field pictures and MBAFF).
3. 2 monochrome streams.

10-bit and 4:2:2 streams (18) are out of scope until the frame pipeline carries them.

## Acceptance criteria

- [x] All 25 libx264 feature-matrix streams decode byte-identically to FFmpeg, in output order.
- [x] All 95 tier-1 FATE conformance streams decode byte-identically to FFmpeg.
- [ ] All 69 tier-2 (interlaced) FATE conformance streams decode byte-identically to FFmpeg.
- [x] Both tier-3 (monochrome) streams decode byte-identically to the JM reference decoder (FFmpeg 6.1 disagrees with JM on these).
- [x] No input in either corpus panics; malformed slices return errors.
- [x] Ragdoll-encoded streams still round-trip (BUG-0060 tests pass).
- [x] A test in the workspace decodes a committed subset of the corpus and checks output hashes, so conformance cannot silently regress before the `qa` runner (`0063`) exists.

## 2026-10-28

2026-10-28: new decoder core in ragdoll-codec/src/h264/dec (parameter sets, slice header, DPB, CAVLC/CABAC macroblock layer, MV prediction incl. spatial/temporal direct, weighted prediction, scaling matrices, deblocking). Verified with two oracles: all 25 x264 streams and all 95 tier-1 FATE streams are byte-identical to both FFmpeg 6.1 and JM 19.1 (built from vcgit.hhi.fraunhofer.de/jvet/JM). Both monochrome streams match JM on every frame; FFmpeg 6.1 differs from JM on all 100 frames of each, so JM is the oracle there. Bugs found on the way that also affected the old decoder or shared code: BitReader::more_rbsp_data reported more data when the unbuffered tail held only the stop bit (CAVLC slices ran past their end); CabacDecoder::align_for_pcm started PCM samples one byte late whenever the arithmetic decoder sat on a byte boundary; decode_cavlc_block indexed past its tables when coeff_token signalled more coefficients than the block holds (found by fuzzing). FFmpeg crops unaligned left offsets only with -flags unaligned, so comparisons use it. Robustness: all 196 conformance files decode or return an Unsupported/InvalidSyntax error; 4,200 mutated streams produced no panics or hangs. Pipeline GOP splitting now requires an IDR slice, not just a container sync flag, because a fresh decoder cannot start on a P picture (the multitrack test fixture flags frame 15 as a keyframe although the encoder made it a P frame). Speed: 720p is about 90 fps on Ragdoll-encoded Baseline (old decoder 115 fps, but wrong on most third-party streams) and 53 fps on x264 medium; follow-up filed. Remaining: tier 2 (interlaced, 69 streams).
