---
id: 77
uid: f3a4e1c4-60c4-4842-94b8-686b70787041
title: 'Opus decoder: RFC 6716 decoding, sample-exact with fixed-point libopus on the RFC 8251 test vectors'
type: feature
status: done
milestone: m11-validation
assignee: Claude
created: 2026-10-28
updated: 2026-10-28
closed_at: 2026-10-28
priority: p0
effort: xl
area: codec-audio
---

## Problem

`OpusDecoder` cannot decode Opus from any other encoder. On 2026-10-28, decoding libopus-encoded Ogg Opus files (CELT 96 kbit/s stereo, SILK 12 kbit/s mono, hybrid 32 kbit/s) and comparing with FFmpeg 6.1 gave -12 dB SNR on all three: the output is noise. Mono SILK files also come out as stereo with twice the samples.

`opus.rs` reads a bitstream of its own design rather than RFC 6716. After the TOC byte it decodes per-band "active" flags, a uniform 11-bit energy and per-coefficient pulses with the range coder, which matches nothing in the specification. There is no SILK decoder (so no SILK-only or hybrid packets), no coarse/fine energy prediction, no bit allocation, no PVQ codebook decoding, no band folding, transient/TF handling, anti-collapse, post-filter or de-emphasis, and packet framing codes 1-3 (multiple frames per packet) are not parsed.

## Architecture & SIMD Design

A new decoder under `ragdoll-codec/src/opus/` that mirrors libopus's fixed-point decoder, so its output can be compared sample by sample:

- packet framing (RFC 6716 3.1-3.2: TOC, frame-count codes 0-3, padding) and the range decoder (4.1), with `final_range` exposed so every packet can be checked against the encoder's final range;
- SILK (4.2): LP layer indices, pulses, NLSF and LSF-to-LPC conversion with stability checks, LTP, gains, stereo prediction, the decoder core, and the SILK resampler to the API rate;
- CELT (4.3): coarse/fine energy, bit allocation, PVQ with CWRS, band folding, spreading, TF change, anti-collapse, fixed-point IMDCT, pitch post-filter and de-emphasis;
- `opus_decoder.c` behaviour: hybrid mode, mode transitions with CELT redundancy frames and cross-fades.

Fixed-point libopus is the oracle because its decoder is integer-exact. The float build is not reproducible across implementations. SIMD is out of scope; correctness first.

## Acceptance criteria

- [x] All 12 RFC 8251 test vectors decode sample-exact with fixed-point libopus 1.3.1 `opus_demo`, at 48 kHz stereo and mono, with every packet's final range matching.
- [x] All 12 vectors pass `opus_compare` against the official `.dec` and `m.dec` reference outputs.
- [x] libopus-encoded Ogg Opus files (CELT, SILK and hybrid; 10/20/40/60 ms; mono and stereo) decode sample-exact with fixed-point libopus.
- [x] Mutated packets never panic or hang; the fuzz corpus includes every vector.
- [x] A workspace test decodes committed libopus-encoded packets and checks output hashes.

## 2026-10-28

Port of fixed-point libopus 1.3.1 (opus_decoder.c, celt/, silk/ decoder side, silk resampler) under crates/ragdoll-codec/src/opus/. Arithmetic follows the OPUS_FAST_INT64 x86-64 build: 64-bit SMULW*/SMLAW*/MULT16_32_Q15/MULT32_32_Q31 products, opus_val16 truncation at every assignment, wrapping where C wraps. Tables are generated from the C sources (gen scripts: celt_tables.py, silk_tables.py).

## 2026-10-28

Evidence: all 12 RFC 8251 vectors sample-exact with opus_demo at 8/12/16/24/48 kHz, mono and stereo (120 cases), every final range equal; opus_compare passes all 24 with libopus's own scores (tv01 98.9 %, tv02-04 100 %, tv05 91.7 % ...). 24 libopus-encoded streams (FEC/LBRR, DTX, 2.5-120 ms, restricted low delay, CBR/CVBR, force-mono) x clean/10 %/30 % loss x FEC on/off x mono/stereo: 288/288 exact, so PLC, CNG, FEC and glue are covered. OPUS_SET_GAIN at +-6 dB and the Q8 extremes: 40/40 exact. 72 FFmpeg-libopus Ogg files (voip/audio/lowdelay x 10/20/40/60 ms x 12/32/96 kbps x mono/stereo) through ragdoll_pipeline::decode_audio_bytes: 72/72 exact.

## 2026-10-28

Fuzzing: 432 corrupted streams (bit flips, truncation, extension, random TOC and code-3 headers, garbage payloads) match libopus sample for sample, including which packets are rejected; 2M random packets/frame sizes/FEC/PLC calls across every rate and channel count with overflow checks on: no panics, no hangs.

## 2026-10-28

Fixed on the way: the Ogg demuxer timed Opus packets by the TOC frame size only, ignoring the frame count of code 1-3 packets (40-120 ms packets got 20-60 ms durations); a format test had relied on that with a code-2 dummy packet. decode_audio_bytes now drops OpusHead pre-skip and the decoder applies the OpusHead output gain. End trimming to the final granule position is not done yet (follow-up item).

## Result

Decoder sample-exact with fixed-point libopus 1.3.1: all RFC 8251 vectors at every rate and channel count, FEC/PLC/DTX corpora, 72 Ogg files and 432 fuzzed streams. ragdoll_codec::opus::decoder::OpusPacketDecoder is the libopus-equivalent reference for encoder work: an encoder's packets are correct when this decoder's final_range equals the encoder's.
