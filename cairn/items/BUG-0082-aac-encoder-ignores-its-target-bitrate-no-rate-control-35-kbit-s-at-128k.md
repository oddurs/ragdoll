---
id: 82
uid: 18c32a99-2a85-4ec6-b621-783738fdbbd3
title: AAC encoder ignores its target bitrate (no rate control; 35 kbit/s at 128k)
type: bug
status: done
milestone: m11-validation
created: 2026-10-28
updated: 2026-10-29
closed_at: 2026-10-29
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

- [x] On speech, music and synthetic signals the average bitrate is within 10 % of the target from 32 to 320 kbit/s.
- [x] No frame exceeds the bit reservoir (6144 bits per channel).
- [x] A regression test reproduces the failure and passes with the fix.

## 2026-10-29

Fixed in aac/encoder.rs (new; the encoder left aac/mod.rs, which keeps the shared tables): per-band scale factors sf = g + round(2 * 0.25 * log2(rms_b)) (noise grows slowly with band level; beta 0 gave ~1 dB more SNR, beta 0.25 kept for some perceptual shaping), per-band M/S where E_M * E_S < 0.9 E_L * E_R, codebooks 1-11 chosen by DP over bands with section costs, sf deltas kept within +-60 by raising, |q| <= 8191. Rate control: bit reservoir capped at 6144 bits/channel minus the average; each frame's offset g is bisected on exact bit counts into [mean + R - Rmax, min(mean + R, 6144 * ch)] (ADTS header counted for encode_adts_frame), starting from an offset carried from active frames (corrected towards average + 1/8 of the excess above half reservoir; near-silent frames leave it alone, otherwise the next loud frame drained the reservoir: speech at 32k went 4.5 -> 10.5 dB SNR). Fill elements (EXT_FILL) pad only when even the finest step cannot use the bits. Bitrate clamped to 6144 bits/channel/frame (288 kbit/s per channel at 48 kHz).

## 2026-10-29

Sweep (20 s cuts of perf/in/music48 and the lavfi speech/tonal signals at 44.1 kHz, plus this item's 4 s synthetic repro; ADTS size; SNR after FFmpeg decode, identical after Ragdoll decode): Ragdoll within -4.8..-0.1 % of target at 32/64/96/128/192/256/320 kbit/s on all four (worst synthetic 32k: 30.5); largest frame 12152 bits (stereo bound 12288). SNR vs FFmpeg's native aac at the same -b:a: music +6.7..+14.0 dB, speech +3.4..+11.9, tonal +19..+29, synthetic +13..+46 (FFmpeg itself lands -23 %..+15 % off target). The repro now gives 126.3 kbit/s for 128k (was 35). Encode speed: 278 ms vs FFmpeg aac 940 ms for 60 s of music48 at 128k. Regression test aac::encoder::tests::bitrate_tracks_target (synthetic/music/speech at 32-320 kbit/s within 10 %, every frame <= 12288 bits, SNR rising with bitrate, FFmpeg decodes). Not done: short windows/TNS (transients pre-echo; SNR-neutral), a real psychoacoustic model (masking/ATH), PNS/intensity; SNR is the only quality metric compared.
