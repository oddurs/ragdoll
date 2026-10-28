#!/usr/bin/env bash
# Regenerates the AAC-LC decoder fixtures in tests/data/aac (relative to the crate).
#
# Each case is a short ADTS stream from FFmpeg's native AAC encoder (PNS off: the noise
# generator is implementation-defined) and FFmpeg's float decoder output as s16le, which
# `tests/aac_reference.rs` compares against within one LSB.
#
# The click train (a 3 kHz burst every 50 ms) forces EIGHT_SHORT_SEQUENCE windows and
# their LONG_START / LONG_STOP transitions; the default encoder enables M/S, intensity
# stereo and TNS where they pay.
set -euo pipefail

out="$(cd "$(dirname "$0")/../.." && pwd)/tests/data/aac"
mkdir -p "$out"

tone='0.4*sin(2*PI*440*t)+0.2*sin(2*PI*(300+3000*t)*t)'
click='0.6*lt(mod(t\,0.05)\,0.002)*sin(2*PI*3000*t)'

gen() { # name rate channel-exprs seconds encoder-args...
    local name=$1 rate=$2 exprs=$3 secs=$4
    shift 4
    ffmpeg -hide_banner -loglevel error -y \
        -f lavfi -i "aevalsrc=${exprs}:s=${rate}:d=${secs}" \
        -c:a aac -aac_pns 0 "$@" -f adts "$out/$name.aac"
    ffmpeg -hide_banner -loglevel error -y -c:a aac -i "$out/$name.aac" -f s16le "$out/$name.s16"
}

gen mono_8k 8000 "$tone" 0.5 -b:a 16k
gen mono_22k_short 22050 "$tone+$click" 0.3 -b:a 48k
gen stereo_48k_tools 48000 "$tone+$click|0.7*($tone)-$click" 0.25 -b:a 128k
gen stereo_44k_ms 44100 "$tone|0.9*($tone)+0.05*sin(2*PI*5000*t)" 0.2 -b:a 96k -aac_ms 1
gen stereo_32k_320k 32000 "$tone+$click|$click-0.5*($tone)" 0.2 -b:a 320k
gen stereo_96k 96000 "$tone|$tone+$click" 0.05 -b:a 256k
gen surround51_48k 48000 \
    "$tone|0.8*($tone)|$click|0.3*sin(2*PI*60*t)|0.5*($tone)+$click|0.2*sin(2*PI*900*t)" \
    0.06 -b:a 256k
