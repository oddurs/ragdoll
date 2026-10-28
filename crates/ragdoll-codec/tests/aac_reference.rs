//! AAC-LC decoder output checked against FFmpeg's float decoder.
//!
//! `tests/data/aac` holds short ADTS streams from FFmpeg's native encoder and FFmpeg's decoded
//! output as interleaved s16le, both made by `tools/aac_fixtures/gen.sh`. Between them they cover
//! every window sequence and shape switch, all eleven spectral codebooks, M/S (`ms_mask_present`
//! 0, 1 and 2), intensity stereo in both phases, TNS, sample rates from 8 to 96 kHz and the 5.1
//! channel configuration (output reordered to FL, FR, FC, LFE, BL, BR).
//!
//! The decoders agree to about 2^-23 of full scale, far inside the ISO/IEC 14496-26 bounds, so
//! after rounding to 16 bits a sample may differ by one LSB only where FFmpeg's value lies within
//! that distance of a rounding boundary. Every sample must be within one LSB and at most one in a
//! hundred may differ at all; a decoding error moves whole bands and fails both by far.

use ragdoll_codec::{AacDecoder, AdtsHeader};

const CASES: [(&str, u32, usize); 7] = [
    ("mono_8k", 8000, 1),
    ("mono_22k_short", 22050, 1),
    ("stereo_48k_tools", 48000, 2),
    ("stereo_44k_ms", 44100, 2),
    ("stereo_32k_320k", 32000, 2),
    ("stereo_96k", 96000, 2),
    ("surround51_48k", 48000, 6),
];

fn data(name: &str) -> Vec<u8> {
    let path = format!("{}/tests/data/aac/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

#[test]
fn aac_matches_ffmpeg_float_decoder() {
    for (name, rate, channels) in CASES {
        let stream = data(&format!("{name}.aac"));
        let reference: Vec<i16> = data(&format!("{name}.s16"))
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| i16::from_le_bytes(b))
            .collect();

        let mut dec = AacDecoder::new(rate, channels as u8);
        let mut out: Vec<i16> = Vec::new();
        let mut pos = 0;
        while pos < stream.len() {
            let hdr = AdtsHeader::parse(&stream[pos..]).unwrap();
            let len = hdr.frame_length as usize;
            let frame = dec.decode_packet(&stream[pos..pos + len]).unwrap();
            assert_eq!(frame.sample_rate, rate, "{name}");
            assert_eq!(frame.channels as usize, channels, "{name}");
            // Rounded as FFmpeg's float to s16 conversion does.
            out.extend(
                frame
                    .as_f32_slice()
                    .iter()
                    .map(|&v| (v * 32768.0).round_ties_even().clamp(-32768.0, 32767.0) as i16),
            );
            pos += len;
        }

        assert_eq!(out.len(), reference.len(), "{name}: sample count");
        let mut differing = 0;
        for (i, (&a, &b)) in out.iter().zip(&reference).enumerate() {
            let d = (a as i32 - b as i32).abs();
            assert!(
                d <= 1,
                "{name}: sample {} channel {} is {a}, FFmpeg {b}",
                i / channels,
                i % channels
            );
            differing += (d != 0) as usize;
        }
        assert!(
            differing * 100 <= out.len(),
            "{name}: {differing} of {} samples differ by one LSB",
            out.len()
        );
    }
}
