//! Ogg Opus files written by FFmpeg's libopus encoder, decoded through the pipeline, checked
//! against fixed-point libopus 1.3.1.
//!
//! `voip_mono_60ms.opus` carries 60 ms packets of several frames each (TOC code 1-3).
//! `audio_stereo_10ms.opus` has its `OpusHead` output gain set to -2 dB (-512 in Q8).
//! The expected values are the sample count and the 64-bit FNV-1a hash of the 16-bit output of
//! `opus_decode` with `OPUS_SET_GAIN(output_gain)`, after dropping `pre_skip` samples.

use ragdoll_pipeline::decode_audio_bytes;

const CASES: [(&str, u8, usize, u64); 2] = [
    ("voip_mono_60ms.opus", 1, 74568, 0x1477c53321f1aea1),
    ("audio_stereo_10ms.opus", 2, 48168, 0x3da21346730b5908),
];

#[test]
fn decodes_ogg_opus_sample_exact_with_libopus() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/opus");
    for (name, channels, samples, expected) in CASES {
        let data = std::fs::read(dir.join(name)).unwrap();
        let (frame, _) = decode_audio_bytes(&data).unwrap();
        assert_eq!(frame.sample_rate, 48_000, "{name}");
        assert_eq!(frame.channels, channels, "{name}");
        assert_eq!(frame.num_samples, samples, "{name}: sample count");
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for &s in frame.as_f32_slice() {
            // The decoder's float output is s16 / 32768, so this is exact.
            let v = (s * 32768.0) as i16;
            for b in v.to_le_bytes() {
                hash ^= b as u64;
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        assert_eq!(hash, expected, "{name}: output hash");
    }
}
