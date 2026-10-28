//! Ogg Opus files written by FFmpeg's libopus encoder, decoded through the pipeline, checked
//! against fixed-point libopus 1.3.1.
//!
//! `voip_mono_60ms.opus` carries 60 ms packets of several frames each (TOC code 1-3).
//! `audio_stereo_10ms.opus` has its `OpusHead` output gain set to -2 dB (-512 in Q8).
//! The expected values are the sample count and the 64-bit FNV-1a hash of the 16-bit output of
//! `opus_decode` with `OPUS_SET_GAIN(output_gain)`, after dropping `pre_skip` samples and the
//! samples past the final granule position: 1.5 s and 1.0 s, the lengths FFmpeg encoded.

use ragdoll_pipeline::decode_audio_bytes;

const CASES: [(&str, u8, usize, u64); 2] = [
    ("voip_mono_60ms.opus", 1, 72000, 0xc8aff96267cd1738),
    ("audio_stereo_10ms.opus", 2, 48000, 0xdddd70acf745b329),
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

/// Ragdoll's own Ogg Opus output: the final granule position trims the end padding (cairn
/// BUG-0080), so a stream that is not a whole number of 20 ms frames decodes to exactly its input
/// length, aligned by the pre-skip; and pages carry many packets (BUG-0081).
#[test]
fn ogg_opus_roundtrip_keeps_length_and_pages_packets() {
    use ragdoll_core::{AudioFrame, SampleFormat};
    use ragdoll_format::{OggDemuxer, OggMuxer};
    use ragdoll_pipeline::encode_audio_track_packets;

    let n = 50_000;
    let mut frame = AudioFrame::new(48_000, 2, SampleFormat::F32le, n);
    for (i, s) in frame.as_f32_mut_slice().chunks_exact_mut(2).enumerate() {
        let t = i as f64 / 48_000.0;
        s[0] = ((2.0 * std::f64::consts::PI * 440.0 * t).sin() * 0.5) as f32;
        s[1] = ((2.0 * std::f64::consts::PI * (300.0 + 2000.0 * t) * t).sin() * 0.4) as f32;
    }
    let input = frame.as_f32_slice().to_vec();
    let (info, pkts) = encode_audio_track_packets(frame, "opus", 0).unwrap();
    let payload: usize = pkts.iter().map(|p| p.len()).sum();
    let mut mux = OggMuxer::new();
    mux.add_track(info);
    for p in &pkts {
        mux.write_packet(p).unwrap();
    }
    let mut file = Vec::new();
    mux.finish(&mut file).unwrap();

    let demux = OggDemuxer::open(&file).unwrap();
    assert_eq!(
        demux.final_granule(),
        Some(n as i64 + 120),
        "granule counts pre-skip + input"
    );
    assert_eq!(demux.track().duration_ticks, Some(n as i64));
    // Headers (OpusHead, OpusTags) aside, framing must stay under 1 % of the payload.
    let pages = file.windows(4).filter(|w| w == b"OggS").count();
    let audio_overhead = file.len() - payload;
    assert!(pages < 12, "{pages} pages for {} packets", pkts.len());
    assert!(
        (audio_overhead as f64) < 0.01 * payload as f64 + 200.0,
        "{audio_overhead} bytes of Ogg framing for {payload} bytes of Opus"
    );

    let (out, _) = decode_audio_bytes(&file).unwrap();
    assert_eq!(out.num_samples, n);
    let (mut sig, mut err) = (0.0f64, 0.0f64);
    for (&x, &y) in input.iter().zip(out.as_f32_slice()).skip(2 * 960) {
        sig += (x as f64) * (x as f64);
        err += ((x - y) as f64) * ((x - y) as f64);
    }
    let snr = 10.0 * (sig / err).log10();
    assert!(snr > 20.0, "roundtrip SNR {snr:.1} dB");
}
