//! H.264 decoder output checked against the reference decoders.
//!
//! The streams in `tests/data/h264` were encoded with libx264 from FFmpeg's `testsrc2` pattern,
//! each exercising a different part of the decoder. The MBAFF streams use interlaced input (half
//! the picture scrolling vertically) so that field and frame macroblock pairs mix. The expected values are the 64-bit FNV-1a
//! hash of the JM 19.1 reference decoder's output (all frames, in output order, I420, cropped);
//! FFmpeg 6.1 produces byte-identical output for each.

use ragdoll_codec::H264Decoder;

const CASES: [(&str, usize, usize, usize, u64); 6] = [
    // Main: CABAC, 3 B-frames in a pyramid, explicit weighted prediction, 3 references.
    (
        "cabac_bpyramid_weighted.264",
        128,
        96,
        12,
        0x14d68b900013eadb,
    ),
    // High: temporal direct, 8x8 transform, JVT default scaling matrices.
    (
        "cabac_temporal_direct_8x8_cqm.264",
        128,
        96,
        12,
        0x92ad1aa292f98a94,
    ),
    // Baseline: CAVLC, 3 references, 2 slices per picture, IDR every 6 frames.
    (
        "cavlc_baseline_multiref_slices.264",
        128,
        96,
        12,
        0xa048ac9b8b37c261,
    ),
    // High with CAVLC and B-frames; 90x58 needs frame cropping.
    (
        "cavlc_high_cropped_90x58.264",
        90,
        58,
        12,
        0x2b092e8ae3ffbdde,
    ),
    // MBAFF, CABAC, top field first: temporal direct, B pyramid, weighted prediction, 8x8 transform.
    (
        "cabac_mbaff_temporal_weighted.264",
        128,
        96,
        12,
        0x22508a1cfed19049,
    ),
    // MBAFF, CAVLC, bottom field first: spatial direct, 2 references, 8x8 transform.
    ("cavlc_mbaff_spatial.264", 128, 96, 12, 0x05629c5f77e4e6be),
];

fn fnv1a(hash: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *hash ^= b as u64;
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

#[test]
fn decodes_bit_exact_with_reference_decoders() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/h264");
    for (name, w, h, frames, expected) in CASES {
        let data = std::fs::read(dir.join(name)).unwrap();
        let out = H264Decoder::new().decode_stream_pts_ordered(&data).unwrap();
        assert_eq!(out.len(), frames, "{name}: frame count");
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for f in &out {
            assert_eq!((f.width, f.height), (w, h), "{name}: dimensions");
            for p in 0..3 {
                let (pw, ph) = if p == 0 {
                    (w, h)
                } else {
                    (w.div_ceil(2), h.div_ceil(2))
                };
                for y in 0..ph {
                    fnv1a(&mut hash, &f.plane(p).row(y)[..pw]);
                }
            }
        }
        assert_eq!(
            hash, expected,
            "{name}: decoded output differs from the reference decoders"
        );
    }
}
