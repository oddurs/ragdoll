//! Opus decoder output checked against fixed-point libopus 1.3.1.
//!
//! `tests/data/opus` holds streams in the `opus_demo` bitstream format (per packet: 4-byte
//! big-endian length, 4-byte big-endian encoder final range, payload). `tvNN.bit` are excerpts of
//! the RFC 8251 test vectors, each the window of packets with the most mode, bandwidth, frame
//! size and channel switches. `enc_*.bit` were encoded by libopus (`opus_demo -e`) to cover what
//! the vectors do not: in-band FEC (LBRR), DTX, 2.5 ms and 40 to 120 ms packets, and the
//! restricted low-delay mode. The vectors and libopus are BSD-3-Clause (`src/opus/COPYING.libopus`).
//!
//! Each case decodes at the given rate and channel count, optionally with packets dropped by
//! [`lost`] and optionally recovering them with in-band FEC, exactly as `opus_demo -d` does
//! (`-inbandfec` delays output by one packet). The expected values are the sample count and the
//! 64-bit FNV-1a hash of `opus_demo`'s 16-bit output. The final range of every packet decoded
//! normally must also equal the encoder's.

use ragdoll_codec::opus::decoder::OpusPacketDecoder;

type Case = (&'static str, i32, usize, bool, bool, usize, u64);

const CASES: [Case; 75] = [
    ("tv01", 48000, 2, false, false, 15000, 0x4be7034c0520a84d),
    ("tv01", 48000, 1, false, false, 15000, 0x01d34f992468821f),
    ("tv02", 48000, 2, false, false, 159840, 0x05ad87d5ed2bc3fa),
    ("tv02", 48000, 1, false, false, 159840, 0x18407c19aeb2fc8f),
    ("tv03", 48000, 2, false, false, 262080, 0x809f873a3eef7db1),
    ("tv03", 48000, 1, false, false, 262080, 0xb1449fdecd60e568),
    ("tv04", 48000, 2, false, false, 224640, 0x13319558be374c06),
    ("tv04", 48000, 1, false, false, 224640, 0xde513e7d3808e896),
    ("tv05", 48000, 2, false, false, 155040, 0x857537a4e794601d),
    ("tv05", 48000, 1, false, false, 155040, 0xeec00405147b9109),
    ("tv06", 48000, 2, false, false, 121440, 0x4056ef23a9dcc12a),
    ("tv06", 48000, 1, false, false, 121440, 0xd5a9afb0222ec212),
    ("tv07", 48000, 2, false, false, 178560, 0xfbbf32785d88e15d),
    ("tv07", 48000, 1, false, false, 178560, 0x40490136736e23cc),
    ("tv08", 48000, 2, false, false, 140160, 0xf28ac3b9c84b0245),
    ("tv08", 48000, 1, false, false, 140160, 0x7f5832e6c4987c4e),
    ("tv09", 48000, 2, false, false, 63240, 0xd4a120a81c783b37),
    ("tv09", 48000, 1, false, false, 63240, 0xba197cedbe973f4c),
    ("tv10", 48000, 2, false, false, 14640, 0xe181ce9660b8370e),
    ("tv10", 48000, 1, false, false, 14640, 0x2ff01fd3d77b093f),
    ("tv11", 48000, 2, false, false, 48000, 0xf863c88d974381f4),
    ("tv11", 48000, 1, false, false, 48000, 0xcae3a4ea88460ef6),
    ("tv12", 48000, 2, false, false, 247680, 0xaf09a29f9474b439),
    ("tv12", 48000, 1, false, false, 247680, 0xbe7aca33cf421316),
    (
        "enc_audio_hybrid_100ms",
        48000,
        2,
        false,
        false,
        124800,
        0x48ff44541fd08bb6,
    ),
    (
        "enc_audio_hybrid_100ms",
        48000,
        1,
        false,
        false,
        124800,
        0x5b98180f42041667,
    ),
    (
        "enc_celt_120ms",
        48000,
        2,
        false,
        false,
        97920,
        0xc0a2821796a404bd,
    ),
    (
        "enc_celt_120ms",
        48000,
        1,
        false,
        false,
        97920,
        0x85b33002cd33651c,
    ),
    (
        "enc_celt_2_5ms",
        48000,
        2,
        false,
        false,
        48120,
        0x019b4c6375e03a4e,
    ),
    (
        "enc_celt_2_5ms",
        48000,
        1,
        false,
        false,
        48120,
        0x5273fa6de89de4eb,
    ),
    (
        "enc_hybrid_fb_stereo_fec_40ms",
        48000,
        2,
        false,
        false,
        120960,
        0x0516489fff9a8ff0,
    ),
    (
        "enc_hybrid_fb_stereo_fec_40ms",
        48000,
        1,
        false,
        false,
        120960,
        0xcca4e21d8c2150e1,
    ),
    (
        "enc_lowdelay_10ms",
        48000,
        2,
        false,
        false,
        48480,
        0x82e19b8c079106b5,
    ),
    (
        "enc_lowdelay_10ms",
        48000,
        1,
        false,
        false,
        48480,
        0x36b151d0e40bc18a,
    ),
    (
        "enc_silk_mb_fec_40ms",
        48000,
        2,
        false,
        false,
        120960,
        0x584a5572651fa415,
    ),
    (
        "enc_silk_mb_fec_40ms",
        48000,
        1,
        false,
        false,
        120960,
        0xed9f09773744e37d,
    ),
    (
        "enc_silk_nb_fec_60ms",
        48000,
        2,
        false,
        false,
        120960,
        0x74e4fd56b803e425,
    ),
    (
        "enc_silk_nb_fec_60ms",
        48000,
        1,
        false,
        false,
        120960,
        0x5de6ffc917e165ba,
    ),
    (
        "enc_silk_wb_dtx",
        48000,
        2,
        false,
        false,
        120960,
        0x97b2f9a780834809,
    ),
    (
        "enc_silk_wb_dtx",
        48000,
        1,
        false,
        false,
        120960,
        0x4e3aeba84ab8f2fb,
    ),
    (
        "enc_silk_wb_stereo_fec",
        48000,
        2,
        false,
        false,
        120960,
        0x9217ab3fcf308fee,
    ),
    (
        "enc_silk_wb_stereo_fec",
        48000,
        1,
        false,
        false,
        120960,
        0x9a9f68d10f262374,
    ),
    ("tv01", 24000, 2, false, false, 7500, 0xb7c0ca46af747ae5),
    ("tv02", 8000, 1, false, false, 26640, 0xdeae5b3a2256c3a2),
    ("tv03", 12000, 2, false, false, 65520, 0x8241b2e6b1819435),
    ("tv05", 16000, 2, false, false, 51680, 0x13a6b874323d4f5d),
    ("tv08", 24000, 1, false, false, 70080, 0x1aee29ebd702506e),
    ("tv10", 8000, 2, false, false, 2440, 0x28c0a36d1530043a),
    ("tv12", 16000, 1, false, false, 82560, 0xd09619ee91b0b0ec),
    (
        "enc_celt_2_5ms",
        12000,
        1,
        false,
        false,
        12030,
        0x4acc361bf271749a,
    ),
    (
        "enc_silk_wb_stereo_fec",
        16000,
        2,
        false,
        false,
        40320,
        0xcb246c3096808d55,
    ),
    ("tv02", 48000, 2, true, false, 159840, 0x95175270c8e0ebdc),
    ("tv05", 48000, 2, true, false, 155040, 0xf3c79d8492faed67),
    ("tv08", 48000, 2, true, false, 142680, 0x751901e14f9bf70c),
    ("tv10", 48000, 2, true, false, 13680, 0x10299cc66470a423),
    ("tv11", 48000, 2, true, false, 40320, 0xe4a62706369b91a9),
    (
        "enc_audio_hybrid_100ms",
        48000,
        2,
        true,
        false,
        124800,
        0x8d9cf942f14180de,
    ),
    (
        "enc_celt_120ms",
        48000,
        2,
        true,
        false,
        97920,
        0x3c2a10003631a1f2,
    ),
    (
        "enc_celt_2_5ms",
        48000,
        2,
        true,
        false,
        48120,
        0xd435cbb33f0f242f,
    ),
    (
        "enc_hybrid_fb_stereo_fec_40ms",
        48000,
        2,
        true,
        false,
        120960,
        0xcbb86b793989763e,
    ),
    (
        "enc_lowdelay_10ms",
        48000,
        2,
        true,
        false,
        48480,
        0x6dd3fdea5914d73d,
    ),
    (
        "enc_silk_mb_fec_40ms",
        48000,
        2,
        true,
        false,
        120960,
        0xe3711e75b55692fd,
    ),
    (
        "enc_silk_nb_fec_60ms",
        48000,
        2,
        true,
        false,
        120960,
        0xb9a4305ff1052e45,
    ),
    (
        "enc_silk_wb_dtx",
        48000,
        2,
        true,
        false,
        120960,
        0x4e1b41c44b69ca7d,
    ),
    (
        "enc_silk_wb_stereo_fec",
        48000,
        2,
        true,
        false,
        120960,
        0xe239f4af4b8e1e77,
    ),
    (
        "enc_hybrid_fb_stereo_fec_40ms",
        48000,
        2,
        true,
        true,
        119040,
        0x51e78d34cfd9423e,
    ),
    (
        "enc_hybrid_fb_stereo_fec_40ms",
        48000,
        1,
        true,
        true,
        119040,
        0x27acb60675b11158,
    ),
    (
        "enc_silk_mb_fec_40ms",
        48000,
        2,
        true,
        true,
        119040,
        0x6abbc888a6615549,
    ),
    (
        "enc_silk_mb_fec_40ms",
        48000,
        1,
        true,
        true,
        119040,
        0xadc4ffb4f1297f80,
    ),
    (
        "enc_silk_nb_fec_60ms",
        48000,
        2,
        true,
        true,
        118080,
        0x080ae1565ba7b5a5,
    ),
    (
        "enc_silk_nb_fec_60ms",
        48000,
        1,
        true,
        true,
        118080,
        0xbed41a99ccef3033,
    ),
    (
        "enc_silk_wb_stereo_fec",
        48000,
        2,
        true,
        true,
        120000,
        0xaed219f21801f638,
    ),
    (
        "enc_silk_wb_stereo_fec",
        48000,
        1,
        true,
        true,
        120000,
        0x6537b043be365a1a,
    ),
    ("tv04", 48000, 2, true, true, 224160, 0xedde1150987e377e),
    ("tv04", 48000, 1, true, true, 224160, 0xe0f896584ba3c5d3),
];

/// The packet loss pattern: isolated losses plus a burst of four every 31 packets.
fn lost(i: usize) -> bool {
    i > 0 && (i % 7 == 3 || (20..24).contains(&(i % 31)))
}

fn read_bit(data: &[u8]) -> Vec<(&[u8], u32)> {
    let mut out = Vec::new();
    let mut off = 0;
    while off + 8 <= data.len() {
        let len = u32::from_be_bytes(data[off..off + 4].try_into().unwrap()) as usize;
        let rng = u32::from_be_bytes(data[off + 4..off + 8].try_into().unwrap());
        out.push((&data[off + 8..off + 8 + len], rng));
        off += 8 + len;
    }
    out
}

fn fnv1a(hash: &mut u64, samples: &[i16]) {
    for s in samples {
        for b in s.to_le_bytes() {
            *hash ^= b as u64;
            *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

#[test]
fn decodes_sample_exact_with_libopus() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/opus");
    for (name, rate, ch, loss, fec, samples, expected) in CASES {
        let data = std::fs::read(dir.join(format!("{name}.bit"))).unwrap();
        let mut packets = read_bit(&data);
        if loss {
            for (i, p) in packets.iter_mut().enumerate() {
                if lost(i) {
                    *p = (&[], 0);
                }
            }
        }
        let label = format!("{name} {rate} Hz {ch} ch loss={loss} fec={fec}");
        let mut dec = OpusPacketDecoder::new(rate, ch).unwrap();
        let max_frame = rate as usize * 120 / 1000;
        let mut pcm = vec![0i16; max_frame * ch];
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut total = 0;
        let mut lost_prev = true;
        for i in 0..packets.len() {
            let (data, _) = packets[i];
            let is_lost = data.is_empty();
            let packet = (!is_lost).then_some(data);
            let (n, checked_rng) = if fec {
                if i == 0 {
                    lost_prev = is_lost;
                    continue;
                }
                if lost_prev {
                    // Recover the previous packet from this one's LBRR data.
                    let n = dec.last_packet_duration() as usize;
                    (dec.decode(packet, &mut pcm, n, true).unwrap(), None)
                } else {
                    let (prev, rng) = packets[i - 1];
                    let n = dec.decode(Some(prev), &mut pcm, max_frame, false).unwrap();
                    (n, (!is_lost).then_some(rng))
                }
            } else {
                let n = if is_lost {
                    dec.last_packet_duration() as usize
                } else {
                    max_frame
                };
                let n = dec.decode(packet, &mut pcm, n, false).unwrap();
                (n, (!is_lost && !lost_prev).then_some(packets[i].1))
            };
            if let Some(rng) = checked_rng {
                assert_eq!(dec.final_range(), rng, "{label}: final range of packet {i}");
            }
            fnv1a(&mut hash, &pcm[..n * ch]);
            total += n;
            lost_prev = is_lost;
        }
        assert_eq!(total, samples, "{label}: sample count");
        assert_eq!(hash, expected, "{label}: output hash");
    }
}
