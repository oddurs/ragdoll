//! Opus encoder output checked against fixed-point libopus 1.3.1.
//!
//! The input is a deterministic integer test signal (a triangle wave, a square wave with
//! periodic level bursts, LCG noise and clicks), so no audio file is needed. Each case encodes it
//! as `opus_demo -e restricted-lowdelay 48000 CH BPS [-cbr|-cvbr] -complexity CX -framesize F`
//! does, writing the same bitstream format (per packet: 4-byte big-endian length, 4-byte
//! big-endian final range, payload). The expected values are the packet count, the payload bytes
//! and the 64-bit FNV-1a hash of `opus_demo`'s output file, so every packet and final range must
//! be byte-identical. Every packet is also decoded, and the decoder's final range must equal the
//! encoder's.

use ragdoll_codec::opus::decoder::OpusPacketDecoder;
use ragdoll_codec::opus::encoder::OpusPacketEncoder;

#[derive(Clone, Copy, PartialEq)]
enum Rate {
    Vbr,
    Cvbr,
    Cbr,
}

type Case = (usize, i32, usize, Rate, i32, usize, usize, u64);

const CASES: [Case; 12] = [
    (2, 64000, 960, Rate::Vbr, 9, 101, 18096, 0x821cfbfb521b708c),
    (
        2,
        128000,
        480,
        Rate::Cvbr,
        9,
        201,
        32463,
        0x20265ab141badc1b,
    ),
    (2, 32000, 1920, Rate::Cbr, 9, 51, 8160, 0xdc4ba34cb4eb58f0),
    (1, 24000, 960, Rate::Cvbr, 9, 101, 6188, 0xa7910584879805d2),
    (1, 96000, 120, Rate::Vbr, 9, 801, 24821, 0x6087459c370f889d),
    (2, 256000, 240, Rate::Cbr, 8, 401, 64160, 0x987cc87c7a3f38fc),
    (2, 48000, 960, Rate::Cvbr, 5, 101, 12287, 0x3ae32c1e52029c64),
    (1, 16000, 2880, Rate::Vbr, 3, 34, 4061, 0xd72eeb93ec40d70f),
    (2, 12000, 5760, Rate::Cvbr, 9, 17, 3149, 0xdbf8b8c669a5b160),
    (2, 6000, 960, Rate::Vbr, 9, 101, 1198, 0x31c819e50539af1a),
    (
        2,
        510000,
        960,
        Rate::Vbr,
        0,
        101,
        123855,
        0x07abf80d1a20d0e6,
    ),
    (1, 40000, 480, Rate::Cbr, 7, 201, 10050, 0xa5ec8f26e312449a),
];

const LEN: usize = 96000;

fn signal(ch: usize) -> Vec<i16> {
    let mut seed = 1u32;
    let mut out = Vec::with_capacity(LEN * ch);
    for n in 0..LEN as i32 {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let noise = (((seed >> 16) & 0xffff) as i32 - 32768) >> 3;
        let v = n % 109;
        let tri = if v < 55 { v } else { 109 - v } * 300 - 8000;
        let sq = if (n / 240) % 2 == 0 { 3000 } else { -3000 };
        let burst = if (n / 4800) % 5 == 0 { 4 } else { 1 };
        let click = if n % 12000 < 3 { 20000 } else { 0 };
        let l = (tri + sq * burst + noise + click).clamp(-32768, 32767);
        let r = ((tri >> 1) - sq + (noise >> 1) - click).clamp(-32768, 32767);
        if ch == 2 {
            out.extend([l as i16, r as i16]);
        } else {
            out.push(((l + r) >> 1).clamp(-32768, 32767) as i16);
        }
    }
    out
}

fn fnv1a(hash: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *hash ^= b as u64;
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

#[test]
fn encodes_byte_identical_with_libopus() {
    for (ch, bps, frame, rate, complexity, packets, bytes, expected) in CASES {
        let label = format!("{ch} ch {bps} bit/s {frame} samples complexity {complexity}");
        let pcm = signal(ch);
        let mut enc = OpusPacketEncoder::new(ch).unwrap();
        enc.set_bitrate(bps);
        enc.set_vbr(rate != Rate::Cbr);
        enc.set_vbr_constraint(rate == Rate::Cvbr);
        enc.set_complexity(complexity);
        enc.set_lsb_depth(16);
        let mut dec = OpusPacketDecoder::new(48000, ch).unwrap();
        let mut out = vec![0u8; 1500];
        let mut decoded = vec![0i16; 5760 * ch];
        let (mut hash, mut count, mut total) = (0xcbf2_9ce4_8422_2325u64, 0, 0);
        let mut pos = 0;
        loop {
            // Like opus_demo: the last frame is zero-padded, and a stream that ends on a frame
            // boundary gets one more all-zero frame.
            let take = (LEN - pos).min(frame);
            let mut input = vec![0i16; frame * ch];
            input[..take * ch].copy_from_slice(&pcm[pos * ch..(pos + take) * ch]);
            let n = enc.encode(&input, frame, &mut out).unwrap();
            let rng = enc.final_range();
            fnv1a(&mut hash, &(n as u32).to_be_bytes());
            fnv1a(&mut hash, &rng.to_be_bytes());
            fnv1a(&mut hash, &out[..n]);
            dec.decode(Some(&out[..n]), &mut decoded, 5760, false)
                .unwrap();
            if rng != 0 {
                assert_eq!(
                    dec.final_range(),
                    rng,
                    "{label}: decoder final range, packet {count}"
                );
            }
            count += 1;
            total += n;
            pos += take;
            if take < frame {
                break;
            }
        }
        assert_eq!(
            (count, total),
            (packets, bytes),
            "{label}: packets and bytes"
        );
        assert_eq!(hash, expected, "{label}: bitstream hash");
    }
}
