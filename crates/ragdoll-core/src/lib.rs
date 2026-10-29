//! # `ragdoll-core`
//!
//! Foundational multimedia types and zero-allocation memory management for Ragdoll:
//! - [`time`]: 128-bit overflow-safe `Rational64` timebases and `Timestamp` rescaling.
//! - [`color`]: ITU-T H.273 (CICP) color primaries, transfer functions, matrices, and perceptual `Oklab`.
//! - [`format`]: Video `PixelFormat` and audio `SampleFormat` / `ChannelLayout` descriptors.
//! - [`frame`]: 64-byte cache-line-aligned `AlignedBuffer`, `Plane`, `VideoFrame`, `AudioFrame`, and lock-free `FramePool`.
//! - [`packet`]: Zero-copy `PacketSlice<'mmap>` views and 24-byte cache-packed `SampleEntry` index items.

pub mod bitstream;
pub mod color;
pub mod crc;
pub mod dispatch;
pub mod error;
pub mod format;
pub mod frame;
pub mod packet;
pub mod power;
pub mod testing;
pub mod time;

pub use bitstream::{
    BitReader, BitWriter, escape_rbsp, read_ebml_vint, read_leb128, unescape_rbsp,
    unescape_rbsp_into, unescape_rbsp_scalar, write_ebml_vint, write_leb128,
};
pub use color::{
    ChromaLocation, Chromaticities, ColorPrimaries, ColorRange, Colorimetry, MatrixCoefficients,
    TransferCharacteristics,
};
pub use crc::{
    adler32, crc8_flac, crc8_flac_update, crc16_flac, crc16_flac_update, crc16_flac_update_scalar,
    crc32_lsb_update, crc32_mpeg2, crc32_msb_update, crc32_ogg, crc32_png,
};
pub use error::{RagdollError, Result};
pub use format::{ChannelLayout, PixelFormat, SampleFormat};
pub use frame::{
    AlignedBuffer, AudioFrame, CACHE_LINE_ALIGN, FramePool, INPUT_BUFFER_PADDING_SIZE, PictureType,
    Plane, PooledVideoFrame, SharedPlanes, VideoFrame, align_up_64, optimal_plane_stride,
};
pub use packet::{
    AudioTrackParams, CodecId, PacketFlags, PacketPayload, PacketSlice, SampleEntry, TrackInfo,
    TrackType, VideoTrackParams,
};
pub use power::{
    CpuTopology, HardwareMetrics, PowerMode, ProcessRusageSnapshot,
    apply_current_power_mode_to_thread, apply_power_mode_to_current_thread, global_power_mode,
    measure_hardware_metrics, parse_macos_time_l_stderr, set_global_power_mode, with_power_mode,
};
pub use time::{
    NOPTS_VALUE, Rational64, Rounding, Timestamp, bessel_i0, compare_ts, rescale_delta, rescale_q,
    rescale_q_rnd, rescale_rnd,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rational64_reduction_and_exact_ordering() {
        let r1 = Rational64::new(48_000, 2002);
        assert_eq!(r1.num, 24_000);
        assert_eq!(r1.den, 1001);
        assert_eq!(r1, Rational64::FPS_23_976);

        let neg = Rational64::new(30, -10);
        assert_eq!(neg.num, -3);
        assert_eq!(neg.den, 1);

        assert!(Rational64::FPS_23_976 < Rational64::FPS_24);
        assert!(Rational64::FPS_59_94 < Rational64::FPS_60);

        let sum = Rational64::new(1, 3) + Rational64::new(1, 6);
        assert_eq!(sum, Rational64::new(1, 2));

        let prod = Rational64::FPS_29_97 * Rational64::new(1001, 30_000);
        assert_eq!(prod, Rational64::ONE);
    }

    #[test]
    fn timestamp_128bit_rescaling_without_overflow() {
        // 48 hours at 90 kHz MPEG transport clock = 15,552,000,000 ticks.
        // Converting to nanoseconds (1 / 1_000_000_000) would overflow 64-bit (ticks * 10^9 > 2^63),
        // but succeeds cleanly via 128-bit intermediate arithmetic.
        let mpeg_ticks: i64 = 48 * 3600 * 90_000;
        let ts_mpeg = Timestamp::new(mpeg_ticks, Rational64::TIMEBASE_MPEG_90K);
        let ts_ns = ts_mpeg.rescale_to(Rational64::TIMEBASE_NANOS);
        assert_eq!(ts_ns.ticks, 48 * 3600 * 1_000_000_000_i64);
        assert_eq!(ts_mpeg, ts_ns);
        assert_eq!(ts_ns.format_hms(), "48:00:00.000");

        let parsed = Timestamp::parse_hms("01:23:45.678").unwrap();
        assert_eq!(parsed.format_hms(), "01:23:45.678");
    }

    #[test]
    fn rounding_modes_precision() {
        // 5 * 1 / 2 = 2.5
        assert_eq!(rescale_rnd(5, 1, 2, Rounding::Zero), 2);
        assert_eq!(rescale_rnd(5, 1, 2, Rounding::Inf), 3);
        assert_eq!(rescale_rnd(5, 1, 2, Rounding::Down), 2);
        assert_eq!(rescale_rnd(5, 1, 2, Rounding::Up), 3);
        assert_eq!(rescale_rnd(5, 1, 2, Rounding::NearInf), 3);

        // -5 * 1 / 2 = -2.5
        assert_eq!(rescale_rnd(-5, 1, 2, Rounding::Zero), -2);
        assert_eq!(rescale_rnd(-5, 1, 2, Rounding::Inf), -3);
        assert_eq!(rescale_rnd(-5, 1, 2, Rounding::Down), -3);
        assert_eq!(rescale_rnd(-5, 1, 2, Rounding::Up), -2);
        assert_eq!(rescale_rnd(-5, 1, 2, Rounding::NearInf), -3);
    }

    #[test]
    fn plane_and_frame_pool_64_byte_cache_line_alignment() {
        let pool = FramePool::new(8);
        assert_eq!(pool.allocations(), 0);
        assert_eq!(pool.recycled_hits(), 0);

        let ptrs: Vec<usize>;
        {
            // Odd resolution (1919x1079) to stress stride padding
            let mut frame = pool.acquire_scoped(1919, 1079, PixelFormat::Yuv420p);
            assert_eq!(frame.planes.len(), 3);
            for plane in &mut frame.planes {
                assert_eq!(plane.as_ptr() as usize % CACHE_LINE_ALIGN, 0);
                assert_eq!(plane.stride() % CACHE_LINE_ALIGN, 0);
                assert!(plane.stride() >= plane.row_bytes());
                // Every scanline base address is 64-byte aligned
                for y in 0..plane.height() {
                    let row_ptr = plane.row_padded(y).as_ptr() as usize;
                    assert_eq!(row_ptr % CACHE_LINE_ALIGN, 0);
                }
            }
            ptrs = frame.planes.iter().map(|p| p.as_ptr() as usize).collect();
        }
        assert_eq!(pool.allocations(), 1);
        assert_eq!(pool.available_in_pool(), 1);

        // Re-acquiring 100 times in a loop must hit the recycled pool 100 times with 0 new heap allocations
        for _ in 0..100 {
            let frame = pool.acquire_scoped(1919, 1079, PixelFormat::Yuv420p);
            let current_ptrs: Vec<usize> =
                frame.planes.iter().map(|p| p.as_ptr() as usize).collect();
            assert_eq!(current_ptrs, ptrs);
        }
        assert_eq!(pool.allocations(), 1);
        assert_eq!(pool.recycled_hits(), 100);
    }

    #[test]
    fn sample_entry_24_byte_packing_and_negative_cts() {
        assert_eq!(std::mem::size_of::<SampleEntry>(), 24);
        let entry_kf = SampleEntry::new(0x1234_5678_9ABC, 90_000, 65_536, -3003, true);
        assert!(entry_kf.is_keyframe());
        assert_eq!(entry_kf.offset, 0x1234_5678_9ABC);
        assert_eq!(entry_kf.pts, 90_000);
        assert_eq!(entry_kf.size, 65_536);
        assert_eq!(entry_kf.cts_delta(), -3003);
        assert_eq!(entry_kf.dts(), 93_003);

        let entry_p = SampleEntry::new(4096, 120_000, 4096, 6006, false);
        assert!(!entry_p.is_keyframe());
        assert_eq!(entry_p.cts_delta(), 6006);
        assert_eq!(entry_p.dts(), 113_994);
    }

    #[test]
    fn colorimetry_cicp_and_luma_weights() {
        let bt709 = Colorimetry::bt709_limited();
        let (kr, kg, kb) = bt709.matrix.luma_coefficients();
        assert!((kr + kg + kb - 1.0).abs() < 1e-12);

        let hdr = Colorimetry::bt2020_pq_hdr10();
        assert!(hdr.transfer.is_hdr());
        assert_eq!(hdr.chroma_location, ChromaLocation::TopLeft);
    }

    #[test]
    fn bitstream_arbitrary_widths_1_to_64_roundtrip() {
        let mut writer = BitWriter::with_capacity(1024);
        let mut expected = Vec::new();

        for width in 1..=64u32 {
            let mask = if width == 64 {
                u64::MAX
            } else {
                (1u64 << width) - 1
            };
            let val = 0xA5A5_5A5A_F00D_CAFE_u64.wrapping_mul(width as u64 + 17) & mask;
            writer.write_bits(val, width);
            expected.push((width, val));
        }
        let bytes = writer.finish();

        let mut reader = BitReader::new(&bytes);
        for (width, exp) in expected {
            let got = reader.read_bits(width).unwrap();
            assert_eq!(got, exp, "mismatch at bit width {width}");
        }
    }

    #[test]
    fn bitstream_exp_golomb_ue_and_se_clz_roundtrip() {
        let mut writer = BitWriter::with_capacity(4096);
        let ue_vals: Vec<u64> = (0..300)
            .chain([1023, 1024, 65535, 65536, 1_048_575])
            .collect();
        let se_vals: Vec<i64> = (-200..=200)
            .chain([-32768, 32767, -500_000, 500_000])
            .collect();

        for &u in &ue_vals {
            writer.write_ue(u);
        }
        for &s in &se_vals {
            writer.write_se(s);
        }
        writer.write_rbsp_trailing_bits();
        let bytes = writer.finish();

        let mut reader = BitReader::new(&bytes);
        for &u in &ue_vals {
            assert_eq!(reader.read_ue().unwrap(), u);
        }
        for &s in &se_vals {
            assert_eq!(reader.read_se().unwrap(), s);
        }
    }

    #[test]
    fn bitstream_golomb_rice_signed_and_unsigned_roundtrip() {
        for k in 0..=12u32 {
            let mut writer = BitWriter::with_capacity(2048);
            let residuals: Vec<i64> = (-128..=128).chain([-4095, 4096, -16384, 16384]).collect();
            for &r in &residuals {
                writer.write_rice_signed(r, k);
            }
            let bytes = writer.finish();
            let mut reader = BitReader::new(&bytes);
            for &r in &residuals {
                assert_eq!(reader.read_rice_signed(k).unwrap(), r, "Rice k={k}");
            }
        }
    }

    #[test]
    fn rbsp_neon_vs_scalar_differential_and_roundtrip() {
        for len in [1usize, 3, 15, 16, 17, 31, 32, 63, 64, 127, 513] {
            let mut raw = Vec::with_capacity(len);
            for i in 0..len {
                // Inject periodic 0x00 0x00 [0x00..=0x03] sequences
                let b = match i % 11 {
                    3 | 4 => 0x00,
                    5 => (i % 4) as u8,
                    _ => ((i * 37 + 13) & 0xFF) as u8,
                };
                raw.push(b);
            }
            let escaped = escape_rbsp(&raw);
            let unescaped_simd = unescape_rbsp(&escaped);
            let unescaped_scalar = unescape_rbsp_scalar(&escaped);
            assert_eq!(unescaped_simd, unescaped_scalar);
            assert_eq!(unescaped_simd, raw);
        }
    }

    #[test]
    fn leb128_and_ebml_vint_roundtrip() {
        for &val in &[0u64, 1, 127, 128, 16383, 16384, 1_000_000, 0x1234_5678] {
            let mut buf = Vec::new();
            write_leb128(val, &mut buf);
            let (decoded, consumed) = read_leb128(&buf).unwrap();
            assert_eq!(decoded, val);
            assert_eq!(consumed, buf.len());

            let mut ebml = Vec::new();
            write_ebml_vint(val, &mut ebml);
            let (_raw_id, stripped, width) = read_ebml_vint(&ebml).unwrap();
            assert_eq!(stripped, val);
            assert_eq!(width, ebml.len());
        }
    }

    #[test]
    fn continued_fraction_reduce_d2q_and_bessel_i0() {
        // Recover NTSC 29.97 fps (30000 / 1001) from float with max denominator 100_000
        let ntsc = 30_000.0 / 1001.0;
        let q = Rational64::from_f64(ntsc, 100_000);
        assert_eq!(q, Rational64::FPS_29_97);

        // Pi convergent with max <= 400 -> 355 / 113; semiconvergent with max <= 200 -> 179 / 57
        let (pi_355, exact) =
            Rational64::reduce_with_max(3_141_592_653_589, 1_000_000_000_000, 400);
        assert!(!exact);
        assert_eq!(pi_355, Rational64::new(355, 113));

        let (pi_semi, exact_semi) =
            Rational64::reduce_with_max(3_141_592_653_589, 1_000_000_000_000, 200);
        assert!(!exact_semi);
        assert_eq!(pi_semi, Rational64::new(179, 57));

        // Nearer rational comparison
        let candidates = [
            Rational64::FPS_24,
            Rational64::FPS_23_976,
            Rational64::FPS_29_97,
            Rational64::FPS_30,
        ];
        assert_eq!(
            Rational64::new(2997, 100).find_nearest(&candidates),
            Some(2)
        );

        // Bessel I_0(0) == 1.0, strictly increasing for x > 0
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-14);
        assert!((bessel_i0(1.0) - 1.266_065_877_752_008_4).abs() < 1e-12);
        assert!((bessel_i0(20.0) - 43_558_282.559_553_5).abs() < 1e-4);
    }

    #[test]
    fn compare_ts_and_rescale_delta_stateful_audio() {
        assert_eq!(
            compare_ts(
                90_000,
                Rational64::TIMEBASE_MPEG_90K,
                48_000,
                Rational64::new(1, 48_000)
            ),
            std::cmp::Ordering::Equal
        );
        let mut last = NOPTS_VALUE;
        let out0 = rescale_delta(
            Rational64::TIMEBASE_MPEG_90K,
            0,
            Rational64::new(1, 48_000),
            1024,
            &mut last,
            Rational64::new(1, 48_000),
        );
        assert_eq!(out0, 0);
        assert_eq!(last, 1024);
    }

    #[test]
    fn dav1d_cache_set_stride_padding_and_edge_extension() {
        // 1024 and 2048 are multiples of 1024 -> padded by +64 bytes to avoid L1/L2 cache-set aliasing
        assert_eq!(optimal_plane_stride(1024), 1088);
        assert_eq!(optimal_plane_stride(2048), 2112);
        // 1920 is 64-byte aligned and not a multiple of 1024 -> stays 1920
        assert_eq!(optimal_plane_stride(1920), 1920);

        let mut plane = Plane::new(10, 4, 1);
        for y in 0..4 {
            let row = plane.row_mut(y);
            for (x, p) in row.iter_mut().enumerate() {
                *p = (y * 10 + x + 1) as u8;
            }
        }
        plane.extend_right_edge();
        for y in 0..4 {
            let padded = plane.row_padded(y);
            let edge_val = (y * 10 + 10) as u8;
            for &sample in &padded[10..] {
                assert_eq!(sample, edge_val);
            }
        }
        // Tail safety padding is always 64 zero bytes
        assert_eq!(
            plane.buffer().tail_padding().len(),
            INPUT_BUFFER_PADDING_SIZE
        );
        assert!(plane.buffer().tail_padding().iter().all(|&b| b == 0));
    }

    #[test]
    fn bitstream_te_uek_uniform_and_more_rbsp_data() {
        let mut writer = BitWriter::new();
        // Truncated Exp-Golomb te(v) with max=1 and max=5
        writer.write_te(0, 1);
        writer.write_te(1, 1);
        writer.write_te(4, 5);
        // k-th order Exp-Golomb
        writer.write_ue_k(37, 3);
        writer.write_se_k(-19, 2);
        // AV1 non-power-of-two uniform coding
        for max in 2..=13u32 {
            for v in 0..max {
                writer.write_uniform(v, max);
            }
        }
        writer.write_rbsp_trailing_bits();
        let bytes = writer.finish();

        let mut reader = BitReader::new(&bytes);
        assert!(reader.more_rbsp_data());
        assert_eq!(reader.read_te(1).unwrap(), 0);
        assert_eq!(reader.read_te(1).unwrap(), 1);
        assert_eq!(reader.read_te(5).unwrap(), 4);
        assert_eq!(reader.read_ue_k(3).unwrap(), 37);
        assert_eq!(reader.read_se_k(2).unwrap(), -19);
        for max in 2..=13u32 {
            for v in 0..max {
                assert!(reader.more_rbsp_data());
                assert_eq!(reader.read_uniform(max).unwrap(), v);
            }
        }
        // Only the RBSP stop bit + zero padding remain -> more_rbsp_data() must be false!
        assert!(!reader.more_rbsp_data());
    }

    #[test]
    fn crc8_crc16_slice8_and_crc32_mpeg2_vectors() {
        let vec = b"123456789";
        // Standard test vectors for "123456789":
        // CRC-8 (poly=0x07, init=0x00) = 0xF4
        assert_eq!(crc8_flac(vec), 0xF4);
        // CRC-16/UMTS (BUYPASS, poly=0x8005, init=0x0000, MSB-first, no xorout) = 0xFEE8
        assert_eq!(crc16_flac(vec), 0xFEE8);
        // CRC-32/MPEG-2 (poly=0x04C11DB7, init=0xFFFFFFFF, MSB-first, no xorout) = 0x0376E6E7
        assert_eq!(crc32_mpeg2(vec), 0x0376_E6E7);

        // Differential check of Slicing-by-8 vs byte-at-a-time over 257-byte buffer
        let data: Vec<u8> = (0..257).map(|i| (i * 73 + 19) as u8).collect();
        let mut crc16_byte_by_byte = 0u16;
        for b in &data {
            crc16_byte_by_byte = crc16_flac_update(crc16_byte_by_byte, std::slice::from_ref(b));
        }
        assert_eq!(crc16_flac(&data), crc16_byte_by_byte);

        let mut crc32_byte_by_byte = 0xFFFF_FFFFu32;
        for b in &data {
            crc32_byte_by_byte = crc32_msb_update(crc32_byte_by_byte, std::slice::from_ref(b));
        }
        assert_eq!(crc32_mpeg2(&data), crc32_byte_by_byte);
    }
}
