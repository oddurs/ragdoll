//! # `ragdoll-simd`
//!
//! Explicit Apple Silicon `aarch64` NEON and `x86_64` AVX2 SIMD kernels with bit-exact scalar fallbacks.
//!
//! - [`scan`]: Vectorized Annex-B NAL start-code (`0x000001` / `0x00000001`) and MPEG-TS `0x47` sync-byte scanners.
//! - [`metrics`]: NEON-accelerated block/plane SAD, SSE, and single-pass $8\times8$ SSIM window statistics.
//! - [`dct4`]: Power-of-two DCT-IV (the AAC-LC IMDCT core) through a precomputed-twiddle complex FFT.
//! - [`h264_deblock`]: 16-line luma and 8-line chroma H.264 deblocking edges, transposed in registers for vertical edges.
//! - [`h264_mc`]: Partition-sized H.264 quarter-pel luma and eighth-pel chroma inter prediction.
//! - [`fixed`]: Q14 fixed-point polyphase dot products and $4\times4$ integer DCT/IDCT/Hadamard transforms.

pub mod audio_dsp;
pub mod dct4;
pub mod fixed;
pub mod h264_deblock;
pub mod h264_dsp;
pub mod h264_mc;
pub mod metrics;
pub mod scan;

pub use audio_dsp::{
    BiquadCoeffs, ITU_MIX_MINUS_3DB, MdctPlan, TRUE_PEAK_12X4_TAPS, apply_gain_true_peak_limit_f32,
    apply_gain_true_peak_limit_f32_scalar, biquad_process_4ch_f32, biquad_process_4ch_f32_scalar,
    fft_complex_in_place, kbd_window_f32, polyphase_dot_f32, polyphase_dot_f32_scalar,
    polyphase_dot16_f32, polyphase_dot16_f32_scalar, polyphase_dot16_q15_i16,
    polyphase_dot16_q15_i16_scalar, polyphase_dot32_f32, polyphase_dot32_f32_scalar,
    polyphase_dot32_q15_i16, polyphase_dot32_q15_i16_scalar, rematrix_5_1_to_stereo_f32,
    rematrix_5_1_to_stereo_f32_scalar, rematrix_mono_to_stereo_f32,
    rematrix_mono_to_stereo_f32_scalar, rematrix_nxm_f32, rematrix_nxm_f32_scalar,
    rematrix_stereo_to_mono_f32, rematrix_stereo_to_mono_f32_scalar, sine_window_f32,
    true_peak_4x_12tap_sample_f32, true_peak_4x_12tap_sample_f32_scalar, window_overlap_add_f32,
    window_overlap_add_f32_scalar,
};

pub use dct4::{Dct4Plan, imdct_window, imdct_window_scalar};
pub use fixed::{
    Q14_HALF, Q14_ONE, add4x4_idct_u8, clamp_u8, convolve_8tap_q14_u8, convolve_8tap_q14_u8_scalar,
    forward_dct4x4, forward_dct4x4_scalar, forward_dct8x8, hadamard4x4, hadamard4x4_scalar,
    inverse_idct4x4, inverse_idct4x4_scalar, inverse_idct8x8, sub4x4_dct_u8,
};
pub use h264_deblock::{
    h264_chroma_deblock, h264_chroma_deblock_scalar, h264_luma_deblock, h264_luma_deblock_scalar,
};
pub use h264_dsp::{
    H264_DEBLOCK_ALPHA, H264_DEBLOCK_BETA, H264_DEBLOCK_TC0, h264_deblock_chroma_line,
    h264_deblock_luma_line, h264_loop_filter_chroma_edge_h_u8, h264_loop_filter_chroma_edge_v_u8,
    h264_loop_filter_luma_edge_h_u8, h264_loop_filter_luma_edge_v_u8, predict_intra_4x4_u8,
    predict_intra_4x4_u8_scalar, put_h264_chroma_mc8_u8, put_h264_chroma_mc8_u8_scalar,
    put_h264_qpel8_mc_u8, put_h264_qpel16_mc_u8, put_h264_qpel16_mc_u8_scalar,
};
pub use h264_mc::{
    chroma_mc_cols, h264_chroma_mc, h264_chroma_mc_scalar, h264_luma_mc, h264_luma_mc_scalar,
    h264_weight_bi, h264_weight_bi_scalar, h264_weight_uni, h264_weight_uni_scalar, luma_mc_cols,
};
pub use metrics::{
    SSIM_C1_8BIT, SSIM_C2_8BIT, SsimWindowStats, flac_eval_fixed_orders_l1,
    flac_eval_fixed_orders_l1_scalar, sad_16x16_u8, sad_16x16_u8_scalar, sad_block_64_u8,
    sad_block_64_u8_scalar, sad_block_256_u8, sad_block_256_u8_scalar, sad_plane_u8,
    sad_plane_u8_scalar, satd_4x4_u8, satd_4x4_u8_scalar, satd_8x8_u8, sse_plane_u8,
    sse_plane_u8_scalar, ssim_4x4xn_u8, ssim_4x4xn_u8_scalar, ssim_end1_8bit, ssim_plane_8bit,
    ssim_to_db, ssim_window_8x8_u8, ssim_window_8x8_u8_scalar,
};
pub use scan::{
    annex_b_to_avcc, avcc_to_annex_b, find_annex_b_start_code, find_annex_b_start_code_scalar,
    find_mpeg_ts_sync, find_mpeg_ts_sync_scalar, split_annex_b_nals,
};

/// Active hardware SIMD instruction set level detected on the host CPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdLevel {
    /// ARM64 Advanced SIMD / NEON (128-bit x 32 vector registers, standard on Apple M-series).
    Neon,
    /// x86_64 AVX2 (256-bit YMM registers).
    Avx2,
    /// Portable scalar fallback.
    Scalar,
}

impl SimdLevel {
    #[inline]
    pub fn detect() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Self::Neon
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                Self::Avx2
            } else {
                Self::Scalar
            }
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Self::Scalar
        }
    }

    /// The level kernels dispatch to: [`Self::detect`], unless `RAGDOLL_SIMD` forces another
    /// (see [`ragdoll_core::dispatch`]).
    #[inline]
    pub fn active() -> Self {
        match ragdoll_core::dispatch::SimdDispatch::active() {
            ragdoll_core::dispatch::SimdDispatch::Neon => Self::Neon,
            ragdoll_core::dispatch::SimdDispatch::Avx2 => Self::Avx2,
            ragdoll_core::dispatch::SimdDispatch::Scalar => Self::Scalar,
        }
    }

    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Neon => "aarch64 NEON (128-bit x 32 ASIMD)",
            Self::Avx2 => "x86_64 AVX2 (256-bit)",
            Self::Scalar => "Scalar Reference",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn annex_b_start_code_neon_vs_scalar_across_all_alignments() {
        // Test start codes at every offset 0..=140 to stress 16-byte and 64-byte SIMD boundaries
        for offset in 0..=140usize {
            for four_byte in [false, true] {
                let mut buf = vec![0x5Au8; offset + 32];
                // Also test cases where false single 0x00 bytes appear before the start code
                if offset >= 7 {
                    buf[offset - 5] = 0x00;
                }
                if four_byte {
                    buf[offset] = 0x00;
                    buf[offset + 1] = 0x00;
                    buf[offset + 2] = 0x00;
                    buf[offset + 3] = 0x01;
                    buf[offset + 4] = 0x67; // SPS NAL header
                } else {
                    buf[offset] = 0x00;
                    buf[offset + 1] = 0x00;
                    buf[offset + 2] = 0x01;
                    buf[offset + 3] = 0x68; // PPS NAL header
                }

                let simd_res = find_annex_b_start_code(&buf);
                let scalar_res = find_annex_b_start_code_scalar(&buf);
                assert_eq!(
                    simd_res, scalar_res,
                    "mismatch at offset={offset}, four_byte={four_byte}"
                );
                assert_eq!(simd_res, Some((offset, if four_byte { 4 } else { 3 })));
            }
        }
    }

    #[test]
    fn split_annex_b_nals_extracts_exact_slices() {
        let mut stream = Vec::new();
        stream.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1E]);
        stream.extend_from_slice(&[0x00, 0x00, 0x01, 0x68, 0xCE, 0x3C, 0x80]);
        stream.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x21]);

        let nals = split_annex_b_nals(&stream);
        assert_eq!(nals.len(), 3);
        assert_eq!(nals[0], &[0x67, 0x42, 0x00, 0x1E]);
        assert_eq!(nals[1], &[0x68, 0xCE, 0x3C, 0x80]);
        assert_eq!(nals[2], &[0x65, 0x88, 0x84, 0x21]);
    }

    #[test]
    fn mpeg_ts_sync_scanner_neon_vs_scalar() {
        for sync_offset in [0usize, 1, 15, 16, 17, 63, 64, 65, 113] {
            let mut ts = vec![0xAAu8; sync_offset + 188 * 2];
            // Decoy 0x47 without a matching 0x47 at +188
            if sync_offset >= 10 {
                ts[sync_offset - 5] = 0x47;
            }
            ts[sync_offset] = 0x47;
            ts[sync_offset + 188] = 0x47;
            assert_eq!(
                find_mpeg_ts_sync(&ts),
                find_mpeg_ts_sync_scalar(&ts),
                "TS sync mismatch at {sync_offset}"
            );
            assert_eq!(find_mpeg_ts_sync(&ts), Some(sync_offset));
        }
    }

    #[test]
    fn sad_and_sse_neon_vs_scalar_across_odd_dimensions() {
        for &(w, h) in &[
            (1usize, 1usize),
            (7, 5),
            (15, 9),
            (16, 16),
            (33, 17),
            (127, 31),
        ] {
            let stride = (w + 63) & !63;
            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            for y in 0..h {
                for x in 0..w {
                    a[y * stride + x] = ((x * 31 + y * 17 + 7) & 0xFF) as u8;
                    b[y * stride + x] = ((x * 13 + y * 53 + 101) & 0xFF) as u8;
                }
            }
            let sad_simd = sad_plane_u8(&a, stride, &b, stride, w, h);
            let sad_ref = sad_plane_u8_scalar(&a, stride, &b, stride, w, h);
            assert_eq!(sad_simd, sad_ref, "SAD mismatch for {w}x{h}");

            let sse_simd = sse_plane_u8(&a, stride, &b, stride, w, h);
            let sse_ref = sse_plane_u8_scalar(&a, stride, &b, stride, w, h);
            assert_eq!(sse_simd, sse_ref, "SSE mismatch for {w}x{h}");
        }
    }

    #[test]
    fn ssim_8x8_window_neon_vs_scalar() {
        let stride = 64;
        let mut a = vec![0u8; stride * 8];
        let mut b = vec![0u8; stride * 8];
        for y in 0..8 {
            for x in 0..8 {
                a[y * stride + x] = (x * 23 + y * 19 + 12) as u8;
                b[y * stride + x] = (x * 21 + y * 17 + 15) as u8;
            }
        }
        let s_simd = ssim_window_8x8_u8(&a, stride, &b, stride);
        let s_ref = ssim_window_8x8_u8_scalar(&a, stride, &b, stride);
        assert_eq!(s_simd, s_ref);
    }

    #[test]
    fn q14_convolve_8tap_neon_vs_scalar() {
        let samples = [18u8, 42, 120, 200, 195, 110, 50, 20];
        let weights: [i16; 8] = [-256, 768, -2048, 9728, 9728, -2048, 768, -256];
        assert_eq!(weights.iter().map(|&w| w as i32).sum::<i32>(), Q14_ONE);
        assert_eq!(
            convolve_8tap_q14_u8(&samples, &weights),
            convolve_8tap_q14_u8_scalar(&samples, &weights)
        );
    }

    #[test]
    fn avcc_annex_b_roundtrip() {
        let annex_b = [
            0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1E, 0x00, 0x00, 0x01, 0x68, 0xCE, 0x3C,
            0x80,
        ];
        let mut avcc = Vec::new();
        annex_b_to_avcc(&annex_b, &mut avcc);
        let mut back = Vec::new();
        assert!(avcc_to_annex_b(&avcc, 4, &mut back));
        assert_eq!(
            split_annex_b_nals(&back),
            vec![&[0x67, 0x42, 0x00, 0x1E][..], &[0x68, 0xCE, 0x3C, 0x80][..]]
        );
    }

    #[test]
    fn ssim_4x4xn_and_full_plane_neon_vs_scalar() {
        let width = 36; // 9 4x4 blocks (even + odd tail)
        let height = 24;
        let stride = 64;
        let mut a = vec![0u8; stride * height];
        let mut b = vec![0u8; stride * height];
        for y in 0..height {
            for x in 0..width {
                a[y * stride + x] = ((x * 19 + y * 31 + 17) & 0xFF) as u8;
                b[y * stride + x] = ((x * 19 + y * 31 + 21) & 0xFF) as u8;
            }
        }

        let mut sums_simd = vec![[0i32; 4]; width / 4];
        let mut sums_ref = vec![[0i32; 4]; width / 4];
        ssim_4x4xn_u8(&a, stride, &b, stride, &mut sums_simd);
        ssim_4x4xn_u8_scalar(&a, stride, &b, stride, &mut sums_ref);
        assert_eq!(sums_simd, sums_ref);

        // Identical plane must have SSIM == 1.0
        let ssim_ident = ssim_plane_8bit(&a, stride, &a, stride, width, height);
        assert!((ssim_ident - 1.0).abs() < 1e-6);
        assert!(ssim_to_db(ssim_ident).is_infinite());

        let ssim_ab = ssim_plane_8bit(&a, stride, &b, stride, width, height);
        assert!(ssim_ab > 0.90 && ssim_ab < 1.0);
    }

    #[test]
    fn satd_4x4_and_8x8_neon_vs_scalar() {
        let stride = 64;
        let mut a = vec![0u8; stride * 8];
        let mut b = vec![0u8; stride * 8];
        for y in 0..8 {
            for x in 0..8 {
                a[y * stride + x] = ((x * 41 + y * 29 + 11) & 0xFF) as u8;
                b[y * stride + x] = ((x * 17 + y * 53 + 90) & 0xFF) as u8;
            }
        }
        assert_eq!(
            satd_4x4_u8(&a, stride, &b, stride),
            satd_4x4_u8_scalar(&a, stride, &b, stride)
        );
        assert_eq!(satd_8x8_u8(&a, stride, &a, stride), 0);
    }

    #[test]
    fn dct4x4_idct4x4_hadamard4x4_neon_vs_scalar_and_8x8() {
        let mut block = [0i16; 16];
        for (i, v) in block.iter_mut().enumerate() {
            *v = ((i as i16) * 23 - 140) ^ ((i as i16) * 7);
        }
        let dct_simd = forward_dct4x4(&block);
        let dct_ref = forward_dct4x4_scalar(&block);
        assert_eq!(dct_simd, dct_ref);

        let idct_simd = inverse_idct4x4(&dct_simd);
        let idct_ref = inverse_idct4x4_scalar(&dct_ref);
        assert_eq!(idct_simd, idct_ref);

        let had_simd = hadamard4x4(&block);
        let had_ref = hadamard4x4_scalar(&block);
        assert_eq!(had_simd, had_ref);

        // 8x8 DCT -> IDCT energy preservation
        let mut block8 = [0i16; 64];
        for (i, v) in block8.iter_mut().enumerate() {
            *v = ((i as i16) % 9 - 4) * 12;
        }
        let coeff8 = forward_dct8x8(&block8);
        let recon8 = inverse_idct8x8(&coeff8);
        assert_eq!(recon8.len(), 64);

        // A lone coefficient exercises the odd half of both passes (ITU-T H.264 8.5.13.2,
        // `f5 = (e3 >> 2) - e5`); the expected block matches FFmpeg's `h264_idct8_add`.
        let mut lone = [0i16; 64];
        lone[10] = 76;
        #[rustfmt::skip]
        let expected: [i16; 64] = [
             2,  1, -1, -2, -2, -1,  1,  2,
             1,  1, -1, -1, -1, -1,  1,  1,
             1,  0,  0, -1, -1,  0,  0,  1,
             0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,
            -1,  0,  0,  1,  1,  0,  0, -1,
            -1, -1,  1,  1,  1,  1, -1, -1,
            -2, -1,  1,  2,  2,  1, -1, -2,
        ];
        assert_eq!(inverse_idct8x8(&lone), expected);
    }

    #[test]
    fn sad_16x16_256_64_and_flac_fixed_l1_neon_vs_scalar() {
        let mut a256 = [0u8; 256];
        let mut b256 = [0u8; 256];
        for i in 0..256 {
            a256[i] = ((i * 73 + 19) & 0xFF) as u8;
            b256[i] = ((i * 29 + 141) & 0xFF) as u8;
        }
        assert_eq!(
            sad_block_256_u8(&a256, &b256),
            sad_block_256_u8_scalar(&a256, &b256)
        );
        assert_eq!(
            sad_16x16_u8(&a256, 16, &b256, 16),
            sad_16x16_u8_scalar(&a256, 16, &b256, 16)
        );

        let mut a64 = [0u8; 64];
        let mut b64 = [0u8; 64];
        a64.copy_from_slice(&a256[..64]);
        b64.copy_from_slice(&b256[..64]);
        assert_eq!(
            sad_block_64_u8(&a64, &b64),
            sad_block_64_u8_scalar(&a64, &b64)
        );

        for len in [1usize, 3, 4, 7, 8, 15, 64, 513, 4096] {
            let samples: Vec<i32> = (0..len)
                .map(|i| (((i * 1103515245 + 12345) as i32) >> 10) + ((i as i32) * 17))
                .collect();
            assert_eq!(
                flac_eval_fixed_orders_l1(&samples),
                flac_eval_fixed_orders_l1_scalar(&samples),
                "FLAC fixed L1 mismatch at len={len}"
            );
        }
    }

    #[test]
    fn h264_qpel16_chroma_mc8_deblock_and_intra4x4_neon_vs_scalar() {
        let width = 32usize;
        let height = 32usize;
        let stride = 32usize;
        let mut plane = vec![0u8; stride * height];
        for (i, b) in plane.iter_mut().enumerate() {
            *b = ((i * 97 + (i >> 3) * 53 + 19) & 0xFF) as u8;
        }

        // 1. All 16 quarter-pel phases `(mx, my) in 0..4 x 0..4` (interior and border clamped)
        for &(x0, y0) in &[(6isize, 6isize), (-1, 0), (20, 20)] {
            for my in 0..4u8 {
                for mx in 0..4u8 {
                    let mut simd_dst = [0u8; 256];
                    let mut scalar_dst = [0u8; 256];
                    put_h264_qpel16_mc_u8(
                        &plane,
                        stride,
                        width,
                        height,
                        x0,
                        y0,
                        mx,
                        my,
                        &mut simd_dst,
                    );
                    put_h264_qpel16_mc_u8_scalar(
                        &plane,
                        stride,
                        width,
                        height,
                        x0,
                        y0,
                        mx,
                        my,
                        &mut scalar_dst,
                    );
                    assert_eq!(
                        simd_dst, scalar_dst,
                        "qpel16 mismatch at ({x0},{y0}) phase ({mx},{my})"
                    );
                }
            }
        }

        // 2. All 64 1/8-pel chroma phases `(fx, fy) in 0..8 x 0..8`
        for fy in 0..8u8 {
            for fx in 0..8u8 {
                let mut simd_c = [0u8; 64];
                let mut scalar_c = [0u8; 64];
                put_h264_chroma_mc8_u8(&plane, stride, width, height, 4, 4, fx, fy, &mut simd_c);
                put_h264_chroma_mc8_u8_scalar(
                    &plane,
                    stride,
                    width,
                    height,
                    4,
                    4,
                    fx,
                    fy,
                    &mut scalar_c,
                );
                assert_eq!(simd_c, scalar_c, "chroma mc8 mismatch at ({fx},{fy})");
            }
        }

        // 3. All 9 Intra 4x4 spatial prediction modes (`0..=8`)
        let top8 = [20u8, 45, 80, 110, 140, 175, 200, 230];
        let left4 = [30u8, 70, 120, 185];
        for mode in 0..=8u8 {
            let mut simd_4x4 = [0u8; 16];
            let mut scalar_4x4 = [0u8; 16];
            predict_intra_4x4_u8(mode, Some(&top8), Some(&left4), 15, &mut simd_4x4, 4);
            predict_intra_4x4_u8_scalar(mode, Some(&top8), Some(&left4), 15, &mut scalar_4x4, 4);
            assert_eq!(simd_4x4, scalar_4x4, "intra 4x4 mismatch at mode {mode}");
        }

        // 4. Adaptive In-Loop Deblocking Filter across bS = 1..=4
        for bs in 1..=4u8 {
            let mut line = [100u8, 102, 105, 108, 116, 119, 121, 123];
            let orig_gap = (line[4] as i32 - line[3] as i32).abs();
            h264_deblock_luma_line(&mut line, 28, bs);
            let new_gap = (line[4] as i32 - line[3] as i32).abs();
            assert!(
                new_gap <= orig_gap,
                "deblocking bS={bs} must reduce or preserve step gap ({new_gap} vs {orig_gap})"
            );
        }
    }

    #[test]
    fn audio_dsp_polyphase_rematrix_biquad_true_peak_and_mdct_neon_vs_scalar() {
        // 1. SIMD-0042: Polyphase f32 (within 1e-6) and Q15 i16 (0 LSB bit-exact)
        let mut src_f32 = [0.0f32; 32];
        let mut taps16_f32 = [0.0f32; 16];
        let mut taps32_f32 = [0.0f32; 32];
        let mut src_i16 = [0i16; 32];
        let mut taps16_i16 = [0i16; 16];
        let mut taps32_i16 = [0i16; 32];
        for i in 0..32 {
            src_f32[i] = ((i as f32 * 0.37).sin()) * 0.9;
            taps32_f32[i] = ((i as f32 - 15.5) * 0.25).cos() * 0.0625;
            src_i16[i] = (src_f32[i] * 32000.0) as i16;
            taps32_i16[i] = (taps32_f32[i] * 32768.0) as i16;
            if i < 16 {
                taps16_f32[i] = taps32_f32[i];
                taps16_i16[i] = taps32_i16[i];
            }
        }
        let d16_simd = polyphase_dot16_f32(&src_f32[..16], &taps16_f32);
        let d16_scalar = polyphase_dot16_f32_scalar(&src_f32[..16], &taps16_f32);
        assert!(
            (d16_simd - d16_scalar).abs() < 1e-6,
            "polyphase_dot16_f32 diff: {} vs {}",
            d16_simd,
            d16_scalar
        );
        let d32_simd = polyphase_dot32_f32(&src_f32, &taps32_f32);
        let d32_scalar = polyphase_dot32_f32_scalar(&src_f32, &taps32_f32);
        assert!(
            (d32_simd - d32_scalar).abs() < 1e-6,
            "polyphase_dot32_f32 diff: {} vs {}",
            d32_simd,
            d32_scalar
        );
        assert_eq!(
            polyphase_dot16_q15_i16(&src_i16[..16], &taps16_i16),
            polyphase_dot16_q15_i16_scalar(&src_i16[..16], &taps16_i16),
            "Q15 16-tap must be 0 LSB bit-exact"
        );
        assert_eq!(
            polyphase_dot32_q15_i16(&src_i16, &taps32_i16),
            polyphase_dot32_q15_i16_scalar(&src_i16, &taps32_i16),
            "Q15 32-tap must be 0 LSB bit-exact"
        );

        // 2. SIMD-0042: Channel Matrix Mixing (Mono <-> Stereo, 5.1 -> Stereo, NxM)
        let src_5_1: Vec<f32> = (0..(23 * 6))
            .map(|i| ((i as f32) * 0.19).sin() * 0.8)
            .collect();
        let mut st_simd = vec![0.0f32; 23 * 2];
        let mut st_scalar = vec![0.0f32; 23 * 2];
        rematrix_5_1_to_stereo_f32(
            &src_5_1,
            &mut st_simd,
            ITU_MIX_MINUS_3DB,
            ITU_MIX_MINUS_3DB,
            0.0,
            true,
        );
        rematrix_5_1_to_stereo_f32_scalar(
            &src_5_1,
            &mut st_scalar,
            ITU_MIX_MINUS_3DB,
            ITU_MIX_MINUS_3DB,
            0.0,
            true,
        );
        for (a, b) in st_simd.iter().zip(st_scalar.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        // 3. SIMD-0043: Biquad 4ch & 4x True-Peak 12-tap bit-exactness
        let coeffs = BiquadCoeffs {
            b0: 1.5351248,
            b1: -2.6916962,
            b2: 1.1983928,
            a1: -1.6906593,
            a2: 0.73248076,
        };
        let mut z1_simd = [0.0f32; 4];
        let mut z2_simd = [0.0f32; 4];
        let mut z1_scalar = [0.0f32; 4];
        let mut z2_scalar = [0.0f32; 4];
        for step in 0..32 {
            let x = [
                ((step as f32) * 0.1).sin(),
                ((step as f32) * 0.2).cos(),
                ((step as f32) * 0.3).sin() * 0.5,
                -((step as f32) * 0.15).cos(),
            ];
            let mut y_simd = [0.0f32; 4];
            let mut y_scalar = [0.0f32; 4];
            biquad_process_4ch_f32(&coeffs, &x, &mut z1_simd, &mut z2_simd, &mut y_simd);
            biquad_process_4ch_f32_scalar(
                &coeffs,
                &x,
                &mut z1_scalar,
                &mut z2_scalar,
                &mut y_scalar,
            );
            for c in 0..4 {
                assert_eq!(
                    y_simd[c].to_bits(),
                    y_scalar[c].to_bits(),
                    "biquad 4ch bit-exact mismatch at step {step} ch {c}"
                );
            }
        }

        let mut hist = [0.0f32; 12];
        for (k, v) in hist.iter_mut().enumerate() {
            *v = ((k as f32 - 5.5) * std::f32::consts::FRAC_PI_2).cos();
        }
        let tp_simd = true_peak_4x_12tap_sample_f32(&hist);
        let tp_scalar = true_peak_4x_12tap_sample_f32_scalar(&hist);
        assert_eq!(
            tp_simd.to_bits(),
            tp_scalar.to_bits(),
            "true-peak 4x 12-tap must be 100% bit-exact"
        );
        // fs/4 sine wave at 45 deg phase has sample peaks of 0.7071, but true inter-sample peak ~ 1.0!
        assert!(
            tp_simd > 0.98,
            "expected 4x oversampled true-peak > 0.98 on fs/4 45-deg tone, got {tp_simd}"
        );

        // 4. 0044: MDCT / IMDCT + Sine/KBD TDAC perfect reconstruction across AAC (256, 2048) & Opus (120, 240, 480, 960)
        for &n in &[120usize, 240, 256, 480, 960, 2048] {
            let m = n / 2;
            let plan = MdctPlan::new(n);
            let win = if n == 256 || n == 2048 {
                kbd_window_f32(n, 4.0)
            } else {
                sine_window_f32(n)
            };
            let signal: Vec<f32> = (0..(3 * m))
                .map(|i| ((i as f32) * 0.073).sin() * 0.6 + ((i as f32) * 0.21).cos() * 0.25)
                .collect();
            let mut overlap = vec![0.0f32; m];
            let mut coeffs = vec![0.0f32; m];
            let mut imdct_out = vec![0.0f32; n];
            let mut recon_mid = vec![0.0f32; m];

            // Frame 0 (primes overlap)
            let w0: Vec<f32> = (0..n).map(|i| signal[i] * win[i]).collect();
            plan.mdct(&w0, &mut coeffs);
            plan.imdct(&coeffs, &mut imdct_out);
            window_overlap_add_f32(&imdct_out, &win, &mut overlap, &mut recon_mid);

            // Frame 1 (reconstructs signal[m..2*m] via 50% TDAC!)
            let w1: Vec<f32> = (0..n).map(|i| signal[m + i] * win[i]).collect();
            plan.mdct(&w1, &mut coeffs);
            plan.imdct(&coeffs, &mut imdct_out);
            window_overlap_add_f32(&imdct_out, &win, &mut overlap, &mut recon_mid);

            for i in 0..m {
                let err = (recon_mid[i] - signal[m + i]).abs();
                assert!(
                    err < 1e-4,
                    "MDCT/IMDCT TDAC N={n} reconstruction error at sample {i}: got {}, expected {} (err={err})",
                    recon_mid[i],
                    signal[m + i]
                );
            }
        }
    }
}
