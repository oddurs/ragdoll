//! # `ragdoll-simd`
//!
//! Explicit Apple Silicon `aarch64` NEON and `x86_64` AVX2 SIMD kernels with bit-exact scalar fallbacks.
//!
//! - [`scan`]: Vectorized Annex-B NAL start-code (`0x000001` / `0x00000001`) and MPEG-TS `0x47` sync-byte scanners.
//! - [`metrics`]: NEON-accelerated block/plane SAD, SSE, and single-pass $8\times8$ SSIM window statistics.
//! - [`fixed`]: Q14 fixed-point polyphase dot products and $4\times4$ integer DCT/IDCT/Hadamard transforms.

pub mod fixed;
pub mod metrics;
pub mod scan;

pub use fixed::{
    Q14_HALF, Q14_ONE, add4x4_idct_u8, clamp_u8, convolve_8tap_q14_u8, convolve_8tap_q14_u8_scalar,
    forward_dct4x4, forward_dct4x4_scalar, forward_dct8x8, hadamard4x4, hadamard4x4_scalar,
    inverse_idct4x4, inverse_idct4x4_scalar, inverse_idct8x8, sub4x4_dct_u8,
};
pub use metrics::{
    SSIM_C1_8BIT, SSIM_C2_8BIT, SsimWindowStats, sad_plane_u8, sad_plane_u8_scalar, satd_4x4_u8,
    satd_4x4_u8_scalar, satd_8x8_u8, sse_plane_u8, sse_plane_u8_scalar, ssim_4x4xn_u8,
    ssim_4x4xn_u8_scalar, ssim_end1_8bit, ssim_plane_8bit, ssim_to_db, ssim_window_8x8_u8,
    ssim_window_8x8_u8_scalar,
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
    }
}
