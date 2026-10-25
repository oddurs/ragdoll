//! # `ragdoll-pix`
//!
//! Fused L1-cache-resident scanline pixel engine and perceptual quality analyzer:
//! - [`color`]: SIMD `YUV420p` $\leftrightarrow$ `RGB24`/`RGBA8`/`Oklab` conversion across BT.601, BT.709, and BT.2020.
//! - [`scale`]: Separable 8-tap Q14 polyphase `Lanczos3`, `MitchellNetravali`, and `Bilinear` resampler with an 8-row L1 rolling scanline ring buffer.
//! - [`metrics`]: SIMD-accelerated `PSNR`, $8\times8$ windowed `SSIM`, and `VMAF-Lite` perceptual quality analyzer.

pub mod color;
pub mod metrics;
pub mod scale;

pub use color::{
    OklabColor, YuvRgbMatrix, convert_frame, convert_frame_into, oklab_row_to_rgb24,
    rgb24_row_to_oklab, rgb24_to_yuv420p, rgb24_to_yuv420p_scalar, upsample_chroma_420_to_444,
    yuv420p_to_rgb24, yuv420p_to_rgb24_scalar, yuv420p_to_rgba8,
};
pub use metrics::{
    QualityReport, evaluate_frame_quality, psnr_plane_u8, ssim_plane_u8, vmaf_lite_plane_u8,
};
pub use scale::{
    PolyphaseFilterBank, ScaleFilter, hscale_row_8tap_q14, hscale_row_8tap_q14_scalar, scale_frame,
    scale_frame_into, scale_plane, scale_plane_with_banks, vscale_row_8tap_q14,
    vscale_row_8tap_q14_scalar,
};

#[cfg(test)]
mod tests {
    use super::*;
    use ragdoll_core::{
        ChromaLocation, ColorRange, Colorimetry, MatrixCoefficients, PixelFormat, Plane, VideoFrame,
    };
    use ragdoll_simd::Q14_ONE;

    #[test]
    fn yuv420p_rgb24_neon_vs_scalar_differential_across_odd_dimensions() {
        for &(w, h) in &[
            (2usize, 2usize),
            (14, 6),
            (16, 16),
            (19, 11),
            (33, 17),
            (65, 35),
        ] {
            for matrix in [
                MatrixCoefficients::BT601,
                MatrixCoefficients::BT709,
                MatrixCoefficients::BT2020NCL,
            ] {
                for range in [ColorRange::Limited, ColorRange::Full] {
                    let mat = YuvRgbMatrix::from_colorimetry(matrix, range);
                    let cw = w.div_ceil(2);
                    let ch = h.div_ceil(2);
                    let y_stride = (w + 63) & !63;
                    let c_stride = (cw + 63) & !63;
                    let rgb_stride = (w * 3 + 63) & !63;

                    let mut y_plane = vec![0u8; y_stride * h];
                    let mut u_plane = vec![128u8; c_stride * ch];
                    let mut v_plane = vec![128u8; c_stride * ch];

                    for y in 0..h {
                        for x in 0..w {
                            y_plane[y * y_stride + x] = ((x * 29 + y * 43 + 16) % 220 + 16) as u8;
                        }
                    }
                    for cy in 0..ch {
                        for cx in 0..cw {
                            u_plane[cy * c_stride + cx] =
                                ((cx * 37 + cy * 19 + 40) % 200 + 20) as u8;
                            v_plane[cy * c_stride + cx] =
                                ((cx * 53 + cy * 23 + 30) % 200 + 20) as u8;
                        }
                    }

                    let mut rgb_simd = vec![0u8; rgb_stride * h];
                    let mut rgb_ref = vec![0u8; rgb_stride * h];

                    yuv420p_to_rgb24(
                        &y_plane,
                        y_stride,
                        &u_plane,
                        c_stride,
                        &v_plane,
                        c_stride,
                        &mut rgb_simd,
                        rgb_stride,
                        w,
                        h,
                        &mat,
                    );
                    yuv420p_to_rgb24_scalar(
                        &y_plane,
                        y_stride,
                        &u_plane,
                        c_stride,
                        &v_plane,
                        c_stride,
                        &mut rgb_ref,
                        rgb_stride,
                        w,
                        h,
                        &mat,
                    );
                    assert_eq!(rgb_simd, rgb_ref, "YUV->RGB mismatch at {w}x{h} {matrix:?}");

                    let mut y_out_simd = vec![0u8; y_stride * h];
                    let mut u_out_simd = vec![0u8; c_stride * ch];
                    let mut v_out_simd = vec![0u8; c_stride * ch];
                    let mut y_out_ref = vec![0u8; y_stride * h];
                    let mut u_out_ref = vec![0u8; c_stride * ch];
                    let mut v_out_ref = vec![0u8; c_stride * ch];

                    rgb24_to_yuv420p(
                        &rgb_simd,
                        rgb_stride,
                        &mut y_out_simd,
                        y_stride,
                        &mut u_out_simd,
                        c_stride,
                        &mut v_out_simd,
                        c_stride,
                        w,
                        h,
                        &mat,
                    );
                    rgb24_to_yuv420p_scalar(
                        &rgb_ref,
                        rgb_stride,
                        &mut y_out_ref,
                        y_stride,
                        &mut u_out_ref,
                        c_stride,
                        &mut v_out_ref,
                        c_stride,
                        w,
                        h,
                        &mat,
                    );
                    assert_eq!(y_out_simd, y_out_ref, "RGB->Y mismatch at {w}x{h}");
                    assert_eq!(u_out_simd, u_out_ref, "RGB->U mismatch at {w}x{h}");
                    assert_eq!(v_out_simd, v_out_ref, "RGB->V mismatch at {w}x{h}");
                }
            }
        }
    }

    #[test]
    fn yuv_rgb_roundtrip_accuracy_within_1_lsb_on_neutral_and_primary_colors() {
        for matrix in [
            MatrixCoefficients::BT601,
            MatrixCoefficients::BT709,
            MatrixCoefficients::BT2020NCL,
        ] {
            let mat = YuvRgbMatrix::from_colorimetry(matrix, ColorRange::Full);
            for gray in [0u8, 16, 64, 128, 180, 235, 255] {
                let (y, u, v) = mat.rgb_to_yuv_pixel(gray, gray, gray);
                assert_eq!(u, 128, "neutral gray must have U=128");
                assert_eq!(v, 128, "neutral gray must have V=128");
                let (r, g, b) = mat.yuv_to_rgb_pixel(y, u, v);
                assert!(r.abs_diff(gray) <= 1);
                assert!(g.abs_diff(gray) <= 1);
                assert!(b.abs_diff(gray) <= 1);
            }
        }
    }

    #[test]
    fn oklab_scanline_and_chroma_siting_roundtrip() {
        let rgb = [240u8, 120, 45, 30, 180, 210, 128, 128, 128];
        let mut oklab = [OklabColor::default(); 3];
        let mut back = [0u8; 9];
        rgb24_row_to_oklab(&rgb, &mut oklab);
        oklab_row_to_rgb24(&oklab, &mut back);
        for (a, b) in rgb.iter().zip(back.iter()) {
            assert!(a.abs_diff(*b) <= 1);
        }

        let src_c = [64u8, 128, 192, 220];
        let mut dst_444 = [0u8; 16];
        upsample_chroma_420_to_444(&src_c, 2, &mut dst_444, 4, 4, 4, ChromaLocation::TopLeft);
        assert_eq!(dst_444[0], 64);
        assert_eq!(dst_444[1], 96);
    }

    #[test]
    fn polyphase_lanczos3_hscale_and_vscale_neon_vs_scalar() {
        for filter in [
            ScaleFilter::Bilinear,
            ScaleFilter::MitchellNetravali,
            ScaleFilter::Lanczos3,
        ] {
            for &(src_w, dst_w) in &[(64usize, 32usize), (33, 19), (19, 37), (128, 85)] {
                let bank = PolyphaseFilterBank::new(src_w, dst_w, filter);
                for w in &bank.weights {
                    assert_eq!(w.iter().map(|&t| t as i32).sum::<i32>(), Q14_ONE);
                }

                let mut src_row = vec![0u8; (src_w + 63) & !63];
                for (i, p) in src_row[..src_w].iter_mut().enumerate() {
                    *p = ((i * 41 + 19) & 0xFF) as u8;
                }
                let mut dst_simd = vec![0u8; dst_w];
                let mut dst_ref = vec![0u8; dst_w];
                hscale_row_8tap_q14(&src_row, &mut dst_simd, &bank);
                hscale_row_8tap_q14_scalar(&src_row, &mut dst_ref, &bank);
                assert_eq!(
                    dst_simd, dst_ref,
                    "hscale mismatch {src_w}->{dst_w} {filter:?}"
                );
            }
        }
    }

    #[test]
    fn lanczos3_4k_to_1080p_zero_dc_drift_and_odd_plane_scaling() {
        // Verify 4K (3840x2160) -> 1080p (1920x1080) flat luma field preserves exact DC level (142)
        let mut src_4k = Plane::new(3840, 2160, 1);
        src_4k.data_mut().fill(142);
        let mut dst_1080p = Plane::new(1920, 1080, 1);
        scale_plane(&src_4k, &mut dst_1080p, ScaleFilter::Lanczos3);

        for y in 0..1080 {
            let row = dst_1080p.row(y);
            assert!(
                row.iter().all(|&p| p == 142),
                "DC ripple detected at row {y}"
            );
        }
    }

    #[test]
    fn psnr_ssim_and_vmaf_lite_exact_identity_and_analytical_mse() {
        let mut ref_frame = VideoFrame::new(64, 64, PixelFormat::Yuv420p);
        ref_frame.colorimetry = Colorimetry::bt709_limited();
        for p in 0..3 {
            let plane = ref_frame.plane_mut(p);
            let h = plane.height();
            for y in 0..h {
                let row = plane.row_mut(y);
                for (x, px) in row.iter_mut().enumerate() {
                    *px = ((x * 17 + y * 29 + 30) % 180 + 20) as u8;
                }
            }
        }

        // 1. Identical frames -> PSNR = inf, SSIM = 1.0, VMAF-Lite = 100.0
        let ident = evaluate_frame_quality(&ref_frame, &ref_frame, None);
        assert!(ident.psnr_y.is_infinite());
        assert!(ident.psnr_avg.is_infinite());
        assert!((ident.ssim_y - 1.0).abs() < 1e-6);
        assert!((ident.vmaf_lite - 100.0).abs() < 1e-6);

        // 2. Uniform +1 LSB distortion on luma -> MSE = 1.0 -> PSNR = 10 * log10(255^2 / 1) = 48.1308036 dB
        let mut dist_frame = ref_frame.clone();
        for y in 0..64 {
            for p in dist_frame.plane_mut(0).row_mut(y).iter_mut() {
                *p += 1;
            }
        }
        let rep = evaluate_frame_quality(&ref_frame, &dist_frame, None);
        let expected_psnr = 10.0 * (255.0f64 * 255.0).log10();
        assert!((rep.psnr_y - expected_psnr).abs() < 1e-6);
        assert!(rep.ssim_y > 0.99 && rep.ssim_y <= 1.0);
        assert!(rep.vmaf_lite > 90.0 && rep.vmaf_lite < 100.0);
    }
}
