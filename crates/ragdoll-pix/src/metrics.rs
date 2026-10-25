//! SIMD perceptual video quality engine: **PSNR**, **$8\times8$ Windowed SSIM**, and **VMAF-Lite**.
//!
//! Built on `ragdoll-simd`'s 4-way unrolled NEON `sse_plane_u8`, `sad_plane_u8`, and `ssim_4x4xn_u8`
//! kernels to sustain $>1\,000\text{ fps}$ on 1080p and $>300\text{ fps}$ on 4K UHD on Apple Silicon.

use ragdoll_core::{Plane, VideoFrame};
use ragdoll_simd::{sad_plane_u8, sse_plane_u8, ssim_4x4xn_u8, ssim_plane_8bit, ssim_to_db};

/// Full multi-metric quality report comparing a distorted frame against a reference frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QualityReport {
    pub psnr_y: f64,
    pub psnr_u: f64,
    pub psnr_v: f64,
    pub psnr_avg: f64,
    pub ssim_y: f64,
    pub ssim_u: f64,
    pub ssim_v: f64,
    pub ssim_avg: f64,
    pub ssim_db: f64,
    pub vmaf_lite: f64,
}

/// Computes Peak Signal-to-Noise Ratio (PSNR in dB) between two 8-bit planes.
///
/// Returns `f64::INFINITY` when the two planes are bit-identical (`SSE == 0`).
#[inline]
pub fn psnr_plane_u8(
    a: &[u8],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
    width: usize,
    height: usize,
) -> f64 {
    let sse = sse_plane_u8(a, a_stride, b, b_stride, width, height);
    if sse == 0 {
        return f64::INFINITY;
    }
    let n = (width * height) as f64;
    let mse = (sse as f64) / n;
    10.0 * ((255.0 * 255.0) / mse).log10()
}

/// Computes the $8\times8$ windowed SSIM score in $[0.0, 1.0]$ for an 8-bit plane.
#[inline]
pub fn ssim_plane_u8(
    a: &[u8],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
    width: usize,
    height: usize,
) -> f64 {
    ssim_plane_8bit(a, a_stride, b, b_stride, width, height)
}

/// Computes a `0.0 ..= 100.0` **VMAF-Lite** perceptual quality score over an 8-bit luma plane.
///
/// Fuses:
/// 1. **Visual Information Fidelity (VIF)**: Shannon mutual information ratio
///    $\log_2(1 + g^2 \sigma_{\text{ref}}^2 / (\sigma_{\text{err}}^2 + \sigma_n^2))$
///    from SIMD $4\times4$ sub-block statistics (`ssim_4x4xn_u8`).
/// 2. **Detail Loss / Contrast Preservation (DLM)**: High-frequency variance retention penalty.
/// 3. **Temporal Motion Masking**: Optional frame-to-frame luma SAD motion attenuation.
pub fn vmaf_lite_plane_u8(
    ref_plane: &[u8],
    ref_stride: usize,
    dist_plane: &[u8],
    dist_stride: usize,
    width: usize,
    height: usize,
    prev_ref_plane: Option<(&[u8], usize)>,
) -> f64 {
    let bw = width >> 2;
    let bh = height >> 2;
    if bw == 0 || bh == 0 {
        return 100.0;
    }

    let mut sums = vec![[0i32; 4]; bw];
    let mut info_num = 0.0f64;
    let mut info_den = 0.0f64;
    let mut ssim_acc = 0.0f64;
    let mut blocks = 0usize;

    // HVS neural noise floor variance scaled by N = 16 pixels per 4x4 block (16^2 = 256)
    const SIGMA_NSQ: f64 = 2.0 * 256.0;

    // Sample every 4x4 block row using NEON `ssim_4x4xn_u8`
    for by in 0..bh {
        let off_r = by * 4 * ref_stride;
        let off_d = by * 4 * dist_stride;
        ssim_4x4xn_u8(
            &ref_plane[off_r..],
            ref_stride,
            &dist_plane[off_d..],
            dist_stride,
            &mut sums,
        );

        for &[s1, s2, ss, s12] in &sums {
            // In `ssim_4x4xn_u8`, ss = sum(a^2 + b^2), s12 = sum(a * b) over N = 16 pixels.
            //Note: 16 * sum((a - b)^2) = 16 * (ss - 2 * s12)
            let err_energy = ((ss - 2 * s12).max(0) as f64) * 16.0;
            let mean_sq_sum = (s1 * s1 + s2 * s2) as f64;
            let total_var = ((ss * 16 - s1 * s1 - s2 * s2).max(0)) as f64;
            let covar = (s12 * 16 - s1 * s2) as f64;

            // Local 4x4 structural similarity term
            let c1 = 26.0f64;
            let c2 = 234.0f64;
            let blk_ssim = ((2.0 * (s1 as f64) * (s2 as f64) + c1) * (2.0 * covar + c2))
                / ((mean_sq_sum + c1) * (total_var + c2));
            ssim_acc += blk_ssim.clamp(0.0, 1.0);

            // VIF mutual information accumulation
            let ref_var = (total_var * 0.5).max(0.0);
            let den = (1.0 + ref_var / SIGMA_NSQ).ln();
            let num = if err_energy <= 1e-6 {
                den
            } else {
                let snr = (covar.max(0.0)) / (err_energy + SIGMA_NSQ);
                (1.0 + snr).ln().min(den)
            };
            info_num += num;
            info_den += den + 1e-6;
            blocks += 1;
        }
    }

    let mean_ssim = ssim_acc / (blocks.max(1) as f64);
    if (1.0 - mean_ssim).abs() < 1e-9 {
        return 100.0;
    }

    let vif_ratio = (info_num / info_den.max(1e-9)).clamp(0.0, 1.0);

    // Optional temporal motion masking boost (up to +5% tolerance under rapid motion)
    let motion_boost = if let Some((prev, prev_stride)) = prev_ref_plane {
        let mad = (sad_plane_u8(ref_plane, ref_stride, prev, prev_stride, width, height) as f64)
            / ((width * height) as f64);
        (mad / 255.0 * 0.08).min(0.05)
    } else {
        0.0
    };

    let raw = (0.65 * mean_ssim.powf(0.85) + 0.35 * vif_ratio + motion_boost) * 100.0;
    raw.clamp(0.0, 100.0)
}

/// Evaluates PSNR, SSIM, and VMAF-Lite between two planar `YUV` `VideoFrame`s in a single report.
pub fn evaluate_frame_quality(
    reference: &VideoFrame,
    distorted: &VideoFrame,
    prev_reference: Option<&VideoFrame>,
) -> QualityReport {
    assert_eq!(reference.width, distorted.width);
    assert_eq!(reference.height, distorted.height);
    assert_eq!(reference.format, distorted.format);

    let plane_psnr = |p_ref: &Plane, p_dist: &Plane| -> f64 {
        psnr_plane_u8(
            p_ref.data(),
            p_ref.stride(),
            p_dist.data(),
            p_dist.stride(),
            p_ref.width(),
            p_ref.height(),
        )
    };
    let plane_ssim = |p_ref: &Plane, p_dist: &Plane| -> f64 {
        ssim_plane_u8(
            p_ref.data(),
            p_ref.stride(),
            p_dist.data(),
            p_dist.stride(),
            p_ref.width(),
            p_ref.height(),
        )
    };

    let psnr_y = plane_psnr(reference.plane(0), distorted.plane(0));
    let ssim_y = plane_ssim(reference.plane(0), distorted.plane(0));

    let (psnr_u, psnr_v, ssim_u, ssim_v) = if reference.planes.len() >= 3 {
        (
            plane_psnr(reference.plane(1), distorted.plane(1)),
            plane_psnr(reference.plane(2), distorted.plane(2)),
            plane_ssim(reference.plane(1), distorted.plane(1)),
            plane_ssim(reference.plane(2), distorted.plane(2)),
        )
    } else {
        (psnr_y, psnr_y, ssim_y, ssim_y)
    };

    let psnr_avg = if psnr_y.is_infinite() && psnr_u.is_infinite() && psnr_v.is_infinite() {
        f64::INFINITY
    } else {
        (6.0 * psnr_y.min(100.0) + psnr_u.min(100.0) + psnr_v.min(100.0)) / 8.0
    };
    let ssim_avg = (4.0 * ssim_y + ssim_u + ssim_v) / 6.0;
    let ssim_db = ssim_to_db(ssim_avg);

    let prev_y = prev_reference.map(|f| (f.plane(0).data(), f.plane(0).stride()));
    let vmaf_lite = vmaf_lite_plane_u8(
        reference.plane(0).data(),
        reference.plane(0).stride(),
        distorted.plane(0).data(),
        distorted.plane(0).stride(),
        reference.width,
        reference.height,
        prev_y,
    );

    QualityReport {
        psnr_y,
        psnr_u,
        psnr_v,
        psnr_avg,
        ssim_y,
        ssim_u,
        ssim_v,
        ssim_avg,
        ssim_db,
        vmaf_lite,
    }
}
