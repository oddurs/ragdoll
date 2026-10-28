//! SIMD-accelerated audio DSP kernels for Milestone 8 (`m8-audio-pro`):
//! - **`SIMD-0042`**: Windowed-Sinc Kaiser (`bessel_i0`) polyphase FIR dot products (`f32` and `Q15` `i16`)
//!   and ITU-R BS.775 multichannel matrix mixing (`Mono <-> Stereo <-> 5.1 <-> 7.1`).
//! - **`SIMD-0043`**: ITU-R BS.1770-4 / EBU R128 K-weighting biquad cascade (`f32x4`) and
//!   4x oversampled 12-tap polyphase True-Peak (`dBTP`) detector and limiter.
//! - **`0044`**: Shared SIMD Modified Discrete Cosine Transform (`MDCT` / `IMDCT`) and Sine / KBD
//!   windowing + overlap-add synthesis for AAC-LC (`N = 256, 2048`) and Opus CELT (`N = 120, 240, 480, 960`).

use std::f64::consts::PI;

/// Equal-power (`-3.0103 dB`, $1/\sqrt{2}$) ITU-R BS.775 center/surround mix coefficient.
pub const ITU_MIX_MINUS_3DB: f32 = std::f32::consts::FRAC_1_SQRT_2;

// ============================================================================
// SIMD-0042: Polyphase FIR Convolution Kernels (f32 & Q15 i16)
// ============================================================================

/// Scalar reference: 16-tap `f32` polyphase dot product.
#[inline]
pub fn polyphase_dot16_f32_scalar(src: &[f32], taps: &[f32; 16]) -> f32 {
    debug_assert!(src.len() >= 16);
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    for k in (0..16).step_by(4) {
        s0 += src[k] * taps[k];
        s1 += src[k + 1] * taps[k + 1];
        s2 += src[k + 2] * taps[k + 2];
        s3 += src[k + 3] * taps[k + 3];
    }
    (s0 + s1) + (s2 + s3)
}

/// Vectorized 16-tap `f32` polyphase dot product using `aarch64` NEON (`vfmaq_f32` + `vaddvq_f32`).
#[inline]
pub fn polyphase_dot16_f32(src: &[f32], taps: &[f32; 16]) -> f32 {
    debug_assert!(src.len() >= 16);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        return unsafe {
            use std::arch::aarch64::*;
            let sp = src.as_ptr();
            let tp = taps.as_ptr();
            let mut acc0 = vmulq_f32(vld1q_f32(sp), vld1q_f32(tp));
            let mut acc1 = vmulq_f32(vld1q_f32(sp.add(4)), vld1q_f32(tp.add(4)));
            acc0 = vfmaq_f32(acc0, vld1q_f32(sp.add(8)), vld1q_f32(tp.add(8)));
            acc1 = vfmaq_f32(acc1, vld1q_f32(sp.add(12)), vld1q_f32(tp.add(12)));
            vaddvq_f32(vaddq_f32(acc0, acc1))
        };
    }
    polyphase_dot16_f32_scalar(src, taps)
}

/// Scalar reference: 32-tap `f32` polyphase dot product.
#[inline]
pub fn polyphase_dot32_f32_scalar(src: &[f32], taps: &[f32; 32]) -> f32 {
    debug_assert!(src.len() >= 32);
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    for k in (0..32).step_by(4) {
        s0 += src[k] * taps[k];
        s1 += src[k + 1] * taps[k + 1];
        s2 += src[k + 2] * taps[k + 2];
        s3 += src[k + 3] * taps[k + 3];
    }
    (s0 + s1) + (s2 + s3)
}

/// Vectorized 32-tap `f32` polyphase dot product using `aarch64` NEON (`vfmaq_f32` + `vaddvq_f32`).
#[inline]
pub fn polyphase_dot32_f32(src: &[f32], taps: &[f32; 32]) -> f32 {
    debug_assert!(src.len() >= 32);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        return unsafe {
            use std::arch::aarch64::*;
            let sp = src.as_ptr();
            let tp = taps.as_ptr();
            let mut acc0 = vmulq_f32(vld1q_f32(sp), vld1q_f32(tp));
            let mut acc1 = vmulq_f32(vld1q_f32(sp.add(4)), vld1q_f32(tp.add(4)));
            for k in (8..32).step_by(8) {
                acc0 = vfmaq_f32(acc0, vld1q_f32(sp.add(k)), vld1q_f32(tp.add(k)));
                acc1 = vfmaq_f32(acc1, vld1q_f32(sp.add(k + 4)), vld1q_f32(tp.add(k + 4)));
            }
            vaddvq_f32(vaddq_f32(acc0, acc1))
        };
    }
    polyphase_dot32_f32_scalar(src, taps)
}

/// Scalar reference: arbitrary-tap (`16, 32, 64`) `f32` polyphase dot product.
#[inline]
pub fn polyphase_dot_f32_scalar(src: &[f32], taps: &[f32]) -> f32 {
    let n = src.len().min(taps.len());
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut k = 0usize;
    while k + 4 <= n {
        s0 += src[k] * taps[k];
        s1 += src[k + 1] * taps[k + 1];
        s2 += src[k + 2] * taps[k + 2];
        s3 += src[k + 3] * taps[k + 3];
        k += 4;
    }
    let mut sum = (s0 + s1) + (s2 + s3);
    while k < n {
        sum += src[k] * taps[k];
        k += 1;
    }
    sum
}

/// Vectorized `f32` polyphase dot product (`T in {16, 32, 64}` or any length).
#[inline]
pub fn polyphase_dot_f32(src: &[f32], taps: &[f32]) -> f32 {
    let n = src.len().min(taps.len());
    if n == 16 {
        let t: &[f32; 16] = taps[..16].try_into().unwrap();
        return polyphase_dot16_f32(src, t);
    }
    if n == 32 {
        let t: &[f32; 32] = taps[..32].try_into().unwrap();
        return polyphase_dot32_f32(src, t);
    }
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        return unsafe {
            use std::arch::aarch64::*;
            let sp = src.as_ptr();
            let tp = taps.as_ptr();
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut k = 0usize;
            while k + 8 <= n {
                acc0 = vfmaq_f32(acc0, vld1q_f32(sp.add(k)), vld1q_f32(tp.add(k)));
                acc1 = vfmaq_f32(acc1, vld1q_f32(sp.add(k + 4)), vld1q_f32(tp.add(k + 4)));
                k += 8;
            }
            let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));
            while k < n {
                sum += src[k] * taps[k];
                k += 1;
            }
            sum
        };
    }
    polyphase_dot_f32_scalar(src, taps)
}

/// Scalar reference: 16-tap Q15 fixed-point (`32768 = 1.0`) `i16` polyphase dot product.
#[inline]
pub fn polyphase_dot16_q15_i16_scalar(src: &[i16], taps: &[i16; 16]) -> i16 {
    debug_assert!(src.len() >= 16);
    let mut sum = 16384i32;
    for k in 0..16 {
        sum += (src[k] as i32) * (taps[k] as i32);
    }
    (sum >> 15).clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

/// Vectorized 16-tap Q15 fixed-point `i16` polyphase dot product using `aarch64` NEON (`vmull_s16` + `vmlal_s16` + `vaddvq_s32`).
#[inline]
pub fn polyphase_dot16_q15_i16(src: &[i16], taps: &[i16; 16]) -> i16 {
    debug_assert!(src.len() >= 16);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        return unsafe {
            use std::arch::aarch64::*;
            let s0 = vld1q_s16(src.as_ptr());
            let s1 = vld1q_s16(src.as_ptr().add(8));
            let t0 = vld1q_s16(taps.as_ptr());
            let t1 = vld1q_s16(taps.as_ptr().add(8));

            let mut acc = vmull_s16(vget_low_s16(s0), vget_low_s16(t0));
            acc = vmlal_high_s16(acc, s0, t0);
            acc = vmlal_s16(acc, vget_low_s16(s1), vget_low_s16(t1));
            acc = vmlal_high_s16(acc, s1, t1);

            let sum = vaddvq_s32(acc) + 16384;
            (sum >> 15).clamp(i16::MIN as i32, i16::MAX as i32) as i16
        };
    }
    polyphase_dot16_q15_i16_scalar(src, taps)
}

/// Scalar reference: 32-tap Q15 fixed-point `i16` polyphase dot product.
#[inline]
pub fn polyphase_dot32_q15_i16_scalar(src: &[i16], taps: &[i16; 32]) -> i16 {
    debug_assert!(src.len() >= 32);
    let mut sum = 16384i32;
    for k in 0..32 {
        sum += (src[k] as i32) * (taps[k] as i32);
    }
    (sum >> 15).clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

/// Vectorized 32-tap Q15 fixed-point `i16` polyphase dot product using `aarch64` NEON.
#[inline]
pub fn polyphase_dot32_q15_i16(src: &[i16], taps: &[i16; 32]) -> i16 {
    debug_assert!(src.len() >= 32);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        return unsafe {
            use std::arch::aarch64::*;
            let mut acc = vdupq_n_s32(0);
            for k in (0..32).step_by(8) {
                let s = vld1q_s16(src.as_ptr().add(k));
                let t = vld1q_s16(taps.as_ptr().add(k));
                acc = vmlal_s16(acc, vget_low_s16(s), vget_low_s16(t));
                acc = vmlal_high_s16(acc, s, t);
            }
            let sum = vaddvq_s32(acc) + 16384;
            (sum >> 15).clamp(i16::MIN as i32, i16::MAX as i32) as i16
        };
    }
    polyphase_dot32_q15_i16_scalar(src, taps)
}

// ============================================================================
// SIMD-0042: Channel Matrix Mixer (ITU-R BS.775 / libswresample/rematrix.c)
// ============================================================================

/// Scalar reference: Stereo `[L0, R0, L1, R1, ...]` to Mono downmix (`gain * (L + R)`).
pub fn rematrix_stereo_to_mono_f32_scalar(stereo: &[f32], mono: &mut [f32], gain: f32) {
    let frames = (stereo.len() / 2).min(mono.len());
    for i in 0..frames {
        mono[i] = (stereo[i * 2] + stereo[i * 2 + 1]) * gain;
    }
}

/// Vectorized Stereo to Mono downmix using `aarch64` NEON (`vld2q_f32`).
pub fn rematrix_stereo_to_mono_f32(stereo: &[f32], mono: &mut [f32], gain: f32) {
    let frames = (stereo.len() / 2).min(mono.len());
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        let mut i = 0usize;
        unsafe {
            use std::arch::aarch64::*;
            let vg = vdupq_n_f32(gain);
            while i + 4 <= frames {
                let lr = vld2q_f32(stereo.as_ptr().add(i * 2));
                let m = vmulq_f32(vaddq_f32(lr.0, lr.1), vg);
                vst1q_f32(mono.as_mut_ptr().add(i), m);
                i += 4;
            }
        }
        if i < frames {
            rematrix_stereo_to_mono_f32_scalar(
                &stereo[i * 2..frames * 2],
                &mut mono[i..frames],
                gain,
            );
        }
        return;
    }
    rematrix_stereo_to_mono_f32_scalar(&stereo[..frames * 2], &mut mono[..frames], gain);
}

/// Scalar reference: Mono to Stereo upmix (`[M0 * gain, M0 * gain, ...]`).
pub fn rematrix_mono_to_stereo_f32_scalar(mono: &[f32], stereo: &mut [f32], gain: f32) {
    let frames = mono.len().min(stereo.len() / 2);
    for i in 0..frames {
        let v = mono[i] * gain;
        stereo[i * 2] = v;
        stereo[i * 2 + 1] = v;
    }
}

/// Vectorized Mono to Stereo upmix using `aarch64` NEON (`vst2q_f32`).
pub fn rematrix_mono_to_stereo_f32(mono: &[f32], stereo: &mut [f32], gain: f32) {
    let frames = mono.len().min(stereo.len() / 2);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        let mut i = 0usize;
        unsafe {
            use std::arch::aarch64::*;
            let vg = vdupq_n_f32(gain);
            while i + 4 <= frames {
                let m = vmulq_f32(vld1q_f32(mono.as_ptr().add(i)), vg);
                vst2q_f32(stereo.as_mut_ptr().add(i * 2), float32x4x2_t(m, m));
                i += 4;
            }
        }
        if i < frames {
            rematrix_mono_to_stereo_f32_scalar(
                &mono[i..frames],
                &mut stereo[i * 2..frames * 2],
                gain,
            );
        }
        return;
    }
    rematrix_mono_to_stereo_f32_scalar(&mono[..frames], &mut stereo[..frames * 2], gain);
}

/// Scalar reference: ITU-R BS.775 `5.1` (`L, R, C, LFE, Ls, Rs`) to Stereo downmix:
/// $L' = \text{norm} \cdot (L + c_{\text{mix}} C + s_{\text{mix}} L_s + \text{lfe}_{\text{mix}} \text{LFE})$,
/// $R' = \text{norm} \cdot (R + c_{\text{mix}} C + s_{\text{mix}} R_s + \text{lfe}_{\text{mix}} \text{LFE})$.
pub fn rematrix_5_1_to_stereo_f32_scalar(
    src_5_1: &[f32],
    dst_stereo: &mut [f32],
    c_mix: f32,
    s_mix: f32,
    lfe_mix: f32,
    normalize: bool,
) {
    let frames = (src_5_1.len() / 6).min(dst_stereo.len() / 2);
    let norm = if normalize {
        1.0 / (1.0 + c_mix + s_mix + lfe_mix)
    } else {
        1.0
    };
    let g_lr = norm;
    let g_c = c_mix * norm;
    let g_s = s_mix * norm;
    let g_lfe = lfe_mix * norm;

    for i in 0..frames {
        let base = i * 6;
        let l = src_5_1[base];
        let r = src_5_1[base + 1];
        let c = src_5_1[base + 2];
        let lfe = src_5_1[base + 3];
        let ls = src_5_1[base + 4];
        let rs = src_5_1[base + 5];
        let common = c * g_c + lfe * g_lfe;
        dst_stereo[i * 2] = (l * g_lr + common) + ls * g_s;
        dst_stereo[i * 2 + 1] = (r * g_lr + common) + rs * g_s;
    }
}

/// Vectorized ITU-R BS.775 `5.1` to Stereo downmix over 4-frame blocks in L1 cache.
pub fn rematrix_5_1_to_stereo_f32(
    src_5_1: &[f32],
    dst_stereo: &mut [f32],
    c_mix: f32,
    s_mix: f32,
    lfe_mix: f32,
    normalize: bool,
) {
    let frames = (src_5_1.len() / 6).min(dst_stereo.len() / 2);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        let norm = if normalize {
            1.0 / (1.0 + c_mix + s_mix + lfe_mix)
        } else {
            1.0
        };
        let g_lr = norm;
        let g_c = c_mix * norm;
        let g_s = s_mix * norm;
        let g_lfe = lfe_mix * norm;

        let mut i = 0usize;
        unsafe {
            use std::arch::aarch64::*;
            let vg_lr = vdupq_n_f32(g_lr);
            let vg_c = vdupq_n_f32(g_c);
            let vg_s = vdupq_n_f32(g_s);
            let vg_lfe = vdupq_n_f32(g_lfe);

            while i + 4 <= frames {
                let mut l_arr = [0.0f32; 4];
                let mut r_arr = [0.0f32; 4];
                let mut c_arr = [0.0f32; 4];
                let mut lfe_arr = [0.0f32; 4];
                let mut ls_arr = [0.0f32; 4];
                let mut rs_arr = [0.0f32; 4];
                for k in 0..4 {
                    let b = (i + k) * 6;
                    l_arr[k] = src_5_1[b];
                    r_arr[k] = src_5_1[b + 1];
                    c_arr[k] = src_5_1[b + 2];
                    lfe_arr[k] = src_5_1[b + 3];
                    ls_arr[k] = src_5_1[b + 4];
                    rs_arr[k] = src_5_1[b + 5];
                }
                let vl = vmulq_f32(vld1q_f32(l_arr.as_ptr()), vg_lr);
                let vr = vmulq_f32(vld1q_f32(r_arr.as_ptr()), vg_lr);
                let vc = vmulq_f32(vld1q_f32(c_arr.as_ptr()), vg_c);
                let vlfe = vmulq_f32(vld1q_f32(lfe_arr.as_ptr()), vg_lfe);
                let vls = vmulq_f32(vld1q_f32(ls_arr.as_ptr()), vg_s);
                let vrs = vmulq_f32(vld1q_f32(rs_arr.as_ptr()), vg_s);

                let vcommon = vaddq_f32(vc, vlfe);
                let out_l = vaddq_f32(vaddq_f32(vl, vcommon), vls);
                let out_r = vaddq_f32(vaddq_f32(vr, vcommon), vrs);
                vst2q_f32(
                    dst_stereo.as_mut_ptr().add(i * 2),
                    float32x4x2_t(out_l, out_r),
                );
                i += 4;
            }
        }
        if i < frames {
            rematrix_5_1_to_stereo_f32_scalar(
                &src_5_1[i * 6..frames * 6],
                &mut dst_stereo[i * 2..frames * 2],
                c_mix,
                s_mix,
                lfe_mix,
                normalize,
            );
        }
        return;
    }
    rematrix_5_1_to_stereo_f32_scalar(
        &src_5_1[..frames * 6],
        &mut dst_stereo[..frames * 2],
        c_mix,
        s_mix,
        lfe_mix,
        normalize,
    );
}

/// Scalar reference: general $N_{\text{in}} \times M_{\text{out}}$ channel rematrixing over interleaved `f32` frames.
/// `matrix` has layout `matrix[out_ch * in_channels + in_ch]`.
pub fn rematrix_nxm_f32_scalar(
    src: &[f32],
    dst: &mut [f32],
    in_channels: usize,
    out_channels: usize,
    matrix: &[f32],
) {
    assert!(in_channels > 0 && out_channels > 0);
    assert!(matrix.len() >= in_channels * out_channels);
    let frames = (src.len() / in_channels).min(dst.len() / out_channels);
    for f in 0..frames {
        let s_row = &src[f * in_channels..(f + 1) * in_channels];
        let d_row = &mut dst[f * out_channels..(f + 1) * out_channels];
        for (oc, out_val) in d_row.iter_mut().enumerate() {
            let m_row = &matrix[oc * in_channels..(oc + 1) * in_channels];
            let mut sum = 0.0f32;
            for ic in 0..in_channels {
                sum += s_row[ic] * m_row[ic];
            }
            *out_val = sum;
        }
    }
}

/// Vectorized $N_{\text{in}} \times M_{\text{out}}$ channel rematrixing over 16-sample blocks in L1 cache.
#[allow(clippy::needless_range_loop)]
#[cfg_attr(not(target_arch = "aarch64"), allow(unused_mut))]
pub fn rematrix_nxm_f32(
    src: &[f32],
    dst: &mut [f32],
    in_channels: usize,
    out_channels: usize,
    matrix: &[f32],
) {
    assert!(in_channels > 0 && out_channels > 0 && in_channels <= 8 && out_channels <= 8);
    assert!(matrix.len() >= in_channels * out_channels);
    let frames = (src.len() / in_channels).min(dst.len() / out_channels);
    let mut f = 0usize;

    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        unsafe {
            use std::arch::aarch64::*;
            let mut in_planar = [[0.0f32; 16]; 8];
            let mut out_planar = [[0.0f32; 16]; 8];
            while f + 16 <= frames {
                for k in 0..16 {
                    let s_base = (f + k) * in_channels;
                    for ic in 0..in_channels {
                        in_planar[ic][k] = src[s_base + ic];
                    }
                }
                for oc in 0..out_channels {
                    let m_row = &matrix[oc * in_channels..(oc + 1) * in_channels];
                    for blk in (0..16).step_by(4) {
                        let mut acc = vdupq_n_f32(0.0);
                        for ic in 0..in_channels {
                            let g = vdupq_n_f32(m_row[ic]);
                            let v_in = vld1q_f32(in_planar[ic].as_ptr().add(blk));
                            acc = vaddq_f32(acc, vmulq_f32(v_in, g));
                        }
                        vst1q_f32(out_planar[oc].as_mut_ptr().add(blk), acc);
                    }
                }
                for k in 0..16 {
                    let d_base = (f + k) * out_channels;
                    for oc in 0..out_channels {
                        dst[d_base + oc] = out_planar[oc][k];
                    }
                }
                f += 16;
            }
        }
    }

    if f < frames {
        rematrix_nxm_f32_scalar(
            &src[f * in_channels..frames * in_channels],
            &mut dst[f * out_channels..frames * out_channels],
            in_channels,
            out_channels,
            matrix,
        );
    }
}

// ============================================================================
// SIMD-0043: EBU R128 K-Weighting Biquad Cascade & 4x True-Peak Detector
// ============================================================================

/// Second-order Direct Form II Transposed IIR biquad coefficients (`b0, b1, b2, a1, a2` with `a0 = 1.0`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiquadCoeffs {
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
    pub a2: f32,
}

impl BiquadCoeffs {
    /// Processes a single scalar sample `x` with Direct Form II Transposed state `(z1, z2)`.
    #[inline]
    pub fn process_sample_scalar(&self, x: f32, z1: &mut f32, z2: &mut f32) -> f32 {
        let y = self.b0 * x + *z1;
        *z1 = (self.b1 * x - self.a1 * y) + *z2;
        *z2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// Scalar reference: processes 4 channels in parallel through an identical biquad stage.
#[inline]
pub fn biquad_process_4ch_f32_scalar(
    coeffs: &BiquadCoeffs,
    x: &[f32; 4],
    z1: &mut [f32; 4],
    z2: &mut [f32; 4],
    y_out: &mut [f32; 4],
) {
    for c in 0..4 {
        let y = coeffs.b0 * x[c] + z1[c];
        z1[c] = (coeffs.b1 * x[c] - coeffs.a1 * y) + z2[c];
        z2[c] = coeffs.b2 * x[c] - coeffs.a2 * y;
        y_out[c] = y;
    }
}

/// Vectorized 4-channel biquad state update using `aarch64` NEON (`f32x4`), bit-exact with [`biquad_process_4ch_f32_scalar`].
#[inline]
pub fn biquad_process_4ch_f32(
    coeffs: &BiquadCoeffs,
    x: &[f32; 4],
    z1: &mut [f32; 4],
    z2: &mut [f32; 4],
    y_out: &mut [f32; 4],
) {
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        unsafe {
            use std::arch::aarch64::*;
            let vx = vld1q_f32(x.as_ptr());
            let vz1 = vld1q_f32(z1.as_ptr());
            let vz2 = vld1q_f32(z2.as_ptr());

            let vb0 = vdupq_n_f32(coeffs.b0);
            let vb1 = vdupq_n_f32(coeffs.b1);
            let vb2 = vdupq_n_f32(coeffs.b2);
            let va1 = vdupq_n_f32(coeffs.a1);
            let va2 = vdupq_n_f32(coeffs.a2);

            let vy = vaddq_f32(vmulq_f32(vb0, vx), vz1);
            let new_z1 = vaddq_f32(vsubq_f32(vmulq_f32(vb1, vx), vmulq_f32(va1, vy)), vz2);
            let new_z2 = vsubq_f32(vmulq_f32(vb2, vx), vmulq_f32(va2, vy));

            vst1q_f32(z1.as_mut_ptr(), new_z1);
            vst1q_f32(z2.as_mut_ptr(), new_z2);
            vst1q_f32(y_out.as_mut_ptr(), vy);
        }
        return;
    }
    biquad_process_4ch_f32_scalar(coeffs, x, z1, z2, y_out);
}

/// ITU-R BS.1770-4 Annex 2 Table 3 4-phase 12-tap polyphase FIR filter bank (`h[0..48]` centered at $n = 24$),
/// transposed as `[tap_k][phase_0..4]` so all 4 oversampled phases can be accumulated simultaneously in a single `float32x4_t` vector.
pub const TRUE_PEAK_12X4_TAPS: [[f32; 4]; 12] = [
    [0.0017089844, -0.029174805, -0.018920898, -0.008300781],
    [0.010986328, 0.029296875, 0.033081055, 0.014892578],
    [-0.01965332, -0.051757813, -0.05822754, -0.026611328],
    [0.033203125, 0.08911133, 0.1015625, 0.047607422],
    [-0.059448242, -0.1665039, -0.20031738, -0.10229492],
    [0.1373291, 0.4650879, 0.77978516, 0.97216797],
    [0.97216797, 0.77978516, 0.4650879, 0.1373291],
    [-0.10229492, -0.20031738, -0.1665039, -0.059448242],
    [0.047607422, 0.1015625, 0.08911133, 0.033203125],
    [-0.026611328, -0.05822754, -0.051757813, -0.01965332],
    [0.014892578, 0.033081055, 0.029296875, 0.010986328],
    [-0.008300781, -0.018920898, -0.029174805, 0.0017089844],
];

/// Scalar reference: evaluates all 4 oversampled phases of the 12-tap ITU-R BS.1770-4 True-Peak FIR
/// over `history` (12 consecutive samples of one channel) and returns the maximum absolute amplitude.
#[inline]
pub fn true_peak_4x_12tap_sample_f32_scalar(history: &[f32; 12]) -> f32 {
    let mut acc = [0.0f32; 4];
    for k in 0..12 {
        let x = history[k];
        let t = &TRUE_PEAK_12X4_TAPS[k];
        acc[0] += t[0] * x;
        acc[1] += t[1] * x;
        acc[2] += t[2] * x;
        acc[3] += t[3] * x;
    }
    let m01 = acc[0].abs().max(acc[1].abs());
    let m23 = acc[2].abs().max(acc[3].abs());
    m01.max(m23)
}

/// Vectorized 4x oversampled 12-tap ITU-R BS.1770-4 True-Peak FIR evaluation using `aarch64` NEON
/// (`vmulq_n_f32` + `vaddq_f32` + `vmaxvq_f32(vabsq_f32(acc))`), bit-exact with [`true_peak_4x_12tap_sample_f32_scalar`].
#[inline]
pub fn true_peak_4x_12tap_sample_f32(history: &[f32; 12]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        return unsafe {
            use std::arch::aarch64::*;
            let mut acc = vdupq_n_f32(0.0);
            for (k, &x) in history.iter().enumerate() {
                let vt = vld1q_f32(TRUE_PEAK_12X4_TAPS[k].as_ptr());
                acc = vaddq_f32(acc, vmulq_n_f32(vt, x));
            }
            vmaxvq_f32(vabsq_f32(acc))
        };
    }
    true_peak_4x_12tap_sample_f32_scalar(history)
}

/// Scalar reference: applies linear `gain` to `samples` in-place while clamping to `[-ceiling, ceiling]`.
pub fn apply_gain_true_peak_limit_f32_scalar(samples: &mut [f32], gain: f32, ceiling: f32) {
    let neg_ceil = -ceiling;
    for s in samples.iter_mut() {
        *s = (*s * gain).clamp(neg_ceil, ceiling);
    }
}

/// Vectorized gain normalizer and ceiling limiter using `aarch64` NEON (`vmulq_f32` + `vminq_f32` + `vmaxq_f32`),
/// bit-exact with [`apply_gain_true_peak_limit_f32_scalar`].
#[cfg_attr(not(target_arch = "aarch64"), allow(unused_mut))]
pub fn apply_gain_true_peak_limit_f32(samples: &mut [f32], gain: f32, ceiling: f32) {
    let n = samples.len();
    let mut i = 0usize;
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        unsafe {
            use std::arch::aarch64::*;
            let vg = vdupq_n_f32(gain);
            let vmax = vdupq_n_f32(ceiling);
            let vmin = vdupq_n_f32(-ceiling);
            while i + 4 <= n {
                let v = vld1q_f32(samples.as_ptr().add(i));
                let scaled = vminq_f32(vmaxq_f32(vmulq_f32(v, vg), vmin), vmax);
                vst1q_f32(samples.as_mut_ptr().add(i), scaled);
                i += 4;
            }
        }
    }
    if i < n {
        apply_gain_true_peak_limit_f32_scalar(&mut samples[i..n], gain, ceiling);
    }
}

// ============================================================================
// 0044: Shared SIMD MDCT / IMDCT & Sine / KBD Windowing Engine
// ============================================================================

/// Generates a symmetric Sine window of length `n`:
/// $w[k] = \sin\!\left(\frac{\pi}{n} \left(k + 0.5\right)\right)$ for $0 \le k < n$.
pub fn sine_window_f32(n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| ((PI / (n as f64)) * (k as f64 + 0.5)).sin() as f32)
        .collect()
}

/// Generates an ISO/IEC 14496-3 Kaiser-Bessel-Derived (KBD) window of full length `n` (where `n` is even, e.g. `2048` or `256`)
/// with shape parameter `alpha` (`4.0` for AAC-LC long blocks, `6.0` for short blocks) using `ragdoll_core::bessel_i0`.
pub fn kbd_window_f32(n: usize, alpha: f64) -> Vec<f32> {
    assert!(n.is_multiple_of(2) && n > 0);
    let half = n / 2;
    let beta = PI * alpha;
    let denom = ragdoll_core::bessel_i0(beta);
    let mut cum = Vec::with_capacity(half + 1);
    let mut running = 0.0f64;
    let m = half as f64;
    for k in 0..=half {
        let r = (2.0 * (k as f64) / m) - 1.0;
        let arg = (1.0 - r * r).max(0.0).sqrt();
        let w = ragdoll_core::bessel_i0(beta * arg) / denom;
        running += w;
        cum.push(running);
    }
    let total = cum[half];
    let mut win = vec![0.0f32; n];
    for k in 0..half {
        let v = (cum[k] / total).sqrt() as f32;
        win[k] = v;
        win[n - 1 - k] = v;
    }
    win
}

/// Precomputed twiddle tables for an $N$-sample MDCT / IMDCT (producing $M = N / 2$ spectral coefficients).
#[derive(Clone, Debug)]
pub struct MdctPlan {
    /// Transform block length $N$ (`2 * M` time-domain samples, `M` frequency bins, `K = N / 4` complex FFT points).
    pub n: usize,
    /// Pre-twiddle cosine table of length $K = N / 4$: $\cos(2\pi (k + 1/8) / N)$.
    pub twiddle_cos: Vec<f32>,
    /// Pre-twiddle sine table of length $K = N / 4$: $\sin(2\pi (k + 1/8) / N)$.
    pub twiddle_sin: Vec<f32>,
}

impl MdctPlan {
    /// Creates an `MdctPlan` for transform size `n` (must be divisible by 4, e.g., `120, 240, 256, 480, 960, 2048`).
    pub fn new(n: usize) -> Self {
        assert!(n.is_multiple_of(4) && n >= 4);
        let k_len = n / 4;
        let mut twiddle_cos = Vec::with_capacity(k_len);
        let mut twiddle_sin = Vec::with_capacity(k_len);
        for k in 0..k_len {
            let angle = 2.0 * PI * (k as f64 + 0.125) / (n as f64);
            twiddle_cos.push(angle.cos() as f32);
            twiddle_sin.push(angle.sin() as f32);
        }
        Self {
            n,
            twiddle_cos,
            twiddle_sin,
        }
    }

    /// Self-inverse Type-IV Discrete Cosine Transform (`DCT-IV`) of length $M = N / 2$ via $K = N / 4$-point complex FFT.
    fn dct4(&self, u_in: &[f32], x_out: &mut [f32]) {
        let m = self.n / 2;
        let k_len = self.n / 4;
        debug_assert!(u_in.len() >= m && x_out.len() >= m);

        let mut z_re = vec![0.0f32; k_len];
        let mut z_im = vec![0.0f32; k_len];

        for k in 0..k_len {
            let a = u_in[2 * k];
            let b = u_in[m - 1 - 2 * k];
            let c = self.twiddle_cos[k];
            let s = self.twiddle_sin[k];
            z_re[k] = a * c + b * s;
            z_im[k] = b * c - a * s;
        }

        fft_complex_in_place(&mut z_re, &mut z_im, false);

        for r in 0..k_len {
            let c = self.twiddle_cos[r];
            let s = self.twiddle_sin[r];
            x_out[2 * r] = z_re[r] * c + z_im[r] * s;
            x_out[m - 1 - 2 * r] = z_re[r] * s - z_im[r] * c;
        }
    }

    /// Forward MDCT (`time_in` of length $N \to$ `coeffs_out` of length $M = N / 2$):
    /// $$X[k] = \sum_{n=0}^{N-1} x[n] \cos\!\left(\frac{\pi}{M}\left(n + \frac{1}{2} + \frac{M}{2}\right)\left(k + \frac{1}{2}\right)\right)$$
    pub fn mdct(&self, time_in: &[f32], coeffs_out: &mut [f32]) {
        let n = self.n;
        let m = n / 2;
        let n4 = n / 4;
        let n34 = 3 * n / 4;
        assert!(time_in.len() >= n && coeffs_out.len() >= m);

        let mut u = vec![0.0f32; m];
        for k in 0..n4 {
            u[k] = -time_in[n34 + k] - time_in[n34 - 1 - k];
            u[n4 + k] = time_in[k] - time_in[m - 1 - k];
        }
        self.dct4(&u, coeffs_out);
    }

    /// Inverse MDCT (`coeffs_in` of length $M = N / 2 \to$ `time_out` of length $N$), scaled by $2 / M$
    /// so that $\text{IMDCT}(\text{MDCT}(w \cdot x)) \cdot w$ with $50\%$ TDAC overlap-add reconstructs $x$ with unity gain.
    pub fn imdct(&self, coeffs_in: &[f32], time_out: &mut [f32]) {
        let n = self.n;
        let m = n / 2;
        let n4 = n / 4;
        let n34 = 3 * n / 4;
        assert!(coeffs_in.len() >= m && time_out.len() >= n);

        let mut u = vec![0.0f32; m];
        self.dct4(coeffs_in, &mut u);
        let scale = 2.0 / (m as f32);

        for k in 0..n4 {
            let u_hi = u[n4 + k] * scale;
            let u_lo = u[k] * scale;
            time_out[k] = u_hi;
            time_out[m - 1 - k] = -u_hi;
            time_out[n34 + k] = -u_lo;
            time_out[n34 - 1 - k] = -u_lo;
        }
    }
}

/// In-place complex FFT (`inverse = false` computes $\sum x[n] e^{-j 2\pi k n / K}$,
/// `inverse = true` computes unnormalized $\sum x[n] e^{+j 2\pi k n / K}$).
/// Supports both power-of-two lengths (Cooley-Tukey radix-2, e.g. `K = 64, 512` for AAC-LC)
/// and mixed-radix $15 \times 2^p$ lengths (`K = 30, 60, 120, 240` for Opus CELT).
pub fn fft_complex_in_place(re: &mut [f32], im: &mut [f32], inverse: bool) {
    let n = re.len();
    assert_eq!(n, im.len());
    if n <= 1 {
        return;
    }
    let sign = if inverse { 1.0f64 } else { -1.0f64 };

    if n.is_power_of_two() {
        // Bit-reversal permutation
        let mut j = 0usize;
        for i in 1..n {
            let mut bit = n >> 1;
            while (j & bit) != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        // Cooley-Tukey radix-2 butterflies
        let mut len = 2usize;
        while len <= n {
            let half = len / 2;
            let step_angle = sign * 2.0 * PI / (len as f64);
            for k in (0..n).step_by(len) {
                for m in 0..half {
                    let ang = step_angle * (m as f64);
                    let wr = ang.cos() as f32;
                    let wi = ang.sin() as f32;
                    let u_r = re[k + m];
                    let u_i = im[k + m];
                    let v_r = re[k + m + half] * wr - im[k + m + half] * wi;
                    let v_i = re[k + m + half] * wi + im[k + m + half] * wr;
                    re[k + m] = u_r + v_r;
                    im[k + m] = u_i + v_i;
                    re[k + m + half] = u_r - v_r;
                    im[k + m + half] = u_i - v_i;
                }
            }
            len <<= 1;
        }
        return;
    }

    // Mixed-radix Cooley-Tukey decomposition for N = R * M (e.g. 15 * 2^p for Opus CELT)
    let radix = if n.is_multiple_of(15) {
        15
    } else if n.is_multiple_of(5) {
        5
    } else if n.is_multiple_of(3) {
        3
    } else {
        n
    };
    if radix == n {
        let orig_re = re.to_vec();
        let orig_im = im.to_vec();
        for k in 0..n {
            let mut sr = 0.0f32;
            let mut si = 0.0f32;
            for t in 0..n {
                let ang = sign * 2.0 * PI * ((k * t) % n) as f64 / (n as f64);
                let c = ang.cos() as f32;
                let s = ang.sin() as f32;
                sr += orig_re[t] * c - orig_im[t] * s;
                si += orig_re[t] * s + orig_im[t] * c;
            }
            re[k] = sr;
            im[k] = si;
        }
        return;
    }

    let m = n / radix;
    let mut sub_re = vec![0.0f32; m];
    let mut sub_im = vec![0.0f32; m];
    let mut cols_re = vec![0.0f32; n];
    let mut cols_im = vec![0.0f32; n];

    for r in 0..radix {
        for q in 0..m {
            sub_re[q] = re[q * radix + r];
            sub_im[q] = im[q * radix + r];
        }
        fft_complex_in_place(&mut sub_re, &mut sub_im, inverse);
        for q in 0..m {
            let ang = sign * 2.0 * PI * (r * q) as f64 / (n as f64);
            let c = ang.cos() as f32;
            let s = ang.sin() as f32;
            cols_re[r * m + q] = sub_re[q] * c - sub_im[q] * s;
            cols_im[r * m + q] = sub_re[q] * s + sub_im[q] * c;
        }
    }

    for q in 0..m {
        for p in 0..radix {
            let mut sr = 0.0f32;
            let mut si = 0.0f32;
            for r in 0..radix {
                let ang = sign * 2.0 * PI * ((p * r) % radix) as f64 / (radix as f64);
                let c = ang.cos() as f32;
                let s = ang.sin() as f32;
                let xr = cols_re[r * m + q];
                let xi = cols_im[r * m + q];
                sr += xr * c - xi * s;
                si += xr * s + xi * c;
            }
            re[p * m + q] = sr;
            im[p * m + q] = si;
        }
    }
}

/// Scalar reference: IMDCT 50% TDAC windowing and overlap-add synthesis.
/// Outputs `m = n / 2` samples into `pcm_out` and updates `overlap_state` (`m` samples) for the next frame.
pub fn window_overlap_add_f32_scalar(
    imdct_buf: &[f32],
    window: &[f32],
    overlap_state: &mut [f32],
    pcm_out: &mut [f32],
) {
    let n = imdct_buf.len().min(window.len());
    let m = n / 2;
    assert!(overlap_state.len() >= m && pcm_out.len() >= m);
    for i in 0..m {
        pcm_out[i] = overlap_state[i] + imdct_buf[i] * window[i];
        overlap_state[i] = imdct_buf[m + i] * window[m + i];
    }
}

/// Vectorized IMDCT 50% TDAC windowing and overlap-add synthesis using `aarch64` NEON (`vmulq_f32` + `vaddq_f32`),
/// bit-exact with [`window_overlap_add_f32_scalar`].
#[cfg_attr(not(target_arch = "aarch64"), allow(unused_mut))]
pub fn window_overlap_add_f32(
    imdct_buf: &[f32],
    window: &[f32],
    overlap_state: &mut [f32],
    pcm_out: &mut [f32],
) {
    let n = imdct_buf.len().min(window.len());
    let m = n / 2;
    assert!(overlap_state.len() >= m && pcm_out.len() >= m);
    let mut i = 0usize;
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        unsafe {
            use std::arch::aarch64::*;
            while i + 4 <= m {
                let v_prev = vld1q_f32(overlap_state.as_ptr().add(i));
                let v_lo = vld1q_f32(imdct_buf.as_ptr().add(i));
                let w_lo = vld1q_f32(window.as_ptr().add(i));
                let v_hi = vld1q_f32(imdct_buf.as_ptr().add(m + i));
                let w_hi = vld1q_f32(window.as_ptr().add(m + i));

                let out = vaddq_f32(v_prev, vmulq_f32(v_lo, w_lo));
                let next_ov = vmulq_f32(v_hi, w_hi);
                vst1q_f32(pcm_out.as_mut_ptr().add(i), out);
                vst1q_f32(overlap_state.as_mut_ptr().add(i), next_ov);
                i += 4;
            }
        }
    }
    if i < m {
        for k in i..m {
            pcm_out[k] = overlap_state[k] + imdct_buf[k] * window[k];
            overlap_state[k] = imdct_buf[m + k] * window[m + k];
        }
    }
}
