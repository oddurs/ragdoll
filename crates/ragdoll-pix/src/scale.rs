//! Fused L1-cache-resident separable polyphase resampler (`Lanczos3`, `MitchellNetravali`, `Bilinear`).
//!
//! Modeled after `libswscale/aarch64/hscale.S` and upgraded with:
//! 1. **Boundary-Folded 8-Tap Q14 Polyphase Filter Banks**: Edge clamping is folded into a contiguous
//!    8-sample window `[win_start .. win_start + 8]` at table construction time, with exact $\sum w_i = 16\,384$
//!    DC-preservation so horizontal NEON filtering uses single-instruction `vld1_u8` loads with zero branches.
//! 2. **4-Pixel-Parallel `vpaddq_s32` Horizontal NEON Kernel**: Computes 4 output pixels per iteration
//!    using `vmull_s16` + `vmlal_high_s16` + two-stage `vpaddq_s32` pairwise vector reduction.
//! 3. **16-Pixel-Parallel Vertical NEON Kernel over an 8-Row L1 Rolling Ring Buffer**: Eliminates
//!    full-frame intermediate DRAM allocations; only 8 horizontally-scaled scanlines (~15 KB for 1080p)
//!    stay resident in L1 cache.

use std::f64::consts::PI;

use ragdoll_core::{Plane, VideoFrame};
use ragdoll_simd::{Q14_HALF, Q14_ONE, clamp_u8, convolve_8tap_q14_u8_scalar};

/// Resampling reconstruction kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ScaleFilter {
    /// 2-tap linear tent interpolation.
    Bilinear,
    /// 4-tap Mitchell-Netravali bicubic spline ($B = 1/3, C = 1/3$).
    MitchellNetravali,
    /// 6-tap piecewise cubic `Spline36` mastering filter (`zimg` / `zscale`).
    Spline36,
    /// 6-tap / 8-tap windowed sinc Lanczos-3 ($a = 3$).
    #[default]
    Lanczos3,
}

impl ScaleFilter {
    #[inline]
    pub fn support_radius(self) -> f64 {
        match self {
            Self::Bilinear => 1.0,
            Self::MitchellNetravali => 2.0,
            Self::Spline36 => 3.0,
            Self::Lanczos3 => 3.0,
        }
    }

    #[inline]
    pub fn eval(self, x: f64) -> f64 {
        let ax = x.abs();
        match self {
            Self::Bilinear => {
                if ax < 1.0 {
                    1.0 - ax
                } else {
                    0.0
                }
            }
            Self::MitchellNetravali => {
                // B = 1/3, C = 1/3
                if ax < 1.0 {
                    (7.0 * ax * ax * ax - 12.0 * ax * ax + 16.0 / 3.0) / 6.0
                } else if ax < 2.0 {
                    (-7.0 / 3.0 * ax * ax * ax + 12.0 * ax * ax - 20.0 * ax + 32.0 / 3.0) / 6.0
                } else {
                    0.0
                }
            }
            Self::Spline36 => {
                // Exact `zimg::resize::Spline36Filter` polynomials
                if ax < 1.0 {
                    1.0 + ax * (-3.0 / 209.0 + ax * (-453.0 / 209.0 + ax * (13.0 / 11.0)))
                } else if ax < 2.0 {
                    let t = ax - 1.0;
                    t * (-156.0 / 209.0 + t * (270.0 / 209.0 + t * (-6.0 / 11.0)))
                } else if ax < 3.0 {
                    let t = ax - 2.0;
                    t * (26.0 / 209.0 + t * (-45.0 / 209.0 + t * (1.0 / 11.0)))
                } else {
                    0.0
                }
            }
            Self::Lanczos3 => {
                if ax < 1e-9 {
                    1.0
                } else if ax < 3.0 {
                    let pix = PI * ax;
                    let pix3 = pix / 3.0;
                    (pix.sin() / pix) * (pix3.sin() / pix3)
                } else {
                    0.0
                }
            }
        }
    }
}

/// Precomputed 8-tap Q14 (`sum == 16384`) polyphase filter bank for a 1D `src_len -> dst_len` axis.
#[derive(Clone, Debug)]
pub struct PolyphaseFilterBank {
    pub src_len: usize,
    pub dst_len: usize,
    /// Contiguous 8-sample window start index in `0..=src_len.saturating_sub(8)` for each output coordinate.
    pub win_starts: Vec<usize>,
    /// 8-tap Q14 weights (`sum == 16384`) for each output coordinate.
    pub weights: Vec<[i16; 8]>,
}

impl PolyphaseFilterBank {
    pub fn new(src_len: usize, dst_len: usize, filter: ScaleFilter) -> Self {
        assert!(src_len > 0 && dst_len > 0);
        let mut win_starts = Vec::with_capacity(dst_len);
        let mut weights = Vec::with_capacity(dst_len);

        let ratio = (src_len as f64) / (dst_len as f64);
        // Widen filter support when downscaling for proper anti-aliasing, capped so 8 taps cover the window
        let base_radius = filter.support_radius();
        let max_scale = 3.95 / base_radius;
        let filter_scale = ratio.clamp(1.0, max_scale);
        let inv_scale = 1.0 / filter_scale;

        let max_win_start = src_len.saturating_sub(8);

        for d in 0..dst_len {
            let center = ((d as f64) + 0.5) * ratio - 0.5;
            let ideal_start = (center.floor() as isize) - 3;
            let win_start = if src_len <= 8 {
                0usize
            } else {
                ideal_start.clamp(0, max_win_start as isize) as usize
            };

            // Evaluate continuous weights at the 8 ideal taps `ideal_start + k` and fold clamped
            // boundary coordinates into `win_start .. win_start + 8`.
            let mut folded_f64 = [0.0f64; 8];
            for k in 0..8isize {
                let tap_coord = ideal_start + k;
                let dist = ((tap_coord as f64) - center) * inv_scale;
                let w = filter.eval(dist);
                let clamped = tap_coord.clamp(0, (src_len as isize) - 1) as usize;
                let slot = (clamped - win_start).min(7);
                folded_f64[slot] += w;
            }

            let sum_f64: f64 = folded_f64.iter().sum();
            let mut q14_taps = [0i16; 8];
            let mut q14_sum = 0i32;
            let mut max_slot = 0usize;
            let mut max_abs = -1.0f64;

            if sum_f64.abs() > 1e-12 {
                // Error-diffusion coefficient quantization (`zimg::resize::matrix_to_filter`)
                // preserves the sub-pixel phase center of mass across the 8 taps.
                let mut i16_err = 0.0f64;
                for (slot, &wf) in folded_f64.iter().enumerate() {
                    let norm = wf / sum_f64;
                    let expected_i16 = norm * (Q14_ONE as f64) - i16_err;
                    let q = expected_i16.round().clamp(-32768.0, 32767.0) as i16;
                    i16_err = (q as f64) - expected_i16;
                    q14_taps[slot] = q;
                    q14_sum += q as i32;
                    if norm.abs() > max_abs {
                        max_abs = norm.abs();
                        max_slot = slot;
                    }
                }
                // Fold final <= 1 LSB residual into the dominant center tap so sum == 16384
                let diff = Q14_ONE - q14_sum;
                q14_taps[max_slot] = (q14_taps[max_slot] as i32 + diff) as i16;
            } else {
                q14_taps[0] = Q14_ONE as i16;
            }

            debug_assert_eq!(q14_taps.iter().map(|&w| w as i32).sum::<i32>(), Q14_ONE);
            win_starts.push(win_start);
            weights.push(q14_taps);
        }

        Self {
            src_len,
            dst_len,
            win_starts,
            weights,
        }
    }
}

/// Horizontally scales a single `u8` scanline using an 8-tap Q14 `PolyphaseFilterBank`.
///
/// NOTE: `src_row` must have at least `max(src_len, 8)` accessible bytes (guaranteed by 64-byte aligned `Plane` rows).
#[inline]
pub fn hscale_row_8tap_q14(src_row: &[u8], dst_row: &mut [u8], bank: &PolyphaseFilterBank) {
    #[cfg(target_arch = "aarch64")]
    {
        if src_row.len() >= 8 {
            unsafe { hscale_row_8tap_q14_neon(src_row, dst_row, bank) };
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if src_row.len() >= 8 && std::arch::is_x86_feature_detected!("avx2") {
            unsafe { hscale_row_8tap_q14_avx2(src_row, dst_row, bank) };
            return;
        }
    }
    hscale_row_8tap_q14_scalar(src_row, dst_row, bank);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hscale_row_8tap_q14_avx2(src_row: &[u8], dst_row: &mut [u8], bank: &PolyphaseFilterBank) {
    use std::arch::x86_64::*;
    let dst_len = bank.dst_len;
    assert!(dst_row.len() >= dst_len);
    for x in 0..dst_len {
        let ws = bank.win_starts[x];
        let s8 = unsafe { _mm_loadl_epi64(src_row.as_ptr().add(ws) as *const __m128i) };
        let s16 = _mm_cvtepu8_epi16(s8);
        let w16 = unsafe { _mm_loadu_si128(bank.weights[x].as_ptr() as *const __m128i) };
        let prod = _mm_madd_epi16(s16, w16);
        let sum2 = _mm_hadd_epi32(prod, prod);
        let sum1 = _mm_hadd_epi32(sum2, sum2);
        let acc = _mm_cvtsi128_si32(sum1) + Q14_HALF;
        dst_row[x] = clamp_u8(acc >> 14);
    }
}

#[allow(clippy::needless_range_loop)]
pub fn hscale_row_8tap_q14_scalar(src_row: &[u8], dst_row: &mut [u8], bank: &PolyphaseFilterBank) {
    let dst_len = bank.dst_len;
    if src_row.len() >= 8 {
        for x in 0..dst_len {
            let ws = bank.win_starts[x];
            let samples: &[u8; 8] = src_row[ws..ws + 8].try_into().unwrap();
            dst_row[x] = convolve_8tap_q14_u8_scalar(samples, &bank.weights[x]);
        }
    } else {
        let mut padded = [0u8; 8];
        padded[..src_row.len()].copy_from_slice(src_row);
        for x in 0..dst_len {
            dst_row[x] = convolve_8tap_q14_u8_scalar(&padded, &bank.weights[x]);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn hscale_row_8tap_q14_neon(src_row: &[u8], dst_row: &mut [u8], bank: &PolyphaseFilterBank) {
    use std::arch::aarch64::*;
    let dst_len = bank.dst_len;
    assert!(dst_row.len() >= dst_len);
    assert!(bank.win_starts.len() >= dst_len && bank.weights.len() >= dst_len);

    let src_ptr = src_row.as_ptr();
    let dst_ptr = dst_row.as_mut_ptr();
    let starts_ptr = bank.win_starts.as_ptr();
    let weights_ptr = bank.weights.as_ptr();
    let mut x = 0usize;

    // Helper for 4 output pixels -> int32x4_t sum
    let eval4 = |base: usize| unsafe {
        let s0 = vld1_u8(src_ptr.add(*starts_ptr.add(base)));
        let s1 = vld1_u8(src_ptr.add(*starts_ptr.add(base + 1)));
        let s2 = vld1_u8(src_ptr.add(*starts_ptr.add(base + 2)));
        let s3 = vld1_u8(src_ptr.add(*starts_ptr.add(base + 3)));

        let w0 = vld1q_s16((*weights_ptr.add(base)).as_ptr());
        let w1 = vld1q_s16((*weights_ptr.add(base + 1)).as_ptr());
        let w2 = vld1q_s16((*weights_ptr.add(base + 2)).as_ptr());
        let w3 = vld1q_s16((*weights_ptr.add(base + 3)).as_ptr());

        let s0_16 = vreinterpretq_s16_u16(vmovl_u8(s0));
        let s1_16 = vreinterpretq_s16_u16(vmovl_u8(s1));
        let s2_16 = vreinterpretq_s16_u16(vmovl_u8(s2));
        let s3_16 = vreinterpretq_s16_u16(vmovl_u8(s3));

        let acc0 = vmlal_high_s16(vmull_s16(vget_low_s16(s0_16), vget_low_s16(w0)), s0_16, w0);
        let acc1 = vmlal_high_s16(vmull_s16(vget_low_s16(s1_16), vget_low_s16(w1)), s1_16, w1);
        let acc2 = vmlal_high_s16(vmull_s16(vget_low_s16(s2_16), vget_low_s16(w2)), s2_16, w2);
        let acc3 = vmlal_high_s16(vmull_s16(vget_low_s16(s3_16), vget_low_s16(w3)), s3_16, w3);

        let p01 = vpaddq_s32(acc0, acc1);
        let p23 = vpaddq_s32(acc2, acc3);
        vpaddq_s32(p01, p23)
    };

    // Process 8 output pixels per iteration with direct NEON `vst1_u8` store (zero NEON->GPR stall)
    while x + 8 <= dst_len {
        let sum_lo = eval4(x);
        let sum_hi = eval4(x + 4);
        let shifted16 = vrshrn_high_n_s32::<14>(vrshrn_n_s32::<14>(sum_lo), sum_hi);
        let out8 = vqmovun_s16(shifted16);
        unsafe {
            vst1_u8(dst_ptr.add(x), out8);
        }
        x += 8;
    }

    while x + 4 <= dst_len {
        let sum4 = eval4(x);
        let shifted16 = vrshrn_n_s32::<14>(sum4);
        let out8 = vqmovun_s16(vcombine_s16(shifted16, shifted16));
        unsafe {
            vst1_lane_u32::<0>(dst_ptr.add(x) as *mut u32, vreinterpret_u32_u8(out8));
        }
        x += 4;
    }

    while x < dst_len {
        let ws = bank.win_starts[x];
        let samples: &[u8; 8] = src_row[ws..ws + 8].try_into().unwrap();
        dst_row[x] = convolve_8tap_q14_u8_scalar(samples, &bank.weights[x]);
        x += 1;
    }
}

/// Vertically convolves 8 horizontally-scaled rows from an L1 ring buffer into `dst_row`.
#[inline]
pub fn vscale_row_8tap_q14(rows: [&[u8]; 8], weights: &[i16; 8], dst_row: &mut [u8], width: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { vscale_row_8tap_q14_neon(rows, weights, dst_row, width) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe { vscale_row_8tap_q14_avx2(rows, weights, dst_row, width) }
        } else {
            vscale_row_8tap_q14_scalar(rows, weights, dst_row, width)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        vscale_row_8tap_q14_scalar(rows, weights, dst_row, width)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vscale_row_8tap_q14_avx2(
    rows: [&[u8]; 8],
    weights: &[i16; 8],
    dst_row: &mut [u8],
    width: usize,
) {
    use std::arch::x86_64::*;
    let mut x = 0usize;
    let v_bias = _mm256_set1_epi32(Q14_HALF);
    while x + 8 <= width {
        let mut acc = v_bias;
        for t in 0..8 {
            let w = weights[t] as i32;
            if w == 0 {
                continue;
            }
            let r8 = unsafe { _mm_loadl_epi64(rows[t].as_ptr().add(x) as *const __m128i) };
            let r32 = _mm256_cvtepu8_epi32(r8);
            acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(r32, _mm256_set1_epi32(w)));
        }
        let shifted = _mm256_srai_epi32(acc, 14);
        let mut lanes = [0i32; 8];
        unsafe { _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, shifted) };
        for k in 0..8 {
            dst_row[x + k] = clamp_u8(lanes[k]);
        }
        x += 8;
    }
    while x < width {
        let mut acc = Q14_HALF;
        for t in 0..8 {
            acc += (rows[t][x] as i32) * (weights[t] as i32);
        }
        dst_row[x] = clamp_u8(acc >> 14);
        x += 1;
    }
}

pub fn vscale_row_8tap_q14_scalar(
    rows: [&[u8]; 8],
    weights: &[i16; 8],
    dst_row: &mut [u8],
    width: usize,
) {
    for x in 0..width {
        let mut acc = Q14_HALF;
        for t in 0..8 {
            acc += (rows[t][x] as i32) * (weights[t] as i32);
        }
        dst_row[x] = clamp_u8(acc >> 14);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn vscale_row_8tap_q14_neon(
    rows: [&[u8]; 8],
    weights: &[i16; 8],
    dst_row: &mut [u8],
    width: usize,
) {
    use std::arch::aarch64::*;
    let v_bias = vdupq_n_s32(Q14_HALF);
    let mut x = 0usize;

    // Process 16 horizontal pixels simultaneously per vertical NEON pass
    while x + 16 <= width {
        let mut acc0 = v_bias;
        let mut acc1 = v_bias;
        let mut acc2 = v_bias;
        let mut acc3 = v_bias;

        for t in 0..8 {
            let w = weights[t];
            if w == 0 {
                continue;
            }
            let r8 = unsafe { vld1q_u8(rows[t].as_ptr().add(x)) };
            let r_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(r8)));
            let r_hi = vreinterpretq_s16_u16(vmovl_high_u8(r8));

            acc0 = vmlal_n_s16(acc0, vget_low_s16(r_lo), w);
            acc1 = vmlal_high_n_s16(acc1, r_lo, w);
            acc2 = vmlal_n_s16(acc2, vget_low_s16(r_hi), w);
            acc3 = vmlal_high_n_s16(acc3, r_hi, w);
        }

        let s01 = vcombine_s16(vshrn_n_s32::<14>(acc0), vshrn_n_s32::<14>(acc1));
        let s23 = vcombine_s16(vshrn_n_s32::<14>(acc2), vshrn_n_s32::<14>(acc3));
        let out8 = vcombine_u8(vqmovun_s16(s01), vqmovun_s16(s23));
        unsafe {
            vst1q_u8(dst_row.as_mut_ptr().add(x), out8);
        }
        x += 16;
    }

    while x < width {
        let mut acc = Q14_HALF;
        for t in 0..8 {
            acc += (rows[t][x] as i32) * (weights[t] as i32);
        }
        dst_row[x] = clamp_u8(acc >> 14);
        x += 1;
    }
}

/// Horizontally scales a `u8` scanline into a 14-bit signed `Q6` (`u8 << 6`) intermediate `i16` row.
///
/// Modeled after `libswscale` `ff_hscale8to15_X8_neon` and `zimg`: preserves 6 fractional bits
/// (`1/64`th of an 8-bit LSB) and keeps negative / $>255$ Lanczos-3 ringing lobes unclipped
/// between the horizontal and vertical passes.
#[inline]
pub fn hscale_row_8tap_q14_to_q6(src_row: &[u8], dst_row: &mut [i16], bank: &PolyphaseFilterBank) {
    #[cfg(target_arch = "aarch64")]
    {
        if src_row.len() >= 8 {
            unsafe { hscale_row_8tap_q14_to_q6_neon(src_row, dst_row, bank) };
            return;
        }
    }
    hscale_row_8tap_q14_to_q6_scalar(src_row, dst_row, bank);
}

#[allow(clippy::needless_range_loop)]
pub fn hscale_row_8tap_q14_to_q6_scalar(
    src_row: &[u8],
    dst_row: &mut [i16],
    bank: &PolyphaseFilterBank,
) {
    let dst_len = bank.dst_len;
    if src_row.len() >= 8 {
        for x in 0..dst_len {
            let ws = bank.win_starts[x];
            let w = &bank.weights[x];
            let mut acc = 128i32; // rounding bias for >> 8 (Q14 -> Q6)
            for k in 0..8 {
                acc += (src_row[ws + k] as i32) * (w[k] as i32);
            }
            dst_row[x] = (acc >> 8).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }
    } else {
        let mut padded = [0u8; 8];
        padded[..src_row.len()].copy_from_slice(src_row);
        for x in 0..dst_len {
            let w = &bank.weights[x];
            let mut acc = 128i32;
            for k in 0..8 {
                acc += (padded[k] as i32) * (w[k] as i32);
            }
            dst_row[x] = (acc >> 8).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn hscale_row_8tap_q14_to_q6_neon(
    src_row: &[u8],
    dst_row: &mut [i16],
    bank: &PolyphaseFilterBank,
) {
    use std::arch::aarch64::*;
    let dst_len = bank.dst_len;
    assert!(dst_row.len() >= dst_len);
    assert!(bank.win_starts.len() >= dst_len && bank.weights.len() >= dst_len);

    let src_ptr = src_row.as_ptr();
    let dst_ptr = dst_row.as_mut_ptr();
    let starts_ptr = bank.win_starts.as_ptr();
    let weights_ptr = bank.weights.as_ptr();
    let mut x = 0usize;

    let eval4 = |base: usize| unsafe {
        let s0 = vld1_u8(src_ptr.add(*starts_ptr.add(base)));
        let s1 = vld1_u8(src_ptr.add(*starts_ptr.add(base + 1)));
        let s2 = vld1_u8(src_ptr.add(*starts_ptr.add(base + 2)));
        let s3 = vld1_u8(src_ptr.add(*starts_ptr.add(base + 3)));

        let w0 = vld1q_s16((*weights_ptr.add(base)).as_ptr());
        let w1 = vld1q_s16((*weights_ptr.add(base + 1)).as_ptr());
        let w2 = vld1q_s16((*weights_ptr.add(base + 2)).as_ptr());
        let w3 = vld1q_s16((*weights_ptr.add(base + 3)).as_ptr());

        let s0_16 = vreinterpretq_s16_u16(vmovl_u8(s0));
        let s1_16 = vreinterpretq_s16_u16(vmovl_u8(s1));
        let s2_16 = vreinterpretq_s16_u16(vmovl_u8(s2));
        let s3_16 = vreinterpretq_s16_u16(vmovl_u8(s3));

        let acc0 = vmlal_high_s16(vmull_s16(vget_low_s16(s0_16), vget_low_s16(w0)), s0_16, w0);
        let acc1 = vmlal_high_s16(vmull_s16(vget_low_s16(s1_16), vget_low_s16(w1)), s1_16, w1);
        let acc2 = vmlal_high_s16(vmull_s16(vget_low_s16(s2_16), vget_low_s16(w2)), s2_16, w2);
        let acc3 = vmlal_high_s16(vmull_s16(vget_low_s16(s3_16), vget_low_s16(w3)), s3_16, w3);

        let p01 = vpaddq_s32(acc0, acc1);
        let p23 = vpaddq_s32(acc2, acc3);
        vpaddq_s32(p01, p23)
    };

    while x + 8 <= dst_len {
        let sum_lo = eval4(x);
        let sum_hi = eval4(x + 4);
        let q6_8 = vqrshrn_high_n_s32::<8>(vqrshrn_n_s32::<8>(sum_lo), sum_hi);
        unsafe {
            vst1q_s16(dst_ptr.add(x), q6_8);
        }
        x += 8;
    }

    while x + 4 <= dst_len {
        let sum4 = eval4(x);
        let q6_4 = vqrshrn_n_s32::<8>(sum4);
        unsafe {
            vst1_s16(dst_ptr.add(x), q6_4);
        }
        x += 4;
    }

    while x < dst_len {
        let ws = bank.win_starts[x];
        let w = &bank.weights[x];
        let mut acc = 128i32;
        for k in 0..8 {
            acc += (src_row[ws + k] as i32) * (w[k] as i32);
        }
        dst_row[x] = (acc >> 8).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        x += 1;
    }
}

/// Vertically convolves 8 `Q6` (`i16`) intermediate ring-buffer rows with `Q14` weights into `u8`.
///
/// Uses `libswscale/aarch64/output.S`'s two-stage narrowing shift (`vqshrun_n_s32::<16>` + `vqshrn_n_u16::<4>`)
/// to shift right by `20` (`Q6 * Q14 = Q20`) and saturate to `0..=255` in two instructions.
#[inline]
pub fn vscale_row_8tap_q6_to_u8(
    rows: [&[i16]; 8],
    weights: &[i16; 8],
    dst_row: &mut [u8],
    width: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { vscale_row_8tap_q6_to_u8_neon(rows, weights, dst_row, width) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        vscale_row_8tap_q6_to_u8_scalar(rows, weights, dst_row, width)
    }
}

pub fn vscale_row_8tap_q6_to_u8_scalar(
    rows: [&[i16]; 8],
    weights: &[i16; 8],
    dst_row: &mut [u8],
    width: usize,
) {
    const Q20_HALF: i32 = 1 << 19;
    for x in 0..width {
        let mut acc = Q20_HALF;
        for t in 0..8 {
            acc += (rows[t][x] as i32) * (weights[t] as i32);
        }
        dst_row[x] = clamp_u8(acc >> 20);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn vscale_row_8tap_q6_to_u8_neon(
    rows: [&[i16]; 8],
    weights: &[i16; 8],
    dst_row: &mut [u8],
    width: usize,
) {
    use std::arch::aarch64::*;
    const Q20_HALF: i32 = 1 << 19;
    let v_bias = vdupq_n_s32(Q20_HALF);
    let mut x = 0usize;

    // Process 16 horizontal pixels per iteration directly from `i16` ring-buffer rows
    while x + 16 <= width {
        let mut acc0 = v_bias;
        let mut acc1 = v_bias;
        let mut acc2 = v_bias;
        let mut acc3 = v_bias;

        for t in 0..8 {
            let w = weights[t];
            if w == 0 {
                continue;
            }
            let r_lo = unsafe { vld1q_s16(rows[t].as_ptr().add(x)) };
            let r_hi = unsafe { vld1q_s16(rows[t].as_ptr().add(x + 8)) };

            acc0 = vmlal_n_s16(acc0, vget_low_s16(r_lo), w);
            acc1 = vmlal_high_n_s16(acc1, r_lo, w);
            acc2 = vmlal_n_s16(acc2, vget_low_s16(r_hi), w);
            acc3 = vmlal_high_n_s16(acc3, r_hi, w);
        }

        // Two-stage narrowing shift >> 20 (`>> 16` via `vqshrun_n_s32::<16>` + `>> 4` via `vqshrn_n_u16::<4>`)
        let u01 = vcombine_u16(vqshrun_n_s32::<16>(acc0), vqshrun_n_s32::<16>(acc1));
        let u23 = vcombine_u16(vqshrun_n_s32::<16>(acc2), vqshrun_n_s32::<16>(acc3));
        let out8 = vcombine_u8(vqshrn_n_u16::<4>(u01), vqshrn_n_u16::<4>(u23));
        unsafe {
            vst1q_u8(dst_row.as_mut_ptr().add(x), out8);
        }
        x += 16;
    }

    while x < width {
        let mut acc = Q20_HALF;
        for t in 0..8 {
            acc += (rows[t][x] as i32) * (weights[t] as i32);
        }
        dst_row[x] = clamp_u8(acc >> 20);
        x += 1;
    }
}

/// Scales a 2D `Plane` into `dst` using a **fused 8-row 14-bit (`Q6`) L1-cache-resident rolling scanline ring buffer**.
///
/// Each input row is horizontally filtered at most once into the 8-row `i16` (`Q6`) ring buffer, and vertical
/// NEON dot products immediately emit finished output rows into `dst` with zero full-frame intermediate buffers.
pub fn scale_plane_with_banks(
    src: &Plane,
    dst: &mut Plane,
    h_bank: &PolyphaseFilterBank,
    v_bank: &PolyphaseFilterBank,
) {
    assert_eq!(src.width(), h_bank.src_len);
    assert_eq!(dst.width(), h_bank.dst_len);
    assert_eq!(src.height(), v_bank.src_len);
    assert_eq!(dst.height(), v_bank.dst_len);

    let dst_w = dst.width();
    let dst_h = dst.height();
    let src_h = src.height();

    // 8-slot 14-bit (`Q6` `i16`) rolling scanline ring buffer resident in L1 cache (~30 KB for 1080p)
    let ring_stride = (dst_w + 31) & !31;
    let mut ring_buf = vec![0i16; ring_stride * 8];
    let mut cached_src_y = [usize::MAX; 8];

    for dy in 0..dst_h {
        let win_start = v_bank.win_starts[dy];
        let v_weights = &v_bank.weights[dy];

        // Ensure all 8 required source rows [win_start .. win_start + 8] are horizontally scaled in the ring buffer
        for k in 0..8 {
            let sy = (win_start + k).min(src_h - 1);
            let slot = sy & 7;
            if cached_src_y[slot] != sy {
                let src_row = src.row_padded(sy);
                let ring_row = &mut ring_buf[slot * ring_stride..slot * ring_stride + dst_w];
                hscale_row_8tap_q14_to_q6(src_row, ring_row, h_bank);
                cached_src_y[slot] = sy;
            }
        }

        let s0 = (win_start.min(src_h - 1)) & 7;
        let s1 = ((win_start + 1).min(src_h - 1)) & 7;
        let s2 = ((win_start + 2).min(src_h - 1)) & 7;
        let s3 = ((win_start + 3).min(src_h - 1)) & 7;
        let s4 = ((win_start + 4).min(src_h - 1)) & 7;
        let s5 = ((win_start + 5).min(src_h - 1)) & 7;
        let s6 = ((win_start + 6).min(src_h - 1)) & 7;
        let s7 = ((win_start + 7).min(src_h - 1)) & 7;

        let rows: [&[i16]; 8] = [
            &ring_buf[s0 * ring_stride..s0 * ring_stride + dst_w],
            &ring_buf[s1 * ring_stride..s1 * ring_stride + dst_w],
            &ring_buf[s2 * ring_stride..s2 * ring_stride + dst_w],
            &ring_buf[s3 * ring_stride..s3 * ring_stride + dst_w],
            &ring_buf[s4 * ring_stride..s4 * ring_stride + dst_w],
            &ring_buf[s5 * ring_stride..s5 * ring_stride + dst_w],
            &ring_buf[s6 * ring_stride..s6 * ring_stride + dst_w],
            &ring_buf[s7 * ring_stride..s7 * ring_stride + dst_w],
        ];

        vscale_row_8tap_q6_to_u8(rows, v_weights, dst.row_mut(dy), dst_w);
    }
}

/// Scales a `Plane` into `dst` using the specified `ScaleFilter`.
pub fn scale_plane(src: &Plane, dst: &mut Plane, filter: ScaleFilter) {
    let h_bank = PolyphaseFilterBank::new(src.width(), dst.width(), filter);
    let v_bank = PolyphaseFilterBank::new(src.height(), dst.height(), filter);
    scale_plane_with_banks(src, dst, &h_bank, &v_bank);
}

/// Scales a planar `VideoFrame` (`YUV420p`, `YUV422p`, `YUV444p`, `Gray8`) to `dst_width x dst_height`.
pub fn scale_frame(
    src: &VideoFrame,
    dst_width: usize,
    dst_height: usize,
    filter: ScaleFilter,
) -> VideoFrame {
    let mut dst = VideoFrame::new(dst_width, dst_height, src.format);
    dst.colorimetry = src.colorimetry;
    dst.pts = src.pts;
    dst.duration = src.duration;
    dst.keyframe = src.keyframe;
    dst.pict_type = src.pict_type;
    scale_frame_into(src, &mut dst, filter);
    dst
}

/// Scales `src` into pre-allocated `dst` across all planes using L1 rolling scanline ring buffers.
pub fn scale_frame_into(src: &VideoFrame, dst: &mut VideoFrame, filter: ScaleFilter) {
    assert_eq!(src.format, dst.format);
    for (sp, dp) in src.planes.iter().zip(dst.planes.iter_mut()) {
        scale_plane(sp, dp, filter);
    }
}
