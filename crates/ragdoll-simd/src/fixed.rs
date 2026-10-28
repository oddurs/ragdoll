/// Q14 fixed-point scale factor ($2^{14} = 16\,384$).
pub const Q14_ONE: i32 = 1 << 14;
/// Q14 rounding bias ($2^{13} = 8\,192$).
pub const Q14_HALF: i32 = 1 << 13;

/// Clamps an `i32` into `0..=255` branchlessly.
#[inline(always)]
pub const fn clamp_u8(v: i32) -> u8 {
    if v < 0 {
        0
    } else if v > 255 {
        255
    } else {
        v as u8
    }
}

/// Convolves a horizontal row of `u8` samples with an 8-tap Q14 fixed-point filter bank
/// (`weights` length = 8, sum = 16384), returning a clamped `u8` pixel.
#[inline(always)]
pub fn convolve_8tap_q14_u8(samples: &[u8; 8], weights: &[i16; 8]) -> u8 {
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        return unsafe { convolve_8tap_q14_u8_neon(samples, weights) };
    }
    convolve_8tap_q14_u8_scalar(samples, weights)
}

#[inline(always)]
pub fn convolve_8tap_q14_u8_scalar(samples: &[u8; 8], weights: &[i16; 8]) -> u8 {
    let mut acc = Q14_HALF;
    for i in 0..8 {
        acc += (samples[i] as i32) * (weights[i] as i32);
    }
    clamp_u8(acc >> 14)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn convolve_8tap_q14_u8_neon(samples: &[u8; 8], weights: &[i16; 8]) -> u8 {
    use std::arch::aarch64::*;
    let s8 = unsafe { vld1_u8(samples.as_ptr()) };
    let s16 = vreinterpretq_s16_u16(vmovl_u8(s8));
    let w16 = unsafe { vld1q_s16(weights.as_ptr()) };

    let prod_lo = vmull_s16(vget_low_s16(s16), vget_low_s16(w16));
    let prod = vmlal_high_s16(prod_lo, s16, w16);
    let sum = vaddvq_s32(prod) + Q14_HALF;
    clamp_u8(sum >> 14)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn transpose_4x4_s16_neon(
    r0: std::arch::aarch64::int16x4_t,
    r1: std::arch::aarch64::int16x4_t,
    r2: std::arch::aarch64::int16x4_t,
    r3: std::arch::aarch64::int16x4_t,
) -> (
    std::arch::aarch64::int16x4_t,
    std::arch::aarch64::int16x4_t,
    std::arch::aarch64::int16x4_t,
    std::arch::aarch64::int16x4_t,
) {
    use std::arch::aarch64::*;
    let t01_lo = vtrn1_s16(r0, r1);
    let t01_hi = vtrn2_s16(r0, r1);
    let t23_lo = vtrn1_s16(r2, r3);
    let t23_hi = vtrn2_s16(r2, r3);

    let c0 = vreinterpret_s16_s32(vtrn1_s32(
        vreinterpret_s32_s16(t01_lo),
        vreinterpret_s32_s16(t23_lo),
    ));
    let c1 = vreinterpret_s16_s32(vtrn1_s32(
        vreinterpret_s32_s16(t01_hi),
        vreinterpret_s32_s16(t23_hi),
    ));
    let c2 = vreinterpret_s16_s32(vtrn2_s32(
        vreinterpret_s32_s16(t01_lo),
        vreinterpret_s32_s16(t23_lo),
    ));
    let c3 = vreinterpret_s16_s32(vtrn2_s32(
        vreinterpret_s32_s16(t01_hi),
        vreinterpret_s32_s16(t23_hi),
    ));
    (c0, c1, c2, c3)
}

/// Exact H.264 / AVC forward $4\times4$ integer DCT transform on a $4\times4$ residual block.
///
/// Dispatches to an in-register `vtrn1`/`vtrn2` NEON kernel on `aarch64` (modeled after `x264/common/aarch64/dct-a.S`).
#[inline]
pub fn forward_dct4x4(block: &[i16; 16]) -> [i16; 16] {
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        return unsafe { forward_dct4x4_neon(block) };
    }
    forward_dct4x4_scalar(block)
}

pub fn forward_dct4x4_scalar(block: &[i16; 16]) -> [i16; 16] {
    let mut tmp = [0i32; 16];
    // 1. Horizontal 1D butterfly
    for r in 0..4 {
        let i = r * 4;
        let s03 = (block[i] as i32) + (block[i + 3] as i32);
        let s12 = (block[i + 1] as i32) + (block[i + 2] as i32);
        let d03 = (block[i] as i32) - (block[i + 3] as i32);
        let d12 = (block[i + 1] as i32) - (block[i + 2] as i32);

        tmp[i] = s03 + s12;
        tmp[i + 1] = (d03 << 1) + d12;
        tmp[i + 2] = s03 - s12;
        tmp[i + 3] = d03 - (d12 << 1);
    }

    // 2. Vertical 1D butterfly
    let mut out = [0i16; 16];
    for c in 0..4 {
        let s03 = tmp[c] + tmp[c + 12];
        let s12 = tmp[c + 4] + tmp[c + 8];
        let d03 = tmp[c] - tmp[c + 12];
        let d12 = tmp[c + 4] - tmp[c + 8];

        out[c] = (s03 + s12) as i16;
        out[c + 4] = ((d03 << 1) + d12) as i16;
        out[c + 8] = (s03 - s12) as i16;
        out[c + 12] = (d03 - (d12 << 1)) as i16;
    }
    out
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn forward_dct4x4_neon(block: &[i16; 16]) -> [i16; 16] {
    use std::arch::aarch64::*;
    let r0 = unsafe { vld1_s16(block.as_ptr()) };
    let r1 = unsafe { vld1_s16(block.as_ptr().add(4)) };
    let r2 = unsafe { vld1_s16(block.as_ptr().add(8)) };
    let r3 = unsafe { vld1_s16(block.as_ptr().add(12)) };

    // 1. Vertical butterfly across rows r0..r3
    let s03 = vadd_s16(r0, r3);
    let s12 = vadd_s16(r1, r2);
    let d03 = vsub_s16(r0, r3);
    let d12 = vsub_s16(r1, r2);

    let v0 = vadd_s16(s03, s12);
    let v1 = vadd_s16(vshl_n_s16::<1>(d03), d12);
    let v2 = vsub_s16(s03, s12);
    let v3 = vsub_s16(d03, vshl_n_s16::<1>(d12));

    // 2. Transpose 4x4 in-register
    let (c0, c1, c2, c3) = unsafe { transpose_4x4_s16_neon(v0, v1, v2, v3) };

    // 3. Horizontal butterfly across columns c0..c3
    let hs03 = vadd_s16(c0, c3);
    let hs12 = vadd_s16(c1, c2);
    let hd03 = vsub_s16(c0, c3);
    let hd12 = vsub_s16(c1, c2);

    let h0 = vadd_s16(hs03, hs12);
    let h1 = vadd_s16(vshl_n_s16::<1>(hd03), hd12);
    let h2 = vsub_s16(hs03, hs12);
    let h3 = vsub_s16(hd03, vshl_n_s16::<1>(hd12));

    // 4. Transpose back to row-major output
    let (o0, o1, o2, o3) = unsafe { transpose_4x4_s16_neon(h0, h1, h2, h3) };
    let mut out = [0i16; 16];
    unsafe {
        vst1_s16(out.as_mut_ptr(), o0);
        vst1_s16(out.as_mut_ptr().add(4), o1);
        vst1_s16(out.as_mut_ptr().add(8), o2);
        vst1_s16(out.as_mut_ptr().add(12), o3);
    }
    out
}

/// Exact H.264 / AVC inverse $4\times4$ integer IDCT transform (with $+32 \gg 6$ normalization).
#[inline]
pub fn inverse_idct4x4(coeff: &[i16; 16]) -> [i16; 16] {
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        return unsafe { inverse_idct4x4_neon(coeff) };
    }
    inverse_idct4x4_scalar(coeff)
}

pub fn inverse_idct4x4_scalar(coeff: &[i16; 16]) -> [i16; 16] {
    let mut tmp = [0i32; 16];
    // 1. Horizontal 1D inverse butterfly
    for r in 0..4 {
        let i = r * 4;
        let e0 = (coeff[i] as i32) + (coeff[i + 2] as i32);
        let e1 = (coeff[i] as i32) - (coeff[i + 2] as i32);
        let e2 = ((coeff[i + 1] as i32) >> 1) - (coeff[i + 3] as i32);
        let e3 = (coeff[i + 1] as i32) + ((coeff[i + 3] as i32) >> 1);

        tmp[i] = e0 + e3;
        tmp[i + 1] = e1 + e2;
        tmp[i + 2] = e1 - e2;
        tmp[i + 3] = e0 - e3;
    }

    // 2. Vertical 1D inverse butterfly + rounding shift >> 6
    let mut out = [0i16; 16];
    for c in 0..4 {
        let e0 = tmp[c] + tmp[c + 8];
        let e1 = tmp[c] - tmp[c + 8];
        let e2 = (tmp[c + 4] >> 1) - tmp[c + 12];
        let e3 = tmp[c + 4] + (tmp[c + 12] >> 1);

        out[c] = ((e0 + e3 + 32) >> 6) as i16;
        out[c + 4] = ((e1 + e2 + 32) >> 6) as i16;
        out[c + 8] = ((e1 - e2 + 32) >> 6) as i16;
        out[c + 12] = ((e0 - e3 + 32) >> 6) as i16;
    }
    out
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn inverse_idct4x4_neon(coeff: &[i16; 16]) -> [i16; 16] {
    use std::arch::aarch64::*;
    let r0 = unsafe { vld1_s16(coeff.as_ptr()) };
    let r1 = unsafe { vld1_s16(coeff.as_ptr().add(4)) };
    let r2 = unsafe { vld1_s16(coeff.as_ptr().add(8)) };
    let r3 = unsafe { vld1_s16(coeff.as_ptr().add(12)) };

    // 1. Transpose rows to columns first so horizontal butterfly runs parallel across 4 rows
    let (c0, c1, c2, c3) = unsafe { transpose_4x4_s16_neon(r0, r1, r2, r3) };

    let e0 = vadd_s16(c0, c2);
    let e1 = vsub_s16(c0, c2);
    let e2 = vsub_s16(vshr_n_s16::<1>(c1), c3);
    let e3 = vadd_s16(c1, vshr_n_s16::<1>(c3));

    let t0 = vadd_s16(e0, e3);
    let t1 = vadd_s16(e1, e2);
    let t2 = vsub_s16(e1, e2);
    let t3 = vsub_s16(e0, e3);

    // 2. Transpose back to rows for vertical butterfly
    let (v0, v1, v2, v3) = unsafe { transpose_4x4_s16_neon(t0, t1, t2, t3) };

    let ve0 = vadd_s16(v0, v2);
    let ve1 = vsub_s16(v0, v2);
    let ve2 = vsub_s16(vshr_n_s16::<1>(v1), v3);
    let ve3 = vadd_s16(v1, vshr_n_s16::<1>(v3));

    // Hardware rounding right shift `vrshr_n_s16::<6>` computes `(x + 32) >> 6` in 1 cycle!
    let o0 = vrshr_n_s16::<6>(vadd_s16(ve0, ve3));
    let o1 = vrshr_n_s16::<6>(vadd_s16(ve1, ve2));
    let o2 = vrshr_n_s16::<6>(vsub_s16(ve1, ve2));
    let o3 = vrshr_n_s16::<6>(vsub_s16(ve0, ve3));

    let mut out = [0i16; 16];
    unsafe {
        vst1_s16(out.as_mut_ptr(), o0);
        vst1_s16(out.as_mut_ptr().add(4), o1);
        vst1_s16(out.as_mut_ptr().add(8), o2);
        vst1_s16(out.as_mut_ptr().add(12), o3);
    }
    out
}

/// Fused $4\times4$ residual subtraction (`src - pred`) and forward $4\times4$ DCT (`x264_sub4x4_dct`).
#[inline]
pub fn sub4x4_dct_u8(src: &[u8], src_stride: usize, pred: &[u8], pred_stride: usize) -> [i16; 16] {
    let mut diff = [0i16; 16];
    for y in 0..4 {
        let rs = &src[y * src_stride..y * src_stride + 4];
        let rp = &pred[y * pred_stride..y * pred_stride + 4];
        for x in 0..4 {
            diff[y * 4 + x] = (rs[x] as i16) - (rp[x] as i16);
        }
    }
    forward_dct4x4(&diff)
}

/// Fused inverse $4\times4$ IDCT and in-place prediction reconstruction (`dst = clamp_u8(dst + idct(coeff))`).
///
/// Includes `x264`'s DC-only fast path when all AC coefficients `coeff[1..16]` are zero.
#[inline]
pub fn add4x4_idct_u8(dst: &mut [u8], dst_stride: usize, coeff: &[i16; 16]) {
    if coeff[1..].iter().all(|&c| c == 0) {
        let dc = ((coeff[0] as i32) + 32) >> 6;
        if dc == 0 {
            return;
        }
        for y in 0..4 {
            let row = &mut dst[y * dst_stride..y * dst_stride + 4];
            for p in row.iter_mut() {
                *p = clamp_u8((*p as i32) + dc);
            }
        }
        return;
    }

    let res = inverse_idct4x4(coeff);
    for y in 0..4 {
        let row = &mut dst[y * dst_stride..y * dst_stride + 4];
        for x in 0..4 {
            row[x] = clamp_u8((row[x] as i32) + (res[y * 4 + x] as i32));
        }
    }
}

/// Exact H.264 $4\times4$ Luma DC Hadamard transform.
#[inline]
pub fn hadamard4x4(block: &[i16; 16]) -> [i16; 16] {
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        return unsafe { hadamard4x4_neon(block) };
    }
    hadamard4x4_scalar(block)
}

pub fn hadamard4x4_scalar(block: &[i16; 16]) -> [i16; 16] {
    let mut tmp = [0i32; 16];
    for r in 0..4 {
        let i = r * 4;
        let s03 = (block[i] as i32) + (block[i + 3] as i32);
        let s12 = (block[i + 1] as i32) + (block[i + 2] as i32);
        let d03 = (block[i] as i32) - (block[i + 3] as i32);
        let d12 = (block[i + 1] as i32) - (block[i + 2] as i32);
        tmp[i] = s03 + s12;
        tmp[i + 1] = d03 + d12;
        tmp[i + 2] = s03 - s12;
        tmp[i + 3] = d03 - d12;
    }
    let mut out = [0i16; 16];
    for c in 0..4 {
        let s03 = tmp[c] + tmp[c + 12];
        let s12 = tmp[c + 4] + tmp[c + 8];
        let d03 = tmp[c] - tmp[c + 12];
        let d12 = tmp[c + 4] - tmp[c + 8];
        out[c] = ((s03 + s12) >> 1) as i16;
        out[c + 4] = ((d03 + d12) >> 1) as i16;
        out[c + 8] = ((s03 - s12) >> 1) as i16;
        out[c + 12] = ((d03 - d12) >> 1) as i16;
    }
    out
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn hadamard4x4_neon(block: &[i16; 16]) -> [i16; 16] {
    use std::arch::aarch64::*;
    let r0 = unsafe { vld1_s16(block.as_ptr()) };
    let r1 = unsafe { vld1_s16(block.as_ptr().add(4)) };
    let r2 = unsafe { vld1_s16(block.as_ptr().add(8)) };
    let r3 = unsafe { vld1_s16(block.as_ptr().add(12)) };

    let s03 = vadd_s16(r0, r3);
    let s12 = vadd_s16(r1, r2);
    let d03 = vsub_s16(r0, r3);
    let d12 = vsub_s16(r1, r2);

    let v0 = vadd_s16(s03, s12);
    let v1 = vadd_s16(d03, d12);
    let v2 = vsub_s16(s03, s12);
    let v3 = vsub_s16(d03, d12);

    let (c0, c1, c2, c3) = unsafe { transpose_4x4_s16_neon(v0, v1, v2, v3) };

    let hs03 = vadd_s16(c0, c3);
    let hs12 = vadd_s16(c1, c2);
    let hd03 = vsub_s16(c0, c3);
    let hd12 = vsub_s16(c1, c2);

    let h0 = vshr_n_s16::<1>(vadd_s16(hs03, hs12));
    let h1 = vshr_n_s16::<1>(vadd_s16(hd03, hd12));
    let h2 = vshr_n_s16::<1>(vsub_s16(hs03, hs12));
    let h3 = vshr_n_s16::<1>(vsub_s16(hd03, hd12));

    let (o0, o1, o2, o3) = unsafe { transpose_4x4_s16_neon(h0, h1, h2, h3) };
    let mut out = [0i16; 16];
    unsafe {
        vst1_s16(out.as_mut_ptr(), o0);
        vst1_s16(out.as_mut_ptr().add(4), o1);
        vst1_s16(out.as_mut_ptr().add(8), o2);
        vst1_s16(out.as_mut_ptr().add(12), o3);
    }
    out
}

/// Exact H.264 High Profile $8\times8$ forward integer DCT transform (`transform_8x8_mode_flag = 1`).
pub fn forward_dct8x8(block: &[i16; 64]) -> [i16; 64] {
    let mut tmp = [0i32; 64];
    for r in 0..8 {
        let i = r * 8;
        let a0 = (block[i] as i32) + (block[i + 7] as i32);
        let a1 = (block[i + 1] as i32) + (block[i + 6] as i32);
        let a2 = (block[i + 2] as i32) + (block[i + 5] as i32);
        let a3 = (block[i + 3] as i32) + (block[i + 4] as i32);
        let a4 = (block[i] as i32) - (block[i + 7] as i32);
        let a5 = (block[i + 1] as i32) - (block[i + 6] as i32);
        let a6 = (block[i + 2] as i32) - (block[i + 5] as i32);
        let a7 = (block[i + 3] as i32) - (block[i + 4] as i32);

        let b0 = a0 + a3;
        let b1 = a1 + a2;
        let b2 = a0 - a3;
        let b3 = a1 - a2;
        let b4 = a5 + a6 + ((a4 >> 1) + a4);
        let b5 = a4 - a7 - ((a6 >> 1) + a6);
        let b6 = a4 + a7 - ((a5 >> 1) + a5);
        let b7 = a5 - a6 + ((a7 >> 1) + a7);

        tmp[i] = b0 + b1;
        tmp[i + 1] = b4 + (b7 >> 2);
        tmp[i + 2] = b2 + (b3 >> 1);
        tmp[i + 3] = b5 + (b6 >> 2);
        tmp[i + 4] = b0 - b1;
        tmp[i + 5] = b6 - (b5 >> 2);
        tmp[i + 6] = (b2 >> 1) - b3;
        tmp[i + 7] = (b4 >> 2) - b7;
    }

    let mut out = [0i16; 64];
    for c in 0..8 {
        let a0 = tmp[c] + tmp[c + 56];
        let a1 = tmp[c + 8] + tmp[c + 48];
        let a2 = tmp[c + 16] + tmp[c + 40];
        let a3 = tmp[c + 24] + tmp[c + 32];
        let a4 = tmp[c] - tmp[c + 56];
        let a5 = tmp[c + 8] - tmp[c + 48];
        let a6 = tmp[c + 16] - tmp[c + 40];
        let a7 = tmp[c + 24] - tmp[c + 32];

        let b0 = a0 + a3;
        let b1 = a1 + a2;
        let b2 = a0 - a3;
        let b3 = a1 - a2;
        let b4 = a5 + a6 + ((a4 >> 1) + a4);
        let b5 = a4 - a7 - ((a6 >> 1) + a6);
        let b6 = a4 + a7 - ((a5 >> 1) + a5);
        let b7 = a5 - a6 + ((a7 >> 1) + a7);

        out[c] = (b0 + b1) as i16;
        out[c + 8] = (b4 + (b7 >> 2)) as i16;
        out[c + 16] = (b2 + (b3 >> 1)) as i16;
        out[c + 24] = (b5 + (b6 >> 2)) as i16;
        out[c + 32] = (b0 - b1) as i16;
        out[c + 40] = (b6 - (b5 >> 2)) as i16;
        out[c + 48] = ((b2 >> 1) - b3) as i16;
        out[c + 56] = ((b4 >> 2) - b7) as i16;
    }
    out
}

/// Exact H.264 High Profile $8\times8$ inverse integer IDCT transform (with $+32 \gg 6$ normalization).
pub fn inverse_idct8x8(coeff: &[i16; 64]) -> [i16; 64] {
    let mut tmp = [0i32; 64];
    for r in 0..8 {
        let i = r * 8;
        let c0 = coeff[i] as i32;
        let c1 = coeff[i + 1] as i32;
        let c2 = coeff[i + 2] as i32;
        let c3 = coeff[i + 3] as i32;
        let c4 = coeff[i + 4] as i32;
        let c5 = coeff[i + 5] as i32;
        let c6 = coeff[i + 6] as i32;
        let c7 = coeff[i + 7] as i32;

        let a0 = c0 + c4;
        let a2 = c0 - c4;
        let a4 = (c2 >> 1) - c6;
        let a6 = c2 + (c6 >> 1);

        let b0 = a0 + a6;
        let b2 = a2 + a4;
        let b4 = a2 - a4;
        let b6 = a0 - a6;

        let a1 = -c3 + c5 - c7 - (c7 >> 1);
        let a3 = c1 + c7 - c3 - (c3 >> 1);
        let a5 = -c1 + c7 + c5 + (c5 >> 1);
        let a7 = c3 + c5 + c1 + (c1 >> 1);

        let b1 = a1 + (a7 >> 2);
        let b3 = a3 + (a5 >> 2);
        let b5 = (a3 >> 2) - a5;
        let b7 = a7 - (a1 >> 2);

        tmp[i] = b0 + b7;
        tmp[i + 1] = b2 + b5;
        tmp[i + 2] = b4 + b3;
        tmp[i + 3] = b6 + b1;
        tmp[i + 4] = b6 - b1;
        tmp[i + 5] = b4 - b3;
        tmp[i + 6] = b2 - b5;
        tmp[i + 7] = b0 - b7;
    }

    let mut out = [0i16; 64];
    for c in 0..8 {
        let c0 = tmp[c];
        let c1 = tmp[c + 8];
        let c2 = tmp[c + 16];
        let c3 = tmp[c + 24];
        let c4 = tmp[c + 32];
        let c5 = tmp[c + 40];
        let c6 = tmp[c + 48];
        let c7 = tmp[c + 56];

        let a0 = c0 + c4;
        let a2 = c0 - c4;
        let a4 = (c2 >> 1) - c6;
        let a6 = c2 + (c6 >> 1);

        let b0 = a0 + a6;
        let b2 = a2 + a4;
        let b4 = a2 - a4;
        let b6 = a0 - a6;

        let a1 = -c3 + c5 - c7 - (c7 >> 1);
        let a3 = c1 + c7 - c3 - (c3 >> 1);
        let a5 = -c1 + c7 + c5 + (c5 >> 1);
        let a7 = c3 + c5 + c1 + (c1 >> 1);

        let b1 = a1 + (a7 >> 2);
        let b3 = a3 + (a5 >> 2);
        let b5 = (a3 >> 2) - a5;
        let b7 = a7 - (a1 >> 2);

        out[c] = ((b0 + b7 + 32) >> 6) as i16;
        out[c + 8] = ((b2 + b5 + 32) >> 6) as i16;
        out[c + 16] = ((b4 + b3 + 32) >> 6) as i16;
        out[c + 24] = ((b6 + b1 + 32) >> 6) as i16;
        out[c + 32] = ((b6 - b1 + 32) >> 6) as i16;
        out[c + 40] = ((b4 - b3 + 32) >> 6) as i16;
        out[c + 48] = ((b2 - b5 + 32) >> 6) as i16;
        out[c + 56] = ((b0 - b7 + 32) >> 6) as i16;
    }
    out
}
