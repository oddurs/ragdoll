//! CELT signal-domain helpers (libopus `pitch.c`, `celt_lpc.c`, `celt.c`): pitch search and LPC
//! for packet-loss concealment, and the pitch pre/post-filter.

use super::math::{
    ilog2, inner_prod, mac16, maxabs16, maxabs32, mul16, mul16_32_q15, mul16_p15, mul16_q15,
    mul32_q31, pshr32, round16, saturate, shl32, sround16, vshr32,
};
use super::range::ilog;

pub const SIG_SHIFT: i32 = 12;
pub const SIG_SAT: i32 = 300000000;
pub const LPC_ORDER: usize = 24;
pub const COMBFILTER_MINPERIOD: i32 = 15;

/// Plain correlation `xcorr[i] = sum_j x[j] * y[i + j]` (`celt_pitch_xcorr`); returns the
/// maximum, at least 1. Integer sums are exact, so libopus's unrolling does not matter.
pub fn pitch_xcorr(x: &[i16], y: &[i16], xcorr: &mut [i32], len: usize, max_pitch: usize) -> i32 {
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        return unsafe { super::simd_enc::pitch_xcorr_avx2(x, y, xcorr, len, max_pitch) };
    }
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: NEON is baseline on aarch64.
        return unsafe { super::simd_enc::pitch_xcorr_neon(x, y, xcorr, len, max_pitch) };
    }
    pitch_xcorr_scalar(x, y, xcorr, len, max_pitch)
}

/// Scalar reference of [`pitch_xcorr`].
pub fn pitch_xcorr_scalar(
    x: &[i16],
    y: &[i16],
    xcorr: &mut [i32],
    len: usize,
    max_pitch: usize,
) -> i32 {
    let mut maxcorr = 1i32;
    for i in 0..max_pitch {
        let mut sum = 0i32;
        for j in 0..len {
            sum = mac16(sum, x[j] as i32, y[i + j] as i32);
        }
        xcorr[i] = sum;
        maxcorr = maxcorr.max(sum);
    }
    maxcorr
}

#[allow(clippy::too_many_arguments)]
fn find_best_pitch(
    xcorr: &[i32],
    y: &[i16],
    len: usize,
    max_pitch: usize,
    best_pitch: &mut [usize; 2],
    yshift: i32,
    maxcorr: i32,
) {
    let mut syy = 1i32;
    let mut best_num = [-1i32; 2];
    let mut best_den = [0i32; 2];
    let xshift = ilog2(maxcorr) - 14;
    *best_pitch = [0, 1];
    for &v in &y[..len] {
        syy = syy.wrapping_add(mul16(v as i32, v as i32) >> yshift);
    }
    for i in 0..max_pitch {
        if xcorr[i] > 0 {
            let xcorr16 = vshr32(xcorr[i], xshift) as i16 as i32;
            let num = mul16_q15(xcorr16, xcorr16) as i16 as i32;
            if mul16_32_q15(num, best_den[1]) > mul16_32_q15(best_num[1], syy) {
                if mul16_32_q15(num, best_den[0]) > mul16_32_q15(best_num[0], syy) {
                    best_num[1] = best_num[0];
                    best_den[1] = best_den[0];
                    best_pitch[1] = best_pitch[0];
                    best_num[0] = num;
                    best_den[0] = syy;
                    best_pitch[0] = i;
                } else {
                    best_num[1] = num;
                    best_den[1] = syy;
                    best_pitch[1] = i;
                }
            }
        }
        syy = syy
            .wrapping_add(mul16(y[i + len] as i32, y[i + len] as i32) >> yshift)
            .wrapping_sub(mul16(y[i] as i32, y[i] as i32) >> yshift);
        syy = syy.max(1);
    }
}

fn fir5(x: &mut [i16], num: &[i32; 5], n: usize) {
    let mut mem = [0i32; 5];
    for v in x[..n].iter_mut() {
        let mut sum = shl32(*v as i32, SIG_SHIFT);
        for k in 0..5 {
            sum = mac16(sum, num[k], mem[k]);
        }
        mem[4] = mem[3];
        mem[3] = mem[2];
        mem[2] = mem[1];
        mem[1] = mem[0];
        mem[0] = *v as i32;
        *v = round16(sum, SIG_SHIFT);
    }
}

/// `pitch_downsample`: 2x decimation with a 5-tap whitening filter.
pub fn pitch_downsample(x: &[&[i32]], x_lp: &mut [i16], len: usize) {
    let c = x.len();
    let mut maxabs = maxabs32(&x[0][..len]);
    if c == 2 {
        maxabs = maxabs.max(maxabs32(&x[1][..len]));
    }
    if maxabs < 1 {
        maxabs = 1;
    }
    let mut shift = (ilog2(maxabs) - 10).max(0);
    if c == 2 {
        shift += 1;
    }
    let half = len >> 1;
    for ch in 0..c {
        let xs = x[ch];
        for i in 1..half {
            let v = ((((xs[2 * i - 1].wrapping_add(xs[2 * i + 1])) >> 1).wrapping_add(xs[2 * i]))
                >> 1)
                >> shift;
            x_lp[i] = if ch == 0 {
                v as i16
            } else {
                x_lp[i].wrapping_add(v as i16)
            };
        }
        let v = (((xs[1] >> 1).wrapping_add(xs[0])) >> 1) >> shift;
        x_lp[0] = if ch == 0 {
            v as i16
        } else {
            x_lp[0].wrapping_add(v as i16)
        };
    }
    let mut ac = [0i32; 5];
    autocorr(&x_lp[..half], &mut ac, None, 0, 4, half);
    ac[0] = ac[0].wrapping_add(ac[0] >> 13);
    for (i, a) in ac.iter_mut().enumerate().skip(1) {
        *a = a.wrapping_sub(mul16_32_q15((2 * i * i) as i32, *a));
    }
    let mut lpc = [0i16; 4];
    lpc_from_autocorr(&mut lpc, &ac, 4);
    let mut tmp = 32767i32;
    for v in lpc.iter_mut() {
        tmp = mul16_q15(29491, tmp);
        *v = mul16_q15(*v as i32, tmp) as i16;
    }
    let c1 = 26214;
    let l = lpc.map(|v| v as i32);
    let lpc2 = [
        (l[0] + 3277) as i16 as i32,
        (l[1] + mul16_q15(c1, l[0])) as i16 as i32,
        (l[2] + mul16_q15(c1, l[1])) as i16 as i32,
        (l[3] + mul16_q15(c1, l[2])) as i16 as i32,
        mul16_q15(c1, l[3]) as i16 as i32,
    ];
    fir5(x_lp, &lpc2, half);
}

/// `pitch_search`: returns the pitch period of `y` against `x_lp`.
pub fn pitch_search(x_lp: &[i16], y: &[i16], len: usize, max_pitch: usize) -> usize {
    let lag = len + max_pitch;
    // Stack buffers: the decoder's and encoder's searches need under 512 entries each.
    let (mut x4, mut y4) = ([0i16; 1024], [0i16; 1024]);
    let x_lp4 = &mut x4[..len >> 2];
    for (j, v) in x_lp4.iter_mut().enumerate() {
        *v = x_lp[2 * j];
    }
    let y_lp4 = &mut y4[..lag >> 2];
    for (j, v) in y_lp4.iter_mut().enumerate() {
        *v = y[2 * j];
    }
    let xmax = maxabs16(x_lp4);
    let ymax = maxabs16(y_lp4);
    let mut shift = ilog2(1.max(xmax.max(ymax))) - 11;
    if shift > 0 {
        for v in x_lp4.iter_mut() {
            *v >>= shift;
        }
        for v in y_lp4.iter_mut() {
            *v >>= shift;
        }
        shift *= 2;
    } else {
        shift = 0;
    }
    let mut xc = [0i32; 1024];
    let xcorr = &mut xc[..max_pitch >> 1];
    let maxcorr = pitch_xcorr(x_lp4, y_lp4, xcorr, len >> 2, max_pitch >> 2);
    let mut best = [0usize; 2];
    find_best_pitch(
        xcorr,
        y_lp4,
        len >> 2,
        max_pitch >> 2,
        &mut best,
        0,
        maxcorr,
    );
    let mut maxcorr = 1i32;
    for i in 0..max_pitch >> 1 {
        xcorr[i] = 0;
        if (i as i32 - 2 * best[0] as i32).abs() > 2 && (i as i32 - 2 * best[1] as i32).abs() > 2 {
            continue;
        }
        let sum = dot_shr(&x_lp[..len >> 1], &y[i..i + (len >> 1)], shift);
        xcorr[i] = (-1).max(sum);
        maxcorr = maxcorr.max(sum);
    }
    find_best_pitch(
        xcorr,
        y,
        len >> 1,
        max_pitch >> 1,
        &mut best,
        shift + 1,
        maxcorr,
    );
    let offset: i32 = if best[0] > 0 && best[0] < (max_pitch >> 1) - 1 {
        let a = xcorr[best[0] - 1];
        let b = xcorr[best[0]];
        let c = xcorr[best[0] + 1];
        if (c - a) > mul16_32_q15(22938, b - a) {
            1
        } else if (a - c) > mul16_32_q15(22938, b - c) {
            -1
        } else {
            0
        }
    } else {
        0
    };
    (2 * best[0] as i32 - offset) as usize
}

/// `sum_j MULT16_16(a[j], b[j])` (wrapping) of equal-length slices: [`inner_prod`] for long
/// vectors.
pub fn dot16(a: &[i16], b: &[i16]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        return unsafe { super::simd_enc::dot16_avx2(a, b) };
    }
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: NEON is baseline on aarch64.
        return unsafe { super::simd_enc::dot16_neon(a, b) };
    }
    assert!(b.len() == a.len());
    inner_prod(a, b)
}

/// `sum_j MULT16_16(a[j], b[j]) >> shift` (wrapping) of equal-length slices, `shift < 32`.
pub fn dot_shr(a: &[i16], b: &[i16], shift: i32) -> i32 {
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        return unsafe { super::simd_enc::dot16_shr_avx2(a, b, shift) };
    }
    dot_shr_scalar(a, b, shift)
}

/// Scalar reference of [`dot_shr`].
pub fn dot_shr_scalar(a: &[i16], b: &[i16], shift: i32) -> i32 {
    assert!(b.len() == a.len());
    a.iter().zip(b).fold(0i32, |s, (&u, &v)| {
        s.wrapping_add(mul16(u as i32, v as i32) >> shift)
    })
}

/// `_celt_lpc`: LPC coefficients (Q12) from autocorrelation.
pub fn lpc_from_autocorr(lpc_out: &mut [i16], ac: &[i32], p: usize) {
    let mut lpc = [0i32; LPC_ORDER];
    let mut error = ac[0];
    if ac[0] != 0 {
        for i in 0..p {
            let mut rr = 0i32;
            for j in 0..i {
                rr = rr.wrapping_add(mul32_q31(lpc[j], ac[i - j]));
            }
            rr = rr.wrapping_add(ac[i + 1] >> 3);
            let r = super::math::frac_div32(shl32(rr, 3), error).wrapping_neg();
            lpc[i] = r >> 3;
            for j in 0..(i + 1) >> 1 {
                let tmp1 = lpc[j];
                let tmp2 = lpc[i - 1 - j];
                lpc[j] = tmp1.wrapping_add(mul32_q31(r, tmp2));
                lpc[i - 1 - j] = tmp2.wrapping_add(mul32_q31(r, tmp1));
            }
            error = error.wrapping_sub(mul32_q31(mul32_q31(r, r), error));
            if error < (ac[0] >> 10) {
                break;
            }
        }
    }
    for i in 0..p {
        lpc_out[i] = round16(lpc[i], 16);
    }
}

/// `celt_fir`: `y[i] = x[i] + sum_j num[j] * x[i - j - 1]`, reading `ord` samples of history
/// before `x_off`.
pub fn fir(x: &[i16], x_off: usize, num: &[i16], y: &mut [i16], n: usize, ord: usize) {
    for i in 0..n {
        let mut sum = shl32(x[x_off + i] as i32, SIG_SHIFT);
        for j in 0..ord {
            sum = mac16(sum, num[ord - 1 - j] as i32, x[x_off + i + j - ord] as i32);
        }
        y[i] = round16(sum, SIG_SHIFT);
    }
}

/// `celt_iir` (non-`SMALL_FOOTPRINT` form): all-pole filter, in place on `x[..n]`.
pub fn iir(x: &mut [i32], den: &[i16], n: usize, ord: usize, mem: &mut [i16]) {
    let mut rden = [0i32; LPC_ORDER];
    let rden = &mut rden[..ord];
    for (i, r) in rden.iter_mut().enumerate() {
        *r = den[ord - i - 1] as i32;
    }
    // The decoder extrapolates at most a frame plus its overlap.
    let mut y = [0i16; 2048 + LPC_ORDER];
    let y = &mut y[..n + ord];
    for i in 0..ord {
        y[i] = mem[ord - i - 1].wrapping_neg();
    }
    let mut i = 0usize;
    while i + 3 < n {
        let mut sum = [x[i], x[i + 1], x[i + 2], x[i + 3]];
        for j in 0..ord {
            for (k, s) in sum.iter_mut().enumerate() {
                *s = mac16(*s, rden[j], y[i + j + k] as i32);
            }
        }
        y[i + ord] = sround16(sum[0], SIG_SHIFT).wrapping_neg();
        x[i] = sum[0];
        sum[1] = mac16(sum[1], y[i + ord] as i32, den[0] as i32);
        y[i + ord + 1] = sround16(sum[1], SIG_SHIFT).wrapping_neg();
        x[i + 1] = sum[1];
        sum[2] = mac16(sum[2], y[i + ord + 1] as i32, den[0] as i32);
        sum[2] = mac16(sum[2], y[i + ord] as i32, den[1] as i32);
        y[i + ord + 2] = sround16(sum[2], SIG_SHIFT).wrapping_neg();
        x[i + 2] = sum[2];
        sum[3] = mac16(sum[3], y[i + ord + 2] as i32, den[0] as i32);
        sum[3] = mac16(sum[3], y[i + ord + 1] as i32, den[1] as i32);
        sum[3] = mac16(sum[3], y[i + ord] as i32, den[2] as i32);
        y[i + ord + 3] = sround16(sum[3], SIG_SHIFT).wrapping_neg();
        x[i + 3] = sum[3];
        i += 4;
    }
    while i < n {
        let mut sum = x[i];
        for j in 0..ord {
            sum = sum.wrapping_sub(mul16(rden[j], y[i + j] as i32));
        }
        y[i + ord] = sround16(sum, SIG_SHIFT);
        x[i] = sum;
        i += 1;
    }
    for i in 0..ord {
        mem[i] = x[n - i - 1] as i16;
    }
}

/// `_celt_autocorr`: autocorrelation of `x[..n]` up to `lag`, optionally windowed at both ends;
/// returns the scaling shift.
pub fn autocorr(
    x: &[i16],
    ac: &mut [i32],
    window: Option<&[i16]>,
    overlap: usize,
    lag: usize,
    n: usize,
) -> i32 {
    let fast_n = n - lag;
    let mut xx = [0i16; 2048];
    let xx = &mut xx[..n];
    xx.copy_from_slice(&x[..n]);
    if let (Some(w), true) = (window, overlap > 0) {
        for i in 0..overlap {
            xx[i] = mul16_q15(x[i] as i32, w[i] as i32) as i16;
            xx[n - i - 1] = mul16_q15(x[n - i - 1] as i32, w[i] as i32) as i16;
        }
    }
    let mut ac0 = 1 + ((n as i32) << 7);
    if n & 1 != 0 {
        ac0 = ac0.wrapping_add(mul16(xx[0] as i32, xx[0] as i32) >> 9);
    }
    let mut i = n & 1;
    while i < n {
        ac0 = ac0.wrapping_add(mul16(xx[i] as i32, xx[i] as i32) >> 9);
        ac0 = ac0.wrapping_add(mul16(xx[i + 1] as i32, xx[i + 1] as i32) >> 9);
        i += 2;
    }
    let mut shift = (ilog2(ac0) - 30 + 10) / 2;
    if shift > 0 {
        for v in xx.iter_mut() {
            *v = pshr32(*v as i32, shift) as i16;
        }
    } else {
        shift = 0;
    }
    pitch_xcorr(xx, xx, ac, fast_n, lag + 1);
    for k in 0..=lag {
        let mut d = 0i32;
        for i in k + fast_n..n {
            d = mac16(d, xx[i] as i32, xx[i - k] as i32);
        }
        ac[k] = ac[k].wrapping_add(d);
    }
    shift *= 2;
    if shift <= 0 {
        ac[0] = ac[0].wrapping_add(shl32(1, -shift));
    }
    if ac[0] < 268435456 {
        let shift2 = 29 - ilog(ac[0] as u32);
        for v in ac[..=lag].iter_mut() {
            *v = shl32(*v, shift2);
        }
        shift -= shift2;
    } else if ac[0] >= 536870912 {
        let mut shift2 = 1;
        if ac[0] >= 1073741824 {
            shift2 += 1;
        }
        for v in ac[..=lag].iter_mut() {
            *v >>= shift2;
        }
        shift += shift2;
    }
    shift
}

const COMB_GAINS: [[i32; 3]; 3] = [[10048, 7112, 4248], [15200, 8784, 0], [26208, 3280, 0]];

/// One output sample of the comb filter. `x(k)` reads the input relative to the sample.
#[inline(always)]
fn comb_overlap_sample(
    xi: i32,
    xt0: [i32; 5],
    xt1: [i32; 5],
    f: i32,
    g0: [i32; 3],
    g1: [i32; 3],
) -> i32 {
    let nf = 32767 - f;
    let y = xi
        .wrapping_add(mul16_32_q15(mul16_q15(nf, g0[0]), xt0[2]))
        .wrapping_add(mul16_32_q15(
            mul16_q15(nf, g0[1]),
            xt0[1].wrapping_add(xt0[3]),
        ))
        .wrapping_add(mul16_32_q15(
            mul16_q15(nf, g0[2]),
            xt0[0].wrapping_add(xt0[4]),
        ))
        .wrapping_add(mul16_32_q15(mul16_q15(f, g1[0]), xt1[2]))
        .wrapping_add(mul16_32_q15(
            mul16_q15(f, g1[1]),
            xt1[1].wrapping_add(xt1[3]),
        ))
        .wrapping_add(mul16_32_q15(
            mul16_q15(f, g1[2]),
            xt1[0].wrapping_add(xt1[4]),
        ));
    saturate(y, SIG_SAT)
}

/// Comb-filter parameters (`comb_filter` arguments besides the signal).
#[derive(Clone, Copy)]
pub struct Comb {
    pub t0: i32,
    pub t1: i32,
    pub g0: i32,
    pub g1: i32,
    pub tapset0: usize,
    pub tapset1: usize,
}

fn comb_gains(c: &Comb) -> ([i32; 3], [i32; 3]) {
    let g0 = COMB_GAINS[c.tapset0].map(|g| mul16_p15(c.g0, g));
    let g1 = COMB_GAINS[c.tapset1].map(|g| mul16_p15(c.g1, g));
    (g0, g1)
}

/// Samples `i0..n` of the cross-fading part of the comb filter on raw pointers, as
/// [`super::simd::comb_overlap_avx2`] defines it.
///
/// # Safety
///
/// As [`super::simd::comb_overlap_avx2`].
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn comb_overlap_tail(
    y: *mut i32,
    x: *const i32,
    i0: usize,
    n: usize,
    t0: usize,
    t1: usize,
    g0: [i32; 3],
    g1: [i32; 3],
    window: &[i16],
) {
    for i in i0..n {
        // Safety: per the function contract.
        unsafe {
            let r = |t: usize, k: usize| *x.add(i + k).sub(t + 2);
            let f = mul16_q15(window[i] as i32, window[i] as i32);
            let xt0 = [r(t0, 0), r(t0, 1), r(t0, 2), r(t0, 3), r(t0, 4)];
            let xt1 = [r(t1, 0), r(t1, 1), r(t1, 2), r(t1, 3), r(t1, 4)];
            *y.add(i) = comb_overlap_sample(*x.add(i), xt0, xt1, f, g0, g1);
        }
    }
}

/// `comb_filter` of `n` samples from `x` (history before it) into `y` on raw pointers.
///
/// # Safety
///
/// `x - max(t0, t1) - 2..x + n` must be readable and `y..y + n` writable; `y` either equals
/// `x` (in-place filtering, which reads earlier outputs as history) or does not overlap the
/// reads.
unsafe fn comb_filter_raw(
    y: *mut i32,
    x: *const i32,
    n: usize,
    c: Comb,
    window: &[i16],
    overlap: usize,
) {
    let t0 = c.t0.max(COMBFILTER_MINPERIOD) as usize;
    let t1 = c.t1.max(COMBFILTER_MINPERIOD) as usize;
    let (g0, g1) = comb_gains(&c);
    let overlap = if c.g0 == c.g1 && t0 == t1 && c.tapset0 == c.tapset1 {
        0
    } else {
        overlap.min(n)
    };
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2; the periods are at least
        // COMBFILTER_MINPERIOD (15), and the bounds are the caller's.
        unsafe {
            super::simd::comb_overlap_avx2(y, x, overlap, t0, t1, g0, g1, window);
            if c.g1 == 0 {
                if !std::ptr::eq(y, x) {
                    std::ptr::copy_nonoverlapping(x.add(overlap), y.add(overlap), n - overlap);
                }
                return;
            }
            super::simd::comb_const_avx2(y.add(overlap), x.add(overlap), n - overlap, t1, g1);
        }
        return;
    }
    // Safety: the bounds are the caller's.
    unsafe { comb_overlap_tail(y, x, 0, overlap, t0, t1, g0, g1, window) };
    if c.g1 == 0 {
        if !std::ptr::eq(y, x) {
            // Safety: the bounds are the caller's and the ranges do not overlap.
            unsafe { std::ptr::copy_nonoverlapping(x.add(overlap), y.add(overlap), n - overlap) };
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() && g1.iter().all(|&g| g as i16 != i16::MIN) {
        // Safety: NEON is baseline on aarch64; as above for the rest.
        unsafe {
            super::simd::comb_const_neon(y.add(overlap), x.add(overlap), n - overlap, t1, g1)
        };
        return;
    }
    // Safety: the bounds are the caller's.
    unsafe { comb_const_scalar(y.add(overlap), x.add(overlap), n - overlap, t1, g1) };
}

/// Scalar reference of [`super::simd::comb_const_avx2`].
///
/// # Safety
///
/// As [`super::simd::comb_const_avx2`].
pub(super) unsafe fn comb_const_scalar(
    y: *mut i32,
    x: *const i32,
    n: usize,
    t: usize,
    g: [i32; 3],
) {
    for i in 0..n {
        // Safety: per the function contract.
        unsafe {
            let r = |k: usize| *x.add(i + k).sub(t + 2);
            let v = (*x.add(i))
                .wrapping_add(mul16_32_q15(g[0], r(2)))
                .wrapping_add(mul16_32_q15(g[1], r(3).wrapping_add(r(1))))
                .wrapping_add(mul16_32_q15(g[2], r(4).wrapping_add(r(0))));
            *y.add(i) = saturate(v, SIG_SAT);
        }
    }
}

/// Scalar reference of [`comb_filter_inplace`].
#[cfg(test)]
pub(super) fn comb_filter_inplace_scalar(
    buf: &mut [i32],
    off: usize,
    n: usize,
    c: Comb,
    window: &[i16],
    overlap: usize,
) {
    if c.g0 == 0 && c.g1 == 0 {
        return;
    }
    let t0 = c.t0.max(COMBFILTER_MINPERIOD) as usize;
    let t1 = c.t1.max(COMBFILTER_MINPERIOD) as usize;
    let (g0, g1) = comb_gains(&c);
    let overlap = if c.g0 == c.g1 && t0 == t1 && c.tapset0 == c.tapset1 {
        0
    } else {
        overlap.min(n)
    };
    assert!(off >= t0.max(t1) + 2 && off + n <= buf.len());
    // Safety: bounded by the assert.
    unsafe {
        let p = buf.as_mut_ptr().add(off);
        comb_overlap_tail(p, p, 0, overlap, t0, t1, g0, g1, window);
        if c.g1 != 0 {
            comb_const_scalar(p.add(overlap), p.add(overlap), n - overlap, t1, g1);
        }
    }
}

/// `comb_filter` in place: `buf[off..off + n]` is filtered with its own (already filtered)
/// history before `off`, as libopus does when input and output alias.
pub fn comb_filter_inplace(
    buf: &mut [i32],
    off: usize,
    n: usize,
    c: Comb,
    window: &[i16],
    overlap: usize,
) {
    if c.g0 == 0 && c.g1 == 0 {
        return;
    }
    let t = c.t0.max(c.t1).max(COMBFILTER_MINPERIOD) as usize;
    assert!(off >= t + 2 && off + n <= buf.len() && window.len() >= overlap.min(n));
    // Safety: the asserts bound every read and write to `buf`.
    unsafe {
        let p = buf.as_mut_ptr().add(off);
        comb_filter_raw(p, p, n, c, window, overlap);
    }
}

/// `comb_filter` with separate input `x` (history before `x_off`) and output `y`.
pub fn comb_filter_to(
    y: &mut [i32],
    x: &[i32],
    x_off: usize,
    n: usize,
    c: Comb,
    window: &[i16],
    overlap: usize,
) {
    if c.g0 == 0 && c.g1 == 0 {
        y[..n].copy_from_slice(&x[x_off..x_off + n]);
        return;
    }
    let t = c.t0.max(c.t1).max(COMBFILTER_MINPERIOD) as usize;
    assert!(
        x_off >= t + 2 && x_off + n <= x.len() && y.len() >= n && window.len() >= overlap.min(n)
    );
    // Safety: the asserts bound every access; `y` and `x` are distinct borrows.
    unsafe { comb_filter_raw(y.as_mut_ptr(), x.as_ptr().add(x_off), n, c, window, overlap) };
}
