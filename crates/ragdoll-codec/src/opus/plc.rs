//! CELT signal-domain helpers (libopus `pitch.c`, `celt_lpc.c`, `celt.c`): pitch search and LPC
//! for packet-loss concealment, and the pitch pre/post-filter.

use super::math::{
    ilog2, mac16, maxabs16, maxabs32, mul16, mul16_32_q15, mul16_p15, mul16_q15, mul32_q31, pshr32,
    round16, saturate, shl32, sround16, vshr32,
};
use super::range::ilog;

pub const SIG_SHIFT: i32 = 12;
pub const SIG_SAT: i32 = 300000000;
pub const LPC_ORDER: usize = 24;
pub const COMBFILTER_MINPERIOD: i32 = 15;

/// Plain correlation `xcorr[i] = sum_j x[j] * y[i + j]` (`celt_pitch_xcorr`); returns the
/// maximum, at least 1. Integer sums are exact, so libopus's unrolling does not matter.
fn pitch_xcorr(x: &[i16], y: &[i16], xcorr: &mut [i32], len: usize, max_pitch: usize) -> i32 {
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
    let mut x_lp4: Vec<i16> = (0..len >> 2).map(|j| x_lp[2 * j]).collect();
    let mut y_lp4: Vec<i16> = (0..lag >> 2).map(|j| y[2 * j]).collect();
    let xmax = maxabs16(&x_lp4);
    let ymax = maxabs16(&y_lp4);
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
    let mut xcorr = vec![0i32; max_pitch >> 1];
    let maxcorr = pitch_xcorr(&x_lp4, &y_lp4, &mut xcorr, len >> 2, max_pitch >> 2);
    let mut best = [0usize; 2];
    find_best_pitch(
        &xcorr,
        &y_lp4,
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
        let mut sum = 0i32;
        for j in 0..len >> 1 {
            sum = sum.wrapping_add(mul16(x_lp[j] as i32, y[i + j] as i32) >> shift);
        }
        xcorr[i] = (-1).max(sum);
        maxcorr = maxcorr.max(sum);
    }
    find_best_pitch(
        &xcorr,
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
    let rden: Vec<i32> = (0..ord).map(|i| den[ord - i - 1] as i32).collect();
    let mut y = vec![0i16; n + ord];
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
    let mut xx: Vec<i16> = x[..n].to_vec();
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
    pitch_xcorr(&xx, &xx, ac, fast_n, lag + 1);
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
    let t0 = c.t0.max(COMBFILTER_MINPERIOD) as usize;
    let t1 = c.t1.max(COMBFILTER_MINPERIOD) as usize;
    let (g0, g1) = comb_gains(&c);
    let overlap = if c.g0 == c.g1 && t0 == t1 && c.tapset0 == c.tapset1 {
        0
    } else {
        overlap
    };
    let mut i = 0usize;
    while i < overlap {
        let p = off + i;
        let f = mul16_q15(window[i] as i32, window[i] as i32);
        let xt0 = [
            buf[p - t0 - 2],
            buf[p - t0 - 1],
            buf[p - t0],
            buf[p - t0 + 1],
            buf[p - t0 + 2],
        ];
        let xt1 = [
            buf[p - t1 - 2],
            buf[p - t1 - 1],
            buf[p - t1],
            buf[p - t1 + 1],
            buf[p - t1 + 2],
        ];
        buf[p] = comb_overlap_sample(buf[p], xt0, xt1, f, g0, g1);
        i += 1;
    }
    if c.g1 == 0 {
        return;
    }
    while i < n {
        let p = off + i;
        let y = buf[p]
            .wrapping_add(mul16_32_q15(g1[0], buf[p - t1]))
            .wrapping_add(mul16_32_q15(
                g1[1],
                buf[p - t1 + 1].wrapping_add(buf[p - t1 - 1]),
            ))
            .wrapping_add(mul16_32_q15(
                g1[2],
                buf[p - t1 + 2].wrapping_add(buf[p - t1 - 2]),
            ));
        buf[p] = saturate(y, SIG_SAT);
        i += 1;
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
    let t0 = c.t0.max(COMBFILTER_MINPERIOD) as usize;
    let t1 = c.t1.max(COMBFILTER_MINPERIOD) as usize;
    let (g0, g1) = comb_gains(&c);
    let overlap = if c.g0 == c.g1 && t0 == t1 && c.tapset0 == c.tapset1 {
        0
    } else {
        overlap
    };
    let mut i = 0usize;
    while i < overlap {
        let p = x_off + i;
        let f = mul16_q15(window[i] as i32, window[i] as i32);
        let xt0 = [
            x[p - t0 - 2],
            x[p - t0 - 1],
            x[p - t0],
            x[p - t0 + 1],
            x[p - t0 + 2],
        ];
        let xt1 = [
            x[p - t1 - 2],
            x[p - t1 - 1],
            x[p - t1],
            x[p - t1 + 1],
            x[p - t1 + 2],
        ];
        y[i] = comb_overlap_sample(x[p], xt0, xt1, f, g0, g1);
        i += 1;
    }
    if c.g1 == 0 {
        y[overlap..n].copy_from_slice(&x[x_off + overlap..x_off + n]);
        return;
    }
    while i < n {
        let p = x_off + i;
        let v = x[p]
            .wrapping_add(mul16_32_q15(g1[0], x[p - t1]))
            .wrapping_add(mul16_32_q15(
                g1[1],
                x[p - t1 + 1].wrapping_add(x[p - t1 - 1]),
            ))
            .wrapping_add(mul16_32_q15(
                g1[2],
                x[p - t1 + 2].wrapping_add(x[p - t1 - 2]),
            ));
        y[i] = saturate(v, SIG_SAT);
        i += 1;
    }
}
