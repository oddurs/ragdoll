//! Fixed-point mixed-radix FFT and MDCT of CELT (libopus `kiss_fft.c`, `mdct.c`), for the four
//! transform sizes of the 48 kHz mode (N = 1920, 960, 480, 240).

use super::celt_tables::{
    FFT_BITREV60, FFT_BITREV120, FFT_BITREV240, FFT_BITREV480, FFT_TWIDDLES, MDCT_TWIDDLES,
};
use super::math::{mul16_32_q15, mul16_32_q16, pshr32};

#[derive(Clone, Copy, Default, Debug)]
pub struct Cpx {
    pub r: i32,
    pub i: i32,
}

/// A static FFT configuration (`kiss_fft_state`).
#[allow(dead_code)] // Encoder side, used by the CELT encoder (BUG-0078).
pub struct FftState {
    pub nfft: usize,
    pub scale: i32,
    pub scale_shift: i32,
    pub shift: i32,
    pub factors: [i32; 16],
    pub bitrev: &'static [i16],
}

pub static FFT_STATES: [FftState; 4] = [
    FftState {
        nfft: 480,
        scale: 17476,
        scale_shift: 8,
        shift: -1,
        factors: [5, 96, 3, 32, 4, 8, 2, 4, 4, 1, 0, 0, 0, 0, 0, 0],
        bitrev: &FFT_BITREV480,
    },
    FftState {
        nfft: 240,
        scale: 17476,
        scale_shift: 7,
        shift: 1,
        factors: [5, 48, 3, 16, 4, 4, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        bitrev: &FFT_BITREV240,
    },
    FftState {
        nfft: 120,
        scale: 17476,
        scale_shift: 6,
        shift: 2,
        factors: [5, 24, 3, 8, 2, 4, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        bitrev: &FFT_BITREV120,
    },
    FftState {
        nfft: 60,
        scale: 17476,
        scale_shift: 5,
        shift: 3,
        factors: [5, 12, 3, 4, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        bitrev: &FFT_BITREV60,
    },
];

#[inline(always)]
fn smul(a: i32, b: i32) -> i32 {
    mul16_32_q15(b, a)
}

#[inline(always)]
fn cmul(a: Cpx, b: (i16, i16)) -> Cpx {
    let (br, bi) = (b.0 as i32, b.1 as i32);
    Cpx {
        r: smul(a.r, br).wrapping_sub(smul(a.i, bi)),
        i: smul(a.r, bi).wrapping_add(smul(a.i, br)),
    }
}

#[inline(always)]
fn cadd(a: Cpx, b: Cpx) -> Cpx {
    Cpx {
        r: a.r.wrapping_add(b.r),
        i: a.i.wrapping_add(b.i),
    }
}

#[inline(always)]
fn csub(a: Cpx, b: Cpx) -> Cpx {
    Cpx {
        r: a.r.wrapping_sub(b.r),
        i: a.i.wrapping_sub(b.i),
    }
}

fn bfly2(f: &mut [Cpx], m: usize, n: usize) {
    debug_assert_eq!(m, 4);
    let tw = 23170; // QCONST16(0.7071067812, 15)
    for i in 0..n {
        let base = i * 8;
        let (a, b) = f[base..base + 8].split_at_mut(4);
        let t = b[0];
        b[0] = csub(a[0], t);
        a[0] = cadd(a[0], t);
        let t = Cpx {
            r: smul(b[1].r.wrapping_add(b[1].i), tw),
            i: smul(b[1].i.wrapping_sub(b[1].r), tw),
        };
        b[1] = csub(a[1], t);
        a[1] = cadd(a[1], t);
        let t = Cpx {
            r: b[2].i,
            i: b[2].r.wrapping_neg(),
        };
        b[2] = csub(a[2], t);
        a[2] = cadd(a[2], t);
        let t = Cpx {
            r: smul(b[3].i.wrapping_sub(b[3].r), tw),
            i: smul(b[3].i.wrapping_add(b[3].r).wrapping_neg(), tw),
        };
        b[3] = csub(a[3], t);
        a[3] = cadd(a[3], t);
    }
}

fn bfly4(f: &mut [Cpx], fstride: usize, m: usize, n: usize, mm: usize) {
    if m == 1 {
        for i in 0..n {
            let o = i * 4;
            let scratch0 = csub(f[o], f[o + 2]);
            f[o] = cadd(f[o], f[o + 2]);
            let mut scratch1 = cadd(f[o + 1], f[o + 3]);
            f[o + 2] = csub(f[o], scratch1);
            f[o] = cadd(f[o], scratch1);
            scratch1 = csub(f[o + 1], f[o + 3]);
            f[o + 1] = Cpx {
                r: scratch0.r.wrapping_add(scratch1.i),
                i: scratch0.i.wrapping_sub(scratch1.r),
            };
            f[o + 3] = Cpx {
                r: scratch0.r.wrapping_sub(scratch1.i),
                i: scratch0.i.wrapping_add(scratch1.r),
            };
        }
        return;
    }
    let (m2, m3) = (2 * m, 3 * m);
    for i in 0..n {
        let base = i * mm;
        for j in 0..m {
            let o = base + j;
            let s0 = cmul(f[o + m], FFT_TWIDDLES[j * fstride]);
            let s1 = cmul(f[o + m2], FFT_TWIDDLES[j * fstride * 2]);
            let s2 = cmul(f[o + m3], FFT_TWIDDLES[j * fstride * 3]);
            let s5 = csub(f[o], s1);
            f[o] = cadd(f[o], s1);
            let s3 = cadd(s0, s2);
            let s4 = csub(s0, s2);
            f[o + m2] = csub(f[o], s3);
            f[o] = cadd(f[o], s3);
            f[o + m] = Cpx {
                r: s5.r.wrapping_add(s4.i),
                i: s5.i.wrapping_sub(s4.r),
            };
            f[o + m3] = Cpx {
                r: s5.r.wrapping_sub(s4.i),
                i: s5.i.wrapping_add(s4.r),
            };
        }
    }
}

fn bfly3(f: &mut [Cpx], fstride: usize, m: usize, n: usize, mm: usize) {
    let m2 = 2 * m;
    let epi3_i: i32 = -28378;
    for i in 0..n {
        let base = i * mm;
        for k in 0..m {
            let o = base + k;
            let s1 = cmul(f[o + m], FFT_TWIDDLES[k * fstride]);
            let s2 = cmul(f[o + m2], FFT_TWIDDLES[k * fstride * 2]);
            let s3 = cadd(s1, s2);
            let mut s0 = csub(s1, s2);
            f[o + m] = Cpx {
                r: f[o].r.wrapping_sub(s3.r >> 1),
                i: f[o].i.wrapping_sub(s3.i >> 1),
            };
            s0 = Cpx {
                r: smul(s0.r, epi3_i),
                i: smul(s0.i, epi3_i),
            };
            f[o] = cadd(f[o], s3);
            f[o + m2] = Cpx {
                r: f[o + m].r.wrapping_add(s0.i),
                i: f[o + m].i.wrapping_sub(s0.r),
            };
            f[o + m] = Cpx {
                r: f[o + m].r.wrapping_sub(s0.i),
                i: f[o + m].i.wrapping_add(s0.r),
            };
        }
    }
}

fn bfly5(f: &mut [Cpx], fstride: usize, m: usize, n: usize, mm: usize) {
    let (yar, yai) = (10126i32, -31164i32);
    let (ybr, ybi) = (-26510i32, -19261i32);
    for i in 0..n {
        let base = i * mm;
        for u in 0..m {
            let (o0, o1, o2, o3, o4) = (
                base + u,
                base + u + m,
                base + u + 2 * m,
                base + u + 3 * m,
                base + u + 4 * m,
            );
            let s0 = f[o0];
            let s1 = cmul(f[o1], FFT_TWIDDLES[u * fstride]);
            let s2 = cmul(f[o2], FFT_TWIDDLES[2 * u * fstride]);
            let s3 = cmul(f[o3], FFT_TWIDDLES[3 * u * fstride]);
            let s4 = cmul(f[o4], FFT_TWIDDLES[4 * u * fstride]);
            let s7 = cadd(s1, s4);
            let s10 = csub(s1, s4);
            let s8 = cadd(s2, s3);
            let s9 = csub(s2, s3);
            f[o0] = Cpx {
                r: f[o0].r.wrapping_add(s7.r.wrapping_add(s8.r)),
                i: f[o0].i.wrapping_add(s7.i.wrapping_add(s8.i)),
            };
            let s5 = Cpx {
                r: s0
                    .r
                    .wrapping_add(smul(s7.r, yar).wrapping_add(smul(s8.r, ybr))),
                i: s0
                    .i
                    .wrapping_add(smul(s7.i, yar).wrapping_add(smul(s8.i, ybr))),
            };
            let s6 = Cpx {
                r: smul(s10.i, yai).wrapping_add(smul(s9.i, ybi)),
                i: smul(s10.r, yai)
                    .wrapping_add(smul(s9.r, ybi))
                    .wrapping_neg(),
            };
            f[o1] = csub(s5, s6);
            f[o4] = cadd(s5, s6);
            let s11 = Cpx {
                r: s0
                    .r
                    .wrapping_add(smul(s7.r, ybr).wrapping_add(smul(s8.r, yar))),
                i: s0
                    .i
                    .wrapping_add(smul(s7.i, ybr).wrapping_add(smul(s8.i, yar))),
            };
            let s12 = Cpx {
                r: smul(s9.i, yai).wrapping_sub(smul(s10.i, ybi)),
                i: smul(s10.r, ybi).wrapping_sub(smul(s9.r, yai)),
            };
            f[o2] = cadd(s11, s12);
            f[o3] = csub(s11, s12);
        }
    }
}

/// `opus_fft_impl`: in-place FFT of bit-reversed input.
pub fn fft_impl(st: &FftState, f: &mut [Cpx]) {
    let shift = st.shift.max(0) as usize;
    let mut fstride = [0usize; 8];
    fstride[0] = 1;
    let mut l = 0usize;
    loop {
        let p = st.factors[2 * l] as usize;
        let m = st.factors[2 * l + 1];
        fstride[l + 1] = fstride[l] * p;
        l += 1;
        if m == 1 {
            break;
        }
    }
    let mut m = st.factors[2 * l - 1] as usize;
    for i in (0..l).rev() {
        let m2 = if i != 0 {
            st.factors[2 * i - 1] as usize
        } else {
            1
        };
        match st.factors[2 * i] {
            2 => bfly2(f, m, fstride[i]),
            4 => bfly4(f, fstride[i] << shift, m, fstride[i], m2),
            3 => bfly3(f, fstride[i] << shift, m, fstride[i], m2),
            5 => bfly5(f, fstride[i] << shift, m, fstride[i], m2),
            _ => unreachable!(),
        }
        m = m2;
    }
}

/// Offset of the twiddles for transform `shift` in [`MDCT_TWIDDLES`] and the transform size.
fn trig(shift: usize) -> (&'static [i16], usize) {
    let mut n = 1920usize;
    let mut off = 0usize;
    for _ in 0..shift {
        n >>= 1;
        off += n;
    }
    (&MDCT_TWIDDLES[off..], n)
}

/// `clt_mdct_backward`: inverse MDCT of the `stride`-interleaved coefficients `input[..]` with
/// TDAC windowing; `out` holds `overlap/2` samples of history before the new output.
pub fn mdct_backward(
    input: &[i32],
    out: &mut [i32],
    window: &[i16],
    overlap: usize,
    shift: usize,
    stride: usize,
) {
    let (t, n) = trig(shift);
    let n2 = n >> 1;
    let n4 = n >> 2;
    let st = &FFT_STATES[shift];
    let yo = overlap >> 1;
    {
        // Pre-rotation, writing bit-reversed complex values into out[yo..].
        let mut xp1 = 0usize;
        let mut xp2 = stride * (n2 - 1);
        for i in 0..n4 {
            let rev = st.bitrev[i] as usize;
            let (x1, x2) = (input[xp1], input[xp2]);
            let yr = smul(x2, t[i] as i32).wrapping_add(smul(x1, t[n4 + i] as i32));
            let yi = smul(x1, t[i] as i32).wrapping_sub(smul(x2, t[n4 + i] as i32));
            out[yo + 2 * rev + 1] = yr;
            out[yo + 2 * rev] = yi;
            xp1 += 2 * stride;
            xp2 = xp2.wrapping_sub(2 * stride);
        }
    }
    {
        let mut buf: Vec<Cpx> = (0..n4)
            .map(|k| Cpx {
                r: out[yo + 2 * k],
                i: out[yo + 2 * k + 1],
            })
            .collect();
        fft_impl(st, &mut buf);
        for (k, c) in buf.iter().enumerate() {
            out[yo + 2 * k] = c.r;
            out[yo + 2 * k + 1] = c.i;
        }
    }
    {
        // Post-rotation and de-shuffling, in place from both ends.
        let mut yp0 = yo;
        let mut yp1 = yo + n2 - 2;
        for i in 0..(n4 + 1) >> 1 {
            let (re, im) = (out[yp0 + 1], out[yp0]);
            let (t0, t1) = (t[i] as i32, t[n4 + i] as i32);
            let yr = smul(re, t0).wrapping_add(smul(im, t1));
            let yi = smul(re, t1).wrapping_sub(smul(im, t0));
            let (re, im) = (out[yp1 + 1], out[yp1]);
            out[yp0] = yr;
            out[yp1 + 1] = yi;
            let (t0, t1) = (t[n4 - i - 1] as i32, t[n2 - i - 1] as i32);
            let yr = smul(re, t0).wrapping_add(smul(im, t1));
            let yi = smul(re, t1).wrapping_sub(smul(im, t0));
            out[yp1] = yr;
            out[yp0 + 1] = yi;
            yp0 += 2;
            yp1 = yp1.wrapping_sub(2);
        }
    }
    // Mirror on both sides for TDAC.
    for i in 0..overlap / 2 {
        let xp1 = overlap - 1 - i;
        let (w1, w2) = (window[i] as i32, window[overlap - 1 - i] as i32);
        let x1 = out[xp1];
        let x2 = out[i];
        out[i] = mul16_32_q15(w2, x2).wrapping_sub(mul16_32_q15(w1, x1));
        out[xp1] = mul16_32_q15(w1, x2).wrapping_add(mul16_32_q15(w2, x1));
    }
}

/// `clt_mdct_forward`: forward MDCT of `input` (`N/2 + overlap` samples) into `stride`-interleaved
/// coefficients.
#[allow(dead_code)] // Encoder side, used by the CELT encoder (BUG-0078).
pub fn mdct_forward(
    input: &[i32],
    out: &mut [i32],
    window: &[i16],
    overlap: usize,
    shift: usize,
    stride: usize,
) {
    let (t, n) = trig(shift);
    let st = &FFT_STATES[shift];
    let scale = st.scale;
    let scale_shift = st.scale_shift - 1;
    let n2 = n >> 1;
    let n4 = n >> 2;
    let mut f = vec![0i32; n2];
    {
        // Window, shuffle and fold.
        let mut xp1 = overlap >> 1;
        let mut xp2 = n2 - 1 + (overlap >> 1);
        let mut wp1 = overlap >> 1;
        let mut wp2 = (overlap >> 1).wrapping_sub(1);
        let mut yp = 0usize;
        let mut i = 0usize;
        while i < (overlap + 3) >> 2 {
            f[yp] = mul16_32_q15(window[wp2] as i32, input[xp1 + n2])
                .wrapping_add(mul16_32_q15(window[wp1] as i32, input[xp2]));
            f[yp + 1] = mul16_32_q15(window[wp1] as i32, input[xp1])
                .wrapping_sub(mul16_32_q15(window[wp2] as i32, input[xp2 - n2]));
            yp += 2;
            xp1 += 2;
            xp2 = xp2.wrapping_sub(2);
            wp1 += 2;
            wp2 = wp2.wrapping_sub(2);
            i += 1;
        }
        wp1 = 0;
        wp2 = overlap - 1;
        while i < n4 - ((overlap + 3) >> 2) {
            f[yp] = input[xp2];
            f[yp + 1] = input[xp1];
            yp += 2;
            xp1 += 2;
            xp2 = xp2.wrapping_sub(2);
            i += 1;
        }
        while i < n4 {
            f[yp] = mul16_32_q15(window[wp2] as i32, input[xp2])
                .wrapping_sub(mul16_32_q15(window[wp1] as i32, input[xp1 - n2]));
            f[yp + 1] = mul16_32_q15(window[wp2] as i32, input[xp1])
                .wrapping_add(mul16_32_q15(window[wp1] as i32, input[xp2 + n2]));
            yp += 2;
            xp1 += 2;
            xp2 = xp2.wrapping_sub(2);
            wp1 += 2;
            wp2 = wp2.wrapping_sub(2);
            i += 1;
        }
    }
    let mut f2 = vec![Cpx::default(); n4];
    for i in 0..n4 {
        let (t0, t1) = (t[i] as i32, t[n4 + i] as i32);
        let (re, im) = (f[2 * i], f[2 * i + 1]);
        let yr = smul(re, t0).wrapping_sub(smul(im, t1));
        let yi = smul(im, t0).wrapping_add(smul(re, t1));
        f2[st.bitrev[i] as usize] = Cpx {
            r: pshr32(mul16_32_q16(scale, yr), scale_shift),
            i: pshr32(mul16_32_q16(scale, yi), scale_shift),
        };
    }
    fft_impl(st, &mut f2);
    let mut yp1 = 0usize;
    let mut yp2 = stride * (n2 - 1);
    for (i, fp) in f2.iter().enumerate() {
        let (t0, t1) = (t[i] as i32, t[n4 + i] as i32);
        let yr = smul(fp.i, t1).wrapping_sub(smul(fp.r, t0));
        let yi = smul(fp.r, t1).wrapping_add(smul(fp.i, t0));
        out[yp1] = yr;
        out[yp2] = yi;
        yp1 += 2 * stride;
        yp2 = yp2.wrapping_sub(2 * stride);
    }
}
