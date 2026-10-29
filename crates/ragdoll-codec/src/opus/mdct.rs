//! Fixed-point mixed-radix FFT and MDCT of CELT (libopus `kiss_fft.c`, `mdct.c`), for the four
//! transform sizes of the 48 kHz mode (N = 1920, 960, 480, 240).
//!
//! The FFT works in place on interleaved `(r, i)` pairs of `i32`, so the MDCT runs it directly
//! on its output buffer as libopus does. Its butterflies have AVX2 and NEON versions in
//! [`super::simd`], bit-exact with the scalar ones here.

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
pub struct FftState {
    pub scale: i32,
    pub scale_shift: i32,
    pub shift: i32,
    pub factors: [i32; 16],
    pub bitrev: &'static [i16],
}

pub static FFT_STATES: [FftState; 4] = [
    FftState {
        scale: 17476,
        scale_shift: 8,
        shift: -1,
        factors: [5, 96, 3, 32, 4, 8, 2, 4, 4, 1, 0, 0, 0, 0, 0, 0],
        bitrev: &FFT_BITREV480,
    },
    FftState {
        scale: 17476,
        scale_shift: 7,
        shift: 1,
        factors: [5, 48, 3, 16, 4, 4, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        bitrev: &FFT_BITREV240,
    },
    FftState {
        scale: 17476,
        scale_shift: 6,
        shift: 2,
        factors: [5, 24, 3, 8, 2, 4, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        bitrev: &FFT_BITREV120,
    },
    FftState {
        scale: 17476,
        scale_shift: 5,
        shift: 3,
        factors: [5, 12, 3, 4, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        bitrev: &FFT_BITREV60,
    },
];

/// One radix pass of `opus_fft_impl`: `n` groups `mm` apart of `p` interleaved sub-transforms
/// of length `m`, with twiddle stride `fstride` into [`FFT_TWIDDLES`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Stage {
    pub p: usize,
    pub m: usize,
    pub n: usize,
    pub mm: usize,
    pub fstride: usize,
}

/// The radix passes of `st` in execution order.
pub fn fft_stages(st: &FftState) -> ([Stage; 5], usize) {
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
    let mut stages = [Stage::default(); 5];
    let mut m = st.factors[2 * l - 1] as usize;
    for (k, i) in (0..l).rev().enumerate() {
        let m2 = if i != 0 {
            st.factors[2 * i - 1] as usize
        } else {
            1
        };
        stages[k] = Stage {
            p: st.factors[2 * i] as usize,
            m,
            n: fstride[i],
            mm: m2,
            fstride: fstride[i] << shift,
        };
        m = m2;
    }
    (stages, l)
}

#[inline(always)]
fn smul(a: i32, b: i32) -> i32 {
    mul16_32_q15(b, a)
}

#[inline(always)]
fn ld(f: &[[i32; 2]], k: usize) -> Cpx {
    Cpx {
        r: f[k][0],
        i: f[k][1],
    }
}

#[inline(always)]
fn st(f: &mut [[i32; 2]], k: usize, c: Cpx) {
    f[k] = [c.r, c.i];
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

pub(super) fn bfly2(f: &mut [[i32; 2]], m: usize, n: usize) {
    debug_assert_eq!(m, 4);
    let tw = 23170; // QCONST16(0.7071067812, 15)
    for i in 0..n {
        let base = i * 8;
        let (a, b) = f[base..base + 8].split_at_mut(4);
        let t = ld(b, 0);
        st(b, 0, csub(ld(a, 0), t));
        st(a, 0, cadd(ld(a, 0), t));
        let b1 = ld(b, 1);
        let t = Cpx {
            r: smul(b1.r.wrapping_add(b1.i), tw),
            i: smul(b1.i.wrapping_sub(b1.r), tw),
        };
        st(b, 1, csub(ld(a, 1), t));
        st(a, 1, cadd(ld(a, 1), t));
        let b2 = ld(b, 2);
        let t = Cpx {
            r: b2.i,
            i: b2.r.wrapping_neg(),
        };
        st(b, 2, csub(ld(a, 2), t));
        st(a, 2, cadd(ld(a, 2), t));
        let b3 = ld(b, 3);
        let t = Cpx {
            r: smul(b3.i.wrapping_sub(b3.r), tw),
            i: smul(b3.i.wrapping_add(b3.r).wrapping_neg(), tw),
        };
        st(b, 3, csub(ld(a, 3), t));
        st(a, 3, cadd(ld(a, 3), t));
    }
}

/// The first radix-4 pass (`m == 1`), which needs no twiddles.
pub(super) fn bfly4_m1(f: &mut [[i32; 2]], n: usize) {
    for i in 0..n {
        let o = i * 4;
        let (f0, f1, f2, f3) = (ld(f, o), ld(f, o + 1), ld(f, o + 2), ld(f, o + 3));
        let scratch0 = csub(f0, f2);
        let f0 = cadd(f0, f2);
        let scratch1 = cadd(f1, f3);
        st(f, o + 2, csub(f0, scratch1));
        st(f, o, cadd(f0, scratch1));
        let scratch1 = csub(f1, f3);
        st(
            f,
            o + 1,
            Cpx {
                r: scratch0.r.wrapping_add(scratch1.i),
                i: scratch0.i.wrapping_sub(scratch1.r),
            },
        );
        st(
            f,
            o + 3,
            Cpx {
                r: scratch0.r.wrapping_sub(scratch1.i),
                i: scratch0.i.wrapping_add(scratch1.r),
            },
        );
    }
}

fn bfly4(f: &mut [[i32; 2]], fstride: usize, m: usize, n: usize, mm: usize) {
    if m == 1 {
        bfly4_m1(f, n);
        return;
    }
    let (m2, m3) = (2 * m, 3 * m);
    for i in 0..n {
        let base = i * mm;
        for j in 0..m {
            let o = base + j;
            let s0 = cmul(ld(f, o + m), FFT_TWIDDLES[j * fstride]);
            let s1 = cmul(ld(f, o + m2), FFT_TWIDDLES[j * fstride * 2]);
            let s2 = cmul(ld(f, o + m3), FFT_TWIDDLES[j * fstride * 3]);
            let f0 = ld(f, o);
            let s5 = csub(f0, s1);
            let f0 = cadd(f0, s1);
            let s3 = cadd(s0, s2);
            let s4 = csub(s0, s2);
            st(f, o + m2, csub(f0, s3));
            st(f, o, cadd(f0, s3));
            st(
                f,
                o + m,
                Cpx {
                    r: s5.r.wrapping_add(s4.i),
                    i: s5.i.wrapping_sub(s4.r),
                },
            );
            st(
                f,
                o + m3,
                Cpx {
                    r: s5.r.wrapping_sub(s4.i),
                    i: s5.i.wrapping_add(s4.r),
                },
            );
        }
    }
}

fn bfly3(f: &mut [[i32; 2]], fstride: usize, m: usize, n: usize, mm: usize) {
    let m2 = 2 * m;
    let epi3_i: i32 = -28378;
    for i in 0..n {
        let base = i * mm;
        for k in 0..m {
            let o = base + k;
            let s1 = cmul(ld(f, o + m), FFT_TWIDDLES[k * fstride]);
            let s2 = cmul(ld(f, o + m2), FFT_TWIDDLES[k * fstride * 2]);
            let s3 = cadd(s1, s2);
            let s0 = csub(s1, s2);
            let f0 = ld(f, o);
            let fm = Cpx {
                r: f0.r.wrapping_sub(s3.r >> 1),
                i: f0.i.wrapping_sub(s3.i >> 1),
            };
            let s0 = Cpx {
                r: smul(s0.r, epi3_i),
                i: smul(s0.i, epi3_i),
            };
            st(f, o, cadd(f0, s3));
            st(
                f,
                o + m2,
                Cpx {
                    r: fm.r.wrapping_add(s0.i),
                    i: fm.i.wrapping_sub(s0.r),
                },
            );
            st(
                f,
                o + m,
                Cpx {
                    r: fm.r.wrapping_sub(s0.i),
                    i: fm.i.wrapping_add(s0.r),
                },
            );
        }
    }
}

fn bfly5(f: &mut [[i32; 2]], fstride: usize, m: usize, n: usize, mm: usize) {
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
            let s0 = ld(f, o0);
            let s1 = cmul(ld(f, o1), FFT_TWIDDLES[u * fstride]);
            let s2 = cmul(ld(f, o2), FFT_TWIDDLES[2 * u * fstride]);
            let s3 = cmul(ld(f, o3), FFT_TWIDDLES[3 * u * fstride]);
            let s4 = cmul(ld(f, o4), FFT_TWIDDLES[4 * u * fstride]);
            let s7 = cadd(s1, s4);
            let s10 = csub(s1, s4);
            let s8 = cadd(s2, s3);
            let s9 = csub(s2, s3);
            st(
                f,
                o0,
                Cpx {
                    r: s0.r.wrapping_add(s7.r.wrapping_add(s8.r)),
                    i: s0.i.wrapping_add(s7.i.wrapping_add(s8.i)),
                },
            );
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
            st(f, o1, csub(s5, s6));
            st(f, o4, cadd(s5, s6));
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
            st(f, o2, cadd(s11, s12));
            st(f, o3, csub(s11, s12));
        }
    }
}

/// `opus_fft_impl`: in-place FFT of the bit-reversed interleaved `(r, i)` pairs in
/// `f[..2 * nfft]`.
pub fn fft_impl(st: &FftState, f: &mut [i32]) {
    let f = &mut f[..2 * st.bitrev.len()];
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        unsafe { super::simd::fft_avx2(st, f) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: NEON is baseline on aarch64.
        unsafe { super::simd::fft_neon(st, f) };
        return;
    }
    fft_impl_scalar(st, f);
}

/// Scalar reference of [`fft_impl`].
pub fn fft_impl_scalar(st: &FftState, f: &mut [i32]) {
    let f = f[..2 * st.bitrev.len()].as_chunks_mut::<2>().0;
    let (stages, l) = fft_stages(st);
    for s in &stages[..l] {
        match s.p {
            2 => bfly2(f, s.m, s.n),
            4 => bfly4(f, s.fstride, s.m, s.n, s.mm),
            3 => bfly3(f, s.fstride, s.m, s.n, s.mm),
            5 => bfly5(f, s.fstride, s.m, s.n, s.mm),
            _ => unreachable!(),
        }
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
    let st = &FFT_STATES[shift];
    let yo = overlap >> 1;
    let t = &t[..n2];
    pre_rotate(
        &input[..stride * (n2 - 1) + 1],
        &mut out[yo..yo + n2],
        t,
        st.bitrev,
        stride,
    );
    fft_impl(st, &mut out[yo..yo + n2]);
    post_rotate(&mut out[yo..yo + n2], t);
    tdac_mirror(&mut out[..overlap], &window[..overlap]);
}

/// The TDAC mirroring at the start of the inverse MDCT output: `out` and `window` hold
/// `overlap` values, folded about the middle.
pub(super) fn tdac_mirror(out: &mut [i32], window: &[i16]) {
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        let done = unsafe { super::simd::tdac_mirror_avx2(out, window) };
        tdac_mirror_scalar(out, window, done);
        return;
    }
    tdac_mirror_scalar(out, window, 0);
}

/// Scalar reference of [`tdac_mirror`], from pair `i0` on.
pub fn tdac_mirror_scalar(out: &mut [i32], window: &[i16], i0: usize) {
    let overlap = out.len();
    let (lo, hi) = out.split_at_mut(overlap / 2);
    let (wlo, whi) = window[..overlap].split_at(overlap / 2);
    for (i, (x2, x1)) in lo.iter_mut().zip(hi.iter_mut().rev()).enumerate().skip(i0) {
        let (w1, w2) = (wlo[i] as i32, whi[overlap / 2 - 1 - i] as i32);
        let (v1, v2) = (*x1, *x2);
        *x2 = mul16_32_q15(w2, v2).wrapping_sub(mul16_32_q15(w1, v1));
        *x1 = mul16_32_q15(w1, v2).wrapping_add(mul16_32_q15(w2, v1));
    }
}

/// Pre-rotation of the inverse MDCT: the `stride`-interleaved coefficients `x` (`n2` of them)
/// rotated into the bit-reversed complex values `y` (`n2` values) the FFT takes.
pub(super) fn pre_rotate(x: &[i32], y: &mut [i32], t: &[i16], bitrev: &[i16], stride: usize) {
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        unsafe { super::simd::pre_rotate_avx2(x, y, t, bitrev, stride) };
        return;
    }
    pre_rotate_scalar(x, y, t, bitrev, stride, 0);
}

/// Scalar reference of [`pre_rotate`], from step `i0` on.
pub fn pre_rotate_scalar(
    x: &[i32],
    y: &mut [i32],
    t: &[i16],
    bitrev: &[i16],
    stride: usize,
    i0: usize,
) {
    let n2 = y.len();
    let n4 = n2 >> 1;
    let (t0, t1) = t[..n2].split_at(n4);
    for (i, &rev) in bitrev[..n4].iter().enumerate().skip(i0) {
        let rev = rev as usize;
        let (x1, x2) = (x[2 * stride * i], x[stride * (n2 - 1 - 2 * i)]);
        let yr = smul(x2, t0[i] as i32).wrapping_add(smul(x1, t1[i] as i32));
        let yi = smul(x1, t0[i] as i32).wrapping_sub(smul(x2, t1[i] as i32));
        // Real and imaginary parts swap because an FFT stands in for the IFFT.
        y[2 * rev + 1] = yr;
        y[2 * rev] = yi;
    }
}

/// Post-rotation and de-shuffling of the FFT output `y` (`n2` values), in place from both ends.
pub(super) fn post_rotate(y: &mut [i32], t: &[i16]) {
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        unsafe { super::simd::post_rotate_avx2(y, t) };
        return;
    }
    post_rotate_scalar(y, t);
}

/// Scalar reference of the MDCT post-rotation.
pub fn post_rotate_scalar(y: &mut [i32], t: &[i16]) {
    for i in 0..(y.len() / 2 + 1) >> 1 {
        post_rotate_one(y, t, i);
    }
}

/// Step `i` of the post-rotation: the pairs `i` from the front and from the back.
#[inline(always)]
pub(super) fn post_rotate_one(y: &mut [i32], t: &[i16], i: usize) {
    let n2 = y.len();
    let n4 = n2 >> 1;
    let yp0 = 2 * i;
    let yp1 = n2 - 2 - 2 * i;
    let (re, im) = (y[yp0 + 1], y[yp0]);
    let (t0, t1) = (t[i] as i32, t[n4 + i] as i32);
    let yr = smul(re, t0).wrapping_add(smul(im, t1));
    let yi = smul(re, t1).wrapping_sub(smul(im, t0));
    let (re, im) = (y[yp1 + 1], y[yp1]);
    y[yp0] = yr;
    y[yp1 + 1] = yi;
    let (t0, t1) = (t[n4 - i - 1] as i32, t[n2 - i - 1] as i32);
    let yr = smul(re, t0).wrapping_add(smul(im, t1));
    let yi = smul(re, t1).wrapping_sub(smul(im, t0));
    y[yp1] = yr;
    y[yp0 + 1] = yi;
}

/// `clt_mdct_forward`: forward MDCT of `input` (`N/2 + overlap` samples) into `stride`-interleaved
/// coefficients. `f` and `f2` are working space of at least `N/2` values.
#[allow(clippy::too_many_arguments)]
pub fn mdct_forward(
    input: &[i32],
    out: &mut [i32],
    window: &[i16],
    overlap: usize,
    shift: usize,
    stride: usize,
    f: &mut [i32],
    f2: &mut [i32],
) {
    let (t, n) = trig(shift);
    let st = &FFT_STATES[shift];
    let n2 = n >> 1;
    let (f, f2) = (&mut f[..n2], &mut f2[..n2]);
    fold_forward(input, f, window, overlap);
    pre_rotate_forward(f, f2, &t[..n2], st);
    fft_impl(st, f2);
    post_rotate_forward(f2, out, &t[..n2], stride);
}

/// [`mdct_forward`] as libopus writes it, the reference of the stages' SIMD versions.
#[cfg(test)]
pub fn mdct_forward_scalar(
    input: &[i32],
    out: &mut [i32],
    window: &[i16],
    overlap: usize,
    shift: usize,
    stride: usize,
) {
    let (t, n) = trig(shift);
    let st = &FFT_STATES[shift];
    let n2 = n >> 1;
    let (mut f, mut f2) = (vec![0i32; n2], vec![0i32; n2]);
    let q = (overlap + 3) >> 2;
    fold_window_scalar(input, &mut f, window, overlap);
    fold_copy_scalar(input, &mut f, overlap >> 1, q, (n >> 2) - q);
    pre_rotate_forward_scalar(&f, &mut f2, &t[..n2], st, 0);
    fft_impl_scalar(st, &mut f2);
    post_rotate_forward_scalar(&f2, out, &t[..n2], stride, 0, n >> 2);
}

/// Window, shuffle and fold of the forward MDCT: `f` (`N/2` values) from `input`.
fn fold_forward(input: &[i32], f: &mut [i32], window: &[i16], overlap: usize) {
    let n4 = f.len() >> 1;
    let q = (overlap + 3) >> 2;
    fold_window_scalar(input, f, window, overlap);
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        let done = unsafe { super::simd_enc::fold_copy_avx2(input, f, overlap >> 1, q, n4 - q) };
        fold_copy_scalar(input, f, overlap >> 1, done, n4 - q);
        return;
    }
    fold_copy_scalar(input, f, overlap >> 1, q, n4 - q);
}

/// The windowed steps of the forward MDCT's folding: the first and last `(overlap + 3) / 4`.
pub fn fold_window_scalar(input: &[i32], f: &mut [i32], window: &[i16], overlap: usize) {
    let n2 = f.len();
    let n4 = n2 >> 1;
    let q = (overlap + 3) >> 2;
    let h = overlap >> 1;
    for i in 0..q {
        let (xp1, xp2) = (h + 2 * i, n2 - 1 + h - 2 * i);
        let (w1, w2) = (window[h + 2 * i] as i32, window[h - 1 - 2 * i] as i32);
        f[2 * i] = mul16_32_q15(w2, input[xp1 + n2]).wrapping_add(mul16_32_q15(w1, input[xp2]));
        f[2 * i + 1] = mul16_32_q15(w1, input[xp1]).wrapping_sub(mul16_32_q15(w2, input[xp2 - n2]));
    }
    for i in n4 - q..n4 {
        let k = i - (n4 - q);
        let (xp1, xp2) = (h + 2 * i, n2 - 1 + h - 2 * i);
        let (w1, w2) = (window[2 * k] as i32, window[overlap - 1 - 2 * k] as i32);
        f[2 * i] = mul16_32_q15(w2, input[xp2]).wrapping_sub(mul16_32_q15(w1, input[xp1 - n2]));
        f[2 * i + 1] = mul16_32_q15(w2, input[xp1]).wrapping_add(mul16_32_q15(w1, input[xp2 + n2]));
    }
}

/// The unwindowed steps `i0..i1` of the forward MDCT's folding (`h` is half the overlap).
pub fn fold_copy_scalar(input: &[i32], f: &mut [i32], h: usize, i0: usize, i1: usize) {
    let n2 = f.len();
    for i in i0..i1 {
        f[2 * i] = input[n2 - 1 + h - 2 * i];
        f[2 * i + 1] = input[h + 2 * i];
    }
}

/// Pre-rotation of the forward MDCT: the folded `f` rotated, scaled and bit-reversed into
/// `f2`, the FFT input.
fn pre_rotate_forward(f: &[i32], f2: &mut [i32], t: &[i16], st: &FftState) {
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2; `bitrev` is a permutation.
        let done = unsafe { super::simd_enc::pre_rotate_forward_avx2(f, f2, t, st) };
        pre_rotate_forward_scalar(f, f2, t, st, done);
        return;
    }
    pre_rotate_forward_scalar(f, f2, t, st, 0);
}

/// Scalar reference of [`pre_rotate_forward`], from step `i0` on.
pub fn pre_rotate_forward_scalar(f: &[i32], f2: &mut [i32], t: &[i16], st: &FftState, i0: usize) {
    let n4 = f.len() >> 1;
    let (scale, scale_shift) = (st.scale, st.scale_shift - 1);
    for i in i0..n4 {
        let (t0, t1) = (t[i] as i32, t[n4 + i] as i32);
        let (re, im) = (f[2 * i], f[2 * i + 1]);
        let yr = smul(re, t0).wrapping_sub(smul(im, t1));
        let yi = smul(im, t0).wrapping_add(smul(re, t1));
        let rev = st.bitrev[i] as usize;
        f2[2 * rev] = pshr32(mul16_32_q16(scale, yr), scale_shift);
        f2[2 * rev + 1] = pshr32(mul16_32_q16(scale, yi), scale_shift);
    }
}

/// Post-rotation of the forward MDCT: the FFT output `f2` rotated into `stride`-interleaved
/// coefficients, from both ends of `out`.
fn post_rotate_forward(f2: &[i32], out: &mut [i32], t: &[i16], stride: usize) {
    let n4 = f2.len() >> 1;
    #[cfg(target_arch = "x86_64")]
    if stride == 1 && ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        let done = unsafe { super::simd_enc::post_rotate_forward_avx2(f2, out, t) };
        post_rotate_forward_scalar(f2, out, t, stride, done, n4 - done);
        return;
    }
    post_rotate_forward_scalar(f2, out, t, stride, 0, n4);
}

/// Scalar reference of [`post_rotate_forward`], steps `i0..i1`.
pub fn post_rotate_forward_scalar(
    f2: &[i32],
    out: &mut [i32],
    t: &[i16],
    stride: usize,
    i0: usize,
    i1: usize,
) {
    let n2 = f2.len();
    let n4 = n2 >> 1;
    for i in i0..i1 {
        let (t0, t1) = (t[i] as i32, t[n4 + i] as i32);
        let (re, im) = (f2[2 * i], f2[2 * i + 1]);
        out[2 * stride * i] = smul(im, t1).wrapping_sub(smul(re, t0));
        out[stride * (n2 - 1 - 2 * i)] = smul(re, t1).wrapping_add(smul(im, t0));
    }
}
