//! AVX2 and NEON kernels of the fixed-point Opus decoder (and the encoder parts it shares),
//! each bit-exact with the scalar reference it replaces.
//!
//! `MULT16_32_Q15(a, b)` (`(a * b) >> 15` on a 16-bit `a` and a 32-bit `b`, in 64 bits and
//! truncated to 32) is the core operation. AVX2 forms the 64-bit products with
//! `_mm256_mul_epi32` on the even and odd lanes and keeps bits 15..46 of each. NEON uses
//! `vqdmulhq_s32(b, a << 16)`, `(2 * b * (a << 16)) >> 32`, which is the same value and never
//! saturates because no operand here is `-32768` (the twiddles and constants are within
//! `±32767`).

#![allow(clippy::too_many_arguments)]

use std::sync::OnceLock;

use super::celt_tables::FFT_TWIDDLES;
use super::mdct::{FFT_STATES, FftState, Stage, fft_stages};

/// Index of `st` in [`FFT_STATES`].
fn state_index(st: &FftState) -> usize {
    match st.bitrev.len() {
        480 => 0,
        240 => 1,
        120 => 2,
        _ => 3,
    }
}

/// Twiddles of every radix pass with `m >= 4`, per FFT size and pass: for each run of four
/// butterflies `j..j + 4` and each multiplier `q` in `1..p`, eight `i32` built by `layout` from
/// the four twiddles `FFT_TWIDDLES[q * j * fstride]`.
type Twiddles = Vec<Vec<Vec<i32>>>;

fn build_twiddles(layout: fn(&[(i16, i16); 4]) -> [i32; 8]) -> Twiddles {
    FFT_STATES
        .iter()
        .map(|st| {
            let (stages, l) = fft_stages(st);
            stages[..l]
                .iter()
                .map(|s| {
                    let mut v = Vec::new();
                    if s.m >= 4 {
                        for j in (0..s.m).step_by(4) {
                            for q in 1..s.p {
                                let tw: [(i16, i16); 4] =
                                    std::array::from_fn(|k| FFT_TWIDDLES[q * (j + k) * s.fstride]);
                                v.extend_from_slice(&layout(&tw));
                            }
                        }
                    }
                    v
                })
                .collect()
        })
        .collect()
}

#[cfg(target_arch = "x86_64")]
fn twiddles_avx2() -> &'static Twiddles {
    static T: OnceLock<Twiddles> = OnceLock::new();
    // Interleaved (r, i) pairs, the layout of the data.
    T.get_or_init(|| {
        build_twiddles(|tw| std::array::from_fn(|k| [tw[k / 2].0, tw[k / 2].1][k & 1] as i32))
    })
}

#[cfg(target_arch = "aarch64")]
fn twiddles_neon() -> &'static Twiddles {
    static T: OnceLock<Twiddles> = OnceLock::new();
    // Four real parts then four imaginary parts, shifted for vqdmulhq_s32.
    T.get_or_init(|| {
        build_twiddles(|tw| {
            std::array::from_fn(|k| {
                let c = tw[k & 3];
                (if k < 4 { c.0 } else { c.1 } as i32) << 16
            })
        })
    })
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    /// `MULT16_32_Q15(c, x)` per 32-bit lane, `c` holding 16-bit values in every lane.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn mulq15(x: __m256i, c: __m256i) -> __m256i {
        let even = _mm256_mul_epi32(x, c);
        let odd = _mm256_mul_epi32(_mm256_srli_epi64::<32>(x), _mm256_srli_epi64::<32>(c));
        _mm256_blend_epi32::<0xAA>(_mm256_srli_epi64::<15>(even), _mm256_slli_epi64::<17>(odd))
    }

    /// [`mulq15`] with the same constant in every lane.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn mulq15c(x: __m256i, c: __m256i) -> __m256i {
        let even = _mm256_mul_epi32(x, c);
        let odd = _mm256_mul_epi32(_mm256_srli_epi64::<32>(x), c);
        _mm256_blend_epi32::<0xAA>(_mm256_srli_epi64::<15>(even), _mm256_slli_epi64::<17>(odd))
    }

    /// Four complex products `x * t` of interleaved `(r, i)` pairs, each part as `S_MUL` sums.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn cmul(x: __m256i, t: __m256i) -> __m256i {
        let ts = _mm256_shuffle_epi32::<0xB1>(t);
        let xi = _mm256_srli_epi64::<32>(x);
        let rr = _mm256_mul_epi32(x, t);
        let ii = _mm256_mul_epi32(xi, ts);
        let ri = _mm256_mul_epi32(x, ts);
        let ir = _mm256_mul_epi32(xi, t);
        let re = _mm256_sub_epi64(_mm256_srli_epi64::<15>(rr), _mm256_srli_epi64::<15>(ii));
        let im = _mm256_add_epi64(_mm256_srli_epi64::<15>(ri), _mm256_srli_epi64::<15>(ir));
        _mm256_blend_epi32::<0xAA>(re, _mm256_slli_epi64::<32>(im))
    }

    /// `(a.r + b.i, a.i - b.r)` and `(a.r - b.i, a.i + b.r)`: `a -/+ j * b`.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn rot_pair(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
        let sw = _mm256_shuffle_epi32::<0xB1>(b);
        let p = _mm256_add_epi32(a, sw);
        let m = _mm256_sub_epi32(a, sw);
        (
            _mm256_blend_epi32::<0xAA>(p, m),
            _mm256_blend_epi32::<0xAA>(m, p),
        )
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn ld(p: *const i32) -> __m256i {
        unsafe { _mm256_loadu_si256(p as *const __m256i) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn st(p: *mut i32, v: __m256i) {
        unsafe { _mm256_storeu_si256(p as *mut __m256i, v) }
    }
}

/// AVX2 [`super::mdct::fft_impl`].
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn fft_avx2(st: &FftState, f: &mut [i32]) {
    let tw = &twiddles_avx2()[state_index(st)];
    let (stages, l) = fft_stages(st);
    assert!(f.len() >= 2 * st.bitrev.len());
    for (k, s) in stages[..l].iter().enumerate() {
        // Safety: every pass stays within the `nfft` pairs of `f`, and `tw[k]` holds the
        // twiddles of pass `k`.
        unsafe {
            match (s.p, s.m) {
                (4, 1) => fft4_m1_avx2(f, s.n),
                (2, _) => fft2_avx2(f, s.n),
                (4, _) => fft4_avx2(f.as_mut_ptr(), &tw[k], s),
                (3, _) => fft3_avx2(f.as_mut_ptr(), &tw[k], s),
                (5, _) => fft5_avx2(f.as_mut_ptr(), &tw[k], s),
                _ => unreachable!(),
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fft4_m1_avx2(f: &mut [i32], n: usize) {
    use avx2::*;
    use std::arch::x86_64::*;
    let neg3 = _mm256_setr_epi32(1, 1, 1, -1, 1, 1, 1, -1);
    let pairs = n / 2;
    let p = f.as_mut_ptr();
    for g in 0..pairs {
        // Safety: groups 2g and 2g + 1 are within the n groups of 8 values.
        unsafe {
            let q = p.add(16 * g);
            let g0 = ld(q);
            let g1 = ld(q.add(8));
            let x = _mm256_permute2x128_si256::<0x20>(g0, g1);
            let y = _mm256_permute2x128_si256::<0x31>(g0, g1);
            let a = _mm256_add_epi32(x, y);
            let b = _mm256_sub_epi32(x, y);
            let c = _mm256_unpacklo_epi64(a, b);
            let d = _mm256_unpackhi_epi64(a, b);
            let e = _mm256_sign_epi32(_mm256_shuffle_epi32::<0b10_11_01_00>(d), neg3);
            let o01 = _mm256_add_epi32(c, e);
            let o23 = _mm256_sub_epi32(c, e);
            st(q, _mm256_permute2x128_si256::<0x20>(o01, o23));
            st(q.add(8), _mm256_permute2x128_si256::<0x31>(o01, o23));
        }
    }
    if n & 1 != 0 {
        super::mdct::bfly4_m1(f[16 * pairs..16 * pairs + 8].as_chunks_mut::<2>().0, 1);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fft2_avx2(f: &mut [i32], n: usize) {
    use avx2::*;
    use std::arch::x86_64::*;
    let tw = _mm256_set1_epi32(23170);
    let sign = _mm256_setr_epi32(1, 1, 1, 1, 1, -1, -1, -1);
    assert!(f.len() >= 16 * n);
    let p = f.as_mut_ptr();
    for i in 0..n {
        // Safety: group i spans values 16i..16i + 16, checked above.
        unsafe {
            let q = p.add(16 * i);
            let a = ld(q);
            let b = ld(q.add(8));
            let sw = _mm256_shuffle_epi32::<0xB1>(b);
            let pl = _mm256_add_epi32(b, sw);
            let mi = _mm256_sub_epi32(b, sw);
            // [b0.r, b0.i, b1.r + b1.i, b1.i - b1.r, b2.i, -b2.r, b3.i - b3.r, -(b3.i + b3.r)]
            let v = _mm256_blend_epi32::<0x84>(b, pl);
            let v = _mm256_blend_epi32::<0x48>(v, mi);
            let v = _mm256_sign_epi32(_mm256_blend_epi32::<0x30>(v, sw), sign);
            let t = _mm256_blend_epi32::<0xCC>(v, mulq15c(v, tw));
            st(q, _mm256_add_epi32(a, t));
            st(q.add(8), _mm256_sub_epi32(a, t));
        }
    }
}

/// # Safety
///
/// `f` must hold the `s.n * s.mm` pairs of the pass and `tw` its twiddles.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fft4_avx2(f: *mut i32, tw: &[i32], s: &Stage) {
    use avx2::*;
    use std::arch::x86_64::*;
    let m = s.m;
    for i in 0..s.n {
        for (q, t) in tw.as_chunks::<24>().0.iter().enumerate() {
            // Safety: per the function contract.
            unsafe {
                let p0 = f.add(2 * (i * s.mm + 4 * q));
                let (p1, p2, p3) = (p0.add(2 * m), p0.add(4 * m), p0.add(6 * m));
                let s0 = cmul(ld(p1), ld(t.as_ptr()));
                let s1 = cmul(ld(p2), ld(t.as_ptr().add(8)));
                let s2 = cmul(ld(p3), ld(t.as_ptr().add(16)));
                let f0 = ld(p0);
                let s5 = _mm256_sub_epi32(f0, s1);
                let f0 = _mm256_add_epi32(f0, s1);
                let s3 = _mm256_add_epi32(s0, s2);
                let s4 = _mm256_sub_epi32(s0, s2);
                st(p2, _mm256_sub_epi32(f0, s3));
                st(p0, _mm256_add_epi32(f0, s3));
                let (a, b) = rot_pair(s5, s4);
                st(p1, a);
                st(p3, b);
            }
        }
    }
}

/// # Safety
///
/// As [`fft4_avx2`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fft3_avx2(f: *mut i32, tw: &[i32], s: &Stage) {
    use avx2::*;
    use std::arch::x86_64::*;
    let m = s.m;
    let epi3 = _mm256_set1_epi32(-28378);
    for i in 0..s.n {
        for (q, t) in tw.as_chunks::<16>().0.iter().enumerate() {
            // Safety: per the function contract.
            unsafe {
                let p0 = f.add(2 * (i * s.mm + 4 * q));
                let (p1, p2) = (p0.add(2 * m), p0.add(4 * m));
                let s1 = cmul(ld(p1), ld(t.as_ptr()));
                let s2 = cmul(ld(p2), ld(t.as_ptr().add(8)));
                let s3 = _mm256_add_epi32(s1, s2);
                let s0 = mulq15c(_mm256_sub_epi32(s1, s2), epi3);
                let f0 = ld(p0);
                let fm = _mm256_sub_epi32(f0, _mm256_srai_epi32::<1>(s3));
                st(p0, _mm256_add_epi32(f0, s3));
                let (a, b) = rot_pair(fm, s0);
                st(p2, a);
                st(p1, b);
            }
        }
    }
}

/// # Safety
///
/// As [`fft4_avx2`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fft5_avx2(f: *mut i32, tw: &[i32], s: &Stage) {
    use avx2::*;
    use std::arch::x86_64::*;
    let m = s.m;
    let yar = _mm256_set1_epi32(10126);
    let yai = _mm256_set1_epi32(-31164);
    let ybr = _mm256_set1_epi32(-26510);
    let ybi = _mm256_set1_epi32(-19261);
    for i in 0..s.n {
        for (q, t) in tw.as_chunks::<32>().0.iter().enumerate() {
            // Safety: per the function contract.
            unsafe {
                let p0 = f.add(2 * (i * s.mm + 4 * q));
                let (p1, p2, p3, p4) = (p0.add(2 * m), p0.add(4 * m), p0.add(6 * m), p0.add(8 * m));
                let s0 = ld(p0);
                let s1 = cmul(ld(p1), ld(t.as_ptr()));
                let s2 = cmul(ld(p2), ld(t.as_ptr().add(8)));
                let s3 = cmul(ld(p3), ld(t.as_ptr().add(16)));
                let s4 = cmul(ld(p4), ld(t.as_ptr().add(24)));
                let s7 = _mm256_add_epi32(s1, s4);
                let s10 = _mm256_sub_epi32(s1, s4);
                let s8 = _mm256_add_epi32(s2, s3);
                let s9 = _mm256_sub_epi32(s2, s3);
                st(p0, _mm256_add_epi32(s0, _mm256_add_epi32(s7, s8)));
                let s5 = _mm256_add_epi32(s0, _mm256_add_epi32(mulq15c(s7, yar), mulq15c(s8, ybr)));
                let qv = _mm256_add_epi32(mulq15c(s10, yai), mulq15c(s9, ybi));
                // s6 = (q.i, -q.r): out1 = s5 - s6, out4 = s5 + s6.
                let (o4, o1) = rot_pair(s5, qv);
                st(p1, o1);
                st(p4, o4);
                let s11 =
                    _mm256_add_epi32(s0, _mm256_add_epi32(mulq15c(s7, ybr), mulq15c(s8, yar)));
                let w = _mm256_sub_epi32(mulq15c(s9, yai), mulq15c(s10, ybi));
                // s12 = (w.i, -w.r): out2 = s11 + s12, out3 = s11 - s12.
                let (o2, o3) = rot_pair(s11, w);
                st(p2, o2);
                st(p3, o3);
            }
        }
    }
}

/// AVX2 [`super::mdct::post_rotate_scalar`].
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn post_rotate_avx2(y: &mut [i32], t: &[i16]) {
    use avx2::*;
    use std::arch::x86_64::*;
    let n2 = y.len();
    let n4 = n2 >> 1;
    let count = (n4 + 1) >> 1;
    let blocks = count / 4;
    assert!(t.len() >= n2 && n2.is_multiple_of(4));
    let rev = _mm256_setr_epi32(6, 7, 4, 5, 2, 3, 0, 1);
    let rev16 = _mm_setr_epi8(6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9);
    let yp = y.as_mut_ptr();
    let tp = t.as_ptr();
    for b in 0..blocks {
        let i = 4 * b;
        // Safety: the front values 2i..2i + 8 and the back values n2 - 2i - 8..n2 - 2i are
        // in `y` and disjoint (i + 4 <= n4 / 2); the twiddle reads stay below n2.
        unsafe {
            // Twiddles of the front (i + k) and back (n4 - 1 - i - k) halves, interleaved.
            let lo = |a: usize, z: usize| {
                let f = _mm_loadl_epi64(tp.add(a) as *const __m128i);
                let r = _mm_shuffle_epi8(_mm_loadl_epi64(tp.add(z - 4) as *const __m128i), rev16);
                _mm256_cvtepi16_epi32(_mm_unpacklo_epi16(f, r))
            };
            let t0 = lo(i, n4 - i);
            let t1 = lo(n4 + i, n2 - i);
            let pf = yp.add(2 * i);
            let pb = yp.add(n2 - 2 * i - 8);
            let fv = ld(pf);
            let bv = _mm256_permutevar8x32_epi32(ld(pb), rev);
            // Lanes alternate front and back: [k=0 front, k=0 back, k=1 front, ...].
            let xim = _mm256_blend_epi32::<0xAA>(fv, _mm256_slli_epi64::<32>(bv));
            let xre = _mm256_blend_epi32::<0xAA>(_mm256_srli_epi64::<32>(fv), bv);
            let yr = _mm256_add_epi32(mulq15(xre, t0), mulq15(xim, t1));
            let yi = _mm256_sub_epi32(mulq15(xre, t1), mulq15(xim, t0));
            st(pf, _mm256_blend_epi32::<0xAA>(yr, yi));
            let bo = _mm256_shuffle_epi32::<0xB1>(_mm256_blend_epi32::<0xAA>(yi, yr));
            st(pb, _mm256_permutevar8x32_epi32(bo, rev));
        }
    }
    for i in 4 * blocks..count {
        super::mdct::post_rotate_one(y, t, i);
    }
}

/// NEON [`super::mdct::fft_impl`].
///
/// # Safety
///
/// NEON must be available (it is on every aarch64 CPU).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn fft_neon(st: &FftState, f: &mut [i32]) {
    let tw = &twiddles_neon()[state_index(st)];
    let (stages, l) = fft_stages(st);
    assert!(f.len() >= 2 * st.bitrev.len());
    for (k, s) in stages[..l].iter().enumerate() {
        match (s.p, s.m) {
            (4, 1) => super::mdct::bfly4_m1(f.as_chunks_mut::<2>().0, s.n),
            (2, _) => super::mdct::bfly2(f.as_chunks_mut::<2>().0, s.m, s.n),
            // Safety: every pass stays within the `nfft` pairs of `f`, and `tw[k]` holds the
            // twiddles of pass `k`.
            (4, _) => unsafe { fft4_neon(f.as_mut_ptr(), &tw[k], s) },
            (3, _) => unsafe { fft3_neon(f.as_mut_ptr(), &tw[k], s) },
            (5, _) => unsafe { fft5_neon(f.as_mut_ptr(), &tw[k], s) },
            _ => unreachable!(),
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// Four complex values as separate real and imaginary vectors.
    #[derive(Clone, Copy)]
    pub struct C4 {
        pub r: int32x4_t,
        pub i: int32x4_t,
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn ld(p: *const i32) -> C4 {
        let v = unsafe { vld2q_s32(p) };
        C4 { r: v.0, i: v.1 }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn st(p: *mut i32, c: C4) {
        unsafe { vst2q_s32(p, int32x4x2_t(c.r, c.i)) }
    }

    /// Twiddles at `p`: four real parts then four imaginary parts, pre-shifted by 16.
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn ldt(p: *const i32) -> C4 {
        unsafe {
            C4 {
                r: vld1q_s32(p),
                i: vld1q_s32(p.add(4)),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub fn add(a: C4, b: C4) -> C4 {
        C4 {
            r: vaddq_s32(a.r, b.r),
            i: vaddq_s32(a.i, b.i),
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub fn sub(a: C4, b: C4) -> C4 {
        C4 {
            r: vsubq_s32(a.r, b.r),
            i: vsubq_s32(a.i, b.i),
        }
    }

    /// `S_MUL` of both parts by a constant pre-shifted by 16.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn mulc(a: C4, c: int32x4_t) -> C4 {
        C4 {
            r: vqdmulhq_s32(a.r, c),
            i: vqdmulhq_s32(a.i, c),
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub fn cmul(a: C4, t: C4) -> C4 {
        C4 {
            r: vsubq_s32(vqdmulhq_s32(a.r, t.r), vqdmulhq_s32(a.i, t.i)),
            i: vaddq_s32(vqdmulhq_s32(a.r, t.i), vqdmulhq_s32(a.i, t.r)),
        }
    }

    /// `a - j * b` and `a + j * b`.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn rot_pair(a: C4, b: C4) -> (C4, C4) {
        (
            C4 {
                r: vaddq_s32(a.r, b.i),
                i: vsubq_s32(a.i, b.r),
            },
            C4 {
                r: vsubq_s32(a.r, b.i),
                i: vaddq_s32(a.i, b.r),
            },
        )
    }
}

/// # Safety
///
/// As [`fft4_avx2`].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fft4_neon(f: *mut i32, tw: &[i32], s: &Stage) {
    use neon::*;
    let m = s.m;
    for i in 0..s.n {
        for (q, t) in tw.as_chunks::<24>().0.iter().enumerate() {
            // Safety: per the function contract.
            unsafe {
                let p0 = f.add(2 * (i * s.mm + 4 * q));
                let (p1, p2, p3) = (p0.add(2 * m), p0.add(4 * m), p0.add(6 * m));
                let s0 = cmul(ld(p1), ldt(t.as_ptr()));
                let s1 = cmul(ld(p2), ldt(t.as_ptr().add(8)));
                let s2 = cmul(ld(p3), ldt(t.as_ptr().add(16)));
                let f0 = ld(p0);
                let s5 = sub(f0, s1);
                let f0 = add(f0, s1);
                let s3 = add(s0, s2);
                let s4 = sub(s0, s2);
                st(p2, sub(f0, s3));
                st(p0, add(f0, s3));
                let (a, b) = rot_pair(s5, s4);
                st(p1, a);
                st(p3, b);
            }
        }
    }
}

/// # Safety
///
/// As [`fft4_avx2`].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fft3_neon(f: *mut i32, tw: &[i32], s: &Stage) {
    use neon::*;
    use std::arch::aarch64::*;
    let m = s.m;
    let epi3 = vdupq_n_s32(-28378 << 16);
    for i in 0..s.n {
        for (q, t) in tw.as_chunks::<16>().0.iter().enumerate() {
            // Safety: per the function contract.
            unsafe {
                let p0 = f.add(2 * (i * s.mm + 4 * q));
                let (p1, p2) = (p0.add(2 * m), p0.add(4 * m));
                let s1 = cmul(ld(p1), ldt(t.as_ptr()));
                let s2 = cmul(ld(p2), ldt(t.as_ptr().add(8)));
                let s3 = add(s1, s2);
                let s0 = mulc(sub(s1, s2), epi3);
                let f0 = ld(p0);
                let fm = C4 {
                    r: vsubq_s32(f0.r, vshrq_n_s32::<1>(s3.r)),
                    i: vsubq_s32(f0.i, vshrq_n_s32::<1>(s3.i)),
                };
                st(p0, add(f0, s3));
                let (a, b) = rot_pair(fm, s0);
                st(p2, a);
                st(p1, b);
            }
        }
    }
}

/// # Safety
///
/// As [`fft4_avx2`].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fft5_neon(f: *mut i32, tw: &[i32], s: &Stage) {
    use neon::*;
    use std::arch::aarch64::*;
    let m = s.m;
    let yar = vdupq_n_s32(10126 << 16);
    let yai = vdupq_n_s32(-31164 << 16);
    let ybr = vdupq_n_s32(-26510 << 16);
    let ybi = vdupq_n_s32(-19261 << 16);
    for i in 0..s.n {
        for (q, t) in tw.as_chunks::<32>().0.iter().enumerate() {
            // Safety: per the function contract.
            unsafe {
                let p0 = f.add(2 * (i * s.mm + 4 * q));
                let (p1, p2, p3, p4) = (p0.add(2 * m), p0.add(4 * m), p0.add(6 * m), p0.add(8 * m));
                let s0 = ld(p0);
                let s1 = cmul(ld(p1), ldt(t.as_ptr()));
                let s2 = cmul(ld(p2), ldt(t.as_ptr().add(8)));
                let s3 = cmul(ld(p3), ldt(t.as_ptr().add(16)));
                let s4 = cmul(ld(p4), ldt(t.as_ptr().add(24)));
                let s7 = add(s1, s4);
                let s10 = sub(s1, s4);
                let s8 = add(s2, s3);
                let s9 = sub(s2, s3);
                st(p0, add(s0, add(s7, s8)));
                let s5 = add(s0, add(mulc(s7, yar), mulc(s8, ybr)));
                let qv = add(mulc(s10, yai), mulc(s9, ybi));
                let (o4, o1) = rot_pair(s5, qv);
                st(p1, o1);
                st(p4, o4);
                let s11 = add(s0, add(mulc(s7, ybr), mulc(s8, yar)));
                let w = sub(mulc(s9, yai), mulc(s10, ybi));
                let (o2, o3) = rot_pair(s11, w);
                st(p2, o2);
                st(p3, o3);
            }
        }
    }
}

/// AVX2 [`super::mdct::tdac_mirror_scalar`] over whole vectors of eight pairs; returns how
/// many pairs it did.
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn tdac_mirror_avx2(out: &mut [i32], window: &[i16]) -> usize {
    use avx2::*;
    use std::arch::x86_64::*;
    let overlap = out.len();
    let half = overlap / 2;
    assert!(window.len() >= overlap);
    let rev = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    let op = out.as_mut_ptr();
    let wp = window.as_ptr();
    let mut i = 0;
    while i + 8 <= half {
        // Safety: the front values i..i + 8 and the back values overlap - 8 - i..overlap - i
        // are in `out` and disjoint (i + 8 <= overlap / 2), and so are the window reads.
        unsafe {
            let w16 = |p: *const i16| _mm256_cvtepi16_epi32(_mm_loadu_si128(p as *const __m128i));
            let back = overlap - 8 - i;
            let x2 = ld(op.add(i));
            let x1 = _mm256_permutevar8x32_epi32(ld(op.add(back)), rev);
            let w1 = w16(wp.add(i));
            let w2 = _mm256_permutevar8x32_epi32(w16(wp.add(back)), rev);
            let a = _mm256_sub_epi32(mulq15(x2, w2), mulq15(x1, w1));
            let b = _mm256_add_epi32(mulq15(x2, w1), mulq15(x1, w2));
            st(op.add(i), a);
            st(op.add(back), _mm256_permutevar8x32_epi32(b, rev));
        }
        i += 8;
    }
    i
}

/// AVX2 `SATURATE(x, lim)` in place.
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn saturate_avx2(x: &mut [i32], lim: i32) {
    use std::arch::x86_64::*;
    let (hi, lo) = (_mm256_set1_epi32(lim), _mm256_set1_epi32(-lim));
    let (chunks, rest) = x.as_chunks_mut::<8>();
    for c in chunks {
        // Safety: `c` holds 8 values.
        unsafe {
            let v = _mm256_loadu_si256(c.as_ptr() as *const __m256i);
            let v = _mm256_max_epi32(_mm256_min_epi32(v, hi), lo);
            _mm256_storeu_si256(c.as_mut_ptr() as *mut __m256i, v);
        }
    }
    for v in rest {
        *v = (*v).clamp(-lim, lim);
    }
}

/// AVX2 [`super::mdct::pre_rotate`].
///
/// # Safety
///
/// The CPU must support AVX2, and the first `y.len() / 2` entries of `bitrev` must be below
/// `y.len() / 2` (they are a permutation in every [`FftState`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn pre_rotate_avx2(x: &[i32], y: &mut [i32], t: &[i16], bitrev: &[i16], stride: usize) {
    use std::arch::x86_64::*;
    let n2 = y.len();
    let n4 = n2 >> 1;
    assert!(
        n2.is_multiple_of(2) && x.len() > stride * (n2 - 1) && t.len() >= n2 && bitrev.len() >= n4
    );
    let odd = _mm256_setr_epi32(7, 7, 5, 5, 3, 3, 1, 1);
    let (xp, tp) = (x.as_ptr(), t.as_ptr());
    let s = stride as i32;
    let step1 = _mm_setr_epi32(0, 2 * s, 4 * s, 6 * s);
    let step2 = _mm_setr_epi32(0, -2 * s, -4 * s, -6 * s);
    let mut i = 0usize;
    while i + 4 <= n4 {
        // Safety: the reads x[2 stride i..] and x[stride (n2 - 1 - 2i)..] for these four
        // steps stay in `x` and the twiddle reads below n2.
        let v = unsafe {
            // x1 = x[2 stride (i + k)] and x2 = x[stride (n2 - 1 - 2 (i + k))] in the even lanes.
            let (x1, x2) = if stride == 1 {
                (
                    _mm256_loadu_si256(xp.add(2 * i) as *const __m256i),
                    _mm256_permutevar8x32_epi32(
                        _mm256_loadu_si256(xp.add(n2 - 8 - 2 * i) as *const __m256i),
                        odd,
                    ),
                )
            } else {
                let b1 = _mm_add_epi32(_mm_set1_epi32(2 * s * i as i32), step1);
                let b2 = _mm_add_epi32(_mm_set1_epi32(s * (n2 - 1 - 2 * i) as i32), step2);
                (
                    _mm256_cvtepi32_epi64(_mm_i32gather_epi32::<4>(xp, b1)),
                    _mm256_cvtepi32_epi64(_mm_i32gather_epi32::<4>(xp, b2)),
                )
            };
            let t0 = _mm256_cvtepi16_epi64(_mm_loadl_epi64(tp.add(i) as *const __m128i));
            let t1 = _mm256_cvtepi16_epi64(_mm_loadl_epi64(tp.add(n4 + i) as *const __m128i));
            let yr = _mm256_add_epi64(
                _mm256_srli_epi64::<15>(_mm256_mul_epi32(x2, t0)),
                _mm256_srli_epi64::<15>(_mm256_mul_epi32(x1, t1)),
            );
            let yi = _mm256_sub_epi64(
                _mm256_srli_epi64::<15>(_mm256_mul_epi32(x1, t0)),
                _mm256_srli_epi64::<15>(_mm256_mul_epi32(x2, t1)),
            );
            _mm256_blend_epi32::<0xAA>(yi, _mm256_slli_epi64::<32>(yr))
        };
        // Safety: per the function contract every pair 2 bitrev[i + k] is within `y`.
        unsafe {
            let yp = y.as_mut_ptr();
            let r = |k: usize| 2 * *bitrev.get_unchecked(i + k) as usize;
            debug_assert!((0..4).all(|k| r(k) + 2 <= n2));
            _mm_storel_epi64(yp.add(r(0)) as *mut __m128i, _mm256_castsi256_si128(v));
            _mm_storeh_pd(
                yp.add(r(1)) as *mut f64,
                _mm_castsi128_pd(_mm256_castsi256_si128(v)),
            );
            let hi = _mm256_extracti128_si256::<1>(v);
            _mm_storel_epi64(yp.add(r(2)) as *mut __m128i, hi);
            _mm_storeh_pd(yp.add(r(3)) as *mut f64, _mm_castsi128_pd(hi));
        }
        i += 4;
    }
    super::mdct::pre_rotate_scalar(x, y, t, bitrev, stride, i);
}

// ----- 16-bit band kernels ------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx2_i16 {
    use std::arch::x86_64::*;

    /// The 32-bit products `x * g` of 16-bit lanes, in `unpacklo`/`unpackhi` order.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn mul32(x: __m256i, g: __m256i) -> (__m256i, __m256i) {
        let lo = _mm256_mullo_epi16(x, g);
        let hi = _mm256_mulhi_epi16(x, g);
        (_mm256_unpacklo_epi16(lo, hi), _mm256_unpackhi_epi16(lo, hi))
    }

    /// The low 16 bits of the 32-bit lanes of `a` and `b` (as [`mul32`] split them), in
    /// order: a truncating pack.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn pack_trunc(a: __m256i, b: __m256i) -> __m256i {
        let a = _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(a));
        let b = _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(b));
        _mm256_packs_epi32(a, b)
    }

    /// `PSHR32(MULT16_16(g, x), sh)` truncated to 16 bits; `round` is `1 << (sh - 1)` (or 0)
    /// and `cnt` holds `sh`.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn pshr_mul(x: __m256i, g: __m256i, round: __m256i, cnt: __m128i) -> __m256i {
        let (p0, p1) = mul32(x, g);
        pack_trunc(
            _mm256_sra_epi32(_mm256_add_epi32(p0, round), cnt),
            _mm256_sra_epi32(_mm256_add_epi32(p1, round), cnt),
        )
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn ld(p: *const i16) -> __m256i {
        unsafe { _mm256_loadu_si256(p as *const __m256i) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn st(p: *mut i16, v: __m256i) {
        unsafe { _mm256_storeu_si256(p as *mut __m256i, v) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn ld8(p: *const i16) -> __m256i {
        unsafe { _mm256_castsi128_si256(_mm_loadu_si128(p as *const __m128i)) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn st8(p: *mut i16, v: __m256i) {
        unsafe { _mm_storeu_si128(p as *mut __m128i, _mm256_castsi256_si128(v)) }
    }

    /// Applies the lane-wise `f` to all of `x` and `y` (equal lengths) in place: whole
    /// vectors, then the last 16 lanes once more from their original values (below 16 lanes,
    /// the first and the last 8). Returns false, doing nothing, below 8 lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn map2(
        x: &mut [i16],
        y: &mut [i16],
        f: impl Fn(__m256i, __m256i) -> (__m256i, __m256i),
    ) -> bool {
        let n = x.len();
        assert!(y.len() == n);
        if n < 8 {
            return false;
        }
        if n < 16 {
            // Safety: both accesses are to 8 values at or below n - 8.
            unsafe {
                let (xp, yp) = (x.as_mut_ptr(), y.as_mut_ptr());
                let a = f(ld8(xp), ld8(yp));
                let b = f(ld8(xp.add(n - 8)), ld8(yp.add(n - 8)));
                st8(xp, a.0);
                st8(yp, a.1);
                st8(xp.add(n - 8), b.0);
                st8(yp.add(n - 8), b.1);
            }
            return true;
        }
        // Safety: every access is to 16 values at or below n - 16.
        unsafe {
            let (xp, yp) = (x.as_mut_ptr(), y.as_mut_ptr());
            let last = f(ld(xp.add(n - 16)), ld(yp.add(n - 16)));
            let mut i = 0;
            while i + 16 < n {
                let (u, v) = f(ld(xp.add(i)), ld(yp.add(i)));
                st(xp.add(i), u);
                st(yp.add(i), v);
                i += 16;
            }
            st(xp.add(n - 16), last.0);
            st(yp.add(n - 16), last.1);
        }
        true
    }

    /// [`map2`] for one slice.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn map1(x: &mut [i16], f: impl Fn(__m256i) -> __m256i) -> bool {
        let n = x.len();
        if n < 8 {
            return false;
        }
        if n < 16 {
            // Safety: both accesses are to 8 values at or below n - 8.
            unsafe {
                let xp = x.as_mut_ptr();
                let a = f(ld8(xp));
                let b = f(ld8(xp.add(n - 8)));
                st8(xp, a);
                st8(xp.add(n - 8), b);
            }
            return true;
        }
        // Safety: every access is to 16 values at or below n - 16.
        unsafe {
            let xp = x.as_mut_ptr();
            let last = f(ld(xp.add(n - 16)));
            let mut i = 0;
            while i + 16 < n {
                st(xp.add(i), f(ld(xp.add(i))));
                i += 16;
            }
            st(xp.add(n - 16), last);
        }
        true
    }
}

/// AVX2 [`super::vq::scale_shr_scalar`]; returns false, doing nothing, below 8 values.
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn scale_shr_avx2(x: &mut [i16], g: i32, sh: i32) -> bool {
    use avx2_i16::*;
    use std::arch::x86_64::*;
    let gv = _mm256_set1_epi16(g as i16);
    let round = _mm256_set1_epi32((1i32 << sh) >> 1);
    let cnt = _mm_cvtsi32_si128(sh);
    map1(x, |v| pshr_mul(v, gv, round, cnt))
}

/// AVX2 [`super::bands::mul_q15_scalar`] (`out` and `x` of equal lengths); returns false,
/// doing nothing, below 8 values.
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn mul_q15_avx2(out: &mut [i16], x: &[i16], g: i32) -> bool {
    use avx2_i16::*;
    use std::arch::x86_64::*;
    let gv = _mm256_set1_epi16(g as i16);
    // (x * g) >> 15 in 16 bits: the high half doubled, plus bit 15 of the low half.
    let f = |xv: __m256i| {
        let hi = _mm256_slli_epi16::<1>(_mm256_mulhi_epi16(xv, gv));
        let lo = _mm256_srli_epi16::<15>(_mm256_mullo_epi16(xv, gv));
        _mm256_or_si256(hi, lo)
    };
    let n = x.len();
    assert!(out.len() == n);
    if n < 8 {
        return false;
    }
    if n < 16 {
        // Safety: both accesses are to 8 values at or below n - 8.
        unsafe {
            st8(out.as_mut_ptr(), f(ld8(x.as_ptr())));
            st8(out.as_mut_ptr().add(n - 8), f(ld8(x.as_ptr().add(n - 8))));
        }
        return true;
    }
    // Safety: every access is to 16 values at or below n - 16; the last vector may repeat
    // some outputs, from the unchanged input.
    unsafe {
        let mut i = 0;
        while i + 16 < n {
            st(out.as_mut_ptr().add(i), f(ld(x.as_ptr().add(i))));
            i += 16;
        }
        st(out.as_mut_ptr().add(n - 16), f(ld(x.as_ptr().add(n - 16))));
    }
    true
}

/// AVX2 final loop of [`super::bands`] `stereo_merge` (`x` and `y` of equal lengths);
/// returns false, doing nothing, below 8 values.
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn stereo_merge_avx2(
    x: &mut [i16],
    y: &mut [i16],
    mid: i32,
    lgain: i32,
    lsh: i32,
    rgain: i32,
    rsh: i32,
) -> bool {
    use avx2_i16::*;
    use std::arch::x86_64::*;
    let midv = _mm256_set1_epi16(mid as i16);
    let (lg, rg) = (
        _mm256_set1_epi16(lgain as i16),
        _mm256_set1_epi16(rgain as i16),
    );
    let (lr, rr) = (
        _mm256_set1_epi32((1i32 << lsh) >> 1),
        _mm256_set1_epi32((1i32 << rsh) >> 1),
    );
    let (lc, rc) = (_mm_cvtsi32_si128(lsh), _mm_cvtsi32_si128(rsh));
    map2(x, y, |xv, r| {
        // MULT16_16_P15(mid, x) is pmulhrsw.
        let l = _mm256_mulhrs_epi16(xv, midv);
        (
            pshr_mul(_mm256_sub_epi16(l, r), lg, lr, lc),
            pshr_mul(_mm256_add_epi16(l, r), rg, rr, rc),
        )
    })
}

/// AVX2 [`super::bands::haar1_scalar`] for strides 1, 2, 4, 8 and multiples of 16 over the
/// first values in whole vectors; returns how many values it did (0 for other strides).
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn haar1_avx2(x: &mut [i16], n0: usize, stride: usize) -> usize {
    use avx2_i16::*;
    use std::arch::x86_64::*;
    let total = (n0 >> 1) * 2 * stride;
    assert!(x.len() >= total);
    let c = _mm256_set1_epi16(23170);
    let round = _mm256_set1_epi32(16384);
    // tmp1 = 23170 a, tmp2 = 23170 b; a' = PSHR32(tmp1 + tmp2, 15), b' = PSHR32(tmp1 - tmp2, 15).
    let pair = |a: __m256i, b: __m256i| -> (__m256i, __m256i) {
        let (a0, a1) = mul32(a, c);
        let (b0, b1) = mul32(b, c);
        let f = |v: __m256i| _mm256_srai_epi32::<15>(_mm256_add_epi32(v, round));
        (
            pack_trunc(f(_mm256_add_epi32(a0, b0)), f(_mm256_add_epi32(a1, b1))),
            pack_trunc(f(_mm256_sub_epi32(a0, b0)), f(_mm256_sub_epi32(a1, b1))),
        )
    };
    if stride.is_multiple_of(16) {
        for blk in x[..total].chunks_exact_mut(2 * stride) {
            let (a, b) = blk.split_at_mut(stride);
            for (av, bv) in a
                .as_chunks_mut::<16>()
                .0
                .iter_mut()
                .zip(b.as_chunks_mut::<16>().0)
            {
                // Safety: `av` and `bv` hold 16 values each.
                unsafe {
                    let (na, nb) = pair(ld(av.as_ptr()), ld(bv.as_ptr()));
                    st(av.as_mut_ptr(), na);
                    st(bv.as_mut_ptr(), nb);
                }
            }
        }
        return total;
    }
    if !matches!(stride, 1 | 2 | 4 | 8) {
        return 0;
    }
    // Within a vector, lanes alternate between runs of `stride` a values and b values: swap
    // each run with its partner, then merge the results by lane parity.
    let (swap, amask) = match stride {
        1 => (
            _mm256_setr_epi8(
                2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5, 10,
                11, 8, 9, 14, 15, 12, 13,
            ),
            _mm256_set1_epi32(0x0000_ffff),
        ),
        2 => (
            _mm256_setr_epi8(
                4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 12,
                13, 14, 15, 8, 9, 10, 11,
            ),
            _mm256_set1_epi64x(0x0000_0000_ffff_ffff),
        ),
        4 => (
            _mm256_setr_epi8(
                8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                0, 1, 2, 3, 4, 5, 6, 7,
            ),
            _mm256_setr_epi64x(-1, 0, -1, 0),
        ),
        _ => (
            _mm256_setr_epi8(
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                10, 11, 12, 13, 14, 15,
            ),
            _mm256_setr_epi64x(-1, -1, 0, 0),
        ),
    };
    let n = total & !15;
    for v in x[..n].as_chunks_mut::<16>().0 {
        // Safety: `v` holds 16 values.
        unsafe {
            let w = ld(v.as_ptr());
            let p = if stride == 8 {
                _mm256_permute4x64_epi64::<0x4E>(w)
            } else {
                _mm256_shuffle_epi8(w, swap)
            };
            // a lanes: (w, p) are (a, b); b lanes: (p, w) are (a, b).
            let a = _mm256_blendv_epi8(p, w, amask);
            let b = _mm256_blendv_epi8(w, p, amask);
            let (na, nb) = pair(a, b);
            st(v.as_mut_ptr(), _mm256_blendv_epi8(nb, na, amask));
        }
    }
    n
}

/// AVX2 [`super::bands`] denormalisation of one band: `out[i] = MULT16_16(x[i], g)` shifted
/// right by `shift` (left for a negative one); returns how many values it did (all of them
/// from 8 up, else none).
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn denormalise_avx2(out: &mut [i32], x: &[i16], g: i32, shift: i32) -> usize {
    use std::arch::x86_64::*;
    let gv = _mm256_set1_epi32(g as i16 as i32);
    let n = x.len();
    assert!(out.len() == n);
    if n < 8 {
        return 0;
    }
    let (rc, lc) = (
        _mm_cvtsi32_si128(shift.max(0)),
        _mm_cvtsi32_si128((-shift).max(0)),
    );
    let mut f = |i: usize| {
        // Safety: i + 8 <= n.
        unsafe {
            let p = _mm256_mullo_epi32(
                _mm256_cvtepi16_epi32(_mm_loadu_si128(x.as_ptr().add(i) as *const __m128i)),
                gv,
            );
            let r = _mm256_sll_epi32(_mm256_sra_epi32(p, rc), lc);
            _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, r);
        }
    };
    let mut i = 0;
    while i + 8 < n {
        f(i);
        i += 8;
    }
    // The last vector may repeat some outputs, from the unchanged input.
    f(n - 8);
    n
}

/// Samples per channel that [`deemph_stereo_avx2`] takes.
#[cfg(target_arch = "x86_64")]
pub const DEEMPH_N: usize = 960;

/// AVX2 [`super::celt::deemph_stereo_scalar`] of 960 samples per channel with coefficient `p`.
///
/// The filter `m' = (p * (x + m)) >> 15` is a serial chain per channel. Here each channel is
/// cut into four segments whose chains run side by side in the 64-bit lanes of one vector.
/// A segment after the first starts from an estimate of its true starting state: without the
/// floor, the state is a geometric sum of the preceding inputs, and the floors only take it
/// below that sum by less than `1 / (1 - p / 32768)`. Once the true state is known (the end of
/// the previous segment), the segment is redone from it one step at a time until it meets
/// the speculative run, after which the two agree. So the result is exact whatever the
/// estimate; a close one just makes the redo short.
///
/// Returns the new states, or `None` (leaving `t` unspecified) when an input or state is too
/// large for the sums to be sure to fit 32 bits, for the scalar code to take the frame.
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn deemph_stereo_avx2(
    x0: &[i32],
    x1: &[i32],
    mem: [i32; 2],
    p: i32,
    t: &mut [i32],
) -> Option<[i32; 2]> {
    use std::arch::x86_64::*;
    const N: usize = DEEMPH_N;
    const SEG: usize = N / 4;
    /// Inputs taken into a starting-state estimate.
    const TAPS: usize = 152;
    assert!(x0.len() == N && x1.len() == N && t.len() >= 2 * N && (0..32768).contains(&p));
    // With |x| <= MAX_X and |m| <= MAX_M, every state stays within MAX_M (the floor adds at
    // most 1 to p/32768 of the sum) and every sum within MAX_X + MAX_M < 2^31.
    let r = p as f64 / 32768.0;
    let max_m = ((r * MAX_X as f64 + 1.0) / (1.0 - r)) as i64;
    if MAX_X as i64 + max_m + 8 >= 1 << 31 || mem.iter().any(|&m| (m as i64).abs() > max_m) {
        return None;
    }
    const MAX_X: u32 = 300_000_000;
    // Safety (all blocks below): x0 and x1 hold N values and t 2N, which every index stays
    // within.
    let max_abs = |x: &[i32]| unsafe {
        let mut m = _mm256_setzero_si256();
        for v in x.as_chunks::<8>().0 {
            let a = _mm256_abs_epi32(_mm256_loadu_si256(v.as_ptr() as *const __m256i));
            m = _mm256_max_epu32(m, a);
        }
        let m = _mm_max_epu32(_mm256_castsi256_si128(m), _mm256_extracti128_si256::<1>(m));
        let m = _mm_max_epu32(m, _mm_shuffle_epi32::<0x4E>(m));
        let m = _mm_max_epu32(m, _mm_shuffle_epi32::<0xB1>(m));
        _mm_cvtsi128_si32(m) as u32
    };
    if max_abs(x0) > MAX_X || max_abs(x1) > MAX_X {
        return None;
    }
    // Powers r^1..r^TAPS, for the estimates.
    let mut pw = [0f64; TAPS];
    let mut q = 1.0;
    for v in pw.iter_mut() {
        q *= r;
        *v = q;
    }
    // The state at `start`: the floorless geometric sum of the previous TAPS inputs, less
    // half the floors' largest total effect.
    let estimate = |x: &[i32], start: usize| unsafe {
        let mut acc = _mm256_setzero_pd();
        for j in (0..TAPS).step_by(4) {
            // Inputs x[start - 1 - j - l] for l = 0..3, newest first.
            let v = _mm_loadu_si128(x.as_ptr().add(start - 4 - j) as *const __m128i);
            let v = _mm_shuffle_epi32::<0x1B>(v);
            let w = _mm256_loadu_pd(pw.as_ptr().add(j));
            acc = _mm256_add_pd(acc, _mm256_mul_pd(_mm256_cvtepi32_pd(v), w));
        }
        let h = _mm_add_pd(_mm256_castpd256_pd128(acc), _mm256_extractf128_pd::<1>(acc));
        let s = _mm_cvtsd_f64(_mm_add_sd(h, _mm_unpackhi_pd(h, h)));
        (s - 0.5 / (1.0 - r)).round() as i32
    };
    let starts = |x: &[i32], m: i32| {
        [
            m,
            estimate(x, SEG),
            estimate(x, 2 * SEG),
            estimate(x, 3 * SEG),
        ]
    };
    let (e0, e1) = (starts(x0, mem[0]), starts(x1, mem[1]));
    unsafe {
        let pv = _mm256_set1_epi64x(p as i64);
        let mut f0 = _mm256_setr_epi64x(e0[0] as i64, e0[1] as i64, e0[2] as i64, e0[3] as i64);
        let mut f1 = _mm256_setr_epi64x(e1[0] as i64, e1[1] as i64, e1[2] as i64, e1[3] as i64);
        let pick = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
        // Four steps of the four segments of channel `x` from states `f`: returns the sums as
        // four rows (one per segment) of four steps.
        let block = |x: &[i32], k: usize, f: &mut __m256i| {
            let row = |j: usize| _mm_loadu_si128(x.as_ptr().add(j * SEG + k) as *const __m128i);
            let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
            let t0 = _mm_unpacklo_epi32(r0, r1);
            let t1 = _mm_unpackhi_epi32(r0, r1);
            let t2 = _mm_unpacklo_epi32(r2, r3);
            let t3 = _mm_unpackhi_epi32(r2, r3);
            let cols = [
                _mm_unpacklo_epi64(t0, t2),
                _mm_unpackhi_epi64(t0, t2),
                _mm_unpacklo_epi64(t1, t3),
                _mm_unpackhi_epi64(t1, t3),
            ];
            let mut s = [_mm_setzero_si128(); 4];
            for (s, &c) in s.iter_mut().zip(&cols) {
                // The sums are exact in the low 32 bits of each lane, which is all the
                // multiply reads; the product's bits 15..46 are the new state.
                let v = _mm256_add_epi32(_mm256_cvtepi32_epi64(c), *f);
                *f = _mm256_srli_epi64::<15>(_mm256_mul_epi32(v, pv));
                *s = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(v, pick));
            }
            let t0 = _mm_unpacklo_epi32(s[0], s[1]);
            let t1 = _mm_unpackhi_epi32(s[0], s[1]);
            let t2 = _mm_unpacklo_epi32(s[2], s[3]);
            let t3 = _mm_unpackhi_epi32(s[2], s[3]);
            [
                _mm_unpacklo_epi64(t0, t2),
                _mm_unpackhi_epi64(t0, t2),
                _mm_unpacklo_epi64(t1, t3),
                _mm_unpackhi_epi64(t1, t3),
            ]
        };
        for k in (0..SEG).step_by(4) {
            let a = block(x0, k, &mut f0);
            let b = block(x1, k, &mut f1);
            for j in 0..4 {
                let o = t.as_mut_ptr().add(2 * (j * SEG + k));
                _mm_storeu_si128(o as *mut __m128i, _mm_unpacklo_epi32(a[j], b[j]));
                _mm_storeu_si128(o.add(4) as *mut __m128i, _mm_unpackhi_epi32(a[j], b[j]));
            }
        }
    }
    // Redo each later segment from its true starting state until it meets the speculative run.
    let next = |s: i32| ((p as i64 * s as i64) >> 15) as i32;
    let mut out = [0i32; 2];
    for (ch, (x, e)) in [(x0, e0), (x1, e1)].into_iter().enumerate() {
        for seg in 1..4 {
            let start = seg * SEG;
            let mut f = next(t[2 * (start - 1) + ch]);
            if f == e[seg] {
                continue;
            }
            for k in start..start + SEG {
                let s = x[k] + f;
                if s == t[2 * k + ch] {
                    break;
                }
                t[2 * k + ch] = s;
                f = next(s);
            }
        }
        out[ch] = next(t[2 * (N - 1) + ch]);
    }
    Some(out)
}

/// AVX2 [`super::celt`] `sig2word16_slice_scalar` over the first `n - n % 16` values;
/// returns how many it did.
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn sig2word16_avx2(out: &mut [i16], x: &[i32]) -> usize {
    use std::arch::x86_64::*;
    let n = out.len().min(x.len()) & !15;
    let r = _mm256_set1_epi32(2048);
    for (o, v) in out[..n]
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(x[..n].as_chunks::<16>().0)
    {
        // Safety: `o` and `v` hold 16 values each.
        unsafe {
            // PSHR32(x, 12) then saturation, which the pack does.
            let a = _mm256_srai_epi32::<12>(_mm256_add_epi32(
                _mm256_loadu_si256(v.as_ptr() as *const __m256i),
                r,
            ));
            let b = _mm256_srai_epi32::<12>(_mm256_add_epi32(
                _mm256_loadu_si256(v.as_ptr().add(8) as *const __m256i),
                r,
            ));
            let p = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(a, b));
            _mm256_storeu_si256(o.as_mut_ptr() as *mut __m256i, p);
        }
    }
    n
}

/// AVX2 [`super::rate::bits2pulses_scalar`] on the band's pulse cache `c`: the cache is
/// non-decreasing, so the binary search ends where the entries reach `bits - 1`, which a
/// vector compare counts without the search's dependent loads and branches.
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn bits2pulses_avx2(c: &[u8], bits: i32) -> i32 {
    use std::arch::x86_64::*;
    let max = c[0] as usize;
    let t = (bits - 1).clamp(0, 256);
    let lo = if t == 0 {
        0
    } else {
        // The entries c[1..=max] (at most 40), read 64 at a time; near the end of the table
        // they are copied out first. Lanes past `max` are masked off below.
        let mut buf = [0u8; 64];
        let src: &[u8] = if c.len() > 64 {
            &c[1..65]
        } else {
            buf[..max].copy_from_slice(&c[1..=max]);
            &buf
        };
        let lim = _mm256_set1_epi8((t - 1) as u8 as i8);
        // Safety: `src` is 64 bytes.
        let (a, b) = unsafe {
            (
                _mm256_loadu_si256(src.as_ptr() as *const __m256i),
                _mm256_loadu_si256(src.as_ptr().add(32) as *const __m256i),
            )
        };
        // Entries at most t - 1 are those below t.
        let ma = _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_min_epu8(a, lim), a)) as u32;
        let mb = _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_min_epu8(b, lim), b)) as u32;
        let m = (ma as u64 | ((mb as u64) << 32)) & ((1u64 << max) - 1);
        m.count_ones() as i32
    };
    super::rate::bits2pulses_pick(c, bits, lo)
}

/// NEON [`bits2pulses_avx2`].
///
/// # Safety
///
/// NEON must be available (it is on every aarch64 CPU).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn bits2pulses_neon(c: &[u8], bits: i32) -> i32 {
    use std::arch::aarch64::*;
    let max = c[0] as usize;
    let t = (bits - 1).clamp(0, 256);
    let lo = if t == 0 {
        0
    } else {
        let mut buf = [0u8; 64];
        let src: &[u8] = if c.len() > 64 {
            &c[1..65]
        } else {
            buf[..max].copy_from_slice(&c[1..=max]);
            &buf
        };
        const IDX: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        // Safety: `src` is 64 bytes and `IDX` 16.
        unsafe {
            let lim = vdupq_n_u8((t - 1) as u8);
            let idx = vld1q_u8(IDX.as_ptr());
            let one = vdupq_n_u8(1);
            let mut count = 0u32;
            for k in 0..4 {
                let x = vld1q_u8(src.as_ptr().add(16 * k));
                // Entries at most t - 1 are those below t; lanes past `max` do not count.
                let valid = vcltq_u8(
                    vaddq_u8(idx, vdupq_n_u8(16 * k as u8)),
                    vdupq_n_u8(max as u8),
                );
                let m = vandq_u8(vandq_u8(vcleq_u8(x, lim), valid), one);
                count += vaddvq_u8(m) as u32;
            }
            count as i32
        }
    };
    super::rate::bits2pulses_pick(c, bits, lo)
}

/// AVX2 [`super::rate::alloc_psum_scalar`]: the 21 bands in three vectors, the band where
/// the scan turns `done` found from a mask instead of a branch per band.
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn alloc_psum_avx2(
    tmp: &[i32; 21],
    thresh: &[i32],
    cap: &[i32],
    start: usize,
    end: usize,
    floor: i32,
) -> i32 {
    use std::arch::x86_64::*;
    assert!(start <= end && end <= 21 && thresh.len() >= end && cap.len() >= end);
    let pad = |s: &[i32]| -> [i32; 24] {
        let mut a = [0i32; 24];
        a[start..end].copy_from_slice(&s[start..end]);
        a
    };
    let (t, th, cp) = (pad(tmp), pad(thresh), pad(cap));
    let fl = _mm256_set1_epi32(floor);
    let lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let (s, e) = (
        _mm256_set1_epi32(start as i32),
        _mm256_set1_epi32(end as i32),
    );
    let mut tv = [_mm256_setzero_si256(); 3];
    let mut valid = [_mm256_setzero_si256(); 3];
    let mut reach = 0u32;
    for v in 0..3 {
        // Safety: the arrays hold 24 values.
        unsafe {
            let idx = _mm256_add_epi32(lanes, _mm256_set1_epi32(8 * v as i32));
            valid[v] = _mm256_andnot_si256(_mm256_cmpgt_epi32(s, idx), _mm256_cmpgt_epi32(e, idx));
            tv[v] = _mm256_loadu_si256(t.as_ptr().add(8 * v) as *const __m256i);
            let thv = _mm256_loadu_si256(th.as_ptr().add(8 * v) as *const __m256i);
            // tmp >= thresh is !(thresh > tmp).
            let ge = _mm256_andnot_si256(_mm256_cmpgt_epi32(thv, tv[v]), valid[v]);
            reach |= (_mm256_movemask_ps(_mm256_castsi256_ps(ge)) as u32) << (8 * v);
        }
    }
    // Bands up to the highest one reaching its threshold are `done`.
    let top = if reach == 0 {
        -1
    } else {
        31 - reach.leading_zeros() as i32
    };
    let topv = _mm256_set1_epi32(top);
    let mut sum = _mm256_setzero_si256();
    for v in 0..3 {
        // Safety: the arrays hold 24 values.
        unsafe {
            let idx = _mm256_add_epi32(lanes, _mm256_set1_epi32(8 * v as i32));
            let done = _mm256_cmpgt_epi32(_mm256_add_epi32(topv, _mm256_set1_epi32(1)), idx);
            let cpv = _mm256_loadu_si256(cp.as_ptr().add(8 * v) as *const __m256i);
            let capped = _mm256_min_epi32(tv[v], cpv);
            let floored = _mm256_andnot_si256(_mm256_cmpgt_epi32(fl, tv[v]), fl);
            let c = _mm256_blendv_epi8(floored, capped, done);
            sum = _mm256_add_epi32(sum, _mm256_and_si256(c, valid[v]));
        }
    }
    let h = _mm_add_epi32(
        _mm256_castsi256_si128(sum),
        _mm256_extracti128_si256::<1>(sum),
    );
    let h = _mm_add_epi32(h, _mm_shuffle_epi32::<0x4E>(h));
    let h = _mm_add_epi32(h, _mm_shuffle_epi32::<0xB1>(h));
    _mm_cvtsi128_si32(h)
}

/// One sample of the constant-gain comb filter, as [`comb_const_avx2`] defines it.
///
/// # Safety
///
/// `x - t - 2..=x + i` must be readable and `y + i` writable.
#[inline(always)]
unsafe fn comb_const_one(y: *mut i32, x: *const i32, i: usize, t: usize, g: [i32; 3]) {
    use super::math::mul16_32_q15;
    // Safety: per the function contract.
    unsafe {
        let r = |k: isize| *x.offset(i as isize - t as isize + k);
        let v = (*x.add(i))
            .wrapping_add(mul16_32_q15(g[0], r(0)))
            .wrapping_add(mul16_32_q15(g[1], r(1).wrapping_add(r(-1))))
            .wrapping_add(mul16_32_q15(g[2], r(2).wrapping_add(r(-2))));
        *y.add(i) = v.clamp(-super::plc::SIG_SAT, super::plc::SIG_SAT);
    }
}

/// AVX2 constant-gain part of `comb_filter`: for `i` in `0..n`,
/// `y[i] = SATURATE(x[i] + g0 x[i-t] + g1 (x[i-t+1] + x[i-t-1]) + g2 (x[i-t+2] + x[i-t-2]))`
/// with `MULT16_32_Q15` products.
///
/// # Safety
///
/// The CPU must support AVX2; `x - t - 2..x + n` must be readable and `y..y + n` writable.
/// `y` either equals `x` with `t >= 15`, so that every read of a vector of eight outputs is of
/// an earlier, already filtered sample (libopus filters in place this way), or does not
/// overlap the reads.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn comb_const_avx2(y: *mut i32, x: *const i32, n: usize, t: usize, g: [i32; 3]) {
    use avx2::*;
    use std::arch::x86_64::*;
    let g = g.map(|v| v as i16 as i32);
    let (g0, g1, g2) = (
        _mm256_set1_epi32(g[0]),
        _mm256_set1_epi32(g[1]),
        _mm256_set1_epi32(g[2]),
    );
    let lim = _mm256_set1_epi32(super::plc::SIG_SAT);
    let nlim = _mm256_set1_epi32(-super::plc::SIG_SAT);
    let mut i = 0usize;
    while i + 8 <= n {
        // Safety: per the function contract.
        unsafe {
            let h = x.add(i).sub(t);
            let v = _mm256_add_epi32(ld(x.add(i)), mulq15c(ld(h), g0));
            let v = _mm256_add_epi32(v, mulq15c(_mm256_add_epi32(ld(h.add(1)), ld(h.sub(1))), g1));
            let v = _mm256_add_epi32(v, mulq15c(_mm256_add_epi32(ld(h.add(2)), ld(h.sub(2))), g2));
            st(y.add(i), _mm256_max_epi32(_mm256_min_epi32(v, lim), nlim));
        }
        i += 8;
    }
    while i < n {
        // Safety: per the function contract.
        unsafe { comb_const_one(y, x, i, t, g) };
        i += 1;
    }
}

/// AVX2 cross-fading part of `comb_filter` over the first `n` samples: the filter with
/// period `t0` and gains `g0` fades out with `1 - w^2` while the one with `t1`, `g1` fades in.
///
/// # Safety
///
/// As [`comb_const_avx2`], with `t = max(t0, t1)`; `window` must hold `n` values.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn comb_overlap_avx2(
    y: *mut i32,
    x: *const i32,
    n: usize,
    t0: usize,
    t1: usize,
    g0: [i32; 3],
    g1: [i32; 3],
    window: &[i16],
) {
    use avx2::*;
    use std::arch::x86_64::*;
    assert!(window.len() >= n);
    let gv = |g: [i32; 3]| g.map(|v| _mm256_set1_epi32(v as i16 as i32));
    let (g0v, g1v) = (gv(g0), gv(g1));
    let one = _mm256_set1_epi32(32767);
    let lim = _mm256_set1_epi32(super::plc::SIG_SAT);
    let nlim = _mm256_set1_epi32(-super::plc::SIG_SAT);
    let coef = |w: __m256i, g: __m256i| _mm256_srai_epi32::<15>(_mm256_mullo_epi32(w, g));
    let mut i = 0usize;
    while i + 8 <= n {
        // Safety: per the function contract; window[i..i + 8] is in bounds.
        unsafe {
            let w =
                _mm256_cvtepi16_epi32(_mm_loadu_si128(window.as_ptr().add(i) as *const __m128i));
            let f = _mm256_srai_epi32::<15>(_mm256_mullo_epi32(w, w));
            let nf = _mm256_sub_epi32(one, f);
            let h0 = x.add(i).sub(t0);
            let h1 = x.add(i).sub(t1);
            let mut v = ld(x.add(i));
            v = _mm256_add_epi32(v, mulq15(ld(h0), coef(nf, g0v[0])));
            v = _mm256_add_epi32(
                v,
                mulq15(
                    _mm256_add_epi32(ld(h0.sub(1)), ld(h0.add(1))),
                    coef(nf, g0v[1]),
                ),
            );
            v = _mm256_add_epi32(
                v,
                mulq15(
                    _mm256_add_epi32(ld(h0.sub(2)), ld(h0.add(2))),
                    coef(nf, g0v[2]),
                ),
            );
            v = _mm256_add_epi32(v, mulq15(ld(h1), coef(f, g1v[0])));
            v = _mm256_add_epi32(
                v,
                mulq15(
                    _mm256_add_epi32(ld(h1.sub(1)), ld(h1.add(1))),
                    coef(f, g1v[1]),
                ),
            );
            v = _mm256_add_epi32(
                v,
                mulq15(
                    _mm256_add_epi32(ld(h1.sub(2)), ld(h1.add(2))),
                    coef(f, g1v[2]),
                ),
            );
            st(y.add(i), _mm256_max_epi32(_mm256_min_epi32(v, lim), nlim));
        }
        i += 8;
    }
    if i < n {
        // Safety: per the function contract.
        unsafe { super::plc::comb_overlap_tail(y, x, i, n, t0, t1, g0, g1, window) };
    }
}

/// NEON [`comb_const_avx2`]; `g` must not hold `-32768` (see the module notes).
///
/// # Safety
///
/// As [`comb_const_avx2`], with NEON instead of AVX2.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn comb_const_neon(y: *mut i32, x: *const i32, n: usize, t: usize, g: [i32; 3]) {
    use std::arch::aarch64::*;
    let g = g.map(|v| v as i16 as i32);
    let gq = g.map(|v| vdupq_n_s32(v << 16));
    let lim = vdupq_n_s32(super::plc::SIG_SAT);
    let nlim = vdupq_n_s32(-super::plc::SIG_SAT);
    let mut i = 0usize;
    while i + 4 <= n {
        // Safety: per the function contract; four lanes read at most x[i + 5 - t] < x[i].
        unsafe {
            let h = x.add(i).sub(t);
            let mut v = vaddq_s32(vld1q_s32(x.add(i)), vqdmulhq_s32(vld1q_s32(h), gq[0]));
            v = vaddq_s32(
                v,
                vqdmulhq_s32(vaddq_s32(vld1q_s32(h.add(1)), vld1q_s32(h.sub(1))), gq[1]),
            );
            v = vaddq_s32(
                v,
                vqdmulhq_s32(vaddq_s32(vld1q_s32(h.add(2)), vld1q_s32(h.sub(2))), gq[2]),
            );
            vst1q_s32(y.add(i), vmaxq_s32(vminq_s32(v, lim), nlim));
        }
        i += 4;
    }
    while i < n {
        // Safety: per the function contract.
        unsafe { comb_const_one(y, x, i, t, g) };
        i += 1;
    }
}

/// AVX2 [`super::silk_dsp::fir_interpol_scalar`] from the start: four outputs at a time, each
/// an 8-tap `pmaddwd` dot product (exact: the reference sums wrap the same way).
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn fir_interpol_avx2(
    out: &mut [i16],
    buf: &[i16],
    max_index_q16: i32,
    inc: i32,
) -> usize {
    use super::silk_dsp::{FIR_12_TAPS, fir_interpol_scalar};
    use std::arch::x86_64::*;
    let mut index_q16 = 0i32;
    let mut o = 0usize;
    let taps = FIR_12_TAPS.as_ptr() as *const __m128i;
    let one = _mm_set1_epi32(1);
    // Four outputs need positions up to index + 3 inc, each reading 8 values.
    while index_q16 + 3 * inc < max_index_q16 && o + 4 <= out.len() {
        let mut acc = [_mm_setzero_si128(); 4];
        for (k, a) in acc.iter_mut().enumerate() {
            let idx = index_q16 + k as i32 * inc;
            let i0 = (idx >> 16) as usize;
            let ti = ((idx & 0xffff) * 12) >> 16;
            assert!(i0 + 8 <= buf.len());
            // Safety: buf[i0..i0 + 8] is checked above; ti < 12.
            unsafe {
                let x = _mm_loadu_si128(buf.as_ptr().add(i0) as *const __m128i);
                *a = _mm_madd_epi16(x, _mm_loadu_si128(taps.add(ti as usize)));
            }
        }
        let s = _mm_hadd_epi32(
            _mm_hadd_epi32(acc[0], acc[1]),
            _mm_hadd_epi32(acc[2], acc[3]),
        );
        // RSHIFT_ROUND(s, 15) then SAT16.
        let r = _mm_srai_epi32::<1>(_mm_add_epi32(_mm_srai_epi32::<14>(s), one));
        let v = _mm_packs_epi32(r, r);
        let mut tmp = [0i16; 8];
        // Safety: `tmp` is 16 bytes.
        unsafe { _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, v) };
        out[o..o + 4].copy_from_slice(&tmp[..4]);
        o += 4;
        index_q16 += 4 * inc;
    }
    fir_interpol_scalar(out, buf, max_index_q16, inc, index_q16, o)
}

/// AVX2 [`super::silk_dsp::fir_interpol_scalar`] for the 2x-to-3x step of 16 to 48 kHz
/// (`inc == 43691`) over `n_in <= 160` input samples (`2 n_in` in `buf` after the 8 history
/// values). From position 0 the steps then follow a fixed pattern: output `3j` is phase 0 and
/// `3j + 1` phase 8 at `buf[2j..]`, `3j + 2` phase 4 at `buf[2j + 1..]` (position `3j inc` is
/// `2j * 65536 + j`, and `j < 160` keeps every fraction inside its phase). Eight `j` at once,
/// each `pmaddwd` taking two adjacent taps; returns the `3 n_in` outputs written.
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn fir_interpol_3x_avx2(out: &mut [i16], buf: &[i16], n_in: usize) -> usize {
    use super::silk_dsp::{FIR_12_TAPS, fir_interpol_scalar};
    use std::arch::x86_64::*;
    assert!(n_in <= 160 && buf.len() >= 2 * n_in + 8 && out.len() >= 3 * n_in);
    let pairs = |ti: usize| -> [__m256i; 4] {
        let t = &FIR_12_TAPS[ti];
        std::array::from_fn(|m| {
            _mm256_set1_epi32((t[2 * m] as u16 as i32) | ((t[2 * m + 1] as i32) << 16))
        })
    };
    let (p0, p8, p4) = (pairs(0), pairs(8), pairs(4));
    let one = _mm256_set1_epi32(1);
    let bp = buf.as_ptr();
    let mut j0 = 0usize;
    while j0 + 8 <= n_in {
        // Safety: the loads read buf[2 j0..2 j0 + 23] and 2 j0 + 23 <= 2 n_in + 7.
        let (a, b, c) = unsafe {
            let dot = |off: usize, p: &[__m256i; 4]| {
                let mut acc = _mm256_setzero_si256();
                for (m, pm) in p.iter().enumerate() {
                    let x = _mm256_loadu_si256(bp.add(2 * j0 + off + 2 * m) as *const __m256i);
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(x, *pm));
                }
                // RSHIFT_ROUND(acc, 15); SAT16 comes with the pack.
                let r = _mm256_srai_epi32::<1>(_mm256_add_epi32(_mm256_srai_epi32::<14>(acc), one));
                _mm_packs_epi32(_mm256_castsi256_si128(r), _mm256_extracti128_si256::<1>(r))
            };
            (dot(0, &p0), dot(0, &p8), dot(1, &p4))
        };
        let mut t = [[0i16; 8]; 3];
        // Safety: each row of `t` is 16 bytes.
        unsafe {
            _mm_storeu_si128(t[0].as_mut_ptr() as *mut __m128i, a);
            _mm_storeu_si128(t[1].as_mut_ptr() as *mut __m128i, b);
            _mm_storeu_si128(t[2].as_mut_ptr() as *mut __m128i, c);
        }
        for (k, o) in out[3 * j0..3 * j0 + 24]
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .enumerate()
        {
            o.copy_from_slice(&[t[0][k], t[1][k], t[2][k]]);
        }
        j0 += 8;
    }
    fir_interpol_scalar(
        out,
        buf,
        (n_in as i32) << 17,
        43691,
        j0 as i32 * 131073,
        3 * j0,
    )
}

/// AVX2 short-term synthesis of `silk_decode_core` (see `silk::lpc_synthesis_scalar`) over
/// blocks of eight outputs: the terms whose inputs precede the block are formed for all eight
/// at once, then each output adds the terms of the block's own earlier outputs in turn. The
/// `SMLAWB` terms are truncated one by one and summed with wrapping adds, so the order of the
/// sum does not matter. Returns the number of outputs done (a multiple of 8).
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn lpc_synthesis_avx2(
    s_lpc: &mut [i32],
    a_q12: &[i16; 16],
    order: usize,
    r: &[i32],
    xq: &mut [i16],
    gain_q10: i32,
) -> usize {
    use super::silk_dsp::{lshift_sat32, rshift_round, sat16, smlawb, smulww};
    use std::arch::x86_64::*;
    let n = r.len();
    assert!(s_lpc.len() >= 16 + n && xq.len() >= n);
    if !(8..=16).contains(&order) {
        return 0;
    }
    let a: [i32; 16] = std::array::from_fn(|j| a_q12[j] as i32);
    let av: [__m256i; 16] = std::array::from_fn(|j| _mm256_set1_epi32(a[j]));
    let lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let mut i0 = 0usize;
    while i0 + 8 <= n {
        // Lane b: p = order / 2 + the terms j >= b, whose inputs s[16 + i0 + b - 1 - j]
        // precede the block.
        let mut acc = _mm256_set1_epi32((order >> 1) as i32);
        for j in 0..order {
            // Safety: reads s_lpc[16 + i0 - 1 - j..][..8], within 0..16 + n.
            let x = unsafe {
                _mm256_loadu_si256(s_lpc.as_ptr().add(16 + i0 - 1 - j) as *const __m256i)
            };
            let even = _mm256_srli_epi64::<16>(_mm256_mul_epi32(x, av[j]));
            let odd = _mm256_slli_epi64::<16>(_mm256_mul_epi32(_mm256_srli_epi64::<32>(x), av[j]));
            let mut t = _mm256_blend_epi32::<0xAA>(even, odd);
            if j < 7 {
                t = _mm256_and_si256(
                    t,
                    _mm256_cmpgt_epi32(_mm256_set1_epi32(j as i32 + 1), lanes),
                );
            }
            acc = _mm256_add_epi32(acc, t);
        }
        let mut known = [0i32; 8];
        // Safety: `known` is 32 bytes.
        unsafe { _mm256_storeu_si256(known.as_mut_ptr() as *mut __m256i, acc) };
        let r8: &[i32; 8] = r[i0..i0 + 8].try_into().unwrap();
        let mut v = [0i32; 8];
        // Output b adds the terms of outputs i0..i0 + b (order >= 8, so all of them), the
        // newest last; unrolled so that every index is a constant.
        #[inline(always)]
        fn step<const B: usize>(v: &mut [i32; 8], known: &[i32; 8], a: &[i32; 16], r: &[i32; 8]) {
            let mut p = known[B];
            for j in 1..B {
                p = smlawb(p, v[B - 1 - j], a[j]);
            }
            if B > 0 {
                p = smlawb(p, v[B.saturating_sub(1)], a[0]);
            }
            v[B] = r[B].saturating_add(lshift_sat32(p, 4));
        }
        step::<0>(&mut v, &known, &a, r8);
        step::<1>(&mut v, &known, &a, r8);
        step::<2>(&mut v, &known, &a, r8);
        step::<3>(&mut v, &known, &a, r8);
        step::<4>(&mut v, &known, &a, r8);
        step::<5>(&mut v, &known, &a, r8);
        step::<6>(&mut v, &known, &a, r8);
        step::<7>(&mut v, &known, &a, r8);
        for (o, &x) in xq[i0..i0 + 8].iter_mut().zip(&v) {
            *o = sat16(rshift_round(smulww(x, gain_q10), 8)) as i16;
        }
        s_lpc[16 + i0..16 + i0 + 8].copy_from_slice(&v);
        i0 += 8;
    }
    i0
}

/// AVX2 [`super::silk_dsp::lpc_analysis_filter`] for `d <= 16`: eight outputs at a time, each
/// a 16-tap `pmaddwd` dot product over the previous 16 inputs (taps beyond `d` are zero).
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn lpc_analysis_filter_avx2(
    out: &mut [i16],
    input: &[i16],
    b: &[i16],
    len: usize,
    d: usize,
) {
    use super::silk_dsp::lpc_analysis_filter_scalar;
    use std::arch::x86_64::*;
    assert!(d <= 16 && b.len() >= d && input.len() >= len && out.len() >= len);
    let start = d.max(16);
    if len <= start {
        lpc_analysis_filter_scalar(out, input, b, len, d, d);
        return;
    }
    lpc_analysis_filter_scalar(out, input, b, start, d, d);
    // brev[15 - j] = b[j]: lane l multiplies input[ix - 16 + l].
    let mut brev = [0i16; 16];
    for (j, &c) in b[..d].iter().enumerate() {
        brev[15 - j] = c;
    }
    // Safety: `brev` is 32 bytes.
    let bv = unsafe { _mm256_loadu_si256(brev.as_ptr() as *const __m256i) };
    let one = _mm_set1_epi32(1);
    let ip = input.as_ptr();
    let mut ix = start;
    while ix + 8 <= len {
        // Safety: the loads read input[ix - 16..ix + 8], within `input` as ix >= 16 and
        // ix + 8 <= len.
        let v = unsafe {
            let m: [__m256i; 8] = std::array::from_fn(|k| {
                _mm256_madd_epi16(
                    _mm256_loadu_si256(ip.add(ix + k - 16) as *const __m256i),
                    bv,
                )
            });
            let a = _mm256_hadd_epi32(_mm256_hadd_epi32(m[0], m[1]), _mm256_hadd_epi32(m[2], m[3]));
            let c = _mm256_hadd_epi32(_mm256_hadd_epi32(m[4], m[5]), _mm256_hadd_epi32(m[6], m[7]));
            let sum = _mm256_add_epi32(
                _mm256_permute2x128_si256::<0x20>(a, c),
                _mm256_permute2x128_si256::<0x31>(a, c),
            );
            let x = _mm256_cvtepi16_epi32(_mm_loadu_si128(ip.add(ix) as *const __m128i));
            let acc = _mm256_sub_epi32(_mm256_slli_epi32::<12>(x), sum);
            // RSHIFT_ROUND(acc, 12) then SAT16.
            let lo = _mm256_castsi256_si128(acc);
            let hi = _mm256_extracti128_si256::<1>(acc);
            let r = |v: __m128i| _mm_srai_epi32::<1>(_mm_add_epi32(_mm_srai_epi32::<11>(v), one));
            _mm_packs_epi32(r(lo), r(hi))
        };
        let mut tmp = [0i16; 8];
        // Safety: `tmp` is 16 bytes.
        unsafe { _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, v) };
        out[ix..ix + 8].copy_from_slice(&tmp);
        ix += 8;
    }
    lpc_analysis_filter_scalar(out, input, b, len, d, ix);
}

/// NEON [`super::silk_dsp::fir_interpol_scalar`] from the start.
///
/// # Safety
///
/// NEON must be available (it is on every aarch64 CPU).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn fir_interpol_neon(
    out: &mut [i16],
    buf: &[i16],
    max_index_q16: i32,
    inc: i32,
) -> usize {
    use super::silk_dsp::{FIR_12_TAPS, rshift_round, sat16};
    use std::arch::aarch64::*;
    let mut index_q16 = 0i32;
    let mut o = 0usize;
    while index_q16 < max_index_q16 {
        let i0 = (index_q16 >> 16) as usize;
        let ti = (((index_q16 & 0xffff) * 12) >> 16) as usize;
        let x = &buf[i0..i0 + 8];
        let t = &FIR_12_TAPS[ti];
        // Safety: `x` and `t` hold eight values each.
        let res = unsafe {
            let xv = vld1q_s16(x.as_ptr());
            let tv = vld1q_s16(t.as_ptr());
            let p = vmull_s16(vget_low_s16(xv), vget_low_s16(tv));
            vaddvq_s32(vmlal_high_s16(p, xv, tv))
        };
        out[o] = sat16(rshift_round(res, 15)) as i16;
        o += 1;
        index_q16 += inc;
    }
    o
}

#[cfg(test)]
mod tests {
    use super::super::mdct::{FFT_STATES, fft_impl, fft_impl_scalar, post_rotate_scalar};

    /// Deterministic test signal mixing small, large and extreme values.
    fn signal(n: usize, seed: u32, amp: i32) -> Vec<i32> {
        let mut s = seed;
        (0..n)
            .map(|k| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                match k % 37 {
                    0 => i32::MAX,
                    1 => i32::MIN,
                    2 => -amp,
                    _ => ((s as i32) >> 8) % amp.max(1),
                }
            })
            .collect()
    }

    #[test]
    fn fft_matches_scalar() {
        for st in &FFT_STATES {
            let n = 2 * st.bitrev.len();
            for (seed, amp) in [(1, 1 << 12), (2, 1 << 20), (3, 1 << 28), (4, i32::MAX)] {
                let x = signal(n, seed, amp);
                let mut a = x.clone();
                let mut b = x.clone();
                fft_impl_scalar(st, &mut a);
                fft_impl(st, &mut b);
                assert_eq!(a, b, "nfft {} amp {amp}", n / 2);
            }
        }
    }

    #[test]
    fn bits2pulses_matches_scalar() {
        use super::super::celt_tables::CACHE_INDEX;
        use super::super::rate::{bits2pulses, bits2pulses_scalar};
        for lm in -1..=3 {
            for band in 0..21 {
                if CACHE_INDEX[(lm + 1) as usize * 21 + band] < 0 {
                    continue;
                }
                for bits in -300..3000 {
                    assert_eq!(
                        bits2pulses(band, lm, bits),
                        bits2pulses_scalar(band, lm, bits),
                        "band {band} lm {lm} bits {bits}"
                    );
                }
            }
        }
    }

    /// Deterministic 16-bit test values, including the extremes.
    fn signal16(n: usize, seed: u32) -> Vec<i16> {
        signal(n, seed, 1 << 16)
            .iter()
            .enumerate()
            .map(|(k, &v)| match k % 29 {
                3 => i16::MIN,
                4 => i16::MAX,
                _ => v as i16,
            })
            .collect()
    }

    #[test]
    fn band_kernels_match_scalar() {
        use super::super::bands::*;
        use super::super::vq::{scale_shr, scale_shr_scalar};
        for n in (1..40).chain([48, 64, 96, 176]) {
            let x = signal16(n, n as u32);
            let y = signal16(n, 7 * n as u32 + 1);
            for (g, sh) in [(32767, 1), (-32768, 15), (12345, 9), (-777, 3)] {
                let (mut a, mut b) = (x.clone(), x.clone());
                scale_shr(&mut a, g, sh);
                scale_shr_scalar(&mut b, g, sh);
                assert_eq!(a, b, "scale_shr n {n}");
                let (mut a, mut b) = (vec![0i16; n], vec![0i16; n]);
                mul_q15(&mut a, &x, g);
                mul_q15_scalar(&mut b, &x, g);
                assert_eq!(a, b, "mul_q15 n {n}");
                for shift in [-2, -1, 0, 3, 17] {
                    let (mut a, mut b) = (vec![0i32; n], vec![0i32; n]);
                    denormalise_band(&mut a, &x, g, shift);
                    denormalise_band_scalar(&mut b, &x, g, shift);
                    assert_eq!(a, b, "denormalise n {n} shift {shift}");
                }
            }
            for mid in [0, 1, 16384, 32767, -32768] {
                let (mut xb, mut yb) = (x.clone(), y.clone());
                stereo_merge_scalar(&mut xb, &mut yb, mid, 23170, 9, -18000, 12);
                let (mut xc, mut yc) = (x.clone(), y.clone());
                #[cfg(target_arch = "x86_64")]
                if ragdoll_core::dispatch::avx2() {
                    // Safety: AVX2 is available.
                    if unsafe {
                        super::stereo_merge_avx2(&mut xc, &mut yc, mid, 23170, 9, -18000, 12)
                    } {
                        assert_eq!((&xc, &yc), (&xb, &yb), "stereo_merge n {n}");
                    }
                }
                let _ = (&mut xc, &mut yc);
            }
            for stride in [1usize, 2, 4, 8, 16, 32] {
                let n0 = n / stride;
                if n0 < 2 {
                    continue;
                }
                let (mut a, mut b) = (x.clone(), x.clone());
                haar1(&mut a, n0, stride);
                haar1_scalar(&mut b, n0, stride, 0);
                assert_eq!(a, b, "haar1 n0 {n0} stride {stride}");
            }
        }
    }

    #[test]
    fn deemph_matches_scalar() {
        use super::super::celt::{deemph_stereo, deemph_stereo_scalar};
        const N: usize = 960;
        let mut seed = 5u32;
        let mut rnd = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed as i32
        };
        for trial in 0..300 {
            // Audio-like walks of many sizes, steps at the segment edges, and inputs past the
            // vector path's limit (which then falls back).
            let amp: i32 = [
                1 << 8,
                1 << 16,
                1 << 24,
                299_999_999,
                300_000_000,
                400_000_000,
            ][trial % 6];
            let mut x = [vec![0i32; N], vec![0i32; N]];
            for xc in x.iter_mut() {
                let mut v = 0i64;
                for (k, o) in xc.iter_mut().enumerate() {
                    v += (rnd() >> 4) as i64;
                    v = v.clamp(-(amp as i64), amp as i64);
                    *o = match trial % 5 {
                        0 => v as i32,
                        1 => rnd() % amp,
                        2 => {
                            if (k / 240) % 2 == 0 {
                                amp
                            } else {
                                -amp
                            }
                        }
                        3 => {
                            if k % 240 < 3 {
                                -amp
                            } else {
                                amp / 3
                            }
                        }
                        _ => (v as i32) / 7,
                    };
                }
            }
            let mem = match trial % 4 {
                0 => [0, 0],
                1 => [rnd() % (1 << 30), rnd() % (1 << 20)],
                2 => [1_700_000_000, -1_700_000_000],
                _ => [i32::MAX, i32::MIN],
            };
            let (mut a, mut b) = (vec![0i32; 2 * N], vec![0i32; 2 * N]);
            let ma = deemph_stereo(&x[0], &x[1], mem, &mut a);
            let mb = deemph_stereo_scalar(&x[0], &x[1], mem, &mut b);
            assert_eq!((ma, &a), (mb, &b), "trial {trial}");
            #[cfg(target_arch = "x86_64")]
            if ragdoll_core::dispatch::avx2() {
                // Safety: AVX2 is available.
                let mut c = vec![0i32; 2 * N];
                if let Some(mc) =
                    unsafe { super::deemph_stereo_avx2(&x[0], &x[1], mem, 27853, &mut c) }
                {
                    assert_eq!((mc, &c), (mb, &b), "avx2 trial {trial}");
                }
            }
        }
    }

    #[test]
    fn signal_kernels_match_scalar() {
        use super::super::celt::{saturate_slice, sig2word16_slice, sig2word16_slice_scalar};
        use super::super::celt_tables::WINDOW;
        use super::super::mdct::{tdac_mirror, tdac_mirror_scalar};
        use super::super::plc::{
            COMBFILTER_MINPERIOD, Comb, SIG_SAT, comb_filter_inplace, comb_filter_inplace_scalar,
        };
        for n in (1..40).chain([120, 240, 960, 1000]) {
            let x = signal(n, n as u32, i32::MAX);
            let (mut a, mut b) = (vec![0i16; n], vec![0i16; n]);
            sig2word16_slice(&mut a, &x);
            sig2word16_slice_scalar(&mut b, &x);
            assert_eq!(a, b, "sig2word16 n {n}");
            let mut s = x.clone();
            saturate_slice(&mut s);
            let r: Vec<i32> = x.iter().map(|&v| v.clamp(-SIG_SAT, SIG_SAT)).collect();
            assert_eq!(s, r, "saturate n {n}");
        }
        for overlap in [120usize, 24, 8] {
            let x = signal(overlap, 3, 1 << 30);
            let (mut a, mut b) = (x.clone(), x.clone());
            tdac_mirror(&mut a, &WINDOW[..overlap]);
            tdac_mirror_scalar(&mut b, &WINDOW[..overlap], 0);
            assert_eq!(a, b, "tdac overlap {overlap}");
        }
        let combs = [
            (15, 15, 3072, 3072, 0, 0),
            (15, 300, 0, 24576, 1, 2),
            (1023, 17, -24576, 9216, 2, 0),
            (100, 100, 6144, 0, 1, 1),
            (40, 1022, 12288, -3072, 0, 2),
        ];
        for (t0, t1, g0, g1, tapset0, tapset1) in combs {
            for n in [120usize, 240, 840, 13] {
                let base = signal(1200 + n, t0 as u32 + n as u32, SIG_SAT);
                let c = Comb {
                    t0,
                    t1,
                    g0,
                    g1,
                    tapset0,
                    tapset1,
                };
                let (mut a, mut b) = (base.clone(), base.clone());
                comb_filter_inplace(&mut a, 1100, n, c, &WINDOW, 120);
                comb_filter_inplace_scalar(&mut b, 1100, n, c, &WINDOW, 120);
                assert_eq!(a, b, "comb t0 {t0} t1 {t1} n {n}");
            }
        }
        let _ = COMBFILTER_MINPERIOD;
    }

    #[test]
    fn silk_kernels_match_scalar() {
        use super::super::silk::{lpc_synthesis, lpc_synthesis_scalar};
        use super::super::silk_dsp::{
            fir_interpol, fir_interpol_scalar, lpc_analysis_filter, lpc_analysis_filter_scalar,
        };
        // Resampler interpolation: the 16 -> 48 kHz step and others.
        for (inc, n_in) in [
            (43691, 160),
            (43691, 16),
            (43691, 144),
            (65537, 80),
            (21846, 120),
        ] {
            let buf = signal16(2 * n_in + 8, inc as u32);
            let max = (n_in as i32) << 17;
            let (mut a, mut b) = (vec![0i16; 8 * n_in + 8], vec![0i16; 8 * n_in + 8]);
            let na = fir_interpol(&mut a, &buf, max, inc);
            let nb = fir_interpol_scalar(&mut b, &buf, max, inc, 0, 0);
            assert_eq!((na, &a), (nb, &b), "fir_interpol inc {inc} n_in {n_in}");
        }
        for d in [10usize, 16] {
            for len in [16usize, 17, 23, 40, 80, 320] {
                let input = signal16(len, len as u32 + d as u32);
                let coef: Vec<i16> = signal16(d, 99).iter().map(|&v| v >> 3).collect();
                let (mut a, mut b) = (vec![0i16; len], vec![0i16; len]);
                lpc_analysis_filter(&mut a, &input, &coef, len, d);
                lpc_analysis_filter_scalar(&mut b, &input, &coef, len, d, d);
                assert_eq!(a, b, "lpc_analysis d {d} len {len}");
            }
        }
        for sub in [40usize, 60, 80, 7] {
            let mut a_q12 = [0i16; 16];
            for (k, v) in signal16(16, sub as u32).iter().enumerate() {
                a_q12[k] = v >> 2;
            }
            for amp in [1 << 10, 1 << 20, i32::MAX] {
                let r = signal(sub, amp as u32, amp);
                let hist = signal(16, 5, amp);
                let mut sa = hist.clone();
                sa.resize(16 + sub, 0);
                let mut sb = sa.clone();
                let (mut xa, mut xb) = (vec![0i16; sub], vec![0i16; sub]);
                lpc_synthesis::<16>(&mut sa, &a_q12, &r, &mut xa, 12345);
                lpc_synthesis_scalar::<16>(&mut sb, &a_q12, &r, &mut xb, 12345, 0);
                assert_eq!((&sa, &xa), (&sb, &xb), "lpc_synthesis 16 sub {sub}");
                let mut sa = hist.clone();
                sa.resize(16 + sub, 0);
                let mut sb = sa.clone();
                lpc_synthesis::<10>(&mut sa, &a_q12, &r, &mut xa, -4321);
                lpc_synthesis_scalar::<10>(&mut sb, &a_q12, &r, &mut xb, -4321, 0);
                assert_eq!((&sa, &xa), (&sb, &xb), "lpc_synthesis 10 sub {sub}");
            }
        }
    }

    #[test]
    fn alloc_psum_matches_scalar() {
        use super::super::rate::alloc_psum_scalar;
        for seed in 0..400u32 {
            let r: Vec<i32> = signal(63, seed, 4000).iter().map(|v| v % 4000).collect();
            let tmp: [i32; 21] = std::array::from_fn(|j| r[j]);
            let thresh: Vec<i32> = r[21..42].iter().map(|v| v.abs() % 300).collect();
            let cap: Vec<i32> = r[42..63].iter().map(|v| v.abs()).collect();
            for (start, end) in [(0usize, 21usize), (17, 21), (0, 13), (5, 6), (3, 3)] {
                for floor in [8, 16] {
                    let want = alloc_psum_scalar(&tmp, &thresh, &cap, start, end, floor);
                    #[cfg(target_arch = "x86_64")]
                    if ragdoll_core::dispatch::avx2() {
                        // Safety: AVX2 is available.
                        let got = unsafe {
                            super::alloc_psum_avx2(&tmp, &thresh, &cap, start, end, floor)
                        };
                        assert_eq!(got, want, "seed {seed} {start}..{end}");
                    }
                    let _ = want;
                }
            }
        }
    }

    #[test]
    fn bitrev_tables_are_permutations() {
        for st in &FFT_STATES {
            let mut seen = vec![false; st.bitrev.len()];
            for &r in st.bitrev {
                assert!(!std::mem::replace(&mut seen[r as usize], true));
            }
        }
    }

    #[test]
    fn pre_rotate_matches_scalar() {
        use super::super::mdct::{pre_rotate, pre_rotate_scalar};
        for (st, n2) in FFT_STATES.iter().zip([960usize, 480, 240, 120]) {
            let t: Vec<i16> = signal(n2, 11, 32767).iter().map(|&v| v as i16).collect();
            for stride in [1usize, 2, 4, 8] {
                let x = signal(stride * n2, 12, i32::MAX);
                let mut a = vec![0i32; n2];
                let mut b = vec![0i32; n2];
                pre_rotate_scalar(&x, &mut a, &t, st.bitrev, stride, 0);
                pre_rotate(&x, &mut b, &t, st.bitrev, stride);
                assert_eq!(a, b, "n2 {n2} stride {stride}");
            }
        }
    }

    #[test]
    fn post_rotate_matches_scalar() {
        for n2 in [960usize, 480, 240, 120] {
            let t: Vec<i16> = signal(n2, 9, 32767).iter().map(|&v| v as i16).collect();
            for (seed, amp) in [(5, 1 << 14), (6, 1 << 30), (7, i32::MAX)] {
                let x = signal(n2, seed, amp);
                let mut a = x.clone();
                let mut b = x.clone();
                post_rotate_scalar(&mut a, &t);
                super::super::mdct::post_rotate(&mut b, &t);
                assert_eq!(a, b, "n2 {n2}");
            }
        }
    }
}
