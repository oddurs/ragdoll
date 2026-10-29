//! AVX2 and NEON kernels of the fixed-point CELT encoder, each bit-exact with the scalar
//! reference it replaces: the PVQ pulse search and the pitch analysis correlations.

#![allow(clippy::too_many_arguments)]

#[cfg(target_arch = "x86_64")]
use super::bands::MAX_BAND;

/// AVX2 helpers on eight 32-bit lanes.
#[cfg(target_arch = "x86_64")]
mod avx2_i32 {
    use std::arch::x86_64::*;

    /// The low 16 bits of each lane, sign-extended (`as i16 as i32`).
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn sext16(v: __m256i) -> __m256i {
        _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(v))
    }

    /// `MULT16_16(a, b)` of lanes holding 16-bit values: `pmaddwd` with the high half of `a`
    /// cleared adds nothing to the low product.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn mul16(a: __m256i, b: __m256i) -> __m256i {
        _mm256_madd_epi16(_mm256_and_si256(a, _mm256_set1_epi32(0xffff)), b)
    }

    /// Wrapping sum of the lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn hsum(v: __m256i) -> i32 {
        let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
        _mm_cvtsi128_si32(s)
    }

    /// Per lane, the larger of the fractions `an / ad` and `bn / bd` (16-bit numerators,
    /// positive 16-bit denominators), the first on ties.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn larger(an: __m256i, ad: __m256i, bn: __m256i, bd: __m256i) -> (__m256i, __m256i) {
        let b = _mm256_cmpgt_epi32(mul16(bn, ad), mul16(an, bd));
        (_mm256_blendv_epi8(an, bn, b), _mm256_blendv_epi8(ad, bd, b))
    }

    /// Minimum of the lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn hmin(v: __m256i) -> i32 {
        let s = _mm_min_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
        let s = _mm_min_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
        let s = _mm_min_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
        _mm_cvtsi128_si32(s)
    }
}

/// AVX2 [`super::vq::pvq_search_scalar`] for `x` of exactly `n <= MAX_BAND` values.
///
/// Each pulse goes to the first `j` maximising `rxy_j / ryy_j`, which the scalar loop finds
/// with the cross-multiplied test `best_den * rxy > ryy * best_num`. While every `ryy_j` is
/// positive that test is the exact order of the fractions (the 16-bit products fit 32 bits),
/// so each lane keeps the best of its own values (the earlier on ties), and the lanes are then
/// merged, the lowest index winning among equal fractions. Should some `ryy_j` not be positive
/// (a 16-bit wrap-around normalised bands never reach), the remaining pulses take the scalar
/// loop. Lanes past `n` hold the fraction -32768 / 1, which no real value is below.
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn pvq_search_avx2(x: &mut [i16], iy: &mut [i32], k: i32, n: usize) -> i32 {
    use super::math::{add16, ilog2, mac16, mul16_32_q16, rcp};
    use super::vq::pvq_pulses;
    use avx2_i32::*;
    use std::arch::x86_64::*;
    use std::mem::MaybeUninit;
    const C: usize = MAX_BAND / 8;
    assert!(n >= 1 && n <= MAX_BAND && x.len() == n && iy.len() >= n);
    let chunks = n.div_ceil(8);
    let tail = n - 8 * (chunks - 1);
    let lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let pad = _mm256_cmpgt_epi32(lanes, _mm256_set1_epi32(tail as i32 - 1));
    let zero = _mm256_setzero_si256();
    // Per chunk of eight: |x|, the signs (0 or 1), y = 2 * iy (low 16 bits) and iy. Only the
    // first `chunks` entries are written and read.
    let mut xb = MaybeUninit::<[__m256i; C]>::uninit();
    let mut sb = MaybeUninit::<[__m256i; C]>::uninit();
    let mut yb = MaybeUninit::<[__m256i; C]>::uninit();
    let mut iyb = MaybeUninit::<[__m256i; C]>::uninit();
    let (xv, sv, yv, iyv) = (
        xb.as_mut_ptr() as *mut __m256i,
        sb.as_mut_ptr() as *mut __m256i,
        yb.as_mut_ptr() as *mut __m256i,
        iyb.as_mut_ptr() as *mut __m256i,
    );
    let xp = x.as_mut_ptr();
    // Safety: chunk `c < chunks` of the buffers is in bounds, and of `x` the first `len`
    // values are; every buffer entry is written before it is read.
    unsafe {
        for c in 0..chunks {
            let len = if c + 1 == chunks { tail } else { 8 };
            let mut t = [0i16; 8];
            std::ptr::copy_nonoverlapping(xp.add(8 * c), t.as_mut_ptr(), len);
            let v = _mm_loadu_si128(t.as_ptr() as *const __m128i);
            // abs_epi16 wraps -32768 to itself, as `wrapping_abs` does.
            let a = _mm_abs_epi16(v);
            _mm_storeu_si128(t.as_mut_ptr() as *mut __m128i, a);
            std::ptr::copy_nonoverlapping(t.as_ptr(), xp.add(8 * c), len);
            xv.add(c).write(_mm256_cvtepi16_epi32(a));
            sv.add(c)
                .write(_mm256_cvtepi16_epi32(_mm_srli_epi16::<15>(v)));
            yv.add(c).write(zero);
            iyv.add(c).write(zero);
        }
        let mut xy = 0i32;
        let mut yy = 0i32;
        let mut pulses_left = k;
        if k > (n as i32 >> 1) {
            let mut s = zero;
            for c in 0..chunks {
                s = _mm256_add_epi32(s, *xv.add(c));
            }
            let mut sum = hsum(s);
            if sum <= k {
                *xp = 16384;
                for j in 1..n {
                    *xp.add(j) = 0;
                }
                for c in 0..chunks {
                    *xv.add(c) = zero;
                }
                *xv = _mm256_setr_epi32(16384, 0, 0, 0, 0, 0, 0, 0);
                sum = 16384;
            }
            let r = _mm256_set1_epi32(mul16_32_q16(k, rcp(sum)) as i16 as i32);
            let (mut ayy, mut axy, mut apl) = (zero, zero, zero);
            for c in 0..chunks {
                // iy = MULT16_16_Q15(x, r); y = iy as i16.
                let t = _mm256_srai_epi32::<15>(_mm256_mullo_epi32(*xv.add(c), r));
                let t16 = sext16(t);
                *iyv.add(c) = t;
                *yv.add(c) = _mm256_slli_epi32::<1>(t16);
                ayy = _mm256_add_epi32(ayy, _mm256_mullo_epi32(t16, t16));
                axy = _mm256_add_epi32(axy, _mm256_mullo_epi32(*xv.add(c), t16));
                apl = _mm256_add_epi32(apl, t);
            }
            yy = hsum(ayy) as i16 as i32;
            xy = hsum(axy);
            pulses_left -= hsum(apl);
        }
        if pulses_left > n as i32 + 3 {
            let tmp = pulses_left;
            let y0 = _mm256_extract_epi32::<0>(*yv);
            yy = mac16(yy, tmp, tmp) as i16 as i32;
            yy = mac16(yy, tmp, y0) as i16 as i32;
            *iyv = _mm256_add_epi32(*iyv, _mm256_setr_epi32(pulses_left, 0, 0, 0, 0, 0, 0, 0));
            pulses_left = 0;
        }
        let one = _mm256_set1_epi32(1);
        let two = _mm256_set1_epi32(2);
        let mut i = 0;
        while i < pulses_left {
            let rshift = 1 + ilog2(k - pulses_left + i + 1);
            let yy1 = add16(yy, 1);
            let cnt = _mm_cvtsi32_si128(rshift & 31);
            let (xyb, yyb) = (_mm256_set1_epi32(xy), _mm256_set1_epi32(yy1));
            let (mut bnum, mut bden, mut bc) = (zero, one, zero);
            let mut bad = zero;
            for c in 0..chunks {
                let rxy = sext16(_mm256_sra_epi32(_mm256_add_epi32(xyb, *xv.add(c)), cnt));
                let mut num = sext16(_mm256_srai_epi32::<15>(mul16(rxy, rxy)));
                let mut den = sext16(_mm256_add_epi32(yyb, *yv.add(c)));
                if c + 1 == chunks {
                    num = _mm256_blendv_epi8(num, _mm256_set1_epi32(-32768), pad);
                    den = _mm256_blendv_epi8(den, one, pad);
                }
                bad = _mm256_or_si256(bad, _mm256_cmpgt_epi32(one, den));
                if c == 0 {
                    (bnum, bden) = (num, den);
                } else {
                    let better = _mm256_cmpgt_epi32(mul16(num, bden), mul16(bnum, den));
                    bnum = _mm256_blendv_epi8(bnum, num, better);
                    bden = _mm256_blendv_epi8(bden, den, better);
                    bc = _mm256_blendv_epi8(bc, _mm256_set1_epi32(c as i32), better);
                }
            }
            if _mm256_movemask_epi8(bad) != 0 {
                break;
            }
            // The largest fraction, in every lane.
            let (mut mnum, mut mden) = (bnum, bden);
            (mnum, mden) = larger(
                mnum,
                mden,
                _mm256_permute2x128_si256::<1>(mnum, mnum),
                _mm256_permute2x128_si256::<1>(mden, mden),
            );
            (mnum, mden) = larger(
                mnum,
                mden,
                _mm256_shuffle_epi32::<0b01_00_11_10>(mnum),
                _mm256_shuffle_epi32::<0b01_00_11_10>(mden),
            );
            (mnum, mden) = larger(
                mnum,
                mden,
                _mm256_shuffle_epi32::<0b10_11_00_01>(mnum),
                _mm256_shuffle_epi32::<0b10_11_00_01>(mden),
            );
            let eq = _mm256_cmpeq_epi32(mul16(bnum, mden), mul16(mnum, bden));
            let best = if chunks == 1 {
                _mm256_movemask_ps(_mm256_castsi256_ps(eq)).trailing_zeros() as usize
            } else {
                let idx = _mm256_add_epi32(_mm256_slli_epi32::<3>(bc), lanes);
                hmin(_mm256_blendv_epi8(_mm256_set1_epi32(i32::MAX), idx, eq)) as usize
            };
            let (bv, bl) = (yv.add(best >> 3), (best & 7) as i32);
            xy = xy.wrapping_add(*xp.add(best) as i32);
            yy = add16(yy1, *(bv as *const i32).add(bl as usize));
            let hit = _mm256_cmpeq_epi32(lanes, _mm256_set1_epi32(bl));
            *bv = _mm256_add_epi32(*bv, _mm256_and_si256(hit, two));
            let iv = iyv.add(best >> 3);
            *iv = _mm256_sub_epi32(*iv, hit);
            i += 1;
        }
        if i < pulses_left {
            // A wrapped energy: the scalar loop places the rest.
            let (mut y, mut iys) = ([0i16; MAX_BAND], [0i32; MAX_BAND]);
            for c in 0..chunks {
                let (yc, ic) = (yv.add(c) as *const i32, iyv.add(c) as *const i32);
                for l in 0..8 {
                    y[8 * c + l] = *yc.add(l) as i16;
                    iys[8 * c + l] = *ic.add(l);
                }
            }
            let xs = std::slice::from_raw_parts(xp, n);
            yy = pvq_pulses(xs, &mut y, &mut iys, n, k, pulses_left, i, xy, yy as i16) as i32;
            for c in 0..chunks {
                *iyv.add(c) = _mm256_loadu_si256(iys.as_ptr().add(8 * c) as *const __m256i);
            }
        }
        let ip = iy.as_mut_ptr();
        for c in 0..chunks {
            let len = if c + 1 == chunks { tail } else { 8 };
            let s = *sv.add(c);
            let v = _mm256_add_epi32(_mm256_xor_si256(*iyv.add(c), _mm256_sub_epi32(zero, s)), s);
            if len == 8 {
                _mm256_storeu_si256(ip.add(8 * c) as *mut __m256i, v);
            } else {
                let mut t = [0i32; 8];
                _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, v);
                std::ptr::copy_nonoverlapping(t.as_ptr(), ip.add(8 * c), len);
            }
        }
        yy
    }
}

/// AVX2 [`super::mdct::fold_copy_scalar`] from step `i0`, four steps at a time; returns the
/// first step it left (at most `i1`).
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn fold_copy_avx2(
    input: &[i32],
    f: &mut [i32],
    h: usize,
    i0: usize,
    i1: usize,
) -> usize {
    use std::arch::x86_64::*;
    let n2 = f.len();
    assert!(2 * i1 <= n2 && n2 + h <= input.len());
    let rev = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    let (ip, fp) = (input.as_ptr(), f.as_mut_ptr());
    let mut i = i0;
    while i + 4 <= i1 {
        // Safety: f[2i..2i + 8] is in `f` as i + 4 <= i1 <= n2 / 2; the reads
        // input[h + 2i..h + 2i + 8] end at most at h + 2 i1 <= n2 + h, and
        // input[n2 + h - 2i - 8..n2 + h - 2i] start at least at n2 + h - 2 i1 >= 0.
        unsafe {
            // f[2i + 2k] = input[n2 - 1 + h - 2i - 2k], f[2i + 2k + 1] = input[h + 2i + 2k].
            let a = _mm256_loadu_si256(ip.add(h + 2 * i) as *const __m256i);
            let b = _mm256_loadu_si256(ip.add(n2 + h - 2 * i - 8) as *const __m256i);
            let r = _mm256_permutevar8x32_epi32(b, rev);
            let v = _mm256_blend_epi32::<0xAA>(r, _mm256_slli_epi64::<32>(a));
            _mm256_storeu_si256(fp.add(2 * i) as *mut __m256i, v);
        }
        i += 4;
    }
    i
}

/// `(a * t) >> 15` of the 32-bit values in the even lanes of `a` and the 16-bit `t`, in the
/// low halves of the 64-bit lanes (`MULT16_32_Q15`).
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
fn q15_64(
    a: std::arch::x86_64::__m256i,
    t: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    _mm256_srli_epi64::<15>(_mm256_mul_epi32(a, t))
}

/// AVX2 [`super::mdct::pre_rotate_forward_scalar`], four steps at a time; returns the first
/// step it left.
///
/// # Safety
///
/// The CPU must support AVX2, and the entries of `st.bitrev` must be below `f.len() / 2`
/// (they are a permutation in every [`super::mdct::FftState`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn pre_rotate_forward_avx2(
    f: &[i32],
    f2: &mut [i32],
    t: &[i16],
    st: &super::mdct::FftState,
) -> usize {
    use std::arch::x86_64::*;
    let n2 = f.len();
    let n4 = n2 >> 1;
    assert!(f2.len() == n2 && t.len() >= n2 && st.bitrev.len() == n4);
    debug_assert!(st.bitrev.iter().all(|&r| (r as usize) < n4));
    let sh = st.scale_shift - 1;
    let scale = _mm256_set1_epi32(st.scale as i16 as i32);
    let round = _mm256_set1_epi32((1 << sh) >> 1);
    let cnt = _mm_cvtsi32_si128(sh);
    let (fp, tp, yp) = (f.as_ptr(), t.as_ptr(), f2.as_mut_ptr());
    let mut i = 0;
    while i + 4 <= n4 {
        // Safety: the reads are of f[2i..2i + 8] and t[i..i + 4], t[n4 + i..n4 + i + 4]; the
        // stores of pairs at 2 bitrev[i + k] < n2.
        unsafe {
            let v = _mm256_loadu_si256(fp.add(2 * i) as *const __m256i);
            let (re, im) = (v, _mm256_srli_epi64::<32>(v));
            let t0 = _mm256_cvtepi16_epi64(_mm_loadl_epi64(tp.add(i) as *const __m128i));
            let t1 = _mm256_cvtepi16_epi64(_mm_loadl_epi64(tp.add(n4 + i) as *const __m128i));
            let yr = _mm256_sub_epi64(q15_64(re, t0), q15_64(im, t1));
            let yi = _mm256_add_epi64(q15_64(im, t0), q15_64(re, t1));
            // MULT16_32_Q16(scale, y), then PSHR32.
            let yr = _mm256_srli_epi64::<16>(_mm256_mul_epi32(yr, scale));
            let yi = _mm256_srli_epi64::<16>(_mm256_mul_epi32(yi, scale));
            let y = _mm256_blend_epi32::<0xAA>(yr, _mm256_slli_epi64::<32>(yi));
            let y = _mm256_sra_epi32(_mm256_add_epi32(y, round), cnt);
            let r = |k: usize| 2 * *st.bitrev.get_unchecked(i + k) as usize;
            let lo = _mm256_castsi256_si128(y);
            let hi = _mm256_extracti128_si256::<1>(y);
            _mm_storel_epi64(yp.add(r(0)) as *mut __m128i, lo);
            _mm_storeh_pd(yp.add(r(1)) as *mut f64, _mm_castsi128_pd(lo));
            _mm_storel_epi64(yp.add(r(2)) as *mut __m128i, hi);
            _mm_storeh_pd(yp.add(r(3)) as *mut f64, _mm_castsi128_pd(hi));
        }
        i += 4;
    }
    i
}

/// AVX2 [`super::mdct::post_rotate_forward_scalar`] with stride 1: steps `i` from the front and
/// `n4 - 1 - i` from the back, four of each at a time; returns how many front steps it did
/// (the scalar loop does the steps between).
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn post_rotate_forward_avx2(f2: &[i32], out: &mut [i32], t: &[i16]) -> usize {
    use std::arch::x86_64::*;
    let n2 = f2.len();
    let n4 = n2 >> 1;
    assert!(out.len() >= n2 && t.len() >= n2);
    let swap = _mm256_setr_epi32(6, 7, 4, 5, 2, 3, 0, 1);
    let (fp, tp, op) = (f2.as_ptr(), t.as_ptr(), out.as_mut_ptr());
    // (yr, yi) of the four steps from `i`, interleaved.
    let rot = |i: usize| {
        // Safety: the callers' steps are below n4.
        unsafe {
            let v = _mm256_loadu_si256(fp.add(2 * i) as *const __m256i);
            let (re, im) = (v, _mm256_srli_epi64::<32>(v));
            let t0 = _mm256_cvtepi16_epi64(_mm_loadl_epi64(tp.add(i) as *const __m128i));
            let t1 = _mm256_cvtepi16_epi64(_mm_loadl_epi64(tp.add(n4 + i) as *const __m128i));
            let yr = _mm256_sub_epi64(q15_64(im, t1), q15_64(re, t0));
            let yi = _mm256_add_epi64(q15_64(re, t1), q15_64(im, t0));
            _mm256_blend_epi32::<0xAA>(yr, _mm256_slli_epi64::<32>(yi))
        }
    };
    let mut i = 0;
    while i + 4 <= n4 / 2 {
        let (a, b) = (rot(i), rot(n4 - 4 - i));
        // out[2i + 2k] = yr(i + k) and out[2i + 2k + 1] = yi(n4 - 1 - i - k); at the back,
        // out[n2 - 2i - 8 + 2k] = yr(n4 - 4 - i + k) and the odd values yi(i + 3 - k).
        let front = _mm256_blend_epi32::<0xAA>(a, _mm256_permutevar8x32_epi32(b, swap));
        let back = _mm256_blend_epi32::<0xAA>(b, _mm256_permutevar8x32_epi32(a, swap));
        // Safety: the front values 2i..2i + 8 and the back ones n2 - 2i - 8..n2 - 2i are in
        // `out` and apart, as i + 4 <= n4 / 2.
        unsafe {
            _mm256_storeu_si256(op.add(2 * i) as *mut __m256i, front);
            _mm256_storeu_si256(op.add(n2 - 2 * i - 8) as *mut __m256i, back);
        }
        i += 4;
    }
    i
}

/// AVX2 [`super::celt_enc::preemphasis_scalar`] of samples `1..`, eight at a time (each needs
/// only its input and the one before); returns the first sample it left (at least 1).
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn preemphasis_avx2(pcm: &[i16], cc: usize, ch: usize, out: &mut [i32]) -> usize {
    use std::arch::x86_64::*;
    let n = out.len();
    assert!((1..=2).contains(&cc) && ch < cc && pcm.len() >= cc * n);
    let coef = _mm256_set1_epi32(27853);
    let (pp, op) = (pcm.as_ptr(), out.as_mut_ptr());
    // Samples i..i + 8 of the channel, sign-extended.
    let load = |i: usize| {
        // Safety: the callers' samples are below n, so the reads below cc * n.
        unsafe {
            if cc == 1 {
                _mm256_cvtepi16_epi32(_mm_loadu_si128(pp.add(i) as *const __m128i))
            } else {
                let v = _mm256_loadu_si256(pp.add(2 * i) as *const __m256i);
                if ch == 0 {
                    _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(v))
                } else {
                    _mm256_srai_epi32::<16>(v)
                }
            }
        }
    };
    let mut i = 1;
    while i + 8 <= n {
        let (x, prev) = (load(i), load(i - 1));
        // SHL32(x, SIG_SHIFT) - MULT16_16(PREEMPH, prev) >> (15 - SIG_SHIFT).
        let m = _mm256_srai_epi32::<3>(_mm256_mullo_epi32(prev, coef));
        let v = _mm256_sub_epi32(_mm256_slli_epi32::<12>(x), m);
        // Safety: out[i..i + 8] is in `out`.
        unsafe { _mm256_storeu_si256(op.add(i) as *mut __m256i, v) };
        i += 8;
    }
    i
}

/// AVX2 `celt_maxabs32`.
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn maxabs32_avx2(x: &[i32]) -> i32 {
    use std::arch::x86_64::*;
    let n8 = x.len() & !7;
    let (mut hi, mut lo) = (_mm256_setzero_si256(), _mm256_setzero_si256());
    for c in x[..n8].chunks_exact(8) {
        // Safety: `c` holds 8 values.
        let v = unsafe { _mm256_loadu_si256(c.as_ptr() as *const __m256i) };
        hi = _mm256_max_epi32(hi, v);
        lo = _mm256_min_epi32(lo, v);
    }
    let (mut a, mut b) = ([0i32; 8], [0i32; 8]);
    // Safety: `a` and `b` hold 8 values.
    unsafe {
        _mm256_storeu_si256(a.as_mut_ptr() as *mut __m256i, hi);
        _mm256_storeu_si256(b.as_mut_ptr() as *mut __m256i, lo);
    }
    let mut maxval = a.iter().fold(0, |m, &v| v.max(m));
    let mut minval = b.iter().fold(0, |m, &v| v.min(m));
    for &v in &x[n8..] {
        maxval = maxval.max(v);
        minval = minval.min(v);
    }
    maxval.max(minval.wrapping_neg())
}

/// AVX2 [`super::plc::decimate_scalar`] from output 1, eight outputs at a time; returns the
/// first output it left.
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn decimate_avx2(xs: &[i32], x_lp: &mut [i16], shift: i32, add: bool) -> usize {
    use std::arch::x86_64::*;
    let half = x_lp.len();
    assert!(xs.len() >= 2 * half);
    let cnt = _mm_cvtsi32_si128(shift);
    let (xp, op) = (xs.as_ptr(), x_lp.as_mut_ptr());
    // The even and the odd lanes of a and b, in order.
    let even = |a: __m256i, b: __m256i| {
        let v = _mm256_shuffle_ps::<0x88>(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b));
        _mm256_permute4x64_epi64::<0xD8>(_mm256_castps_si256(v))
    };
    let odd = |a: __m256i, b: __m256i| {
        let v = _mm256_shuffle_ps::<0xDD>(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b));
        _mm256_permute4x64_epi64::<0xD8>(_mm256_castps_si256(v))
    };
    let mut i = 1;
    while i + 8 <= half {
        // Safety: the reads xs[2i - 1..2i + 16] stay below 2 half, the 8 outputs below half.
        unsafe {
            let ld = |p: *const i32| _mm256_loadu_si256(p as *const __m256i);
            let (a0, a1) = (ld(xp.add(2 * i)), ld(xp.add(2 * i + 8)));
            let (b0, b1) = (ld(xp.add(2 * i - 1)), ld(xp.add(2 * i + 7)));
            let (e, o, op1) = (even(a0, a1), odd(a0, a1), even(b0, b1));
            let s = _mm256_srai_epi32::<1>(_mm256_add_epi32(op1, o));
            let v = _mm256_sra_epi32(_mm256_srai_epi32::<1>(_mm256_add_epi32(s, e)), cnt);
            // The low 16 bits of each lane, in order.
            let v = _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(v));
            let v = _mm256_permute4x64_epi64::<0x08>(_mm256_packs_epi32(v, v));
            let mut v = _mm256_castsi256_si128(v);
            let o = op.add(i) as *mut __m128i;
            if add {
                v = _mm_add_epi16(v, _mm_loadu_si128(o));
            }
            _mm_storeu_si128(o, v);
        }
        i += 8;
    }
    i
}

/// AVX2 [`super::plc::fir5_scalar`] on all but a prefix of `x`, sixteen outputs at a time from
/// the end down (so each block still reads its inputs); returns the length of the prefix left.
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn fir5_avx2(x: &mut [i16], num: &[i32; 5]) -> usize {
    use std::arch::x86_64::*;
    let n = x.len();
    if n < 21 {
        return n;
    }
    let blocks = (n - 5) / 16;
    let start = n - 16 * blocks;
    // Coefficient pairs for pmaddwd against (x[i - 1], x[i - 2]), (x[i - 3], x[i - 4]) and
    // (x[i - 5], 0).
    let pair = |a: i32, b: i32| _mm256_set1_epi32((b << 16) | (a & 0xffff));
    let (c01, c23, c4) = (pair(num[0], num[1]), pair(num[2], num[3]), pair(num[4], 0));
    let round = _mm256_set1_epi32(1 << 11);
    let z = _mm256_setzero_si256();
    let xp = x.as_mut_ptr();
    for b in (0..blocks).rev() {
        let i = start + 16 * b;
        // Safety: the reads x[i - 5..i + 16] and the store x[i..i + 16] are in `x` (i >= 5).
        unsafe {
            let ld = |k: usize| _mm256_loadu_si256(xp.add(i - k) as *const __m256i);
            let (u0, u1, u2, u3, u4, u5) = (ld(0), ld(1), ld(2), ld(3), ld(4), ld(5));
            // SHL32(x[i], 12) + the taps, in unpacklo / unpackhi order.
            let half = |x0: __m256i, p1: __m256i, p2: __m256i, p3: __m256i| {
                let s = _mm256_srai_epi32::<4>(x0);
                let s = _mm256_add_epi32(s, _mm256_madd_epi16(p1, c01));
                let s = _mm256_add_epi32(s, _mm256_madd_epi16(p2, c23));
                let s = _mm256_add_epi32(s, _mm256_madd_epi16(p3, c4));
                // ROUND16(s, 12), truncated.
                let s = _mm256_srai_epi32::<12>(_mm256_add_epi32(s, round));
                _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(s))
            };
            let lo = half(
                _mm256_unpacklo_epi16(z, u0),
                _mm256_unpacklo_epi16(u1, u2),
                _mm256_unpacklo_epi16(u3, u4),
                _mm256_unpacklo_epi16(u5, z),
            );
            let hi = half(
                _mm256_unpackhi_epi16(z, u0),
                _mm256_unpackhi_epi16(u1, u2),
                _mm256_unpackhi_epi16(u3, u4),
                _mm256_unpackhi_epi16(u5, z),
            );
            _mm256_storeu_si256(xp.add(i) as *mut __m256i, _mm256_packs_epi32(lo, hi));
        }
    }
    start
}

/// AVX2 [`super::plc::pitch_xcorr_scalar`]: `xcorr[i] = sum_j x[j] * y[i + j]` for
/// `i < max_pitch` and `j < len`; returns the largest, at least 1. `pmaddwd` forms the
/// products in pairs; integer sums wrap the same in any order, so four lags at a time share
/// the loads of `x` and reduce together.
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn pitch_xcorr_avx2(
    x: &[i16],
    y: &[i16],
    xcorr: &mut [i32],
    len: usize,
    max_pitch: usize,
) -> i32 {
    use std::arch::x86_64::*;
    assert!(x.len() >= len && y.len() + 1 >= len + max_pitch && xcorr.len() >= max_pitch);
    let n16 = len & !15;
    let (xp, yp) = (x.as_ptr(), y.as_ptr());
    let tail = |i: usize, mut s: i32| {
        for j in n16..len {
            s = s.wrapping_add(x[j] as i32 * y[i + j] as i32);
        }
        s
    };
    let mut maxcorr = 1i32;
    let mut i = 0;
    // Safety: every load is of 16 values of `x` below `len`, or of `y` below `i + 3 + len`
    // with `i + 3 < max_pitch`.
    unsafe {
        let ld = |p: *const i16| _mm256_loadu_si256(p as *const __m256i);
        while i + 4 <= max_pitch {
            let z = _mm256_setzero_si256();
            let (mut s0, mut s1, mut s2, mut s3) = (z, z, z, z);
            let mut j = 0;
            while j < n16 {
                let xv = ld(xp.add(j));
                let yq = yp.add(i + j);
                s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(xv, ld(yq)));
                s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(xv, ld(yq.add(1))));
                s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(xv, ld(yq.add(2))));
                s3 = _mm256_add_epi32(s3, _mm256_madd_epi16(xv, ld(yq.add(3))));
                j += 16;
            }
            let h = _mm256_hadd_epi32(_mm256_hadd_epi32(s0, s1), _mm256_hadd_epi32(s2, s3));
            let h = _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256::<1>(h));
            let mut s = [0i32; 4];
            _mm_storeu_si128(s.as_mut_ptr() as *mut __m128i, h);
            for (q, &v) in s.iter().enumerate() {
                let v = tail(i + q, v);
                xcorr[i + q] = v;
                maxcorr = maxcorr.max(v);
            }
            i += 4;
        }
        while i < max_pitch {
            let v = tail(i, dot16_avx2(&x[..n16], &y[i..i + n16]));
            xcorr[i] = v;
            maxcorr = maxcorr.max(v);
            i += 1;
        }
    }
    maxcorr
}

/// AVX2 wrapping dot product `sum_j a[j] * b[j]` of equal-length slices (`inner_prod`).
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dot16_avx2(a: &[i16], b: &[i16]) -> i32 {
    use std::arch::x86_64::*;
    let n = a.len();
    assert!(b.len() == n);
    let n16 = n & !15;
    let mut s = _mm256_setzero_si256();
    // Safety: every load is of 16 values below `n`.
    unsafe {
        let mut j = 0;
        while j < n16 {
            let av = _mm256_loadu_si256(a.as_ptr().add(j) as *const __m256i);
            let bv = _mm256_loadu_si256(b.as_ptr().add(j) as *const __m256i);
            s = _mm256_add_epi32(s, _mm256_madd_epi16(av, bv));
            j += 16;
        }
    }
    let mut v = avx2_i32::hsum(s);
    for j in n16..n {
        v = v.wrapping_add(a[j] as i32 * b[j] as i32);
    }
    v
}

/// AVX2 `sum_j (a[j] * b[j]) >> shift` (wrapping) of equal-length slices, `shift < 32`.
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dot16_shr_avx2(a: &[i16], b: &[i16], shift: i32) -> i32 {
    use std::arch::x86_64::*;
    let n = a.len();
    assert!(b.len() == n && (0..32).contains(&shift));
    let n16 = n & !15;
    let cnt = _mm_cvtsi32_si128(shift);
    let mut s = _mm256_setzero_si256();
    // Safety: every load is of 16 values below `n`.
    unsafe {
        let mut j = 0;
        while j < n16 {
            let av = _mm256_loadu_si256(a.as_ptr().add(j) as *const __m256i);
            let bv = _mm256_loadu_si256(b.as_ptr().add(j) as *const __m256i);
            // The 32-bit products, each shifted on its own.
            let lo = _mm256_mullo_epi16(av, bv);
            let hi = _mm256_mulhi_epi16(av, bv);
            let p0 = _mm256_sra_epi32(_mm256_unpacklo_epi16(lo, hi), cnt);
            let p1 = _mm256_sra_epi32(_mm256_unpackhi_epi16(lo, hi), cnt);
            s = _mm256_add_epi32(s, _mm256_add_epi32(p0, p1));
            j += 16;
        }
    }
    let mut v = avx2_i32::hsum(s);
    for j in n16..n {
        v = v.wrapping_add((a[j] as i32 * b[j] as i32) >> shift);
    }
    v
}

/// NEON [`super::plc::pitch_xcorr_scalar`].
///
/// # Safety
///
/// NEON must be available (it is on every aarch64 CPU).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn pitch_xcorr_neon(
    x: &[i16],
    y: &[i16],
    xcorr: &mut [i32],
    len: usize,
    max_pitch: usize,
) -> i32 {
    assert!(x.len() >= len && y.len() + 1 >= len + max_pitch && xcorr.len() >= max_pitch);
    let mut maxcorr = 1i32;
    for i in 0..max_pitch {
        // Safety: NEON is available.
        let v = unsafe { dot16_neon(&x[..len], &y[i..i + len]) };
        xcorr[i] = v;
        maxcorr = maxcorr.max(v);
    }
    maxcorr
}

/// NEON wrapping dot product `sum_j a[j] * b[j]` of equal-length slices.
///
/// # Safety
///
/// NEON must be available (it is on every aarch64 CPU).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dot16_neon(a: &[i16], b: &[i16]) -> i32 {
    use std::arch::aarch64::*;
    let n = a.len();
    assert!(b.len() == n);
    let n8 = n & !7;
    let mut s = vdupq_n_s32(0);
    // Safety: every load is of 8 values below `n`.
    unsafe {
        let mut j = 0;
        while j < n8 {
            let av = vld1q_s16(a.as_ptr().add(j));
            let bv = vld1q_s16(b.as_ptr().add(j));
            s = vmlal_s16(s, vget_low_s16(av), vget_low_s16(bv));
            s = vmlal_high_s16(s, av, bv);
            j += 8;
        }
    }
    // Lane sums wrap like the scalar sum: reduce in 32 bits.
    let mut v = vgetq_lane_s32::<0>(s)
        .wrapping_add(vgetq_lane_s32::<1>(s))
        .wrapping_add(vgetq_lane_s32::<2>(s))
        .wrapping_add(vgetq_lane_s32::<3>(s));
    for j in n8..n {
        v = v.wrapping_add(a[j] as i32 * b[j] as i32);
    }
    v
}

#[cfg(test)]
mod tests {
    /// Deterministic test values: `amp`-bounded noise with extreme values mixed in.
    fn signal16(n: usize, seed: u32, amp: i32) -> Vec<i16> {
        let mut s = seed;
        (0..n)
            .map(|k| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                match (k as u32 + seed) % 41 {
                    0 => i16::MIN,
                    1 => i16::MAX,
                    _ => (((s as i32) >> 12) % amp.max(1)) as i16,
                }
            })
            .collect()
    }

    #[test]
    fn pvq_search_matches_scalar() {
        use super::super::vq::{pvq_search, pvq_search_scalar};
        for n in (1..=40).chain([48, 64, 72, 96, 128, 144, 176]) {
            for k in [1usize, 2, 3, 5, 8, 13, 31, 64, 100, 181, 255, 600] {
                for (seed, amp) in [(1, 16384), (2, 3000), (3, 40), (4, 32768), (5, 1)] {
                    let x = signal16(n, seed * 977 + n as u32 + k as u32, amp);
                    let (mut xa, mut xb) = (x.clone(), x.clone());
                    let (mut ia, mut ib) = (vec![7i32; n], vec![-7i32; n]);
                    let ya = pvq_search(&mut xa, &mut ia, k as i32, n);
                    let yb = pvq_search_scalar(&mut xb, &mut ib, k as i32, n);
                    assert_eq!((ya, &ia, &xa), (yb, &ib, &xb), "n {n} k {k} amp {amp}");
                }
            }
        }
    }

    #[test]
    fn mdct_forward_matches_scalar() {
        use super::super::celt_tables::WINDOW;
        use super::super::mdct::{mdct_forward, mdct_forward_scalar};
        let mut seed = 3u32;
        let mut rnd = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed as i32
        };
        for shift in 0..4 {
            let n2 = 960 >> shift;
            for stride in [1usize, 2, 4, 8] {
                // Pre-emphasised signal levels, full scale and beyond, and extremes.
                for amp in [1 << 12, 1 << 20, 1 << 27, i32::MAX] {
                    let input: Vec<i32> = (0..n2 + 120)
                        .map(|k| match k % 53 {
                            0 => i32::MIN,
                            1 => i32::MAX,
                            _ => rnd() % amp,
                        })
                        .collect();
                    let (mut a, mut b) = (vec![0i32; n2 * stride], vec![0i32; n2 * stride]);
                    let (mut f, mut f2) = (vec![0i32; 960], vec![0i32; 960]);
                    mdct_forward(&input, &mut a, &WINDOW, 120, shift, stride, &mut f, &mut f2);
                    mdct_forward_scalar(&input, &mut b, &WINDOW, 120, shift, stride);
                    assert_eq!(a, b, "shift {shift} stride {stride} amp {amp}");
                }
            }
        }
    }

    #[test]
    fn pitch_downsample_kernels_match_scalar() {
        use super::super::math::maxabs32;
        use super::super::plc::{decimate_scalar, fir5, fir5_scalar};
        let mut seed = 5u32;
        let mut rnd = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed as i32
        };
        for n in (1..60).chain([255, 256, 257, 992, 1000]) {
            for amp in [1 << 10, 1 << 24, i32::MAX] {
                let xs: Vec<i32> = (0..2 * n + 1)
                    .map(|k| match k % 31 {
                        0 => i32::MIN,
                        1 => i32::MAX,
                        _ => rnd() % amp,
                    })
                    .collect();
                assert_eq!(
                    super::super::plc::maxabs32_long(&xs),
                    maxabs32(&xs),
                    "maxabs n {n}"
                );
                for (shift, add) in [(0, false), (1, true), (11, false), (22, true)] {
                    let init: Vec<i16> = (0..n).map(|_| rnd() as i16).collect();
                    let (mut a, mut b) = (init.clone(), init.clone());
                    super::super::plc::decimate(&xs, &mut a, shift, add);
                    decimate_scalar(&xs, &mut b, shift, add, 1);
                    assert_eq!(a, b, "decimate n {n} shift {shift}");
                }
            }
            let x = signal16(n, n as u32 + 77, 32768);
            for num in [
                [3277, -8000, 12000, -16384, 300],
                [-32768, 32767, -32768, 32767, -32768],
                [26214, 0, 0, 0, 0],
            ] {
                let (mut a, mut b) = (x.clone(), x.clone());
                fir5(&mut a, &num, n);
                fir5_scalar(&mut b, &num, n);
                assert_eq!(a, b, "fir5 n {n}");
            }
        }
    }

    #[test]
    fn preemphasis_matches_scalar() {
        use super::super::celt_enc::{preemphasis, preemphasis_scalar};
        for n in (1..40).chain([120, 240, 480, 960]) {
            for cc in [1usize, 2] {
                let pcm = signal16(cc * n, (n * cc) as u32, 32768);
                for ch in 0..cc {
                    for mem in [0, 12345, -(1 << 20), i32::MIN] {
                        let (mut a, mut b) = (vec![0i32; n], vec![0i32; n]);
                        let ma = preemphasis(&pcm, cc, ch, &mut a, mem);
                        let mb = preemphasis_scalar(&pcm, cc, ch, &mut b, mem, 0);
                        assert_eq!((ma, &a), (mb, &b), "n {n} cc {cc} ch {ch}");
                    }
                }
            }
        }
    }

    #[test]
    fn correlations_match_scalar() {
        use super::super::math::inner_prod;
        use super::super::plc::{dot_shr, dot_shr_scalar, dot16, pitch_xcorr, pitch_xcorr_scalar};
        // Odd lengths and lag counts, the encoder's sizes, and full-scale values (whose sums
        // wrap).
        for len in (1..40).chain([63, 64, 65, 240, 480, 988]) {
            for max_pitch in [1usize, 2, 3, 4, 5, 7, 9, 25, 244] {
                for (seed, amp) in [(1, 32768), (2, 2048), (3, 9)] {
                    let x = signal16(len, seed + len as u32, amp);
                    let y = signal16(len + max_pitch, seed * 31 + max_pitch as u32, amp);
                    let (mut a, mut b) = (vec![0i32; max_pitch], vec![0i32; max_pitch]);
                    let ma = pitch_xcorr(&x, &y, &mut a, len, max_pitch);
                    let mb = pitch_xcorr_scalar(&x, &y, &mut b, len, max_pitch);
                    assert_eq!((ma, &a), (mb, &b), "xcorr len {len} lags {max_pitch}");
                }
            }
            let x = signal16(len, 7 * len as u32, 32768);
            let y = signal16(len, 11 * len as u32 + 3, 32768);
            assert_eq!(dot16(&x, &y), inner_prod(&x, &y), "dot16 len {len}");
            for shift in [0, 1, 2, 7, 8, 15, 31] {
                assert_eq!(
                    dot_shr(&x, &y, shift),
                    dot_shr_scalar(&x, &y, shift),
                    "dot_shr len {len} shift {shift}"
                );
            }
        }
    }
}
