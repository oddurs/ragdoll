//! H.264 encoder kernels of the $8\times8$ transform (`SIMD-0089`, `x264` `sub16x8_dct8` and
//! `quant_8x8`):
//!
//! - [`sub_dct8x8_x2`]: forward $8\times8$ transforms of the residuals of two $8\times8$ blocks
//!   side by side, both at once in the two lanes of AVX2 registers.
//! - [`quant_8x8`]: quantization of one $8\times8$ block with a multiplier per position.
//!
//! Every kernel has a `_scalar` reference that the AVX2 and NEON paths match bit for bit. For
//! 8-bit samples every intermediate of the transform fits 16 bits (at most $8.75^2 \cdot 255$),
//! so the SIMD paths compute in 16-bit lanes what the reference computes in 32 bits.

/// Forward $8\times8$ transforms (`forward_dct8x8`) of `src - pred` for the two $8\times8$
/// blocks at columns 0 and 8 of the $16\times8$ regions at the start of `src` (stride `ss`) and
/// `pred` (stride `ps`), left block first, coefficients in raster order.
///
/// # Panics
/// If either buffer is shorter than 8 rows of 16 samples at its stride.
#[inline]
pub fn sub_dct8x8_x2(src: &[u8], ss: usize, pred: &[u8], ps: usize, out: &mut [[i16; 64]; 2]) {
    assert!(7 * ss + 16 <= src.len() && 7 * ps + 16 <= pred.len());
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above; NEON is baseline on aarch64.
        unsafe {
            for (b, o) in out.iter_mut().enumerate() {
                *o = neon::sub_dct8(src.as_ptr().add(8 * b), ss, pred.as_ptr().add(8 * b), ps);
            }
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        unsafe { x86::sub_dct8_x2(src.as_ptr(), ss, pred.as_ptr(), ps, out) };
        return;
    }
    sub_dct8x8_x2_scalar(src, ss, pred, ps, out);
}

/// Scalar reference for [`sub_dct8x8_x2`].
pub fn sub_dct8x8_x2_scalar(
    src: &[u8],
    ss: usize,
    pred: &[u8],
    ps: usize,
    out: &mut [[i16; 64]; 2],
) {
    for (b, o) in out.iter_mut().enumerate() {
        let mut d = [0i16; 64];
        for y in 0..8 {
            for x in 0..8 {
                d[y * 8 + x] = src[y * ss + 8 * b + x] as i16 - pred[y * ps + 8 * b + x] as i16;
            }
        }
        *o = crate::forward_dct8x8(&d);
    }
}

/// Quantizes the $8\times8$ block `c` (raster order) in place:
/// `level = sign(c) min((|c| mf[i] + bias) >> shift, 32767)`. Returns bit 0 set when a level is
/// non-zero and bit 1 when one exceeds 1 in magnitude.
///
/// # Panics
/// If `shift` is not in `1..=31`, `bias` is not below `1 << shift` or a multiplier exceeds 15
/// bits.
#[inline]
pub fn quant_8x8(c: &mut [i16; 64], mf: &[u16; 64], bias: u32, shift: u32) -> u32 {
    assert!((1..32).contains(&shift) && bias < 1 << shift);
    assert!(mf.iter().fold(0, |a, &m| a | m) <= i16::MAX as u16);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: NEON is baseline on aarch64.
        return unsafe { neon::quant8(c, mf, bias, shift) };
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        return unsafe { x86::quant8(c, mf, bias, shift) };
    }
    quant_8x8_scalar(c, mf, bias, shift)
}

/// Scalar reference for [`quant_8x8`].
pub fn quant_8x8_scalar(c: &mut [i16; 64], mf: &[u16; 64], bias: u32, shift: u32) -> u32 {
    let mut nz = 0;
    for (v, &m) in c.iter_mut().zip(mf) {
        let level =
            ((v.unsigned_abs() as u64 * m as u64 + bias as u64) >> shift).min(i16::MAX as u64);
        nz |= u32::from(level != 0) | (u32::from(level > 1) << 1);
        *v = if *v < 0 {
            -(level as i16)
        } else {
            level as i16
        };
    }
    nz
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    /// Transposes the $8\times8$ block of 16-bit values held in each 128-bit lane of `r`
    /// (register `i` row `i`).
    #[inline]
    #[target_feature(enable = "avx2")]
    fn transpose(r: [__m256i; 8]) -> [__m256i; 8] {
        let a = [
            _mm256_unpacklo_epi16(r[0], r[1]),
            _mm256_unpackhi_epi16(r[0], r[1]),
            _mm256_unpacklo_epi16(r[2], r[3]),
            _mm256_unpackhi_epi16(r[2], r[3]),
            _mm256_unpacklo_epi16(r[4], r[5]),
            _mm256_unpackhi_epi16(r[4], r[5]),
            _mm256_unpacklo_epi16(r[6], r[7]),
            _mm256_unpackhi_epi16(r[6], r[7]),
        ];
        let b = [
            _mm256_unpacklo_epi32(a[0], a[2]),
            _mm256_unpackhi_epi32(a[0], a[2]),
            _mm256_unpacklo_epi32(a[1], a[3]),
            _mm256_unpackhi_epi32(a[1], a[3]),
            _mm256_unpacklo_epi32(a[4], a[6]),
            _mm256_unpackhi_epi32(a[4], a[6]),
            _mm256_unpacklo_epi32(a[5], a[7]),
            _mm256_unpackhi_epi32(a[5], a[7]),
        ];
        [
            _mm256_unpacklo_epi64(b[0], b[4]),
            _mm256_unpackhi_epi64(b[0], b[4]),
            _mm256_unpacklo_epi64(b[1], b[5]),
            _mm256_unpackhi_epi64(b[1], b[5]),
            _mm256_unpacklo_epi64(b[2], b[6]),
            _mm256_unpackhi_epi64(b[2], b[6]),
            _mm256_unpacklo_epi64(b[3], b[7]),
            _mm256_unpackhi_epi64(b[3], b[7]),
        ]
    }

    /// The 1-D $8$-point transform of `forward_dct8x8` across the registers, per 16-bit lane.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn dct8_1d(x: [__m256i; 8]) -> [__m256i; 8] {
        let (add, sub) = (_mm256_add_epi16, _mm256_sub_epi16);
        let half = |v| _mm256_srai_epi16::<1>(v);
        let quarter = |v| _mm256_srai_epi16::<2>(v);
        let a0 = add(x[0], x[7]);
        let a1 = add(x[1], x[6]);
        let a2 = add(x[2], x[5]);
        let a3 = add(x[3], x[4]);
        let a4 = sub(x[0], x[7]);
        let a5 = sub(x[1], x[6]);
        let a6 = sub(x[2], x[5]);
        let a7 = sub(x[3], x[4]);
        let b0 = add(a0, a3);
        let b1 = add(a1, a2);
        let b2 = sub(a0, a3);
        let b3 = sub(a1, a2);
        let b4 = add(add(a5, a6), add(half(a4), a4));
        let b5 = sub(sub(a4, a7), add(half(a6), a6));
        let b6 = sub(add(a4, a7), add(half(a5), a5));
        let b7 = add(sub(a5, a6), add(half(a7), a7));
        [
            add(b0, b1),
            add(b4, quarter(b7)),
            add(b2, half(b3)),
            add(b5, quarter(b6)),
            sub(b0, b1),
            sub(b6, quarter(b5)),
            sub(half(b2), b3),
            sub(quarter(b4), b7),
        ]
    }

    /// [`super::sub_dct8x8_x2`]: the left block in the low lanes, the right one in the high.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn sub_dct8_x2(
        src: *const u8,
        ss: usize,
        pred: *const u8,
        ps: usize,
        out: &mut [[i16; 64]; 2],
    ) {
        unsafe {
            let rows: [__m256i; 8] = std::array::from_fn(|y| {
                let s = _mm_loadu_si128(src.add(y * ss) as *const __m128i);
                let p = _mm_loadu_si128(pred.add(y * ps) as *const __m128i);
                _mm256_sub_epi16(_mm256_cvtepu8_epi16(s), _mm256_cvtepu8_epi16(p))
            });
            // Rows first, as the reference: the transposed block's registers are its columns,
            // so the transform across them is the row transform.
            let t = dct8_1d(transpose(rows));
            let o = dct8_1d(transpose(t));
            for (j, v) in o.iter().enumerate() {
                _mm_storeu_si128(
                    out[0].as_mut_ptr().add(8 * j) as *mut __m128i,
                    _mm256_castsi256_si128(*v),
                );
                _mm_storeu_si128(
                    out[1].as_mut_ptr().add(8 * j) as *mut __m128i,
                    _mm256_extracti128_si256::<1>(*v),
                );
            }
        }
    }

    /// [`super::quant_8x8`].
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn quant8(c: &mut [i16; 64], mf: &[u16; 64], bias: u32, shift: u32) -> u32 {
        unsafe {
            let bias = _mm256_set1_epi32(bias as i32);
            let shift = _mm_cvtsi32_si128(shift as i32);
            let one = _mm256_set1_epi16(1);
            let (mut any, mut big) = (_mm256_setzero_si256(), _mm256_setzero_si256());
            for k in 0..4 {
                let p = c.as_mut_ptr().add(16 * k) as *mut __m256i;
                let v = _mm256_loadu_si256(p);
                let m = _mm256_loadu_si256(mf.as_ptr().add(16 * k) as *const __m256i);
                // |c| as unsigned 16-bit (|-32768| wraps to 32768), products widened to 32 bits.
                let a = _mm256_abs_epi16(v);
                let (lo, hi) = (_mm256_mullo_epi16(a, m), _mm256_mulhi_epu16(a, m));
                let p0 =
                    _mm256_srl_epi32(_mm256_add_epi32(_mm256_unpacklo_epi16(lo, hi), bias), shift);
                let p1 =
                    _mm256_srl_epi32(_mm256_add_epi32(_mm256_unpackhi_epi16(lo, hi), bias), shift);
                // packs saturates at 32767; the products reach at most 2^31, below its sign.
                let mag = _mm256_packs_epi32(p0, p1);
                _mm256_storeu_si256(p, _mm256_sign_epi16(mag, v));
                any = _mm256_or_si256(any, mag);
                big = _mm256_or_si256(big, _mm256_cmpgt_epi16(mag, one));
            }
            ((_mm256_testz_si256(any, any) ^ 1) as u32)
                | (((_mm256_testz_si256(big, big) ^ 1) as u32) << 1)
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// Transposes the $8\times8$ block of rows `r`.
    #[inline]
    #[target_feature(enable = "neon")]
    fn transpose(r: [int16x8_t; 8]) -> [int16x8_t; 8] {
        let t: [int16x8_t; 8] = std::array::from_fn(|i| {
            let (a, b) = (r[i & !1], r[i | 1]);
            if i & 1 == 0 {
                vtrn1q_s16(a, b)
            } else {
                vtrn2q_s16(a, b)
            }
        });
        let w = |v: int16x8_t| vreinterpretq_s32_s16(v);
        // u[0] = cols 0,4 of rows 0-3; u[1] = cols 1,5; u[2] = cols 2,6; u[3] = cols 3,7;
        // u[4..] the same of rows 4-7.
        let u = [
            vtrn1q_s32(w(t[0]), w(t[2])),
            vtrn1q_s32(w(t[1]), w(t[3])),
            vtrn2q_s32(w(t[0]), w(t[2])),
            vtrn2q_s32(w(t[1]), w(t[3])),
            vtrn1q_s32(w(t[4]), w(t[6])),
            vtrn1q_s32(w(t[5]), w(t[7])),
            vtrn2q_s32(w(t[4]), w(t[6])),
            vtrn2q_s32(w(t[5]), w(t[7])),
        ];
        let d = |v: int32x4_t| vreinterpretq_s64_s32(v);
        let h = |v: int64x2_t| vreinterpretq_s16_s64(v);
        std::array::from_fn(|c| {
            let (a, b) = (d(u[c & 3]), d(u[4 + (c & 3)]));
            h(if c < 4 {
                vtrn1q_s64(a, b)
            } else {
                vtrn2q_s64(a, b)
            })
        })
    }

    /// The 1-D $8$-point transform of `forward_dct8x8` across the registers, per lane.
    #[inline]
    #[target_feature(enable = "neon")]
    fn dct8_1d(x: [int16x8_t; 8]) -> [int16x8_t; 8] {
        let (add, sub) = (vaddq_s16, vsubq_s16);
        let half = |v| vshrq_n_s16::<1>(v);
        let quarter = |v| vshrq_n_s16::<2>(v);
        let a0 = add(x[0], x[7]);
        let a1 = add(x[1], x[6]);
        let a2 = add(x[2], x[5]);
        let a3 = add(x[3], x[4]);
        let a4 = sub(x[0], x[7]);
        let a5 = sub(x[1], x[6]);
        let a6 = sub(x[2], x[5]);
        let a7 = sub(x[3], x[4]);
        let b0 = add(a0, a3);
        let b1 = add(a1, a2);
        let b2 = sub(a0, a3);
        let b3 = sub(a1, a2);
        let b4 = add(add(a5, a6), add(half(a4), a4));
        let b5 = sub(sub(a4, a7), add(half(a6), a6));
        let b6 = sub(add(a4, a7), add(half(a5), a5));
        let b7 = add(sub(a5, a6), add(half(a7), a7));
        [
            add(b0, b1),
            add(b4, quarter(b7)),
            add(b2, half(b3)),
            add(b5, quarter(b6)),
            sub(b0, b1),
            sub(b6, quarter(b5)),
            sub(half(b2), b3),
            sub(quarter(b4), b7),
        ]
    }

    /// One block of [`super::sub_dct8x8_x2`].
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn sub_dct8(
        src: *const u8,
        ss: usize,
        pred: *const u8,
        ps: usize,
    ) -> [i16; 64] {
        unsafe {
            let rows: [int16x8_t; 8] = std::array::from_fn(|y| {
                let (s, p) = (vld1_u8(src.add(y * ss)), vld1_u8(pred.add(y * ps)));
                vreinterpretq_s16_u16(vsubl_u8(s, p))
            });
            let o = dct8_1d(transpose(dct8_1d(transpose(rows))));
            let mut out = [0i16; 64];
            for (j, v) in o.iter().enumerate() {
                vst1q_s16(out.as_mut_ptr().add(8 * j), *v);
            }
            out
        }
    }

    /// [`super::quant_8x8`].
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn quant8(c: &mut [i16; 64], mf: &[u16; 64], bias: u32, shift: u32) -> u32 {
        unsafe {
            let bias = vdupq_n_u32(bias);
            let shift = vdupq_n_s32(-(shift as i32));
            let cap = vdupq_n_u32(i16::MAX as u32);
            let mut max = vdupq_n_u16(0);
            for k in 0..8 {
                let v = vld1q_s16(c.as_ptr().add(8 * k));
                let m = vld1q_u16(mf.as_ptr().add(8 * k));
                let a = vreinterpretq_u16_s16(vabsq_s16(v));
                let lo = vshlq_u32(
                    vaddq_u32(vmull_u16(vget_low_u16(a), vget_low_u16(m)), bias),
                    shift,
                );
                let hi = vshlq_u32(vaddq_u32(vmull_high_u16(a, m), bias), shift);
                let l = vreinterpretq_s16_u16(vcombine_u16(
                    vmovn_u32(vminq_u32(lo, cap)),
                    vmovn_u32(vminq_u32(hi, cap)),
                ));
                vst1q_s16(
                    c.as_mut_ptr().add(8 * k),
                    vbslq_s16(vcltzq_s16(v), vnegq_s16(l), l),
                );
                max = vmaxq_u16(max, vreinterpretq_u16_s16(l));
            }
            match vmaxvq_u16(max) {
                0 => 0,
                1 => 1,
                _ => 3,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic noise with runs of extremes, so residuals reach both ends of their range.
    fn plane(len: usize, seed: u32) -> Vec<u8> {
        let mut x = seed;
        (0..len)
            .map(|i| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                match (i / 5) % 6 {
                    0 => 0,
                    1 => 255,
                    _ => (x >> 24) as u8,
                }
            })
            .collect()
    }

    #[test]
    fn dct8_matches_scalar() {
        let (w, h) = (37usize, 29usize);
        let flat = [vec![0u8; w * h], vec![255u8; w * h]];
        for seed in 0..40u32 {
            let (s, p) = (plane(w * h, seed), plane(w * h, seed + 100));
            let cases: [(&[u8], &[u8]); 3] = [(&s, &p), (&flat[1], &flat[0]), (&flat[0], &flat[1])];
            for (s, p) in cases {
                let off = (seed as usize * 7) % (w - 16) + (seed as usize % (h - 8)) * w;
                let (mut a, mut b) = ([[0i16; 64]; 2], [[0i16; 64]; 2]);
                sub_dct8x8_x2(&s[off..], w, &p[off..], w, &mut a);
                sub_dct8x8_x2_scalar(&s[off..], w, &p[off..], w, &mut b);
                assert_eq!(a, b, "seed {seed}");
            }
        }
    }

    #[test]
    fn quant8_matches_scalar() {
        let mut x = 0x1234_5678u32;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        for round in 0..200 {
            let mut c: [i16; 64] = std::array::from_fn(|_| match next() % 9 {
                0 => i16::MIN,
                1 => i16::MAX,
                2 => 0,
                _ => (next() >> 16) as i16 >> (next() % 12),
            });
            let mf: [u16; 64] = std::array::from_fn(|_| (next() % 32768) as u16);
            let shift = 1 + next() % 31;
            let bias = next() % (1 << shift);
            let mut d = c;
            let (na, nb) = (
                quant_8x8(&mut c, &mf, bias, shift),
                quant_8x8_scalar(&mut d, &mf, bias, shift),
            );
            assert_eq!((c, na), (d, nb), "round {round}");
        }
    }
}
