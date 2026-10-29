//! H.264 inverse 4x4 and 8x8 transforms and reconstruction (ITU-T H.264 8.5.12.2, 8.5.13.2,
//! `SIMD-0076`).
//!
//! [`h264_idct4_add`] and [`h264_idct8_add`] keep the scalar references' wrapping `i32`
//! arithmetic: transposes around the row and column butterflies, so results match them for every
//! input, not only conforming ones.

#[inline(always)]
fn clip(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Inverse-transforms raster coefficients `d` and adds the result to the 4x4 block of `dst` at
/// stride `stride`, clipping to 8 bits.
///
/// # Panics
/// If `dst` is shorter than `3 * stride + 4`.
#[inline]
pub fn h264_idct4_add(dst: &mut [u8], stride: usize, d: &[i32; 16]) {
    assert!(3 * stride + 4 <= dst.len());
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        unsafe { idct4_add_avx2(dst, stride, d) };
        return;
    }
    h264_idct4_add_scalar(dst, stride, d);
}

/// Scalar reference for [`h264_idct4_add`].
pub fn h264_idct4_add_scalar(dst: &mut [u8], stride: usize, d: &[i32; 16]) {
    let mut f = [0i32; 16];
    for i in 0..4 {
        let r = &d[i * 4..i * 4 + 4];
        let e0 = r[0].wrapping_add(r[2]);
        let e1 = r[0].wrapping_sub(r[2]);
        let e2 = (r[1] >> 1).wrapping_sub(r[3]);
        let e3 = r[1].wrapping_add(r[3] >> 1);
        f[i * 4] = e0.wrapping_add(e3);
        f[i * 4 + 1] = e1.wrapping_add(e2);
        f[i * 4 + 2] = e1.wrapping_sub(e2);
        f[i * 4 + 3] = e0.wrapping_sub(e3);
    }
    for j in 0..4 {
        let (f0, f1, f2, f3) = (f[j], f[4 + j], f[8 + j], f[12 + j]);
        let g0 = f0.wrapping_add(f2);
        let g1 = f0.wrapping_sub(f2);
        let g2 = (f1 >> 1).wrapping_sub(f3);
        let g3 = f1.wrapping_add(f3 >> 1);
        let h = [
            g0.wrapping_add(g3),
            g1.wrapping_add(g2),
            g1.wrapping_sub(g2),
            g0.wrapping_sub(g3),
        ];
        for (i, v) in h.iter().enumerate() {
            let p = &mut dst[i * stride + j];
            *p = clip((*p as i32).wrapping_add(v.wrapping_add(32) >> 6));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn idct4_add_avx2(dst: &mut [u8], stride: usize, d: &[i32; 16]) {
    use std::arch::x86_64::*;
    #[inline(always)]
    unsafe fn transpose(r: [__m128i; 4]) -> [__m128i; 4] {
        unsafe {
            let a = _mm_unpacklo_epi32(r[0], r[1]);
            let b = _mm_unpackhi_epi32(r[0], r[1]);
            let c = _mm_unpacklo_epi32(r[2], r[3]);
            let e = _mm_unpackhi_epi32(r[2], r[3]);
            [
                _mm_unpacklo_epi64(a, c),
                _mm_unpackhi_epi64(a, c),
                _mm_unpacklo_epi64(b, e),
                _mm_unpackhi_epi64(b, e),
            ]
        }
    }
    /// The 1-D transform of 8.5.12.2 on four lanes at once: `x[k]` holds element `k`.
    #[inline(always)]
    unsafe fn butterfly(x: [__m128i; 4]) -> [__m128i; 4] {
        unsafe {
            let e0 = _mm_add_epi32(x[0], x[2]);
            let e1 = _mm_sub_epi32(x[0], x[2]);
            let e2 = _mm_sub_epi32(_mm_srai_epi32::<1>(x[1]), x[3]);
            let e3 = _mm_add_epi32(x[1], _mm_srai_epi32::<1>(x[3]));
            [
                _mm_add_epi32(e0, e3),
                _mm_add_epi32(e1, e2),
                _mm_sub_epi32(e1, e2),
                _mm_sub_epi32(e0, e3),
            ]
        }
    }
    unsafe {
        let p = d.as_ptr() as *const __m128i;
        let rows = [
            _mm_loadu_si128(p),
            _mm_loadu_si128(p.add(1)),
            _mm_loadu_si128(p.add(2)),
            _mm_loadu_si128(p.add(3)),
        ];
        // Rows: transpose so lane i is row i, then the row transform across columns.
        let f = butterfly(transpose(rows));
        // f[j] lane i = f[i][j]; transpose back so lane j is column j, then the column transform.
        let h = butterfly(transpose(f));
        let r32 = _mm_set1_epi32(32);
        let zero = _mm_setzero_si128();
        for (i, v) in h.iter().enumerate() {
            let q = dst.as_mut_ptr().add(i * stride);
            let px = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((q as *const i32).read_unaligned()));
            let s = _mm_add_epi32(px, _mm_srai_epi32::<6>(_mm_add_epi32(*v, r32)));
            let s = _mm_packus_epi16(_mm_packs_epi32(s, zero), zero);
            (q as *mut i32).write_unaligned(_mm_cvtsi128_si32(s));
        }
    }
}

/// Inverse-transforms raster coefficients `d` (8.5.13.2) and adds the result to the 8x8 block of
/// `dst` at stride `stride`, clipping to 8 bits.
///
/// # Panics
/// If `dst` is shorter than `7 * stride + 8`.
#[inline]
pub fn h264_idct8_add(dst: &mut [u8], stride: usize, d: &[i32; 64]) {
    assert!(7 * stride + 8 <= dst.len());
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        unsafe { idct8_add_avx2(dst, stride, d) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above; neon() is true only when the CPU supports NEON.
        unsafe { idct8_add_neon(dst, stride, d) };
        return;
    }
    h264_idct8_add_scalar(dst, stride, d);
}

/// The 1-D transform of 8.5.13.2 in wrapping `i32` arithmetic.
#[inline(always)]
fn idct8_1d(d: [i32; 8]) -> [i32; 8] {
    let a0 = d[0].wrapping_add(d[4]);
    let a1 = d[5]
        .wrapping_sub(d[3])
        .wrapping_sub(d[7])
        .wrapping_sub(d[7] >> 1);
    let a2 = d[0].wrapping_sub(d[4]);
    let a3 = d[1]
        .wrapping_add(d[7])
        .wrapping_sub(d[3])
        .wrapping_sub(d[3] >> 1);
    let a4 = (d[2] >> 1).wrapping_sub(d[6]);
    let a5 = d[7]
        .wrapping_sub(d[1])
        .wrapping_add(d[5])
        .wrapping_add(d[5] >> 1);
    let a6 = d[2].wrapping_add(d[6] >> 1);
    let a7 = d[3]
        .wrapping_add(d[5])
        .wrapping_add(d[1])
        .wrapping_add(d[1] >> 1);
    let b0 = a0.wrapping_add(a6);
    let b1 = a1.wrapping_add(a7 >> 2);
    let b2 = a2.wrapping_add(a4);
    let b3 = a3.wrapping_add(a5 >> 2);
    let b4 = a2.wrapping_sub(a4);
    let b5 = (a3 >> 2).wrapping_sub(a5);
    let b6 = a0.wrapping_sub(a6);
    let b7 = a7.wrapping_sub(a1 >> 2);
    [
        b0.wrapping_add(b7),
        b2.wrapping_add(b5),
        b4.wrapping_add(b3),
        b6.wrapping_add(b1),
        b6.wrapping_sub(b1),
        b4.wrapping_sub(b3),
        b2.wrapping_sub(b5),
        b0.wrapping_sub(b7),
    ]
}

/// Scalar reference for [`h264_idct8_add`].
pub fn h264_idct8_add_scalar(dst: &mut [u8], stride: usize, d: &[i32; 64]) {
    let mut g = [0i32; 64];
    for i in 0..8 {
        let row = std::array::from_fn(|j| d[i * 8 + j]);
        g[i * 8..i * 8 + 8].copy_from_slice(&idct8_1d(row));
    }
    for j in 0..8 {
        let col = idct8_1d(std::array::from_fn(|i| g[i * 8 + j]));
        for (i, v) in col.iter().enumerate() {
            let p = &mut dst[i * stride + j];
            *p = clip((*p as i32).wrapping_add(v.wrapping_add(32) >> 6));
        }
    }
}

/// [`idct8_1d`] across eight vectors of any lane type: `x[k]` holds element `k` of each lane's
/// input, the result element `k` of each lane's output.
macro_rules! idct8_lanes {
    ($x:expr, $add:ident, $sub:ident, $sr1:expr, $sr2:expr) => {{
        let x = $x;
        let a0 = $add(x[0], x[4]);
        let a1 = $sub($sub($sub(x[5], x[3]), x[7]), $sr1(x[7]));
        let a2 = $sub(x[0], x[4]);
        let a3 = $sub($sub($add(x[1], x[7]), x[3]), $sr1(x[3]));
        let a4 = $sub($sr1(x[2]), x[6]);
        let a5 = $add($add($sub(x[7], x[1]), x[5]), $sr1(x[5]));
        let a6 = $add(x[2], $sr1(x[6]));
        let a7 = $add($add($add(x[3], x[5]), x[1]), $sr1(x[1]));
        let b0 = $add(a0, a6);
        let b1 = $add(a1, $sr2(a7));
        let b2 = $add(a2, a4);
        let b3 = $add(a3, $sr2(a5));
        let b4 = $sub(a2, a4);
        let b5 = $sub($sr2(a3), a5);
        let b6 = $sub(a0, a6);
        let b7 = $sub(a7, $sr2(a1));
        [
            $add(b0, b7),
            $add(b2, b5),
            $add(b4, b3),
            $add(b6, b1),
            $sub(b6, b1),
            $sub(b4, b3),
            $sub(b2, b5),
            $sub(b0, b7),
        ]
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn idct8_add_avx2(dst: &mut [u8], stride: usize, d: &[i32; 64]) {
    use std::arch::x86_64::*;
    /// Transposes an 8x8 matrix of `i32`, one row per register.
    #[inline(always)]
    unsafe fn transpose(r: [__m256i; 8]) -> [__m256i; 8] {
        unsafe {
            // Pairs of rows interleaved: columns 0,1 | 4,5 and 2,3 | 6,7.
            let t = [
                _mm256_unpacklo_epi32(r[0], r[1]),
                _mm256_unpackhi_epi32(r[0], r[1]),
                _mm256_unpacklo_epi32(r[2], r[3]),
                _mm256_unpackhi_epi32(r[2], r[3]),
                _mm256_unpacklo_epi32(r[4], r[5]),
                _mm256_unpackhi_epi32(r[4], r[5]),
                _mm256_unpacklo_epi32(r[6], r[7]),
                _mm256_unpackhi_epi32(r[6], r[7]),
            ];
            // u[4h + c] holds rows 4h..4h+4 of columns c | c + 4.
            let u = [
                _mm256_unpacklo_epi64(t[0], t[2]),
                _mm256_unpackhi_epi64(t[0], t[2]),
                _mm256_unpacklo_epi64(t[1], t[3]),
                _mm256_unpackhi_epi64(t[1], t[3]),
                _mm256_unpacklo_epi64(t[4], t[6]),
                _mm256_unpackhi_epi64(t[4], t[6]),
                _mm256_unpacklo_epi64(t[5], t[7]),
                _mm256_unpackhi_epi64(t[5], t[7]),
            ];
            [
                _mm256_permute2x128_si256::<0x20>(u[0], u[4]),
                _mm256_permute2x128_si256::<0x20>(u[1], u[5]),
                _mm256_permute2x128_si256::<0x20>(u[2], u[6]),
                _mm256_permute2x128_si256::<0x20>(u[3], u[7]),
                _mm256_permute2x128_si256::<0x31>(u[0], u[4]),
                _mm256_permute2x128_si256::<0x31>(u[1], u[5]),
                _mm256_permute2x128_si256::<0x31>(u[2], u[6]),
                _mm256_permute2x128_si256::<0x31>(u[3], u[7]),
            ]
        }
    }
    #[inline(always)]
    fn add(a: __m256i, b: __m256i) -> __m256i {
        // Safety: only reached from the AVX2 kernel below.
        unsafe { _mm256_add_epi32(a, b) }
    }
    #[inline(always)]
    fn sub(a: __m256i, b: __m256i) -> __m256i {
        // Safety: as for `add`.
        unsafe { _mm256_sub_epi32(a, b) }
    }
    #[inline(always)]
    fn sr1(a: __m256i) -> __m256i {
        // Safety: as for `add`.
        unsafe { _mm256_srai_epi32::<1>(a) }
    }
    #[inline(always)]
    fn sr2(a: __m256i) -> __m256i {
        // Safety: as for `add`.
        unsafe { _mm256_srai_epi32::<2>(a) }
    }
    unsafe {
        let p = d.as_ptr() as *const __m256i;
        let rows: [__m256i; 8] = std::array::from_fn(|i| _mm256_loadu_si256(p.add(i)));
        // Register k holds column k: the row transforms run across registers.
        let f = idct8_lanes!(transpose(rows), add, sub, sr1, sr2);
        // Back to one row per register; the column transforms run across registers again.
        let h = idct8_lanes!(transpose(f), add, sub, sr1, sr2);
        let r32 = _mm256_set1_epi32(32);
        for k in 0..4 {
            let v0 = _mm256_srai_epi32::<6>(_mm256_add_epi32(h[2 * k], r32));
            let v1 = _mm256_srai_epi32::<6>(_mm256_add_epi32(h[2 * k + 1], r32));
            // Saturating to i16 first keeps clip(p + v): |v| > 32767 clips either way.
            let v = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packs_epi32(v0, v1));
            let q0 = dst.as_mut_ptr().add(2 * k * stride);
            let q1 = q0.add(stride);
            let px = _mm_unpacklo_epi64(
                _mm_loadl_epi64(q0 as *const __m128i),
                _mm_loadl_epi64(q1 as *const __m128i),
            );
            let s = _mm256_adds_epi16(v, _mm256_cvtepu8_epi16(px));
            let s = _mm256_packus_epi16(s, s);
            _mm_storel_epi64(q0 as *mut __m128i, _mm256_castsi256_si128(s));
            _mm_storel_epi64(q1 as *mut __m128i, _mm256_extracti128_si256::<1>(s));
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn idct8_add_neon(dst: &mut [u8], stride: usize, d: &[i32; 64]) {
    use std::arch::aarch64::*;
    /// Transposes the 4x4 block held in `r`.
    #[inline(always)]
    unsafe fn transpose4(r: [int32x4_t; 4]) -> [int32x4_t; 4] {
        unsafe {
            let a = vtrn1q_s32(r[0], r[1]);
            let b = vtrn2q_s32(r[0], r[1]);
            let c = vtrn1q_s32(r[2], r[3]);
            let e = vtrn2q_s32(r[2], r[3]);
            let (a, b) = (vreinterpretq_s64_s32(a), vreinterpretq_s64_s32(b));
            let (c, e) = (vreinterpretq_s64_s32(c), vreinterpretq_s64_s32(e));
            [
                vreinterpretq_s32_s64(vtrn1q_s64(a, c)),
                vreinterpretq_s32_s64(vtrn1q_s64(b, e)),
                vreinterpretq_s32_s64(vtrn2q_s64(a, c)),
                vreinterpretq_s32_s64(vtrn2q_s64(b, e)),
            ]
        }
    }
    /// Transposes an 8x8 matrix of `i32`; `r[i][h]` holds columns `4h..4h+4` of row `i`.
    #[inline(always)]
    unsafe fn transpose(r: [[int32x4_t; 2]; 8]) -> [[int32x4_t; 2]; 8] {
        let mut o = r;
        for bi in 0..2 {
            for bj in 0..2 {
                let t = unsafe { transpose4(std::array::from_fn(|l| r[4 * bi + l][bj])) };
                for (c, v) in t.into_iter().enumerate() {
                    o[4 * bj + c][bi] = v;
                }
            }
        }
        o
    }
    #[inline(always)]
    fn add(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        // Safety: only reached from the NEON kernel below.
        unsafe { vaddq_s32(a, b) }
    }
    #[inline(always)]
    fn sub(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        // Safety: as for `add`.
        unsafe { vsubq_s32(a, b) }
    }
    #[inline(always)]
    fn sr1(a: int32x4_t) -> int32x4_t {
        // Safety: as for `add`.
        unsafe { vshrq_n_s32::<1>(a) }
    }
    #[inline(always)]
    fn sr2(a: int32x4_t) -> int32x4_t {
        // Safety: as for `add`.
        unsafe { vshrq_n_s32::<2>(a) }
    }
    /// The 1-D transform across rows `x[k]`, on both column halves.
    #[inline(always)]
    fn pass(x: [[int32x4_t; 2]; 8]) -> [[int32x4_t; 2]; 8] {
        let lo = idct8_lanes!(
            std::array::from_fn::<_, 8, _>(|k| x[k][0]),
            add,
            sub,
            sr1,
            sr2
        );
        let hi = idct8_lanes!(
            std::array::from_fn::<_, 8, _>(|k| x[k][1]),
            add,
            sub,
            sr1,
            sr2
        );
        std::array::from_fn(|k| [lo[k], hi[k]])
    }
    unsafe {
        let p = d.as_ptr();
        let rows: [[int32x4_t; 2]; 8] =
            std::array::from_fn(|i| [vld1q_s32(p.add(8 * i)), vld1q_s32(p.add(8 * i + 4))]);
        let h = pass(transpose(pass(transpose(rows))));
        let r32 = vdupq_n_s32(32);
        for (i, r) in h.iter().enumerate() {
            let q = dst.as_mut_ptr().add(i * stride);
            // Saturating to i16 first keeps clip(p + v): |v| > 32767 clips either way.
            let v = vcombine_s16(
                vqmovn_s32(vshrq_n_s32::<6>(vaddq_s32(r[0], r32))),
                vqmovn_s32(vshrq_n_s32::<6>(vaddq_s32(r[1], r32))),
            );
            let px = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(q)));
            vst1_u8(q, vqmovun_s16(vqaddq_s16(v, px)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idct4_add_matches_scalar() {
        let mut x = 0x9e37_79b9_u32;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        for round in 0..2000 {
            // Mostly the conforming range, sometimes far beyond it.
            let span = if round % 10 == 0 { 1 << 20 } else { 4096 };
            let d: [i32; 16] = std::array::from_fn(|_| (next() % (2 * span)) as i32 - span as i32);
            let pix: Vec<u8> = (0..4 * 7).map(|_| next() as u8).collect();
            let (mut a, mut b) = (pix.clone(), pix.clone());
            h264_idct4_add(&mut a, 7, &d);
            h264_idct4_add_scalar(&mut b, 7, &d);
            assert_eq!(a, b, "{d:?}");
        }
    }

    #[test]
    fn idct8_add_matches_scalar() {
        let mut x = 0x2545_f491_u32;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        for round in 0..4000 {
            // The conforming range, far beyond it, the i32 extremes, and sparse blocks.
            let d: [i32; 64] = std::array::from_fn(|_| {
                let r = next();
                match round % 8 {
                    0 => [i32::MIN, i32::MAX, 0, -1][(r & 3) as usize],
                    1 => r as i32,
                    2 => (r % (1 << 21)) as i32 - (1 << 20),
                    3 if r % 5 != 0 => 0,
                    _ => (r % 8192) as i32 - 4096,
                }
            });
            for stride in [8, 13, 31] {
                let pix: Vec<u8> = (0..7 * stride + 8)
                    .map(|i| match (i + round) % 9 {
                        0 => 0,
                        1 => 255,
                        _ => next() as u8,
                    })
                    .collect();
                let (mut a, mut b) = (pix.clone(), pix.clone());
                h264_idct8_add(&mut a, stride, &d);
                h264_idct8_add_scalar(&mut b, stride, &d);
                assert_eq!(a, b, "stride {stride}: {d:?}");
            }
        }
    }
}
