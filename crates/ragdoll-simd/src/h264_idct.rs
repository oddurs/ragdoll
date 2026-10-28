//! H.264 inverse 4x4 transform and reconstruction (ITU-T H.264 8.5.12.2, `SIMD-0076`).
//!
//! [`h264_idct4_add`] keeps the scalar reference's `i32` arithmetic: two 4x4 transposes around
//! the row and column butterflies, so results match it for every input, not only conforming ones.

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
}
