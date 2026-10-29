//! H.264 in-loop deblocking of whole 16-line luma edges (ITU-T H.264 8.7.2, `SIMD-0076`).
//!
//! [`h264_luma_deblock`] filters the four 4-line segments of one macroblock edge at once, given
//! the per-segment `tC0` (`-1` for a segment with bS 0), or with the bS 4 filter. A vertical
//! edge (samples along a row) is transposed into registers, filtered, and transposed back.

#[inline(always)]
fn clip1(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Filters one 16-line luma edge in place. Line `k` has `q0` at `q0 + k * ystride`, and its
/// samples `p3..p0, q0..q3` step by `xstride` across the edge. `tc0[s]` applies to lines
/// `4s..4s + 4`; `-1` leaves a segment unfiltered. `strong` selects the bS 4 filter (8.7.2.4)
/// for the whole edge, ignoring `tc0`.
///
/// # Panics
/// If `buf` does not hold all 16 lines of samples `p3..q3`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn h264_luma_deblock(
    buf: &mut [u8],
    q0: usize,
    xstride: usize,
    ystride: usize,
    alpha: i32,
    beta: i32,
    tc0: [i8; 4],
    strong: bool,
) {
    match (xstride, ystride, strong) {
        (1, s, false) => h264_luma_deblock_edge::<false, false>(buf, q0, s, alpha, beta, tc0),
        (1, s, true) => h264_luma_deblock_edge::<false, true>(buf, q0, s, alpha, beta, tc0),
        (s, 1, false) => h264_luma_deblock_edge::<true, false>(buf, q0, s, alpha, beta, tc0),
        (s, 1, true) => h264_luma_deblock_edge::<true, true>(buf, q0, s, alpha, beta, tc0),
        _ => {
            assert!(q0 >= 4 * xstride && q0 + 3 * xstride + 15 * ystride < buf.len());
            h264_luma_deblock_scalar(buf, q0, xstride, ystride, alpha, beta, tc0, strong);
        }
    }
}

/// [`h264_luma_deblock`] of a vertical edge (`HORIZONTAL == false`: samples across it step by 1,
/// lines by `stride`) or a horizontal one (the other way round), with the bS 4 filter when
/// `STRONG`. The edge kind and filter are known at compile time.
///
/// # Panics
/// If `buf` does not hold all 16 lines of samples `p3..q3`.
#[inline]
pub fn h264_luma_deblock_edge<const HORIZONTAL: bool, const STRONG: bool>(
    buf: &mut [u8],
    q0: usize,
    stride: usize,
    alpha: i32,
    beta: i32,
    tc0: [i8; 4],
) {
    let (xstride, ystride) = if HORIZONTAL { (stride, 1) } else { (1, stride) };
    assert!(q0 >= 4 * xstride && q0 + 3 * xstride + 15 * ystride < buf.len());
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        unsafe { luma_avx2::<HORIZONTAL, STRONG>(buf, q0, xstride, ystride, alpha, beta, tc0) };
        return;
    }
    h264_luma_deblock_scalar(buf, q0, xstride, ystride, alpha, beta, tc0, STRONG);
}

/// Scalar reference for [`h264_luma_deblock`].
#[allow(clippy::too_many_arguments)]
pub fn h264_luma_deblock_scalar(
    buf: &mut [u8],
    q0: usize,
    xstride: usize,
    ystride: usize,
    alpha: i32,
    beta: i32,
    tc0: [i8; 4],
    strong: bool,
) {
    for k in 0..16 {
        let tc0 = tc0[k / 4] as i32;
        if !strong && tc0 < 0 {
            continue;
        }
        let o = q0 + k * ystride;
        let at = |i: isize| (o as isize + i * xstride as isize) as usize;
        let (p0, q0v) = (buf[at(-1)] as i32, buf[at(0)] as i32);
        let (p1, q1) = (buf[at(-2)] as i32, buf[at(1)] as i32);
        if (p0 - q0v).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0v).abs() >= beta {
            continue;
        }
        let (p2, q2) = (buf[at(-3)] as i32, buf[at(2)] as i32);
        let ap = (p2 - p0).abs() < beta;
        let aq = (q2 - q0v).abs() < beta;
        if !strong {
            let tc = tc0 + ap as i32 + aq as i32;
            let d = ((((q0v - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
            buf[at(-1)] = clip1(p0 + d);
            buf[at(0)] = clip1(q0v - d);
            if ap {
                buf[at(-2)] =
                    (p1 + ((p2 + ((p0 + q0v + 1) >> 1) - (p1 << 1)) >> 1).clamp(-tc0, tc0)) as u8;
            }
            if aq {
                buf[at(1)] =
                    (q1 + ((q2 + ((p0 + q0v + 1) >> 1) - (q1 << 1)) >> 1).clamp(-tc0, tc0)) as u8;
            }
        } else {
            let (p3, q3) = (buf[at(-4)] as i32, buf[at(3)] as i32);
            let small = (p0 - q0v).abs() < ((alpha >> 2) + 2);
            if ap && small {
                buf[at(-1)] = ((p2 + 2 * p1 + 2 * p0 + 2 * q0v + q1 + 4) >> 3) as u8;
                buf[at(-2)] = ((p2 + p1 + p0 + q0v + 2) >> 2) as u8;
                buf[at(-3)] = ((2 * p3 + 3 * p2 + p1 + p0 + q0v + 4) >> 3) as u8;
            } else {
                buf[at(-1)] = ((2 * p1 + p0 + q1 + 2) >> 2) as u8;
            }
            if aq && small {
                buf[at(0)] = ((p1 + 2 * p0 + 2 * q0v + 2 * q1 + q2 + 4) >> 3) as u8;
                buf[at(1)] = ((p0 + q0v + q1 + q2 + 2) >> 2) as u8;
                buf[at(2)] = ((2 * q3 + 3 * q2 + q1 + q0v + p0 + 4) >> 3) as u8;
            } else {
                buf[at(0)] = ((2 * q1 + q0v + p1 + 2) >> 2) as u8;
            }
        }
    }
}

/// Filters one 8-line 4:2:0 chroma edge in place (8.7.2.3 and 8.7.2.4 with `chromaStyleFilteringFlag`).
/// Line `k` has `q0` at `q0 + k * ystride` and samples `p1, p0, q0, q1` stepping by `xstride`;
/// `tc0[s]` applies to lines `2s, 2s + 1` (`-1` leaves them unfiltered); `strong` selects bS 4.
///
/// # Panics
/// If `buf` does not hold all 8 lines of samples `p1..q1`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn h264_chroma_deblock(
    buf: &mut [u8],
    q0: usize,
    xstride: usize,
    ystride: usize,
    alpha: i32,
    beta: i32,
    tc0: [i8; 4],
    strong: bool,
) {
    match (xstride, ystride, strong) {
        (1, s, false) => h264_chroma_deblock_edge::<false, false>(buf, q0, s, alpha, beta, tc0),
        (1, s, true) => h264_chroma_deblock_edge::<false, true>(buf, q0, s, alpha, beta, tc0),
        (s, 1, false) => h264_chroma_deblock_edge::<true, false>(buf, q0, s, alpha, beta, tc0),
        (s, 1, true) => h264_chroma_deblock_edge::<true, true>(buf, q0, s, alpha, beta, tc0),
        _ => {
            assert!(q0 >= 2 * xstride && q0 + xstride + 7 * ystride < buf.len());
            h264_chroma_deblock_scalar(buf, q0, xstride, ystride, alpha, beta, tc0, strong);
        }
    }
}

/// [`h264_chroma_deblock`] with the edge kind and filter known at compile time, as for
/// [`h264_luma_deblock_edge`].
///
/// # Panics
/// If `buf` does not hold all 8 lines of samples `p1..q1`.
#[inline]
pub fn h264_chroma_deblock_edge<const HORIZONTAL: bool, const STRONG: bool>(
    buf: &mut [u8],
    q0: usize,
    stride: usize,
    alpha: i32,
    beta: i32,
    tc0: [i8; 4],
) {
    let (xstride, ystride) = if HORIZONTAL { (stride, 1) } else { (1, stride) };
    assert!(q0 >= 2 * xstride && q0 + xstride + 7 * ystride < buf.len());
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        unsafe { chroma_avx2::<HORIZONTAL, STRONG>(buf, q0, xstride, ystride, alpha, beta, tc0) };
        return;
    }
    h264_chroma_deblock_scalar(buf, q0, xstride, ystride, alpha, beta, tc0, STRONG);
}

/// Scalar reference for [`h264_chroma_deblock`].
#[allow(clippy::too_many_arguments)]
pub fn h264_chroma_deblock_scalar(
    buf: &mut [u8],
    q0: usize,
    xstride: usize,
    ystride: usize,
    alpha: i32,
    beta: i32,
    tc0: [i8; 4],
    strong: bool,
) {
    for k in 0..8 {
        let tc0 = tc0[k / 2] as i32;
        if !strong && tc0 < 0 {
            continue;
        }
        let o = q0 + k * ystride;
        let (p1, p0) = (buf[o - 2 * xstride] as i32, buf[o - xstride] as i32);
        let (q0v, q1) = (buf[o] as i32, buf[o + xstride] as i32);
        if (p0 - q0v).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0v).abs() >= beta {
            continue;
        }
        if strong {
            buf[o - xstride] = ((2 * p1 + p0 + q1 + 2) >> 2) as u8;
            buf[o] = ((2 * q1 + q0v + p1 + 2) >> 2) as u8;
        } else {
            let tc = tc0 + 1;
            let d = ((((q0v - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
            buf[o - xstride] = clip1(p0 + d);
            buf[o] = clip1(q0v - d);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn chroma_avx2<const HORIZONTAL: bool, const STRONG: bool>(
    buf: &mut [u8],
    q0: usize,
    xstride: usize,
    ystride: usize,
    alpha: i32,
    beta: i32,
    tc0: [i8; 4],
) {
    use std::arch::x86_64::*;
    let base = unsafe { buf.as_mut_ptr().add(q0) };
    // p1, p0, q0, q1 of the 8 lines, 8 bytes each in the low halves.
    let (p1, p0, q0v, q1);
    if HORIZONTAL {
        let ld = |i: isize| unsafe {
            _mm_loadl_epi64(base.offset(i * xstride as isize) as *const __m128i)
        };
        (p1, p0, q0v, q1) = (ld(-2), ld(-1), ld(0), ld(1));
    } else {
        let row = |k: usize| unsafe {
            _mm_cvtsi32_si128((base.add(k * ystride).sub(2) as *const i32).read_unaligned())
        };
        let a: [__m128i; 4] =
            std::array::from_fn(|i| _mm_unpacklo_epi8(row(2 * i), row(2 * i + 1)));
        let b0 = _mm_unpacklo_epi16(a[0], a[1]);
        let b1 = _mm_unpacklo_epi16(a[2], a[3]);
        let c01 = _mm_unpacklo_epi32(b0, b1);
        let c23 = _mm_unpackhi_epi32(b0, b1);
        (p1, p0) = (c01, _mm_unpackhi_epi64(c01, c01));
        (q0v, q1) = (c23, _mm_unpackhi_epi64(c23, c23));
    }
    let zero = _mm_setzero_si128();
    let w = |v: __m128i| _mm_unpacklo_epi8(v, zero);
    let (wp1, wp0, wq0, wq1) = (w(p1), w(p0), w(q0v), w(q1));
    let absd = |a: __m128i, b: __m128i| _mm_abs_epi16(_mm_sub_epi16(a, b));
    let lt = |a: __m128i, b: __m128i| _mm_cmpgt_epi16(b, a);
    let vb = _mm_set1_epi16(beta as i16);
    let mut mask = _mm_and_si128(
        lt(absd(wp0, wq0), _mm_set1_epi16(alpha as i16)),
        _mm_and_si128(lt(absd(wp1, wp0), vb), lt(absd(wq1, wq0), vb)),
    );
    let two = _mm_set1_epi16(2);
    let (np0, nq0) = if STRONG {
        (
            _mm_srai_epi16::<2>(_mm_add_epi16(
                _mm_add_epi16(_mm_slli_epi16::<1>(wp1), wp0),
                _mm_add_epi16(wq1, two),
            )),
            _mm_srai_epi16::<2>(_mm_add_epi16(
                _mm_add_epi16(_mm_slli_epi16::<1>(wq1), wq0),
                _mm_add_epi16(wp1, two),
            )),
        )
    } else {
        let t: [i16; 8] = std::array::from_fn(|k| tc0[k / 2] as i16);
        let vt0 = unsafe { _mm_loadu_si128(t.as_ptr() as *const __m128i) };
        mask = _mm_and_si128(mask, _mm_cmpgt_epi16(vt0, _mm_set1_epi16(-1)));
        let tc = _mm_add_epi16(vt0, _mm_set1_epi16(1));
        let d = _mm_add_epi16(
            _mm_slli_epi16::<2>(_mm_sub_epi16(wq0, wp0)),
            _mm_add_epi16(_mm_sub_epi16(wp1, wq1), _mm_set1_epi16(4)),
        );
        let d = _mm_min_epi16(
            _mm_max_epi16(_mm_srai_epi16::<3>(d), _mm_sub_epi16(zero, tc)),
            tc,
        );
        (_mm_add_epi16(wp0, d), _mm_sub_epi16(wq0, d))
    };
    // packus clips to 0..=255 (the strong results are already in range).
    let np0 = _mm_packus_epi16(_mm_blendv_epi8(wp0, np0, mask), zero);
    let nq0 = _mm_packus_epi16(_mm_blendv_epi8(wq0, nq0, mask), zero);
    if HORIZONTAL {
        unsafe {
            _mm_storel_epi64(base.sub(xstride) as *mut __m128i, np0);
            _mm_storel_epi64(base as *mut __m128i, nq0);
        }
    } else {
        let x = _mm_unpacklo_epi8(p1, np0);
        let y = _mm_unpacklo_epi8(nq0, q1);
        let rows = [_mm_unpacklo_epi16(x, y), _mm_unpackhi_epi16(x, y)];
        for k in 0..8 {
            let v = rows[k / 4];
            let v = match k % 4 {
                0 => _mm_cvtsi128_si32(v),
                1 => _mm_cvtsi128_si32(_mm_srli_si128::<4>(v)),
                2 => _mm_cvtsi128_si32(_mm_srli_si128::<8>(v)),
                _ => _mm_cvtsi128_si32(_mm_srli_si128::<12>(v)),
            };
            unsafe { (base.add(k * ystride).sub(2) as *mut i32).write_unaligned(v) };
        }
    }
}

/// Transposes 16 rows of 8 bytes (`rows[k]` in the low half) into 8 columns of 16 bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transpose_16x8(r: &[std::arch::x86_64::__m128i; 16]) -> [std::arch::x86_64::__m128i; 8] {
    use std::arch::x86_64::*;
    let a: [__m128i; 8] = std::array::from_fn(|i| _mm_unpacklo_epi8(r[2 * i], r[2 * i + 1]));
    // b[2g]: columns 0..3, b[2g + 1]: columns 4..7, of rows 4g..4g + 4.
    let b: [__m128i; 8] = std::array::from_fn(|i| {
        let g = i / 2;
        if i % 2 == 0 {
            _mm_unpacklo_epi16(a[2 * g], a[2 * g + 1])
        } else {
            _mm_unpackhi_epi16(a[2 * g], a[2 * g + 1])
        }
    });
    // Rows 0..7 then 8..15: columns (0,1), (2,3), (4,5), (6,7).
    let half = |lo: usize| -> [__m128i; 4] {
        [
            _mm_unpacklo_epi32(b[lo], b[lo + 2]),
            _mm_unpackhi_epi32(b[lo], b[lo + 2]),
            _mm_unpacklo_epi32(b[lo + 1], b[lo + 3]),
            _mm_unpackhi_epi32(b[lo + 1], b[lo + 3]),
        ]
    };
    let (c, d) = (half(0), half(4));
    [
        _mm_unpacklo_epi64(c[0], d[0]),
        _mm_unpackhi_epi64(c[0], d[0]),
        _mm_unpacklo_epi64(c[1], d[1]),
        _mm_unpackhi_epi64(c[1], d[1]),
        _mm_unpacklo_epi64(c[2], d[2]),
        _mm_unpackhi_epi64(c[2], d[2]),
        _mm_unpacklo_epi64(c[3], d[3]),
        _mm_unpackhi_epi64(c[3], d[3]),
    ]
}

/// Inverse of [`transpose_16x8`]: 8 columns of 16 bytes back to 16 rows of 8 bytes (low halves).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transpose_8x16(c: &[std::arch::x86_64::__m128i; 8]) -> [std::arch::x86_64::__m128i; 16] {
    use std::arch::x86_64::*;
    // e[2j]: rows 0..7, e[2j + 1]: rows 8..15, of columns (2j, 2j + 1).
    let e: [__m128i; 8] = std::array::from_fn(|i| {
        let j = i / 2;
        if i % 2 == 0 {
            _mm_unpacklo_epi8(c[2 * j], c[2 * j + 1])
        } else {
            _mm_unpackhi_epi8(c[2 * j], c[2 * j + 1])
        }
    });
    // For rows 0..7 (h = 0) and 8..15 (h = 1): f = [rows 0..3 cols 0..3, rows 4..7 cols 0..3,
    // rows 0..3 cols 4..7, rows 4..7 cols 4..7].
    let quad = |h: usize| -> [__m128i; 4] {
        [
            _mm_unpacklo_epi16(e[h], e[2 + h]),
            _mm_unpackhi_epi16(e[h], e[2 + h]),
            _mm_unpacklo_epi16(e[4 + h], e[6 + h]),
            _mm_unpackhi_epi16(e[4 + h], e[6 + h]),
        ]
    };
    let mut out = [_mm_setzero_si128(); 16];
    for h in 0..2 {
        let f = quad(h);
        let g = [
            _mm_unpacklo_epi32(f[0], f[2]),
            _mm_unpackhi_epi32(f[0], f[2]),
            _mm_unpacklo_epi32(f[1], f[3]),
            _mm_unpackhi_epi32(f[1], f[3]),
        ];
        for (k, v) in g.iter().enumerate() {
            out[8 * h + 2 * k] = *v;
            out[8 * h + 2 * k + 1] = _mm_unpackhi_epi64(*v, *v);
        }
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn luma_avx2<const HORIZONTAL: bool, const STRONG: bool>(
    buf: &mut [u8],
    q0: usize,
    xstride: usize,
    ystride: usize,
    alpha: i32,
    beta: i32,
    tc0: [i8; 4],
) {
    use std::arch::x86_64::*;
    let base = unsafe { buf.as_mut_ptr().add(q0) };
    // s[i] = samples p3, p2, p1, p0, q0, q1, q2, q3 of the 16 lines, as bytes.
    let mut s = [_mm_setzero_si128(); 8];
    if HORIZONTAL {
        for (i, v) in s.iter_mut().enumerate() {
            *v = unsafe {
                _mm_loadu_si128(base.sub(4 * xstride).add(i * xstride) as *const __m128i)
            };
        }
    } else {
        let rows: [__m128i; 16] = std::array::from_fn(|k| unsafe {
            _mm_loadl_epi64(base.add(k * ystride).sub(4) as *const __m128i)
        });
        s = unsafe { transpose_16x8(&rows) };
    }
    let w = |v: __m128i| _mm256_cvtepu8_epi16(v);
    let (p3, p2, p1, p0) = (w(s[0]), w(s[1]), w(s[2]), w(s[3]));
    let (q0v, q1, q2, q3) = (w(s[4]), w(s[5]), w(s[6]), w(s[7]));
    let absd = |a: __m256i, b: __m256i| _mm256_abs_epi16(_mm256_sub_epi16(a, b));
    let lt = |a: __m256i, b: __m256i| _mm256_cmpgt_epi16(b, a);
    let va = _mm256_set1_epi16(alpha as i16);
    let vb = _mm256_set1_epi16(beta as i16);
    let d_pq = absd(p0, q0v);
    let mut mask = _mm256_and_si256(
        lt(d_pq, va),
        _mm256_and_si256(lt(absd(p1, p0), vb), lt(absd(q1, q0v), vb)),
    );
    let ap = lt(absd(p2, p0), vb);
    let aq = lt(absd(q2, q0v), vb);
    let two = _mm256_set1_epi16(2);
    let four = _mm256_set1_epi16(4);
    let add = _mm256_add_epi16;
    let sel = |m: __m256i, new: __m256i, old: __m256i| _mm256_blendv_epi8(old, new, m);
    let (np2, np1, np0, nq0, nq1, nq2);
    if STRONG {
        let small = lt(d_pq, _mm256_set1_epi16(((alpha >> 2) + 2) as i16));
        let pq = add(p0, q0v);
        // p side
        let sp = _mm256_and_si256(ap, small);
        let f0 = _mm256_srai_epi16::<3>(add(
            add(add(p2, _mm256_slli_epi16::<1>(add(p1, pq))), q1),
            four,
        ));
        let f1 = _mm256_srai_epi16::<2>(add(add(add(p2, p1), pq), two));
        let f2 = _mm256_srai_epi16::<3>(add(
            add(
                add(
                    _mm256_slli_epi16::<1>(p3),
                    add(_mm256_slli_epi16::<1>(p2), p2),
                ),
                add(p1, pq),
            ),
            four,
        ));
        let w0 = _mm256_srai_epi16::<2>(add(add(_mm256_slli_epi16::<1>(p1), add(p0, q1)), two));
        np0 = sel(sp, f0, w0);
        np1 = sel(sp, f1, p1);
        np2 = sel(sp, f2, p2);
        // q side
        let sq = _mm256_and_si256(aq, small);
        let g0 = _mm256_srai_epi16::<3>(add(
            add(add(p1, _mm256_slli_epi16::<1>(add(q1, pq))), q2),
            four,
        ));
        let g1 = _mm256_srai_epi16::<2>(add(add(add(q2, q1), pq), two));
        let g2 = _mm256_srai_epi16::<3>(add(
            add(
                add(
                    _mm256_slli_epi16::<1>(q3),
                    add(_mm256_slli_epi16::<1>(q2), q2),
                ),
                add(q1, pq),
            ),
            four,
        ));
        let v0 = _mm256_srai_epi16::<2>(add(add(_mm256_slli_epi16::<1>(q1), add(q0v, p1)), two));
        nq0 = sel(sq, g0, v0);
        nq1 = sel(sq, g1, q1);
        nq2 = sel(sq, g2, q2);
    } else {
        let t: [i16; 16] = std::array::from_fn(|k| tc0[k / 4] as i16);
        let vt0 = unsafe { _mm256_loadu_si256(t.as_ptr() as *const __m256i) };
        mask = _mm256_and_si256(mask, _mm256_cmpgt_epi16(vt0, _mm256_set1_epi16(-1)));
        // Masks are -1 where true: subtracting them adds 1.
        let tc = _mm256_sub_epi16(_mm256_sub_epi16(vt0, ap), aq);
        let ntc = _mm256_sub_epi16(_mm256_setzero_si256(), tc);
        let d = add(
            _mm256_slli_epi16::<2>(_mm256_sub_epi16(q0v, p0)),
            add(_mm256_sub_epi16(p1, q1), four),
        );
        let d = _mm256_min_epi16(_mm256_max_epi16(_mm256_srai_epi16::<3>(d), ntc), tc);
        let zero = _mm256_setzero_si256();
        let max = _mm256_set1_epi16(255);
        let clip = |v: __m256i| _mm256_min_epi16(_mm256_max_epi16(v, zero), max);
        np0 = clip(add(p0, d));
        nq0 = clip(_mm256_sub_epi16(q0v, d));
        let avg = _mm256_avg_epu16(p0, q0v);
        let nt0 = _mm256_sub_epi16(zero, vt0);
        let c1 = |x2: __m256i, x1: __m256i| {
            let v =
                _mm256_srai_epi16::<1>(_mm256_sub_epi16(add(x2, avg), _mm256_slli_epi16::<1>(x1)));
            add(x1, _mm256_min_epi16(_mm256_max_epi16(v, nt0), vt0))
        };
        np1 = sel(ap, c1(p2, p1), p1);
        nq1 = sel(aq, c1(q2, q1), q1);
        np2 = p2;
        nq2 = q2;
    }
    let out = |new: __m256i, old: __m256i| {
        let v = sel(mask, new, old);
        let p = _mm256_packus_epi16(v, v);
        _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0b1000>(p))
    };
    s[1] = out(np2, p2);
    s[2] = out(np1, p1);
    s[3] = out(np0, p0);
    s[4] = out(nq0, q0v);
    s[5] = out(nq1, q1);
    s[6] = out(nq2, q2);
    if HORIZONTAL {
        // Rows p2..q2 (an iterator adaptor here is not inlined into this target_feature function).
        #[allow(clippy::needless_range_loop)]
        for i in 1..7 {
            unsafe {
                _mm_storeu_si128(base.sub(4 * xstride).add(i * xstride) as *mut __m128i, s[i])
            };
        }
    } else {
        let rows = unsafe { transpose_8x16(&s) };
        for (k, r) in rows.iter().enumerate() {
            unsafe { _mm_storel_epi64(base.add(k * ystride).sub(4) as *mut __m128i, *r) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A smooth ramp with noise: small steps across the edge so the filters engage.
    fn picture(seed: u32) -> Vec<u8> {
        let mut x = seed;
        (0..32 * 32)
            .map(|i| {
                x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let base = 100 + (i % 32) * 2 + i / 32;
                (base + ((x >> 16) % 9) as i32 - 4 + if (x >> 28) == 0 { 60 } else { 0 })
                    .clamp(0, 255) as u8
            })
            .collect()
    }

    #[test]
    fn chroma_deblock_matches_scalar() {
        let cases: &[(i32, i32, [i8; 4], bool)] = &[
            (40, 8, [0, 1, 2, 3], false),
            (80, 13, [-1, 4, -1, 10], false),
            (255, 18, [25, 25, 25, 25], false),
            (60, 10, [0; 4], true),
            (12, 2, [0; 4], true),
        ];
        let mut changed = 0;
        for seed in 0..20 {
            let pic = picture(seed);
            for &(alpha, beta, tc0, strong) in cases {
                for &(q0, xs, ys) in &[(8 * 32 + 8, 1, 32), (8 * 32 + 8, 32, 1)] {
                    let mut a = pic.clone();
                    let mut b = pic.clone();
                    h264_chroma_deblock(&mut a, q0, xs, ys, alpha, beta, tc0, strong);
                    h264_chroma_deblock_scalar(&mut b, q0, xs, ys, alpha, beta, tc0, strong);
                    assert_eq!(
                        a, b,
                        "seed {seed} {alpha} {beta} {tc0:?} {strong} stride {xs}"
                    );
                    changed += (a != pic) as usize;
                }
            }
        }
        assert!(changed > 150, "filters engaged on only {changed} edges");
    }

    #[test]
    fn luma_deblock_matches_scalar() {
        let cases: &[(i32, i32, [i8; 4], bool)] = &[
            (40, 8, [0, 1, 2, 3], false),
            (80, 13, [-1, 4, -1, 10], false),
            (255, 18, [25, 25, 25, 25], false),
            (15, 3, [1, 1, 1, 1], false),
            (60, 10, [0; 4], true),
            (255, 18, [0; 4], true),
            (12, 2, [0; 4], true),
        ];
        let mut changed = 0;
        for seed in 0..20 {
            let pic = picture(seed);
            for &(alpha, beta, tc0, strong) in cases {
                // Vertical edge at column 8 (lines are rows), horizontal edge at row 8.
                for &(q0, xs, ys) in &[(8 * 32 + 8, 1, 32), (8 * 32 + 8, 32, 1)] {
                    let mut a = pic.clone();
                    let mut b = pic.clone();
                    h264_luma_deblock(&mut a, q0, xs, ys, alpha, beta, tc0, strong);
                    h264_luma_deblock_scalar(&mut b, q0, xs, ys, alpha, beta, tc0, strong);
                    assert_eq!(
                        a, b,
                        "seed {seed} {alpha} {beta} {tc0:?} {strong} stride {xs}"
                    );
                    changed += (a != pic) as usize;
                }
            }
        }
        assert!(changed > 200, "filters engaged on only {changed} edges");
    }
}
