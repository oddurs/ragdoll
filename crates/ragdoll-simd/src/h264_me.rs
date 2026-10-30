//! Block costs of H.264 motion search over every partition size from $8\times8$ to
//! $16\times16$ (`SIMD-0089`, `x264` `pixel.c`): SAD and SATD of a source block against a
//! prediction that is either one strided block or the rounded average of two, as quarter-sample
//! positions read from half-sample planes are (`x264` `get_ref`).
//!
//! - [`pixel_sad_wxh`]: SAD of `W`x`H` blocks.
//! - [`pixel_sad_avg_wxh`]: SAD against `(a + b + 1) >> 1` without storing the average.
//! - [`pixel_satd_wxh`]: sum of absolute $4\times4$ Hadamard-transformed differences, halved,
//!   against one block or the average of two.
//! - [`pixel_avg_wxh`]: `(a + b + 1) >> 1` into a strided destination.
//! - [`pixel_sa8d_16x16`]: the transform decision's cost, the $8\times8$ Hadamard SATD of each
//!   $8\times8$ block, `(sum + 2) >> 2`, summed (`x264_pixel_sa8d_8x8`).
//!
//! `W` and `H` are 8 or 16. Every kernel has a `_scalar` reference that the AVX2 and NEON paths
//! match bit for bit.

#[inline(always)]
fn check<const W: usize, const H: usize>(buf: &[u8], stride: usize) {
    assert!((W == 8 || W == 16) && (H == 8 || H == 16));
    assert!((H - 1) * stride + W <= buf.len());
}

/// SAD of the `W`x`H` blocks at the start of `a` (stride `sa`) and `b` (stride `sb`).
///
/// # Panics
/// If `W` or `H` is not 8 or 16, or a buffer is shorter than `H` rows of its stride.
#[inline(always)]
pub fn pixel_sad_wxh<const W: usize, const H: usize>(
    a: &[u8],
    sa: usize,
    b: &[u8],
    sb: usize,
) -> u32 {
    check::<W, H>(a, sa);
    check::<W, H>(b, sb);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above; NEON is baseline on aarch64.
        return unsafe { neon::sad::<W, H, false>(a.as_ptr(), sa, b.as_ptr(), sb, b.as_ptr(), sb) };
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        return unsafe { x86::sad::<W, H, false>(a.as_ptr(), sa, b.as_ptr(), sb, b.as_ptr(), sb) };
    }
    pixel_sad_wxh_scalar::<W, H>(a, sa, b, sb)
}

/// Scalar reference for [`pixel_sad_wxh`].
pub fn pixel_sad_wxh_scalar<const W: usize, const H: usize>(
    a: &[u8],
    sa: usize,
    b: &[u8],
    sb: usize,
) -> u32 {
    let mut sum = 0;
    for y in 0..H {
        for x in 0..W {
            sum += a[y * sa + x].abs_diff(b[y * sb + x]) as u32;
        }
    }
    sum
}

/// SAD of the `W`x`H` block `src` (stride `ss`) against `(a + b + 1) >> 1` of the blocks `a`
/// (stride `sa`) and `b` (stride `sb`).
///
/// # Panics
/// As [`pixel_sad_wxh`].
#[inline(always)]
pub fn pixel_sad_avg_wxh<const W: usize, const H: usize>(
    src: &[u8],
    ss: usize,
    a: &[u8],
    sa: usize,
    b: &[u8],
    sb: usize,
) -> u32 {
    check::<W, H>(src, ss);
    check::<W, H>(a, sa);
    check::<W, H>(b, sb);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above; NEON is baseline on aarch64.
        return unsafe {
            neon::sad::<W, H, true>(src.as_ptr(), ss, a.as_ptr(), sa, b.as_ptr(), sb)
        };
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        return unsafe { x86::sad::<W, H, true>(src.as_ptr(), ss, a.as_ptr(), sa, b.as_ptr(), sb) };
    }
    pixel_sad_avg_wxh_scalar::<W, H>(src, ss, a, sa, b, sb)
}

/// Scalar reference for [`pixel_sad_avg_wxh`].
pub fn pixel_sad_avg_wxh_scalar<const W: usize, const H: usize>(
    src: &[u8],
    ss: usize,
    a: &[u8],
    sa: usize,
    b: &[u8],
    sb: usize,
) -> u32 {
    let mut sum = 0;
    for y in 0..H {
        for x in 0..W {
            let p = ((a[y * sa + x] as u16 + b[y * sb + x] as u16 + 1) >> 1) as u8;
            sum += src[y * ss + x].abs_diff(p) as u32;
        }
    }
    sum
}

/// SATD of the `W`x`H` block `src` (stride `ss`) against the block `a` (stride `sa`), or with
/// `b` against `(a + b + 1) >> 1`: the sum over its $4\times4$ blocks of `|H d H^T|`, halved, as
/// [`crate::pixel_satd_16x16`].
///
/// # Panics
/// As [`pixel_sad_wxh`].
#[inline(always)]
pub fn pixel_satd_wxh<const W: usize, const H: usize>(
    src: &[u8],
    ss: usize,
    a: &[u8],
    sa: usize,
    b: Option<(&[u8], usize)>,
) -> u32 {
    check::<W, H>(src, ss);
    check::<W, H>(a, sa);
    if let Some((b, sb)) = b {
        check::<W, H>(b, sb);
    }
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above; NEON is baseline on aarch64.
        return unsafe {
            match b {
                Some((b, sb)) => {
                    neon::satd::<W, H, true>(src.as_ptr(), ss, a.as_ptr(), sa, b.as_ptr(), sb)
                }
                None => neon::satd::<W, H, false>(src.as_ptr(), ss, a.as_ptr(), sa, a.as_ptr(), sa),
            }
        };
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        return unsafe {
            match b {
                Some((b, sb)) => {
                    x86::satd::<W, H, true>(src.as_ptr(), ss, a.as_ptr(), sa, b.as_ptr(), sb)
                }
                None => x86::satd::<W, H, false>(src.as_ptr(), ss, a.as_ptr(), sa, a.as_ptr(), sa),
            }
        };
    }
    pixel_satd_wxh_scalar::<W, H>(src, ss, a, sa, b)
}

/// Scalar reference for [`pixel_satd_wxh`].
pub fn pixel_satd_wxh_scalar<const W: usize, const H: usize>(
    src: &[u8],
    ss: usize,
    a: &[u8],
    sa: usize,
    b: Option<(&[u8], usize)>,
) -> u32 {
    let pred = |y: usize, x: usize| match b {
        Some((b, sb)) => (a[y * sa + x] as i32 + b[y * sb + x] as i32 + 1) >> 1,
        None => a[y * sa + x] as i32,
    };
    let mut sum = 0u32;
    for by in (0..H).step_by(4) {
        for bx in (0..W).step_by(4) {
            let mut d = [[0i32; 4]; 4];
            for (y, row) in d.iter_mut().enumerate() {
                for (x, v) in row.iter_mut().enumerate() {
                    let (r, c) = (by + y, bx + x);
                    *v = src[r * ss + c] as i32 - pred(r, c);
                }
            }
            for row in d.iter_mut() {
                let (s01, d01) = (row[0] + row[1], row[0] - row[1]);
                let (s23, d23) = (row[2] + row[3], row[2] - row[3]);
                *row = [s01 + s23, s01 - s23, d01 + d23, d01 - d23];
            }
            let [r0, r1, r2, r3] = d;
            for c in 0..4 {
                let (s01, d01) = (r0[c] + r1[c], r0[c] - r1[c]);
                let (s23, d23) = (r2[c] + r3[c], r2[c] - r3[c]);
                sum +=
                    ((s01 + s23).abs() + (s01 - s23).abs() + (d01 + d23).abs() + (d01 - d23).abs())
                        as u32;
            }
        }
    }
    sum >> 1
}

/// SADs of the `W`x`H` block `src` (stride `ss`) against the four blocks at the start of
/// `refs`, all with stride `rs`, loading the source once (`x264_pixel_sad_x4`).
///
/// # Panics
/// As [`pixel_sad_wxh`], for each block.
#[inline(always)]
pub fn pixel_sad_x4_wxh<const W: usize, const H: usize>(
    src: &[u8],
    ss: usize,
    refs: [&[u8]; 4],
    rs: usize,
) -> [u32; 4] {
    check::<W, H>(src, ss);
    for r in refs {
        check::<W, H>(r, rs);
    }
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above; NEON is baseline on aarch64.
        return refs.map(|r| unsafe {
            neon::sad::<W, H, false>(src.as_ptr(), ss, r.as_ptr(), rs, src.as_ptr(), ss)
        });
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        return unsafe { x86::sad_x4::<W, H>(src.as_ptr(), ss, refs.map(|r| r.as_ptr()), rs) };
    }
    pixel_sad_x4_wxh_scalar::<W, H>(src, ss, refs, rs)
}

/// Scalar reference for [`pixel_sad_x4_wxh`].
pub fn pixel_sad_x4_wxh_scalar<const W: usize, const H: usize>(
    src: &[u8],
    ss: usize,
    refs: [&[u8]; 4],
    rs: usize,
) -> [u32; 4] {
    refs.map(|r| pixel_sad_wxh_scalar::<W, H>(src, ss, r, rs))
}

/// `dst = (a + b + 1) >> 1` over a `W`x`H` block, all strided.
///
/// # Panics
/// As [`pixel_sad_wxh`], for `dst` too.
#[inline(always)]
pub fn pixel_avg_wxh<const W: usize, const H: usize>(
    dst: &mut [u8],
    ds: usize,
    a: &[u8],
    sa: usize,
    b: &[u8],
    sb: usize,
) {
    check::<W, H>(dst, ds);
    check::<W, H>(a, sa);
    check::<W, H>(b, sb);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above; NEON is baseline on aarch64.
        unsafe { neon::avg::<W, H>(dst.as_mut_ptr(), ds, a.as_ptr(), sa, b.as_ptr(), sb) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        unsafe { x86::avg::<W, H>(dst.as_mut_ptr(), ds, a.as_ptr(), sa, b.as_ptr(), sb) };
        return;
    }
    pixel_avg_wxh_scalar::<W, H>(dst, ds, a, sa, b, sb);
}

/// Scalar reference for [`pixel_avg_wxh`].
pub fn pixel_avg_wxh_scalar<const W: usize, const H: usize>(
    dst: &mut [u8],
    ds: usize,
    a: &[u8],
    sa: usize,
    b: &[u8],
    sb: usize,
) {
    for y in 0..H {
        for x in 0..W {
            dst[y * ds + x] = ((a[y * sa + x] as u16 + b[y * sb + x] as u16 + 1) >> 1) as u8;
        }
    }
}

/// SA8D of the $16\times16$ blocks `a` (stride `sa`) and `b` (stride `sb`): over each of
/// their $8\times8$ blocks, the sum of absolute $8\times8$ Hadamard-transformed differences,
/// `(sum + 2) >> 2`; the four summed.
///
/// # Panics
/// If a buffer is shorter than 16 rows of its stride.
#[inline(always)]
pub fn pixel_sa8d_16x16(a: &[u8], sa: usize, b: &[u8], sb: usize) -> u32 {
    check::<16, 16>(a, sa);
    check::<16, 16>(b, sb);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above; NEON is baseline on aarch64.
        return unsafe { neon::sa8d_16x16(a.as_ptr(), sa, b.as_ptr(), sb) };
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        return unsafe { x86::sa8d_16x16(a.as_ptr(), sa, b.as_ptr(), sb) };
    }
    pixel_sa8d_16x16_scalar(a, sa, b, sb)
}

/// Scalar reference for [`pixel_sa8d_16x16`].
pub fn pixel_sa8d_16x16_scalar(a: &[u8], sa: usize, b: &[u8], sb: usize) -> u32 {
    // One stage of butterflies at distance 1, 2 and 4 each: an 8-point Hadamard transform up
    // to the order and signs of its outputs, which leave the sum of magnitudes unchanged.
    let had8 = |mut v: [i32; 8]| {
        for d in [1, 2, 4] {
            for i in (0..8).filter(|i| i & d == 0) {
                (v[i], v[i + d]) = (v[i] + v[i + d], v[i] - v[i + d]);
            }
        }
        v
    };
    let mut total = 0;
    for q in 0..4 {
        let (x0, y0) = ((q & 1) * 8, (q >> 1) * 8);
        let mut d = [[0i32; 8]; 8];
        for (y, row) in d.iter_mut().enumerate() {
            for (x, v) in row.iter_mut().enumerate() {
                *v = a[(y0 + y) * sa + x0 + x] as i32 - b[(y0 + y) * sb + x0 + x] as i32;
            }
            *row = had8(*row);
        }
        let sum: u32 = (0..8)
            .map(|x| {
                had8(std::array::from_fn(|y| d[y][x]))
                    .iter()
                    .map(|v| v.unsigned_abs())
                    .sum::<u32>()
            })
            .sum();
        total += (sum + 2) >> 2;
    }
    total
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    /// Two rows of `W` bytes: for `W = 16` row `p` in the low lane and `p + s` in the high
    /// lane; for `W = 8` rows `p .. p + 3s`, eight bytes each.
    #[inline(always)]
    unsafe fn rows<const W: usize>(p: *const u8, s: usize) -> __m256i {
        unsafe {
            if W == 16 {
                let lo = _mm_loadu_si128(p as *const __m128i);
                let hi = _mm_loadu_si128(p.add(s) as *const __m128i);
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi)
            } else {
                let lo = pair8(p, s);
                let hi = pair8(p.add(2 * s), s);
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi)
            }
        }
    }

    /// Eight bytes of row `p` in the low half and of row `p + s` in the high half.
    #[inline(always)]
    unsafe fn pair8(p: *const u8, s: usize) -> __m128i {
        unsafe {
            let lo = _mm_castsi128_pd(_mm_loadl_epi64(p as *const __m128i));
            _mm_castpd_si128(_mm_loadh_pd(lo, p.add(s) as *const f64))
        }
    }

    /// SAD of `W`x`H` blocks `p` and `a` (or the average of `a` and `b` with `AVG`).
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn sad<const W: usize, const H: usize, const AVG: bool>(
        p: *const u8,
        sp: usize,
        a: *const u8,
        sa: usize,
        b: *const u8,
        sb: usize,
    ) -> u32 {
        unsafe {
            // Rows per register.
            let step = if W == 16 { 2 } else { 4 };
            let mut acc = _mm256_setzero_si256();
            for y in (0..H).step_by(step) {
                let mut q = rows::<W>(a.add(y * sa), sa);
                if AVG {
                    q = _mm256_avg_epu8(q, rows::<W>(b.add(y * sb), sb));
                }
                acc = _mm256_add_epi64(acc, _mm256_sad_epu8(rows::<W>(p.add(y * sp), sp), q));
            }
            let s = _mm_add_epi64(
                _mm256_castsi256_si128(acc),
                _mm256_extracti128_si256::<1>(acc),
            );
            (_mm_cvtsi128_si64(s) + _mm_extract_epi64::<1>(s)) as u32
        }
    }

    /// [`sad`] of one source block against four blocks of one plane.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn sad_x4<const W: usize, const H: usize>(
        p: *const u8,
        sp: usize,
        q: [*const u8; 4],
        ps: usize,
    ) -> [u32; 4] {
        unsafe {
            let step = if W == 16 { 2 } else { 4 };
            let mut acc = [_mm256_setzero_si256(); 4];
            for y in (0..H).step_by(step) {
                let v = rows::<W>(p.add(y * sp), sp);
                for k in 0..4 {
                    let r = rows::<W>(q[k].add(y * ps), ps);
                    acc[k] = _mm256_add_epi64(acc[k], _mm256_sad_epu8(v, r));
                }
            }
            // The partial sums fit 32 bits: two blocks' per 64-bit lane, then one 32-bit lane
            // per block.
            let t01 = _mm256_or_si256(acc[0], _mm256_slli_epi64::<32>(acc[1]));
            let t23 = _mm256_or_si256(acc[2], _mm256_slli_epi64::<32>(acc[3]));
            let u = _mm256_add_epi32(
                _mm256_unpacklo_epi64(t01, t23),
                _mm256_unpackhi_epi64(t01, t23),
            );
            let s = _mm_add_epi32(_mm256_castsi256_si128(u), _mm256_extracti128_si256::<1>(u));
            let mut out = [0u32; 4];
            _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, s);
            out
        }
    }

    /// `W` bytes of row `p` widened to 16 bits: for `W = 16` the row's sixteen samples; for
    /// `W = 8` the row's eight in the low lane and those of the row four below in the high one.
    #[inline(always)]
    unsafe fn wide<const W: usize, const AVG: bool>(
        a: *const u8,
        sa: usize,
        b: *const u8,
        sb: usize,
    ) -> __m256i {
        unsafe {
            let load = |p: *const u8, s: usize| {
                if W == 16 {
                    _mm_loadu_si128(p as *const __m128i)
                } else {
                    pair8(p, 4 * s)
                }
            };
            let mut v = load(a, sa);
            if AVG {
                v = _mm_avg_epu8(v, load(b, sb));
            }
            _mm256_cvtepu8_epi16(v)
        }
    }

    /// SATD of `W`x`H` blocks `p` and `a` (or the average of `a` and `b` with `AVG`): four
    /// $4\times4$ blocks per pass, one row of each per register.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn satd<const W: usize, const H: usize, const AVG: bool>(
        p: *const u8,
        sp: usize,
        a: *const u8,
        sa: usize,
        b: *const u8,
        sb: usize,
    ) -> u32 {
        unsafe {
            let ones = _mm256_set1_epi16(1);
            let mut acc = _mm256_setzero_si256();
            // A pass covers 4 rows of 16 columns, or 8 rows of 8 columns.
            let pass_rows = if W == 16 { 4 } else { 8 };
            for y0 in (0..H).step_by(pass_rows) {
                {
                    let row = |k: usize| {
                        let r = y0 + k;
                        _mm256_sub_epi16(
                            wide::<W, false>(p.add(r * sp), sp, p, sp),
                            wide::<W, AVG>(a.add(r * sa), sa, b.add(r * sb), sb),
                        )
                    };
                    let (d0, d1, d2, d3) = (row(0), row(1), row(2), row(3));
                    let (s01, d01) = (_mm256_add_epi16(d0, d1), _mm256_sub_epi16(d0, d1));
                    let (s23, d23) = (_mm256_add_epi16(d2, d3), _mm256_sub_epi16(d2, d3));
                    let v = [
                        _mm256_add_epi16(s01, s23),
                        _mm256_sub_epi16(s01, s23),
                        _mm256_add_epi16(d01, d23),
                        _mm256_sub_epi16(d01, d23),
                    ];
                    let mut sum = _mm256_setzero_si256();
                    for x in v {
                        // Horizontal butterflies over adjacent columns, the last one as
                        // |p + q| + |p - q| = 2 max(|p|, |q|), counted in both lanes of a pair.
                        let t = _mm256_shufflehi_epi16::<0b10_11_00_01>(_mm256_shufflelo_epi16::<
                            0b10_11_00_01,
                        >(
                            x
                        ));
                        let w = _mm256_blend_epi16::<0b1010_1010>(
                            _mm256_add_epi16(x, t),
                            _mm256_sub_epi16(t, x),
                        );
                        let u = _mm256_shufflehi_epi16::<0b01_00_11_10>(_mm256_shufflelo_epi16::<
                            0b01_00_11_10,
                        >(
                            w
                        ));
                        sum = _mm256_add_epi16(
                            sum,
                            _mm256_max_epi16(_mm256_abs_epi16(w), _mm256_abs_epi16(u)),
                        );
                    }
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(sum, ones));
                }
            }
            let s = _mm_add_epi32(
                _mm256_castsi256_si128(acc),
                _mm256_extracti128_si256::<1>(acc),
            );
            let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
            (_mm_cvtsi128_si32(s) as u32) >> 1
        }
    }

    /// `(a + b + 1) >> 1` of `W`x`H` blocks into `d`.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn avg<const W: usize, const H: usize>(
        d: *mut u8,
        sd: usize,
        a: *const u8,
        sa: usize,
        b: *const u8,
        sb: usize,
    ) {
        unsafe {
            for y in 0..H {
                if W == 16 {
                    let v = _mm_avg_epu8(
                        _mm_loadu_si128(a.add(y * sa) as *const __m128i),
                        _mm_loadu_si128(b.add(y * sb) as *const __m128i),
                    );
                    _mm_storeu_si128(d.add(y * sd) as *mut __m128i, v);
                } else {
                    let v = _mm_avg_epu8(
                        _mm_loadl_epi64(a.add(y * sa) as *const __m128i),
                        _mm_loadl_epi64(b.add(y * sb) as *const __m128i),
                    );
                    _mm_storel_epi64(d.add(y * sd) as *mut __m128i, v);
                }
            }
        }
    }

    /// Butterflies of `r` at distance 1, 2 and 4: an 8-point Hadamard transform across the
    /// registers, up to the order and signs of its outputs.
    #[inline(always)]
    unsafe fn had8(r: &mut [__m256i; 8]) {
        unsafe {
            for d in [1, 2, 4] {
                for i in (0..8).filter(|i| i & d == 0) {
                    let (a, b) = (r[i], r[i + d]);
                    r[i] = _mm256_add_epi16(a, b);
                    r[i + d] = _mm256_sub_epi16(a, b);
                }
            }
        }
    }

    /// Transposes the $8\times8$ 16-bit matrix in each 128-bit lane of `r`.
    #[inline(always)]
    unsafe fn transpose8(r: &mut [__m256i; 8]) {
        unsafe {
            let lo16 = |a, b| _mm256_unpacklo_epi16(a, b);
            let hi16 = |a, b| _mm256_unpackhi_epi16(a, b);
            let (t0, t1) = (lo16(r[0], r[1]), hi16(r[0], r[1]));
            let (t2, t3) = (lo16(r[2], r[3]), hi16(r[2], r[3]));
            let (t4, t5) = (lo16(r[4], r[5]), hi16(r[4], r[5]));
            let (t6, t7) = (lo16(r[6], r[7]), hi16(r[6], r[7]));
            let (u0, u1) = (_mm256_unpacklo_epi32(t0, t2), _mm256_unpackhi_epi32(t0, t2));
            let (u2, u3) = (_mm256_unpacklo_epi32(t1, t3), _mm256_unpackhi_epi32(t1, t3));
            let (u4, u5) = (_mm256_unpacklo_epi32(t4, t6), _mm256_unpackhi_epi32(t4, t6));
            let (u6, u7) = (_mm256_unpacklo_epi32(t5, t7), _mm256_unpackhi_epi32(t5, t7));
            r[0] = _mm256_unpacklo_epi64(u0, u4);
            r[1] = _mm256_unpackhi_epi64(u0, u4);
            r[2] = _mm256_unpacklo_epi64(u1, u5);
            r[3] = _mm256_unpackhi_epi64(u1, u5);
            r[4] = _mm256_unpacklo_epi64(u2, u6);
            r[5] = _mm256_unpackhi_epi64(u2, u6);
            r[6] = _mm256_unpacklo_epi64(u3, u7);
            r[7] = _mm256_unpackhi_epi64(u3, u7);
        }
    }

    /// SA8D of $16\times16$ blocks: the left and right $8\times8$ blocks of eight rows side by
    /// side in the two lanes, a row per register.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn sa8d_16x16(a: *const u8, sa: usize, b: *const u8, sb: usize) -> u32 {
        unsafe {
            let ones = _mm256_set1_epi16(1);
            let mut total = 0;
            for half in 0..2 {
                let mut r: [__m256i; 8] = std::array::from_fn(|y| {
                    let row = half * 8 + y;
                    let pa =
                        _mm256_cvtepu8_epi16(_mm_loadu_si128(a.add(row * sa) as *const __m128i));
                    let pb =
                        _mm256_cvtepu8_epi16(_mm_loadu_si128(b.add(row * sb) as *const __m128i));
                    _mm256_sub_epi16(pa, pb)
                });
                had8(&mut r);
                transpose8(&mut r);
                had8(&mut r);
                // Magnitudes reach 64 * 255: two lanes of 32 bits per pair of them.
                let mut acc = _mm256_setzero_si256();
                for v in r {
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_abs_epi16(v), ones));
                }
                let hsum = |v: __m128i| {
                    let v = _mm_add_epi32(v, _mm_shuffle_epi32::<0b01_00_11_10>(v));
                    let v = _mm_add_epi32(v, _mm_shuffle_epi32::<0b10_11_00_01>(v));
                    _mm_cvtsi128_si32(v) as u32
                };
                total += (hsum(_mm256_castsi256_si128(acc)) + 2) >> 2;
                total += (hsum(_mm256_extracti128_si256::<1>(acc)) + 2) >> 2;
            }
            total
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// Eight samples of `a` (or the rounded average with `b` with `AVG`).
    #[inline(always)]
    unsafe fn pred8<const AVG: bool>(a: *const u8, b: *const u8) -> uint8x8_t {
        unsafe {
            if AVG {
                vrhadd_u8(vld1_u8(a), vld1_u8(b))
            } else {
                vld1_u8(a)
            }
        }
    }

    /// SAD of `W`x`H` blocks `p` and `a` (or the average of `a` and `b` with `AVG`).
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn sad<const W: usize, const H: usize, const AVG: bool>(
        p: *const u8,
        sp: usize,
        a: *const u8,
        sa: usize,
        b: *const u8,
        sb: usize,
    ) -> u32 {
        unsafe {
            let mut acc = vdupq_n_u16(0);
            for y in 0..H {
                for h in 0..W / 8 {
                    let q = pred8::<AVG>(a.add(y * sa + h * 8), b.add(y * sb + h * 8));
                    acc = vabal_u8(acc, vld1_u8(p.add(y * sp + h * 8)), q);
                }
            }
            vaddlvq_u16(acc)
        }
    }

    /// SATD of `W`x`H` blocks `p` and `a` (or the average of `a` and `b` with `AVG`), eight
    /// columns (two $4\times4$ blocks) of four rows per pass.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn satd<const W: usize, const H: usize, const AVG: bool>(
        p: *const u8,
        sp: usize,
        a: *const u8,
        sa: usize,
        b: *const u8,
        sb: usize,
    ) -> u32 {
        unsafe {
            let odd: [u16; 8] = [0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF];
            let odd = vld1q_u16(odd.as_ptr());
            let mut acc = vdupq_n_u32(0);
            for strip in 0..H / 4 {
                for half in 0..W / 8 {
                    let row = |k: usize| {
                        let r = strip * 4 + k;
                        let q = pred8::<AVG>(a.add(r * sa + half * 8), b.add(r * sb + half * 8));
                        vreinterpretq_s16_u16(vsubl_u8(vld1_u8(p.add(r * sp + half * 8)), q))
                    };
                    let (d0, d1, d2, d3) = (row(0), row(1), row(2), row(3));
                    let (s01, d01) = (vaddq_s16(d0, d1), vsubq_s16(d0, d1));
                    let (s23, d23) = (vaddq_s16(d2, d3), vsubq_s16(d2, d3));
                    let v = [
                        vaddq_s16(s01, s23),
                        vsubq_s16(s01, s23),
                        vaddq_s16(d01, d23),
                        vsubq_s16(d01, d23),
                    ];
                    let mut sum = vdupq_n_u16(0);
                    for x in v {
                        let t = vrev32q_s16(x);
                        let w = vbslq_s16(odd, vsubq_s16(t, x), vaddq_s16(x, t));
                        let u = vreinterpretq_s16_s32(vrev64q_s32(vreinterpretq_s32_s16(w)));
                        let m = vmaxq_s16(vabsq_s16(w), vabsq_s16(u));
                        sum = vaddq_u16(sum, vreinterpretq_u16_s16(m));
                    }
                    acc = vpadalq_u16(acc, sum);
                }
            }
            vaddvq_u32(acc) >> 1
        }
    }

    /// `(a + b + 1) >> 1` of `W`x`H` blocks into `d`.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn avg<const W: usize, const H: usize>(
        d: *mut u8,
        sd: usize,
        a: *const u8,
        sa: usize,
        b: *const u8,
        sb: usize,
    ) {
        unsafe {
            for y in 0..H {
                for h in 0..W / 8 {
                    let v = pred8::<true>(a.add(y * sa + h * 8), b.add(y * sb + h * 8));
                    vst1_u8(d.add(y * sd + h * 8), v);
                }
            }
        }
    }

    /// SA8D of $16\times16$ blocks, one $8\times8$ block at a time, a row per register.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn sa8d_16x16(a: *const u8, sa: usize, b: *const u8, sb: usize) -> u32 {
        unsafe {
            let had8 = |r: &mut [int16x8_t; 8]| {
                for d in [1, 2, 4] {
                    for i in (0..8).filter(|i| i & d == 0) {
                        let (x, y) = (r[i], r[i + d]);
                        r[i] = vaddq_s16(x, y);
                        r[i + d] = vsubq_s16(x, y);
                    }
                }
            };
            let mut total = 0;
            for q in 0..4 {
                let (x0, y0) = ((q & 1) * 8, (q >> 1) * 8);
                let mut r: [int16x8_t; 8] = std::array::from_fn(|y| {
                    let (pa, pb) = (a.add((y0 + y) * sa + x0), b.add((y0 + y) * sb + x0));
                    vreinterpretq_s16_u16(vsubl_u8(vld1_u8(pa), vld1_u8(pb)))
                });
                had8(&mut r);
                // Transpose, interleaving 16-, 32- and 64-bit elements.
                let z16 = |x, y| (vzip1q_s16(x, y), vzip2q_s16(x, y));
                let z32 = |x: int16x8_t, y: int16x8_t| {
                    let (x, y) = (vreinterpretq_s32_s16(x), vreinterpretq_s32_s16(y));
                    (
                        vreinterpretq_s16_s32(vzip1q_s32(x, y)),
                        vreinterpretq_s16_s32(vzip2q_s32(x, y)),
                    )
                };
                let z64 = |x: int16x8_t, y: int16x8_t| {
                    let (x, y) = (vreinterpretq_s64_s16(x), vreinterpretq_s64_s16(y));
                    (
                        vreinterpretq_s16_s64(vzip1q_s64(x, y)),
                        vreinterpretq_s16_s64(vzip2q_s64(x, y)),
                    )
                };
                let ((t0, t1), (t2, t3)) = (z16(r[0], r[1]), z16(r[2], r[3]));
                let ((t4, t5), (t6, t7)) = (z16(r[4], r[5]), z16(r[6], r[7]));
                let ((u0, u1), (u2, u3)) = (z32(t0, t2), z32(t1, t3));
                let ((u4, u5), (u6, u7)) = (z32(t4, t6), z32(t5, t7));
                (r[0], r[1]) = z64(u0, u4);
                (r[2], r[3]) = z64(u1, u5);
                (r[4], r[5]) = z64(u2, u6);
                (r[6], r[7]) = z64(u3, u7);
                had8(&mut r);
                let mut acc = vdupq_n_u32(0);
                for v in r {
                    acc = vpadalq_u16(acc, vreinterpretq_u16_s16(vabsq_s16(v)));
                }
                total += (vaddvq_u32(acc) + 2) >> 2;
            }
            total
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic noise with runs of extremes, so differences reach both ends of their range.
    fn plane(w: usize, h: usize, seed: u32) -> Vec<u8> {
        let mut x = seed;
        (0..w * h)
            .map(|i| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                match (i / 7) % 5 {
                    0 => 0,
                    1 => 255,
                    _ => (x >> 24) as u8,
                }
            })
            .collect()
    }

    fn check_size<const W: usize, const H: usize>() {
        let (pw, ph) = (53usize, 41usize);
        let s = plane(pw, ph, 5);
        let a = plane(pw, ph, 17);
        let b = plane(pw, ph, 29);
        let flat = vec![255u8; pw * ph];
        let zero = vec![0u8; pw * ph];
        for &(so, ao, bo) in &[
            (0usize, 0usize, 0usize),
            (3, 1, 7),
            (pw + 5, 2 * pw + 9, 11),
            (19, 33, 3 * pw),
        ] {
            for (p, q, r) in [
                (&s, &a, &b),
                (&flat, &zero, &zero),
                (&zero, &flat, &flat),
                (&s, &s, &s),
                (&zero, &flat, &zero),
            ] {
                let (p, q, r) = (&p[so..], &q[ao..], &r[bo..]);
                assert_eq!(
                    pixel_sad_wxh::<W, H>(p, pw, q, pw),
                    pixel_sad_wxh_scalar::<W, H>(p, pw, q, pw)
                );
                let refs = [&q[0..], &q[1..], &r[pw + 3..], &q[2 * pw + 5..]];
                assert_eq!(
                    pixel_sad_x4_wxh::<W, H>(p, pw, refs, pw),
                    pixel_sad_x4_wxh_scalar::<W, H>(p, pw, refs, pw)
                );
                assert_eq!(
                    pixel_sad_avg_wxh::<W, H>(p, pw, q, pw, r, pw),
                    pixel_sad_avg_wxh_scalar::<W, H>(p, pw, q, pw, r, pw)
                );
                assert_eq!(
                    pixel_satd_wxh::<W, H>(p, pw, q, pw, None),
                    pixel_satd_wxh_scalar::<W, H>(p, pw, q, pw, None),
                    "{W}x{H} at {so} {ao}"
                );
                assert_eq!(
                    pixel_satd_wxh::<W, H>(p, pw, q, pw, Some((r, pw))),
                    pixel_satd_wxh_scalar::<W, H>(p, pw, q, pw, Some((r, pw))),
                    "{W}x{H} avg at {so} {ao} {bo}"
                );
                let (mut d1, mut d2) = (vec![1u8; 16 * 21], vec![1u8; 16 * 21]);
                pixel_avg_wxh::<W, H>(&mut d1[5..], 21, q, pw, r, pw);
                pixel_avg_wxh_scalar::<W, H>(&mut d2[5..], 21, q, pw, r, pw);
                assert_eq!(d1, d2);
                if W == 16 && H == 16 {
                    assert_eq!(
                        pixel_sa8d_16x16(p, pw, q, pw),
                        pixel_sa8d_16x16_scalar(p, pw, q, pw)
                    );
                }
            }
        }
    }

    #[test]
    fn block_costs_match_scalar_at_every_size() {
        check_size::<8, 8>();
        check_size::<16, 8>();
        check_size::<8, 16>();
        check_size::<16, 16>();
    }

    #[test]
    fn satd_matches_the_16x16_kernel() {
        let a = plane(40, 40, 1);
        let b = plane(40, 40, 2);
        assert_eq!(
            pixel_satd_wxh::<16, 16>(&a, 40, &b, 40, None),
            crate::pixel_satd_16x16(&a, 40, &b, 40)
        );
    }
}
