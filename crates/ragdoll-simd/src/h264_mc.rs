//! Partition-sized H.264 inter prediction (`SIMD-0076`): quarter-pel luma (ITU-T H.264 8.4.2.2.1)
//! and eighth-pel chroma (8.4.2.2.2) samples read straight from a reference plane.
//!
//! - [`h264_luma_mc`]: `w`x`h` luma blocks, `w, h` in `{4, 8, 16}`, computing only the filters the
//!   quarter-pel phase needs (`h264qpel_template.c`).
//! - [`h264_chroma_mc`]: `w`x`h` 4:2:0 chroma blocks, `w, h` in `{2, 4, 8}` (`h264chroma_template.c`).
//! - [`h264_weight_bi`] and [`h264_weight_uni`]: weighted sample prediction (8.4.2.3.2, `h264dsp_template.c`).
//!
//! The kernels never clamp coordinates: the caller guarantees the margins given by
//! [`luma_mc_cols`] and [`chroma_mc_cols`], and copies an edge-extended window (FFmpeg's
//! `emulated_edge_mc`) when a block reaches past the picture edge. AVX2 and NEON compute at least
//! 8 columns, so those margins cover 8 columns even for narrower blocks.

#[inline(always)]
fn clip(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

#[inline(always)]
fn avg(a: u8, b: u8) -> u8 {
    ((a as u16 + b as u16 + 1) >> 1) as u8
}

/// Columns a luma kernel reads at and right of the block origin for a `w`-wide block; it also
/// reads 2 columns to the left, 2 rows above and `h + 3` rows from the origin down.
pub const fn luma_mc_cols(w: usize) -> usize {
    if w < 8 { 8 + 3 } else { w + 3 }
}

/// Columns a chroma kernel reads at and right of the block origin for a `w`-wide block; it reads
/// `h + 1` rows from the origin down.
pub const fn chroma_mc_cols(w: usize) -> usize {
    if w < 8 { 9 } else { w + 1 }
}

#[allow(clippy::too_many_arguments)]
fn check_luma(dst: &[u8], ds: usize, src: &[u8], off: usize, ss: usize, w: usize, h: usize) {
    assert!(matches!(w, 4 | 8 | 16) && matches!(h, 4 | 8 | 16));
    assert!(off >= 2 * ss + 2 && off + (h + 2) * ss + luma_mc_cols(w) <= src.len());
    assert!((h - 1) * ds + w <= dst.len());
}

#[allow(clippy::too_many_arguments)]
fn check_chroma(dst: &[u8], ds: usize, src: &[u8], off: usize, ss: usize, w: usize, h: usize) {
    assert!(matches!(w, 2 | 4 | 8) && matches!(h, 2 | 4 | 8));
    assert!(off + h * ss + chroma_mc_cols(w) <= src.len());
    assert!((h - 1) * ds + w <= dst.len());
}

/// Writes the `w`x`h` luma prediction at quarter-pel phase `(mx, my)` (`0..=3`) for the block whose
/// integer-sample origin is `src[off]`, into `dst` with stride `ds`.
///
/// # Panics
/// If `w` or `h` is not 4, 8 or 16, or `src` lacks the margins of [`luma_mc_cols`].
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn h264_luma_mc(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    off: usize,
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
) {
    luma_mc::<false>(dst, ds, src, off, ss, w, h, mx, my);
}

/// [`h264_luma_mc`] averaged into `dst`: `dst = (dst + pred + 1) >> 1`, the second list of
/// default bi-prediction (8.4.2.3.1).
///
/// # Panics
/// As [`h264_luma_mc`].
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn h264_luma_mc_avg(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    off: usize,
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
) {
    luma_mc::<true>(dst, ds, src, off, ss, w, h, mx, my);
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn luma_mc<const AVG: bool>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    off: usize,
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
) {
    check_luma(dst, ds, src, off, ss, w, h);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above; NEON is baseline on aarch64.
        unsafe { luma_neon::<AVG>(dst, ds, src, off, ss, w, h, mx & 3, my & 3) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        unsafe { luma_avx2::<AVG>(dst, ds, src, off, ss, w, h, mx & 3, my & 3) };
        return;
    }
    if AVG {
        h264_luma_mc_avg_scalar(dst, ds, src, off, ss, w, h, mx, my);
    } else {
        h264_luma_mc_scalar(dst, ds, src, off, ss, w, h, mx, my);
    }
}

/// Scalar reference for [`h264_luma_mc_avg`].
#[allow(clippy::too_many_arguments)]
pub fn h264_luma_mc_avg_scalar(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    off: usize,
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
) {
    let mut t = [0u8; 256];
    h264_luma_mc_scalar(&mut t, 16, src, off, ss, w, h, mx, my);
    h264_avg_scalar(dst, ds, &t, 16, w, h);
}

/// Scalar reference for [`h264_luma_mc`].
#[allow(clippy::too_many_arguments)]
pub fn h264_luma_mc_scalar(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    off: usize,
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
) {
    check_luma(dst, ds, src, off, ss, w, h);
    let s = ss as isize;
    let at = |r: isize, c: isize| src[(off as isize + r * s + c) as usize] as i32;
    let tap6 =
        |a: i32, b: i32, c: i32, d: i32, e: i32, f: i32| a - 5 * b + 20 * c + 20 * d - 5 * e + f;
    let h_raw = |r: isize, c: isize| {
        tap6(
            at(r, c - 2),
            at(r, c - 1),
            at(r, c),
            at(r, c + 1),
            at(r, c + 2),
            at(r, c + 3),
        )
    };
    let g = |r: isize, c: isize| at(r, c) as u8;
    let b = |r: isize, c: isize| clip((h_raw(r, c) + 16) >> 5);
    let v = |r: isize, c: isize| {
        let raw = tap6(
            at(r - 2, c),
            at(r - 1, c),
            at(r, c),
            at(r + 1, c),
            at(r + 2, c),
            at(r + 3, c),
        );
        clip((raw + 16) >> 5)
    };
    let j = |r: isize, c: isize| {
        let raw = tap6(
            h_raw(r - 2, c),
            h_raw(r - 1, c),
            h_raw(r, c),
            h_raw(r + 1, c),
            h_raw(r + 2, c),
            h_raw(r + 3, c),
        );
        clip((raw + 512) >> 10)
    };
    for r in 0..h as isize {
        for c in 0..w as isize {
            dst[r as usize * ds + c as usize] = match (mx & 3, my & 3) {
                (0, 0) => g(r, c),
                (1, 0) => avg(g(r, c), b(r, c)),
                (2, 0) => b(r, c),
                (3, 0) => avg(b(r, c), g(r, c + 1)),
                (0, 1) => avg(g(r, c), v(r, c)),
                (1, 1) => avg(b(r, c), v(r, c)),
                (2, 1) => avg(b(r, c), j(r, c)),
                (3, 1) => avg(b(r, c), v(r, c + 1)),
                (0, 2) => v(r, c),
                (1, 2) => avg(v(r, c), j(r, c)),
                (2, 2) => j(r, c),
                (3, 2) => avg(j(r, c), v(r, c + 1)),
                (0, 3) => avg(v(r, c), g(r + 1, c)),
                (1, 3) => avg(v(r, c), b(r + 1, c)),
                (2, 3) => avg(j(r, c), b(r + 1, c)),
                _ => avg(v(r, c + 1), b(r + 1, c)),
            };
        }
    }
}

/// [`h264_luma_mc`] (with `avg`, [`h264_luma_mc_avg`]) for the `w`x`h` block at `(x, y)` of a
/// `width`x`height` plane displaced by the quarter-sample vector `mv`, when every sample its
/// filter reads lies inside the plane; otherwise nothing is written and the result is `false`.
/// One test of the block position covers the kernel's bounds.
///
/// # Panics
/// If `w` or `h` is not 4, 8 or 16, `src` holds fewer than `width * height` samples or `dst`
/// is too short for the block.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn h264_luma_mc_in_plane(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    width: usize,
    height: usize,
    (x, y): (usize, usize),
    (w, h): (usize, usize),
    mv: [i16; 2],
    avg: bool,
) -> bool {
    let ix = x as isize + (mv[0] as isize >> 2);
    let iy = y as isize + (mv[1] as isize >> 2);
    if !(ix >= 2
        && iy >= 2
        && ix as usize + luma_mc_cols(w) <= width
        && iy as usize + h + 3 <= height)
    {
        return false;
    }
    // Rows iy - 2 ..= iy + h + 2 and columns ix - 2 .. ix + luma_mc_cols(w) lie in the plane.
    assert!(matches!(w, 4 | 8 | 16) && matches!(h, 4 | 8 | 16));
    assert!(width.checked_mul(height).is_some_and(|n| n <= src.len()));
    assert!((h - 1) * ds + w <= dst.len());
    let off = iy as usize * width + ix as usize;
    let (mx, my) = ((mv[0] & 3) as u8, (mv[1] & 3) as u8);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds established above; NEON is baseline on aarch64.
        unsafe {
            if avg {
                luma_neon::<true>(dst, ds, src, off, width, w, h, mx, my)
            } else {
                luma_neon::<false>(dst, ds, src, off, width, w, h, mx, my)
            }
        };
        return true;
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds established above; avx2() is true only when the CPU supports AVX2.
        unsafe {
            if avg {
                luma_avx2::<true>(dst, ds, src, off, width, w, h, mx, my)
            } else {
                luma_avx2::<false>(dst, ds, src, off, width, w, h, mx, my)
            }
        };
        return true;
    }
    if avg {
        h264_luma_mc_avg_scalar(dst, ds, src, off, width, w, h, mx, my);
    } else {
        h264_luma_mc_scalar(dst, ds, src, off, width, w, h, mx, my);
    }
    true
}

/// [`h264_chroma_mc`] (with `avg`, [`h264_chroma_mc_avg`]) for the `w`x`h` block at `(x, y)`
/// of a `width`x`height` plane displaced by the eighth-sample vector `mv`, when every sample
/// it reads lies inside the plane; otherwise nothing is written and the result is `false`.
///
/// # Panics
/// If `w` or `h` is not 2, 4 or 8, `src` holds fewer than `width * height` samples or `dst`
/// is too short for the block.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn h264_chroma_mc_in_plane(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    width: usize,
    height: usize,
    (x, y): (usize, usize),
    (w, h): (usize, usize),
    mv: [i16; 2],
    avg: bool,
) -> bool {
    let ix = x as isize + (mv[0] as isize >> 3);
    let iy = y as isize + (mv[1] as isize >> 3);
    if !(ix >= 0
        && iy >= 0
        && ix as usize + chroma_mc_cols(w) <= width
        && (iy as usize + h) < height)
    {
        return false;
    }
    // Rows iy ..= iy + h and columns ix .. ix + chroma_mc_cols(w) lie in the plane.
    assert!(matches!(w, 2 | 4 | 8) && matches!(h, 2 | 4 | 8));
    assert!(width.checked_mul(height).is_some_and(|n| n <= src.len()));
    assert!((h - 1) * ds + w <= dst.len());
    let off = iy as usize * width + ix as usize;
    let (fx, fy) = ((mv[0] & 7) as u8, (mv[1] & 7) as u8);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds established above; NEON is baseline on aarch64.
        unsafe {
            if avg {
                chroma_neon::<true>(dst, ds, src, off, width, w, h, fx, fy)
            } else {
                chroma_neon::<false>(dst, ds, src, off, width, w, h, fx, fy)
            }
        };
        return true;
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds established above; avx2() is true only when the CPU supports AVX2.
        unsafe {
            if avg {
                chroma_avx2::<true>(dst, ds, src, off, width, w, h, fx, fy)
            } else {
                chroma_avx2::<false>(dst, ds, src, off, width, w, h, fx, fy)
            }
        };
        return true;
    }
    if avg {
        h264_chroma_mc_avg_scalar(dst, ds, src, off, width, w, h, fx, fy);
    } else {
        h264_chroma_mc_scalar(dst, ds, src, off, width, w, h, fx, fy);
    }
    true
}

/// Writes the `w`x`h` chroma prediction at eighth-pel phase `(fx, fy)` (`0..=7`) for the block whose
/// integer-sample origin is `src[off]`, into `dst` with stride `ds`.
///
/// # Panics
/// If `w` or `h` is not 2, 4 or 8, or `src` lacks the margins of [`chroma_mc_cols`].
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn h264_chroma_mc(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    off: usize,
    ss: usize,
    w: usize,
    h: usize,
    fx: u8,
    fy: u8,
) {
    chroma_mc::<false>(dst, ds, src, off, ss, w, h, fx, fy);
}

/// [`h264_chroma_mc`] averaged into `dst`: `dst = (dst + pred + 1) >> 1`.
///
/// # Panics
/// As [`h264_chroma_mc`].
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn h264_chroma_mc_avg(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    off: usize,
    ss: usize,
    w: usize,
    h: usize,
    fx: u8,
    fy: u8,
) {
    chroma_mc::<true>(dst, ds, src, off, ss, w, h, fx, fy);
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn chroma_mc<const AVG: bool>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    off: usize,
    ss: usize,
    w: usize,
    h: usize,
    fx: u8,
    fy: u8,
) {
    check_chroma(dst, ds, src, off, ss, w, h);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above; NEON is baseline on aarch64.
        unsafe { chroma_neon::<AVG>(dst, ds, src, off, ss, w, h, fx & 7, fy & 7) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        unsafe { chroma_avx2::<AVG>(dst, ds, src, off, ss, w, h, fx & 7, fy & 7) };
        return;
    }
    if AVG {
        h264_chroma_mc_avg_scalar(dst, ds, src, off, ss, w, h, fx, fy);
    } else {
        h264_chroma_mc_scalar(dst, ds, src, off, ss, w, h, fx, fy);
    }
}

/// Scalar reference for [`h264_chroma_mc_avg`].
#[allow(clippy::too_many_arguments)]
pub fn h264_chroma_mc_avg_scalar(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    off: usize,
    ss: usize,
    w: usize,
    h: usize,
    fx: u8,
    fy: u8,
) {
    let mut t = [0u8; 64];
    h264_chroma_mc_scalar(&mut t, 8, src, off, ss, w, h, fx, fy);
    h264_avg_scalar(dst, ds, &t, 8, w, h);
}

/// Scalar reference for [`h264_chroma_mc`].
#[allow(clippy::too_many_arguments)]
pub fn h264_chroma_mc_scalar(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    off: usize,
    ss: usize,
    w: usize,
    h: usize,
    fx: u8,
    fy: u8,
) {
    check_chroma(dst, ds, src, off, ss, w, h);
    let (fx, fy) = ((fx & 7) as u32, (fy & 7) as u32);
    let (wa, wb, wc, wd) = ((8 - fx) * (8 - fy), fx * (8 - fy), (8 - fx) * fy, fx * fy);
    for r in 0..h {
        let s0 = &src[off + r * ss..];
        let s1 = &src[off + (r + 1) * ss..];
        for c in 0..w {
            let v = wa * s0[c] as u32
                + wb * s0[c + 1] as u32
                + wc * s1[c] as u32
                + wd * s1[c + 1] as u32;
            dst[r * ds + c] = ((v + 32) >> 6) as u8;
        }
    }
}

/// Default bi-prediction (8.4.2.3.1): `dst = (dst + src + 1) >> 1` over `w`x`h` samples.
///
/// # Panics
/// If a buffer is too short for `h` rows of `w` samples.
#[inline]
pub fn h264_avg(dst: &mut [u8], ds: usize, src: &[u8], ss: usize, w: usize, h: usize) {
    assert!(matches!(w, 2 | 4 | 8 | 16) && h > 0);
    assert!((h - 1) * ds + w <= dst.len() && (h - 1) * ss + w <= src.len());
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        unsafe { avg_avx2(dst, ds, src, ss, w, h) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above; NEON is baseline on aarch64.
        unsafe { avg_neon(dst, ds, src, ss, w, h) };
        return;
    }
    h264_avg_scalar(dst, ds, src, ss, w, h);
}

/// Scalar reference for [`h264_avg`].
pub fn h264_avg_scalar(dst: &mut [u8], ds: usize, src: &[u8], ss: usize, w: usize, h: usize) {
    for r in 0..h {
        for c in 0..w {
            dst[r * ds + c] = avg(dst[r * ds + c], src[r * ss + c]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avg_avx2(dst: &mut [u8], ds: usize, src: &[u8], ss: usize, w: usize, h: usize) {
    use std::arch::x86_64::*;
    let (d, s) = (dst.as_mut_ptr(), src.as_ptr());
    for r in 0..h {
        unsafe {
            let (d, s) = (d.add(r * ds), s.add(r * ss));
            match w {
                16 => {
                    let v = _mm_avg_epu8(
                        _mm_loadu_si128(d as *const __m128i),
                        _mm_loadu_si128(s as *const __m128i),
                    );
                    _mm_storeu_si128(d as *mut __m128i, v);
                }
                8 => {
                    let v = _mm_avg_epu8(
                        _mm_loadl_epi64(d as *const __m128i),
                        _mm_loadl_epi64(s as *const __m128i),
                    );
                    _mm_storel_epi64(d as *mut __m128i, v);
                }
                4 => {
                    let v = _mm_avg_epu8(
                        _mm_cvtsi32_si128((d as *const i32).read_unaligned()),
                        _mm_cvtsi32_si128((s as *const i32).read_unaligned()),
                    );
                    (d as *mut i32).write_unaligned(_mm_cvtsi128_si32(v));
                }
                _ => {
                    for c in 0..w {
                        *d.add(c) = avg(*d.add(c), *s.add(c));
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn avg_neon(dst: &mut [u8], ds: usize, src: &[u8], ss: usize, w: usize, h: usize) {
    use std::arch::aarch64::*;
    let (d, s) = (dst.as_mut_ptr(), src.as_ptr());
    for r in 0..h {
        unsafe {
            let (d, s) = (d.add(r * ds), s.add(r * ss));
            match w {
                16 => vst1q_u8(d, vrhaddq_u8(vld1q_u8(d), vld1q_u8(s))),
                8 => vst1_u8(d, vrhadd_u8(vld1_u8(d), vld1_u8(s))),
                _ => {
                    for c in 0..w {
                        *d.add(c) = avg(*d.add(c), *s.add(c));
                    }
                }
            }
        }
    }
}

/// Weighted bi-prediction (8.4.2.3.2) of `w`x`h` samples from `a` and `b` (sharing stride `s`):
/// `clip(((a * w0 + b * w1 + 2^log_wd) >> (log_wd + 1)) + off)`, where `off` is the already
/// rounded `(o0 + o1 + 1) >> 1`.
///
/// # Panics
/// If a buffer is too short for `h` rows of `w` samples.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn h264_weight_bi(
    dst: &mut [u8],
    ds: usize,
    a: &[u8],
    b: &[u8],
    s: usize,
    w: usize,
    h: usize,
    log_wd: u32,
    w0: i32,
    w1: i32,
    off: i32,
) {
    assert!(w > 0 && h > 0 && log_wd <= 7 && (h - 1) * ds + w <= dst.len());
    assert!((h - 1) * s + w.max(8) <= a.len().min(b.len()));
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        unsafe {
            weight_avx2(dst, ds, a, Some(b), s, w, h, log_wd + 1, [w0, w1], {
                (1 << log_wd) + (off << (log_wd + 1))
            })
        };
        return;
    }
    h264_weight_bi_scalar(dst, ds, a, b, s, w, h, log_wd, w0, w1, off);
}

/// Scalar reference for [`h264_weight_bi`].
#[allow(clippy::too_many_arguments)]
pub fn h264_weight_bi_scalar(
    dst: &mut [u8],
    ds: usize,
    a: &[u8],
    b: &[u8],
    s: usize,
    w: usize,
    h: usize,
    log_wd: u32,
    w0: i32,
    w1: i32,
    off: i32,
) {
    for r in 0..h {
        for c in 0..w {
            let v = (a[r * s + c] as i32 * w0 + b[r * s + c] as i32 * w1 + (1 << log_wd))
                >> (log_wd + 1);
            dst[r * ds + c] = clip(v + off);
        }
    }
}

/// Weighted uni-prediction (8.4.2.3.2) of `w`x`h` samples from `a` (stride `s`):
/// `clip(((a * wt + 2^(log_wd - 1)) >> log_wd) + off)`, or `clip(a * wt + off)` when `log_wd` is 0.
///
/// # Panics
/// If a buffer is too short for `h` rows of `w` samples.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn h264_weight_uni(
    dst: &mut [u8],
    ds: usize,
    a: &[u8],
    s: usize,
    w: usize,
    h: usize,
    log_wd: u32,
    wt: i32,
    off: i32,
) {
    assert!(w > 0 && h > 0 && log_wd <= 7 && (h - 1) * ds + w <= dst.len());
    assert!((h - 1) * s + w.max(8) <= a.len());
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        let round = if log_wd >= 1 { 1 << (log_wd - 1) } else { 0 };
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        unsafe {
            weight_avx2(
                dst,
                ds,
                a,
                None,
                s,
                w,
                h,
                log_wd,
                [wt, 0],
                round + (off << log_wd),
            )
        };
        return;
    }
    h264_weight_uni_scalar(dst, ds, a, s, w, h, log_wd, wt, off);
}

/// Scalar reference for [`h264_weight_uni`].
#[allow(clippy::too_many_arguments)]
pub fn h264_weight_uni_scalar(
    dst: &mut [u8],
    ds: usize,
    a: &[u8],
    s: usize,
    w: usize,
    h: usize,
    log_wd: u32,
    wt: i32,
    off: i32,
) {
    for r in 0..h {
        for c in 0..w {
            let x = a[r * s + c] as i32 * wt;
            let v = if log_wd >= 1 {
                (x + (1 << (log_wd - 1))) >> log_wd
            } else {
                x
            };
            dst[r * ds + c] = clip(v + off);
        }
    }
}

// ----- AVX2 ------------------------------------------------------------------------------------

/// Luma prediction into a 16-stride scratch block, then stored (`w` columns) into `dst`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn luma_avx2<const AVG: bool>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    off: usize,
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
) {
    if (mx, my) == (0, 0) {
        // Whole-sample blocks (the most common kind): copied, or averaged into `dst`, straight
        // from the reference, two rows per step (`h` is even).
        use std::arch::x86_64::*;
        unsafe {
            let (mut d, mut s) = (dst.as_mut_ptr(), src.as_ptr().add(off));
            for _ in 0..h / 2 {
                match w {
                    16 => {
                        let mut a = _mm_loadu_si128(s as *const __m128i);
                        let mut b = _mm_loadu_si128(s.add(ss) as *const __m128i);
                        if AVG {
                            a = _mm_avg_epu8(a, _mm_loadu_si128(d as *const __m128i));
                            b = _mm_avg_epu8(b, _mm_loadu_si128(d.add(ds) as *const __m128i));
                        }
                        _mm_storeu_si128(d as *mut __m128i, a);
                        _mm_storeu_si128(d.add(ds) as *mut __m128i, b);
                    }
                    8 => {
                        let mut a = _mm_loadl_epi64(s as *const __m128i);
                        let mut b = _mm_loadl_epi64(s.add(ss) as *const __m128i);
                        if AVG {
                            a = _mm_avg_epu8(a, _mm_loadl_epi64(d as *const __m128i));
                            b = _mm_avg_epu8(b, _mm_loadl_epi64(d.add(ds) as *const __m128i));
                        }
                        _mm_storel_epi64(d as *mut __m128i, a);
                        _mm_storel_epi64(d.add(ds) as *mut __m128i, b);
                    }
                    _ => {
                        let ld =
                            |p: *const u8| _mm_cvtsi32_si128((p as *const i32).read_unaligned());
                        let mut a = ld(s);
                        let mut b = ld(s.add(ss));
                        if AVG {
                            a = _mm_avg_epu8(a, ld(d));
                            b = _mm_avg_epu8(b, ld(d.add(ds)));
                        }
                        (d as *mut i32).write_unaligned(_mm_cvtsi128_si32(a));
                        (d.add(ds) as *mut i32).write_unaligned(_mm_cvtsi128_si32(b));
                    }
                }
                s = s.add(2 * ss);
                d = d.add(2 * ds);
            }
        }
        return;
    }
    if w == 16 {
        unsafe { luma_avx2_w::<16, AVG>(dst, ds, src, off, ss, w, h, mx, my) }
    } else {
        unsafe { luma_avx2_w::<8, AVG>(dst, ds, src, off, ss, w, h, mx, my) }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    /// Loads `W` (8 or 16) bytes at `p` widened to `i16` lanes (upper half unused when `W == 8`).
    #[inline(always)]
    pub unsafe fn load_w16<const W: usize>(p: *const u8) -> __m256i {
        unsafe {
            if W == 16 {
                _mm256_cvtepu8_epi16(_mm_loadu_si128(p as *const __m128i))
            } else {
                _mm256_cvtepu8_epi16(_mm_loadl_epi64(p as *const __m128i))
            }
        }
    }

    /// `a + f - 5 (b + e) + 20 (c + d)` in `i16` lanes (exact for 8-bit samples).
    #[inline(always)]
    pub unsafe fn tap6(
        a: __m256i,
        b: __m256i,
        c: __m256i,
        d: __m256i,
        e: __m256i,
        f: __m256i,
    ) -> __m256i {
        unsafe {
            let af = _mm256_add_epi16(a, f);
            let be = _mm256_add_epi16(b, e);
            let cd = _mm256_add_epi16(c, d);
            _mm256_add_epi16(
                _mm256_sub_epi16(af, _mm256_mullo_epi16(be, _mm256_set1_epi16(5))),
                _mm256_mullo_epi16(cd, _mm256_set1_epi16(20)),
            )
        }
    }

    /// Packs 16 `i16` lanes to `u8` with unsigned saturation, in order, into the low 128 bits.
    #[inline(always)]
    pub unsafe fn pack16(v: __m256i) -> __m128i {
        unsafe {
            let p = _mm256_packus_epi16(v, v);
            _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0b1000>(p))
        }
    }

    #[inline(always)]
    pub unsafe fn store<const W: usize>(p: *mut u8, v: __m128i) {
        unsafe {
            if W == 16 {
                _mm_storeu_si128(p as *mut __m128i, v)
            } else {
                _mm_storel_epi64(p as *mut __m128i, v)
            }
        }
    }

    #[inline(always)]
    pub unsafe fn load<const W: usize>(p: *const u8) -> __m128i {
        unsafe {
            if W == 16 {
                _mm_loadu_si128(p as *const __m128i)
            } else {
                _mm_loadl_epi64(p as *const __m128i)
            }
        }
    }
}

/// Horizontal half-sample filter `b` (8.4.2.2.1) for `rows` rows from `s` into `t` (stride 16).
/// Row `r` goes to `t + r * ts`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn h_lowpass_avx2<const W: usize>(
    t: *mut u8,
    ts: usize,
    s: *const u8,
    ss: usize,
    rows: usize,
) {
    use std::arch::x86_64::*;
    use x86::*;
    let r16 = _mm256_set1_epi16(16);
    for r in 0..rows {
        unsafe {
            let p = s.add(r * ss);
            let v = tap6(
                load_w16::<W>(p.sub(2)),
                load_w16::<W>(p.sub(1)),
                load_w16::<W>(p),
                load_w16::<W>(p.add(1)),
                load_w16::<W>(p.add(2)),
                load_w16::<W>(p.add(3)),
            );
            let v = _mm256_srai_epi16::<5>(_mm256_add_epi16(v, r16));
            store::<W>(t.add(r * ts), pack16(v));
        }
    }
}

/// Vertical half-sample filter `h` for `rows` rows from `s` into `t` (stride 16).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn v_lowpass_avx2<const W: usize>(
    t: *mut u8,
    ts: usize,
    s: *const u8,
    ss: usize,
    rows: usize,
) {
    use std::arch::x86_64::*;
    use x86::*;
    let r16 = _mm256_set1_epi16(16);
    unsafe {
        let mut a = load_w16::<W>(s.sub(2 * ss));
        let mut b = load_w16::<W>(s.sub(ss));
        let mut c = load_w16::<W>(s);
        let mut d = load_w16::<W>(s.add(ss));
        let mut e = load_w16::<W>(s.add(2 * ss));
        for r in 0..rows {
            let f = load_w16::<W>(s.add((r + 3) * ss));
            let v = _mm256_srai_epi16::<5>(_mm256_add_epi16(tap6(a, b, c, d, e, f), r16));
            store::<W>(t.add(r * ts), pack16(v));
            (a, b, c, d, e) = (b, c, d, e, f);
        }
    }
}

/// Centre half-sample filter `j`: horizontal `i16` intermediates, then the vertical filter in `i32`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hv_lowpass_avx2<const W: usize>(
    t: *mut u8,
    ts: usize,
    s: *const u8,
    ss: usize,
    rows: usize,
) {
    use std::arch::x86_64::*;
    use x86::*;
    let mut mid = [_mm256_setzero_si256(); 21];
    for (k, m) in mid.iter_mut().enumerate().take(rows + 5) {
        unsafe {
            let p = s.add(k * ss).sub(2 * ss);
            *m = tap6(
                load_w16::<W>(p.sub(2)),
                load_w16::<W>(p.sub(1)),
                load_w16::<W>(p),
                load_w16::<W>(p.add(1)),
                load_w16::<W>(p.add(2)),
                load_w16::<W>(p.add(3)),
            );
        }
    }
    // (a + f) - 5 (b + e) + 20 (c + d): each pair sum fits i16 (|tap6| <= 10710); the weighted sum
    // needs i32, formed with madd over interleaved (af, be) and (cd, 0) pairs.
    let w_af_be = _mm256_set1_epi32(((-5i32) << 16) | 1);
    let w_cd = _mm256_set1_epi32(20);
    let zero = _mm256_setzero_si256();
    let r512 = _mm256_set1_epi32(512);
    for r in 0..rows {
        let af = _mm256_add_epi16(mid[r], mid[r + 5]);
        let be = _mm256_add_epi16(mid[r + 1], mid[r + 4]);
        let cd = _mm256_add_epi16(mid[r + 2], mid[r + 3]);
        let lo = _mm256_add_epi32(
            _mm256_madd_epi16(_mm256_unpacklo_epi16(af, be), w_af_be),
            _mm256_madd_epi16(_mm256_unpacklo_epi16(cd, zero), w_cd),
        );
        let hi = _mm256_add_epi32(
            _mm256_madd_epi16(_mm256_unpackhi_epi16(af, be), w_af_be),
            _mm256_madd_epi16(_mm256_unpackhi_epi16(cd, zero), w_cd),
        );
        let lo = _mm256_srai_epi32::<10>(_mm256_add_epi32(lo, r512));
        let hi = _mm256_srai_epi32::<10>(_mm256_add_epi32(hi, r512));
        unsafe { store::<W>(t.add(r * ts), pack16(_mm256_packs_epi32(lo, hi))) };
    }
}

/// Copies `rows` rows of `W` samples from `s` into `t` (stride 16).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn copy_avx2<const W: usize>(t: *mut u8, ts: usize, s: *const u8, ss: usize, rows: usize) {
    use x86::*;
    for r in 0..rows {
        unsafe { store::<W>(t.add(r * ts), load::<W>(s.add(r * ss))) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn luma_avx2_w<const W: usize, const AVG: bool>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    off: usize,
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
) {
    use std::arch::x86_64::*;
    use x86::*;
    let s = unsafe { src.as_ptr().add(off) };
    // Scratch rows are only read after being written, through raw pointers, so they need no
    // initialisation.
    let mut b0 = std::mem::MaybeUninit::<[u8; 256]>::uninit();
    let mut b1 = std::mem::MaybeUninit::<[u8; 256]>::uninit();
    let (t0, t1) = (b0.as_mut_ptr() as *mut u8, b1.as_mut_ptr() as *mut u8);
    // A phase that is one filter's output is written straight to `dst` when the block is as
    // wide as the vectors; otherwise t0 (and t1 when two samples are averaged) per Table 8-12.
    let direct = !AVG && w == W && matches!((mx, my), (0, 0) | (2, 0) | (0, 2) | (2, 2));
    let (o, os) = if direct {
        (dst.as_mut_ptr(), ds)
    } else {
        (t0, 16)
    };
    let two = unsafe {
        match (mx, my) {
            (0, 0) => {
                copy_avx2::<W>(o, os, s, ss, h);
                false
            }
            (_, 0) => {
                h_lowpass_avx2::<W>(o, os, s, ss, h);
                if mx != 2 {
                    copy_avx2::<W>(t1, 16, s.add((mx == 3) as usize), ss, h);
                }
                mx != 2
            }
            (0, _) => {
                v_lowpass_avx2::<W>(o, os, s, ss, h);
                if my != 2 {
                    copy_avx2::<W>(t1, 16, s.add((my == 3) as usize * ss), ss, h);
                }
                my != 2
            }
            (2, 2) => {
                hv_lowpass_avx2::<W>(o, os, s, ss, h);
                false
            }
            (2, _) => {
                hv_lowpass_avx2::<W>(t0, 16, s, ss, h);
                h_lowpass_avx2::<W>(t1, 16, s.add((my == 3) as usize * ss), ss, h);
                true
            }
            (_, 2) => {
                hv_lowpass_avx2::<W>(t0, 16, s, ss, h);
                v_lowpass_avx2::<W>(t1, 16, s.add((mx == 3) as usize), ss, h);
                true
            }
            _ => {
                h_lowpass_avx2::<W>(t0, 16, s.add((my == 3) as usize * ss), ss, h);
                v_lowpass_avx2::<W>(t1, 16, s.add((mx == 3) as usize), ss, h);
                true
            }
        }
    };
    if direct {
        return;
    }
    for r in 0..h {
        unsafe {
            let mut v = load::<W>(t0.add(r * 16));
            if two {
                v = _mm_avg_epu8(v, load::<W>(t1.add(r * 16)));
            }
            let d = dst.as_mut_ptr().add(r * ds);
            match w {
                16 => {
                    if AVG {
                        v = _mm_avg_epu8(v, _mm_loadu_si128(d as *const __m128i));
                    }
                    _mm_storeu_si128(d as *mut __m128i, v)
                }
                8 => {
                    if AVG {
                        v = _mm_avg_epu8(v, _mm_loadl_epi64(d as *const __m128i));
                    }
                    _mm_storel_epi64(d as *mut __m128i, v)
                }
                _ => {
                    if AVG {
                        let p = _mm_cvtsi32_si128((d as *const i32).read_unaligned());
                        v = _mm_avg_epu8(v, p);
                    }
                    (d as *mut i32).write_unaligned(_mm_cvtsi128_si32(v))
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn chroma_avx2<const AVG: bool>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    off: usize,
    ss: usize,
    w: usize,
    h: usize,
    fx: u8,
    fy: u8,
) {
    use std::arch::x86_64::*;
    let (fx, fy) = (fx as i16, fy as i16);
    let r32 = _mm_set1_epi16(32);
    let s = unsafe { src.as_ptr().add(off) };
    let mut store = |r: usize, v: __m128i| unsafe {
        let d = dst.as_mut_ptr().add(r * ds);
        match w {
            8 => {
                let v = if AVG {
                    _mm_avg_epu8(v, _mm_loadl_epi64(d as *const __m128i))
                } else {
                    v
                };
                _mm_storel_epi64(d as *mut __m128i, v)
            }
            4 => {
                let v = if AVG {
                    _mm_avg_epu8(v, _mm_cvtsi32_si128((d as *const i32).read_unaligned()))
                } else {
                    v
                };
                (d as *mut i32).write_unaligned(_mm_cvtsi128_si32(v))
            }
            _ => {
                let v = if AVG {
                    let p = (d as *const u16).read_unaligned() as i32;
                    _mm_avg_epu8(v, _mm_cvtsi32_si128(p))
                } else {
                    v
                };
                (d as *mut u16).write_unaligned(_mm_cvtsi128_si32(v) as u16)
            }
        }
    };
    let ld = |r: usize, c: usize| unsafe { _mm_loadl_epi64(s.add(r * ss + c) as *const __m128i) };
    let round = |v: __m128i| {
        let v = _mm_srli_epi16::<6>(_mm_add_epi16(v, r32));
        _mm_packus_epi16(v, v)
    };
    match (fx, fy) {
        // Whole-sample position: (64 * a + 32) >> 6 = a.
        (0, 0) => {
            for r in 0..h {
                store(r, ld(r, 0));
            }
        }
        // One-dimensional cases (8.4.2.2.2 with B = D = 0 or C = D = 0): one maddubs per row
        // over (a, b) pairs, horizontally or vertically.
        (_, 0) => {
            let ab = _mm_set1_epi16((fx << 11) | ((8 - fx) << 3));
            for r in 0..h {
                store(
                    r,
                    round(_mm_maddubs_epi16(_mm_unpacklo_epi8(ld(r, 0), ld(r, 1)), ab)),
                );
            }
        }
        (0, _) => {
            let ac = _mm_set1_epi16((fy << 11) | ((8 - fy) << 3));
            let mut top = ld(0, 0);
            for r in 0..h {
                let bot = ld(r + 1, 0);
                store(r, round(_mm_maddubs_epi16(_mm_unpacklo_epi8(top, bot), ac)));
                top = bot;
            }
        }
        _ => {
            // maddubs pairs (s[c], s[c + 1]) with weights (A, B) for the upper row and (C, D)
            // for the lower.
            let ab = _mm_set1_epi16(((fx * (8 - fy)) << 8) | ((8 - fx) * (8 - fy)));
            let cd = _mm_set1_epi16(((fx * fy) << 8) | ((8 - fx) * fy));
            let row = |r: usize| _mm_unpacklo_epi8(ld(r, 0), ld(r, 1));
            let mut top = row(0);
            for r in 0..h {
                let bot = row(r + 1);
                let v = _mm_add_epi16(_mm_maddubs_epi16(top, ab), _mm_maddubs_epi16(bot, cd));
                store(r, round(v));
                top = bot;
            }
        }
    }
}

/// `clip((a * wt[0] + b * wt[1] + round) >> shift)` over 8-sample groups; `b` absent means
/// uni-prediction. `round` folds the offset in, so the shift of the sum rounds exactly as 8.4.2.3.2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn weight_avx2(
    dst: &mut [u8],
    ds: usize,
    a: &[u8],
    b: Option<&[u8]>,
    s: usize,
    w: usize,
    h: usize,
    shift: u32,
    wt: [i32; 2],
    round: i32,
) {
    use std::arch::x86_64::*;
    // madd pairs (a, b) with (w0, w1); for uni-prediction (a, 0). 16 samples per step: whole
    // rows of 16, two rows of 8, or four rows of 4 (or 2).
    let weights = _mm256_set1_epi32((wt[1] << 16) | (wt[0] & 0xffff));
    let rnd = _mm256_set1_epi32(round);
    let sh = _mm_cvtsi32_si128(shift as i32);
    let weigh = |pa: __m128i, pb: __m128i| -> __m128i {
        let (a, b) = (_mm256_cvtepu8_epi16(pa), _mm256_cvtepu8_epi16(pb));
        let lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), weights);
        let hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), weights);
        let lo = _mm256_sra_epi32(_mm256_add_epi32(lo, rnd), sh);
        let hi = _mm256_sra_epi32(_mm256_add_epi32(hi, rnd), sh);
        let v = _mm256_packs_epi32(lo, hi);
        let v = _mm256_packus_epi16(v, v);
        _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0b1000>(v))
    };
    let pa = a.as_ptr();
    let pb = b.map(|b| b.as_ptr());
    let d = dst.as_mut_ptr();
    let rows = match w {
        16 => 1,
        8 => 2,
        _ => 4,
    };
    // Loads 16 samples of `p` from rows `r..r + n` (`n <= rows`), `16 / rows` from each.
    let gather = |p: *const u8, r: usize, n: usize| -> __m128i {
        unsafe {
            match rows {
                1 => _mm_loadu_si128(p.add(r * s) as *const __m128i),
                2 => _mm_unpacklo_epi64(
                    _mm_loadl_epi64(p.add(r * s) as *const __m128i),
                    _mm_loadl_epi64(p.add((r + 1) * s) as *const __m128i),
                ),
                _ => {
                    let row = |k: usize| {
                        if k < n {
                            (p.add((r + k) * s) as *const i32).read_unaligned()
                        } else {
                            0
                        }
                    };
                    _mm_setr_epi32(row(0), row(1), row(2), row(3))
                }
            }
        }
    };
    let mut r = 0;
    while r < h {
        let n = rows.min(h - r);
        let va = gather(pa, r, n);
        let vb = match pb {
            Some(pb) => gather(pb, r, n),
            None => _mm_setzero_si128(),
        };
        let v = weigh(va, vb);
        unsafe {
            match (w, n) {
                (16, _) => _mm_storeu_si128(d.add(r * ds) as *mut __m128i, v),
                (8, _) => {
                    _mm_storel_epi64(d.add(r * ds) as *mut __m128i, v);
                    _mm_storel_epi64(
                        d.add((r + 1) * ds) as *mut __m128i,
                        _mm_unpackhi_epi64(v, v),
                    );
                }
                _ => {
                    let lanes: [i32; 4] = std::mem::transmute(v);
                    for (k, &l) in lanes.iter().enumerate().take(n) {
                        let p = d.add((r + k) * ds);
                        if w == 4 {
                            (p as *mut i32).write_unaligned(l);
                        } else {
                            (p as *mut u16).write_unaligned(l as u16);
                        }
                    }
                }
            }
        }
        r += n;
    }
}

// ----- NEON ------------------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// Loads 8 bytes at `p` widened to `i16`.
    #[inline(always)]
    pub unsafe fn load8w(p: *const u8) -> int16x8_t {
        unsafe { vreinterpretq_s16_u16(vmovl_u8(vld1_u8(p))) }
    }

    /// `a + f - 5 (b + e) + 20 (c + d)` in `i16` lanes.
    #[inline(always)]
    pub unsafe fn tap6(
        a: int16x8_t,
        b: int16x8_t,
        c: int16x8_t,
        d: int16x8_t,
        e: int16x8_t,
        f: int16x8_t,
    ) -> int16x8_t {
        unsafe {
            let af = vaddq_s16(a, f);
            let be = vaddq_s16(b, e);
            let cd = vaddq_s16(c, d);
            vmlaq_n_s16(vmlsq_n_s16(af, be, 5), cd, 20)
        }
    }

    /// Horizontal `tap6` intermediates for 8 columns of the row at `p`.
    #[inline(always)]
    pub unsafe fn h_raw(p: *const u8) -> int16x8_t {
        unsafe {
            tap6(
                load8w(p.sub(2)),
                load8w(p.sub(1)),
                load8w(p),
                load8w(p.add(1)),
                load8w(p.add(2)),
                load8w(p.add(3)),
            )
        }
    }
}

/// NEON luma prediction, 8 columns at a time, into a 16-stride scratch block.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn luma_neon<const AVG: bool>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    off: usize,
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
) {
    use neon::*;
    use std::arch::aarch64::*;
    let s0 = unsafe { src.as_ptr().add(off) };
    let mut t0 = [0u8; 256];
    let mut t1 = [0u8; 256];
    let halves = if w == 16 { 2 } else { 1 };
    let hl = |t: &mut [u8; 256], s: *const u8, col: usize| {
        for r in 0..h {
            unsafe {
                let v = h_raw(s.add(r * ss));
                vst1_u8(t.as_mut_ptr().add(r * 16 + col), vqrshrun_n_s16::<5>(v));
            }
        }
    };
    let vl = |t: &mut [u8; 256], s: *const u8, col: usize| {
        for r in 0..h {
            unsafe {
                let p = s.add(r * ss);
                let v = tap6(
                    load8w(p.sub(2 * ss)),
                    load8w(p.sub(ss)),
                    load8w(p),
                    load8w(p.add(ss)),
                    load8w(p.add(2 * ss)),
                    load8w(p.add(3 * ss)),
                );
                vst1_u8(t.as_mut_ptr().add(r * 16 + col), vqrshrun_n_s16::<5>(v));
            }
        }
    };
    let hv = |t: &mut [u8; 256], s: *const u8, col: usize| {
        let mut mid = [vdupq_n_s16(0); 21];
        for (k, m) in mid.iter_mut().enumerate().take(h + 5) {
            *m = unsafe { h_raw(s.add(k * ss).sub(2 * ss)) };
        }
        for r in 0..h {
            let af = vaddq_s16(mid[r], mid[r + 5]);
            let be = vaddq_s16(mid[r + 1], mid[r + 4]);
            let cd = vaddq_s16(mid[r + 2], mid[r + 3]);
            let lo = vmlal_n_s16(
                vmlsl_n_s16(vmovl_s16(vget_low_s16(af)), vget_low_s16(be), 5),
                vget_low_s16(cd),
                20,
            );
            let hi = vmlal_n_s16(
                vmlsl_n_s16(vmovl_s16(vget_high_s16(af)), vget_high_s16(be), 5),
                vget_high_s16(cd),
                20,
            );
            // (x + 512) >> 10 saturated to i16, then to u8.
            let v = vcombine_s16(vqrshrn_n_s32::<10>(lo), vqrshrn_n_s32::<10>(hi));
            unsafe { vst1_u8(t.as_mut_ptr().add(r * 16 + col), vqmovun_s16(v)) };
        }
    };
    let cp = |t: &mut [u8; 256], s: *const u8, col: usize| {
        for r in 0..h {
            unsafe { vst1_u8(t.as_mut_ptr().add(r * 16 + col), vld1_u8(s.add(r * ss))) };
        }
    };
    let mut two = false;
    for half in 0..halves {
        let col = half * 8;
        let s = unsafe { s0.add(col) };
        let down = |b: bool| unsafe { s.add(b as usize * ss) };
        let right = |b: bool| unsafe { s.add(b as usize) };
        two = match (mx, my) {
            (0, 0) => {
                cp(&mut t0, s, col);
                false
            }
            (_, 0) => {
                hl(&mut t0, s, col);
                if mx != 2 {
                    cp(&mut t1, right(mx == 3), col);
                }
                mx != 2
            }
            (0, _) => {
                vl(&mut t0, s, col);
                if my != 2 {
                    cp(&mut t1, down(my == 3), col);
                }
                my != 2
            }
            (2, 2) => {
                hv(&mut t0, s, col);
                false
            }
            (2, _) => {
                hv(&mut t0, s, col);
                hl(&mut t1, down(my == 3), col);
                true
            }
            (_, 2) => {
                hv(&mut t0, s, col);
                vl(&mut t1, right(mx == 3), col);
                true
            }
            _ => {
                hl(&mut t0, down(my == 3), col);
                vl(&mut t1, right(mx == 3), col);
                true
            }
        };
    }
    for r in 0..h {
        unsafe {
            let d = dst.as_mut_ptr().add(r * ds);
            let a = t0.as_ptr().add(r * 16);
            let b = t1.as_ptr().add(r * 16);
            match w {
                16 => {
                    let mut v = vld1q_u8(a);
                    if two {
                        v = vrhaddq_u8(v, vld1q_u8(b));
                    }
                    if AVG {
                        v = vrhaddq_u8(v, vld1q_u8(d));
                    }
                    vst1q_u8(d, v);
                }
                8 => {
                    let mut v = vld1_u8(a);
                    if two {
                        v = vrhadd_u8(v, vld1_u8(b));
                    }
                    if AVG {
                        v = vrhadd_u8(v, vld1_u8(d));
                    }
                    vst1_u8(d, v);
                }
                _ => {
                    let mut v = vld1_u8(a);
                    if two {
                        v = vrhadd_u8(v, vld1_u8(b));
                    }
                    if AVG {
                        let p = vdup_n_u32((d as *const u32).read_unaligned());
                        v = vrhadd_u8(v, vreinterpret_u8_u32(p));
                    }
                    (d as *mut u32).write_unaligned(vget_lane_u32::<0>(vreinterpret_u32_u8(v)));
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn chroma_neon<const AVG: bool>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    off: usize,
    ss: usize,
    w: usize,
    h: usize,
    fx: u8,
    fy: u8,
) {
    use std::arch::aarch64::*;
    let wa = vdup_n_u8((8 - fx) * (8 - fy));
    let wb = vdup_n_u8(fx * (8 - fy));
    let wc = vdup_n_u8((8 - fx) * fy);
    let wd = vdup_n_u8(fx * fy);
    let s = unsafe { src.as_ptr().add(off) };
    for r in 0..h {
        unsafe {
            let p = s.add(r * ss);
            let mut acc = vmull_u8(vld1_u8(p), wa);
            acc = vmlal_u8(acc, vld1_u8(p.add(1)), wb);
            acc = vmlal_u8(acc, vld1_u8(p.add(ss)), wc);
            acc = vmlal_u8(acc, vld1_u8(p.add(ss + 1)), wd);
            let mut v = vrshrn_n_u16::<6>(acc);
            let d = dst.as_mut_ptr().add(r * ds);
            match w {
                8 => {
                    if AVG {
                        v = vrhadd_u8(v, vld1_u8(d));
                    }
                    vst1_u8(d, v)
                }
                4 => {
                    if AVG {
                        let p = vdup_n_u32((d as *const u32).read_unaligned());
                        v = vrhadd_u8(v, vreinterpret_u8_u32(p));
                    }
                    (d as *mut u32).write_unaligned(vget_lane_u32::<0>(vreinterpret_u32_u8(v)))
                }
                _ => {
                    if AVG {
                        let p = vdup_n_u16((d as *const u16).read_unaligned());
                        v = vrhadd_u8(v, vreinterpret_u8_u16(p));
                    }
                    (d as *mut u16).write_unaligned(vget_lane_u16::<0>(vreinterpret_u16_u8(v)))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic noise with runs of extremes, so the filters reach both clip bounds.
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

    #[test]
    fn luma_mc_matches_scalar_for_every_size_and_phase() {
        let (pw, ph) = (61, 47);
        let src = plane(pw, ph, 7);
        for &(w, h) in &[(16, 16), (16, 8), (8, 16), (8, 8), (8, 4), (4, 8), (4, 4)] {
            for mx in 0..4 {
                for my in 0..4 {
                    for &(x, y) in &[(2, 2), (5, 3), (pw - luma_mc_cols(w), ph - h - 3)] {
                        let off = y * pw + x;
                        let ds = 19;
                        let mut a = vec![0xAAu8; ds * 16];
                        let mut b = a.clone();
                        h264_luma_mc(&mut a, ds, &src, off, pw, w, h, mx, my);
                        h264_luma_mc_scalar(&mut b, ds, &src, off, pw, w, h, mx, my);
                        assert_eq!(a, b, "{w}x{h} phase ({mx}, {my}) at ({x}, {y})");
                        // Averaged into an existing prediction.
                        let mut a = plane(ds, 16, (x * 7 + y) as u32);
                        let mut b = a.clone();
                        h264_luma_mc_avg(&mut a, ds, &src, off, pw, w, h, mx, my);
                        h264_luma_mc_avg_scalar(&mut b, ds, &src, off, pw, w, h, mx, my);
                        assert_eq!(a, b, "avg {w}x{h} phase ({mx}, {my}) at ({x}, {y})");
                    }
                }
            }
        }
    }

    #[test]
    fn luma_mc_scalar_matches_the_16x16_reference() {
        let (pw, ph) = (40, 40);
        let src = plane(pw, ph, 3);
        for mx in 0..4 {
            for my in 0..4 {
                let (x, y) = (5, 6);
                let mut a = [0u8; 256];
                h264_luma_mc_scalar(&mut a, 16, &src, y * pw + x, pw, 16, 16, mx, my);
                let mut b = [0u8; 256];
                crate::put_h264_qpel16_mc_u8_scalar(
                    &src, pw, pw, ph, x as isize, y as isize, mx, my, &mut b,
                );
                assert_eq!(a, b, "phase ({mx}, {my})");
            }
        }
    }

    #[test]
    fn chroma_mc_matches_scalar_for_every_size_and_phase() {
        let (pw, ph) = (37, 29);
        let src = plane(pw, ph, 11);
        for &(w, h) in &[(8, 8), (8, 4), (4, 8), (4, 4), (4, 2), (2, 4), (2, 2)] {
            for fx in 0..8 {
                for fy in 0..8 {
                    for &(x, y) in &[(0, 0), (3, 5), (pw - chroma_mc_cols(w), ph - h - 1)] {
                        let off = y * pw + x;
                        let ds = 11;
                        let mut a = vec![0x55u8; ds * 8];
                        let mut b = a.clone();
                        h264_chroma_mc(&mut a, ds, &src, off, pw, w, h, fx, fy);
                        h264_chroma_mc_scalar(&mut b, ds, &src, off, pw, w, h, fx, fy);
                        assert_eq!(a, b, "{w}x{h} phase ({fx}, {fy}) at ({x}, {y})");
                        let mut a = plane(ds, 8, (x * 5 + y) as u32);
                        let mut b = a.clone();
                        h264_chroma_mc_avg(&mut a, ds, &src, off, pw, w, h, fx, fy);
                        h264_chroma_mc_avg_scalar(&mut b, ds, &src, off, pw, w, h, fx, fy);
                        assert_eq!(a, b, "avg {w}x{h} phase ({fx}, {fy}) at ({x}, {y})");
                    }
                }
            }
        }
    }

    #[test]
    fn in_plane_entries_match_offset_entries() {
        let (pw, ph) = (48, 40);
        let src = plane(pw, ph, 29);
        let mvs = [
            [0, 0],
            [5, -3],
            [-9, 14],
            [-27, -30],
            [60, 50],
            [3, 7],
            [-1, -1],
        ];
        for &(w, h) in &[(16, 16), (8, 8), (8, 4), (4, 8), (4, 4)] {
            for &(x, y) in &[(0, 0), (8, 4), (16, 16), (pw - w, ph - h)] {
                for mv in mvs {
                    for avg in [false, true] {
                        let ds = 17;
                        let mut a = plane(ds, 16, 3);
                        let mut b = a.clone();
                        let ok = h264_luma_mc_in_plane(
                            &mut a,
                            ds,
                            &src,
                            pw,
                            ph,
                            (x, y),
                            (w, h),
                            mv,
                            avg,
                        );
                        let ix = x as isize + (mv[0] as isize >> 2);
                        let iy = y as isize + (mv[1] as isize >> 2);
                        let inside = ix >= 2
                            && iy >= 2
                            && ix as usize + luma_mc_cols(w) <= pw
                            && iy as usize + h + 3 <= ph;
                        assert_eq!(ok, inside, "luma {w}x{h} at ({x}, {y}) mv {mv:?}");
                        if inside {
                            let off = iy as usize * pw + ix as usize;
                            let (mx, my) = ((mv[0] & 3) as u8, (mv[1] & 3) as u8);
                            if avg {
                                h264_luma_mc_avg_scalar(&mut b, ds, &src, off, pw, w, h, mx, my);
                            } else {
                                h264_luma_mc_scalar(&mut b, ds, &src, off, pw, w, h, mx, my);
                            }
                        }
                        assert_eq!(a, b, "luma {w}x{h} at ({x}, {y}) mv {mv:?} avg {avg}");
                        let (cw, ch) = (w / 2, h / 2);
                        let (cx, cy) = (x / 2, y / 2);
                        let (cpw, cph) = (pw / 2, ph / 2);
                        let csrc = &src[..cpw * cph];
                        let mut a = plane(ds, 8, 4);
                        let mut b = a.clone();
                        let ok = h264_chroma_mc_in_plane(
                            &mut a,
                            ds,
                            csrc,
                            cpw,
                            cph,
                            (cx, cy),
                            (cw, ch),
                            mv,
                            avg,
                        );
                        let ix = cx as isize + (mv[0] as isize >> 3);
                        let iy = cy as isize + (mv[1] as isize >> 3);
                        let inside = ix >= 0
                            && iy >= 0
                            && ix as usize + chroma_mc_cols(cw) <= cpw
                            && (iy as usize + ch) < cph;
                        assert_eq!(ok, inside, "chroma {cw}x{ch} at ({cx}, {cy}) mv {mv:?}");
                        if inside {
                            let off = iy as usize * cpw + ix as usize;
                            let (fx, fy) = ((mv[0] & 7) as u8, (mv[1] & 7) as u8);
                            if avg {
                                h264_chroma_mc_avg_scalar(
                                    &mut b, ds, csrc, off, cpw, cw, ch, fx, fy,
                                );
                            } else {
                                h264_chroma_mc_scalar(&mut b, ds, csrc, off, cpw, cw, ch, fx, fy);
                            }
                        }
                        assert_eq!(a, b, "chroma {cw}x{ch} at ({cx}, {cy}) mv {mv:?} avg {avg}");
                    }
                }
            }
        }
    }

    #[test]
    fn avg_matches_scalar() {
        let a = plane(16, 16, 21);
        let b = plane(16, 16, 22);
        for &(w, h) in &[
            (16, 16),
            (16, 8),
            (8, 8),
            (8, 4),
            (4, 4),
            (4, 2),
            (2, 2),
            (2, 4),
        ] {
            let mut x = a.clone();
            let mut y = a.clone();
            h264_avg(&mut x, 16, &b, 16, w, h);
            h264_avg_scalar(&mut y, 16, &b, 16, w, h);
            assert_eq!(x, y, "{w}x{h}");
        }
    }

    #[test]
    fn weighting_matches_scalar() {
        let a = plane(16, 16, 5);
        let b = plane(16, 16, 9);
        let cases = [
            (5, 32, 32, 0),
            (5, 128, -64, 0),
            (0, 1, 1, 0),
            (7, -128, 127, -128),
            (6, 80, -20, 127),
            (1, 3, 5, -7),
        ];
        for &(log_wd, w0, w1, off) in &cases {
            for &(w, h) in &[
                (16, 16),
                (8, 4),
                (4, 8),
                (2, 2),
                (16, 8),
                (8, 16),
                (8, 2),
                (4, 2),
                (2, 4),
                (4, 4),
                (2, 8),
            ] {
                let mut x = vec![0x11u8; 21 * 16];
                let mut y = x.clone();
                h264_weight_bi(&mut x, 21, &a, &b, 16, w, h, log_wd, w0, w1, off);
                h264_weight_bi_scalar(&mut y, 21, &a, &b, 16, w, h, log_wd, w0, w1, off);
                assert_eq!(x, y, "bi {w}x{h} {log_wd} {w0} {w1} {off}");
                h264_weight_uni(&mut x, 21, &a, 16, w, h, log_wd, w0, off);
                h264_weight_uni_scalar(&mut y, 21, &a, 16, w, h, log_wd, w0, off);
                assert_eq!(x, y, "uni {w}x{h} {log_wd} {w0} {off}");
            }
        }
    }

    #[test]
    #[should_panic]
    fn luma_mc_rejects_a_source_without_margins() {
        let src = [0u8; 64];
        let mut dst = [0u8; 16];
        h264_luma_mc(&mut dst, 4, &src, 0, 8, 4, 4, 1, 1);
    }
}
