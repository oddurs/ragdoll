//! H.264 encoder pixel kernels (`SIMD-0089`): block matching costs for motion search and mode
//! decision (`x264` `pixel.c`).
//!
//! - [`pixel_sad_16x16`], [`pixel_sad_8x8`]: sums of absolute differences between strided blocks.
//! - [`pixel_sad_x4_16x16`]: one source macroblock against four candidate positions of a reference
//!   plane, loading the source once (`x264_pixel_sad_x4_16x16`).
//! - [`pixel_satd_16x16`]: sum of absolute $4\times4$ Hadamard-transformed differences, halved.
//!
//! Every kernel has a `_scalar` reference that the AVX2 and NEON paths match bit for bit.

#[inline(always)]
fn check_block(buf: &[u8], stride: usize, w: usize, h: usize) {
    assert!((h - 1) * stride + w <= buf.len());
}

/// SAD of the $16\times16$ blocks at the start of `a` (stride `sa`) and `b` (stride `sb`).
///
/// # Panics
/// If either buffer is shorter than 16 rows of its stride.
#[inline]
pub fn pixel_sad_16x16(a: &[u8], sa: usize, b: &[u8], sb: usize) -> u32 {
    check_block(a, sa, 16, 16);
    check_block(b, sb, 16, 16);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above; NEON is baseline on aarch64.
        return unsafe { sad_neon::<16, 16>(a.as_ptr(), sa, b.as_ptr(), sb) };
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        return unsafe { sad16_avx2(a.as_ptr(), sa, b.as_ptr(), sb) };
    }
    pixel_sad_16x16_scalar(a, sa, b, sb)
}

/// Scalar reference for [`pixel_sad_16x16`].
pub fn pixel_sad_16x16_scalar(a: &[u8], sa: usize, b: &[u8], sb: usize) -> u32 {
    sad_scalar(a, sa, b, sb, 16, 16)
}

/// SAD of the $8\times8$ blocks at the start of `a` (stride `sa`) and `b` (stride `sb`).
///
/// # Panics
/// If either buffer is shorter than 8 rows of its stride.
#[inline]
pub fn pixel_sad_8x8(a: &[u8], sa: usize, b: &[u8], sb: usize) -> u32 {
    check_block(a, sa, 8, 8);
    check_block(b, sb, 8, 8);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above; NEON is baseline on aarch64.
        return unsafe { sad_neon::<8, 8>(a.as_ptr(), sa, b.as_ptr(), sb) };
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        return unsafe { sad8_avx2(a.as_ptr(), sa, b.as_ptr(), sb) };
    }
    pixel_sad_8x8_scalar(a, sa, b, sb)
}

/// Scalar reference for [`pixel_sad_8x8`].
pub fn pixel_sad_8x8_scalar(a: &[u8], sa: usize, b: &[u8], sb: usize) -> u32 {
    sad_scalar(a, sa, b, sb, 8, 8)
}

fn sad_scalar(a: &[u8], sa: usize, b: &[u8], sb: usize, w: usize, h: usize) -> u32 {
    let mut sum = 0u32;
    for y in 0..h {
        let (ra, rb) = (&a[y * sa..y * sa + w], &b[y * sb..y * sb + w]);
        sum += ra
            .iter()
            .zip(rb)
            .map(|(&p, &q)| p.abs_diff(q) as u32)
            .sum::<u32>();
    }
    sum
}

/// Sum of squared differences of the `N`x`N` blocks (`N` = 8 or 16) at the start of `a` (stride
/// `sa`) and `b` (stride `sb`).
///
/// # Panics
/// If `N` is not 8 or 16, or either buffer is shorter than `N` rows of its stride.
#[inline]
pub fn pixel_ssd<const N: usize>(a: &[u8], sa: usize, b: &[u8], sb: usize) -> u32 {
    assert!(N == 8 || N == 16);
    check_block(a, sa, N, N);
    check_block(b, sb, N, N);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above; NEON is baseline on aarch64.
        return unsafe { ssd_neon::<N>(a.as_ptr(), sa, b.as_ptr(), sb) };
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        return unsafe { ssd_avx2::<N>(a.as_ptr(), sa, b.as_ptr(), sb) };
    }
    pixel_ssd_scalar::<N>(a, sa, b, sb)
}

/// Scalar reference for [`pixel_ssd`].
pub fn pixel_ssd_scalar<const N: usize>(a: &[u8], sa: usize, b: &[u8], sb: usize) -> u32 {
    let mut sum = 0u32;
    for y in 0..N {
        for x in 0..N {
            let d = a[y * sa + x] as i32 - b[y * sb + x] as i32;
            sum += (d * d) as u32;
        }
    }
    sum
}

/// SADs of the contiguous $16\times16$ block `a` against the four $16\times16$ blocks of `b`
/// (stride `sb`) starting at `offs`.
///
/// # Panics
/// If a block at one of `offs` reaches past the end of `b`.
#[inline]
pub fn pixel_sad_x4_16x16(a: &[u8; 256], b: &[u8], offs: [usize; 4], sb: usize) -> [u32; 4] {
    for &o in &offs {
        assert!(o + 15 * sb + 16 <= b.len());
    }
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above; NEON is baseline on aarch64.
        return offs.map(|o| unsafe { sad_neon::<16, 16>(a.as_ptr(), 16, b.as_ptr().add(o), sb) });
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        return unsafe { sad_x4_avx2(a, b.as_ptr(), offs, sb) };
    }
    pixel_sad_x4_16x16_scalar(a, b, offs, sb)
}

/// Scalar reference for [`pixel_sad_x4_16x16`].
pub fn pixel_sad_x4_16x16_scalar(a: &[u8; 256], b: &[u8], offs: [usize; 4], sb: usize) -> [u32; 4] {
    offs.map(|o| sad_scalar(a, 16, &b[o..], sb, 16, 16))
}

/// Sum over the sixteen $4\times4$ blocks of `|H (a - b) H^T|`, halved, where `H` is the $4\times4$
/// Hadamard matrix (`x264_pixel_satd_16x16`).
///
/// # Panics
/// If either buffer is shorter than 16 rows of its stride.
#[inline]
pub fn pixel_satd_16x16(a: &[u8], sa: usize, b: &[u8], sb: usize) -> u32 {
    check_block(a, sa, 16, 16);
    check_block(b, sb, 16, 16);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above; NEON is baseline on aarch64.
        return unsafe { satd16_neon(a.as_ptr(), sa, b.as_ptr(), sb) };
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        return unsafe { satd16_avx2(a.as_ptr(), sa, b.as_ptr(), sb) };
    }
    pixel_satd_16x16_scalar(a, sa, b, sb)
}

/// Scalar reference for [`pixel_satd_16x16`].
pub fn pixel_satd_16x16_scalar(a: &[u8], sa: usize, b: &[u8], sb: usize) -> u32 {
    let mut sum = 0u32;
    for by in 0..4 {
        for bx in 0..4 {
            let mut d = [[0i32; 4]; 4];
            for (y, row) in d.iter_mut().enumerate() {
                for (x, v) in row.iter_mut().enumerate() {
                    let (r, c) = (by * 4 + y, bx * 4 + x);
                    *v = a[r * sa + c] as i32 - b[r * sb + c] as i32;
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

/// SATD of the $4\times4$ blocks at the start of `a` (stride `sa`) and `b` (stride `sb`), with
/// the halving of [`pixel_satd_16x16`] (`x264_pixel_satd_4x4`).
///
/// # Panics
/// If either buffer is shorter than 4 rows of its stride.
#[inline]
pub fn pixel_satd_4x4(a: &[u8], sa: usize, b: &[u8], sb: usize) -> u32 {
    check_block(a, sa, 4, 4);
    check_block(b, sb, 4, 4);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        return crate::satd_4x4_u8(a, sa, b, sb);
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        return unsafe { satd4_avx2(a.as_ptr(), sa, b.as_ptr(), sb) };
    }
    pixel_satd_4x4_scalar(a, sa, b, sb)
}

/// Scalar reference for [`pixel_satd_4x4`].
pub fn pixel_satd_4x4_scalar(a: &[u8], sa: usize, b: &[u8], sb: usize) -> u32 {
    crate::satd_4x4_u8_scalar(a, sa, b, sb)
}

/// Forward $4\times4$ H.264 core transforms of the residual `src - pred` for the four $4\times4$
/// blocks of a $16\times4$ strip, left to right (`x264_sub16x4_dct` order).
///
/// # Panics
/// If either buffer is shorter than 4 rows of 16 samples at its stride.
#[inline]
pub fn sub_dct4x4_16x4(src: &[u8], ss: usize, pred: &[u8], ps: usize, out: &mut [[i16; 16]; 4]) {
    check_block(src, ss, 16, 4);
    check_block(pred, ps, 16, 4);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above; NEON is baseline on aarch64.
        unsafe {
            dct_strip_neon::<false>(src.as_ptr(), ss, pred.as_ptr(), ps, 0, &mut out[..2]);
            dct_strip_neon::<false>(
                src.as_ptr().add(8),
                ss,
                pred.as_ptr().add(8),
                ps,
                0,
                &mut out[2..],
            );
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        unsafe { dct_strip_avx2::<false>(src.as_ptr(), ss, pred.as_ptr(), ps, false, out) };
        return;
    }
    sub_dct4x4_16x4_scalar(src, ss, pred, ps, out);
}

/// Scalar reference for [`sub_dct4x4_16x4`].
pub fn sub_dct4x4_16x4_scalar(
    src: &[u8],
    ss: usize,
    pred: &[u8],
    ps: usize,
    out: &mut [[i16; 16]; 4],
) {
    for (b, o) in out.iter_mut().enumerate() {
        *o = sub_dct4x4_scalar(&src[b * 4..], ss, &pred[b * 4..], ps);
    }
}

/// Forward $4\times4$ transforms of the residual of an $8\times8$ block, as its four $4\times4$
/// blocks in raster order (`x264_sub8x8_dct`).
///
/// # Panics
/// If either buffer is shorter than 8 rows of 8 samples at its stride.
#[inline]
pub fn sub_dct4x4_8x8(src: &[u8], ss: usize, pred: &[u8], ps: usize, out: &mut [[i16; 16]; 4]) {
    check_block(src, ss, 8, 8);
    check_block(pred, ps, 8, 8);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above; NEON is baseline on aarch64.
        unsafe {
            dct_strip_neon::<false>(src.as_ptr(), ss, pred.as_ptr(), ps, 0, &mut out[..2]);
            dct_strip_neon::<false>(src.as_ptr(), ss, pred.as_ptr(), ps, 4, &mut out[2..]);
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        unsafe { dct_strip_avx2::<false>(src.as_ptr(), ss, pred.as_ptr(), ps, true, out) };
        return;
    }
    sub_dct4x4_8x8_scalar(src, ss, pred, ps, out);
}

/// Scalar reference for [`sub_dct4x4_8x8`].
pub fn sub_dct4x4_8x8_scalar(
    src: &[u8],
    ss: usize,
    pred: &[u8],
    ps: usize,
    out: &mut [[i16; 16]; 4],
) {
    for (b, o) in out.iter_mut().enumerate() {
        let (so, po) = (
            (b >> 1) * 4 * ss + (b & 1) * 4,
            (b >> 1) * 4 * ps + (b & 1) * 4,
        );
        *o = sub_dct4x4_scalar(&src[so..], ss, &pred[po..], ps);
    }
}

fn sub_dct4x4_scalar(src: &[u8], ss: usize, pred: &[u8], ps: usize) -> [i16; 16] {
    let mut d = [0i16; 16];
    for y in 0..4 {
        for x in 0..4 {
            d[y * 4 + x] = src[y * ss + x] as i16 - pred[y * ps + x] as i16;
        }
    }
    crate::forward_dct4x4_scalar(&d)
}

/// Unnormalized $4\times4$ Hadamard transforms of the four $4\times4$ blocks of the $16\times4$
/// strip at the start of `src` (stride `ss`), left to right, as `out[b][4 v + h]` for vertical and
/// horizontal sequency `v`, `h`. The 1-D transform of `x` is `[x0 + x1 + x2 + x3,
/// x0 + x1 - x2 - x3, x0 - x1 + x2 - x3, x0 - x1 - x2 + x3]`, so a block's SATD against a
/// prediction is half the sum of `|out - transform(pred)|` (`x264_intra_satd_x3`).
///
/// # Panics
/// If `src` is shorter than 4 rows of 16 samples at its stride.
#[inline]
pub fn hadamard4x4_16x4(src: &[u8], ss: usize, out: &mut [[i16; 16]; 4]) {
    static ZERO: [u8; 16] = [0; 16];
    check_block(src, ss, 16, 4);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above (the zero row is read at stride 0); NEON is baseline.
        unsafe {
            dct_strip_neon::<true>(src.as_ptr(), ss, ZERO.as_ptr(), 0, 0, &mut out[..2]);
            dct_strip_neon::<true>(src.as_ptr().add(8), ss, ZERO.as_ptr(), 0, 0, &mut out[2..]);
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above (the zero row is read at stride 0); avx2() is true only
        // when the CPU supports AVX2.
        unsafe { dct_strip_avx2::<true>(src.as_ptr(), ss, ZERO.as_ptr(), 0, false, out) };
        return;
    }
    hadamard4x4_16x4_scalar(src, ss, out);
}

/// Scalar reference for [`hadamard4x4_16x4`].
pub fn hadamard4x4_16x4_scalar(src: &[u8], ss: usize, out: &mut [[i16; 16]; 4]) {
    let h1 = |x: [i32; 4]| {
        let (s01, d01, s23, d23) = (x[0] + x[1], x[0] - x[1], x[2] + x[3], x[2] - x[3]);
        [s01 + s23, s01 - s23, d01 + d23, d01 - d23]
    };
    for (b, o) in out.iter_mut().enumerate() {
        let rows: [[i32; 4]; 4] =
            std::array::from_fn(|y| h1(std::array::from_fn(|x| src[y * ss + b * 4 + x] as i32)));
        for h in 0..4 {
            let col = h1(std::array::from_fn(|y| rows[y][h]));
            for v in 0..4 {
                o[4 * v + h] = col[v] as i16;
            }
        }
    }
}

/// Quantization of a $4\times4$ block at one QP: `level = sign(c) ((|c| mf[i] + bias) >> shift)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quant4x4 {
    /// Multiplier per coefficient position (raster order).
    pub mf: [u16; 16],
    /// Rounding offset added before the shift; below `1 << shift`.
    pub bias: u32,
    /// Right shift, `15..=23` for H.264.
    pub shift: u32,
}

/// Quantizes the four $4\times4$ blocks of `blocks` in place; bit `b` of the result is set when
/// block `b` has a non-zero level, and bit `4 + b` when it has a level of magnitude above 1.
///
/// # Panics
/// If `q.shift` is not in `1..=31` or `q.bias` is not below `1 << q.shift`.
#[inline(always)]
pub fn quant_4x4_x4(blocks: &mut [[i16; 16]; 4], q: &Quant4x4) -> u32 {
    assert!((1..32).contains(&q.shift) && q.bias < 1 << q.shift);
    // Every multiplier fits in 15 bits.
    assert!(q.mf.iter().fold(0, |a, &m| a | m) <= i16::MAX as u16);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: NEON is baseline on aarch64.
        return unsafe { quant_neon(blocks, q) };
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        return unsafe { quant_avx2(blocks, q) };
    }
    quant_4x4_x4_scalar(blocks, q)
}

/// Scalar reference for [`quant_4x4_x4`].
#[inline(never)]
pub fn quant_4x4_x4_scalar(blocks: &mut [[i16; 16]; 4], q: &Quant4x4) -> u32 {
    let mut nz = 0;
    for (b, blk) in blocks.iter_mut().enumerate() {
        for (c, &mf) in blk.iter_mut().zip(&q.mf) {
            let level = ((c.unsigned_abs() as u64 * mf as u64 + q.bias as u64) >> q.shift)
                .min(i16::MAX as u64) as i16;
            *c = if *c < 0 { -level } else { level };
        }
        if blk.iter().any(|&c| c != 0) {
            nz |= 1 << b;
        }
        if blk.iter().any(|&c| c.unsigned_abs() > 1) {
            nz |= 0x10 << b;
        }
    }
    nz
}

/// Inverse quantization of four $4\times4$ blocks in place: `c = clamp_i16(c * scale[i])`, where
/// `scale` already includes the `<< (qp / 6)` of ITU-T H.264 8.5.12.1.
///
/// # Panics
/// If a scale exceeds 65535 in magnitude.
#[inline(always)]
pub fn dequant_4x4_x4(blocks: &mut [[i16; 16]; 4], scale: &[i32; 16]) {
    check_scale(scale);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: NEON is baseline on aarch64.
        unsafe { dequant_neon(blocks, scale) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        unsafe { dequant_avx2(blocks, scale) };
        return;
    }
    dequant_4x4_x4_scalar(blocks, scale);
}

#[inline(always)]
fn check_scale(scale: &[i32; 16]) {
    // Keeps |c * scale| below 2^31 for every 16-bit level.
    assert!(scale.iter().all(|s| s.unsigned_abs() <= u16::MAX as u32));
}

/// Scalar reference for [`dequant_4x4_x4`].
#[inline(never)]
pub fn dequant_4x4_x4_scalar(blocks: &mut [[i16; 16]; 4], scale: &[i32; 16]) {
    check_scale(scale);
    for blk in blocks.iter_mut() {
        for (c, &s) in blk.iter_mut().zip(scale) {
            *c = (*c as i64 * s as i64).clamp(i16::MIN as i64, i16::MAX as i64) as i16;
        }
    }
}

/// Adds the H.264 inverse $4\times4$ transforms of four blocks, laid out as the four quadrants
/// of an $8\times8$ region in raster order, onto `dst`: `dst = clamp_u8(dst + ((idct(c) + 32) >> 6))`.
///
/// # Panics
/// If `dst` is shorter than 8 rows of 8 samples at stride `ds`.
#[inline]
pub fn add_idct4x4_8x8(dst: &mut [u8], ds: usize, blocks: &[[i16; 16]; 4]) {
    check_block(dst, ds, 8, 8);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above; NEON is baseline on aarch64.
        unsafe {
            idct_add_pair_neon(dst.as_mut_ptr(), ds, &blocks[0], &blocks[1]);
            idct_add_pair_neon(dst.as_mut_ptr().add(4 * ds), ds, &blocks[2], &blocks[3]);
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        unsafe { idct_add_8x8_avx2(dst.as_mut_ptr(), ds, blocks) };
        return;
    }
    add_idct4x4_8x8_scalar(dst, ds, blocks);
}

/// Scalar reference for [`add_idct4x4_8x8`].
pub fn add_idct4x4_8x8_scalar(dst: &mut [u8], ds: usize, blocks: &[[i16; 16]; 4]) {
    for (b, c) in blocks.iter().enumerate() {
        let res = crate::inverse_idct4x4_scalar(c);
        let o = (b >> 1) * 4 * ds + (b & 1) * 4;
        for y in 0..4 {
            for x in 0..4 {
                let p = &mut dst[o + y * ds + x];
                *p = (*p as i32 + res[y * 4 + x] as i32).clamp(0, 255) as u8;
            }
        }
    }
}

/// The 1-D transform of [`hadamard4x4_16x4`] of `x`, times 4: the non-zero row (column) of the
/// transform of a $4\times4$ block whose rows (columns) all equal `x`.
#[inline(always)]
fn hadamard_edge(x: &[u8]) -> [i16; 4] {
    let x = [x[0] as i16, x[1] as i16, x[2] as i16, x[3] as i16];
    let (s01, d01, s23, d23) = (x[0] + x[1], x[0] - x[1], x[2] + x[3], x[2] - x[3]);
    [s01 + s23, s01 - s23, d01 + d23, d01 - d23].map(|c| 4 * c)
}

/// SATDs, as [`pixel_satd_16x16`], of the $16\times16$ block `src` against its vertical,
/// horizontal and DC intra predictions from the `top` and `left` edge samples and the DC value
/// `dc`, in that order (`x264_intra_satd_x3_16x16`). One Hadamard transform of the source serves
/// all three: a prediction with equal rows (columns) transforms to a single non-zero row
/// (column), and the DC prediction to a single coefficient.
#[inline(always)]
pub fn intra_satd_x3_16x16(src: &[u8; 256], top: &[u8; 16], left: &[u8; 16], dc: u8) -> [u32; 3] {
    let tv: [[i16; 4]; 4] = std::array::from_fn(|b| hadamard_edge(&top[b * 4..]));
    let th: [[i16; 4]; 4] = std::array::from_fn(|b| hadamard_edge(&left[b * 4..]));
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        return unsafe { intra_satd_x3_avx2(src, &tv, &th, 16 * dc as i16) };
    }
    intra_satd_x3_strips(src, &tv, &th, 16 * dc as i16)
}

/// [`intra_satd_x3_16x16`] through the dispatched $4\times4$ Hadamard of each strip.
#[inline(never)]
fn intra_satd_x3_strips(
    src: &[u8; 256],
    tv: &[[i16; 4]; 4],
    th: &[[i16; 4]; 4],
    dc16: i16,
) -> [u32; 3] {
    let mut out = [[[0i16; 16]; 4]; 4];
    for (strip, o) in out.iter_mut().enumerate() {
        hadamard4x4_16x4(&src[strip * 64..], 16, o);
    }
    intra_satd_x3_from(&out, tv, th, dc16)
}

/// Scalar reference for [`intra_satd_x3_16x16`].
pub fn intra_satd_x3_16x16_scalar(
    src: &[u8; 256],
    top: &[u8; 16],
    left: &[u8; 16],
    dc: u8,
) -> [u32; 3] {
    let tv: [[i16; 4]; 4] = std::array::from_fn(|b| hadamard_edge(&top[b * 4..]));
    let th: [[i16; 4]; 4] = std::array::from_fn(|b| hadamard_edge(&left[b * 4..]));
    let mut out = [[[0i16; 16]; 4]; 4];
    for (strip, o) in out.iter_mut().enumerate() {
        hadamard4x4_16x4_scalar(&src[strip * 64..], 16, o);
    }
    intra_satd_x3_from(&out, &tv, &th, 16 * dc as i16)
}

/// [`intra_satd_x3_16x16`] from the source's transforms (`out[strip][block]`), the transformed
/// edges per block column (`tv`) and row (`th`) and the transformed DC prediction.
fn intra_satd_x3_from(
    out: &[[[i16; 16]; 4]; 4],
    tv: &[[i16; 4]; 4],
    th: &[[i16; 4]; 4],
    dc16: i16,
) -> [u32; 3] {
    let mut sum = [0u32; 3];
    for (by, strip) in out.iter().enumerate() {
        for (bx, blk) in strip.iter().enumerate() {
            for (k, &c) in blk.iter().enumerate() {
                let (v, h) = (k / 4, k % 4);
                let c = c as i32;
                let pv = if v == 0 { tv[bx][h] as i32 } else { 0 };
                let ph = if h == 0 { th[by][v] as i32 } else { 0 };
                let pd = if k == 0 { dc16 as i32 } else { 0 };
                sum[0] += (c - pv).unsigned_abs();
                sum[1] += (c - ph).unsigned_abs();
                sum[2] += (c - pd).unsigned_abs();
            }
        }
    }
    sum.map(|s| s >> 1)
}

/// ITU-T H.264 intra plane prediction of an `N`x`N` block (`N` = 16 luma or 8 chroma) into
/// `dst` (stride `N`): `clip((a + b (x - c0) + c (y - c0) + 16) >> 5)` with `c0 = N / 2 - 1`.
/// The sums must fit in 16 bits, as they do for 8-bit edges.
///
/// # Panics
/// If `N` is not 8 or 16, or `dst` is shorter than `N * N`.
#[inline]
pub fn intra_pred_plane<const N: usize>(a: i32, b: i32, c: i32, dst: &mut [u8]) {
    assert!((N == 8 || N == 16) && dst.len() >= N * N);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above; NEON is baseline on aarch64.
        unsafe { plane_neon::<N>(a, b, c, dst) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        unsafe { plane_avx2::<N>(a, b, c, dst) };
        return;
    }
    intra_pred_plane_scalar::<N>(a, b, c, dst);
}

/// Scalar reference for [`intra_pred_plane`].
pub fn intra_pred_plane_scalar<const N: usize>(a: i32, b: i32, c: i32, dst: &mut [u8]) {
    let c0 = (N / 2 - 1) as i32;
    for y in 0..N {
        for x in 0..N {
            let v = (a + b * (x as i32 - c0) + c * (y as i32 - c0) + 16) >> 5;
            dst[y * N + x] = v.clamp(0, 255) as u8;
        }
    }
}

/// Row stride and width of the planes of a [`HpelWindow`].
pub const HPEL_WIN_W: usize = 32;
/// Columns of each row of a [`HpelWindow`] that [`hpel_window`] interpolates: a $16\times16$
/// block displaced by up to one sample either way spans 18.
pub const HPEL_COLS: usize = 24;
/// Rows of the planes of a [`HpelWindow`].
pub const HPEL_WIN_H: usize = 18;

/// Rows of unrounded horizontal taps a [`HpelWindow`] keeps for its centre plane.
const HPEL_HR_ROWS: usize = HPEL_WIN_H + 5;

/// The three half-sample planes of ITU-T H.264 8.4.2.2.1 over a `HPEL_COLS` x `HPEL_WIN_H`
/// window, row-major with stride `HPEL_WIN_W`. With `(x, y)` the window's top-left integer sample:
/// `h` holds `b` at `(x + c + 1/2, y + r)`, `v` holds `h` at `(x + c, y + r + 1/2)` and `c` holds
/// `j` at `(x + c + 1/2, y + r + 1/2)`. [`hpel_window`] fills `h` and `v`, and `c` only once
/// [`hpel_window_centre`] derives it from the unrounded taps kept alongside.
#[derive(Clone, Debug)]
pub struct HpelWindow {
    pub h: [u8; HPEL_WIN_W * HPEL_WIN_H],
    pub v: [u8; HPEL_WIN_W * HPEL_WIN_H],
    pub c: [u8; HPEL_WIN_W * HPEL_WIN_H],
    /// Unrounded horizontal 6-tap sums of rows `-2..HPEL_WIN_H + 3`, stride `HPEL_COLS`.
    hr: [i16; HPEL_HR_ROWS * HPEL_COLS],
}

impl Default for HpelWindow {
    fn default() -> Self {
        Self {
            h: [0; HPEL_WIN_W * HPEL_WIN_H],
            v: [0; HPEL_WIN_W * HPEL_WIN_H],
            c: [0; HPEL_WIN_W * HPEL_WIN_H],
            hr: [0; HPEL_HR_ROWS * HPEL_COLS],
        }
    }
}

/// Interpolates the `h` and `v` planes of the window whose top-left integer sample is `src[off]`
/// (stride `ss`), columns `0..HPEL_COLS` of each row; the others are left as they are. Reads rows
/// `-2..=HPEL_WIN_H + 2` and columns `-2..=HPEL_WIN_W + 2` around it.
///
/// # Panics
/// If `src` lacks those margins.
#[inline]
pub fn hpel_window(src: &[u8], off: usize, ss: usize, out: &mut HpelWindow) {
    assert!(off >= 2 * ss + 2 && off + (HPEL_WIN_H + 2) * ss + HPEL_WIN_W + 3 <= src.len());
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above; NEON is baseline on aarch64.
        unsafe { hpel_neon(src.as_ptr().add(off), ss, out) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        unsafe { hpel_avx2(src.as_ptr().add(off), ss, out) };
        return;
    }
    hpel_window_scalar(src, off, ss, out);
}

/// Scalar reference for [`hpel_window`].
pub fn hpel_window_scalar(src: &[u8], off: usize, ss: usize, out: &mut HpelWindow) {
    assert!(off >= 2 * ss + 2 && off + (HPEL_WIN_H + 2) * ss + HPEL_WIN_W + 3 <= src.len());
    let at = |r: isize, c: isize| src[(off as isize + r * ss as isize + c) as usize] as i32;
    let tap6 =
        |a: i32, b: i32, c: i32, d: i32, e: i32, f: i32| a - 5 * b + 20 * c + 20 * d - 5 * e + f;
    let clip = |v: i32| v.clamp(0, 255) as u8;
    for k in 0..HPEL_HR_ROWS {
        let r = k as isize - 2;
        for c in 0..HPEL_COLS as isize {
            let h = tap6(
                at(r, c - 2),
                at(r, c - 1),
                at(r, c),
                at(r, c + 1),
                at(r, c + 2),
                at(r, c + 3),
            );
            out.hr[k * HPEL_COLS + c as usize] = h as i16;
            if (0..HPEL_WIN_H as isize).contains(&r) {
                let i = r as usize * HPEL_WIN_W + c as usize;
                out.h[i] = clip((h + 16) >> 5);
                let v = tap6(
                    at(r - 2, c),
                    at(r - 1, c),
                    at(r, c),
                    at(r + 1, c),
                    at(r + 2, c),
                    at(r + 3, c),
                );
                out.v[i] = clip((v + 16) >> 5);
            }
        }
    }
}

/// Interpolates the `c` plane of a window that [`hpel_window`] has filled, from the horizontal
/// taps it kept, columns `0..HPEL_COLS` of each row.
#[inline]
pub fn hpel_window_centre(out: &mut HpelWindow) {
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: NEON is baseline on aarch64.
        unsafe { hpel_centre_neon(out) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        unsafe { hpel_centre_avx2(out) };
        return;
    }
    hpel_window_centre_scalar(out);
}

/// Scalar reference for [`hpel_window_centre`].
pub fn hpel_window_centre_scalar(out: &mut HpelWindow) {
    for r in 0..HPEL_WIN_H {
        for c in 0..HPEL_COLS {
            let t = |k: usize| out.hr[(r + k) * HPEL_COLS + c] as i32;
            let j = t(0) - 5 * t(1) + 20 * t(2) + 20 * t(3) - 5 * t(4) + t(5);
            out.c[r * HPEL_WIN_W + c] = ((j + 512) >> 10).clamp(0, 255) as u8;
        }
    }
}

/// `dst = (a + b + 1) >> 1` over a $16\times16$ block (stride 16), from strided sources.
///
/// # Panics
/// If a source is shorter than 16 rows of 16 samples at its stride.
#[inline]
pub fn pixel_avg_16x16(dst: &mut [u8; 256], a: &[u8], sa: usize, b: &[u8], sb: usize) {
    check_block(a, sa, 16, 16);
    check_block(b, sb, 16, 16);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above; NEON is baseline on aarch64.
        unsafe {
            use std::arch::aarch64::*;
            for r in 0..16 {
                let v = vrhaddq_u8(
                    vld1q_u8(a.as_ptr().add(r * sa)),
                    vld1q_u8(b.as_ptr().add(r * sb)),
                );
                vst1q_u8(dst.as_mut_ptr().add(r * 16), v);
            }
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        unsafe { avg16_avx2(dst, a.as_ptr(), sa, b.as_ptr(), sb) };
        return;
    }
    pixel_avg_16x16_scalar(dst, a, sa, b, sb);
}

/// Scalar reference for [`pixel_avg_16x16`].
pub fn pixel_avg_16x16_scalar(dst: &mut [u8; 256], a: &[u8], sa: usize, b: &[u8], sb: usize) {
    for r in 0..16 {
        for c in 0..16 {
            dst[r * 16 + c] = ((a[r * sa + c] as u16 + b[r * sb + c] as u16 + 1) >> 1) as u8;
        }
    }
}

/// Positions in `[avg2 | avg3]` of the edge `[L, L, K, J, I, M, A, B, C, D, E, F, G, H, H, H]`
/// (`avg2[i]` of samples `i, i + 1`, `avg3[i]` centred on `i + 1`) of every sample of the six
/// directional intra $4\times4$ predictions, modes 3 to 8, raster order.
const PRED4_DIR: [[u8; 16]; 6] = [
    [
        22, 23, 24, 25, 23, 24, 25, 26, 24, 25, 26, 27, 25, 26, 27, 28,
    ],
    [
        20, 21, 22, 23, 19, 20, 21, 22, 18, 19, 20, 21, 17, 18, 19, 20,
    ],
    [5, 6, 7, 8, 20, 21, 22, 23, 19, 5, 6, 7, 18, 20, 21, 22],
    [4, 20, 21, 22, 3, 19, 4, 20, 2, 18, 3, 19, 1, 17, 2, 18],
    [6, 7, 8, 9, 22, 23, 24, 25, 7, 8, 9, 10, 23, 24, 25, 26],
    [3, 18, 2, 17, 2, 17, 1, 16, 1, 16, 0, 0, 0, 0, 0, 0],
];

/// The six directional intra $4\times4$ predictions (ITU-T H.264 8.3.1.2.4-9, modes 3 to 8,
/// 16 raster samples each) from the top and top-right samples `top`, the left column `left`
/// and the corner `tl`: every sample is a 2- or 3-tap average of neighbouring edge samples,
/// so all six are byte shuffles of the edge's two filtered versions (`x264` `predict_4x4`).
#[inline]
pub fn intra4x4_pred_dir(top: &[u8; 8], left: &[u8; 4], tl: u8) -> [[u8; 16]; 6] {
    let [a, b, c, d, e, f, g, h] = *top;
    let [i, j, k, l] = *left;
    let edge = [l, l, k, j, i, tl, a, b, c, d, e, f, g, h, h, h];
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: NEON is baseline on aarch64.
        return unsafe { pred4_dir_neon(&edge) };
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2 (and so SSSE3).
        return unsafe { pred4_dir_avx2(&edge) };
    }
    pred4_dir_from_edge(&edge)
}

/// [`intra4x4_pred_dir`] through the shuffle table, without SIMD.
fn pred4_dir_from_edge(edge: &[u8; 16]) -> [[u8; 16]; 6] {
    let mut t = [0u8; 32];
    for n in 0..15 {
        t[n] = ((edge[n] as u16 + edge[n + 1] as u16 + 1) >> 1) as u8;
    }
    for n in 0..14 {
        t[16 + n] = ((edge[n] as u16 + 2 * edge[n + 1] as u16 + edge[n + 2] as u16 + 2) >> 2) as u8;
    }
    PRED4_DIR.map(|idx| idx.map(|x| t[x as usize]))
}

/// Scalar reference for [`intra4x4_pred_dir`]: the generic 4x4 predictor, mode by mode.
pub fn intra4x4_pred_dir_scalar(top: &[u8; 8], left: &[u8; 4], tl: u8) -> [[u8; 16]; 6] {
    std::array::from_fn(|n| {
        let mut out = [0u8; 16];
        crate::predict_intra_4x4_u8_scalar(n as u8 + 3, Some(top), Some(left), tl, &mut out, 4);
        out
    })
}

/// Chroma motion compensation of the $8\times8$ blocks of both planes (ITU-T H.264 8.4.2.2.2):
/// `out = (wa A + wb B + wc C + wd D + 32) >> 6` with the bilinear weights of the eighth-sample
/// fractions `fx`, `fy` (`& 7`), where `A` is `u[off]` / `v[off]` (stride `ss`).
///
/// # Panics
/// If a plane is shorter than `off + 8 * ss + 16` (the vector kernels read 16 samples per row).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn chroma_mc_8x8_x2(
    u: &[u8],
    v: &[u8],
    off: usize,
    ss: usize,
    fx: u8,
    fy: u8,
    out_u: &mut [u8; 64],
    out_v: &mut [u8; 64],
) {
    let end = ss.checked_mul(8).and_then(|d| d.checked_add(off + 16));
    assert!(end.is_some_and(|e| e <= u.len() && e <= v.len()));
    let (fx, fy) = (fx & 7, fy & 7);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above; NEON is baseline on aarch64.
        unsafe {
            chroma_mc_neon(u.as_ptr().add(off), ss, fx, fy, out_u);
            chroma_mc_neon(v.as_ptr().add(off), ss, fx, fy, out_v);
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        unsafe {
            chroma_mc_avx2(
                u.as_ptr().add(off),
                v.as_ptr().add(off),
                ss,
                fx,
                fy,
                out_u,
                out_v,
            )
        };
        return;
    }
    chroma_mc_8x8_x2_scalar(u, v, off, ss, fx, fy, out_u, out_v);
}

/// Scalar reference for [`chroma_mc_8x8_x2`].
#[allow(clippy::too_many_arguments)]
pub fn chroma_mc_8x8_x2_scalar(
    u: &[u8],
    v: &[u8],
    off: usize,
    ss: usize,
    fx: u8,
    fy: u8,
    out_u: &mut [u8; 64],
    out_v: &mut [u8; 64],
) {
    let (fx, fy) = ((fx & 7) as u32, (fy & 7) as u32);
    let (wa, wb, wc, wd) = ((8 - fx) * (8 - fy), fx * (8 - fy), (8 - fx) * fy, fx * fy);
    for (src, out) in [(u, out_u), (v, out_v)] {
        for r in 0..8 {
            let (s0, s1) = (&src[off + r * ss..], &src[off + (r + 1) * ss..]);
            for c in 0..8 {
                let x = wa * s0[c] as u32
                    + wb * s0[c + 1] as u32
                    + wc * s1[c] as u32
                    + wd * s1[c + 1] as u32;
                out[r * 8 + c] = ((x + 32) >> 6) as u8;
            }
        }
    }
}

/// The two half-sample $16\times16$ predictions next to the block whose top-left integer sample
/// is `src[off]` (stride `ss`): `out_h` the `b` samples half a sample left (`left`) or right of
/// it, `out_v` the `h` samples half a sample above (`up`) or below it (ITU-T H.264 8.4.2.2.1).
/// Reads rows and columns `-3..=18` around the block.
///
/// # Panics
/// If `src` lacks those margins.
#[inline]
pub fn hpel_hv_16x16(
    src: &[u8],
    off: usize,
    ss: usize,
    left: bool,
    up: bool,
    out_h: &mut [u8; 256],
    out_v: &mut [u8; 256],
) {
    assert!(off >= 3 * ss + 3 && off + 18 * ss + 19 <= src.len());
    let (h0, v0) = (off - usize::from(left), off - usize::from(up) * ss);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above; NEON is baseline on aarch64.
        unsafe { hv_neon(src.as_ptr(), h0, v0, ss, out_h, out_v) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        unsafe { hv_avx2(src.as_ptr(), h0, v0, ss, out_h, out_v) };
        return;
    }
    hpel_hv_16x16_scalar(src, off, ss, left, up, out_h, out_v);
}

/// Scalar reference for [`hpel_hv_16x16`].
pub fn hpel_hv_16x16_scalar(
    src: &[u8],
    off: usize,
    ss: usize,
    left: bool,
    up: bool,
    out_h: &mut [u8; 256],
    out_v: &mut [u8; 256],
) {
    let (h0, v0) = (off - usize::from(left), off - usize::from(up) * ss);
    let tap = |o: usize, step: usize| {
        let p = |k: usize| src[o + k * step - 2 * step] as i32;
        let v = p(0) - 5 * p(1) + 20 * p(2) + 20 * p(3) - 5 * p(4) + p(5);
        ((v + 16) >> 5).clamp(0, 255) as u8
    };
    for r in 0..16 {
        for c in 0..16 {
            out_h[r * 16 + c] = tap(h0 + r * ss + c, 1);
            out_v[r * 16 + c] = tap(v0 + r * ss + c, ss);
        }
    }
}

/// SAD of the contiguous $16\times16$ block `src` against `(a + b + 1) >> 1` of the strided
/// blocks `a` and `b`, without storing the average (`x264` `pixel_avg` then `sad`).
///
/// # Panics
/// If a source is shorter than 16 rows of 16 samples at its stride.
#[inline]
pub fn pixel_sad_avg_16x16(src: &[u8; 256], a: &[u8], sa: usize, b: &[u8], sb: usize) -> u32 {
    check_block(a, sa, 16, 16);
    check_block(b, sb, 16, 16);
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above; NEON is baseline on aarch64.
        unsafe {
            use std::arch::aarch64::*;
            let mut acc = vdupq_n_u16(0);
            for r in 0..16 {
                let v = vrhaddq_u8(
                    vld1q_u8(a.as_ptr().add(r * sa)),
                    vld1q_u8(b.as_ptr().add(r * sb)),
                );
                let s = vld1q_u8(src.as_ptr().add(r * 16));
                acc = vabal_u8(acc, vget_low_u8(s), vget_low_u8(v));
                acc = vabal_high_u8(acc, s, v);
            }
            return vaddlvq_u16(acc);
        }
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        return unsafe { sad_avg16_avx2(src, a.as_ptr(), sa, b.as_ptr(), sb) };
    }
    pixel_sad_avg_16x16_scalar(src, a, sa, b, sb)
}

/// Scalar reference for [`pixel_sad_avg_16x16`].
pub fn pixel_sad_avg_16x16_scalar(
    src: &[u8; 256],
    a: &[u8],
    sa: usize,
    b: &[u8],
    sb: usize,
) -> u32 {
    let mut avg = [0u8; 256];
    pixel_avg_16x16_scalar(&mut avg, a, sa, b, sb);
    pixel_sad_16x16_scalar(src, 16, &avg, 16)
}

/// H.264 frame zig-zag scan of a $4\times4$ block (Table 8-13), as raster positions.
const ZIGZAG4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// Reorders four raster-order $4\times4$ blocks into zig-zag scan order.
#[inline]
pub fn zigzag_4x4_x4(blocks: &[[i16; 16]; 4]) -> [[i16; 16]; 4] {
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: NEON is baseline on aarch64.
        return unsafe { zigzag_neon(blocks) };
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        return unsafe { zigzag_avx2(blocks) };
    }
    zigzag_4x4_x4_scalar(blocks)
}

/// Bit `i` of entry `b` is set when coefficient `i` of block `b` is non-zero.
#[inline]
pub fn nonzero_mask_4x4_x4(blocks: &[[i16; 16]; 4]) -> [u32; 4] {
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: NEON is baseline on aarch64.
        return unsafe { nonzero_mask_neon(blocks) };
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        return unsafe { nonzero_mask_avx2(blocks) };
    }
    nonzero_mask_4x4_x4_scalar(blocks)
}

/// Scalar reference for [`nonzero_mask_4x4_x4`].
pub fn nonzero_mask_4x4_x4_scalar(blocks: &[[i16; 16]; 4]) -> [u32; 4] {
    blocks.map(|b| {
        b.iter()
            .enumerate()
            .fold(0, |m, (i, &c)| m | (u32::from(c != 0) << i))
    })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn nonzero_mask_avx2(blocks: &[[i16; 16]; 4]) -> [u32; 4] {
    use std::arch::x86_64::*;
    unsafe {
        let zero = _mm256_setzero_si256();
        blocks.map(|b| {
            let v = _mm256_loadu_si256(b.as_ptr() as *const __m256i);
            let z = _mm256_cmpeq_epi16(v, zero);
            // Bytes of the 16-bit compare, low then high lane; keep one bit per coefficient.
            let bytes =
                _mm_packs_epi16(_mm256_castsi256_si128(z), _mm256_extracti128_si256::<1>(z));
            !(_mm_movemask_epi8(bytes) as u32) & 0xFFFF
        })
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn nonzero_mask_neon(blocks: &[[i16; 16]; 4]) -> [u32; 4] {
    use std::arch::aarch64::*;
    unsafe {
        let bits: [u16; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
        let bits = vld1q_u16(bits.as_ptr());
        blocks.map(|b| {
            let lo = vandq_u16(vtstq_s16(vld1q_s16(b.as_ptr()), vdupq_n_s16(-1)), bits);
            let hi = vandq_u16(
                vtstq_s16(vld1q_s16(b.as_ptr().add(8)), vdupq_n_s16(-1)),
                bits,
            );
            vaddvq_u16(lo) as u32 | (vaddvq_u16(hi) as u32) << 8
        })
    }
}

/// Scalar reference for [`zigzag_4x4_x4`].
pub fn zigzag_4x4_x4_scalar(blocks: &[[i16; 16]; 4]) -> [[i16; 16]; 4] {
    blocks.map(|b| ZIGZAG4.map(|i| b[i]))
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    /// Rows `p` (low lane) and `p + s` (high lane), 16 bytes each.
    #[inline(always)]
    pub unsafe fn load_2x16(p: *const u8, s: usize) -> __m256i {
        unsafe {
            let lo = _mm_loadu_si128(p as *const __m128i);
            let hi = _mm_loadu_si128(p.add(s) as *const __m128i);
            _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi)
        }
    }

    /// Rows `p .. p + 3s`, 8 bytes each, packed into one register.
    #[inline(always)]
    pub unsafe fn load_4x8(p: *const u8, s: usize) -> __m256i {
        unsafe {
            let r = |k: usize| _mm_loadl_epi64(p.add(k * s) as *const __m128i);
            let lo = _mm_unpacklo_epi64(r(0), r(1));
            let hi = _mm_unpacklo_epi64(r(2), r(3));
            _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi)
        }
    }

    #[inline(always)]
    pub unsafe fn hsum_epi64(v: __m256i) -> u32 {
        unsafe {
            let s = _mm_add_epi64(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
            (_mm_cvtsi128_si64(s) + _mm_extract_epi64::<1>(s)) as u32
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sad16_avx2(a: *const u8, sa: usize, b: *const u8, sb: usize) -> u32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut acc = _mm256_setzero_si256();
        for y in (0..16).step_by(2) {
            let va = x86::load_2x16(a.add(y * sa), sa);
            let vb = x86::load_2x16(b.add(y * sb), sb);
            acc = _mm256_add_epi64(acc, _mm256_sad_epu8(va, vb));
        }
        x86::hsum_epi64(acc)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sad8_avx2(a: *const u8, sa: usize, b: *const u8, sb: usize) -> u32 {
    use std::arch::x86_64::*;
    unsafe {
        let s0 = _mm256_sad_epu8(x86::load_4x8(a, sa), x86::load_4x8(b, sb));
        let s1 = _mm256_sad_epu8(
            x86::load_4x8(a.add(4 * sa), sa),
            x86::load_4x8(b.add(4 * sb), sb),
        );
        x86::hsum_epi64(_mm256_add_epi64(s0, s1))
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn ssd_avx2<const N: usize>(a: *const u8, sa: usize, b: *const u8, sb: usize) -> u32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut acc = _mm256_setzero_si256();
        for y in 0..N {
            let (va, vb) = if N == 16 {
                (
                    _mm256_cvtepu8_epi16(_mm_loadu_si128(a.add(y * sa) as *const __m128i)),
                    _mm256_cvtepu8_epi16(_mm_loadu_si128(b.add(y * sb) as *const __m128i)),
                )
            } else {
                (
                    _mm256_cvtepu8_epi16(_mm_loadl_epi64(a.add(y * sa) as *const __m128i)),
                    _mm256_cvtepu8_epi16(_mm_loadl_epi64(b.add(y * sb) as *const __m128i)),
                )
            };
            let d = _mm256_sub_epi16(va, vb);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d, d));
        }
        let s = _mm_add_epi32(
            _mm256_castsi256_si128(acc),
            _mm256_extracti128_si256::<1>(acc),
        );
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
        _mm_cvtsi128_si32(s) as u32
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn ssd_neon<const N: usize>(a: *const u8, sa: usize, b: *const u8, sb: usize) -> u32 {
    use std::arch::aarch64::*;
    unsafe {
        let mut acc = vdupq_n_u32(0);
        for y in 0..N {
            for h in 0..N / 8 {
                let d = vabd_u8(
                    vld1_u8(a.add(y * sa + h * 8)),
                    vld1_u8(b.add(y * sb + h * 8)),
                );
                let d2 = vmull_u8(d, d);
                acc = vpadalq_u16(acc, d2);
            }
        }
        vaddvq_u32(acc)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sad_x4_avx2(a: &[u8; 256], b: *const u8, offs: [usize; 4], sb: usize) -> [u32; 4] {
    use std::arch::x86_64::*;
    unsafe {
        let (b0, b1, b2, b3) = (
            b.add(offs[0]),
            b.add(offs[1]),
            b.add(offs[2]),
            b.add(offs[3]),
        );
        let mut acc = [_mm256_setzero_si256(); 4];
        for y in (0..16).step_by(2) {
            let va = _mm256_loadu_si256(a.as_ptr().add(y * 16) as *const __m256i);
            let o = y * sb;
            acc[0] = _mm256_add_epi64(acc[0], _mm256_sad_epu8(va, x86::load_2x16(b0.add(o), sb)));
            acc[1] = _mm256_add_epi64(acc[1], _mm256_sad_epu8(va, x86::load_2x16(b1.add(o), sb)));
            acc[2] = _mm256_add_epi64(acc[2], _mm256_sad_epu8(va, x86::load_2x16(b2.add(o), sb)));
            acc[3] = _mm256_add_epi64(acc[3], _mm256_sad_epu8(va, x86::load_2x16(b3.add(o), sb)));
        }
        acc.map(|v| x86::hsum_epi64(v))
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn satd4_avx2(a: *const u8, sa: usize, b: *const u8, sb: usize) -> u32 {
    use std::arch::x86_64::*;
    unsafe {
        let rows = |p: *const u8, s: usize| {
            let r = |k: usize| std::ptr::read_unaligned(p.add(k * s) as *const i32);
            _mm256_cvtepu8_epi16(_mm_setr_epi32(r(0), r(1), r(2), r(3)))
        };
        // One block per register: rows 0, 1 in the low lane, rows 2, 3 in the high lane.
        let d = _mm256_sub_epi16(rows(a, sa), rows(b, sb));
        let t = _mm256_shufflehi_epi16::<0b10_11_00_01>(_mm256_shufflelo_epi16::<0b10_11_00_01>(d));
        let h1 = _mm256_blend_epi16::<0b1010_1010>(_mm256_add_epi16(d, t), _mm256_sub_epi16(t, d));
        let t =
            _mm256_shufflehi_epi16::<0b01_00_11_10>(_mm256_shufflelo_epi16::<0b01_00_11_10>(h1));
        let h2 =
            _mm256_blend_epi16::<0b1100_1100>(_mm256_add_epi16(h1, t), _mm256_sub_epi16(t, h1));
        let t = _mm256_shuffle_epi32::<0b01_00_11_10>(h2);
        let v1 =
            _mm256_blend_epi32::<0b1100_1100>(_mm256_add_epi16(h2, t), _mm256_sub_epi16(t, h2));
        // Last butterfly as 2 max(|p|, |q|), counted once in each lane of the pair.
        let t = _mm256_permute2x128_si256::<0x01>(v1, v1);
        let m = _mm256_max_epi16(_mm256_abs_epi16(v1), _mm256_abs_epi16(t));
        let s = _mm256_madd_epi16(m, _mm256_set1_epi16(1));
        let s = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256::<1>(s));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
        (_mm_cvtsi128_si32(s) as u32) >> 1
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn satd16_avx2(a: *const u8, sa: usize, b: *const u8, sb: usize) -> u32 {
    use std::arch::x86_64::*;
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let mut acc = _mm256_setzero_si256();
        for strip in 0..4 {
            // One 4-row strip: each register holds a row of all four 4x4 blocks.
            let row = |k: usize| {
                let r = strip * 4 + k;
                let va = _mm256_cvtepu8_epi16(_mm_loadu_si128(a.add(r * sa) as *const __m128i));
                let vb = _mm256_cvtepu8_epi16(_mm_loadu_si128(b.add(r * sb) as *const __m128i));
                _mm256_sub_epi16(va, vb)
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
            let mut strip_sum = _mm256_setzero_si256();
            for x in v {
                // Horizontal butterfly over adjacent columns, then the last stage as
                // |p + q| + |p - q| = 2 max(|p|, |q|), summed over both lanes of each pair.
                let t = _mm256_shufflehi_epi16::<0b10_11_00_01>(_mm256_shufflelo_epi16::<
                    0b10_11_00_01,
                >(x));
                let w = _mm256_blend_epi16::<0b1010_1010>(
                    _mm256_add_epi16(x, t),
                    _mm256_sub_epi16(t, x),
                );
                let u = _mm256_shufflehi_epi16::<0b01_00_11_10>(_mm256_shufflelo_epi16::<
                    0b01_00_11_10,
                >(w));
                strip_sum = _mm256_add_epi16(
                    strip_sum,
                    _mm256_max_epi16(_mm256_abs_epi16(w), _mm256_abs_epi16(u)),
                );
            }
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(strip_sum, ones));
        }
        let s = _mm_add_epi32(
            _mm256_castsi256_si128(acc),
            _mm256_extracti128_si256::<1>(acc),
        );
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
        // Each pair maximum was counted once per lane of the pair.
        (_mm_cvtsi128_si32(s) as u32) >> 1
    }
}

/// Transposes the $4\times4$ blocks held one row per register (two blocks per 128-bit lane) into
/// one column per register, so the same element-wise butterflies serve both dimensions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn transpose_4x4_epi16_avx2(v: [std::arch::x86_64::__m256i; 4]) -> [std::arch::x86_64::__m256i; 4] {
    use std::arch::x86_64::*;
    let t0 = _mm256_unpacklo_epi16(v[0], v[1]);
    let t1 = _mm256_unpackhi_epi16(v[0], v[1]);
    let t2 = _mm256_unpacklo_epi16(v[2], v[3]);
    let t3 = _mm256_unpackhi_epi16(v[2], v[3]);
    let u0 = _mm256_unpacklo_epi32(t0, t2);
    let u1 = _mm256_unpackhi_epi32(t0, t2);
    let u2 = _mm256_unpacklo_epi32(t1, t3);
    let u3 = _mm256_unpackhi_epi32(t1, t3);
    [
        _mm256_unpacklo_epi64(u0, u2),
        _mm256_unpackhi_epi64(u0, u2),
        _mm256_unpacklo_epi64(u1, u3),
        _mm256_unpackhi_epi64(u1, u3),
    ]
}

/// The H.264 forward core transform butterfly applied element-wise across four registers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn dct4_butterfly_avx2(d: [std::arch::x86_64::__m256i; 4]) -> [std::arch::x86_64::__m256i; 4] {
    use std::arch::x86_64::*;
    let (s03, s12) = (_mm256_add_epi16(d[0], d[3]), _mm256_add_epi16(d[1], d[2]));
    let (d03, d12) = (_mm256_sub_epi16(d[0], d[3]), _mm256_sub_epi16(d[1], d[2]));
    [
        _mm256_add_epi16(s03, s12),
        _mm256_add_epi16(_mm256_slli_epi16::<1>(d03), d12),
        _mm256_sub_epi16(s03, s12),
        _mm256_sub_epi16(d03, _mm256_slli_epi16::<1>(d12)),
    ]
}

/// [`intra_satd_x3_16x16`]: the transform of each strip stays in registers holding one column
/// of coefficients per block (`o[h]` lane `4 b + v`), so each prediction only replaces a few
/// lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn intra_satd_x3_avx2(
    src: &[u8; 256],
    tv: &[[i16; 4]; 4],
    th: &[[i16; 4]; 4],
    dc16: i16,
) -> [u32; 3] {
    use std::arch::x86_64::*;
    unsafe {
        let ones = _mm256_set1_epi16(1);
        // Vertical prediction of column h: lane 4 b holds block b's transformed top edge.
        let pv: [__m256i; 4] = std::array::from_fn(|h| {
            _mm256_setr_epi16(
                tv[0][h], 0, 0, 0, tv[1][h], 0, 0, 0, tv[2][h], 0, 0, 0, tv[3][h], 0, 0, 0,
            )
        });
        let pd = _mm256_setr_epi16(dc16, 0, 0, 0, dc16, 0, 0, 0, dc16, 0, 0, 0, dc16, 0, 0, 0);
        let (mut acc_v, mut acc_h, mut acc_d) = (
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
        );
        for (strip, t) in th.iter().enumerate() {
            let row = |k: usize| {
                _mm256_cvtepu8_epi16(_mm_loadu_si128(
                    src.as_ptr().add((strip * 4 + k) * 16) as *const __m128i
                ))
            };
            let v = hadamard4_butterfly_avx2([row(0), row(1), row(2), row(3)]);
            let o = hadamard4_butterfly_avx2(transpose_4x4_epi16_avx2(v));
            // Column 0 under the horizontal prediction: every block holds the transformed left
            // edge of this block row.
            let ph = _mm256_set1_epi64x(
                (t[0] as u16 as i64)
                    | (t[1] as u16 as i64) << 16
                    | (t[2] as u16 as i64) << 32
                    | (t[3] as u16 as i64) << 48,
            );
            let abs_sum = |x: __m256i| _mm256_madd_epi16(_mm256_abs_epi16(x), ones);
            let rest = _mm256_add_epi32(
                _mm256_add_epi32(abs_sum(o[1]), abs_sum(o[2])),
                abs_sum(o[3]),
            );
            acc_h = _mm256_add_epi32(acc_h, rest);
            acc_d = _mm256_add_epi32(acc_d, rest);
            acc_h = _mm256_add_epi32(acc_h, abs_sum(_mm256_sub_epi16(o[0], ph)));
            acc_d = _mm256_add_epi32(acc_d, abs_sum(_mm256_sub_epi16(o[0], pd)));
            for h in 0..4 {
                acc_v = _mm256_add_epi32(acc_v, abs_sum(_mm256_sub_epi16(o[h], pv[h])));
            }
        }
        let hsum = |a: __m256i| {
            let s = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256::<1>(a));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
            _mm_cvtsi128_si32(s) as u32
        };
        [hsum(acc_v) >> 1, hsum(acc_h) >> 1, hsum(acc_d) >> 1]
    }
}

/// The $4\times4$ Hadamard butterfly of [`hadamard4x4_16x4`] applied element-wise across four
/// registers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn hadamard4_butterfly_avx2(d: [std::arch::x86_64::__m256i; 4]) -> [std::arch::x86_64::__m256i; 4] {
    use std::arch::x86_64::*;
    let (s01, d01) = (_mm256_add_epi16(d[0], d[1]), _mm256_sub_epi16(d[0], d[1]));
    let (s23, d23) = (_mm256_add_epi16(d[2], d[3]), _mm256_sub_epi16(d[2], d[3]));
    [
        _mm256_add_epi16(s01, s23),
        _mm256_sub_epi16(s01, s23),
        _mm256_add_epi16(d01, d23),
        _mm256_sub_epi16(d01, d23),
    ]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dct_strip_avx2<const HAD: bool>(
    src: *const u8,
    ss: usize,
    pred: *const u8,
    ps: usize,
    from_8x8: bool,
    out: &mut [[i16; 16]; 4],
) {
    use std::arch::x86_64::*;
    unsafe {
        // Each register holds one row of all four blocks: blocks 0, 1 in the low lane, 2, 3 in the
        // high lane (an 8x8 block's rows k and k + 4 side by side).
        let row = |k: usize| {
            let (s, p) = if from_8x8 {
                (
                    _mm_unpacklo_epi64(
                        _mm_loadl_epi64(src.add(k * ss) as *const __m128i),
                        _mm_loadl_epi64(src.add((k + 4) * ss) as *const __m128i),
                    ),
                    _mm_unpacklo_epi64(
                        _mm_loadl_epi64(pred.add(k * ps) as *const __m128i),
                        _mm_loadl_epi64(pred.add((k + 4) * ps) as *const __m128i),
                    ),
                )
            } else {
                (
                    _mm_loadu_si128(src.add(k * ss) as *const __m128i),
                    _mm_loadu_si128(pred.add(k * ps) as *const __m128i),
                )
            };
            _mm256_sub_epi16(_mm256_cvtepu8_epi16(s), _mm256_cvtepu8_epi16(p))
        };
        // Vertical pass on rows, then the horizontal pass on columns; the transform is exact
        // integer arithmetic within 16 bits, so the order of the passes does not matter.
        let butterfly = |d| {
            if HAD {
                hadamard4_butterfly_avx2(d)
            } else {
                dct4_butterfly_avx2(d)
            }
        };
        let v = butterfly([row(0), row(1), row(2), row(3)]);
        let o = butterfly(transpose_4x4_epi16_avx2(v));
        // o[m] holds column m of each coefficient block; interleave back to raster rows.
        let t0 = _mm256_unpacklo_epi16(o[0], o[1]);
        let t1 = _mm256_unpackhi_epi16(o[0], o[1]);
        let t2 = _mm256_unpacklo_epi16(o[2], o[3]);
        let t3 = _mm256_unpackhi_epi16(o[2], o[3]);
        let (a01, a23) = (_mm256_unpacklo_epi32(t0, t2), _mm256_unpackhi_epi32(t0, t2));
        let (b01, b23) = (_mm256_unpacklo_epi32(t1, t3), _mm256_unpackhi_epi32(t1, t3));
        let p = out.as_mut_ptr() as *mut __m256i;
        _mm256_storeu_si256(p, _mm256_permute2x128_si256::<0x20>(a01, a23));
        _mm256_storeu_si256(p.add(1), _mm256_permute2x128_si256::<0x20>(b01, b23));
        _mm256_storeu_si256(p.add(2), _mm256_permute2x128_si256::<0x31>(a01, a23));
        _mm256_storeu_si256(p.add(3), _mm256_permute2x128_si256::<0x31>(b01, b23));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quant_avx2(blocks: &mut [[i16; 16]; 4], q: &Quant4x4) -> u32 {
    use std::arch::x86_64::*;
    unsafe {
        let mf = _mm256_loadu_si256(q.mf.as_ptr() as *const __m256i);
        let bias = _mm256_set1_epi32(q.bias as i32);
        let shift = _mm_cvtsi32_si128(q.shift as i32);
        let one = _mm256_set1_epi16(1);
        let mut nz = 0;
        for (b, blk) in blocks.iter_mut().enumerate() {
            let p = blk.as_mut_ptr() as *mut __m256i;
            let c = _mm256_loadu_si256(p);
            // |c| as unsigned 16-bit (|-32768| wraps to 32768), widened products in 32 bits.
            let a = _mm256_abs_epi16(c);
            let (lo, hi) = (_mm256_mullo_epi16(a, mf), _mm256_mulhi_epu16(a, mf));
            let p0 = _mm256_srl_epi32(_mm256_add_epi32(_mm256_unpacklo_epi16(lo, hi), bias), shift);
            let p1 = _mm256_srl_epi32(_mm256_add_epi32(_mm256_unpackhi_epi16(lo, hi), bias), shift);
            let mag = _mm256_packs_epi32(p0, p1);
            let level = _mm256_sign_epi16(mag, c);
            _mm256_storeu_si256(p, level);
            if _mm256_testz_si256(level, level) == 0 {
                nz |= 1 << b;
                let big = _mm256_cmpgt_epi16(mag, one);
                if _mm256_testz_si256(big, big) == 0 {
                    nz |= 0x10 << b;
                }
            }
        }
        nz
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dequant_avx2(blocks: &mut [[i16; 16]; 4], scale: &[i32; 16]) {
    use std::arch::x86_64::*;
    unsafe {
        let s0 = _mm256_loadu_si256(scale.as_ptr() as *const __m256i);
        let s1 = _mm256_loadu_si256(scale.as_ptr().add(8) as *const __m256i);
        for blk in blocks.iter_mut() {
            let p = blk.as_mut_ptr();
            let c0 = _mm256_cvtepi16_epi32(_mm_loadu_si128(p as *const __m128i));
            let c1 = _mm256_cvtepi16_epi32(_mm_loadu_si128(p.add(8) as *const __m128i));
            let d = _mm256_packs_epi32(_mm256_mullo_epi32(c0, s0), _mm256_mullo_epi32(c1, s1));
            // packs works per 128-bit lane: restore coefficient order.
            let d = _mm256_permute4x64_epi64::<0b11_01_10_00>(d);
            _mm256_storeu_si256(p as *mut __m256i, d);
        }
    }
}

/// Transposes the $4\times4$ 32-bit block held in each 128-bit lane of four registers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn transpose_4x4_epi32_avx2(r: [std::arch::x86_64::__m256i; 4]) -> [std::arch::x86_64::__m256i; 4] {
    use std::arch::x86_64::*;
    let t0 = _mm256_unpacklo_epi32(r[0], r[1]);
    let t1 = _mm256_unpackhi_epi32(r[0], r[1]);
    let t2 = _mm256_unpacklo_epi32(r[2], r[3]);
    let t3 = _mm256_unpackhi_epi32(r[2], r[3]);
    [
        _mm256_unpacklo_epi64(t0, t2),
        _mm256_unpackhi_epi64(t0, t2),
        _mm256_unpacklo_epi64(t1, t3),
        _mm256_unpackhi_epi64(t1, t3),
    ]
}

/// The H.264 inverse core transform butterfly applied element-wise across four registers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn idct4_butterfly_avx2(c: [std::arch::x86_64::__m256i; 4]) -> [std::arch::x86_64::__m256i; 4] {
    use std::arch::x86_64::*;
    let e0 = _mm256_add_epi32(c[0], c[2]);
    let e1 = _mm256_sub_epi32(c[0], c[2]);
    let e2 = _mm256_sub_epi32(_mm256_srai_epi32::<1>(c[1]), c[3]);
    let e3 = _mm256_add_epi32(c[1], _mm256_srai_epi32::<1>(c[3]));
    [
        _mm256_add_epi32(e0, e3),
        _mm256_add_epi32(e1, e2),
        _mm256_sub_epi32(e1, e2),
        _mm256_sub_epi32(e0, e3),
    ]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn idct_add_8x8_avx2(dst: *mut u8, ds: usize, blocks: &[[i16; 16]; 4]) {
    use std::arch::x86_64::*;
    unsafe {
        let round = _mm256_set1_epi32(32);
        for half in 0..2 {
            let (a, b) = (&blocks[half * 2], &blocks[half * 2 + 1]);
            // Register k: row k of block a (low lane) and of block b (high lane), as 32 bits.
            let rows: [__m256i; 4] = std::array::from_fn(|k| {
                let ra = _mm_loadl_epi64(a.as_ptr().add(k * 4) as *const __m128i);
                let rb = _mm_loadl_epi64(b.as_ptr().add(k * 4) as *const __m128i);
                _mm256_cvtepi16_epi32(_mm_unpacklo_epi64(ra, rb))
            });
            // Rows first (on coefficient columns), then columns, as in ITU-T H.264 8.5.12.2.
            let h = idct4_butterfly_avx2(transpose_4x4_epi32_avx2(rows));
            let v = idct4_butterfly_avx2(transpose_4x4_epi32_avx2(h));
            for (k, o) in v.iter().enumerate() {
                let res = _mm256_srai_epi32::<6>(_mm256_add_epi32(*o, round));
                // [a row k, b row k] as 8 x i16 (packs pairs the lanes, the permute regroups).
                let r16 = _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0b00_00_10_00>(
                    _mm256_packs_epi32(res, res),
                ));
                let p = dst.add((half * 4 + k) * ds);
                let px = _mm_cvtepu8_epi16(_mm_loadl_epi64(p as *const __m128i));
                let sum = _mm_packus_epi16(_mm_add_epi16(px, r16), r16);
                _mm_storel_epi64(p as *mut __m128i, sum);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dequant_neon(blocks: &mut [[i16; 16]; 4], scale: &[i32; 16]) {
    use std::arch::aarch64::*;
    unsafe {
        let s: [int32x4_t; 4] = std::array::from_fn(|k| vld1q_s32(scale.as_ptr().add(k * 4)));
        for blk in blocks.iter_mut() {
            for h in 0..2 {
                let c = vld1q_s16(blk.as_ptr().add(h * 8));
                let lo = vmulq_s32(vmovl_s16(vget_low_s16(c)), s[h * 2]);
                let hi = vmulq_s32(vmovl_high_s16(c), s[h * 2 + 1]);
                vst1q_s16(
                    blk.as_mut_ptr().add(h * 8),
                    vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi)),
                );
            }
        }
    }
}

/// Adds the inverse transforms of blocks `a` and `b` onto the 8-sample rows of `dst`, `a` on the
/// left half.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn idct_add_pair_neon(dst: *mut u8, ds: usize, a: &[i16; 16], b: &[i16; 16]) {
    use std::arch::aarch64::*;
    fn transpose(r: [int32x4_t; 4]) -> [int32x4_t; 4] {
        // Safety: NEON is baseline on aarch64.
        unsafe {
            let t0 = vtrnq_s32(r[0], r[1]);
            let t1 = vtrnq_s32(r[2], r[3]);
            [
                vcombine_s32(vget_low_s32(t0.0), vget_low_s32(t1.0)),
                vcombine_s32(vget_low_s32(t0.1), vget_low_s32(t1.1)),
                vcombine_s32(vget_high_s32(t0.0), vget_high_s32(t1.0)),
                vcombine_s32(vget_high_s32(t0.1), vget_high_s32(t1.1)),
            ]
        }
    }
    fn butterfly(c: [int32x4_t; 4]) -> [int32x4_t; 4] {
        // Safety: NEON is baseline on aarch64.
        unsafe {
            let e0 = vaddq_s32(c[0], c[2]);
            let e1 = vsubq_s32(c[0], c[2]);
            let e2 = vsubq_s32(vshrq_n_s32::<1>(c[1]), c[3]);
            let e3 = vaddq_s32(c[1], vshrq_n_s32::<1>(c[3]));
            [
                vaddq_s32(e0, e3),
                vaddq_s32(e1, e2),
                vsubq_s32(e1, e2),
                vsubq_s32(e0, e3),
            ]
        }
    }
    unsafe {
        let idct = |blk: &[i16; 16]| {
            let rows: [int32x4_t; 4] =
                std::array::from_fn(|k| vmovl_s16(vld1_s16(blk.as_ptr().add(k * 4))));
            butterfly(transpose(butterfly(transpose(rows))))
        };
        let (ra, rb) = (idct(a), idct(b));
        for k in 0..4 {
            let res = vcombine_s16(vqrshrn_n_s32::<6>(ra[k]), vqrshrn_n_s32::<6>(rb[k]));
            let p = dst.add(k * ds);
            let px = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(p)));
            vst1_u8(p, vqmovun_s16(vqaddq_s16(px, res)));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avg16_avx2(dst: &mut [u8; 256], a: *const u8, sa: usize, b: *const u8, sb: usize) {
    use std::arch::x86_64::*;
    unsafe {
        for r in (0..16).step_by(2) {
            let v = _mm256_avg_epu8(
                x86::load_2x16(a.add(r * sa), sa),
                x86::load_2x16(b.add(r * sb), sb),
            );
            _mm256_storeu_si256(dst.as_mut_ptr().add(r * 16) as *mut __m256i, v);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sad_avg16_avx2(src: &[u8; 256], a: *const u8, sa: usize, b: *const u8, sb: usize) -> u32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut acc = _mm256_setzero_si256();
        for r in (0..16).step_by(2) {
            let v = _mm256_avg_epu8(
                x86::load_2x16(a.add(r * sa), sa),
                x86::load_2x16(b.add(r * sb), sb),
            );
            let s = _mm256_loadu_si256(src.as_ptr().add(r * 16) as *const __m256i);
            acc = _mm256_add_epi64(acc, _mm256_sad_epu8(s, v));
        }
        x86::hsum_epi64(acc)
    }
}

/// `a - 5 b + 20 c + 20 d - 5 e + f` on 16-bit lanes, as `(a + f) + 5 (4 (c + d) - (b + e))`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn tap6_epi16(t: [std::arch::x86_64::__m256i; 6]) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let s0 = _mm256_add_epi16(t[0], t[5]);
    let s1 = _mm256_add_epi16(t[1], t[4]);
    let s2 = _mm256_add_epi16(t[2], t[3]);
    let k = _mm256_sub_epi16(_mm256_slli_epi16::<2>(s2), s1);
    _mm256_add_epi16(s0, _mm256_add_epi16(k, _mm256_slli_epi16::<2>(k)))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn plane_avx2<const N: usize>(a: i32, b: i32, c: i32, dst: &mut [u8]) {
    use std::arch::x86_64::*;
    unsafe {
        let c0 = (N / 2 - 1) as i32;
        // Lane x holds b (x - c0) for N = 16; for N = 8 the two 128-bit lanes hold two rows.
        let bx: [i16; 16] = std::array::from_fn(|i| (b * ((i % N) as i32 - c0)) as i16);
        let bx = _mm256_loadu_si256(bx.as_ptr() as *const __m256i);
        let row_base = |y: usize| (a + c * (y as i32 - c0) + 16) as i16;
        let out = dst.as_mut_ptr();
        if N == 16 {
            for y in 0..16 {
                let v =
                    _mm256_srai_epi16::<5>(_mm256_add_epi16(bx, _mm256_set1_epi16(row_base(y))));
                // packus saturates to 0..=255; the permute restores the lane order.
                let p = _mm256_permute4x64_epi64::<0b00_00_10_00>(_mm256_packus_epi16(v, v));
                _mm_storeu_si128(out.add(y * 16) as *mut __m128i, _mm256_castsi256_si128(p));
            }
        } else {
            for y in (0..8).step_by(2) {
                let base =
                    _mm256_setr_m128i(_mm_set1_epi16(row_base(y)), _mm_set1_epi16(row_base(y + 1)));
                let v = _mm256_srai_epi16::<5>(_mm256_add_epi16(bx, base));
                let p = _mm256_permute4x64_epi64::<0b00_00_10_00>(_mm256_packus_epi16(v, v));
                _mm_storeu_si128(out.add(y * 8) as *mut __m128i, _mm256_castsi256_si128(p));
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn plane_neon<const N: usize>(a: i32, b: i32, c: i32, dst: &mut [u8]) {
    use std::arch::aarch64::*;
    unsafe {
        let c0 = (N / 2 - 1) as i32;
        let bx: [i16; 16] = std::array::from_fn(|i| (b * (i as i32 - c0)) as i16);
        let (lo, hi) = (vld1q_s16(bx.as_ptr()), vld1q_s16(bx.as_ptr().add(8)));
        for y in 0..N {
            let base = vdupq_n_s16((a + c * (y as i32 - c0) + 16) as i16);
            let out = dst.as_mut_ptr().add(y * N);
            vst1_u8(out, vqshrun_n_s16::<5>(vaddq_s16(lo, base)));
            if N == 16 {
                vst1_u8(out.add(8), vqshrun_n_s16::<5>(vaddq_s16(hi, base)));
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pred4_dir_avx2(edge: &[u8; 16]) -> [[u8; 16]; 6] {
    use std::arch::x86_64::*;
    unsafe {
        let e = _mm_loadu_si128(edge.as_ptr() as *const __m128i);
        let (e1, e2) = (_mm_srli_si128::<1>(e), _mm_srli_si128::<2>(e));
        let avg2 = _mm_avg_epu8(e, e1);
        // (x + 2 y + z + 2) >> 2 = avg(avg(x, z) - ((x ^ z) & 1), y), exactly.
        let xz = _mm_sub_epi8(
            _mm_avg_epu8(e, e2),
            _mm_and_si128(_mm_xor_si128(e, e2), _mm_set1_epi8(1)),
        );
        let avg3 = _mm_avg_epu8(xz, e1);
        let mut out = [[0u8; 16]; 6];
        for (o, idx) in out.iter_mut().zip(&PRED4_DIR) {
            let idx = _mm_loadu_si128(idx.as_ptr() as *const __m128i);
            // Indices 16.. select avg3: pshufb zeroes a lane whose index has the top bit set.
            let lo = _mm_shuffle_epi8(
                avg2,
                _mm_or_si128(idx, _mm_cmpgt_epi8(idx, _mm_set1_epi8(15))),
            );
            let hi = _mm_shuffle_epi8(
                avg3,
                _mm_or_si128(
                    _mm_sub_epi8(idx, _mm_set1_epi8(16)),
                    _mm_cmplt_epi8(idx, _mm_set1_epi8(16)),
                ),
            );
            _mm_storeu_si128(o.as_mut_ptr() as *mut __m128i, _mm_or_si128(lo, hi));
        }
        out
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn pred4_dir_neon(edge: &[u8; 16]) -> [[u8; 16]; 6] {
    use std::arch::aarch64::*;
    unsafe {
        let e = vld1q_u8(edge.as_ptr());
        let z = vdupq_n_u8(0);
        let (e1, e2) = (vextq_u8::<1>(e, z), vextq_u8::<2>(e, z));
        let avg2 = vrhaddq_u8(e, e1);
        let xz = vsubq_u8(vrhaddq_u8(e, e2), vandq_u8(veorq_u8(e, e2), vdupq_n_u8(1)));
        let avg3 = vrhaddq_u8(xz, e1);
        let t = uint8x16x2_t(avg2, avg3);
        let mut out = [[0u8; 16]; 6];
        for (o, idx) in out.iter_mut().zip(&PRED4_DIR) {
            vst1q_u8(o.as_mut_ptr(), vqtbl2q_u8(t, vld1q_u8(idx.as_ptr())));
        }
        out
    }
}

/// [`chroma_mc_8x8_x2`]: `u` rows in the low lane, `v` rows in the high lane; each row's
/// sample pairs `(A, B)` meet their weights in one `pmaddubsw`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn chroma_mc_avx2(
    u: *const u8,
    v: *const u8,
    ss: usize,
    fx: u8,
    fy: u8,
    out_u: &mut [u8; 64],
    out_v: &mut [u8; 64],
) {
    use std::arch::x86_64::*;
    unsafe {
        let (fx, fy) = (fx as i16, fy as i16);
        let pair = |a: i16, b: i16| _mm256_set1_epi16((b << 8) | a);
        let top = pair((8 - fx) * (8 - fy), fx * (8 - fy));
        let bottom = pair((8 - fx) * fy, fx * fy);
        let r32 = _mm256_set1_epi16(32);
        // The (A, B) byte pairs of row r of both planes.
        let pairs = |r: usize| {
            let x = _mm256_inserti128_si256::<1>(
                _mm256_castsi128_si256(_mm_loadu_si128(u.add(r * ss) as *const __m128i)),
                _mm_loadu_si128(v.add(r * ss) as *const __m128i),
            );
            _mm256_unpacklo_epi8(x, _mm256_srli_si256::<1>(x))
        };
        let mut cur = pairs(0);
        for r in 0..8 {
            let next = pairs(r + 1);
            let sum = _mm256_add_epi16(
                _mm256_maddubs_epi16(cur, top),
                _mm256_maddubs_epi16(next, bottom),
            );
            let x = _mm256_srli_epi16::<6>(_mm256_add_epi16(sum, r32));
            let packed = _mm256_packus_epi16(x, x);
            _mm_storel_epi64(
                out_u.as_mut_ptr().add(r * 8) as *mut __m128i,
                _mm256_castsi256_si128(packed),
            );
            _mm_storel_epi64(
                out_v.as_mut_ptr().add(r * 8) as *mut __m128i,
                _mm256_extracti128_si256::<1>(packed),
            );
            cur = next;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hv_avx2(
    p: *const u8,
    h0: usize,
    v0: usize,
    ss: usize,
    out_h: &mut [u8; 256],
    out_v: &mut [u8; 256],
) {
    use std::arch::x86_64::*;
    unsafe {
        let r16 = _mm256_set1_epi16(16);
        let at = |q: *const u8| _mm256_cvtepu8_epi16(_mm_loadu_si128(q as *const __m128i));
        let store = |dst: *mut u8, x: __m256i| {
            let x = _mm256_srai_epi16::<5>(_mm256_add_epi16(x, r16));
            let packed = _mm256_permute4x64_epi64::<0b00_00_10_00>(_mm256_packus_epi16(x, x));
            _mm_storeu_si128(dst as *mut __m128i, _mm256_castsi256_si128(packed));
        };
        for r in 0..16 {
            let q = p.add(h0 + r * ss);
            let h = tap6_epi16([
                at(q.sub(2)),
                at(q.sub(1)),
                at(q),
                at(q.add(1)),
                at(q.add(2)),
                at(q.add(3)),
            ]);
            store(out_h.as_mut_ptr().add(r * 16), h);
        }
        // The vertical taps slide down six rows held in registers.
        let mut rows: [__m256i; 6] = std::array::from_fn(|k| {
            if k < 5 {
                at(p.add(v0 + k * ss - 2 * ss))
            } else {
                _mm256_setzero_si256()
            }
        });
        for r in 0..16 {
            rows[5] = at(p.add(v0 + (r + 3) * ss));
            store(out_v.as_mut_ptr().add(r * 16), tap6_epi16(rows));
            rows = [rows[1], rows[2], rows[3], rows[4], rows[5], rows[5]];
        }
    }
}

/// Rows of the second (packed) half of an AVX2 window pass.
#[cfg(target_arch = "x86_64")]
const HPEL_HALF: usize = HPEL_WIN_H / 2;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hpel_avx2(p: *const u8, ss: usize, out: &mut HpelWindow) {
    use std::arch::x86_64::*;
    unsafe {
        let r16 = _mm256_set1_epi16(16);
        // The `h` and `v` planes of `OUT` register rows from `ld(row, column)` (16 widened
        // samples): `st(plane, row, values)` stores a plane row and `keep(k, taps)` the
        // horizontal taps of row k - 2; rows -2..OUT + 3 are read.
        macro_rules! pass {
            ($out:expr, $c0:expr, $ld:expr, $st:expr, $keep:expr) => {{
                const ROWS: usize = $out + 5;
                let (ld, st, keep) = ($ld, $st, $keep);
                let mut sr = [_mm256_setzero_si256(); ROWS];
                for (k, s) in sr.iter_mut().enumerate() {
                    let r = k as isize - 2;
                    let h = tap6_epi16(std::array::from_fn(|t| ld(r, $c0 - 2 + t as isize)));
                    keep(k, h);
                    if (2..$out + 2).contains(&k) {
                        st(0, k - 2, _mm256_srai_epi16::<5>(_mm256_add_epi16(h, r16)));
                    }
                    *s = ld(r, $c0);
                }
                for r in 0..$out {
                    let v =
                        tap6_epi16([sr[r], sr[r + 1], sr[r + 2], sr[r + 3], sr[r + 4], sr[r + 5]]);
                    st(1, r, _mm256_srai_epi16::<5>(_mm256_add_epi16(v, r16)));
                }
            }};
        }
        let planes = [out.h.as_mut_ptr(), out.v.as_mut_ptr()];
        let hr = out.hr.as_mut_ptr();
        let row = |r: isize, c: isize| p.offset(r * ss as isize + c);
        // Columns 0..16, one row per register.
        pass!(
            HPEL_WIN_H,
            0,
            |r: isize, c: isize| _mm256_cvtepu8_epi16(_mm_loadu_si128(row(r, c) as *const __m128i)),
            |plane: usize, r: usize, x: __m256i| {
                let packed = _mm256_permute4x64_epi64::<0b00_00_10_00>(_mm256_packus_epi16(x, x));
                let dst = planes[plane].add(r * HPEL_WIN_W) as *mut __m128i;
                _mm_storeu_si128(dst, _mm256_castsi256_si128(packed));
            },
            |k: usize, h: __m256i| {
                _mm256_storeu_si256(hr.add(k * HPEL_COLS) as *mut __m256i, h);
            }
        );
        // Columns 16..24: rows r and r + HPEL_HALF side by side in the two 128-bit lanes.
        pass!(
            HPEL_HALF,
            16,
            |r: isize, c: isize| {
                let lo = _mm_loadl_epi64(row(r, c) as *const __m128i);
                let hi = _mm_loadl_epi64(row(r + HPEL_HALF as isize, c) as *const __m128i);
                _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(lo, hi))
            },
            |plane: usize, r: usize, x: __m256i| {
                let packed = _mm256_packus_epi16(x, x);
                let dst = planes[plane].add(r * HPEL_WIN_W + 16);
                _mm_storel_epi64(dst as *mut __m128i, _mm256_castsi256_si128(packed));
                _mm_storel_epi64(
                    dst.add(HPEL_HALF * HPEL_WIN_W) as *mut __m128i,
                    _mm256_extracti128_si256::<1>(packed),
                );
            },
            |k: usize, h: __m256i| {
                let dst = hr.add(k * HPEL_COLS + 16);
                _mm_storeu_si128(dst as *mut __m128i, _mm256_castsi256_si128(h));
                _mm_storeu_si128(
                    dst.add(HPEL_HALF * HPEL_COLS) as *mut __m128i,
                    _mm256_extracti128_si256::<1>(h),
                );
            }
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hpel_centre_avx2(out: &mut HpelWindow) {
    use std::arch::x86_64::*;
    unsafe {
        let r512 = _mm256_set1_epi32(512);
        let (one, m5, p20) = (
            _mm256_set1_epi16(1),
            _mm256_set1_epi16(-5),
            _mm256_set1_epi16(20),
        );
        // The vertical tap on the unrounded intermediates needs 32 bits: multiply-add the
        // symmetric pairs, interleaved per 128-bit lane (packs restores the order).
        let centre = |t: [__m256i; 6]| {
            let pairs = [(t[0], t[5], one), (t[1], t[4], m5), (t[2], t[3], p20)];
            let (mut lo, mut hi) = (r512, r512);
            for (a, b, k) in pairs {
                lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), k));
                hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), k));
            }
            let (lo, hi) = (_mm256_srai_epi32::<10>(lo), _mm256_srai_epi32::<10>(hi));
            let j = _mm256_packs_epi32(lo, hi);
            _mm256_packus_epi16(j, j)
        };
        let hr = out.hr.as_ptr();
        let c = out.c.as_mut_ptr();
        let full: [__m256i; HPEL_HR_ROWS] =
            std::array::from_fn(|k| _mm256_loadu_si256(hr.add(k * HPEL_COLS) as *const __m256i));
        for r in 0..HPEL_WIN_H {
            let packed =
                _mm256_permute4x64_epi64::<0b00_00_10_00>(centre(std::array::from_fn(|t| {
                    full[r + t]
                })));
            _mm_storeu_si128(
                c.add(r * HPEL_WIN_W) as *mut __m128i,
                _mm256_castsi256_si128(packed),
            );
        }
        let half: [__m256i; HPEL_HALF + 5] = std::array::from_fn(|k| {
            let lo = _mm_loadu_si128(hr.add(k * HPEL_COLS + 16) as *const __m128i);
            let hi = _mm_loadu_si128(hr.add((k + HPEL_HALF) * HPEL_COLS + 16) as *const __m128i);
            _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi)
        });
        for r in 0..HPEL_HALF {
            let packed = centre(std::array::from_fn(|t| half[r + t]));
            let dst = c.add(r * HPEL_WIN_W + 16);
            _mm_storel_epi64(dst as *mut __m128i, _mm256_castsi256_si128(packed));
            _mm_storel_epi64(
                dst.add(HPEL_HALF * HPEL_WIN_W) as *mut __m128i,
                _mm256_extracti128_si256::<1>(packed),
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn hpel_neon(p: *const u8, ss: usize, out: &mut HpelWindow) {
    use std::arch::aarch64::*;
    fn tap6(t: [int16x8_t; 6]) -> int16x8_t {
        // Safety: NEON is baseline on aarch64.
        unsafe {
            let s0 = vaddq_s16(t[0], t[5]);
            let s1 = vaddq_s16(t[1], t[4]);
            let s2 = vaddq_s16(t[2], t[3]);
            let k = vsubq_s16(vshlq_n_s16::<2>(s2), s1);
            vaddq_s16(s0, vaddq_s16(k, vshlq_n_s16::<2>(k)))
        }
    }
    unsafe {
        let ld = |r: isize, c: isize| {
            vreinterpretq_s16_u16(vmovl_u8(vld1_u8(p.offset(r * ss as isize + c))))
        };
        for chunk in 0..HPEL_COLS / 8 {
            let c0 = (chunk * 8) as isize;
            let mut sr = [vdupq_n_s16(0); HPEL_HR_ROWS];
            for (k, s) in sr.iter_mut().enumerate() {
                let r = k as isize - 2;
                let h = tap6(std::array::from_fn(|t| ld(r, c0 - 2 + t as isize)));
                vst1q_s16(out.hr.as_mut_ptr().add(k * HPEL_COLS + chunk * 8), h);
                if (2..HPEL_WIN_H + 2).contains(&k) {
                    let o = (k - 2) * HPEL_WIN_W + chunk * 8;
                    vst1_u8(out.h.as_mut_ptr().add(o), vqrshrun_n_s16::<5>(h));
                }
                *s = ld(r, c0);
            }
            for r in 0..HPEL_WIN_H {
                let o = r * HPEL_WIN_W + chunk * 8;
                let v = tap6([sr[r], sr[r + 1], sr[r + 2], sr[r + 3], sr[r + 4], sr[r + 5]]);
                vst1_u8(out.v.as_mut_ptr().add(o), vqrshrun_n_s16::<5>(v));
            }
        }
    }
}

/// One plane of [`chroma_mc_8x8_x2`].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn chroma_mc_neon(src: *const u8, ss: usize, fx: u8, fy: u8, out: &mut [u8; 64]) {
    use std::arch::aarch64::*;
    unsafe {
        let w = |x: u8| vdup_n_u8(x);
        let (wa, wb) = (w((8 - fx) * (8 - fy)), w(fx * (8 - fy)));
        let (wc, wd) = (w((8 - fx) * fy), w(fx * fy));
        let row = |r: usize| {
            let x = vld1q_u8(src.add(r * ss));
            (vget_low_u8(x), vget_low_u8(vextq_u8::<1>(x, x)))
        };
        let mut cur = row(0);
        for r in 0..8 {
            let next = row(r + 1);
            let mut acc = vmull_u8(cur.0, wa);
            acc = vmlal_u8(acc, cur.1, wb);
            acc = vmlal_u8(acc, next.0, wc);
            acc = vmlal_u8(acc, next.1, wd);
            vst1_u8(out.as_mut_ptr().add(r * 8), vrshrn_n_u16::<6>(acc));
            cur = next;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn hv_neon(
    p: *const u8,
    h0: usize,
    v0: usize,
    ss: usize,
    out_h: &mut [u8; 256],
    out_v: &mut [u8; 256],
) {
    use std::arch::aarch64::*;
    fn tap6(t: [int16x8_t; 6]) -> int16x8_t {
        // Safety: NEON is baseline on aarch64.
        unsafe {
            let s0 = vaddq_s16(t[0], t[5]);
            let s1 = vaddq_s16(t[1], t[4]);
            let s2 = vaddq_s16(t[2], t[3]);
            let k = vsubq_s16(vshlq_n_s16::<2>(s2), s1);
            vaddq_s16(s0, vaddq_s16(k, vshlq_n_s16::<2>(k)))
        }
    }
    unsafe {
        let at = |q: *const u8| vreinterpretq_s16_u16(vmovl_u8(vld1_u8(q)));
        for half in 0..2 {
            for r in 0..16 {
                let q = p.add(h0 + r * ss + half * 8);
                let h = tap6(std::array::from_fn(|k| at(q.add(k).sub(2))));
                vst1_u8(
                    out_h.as_mut_ptr().add(r * 16 + half * 8),
                    vqrshrun_n_s16::<5>(h),
                );
                let q = p.add(v0 + r * ss + half * 8);
                let v = tap6(std::array::from_fn(|k| at(q.add(k * ss).sub(2 * ss))));
                vst1_u8(
                    out_v.as_mut_ptr().add(r * 16 + half * 8),
                    vqrshrun_n_s16::<5>(v),
                );
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn hpel_centre_neon(out: &mut HpelWindow) {
    use std::arch::aarch64::*;
    unsafe {
        for chunk in 0..HPEL_COLS / 8 {
            let hr: [int16x8_t; HPEL_HR_ROWS] =
                std::array::from_fn(|k| vld1q_s16(out.hr.as_ptr().add(k * HPEL_COLS + chunk * 8)));
            for r in 0..HPEL_WIN_H {
                let j = |lo: bool| {
                    let half = |x: int16x8_t| {
                        if lo {
                            vget_low_s16(x)
                        } else {
                            vget_high_s16(x)
                        }
                    };
                    let s0 = vaddl_s16(half(hr[r]), half(hr[r + 5]));
                    let s1 = vaddl_s16(half(hr[r + 1]), half(hr[r + 4]));
                    let s2 = vaddl_s16(half(hr[r + 2]), half(hr[r + 3]));
                    let sum = vmlaq_n_s32(vmlaq_n_s32(s0, s1, -5), s2, 20);
                    vqrshrn_n_s32::<10>(sum)
                };
                let jc = vcombine_s16(j(true), j(false));
                let o = r * HPEL_WIN_W + chunk * 8;
                vst1_u8(out.c.as_mut_ptr().add(o), vqmovun_s16(jc));
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn zigzag_avx2(blocks: &[[i16; 16]; 4]) -> [[i16; 16]; 4] {
    use std::arch::x86_64::*;
    // Byte shuffles of each 128-bit lane: from the lane itself, and from the other half of the
    // block (positions 8 and 7 cross halves); -1 selects zero.
    const OWN: [i8; 32] = [
        0, 1, 2, 3, 8, 9, -1, -1, 10, 11, 4, 5, 6, 7, 12, 13, //
        2, 3, 8, 9, 10, 11, 4, 5, -1, -1, 6, 7, 12, 13, 14, 15,
    ];
    const SWAP: [i8; 32] = [
        -1, -1, -1, -1, -1, -1, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1, //
        -1, -1, -1, -1, -1, -1, -1, -1, 14, 15, -1, -1, -1, -1, -1, -1,
    ];
    unsafe {
        let own = _mm256_loadu_si256(OWN.as_ptr() as *const __m256i);
        let swap = _mm256_loadu_si256(SWAP.as_ptr() as *const __m256i);
        let mut out = [[0i16; 16]; 4];
        for (o, b) in out.iter_mut().zip(blocks) {
            let x = _mm256_loadu_si256(b.as_ptr() as *const __m256i);
            let y = _mm256_permute2x128_si256::<0x01>(x, x);
            let z = _mm256_or_si256(_mm256_shuffle_epi8(x, own), _mm256_shuffle_epi8(y, swap));
            _mm256_storeu_si256(o.as_mut_ptr() as *mut __m256i, z);
        }
        out
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn zigzag_neon(blocks: &[[i16; 16]; 4]) -> [[i16; 16]; 4] {
    use std::arch::aarch64::*;
    let idx: [u8; 32] = std::array::from_fn(|k| (2 * ZIGZAG4[k / 2] + (k & 1)) as u8);
    unsafe {
        let (i0, i1) = (vld1q_u8(idx.as_ptr()), vld1q_u8(idx.as_ptr().add(16)));
        let mut out = [[0i16; 16]; 4];
        for (o, b) in out.iter_mut().zip(blocks) {
            let p = b.as_ptr() as *const u8;
            let t = uint8x16x2_t(vld1q_u8(p), vld1q_u8(p.add(16)));
            let q = o.as_mut_ptr() as *mut u8;
            vst1q_u8(q, vqtbl2q_u8(t, i0));
            vst1q_u8(q.add(16), vqtbl2q_u8(t, i1));
        }
        out
    }
}

/// NEON counterpart of [`dct_strip_avx2`] for two horizontally adjacent blocks whose rows start
/// `row_off` rows below `src` / `pred`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dct_strip_neon<const HAD: bool>(
    src: *const u8,
    ss: usize,
    pred: *const u8,
    ps: usize,
    row_off: usize,
    out: &mut [[i16; 16]],
) {
    use std::arch::aarch64::*;
    fn butterfly<const HAD: bool>(d: [int16x8_t; 4]) -> [int16x8_t; 4] {
        // Safety: NEON is baseline on aarch64.
        unsafe {
            if HAD {
                let (s01, d01) = (vaddq_s16(d[0], d[1]), vsubq_s16(d[0], d[1]));
                let (s23, d23) = (vaddq_s16(d[2], d[3]), vsubq_s16(d[2], d[3]));
                return [
                    vaddq_s16(s01, s23),
                    vsubq_s16(s01, s23),
                    vaddq_s16(d01, d23),
                    vsubq_s16(d01, d23),
                ];
            }
            let (s03, s12) = (vaddq_s16(d[0], d[3]), vaddq_s16(d[1], d[2]));
            let (d03, d12) = (vsubq_s16(d[0], d[3]), vsubq_s16(d[1], d[2]));
            [
                vaddq_s16(s03, s12),
                vaddq_s16(vshlq_n_s16::<1>(d03), d12),
                vsubq_s16(s03, s12),
                vsubq_s16(d03, vshlq_n_s16::<1>(d12)),
            ]
        }
    }
    fn transpose(v: [int16x8_t; 4]) -> [int16x8_t; 4] {
        // Safety: NEON is baseline on aarch64.
        unsafe {
            let t01 = vtrnq_s16(v[0], v[1]);
            let t23 = vtrnq_s16(v[2], v[3]);
            let e = vtrnq_s32(vreinterpretq_s32_s16(t01.0), vreinterpretq_s32_s16(t23.0));
            let o = vtrnq_s32(vreinterpretq_s32_s16(t01.1), vreinterpretq_s32_s16(t23.1));
            [
                vreinterpretq_s16_s32(e.0),
                vreinterpretq_s16_s32(o.0),
                vreinterpretq_s16_s32(e.1),
                vreinterpretq_s16_s32(o.1),
            ]
        }
    }
    unsafe {
        let row = |k: usize| {
            let r = row_off + k;
            vreinterpretq_s16_u16(vsubl_u8(
                vld1_u8(src.add(r * ss)),
                vld1_u8(pred.add(r * ps)),
            ))
        };
        let v = butterfly::<HAD>([row(0), row(1), row(2), row(3)]);
        let r = transpose(butterfly::<HAD>(transpose(v)));
        for (h, blk) in out.iter_mut().enumerate().take(2) {
            for (k, rv) in r.iter().enumerate() {
                let half = if h == 0 {
                    vget_low_s16(*rv)
                } else {
                    vget_high_s16(*rv)
                };
                vst1_s16(blk.as_mut_ptr().add(k * 4), half);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn quant_neon(blocks: &mut [[i16; 16]; 4], q: &Quant4x4) -> u32 {
    use std::arch::aarch64::*;
    unsafe {
        let bias = vdupq_n_u32(q.bias);
        let shift = vdupq_n_s32(-(q.shift as i32));
        let cap = vdupq_n_u32(i16::MAX as u32);
        let mut nz = 0;
        for (b, blk) in blocks.iter_mut().enumerate() {
            let mut max = vdupq_n_u16(0);
            for h in 0..2 {
                let c = vld1q_s16(blk.as_ptr().add(h * 8));
                let mf = vld1q_u16(q.mf.as_ptr().add(h * 8));
                let a = vreinterpretq_u16_s16(vabsq_s16(c));
                let lo = vshlq_u32(
                    vaddq_u32(vmull_u16(vget_low_u16(a), vget_low_u16(mf)), bias),
                    shift,
                );
                let hi = vshlq_u32(vaddq_u32(vmull_high_u16(a, mf), bias), shift);
                let l = vreinterpretq_s16_u16(vcombine_u16(
                    vmovn_u32(vminq_u32(lo, cap)),
                    vmovn_u32(vminq_u32(hi, cap)),
                ));
                let level = vbslq_s16(vcltzq_s16(c), vnegq_s16(l), l);
                vst1q_s16(blk.as_mut_ptr().add(h * 8), level);
                max = vmaxq_u16(max, vreinterpretq_u16_s16(l));
            }
            match vmaxvq_u16(max) {
                0 => {}
                1 => nz |= 1 << b,
                _ => nz |= 0x11 << b,
            }
        }
        nz
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sad_neon<const W: usize, const H: usize>(
    a: *const u8,
    sa: usize,
    b: *const u8,
    sb: usize,
) -> u32 {
    use std::arch::aarch64::*;
    unsafe {
        let mut acc = vdupq_n_u16(0);
        for y in 0..H {
            if W == 16 {
                let (va, vb) = (vld1q_u8(a.add(y * sa)), vld1q_u8(b.add(y * sb)));
                acc = vabal_u8(acc, vget_low_u8(va), vget_low_u8(vb));
                acc = vabal_high_u8(acc, va, vb);
            } else {
                acc = vabal_u8(acc, vld1_u8(a.add(y * sa)), vld1_u8(b.add(y * sb)));
            }
        }
        vaddlvq_u16(acc)
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn satd16_neon(a: *const u8, sa: usize, b: *const u8, sb: usize) -> u32 {
    use std::arch::aarch64::*;
    unsafe {
        let odd: [u16; 8] = [0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF, 0, 0xFFFF];
        let odd = vld1q_u16(odd.as_ptr());
        let mut acc = vdupq_n_u32(0);
        for strip in 0..4 {
            for half in 0..2 {
                let row = |k: usize| {
                    let r = strip * 4 + k;
                    let va = vld1_u8(a.add(r * sa + half * 8));
                    let vb = vld1_u8(b.add(r * sb + half * 8));
                    vreinterpretq_s16_u16(vsubl_u8(va, vb))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic noise with runs of extremes, so differences reach both ends of their range.
    fn plane(w: usize, h: usize, seed: u32) -> Vec<u8> {
        let mut x = seed;
        (0..w * h)
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
    fn sad_and_satd_match_scalar_at_odd_strides_and_offsets() {
        for &(pw, ph) in &[(16usize, 16usize), (37, 23), (61, 40)] {
            let a = plane(pw, ph, 3);
            let b = plane(pw, ph, 11);
            let flat = vec![255u8; pw * ph];
            let zero = vec![0u8; pw * ph];
            for &(x, y) in &[(0usize, 0usize), (1, 3), (pw - 16, ph - 16)] {
                if x + 21 > pw || y + 16 > ph {
                    continue;
                }
                let o = y * pw + x;
                for (p, q) in [(&a, &b), (&flat, &zero), (&zero, &flat), (&a, &a)] {
                    assert_eq!(
                        pixel_sad_16x16(&p[o..], pw, &q[o..], pw),
                        pixel_sad_16x16_scalar(&p[o..], pw, &q[o..], pw)
                    );
                    assert_eq!(
                        pixel_satd_16x16(&p[o..], pw, &q[o..], pw),
                        pixel_satd_16x16_scalar(&p[o..], pw, &q[o..], pw),
                        "{pw}x{ph} at ({x}, {y})"
                    );
                    assert_eq!(
                        pixel_sad_8x8(&p[o + 3..], pw, &q[o + 5..], pw),
                        pixel_sad_8x8_scalar(&p[o + 3..], pw, &q[o + 5..], pw)
                    );
                    assert_eq!(
                        pixel_satd_4x4(&p[o + 1..], pw, &q[o + 7..], pw),
                        pixel_satd_4x4_scalar(&p[o + 1..], pw, &q[o + 7..], pw)
                    );
                    assert_eq!(
                        pixel_ssd::<16>(&p[o..], pw, &q[o..], pw),
                        pixel_ssd_scalar::<16>(&p[o..], pw, &q[o..], pw)
                    );
                    assert_eq!(
                        pixel_ssd::<8>(&p[o + 3..], pw, &q[o + 5..], pw),
                        pixel_ssd_scalar::<8>(&p[o + 3..], pw, &q[o + 5..], pw)
                    );
                }
            }
        }
        // A saturated difference in every coefficient's direction.
        let ones = [255u8; 256];
        let checker: Vec<u8> = (0..256)
            .map(|i| if (i ^ (i >> 4)) & 1 == 0 { 255 } else { 0 })
            .collect();
        assert_eq!(
            pixel_satd_16x16(&ones, 16, &checker, 16),
            pixel_satd_16x16_scalar(&ones, 16, &checker, 16)
        );
    }

    #[test]
    fn plane_prediction_matches_scalar_at_the_16_bit_extremes() {
        for (a, b, c) in [
            (16 * 510, 717i32, 717),
            (16 * 510, -717, -717),
            (0, 717, -717),
            (0, -717, 717),
            (4000, 3, -5),
            (16 * 510, 1355, 1355),
            (0, -1355, -1355),
        ] {
            let (mut s16, mut r16) = ([0u8; 256], [0u8; 256]);
            if b.abs() <= 717 {
                intra_pred_plane::<16>(a, b, c, &mut s16);
                intra_pred_plane_scalar::<16>(a, b, c, &mut r16);
                assert_eq!(s16, r16, "16x16 {a} {b} {c}");
            }
            let (mut s8, mut r8) = ([0u8; 64], [0u8; 64]);
            intra_pred_plane::<8>(a, b, c, &mut s8);
            intra_pred_plane_scalar::<8>(a, b, c, &mut r8);
            assert_eq!(s8, r8, "8x8 {a} {b} {c}");
        }
    }

    #[test]
    fn directional_4x4_predictions_match_the_generic_predictor() {
        let e = plane(13, 40, 51);
        for n in 0..40 {
            let r = &e[n * 13..n * 13 + 13];
            let top: [u8; 8] = r[..8].try_into().unwrap();
            let left: [u8; 4] = r[8..12].try_into().unwrap();
            let want = intra4x4_pred_dir_scalar(&top, &left, r[12]);
            assert_eq!(intra4x4_pred_dir(&top, &left, r[12]), want, "edge {r:?}");
            let edge = [
                left[3], left[3], left[2], left[1], left[0], r[12], top[0], top[1], top[2], top[3],
                top[4], top[5], top[6], top[7], top[7], top[7],
            ];
            assert_eq!(pred4_dir_from_edge(&edge), want);
        }
        for v in [0u8, 255] {
            assert_eq!(
                intra4x4_pred_dir(&[v; 8], &[255 - v; 4], v),
                intra4x4_pred_dir_scalar(&[v; 8], &[255 - v; 4], v)
            );
        }
    }

    #[test]
    fn intra_satd_x3_matches_explicit_predictions() {
        let a = plane(16, 16, 31);
        let e = plane(40, 1, 32);
        let src: [u8; 256] = a.try_into().unwrap();
        let cases: [([u8; 16], [u8; 16], u8); 4] = [
            (
                e[..16].try_into().unwrap(),
                e[20..36].try_into().unwrap(),
                77,
            ),
            ([255; 16], [0; 16], 255),
            ([0; 16], [255; 16], 0),
            (
                e[3..19].try_into().unwrap(),
                e[1..17].try_into().unwrap(),
                128,
            ),
        ];
        for (top, left, dc) in cases {
            let preds: [[u8; 256]; 3] = [
                std::array::from_fn(|i| top[i % 16]),
                std::array::from_fn(|i| left[i / 16]),
                [dc; 256],
            ];
            let want = preds.map(|p| pixel_satd_16x16_scalar(&src, 16, &p, 16));
            assert_eq!(intra_satd_x3_16x16_scalar(&src, &top, &left, dc), want);
            assert_eq!(intra_satd_x3_16x16(&src, &top, &left, dc), want);
            for (flat, other) in [([255u8; 256], 0u8), ([0; 256], 255)] {
                let want = [
                    pixel_satd_16x16_scalar(&flat, 16, &[other; 256], 16),
                    pixel_satd_16x16_scalar(&flat, 16, &[other; 256], 16),
                    pixel_satd_16x16_scalar(&flat, 16, &[other; 256], 16),
                ];
                assert_eq!(
                    intra_satd_x3_16x16(&flat, &[other; 16], &[other; 16], other),
                    want
                );
            }
        }
    }

    #[test]
    fn residual_transforms_match_scalar() {
        let (pw, ph) = (45usize, 29usize);
        let a = plane(pw, ph, 21);
        let b = plane(pw, ph, 22);
        let (white, black) = (vec![255u8; pw * ph], vec![0u8; pw * ph]);
        for (p, q) in [(&a, &b), (&white, &black), (&black, &white)] {
            for &(x, y) in &[(0usize, 0usize), (3, 1), (pw - 16, ph - 8)] {
                let o = y * pw + x;
                let (mut s, mut r) = ([[0i16; 16]; 4], [[0i16; 16]; 4]);
                sub_dct4x4_16x4(&p[o..], pw, &q[o + 1..], pw, &mut s);
                sub_dct4x4_16x4_scalar(&p[o..], pw, &q[o + 1..], pw, &mut r);
                assert_eq!(s, r, "16x4 at ({x}, {y})");
                sub_dct4x4_8x8(&p[o..], pw, &q[o..], pw, &mut s);
                sub_dct4x4_8x8_scalar(&p[o..], pw, &q[o..], pw, &mut r);
                assert_eq!(s, r, "8x8 at ({x}, {y})");
                // The fused transform equals transforming the difference.
                let mut d = [0i16; 16];
                for k in 0..16 {
                    d[k] = p[o + (k / 4) * pw + k % 4] as i16 - q[o + (k / 4) * pw + k % 4] as i16;
                }
                assert_eq!(r[0], crate::forward_dct4x4_scalar(&d));
                hadamard4x4_16x4(&p[o..], pw, &mut s);
                hadamard4x4_16x4_scalar(&p[o..], pw, &mut r);
                assert_eq!(s, r, "Hadamard at ({x}, {y})");
            }
        }
        // The transforms are linear: SATD is half the sum of the coefficient differences.
        for (p, q) in [(&a, &b), (&white, &black)] {
            let mut sum = 0u32;
            for strip in 0..4 {
                let (mut hp, mut hq) = ([[0i16; 16]; 4], [[0i16; 16]; 4]);
                hadamard4x4_16x4(&p[strip * 4 * pw..], pw, &mut hp);
                hadamard4x4_16x4(&q[strip * 4 * pw + 2..], pw, &mut hq);
                for (bp, bq) in hp.iter().zip(&hq) {
                    sum += bp
                        .iter()
                        .zip(bq)
                        .map(|(&u, &v)| (u as i32 - v as i32).unsigned_abs())
                        .sum::<u32>();
                }
            }
            assert_eq!(sum >> 1, pixel_satd_16x16_scalar(p, pw, &q[2..], pw));
        }
    }

    #[test]
    fn quantization_matches_scalar_at_every_shift_and_extreme() {
        let mut x = 7u32;
        let mut next = || {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            x
        };
        for shift in [15u32, 16, 19, 23, 1, 31] {
            for _ in 0..50 {
                let mf: [u16; 16] = std::array::from_fn(|_| (next() >> 17) as u16);
                let bias = next() % (1 << shift);
                let q = Quant4x4 { mf, bias, shift };
                let mut a = [[0i16; 16]; 4];
                for (i, c) in a.iter_mut().flatten().enumerate() {
                    *c = match i % 7 {
                        0 => i16::MIN,
                        1 => i16::MAX,
                        2 => 0,
                        3 => (next() >> 28) as i16 - 8,
                        _ => (next() >> 16) as i16,
                    };
                }
                a[2] = [0; 16];
                let mut b = a;
                assert_eq!(quant_4x4_x4(&mut a, &q), quant_4x4_x4_scalar(&mut b, &q));
                assert_eq!(a, b, "shift {shift}");
            }
        }
        // Blocks whose levels are all within +-1, or hold one larger level, set only the
        // non-zero bits, or both bits, of their block.
        let q = Quant4x4 {
            mf: [13107; 16],
            bias: (1 << 15) / 6,
            shift: 15,
        };
        for round in 0..200u32 {
            let mut a = [[0i16; 16]; 4];
            for (i, c) in a.iter_mut().flatten().enumerate() {
                let r = next();
                *c = match (r >> 20) % 16 {
                    0..=5 => 0,
                    6..=9 => 2,
                    10..=14 => -3,
                    _ => (r >> 24) as i16 - 128,
                };
                if i == 16 * (round as usize % 4) + (r as usize % 16) && round % 3 == 0 {
                    *c = -60;
                }
            }
            a[(round as usize + 1) % 4] = [0; 16];
            let mut b = a;
            let nz = quant_4x4_x4(&mut a, &q);
            assert_eq!(nz, quant_4x4_x4_scalar(&mut b, &q), "round {round}");
            assert_eq!(a, b);
            for (k, blk) in a.iter().enumerate() {
                assert_eq!(nz >> k & 1 != 0, blk.iter().any(|&c| c != 0));
                assert_eq!(nz >> (4 + k) & 1 != 0, blk.iter().any(|&c| c.abs() > 1));
            }
        }
    }

    #[test]
    fn dequant_and_inverse_transform_match_scalar() {
        let mut x = 99u32;
        let mut next = || {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            x
        };
        for round in 0..400 {
            let scale: [i32; 16] = std::array::from_fn(|i| match (round + i) % 5 {
                0 => 65535,
                1 => -65535,
                _ => (next() >> 16) as i32 - 32768,
            });
            let mut a = [[0i16; 16]; 4];
            for (i, c) in a.iter_mut().flatten().enumerate() {
                *c = match (i + round) % 6 {
                    0 => i16::MIN,
                    1 => i16::MAX,
                    2 => 0,
                    3 => (next() >> 29) as i16 - 4,
                    _ => (next() >> 16) as i16,
                };
            }
            let mut b = a;
            dequant_4x4_x4(&mut a, &scale);
            dequant_4x4_x4_scalar(&mut b, &scale);
            assert_eq!(a, b);

            let ds = 13;
            let mut p: Vec<u8> = (0..ds * 8).map(|_| (next() >> 24) as u8).collect();
            if round % 3 == 0 {
                p.fill(if round % 2 == 0 { 0 } else { 255 });
            }
            let mut q = p.clone();
            add_idct4x4_8x8(&mut p, ds, &a);
            add_idct4x4_8x8_scalar(&mut q, ds, &a);
            assert_eq!(p, q, "round {round}");
        }
    }

    #[test]
    fn chroma_mc_matches_scalar_and_the_generic_kernel() {
        let (pw, ph) = (40usize, 24usize);
        let (u, v) = (plane(pw, ph, 71), plane(pw, ph, 72));
        for (x, y) in [(0usize, 0usize), (5, 3), (17, 9), (24, 6)] {
            let off = y * pw + x;
            for fx in 0..8u8 {
                for fy in 0..8u8 {
                    let (mut a, mut b, mut c, mut d) = ([0u8; 64], [0u8; 64], [0u8; 64], [0u8; 64]);
                    chroma_mc_8x8_x2(&u, &v, off, pw, fx, fy, &mut a, &mut b);
                    chroma_mc_8x8_x2_scalar(&u, &v, off, pw, fx, fy, &mut c, &mut d);
                    assert_eq!((a, b), (c, d), "({x}, {y}) {fx} {fy}");
                    let mut g = [0u8; 64];
                    crate::h264_chroma_mc_scalar(&mut g, 8, &u, off, pw, 8, 8, fx, fy);
                    assert_eq!(a, g);
                    crate::h264_chroma_mc_scalar(&mut g, 8, &v, off, pw, 8, 8, fx, fy);
                    assert_eq!(b, g);
                }
            }
        }
    }

    #[test]
    fn half_pel_blocks_match_scalar_and_the_window() {
        let (pw, ph) = (64usize, 48usize);
        for seed in [61u32, 62] {
            let mut src = plane(pw, ph, seed);
            if seed == 62 {
                for (i, s) in src.iter_mut().enumerate() {
                    *s = if (i / 2) % 2 == 0 { 0 } else { 255 };
                }
            }
            for (x, y) in [(3usize, 3usize), (7, 5), (26, 21)] {
                let off = y * pw + x;
                let mut win = HpelWindow::default();
                hpel_window_scalar(&src, off - pw - 1, pw, &mut win);
                for (left, up) in [(false, false), (true, false), (false, true), (true, true)] {
                    let (mut a, mut b, mut c, mut d) =
                        ([0u8; 256], [0u8; 256], [0u8; 256], [0u8; 256]);
                    hpel_hv_16x16(&src, off, pw, left, up, &mut a, &mut b);
                    hpel_hv_16x16_scalar(&src, off, pw, left, up, &mut c, &mut d);
                    assert_eq!((a, b), (c, d), "({x}, {y}) {left} {up}");
                    // The window starts one sample up and left of the block.
                    for r in 0..16 {
                        for k in 0..16 {
                            let hx = k + usize::from(!left);
                            assert_eq!(a[r * 16 + k], win.h[(r + 1) * HPEL_WIN_W + hx]);
                            let vy = r + usize::from(!up);
                            assert_eq!(b[r * 16 + k], win.v[vy * HPEL_WIN_W + k + 1]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn hpel_window_and_average_match_scalar() {
        let (pw, ph) = (HPEL_WIN_W + 9, HPEL_WIN_H + 7);
        for (seed, flat) in [(31u32, None), (32, Some(0u8)), (33, Some(255u8))] {
            let mut src = plane(pw, ph, seed);
            if let Some(v) = flat {
                // Saturated stripes drive every filter to both clip bounds.
                for (i, s) in src.iter_mut().enumerate() {
                    *s = if (i / 3) % 2 == 0 { v } else { 255 - v };
                }
            }
            for off in [2 * pw + 2, 3 * pw + 4] {
                let (mut a, mut b) = (HpelWindow::default(), HpelWindow::default());
                hpel_window(&src, off, pw, &mut a);
                hpel_window_scalar(&src, off, pw, &mut b);
                assert_eq!(a.hr, b.hr);
                hpel_window_centre(&mut a);
                hpel_window_centre_scalar(&mut b);
                assert_eq!(a.h, b.h);
                assert_eq!(a.v, b.v);
                assert_eq!(a.c, b.c, "seed {seed} off {off}");
            }
            let (mut a, mut b) = ([0u8; 256], [0u8; 256]);
            pixel_avg_16x16(&mut a, &src[1..], pw, &src[pw..], pw);
            pixel_avg_16x16_scalar(&mut b, &src[1..], pw, &src[pw..], pw);
            assert_eq!(a, b);
            let blk: [u8; 256] = std::array::from_fn(|i| src[(i / 16) * pw + i % 16 + 3]);
            assert_eq!(
                pixel_sad_avg_16x16(&blk, &src[1..], pw, &src[pw..], pw),
                pixel_sad_avg_16x16_scalar(&blk, &src[1..], pw, &src[pw..], pw)
            );
            assert_eq!(
                pixel_sad_avg_16x16(&[255; 256], &src[2..], pw, &src[pw + 1..], pw),
                pixel_sad_avg_16x16_scalar(&[255; 256], &src[2..], pw, &src[pw + 1..], pw)
            );
        }
    }

    #[test]
    fn zigzag_matches_scalar() {
        let blocks: [[i16; 16]; 4] =
            std::array::from_fn(|b| std::array::from_fn(|i| (b * 100 + i) as i16 - 150));
        assert_eq!(zigzag_4x4_x4(&blocks), zigzag_4x4_x4_scalar(&blocks));
        let mut sparse = blocks;
        for (b, blk) in sparse.iter_mut().enumerate() {
            for (i, c) in blk.iter_mut().enumerate() {
                if (i * 7 + b) % 3 != 0 {
                    *c = 0;
                }
            }
        }
        sparse[3] = [0; 16];
        sparse[2][15] = i16::MIN;
        assert_eq!(
            nonzero_mask_4x4_x4(&sparse),
            nonzero_mask_4x4_x4_scalar(&sparse)
        );
        assert_eq!(nonzero_mask_4x4_x4(&blocks), [0xFFFF; 4]);
        assert_eq!(
            zigzag_4x4_x4_scalar(&blocks)[0][..4],
            [-150, -149, -146, -142]
        );
    }

    #[test]
    fn sad_x4_matches_scalar() {
        let (pw, ph) = (53usize, 41usize);
        let r = plane(pw, ph, 5);
        let s = plane(16, 16, 9);
        let src: &[u8; 256] = s.as_slice().try_into().unwrap();
        let last = (ph - 16) * pw + pw - 16;
        for offs in [
            [0, 1, pw, pw + 1],
            [last, 7, 3 * pw + 2, last - pw],
            [5, 5, 5, 5],
        ] {
            assert_eq!(
                pixel_sad_x4_16x16(src, &r, offs, pw),
                pixel_sad_x4_16x16_scalar(src, &r, offs, pw)
            );
        }
    }
}
