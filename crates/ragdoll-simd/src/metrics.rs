/// Computes the Sum of Absolute Differences ($\sum |a - b|$) over a 2D block or plane.
#[inline]
pub fn sad_plane_u8(
    a: &[u8],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
    width: usize,
    height: usize,
) -> u64 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { sad_plane_u8_neon(a, a_stride, b, b_stride, width, height) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        sad_plane_u8_scalar(a, a_stride, b, b_stride, width, height)
    }
}

pub fn sad_plane_u8_scalar(
    a: &[u8],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
    width: usize,
    height: usize,
) -> u64 {
    let mut total = 0u64;
    for y in 0..height {
        let row_a = &a[y * a_stride..y * a_stride + width];
        let row_b = &b[y * b_stride..y * b_stride + width];
        let mut row_sum = 0u32;
        for x in 0..width {
            row_sum += row_a[x].abs_diff(row_b[x]) as u32;
        }
        total += row_sum as u64;
    }
    total
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sad_plane_u8_neon(
    a: &[u8],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
    width: usize,
    height: usize,
) -> u64 {
    use std::arch::aarch64::*;
    let mut total = 0u64;

    for y in 0..height {
        let row_a = &a[y * a_stride..y * a_stride + width];
        let row_b = &b[y * b_stride..y * b_stride + width];
        let pa = row_a.as_ptr();
        let pb = row_b.as_ptr();
        let mut x = 0usize;

        // 4 independent u16x8 accumulators saturate all 4 NEON pipes on Apple M-series.
        // Up to 16,384 pixels per row (256 iterations x 2 x 255 = 130,560 -> chunked at 8192px)
        // never overflows u16 lanes for any resolution up to 8K UHD.
        let mut acc0 = vdupq_n_u16(0);
        let mut acc1 = vdupq_n_u16(0);
        let mut acc2 = vdupq_n_u16(0);
        let mut acc3 = vdupq_n_u16(0);

        while x + 64 <= width {
            let va0 = unsafe { vld1q_u8(pa.add(x)) };
            let vb0 = unsafe { vld1q_u8(pb.add(x)) };
            let va1 = unsafe { vld1q_u8(pa.add(x + 16)) };
            let vb1 = unsafe { vld1q_u8(pb.add(x + 16)) };
            let va2 = unsafe { vld1q_u8(pa.add(x + 32)) };
            let vb2 = unsafe { vld1q_u8(pb.add(x + 32)) };
            let va3 = unsafe { vld1q_u8(pa.add(x + 48)) };
            let vb3 = unsafe { vld1q_u8(pb.add(x + 48)) };

            acc0 = vabal_u8(acc0, vget_low_u8(va0), vget_low_u8(vb0));
            acc1 = vabal_high_u8(acc1, va0, vb0);
            acc2 = vabal_u8(acc2, vget_low_u8(va1), vget_low_u8(vb1));
            acc3 = vabal_high_u8(acc3, va1, vb1);

            acc0 = vabal_u8(acc0, vget_low_u8(va2), vget_low_u8(vb2));
            acc1 = vabal_high_u8(acc1, va2, vb2);
            acc2 = vabal_u8(acc2, vget_low_u8(va3), vget_low_u8(vb3));
            acc3 = vabal_high_u8(acc3, va3, vb3);

            x += 64;
        }

        while x + 16 <= width {
            let va = unsafe { vld1q_u8(pa.add(x)) };
            let vb = unsafe { vld1q_u8(pb.add(x)) };
            acc0 = vabal_u8(acc0, vget_low_u8(va), vget_low_u8(vb));
            acc1 = vabal_high_u8(acc1, va, vb);
            x += 16;
        }

        let sum01 = vaddl_u16(vget_low_u16(acc0), vget_high_u16(acc0));
        let sum23 = vaddl_u16(vget_low_u16(acc1), vget_high_u16(acc1));
        let sum45 = vaddl_u16(vget_low_u16(acc2), vget_high_u16(acc2));
        let sum67 = vaddl_u16(vget_low_u16(acc3), vget_high_u16(acc3));
        let sum_all = vaddq_u32(vaddq_u32(sum01, sum23), vaddq_u32(sum45, sum67));
        let mut row_sum = vaddlvq_u32(sum_all);

        while x < width {
            row_sum += row_a[x].abs_diff(row_b[x]) as u64;
            x += 1;
        }
        total += row_sum;
    }
    total
}

/// Computes the exact Sum of Squared Errors ($\sum (a - b)^2$) over a 2D plane.
#[inline]
pub fn sse_plane_u8(
    a: &[u8],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
    width: usize,
    height: usize,
) -> u64 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { sse_plane_u8_neon(a, a_stride, b, b_stride, width, height) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        sse_plane_u8_scalar(a, a_stride, b, b_stride, width, height)
    }
}

pub fn sse_plane_u8_scalar(
    a: &[u8],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
    width: usize,
    height: usize,
) -> u64 {
    let mut total = 0u64;
    for y in 0..height {
        let row_a = &a[y * a_stride..y * a_stride + width];
        let row_b = &b[y * b_stride..y * b_stride + width];
        let mut row_sse = 0u64;
        for x in 0..width {
            let d = row_a[x].abs_diff(row_b[x]) as u64;
            row_sse += d * d;
        }
        total += row_sse;
    }
    total
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sse_plane_u8_neon(
    a: &[u8],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
    width: usize,
    height: usize,
) -> u64 {
    use std::arch::aarch64::*;
    let mut total = 0u64;

    // Accumulate up to 8 rows in u32x4 lanes before horizontal reduction (8 * 7680 * 65025 << u32::MAX)
    let mut y = 0usize;
    while y < height {
        let batch_end = (y + 8).min(height);
        let mut acc0 = vdupq_n_u32(0);
        let mut acc1 = vdupq_n_u32(0);
        let mut acc2 = vdupq_n_u32(0);
        let mut acc3 = vdupq_n_u32(0);
        let mut tail_sse = 0u64;

        for row_y in y..batch_end {
            let row_a = &a[row_y * a_stride..row_y * a_stride + width];
            let row_b = &b[row_y * b_stride..row_y * b_stride + width];
            let pa = row_a.as_ptr();
            let pb = row_b.as_ptr();
            let mut x = 0usize;

            while x + 32 <= width {
                let va0 = unsafe { vld1q_u8(pa.add(x)) };
                let vb0 = unsafe { vld1q_u8(pb.add(x)) };
                let va1 = unsafe { vld1q_u8(pa.add(x + 16)) };
                let vb1 = unsafe { vld1q_u8(pb.add(x + 16)) };

                let d0_lo = vabdl_u8(vget_low_u8(va0), vget_low_u8(vb0));
                let d0_hi = vabdl_high_u8(va0, vb0);
                let d1_lo = vabdl_u8(vget_low_u8(va1), vget_low_u8(vb1));
                let d1_hi = vabdl_high_u8(va1, vb1);

                acc0 = vmlal_u16(acc0, vget_low_u16(d0_lo), vget_low_u16(d0_lo));
                acc1 = vmlal_high_u16(acc1, d0_lo, d0_lo);
                acc2 = vmlal_u16(acc2, vget_low_u16(d0_hi), vget_low_u16(d0_hi));
                acc3 = vmlal_high_u16(acc3, d0_hi, d0_hi);

                acc0 = vmlal_u16(acc0, vget_low_u16(d1_lo), vget_low_u16(d1_lo));
                acc1 = vmlal_high_u16(acc1, d1_lo, d1_lo);
                acc2 = vmlal_u16(acc2, vget_low_u16(d1_hi), vget_low_u16(d1_hi));
                acc3 = vmlal_high_u16(acc3, d1_hi, d1_hi);
                x += 32;
            }

            while x + 16 <= width {
                let va = unsafe { vld1q_u8(pa.add(x)) };
                let vb = unsafe { vld1q_u8(pb.add(x)) };
                let diff_lo = vabdl_u8(vget_low_u8(va), vget_low_u8(vb));
                let diff_hi = vabdl_high_u8(va, vb);

                acc0 = vmlal_u16(acc0, vget_low_u16(diff_lo), vget_low_u16(diff_lo));
                acc1 = vmlal_high_u16(acc1, diff_lo, diff_lo);
                acc2 = vmlal_u16(acc2, vget_low_u16(diff_hi), vget_low_u16(diff_hi));
                acc3 = vmlal_high_u16(acc3, diff_hi, diff_hi);
                x += 16;
            }

            while x < width {
                let d = row_a[x].abs_diff(row_b[x]) as u64;
                tail_sse += d * d;
                x += 1;
            }
        }

        let sum01 = vaddq_u32(acc0, acc1);
        let sum23 = vaddq_u32(acc2, acc3);
        total += vaddlvq_u32(vaddq_u32(sum01, sum23)) + tail_sse;
        y = batch_end;
    }
    total
}

/// Sufficient statistics for an $8\times8$ SSIM window:
/// $\sum x, \sum y, \sum x^2, \sum y^2, \sum xy$.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct SsimWindowStats {
    pub sum_a: u32,
    pub sum_b: u32,
    pub sum_aa: u32,
    pub sum_bb: u32,
    pub sum_ab: u32,
}

/// Computes the 5 sufficient statistics for an $8\times8$ pixel window in a single SIMD pass.
#[inline]
pub fn ssim_window_8x8_u8(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize) -> SsimWindowStats {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { ssim_window_8x8_u8_neon(a, a_stride, b, b_stride) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        ssim_window_8x8_u8_scalar(a, a_stride, b, b_stride)
    }
}

pub fn ssim_window_8x8_u8_scalar(
    a: &[u8],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
) -> SsimWindowStats {
    let mut stats = SsimWindowStats::default();
    for y in 0..8 {
        let ra = &a[y * a_stride..y * a_stride + 8];
        let rb = &b[y * b_stride..y * b_stride + 8];
        for x in 0..8 {
            let va = ra[x] as u32;
            let vb = rb[x] as u32;
            stats.sum_a += va;
            stats.sum_b += vb;
            stats.sum_aa += va * va;
            stats.sum_bb += vb * vb;
            stats.sum_ab += va * vb;
        }
    }
    stats
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn ssim_window_8x8_u8_neon(
    a: &[u8],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
) -> SsimWindowStats {
    use std::arch::aarch64::*;
    let mut sum_a16 = vdupq_n_u16(0);
    let mut sum_b16 = vdupq_n_u16(0);
    let mut sum_aa32 = vdupq_n_u32(0);
    let mut sum_bb32 = vdupq_n_u32(0);
    let mut sum_ab32 = vdupq_n_u32(0);

    for y in 0..8 {
        let va8 = unsafe { vld1_u8(a.as_ptr().add(y * a_stride)) };
        let vb8 = unsafe { vld1_u8(b.as_ptr().add(y * b_stride)) };

        sum_a16 = vaddw_u8(sum_a16, va8);
        sum_b16 = vaddw_u8(sum_b16, vb8);

        let aa16 = vmull_u8(va8, va8);
        let bb16 = vmull_u8(vb8, vb8);
        let ab16 = vmull_u8(va8, vb8);

        sum_aa32 = vpadalq_u16(sum_aa32, aa16);
        sum_bb32 = vpadalq_u16(sum_bb32, bb16);
        sum_ab32 = vpadalq_u16(sum_ab32, ab16);
    }

    SsimWindowStats {
        sum_a: vaddlvq_u16(sum_a16),
        sum_b: vaddlvq_u16(sum_b16),
        sum_aa: vaddvq_u32(sum_aa32),
        sum_bb: vaddvq_u32(sum_bb32),
        sum_ab: vaddvq_u32(sum_ab32),
    }
}

/// FFmpeg `vf_ssim.c` $C_1$ constant scaled by $N = 64$: $\lfloor (0.01 \times 255)^2 \times 64 + 0.5 \rfloor = 416$.
pub const SSIM_C1_8BIT: i32 = 416;
/// FFmpeg `vf_ssim.c` $C_2$ constant scaled by Bessel sample-variance factor $N(N-1) = 64 \times 63$:
/// $\lfloor (0.03 \times 255)^2 \times 64 \times 63 + 0.5 \rfloor = 235\,963$.
pub const SSIM_C2_8BIT: i32 = 235_963;

/// Evaluates the exact FFmpeg `ssim_end1` formula for an $8\times8$ window from its integer sums:
/// `s1` = $\sum a$, `s2` = $\sum b$, `ss` = $\sum (a^2 + b^2)$, `s12` = $\sum ab$.
#[inline(always)]
pub fn ssim_end1_8bit(s1: i32, s2: i32, ss: i32, s12: i32) -> f32 {
    let vars = ss * 64 - s1 * s1 - s2 * s2;
    let covar = s12 * 64 - s1 * s2;
    ((2 * s1 * s2 + SSIM_C1_8BIT) as f32 * (2 * covar + SSIM_C2_8BIT) as f32)
        / ((s1 * s1 + s2 * s2 + SSIM_C1_8BIT) as f32 * (vars + SSIM_C2_8BIT) as f32)
}

/// Computes `n` contiguous non-overlapping $4\times4$ sub-block SSIM statistics `[sum_a, sum_b, sum_aa_bb, sum_ab]`
/// across 4 scanlines (`vf_ssim.c::ssim_4x4xn_8bit`).
#[inline]
pub fn ssim_4x4xn_u8(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize, sums: &mut [[i32; 4]]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { ssim_4x4xn_u8_neon(a, a_stride, b, b_stride, sums) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        ssim_4x4xn_u8_scalar(a, a_stride, b, b_stride, sums)
    }
}

pub fn ssim_4x4xn_u8_scalar(
    a: &[u8],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
    sums: &mut [[i32; 4]],
) {
    for (blk, out) in sums.iter_mut().enumerate() {
        let x0 = blk * 4;
        let mut s1 = 0i32;
        let mut s2 = 0i32;
        let mut ss = 0i32;
        let mut s12 = 0i32;
        for y in 0..4 {
            let ra = &a[y * a_stride + x0..y * a_stride + x0 + 4];
            let rb = &b[y * b_stride + x0..y * b_stride + x0 + 4];
            for x in 0..4 {
                let va = ra[x] as i32;
                let vb = rb[x] as i32;
                s1 += va;
                s2 += vb;
                ss += va * va + vb * vb;
                s12 += va * vb;
            }
        }
        *out = [s1, s2, ss, s12];
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn ssim_4x4xn_u8_neon(
    a: &[u8],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
    sums: &mut [[i32; 4]],
) {
    use std::arch::aarch64::*;
    let n = sums.len();
    let mut blk = 0usize;

    // Process 4 adjacent 4x4 blocks (16 pixels wide x 4 rows) per NEON iteration using full 128-bit loads
    while blk + 4 <= n {
        let x0 = blk * 4;
        let mut sa_01 = vdupq_n_u16(0);
        let mut sa_23 = vdupq_n_u16(0);
        let mut sb_01 = vdupq_n_u16(0);
        let mut sb_23 = vdupq_n_u16(0);

        let mut ss0 = vdupq_n_u32(0);
        let mut ss1 = vdupq_n_u32(0);
        let mut ss2 = vdupq_n_u32(0);
        let mut ss3 = vdupq_n_u32(0);

        let mut sab0 = vdupq_n_u32(0);
        let mut sab1 = vdupq_n_u32(0);
        let mut sab2 = vdupq_n_u32(0);
        let mut sab3 = vdupq_n_u32(0);

        for y in 0..4 {
            let va = unsafe { vld1q_u8(a.as_ptr().add(y * a_stride + x0)) };
            let vb = unsafe { vld1q_u8(b.as_ptr().add(y * b_stride + x0)) };

            let va_lo = vget_low_u8(va);
            let va_hi = vget_high_u8(va);
            let vb_lo = vget_low_u8(vb);
            let vb_hi = vget_high_u8(vb);

            sa_01 = vaddw_u8(sa_01, va_lo);
            sa_23 = vaddw_u8(sa_23, va_hi);
            sb_01 = vaddw_u8(sb_01, vb_lo);
            sb_23 = vaddw_u8(sb_23, vb_hi);

            let aa_01 = vmull_u8(va_lo, va_lo);
            let aa_23 = vmull_high_u8(va, va);
            let bb_01 = vmull_u8(vb_lo, vb_lo);
            let bb_23 = vmull_high_u8(vb, vb);
            let ab_01 = vmull_u8(va_lo, vb_lo);
            let ab_23 = vmull_high_u8(va, vb);

            let aa_bb_01_lo = vaddl_u16(vget_low_u16(aa_01), vget_low_u16(bb_01));
            let aa_bb_01_hi = vaddl_high_u16(aa_01, bb_01);
            let aa_bb_23_lo = vaddl_u16(vget_low_u16(aa_23), vget_low_u16(bb_23));
            let aa_bb_23_hi = vaddl_high_u16(aa_23, bb_23);

            ss0 = vaddq_u32(ss0, aa_bb_01_lo);
            ss1 = vaddq_u32(ss1, aa_bb_01_hi);
            ss2 = vaddq_u32(ss2, aa_bb_23_lo);
            ss3 = vaddq_u32(ss3, aa_bb_23_hi);

            sab0 = vaddw_u16(sab0, vget_low_u16(ab_01));
            sab1 = vaddw_high_u16(sab1, ab_01);
            sab2 = vaddw_u16(sab2, vget_low_u16(ab_23));
            sab3 = vaddw_high_u16(sab3, ab_23);
        }

        sums[blk] = [
            vaddlv_u16(vget_low_u16(sa_01)) as i32,
            vaddlv_u16(vget_low_u16(sb_01)) as i32,
            vaddvq_u32(ss0) as i32,
            vaddvq_u32(sab0) as i32,
        ];
        sums[blk + 1] = [
            vaddlv_u16(vget_high_u16(sa_01)) as i32,
            vaddlv_u16(vget_high_u16(sb_01)) as i32,
            vaddvq_u32(ss1) as i32,
            vaddvq_u32(sab1) as i32,
        ];
        sums[blk + 2] = [
            vaddlv_u16(vget_low_u16(sa_23)) as i32,
            vaddlv_u16(vget_low_u16(sb_23)) as i32,
            vaddvq_u32(ss2) as i32,
            vaddvq_u32(sab2) as i32,
        ];
        sums[blk + 3] = [
            vaddlv_u16(vget_high_u16(sa_23)) as i32,
            vaddlv_u16(vget_high_u16(sb_23)) as i32,
            vaddvq_u32(ss3) as i32,
            vaddvq_u32(sab3) as i32,
        ];
        blk += 4;
    }

    if blk < n {
        let x0 = blk * 4;
        ssim_4x4xn_u8_scalar(&a[x0..], a_stride, &b[x0..], b_stride, &mut sums[blk..]);
    }
}

/// Computes the full-plane SSIM score in $[0.0, 1.0]$ using FFmpeg `vf_ssim.c`'s
/// $4\times4$ sub-block reuse algorithm (stepping $8\times8$ windows by 4 pixels).
pub fn ssim_plane_8bit(
    a: &[u8],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
    width: usize,
    height: usize,
) -> f64 {
    let bw = width >> 2;
    let bh = height >> 2;
    if bw < 2 || bh < 2 {
        let stats = ssim_window_8x8_u8(a, a_stride, b, b_stride);
        return ssim_end1_8bit(
            stats.sum_a as i32,
            stats.sum_b as i32,
            (stats.sum_aa + stats.sum_bb) as i32,
            stats.sum_ab as i32,
        ) as f64;
    }

    let mut row0 = vec![[0i32; 4]; bw];
    let mut row1 = vec![[0i32; 4]; bw];
    ssim_4x4xn_u8(a, a_stride, b, b_stride, &mut row0);

    let mut total_ssim = 0.0f64;
    for by in 1..bh {
        let y_off_a = by * 4 * a_stride;
        let y_off_b = by * 4 * b_stride;
        ssim_4x4xn_u8(&a[y_off_a..], a_stride, &b[y_off_b..], b_stride, &mut row1);

        let mut row_ssim = 0.0f32;
        for bx in 0..(bw - 1) {
            let s1 = row0[bx][0] + row0[bx + 1][0] + row1[bx][0] + row1[bx + 1][0];
            let s2 = row0[bx][1] + row0[bx + 1][1] + row1[bx][1] + row1[bx + 1][1];
            let ss = row0[bx][2] + row0[bx + 1][2] + row1[bx][2] + row1[bx + 1][2];
            let s12 = row0[bx][3] + row0[bx + 1][3] + row1[bx][3] + row1[bx + 1][3];
            row_ssim += ssim_end1_8bit(s1, s2, ss, s12);
        }
        total_ssim += row_ssim as f64;
        std::mem::swap(&mut row0, &mut row1);
    }

    total_ssim / (((bw - 1) * (bh - 1)) as f64)
}

/// Converts a linear SSIM score in $[0.0, 1.0]$ into decibels (`-10.0 * log10(1.0 - ssim)`).
#[inline]
pub fn ssim_to_db(ssim: f64) -> f64 {
    if ssim >= 1.0 {
        f64::INFINITY
    } else {
        -10.0 * (1.0 - ssim).log10()
    }
}

/// Computes the $4\times4$ Sum of Absolute Transformed Differences (Hadamard SATD) between two `u8` blocks.
///
/// Modeled after `x264/common/aarch64/pixel-a.S::pixel_satd_4x4_neon`.
#[inline]
pub fn satd_4x4_u8(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { satd_4x4_u8_neon(a, a_stride, b, b_stride) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        satd_4x4_u8_scalar(a, a_stride, b, b_stride)
    }
}

#[allow(clippy::needless_range_loop)]
pub fn satd_4x4_u8_scalar(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize) -> u32 {
    let mut tmp = [[0i32; 4]; 4];
    for y in 0..4 {
        let ra = &a[y * a_stride..y * a_stride + 4];
        let rb = &b[y * b_stride..y * b_stride + 4];
        let d0 = (ra[0] as i32) - (rb[0] as i32);
        let d1 = (ra[1] as i32) - (rb[1] as i32);
        let d2 = (ra[2] as i32) - (rb[2] as i32);
        let d3 = (ra[3] as i32) - (rb[3] as i32);

        let s03 = d0 + d3;
        let s12 = d1 + d2;
        let d03 = d0 - d3;
        let d12 = d1 - d2;

        tmp[y][0] = s03 + s12;
        tmp[y][1] = d03 + d12;
        tmp[y][2] = s03 - s12;
        tmp[y][3] = d03 - d12;
    }

    let mut sum = 0u32;
    for x in 0..4 {
        let s03 = tmp[0][x] + tmp[3][x];
        let s12 = tmp[1][x] + tmp[2][x];
        let d03 = tmp[0][x] - tmp[3][x];
        let d12 = tmp[1][x] - tmp[2][x];

        sum += (s03 + s12).unsigned_abs();
        sum += (d03 + d12).unsigned_abs();
        sum += (s03 - s12).unsigned_abs();
        sum += (d03 - d12).unsigned_abs();
    }
    sum >> 1
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn satd_4x4_u8_neon(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize) -> u32 {
    use std::arch::aarch64::*;
    let load_diff4 = |row: usize| -> int16x4_t {
        let pa = unsafe { a.as_ptr().add(row * a_stride) as *const u32 };
        let pb = unsafe { b.as_ptr().add(row * b_stride) as *const u32 };
        let ua = unsafe { vcreate_u8(pa.read_unaligned() as u64) };
        let ub = unsafe { vcreate_u8(pb.read_unaligned() as u64) };
        vget_low_s16(vreinterpretq_s16_u16(vsubl_u8(ua, ub)))
    };

    let r0 = load_diff4(0);
    let r1 = load_diff4(1);
    let r2 = load_diff4(2);
    let r3 = load_diff4(3);

    // 1. Vertical 4-point Hadamard butterfly across rows r0..r3
    let s03 = vadd_s16(r0, r3);
    let s12 = vadd_s16(r1, r2);
    let d03 = vsub_s16(r0, r3);
    let d12 = vsub_s16(r1, r2);

    let v0 = vadd_s16(s03, s12);
    let v1 = vadd_s16(d03, d12);
    let v2 = vsub_s16(s03, s12);
    let v3 = vsub_s16(d03, d12);

    // 2. In-register 4x4 16-bit transpose via vtrn1/vtrn2
    let t01_lo = vtrn1_s16(v0, v1);
    let t01_hi = vtrn2_s16(v0, v1);
    let t23_lo = vtrn1_s16(v2, v3);
    let t23_hi = vtrn2_s16(v2, v3);

    let c0 = vreinterpret_s16_s32(vtrn1_s32(
        vreinterpret_s32_s16(t01_lo),
        vreinterpret_s32_s16(t23_lo),
    ));
    let c1 = vreinterpret_s16_s32(vtrn1_s32(
        vreinterpret_s32_s16(t01_hi),
        vreinterpret_s32_s16(t23_hi),
    ));
    let c2 = vreinterpret_s16_s32(vtrn2_s32(
        vreinterpret_s32_s16(t01_lo),
        vreinterpret_s32_s16(t23_lo),
    ));
    let c3 = vreinterpret_s16_s32(vtrn2_s32(
        vreinterpret_s32_s16(t01_hi),
        vreinterpret_s32_s16(t23_hi),
    ));

    // 3. Horizontal 4-point Hadamard butterfly across transposed columns c0..c3
    let hs03 = vadd_s16(c0, c3);
    let hs12 = vadd_s16(c1, c2);
    let hd03 = vsub_s16(c0, c3);
    let hd12 = vsub_s16(c1, c2);

    let h0 = vreinterpret_u16_s16(vabs_s16(vadd_s16(hs03, hs12)));
    let h1 = vreinterpret_u16_s16(vabs_s16(vadd_s16(hd03, hd12)));
    let h2 = vreinterpret_u16_s16(vabs_s16(vsub_s16(hs03, hs12)));
    let h3 = vreinterpret_u16_s16(vabs_s16(vsub_s16(hd03, hd12)));

    let acc01 = vcombine_u16(h0, h1);
    let acc23 = vcombine_u16(h2, h3);
    (vaddlvq_u16(acc01) + vaddlvq_u16(acc23)) >> 1
}

/// Computes the $8\times8$ Hadamard SATD by summing the four $4\times4$ sub-block SATDs.
#[inline]
pub fn satd_8x8_u8(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize) -> u32 {
    satd_4x4_u8(a, a_stride, b, b_stride)
        + satd_4x4_u8(&a[4..], a_stride, &b[4..], b_stride)
        + satd_4x4_u8(&a[4 * a_stride..], a_stride, &b[4 * b_stride..], b_stride)
        + satd_4x4_u8(
            &a[4 * a_stride + 4..],
            a_stride,
            &b[4 * b_stride + 4..],
            b_stride,
        )
}

/// Computes the $16\times16$ Sum of Absolute Differences (`SAD`) between two strided 2D blocks
/// with a single terminal horizontal reduction (`libavcodec/aarch64/pixblockdsp_neon.S`).
#[inline]
pub fn sad_16x16_u8(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { sad_16x16_u8_neon(a, a_stride, b, b_stride) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        sad_16x16_u8_scalar(a, a_stride, b, b_stride)
    }
}

pub fn sad_16x16_u8_scalar(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize) -> u32 {
    let mut sum = 0u32;
    for y in 0..16 {
        let ra = &a[y * a_stride..y * a_stride + 16];
        let rb = &b[y * b_stride..y * b_stride + 16];
        for x in 0..16 {
            sum += ra[x].abs_diff(rb[x]) as u32;
        }
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sad_16x16_u8_neon(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize) -> u32 {
    use std::arch::aarch64::*;
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let mut acc_lo = vdupq_n_u16(0);
    let mut acc_hi = vdupq_n_u16(0);

    for y in 0..16 {
        let va = unsafe { vld1q_u8(pa.add(y * a_stride)) };
        let vb = unsafe { vld1q_u8(pb.add(y * b_stride)) };
        acc_lo = vabal_u8(acc_lo, vget_low_u8(va), vget_low_u8(vb));
        acc_hi = vabal_high_u8(acc_hi, va, vb);
    }
    vaddlvq_u16(vaddq_u16(acc_lo, acc_hi))
}

/// Computes the SAD between two contiguous 256-byte ($16\times16$ luma macroblock) buffers in 4 NEON iterations.
#[inline]
pub fn sad_block_256_u8(a: &[u8; 256], b: &[u8; 256]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { sad_block_256_u8_neon(a, b) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        sad_block_256_u8_scalar(a, b)
    }
}

pub fn sad_block_256_u8_scalar(a: &[u8; 256], b: &[u8; 256]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x.abs_diff(y) as u32)
        .sum()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sad_block_256_u8_neon(a: &[u8; 256], b: &[u8; 256]) -> u32 {
    use std::arch::aarch64::*;
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let mut acc0 = vdupq_n_u16(0);
    let mut acc1 = vdupq_n_u16(0);
    let mut acc2 = vdupq_n_u16(0);
    let mut acc3 = vdupq_n_u16(0);

    for chunk in 0..4 {
        let off = chunk * 64;
        let va0 = unsafe { vld1q_u8(pa.add(off)) };
        let vb0 = unsafe { vld1q_u8(pb.add(off)) };
        let va1 = unsafe { vld1q_u8(pa.add(off + 16)) };
        let vb1 = unsafe { vld1q_u8(pb.add(off + 16)) };
        let va2 = unsafe { vld1q_u8(pa.add(off + 32)) };
        let vb2 = unsafe { vld1q_u8(pb.add(off + 32)) };
        let va3 = unsafe { vld1q_u8(pa.add(off + 48)) };
        let vb3 = unsafe { vld1q_u8(pb.add(off + 48)) };

        acc0 = vabal_u8(acc0, vget_low_u8(va0), vget_low_u8(vb0));
        acc1 = vabal_high_u8(acc1, va0, vb0);
        acc2 = vabal_u8(acc2, vget_low_u8(va1), vget_low_u8(vb1));
        acc3 = vabal_high_u8(acc3, va1, vb1);

        acc0 = vabal_u8(acc0, vget_low_u8(va2), vget_low_u8(vb2));
        acc1 = vabal_high_u8(acc1, va2, vb2);
        acc2 = vabal_u8(acc2, vget_low_u8(va3), vget_low_u8(vb3));
        acc3 = vabal_high_u8(acc3, va3, vb3);
    }

    let s01 = vaddq_u16(acc0, acc1);
    let s23 = vaddq_u16(acc2, acc3);
    vaddlvq_u16(vaddq_u16(s01, s23))
}

/// Computes the SAD between two contiguous 64-byte ($8\times8$ chroma macroblock) buffers in 1 NEON pass.
#[inline]
pub fn sad_block_64_u8(a: &[u8; 64], b: &[u8; 64]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { sad_block_64_u8_neon(a, b) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        sad_block_64_u8_scalar(a, b)
    }
}

pub fn sad_block_64_u8_scalar(a: &[u8; 64], b: &[u8; 64]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x.abs_diff(y) as u32)
        .sum()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sad_block_64_u8_neon(a: &[u8; 64], b: &[u8; 64]) -> u32 {
    use std::arch::aarch64::*;
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let va0 = unsafe { vld1q_u8(pa) };
    let vb0 = unsafe { vld1q_u8(pb) };
    let va1 = unsafe { vld1q_u8(pa.add(16)) };
    let vb1 = unsafe { vld1q_u8(pb.add(16)) };
    let va2 = unsafe { vld1q_u8(pa.add(32)) };
    let vb2 = unsafe { vld1q_u8(pb.add(32)) };
    let va3 = unsafe { vld1q_u8(pa.add(48)) };
    let vb3 = unsafe { vld1q_u8(pb.add(48)) };

    let mut acc0 = vabdl_u8(vget_low_u8(va0), vget_low_u8(vb0));
    let mut acc1 = vabdl_high_u8(va0, vb0);
    let mut acc2 = vabdl_u8(vget_low_u8(va1), vget_low_u8(vb1));
    let mut acc3 = vabdl_high_u8(va1, vb1);

    acc0 = vabal_u8(acc0, vget_low_u8(va2), vget_low_u8(vb2));
    acc1 = vabal_high_u8(acc1, va2, vb2);
    acc2 = vabal_u8(acc2, vget_low_u8(va3), vget_low_u8(vb3));
    acc3 = vabal_high_u8(acc3, va3, vb3);

    let s01 = vaddq_u16(acc0, acc1);
    let s23 = vaddq_u16(acc2, acc3);
    vaddlvq_u16(vaddq_u16(s01, s23))
}

/// Evaluates the $L_1$ residual error sums ($\sum_{i=k}^{n-1} |\Delta^{(k)} x[i]|$) for all 5 FLAC
/// Fixed predictor orders $k \in \{0, 1, 2, 3, 4\}$ in a **single fused memory pass**
/// (`libFLAC/fixed.c:244-250`).
#[inline]
pub fn flac_eval_fixed_orders_l1(samples: &[i32]) -> [u64; 5] {
    #[cfg(target_arch = "aarch64")]
    {
        if samples.len() >= 8 {
            return unsafe { flac_eval_fixed_orders_l1_neon(samples) };
        }
    }
    flac_eval_fixed_orders_l1_scalar(samples)
}

pub fn flac_eval_fixed_orders_l1_scalar(samples: &[i32]) -> [u64; 5] {
    let n = samples.len();
    let mut l1 = [0u64; 5];
    if n == 0 {
        return l1;
    }

    // Handle initial warm-up samples 0..min(4, n)
    let init_end = n.min(4);
    for i in 0..init_end {
        let s0 = samples[i] as i64;
        l1[0] += s0.unsigned_abs();
        if i >= 1 {
            let e1 = s0 - (samples[i - 1] as i64);
            l1[1] += e1.unsigned_abs();
        }
        if i >= 2 {
            let e2 = s0 - 2 * (samples[i - 1] as i64) + (samples[i - 2] as i64);
            l1[2] += e2.unsigned_abs();
        }
        if i >= 3 {
            let e3 = s0 - 3 * (samples[i - 1] as i64) + 3 * (samples[i - 2] as i64)
                - (samples[i - 3] as i64);
            l1[3] += e3.unsigned_abs();
        }
    }
    if n <= 4 {
        return l1;
    }

    // Cascaded single-pass recurrence: e_k[i] = e_{k-1}[i] - e_{k-1}[i-1]
    let mut last0 = samples[3] as i64;
    let mut last1 = (samples[3] as i64) - (samples[2] as i64);
    let mut last2 = last1 - ((samples[2] as i64) - (samples[1] as i64));
    let mut last3 = last2 - ((samples[2] as i64) - 2 * (samples[1] as i64) + (samples[0] as i64));

    for &s in &samples[4..] {
        let e0 = s as i64;
        let e1 = e0 - last0;
        let e2 = e1 - last1;
        let e3 = e2 - last2;
        let e4 = e3 - last3;

        last0 = e0;
        last1 = e1;
        last2 = e2;
        last3 = e3;

        l1[0] += e0.unsigned_abs();
        l1[1] += e1.unsigned_abs();
        l1[2] += e2.unsigned_abs();
        l1[3] += e3.unsigned_abs();
        l1[4] += e4.unsigned_abs();
    }
    l1
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn flac_eval_fixed_orders_l1_neon(samples: &[i32]) -> [u64; 5] {
    use std::arch::aarch64::*;
    let n = samples.len();
    // Guard against 32-bit overflow on extreme 28+ bit inputs (`16 * |s| >= i32::MAX`)
    // Standard 8/16/24-bit + Side-channel 25-bit audio satisfies `|s| <= (1 << 24)`.
    let mut l1 = flac_eval_fixed_orders_l1_scalar(&samples[..4]);

    let s0 = samples[0] as i64;
    let s1 = samples[1] as i64;
    let s2 = samples[2] as i64;
    let s3 = samples[3] as i64;
    if s0.abs() > (1 << 25) || s1.abs() > (1 << 25) || s2.abs() > (1 << 25) || s3.abs() > (1 << 25)
    {
        return flac_eval_fixed_orders_l1_scalar(samples);
    }

    // Seed the 4-lane history registers so lane 3 holds the `i = 3` cascaded difference
    let mut prev_e0 = unsafe { vld1q_s32(samples.as_ptr()) };
    let e1_3 = (s3 - s2) as i32;
    let e1_2 = (s2 - s1) as i32;
    let e1_1 = (s1 - s0) as i32;
    let mut prev_e1 = unsafe { vld1q_s32([0, e1_1, e1_2, e1_3].as_ptr()) };
    let e2_3 = e1_3 - e1_2;
    let e2_2 = e1_2 - e1_1;
    let mut prev_e2 = unsafe { vld1q_s32([0, 0, e2_2, e2_3].as_ptr()) };
    let e3_3 = e2_3 - e2_2;
    let mut prev_e3 = unsafe { vld1q_s32([0, 0, 0, e3_3].as_ptr()) };

    let mut acc0 = vdupq_n_u64(0);
    let mut acc1 = vdupq_n_u64(0);
    let mut acc2 = vdupq_n_u64(0);
    let mut acc3 = vdupq_n_u64(0);
    let mut acc4 = vdupq_n_u64(0);

    let ptr = samples.as_ptr();
    let mut i = 4usize;
    let limit = vdupq_n_s32(1 << 25);

    while i + 4 <= n {
        let e0 = unsafe { vld1q_s32(ptr.add(i)) };
        if vmaxvq_s32(vabsq_s32(e0)) > (1 << 25) {
            let _ = limit;
            return flac_eval_fixed_orders_l1_scalar(samples);
        }

        let e0_m1 = vextq_s32::<3>(prev_e0, e0);
        let e1 = vsubq_s32(e0, e0_m1);

        let e1_m1 = vextq_s32::<3>(prev_e1, e1);
        let e2 = vsubq_s32(e1, e1_m1);

        let e2_m1 = vextq_s32::<3>(prev_e2, e2);
        let e3 = vsubq_s32(e2, e2_m1);

        let e3_m1 = vextq_s32::<3>(prev_e3, e3);
        let e4 = vsubq_s32(e3, e3_m1);

        prev_e0 = e0;
        prev_e1 = e1;
        prev_e2 = e2;
        prev_e3 = e3;

        acc0 = vpadalq_u32(acc0, vreinterpretq_u32_s32(vabsq_s32(e0)));
        acc1 = vpadalq_u32(acc1, vreinterpretq_u32_s32(vabsq_s32(e1)));
        acc2 = vpadalq_u32(acc2, vreinterpretq_u32_s32(vabsq_s32(e2)));
        acc3 = vpadalq_u32(acc3, vreinterpretq_u32_s32(vabsq_s32(e3)));
        acc4 = vpadalq_u32(acc4, vreinterpretq_u32_s32(vabsq_s32(e4)));

        i += 4;
    }

    l1[0] += vaddvq_u64(acc0);
    l1[1] += vaddvq_u64(acc1);
    l1[2] += vaddvq_u64(acc2);
    l1[3] += vaddvq_u64(acc3);
    l1[4] += vaddvq_u64(acc4);

    if i < n {
        let mut last0 = samples[i - 1] as i64;
        let mut last1 = (samples[i - 1] as i64) - (samples[i - 2] as i64);
        let mut last2 = last1 - ((samples[i - 2] as i64) - (samples[i - 3] as i64));
        let mut last3 = last2
            - ((samples[i - 2] as i64) - 2 * (samples[i - 3] as i64) + (samples[i - 4] as i64));

        for &s in &samples[i..] {
            let e0 = s as i64;
            let e1 = e0 - last0;
            let e2 = e1 - last1;
            let e3 = e2 - last2;
            let e4 = e3 - last3;

            last0 = e0;
            last1 = e1;
            last2 = e2;
            last3 = e3;

            l1[0] += e0.unsigned_abs();
            l1[1] += e1.unsigned_abs();
            l1[2] += e2.unsigned_abs();
            l1[3] += e3.unsigned_abs();
            l1[4] += e4.unsigned_abs();
        }
    }

    l1
}
