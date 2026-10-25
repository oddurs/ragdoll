//! H.264 / AVC (`ITU-T H.264 / ISO/IEC 14496-10`) DSP & SIMD Kernels (`SIMD-0038`).
//!
//! Reference-hardened against `libavcodec/h264qpel_template.c`, `libavcodec/h264chroma_template.c`,
//! `libavcodec/h264_loopfilter.c`, and `libavcodec/h264pred_template.c`:
//! - **6-Tap FIR Quarter-Pel (`qpel`) Luma Motion Interpolation**: [`put_h264_qpel16_mc_u8`] and [`put_h264_qpel8_mc_u8`]
//!   across all 16 `(mx, my)` fractional phases with `aarch64` NEON (`vrhaddq_u8`, 16-bit 6-tap FIR) and scalar parity.
//! - **Bilinear $1/8$-Pel Chroma Motion Interpolation**: [`put_h264_chroma_mc8_u8`] with `aarch64` NEON (`vmull_u8` / `vmlal_u8`).
//! - **Adaptive In-Loop Deblocking Filter (ITU-T H.264 8.7)**: [`h264_loop_filter_luma_edge_h_u8`],
//!   [`h264_loop_filter_luma_edge_v_u8`], [`h264_loop_filter_chroma_edge_h_u8`], [`h264_loop_filter_chroma_edge_v_u8`]
//!   for boundary strengths $bS \in \{1..4\}$ with exact Table 8-16 $\alpha, \beta, t_{C0}$ thresholds.
//! - **Intra $4\times4$ 9-Mode Spatial Prediction (ITU-T H.264 8.3.1.2)**: [`predict_intra_4x4_u8`] for modes `0..=8`.

use crate::fixed::clamp_u8;

/// ITU-T H.264 Table 8-16 $\alpha(\text{index}_A)$ deblocking threshold table (`QP = 0..51`).
pub const H264_DEBLOCK_ALPHA: [u8; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 20,
    22, 25, 28, 32, 36, 40, 45, 50, 56, 63, 71, 80, 90, 101, 113, 127, 144, 162, 182, 203, 226,
    255, 255,
];

/// ITU-T H.264 Table 8-16 $\beta(\text{index}_B)$ deblocking threshold table (`QP = 0..51`).
pub const H264_DEBLOCK_BETA: [u8; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8, 8,
    9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
];

/// ITU-T H.264 Table 8-16 $t_{C0}(\text{index}_A, bS - 1)$ clipping table for $bS \in \{1, 2, 3\}$.
pub const H264_DEBLOCK_TC0: [[u8; 3]; 52] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 1],
    [0, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 2],
    [1, 1, 2],
    [1, 1, 2],
    [1, 1, 2],
    [1, 2, 3],
    [1, 2, 3],
    [2, 2, 3],
    [2, 2, 4],
    [2, 3, 4],
    [2, 3, 4],
    [3, 3, 5],
    [3, 4, 6],
    [3, 4, 6],
    [4, 5, 7],
    [4, 5, 8],
    [4, 6, 9],
    [5, 7, 10],
    [6, 8, 11],
    [6, 8, 13],
    [7, 10, 14],
    [8, 11, 16],
    [9, 12, 18],
    [10, 13, 20],
    [11, 15, 23],
    [13, 17, 25],
];

#[inline(always)]
fn sample_clamped(
    src: &[u8],
    stride: usize,
    width: usize,
    height: usize,
    x: isize,
    y: isize,
) -> u8 {
    let cx = x.clamp(0, (width as isize) - 1) as usize;
    let cy = y.clamp(0, (height as isize) - 1) as usize;
    src[cy * stride + cx]
}

#[inline(always)]
const fn fir6_raw(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32) -> i32 {
    a - 5 * b + 20 * c + 20 * d - 5 * e + f
}

#[inline(always)]
const fn avg_u8(a: u8, b: u8) -> u8 {
    ((a as u16 + b as u16 + 1) >> 1) as u8
}

/// Interpolates a $16\times16$ luma block at integer position `(x0, y0)` and quarter-pel phase `(mx, my)`
/// (`mx, my in 0..=3`) per ITU-T H.264 8.4.2.2.1 (`put_h264_qpel16_mc{mx}{my}_8_c`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn put_h264_qpel16_mc_u8(
    src: &[u8],
    src_stride: usize,
    width: usize,
    height: usize,
    x0: isize,
    y0: isize,
    mx: u8,
    my: u8,
    dst: &mut [u8; 256],
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            put_h264_qpel16_mc_u8_neon(src, src_stride, width, height, x0, y0, mx, my, dst);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        put_h264_qpel16_mc_u8_scalar(src, src_stride, width, height, x0, y0, mx, my, dst);
    }
}

/// Interpolates an $8\times8$ luma sub-block at integer position `(x0, y0)` and quarter-pel phase `(mx, my)` (`0..=3`).
#[allow(clippy::too_many_arguments)]
pub fn put_h264_qpel8_mc_u8(
    src: &[u8],
    src_stride: usize,
    width: usize,
    height: usize,
    x0: isize,
    y0: isize,
    mx: u8,
    my: u8,
    dst: &mut [u8; 64],
) {
    let mut tmp16 = [0u8; 256];
    put_h264_qpel16_mc_u8(src, src_stride, width, height, x0, y0, mx, my, &mut tmp16);
    for r in 0..8 {
        dst[r * 8..r * 8 + 8].copy_from_slice(&tmp16[r * 16..r * 16 + 8]);
    }
}

/// Scalar reference for 16x16 H.264 6-tap quarter-pel luma motion compensation (`mx, my in 0..=3`).
#[allow(clippy::too_many_arguments)]
pub fn put_h264_qpel16_mc_u8_scalar(
    src: &[u8],
    src_stride: usize,
    width: usize,
    height: usize,
    x0: isize,
    y0: isize,
    mx: u8,
    my: u8,
    dst: &mut [u8; 256],
) {
    let mx = mx & 3;
    let my = my & 3;

    // Fetch a 22x22 padded integer window around (x0 - 2, y0 - 2) on the stack (zero heap allocation!)
    let mut win = [[0u8; 22]; 22];
    for (wy, row) in win.iter_mut().enumerate() {
        let sy = y0 + (wy as isize) - 2;
        for (wx, slot) in row.iter_mut().enumerate() {
            let sx = x0 + (wx as isize) - 2;
            *slot = sample_clamped(src, src_stride, width, height, sx, sy);
        }
    }

    if mx == 0 && my == 0 {
        for r in 0..16 {
            dst[r * 16..r * 16 + 16].copy_from_slice(&win[r + 2][2..18]);
        }
        return;
    }

    // Precompute horizontal 6-tap raw values b1[y][x] for y in 0..22, x in 0..16
    let mut b1 = [[0i32; 16]; 22];
    for wy in 0..22 {
        for x in 0..16 {
            b1[wy][x] = fir6_raw(
                win[wy][x] as i32,
                win[wy][x + 1] as i32,
                win[wy][x + 2] as i32,
                win[wy][x + 3] as i32,
                win[wy][x + 4] as i32,
                win[wy][x + 5] as i32,
            );
        }
    }

    for r in 0..16 {
        let wy = r + 2;
        for c in 0..16 {
            let wx = c + 2;
            let g00 = win[wy][wx];
            let g10 = win[wy][wx + 1];
            let g01 = win[wy + 1][wx];

            let b00 = clamp_u8((b1[wy][c] + 16) >> 5);
            let b01 = clamp_u8((b1[wy + 1][c] + 16) >> 5);

            let h00 = clamp_u8(
                (fir6_raw(
                    win[r][wx] as i32,
                    win[r + 1][wx] as i32,
                    win[r + 2][wx] as i32,
                    win[r + 3][wx] as i32,
                    win[r + 4][wx] as i32,
                    win[r + 5][wx] as i32,
                ) + 16)
                    >> 5,
            );
            let h10 = clamp_u8(
                (fir6_raw(
                    win[r][wx + 1] as i32,
                    win[r + 1][wx + 1] as i32,
                    win[r + 2][wx + 1] as i32,
                    win[r + 3][wx + 1] as i32,
                    win[r + 4][wx + 1] as i32,
                    win[r + 5][wx + 1] as i32,
                ) + 16)
                    >> 5,
            );

            let j00 = clamp_u8(
                (fir6_raw(
                    b1[r][c],
                    b1[r + 1][c],
                    b1[r + 2][c],
                    b1[r + 3][c],
                    b1[r + 4][c],
                    b1[r + 5][c],
                ) + 512)
                    >> 10,
            );

            dst[r * 16 + c] = match (mx, my) {
                (0, 0) => g00,
                (1, 0) => avg_u8(g00, b00),
                (2, 0) => b00,
                (3, 0) => avg_u8(b00, g10),
                (0, 1) => avg_u8(g00, h00),
                (1, 1) => avg_u8(b00, h00),
                (2, 1) => avg_u8(b00, j00),
                (3, 1) => avg_u8(b00, h10),
                (0, 2) => h00,
                (1, 2) => avg_u8(h00, j00),
                (2, 2) => j00,
                (3, 2) => avg_u8(j00, h10),
                (0, 3) => avg_u8(h00, g01),
                (1, 3) => avg_u8(h00, b01),
                (2, 3) => avg_u8(j00, b01),
                (3, 3) => avg_u8(h10, b01),
                _ => unreachable!(),
            };
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn put_h264_qpel16_mc_u8_neon(
    src: &[u8],
    src_stride: usize,
    width: usize,
    height: usize,
    x0: isize,
    y0: isize,
    mx: u8,
    my: u8,
    dst: &mut [u8; 256],
) {
    use std::arch::aarch64::*;
    let mx = mx & 3;
    let my = my & 3;

    let mut win = [[0u8; 22]; 22];
    for (wy, row) in win.iter_mut().enumerate() {
        let sy = y0 + (wy as isize) - 2;
        for (wx, slot) in row.iter_mut().enumerate() {
            let sx = x0 + (wx as isize) - 2;
            *slot = sample_clamped(src, src_stride, width, height, sx, sy);
        }
    }

    if mx == 0 && my == 0 {
        for r in 0..16 {
            let v = unsafe { vld1q_u8(win[r + 2].as_ptr().add(2)) };
            unsafe { vst1q_u8(dst.as_mut_ptr().add(r * 16), v) };
        }
        return;
    }

    // Vectorized 6-tap horizontal raw filter b1[22][16] (`i16`) using NEON 8-lane i16 math
    let mut b1 = [[0i16; 16]; 22];
    let mut b_clipped = [[0u8; 16]; 22];
    for wy in 0..22 {
        let row_ptr = win[wy].as_ptr();
        for half in 0..2 {
            let off = half * 8;
            let a = vreinterpretq_s16_u16(vmovl_u8(unsafe { vld1_u8(row_ptr.add(off)) }));
            let b = vreinterpretq_s16_u16(vmovl_u8(unsafe { vld1_u8(row_ptr.add(off + 1)) }));
            let c = vreinterpretq_s16_u16(vmovl_u8(unsafe { vld1_u8(row_ptr.add(off + 2)) }));
            let d = vreinterpretq_s16_u16(vmovl_u8(unsafe { vld1_u8(row_ptr.add(off + 3)) }));
            let e = vreinterpretq_s16_u16(vmovl_u8(unsafe { vld1_u8(row_ptr.add(off + 4)) }));
            let f = vreinterpretq_s16_u16(vmovl_u8(unsafe { vld1_u8(row_ptr.add(off + 5)) }));

            let sum_af = vaddq_s16(a, f);
            let sum_be = vaddq_s16(b, e);
            let sum_cd = vaddq_s16(c, d);
            let raw = vaddq_s16(
                vsubq_s16(sum_af, vmulq_n_s16(sum_be, 5)),
                vmulq_n_s16(sum_cd, 20),
            );
            unsafe { vst1q_s16(b1[wy].as_mut_ptr().add(off), raw) };
            let clipped = vqrshrun_n_s16::<5>(raw);
            unsafe { vst1_u8(b_clipped[wy].as_mut_ptr().add(off), clipped) };
        }
    }

    for r in 0..16 {
        let wy = r + 2;
        let g00 = unsafe { vld1q_u8(win[wy].as_ptr().add(2)) };
        let g10 = unsafe { vld1q_u8(win[wy].as_ptr().add(3)) };
        let g01 = unsafe { vld1q_u8(win[wy + 1].as_ptr().add(2)) };
        let b00 = unsafe { vld1q_u8(b_clipped[wy].as_ptr()) };
        let b01 = unsafe { vld1q_u8(b_clipped[wy + 1].as_ptr()) };

        // Compute h00 and h10 via NEON 6-tap vertical FIR
        let mut h00_arr = [0u8; 16];
        let mut h10_arr = [0u8; 16];
        let mut j00_arr = [0u8; 16];
        for half in 0..2 {
            let off = half * 8;
            let va0 =
                vreinterpretq_s16_u16(vmovl_u8(unsafe { vld1_u8(win[r].as_ptr().add(2 + off)) }));
            let vb0 = vreinterpretq_s16_u16(vmovl_u8(unsafe {
                vld1_u8(win[r + 1].as_ptr().add(2 + off))
            }));
            let vc0 = vreinterpretq_s16_u16(vmovl_u8(unsafe {
                vld1_u8(win[r + 2].as_ptr().add(2 + off))
            }));
            let vd0 = vreinterpretq_s16_u16(vmovl_u8(unsafe {
                vld1_u8(win[r + 3].as_ptr().add(2 + off))
            }));
            let ve0 = vreinterpretq_s16_u16(vmovl_u8(unsafe {
                vld1_u8(win[r + 4].as_ptr().add(2 + off))
            }));
            let vf0 = vreinterpretq_s16_u16(vmovl_u8(unsafe {
                vld1_u8(win[r + 5].as_ptr().add(2 + off))
            }));
            let raw_h0 = vaddq_s16(
                vsubq_s16(vaddq_s16(va0, vf0), vmulq_n_s16(vaddq_s16(vb0, ve0), 5)),
                vmulq_n_s16(vaddq_s16(vc0, vd0), 20),
            );
            unsafe { vst1_u8(h00_arr.as_mut_ptr().add(off), vqrshrun_n_s16::<5>(raw_h0)) };

            let va1 =
                vreinterpretq_s16_u16(vmovl_u8(unsafe { vld1_u8(win[r].as_ptr().add(3 + off)) }));
            let vb1 = vreinterpretq_s16_u16(vmovl_u8(unsafe {
                vld1_u8(win[r + 1].as_ptr().add(3 + off))
            }));
            let vc1 = vreinterpretq_s16_u16(vmovl_u8(unsafe {
                vld1_u8(win[r + 2].as_ptr().add(3 + off))
            }));
            let vd1 = vreinterpretq_s16_u16(vmovl_u8(unsafe {
                vld1_u8(win[r + 3].as_ptr().add(3 + off))
            }));
            let ve1 = vreinterpretq_s16_u16(vmovl_u8(unsafe {
                vld1_u8(win[r + 4].as_ptr().add(3 + off))
            }));
            let vf1 = vreinterpretq_s16_u16(vmovl_u8(unsafe {
                vld1_u8(win[r + 5].as_ptr().add(3 + off))
            }));
            let raw_h1 = vaddq_s16(
                vsubq_s16(vaddq_s16(va1, vf1), vmulq_n_s16(vaddq_s16(vb1, ve1), 5)),
                vmulq_n_s16(vaddq_s16(vc1, vd1), 20),
            );
            unsafe { vst1_u8(h10_arr.as_mut_ptr().add(off), vqrshrun_n_s16::<5>(raw_h1)) };
        }

        for c in 0..16 {
            let j_raw = fir6_raw(
                b1[r][c] as i32,
                b1[r + 1][c] as i32,
                b1[r + 2][c] as i32,
                b1[r + 3][c] as i32,
                b1[r + 4][c] as i32,
                b1[r + 5][c] as i32,
            );
            j00_arr[c] = clamp_u8((j_raw + 512) >> 10);
        }

        let h00 = unsafe { vld1q_u8(h00_arr.as_ptr()) };
        let h10 = unsafe { vld1q_u8(h10_arr.as_ptr()) };
        let j00 = unsafe { vld1q_u8(j00_arr.as_ptr()) };

        let out_row = match (mx, my) {
            (0, 0) => g00,
            (1, 0) => vrhaddq_u8(g00, b00),
            (2, 0) => b00,
            (3, 0) => vrhaddq_u8(b00, g10),
            (0, 1) => vrhaddq_u8(g00, h00),
            (1, 1) => vrhaddq_u8(b00, h00),
            (2, 1) => vrhaddq_u8(b00, j00),
            (3, 1) => vrhaddq_u8(b00, h10),
            (0, 2) => h00,
            (1, 2) => vrhaddq_u8(h00, j00),
            (2, 2) => j00,
            (3, 2) => vrhaddq_u8(j00, h10),
            (0, 3) => vrhaddq_u8(h00, g01),
            (1, 3) => vrhaddq_u8(h00, b01),
            (2, 3) => vrhaddq_u8(j00, b01),
            (3, 3) => vrhaddq_u8(h10, b01),
            _ => unreachable!(),
        };
        unsafe { vst1q_u8(dst.as_mut_ptr().add(r * 16), out_row) };
    }
}

/// Bilinear $1/8$-pel $8\times8$ H.264 chroma motion compensation (`fx, fy in 0..=7`)
/// per ITU-T H.264 8.4.2.2.2 (`ff_put_h264_chroma_mc8_neon`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn put_h264_chroma_mc8_u8(
    src: &[u8],
    src_stride: usize,
    width: usize,
    height: usize,
    cx0: isize,
    cy0: isize,
    fx: u8,
    fy: u8,
    dst: &mut [u8; 64],
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            put_h264_chroma_mc8_u8_neon(src, src_stride, width, height, cx0, cy0, fx, fy, dst);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        put_h264_chroma_mc8_u8_scalar(src, src_stride, width, height, cx0, cy0, fx, fy, dst);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn put_h264_chroma_mc8_u8_scalar(
    src: &[u8],
    src_stride: usize,
    width: usize,
    height: usize,
    cx0: isize,
    cy0: isize,
    fx: u8,
    fy: u8,
    dst: &mut [u8; 64],
) {
    let fx = (fx & 7) as u32;
    let fy = (fy & 7) as u32;
    let wa = (8 - fx) * (8 - fy);
    let wb = fx * (8 - fy);
    let wc = (8 - fx) * fy;
    let wd = fx * fy;

    for r in 0..8isize {
        let sy0 = cy0 + r;
        let sy1 = cy0 + r + 1;
        for c in 0..8isize {
            let sx0 = cx0 + c;
            let sx1 = cx0 + c + 1;
            let a = sample_clamped(src, src_stride, width, height, sx0, sy0) as u32;
            let b = sample_clamped(src, src_stride, width, height, sx1, sy0) as u32;
            let c_s = sample_clamped(src, src_stride, width, height, sx0, sy1) as u32;
            let d = sample_clamped(src, src_stride, width, height, sx1, sy1) as u32;
            dst[(r as usize) * 8 + (c as usize)] =
                ((wa * a + wb * b + wc * c_s + wd * d + 32) >> 6) as u8;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn put_h264_chroma_mc8_u8_neon(
    src: &[u8],
    src_stride: usize,
    width: usize,
    height: usize,
    cx0: isize,
    cy0: isize,
    fx: u8,
    fy: u8,
    dst: &mut [u8; 64],
) {
    use std::arch::aarch64::*;
    let fx = fx & 7;
    let fy = fy & 7;
    let mut win = [[0u8; 9]; 9];
    for (wy, row) in win.iter_mut().enumerate() {
        let sy = cy0 + (wy as isize);
        for (wx, slot) in row.iter_mut().enumerate() {
            let sx = cx0 + (wx as isize);
            *slot = sample_clamped(src, src_stride, width, height, sx, sy);
        }
    }

    let wa = vdup_n_u8((8 - fx) * (8 - fy));
    let wb = vdup_n_u8(fx * (8 - fy));
    let wc = vdup_n_u8((8 - fx) * fy);
    let wd = vdup_n_u8(fx * fy);

    for r in 0..8 {
        let a = unsafe { vld1_u8(win[r].as_ptr()) };
        let b = unsafe { vld1_u8(win[r].as_ptr().add(1)) };
        let c = unsafe { vld1_u8(win[r + 1].as_ptr()) };
        let d = unsafe { vld1_u8(win[r + 1].as_ptr().add(1)) };

        let mut acc = vmull_u8(a, wa);
        acc = vmlal_u8(acc, b, wb);
        acc = vmlal_u8(acc, c, wc);
        acc = vmlal_u8(acc, d, wd);
        let out = vrshrn_n_u16::<6>(acc);
        unsafe { vst1_u8(dst.as_mut_ptr().add(r * 8), out) };
    }
}

/// Filters a single 8-sample luma edge (`[p3, p2, p1, p0, q0, q1, q2, q3]`) in-place
/// according to ITU-T H.264 8.7.2.
#[inline]
pub fn h264_deblock_luma_line(line: &mut [u8; 8], qp: u8, bs: u8) {
    if bs == 0 || bs > 4 {
        return;
    }
    let idx = qp.min(51) as usize;
    let alpha = H264_DEBLOCK_ALPHA[idx] as i32;
    let beta = H264_DEBLOCK_BETA[idx] as i32;
    if alpha == 0 || beta == 0 {
        return;
    }

    let p3 = line[0] as i32;
    let p2 = line[1] as i32;
    let p1 = line[2] as i32;
    let p0 = line[3] as i32;
    let q0 = line[4] as i32;
    let q1 = line[5] as i32;
    let q2 = line[6] as i32;
    let q3 = line[7] as i32;

    if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
        return;
    }

    let ap = (p2 - p0).abs() < beta;
    let aq = (q2 - q0).abs() < beta;

    if bs < 4 {
        let tc0 = H264_DEBLOCK_TC0[idx][(bs - 1) as usize] as i32;
        let tc = tc0 + (ap as i32) + (aq as i32);
        let delta = ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
        line[3] = clamp_u8(p0 + delta);
        line[4] = clamp_u8(q0 - delta);

        if ap {
            let dp1 = ((p2 + ((p0 + q0 + 1) >> 1) - (p1 << 1)) >> 1).clamp(-tc0, tc0);
            line[2] = clamp_u8(p1 + dp1);
        }
        if aq {
            let dq1 = ((q2 + ((p0 + q0 + 1) >> 1) - (q1 << 1)) >> 1).clamp(-tc0, tc0);
            line[5] = clamp_u8(q1 + dq1);
        }
    } else {
        // Strong intra boundary filter (bS == 4)
        let small_gap = (p0 - q0).abs() < ((alpha >> 2) + 2);
        if ap && small_gap {
            line[3] = ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3) as u8;
            line[2] = ((p2 + p1 + p0 + q0 + 2) >> 2) as u8;
            line[1] = ((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3) as u8;
        } else {
            line[3] = ((2 * p1 + p0 + q1 + 2) >> 2) as u8;
        }
        if aq && small_gap {
            line[4] = ((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3) as u8;
            line[5] = ((p0 + q0 + q1 + q2 + 2) >> 2) as u8;
            line[6] = ((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3) as u8;
        } else {
            line[4] = ((2 * q1 + q0 + p1 + 2) >> 2) as u8;
        }
    }
}

/// Filters a 16-pixel horizontal luma edge at `(x0, edge_y)` (`edge_y >= 4`) in-place (`SIMD-0038`).
pub fn h264_loop_filter_luma_edge_h_u8(
    plane: &mut [u8],
    stride: usize,
    x0: usize,
    edge_y: usize,
    qp: u8,
    bs: [u8; 4],
) {
    if edge_y < 4 {
        return;
    }
    for (sub, &b) in bs.iter().enumerate() {
        if b == 0 {
            continue;
        }
        for dx in 0..4 {
            let x = x0 + sub * 4 + dx;
            let mut line = [0u8; 8];
            for (k, slot) in line.iter_mut().enumerate() {
                *slot = plane[(edge_y + k - 4) * stride + x];
            }
            h264_deblock_luma_line(&mut line, qp, b);
            for (k, &val) in line.iter().enumerate() {
                plane[(edge_y + k - 4) * stride + x] = val;
            }
        }
    }
}

/// Filters a 16-pixel vertical luma edge at `(edge_x, y0)` (`edge_x >= 4`) in-place (`SIMD-0038`).
pub fn h264_loop_filter_luma_edge_v_u8(
    plane: &mut [u8],
    stride: usize,
    edge_x: usize,
    y0: usize,
    qp: u8,
    bs: [u8; 4],
) {
    if edge_x < 4 {
        return;
    }
    for (sub, &b) in bs.iter().enumerate() {
        if b == 0 {
            continue;
        }
        for dy in 0..4 {
            let y = y0 + sub * 4 + dy;
            let row_off = y * stride + (edge_x - 4);
            let mut line = [0u8; 8];
            line.copy_from_slice(&plane[row_off..row_off + 8]);
            h264_deblock_luma_line(&mut line, qp, b);
            plane[row_off..row_off + 8].copy_from_slice(&line);
        }
    }
}

/// Filters a single 4-sample chroma edge (`[p1, p0, q0, q1]`) in-place according to ITU-T H.264 8.7.2.
#[inline]
pub fn h264_deblock_chroma_line(line: &mut [u8; 4], qpc: u8, bs: u8) {
    if bs == 0 || bs > 4 {
        return;
    }
    let idx = qpc.min(51) as usize;
    let alpha = H264_DEBLOCK_ALPHA[idx] as i32;
    let beta = H264_DEBLOCK_BETA[idx] as i32;
    if alpha == 0 || beta == 0 {
        return;
    }

    let p1 = line[0] as i32;
    let p0 = line[1] as i32;
    let q0 = line[2] as i32;
    let q1 = line[3] as i32;

    if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
        return;
    }

    if bs < 4 {
        let tc = (H264_DEBLOCK_TC0[idx][(bs - 1) as usize] as i32) + 1;
        let delta = ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
        line[1] = clamp_u8(p0 + delta);
        line[2] = clamp_u8(q0 - delta);
    } else {
        line[1] = ((2 * p1 + p0 + q1 + 2) >> 2) as u8;
        line[2] = ((2 * q1 + q0 + p1 + 2) >> 2) as u8;
    }
}

/// Filters an 8-pixel horizontal chroma edge at `(cx0, edge_cy)` (`edge_cy >= 2`) in-place.
pub fn h264_loop_filter_chroma_edge_h_u8(
    plane: &mut [u8],
    stride: usize,
    cx0: usize,
    edge_cy: usize,
    qpc: u8,
    bs: [u8; 4],
) {
    if edge_cy < 2 {
        return;
    }
    for (sub, &b) in bs.iter().enumerate() {
        if b == 0 {
            continue;
        }
        for dx in 0..2 {
            let x = cx0 + sub * 2 + dx;
            let mut line = [
                plane[(edge_cy - 2) * stride + x],
                plane[(edge_cy - 1) * stride + x],
                plane[edge_cy * stride + x],
                plane[(edge_cy + 1) * stride + x],
            ];
            h264_deblock_chroma_line(&mut line, qpc, b);
            plane[(edge_cy - 1) * stride + x] = line[1];
            plane[edge_cy * stride + x] = line[2];
        }
    }
}

/// Filters an 8-pixel vertical chroma edge at `(edge_cx, cy0)` (`edge_cx >= 2`) in-place.
pub fn h264_loop_filter_chroma_edge_v_u8(
    plane: &mut [u8],
    stride: usize,
    edge_cx: usize,
    cy0: usize,
    qpc: u8,
    bs: [u8; 4],
) {
    if edge_cx < 2 {
        return;
    }
    for (sub, &b) in bs.iter().enumerate() {
        if b == 0 {
            continue;
        }
        for dy in 0..2 {
            let y = cy0 + sub * 2 + dy;
            let off = y * stride + (edge_cx - 2);
            let mut line = [plane[off], plane[off + 1], plane[off + 2], plane[off + 3]];
            h264_deblock_chroma_line(&mut line, qpc, b);
            plane[off + 1] = line[1];
            plane[off + 2] = line[2];
        }
    }
}

/// ITU-T H.264 8.3.1.2 Intra $4\times4$ 9-mode spatial prediction (`mode in 0..=8`).
///
/// - `top`: 8 top reference samples `[A, B, C, D, E, F, G, H]` (where `E..H` are top-right or replicated `D`).
/// - `left`: 4 left reference samples `[I, J, K, L]`.
/// - `top_left`: top-left corner sample `M`.
#[inline]
pub fn predict_intra_4x4_u8(
    mode: u8,
    top: Option<&[u8; 8]>,
    left: Option<&[u8; 4]>,
    top_left: u8,
    dst: &mut [u8],
    stride: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if dst.len() >= 3 * stride + 4 {
            unsafe { predict_intra_4x4_u8_neon(mode, top, left, top_left, dst, stride) };
            return;
        }
    }
    predict_intra_4x4_u8_scalar(mode, top, left, top_left, dst, stride);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn predict_intra_4x4_u8_neon(
    mode: u8,
    top: Option<&[u8; 8]>,
    left: Option<&[u8; 4]>,
    top_left: u8,
    dst: &mut [u8],
    stride: usize,
) {
    use std::arch::aarch64::*;
    if mode == 0
        && let Some(t) = top
    {
        let v32 = u32::from_ne_bytes([t[0], t[1], t[2], t[3]]);
        let vec = vdup_n_u32(v32);
        for y in 0..4 {
            unsafe {
                vst1_lane_u32::<0>(dst.as_mut_ptr().add(y * stride) as *mut u32, vec);
            }
        }
        return;
    }
    predict_intra_4x4_u8_scalar(mode, top, left, top_left, dst, stride);
}

pub fn predict_intra_4x4_u8_scalar(
    mode: u8,
    top: Option<&[u8; 8]>,
    left: Option<&[u8; 4]>,
    top_left: u8,
    dst: &mut [u8],
    stride: usize,
) {
    let t = top.copied().unwrap_or([128u8; 8]);
    let l = left.copied().unwrap_or([128u8; 4]);
    let a = t[0] as u32;
    let b = t[1] as u32;
    let c = t[2] as u32;
    let d = t[3] as u32;
    let e = t[4] as u32;
    let f = t[5] as u32;
    let g = t[6] as u32;
    let h = t[7] as u32;
    let i = l[0] as u32;
    let j = l[1] as u32;
    let k = l[2] as u32;
    let l_s = l[3] as u32;
    let m = top_left as u32;

    let avg2 = |x: u32, y: u32| -> u8 { ((x + y + 1) >> 1) as u8 };
    let avg3 = |x: u32, y: u32, z: u32| -> u8 { ((x + 2 * y + z + 2) >> 2) as u8 };

    match mode {
        0 => {
            // Vertical
            for y in 0..4 {
                dst[y * stride..y * stride + 4].copy_from_slice(&t[0..4]);
            }
        }
        1 => {
            // Horizontal
            for y in 0..4 {
                dst[y * stride..y * stride + 4].fill(l[y]);
            }
        }
        2 => {
            // DC
            let dc = match (top, left) {
                (Some(_), Some(_)) => ((a + b + c + d + i + j + k + l_s + 4) >> 3) as u8,
                (Some(_), None) => ((a + b + c + d + 2) >> 2) as u8,
                (None, Some(_)) => ((i + j + k + l_s + 2) >> 2) as u8,
                (None, None) => 128u8,
            };
            for y in 0..4 {
                dst[y * stride..y * stride + 4].fill(dc);
            }
        }
        3 => {
            // Diagonal Down-Left
            let p0 = avg3(a, b, c);
            let p1 = avg3(b, c, d);
            let p2 = avg3(c, d, e);
            let p3 = avg3(d, e, f);
            let p4 = avg3(e, f, g);
            let p5 = avg3(f, g, h);
            let p6 = avg3(g, h, h);
            dst[0..4].copy_from_slice(&[p0, p1, p2, p3]);
            dst[stride..stride + 4].copy_from_slice(&[p1, p2, p3, p4]);
            dst[2 * stride..2 * stride + 4].copy_from_slice(&[p2, p3, p4, p5]);
            dst[3 * stride..3 * stride + 4].copy_from_slice(&[p3, p4, p5, p6]);
        }
        4 => {
            // Diagonal Down-Right
            let v_lm = avg3(k, j, i);
            let v_kj = avg3(l_s, k, j);
            let v_ji = avg3(j, i, m);
            let v_im = avg3(i, m, a);
            let v_ma = avg3(m, a, b);
            let v_ab = avg3(a, b, c);
            let v_bc = avg3(b, c, d);
            dst[0..4].copy_from_slice(&[v_im, v_ma, v_ab, v_bc]);
            dst[stride..stride + 4].copy_from_slice(&[v_ji, v_im, v_ma, v_ab]);
            dst[2 * stride..2 * stride + 4].copy_from_slice(&[v_lm, v_ji, v_im, v_ma]);
            dst[3 * stride..3 * stride + 4].copy_from_slice(&[v_kj, v_lm, v_ji, v_im]);
        }
        5 => {
            // Vertical-Right
            let r0 = [avg2(m, a), avg2(a, b), avg2(b, c), avg2(c, d)];
            let r1 = [avg3(i, m, a), avg3(m, a, b), avg3(a, b, c), avg3(b, c, d)];
            let r2 = [avg3(j, i, m), r0[0], r0[1], r0[2]];
            let r3 = [avg3(k, j, i), r1[0], r1[1], r1[2]];
            dst[0..4].copy_from_slice(&r0);
            dst[stride..stride + 4].copy_from_slice(&r1);
            dst[2 * stride..2 * stride + 4].copy_from_slice(&r2);
            dst[3 * stride..3 * stride + 4].copy_from_slice(&r3);
        }
        6 => {
            // Horizontal-Down
            let r0 = [avg2(m, i), avg3(i, m, a), avg3(m, a, b), avg3(a, b, c)];
            let r1 = [avg2(i, j), avg3(m, i, j), r0[0], r0[1]];
            let r2 = [avg2(j, k), avg3(i, j, k), r1[0], r1[1]];
            let r3 = [avg2(k, l_s), avg3(j, k, l_s), r2[0], r2[1]];
            dst[0..4].copy_from_slice(&r0);
            dst[stride..stride + 4].copy_from_slice(&r1);
            dst[2 * stride..2 * stride + 4].copy_from_slice(&r2);
            dst[3 * stride..3 * stride + 4].copy_from_slice(&r3);
        }
        7 => {
            // Vertical-Left
            let r0 = [avg2(a, b), avg2(b, c), avg2(c, d), avg2(d, e)];
            let r1 = [avg3(a, b, c), avg3(b, c, d), avg3(c, d, e), avg3(d, e, f)];
            let r2 = [r0[1], r0[2], r0[3], avg2(e, f)];
            let r3 = [r1[1], r1[2], r1[3], avg3(e, f, g)];
            dst[0..4].copy_from_slice(&r0);
            dst[stride..stride + 4].copy_from_slice(&r1);
            dst[2 * stride..2 * stride + 4].copy_from_slice(&r2);
            dst[3 * stride..3 * stride + 4].copy_from_slice(&r3);
        }
        _ => {
            // Horizontal-Up (mode 8)
            let p0 = avg2(i, j);
            let p1 = avg3(i, j, k);
            let p2 = avg2(j, k);
            let p3 = avg3(j, k, l_s);
            let p4 = avg2(k, l_s);
            let p5 = avg3(k, l_s, l_s);
            let p6 = l[3];
            dst[0..4].copy_from_slice(&[p0, p1, p2, p3]);
            dst[stride..stride + 4].copy_from_slice(&[p2, p3, p4, p5]);
            dst[2 * stride..2 * stride + 4].copy_from_slice(&[p4, p5, p6, p6]);
            dst[3 * stride..3 * stride + 4].copy_from_slice(&[p6, p6, p6, p6]);
        }
    }
}
