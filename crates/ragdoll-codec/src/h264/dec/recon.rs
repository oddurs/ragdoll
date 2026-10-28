//! Residual reconstruction (ITU-T H.264 8.5) and inter prediction sample generation (8.4.2).

#![allow(clippy::needless_range_loop)]

use ragdoll_simd::{
    chroma_mc_cols, h264_avg, h264_chroma_mc, h264_idct4_add, h264_luma_mc, h264_weight_bi,
    h264_weight_uni, luma_mc_cols,
};

use super::ps::WeightScales;
use super::tables::{NORM_ADJUST_4X4, NORM_ADJUST_8X8, SCAN4X4, SCAN8X8, norm4_class, norm8_class};
use crate::cabac::ResidualCoeffs;

/// `LevelScale4x4` and `LevelScale8x8` for the active scaling matrices, indexed by raster position.
pub struct Dequant {
    /// `[list][qp % 6][raster]`, lists Intra Y/Cb/Cr then Inter Y/Cb/Cr.
    pub ls4: [[[i32; 16]; 6]; 6],
    /// `[list][qp % 6][raster]`, lists Intra Y then Inter Y.
    pub ls8: [[[i32; 64]; 6]; 2],
}

impl Dequant {
    pub fn new(ws: &WeightScales) -> Self {
        let mut ls4 = [[[0i32; 16]; 6]; 6];
        let mut ls8 = [[[0i32; 64]; 6]; 2];
        for list in 0..6 {
            for m in 0..6 {
                for k in 0..16 {
                    let pos = SCAN4X4[k];
                    ls4[list][m][pos] =
                        ws.w4[list][k] as i32 * NORM_ADJUST_4X4[m][norm4_class(pos)];
                }
            }
        }
        for list in 0..2 {
            for m in 0..6 {
                for k in 0..64 {
                    let pos = SCAN8X8[k];
                    ls8[list][m][pos] =
                        ws.w8[list][k] as i32 * NORM_ADJUST_8X8[m][norm8_class(pos)];
                }
            }
        }
        Self { ls4, ls8 }
    }

    /// Scales scanned 4x4 coefficients `sc[k]` (raster position `scan[k + off]`) into `out`
    /// (8.5.12.1). Every position is written: a zero coefficient scales to zero, so no branch
    /// on the (unpredictable) coefficient values is needed.
    #[inline]
    pub fn scale4_block(
        &self,
        list: usize,
        qp: i32,
        sc: &[i16],
        off: usize,
        scan: &[usize; 16],
        out: &mut [i32; 16],
    ) {
        let ls = &self.ls4[list][(qp % 6) as usize];
        let scan = &scan[off..off + sc.len()];
        if qp >= 24 {
            let sh = qp / 6 - 4;
            for (&c, &pos) in sc.iter().zip(scan) {
                out[pos] = (c as i32 * ls[pos]) << sh;
            }
        } else {
            let (sh, rnd) = (4 - qp / 6, 1 << (3 - qp / 6));
            for (&c, &pos) in sc.iter().zip(scan) {
                out[pos] = (c as i32 * ls[pos] + rnd) >> sh;
            }
        }
    }

    /// [`Dequant::scale4_block`] for the `n` nonzero coefficients of `c`, whose scan positions
    /// start at `off`; other positions of `out` are left as they are.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn scale4_coeffs(
        &self,
        list: usize,
        qp: i32,
        c: &ResidualCoeffs,
        n: usize,
        off: usize,
        scan: &[usize; 16],
        out: &mut [i32; 16],
    ) {
        let ls = &self.ls4[list][(qp % 6) as usize];
        let (pos, level) = (&c.pos[..n], &c.level[..n]);
        if qp >= 24 {
            let sh = qp / 6 - 4;
            for (&k, &v) in pos.iter().zip(level) {
                let p = scan[(k as usize + off) & 15] & 15;
                out[p] = (v as i32 * ls[p]) << sh;
            }
        } else {
            let (sh, rnd) = (4 - qp / 6, 1 << (3 - qp / 6));
            for (&k, &v) in pos.iter().zip(level) {
                let p = scan[(k as usize + off) & 15] & 15;
                out[p] = (v as i32 * ls[p] + rnd) >> sh;
            }
        }
    }

    /// [`Dequant::scale4_coeffs`] for an 8x8 block.
    #[inline]
    pub fn scale8_coeffs(
        &self,
        list: usize,
        qp: i32,
        c: &ResidualCoeffs,
        n: usize,
        scan: &[usize; 64],
        out: &mut [i32; 64],
    ) {
        let ls = &self.ls8[list][(qp % 6) as usize];
        let (pos, level) = (&c.pos[..n], &c.level[..n]);
        if qp >= 36 {
            let sh = qp / 6 - 6;
            for (&k, &v) in pos.iter().zip(level) {
                let p = scan[k as usize & 63];
                out[p & 63] = (v as i32 * ls[p & 63]) << sh;
            }
        } else {
            let (sh, rnd) = (6 - qp / 6, 1 << (5 - qp / 6));
            for (&k, &v) in pos.iter().zip(level) {
                let p = scan[k as usize & 63];
                out[p & 63] = (v as i32 * ls[p & 63] + rnd) >> sh;
            }
        }
    }

    /// [`Dequant::scale4_block`] for an 8x8 block (8.5.13.1).
    #[inline]
    pub fn scale8_block(
        &self,
        list: usize,
        qp: i32,
        sc: &[i16; 64],
        scan: &[usize; 64],
        out: &mut [i32; 64],
    ) {
        let ls = &self.ls8[list][(qp % 6) as usize];
        if qp >= 36 {
            let sh = qp / 6 - 6;
            for (&c, &pos) in sc.iter().zip(scan) {
                out[pos] = (c as i32 * ls[pos]) << sh;
            }
        } else {
            let (sh, rnd) = (6 - qp / 6, 1 << (5 - qp / 6));
            for (&c, &pos) in sc.iter().zip(scan) {
                out[pos] = (c as i32 * ls[pos] + rnd) >> sh;
            }
        }
    }
}

/// Intra_16x16 luma DC: inverse Hadamard and scaling (8.5.10). `c` is raster order; returns `dcY` raster.
pub fn luma_dc(dq: &Dequant, qp: i32, c: &[i32; 16]) -> [i32; 16] {
    let mut t = [0i32; 16];
    for i in 0..4 {
        let r = &c[i * 4..i * 4 + 4];
        let (a, b) = (r[0] + r[1], r[0] - r[1]);
        let (cc, d) = (r[2] + r[3], r[2] - r[3]);
        t[i * 4] = a + cc;
        t[i * 4 + 1] = a - cc;
        t[i * 4 + 2] = b - d;
        t[i * 4 + 3] = b + d;
    }
    // t = c * H; now f = H * t. H's columns are (1,1,1,1), (1,1,-1,-1), (1,-1,-1,1), (1,-1,1,-1).
    let mut f = [0i32; 16];
    for j in 0..4 {
        let (r0, r1, r2, r3) = (t[j], t[4 + j], t[8 + j], t[12 + j]);
        let (a, b) = (r0 + r1, r0 - r1);
        let (cc, d) = (r2 + r3, r2 - r3);
        f[j] = a + cc;
        f[4 + j] = a - cc;
        f[8 + j] = b - d;
        f[12 + j] = b + d;
    }
    let ls = dq.ls4[0][(qp % 6) as usize][0];
    let mut out = [0i32; 16];
    for i in 0..16 {
        out[i] = if qp >= 36 {
            (f[i] * ls) << (qp / 6 - 6)
        } else {
            (f[i] * ls + (1 << (5 - qp / 6))) >> (6 - qp / 6)
        };
    }
    out
}

/// Chroma DC for 4:2:0: 2x2 transform and scaling (8.5.11). `c` is raster `[c00, c01, c10, c11]`.
pub fn chroma_dc(dq: &Dequant, list: usize, qpc: i32, c: &[i32; 4]) -> [i32; 4] {
    let f = [
        c[0] + c[1] + c[2] + c[3],
        c[0] - c[1] + c[2] - c[3],
        c[0] + c[1] - c[2] - c[3],
        c[0] - c[1] - c[2] + c[3],
    ];
    let ls = dq.ls4[list][(qpc % 6) as usize][0];
    f.map(|v| ((v * ls) << (qpc / 6)) >> 5)
}

#[inline]
fn clip(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Inverse 4x4 transform of raster coefficients `d` and addition to `dst` (8.5.12.2); `d` is
/// zeroed for the next block.
#[inline]
pub fn idct4_add(dst: &mut [u8], stride: usize, d: &mut [i32; 16]) {
    h264_idct4_add(dst, stride, d);
    *d = [0; 16];
}

fn idct8_1d(d: [i32; 8]) -> [i32; 8] {
    let a0 = d[0] + d[4];
    let a1 = -d[3] + d[5] - d[7] - (d[7] >> 1);
    let a2 = d[0] - d[4];
    let a3 = d[1] + d[7] - d[3] - (d[3] >> 1);
    let a4 = (d[2] >> 1) - d[6];
    let a5 = -d[1] + d[7] + d[5] + (d[5] >> 1);
    let a6 = d[2] + (d[6] >> 1);
    let a7 = d[3] + d[5] + d[1] + (d[1] >> 1);
    let b0 = a0 + a6;
    let b1 = a1 + (a7 >> 2);
    let b2 = a2 + a4;
    let b3 = a3 + (a5 >> 2);
    let b4 = a2 - a4;
    let b5 = (a3 >> 2) - a5;
    let b6 = a0 - a6;
    let b7 = a7 - (a1 >> 2);
    [
        b0 + b7,
        b2 + b5,
        b4 + b3,
        b6 + b1,
        b6 - b1,
        b4 - b3,
        b2 - b5,
        b0 - b7,
    ]
}

/// Inverse 8x8 transform of raster coefficients `d` and addition to `dst` (8.5.13.2); `d` is
/// zeroed for the next block.
pub fn idct8_add(dst: &mut [u8], stride: usize, d: &mut [i32; 64]) {
    let mut g = [0i32; 64];
    for i in 0..8 {
        let mut row = [0i32; 8];
        row.copy_from_slice(&d[i * 8..i * 8 + 8]);
        g[i * 8..i * 8 + 8].copy_from_slice(&idct8_1d(row));
    }
    for j in 0..8 {
        let col = idct8_1d(std::array::from_fn(|i| g[i * 8 + j]));
        for (i, v) in col.iter().enumerate() {
            let p = &mut dst[i * stride + j];
            *p = clip(*p as i32 + ((v + 32) >> 6));
        }
    }
    *d = [0; 64];
}

/// Columns of the stack window an edge-extended luma or chroma block is copied into.
const EMU_STRIDE: usize = 24;

/// Luma prediction samples for a `w`x`h` block at pixel `(x, y)` displaced by quarter-pel `mv`,
/// written to `dst` with stride `ds`. Blocks whose filter taps reach past the picture read an
/// edge-extended copy (8.4.2.2.1 clamps reference coordinates to the picture).
#[allow(clippy::too_many_arguments)]
pub fn mc_luma(
    src: &[u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    mv: [i16; 2],
    dst: &mut [u8],
    ds: usize,
) {
    let ix = x as isize + (mv[0] as isize >> 2);
    let iy = y as isize + (mv[1] as isize >> 2);
    let (fx, fy) = ((mv[0] & 3) as u8, (mv[1] & 3) as u8);
    let cols = luma_mc_cols(w);
    if ix >= 2 && iy >= 2 && ix as usize + cols <= width && iy as usize + h + 3 <= height {
        h264_luma_mc(
            dst,
            ds,
            src,
            iy as usize * width + ix as usize,
            width,
            w,
            h,
            fx,
            fy,
        );
        return;
    }
    // Whole-sample blocks at the picture edge need no interpolation margins.
    if fx == 0 && fy == 0 && copy_inside(src, width, height, ix, iy, w, h, dst, ds) {
        return;
    }
    let mut win = [0u8; EMU_STRIDE * 21];
    edge_window(
        &mut win,
        src,
        width,
        height,
        ix - 2,
        iy - 2,
        cols + 2,
        h + 5,
    );
    h264_luma_mc(dst, ds, &win, 2 * EMU_STRIDE + 2, EMU_STRIDE, w, h, fx, fy);
}

/// Whole-sample prediction: copies the `w`x`h` block at `(x, y)` of a `width`x`height` plane to
/// `dst` when it lies inside the plane, which needs none of the interpolation margins.
#[allow(clippy::too_many_arguments)]
#[inline]
fn copy_inside(
    src: &[u8],
    width: usize,
    height: usize,
    x: isize,
    y: isize,
    w: usize,
    h: usize,
    dst: &mut [u8],
    ds: usize,
) -> bool {
    if x < 0 || y < 0 || x as usize + w > width || y as usize + h > height {
        return false;
    }
    let o = y as usize * width + x as usize;
    fn rows<const W: usize>(src: &[u8], o: usize, ss: usize, h: usize, dst: &mut [u8], ds: usize) {
        for r in 0..h {
            let s: &[u8; W] = src[o + r * ss..][..W].try_into().expect("W samples");
            dst[r * ds..r * ds + W].copy_from_slice(s);
        }
    }
    match w {
        16 => rows::<16>(src, o, width, h, dst, ds),
        8 => rows::<8>(src, o, width, h, dst, ds),
        4 => rows::<4>(src, o, width, h, dst, ds),
        _ => rows::<2>(src, o, width, h, dst, ds),
    }
    true
}

/// Copies the `cols`x`rows` window at `(x0, y0)` of a `width`x`height` plane into `win` (stride
/// [`EMU_STRIDE`]), clamping coordinates to the plane (FFmpeg's `emulated_edge_mc`).
#[allow(clippy::too_many_arguments)]
fn edge_window(
    win: &mut [u8],
    src: &[u8],
    width: usize,
    height: usize,
    x0: isize,
    y0: isize,
    cols: usize,
    rows: usize,
) {
    let inside = x0 >= 0 && x0 as usize + cols <= width;
    for r in 0..rows {
        let sy = (y0 + r as isize).clamp(0, height as isize - 1) as usize;
        let row = &src[sy * width..sy * width + width];
        let out = &mut win[r * EMU_STRIDE..r * EMU_STRIDE + cols];
        if inside {
            out.copy_from_slice(&row[x0 as usize..x0 as usize + cols]);
        } else {
            for (c, o) in out.iter_mut().enumerate() {
                *o = row[(x0 + c as isize).clamp(0, width as isize - 1) as usize];
            }
        }
    }
}

/// Chroma (4:2:0) prediction samples for a `w`x`h` chroma block at chroma pixel `(x, y)`,
/// written to `dst` with stride `ds`.
#[allow(clippy::too_many_arguments)]
pub fn mc_chroma(
    src: &[u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    mv: [i16; 2],
    dst: &mut [u8],
    ds: usize,
) {
    let ix = x as isize + (mv[0] as isize >> 3);
    let iy = y as isize + (mv[1] as isize >> 3);
    let (fx, fy) = ((mv[0] & 7) as u8, (mv[1] & 7) as u8);
    let cols = chroma_mc_cols(w);
    if ix >= 0 && iy >= 0 && ix as usize + cols <= width && iy as usize + h < height {
        h264_chroma_mc(
            dst,
            ds,
            src,
            iy as usize * width + ix as usize,
            width,
            w,
            h,
            fx,
            fy,
        );
        return;
    }
    if fx == 0 && fy == 0 && copy_inside(src, width, height, ix, iy, w, h, dst, ds) {
        return;
    }
    let mut win = [0u8; EMU_STRIDE * 9];
    edge_window(&mut win, src, width, height, ix, iy, cols, h + 1);
    h264_chroma_mc(dst, ds, &win, 0, EMU_STRIDE, w, h, fx, fy);
}

/// Predicts a `w`x`h` block from the lists in `used` and combines them into `dst` (stride `ds`)
/// per 8.4.2.3. `predict(list, out, stride)` writes one list's prediction samples.
#[allow(clippy::too_many_arguments)]
pub fn predict_block(
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
    used: [bool; 2],
    wt: Weighting,
    scratch: &mut [[u8; 256]; 2],
    mut predict: impl FnMut(usize, &mut [u8], usize),
) {
    match (used, effective_weighting(used, wt)) {
        ([true, false], Weighting::Default) => predict(0, dst, ds),
        ([false, true], Weighting::Default) => predict(1, dst, ds),
        ([true, true], Weighting::Default) => {
            predict(0, dst, ds);
            let t = &mut scratch[0];
            predict(1, t, 16);
            h264_avg(dst, ds, t, 16, w, h);
        }
        (_, Weighting::Weighted(log_wd, [(w0, o0), (w1, o1)])) => {
            let t = scratch;
            for list in 0..2 {
                if used[list] {
                    predict(list, &mut t[list], 16);
                }
            }
            match used {
                [true, true] => {
                    let off = (o0 + o1 + 1) >> 1;
                    h264_weight_bi(dst, ds, &t[0], &t[1], 16, w, h, log_wd, w0, w1, off);
                }
                [true, false] => h264_weight_uni(dst, ds, &t[0], 16, w, h, log_wd, w0, o0),
                [false, true] => h264_weight_uni(dst, ds, &t[1], 16, w, h, log_wd, w1, o1),
                [false, false] => {}
            }
        }
        ([false, false], Weighting::Default) => {}
    }
}

/// `wt` for the lists in `used`, with weights that reduce exactly to default prediction
/// (2^logWD with zero offsets) replaced by [`Weighting::Default`]:
/// ((a + b + 1) * 2^L) >> (L + 1) = (a + b + 1) >> 1, and (a * 2^L + 2^(L-1)) >> L = a.
pub fn effective_weighting(used: [bool; 2], wt: Weighting) -> Weighting {
    let unit = |l: u32, (w, o): (i32, i32)| w == 1 << l && o == 0;
    match (used, wt) {
        ([true, true], Weighting::Weighted(l, [e0, e1])) if unit(l, e0) && unit(l, e1) => {
            Weighting::Default
        }
        ([true, false], Weighting::Weighted(l, [e0, _])) if unit(l, e0) => Weighting::Default,
        ([false, true], Weighting::Weighted(l, [_, e1])) if unit(l, e1) => Weighting::Default,
        _ => wt,
    }
}

/// How the two prediction lists are combined for one block.
#[derive(Clone, Copy, Debug)]
pub enum Weighting {
    Default,
    /// `(logWD, [(w0, o0), (w1, o1)])` for one component.
    Weighted(u32, [(i32, i32); 2]),
}
