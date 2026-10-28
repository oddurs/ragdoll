//! Residual reconstruction (ITU-T H.264 8.5) and inter prediction sample generation (8.4.2).

#![allow(clippy::needless_range_loop)]

use ragdoll_simd::{put_h264_chroma_mc8_u8, put_h264_qpel16_mc_u8};

use super::ps::WeightScales;
use super::tables::{NORM_ADJUST_4X4, NORM_ADJUST_8X8, SCAN4X4, SCAN8X8, norm4_class, norm8_class};

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

    /// Scales one 4x4 AC (or full) coefficient at raster `pos` (8.5.12.1).
    #[inline]
    pub fn scale4(&self, list: usize, qp: i32, pos: usize, c: i32) -> i32 {
        let ls = self.ls4[list][(qp % 6) as usize][pos];
        if qp >= 24 {
            (c * ls) << (qp / 6 - 4)
        } else {
            (c * ls + (1 << (3 - qp / 6))) >> (4 - qp / 6)
        }
    }

    /// Scales one 8x8 coefficient at raster `pos` (8.5.13.1).
    #[inline]
    pub fn scale8(&self, list: usize, qp: i32, pos: usize, c: i32) -> i32 {
        let ls = self.ls8[list][(qp % 6) as usize][pos];
        if qp >= 36 {
            (c * ls) << (qp / 6 - 6)
        } else {
            (c * ls + (1 << (5 - qp / 6))) >> (6 - qp / 6)
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

/// Inverse 4x4 transform of raster coefficients `d` and addition to `dst` (8.5.12.2).
pub fn idct4_add(dst: &mut [u8], stride: usize, d: &[i32; 16]) {
    let mut f = [0i32; 16];
    for i in 0..4 {
        let r = &d[i * 4..i * 4 + 4];
        let e0 = r[0] + r[2];
        let e1 = r[0] - r[2];
        let e2 = (r[1] >> 1) - r[3];
        let e3 = r[1] + (r[3] >> 1);
        f[i * 4] = e0 + e3;
        f[i * 4 + 1] = e1 + e2;
        f[i * 4 + 2] = e1 - e2;
        f[i * 4 + 3] = e0 - e3;
    }
    for j in 0..4 {
        let (f0, f1, f2, f3) = (f[j], f[4 + j], f[8 + j], f[12 + j]);
        let g0 = f0 + f2;
        let g1 = f0 - f2;
        let g2 = (f1 >> 1) - f3;
        let g3 = f1 + (f3 >> 1);
        let h = [g0 + g3, g1 + g2, g1 - g2, g0 - g3];
        for (i, v) in h.iter().enumerate() {
            let p = &mut dst[i * stride + j];
            *p = clip(*p as i32 + ((v + 32) >> 6));
        }
    }
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

/// Inverse 8x8 transform of raster coefficients `d` and addition to `dst` (8.5.13.2).
pub fn idct8_add(dst: &mut [u8], stride: usize, d: &[i32; 64]) {
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
}

/// Luma prediction samples for a `w`x`h` block at pixel `(x, y)` displaced by quarter-pel `mv`.
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
    out: &mut [u8],
) {
    let mut tmp = [0u8; 256];
    let ix = x as isize + (mv[0] as isize >> 2);
    let iy = y as isize + (mv[1] as isize >> 2);
    put_h264_qpel16_mc_u8(
        src,
        width,
        width,
        height,
        ix,
        iy,
        (mv[0] & 3) as u8,
        (mv[1] & 3) as u8,
        &mut tmp,
    );
    for r in 0..h {
        out[r * w..r * w + w].copy_from_slice(&tmp[r * 16..r * 16 + w]);
    }
}

/// Chroma (4:2:0) prediction samples for a `w`x`h` chroma block at chroma pixel `(x, y)`.
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
    out: &mut [u8],
) {
    let mut tmp = [0u8; 64];
    let ix = x as isize + (mv[0] as isize >> 3);
    let iy = y as isize + (mv[1] as isize >> 3);
    put_h264_chroma_mc8_u8(
        src,
        width,
        width,
        height,
        ix,
        iy,
        (mv[0] & 7) as u8,
        (mv[1] & 7) as u8,
        &mut tmp,
    );
    for r in 0..h {
        out[r * w..r * w + w].copy_from_slice(&tmp[r * 8..r * 8 + w]);
    }
}

/// How the two prediction lists are combined for one block.
#[derive(Clone, Copy, Debug)]
pub enum Weighting {
    Default,
    /// `(logWD, [(w0, o0), (w1, o1)])` for one component.
    Weighted(u32, [(i32, i32); 2]),
}

/// Combines prediction samples `p0`/`p1` (either may be absent) into `dst` (8.4.2.3).
pub fn weighted_combine(p0: Option<&[u8]>, p1: Option<&[u8]>, wt: Weighting, dst: &mut [u8]) {
    match (p0, p1, wt) {
        (Some(a), Some(b), Weighting::Default) => {
            for i in 0..dst.len() {
                dst[i] = ((a[i] as u16 + b[i] as u16 + 1) >> 1) as u8;
            }
        }
        (Some(a), None, Weighting::Default) | (None, Some(a), Weighting::Default) => {
            dst.copy_from_slice(&a[..dst.len()]);
        }
        (Some(a), Some(b), Weighting::Weighted(log_wd, [(w0, o0), (w1, o1)])) => {
            let round = 1 << log_wd;
            let off = (o0 + o1 + 1) >> 1;
            for i in 0..dst.len() {
                let v = ((a[i] as i32 * w0 + b[i] as i32 * w1 + round) >> (log_wd + 1)) + off;
                dst[i] = clip(v);
            }
        }
        (Some(a), None, Weighting::Weighted(log_wd, [(w, o), _]))
        | (None, Some(a), Weighting::Weighted(log_wd, [_, (w, o)])) => {
            for i in 0..dst.len() {
                let v = if log_wd >= 1 {
                    ((a[i] as i32 * w + (1 << (log_wd - 1))) >> log_wd) + o
                } else {
                    a[i] as i32 * w + o
                };
                dst[i] = clip(v);
            }
        }
        (None, None, _) => {}
    }
}
