//! Intra prediction (ITU-T H.264 8.3).

/// Neighbouring samples of an NxN luma block (`N` = 4 or 8).
///
/// `top` holds `p[0..2N, -1]` (top-right already substituted when unavailable), `left` holds
/// `p[-1, 0..N]`, `tl` is `p[-1, -1]`.
pub struct Edges {
    pub top: [u8; 16],
    pub left: [u8; 8],
    pub tl: u8,
    pub has_top: bool,
    pub has_left: bool,
    pub has_tl: bool,
}

impl Edges {
    /// Gathers the edges of the `n`x`n` block at pixel `(x, y)` from `plane`.
    #[allow(clippy::too_many_arguments)]
    pub fn gather(
        plane: &[u8],
        stride: usize,
        x: usize,
        y: usize,
        n: usize,
        has_left: bool,
        has_top: bool,
        has_tr: bool,
        has_tl: bool,
    ) -> Self {
        let mut e = Edges {
            top: [0; 16],
            left: [0; 8],
            tl: 0,
            has_top,
            has_left,
            has_tl,
        };
        if has_top {
            let row = (y - 1) * stride;
            e.top[..n].copy_from_slice(&plane[row + x..row + x + n]);
            if has_tr {
                e.top[n..2 * n].copy_from_slice(&plane[row + x + n..row + x + 2 * n]);
            } else {
                let last = e.top[n - 1];
                e.top[n..2 * n].fill(last);
            }
        }
        if has_left {
            for i in 0..n {
                e.left[i] = plane[(y + i) * stride + x - 1];
            }
        }
        if has_tl {
            e.tl = plane[(y - 1) * stride + x - 1];
        }
        e
    }

    /// Reference sample filtering for Intra_8x8 (8.3.2.2.1).
    pub fn filter8(&self) -> Self {
        let mut f = Edges {
            top: self.top,
            left: self.left,
            tl: self.tl,
            has_top: self.has_top,
            has_left: self.has_left,
            has_tl: self.has_tl,
        };
        let p = |v: u8| v as u32;
        if self.has_top {
            let t = &self.top;
            f.top[0] = if self.has_tl {
                ((p(self.tl) + 2 * p(t[0]) + p(t[1]) + 2) >> 2) as u8
            } else {
                ((3 * p(t[0]) + p(t[1]) + 2) >> 2) as u8
            };
            for x in 1..15 {
                f.top[x] = ((p(t[x - 1]) + 2 * p(t[x]) + p(t[x + 1]) + 2) >> 2) as u8;
            }
            f.top[15] = ((p(t[14]) + 3 * p(t[15]) + 2) >> 2) as u8;
        }
        if self.has_tl {
            f.tl = match (self.has_top, self.has_left) {
                (true, true) => {
                    ((p(self.top[0]) + 2 * p(self.tl) + p(self.left[0]) + 2) >> 2) as u8
                }
                (true, false) => ((3 * p(self.tl) + p(self.top[0]) + 2) >> 2) as u8,
                (false, true) => ((3 * p(self.tl) + p(self.left[0]) + 2) >> 2) as u8,
                (false, false) => self.tl,
            };
        }
        if self.has_left {
            let l = &self.left;
            f.left[0] = if self.has_tl {
                ((p(self.tl) + 2 * p(l[0]) + p(l[1]) + 2) >> 2) as u8
            } else {
                ((3 * p(l[0]) + p(l[1]) + 2) >> 2) as u8
            };
            for y in 1..7 {
                f.left[y] = ((p(l[y - 1]) + 2 * p(l[y]) + p(l[y + 1]) + 2) >> 2) as u8;
            }
            f.left[7] = ((p(l[6]) + 3 * p(l[7]) + 2) >> 2) as u8;
        }
        f
    }
}

/// Predicts an `n`x`n` block (`n` = 4 or 8) with Intra4x4/Intra8x8 `mode` into `out` (row-major, stride `n`).
pub fn predict_nxn(n: usize, mode: u8, e: &Edges, out: &mut [u8]) {
    let t = |x: i32| e.top[x as usize] as i32;
    let l = |y: i32| e.left[y as usize] as i32;
    // p[x, -1] for x >= -1 and p[-1, y] for y >= -1.
    let pt = |x: i32| if x < 0 { e.tl as i32 } else { t(x) };
    let pl = |y: i32| if y < 0 { e.tl as i32 } else { l(y) };
    let ni = n as i32;
    for y in 0..ni {
        for x in 0..ni {
            let v = match mode {
                0 => t(x),
                1 => l(y),
                2 => {
                    let st: i32 = (0..ni).map(t).sum();
                    let sl: i32 = (0..ni).map(l).sum();
                    let sh = if n == 4 { 2 } else { 3 };
                    match (e.has_top, e.has_left) {
                        (true, true) => (st + sl + ni) >> (sh + 1),
                        (false, true) => (sl + (ni >> 1)) >> sh,
                        (true, false) => (st + (ni >> 1)) >> sh,
                        (false, false) => 128,
                    }
                }
                3 => {
                    if x == ni - 1 && y == ni - 1 {
                        (t(2 * ni - 2) + 3 * t(2 * ni - 1) + 2) >> 2
                    } else {
                        (t(x + y) + 2 * t(x + y + 1) + t(x + y + 2) + 2) >> 2
                    }
                }
                4 => {
                    if x > y {
                        (pt(x - y - 2) + 2 * pt(x - y - 1) + t(x - y) + 2) >> 2
                    } else if x < y {
                        (pl(y - x - 2) + 2 * pl(y - x - 1) + l(y - x) + 2) >> 2
                    } else {
                        (t(0) + 2 * e.tl as i32 + l(0) + 2) >> 2
                    }
                }
                5 => {
                    let z = 2 * x - y;
                    if z >= 0 && z % 2 == 0 {
                        (pt(x - (y >> 1) - 1) + t(x - (y >> 1)) + 1) >> 1
                    } else if z > 0 {
                        (pt(x - (y >> 1) - 2) + 2 * pt(x - (y >> 1) - 1) + t(x - (y >> 1)) + 2) >> 2
                    } else if z == -1 {
                        (l(0) + 2 * e.tl as i32 + t(0) + 2) >> 2
                    } else {
                        (pl(y - 2 * x - 1) + 2 * pl(y - 2 * x - 2) + pl(y - 2 * x - 3) + 2) >> 2
                    }
                }
                6 => {
                    let z = 2 * y - x;
                    if z >= 0 && z % 2 == 0 {
                        (pl(y - (x >> 1) - 1) + l(y - (x >> 1)) + 1) >> 1
                    } else if z > 0 {
                        (pl(y - (x >> 1) - 2) + 2 * pl(y - (x >> 1) - 1) + l(y - (x >> 1)) + 2) >> 2
                    } else if z == -1 {
                        (l(0) + 2 * e.tl as i32 + t(0) + 2) >> 2
                    } else {
                        (pt(x - 2 * y - 1) + 2 * pt(x - 2 * y - 2) + pt(x - 2 * y - 3) + 2) >> 2
                    }
                }
                7 => {
                    let i = x + (y >> 1);
                    if y % 2 == 0 {
                        (t(i) + t(i + 1) + 1) >> 1
                    } else {
                        (t(i) + 2 * t(i + 1) + t(i + 2) + 2) >> 2
                    }
                }
                _ => {
                    let z = x + 2 * y;
                    let i = y + (x >> 1);
                    if z > 2 * ni - 3 {
                        l(ni - 1)
                    } else if z == 2 * ni - 3 {
                        (l(ni - 2) + 3 * l(ni - 1) + 2) >> 2
                    } else if z % 2 == 0 {
                        (l(i) + l(i + 1) + 1) >> 1
                    } else {
                        (l(i) + 2 * l(i + 1) + l(i + 2) + 2) >> 2
                    }
                }
            };
            out[(y as usize) * n + x as usize] = v as u8;
        }
    }
}

/// Whether a 16x16 / chroma mode (V=0, H=1, DC=2, Plane=3) only reads available samples.
pub fn block_mode_ok(mode: u8, has_left: bool, has_top: bool, has_tl: bool) -> bool {
    match mode {
        0 => has_top,
        1 => has_left,
        2 => true,
        _ => has_left && has_top && has_tl,
    }
}

#[inline]
fn clip(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Intra_16x16 prediction into `out` (16x16, stride 16). Samples are read from `plane` around `(x, y)`.
#[allow(clippy::too_many_arguments)]
pub fn predict_16x16(
    plane: &[u8],
    stride: usize,
    x: usize,
    y: usize,
    mode: u8,
    has_left: bool,
    has_top: bool,
    has_tl: bool,
    out: &mut [u8; 256],
) {
    predict_block(
        plane,
        stride,
        x,
        y,
        16,
        16,
        mode,
        [has_left; 2],
        has_top,
        has_tl,
        out,
        false,
    );
}

/// Chroma intra prediction for a 4:2:0 8x8 block (`intra_chroma_pred_mode`: 0 DC, 1 H, 2 V, 3 Plane).
/// `has_left` gives the availability of the upper and lower four left samples, which can differ
/// in MBAFF frames with constrained intra prediction.
#[allow(clippy::too_many_arguments)]
pub fn predict_chroma(
    plane: &[u8],
    stride: usize,
    x: usize,
    y: usize,
    mode: u8,
    has_left: [bool; 2],
    has_top: bool,
    has_tl: bool,
    out: &mut [u8; 64],
) {
    // Map chroma modes onto the 16x16 numbering: V=0, H=1, DC=2, Plane=3.
    let m = match mode {
        0 => 2,
        1 => 1,
        2 => 0,
        _ => 3,
    };
    predict_block(
        plane, stride, x, y, 8, 8, m, has_left, has_top, has_tl, out, true,
    );
}

#[allow(clippy::too_many_arguments)]
fn predict_block(
    plane: &[u8],
    stride: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    mode: u8,
    left_avail: [bool; 2],
    has_top: bool,
    has_tl: bool,
    out: &mut [u8],
    chroma: bool,
) {
    let has_left = left_avail[0] && left_avail[1];
    let top = |i: usize| plane[(y - 1) * stride + x + i] as i32;
    let left = |i: usize| plane[(y + i) * stride + x - 1] as i32;
    match mode {
        0 => {
            for r in 0..h {
                for c in 0..w {
                    out[r * w + c] = top(c) as u8;
                }
            }
        }
        1 => {
            for r in 0..h {
                let v = left(r) as u8;
                out[r * w..r * w + w].fill(v);
            }
        }
        2 if !chroma => {
            let st: i32 = if has_top { (0..16).map(top).sum() } else { 0 };
            let sl: i32 = if has_left { (0..16).map(left).sum() } else { 0 };
            let v = match (has_top, has_left) {
                (true, true) => (st + sl + 16) >> 5,
                (false, true) => (sl + 8) >> 4,
                (true, false) => (st + 8) >> 4,
                (false, false) => 128,
            };
            out[..256].fill(v as u8);
        }
        2 => {
            // Chroma DC per 4x4 block (8.3.4.1 - 8.3.4.3).
            for by in 0..2 {
                let has_left = left_avail[by];
                for bx in 0..2 {
                    let st: i32 = if has_top {
                        (0..4).map(|i| top(bx * 4 + i)).sum()
                    } else {
                        0
                    };
                    let sl: i32 = if has_left {
                        (0..4).map(|i| left(by * 4 + i)).sum()
                    } else {
                        0
                    };
                    let both = (st + sl + 4) >> 3;
                    let t_only = (st + 2) >> 2;
                    let l_only = (sl + 2) >> 2;
                    let v = if (bx == 0 && by == 0) || (bx > 0 && by > 0) {
                        match (has_top, has_left) {
                            (true, true) => both,
                            (true, false) => t_only,
                            (false, true) => l_only,
                            _ => 128,
                        }
                    } else if bx > 0 {
                        if has_top {
                            t_only
                        } else if has_left {
                            l_only
                        } else {
                            128
                        }
                    } else if has_left {
                        l_only
                    } else if has_top {
                        t_only
                    } else {
                        128
                    };
                    for r in 0..4 {
                        for c in 0..4 {
                            out[(by * 4 + r) * w + bx * 4 + c] = v as u8;
                        }
                    }
                }
            }
        }
        _ => {
            let _ = has_tl;
            let pt = |i: i32| {
                if i < 0 {
                    plane[(y - 1) * stride + x - 1] as i32
                } else {
                    top(i as usize)
                }
            };
            let pl = |i: i32| {
                if i < 0 {
                    plane[(y - 1) * stride + x - 1] as i32
                } else {
                    left(i as usize)
                }
            };
            let (hw, hh) = ((w / 2) as i32, (h / 2) as i32);
            let mut hs = 0;
            for i in 0..hw {
                hs += (i + 1) * (pt(hw + i) - pt(hw - 2 - i));
            }
            let mut vs = 0;
            for i in 0..hh {
                vs += (i + 1) * (pl(hh + i) - pl(hh - 2 - i));
            }
            let a = 16 * (pl(h as i32 - 1) + pt(w as i32 - 1));
            let (b, c) = if chroma {
                ((34 * hs + 32) >> 6, (34 * vs + 32) >> 6)
            } else {
                ((5 * hs + 32) >> 6, (5 * vs + 32) >> 6)
            };
            for r in 0..h as i32 {
                for cc in 0..w as i32 {
                    out[(r as usize) * w + cc as usize] =
                        clip((a + b * (cc - (hw - 1)) + c * (r - (hh - 1)) + 16) >> 5);
                }
            }
        }
    }
}
