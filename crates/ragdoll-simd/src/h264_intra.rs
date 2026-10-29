//! H.264 intra prediction (ITU-T H.264 8.3, `SIMD-0076`): Intra_4x4, Intra_8x8 with its
//! reference sample filtering (8.3.2.2.1), Intra_16x16 and 4:2:0 chroma, written straight into
//! the picture plane.
//!
//! Neighbours are flagged available with [`INTRA_LEFT`] and friends. The 4x4 and 8x8 predictors
//! read unavailable neighbours as 0 (an unavailable top-right as `p[N-1, -1]`), so every mode is
//! defined for every input; the 16x16 and chroma predictors expect a mode whose samples are
//! available ([`h264_pred16x16`]).
//!
//! The directional 4x4 and 8x8 modes are byte shuffles of two filtered versions of the edge line
//! `[pad, p[-1, N-1] .. p[-1, 0], p[-1, -1], p[0, -1] .. p[2N-1, -1], pad]`: its three-tap lowpass
//! `(a + 2b + c + 2) >> 2` and its two-tap average `(a + b + 1) >> 1`. The pads repeat the end
//! samples, which yields the end cases such as `(p[6, -1] + 3 p[7, -1] + 2) >> 2`. The Intra_8x8
//! reference filter is the same lowpass over the same line.

/// The left neighbours are available (for chroma: the upper four).
pub const INTRA_LEFT: u8 = 1;
/// The top neighbours are available.
pub const INTRA_TOP: u8 = 2;
/// The top-right neighbours are available (4x4 and 8x8 only).
pub const INTRA_TOP_RIGHT: u8 = 4;
/// The top-left neighbour is available.
pub const INTRA_TOP_LEFT: u8 = 8;
/// Chroma only: the lower four left neighbours are available.
pub const INTRA_LEFT_LOWER: u8 = 16;

#[inline(always)]
fn lp(a: u8, b: u8, c: u8) -> u8 {
    ((a as u16 + 2 * b as u16 + c as u16 + 2) >> 2) as u8
}

#[inline(always)]
fn clip(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Panics unless the `n`x`n` block at `off` and the neighbours flagged in `avail` lie in a plane
/// of `len` samples. Top-right neighbours then do too: they end before the block's last row.
#[inline(always)]
fn check_block(len: usize, off: usize, stride: usize, n: usize, avail: u8) {
    assert!(stride >= n && off + (n - 1) * stride + n <= len);
    if avail & INTRA_TOP_LEFT != 0 {
        assert!(off > stride);
    } else if avail & INTRA_TOP != 0 {
        assert!(off >= stride);
    }
    if avail & INTRA_LEFT != 0 {
        assert!(off >= 1);
    }
}

// ----- Intra_4x4 and Intra_8x8 ----------------------------------------------------------------

/// Where sample `(x, y)` of an `n`x`n` block predicted with directional `mode` (3..=8) comes from
/// on the edge line (layout in the module docs: `p[-1, y]` at `n - y`, `p[-1, -1]` at `n + 1`,
/// `p[x, -1]` at `n + 2 + x`): `i` for lowpass sample `i`, `32 + i` for average sample `i`.
const fn dir_src(n: usize, mode: usize, x: usize, y: usize) -> usize {
    let (n, x, y) = (n as isize, x as isize, y as isize);
    let (lowpass, avg) = (0isize, 32isize);
    let s = match mode {
        3 => lowpass + n + 3 + x + y,
        4 => lowpass + n + 1 + x - y,
        5 => {
            let (z, k) = (2 * x - y, x - (y >> 1));
            if z >= 0 && z % 2 == 0 {
                avg + n + 1 + k
            } else if z > 0 {
                lowpass + n + 1 + k
            } else if z == -1 {
                lowpass + n + 1
            } else {
                lowpass + n + 2 + z
            }
        }
        6 => {
            let (z, k) = (2 * y - x, y - (x >> 1));
            if z >= 0 && z % 2 == 0 {
                avg + n - k
            } else if z > 0 {
                lowpass + n + 1 - k
            } else if z == -1 {
                lowpass + n + 1
            } else {
                lowpass + n - z
            }
        }
        7 => {
            let i = x + (y >> 1);
            if y % 2 == 0 {
                avg + n + 2 + i
            } else {
                lowpass + n + 3 + i
            }
        }
        _ => {
            let (z, i) = (x + 2 * y, y + (x >> 1));
            if z > 2 * n - 3 {
                // p[-1, n-1]: the average of it and the pad repeating it.
                avg
            } else if z == 2 * n - 3 {
                lowpass + 1
            } else if z % 2 == 0 {
                avg + n - i - 1
            } else {
                lowpass + n - i - 1
            }
        }
    };
    s as usize
}

/// Byte-shuffle masks (`0x80` selects 0) building a 4x4 block, rows in order, from the lowpass
/// line (`[mode][0]`) and the average line (`[mode][1]`). Modes 0..=2 are not shuffles.
const MASKS4: [[[u8; 16]; 2]; 9] = {
    let mut m = [[[0x80u8; 16]; 2]; 9];
    let mut mode = 3;
    while mode < 9 {
        let mut p = 0;
        while p < 16 {
            let s = dir_src(4, mode, p % 4, p / 4);
            if s < 32 {
                m[mode][0][p] = s as u8;
            } else {
                m[mode][1][p] = (s - 32) as u8;
            }
            p += 1;
        }
        mode += 1;
    }
    m
};

/// Start of the 16-sample windows of the 8x8 lowpass and average lines (27 samples each) that a
/// directional mode reads.
const fn win8(mode: usize) -> (usize, usize) {
    (
        if mode == 3 || mode == 7 { 10 } else { 1 },
        if mode == 5 || mode == 7 { 5 } else { 0 },
    )
}

/// [`MASKS4`] for 8x8 blocks, two rows per register (`[mode][pair]`), indexing the [`win8`]
/// windows.
const MASKS8: [[[[u8; 16]; 2]; 4]; 9] = {
    let mut m = [[[[0x80u8; 16]; 2]; 4]; 9];
    let mut mode = 3;
    while mode < 9 {
        let (wl, wa) = win8(mode);
        let mut p = 0;
        while p < 64 {
            let s = dir_src(8, mode, p % 8, p / 8);
            let (src, i) = if s < 32 {
                (0, s - wl)
            } else {
                (1, s - 32 - wa)
            };
            assert!(i < 16, "8x8 shuffle outside its window");
            m[mode][p / 16][src][p % 16] = i as u8;
            p += 1;
        }
        mode += 1;
    }
    m
};

/// Neighbouring samples of an NxN block: `top` holds `p[0..2N, -1]` (top-right substituted),
/// `left` holds `p[-1, 0..N]`, `tl` is `p[-1, -1]`; unavailable ones are 0.
struct Edges {
    top: [u8; 16],
    left: [u8; 8],
    tl: u8,
    has_top: bool,
    has_left: bool,
    has_tl: bool,
}

impl Edges {
    fn gather(plane: &[u8], off: usize, stride: usize, n: usize, avail: u8) -> Self {
        let mut e = Edges {
            top: [0; 16],
            left: [0; 8],
            tl: 0,
            has_top: avail & INTRA_TOP != 0,
            has_left: avail & INTRA_LEFT != 0,
            has_tl: avail & INTRA_TOP_LEFT != 0,
        };
        if e.has_top {
            let row = off - stride;
            e.top[..n].copy_from_slice(&plane[row..row + n]);
            if avail & INTRA_TOP_RIGHT != 0 {
                e.top[n..2 * n].copy_from_slice(&plane[row + n..row + 2 * n]);
            } else {
                let last = e.top[n - 1];
                e.top[n..2 * n].fill(last);
            }
        }
        if e.has_left {
            for i in 0..n {
                e.left[i] = plane[off + i * stride - 1];
            }
        }
        if e.has_tl {
            e.tl = plane[off - stride - 1];
        }
        e
    }

    /// Reference sample filtering for Intra_8x8 (8.3.2.2.1).
    fn filter8(&self) -> Self {
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
                lp(self.tl, t[0], t[1])
            } else {
                ((3 * p(t[0]) + p(t[1]) + 2) >> 2) as u8
            };
            for x in 1..15 {
                f.top[x] = lp(t[x - 1], t[x], t[x + 1]);
            }
            f.top[15] = ((p(t[14]) + 3 * p(t[15]) + 2) >> 2) as u8;
        }
        if self.has_tl {
            f.tl = match (self.has_top, self.has_left) {
                (true, true) => lp(self.top[0], self.tl, self.left[0]),
                (true, false) => ((3 * p(self.tl) + p(self.top[0]) + 2) >> 2) as u8,
                (false, true) => ((3 * p(self.tl) + p(self.left[0]) + 2) >> 2) as u8,
                (false, false) => self.tl,
            };
        }
        if self.has_left {
            let l = &self.left;
            f.left[0] = if self.has_tl {
                lp(self.tl, l[0], l[1])
            } else {
                ((3 * p(l[0]) + p(l[1]) + 2) >> 2) as u8
            };
            for y in 1..7 {
                f.left[y] = lp(l[y - 1], l[y], l[y + 1]);
            }
            f.left[7] = ((p(l[6]) + 3 * p(l[7]) + 2) >> 2) as u8;
        }
        f
    }
}

/// Predicts an `n`x`n` block (`n` = 4 or 8) with Intra4x4/Intra8x8 `mode` into the plane, sample
/// by sample as 8.3.1.2 and 8.3.2.2 state it.
fn predict_nxn(plane: &mut [u8], off: usize, stride: usize, n: usize, mode: u8, e: &Edges) {
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
            plane[off + y as usize * stride + x as usize] = v as u8;
        }
    }
}

/// Intra_4x4 prediction (8.3.1.2) with `mode` (0..=8; larger values act as 8) of the 4x4 block
/// at `off` in `plane` (line stride `stride`), from the neighbours flagged in `avail`.
///
/// # Panics
/// If the block or a neighbour flagged available lies outside `plane`.
#[inline]
pub fn h264_pred4x4(plane: &mut [u8], off: usize, stride: usize, mode: u8, avail: u8) {
    check_block(plane.len(), off, stride, 4, avail);
    if mode < 3 {
        pred4x4_fill(plane, off, stride, mode, avail);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        unsafe { pred4x4_avx2(plane, off, stride, mode, avail) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above; neon() is true only when the CPU supports NEON.
        unsafe { pred4x4_neon(plane, off, stride, mode, avail) };
        return;
    }
    h264_pred4x4_scalar(plane, off, stride, mode, avail);
}

/// Scalar reference for [`h264_pred4x4`].
pub fn h264_pred4x4_scalar(plane: &mut [u8], off: usize, stride: usize, mode: u8, avail: u8) {
    check_block(plane.len(), off, stride, 4, avail);
    let e = Edges::gather(plane, off, stride, 4, avail);
    predict_nxn(plane, off, stride, 4, mode, &e);
}

/// Vertical, horizontal and DC Intra_4x4 prediction, a row at a time.
#[inline(always)]
fn pred4x4_fill(plane: &mut [u8], off: usize, stride: usize, mode: u8, avail: u8) {
    let has_top = avail & INTRA_TOP != 0;
    let has_left = avail & INTRA_LEFT != 0;
    let top = if has_top {
        u32::from_le_bytes(plane[off - stride..off - stride + 4].try_into().unwrap())
    } else {
        0
    };
    let left = |plane: &[u8], r: usize| {
        if has_left {
            plane[off + r * stride - 1] as u32
        } else {
            0
        }
    };
    let row = |plane: &mut [u8], r: usize, v: u32| {
        plane[off + r * stride..off + r * stride + 4].copy_from_slice(&v.to_le_bytes());
    };
    match mode {
        0 => (0..4).for_each(|r| row(plane, r, top)),
        1 => (0..4).for_each(|r| {
            let v = left(plane, r) * 0x0101_0101;
            row(plane, r, v);
        }),
        _ => {
            let st = top.to_le_bytes().iter().map(|&v| v as u32).sum::<u32>();
            let sl = (0..4).map(|r| left(plane, r)).sum::<u32>();
            let dc = match (has_top, has_left) {
                (true, true) => (st + sl + 4) >> 3,
                (false, true) => (sl + 2) >> 2,
                (true, false) => (st + 2) >> 2,
                (false, false) => 128,
            };
            (0..4).for_each(|r| row(plane, r, dc * 0x0101_0101));
        }
    }
}

/// Intra_8x8 prediction (8.3.2.2) with `mode` (0..=8; larger values act as 8) of the 8x8 block
/// at `off` in `plane`, from the neighbours flagged in `avail` after reference sample filtering
/// (8.3.2.2.1).
///
/// # Panics
/// If the block or a neighbour flagged available lies outside `plane`.
#[inline]
pub fn h264_pred8x8l(plane: &mut [u8], off: usize, stride: usize, mode: u8, avail: u8) {
    check_block(plane.len(), off, stride, 8, avail);
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: bounds checked above; avx2() is true only when the CPU supports AVX2.
        unsafe { pred8x8l_avx2(plane, off, stride, mode, avail) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: bounds checked above; neon() is true only when the CPU supports NEON.
        unsafe { pred8x8l_neon(plane, off, stride, mode, avail) };
        return;
    }
    h264_pred8x8l_scalar(plane, off, stride, mode, avail);
}

/// Scalar reference for [`h264_pred8x8l`].
pub fn h264_pred8x8l_scalar(plane: &mut [u8], off: usize, stride: usize, mode: u8, avail: u8) {
    check_block(plane.len(), off, stride, 8, avail);
    let e = Edges::gather(plane, off, stride, 8, avail).filter8();
    predict_nxn(plane, off, stride, 8, mode, &e);
}

/// The raw edge samples of an 8x8 block, arranged so that one lowpass over the edge line gives
/// the filtered edge (8.3.2.2.1) wherever a neighbour is available: `(left, tl_slot, tl)` where
/// `left` holds `p[-1, 7] .. p[-1, 0]` from its low byte up, and `tl_slot` stands at
/// `p[-1, -1]`. Unavailable left samples repeat `p[-1, -1]` (giving its filter
/// `(3 p[-1,-1] + p[0,-1] + 2) >> 2`); an unavailable `p[-1, -1]` repeats `p[0, -1]` (or
/// `p[-1, 0]`), which filters `p[0, -1]` as `(3 p[0,-1] + p[1,-1] + 2) >> 2`.
#[inline(always)]
fn edge8_left(plane: &[u8], off: usize, stride: usize, avail: u8) -> (u64, u8, u8) {
    let tl = if avail & INTRA_TOP_LEFT != 0 {
        plane[off - stride - 1]
    } else {
        0
    };
    let left = if avail & INTRA_LEFT != 0 {
        (0..8).fold(0u64, |v, i| {
            v | (plane[off + i * stride - 1] as u64) << (56 - 8 * i)
        })
    } else {
        tl as u64 * 0x0101_0101_0101_0101
    };
    let tl_slot = if avail & INTRA_TOP_LEFT != 0 {
        tl
    } else if avail & INTRA_TOP != 0 {
        plane[off - stride]
    } else {
        (left >> 56) as u8
    };
    (left, tl_slot, tl)
}

/// The filtered `p[-1, 0]` when `p[-1, -1]` is unavailable but `p[0, -1]` is, which
/// [`edge8_left`]'s substitution cannot give at the same time as the filtered `p[0, -1]`.
#[inline(always)]
fn left0_without_tl(left: u64) -> u8 {
    let (l0, l1) = ((left >> 56) as u32, (left >> 48) as u32 & 0xff);
    ((3 * l0 + l1 + 2) >> 2) as u8
}

/// Masks clearing the filtered edge line (two 16-byte halves) where neighbours are unavailable,
/// and its pads, indexed by `avail & 11` (left, top and top-left).
const EDGE8_KEEP: [[u8; 32]; 12] = {
    let mut m = [[0u8; 32]; 12];
    let mut a = 0;
    while a < 12 {
        let mut i = 1;
        while i < 26 {
            let keep = if i < 9 {
                a & INTRA_LEFT as usize != 0
            } else if i == 9 {
                a & INTRA_TOP_LEFT as usize != 0
            } else {
                a & INTRA_TOP as usize != 0
            };
            m[a][i] = if keep { 0xff } else { 0 };
            i += 1;
        }
        a += 1;
    }
    m
};

/// Shuffle repeating byte 1 into byte 0 (the low pad of the 8x8 edge line).
const PAD_LO: [u8; 16] = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
/// Shuffle laying `p[6..16, -1]` and the high pad into the upper half of the 8x8 edge line.
const TOP_HI: [u8; 16] = [
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 0x80, 0x80, 0x80, 0x80, 0x80,
];
/// Shuffle repeating byte 9 into byte 10 (the high pad of the filtered 8x8 edge line).
const PAD_HI: [u8; 16] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 0x80, 0x80, 0x80, 0x80, 0x80,
];
/// Shuffle replacing the top-right samples `p[N..2N, -1]` with `p[N-1, -1]`: N = 8, then N = 4.
const NO_TR8: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7];
/// Shuffles laying `p[0..8, -1]` into the 4x4 edge line at byte 6, with or without top-right
/// samples, and the high pad.
const TOP4: [[u8; 16]; 2] = [
    [
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0, 1, 2, 3, 3, 3, 3, 3, 3, 0x80,
    ],
    [
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0, 1, 2, 3, 4, 5, 6, 7, 7, 0x80,
    ],
];
/// Shuffles gathering `p[-1, 0..8]` (low half) and `p[0..8, -1]` (high half) from the two halves
/// of the 8x8 edge line, for the DC sums.
const DC8: [[u8; 16]; 2] = [
    [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 0x80, 0x80],
    [
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0, 1,
    ],
];
/// Shuffles broadcasting `p[-1, y]` and `p[-1, y + 1]` over two 8-sample rows, `y = 2k`.
const HOR8: [[u8; 16]; 4] = [
    [8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7],
    [6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5],
    [4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3],
    [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
];

/// DC of an NxN block from the sums of its top and left samples (8.3.1.2.3, 8.3.2.2.4).
#[inline(always)]
fn dc_nxn(sh: u32, st: u32, sl: u32, avail: u8) -> u8 {
    let n = 1 << sh;
    (match (avail & INTRA_TOP != 0, avail & INTRA_LEFT != 0) {
        (true, true) => (st + sl + n) >> (sh + 1),
        (false, true) => (sl + (n >> 1)) >> sh,
        (true, false) => (st + (n >> 1)) >> sh,
        (false, false) => 128,
    }) as u8
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    #[inline(always)]
    pub unsafe fn load(m: &[u8; 16]) -> __m128i {
        unsafe { _mm_loadu_si128(m.as_ptr() as *const __m128i) }
    }

    /// `(a + 2b + c + 2) >> 2` per byte: `((a + c) >> 1 + b + 1) >> 1` is the same value.
    #[inline(always)]
    pub unsafe fn lowpass(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
        unsafe {
            let odd = _mm_and_si128(_mm_xor_si128(a, c), _mm_set1_epi8(1));
            _mm_avg_epu8(_mm_sub_epi8(_mm_avg_epu8(a, c), odd), b)
        }
    }

    /// The lowpass and average lines of the 16-byte edge line `f`.
    #[inline(always)]
    pub unsafe fn lines(f: __m128i) -> (__m128i, __m128i) {
        unsafe {
            let next = _mm_srli_si128::<1>(f);
            (
                lowpass(_mm_slli_si128::<1>(f), f, next),
                _mm_avg_epu8(f, next),
            )
        }
    }

    /// [`lines`] for a 32-byte edge line in two halves: `(lowpass, average)`, each `[lo, hi]`.
    #[inline(always)]
    pub unsafe fn lines2(lo: __m128i, hi: __m128i) -> ([__m128i; 2], [__m128i; 2]) {
        unsafe {
            let (lo_next, hi_next) = (_mm_alignr_epi8::<1>(hi, lo), _mm_srli_si128::<1>(hi));
            let (lo_prev, hi_prev) = (_mm_slli_si128::<1>(lo), _mm_alignr_epi8::<15>(hi, lo));
            (
                [lowpass(lo_prev, lo, lo_next), lowpass(hi_prev, hi, hi_next)],
                [_mm_avg_epu8(lo, lo_next), _mm_avg_epu8(hi, hi_next)],
            )
        }
    }

    /// Stores the low and high 8 bytes of `v` at `p` and `p + stride`.
    #[inline(always)]
    pub unsafe fn store_rows8(p: *mut u8, stride: usize, v: __m128i) {
        unsafe {
            _mm_storel_epi64(p as *mut __m128i, v);
            _mm_storel_epi64(p.add(stride) as *mut __m128i, _mm_unpackhi_epi64(v, v));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pred4x4_avx2(plane: &mut [u8], off: usize, stride: usize, mode: u8, avail: u8) {
    use std::arch::x86_64::*;
    use x86::*;
    let mode = mode.min(8) as usize;
    let p = plane.as_mut_ptr();
    unsafe {
        // Edge line [p[-1,3], p[-1,3..=0], p[-1,-1], p[0..8,-1], p[7,-1]]; absent samples are 0.
        let mut low = 0u64;
        if avail & INTRA_LEFT != 0 {
            let l = |r: usize| *p.add(off + r * stride - 1) as u64;
            let l3 = l(3);
            low = l3 | l3 << 8 | l(2) << 16 | l(1) << 24 | l(0) << 32;
        }
        if avail & INTRA_TOP_LEFT != 0 {
            low |= (*p.add(off - stride - 1) as u64) << 40;
        }
        let mut f = _mm_cvtsi64_si128(low as i64);
        if avail & INTRA_TOP != 0 {
            let t = _mm_loadl_epi64(p.add(off - stride) as *const __m128i);
            let m = &TOP4[(avail & INTRA_TOP_RIGHT != 0) as usize];
            f = _mm_or_si128(f, _mm_shuffle_epi8(t, load(m)));
        }
        let (h, a) = lines(f);
        let m = &MASKS4[mode];
        let v = _mm_or_si128(
            _mm_shuffle_epi8(h, load(&m[0])),
            _mm_shuffle_epi8(a, load(&m[1])),
        );
        let q = p.add(off);
        (q as *mut i32).write_unaligned(_mm_cvtsi128_si32(v));
        (q.add(stride) as *mut i32).write_unaligned(_mm_extract_epi32::<1>(v));
        (q.add(2 * stride) as *mut i32).write_unaligned(_mm_extract_epi32::<2>(v));
        (q.add(3 * stride) as *mut i32).write_unaligned(_mm_extract_epi32::<3>(v));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pred8x8l_avx2(plane: &mut [u8], off: usize, stride: usize, mode: u8, avail: u8) {
    use std::arch::x86_64::*;
    use x86::*;
    let mode = mode.min(8) as usize;
    let (left, tl_slot, tl) = edge8_left(plane, off, stride, avail);
    let p = plane.as_mut_ptr();
    unsafe {
        let top = if avail & INTRA_TOP != 0 {
            let t = _mm_loadu_si128(p.add(off - stride) as *const __m128i);
            if avail & INTRA_TOP_RIGHT != 0 {
                t
            } else {
                _mm_shuffle_epi8(t, load(&NO_TR8))
            }
        } else {
            _mm_set1_epi8(tl as i8)
        };
        // Raw edge line: [pad, p[-1,7..=0], tl_slot, p[0..6,-1]] and [p[6..16,-1], pad].
        let lo = _mm_slli_si128::<1>(_mm_cvtsi64_si128(left as i64));
        let lo = _mm_insert_epi8::<0>(lo, left as u8 as i32);
        let lo = _mm_insert_epi8::<9>(lo, tl_slot as i32);
        let lo = _mm_or_si128(lo, _mm_slli_si128::<10>(top));
        let hi = _mm_shuffle_epi8(top, load(&TOP_HI));
        // Filtered edge line, with unavailable samples cleared and the pads restored.
        let (f, _) = lines2(lo, hi);
        let keep = &EDGE8_KEEP[(avail & 11) as usize];
        let mut flo = _mm_and_si128(f[0], _mm_loadu_si128(keep.as_ptr() as *const __m128i));
        let fhi = _mm_and_si128(
            f[1],
            _mm_loadu_si128(keep.as_ptr().add(16) as *const __m128i),
        );
        if avail & (INTRA_LEFT | INTRA_TOP | INTRA_TOP_LEFT) == INTRA_LEFT | INTRA_TOP {
            flo = _mm_insert_epi8::<8>(flo, left0_without_tl(left) as i32);
        }
        let flo = _mm_shuffle_epi8(flo, load(&PAD_LO));
        let fhi = _mm_shuffle_epi8(fhi, load(&PAD_HI));
        let q = p.add(off);
        match mode {
            0 => {
                let v = _mm_alignr_epi8::<10>(fhi, flo);
                for r in 0..8 {
                    _mm_storel_epi64(q.add(r * stride) as *mut __m128i, v);
                }
            }
            1 => {
                for (k, m) in HOR8.iter().enumerate() {
                    store_rows8(
                        q.add(2 * k * stride),
                        stride,
                        _mm_shuffle_epi8(flo, load(m)),
                    );
                }
            }
            2 => {
                let e = _mm_or_si128(
                    _mm_shuffle_epi8(flo, load(&DC8[0])),
                    _mm_shuffle_epi8(fhi, load(&DC8[1])),
                );
                let s = _mm_sad_epu8(e, _mm_setzero_si128());
                let (sl, st) = (_mm_cvtsi128_si32(s), _mm_extract_epi32::<2>(s));
                let v = _mm_set1_epi8(dc_nxn(3, st as u32, sl as u32, avail) as i8);
                for r in 0..8 {
                    _mm_storel_epi64(q.add(r * stride) as *mut __m128i, v);
                }
            }
            _ => {
                let (h, a) = lines2(flo, fhi);
                let (wl, wa) = win8(mode);
                let hw = if wl == 10 {
                    _mm_alignr_epi8::<10>(h[1], h[0])
                } else {
                    _mm_alignr_epi8::<1>(h[1], h[0])
                };
                let aw = if wa == 5 {
                    _mm_alignr_epi8::<5>(a[1], a[0])
                } else {
                    a[0]
                };
                for (k, m) in MASKS8[mode].iter().enumerate() {
                    let v = _mm_or_si128(
                        _mm_shuffle_epi8(hw, load(&m[0])),
                        _mm_shuffle_epi8(aw, load(&m[1])),
                    );
                    store_rows8(q.add(2 * k * stride), stride, v);
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use std::arch::aarch64::*;

    #[inline(always)]
    pub unsafe fn load(m: &[u8; 16]) -> uint8x16_t {
        unsafe { vld1q_u8(m.as_ptr()) }
    }

    /// `(a + 2b + c + 2) >> 2` per byte: `((a + c) >> 1 + b + 1) >> 1` is the same value.
    #[inline(always)]
    pub unsafe fn lowpass(a: uint8x16_t, b: uint8x16_t, c: uint8x16_t) -> uint8x16_t {
        unsafe { vrhaddq_u8(vhaddq_u8(a, c), b) }
    }

    /// The lowpass and average lines of the 16-byte edge line `f`.
    #[inline(always)]
    pub unsafe fn lines(f: uint8x16_t) -> (uint8x16_t, uint8x16_t) {
        unsafe {
            let z = vdupq_n_u8(0);
            let next = vextq_u8::<1>(f, z);
            (lowpass(vextq_u8::<15>(z, f), f, next), vrhaddq_u8(f, next))
        }
    }

    /// [`lines`] for a 32-byte edge line in two halves: `(lowpass, average)`, each `[lo, hi]`.
    #[inline(always)]
    pub unsafe fn lines2(lo: uint8x16_t, hi: uint8x16_t) -> ([uint8x16_t; 2], [uint8x16_t; 2]) {
        unsafe {
            let z = vdupq_n_u8(0);
            let (lo_next, hi_next) = (vextq_u8::<1>(lo, hi), vextq_u8::<1>(hi, z));
            let (lo_prev, hi_prev) = (vextq_u8::<15>(z, lo), vextq_u8::<15>(lo, hi));
            (
                [lowpass(lo_prev, lo, lo_next), lowpass(hi_prev, hi, hi_next)],
                [vrhaddq_u8(lo, lo_next), vrhaddq_u8(hi, hi_next)],
            )
        }
    }

    /// Stores the low and high 8 bytes of `v` at `p` and `p + stride`.
    #[inline(always)]
    pub unsafe fn store_rows8(p: *mut u8, stride: usize, v: uint8x16_t) {
        unsafe {
            vst1_u8(p, vget_low_u8(v));
            vst1_u8(p.add(stride), vget_high_u8(v));
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn pred4x4_neon(plane: &mut [u8], off: usize, stride: usize, mode: u8, avail: u8) {
    use arm::*;
    use std::arch::aarch64::*;
    let mode = mode.min(8) as usize;
    let p = plane.as_mut_ptr();
    unsafe {
        // Edge line [p[-1,3], p[-1,3..=0], p[-1,-1], p[0..8,-1], p[7,-1]]; absent samples are 0.
        let mut low = 0u64;
        if avail & INTRA_LEFT != 0 {
            let l = |r: usize| *p.add(off + r * stride - 1) as u64;
            let l3 = l(3);
            low = l3 | l3 << 8 | l(2) << 16 | l(1) << 24 | l(0) << 32;
        }
        if avail & INTRA_TOP_LEFT != 0 {
            low |= (*p.add(off - stride - 1) as u64) << 40;
        }
        let mut f = vreinterpretq_u8_u64(vsetq_lane_u64::<0>(low, vdupq_n_u64(0)));
        if avail & INTRA_TOP != 0 {
            let t = vcombine_u8(vld1_u8(p.add(off - stride)), vdup_n_u8(0));
            let m = &TOP4[(avail & INTRA_TOP_RIGHT != 0) as usize];
            f = vorrq_u8(f, vqtbl1q_u8(t, load(m)));
        }
        let (h, a) = lines(f);
        let m = &MASKS4[mode];
        let v = vreinterpretq_u32_u8(vorrq_u8(
            vqtbl1q_u8(h, load(&m[0])),
            vqtbl1q_u8(a, load(&m[1])),
        ));
        let q = p.add(off);
        (q as *mut u32).write_unaligned(vgetq_lane_u32::<0>(v));
        (q.add(stride) as *mut u32).write_unaligned(vgetq_lane_u32::<1>(v));
        (q.add(2 * stride) as *mut u32).write_unaligned(vgetq_lane_u32::<2>(v));
        (q.add(3 * stride) as *mut u32).write_unaligned(vgetq_lane_u32::<3>(v));
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn pred8x8l_neon(plane: &mut [u8], off: usize, stride: usize, mode: u8, avail: u8) {
    use arm::*;
    use std::arch::aarch64::*;
    let mode = mode.min(8) as usize;
    let (left, tl_slot, tl) = edge8_left(plane, off, stride, avail);
    let p = plane.as_mut_ptr();
    unsafe {
        let top = if avail & INTRA_TOP != 0 {
            let t = vld1q_u8(p.add(off - stride));
            if avail & INTRA_TOP_RIGHT != 0 {
                t
            } else {
                vqtbl1q_u8(t, load(&NO_TR8))
            }
        } else {
            vdupq_n_u8(tl)
        };
        // Raw edge line: [pad, p[-1,7..=0], tl_slot, p[0..6,-1]] and [p[6..16,-1], pad].
        let z = vdupq_n_u8(0);
        let l = vreinterpretq_u8_u64(vsetq_lane_u64::<0>(left, vdupq_n_u64(0)));
        let lo = vextq_u8::<15>(z, l);
        let lo = vsetq_lane_u8::<0>(left as u8, lo);
        let lo = vsetq_lane_u8::<9>(tl_slot, lo);
        let lo = vorrq_u8(lo, vextq_u8::<6>(z, top));
        let hi = vqtbl1q_u8(top, load(&TOP_HI));
        // Filtered edge line, with unavailable samples cleared and the pads restored.
        let (f, _) = lines2(lo, hi);
        let keep = &EDGE8_KEEP[(avail & 11) as usize];
        let mut flo = vandq_u8(f[0], vld1q_u8(keep.as_ptr()));
        let fhi = vandq_u8(f[1], vld1q_u8(keep.as_ptr().add(16)));
        if avail & (INTRA_LEFT | INTRA_TOP | INTRA_TOP_LEFT) == INTRA_LEFT | INTRA_TOP {
            flo = vsetq_lane_u8::<8>(left0_without_tl(left), flo);
        }
        let flo = vqtbl1q_u8(flo, load(&PAD_LO));
        let fhi = vqtbl1q_u8(fhi, load(&PAD_HI));
        let q = p.add(off);
        match mode {
            0 => {
                let v = vget_low_u8(vextq_u8::<10>(flo, fhi));
                for r in 0..8 {
                    vst1_u8(q.add(r * stride), v);
                }
            }
            1 => {
                for (k, m) in HOR8.iter().enumerate() {
                    store_rows8(q.add(2 * k * stride), stride, vqtbl1q_u8(flo, load(m)));
                }
            }
            2 => {
                let e = vorrq_u8(
                    vqtbl1q_u8(flo, load(&DC8[0])),
                    vqtbl1q_u8(fhi, load(&DC8[1])),
                );
                let sl = vaddlv_u8(vget_low_u8(e)) as u32;
                let st = vaddlv_u8(vget_high_u8(e)) as u32;
                let v = vdup_n_u8(dc_nxn(3, st, sl, avail));
                for r in 0..8 {
                    vst1_u8(q.add(r * stride), v);
                }
            }
            _ => {
                let (h, a) = lines2(flo, fhi);
                let (wl, wa) = win8(mode);
                let hw = if wl == 10 {
                    vextq_u8::<10>(h[0], h[1])
                } else {
                    vextq_u8::<1>(h[0], h[1])
                };
                let aw = if wa == 5 {
                    vextq_u8::<5>(a[0], a[1])
                } else {
                    a[0]
                };
                for (k, m) in MASKS8[mode].iter().enumerate() {
                    let v = vorrq_u8(vqtbl1q_u8(hw, load(&m[0])), vqtbl1q_u8(aw, load(&m[1])));
                    store_rows8(q.add(2 * k * stride), stride, v);
                }
            }
        }
    }
}

// ----- Intra_16x16 and chroma -----------------------------------------------------------------

/// Intra_16x16 prediction (8.3.3) with `mode` (0 vertical, 1 horizontal, 2 DC, 3 plane) of the
/// 16x16 block at `off` in `plane`. `avail` flags the left, top and top-left neighbours; `mode`
/// must only use available ones (DC adapts to them).
///
/// # Panics
/// If the block or the neighbours `mode` reads lie outside `plane`.
#[inline]
pub fn h264_pred16x16(plane: &mut [u8], off: usize, stride: usize, mode: u8, avail: u8) {
    if mode >= 3 {
        let (a, b, c) = plane_params::<16>(plane, off, stride);
        #[cfg(target_arch = "x86_64")]
        if ragdoll_core::dispatch::avx2() {
            // Safety: plane_params checked the block's bounds; avx2() is true only when the
            // CPU supports AVX2.
            unsafe { plane_avx2::<16>(plane, off, stride, a, b, c) };
            return;
        }
        #[cfg(target_arch = "aarch64")]
        if ragdoll_core::dispatch::neon() {
            // Safety: plane_params checked the block's bounds; neon() is true only when the
            // CPU supports NEON.
            unsafe { plane_neon::<16>(plane, off, stride, a, b, c) };
            return;
        }
        plane_fill::<16>(plane, off, stride, a, b, c);
        return;
    }
    fill_block::<16>(plane, off, stride, mode, avail);
}

/// Scalar reference for [`h264_pred16x16`].
pub fn h264_pred16x16_scalar(plane: &mut [u8], off: usize, stride: usize, mode: u8, avail: u8) {
    predict_block(plane, off, stride, 16, mode, avail, false);
}

/// 4:2:0 chroma intra prediction (8.3.4) of the 8x8 block at `off` in `plane`, with `mode` in
/// the Intra_16x16 numbering (0 vertical, 1 horizontal, 2 DC, 3 plane). `avail` flags the upper
/// ([`INTRA_LEFT`]) and lower ([`INTRA_LEFT_LOWER`]) left neighbours, the top and the top-left
/// one; `mode` must only use available ones (DC adapts to them).
///
/// # Panics
/// If the block or the neighbours `mode` reads lie outside `plane`.
#[inline]
pub fn h264_pred_chroma(plane: &mut [u8], off: usize, stride: usize, mode: u8, avail: u8) {
    match mode {
        2 => chroma_dc(plane, off, stride, avail),
        3.. => {
            let (a, b, c) = plane_params::<8>(plane, off, stride);
            #[cfg(target_arch = "x86_64")]
            if ragdoll_core::dispatch::avx2() {
                // Safety: plane_params checked the block's bounds; avx2() is true only when
                // the CPU supports AVX2.
                unsafe { plane_avx2::<8>(plane, off, stride, a, b, c) };
                return;
            }
            #[cfg(target_arch = "aarch64")]
            if ragdoll_core::dispatch::neon() {
                // Safety: plane_params checked the block's bounds; neon() is true only when the
                // CPU supports NEON.
                unsafe { plane_neon::<8>(plane, off, stride, a, b, c) };
                return;
            }
            plane_fill::<8>(plane, off, stride, a, b, c);
        }
        _ => fill_block::<8>(plane, off, stride, mode, avail),
    }
}

/// Scalar reference for [`h264_pred_chroma`].
pub fn h264_pred_chroma_scalar(plane: &mut [u8], off: usize, stride: usize, mode: u8, avail: u8) {
    predict_block(plane, off, stride, 8, mode, avail, true);
}

/// Vertical, horizontal and luma DC prediction (chroma DC is [`chroma_dc`]) of an NxN block, a
/// row at a time.
#[inline(always)]
fn fill_block<const N: usize>(plane: &mut [u8], off: usize, stride: usize, mode: u8, avail: u8) {
    assert!(stride >= N && off + (N - 1) * stride + N <= plane.len());
    let left = |plane: &[u8], r: usize| plane[off + r * stride - 1];
    match mode {
        0 => {
            let top: [u8; N] = plane[off - stride..off - stride + N].try_into().unwrap();
            for r in 0..N {
                plane[off + r * stride..off + r * stride + N].copy_from_slice(&top);
            }
        }
        1 => {
            for r in 0..N {
                let v = left(plane, r);
                plane[off + r * stride..off + r * stride + N].fill(v);
            }
        }
        _ => {
            // Luma DC (8.3.3.3).
            let has_top = avail & INTRA_TOP != 0;
            let has_left = avail & INTRA_LEFT != 0;
            let st: u32 = if has_top {
                plane[off - stride..off - stride + N]
                    .iter()
                    .map(|&v| v as u32)
                    .sum()
            } else {
                0
            };
            let sl: u32 = if has_left {
                (0..N).map(|r| left(plane, r) as u32).sum()
            } else {
                0
            };
            let v = match (has_top, has_left) {
                (true, true) => (st + sl + 16) >> 5,
                (false, true) => (sl + 8) >> 4,
                (true, false) => (st + 8) >> 4,
                (false, false) => 128,
            } as u8;
            for r in 0..N {
                plane[off + r * stride..off + r * stride + N].fill(v);
            }
        }
    }
}

/// Chroma DC prediction (8.3.4.1 - 8.3.4.3): each 4x4 block from its own top and left sums.
#[inline(always)]
fn chroma_dc(plane: &mut [u8], off: usize, stride: usize, avail: u8) {
    assert!(stride >= 8 && off + 7 * stride + 8 <= plane.len());
    let has_top = avail & INTRA_TOP != 0;
    let (mut st, mut sl) = ([0u32; 2], [0u32; 2]);
    if has_top {
        for (i, &v) in plane[off - stride..off - stride + 8].iter().enumerate() {
            st[i >> 2] += v as u32;
        }
    }
    let has_left = [avail & INTRA_LEFT != 0, avail & INTRA_LEFT_LOWER != 0];
    for h in 0..2 {
        if has_left[h] {
            sl[h] = (0..4)
                .map(|r| plane[off + (4 * h + r) * stride - 1] as u32)
                .sum();
        }
    }
    for by in 0..2 {
        let has_left = has_left[by];
        let l_only = (sl[by] + 2) >> 2;
        let mut row = [0u8; 8];
        for bx in 0..2 {
            let t_only = (st[bx] + 2) >> 2;
            // The diagonal blocks use both edges; the others prefer their own (top for the
            // upper right, left for the lower left).
            let v = match (has_top, has_left) {
                (true, true) if bx == by => (st[bx] + sl[by] + 4) >> 3,
                (true, true) if bx > 0 => t_only,
                (true, true) => l_only,
                (true, false) => t_only,
                (false, true) => l_only,
                (false, false) => 128,
            };
            row[4 * bx..4 * bx + 4].fill(v as u8);
        }
        for r in 0..4 {
            let o = off + (4 * by + r) * stride;
            plane[o..o + 8].copy_from_slice(&row);
        }
    }
}

/// Plane prediction parameters `(base, b, c)` with `base = a + 16 - (N/2 - 1)(b + c)` of an NxN
/// block (N = 16 luma, 8 chroma; 8.3.3.4, 8.3.4.4), which needs all its neighbours. Sample
/// `(x, y)` is `clip((base + b x + c y) >> 5)`; `base + b x` and every `base + b x + c y` lie
/// within `i16`.
#[inline(always)]
fn plane_params<const N: usize>(plane: &[u8], off: usize, stride: usize) -> (i32, i32, i32) {
    assert!(off > stride && stride >= N && off + (N - 1) * stride + N <= plane.len());
    let top = &plane[off - stride - 1..off - stride + N];
    let pl = |i: usize| plane[off - stride - 1 + i * stride] as i32;
    let h = N / 2;
    let (mut hs, mut vs) = (0, 0);
    for i in 0..h {
        hs += (i as i32 + 1) * (top[h + 1 + i] as i32 - top[h - 1 - i] as i32);
        vs += (i as i32 + 1) * (pl(h + 1 + i) - pl(h - 1 - i));
    }
    let a = 16 * (pl(N) + top[N] as i32);
    let (b, c) = if N == 8 {
        ((34 * hs + 32) >> 6, (34 * vs + 32) >> 6)
    } else {
        ((5 * hs + 32) >> 6, (5 * vs + 32) >> 6)
    };
    let k = h as i32 - 1;
    (a + 16 - k * (b + c), b, c)
}

/// Scalar plane prediction from [`plane_params`].
fn plane_fill<const N: usize>(
    plane: &mut [u8],
    off: usize,
    stride: usize,
    base: i32,
    b: i32,
    c: i32,
) {
    for y in 0..N {
        let row = &mut plane[off + y * stride..off + y * stride + N];
        for (x, v) in row.iter_mut().enumerate() {
            *v = clip((base + b * x as i32 + c * y as i32) >> 5);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn plane_avx2<const N: usize>(
    plane: &mut [u8],
    off: usize,
    stride: usize,
    base: i32,
    b: i32,
    c: i32,
) {
    use std::arch::x86_64::*;
    // i16 lanes: plane_params bounds every sample and partial sum.
    let ramp = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let mut v = _mm256_add_epi16(
        _mm256_set1_epi16(base as i16),
        _mm256_mullo_epi16(_mm256_set1_epi16(b as i16), ramp),
    );
    let dc = _mm256_set1_epi16(c as i16);
    let q = plane.as_mut_ptr();
    for y in 0..N {
        let s = _mm256_srai_epi16::<5>(v);
        let s = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packus_epi16(s, s));
        // Safety: plane_params checked the block's bounds.
        unsafe {
            let o = q.add(off + y * stride) as *mut __m128i;
            if N == 16 {
                _mm_storeu_si128(o, _mm256_castsi256_si128(s));
            } else {
                _mm_storel_epi64(o, _mm256_castsi256_si128(s));
            }
        }
        v = _mm256_add_epi16(v, dc);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn plane_neon<const N: usize>(
    plane: &mut [u8],
    off: usize,
    stride: usize,
    base: i32,
    b: i32,
    c: i32,
) {
    use std::arch::aarch64::*;
    // i16 lanes: plane_params bounds every sample and partial sum.
    const RAMP: [i16; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let q = plane.as_mut_ptr();
    // Safety: plane_params checked the block's bounds.
    unsafe {
        let base = vdupq_n_s16(base as i16);
        let mut v0 = vmlaq_n_s16(base, vld1q_s16(RAMP.as_ptr()), b as i16);
        let mut v1 = vmlaq_n_s16(base, vld1q_s16(RAMP.as_ptr().add(8)), b as i16);
        let dc = vdupq_n_s16(c as i16);
        for y in 0..N {
            let o = q.add(off + y * stride);
            let lo = vqmovun_s16(vshrq_n_s16::<5>(v0));
            if N == 16 {
                vst1q_u8(o, vcombine_u8(lo, vqmovun_s16(vshrq_n_s16::<5>(v1))));
            } else {
                vst1_u8(o, lo);
            }
            v0 = vaddq_s16(v0, dc);
            v1 = vaddq_s16(v1, dc);
        }
    }
}

/// The 16x16 and chroma predictors sample by sample as 8.3.3 and 8.3.4 state them.
fn predict_block(
    plane: &mut [u8],
    off: usize,
    stride: usize,
    w: usize,
    mode: u8,
    avail: u8,
    chroma: bool,
) {
    let h = w;
    let has_top = avail & INTRA_TOP != 0;
    let left_avail = if chroma {
        [avail & INTRA_LEFT != 0, avail & INTRA_LEFT_LOWER != 0]
    } else {
        [avail & INTRA_LEFT != 0; 2]
    };
    let has_left = left_avail[0] && left_avail[1];
    let mut out = [0u8; 256];
    {
        let plane = &*plane;
        let top = |i: usize| plane[off - stride + i] as i32;
        let left = |i: usize| plane[off + i * stride - 1] as i32;
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
                let corner = plane[off - stride - 1] as i32;
                let pt = |i: i32| if i < 0 { corner } else { top(i as usize) };
                let pl = |i: i32| if i < 0 { corner } else { left(i as usize) };
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
    for r in 0..h {
        plane[off + r * stride..off + r * stride + w].copy_from_slice(&out[r * w..r * w + w]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic noise with runs of extremes.
    fn noise(n: usize, seed: u32) -> Vec<u8> {
        let mut x = seed | 1;
        (0..n)
            .map(|i| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                match (i / 5 + seed as usize) % 7 {
                    0 => 0,
                    1 => 255,
                    _ => (x >> 24) as u8,
                }
            })
            .collect()
    }

    /// Runs `fast` and `reference` on the same planes for every mode and availability set and a
    /// few positions and strides, comparing whole planes.
    fn check(
        n: usize,
        modes: &[u8],
        avails: &[u8],
        fast: impl Fn(&mut [u8], usize, usize, u8, u8),
        reference: impl Fn(&mut [u8], usize, usize, u8, u8),
    ) {
        for (seed, stride) in [(1u32, n + 1), (2, n + 3), (3, 2 * n + 5), (4, 37)] {
            let rows = n + 4;
            for avail in avails.iter().copied() {
                for &mode in modes {
                    for x in [1, stride - n] {
                        let plane =
                            noise(rows * stride, seed * 977 + avail as u32 * 31 + mode as u32);
                        let off = 2 * stride + x;
                        let (mut a, mut b) = (plane.clone(), plane);
                        fast(&mut a, off, stride, mode, avail);
                        reference(&mut b, off, stride, mode, avail);
                        assert_eq!(
                            a, b,
                            "n {n} mode {mode} avail {avail:#x} stride {stride} x {x}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn pred4x4_matches_scalar() {
        let avails: Vec<u8> = (0..16).collect();
        let modes: Vec<u8> = (0..10).collect();
        check(4, &modes, &avails, h264_pred4x4, h264_pred4x4_scalar);
    }

    #[test]
    fn pred8x8l_matches_scalar() {
        let avails: Vec<u8> = (0..16).collect();
        let modes: Vec<u8> = (0..10).collect();
        check(8, &modes, &avails, h264_pred8x8l, h264_pred8x8l_scalar);
    }

    #[test]
    fn pred16x16_matches_scalar() {
        // Each mode only with the neighbours it needs.
        let all = INTRA_LEFT | INTRA_TOP | INTRA_TOP_LEFT;
        for (mode, avails) in [
            (0, vec![INTRA_TOP, all]),
            (1, vec![INTRA_LEFT, all]),
            (2, vec![0, INTRA_LEFT, INTRA_TOP, all]),
            (3, vec![all]),
        ] {
            check(16, &[mode], &avails, h264_pred16x16, h264_pred16x16_scalar);
        }
    }

    #[test]
    fn pred_chroma_matches_scalar() {
        let all = INTRA_LEFT | INTRA_LEFT_LOWER | INTRA_TOP | INTRA_TOP_LEFT;
        let dc: Vec<u8> = (0..32).filter(|a| a & INTRA_TOP_RIGHT == 0).collect();
        for (mode, avails) in [
            (0, vec![INTRA_TOP, all]),
            (1, vec![INTRA_LEFT | INTRA_LEFT_LOWER, all]),
            (2, dc),
            (3, vec![all]),
        ] {
            check(
                8,
                &[mode],
                &avails,
                h264_pred_chroma,
                h264_pred_chroma_scalar,
            );
        }
    }

    #[test]
    fn plane_prediction_reaches_both_clip_bounds() {
        // Steepest gradients: the i16 bounds of plane_params are tight here.
        for (t, l) in [(0u8, 255u8), (255, 0)] {
            for n in [8usize, 16] {
                let stride = n + 1;
                let mut plane = vec![0u8; (n + 1) * stride];
                for i in 0..n {
                    plane[i + 1] = if i < n / 2 { l } else { t };
                    plane[(i + 1) * stride] = if i < n / 2 { t } else { l };
                }
                plane[0] = l;
                let mut a = plane.clone();
                let mut b = plane;
                if n == 16 {
                    h264_pred16x16(&mut a, stride + 1, stride, 3, 11);
                    h264_pred16x16_scalar(&mut b, stride + 1, stride, 3, 11);
                } else {
                    h264_pred_chroma(&mut a, stride + 1, stride, 3, 27);
                    h264_pred_chroma_scalar(&mut b, stride + 1, stride, 3, 27);
                }
                assert_eq!(a, b);
            }
        }
    }
}
