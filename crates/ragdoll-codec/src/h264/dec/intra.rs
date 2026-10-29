//! Intra prediction (ITU-T H.264 8.3): neighbour availability for the predictors of
//! `ragdoll_simd::h264_intra`, which write straight into the picture.

pub use ragdoll_simd::{
    INTRA_LEFT, INTRA_LEFT_LOWER, INTRA_TOP, INTRA_TOP_LEFT, INTRA_TOP_RIGHT,
    h264_pred_chroma as pred_chroma, h264_pred4x4 as pred4x4, h264_pred8x8l as pred8x8l,
    h264_pred16x16 as pred16x16,
};

use super::tables::{BLK4_AT, BLK4_XY};

/// Availability flags from `(left, top, top_right, top_left)`; the top-right counts only with
/// the top.
#[inline]
pub const fn flags((l, t, tr, tl): (bool, bool, bool, bool)) -> u8 {
    const fn bit(b: bool, f: u8) -> u8 {
        if b { f } else { 0 }
    }
    bit(l, INTRA_LEFT) | bit(t, INTRA_TOP) | bit(tr && t, INTRA_TOP_RIGHT) | bit(tl, INTRA_TOP_LEFT)
}

/// Availability flags of the `4 n4`x`4 n4` luma block at MB-relative 4x4 `(bx, by)`, `cur` in
/// `luma4x4BlkIdx` order, when the neighbouring macroblocks' availability is `nb` (the same
/// flags: left A, top B, top-right C, top-left D) for all their rows, as without MBAFF. Blocks of
/// the current macroblock are available once decoded; the one to its right never is.
const fn block_flags(bx: usize, by: usize, n4: usize, cur: usize, nb: u8) -> u8 {
    let (a, b, c, d) = (
        nb & INTRA_LEFT != 0,
        nb & INTRA_TOP != 0,
        nb & INTRA_TOP_RIGHT != 0,
        nb & INTRA_TOP_LEFT != 0,
    );
    let left = bx > 0 || a;
    let top = by > 0 || b;
    let tr = if by == 0 {
        if bx + n4 < 4 { b } else { c }
    } else {
        bx + n4 < 4 && BLK4_AT[by - 1][bx + n4] < cur
    };
    let tl = match (bx > 0, by > 0) {
        (true, true) => true,
        (true, false) => b,
        (false, true) => a,
        (false, false) => d,
    };
    flags((left, top, tr, tl))
}

/// [`block_flags`] of each 4x4 block (`[luma4x4BlkIdx][nb]`).
pub const AVAIL4: [[u8; 16]; 16] = {
    let mut t = [[0; 16]; 16];
    let mut blk = 0;
    while blk < 16 {
        let (bx, by) = BLK4_XY[blk];
        let mut nb = 0;
        while nb < 16 {
            t[blk][nb] = block_flags(bx, by, 1, blk, nb as u8);
            nb += 1;
        }
        blk += 1;
    }
    t
};

/// [`block_flags`] of each 8x8 block (`[luma8x8BlkIdx][nb]`).
pub const AVAIL8: [[u8; 16]; 4] = {
    let mut t = [[0; 16]; 4];
    let mut b8 = 0;
    while b8 < 4 {
        let mut nb = 0;
        while nb < 16 {
            t[b8][nb] = block_flags((b8 & 1) * 2, (b8 >> 1) * 2, 2, b8 * 4, nb as u8);
            nb += 1;
        }
        b8 += 1;
    }
    t
};

/// Whether a 16x16 / chroma mode (V=0, H=1, DC=2, Plane=3) only reads available samples.
pub fn block_mode_ok(mode: u8, has_left: bool, has_top: bool, has_tl: bool) -> bool {
    match mode {
        0 => has_top,
        1 => has_left,
        2 => true,
        _ => has_left && has_top && has_tl,
    }
}
