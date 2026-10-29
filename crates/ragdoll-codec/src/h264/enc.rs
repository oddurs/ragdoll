//! [`H264Encoder`] frame coding (`SIMD-0089`): macroblock analysis, mode decision, residual coding
//! and in-loop deblocking over edge-padded reconstruction planes.
//!
//! Decisions follow `x264`'s fast presets: a `P_Skip` probe that quantizes the skip residual before
//! any search, a predictor-seeded diamond motion search refined to quarter-pel, SATD-ranked intra
//! modes, and decimation of nearly empty inter blocks. The reconstruction is written straight into
//! planes padded by edge replication, so motion search and compensation of the next frame read
//! reference samples without clamping.

use super::*;
use crate::cabac::NUM_CABAC_CONTEXTS;
use ragdoll_simd::{
    H264_DEBLOCK_ALPHA, H264_DEBLOCK_BETA, H264_DEBLOCK_TC0, HPEL_WIN_W, HpelWindow, Quant4x4,
    add_idct4x4_8x8, dequant_4x4_x4, h264_chroma_deblock, h264_chroma_mc, h264_luma_deblock,
    h264_luma_mc, hadamard4x4_16x4, hpel_window, hpel_window_centre, intra_pred_plane,
    intra_satd_x3_16x16, intra4x4_pred_dir, nonzero_mask_4x4_x4, pixel_avg_16x16, pixel_sad_8x8,
    pixel_sad_16x16, pixel_sad_avg_16x16, pixel_sad_x4_16x16, pixel_satd_4x4, pixel_satd_16x16,
    pixel_ssd, quant_4x4_x4, sub_dct4x4_8x8, sub_dct4x4_16x4, zigzag_4x4_x4,
};

/// Luma padding of reconstructed planes, in samples: the half-pel window of a block displaced
/// by [`MV_MARGIN`] past the right edge reads 45 samples beyond it.
const PAD_Y: usize = 48;
/// Chroma padding of reconstructed planes, in samples.
const PAD_C: usize = 16;
/// How far past the picture edge (luma samples) a motion vector may move a block; the chroma
/// block then stays two samples inside [`PAD_C`].
const MV_MARGIN: i32 = 28;

/// `x264_lambda_tab`: Lagrangian multiplier of the SAD / SATD domain per QP.
const LAMBDA: [u16; 52] = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 6, 7, 8, 9,
    10, 11, 13, 14, 16, 18, 20, 23, 25, 29, 32, 36, 40, 45, 51, 57, 64, 72, 81, 91,
];

/// Coefficient decimation weights by preceding zero run (`x264_decimate_table4/8`).
const DECIMATE4: [u8; 16] = [3, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const DECIMATE8: [u8; 64] = [
    3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// A plane whose samples are surrounded by `pad` replicated edge samples on every side.
#[derive(Clone, Debug, Default)]
pub(super) struct Plane {
    buf: Vec<u8>,
    stride: usize,
    w: usize,
    h: usize,
    pad: usize,
}

impl Plane {
    fn alloc(w: usize, h: usize, pad: usize) -> Self {
        let stride = (w + 2 * pad + 31) & !31;
        Self {
            buf: vec![0; stride * (h + 2 * pad)],
            stride,
            w,
            h,
            pad,
        }
    }

    /// Buffer index of sample `(x, y)`, which may lie in the padding.
    #[inline(always)]
    fn at(&self, x: isize, y: isize) -> usize {
        ((y + self.pad as isize) * self.stride as isize + x + self.pad as isize) as usize
    }

    #[inline(always)]
    fn idx(&self, x: usize, y: usize) -> usize {
        (y + self.pad) * self.stride + x + self.pad
    }

    /// Writes the `N`x`N` block `src` (stride `N`) with its top-left sample at `(x0, y0)`.
    #[inline(always)]
    fn store<const N: usize>(&mut self, x0: usize, y0: usize, src: &[u8]) {
        let o = self.idx(x0, y0);
        copy_rows::<N>(src, 0, N, &mut self.buf, o, self.stride, N);
    }

    /// Writes the `N`x`N` block of `src` whose top-left sample is at buffer index `so` with its
    /// top-left sample at `(x0, y0)`.
    #[inline(always)]
    fn copy_from<const N: usize>(&mut self, src: &Plane, so: usize, x0: usize, y0: usize) {
        let o = self.idx(x0, y0);
        copy_rows::<N>(&src.buf, so, src.stride, &mut self.buf, o, self.stride, N);
    }

    /// Reads the `N`x`N` block whose top-left sample is at buffer index `o` into `dst`.
    #[inline(always)]
    fn load<const N: usize>(&self, o: usize, dst: &mut [u8]) {
        copy_rows::<N>(&self.buf, o, self.stride, dst, 0, N, N);
    }

    /// Replicates the outermost samples into the padding.
    fn extend_borders(&mut self) {
        let (p, s, w, h) = (self.pad, self.stride, self.w, self.h);
        for y in p..p + h {
            let row = &mut self.buf[y * s..(y + 1) * s];
            let (l, r) = (row[p], row[p + w - 1]);
            row[..p].fill(l);
            row[p + w..].fill(r);
        }
        for y in 0..p {
            self.buf.copy_within(p * s..(p + 1) * s, y * s);
            self.buf
                .copy_within((p + h - 1) * s..(p + h) * s, (p + h + y) * s);
        }
    }

    #[cfg(test)]
    fn to_plane_rows(&self, dst: &mut VideoFrame, plane: usize) {
        let (w, h) = (dst.plane(plane).width(), dst.plane(plane).height());
        for y in 0..h.min(self.h) {
            let o = self.idx(0, y);
            dst.plane_mut(plane).row_mut(y)[..w.min(self.w)]
                .copy_from_slice(&self.buf[o..o + w.min(self.w)]);
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct Pic {
    y: Plane,
    u: Plane,
    v: Plane,
}

impl Pic {
    fn alloc(w: usize, h: usize) -> Self {
        Self {
            y: Plane::alloc(w, h, PAD_Y),
            u: Plane::alloc(w / 2, h / 2, PAD_C),
            v: Plane::alloc(w / 2, h / 2, PAD_C),
        }
    }

    fn store_mb(&mut self, mb_x: usize, mb_y: usize, y: &[u8; 256], u: &[u8; 64], v: &[u8; 64]) {
        self.y.store::<16>(mb_x * 16, mb_y * 16, y);
        self.u.store::<8>(mb_x * 8, mb_y * 8, u);
        self.v.store::<8>(mb_x * 8, mb_y * 8, v);
    }
}

/// Picture buffers and per-frame macroblock state, kept across frames so steady-state encoding
/// allocates nothing per macroblock.
#[derive(Clone, Debug, Default)]
pub(super) struct EncState {
    mb_w: usize,
    mb_h: usize,
    cur: Pic,
    refp: Pic,
    pub(super) has_ref: bool,
    luma_nnz: Vec<u8>,
    chroma_nnz: [Vec<u8>; 2],
    intra4x4_modes: Vec<u8>,
    ref_grid: Vec<i8>,
    mv_grid: Vec<[i16; 2]>,
    mvd_grid: Vec<[i16; 2]>,
    prev_mb_mv: Vec<[i16; 2]>,
    mb_is_intra: Vec<bool>,
    mb_is_i16x16: Vec<bool>,
    mb_is_skip: Vec<bool>,
    mb_cbp_luma: Vec<u8>,
    mb_cbp_chroma: Vec<u8>,
    mb_chroma_mode: Vec<u8>,
    mb_luma_dc_cbf: Vec<bool>,
    mb_chroma_dc_cbf: [Vec<bool>; 2],
    mb_8x8_flag: Vec<bool>,
    mb_qps: Vec<u8>,
    mb_vars: Vec<f32>,
    bits_buf: Vec<u8>,
    win: Box<HpelWindow>,
}

impl EncState {
    /// Sizes the pictures and grids and clears the per-frame state. `ref_grid` and `mv_grid` are
    /// left alone: every macroblock writes its own entries before a later one reads them.
    fn prepare(&mut self, mb_w: usize, mb_h: usize, slice_qp: u8, cabac: bool) {
        let (n4, n2, n) = (mb_w * mb_h * 16, mb_w * mb_h * 4, mb_w * mb_h);
        if self.mb_w != mb_w || self.mb_h != mb_h {
            *self = Self {
                mb_w,
                mb_h,
                cur: Pic::alloc(mb_w * 16, mb_h * 16),
                refp: Pic::alloc(mb_w * 16, mb_h * 16),
                ref_grid: vec![-2; n4],
                mv_grid: vec![[0; 2]; n4],
                prev_mb_mv: vec![[0; 2]; n],
                ..Self::default()
            };
        }
        let reset = |v: &mut Vec<u8>, len: usize, x: u8| {
            v.clear();
            v.resize(len, x);
        };
        reset(&mut self.luma_nnz, n4, 0);
        reset(&mut self.chroma_nnz[0], n2, 0);
        reset(&mut self.chroma_nnz[1], n2, 0);
        reset(&mut self.intra4x4_modes, n4, 2);
        reset(&mut self.mb_cbp_luma, n, 0);
        reset(&mut self.mb_cbp_chroma, n, 0);
        reset(&mut self.mb_chroma_mode, n, 0);
        reset(&mut self.mb_qps, n, slice_qp);
        let reset_b = |v: &mut Vec<bool>| {
            v.clear();
            v.resize(n, false);
        };
        reset_b(&mut self.mb_is_intra);
        reset_b(&mut self.mb_is_i16x16);
        reset_b(&mut self.mb_is_skip);
        reset_b(&mut self.mb_luma_dc_cbf);
        reset_b(&mut self.mb_chroma_dc_cbf[0]);
        reset_b(&mut self.mb_chroma_dc_cbf[1]);
        reset_b(&mut self.mb_8x8_flag);
        // Only CABAC reads the neighbours' motion vector differences.
        if cabac {
            self.mvd_grid.clear();
            self.mvd_grid.resize(n4, [0; 2]);
        }
    }

    /// Marks macroblock `(mb_x, mb_y)` intra in the motion grids: no reference, zero vectors.
    fn set_intra_motion(&mut self, mb_x: usize, mb_y: usize) {
        let g_stride = self.mb_w * 4;
        fill_grid_4x4(&mut self.ref_grid, g_stride, mb_x * 4, mb_y * 4, 4, 4, -1);
        fill_grid_4x4(
            &mut self.mv_grid,
            g_stride,
            mb_x * 4,
            mb_y * 4,
            4,
            4,
            [0; 2],
        );
    }

    /// The latest reconstructed (deblocked) picture at coded size.
    #[cfg(test)]
    pub(super) fn recon_frame(&self) -> VideoFrame {
        let mut f = VideoFrame::new(self.mb_w * 16, self.mb_h * 16, PixelFormat::Yuv420p);
        self.refp.y.to_plane_rows(&mut f, 0);
        self.refp.u.to_plane_rows(&mut f, 1);
        self.refp.v.to_plane_rows(&mut f, 2);
        f
    }
}

/// Source samples of one macroblock.
struct MbSrc {
    y: [u8; 256],
    u: [u8; 64],
    v: [u8; 64],
}

impl MbSrc {
    /// Loads macroblock `(mb_x, mb_y)` of `frame`, replicating the last column and row past the
    /// picture's right and bottom edges.
    fn load(&mut self, frame: &VideoFrame, mb_x: usize, mb_y: usize) {
        let (x0, y0) = (mb_x * 16, mb_y * 16);
        if x0 + 16 > frame.width || y0 + 16 > frame.height {
            extract_mb_planes(frame, x0, y0, &mut self.y, &mut self.u, &mut self.v);
            return;
        }
        let (py, pu, pv) = (frame.plane(0), frame.plane(1), frame.plane(2));
        let (s, cs) = (py.stride(), pu.stride());
        copy_rows::<16>(py.data(), y0 * s + x0, s, &mut self.y, 0, 16, 16);
        let co = (y0 / 2) * cs + x0 / 2;
        copy_rows::<8>(pu.data(), co, cs, &mut self.u, 0, 8, 8);
        copy_rows::<8>(pv.data(), co, pv.stride(), &mut self.v, 0, 8, 8);
    }
}

/// Copies `rows` rows of `N` bytes from `src` (stride `ss`, first byte `so`) to `dst` (stride
/// `ds`, first byte `d0`), checking the bounds once.
#[inline(always)]
fn copy_rows<const N: usize>(
    src: &[u8],
    so: usize,
    ss: usize,
    dst: &mut [u8],
    d0: usize,
    ds: usize,
    rows: usize,
) {
    assert!(
        rows > 0 && so + (rows - 1) * ss + N <= src.len() && d0 + (rows - 1) * ds + N <= dst.len()
    );
    for r in 0..rows {
        // Safety: the assertion bounds every row of both blocks; `dst` is a distinct borrow.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr().add(so + r * ss),
                dst.as_mut_ptr().add(d0 + r * ds),
                N,
            );
        }
    }
}

/// Entropy coder of the slice being written.
type Cabac = Option<(CabacEncoder, [CabacContext; NUM_CABAC_CONTEXTS])>;

/// Slice-level coding state carried from macroblock to macroblock.
struct SliceState {
    is_idr: bool,
    use_cabac: bool,
    t8x8: bool,
    chroma_qp_offset: i32,
    skip_run: u32,
    cur_qp: i32,
    prev_qp_delta_nz: bool,
    skip_bounds: SkipBounds,
}

#[inline(always)]
fn blk_xy(b: usize) -> (usize, usize) {
    ((b & 1) + ((b >> 1) & 2), ((b >> 1) & 1) + ((b >> 2) & 2))
}

#[inline(always)]
fn chroma_qp(qp: i32, offset: i32) -> u8 {
    CHROMA_QP_TABLE[(qp + offset).clamp(0, 51) as usize]
}

/// Bits of the signed Exp-Golomb code of `d`.
#[inline(always)]
fn se_bits(d: i32) -> u32 {
    let v = d.unsigned_abs() * 2 - u32::from(d > 0);
    2 * (32 - (v + 1).leading_zeros()) - 1
}

/// `x264_decimate_score`: 9 when a level exceeds 1, else a weight that falls with the zero runs.
fn decimate_score(scanned: &[i16]) -> u32 {
    let table: &[u8] = if scanned.len() == 64 {
        &DECIMATE8
    } else {
        &DECIMATE4
    };
    let mut idx = scanned.len();
    while idx > 0 && scanned[idx - 1] == 0 {
        idx -= 1;
    }
    let mut score = 0u32;
    while idx > 0 {
        idx -= 1;
        if scanned[idx].unsigned_abs() > 1 {
            return 9;
        }
        let mut run = 0;
        while idx > 0 && scanned[idx - 1] == 0 {
            idx -= 1;
            run += 1;
        }
        score += table[run] as u32;
    }
    score
}

/// [`decimate_score`] of a scanned $4\times4$ block (16 or 15 levels) given the mask of its
/// non-zero levels (bit `i` for `scanned[i]`).
#[inline]
fn decimate_masked(scanned: &[i16], mask: u32) -> u32 {
    let mut score = 0u32;
    let mut m = mask;
    while m != 0 {
        let p = 31 - m.leading_zeros() as i32;
        if scanned[p as usize].unsigned_abs() > 1 {
            return 9;
        }
        m ^= 1 << p;
        let below = 31 - m.leading_zeros() as i32; // -1 once `m` is empty
        score += DECIMATE4[(p - below - 1) as usize] as u32;
    }
    score
}

#[inline(always)]
fn zigzag16(q: &[i16; 16]) -> [i16; 16] {
    ZIGZAG_4X4.map(|i| q[i])
}

fn zigzag64(q: &[i16; 64]) -> [i16; 64] {
    ZIGZAG_8X8.map(|i| q[i])
}

/// CAVLC `level_prefix` / `level_suffix` code of a `levelCode` at a `suffixLength`, as
/// `(bits, length)` (ITU-T H.264 9.2.2.1).
const fn level_vlc(code: u32, suffix_len: u32) -> (u32, u32) {
    if code < (14 << suffix_len) {
        // level_prefix zeros, a one, then suffix_len suffix bits.
        let prefix = code >> suffix_len;
        (
            (1 << suffix_len) | (code & ((1 << suffix_len) - 1)),
            prefix + 1 + suffix_len,
        )
    } else if suffix_len == 0 && code < 30 {
        ((1 << 4) | (code - 14), 19)
    } else if suffix_len > 0 && code < (15 << suffix_len) {
        (
            (1 << suffix_len) | (code - (14 << suffix_len)),
            15 + suffix_len,
        )
    } else {
        let mut esc = code - (15 << suffix_len);
        if suffix_len == 0 {
            esc -= 15;
        }
        ((1 << 12) | (esc & 0xFFF), 28)
    }
}

/// [`level_vlc`] of the small level codes, packed as `bits << 8 | length`.
static LEVEL_VLC: [[u32; 64]; 7] = {
    let mut t = [[0u32; 64]; 7];
    let mut s = 0;
    while s < 7 {
        let mut c = 0;
        while c < 64 {
            let (bits, len) = level_vlc(c as u32, s as u32);
            t[s][c] = bits << 8 | len;
            c += 1;
        }
        s += 1;
    }
    t
};

/// `suffixLength` after coding a level of magnitude `abs` at `suffix_len` (ITU-T H.264 9.2.2.1).
#[inline(always)]
const fn next_suffix_len(suffix_len: u32, abs: u32) -> u32 {
    let s = if suffix_len == 0 { 1 } else { suffix_len };
    if abs > (3 << (s - 1)) && s < 6 {
        s + 1
    } else {
        s
    }
}

/// Largest level magnitude [`LEVEL_TAB`] covers.
const LEVEL_TAB_MAX: i32 = 32;

/// Level codes by `suffixLength` and `level + LEVEL_TAB_MAX` for the levels that need no
/// `levelCode` adjustment, packed as `bits << 16 | next suffixLength << 8 | length`
/// (`x264_level_token`).
static LEVEL_TAB: [[u32; 2 * LEVEL_TAB_MAX as usize]; 7] = {
    let mut t = [[0u32; 2 * LEVEL_TAB_MAX as usize]; 7];
    let mut s = 0;
    while s < 7 {
        let mut i = 0;
        while i < 2 * LEVEL_TAB_MAX {
            let lv = i - LEVEL_TAB_MAX;
            if lv != 0 {
                let code = if lv > 0 { 2 * lv - 2 } else { -2 * lv - 1 } as u32;
                let (bits, len) = level_vlc(code, s as u32);
                let next = next_suffix_len(s as u32, lv.unsigned_abs());
                t[s][i as usize] = bits << 16 | next << 8 | len;
            }
            i += 1;
        }
        s += 1;
    }
    t
};

/// CAVLC `residual_block` of a zig-zag scanned block of `N` = 16 or 15 coefficients (`nc` selects
/// the `coeff_token` table), or of the 4 chroma DC coefficients (`N` = 4); the bits equal
/// [`encode_cavlc_block`]'s.
fn write_cavlc_block<const N: usize>(w: &mut Bits, coeffs: &[i16; N], nc: u8) {
    let mut mask = 0u32;
    for (i, &c) in coeffs.iter().enumerate() {
        mask |= u32::from(c != 0) << i;
    }
    write_cavlc_masked::<N>(w, coeffs, nc, mask);
}

/// [`write_cavlc_block`] of the first `N` levels of `coeffs` given the mask of the non-zero ones
/// (bit `i` for `coeffs[i]`). Levels and runs are visited from the highest frequency down through
/// the mask, so the cost follows the number of coded levels.
#[inline]
fn write_cavlc_masked<const N: usize>(w: &mut Bits, coeffs: &[i16], nc: u8, mask: u32) {
    let (token_len, token_bits): (&[u8], &[u8]) = if N == 4 {
        (&CHROMA_DC_COEFF_TOKEN_LEN, &CHROMA_DC_COEFF_TOKEN_BITS)
    } else {
        let t = nc_to_table_idx(nc);
        (&COEFF_TOKEN_LEN[t], &COEFF_TOKEN_BITS[t])
    };
    if mask == 0 {
        w.put32(token_bits[0] as u32, token_len[0] as u32);
        return;
    }
    let n = mask.count_ones() as usize;
    let top = 31 - mask.leading_zeros();
    let total_zeros = top as usize + 1 - n;

    // Up to three trailing +-1 levels, coded as sign bits after the token.
    let mut m = mask;
    let mut trailing_ones = 0usize;
    let mut signs = 0u32;
    while trailing_ones < 3 && m != 0 {
        let p = 31 - m.leading_zeros();
        let c = coeffs[p as usize];
        if c.unsigned_abs() != 1 {
            break;
        }
        signs = (signs << 1) | u32::from(c < 0);
        trailing_ones += 1;
        m ^= 1 << p;
    }
    let tok = n * 4 + trailing_ones;
    w.put32(
        ((token_bits[tok] as u32) << trailing_ones) | signs,
        token_len[tok] as u32 + trailing_ones as u32,
    );

    if m != 0 {
        let mut suffix_len = u32::from(n > 10 && trailing_ones < 3);
        // The first remaining level cannot be +-1 when fewer than three trailing ones were
        // taken, so its `levelCode` is lowered by 2.
        let p = 31 - m.leading_zeros();
        m ^= 1 << p;
        let lv = coeffs[p as usize] as i32;
        let mut code = if lv > 0 { 2 * lv - 2 } else { -2 * lv - 1 } as u32;
        if trailing_ones < 3 {
            code -= 2;
        }
        let (bits, len) = if code < 64 {
            let e = LEVEL_VLC[suffix_len as usize][code as usize];
            (e >> 8, e & 0xFF)
        } else {
            level_vlc(code, suffix_len)
        };
        w.put32(bits, len);
        suffix_len = next_suffix_len(suffix_len, lv.unsigned_abs());
        while m != 0 {
            let p = 31 - m.leading_zeros();
            m ^= 1 << p;
            let lv = coeffs[p as usize] as i32;
            let i = lv + LEVEL_TAB_MAX;
            if (i as u32) < 2 * LEVEL_TAB_MAX as u32 {
                let e = LEVEL_TAB[suffix_len as usize][i as usize];
                w.put32(e >> 16, e & 0xFF);
                suffix_len = (e >> 8) & 0xFF;
            } else {
                let code = if lv > 0 { 2 * lv - 2 } else { -2 * lv - 1 } as u32;
                let (bits, len) = level_vlc(code, suffix_len);
                w.put32(bits, len);
                suffix_len = next_suffix_len(suffix_len, lv.unsigned_abs());
            }
        }
    }

    if n < N {
        let (l, b) = if N == 4 {
            (
                CHROMA_DC_TOTAL_ZEROS_LEN[n - 1][total_zeros],
                CHROMA_DC_TOTAL_ZEROS_BITS[n - 1][total_zeros],
            )
        } else {
            (
                TOTAL_ZEROS_LEN[n - 1][total_zeros],
                TOTAL_ZEROS_BITS[n - 1][total_zeros],
            )
        };
        w.put32(b as u32, l as u32);
    }
    // run_before of each level from the highest down while zeros remain, except the last.
    let mut zeros_left = total_zeros as u32;
    let mut m = mask ^ (1 << top);
    let mut p = top;
    while zeros_left != 0 && m != 0 {
        let below = 31 - m.leading_zeros();
        m ^= 1 << below;
        let run = (p - below - 1) as usize;
        let k = (zeros_left - 1).min(6) as usize;
        w.put32(RUN_BITS[k][run] as u32, RUN_LEN[k][run] as u32);
        zeros_left -= run as u32;
        p = below;
    }
}

/// Writes the CAVLC luma $4\times4$ blocks of the coded $8\times8$ quadrants of `cbp_luma` and
/// the chroma residual, updating the non-zero counts that predict `nC`.
#[allow(clippy::too_many_arguments)]
fn write_cavlc_residuals(
    w: &mut Bits,
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    cbp_luma: u8,
    cbp_chroma: u8,
    luma_q4: &[[i16; 16]; 16],
    chroma: &ChromaLevels,
    luma_nnz: &mut [u8],
    chroma_nnz: &mut [Vec<u8>; 2],
) {
    let g_stride = mb_w * 4;
    for i8x8 in 0..4 {
        let coded = cbp_luma & (1 << i8x8) != 0;
        let group: &[[i16; 16]; 4] = luma_q4[i8x8 * 4..i8x8 * 4 + 4].try_into().unwrap();
        let (scanned, masks) = if coded {
            let zz = zigzag_4x4_x4(group);
            let masks = nonzero_mask_4x4_x4(&zz);
            (zz, masks)
        } else {
            ([[0; 16]; 4], [0; 4])
        };
        for (k, (zz, &mask)) in scanned.iter().zip(&masks).enumerate() {
            let (bx, by) = blk_xy(i8x8 * 4 + k);
            let g = (mb_y * 4 + by) * g_stride + mb_x * 4 + bx;
            if coded {
                let nc = predict_luma_nc(luma_nnz, g_stride, mb_x * 4 + bx, mb_y * 4 + by);
                write_cavlc_masked::<16>(w, zz, nc, mask);
            }
            luma_nnz[g] = mask.count_ones() as u8;
        }
    }
    write_cavlc_chroma(w, mb_x, mb_y, mb_w, cbp_chroma, chroma, chroma_nnz);
}

/// Writes the CAVLC chroma DC and AC blocks that `cbp_chroma` signals.
fn write_cavlc_chroma(
    w: &mut Bits,
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    cbp_chroma: u8,
    chroma: &ChromaLevels,
    chroma_nnz: &mut [Vec<u8>; 2],
) {
    if cbp_chroma >= 1 {
        for dc in &chroma.dc {
            write_cavlc_block(w, dc, 0);
        }
    }
    if cbp_chroma >= 2 {
        let cg_stride = mb_w * 2;
        for (p, nnz) in chroma_nnz.iter_mut().enumerate() {
            for (b, (ac, &mask)) in chroma.ac[p].iter().zip(&chroma.ac_masks[p]).enumerate() {
                let (cgx, cgy) = (mb_x * 2 + (b & 1), mb_y * 2 + (b >> 1));
                let nc = predict_chroma_nc(nnz, cg_stride, cgx, cgy);
                write_cavlc_masked::<15>(w, ac, nc, mask);
                nnz[cgy * cg_stride + cgx] = mask.count_ones() as u8;
            }
        }
    }
}

/// Luma motion compensation of the macroblock at `(x0, y0)` displaced by quarter-pel `mv`.
#[inline(always)]
fn mc_luma(r: &Plane, x0: usize, y0: usize, mv: [i16; 2], dst: &mut [u8; 256]) {
    let off = r.at(
        x0 as isize + (mv[0] >> 2) as isize,
        y0 as isize + (mv[1] >> 2) as isize,
    );
    if (mv[0] | mv[1]) & 3 == 0 {
        r.load::<16>(off, dst);
        return;
    }
    h264_luma_mc(
        dst,
        16,
        &r.buf,
        off,
        r.stride,
        16,
        16,
        (mv[0] & 3) as u8,
        (mv[1] & 3) as u8,
    );
}

/// Chroma motion compensation of the macroblock whose chroma origin is `(cx0, cy0)`.
#[inline(always)]
fn mc_chroma(p: &Pic, cx0: usize, cy0: usize, mv: [i16; 2], u: &mut [u8; 64], v: &mut [u8; 64]) {
    let off = p.u.at(
        cx0 as isize + (mv[0] >> 3) as isize,
        cy0 as isize + (mv[1] >> 3) as isize,
    );
    let (fx, fy) = ((mv[0] & 7) as u8, (mv[1] & 7) as u8);
    if fx | fy == 0 {
        p.u.load::<8>(off, u);
        p.v.load::<8>(off, v);
        return;
    }
    h264_chroma_mc(u, 8, &p.u.buf, off, p.u.stride, 8, 8, fx, fy);
    h264_chroma_mc(v, 8, &p.v.buf, off, p.v.stride, 8, 8, fx, fy);
}

/// Quarter-pel motion vector bounds of one macroblock that keep compensation inside the padding.
#[derive(Clone, Copy)]
struct MvBounds {
    min: [i32; 2],
    max: [i32; 2],
}

impl MvBounds {
    fn new(x0: usize, y0: usize, w: usize, h: usize) -> Self {
        let (x0, y0, w, h) = (x0 as i32, y0 as i32, w as i32, h as i32);
        // Level limit on vertical vectors (Table A-1, levels >= 3.1) bounds very tall pictures.
        Self {
            min: [-4 * (x0 + MV_MARGIN), (-4 * (y0 + MV_MARGIN)).max(-2048)],
            max: [
                4 * (w - 16 - x0 + MV_MARGIN),
                (4 * (h - 16 - y0 + MV_MARGIN)).min(2044),
            ],
        }
    }

    #[inline(always)]
    fn contains(&self, mv: [i16; 2]) -> bool {
        (self.min[0]..=self.max[0]).contains(&(mv[0] as i32))
            && (self.min[1]..=self.max[1]).contains(&(mv[1] as i32))
    }

    #[inline(always)]
    fn clamp(&self, mv: [i32; 2]) -> [i32; 2] {
        [
            mv[0].clamp(self.min[0], self.max[0]),
            mv[1].clamp(self.min[1], self.max[1]),
        ]
    }
}

/// Motion search of one $16\times16$ partition against the reference luma plane.
struct MotionSearch<'a> {
    r: &'a Plane,
    src: &'a [u8; 256],
    x0: usize,
    y0: usize,
    mvp: [i16; 2],
    lambda: u32,
    bounds: MvBounds,
}

impl MotionSearch<'_> {
    #[inline(always)]
    fn mv_cost(&self, mv: [i32; 2]) -> u32 {
        self.lambda * (se_bits(mv[0] - self.mvp[0] as i32) + se_bits(mv[1] - self.mvp[1] as i32))
    }

    #[inline(always)]
    fn full_off(&self, mx: i32, my: i32) -> usize {
        self.r.at(
            self.x0 as isize + mx as isize,
            self.y0 as isize + my as isize,
        )
    }

    /// Integer-pel diamond search seeded by the best of `cands` (quarter-pel); returns the
    /// full-pel vector and its cost.
    fn full_pel(&self, cands: &[[i16; 2]]) -> ([i32; 2], u32) {
        let int_min = [(self.bounds.min[0] + 3) >> 2, (self.bounds.min[1] + 3) >> 2];
        let int_max = [self.bounds.max[0] >> 2, self.bounds.max[1] >> 2];
        let cost_at = |m: [i32; 2]| {
            let off = self.full_off(m[0], m[1]);
            pixel_sad_16x16(self.src, 16, &self.r.buf[off..], self.r.stride)
                + self.mv_cost([m[0] * 4, m[1] * 4])
        };
        let mut best = [0i32; 2];
        let mut best_cost = u32::MAX;
        let mut seen = [[i32::MAX; 2]; 8];
        for (k, c) in cands.iter().enumerate().take(seen.len()) {
            let m = [
                ((c[0] as i32 + 2) >> 2).clamp(int_min[0], int_max[0]),
                ((c[1] as i32 + 2) >> 2).clamp(int_min[1], int_max[1]),
            ];
            if seen[..k].contains(&m) {
                continue;
            }
            seen[k] = m;
            let cost = cost_at(m);
            if cost < best_cost {
                best_cost = cost;
                best = m;
            }
        }
        for _ in 0..16 {
            let pts = [
                [best[0], best[1] - 1],
                [best[0] - 1, best[1]],
                [best[0] + 1, best[1]],
                [best[0], best[1] + 1],
            ];
            let inside = pts.iter().all(|p| {
                (int_min[0]..=int_max[0]).contains(&p[0])
                    && (int_min[1]..=int_max[1]).contains(&p[1])
            });
            let mut moved = None;
            if inside {
                let sads = pixel_sad_x4_16x16(
                    self.src,
                    &self.r.buf,
                    pts.map(|p| self.full_off(p[0], p[1])),
                    self.r.stride,
                );
                for (p, sad) in pts.iter().zip(sads) {
                    let cost = sad + self.mv_cost([p[0] * 4, p[1] * 4]);
                    if cost < best_cost {
                        best_cost = cost;
                        moved = Some(*p);
                    }
                }
            } else {
                for p in pts {
                    if (int_min[0]..=int_max[0]).contains(&p[0])
                        && (int_min[1]..=int_max[1]).contains(&p[1])
                    {
                        let cost = cost_at(p);
                        if cost < best_cost {
                            best_cost = cost;
                            moved = Some(p);
                        }
                    }
                }
            }
            match moved {
                Some(p) => best = p,
                None => break,
            }
        }
        (best, best_cost)
    }

    /// Half-pel then quarter-pel refinement around a full-pel vector; leaves the winning
    /// prediction in `pred` and returns the quarter-pel vector and its cost.
    ///
    /// The half-sample planes of a window around the full-pel position are interpolated once;
    /// every candidate within three quarter samples of it is then a copy or an average of two
    /// of the integer, `b`, `h` and `j` planes (`x264` `get_ref`).
    fn sub_pel(
        &self,
        full: [i32; 2],
        full_cost: u32,
        win: &mut HpelWindow,
        pred: &mut [u8; 256],
    ) -> ([i16; 2], u32) {
        let center = [full[0] * 4, full[1] * 4];
        let (bx, by) = (
            self.x0 as isize + full[0] as isize,
            self.y0 as isize + full[1] as isize,
        );
        hpel_window(&self.r.buf, self.r.at(bx - 1, by - 1), self.r.stride, win);
        let mut best = center;
        let mut best_cost = full_cost;
        let mut have_centre = false;
        for (step, iters) in [(2i32, 1), (1, 2)] {
            for _ in 0..iters {
                let from = best;
                for d in [[0, -step], [-step, 0], [step, 0], [0, step]] {
                    let mv = [from[0] + d[0], from[1] + d[1]];
                    if (mv[0] - center[0]).abs() > 3
                        || (mv[1] - center[1]).abs() > 3
                        || self.bounds.clamp(mv) != mv
                    {
                        continue;
                    }
                    // The centre plane is interpolated only once a candidate reads it.
                    if !have_centre
                        && (mv[0] & 3 == 2 && mv[1] & 1 == 1
                            || mv[1] & 3 == 2 && mv[0] & 1 == 1
                            || (mv[0] & 3, mv[1] & 3) == (2, 2))
                    {
                        hpel_window_centre(win);
                        have_centre = true;
                    }
                    let cost = match self.window_src(win, bx, by, mv) {
                        WinSrc::One(a, sa) => pixel_sad_16x16(self.src, 16, a, sa),
                        WinSrc::Avg(a, sa, b, sb) => pixel_sad_avg_16x16(self.src, a, sa, b, sb),
                    } + self.mv_cost(mv);
                    if cost < best_cost {
                        best_cost = cost;
                        best = mv;
                    }
                }
                if best == from {
                    break;
                }
            }
        }
        self.window_pred(win, bx, by, best, pred);
        ([best[0] as i16, best[1] as i16], best_cost)
    }

    /// The block or pair of blocks whose (rounded) average is the prediction at quarter-pel
    /// `mv` from the window interpolated around full-pel `(bx, by)`; `mv` lies within three
    /// quarter samples of that position.
    fn window_src<'a>(
        &'a self,
        win: &'a HpelWindow,
        bx: isize,
        by: isize,
        mv: [i32; 2],
    ) -> WinSrc<'a> {
        let (ix, iy) = (
            self.x0 as isize + (mv[0] >> 2) as isize,
            self.y0 as isize + (mv[1] >> 2) as isize,
        );
        let (wx, wy) = ((ix - bx + 1) as usize, (iy - by + 1) as usize);
        let r = self.r;
        // Integer samples displaced by (dx, dy), and the window planes displaced likewise.
        let g = |dx: isize, dy: isize| &r.buf[r.at(ix + dx, iy + dy)..];
        let w = |p: &'a [u8], dx: usize, dy: usize| &p[(wy + dy) * HPEL_WIN_W + wx + dx..];
        let (sg, sw) = (r.stride, HPEL_WIN_W);
        let (h, v, c) = (&win.h[..], &win.v[..], &win.c[..]);
        match (mv[0] & 3, mv[1] & 3) {
            (0, 0) => WinSrc::One(g(0, 0), sg),
            (2, 0) => WinSrc::One(w(h, 0, 0), sw),
            (0, 2) => WinSrc::One(w(v, 0, 0), sw),
            (2, 2) => WinSrc::One(w(c, 0, 0), sw),
            (1, 0) => WinSrc::Avg(g(0, 0), sg, w(h, 0, 0), sw),
            (3, 0) => WinSrc::Avg(w(h, 0, 0), sw, g(1, 0), sg),
            (0, 1) => WinSrc::Avg(g(0, 0), sg, w(v, 0, 0), sw),
            (0, 3) => WinSrc::Avg(w(v, 0, 0), sw, g(0, 1), sg),
            (1, 1) => WinSrc::Avg(w(h, 0, 0), sw, w(v, 0, 0), sw),
            (3, 1) => WinSrc::Avg(w(h, 0, 0), sw, w(v, 1, 0), sw),
            (1, 3) => WinSrc::Avg(w(v, 0, 0), sw, w(h, 0, 1), sw),
            (3, 3) => WinSrc::Avg(w(v, 1, 0), sw, w(h, 0, 1), sw),
            (2, 1) => WinSrc::Avg(w(h, 0, 0), sw, w(c, 0, 0), sw),
            (2, 3) => WinSrc::Avg(w(c, 0, 0), sw, w(h, 0, 1), sw),
            (1, 2) => WinSrc::Avg(w(v, 0, 0), sw, w(c, 0, 0), sw),
            _ => WinSrc::Avg(w(c, 0, 0), sw, w(v, 1, 0), sw),
        }
    }

    /// The prediction at quarter-pel `mv` from the window interpolated around full-pel
    /// `(bx, by)`; `mv` lies within three quarter samples of that position.
    fn window_pred(
        &self,
        win: &HpelWindow,
        bx: isize,
        by: isize,
        mv: [i32; 2],
        dst: &mut [u8; 256],
    ) {
        match self.window_src(win, bx, by, mv) {
            WinSrc::One(a, sa) => copy_rows::<16>(a, 0, sa, dst, 0, 16, 16),
            WinSrc::Avg(a, sa, b, sb) => pixel_avg_16x16(dst, a, sa, b, sb),
        }
    }
}

/// Source of a sub-pel prediction: one strided block, or two whose rounded average it is.
enum WinSrc<'a> {
    One(&'a [u8], usize),
    Avg(&'a [u8], usize, &'a [u8], usize),
}

/// Edge samples for intra $16\times16$ prediction: top row, left column and top-left corner.
fn i16_edges(p: &Plane, mb_x: usize, mb_y: usize) -> (Option<[u8; 16]>, Option<[u8; 16]>, u8) {
    let (x0, y0) = (mb_x * 16, mb_y * 16);
    let top = (mb_y > 0).then(|| {
        let o = p.idx(x0, y0 - 1);
        p.buf[o..o + 16].try_into().unwrap()
    });
    let left = (mb_x > 0).then(|| {
        let col = &p.buf[p.idx(x0 - 1, y0)..][..15 * p.stride + 1];
        std::array::from_fn(|r| col[r * p.stride])
    });
    let tl = if mb_x > 0 && mb_y > 0 {
        p.buf[p.idx(x0 - 1, y0 - 1)]
    } else {
        128
    };
    (top, left, tl)
}

/// Edge samples for intra $8\times8$ chroma prediction.
fn chroma_edges(p: &Plane, mb_x: usize, mb_y: usize) -> (Option<[u8; 8]>, Option<[u8; 8]>, u8) {
    let (x0, y0) = (mb_x * 8, mb_y * 8);
    let top = (mb_y > 0).then(|| {
        let o = p.idx(x0, y0 - 1);
        p.buf[o..o + 8].try_into().unwrap()
    });
    let left = (mb_x > 0).then(|| std::array::from_fn(|r| p.buf[p.idx(x0 - 1, y0 + r)]));
    let tl = if mb_x > 0 && mb_y > 0 {
        p.buf[p.idx(x0 - 1, y0 - 1)]
    } else {
        128
    };
    (top, left, tl)
}

/// Edge samples for intra $4\times4$ block `b` (decoding order) of the macroblock at column `mb_x`.
fn i4_edges(
    p: &Plane,
    px: usize,
    py: usize,
    b: usize,
    mb_x: usize,
    mb_w: usize,
) -> (Option<[u8; 8]>, Option<[u8; 4]>, u8) {
    let top = (py > 0).then(|| {
        let o = p.idx(px, py - 1);
        let mut t = [0u8; 8];
        t[..4].copy_from_slice(&p.buf[o..o + 4]);
        let tr_avail = !matches!(b, 3 | 7 | 11 | 13 | 15) && (b != 5 || mb_x + 1 < mb_w);
        if tr_avail {
            t[4..].copy_from_slice(&p.buf[o + 4..o + 8]);
        } else {
            let pad = t[3];
            t[4..].fill(pad);
        }
        t
    });
    let left = (px > 0).then(|| std::array::from_fn(|r| p.buf[p.idx(px - 1, py + r)]));
    let tl = if px > 0 && py > 0 {
        p.buf[p.idx(px - 1, py - 1)]
    } else {
        128
    };
    (top, left, tl)
}

/// Picks the intra chroma mode with the least SAD; returns it with both predictions.
fn analyse_chroma_intra(
    pic: &Pic,
    mb_x: usize,
    mb_y: usize,
    src: &MbSrc,
) -> (u8, [u8; 64], [u8; 64]) {
    let (tu, lu, tlu) = chroma_edges(&pic.u, mb_x, mb_y);
    let (tv, lv, tlv) = chroma_edges(&pic.v, mb_x, mb_y);
    let mut best = (u32::MAX, 0u8, [0u8; 64], [0u8; 64]);
    for mode in [0u8, 1, 2, 3] {
        let ok = match mode {
            1 => lu.is_some(),
            2 => tu.is_some(),
            3 => tu.is_some() && lu.is_some(),
            _ => true,
        };
        if !ok {
            continue;
        }
        let (mut pu, mut pv) = ([0u8; 64], [0u8; 64]);
        pred_chroma(mode, tu.as_ref(), lu.as_ref(), tlu, &mut pu);
        pred_chroma(mode, tv.as_ref(), lv.as_ref(), tlv, &mut pv);
        let cost = pixel_sad_8x8(&src.u, 8, &pu, 8) + pixel_sad_8x8(&src.v, 8, &pv, 8);
        // DC is the cheapest mode to signal.
        let cost = cost + if mode == 0 { 0 } else { 8 };
        if cost < best.0 {
            best = (cost, mode, pu, pv);
        }
    }
    (best.1, best.2, best.3)
}

/// Intra $16\times16$ prediction into a 16-stride block, as [`predict_intra_16x16`].
fn pred16(mode: u8, top: Option<&[u8; 16]>, left: Option<&[u8; 16]>, tl: u8, dst: &mut [u8; 256]) {
    if mode != 3 {
        predict_intra_16x16(mode, top, left, tl, dst, 16);
        return;
    }
    let (t, l) = (
        top.copied().unwrap_or([128; 16]),
        left.copied().unwrap_or([128; 16]),
    );
    let (mut h, mut v) = (0i32, 0i32);
    for i in 0..8 {
        let (pt, pl) = if i == 7 {
            (tl as i32, tl as i32)
        } else {
            (t[6 - i] as i32, l[6 - i] as i32)
        };
        h += (i as i32 + 1) * (t[8 + i] as i32 - pt);
        v += (i as i32 + 1) * (l[8 + i] as i32 - pl);
    }
    let a = 16 * (t[15] as i32 + l[15] as i32);
    intra_pred_plane::<16>(a, (5 * h + 32) >> 6, (5 * v + 32) >> 6, dst);
}

/// Intra $8\times8$ chroma prediction into an 8-stride block, as [`predict_intra_chroma_8x8`].
fn pred_chroma(
    mode: u8,
    top: Option<&[u8; 8]>,
    left: Option<&[u8; 8]>,
    tl: u8,
    dst: &mut [u8; 64],
) {
    if mode != 3 {
        predict_intra_chroma_8x8(mode, top, left, tl, dst, 8);
        return;
    }
    let (t, l) = (
        top.copied().unwrap_or([128; 8]),
        left.copied().unwrap_or([128; 8]),
    );
    let (mut h, mut v) = (0i32, 0i32);
    for i in 0..4 {
        let (pt, pl) = if i == 3 {
            (tl as i32, tl as i32)
        } else {
            (t[2 - i] as i32, l[2 - i] as i32)
        };
        h += (i as i32 + 1) * (t[4 + i] as i32 - pt);
        v += (i as i32 + 1) * (l[4 + i] as i32 - pl);
    }
    let a = 16 * (t[7] as i32 + l[7] as i32);
    intra_pred_plane::<8>(a, (17 * h + 16) >> 5, (17 * v + 16) >> 5, dst);
}

/// Picks the intra $16\times16$ mode with the least SATD; returns that SATD and the mode. The
/// vertical, horizontal and DC costs come from one transform of the source
/// ([`intra_satd_x3_16x16`]); ties keep DC, then vertical, then horizontal, then plane.
fn analyse_i16(pic: &Pic, mb_x: usize, mb_y: usize, src: &MbSrc) -> (u32, u8) {
    let (top, left, tl) = i16_edges(&pic.y, mb_x, mb_y);
    let sum = |e: &[u8; 16]| e.iter().map(|&v| v as u32).sum::<u32>();
    let dc = match (&top, &left) {
        (Some(t), Some(l)) => ((sum(t) + sum(l) + 16) >> 5) as u8,
        (Some(e), None) | (None, Some(e)) => ((sum(e) + 8) >> 4) as u8,
        (None, None) => 128,
    };
    let [v, h, d] = intra_satd_x3_16x16(
        &src.y,
        top.as_ref().unwrap_or(&[0; 16]),
        left.as_ref().unwrap_or(&[0; 16]),
        dc,
    );
    let mut best = (d, 2u8);
    if top.is_some() && v < best.0 {
        best = (v, 0);
    }
    if left.is_some() && h < best.0 {
        best = (h, 1);
    }
    if top.is_some() && left.is_some() {
        let mut pred = [0u8; 256];
        pred16(3, top.as_ref(), left.as_ref(), tl, &mut pred);
        let cost = pixel_satd_16x16(&src.y, 16, &pred, 16);
        if cost < best.0 {
            best = (cost, 3);
        }
    }
    best
}

/// SATDs, as [`pixel_satd_4x4`], of a block with Hadamard transform `had` (from
/// [`hadamard4x4_16x4`]) against its vertical, horizontal and DC intra $4\times4$ predictions;
/// unavailable edges leave their mode's cost meaningless.
fn intra_satd_x3_4x4(had: &[i16; 16], top: Option<&[u8; 8]>, left: Option<&[u8; 4]>) -> [u32; 3] {
    let h1 = |x: [i32; 4]| {
        let (s01, d01, s23, d23) = (x[0] + x[1], x[0] - x[1], x[2] + x[3], x[2] - x[3]);
        [s01 + s23, s01 - s23, d01 + d23, d01 - d23].map(|c| 4 * c)
    };
    let c = |i: usize| had[i] as i32;
    let total: i32 = had.iter().map(|&v| (v as i32).abs()).sum();
    let sum = |e: &[u8]| e.iter().take(4).map(|&v| v as i32).sum::<i32>();
    let dc = match (top, left) {
        (Some(t), Some(l)) => (sum(t) + sum(l) + 4) >> 3,
        (Some(t), None) => (sum(t) + 2) >> 2,
        (None, Some(l)) => (sum(l) + 2) >> 2,
        (None, None) => 128,
    };
    let sd = total - c(0).abs() + (c(0) - 16 * dc).abs();
    let sv = top.map_or(0, |t| {
        let p = h1([t[0], t[1], t[2], t[3]].map(i32::from));
        total
            + (0..4)
                .map(|h| (c(h) - p[h]).abs() - c(h).abs())
                .sum::<i32>()
    });
    let sh = left.map_or(0, |l| {
        let p = h1(l.map(i32::from));
        total
            + (0..4)
                .map(|v| (c(4 * v) - p[v]).abs() - c(4 * v).abs())
                .sum::<i32>()
    });
    [sv, sh, sd].map(|x| (x >> 1) as u32)
}

/// `quantize_block4x4` with a precomputed quantizer; also reports whether a level is non-zero.
#[inline]
fn quant_one(c: &[i16; 16], q: &Quant4x4) -> ([i16; 16], bool) {
    let mut out = [0i16; 16];
    for i in 0..16 {
        let l = ((c[i].unsigned_abs() as u32 * q.mf[i] as u32 + q.bias) >> q.shift) as i16;
        out[i] = if c[i] < 0 { -l } else { l };
    }
    (out, out.iter().any(|&l| l != 0))
}

/// Residual of an inter-predicted macroblock after quantization and decimation.
struct InterResidual {
    luma_q4: [[i16; 16]; 16],
    luma_q8: [[i16; 64]; 4],
    use_8x8: bool,
    cbp_luma: u8,
    chroma: ChromaLevels,
    cbp_chroma: u8,
}

/// Raster index (`4 by + bx`) of each $4\times4$ luma block in decoding order; the mapping is its
/// own inverse.
const Z_TO_RASTER: [usize; 16] = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];

/// The $4\times4$ quantizers of `quantize_block4x4` per QP, inter (`f = 2^qbits / 6`) then intra
/// (`f = 2^qbits / 3`).
static QUANT4: [[Quant4x4; 2]; 52] = {
    let mut t = [[Quant4x4 {
        mf: [0; 16],
        bias: 0,
        shift: 0,
    }; 2]; 52];
    let mut qp = 0;
    while qp < 52 {
        let shift = 15 + (qp / 6) as u32;
        let mut mf = [0u16; 16];
        let mut i = 0;
        while i < 16 {
            mf[i] = QUANT4_MF[qp % 6][coeff_pos_type(i)] as u16;
            i += 1;
        }
        t[qp][0] = Quant4x4 {
            mf,
            bias: (1 << shift) / 6,
            shift,
        };
        t[qp][1] = Quant4x4 {
            mf,
            bias: (1 << shift) / 3,
            shift,
        };
        qp += 1;
    }
    t
};

/// `LevelScale4x4 << (qp / 6)` per QP and raster coefficient position (`dequantize_block4x4`).
static DEQUANT4: [[i32; 16]; 52] = {
    let mut t = [[0i32; 16]; 52];
    let mut qp = 0;
    while qp < 52 {
        let mut i = 0;
        while i < 16 {
            t[qp][i] = DEQUANT4_COEFF_INIT[qp % 6][coeff_pos_type(i)] << (qp / 6);
            i += 1;
        }
        qp += 1;
    }
    t
};

/// $4\times4$ quantizer of `quantize_block4x4` at `qp`.
#[inline(always)]
fn quant4(qp: u8, intra: bool) -> &'static Quant4x4 {
    &QUANT4[qp.min(51) as usize][intra as usize]
}

/// Largest SAD of a $4\times4$ residual whose coefficients all quantize to zero under `q`: the
/// core transform weights a residual sample by at most `w_v w_h`, `w = [1, 2, 1, 2]`.
fn zero_sad_bound(q: &Quant4x4) -> u32 {
    const W: [u32; 4] = [1, 2, 1, 2];
    let room = (1u32 << q.shift) - q.bias - 1;
    (0..16)
        .map(|i| room / (q.mf[i] as u32 * W[i >> 2] * W[i & 3]))
        .min()
        .unwrap_or(0)
}

/// Forward transforms of the sixteen $4\times4$ residual blocks of a macroblock, decoding order.
fn dct_luma(src: &[u8; 256], pred: &[u8; 256], out: &mut [[i16; 16]; 16]) {
    let mut strip = [[0i16; 16]; 4];
    for s in 0..4 {
        sub_dct4x4_16x4(&src[s * 64..], 16, &pred[s * 64..], 16, &mut strip);
        for (bx, blk) in strip.iter().enumerate() {
            out[Z_TO_RASTER[s * 4 + bx]] = *blk;
        }
    }
}

#[inline(always)]
fn group4(blocks: &mut [[i16; 16]; 16], i8x8: usize) -> &mut [[i16; 16]; 4] {
    (&mut blocks[i8x8 * 4..i8x8 * 4 + 4]).try_into().unwrap()
}

/// Quantizes the luma residual with $4\times4$ transforms and decimates nearly empty $8\times8$
/// blocks; returns the levels and the coded block pattern.
fn quant_luma_4x4_inter(src: &[u8; 256], pred: &[u8; 256], qp: u8) -> ([[i16; 16]; 16], u8) {
    let mut q4 = [[0i16; 16]; 16];
    dct_luma(src, pred, &mut q4);
    let q = quant4(qp, false);
    let mut cbp = 0u8;
    let mut total = 0u32;
    for i8x8 in 0..4 {
        let group = group4(&mut q4, i8x8);
        let nz = quant_4x4_x4(group, q);
        if nz == 0 {
            continue;
        }
        let zz = zigzag_4x4_x4(group);
        let masks = nonzero_mask_4x4_x4(&zz);
        let score: u32 = (0..4).map(|b| decimate_masked(&zz[b], masks[b])).sum();
        if score < 4 {
            *group = [[0; 16]; 4];
        } else {
            cbp |= 1 << i8x8;
        }
        total += score;
    }
    if total < 6 && cbp != 0 {
        q4 = [[0; 16]; 16];
        cbp = 0;
    }
    (q4, cbp)
}

/// Transformed and quantized chroma residual of a macroblock.
struct ChromaLevels {
    /// The $2\times2$ DC levels of each plane.
    dc: [[i16; 4]; 2],
    /// The zig-zag scanned AC levels of each $4\times4$ block.
    ac: [[[i16; 15]; 4]; 2],
    /// Masks of the non-zero AC levels (bit `i` for `ac[..][..][i]`).
    ac_masks: [[u32; 4]; 2],
    /// The AC levels in raster order; the DC slot holds no level.
    raster: [[[i16; 16]; 4]; 2],
}

impl ChromaLevels {
    const ZERO: Self = Self {
        dc: [[0; 4]; 2],
        ac: [[[0; 15]; 4]; 2],
        ac_masks: [[0; 4]; 2],
        raster: [[[0; 16]; 4]; 2],
    };

    /// `coded_block_pattern` chroma: 2 with AC levels, 1 with DC levels only, else 0.
    fn cbp(&self) -> u8 {
        if self.ac_masks.iter().flatten().any(|&m| m != 0) {
            2
        } else if self.dc.iter().flatten().any(|&c| c != 0) {
            1
        } else {
            0
        }
    }

    /// The decimation score of plane `p`'s AC levels.
    fn ac_score(&self, p: usize) -> u32 {
        (0..4)
            .map(|b| decimate_masked(&self.ac[p][b], self.ac_masks[p][b]))
            .sum()
    }
}

/// Transforms and quantizes the chroma residual of both planes.
fn quant_chroma(src: &MbSrc, pu: &[u8; 64], pv: &[u8; 64], qpc: u8, intra: bool) -> ChromaLevels {
    let q = quant4(qpc, intra);
    let mut out = ChromaLevels::ZERO;
    for (p, (s, pred)) in [(&src.u, pu), (&src.v, pv)].into_iter().enumerate() {
        let mut c = [[0i16; 16]; 4];
        sub_dct4x4_8x8(s, 8, pred, 8, &mut c);
        let dc_raw = [c[0][0], c[1][0], c[2][0], c[3][0]];
        if quant_4x4_x4(&mut c, q) != 0 {
            let zz = zigzag_4x4_x4(&c);
            // Bit 0 is the block's DC slot, which the 2x2 DC transform codes instead.
            out.ac_masks[p] = nonzero_mask_4x4_x4(&zz).map(|m| m >> 1);
            for (a, z) in out.ac[p].iter_mut().zip(&zz) {
                a.copy_from_slice(&z[1..]);
            }
            out.raster[p] = c;
        }
        out.dc[p] = quantize_chroma_dc2x2(&dc_raw, qpc);
    }
    out
}

#[inline(always)]
fn dequant4(qp: u8) -> &'static [i32; 16] {
    &DEQUANT4[qp.min(51) as usize]
}

/// Adds the dequantized $4\times4$ luma residual of the coded $8\times8$ quadrants to `pred`.
fn recon_luma_4x4(pred: &mut [u8; 256], q4: &[[i16; 16]; 16], cbp: u8, qp: u8) {
    let scale = dequant4(qp);
    for i8x8 in 0..4 {
        if cbp & (1 << i8x8) != 0 {
            let mut group: [[i16; 16]; 4] = q4[i8x8 * 4..i8x8 * 4 + 4].try_into().unwrap();
            dequant_4x4_x4(&mut group, scale);
            let o = (i8x8 >> 1) * 128 + (i8x8 & 1) * 8;
            add_idct4x4_8x8(&mut pred[o..], 16, &group);
        }
    }
}

/// Adds the dequantized chroma residual (DC and, for `cbp_chroma == 2`, AC) to the predictions.
fn recon_chroma(
    pu: &mut [u8; 64],
    pv: &mut [u8; 64],
    chroma: &ChromaLevels,
    cbp_chroma: u8,
    qpc: u8,
) {
    if cbp_chroma == 0 {
        return;
    }
    let scale = dequant4(qpc);
    for (p, pred) in [pu, pv].into_iter().enumerate() {
        let dc_recon = dequantize_idct_chroma_dc2x2(&chroma.dc[p], qpc);
        let mut blocks = [[0i16; 16]; 4];
        if cbp_chroma >= 2 {
            blocks = chroma.raster[p];
            dequant_4x4_x4(&mut blocks, scale);
        }
        for (blk, &d) in blocks.iter_mut().zip(&dc_recon) {
            blk[0] = d;
        }
        add_idct4x4_8x8(pred, 8, &blocks);
    }
}

/// $8\times8$-transform counterpart of [`quant_luma_4x4_inter`].
fn quant_luma_8x8_inter(src: &[u8; 256], pred: &[u8; 256], qp: u8) -> ([[i16; 64]; 4], u8, u32) {
    let mut q8 = [[0i16; 64]; 4];
    let mut cbp = 0u8;
    let mut total = 0u32;
    for (i8x8, slot) in q8.iter_mut().enumerate() {
        let o = (i8x8 >> 1) * 128 + (i8x8 & 1) * 8;
        let dct = sub8x8_dct_u8(&src[o..], 16, &pred[o..], 16);
        let q = quantize_block8x8(&dct, qp, false);
        if q.iter().any(|&c| c != 0) {
            let score = decimate_score(&zigzag64(&q));
            total += score;
            if score >= 4 {
                cbp |= 1 << i8x8;
                *slot = q;
            }
        }
    }
    if total < 6 && cbp != 0 {
        q8 = [[0; 64]; 4];
        cbp = 0;
    }
    (q8, cbp, total)
}

/// Quantizes the chroma residual of an inter macroblock, dropping a plane's AC levels when they
/// are nearly empty.
fn quant_chroma_inter(src: &MbSrc, pu: &[u8; 64], pv: &[u8; 64], qpc: u8) -> ChromaLevels {
    let mut c = quant_chroma(src, pu, pv, qpc, false);
    for p in 0..2 {
        if c.ac_masks[p] != [0; 4] && c.ac_score(p) < 7 {
            c.ac[p] = [[0; 15]; 4];
            c.ac_masks[p] = [0; 4];
            c.raster[p] = [[0; 16]; 4];
        }
    }
    c
}

/// SAD thresholds of the skip decision at one QP.
struct SkipBounds {
    qp: u8,
    /// Luma SAD below which the skip residual certainly quantizes to nothing.
    luma: u32,
    /// Luma SSD below which the same holds.
    luma_ssd: u32,
    /// The SAD bound for each chroma plane.
    chroma: u32,
    /// The SSD bound for each chroma plane.
    chroma_ssd: u32,
    /// Luma SAD from which probing is skipped: some $4\times4$ block then holds more residual
    /// than a zero DC level allows, were its residual all of one sign.
    probe: u32,
}

/// Largest SSD of a $4\times4$ residual whose coefficients all quantize to zero under `q`: by
/// Cauchy-Schwarz a coefficient is at most `|w| sqrt(SSD)`, where the transform's basis rows
/// have squared norms `[4, 10, 4, 10]`.
fn zero_ssd_bound(q: &Quant4x4) -> u32 {
    const N: [u32; 4] = [4, 10, 4, 10];
    let room = (1u32 << q.shift) - q.bias - 1;
    (0..16)
        .map(|i| {
            let lim = room / q.mf[i] as u32;
            lim * lim / (N[i >> 2] * N[i & 3])
        })
        .min()
        .unwrap_or(0)
}

impl SkipBounds {
    fn new(qp: u8, qpc: u8) -> Self {
        let q = quant4(qp, false);
        let qc = quant4(qpc, false);
        // The chroma DC Hadamard sums the plane's residual, at most 8 sqrt(SSD) in magnitude; it
        // quantizes with qbits + 1 and an intra-style f = 2^qbits / 3 (`quantize_chroma_dc2x2`).
        let dc_shift = 16 + (qpc as u32 / 6);
        let dc_room = (1u32 << dc_shift) - (1u32 << dc_shift) / 3 - 1;
        let dc = dc_room / QUANT4_MF[qpc as usize % 6][0] as u32;
        Self {
            qp,
            luma: zero_sad_bound(q),
            luma_ssd: zero_ssd_bound(q),
            chroma: zero_sad_bound(qc).min(dc),
            chroma_ssd: zero_ssd_bound(qc).min(dc * dc / 64),
            probe: 16 * (((1u32 << q.shift) - q.bias) / q.mf[0] as u32 + 1),
        }
    }

    /// Whether a residual of this SAD and lazily computed SSD certainly quantizes to nothing.
    #[inline(always)]
    fn certain(sad: u32, sad_bound: u32, ssd: impl FnOnce() -> u32, ssd_bound: u32) -> bool {
        sad <= sad_bound || ssd() <= ssd_bound
    }
}

/// `x264_macroblock_probe_pskip`, luma part: whether the luma residual against the skip
/// prediction would be decimated to nothing.
fn probe_skip_luma(src: &MbSrc, py: &[u8; 256], qp: u8) -> bool {
    let mut c = [[0i16; 16]; 16];
    dct_luma(&src.y, py, &mut c);
    let q = quant4(qp, false);
    let mut total = 0u32;
    for i8x8 in 0..4 {
        let group = group4(&mut c, i8x8);
        let nz = quant_4x4_x4(group, q);
        if nz == 0 {
            continue;
        }
        let zz = zigzag_4x4_x4(group);
        for (z, &mask) in zz.iter().zip(&nonzero_mask_4x4_x4(&zz)) {
            total += decimate_masked(z, mask);
            if total >= 6 {
                return false;
            }
        }
    }
    true
}

/// `x264_macroblock_probe_pskip`, chroma part: no DC level, and AC levels decimated per plane.
fn probe_skip_chroma(src: &MbSrc, pu: &[u8; 64], pv: &[u8; 64], qpc: u8) -> bool {
    let c = quant_chroma(src, pu, pv, qpc, false);
    if c.dc.iter().flatten().any(|&l| l != 0) {
        return false;
    }
    (0..2).all(|p| c.ac_score(p) < 7)
}

impl H264Encoder {
    /// Codes one frame; see [`H264Encoder::encode_frame`].
    pub(super) fn encode_frame_impl(&mut self, frame: &VideoFrame) -> Result<Vec<u8>> {
        if frame.format != PixelFormat::Yuv420p {
            return Err(RagdollError::Unsupported(format!(
                "H264Encoder requires Yuv420p input, got {}",
                frame.format
            )));
        }
        let mb_w = self.sps.pic_width_in_mbs as usize;
        let mb_h = self.sps.pic_height_in_map_units as usize;
        let total_mbs = mb_w * mb_h;
        let is_idr = self.frame_count.is_multiple_of(self.keyint) || !self.state.has_ref;
        let nal_type = if is_idr {
            NalUnitType::IdrSlice
        } else {
            NalUnitType::NonIdrSlice
        };
        let nal_header = NalHeader {
            nal_ref_idc: 3,
            nal_unit_type: nal_type,
        };
        let fps = self.sps.fps.unwrap_or(Rational64::FPS_30).as_f64().max(1.0);

        // Per-MB spatial variance and its geometric mean drive CRF variance-adaptive quantization.
        let mut avg_log_var = 0.0f32;
        let mut mb_vars = std::mem::take(&mut self.state.mb_vars);
        mb_vars.clear();
        if matches!(self.rate_control, RateControlMode::Crf { .. }) {
            let mut sum_log_var = 0.0f32;
            let mut src = MbSrc {
                y: [0; 256],
                u: [0; 64],
                v: [0; 64],
            };
            for mb_y in 0..mb_h {
                for mb_x in 0..mb_w {
                    src.load(frame, mb_x, mb_y);
                    let v = compute_mb_luma_variance(&src.y);
                    mb_vars.push(v);
                    sum_log_var += v.max(1.0).log2();
                }
            }
            avg_log_var = sum_log_var / (total_mbs.max(1) as f32);
        }

        let slice_qp = self.frame_slice_qp(is_idr, total_mbs, fps);
        let gop_frame_idx = self.frame_count % self.keyint.max(1);
        let slice_qp_delta = (slice_qp as i32) - (26 + (self.pps.pic_init_qp_minus26 as i32));
        let slice_header = SliceHeader {
            first_mb_in_slice: 0,
            slice_type: if is_idr { 7 } else { 5 },
            pict_type: if is_idr {
                PictureType::I
            } else {
                PictureType::P
            },
            pps_id: 0,
            frame_num: (gop_frame_idx & ((1 << self.sps.log2_max_frame_num) - 1)) as u16,
            idr_pic_id: if is_idr {
                Some((self.frame_count / self.keyint.max(1)) as u16)
            } else {
                None
            },
            pic_order_cnt_lsb: ((gop_frame_idx * 2)
                & ((1 << self.sps.log2_max_pic_order_cnt_lsb) - 1))
                as u16,
            direct_spatial_mv_pred_flag: true,
            num_ref_idx_l0_active_minus1: 0,
            num_ref_idx_l1_active_minus1: 0,
            cabac_init_idc: 0,
            slice_qp_delta: slice_qp_delta.clamp(-51, 51) as i8,
            slice_qp,
            disable_deblocking_filter_idc: if slice_qp == 0 { 1 } else { 0 },
            slice_alpha_c0_offset_div2: 0,
            slice_beta_offset_div2: 0,
        };

        let mut hw = BitWriter::with_capacity(64);
        slice_header.write(&mut hw, nal_header, &self.sps, &self.pps);
        let mut w = Bits::new(std::mem::take(&mut self.state.bits_buf));
        let hdr_bits = hw.bits_written();
        w.append(&hw.finish(), hdr_bits);
        let use_cabac = self.pps.entropy_coding_mode_flag;
        let mut cabac: Cabac = if use_cabac {
            w.align_with_ones();
            Some((
                CabacEncoder::new(),
                init_cabac_contexts(slice_qp, is_idr, slice_header.cabac_init_idc),
            ))
        } else {
            None
        };

        self.state.prepare(mb_w, mb_h, slice_qp, use_cabac);
        let mut sl = SliceState {
            is_idr,
            use_cabac,
            t8x8: self.pps.transform_8x8_mode_flag,
            chroma_qp_offset: self.pps.chroma_qp_index_offset as i32,
            skip_run: 0,
            cur_qp: slice_qp as i32,
            prev_qp_delta_nz: false,
            skip_bounds: SkipBounds::new(
                slice_qp,
                chroma_qp(slice_qp as i32, self.pps.chroma_qp_index_offset as i32),
            ),
        };
        let mut src = MbSrc {
            y: [0; 256],
            u: [0; 64],
            v: [0; 64],
        };

        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let mb_idx = mb_y * mb_w + mb_x;
                let qp = match self.rate_control {
                    RateControlMode::FixedQp(_) => slice_qp,
                    RateControlMode::Crf { .. } => {
                        let log_var = mb_vars[mb_idx].max(1.0).log2();
                        // Auto-variance AQ around the geometric mean variance (`x264` log-domain
                        // `ac_energy`): flat regions get a negative QP offset, masked textures a
                        // positive one.
                        let delta = ((log_var - avg_log_var) * 2.5).round().clamp(-8.0, 8.0) as i32;
                        ((slice_qp as i32) + delta).clamp(1, 51) as u8
                    }
                    RateControlMode::Abr { bitrate_kbps, .. } => {
                        let target_frame_bits = (bitrate_kbps as f64) * 1000.0 / fps;
                        let frac_done = (mb_idx as f64) / (total_mbs.max(1) as f64);
                        if frac_done > 0.02 {
                            let cur_bits = match &cabac {
                                Some((ce, _)) => (ce.bytes_written() * 8) as f64,
                                None => w.bits_written() as f64,
                            };
                            let rem_budget = ((target_frame_bits
                                * ((self.frame_count + 1) as f64))
                                - (self.total_bits_emitted as f64))
                                .max(target_frame_bits * 0.25);
                            let expected = rem_budget * frac_done;
                            let row_adj =
                                (((cur_bits / expected.max(64.0)).clamp(0.125, 8.0)).log2() * 6.0)
                                    .round() as i32;
                            ((slice_qp as i32) + row_adj).clamp(1, 51) as u8
                        } else {
                            slice_qp
                        }
                    }
                };
                src.load(frame, mb_x, mb_y);
                let st = &mut self.state;
                if qp == 0 {
                    st.code_pcm(&mut sl, &mut w, &mut cabac, mb_x, mb_y, &src);
                } else if is_idr {
                    st.code_intra(&mut sl, &mut w, &mut cabac, mb_x, mb_y, qp, &src);
                } else {
                    st.code_p_mb(&mut sl, &mut w, &mut cabac, mb_x, mb_y, qp, &src);
                }
                if let Some((ce, _)) = &mut cabac {
                    ce.encode_terminate(u8::from(mb_idx + 1 == total_mbs));
                }
            }
        }
        self.state.mb_vars = mb_vars;

        let slice_rbsp = if let Some((ce, _)) = cabac {
            let mut bytes = w.finish();
            bytes.extend_from_slice(&ce.finish());
            bytes
        } else {
            if !is_idr && sl.skip_run > 0 {
                w.write_ue(sl.skip_run as u64);
            }
            w.write_rbsp_trailing_bits();
            w.finish()
        };

        if slice_header.disable_deblocking_filter_idc != 1 {
            self.state.deblock(sl.chroma_qp_offset);
        }
        let st = &mut self.state;
        st.cur.y.extend_borders();
        st.cur.u.extend_borders();
        st.cur.v.extend_borders();
        std::mem::swap(&mut st.cur, &mut st.refp);
        st.has_ref = true;
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                st.prev_mb_mv[mb_y * mb_w + mb_x] = st.mv_grid[(mb_y * 4) * mb_w * 4 + mb_x * 4];
            }
        }

        let mut out = if is_idr {
            self.headers_annex_b()
        } else {
            Vec::new()
        };
        append_annex_b_nal(&mut out, nal_header, &slice_rbsp);
        self.state.bits_buf = slice_rbsp;
        self.finish_frame_rate_control(&mut out, fps);
        Ok(out)
    }
}

/// Appends the Annex-B NAL unit of `rbsp` to `out`, inserting emulation prevention bytes exactly
/// as [`encode_annex_b_nal`] does. Eight bytes at a time are skipped while no zero byte can
/// start an escape.
fn append_annex_b_nal(out: &mut Vec<u8>, header: NalHeader, rbsp: &[u8]) {
    const ONES: u64 = 0x0101_0101_0101_0101;
    const HIGHS: u64 = 0x8080_8080_8080_8080;
    out.reserve(rbsp.len() + rbsp.len() / 64 + 8);
    out.extend_from_slice(&[0, 0, 0, 1, header.to_byte()]);
    let (mut zeros, mut copied, mut i) = (0u32, 0usize, 0usize);
    while i < rbsp.len() {
        if zeros == 0
            && let Some(word) = rbsp.get(i..i + 8)
        {
            let x = u64::from_le_bytes(word.try_into().unwrap());
            if x.wrapping_sub(ONES) & !x & HIGHS == 0 {
                i += 8;
                continue;
            }
        }
        let b = rbsp[i];
        if zeros == 2 && b <= 3 {
            out.extend_from_slice(&rbsp[copied..i]);
            out.push(3);
            copied = i;
            zeros = 0;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        i += 1;
    }
    out.extend_from_slice(&rbsp[copied..]);
}

/// MSB-first bit writer for slice data (`x264` `bs_t`): bits gather in a 64-bit word that is
/// flushed 32 bits at a time into a buffer reused across frames.
struct Bits {
    out: Vec<u8>,
    acc: u64,
    n: u32,
}

impl Bits {
    fn new(mut buf: Vec<u8>) -> Self {
        buf.clear();
        Self {
            out: buf,
            acc: 0,
            n: 0,
        }
    }

    /// Appends the first `bits` bits of `bytes`.
    fn append(&mut self, bytes: &[u8], bits: usize) {
        for (i, &b) in bytes.iter().enumerate().take(bits.div_ceil(8)) {
            let len = (bits - i * 8).min(8) as u32;
            self.put32((b >> (8 - len)) as u32, len);
        }
    }

    /// Writes the low `len` bits of `val`, `len <= 32`.
    #[inline(always)]
    fn put32(&mut self, val: u32, len: u32) {
        self.acc = (self.acc << len) | val as u64;
        self.n += len;
        if self.n >= 32 {
            self.n -= 32;
            self.out
                .extend_from_slice(&((self.acc >> self.n) as u32).to_be_bytes());
        }
    }

    #[inline(always)]
    fn write_bits(&mut self, val: u64, len: u32) {
        if len > 32 {
            self.put32((val >> 32) as u32 & (u32::MAX >> (64 - len)), len - 32);
            self.put32(val as u32, 32);
        } else if len > 0 {
            self.put32(val as u32 & (u32::MAX >> (32 - len)), len);
        }
    }

    #[inline(always)]
    fn write_bit(&mut self, bit: bool) {
        self.put32(bit as u32, 1);
    }

    #[inline(always)]
    fn write_ue(&mut self, val: u64) {
        let code = val + 1;
        let len = 64 - code.leading_zeros();
        self.write_bits(0, len - 1);
        self.write_bits(code, len);
    }

    #[inline(always)]
    fn write_se(&mut self, val: i64) {
        self.write_ue(if val > 0 {
            (2 * val - 1) as u64
        } else {
            (-2 * val) as u64
        });
    }

    fn bits_written(&self) -> usize {
        self.out.len() * 8 + self.n as usize
    }

    fn align_to_byte(&mut self) {
        let rem = self.n & 7;
        if rem != 0 {
            self.put32(0, 8 - rem);
        }
    }

    /// `cabac_alignment_one_bit`s up to the next byte boundary.
    fn align_with_ones(&mut self) {
        let rem = self.n & 7;
        if rem != 0 {
            self.put32((1 << (8 - rem)) - 1, 8 - rem);
        }
    }

    fn write_rbsp_trailing_bits(&mut self) {
        self.write_bit(true);
        self.align_to_byte();
    }

    fn finish(mut self) -> Vec<u8> {
        self.align_to_byte();
        while self.n >= 8 {
            self.n -= 8;
            self.out.push((self.acc >> self.n) as u8);
        }
        self.out
    }
}

impl EncState {
    /// Writes `mb_skip_run` / `mb_skip_flag = 0` ahead of a coded macroblock of a P slice.
    fn begin_coded_mb(
        &self,
        sl: &mut SliceState,
        w: &mut Bits,
        cabac: &mut Cabac,
        mb_x: usize,
        mb_y: usize,
    ) {
        if sl.is_idr {
            return;
        }
        match cabac {
            Some((ce, ctxs)) => {
                let mb_idx = mb_y * self.mb_w + mb_x;
                let l = u8::from(mb_x > 0 && !self.mb_is_skip[mb_idx - 1]);
                let t = u8::from(mb_y > 0 && !self.mb_is_skip[mb_idx - self.mb_w]);
                ce.encode_p_skip(ctxs, l, t, false);
            }
            None => {
                w.write_ue(sl.skip_run as u64);
                sl.skip_run = 0;
            }
        }
    }

    /// Lossless `I_PCM` macroblock (QP 0).
    fn code_pcm(
        &mut self,
        sl: &mut SliceState,
        w: &mut Bits,
        cabac: &mut Cabac,
        mb_x: usize,
        mb_y: usize,
        src: &MbSrc,
    ) {
        let mb_w = self.mb_w;
        let mb_idx = mb_y * mb_w + mb_x;
        self.begin_coded_mb(sl, w, cabac, mb_x, mb_y);
        match cabac {
            Some((ce, ctxs)) => {
                if !sl.is_idr {
                    ce.encode_p_mb_type(ctxs, 30); // I_PCM in a P slice (5 + 25)
                } else {
                    let l = u8::from(mb_x > 0 && self.mb_is_i16x16[mb_idx - 1]);
                    let t = u8::from(mb_y > 0 && self.mb_is_i16x16[mb_idx - mb_w]);
                    ce.encode_intra_mb_type(ctxs, 3, l, t, 25);
                }
                for &b in src.y.iter().chain(&src.u).chain(&src.v) {
                    ce.write_raw_byte(b);
                }
                ce.reinit_after_pcm();
            }
            None => {
                w.write_ue(if sl.is_idr { 25 } else { 30 });
                w.align_to_byte();
                for &b in src.y.iter().chain(&src.u).chain(&src.v) {
                    w.write_bits(b as u64, 8);
                }
            }
        }
        let g_stride = mb_w * 4;
        fill_grid_4x4(&mut self.luma_nnz, g_stride, mb_x * 4, mb_y * 4, 4, 4, 16);
        self.set_intra_motion(mb_x, mb_y);
        for c in 0..2 {
            fill_grid_4x4(
                &mut self.chroma_nnz[c],
                mb_w * 2,
                mb_x * 2,
                mb_y * 2,
                2,
                2,
                16,
            );
            self.mb_chroma_dc_cbf[c][mb_idx] = true;
        }
        self.mb_is_intra[mb_idx] = true;
        self.mb_is_i16x16[mb_idx] = true;
        self.mb_cbp_luma[mb_idx] = 15;
        self.mb_cbp_chroma[mb_idx] = 2;
        self.mb_luma_dc_cbf[mb_idx] = true;
        self.mb_qps[mb_idx] = sl.cur_qp as u8;
        sl.prev_qp_delta_nz = false;
        self.cur.store_mb(mb_x, mb_y, &src.y, &src.u, &src.v);
    }

    /// Signals a `P_Skip` macroblock and records its state; the caller stores its prediction.
    fn code_skip(
        &mut self,
        sl: &mut SliceState,
        cabac: &mut Cabac,
        mb_x: usize,
        mb_y: usize,
        mv: [i16; 2],
    ) {
        let mb_w = self.mb_w;
        let mb_idx = mb_y * mb_w + mb_x;
        match cabac {
            Some((ce, ctxs)) => {
                let l = u8::from(mb_x > 0 && !self.mb_is_skip[mb_idx - 1]);
                let t = u8::from(mb_y > 0 && !self.mb_is_skip[mb_idx - mb_w]);
                ce.encode_p_skip(ctxs, l, t, true);
            }
            None => sl.skip_run += 1,
        }
        let g_stride = mb_w * 4;
        fill_grid_4x4(&mut self.ref_grid, g_stride, mb_x * 4, mb_y * 4, 4, 4, 0);
        fill_grid_4x4(&mut self.mv_grid, g_stride, mb_x * 4, mb_y * 4, 4, 4, mv);
        self.mb_is_skip[mb_idx] = true;
        self.mb_qps[mb_idx] = sl.cur_qp as u8;
        sl.prev_qp_delta_nz = false;
    }

    /// Analyses and codes one macroblock of a P slice.
    #[allow(clippy::too_many_arguments)]
    fn code_p_mb(
        &mut self,
        sl: &mut SliceState,
        w: &mut Bits,
        cabac: &mut Cabac,
        mb_x: usize,
        mb_y: usize,
        qp: u8,
        src: &MbSrc,
    ) {
        let mb_w = self.mb_w;
        let g_stride = mb_w * 4;
        let (gx0, gy0) = (mb_x * 4, mb_y * 4);
        let (x0, y0) = (mb_x * 16, mb_y * 16);
        let qpc = chroma_qp(qp as i32, sl.chroma_qp_offset);
        let bounds = MvBounds::new(x0, y0, self.refp.y.w, self.refp.y.h);
        let lambda = LAMBDA[qp as usize] as u32;

        let pskip_mv = pred_pskip_motion_grid(&self.ref_grid, &self.mv_grid, g_stride, gx0, gy0);
        let mvp = pred_motion_partition(&self.ref_grid, &self.mv_grid, g_stride, gx0, gy0, 4, 0);
        let mut py = [0u8; 256];
        let mut pu = [0u8; 64];
        let mut pv = [0u8; 64];

        // 1. Skip probe: the skip prediction's residual must decimate to nothing.
        if sl.skip_bounds.qp != qp {
            sl.skip_bounds = SkipBounds::new(qp, qpc);
        }
        if bounds.contains(pskip_mv) {
            // Whole-sample vectors are measured on the reference planes in place, and the
            // prediction is copied only when a probe or the reconstruction needs it.
            let int_y = (pskip_mv[0] | pskip_mv[1]) & 3 == 0;
            let int_c = (pskip_mv[0] | pskip_mv[1]) & 7 == 0;
            let (ry, ru, rv) = (&self.refp.y, &self.refp.u, &self.refp.v);
            let yo = ry.at(
                x0 as isize + (pskip_mv[0] >> 2) as isize,
                y0 as isize + (pskip_mv[1] >> 2) as isize,
            );
            let co = ru.at(
                (mb_x * 8) as isize + (pskip_mv[0] >> 3) as isize,
                (mb_y * 8) as isize + (pskip_mv[1] >> 3) as isize,
            );
            let sad = if int_y {
                pixel_sad_16x16(&src.y, 16, &ry.buf[yo..], ry.stride)
            } else {
                mc_luma(ry, x0, y0, pskip_mv, &mut py);
                pixel_sad_16x16(&src.y, 16, &py, 16)
            };
            let zero = &sl.skip_bounds;
            if sad < zero.probe {
                let mut luma_zero = SkipBounds::certain(
                    sad,
                    zero.luma,
                    || match int_y {
                        true => pixel_ssd::<16>(&src.y, 16, &ry.buf[yo..], ry.stride),
                        false => pixel_ssd::<16>(&src.y, 16, &py, 16),
                    },
                    zero.luma_ssd,
                );
                if !luma_zero {
                    if int_y {
                        ry.load::<16>(yo, &mut py);
                    }
                    luma_zero = probe_skip_luma(src, &py, qp);
                }
                if luma_zero {
                    if !int_c {
                        mc_chroma(&self.refp, mb_x * 8, mb_y * 8, pskip_mv, &mut pu, &mut pv);
                    }
                    let plane_zero = |s: &[u8; 64], r: &Plane, p: &[u8; 64]| {
                        let (b, o, st) = if int_c {
                            (&r.buf[..], co, r.stride)
                        } else {
                            (&p[..], 0, 8)
                        };
                        let sad = pixel_sad_8x8(s, 8, &b[o..], st);
                        let ssd = || pixel_ssd::<8>(s, 8, &b[o..], st);
                        SkipBounds::certain(sad, zero.chroma, ssd, zero.chroma_ssd)
                    };
                    let mut chroma_zero =
                        plane_zero(&src.u, ru, &pu) && plane_zero(&src.v, rv, &pv);
                    if !chroma_zero {
                        if int_c {
                            ru.load::<8>(co, &mut pu);
                            rv.load::<8>(co, &mut pv);
                        }
                        chroma_zero = probe_skip_chroma(src, &pu, &pv, qpc);
                    }
                    if chroma_zero {
                        self.code_skip(sl, cabac, mb_x, mb_y, pskip_mv);
                        let (cur, refp) = (&mut self.cur, &self.refp);
                        let (cx, cy) = (mb_x * 8, mb_y * 8);
                        match int_y {
                            true => cur.y.copy_from::<16>(&refp.y, yo, x0, y0),
                            false => cur.y.store::<16>(x0, y0, &py),
                        }
                        match int_c {
                            true => {
                                cur.u.copy_from::<8>(&refp.u, co, cx, cy);
                                cur.v.copy_from::<8>(&refp.v, co, cx, cy);
                            }
                            false => {
                                cur.u.store::<8>(cx, cy, &pu);
                                cur.v.store::<8>(cx, cy, &pv);
                            }
                        }
                        return;
                    }
                }
            }
        }

        // 2. Motion search from the median, skip, zero, neighbouring and co-located vectors.
        let mb_idx = mb_y * mb_w + mb_x;
        let mut cands = [mvp, pskip_mv, [0, 0], [0, 0], [0, 0], [0, 0]];
        let mut n = 3;
        if mb_x > 0 {
            cands[n] = self.mv_grid[gy0 * g_stride + gx0 - 1];
            n += 1;
        }
        if mb_y > 0 {
            cands[n] = self.mv_grid[(gy0 - 1) * g_stride + gx0];
            n += 1;
        }
        cands[n] = self.prev_mb_mv[mb_idx];
        n += 1;
        let ms = MotionSearch {
            r: &self.refp.y,
            src: &src.y,
            x0,
            y0,
            mvp,
            lambda,
            bounds,
        };
        let (full, full_cost) = ms.full_pel(&cands[..n]);
        // A full-pel match whose luma residual already quantizes to nothing needs no refinement.
        let full_mv = [full[0] * 4, full[1] * 4];
        let mv = if full_cost - ms.mv_cost(full_mv) <= sl.skip_bounds.luma {
            let r = &self.refp.y;
            r.load::<16>(ms.full_off(full[0], full[1]), &mut py);
            [full_mv[0] as i16, full_mv[1] as i16]
        } else {
            ms.sub_pel(full, full_cost, &mut self.win, &mut py).0
        };
        let inter_satd = pixel_satd_16x16(&src.y, 16, &py, 16);
        let inter_cost = inter_satd
            + lambda * (se_bits((mv[0] - mvp[0]) as i32) + se_bits((mv[1] - mvp[1]) as i32) + 1);

        // 3. Intra 16x16 when inter prediction is poor.
        if inter_satd > 24 * lambda {
            let (i16_satd, i16_mode) = analyse_i16(&self.cur, mb_x, mb_y, src);
            if i16_satd + lambda * 8 < inter_cost {
                self.begin_coded_mb(sl, w, cabac, mb_x, mb_y);
                self.code_i16(sl, w, cabac, mb_x, mb_y, qp, src, i16_mode);
                return;
            }
        }

        // 4. P_L0_16x16 (or P_Skip when its residual vanishes at the skip vector).
        mc_chroma(&self.refp, mb_x * 8, mb_y * 8, mv, &mut pu, &mut pv);
        let (luma_q4, cbp4) = quant_luma_4x4_inter(&src.y, &py, qp);
        let mut res = InterResidual {
            luma_q4,
            luma_q8: [[0; 64]; 4],
            use_8x8: false,
            cbp_luma: cbp4,
            chroma: ChromaLevels::ZERO,
            cbp_chroma: 0,
        };
        if sl.t8x8 && sl.use_cabac && cbp4 != 0 {
            let (q8, cbp8, _) = quant_luma_8x8_inter(&src.y, &py, qp);
            let nnz4: usize = res.luma_q4.iter().flatten().filter(|&&c| c != 0).count();
            let nnz8: usize = q8.iter().flatten().filter(|&&c| c != 0).count();
            if nnz8 <= nnz4 {
                res.luma_q8 = q8;
                res.use_8x8 = true;
                res.cbp_luma = cbp8;
            }
        }
        res.chroma = quant_chroma_inter(src, &pu, &pv, qpc);
        res.cbp_chroma = res.chroma.cbp();
        if res.cbp_luma == 0 && res.cbp_chroma == 0 && mv == pskip_mv {
            self.code_skip(sl, cabac, mb_x, mb_y, mv);
            self.cur.store_mb(mb_x, mb_y, &py, &pu, &pv);
            return;
        }
        self.begin_coded_mb(sl, w, cabac, mb_x, mb_y);
        self.code_inter(
            sl, w, cabac, mb_x, mb_y, qp, mv, mvp, &res, &mut py, &mut pu, &mut pv,
        );
    }

    /// Writes and reconstructs a `P_L0_16x16` macroblock.
    #[allow(clippy::too_many_arguments)]
    fn code_inter(
        &mut self,
        sl: &mut SliceState,
        w: &mut Bits,
        cabac: &mut Cabac,
        mb_x: usize,
        mb_y: usize,
        qp: u8,
        mv: [i16; 2],
        mvp: [i16; 2],
        res: &InterResidual,
        py: &mut [u8; 256],
        pu: &mut [u8; 64],
        pv: &mut [u8; 64],
    ) {
        let mb_w = self.mb_w;
        let mb_idx = mb_y * mb_w + mb_x;
        let (g_stride, cg_stride) = (mb_w * 4, mb_w * 2);
        let (gx0, gy0) = (mb_x * 4, mb_y * 4);
        let cbp_luma = res.cbp_luma;
        let cbp_chroma = res.cbp_chroma;
        let cbp = cbp_luma | (cbp_chroma << 4);
        let mvd = [mv[0] - mvp[0], mv[1] - mvp[1]];
        let qp_delta = if cbp != 0 { qp as i32 - sl.cur_qp } else { 0 };
        let eff_qp = (sl.cur_qp + qp_delta).clamp(0, 51) as u8;
        let eff_qpc = chroma_qp(eff_qp as i32, sl.chroma_qp_offset);
        if res.use_8x8 && cbp_luma != 0 {
            self.mb_8x8_flag[mb_idx] = true;
        }
        match cabac {
            Some((ce, ctxs)) => {
                ce.encode_p_mb_type(ctxs, 0); // P_L0_16x16
                let (amvd_x, amvd_y) = neighbor_mvd_sum_16x16(&self.mvd_grid, g_stride, gx0, gy0);
                ce.encode_mvd(ctxs, 0, amvd_x, mvd[0]);
                ce.encode_mvd(ctxs, 1, amvd_y, mvd[1]);
                encode_cabac_cbp_and_residuals(
                    ce,
                    ctxs,
                    mb_x,
                    mb_y,
                    mb_w,
                    g_stride,
                    cg_stride,
                    false,
                    sl.t8x8,
                    cbp_luma,
                    cbp_chroma,
                    qp_delta,
                    &mut sl.prev_qp_delta_nz,
                    &res.luma_q4,
                    if res.use_8x8 {
                        Some(&res.luma_q8)
                    } else {
                        None
                    },
                    &res.chroma.dc,
                    &res.chroma.ac,
                    &mut self.luma_nnz,
                    &mut self.chroma_nnz,
                    &self.mb_is_intra,
                    &self.mb_is_i16x16,
                    &self.mb_is_skip,
                    &self.mb_8x8_flag,
                    &mut self.mb_cbp_luma,
                    &mut self.mb_cbp_chroma,
                    &mut self.mb_chroma_dc_cbf,
                );
            }
            None => {
                w.write_ue(0); // P_L0_16x16
                w.write_se(mvd[0] as i64);
                w.write_se(mvd[1] as i64);
                w.write_ue(inter_cbp_to_golomb(cbp));
                if cbp != 0 {
                    if sl.t8x8 {
                        w.write_bit(false);
                    }
                    w.write_se(qp_delta as i64);
                    write_cavlc_residuals(
                        w,
                        mb_x,
                        mb_y,
                        mb_w,
                        cbp_luma,
                        cbp_chroma,
                        &res.luma_q4,
                        &res.chroma,
                        &mut self.luma_nnz,
                        &mut self.chroma_nnz,
                    );
                }
                self.mb_cbp_luma[mb_idx] = cbp_luma;
                self.mb_cbp_chroma[mb_idx] = cbp_chroma;
            }
        }
        sl.cur_qp = eff_qp as i32;
        self.mb_qps[mb_idx] = eff_qp;
        if res.use_8x8 {
            for (i8x8, q8) in res.luma_q8.iter().enumerate() {
                if cbp_luma & (1 << i8x8) != 0 {
                    let o = (i8x8 >> 1) * 128 + (i8x8 & 1) * 8;
                    add8x8_idct_u8(&mut py[o..], 16, &dequantize_block8x8(q8, eff_qp));
                }
            }
        } else {
            recon_luma_4x4(py, &res.luma_q4, cbp_luma, eff_qp);
        }
        recon_chroma(pu, pv, &res.chroma, cbp_chroma, eff_qpc);
        fill_grid_4x4(&mut self.ref_grid, g_stride, gx0, gy0, 4, 4, 0);
        fill_grid_4x4(&mut self.mv_grid, g_stride, gx0, gy0, 4, 4, mv);
        if cabac.is_some() {
            fill_grid_4x4(&mut self.mvd_grid, g_stride, gx0, gy0, 4, 4, mvd);
        }
        self.cur.store_mb(mb_x, mb_y, py, pu, pv);
    }

    /// Analyses and codes an intra macroblock of an I slice (`I_4x4` or `I_16x16`).
    #[allow(clippy::too_many_arguments)]
    fn code_intra(
        &mut self,
        sl: &mut SliceState,
        w: &mut Bits,
        cabac: &mut Cabac,
        mb_x: usize,
        mb_y: usize,
        qp: u8,
        src: &MbSrc,
    ) {
        let lambda = LAMBDA[qp as usize] as u32;
        let (i16_satd, i16_mode) = analyse_i16(&self.cur, mb_x, mb_y, src);
        if i16_satd > 40 * lambda
            && let Some(i4) = self.analyse_i4(mb_x, mb_y, qp, src, i16_satd + lambda * 4)
        {
            self.code_i4(sl, w, cabac, mb_x, mb_y, qp, src, i4);
            return;
        }
        self.code_i16(sl, w, cabac, mb_x, mb_y, qp, src, i16_mode);
    }

    /// Chooses, codes and reconstructs the sixteen $4\times4$ blocks of an `I_4x4` candidate in
    /// place; returns `None` (restoring the mode grid) when its cost does not beat `budget`.
    fn analyse_i4(
        &mut self,
        mb_x: usize,
        mb_y: usize,
        qp: u8,
        src: &MbSrc,
        budget: u32,
    ) -> Option<I4Choice> {
        let lambda = LAMBDA[qp as usize] as u32;
        let g_stride = self.mb_w * 4;
        let mut ch = I4Choice {
            modes: [2; 16],
            q4: [[0; 16]; 16],
            cbp_luma: 0,
            recon: [0; 256],
        };
        let mut saved = [2u8; 16];
        for (b, s) in saved.iter_mut().enumerate() {
            let (bx, by) = blk_xy(b);
            *s = self.intra4x4_modes[(mb_y * 4 + by) * g_stride + mb_x * 4 + bx];
        }
        // Hadamard transforms of the source blocks (raster order) give the vertical, horizontal
        // and DC SATDs without building those predictions (`x264_intra_satd_x3_4x4`).
        let mut had = [[0i16; 16]; 16];
        for (strip, h) in had.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            hadamard4x4_16x4(&src.y[strip * 64..], 16, h);
        }
        let quant = quant4(qp, true);
        let scale = dequant4(qp);
        let mut total = lambda * 24;
        for b in 0..16 {
            let (bx, by) = blk_xy(b);
            let (px, py) = (mb_x * 16 + bx * 4, mb_y * 16 + by * 4);
            let (gx, gy) = (mb_x * 4 + bx, mb_y * 4 + by);
            let (top, left, tl) = i4_edges(&self.cur.y, px, py, b, mb_x, self.mb_w);
            let pred_mode = pred_intra4x4_mode(&self.intra4x4_modes, g_stride, gx, gy);
            let so = by * 64 + bx * 4;
            let mode_bits = |m: u8| lambda * if m == pred_mode { 1 } else { 4 };
            let [sv, sh, sd] = intra_satd_x3_4x4(&had[by * 4 + bx], top.as_ref(), left.as_ref());
            // Modes in order, so ties keep the lowest mode as before.
            let mut best = (u32::MAX, 2u8);
            for (m, ok, satd) in [
                (0, top.is_some(), sv),
                (1, left.is_some(), sh),
                (2, true, sd),
            ] {
                if ok && satd + mode_bits(m) < best.0 {
                    best = (satd + mode_bits(m), m);
                }
            }
            // The six directional modes at once; unavailable edges only feed modes skipped here.
            let dirs = intra4x4_pred_dir(
                top.as_ref().unwrap_or(&[128; 8]),
                left.as_ref().unwrap_or(&[128; 4]),
                tl,
            );
            for m in 3..=8u8 {
                if !intra4x4_mode_available(m, top.is_some(), left.is_some()) {
                    continue;
                }
                let cost =
                    pixel_satd_4x4(&src.y[so..], 16, &dirs[m as usize - 3], 4) + mode_bits(m);
                if cost < best.0 {
                    best = (cost, m);
                }
            }
            let pred = if best.1 >= 3 {
                dirs[best.1 as usize - 3]
            } else {
                let mut pred = [0u8; 16];
                predict_intra_4x4_u8(best.1, top.as_ref(), left.as_ref(), tl, &mut pred, 4);
                pred
            };
            total += best.0;
            if total >= budget {
                for (b, &m) in saved.iter().enumerate() {
                    let (bx, by) = blk_xy(b);
                    self.intra4x4_modes[(mb_y * 4 + by) * g_stride + mb_x * 4 + bx] = m;
                }
                return None;
            }
            ch.modes[b] = best.1;
            self.intra4x4_modes[gy * g_stride + gx] = best.1;
            let (q, nz) = quant_one(&sub4x4_dct_u8(&src.y[so..], 16, &pred, 4), quant);
            let mut rec = pred;
            if nz {
                ch.cbp_luma |= 1 << (b >> 2);
                let d: [i16; 16] = std::array::from_fn(|i| {
                    (q[i] as i32 * scale[i]).clamp(i16::MIN as i32, i16::MAX as i32) as i16
                });
                add4x4_idct_u8(&mut rec, 4, &d);
            }
            ch.q4[b] = q;
            let o = self.cur.y.idx(px, py);
            let s = self.cur.y.stride;
            for r in 0..4 {
                self.cur.y.buf[o + r * s..o + r * s + 4].copy_from_slice(&rec[r * 4..r * 4 + 4]);
                ch.recon[so + r * 16..so + r * 16 + 4].copy_from_slice(&rec[r * 4..r * 4 + 4]);
            }
        }
        Some(ch)
    }

    /// Writes and reconstructs an `I_4x4` macroblock chosen by [`Self::analyse_i4`].
    #[allow(clippy::too_many_arguments)]
    fn code_i4(
        &mut self,
        sl: &mut SliceState,
        w: &mut Bits,
        cabac: &mut Cabac,
        mb_x: usize,
        mb_y: usize,
        qp: u8,
        src: &MbSrc,
        i4: I4Choice,
    ) {
        let mb_w = self.mb_w;
        let mb_idx = mb_y * mb_w + mb_x;
        let (g_stride, cg_stride) = (mb_w * 4, mb_w * 2);
        let (chroma_mode, mut pu, mut pv) = analyse_chroma_intra(&self.cur, mb_x, mb_y, src);
        let qpc = chroma_qp(qp as i32, sl.chroma_qp_offset);
        let c = quant_chroma(src, &pu, &pv, qpc, true);
        let cbp_chroma = c.cbp();
        let (chroma_dc, chroma_ac) = (&c.dc, &c.ac);
        let cbp = i4.cbp_luma | (cbp_chroma << 4);
        let qp_delta = if cbp != 0 { qp as i32 - sl.cur_qp } else { 0 };
        let eff_qp = (sl.cur_qp + qp_delta).clamp(0, 51) as u8;
        let eff_qpc = chroma_qp(eff_qp as i32, sl.chroma_qp_offset);
        self.mb_is_intra[mb_idx] = true;
        self.mb_chroma_mode[mb_idx] = chroma_mode;
        self.set_intra_motion(mb_x, mb_y);
        // Mode prediction while writing must see the neighbours' modes, not this MB's choices.
        for b in 0..16 {
            let (bx, by) = blk_xy(b);
            self.intra4x4_modes[(mb_y * 4 + by) * g_stride + mb_x * 4 + bx] = 2;
        }
        match cabac {
            Some((ce, ctxs)) => {
                if !sl.is_idr {
                    ce.encode_p_mb_type(ctxs, 5); // I_NxN in a P slice
                } else {
                    let l = u8::from(mb_x > 0 && self.mb_is_i16x16[mb_idx - 1]);
                    let t = u8::from(mb_y > 0 && self.mb_is_i16x16[mb_idx - mb_w]);
                    ce.encode_intra_mb_type(ctxs, 3, l, t, 0);
                }
                if sl.t8x8 {
                    let l8 = mb_x > 0 && self.mb_8x8_flag[mb_idx - 1];
                    let t8 = mb_y > 0 && self.mb_8x8_flag[mb_idx - mb_w];
                    ce.encode_decision(&mut ctxs[399 + l8 as usize + t8 as usize], 0);
                }
                for (b, &m) in i4.modes.iter().enumerate() {
                    let (bx, by) = blk_xy(b);
                    let (gx, gy) = (mb_x * 4 + bx, mb_y * 4 + by);
                    let pm = pred_intra4x4_mode(&self.intra4x4_modes, g_stride, gx, gy);
                    let rem = if m < pm { m } else { m.saturating_sub(1) };
                    ce.encode_intra4x4_pred_mode(ctxs, m == pm, rem);
                    self.intra4x4_modes[gy * g_stride + gx] = m;
                }
                let l_cm = u8::from(
                    mb_x > 0
                        && self.mb_is_intra[mb_idx - 1]
                        && self.mb_chroma_mode[mb_idx - 1] != 0,
                );
                let t_cm = u8::from(
                    mb_y > 0
                        && self.mb_is_intra[mb_idx - mb_w]
                        && self.mb_chroma_mode[mb_idx - mb_w] != 0,
                );
                ce.encode_chroma_pred_mode(ctxs, l_cm, t_cm, chroma_mode);
                encode_cabac_cbp_and_residuals(
                    ce,
                    ctxs,
                    mb_x,
                    mb_y,
                    mb_w,
                    g_stride,
                    cg_stride,
                    true,
                    false,
                    i4.cbp_luma,
                    cbp_chroma,
                    qp_delta,
                    &mut sl.prev_qp_delta_nz,
                    &i4.q4,
                    None,
                    chroma_dc,
                    chroma_ac,
                    &mut self.luma_nnz,
                    &mut self.chroma_nnz,
                    &self.mb_is_intra,
                    &self.mb_is_i16x16,
                    &self.mb_is_skip,
                    &self.mb_8x8_flag,
                    &mut self.mb_cbp_luma,
                    &mut self.mb_cbp_chroma,
                    &mut self.mb_chroma_dc_cbf,
                );
            }
            None => {
                w.write_ue(if sl.is_idr { 0 } else { 5 }); // I_4x4
                if sl.t8x8 {
                    w.write_bit(false);
                }
                for (b, &m) in i4.modes.iter().enumerate() {
                    let (bx, by) = blk_xy(b);
                    let (gx, gy) = (mb_x * 4 + bx, mb_y * 4 + by);
                    let pm = pred_intra4x4_mode(&self.intra4x4_modes, g_stride, gx, gy);
                    if m == pm {
                        w.write_bit(true);
                    } else {
                        w.write_bit(false);
                        w.write_bits((if m < pm { m } else { m - 1 }) as u64, 3);
                    }
                    self.intra4x4_modes[gy * g_stride + gx] = m;
                }
                w.write_ue(chroma_mode as u64);
                w.write_ue(intra4x4_cbp_to_golomb(cbp));
                if cbp != 0 {
                    w.write_se(qp_delta as i64);
                    write_cavlc_residuals(
                        w,
                        mb_x,
                        mb_y,
                        mb_w,
                        i4.cbp_luma,
                        cbp_chroma,
                        &i4.q4,
                        &c,
                        &mut self.luma_nnz,
                        &mut self.chroma_nnz,
                    );
                }
                self.mb_cbp_luma[mb_idx] = i4.cbp_luma;
                self.mb_cbp_chroma[mb_idx] = cbp_chroma;
            }
        }
        sl.cur_qp = eff_qp as i32;
        self.mb_qps[mb_idx] = eff_qp;
        recon_chroma(&mut pu, &mut pv, &c, cbp_chroma, eff_qpc);
        self.cur.store_mb(mb_x, mb_y, &i4.recon, &pu, &pv);
    }

    /// Writes and reconstructs an `I_16x16` macroblock predicted with `mode`.
    #[allow(clippy::too_many_arguments)]
    fn code_i16(
        &mut self,
        sl: &mut SliceState,
        w: &mut Bits,
        cabac: &mut Cabac,
        mb_x: usize,
        mb_y: usize,
        qp: u8,
        src: &MbSrc,
        mode: u8,
    ) {
        let mut pred = [0u8; 256];
        let (top, left, tl) = i16_edges(&self.cur.y, mb_x, mb_y);
        pred16(mode, top.as_ref(), left.as_ref(), tl, &mut pred);
        let mb_w = self.mb_w;
        let mb_idx = mb_y * mb_w + mb_x;
        let (g_stride, cg_stride) = (mb_w * 4, mb_w * 2);
        let (chroma_mode, mut pu, mut pv) = analyse_chroma_intra(&self.cur, mb_x, mb_y, src);
        self.mb_is_intra[mb_idx] = true;
        self.mb_is_i16x16[mb_idx] = true;
        self.mb_chroma_mode[mb_idx] = chroma_mode;
        self.set_intra_motion(mb_x, mb_y);
        let qp_delta = qp as i32 - sl.cur_qp;
        let eff_qp = (sl.cur_qp + qp_delta).clamp(0, 51) as u8;
        let eff_qpc = chroma_qp(eff_qp as i32, sl.chroma_qp_offset);

        let mut dc_raw = [0i16; 16];
        let mut ac_q4 = [[0i16; 16]; 16];
        dct_luma(&src.y, &pred, &mut ac_q4);
        for (b, ac) in ac_q4.iter().enumerate() {
            dc_raw[Z_TO_RASTER[b]] = ac[0];
        }
        let q = quant4(eff_qp, true);
        // Zig-zag scanned AC levels (slot 0 empty) and masks of the non-zero ones.
        let mut ac_zz = [[0i16; 16]; 16];
        let mut ac_masks = [0u32; 16];
        for i8x8 in 0..4 {
            let group = group4(&mut ac_q4, i8x8);
            if quant_4x4_x4(group, q) != 0 {
                for blk in group.iter_mut() {
                    blk[0] = 0;
                }
                let zz = zigzag_4x4_x4(group);
                let masks = nonzero_mask_4x4_x4(&zz);
                for k in 0..4 {
                    ac_zz[i8x8 * 4 + k] = zz[k];
                    ac_masks[i8x8 * 4 + k] = masks[k] >> 1;
                }
            }
        }
        let any_ac = ac_masks.iter().any(|&m| m != 0);
        let dc_q = quantize_luma_dc4x4(&dc_raw, eff_qp);
        let dc_scanned = zigzag16(&dc_q);
        let cbp_luma = if any_ac { 15u8 } else { 0 };
        let c = quant_chroma(src, &pu, &pv, eff_qpc, true);
        let cbp_chroma = c.cbp();
        let (chroma_dc, chroma_ac) = (&c.dc, &c.ac);
        let mb_type = 1 + mode as u32 + 4 * cbp_chroma as u32 + if cbp_luma != 0 { 12 } else { 0 };
        self.mb_cbp_luma[mb_idx] = cbp_luma;
        self.mb_cbp_chroma[mb_idx] = cbp_chroma;

        match cabac {
            Some((ce, ctxs)) => {
                if !sl.is_idr {
                    ce.encode_p_mb_type(ctxs, 5 + mb_type);
                } else {
                    let l = u8::from(mb_x > 0 && self.mb_is_i16x16[mb_idx - 1]);
                    let t = u8::from(mb_y > 0 && self.mb_is_i16x16[mb_idx - mb_w]);
                    ce.encode_intra_mb_type(ctxs, 3, l, t, mb_type);
                }
                let l_cm = u8::from(
                    mb_x > 0
                        && self.mb_is_intra[mb_idx - 1]
                        && self.mb_chroma_mode[mb_idx - 1] != 0,
                );
                let t_cm = u8::from(
                    mb_y > 0
                        && self.mb_is_intra[mb_idx - mb_w]
                        && self.mb_chroma_mode[mb_idx - mb_w] != 0,
                );
                ce.encode_chroma_pred_mode(ctxs, l_cm, t_cm, chroma_mode);
                ce.encode_mb_qp_delta(ctxs, u8::from(sl.prev_qp_delta_nz), qp_delta);
                sl.prev_qp_delta_nz = qp_delta != 0;
                let l_dc = (mb_x > 0).then(|| self.mb_luma_dc_cbf[mb_idx - 1]);
                let t_dc = (mb_y > 0).then(|| self.mb_luma_dc_cbf[mb_idx - mb_w]);
                ce.encode_residual_block_cat(
                    ctxs,
                    0,
                    cabac_cbf_ctx_inc(l_dc, t_dc, true),
                    &dc_scanned,
                );
                self.mb_luma_dc_cbf[mb_idx] = dc_scanned.iter().any(|&c| c != 0);
                if cbp_luma != 0 {
                    for (b, zz) in ac_zz.iter().enumerate() {
                        let (bx, by) = blk_xy(b);
                        let (gx, gy) = (mb_x * 4 + bx, mb_y * 4 + by);
                        let scanned: &[i16; 15] = zz[1..].try_into().unwrap();
                        let nb = |left| {
                            luma_neighbor_cbf(
                                &self.luma_nnz,
                                &self.mb_is_intra,
                                &self.mb_is_i16x16,
                                &self.mb_is_skip,
                                &self.mb_cbp_luma,
                                mb_w,
                                g_stride,
                                gx,
                                gy,
                                left,
                            )
                        };
                        let inc = cabac_cbf_ctx_inc(nb(true), nb(false), true);
                        ce.encode_residual_block_cat(ctxs, 1, inc, scanned);
                        self.luma_nnz[gy * g_stride + gx] = ac_masks[b].count_ones() as u8;
                    }
                }
                encode_cabac_chroma_residuals(
                    ce,
                    ctxs,
                    mb_x,
                    mb_y,
                    mb_w,
                    cg_stride,
                    true,
                    cbp_chroma,
                    chroma_dc,
                    chroma_ac,
                    &mut self.chroma_nnz,
                    &self.mb_is_intra,
                    &self.mb_is_skip,
                    &self.mb_cbp_chroma,
                    &mut self.mb_chroma_dc_cbf,
                );
            }
            None => {
                w.write_ue((if sl.is_idr { 0 } else { 5 }) + mb_type as u64);
                w.write_ue(chroma_mode as u64);
                w.write_se(qp_delta as i64);
                let nc_dc = predict_luma_nc(&self.luma_nnz, g_stride, mb_x * 4, mb_y * 4);
                write_cavlc_block(w, &dc_scanned, nc_dc);
                if cbp_luma != 0 {
                    for (b, zz) in ac_zz.iter().enumerate() {
                        let (bx, by) = blk_xy(b);
                        let (gx, gy) = (mb_x * 4 + bx, mb_y * 4 + by);
                        let nc = predict_luma_nc(&self.luma_nnz, g_stride, gx, gy);
                        write_cavlc_masked::<15>(w, &zz[1..], nc, ac_masks[b]);
                        self.luma_nnz[gy * g_stride + gx] = ac_masks[b].count_ones() as u8;
                    }
                }
                write_cavlc_chroma(w, mb_x, mb_y, mb_w, cbp_chroma, &c, &mut self.chroma_nnz);
            }
        }
        sl.cur_qp = eff_qp as i32;
        self.mb_qps[mb_idx] = eff_qp;

        let dc_recon = dequantize_idct_luma_dc4x4(&dc_q, eff_qp);
        let scale = dequant4(eff_qp);
        for i8x8 in 0..4 {
            let group = group4(&mut ac_q4, i8x8);
            if cbp_luma != 0 {
                dequant_4x4_x4(group, scale);
            }
            for (k, blk) in group.iter_mut().enumerate() {
                blk[0] = dc_recon[Z_TO_RASTER[i8x8 * 4 + k]];
            }
            if group.iter().flatten().any(|&c| c != 0) {
                let o = (i8x8 >> 1) * 128 + (i8x8 & 1) * 8;
                add_idct4x4_8x8(&mut pred[o..], 16, group);
            }
        }
        recon_chroma(&mut pu, &mut pv, &c, cbp_chroma, eff_qpc);
        self.cur.store_mb(mb_x, mb_y, &pred, &pu, &pv);
    }

    /// ITU-T H.264 8.7 deblocking of the current picture, macroblock by macroblock: vertical
    /// edges left to right, then horizontal edges top to bottom.
    fn deblock(&mut self, chroma_qp_offset: i32) {
        let (mb_w, mb_h) = (self.mb_w, self.mb_h);
        let g_stride = mb_w * 4;
        // An inter macroblock without luma residual next to others with the same motion has
        // boundary strength 0 on every edge.
        let quiet = |m: usize| !self.mb_is_intra[m] && self.mb_cbp_luma[m] == 0;
        let mv_of = |mb_x: usize, mb_y: usize| self.mv_grid[mb_y * 4 * g_stride + mb_x * 4];
        // Non-zero 4x4 blocks of one grid row of four, bit `c` for column `c`.
        let nz_row = |nnz: &[u8], g: usize| -> u32 {
            let w = u32::from_le_bytes(nnz[g..g + 4].try_into().unwrap());
            let hi = (((w & 0x7f7f_7f7f) + 0x7f7f_7f7f) | w) & 0x8080_8080;
            ((hi >> 7) & 1) | ((hi >> 14) & 2) | ((hi >> 21) & 4) | ((hi >> 28) & 8)
        };
        // Column `c` of four such rows, bit `k` for row `k`.
        let nz_col = |rows: &[u32; 4], c: usize| -> u32 {
            ((rows[0] >> c) & 1)
                | (((rows[1] >> c) & 1) << 1)
                | (((rows[2] >> c) & 1) << 2)
                | (((rows[3] >> c) & 1) << 3)
        };
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let m = mb_y * mb_w + mb_x;
                if quiet(m)
                    && (mb_x == 0 || (quiet(m - 1) && mv_of(mb_x - 1, mb_y) == mv_of(mb_x, mb_y)))
                    && (mb_y == 0
                        || (quiet(m - mb_w) && mv_of(mb_x, mb_y - 1) == mv_of(mb_x, mb_y)))
                {
                    continue;
                }
                let g0 = mb_y * 4 * g_stride + mb_x * 4;
                let intra = self.mb_is_intra[m];
                // Every partition is 16x16, so motion only differs across macroblock edges, and
                // a macroblock without luma residual needs none of its internal edges filtered.
                let internal = intra || self.mb_cbp_luma[m] != 0;
                let skip_odd = self.mb_8x8_flag[m];
                let rows: [u32; 4] =
                    std::array::from_fn(|r| nz_row(&self.luma_nnz, g0 + r * g_stride));
                let qp_q = self.mb_qps[m] as i32;
                let qpc_q = chroma_qp(qp_q, chroma_qp_offset) as i32;
                for dir in 0..2 {
                    for e in 0..4 {
                        let pm = match (e, dir) {
                            (0, 0) if mb_x > 0 => m - 1,
                            (0, 1) if mb_y > 0 => m - mb_w,
                            (0, _) => continue,
                            _ if !internal || (e & 1 == 1 && skip_odd) => continue,
                            _ => m,
                        };
                        let bs = if intra || self.mb_is_intra[pm] {
                            [if e == 0 { 4 } else { 3 }; 4]
                        } else {
                            // Bit k: the 4x4 blocks on either side of segment k hold levels.
                            let coded = match (dir, e) {
                                (0, 0) => {
                                    nz_col(&rows, 0)
                                        | (0..4).fold(0, |a, k| {
                                            a | u32::from(self.luma_nnz[g0 + k * g_stride - 1] != 0)
                                                << k
                                        })
                                }
                                (0, _) => nz_col(&rows, e) | nz_col(&rows, e - 1),
                                (_, 0) => rows[0] | nz_row(&self.luma_nnz, g0 - g_stride),
                                _ => rows[e] | rows[e - 1],
                            };
                            let motion = e == 0 && {
                                let gp = if dir == 0 { g0 - 1 } else { g0 - g_stride };
                                let (mq, mp) = (self.mv_grid[g0], self.mv_grid[gp]);
                                self.ref_grid[g0] != self.ref_grid[gp]
                                    || (mq[0] - mp[0]).abs() >= 4
                                    || (mq[1] - mp[1]).abs() >= 4
                            };
                            if coded == 0 && !motion {
                                continue;
                            }
                            let other = u8::from(motion);
                            std::array::from_fn(|k| if coded >> k & 1 != 0 { 2 } else { other })
                        };
                        let (qp, qpc) = if pm == m {
                            (qp_q, qpc_q)
                        } else {
                            let qp_p = self.mb_qps[pm] as i32;
                            (
                                (qp_q + qp_p + 1) >> 1,
                                (qpc_q + chroma_qp(qp_p, chroma_qp_offset) as i32 + 1) >> 1,
                            )
                        };
                        let (x, y) = if dir == 0 {
                            (mb_x * 16 + e * 4, mb_y * 16)
                        } else {
                            (mb_x * 16, mb_y * 16 + e * 4)
                        };
                        if let Some(f) = EdgeFilter::new(qp, bs) {
                            f.apply(&mut self.cur.y, x, y, dir, false);
                        }
                        if e & 1 == 0
                            && let Some(f) = EdgeFilter::new(qpc, bs)
                        {
                            f.apply(&mut self.cur.u, x / 2, y / 2, dir, true);
                            f.apply(&mut self.cur.v, x / 2, y / 2, dir, true);
                        }
                    }
                }
            }
        }
    }
}

/// `tc0` per boundary strength 0..=4 at each indexA (`-1`: no filtering; `0` with strength 4,
/// which the strong filter ignores).
static DEBLOCK_TC0: [[i8; 5]; 52] = {
    let mut t = [[0i8; 5]; 52];
    let mut q = 0;
    while q < 52 {
        t[q] = [
            -1,
            H264_DEBLOCK_TC0[q][0] as i8,
            H264_DEBLOCK_TC0[q][1] as i8,
            H264_DEBLOCK_TC0[q][2] as i8,
            0,
        ];
        q += 1;
    }
    t
};

/// Thresholds of one edge for the `ragdoll-simd` deblocking kernels (`FFmpeg` conventions).
struct EdgeFilter {
    alpha: i32,
    beta: i32,
    tc0: [i8; 4],
    strong: bool,
}

impl EdgeFilter {
    /// `None` when `qp` disables filtering (`alpha` or `beta` zero).
    #[inline(always)]
    fn new(qp: i32, bs: [u8; 4]) -> Option<Self> {
        let ia = qp.clamp(0, 51) as usize;
        let (alpha, beta) = (H264_DEBLOCK_ALPHA[ia] as i32, H264_DEBLOCK_BETA[ia] as i32);
        let tc = &DEBLOCK_TC0[ia];
        (alpha != 0 && beta != 0).then(|| Self {
            alpha,
            beta,
            tc0: bs.map(|b| tc[b.min(4) as usize]),
            strong: bs[0] == 4,
        })
    }

    /// Filters one 16-sample luma or 8-sample chroma edge starting at `(x, y)`; `dir` 0 is a
    /// vertical edge, 1 a horizontal one.
    #[inline(always)]
    fn apply(&self, p: &mut Plane, x: usize, y: usize, dir: usize, chroma: bool) {
        let q0 = p.idx(x, y);
        let (xs, ys) = if dir == 0 {
            (1, p.stride)
        } else {
            (p.stride, 1)
        };
        let (a, b, tc0, s) = (self.alpha, self.beta, self.tc0, self.strong);
        if chroma {
            h264_chroma_deblock(&mut p.buf, q0, xs, ys, a, b, tc0, s);
        } else {
            h264_luma_deblock(&mut p.buf, q0, xs, ys, a, b, tc0, s);
        }
    }
}

/// An `I_4x4` candidate: modes, levels and reconstruction of its sixteen blocks.
struct I4Choice {
    modes: [u8; 16],
    q4: [[i16; 16]; 16],
    cbp_luma: u8,
    recon: [u8; 256],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nal_escaping_matches_the_reference() {
        let mut x = 0x2545_f491_u32;
        for len in 0..300usize {
            let data: Vec<u8> = (0..len)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5;
                    // Mostly zeros and small values, so escapes and long zero runs occur.
                    match x % 8 {
                        0..=3 => 0,
                        4 | 5 => (x >> 8) as u8 & 3,
                        _ => (x >> 8) as u8,
                    }
                })
                .collect();
            let header = NalHeader {
                nal_ref_idc: 3,
                nal_unit_type: NalUnitType::NonIdrSlice,
            };
            let mut out = vec![9];
            append_annex_b_nal(&mut out, header, &data);
            assert_eq!(
                out[1..],
                encode_annex_b_nal(3, NalUnitType::NonIdrSlice, &data)[..]
            );
        }
    }

    #[test]
    fn cavlc_writer_matches_reference_block_coder() {
        let mut x = 0x9e37_79b9_u32;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        for _ in 0..20_000 {
            let len = [16usize, 15, 4][(next() % 3) as usize];
            let density = next() % 17;
            let range = [4u32, 4, 40, 3000][(next() % 4) as usize];
            let mut c = [0i16; 16];
            for v in c.iter_mut().take(len) {
                if next() % 16 < density {
                    let mag = (next() % range) as i16 + 1;
                    *v = if next() & 1 == 0 { mag } else { -mag };
                }
            }
            if len != 4 {
                let mask = (0..len).fold(0u32, |m, i| m | u32::from(c[i] != 0) << i);
                assert_eq!(decimate_masked(&c[..len], mask), decimate_score(&c[..len]));
            }
            let nc = (next() % 17) as u8;
            let chroma_dc = len == 4;
            let (mut a, mut b) = (Bits::new(Vec::new()), BitWriter::new());
            match len {
                16 => write_cavlc_block(&mut a, &c, nc),
                15 => write_cavlc_block::<15>(&mut a, c[..15].try_into().unwrap(), nc),
                _ => write_cavlc_block::<4>(&mut a, c[..4].try_into().unwrap(), nc),
            }
            encode_cavlc_block(&mut b, &c[..len], nc, chroma_dc);
            assert_eq!(
                a.bits_written(),
                b.bits_written(),
                "{:?} nc {nc}",
                &c[..len]
            );
            assert_eq!(a.finish(), b.finish(), "{:?} nc {nc}", &c[..len]);
        }
    }

    #[test]
    fn window_prediction_matches_luma_mc_up_to_the_vector_bounds() {
        let (w, h) = (64, 48);
        let mut p = Plane::alloc(w, h, PAD_Y);
        let mut x = 0x1234_5678_u32;
        for y in 0..h {
            for c in 0..w {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let i = p.idx(c, y);
                p.buf[i] = if (c + y) % 7 == 0 {
                    255
                } else {
                    (x >> 24) as u8
                };
            }
        }
        p.extend_borders();
        let src = [0u8; 256];
        let mut win = HpelWindow::default();
        for (x0, y0) in [(0, 0), (16, 16), (48, 32)] {
            let bounds = MvBounds::new(x0, y0, w, h);
            let ms = MotionSearch {
                r: &p,
                src: &src,
                x0,
                y0,
                mvp: [0, 0],
                lambda: 1,
                bounds,
            };
            let lo = [(bounds.min[0] + 3) >> 2, (bounds.min[1] + 3) >> 2];
            let hi = [bounds.max[0] >> 2, bounds.max[1] >> 2];
            for full in [[0, 0], [-3, 2], lo, hi, [lo[0], hi[1]]] {
                let (bx, by) = (
                    x0 as isize + full[0] as isize,
                    y0 as isize + full[1] as isize,
                );
                hpel_window(&p.buf, p.at(bx - 1, by - 1), p.stride, &mut win);
                hpel_window_centre(&mut win);
                for dy in -3..=3 {
                    for dx in -3..=3 {
                        let mv = [full[0] * 4 + dx, full[1] * 4 + dy];
                        if bounds.clamp(mv) != mv {
                            continue;
                        }
                        let (mut a, mut b) = ([0u8; 256], [0u8; 256]);
                        ms.window_pred(&win, bx, by, mv, &mut a);
                        mc_luma(&p, x0, y0, [mv[0] as i16, mv[1] as i16], &mut b);
                        assert_eq!(a, b, "block ({x0}, {y0}) mv {mv:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn fast_intra_predictors_match_the_reference_ones() {
        let mut x = 0xdead_beef_u32;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        for round in 0..300 {
            let pick = |v: u32| match round % 3 {
                0 => (v >> 24) as u8,
                1 => [0u8, 255][(v & 1) as usize],
                _ => 128 + (v % 8) as u8,
            };
            let t16: [u8; 16] = std::array::from_fn(|_| pick(next()));
            let l16: [u8; 16] = std::array::from_fn(|_| pick(next()));
            let tl = pick(next());
            for mode in 0..4u8 {
                let (mut a, mut b) = ([0u8; 256], [0u8; 256]);
                pred16(mode, Some(&t16), Some(&l16), tl, &mut a);
                predict_intra_16x16_scalar(mode, Some(&t16), Some(&l16), tl, &mut b, 16);
                assert_eq!(a, b, "16x16 mode {mode}");
                let (t8, l8): ([u8; 8], [u8; 8]) =
                    (t16[..8].try_into().unwrap(), l16[8..].try_into().unwrap());
                let (mut c, mut d) = ([0u8; 64], [0u8; 64]);
                pred_chroma(mode, Some(&t8), Some(&l8), tl, &mut c);
                predict_intra_chroma_8x8_scalar(mode, Some(&t8), Some(&l8), tl, &mut d, 8);
                assert_eq!(c, d, "chroma mode {mode}");
            }
        }
    }

    #[test]
    fn decimation_scores_follow_x264() {
        let one_at = |i: usize, v: i16| {
            let mut c = [0i16; 16];
            c[i] = v;
            c
        };
        assert_eq!(decimate_score(&[0; 16]), 0);
        assert_eq!(decimate_score(&one_at(0, 1)), 3);
        assert_eq!(decimate_score(&one_at(15, -1)), 0);
        assert_eq!(decimate_score(&one_at(1, 2)), 9);
        let mut two = one_at(0, 1);
        two[1] = -1;
        assert_eq!(decimate_score(&two), 6);
    }
}
