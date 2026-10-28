//! Slice data and macroblock layer decoding (ITU-T H.264 7.3.4, 7.3.5, 8.3, 8.4, 8.5).

#![allow(clippy::needless_range_loop)]

use std::sync::Arc;

use ragdoll_core::{BitReader, RagdollError, Result};

use crate::cabac::{
    CabacContext, CabacDecoder, NUM_CABAC_CONTEXTS, ResidualCoeffs, init_cabac_contexts,
};
use crate::h264::{decode_cavlc_block, decode_cavlc_coeffs};

use super::cabac_se;
use super::intra::{self, Edges};
use super::picture::{MbType, Motion, NO_MOTION, NO_PIC, PicState, Structure, fill_rect};
use super::ps::{Pps, Sps};
use super::recon::{self, Dequant, Weighting};
use super::slice::{SliceHeader, SliceType};
use super::tables::{
    B_MB_TYPES, B_SUB_TYPES, BLK4_AT, BLK4_XY, CBP_ME_420, CBP_ME_MONO, FIELD_SCAN4X4,
    FIELD_SCAN8X8, P_SUB_TYPES, PRED_L0, Part, SCAN4X4, SCAN8X8, chroma_qp,
};

/// A reference picture as seen by one slice.
pub struct RefPic {
    /// Frame or field reference id (see `FrameStore::frame_ref_id`).
    pub id: u32,
    pub poc: i32,
    pub long: bool,
    /// Samples of the frame, or of the field for a field reference.
    pub planes: Arc<[Vec<u8>; 3]>,
    /// Motion of the frame store, in frame layout.
    pub motion: Option<Arc<Motion>>,
    /// Field parity for a field reference.
    pub parity: Option<u8>,
    /// POCs of the store's two fields.
    pub field_poc: [i32; 2],
}

/// Everything a slice needs besides the picture state.
pub struct SliceCtx<'a> {
    pub sps: &'a Sps,
    pub pps: &'a Pps,
    pub sh: &'a SliceHeader,
    pub slice_num: u32,
    pub refs: [Vec<RefPic>; 2],
    /// MBAFF only: the field reference lists of field macroblocks, by MB parity then list.
    pub field_refs: Option<[[Vec<RefPic>; 2]; 2]>,
    pub cur_poc: i32,
    /// Top and bottom field POCs of the current frame (field macroblocks of MBAFF frames).
    pub field_poc: [i32; 2],
    pub dq: &'a Dequant,
}

enum Entropy<'a> {
    Cavlc(BitReader<'a>),
    Cabac(CabacDecoder<'a>, Box<[CabacContext; NUM_CABAC_CONTEXTS]>),
}

fn bad(msg: &'static str) -> RagdollError {
    RagdollError::InvalidSyntax(msg)
}

/// Decodes the slice data of one slice into `st`.
pub fn decode_slice(cx: &SliceCtx<'_>, st: &mut PicState, rbsp: &[u8]) -> Result<()> {
    let sh = cx.sh;
    let mut r = BitReader::new(rbsp);
    r.skip_bits(sh.data_bit_offset)?;
    let ent = if cx.pps.cabac {
        r.align_to_byte();
        let pos = r.bits_read() / 8;
        let data = rbsp.get(pos..).ok_or(RagdollError::BitstreamEndOfData)?;
        let ctxs = Box::new(init_cabac_contexts(
            sh.qp as u8,
            sh.slice_type.is_intra(),
            sh.cabac_init_idc as u8,
        ));
        Entropy::Cabac(CabacDecoder::new(data)?, ctxs)
    } else {
        Entropy::Cavlc(r)
    };
    let total = st.mb_w * st.mb_h;
    let first = sh.first_mb as usize * if st.mbaff { 2 } else { 1 };
    if first >= total {
        return Err(bad("H.264 first_mb_in_slice beyond picture"));
    }
    let implicit = build_implicit_weights(cx, &cx.refs, cx.cur_poc);
    let implicit_field = match &cx.field_refs {
        Some(f) => [0, 1].map(|p| build_implicit_weights(cx, &f[p], cx.field_poc[p])),
        None => [Vec::new(), Vec::new()],
    };
    let mut d = Dec {
        cx,
        st,
        ent,
        qp: sh.qp,
        last_dqp_nz: false,
        mb: MbPos {
            addr: 0,
            x: 0,
            y: 0,
        },
        field: cx.sh.field_pic,
        implicit,
        implicit_field,
        scratch: [[0; 256]; 2],
        partial_motion: false,
        res: None,
    };
    d.run(first, total)
}

/// Position of a macroblock: `addr` indexes per-MB state, `(x, y)` its MB column and row
/// (for MBAFF, `y = 2 * pair_row + bottom`).
#[derive(Clone, Copy)]
struct MbPos {
    addr: usize,
    x: usize,
    y: usize,
}

/// Implicit bi-prediction weights `[w0, w1]` per `(refIdxL0, refIdxL1)` (8.4.2.3.1), empty when unused.
fn build_implicit_weights(
    cx: &SliceCtx<'_>,
    refs: &[Vec<RefPic>; 2],
    cur_poc: i32,
) -> Vec<[i32; 2]> {
    if !(cx.sh.slice_type.is_b() && cx.pps.weighted_bipred_idc == 2) {
        return Vec::new();
    }
    let (l0, l1) = (&refs[0], &refs[1]);
    let mut out = Vec::with_capacity(l0.len() * l1.len());
    for r0 in l0 {
        for r1 in l1 {
            let td = (r1.poc - r0.poc).clamp(-128, 127);
            let w = if r0.long || r1.long || td == 0 {
                [32, 32]
            } else {
                let tb = (cur_poc - r0.poc).clamp(-128, 127);
                let tx = (16384 + (td / 2).abs()) / td;
                let dsf = ((tb * tx + 32) >> 6).clamp(-1024, 1023);
                if (dsf >> 2) < -64 || (dsf >> 2) > 128 {
                    [32, 32]
                } else {
                    [64 - (dsf >> 2), dsf >> 2]
                }
            };
            out.push(w);
        }
    }
    out
}

/// Motion of one block for one list: reference index (`-1` = list unused) and vector.
type Mv = (i8, [i16; 2]);

/// Parsed residual of one macroblock, dequantised, raster order per block.
struct Residual {
    luma: [[i32; 16]; 16],
    luma_coded: u16,
    luma8: [[i32; 64]; 4],
    chroma: [[[i32; 16]; 4]; 2],
    chroma_coded: [u8; 2],
}

impl Residual {
    fn is_clear(&self) -> bool {
        self.luma.iter().flatten().all(|&v| v == 0)
            && self.luma8.iter().flatten().all(|&v| v == 0)
            && self.chroma.iter().flatten().flatten().all(|&v| v == 0)
    }

    fn new() -> Self {
        Self {
            luma: [[0; 16]; 16],
            luma_coded: 0,
            luma8: [[0; 64]; 4],
            chroma: [[[0; 16]; 4]; 2],
            chroma_coded: [0; 2],
        }
    }
}

struct Dec<'a, 'c> {
    cx: &'c SliceCtx<'a>,
    st: &'c mut PicState,
    ent: Entropy<'a>,
    qp: i32,
    last_dqp_nz: bool,
    mb: MbPos,
    /// The current macroblock is a field macroblock (field picture or MBAFF field pair).
    field: bool,
    implicit: Vec<[i32; 2]>,
    implicit_field: [Vec<[i32; 2]>; 2],
    /// Prediction samples of the second list, or of both for weighted prediction.
    scratch: [[u8; 256]; 2],
    /// The current MB's motion was stored for less than the whole MB, so it may vary.
    partial_motion: bool,
    /// Residual blocks, kept zeroed between macroblocks: the inverse transforms clear what they
    /// read, so only coded coefficients are ever written.
    res: Option<Box<Residual>>,
}

impl<'a, 'c> Dec<'a, 'c> {
    fn run(&mut self, first: usize, total: usize) -> Result<()> {
        let slice_type = self.cx.sh.slice_type;
        let mbaff = self.st.mbaff;
        let mut addr = first;
        if !self.is_cabac() {
            loop {
                if !slice_type.is_intra() {
                    let run = self.reader().read_ue()? as usize;
                    for left in (0..run).rev() {
                        if addr >= total {
                            return Err(bad("H.264 mb_skip_run beyond picture"));
                        }
                        if mbaff && addr.is_multiple_of(2) {
                            self.field = self.inferred_field(addr);
                            // A coded bottom MB follows: its mb_field_decoding_flag also
                            // applies to this skipped top MB.
                            if left == 0 && self.reader().more_rbsp_data() {
                                self.field = self.reader().read_bit()?;
                            }
                        }
                        self.decode_skip(addr)?;
                        addr += 1;
                    }
                    if run > 0 && !self.reader().more_rbsp_data() {
                        return Ok(());
                    }
                }
                if addr >= total {
                    return Err(bad("H.264 slice data beyond picture"));
                }
                if mbaff && addr.is_multiple_of(2) {
                    self.field = self.reader().read_bit()?;
                }
                self.decode_mb(addr)?;
                addr += 1;
                if !self.reader().more_rbsp_data() {
                    return Ok(());
                }
            }
        } else {
            let mut next_skipped = false;
            let mut top_skipped = false;
            loop {
                if addr >= total {
                    return Err(bad("H.264 slice data beyond picture"));
                }
                let top = addr.is_multiple_of(2);
                if mbaff && top {
                    self.field = self.inferred_field(addr);
                }
                let skipped = if slice_type.is_intra() {
                    false
                } else if mbaff && !top && top_skipped {
                    next_skipped
                } else {
                    self.decode_skip_flag(addr)
                };
                if skipped {
                    if mbaff && top {
                        // The bottom MB's skip flag (and the pair's field flag when it is coded)
                        // precede the skipped top MB's reconstruction.
                        self.begin_mb(addr);
                        let m = self.mb.addr;
                        self.st.mb_type[m] = if slice_type.is_b() {
                            MbType::BSkip
                        } else {
                            MbType::PSkip
                        };
                        next_skipped = self.decode_skip_flag(addr + 1);
                        if !next_skipped {
                            self.field = self.decode_field_flag(addr);
                        }
                    }
                    self.decode_skip(addr)?;
                } else {
                    if mbaff && top {
                        self.field = self.decode_field_flag(addr);
                    }
                    self.decode_mb(addr)?;
                }
                if top {
                    top_skipped = skipped;
                }
                if !(mbaff && top) {
                    let (cd, _) = self.cabac();
                    if cd.decode_terminate() != 0 {
                        return Ok(());
                    }
                }
                addr += 1;
            }
        }
    }

    /// Position of the macroblock at decoding address `addr`.
    fn pos(&self, addr: usize) -> MbPos {
        let w = self.st.mb_w;
        let (x, y) = if self.st.mbaff {
            let pair = addr / 2;
            (pair % w, 2 * (pair / w) + addr % 2)
        } else {
            (addr % w, addr / w)
        };
        MbPos {
            addr: y * w + x,
            x,
            y,
        }
    }

    /// Inferred mb_field_decoding_flag of the pair at decoding address `addr` (7.4.4): that of the
    /// left pair, else of the pair above, else frame.
    fn inferred_field(&self, addr: usize) -> bool {
        let p = self.pos(addr);
        let w = self.st.mb_w;
        let same = |a: usize| self.st.slice_num[a] == self.cx.slice_num;
        if p.x > 0 && same(p.addr - 1) {
            self.st.mb_field[p.addr - 1]
        } else if p.y >= 2 && same(p.addr - 2 * w) {
            self.st.mb_field[p.addr - 2 * w]
        } else {
            false
        }
    }

    /// CABAC mb_field_decoding_flag of the pair whose top MB has decoding address `addr`.
    fn decode_field_flag(&mut self, addr: usize) -> bool {
        let p = self.pos(addr);
        let w = self.st.mb_w;
        let cond =
            |a: usize| (self.st.slice_num[a] == self.cx.slice_num && self.st.mb_field[a]) as usize;
        let inc = if p.x > 0 { cond(p.addr - 1) } else { 0 }
            + if p.y >= 2 { cond(p.addr - 2 * w) } else { 0 };
        let (cd, ctxs) = self.cabac();
        cd.decode_decision(&mut ctxs[70 + inc]) != 0
    }

    /// CABAC mb_skip_flag of the macroblock at decoding address `addr`.
    fn decode_skip_flag(&mut self, addr: usize) -> bool {
        let saved = self.mb;
        self.mb = self.pos(addr);
        let a = self
            .mb_left()
            .filter(|&m| !self.st.mb_type[m].is_skip())
            .is_some() as u8;
        let b = self
            .mb_top()
            .filter(|&m| !self.st.mb_type[m].is_skip())
            .is_some() as u8;
        self.mb = saved;
        let is_b = self.cx.sh.slice_type.is_b();
        let (cd, ctxs) = self.cabac();
        if is_b {
            cd.decode_b_skip(&mut ctxs[..], a, b)
        } else {
            cd.decode_p_skip(&mut ctxs[..], a, b)
        }
    }

    fn reader(&mut self) -> &mut BitReader<'a> {
        match &mut self.ent {
            Entropy::Cavlc(r) => r,
            Entropy::Cabac(..) => unreachable!("CAVLC syntax read in a CABAC slice"),
        }
    }

    fn cabac(
        &mut self,
    ) -> (
        &mut CabacDecoder<'a>,
        &mut [CabacContext; NUM_CABAC_CONTEXTS],
    ) {
        match &mut self.ent {
            Entropy::Cabac(cd, ctxs) => (cd, ctxs),
            Entropy::Cavlc(_) => unreachable!("CABAC syntax read in a CAVLC slice"),
        }
    }

    /// Whether the current macroblock is a field macroblock.
    fn field_mb(&self) -> bool {
        self.field
    }

    /// MBAFF field macroblock: uses the field reference lists of its parity.
    fn mbaff_field(&self) -> bool {
        self.st.mbaff && self.field
    }

    /// Reference list `list` as seen by the current macroblock.
    fn refs(&self, list: usize) -> &'c [RefPic] {
        let cx = self.cx;
        match &cx.field_refs {
            Some(f) if self.field => &f[self.mb.y & 1][list],
            _ => &cx.refs[list],
        }
    }

    /// POC of the current picture, or of the current MB's field for MBAFF field macroblocks.
    fn cur_poc(&self) -> i32 {
        if self.mbaff_field() {
            self.cx.field_poc[self.mb.y & 1]
        } else {
            self.cx.cur_poc
        }
    }

    /// Parity of the current field picture or MBAFF field macroblock; `None` for frame MBs.
    fn field_parity(&self) -> Option<u8> {
        match self.st.structure {
            Structure::Field(p) => Some(p),
            Structure::Frame if self.field => Some((self.mb.y & 1) as u8),
            Structure::Frame => None,
        }
    }

    /// Number of active references of `list` for the current macroblock.
    fn num_refs(&self, list: usize) -> u32 {
        self.cx.sh.num_ref_idx_active[list] * if self.mbaff_field() { 2 } else { 1 }
    }

    fn is_cabac(&self) -> bool {
        matches!(self.ent, Entropy::Cabac(..))
    }

    // ----- neighbourhood -------------------------------------------------------------------

    fn begin_mb(&mut self, addr: usize) {
        self.mb = self.pos(addr);
        let a = self.mb.addr;
        if self.st.recycled {
            self.st.clear_mb(a);
        }
        self.partial_motion = false;
        self.st.slice_num[a] = self.cx.slice_num;
        self.st.mb_field[a] = self.field;
    }

    fn mb_avail(&self, addr: usize) -> bool {
        self.st.slice_num[addr] == self.cx.slice_num && addr != self.mb.addr
    }

    /// Neighbouring location (6.4.12): the macroblock covering the luma (or, with `chroma`, 4:2:0
    /// chroma) location `(xn, yn)` relative to the current macroblock's top-left sample, and the
    /// location `(xw, yw)` inside it. Availability is not checked.
    #[inline]
    fn locate(&self, xn: isize, yn: isize, chroma: bool) -> Option<(usize, usize, usize)> {
        // MB size `s = 1 << sh`; positions are non-negative once range-checked, so shifts and
        // masks stand in for division.
        let sh = if chroma { 3 } else { 4 };
        let s: isize = 1 << sh;
        let (w, h) = (self.st.mb_w as isize, self.st.mb_h as isize);
        let px = self.mb.x as isize * s + xn;
        if px < 0 || px >= w * s {
            return None;
        }
        let nx = (px >> sh) as usize;
        let xw = (px & (s - 1)) as usize;
        if !self.st.mbaff {
            let py = self.mb.y as isize * s + yn;
            if py < 0 || py >= h * s {
                return None;
            }
            let addr = (py >> sh) as usize * self.st.mb_w + nx;
            return Some((addr, xw, (py & (s - 1)) as usize));
        }
        self.locate_mbaff(yn, sh, nx, xw)
    }

    /// [`Self::locate`] in an MBAFF frame, for column `nx` and `xw` already found.
    #[inline(never)]
    fn locate_mbaff(
        &self,
        yn: isize,
        sh: u32,
        nx: usize,
        xw: usize,
    ) -> Option<(usize, usize, usize)> {
        let s: isize = 1 << sh;
        let h = self.st.mb_h as isize;
        // MBAFF: the location is geometric in the frame; a field MB of parity p covers the
        // frame lines `pair_top + p + 2k`.
        let my = self.mb.y as isize;
        let line = if self.field {
            (my & !1) * s + (my & 1) + 2 * yn
        } else {
            my * s + yn
        };
        if line < 0 || line >= h * s {
            return None;
        }
        let pair_row = (line / (2 * s)) as usize;
        let top = 2 * pair_row * self.st.mb_w + nx;
        let pair_field = if pair_row == self.mb.y / 2 && nx == self.mb.x {
            self.field
        } else {
            self.st.mb_field[top]
        };
        let l = line % (2 * s);
        let (bottom, yw) = if pair_field {
            (l % 2, l / 2)
        } else {
            (l / s, l % s)
        };
        Some((top + bottom as usize * self.st.mb_w, xw, yw as usize))
    }

    /// Index into the per-4x4 arrays of the block at `(xw, yw)` (samples) inside MB `addr`.
    fn b4_of(&self, addr: usize, xw: usize, yw: usize) -> usize {
        let (x, y) = self.mb_pos(addr);
        self.st.b4(x * 4 + xw / 4, y * 4 + yw / 4)
    }

    /// Column and row of macroblock `addr`. Neighbours lie within two rows of the current MB, so
    /// those rows are tried before dividing.
    #[inline]
    fn mb_pos(&self, addr: usize) -> (usize, usize) {
        let w = self.st.mb_w;
        let y = self.mb.y;
        let row = y * w;
        if addr >= row {
            if addr < row + w {
                return (addr - row, y);
            }
            if addr < row + 2 * w {
                return (addr - row - w, y + 1);
            }
        } else if addr + w >= row {
            return (addr + w - row, y - 1);
        } else if addr + 2 * w >= row {
            return (addr + 2 * w - row, y - 2);
        }
        (addr % w, addr / w)
    }

    /// Index into the per-chroma-4x4 arrays of the block at chroma `(xw, yw)` inside MB `addr`.
    fn c4_of(&self, addr: usize, xw: usize, yw: usize) -> usize {
        let (x, y) = self.mb_pos(addr);
        self.st.c4(x * 2 + xw / 4, y * 2 + yw / 4)
    }

    /// Available neighbouring macroblock covering luma `(xn, yn)` (outside the current MB).
    fn mb_nb(&self, xn: isize, yn: isize) -> Option<usize> {
        self.locate(xn, yn, false)
            .map(|n| n.0)
            .filter(|&a| self.mb_avail(a))
    }

    /// The available macroblock left of the current one.
    #[inline]
    fn mb_left(&self) -> Option<usize> {
        if self.st.mbaff {
            return self.mb_nb(-1, 0);
        }
        let a = self.mb.addr;
        (self.mb.x > 0 && self.st.slice_num[a - 1] == self.cx.slice_num).then(|| a - 1)
    }

    /// The available macroblock above the current one.
    #[inline]
    fn mb_top(&self) -> Option<usize> {
        if self.st.mbaff {
            return self.mb_nb(0, -1);
        }
        let (a, w) = (self.mb.addr, self.st.mb_w);
        (self.mb.y > 0 && self.st.slice_num[a - w] == self.cx.slice_num).then(|| a - w)
    }

    /// The available 4x4 block covering luma `(xn, yn)`: `(macroblock, 4x4 index)`. Blocks of the
    /// current MB are available when their bit is set in `done` (by `luma4x4BlkIdx`).
    fn blk(&self, xn: isize, yn: isize, done: u16) -> Option<(usize, usize)> {
        if !self.st.mbaff {
            // Without MBAFF the 4x4 block follows from absolute luma coordinates directly.
            let px = (self.mb.x * 16) as isize + xn;
            let py = (self.mb.y * 16) as isize + yn;
            if px < 0
                || py < 0
                || px >= (self.st.mb_w * 16) as isize
                || py >= (self.st.mb_h * 16) as isize
            {
                return None;
            }
            let (px, py) = (px as usize, py as usize);
            let a = (py >> 4) * self.st.mb_w + (px >> 4);
            if a == self.mb.addr {
                if done & (1 << BLK4_AT[(py & 15) >> 2][(px & 15) >> 2]) == 0 {
                    return None;
                }
            } else if !self.mb_avail(a) {
                return None;
            }
            return Some((a, self.st.b4(px >> 2, py >> 2)));
        }
        let (a, xw, yw) = self.locate(xn, yn, false)?;
        if a == self.mb.addr {
            if done & (1 << BLK4_AT[yw / 4][xw / 4]) == 0 {
                return None;
            }
        } else if !self.mb_avail(a) {
            return None;
        }
        Some((a, self.b4_of(a, xw, yw)))
    }

    /// Whether a macroblock's samples may be used for intra prediction (constrained intra pred).
    fn intra_ok(&self, addr: Option<usize>) -> bool {
        match addr {
            None => false,
            Some(a) if a == self.mb.addr => true,
            Some(a) => {
                self.mb_avail(a)
                    && !(self.cx.pps.constrained_intra_pred && !self.st.mb_type[a].is_intra())
            }
        }
    }

    /// Intra availability of the samples left of luma rows `y0..y0 + n` of the current MB. In MBAFF
    /// frames these rows can come from both macroblocks of the left pair.
    fn left_ok(&self, y0: usize, n: usize) -> bool {
        let rows = if self.st.mbaff { n } else { 1 };
        (y0..y0 + rows).all(|y| self.intra_ok(self.locate(-1, y as isize, false).map(|l| l.0)))
    }

    /// Intra availability `(left, top, top_right, top_left)` of the `n`x`n` luma block at MB-relative 4x4 `(bx, by)`.
    fn intra_edges(
        &self,
        bx: usize,
        by: usize,
        n4: usize,
        cur_blk: usize,
    ) -> (bool, bool, bool, bool) {
        let at = |xn: isize, yn: isize| -> bool {
            match self.locate(xn, yn, false) {
                // Inside the current MB: decoded iff earlier in block order.
                Some((a, xw, yw)) if a == self.mb.addr => BLK4_AT[yw / 4][xw / 4] < cur_blk,
                l => self.intra_ok(l.map(|l| l.0)),
            }
        };
        let (x, y) = ((bx * 4) as isize, (by * 4) as isize);
        let left = if bx == 0 {
            self.left_ok(by * 4, n4 * 4)
        } else {
            true
        };
        (
            left,
            at(x, y - 1),
            at(x + (n4 * 4) as isize, y - 1),
            at(x - 1, y - 1),
        )
    }

    /// Byte offset of the current MB's sample view in plane `c`, its line stride, and the MB's
    /// top-left `(x, y)` within that view. A field MB of an MBAFF frame sees its field's lines.
    fn view(&self, c: usize) -> (usize, usize, usize, usize) {
        let s = if c == 0 { 16 } else { 8 };
        let w = if c == 0 {
            self.st.width
        } else {
            self.st.width / 2
        };
        if self.mbaff_field() {
            let p = self.mb.y & 1;
            (p * w, 2 * w, self.mb.x * s, (self.mb.y / 2) * s)
        } else {
            (0, w, self.mb.x * s, self.mb.y * s)
        }
    }

    // ----- syntax helpers -------------------------------------------------------------------

    fn read_te(&mut self, range: u32) -> Result<i8> {
        let r = self.reader();
        let v = if range == 1 {
            u64::from(!r.read_bit()?)
        } else {
            r.read_ue()?
        };
        if v > range as u64 {
            return Err(bad("H.264 ref_idx out of range"));
        }
        Ok(v as i8)
    }

    // ----- macroblock decoding --------------------------------------------------------------

    fn decode_skip(&mut self, addr: usize) -> Result<()> {
        self.begin_mb(addr);
        let a = self.mb.addr;
        self.st.qp[a] = self.qp as u8;
        self.st.qpc[a] = self.chroma_qps(self.qp);
        self.last_dqp_nz = false;
        if self.cx.sh.slice_type.is_b() {
            self.st.mb_type[a] = MbType::BSkip;
            self.direct_motion(0b1111)?;
        } else {
            self.st.mb_type[a] = MbType::PSkip;
            let mv = self.p_skip_mv();
            let id = self.ref_id(0, 0)?;
            self.fill_motion(0, 0, 4, 4, 0, (0, mv), id, [0, 0]);
        }
        self.inter_predict_mb()?;
        Ok(())
    }

    fn chroma_qps(&self, qp: i32) -> [u8; 2] {
        let o = self.cx.pps.chroma_qp_index_offset;
        [chroma_qp(qp, o[0]), chroma_qp(qp, o[1])]
    }

    fn decode_mb(&mut self, addr: usize) -> Result<()> {
        self.begin_mb(addr);
        let slice_type = self.cx.sh.slice_type;
        let raw = self.read_mb_type()?;
        let intra_base = match slice_type {
            SliceType::I | SliceType::Si => 0,
            SliceType::P | SliceType::Sp => 5,
            SliceType::B => 23,
        };
        if slice_type == SliceType::Si {
            return Err(RagdollError::Unsupported(
                "H.264 SI slices are not supported".into(),
            ));
        }
        if raw >= intra_base {
            let t = raw - intra_base;
            if t > 25 {
                return Err(bad("H.264 mb_type out of range"));
            }
            return self.decode_intra_mb(t);
        }
        if slice_type.is_b() {
            self.decode_b_mb(raw)
        } else {
            self.decode_p_mb(raw)
        }
    }

    fn read_mb_type(&mut self) -> Result<u32> {
        let slice_type = self.cx.sh.slice_type;
        if !self.is_cabac() {
            return Ok(self.reader().read_ue()? as u32);
        }
        let left = self.mb_left();
        let top = self.mb_top();
        Ok(match slice_type {
            SliceType::I => {
                let c = |m: Option<usize>| {
                    m.is_some_and(|a| {
                        self.st.mb_type[a] != MbType::I4x4 && self.st.mb_type[a] != MbType::I8x8
                    }) as u8
                };
                let (ca, cb) = (c(left), c(top));
                let (cd, ctxs) = self.cabac();
                cd.decode_intra_mb_type(&mut ctxs[..], 3, ca, cb)
            }
            SliceType::B => {
                let c = |m: Option<usize>| {
                    m.is_some_and(|a| {
                        !matches!(self.st.mb_type[a], MbType::BSkip | MbType::BDirect16x16)
                    }) as usize
                };
                let inc = c(left) + c(top);
                let (cd, ctxs) = self.cabac();
                cabac_se::b_mb_type(cd, &mut ctxs[..], inc)
            }
            _ => {
                let (cd, ctxs) = self.cabac();
                cd.decode_p_mb_type(&mut ctxs[..])
            }
        })
    }

    fn read_transform_8x8(&mut self) -> Result<bool> {
        if !self.is_cabac() {
            return self.reader().read_bit();
        }
        let c = |m: Option<usize>| m.is_some_and(|a| self.st.t8x8[a]) as usize;
        let inc = c(self.mb_left()) + c(self.mb_top());
        let (cd, ctxs) = self.cabac();
        Ok(cabac_se::transform_8x8(cd, &mut ctxs[..], inc))
    }

    fn read_chroma_pred_mode(&mut self) -> Result<u8> {
        let m = if self.is_cabac() {
            let c = |m: Option<usize>| {
                m.is_some_and(|a| {
                    let t = self.st.mb_type[a];
                    t.is_intra() && t != MbType::IPcm && self.st.chroma_pred_mode[a] != 0
                }) as u8
            };
            let (a, b) = (c(self.mb_left()), c(self.mb_top()));
            let (cd, ctxs) = self.cabac();
            cd.decode_chroma_pred_mode(&mut ctxs[..], a, b)
        } else {
            self.reader().read_ue()? as u8
        };
        if m > 3 {
            return Err(bad("H.264 intra_chroma_pred_mode > 3"));
        }
        Ok(m)
    }

    fn read_pred_mode(&mut self, predicted: u8) -> Result<u8> {
        let (prev, rem) = if self.is_cabac() {
            let (cd, ctxs) = self.cabac();
            cd.decode_intra4x4_pred_mode(&mut ctxs[..])
        } else {
            let r = self.reader();
            if r.read_bit()? {
                (true, 0)
            } else {
                (false, r.read_bits(3)? as u8)
            }
        };
        Ok(if prev {
            predicted
        } else if rem < predicted {
            rem
        } else {
            rem + 1
        })
    }

    /// Predicted Intra4x4/Intra8x8 mode for the block whose top-left 4x4 is at MB-relative `(bx, by)` (8.3.1.1).
    fn predicted_intra_mode(&self, bx: usize, by: usize) -> u8 {
        let (x, y) = ((bx * 4) as isize, (by * 4) as isize);
        let fetch = |xn: isize, yn: isize| -> Option<i8> {
            let (addr, xw, yw) = self.locate(xn, yn, false)?;
            if addr != self.mb.addr {
                if !self.mb_avail(addr) {
                    return None;
                }
                let t = self.st.mb_type[addr];
                if !t.is_intra() && self.cx.pps.constrained_intra_pred {
                    return None;
                }
                if !matches!(t, MbType::I4x4 | MbType::I8x8) {
                    return Some(2);
                }
            }
            Some(self.st.intra_mode[self.b4_of(addr, xw, yw)])
        };
        match (fetch(x - 1, y), fetch(x, y - 1)) {
            (Some(a), Some(b)) => a.min(b) as u8,
            _ => 2,
        }
    }

    fn read_cbp(&mut self, intra: bool) -> Result<u8> {
        let mono = self.cx.sps.chroma_array_type() == 0 || self.cx.sps.chroma_array_type() == 3;
        if !self.is_cabac() {
            let code = self.reader().read_ue()? as usize;
            let entry = if mono {
                CBP_ME_MONO.get(code)
            } else {
                CBP_ME_420.get(code)
            };
            let (i, p) = *entry.ok_or_else(|| bad("H.264 coded_block_pattern out of range"))?;
            return Ok(if intra { i } else { p });
        }
        // Luma: condTermFlagN = 1 when the neighbouring 8x8 block has its CBP bit clear (skip counts as clear).
        let nb_bit = |xn: isize, yn: isize| -> u8 {
            let Some((a, xw, yw)) = self.locate(xn, yn, false) else {
                return 1;
            };
            if !self.mb_avail(a) {
                return 1;
            }
            match self.st.mb_type[a] {
                MbType::IPcm => 1,
                t if t.is_skip() => 0,
                _ => (self.st.cbp[a] >> ((yw / 8) * 2 + xw / 8)) & 1,
            }
        };
        let la = [nb_bit(-1, 0), nb_bit(-1, 8)];
        let tb = [nb_bit(0, -1), nb_bit(8, -1)];
        let chroma_of = |m: Option<usize>| -> u8 {
            match m {
                None => 0,
                Some(a) if self.st.mb_type[a] == MbType::IPcm => 2,
                Some(a) if self.st.mb_type[a].is_skip() => 0,
                Some(a) => self.st.cbp[a] >> 4,
            }
        };
        let (left, top) = (self.mb_left(), self.mb_top());
        let (ca, cb) = (chroma_of(left), chroma_of(top));
        let (cd, ctxs) = self.cabac();
        let mut cbp = 0u8;
        for b8 in 0..4 {
            let a_bit = match b8 {
                0 => la[0],
                1 => cbp & 1,
                2 => la[1],
                _ => (cbp >> 2) & 1,
            };
            let b_bit = match b8 {
                0 => tb[0],
                1 => tb[1],
                2 => cbp & 1,
                _ => (cbp >> 1) & 1,
            };
            let bit = cd.decode_cbp_luma_bit(&mut ctxs[..], (a_bit == 0) as u8, (b_bit == 0) as u8);
            cbp |= bit << b8;
        }
        if !mono {
            cbp |= cd.decode_cbp_chroma(&mut ctxs[..], ca, cb) << 4;
        }
        Ok(cbp)
    }

    fn read_qp_delta(&mut self) -> Result<()> {
        let d = if self.is_cabac() {
            let nz = self.last_dqp_nz as u8;
            let (cd, ctxs) = self.cabac();
            cd.decode_mb_qp_delta(&mut ctxs[..], nz)
        } else {
            self.reader().read_se()? as i32
        };
        if !(-26..=25).contains(&d) {
            return Err(bad("H.264 mb_qp_delta out of range"));
        }
        self.last_dqp_nz = d != 0;
        self.qp = (self.qp + d + 52) % 52;
        Ok(())
    }

    fn set_qp(&mut self) {
        let a = self.mb.addr;
        self.st.qp[a] = self.qp as u8;
        self.st.qpc[a] = self.chroma_qps(self.qp);
    }

    // ----- intra macroblocks ----------------------------------------------------------------

    fn decode_intra_mb(&mut self, t: u32) -> Result<()> {
        let a = self.mb.addr;
        if t == 25 {
            return self.decode_pcm();
        }
        let mut modes = [0u8; 16];
        let mut t8 = false;
        let (cbp, kind) = if t == 0 {
            if self.cx.pps.transform_8x8_mode {
                t8 = self.read_transform_8x8()?;
            }
            self.st.t8x8[a] = t8;
            self.st.mb_type[a] = if t8 { MbType::I8x8 } else { MbType::I4x4 };
            if t8 {
                for b8 in 0..4 {
                    let (bx, by) = ((b8 & 1) * 2, (b8 >> 1) * 2);
                    let pred = self.predicted_intra_mode(bx, by);
                    let m = self.read_pred_mode(pred)?;
                    modes[b8] = m;
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let i = self.st.b4(self.mb.x * 4 + bx + dx, self.mb.y * 4 + by + dy);
                        self.st.intra_mode[i] = m as i8;
                    }
                }
            } else {
                for (blk, slot) in modes.iter_mut().enumerate() {
                    let (bx, by) = BLK4_XY[blk];
                    let pred = self.predicted_intra_mode(bx, by);
                    let m = self.read_pred_mode(pred)?;
                    *slot = m;
                    let i = self.st.b4(self.mb.x * 4 + bx, self.mb.y * 4 + by);
                    self.st.intra_mode[i] = m as i8;
                }
            }
            let chroma = if self.cx.sps.chroma_array_type() != 0 {
                self.read_chroma_pred_mode()?
            } else {
                0
            };
            self.st.chroma_pred_mode[a] = chroma;
            let cbp = self.read_cbp(true)?;
            (cbp, self.st.mb_type[a])
        } else {
            let pred = ((t - 1) % 4) as u8;
            let cbp_c = ((t - 1) / 4 % 3) as u8;
            let cbp_l = if t >= 13 { 0x0F } else { 0 };
            self.st.mb_type[a] = MbType::I16x16(pred);
            let chroma = if self.cx.sps.chroma_array_type() != 0 {
                self.read_chroma_pred_mode()?
            } else {
                0
            };
            self.st.chroma_pred_mode[a] = chroma;
            (cbp_l | (cbp_c << 4), MbType::I16x16(pred))
        };
        self.st.cbp[a] = cbp;
        let is_i16 = matches!(kind, MbType::I16x16(_));
        let mut res = self.take_residual();
        if cbp != 0 || is_i16 {
            self.read_qp_delta()?;
            self.set_qp();
            self.read_residual(cbp, is_i16, t8, true, &mut res)?;
        } else {
            self.last_dqp_nz = false;
            self.set_qp();
        }
        self.reconstruct_intra(kind, &modes, &mut res)?;
        self.res = Some(res);
        Ok(())
    }

    /// The zeroed residual buffer; a macroblock that fails part-way drops it, dirty.
    fn take_residual(&mut self) -> Box<Residual> {
        let mut res = self.res.take().unwrap_or_else(|| Box::new(Residual::new()));
        debug_assert!(res.is_clear());
        res.luma_coded = 0;
        res.chroma_coded = [0; 2];
        res
    }

    fn decode_pcm(&mut self) -> Result<()> {
        let a = self.mb.addr;
        let mut samples = [0u8; 384];
        let n = if self.cx.sps.chroma_array_type() == 0 {
            256
        } else {
            384
        };
        match &mut self.ent {
            Entropy::Cavlc(r) => {
                r.align_to_byte();
                for s in samples.iter_mut().take(n) {
                    *s = r.read_bits(8)? as u8;
                }
            }
            Entropy::Cabac(cd, _) => cd.align_for_pcm(&mut samples[..n])?,
        }
        self.st.mb_type[a] = MbType::IPcm;
        self.st.cbp[a] = 0x2F;
        self.st.cbf_dc[a] = 7;
        self.st.qp[a] = 0;
        self.st.qpc[a] = self.chroma_qps(0);
        self.last_dqp_nz = false;
        let (x4, y4) = (self.mb.x * 4, self.mb.y * 4);
        for by in 0..4 {
            for bx in 0..4 {
                let i = self.st.b4(x4 + bx, y4 + by);
                self.st.nnz[i] = 16;
                self.st.nz[i] = true;
            }
        }
        for p in 0..2 {
            for by in 0..2 {
                for bx in 0..2 {
                    let i = self.st.c4(self.mb.x * 2 + bx, self.mb.y * 2 + by);
                    self.st.nnz_c[p][i] = 16;
                }
            }
        }
        let (off, ls, px, py) = self.view(0);
        for y in 0..16 {
            let o = off + (py + y) * ls + px;
            self.st.planes[0][o..o + 16].copy_from_slice(&samples[y * 16..y * 16 + 16]);
        }
        if n == 384 {
            let (off, cs, cx, cy) = self.view(1);
            for p in 0..2 {
                for y in 0..8 {
                    let src = &samples[256 + p * 64 + y * 8..256 + p * 64 + y * 8 + 8];
                    let o = off + (cy + y) * cs + cx;
                    self.st.planes[1 + p][o..o + 8].copy_from_slice(src);
                }
            }
        }
        Ok(())
    }

    fn reconstruct_intra(
        &mut self,
        kind: MbType,
        modes: &[u8; 16],
        res: &mut Residual,
    ) -> Result<()> {
        let (off, ls, px, py) = self.view(0);
        match kind {
            MbType::I4x4 => {
                for (blk, &mode) in modes.iter().enumerate() {
                    let (bx, by) = BLK4_XY[blk];
                    let (l, t, tr, tl) = self.intra_edges(bx, by, 1, blk);
                    let (x, y) = (px + bx * 4, py + by * 4);
                    let plane = &mut self.st.planes[0][off..];
                    let e = Edges::gather(plane, ls, x, y, 4, l, t, tr && t, tl);
                    let mut out = [0u8; 16];
                    intra::predict_nxn(4, mode, &e, &mut out);
                    for r in 0..4 {
                        plane[(y + r) * ls + x..(y + r) * ls + x + 4]
                            .copy_from_slice(&out[r * 4..r * 4 + 4]);
                    }
                    if res.luma_coded & (1 << blk) != 0 {
                        recon::idct4_add(&mut plane[y * ls + x..], ls, &mut res.luma[blk]);
                    }
                }
            }
            MbType::I8x8 => {
                for b8 in 0..4 {
                    let (bx, by) = ((b8 & 1) * 2, (b8 >> 1) * 2);
                    let (l, t, tr, tl) = self.intra_edges(bx, by, 2, b8 * 4);
                    let (x, y) = (px + bx * 4, py + by * 4);
                    let plane = &mut self.st.planes[0][off..];
                    let e = Edges::gather(plane, ls, x, y, 8, l, t, tr && t, tl).filter8();
                    let mut out = [0u8; 64];
                    intra::predict_nxn(8, modes[b8], &e, &mut out);
                    for r in 0..8 {
                        plane[(y + r) * ls + x..(y + r) * ls + x + 8]
                            .copy_from_slice(&out[r * 8..r * 8 + 8]);
                    }
                    if res.luma_coded & (0xF << (b8 * 4)) != 0 {
                        recon::idct8_add(&mut plane[y * ls + x..], ls, &mut res.luma8[b8]);
                    }
                }
            }
            MbType::I16x16(mode) => {
                let mb_top = self.intra_ok(self.mb_top());
                let mb_left = self.left_ok(0, 16);
                let mb_tl = self.intra_ok(self.mb_nb(-1, -1));
                if !intra::block_mode_ok(mode, mb_left, mb_top, mb_tl) {
                    return Err(bad("H.264 Intra16x16 mode uses unavailable samples"));
                }
                let mut out = [0u8; 256];
                let plane = &mut self.st.planes[0][off..];
                intra::predict_16x16(plane, ls, px, py, mode, mb_left, mb_top, mb_tl, &mut out);
                for r in 0..16 {
                    plane[(py + r) * ls + px..(py + r) * ls + px + 16]
                        .copy_from_slice(&out[r * 16..r * 16 + 16]);
                }
                for blk in 0..16 {
                    if res.luma_coded & (1 << blk) != 0 {
                        let (bx, by) = BLK4_XY[blk];
                        let o = (py + by * 4) * ls + px + bx * 4;
                        recon::idct4_add(&mut plane[o..], ls, &mut res.luma[blk]);
                    }
                }
            }
            _ => unreachable!(),
        }
        if self.cx.sps.chroma_array_type() != 0 {
            let mb_top = self.intra_ok(self.mb_top());
            let left = [self.left_ok(0, 8), self.left_ok(8, 8)];
            let mb_tl = self.intra_ok(self.mb_nb(-1, -1));
            let mode = self.st.chroma_pred_mode[self.mb.addr];
            let as_block = [2, 1, 0, 3][mode as usize];
            if !intra::block_mode_ok(as_block, left[0] && left[1], mb_top, mb_tl) {
                return Err(bad("H.264 chroma intra mode uses unavailable samples"));
            }
            let (off, cs, cx, cy) = self.view(1);
            for p in 0..2 {
                let mut out = [0u8; 64];
                let plane = &mut self.st.planes[1 + p][off..];
                intra::predict_chroma(plane, cs, cx, cy, mode, left, mb_top, mb_tl, &mut out);
                for r in 0..8 {
                    let o = (cy + r) * cs + cx;
                    plane[o..o + 8].copy_from_slice(&out[r * 8..r * 8 + 8]);
                }
            }
            self.add_chroma_residual(res);
        }
        Ok(())
    }

    fn add_chroma_residual(&mut self, res: &mut Residual) {
        let (off, cs, cx, cy) = self.view(1);
        for p in 0..2 {
            for blk in 0..4 {
                if res.chroma_coded[p] & (1 << blk) != 0 {
                    let o = off + (cy + (blk >> 1) * 4) * cs + cx + (blk & 1) * 4;
                    recon::idct4_add(&mut self.st.planes[1 + p][o..], cs, &mut res.chroma[p][blk]);
                }
            }
        }
    }

    // ----- residual -------------------------------------------------------------------------

    /// Neighbours A and B of the 4x4 block at MB-relative 4x4 `(bx, by)` of plane `c` (0 luma,
    /// 1 chroma): each `(macroblock, block index)` when available.
    #[allow(clippy::type_complexity)]
    fn nb_blocks(&self, c: usize, bx: usize, by: usize) -> [Option<(usize, usize)>; 2] {
        let chroma = c > 0;
        if !self.st.mbaff {
            // Without MBAFF the left and top blocks follow from block coordinates directly.
            // Blocks per MB side: 1 << sh.
            let sh = if chroma { 1 } else { 2 };
            let w4 = self.st.mb_w << sh;
            let i = ((self.mb.y << sh) + by) * w4 + (self.mb.x << sh) + bx;
            let left = if bx > 0 {
                Some(self.mb.addr)
            } else {
                self.mb_left()
            };
            let top = if by > 0 {
                Some(self.mb.addr)
            } else {
                self.mb_top()
            };
            return [left.map(|a| (a, i - 1)), top.map(|a| (a, i - w4))];
        }
        let (x, y) = ((bx * 4) as isize, (by * 4) as isize);
        [(x - 1, y), (x, y - 1)].map(|(xn, yn)| {
            let (a, xw, yw) = self.locate(xn, yn, chroma)?;
            if a != self.mb.addr && !self.mb_avail(a) {
                return None;
            }
            Some((
                a,
                if chroma {
                    self.c4_of(a, xw, yw)
                } else {
                    self.b4_of(a, xw, yw)
                },
            ))
        })
    }

    /// CAVLC `nC` for the luma 4x4 block at MB-relative 4x4 `(bx, by)` (9.2.1).
    fn nc_luma(&self, bx: usize, by: usize) -> u8 {
        let [a, b] = self
            .nb_blocks(0, bx, by)
            .map(|n| n.map(|(_, i)| self.st.nnz[i]));
        combine_nc(a, b)
    }

    fn nc_chroma(&self, p: usize, bx: usize, by: usize) -> u8 {
        let [a, b] = self
            .nb_blocks(1, bx, by)
            .map(|n| n.map(|(_, i)| self.st.nnz_c[p][i]));
        combine_nc(a, b)
    }

    /// CABAC coded_block_flag ctxIdxInc for a luma 4x4 block (9.3.3.1.1.9).
    fn cbf_luma_inc(&self, bx: usize, by: usize, intra: bool) -> u8 {
        let [a, b] = self.nb_blocks(0, bx, by).map(|n| match n {
            Some((m, _)) if m != self.mb.addr && self.st.mb_type[m] == MbType::IPcm => 1,
            Some((_, i)) => self.st.nz[i] as u8,
            None => intra as u8,
        });
        a + 2 * b
    }

    fn cbf_chroma_ac_inc(&self, p: usize, bx: usize, by: usize, intra: bool) -> u8 {
        let [a, b] = self.nb_blocks(1, bx, by).map(|n| match n {
            Some((m, _)) if m != self.mb.addr && self.st.mb_type[m] == MbType::IPcm => 1,
            Some((_, i)) => (self.st.nnz_c[p][i] > 0) as u8,
            None => intra as u8,
        });
        a + 2 * b
    }

    /// condTermFlagN for a DC block (`bit` 0 luma, 1 Cb, 2 Cr) of a neighbouring macroblock.
    fn cbf_dc(&self, m: Option<usize>, bit: u8, intra: bool) -> u8 {
        match m {
            None => intra as u8,
            Some(a) => {
                let t = self.st.mb_type[a];
                let has_dc = if bit == 0 {
                    matches!(t, MbType::I16x16(_))
                } else {
                    self.st.cbp[a] >> 4 != 0
                };
                if t == MbType::IPcm {
                    1
                } else if t.is_skip() || !has_dc {
                    0
                } else {
                    (self.st.cbf_dc[a] >> bit) & 1
                }
            }
        }
    }

    /// Reads one residual block. CAVLC: `nc`; CABAC: `cat` and `inc`. Returns total_coeff.
    fn read_block(
        &mut self,
        out: &mut [i16],
        nc: u8,
        chroma_dc: bool,
        cat: usize,
        inc: u8,
    ) -> Result<usize> {
        let field = self.field_mb();
        match &mut self.ent {
            Entropy::Cavlc(r) => decode_cavlc_block(r, out, nc, chroma_dc),
            Entropy::Cabac(cd, ctxs) => Ok(cd
                .decode_residual_block(&mut ctxs[..], cat, inc, out, field)
                .1),
        }
    }

    fn read_residual(
        &mut self,
        cbp: u8,
        is_i16: bool,
        t8: bool,
        intra: bool,
        res: &mut Residual,
    ) -> Result<()> {
        let field = self.field_mb();
        let (scan4, scan8) = if field {
            (&FIELD_SCAN4X4, &FIELD_SCAN8X8)
        } else {
            (&SCAN4X4, &SCAN8X8)
        };
        let a = self.mb.addr;
        let qp = self.qp;
        let (mx4, my4) = (self.mb.x * 4, self.mb.y * 4);
        let lst = if intra { 0 } else { 3 };
        let cabac = self.is_cabac();
        let dq = self.cx.dq;
        let mut coeffs = ResidualCoeffs::new();

        if is_i16 {
            let mut sc = [0i16; 16];
            let nc = if cabac { 0 } else { self.nc_luma(0, 0) };
            let inc = if cabac {
                self.cbf_dc(self.mb_left(), 0, true) + 2 * self.cbf_dc(self.mb_top(), 0, true)
            } else {
                0
            };
            let n = self.read_block(&mut sc, nc, false, 0, inc)?;
            if n > 0 {
                self.st.cbf_dc[a] |= 1;
            }
            let mut c = [0i32; 16];
            for k in 0..16 {
                c[scan4[k]] = sc[k] as i32;
            }
            let dc = recon::luma_dc(dq, qp, &c);
            for blk in 0..16 {
                let (bx, by) = BLK4_XY[blk];
                res.luma[blk][0] = dc[by * 4 + bx];
                if dc[by * 4 + bx] != 0 {
                    res.luma_coded |= 1 << blk;
                }
            }
        }

        for b8 in 0..4 {
            if cbp & (1 << b8) == 0 {
                continue;
            }
            if t8 {
                let l8 = if intra { 0 } else { 1 };
                if cabac {
                    let (cd, ctxs) = self.cabac();
                    let total =
                        cd.decode_residual_coeffs(&mut ctxs[..], 5, 0, 64, field, &mut coeffs);
                    let (bx, by) = BLK4_XY[b8 * 4];
                    let (j0, j1) = (
                        self.st.b4(mx4 + bx, my4 + by),
                        self.st.b4(mx4 + bx, my4 + by + 1),
                    );
                    for j in [j0, j0 + 1, j1, j1 + 1] {
                        self.st.nnz[j] = total as u8;
                        self.st.nz[j] = total > 0;
                    }
                    if total > 0 {
                        dq.scale8_coeffs(l8, qp, &coeffs, total, scan8, &mut res.luma8[b8]);
                        res.luma_coded |= 0xF << (b8 * 4);
                    }
                    continue;
                }
                let mut sc = [0i16; 64];
                let mut total = 0;
                {
                    for i in 0..4 {
                        let blk = b8 * 4 + i;
                        let (bx, by) = BLK4_XY[blk];
                        let nc = self.nc_luma(bx, by);
                        let mut sub = [0i16; 16];
                        let n = {
                            let r = self.reader();
                            decode_cavlc_block(r, &mut sub, nc, false)?
                        };
                        let j = self.st.b4(mx4 + bx, my4 + by);
                        self.st.nnz[j] = n as u8;
                        total += n;
                        for k in 0..16 {
                            sc[4 * k + i] = sub[k];
                        }
                    }
                    for i in 0..4 {
                        let (bx, by) = BLK4_XY[b8 * 4 + i];
                        let j = self.st.b4(mx4 + bx, my4 + by);
                        self.st.nz[j] = total > 0;
                    }
                }
                if total > 0 {
                    dq.scale8_block(l8, qp, &sc, scan8, &mut res.luma8[b8]);
                    res.luma_coded |= 0xF << (b8 * 4);
                }
                continue;
            }
            for i in 0..4 {
                let blk = b8 * 4 + i;
                let (bx, by) = BLK4_XY[blk];
                let (x4, y4) = (mx4 + bx, my4 + by);
                let max = if is_i16 { 15 } else { 16 };
                // Intra_16x16 blocks keep their DC at position 0 (scan offset 1).
                let out = &mut res.luma[blk];
                let n = if cabac {
                    let inc = self.cbf_luma_inc(bx, by, intra);
                    let cat = if is_i16 { 1 } else { 2 };
                    let (cd, ctxs) = self.cabac();
                    let n =
                        cd.decode_residual_coeffs(&mut ctxs[..], cat, inc, max, field, &mut coeffs);
                    dq.scale4_coeffs(lst, qp, &coeffs, n, 16 - max, scan4, out);
                    n
                } else {
                    let nc = self.nc_luma(bx, by);
                    let r = self.reader();
                    let n = decode_cavlc_coeffs(r, max, nc, &mut coeffs)?;
                    dq.scale4_coeffs(lst, qp, &coeffs, n, 16 - max, scan4, out);
                    n
                };
                let j = self.st.b4(x4, y4);
                self.st.nnz[j] = n as u8;
                self.st.nz[j] = n > 0;
                if n > 0 {
                    res.luma_coded |= 1 << blk;
                }
            }
        }

        let cbp_c = cbp >> 4;
        if self.cx.sps.chroma_array_type() == 0 || cbp_c == 0 {
            return Ok(());
        }
        let qpc = self.st.qpc[a];
        let mut dcs = [[0i32; 4]; 2];
        for p in 0..2 {
            let mut sc = [0i16; 4];
            let inc = if cabac {
                let bit = 1 + p as u8;
                self.cbf_dc(self.mb_left(), bit, intra) + 2 * self.cbf_dc(self.mb_top(), bit, intra)
            } else {
                0
            };
            let n = self.read_block(&mut sc, 0, true, 3, inc)?;
            if n > 0 {
                self.st.cbf_dc[a] |= 2 << p;
            }
            let c = [sc[0] as i32, sc[1] as i32, sc[2] as i32, sc[3] as i32];
            dcs[p] = recon::chroma_dc(dq, lst + 1 + p, qpc[p] as i32, &c);
        }
        for p in 0..2 {
            for blk in 0..4 {
                res.chroma[p][blk][0] = dcs[p][blk];
                if dcs[p][blk] != 0 {
                    res.chroma_coded[p] |= 1 << blk;
                }
            }
        }
        if cbp_c & 2 == 0 {
            return Ok(());
        }
        let (cx4, cy4) = (self.mb.x * 2, self.mb.y * 2);
        for p in 0..2 {
            for blk in 0..4 {
                let (bx, by) = (blk & 1, blk >> 1);
                let (x4, y4) = (cx4 + bx, cy4 + by);
                let out = &mut res.chroma[p][blk];
                let n = if cabac {
                    let inc = self.cbf_chroma_ac_inc(p, bx, by, intra);
                    let (cd, ctxs) = self.cabac();
                    let n =
                        cd.decode_residual_coeffs(&mut ctxs[..], 4, inc, 15, field, &mut coeffs);
                    dq.scale4_coeffs(lst + 1 + p, qpc[p] as i32, &coeffs, n, 1, scan4, out);
                    n
                } else {
                    let nc = self.nc_chroma(p, bx, by);
                    let r = self.reader();
                    let n = decode_cavlc_coeffs(r, 15, nc, &mut coeffs)?;
                    dq.scale4_coeffs(lst + 1 + p, qpc[p] as i32, &coeffs, n, 1, scan4, out);
                    n
                };
                let j = self.st.c4(x4, y4);
                self.st.nnz_c[p][j] = n as u8;
                if n > 0 {
                    res.chroma_coded[p] |= 1 << blk;
                }
            }
        }
        Ok(())
    }

    // ----- inter macroblocks ----------------------------------------------------------------

    fn ref_id(&self, list: usize, idx: i8) -> Result<u32> {
        if idx < 0 {
            return Ok(NO_PIC);
        }
        self.refs(list)
            .get(idx as usize)
            .map(|r| r.id)
            .ok_or_else(|| bad("H.264 ref_idx beyond the reference list"))
    }

    /// Writes motion for the MB-relative 4x4 rectangle `(bx, by, w, h)` of one list.
    #[allow(clippy::too_many_arguments)]
    fn fill_motion(
        &mut self,
        bx: usize,
        by: usize,
        w: usize,
        h: usize,
        list: usize,
        m: Mv,
        pic: u32,
        mvd: [u8; 2],
    ) {
        // Fixed-size stores per width: a slice fill of a few elements becomes a memset call.
        #[allow(clippy::too_many_arguments)]
        fn rect<const N: usize>(
            st: &mut PicState,
            list: usize,
            i: usize,
            h: usize,
            m: Mv,
            pic: u32,
            mvd: [u8; 2],
        ) {
            let s = st.mb_w * 4;
            let rows = &mut st.blocks[i..i + (h - 1) * s + N];
            for r in 0..h {
                for b in &mut rows[r * s..r * s + N] {
                    b.ref_idx[list] = m.0;
                    b.mv[list] = m.1;
                    b.ref_pic[list] = pic;
                    b.mvd[list] = mvd;
                }
            }
        }
        self.partial_motion |= (bx, by, w, h) != (0, 0, 4, 4);
        let i = self.st.b4(self.mb.x * 4 + bx, self.mb.y * 4 + by);
        match w {
            1 => rect::<1>(self.st, list, i, h, m, pic, mvd),
            2 => rect::<2>(self.st, list, i, h, m, pic, mvd),
            3 => rect::<3>(self.st, list, i, h, m, pic, mvd),
            _ => rect::<4>(self.st, list, i, h, m, pic, mvd),
        }
    }

    /// Motion of the available block covering luma `(xn, yn)` relative to the current MB, in the
    /// current MB's frame/field units (8.4.1.3.2): an MBAFF neighbour of the other kind has its
    /// reference index and vertical vector rescaled.
    /// [`Self::neighbour`] for both lists, locating the block once.
    fn neighbour_pair(&self, xn: isize, yn: isize, done: u16) -> [Option<Mv>; 2] {
        let Some((a, i)) = self.blk(xn, yn, done) else {
            return [None, None];
        };
        [0, 1].map(|list| {
            let (r, mv) = (self.st.blocks[i].ref_idx[list], self.st.blocks[i].mv[list]);
            if r < 0 || !self.st.mbaff {
                return Some((r, mv));
            }
            Some(match (self.field, self.st.mb_field[a]) {
                (true, false) => (r * 2, [mv[0], mv[1] / 2]),
                (false, true) => (r >> 1, [mv[0], mv[1].wrapping_mul(2)]),
                _ => (r, mv),
            })
        })
    }

    fn neighbour(&self, list: usize, xn: isize, yn: isize, done: u16) -> Option<Mv> {
        let (a, i) = self.blk(xn, yn, done)?;
        let (r, mv) = (self.st.blocks[i].ref_idx[list], self.st.blocks[i].mv[list]);
        if r < 0 || !self.st.mbaff {
            return Some((r, mv));
        }
        Some(match (self.field, self.st.mb_field[a]) {
            (true, false) => (r * 2, [mv[0], mv[1] / 2]),
            (false, true) => (r >> 1, [mv[0], mv[1].wrapping_mul(2)]),
            _ => (r, mv),
        })
    }

    /// Motion vector predictor (8.4.1.3) for the MB-relative 4x4 rectangle `(bx, by, w, h)`.
    #[allow(clippy::too_many_arguments)]
    fn mvp(
        &self,
        list: usize,
        bx: usize,
        by: usize,
        w: usize,
        refidx: i8,
        part: Option<(Part, usize)>,
        done: u16,
    ) -> [i16; 2] {
        let (x, y) = ((bx * 4) as isize, (by * 4) as isize);
        let a = self.neighbour(list, x - 1, y, done);
        let b = self.neighbour(list, x, y - 1, done);
        let c = self
            .neighbour(list, x + (w * 4) as isize, y - 1, done)
            .or_else(|| self.neighbour(list, x - 1, y - 1, done));
        mvp_from(a, b, c, refidx, part)
    }

    fn p_skip_mv(&self) -> [i16; 2] {
        let [a, b, c] = self.mb_neighbours()[0];
        match (a, b) {
            (None, _) | (_, None) => [0, 0],
            (Some((0, [0, 0])), _) | (_, Some((0, [0, 0]))) => [0, 0],
            _ => mvp_from(a, b, c, 0, None),
        }
    }

    /// Neighbours A, B and C (D where C is unavailable) of the whole macroblock in both lists,
    /// as for a 16x16 partition's predictor (8.4.1.3.2).
    fn mb_neighbours(&self) -> [[Option<Mv>; 3]; 2] {
        if self.st.mbaff {
            let a = self.neighbour_pair(-1, 0, 0);
            let b = self.neighbour_pair(0, -1, 0);
            let c = self.neighbour_pair(16, -1, 0);
            // C falls back to D when unavailable; availability is the same for both lists.
            let c = if c[0].is_none() {
                self.neighbour_pair(-1, -1, 0)
            } else {
                c
            };
            return [[a[0], b[0], c[0]], [a[1], b[1], c[1]]];
        }
        // Without MBAFF the neighbours are the adjacent macroblocks' corner blocks.
        let (x, y, w, addr) = (self.mb.x, self.mb.y, self.st.mb_w, self.mb.addr);
        let i = self.st.b4(x * 4, y * 4);
        let s = w * 4;
        let st = &*self.st;
        let at = |i: usize| {
            let b = st.blocks[i];
            [0, 1].map(|l| Some((b.ref_idx[l], b.mv[l])))
        };
        let none = [None, None];
        let a = if x > 0 && self.mb_avail(addr - 1) {
            at(i - 1)
        } else {
            none
        };
        let (b, c) = if y > 0 {
            let b = if self.mb_avail(addr - w) {
                at(i - s)
            } else {
                none
            };
            let c = if x + 1 < w && self.mb_avail(addr - w + 1) {
                at(i - s + 4)
            } else if x > 0 && self.mb_avail(addr - w - 1) {
                at(i - s - 1)
            } else {
                none
            };
            (b, c)
        } else {
            (none, none)
        };
        [[a[0], b[0], c[0]], [a[1], b[1], c[1]]]
    }

    fn read_ref_idx(&mut self, list: usize, bx: usize, by: usize) -> Result<i8> {
        let n = self.num_refs(list);
        if !self.is_cabac() {
            return self.read_te(n - 1);
        }
        let (x, y) = ((bx * 4) as isize, (by * 4) as isize);
        let is_b = self.cx.sh.slice_type.is_b();
        let cond = |xn: isize, yn: isize| -> u8 {
            let Some((addr, i)) = self.blk(xn, yn, 0xFFFF) else {
                return 0;
            };
            let t = self.st.mb_type[addr];
            if t.is_skip() || t.is_intra() {
                return 0;
            }
            if is_b && self.st.direct[i] {
                return 0;
            }
            // A field neighbour of a frame MB counts reference indices in fields (9.3.3.1.1.6).
            let thresh = if self.st.mbaff && !self.field && self.st.mb_field[addr] {
                1
            } else {
                0
            };
            (self.st.blocks[i].ref_idx[list] > thresh) as u8
        };
        let (ca, cb) = (cond(x - 1, y), cond(x, y - 1));
        let (cd, ctxs) = self.cabac();
        let v = cd.decode_ref_idx(&mut ctxs[..], ca, cb);
        if v as u32 >= n {
            return Err(bad("H.264 ref_idx out of range"));
        }
        Ok(v as i8)
    }

    fn read_mvd(&mut self, list: usize, bx: usize, by: usize) -> Result<([i16; 2], [u8; 2])> {
        if !self.is_cabac() {
            let r = self.reader();
            let x = r.read_se()?;
            let y = r.read_se()?;
            if !(-8192..=8191).contains(&x) || !(-2048..=2047).contains(&y) {
                return Err(bad("H.264 mvd out of range"));
            }
            return Ok(([x as i16, y as i16], [0, 0]));
        }
        let (x, y) = ((bx * 4) as isize, (by * 4) as isize);
        let absn = |xn: isize, yn: isize, comp: usize| -> u32 {
            let Some((addr, i)) = self.blk(xn, yn, 0xFFFF) else {
                return 0;
            };
            let v = self.st.blocks[i].mvd[list][comp] as u32;
            // Vertical components between frame and field MBs are rescaled (9.3.3.1.1.7).
            match (comp, self.st.mbaff, self.field, self.st.mb_field[addr]) {
                (1, true, false, true) => v * 2,
                (1, true, true, false) => v / 2,
                _ => v,
            }
        };
        let sums = [0, 1].map(|comp| absn(x - 1, y, comp) + absn(x, y - 1, comp));
        let mut mvd = [0i16; 2];
        let mut amvd = [0u8; 2];
        for comp in 0..2 {
            let sum = sums[comp];
            let (cd, ctxs) = self.cabac();
            mvd[comp] = cd.decode_mvd(&mut ctxs[..], comp, sum);
            amvd[comp] = mvd[comp].unsigned_abs().min(255) as u8;
        }
        Ok((mvd, amvd))
    }

    fn decode_p_mb(&mut self, raw: u32) -> Result<()> {
        let a = self.mb.addr;
        if raw > 4 {
            return Err(bad("H.264 P mb_type out of range"));
        }
        if raw <= 2 {
            let part = [Part::P16x16, Part::P16x8, Part::P8x16][raw as usize];
            self.st.mb_type[a] = MbType::Inter(part);
            self.decode_partitions(part, [PRED_L0, PRED_L0])?;
            return self.finish_inter(true);
        }
        self.st.mb_type[a] = MbType::Inter(Part::P8x8);
        let mut subs = [0usize; 4];
        for s in &mut subs {
            *s = if self.is_cabac() {
                let (cd, ctxs) = self.cabac();
                cd.decode_p_sub_mb_type(&mut ctxs[..]) as usize
            } else {
                self.reader().read_ue()? as usize
            };
            if *s > 3 {
                return Err(bad("H.264 P sub_mb_type out of range"));
            }
        }
        let info: [(u8, u8, u8, u8); 4] = subs.map(|s| P_SUB_TYPES[s]);
        self.decode_sub_mbs(&info, raw == 4)?;
        let no_small = subs.iter().all(|&s| s == 0);
        self.finish_inter(no_small)
    }

    fn decode_b_mb(&mut self, raw: u32) -> Result<()> {
        let a = self.mb.addr;
        if raw == 0 {
            self.st.mb_type[a] = MbType::BDirect16x16;
            self.direct_motion(0b1111)?;
            return self.finish_inter(self.cx.sps.direct_8x8_inference);
        }
        if raw > 22 {
            return Err(bad("H.264 B mb_type out of range"));
        }
        if raw < 22 {
            let (part, pred) = B_MB_TYPES[raw as usize];
            self.st.mb_type[a] = MbType::Inter(part);
            self.decode_partitions(part, pred)?;
            return self.finish_inter(true);
        }
        self.st.mb_type[a] = MbType::Inter(Part::P8x8);
        let mut subs = [0usize; 4];
        for s in &mut subs {
            *s = if self.is_cabac() {
                let (cd, ctxs) = self.cabac();
                cd.decode_b_sub_mb_type(&mut ctxs[..]) as usize
            } else {
                self.reader().read_ue()? as usize
            };
            if *s > 12 {
                return Err(bad("H.264 B sub_mb_type out of range"));
            }
        }
        let mut direct_mask = 0u8;
        for (i, &s) in subs.iter().enumerate() {
            if s == 0 {
                direct_mask |= 1 << i;
            }
        }
        if direct_mask != 0 {
            self.direct_motion(direct_mask)?;
        }
        let info: [(u8, u8, u8, u8); 4] = subs.map(|s| B_SUB_TYPES[s]);
        self.decode_sub_mbs(&info, false)?;
        let no_small = subs.iter().all(|&s| {
            if s == 0 {
                self.cx.sps.direct_8x8_inference
            } else {
                B_SUB_TYPES[s].0 == 1
            }
        });
        self.finish_inter(no_small)
    }

    /// 16x16, 16x8 and 8x16 partitions: ref_idx and mvd syntax, motion derivation.
    fn decode_partitions(&mut self, part: Part, pred: [u8; 2]) -> Result<()> {
        let rects: &[(usize, usize, usize, usize)] = match part {
            Part::P16x16 => &[(0, 0, 4, 4)],
            Part::P16x8 => &[(0, 0, 4, 2), (0, 2, 4, 2)],
            Part::P8x16 => &[(0, 0, 2, 4), (2, 0, 2, 4)],
            Part::P8x8 => unreachable!(),
        };
        let mut refs = [[-1i8; 2]; 2];
        for list in 0..2 {
            for (p, &(bx, by, w, h)) in rects.iter().enumerate() {
                if pred[p] & (1 << list) == 0 {
                    continue;
                }
                let r = if self.num_refs(list) > 1 {
                    self.read_ref_idx(list, bx, by)?
                } else {
                    0
                };
                refs[p][list] = r;
                // Store early so CABAC contexts of later partitions see it.
                for y in by..by + h {
                    for x in bx..bx + w {
                        let i = self.st.b4(self.mb.x * 4 + x, self.mb.y * 4 + y);
                        self.st.blocks[i].ref_idx[list] = r;
                    }
                }
            }
        }
        for list in 0..2 {
            let mut done = 0u16;
            for (p, &(bx, by, w, h)) in rects.iter().enumerate() {
                if pred[p] & (1 << list) != 0 {
                    let (mvd, amvd) = self.read_mvd(list, bx, by)?;
                    let r = refs[p][list];
                    let shape = if rects.len() > 1 {
                        Some((part, p))
                    } else {
                        None
                    };
                    let mvp = self.mvp(list, bx, by, w, r, shape, done);
                    let mv = [mvp[0].wrapping_add(mvd[0]), mvp[1].wrapping_add(mvd[1])];
                    let id = self.ref_id(list, r)?;
                    self.fill_motion(bx, by, w, h, list, (r, mv), id, amvd);
                } else {
                    self.fill_motion(bx, by, w, h, list, (-1, [0, 0]), NO_PIC, [0, 0]);
                }
                done |= rect_mask(bx, by, w, h);
            }
        }
        Ok(())
    }

    /// P_8x8 / B_8x8 sub-macroblocks. `info[i] = (count, w, h, pred)`; `count == 0` is B_Direct_8x8.
    fn decode_sub_mbs(&mut self, info: &[(u8, u8, u8, u8); 4], ref0: bool) -> Result<()> {
        let mut refs = [[-1i8; 2]; 4];
        for list in 0..2 {
            for (i, &(count, _, _, pred)) in info.iter().enumerate() {
                if count == 0 || pred & (1 << list) == 0 {
                    continue;
                }
                let (bx, by) = ((i & 1) * 2, (i >> 1) * 2);
                let r = if self.num_refs(list) > 1 && !ref0 {
                    self.read_ref_idx(list, bx, by)?
                } else {
                    0
                };
                refs[i][list] = r;
                for y in by..by + 2 {
                    for x in bx..bx + 2 {
                        let j = self.st.b4(self.mb.x * 4 + x, self.mb.y * 4 + y);
                        self.st.blocks[j].ref_idx[list] = r;
                    }
                }
            }
        }
        for list in 0..2 {
            let mut done = 0u16;
            for (i, &(count, w, h, pred)) in info.iter().enumerate() {
                let (bx, by) = ((i & 1) * 2, (i >> 1) * 2);
                if count == 0 {
                    done |= rect_mask(bx, by, 2, 2);
                    continue;
                }
                let (w, h) = (w as usize, h as usize);
                for s in 0..count as usize {
                    let (sx, sy) = match (w, h) {
                        (2, 2) => (0, 0),
                        (2, 1) => (0, s),
                        (1, 2) => (s, 0),
                        _ => (s & 1, s >> 1),
                    };
                    let (x, y) = (bx + sx, by + sy);
                    if pred & (1 << list) != 0 {
                        let (mvd, amvd) = self.read_mvd(list, x, y)?;
                        let r = refs[i][list];
                        let mvp = self.mvp(list, x, y, w, r, None, done);
                        let mv = [mvp[0].wrapping_add(mvd[0]), mvp[1].wrapping_add(mvd[1])];
                        let id = self.ref_id(list, r)?;
                        self.fill_motion(x, y, w, h, list, (r, mv), id, amvd);
                    } else {
                        self.fill_motion(x, y, w, h, list, (-1, [0, 0]), NO_PIC, [0, 0]);
                    }
                    done |= rect_mask(x, y, w, h);
                }
            }
        }
        Ok(())
    }

    /// Direct prediction (8.4.1.2) for the 8x8 quadrants in `mask`.
    fn direct_motion(&mut self, mask: u8) -> Result<()> {
        if self.refs(1).is_empty() || self.refs(0).is_empty() {
            return Err(bad("H.264 direct prediction without reference lists"));
        }
        let col_short = !self.refs(1)[0].long;
        let inference = self.cx.sps.direct_8x8_inference;
        let spatial = self.cx.sh.direct_spatial;

        // Spatial: reference indices and predictors from the MB neighbours (8.4.1.2.2).
        let mut sp_ref = [-1i8; 2];
        let mut sp_mv = [[0i16; 2]; 2];
        let mut direct_zero = false;
        if spatial {
            // Neighbours A, B and C (or D) of the whole MB, as for a 16x16 partition's predictor.
            let nb = self.mb_neighbours();
            for (list, &[a, b, c]) in nb.iter().enumerate() {
                let rr = |n: Option<Mv>| n.map_or(-1, |n| n.0);
                sp_ref[list] = min_positive(rr(a), min_positive(rr(b), rr(c)));
            }
            if sp_ref[0] < 0 && sp_ref[1] < 0 {
                sp_ref = [0, 0];
                direct_zero = true;
            } else {
                for list in 0..2 {
                    if sp_ref[list] >= 0 {
                        let [a, b, c] = nb[list];
                        sp_mv[list] = mvp_from(a, b, c, sp_ref[list], None);
                    }
                }
            }
        }

        // Spatial motion of an 8x8 from its co-located block.
        let spatial_mvs = |col_ref: i8, col_mv: [i16; 2]| {
            let col_zero = col_short
                && col_ref == 0
                && (-1..=1).contains(&col_mv[0])
                && (-1..=1).contains(&col_mv[1]);
            let mut mvs = [[0i16; 2]; 2];
            for list in 0..2 {
                if !(direct_zero || sp_ref[list] < 0 || (sp_ref[list] == 0 && col_zero)) {
                    mvs[list] = sp_mv[list];
                }
            }
            mvs
        };
        // Spatial prediction only looks at the co-located block for a reference index 0.
        let need_col = !spatial || !(direct_zero || (sp_ref[0] != 0 && sp_ref[1] != 0));
        let mut col_mb = None;
        // A whole spatial-direct MB with 8x8 inference has one motion when the co-located MB is
        // not consulted or has one motion itself: derive and store it once.
        if spatial && inference && mask == 0xF {
            let col = if need_col {
                let c = col_mb.get_or_insert_with(|| self.colocated_mb());
                c.uniform().then(|| c.block(0, 0))
            } else {
                Some((-1, [0, 0], NO_PIC, 0))
            };
            if let Some((col_ref, col_mv, _, _)) = col {
                return self.store_direct(0, 0, 4, sp_ref, spatial_mvs(col_ref, col_mv));
            }
        }

        // With 8x8 inference, motion of each 8x8 is collected first so that a macroblock whose
        // 8x8s agree is stored once.
        let mut per8: [Option<DirectMotion>; 4] = [None; 4];
        for b8 in 0..4 {
            if mask & (1 << b8) == 0 {
                continue;
            }
            let (bx8, by8) = ((b8 & 1) * 2, (b8 >> 1) * 2);
            // With direct_8x8_inference the four 4x4 blocks share the corner co-located block,
            // so the 8x8 is derived once.
            for sub in 0..if inference { 1 } else { 4 } {
                let (bx, by) = (bx8 + (sub & 1), by8 + (sub >> 1));
                // Co-located 4x4 block (corner of the 8x8 with direct_8x8_inference).
                let (cbx, cby) = if inference {
                    ((b8 & 1) * 3, (b8 >> 1) * 3)
                } else {
                    (bx, by)
                };
                let (col_ref, col_mv, col_pic, vscale) = if need_col {
                    col_mb
                        .get_or_insert_with(|| self.colocated_mb())
                        .block(cbx, cby)
                } else {
                    (-1, [0, 0], NO_PIC, 0)
                };
                let (refs, mvs) = if spatial {
                    (sp_ref, spatial_mvs(col_ref, col_mv))
                } else {
                    // Temporal (8.4.1.2.3): map the co-located reference into RefPicList0 and
                    // convert its vertical vector between frame and field units.
                    let target = match self.field_parity() {
                        None => col_pic & !3,
                        Some(p) if col_pic % 4 == 0 => col_pic + 1 + p as u32,
                        Some(_) => col_pic,
                    };
                    let ref0 = if col_ref < 0 {
                        0
                    } else {
                        self.refs(0)
                            .iter()
                            .position(|r| r.id == target)
                            .unwrap_or(0) as i8
                    };
                    let col_mv = match vscale {
                        1 => [col_mv[0], col_mv[1].wrapping_mul(2)],
                        -1 => [col_mv[0], col_mv[1] / 2],
                        _ => col_mv,
                    };
                    let p0 = &self.refs(0)[ref0 as usize];
                    let p1 = &self.refs(1)[0];
                    let td = (p1.poc - p0.poc).clamp(-128, 127);
                    let (mv0, mv1) = if p0.long || td == 0 {
                        (col_mv, [0, 0])
                    } else {
                        let tb = (self.cur_poc() - p0.poc).clamp(-128, 127);
                        let tx = (16384 + (td / 2).abs()) / td;
                        let dsf = ((tb * tx + 32) >> 6).clamp(-1024, 1023);
                        let s = |v: i16| ((dsf * v as i32 + 128) >> 8) as i16;
                        let mv0 = [s(col_mv[0]), s(col_mv[1])];
                        (
                            mv0,
                            [
                                mv0[0].wrapping_sub(col_mv[0]),
                                mv0[1].wrapping_sub(col_mv[1]),
                            ],
                        )
                    };
                    ([ref0, 0], [mv0, mv1])
                };
                if inference {
                    per8[b8] = Some((refs, mvs));
                } else {
                    self.store_direct(bx, by, 1, refs, mvs)?;
                }
            }
        }
        if inference {
            let first = per8[0];
            if mask == 0xF && per8.iter().all(|p| *p == first) {
                let (refs, mvs) = first.expect("all four 8x8s");
                return self.store_direct(0, 0, 4, refs, mvs);
            }
            for (b8, p) in per8.iter().enumerate() {
                if let Some((refs, mvs)) = *p {
                    self.store_direct((b8 & 1) * 2, (b8 >> 1) * 2, 2, refs, mvs)?;
                }
            }
        }
        Ok(())
    }

    /// Stores direct-mode motion for the MB-relative `n`x`n` 4x4 blocks at `(bx, by)`.
    fn store_direct(
        &mut self,
        bx: usize,
        by: usize,
        n: usize,
        refs: [i8; 2],
        mvs: [[i16; 2]; 2],
    ) -> Result<()> {
        let mut b = NO_MOTION;
        for list in 0..2 {
            if refs[list] >= 0 {
                b.ref_idx[list] = refs[list];
                b.mv[list] = mvs[list];
            }
            b.ref_pic[list] = self.ref_id(list, b.ref_idx[list])?;
        }
        self.partial_motion |= n != 4;
        let i = self.st.b4(self.mb.x * 4 + bx, self.mb.y * 4 + by);
        let s = self.st.mb_w * 4;
        match n {
            4 => {
                fill_rect::<_, 4>(&mut self.st.blocks, i, s, 4, b);
                fill_rect::<_, 4>(&mut self.st.direct, i, s, 4, true);
            }
            2 => {
                fill_rect::<_, 2>(&mut self.st.blocks, i, s, 2, b);
                fill_rect::<_, 2>(&mut self.st.direct, i, s, 2, true);
            }
            _ => {
                fill_rect::<_, 1>(&mut self.st.blocks, i, s, 1, b);
                fill_rect::<_, 1>(&mut self.st.direct, i, s, 1, true);
            }
        }
        Ok(())
    }

    /// The co-located macroblock of direct prediction (8.4.1.2.1), for [`ColMb::block`].
    fn colocated_mb(&self) -> ColMb<'c> {
        let col = &self.refs(1)[0];
        let Some(m) = col.motion.as_deref() else {
            return ColMb {
                m: None,
                mb_w: 0,
                mbx: 0,
                layout: ColLayout::Same(0),
            };
        };
        let mb_w = self.st.mb_w;
        let mbx = self.mb.x;
        // Motion is stored in frame layout; pick the co-located MB row and block row there.
        // An MBAFF field MB uses the field of RefPicList1[0] with its own parity, at its pair row.
        let (structure, cur_row) = if self.mbaff_field() {
            (Structure::Field((self.mb.y & 1) as u8), self.mb.y / 2)
        } else {
            (self.st.structure, self.mb.y)
        };
        let layout = match structure {
            Structure::Frame => {
                let row = cur_row;
                if m.mb_field.get(row * mb_w + mbx).copied().unwrap_or(false) {
                    let cur = self.cx.cur_poc;
                    let cp = usize::from(
                        (col.field_poc[0] - cur).abs() >= (col.field_poc[1] - cur).abs(),
                    );
                    ColLayout::FrameFromField((row & !1) + cp, 2 * (row % 2))
                } else {
                    ColLayout::Same(row)
                }
            }
            Structure::Field(cur_parity) => {
                // The co-located field of an MBAFF frame is taken at the current parity (Table 8-6).
                let cp = if m.mbaff {
                    cur_parity as usize
                } else {
                    col.parity.unwrap_or(0) as usize
                };
                let r = cur_row;
                if m.mb_field
                    .get((2 * r + cp) * mb_w + mbx)
                    .copied()
                    .unwrap_or(false)
                {
                    ColLayout::Same(2 * r + cp)
                } else {
                    ColLayout::FieldFromFrame(2 * r)
                }
            }
        };
        ColMb {
            m: Some(m),
            mb_w,
            mbx,
            layout,
        }
    }

    /// Coded block pattern, optional transform_size_8x8_flag, residual and reconstruction of an inter MB.
    fn finish_inter(&mut self, no_small: bool) -> Result<()> {
        let a = self.mb.addr;
        let cbp = self.read_cbp(false)?;
        self.st.cbp[a] = cbp;
        let mut t8 = false;
        if cbp & 0x0F != 0 && self.cx.pps.transform_8x8_mode && no_small {
            t8 = self.read_transform_8x8()?;
        }
        self.st.t8x8[a] = t8;
        let mut res = self.take_residual();
        if cbp != 0 {
            self.read_qp_delta()?;
            self.set_qp();
            self.read_residual(cbp, false, t8, false, &mut res)?;
        } else {
            self.last_dqp_nz = false;
            self.set_qp();
        }
        self.inter_predict_mb()?;
        let (o, ls, px, py) = self.view(0);
        if t8 {
            for b8 in 0..4 {
                if res.luma_coded & (0xF << (b8 * 4)) != 0 {
                    let off = o + (py + (b8 >> 1) * 8) * ls + px + (b8 & 1) * 8;
                    recon::idct8_add(&mut self.st.planes[0][off..], ls, &mut res.luma8[b8]);
                }
            }
        } else {
            for blk in 0..16 {
                if res.luma_coded & (1 << blk) != 0 {
                    let (bx, by) = BLK4_XY[blk];
                    let off = o + (py + by * 4) * ls + px + bx * 4;
                    recon::idct4_add(&mut self.st.planes[0][off..], ls, &mut res.luma[blk]);
                }
            }
        }
        if self.cx.sps.chroma_array_type() != 0 {
            self.add_chroma_residual(&mut res);
        }
        self.res = Some(res);
        Ok(())
    }

    fn weighting(&self, refs: [i8; 2], comp: usize) -> Weighting {
        if let Some(t) = &self.cx.sh.weights {
            let log = if comp == 0 {
                t.luma_log2
            } else {
                t.chroma_log2
            };
            // MBAFF field MBs index the table by frame (refIdxL0WP = refIdxL0 >> 1, 8.4.2.3).
            let shift = self.mbaff_field() as u32;
            let e = |list: usize| -> (i32, i32) {
                let r = refs[list];
                if r < 0 {
                    return (1, 0);
                }
                let Some(e) = t.l[list].get((r as usize) >> shift) else {
                    return (1 << log, 0);
                };
                if comp == 0 {
                    e.luma
                } else {
                    e.chroma[comp - 1]
                }
            };
            return Weighting::Weighted(log, [e(0), e(1)]);
        }
        let implicit = if self.mbaff_field() {
            &self.implicit_field[self.mb.y & 1]
        } else {
            &self.implicit
        };
        if !implicit.is_empty() && refs[0] >= 0 && refs[1] >= 0 {
            let n1 = self.refs(1).len();
            let w = implicit[refs[0] as usize * n1 + refs[1] as usize];
            return Weighting::Weighted(5, [(w[0], 0), (w[1], 0)]);
        }
        Weighting::Default
    }

    /// Motion-compensated prediction of the whole macroblock from the stored motion. Blocks with
    /// identical motion are predicted together; the result is the same for any grouping.
    fn inter_predict_mb(&mut self) -> Result<()> {
        let (mx4, my4) = (self.mb.x * 4, self.mb.y * 4);
        let key = |s: &PicState, bx: usize, by: usize| {
            let j = s.b4(mx4 + bx, my4 + by);
            let b = &s.blocks[j];
            (b.ref_idx[0], b.mv[0], b.ref_idx[1], b.mv[1])
        };
        let same = |s: &PicState, bx: usize, by: usize, n: usize| {
            let k = key(s, bx, by);
            (by..by + n).all(|y| (bx..bx + n).all(|x| key(s, x, y) == k))
        };
        // Motion stored only for the whole MB (the other list left at its reset value) is one
        // motion without comparing.
        if !self.partial_motion || same(self.st, 0, 0, 4) {
            self.st.motion_edges[self.mb.addr] = 0;
            let k = key(self.st, 0, 0);
            return self.mc_rect(0, 0, 4, 4, [k.0, k.2], [k.1, k.3]);
        }
        let uni: [bool; 4] =
            std::array::from_fn(|b8| same(self.st, (b8 & 1) * 2, (b8 >> 1) * 2, 2));
        let k8 = |b8: usize| key(self.st, (b8 & 1) * 2, (b8 >> 1) * 2);
        // Inner edges of each 8x8, then the edges between 8x8s that are uniform and agree.
        let mut m = 0u8;
        for (bit, a, b) in [(1, 0, 2), (3, 1, 3), (5, 0, 1), (7, 2, 3)] {
            if !uni[a] || !uni[b] {
                m |= 1 << bit;
            }
        }
        let all = uni.iter().all(|&u| u);
        if !(all && k8(0) == k8(1) && k8(2) == k8(3)) {
            m |= 1 << 2;
        }
        if !(all && k8(0) == k8(2) && k8(1) == k8(3)) {
            m |= 1 << 6;
        }
        self.st.motion_edges[self.mb.addr] = m;
        for (b8, &uniform) in uni.iter().enumerate() {
            let (bx, by) = ((b8 & 1) * 2, (b8 >> 1) * 2);
            if uniform {
                let k = key(self.st, bx, by);
                self.mc_rect(bx, by, 2, 2, [k.0, k.2], [k.1, k.3])?;
            } else {
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let k = key(self.st, bx + dx, by + dy);
                    self.mc_rect(bx + dx, by + dy, 1, 1, [k.0, k.2], [k.1, k.3])?;
                }
            }
        }
        Ok(())
    }

    /// [`Self::mc_rect`] with default weighted prediction for every component (8.4.2.3.1): one
    /// list is predicted, a second is averaged into it.
    #[allow(clippy::too_many_arguments)]
    fn mc_rect_default(
        &mut self,
        rp: [Option<&RefPic>; 2],
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        pw: usize,
        ph: usize,
        mvs: [[i16; 2]; 2],
    ) -> Result<()> {
        let (o, ls, _, _) = self.view(0);
        let d = o + y * ls + x;
        let dst = &mut self.st.planes[0][d..];
        match rp {
            [Some(r), None] | [None, Some(r)] => {
                let l = rp[0].is_none() as usize;
                recon::mc_luma::<false>(&r.planes[0], pw, ph, x, y, w, h, mvs[l], dst, ls);
            }
            [Some(r0), Some(r1)] => {
                recon::mc_luma::<false>(&r0.planes[0], pw, ph, x, y, w, h, mvs[0], dst, ls);
                recon::mc_luma::<true>(&r1.planes[0], pw, ph, x, y, w, h, mvs[1], dst, ls);
            }
            [None, None] => {}
        }
        if self.cx.sps.chroma_array_type() == 0 {
            return Ok(());
        }
        let (co, cs, _, _) = self.view(1);
        let (cw, ch, cpw, cph) = (w / 2, h / 2, pw / 2, ph / 2);
        let (cx, cy) = (x / 2, y / 2);
        let mut cmv = mvs;
        if let Some(cur) = self.field_parity() {
            for list in 0..2 {
                // Between fields of opposite parity the chroma vector is offset by a quarter
                // chroma sample (Table 8-9).
                if let Some(Some(r)) = rp[list].map(|r| r.parity) {
                    cmv[list][1] = cmv[list][1].wrapping_add(2 * (cur as i16 - r as i16));
                }
            }
        }
        let d = co + cy * cs + cx;
        for p in 1..3 {
            let dst = &mut self.st.planes[p][d..];
            match rp {
                [Some(r), None] | [None, Some(r)] => {
                    let l = rp[0].is_none() as usize;
                    let mv = cmv[l];
                    recon::mc_chroma::<false>(&r.planes[p], cpw, cph, cx, cy, cw, ch, mv, dst, cs);
                }
                [Some(r0), Some(r1)] => {
                    let (m0, m1) = (cmv[0], cmv[1]);
                    recon::mc_chroma::<false>(&r0.planes[p], cpw, cph, cx, cy, cw, ch, m0, dst, cs);
                    recon::mc_chroma::<true>(&r1.planes[p], cpw, cph, cx, cy, cw, ch, m1, dst, cs);
                }
                [None, None] => {}
            }
        }
        Ok(())
    }

    fn mc_rect(
        &mut self,
        bx: usize,
        by: usize,
        w4: usize,
        h4: usize,
        refs: [i8; 2],
        mvs: [[i16; 2]; 2],
    ) -> Result<()> {
        let (w, h) = (w4 * 4, h4 * 4);
        let (o, ls, px, py) = self.view(0);
        let (x, y) = (px + bx * 4, py + by * 4);
        // Size of the picture (or, for an MBAFF field MB, of the field) the vectors point into.
        let (pw, ph) = if self.mbaff_field() {
            (self.st.width, self.st.height / 2)
        } else {
            (self.st.width, self.st.height)
        };
        let mut rp: [Option<&RefPic>; 2] = [None, None];
        for list in 0..2 {
            if refs[list] >= 0 {
                rp[list] = Some(
                    self.refs(list)
                        .get(refs[list] as usize)
                        .ok_or_else(|| bad("H.264 ref_idx beyond list"))?,
                );
            }
        }
        if rp.iter().all(Option::is_none) {
            return Err(bad("H.264 inter block without prediction"));
        }
        let used = rp.map(|r| r.is_some());
        let wt = recon::effective_weighting(used, self.weighting(refs, 0));
        if self.cx.sh.weights.is_none() && matches!(wt, Weighting::Default) {
            return self.mc_rect_default(rp, x, y, w, h, pw, ph, mvs);
        }
        let planes = |list: usize| &rp[list].expect("used list").planes;
        let d = o + y * ls + x;
        recon::predict_block(
            &mut self.st.planes[0][d..],
            ls,
            w,
            h,
            used,
            wt,
            &mut self.scratch,
            |list, out, os| {
                recon::mc_luma::<false>(&planes(list)[0], pw, ph, x, y, w, h, mvs[list], out, os)
            },
        );
        if self.cx.sps.chroma_array_type() == 0 {
            return Ok(());
        }
        let (co, cs, _, _) = self.view(1);
        let (cw, ch, cpw, cph) = (w / 2, h / 2, pw / 2, ph / 2);
        let (cx, cy) = (x / 2, y / 2);
        let cur_parity = self.field_parity();
        let mut cmv = mvs;
        for list in 0..2 {
            // Between fields of opposite parity the chroma vector is offset by a quarter chroma
            // sample (Table 8-9).
            if let (Some(cur), Some(Some(r))) = (cur_parity, rp[list].map(|r| r.parity)) {
                cmv[list][1] = cmv[list][1].wrapping_add(2 * (cur as i16 - r as i16));
            }
        }
        for p in 0..2 {
            // Implicit and default weights are the same for every component.
            let wt = if self.cx.sh.weights.is_some() {
                self.weighting(refs, 1 + p)
            } else {
                wt
            };
            let d = co + cy * cs + cx;
            recon::predict_block(
                &mut self.st.planes[1 + p][d..],
                cs,
                cw,
                ch,
                used,
                wt,
                &mut self.scratch,
                |list, out, os| {
                    recon::mc_chroma::<false>(
                        &planes(list)[1 + p],
                        cpw,
                        cph,
                        cx,
                        cy,
                        cw,
                        ch,
                        cmv[list],
                        out,
                        os,
                    )
                },
            );
        }
        Ok(())
    }
}

/// Where the co-located 4x4 blocks of the current macroblock lie in `RefPicList1[0]`'s motion
/// (8.4.1.2.1, Table 8-8).
#[derive(Clone, Copy)]
enum ColLayout {
    /// Frame from frame, or field from field: MB row `.0`, block rows as they are.
    Same(usize),
    /// A frame MB from a field MB pair: MB row `.0`, block rows `.1 + cby / 2`.
    FrameFromField(usize, usize),
    /// A field MB from frame MBs: MB rows `.0 + cby / 2`, block rows `2 * (cby % 2)`.
    FieldFromFrame(usize),
}

/// The co-located macroblock of the current one; `m` is `None` without stored motion.
#[derive(Clone, Copy)]
struct ColMb<'r> {
    m: Option<&'r Motion>,
    mb_w: usize,
    mbx: usize,
    layout: ColLayout,
}

impl ColMb<'_> {
    /// Every 4x4 block of the co-located MB gives the same [`Self::block`] result.
    fn uniform(&self) -> bool {
        let Some(m) = self.m else {
            return true;
        };
        let ColLayout::Same(row) = self.layout else {
            return false;
        };
        let mb = row * self.mb_w + self.mbx;
        m.mb_intra.get(mb).copied().unwrap_or(true)
            || m.mb_uniform.get(mb).copied().unwrap_or(false)
    }

    /// `(refIdxCol, mvCol, referenced picture id, vertical scale)` of MB-relative 4x4 block
    /// `(cbx, cby)`; the scale is `1` for field-to-frame, `-1` for frame-to-field, else `0`.
    fn block(&self, cbx: usize, cby: usize) -> (i8, [i16; 2], u32, i32) {
        let Some(m) = self.m else {
            return (-1, [0, 0], NO_PIC, 0);
        };
        let (row, by, vscale) = match self.layout {
            ColLayout::Same(row) => (row, cby, 0),
            ColLayout::FrameFromField(row, half) => (row, half + cby / 2, 1),
            ColLayout::FieldFromFrame(r2) => (r2 + cby / 2, 2 * (cby % 2), -1),
        };
        let mb = row * self.mb_w + self.mbx;
        if mb >= m.mb_intra.len() || m.mb_intra[mb] {
            return (-1, [0, 0], NO_PIC, vscale);
        }
        let i = (row * 4 + by) * self.mb_w * 4 + self.mbx * 4 + cbx;
        let b = &m.blocks[i];
        let l = if b.ref_idx[0] >= 0 { 0 } else { 1 };
        (b.ref_idx[l], b.mv[l], b.ref_pic[l], vscale)
    }
}

/// Motion vector predictor (8.4.1.3) from the neighbours A, B and C (or D) of a partition.
fn mvp_from(
    a: Option<Mv>,
    b: Option<Mv>,
    c: Option<Mv>,
    refidx: i8,
    part: Option<(Part, usize)>,
) -> [i16; 2] {
    match part {
        Some((Part::P16x8, 0)) if b.is_some_and(|n| n.0 == refidx) => return b.unwrap().1,
        Some((Part::P16x8, 1)) if a.is_some_and(|n| n.0 == refidx) => return a.unwrap().1,
        Some((Part::P8x16, 0)) if a.is_some_and(|n| n.0 == refidx) => return a.unwrap().1,
        Some((Part::P8x16, 1)) if c.is_some_and(|n| n.0 == refidx) => return c.unwrap().1,
        _ => {}
    }
    let (a, b, c) = if b.is_none() && c.is_none() && a.is_some() {
        (a, a, a)
    } else {
        (a, b, c)
    };
    let un = (-1i8, [0i16; 2]);
    let (a, b, c) = (a.unwrap_or(un), b.unwrap_or(un), c.unwrap_or(un));
    let matches = [a.0 == refidx, b.0 == refidx, c.0 == refidx];
    if matches.iter().filter(|&&m| m).count() == 1 {
        return if matches[0] {
            a.1
        } else if matches[1] {
            b.1
        } else {
            c.1
        };
    }
    [
        median(a.1[0], b.1[0], c.1[0]),
        median(a.1[1], b.1[1], c.1[1]),
    ]
}

/// Reference indices and vectors of both lists for one direct-mode block.
type DirectMotion = ([i8; 2], [[i16; 2]; 2]);

fn combine_nc(a: Option<u8>, b: Option<u8>) -> u8 {
    match (a, b) {
        (Some(a), Some(b)) => ((a as u16 + b as u16 + 1) >> 1) as u8,
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => 0,
    }
}

fn median(a: i16, b: i16, c: i16) -> i16 {
    a.max(b).min(a.min(b).max(c))
}

fn min_positive(a: i8, b: i8) -> i8 {
    if a >= 0 && b >= 0 { a.min(b) } else { a.max(b) }
}

/// Bit mask of MB-relative 4x4 blocks (by `luma4x4BlkIdx`) covered by a rectangle.
fn rect_mask(bx: usize, by: usize, w: usize, h: usize) -> u16 {
    let mut m = 0u16;
    for y in by..by + h {
        for x in bx..bx + w {
            m |= 1 << BLK4_AT[y][x];
        }
    }
    m
}
