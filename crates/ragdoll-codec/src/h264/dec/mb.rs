//! Slice data and macroblock layer decoding (ITU-T H.264 7.3.4, 7.3.5, 8.3, 8.4, 8.5).

#![allow(clippy::needless_range_loop)]

use std::sync::Arc;

use ragdoll_core::{BitReader, RagdollError, Result};

use crate::cabac::{CabacContext, CabacDecoder, NUM_CABAC_CONTEXTS, init_cabac_contexts};
use crate::h264::decode_cavlc_block;

use super::cabac_se;
use super::intra::{self, Edges};
use super::picture::{MbType, Motion, NO_PIC, PicState, Structure};
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
    pub cur_poc: i32,
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
    let first = sh.first_mb as usize;
    if first >= total {
        return Err(bad("H.264 first_mb_in_slice beyond picture"));
    }
    let implicit = build_implicit_weights(cx);
    let mut d = Dec {
        cx,
        st,
        ent,
        qp: sh.qp,
        last_dqp_nz: false,
        mb: MbPos::new(first, 0),
        implicit,
    };
    d.run(first, total)
}

#[derive(Clone, Copy)]
struct MbPos {
    addr: usize,
    x: usize,
    y: usize,
}

impl MbPos {
    fn new(addr: usize, mb_w: usize) -> Self {
        if mb_w == 0 {
            return Self { addr, x: 0, y: 0 };
        }
        Self {
            addr,
            x: addr % mb_w,
            y: addr / mb_w,
        }
    }
}

/// Implicit bi-prediction weights `[w0, w1]` per `(refIdxL0, refIdxL1)` (8.4.2.3.1), empty when unused.
fn build_implicit_weights(cx: &SliceCtx<'_>) -> Vec<[i32; 2]> {
    if !(cx.sh.slice_type.is_b() && cx.pps.weighted_bipred_idc == 2) {
        return Vec::new();
    }
    let (l0, l1) = (&cx.refs[0], &cx.refs[1]);
    let mut out = Vec::with_capacity(l0.len() * l1.len());
    for r0 in l0 {
        for r1 in l1 {
            let td = (r1.poc - r0.poc).clamp(-128, 127);
            let w = if r0.long || r1.long || td == 0 {
                [32, 32]
            } else {
                let tb = (cx.cur_poc - r0.poc).clamp(-128, 127);
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
    implicit: Vec<[i32; 2]>,
}

impl<'a, 'c> Dec<'a, 'c> {
    fn run(&mut self, first: usize, total: usize) -> Result<()> {
        let slice_type = self.cx.sh.slice_type;
        let mut addr = first;
        if !self.is_cabac() {
            loop {
                if !slice_type.is_intra() {
                    let run = self.reader().read_ue()? as usize;
                    for _ in 0..run {
                        if addr >= total {
                            return Err(bad("H.264 mb_skip_run beyond picture"));
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
                self.decode_mb(addr)?;
                addr += 1;
                if !self.reader().more_rbsp_data() {
                    return Ok(());
                }
            }
        } else {
            loop {
                if addr >= total {
                    return Err(bad("H.264 slice data beyond picture"));
                }
                let skipped = if slice_type.is_intra() {
                    false
                } else {
                    self.begin_mb(addr);
                    let a = self
                        .mb_left()
                        .filter(|&m| !self.st.mb_type[m].is_skip())
                        .is_some() as u8;
                    let b = self
                        .mb_top()
                        .filter(|&m| !self.st.mb_type[m].is_skip())
                        .is_some() as u8;
                    let (cd, ctxs) = self.cabac();
                    if slice_type.is_b() {
                        cd.decode_b_skip(&mut ctxs[..], a, b)
                    } else {
                        cd.decode_p_skip(&mut ctxs[..], a, b)
                    }
                };
                if skipped {
                    self.decode_skip(addr)?;
                } else {
                    self.decode_mb(addr)?;
                }
                let (cd, _) = self.cabac();
                if cd.decode_terminate() != 0 {
                    return Ok(());
                }
                addr += 1;
            }
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

    /// Whether the current macroblock is a field macroblock (field picture).
    fn field_mb(&self) -> bool {
        matches!(self.st.structure, Structure::Field(_))
    }

    fn is_cabac(&self) -> bool {
        matches!(self.ent, Entropy::Cabac(..))
    }

    // ----- neighbourhood -------------------------------------------------------------------

    fn begin_mb(&mut self, addr: usize) {
        self.mb = MbPos::new(addr, self.st.mb_w);
        self.st.slice_num[addr] = self.cx.slice_num;
    }

    fn mb_avail(&self, addr: usize) -> bool {
        self.st.slice_num[addr] == self.cx.slice_num && addr != self.mb.addr
    }

    fn mb_left(&self) -> Option<usize> {
        (self.mb.x > 0)
            .then(|| self.mb.addr - 1)
            .filter(|&a| self.mb_avail(a))
    }

    fn mb_top(&self) -> Option<usize> {
        (self.mb.y > 0)
            .then(|| self.mb.addr - self.st.mb_w)
            .filter(|&a| self.mb_avail(a))
    }

    /// Macroblock containing absolute 4x4 position `(x4, y4)`, if inside the picture.
    fn mb_at4(&self, x4: isize, y4: isize) -> Option<usize> {
        if x4 < 0
            || y4 < 0
            || x4 >= (self.st.mb_w * 4) as isize
            || y4 >= (self.st.mb_h * 4) as isize
        {
            return None;
        }
        Some((y4 as usize / 4) * self.st.mb_w + x4 as usize / 4)
    }

    /// Whether the 4x4 block at absolute `(x4, y4)` is available; `done` marks decoded blocks of the current MB.
    fn blk_avail(&self, x4: isize, y4: isize, done: u16) -> bool {
        match self.mb_at4(x4, y4) {
            None => false,
            Some(a) if a == self.mb.addr => {
                done & (1 << BLK4_AT[y4 as usize % 4][x4 as usize % 4]) != 0
            }
            Some(a) => self.mb_avail(a),
        }
    }

    /// Availability of a neighbouring macroblock for intra prediction (constrained intra pred).
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

    /// Intra availability `(left, top, top_right, top_left)` of the `n`x`n` luma block at MB-relative 4x4 `(bx, by)`.
    fn intra_edges(
        &self,
        bx: usize,
        by: usize,
        n4: usize,
        cur_blk: usize,
    ) -> (bool, bool, bool, bool) {
        let (mx4, my4) = ((self.mb.x * 4) as isize, (self.mb.y * 4) as isize);
        let at = |dx: isize, dy: isize| -> bool {
            let (x, y) = (bx as isize + dx, by as isize + dy);
            let addr = self.mb_at4(mx4 + x, my4 + y);
            if (0..4).contains(&x) && (0..4).contains(&y) {
                // Inside the current MB: decoded iff earlier in block order.
                BLK4_AT[y as usize][x as usize] < cur_blk
            } else if x >= 4 && y >= 0 {
                false
            } else {
                self.intra_ok(addr)
            }
        };
        (at(-1, 0), at(0, -1), at(n4 as isize, -1), at(-1, -1))
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
        let (x4, y4) = ((self.mb.x * 4 + bx) as isize, (self.mb.y * 4 + by) as isize);
        let fetch = |nx: isize, ny: isize| -> Option<i8> {
            let addr = self.mb_at4(nx, ny)?;
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
            Some(self.st.intra_mode[self.st.b4(nx as usize, ny as usize)])
        };
        match (fetch(x4 - 1, y4), fetch(x4, y4 - 1)) {
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
            let (i, p) = *entry.ok_or(bad("H.264 coded_block_pattern out of range"))?;
            return Ok(if intra { i } else { p });
        }
        // Luma: condTermFlagN = 1 when the neighbouring 8x8 block has its CBP bit clear (skip counts as clear).
        let luma_bits = |m: Option<usize>| -> u8 {
            match m {
                None => 0x0F,
                Some(a) if self.st.mb_type[a] == MbType::IPcm => 0x0F,
                Some(a) if self.st.mb_type[a].is_skip() => 0,
                Some(a) => self.st.cbp[a] & 0x0F,
            }
        };
        let chroma_of = |m: Option<usize>| -> u8 {
            match m {
                None => 0,
                Some(a) if self.st.mb_type[a] == MbType::IPcm => 2,
                Some(a) if self.st.mb_type[a].is_skip() => 0,
                Some(a) => self.st.cbp[a] >> 4,
            }
        };
        let (left, top) = (self.mb_left(), self.mb_top());
        let (la, tb) = (luma_bits(left), luma_bits(top));
        let (ca, cb) = (chroma_of(left), chroma_of(top));
        let (cd, ctxs) = self.cabac();
        let mut cbp = 0u8;
        for b8 in 0..4 {
            let a_bit = match b8 {
                0 => (la >> 1) & 1,
                1 => cbp & 1,
                2 => (la >> 3) & 1,
                _ => (cbp >> 2) & 1,
            };
            let b_bit = match b8 {
                0 => (tb >> 2) & 1,
                1 => (tb >> 3) & 1,
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
        let mut res = Residual::new();
        if cbp != 0 || is_i16 {
            self.read_qp_delta()?;
            self.set_qp();
            self.read_residual(cbp, is_i16, t8, true, &mut res)?;
        } else {
            self.last_dqp_nz = false;
            self.set_qp();
        }
        self.reconstruct_intra(kind, &modes, &res)
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
        let w = self.st.width;
        let (px, py) = (self.mb.x * 16, self.mb.y * 16);
        for y in 0..16 {
            self.st.planes[0][(py + y) * w + px..(py + y) * w + px + 16]
                .copy_from_slice(&samples[y * 16..y * 16 + 16]);
        }
        if n == 384 {
            let cw = w / 2;
            for p in 0..2 {
                for y in 0..8 {
                    let src = &samples[256 + p * 64 + y * 8..256 + p * 64 + y * 8 + 8];
                    self.st.planes[1 + p]
                        [(py / 2 + y) * cw + px / 2..(py / 2 + y) * cw + px / 2 + 8]
                        .copy_from_slice(src);
                }
            }
        }
        Ok(())
    }

    fn reconstruct_intra(&mut self, kind: MbType, modes: &[u8; 16], res: &Residual) -> Result<()> {
        let w = self.st.width;
        let (px, py) = (self.mb.x * 16, self.mb.y * 16);
        match kind {
            MbType::I4x4 => {
                for (blk, &mode) in modes.iter().enumerate() {
                    let (bx, by) = BLK4_XY[blk];
                    let (l, t, tr, tl) = self.intra_edges(bx, by, 1, blk);
                    let (x, y) = (px + bx * 4, py + by * 4);
                    let e = Edges::gather(&self.st.planes[0], w, x, y, 4, l, t, tr && t, tl);
                    let mut out = [0u8; 16];
                    intra::predict_nxn(4, mode, &e, &mut out);
                    for r in 0..4 {
                        self.st.planes[0][(y + r) * w + x..(y + r) * w + x + 4]
                            .copy_from_slice(&out[r * 4..r * 4 + 4]);
                    }
                    if res.luma_coded & (1 << blk) != 0 {
                        recon::idct4_add(&mut self.st.planes[0][y * w + x..], w, &res.luma[blk]);
                    }
                }
            }
            MbType::I8x8 => {
                for b8 in 0..4 {
                    let (bx, by) = ((b8 & 1) * 2, (b8 >> 1) * 2);
                    let (l, t, tr, tl) = self.intra_edges(bx, by, 2, b8 * 4);
                    let (x, y) = (px + bx * 4, py + by * 4);
                    let e =
                        Edges::gather(&self.st.planes[0], w, x, y, 8, l, t, tr && t, tl).filter8();
                    let mut out = [0u8; 64];
                    intra::predict_nxn(8, modes[b8], &e, &mut out);
                    for r in 0..8 {
                        self.st.planes[0][(y + r) * w + x..(y + r) * w + x + 8]
                            .copy_from_slice(&out[r * 8..r * 8 + 8]);
                    }
                    if res.luma_coded & (0xF << (b8 * 4)) != 0 {
                        recon::idct8_add(&mut self.st.planes[0][y * w + x..], w, &res.luma8[b8]);
                    }
                }
            }
            MbType::I16x16(mode) => {
                let mb_top = self.mb.y > 0 && self.intra_ok(Some(self.mb.addr - self.st.mb_w));
                let mb_left = self.mb.x > 0 && self.intra_ok(Some(self.mb.addr - 1));
                let mb_tl = self.mb.x > 0
                    && self.mb.y > 0
                    && self.intra_ok(Some(self.mb.addr - self.st.mb_w - 1));
                if !intra::block_mode_ok(mode, mb_left, mb_top, mb_tl) {
                    return Err(bad("H.264 Intra16x16 mode uses unavailable samples"));
                }
                let mut out = [0u8; 256];
                intra::predict_16x16(
                    &self.st.planes[0],
                    w,
                    px,
                    py,
                    mode,
                    mb_left,
                    mb_top,
                    mb_tl,
                    &mut out,
                );
                for r in 0..16 {
                    self.st.planes[0][(py + r) * w + px..(py + r) * w + px + 16]
                        .copy_from_slice(&out[r * 16..r * 16 + 16]);
                }
                for blk in 0..16 {
                    if res.luma_coded & (1 << blk) != 0 {
                        let (bx, by) = BLK4_XY[blk];
                        let off = (py + by * 4) * w + px + bx * 4;
                        recon::idct4_add(&mut self.st.planes[0][off..], w, &res.luma[blk]);
                    }
                }
            }
            _ => unreachable!(),
        }
        if self.cx.sps.chroma_array_type() != 0 {
            let mb_top = self.mb.y > 0 && self.intra_ok(Some(self.mb.addr - self.st.mb_w));
            let mb_left = self.mb.x > 0 && self.intra_ok(Some(self.mb.addr - 1));
            let mb_tl = self.mb.x > 0
                && self.mb.y > 0
                && self.intra_ok(Some(self.mb.addr - self.st.mb_w - 1));
            let mode = self.st.chroma_pred_mode[self.mb.addr];
            let as_block = [2, 1, 0, 3][mode as usize];
            if !intra::block_mode_ok(as_block, mb_left, mb_top, mb_tl) {
                return Err(bad("H.264 chroma intra mode uses unavailable samples"));
            }
            let cw = w / 2;
            for p in 0..2 {
                let mut out = [0u8; 64];
                intra::predict_chroma(
                    &self.st.planes[1 + p],
                    cw,
                    px / 2,
                    py / 2,
                    mode,
                    mb_left,
                    mb_top,
                    mb_tl,
                    &mut out,
                );
                for r in 0..8 {
                    let o = (py / 2 + r) * cw + px / 2;
                    self.st.planes[1 + p][o..o + 8].copy_from_slice(&out[r * 8..r * 8 + 8]);
                }
            }
            self.add_chroma_residual(res);
        }
        Ok(())
    }

    fn add_chroma_residual(&mut self, res: &Residual) {
        let cw = self.st.width / 2;
        let (cx, cy) = (self.mb.x * 8, self.mb.y * 8);
        for p in 0..2 {
            for blk in 0..4 {
                if res.chroma_coded[p] & (1 << blk) != 0 {
                    let off = (cy + (blk >> 1) * 4) * cw + cx + (blk & 1) * 4;
                    recon::idct4_add(&mut self.st.planes[1 + p][off..], cw, &res.chroma[p][blk]);
                }
            }
        }
    }

    // ----- residual -------------------------------------------------------------------------

    /// CAVLC `nC` for the luma 4x4 block at absolute `(x4, y4)` (9.2.1).
    fn nc_luma(&self, x4: usize, y4: usize) -> u8 {
        let (x4, y4) = (x4 as isize, y4 as isize);
        let get = |nx: isize, ny: isize| -> Option<u8> {
            let a = self.mb_at4(nx, ny)?;
            (a == self.mb.addr || self.mb_avail(a))
                .then(|| self.st.nnz[self.st.b4(nx as usize, ny as usize)])
        };
        combine_nc(get(x4 - 1, y4), get(x4, y4 - 1))
    }

    fn nc_chroma(&self, p: usize, x4: usize, y4: usize) -> u8 {
        let (x4, y4) = (x4 as isize, y4 as isize);
        let get = |nx: isize, ny: isize| -> Option<u8> {
            let a = self.mb_at4(nx * 2, ny * 2)?;
            (a == self.mb.addr || self.mb_avail(a))
                .then(|| self.st.nnz_c[p][self.st.c4(nx as usize, ny as usize)])
        };
        combine_nc(get(x4 - 1, y4), get(x4, y4 - 1))
    }

    /// CABAC coded_block_flag condTermFlagN for a luma 4x4 neighbour (9.3.3.1.1.9).
    fn cbf_luma(&self, nx: isize, ny: isize, intra: bool) -> u8 {
        match self.mb_at4(nx, ny) {
            Some(a) if a == self.mb.addr => self.st.nz[self.st.b4(nx as usize, ny as usize)] as u8,
            Some(a) if self.mb_avail(a) => {
                if self.st.mb_type[a] == MbType::IPcm {
                    1
                } else {
                    self.st.nz[self.st.b4(nx as usize, ny as usize)] as u8
                }
            }
            _ => intra as u8,
        }
    }

    fn cbf_chroma_ac(&self, p: usize, nx: isize, ny: isize, intra: bool) -> u8 {
        match self.mb_at4(nx * 2, ny * 2) {
            Some(a) if a == self.mb.addr => {
                (self.st.nnz_c[p][self.st.c4(nx as usize, ny as usize)] > 0) as u8
            }
            Some(a) if self.mb_avail(a) => {
                if self.st.mb_type[a] == MbType::IPcm {
                    1
                } else {
                    (self.st.nnz_c[p][self.st.c4(nx as usize, ny as usize)] > 0) as u8
                }
            }
            _ => intra as u8,
        }
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

        if is_i16 {
            let mut sc = [0i16; 16];
            let nc = if cabac { 0 } else { self.nc_luma(mx4, my4) };
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
                let mut sc = [0i16; 64];
                let mut total = 0;
                if cabac {
                    let (cd, ctxs) = self.cabac();
                    total = cd
                        .decode_residual_block(&mut ctxs[..], 5, 0, &mut sc, field)
                        .1;
                    for i in 0..4 {
                        let blk = b8 * 4 + i;
                        let (bx, by) = BLK4_XY[blk];
                        let j = self.st.b4(mx4 + bx, my4 + by);
                        self.st.nnz[j] = total as u8;
                        self.st.nz[j] = total > 0;
                    }
                } else {
                    for i in 0..4 {
                        let blk = b8 * 4 + i;
                        let (bx, by) = BLK4_XY[blk];
                        let nc = self.nc_luma(mx4 + bx, my4 + by);
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
                    let l8 = if intra { 0 } else { 1 };
                    for k in 0..64 {
                        if sc[k] != 0 {
                            let pos = scan8[k];
                            res.luma8[b8][pos] = dq.scale8(l8, qp, pos, sc[k] as i32);
                        }
                    }
                    res.luma_coded |= 0xF << (b8 * 4);
                }
                continue;
            }
            for i in 0..4 {
                let blk = b8 * 4 + i;
                let (bx, by) = BLK4_XY[blk];
                let (x4, y4) = (mx4 + bx, my4 + by);
                let max = if is_i16 { 15 } else { 16 };
                let mut sc = [0i16; 16];
                let n = if cabac {
                    let inc = self.cbf_luma(x4 as isize - 1, y4 as isize, intra)
                        + 2 * self.cbf_luma(x4 as isize, y4 as isize - 1, intra);
                    let cat = if is_i16 { 1 } else { 2 };
                    let (cd, ctxs) = self.cabac();
                    cd.decode_residual_block(&mut ctxs[..], cat, inc, &mut sc[..max], field)
                        .1
                } else {
                    let nc = self.nc_luma(x4, y4);
                    let r = self.reader();
                    decode_cavlc_block(r, &mut sc[..max], nc, false)?
                };
                let j = self.st.b4(x4, y4);
                self.st.nnz[j] = n as u8;
                self.st.nz[j] = n > 0;
                if n > 0 {
                    let off = 16 - max;
                    for k in 0..max {
                        if sc[k] != 0 {
                            let pos = scan4[k + off];
                            res.luma[blk][pos] = dq.scale4(lst, qp, pos, sc[k] as i32);
                        }
                    }
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
                let (x4, y4) = (cx4 + (blk & 1), cy4 + (blk >> 1));
                let mut sc = [0i16; 15];
                let n = if cabac {
                    let inc = self.cbf_chroma_ac(p, x4 as isize - 1, y4 as isize, intra)
                        + 2 * self.cbf_chroma_ac(p, x4 as isize, y4 as isize - 1, intra);
                    let (cd, ctxs) = self.cabac();
                    cd.decode_residual_block(&mut ctxs[..], 4, inc, &mut sc, field)
                        .1
                } else {
                    let nc = self.nc_chroma(p, x4, y4);
                    let r = self.reader();
                    decode_cavlc_block(r, &mut sc, nc, false)?
                };
                let j = self.st.c4(x4, y4);
                self.st.nnz_c[p][j] = n as u8;
                if n > 0 {
                    for k in 0..15 {
                        if sc[k] != 0 {
                            let pos = scan4[k + 1];
                            res.chroma[p][blk][pos] =
                                dq.scale4(lst + 1 + p, qpc[p] as i32, pos, sc[k] as i32);
                        }
                    }
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
        self.cx.refs[list]
            .get(idx as usize)
            .map(|r| r.id)
            .ok_or(bad("H.264 ref_idx beyond the reference list"))
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
        for y in by..by + h {
            for x in bx..bx + w {
                let i = self.st.b4(self.mb.x * 4 + x, self.mb.y * 4 + y);
                self.st.ref_idx[list][i] = m.0;
                self.st.mv[list][i] = m.1;
                self.st.ref_pic[list][i] = pic;
                self.st.mvd[list][i] = mvd;
            }
        }
    }

    fn neighbour(&self, list: usize, x4: isize, y4: isize, done: u16) -> Option<Mv> {
        if !self.blk_avail(x4, y4, done) {
            return None;
        }
        let i = self.st.b4(x4 as usize, y4 as usize);
        Some((self.st.ref_idx[list][i], self.st.mv[list][i]))
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
        let (x4, y4) = ((self.mb.x * 4 + bx) as isize, (self.mb.y * 4 + by) as isize);
        let a = self.neighbour(list, x4 - 1, y4, done);
        let b = self.neighbour(list, x4, y4 - 1, done);
        let c = self
            .neighbour(list, x4 + w as isize, y4 - 1, done)
            .or_else(|| self.neighbour(list, x4 - 1, y4 - 1, done));
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

    fn p_skip_mv(&self) -> [i16; 2] {
        let (x4, y4) = ((self.mb.x * 4) as isize, (self.mb.y * 4) as isize);
        let a = self.neighbour(0, x4 - 1, y4, 0);
        let b = self.neighbour(0, x4, y4 - 1, 0);
        match (a, b) {
            (None, _) | (_, None) => [0, 0],
            (Some((0, [0, 0])), _) | (_, Some((0, [0, 0]))) => [0, 0],
            _ => self.mvp(0, 0, 0, 4, 0, None, 0),
        }
    }

    fn read_ref_idx(&mut self, list: usize, bx: usize, by: usize) -> Result<i8> {
        let n = self.cx.sh.num_ref_idx_active[list];
        if !self.is_cabac() {
            return self.read_te(n - 1);
        }
        let (x4, y4) = ((self.mb.x * 4 + bx) as isize, (self.mb.y * 4 + by) as isize);
        let is_b = self.cx.sh.slice_type.is_b();
        let cond = |nx: isize, ny: isize| -> u8 {
            let Some(addr) = self.mb_at4(nx, ny) else {
                return 0;
            };
            if addr != self.mb.addr && !self.mb_avail(addr) {
                return 0;
            }
            let t = self.st.mb_type[addr];
            if t.is_skip() || t.is_intra() {
                return 0;
            }
            let i = self.st.b4(nx as usize, ny as usize);
            if is_b && self.st.direct[i] {
                return 0;
            }
            (self.st.ref_idx[list][i] > 0) as u8
        };
        let (ca, cb) = (cond(x4 - 1, y4), cond(x4, y4 - 1));
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
        let (x4, y4) = ((self.mb.x * 4 + bx) as isize, (self.mb.y * 4 + by) as isize);
        let absn = |nx: isize, ny: isize, comp: usize| -> u32 {
            let Some(addr) = self.mb_at4(nx, ny) else {
                return 0;
            };
            if addr != self.mb.addr && !self.mb_avail(addr) {
                return 0;
            }
            self.st.mvd[list][self.st.b4(nx as usize, ny as usize)][comp] as u32
        };
        let sums = [0, 1].map(|comp| absn(x4 - 1, y4, comp) + absn(x4, y4 - 1, comp));
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
                let r = if self.cx.sh.num_ref_idx_active[list] > 1 {
                    self.read_ref_idx(list, bx, by)?
                } else {
                    0
                };
                refs[p][list] = r;
                // Store early so CABAC contexts of later partitions see it.
                for y in by..by + h {
                    for x in bx..bx + w {
                        let i = self.st.b4(self.mb.x * 4 + x, self.mb.y * 4 + y);
                        self.st.ref_idx[list][i] = r;
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
                let r = if self.cx.sh.num_ref_idx_active[list] > 1 && !ref0 {
                    self.read_ref_idx(list, bx, by)?
                } else {
                    0
                };
                refs[i][list] = r;
                for y in by..by + 2 {
                    for x in bx..bx + 2 {
                        let j = self.st.b4(self.mb.x * 4 + x, self.mb.y * 4 + y);
                        self.st.ref_idx[list][j] = r;
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
        if self.cx.refs[1].is_empty() || self.cx.refs[0].is_empty() {
            return Err(bad("H.264 direct prediction without reference lists"));
        }
        let col = &self.cx.refs[1][0];
        let inference = self.cx.sps.direct_8x8_inference;
        let spatial = self.cx.sh.direct_spatial;
        let (mx4, my4) = (self.mb.x * 4, self.mb.y * 4);

        // Spatial: reference indices and predictors from the MB neighbours (8.4.1.2.2).
        let mut sp_ref = [-1i8; 2];
        let mut sp_mv = [[0i16; 2]; 2];
        let mut direct_zero = false;
        if spatial {
            for list in 0..2 {
                let (x4, y4) = (mx4 as isize, my4 as isize);
                let a = self.neighbour(list, x4 - 1, y4, 0);
                let b = self.neighbour(list, x4, y4 - 1, 0);
                let c = self
                    .neighbour(list, x4 + 4, y4 - 1, 0)
                    .or_else(|| self.neighbour(list, x4 - 1, y4 - 1, 0));
                let rr = |n: Option<Mv>| n.map_or(-1, |n| n.0);
                sp_ref[list] = min_positive(rr(a), min_positive(rr(b), rr(c)));
            }
            if sp_ref[0] < 0 && sp_ref[1] < 0 {
                sp_ref = [0, 0];
                direct_zero = true;
            } else {
                for list in 0..2 {
                    if sp_ref[list] >= 0 {
                        sp_mv[list] = self.mvp(list, 0, 0, 4, sp_ref[list], None, 0);
                    }
                }
            }
        }

        let col_short = !col.long;
        for b8 in 0..4 {
            if mask & (1 << b8) == 0 {
                continue;
            }
            let (bx8, by8) = ((b8 & 1) * 2, (b8 >> 1) * 2);
            for sub in 0..4 {
                let (bx, by) = (bx8 + (sub & 1), by8 + (sub >> 1));
                // Co-located 4x4 block (corner of the 8x8 with direct_8x8_inference).
                let (cbx, cby) = if inference {
                    ((b8 & 1) * 3, (b8 >> 1) * 3)
                } else {
                    (bx, by)
                };
                let (col_ref, col_mv, col_pic, vscale) = self.colocated(cbx, cby);
                let (refs, mvs) = if spatial {
                    let col_zero = col_short
                        && col_ref == 0
                        && (-1..=1).contains(&col_mv[0])
                        && (-1..=1).contains(&col_mv[1]);
                    let mut mvs = [[0i16; 2]; 2];
                    for list in 0..2 {
                        if direct_zero || sp_ref[list] < 0 || (sp_ref[list] == 0 && col_zero) {
                            mvs[list] = [0, 0];
                        } else {
                            mvs[list] = sp_mv[list];
                        }
                    }
                    (sp_ref, mvs)
                } else {
                    // Temporal (8.4.1.2.3): map the co-located reference into RefPicList0 and
                    // convert its vertical vector between frame and field units.
                    let target = match self.st.structure {
                        Structure::Frame => col_pic & !3,
                        Structure::Field(p) if col_pic % 4 == 0 => col_pic + 1 + p as u32,
                        Structure::Field(_) => col_pic,
                    };
                    let ref0 = if col_ref < 0 {
                        0
                    } else {
                        self.cx.refs[0]
                            .iter()
                            .position(|r| r.id == target)
                            .unwrap_or(0) as i8
                    };
                    let col_mv = match vscale {
                        1 => [col_mv[0], col_mv[1].wrapping_mul(2)],
                        -1 => [col_mv[0], col_mv[1] / 2],
                        _ => col_mv,
                    };
                    let p0 = &self.cx.refs[0][ref0 as usize];
                    let p1 = &self.cx.refs[1][0];
                    let td = (p1.poc - p0.poc).clamp(-128, 127);
                    let (mv0, mv1) = if p0.long || td == 0 {
                        (col_mv, [0, 0])
                    } else {
                        let tb = (self.cx.cur_poc - p0.poc).clamp(-128, 127);
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
                for list in 0..2 {
                    let r = refs[list];
                    let (r, mv) = if r < 0 { (-1, [0, 0]) } else { (r, mvs[list]) };
                    let id = self.ref_id(list, r)?;
                    self.fill_motion(bx, by, 1, 1, list, (r, mv), id, [0, 0]);
                }
                let i = self.st.b4(mx4 + bx, my4 + by);
                self.st.direct[i] = true;
            }
        }
        Ok(())
    }

    /// Co-located block for direct prediction (8.4.1.2.1): `(refIdxCol, mvCol, referenced picture id,
    /// vertical scale)` where the scale is `1` for field-to-frame, `-1` for frame-to-field, else `0`.
    /// `(cbx, cby)` is the MB-relative 4x4 block, after the 8x8-inference corner selection.
    fn colocated(&self, cbx: usize, cby: usize) -> (i8, [i16; 2], u32, i32) {
        let col = &self.cx.refs[1][0];
        let Some(m) = &col.motion else {
            return (-1, [0, 0], NO_PIC, 0);
        };
        let mb_w = self.st.mb_w;
        let mbx = self.mb.x;
        // Motion is stored in frame layout; pick the co-located MB row and block row there.
        let (row, by, vscale) = match self.st.structure {
            Structure::Frame => {
                let row = self.mb.y;
                if m.mb_field[row * mb_w + mbx] {
                    let cur = self.cx.cur_poc;
                    let cp = usize::from(
                        (col.field_poc[0] - cur).abs() >= (col.field_poc[1] - cur).abs(),
                    );
                    ((row & !1) + cp, 2 * (row % 2) + cby / 2, 1)
                } else {
                    (row, cby, 0)
                }
            }
            Structure::Field(_) => {
                let cp = col.parity.unwrap_or(0) as usize;
                let r = self.mb.y;
                if m.mb_field[(2 * r + cp) * mb_w + mbx] {
                    (2 * r + cp, cby, 0)
                } else {
                    (2 * r + cby / 2, 2 * (cby % 2), -1)
                }
            }
        };
        let mb = row * mb_w + mbx;
        if mb >= m.mb_intra.len() || m.mb_intra[mb] {
            return (-1, [0, 0], NO_PIC, vscale);
        }
        let i = (row * 4 + by) * mb_w * 4 + mbx * 4 + cbx;
        let l = if m.ref_idx[0][i] >= 0 { 0 } else { 1 };
        (m.ref_idx[l][i], m.mv[l][i], m.ref_pic[l][i], vscale)
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
        let mut res = Residual::new();
        if cbp != 0 {
            self.read_qp_delta()?;
            self.set_qp();
            self.read_residual(cbp, false, t8, false, &mut res)?;
        } else {
            self.last_dqp_nz = false;
            self.set_qp();
        }
        self.inter_predict_mb()?;
        let w = self.st.width;
        let (px, py) = (self.mb.x * 16, self.mb.y * 16);
        if t8 {
            for b8 in 0..4 {
                if res.luma_coded & (0xF << (b8 * 4)) != 0 {
                    let off = (py + (b8 >> 1) * 8) * w + px + (b8 & 1) * 8;
                    recon::idct8_add(&mut self.st.planes[0][off..], w, &res.luma8[b8]);
                }
            }
        } else {
            for blk in 0..16 {
                if res.luma_coded & (1 << blk) != 0 {
                    let (bx, by) = BLK4_XY[blk];
                    let off = (py + by * 4) * w + px + bx * 4;
                    recon::idct4_add(&mut self.st.planes[0][off..], w, &res.luma[blk]);
                }
            }
        }
        if self.cx.sps.chroma_array_type() != 0 {
            self.add_chroma_residual(&res);
        }
        Ok(())
    }

    fn weighting(&self, refs: [i8; 2], comp: usize) -> Weighting {
        if let Some(t) = &self.cx.sh.weights {
            let log = if comp == 0 {
                t.luma_log2
            } else {
                t.chroma_log2
            };
            let e = |list: usize| -> (i32, i32) {
                let r = refs[list];
                if r < 0 {
                    return (1, 0);
                }
                let Some(e) = t.l[list].get(r as usize) else {
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
        if !self.implicit.is_empty() && refs[0] >= 0 && refs[1] >= 0 {
            let n1 = self.cx.refs[1].len();
            let w = self.implicit[refs[0] as usize * n1 + refs[1] as usize];
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
            (s.ref_idx[0][j], s.mv[0][j], s.ref_idx[1][j], s.mv[1][j])
        };
        let same = |s: &PicState, bx: usize, by: usize, n: usize| {
            let k = key(s, bx, by);
            (by..by + n).all(|y| (bx..bx + n).all(|x| key(s, x, y) == k))
        };
        if same(self.st, 0, 0, 4) {
            let k = key(self.st, 0, 0);
            return self.mc_rect(0, 0, 4, 4, [k.0, k.2], [k.1, k.3]);
        }
        for b8 in 0..4 {
            let (bx, by) = ((b8 & 1) * 2, (b8 >> 1) * 2);
            if same(self.st, bx, by, 2) {
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
        let (x, y) = (self.mb.x * 16 + bx * 4, self.mb.y * 16 + by * 4);
        let (pw, ph) = (self.st.width, self.st.height);
        let mut pred = [[0u8; 256]; 2];
        let mut used = [false; 2];
        for list in 0..2 {
            if refs[list] < 0 {
                continue;
            }
            let rp = self.cx.refs[list]
                .get(refs[list] as usize)
                .ok_or(bad("H.264 ref_idx beyond list"))?;
            recon::mc_luma(
                &rp.planes[0],
                pw,
                ph,
                x,
                y,
                w,
                h,
                mvs[list],
                &mut pred[list],
            );
            used[list] = true;
        }
        if !used[0] && !used[1] {
            return Err(bad("H.264 inter block without prediction"));
        }
        let wt = self.weighting(refs, 0);
        let mut out = [0u8; 256];
        recon::weighted_combine(
            used[0].then_some(&pred[0][..w * h]),
            used[1].then_some(&pred[1][..w * h]),
            wt,
            &mut out[..w * h],
        );
        for r in 0..h {
            self.st.planes[0][(y + r) * pw + x..(y + r) * pw + x + w]
                .copy_from_slice(&out[r * w..r * w + w]);
        }
        if self.cx.sps.chroma_array_type() == 0 {
            return Ok(());
        }
        let (cw, ch, cpw, cph) = (w / 2, h / 2, pw / 2, ph / 2);
        let (cx, cy) = (x / 2, y / 2);
        for p in 0..2 {
            let mut pc = [[0u8; 64]; 2];
            for list in 0..2 {
                if used[list] {
                    let rp = &self.cx.refs[list][refs[list] as usize];
                    // Between fields of opposite parity the chroma vector is offset by a
                    // quarter chroma sample (Table 8-9).
                    let mut mv = mvs[list];
                    if let (Structure::Field(cur), Some(r)) = (self.st.structure, rp.parity) {
                        mv[1] = mv[1].wrapping_add(2 * (cur as i16 - r as i16));
                    }
                    recon::mc_chroma(
                        &rp.planes[1 + p],
                        cpw,
                        cph,
                        cx,
                        cy,
                        cw,
                        ch,
                        mv,
                        &mut pc[list],
                    );
                }
            }
            let wt = self.weighting(refs, 1 + p);
            let mut o = [0u8; 64];
            recon::weighted_combine(
                used[0].then_some(&pc[0][..cw * ch]),
                used[1].then_some(&pc[1][..cw * ch]),
                wt,
                &mut o[..cw * ch],
            );
            for r in 0..ch {
                self.st.planes[1 + p][(cy + r) * cpw + cx..(cy + r) * cpw + cx + cw]
                    .copy_from_slice(&o[r * cw..r * cw + cw]);
            }
        }
        Ok(())
    }
}

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
