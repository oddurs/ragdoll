//! Slice data and macroblock layer decoding (ITU-T H.264 7.3.4, 7.3.5, 8.3, 8.4, 8.5).

#![allow(clippy::needless_range_loop)]

use std::sync::Arc;

use ragdoll_core::{BitReader, RagdollError, Result};

use crate::cabac::{
    CabacContext, CabacDecoder, NUM_CABAC_CONTEXTS, ResidualCoeffs, init_cabac_contexts,
};
use crate::h264::{decode_cavlc_block, decode_cavlc_coeffs};

use super::cabac_se;
use super::intra;
use super::picture::{
    BlockInfo, MbType, Motion, MotionView, NO_BLOCK, NO_PIC, PicState, PicSync, RefIds, Structure,
    fill_block_rect,
};
use super::ps::{BASELINE, Pps, Sps};
use super::recon::{self, Dequant, Weighting};
use super::slice::{SliceHeader, SliceType};
use super::tables::{
    B_MB_TYPES, B_SUB_TYPES, BLK4_AT, BLK4_XY, CBP_ME_420, CBP_ME_MONO, FIELD_SCAN4X4,
    FIELD_SCAN8X8, P_SUB_TYPES, PRED_L0, Part, SCAN_MAP4, SCAN4X4, SCAN8X8, chroma_qp,
};

/// A reference picture as seen by one slice.
pub struct RefPic {
    /// Frame or field reference id (see `FrameStore::frame_ref_id`).
    pub id: u32,
    pub poc: i32,
    pub long: bool,
    /// Samples of the frame, or of the field for a field reference.
    pub planes: Arc<[ragdoll_core::AlignedBuffer; 3]>,
    /// Motion of the frame store, in frame layout.
    pub motion: Option<Arc<Motion>>,
    /// Field parity for a field reference.
    pub parity: Option<u8>,
    /// POCs of the store's two fields.
    pub field_poc: [i32; 2],
    /// Progress of a frame still being decoded on another thread, which prediction from it
    /// waits on (its `motion` is then `None` until it is complete).
    pub sync: Option<Arc<PicSync>>,
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

/// [`Dec::nb`] bits.
const NB_L: u8 = 1;
const NB_T: u8 = 2;
const NB_TR: u8 = 4;
const NB_TL: u8 = 8;

fn bad(msg: &'static str) -> RagdollError {
    RagdollError::InvalidSyntax(msg)
}

/// Decodes the slice data of one slice into `st`.
pub fn decode_slice(cx: &SliceCtx<'_>, st: &mut PicState, rbsp: &[u8]) -> Result<()> {
    // One decoder instance per entropy coder and MBAFF keeps each one's hot code small.
    match (cx.pps.cabac, st.mbaff) {
        (true, false) => decode_slice_as::<true, false>(cx, st, rbsp),
        (false, false) => decode_slice_as::<false, false>(cx, st, rbsp),
        (true, true) => decode_slice_as::<true, true>(cx, st, rbsp),
        (false, true) => decode_slice_as::<false, true>(cx, st, rbsp),
    }
}

/// [`decode_slice`] for CABAC (`CABAC`) or CAVLC slices, of an MBAFF frame (`MBAFF`) or not.
fn decode_slice_as<const CABAC: bool, const MBAFF: bool>(
    cx: &SliceCtx<'_>,
    st: &mut PicState,
    rbsp: &[u8],
) -> Result<()> {
    let sh = cx.sh;
    let mut rd = BitReader::new(rbsp);
    rd.skip_bits(sh.data_bit_offset)?;
    // A CAVLC slice's CABAC decoder is never used: it reads two zero bytes.
    let cd = if CABAC {
        rd.align_to_byte();
        let pos = rd.bits_read() / 8;
        CabacDecoder::new(rbsp.get(pos..).ok_or(RagdollError::BitstreamEndOfData)?)?
    } else {
        CabacDecoder::new(&[0, 0])?
    };
    let ctxs = Box::new(init_cabac_contexts(
        sh.qp as u8,
        sh.slice_type.is_intra(),
        sh.cabac_init_idc as u8,
    ));
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
    let (st_mbaff, st_structure) = (st.mbaff, st.structure);
    let table_base = st.ref_tables.len() as u16;
    st.push_ref_table(ref_ids(&cx.refs));
    if let Some(f) = &cx.field_refs {
        st.push_ref_table(ref_ids(&f[0]));
        st.push_ref_table(ref_ids(&f[1]));
    }
    let mut d = Dec::<CABAC, MBAFF> {
        cx,
        st,
        rd,
        cd,
        ctxs,
        qp: sh.qp,
        qpc: {
            let o = cx.pps.chroma_qp_index_offset;
            [chroma_qp(sh.qp, o[0]), chroma_qp(sh.qp, o[1])]
        },
        last_dqp_nz: false,
        mb: MbPos {
            addr: 0,
            x: 0,
            y: 0,
        },
        field: cx.sh.field_pic,
        implicit,
        implicit_field,
        unit_weights: cx.sh.weights.as_ref().map_or([0; 2], |t| {
            [0, 1].map(|l| {
                (0..32).fold(0u32, |m, r| {
                    let unit = |(w, o): (i32, i32), log: u32| w == 1 << log && o == 0;
                    let def = t.l[l].get(r).is_none_or(|e| {
                        unit(e.luma, t.luma_log2)
                            && unit(e.chroma[0], t.chroma_log2)
                            && unit(e.chroma[1], t.chroma_log2)
                    });
                    m | (def as u32) << r
                })
            })
        }),
        scratch: [[0; 256]; 2],
        partial_motion: false,
        simple: !st_mbaff
            && matches!(st_structure, Structure::Frame)
            && cx.sps.chroma_array_type() == 1,
        nb: 0,
        nb_addr: usize::MAX,
        table_base,
        whole: cx.slice_num == 0 && cx.sh.first_mb == 0,
        res: None,
        waits: cx.refs.iter().flatten().any(|r| r.sync.is_some()),
        col_prefetch: cx.sh.slice_type.is_b() && !st_mbaff && st_structure == Structure::Frame,
        mck: ragdoll_simd::H264McKernels::new(),
        mc_fast: false,
        mc_direct: false,
        weighted: false,
    };
    let n = d.st.width * d.st.height;
    d.mc_fast = d.simple
        && d.st.planes[0].len() >= n
        && d.cx.refs.iter().flatten().all(|r| {
            r.planes[0].len() >= n && r.planes[1].len() >= n / 4 && r.planes[2].len() >= n / 4
        });
    d.mc_direct = d.mc_fast && !d.waits;
    d.weighted = d.cx.sh.weights.is_some() || !d.implicit.is_empty();
    let r = d.run(first, total);
    if r.is_err() {
        // Filtering resumes after undecoded macroblocks are restored.
        st.rows.on = false;
    }
    r
}

/// The ids of the pictures in reference lists `lists`, as a table of [`PicState::ref_tables`].
pub fn ref_ids(lists: &[Vec<RefPic>; 2]) -> RefIds {
    let mut t = [[NO_PIC; 32]; 2];
    for (row, list) in t.iter_mut().zip(lists) {
        for (id, r) in row.iter_mut().zip(list) {
            *id = r.id;
        }
    }
    t
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

/// A slice's decoder, for a CABAC (`CABAC`) or CAVLC slice, of an MBAFF frame (`MBAFF`, as
/// `st.mbaff`) or not.
struct Dec<'a, 'c, const CABAC: bool, const MBAFF: bool> {
    cx: &'c SliceCtx<'a>,
    st: &'c mut PicState,
    /// CAVLC: the slice data.
    rd: BitReader<'a>,
    /// CABAC: the arithmetic decoder and its context variables.
    cd: CabacDecoder<'a>,
    ctxs: Box<[CabacContext; NUM_CABAC_CONTEXTS]>,
    qp: i32,
    /// Chroma QPs (Cb, Cr) of [`Self::qp`].
    qpc: [u8; 2],
    last_dqp_nz: bool,
    mb: MbPos,
    /// The current macroblock is a field macroblock (field picture or MBAFF field pair).
    field: bool,
    implicit: Vec<[i32; 2]>,
    implicit_field: [Vec<[i32; 2]>; 2],
    /// Explicit weights: per list, bit `r` set when reference index `r` (0..32) has default
    /// weights and zero offsets in every component.
    unit_weights: [u32; 2],
    /// Prediction samples of the second list, or of both for weighted prediction.
    scratch: [[u8; 256]; 2],
    /// The current MB's motion was stored for less than the whole MB, so it may vary.
    partial_motion: bool,
    /// Progressive 4:2:0 frame without MBAFF or explicit weights: inter prediction takes
    /// [`Self::mc_rect_simple`].
    simple: bool,
    /// Without MBAFF: which of the current MB's left, top, top-right and top-left neighbours
    /// are available ([`NB_L`] .. [`NB_TL`]), found once per macroblock.
    nb: u8,
    /// The address [`Self::nb`] was found for.
    nb_addr: usize,
    /// This slice's first entry of [`PicState::ref_tables`]: its frame lists, then (MBAFF) its
    /// top and bottom field lists.
    table_base: u16,
    /// The slice is the picture's first and starts at its first macroblock: every macroblock
    /// before the current one belongs to it.
    whole: bool,
    /// Residual blocks, kept zeroed between macroblocks: the inverse transforms clear what they
    /// read, so only coded coefficients are ever written.
    res: Option<Box<Residual>>,
    /// Some reference picture is still being decoded on another thread: inter prediction waits
    /// for the rows it reads.
    waits: bool,
    /// A frame B slice: direct prediction reads RefPicList1[0]'s motion, prefetched ahead.
    col_prefetch: bool,
    /// Inter-prediction kernels of [`Self::mc_pred`].
    mck: ragdoll_simd::H264McKernels,
    /// [`Self::simple`], and every reference picture's planes are as large as the picture's:
    /// inter prediction takes [`Self::mc_rect_simple`].
    mc_fast: bool,
    /// [`Self::mc_fast`] without references still being decoded: macroblocks of one motion
    /// take [`Self::mc_whole`].
    mc_direct: bool,
    /// The slice has explicit or implicit weights.
    weighted: bool,
}

impl<'a, 'c, const CABAC: bool, const MBAFF: bool> Dec<'a, 'c, CABAC, MBAFF> {
    fn run(&mut self, first: usize, total: usize) -> Result<()> {
        let slice_type = self.cx.sh.slice_type;
        let mbaff = MBAFF;
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
                        self.st.mbs[m].mb_type = if slice_type.is_b() {
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
        let (x, y) = if MBAFF {
            let pair = addr / 2;
            (pair % w, 2 * (pair / w) + addr % 2)
        } else if addr == self.mb.addr + 1 {
            // Decoding order without MBAFF is raster order: the next MB needs no division.
            if self.mb.x + 1 < w {
                (self.mb.x + 1, self.mb.y)
            } else {
                (0, self.mb.y + 1)
            }
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
        let same = |a: usize| self.st.mbs[a].slice_num == self.cx.slice_num;
        if p.x > 0 && same(p.addr - 1) {
            self.st.mbs[p.addr - 1].mb_field
        } else if p.y >= 2 && same(p.addr - 2 * w) {
            self.st.mbs[p.addr - 2 * w].mb_field
        } else {
            false
        }
    }

    /// CABAC mb_field_decoding_flag of the pair whose top MB has decoding address `addr`.
    fn decode_field_flag(&mut self, addr: usize) -> bool {
        let p = self.pos(addr);
        let w = self.st.mb_w;
        let cond = |a: usize| {
            (self.st.mbs[a].slice_num == self.cx.slice_num && self.st.mbs[a].mb_field) as usize
        };
        let inc = if p.x > 0 { cond(p.addr - 1) } else { 0 }
            + if p.y >= 2 { cond(p.addr - 2 * w) } else { 0 };
        let (cd, ctxs) = self.cabac();
        cd.decode_decision(&mut ctxs[70 + inc]) != 0
    }

    /// CABAC mb_skip_flag of the macroblock at decoding address `addr`.
    #[inline(always)]
    fn decode_skip_flag(&mut self, addr: usize) -> bool {
        // MBAFF reads a pair's second skip flag before the first macroblock is decoded.
        let saved = (self.mb, self.nb);
        self.set_mb(addr);
        let a = self
            .mb_left()
            .filter(|&m| !self.st.mbs[m].mb_type.is_skip())
            .is_some() as u8;
        let b = self
            .mb_top()
            .filter(|&m| !self.st.mbs[m].mb_type.is_skip())
            .is_some() as u8;
        if MBAFF {
            (self.mb, self.nb) = saved;
        }
        let is_b = self.cx.sh.slice_type.is_b();
        let (cd, ctxs) = self.cabac();
        if is_b {
            cd.decode_b_skip(&mut ctxs[..], a, b)
        } else {
            cd.decode_p_skip(&mut ctxs[..], a, b)
        }
    }

    fn reader(&mut self) -> &mut BitReader<'a> {
        debug_assert!(!CABAC, "CAVLC syntax read in a CABAC slice");
        &mut self.rd
    }

    fn cabac(
        &mut self,
    ) -> (
        &mut CabacDecoder<'a>,
        &mut [CabacContext; NUM_CABAC_CONTEXTS],
    ) {
        debug_assert!(CABAC, "CABAC syntax read in a CAVLC slice");
        (&mut self.cd, &mut self.ctxs)
    }

    /// Whether the current macroblock is a field macroblock.
    fn field_mb(&self) -> bool {
        self.field
    }

    /// MBAFF field macroblock: uses the field reference lists of its parity.
    fn mbaff_field(&self) -> bool {
        MBAFF && self.field
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
        CABAC
    }

    // ----- neighbourhood -------------------------------------------------------------------

    /// Makes `addr` the current macroblock, finding its neighbours' availability.
    fn set_mb(&mut self, addr: usize) {
        self.mb = self.pos(addr);
        if MBAFF {
            return;
        }
        let (x, y, w, a) = (self.mb.x, self.mb.y, self.st.mb_w, self.mb.addr);
        self.nb_addr = addr;
        self.st.row_mb(addr, x, y);
        self.st.prefetch_mb_blocks(x + 2, y);
        if self.simple {
            self.st.prefetch_mb_samples(x, y);
        }
        // Direct prediction reads the co-located macroblock's corner blocks.
        if self.col_prefetch
            && let Some(m) = self.cx.refs[1].first().and_then(|r| r.motion.as_deref())
        {
            m.prefetch_corners(x + 2, y, w);
        }
        if self.whole {
            let (l, t, r) = (x > 0, y > 0, x + 1 < w);
            self.nb = (l as u8 * NB_L)
                | (t as u8 * NB_T)
                | ((t && r) as u8 * NB_TR)
                | ((t && l) as u8 * NB_TL);
            return;
        }
        let (mbs, cur) = (&self.st.mbs, self.cx.slice_num);
        let mut nb = 0;
        if x > 0 && mbs[a - 1].slice_num == cur {
            nb |= NB_L;
        }
        if y > 0 {
            if mbs[a - w].slice_num == cur {
                nb |= NB_T;
            }
            if x + 1 < w && mbs[a - w + 1].slice_num == cur {
                nb |= NB_TR;
            }
            if x > 0 && mbs[a - w - 1].slice_num == cur {
                nb |= NB_TL;
            }
        }
        self.nb = nb;
    }

    fn begin_mb(&mut self, addr: usize) {
        if MBAFF || self.nb_addr != addr {
            self.set_mb(addr);
        }
        let a = self.mb.addr;
        // Every macroblock type writes all of its blocks' motion (see `reset_blocks`).
        if self.st.recycled {
            self.st.clear_mb_state(self.mb.x, self.mb.y);
        }
        self.partial_motion = false;
        // The reference lists `refs` reads, as a table of `st.ref_tables`.
        let table = self.table_base
            + match &self.cx.field_refs {
                Some(_) if self.field => 1 + (self.mb.y & 1) as u16,
                _ => 0,
            };
        let m = &mut self.st.mbs[a];
        m.mb_table = table;
        m.slice_num = self.cx.slice_num;
        m.mb_field = self.field;
    }

    fn mb_avail(&self, addr: usize) -> bool {
        self.st.mbs[addr].slice_num == self.cx.slice_num && addr != self.mb.addr
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
        if !MBAFF {
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
            self.st.mbs[top].mb_field
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
        if MBAFF {
            return self.mb_nb(-1, 0);
        }
        (self.nb & NB_L != 0).then(|| self.mb.addr - 1)
    }

    /// The available macroblock above the current one.
    #[inline]
    fn mb_top(&self) -> Option<usize> {
        if MBAFF {
            return self.mb_nb(0, -1);
        }
        (self.nb & NB_T != 0).then(|| self.mb.addr - self.st.mb_w)
    }

    /// The available 4x4 block covering luma `(xn, yn)`: `(macroblock, 4x4 index)`. Blocks of the
    /// current MB are available when their bit is set in `done` (by `luma4x4BlkIdx`).
    fn blk(&self, xn: isize, yn: isize, done: u16) -> Option<(usize, usize)> {
        if !MBAFF {
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
            } else {
                // Neighbour MB by offset `(dx + 1, dy + 1)`: only those above or to the left
                // precede the current one in decoding order.
                let (dx, dy) = ((px >> 4) + 1 - self.mb.x, (py >> 4) + 1 - self.mb.y);
                const NB_AT: [[u8; 3]; 3] = [[NB_TL, NB_T, NB_TR], [NB_L, 0, 0], [0, 0, 0]];
                let bit = NB_AT.get(dy).and_then(|r| r.get(dx)).copied().unwrap_or(0);
                if self.nb & bit == 0 {
                    return None;
                }
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
                    && !(self.cx.pps.constrained_intra_pred && !self.st.mbs[a].mb_type.is_intra())
            }
        }
    }

    /// Intra availability of the samples left of luma rows `y0..y0 + n` of the current MB. In MBAFF
    /// frames these rows can come from both macroblocks of the left pair.
    fn left_ok(&self, y0: usize, n: usize) -> bool {
        let rows = if MBAFF { n } else { 1 };
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

    /// Decodes a skipped macroblock; on an error its blocks are reset rather than left
    /// half-written.
    fn decode_skip(&mut self, addr: usize) -> Result<()> {
        self.decode_skip_mb(addr)
            .inspect_err(|_| self.reset_blocks())
    }

    fn decode_skip_mb(&mut self, addr: usize) -> Result<()> {
        self.begin_mb(addr);
        let a = self.mb.addr;
        let is_b = self.cx.sh.slice_type.is_b();
        let m = &mut self.st.mbs[a];
        m.qp = self.qp as u8;
        m.qpc = self.qpc;
        m.cbp = 0;
        m.t8x8 = false;
        m.mb_type = if is_b { MbType::BSkip } else { MbType::PSkip };
        self.last_dqp_nz = false;
        if is_b {
            self.direct_motion(0b1111)?;
        } else {
            let mv = self.p_skip_mv();
            self.check_ref(0, 0)?;
            self.fill_blocks(0, 0, 4, 4, 0, (0, mv), [0, 0]);
        }
        self.inter_predict_mb()?;
        Ok(())
    }

    fn chroma_qps(&self, qp: i32) -> [u8; 2] {
        let o = self.cx.pps.chroma_qp_index_offset;
        [chroma_qp(qp, o[0]), chroma_qp(qp, o[1])]
    }

    /// Decodes a coded macroblock; on an error its blocks are reset rather than left
    /// half-written.
    fn decode_mb(&mut self, addr: usize) -> Result<()> {
        self.decode_coded_mb(addr)
            .inspect_err(|_| self.reset_blocks())
    }

    fn decode_coded_mb(&mut self, addr: usize) -> Result<()> {
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
            self.reset_blocks();
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
                        self.st.mbs[a].mb_type != MbType::I4x4
                            && self.st.mbs[a].mb_type != MbType::I8x8
                    }) as u8
                };
                let (ca, cb) = (c(left), c(top));
                let (cd, ctxs) = self.cabac();
                cd.decode_intra_mb_type(&mut ctxs[..], 3, ca, cb)
            }
            SliceType::B => {
                let c = |m: Option<usize>| {
                    m.is_some_and(|a| {
                        !matches!(self.st.mbs[a].mb_type, MbType::BSkip | MbType::BDirect16x16)
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
        let c = |m: Option<usize>| m.is_some_and(|a| self.st.mbs[a].t8x8) as usize;
        let inc = c(self.mb_left()) + c(self.mb_top());
        let (cd, ctxs) = self.cabac();
        Ok(cabac_se::transform_8x8(cd, &mut ctxs[..], inc))
    }

    fn read_chroma_pred_mode(&mut self) -> Result<u8> {
        let m = if self.is_cabac() {
            let c = |m: Option<usize>| {
                m.is_some_and(|a| {
                    let t = self.st.mbs[a].mb_type;
                    t.is_intra() && t != MbType::IPcm && self.st.mbs[a].chroma_pred_mode != 0
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
                let t = self.st.mbs[addr].mb_type;
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

    /// Without MBAFF: the Intra4x4/Intra8x8 prediction modes around the current macroblock for
    /// [`intra::predicted_mode`], row 0 those of the 4x4 blocks above it and column 0 those left
    /// of it; [`intra::NO_MODE`] where a neighbour is unavailable and 2 where it is not coded in
    /// Intra_4x4 or Intra_8x8 (8.3.1.1).
    fn intra_mode_grid(&self) -> Option<intra::ModeGrid> {
        if MBAFF {
            return None;
        }
        let w4 = self.st.mb_w * 4;
        let i = self.st.b4(self.mb.x * 4, self.mb.y * 4);
        let modes = |mb: Option<usize>, j: usize, step: usize| -> [i8; 4] {
            let Some(a) = mb else {
                return [intra::NO_MODE; 4];
            };
            let t = self.st.mbs[a].mb_type;
            if !t.is_intra() && self.cx.pps.constrained_intra_pred {
                [intra::NO_MODE; 4]
            } else if !matches!(t, MbType::I4x4 | MbType::I8x8) {
                [2; 4]
            } else {
                std::array::from_fn(|k| self.st.intra_mode[j + k * step])
            }
        };
        let left = modes(self.mb_left(), i.wrapping_sub(1), w4);
        let top = modes(self.mb_top(), i.wrapping_sub(w4), 1);
        let mut g = [[0; 8]; 8];
        g[0][1..5].copy_from_slice(&top);
        for (row, l) in g[1..5].iter_mut().zip(left) {
            row[0] = l;
        }
        Some(g)
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
            match self.st.mbs[a].mb_type {
                MbType::IPcm => 1,
                t if t.is_skip() => 0,
                _ => (self.st.mbs[a].cbp >> ((yw / 8) * 2 + xw / 8)) & 1,
            }
        };
        let (la, tb) = if MBAFF {
            (
                [nb_bit(-1, 0), nb_bit(-1, 8)],
                [nb_bit(0, -1), nb_bit(8, -1)],
            )
        } else {
            // The left macroblock's right 8x8s (bits 1, 3) and the top one's bottom 8x8s
            // (bits 2, 3); all set when unavailable or I_PCM, clear when skipped.
            let bits = |m: Option<usize>| -> u8 {
                match m {
                    None => 0xF,
                    Some(a) => match self.st.mbs[a].mb_type {
                        MbType::IPcm => 0xF,
                        t if t.is_skip() => 0,
                        _ => self.st.mbs[a].cbp & 0xF,
                    },
                }
            };
            let (l, t) = (bits(self.mb_left()), bits(self.mb_top()));
            ([(l >> 1) & 1, (l >> 3) & 1], [(t >> 2) & 1, (t >> 3) & 1])
        };
        let chroma_of = |m: Option<usize>| -> u8 {
            match m {
                None => 0,
                Some(a) if self.st.mbs[a].mb_type == MbType::IPcm => 2,
                Some(a) if self.st.mbs[a].mb_type.is_skip() => 0,
                Some(a) => self.st.mbs[a].cbp >> 4,
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
        self.qpc = self.chroma_qps(self.qp);
        Ok(())
    }

    fn set_qp(&mut self) {
        let a = self.mb.addr;
        self.st.mbs[a].qp = self.qp as u8;
        self.st.mbs[a].qpc = self.qpc;
    }

    // ----- intra macroblocks ----------------------------------------------------------------

    #[inline(never)]
    fn decode_intra_mb(&mut self, t: u32) -> Result<()> {
        let a = self.mb.addr;
        self.st.mbs[a].t8x8 = false;
        if t == 25 {
            return self.decode_pcm();
        }
        let mut modes = [0u8; 16];
        let mut t8 = false;
        let (cbp, kind) = if t == 0 {
            if self.cx.pps.transform_8x8_mode {
                t8 = self.read_transform_8x8()?;
            }
            self.st.mbs[a].t8x8 = t8;
            self.st.mbs[a].mb_type = if t8 { MbType::I8x8 } else { MbType::I4x4 };
            let mut grid = self.intra_mode_grid();
            let (i0, w4) = (self.st.b4(self.mb.x * 4, self.mb.y * 4), self.st.mb_w * 4);
            if t8 {
                for b8 in 0..4 {
                    let (bx, by) = ((b8 & 1) * 2, (b8 >> 1) * 2);
                    let pred = match &grid {
                        Some(g) => intra::predicted_mode(g, bx, by),
                        None => self.predicted_intra_mode(bx, by),
                    };
                    let m = self.read_pred_mode(pred)?;
                    modes[b8] = m;
                    let i = i0 + by * w4 + bx;
                    for j in [i, i + 1, i + w4, i + w4 + 1] {
                        self.st.intra_mode[j] = m as i8;
                    }
                    if let Some(g) = &mut grid {
                        intra::set_mode(g, bx, by, 2, m);
                    }
                }
            } else {
                for (blk, slot) in modes.iter_mut().enumerate() {
                    let (bx, by) = BLK4_XY[blk];
                    let pred = match &grid {
                        Some(g) => intra::predicted_mode(g, bx, by),
                        None => self.predicted_intra_mode(bx, by),
                    };
                    let m = self.read_pred_mode(pred)?;
                    *slot = m;
                    self.st.intra_mode[i0 + by * w4 + bx] = m as i8;
                    if let Some(g) = &mut grid {
                        intra::set_mode(g, bx, by, 1, m);
                    }
                }
            }
            let chroma = if self.cx.sps.chroma_array_type() != 0 {
                self.read_chroma_pred_mode()?
            } else {
                0
            };
            self.st.mbs[a].chroma_pred_mode = chroma;
            let cbp = self.read_cbp(true)?;
            (cbp, self.st.mbs[a].mb_type)
        } else {
            let pred = ((t - 1) % 4) as u8;
            let cbp_c = ((t - 1) / 4 % 3) as u8;
            let cbp_l = if t >= 13 { 0x0F } else { 0 };
            self.st.mbs[a].mb_type = MbType::I16x16(pred);
            let chroma = if self.cx.sps.chroma_array_type() != 0 {
                self.read_chroma_pred_mode()?
            } else {
                0
            };
            self.st.mbs[a].chroma_pred_mode = chroma;
            (cbp_l | (cbp_c << 4), MbType::I16x16(pred))
        };
        self.st.mbs[a].cbp = cbp;
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
        if CABAC {
            self.cd.align_for_pcm(&mut samples[..n])?;
        } else {
            let r = &mut self.rd;
            r.align_to_byte();
            for s in samples.iter_mut().take(n) {
                *s = r.read_bits(8)? as u8;
            }
        }
        self.st.mbs[a].mb_type = MbType::IPcm;
        self.st.mbs[a].cbp = 0x2F;
        self.st.mbs[a].cbf_dc = 7;
        self.st.mbs[a].qp = 0;
        self.st.mbs[a].qpc = self.chroma_qps(0);
        self.last_dqp_nz = false;
        let (x4, y4) = (self.mb.x * 4, self.mb.y * 4);
        for by in 0..4 {
            for bx in 0..4 {
                let i = self.st.b4(x4 + bx, y4 + by);
                self.st.nnz[i] = 16;
                self.st.blocks[i].nz = true;
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
        // Intra availability of the neighbouring macroblocks as intra::INTRA_* flags; without
        // MBAFF it is the same for all their rows.
        let nb = (!MBAFF).then(|| {
            intra::flags((
                self.intra_ok(self.mb_left()),
                self.intra_ok(self.mb_top()),
                self.intra_ok(self.mb_nb(16, -1)),
                self.intra_ok(self.mb_nb(-1, -1)),
            ))
        });
        match kind {
            MbType::I4x4 => {
                for (blk, &mode) in modes.iter().enumerate() {
                    let (bx, by) = BLK4_XY[blk];
                    let avail = match nb {
                        Some(nb) => intra::AVAIL4[blk][nb as usize],
                        None => intra::flags(self.intra_edges(bx, by, 1, blk)),
                    };
                    let o = (py + by * 4) * ls + px + bx * 4;
                    let plane = &mut self.st.planes[0][off..];
                    intra::pred4x4(plane, o, ls, mode, avail);
                    if res.luma_coded & (1 << blk) != 0 {
                        recon::idct4_add(&mut plane[o..], ls, &mut res.luma[blk]);
                    }
                }
            }
            MbType::I8x8 => {
                for b8 in 0..4 {
                    let (bx, by) = ((b8 & 1) * 2, (b8 >> 1) * 2);
                    let avail = match nb {
                        Some(nb) => intra::AVAIL8[b8][nb as usize],
                        None => intra::flags(self.intra_edges(bx, by, 2, b8 * 4)),
                    };
                    let o = (py + by * 4) * ls + px + bx * 4;
                    let plane = &mut self.st.planes[0][off..];
                    intra::pred8x8l(plane, o, ls, modes[b8], avail);
                    if res.luma_coded & (0xF << (b8 * 4)) != 0 {
                        recon::idct8_add(&mut plane[o..], ls, &mut res.luma8[b8]);
                    }
                }
            }
            MbType::I16x16(mode) => {
                let (mb_left, mb_top, mb_tl) = match nb {
                    Some(nb) => (
                        nb & intra::INTRA_LEFT != 0,
                        nb & intra::INTRA_TOP != 0,
                        nb & intra::INTRA_TOP_LEFT != 0,
                    ),
                    None => (
                        self.left_ok(0, 16),
                        self.intra_ok(self.mb_top()),
                        self.intra_ok(self.mb_nb(-1, -1)),
                    ),
                };
                if !intra::block_mode_ok(mode, mb_left, mb_top, mb_tl) {
                    return Err(bad("H.264 Intra16x16 mode uses unavailable samples"));
                }
                let plane = &mut self.st.planes[0][off..];
                let avail = intra::flags((mb_left, mb_top, false, mb_tl));
                intra::pred16x16(plane, py * ls + px, ls, mode, avail);
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
            let (left, mb_top, mb_tl) = match nb {
                Some(nb) => (
                    [nb & intra::INTRA_LEFT != 0; 2],
                    nb & intra::INTRA_TOP != 0,
                    nb & intra::INTRA_TOP_LEFT != 0,
                ),
                None => (
                    [self.left_ok(0, 8), self.left_ok(8, 8)],
                    self.intra_ok(self.mb_top()),
                    self.intra_ok(self.mb_nb(-1, -1)),
                ),
            };
            let mode = self.st.mbs[self.mb.addr].chroma_pred_mode;
            let as_block = [2, 1, 0, 3][mode as usize];
            if !intra::block_mode_ok(as_block, left[0] && left[1], mb_top, mb_tl) {
                return Err(bad("H.264 chroma intra mode uses unavailable samples"));
            }
            let mut avail = intra::flags((left[0], mb_top, false, mb_tl));
            if left[1] {
                avail |= intra::INTRA_LEFT_LOWER;
            }
            let (off, cs, cx, cy) = self.view(1);
            for p in 0..2 {
                let plane = &mut self.st.planes[1 + p][off..];
                intra::pred_chroma(plane, cy * cs + cx, cs, as_block, avail);
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
        if !MBAFF {
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

    /// Without MBAFF: the left and top neighbours of the 4x4 block at MB-relative `(bx, by)` in
    /// a per-block array with `1 << sh` blocks per macroblock side, as `(index, in another
    /// macroblock, macroblock)` when available.
    #[inline(always)]
    fn nb_flat(&self, sh: usize, bx: usize, by: usize) -> [Option<(usize, bool, usize)>; 2] {
        let w = self.st.mb_w << sh;
        let i = ((self.mb.y << sh) + by) * w + (self.mb.x << sh) + bx;
        let a = self.mb.addr;
        [
            (bx > 0 || self.nb & NB_L != 0).then(|| (i - 1, bx == 0, a.wrapping_sub(1))),
            (by > 0 || self.nb & NB_T != 0).then(|| (i - w, by == 0, a.wrapping_sub(self.st.mb_w))),
        ]
    }

    /// Neighbours A and B of the luma 4x4 block at MB-relative `(bx, by)`, each `(macroblock,
    /// block index)` when available (as [`Self::blk`], every block of the current macroblock
    /// counting as decoded).
    #[inline]
    fn nb_ab(&self, bx: usize, by: usize) -> [Option<(usize, usize)>; 2] {
        if !MBAFF {
            let a = self.mb.addr;
            return self
                .nb_flat(2, bx, by)
                .map(|n| n.map(|(i, other, m)| (if other { m } else { a }, i)));
        }
        let (x, y) = ((bx * 4) as isize, (by * 4) as isize);
        [self.blk(x - 1, y, 0xFFFF), self.blk(x, y - 1, 0xFFFF)]
    }

    /// CAVLC `nC` for the luma 4x4 block at MB-relative 4x4 `(bx, by)` (9.2.1).
    #[inline]
    fn nc_luma(&self, bx: usize, by: usize) -> u8 {
        if !MBAFF {
            let [a, b] = self
                .nb_flat(2, bx, by)
                .map(|n| n.map(|(i, _, _)| self.st.nnz[i]));
            return combine_nc(a, b);
        }
        let [a, b] = self
            .nb_blocks(0, bx, by)
            .map(|n| n.map(|(_, i)| self.st.nnz[i]));
        combine_nc(a, b)
    }

    #[inline]
    fn nc_chroma(&self, p: usize, bx: usize, by: usize) -> u8 {
        if !MBAFF {
            let [a, b] = self
                .nb_flat(1, bx, by)
                .map(|n| n.map(|(i, _, _)| self.st.nnz_c[p][i]));
            return combine_nc(a, b);
        }
        let [a, b] = self
            .nb_blocks(1, bx, by)
            .map(|n| n.map(|(_, i)| self.st.nnz_c[p][i]));
        combine_nc(a, b)
    }

    /// CABAC coded_block_flag ctxIdxInc for a luma 4x4 block (9.3.3.1.1.9).
    #[inline]
    fn cbf_luma_inc(&self, bx: usize, by: usize, intra: bool) -> u8 {
        if !MBAFF {
            let [a, b] = self.nb_flat(2, bx, by).map(|n| match n {
                Some((_, true, m)) if self.st.mbs[m].mb_type == MbType::IPcm => 1,
                Some((i, _, _)) => self.st.blocks[i].nz as u8,
                None => intra as u8,
            });
            return a + 2 * b;
        }
        let [a, b] = self.nb_blocks(0, bx, by).map(|n| match n {
            Some((m, _)) if m != self.mb.addr && self.st.mbs[m].mb_type == MbType::IPcm => 1,
            Some((_, i)) => self.st.blocks[i].nz as u8,
            None => intra as u8,
        });
        a + 2 * b
    }

    #[inline]
    fn cbf_chroma_ac_inc(&self, p: usize, bx: usize, by: usize, intra: bool) -> u8 {
        if !MBAFF {
            let [a, b] = self.nb_flat(1, bx, by).map(|n| match n {
                Some((_, true, m)) if self.st.mbs[m].mb_type == MbType::IPcm => 1,
                Some((i, _, _)) => (self.st.nnz_c[p][i] > 0) as u8,
                None => intra as u8,
            });
            return a + 2 * b;
        }
        let [a, b] = self.nb_blocks(1, bx, by).map(|n| match n {
            Some((m, _)) if m != self.mb.addr && self.st.mbs[m].mb_type == MbType::IPcm => 1,
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
                let t = self.st.mbs[a].mb_type;
                let has_dc = if bit == 0 {
                    matches!(t, MbType::I16x16(_))
                } else {
                    self.st.mbs[a].cbp >> 4 != 0
                };
                if t == MbType::IPcm {
                    1
                } else if t.is_skip() || !has_dc {
                    0
                } else {
                    (self.st.mbs[a].cbf_dc >> bit) & 1
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
        if CABAC {
            Ok(self
                .cd
                .decode_residual_block(&mut self.ctxs[..], cat, inc, out, field)
                .1)
        } else {
            decode_cavlc_block(&mut self.rd, out, nc, chroma_dc)
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
        self.read_residual_with(cbp, is_i16, t8, intra, res)
    }

    #[inline(never)]
    fn read_residual_with(
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
        let map4 = [&SCAN_MAP4[field as usize][0], &SCAN_MAP4[field as usize][1]];
        let a = self.mb.addr;
        let qp = self.qp;
        let (mx4, my4) = (self.mb.x * 4, self.mb.y * 4);
        let lst = if intra { 0 } else { 3 };
        let cabac = CABAC;
        // DC coded_block_flags are set below as DC blocks are read.
        self.st.mbs[a].cbf_dc = 0;
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
                self.st.mbs[a].cbf_dc |= 1;
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
                        self.st.blocks[j].nz = total > 0;
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
                        self.st.blocks[j].nz = total > 0;
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
                    dq.scale4_coeffs(lst, qp, &coeffs, n, map4[16 - max], out);
                    n
                } else {
                    let nc = self.nc_luma(bx, by);
                    let r = self.reader();
                    let n = decode_cavlc_coeffs(r, max, nc, &mut coeffs)?;
                    dq.scale4_coeffs(lst, qp, &coeffs, n, map4[16 - max], out);
                    n
                };
                let j = self.st.b4(x4, y4);
                self.st.nnz[j] = n as u8;
                self.st.blocks[j].nz = n > 0;
                if n > 0 {
                    res.luma_coded |= 1 << blk;
                }
            }
        }

        let cbp_c = cbp >> 4;
        if self.cx.sps.chroma_array_type() == 0 || cbp_c == 0 {
            return Ok(());
        }
        let qpc = self.st.mbs[a].qpc;
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
                self.st.mbs[a].cbf_dc |= 2 << p;
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
                    dq.scale4_coeffs(lst + 1 + p, qpc[p] as i32, &coeffs, n, map4[1], out);
                    n
                } else {
                    let nc = self.nc_chroma(p, bx, by);
                    let r = self.reader();
                    let n = decode_cavlc_coeffs(r, 15, nc, &mut coeffs)?;
                    dq.scale4_coeffs(lst + 1 + p, qpc[p] as i32, &coeffs, n, map4[1], out);
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

    /// Fails when a used reference index lies beyond its list.
    #[inline]
    fn check_ref(&self, list: usize, idx: i8) -> Result<()> {
        if idx >= 0 && idx as usize >= self.refs(list).len() {
            return Err(bad("H.264 ref_idx beyond the reference list"));
        }
        Ok(())
    }

    /// Resets the current macroblock's blocks to no motion, no coefficients.
    fn reset_blocks(&mut self) {
        self.st.reset_mb_blocks(self.mb.x, self.mb.y);
    }

    /// Writes the whole [`BlockInfo`] of the MB-relative 4x4 rectangle `(bx, by, w, h)`: motion
    /// `m` for `list`, none for the other, no coefficients.
    #[allow(clippy::too_many_arguments)]
    fn fill_blocks(
        &mut self,
        bx: usize,
        by: usize,
        w: usize,
        h: usize,
        list: usize,
        m: Mv,
        mvd: [u8; 2],
    ) {
        // Built per list with constant indices, so that it stays in a register.
        let b = if list == 0 {
            BlockInfo {
                mv: [m.1, [0, 0]],
                ref_idx: [m.0, -1],
                mvd: [mvd, [0, 0]],
                ..NO_BLOCK
            }
        } else {
            BlockInfo {
                mv: [[0, 0], m.1],
                ref_idx: [-1, m.0],
                mvd: [[0, 0], mvd],
                ..NO_BLOCK
            }
        };
        self.partial_motion |= (bx, by, w, h) != (0, 0, 4, 4);
        let i = self.st.b4(self.mb.x * 4 + bx, self.mb.y * 4 + by);
        let s = self.st.mb_w * 4;
        match w {
            1 => fill_block_rect::<1>(&mut self.st.blocks, i, s, h, b),
            2 => fill_block_rect::<2>(&mut self.st.blocks, i, s, h, b),
            3 => fill_block_rect::<3>(&mut self.st.blocks, i, s, h, b),
            _ => fill_block_rect::<4>(&mut self.st.blocks, i, s, h, b),
        }
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
            mvd: [u8; 2],
        ) {
            let s = st.mb_w * 4;
            let rows = &mut st.blocks[i..i + (h - 1) * s + N];
            for r in 0..h {
                for b in &mut rows[r * s..r * s + N] {
                    b.ref_idx[list] = m.0;
                    b.mv[list] = m.1;
                    b.mvd[list] = mvd;
                }
            }
        }
        self.partial_motion |= (bx, by, w, h) != (0, 0, 4, 4);
        let i = self.st.b4(self.mb.x * 4 + bx, self.mb.y * 4 + by);
        match w {
            1 => rect::<1>(self.st, list, i, h, m, mvd),
            2 => rect::<2>(self.st, list, i, h, m, mvd),
            3 => rect::<3>(self.st, list, i, h, m, mvd),
            _ => rect::<4>(self.st, list, i, h, m, mvd),
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
            if r < 0 || !MBAFF {
                return Some((r, mv));
            }
            Some(match (self.field, self.st.mbs[a].mb_field) {
                (true, false) => (r * 2, [mv[0], mv[1] / 2]),
                (false, true) => (r >> 1, [mv[0], mv[1].wrapping_mul(2)]),
                _ => (r, mv),
            })
        })
    }

    /// Motion of the available block covering luma `(xn, yn)`, [`NO_MV`] if there is none.
    fn neighbour(&self, list: usize, xn: isize, yn: isize, done: u16) -> Mv {
        let Some((a, i)) = self.blk(xn, yn, done) else {
            return NO_MV;
        };
        let (r, mv) = (self.st.blocks[i].ref_idx[list], self.st.blocks[i].mv[list]);
        if r < 0 || !MBAFF {
            return (r, mv);
        }
        match (self.field, self.st.mbs[a].mb_field) {
            (true, false) => (r * 2, [mv[0], mv[1] / 2]),
            (false, true) => (r >> 1, [mv[0], mv[1].wrapping_mul(2)]),
            _ => (r, mv),
        }
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
        if !MBAFF {
            // Neighbours by block coordinates: inside the macroblock when decoded (`done`),
            // else in the left, top, top-right or top-left macroblock when available.
            let st = &*self.st;
            let s = st.mb_w * 4;
            let i = st.b4(self.mb.x * 4 + bx, self.mb.y * 4 + by);
            let at = |j: usize| (st.blocks[j].ref_idx[list], st.blocks[j].mv[list]);
            let inside = |x: usize, y: usize| done & (1 << BLK4_AT[y][x]) != 0;
            let nb = self.nb;
            let get = |ok: bool, j: usize| if ok { at(j) } else { NO_MV };
            let a = get(
                if bx > 0 {
                    inside(bx - 1, by)
                } else {
                    nb & NB_L != 0
                },
                i.wrapping_sub(1),
            );
            let b = get(
                if by > 0 {
                    inside(bx, by - 1)
                } else {
                    nb & NB_T != 0
                },
                i.wrapping_sub(s),
            );
            let cx = bx + w;
            let c_ok = match (by > 0, cx < 4) {
                (true, true) => inside(cx, by - 1),
                // The macroblock to the right follows in decoding order.
                (true, false) => false,
                (false, true) => nb & NB_T != 0,
                (false, false) => nb & NB_TR != 0,
            };
            let c = if c_ok {
                at((i + w).wrapping_sub(s))
            } else {
                let d_ok = match (bx > 0, by > 0) {
                    (true, true) => inside(bx - 1, by - 1),
                    (false, true) => nb & NB_L != 0,
                    (true, false) => nb & NB_T != 0,
                    (false, false) => nb & NB_TL != 0,
                };
                get(d_ok, i.wrapping_sub(s + 1))
            };
            return mvp_from(a, b, c, refidx, part);
        }
        let (x, y) = ((bx * 4) as isize, (by * 4) as isize);
        let a = self.neighbour(list, x - 1, y, done);
        let b = self.neighbour(list, x, y - 1, done);
        let mut c = self.neighbour(list, x + (w * 4) as isize, y - 1, done);
        if c.0 == UNAV {
            c = self.neighbour(list, x - 1, y - 1, done);
        }
        mvp_from(a, b, c, refidx, part)
    }

    fn p_skip_mv(&self) -> [i16; 2] {
        let [a, b, c] = if MBAFF {
            self.mb_neighbours()[0]
        } else {
            self.mb_neighbours_flat(0)
        };
        if a.0 == UNAV || b.0 == UNAV || a == (0, [0, 0]) || b == (0, [0, 0]) {
            return [0, 0];
        }
        mvp_raw(a, b, c, 0)
    }

    /// [`Self::mb_neighbours`] of one list, without MBAFF.
    #[inline]
    fn mb_neighbours_flat(&self, list: usize) -> [Mv; 3] {
        let (x, y, w) = (self.mb.x, self.mb.y, self.st.mb_w);
        let i = self.st.b4(x * 4, y * 4);
        let s = w * 4;
        let st = &*self.st;
        let at = |i: usize| (st.blocks[i].ref_idx[list], st.blocks[i].mv[list]);
        let nb = self.nb;
        let a = if nb & NB_L != 0 { at(i - 1) } else { NO_MV };
        let b = if nb & NB_T != 0 { at(i - s) } else { NO_MV };
        let c = if nb & NB_TR != 0 {
            at(i - s + 4)
        } else if nb & NB_TL != 0 {
            at(i - s - 1)
        } else {
            NO_MV
        };
        [a, b, c]
    }

    /// Neighbours A, B and C (D where C is unavailable) of the whole macroblock in both lists,
    /// as for a 16x16 partition's predictor (8.4.1.3.2).
    #[inline(always)]
    fn mb_neighbours(&self) -> [[Mv; 3]; 2] {
        if MBAFF {
            let a = self.neighbour_pair(-1, 0, 0);
            let b = self.neighbour_pair(0, -1, 0);
            let c = self.neighbour_pair(16, -1, 0);
            // C falls back to D when unavailable; availability is the same for both lists.
            let c = if c[0].is_none() {
                self.neighbour_pair(-1, -1, 0)
            } else {
                c
            };
            let r = |n: Option<Mv>| n.unwrap_or(NO_MV);
            return [[r(a[0]), r(b[0]), r(c[0])], [r(a[1]), r(b[1]), r(c[1])]];
        }
        // Without MBAFF the neighbours are the adjacent macroblocks' corner blocks.
        let (x, y, w) = (self.mb.x, self.mb.y, self.st.mb_w);
        let i = self.st.b4(x * 4, y * 4);
        let s = w * 4;
        let st = &*self.st;
        let at = |i: usize| {
            let b = st.blocks[i];
            [0, 1].map(|l| (b.ref_idx[l], b.mv[l]))
        };
        let none = [NO_MV, NO_MV];
        let nb = self.nb;
        let a = if nb & NB_L != 0 { at(i - 1) } else { none };
        let b = if nb & NB_T != 0 { at(i - s) } else { none };
        let c = if nb & NB_TR != 0 {
            at(i - s + 4)
        } else if nb & NB_TL != 0 {
            at(i - s - 1)
        } else {
            none
        };
        [[a[0], b[0], c[0]], [a[1], b[1], c[1]]]
    }

    fn read_ref_idx(&mut self, list: usize, bx: usize, by: usize) -> Result<i8> {
        let n = self.num_refs(list);
        if !self.is_cabac() {
            return self.read_te(n - 1);
        }
        let is_b = self.cx.sh.slice_type.is_b();
        let cond = |n: Option<(usize, usize)>| -> u8 {
            let Some((addr, i)) = n else {
                return 0;
            };
            let t = self.st.mbs[addr].mb_type;
            if t.is_skip() || t.is_intra() {
                return 0;
            }
            if is_b && self.st.blocks[i].direct {
                return 0;
            }
            // A field neighbour of a frame MB counts reference indices in fields (9.3.3.1.1.6).
            let thresh = if MBAFF && !self.field && self.st.mbs[addr].mb_field {
                1
            } else {
                0
            };
            (self.st.blocks[i].ref_idx[list] > thresh) as u8
        };
        let [na, nb] = self.nb_ab(bx, by);
        let (ca, cb) = (cond(na), cond(nb));
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
        let [na, nb] = self.nb_ab(bx, by);
        let absn = |n: Option<(usize, usize)>, comp: usize| -> u32 {
            let Some((addr, i)) = n else {
                return 0;
            };
            let v = self.st.blocks[i].mvd[list][comp] as u32;
            // Vertical components between frame and field MBs are rescaled (9.3.3.1.1.7).
            match (comp, MBAFF, self.field, self.st.mbs[addr].mb_field) {
                (1, true, false, true) => v * 2,
                (1, true, true, false) => v / 2,
                _ => v,
            }
        };
        let sums = [0, 1].map(|comp| absn(na, comp) + absn(nb, comp));
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
            self.st.mbs[a].mb_type = MbType::Inter(part);
            self.decode_partitions(part, [PRED_L0, PRED_L0])?;
            return self.finish_inter(true);
        }
        self.st.mbs[a].mb_type = MbType::Inter(Part::P8x8);
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
        self.reset_blocks();
        self.decode_sub_mbs(&info, raw == 4)?;
        let no_small = subs.iter().all(|&s| s == 0);
        self.finish_inter(no_small)
    }

    #[inline(never)]
    fn decode_b_mb(&mut self, raw: u32) -> Result<()> {
        let a = self.mb.addr;
        if raw == 0 {
            self.st.mbs[a].mb_type = MbType::BDirect16x16;
            self.direct_motion(0b1111)?;
            return self.finish_inter(self.cx.sps.direct_8x8_inference);
        }
        if raw > 22 {
            return Err(bad("H.264 B mb_type out of range"));
        }
        if raw < 22 {
            let (part, pred) = B_MB_TYPES[raw as usize];
            self.st.mbs[a].mb_type = MbType::Inter(part);
            self.decode_partitions(part, pred)?;
            return self.finish_inter(true);
        }
        self.st.mbs[a].mb_type = MbType::Inter(Part::P8x8);
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
        self.reset_blocks();
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
        // One instance per shape: its partitions and their sizes become constants, and the
        // loops over them branch no more.
        match part {
            Part::P16x16 => self.decode_partitions_as::<0>(pred),
            Part::P16x8 => self.decode_partitions_as::<1>(pred),
            Part::P8x16 => self.decode_partitions_as::<2>(pred),
            Part::P8x8 => unreachable!(),
        }
    }

    /// [`Self::decode_partitions`] for shape `SHAPE`: 16x16, 16x8 or 8x16.
    #[inline(always)]
    fn decode_partitions_as<const SHAPE: u8>(&mut self, pred: [u8; 2]) -> Result<()> {
        let (part, rects): (Part, &[(usize, usize, usize, usize)]) = match SHAPE {
            0 => (Part::P16x16, &[(0, 0, 4, 4)]),
            1 => (Part::P16x8, &[(0, 0, 4, 2), (0, 2, 4, 2)]),
            _ => (Part::P8x16, &[(0, 0, 2, 4), (2, 0, 2, 4)]),
        };
        // In P slices each partition's blocks are written in full when its vector is known;
        // otherwise the lists are written separately over reset blocks, whose list 1 and direct
        // flags the CABAC contexts of later partitions also read.
        let l0_only = !self.cx.sh.slice_type.is_b();
        if !l0_only {
            self.reset_blocks();
        }
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
                    self.check_ref(list, r)?;
                    if l0_only {
                        self.fill_blocks(bx, by, w, h, list, (r, mv), amvd);
                    } else {
                        self.fill_motion(bx, by, w, h, list, (r, mv), amvd);
                    }
                }
                // An unused list keeps the macroblock's reset motion (no reference).
                done |= rect_mask(bx, by, w, h);
            }
        }
        Ok(())
    }

    /// P_8x8 / B_8x8 sub-macroblocks. `info[i] = (count, w, h, pred)`; `count == 0` is B_Direct_8x8.
    /// The blocks must have been reset (B_Direct_8x8 sub-macroblocks are derived first).
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
                        self.check_ref(list, r)?;
                        self.fill_motion(x, y, w, h, list, (r, mv), amvd);
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
                sp_ref[list] = min_positive(a.0.max(-1), min_positive(b.0.max(-1), c.0.max(-1)));
            }
            if sp_ref[0] < 0 && sp_ref[1] < 0 {
                sp_ref = [0, 0];
                direct_zero = true;
            } else {
                for list in 0..2 {
                    if sp_ref[list] >= 0 {
                        let [a, b, c] = nb[list];
                        sp_mv[list] = mvp_raw(a, b, c, sp_ref[list]);
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
                c.uniform().then(|| c.ref_mv(0, 0))
            } else {
                Some((-1, [0, 0]))
            };
            if let Some((col_ref, col_mv)) = col {
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
                let (col_ref, col_mv, col_pic, vscale) = if !need_col {
                    (-1, [0, 0], NO_PIC, 0)
                } else if spatial {
                    let (r, mv) = col_mb
                        .get_or_insert_with(|| self.colocated_mb())
                        .ref_mv(cbx, cby);
                    (r, mv, NO_PIC, 0)
                } else {
                    col_mb
                        .get_or_insert_with(|| self.colocated_mb())
                        .block(cbx, cby)
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
        let mut b = NO_BLOCK;
        b.direct = true;
        for list in 0..2 {
            if refs[list] >= 0 {
                b.ref_idx[list] = refs[list];
                b.mv[list] = mvs[list];
            }
            self.check_ref(list, b.ref_idx[list])?;
        }
        self.partial_motion |= n != 4;
        let i = self.st.b4(self.mb.x * 4 + bx, self.mb.y * 4 + by);
        let s = self.st.mb_w * 4;
        match n {
            4 => {
                fill_block_rect::<4>(&mut self.st.blocks, i, s, 4, b);
            }
            2 => {
                fill_block_rect::<2>(&mut self.st.blocks, i, s, 2, b);
            }
            _ => {
                fill_block_rect::<1>(&mut self.st.blocks, i, s, 1, b);
            }
        }
        Ok(())
    }

    /// The co-located macroblock of direct prediction (8.4.1.2.1), for [`ColMb::block`].
    fn colocated_mb(&self) -> ColMb<'c> {
        let col = &self.refs(1)[0];
        let m = match (&col.motion, &col.sync) {
            // A B slice in a Baseline stream (which has none) predicts from a co-located picture
            // without motion: frame threads keep none for Baseline (see `Core`).
            _ if self.cx.sps.profile_idc == BASELINE => None,
            (Some(m), _) => Some(MotionView::of(m)),
            // A frame still decoding on another thread (frames only): its co-located row.
            (None, Some(sync)) => sync.live().map(|live| {
                sync.wait_motion(self.mb.y, 2);
                live.view()
            }),
            (None, None) => None,
        };
        let Some(m) = m else {
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
        self.st.mbs[a].cbp = cbp;
        let mut t8 = false;
        if cbp & 0x0F != 0 && self.cx.pps.transform_8x8_mode && no_small {
            t8 = self.read_transform_8x8()?;
        }
        self.st.mbs[a].t8x8 = t8;
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
    /// FFmpeg's `prefetch_motion`: starts loading the reference samples a macroblock four to
    /// the right will likely read, from the motion of this one's first block in `list` (four
    /// luma rows, staggered by column so that four macroblocks cover sixteen, and a chroma row).
    #[inline]
    fn prefetch_motion(&self, list: usize) {
        let b = &self.st.blocks[self.st.b4(self.mb.x * 4, self.mb.y * 4)];
        let r = b.ref_idx[list];
        if r < 0 || !self.simple {
            return;
        }
        if let Some(rp) = self.cx.refs[list].get(r as usize) {
            self.prefetch_ref(rp, b.mv[list]);
        }
    }

    /// [`Self::prefetch_motion`] from reference `r` and vector `mv`.
    #[inline(always)]
    fn prefetch_ref(&self, r: &RefPic, mv: [i16; 2]) {
        let w = self.st.width as isize;
        let (x, y) = ((self.mb.x * 16) as isize, (self.mb.y * 16) as isize);
        let mx = (mv[0] >> 2) as isize + x + 8;
        let my = (mv[1] >> 2) as isize + y;
        // Negative offsets wrap to huge ones, which the prefetch skips.
        let luma = (my + (self.mb.x & 3) as isize * 4) * w + mx + 64;
        ragdoll_simd::prefetch_rows::<4>(&r.planes[0], luma as usize, w as usize);
        let chroma = ((my >> 1) + (self.mb.x & 7) as isize) * (w / 2) + (mx >> 1) + 64;
        ragdoll_simd::prefetch_rows::<1>(&r.planes[1], chroma as usize, 0);
        ragdoll_simd::prefetch_rows::<1>(&r.planes[2], chroma as usize, 0);
    }

    fn inter_predict_mb(&mut self) -> Result<()> {
        if self.mc_direct && !self.partial_motion {
            return self.mc_whole();
        }
        self.prefetch_motion(0);
        let r = self.inter_predict_parts();
        self.prefetch_motion(1);
        r
    }

    /// [`Self::inter_predict_mb`] for a macroblock with one motion throughout when
    /// [`Self::mc_direct`]: straight to [`Self::mc_pred`], or [`Self::mc_pred_weighted`].
    fn mc_whole(&mut self) -> Result<()> {
        let a = self.mb.addr;
        self.st.mbs[a].motion_edges = 0;
        let b = self.st.blocks[self.st.b4(self.mb.x * 4, self.mb.y * 4)];
        let lists = &self.cx.refs;
        let pic = |l: usize| -> Result<Option<&'c RefPic>> {
            match b.ref_idx[l] {
                r if r < 0 => Ok(None),
                r => lists[l]
                    .get(r as usize)
                    .map(Some)
                    .ok_or_else(|| bad("H.264 ref_idx beyond list")),
            }
        };
        let (x, y) = (self.mb.x * 16, self.mb.y * 16);
        let rp = [pic(0)?, pic(1)?];
        for (r, mv) in rp.into_iter().zip(b.mv) {
            if let Some(r) = r {
                self.prefetch_ref(r, mv);
            }
        }
        if self.weighted
            && let Some(wt) = self.block_weights(rp.map(|r| r.is_some()), b.ref_idx)
        {
            self.mc_pred_weighted(rp, b.mv, x, y, 16, 16, wt);
            return Ok(());
        }
        match rp {
            [Some(r), None] => self.mc_pred(r, b.mv[0], x, y, 16, 16, false),
            [None, Some(r)] => self.mc_pred(r, b.mv[1], x, y, 16, 16, false),
            [Some(r0), Some(r1)] => {
                self.mc_pred(r0, b.mv[0], x, y, 16, 16, false);
                self.mc_pred(r1, b.mv[1], x, y, 16, 16, true);
            }
            [None, None] => return Err(bad("H.264 inter block without prediction")),
        }
        Ok(())
    }

    fn inter_predict_parts(&mut self) -> Result<()> {
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
            self.st.mbs[self.mb.addr].motion_edges = 0;
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
        self.st.mbs[self.mb.addr].motion_edges = m;
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
        let [pcb, pcr] = self.st.planes.chroma_mut();
        let (dcb, dcr) = (&mut pcb[d..], &mut pcr[d..]);
        fn src(r: &RefPic) -> [&[u8]; 2] {
            [&r.planes[1], &r.planes[2]]
        }
        match rp {
            [Some(r), None] | [None, Some(r)] => {
                let mv = cmv[rp[0].is_none() as usize];
                let dst = [dcb, dcr];
                recon::mc_chroma_pair::<false>(src(r), cpw, cph, cx, cy, cw, ch, mv, dst, cs);
            }
            [Some(r0), Some(r1)] => {
                let (m0, m1) = (cmv[0], cmv[1]);
                let dst = [&mut *dcb, &mut *dcr];
                recon::mc_chroma_pair::<false>(src(r0), cpw, cph, cx, cy, cw, ch, m0, dst, cs);
                let dst = [dcb, dcr];
                recon::mc_chroma_pair::<true>(src(r1), cpw, cph, cx, cy, cw, ch, m1, dst, cs);
            }
            [None, None] => {}
        }
        Ok(())
    }

    /// [`Self::mc_rect_simple`] in a slice with explicit weights, kept apart so that the
    /// unweighted path stays as it is.
    #[inline(never)]
    fn mc_rect_simple_explicit(
        &mut self,
        bx: usize,
        by: usize,
        w4: usize,
        h4: usize,
        refs: [i8; 2],
        mvs: [[i16; 2]; 2],
    ) -> Result<()> {
        self.mc_rect_simple::<true>(bx, by, w4, h4, refs, mvs)
    }

    /// With explicit weights: the luma weighting of a block predicted from the lists in `used`
    /// with `refs`, or `None` when every component reduces to default prediction (as for most
    /// references of a P slice).
    #[inline(never)]
    fn explicit_weighting(&self, used: [bool; 2], refs: [i8; 2]) -> Option<Weighting> {
        if used == [false, false] {
            return None;
        }
        // References with default weights in every component predict as without weights.
        let unit = |l: usize| !used[l] || self.unit_weights[l] >> (refs[l] as u32 & 31) & 1 != 0;
        if unit(0) && unit(1) {
            return None;
        }
        let eff = |c: usize| recon::effective_weighting(used, self.weighting(refs, c));
        let wl = eff(0);
        let default = |wt: Weighting| matches!(wt, Weighting::Default);
        (!(default(wl) && default(eff(1)) && default(eff(2)))).then_some(wl)
    }

    /// [`Self::mc_rect`] when [`Self::simple`]: frame coordinates throughout; blocks whose
    /// weights reduce to default prediction take the default path.
    #[inline(always)]
    fn mc_rect_simple<const EXPLICIT: bool>(
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
        let lists = &self.cx.refs;
        let pic = |l: usize| -> Result<Option<&'c RefPic>> {
            if refs[l] < 0 {
                return Ok(None);
            }
            match lists[l].get(refs[l] as usize) {
                Some(r) => Ok(Some(r)),
                None => Err(bad("H.264 ref_idx beyond list")),
            }
        };
        let rp = [pic(0)?, pic(1)?];
        if EXPLICIT
            && self
                .explicit_weighting(rp.map(|r| r.is_some()), refs)
                .is_some()
        {
            let wt = [0, 1, 2].map(|c| self.weighting(refs, c));
            self.mc_pred_weighted(rp, mvs, x, y, w, h, wt);
            return Ok(());
        }
        match rp {
            [None, None] => return Err(bad("H.264 inter block without prediction")),
            [Some(_), Some(_)] if !self.implicit.is_empty() => {
                let n1 = lists[1].len();
                let [w0, w1] = self.implicit[refs[0] as usize * n1 + refs[1] as usize];
                // Implicit weights of 32 each are default bi-prediction (8.4.2.3).
                if (w0, w1) != (32, 32) {
                    let wt = Weighting::Weighted(5, [(w0, 0), (w1, 0)]);
                    self.mc_pred_weighted(rp, mvs, x, y, w, h, [wt; 3]);
                    return Ok(());
                }
            }
            _ => {}
        }
        match rp {
            [Some(r), None] => self.mc_pred(r, mvs[0], x, y, w, h, false),
            [None, Some(r)] => self.mc_pred(r, mvs[1], x, y, w, h, false),
            [Some(r0), Some(r1)] => {
                self.mc_pred(r0, mvs[0], x, y, w, h, false);
                self.mc_pred(r1, mvs[1], x, y, w, h, true);
            }
            [None, None] => {}
        }
        Ok(())
    }

    /// Predicts the `w`x`h` luma block at `(x, y)` and its chroma blocks from `r` displaced by
    /// `mv`, written to the picture (with `avg`, averaged into it), for a progressive 4:2:0
    /// frame whose references' planes are as large as its own ([`Self::mc_fast`]).
    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    fn mc_pred(&self, r: &RefPic, mv: [i16; 2], x: usize, y: usize, w: usize, h: usize, avg: bool) {
        let pw = self.st.width;
        let [(dy, _), (dcb, _), (dcr, _)] = self.st.planes.raw_all();
        let (cs, cx, cy) = (pw / 2, x / 2, y / 2);
        // Safety: the destination blocks lie inside the picture's planes, which no reference
        // shares.
        unsafe {
            self.luma_into(r, mv, x, y, w, h, avg, dy.add(y * pw + x), pw);
            let d = cy * cs + cx;
            self.chroma_into(
                r,
                mv,
                cx,
                cy,
                w / 2,
                h / 2,
                avg,
                [dcb.add(d), dcr.add(d)],
                cs,
            );
        }
    }

    /// [`Self::mc_pred`] with weighted prediction `wt` (luma, Cb, Cr) from the references in
    /// `rp`: each list is predicted into scratch blocks, which the weights then combine.
    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    fn mc_pred_weighted(
        &self,
        rp: [Option<&RefPic>; 2],
        mvs: [[i16; 2]; 2],
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        wt: [Weighting; 3],
    ) {
        let pw = self.st.width;
        let [(dy, ny), (dcb, nc), (dcr, _)] = self.st.planes.raw_all();
        let (cs, cx, cy, cw, ch) = (pw / 2, x / 2, y / 2, w / 2, h / 2);
        let used = rp.map(|r| r.is_some());
        // Per list: luma at 0 (stride 16), or Cb at 0 and Cr at 128 (stride 16, 8 rows).
        let mut t = [[0u8; 256]; 2];
        for (l, r) in rp.iter().enumerate() {
            if let Some(r) = r {
                // Safety: the scratch block holds `h` rows of 16.
                unsafe { self.luma_into(r, mvs[l], x, y, w, h, false, t[l].as_mut_ptr(), 16) };
            }
        }
        let d = y * pw + x;
        // Safety: the destination block lies inside the picture's plane, which no reference
        // shares.
        let dst = unsafe { std::slice::from_raw_parts_mut(dy.add(d), ny - d) };
        combine(dst, pw, [&t[0][..], &t[1][..]], w, h, used, wt[0]);
        for (l, r) in rp.iter().enumerate() {
            if let Some(r) = r {
                let p = t[l].as_mut_ptr();
                // Safety: the scratch blocks hold `ch` rows of 16 each.
                unsafe { self.chroma_into(r, mvs[l], cx, cy, cw, ch, false, [p, p.add(128)], 16) };
            }
        }
        let d = cy * cs + cx;
        for (c, base) in [(1, dcb), (2, dcr)] {
            // Safety: as for luma.
            let dst = unsafe { std::slice::from_raw_parts_mut(base.add(d), nc - d) };
            let o = (c - 1) * 128;
            combine(
                dst,
                cs,
                [&t[0][o..o + 128], &t[1][o..o + 128]],
                cw,
                ch,
                used,
                wt[c],
            );
        }

        /// `dst = ` the lists' predictions `t` combined per `wt` (8.4.2.3).
        fn combine(
            dst: &mut [u8],
            ds: usize,
            t: [&[u8]; 2],
            w: usize,
            h: usize,
            used: [bool; 2],
            wt: Weighting,
        ) {
            use ragdoll_simd::{h264_avg, h264_weight_bi, h264_weight_uni};
            match (used, recon::effective_weighting(used, wt)) {
                ([true, true], Weighting::Weighted(lw, [(w0, o0), (w1, o1)])) => h264_weight_bi(
                    dst,
                    ds,
                    t[0],
                    t[1],
                    16,
                    w,
                    h,
                    lw,
                    w0,
                    w1,
                    (o0 + o1 + 1) >> 1,
                ),
                ([true, false], Weighting::Weighted(lw, [(w0, o0), _])) => {
                    h264_weight_uni(dst, ds, t[0], 16, w, h, lw, w0, o0)
                }
                ([false, true], Weighting::Weighted(lw, [_, (w1, o1)])) => {
                    h264_weight_uni(dst, ds, t[1], 16, w, h, lw, w1, o1)
                }
                (u, _) => {
                    let first = if u[0] { 0 } else { 1 };
                    for r in 0..h {
                        dst[r * ds..r * ds + w].copy_from_slice(&t[first][r * 16..r * 16 + w]);
                    }
                    if u == [true, true] {
                        h264_avg(dst, ds, t[1], 16, w, h);
                    }
                }
            }
        }
    }

    /// When [`Self::weighted`]: the weights (luma, Cb, Cr) of a block predicted from the lists
    /// in `used` with `refs`, or `None` when they reduce to default prediction.
    #[inline]
    fn block_weights(&self, used: [bool; 2], refs: [i8; 2]) -> Option<[Weighting; 3]> {
        if self.cx.sh.weights.is_some() {
            self.explicit_weighting(used, refs)?;
            return Some([0, 1, 2].map(|c| self.weighting(refs, c)));
        }
        if used != [true, true] || self.implicit.is_empty() {
            return None;
        }
        let n1 = self.cx.refs[1].len();
        let [w0, w1] = self.implicit[refs[0] as usize * n1 + refs[1] as usize];
        // Implicit weights of 32 each are default bi-prediction (8.4.2.3).
        ((w0, w1) != (32, 32)).then_some([Weighting::Weighted(5, [(w0, 0), (w1, 0)]); 3])
    }

    /// Luma inter prediction of the `w`x`h` block at `(x, y)` from `r` displaced by `mv`, into
    /// `dst` (stride `ds`; with `avg`, averaged into it).
    ///
    /// # Safety
    /// Rows `0..h` of `dst`, `w` samples each, are writable and are no reference's samples.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn luma_into(
        &self,
        r: &RefPic,
        mv: [i16; 2],
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        avg: bool,
        dst: *mut u8,
        ds: usize,
    ) {
        let (pw, ph) = (self.st.width, self.st.height);
        let (mvx, mvy) = (mv[0] as isize, mv[1] as isize);
        let (ix, iy) = (x as isize + (mvx >> 2), y as isize + (mvy >> 2));
        if ix >= 2
            && iy >= 2
            && ix as usize + ragdoll_simd::h264_mc::luma_mc_cols(w) <= pw
            && iy as usize + h + 3 <= ph
        {
            // Safety: the reads lie inside the reference's plane, as large as the picture's
            // (`mc_fast`); the caller vouches for `dst`.
            unsafe {
                let src = r.planes[0].as_ptr().add(iy as usize * pw + ix as usize);
                let (mx, my) = ((mvx & 3) as u8, (mvy & 3) as u8);
                self.mck.luma(avg, dst, ds, src, pw, w, h, mx, my);
            }
        } else {
            // Safety: as the caller vouches.
            let dst = unsafe { std::slice::from_raw_parts_mut(dst, (h - 1) * ds + w) };
            mc_luma_edge(r, mv, (pw, ph), x, y, w, h, avg, dst, ds);
        }
    }

    /// [`Self::luma_into`] for the `w`x`h` chroma blocks at `(x, y)` of both chroma planes.
    ///
    /// # Safety
    /// As [`Self::luma_into`], for each of `dst`.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn chroma_into(
        &self,
        r: &RefPic,
        mv: [i16; 2],
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        avg: bool,
        dst: [*mut u8; 2],
        ds: usize,
    ) {
        let (cs, cph) = (self.st.width / 2, self.st.height / 2);
        let (mvx, mvy) = (mv[0] as isize, mv[1] as isize);
        let (ix, iy) = (x as isize + (mvx >> 3), y as isize + (mvy >> 3));
        if ix >= 0
            && iy >= 0
            && ix as usize + ragdoll_simd::h264_mc::chroma_mc_cols(w) <= cs
            && iy as usize + h < cph
        {
            // Safety: as for luma. The kernel reads 16 bytes from each row's start, up to 7
            // past the block's last column: inside the plane, or in the zeroed padding every
            // `AlignedBuffer` has past its end.
            unsafe {
                let o = iy as usize * cs + ix as usize;
                let src = [r.planes[1].as_ptr().add(o), r.planes[2].as_ptr().add(o)];
                let (fx, fy) = ((mvx & 7) as u8, (mvy & 7) as u8);
                self.mck.chroma_pair(avg, dst, ds, src, cs, w, h, fx, fy);
            }
        } else {
            // Safety: as the caller vouches.
            let dst = dst.map(|d| unsafe { std::slice::from_raw_parts_mut(d, (h - 1) * ds + w) });
            mc_chroma_edge(r, mv, (cs, cph), x, y, w, h, avg, dst, ds);
        }
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
        if self.simple {
            if self.waits {
                self.wait_refs(by, h4, refs, mvs);
            }
            if self.cx.sh.weights.is_some() && self.mc_fast {
                return self.mc_rect_simple_explicit(bx, by, w4, h4, refs, mvs);
            }
            if self.mc_fast {
                return self.mc_rect_simple::<false>(bx, by, w4, h4, refs, mvs);
            }
        }
        self.mc_rect_general(bx, by, w4, h4, refs, mvs)
    }

    /// [`Self::mc_rect`] for fields, MBAFF and 4:0:0, kept out of line so that the progressive
    /// path stays small.
    #[cold]
    #[inline(never)]
    fn mc_rect_general(
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
        self.mc_rect_weighted(rp, refs, wt, o, ls, x, y, w, h, pw, ph, mvs);
        Ok(())
    }

    /// Waits until the reference samples a `4 * h4` lines high block at block row `by` of the
    /// current macroblock reads are final: the luma rows its vectors reach, three more for the
    /// interpolation filter (chroma reaches no further, see [`PicSync`]).
    #[inline(never)]
    fn wait_refs(&self, by: usize, h4: usize, refs: [i8; 2], mvs: [[i16; 2]; 2]) {
        let bottom = (self.mb.y * 16 + (by + h4) * 4) as isize - 1;
        for list in 0..2 {
            let Some(sync) = usize::try_from(refs[list])
                .ok()
                .and_then(|r| self.cx.refs[list].get(r))
                .and_then(|r| r.sync.as_deref())
            else {
                continue;
            };
            let line = bottom + (mvs[list][1] >> 2) as isize + 3;
            let last = self.st.height as isize - 1;
            if !sync.has(line.clamp(0, last) as usize) {
                // Waking costs more than decoding a few macroblocks: wait for some slack too,
                // rather than again for every next line.
                sync.wait((line + WAIT_SLACK).clamp(0, last) as usize);
            }
        }
    }

    /// [`Self::mc_rect`] with explicit or implicit weights, kept out of line: the default path
    /// stays small.
    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    fn mc_rect_weighted(
        &mut self,
        rp: [Option<&RefPic>; 2],
        refs: [i8; 2],
        wt: Weighting,
        o: usize,
        ls: usize,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        pw: usize,
        ph: usize,
        mvs: [[i16; 2]; 2],
    ) {
        let used = rp.map(|r| r.is_some());
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
            return;
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
    m: Option<MotionView<'r>>,
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

    /// `(refIdxCol, mvCol)` of MB-relative 4x4 block `(cbx, cby)`: [`Self::block`] without the
    /// picture and scale, which spatial direct prediction does not use.
    #[inline]
    fn ref_mv(&self, cbx: usize, cby: usize) -> (i8, [i16; 2]) {
        let Some(m) = self.m else {
            return (-1, [0, 0]);
        };
        let (row, by) = match self.layout {
            ColLayout::Same(row) => (row, cby),
            ColLayout::FrameFromField(row, half) => (row, half + cby / 2),
            ColLayout::FieldFromFrame(r2) => (r2 + cby / 2, 2 * (cby % 2)),
        };
        let mb = row * self.mb_w + self.mbx;
        if mb >= m.mb_intra.len() || m.mb_intra[mb] {
            return (-1, [0, 0]);
        }
        let b = &m.blocks[(row * 4 + by) * self.mb_w * 4 + self.mbx * 4 + cbx];
        let l = if b.ref_idx[0] >= 0 { 0 } else { 1 };
        (b.ref_idx[l], b.mv[l])
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
        (b.ref_idx[l], b.mv[l], m.ref_pic(mb, i, l), vscale)
    }
}

/// Lines beyond those needed a block waits for in a reference still being decoded.
const WAIT_SLACK: isize = 16;

/// Reference index of an unavailable neighbour in [`mvp_raw`] form; its vector is zero.
const UNAV: i8 = i8::MIN;
const NO_MV: Mv = (UNAV, [0, 0]);

/// [`mvp_from`] without a partition shape, for neighbours given with [`UNAV`] for unavailable
/// (`refidx` is at least 0).
fn mvp_raw(a: Mv, b: Mv, c: Mv, refidx: i8) -> [i16; 2] {
    let (a, b, c) = if b.0 == UNAV && c.0 == UNAV && a.0 != UNAV {
        (a, a, a)
    } else {
        (a, b, c)
    };
    let m = (a.0 == refidx) as u8 | ((b.0 == refidx) as u8) << 1 | ((c.0 == refidx) as u8) << 2;
    match m {
        1 => a.1,
        2 => b.1,
        4 => c.1,
        _ => [
            median(a.1[0], b.1[0], c.1[0]),
            median(a.1[1], b.1[1], c.1[1]),
        ],
    }
}

/// Motion vector predictor (8.4.1.3) from the neighbours A, B and C (or D) of a partition,
/// [`NO_MV`] where unavailable.
fn mvp_from(a: Mv, b: Mv, c: Mv, refidx: i8, part: Option<(Part, usize)>) -> [i16; 2] {
    match part {
        Some((Part::P16x8, 0)) if b.0 == refidx => return b.1,
        Some((Part::P16x8, 1)) if a.0 == refidx => return a.1,
        Some((Part::P8x16, 0)) if a.0 == refidx => return a.1,
        Some((Part::P8x16, 1)) if c.0 == refidx => return c.1,
        _ => {}
    }
    mvp_raw(a, b, c, refidx)
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

/// [`Dec::luma_into`] for a block reading past the reference picture (`pw`x`ph`).
#[allow(clippy::too_many_arguments)]
#[cold]
#[inline(never)]
fn mc_luma_edge(
    r: &RefPic,
    mv: [i16; 2],
    (pw, ph): (usize, usize),
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    avg: bool,
    dst: &mut [u8],
    ds: usize,
) {
    if avg {
        recon::mc_luma::<true>(&r.planes[0], pw, ph, x, y, w, h, mv, dst, ds);
    } else {
        recon::mc_luma::<false>(&r.planes[0], pw, ph, x, y, w, h, mv, dst, ds);
    }
}

/// [`Dec::chroma_into`] for blocks reading past the reference picture (chroma `cw`x`ch`).
#[allow(clippy::too_many_arguments)]
#[cold]
#[inline(never)]
fn mc_chroma_edge(
    r: &RefPic,
    mv: [i16; 2],
    (cw, ch): (usize, usize),
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    avg: bool,
    dst: [&mut [u8]; 2],
    ds: usize,
) {
    let src = [&r.planes[1][..], &r.planes[2][..]];
    if avg {
        recon::mc_chroma_pair::<true>(src, cw, ch, x, y, w, h, mv, dst, ds);
    } else {
        recon::mc_chroma_pair::<false>(src, cw, ch, x, y, w, h, mv, dst, ds);
    }
}
