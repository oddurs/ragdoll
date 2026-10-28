//! Decoded frame stores and the per-picture macroblock state used while decoding them.

use std::sync::{Arc, OnceLock};

use super::tables::Part;

/// Macroblock prediction type after parsing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MbType {
    /// Not yet decoded in this picture.
    None,
    I4x4,
    I8x8,
    /// `I_16x16` with `Intra16x16PredMode`.
    I16x16(u8),
    IPcm,
    PSkip,
    BSkip,
    BDirect16x16,
    /// P or B inter macroblock with a partition shape (B_8x8 and P_8x8 use `Part::P8x8`).
    Inter(Part),
}

impl MbType {
    pub fn is_intra(self) -> bool {
        matches!(self, Self::I4x4 | Self::I8x8 | Self::I16x16(_) | Self::IPcm)
    }
    pub fn is_skip(self) -> bool {
        matches!(self, Self::PSkip | Self::BSkip)
    }
}

/// Coding structure of the picture being decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Structure {
    Frame,
    /// A field picture of the given parity (0 top, 1 bottom).
    Field(u8),
}

/// Motion a frame store keeps for direct prediction of later B pictures, in frame layout:
/// macroblock rows of field pictures are interleaved (field row `r` of parity `p` is frame row `2r + p`).
#[derive(Clone, Debug)]
pub struct Motion {
    /// Per 4x4 block, per list.
    pub mv: [Vec<[i16; 2]>; 2],
    pub ref_idx: [Vec<i8>; 2],
    /// Reference id of the picture each block references (see [`FrameStore::frame_ref_id`]), or [`NO_PIC`].
    pub ref_pic: [Vec<u32>; 2],
    /// Per macroblock.
    pub mb_intra: Vec<bool>,
    /// Per macroblock: decoded as a field macroblock (field picture or field MB pair).
    pub mb_field: Vec<bool>,
    /// Decoded as an MBAFF frame (rather than as a frame or two field pictures).
    pub mbaff: bool,
}

impl Motion {
    pub fn new(mb_w: usize, mb_h: usize) -> Self {
        let b4 = mb_w * mb_h * 16;
        Self {
            mv: [vec![[0; 2]; b4], vec![[0; 2]; b4]],
            ref_idx: [vec![-1; b4], vec![-1; b4]],
            ref_pic: [vec![NO_PIC; b4], vec![NO_PIC; b4]],
            mb_intra: vec![true; mb_w * mb_h],
            mb_field: vec![false; mb_w * mb_h],
            mbaff: false,
        }
    }
}

pub const NO_PIC: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefKind {
    Unused,
    Short,
    Long,
}

/// A decoded frame, or a field pair being assembled, held in the DPB.
#[derive(Clone, Debug)]
pub struct FrameStore {
    pub id: u32,
    pub planes: Arc<[Vec<u8>; 3]>,
    field_planes: [OnceLock<Arc<[Vec<u8>; 3]>>; 2],
    pub width: usize,
    pub height: usize,
    pub field_poc: [i32; 2],
    pub decoded: [bool; 2],
    pub kind: [RefKind; 2],
    pub frame_num: u32,
    pub frame_num_wrap: i32,
    pub long_term_frame_idx: u32,
    pub motion: Arc<Motion>,
}

impl FrameStore {
    pub fn new(id: u32, width: usize, height: usize, mb_w: usize, mb_h: usize) -> Self {
        Self {
            id,
            planes: Arc::new([
                vec![0; width * height],
                vec![128; width * height / 4],
                vec![128; width * height / 4],
            ]),
            field_planes: Default::default(),
            width,
            height,
            field_poc: [0; 2],
            decoded: [false; 2],
            kind: [RefKind::Unused; 2],
            frame_num: 0,
            frame_num_wrap: 0,
            long_term_frame_idx: 0,
            motion: Arc::new(Motion::new(mb_w, mb_h)),
        }
    }

    /// POC of the frame: the smaller POC of its decoded fields.
    /// Makes a recycled store a new, empty one with id `id`. With `init`, samples and motion are
    /// reset too (a field picture fills only half of them); a frame picture replaces both.
    pub fn reuse(&mut self, id: u32, init: bool) {
        self.id = id;
        self.field_planes = Default::default();
        self.field_poc = [0; 2];
        self.decoded = [false; 2];
        self.kind = [RefKind::Unused; 2];
        self.frame_num = 0;
        self.frame_num_wrap = 0;
        self.long_term_frame_idx = 0;
        if init {
            let planes = Arc::make_mut(&mut self.planes);
            planes[0].fill(0);
            planes[1].fill(128);
            planes[2].fill(128);
            let n = self.motion.mb_intra.len();
            let m = Arc::make_mut(&mut self.motion);
            for l in 0..2 {
                m.mv[l].fill([0; 2]);
                m.ref_idx[l].fill(-1);
                m.ref_pic[l].fill(NO_PIC);
            }
            m.mb_intra.fill(true);
            m.mb_field.fill(false);
            debug_assert_eq!(m.mb_intra.len(), n);
        }
        Arc::make_mut(&mut self.motion).mbaff = false;
    }

    pub fn poc(&self) -> i32 {
        match self.decoded {
            [true, true] => self.field_poc[0].min(self.field_poc[1]),
            [false, true] => self.field_poc[1],
            _ => self.field_poc[0],
        }
    }

    pub fn frame_ref_id(&self) -> u32 {
        self.id * 4
    }

    pub fn field_ref_id(&self, parity: u8) -> u32 {
        self.id * 4 + 1 + parity as u32
    }

    pub fn any(&self, k: RefKind) -> bool {
        self.kind[0] == k || self.kind[1] == k
    }

    pub fn is_reference(&self) -> bool {
        self.any(RefKind::Short) || self.any(RefKind::Long)
    }

    /// The samples of one field as a half-height picture.
    pub fn field_view(&self, parity: u8) -> Arc<[Vec<u8>; 3]> {
        self.field_planes[parity as usize]
            .get_or_init(|| {
                let p = parity as usize;
                let extract = |plane: &[u8], w: usize, h: usize| -> Vec<u8> {
                    let mut out = Vec::with_capacity(w * h / 2);
                    for y in (p..h).step_by(2) {
                        out.extend_from_slice(&plane[y * w..y * w + w]);
                    }
                    out
                };
                let (w, h) = (self.width, self.height);
                Arc::new([
                    extract(&self.planes[0], w, h),
                    extract(&self.planes[1], w / 2, h / 2),
                    extract(&self.planes[2], w / 2, h / 2),
                ])
            })
            .clone()
    }

    /// Mutable access to the frame samples; drops cached field views of `parity`.
    pub fn planes_mut(&mut self, parity: Option<u8>) -> &mut [Vec<u8>; 3] {
        match parity {
            Some(p) => self.field_planes[p as usize] = OnceLock::new(),
            None => self.field_planes = Default::default(),
        }
        Arc::make_mut(&mut self.planes)
    }
}

/// Per-picture decoding state, indexed by macroblock address or by 4x4 block in raster order.
/// For a field picture the geometry is that of the field.
pub struct PicState {
    pub structure: Structure,
    /// Macroblock-adaptive frame/field frame: macroblocks are decoded in vertical pairs and
    /// `mb_y = 2 * pair_row + (bottom MB)`; a field pair's MBs hold the pair's top and bottom field.
    pub mbaff: bool,
    pub mb_w: usize,
    pub mb_h: usize,
    pub width: usize,
    pub height: usize,
    pub planes: [Vec<u8>; 3],

    pub slice_num: Vec<u32>,
    pub mb_type: Vec<MbType>,
    /// Field macroblock (field picture, or field pair of an MBAFF frame).
    pub mb_field: Vec<bool>,
    /// Luma CBP in bits 0..3, chroma CBP in bits 4..5.
    pub cbp: Vec<u8>,
    pub qp: Vec<u8>,
    pub qpc: Vec<[u8; 2]>,
    pub t8x8: Vec<bool>,
    pub chroma_pred_mode: Vec<u8>,
    /// CABAC DC coded_block_flags: bit 0 luma DC, bit 1 Cb DC, bit 2 Cr DC.
    pub cbf_dc: Vec<u8>,

    /// Per 4x4 luma block: total_coeff (CAVLC nC) / coefficient count.
    pub nnz: Vec<u8>,
    /// Per 4x4 luma block: non-zero coefficients in the transform block containing it (deblocking, CABAC cbf).
    pub nz: Vec<bool>,
    pub intra_mode: Vec<i8>,
    pub mv: [Vec<[i16; 2]>; 2],
    pub mvd: [Vec<[u8; 2]>; 2],
    pub ref_idx: [Vec<i8>; 2],
    pub ref_pic: [Vec<u32>; 2],
    pub direct: Vec<bool>,
    /// Per 4x4 chroma block of each plane: total_coeff of the AC block.
    pub nnz_c: [Vec<u8>; 2],
    /// The buffers were reused from an earlier picture: each macroblock's state is restored by
    /// [`PicState::clear_mb`] when it is decoded, and [`PicState::restore_undecoded`] restores
    /// what no slice covered.
    pub recycled: bool,
}

impl PicState {
    pub fn new(structure: Structure, mb_w: usize, mb_h: usize, mbaff: bool) -> Self {
        let mbs = mb_w * mb_h;
        let b4 = mbs * 16;
        let c4 = mbs * 4;
        let (w, h) = (mb_w * 16, mb_h * 16);
        Self {
            structure,
            mbaff,
            mb_w,
            mb_h,
            width: w,
            height: h,
            planes: [vec![0; w * h], vec![128; w * h / 4], vec![128; w * h / 4]],
            slice_num: vec![u32::MAX; mbs],
            mb_type: vec![MbType::None; mbs],
            mb_field: vec![matches!(structure, Structure::Field(_)); mbs],
            cbp: vec![0; mbs],
            qp: vec![0; mbs],
            qpc: vec![[0; 2]; mbs],
            t8x8: vec![false; mbs],
            chroma_pred_mode: vec![0; mbs],
            cbf_dc: vec![0; mbs],
            nnz: vec![0; b4],
            nz: vec![false; b4],
            intra_mode: vec![-1; b4],
            mv: [vec![[0; 2]; b4], vec![[0; 2]; b4]],
            mvd: [vec![[0; 2]; b4], vec![[0; 2]; b4]],
            ref_idx: [vec![-1; b4], vec![-1; b4]],
            ref_pic: [vec![NO_PIC; b4], vec![NO_PIC; b4]],
            direct: vec![false; b4],
            nnz_c: [vec![0; c4], vec![0; c4]],
            recycled: false,
        }
    }

    /// A state for a new picture, reusing `old`'s buffers when the geometry matches. Only the
    /// availability state (`slice_num`, `mb_type`, `mb_field`) is reset here.
    pub fn recycle(
        old: Option<Self>,
        structure: Structure,
        mb_w: usize,
        mb_h: usize,
        mbaff: bool,
    ) -> Self {
        match old {
            Some(mut st) if st.mb_w == mb_w && st.mb_h == mb_h => {
                st.structure = structure;
                st.mbaff = mbaff;
                st.slice_num.fill(u32::MAX);
                st.mb_type.fill(MbType::None);
                st.mb_field.fill(matches!(structure, Structure::Field(_)));
                st.recycled = true;
                st
            }
            _ => Self::new(structure, mb_w, mb_h, mbaff),
        }
    }

    /// Restores macroblock `addr`'s coding state (all but availability) to that of a new picture.
    pub fn clear_mb(&mut self, addr: usize) {
        self.cbp[addr] = 0;
        self.qp[addr] = 0;
        self.qpc[addr] = [0; 2];
        self.t8x8[addr] = false;
        self.chroma_pred_mode[addr] = 0;
        self.cbf_dc[addr] = 0;
        let (x, y) = (addr % self.mb_w, addr / self.mb_w);
        for r in 0..4 {
            let i = self.b4(x * 4, y * 4 + r);
            self.nnz[i..i + 4].copy_from_slice(&[0; 4]);
            self.nz[i..i + 4].copy_from_slice(&[false; 4]);
            self.intra_mode[i..i + 4].copy_from_slice(&[-1; 4]);
            self.direct[i..i + 4].copy_from_slice(&[false; 4]);
            for l in 0..2 {
                self.mv[l][i..i + 4].copy_from_slice(&[[0; 2]; 4]);
                self.mvd[l][i..i + 4].copy_from_slice(&[[0; 2]; 4]);
                self.ref_idx[l][i..i + 4].copy_from_slice(&[-1; 4]);
                self.ref_pic[l][i..i + 4].copy_from_slice(&[NO_PIC; 4]);
            }
        }
        for r in 0..2 {
            let i = self.c4(x * 2, y * 2 + r);
            for c in &mut self.nnz_c {
                c[i..i + 2].copy_from_slice(&[0; 2]);
            }
        }
    }

    /// For a recycled state: gives every macroblock no slice decoded the state and samples of a
    /// new picture (luma 0, chroma 128).
    pub fn restore_undecoded(&mut self) {
        if !self.recycled {
            return;
        }
        let (w, cw) = (self.width, self.width / 2);
        for addr in 0..self.mb_w * self.mb_h {
            if self.mb_type[addr] != MbType::None {
                continue;
            }
            self.clear_mb(addr);
            let (x, y) = (addr % self.mb_w, addr / self.mb_w);
            // Frame lines the macroblock covers: in an MBAFF frame a field pair interleaves.
            // The pair's other macroblock decides whether the pair is a field pair.
            let pair_field = self.mbaff && {
                let p = if y % 2 == 0 {
                    addr + self.mb_w
                } else {
                    addr - self.mb_w
                };
                self.mb_type[p] != MbType::None && self.mb_field[p]
            };
            let lines = |n: usize, top: usize| -> Vec<usize> {
                if pair_field {
                    (0..n).map(|k| (y & !1) * n + (y & 1) + 2 * k).collect()
                } else {
                    (0..n).map(|k| top + k).collect()
                }
            };
            for line in lines(16, y * 16) {
                self.planes[0][line * w + x * 16..line * w + x * 16 + 16].fill(0);
            }
            for line in lines(8, y * 8) {
                for c in 1..3 {
                    self.planes[c][line * cw + x * 8..line * cw + x * 8 + 8].fill(128);
                }
            }
        }
    }

    /// Index of the 4x4 block at absolute 4x4 coordinates.
    #[inline]
    pub fn b4(&self, x4: usize, y4: usize) -> usize {
        y4 * self.mb_w * 4 + x4
    }

    /// Index of the chroma 4x4 block at absolute chroma-4x4 coordinates.
    #[inline]
    pub fn c4(&self, x4: usize, y4: usize) -> usize {
        y4 * self.mb_w * 2 + x4
    }

    /// Moves samples and motion into `store`: a frame picture swaps buffers with it (leaving this
    /// state with the store's old buffers, for [`PicState::recycle`]); a field picture's rows are
    /// copied into its lines.
    pub fn write_into(&mut self, store: &mut FrameStore) {
        let parity = match self.structure {
            Structure::Frame => None,
            Structure::Field(p) => Some(p),
        };
        let planes = store.planes_mut(parity);
        for (c, (dst, src)) in planes.iter_mut().zip(self.planes.iter_mut()).enumerate() {
            let (w, h) = if c == 0 {
                (self.width, self.height)
            } else {
                (self.width / 2, self.height / 2)
            };
            match parity {
                None if dst.len() == src.len() => std::mem::swap(dst, src),
                None => dst[..w * h].copy_from_slice(&src[..w * h]),
                Some(p) => {
                    for y in 0..h {
                        let dy = 2 * y + p as usize;
                        dst[dy * w..dy * w + w].copy_from_slice(&src[y * w..y * w + w]);
                    }
                }
            }
        }
        let m = Arc::make_mut(&mut store.motion);
        m.mbaff = self.mbaff;
        let stride = self.mb_w * 4;
        if parity.is_none() && m.mv[0].len() == self.mv[0].len() {
            for l in 0..2 {
                std::mem::swap(&mut m.mv[l], &mut self.mv[l]);
                std::mem::swap(&mut m.ref_idx[l], &mut self.ref_idx[l]);
                std::mem::swap(&mut m.ref_pic[l], &mut self.ref_pic[l]);
            }
            for (i, t) in self.mb_type.iter().enumerate() {
                m.mb_intra[i] = t.is_intra();
            }
            m.mb_field.copy_from_slice(&self.mb_field);
            return;
        }
        for mby in 0..self.mb_h {
            let fy = match parity {
                None => mby,
                Some(p) => 2 * mby + p as usize,
            };
            for mbx in 0..self.mb_w {
                let src_mb = mby * self.mb_w + mbx;
                let dst_mb = fy * self.mb_w + mbx;
                m.mb_intra[dst_mb] = self.mb_type[src_mb].is_intra();
                m.mb_field[dst_mb] = self.mb_field[src_mb];
                for r in 0..4 {
                    let s = (mby * 4 + r) * stride + mbx * 4;
                    let d = (fy * 4 + r) * stride + mbx * 4;
                    for l in 0..2 {
                        m.mv[l][d..d + 4].copy_from_slice(&self.mv[l][s..s + 4]);
                        m.ref_idx[l][d..d + 4].copy_from_slice(&self.ref_idx[l][s..s + 4]);
                        m.ref_pic[l][d..d + 4].copy_from_slice(&self.ref_pic[l][s..s + 4]);
                    }
                }
            }
        }
    }
}
