//! Decoded pictures and the per-picture macroblock state used while decoding them.

use std::sync::Arc;

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

/// Motion data a reference picture keeps for direct prediction of later B pictures.
#[derive(Clone, Debug)]
pub struct Motion {
    /// Per 4x4 block, per list.
    pub mv: [Vec<[i16; 2]>; 2],
    pub ref_idx: [Vec<i8>; 2],
    /// Unique id of the picture each block references, or [`NO_PIC`].
    pub ref_pic: [Vec<u32>; 2],
    /// Per macroblock.
    pub mb_intra: Vec<bool>,
}

pub const NO_PIC: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefKind {
    Unused,
    Short,
    Long,
}

/// A decoded frame held in the DPB.
#[derive(Clone, Debug)]
pub struct Picture {
    pub id: u32,
    pub planes: Arc<[Vec<u8>; 3]>,
    pub width: usize,
    pub height: usize,
    pub poc: i32,
    pub top_poc: i32,
    pub bottom_poc: i32,
    pub frame_num: u32,
    pub frame_num_wrap: i32,
    pub long_term_frame_idx: u32,
    pub kind: RefKind,
    pub motion: Option<Arc<Motion>>,
}

impl Picture {
    pub fn is_short(&self) -> bool {
        self.kind == RefKind::Short
    }
    pub fn is_long(&self) -> bool {
        self.kind == RefKind::Long
    }
}

/// Per-picture decoding state, indexed by macroblock address or by 4x4 block in raster order.
pub struct PicState {
    pub mb_w: usize,
    pub mb_h: usize,
    pub width: usize,
    pub height: usize,
    pub planes: [Vec<u8>; 3],

    pub slice_num: Vec<u32>,
    pub mb_type: Vec<MbType>,
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
}

impl PicState {
    pub fn new(mb_w: usize, mb_h: usize) -> Self {
        let mbs = mb_w * mb_h;
        let b4 = mbs * 16;
        let c4 = mbs * 4;
        let (w, h) = (mb_w * 16, mb_h * 16);
        Self {
            mb_w,
            mb_h,
            width: w,
            height: h,
            planes: [vec![0; w * h], vec![128; w * h / 4], vec![128; w * h / 4]],
            slice_num: vec![u32::MAX; mbs],
            mb_type: vec![MbType::None; mbs],
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

    pub fn take_motion(&mut self) -> Motion {
        Motion {
            mv: [
                std::mem::take(&mut self.mv[0]),
                std::mem::take(&mut self.mv[1]),
            ],
            ref_idx: [
                std::mem::take(&mut self.ref_idx[0]),
                std::mem::take(&mut self.ref_idx[1]),
            ],
            ref_pic: [
                std::mem::take(&mut self.ref_pic[0]),
                std::mem::take(&mut self.ref_pic[1]),
            ],
            mb_intra: self.mb_type.iter().map(|t| t.is_intra()).collect(),
        }
    }
}
