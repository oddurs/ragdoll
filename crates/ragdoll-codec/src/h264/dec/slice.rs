//! Slice header (ITU-T H.264 7.3.3).

use ragdoll_core::{BitReader, RagdollError, Result};

use super::ps::{Pps, Sps};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SliceType {
    P,
    B,
    I,
    Sp,
    Si,
}

impl SliceType {
    pub fn is_intra(self) -> bool {
        matches!(self, Self::I | Self::Si)
    }
    pub fn is_b(self) -> bool {
        self == Self::B
    }
}

/// Reference picture list modification operation (7.3.3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefMod {
    Subtract(u32),
    Add(u32),
    LongTerm(u32),
}

/// Memory management control operation (7.3.3.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mmco {
    UnmarkShort {
        diff_pic_nums_minus1: u32,
    },
    UnmarkLong {
        long_term_pic_num: u32,
    },
    ShortToLong {
        diff_pic_nums_minus1: u32,
        long_term_frame_idx: u32,
    },
    MaxLongTermIdx {
        max_plus1: u32,
    },
    Reset,
    CurrentToLong {
        long_term_frame_idx: u32,
    },
}

/// Explicit weighted prediction entry `(weight, offset)` for luma and both chroma planes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightEntry {
    pub luma: (i32, i32),
    pub chroma: [(i32, i32); 2],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PredWeightTable {
    pub luma_log2: u32,
    pub chroma_log2: u32,
    pub l: [Vec<WeightEntry>; 2],
}

#[derive(Clone, Debug)]
pub struct SliceHeader {
    pub nal_ref_idc: u8,
    pub idr: bool,
    pub first_mb: u32,
    pub slice_type: SliceType,
    pub pps_id: u32,
    pub frame_num: u32,
    pub field_pic: bool,
    pub bottom_field: bool,
    pub idr_pic_id: u32,
    pub poc_lsb: u32,
    pub delta_poc_bottom: i32,
    pub delta_poc: [i32; 2],
    pub redundant_pic_cnt: u32,
    pub direct_spatial: bool,
    pub num_ref_idx_active: [u32; 2],
    pub ref_mods: [Vec<RefMod>; 2],
    pub weights: Option<PredWeightTable>,
    pub long_term_reference: bool,
    pub mmcos: Option<Vec<Mmco>>,
    pub cabac_init_idc: u32,
    pub qp: i32,
    pub disable_deblocking_filter_idc: u32,
    pub alpha_offset: i32,
    pub beta_offset: i32,
    /// Bit offset of `slice_data()` within the RBSP.
    pub data_bit_offset: usize,
}

impl SliceHeader {
    /// Parses the first fields up to `pic_parameter_set_id`, enough to select parameter sets.
    pub fn peek_pps_id(rbsp: &[u8]) -> Result<u32> {
        let mut r = BitReader::new(rbsp);
        r.read_ue()?;
        r.read_ue()?;
        Ok(r.read_ue()? as u32)
    }

    pub fn parse(
        rbsp: &[u8],
        nal_unit_type: u8,
        nal_ref_idc: u8,
        sps: &Sps,
        pps: &Pps,
    ) -> Result<Self> {
        let mut r = BitReader::new(rbsp);
        let idr = nal_unit_type == 5;
        let first_mb = r.read_ue()? as u32;
        let raw_type = r.read_ue()?;
        let slice_type = match raw_type % 5 {
            0 => SliceType::P,
            1 => SliceType::B,
            2 => SliceType::I,
            3 => SliceType::Sp,
            _ => SliceType::Si,
        };
        if raw_type > 9 {
            return Err(RagdollError::InvalidSyntax("H.264 slice_type > 9"));
        }
        let pps_id = r.read_ue()? as u32;
        if sps.separate_colour_plane {
            r.read_bits(2)?; // colour_plane_id
        }
        let frame_num = r.read_bits(sps.log2_max_frame_num)? as u32;
        let (mut field_pic, mut bottom_field) = (false, false);
        if !sps.frame_mbs_only {
            field_pic = r.read_bit()?;
            if field_pic {
                bottom_field = r.read_bit()?;
            }
        }
        let idr_pic_id = if idr { r.read_ue()? as u32 } else { 0 };
        let mut poc_lsb = 0;
        let mut delta_poc_bottom = 0;
        let mut delta_poc = [0i32; 2];
        if sps.poc_type == 0 {
            poc_lsb = r.read_bits(sps.log2_max_poc_lsb)? as u32;
            if pps.bottom_field_pic_order_in_frame_present && !field_pic {
                delta_poc_bottom = r.read_se()? as i32;
            }
        }
        if sps.poc_type == 1 && !sps.delta_pic_order_always_zero {
            delta_poc[0] = r.read_se()? as i32;
            if pps.bottom_field_pic_order_in_frame_present && !field_pic {
                delta_poc[1] = r.read_se()? as i32;
            }
        }
        let redundant_pic_cnt = if pps.redundant_pic_cnt_present {
            r.read_ue()? as u32
        } else {
            0
        };
        let direct_spatial = if slice_type == SliceType::B {
            r.read_bit()?
        } else {
            false
        };
        let mut num_ref_idx_active = pps.num_ref_idx_default;
        if matches!(slice_type, SliceType::P | SliceType::Sp | SliceType::B) && r.read_bit()? {
            num_ref_idx_active[0] = 1 + r.read_ue()? as u32;
            if slice_type == SliceType::B {
                num_ref_idx_active[1] = 1 + r.read_ue()? as u32;
            }
        }
        let max_refs = if field_pic { 32 } else { 16 };
        if num_ref_idx_active[0] > max_refs || num_ref_idx_active[1] > max_refs {
            return Err(RagdollError::InvalidSyntax(
                "H.264 num_ref_idx_active out of range",
            ));
        }
        if slice_type != SliceType::B {
            num_ref_idx_active[1] = 0;
        }
        if slice_type.is_intra() {
            num_ref_idx_active[0] = 0;
        }

        let mut ref_mods: [Vec<RefMod>; 2] = [Vec::new(), Vec::new()];
        for (list, mods) in ref_mods.iter_mut().enumerate() {
            let present = match list {
                0 => !slice_type.is_intra(),
                _ => slice_type == SliceType::B,
            };
            if present && r.read_bit()? {
                loop {
                    let idc = r.read_ue()?;
                    let op = match idc {
                        0 => RefMod::Subtract(r.read_ue()? as u32),
                        1 => RefMod::Add(r.read_ue()? as u32),
                        2 => RefMod::LongTerm(r.read_ue()? as u32),
                        3 => break,
                        _ => {
                            return Err(RagdollError::InvalidSyntax(
                                "H.264 modification_of_pic_nums_idc > 3",
                            ));
                        }
                    };
                    if mods.len() > max_refs as usize {
                        return Err(RagdollError::InvalidSyntax(
                            "H.264 too many ref list modifications",
                        ));
                    }
                    mods.push(op);
                }
            }
        }

        let explicit = (pps.weighted_pred && matches!(slice_type, SliceType::P | SliceType::Sp))
            || (pps.weighted_bipred_idc == 1 && slice_type == SliceType::B);
        let weights = if explicit {
            Some(parse_pred_weight_table(
                &mut r,
                sps,
                slice_type,
                num_ref_idx_active,
            )?)
        } else {
            None
        };

        let mut long_term_reference = false;
        let mut mmcos = None;
        if nal_ref_idc != 0 {
            if idr {
                r.read_bit()?; // no_output_of_prior_pics_flag: every picture is output
                long_term_reference = r.read_bit()?;
            } else if r.read_bit()? {
                let mut ops = Vec::new();
                loop {
                    let op = match r.read_ue()? {
                        0 => break,
                        1 => Mmco::UnmarkShort {
                            diff_pic_nums_minus1: r.read_ue()? as u32,
                        },
                        2 => Mmco::UnmarkLong {
                            long_term_pic_num: r.read_ue()? as u32,
                        },
                        3 => Mmco::ShortToLong {
                            diff_pic_nums_minus1: r.read_ue()? as u32,
                            long_term_frame_idx: r.read_ue()? as u32,
                        },
                        4 => Mmco::MaxLongTermIdx {
                            max_plus1: r.read_ue()? as u32,
                        },
                        5 => Mmco::Reset,
                        6 => Mmco::CurrentToLong {
                            long_term_frame_idx: r.read_ue()? as u32,
                        },
                        _ => {
                            return Err(RagdollError::InvalidSyntax(
                                "H.264 memory_management_control_operation > 6",
                            ));
                        }
                    };
                    if ops.len() >= 66 {
                        return Err(RagdollError::InvalidSyntax(
                            "H.264 too many MMCO operations",
                        ));
                    }
                    ops.push(op);
                }
                mmcos = Some(ops);
            }
        }
        let cabac_init_idc = if pps.cabac && !slice_type.is_intra() {
            let v = r.read_ue()? as u32;
            if v > 2 {
                return Err(RagdollError::InvalidSyntax("H.264 cabac_init_idc > 2"));
            }
            v
        } else {
            0
        };
        let qp = pps.pic_init_qp + r.read_se()? as i32;
        if !(0..=51).contains(&qp) {
            return Err(RagdollError::InvalidSyntax("H.264 SliceQPY out of range"));
        }
        if matches!(slice_type, SliceType::Sp | SliceType::Si) {
            return Err(RagdollError::Unsupported(
                "H.264 SP and SI slices are not supported".into(),
            ));
        }
        let mut disable_deblocking_filter_idc = 0;
        let mut alpha_offset = 0;
        let mut beta_offset = 0;
        if pps.deblocking_filter_control_present {
            disable_deblocking_filter_idc = r.read_ue()? as u32;
            if disable_deblocking_filter_idc > 2 {
                return Err(RagdollError::InvalidSyntax(
                    "H.264 disable_deblocking_filter_idc > 2",
                ));
            }
            if disable_deblocking_filter_idc != 1 {
                alpha_offset = 2 * r.read_se()? as i32;
                beta_offset = 2 * r.read_se()? as i32;
                if !(-12..=12).contains(&alpha_offset) || !(-12..=12).contains(&beta_offset) {
                    return Err(RagdollError::InvalidSyntax(
                        "H.264 deblocking offset out of range",
                    ));
                }
            }
        }
        if pps.num_slice_groups > 1 {
            return Err(RagdollError::Unsupported(
                "H.264 slice groups (FMO) are not supported".into(),
            ));
        }
        Ok(Self {
            nal_ref_idc,
            idr,
            first_mb,
            slice_type,
            pps_id,
            frame_num,
            field_pic,
            bottom_field,
            idr_pic_id,
            poc_lsb,
            delta_poc_bottom,
            delta_poc,
            redundant_pic_cnt,
            direct_spatial,
            num_ref_idx_active,
            ref_mods,
            weights,
            long_term_reference,
            mmcos,
            cabac_init_idc,
            qp,
            disable_deblocking_filter_idc,
            alpha_offset,
            beta_offset,
            data_bit_offset: r.bits_read(),
        })
    }

    /// True when `self` starts a new primary coded picture after `prev` (7.4.1.2.4).
    pub fn starts_new_picture(&self, prev: &Self, sps: &Sps) -> bool {
        self.frame_num != prev.frame_num
            || self.pps_id != prev.pps_id
            || self.field_pic != prev.field_pic
            || (self.field_pic && self.bottom_field != prev.bottom_field)
            || ((self.nal_ref_idc == 0) != (prev.nal_ref_idc == 0))
            || (sps.poc_type == 0
                && (self.poc_lsb != prev.poc_lsb || self.delta_poc_bottom != prev.delta_poc_bottom))
            || (sps.poc_type == 1 && self.delta_poc != prev.delta_poc)
            || self.idr != prev.idr
            || (self.idr && prev.idr && self.idr_pic_id != prev.idr_pic_id)
    }
}

fn parse_pred_weight_table(
    r: &mut BitReader<'_>,
    sps: &Sps,
    slice_type: SliceType,
    num_ref: [u32; 2],
) -> Result<PredWeightTable> {
    let luma_log2 = r.read_ue()? as u32;
    if luma_log2 > 7 {
        return Err(RagdollError::InvalidSyntax(
            "H.264 luma_log2_weight_denom > 7",
        ));
    }
    let has_chroma = sps.chroma_array_type() != 0;
    let chroma_log2 = if has_chroma { r.read_ue()? as u32 } else { 0 };
    if chroma_log2 > 7 {
        return Err(RagdollError::InvalidSyntax(
            "H.264 chroma_log2_weight_denom > 7",
        ));
    }
    let lists = if slice_type == SliceType::B { 2 } else { 1 };
    let mut l: [Vec<WeightEntry>; 2] = [Vec::new(), Vec::new()];
    for (list, entries) in l.iter_mut().enumerate().take(lists) {
        for _ in 0..num_ref[list] {
            let mut e = WeightEntry {
                luma: (1 << luma_log2, 0),
                chroma: [(1 << chroma_log2, 0); 2],
            };
            if r.read_bit()? {
                e.luma = (r.read_se()? as i32, r.read_se()? as i32);
            }
            if has_chroma && r.read_bit()? {
                for c in &mut e.chroma {
                    *c = (r.read_se()? as i32, r.read_se()? as i32);
                }
            }
            entries.push(e);
        }
    }
    Ok(PredWeightTable {
        luma_log2,
        chroma_log2,
        l,
    })
}
