//! Sequence and picture parameter sets (ITU-T H.264 7.3.2.1 and 7.3.2.2).

use ragdoll_core::{
    BitReader, ColorPrimaries, ColorRange, Colorimetry, MatrixCoefficients, RagdollError,
    Rational64, Result, TransferCharacteristics,
};

use super::tables::{DEFAULT_4X4_INTER, DEFAULT_4X4_INTRA, DEFAULT_8X8_INTER, DEFAULT_8X8_INTRA};

/// One parsed scaling list: absent, "use the default list", or explicit values in zig-zag order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ScalingList {
    #[default]
    Absent,
    UseDefault,
    Explicit(Vec<u8>),
}

/// Scaling matrices as transmitted: 6 4x4 lists and up to 6 8x8 lists (only 2 used for 4:2:0).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalingLists {
    pub lists4: [ScalingList; 6],
    pub lists8: [ScalingList; 6],
}

/// Resolved weight scales in zig-zag order: `w4[0..6]` (Intra Y/Cb/Cr, Inter Y/Cb/Cr) and `w8[0..2]` (Intra Y, Inter Y).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightScales {
    pub w4: [[u8; 16]; 6],
    pub w8: [[u8; 64]; 2],
}

impl WeightScales {
    pub fn flat() -> Self {
        Self {
            w4: [[16; 16]; 6],
            w8: [[16; 64]; 2],
        }
    }
}

/// `profile_idc` of the Baseline profile, whose streams have no B slices.
pub const BASELINE: u8 = 66;

#[derive(Clone, Debug)]
pub struct Sps {
    pub profile_idc: u8,
    pub id: u32,
    pub chroma_format_idc: u32,
    pub separate_colour_plane: bool,
    pub bit_depth_luma: u32,
    pub bit_depth_chroma: u32,
    pub scaling: Option<ScalingLists>,
    pub log2_max_frame_num: u32,
    pub poc_type: u32,
    pub log2_max_poc_lsb: u32,
    pub delta_pic_order_always_zero: bool,
    pub offset_for_non_ref_pic: i32,
    pub offset_for_top_to_bottom_field: i32,
    pub offset_for_ref_frame: Vec<i32>,
    pub max_num_ref_frames: u32,
    pub width_mbs: u32,
    pub height_map_units: u32,
    pub frame_mbs_only: bool,
    pub mbaff: bool,
    pub direct_8x8_inference: bool,
    pub crop: [u32; 4],
    pub fps: Option<Rational64>,
    pub sar: Rational64,
    pub colorimetry: Colorimetry,
    pub max_dec_frame_buffering: Option<u32>,
    pub num_reorder_frames: Option<u32>,
}

impl Sps {
    /// Frame height in macroblocks.
    pub fn height_mbs(&self) -> u32 {
        self.height_map_units * if self.frame_mbs_only { 1 } else { 2 }
    }

    /// `ChromaArrayType`.
    pub fn chroma_array_type(&self) -> u32 {
        if self.separate_colour_plane {
            0
        } else {
            self.chroma_format_idc
        }
    }

    /// Display rectangle `(x, y, width, height)` after frame cropping.
    pub fn crop_rect(&self) -> (u32, u32, u32, u32) {
        let (sub_w, sub_h) = match self.chroma_array_type() {
            1 => (2, 2),
            2 => (2, 1),
            _ => (1, 1),
        };
        let unit_x = sub_w;
        let unit_y = sub_h * if self.frame_mbs_only { 1 } else { 2 };
        let w = self.width_mbs * 16;
        let h = self.height_mbs() * 16;
        let x0 = (self.crop[0] * unit_x).min(w - 1);
        let y0 = (self.crop[2] * unit_y).min(h - 1);
        let cw = w
            .saturating_sub((self.crop[0] + self.crop[1]) * unit_x)
            .max(1);
        let ch = h
            .saturating_sub((self.crop[2] + self.crop[3]) * unit_y)
            .max(1);
        (x0, y0, cw.min(w - x0), ch.min(h - y0))
    }

    pub fn parse(rbsp: &[u8]) -> Result<Self> {
        let mut r = BitReader::new(rbsp);
        let profile_idc = r.read_bits(8)? as u8;
        r.read_bits(16)?; // constraint_set flags, level_idc
        let id = r.read_ue()? as u32;
        if id > 31 {
            return Err(RagdollError::InvalidSyntax(
                "H.264 seq_parameter_set_id > 31",
            ));
        }
        let mut chroma_format_idc = 1;
        let mut separate_colour_plane = false;
        let mut bit_depth_luma = 8;
        let mut bit_depth_chroma = 8;
        let mut scaling = None;
        if matches!(
            profile_idc,
            100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
        ) {
            chroma_format_idc = r.read_ue()? as u32;
            if chroma_format_idc > 3 {
                return Err(RagdollError::InvalidSyntax("H.264 chroma_format_idc > 3"));
            }
            if chroma_format_idc == 3 {
                separate_colour_plane = r.read_bit()?;
            }
            bit_depth_luma = 8 + r.read_ue()? as u32;
            bit_depth_chroma = 8 + r.read_ue()? as u32;
            if bit_depth_luma > 14 || bit_depth_chroma > 14 {
                return Err(RagdollError::InvalidSyntax("H.264 bit depth > 14"));
            }
            if r.read_bit()? {
                return Err(RagdollError::Unsupported(
                    "H.264 lossless (qpprime_y_zero_transform_bypass) is not supported".into(),
                ));
            }
            if r.read_bit()? {
                let n8 = if chroma_format_idc == 3 { 6 } else { 2 };
                scaling = Some(parse_scaling_lists(&mut r, n8)?);
            }
        }
        let log2_max_frame_num = 4 + r.read_ue()? as u32;
        if log2_max_frame_num > 16 {
            return Err(RagdollError::InvalidSyntax("H.264 log2_max_frame_num > 16"));
        }
        let poc_type = r.read_ue()? as u32;
        let mut log2_max_poc_lsb = 4;
        let mut delta_pic_order_always_zero = false;
        let mut offset_for_non_ref_pic = 0;
        let mut offset_for_top_to_bottom_field = 0;
        let mut offset_for_ref_frame = Vec::new();
        match poc_type {
            0 => {
                log2_max_poc_lsb = 4 + r.read_ue()? as u32;
                if log2_max_poc_lsb > 16 {
                    return Err(RagdollError::InvalidSyntax(
                        "H.264 log2_max_pic_order_cnt_lsb > 16",
                    ));
                }
            }
            1 => {
                delta_pic_order_always_zero = r.read_bit()?;
                offset_for_non_ref_pic = r.read_se()? as i32;
                offset_for_top_to_bottom_field = r.read_se()? as i32;
                let n = r.read_ue()?;
                if n > 255 {
                    return Err(RagdollError::InvalidSyntax(
                        "H.264 num_ref_frames_in_pic_order_cnt_cycle > 255",
                    ));
                }
                for _ in 0..n {
                    offset_for_ref_frame.push(r.read_se()? as i32);
                }
            }
            2 => {}
            _ => return Err(RagdollError::InvalidSyntax("H.264 pic_order_cnt_type > 2")),
        }
        let max_num_ref_frames = r.read_ue()? as u32;
        r.read_bit()?; // gaps_in_frame_num_value_allowed_flag: gaps are always filled
        let width_mbs = 1 + r.read_ue()? as u32;
        let height_map_units = 1 + r.read_ue()? as u32;
        let frame_mbs_only = r.read_bit()?;
        // Level 6.2 (MaxFS 139264) is the largest picture H.264 defines.
        let frame_mbs =
            width_mbs as u64 * height_map_units as u64 * if frame_mbs_only { 1 } else { 2 };
        if width_mbs > 1024 || height_map_units > 1024 || frame_mbs > 139_264 {
            return Err(RagdollError::InvalidSyntax(
                "H.264 picture dimensions out of range",
            ));
        }
        let mbaff = !frame_mbs_only && r.read_bit()?;
        let direct_8x8_inference = r.read_bit()?;
        let mut crop = [0u32; 4];
        if r.read_bit()? {
            for c in &mut crop {
                *c = r.read_ue()? as u32;
            }
        }
        let mut sps = Self {
            profile_idc,
            id,
            chroma_format_idc,
            separate_colour_plane,
            bit_depth_luma,
            bit_depth_chroma,
            scaling,
            log2_max_frame_num,
            poc_type,
            log2_max_poc_lsb,
            delta_pic_order_always_zero,
            offset_for_non_ref_pic,
            offset_for_top_to_bottom_field,
            offset_for_ref_frame,
            max_num_ref_frames,
            width_mbs,
            height_map_units,
            frame_mbs_only,
            mbaff,
            direct_8x8_inference,
            crop,
            fps: None,
            sar: Rational64::ONE,
            colorimetry: Colorimetry::bt709_limited(),
            max_dec_frame_buffering: None,
            num_reorder_frames: None,
        };
        if r.read_bit().unwrap_or(false) {
            // A malformed VUI must not reject an otherwise decodable stream.
            let _ = parse_vui(&mut r, &mut sps);
        }
        Ok(sps)
    }
}

fn parse_scaling_list(r: &mut BitReader<'_>, size: usize) -> Result<ScalingList> {
    let mut list = vec![0u8; size];
    let mut last = 8i64;
    let mut next = 8i64;
    for (j, slot) in list.iter_mut().enumerate() {
        if next != 0 {
            let delta = r.read_se()?;
            if !(-128..=127).contains(&delta) {
                return Err(RagdollError::InvalidSyntax(
                    "H.264 delta_scale out of range",
                ));
            }
            next = (last + delta + 256).rem_euclid(256);
            if j == 0 && next == 0 {
                return Ok(ScalingList::UseDefault);
            }
        }
        *slot = if next == 0 { last as u8 } else { next as u8 };
        last = *slot as i64;
    }
    Ok(ScalingList::Explicit(list))
}

fn parse_scaling_lists(r: &mut BitReader<'_>, n8: usize) -> Result<ScalingLists> {
    let mut lists4: [ScalingList; 6] = Default::default();
    let mut lists8: [ScalingList; 6] = Default::default();
    for l in &mut lists4 {
        *l = if r.read_bit()? {
            parse_scaling_list(r, 16)?
        } else {
            ScalingList::Absent
        };
    }
    for l in lists8.iter_mut().take(n8) {
        *l = if r.read_bit()? {
            parse_scaling_list(r, 64)?
        } else {
            ScalingList::Absent
        };
    }
    Ok(ScalingLists { lists4, lists8 })
}

fn parse_hrd(r: &mut BitReader<'_>) -> Result<()> {
    let cpb_cnt = r.read_ue()? + 1;
    if cpb_cnt > 32 {
        return Err(RagdollError::InvalidSyntax("H.264 cpb_cnt_minus1 > 31"));
    }
    r.read_bits(8)?; // bit_rate_scale, cpb_size_scale
    for _ in 0..cpb_cnt {
        r.read_ue()?;
        r.read_ue()?;
        r.read_bit()?;
    }
    r.read_bits(20)?; // four 5-bit length fields
    Ok(())
}

fn parse_vui(r: &mut BitReader<'_>, sps: &mut Sps) -> Result<()> {
    if r.read_bit()? {
        let idc = r.read_bits(8)?;
        const SAR: [(i64, i64); 17] = [
            (1, 1),
            (1, 1),
            (12, 11),
            (10, 11),
            (16, 11),
            (40, 33),
            (24, 11),
            (20, 11),
            (32, 11),
            (80, 33),
            (18, 11),
            (15, 11),
            (64, 33),
            (160, 99),
            (4, 3),
            (3, 2),
            (2, 1),
        ];
        if idc == 255 {
            let w = r.read_bits(16)? as i64;
            let h = r.read_bits(16)? as i64;
            if w > 0 && h > 0 {
                sps.sar = Rational64::new(w, h);
            }
        } else if let Some(&(w, h)) = SAR.get(idc as usize) {
            sps.sar = Rational64::new(w, h);
        }
    }
    if r.read_bit()? {
        r.read_bit()?; // overscan_appropriate_flag
    }
    if r.read_bit()? {
        r.read_bits(3)?; // video_format
        sps.colorimetry.range = if r.read_bit()? {
            ColorRange::Full
        } else {
            ColorRange::Limited
        };
        if r.read_bit()? {
            sps.colorimetry.primaries = ColorPrimaries::from_cicp(r.read_bits(8)? as u8);
            sps.colorimetry.transfer = TransferCharacteristics::from_cicp(r.read_bits(8)? as u8);
            sps.colorimetry.matrix = MatrixCoefficients::from_cicp(r.read_bits(8)? as u8);
        }
    }
    if r.read_bit()? {
        r.read_ue()?;
        r.read_ue()?;
    }
    if r.read_bit()? {
        let units = r.read_bits(32)?;
        let scale = r.read_bits(32)?;
        r.read_bit()?; // fixed_frame_rate_flag
        if units > 0 && scale > 0 {
            sps.fps = Some(Rational64::new(scale as i64, (units * 2) as i64));
        }
    }
    let nal_hrd = r.read_bit()?;
    if nal_hrd {
        parse_hrd(r)?;
    }
    let vcl_hrd = r.read_bit()?;
    if vcl_hrd {
        parse_hrd(r)?;
    }
    if nal_hrd || vcl_hrd {
        r.read_bit()?; // low_delay_hrd_flag
    }
    r.read_bit()?; // pic_struct_present_flag
    if r.read_bit()? {
        r.read_bit()?; // motion_vectors_over_pic_boundaries_flag
        r.read_ue()?; // max_bytes_per_pic_denom
        r.read_ue()?; // max_bits_per_mb_denom
        r.read_ue()?; // log2_max_mv_length_horizontal
        r.read_ue()?; // log2_max_mv_length_vertical
        sps.num_reorder_frames = Some(r.read_ue()? as u32);
        sps.max_dec_frame_buffering = Some(r.read_ue()? as u32);
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct Pps {
    pub id: u32,
    pub sps_id: u32,
    pub cabac: bool,
    pub bottom_field_pic_order_in_frame_present: bool,
    pub num_slice_groups: u32,
    pub num_ref_idx_default: [u32; 2],
    pub weighted_pred: bool,
    pub weighted_bipred_idc: u32,
    pub pic_init_qp: i32,
    pub chroma_qp_index_offset: [i32; 2],
    pub deblocking_filter_control_present: bool,
    pub constrained_intra_pred: bool,
    pub redundant_pic_cnt_present: bool,
    pub transform_8x8_mode: bool,
    pub scaling: Option<ScalingLists>,
}

impl Pps {
    pub fn parse(rbsp: &[u8], sps_table: &[Option<Sps>]) -> Result<Self> {
        let mut r = BitReader::new(rbsp);
        let id = r.read_ue()? as u32;
        if id > 255 {
            return Err(RagdollError::InvalidSyntax(
                "H.264 pic_parameter_set_id > 255",
            ));
        }
        let sps_id = r.read_ue()? as u32;
        if sps_id > 31 {
            return Err(RagdollError::InvalidSyntax(
                "H.264 seq_parameter_set_id > 31",
            ));
        }
        let cabac = r.read_bit()?;
        let bottom_field_pic_order_in_frame_present = r.read_bit()?;
        let num_slice_groups = 1 + r.read_ue()? as u32;
        if num_slice_groups > 8 {
            return Err(RagdollError::InvalidSyntax(
                "H.264 num_slice_groups_minus1 > 7",
            ));
        }
        if num_slice_groups > 1 {
            let map_type = r.read_ue()?;
            match map_type {
                0 => {
                    for _ in 0..num_slice_groups {
                        r.read_ue()?;
                    }
                }
                2 => {
                    for _ in 0..num_slice_groups - 1 {
                        r.read_ue()?;
                        r.read_ue()?;
                    }
                }
                3..=5 => {
                    r.read_bit()?;
                    r.read_ue()?;
                }
                6 => {
                    let n = r.read_ue()? + 1;
                    let bits = 32 - (num_slice_groups - 1).leading_zeros();
                    for _ in 0..n {
                        r.read_bits(bits)?;
                    }
                }
                _ => {}
            }
        }
        let l0 = 1 + r.read_ue()? as u32;
        let l1 = 1 + r.read_ue()? as u32;
        if l0 > 32 || l1 > 32 {
            return Err(RagdollError::InvalidSyntax(
                "H.264 num_ref_idx_default_active > 32",
            ));
        }
        let weighted_pred = r.read_bit()?;
        let weighted_bipred_idc = r.read_bits(2)? as u32;
        let pic_init_qp = 26 + r.read_se()?;
        // pic_init_qp_minus26 is at least -(26 + QpBdOffset) and at most 25 (7.4.2.2).
        if !(-36..=51).contains(&pic_init_qp) {
            return Err(RagdollError::InvalidSyntax(
                "H.264 pic_init_qp out of range",
            ));
        }
        let pic_init_qp = pic_init_qp as i32;
        r.read_se()?; // pic_init_qs_minus26 (SP/SI only)
        let cqp = r.read_se()? as i32;
        if !(-12..=12).contains(&cqp) {
            return Err(RagdollError::InvalidSyntax(
                "H.264 chroma_qp_index_offset out of range",
            ));
        }
        let deblocking_filter_control_present = r.read_bit()?;
        let constrained_intra_pred = r.read_bit()?;
        let redundant_pic_cnt_present = r.read_bit()?;
        let mut transform_8x8_mode = false;
        let mut scaling = None;
        let mut second = cqp;
        if r.more_rbsp_data() {
            transform_8x8_mode = r.read_bit()?;
            if r.read_bit()? {
                let cf = sps_table
                    .get(sps_id as usize)
                    .and_then(|s| s.as_ref())
                    .map_or(1, |s| s.chroma_format_idc);
                let n8 = if transform_8x8_mode {
                    if cf == 3 { 6 } else { 2 }
                } else {
                    0
                };
                scaling = Some(parse_scaling_lists(&mut r, n8)?);
            }
            second = r.read_se()? as i32;
            if !(-12..=12).contains(&second) {
                return Err(RagdollError::InvalidSyntax(
                    "H.264 second_chroma_qp_index_offset out of range",
                ));
            }
        }
        Ok(Self {
            id,
            sps_id,
            cabac,
            bottom_field_pic_order_in_frame_present,
            num_slice_groups,
            num_ref_idx_default: [l0, l1],
            weighted_pred,
            weighted_bipred_idc,
            pic_init_qp,
            chroma_qp_index_offset: [cqp, second],
            deblocking_filter_control_present,
            constrained_intra_pred,
            redundant_pic_cnt_present,
            transform_8x8_mode,
            scaling,
        })
    }
}

fn resolve(list: &ScalingList, default: &[u8], fallback: &[u8]) -> Vec<u8> {
    match list {
        ScalingList::Absent => fallback.to_vec(),
        ScalingList::UseDefault => default.to_vec(),
        ScalingList::Explicit(v) => v.clone(),
    }
}

/// Resolves the scaling matrices in effect for `pps` under `sps` (7.4.2.1.1 and 7.4.2.2, fall-back rules A and B).
pub fn weight_scales(sps: &Sps, pps: &Pps) -> WeightScales {
    let defaults4 = |i: usize| {
        if i < 3 {
            &DEFAULT_4X4_INTRA[..]
        } else {
            &DEFAULT_4X4_INTER[..]
        }
    };
    let defaults8 = |i: usize| {
        if i.is_multiple_of(2) {
            &DEFAULT_8X8_INTRA[..]
        } else {
            &DEFAULT_8X8_INTER[..]
        }
    };

    // Sequence-level lists: Flat_16 when absent, otherwise fall-back rule A.
    let mut seq = WeightScales::flat();
    if let Some(s) = &sps.scaling {
        let mut w4: Vec<Vec<u8>> = Vec::with_capacity(6);
        for i in 0..6 {
            let fb = if i == 0 || i == 3 {
                defaults4(i).to_vec()
            } else {
                w4[i - 1].clone()
            };
            w4.push(resolve(&s.lists4[i], defaults4(i), &fb));
        }
        for (i, v) in w4.iter().enumerate() {
            seq.w4[i].copy_from_slice(v);
        }
        for i in 0..2 {
            seq.w8[i].copy_from_slice(&resolve(&s.lists8[i], defaults8(i), defaults8(i)));
        }
    }
    let Some(p) = &pps.scaling else {
        return seq;
    };
    // Picture-level lists: rule A when the SPS carries no matrix, otherwise rule B.
    let rule_a = sps.scaling.is_none();
    let mut out = WeightScales::flat();
    let mut w4: Vec<Vec<u8>> = Vec::with_capacity(6);
    for i in 0..6 {
        let fb = if i == 0 || i == 3 {
            if rule_a {
                defaults4(i).to_vec()
            } else {
                seq.w4[i].to_vec()
            }
        } else {
            w4[i - 1].clone()
        };
        w4.push(resolve(&p.lists4[i], defaults4(i), &fb));
    }
    for (i, v) in w4.iter().enumerate() {
        out.w4[i].copy_from_slice(v);
    }
    for i in 0..2 {
        let fb = if rule_a {
            defaults8(i).to_vec()
        } else {
            seq.w8[i].to_vec()
        };
        if pps.transform_8x8_mode {
            out.w8[i].copy_from_slice(&resolve(&p.lists8[i], defaults8(i), &fb));
        } else {
            out.w8[i].copy_from_slice(&fb);
        }
    }
    out
}
