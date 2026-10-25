//! 100% pure-Rust H.264 / AVC (ITU-T H.264 / ISO/IEC 14496-10) bitstream parser,
//! transform & quantization engine, CAVLC entropy coder, and Baseline frame encoder/decoder.
//!
//! Reference-hardened against FFmpeg's `libavcodec/h264_ps.c`, `h264_slice.c`, `h264_cavlc.c`,
//! `h264idct_template.c`, and `h264pred_template.c`:
//! - Parses and serializes [`Sps`] (with VUI timing, aspect ratio, CICP colorimetry, and crop offsets),
//!   [`Pps`], [`SliceHeader`], and [`SeiMessage`] RBSP payloads.
//! - Implements H.264 $4\times4$ forward/inverse integer DCT (via `ragdoll-simd`), $4\times4$ Luma DC
//!   Hadamard, $2\times2$ Chroma DC Hadamard, and QP $0..=51$ scaling tables (`LEVEL_SCALE_4X4`).
//! - Implements H.264 Intra $16\times16$ (Vertical, Horizontal, DC, Plane) and Intra $8\times8$ Chroma
//!   spatial prediction, CAVLC `coeff_token` / `total_zeros` / `run_before` coding, `I_16x16`, `I_PCM`,
//!   and `P_Skip` macroblock encoding and decoding.

use ragdoll_core::{
    BitReader, BitWriter, ColorPrimaries, ColorRange, Colorimetry, MatrixCoefficients, PictureType,
    PixelFormat, RagdollError, Rational64, Result, Timestamp, TransferCharacteristics, VideoFrame,
    escape_rbsp, unescape_rbsp,
};
use ragdoll_simd::{
    add4x4_idct_u8, clamp_u8, forward_dct4x4, hadamard4x4, inverse_idct4x4, sad_16x16_u8,
    sad_block_64_u8, sad_block_256_u8, split_annex_b_nals, sub4x4_dct_u8,
};

/// H.264 Network Abstraction Layer (NAL) unit type (`nal_unit_type` in bits `0..=4` of byte 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NalUnitType {
    Unspecified(u8),
    NonIdrSlice, // 1
    SliceDataA,  // 2
    SliceDataB,  // 3
    SliceDataC,  // 4
    IdrSlice,    // 5
    Sei,         // 6
    Sps,         // 7
    Pps,         // 8
    Aud,         // 9
    EndOfSeq,    // 10
    EndOfStream, // 11
    FillerData,  // 12
}

impl NalUnitType {
    #[inline]
    pub const fn from_u8(val: u8) -> Self {
        match val & 0x1F {
            1 => Self::NonIdrSlice,
            2 => Self::SliceDataA,
            3 => Self::SliceDataB,
            4 => Self::SliceDataC,
            5 => Self::IdrSlice,
            6 => Self::Sei,
            7 => Self::Sps,
            8 => Self::Pps,
            9 => Self::Aud,
            10 => Self::EndOfSeq,
            11 => Self::EndOfStream,
            12 => Self::FillerData,
            other => Self::Unspecified(other),
        }
    }

    #[inline]
    pub const fn as_u8(self) -> u8 {
        match self {
            Self::NonIdrSlice => 1,
            Self::SliceDataA => 2,
            Self::SliceDataB => 3,
            Self::SliceDataC => 4,
            Self::IdrSlice => 5,
            Self::Sei => 6,
            Self::Sps => 7,
            Self::Pps => 8,
            Self::Aud => 9,
            Self::EndOfSeq => 10,
            Self::EndOfStream => 11,
            Self::FillerData => 12,
            Self::Unspecified(v) => v & 0x1F,
        }
    }
}

/// 1-byte H.264 NAL unit header (`forbidden_zero_bit | nal_ref_idc | nal_unit_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NalHeader {
    pub nal_ref_idc: u8,
    pub nal_unit_type: NalUnitType,
}

impl NalHeader {
    pub fn parse(byte: u8) -> Result<Self> {
        if (byte & 0x80) != 0 {
            return Err(RagdollError::InvalidSyntax(
                "H.264 NAL forbidden_zero_bit is non-zero",
            ));
        }
        Ok(Self {
            nal_ref_idc: (byte >> 5) & 0x03,
            nal_unit_type: NalUnitType::from_u8(byte & 0x1F),
        })
    }

    #[inline]
    pub const fn to_byte(self) -> u8 {
        ((self.nal_ref_idc & 0x03) << 5) | self.nal_unit_type.as_u8()
    }
}

/// Unescapes a complete NAL unit (starting with the 1-byte NAL header) into `(NalHeader, rbsp_bytes)`.
pub fn parse_nal_rbsp(nal_unit: &[u8]) -> Result<(NalHeader, Vec<u8>)> {
    let Some(&first) = nal_unit.first() else {
        return Err(RagdollError::BitstreamEndOfData);
    };
    let header = NalHeader::parse(first)?;
    let rbsp = unescape_rbsp(&nal_unit[1..]);
    Ok((header, rbsp))
}

/// Packages raw RBSP payload bytes into an escaped NAL unit with a 4-byte Annex-B start code (`00 00 00 01`).
pub fn encode_annex_b_nal(nal_ref_idc: u8, nal_unit_type: NalUnitType, rbsp: &[u8]) -> Vec<u8> {
    let escaped = escape_rbsp(rbsp);
    let mut out = Vec::with_capacity(5 + escaped.len());
    out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
    out.push(
        NalHeader {
            nal_ref_idc,
            nal_unit_type,
        }
        .to_byte(),
    );
    out.extend_from_slice(&escaped);
    out
}

/// Parsed H.264 Sequence Parameter Set (`SPS`, `nal_unit_type = 7`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sps {
    pub profile_idc: u8,
    pub constraint_set_flags: u8,
    pub level_idc: u8,
    pub sps_id: u8,
    pub chroma_format_idc: u8,
    pub bit_depth_luma: u8,
    pub bit_depth_chroma: u8,
    pub log2_max_frame_num: u8,
    pub pic_order_cnt_type: u8,
    pub log2_max_pic_order_cnt_lsb: u8,
    pub max_num_ref_frames: u8,
    pub pic_width_in_mbs: u32,
    pub pic_height_in_map_units: u32,
    pub frame_mbs_only_flag: bool,
    pub crop_left: u32,
    pub crop_right: u32,
    pub crop_top: u32,
    pub crop_bottom: u32,
    pub sar: Rational64,
    pub fps: Option<Rational64>,
    pub colorimetry: Colorimetry,
}

impl Sps {
    /// Creates a canonical Baseline Profile (`profile_idc = 66`) 4:2:0 8-bit SPS for `(width, height)`.
    pub fn new_baseline(width: u32, height: u32, fps: Rational64) -> Self {
        let mb_w = width.div_ceil(16);
        let mb_h = height.div_ceil(16);
        let coded_w = mb_w * 16;
        let coded_h = mb_h * 16;
        // For 4:2:0 frame_mbs_only=true, CropUnitX = 2, CropUnitY = 2
        let crop_right = (coded_w - width) / 2;
        let crop_bottom = (coded_h - height) / 2;

        Self {
            profile_idc: 66,
            constraint_set_flags: 0xC0, // constraint_set0_flag | constraint_set1_flag
            level_idc: 40,
            sps_id: 0,
            chroma_format_idc: 1,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            log2_max_frame_num: 4,
            pic_order_cnt_type: 0,
            log2_max_pic_order_cnt_lsb: 4,
            max_num_ref_frames: 1,
            pic_width_in_mbs: mb_w,
            pic_height_in_map_units: mb_h,
            frame_mbs_only_flag: true,
            crop_left: 0,
            crop_right,
            crop_top: 0,
            crop_bottom,
            sar: Rational64::ONE,
            fps: Some(fps),
            colorimetry: Colorimetry::bt709_limited(),
        }
    }

    /// Active cropped display width in pixels (`libavcodec/h264_ps.c`).
    #[inline]
    pub const fn width(&self) -> u32 {
        let crop_unit_x = match self.chroma_format_idc {
            0 | 3 => 1,
            _ => 2,
        };
        let coded = self.pic_width_in_mbs * 16;
        coded.saturating_sub((self.crop_left + self.crop_right) * crop_unit_x)
    }

    /// Active cropped display height in pixels.
    #[inline]
    pub const fn height(&self) -> u32 {
        let sub_height_c = match self.chroma_format_idc {
            1 => 2,
            _ => 1,
        };
        let frame_mbs = if self.frame_mbs_only_flag { 1 } else { 2 };
        let crop_unit_y = sub_height_c * frame_mbs;
        let coded = self.pic_height_in_map_units * 16 * frame_mbs;
        coded.saturating_sub((self.crop_top + self.crop_bottom) * crop_unit_y)
    }

    /// Human-readable profile name (`Baseline`, `Constrained Baseline`, `Main`, `High`, `High 10`, etc.).
    pub const fn profile_name(&self) -> &'static str {
        match self.profile_idc {
            66 => {
                if (self.constraint_set_flags & 0x40) != 0 {
                    "Constrained Baseline"
                } else {
                    "Baseline"
                }
            }
            77 => "Main",
            88 => "Extended",
            100 => "High",
            110 => "High 10",
            122 => "High 4:2:2",
            244 => "High 4:4:4 Predictive",
            _ => "Unknown",
        }
    }

    /// Parses an SPS from either a full NAL unit (`nal_unit_type = 7`) or unescaped RBSP bytes.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(RagdollError::BitstreamEndOfData);
        }
        let rbsp = if (data[0] & 0x1F) == 7 && (data[0] & 0x80) == 0 && data.len() > 3 {
            unescape_rbsp(&data[1..])
        } else {
            data.to_vec()
        };
        Self::parse_rbsp(&rbsp)
    }

    /// Parses an SPS from unescaped RBSP payload bytes (excluding the 1-byte NAL header).
    pub fn parse_rbsp(rbsp: &[u8]) -> Result<Self> {
        let mut r = BitReader::new(rbsp);
        let profile_idc = r.read_bits(8)? as u8;
        let constraint_set_flags = r.read_bits(8)? as u8;
        let level_idc = r.read_bits(8)? as u8;
        let sps_id = r.read_ue()? as u8;
        if sps_id >= 32 {
            return Err(RagdollError::InvalidSyntax("H.264 sps_id >= 32"));
        }

        let mut chroma_format_idc = 1u8;
        let mut bit_depth_luma = 8u8;
        let mut bit_depth_chroma = 8u8;

        if matches!(
            profile_idc,
            100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
        ) {
            chroma_format_idc = r.read_ue()? as u8;
            if chroma_format_idc == 3 {
                let _separate_colour_plane = r.read_bit()?;
            }
            bit_depth_luma = (r.read_ue()? as u8) + 8;
            bit_depth_chroma = (r.read_ue()? as u8) + 8;
            let _qpprime_y_zero_transform_bypass = r.read_bit()?;
            let seq_scaling_matrix_present = r.read_bit()?;
            if seq_scaling_matrix_present {
                let num_lists = if chroma_format_idc != 3 { 8 } else { 12 };
                for i in 0..num_lists {
                    let list_present = r.read_bit()?;
                    if list_present {
                        let size = if i < 6 { 16 } else { 64 };
                        let mut last_scale = 8i64;
                        let mut next_scale = 8i64;
                        for _ in 0..size {
                            if next_scale != 0 {
                                let delta_scale = r.read_se()?;
                                next_scale = (last_scale + delta_scale + 256) % 256;
                            }
                            last_scale = if next_scale == 0 {
                                last_scale
                            } else {
                                next_scale
                            };
                        }
                    }
                }
            }
        }

        let log2_max_frame_num = (r.read_ue()? as u8) + 4;
        let pic_order_cnt_type = r.read_ue()? as u8;
        let mut log2_max_pic_order_cnt_lsb = 4u8;
        if pic_order_cnt_type == 0 {
            log2_max_pic_order_cnt_lsb = (r.read_ue()? as u8) + 4;
        } else if pic_order_cnt_type == 1 {
            let _delta_pic_order_always_zero = r.read_bit()?;
            let _offset_for_non_ref_pic = r.read_se()?;
            let _offset_for_top_to_bottom = r.read_se()?;
            let cycle_len = r.read_ue()?;
            for _ in 0..cycle_len {
                let _ = r.read_se()?;
            }
        }

        let max_num_ref_frames = r.read_ue()? as u8;
        let _gaps_in_frame_num_allowed = r.read_bit()?;
        let pic_width_in_mbs = (r.read_ue()? as u32) + 1;
        let pic_height_in_map_units = (r.read_ue()? as u32) + 1;
        let frame_mbs_only_flag = r.read_bit()?;
        if !frame_mbs_only_flag {
            let _mb_adaptive_frame_field = r.read_bit()?;
        }
        let _direct_8x8_inference = r.read_bit()?;

        let mut crop_left = 0;
        let mut crop_right = 0;
        let mut crop_top = 0;
        let mut crop_bottom = 0;
        if r.read_bit()? {
            crop_left = r.read_ue()? as u32;
            crop_right = r.read_ue()? as u32;
            crop_top = r.read_ue()? as u32;
            crop_bottom = r.read_ue()? as u32;
        }

        let mut sar = Rational64::ONE;
        let mut fps = None;
        let mut colorimetry = Colorimetry::bt709_limited();

        if r.bits_remaining() > 0 && r.read_bit()? {
            // VUI parameters
            if r.read_bit()? {
                let aspect_ratio_idc = r.read_bits(8)? as u8;
                sar = match aspect_ratio_idc {
                    1 => Rational64::ONE,
                    2 => Rational64::new(12, 11),
                    3 => Rational64::new(10, 11),
                    4 => Rational64::new(16, 11),
                    5 => Rational64::new(40, 33),
                    6 => Rational64::new(24, 11),
                    7 => Rational64::new(20, 11),
                    8 => Rational64::new(32, 11),
                    9 => Rational64::new(80, 33),
                    10 => Rational64::new(18, 11),
                    11 => Rational64::new(15, 11),
                    12 => Rational64::new(64, 33),
                    13 => Rational64::new(160, 99),
                    14 => Rational64::new(4, 3),
                    15 => Rational64::new(3, 2),
                    16 => Rational64::new(2, 1),
                    255 => {
                        let w = r.read_bits(16)? as i64;
                        let h = r.read_bits(16)? as i64;
                        if w > 0 && h > 0 {
                            Rational64::new(w, h)
                        } else {
                            Rational64::ONE
                        }
                    }
                    _ => Rational64::ONE,
                };
            }
            if r.read_bit()? {
                let _overscan_appropriate = r.read_bit()?;
            }
            if r.read_bit()? {
                let _video_format = r.read_bits(3)?;
                let full_range = r.read_bit()?;
                colorimetry.range = if full_range {
                    ColorRange::Full
                } else {
                    ColorRange::Limited
                };
                if r.read_bit()? {
                    let prim = r.read_bits(8)? as u8;
                    let trc = r.read_bits(8)? as u8;
                    let mat = r.read_bits(8)? as u8;
                    colorimetry.primaries = ColorPrimaries::from_cicp(prim);
                    colorimetry.transfer = TransferCharacteristics::from_cicp(trc);
                    colorimetry.matrix = MatrixCoefficients::from_cicp(mat);
                }
            }
            if r.read_bit()? {
                let _chroma_loc_top = r.read_ue()?;
                let _chroma_loc_bottom = r.read_ue()?;
            }
            if r.bits_remaining() >= 66 && r.read_bit()? {
                let num_units_in_tick = r.read_bits(32)?;
                let time_scale = r.read_bits(32)?;
                let _fixed_frame_rate = r.read_bit()?;
                if num_units_in_tick > 0 && time_scale > 0 {
                    fps = Some(Rational64::new(
                        time_scale as i64,
                        (num_units_in_tick * 2) as i64,
                    ));
                }
            }
        }

        Ok(Self {
            profile_idc,
            constraint_set_flags,
            level_idc,
            sps_id,
            chroma_format_idc,
            bit_depth_luma,
            bit_depth_chroma,
            log2_max_frame_num,
            pic_order_cnt_type,
            log2_max_pic_order_cnt_lsb,
            max_num_ref_frames,
            pic_width_in_mbs,
            pic_height_in_map_units,
            frame_mbs_only_flag,
            crop_left,
            crop_right,
            crop_top,
            crop_bottom,
            sar,
            fps,
            colorimetry,
        })
    }

    /// Serializes this SPS into RBSP payload bytes (without the 1-byte NAL header).
    pub fn to_rbsp(&self) -> Vec<u8> {
        let mut w = BitWriter::with_capacity(64);
        w.write_bits(self.profile_idc as u64, 8);
        w.write_bits(self.constraint_set_flags as u64, 8);
        w.write_bits(self.level_idc as u64, 8);
        w.write_ue(self.sps_id as u64);

        if matches!(
            self.profile_idc,
            100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
        ) {
            w.write_ue(self.chroma_format_idc as u64);
            if self.chroma_format_idc == 3 {
                w.write_bit(false);
            }
            w.write_ue(self.bit_depth_luma.saturating_sub(8) as u64);
            w.write_ue(self.bit_depth_chroma.saturating_sub(8) as u64);
            w.write_bit(false); // qpprime_y_zero_transform_bypass_flag
            w.write_bit(false); // seq_scaling_matrix_present_flag
        }

        w.write_ue(self.log2_max_frame_num.saturating_sub(4) as u64);
        w.write_ue(self.pic_order_cnt_type as u64);
        if self.pic_order_cnt_type == 0 {
            w.write_ue(self.log2_max_pic_order_cnt_lsb.saturating_sub(4) as u64);
        }
        w.write_ue(self.max_num_ref_frames as u64);
        w.write_bit(false); // gaps_in_frame_num_value_allowed_flag
        w.write_ue(self.pic_width_in_mbs.saturating_sub(1) as u64);
        w.write_ue(self.pic_height_in_map_units.saturating_sub(1) as u64);
        w.write_bit(self.frame_mbs_only_flag);
        if !self.frame_mbs_only_flag {
            w.write_bit(false);
        }
        w.write_bit(true); // direct_8x8_inference_flag

        let has_crop = self.crop_left != 0
            || self.crop_right != 0
            || self.crop_top != 0
            || self.crop_bottom != 0;
        w.write_bit(has_crop);
        if has_crop {
            w.write_ue(self.crop_left as u64);
            w.write_ue(self.crop_right as u64);
            w.write_ue(self.crop_top as u64);
            w.write_ue(self.crop_bottom as u64);
        }

        // Write VUI with colorimetry and timing_info
        w.write_bit(true); // vui_parameters_present_flag
        w.write_bit(true); // aspect_ratio_info_present_flag
        if self.sar == Rational64::ONE {
            w.write_bits(1, 8);
        } else {
            w.write_bits(255, 8);
            w.write_bits(self.sar.num.clamp(1, 65535) as u64, 16);
            w.write_bits(self.sar.den.clamp(1, 65535) as u64, 16);
        }
        w.write_bit(false); // overscan_info_present_flag
        w.write_bit(true); // video_signal_type_present_flag
        w.write_bits(5, 3); // unspecified video_format
        w.write_bit(self.colorimetry.range == ColorRange::Full);
        w.write_bit(true); // colour_description_present_flag
        w.write_bits(self.colorimetry.primaries as u64, 8);
        w.write_bits(self.colorimetry.transfer as u64, 8);
        w.write_bits(self.colorimetry.matrix as u64, 8);
        w.write_bit(false); // chroma_loc_info_present_flag

        if let Some(fps) = self.fps
            && fps.num > 0
            && fps.den > 0
        {
            w.write_bit(true); // timing_info_present_flag
            w.write_bits(fps.den as u64, 32);
            w.write_bits((fps.num * 2) as u64, 32);
            w.write_bit(true); // fixed_frame_rate_flag
        } else {
            w.write_bit(false);
        }
        w.write_bit(false); // nal_hrd_parameters_present_flag
        w.write_bit(false); // vcl_hrd_parameters_present_flag
        w.write_bit(false); // pic_struct_present_flag
        w.write_bit(false); // bitstream_restriction_flag

        w.write_rbsp_trailing_bits();
        w.finish()
    }

    /// Serializes this SPS as a complete NAL unit (1-byte `0x67` header + escaped RBSP).
    pub fn to_nal_unit(&self) -> Vec<u8> {
        let rbsp = self.to_rbsp();
        let escaped = escape_rbsp(&rbsp);
        let mut out = Vec::with_capacity(1 + escaped.len());
        out.push(
            NalHeader {
                nal_ref_idc: 3,
                nal_unit_type: NalUnitType::Sps,
            }
            .to_byte(),
        );
        out.extend_from_slice(&escaped);
        out
    }
}

/// Parsed H.264 Picture Parameter Set (`PPS`, `nal_unit_type = 8`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pps {
    pub pps_id: u8,
    pub sps_id: u8,
    pub entropy_coding_mode_flag: bool,
    pub bottom_field_pic_order_in_frame_present_flag: bool,
    pub num_ref_idx_l0_default_active_minus1: u8,
    pub num_ref_idx_l1_default_active_minus1: u8,
    pub weighted_pred_flag: bool,
    pub weighted_bipred_idc: u8,
    pub pic_init_qp_minus26: i8,
    pub pic_init_qs_minus26: i8,
    pub chroma_qp_index_offset: i8,
    pub deblocking_filter_control_present_flag: bool,
    pub constrained_intra_pred_flag: bool,
    pub redundant_pic_cnt_present_flag: bool,
}

impl Pps {
    pub fn new_baseline(qp: u8) -> Self {
        Self {
            pps_id: 0,
            sps_id: 0,
            entropy_coding_mode_flag: false, // CAVLC
            bottom_field_pic_order_in_frame_present_flag: false,
            num_ref_idx_l0_default_active_minus1: 0,
            num_ref_idx_l1_default_active_minus1: 0,
            weighted_pred_flag: false,
            weighted_bipred_idc: 0,
            pic_init_qp_minus26: (qp.clamp(0, 51) as i8) - 26,
            pic_init_qs_minus26: 0,
            chroma_qp_index_offset: 0,
            deblocking_filter_control_present_flag: true,
            constrained_intra_pred_flag: false,
            redundant_pic_cnt_present_flag: false,
        }
    }

    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(RagdollError::BitstreamEndOfData);
        }
        let rbsp = if (data[0] & 0x1F) == 8 && (data[0] & 0x80) == 0 && data.len() > 2 {
            unescape_rbsp(&data[1..])
        } else {
            data.to_vec()
        };
        Self::parse_rbsp(&rbsp)
    }

    pub fn parse_rbsp(rbsp: &[u8]) -> Result<Self> {
        let mut r = BitReader::new(rbsp);
        let pps_id = r.read_ue()? as u8;
        let sps_id = r.read_ue()? as u8;
        let entropy_coding_mode_flag = r.read_bit()?;
        let bottom_field_pic_order_in_frame_present_flag = r.read_bit()?;
        let num_slice_groups_minus1 = r.read_ue()?;
        if num_slice_groups_minus1 > 0 {
            return Err(RagdollError::Unsupported(
                "H.264 Flexible Macroblock Ordering (FMO) is not supported".into(),
            ));
        }
        let num_ref_idx_l0_default_active_minus1 = r.read_ue()? as u8;
        let num_ref_idx_l1_default_active_minus1 = r.read_ue()? as u8;
        let weighted_pred_flag = r.read_bit()?;
        let weighted_bipred_idc = r.read_bits(2)? as u8;
        let pic_init_qp_minus26 = r.read_se()? as i8;
        let pic_init_qs_minus26 = r.read_se()? as i8;
        let chroma_qp_index_offset = r.read_se()? as i8;
        let deblocking_filter_control_present_flag = r.read_bit()?;
        let constrained_intra_pred_flag = r.read_bit()?;
        let redundant_pic_cnt_present_flag = r.read_bit()?;

        Ok(Self {
            pps_id,
            sps_id,
            entropy_coding_mode_flag,
            bottom_field_pic_order_in_frame_present_flag,
            num_ref_idx_l0_default_active_minus1,
            num_ref_idx_l1_default_active_minus1,
            weighted_pred_flag,
            weighted_bipred_idc,
            pic_init_qp_minus26,
            pic_init_qs_minus26,
            chroma_qp_index_offset,
            deblocking_filter_control_present_flag,
            constrained_intra_pred_flag,
            redundant_pic_cnt_present_flag,
        })
    }

    pub fn to_rbsp(&self) -> Vec<u8> {
        let mut w = BitWriter::with_capacity(16);
        w.write_ue(self.pps_id as u64);
        w.write_ue(self.sps_id as u64);
        w.write_bit(self.entropy_coding_mode_flag);
        w.write_bit(self.bottom_field_pic_order_in_frame_present_flag);
        w.write_ue(0); // num_slice_groups_minus1
        w.write_ue(self.num_ref_idx_l0_default_active_minus1 as u64);
        w.write_ue(self.num_ref_idx_l1_default_active_minus1 as u64);
        w.write_bit(self.weighted_pred_flag);
        w.write_bits(self.weighted_bipred_idc as u64, 2);
        w.write_se(self.pic_init_qp_minus26 as i64);
        w.write_se(self.pic_init_qs_minus26 as i64);
        w.write_se(self.chroma_qp_index_offset as i64);
        w.write_bit(self.deblocking_filter_control_present_flag);
        w.write_bit(self.constrained_intra_pred_flag);
        w.write_bit(self.redundant_pic_cnt_present_flag);
        w.write_rbsp_trailing_bits();
        w.finish()
    }

    pub fn to_nal_unit(&self) -> Vec<u8> {
        let rbsp = self.to_rbsp();
        let escaped = escape_rbsp(&rbsp);
        let mut out = Vec::with_capacity(1 + escaped.len());
        out.push(
            NalHeader {
                nal_ref_idc: 3,
                nal_unit_type: NalUnitType::Pps,
            }
            .to_byte(),
        );
        out.extend_from_slice(&escaped);
        out
    }
}

/// Parsed H.264 Slice Header (`nal_unit_type = 1` or `5`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SliceHeader {
    pub first_mb_in_slice: u32,
    pub slice_type: u8,
    pub pict_type: PictureType,
    pub pps_id: u8,
    pub frame_num: u16,
    pub idr_pic_id: Option<u16>,
    pub pic_order_cnt_lsb: u16,
    pub slice_qp_delta: i8,
    pub slice_qp: u8,
    pub disable_deblocking_filter_idc: u8,
}

impl SliceHeader {
    pub fn parse_from_reader(
        r: &mut BitReader<'_>,
        nal_header: NalHeader,
        sps: &Sps,
        pps: &Pps,
    ) -> Result<Self> {
        let first_mb_in_slice = r.read_ue()? as u32;
        let slice_type = r.read_ue()? as u8;
        let pict_type = match slice_type % 5 {
            0 | 3 => PictureType::P,
            1 => PictureType::B,
            2 | 4 => PictureType::I,
            _ => PictureType::Unknown,
        };
        let pps_id = r.read_ue()? as u8;
        let frame_num = r.read_bits(sps.log2_max_frame_num as u32)? as u16;

        let mut field_pic_flag = false;
        if !sps.frame_mbs_only_flag {
            field_pic_flag = r.read_bit()?;
            if field_pic_flag {
                let _bottom_field_flag = r.read_bit()?;
            }
        }

        let idr_pic_id = if nal_header.nal_unit_type == NalUnitType::IdrSlice {
            Some(r.read_ue()? as u16)
        } else {
            None
        };

        let mut pic_order_cnt_lsb = 0u16;
        if sps.pic_order_cnt_type == 0 {
            pic_order_cnt_lsb = r.read_bits(sps.log2_max_pic_order_cnt_lsb as u32)? as u16;
            if pps.bottom_field_pic_order_in_frame_present_flag && !field_pic_flag {
                let _delta_poc_bottom = r.read_se()?;
            }
        }

        if pict_type == PictureType::B {
            let _direct_spatial_mv_pred = r.read_bit()?;
        }

        if matches!(pict_type, PictureType::P | PictureType::B) {
            let num_ref_override = r.read_bit()?;
            if num_ref_override {
                let _l0_active = r.read_ue()?;
                if pict_type == PictureType::B {
                    let _l1_active = r.read_ue()?;
                }
            }
        }

        // ref_pic_list_modification
        if pict_type != PictureType::I {
            let mod_l0 = r.read_bit()?;
            if mod_l0 {
                loop {
                    let idc = r.read_ue()?;
                    if idc == 3 {
                        break;
                    }
                    let _ = r.read_ue()?;
                }
            }
            if pict_type == PictureType::B {
                let mod_l1 = r.read_bit()?;
                if mod_l1 {
                    loop {
                        let idc = r.read_ue()?;
                        if idc == 3 {
                            break;
                        }
                        let _ = r.read_ue()?;
                    }
                }
            }
        }

        // dec_ref_pic_marking
        if nal_header.nal_ref_idc != 0 {
            if nal_header.nal_unit_type == NalUnitType::IdrSlice {
                let _no_output_of_prior_pics = r.read_bit()?;
                let _long_term_reference = r.read_bit()?;
            } else {
                let adaptive_marking = r.read_bit()?;
                if adaptive_marking {
                    loop {
                        let mmco = r.read_ue()?;
                        if mmco == 0 {
                            break;
                        }
                        if matches!(mmco, 1..=6) {
                            let _ = r.read_ue()?;
                        }
                        if mmco == 3 {
                            let _ = r.read_ue()?;
                        }
                    }
                }
            }
        }

        if pps.entropy_coding_mode_flag && pict_type != PictureType::I {
            let _cabac_init_idc = r.read_ue()?;
        }

        let slice_qp_delta = r.read_se()? as i8;
        let slice_qp =
            (26i32 + (pps.pic_init_qp_minus26 as i32) + (slice_qp_delta as i32)).clamp(0, 51) as u8;

        let mut disable_deblocking_filter_idc = 0u8;
        if pps.deblocking_filter_control_present_flag {
            disable_deblocking_filter_idc = r.read_ue()? as u8;
            if disable_deblocking_filter_idc != 1 {
                let _alpha = r.read_se()?;
                let _beta = r.read_se()?;
            }
        }

        Ok(Self {
            first_mb_in_slice,
            slice_type,
            pict_type,
            pps_id,
            frame_num,
            idr_pic_id,
            pic_order_cnt_lsb,
            slice_qp_delta,
            slice_qp,
            disable_deblocking_filter_idc,
        })
    }

    pub fn write(&self, w: &mut BitWriter, nal_header: NalHeader, sps: &Sps, pps: &Pps) {
        w.write_ue(self.first_mb_in_slice as u64);
        w.write_ue(self.slice_type as u64);
        w.write_ue(self.pps_id as u64);
        w.write_bits(self.frame_num as u64, sps.log2_max_frame_num as u32);
        if nal_header.nal_unit_type == NalUnitType::IdrSlice {
            w.write_ue(self.idr_pic_id.unwrap_or(0) as u64);
        }
        if sps.pic_order_cnt_type == 0 {
            w.write_bits(
                self.pic_order_cnt_lsb as u64,
                sps.log2_max_pic_order_cnt_lsb as u32,
            );
        }
        if matches!(self.pict_type, PictureType::P | PictureType::B) {
            w.write_bit(false); // num_ref_idx_active_override_flag
            w.write_bit(false); // ref_pic_list_modification_flag_l0
        }
        if nal_header.nal_ref_idc != 0 {
            if nal_header.nal_unit_type == NalUnitType::IdrSlice {
                w.write_bit(false); // no_output_of_prior_pics_flag
                w.write_bit(false); // long_term_reference_flag
            } else {
                w.write_bit(false); // adaptive_ref_pic_marking_mode_flag
            }
        }
        w.write_se(self.slice_qp_delta as i64);
        if pps.deblocking_filter_control_present_flag {
            w.write_ue(self.disable_deblocking_filter_idc as u64);
            if self.disable_deblocking_filter_idc != 1 {
                w.write_se(0);
                w.write_se(0);
            }
        }
    }
}

/// Parsed H.264 Supplemental Enhancement Information (`SEI`, `nal_unit_type = 6`) message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeiMessage {
    pub payload_type: u32,
    pub payload: Vec<u8>,
}

impl SeiMessage {
    /// Parses all SEI messages from an unescaped SEI RBSP payload.
    pub fn parse_rbsp(rbsp: &[u8]) -> Vec<Self> {
        let mut msgs = Vec::new();
        let mut pos = 0usize;
        while pos + 2 <= rbsp.len() && rbsp[pos] != 0x80 {
            let mut payload_type = 0u32;
            while pos < rbsp.len() && rbsp[pos] == 0xFF {
                payload_type += 255;
                pos += 1;
            }
            if pos >= rbsp.len() {
                break;
            }
            payload_type += rbsp[pos] as u32;
            pos += 1;

            let mut payload_size = 0usize;
            while pos < rbsp.len() && rbsp[pos] == 0xFF {
                payload_size += 255;
                pos += 1;
            }
            if pos >= rbsp.len() {
                break;
            }
            payload_size += rbsp[pos] as usize;
            pos += 1;

            let end = (pos + payload_size).min(rbsp.len());
            msgs.push(Self {
                payload_type,
                payload: rbsp[pos..end].to_vec(),
            });
            pos = end;
        }
        msgs
    }
}

/// H.264 zig-zag scan order for $4\times4$ blocks (ITU-T H.264 Table 8-13).
pub const ZIGZAG_4X4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// H.264 dequantization scaling matrix `LevelScale4x4[qp % 6][pos_type]` (`libavcodec/h264data.h`).
pub const DEQUANT4_COEFF_INIT: [[i32; 3]; 6] = [
    [10, 16, 13],
    [11, 18, 14],
    [13, 20, 16],
    [14, 23, 18],
    [16, 25, 20],
    [18, 29, 23],
];

/// Forward quantization multiplier table `MF[qp % 6][pos_type]` (`2^15 / LevelScale4x4`).
pub const QUANT4_MF: [[i32; 3]; 6] = [
    [13107, 5243, 8066],
    [11916, 4660, 7490],
    [10082, 4194, 6554],
    [9362, 3647, 5825],
    [8192, 3355, 5243],
    [7282, 2893, 4559],
];

/// ITU-T H.264 Table 8-15 mapping luma $QP_Y$ (`0..=51`) to chroma $QP_C$.
pub const CHROMA_QP_TABLE: [u8; 52] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39,
    39, 39,
];

#[inline]
const fn coeff_pos_type(idx: usize) -> usize {
    let r = idx >> 2;
    let c = idx & 3;
    match (r & 1, c & 1) {
        (0, 0) => 0,
        (1, 1) => 1,
        _ => 2,
    }
}

/// Forward $4\times4$ H.264 quantization for a transformed $4\times4$ block at `qp` (`0..=51`).
pub fn quantize_block4x4(dct_coeffs: &[i16; 16], qp: u8, is_intra: bool) -> [i16; 16] {
    let qp = qp.min(51) as usize;
    let qbits = 15 + (qp / 6);
    let rem = qp % 6;
    let f = ((1i32 << qbits) / if is_intra { 3 } else { 6 }) as i64;
    let mut out = [0i16; 16];
    for i in 0..16 {
        let c = dct_coeffs[i] as i64;
        let sign = if c < 0 { -1 } else { 1 };
        let mf = QUANT4_MF[rem][coeff_pos_type(i)] as i64;
        let level = ((c.abs() * mf + f) >> qbits) * sign;
        out[i] = level.clamp(i16::MIN as i64, i16::MAX as i64) as i16;
    }
    out
}

/// Inverse $4\times4$ H.264 dequantization at `qp` (`0..=51`).
pub fn dequantize_block4x4(levels: &[i16; 16], qp: u8) -> [i16; 16] {
    let qp = qp.min(51) as usize;
    let shift = qp / 6;
    let rem = qp % 6;
    let mut out = [0i16; 16];
    for i in 0..16 {
        let v = DEQUANT4_COEFF_INIT[rem][coeff_pos_type(i)] << shift;
        out[i] = ((levels[i] as i32) * v).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
    out
}

/// Forward + inverse $4\times4$ Luma DC Hadamard transform + quantization at `qp`.
pub fn quantize_luma_dc4x4(dc_block: &[i16; 16], qp: u8) -> [i16; 16] {
    let had = hadamard4x4(dc_block);
    let qp = qp.min(51) as usize;
    let qbits = 16 + (qp / 6);
    let mf = QUANT4_MF[qp % 6][0] as i64;
    let f = (1i64 << qbits) / 3;
    let mut out = [0i16; 16];
    for i in 0..16 {
        // Note:ragdoll_simd::hadamard4x4 divides by 2 (`(a + d) >> 1`), so scale by 2 to match raw H.264 Hadamard
        let c = (had[i] as i64) * 2;
        let sign = if c < 0 { -1 } else { 1 };
        let level = ((c.abs() * mf + f) >> qbits) * sign;
        out[i] = level.clamp(i16::MIN as i64, i16::MAX as i64) as i16;
    }
    out
}

/// Inverse $4\times4$ Luma DC Hadamard + dequantization (`libavcodec/h264idct_template.c:ff_h264_luma_dc_dequant_idct`).
pub fn dequantize_idct_luma_dc4x4(levels: &[i16; 16], qp: u8) -> [i16; 16] {
    let qp = qp.min(51) as usize;
    let qmul = (DEQUANT4_COEFF_INIT[qp % 6][0] * 16) << ((qp / 6) + 2);
    let mut temp = [0i32; 16];
    for i in 0..4 {
        let z0 = (levels[4 * i] as i32) + (levels[4 * i + 1] as i32);
        let z1 = (levels[4 * i] as i32) - (levels[4 * i + 1] as i32);
        let z2 = (levels[4 * i + 2] as i32) - (levels[4 * i + 3] as i32);
        let z3 = (levels[4 * i + 2] as i32) + (levels[4 * i + 3] as i32);
        temp[4 * i] = z0 + z3;
        temp[4 * i + 1] = z0 - z3;
        temp[4 * i + 2] = z1 - z2;
        temp[4 * i + 3] = z1 + z2;
    }
    let mut out = [0i16; 16];
    for i in 0..4 {
        let z0 = temp[i] + temp[8 + i];
        let z1 = temp[i] - temp[8 + i];
        let z2 = temp[4 + i] - temp[12 + i];
        let z3 = temp[4 + i] + temp[12 + i];
        out[i] = (((z0 + z3) * qmul + 128) >> 8) as i16;
        out[4 + i] = (((z1 + z2) * qmul + 128) >> 8) as i16;
        out[8 + i] = (((z1 - z2) * qmul + 128) >> 8) as i16;
        out[12 + i] = (((z0 - z3) * qmul + 128) >> 8) as i16;
    }
    out
}

/// Inverse $2\times2$ Chroma DC Hadamard + dequantization (`ff_h264_chroma_dc_dequant_idct`).
pub fn dequantize_idct_chroma_dc2x2(levels: &[i16; 4], qpc: u8) -> [i16; 4] {
    let qpc = qpc.min(51) as usize;
    let qmul = (DEQUANT4_COEFF_INIT[qpc % 6][0] * 16) << ((qpc / 6) + 2);
    let a = levels[0] as i32;
    let b = levels[1] as i32;
    let c = levels[2] as i32;
    let d = levels[3] as i32;
    let e = a - b;
    let apb = a + b;
    let cmd = c - d;
    let cpd = c + d;
    [
        (((apb + cpd) * qmul) >> 7) as i16,
        (((e + cmd) * qmul) >> 7) as i16,
        (((apb - cpd) * qmul) >> 7) as i16,
        (((e - cmd) * qmul) >> 7) as i16,
    ]
}

/// H.264 Intra $16\times16$ Luma spatial prediction modes (`0` = Vertical, `1` = Horizontal, `2` = DC, `3` = Plane).
///
/// Reference-hardened against `libavcodec/aarch64/h264pred_neon.S` (`ff_pred16x16_{vert,hor,dc,plane}_neon`).
#[inline]
pub fn predict_intra_16x16(
    mode: u8,
    top: Option<&[u8; 16]>,
    left: Option<&[u8; 16]>,
    top_left: u8,
    dst: &mut [u8],
    stride: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if dst.len() >= 15 * stride + 16 {
            unsafe { predict_intra_16x16_neon(mode, top, left, top_left, dst, stride) };
            return;
        }
    }
    predict_intra_16x16_scalar(mode, top, left, top_left, dst, stride);
}

pub fn predict_intra_16x16_scalar(
    mode: u8,
    top: Option<&[u8; 16]>,
    left: Option<&[u8; 16]>,
    top_left: u8,
    dst: &mut [u8],
    stride: usize,
) {
    match mode {
        0 => {
            // Vertical
            let row_src = top.copied().unwrap_or([128u8; 16]);
            for y in 0..16 {
                dst[y * stride..y * stride + 16].copy_from_slice(&row_src);
            }
        }
        1 => {
            // Horizontal
            let col_src = left.copied().unwrap_or([128u8; 16]);
            for y in 0..16 {
                dst[y * stride..y * stride + 16].fill(col_src[y]);
            }
        }
        2 => {
            // DC (`ff_h264_pred16x16_dc_8_c`)
            let dc = match (top, left) {
                (Some(t), Some(l)) => {
                    let sum: u32 = t.iter().chain(l.iter()).map(|&v| v as u32).sum();
                    ((sum + 16) >> 5) as u8
                }
                (Some(t), None) => {
                    let sum: u32 = t.iter().map(|&v| v as u32).sum();
                    ((sum + 8) >> 4) as u8
                }
                (None, Some(l)) => {
                    let sum: u32 = l.iter().map(|&v| v as u32).sum();
                    ((sum + 8) >> 4) as u8
                }
                (None, None) => 128u8,
            };
            for y in 0..16 {
                dst[y * stride..y * stride + 16].fill(dc);
            }
        }
        _ => {
            // Plane (`ff_h264_pred16x16_plane_8_c`)
            let t = top.copied().unwrap_or([128u8; 16]);
            let l = left.copied().unwrap_or([128u8; 16]);
            let mut h = 0i32;
            let mut v = 0i32;
            for i in 0..8 {
                let p_top_left = if i == 7 {
                    top_left as i32
                } else {
                    t[6 - i] as i32
                };
                let p_left_top = if i == 7 {
                    top_left as i32
                } else {
                    l[6 - i] as i32
                };
                h += ((i + 1) as i32) * ((t[8 + i] as i32) - p_top_left);
                v += ((i + 1) as i32) * ((l[8 + i] as i32) - p_left_top);
            }
            let a = 16 * ((t[15] as i32) + (l[15] as i32));
            let b = (5 * h + 32) >> 6;
            let c = (5 * v + 32) >> 6;
            for y in 0..16 {
                for x in 0..16 {
                    let val = (a + b * (x as i32 - 7) + c * (y as i32 - 7) + 16) >> 5;
                    dst[y * stride + x] = clamp_u8(val);
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn predict_intra_16x16_neon(
    mode: u8,
    top: Option<&[u8; 16]>,
    left: Option<&[u8; 16]>,
    top_left: u8,
    dst: &mut [u8],
    stride: usize,
) {
    use std::arch::aarch64::*;
    let dst_ptr = dst.as_mut_ptr();
    match mode {
        0 => {
            let v_row = match top {
                Some(t) => unsafe { vld1q_u8(t.as_ptr()) },
                None => vdupq_n_u8(128),
            };
            for y in 0..16 {
                unsafe { vst1q_u8(dst_ptr.add(y * stride), v_row) };
            }
        }
        1 => {
            let col_src = left.copied().unwrap_or([128u8; 16]);
            for (y, &val) in col_src.iter().enumerate() {
                let v_row = vdupq_n_u8(val);
                unsafe { vst1q_u8(dst_ptr.add(y * stride), v_row) };
            }
        }
        2 => {
            let dc = match (top, left) {
                (Some(t), Some(l)) => {
                    let vt = unsafe { vld1q_u8(t.as_ptr()) };
                    let vl = unsafe { vld1q_u8(l.as_ptr()) };
                    let sum = (vaddlvq_u8(vt) as u32) + (vaddlvq_u8(vl) as u32);
                    ((sum + 16) >> 5) as u8
                }
                (Some(t), None) => {
                    let vt = unsafe { vld1q_u8(t.as_ptr()) };
                    (((vaddlvq_u8(vt) as u32) + 8) >> 4) as u8
                }
                (None, Some(l)) => {
                    let vl = unsafe { vld1q_u8(l.as_ptr()) };
                    (((vaddlvq_u8(vl) as u32) + 8) >> 4) as u8
                }
                (None, None) => 128u8,
            };
            let v_dc = vdupq_n_u8(dc);
            for y in 0..16 {
                unsafe { vst1q_u8(dst_ptr.add(y * stride), v_dc) };
            }
        }
        _ => {
            let t = top.copied().unwrap_or([128u8; 16]);
            let l = left.copied().unwrap_or([128u8; 16]);
            let mut h = 0i32;
            let mut v = 0i32;
            for i in 0..8 {
                let p_top_left = if i == 7 {
                    top_left as i32
                } else {
                    t[6 - i] as i32
                };
                let p_left_top = if i == 7 {
                    top_left as i32
                } else {
                    l[6 - i] as i32
                };
                h += ((i + 1) as i32) * ((t[8 + i] as i32) - p_top_left);
                v += ((i + 1) as i32) * ((l[8 + i] as i32) - p_left_top);
            }
            let a = 16 * ((t[15] as i32) + (l[15] as i32));
            let b = ((5 * h + 32) >> 6) as i16;
            let c = ((5 * v + 32) >> 6) as i16;
            let base = (a - 7 * (b as i32) - 7 * (c as i32) + 16) as i16;

            let ramp_lo: [i16; 8] = [
                base,
                base.wrapping_add(b),
                base.wrapping_add(b * 2),
                base.wrapping_add(b * 3),
                base.wrapping_add(b * 4),
                base.wrapping_add(b * 5),
                base.wrapping_add(b * 6),
                base.wrapping_add(b * 7),
            ];
            let mut v_lo = unsafe { vld1q_s16(ramp_lo.as_ptr()) };
            let mut v_hi = vaddq_s16(v_lo, vdupq_n_s16(b.wrapping_mul(8)));
            let v_c = vdupq_n_s16(c);

            for y in 0..16 {
                let out_lo = vqshrun_n_s16::<5>(v_lo);
                let out_hi = vqshrun_n_s16::<5>(v_hi);
                unsafe { vst1q_u8(dst_ptr.add(y * stride), vcombine_u8(out_lo, out_hi)) };
                v_lo = vaddq_s16(v_lo, v_c);
                v_hi = vaddq_s16(v_hi, v_c);
            }
        }
    }
}

/// H.264 Intra $8\times8$ Chroma spatial prediction (`0` = DC, `1` = Horizontal, `2` = Vertical, `3` = Plane).
#[inline]
pub fn predict_intra_chroma_8x8(
    mode: u8,
    top: Option<&[u8; 8]>,
    left: Option<&[u8; 8]>,
    top_left: u8,
    dst: &mut [u8],
    stride: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if mode == 3 && dst.len() >= 7 * stride + 8 {
            unsafe { predict_intra_chroma_8x8_plane_neon(top, left, top_left, dst, stride) };
            return;
        }
    }
    predict_intra_chroma_8x8_scalar(mode, top, left, top_left, dst, stride);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn predict_intra_chroma_8x8_plane_neon(
    top: Option<&[u8; 8]>,
    left: Option<&[u8; 8]>,
    top_left: u8,
    dst: &mut [u8],
    stride: usize,
) {
    use std::arch::aarch64::*;
    let t = top.copied().unwrap_or([128u8; 8]);
    let l = left.copied().unwrap_or([128u8; 8]);
    let mut h = 0i32;
    let mut v = 0i32;
    for i in 0..4 {
        let p_tl = if i == 3 {
            top_left as i32
        } else {
            t[2 - i] as i32
        };
        let p_lt = if i == 3 {
            top_left as i32
        } else {
            l[2 - i] as i32
        };
        h += ((i + 1) as i32) * ((t[4 + i] as i32) - p_tl);
        v += ((i + 1) as i32) * ((l[4 + i] as i32) - p_lt);
    }
    let a = 16 * ((t[7] as i32) + (l[7] as i32));
    let b = ((17 * h + 16) >> 5) as i16;
    let c = ((17 * v + 16) >> 5) as i16;
    let base = (a - 3 * (b as i32) - 3 * (c as i32) + 16) as i16;

    let ramp: [i16; 8] = [
        base,
        base.wrapping_add(b),
        base.wrapping_add(b * 2),
        base.wrapping_add(b * 3),
        base.wrapping_add(b * 4),
        base.wrapping_add(b * 5),
        base.wrapping_add(b * 6),
        base.wrapping_add(b * 7),
    ];
    let mut v_row = unsafe { vld1q_s16(ramp.as_ptr()) };
    let v_c = vdupq_n_s16(c);
    let dst_ptr = dst.as_mut_ptr();

    for y in 0..8 {
        let out8 = vqshrun_n_s16::<5>(v_row);
        unsafe { vst1_u8(dst_ptr.add(y * stride), out8) };
        v_row = vaddq_s16(v_row, v_c);
    }
}

pub fn predict_intra_chroma_8x8_scalar(
    mode: u8,
    top: Option<&[u8; 8]>,
    left: Option<&[u8; 8]>,
    top_left: u8,
    dst: &mut [u8],
    stride: usize,
) {
    match mode {
        0 => {
            // 4x4 sub-block DC prediction per ITU-T H.264 8.3.4.1 (`pred8x8_dc_8_c`)
            let (dc00, dc01, dc10, dc11) = match (top, left) {
                (Some(t), Some(l)) => {
                    let s_t0: u32 = t[0..4].iter().map(|&v| v as u32).sum();
                    let s_t1: u32 = t[4..8].iter().map(|&v| v as u32).sum();
                    let s_l0: u32 = l[0..4].iter().map(|&v| v as u32).sum();
                    let s_l1: u32 = l[4..8].iter().map(|&v| v as u32).sum();
                    (
                        ((s_t0 + s_l0 + 4) >> 3) as u8,
                        ((s_t1 + 2) >> 2) as u8,
                        ((s_l1 + 2) >> 2) as u8,
                        ((s_t1 + s_l1 + 4) >> 3) as u8,
                    )
                }
                (Some(t), None) => {
                    let s_t0 = ((t[0..4].iter().map(|&v| v as u32).sum::<u32>() + 2) >> 2) as u8;
                    let s_t1 = ((t[4..8].iter().map(|&v| v as u32).sum::<u32>() + 2) >> 2) as u8;
                    (s_t0, s_t1, s_t0, s_t1)
                }
                (None, Some(l)) => {
                    let s_l0 = ((l[0..4].iter().map(|&v| v as u32).sum::<u32>() + 2) >> 2) as u8;
                    let s_l1 = ((l[4..8].iter().map(|&v| v as u32).sum::<u32>() + 2) >> 2) as u8;
                    (s_l0, s_l0, s_l1, s_l1)
                }
                (None, None) => (128, 128, 128, 128),
            };
            for y in 0..4 {
                dst[y * stride..y * stride + 4].fill(dc00);
                dst[y * stride + 4..y * stride + 8].fill(dc01);
            }
            for y in 4..8 {
                dst[y * stride..y * stride + 4].fill(dc10);
                dst[y * stride + 4..y * stride + 8].fill(dc11);
            }
        }
        1 => {
            // Horizontal
            let l = left.copied().unwrap_or([128u8; 8]);
            for y in 0..8 {
                dst[y * stride..y * stride + 8].fill(l[y]);
            }
        }
        2 => {
            // Vertical
            let t = top.copied().unwrap_or([128u8; 8]);
            for y in 0..8 {
                dst[y * stride..y * stride + 8].copy_from_slice(&t);
            }
        }
        _ => {
            // Plane
            let t = top.copied().unwrap_or([128u8; 8]);
            let l = left.copied().unwrap_or([128u8; 8]);
            let mut h = 0i32;
            let mut v = 0i32;
            for i in 0..4 {
                let p_tl = if i == 3 {
                    top_left as i32
                } else {
                    t[2 - i] as i32
                };
                let p_lt = if i == 3 {
                    top_left as i32
                } else {
                    l[2 - i] as i32
                };
                h += ((i + 1) as i32) * ((t[4 + i] as i32) - p_tl);
                v += ((i + 1) as i32) * ((l[4 + i] as i32) - p_lt);
            }
            let a = 16 * ((t[7] as i32) + (l[7] as i32));
            let b = (17 * h + 16) >> 5;
            let c = (17 * v + 16) >> 5;
            for y in 0..8 {
                for x in 0..8 {
                    let val = (a + b * (x as i32 - 3) + c * (y as i32 - 3) + 16) >> 5;
                    dst[y * stride + x] = clamp_u8(val);
                }
            }
        }
    }
}

/// CAVLC `coeff_token` code lengths for the 4 `nC` categories (`0..2`, `2..4`, `4..8`, `>=8`)
/// from `libavcodec/h264_cavlc.c`.
const COEFF_TOKEN_LEN: [[u8; 68]; 4] = [
    [
        1, 0, 0, 0, 6, 2, 0, 0, 8, 6, 3, 0, 9, 8, 7, 5, 10, 9, 8, 6, 11, 10, 9, 7, 13, 11, 10, 8,
        13, 13, 11, 9, 13, 13, 13, 10, 14, 14, 13, 11, 14, 14, 14, 13, 15, 15, 14, 14, 15, 15, 15,
        14, 16, 15, 15, 15, 16, 16, 16, 15, 16, 16, 16, 16, 16, 16, 16, 16,
    ],
    [
        2, 0, 0, 0, 6, 2, 0, 0, 6, 5, 3, 0, 7, 6, 6, 4, 8, 6, 6, 4, 8, 7, 7, 5, 9, 8, 8, 6, 11, 9,
        9, 6, 11, 11, 11, 7, 12, 11, 11, 9, 12, 12, 12, 11, 12, 12, 12, 11, 13, 13, 13, 12, 13, 13,
        13, 13, 13, 14, 13, 13, 14, 14, 14, 13, 14, 14, 14, 14,
    ],
    [
        4, 0, 0, 0, 6, 4, 0, 0, 6, 5, 4, 0, 6, 5, 5, 4, 7, 5, 5, 4, 7, 5, 5, 4, 7, 6, 6, 4, 7, 6,
        6, 4, 8, 7, 7, 5, 8, 8, 7, 6, 9, 8, 8, 7, 9, 9, 8, 8, 9, 9, 9, 8, 10, 9, 9, 9, 10, 10, 10,
        10, 10, 10, 10, 10, 10, 10, 10, 10,
    ],
    [
        6, 0, 0, 0, 6, 6, 0, 0, 6, 6, 6, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6,
    ],
];

/// CAVLC `coeff_token` code bits for the 4 `nC` categories (`libavcodec/h264_cavlc.c`).
const COEFF_TOKEN_BITS: [[u8; 68]; 4] = [
    [
        1, 0, 0, 0, 5, 1, 0, 0, 7, 4, 1, 0, 7, 6, 5, 3, 7, 6, 5, 3, 7, 6, 5, 4, 15, 6, 5, 4, 11,
        14, 5, 4, 8, 10, 13, 4, 15, 14, 9, 4, 11, 10, 13, 12, 15, 14, 9, 12, 11, 10, 13, 8, 15, 1,
        9, 12, 11, 14, 13, 8, 7, 10, 9, 12, 4, 6, 5, 8,
    ],
    [
        3, 0, 0, 0, 11, 2, 0, 0, 7, 7, 3, 0, 7, 10, 9, 5, 7, 6, 5, 4, 4, 6, 5, 6, 7, 6, 5, 8, 15,
        6, 5, 4, 11, 14, 13, 4, 15, 10, 9, 4, 11, 14, 13, 12, 8, 10, 9, 8, 15, 14, 13, 12, 11, 10,
        9, 12, 7, 11, 6, 8, 9, 8, 10, 1, 7, 6, 5, 4,
    ],
    [
        15, 0, 0, 0, 15, 14, 0, 0, 11, 15, 13, 0, 8, 12, 14, 12, 15, 10, 11, 11, 11, 8, 9, 10, 9,
        14, 13, 9, 8, 10, 9, 8, 15, 14, 13, 13, 11, 14, 10, 12, 15, 10, 13, 12, 11, 14, 9, 12, 8,
        10, 13, 8, 13, 7, 9, 12, 9, 12, 11, 10, 5, 8, 7, 6, 1, 4, 3, 2,
    ],
    [
        3, 0, 0, 0, 0, 1, 0, 0, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
        45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    ],
];

#[inline]
const fn nc_to_table_idx(nc: u8) -> usize {
    match nc {
        0 | 1 => 0,
        2 | 3 => 1,
        4..=7 => 2,
        _ => 3,
    }
}

#[inline]
fn predict_mb_nc(mb_nnz: &[u8], mb_w: usize, mb_x: usize, mb_y: usize) -> u8 {
    let left = if mb_x > 0 {
        Some(mb_nnz[mb_y * mb_w + (mb_x - 1)] as u16)
    } else {
        None
    };
    let top = if mb_y > 0 {
        Some(mb_nnz[(mb_y - 1) * mb_w + mb_x] as u16)
    } else {
        None
    };
    match (left, top) {
        (Some(l), Some(t)) => ((l + t + 1) >> 1) as u8,
        (Some(l), None) => l as u8,
        (None, Some(t)) => t as u8,
        (None, None) => 0,
    }
}

/// CAVLC `chroma_dc_coeff_token` code lengths (`4 * 5` entries).
const CHROMA_DC_COEFF_TOKEN_LEN: [u8; 20] =
    [2, 0, 0, 0, 6, 1, 0, 0, 6, 6, 3, 0, 6, 7, 7, 6, 6, 8, 8, 7];

/// CAVLC `chroma_dc_coeff_token` code bits (`4 * 5` entries).
const CHROMA_DC_COEFF_TOKEN_BITS: [u8; 20] =
    [1, 0, 0, 0, 7, 1, 0, 0, 4, 6, 1, 0, 3, 3, 2, 5, 2, 3, 2, 0];

/// CAVLC `total_zeros` code lengths for 16-coefficient blocks (`total_zeros_len[total_coeff - 1][zeros_left]`).
const TOTAL_ZEROS_LEN: [[u8; 16]; 15] = [
    [1, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 9],
    [3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 0],
    [4, 3, 3, 3, 4, 4, 3, 3, 4, 5, 5, 6, 5, 6, 0, 0],
    [5, 3, 4, 4, 3, 3, 3, 4, 3, 4, 5, 5, 5, 0, 0, 0],
    [4, 4, 4, 3, 3, 3, 3, 3, 4, 5, 4, 5, 0, 0, 0, 0],
    [6, 5, 3, 3, 3, 3, 3, 3, 4, 3, 6, 0, 0, 0, 0, 0],
    [6, 5, 3, 3, 3, 2, 3, 4, 3, 6, 0, 0, 0, 0, 0, 0],
    [6, 4, 5, 3, 2, 2, 3, 3, 6, 0, 0, 0, 0, 0, 0, 0],
    [6, 6, 4, 2, 2, 3, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    [5, 5, 3, 2, 2, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 4, 3, 3, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 4, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
];

const TOTAL_ZEROS_BITS: [[u8; 16]; 15] = [
    [1, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 1],
    [7, 6, 5, 4, 3, 5, 4, 3, 2, 3, 2, 3, 2, 1, 0, 0],
    [5, 7, 6, 5, 4, 3, 4, 3, 2, 3, 2, 1, 1, 0, 0, 0],
    [3, 7, 5, 4, 6, 5, 4, 3, 3, 2, 2, 1, 0, 0, 0, 0],
    [5, 4, 3, 7, 6, 5, 4, 3, 2, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 7, 6, 5, 4, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 5, 4, 3, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 3, 3, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 3, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
];

const CHROMA_DC_TOTAL_ZEROS_LEN: [[u8; 4]; 3] = [[1, 2, 3, 3], [1, 2, 2, 0], [1, 1, 0, 0]];

const CHROMA_DC_TOTAL_ZEROS_BITS: [[u8; 4]; 3] = [[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]];

const RUN_LEN: [[u8; 16]; 7] = [
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0],
];

const RUN_BITS: [[u8; 16]; 7] = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 0, 1, 3, 2, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
];

/// Encodes a 16-coefficient (or 4-coefficient Chroma DC) zig-zag scanned block using H.264 CAVLC (`nc` or `is_chroma_dc = true`).
pub fn encode_cavlc_block(w: &mut BitWriter, scanned_coeffs: &[i16], nc: u8, is_chroma_dc: bool) {
    let max_coeff = scanned_coeffs.len();
    let tbl = nc_to_table_idx(nc);
    // Find last non-zero index
    let mut total_coeff = 0usize;
    let mut levels = Vec::new();
    let mut runs = Vec::new();
    let mut total_zeros = 0usize;

    let last_nz = scanned_coeffs.iter().rposition(|&c| c != 0);
    let Some(last_idx) = last_nz else {
        // TotalCoeffs = 0, TrailingOnes = 0
        if is_chroma_dc {
            w.write_bits(
                CHROMA_DC_COEFF_TOKEN_BITS[0] as u64,
                CHROMA_DC_COEFF_TOKEN_LEN[0] as u32,
            );
        } else {
            w.write_bits(
                COEFF_TOKEN_BITS[tbl][0] as u64,
                COEFF_TOKEN_LEN[tbl][0] as u32,
            );
        }
        return;
    };

    let mut cur_run = 0usize;
    for i in (0..=last_idx).rev() {
        let c = scanned_coeffs[i];
        if c != 0 {
            if !levels.is_empty() {
                runs.push(cur_run);
                total_zeros += cur_run;
            }
            levels.push(c);
            total_coeff += 1;
            cur_run = 0;
        } else {
            cur_run += 1;
        }
    }
    runs.push(cur_run);
    total_zeros += cur_run;

    let mut trailing_ones = 0usize;
    for &lv in levels.iter().take(3) {
        if lv.abs() == 1 {
            trailing_ones += 1;
        } else {
            break;
        }
    }

    let token_idx = total_coeff * 4 + trailing_ones;
    if is_chroma_dc {
        w.write_bits(
            CHROMA_DC_COEFF_TOKEN_BITS[token_idx] as u64,
            CHROMA_DC_COEFF_TOKEN_LEN[token_idx] as u32,
        );
    } else {
        w.write_bits(
            COEFF_TOKEN_BITS[tbl][token_idx] as u64,
            COEFF_TOKEN_LEN[tbl][token_idx] as u32,
        );
    }

    // Trailing ones sign bits (0 = +1, 1 = -1)
    for &lv in levels.iter().take(trailing_ones) {
        w.write_bit(lv < 0);
    }

    // Remaining non-zero levels
    let mut suffix_length = if total_coeff > 10 && trailing_ones < 3 {
        1u32
    } else {
        0u32
    };
    for (idx, &lv) in levels.iter().enumerate().skip(trailing_ones) {
        let mut level_code = if lv > 0 {
            ((lv as i32) << 1) - 2
        } else {
            (-(lv as i32) << 1) - 1
        };
        if idx == trailing_ones && trailing_ones < 3 {
            level_code -= 2;
        }
        let level_prefix = (level_code as u32) >> suffix_length;
        if level_prefix < 14 || (level_prefix == 14 && suffix_length > 0) {
            w.write_bits(0, level_prefix);
            w.write_bit(true);
            if suffix_length > 0 {
                let level_suffix = (level_code as u32) & ((1 << suffix_length) - 1);
                w.write_bits(level_suffix as u64, suffix_length);
            }
        } else if level_prefix == 14 && suffix_length == 0 {
            w.write_bits(0, 14);
            w.write_bit(true);
            w.write_bits((level_code - 14) as u64, 4);
        } else {
            w.write_bits(0, 15);
            w.write_bit(true);
            let esc_suffix = (level_code as u32) - (15 << suffix_length);
            w.write_bits(esc_suffix as u64, 12);
        }

        if suffix_length == 0 {
            suffix_length = 1;
        }
        let threshold = 3i32 << (suffix_length - 1);
        if (lv.abs() as i32) > threshold && suffix_length < 6 {
            suffix_length += 1;
        }
    }

    // Total zeros
    if total_coeff < max_coeff {
        if is_chroma_dc {
            let l = CHROMA_DC_TOTAL_ZEROS_LEN[total_coeff - 1][total_zeros];
            let b = CHROMA_DC_TOTAL_ZEROS_BITS[total_coeff - 1][total_zeros];
            w.write_bits(b as u64, l as u32);
        } else {
            let l = TOTAL_ZEROS_LEN[total_coeff - 1][total_zeros];
            let b = TOTAL_ZEROS_BITS[total_coeff - 1][total_zeros];
            w.write_bits(b as u64, l as u32);
        }
    }

    // run_before for each coefficient except the last
    let mut zeros_left = total_zeros;
    for &run in runs.iter().take(total_coeff.saturating_sub(1)) {
        if zeros_left == 0 {
            break;
        }
        let vlc_idx = (zeros_left - 1).min(6);
        let l = RUN_LEN[vlc_idx][run];
        let b = RUN_BITS[vlc_idx][run];
        w.write_bits(b as u64, l as u32);
        zeros_left -= run;
    }
}

/// Decodes a CAVLC-coded block into `dst_scanned` (`max_coeff = dst_scanned.len()`, `16` or `4`).
pub fn decode_cavlc_block(
    r: &mut BitReader<'_>,
    dst_scanned: &mut [i16],
    nc: u8,
    is_chroma_dc: bool,
) -> Result<usize> {
    dst_scanned.fill(0);
    let max_coeff = dst_scanned.len();
    let tbl = nc_to_table_idx(nc);

    // Match coeff_token prefix
    let (total_coeff, trailing_ones) = if is_chroma_dc {
        match_vlc_pair(
            r,
            &CHROMA_DC_COEFF_TOKEN_LEN,
            &CHROMA_DC_COEFF_TOKEN_BITS,
            20,
        )?
    } else {
        match_vlc_pair(r, &COEFF_TOKEN_LEN[tbl], &COEFF_TOKEN_BITS[tbl], 68)?
    };

    if total_coeff == 0 {
        return Ok(0);
    }

    let mut levels = [0i16; 16];
    for lv in levels.iter_mut().take(trailing_ones) {
        let sign = r.read_bit()?;
        *lv = if sign { -1 } else { 1 };
    }

    let mut suffix_length = if total_coeff > 10 && trailing_ones < 3 {
        1u32
    } else {
        0u32
    };
    for (i, lv_slot) in levels
        .iter_mut()
        .enumerate()
        .take(total_coeff)
        .skip(trailing_ones)
    {
        let level_prefix = r.read_unary_zeros()? as u32;
        let mut level_code = if level_prefix < 14 || (level_prefix == 14 && suffix_length > 0) {
            let suffix = if suffix_length > 0 {
                r.read_bits(suffix_length)? as u32
            } else {
                0
            };
            (level_prefix << suffix_length) + suffix
        } else if level_prefix == 14 && suffix_length == 0 {
            14 + (r.read_bits(4)? as u32)
        } else {
            (15 << suffix_length) + (r.read_bits(12)? as u32)
        } as i32;

        if i == trailing_ones && trailing_ones < 3 {
            level_code += 2;
        }
        let lv = if (level_code & 1) == 0 {
            (level_code + 2) >> 1
        } else {
            -(level_code + 1) >> 1
        };
        *lv_slot = lv as i16;

        if suffix_length == 0 {
            suffix_length = 1;
        }
        let threshold = 3i32 << (suffix_length - 1);
        if lv.abs() > threshold && suffix_length < 6 {
            suffix_length += 1;
        }
    }

    let mut zeros_left = if total_coeff == max_coeff {
        0usize
    } else if is_chroma_dc {
        match_vlc_index(
            r,
            &CHROMA_DC_TOTAL_ZEROS_LEN[total_coeff - 1],
            &CHROMA_DC_TOTAL_ZEROS_BITS[total_coeff - 1],
            max_coeff - total_coeff + 1,
        )?
    } else {
        match_vlc_index(
            r,
            &TOTAL_ZEROS_LEN[total_coeff - 1],
            &TOTAL_ZEROS_BITS[total_coeff - 1],
            max_coeff - total_coeff + 1,
        )?
    };

    let mut pos = total_coeff + zeros_left - 1;
    if pos >= max_coeff {
        return Err(RagdollError::InvalidSyntax(
            "CAVLC coefficient position exceeds block size",
        ));
    }
    dst_scanned[pos] = levels[0];
    for lv in levels.iter().copied().take(total_coeff).skip(1) {
        let run_before = if zeros_left > 0 {
            let vlc_idx = (zeros_left - 1).min(6);
            match_vlc_index(r, &RUN_LEN[vlc_idx], &RUN_BITS[vlc_idx], zeros_left + 1)?
        } else {
            0
        };
        if run_before > zeros_left || pos < 1 + run_before {
            return Err(RagdollError::InvalidSyntax("CAVLC invalid run_before"));
        }
        zeros_left -= run_before;
        pos -= 1 + run_before;
        dst_scanned[pos] = lv;
    }

    Ok(total_coeff)
}

fn match_vlc_pair(
    r: &mut BitReader<'_>,
    lens: &[u8],
    bits: &[u8],
    count: usize,
) -> Result<(usize, usize)> {
    for idx in 0..count {
        let l = lens[idx] as u32;
        if l == 0 {
            continue;
        }
        if let Ok(peek) = r.peek_bits(l)
            && peek == (bits[idx] as u64)
        {
            r.consume_buffered(l);
            return Ok((idx / 4, idx % 4));
        }
    }
    Err(RagdollError::InvalidSyntax(
        "unmatched CAVLC coeff_token VLC",
    ))
}

fn match_vlc_index(r: &mut BitReader<'_>, lens: &[u8], bits: &[u8], count: usize) -> Result<usize> {
    for idx in 0..count.min(lens.len()) {
        let l = lens[idx] as u32;
        if l == 0 {
            continue;
        }
        if let Ok(peek) = r.peek_bits(l)
            && peek == (bits[idx] as u64)
        {
            r.consume_buffered(l);
            return Ok(idx);
        }
    }
    Err(RagdollError::InvalidSyntax(
        "unmatched CAVLC total_zeros/run_before VLC",
    ))
}

/// Pure-Rust H.264 / AVC Baseline (`profile_idc = 66`) video encoder.
///
/// Produces standard ITU-T H.264 Annex-B bitstreams (`SPS` + `PPS` + `IDR` / `P` slices)
/// using `I_16x16` spatial prediction + CAVLC Luma/Chroma DC Hadamard coding, `I_PCM` lossless
/// macroblocks for complex high-frequency blocks, and `P_Skip` inter macroblock coding.
/// All encoded streams decode cleanly in both `ffmpeg` and [`H264Decoder`].
#[derive(Clone, Debug)]
pub struct H264Encoder {
    pub sps: Sps,
    pub pps: Pps,
    pub qp: u8,
    pub keyint: u32,
    frame_count: u32,
    prev_recon: Option<VideoFrame>,
}

impl H264Encoder {
    pub fn new(width: u32, height: u32, fps: Rational64, qp: u8) -> Self {
        let sps = Sps::new_baseline(width, height, fps);
        let pps = Pps::new_baseline(qp);
        Self {
            sps,
            pps,
            qp: qp.clamp(0, 51),
            keyint: 30,
            frame_count: 0,
            prev_recon: None,
        }
    }

    /// Returns the Annex-B SPS + PPS codec header bytes.
    pub fn headers_annex_b(&self) -> Vec<u8> {
        let mut out = encode_annex_b_nal(3, NalUnitType::Sps, &self.sps.to_rbsp());
        out.extend_from_slice(&encode_annex_b_nal(
            3,
            NalUnitType::Pps,
            &self.pps.to_rbsp(),
        ));
        out
    }

    /// Encodes a `Yuv420p` [`VideoFrame`] into an H.264 Annex-B packet (`SPS` + `PPS` + `IDR` on keyframes, or `P` slice).
    pub fn encode_frame(&mut self, frame: &VideoFrame) -> Result<Vec<u8>> {
        if frame.format != PixelFormat::Yuv420p {
            return Err(RagdollError::Unsupported(format!(
                "H264Encoder requires Yuv420p input, got {}",
                frame.format
            )));
        }
        let is_idr = self.frame_count.is_multiple_of(self.keyint) || self.prev_recon.is_none();
        let nal_type = if is_idr {
            NalUnitType::IdrSlice
        } else {
            NalUnitType::NonIdrSlice
        };
        let nal_header = NalHeader {
            nal_ref_idc: 3,
            nal_unit_type: nal_type,
        };
        let slice_header = SliceHeader {
            first_mb_in_slice: 0,
            slice_type: if is_idr { 7 } else { 5 }, // 7 = all-I, 5 = all-P
            pict_type: if is_idr {
                PictureType::I
            } else {
                PictureType::P
            },
            pps_id: 0,
            frame_num: (self.frame_count & ((1 << self.sps.log2_max_frame_num) - 1)) as u16,
            idr_pic_id: if is_idr {
                Some((self.frame_count / self.keyint.max(1)) as u16)
            } else {
                None
            },
            pic_order_cnt_lsb: ((self.frame_count * 2)
                & ((1 << self.sps.log2_max_pic_order_cnt_lsb) - 1))
                as u16,
            slice_qp_delta: 0,
            slice_qp: self.qp,
            disable_deblocking_filter_idc: 1, // disable loop filter for deterministic fast decode
        };

        let mb_w = self.sps.pic_width_in_mbs as usize;
        let mb_h = self.sps.pic_height_in_map_units as usize;
        let mut recon = VideoFrame::new(mb_w * 16, mb_h * 16, PixelFormat::Yuv420p);
        let mut w = BitWriter::with_capacity(frame.width * frame.height + 128);
        slice_header.write(&mut w, nal_header, &self.sps, &self.pps);

        // Threshold for flat/smooth I_16x16_DC_0_0 or P_Skip macroblocks based on QP
        let sad_threshold_luma = if self.qp == 0 {
            0u32
        } else {
            (self.qp as u32) * 8
        };
        let sad_threshold_chroma = sad_threshold_luma / 2;

        let mut skip_run = 0u32;
        let total_mbs = mb_w * mb_h;
        let mut mb_nnz = vec![0u8; total_mbs];
        let mut mb_ref = vec![-2i8; total_mbs]; // 0 = L0 inter, -1 = intra, -2 = unavailable
        let mut mb_mv = vec![[0i16; 2]; total_mbs]; // quarter-pel [mv_x, mv_y]

        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let mb_idx = mb_y * mb_w + mb_x;
                let x0 = mb_x * 16;
                let y0 = mb_y * 16;
                let cx0 = mb_x * 8;
                let cy0 = mb_y * 8;

                // Extract 16x16 luma and 8x8 U/V from input frame (with edge clamping for non-16-multiple dims)
                let mut src_y = [0u8; 256];
                let mut src_u = [0u8; 64];
                let mut src_v = [0u8; 64];
                extract_mb_planes(frame, x0, y0, &mut src_y, &mut src_u, &mut src_v);

                // In P-slices, evaluate P_Skip and diamond motion-estimated P_L0_16x16 against prev_recon
                if !is_idr && let Some(prev) = &self.prev_recon {
                    let pskip_mv = pred_pskip_motion_16x16(&mb_ref, &mb_mv, mb_w, mb_x, mb_y);
                    let mvp = pred_motion_16x16(&mb_ref, &mb_mv, mb_w, mb_x, mb_y);

                    let mut best_mv = pskip_mv;
                    let mut ref_y = [0u8; 256];
                    let mut ref_u = [0u8; 64];
                    let mut ref_v = [0u8; 64];
                    extract_inter_mb_planes(
                        prev, x0, y0, pskip_mv, &mut ref_y, &mut ref_u, &mut ref_v,
                    );

                    let mut best_sad_y = sad_block_256_u8(&src_y, &ref_y);
                    let mut best_sad_uv =
                        sad_block_64_u8(&src_u, &ref_u) + sad_block_64_u8(&src_v, &ref_v);

                    if best_sad_y <= sad_threshold_luma && best_sad_uv <= sad_threshold_chroma {
                        skip_run += 1;
                        mb_nnz[mb_idx] = 0;
                        mb_ref[mb_idx] = 0;
                        mb_mv[mb_idx] = pskip_mv;
                        store_mb_planes(&mut recon, x0, y0, &ref_y, &ref_u, &ref_v);
                        continue;
                    }

                    // Fast even-integer-pel diamond search (`dx, dy in {-8, -6, -4, -2, 0, 2, 4, 6, 8}`)
                    let cand_mv =
                        search_inter_motion_16x16(prev, x0, y0, &src_y, mvp, sad_threshold_luma);
                    if cand_mv != pskip_mv {
                        let mut cand_y = [0u8; 256];
                        let mut cand_u = [0u8; 64];
                        let mut cand_v = [0u8; 64];
                        extract_inter_mb_planes(
                            prev,
                            x0,
                            y0,
                            cand_mv,
                            &mut cand_y,
                            &mut cand_u,
                            &mut cand_v,
                        );
                        let cy_sad = sad_block_256_u8(&src_y, &cand_y);
                        let cuv_sad =
                            sad_block_64_u8(&src_u, &cand_u) + sad_block_64_u8(&src_v, &cand_v);
                        if cy_sad < best_sad_y || (cy_sad == best_sad_y && cuv_sad < best_sad_uv) {
                            best_mv = cand_mv;
                            best_sad_y = cy_sad;
                            best_sad_uv = cuv_sad;
                            ref_y = cand_y;
                            ref_u = cand_u;
                            ref_v = cand_v;
                        }
                    }

                    if best_sad_y <= sad_threshold_luma && best_sad_uv <= sad_threshold_chroma {
                        if best_mv == pskip_mv {
                            skip_run += 1;
                            mb_nnz[mb_idx] = 0;
                            mb_ref[mb_idx] = 0;
                            mb_mv[mb_idx] = pskip_mv;
                            store_mb_planes(&mut recon, x0, y0, &ref_y, &ref_u, &ref_v);
                            continue;
                        }
                        // Emit P_L0_16x16 (`mb_type = 0`, `mvd_l0`, `coded_block_pattern = 0`)
                        w.write_ue(skip_run as u64);
                        skip_run = 0;
                        w.write_ue(0); // mb_type = 0 (P_L0_16x16)
                        w.write_se((best_mv[0] as i64) - (mvp[0] as i64));
                        w.write_se((best_mv[1] as i64) - (mvp[1] as i64));
                        w.write_ue(0); // coded_block_pattern = 0 (no residual blocks)
                        mb_nnz[mb_idx] = 0;
                        mb_ref[mb_idx] = 0;
                        mb_mv[mb_idx] = best_mv;
                        store_mb_planes(&mut recon, x0, y0, &ref_y, &ref_u, &ref_v);
                        continue;
                    }
                }

                if !is_idr {
                    w.write_ue(skip_run as u64);
                    skip_run = 0;
                }

                // Evaluate Intra 16x16 DC predictor (`pred_mode = 2`) and Chroma DC predictor (`mode = 0`)
                let top_y = if mb_y > 0 {
                    let mut arr = [0u8; 16];
                    arr.copy_from_slice(&recon.plane(0).row(y0 - 1)[x0..x0 + 16]);
                    Some(arr)
                } else {
                    None
                };
                let left_y = if mb_x > 0 {
                    let mut arr = [0u8; 16];
                    for (r, slot) in arr.iter_mut().enumerate() {
                        *slot = recon.plane(0).row(y0 + r)[x0 - 1];
                    }
                    Some(arr)
                } else {
                    None
                };
                let tl_y = if mb_x > 0 && mb_y > 0 {
                    recon.plane(0).row(y0 - 1)[x0 - 1]
                } else {
                    128
                };

                let top_u = if mb_y > 0 {
                    let mut arr = [0u8; 8];
                    arr.copy_from_slice(&recon.plane(1).row(cy0 - 1)[cx0..cx0 + 8]);
                    Some(arr)
                } else {
                    None
                };
                let left_u = if mb_x > 0 {
                    let mut arr = [0u8; 8];
                    for (r, slot) in arr.iter_mut().enumerate() {
                        *slot = recon.plane(1).row(cy0 + r)[cx0 - 1];
                    }
                    Some(arr)
                } else {
                    None
                };
                let tl_u = if mb_x > 0 && mb_y > 0 {
                    recon.plane(1).row(cy0 - 1)[cx0 - 1]
                } else {
                    128
                };

                let top_v = if mb_y > 0 {
                    let mut arr = [0u8; 8];
                    arr.copy_from_slice(&recon.plane(2).row(cy0 - 1)[cx0..cx0 + 8]);
                    Some(arr)
                } else {
                    None
                };
                let left_v = if mb_x > 0 {
                    let mut arr = [0u8; 8];
                    for (r, slot) in arr.iter_mut().enumerate() {
                        *slot = recon.plane(2).row(cy0 + r)[cx0 - 1];
                    }
                    Some(arr)
                } else {
                    None
                };
                let tl_v = if mb_x > 0 && mb_y > 0 {
                    recon.plane(2).row(cy0 - 1)[cx0 - 1]
                } else {
                    128
                };

                // Pick best valid I_16x16 luma mode (0=Vert if top, 1=Horiz if left, 2=DC always, 3=Plane if both)
                let mut best_luma_mode = 2u8;
                let mut best_pred_y = [128u8; 256];
                predict_intra_16x16(
                    2,
                    top_y.as_ref(),
                    left_y.as_ref(),
                    tl_y,
                    &mut best_pred_y,
                    16,
                );
                let mut best_sad_y = sad_block_256_u8(&src_y, &best_pred_y);

                for cand_mode in [0u8, 1, 3] {
                    if (cand_mode == 0 && top_y.is_none())
                        || (cand_mode == 1 && left_y.is_none())
                        || (cand_mode == 3 && (top_y.is_none() || left_y.is_none()))
                    {
                        continue;
                    }
                    let mut cand_pred = [0u8; 256];
                    predict_intra_16x16(
                        cand_mode,
                        top_y.as_ref(),
                        left_y.as_ref(),
                        tl_y,
                        &mut cand_pred,
                        16,
                    );
                    let sad = sad_block_256_u8(&src_y, &cand_pred);
                    if sad < best_sad_y {
                        best_sad_y = sad;
                        best_luma_mode = cand_mode;
                        best_pred_y = cand_pred;
                    }
                }

                let mut pred_u = [128u8; 64];
                let mut pred_v = [128u8; 64];
                predict_intra_chroma_8x8(0, top_u.as_ref(), left_u.as_ref(), tl_u, &mut pred_u, 8);
                predict_intra_chroma_8x8(0, top_v.as_ref(), left_v.as_ref(), tl_v, &mut pred_v, 8);
                let sad_uv = sad_block_64_u8(&src_u, &pred_u) + sad_block_64_u8(&src_v, &pred_v);

                let intra_offset = if is_idr { 0u64 } else { 5u64 };

                if best_sad_y <= sad_threshold_luma && sad_uv <= sad_threshold_chroma {
                    // Encode as I_16x16_{best_luma_mode}_0_0 (`mb_type = 1 + best_luma_mode`)
                    let nc = predict_mb_nc(&mb_nnz, mb_w, mb_x, mb_y);
                    w.write_ue(intra_offset + 1 + (best_luma_mode as u64));
                    w.write_ue(0); // intra_chroma_pred_mode = 0 (DC)
                    w.write_se(0); // mb_qp_delta = 0
                    encode_cavlc_block(&mut w, &[0i16; 16], nc, false);
                    mb_nnz[mb_idx] = 0;
                    mb_ref[mb_idx] = -1;
                    mb_mv[mb_idx] = [0, 0];
                    store_mb_planes(&mut recon, x0, y0, &best_pred_y, &pred_u, &pred_v);
                } else {
                    // Encode as I_PCM (`mb_type = 25` in I-slice, `30` in P-slice) for exact pixel fidelity
                    w.write_ue(intra_offset + 25);
                    w.align_to_byte();
                    for &b in &src_y {
                        w.write_bits(b as u64, 8);
                    }
                    for &b in &src_u {
                        w.write_bits(b as u64, 8);
                    }
                    for &b in &src_v {
                        w.write_bits(b as u64, 8);
                    }
                    mb_nnz[mb_idx] = 16;
                    mb_ref[mb_idx] = -1;
                    mb_mv[mb_idx] = [0, 0];
                    store_mb_planes(&mut recon, x0, y0, &src_y, &src_u, &src_v);
                }
            }
        }

        if !is_idr && skip_run > 0 {
            w.write_ue(skip_run as u64);
        }

        w.write_rbsp_trailing_bits();
        let slice_rbsp = w.finish();

        let mut out = if is_idr {
            self.headers_annex_b()
        } else {
            Vec::new()
        };
        out.extend_from_slice(&encode_annex_b_nal(3, nal_type, &slice_rbsp));

        self.prev_recon = Some(recon);
        self.frame_count += 1;
        Ok(out)
    }
}

#[inline]
fn median3(a: i16, b: i16, c: i16) -> i16 {
    a.max(b).min(a.min(b).max(c))
}

/// Computes the ITU-T H.264 8.4.1.3 median motion vector predictor `mvp` (in quarter-pel units)
/// for a $16\times16$ `P_L0_16x16` macroblock at `(mb_x, mb_y)` (`libavcodec/h264_mvpred.h:pred_motion`).
#[inline]
fn pred_motion_16x16(
    mb_ref: &[i8],
    mb_mv: &[[i16; 2]],
    mb_w: usize,
    mb_x: usize,
    mb_y: usize,
) -> [i16; 2] {
    let (left_ref, a) = if mb_x > 0 {
        let idx = mb_y * mb_w + (mb_x - 1);
        (mb_ref[idx], mb_mv[idx])
    } else {
        (-2, [0, 0])
    };
    let (top_ref, b) = if mb_y > 0 {
        let idx = (mb_y - 1) * mb_w + mb_x;
        (mb_ref[idx], mb_mv[idx])
    } else {
        (-2, [0, 0])
    };
    let (diag_ref, c) = if mb_y > 0 && mb_x + 1 < mb_w {
        let idx = (mb_y - 1) * mb_w + (mb_x + 1);
        (mb_ref[idx], mb_mv[idx])
    } else if mb_y > 0 && mb_x > 0 {
        let idx = (mb_y - 1) * mb_w + (mb_x - 1);
        (mb_ref[idx], mb_mv[idx])
    } else {
        (-2, [0, 0])
    };

    let match_count = (left_ref == 0) as u8 + (top_ref == 0) as u8 + (diag_ref == 0) as u8;
    if match_count > 1 {
        [median3(a[0], b[0], c[0]), median3(a[1], b[1], c[1])]
    } else if match_count == 1 {
        if left_ref == 0 {
            a
        } else if top_ref == 0 {
            b
        } else {
            c
        }
    } else if top_ref == -2 && diag_ref == -2 && left_ref != -2 {
        a
    } else {
        [median3(a[0], b[0], c[0]), median3(a[1], b[1], c[1])]
    }
}

/// Computes the ITU-T H.264 8.4.1.1 `P_Skip` motion vector predictor (in quarter-pel units)
/// at `(mb_x, mb_y)` (`libavcodec/h264_mvpred.h:pred_pskip_motion`).
#[inline]
fn pred_pskip_motion_16x16(
    mb_ref: &[i8],
    mb_mv: &[[i16; 2]],
    mb_w: usize,
    mb_x: usize,
    mb_y: usize,
) -> [i16; 2] {
    if mb_x == 0 || mb_y == 0 {
        return [0, 0];
    }
    let left_idx = mb_y * mb_w + (mb_x - 1);
    let top_idx = (mb_y - 1) * mb_w + mb_x;
    if (mb_ref[left_idx] == 0 && mb_mv[left_idx] == [0, 0])
        || (mb_ref[top_idx] == 0 && mb_mv[top_idx] == [0, 0])
    {
        return [0, 0];
    }
    pred_motion_16x16(mb_ref, mb_mv, mb_w, mb_x, mb_y)
}

/// Extracts a motion-compensated $16\times16$ luma and $8\times8$ U/V chroma macroblock from `prev`
/// at quarter-pel motion vector `mv = [mv_x, mv_y]` (`libavcodec/h264chroma_template.c`).
fn extract_inter_mb_planes(
    prev: &VideoFrame,
    x0: usize,
    y0: usize,
    mv: [i16; 2],
    dst_y: &mut [u8; 256],
    dst_u: &mut [u8; 64],
    dst_v: &mut [u8; 64],
) {
    let w = prev.width as isize;
    let h = prev.height as isize;
    let dx = (mv[0] >> 2) as isize;
    let dy = (mv[1] >> 2) as isize;
    let rx0 = (x0 as isize) + dx;
    let ry0 = (y0 as isize) + dy;

    let py = prev.plane(0);
    let pu = prev.plane(1);
    let pv = prev.plane(2);
    let cw = pu.width() as isize;
    let ch = pu.height() as isize;

    // Fast path: in-bounds even-integer pixel shift (exact integer luma and 4:2:0 chroma copy)
    if (mv[0] & 7) == 0
        && (mv[1] & 7) == 0
        && rx0 >= 0
        && rx0 + 16 <= w
        && ry0 >= 0
        && ry0 + 16 <= h
    {
        let ux0 = rx0 as usize;
        let uy0 = ry0 as usize;
        for r in 0..16 {
            dst_y[r * 16..r * 16 + 16].copy_from_slice(&py.row(uy0 + r)[ux0..ux0 + 16]);
        }
        let ucx0 = ux0 / 2;
        let ucy0 = uy0 / 2;
        for r in 0..8 {
            dst_u[r * 8..r * 8 + 8].copy_from_slice(&pu.row(ucy0 + r)[ucx0..ucx0 + 8]);
            dst_v[r * 8..r * 8 + 8].copy_from_slice(&pv.row(ucy0 + r)[ucx0..ucx0 + 8]);
        }
        return;
    }

    for r in 0..16isize {
        let sy = (ry0 + r).clamp(0, h - 1) as usize;
        let row = py.row(sy);
        for c in 0..16isize {
            let sx = (rx0 + c).clamp(0, w - 1) as usize;
            dst_y[(r as usize) * 16 + (c as usize)] = row[sx];
        }
    }

    // 4:2:0 1/8-pel bilinear chroma interpolation (`ff_put_h264_chroma_mc8`)
    let cx_base = ((x0 / 2) as isize) + ((mv[0] >> 3) as isize);
    let cy_base = ((y0 / 2) as isize) + ((mv[1] >> 3) as isize);
    let fx = (mv[0].rem_euclid(8)) as u32;
    let fy = (mv[1].rem_euclid(8)) as u32;

    if fx == 0 && fy == 0 {
        for r in 0..8isize {
            let sy = (cy_base + r).clamp(0, ch - 1) as usize;
            let ru = pu.row(sy);
            let rv = pv.row(sy);
            for c in 0..8isize {
                let sx = (cx_base + c).clamp(0, cw - 1) as usize;
                dst_u[(r as usize) * 8 + (c as usize)] = ru[sx];
                dst_v[(r as usize) * 8 + (c as usize)] = rv[sx];
            }
        }
    } else {
        let wa = (8 - fx) * (8 - fy);
        let wb = fx * (8 - fy);
        let wc = (8 - fx) * fy;
        let wd = fx * fy;
        for r in 0..8isize {
            let sy0 = (cy_base + r).clamp(0, ch - 1) as usize;
            let sy1 = (cy_base + r + 1).clamp(0, ch - 1) as usize;
            let ru0 = pu.row(sy0);
            let ru1 = pu.row(sy1);
            let rv0 = pv.row(sy0);
            let rv1 = pv.row(sy1);
            for c in 0..8isize {
                let sx0 = (cx_base + c).clamp(0, cw - 1) as usize;
                let sx1 = (cx_base + c + 1).clamp(0, cw - 1) as usize;
                let u_val = (wa * (ru0[sx0] as u32)
                    + wb * (ru0[sx1] as u32)
                    + wc * (ru1[sx0] as u32)
                    + wd * (ru1[sx1] as u32)
                    + 32)
                    >> 6;
                let v_val = (wa * (rv0[sx0] as u32)
                    + wb * (rv0[sx1] as u32)
                    + wc * (rv1[sx0] as u32)
                    + wd * (rv1[sx1] as u32)
                    + 32)
                    >> 6;
                dst_u[(r as usize) * 8 + (c as usize)] = u_val as u8;
                dst_v[(r as usize) * 8 + (c as usize)] = v_val as u8;
            }
        }
    }
}

/// Fast NEON diamond motion search over even integer luma shifts `(dx, dy)` within `[-12, 12]`
/// using `sad_16x16_u8` directly on `prev`'s luma plane.
fn search_inter_motion_16x16(
    prev: &VideoFrame,
    x0: usize,
    y0: usize,
    src_y: &[u8; 256],
    mvp: [i16; 2],
    sad_threshold: u32,
) -> [i16; 2] {
    let w = prev.width as isize;
    let h = prev.height as isize;
    let py = prev.plane(0);
    let stride = py.stride();
    let data = py.data();

    let eval_dx_dy = |dx: isize, dy: isize| -> Option<u32> {
        let rx = (x0 as isize) + dx;
        let ry = (y0 as isize) + dy;
        if rx < 0 || rx + 16 > w || ry < 0 || ry + 16 > h {
            return None;
        }
        let offset = (ry as usize) * stride + (rx as usize);
        Some(sad_16x16_u8(src_y, 16, &data[offset..], stride))
    };

    let mut best_dx = 0isize;
    let mut best_dy = 0isize;
    let mut best_sad = eval_dx_dy(0, 0).unwrap_or(u32::MAX);
    if best_sad <= sad_threshold {
        return [0, 0];
    }

    // Check predicted MV (snapped to nearest even integer pixel)
    let mvp_dx = (((mvp[0] >> 2) as isize) & !1).clamp(-12, 12);
    let mvp_dy = (((mvp[1] >> 2) as isize) & !1).clamp(-12, 12);
    if (mvp_dx != 0 || mvp_dy != 0)
        && let Some(s) = eval_dx_dy(mvp_dx, mvp_dy)
        && s < best_sad
    {
        best_sad = s;
        best_dx = mvp_dx;
        best_dy = mvp_dy;
        if best_sad <= sad_threshold {
            return [(best_dx * 4) as i16, (best_dy * 4) as i16];
        }
    }

    // 2-stage diamond + diagonal search (step = 4, then step = 2)
    for step in [4isize, 2isize] {
        let cx = best_dx;
        let cy = best_dy;
        let offsets = [
            (-step, 0),
            (step, 0),
            (0, -step),
            (0, step),
            (-step, -step),
            (step, step),
            (-step, step),
            (step, -step),
        ];
        for (ox, oy) in offsets {
            let cand_dx = cx + ox;
            let cand_dy = cy + oy;
            if cand_dx.abs() > 16 || cand_dy.abs() > 16 {
                continue;
            }
            if let Some(s) = eval_dx_dy(cand_dx, cand_dy)
                && s < best_sad
            {
                best_sad = s;
                best_dx = cand_dx;
                best_dy = cand_dy;
                if best_sad == 0 {
                    return [(best_dx * 4) as i16, (best_dy * 4) as i16];
                }
            }
        }
    }

    [(best_dx * 4) as i16, (best_dy * 4) as i16]
}

fn extract_mb_planes(
    frame: &VideoFrame,
    x0: usize,
    y0: usize,
    dst_y: &mut [u8; 256],
    dst_u: &mut [u8; 64],
    dst_v: &mut [u8; 64],
) {
    let w = frame.width;
    let h = frame.height;
    let py = frame.plane(0);
    if x0 + 16 <= w && y0 + 16 <= h {
        for r in 0..16 {
            dst_y[r * 16..r * 16 + 16].copy_from_slice(&py.row(y0 + r)[x0..x0 + 16]);
        }
    } else {
        for r in 0..16 {
            let sy = (y0 + r).min(h.saturating_sub(1));
            let row = py.row(sy);
            for c in 0..16 {
                let sx = (x0 + c).min(w.saturating_sub(1));
                dst_y[r * 16 + c] = row[sx];
            }
        }
    }
    let cw = frame.plane(1).width();
    let ch = frame.plane(1).height();
    let pu = frame.plane(1);
    let pv = frame.plane(2);
    let cx0 = x0 / 2;
    let cy0 = y0 / 2;
    if cx0 + 8 <= cw && cy0 + 8 <= ch {
        for r in 0..8 {
            dst_u[r * 8..r * 8 + 8].copy_from_slice(&pu.row(cy0 + r)[cx0..cx0 + 8]);
            dst_v[r * 8..r * 8 + 8].copy_from_slice(&pv.row(cy0 + r)[cx0..cx0 + 8]);
        }
    } else {
        for r in 0..8 {
            let sy = (cy0 + r).min(ch.saturating_sub(1));
            let ru = pu.row(sy);
            let rv = pv.row(sy);
            for c in 0..8 {
                let sx = (cx0 + c).min(cw.saturating_sub(1));
                dst_u[r * 8 + c] = ru[sx];
                dst_v[r * 8 + c] = rv[sx];
            }
        }
    }
}

fn store_mb_planes(
    recon: &mut VideoFrame,
    x0: usize,
    y0: usize,
    src_y: &[u8; 256],
    src_u: &[u8; 64],
    src_v: &[u8; 64],
) {
    for r in 0..16 {
        recon.plane_mut(0).row_mut(y0 + r)[x0..x0 + 16]
            .copy_from_slice(&src_y[r * 16..r * 16 + 16]);
    }
    let cx0 = x0 / 2;
    let cy0 = y0 / 2;
    for r in 0..8 {
        recon.plane_mut(1).row_mut(cy0 + r)[cx0..cx0 + 8].copy_from_slice(&src_u[r * 8..r * 8 + 8]);
        recon.plane_mut(2).row_mut(cy0 + r)[cx0..cx0 + 8].copy_from_slice(&src_v[r * 8..r * 8 + 8]);
    }
}

/// Pure-Rust H.264 / AVC Baseline (`profile_idc = 66`) video decoder.
///
/// Decodes Annex-B or length-prefixed NAL packets containing `SPS`, `PPS`, `SEI`, and
/// `IDR` / `NonIdr` slices (`I_16x16` with CAVLC Luma/Chroma DC + AC $4\times4$ IDCT,
/// `I_PCM`, `P_Skip`, and `P_L0_16x16`).
#[derive(Clone, Debug, Default)]
pub struct H264Decoder {
    pub sps: Option<Sps>,
    pub pps: Option<Pps>,
    pub last_sei: Vec<SeiMessage>,
    prev_recon: Option<VideoFrame>,
}

impl H264Decoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds raw `avcC` extradata or Annex-B SPS/PPS headers into the decoder.
    pub fn set_extradata(&mut self, data: &[u8]) -> Result<()> {
        for nal in split_annex_b_nals(data) {
            let (hdr, rbsp) = parse_nal_rbsp(nal)?;
            match hdr.nal_unit_type {
                NalUnitType::Sps => self.sps = Some(Sps::parse_rbsp(&rbsp)?),
                NalUnitType::Pps => self.pps = Some(Pps::parse_rbsp(&rbsp)?),
                NalUnitType::Sei => self.last_sei = SeiMessage::parse_rbsp(&rbsp),
                _ => {}
            }
        }
        Ok(())
    }

    /// Decodes an Annex-B H.264 access unit into a [`VideoFrame`] if a slice NAL is present.
    pub fn decode_annex_b_packet(&mut self, data: &[u8]) -> Result<Option<VideoFrame>> {
        let mut decoded_frame = None;
        for nal in split_annex_b_nals(data) {
            let (hdr, rbsp) = parse_nal_rbsp(nal)?;
            match hdr.nal_unit_type {
                NalUnitType::Sps => {
                    self.sps = Some(Sps::parse_rbsp(&rbsp)?);
                }
                NalUnitType::Pps => {
                    self.pps = Some(Pps::parse_rbsp(&rbsp)?);
                }
                NalUnitType::Sei => {
                    self.last_sei = SeiMessage::parse_rbsp(&rbsp);
                }
                NalUnitType::IdrSlice | NalUnitType::NonIdrSlice => {
                    decoded_frame = Some(self.decode_slice_rbsp(hdr, &rbsp)?);
                }
                _ => {}
            }
        }
        Ok(decoded_frame)
    }

    fn decode_slice_rbsp(&mut self, nal_header: NalHeader, rbsp: &[u8]) -> Result<VideoFrame> {
        let sps = self
            .sps
            .as_ref()
            .ok_or(RagdollError::InvalidSyntax("H.264 slice before SPS"))?
            .clone();
        let pps = self
            .pps
            .as_ref()
            .ok_or(RagdollError::InvalidSyntax("H.264 slice before PPS"))?
            .clone();

        let mut r = BitReader::new(rbsp);
        let sh = SliceHeader::parse_from_reader(&mut r, nal_header, &sps, &pps)?;

        let mb_w = sps.pic_width_in_mbs as usize;
        let mb_h = sps.pic_height_in_map_units as usize;
        let total_mbs = mb_w * mb_h;
        let mut recon = VideoFrame::new(mb_w * 16, mb_h * 16, PixelFormat::Yuv420p);
        let mut cur_qp = sh.slice_qp as i32;
        let is_p_slice = sh.pict_type == PictureType::P;
        let mut skip_run: i64 = -1;
        let mut mb_nnz = vec![0u8; total_mbs];
        let mut mb_ref = vec![-2i8; total_mbs];
        let mut mb_mv = vec![[0i16; 2]; total_mbs];

        for mb_idx in (sh.first_mb_in_slice as usize)..total_mbs {
            let mb_x = mb_idx % mb_w;
            let mb_y = mb_idx / mb_w;
            let x0 = mb_x * 16;
            let y0 = mb_y * 16;
            let cx0 = mb_x * 8;
            let cy0 = mb_y * 8;

            if is_p_slice {
                if skip_run < 0 {
                    if !r.more_rbsp_data() && r.bits_remaining() <= 8 {
                        skip_run = (total_mbs - mb_idx) as i64;
                    } else {
                        skip_run = r.read_ue()? as i64;
                    }
                }
                if skip_run > 0 {
                    skip_run -= 1;
                    let pskip_mv = pred_pskip_motion_16x16(&mb_ref, &mb_mv, mb_w, mb_x, mb_y);
                    mb_nnz[mb_idx] = 0;
                    mb_ref[mb_idx] = 0;
                    mb_mv[mb_idx] = pskip_mv;
                    if let Some(prev) = &self.prev_recon {
                        let mut ref_y = [0u8; 256];
                        let mut ref_u = [0u8; 64];
                        let mut ref_v = [0u8; 64];
                        extract_inter_mb_planes(
                            prev, x0, y0, pskip_mv, &mut ref_y, &mut ref_u, &mut ref_v,
                        );
                        store_mb_planes(&mut recon, x0, y0, &ref_y, &ref_u, &ref_v);
                    }
                    continue;
                }
                skip_run = -1;
            }

            let raw_mb_type = r.read_ue()? as u32;
            let intra_mb_type = if is_p_slice {
                if raw_mb_type == 0 {
                    // P_L0_16x16 macroblock (`mb_type = 0`)
                    let mvp = pred_motion_16x16(&mb_ref, &mb_mv, mb_w, mb_x, mb_y);
                    let mvd_x = r.read_se()? as i16;
                    let mvd_y = r.read_se()? as i16;
                    let mv = [mvp[0].wrapping_add(mvd_x), mvp[1].wrapping_add(mvd_y)];
                    let cbp_code = r.read_ue()? as usize;
                    if cbp_code != 0 {
                        return Err(RagdollError::Unsupported(
                            "P_L0_16x16 with non-zero cbp".into(),
                        ));
                    }
                    mb_nnz[mb_idx] = 0;
                    mb_ref[mb_idx] = 0;
                    mb_mv[mb_idx] = mv;
                    if let Some(prev) = &self.prev_recon {
                        let mut ref_y = [0u8; 256];
                        let mut ref_u = [0u8; 64];
                        let mut ref_v = [0u8; 64];
                        extract_inter_mb_planes(
                            prev, x0, y0, mv, &mut ref_y, &mut ref_u, &mut ref_v,
                        );
                        store_mb_planes(&mut recon, x0, y0, &ref_y, &ref_u, &ref_v);
                    }
                    continue;
                }
                if raw_mb_type < 5 {
                    return Err(RagdollError::Unsupported(
                        "non-16x16 P macroblock partition".into(),
                    ));
                }
                raw_mb_type - 5
            } else {
                raw_mb_type
            };

            if intra_mb_type == 25 {
                // I_PCM macroblock
                r.align_to_byte();
                let mut pcm_y = [0u8; 256];
                let mut pcm_u = [0u8; 64];
                let mut pcm_v = [0u8; 64];
                for b in &mut pcm_y {
                    *b = r.read_bits(8)? as u8;
                }
                for b in &mut pcm_u {
                    *b = r.read_bits(8)? as u8;
                }
                for b in &mut pcm_v {
                    *b = r.read_bits(8)? as u8;
                }
                mb_nnz[mb_idx] = 16;
                mb_ref[mb_idx] = -1;
                mb_mv[mb_idx] = [0, 0];
                store_mb_planes(&mut recon, x0, y0, &pcm_y, &pcm_u, &pcm_v);
                continue;
            }

            if !(1..=24).contains(&intra_mb_type) {
                return Err(RagdollError::Unsupported(format!(
                    "H.264 intra mb_type {intra_mb_type} not supported in baseline fast decoder"
                )));
            }
            mb_ref[mb_idx] = -1;
            mb_mv[mb_idx] = [0, 0];

            // I_16x16_{pred_mode}_{cbp_chroma}_{cbp_luma}
            let code = intra_mb_type - 1;
            let luma_pred_mode = (code % 4) as u8;
            let cbp_chroma = ((code / 4) % 3) as u8;
            let cbp_luma = if code >= 12 { 15u8 } else { 0u8 };

            let chroma_pred_mode = r.read_ue()? as u8;
            let mb_qp_delta = r.read_se()? as i32;
            cur_qp = (cur_qp + mb_qp_delta).rem_euclid(52);
            let qp = cur_qp as u8;
            let qpc_idx = (cur_qp + (pps.chroma_qp_index_offset as i32)).clamp(0, 51) as usize;
            let qpc = CHROMA_QP_TABLE[qpc_idx];

            // Decode Luma 16x16 DC CAVLC block
            let nc = predict_mb_nc(&mb_nnz, mb_w, mb_x, mb_y);
            let mut dc_scanned = [0i16; 16];
            decode_cavlc_block(&mut r, &mut dc_scanned, nc, false)?;
            mb_nnz[mb_idx] = 0;
            let mut dc_matrix = [0i16; 16];
            for i in 0..16 {
                dc_matrix[ZIGZAG_4X4[i]] = dc_scanned[i];
            }
            let luma_dc = dequantize_idct_luma_dc4x4(&dc_matrix, qp);

            // Decode Luma 4x4 AC blocks if cbp_luma != 0
            let mut luma_4x4 = [[0i16; 16]; 16];
            for (b, blk4x4) in luma_4x4.iter_mut().enumerate() {
                // H.264 4x4 block scan index inside 16x16 macroblock:
                let bx = (b & 1) + ((b >> 1) & 2);
                let by = ((b >> 1) & 1) + ((b >> 2) & 2);
                blk4x4[0] = luma_dc[by * 4 + bx];
                if cbp_luma != 0 {
                    let mut ac_scanned = [0i16; 15];
                    decode_cavlc_block(&mut r, &mut ac_scanned, 0, false)?;
                    let mut raw_levels = [0i16; 16];
                    for k in 0..15 {
                        raw_levels[ZIGZAG_4X4[k + 1]] = ac_scanned[k];
                    }
                    let deq = dequantize_block4x4(&raw_levels, qp);
                    blk4x4[1..16].copy_from_slice(&deq[1..16]);
                }
            }

            // Decode Chroma DC & AC if cbp_chroma > 0
            let mut chroma_dc = [[0i16; 4]; 2];
            if cbp_chroma >= 1 {
                for cdc_plane in &mut chroma_dc {
                    let mut cdc_scanned = [0i16; 4];
                    decode_cavlc_block(&mut r, &mut cdc_scanned, 0, true)?;
                    *cdc_plane = dequantize_idct_chroma_dc2x2(&cdc_scanned, qpc);
                }
            }

            // Predict 16x16 Luma + 8x8 Chroma
            let top_y = if mb_y > 0 {
                let mut arr = [0u8; 16];
                arr.copy_from_slice(&recon.plane(0).row(y0 - 1)[x0..x0 + 16]);
                Some(arr)
            } else {
                None
            };
            let left_y = if mb_x > 0 {
                let mut arr = [0u8; 16];
                for (row_i, slot) in arr.iter_mut().enumerate() {
                    *slot = recon.plane(0).row(y0 + row_i)[x0 - 1];
                }
                Some(arr)
            } else {
                None
            };
            let tl_y = if mb_x > 0 && mb_y > 0 {
                recon.plane(0).row(y0 - 1)[x0 - 1]
            } else {
                128
            };

            let mut mb_y_buf = [0u8; 256];
            predict_intra_16x16(
                luma_pred_mode,
                top_y.as_ref(),
                left_y.as_ref(),
                tl_y,
                &mut mb_y_buf,
                16,
            );

            // Add 4x4 IDCT residuals onto predicted Luma
            for (b, blk4x4) in luma_4x4.iter().enumerate() {
                let bx = ((b & 1) + ((b >> 1) & 2)) * 4;
                let by = (((b >> 1) & 1) + ((b >> 2) & 2)) * 4;
                if blk4x4.iter().any(|&v| v != 0) {
                    add4x4_idct_u8(&mut mb_y_buf[by * 16 + bx..], 16, blk4x4);
                }
            }

            let top_u = if mb_y > 0 {
                let mut arr = [0u8; 8];
                arr.copy_from_slice(&recon.plane(1).row(cy0 - 1)[cx0..cx0 + 8]);
                Some(arr)
            } else {
                None
            };
            let left_u = if mb_x > 0 {
                let mut arr = [0u8; 8];
                for (row_i, slot) in arr.iter_mut().enumerate() {
                    *slot = recon.plane(1).row(cy0 + row_i)[cx0 - 1];
                }
                Some(arr)
            } else {
                None
            };
            let tl_u = if mb_x > 0 && mb_y > 0 {
                recon.plane(1).row(cy0 - 1)[cx0 - 1]
            } else {
                128
            };

            let top_v = if mb_y > 0 {
                let mut arr = [0u8; 8];
                arr.copy_from_slice(&recon.plane(2).row(cy0 - 1)[cx0..cx0 + 8]);
                Some(arr)
            } else {
                None
            };
            let left_v = if mb_x > 0 {
                let mut arr = [0u8; 8];
                for (row_i, slot) in arr.iter_mut().enumerate() {
                    *slot = recon.plane(2).row(cy0 + row_i)[cx0 - 1];
                }
                Some(arr)
            } else {
                None
            };
            let tl_v = if mb_x > 0 && mb_y > 0 {
                recon.plane(2).row(cy0 - 1)[cx0 - 1]
            } else {
                128
            };

            let mut mb_u_buf = [0u8; 64];
            let mut mb_v_buf = [0u8; 64];
            predict_intra_chroma_8x8(
                chroma_pred_mode,
                top_u.as_ref(),
                left_u.as_ref(),
                tl_u,
                &mut mb_u_buf,
                8,
            );
            predict_intra_chroma_8x8(
                chroma_pred_mode,
                top_v.as_ref(),
                left_v.as_ref(),
                tl_v,
                &mut mb_v_buf,
                8,
            );

            for (b, (&dc_u, &dc_v)) in chroma_dc[0].iter().zip(chroma_dc[1].iter()).enumerate() {
                let bx = (b & 1) * 4;
                let by = (b >> 1) * 4;
                if dc_u != 0 {
                    let mut blk = [0i16; 16];
                    blk[0] = dc_u;
                    add4x4_idct_u8(&mut mb_u_buf[by * 8 + bx..], 8, &blk);
                }
                if dc_v != 0 {
                    let mut blk = [0i16; 16];
                    blk[0] = dc_v;
                    add4x4_idct_u8(&mut mb_v_buf[by * 8 + bx..], 8, &blk);
                }
            }

            store_mb_planes(&mut recon, x0, y0, &mb_y_buf, &mb_u_buf, &mb_v_buf);
        }

        self.prev_recon = Some(recon.clone());

        // Crop to exact display dimensions if SPS frame_cropping_flag is set
        let out_w = sps.width() as usize;
        let out_h = sps.height() as usize;
        let mut out_frame = if out_w == recon.width && out_h == recon.height {
            recon
        } else {
            let mut cropped = VideoFrame::new(out_w, out_h, PixelFormat::Yuv420p);
            for y in 0..out_h {
                cropped
                    .plane_mut(0)
                    .row_mut(y)
                    .copy_from_slice(&recon.plane(0).row(y)[..out_w]);
            }
            let cw = out_w.div_ceil(2);
            let ch = out_h.div_ceil(2);
            for y in 0..ch {
                cropped
                    .plane_mut(1)
                    .row_mut(y)
                    .copy_from_slice(&recon.plane(1).row(y)[..cw]);
                cropped
                    .plane_mut(2)
                    .row_mut(y)
                    .copy_from_slice(&recon.plane(2).row(y)[..cw]);
            }
            cropped
        };

        out_frame.colorimetry = sps.colorimetry;
        out_frame.pict_type = sh.pict_type;
        out_frame.keyframe = nal_header.nal_unit_type == NalUnitType::IdrSlice;
        out_frame.pts = Timestamp::new(
            (sh.pic_order_cnt_lsb / 2) as i64,
            sps.fps.unwrap_or(Rational64::FPS_30).recip(),
        );
        Ok(out_frame)
    }
}

/// Helper exposing $4\times4$ residual DCT + quantization + dequantization + IDCT reconstruction for a single $4\times4$ block.
pub fn roundtrip_block4x4_dct(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    qp: u8,
) {
    let dct = sub4x4_dct_u8(src, src_stride, pred, pred_stride);
    let q = quantize_block4x4(&dct, qp, true);
    let deq = dequantize_block4x4(&q, qp);
    for r in 0..4 {
        dst[r * dst_stride..r * dst_stride + 4]
            .copy_from_slice(&pred[r * pred_stride..r * pred_stride + 4]);
    }
    let _idct = inverse_idct4x4(&deq);
    add4x4_idct_u8(dst, dst_stride, &deq);
    let _ = forward_dct4x4;
}
