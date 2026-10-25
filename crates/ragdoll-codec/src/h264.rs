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

use crate::cabac::{CabacContext, CabacDecoder, CabacEncoder, init_cabac_contexts};
use ragdoll_core::{
    BitReader, BitWriter, ColorPrimaries, ColorRange, Colorimetry, MatrixCoefficients, PictureType,
    PixelFormat, RagdollError, Rational64, Result, Timestamp, TransferCharacteristics, VideoFrame,
    escape_rbsp, unescape_rbsp,
};
use ragdoll_simd::{
    add4x4_idct_u8, clamp_u8, forward_dct4x4, forward_dct8x8, h264_loop_filter_chroma_edge_h_u8,
    h264_loop_filter_chroma_edge_v_u8, h264_loop_filter_luma_edge_h_u8,
    h264_loop_filter_luma_edge_v_u8, hadamard4x4, inverse_idct4x4, inverse_idct8x8,
    predict_intra_4x4_u8, put_h264_chroma_mc8_u8, put_h264_qpel8_mc_u8, put_h264_qpel16_mc_u8,
    sad_16x16_u8, sad_block_64_u8, sad_block_256_u8, split_annex_b_nals, sub4x4_dct_u8,
};

/// H.264 / AVC coding profile (`0040`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum H264Profile {
    /// Constrained Baseline Profile (`profile_idc = 66`, CAVLC, $4\times4$ transform, I/P slices).
    Baseline,
    /// Main Profile (`profile_idc = 77`, CABAC, $4\times4$ transform, I/P/B slices).
    Main,
    /// High Profile (`profile_idc = 100`, CABAC, $8\times8$ + $4\times4$ adaptive transform, I/P/B slices).
    #[default]
    High,
}

impl H264Profile {
    pub fn parse(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "baseline" | "constrained_baseline" | "cavlc" => Self::Baseline,
            "main" => Self::Main,
            _ => Self::High,
        }
    }
}

/// Rate control mode for [`H264Encoder`] (`0041`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RateControlMode {
    /// Constant Quantization Parameter (`--qp 0..51`).
    FixedQp(u8),
    /// Perceptual Constant Rate Factor (`--crf 0..51`) with Variance-Adaptive Quantization (`VAQ`)
    /// and optional VBV `--maxrate` / `--bufsize` leaky-bucket constraints.
    Crf {
        crf: f32,
        maxrate_kbps: Option<u32>,
        bufsize_kb: Option<u32>,
    },
    /// Target Average Bitrate (`--bitrate <kbps>`) with VBV `--maxrate` / `--bufsize` constraints.
    Abr {
        bitrate_kbps: u32,
        maxrate_kbps: Option<u32>,
        bufsize_kb: Option<u32>,
    },
}

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
        Self::new_profile(width, height, fps, H264Profile::Baseline)
    }

    /// Creates an SPS for the specified [`H264Profile`] (`Baseline = 66`, `Main = 77`, `High = 100`).
    pub fn new_profile(width: u32, height: u32, fps: Rational64, profile: H264Profile) -> Self {
        let mb_w = width.div_ceil(16);
        let mb_h = height.div_ceil(16);
        let coded_w = mb_w * 16;
        let coded_h = mb_h * 16;
        // For 4:2:0 frame_mbs_only=true, CropUnitX = 2, CropUnitY = 2
        let crop_right = (coded_w - width) / 2;
        let crop_bottom = (coded_h - height) / 2;
        let (profile_idc, constraint_set_flags) = match profile {
            H264Profile::Baseline => (66, 0xC0),
            H264Profile::Main => (77, 0x40),
            H264Profile::High => (100, 0x00),
        };

        Self {
            profile_idc,
            constraint_set_flags,
            level_idc: 40,
            sps_id: 0,
            chroma_format_idc: 1,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            log2_max_frame_num: 4,
            pic_order_cnt_type: 0,
            log2_max_pic_order_cnt_lsb: 6,
            max_num_ref_frames: 2,
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
            bit_depth_luma = (r.read_ue()? as u8).saturating_add(8);
            bit_depth_chroma = (r.read_ue()? as u8).saturating_add(8);
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
                                next_scale =
                                    (last_scale.wrapping_add(delta_scale).wrapping_add(256))
                                        .rem_euclid(256);
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

        let log2_max_frame_num = (r.read_ue()? as u8).saturating_add(4).min(16);
        let pic_order_cnt_type = r.read_ue()? as u8;
        let mut log2_max_pic_order_cnt_lsb = 4u8;
        if pic_order_cnt_type == 0 {
            log2_max_pic_order_cnt_lsb = (r.read_ue()? as u8).saturating_add(4).min(16);
        } else if pic_order_cnt_type == 1 {
            let _delta_pic_order_always_zero = r.read_bit()?;
            let _offset_for_non_ref_pic = r.read_se()?;
            let _offset_for_top_to_bottom = r.read_se()?;
            let cycle_len = r.read_ue()?;
            if cycle_len > 255 {
                return Err(RagdollError::InvalidSyntax(
                    "H.264 num_ref_frames_in_pic_order_cnt_cycle > 255",
                ));
            }
            for _ in 0..cycle_len {
                let _ = r.read_se()?;
            }
        }

        let max_num_ref_frames = r.read_ue()? as u8;
        let _gaps_in_frame_num_allowed = r.read_bit()?;
        let pic_width_in_mbs = (r.read_ue()? as u32).saturating_add(1);
        let pic_height_in_map_units = (r.read_ue()? as u32).saturating_add(1);
        if pic_width_in_mbs == 0
            || pic_height_in_map_units == 0
            || pic_width_in_mbs > 512
            || pic_height_in_map_units > 512
        {
            return Err(RagdollError::InvalidSyntax(
                "H.264 SPS macroblock dimensions out of supported range",
            ));
        }
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
    pub transform_8x8_mode_flag: bool,
    pub second_chroma_qp_index_offset: i8,
}

impl Pps {
    pub fn new_baseline(qp: u8) -> Self {
        Self::new_profile(qp, H264Profile::Baseline)
    }

    pub fn new_profile(qp: u8, profile: H264Profile) -> Self {
        Self {
            pps_id: 0,
            sps_id: 0,
            entropy_coding_mode_flag: matches!(profile, H264Profile::Main | H264Profile::High),
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
            transform_8x8_mode_flag: profile == H264Profile::High,
            second_chroma_qp_index_offset: 0,
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

        let mut transform_8x8_mode_flag = false;
        let mut second_chroma_qp_index_offset = chroma_qp_index_offset;
        if r.more_rbsp_data() {
            transform_8x8_mode_flag = r.read_bit()?;
            let pic_scaling_matrix_present_flag = r.read_bit()?;
            if pic_scaling_matrix_present_flag {
                let n_lists = 6 + if transform_8x8_mode_flag { 2 } else { 0 };
                for i in 0..n_lists {
                    if r.read_bit()? {
                        let size = if i < 6 { 16 } else { 64 };
                        let mut last_scale = 8i64;
                        let mut next_scale = 8i64;
                        for _ in 0..size {
                            if next_scale != 0 {
                                let delta_scale = r.read_se()?;
                                next_scale =
                                    (last_scale.wrapping_add(delta_scale).wrapping_add(256))
                                        .rem_euclid(256);
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
            if r.bits_remaining() > 0 {
                second_chroma_qp_index_offset =
                    r.read_se().unwrap_or(chroma_qp_index_offset as i64) as i8;
            }
        }

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
            transform_8x8_mode_flag,
            second_chroma_qp_index_offset,
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
        if self.transform_8x8_mode_flag
            || self.second_chroma_qp_index_offset != self.chroma_qp_index_offset
        {
            w.write_bit(self.transform_8x8_mode_flag);
            w.write_bit(false); // pic_scaling_matrix_present_flag
            w.write_se(self.second_chroma_qp_index_offset as i64);
        }
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
    pub direct_spatial_mv_pred_flag: bool,
    pub num_ref_idx_l0_active_minus1: u8,
    pub num_ref_idx_l1_active_minus1: u8,
    pub cabac_init_idc: u8,
    pub slice_qp_delta: i8,
    pub slice_qp: u8,
    pub disable_deblocking_filter_idc: u8,
    pub slice_alpha_c0_offset_div2: i8,
    pub slice_beta_offset_div2: i8,
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
        } else if sps.pic_order_cnt_type == 1 {
            let _delta_poc0 = r.read_se()?;
            if pps.bottom_field_pic_order_in_frame_present_flag && !field_pic_flag {
                let _delta_poc1 = r.read_se()?;
            }
        }

        if pps.redundant_pic_cnt_present_flag {
            let _redundant_pic_cnt = r.read_ue()?;
        }

        let mut direct_spatial_mv_pred_flag = true;
        if pict_type == PictureType::B {
            direct_spatial_mv_pred_flag = r.read_bit()?;
        }

        let mut num_ref_idx_l0_active_minus1 = pps.num_ref_idx_l0_default_active_minus1;
        let mut num_ref_idx_l1_active_minus1 = pps.num_ref_idx_l1_default_active_minus1;
        if matches!(pict_type, PictureType::P | PictureType::B) {
            let num_ref_override = r.read_bit()?;
            if num_ref_override {
                num_ref_idx_l0_active_minus1 = r.read_ue()? as u8;
                if pict_type == PictureType::B {
                    num_ref_idx_l1_active_minus1 = r.read_ue()? as u8;
                }
            }
        }

        // ref_pic_list_modification
        if pict_type != PictureType::I {
            let mod_l0 = r.read_bit()?;
            if mod_l0 {
                for _ in 0..32 {
                    let idc = r.read_ue()?;
                    if idc >= 3 {
                        break;
                    }
                    let _ = r.read_ue()?;
                }
            }
            if pict_type == PictureType::B {
                let mod_l1 = r.read_bit()?;
                if mod_l1 {
                    for _ in 0..32 {
                        let idc = r.read_ue()?;
                        if idc >= 3 {
                            break;
                        }
                        let _ = r.read_ue()?;
                    }
                }
            }
        }

        // pred_weight_table
        if (pps.weighted_pred_flag && pict_type == PictureType::P)
            || (pps.weighted_bipred_idc == 1 && pict_type == PictureType::B)
        {
            let _luma_log2 = r.read_ue()?;
            if sps.chroma_format_idc != 0 {
                let _chroma_log2 = r.read_ue()?;
            }
            for _ in 0..=num_ref_idx_l0_active_minus1 {
                if r.read_bit()? {
                    let _ = r.read_se()?;
                    let _ = r.read_se()?;
                }
                if sps.chroma_format_idc != 0 && r.read_bit()? {
                    for _ in 0..2 {
                        let _ = r.read_se()?;
                        let _ = r.read_se()?;
                    }
                }
            }
            if pict_type == PictureType::B {
                for _ in 0..=num_ref_idx_l1_active_minus1 {
                    if r.read_bit()? {
                        let _ = r.read_se()?;
                        let _ = r.read_se()?;
                    }
                    if sps.chroma_format_idc != 0 && r.read_bit()? {
                        for _ in 0..2 {
                            let _ = r.read_se()?;
                            let _ = r.read_se()?;
                        }
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
                    for _ in 0..32 {
                        let mmco = r.read_ue()?;
                        if mmco == 0 || mmco > 6 {
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

        let mut cabac_init_idc = 0u8;
        if pps.entropy_coding_mode_flag && pict_type != PictureType::I {
            cabac_init_idc = r.read_ue()? as u8;
        }

        let slice_qp_delta = r.read_se()? as i8;
        let slice_qp =
            (26i32 + (pps.pic_init_qp_minus26 as i32) + (slice_qp_delta as i32)).clamp(0, 51) as u8;

        let mut disable_deblocking_filter_idc = 0u8;
        let mut slice_alpha_c0_offset_div2 = 0i8;
        let mut slice_beta_offset_div2 = 0i8;
        if pps.deblocking_filter_control_present_flag {
            disable_deblocking_filter_idc = r.read_ue()? as u8;
            if disable_deblocking_filter_idc != 1 {
                slice_alpha_c0_offset_div2 = r.read_se()? as i8;
                slice_beta_offset_div2 = r.read_se()? as i8;
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
            direct_spatial_mv_pred_flag,
            num_ref_idx_l0_active_minus1,
            num_ref_idx_l1_active_minus1,
            cabac_init_idc,
            slice_qp_delta,
            slice_qp,
            disable_deblocking_filter_idc,
            slice_alpha_c0_offset_div2,
            slice_beta_offset_div2,
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
        if self.pict_type == PictureType::B {
            w.write_bit(self.direct_spatial_mv_pred_flag);
        }
        if matches!(self.pict_type, PictureType::P | PictureType::B) {
            w.write_bit(false); // num_ref_idx_active_override_flag
            w.write_bit(false); // ref_pic_list_modification_flag_l0
            if self.pict_type == PictureType::B {
                w.write_bit(false); // ref_pic_list_modification_flag_l1
            }
        }
        if nal_header.nal_ref_idc != 0 {
            if nal_header.nal_unit_type == NalUnitType::IdrSlice {
                w.write_bit(false); // no_output_of_prior_pics_flag
                w.write_bit(false); // long_term_reference_flag
            } else {
                w.write_bit(false); // adaptive_ref_pic_marking_mode_flag
            }
        }
        if pps.entropy_coding_mode_flag && self.pict_type != PictureType::I {
            w.write_ue(self.cabac_init_idc as u64);
        }
        w.write_se(self.slice_qp_delta as i64);
        if pps.deblocking_filter_control_present_flag {
            w.write_ue(self.disable_deblocking_filter_idc as u64);
            if self.disable_deblocking_filter_idc != 1 {
                w.write_se(self.slice_alpha_c0_offset_div2 as i64);
                w.write_se(self.slice_beta_offset_div2 as i64);
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
        let c = had[i] as i64;
        let sign = if c < 0 { -1 } else { 1 };
        let level = ((c.abs() * mf + f) >> qbits) * sign;
        out[i] = level.clamp(i16::MIN as i64, i16::MAX as i64) as i16;
    }
    out
}

/// Forward $2\times2$ Chroma DC Hadamard transform + quantization at `qpc`.
pub fn quantize_chroma_dc2x2(dc_2x2: &[i16; 4], qpc: u8) -> [i16; 4] {
    let d0 = dc_2x2[0] as i64;
    let d1 = dc_2x2[1] as i64;
    let d2 = dc_2x2[2] as i64;
    let d3 = dc_2x2[3] as i64;
    let had = [
        d0 + d1 + d2 + d3,
        d0 - d1 + d2 - d3,
        d0 + d1 - d2 - d3,
        d0 - d1 - d2 + d3,
    ];
    let qpc = qpc.min(51) as usize;
    let qbits = 16 + (qpc / 6);
    let mf = QUANT4_MF[qpc % 6][0] as i64;
    let f = (1i64 << qbits) / 3;
    let mut out = [0i16; 4];
    for i in 0..4 {
        let c = had[i];
        let sign = if c < 0 { -1 } else { 1 };
        let level = ((c.abs() * mf + f) >> qbits) * sign;
        out[i] = level.clamp(i16::MIN as i64, i16::MAX as i64) as i16;
    }
    out
}

/// H.264 zig-zag scan order for $8\times8$ High Profile blocks (ITU-T H.264 Table 8-14).
pub const ZIGZAG_8X8: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// ITU-T H.264 Table 9-4 mapping `me(v)` Exp-Golomb `code_num` (`0..48`) to `Intra_4x4` `coded_block_pattern`.
pub const GOLOMB_TO_INTRA4X4_CBP: [u8; 48] = [
    47, 31, 15, 0, 23, 27, 29, 30, 7, 11, 13, 14, 39, 43, 45, 46, 16, 3, 5, 10, 12, 19, 21, 26, 28,
    35, 37, 42, 44, 1, 2, 4, 8, 17, 18, 20, 24, 6, 9, 22, 25, 32, 33, 34, 36, 40, 38, 41,
];

/// ITU-T H.264 Table 9-4 mapping `me(v)` Exp-Golomb `code_num` (`0..48`) to Inter `coded_block_pattern`.
pub const GOLOMB_TO_INTER_CBP: [u8; 48] = [
    0, 16, 1, 2, 4, 8, 32, 3, 5, 10, 12, 15, 47, 7, 11, 13, 14, 6, 9, 31, 35, 37, 42, 44, 33, 34,
    36, 40, 39, 43, 45, 46, 17, 18, 20, 24, 19, 21, 26, 28, 23, 27, 29, 30, 22, 25, 38, 41,
];

#[inline]
pub fn intra4x4_cbp_to_golomb(cbp: u8) -> u64 {
    GOLOMB_TO_INTRA4X4_CBP
        .iter()
        .position(|&v| v == cbp)
        .unwrap_or(3) as u64
}

#[inline]
pub fn inter_cbp_to_golomb(cbp: u8) -> u64 {
    GOLOMB_TO_INTER_CBP
        .iter()
        .position(|&v| v == cbp)
        .unwrap_or(0) as u64
}

/// ITU-T H.264 Table 8-17 $8\times8$ dequantization normAdjust `v_{m, i, j}` (`0040`).
pub const DEQUANT8_COEFF_INIT: [[i32; 6]; 6] = [
    [20, 18, 32, 19, 25, 24],
    [22, 19, 35, 21, 28, 26],
    [26, 23, 42, 24, 33, 31],
    [28, 25, 45, 26, 35, 33],
    [32, 28, 51, 30, 40, 38],
    [36, 32, 58, 34, 46, 43],
];

/// Forward $8\times8$ quantization multiplier table `x264_quant8_mf` (`0040`).
pub const QUANT8_MF: [[i32; 6]; 6] = [
    [13107, 7530, 20972, 9922, 16777, 12565],
    [11916, 7122, 19174, 8990, 14980, 11600],
    [10082, 6190, 15978, 7866, 12710, 9732],
    [9362, 5699, 14913, 7254, 11984, 9142],
    [8192, 5087, 13159, 6289, 10486, 7939],
    [7282, 4449, 11570, 5549, 9118, 7016],
];

#[inline]
const fn coeff8_pos_type(idx: usize) -> usize {
    let r = idx >> 3;
    let c = idx & 7;
    match (r & 3, c & 3) {
        (0, 0) => 0,
        (1 | 3, 1 | 3) => 1,
        (2, 2) => 2,
        (0, 1 | 3) | (1 | 3, 0) => 3,
        (0, 2) | (2, 0) => 4,
        _ => 5,
    }
}

/// Fused $8\times8$ residual subtraction (`src - pred`) and forward $8\times8$ DCT (`0040`).
#[inline]
pub fn sub8x8_dct_u8(src: &[u8], src_stride: usize, pred: &[u8], pred_stride: usize) -> [i16; 64] {
    let mut diff = [0i16; 64];
    for y in 0..8 {
        let rs = &src[y * src_stride..y * src_stride + 8];
        let rp = &pred[y * pred_stride..y * pred_stride + 8];
        for x in 0..8 {
            diff[y * 8 + x] = (rs[x] as i16) - (rp[x] as i16);
        }
    }
    forward_dct8x8(&diff)
}

/// Forward $8\times8$ High Profile quantization (`0040`).
pub fn quantize_block8x8(dct_coeffs: &[i16; 64], qp: u8, is_intra: bool) -> [i16; 64] {
    let qp = qp.min(51) as usize;
    let qbits = 16 + (qp / 6);
    let rem = qp % 6;
    let f = (1i64 << qbits) / if is_intra { 3 } else { 6 };
    let mut out = [0i16; 64];
    for i in 0..64 {
        let c = dct_coeffs[i] as i64;
        let sign = if c < 0 { -1 } else { 1 };
        let mf = QUANT8_MF[rem][coeff8_pos_type(i)] as i64;
        let level = ((c.abs() * mf + f) >> qbits) * sign;
        out[i] = level.clamp(i16::MIN as i64, i16::MAX as i64) as i16;
    }
    out
}

/// Inverse $8\times8$ High Profile dequantization (`ITU-T H.264 8.5.12.1`, default flat scaling matrix `16`).
pub fn dequantize_block8x8(levels: &[i16; 64], qp: u8) -> [i16; 64] {
    let qp = qp.min(51) as usize;
    let q_div = qp / 6;
    let rem = qp % 6;
    let mut out = [0i16; 64];
    for i in 0..64 {
        let v = DEQUANT8_COEFF_INIT[rem][coeff8_pos_type(i)];
        let d = if q_div >= 2 {
            (levels[i] as i32) * (v << (q_div - 2))
        } else {
            let r_shift = 2 - q_div;
            ((levels[i] as i32) * v + (1 << (r_shift - 1))) >> r_shift
        };
        out[i] = d.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
    out
}

/// H.264 High Profile Intra $8\times8$ spatial prediction with 3-tap `[1, 2, 1] >> 2` reference filtering (ITU-T H.264 8.3.2.2).
pub fn predict_intra_8x8_u8(
    mode: u8,
    top: Option<&[u8; 16]>,
    left: Option<&[u8; 8]>,
    top_left: Option<u8>,
    dst: &mut [u8],
    stride: usize,
) {
    // 1. Low-pass filter reference samples (`ff_h264_filter_mb_intra8x8`)
    let mut t = [128u8; 16];
    let mut l = [128u8; 8];
    let mut tl = top_left.unwrap_or(128);

    if let Some(raw_t) = top {
        let p_minus1 = match top_left {
            Some(v) => v as u16,
            None => raw_t[0] as u16,
        };
        t[0] = ((p_minus1 + 2 * (raw_t[0] as u16) + (raw_t[1] as u16) + 2) >> 2) as u8;
        for i in 1..15 {
            t[i] = (((raw_t[i - 1] as u16) + 2 * (raw_t[i] as u16) + (raw_t[i + 1] as u16) + 2)
                >> 2) as u8;
        }
        t[15] = (((raw_t[14] as u16) + 3 * (raw_t[15] as u16) + 2) >> 2) as u8;
    }
    if let Some(raw_l) = left {
        let p_minus1 = match top_left {
            Some(v) => v as u16,
            None => raw_l[0] as u16,
        };
        l[0] = ((p_minus1 + 2 * (raw_l[0] as u16) + (raw_l[1] as u16) + 2) >> 2) as u8;
        for i in 1..7 {
            l[i] = (((raw_l[i - 1] as u16) + 2 * (raw_l[i] as u16) + (raw_l[i + 1] as u16) + 2)
                >> 2) as u8;
        }
        l[7] = (((raw_l[6] as u16) + 3 * (raw_l[7] as u16) + 2) >> 2) as u8;
    }
    if let Some(raw_tl) = top_left {
        tl = match (top, left) {
            (Some(rt), Some(rl)) => {
                (((rt[0] as u16) + 2 * (raw_tl as u16) + (rl[0] as u16) + 2) >> 2) as u8
            }
            (Some(rt), None) => ((3 * (raw_tl as u16) + (rt[0] as u16) + 2) >> 2) as u8,
            (None, Some(rl)) => ((3 * (raw_tl as u16) + (rl[0] as u16) + 2) >> 2) as u8,
            (None, None) => raw_tl,
        };
    }

    match mode {
        0 => {
            for y in 0..8 {
                dst[y * stride..y * stride + 8].copy_from_slice(&t[0..8]);
            }
        }
        1 => {
            for y in 0..8 {
                dst[y * stride..y * stride + 8].fill(l[y]);
            }
        }
        2 => {
            let dc = match (top.is_some(), left.is_some()) {
                (true, true) => {
                    let s: u32 = t[0..8].iter().chain(l.iter()).map(|&v| v as u32).sum();
                    ((s + 8) >> 4) as u8
                }
                (true, false) => {
                    let s: u32 = t[0..8].iter().map(|&v| v as u32).sum();
                    ((s + 4) >> 3) as u8
                }
                (false, true) => {
                    let s: u32 = l.iter().map(|&v| v as u32).sum();
                    ((s + 4) >> 3) as u8
                }
                (false, false) => 128,
            };
            for y in 0..8 {
                dst[y * stride..y * stride + 8].fill(dc);
            }
        }
        3 => {
            // Diagonal Down-Left
            for y in 0..8 {
                for x in 0..8 {
                    let idx = x + y;
                    dst[y * stride + x] = if idx == 14 {
                        (((t[14] as u16) + 3 * (t[15] as u16) + 2) >> 2) as u8
                    } else {
                        (((t[idx] as u16) + 2 * (t[idx + 1] as u16) + (t[idx + 2] as u16) + 2) >> 2)
                            as u8
                    };
                }
            }
        }
        4 => {
            // Diagonal Down-Right
            let mut d = [0u8; 17];
            for i in 0..8 {
                d[i] = l[7 - i];
            }
            d[8] = tl;
            d[9..17].copy_from_slice(&t[0..8]);
            for y in 0..8 {
                for x in 0..8 {
                    let k = 8 + x - y;
                    dst[y * stride + x] =
                        (((d[k - 1] as u16) + 2 * (d[k] as u16) + (d[k + 1] as u16) + 2) >> 2)
                            as u8;
                }
            }
        }
        5 => {
            // Vertical-Right
            let mut p = [0u8; 17];
            for i in 0..8 {
                p[i] = l[7 - i];
            }
            p[8] = tl;
            p[9..17].copy_from_slice(&t[0..8]);
            for y in 0..8 {
                for x in 0..8 {
                    let z_vr = (2 * x as i32) - (y as i32);
                    dst[y * stride + x] = if z_vr >= 0 {
                        let idx = (x - (y >> 1)) + 8;
                        if (z_vr & 1) == 0 {
                            (((p[idx - 1] as u16) + (p[idx] as u16) + 1) >> 1) as u8
                        } else {
                            (((p[idx - 2] as u16) + 2 * (p[idx - 1] as u16) + (p[idx] as u16) + 2)
                                >> 2) as u8
                        }
                    } else if z_vr == -1 {
                        (((l[0] as u16) + 2 * (tl as u16) + (t[0] as u16) + 2) >> 2) as u8
                    } else {
                        let li = y - 2 * x - 1;
                        let a = l[li] as u16;
                        let b = if li >= 1 { l[li - 1] as u16 } else { tl as u16 };
                        let c = if li >= 2 { l[li - 2] as u16 } else { tl as u16 };
                        ((a + 2 * b + c + 2) >> 2) as u8
                    };
                }
            }
        }
        6 => {
            // Horizontal-Down
            let mut p = [0u8; 17];
            for i in 0..8 {
                p[i] = t[7 - i];
            }
            p[8] = tl;
            p[9..17].copy_from_slice(&l[0..8]);
            for y in 0..8 {
                for x in 0..8 {
                    let z_hd = (2 * y as i32) - (x as i32);
                    dst[y * stride + x] = if z_hd >= 0 {
                        let idx = (y - (x >> 1)) + 8;
                        if (z_hd & 1) == 0 {
                            (((p[idx - 1] as u16) + (p[idx] as u16) + 1) >> 1) as u8
                        } else {
                            (((p[idx - 2] as u16) + 2 * (p[idx - 1] as u16) + (p[idx] as u16) + 2)
                                >> 2) as u8
                        }
                    } else if z_hd == -1 {
                        (((l[0] as u16) + 2 * (tl as u16) + (t[0] as u16) + 2) >> 2) as u8
                    } else {
                        let ti = x - 2 * y - 1;
                        let a = t[ti] as u16;
                        let b = if ti >= 1 { t[ti - 1] as u16 } else { tl as u16 };
                        let c = if ti >= 2 { t[ti - 2] as u16 } else { tl as u16 };
                        ((a + 2 * b + c + 2) >> 2) as u8
                    };
                }
            }
        }
        7 => {
            // Vertical-Left
            for y in 0..8 {
                for x in 0..8 {
                    let idx = x + (y >> 1);
                    dst[y * stride + x] = if (y & 1) == 0 {
                        (((t[idx] as u16) + (t[idx + 1] as u16) + 1) >> 1) as u8
                    } else {
                        (((t[idx] as u16) + 2 * (t[idx + 1] as u16) + (t[idx + 2] as u16) + 2) >> 2)
                            as u8
                    };
                }
            }
        }
        _ => {
            // Horizontal-Up
            for y in 0..8 {
                for x in 0..8 {
                    let z_hu = x + 2 * y;
                    dst[y * stride + x] = if z_hu < 13 {
                        let idx = y + (x >> 1);
                        if (z_hu & 1) == 0 {
                            (((l[idx] as u16) + (l[idx + 1] as u16) + 1) >> 1) as u8
                        } else {
                            (((l[idx] as u16) + 2 * (l[idx + 1] as u16) + (l[idx + 2] as u16) + 2)
                                >> 2) as u8
                        }
                    } else if z_hu == 13 {
                        (((l[6] as u16) + 3 * (l[7] as u16) + 2) >> 2) as u8
                    } else {
                        l[7]
                    };
                }
            }
        }
    }
}

/// Adds an $8\times8$ IDCT residual block onto `dst` (`stride`).
pub fn add8x8_idct_u8(dst: &mut [u8], dst_stride: usize, coeff: &[i16; 64]) {
    let res = inverse_idct8x8(coeff);
    for y in 0..8 {
        let row = &mut dst[y * dst_stride..y * dst_stride + 8];
        for x in 0..8 {
            row[x] = clamp_u8((row[x] as i32) + (res[y * 8 + x] as i32));
        }
    }
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
fn predict_luma_nc(luma_nnz: &[u8], g_stride: usize, gx: usize, gy: usize) -> u8 {
    let left = if gx > 0 {
        Some(luma_nnz[gy * g_stride + (gx - 1)] as u16)
    } else {
        None
    };
    let top = if gy > 0 {
        Some(luma_nnz[(gy - 1) * g_stride + gx] as u16)
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

#[inline]
fn predict_chroma_nc(chroma_nnz: &[u8], cg_stride: usize, cgx: usize, cgy: usize) -> u8 {
    let left = if cgx > 0 {
        Some(chroma_nnz[cgy * cg_stride + (cgx - 1)] as u16)
    } else {
        None
    };
    let top = if cgy > 0 {
        Some(chroma_nnz[(cgy - 1) * cg_stride + cgx] as u16)
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

#[inline]
fn cabac_cbf_ctx_inc(left_cbf: Option<bool>, top_cbf: Option<bool>, mb_is_intra: bool) -> u8 {
    let cond_a = left_cbf.unwrap_or(mb_is_intra);
    let cond_b = top_cbf.unwrap_or(mb_is_intra);
    u8::from(cond_a) + (u8::from(cond_b) << 1)
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
        let code_u = level_code as u32;
        if code_u < (14 << suffix_length) {
            let level_prefix = code_u >> suffix_length;
            w.write_bits(0, level_prefix);
            w.write_bit(true);
            if suffix_length > 0 {
                let level_suffix = code_u & ((1 << suffix_length) - 1);
                w.write_bits(level_suffix as u64, suffix_length);
            }
        } else if suffix_length == 0 && code_u < 30 {
            w.write_bits(0, 14);
            w.write_bit(true);
            w.write_bits((code_u - 14) as u64, 4);
        } else if suffix_length > 0 && code_u < (15 << suffix_length) {
            w.write_bits(0, 14);
            w.write_bit(true);
            let level_suffix = code_u - (14 << suffix_length);
            w.write_bits(level_suffix as u64, suffix_length);
        } else {
            let mut esc_suffix = code_u - (15 << suffix_length);
            if suffix_length == 0 {
                esc_suffix -= 15;
            }
            w.write_bits(0, 15);
            w.write_bit(true);
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
        let level_suffix_size = if level_prefix == 14 && suffix_length == 0 {
            4
        } else if level_prefix >= 15 {
            level_prefix - 3
        } else {
            suffix_length
        };
        let level_suffix = if level_suffix_size > 0 {
            r.read_bits(level_suffix_size)? as u32
        } else {
            0
        };
        let mut code_u = (level_prefix.min(15) << suffix_length) + level_suffix;
        if level_prefix >= 15 && suffix_length == 0 {
            code_u += 15;
        }
        if level_prefix >= 16 {
            code_u += (1u32 << (level_prefix - 3)) - 4096;
        }
        let mut level_code = code_u as i32;

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

/// Pure-Rust H.264 / AVC Baseline (`66`), Main (`77`), and High (`100`) video encoder (`0039`, `0040`, `0041`).
///
/// Produces standard ITU-T H.264 Annex-B bitstreams (`SPS` + `PPS` + `IDR` / `P` slices) with:
/// - Full `I_4x4` (9 spatial modes + 16 $4\times4$ DCT blocks), `I_16x16` (4 spatial modes + $4\times4$ DC Hadamard + 16 AC blocks),
///   and `P_L0_16x16` / `P_Skip` inter macroblock coding with 6-tap quarter-pel motion refinement (`0%` `I_PCM` for `qp >= 1`).
/// - Both CAVLC (`Baseline`) and CABAC (`Main` / `High`) entropy coding (`10%–18%` smaller bitstreams at identical PSNR/SSIM).
/// - Adaptive High Profile $8\times8$ integer DCT transform (`transform_8x8_mode_flag = 1`) and in-loop deblocking filter.
/// - Perceptual Constant Rate Factor (`RateControlMode::Crf`), Variance-Adaptive Quantization (`VAQ`), and VBV/ABR rate control.
#[derive(Clone, Debug)]
pub struct H264Encoder {
    pub sps: Sps,
    pub pps: Pps,
    pub profile: H264Profile,
    pub rate_control: RateControlMode,
    pub qp: u8,
    pub keyint: u32,
    frame_count: u32,
    prev_recon: Option<VideoFrame>,
    vbv_buffer_bits: f64,
    total_bits_emitted: u64,
    abr_qp_accum: f32,
}

impl H264Encoder {
    pub fn new(width: u32, height: u32, fps: Rational64, qp: u8) -> Self {
        let qp = qp.clamp(0, 51);
        let profile = H264Profile::Baseline;
        let sps = Sps::new_profile(width, height, fps, profile);
        let pps = Pps::new_profile(qp, profile);
        Self {
            sps,
            pps,
            profile,
            rate_control: RateControlMode::FixedQp(qp),
            qp,
            keyint: 30,
            frame_count: 0,
            prev_recon: None,
            vbv_buffer_bits: 0.0,
            total_bits_emitted: 0,
            abr_qp_accum: qp as f32,
        }
    }

    /// Configures the H.264 coding profile (`Baseline` CAVLC, `Main` CABAC, or `High` CABAC + $8\times8$ transform).
    pub fn with_profile(mut self, profile: H264Profile) -> Self {
        self.profile = profile;
        let fps = self.sps.fps.unwrap_or(Rational64::FPS_30);
        let c = self.sps.colorimetry;
        self.sps = Sps::new_profile(self.sps.width(), self.sps.height(), fps, profile);
        self.sps.colorimetry = c;
        self.pps = Pps::new_profile(self.qp, profile);
        self
    }

    /// Configures perceptual CRF or target ABR / VBV rate control (`0041`).
    pub fn with_rate_control(mut self, rc: RateControlMode) -> Self {
        self.rate_control = rc;
        match rc {
            RateControlMode::FixedQp(q) => {
                self.qp = q.clamp(0, 51);
            }
            RateControlMode::Crf {
                crf, bufsize_kb, ..
            } => {
                self.qp = crf.round().clamp(0.0, 51.0) as u8;
                if let Some(buf) = bufsize_kb {
                    self.vbv_buffer_bits = (buf as f64) * 8000.0 * 0.8;
                }
            }
            RateControlMode::Abr {
                bitrate_kbps,
                bufsize_kb,
                ..
            } => {
                let pixels = (self.sps.width() as f64) * (self.sps.height() as f64);
                let fps = self.sps.fps.unwrap_or(Rational64::FPS_30).as_f64().max(1.0);
                let bpp = ((bitrate_kbps as f64) * 1000.0) / (pixels * fps).max(1.0);
                // Empirical initial QP from bits-per-pixel
                let init_qp = (28.0 - 6.0 * (bpp / 0.08).max(0.05).log2()).clamp(10.0, 48.0);
                self.qp = init_qp.round() as u8;
                self.abr_qp_accum = init_qp as f32;
                let buf = bufsize_kb.unwrap_or(bitrate_kbps * 2);
                self.vbv_buffer_bits = (buf as f64) * 8000.0 * 0.5;
            }
        }
        self.pps = Pps::new_profile(self.qp, self.profile);
        self
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

        let mb_w = self.sps.pic_width_in_mbs as usize;
        let mb_h = self.sps.pic_height_in_map_units as usize;
        let total_mbs = mb_w * mb_h;
        let fps = self.sps.fps.unwrap_or(Rational64::FPS_30).as_f64().max(1.0);

        // Pre-compute per-MB spatial variance and geometric mean log-variance for CRF / VAQ / ABR rate control
        let mut mb_vars = vec![0.0f32; total_mbs];
        let mut sum_log_var = 0.0f32;
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let mut sy = [0u8; 256];
                let mut su = [0u8; 64];
                let mut sv = [0u8; 64];
                extract_mb_planes(frame, mb_x * 16, mb_y * 16, &mut sy, &mut su, &mut sv);
                let v = compute_mb_luma_variance(&sy);
                mb_vars[mb_y * mb_w + mb_x] = v;
                sum_log_var += v.max(1.0).log2();
            }
        }
        let avg_log_var = sum_log_var / (total_mbs.max(1) as f32);

        // Determine frame slice QP from RateControlMode
        let slice_qp = match self.rate_control {
            RateControlMode::FixedQp(q) => q.clamp(0, 51),
            RateControlMode::Crf {
                crf,
                maxrate_kbps,
                bufsize_kb,
            } => {
                // Keyframes anchor the GOP and receive a -3 QP boost; P-frames get +1 offset so
                // average bitrate matches fixed QP while perceptual multi-scale SSIM increases
                let mut q = if is_idr { crf - 3.0 } else { crf + 1.0 };
                if let Some(maxrate) = maxrate_kbps {
                    let max_bits_per_frame = (maxrate as f64) * 1000.0 / fps;
                    let cap = (bufsize_kb.unwrap_or(maxrate * 2) as f64) * 8000.0;
                    if cap > 0.0 && self.vbv_buffer_bits < max_bits_per_frame * 2.0 {
                        q += 4.0;
                    }
                }
                q.round().clamp(0.0, 51.0) as u8
            }
            RateControlMode::Abr {
                bitrate_kbps,
                maxrate_kbps,
                ..
            } => {
                let target_frame_bits = (bitrate_kbps as f64) * 1000.0 / fps;
                if self.frame_count == 0 {
                    let bpp = target_frame_bits / ((total_mbs * 256).max(1) as f64);
                    let init_q = (36.0 - (bpp / 0.05).max(0.05).log2() * 6.0).clamp(18.0, 48.0);
                    self.abr_qp_accum = init_q as f32;
                } else {
                    let expected_bits = target_frame_bits * (self.frame_count as f64);
                    let ratio = (self.total_bits_emitted as f64) / expected_bits.max(1.0);
                    let delta = ((ratio.clamp(0.25, 4.0)).log2() * 6.0) as f32;
                    self.abr_qp_accum = (self.abr_qp_accum + delta * 0.5).clamp(1.0, 51.0);
                }
                let mut q = self.abr_qp_accum;
                if let Some(maxrate) = maxrate_kbps {
                    let max_frame_bits = (maxrate as f64) * 1000.0 / fps;
                    if self.vbv_buffer_bits > max_frame_bits * 4.0 {
                        q += 3.0;
                    }
                }
                q.round().clamp(1.0, 51.0) as u8
            }
        };

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

        let mut recon = VideoFrame::new(mb_w * 16, mb_h * 16, PixelFormat::Yuv420p);
        let mut w = BitWriter::with_capacity(frame.width * frame.height + 256);
        slice_header.write(&mut w, nal_header, &self.sps, &self.pps);

        let use_cabac = self.pps.entropy_coding_mode_flag;
        let mut cabac_enc = if use_cabac {
            w.align_to_byte();
            Some((
                CabacEncoder::new(),
                init_cabac_contexts(slice_qp, is_idr, slice_header.cabac_init_idc),
            ))
        } else {
            None
        };

        // Per-4x4-block state grids (`g_stride = mb_w * 4`, `cg_stride = mb_w * 2`)
        let g_stride = mb_w * 4;
        let g_h = mb_h * 4;
        let cg_stride = mb_w * 2;
        let cg_h = mb_h * 2;
        let mut luma_nnz = vec![0u8; g_stride * g_h];
        let mut chroma_nnz = [vec![0u8; cg_stride * cg_h], vec![0u8; cg_stride * cg_h]];
        let mut intra4x4_modes = vec![2u8; g_stride * g_h];
        let mut ref_grid = vec![-2i8; g_stride * g_h];
        let mut mv_grid = vec![[0i16; 2]; g_stride * g_h];
        let mut mvd_grid = vec![[0i16; 2]; g_stride * g_h];

        // Per-MB state arrays for CABAC context derivation & deblocking
        let mut mb_is_intra = vec![false; total_mbs];
        let mut mb_is_i16x16 = vec![false; total_mbs];
        let mut mb_is_skip = vec![false; total_mbs];
        let mut mb_cbp_luma = vec![0u8; total_mbs];
        let mut mb_cbp_chroma = vec![0u8; total_mbs];
        let mut mb_chroma_mode = vec![0u8; total_mbs];
        let mut mb_luma_dc_cbf = vec![false; total_mbs];
        let mut mb_chroma_dc_cbf = [vec![false; total_mbs], vec![false; total_mbs]];
        let mut mb_8x8_flag = vec![false; total_mbs];
        let mut mb_qps = vec![slice_qp; total_mbs];

        let mut skip_run = 0u32;
        let mut cur_qp = slice_qp as i32;
        let mut prev_mb_qp_delta_nz = false;

        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let mb_idx = mb_y * mb_w + mb_x;
                let x0 = mb_x * 16;
                let y0 = mb_y * 16;
                let cx0 = mb_x * 8;
                let cy0 = mb_y * 8;

                // Compute target MB QP with Variance-Adaptive Quantization (VAQ) or ABR row feedback
                let target_mb_qp = match self.rate_control {
                    RateControlMode::FixedQp(_) => slice_qp,
                    RateControlMode::Crf { .. } => {
                        let log_var = mb_vars[mb_idx].max(1.0).log2();
                        // Auto-variance VAQ around geometric mean variance (`x264`-style `ac_energy` log-domain):
                        // smooth/mid-variance structural regions get negative QP delta (boosting MS-SSIM where
                        // variance masking is weak), while high-entropy masked textures get positive QP delta.
                        let vaq_delta =
                            ((log_var - avg_log_var) * 2.5).round().clamp(-8.0, 8.0) as i32;
                        ((slice_qp as i32) + vaq_delta).clamp(1, 51) as u8
                    }
                    RateControlMode::Abr { bitrate_kbps, .. } => {
                        let target_frame_bits = (bitrate_kbps as f64) * 1000.0 / fps;
                        let frac_done = (mb_idx as f64) / (total_mbs.max(1) as f64);
                        if frac_done > 0.02 {
                            let cur_bits = if let Some((ref ce, _)) = cabac_enc {
                                (ce.bytes_written() * 8) as f64
                            } else {
                                (w.bits_written()) as f64
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

                let mut src_y = [0u8; 256];
                let mut src_u = [0u8; 64];
                let mut src_v = [0u8; 64];
                extract_mb_planes(frame, x0, y0, &mut src_y, &mut src_u, &mut src_v);

                // Lossless bypass only when QP == 0
                if target_mb_qp == 0 {
                    if !is_idr && !use_cabac {
                        w.write_ue(skip_run as u64);
                        skip_run = 0;
                    }
                    if let Some((ref mut ce, ref mut ctxs)) = cabac_enc {
                        if !is_idr {
                            let l_cond = u8::from(mb_x > 0 && !mb_is_skip[mb_idx - 1]);
                            let t_cond = u8::from(mb_y > 0 && !mb_is_skip[mb_idx - mb_w]);
                            ce.encode_p_skip(ctxs, l_cond, t_cond, false);
                            ce.encode_p_mb_type(ctxs, 30); // I_PCM in P-slice (5 + 25)
                        } else {
                            let l_not_i4 = u8::from(mb_x > 0 && mb_is_i16x16[mb_idx - 1]);
                            let t_not_i4 = u8::from(mb_y > 0 && mb_is_i16x16[mb_idx - mb_w]);
                            ce.encode_intra_mb_type(ctxs, 3, l_not_i4, t_not_i4, 25);
                        }
                        for &b in src_y.iter().chain(src_u.iter()).chain(src_v.iter()) {
                            ce.write_raw_byte(b);
                        }
                        ce.reinit_after_pcm();
                        let is_last = mb_idx + 1 == total_mbs;
                        ce.encode_terminate(u8::from(is_last));
                    } else {
                        let intra_offset = if is_idr { 0u64 } else { 5u64 };
                        w.write_ue(intra_offset + 25);
                        w.align_to_byte();
                        for &b in src_y.iter().chain(src_u.iter()).chain(src_v.iter()) {
                            w.write_bits(b as u64, 8);
                        }
                    }
                    for by in 0..4 {
                        for bx in 0..4 {
                            let g = (mb_y * 4 + by) * g_stride + (mb_x * 4 + bx);
                            luma_nnz[g] = 16;
                            intra4x4_modes[g] = 2;
                            ref_grid[g] = -1;
                        }
                    }
                    for by in 0..2 {
                        for bx in 0..2 {
                            let cg = (mb_y * 2 + by) * cg_stride + (mb_x * 2 + bx);
                            chroma_nnz[0][cg] = 16;
                            chroma_nnz[1][cg] = 16;
                        }
                    }
                    mb_is_intra[mb_idx] = true;
                    mb_is_i16x16[mb_idx] = true;
                    mb_cbp_luma[mb_idx] = 15;
                    mb_cbp_chroma[mb_idx] = 2;
                    mb_luma_dc_cbf[mb_idx] = true;
                    mb_chroma_dc_cbf[0][mb_idx] = true;
                    mb_chroma_dc_cbf[1][mb_idx] = true;
                    mb_qps[mb_idx] = cur_qp as u8;
                    prev_mb_qp_delta_nz = false;
                    store_mb_planes(&mut recon, x0, y0, &src_y, &src_u, &src_v);
                    continue;
                }

                // 1. In P-slices, evaluate P_Skip and motion-compensated P_L0_16x16 (with qpel refinement)
                if !is_idr && let Some(prev) = &self.prev_recon {
                    let gx0 = mb_x * 4;
                    let gy0 = mb_y * 4;
                    let pskip_mv = pred_pskip_motion_grid(&ref_grid, &mv_grid, g_stride, gx0, gy0);
                    let mvp = pred_motion_partition(&ref_grid, &mv_grid, g_stride, gx0, gy0, 4, 0);

                    let mut ref_y = [0u8; 256];
                    let mut ref_u = [0u8; 64];
                    let mut ref_v = [0u8; 64];
                    extract_inter_mb_planes(
                        prev, x0, y0, pskip_mv, &mut ref_y, &mut ref_u, &mut ref_v,
                    );
                    let pskip_sad_y = sad_block_256_u8(&src_y, &ref_y);
                    let pskip_sad_uv =
                        sad_block_64_u8(&src_u, &ref_u) + sad_block_64_u8(&src_v, &ref_v);

                    let skip_thresh =
                        ((target_mb_qp as u32) * 5).max(12u32 << (target_mb_qp / 6).min(8));
                    if pskip_sad_y <= skip_thresh && pskip_sad_uv <= skip_thresh / 2 {
                        if let Some((ref mut ce, ref mut ctxs)) = cabac_enc {
                            let l_cond = u8::from(mb_x > 0 && !mb_is_skip[mb_idx - 1]);
                            let t_cond = u8::from(mb_y > 0 && !mb_is_skip[mb_idx - mb_w]);
                            ce.encode_p_skip(ctxs, l_cond, t_cond, true);
                            let is_last = mb_idx + 1 == total_mbs;
                            ce.encode_terminate(u8::from(is_last));
                        } else {
                            skip_run += 1;
                        }
                        fill_grid_4x4(&mut ref_grid, g_stride, gx0, gy0, 4, 4, 0);
                        fill_grid_4x4(&mut mv_grid, g_stride, gx0, gy0, 4, 4, pskip_mv);
                        mb_is_skip[mb_idx] = true;
                        mb_qps[mb_idx] = cur_qp as u8;
                        prev_mb_qp_delta_nz = false;
                        store_mb_planes(&mut recon, x0, y0, &ref_y, &ref_u, &ref_v);
                        continue;
                    }

                    // Integer diamond search + 6-tap half-pel / quarter-pel refinement
                    let int_mv = search_inter_motion_16x16(prev, x0, y0, &src_y, mvp, skip_thresh);
                    let best_mv = search_qpel_refinement_16x16(
                        prev,
                        x0,
                        y0,
                        &src_y,
                        int_mv,
                        mvp,
                        target_mb_qp,
                    );
                    extract_inter_mb_planes(
                        prev, x0, y0, best_mv, &mut ref_y, &mut ref_u, &mut ref_v,
                    );
                    let inter_sad = sad_block_256_u8(&src_y, &ref_y);

                    // Use P_L0_16x16 unless the inter prediction error is huge (scene cut)
                    if inter_sad < 6400 {
                        let mut luma_q4 = [[0i16; 16]; 16];
                        let mut luma_q8 = [[0i16; 64]; 4];
                        let mut cbp_luma_4 = 0u8;
                        let mut nnz4 = 0usize;
                        for (b, q_slot) in luma_q4.iter_mut().enumerate() {
                            let bx = ((b & 1) + ((b >> 1) & 2)) * 4;
                            let by = (((b >> 1) & 1) + ((b >> 2) & 2)) * 4;
                            let dct = sub4x4_dct_u8(
                                &src_y[by * 16 + bx..],
                                16,
                                &ref_y[by * 16 + bx..],
                                16,
                            );
                            let q = quantize_block4x4(&dct, target_mb_qp, false);
                            let blk_nnz = q.iter().filter(|&&c| c != 0).count();
                            if blk_nnz > 0 {
                                cbp_luma_4 |= 1 << (b >> 2);
                                nnz4 += blk_nnz;
                            }
                            *q_slot = q;
                        }

                        let mut use_8x8_inter = false;
                        let mut cbp_luma = cbp_luma_4;
                        if self.pps.transform_8x8_mode_flag && use_cabac && cbp_luma_4 != 0 {
                            let mut cbp_luma_8 = 0u8;
                            let mut nnz8 = 0usize;
                            for (i8x8, q8_slot) in luma_q8.iter_mut().enumerate() {
                                let bx8 = (i8x8 & 1) * 8;
                                let by8 = (i8x8 >> 1) * 8;
                                let dct8 = sub8x8_dct_u8(
                                    &src_y[by8 * 16 + bx8..],
                                    16,
                                    &ref_y[by8 * 16 + bx8..],
                                    16,
                                );
                                let q8 = quantize_block8x8(&dct8, target_mb_qp, false);
                                let blk_nnz = q8.iter().filter(|&&c| c != 0).count();
                                if blk_nnz > 0 {
                                    cbp_luma_8 |= 1 << i8x8;
                                    nnz8 += blk_nnz;
                                }
                                *q8_slot = q8;
                            }
                            if nnz8 <= nnz4 {
                                use_8x8_inter = true;
                                cbp_luma = cbp_luma_8;
                            }
                        }

                        let qpc_idx = (target_mb_qp as i32
                            + (self.pps.chroma_qp_index_offset as i32))
                            .clamp(0, 51) as usize;
                        let qpc = CHROMA_QP_TABLE[qpc_idx];
                        let (chroma_dc_q, chroma_ac_q, cbp_chroma) = quantize_chroma_8x8_residual(
                            &src_u, &src_v, &ref_u, &ref_v, qpc, false,
                        );

                        // If all residuals quantized to zero and MV matches P_Skip MV, emit P_Skip!
                        if cbp_luma == 0 && cbp_chroma == 0 && best_mv == pskip_mv {
                            if let Some((ref mut ce, ref mut ctxs)) = cabac_enc {
                                let l_cond = u8::from(mb_x > 0 && !mb_is_skip[mb_idx - 1]);
                                let t_cond = u8::from(mb_y > 0 && !mb_is_skip[mb_idx - mb_w]);
                                ce.encode_p_skip(ctxs, l_cond, t_cond, true);
                                let is_last = mb_idx + 1 == total_mbs;
                                ce.encode_terminate(u8::from(is_last));
                            } else {
                                skip_run += 1;
                            }
                            fill_grid_4x4(&mut ref_grid, g_stride, gx0, gy0, 4, 4, 0);
                            fill_grid_4x4(&mut mv_grid, g_stride, gx0, gy0, 4, 4, pskip_mv);
                            mb_is_skip[mb_idx] = true;
                            mb_qps[mb_idx] = cur_qp as u8;
                            prev_mb_qp_delta_nz = false;
                            store_mb_planes(&mut recon, x0, y0, &ref_y, &ref_u, &ref_v);
                            continue;
                        }

                        let cbp = cbp_luma | (cbp_chroma << 4);
                        let mvd = [best_mv[0] - mvp[0], best_mv[1] - mvp[1]];
                        let qp_delta = if cbp != 0 {
                            (target_mb_qp as i32) - cur_qp
                        } else {
                            0
                        };
                        let eff_qp = (cur_qp + qp_delta).clamp(0, 51) as u8;
                        let eff_qpc = CHROMA_QP_TABLE[(eff_qp as i32
                            + (self.pps.chroma_qp_index_offset as i32))
                            .clamp(0, 51)
                            as usize];

                        if let Some((ref mut ce, ref mut ctxs)) = cabac_enc {
                            let l_cond = u8::from(mb_x > 0 && !mb_is_skip[mb_idx - 1]);
                            let t_cond = u8::from(mb_y > 0 && !mb_is_skip[mb_idx - mb_w]);
                            ce.encode_p_skip(ctxs, l_cond, t_cond, false);
                            ce.encode_p_mb_type(ctxs, 0); // P_L0_16x16
                            let (amvd_x, amvd_y) =
                                neighbor_mvd_sum_16x16(&mvd_grid, g_stride, gx0, gy0);
                            ce.encode_mvd(ctxs, 0, amvd_x, mvd[0]);
                            ce.encode_mvd(ctxs, 1, amvd_y, mvd[1]);
                            if use_8x8_inter && cbp_luma != 0 {
                                mb_8x8_flag[mb_idx] = true;
                            }
                            encode_cabac_cbp_and_residuals(
                                ce,
                                ctxs,
                                mb_x,
                                mb_y,
                                mb_w,
                                g_stride,
                                cg_stride,
                                false,
                                self.pps.transform_8x8_mode_flag,
                                cbp_luma,
                                cbp_chroma,
                                qp_delta,
                                &mut prev_mb_qp_delta_nz,
                                &luma_q4,
                                if use_8x8_inter { Some(&luma_q8) } else { None },
                                &chroma_dc_q,
                                &chroma_ac_q,
                                &mut luma_nnz,
                                &mut chroma_nnz,
                                &mb_is_intra,
                                &mb_is_i16x16,
                                &mb_is_skip,
                                &mb_8x8_flag,
                                &mut mb_cbp_luma,
                                &mut mb_cbp_chroma,
                                &mut mb_chroma_dc_cbf,
                            );
                            let is_last = mb_idx + 1 == total_mbs;
                            ce.encode_terminate(u8::from(is_last));
                        } else {
                            w.write_ue(skip_run as u64);
                            skip_run = 0;
                            w.write_ue(0); // P_L0_16x16
                            w.write_se(mvd[0] as i64);
                            w.write_se(mvd[1] as i64);
                            w.write_ue(inter_cbp_to_golomb(cbp));
                            if cbp != 0 {
                                if self.pps.transform_8x8_mode_flag {
                                    w.write_bit(false);
                                }
                                w.write_se(qp_delta as i64);
                                encode_cavlc_4x4_and_chroma_residuals(
                                    &mut w,
                                    mb_x,
                                    mb_y,
                                    g_stride,
                                    cg_stride,
                                    cbp_luma,
                                    cbp_chroma,
                                    &luma_q4,
                                    &chroma_dc_q,
                                    &chroma_ac_q,
                                    &mut luma_nnz,
                                    &mut chroma_nnz,
                                );
                            }
                        }

                        cur_qp = eff_qp as i32;
                        mb_qps[mb_idx] = eff_qp;
                        if use_8x8_inter {
                            for (i8x8, q8) in luma_q8.iter().enumerate() {
                                if (cbp_luma & (1 << i8x8)) != 0 && q8.iter().any(|&c| c != 0) {
                                    let bx8 = (i8x8 & 1) * 8;
                                    let by8 = (i8x8 >> 1) * 8;
                                    let deq8 = dequantize_block8x8(q8, eff_qp);
                                    add8x8_idct_u8(&mut ref_y[by8 * 16 + bx8..], 16, &deq8);
                                }
                            }
                            reconstruct_4x4_and_chroma_residuals(
                                &mut ref_y,
                                &mut ref_u,
                                &mut ref_v,
                                0,
                                cbp_chroma,
                                eff_qp,
                                eff_qpc,
                                &luma_q4,
                                &chroma_dc_q,
                                &chroma_ac_q,
                            );
                        } else {
                            reconstruct_4x4_and_chroma_residuals(
                                &mut ref_y,
                                &mut ref_u,
                                &mut ref_v,
                                cbp_luma,
                                cbp_chroma,
                                eff_qp,
                                eff_qpc,
                                &luma_q4,
                                &chroma_dc_q,
                                &chroma_ac_q,
                            );
                        }
                        fill_grid_4x4(&mut ref_grid, g_stride, gx0, gy0, 4, 4, 0);
                        fill_grid_4x4(&mut mv_grid, g_stride, gx0, gy0, 4, 4, best_mv);
                        fill_grid_4x4(&mut mvd_grid, g_stride, gx0, gy0, 4, 4, mvd);
                        store_mb_planes(&mut recon, x0, y0, &ref_y, &ref_u, &ref_v);
                        continue;
                    }
                }

                // 2. Intra macroblock coding (`I_4x4` vs `I_16x16` mode decision with full AC/DC residuals)
                if !is_idr && !use_cabac {
                    w.write_ue(skip_run as u64);
                    skip_run = 0;
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

                // Best 8x8 chroma prediction mode (0=DC, 1=Horiz, 2=Vert, 3=Plane)
                let mut best_chroma_mode = 0u8;
                let mut best_pred_u = [128u8; 64];
                let mut best_pred_v = [128u8; 64];
                predict_intra_chroma_8x8(
                    0,
                    top_u.as_ref(),
                    left_u.as_ref(),
                    tl_u,
                    &mut best_pred_u,
                    8,
                );
                predict_intra_chroma_8x8(
                    0,
                    top_v.as_ref(),
                    left_v.as_ref(),
                    tl_v,
                    &mut best_pred_v,
                    8,
                );
                let mut best_sad_uv =
                    sad_block_64_u8(&src_u, &best_pred_u) + sad_block_64_u8(&src_v, &best_pred_v);
                for cm in [1u8, 2, 3] {
                    if (cm == 1 && left_u.is_none())
                        || (cm == 2 && top_u.is_none())
                        || (cm == 3 && (top_u.is_none() || left_u.is_none()))
                    {
                        continue;
                    }
                    let mut cu = [0u8; 64];
                    let mut cv = [0u8; 64];
                    predict_intra_chroma_8x8(cm, top_u.as_ref(), left_u.as_ref(), tl_u, &mut cu, 8);
                    predict_intra_chroma_8x8(cm, top_v.as_ref(), left_v.as_ref(), tl_v, &mut cv, 8);
                    let sad = sad_block_64_u8(&src_u, &cu) + sad_block_64_u8(&src_v, &cv);
                    if sad + 8 < best_sad_uv {
                        best_sad_uv = sad;
                        best_chroma_mode = cm;
                        best_pred_u = cu;
                        best_pred_v = cv;
                    }
                }

                // Evaluate I_16x16 across valid modes (0=Vert, 1=Horiz, 2=DC, 3=Plane)
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

                let mut best_i16_mode = 2u8;
                let mut best_i16_pred = [128u8; 256];
                predict_intra_16x16(
                    2,
                    top_y.as_ref(),
                    left_y.as_ref(),
                    tl_y,
                    &mut best_i16_pred,
                    16,
                );
                let mut best_i16_sad = sad_block_256_u8(&src_y, &best_i16_pred);
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
                    if sad < best_i16_sad {
                        best_i16_sad = sad;
                        best_i16_mode = cand_mode;
                        best_i16_pred = cand_pred;
                    }
                }

                // Evaluate I_4x4 (in-loop 16-block Z-scan prediction + quantization) when variance is high
                let try_i4x4 = target_mb_qp <= 30
                    && mb_vars[mb_idx] > 180.0
                    && best_i16_sad > (target_mb_qp as u32) * 14;
                let mut use_i4x4 = false;
                let mut i4_modes = [2u8; 16];
                let mut i4_q4 = [[0i16; 16]; 16];
                let mut i4_cbp_luma = 0u8;
                let mut i4_recon_y = [0u8; 256];

                if try_i4x4 {
                    let qp_delta_i4 = (target_mb_qp as i32) - cur_qp;
                    let eff_qp_i4 = (cur_qp + qp_delta_i4).clamp(0, 51) as u8;
                    let mut i4_sad_total = 0u32;
                    let mut saved_modes = [2u8; 16];
                    for (b, slot) in saved_modes.iter_mut().enumerate() {
                        let bx = (b & 1) + ((b >> 1) & 2);
                        let by = ((b >> 1) & 1) + ((b >> 2) & 2);
                        let gx = mb_x * 4 + bx;
                        let gy = mb_y * 4 + by;
                        *slot = intra4x4_modes[gy * g_stride + gx];
                    }

                    for b in 0..16 {
                        let bx = (b & 1) + ((b >> 1) & 2);
                        let by = ((b >> 1) & 1) + ((b >> 2) & 2);
                        let px = x0 + bx * 4;
                        let py = y0 + by * 4;
                        let gx = mb_x * 4 + bx;
                        let gy = mb_y * 4 + by;

                        let (top8, left4, tl) =
                            gather_intra4x4_neighbors(&recon, px, py, b, mb_x, mb_w);
                        let pred_mode = pred_intra4x4_mode(&intra4x4_modes, g_stride, gx, gy);

                        let mut best_m = 2u8;
                        let mut best_blk_pred = [128u8; 16];
                        predict_intra_4x4_u8(
                            2,
                            top8.as_ref(),
                            left4.as_ref(),
                            tl,
                            &mut best_blk_pred,
                            4,
                        );
                        let mut src_blk = [0u8; 16];
                        for r in 0..4 {
                            src_blk[r * 4..r * 4 + 4].copy_from_slice(
                                &src_y[(by * 4 + r) * 16 + bx * 4..(by * 4 + r) * 16 + bx * 4 + 4],
                            );
                        }
                        let mut best_cost = sad_4x4_arr(&src_blk, &best_blk_pred)
                            + if 2 == pred_mode {
                                0
                            } else {
                                target_mb_qp as u32 / 2
                            };

                        for m in 0..=8u8 {
                            if m == 2
                                || !intra4x4_mode_available(m, top8.is_some(), left4.is_some())
                            {
                                continue;
                            }
                            let mut cand_blk = [0u8; 16];
                            predict_intra_4x4_u8(
                                m,
                                top8.as_ref(),
                                left4.as_ref(),
                                tl,
                                &mut cand_blk,
                                4,
                            );
                            let s = sad_4x4_arr(&src_blk, &cand_blk)
                                + if m == pred_mode {
                                    0
                                } else {
                                    target_mb_qp as u32 / 2
                                };
                            if s < best_cost {
                                best_cost = s;
                                best_m = m;
                                best_blk_pred = cand_blk;
                            }
                        }

                        i4_modes[b] = best_m;
                        intra4x4_modes[gy * g_stride + gx] = best_m;
                        i4_sad_total += best_cost;

                        let dct = sub4x4_dct_u8(&src_blk, 4, &best_blk_pred, 4);
                        let q = quantize_block4x4(&dct, eff_qp_i4, true);
                        let mut rec_blk = best_blk_pred;
                        if q.iter().any(|&c| c != 0) {
                            i4_cbp_luma |= 1 << (b >> 2);
                            let deq = dequantize_block4x4(&q, eff_qp_i4);
                            add4x4_idct_u8(&mut rec_blk, 4, &deq);
                        }
                        i4_q4[b] = q;
                        for r in 0..4 {
                            recon.plane_mut(0).row_mut(py + r)[px..px + 4]
                                .copy_from_slice(&rec_blk[r * 4..r * 4 + 4]);
                            i4_recon_y[(by * 4 + r) * 16 + bx * 4..(by * 4 + r) * 16 + bx * 4 + 4]
                                .copy_from_slice(&rec_blk[r * 4..r * 4 + 4]);
                        }
                    }

                    // Mode decision: pick I_4x4 if its prediction cost + 16-mode signaling overhead beats I_16x16
                    if i4_sad_total + (target_mb_qp as u32) * 6 < best_i16_sad {
                        use_i4x4 = true;
                    } else {
                        for (b, &sm) in saved_modes.iter().enumerate() {
                            let bx = (b & 1) + ((b >> 1) & 2);
                            let by = ((b >> 1) & 1) + ((b >> 2) & 2);
                            let gx = mb_x * 4 + bx;
                            let gy = mb_y * 4 + by;
                            intra4x4_modes[gy * g_stride + gx] = sm;
                        }
                    }
                }

                mb_is_intra[mb_idx] = true;
                mb_chroma_mode[mb_idx] = best_chroma_mode;
                fill_grid_4x4(&mut ref_grid, g_stride, mb_x * 4, mb_y * 4, 4, 4, -1);
                fill_grid_4x4(&mut mv_grid, g_stride, mb_x * 4, mb_y * 4, 4, 4, [0, 0]);

                if use_i4x4 {
                    let qpc_idx = (target_mb_qp as i32 + (self.pps.chroma_qp_index_offset as i32))
                        .clamp(0, 51) as usize;
                    let qpc = CHROMA_QP_TABLE[qpc_idx];
                    let (chroma_dc_q, chroma_ac_q, cbp_chroma) = quantize_chroma_8x8_residual(
                        &src_u,
                        &src_v,
                        &best_pred_u,
                        &best_pred_v,
                        qpc,
                        true,
                    );
                    let cbp = i4_cbp_luma | (cbp_chroma << 4);
                    let qp_delta = if cbp != 0 {
                        (target_mb_qp as i32) - cur_qp
                    } else {
                        0
                    };
                    let eff_qp = (cur_qp + qp_delta).clamp(0, 51) as u8;
                    let eff_qpc = CHROMA_QP_TABLE[(eff_qp as i32
                        + (self.pps.chroma_qp_index_offset as i32))
                        .clamp(0, 51) as usize];

                    // Restore intra4x4_modes temporarily so sequential pred_mode encoding matches decoder
                    for b in 0..16 {
                        let bx = (b & 1) + ((b >> 1) & 2);
                        let by = ((b >> 1) & 1) + ((b >> 2) & 2);
                        intra4x4_modes[(mb_y * 4 + by) * g_stride + (mb_x * 4 + bx)] = 2;
                    }

                    if let Some((ref mut ce, ref mut ctxs)) = cabac_enc {
                        if !is_idr {
                            let l_cond = u8::from(mb_x > 0 && !mb_is_skip[mb_idx - 1]);
                            let t_cond = u8::from(mb_y > 0 && !mb_is_skip[mb_idx - mb_w]);
                            ce.encode_p_skip(ctxs, l_cond, t_cond, false);
                            ce.encode_p_mb_type(ctxs, 5); // I_NxN in P-slice
                        } else {
                            let l_not_i4 = u8::from(mb_x > 0 && mb_is_i16x16[mb_idx - 1]);
                            let t_not_i4 = u8::from(mb_y > 0 && mb_is_i16x16[mb_idx - mb_w]);
                            ce.encode_intra_mb_type(ctxs, 3, l_not_i4, t_not_i4, 0);
                        }
                        if self.pps.transform_8x8_mode_flag {
                            let l8 = mb_x > 0 && mb_8x8_flag[mb_idx - 1];
                            let t8 = mb_y > 0 && mb_8x8_flag[mb_idx - mb_w];
                            let inc = (l8 as usize) + (t8 as usize);
                            ce.encode_decision(&mut ctxs[399 + inc], 0);
                        }
                        for (b, &m) in i4_modes.iter().enumerate() {
                            let bx = (b & 1) + ((b >> 1) & 2);
                            let by = ((b >> 1) & 1) + ((b >> 2) & 2);
                            let gx = mb_x * 4 + bx;
                            let gy = mb_y * 4 + by;
                            let pm = pred_intra4x4_mode(&intra4x4_modes, g_stride, gx, gy);
                            let rem = if m < pm { m } else { m.saturating_sub(1) };
                            ce.encode_intra4x4_pred_mode(ctxs, m == pm, rem);
                            intra4x4_modes[gy * g_stride + gx] = m;
                        }
                        let l_cm = u8::from(
                            mb_x > 0 && mb_is_intra[mb_idx - 1] && mb_chroma_mode[mb_idx - 1] != 0,
                        );
                        let t_cm = u8::from(
                            mb_y > 0
                                && mb_is_intra[mb_idx - mb_w]
                                && mb_chroma_mode[mb_idx - mb_w] != 0,
                        );
                        ce.encode_chroma_pred_mode(ctxs, l_cm, t_cm, best_chroma_mode);
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
                            i4_cbp_luma,
                            cbp_chroma,
                            qp_delta,
                            &mut prev_mb_qp_delta_nz,
                            &i4_q4,
                            None,
                            &chroma_dc_q,
                            &chroma_ac_q,
                            &mut luma_nnz,
                            &mut chroma_nnz,
                            &mb_is_intra,
                            &mb_is_i16x16,
                            &mb_is_skip,
                            &mb_8x8_flag,
                            &mut mb_cbp_luma,
                            &mut mb_cbp_chroma,
                            &mut mb_chroma_dc_cbf,
                        );
                        let is_last = mb_idx + 1 == total_mbs;
                        ce.encode_terminate(u8::from(is_last));
                    } else {
                        let intra_offset = if is_idr { 0u64 } else { 5u64 };
                        w.write_ue(intra_offset); // mb_type = 0 (I_4x4)
                        if self.pps.transform_8x8_mode_flag {
                            w.write_bit(false);
                        }
                        for (b, &m) in i4_modes.iter().enumerate() {
                            let bx = (b & 1) + ((b >> 1) & 2);
                            let by = ((b >> 1) & 1) + ((b >> 2) & 2);
                            let gx = mb_x * 4 + bx;
                            let gy = mb_y * 4 + by;
                            let pm = pred_intra4x4_mode(&intra4x4_modes, g_stride, gx, gy);
                            if m == pm {
                                w.write_bit(true);
                            } else {
                                w.write_bit(false);
                                let rem = if m < pm { m } else { m - 1 };
                                w.write_bits(rem as u64, 3);
                            }
                            intra4x4_modes[gy * g_stride + gx] = m;
                        }
                        w.write_ue(best_chroma_mode as u64);
                        w.write_ue(intra4x4_cbp_to_golomb(cbp));
                        if cbp != 0 {
                            w.write_se(qp_delta as i64);
                            encode_cavlc_4x4_and_chroma_residuals(
                                &mut w,
                                mb_x,
                                mb_y,
                                g_stride,
                                cg_stride,
                                i4_cbp_luma,
                                cbp_chroma,
                                &i4_q4,
                                &chroma_dc_q,
                                &chroma_ac_q,
                                &mut luma_nnz,
                                &mut chroma_nnz,
                            );
                        }
                    }

                    cur_qp = eff_qp as i32;
                    mb_qps[mb_idx] = eff_qp;
                    let mut dummy_y = [0u8; 256];
                    reconstruct_4x4_and_chroma_residuals(
                        &mut dummy_y,
                        &mut best_pred_u,
                        &mut best_pred_v,
                        0,
                        cbp_chroma,
                        eff_qp,
                        eff_qpc,
                        &i4_q4,
                        &chroma_dc_q,
                        &chroma_ac_q,
                    );
                    store_mb_planes(&mut recon, x0, y0, &i4_recon_y, &best_pred_u, &best_pred_v);
                    continue;
                }

                // Encode as I_16x16 with full DC Hadamard + 16 4x4 AC blocks + Chroma DC/AC
                mb_is_i16x16[mb_idx] = true;
                let qp_delta = (target_mb_qp as i32) - cur_qp;
                let eff_qp = (cur_qp + qp_delta).clamp(0, 51) as u8;
                let eff_qpc = CHROMA_QP_TABLE[(eff_qp as i32
                    + (self.pps.chroma_qp_index_offset as i32))
                    .clamp(0, 51) as usize];

                let mut dc_raw = [0i16; 16];
                let mut ac_q4 = [[0i16; 16]; 16];
                let mut any_ac_nz = false;
                for (b, ac_slot) in ac_q4.iter_mut().enumerate() {
                    let bx = (b & 1) + ((b >> 1) & 2);
                    let by = ((b >> 1) & 1) + ((b >> 2) & 2);
                    let dct = sub4x4_dct_u8(
                        &src_y[by * 4 * 16 + bx * 4..],
                        16,
                        &best_i16_pred[by * 4 * 16 + bx * 4..],
                        16,
                    );
                    dc_raw[by * 4 + bx] = dct[0];
                    let mut q = quantize_block4x4(&dct, eff_qp, true);
                    q[0] = 0;
                    if q[1..].iter().any(|&c| c != 0) {
                        any_ac_nz = true;
                    }
                    *ac_slot = q;
                }
                let dc_q = quantize_luma_dc4x4(&dc_raw, eff_qp);
                let mut dc_scanned = [0i16; 16];
                for i in 0..16 {
                    dc_scanned[i] = dc_q[ZIGZAG_4X4[i]];
                }
                let cbp_luma = if any_ac_nz { 15u8 } else { 0u8 };
                let (chroma_dc_q, chroma_ac_q, cbp_chroma) = quantize_chroma_8x8_residual(
                    &src_u,
                    &src_v,
                    &best_pred_u,
                    &best_pred_v,
                    eff_qpc,
                    true,
                );

                let i16_mb_type = 1
                    + (best_i16_mode as u32)
                    + 4 * (cbp_chroma as u32)
                    + if cbp_luma != 0 { 12 } else { 0 };

                mb_cbp_luma[mb_idx] = cbp_luma;
                mb_cbp_chroma[mb_idx] = cbp_chroma;

                if let Some((ref mut ce, ref mut ctxs)) = cabac_enc {
                    if !is_idr {
                        let l_cond = u8::from(mb_x > 0 && !mb_is_skip[mb_idx - 1]);
                        let t_cond = u8::from(mb_y > 0 && !mb_is_skip[mb_idx - mb_w]);
                        ce.encode_p_skip(ctxs, l_cond, t_cond, false);
                        ce.encode_p_mb_type(ctxs, 5 + i16_mb_type);
                    } else {
                        let l_not_i4 = u8::from(mb_x > 0 && mb_is_i16x16[mb_idx - 1]);
                        let t_not_i4 = u8::from(mb_y > 0 && mb_is_i16x16[mb_idx - mb_w]);
                        ce.encode_intra_mb_type(ctxs, 3, l_not_i4, t_not_i4, i16_mb_type);
                    }
                    let l_cm = u8::from(
                        mb_x > 0 && mb_is_intra[mb_idx - 1] && mb_chroma_mode[mb_idx - 1] != 0,
                    );
                    let t_cm = u8::from(
                        mb_y > 0
                            && mb_is_intra[mb_idx - mb_w]
                            && mb_chroma_mode[mb_idx - mb_w] != 0,
                    );
                    ce.encode_chroma_pred_mode(ctxs, l_cm, t_cm, best_chroma_mode);
                    ce.encode_mb_qp_delta(ctxs, u8::from(prev_mb_qp_delta_nz), qp_delta);
                    prev_mb_qp_delta_nz = qp_delta != 0;

                    // Luma 16x16 DC (cat 0)
                    let left_dc_cbf = if mb_x > 0 {
                        Some(mb_luma_dc_cbf[mb_idx - 1])
                    } else {
                        None
                    };
                    let top_dc_cbf = if mb_y > 0 {
                        Some(mb_luma_dc_cbf[mb_idx - mb_w])
                    } else {
                        None
                    };
                    let dc_ctx_inc = cabac_cbf_ctx_inc(left_dc_cbf, top_dc_cbf, true);
                    ce.encode_residual_block_cat(ctxs, 0, dc_ctx_inc, &dc_scanned);
                    mb_luma_dc_cbf[mb_idx] = dc_scanned.iter().any(|&c| c != 0);

                    // Luma 16x16 AC (cat 1) if cbp_luma != 0
                    if cbp_luma != 0 {
                        for (b, ac_blk) in ac_q4.iter().enumerate() {
                            let bx = (b & 1) + ((b >> 1) & 2);
                            let by = ((b >> 1) & 1) + ((b >> 2) & 2);
                            let gx = mb_x * 4 + bx;
                            let gy = mb_y * 4 + by;
                            let mut ac_scanned = [0i16; 15];
                            for k in 0..15 {
                                ac_scanned[k] = ac_blk[ZIGZAG_4X4[k + 1]];
                            }
                            let left_cbf = luma_neighbor_cbf(
                                &luma_nnz,
                                &mb_is_intra,
                                &mb_is_i16x16,
                                &mb_is_skip,
                                &mb_cbp_luma,
                                mb_w,
                                g_stride,
                                gx,
                                gy,
                                true,
                            );
                            let top_cbf = luma_neighbor_cbf(
                                &luma_nnz,
                                &mb_is_intra,
                                &mb_is_i16x16,
                                &mb_is_skip,
                                &mb_cbp_luma,
                                mb_w,
                                g_stride,
                                gx,
                                gy,
                                false,
                            );
                            let inc = cabac_cbf_ctx_inc(left_cbf, top_cbf, true);
                            ce.encode_residual_block_cat(ctxs, 1, inc, &ac_scanned);
                            let nnz = ac_scanned.iter().filter(|&&c| c != 0).count() as u8;
                            luma_nnz[gy * g_stride + gx] = nnz;
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
                        &chroma_dc_q,
                        &chroma_ac_q,
                        &mut chroma_nnz,
                        &mb_is_intra,
                        &mb_is_skip,
                        &mb_cbp_chroma,
                        &mut mb_chroma_dc_cbf,
                    );
                    let is_last = mb_idx + 1 == total_mbs;
                    ce.encode_terminate(u8::from(is_last));
                } else {
                    let intra_offset = if is_idr { 0u64 } else { 5u64 };
                    w.write_ue(intra_offset + (i16_mb_type as u64));
                    w.write_ue(best_chroma_mode as u64);
                    w.write_se(qp_delta as i64);

                    let nc_dc = predict_luma_nc(&luma_nnz, g_stride, mb_x * 4, mb_y * 4);
                    encode_cavlc_block(&mut w, &dc_scanned, nc_dc, false);

                    if cbp_luma != 0 {
                        for (b, ac_blk) in ac_q4.iter().enumerate() {
                            let bx = (b & 1) + ((b >> 1) & 2);
                            let by = ((b >> 1) & 1) + ((b >> 2) & 2);
                            let gx = mb_x * 4 + bx;
                            let gy = mb_y * 4 + by;
                            let mut ac_scanned = [0i16; 15];
                            for k in 0..15 {
                                ac_scanned[k] = ac_blk[ZIGZAG_4X4[k + 1]];
                            }
                            let nc = predict_luma_nc(&luma_nnz, g_stride, gx, gy);
                            encode_cavlc_block(&mut w, &ac_scanned, nc, false);
                            let nnz = ac_scanned.iter().filter(|&&c| c != 0).count() as u8;
                            luma_nnz[gy * g_stride + gx] = nnz;
                        }
                    }

                    encode_cavlc_chroma_residuals(
                        &mut w,
                        mb_x,
                        mb_y,
                        cg_stride,
                        cbp_chroma,
                        &chroma_dc_q,
                        &chroma_ac_q,
                        &mut chroma_nnz,
                    );
                }

                cur_qp = eff_qp as i32;
                mb_qps[mb_idx] = eff_qp;

                // Reconstruct I_16x16 luma + chroma
                let luma_dc_recon = dequantize_idct_luma_dc4x4(&dc_q, eff_qp);
                for (b, ac_blk) in ac_q4.iter().enumerate() {
                    let bx = (b & 1) + ((b >> 1) & 2);
                    let by = ((b >> 1) & 1) + ((b >> 2) & 2);
                    let mut deq = if cbp_luma != 0 {
                        dequantize_block4x4(ac_blk, eff_qp)
                    } else {
                        [0i16; 16]
                    };
                    deq[0] = luma_dc_recon[by * 4 + bx];
                    if deq.iter().any(|&c| c != 0) {
                        add4x4_idct_u8(&mut best_i16_pred[by * 4 * 16 + bx * 4..], 16, &deq);
                    }
                }
                let mut dummy_y = [0u8; 256];
                reconstruct_4x4_and_chroma_residuals(
                    &mut dummy_y,
                    &mut best_pred_u,
                    &mut best_pred_v,
                    0,
                    cbp_chroma,
                    eff_qp,
                    eff_qpc,
                    &ac_q4,
                    &chroma_dc_q,
                    &chroma_ac_q,
                );
                store_mb_planes(
                    &mut recon,
                    x0,
                    y0,
                    &best_i16_pred,
                    &best_pred_u,
                    &best_pred_v,
                );
            }
        }

        let slice_rbsp = if let Some((ce, _)) = cabac_enc {
            let mut hdr_bytes = w.finish();
            hdr_bytes.extend_from_slice(&ce.finish());
            hdr_bytes
        } else {
            if !is_idr && skip_run > 0 {
                w.write_ue(skip_run as u64);
            }
            w.write_rbsp_trailing_bits();
            w.finish()
        };

        // Apply ITU-T H.264 8.7 in-loop deblocking filter onto `recon` before storing in DPB
        if slice_header.disable_deblocking_filter_idc != 1 {
            apply_deblocking_filter_frame(
                &mut recon,
                mb_w,
                mb_h,
                &slice_header,
                &self.pps,
                &mb_qps,
                &mb_is_intra,
                &mb_8x8_flag,
                &luma_nnz,
                &ref_grid,
                &mv_grid,
            );
        }

        let mut out = if is_idr {
            self.headers_annex_b()
        } else {
            Vec::new()
        };
        out.extend_from_slice(&encode_annex_b_nal(3, nal_type, &slice_rbsp));

        // Update VBV leaky bucket & emit ITU-T H.264 FillerData NAL (`nal_unit_type = 12`) in ABR mode if under budget
        self.frame_count += 1;
        if let RateControlMode::Abr {
            bitrate_kbps,
            maxrate_kbps,
            bufsize_kb,
        } = self.rate_control
        {
            let target_bits = (bitrate_kbps as f64) * 1000.0 / fps;
            let expected_cum_bits = target_bits * (self.frame_count as f64);
            let cur_cum_bits = self.total_bits_emitted + ((out.len() * 8) as u64);
            if (cur_cum_bits as f64) < expected_cum_bits * 0.99 {
                let deficit_bytes = (((expected_cum_bits - (cur_cum_bits as f64)) / 8.0) as usize)
                    .saturating_sub(6);
                let max_frame_bytes = maxrate_kbps
                    .map(|m| (((m as f64) * 1000.0 / fps / 8.0) as usize).saturating_sub(out.len()))
                    .unwrap_or(deficit_bytes);
                let filler_len = deficit_bytes.min(max_frame_bytes);
                if filler_len > 0 {
                    let mut filler_rbsp = vec![0xFFu8; filler_len];
                    filler_rbsp.push(0x80);
                    out.extend_from_slice(&encode_annex_b_nal(
                        0,
                        NalUnitType::FillerData,
                        &filler_rbsp,
                    ));
                }
            }
            let cap = (bufsize_kb.unwrap_or(bitrate_kbps * 2) as f64) * 8000.0;
            self.vbv_buffer_bits =
                (self.vbv_buffer_bits + ((out.len() * 8) as f64) - target_bits).clamp(0.0, cap);
        } else if let RateControlMode::Crf {
            maxrate_kbps: Some(maxrate),
            bufsize_kb,
            ..
        } = self.rate_control
        {
            let max_bits_per_frame = (maxrate as f64) * 1000.0 / fps;
            let cap = (bufsize_kb.unwrap_or(maxrate * 2) as f64) * 8000.0;
            self.vbv_buffer_bits = (self.vbv_buffer_bits + max_bits_per_frame
                - ((out.len() * 8) as f64))
                .clamp(0.0, cap);
        }
        self.total_bits_emitted += (out.len() * 8) as u64;
        self.prev_recon = Some(recon);
        Ok(out)
    }
}

#[inline]
fn compute_mb_luma_variance(src_y: &[u8; 256]) -> f32 {
    let sum: u32 = src_y.iter().map(|&v| v as u32).sum();
    let sum_sq: u64 = src_y.iter().map(|&v| (v as u64) * (v as u64)).sum();
    let mean = (sum as f32) / 256.0;
    ((sum_sq as f32) / 256.0 - mean * mean).max(0.0)
}

#[inline]
fn sad_4x4_arr(a: &[u8; 16], b: &[u8; 16]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs())
        .sum()
}

#[inline]
fn intra4x4_mode_available(mode: u8, has_top: bool, has_left: bool) -> bool {
    match mode {
        0 | 3 | 7 => has_top,
        1 | 8 => has_left,
        2 => true,
        _ => has_top && has_left,
    }
}

#[inline]
fn pred_intra4x4_mode(intra4x4_modes: &[u8], g_stride: usize, gx: usize, gy: usize) -> u8 {
    if gx == 0 || gy == 0 {
        return 2;
    }
    let ma = intra4x4_modes[gy * g_stride + (gx - 1)];
    let mb = intra4x4_modes[(gy - 1) * g_stride + gx];
    ma.min(mb)
}

fn gather_intra4x4_neighbors(
    recon: &VideoFrame,
    px: usize,
    py: usize,
    b: usize,
    mb_x: usize,
    mb_w: usize,
) -> (Option<[u8; 8]>, Option<[u8; 4]>, u8) {
    let y_plane = recon.plane(0);
    let top8 = if py > 0 {
        let mut t = [0u8; 8];
        let row = y_plane.row(py - 1);
        t[0..4].copy_from_slice(&row[px..px + 4]);
        let tr_avail = !matches!(b, 3 | 7 | 11 | 13 | 15) && (b != 5 || mb_x + 1 < mb_w);
        if tr_avail && px + 8 <= y_plane.width() {
            t[4..8].copy_from_slice(&row[px + 4..px + 8]);
        } else {
            let pad = t[3];
            t[4..8].fill(pad);
        }
        Some(t)
    } else {
        None
    };
    let left4 = if px > 0 {
        let mut l = [0u8; 4];
        for (r, slot) in l.iter_mut().enumerate() {
            *slot = y_plane.row(py + r)[px - 1];
        }
        Some(l)
    } else {
        None
    };
    let tl = if px > 0 && py > 0 {
        y_plane.row(py - 1)[px - 1]
    } else {
        128
    };
    (top8, left4, tl)
}

#[allow(clippy::type_complexity)]
fn quantize_chroma_8x8_residual(
    src_u: &[u8; 64],
    src_v: &[u8; 64],
    pred_u: &[u8; 64],
    pred_v: &[u8; 64],
    qpc: u8,
    is_intra: bool,
) -> ([[i16; 4]; 2], [[[i16; 15]; 4]; 2], u8) {
    let mut dc_q = [[0i16; 4]; 2];
    let mut ac_q = [[[0i16; 15]; 4]; 2];
    let mut any_dc_nz = false;
    let mut any_ac_nz = false;

    for (plane_idx, (src, pred)) in [(src_u, pred_u), (src_v, pred_v)].into_iter().enumerate() {
        let mut dc_raw = [0i16; 4];
        for b in 0..4 {
            let bx = (b & 1) * 4;
            let by = (b >> 1) * 4;
            let dct = sub4x4_dct_u8(&src[by * 8 + bx..], 8, &pred[by * 8 + bx..], 8);
            dc_raw[b] = dct[0];
            let q = quantize_block4x4(&dct, qpc, is_intra);
            for k in 0..15 {
                let c = q[ZIGZAG_4X4[k + 1]];
                ac_q[plane_idx][b][k] = c;
                if c != 0 {
                    any_ac_nz = true;
                }
            }
        }
        let q_dc = quantize_chroma_dc2x2(&dc_raw, qpc);
        if q_dc.iter().any(|&c| c != 0) {
            any_dc_nz = true;
        }
        dc_q[plane_idx] = q_dc;
    }

    let cbp_chroma = if any_ac_nz {
        2
    } else if any_dc_nz {
        1
    } else {
        0
    };
    (dc_q, ac_q, cbp_chroma)
}

#[allow(clippy::too_many_arguments)]
fn encode_cavlc_4x4_and_chroma_residuals(
    w: &mut BitWriter,
    mb_x: usize,
    mb_y: usize,
    g_stride: usize,
    cg_stride: usize,
    cbp_luma: u8,
    cbp_chroma: u8,
    luma_q4: &[[i16; 16]; 16],
    chroma_dc_q: &[[i16; 4]; 2],
    chroma_ac_q: &[[[i16; 15]; 4]; 2],
    luma_nnz: &mut [u8],
    chroma_nnz: &mut [Vec<u8>; 2],
) {
    for (b, blk) in luma_q4.iter().enumerate() {
        let bx = (b & 1) + ((b >> 1) & 2);
        let by = ((b >> 1) & 1) + ((b >> 2) & 2);
        let gx = mb_x * 4 + bx;
        let gy = mb_y * 4 + by;
        if (cbp_luma & (1 << (b >> 2))) != 0 {
            let mut scanned = [0i16; 16];
            for i in 0..16 {
                scanned[i] = blk[ZIGZAG_4X4[i]];
            }
            let nc = predict_luma_nc(luma_nnz, g_stride, gx, gy);
            encode_cavlc_block(w, &scanned, nc, false);
            let nnz = scanned.iter().filter(|&&c| c != 0).count() as u8;
            luma_nnz[gy * g_stride + gx] = nnz;
        } else {
            luma_nnz[gy * g_stride + gx] = 0;
        }
    }
    encode_cavlc_chroma_residuals(
        w,
        mb_x,
        mb_y,
        cg_stride,
        cbp_chroma,
        chroma_dc_q,
        chroma_ac_q,
        chroma_nnz,
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_cavlc_chroma_residuals(
    w: &mut BitWriter,
    mb_x: usize,
    mb_y: usize,
    cg_stride: usize,
    cbp_chroma: u8,
    chroma_dc_q: &[[i16; 4]; 2],
    chroma_ac_q: &[[[i16; 15]; 4]; 2],
    chroma_nnz: &mut [Vec<u8>; 2],
) {
    if cbp_chroma >= 1 {
        for cdc in chroma_dc_q {
            encode_cavlc_block(w, cdc, 0, true);
        }
    }
    if cbp_chroma >= 2 {
        for plane_idx in 0..2 {
            for (b, ac) in chroma_ac_q[plane_idx].iter().enumerate() {
                let bx = b & 1;
                let by = b >> 1;
                let cgx = mb_x * 2 + bx;
                let cgy = mb_y * 2 + by;
                let nc = predict_chroma_nc(&chroma_nnz[plane_idx], cg_stride, cgx, cgy);
                encode_cavlc_block(w, ac, nc, false);
                let nnz = ac.iter().filter(|&&c| c != 0).count() as u8;
                chroma_nnz[plane_idx][cgy * cg_stride + cgx] = nnz;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn luma_neighbor_cbf(
    luma_nnz: &[u8],
    mb_is_intra: &[bool],
    mb_is_i16x16: &[bool],
    mb_is_skip: &[bool],
    mb_cbp_luma: &[u8],
    mb_w: usize,
    g_stride: usize,
    gx: usize,
    gy: usize,
    is_left: bool,
) -> Option<bool> {
    if is_left && gx == 0 {
        return None;
    }
    if !is_left && gy == 0 {
        return None;
    }
    let ngx = if is_left { gx - 1 } else { gx };
    let ngy = if is_left { gy } else { gy - 1 };
    let n_mb = (ngy / 4) * mb_w + (ngx / 4);
    if mb_is_skip[n_mb] {
        return Some(false);
    }
    if !mb_is_i16x16[n_mb] {
        let n_i8x8 = ((ngy & 2) + ((ngx & 2) >> 1)) as u8;
        if (mb_cbp_luma[n_mb] & (1 << n_i8x8)) == 0 && !mb_is_intra[n_mb] {
            return Some(false);
        }
    }
    Some(luma_nnz[ngy * g_stride + ngx] > 0)
}

#[allow(clippy::too_many_arguments)]
fn encode_cabac_cbp_and_residuals(
    ce: &mut CabacEncoder,
    ctxs: &mut [CabacContext; 460],
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    g_stride: usize,
    cg_stride: usize,
    mb_intra: bool,
    write_8x8_flag: bool,
    cbp_luma: u8,
    cbp_chroma: u8,
    qp_delta: i32,
    prev_mb_qp_delta_nz: &mut bool,
    luma_q4: &[[i16; 16]; 16],
    luma_q8: Option<&[[i16; 64]; 4]>,
    chroma_dc_q: &[[i16; 4]; 2],
    chroma_ac_q: &[[[i16; 15]; 4]; 2],
    luma_nnz: &mut [u8],
    chroma_nnz: &mut [Vec<u8>; 2],
    mb_is_intra: &[bool],
    mb_is_i16x16: &[bool],
    mb_is_skip: &[bool],
    mb_8x8_flag: &[bool],
    mb_cbp_luma: &mut [u8],
    mb_cbp_chroma: &mut [u8],
    mb_chroma_dc_cbf: &mut [Vec<bool>; 2],
) {
    let mb_idx = mb_y * mb_w + mb_x;
    for i8x8 in 0..4u8 {
        let bx8 = (i8x8 & 1) as usize;
        let by8 = (i8x8 >> 1) as usize;
        let left_nz = if bx8 == 1 {
            (cbp_luma & (1 << (i8x8 - 1))) != 0
        } else if mb_x > 0 {
            let l_mb = mb_idx - 1;
            !mb_is_skip[l_mb] && (mb_cbp_luma[l_mb] & (1 << (by8 * 2 + 1))) != 0
        } else {
            true
        };
        let top_nz = if by8 == 1 {
            (cbp_luma & (1 << (i8x8 - 2))) != 0
        } else if mb_y > 0 {
            let t_mb = mb_idx - mb_w;
            !mb_is_skip[t_mb] && (mb_cbp_luma[t_mb] & (1 << (2 + bx8))) != 0
        } else {
            true
        };
        ce.encode_cbp_luma_bit(
            ctxs,
            u8::from(!left_nz),
            u8::from(!top_nz),
            u8::from((cbp_luma & (1 << i8x8)) != 0),
        );
    }

    let left_c = if mb_x > 0 && !mb_is_skip[mb_idx - 1] {
        mb_cbp_chroma[mb_idx - 1]
    } else {
        0
    };
    let top_c = if mb_y > 0 && !mb_is_skip[mb_idx - mb_w] {
        mb_cbp_chroma[mb_idx - mb_w]
    } else {
        0
    };
    ce.encode_cbp_chroma(ctxs, left_c, top_c, cbp_chroma);
    mb_cbp_luma[mb_idx] = cbp_luma;
    mb_cbp_chroma[mb_idx] = cbp_chroma;

    if cbp_luma == 0 && cbp_chroma == 0 {
        *prev_mb_qp_delta_nz = false;
        return;
    }

    if write_8x8_flag && cbp_luma != 0 && !mb_intra {
        let l8 = mb_x > 0 && mb_8x8_flag[mb_idx - 1];
        let t8 = mb_y > 0 && mb_8x8_flag[mb_idx - mb_w];
        let inc = (l8 as usize) + (t8 as usize);
        ce.encode_decision(&mut ctxs[399 + inc], u8::from(luma_q8.is_some()));
    }

    ce.encode_mb_qp_delta(ctxs, u8::from(*prev_mb_qp_delta_nz), qp_delta);
    *prev_mb_qp_delta_nz = qp_delta != 0;

    if let Some(q8_arr) = luma_q8 {
        for (i8x8, q8) in q8_arr.iter().enumerate() {
            let bx8 = (i8x8 & 1) * 2;
            let by8 = (i8x8 >> 1) * 2;
            if (cbp_luma & (1 << i8x8)) != 0 {
                let mut scanned64 = [0i16; 64];
                for i in 0..64 {
                    scanned64[i] = q8[ZIGZAG_8X8[i]];
                }
                ce.encode_residual_block_cat(ctxs, 5, 0, &scanned64);
                fill_grid_4x4(luma_nnz, g_stride, mb_x * 4 + bx8, mb_y * 4 + by8, 2, 2, 1);
            } else {
                fill_grid_4x4(luma_nnz, g_stride, mb_x * 4 + bx8, mb_y * 4 + by8, 2, 2, 0);
            }
        }
    } else {
        for (b, blk) in luma_q4.iter().enumerate() {
            let bx = (b & 1) + ((b >> 1) & 2);
            let by = ((b >> 1) & 1) + ((b >> 2) & 2);
            let gx = mb_x * 4 + bx;
            let gy = mb_y * 4 + by;
            if (cbp_luma & (1 << (b >> 2))) != 0 {
                let mut scanned = [0i16; 16];
                for i in 0..16 {
                    scanned[i] = blk[ZIGZAG_4X4[i]];
                }
                let left_cbf = luma_neighbor_cbf(
                    luma_nnz,
                    mb_is_intra,
                    mb_is_i16x16,
                    mb_is_skip,
                    mb_cbp_luma,
                    mb_w,
                    g_stride,
                    gx,
                    gy,
                    true,
                );
                let top_cbf = luma_neighbor_cbf(
                    luma_nnz,
                    mb_is_intra,
                    mb_is_i16x16,
                    mb_is_skip,
                    mb_cbp_luma,
                    mb_w,
                    g_stride,
                    gx,
                    gy,
                    false,
                );
                let inc = cabac_cbf_ctx_inc(left_cbf, top_cbf, mb_intra);
                ce.encode_residual_block_cat(ctxs, 2, inc, &scanned);
                let nnz = scanned.iter().filter(|&&c| c != 0).count() as u8;
                luma_nnz[gy * g_stride + gx] = nnz;
            } else {
                luma_nnz[gy * g_stride + gx] = 0;
            }
        }
    }

    encode_cabac_chroma_residuals(
        ce,
        ctxs,
        mb_x,
        mb_y,
        mb_w,
        cg_stride,
        mb_intra,
        cbp_chroma,
        chroma_dc_q,
        chroma_ac_q,
        chroma_nnz,
        mb_is_intra,
        mb_is_skip,
        mb_cbp_chroma,
        mb_chroma_dc_cbf,
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_cabac_chroma_residuals(
    ce: &mut CabacEncoder,
    ctxs: &mut [CabacContext; 460],
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    cg_stride: usize,
    mb_intra: bool,
    cbp_chroma: u8,
    chroma_dc_q: &[[i16; 4]; 2],
    chroma_ac_q: &[[[i16; 15]; 4]; 2],
    chroma_nnz: &mut [Vec<u8>; 2],
    _mb_is_intra: &[bool],
    mb_is_skip: &[bool],
    _mb_cbp_chroma: &[u8],
    mb_chroma_dc_cbf: &mut [Vec<bool>; 2],
) {
    let mb_idx = mb_y * mb_w + mb_x;
    if cbp_chroma >= 1 {
        for plane_idx in 0..2 {
            let left_cbf = if mb_x > 0 {
                Some(!mb_is_skip[mb_idx - 1] && mb_chroma_dc_cbf[plane_idx][mb_idx - 1])
            } else {
                None
            };
            let top_cbf = if mb_y > 0 {
                Some(!mb_is_skip[mb_idx - mb_w] && mb_chroma_dc_cbf[plane_idx][mb_idx - mb_w])
            } else {
                None
            };
            let inc = cabac_cbf_ctx_inc(left_cbf, top_cbf, mb_intra);
            ce.encode_residual_block_cat(ctxs, 3, inc, &chroma_dc_q[plane_idx]);
            mb_chroma_dc_cbf[plane_idx][mb_idx] = chroma_dc_q[plane_idx].iter().any(|&c| c != 0);
        }
    }
    if cbp_chroma >= 2 {
        for plane_idx in 0..2 {
            for (b, ac) in chroma_ac_q[plane_idx].iter().enumerate() {
                let bx = b & 1;
                let by = b >> 1;
                let cgx = mb_x * 2 + bx;
                let cgy = mb_y * 2 + by;
                let left_cbf = if cgx > 0 {
                    Some(chroma_nnz[plane_idx][cgy * cg_stride + (cgx - 1)] > 0)
                } else {
                    None
                };
                let top_cbf = if cgy > 0 {
                    Some(chroma_nnz[plane_idx][(cgy - 1) * cg_stride + cgx] > 0)
                } else {
                    None
                };
                let inc = cabac_cbf_ctx_inc(left_cbf, top_cbf, mb_intra);
                ce.encode_residual_block_cat(ctxs, 4, inc, ac);
                let nnz = ac.iter().filter(|&&c| c != 0).count() as u8;
                chroma_nnz[plane_idx][cgy * cg_stride + cgx] = nnz;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_4x4_and_chroma_residuals(
    dst_y: &mut [u8; 256],
    dst_u: &mut [u8; 64],
    dst_v: &mut [u8; 64],
    cbp_luma: u8,
    cbp_chroma: u8,
    qp: u8,
    qpc: u8,
    luma_q4: &[[i16; 16]; 16],
    chroma_dc_q: &[[i16; 4]; 2],
    chroma_ac_q: &[[[i16; 15]; 4]; 2],
) {
    if cbp_luma != 0 {
        for (b, blk) in luma_q4.iter().enumerate() {
            if (cbp_luma & (1 << (b >> 2))) != 0 && blk.iter().any(|&c| c != 0) {
                let bx = ((b & 1) + ((b >> 1) & 2)) * 4;
                let by = (((b >> 1) & 1) + ((b >> 2) & 2)) * 4;
                let deq = dequantize_block4x4(blk, qp);
                add4x4_idct_u8(&mut dst_y[by * 16 + bx..], 16, &deq);
            }
        }
    }
    if cbp_chroma >= 1 {
        let dc_u = dequantize_idct_chroma_dc2x2(&chroma_dc_q[0], qpc);
        let dc_v = dequantize_idct_chroma_dc2x2(&chroma_dc_q[1], qpc);
        for b in 0..4 {
            let bx = (b & 1) * 4;
            let by = (b >> 1) * 4;
            let mut raw_u = [0i16; 16];
            let mut raw_v = [0i16; 16];
            if cbp_chroma >= 2 {
                for k in 0..15 {
                    raw_u[ZIGZAG_4X4[k + 1]] = chroma_ac_q[0][b][k];
                    raw_v[ZIGZAG_4X4[k + 1]] = chroma_ac_q[1][b][k];
                }
            }
            let mut deq_u = dequantize_block4x4(&raw_u, qpc);
            let mut deq_v = dequantize_block4x4(&raw_v, qpc);
            deq_u[0] = dc_u[b];
            deq_v[0] = dc_v[b];
            if deq_u.iter().any(|&c| c != 0) {
                add4x4_idct_u8(&mut dst_u[by * 8 + bx..], 8, &deq_u);
            }
            if deq_v.iter().any(|&c| c != 0) {
                add4x4_idct_u8(&mut dst_v[by * 8 + bx..], 8, &deq_v);
            }
        }
    }
}

#[inline]
fn fill_grid_4x4<T: Copy>(
    grid: &mut [T],
    g_stride: usize,
    gx0: usize,
    gy0: usize,
    w4: usize,
    h4: usize,
    val: T,
) {
    for dy in 0..h4 {
        let row = &mut grid[(gy0 + dy) * g_stride + gx0..(gy0 + dy) * g_stride + gx0 + w4];
        row.fill(val);
    }
}

#[inline]
fn median3(a: i16, b: i16, c: i16) -> i16 {
    a.max(b).min(a.min(b).max(c))
}

/// Computes the ITU-T H.264 8.4.1.3 median/directional motion vector predictor `mvp` (in quarter-pel units)
/// for a partition of width `part_w4` ($4\times4$ units) at `(gx, gy)` with reference `ref_idx`.
#[inline]
fn pred_motion_partition(
    ref_grid: &[i8],
    mv_grid: &[[i16; 2]],
    g_stride: usize,
    gx: usize,
    gy: usize,
    part_w4: usize,
    ref_idx: i8,
) -> [i16; 2] {
    let (left_ref, a) = if gx > 0 {
        let idx = gy * g_stride + (gx - 1);
        (ref_grid[idx], mv_grid[idx])
    } else {
        (-2, [0, 0])
    };
    let (top_ref, b) = if gy > 0 {
        let idx = (gy - 1) * g_stride + gx;
        (ref_grid[idx], mv_grid[idx])
    } else {
        (-2, [0, 0])
    };

    // Top-right 4x4 block availability inside macroblock + frame
    let bx = gx & 3;
    let by = gy & 3;
    let tr_gx = gx + part_w4;
    let tr_avail = gy > 0
        && tr_gx < g_stride
        && !((by == 1 && (bx + part_w4 == 2 || bx + part_w4 == 4))
            || (by == 2 && bx + part_w4 == 4)
            || (by == 3 && (bx + part_w4 == 2 || bx + part_w4 == 4)));

    let (diag_ref, c) = if tr_avail && ref_grid[(gy - 1) * g_stride + tr_gx] != -2 {
        let idx = (gy - 1) * g_stride + tr_gx;
        (ref_grid[idx], mv_grid[idx])
    } else if gy > 0 && gx > 0 {
        let idx = (gy - 1) * g_stride + (gx - 1);
        (ref_grid[idx], mv_grid[idx])
    } else {
        (-2, [0, 0])
    };

    let match_count =
        (left_ref == ref_idx) as u8 + (top_ref == ref_idx) as u8 + (diag_ref == ref_idx) as u8;
    if match_count > 1 {
        [median3(a[0], b[0], c[0]), median3(a[1], b[1], c[1])]
    } else if match_count == 1 {
        if left_ref == ref_idx {
            a
        } else if top_ref == ref_idx {
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

/// Computes the ITU-T H.264 8.4.1.1 `P_Skip` motion vector predictor on the $4\times4$ grid.
#[inline]
fn pred_pskip_motion_grid(
    ref_grid: &[i8],
    mv_grid: &[[i16; 2]],
    g_stride: usize,
    gx0: usize,
    gy0: usize,
) -> [i16; 2] {
    if gx0 == 0 || gy0 == 0 {
        return [0, 0];
    }
    let left_idx = gy0 * g_stride + (gx0 - 1);
    let top_idx = (gy0 - 1) * g_stride + gx0;
    if (ref_grid[left_idx] == 0 && mv_grid[left_idx] == [0, 0])
        || (ref_grid[top_idx] == 0 && mv_grid[top_idx] == [0, 0])
    {
        return [0, 0];
    }
    pred_motion_partition(ref_grid, mv_grid, g_stride, gx0, gy0, 4, 0)
}

#[inline]
fn neighbor_mvd_sum_16x16(
    mvd_grid: &[[i16; 2]],
    g_stride: usize,
    gx0: usize,
    gy0: usize,
) -> (u32, u32) {
    let mut amvd_x = 0u32;
    let mut amvd_y = 0u32;
    if gx0 > 0 {
        let l = mvd_grid[gy0 * g_stride + (gx0 - 1)];
        amvd_x += l[0].unsigned_abs() as u32;
        amvd_y += l[1].unsigned_abs() as u32;
    }
    if gy0 > 0 {
        let t = mvd_grid[(gy0 - 1) * g_stride + gx0];
        amvd_x += t[0].unsigned_abs() as u32;
        amvd_y += t[1].unsigned_abs() as u32;
    }
    (amvd_x, amvd_y)
}

/// Extracts a motion-compensated $16\times16$ luma and $8\times8$ U/V chroma macroblock from `prev`
/// at quarter-pel motion vector `mv = [mv_x, mv_y]` using [`put_h264_qpel16_mc_u8`] and [`put_h264_chroma_mc8_u8`].
fn extract_inter_mb_planes(
    prev: &VideoFrame,
    x0: usize,
    y0: usize,
    mv: [i16; 2],
    dst_y: &mut [u8; 256],
    dst_u: &mut [u8; 64],
    dst_v: &mut [u8; 64],
) {
    let py = prev.plane(0);
    let pu = prev.plane(1);
    let pv = prev.plane(2);
    let ix = (x0 as isize) + ((mv[0] >> 2) as isize);
    let iy = (y0 as isize) + ((mv[1] >> 2) as isize);
    let mx = (mv[0] & 3) as u8;
    let my = (mv[1] & 3) as u8;
    put_h264_qpel16_mc_u8(
        py.data(),
        py.stride(),
        prev.width,
        prev.height,
        ix,
        iy,
        mx,
        my,
        dst_y,
    );

    let cx = ((x0 / 2) as isize) + ((mv[0] >> 3) as isize);
    let cy = ((y0 / 2) as isize) + ((mv[1] >> 3) as isize);
    let fx = (mv[0].rem_euclid(8)) as u8;
    let fy = (mv[1].rem_euclid(8)) as u8;
    put_h264_chroma_mc8_u8(
        pu.data(),
        pu.stride(),
        pu.width(),
        pu.height(),
        cx,
        cy,
        fx,
        fy,
        dst_u,
    );
    put_h264_chroma_mc8_u8(
        pv.data(),
        pv.stride(),
        pv.width(),
        pv.height(),
        cx,
        cy,
        fx,
        fy,
        dst_v,
    );
}

/// Fast NEON diamond motion search over integer luma shifts `(dx, dy)` within `[-16, 16]`
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

    let mvp_dx = ((mvp[0] >> 2) as isize).clamp(-16, 16);
    let mvp_dy = ((mvp[1] >> 2) as isize).clamp(-16, 16);
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

    for step in [4isize, 2, 1] {
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

/// Refines an integer-pel motion vector `int_mv` via 2-stage 6-tap half-pel (`±2`) and quarter-pel (`±1`) search (`0039`).
fn search_qpel_refinement_16x16(
    prev: &VideoFrame,
    x0: usize,
    y0: usize,
    src_y: &[u8; 256],
    int_mv: [i16; 2],
    mvp: [i16; 2],
    qp: u8,
) -> [i16; 2] {
    let py = prev.plane(0);
    let lambda = ((qp as u32) / 3).max(1);
    let mv_cost = |mv: [i16; 2]| -> u32 {
        let dx = (mv[0] - mvp[0]).unsigned_abs() as u32;
        let dy = (mv[1] - mvp[1]).unsigned_abs() as u32;
        lambda * ((32 - (dx + 1).leading_zeros()) + (32 - (dy + 1).leading_zeros()))
    };

    let mut best_mv = int_mv;
    let mut cand_y = [0u8; 256];
    put_h264_qpel16_mc_u8(
        py.data(),
        py.stride(),
        prev.width,
        prev.height,
        (x0 as isize) + ((best_mv[0] >> 2) as isize),
        (y0 as isize) + ((best_mv[1] >> 2) as isize),
        (best_mv[0] & 3) as u8,
        (best_mv[1] & 3) as u8,
        &mut cand_y,
    );
    let mut best_score = sad_block_256_u8(src_y, &cand_y) + mv_cost(best_mv);
    if best_score <= ((qp as u32) * 6).max(64) {
        return best_mv;
    }

    for step in [2i16, 1i16] {
        let base = best_mv;
        for (ox, oy) in [(-step, 0), (step, 0), (0, -step), (0, step)] {
            let mv = [base[0] + ox, base[1] + oy];
            put_h264_qpel16_mc_u8(
                py.data(),
                py.stride(),
                prev.width,
                prev.height,
                (x0 as isize) + ((mv[0] >> 2) as isize),
                (y0 as isize) + ((mv[1] >> 2) as isize),
                (mv[0] & 3) as u8,
                (mv[1] & 3) as u8,
                &mut cand_y,
            );
            let score = sad_block_256_u8(src_y, &cand_y) + mv_cost(mv);
            if score + 4 < best_score {
                best_score = score;
                best_mv = mv;
            }
        }
    }
    best_mv
}

/// Applies the ITU-T H.264 8.7 adaptive in-loop deblocking filter across all $4\times4$ vertical and
/// horizontal macroblock edges using `ragdoll-simd`'s NEON/AVX2 edge kernels (`SIMD-0038`).
#[allow(clippy::too_many_arguments)]
fn apply_deblocking_filter_frame(
    recon: &mut VideoFrame,
    mb_w: usize,
    mb_h: usize,
    sh: &SliceHeader,
    pps: &Pps,
    mb_qps: &[u8],
    mb_is_intra: &[bool],
    mb_8x8_flag: &[bool],
    luma_nnz: &[u8],
    ref_grid: &[i8],
    mv_grid: &[[i16; 2]],
) {
    let g_stride = mb_w * 4;
    for mb_y in 0..mb_h {
        for mb_x in 0..mb_w {
            let mb_idx = mb_y * mb_w + mb_x;
            let qp_self = mb_qps[mb_idx] as i32;

            // 1. Filter 4 vertical edges
            for edge_x in 0..4 {
                let x = mb_x * 16 + edge_x * 4;
                if x < 4 {
                    continue;
                }
                let q_av = if edge_x == 0 && mb_x > 0 {
                    (qp_self + (mb_qps[mb_idx - 1] as i32) + 1) >> 1
                } else {
                    qp_self
                };
                let qp = (q_av + (sh.slice_alpha_c0_offset_div2 as i32) * 2).clamp(0, 51) as u8;
                let qpc = CHROMA_QP_TABLE
                    [((q_av + (pps.chroma_qp_index_offset as i32)).clamp(0, 51) as usize).min(51)];
                let gx = mb_x * 4 + edge_x;
                let mut bs_v = [0u8; 4];
                for (by, bs_slot) in bs_v.iter_mut().enumerate() {
                    let gy = mb_y * 4 + by;
                    let g_q = gy * g_stride + gx;
                    let g_p = gy * g_stride + (gx - 1);
                    let p_mb = (gy / 4) * mb_w + ((gx - 1) / 4);
                    *bs_slot = if mb_is_intra[mb_idx] || mb_is_intra[p_mb] {
                        if edge_x == 0 { 4 } else { 3 }
                    } else if luma_nnz[g_q] > 0 || luma_nnz[g_p] > 0 {
                        2
                    } else if ref_grid[g_q] != ref_grid[g_p]
                        || (mv_grid[g_q][0] - mv_grid[g_p][0]).abs() >= 4
                        || (mv_grid[g_q][1] - mv_grid[g_p][1]).abs() >= 4
                    {
                        1
                    } else {
                        0
                    };
                }
                if bs_v.iter().any(|&b| b > 0) {
                    if !(mb_8x8_flag[mb_idx] && (edge_x & 1) == 1) {
                        let ys = recon.plane(0).stride();
                        h264_loop_filter_luma_edge_v_u8(
                            recon.plane_mut(0).data_mut(),
                            ys,
                            x,
                            mb_y * 16,
                            qp,
                            bs_v,
                        );
                    }
                    if (edge_x & 1) == 0 {
                        let cx = mb_x * 8 + (edge_x / 2) * 4;
                        if cx >= 4 {
                            let us = recon.plane(1).stride();
                            let vs = recon.plane(2).stride();
                            h264_loop_filter_chroma_edge_v_u8(
                                recon.plane_mut(1).data_mut(),
                                us,
                                cx,
                                mb_y * 8,
                                qpc,
                                bs_v,
                            );
                            h264_loop_filter_chroma_edge_v_u8(
                                recon.plane_mut(2).data_mut(),
                                vs,
                                cx,
                                mb_y * 8,
                                qpc,
                                bs_v,
                            );
                        }
                    }
                }
            }

            // 2. Filter 4 horizontal edges
            for edge_y in 0..4 {
                let y = mb_y * 16 + edge_y * 4;
                if y < 4 {
                    continue;
                }
                let q_av = if edge_y == 0 && mb_y > 0 {
                    (qp_self + (mb_qps[mb_idx - mb_w] as i32) + 1) >> 1
                } else {
                    qp_self
                };
                let qp = (q_av + (sh.slice_alpha_c0_offset_div2 as i32) * 2).clamp(0, 51) as u8;
                let qpc = CHROMA_QP_TABLE
                    [((q_av + (pps.chroma_qp_index_offset as i32)).clamp(0, 51) as usize).min(51)];
                let gy = mb_y * 4 + edge_y;
                let mut bs_h = [0u8; 4];
                for (bx, bs_slot) in bs_h.iter_mut().enumerate() {
                    let gx = mb_x * 4 + bx;
                    let g_q = gy * g_stride + gx;
                    let g_p = (gy - 1) * g_stride + gx;
                    let p_mb = ((gy - 1) / 4) * mb_w + (gx / 4);
                    *bs_slot = if mb_is_intra[mb_idx] || mb_is_intra[p_mb] {
                        if edge_y == 0 { 4 } else { 3 }
                    } else if luma_nnz[g_q] > 0 || luma_nnz[g_p] > 0 {
                        2
                    } else if ref_grid[g_q] != ref_grid[g_p]
                        || (mv_grid[g_q][0] - mv_grid[g_p][0]).abs() >= 4
                        || (mv_grid[g_q][1] - mv_grid[g_p][1]).abs() >= 4
                    {
                        1
                    } else {
                        0
                    };
                }
                if bs_h.iter().any(|&b| b > 0) {
                    if !(mb_8x8_flag[mb_idx] && (edge_y & 1) == 1) {
                        let ys = recon.plane(0).stride();
                        h264_loop_filter_luma_edge_h_u8(
                            recon.plane_mut(0).data_mut(),
                            ys,
                            mb_x * 16,
                            y,
                            qp,
                            bs_h,
                        );
                    }
                    if (edge_y & 1) == 0 {
                        let cy = mb_y * 8 + (edge_y / 2) * 4;
                        if cy >= 4 {
                            let us = recon.plane(1).stride();
                            let vs = recon.plane(2).stride();
                            h264_loop_filter_chroma_edge_h_u8(
                                recon.plane_mut(1).data_mut(),
                                us,
                                mb_x * 8,
                                cy,
                                qpc,
                                bs_h,
                            );
                            h264_loop_filter_chroma_edge_h_u8(
                                recon.plane_mut(2).data_mut(),
                                vs,
                                mb_x * 8,
                                cy,
                                qpc,
                                bs_h,
                            );
                        }
                    }
                }
            }
        }
    }
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

/// Pure-Rust H.264 / AVC Baseline (`66`), Main (`77`), and High (`100`) video decoder (`0039`, `0040`).
///
/// Decodes Annex-B or length-prefixed NAL packets containing `SPS`, `PPS`, `SEI`, and
/// `IDR` / `NonIdr` slices across:
/// - Both CAVLC (`entropy_coding_mode_flag = 0`) and CABAC (`entropy_coding_mode_flag = 1`).
/// - `I_4x4`, `I_8x8` (High Profile), `I_16x16`, and `I_PCM` intra macroblocks.
/// - `P_Skip`, `P_L0_16x16`, `P_16x8`, `P_8x16`, and `P_8x8` (all sub-partitions `8x8`, `8x4`, `4x8`, `4x4`).
/// - `B` slices (`B_Skip`, `B_Direct_16x16`, `B_L0/L1/Bi_16x16`, `B_16x8`, `B_8x16`, `B_8x8`) with DPB PTS reordering.
#[derive(Clone, Debug, Default)]
pub struct H264Decoder {
    pub sps: Option<Sps>,
    pub pps: Option<Pps>,
    pub last_sei: Vec<SeiMessage>,
    prev_recon: Option<VideoFrame>,
    dpb_refs: Vec<VideoFrame>,
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

    /// Decodes an entire Annex-B bitstream and returns frames sorted in monotonic presentation order (`PTS` / `POC`)
    /// across B-frame pyramids (`0040`).
    pub fn decode_stream_pts_ordered(&mut self, data: &[u8]) -> Result<Vec<VideoFrame>> {
        let mut gop_frames = Vec::new();
        let mut all_frames = Vec::new();
        for nal in split_annex_b_nals(data) {
            let (hdr, rbsp) = parse_nal_rbsp(nal)?;
            match hdr.nal_unit_type {
                NalUnitType::Sps => self.sps = Some(Sps::parse_rbsp(&rbsp)?),
                NalUnitType::Pps => self.pps = Some(Pps::parse_rbsp(&rbsp)?),
                NalUnitType::Sei => self.last_sei = SeiMessage::parse_rbsp(&rbsp),
                NalUnitType::IdrSlice => {
                    if !gop_frames.is_empty() {
                        gop_frames.sort_by_key(|f: &VideoFrame| f.pts.ticks);
                        all_frames.append(&mut gop_frames);
                    }
                    gop_frames.push(self.decode_slice_rbsp(hdr, &rbsp)?);
                }
                NalUnitType::NonIdrSlice => {
                    gop_frames.push(self.decode_slice_rbsp(hdr, &rbsp)?);
                }
                _ => {}
            }
        }
        if !gop_frames.is_empty() {
            gop_frames.sort_by_key(|f: &VideoFrame| f.pts.ticks);
            all_frames.append(&mut gop_frames);
        }
        let fps = self
            .sps
            .as_ref()
            .and_then(|s| s.fps)
            .unwrap_or(Rational64::FPS_30);
        let tb = fps.recip();
        for (idx, f) in all_frames.iter_mut().enumerate() {
            f.pts = Timestamp::new(idx as i64, tb);
        }
        Ok(all_frames)
    }

    fn get_ref_frame(&self, list: usize, ref_idx: i8) -> Option<&VideoFrame> {
        if self.dpb_refs.is_empty() {
            return self.prev_recon.as_ref();
        }
        if list == 1 {
            // For B-slices, L1 defaults to the most recently decoded reference frame (the future anchor in decode order)
            let idx = (ref_idx.max(0) as usize).min(self.dpb_refs.len() - 1);
            Some(&self.dpb_refs[self.dpb_refs.len() - 1 - idx])
        } else {
            // L0: most recent reference frame at index 0
            let idx = (ref_idx.max(0) as usize).min(self.dpb_refs.len() - 1);
            Some(&self.dpb_refs[self.dpb_refs.len() - 1 - idx])
        }
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

        if nal_header.nal_unit_type == NalUnitType::IdrSlice {
            self.dpb_refs.clear();
        }

        let mut r = BitReader::new(rbsp);
        let sh = SliceHeader::parse_from_reader(&mut r, nal_header, &sps, &pps)?;

        let mb_w = sps.pic_width_in_mbs as usize;
        let mb_h = sps.pic_height_in_map_units as usize;
        let total_mbs = mb_w * mb_h;
        let mut recon = VideoFrame::new(mb_w * 16, mb_h * 16, PixelFormat::Yuv420p);
        let mut cur_qp = sh.slice_qp as i32;
        let is_p_slice = sh.pict_type == PictureType::P;
        let is_b_slice = sh.pict_type == PictureType::B;
        let use_cabac = pps.entropy_coding_mode_flag;

        let mut cabac_state = if use_cabac {
            r.align_to_byte();
            let byte_pos = r.bits_read() / 8;
            let slice_bytes = if byte_pos <= rbsp.len() {
                &rbsp[byte_pos..]
            } else {
                &[][..]
            };
            Some((
                CabacDecoder::new(slice_bytes)?,
                init_cabac_contexts(
                    sh.slice_qp,
                    sh.pict_type == PictureType::I,
                    sh.cabac_init_idc,
                ),
            ))
        } else {
            None
        };

        let g_stride = mb_w * 4;
        let g_h = mb_h * 4;
        let cg_stride = mb_w * 2;
        let cg_h = mb_h * 2;
        let mut luma_nnz = vec![0u8; g_stride * g_h];
        let mut chroma_nnz = [vec![0u8; cg_stride * cg_h], vec![0u8; cg_stride * cg_h]];
        let mut intra4x4_modes = vec![2u8; g_stride * g_h];
        let mut ref_grid = vec![-2i8; g_stride * g_h];
        let mut mv_grid = vec![[0i16; 2]; g_stride * g_h];
        let mut mvd_grid = vec![[0i16; 2]; g_stride * g_h];

        let mut mb_is_intra = vec![false; total_mbs];
        let mut mb_is_i16x16 = vec![false; total_mbs];
        let mut mb_is_skip = vec![false; total_mbs];
        let mut mb_is_direct = vec![false; total_mbs];
        let mut mb_cbp_luma = vec![0u8; total_mbs];
        let mut mb_cbp_chroma = vec![0u8; total_mbs];
        let mut mb_chroma_mode = vec![0u8; total_mbs];
        let mut mb_luma_dc_cbf = vec![false; total_mbs];
        let mut mb_chroma_dc_cbf = [vec![false; total_mbs], vec![false; total_mbs]];
        let mut mb_8x8_flag = vec![false; total_mbs];
        let mut mb_qps = vec![sh.slice_qp; total_mbs];

        let mut skip_run: i64 = -1;
        let mut prev_mb_qp_delta_nz = false;

        for mb_idx in (sh.first_mb_in_slice as usize)..total_mbs {
            let mb_x = mb_idx % mb_w;
            let mb_y = mb_idx / mb_w;
            let x0 = mb_x * 16;
            let y0 = mb_y * 16;
            let cx0 = mb_x * 8;
            let cy0 = mb_y * 8;
            let gx0 = mb_x * 4;
            let gy0 = mb_y * 4;

            // 1. Check P_Skip / B_Skip
            let is_skipped = if is_p_slice || is_b_slice {
                if let Some((ref mut cd, ref mut ctxs)) = cabac_state {
                    let l_cond = u8::from(mb_x > 0 && !mb_is_skip[mb_idx - 1]);
                    let t_cond = u8::from(mb_y > 0 && !mb_is_skip[mb_idx - mb_w]);
                    if is_p_slice {
                        cd.decode_p_skip(ctxs, l_cond, t_cond)
                    } else {
                        cd.decode_b_skip(ctxs, l_cond, t_cond)
                    }
                } else {
                    if skip_run < 0 {
                        if !r.more_rbsp_data() && r.bits_remaining() <= 8 {
                            skip_run = (total_mbs - mb_idx) as i64;
                        } else {
                            skip_run = r.read_ue()? as i64;
                        }
                    }
                    if skip_run > 0 {
                        skip_run -= 1;
                        true
                    } else {
                        skip_run = -1;
                        false
                    }
                }
            } else {
                false
            };

            if is_skipped {
                let skip_mv = if is_p_slice {
                    pred_pskip_motion_grid(&ref_grid, &mv_grid, g_stride, gx0, gy0)
                } else {
                    pred_motion_partition(&ref_grid, &mv_grid, g_stride, gx0, gy0, 4, 0)
                };
                fill_grid_4x4(&mut ref_grid, g_stride, gx0, gy0, 4, 4, 0);
                fill_grid_4x4(&mut mv_grid, g_stride, gx0, gy0, 4, 4, skip_mv);
                mb_is_skip[mb_idx] = true;
                mb_is_direct[mb_idx] = is_b_slice;
                mb_qps[mb_idx] = cur_qp as u8;
                prev_mb_qp_delta_nz = false;

                if is_b_slice && self.dpb_refs.len() >= 2 {
                    let r0 = &self.dpb_refs[self.dpb_refs.len() - 2];
                    let r1 = &self.dpb_refs[self.dpb_refs.len() - 1];
                    let mut y0_buf = [0u8; 256];
                    let mut u0_buf = [0u8; 64];
                    let mut v0_buf = [0u8; 64];
                    let mut y1_buf = [0u8; 256];
                    let mut u1_buf = [0u8; 64];
                    let mut v1_buf = [0u8; 64];
                    extract_inter_mb_planes(
                        r0,
                        x0,
                        y0,
                        skip_mv,
                        &mut y0_buf,
                        &mut u0_buf,
                        &mut v0_buf,
                    );
                    extract_inter_mb_planes(
                        r1,
                        x0,
                        y0,
                        skip_mv,
                        &mut y1_buf,
                        &mut u1_buf,
                        &mut v1_buf,
                    );
                    avg_bi_planes(
                        &mut y0_buf,
                        &mut u0_buf,
                        &mut v0_buf,
                        &y1_buf,
                        &u1_buf,
                        &v1_buf,
                    );
                    store_mb_planes(&mut recon, x0, y0, &y0_buf, &u0_buf, &v0_buf);
                } else if let Some(prev) = self.get_ref_frame(0, 0) {
                    let mut ref_y = [0u8; 256];
                    let mut ref_u = [0u8; 64];
                    let mut ref_v = [0u8; 64];
                    extract_inter_mb_planes(
                        prev, x0, y0, skip_mv, &mut ref_y, &mut ref_u, &mut ref_v,
                    );
                    store_mb_planes(&mut recon, x0, y0, &ref_y, &ref_u, &ref_v);
                }
                if let Some((ref mut cd, _)) = cabac_state {
                    let _eos = cd.decode_terminate();
                }
                continue;
            }

            // 2. Decode raw_mb_type
            let raw_mb_type = if let Some((ref mut cd, ref mut ctxs)) = cabac_state {
                if is_p_slice {
                    cd.decode_p_mb_type(ctxs)
                } else if is_b_slice {
                    let l_nd = u8::from(mb_x > 0 && !mb_is_direct[mb_idx - 1]);
                    let t_nd = u8::from(mb_y > 0 && !mb_is_direct[mb_idx - mb_w]);
                    cd.decode_b_mb_type(ctxs, l_nd, t_nd)
                } else {
                    let l_not_i4 = u8::from(mb_x > 0 && mb_is_i16x16[mb_idx - 1]);
                    let t_not_i4 = u8::from(mb_y > 0 && mb_is_i16x16[mb_idx - mb_w]);
                    cd.decode_intra_mb_type(ctxs, 3, l_not_i4, t_not_i4)
                }
            } else {
                r.read_ue()? as u32
            };

            let intra_offset = if is_p_slice {
                5u32
            } else if is_b_slice {
                23u32
            } else {
                0u32
            };

            // 3. Handle Inter Macroblocks (`P` and `B` slices when `raw_mb_type < intra_offset`)
            if (is_p_slice || is_b_slice) && raw_mb_type < intra_offset {
                let mut pred_y = [128u8; 256];
                let mut pred_u = [128u8; 64];
                let mut pred_v = [128u8; 64];
                let mut can_use_8x8_inter = true;

                if is_p_slice {
                    decode_p_inter_prediction(
                        raw_mb_type,
                        &sh,
                        &mut r,
                        &mut cabac_state,
                        &self.dpb_refs,
                        self.prev_recon.as_ref(),
                        x0,
                        y0,
                        gx0,
                        gy0,
                        g_stride,
                        &mut ref_grid,
                        &mut mv_grid,
                        &mut mvd_grid,
                        &mut pred_y,
                        &mut pred_u,
                        &mut pred_v,
                        &mut can_use_8x8_inter,
                    )?;
                } else {
                    if raw_mb_type == 0 {
                        mb_is_direct[mb_idx] = true;
                    }
                    decode_b_inter_prediction(
                        raw_mb_type,
                        &sh,
                        &mut r,
                        &mut cabac_state,
                        &self.dpb_refs,
                        self.prev_recon.as_ref(),
                        x0,
                        y0,
                        gx0,
                        gy0,
                        g_stride,
                        &mut ref_grid,
                        &mut mv_grid,
                        &mut mvd_grid,
                        &mut pred_y,
                        &mut pred_u,
                        &mut pred_v,
                        &mut can_use_8x8_inter,
                    )?;
                }

                // Decode coded_block_pattern & inter residuals
                let (cbp_luma, cbp_chroma) = if let Some((ref mut cd, ref mut ctxs)) = cabac_state {
                    decode_cabac_cbp(
                        cd,
                        ctxs,
                        mb_x,
                        mb_y,
                        mb_w,
                        &mb_is_skip,
                        &mb_cbp_luma,
                        &mb_cbp_chroma,
                    )
                } else {
                    let code = (r.read_ue()? as usize).min(47);
                    let cbp = GOLOMB_TO_INTER_CBP[code];
                    (cbp & 0x0F, (cbp >> 4) & 0x03)
                };
                mb_cbp_luma[mb_idx] = cbp_luma;
                mb_cbp_chroma[mb_idx] = cbp_chroma;

                if cbp_luma > 0 || cbp_chroma > 0 {
                    let transform_8x8 =
                        if pps.transform_8x8_mode_flag && cbp_luma != 0 && can_use_8x8_inter {
                            let flag = if let Some((ref mut cd, ref mut ctxs)) = cabac_state {
                                let l8 = mb_x > 0 && mb_8x8_flag[mb_idx - 1];
                                let t8 = mb_y > 0 && mb_8x8_flag[mb_idx - mb_w];
                                let inc = (l8 as usize) + (t8 as usize);
                                cd.decode_decision(&mut ctxs[399 + inc]) != 0
                            } else {
                                r.read_bit()?
                            };
                            mb_8x8_flag[mb_idx] = flag;
                            flag
                        } else {
                            false
                        };

                    let mb_qp_delta = if let Some((ref mut cd, ref mut ctxs)) = cabac_state {
                        let d = cd.decode_mb_qp_delta(ctxs, u8::from(prev_mb_qp_delta_nz));
                        prev_mb_qp_delta_nz = d != 0;
                        d
                    } else {
                        r.read_se()? as i32
                    };
                    cur_qp = (cur_qp + mb_qp_delta).rem_euclid(52);
                    let qp = cur_qp as u8;
                    let qpc = CHROMA_QP_TABLE
                        [(cur_qp + (pps.chroma_qp_index_offset as i32)).clamp(0, 51) as usize];

                    decode_and_add_luma_and_chroma_residuals(
                        &mut r,
                        &mut cabac_state,
                        mb_x,
                        mb_y,
                        mb_w,
                        g_stride,
                        cg_stride,
                        false,
                        transform_8x8,
                        cbp_luma,
                        cbp_chroma,
                        qp,
                        qpc,
                        &mut pred_y,
                        &mut pred_u,
                        &mut pred_v,
                        &mut luma_nnz,
                        &mut chroma_nnz,
                        &mb_is_intra,
                        &mb_is_i16x16,
                        &mb_is_skip,
                        &mb_cbp_luma,
                        &mb_cbp_chroma,
                        &mut mb_chroma_dc_cbf,
                    )?;
                } else {
                    prev_mb_qp_delta_nz = false;
                }

                mb_qps[mb_idx] = cur_qp as u8;
                store_mb_planes(&mut recon, x0, y0, &pred_y, &pred_u, &pred_v);
                if let Some((ref mut cd, _)) = cabac_state {
                    let _eos = cd.decode_terminate();
                }
                continue;
            }

            // 4. Intra Macroblock (`I_NxN`, `I_16x16`, or `I_PCM`)
            let intra_mb_type = raw_mb_type - intra_offset;
            mb_is_intra[mb_idx] = true;
            mb_is_direct[mb_idx] = true;
            fill_grid_4x4(&mut ref_grid, g_stride, gx0, gy0, 4, 4, -1);
            fill_grid_4x4(&mut mv_grid, g_stride, gx0, gy0, 4, 4, [0, 0]);

            if intra_mb_type == 25 {
                // I_PCM macroblock
                let mut pcm_y = [0u8; 256];
                let mut pcm_u = [0u8; 64];
                let mut pcm_v = [0u8; 64];
                if let Some((ref mut cd, _)) = cabac_state {
                    let mut raw_pcm = [0u8; 384];
                    cd.align_for_pcm(&mut raw_pcm)?;
                    pcm_y.copy_from_slice(&raw_pcm[0..256]);
                    pcm_u.copy_from_slice(&raw_pcm[256..320]);
                    pcm_v.copy_from_slice(&raw_pcm[320..384]);
                    let _eos = cd.decode_terminate();
                } else {
                    r.align_to_byte();
                    for b in &mut pcm_y {
                        *b = r.read_bits(8)? as u8;
                    }
                    for b in &mut pcm_u {
                        *b = r.read_bits(8)? as u8;
                    }
                    for b in &mut pcm_v {
                        *b = r.read_bits(8)? as u8;
                    }
                }
                fill_grid_4x4(&mut luma_nnz, g_stride, gx0, gy0, 4, 4, 16);
                fill_grid_4x4(&mut chroma_nnz[0], cg_stride, mb_x * 2, mb_y * 2, 2, 2, 16);
                fill_grid_4x4(&mut chroma_nnz[1], cg_stride, mb_x * 2, mb_y * 2, 2, 2, 16);
                mb_is_i16x16[mb_idx] = true;
                mb_cbp_luma[mb_idx] = 15;
                mb_cbp_chroma[mb_idx] = 2;
                mb_luma_dc_cbf[mb_idx] = true;
                mb_chroma_dc_cbf[0][mb_idx] = true;
                mb_chroma_dc_cbf[1][mb_idx] = true;
                mb_qps[mb_idx] = cur_qp as u8;
                prev_mb_qp_delta_nz = false;
                store_mb_planes(&mut recon, x0, y0, &pcm_y, &pcm_u, &pcm_v);
                continue;
            }

            if intra_mb_type == 0 {
                // I_NxN (`I_4x4` or High Profile `I_8x8`)
                let transform_8x8 = if pps.transform_8x8_mode_flag {
                    let flag = if let Some((ref mut cd, ref mut ctxs)) = cabac_state {
                        let l8 = mb_x > 0 && mb_8x8_flag[mb_idx - 1];
                        let t8 = mb_y > 0 && mb_8x8_flag[mb_idx - mb_w];
                        let inc = (l8 as usize) + (t8 as usize);
                        cd.decode_decision(&mut ctxs[399 + inc]) != 0
                    } else {
                        r.read_bit()?
                    };
                    mb_8x8_flag[mb_idx] = flag;
                    flag
                } else {
                    false
                };

                let mut modes_16 = [2u8; 16];
                if transform_8x8 {
                    for i8x8 in 0..4 {
                        let bx = (i8x8 & 1) * 2;
                        let by = (i8x8 >> 1) * 2;
                        let gx = gx0 + bx;
                        let gy = gy0 + by;
                        let pm = pred_intra4x4_mode(&intra4x4_modes, g_stride, gx, gy);
                        let m = if let Some((ref mut cd, ref mut ctxs)) = cabac_state {
                            let (prev_flag, rem) = cd.decode_intra4x4_pred_mode(ctxs);
                            if prev_flag {
                                pm
                            } else if rem < pm {
                                rem
                            } else {
                                rem + 1
                            }
                        } else if r.read_bit()? {
                            pm
                        } else {
                            let rem = r.read_bits(3)? as u8;
                            if rem < pm { rem } else { rem + 1 }
                        };
                        modes_16[i8x8 * 4] = m;
                        fill_grid_4x4(&mut intra4x4_modes, g_stride, gx, gy, 2, 2, m);
                    }
                } else {
                    for (b, m_slot) in modes_16.iter_mut().enumerate() {
                        let bx = (b & 1) + ((b >> 1) & 2);
                        let by = ((b >> 1) & 1) + ((b >> 2) & 2);
                        let gx = gx0 + bx;
                        let gy = gy0 + by;
                        let pm = pred_intra4x4_mode(&intra4x4_modes, g_stride, gx, gy);
                        let m = if let Some((ref mut cd, ref mut ctxs)) = cabac_state {
                            let (prev_flag, rem) = cd.decode_intra4x4_pred_mode(ctxs);
                            if prev_flag {
                                pm
                            } else if rem < pm {
                                rem
                            } else {
                                rem + 1
                            }
                        } else if r.read_bit()? {
                            pm
                        } else {
                            let rem = r.read_bits(3)? as u8;
                            if rem < pm { rem } else { rem + 1 }
                        };
                        *m_slot = m;
                        intra4x4_modes[gy * g_stride + gx] = m;
                    }
                }

                let chroma_pred_mode = if let Some((ref mut cd, ref mut ctxs)) = cabac_state {
                    let l_cm = u8::from(
                        mb_x > 0 && mb_is_intra[mb_idx - 1] && mb_chroma_mode[mb_idx - 1] != 0,
                    );
                    let t_cm = u8::from(
                        mb_y > 0
                            && mb_is_intra[mb_idx - mb_w]
                            && mb_chroma_mode[mb_idx - mb_w] != 0,
                    );
                    cd.decode_chroma_pred_mode(ctxs, l_cm, t_cm)
                } else {
                    r.read_ue()? as u8
                };
                mb_chroma_mode[mb_idx] = chroma_pred_mode;

                let (cbp_luma, cbp_chroma) = if let Some((ref mut cd, ref mut ctxs)) = cabac_state {
                    decode_cabac_cbp(
                        cd,
                        ctxs,
                        mb_x,
                        mb_y,
                        mb_w,
                        &mb_is_skip,
                        &mb_cbp_luma,
                        &mb_cbp_chroma,
                    )
                } else {
                    let code = (r.read_ue()? as usize).min(47);
                    let cbp = GOLOMB_TO_INTRA4X4_CBP[code];
                    (cbp & 0x0F, (cbp >> 4) & 0x03)
                };
                mb_cbp_luma[mb_idx] = cbp_luma;
                mb_cbp_chroma[mb_idx] = cbp_chroma;

                if cbp_luma > 0 || cbp_chroma > 0 {
                    let mb_qp_delta = if let Some((ref mut cd, ref mut ctxs)) = cabac_state {
                        let d = cd.decode_mb_qp_delta(ctxs, u8::from(prev_mb_qp_delta_nz));
                        prev_mb_qp_delta_nz = d != 0;
                        d
                    } else {
                        r.read_se()? as i32
                    };
                    cur_qp = (cur_qp + mb_qp_delta).rem_euclid(52);
                } else {
                    prev_mb_qp_delta_nz = false;
                }
                let qp = cur_qp as u8;
                let qpc = CHROMA_QP_TABLE
                    [(cur_qp + (pps.chroma_qp_index_offset as i32)).clamp(0, 51) as usize];
                mb_qps[mb_idx] = qp;

                // Decode and reconstruct I_4x4 / I_8x8 blocks in-loop so neighbor blocks have exact reconstructed pixels
                if transform_8x8 {
                    for i8x8 in 0..4 {
                        let bx = (i8x8 & 1) * 8;
                        let by = (i8x8 >> 1) * 8;
                        let px = x0 + bx;
                        let py = y0 + by;
                        let (top16, left8, tl_opt) =
                            gather_intra8x8_neighbors(&recon, px, py, i8x8, mb_x, mb_w);
                        let mut blk8 = [0u8; 64];
                        predict_intra_8x8_u8(
                            modes_16[i8x8 * 4],
                            top16.as_ref(),
                            left8.as_ref(),
                            tl_opt,
                            &mut blk8,
                            8,
                        );
                        if (cbp_luma & (1 << i8x8)) != 0 {
                            let mut scanned64 = [0i16; 64];
                            if let Some((ref mut cd, ref mut ctxs)) = cabac_state {
                                cd.decode_residual_block_cat(ctxs, 5, 0, &mut scanned64);
                            } else {
                                for sub in 0..4 {
                                    let sbx = (i8x8 & 1) * 2 + (sub & 1);
                                    let sby = (i8x8 >> 1) * 2 + (sub >> 1);
                                    let nc =
                                        predict_luma_nc(&luma_nnz, g_stride, gx0 + sbx, gy0 + sby);
                                    let mut sub16 = [0i16; 16];
                                    let nnz =
                                        decode_cavlc_block(&mut r, &mut sub16, nc, false)? as u8;
                                    luma_nnz[(gy0 + sby) * g_stride + (gx0 + sbx)] = nnz;
                                    for k in 0..16 {
                                        scanned64[k * 4 + sub] = sub16[k];
                                    }
                                }
                            }
                            let mut raw64 = [0i16; 64];
                            let mut nnz_any = 0u8;
                            for i in 0..64 {
                                raw64[ZIGZAG_8X8[i]] = scanned64[i];
                                if scanned64[i] != 0 {
                                    nnz_any = 1;
                                }
                            }
                            if cabac_state.is_some() {
                                fill_grid_4x4(
                                    &mut luma_nnz,
                                    g_stride,
                                    gx0 + (i8x8 & 1) * 2,
                                    gy0 + (i8x8 >> 1) * 2,
                                    2,
                                    2,
                                    nnz_any,
                                );
                            }
                            if nnz_any > 0 {
                                let deq = dequantize_block8x8(&raw64, qp);
                                add8x8_idct_u8(&mut blk8, 8, &deq);
                            }
                        }
                        for row_r in 0..8 {
                            recon.plane_mut(0).row_mut(py + row_r)[px..px + 8]
                                .copy_from_slice(&blk8[row_r * 8..row_r * 8 + 8]);
                        }
                    }
                } else {
                    for (b, &m) in modes_16.iter().enumerate() {
                        let bx = (b & 1) + ((b >> 1) & 2);
                        let by = ((b >> 1) & 1) + ((b >> 2) & 2);
                        let px = x0 + bx * 4;
                        let py = y0 + by * 4;
                        let gx = gx0 + bx;
                        let gy = gy0 + by;
                        let (top8, left4, tl) =
                            gather_intra4x4_neighbors(&recon, px, py, b, mb_x, mb_w);
                        let mut blk4 = [0u8; 16];
                        predict_intra_4x4_u8(m, top8.as_ref(), left4.as_ref(), tl, &mut blk4, 4);
                        if (cbp_luma & (1 << (b >> 2))) != 0 {
                            let mut scanned = [0i16; 16];
                            if let Some((ref mut cd, ref mut ctxs)) = cabac_state {
                                let left_cbf = luma_neighbor_cbf(
                                    &luma_nnz,
                                    &mb_is_intra,
                                    &mb_is_i16x16,
                                    &mb_is_skip,
                                    &mb_cbp_luma,
                                    mb_w,
                                    g_stride,
                                    gx,
                                    gy,
                                    true,
                                );
                                let top_cbf = luma_neighbor_cbf(
                                    &luma_nnz,
                                    &mb_is_intra,
                                    &mb_is_i16x16,
                                    &mb_is_skip,
                                    &mb_cbp_luma,
                                    mb_w,
                                    g_stride,
                                    gx,
                                    gy,
                                    false,
                                );
                                let inc = cabac_cbf_ctx_inc(left_cbf, top_cbf, true);
                                let (_cbf, nnz) =
                                    cd.decode_residual_block_cat(ctxs, 2, inc, &mut scanned);
                                luma_nnz[gy * g_stride + gx] = nnz as u8;
                            } else {
                                let nc = predict_luma_nc(&luma_nnz, g_stride, gx, gy);
                                let nnz = decode_cavlc_block(&mut r, &mut scanned, nc, false)?;
                                luma_nnz[gy * g_stride + gx] = nnz as u8;
                            }
                            if scanned.iter().any(|&c| c != 0) {
                                let mut raw = [0i16; 16];
                                for i in 0..16 {
                                    raw[ZIGZAG_4X4[i]] = scanned[i];
                                }
                                let deq = dequantize_block4x4(&raw, qp);
                                add4x4_idct_u8(&mut blk4, 4, &deq);
                            }
                        }
                        for row_r in 0..4 {
                            recon.plane_mut(0).row_mut(py + row_r)[px..px + 4]
                                .copy_from_slice(&blk4[row_r * 4..row_r * 4 + 4]);
                        }
                    }
                }

                // Predict + reconstruct Chroma for I_NxN
                let (mut mb_u_buf, mut mb_v_buf) =
                    predict_mb_chroma(&recon, mb_x, mb_y, cx0, cy0, chroma_pred_mode);
                decode_and_add_chroma_residuals(
                    &mut r,
                    &mut cabac_state,
                    mb_x,
                    mb_y,
                    mb_w,
                    cg_stride,
                    true,
                    cbp_chroma,
                    qpc,
                    &mut mb_u_buf,
                    &mut mb_v_buf,
                    &mut chroma_nnz,
                    &mb_is_skip,
                    &mut mb_chroma_dc_cbf,
                )?;
                for row_r in 0..8 {
                    recon.plane_mut(1).row_mut(cy0 + row_r)[cx0..cx0 + 8]
                        .copy_from_slice(&mb_u_buf[row_r * 8..row_r * 8 + 8]);
                    recon.plane_mut(2).row_mut(cy0 + row_r)[cx0..cx0 + 8]
                        .copy_from_slice(&mb_v_buf[row_r * 8..row_r * 8 + 8]);
                }
                if let Some((ref mut cd, _)) = cabac_state {
                    let _eos = cd.decode_terminate();
                }
                continue;
            }

            if !(1..=24).contains(&intra_mb_type) {
                return Err(RagdollError::Unsupported(format!(
                    "H.264 intra mb_type {intra_mb_type} out of range"
                )));
            }

            // I_16x16_{pred_mode}_{cbp_chroma}_{cbp_luma}
            mb_is_i16x16[mb_idx] = true;
            let code = intra_mb_type - 1;
            let luma_pred_mode = (code % 4) as u8;
            let cbp_chroma = ((code / 4) % 3) as u8;
            let cbp_luma = if code >= 12 { 15u8 } else { 0u8 };
            mb_cbp_luma[mb_idx] = cbp_luma;
            mb_cbp_chroma[mb_idx] = cbp_chroma;

            let chroma_pred_mode = if let Some((ref mut cd, ref mut ctxs)) = cabac_state {
                let l_cm = u8::from(
                    mb_x > 0 && mb_is_intra[mb_idx - 1] && mb_chroma_mode[mb_idx - 1] != 0,
                );
                let t_cm = u8::from(
                    mb_y > 0 && mb_is_intra[mb_idx - mb_w] && mb_chroma_mode[mb_idx - mb_w] != 0,
                );
                cd.decode_chroma_pred_mode(ctxs, l_cm, t_cm)
            } else {
                r.read_ue()? as u8
            };
            mb_chroma_mode[mb_idx] = chroma_pred_mode;

            let mb_qp_delta = if let Some((ref mut cd, ref mut ctxs)) = cabac_state {
                let d = cd.decode_mb_qp_delta(ctxs, u8::from(prev_mb_qp_delta_nz));
                prev_mb_qp_delta_nz = d != 0;
                d
            } else {
                r.read_se()? as i32
            };
            cur_qp = (cur_qp + mb_qp_delta).rem_euclid(52);
            let qp = cur_qp as u8;
            let qpc = CHROMA_QP_TABLE
                [(cur_qp + (pps.chroma_qp_index_offset as i32)).clamp(0, 51) as usize];
            mb_qps[mb_idx] = qp;

            // Decode Luma 16x16 DC block
            let mut dc_scanned = [0i16; 16];
            if let Some((ref mut cd, ref mut ctxs)) = cabac_state {
                let left_dc = if mb_x > 0 {
                    Some(mb_luma_dc_cbf[mb_idx - 1])
                } else {
                    None
                };
                let top_dc = if mb_y > 0 {
                    Some(mb_luma_dc_cbf[mb_idx - mb_w])
                } else {
                    None
                };
                let inc = cabac_cbf_ctx_inc(left_dc, top_dc, true);
                let (_cbf, nnz) = cd.decode_residual_block_cat(ctxs, 0, inc, &mut dc_scanned);
                mb_luma_dc_cbf[mb_idx] = nnz > 0;
            } else {
                let nc = predict_luma_nc(&luma_nnz, g_stride, gx0, gy0);
                decode_cavlc_block(&mut r, &mut dc_scanned, nc, false)?;
            }
            let mut dc_matrix = [0i16; 16];
            for i in 0..16 {
                dc_matrix[ZIGZAG_4X4[i]] = dc_scanned[i];
            }
            let luma_dc = dequantize_idct_luma_dc4x4(&dc_matrix, qp);

            // Decode Luma 4x4 AC blocks if cbp_luma != 0
            let mut luma_4x4 = [[0i16; 16]; 16];
            for (b, blk4x4) in luma_4x4.iter_mut().enumerate() {
                let bx = (b & 1) + ((b >> 1) & 2);
                let by = ((b >> 1) & 1) + ((b >> 2) & 2);
                let gx = gx0 + bx;
                let gy = gy0 + by;
                blk4x4[0] = luma_dc[by * 4 + bx];
                if cbp_luma != 0 {
                    let mut ac_scanned = [0i16; 15];
                    if let Some((ref mut cd, ref mut ctxs)) = cabac_state {
                        let left_cbf = luma_neighbor_cbf(
                            &luma_nnz,
                            &mb_is_intra,
                            &mb_is_i16x16,
                            &mb_is_skip,
                            &mb_cbp_luma,
                            mb_w,
                            g_stride,
                            gx,
                            gy,
                            true,
                        );
                        let top_cbf = luma_neighbor_cbf(
                            &luma_nnz,
                            &mb_is_intra,
                            &mb_is_i16x16,
                            &mb_is_skip,
                            &mb_cbp_luma,
                            mb_w,
                            g_stride,
                            gx,
                            gy,
                            false,
                        );
                        let inc = cabac_cbf_ctx_inc(left_cbf, top_cbf, true);
                        let (_cbf, nnz) =
                            cd.decode_residual_block_cat(ctxs, 1, inc, &mut ac_scanned);
                        luma_nnz[gy * g_stride + gx] = nnz as u8;
                    } else {
                        let nc = predict_luma_nc(&luma_nnz, g_stride, gx, gy);
                        let nnz = decode_cavlc_block(&mut r, &mut ac_scanned, nc, false)?;
                        luma_nnz[gy * g_stride + gx] = nnz as u8;
                    }
                    let mut raw_levels = [0i16; 16];
                    for k in 0..15 {
                        raw_levels[ZIGZAG_4X4[k + 1]] = ac_scanned[k];
                    }
                    let deq = dequantize_block4x4(&raw_levels, qp);
                    blk4x4[1..16].copy_from_slice(&deq[1..16]);
                }
            }

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
            for (b, blk4x4) in luma_4x4.iter().enumerate() {
                let bx = ((b & 1) + ((b >> 1) & 2)) * 4;
                let by = (((b >> 1) & 1) + ((b >> 2) & 2)) * 4;
                if blk4x4.iter().any(|&v| v != 0) {
                    add4x4_idct_u8(&mut mb_y_buf[by * 16 + bx..], 16, blk4x4);
                }
            }

            let (mut mb_u_buf, mut mb_v_buf) =
                predict_mb_chroma(&recon, mb_x, mb_y, cx0, cy0, chroma_pred_mode);
            decode_and_add_chroma_residuals(
                &mut r,
                &mut cabac_state,
                mb_x,
                mb_y,
                mb_w,
                cg_stride,
                true,
                cbp_chroma,
                qpc,
                &mut mb_u_buf,
                &mut mb_v_buf,
                &mut chroma_nnz,
                &mb_is_skip,
                &mut mb_chroma_dc_cbf,
            )?;

            store_mb_planes(&mut recon, x0, y0, &mb_y_buf, &mb_u_buf, &mb_v_buf);
            if let Some((ref mut cd, _)) = cabac_state {
                let _eos = cd.decode_terminate();
            }
        }

        // Apply in-loop deblocking filter when enabled in SliceHeader
        if sh.disable_deblocking_filter_idc != 1 {
            apply_deblocking_filter_frame(
                &mut recon,
                mb_w,
                mb_h,
                &sh,
                &pps,
                &mb_qps,
                &mb_is_intra,
                &mb_8x8_flag,
                &luma_nnz,
                &ref_grid,
                &mv_grid,
            );
        }

        // Store reference pictures (`nal_ref_idc != 0`) into DPB
        if nal_header.nal_ref_idc != 0 {
            self.prev_recon = Some(recon.clone());
            self.dpb_refs.push(recon.clone());
            let max_refs = (sps.max_num_ref_frames as usize).max(2);
            if self.dpb_refs.len() > max_refs {
                self.dpb_refs.remove(0);
            }
        }

        // Crop to exact display dimensions if SPS frame_cropping_flag is set
        let out_w = (sps.width() as usize).clamp(2, recon.width);
        let out_h = (sps.height() as usize).clamp(2, recon.height);
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

fn gather_intra8x8_neighbors(
    recon: &VideoFrame,
    px: usize,
    py: usize,
    i8x8: usize,
    mb_x: usize,
    mb_w: usize,
) -> (Option<[u8; 16]>, Option<[u8; 8]>, Option<u8>) {
    let y_plane = recon.plane(0);
    let top16 = if py > 0 {
        let mut t = [0u8; 16];
        let row = y_plane.row(py - 1);
        t[0..8].copy_from_slice(&row[px..px + 8]);
        let tr_avail = match i8x8 {
            0 => true,
            1 => mb_x + 1 < mb_w,
            2 => true,
            _ => false,
        };
        if tr_avail && px + 16 <= y_plane.width() {
            t[8..16].copy_from_slice(&row[px + 8..px + 16]);
        } else {
            let pad = t[7];
            t[8..16].fill(pad);
        }
        Some(t)
    } else {
        None
    };
    let left8 = if px > 0 {
        let mut l = [0u8; 8];
        for (r, slot) in l.iter_mut().enumerate() {
            *slot = y_plane.row(py + r)[px - 1];
        }
        Some(l)
    } else {
        None
    };
    let tl = if px > 0 && py > 0 {
        Some(y_plane.row(py - 1)[px - 1])
    } else {
        None
    };
    (top16, left8, tl)
}

fn predict_mb_chroma(
    recon: &VideoFrame,
    mb_x: usize,
    mb_y: usize,
    cx0: usize,
    cy0: usize,
    chroma_pred_mode: u8,
) -> ([u8; 64], [u8; 64]) {
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
    (mb_u_buf, mb_v_buf)
}

#[allow(clippy::too_many_arguments)]
fn decode_cabac_cbp(
    cd: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext; 460],
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    mb_is_skip: &[bool],
    mb_cbp_luma: &[u8],
    mb_cbp_chroma: &[u8],
) -> (u8, u8) {
    let mb_idx = mb_y * mb_w + mb_x;
    let mut cbp_luma = 0u8;
    for i8x8 in 0..4u8 {
        let bx8 = (i8x8 & 1) as usize;
        let by8 = (i8x8 >> 1) as usize;
        let left_nz = if bx8 == 1 {
            (cbp_luma & (1 << (i8x8 - 1))) != 0
        } else if mb_x > 0 {
            let l_mb = mb_idx - 1;
            !mb_is_skip[l_mb] && (mb_cbp_luma[l_mb] & (1 << (by8 * 2 + 1))) != 0
        } else {
            true
        };
        let top_nz = if by8 == 1 {
            (cbp_luma & (1 << (i8x8 - 2))) != 0
        } else if mb_y > 0 {
            let t_mb = mb_idx - mb_w;
            !mb_is_skip[t_mb] && (mb_cbp_luma[t_mb] & (1 << (2 + bx8))) != 0
        } else {
            true
        };
        if cd.decode_cbp_luma_bit(ctxs, u8::from(!left_nz), u8::from(!top_nz)) != 0 {
            cbp_luma |= 1 << i8x8;
        }
    }
    let left_c = if mb_x > 0 && !mb_is_skip[mb_idx - 1] {
        mb_cbp_chroma[mb_idx - 1]
    } else {
        0
    };
    let top_c = if mb_y > 0 && !mb_is_skip[mb_idx - mb_w] {
        mb_cbp_chroma[mb_idx - mb_w]
    } else {
        0
    };
    let cbp_chroma = cd.decode_cbp_chroma(ctxs, left_c, top_c);
    (cbp_luma, cbp_chroma)
}

#[allow(clippy::too_many_arguments)]
fn decode_and_add_luma_and_chroma_residuals(
    r: &mut BitReader<'_>,
    cabac_state: &mut Option<(CabacDecoder<'_>, [CabacContext; 460])>,
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    g_stride: usize,
    cg_stride: usize,
    mb_intra: bool,
    transform_8x8: bool,
    cbp_luma: u8,
    cbp_chroma: u8,
    qp: u8,
    qpc: u8,
    dst_y: &mut [u8; 256],
    dst_u: &mut [u8; 64],
    dst_v: &mut [u8; 64],
    luma_nnz: &mut [u8],
    chroma_nnz: &mut [Vec<u8>; 2],
    mb_is_intra: &[bool],
    mb_is_i16x16: &[bool],
    mb_is_skip: &[bool],
    mb_cbp_luma: &[u8],
    _mb_cbp_chroma: &[u8],
    mb_chroma_dc_cbf: &mut [Vec<bool>; 2],
) -> Result<()> {
    let gx0 = mb_x * 4;
    let gy0 = mb_y * 4;
    if transform_8x8 {
        for i8x8 in 0..4 {
            let bx8 = (i8x8 & 1) * 2;
            let by8 = (i8x8 >> 1) * 2;
            if (cbp_luma & (1 << i8x8)) != 0 {
                let mut scanned64 = [0i16; 64];
                if let Some((cd, ctxs)) = cabac_state {
                    cd.decode_residual_block_cat(ctxs, 5, 0, &mut scanned64);
                } else {
                    for sub in 0..4 {
                        let sbx = bx8 + (sub & 1);
                        let sby = by8 + (sub >> 1);
                        let nc = predict_luma_nc(luma_nnz, g_stride, gx0 + sbx, gy0 + sby);
                        let mut sub16 = [0i16; 16];
                        let nnz = decode_cavlc_block(r, &mut sub16, nc, false)? as u8;
                        luma_nnz[(gy0 + sby) * g_stride + (gx0 + sbx)] = nnz;
                        for k in 0..16 {
                            scanned64[k * 4 + sub] = sub16[k];
                        }
                    }
                }
                let mut raw64 = [0i16; 64];
                let mut nnz_any = 0u8;
                for i in 0..64 {
                    raw64[ZIGZAG_8X8[i]] = scanned64[i];
                    if scanned64[i] != 0 {
                        nnz_any = 1;
                    }
                }
                if cabac_state.is_some() {
                    fill_grid_4x4(luma_nnz, g_stride, gx0 + bx8, gy0 + by8, 2, 2, nnz_any);
                }
                if nnz_any > 0 {
                    let deq = dequantize_block8x8(&raw64, qp);
                    add8x8_idct_u8(&mut dst_y[(by8 * 4) * 16 + (bx8 * 4)..], 16, &deq);
                }
            }
        }
    } else {
        for b in 0..16 {
            let bx = (b & 1) + ((b >> 1) & 2);
            let by = ((b >> 1) & 1) + ((b >> 2) & 2);
            let gx = gx0 + bx;
            let gy = gy0 + by;
            if (cbp_luma & (1 << (b >> 2))) != 0 {
                let mut scanned = [0i16; 16];
                if let Some((cd, ctxs)) = cabac_state {
                    let left_cbf = luma_neighbor_cbf(
                        luma_nnz,
                        mb_is_intra,
                        mb_is_i16x16,
                        mb_is_skip,
                        mb_cbp_luma,
                        mb_w,
                        g_stride,
                        gx,
                        gy,
                        true,
                    );
                    let top_cbf = luma_neighbor_cbf(
                        luma_nnz,
                        mb_is_intra,
                        mb_is_i16x16,
                        mb_is_skip,
                        mb_cbp_luma,
                        mb_w,
                        g_stride,
                        gx,
                        gy,
                        false,
                    );
                    let inc = cabac_cbf_ctx_inc(left_cbf, top_cbf, mb_intra);
                    let (_cbf, nnz) = cd.decode_residual_block_cat(ctxs, 2, inc, &mut scanned);
                    luma_nnz[gy * g_stride + gx] = nnz as u8;
                } else {
                    let nc = predict_luma_nc(luma_nnz, g_stride, gx, gy);
                    let nnz = decode_cavlc_block(r, &mut scanned, nc, false)?;
                    luma_nnz[gy * g_stride + gx] = nnz as u8;
                }
                if scanned.iter().any(|&c| c != 0) {
                    let mut raw = [0i16; 16];
                    for i in 0..16 {
                        raw[ZIGZAG_4X4[i]] = scanned[i];
                    }
                    let deq = dequantize_block4x4(&raw, qp);
                    add4x4_idct_u8(&mut dst_y[(by * 4) * 16 + (bx * 4)..], 16, &deq);
                }
            }
        }
    }

    decode_and_add_chroma_residuals(
        r,
        cabac_state,
        mb_x,
        mb_y,
        mb_w,
        cg_stride,
        mb_intra,
        cbp_chroma,
        qpc,
        dst_u,
        dst_v,
        chroma_nnz,
        mb_is_skip,
        mb_chroma_dc_cbf,
    )
}

#[allow(clippy::too_many_arguments)]
fn decode_and_add_chroma_residuals(
    r: &mut BitReader<'_>,
    cabac_state: &mut Option<(CabacDecoder<'_>, [CabacContext; 460])>,
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    cg_stride: usize,
    mb_intra: bool,
    cbp_chroma: u8,
    qpc: u8,
    dst_u: &mut [u8; 64],
    dst_v: &mut [u8; 64],
    chroma_nnz: &mut [Vec<u8>; 2],
    mb_is_skip: &[bool],
    mb_chroma_dc_cbf: &mut [Vec<bool>; 2],
) -> Result<()> {
    if cbp_chroma == 0 {
        return Ok(());
    }
    let mb_idx = mb_y * mb_w + mb_x;
    let mut chroma_dc = [[0i16; 4]; 2];
    for (plane_idx, cdc_plane) in chroma_dc.iter_mut().enumerate() {
        let mut cdc_scanned = [0i16; 4];
        if let Some((cd, ctxs)) = cabac_state {
            let left_cbf = if mb_x > 0 {
                Some(!mb_is_skip[mb_idx - 1] && mb_chroma_dc_cbf[plane_idx][mb_idx - 1])
            } else {
                None
            };
            let top_cbf = if mb_y > 0 {
                Some(!mb_is_skip[mb_idx - mb_w] && mb_chroma_dc_cbf[plane_idx][mb_idx - mb_w])
            } else {
                None
            };
            let inc = cabac_cbf_ctx_inc(left_cbf, top_cbf, mb_intra);
            let (_cbf, nnz) = cd.decode_residual_block_cat(ctxs, 3, inc, &mut cdc_scanned);
            mb_chroma_dc_cbf[plane_idx][mb_idx] = nnz > 0;
        } else {
            decode_cavlc_block(r, &mut cdc_scanned, 0, true)?;
        }
        *cdc_plane = dequantize_idct_chroma_dc2x2(&cdc_scanned, qpc);
    }

    let mut chroma_ac = [[[0i16; 15]; 4]; 2];
    if cbp_chroma >= 2 {
        for plane_idx in 0..2 {
            for (b, ac_slot) in chroma_ac[plane_idx].iter_mut().enumerate() {
                let bx = b & 1;
                let by = b >> 1;
                let cgx = mb_x * 2 + bx;
                let cgy = mb_y * 2 + by;
                if let Some((cd, ctxs)) = cabac_state {
                    let left_cbf = if cgx > 0 {
                        Some(chroma_nnz[plane_idx][cgy * cg_stride + (cgx - 1)] > 0)
                    } else {
                        None
                    };
                    let top_cbf = if cgy > 0 {
                        Some(chroma_nnz[plane_idx][(cgy - 1) * cg_stride + cgx] > 0)
                    } else {
                        None
                    };
                    let inc = cabac_cbf_ctx_inc(left_cbf, top_cbf, mb_intra);
                    let (_cbf, nnz) = cd.decode_residual_block_cat(ctxs, 4, inc, ac_slot);
                    chroma_nnz[plane_idx][cgy * cg_stride + cgx] = nnz as u8;
                } else {
                    let nc = predict_chroma_nc(&chroma_nnz[plane_idx], cg_stride, cgx, cgy);
                    let nnz = decode_cavlc_block(r, ac_slot, nc, false)?;
                    chroma_nnz[plane_idx][cgy * cg_stride + cgx] = nnz as u8;
                }
            }
        }
    }

    for b in 0..4 {
        let bx = (b & 1) * 4;
        let by = (b >> 1) * 4;
        let mut raw_u = [0i16; 16];
        let mut raw_v = [0i16; 16];
        if cbp_chroma >= 2 {
            for k in 0..15 {
                raw_u[ZIGZAG_4X4[k + 1]] = chroma_ac[0][b][k];
                raw_v[ZIGZAG_4X4[k + 1]] = chroma_ac[1][b][k];
            }
        }
        let mut deq_u = dequantize_block4x4(&raw_u, qpc);
        let mut deq_v = dequantize_block4x4(&raw_v, qpc);
        deq_u[0] = chroma_dc[0][b];
        deq_v[0] = chroma_dc[1][b];
        if deq_u.iter().any(|&c| c != 0) {
            add4x4_idct_u8(&mut dst_u[by * 8 + bx..], 8, &deq_u);
        }
        if deq_v.iter().any(|&c| c != 0) {
            add4x4_idct_u8(&mut dst_v[by * 8 + bx..], 8, &deq_v);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_p_inter_prediction(
    raw_mb_type: u32,
    sh: &SliceHeader,
    r: &mut BitReader<'_>,
    cabac_state: &mut Option<(CabacDecoder<'_>, [CabacContext; 460])>,
    dpb_refs: &[VideoFrame],
    fallback_ref: Option<&VideoFrame>,
    x0: usize,
    y0: usize,
    gx0: usize,
    gy0: usize,
    g_stride: usize,
    ref_grid: &mut [i8],
    mv_grid: &mut [[i16; 2]],
    mvd_grid: &mut [[i16; 2]],
    dst_y: &mut [u8; 256],
    dst_u: &mut [u8; 64],
    dst_v: &mut [u8; 64],
    can_use_8x8_inter: &mut bool,
) -> Result<()> {
    let get_ref = |r_idx: i8| -> Option<&VideoFrame> {
        if dpb_refs.is_empty() {
            fallback_ref
        } else {
            let idx = (r_idx.max(0) as usize).min(dpb_refs.len() - 1);
            Some(&dpb_refs[dpb_refs.len() - 1 - idx])
        }
    };

    if raw_mb_type <= 2 {
        // P_L0_16x16 (0), P_L0_L0_16x8 (1), P_L0_L0_8x16 (2)
        let parts: &[(usize, usize, usize, usize)] = match raw_mb_type {
            0 => &[(0, 0, 4, 4)],
            1 => &[(0, 0, 4, 2), (0, 2, 4, 2)],
            _ => &[(0, 0, 2, 4), (2, 0, 2, 4)],
        };
        let mut ref_idxs = [0i8; 2];
        if sh.num_ref_idx_l0_active_minus1 > 0 {
            for (p_i, &(px4, py4, _, _)) in parts.iter().enumerate() {
                ref_idxs[p_i] = read_ref_idx(
                    r,
                    cabac_state,
                    sh.num_ref_idx_l0_active_minus1,
                    ref_grid,
                    g_stride,
                    gx0 + px4,
                    gy0 + py4,
                )?;
            }
        }
        for (p_i, &(px4, py4, w4, h4)) in parts.iter().enumerate() {
            let gx = gx0 + px4;
            let gy = gy0 + py4;
            let r_idx = ref_idxs[p_i];
            let mvp = if raw_mb_type == 1
                && p_i == 0
                && gy > 0
                && ref_grid[(gy - 1) * g_stride + gx] == r_idx
            {
                mv_grid[(gy - 1) * g_stride + gx]
            } else if ((raw_mb_type == 1 && p_i == 1) || (raw_mb_type == 2 && p_i == 0))
                && gx > 0
                && ref_grid[gy * g_stride + (gx - 1)] == r_idx
            {
                mv_grid[gy * g_stride + (gx - 1)]
            } else if raw_mb_type == 2
                && p_i == 1
                && gy > 0
                && gx + w4 < g_stride
                && ref_grid[(gy - 1) * g_stride + (gx + w4)] == r_idx
            {
                mv_grid[(gy - 1) * g_stride + (gx + w4)]
            } else {
                pred_motion_partition(ref_grid, mv_grid, g_stride, gx, gy, w4, r_idx)
            };
            let mvd = read_mvd(r, cabac_state, mvd_grid, g_stride, gx, gy)?;
            let mv = [mvp[0].wrapping_add(mvd[0]), mvp[1].wrapping_add(mvd[1])];
            fill_grid_4x4(ref_grid, g_stride, gx, gy, w4, h4, r_idx);
            fill_grid_4x4(mv_grid, g_stride, gx, gy, w4, h4, mv);
            fill_grid_4x4(mvd_grid, g_stride, gx, gy, w4, h4, mvd);
            if let Some(ref_f) = get_ref(r_idx) {
                mc_sub_partition(
                    ref_f,
                    x0,
                    y0,
                    px4 * 4,
                    py4 * 4,
                    w4 * 4,
                    h4 * 4,
                    mv,
                    dst_y,
                    dst_u,
                    dst_v,
                );
            }
        }
    } else {
        // P_8x8 (3) or P_8x8ref0 (4)
        let mut sub_types = [0u32; 4];
        for st in &mut sub_types {
            *st = if let Some((cd, ctxs)) = cabac_state {
                cd.decode_p_sub_mb_type(ctxs) as u32
            } else {
                r.read_ue()? as u32
            };
            if *st != 0 {
                *can_use_8x8_inter = false;
            }
        }
        let mut ref_idxs = [0i8; 4];
        if sh.num_ref_idx_l0_active_minus1 > 0 && raw_mb_type != 4 {
            for (i8x8, r_slot) in ref_idxs.iter_mut().enumerate() {
                let px4 = (i8x8 & 1) * 2;
                let py4 = (i8x8 >> 1) * 2;
                *r_slot = read_ref_idx(
                    r,
                    cabac_state,
                    sh.num_ref_idx_l0_active_minus1,
                    ref_grid,
                    g_stride,
                    gx0 + px4,
                    gy0 + py4,
                )?;
            }
        }
        for i8x8 in 0..4 {
            let bx4 = (i8x8 & 1) * 2;
            let by4 = (i8x8 >> 1) * 2;
            let r_idx = ref_idxs[i8x8];
            let sub_parts: &[(usize, usize, usize, usize)] = match sub_types[i8x8] {
                0 => &[(0, 0, 2, 2)],
                1 => &[(0, 0, 2, 1), (0, 1, 2, 1)],
                2 => &[(0, 0, 1, 2), (1, 0, 1, 2)],
                _ => &[(0, 0, 1, 1), (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1)],
            };
            for &(sx4, sy4, sw4, sh4) in sub_parts {
                let gx = gx0 + bx4 + sx4;
                let gy = gy0 + by4 + sy4;
                let mvp = pred_motion_partition(ref_grid, mv_grid, g_stride, gx, gy, sw4, r_idx);
                let mvd = read_mvd(r, cabac_state, mvd_grid, g_stride, gx, gy)?;
                let mv = [mvp[0].wrapping_add(mvd[0]), mvp[1].wrapping_add(mvd[1])];
                fill_grid_4x4(ref_grid, g_stride, gx, gy, sw4, sh4, r_idx);
                fill_grid_4x4(mv_grid, g_stride, gx, gy, sw4, sh4, mv);
                fill_grid_4x4(mvd_grid, g_stride, gx, gy, sw4, sh4, mvd);
                if let Some(ref_f) = get_ref(r_idx) {
                    mc_sub_partition(
                        ref_f,
                        x0,
                        y0,
                        (bx4 + sx4) * 4,
                        (by4 + sy4) * 4,
                        sw4 * 4,
                        sh4 * 4,
                        mv,
                        dst_y,
                        dst_u,
                        dst_v,
                    );
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_b_inter_prediction(
    raw_mb_type: u32,
    sh: &SliceHeader,
    r: &mut BitReader<'_>,
    cabac_state: &mut Option<(CabacDecoder<'_>, [CabacContext; 460])>,
    dpb_refs: &[VideoFrame],
    fallback_ref: Option<&VideoFrame>,
    x0: usize,
    y0: usize,
    gx0: usize,
    gy0: usize,
    g_stride: usize,
    ref_grid: &mut [i8],
    mv_grid: &mut [[i16; 2]],
    mvd_grid: &mut [[i16; 2]],
    dst_y: &mut [u8; 256],
    dst_u: &mut [u8; 64],
    dst_v: &mut [u8; 64],
    can_use_8x8_inter: &mut bool,
) -> Result<()> {
    let ref_l0 = if dpb_refs.len() >= 2 {
        Some(&dpb_refs[dpb_refs.len() - 2])
    } else {
        dpb_refs.last().or(fallback_ref)
    };
    let ref_l1 = dpb_refs.last().or(fallback_ref);

    if raw_mb_type == 0 {
        // B_Direct_16x16
        let mv = pred_motion_partition(ref_grid, mv_grid, g_stride, gx0, gy0, 4, 0);
        fill_grid_4x4(ref_grid, g_stride, gx0, gy0, 4, 4, 0);
        fill_grid_4x4(mv_grid, g_stride, gx0, gy0, 4, 4, mv);
        if let (Some(r0), Some(r1)) = (ref_l0, ref_l1) {
            let mut y1 = [0u8; 256];
            let mut u1 = [0u8; 64];
            let mut v1 = [0u8; 64];
            extract_inter_mb_planes(r0, x0, y0, mv, dst_y, dst_u, dst_v);
            extract_inter_mb_planes(r1, x0, y0, mv, &mut y1, &mut u1, &mut v1);
            avg_bi_planes(dst_y, dst_u, dst_v, &y1, &u1, &v1);
        }
        return Ok(());
    }

    if raw_mb_type <= 3 {
        // B_L0_16x16 (1), B_L1_16x16 (2), B_Bi_16x16 (3)
        let use_l0 = raw_mb_type == 1 || raw_mb_type == 3;
        let use_l1 = raw_mb_type == 2 || raw_mb_type == 3;
        if use_l0 && sh.num_ref_idx_l0_active_minus1 > 0 {
            let _ = read_ref_idx(
                r,
                cabac_state,
                sh.num_ref_idx_l0_active_minus1,
                ref_grid,
                g_stride,
                gx0,
                gy0,
            )?;
        }
        if use_l1 && sh.num_ref_idx_l1_active_minus1 > 0 {
            let _ = read_ref_idx(
                r,
                cabac_state,
                sh.num_ref_idx_l1_active_minus1,
                ref_grid,
                g_stride,
                gx0,
                gy0,
            )?;
        }
        let mut mv0 = [0i16; 2];
        let mut mv1 = [0i16; 2];
        if use_l0 {
            let mvp = pred_motion_partition(ref_grid, mv_grid, g_stride, gx0, gy0, 4, 0);
            let mvd = read_mvd(r, cabac_state, mvd_grid, g_stride, gx0, gy0)?;
            mv0 = [mvp[0].wrapping_add(mvd[0]), mvp[1].wrapping_add(mvd[1])];
            fill_grid_4x4(ref_grid, g_stride, gx0, gy0, 4, 4, 0);
            fill_grid_4x4(mv_grid, g_stride, gx0, gy0, 4, 4, mv0);
            fill_grid_4x4(mvd_grid, g_stride, gx0, gy0, 4, 4, mvd);
        }
        if use_l1 {
            let mvp = pred_motion_partition(ref_grid, mv_grid, g_stride, gx0, gy0, 4, 0);
            let mvd = read_mvd(r, cabac_state, mvd_grid, g_stride, gx0, gy0)?;
            mv1 = [mvp[0].wrapping_add(mvd[0]), mvp[1].wrapping_add(mvd[1])];
            if !use_l0 {
                fill_grid_4x4(ref_grid, g_stride, gx0, gy0, 4, 4, 0);
                fill_grid_4x4(mv_grid, g_stride, gx0, gy0, 4, 4, mv1);
                fill_grid_4x4(mvd_grid, g_stride, gx0, gy0, 4, 4, mvd);
            }
        }
        match (use_l0, use_l1) {
            (true, false) => {
                if let Some(r0) = ref_l0 {
                    extract_inter_mb_planes(r0, x0, y0, mv0, dst_y, dst_u, dst_v);
                }
            }
            (false, true) => {
                if let Some(r1) = ref_l1 {
                    extract_inter_mb_planes(r1, x0, y0, mv1, dst_y, dst_u, dst_v);
                }
            }
            _ => {
                if let (Some(r0), Some(r1)) = (ref_l0, ref_l1) {
                    let mut y1 = [0u8; 256];
                    let mut u1 = [0u8; 64];
                    let mut v1 = [0u8; 64];
                    extract_inter_mb_planes(r0, x0, y0, mv0, dst_y, dst_u, dst_v);
                    extract_inter_mb_planes(r1, x0, y0, mv1, &mut y1, &mut u1, &mut v1);
                    avg_bi_planes(dst_y, dst_u, dst_v, &y1, &u1, &v1);
                }
            }
        }
        return Ok(());
    }

    if raw_mb_type < 22 {
        // B_16x8 (even `raw_mb_type`) or B_8x16 (odd `raw_mb_type`)
        let is_16x8 = (raw_mb_type & 1) == 0;
        let pair_idx = ((raw_mb_type - 4) / 2) as usize;
        // ITU-T H.264 Table 7-14 pred modes for partition 0 and partition 1 (0=L0, 1=L1, 2=Bi)
        const B_PART_MODES: [(u8, u8); 9] = [
            (0, 0),
            (1, 1),
            (2, 2),
            (0, 1),
            (1, 0),
            (0, 2),
            (2, 0),
            (1, 2),
            (2, 1),
        ];
        let (pm0, pm1) = B_PART_MODES[pair_idx.min(8)];
        let parts = if is_16x8 {
            [(0usize, 0usize, 4usize, 2usize, pm0), (0, 2, 4, 2, pm1)]
        } else {
            [(0usize, 0usize, 2usize, 4usize, pm0), (2, 0, 2, 4, pm1)]
        };
        if sh.num_ref_idx_l0_active_minus1 > 0 {
            for &(px4, py4, _, _, pm) in &parts {
                if pm == 0 || pm == 2 {
                    let _ = read_ref_idx(
                        r,
                        cabac_state,
                        sh.num_ref_idx_l0_active_minus1,
                        ref_grid,
                        g_stride,
                        gx0 + px4,
                        gy0 + py4,
                    )?;
                }
            }
        }
        if sh.num_ref_idx_l1_active_minus1 > 0 {
            for &(px4, py4, _, _, pm) in &parts {
                if pm == 1 || pm == 2 {
                    let _ = read_ref_idx(
                        r,
                        cabac_state,
                        sh.num_ref_idx_l1_active_minus1,
                        ref_grid,
                        g_stride,
                        gx0 + px4,
                        gy0 + py4,
                    )?;
                }
            }
        }
        let mut mvs0 = [[0i16; 2]; 2];
        let mut mvs1 = [[0i16; 2]; 2];
        for (p_i, &(px4, py4, w4, h4, pm)) in parts.iter().enumerate() {
            if pm == 0 || pm == 2 {
                let gx = gx0 + px4;
                let gy = gy0 + py4;
                let mvp = pred_motion_partition(ref_grid, mv_grid, g_stride, gx, gy, w4, 0);
                let mvd = read_mvd(r, cabac_state, mvd_grid, g_stride, gx, gy)?;
                mvs0[p_i] = [mvp[0].wrapping_add(mvd[0]), mvp[1].wrapping_add(mvd[1])];
                fill_grid_4x4(ref_grid, g_stride, gx, gy, w4, h4, 0);
                fill_grid_4x4(mv_grid, g_stride, gx, gy, w4, h4, mvs0[p_i]);
                fill_grid_4x4(mvd_grid, g_stride, gx, gy, w4, h4, mvd);
            }
        }
        for (p_i, &(px4, py4, w4, h4, pm)) in parts.iter().enumerate() {
            if pm == 1 || pm == 2 {
                let gx = gx0 + px4;
                let gy = gy0 + py4;
                let mvp = pred_motion_partition(ref_grid, mv_grid, g_stride, gx, gy, w4, 0);
                let mvd = read_mvd(r, cabac_state, mvd_grid, g_stride, gx, gy)?;
                mvs1[p_i] = [mvp[0].wrapping_add(mvd[0]), mvp[1].wrapping_add(mvd[1])];
                if pm == 1 {
                    fill_grid_4x4(ref_grid, g_stride, gx, gy, w4, h4, 0);
                    fill_grid_4x4(mv_grid, g_stride, gx, gy, w4, h4, mvs1[p_i]);
                    fill_grid_4x4(mvd_grid, g_stride, gx, gy, w4, h4, mvd);
                }
            }
        }
        for (p_i, &(px4, py4, w4, h4, pm)) in parts.iter().enumerate() {
            let mv = if pm == 1 { mvs1[p_i] } else { mvs0[p_i] };
            let r_frame = if pm == 1 { ref_l1 } else { ref_l0 };
            if let Some(rf) = r_frame {
                mc_sub_partition(
                    rf,
                    x0,
                    y0,
                    px4 * 4,
                    py4 * 4,
                    w4 * 4,
                    h4 * 4,
                    mv,
                    dst_y,
                    dst_u,
                    dst_v,
                );
            }
        }
        return Ok(());
    }

    // B_8x8 (`raw_mb_type == 22`)
    let mut sub_types = [0u32; 4];
    for st in &mut sub_types {
        *st = if let Some((cd, ctxs)) = cabac_state {
            cd.decode_b_sub_mb_type(ctxs) as u32
        } else {
            r.read_ue()? as u32
        };
        if *st > 3 {
            *can_use_8x8_inter = false;
        }
    }
    if sh.num_ref_idx_l0_active_minus1 > 0 {
        for (i8x8, &st) in sub_types.iter().enumerate() {
            if matches!(st, 1 | 3 | 4 | 5 | 8 | 9 | 10 | 11 | 12) {
                let _ = read_ref_idx(
                    r,
                    cabac_state,
                    sh.num_ref_idx_l0_active_minus1,
                    ref_grid,
                    g_stride,
                    gx0 + (i8x8 & 1) * 2,
                    gy0 + (i8x8 >> 1) * 2,
                )?;
            }
        }
    }
    if sh.num_ref_idx_l1_active_minus1 > 0 {
        for (i8x8, &st) in sub_types.iter().enumerate() {
            if matches!(st, 2 | 3 | 6 | 7 | 8 | 9 | 10 | 11 | 12) {
                let _ = read_ref_idx(
                    r,
                    cabac_state,
                    sh.num_ref_idx_l1_active_minus1,
                    ref_grid,
                    g_stride,
                    gx0 + (i8x8 & 1) * 2,
                    gy0 + (i8x8 >> 1) * 2,
                )?;
            }
        }
    }
    // Decode L0 MVDs then L1 MVDs
    for list in 0..2 {
        for (i8x8, &st) in sub_types.iter().enumerate() {
            let has_list = if list == 0 {
                matches!(st, 1 | 3 | 4 | 5 | 8 | 9 | 10 | 11 | 12)
            } else {
                matches!(st, 2 | 3 | 6 | 7 | 8 | 9 | 10 | 11 | 12)
            };
            if !has_list {
                continue;
            }
            let num_sub = match st {
                1..=3 => 1,
                4..=9 => 2,
                _ => 4,
            };
            for sub_i in 0..num_sub {
                let gx = gx0 + (i8x8 & 1) * 2 + (sub_i & 1);
                let gy = gy0 + (i8x8 >> 1) * 2 + (sub_i >> 1);
                let mvp = pred_motion_partition(ref_grid, mv_grid, g_stride, gx, gy, 2, 0);
                let mvd = read_mvd(r, cabac_state, mvd_grid, g_stride, gx, gy)?;
                let mv = [mvp[0].wrapping_add(mvd[0]), mvp[1].wrapping_add(mvd[1])];
                fill_grid_4x4(ref_grid, g_stride, gx, gy, 2, 2, 0);
                fill_grid_4x4(mv_grid, g_stride, gx, gy, 2, 2, mv);
                fill_grid_4x4(mvd_grid, g_stride, gx, gy, 2, 2, mvd);
            }
        }
    }
    let mv = mv_grid[gy0 * g_stride + gx0];
    if let Some(r0) = ref_l0 {
        extract_inter_mb_planes(r0, x0, y0, mv, dst_y, dst_u, dst_v);
    }
    Ok(())
}

fn read_ref_idx(
    r: &mut BitReader<'_>,
    cabac_state: &mut Option<(CabacDecoder<'_>, [CabacContext; 460])>,
    num_active_minus1: u8,
    ref_grid: &[i8],
    g_stride: usize,
    gx: usize,
    gy: usize,
) -> Result<i8> {
    if let Some((cd, ctxs)) = cabac_state {
        let l_gt0 = u8::from(gx > 0 && ref_grid[gy * g_stride + (gx - 1)] > 0);
        let t_gt0 = u8::from(gy > 0 && ref_grid[(gy - 1) * g_stride + gx] > 0);
        Ok(cd.decode_ref_idx(ctxs, l_gt0, t_gt0) as i8)
    } else if num_active_minus1 == 1 {
        Ok(if r.read_bit()? { 0 } else { 1 })
    } else {
        Ok(r.read_ue()? as i8)
    }
}

fn read_mvd(
    r: &mut BitReader<'_>,
    cabac_state: &mut Option<(CabacDecoder<'_>, [CabacContext; 460])>,
    mvd_grid: &[[i16; 2]],
    g_stride: usize,
    gx: usize,
    gy: usize,
) -> Result<[i16; 2]> {
    if let Some((cd, ctxs)) = cabac_state {
        let mut amvd_x = 0u32;
        let mut amvd_y = 0u32;
        if gx > 0 {
            let l = mvd_grid[gy * g_stride + (gx - 1)];
            amvd_x += l[0].unsigned_abs() as u32;
            amvd_y += l[1].unsigned_abs() as u32;
        }
        if gy > 0 {
            let t = mvd_grid[(gy - 1) * g_stride + gx];
            amvd_x += t[0].unsigned_abs() as u32;
            amvd_y += t[1].unsigned_abs() as u32;
        }
        let mx = cd.decode_mvd(ctxs, 0, amvd_x);
        let my = cd.decode_mvd(ctxs, 1, amvd_y);
        Ok([mx, my])
    } else {
        let mx = r.read_se()? as i16;
        let my = r.read_se()? as i16;
        Ok([mx, my])
    }
}

#[allow(clippy::too_many_arguments)]
fn mc_sub_partition(
    prev: &VideoFrame,
    mb_x0: usize,
    mb_y0: usize,
    sub_x: usize,
    sub_y: usize,
    sub_w: usize,
    sub_h: usize,
    mv: [i16; 2],
    dst_y: &mut [u8; 256],
    dst_u: &mut [u8; 64],
    dst_v: &mut [u8; 64],
) {
    if sub_w == 16 && sub_h == 16 && sub_x == 0 && sub_y == 0 {
        extract_inter_mb_planes(prev, mb_x0, mb_y0, mv, dst_y, dst_u, dst_v);
        return;
    }
    let py = prev.plane(0);
    let pu = prev.plane(1);
    let pv = prev.plane(2);
    for by in (0..sub_h).step_by(8.min(sub_h)) {
        for bx in (0..sub_w).step_by(8.min(sub_w)) {
            let px = mb_x0 + sub_x + bx;
            let py0 = mb_y0 + sub_y + by;
            let ix = (px as isize) + ((mv[0] >> 2) as isize);
            let iy = (py0 as isize) + ((mv[1] >> 2) as isize);
            let mx = (mv[0] & 3) as u8;
            let my = (mv[1] & 3) as u8;
            let mut blk8 = [0u8; 64];
            put_h264_qpel8_mc_u8(
                py.data(),
                py.stride(),
                prev.width,
                prev.height,
                ix,
                iy,
                mx,
                my,
                &mut blk8,
            );
            let copy_w = 8.min(sub_w - bx);
            let copy_h = 8.min(sub_h - by);
            for r in 0..copy_h {
                let dy = sub_y + by + r;
                let dx = sub_x + bx;
                dst_y[dy * 16 + dx..dy * 16 + dx + copy_w]
                    .copy_from_slice(&blk8[r * 8..r * 8 + copy_w]);
            }
        }
    }
    let mut full_u = [0u8; 64];
    let mut full_v = [0u8; 64];
    let cx = (((mb_x0 + sub_x) / 2) as isize) + ((mv[0] >> 3) as isize);
    let cy = (((mb_y0 + sub_y) / 2) as isize) + ((mv[1] >> 3) as isize);
    let fx = (mv[0].rem_euclid(8)) as u8;
    let fy = (mv[1].rem_euclid(8)) as u8;
    put_h264_chroma_mc8_u8(
        pu.data(),
        pu.stride(),
        pu.width(),
        pu.height(),
        cx,
        cy,
        fx,
        fy,
        &mut full_u,
    );
    put_h264_chroma_mc8_u8(
        pv.data(),
        pv.stride(),
        pv.width(),
        pv.height(),
        cx,
        cy,
        fx,
        fy,
        &mut full_v,
    );
    let cw = (sub_w / 2).max(1);
    let ch = (sub_h / 2).max(1);
    let csx = sub_x / 2;
    let csy = sub_y / 2;
    for r in 0..ch {
        dst_u[(csy + r) * 8 + csx..(csy + r) * 8 + csx + cw]
            .copy_from_slice(&full_u[r * 8..r * 8 + cw]);
        dst_v[(csy + r) * 8 + csx..(csy + r) * 8 + csx + cw]
            .copy_from_slice(&full_v[r * 8..r * 8 + cw]);
    }
}

fn avg_bi_planes(
    y0: &mut [u8; 256],
    u0: &mut [u8; 64],
    v0: &mut [u8; 64],
    y1: &[u8; 256],
    u1: &[u8; 64],
    v1: &[u8; 64],
) {
    for (a, &b) in y0.iter_mut().zip(y1.iter()) {
        *a = (((*a as u16) + (b as u16) + 1) >> 1) as u8;
    }
    for (a, &b) in u0.iter_mut().zip(u1.iter()) {
        *a = (((*a as u16) + (b as u16) + 1) >> 1) as u8;
    }
    for (a, &b) in v0.iter_mut().zip(v1.iter()) {
        *a = (((*a as u16) + (b as u16) + 1) >> 1) as u8;
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
