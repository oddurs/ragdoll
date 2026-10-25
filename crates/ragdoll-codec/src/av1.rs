//! 100% pure-Rust AV1 Open Bitstream Unit (OBU) parser, packetizer, `av1C` configuration
//! record builder, and IVF stream container helper.
//!
//! Reference-hardened against `dav1d/src/obu.c` (`parse_seq_hdr` and `dav1d_parse_obu_header`).

use ragdoll_core::{
    BitReader, BitWriter, ColorPrimaries, ColorRange, Colorimetry, MatrixCoefficients,
    RagdollError, Rational64, Result, TransferCharacteristics, read_leb128, write_leb128,
};

/// AV1 Open Bitstream Unit (`OBU`) type (`obu_type` in bits `6..=3` of the OBU header byte).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObuType {
    Reserved(u8),
    SequenceHeader,       // 1
    TemporalDelimiter,    // 2
    FrameHeader,          // 3
    TileGroup,            // 4
    Metadata,             // 5
    Frame,                // 6
    RedundantFrameHeader, // 7
    TileList,             // 8
    Padding,              // 15
}

impl ObuType {
    #[inline]
    pub const fn from_u8(val: u8) -> Self {
        match val & 0x0F {
            1 => Self::SequenceHeader,
            2 => Self::TemporalDelimiter,
            3 => Self::FrameHeader,
            4 => Self::TileGroup,
            5 => Self::Metadata,
            6 => Self::Frame,
            7 => Self::RedundantFrameHeader,
            8 => Self::TileList,
            15 => Self::Padding,
            other => Self::Reserved(other),
        }
    }

    #[inline]
    pub const fn as_u8(self) -> u8 {
        match self {
            Self::SequenceHeader => 1,
            Self::TemporalDelimiter => 2,
            Self::FrameHeader => 3,
            Self::TileGroup => 4,
            Self::Metadata => 5,
            Self::Frame => 6,
            Self::RedundantFrameHeader => 7,
            Self::TileList => 8,
            Self::Padding => 15,
            Self::Reserved(v) => v & 0x0F,
        }
    }
}

/// Parsed AV1 OBU header (`1` or `2` bytes + optional LEB128 payload size).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObuHeader {
    pub obu_type: ObuType,
    pub extension_flag: bool,
    pub has_size_field: bool,
    pub temporal_id: u8,
    pub spatial_id: u8,
    pub header_len: usize,
    pub payload_size: usize,
}

/// A borrowed AV1 OBU slice extracted from a packet or stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObuPacket<'a> {
    pub header: ObuHeader,
    pub payload: &'a [u8],
    pub raw_obu: &'a [u8],
}

/// Parses an AV1 OBU header at the start of `data` (`dav1d/src/obu.c:dav1d_parse_obu_header`).
pub fn parse_obu_header(data: &[u8]) -> Result<ObuHeader> {
    let Some(&first) = data.first() else {
        return Err(RagdollError::BitstreamEndOfData);
    };
    if (first & 0x80) != 0 {
        return Err(RagdollError::InvalidSyntax(
            "AV1 OBU forbidden_bit must be 0",
        ));
    }
    let obu_type = ObuType::from_u8((first >> 3) & 0x0F);
    let extension_flag = (first & 0x04) != 0;
    let has_size_field = (first & 0x02) != 0;
    if (first & 0x01) != 0 {
        return Err(RagdollError::InvalidSyntax(
            "AV1 OBU reserved_1bit must be 0",
        ));
    }

    let mut pos = 1usize;
    let mut temporal_id = 0u8;
    let mut spatial_id = 0u8;
    if extension_flag {
        let Some(&ext) = data.get(pos) else {
            return Err(RagdollError::BitstreamEndOfData);
        };
        temporal_id = (ext >> 5) & 0x07;
        spatial_id = (ext >> 3) & 0x03;
        pos += 1;
    }

    let payload_size = if has_size_field {
        let (sz, leb_len) = read_leb128(&data[pos..])?;
        pos += leb_len;
        sz as usize
    } else {
        data.len().saturating_sub(pos)
    };

    if pos + payload_size > data.len() {
        return Err(RagdollError::BitstreamEndOfData);
    }

    Ok(ObuHeader {
        obu_type,
        extension_flag,
        has_size_field,
        temporal_id,
        spatial_id,
        header_len: pos,
        payload_size,
    })
}

/// Splits a temporal unit / packet into individual [`ObuPacket`] items.
pub fn split_obus(data: &[u8]) -> Result<Vec<ObuPacket<'_>>> {
    let mut obus = Vec::new();
    let mut pos = 0usize;
    while pos < data.len() {
        let hdr = parse_obu_header(&data[pos..])?;
        let total_len = hdr.header_len + hdr.payload_size;
        let raw_obu = &data[pos..pos + total_len];
        let payload = &raw_obu[hdr.header_len..];
        obus.push(ObuPacket {
            header: hdr,
            payload,
            raw_obu,
        });
        pos += total_len;
    }
    Ok(obus)
}

/// Serializes an AV1 OBU with `obu_has_size_field = 1` and LEB128 length.
pub fn encode_obu(obu_type: ObuType, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(2 + payload.len());
    let hdr_byte = ((obu_type.as_u8() & 0x0F) << 3) | 0x02; // has_size_field = 1
    out.push(hdr_byte);
    write_leb128(payload.len() as u64, &mut out);
    out.extend_from_slice(payload);
    out
}

/// Parsed AV1 `OBU_SEQUENCE_HEADER` (`dav1d/src/obu.c:parse_seq_hdr`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Av1SequenceHeader {
    pub seq_profile: u8,
    pub still_picture: bool,
    pub reduced_still_picture_header: bool,
    pub seq_level_idx_0: u8,
    pub seq_tier_0: u8,
    pub max_frame_width: u32,
    pub max_frame_height: u32,
    pub use_128x128_superblock: bool,
    pub enable_filter_intra: bool,
    pub enable_intra_edge_filter: bool,
    pub enable_superres: bool,
    pub enable_cdef: bool,
    pub enable_restoration: bool,
    pub bit_depth: u8,
    pub monochrome: bool,
    pub subsampling_x: u8,
    pub subsampling_y: u8,
    pub colorimetry: Colorimetry,
    pub fps: Option<Rational64>,
    pub film_grain_params_present: bool,
}

/// Reads an AV1 `uvlc()` unsigned variable-length code (`dav1d_get_vlc`).
fn read_av1_uvlc(r: &mut BitReader<'_>) -> Result<u64> {
    let mut leading_zeros = 0u32;
    while !r.read_bit()? {
        leading_zeros += 1;
        if leading_zeros >= 32 {
            return Ok(u32::MAX as u64);
        }
    }
    if leading_zeros == 0 {
        return Ok(0);
    }
    let val = r.read_bits(leading_zeros)?;
    Ok(val + (1u64 << leading_zeros) - 1)
}

impl Av1SequenceHeader {
    /// Parses an `OBU_SEQUENCE_HEADER` from either a full OBU (with header + LEB128 size) or raw payload bytes.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(RagdollError::BitstreamEndOfData);
        }
        // Check if `data` starts with an OBU_SEQUENCE_HEADER byte (`((b >> 3) & 0x0F) == 1 && (b & 0x80) == 0`)
        if let Ok(hdr) = parse_obu_header(data)
            && hdr.obu_type == ObuType::SequenceHeader
            && hdr.header_len + hdr.payload_size <= data.len()
        {
            return Self::parse_payload(&data[hdr.header_len..hdr.header_len + hdr.payload_size]);
        }
        Self::parse_payload(data)
    }

    /// Parses an `OBU_SEQUENCE_HEADER` directly from its bit-packed payload (`dav1d/src/obu.c:parse_seq_hdr`).
    pub fn parse_payload(payload: &[u8]) -> Result<Self> {
        let mut r = BitReader::new(payload);
        let seq_profile = r.read_bits(3)? as u8;
        if seq_profile > 2 {
            return Err(RagdollError::InvalidSyntax("AV1 seq_profile > 2"));
        }
        let still_picture = r.read_bit()?;
        let reduced_still_picture_header = r.read_bit()?;
        if reduced_still_picture_header && !still_picture {
            return Err(RagdollError::InvalidSyntax(
                "AV1 reduced_still_picture_header without still_picture",
            ));
        }

        let mut seq_level_idx_0 = 0u8;
        let mut seq_tier_0 = 0u8;
        let mut fps = None;

        if reduced_still_picture_header {
            seq_level_idx_0 = r.read_bits(5)? as u8;
        } else {
            let timing_info_present = r.read_bit()?;
            let mut decoder_model_info_present = false;
            let mut buffer_delay_length = 0u32;

            if timing_info_present {
                let num_units_in_tick = r.read_bits(32)?;
                let time_scale = r.read_bits(32)?;
                if num_units_in_tick > 0 && time_scale > 0 {
                    fps = Some(Rational64::new(time_scale as i64, num_units_in_tick as i64));
                }
                let equal_picture_interval = r.read_bit()?;
                if equal_picture_interval {
                    let _num_ticks_per_picture_minus_1 = read_av1_uvlc(&mut r)?;
                }
                decoder_model_info_present = r.read_bit()?;
                if decoder_model_info_present {
                    buffer_delay_length = (r.read_bits(5)? as u32) + 1;
                    let _num_units_in_decoding_tick = r.read_bits(32)?;
                    let _buffer_removal_delay_length = r.read_bits(5)?;
                    let _frame_presentation_delay_length = r.read_bits(5)?;
                }
            }

            let initial_display_delay_present = r.read_bit()?;
            let operating_points_cnt = (r.read_bits(5)? as usize) + 1;
            for i in 0..operating_points_cnt {
                let _operating_point_idc = r.read_bits(12)?;
                let seq_level_idx = r.read_bits(5)? as u8;
                let seq_tier = if seq_level_idx > 7 {
                    u8::from(r.read_bit()?)
                } else {
                    0
                };
                if i == 0 {
                    seq_level_idx_0 = seq_level_idx;
                    seq_tier_0 = seq_tier;
                }
                if decoder_model_info_present {
                    let decoder_model_present_for_op = r.read_bit()?;
                    if decoder_model_present_for_op {
                        let _decoder_buffer_delay = r.read_bits(buffer_delay_length)?;
                        let _encoder_buffer_delay = r.read_bits(buffer_delay_length)?;
                        let _low_delay_mode_flag = r.read_bit()?;
                    }
                }
                if initial_display_delay_present {
                    let idd_present_for_op = r.read_bit()?;
                    if idd_present_for_op {
                        let _initial_display_delay_minus_1 = r.read_bits(4)?;
                    }
                }
            }
        }

        let frame_width_bits = (r.read_bits(4)? as u32) + 1;
        let frame_height_bits = (r.read_bits(4)? as u32) + 1;
        let max_frame_width = (r.read_bits(frame_width_bits)? as u32) + 1;
        let max_frame_height = (r.read_bits(frame_height_bits)? as u32) + 1;

        if !reduced_still_picture_header {
            let frame_id_numbers_present = r.read_bit()?;
            if frame_id_numbers_present {
                let _delta_frame_id_length = r.read_bits(4)?;
                let _additional_frame_id_length = r.read_bits(3)?;
            }
        }

        let use_128x128_superblock = r.read_bit()?;
        let enable_filter_intra = r.read_bit()?;
        let enable_intra_edge_filter = r.read_bit()?;

        if !reduced_still_picture_header {
            let _enable_interintra_compound = r.read_bit()?;
            let _enable_masked_compound = r.read_bit()?;
            let _enable_warped_motion = r.read_bit()?;
            let _enable_dual_filter = r.read_bit()?;
            let enable_order_hint = r.read_bit()?;
            if enable_order_hint {
                let _enable_jnt_comp = r.read_bit()?;
                let _enable_ref_frame_mvs = r.read_bit()?;
            }
            let seq_choose_screen_content_tools = r.read_bit()?;
            let seq_force_screen_content_tools = if seq_choose_screen_content_tools {
                2u8
            } else {
                u8::from(r.read_bit()?)
            };
            if seq_force_screen_content_tools > 0 {
                let seq_choose_integer_mv = r.read_bit()?;
                if !seq_choose_integer_mv {
                    let _seq_force_integer_mv = r.read_bit()?;
                }
            }
            if enable_order_hint {
                let _order_hint_bits_minus_1 = r.read_bits(3)?;
            }
        }

        let enable_superres = r.read_bit()?;
        let enable_cdef = r.read_bit()?;
        let enable_restoration = r.read_bit()?;

        // color_config (`dav1d/src/obu.c:222`)
        let high_bitdepth = r.read_bit()?;
        let twelve_bit = if seq_profile == 2 && high_bitdepth {
            r.read_bit()?
        } else {
            false
        };
        let bit_depth = if !high_bitdepth {
            8
        } else if twelve_bit {
            12
        } else {
            10
        };
        let monochrome = if seq_profile != 1 {
            r.read_bit()?
        } else {
            false
        };

        let mut colorimetry = Colorimetry::bt709_limited();
        let color_description_present = r.read_bit()?;
        let (mut pri_code, mut trc_code, mut mc_code) = (2u8, 2u8, 2u8);
        if color_description_present {
            pri_code = r.read_bits(8)? as u8;
            trc_code = r.read_bits(8)? as u8;
            mc_code = r.read_bits(8)? as u8;
            colorimetry.primaries = ColorPrimaries::from_cicp(pri_code);
            colorimetry.transfer = TransferCharacteristics::from_cicp(trc_code);
            colorimetry.matrix = MatrixCoefficients::from_cicp(mc_code);
        }

        let mut subsampling_x = 1u8;
        let mut subsampling_y = 1u8;
        if monochrome {
            colorimetry.range = if r.read_bit()? {
                ColorRange::Full
            } else {
                ColorRange::Limited
            };
        } else if pri_code == 1 && trc_code == 13 && mc_code == 0 {
            colorimetry.range = ColorRange::Full;
            subsampling_x = 0;
            subsampling_y = 0;
        } else {
            colorimetry.range = if r.read_bit()? {
                ColorRange::Full
            } else {
                ColorRange::Limited
            };
            match seq_profile {
                0 => {
                    subsampling_x = 1;
                    subsampling_y = 1;
                }
                1 => {
                    subsampling_x = 0;
                    subsampling_y = 0;
                }
                _ => {
                    if bit_depth == 12 {
                        subsampling_x = u8::from(r.read_bit()?);
                        if subsampling_x != 0 {
                            subsampling_y = u8::from(r.read_bit()?);
                        } else {
                            subsampling_y = 0;
                        }
                    } else {
                        subsampling_x = 1;
                        subsampling_y = 0;
                    }
                }
            }
            if subsampling_x != 0 && subsampling_y != 0 {
                let _chroma_sample_position = r.read_bits(2)?;
            }
        }

        if !monochrome {
            let _separate_uv_delta_q = r.read_bit()?;
        }
        let film_grain_params_present = r.read_bit()?;

        Ok(Self {
            seq_profile,
            still_picture,
            reduced_still_picture_header,
            seq_level_idx_0,
            seq_tier_0,
            max_frame_width,
            max_frame_height,
            use_128x128_superblock,
            enable_filter_intra,
            enable_intra_edge_filter,
            enable_superres,
            enable_cdef,
            enable_restoration,
            bit_depth,
            monochrome,
            subsampling_x,
            subsampling_y,
            colorimetry,
            fps,
            film_grain_params_present,
        })
    }

    /// Serializes this `Av1SequenceHeader` into a complete `OBU_SEQUENCE_HEADER` OBU.
    pub fn to_obu(&self) -> Vec<u8> {
        let mut w = BitWriter::with_capacity(16);
        w.write_bits(self.seq_profile as u64, 3);
        w.write_bit(self.still_picture);
        w.write_bit(false); // reduced_still_picture_header = 0
        w.write_bit(false); // timing_info_present_flag = 0
        w.write_bit(false); // initial_display_delay_present_flag = 0
        w.write_bits(0, 5); // operating_points_cnt_minus_1 = 0
        w.write_bits(0, 12); // operating_point_idc[0] = 0
        w.write_bits(self.seq_level_idx_0 as u64, 5);
        if self.seq_level_idx_0 > 7 {
            w.write_bit(self.seq_tier_0 != 0);
        }

        w.write_bits(15, 4); // frame_width_bits_minus_1 = 15 (16 bits)
        w.write_bits(15, 4); // frame_height_bits_minus_1 = 15 (16 bits)
        w.write_bits(self.max_frame_width.saturating_sub(1) as u64, 16);
        w.write_bits(self.max_frame_height.saturating_sub(1) as u64, 16);
        w.write_bit(false); // frame_id_numbers_present_flag
        w.write_bit(self.use_128x128_superblock);
        w.write_bit(self.enable_filter_intra);
        w.write_bit(self.enable_intra_edge_filter);
        w.write_bit(false); // interintra
        w.write_bit(false); // masked_compound
        w.write_bit(false); // warped_motion
        w.write_bit(false); // dual_filter
        w.write_bit(false); // order_hint
        w.write_bit(true); // seq_choose_screen_content_tools
        w.write_bit(true); // seq_choose_integer_mv
        w.write_bit(self.enable_superres);
        w.write_bit(self.enable_cdef);
        w.write_bit(self.enable_restoration);

        // color_config
        w.write_bit(self.bit_depth > 8);
        if self.seq_profile == 2 && self.bit_depth > 8 {
            w.write_bit(self.bit_depth == 12);
        }
        if self.seq_profile != 1 {
            w.write_bit(self.monochrome);
        }
        w.write_bit(true); // color_description_present_flag
        w.write_bits(self.colorimetry.primaries as u64, 8);
        w.write_bits(self.colorimetry.transfer as u64, 8);
        w.write_bits(self.colorimetry.matrix as u64, 8);
        w.write_bit(self.colorimetry.range == ColorRange::Full);
        if self.seq_profile == 0 && !self.monochrome {
            w.write_bits(0, 2); // chroma_sample_position = unknown
        }
        if !self.monochrome {
            w.write_bit(false); // separate_uv_delta_q
        }
        w.write_bit(self.film_grain_params_present);
        w.write_rbsp_trailing_bits();

        encode_obu(ObuType::SequenceHeader, &w.finish())
    }

    /// Builds a 4-byte + configOBUs `av1C` (`AV1CodecConfigurationRecord`) payload for MP4/MKV muxing.
    pub fn to_av1c_record(&self, seq_hdr_obu: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + seq_hdr_obu.len());
        out.push(0x81); // marker = 1 (bit 7), version = 1
        out.push(((self.seq_profile & 0x07) << 5) | (self.seq_level_idx_0 & 0x1F));
        let b2 = ((self.seq_tier_0 & 1) << 7)
            | (u8::from(self.bit_depth > 8) << 6)
            | (u8::from(self.bit_depth == 12) << 5)
            | (u8::from(self.monochrome) << 4)
            | ((self.subsampling_x & 1) << 3)
            | ((self.subsampling_y & 1) << 2);
        out.push(b2);
        out.push(0x00);
        out.extend_from_slice(seq_hdr_obu);
        out
    }
}

/// IVF container 32-byte file header (`DKIF`, used for raw AV1 / VP9 / RagdollIntra elementary streams).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IvfHeader {
    pub fourcc: [u8; 4],
    pub width: u16,
    pub height: u16,
    pub timebase_num: u32,
    pub timebase_den: u32,
    pub num_frames: u32,
}

/// A single IVF frame packet (`(pts, raw_packet_bytes)`).
pub type IvfPacket = (u64, Vec<u8>);

/// Parses an IVF bitstream (`DKIF` 32-byte header + 12-byte frame headers) into `(IvfHeader, Vec<IvfPacket>)`.
pub fn parse_ivf_stream(data: &[u8]) -> Result<(IvfHeader, Vec<IvfPacket>)> {
    if data.len() < 32 || &data[0..4] != b"DKIF" {
        return Err(RagdollError::InvalidSyntax("missing IVF DKIF magic"));
    }
    let mut fourcc = [0u8; 4];
    fourcc.copy_from_slice(&data[8..12]);
    let width = u16::from_le_bytes(data[12..14].try_into().unwrap());
    let height = u16::from_le_bytes(data[14..16].try_into().unwrap());
    let timebase_den = u32::from_le_bytes(data[16..20].try_into().unwrap());
    let timebase_num = u32::from_le_bytes(data[20..24].try_into().unwrap());
    let num_frames = u32::from_le_bytes(data[24..28].try_into().unwrap());

    let mut pos = 32usize;
    let mut frames = Vec::new();
    while pos + 12 <= data.len() {
        let sz = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        let pts = u64::from_le_bytes(data[pos + 4..pos + 12].try_into().unwrap());
        pos += 12;
        if pos + sz > data.len() {
            break;
        }
        frames.push((pts, data[pos..pos + sz].to_vec()));
        pos += sz;
    }

    Ok((
        IvfHeader {
            fourcc,
            width,
            height,
            timebase_num,
            timebase_den,
            num_frames,
        },
        frames,
    ))
}
