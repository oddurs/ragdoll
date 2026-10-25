//! 100% pure-Rust FLAC (Free Lossless Audio Codec, RFC 9639) decoder and encoder.
//!
//! Reference-hardened against FFmpeg's `libavcodec/flac.c`, `flacdec.c`, `flacenc.c`,
//! and `flacdsp_lpc_template.c`:
//! - Parses and writes `fLaC` stream headers and 34-byte `STREAMINFO` blocks.
//! - Decodes Constant, Verbatim, Fixed (`order 0..=4`), and FIR Linear Predictive Coding
//!   (`order 1..=32`) subframes with `wasted_bits` and Rice / Rice2 residual partitions.
//! - Reconstructs Independent, Left-Side, Right-Side, and Mid-Side stereo decorrelation.
//! - Validates 8-bit header CRC (`crc8_flac`, poly `0x07`) and 16-bit frame footer CRC (`crc16_flac`, poly `0x8005`).
//! - Encodes lossless FLAC streams using stereo Mid-Side energy competition, Fixed (`0..=4`) +
//!   Levinson-Durbin FIR LPC (`order 1..=12`) predictor selection, and optimal Rice parameter $k$ estimation.

use crate::pcm::{
    audio_frame_to_planar_i32, decorrelate_mid_side_i32, planar_i32_to_audio_frame,
    reconstruct_left_side_i32, reconstruct_mid_side_i32, reconstruct_right_side_i32,
};
use ragdoll_core::{AudioFrame, BitReader, BitWriter, RagdollError, Result, crc8_flac, crc16_flac};
use ragdoll_simd::flac_eval_fixed_orders_l1;

/// 4-byte ASCII stream marker at the start of every native `.flac` file.
pub const FLAC_STREAM_MARKER: &[u8; 4] = b"fLaC";

/// Parsed 34-byte FLAC `STREAMINFO` metadata block (RFC 9639 Section 8.2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlacStreamInfo {
    pub min_block_size: u16,
    pub max_block_size: u16,
    pub min_frame_size: u32,
    pub max_frame_size: u32,
    pub sample_rate: u32,
    pub channels: u8,
    pub bits_per_sample: u8,
    pub total_samples: u64,
    pub md5_signature: [u8; 16],
}

impl FlacStreamInfo {
    /// Parses a 34-byte `STREAMINFO` payload (or a buffer starting with `fLaC` + metadata header).
    pub fn parse(data: &[u8]) -> Result<Self> {
        let payload = if data.starts_with(FLAC_STREAM_MARKER) {
            if data.len() < 42 {
                return Err(RagdollError::BitstreamEndOfData);
            }
            &data[8..42]
        } else {
            if data.len() < 34 {
                return Err(RagdollError::BitstreamEndOfData);
            }
            &data[..34]
        };

        let mut r = BitReader::new(payload);
        let min_block_size = r.read_bits(16)? as u16;
        let max_block_size = r.read_bits(16)? as u16;
        let min_frame_size = r.read_bits(24)? as u32;
        let max_frame_size = r.read_bits(24)? as u32;
        let sample_rate = r.read_bits(20)? as u32;
        let channels = (r.read_bits(3)? as u8) + 1;
        let bits_per_sample = (r.read_bits(5)? as u8) + 1;
        let total_samples = r.read_bits(36)?;
        let mut md5_signature = [0u8; 16];
        md5_signature.copy_from_slice(&payload[18..34]);

        if sample_rate == 0 {
            return Err(RagdollError::InvalidSyntax(
                "FLAC STREAMINFO sample_rate cannot be 0",
            ));
        }

        Ok(Self {
            min_block_size,
            max_block_size,
            min_frame_size,
            max_frame_size,
            sample_rate,
            channels,
            bits_per_sample,
            total_samples,
            md5_signature,
        })
    }

    /// Serializes this `STREAMINFO` block into its canonical 34-byte big-endian layout.
    pub fn to_bytes(&self) -> [u8; 34] {
        let mut w = BitWriter::with_capacity(34);
        w.write_bits(self.min_block_size as u64, 16);
        w.write_bits(self.max_block_size as u64, 16);
        w.write_bits(self.min_frame_size as u64, 24);
        w.write_bits(self.max_frame_size as u64, 24);
        w.write_bits(self.sample_rate as u64, 20);
        w.write_bits((self.channels.saturating_sub(1)) as u64, 3);
        w.write_bits((self.bits_per_sample.saturating_sub(1)) as u64, 5);
        w.write_bits(self.total_samples, 36);
        for &b in &self.md5_signature {
            w.write_bits(b as u64, 8);
        }
        let vec = w.finish();
        let mut out = [0u8; 34];
        out.copy_from_slice(&vec[..34]);
        out
    }
}

/// Parses a `.flac` stream's `fLaC` marker and metadata block chain, returning
/// `(FlacStreamInfo, first_audio_frame_byte_offset)`.
pub fn parse_flac_stream_header(data: &[u8]) -> Result<(FlacStreamInfo, usize)> {
    if data.len() < 42 || &data[0..4] != FLAC_STREAM_MARKER {
        return Err(RagdollError::InvalidSyntax("missing fLaC stream marker"));
    }
    let mut pos = 4usize;
    let mut stream_info: Option<FlacStreamInfo> = None;
    loop {
        if pos + 4 > data.len() {
            return Err(RagdollError::BitstreamEndOfData);
        }
        let header = data[pos];
        let is_last = (header & 0x80) != 0;
        let block_type = header & 0x7F;
        let length = ((data[pos + 1] as usize) << 16)
            | ((data[pos + 2] as usize) << 8)
            | (data[pos + 3] as usize);
        pos += 4;
        if pos + length > data.len() {
            return Err(RagdollError::BitstreamEndOfData);
        }
        if block_type == 0 {
            stream_info = Some(FlacStreamInfo::parse(&data[pos..pos + length])?);
        } else if block_type == 127 {
            return Err(RagdollError::InvalidSyntax(
                "invalid FLAC metadata block type 127",
            ));
        }
        pos += length;
        if is_last {
            break;
        }
    }
    let info = stream_info.ok_or(RagdollError::InvalidSyntax("missing FLAC STREAMINFO block"))?;
    Ok((info, pos))
}

/// Stereo / multi-channel assignment mode in a FLAC frame header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlacChannelMode {
    /// 1..=8 independent channels (`ch_assign = 0..=7`).
    Independent(u8),
    /// Stereo Left + Side (`left`, `side = left - right`, `ch_assign = 8`).
    LeftSide,
    /// Stereo Side + Right (`side = left - right`, `right`, `ch_assign = 9`).
    RightSide,
    /// Stereo Mid + Side (`mid = (left + right) >> 1`, `side = left - right`, `ch_assign = 10`).
    MidSide,
}

impl FlacChannelMode {
    #[inline]
    pub const fn channels(self) -> u8 {
        match self {
            Self::Independent(c) => c,
            Self::LeftSide | Self::RightSide | Self::MidSide => 2,
        }
    }

    /// Returns the bit depth for subframe `subframe_idx`, accounting for the extra
    /// difference bit (`bps + 1`) on the `Side` channel in stereo decorrelation modes.
    #[inline]
    pub const fn subframe_bps(self, frame_bps: u8, subframe_idx: usize) -> u8 {
        match (self, subframe_idx) {
            (Self::LeftSide, 1) | (Self::RightSide, 0) | (Self::MidSide, 1) => frame_bps + 1,
            _ => frame_bps,
        }
    }
}

/// Parsed FLAC audio frame header (RFC 9639 Section 9.1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlacFrameHeader {
    pub variable_block_size: bool,
    pub block_size: usize,
    pub sample_rate: u32,
    pub channel_mode: FlacChannelMode,
    pub bits_per_sample: u8,
    pub frame_or_sample_num: u64,
    pub header_crc8: u8,
    pub header_len_bytes: usize,
}

/// Reads a FLAC UTF-8-coded 31-bit frame number or 36-bit sample number (`libavcodec/flac.c:get_utf8`).
fn read_flac_utf8(r: &mut BitReader<'_>) -> Result<u64> {
    let first = r.read_bits(8)? as u8;
    if (first & 0x80) == 0 {
        return Ok(first as u64);
    }
    let ones = (!first).leading_zeros() as usize;
    if !(2..=7).contains(&ones) {
        return Err(RagdollError::InvalidSyntax(
            "invalid FLAC UTF-8 leading byte",
        ));
    }
    let mut val = (first & ((1u8 << (7 - ones)) - 1)) as u64;
    for _ in 1..ones {
        let b = r.read_bits(8)? as u8;
        if (b & 0xC0) != 0x80 {
            return Err(RagdollError::InvalidSyntax(
                "invalid FLAC UTF-8 continuation byte",
            ));
        }
        val = (val << 6) | ((b & 0x3F) as u64);
    }
    Ok(val)
}

/// Writes a 31-bit frame number or 36-bit sample number in FLAC's extended UTF-8 encoding.
fn write_flac_utf8(w: &mut BitWriter, val: u64) {
    if val < 0x80 {
        w.write_bits(val, 8);
        return;
    }
    let bits = 64 - val.leading_zeros() as usize;
    let total_bytes = if bits <= 11 {
        2
    } else if bits <= 16 {
        3
    } else if bits <= 21 {
        4
    } else if bits <= 26 {
        5
    } else if bits <= 31 {
        6
    } else {
        7
    };
    let shift_top = (total_bytes - 1) * 6;
    let prefix = !((1u16 << (8 - total_bytes)) - 1) & 0xFF;
    let first = (prefix as u64) | ((val >> shift_top) & ((1u64 << (7 - total_bytes)) - 1));
    w.write_bits(first, 8);
    for i in (0..total_bytes - 1).rev() {
        let cont = 0x80 | ((val >> (i * 6)) & 0x3F);
        w.write_bits(cont, 8);
    }
}

impl FlacFrameHeader {
    /// Parses a FLAC frame header and verifies its 8-bit `crc8_flac` checksum.
    pub fn parse(data: &[u8], stream_info: Option<&FlacStreamInfo>) -> Result<Self> {
        if data.len() < 6 {
            return Err(RagdollError::BitstreamEndOfData);
        }
        let mut r = BitReader::new(data);
        let sync = r.read_bits(14)?;
        if sync != 0x3FFE {
            return Err(RagdollError::InvalidSyntax(
                "invalid FLAC frame sync code (expected 0x3FFE)",
            ));
        }
        let reserved0 = r.read_bit()?;
        if reserved0 {
            return Err(RagdollError::InvalidSyntax(
                "FLAC frame header reserved bit 14 must be 0",
            ));
        }
        let variable_block_size = r.read_bit()?;
        let bs_code = r.read_bits(4)? as u8;
        let sr_code = r.read_bits(4)? as u8;
        let ch_code = r.read_bits(4)? as u8;
        let bps_code = r.read_bits(3)? as u8;
        let reserved1 = r.read_bit()?;
        if reserved1 {
            return Err(RagdollError::InvalidSyntax(
                "FLAC frame header reserved bit 31 must be 0",
            ));
        }

        let frame_or_sample_num = read_flac_utf8(&mut r)?;

        let block_size = match bs_code {
            0 => {
                return Err(RagdollError::InvalidSyntax(
                    "reserved FLAC blocksize code 0",
                ));
            }
            1 => 192,
            2..=5 => 576 << (bs_code - 2),
            6 => (r.read_bits(8)? as usize) + 1,
            7 => (r.read_bits(16)? as usize) + 1,
            8..=15 => 256 << (bs_code - 8),
            _ => unreachable!(),
        };

        let sample_rate = match sr_code {
            0 => stream_info.map_or(48_000, |s| s.sample_rate),
            1 => 88_200,
            2 => 176_400,
            3 => 192_000,
            4 => 8_000,
            5 => 16_000,
            6 => 22_050,
            7 => 24_000,
            8 => 32_000,
            9 => 44_100,
            10 => 48_000,
            11 => 96_000,
            12 => (r.read_bits(8)? as u32) * 1000,
            13 => r.read_bits(16)? as u32,
            14 => (r.read_bits(16)? as u32) * 10,
            _ => {
                return Err(RagdollError::InvalidSyntax(
                    "invalid FLAC sample rate code 15",
                ));
            }
        };

        let channel_mode = match ch_code {
            0..=7 => FlacChannelMode::Independent(ch_code + 1),
            8 => FlacChannelMode::LeftSide,
            9 => FlacChannelMode::RightSide,
            10 => FlacChannelMode::MidSide,
            _ => {
                return Err(RagdollError::InvalidSyntax(
                    "reserved FLAC channel assignment code > 10",
                ));
            }
        };

        let bits_per_sample = match bps_code {
            0 => stream_info.map_or(16, |s| s.bits_per_sample),
            1 => 8,
            2 => 12,
            3 => {
                return Err(RagdollError::InvalidSyntax(
                    "reserved FLAC sample size code 3",
                ));
            }
            4 => 16,
            5 => 20,
            6 => 24,
            7 => 32,
            _ => unreachable!(),
        };

        let header_bytes_before_crc = r.bits_read() / 8;
        let header_crc8 = r.read_bits(8)? as u8;
        let computed_crc8 = crc8_flac(&data[..header_bytes_before_crc]);
        if header_crc8 != computed_crc8 {
            return Err(RagdollError::ChecksumMismatch {
                expected: header_crc8 as u32,
                actual: computed_crc8 as u32,
            });
        }

        Ok(Self {
            variable_block_size,
            block_size,
            sample_rate,
            channel_mode,
            bits_per_sample,
            frame_or_sample_num,
            header_crc8,
            header_len_bytes: header_bytes_before_crc + 1,
        })
    }

    /// Writes this FLAC frame header (including its `crc8_flac` byte) into `w`.
    pub fn write(&self, w: &mut BitWriter) {
        let mut hdr = BitWriter::with_capacity(16);
        hdr.write_bits(0x3FFE, 14);
        hdr.write_bit(false);
        hdr.write_bit(self.variable_block_size);

        let (bs_code, bs_extra): (u8, Option<(u64, u32)>) = match self.block_size {
            192 => (1, None),
            576 => (2, None),
            1152 => (3, None),
            2304 => (4, None),
            4608 => (5, None),
            256 => (8, None),
            512 => (9, None),
            1024 => (10, None),
            2048 => (11, None),
            4096 => (12, None),
            8192 => (13, None),
            16384 => (14, None),
            32768 => (15, None),
            n if n <= 256 => (6, Some(((n - 1) as u64, 8))),
            n => (7, Some(((n - 1) as u64, 16))),
        };

        let (sr_code, sr_extra): (u8, Option<(u64, u32)>) = match self.sample_rate {
            88_200 => (1, None),
            176_400 => (2, None),
            192_000 => (3, None),
            8_000 => (4, None),
            16_000 => (5, None),
            22_050 => (6, None),
            24_000 => (7, None),
            32_000 => (8, None),
            44_100 => (9, None),
            48_000 => (10, None),
            96_000 => (11, None),
            sr if sr.is_multiple_of(1000) && sr / 1000 <= 255 => {
                (12, Some(((sr / 1000) as u64, 8)))
            }
            sr if sr <= 65535 => (13, Some((sr as u64, 16))),
            sr if sr.is_multiple_of(10) && sr / 10 <= 65535 => (14, Some(((sr / 10) as u64, 16))),
            _ => (0, None),
        };

        let ch_code: u8 = match self.channel_mode {
            FlacChannelMode::Independent(c) => c.saturating_sub(1),
            FlacChannelMode::LeftSide => 8,
            FlacChannelMode::RightSide => 9,
            FlacChannelMode::MidSide => 10,
        };

        let bps_code: u8 = match self.bits_per_sample {
            8 => 1,
            12 => 2,
            16 => 4,
            20 => 5,
            24 => 6,
            32 => 7,
            _ => 0,
        };

        hdr.write_bits(bs_code as u64, 4);
        hdr.write_bits(sr_code as u64, 4);
        hdr.write_bits(ch_code as u64, 4);
        hdr.write_bits(bps_code as u64, 3);
        hdr.write_bit(false);
        write_flac_utf8(&mut hdr, self.frame_or_sample_num);

        if let Some((val, bits)) = bs_extra {
            hdr.write_bits(val, bits);
        }
        if let Some((val, bits)) = sr_extra {
            hdr.write_bits(val, bits);
        }

        let hdr_bytes = hdr.finish();
        let crc8 = crc8_flac(&hdr_bytes);
        for b in hdr_bytes {
            w.write_bits(b as u64, 8);
        }
        w.write_bits(crc8 as u64, 8);
    }
}

/// Decodes a FLAC Rice or Rice2 partitioned residual block into `dst[warm_up..block_size]`.
fn decode_residual_partitions(
    r: &mut BitReader<'_>,
    predictor_order: usize,
    block_size: usize,
    dst: &mut [i32],
) -> Result<()> {
    let coding_method = r.read_bits(2)? as u8;
    let (param_bits, escape_code): (u32, u32) = match coding_method {
        0 => (4, 0b1111),
        1 => (5, 0b11111),
        _ => {
            return Err(RagdollError::InvalidSyntax(
                "reserved FLAC residual coding method >= 2",
            ));
        }
    };
    let partition_order = r.read_bits(4)? as u32;
    let num_partitions = 1usize << partition_order;
    if !block_size.is_multiple_of(num_partitions) {
        return Err(RagdollError::InvalidSyntax(
            "FLAC block_size not divisible by num_partitions",
        ));
    }
    let part_samples = block_size >> partition_order;
    if part_samples < predictor_order && num_partitions > 1 {
        return Err(RagdollError::InvalidSyntax(
            "FLAC partition size smaller than predictor order",
        ));
    }

    let mut idx = predictor_order;
    for p in 0..num_partitions {
        let count = if p == 0 {
            if num_partitions == 1 {
                block_size
                    .checked_sub(predictor_order)
                    .ok_or(RagdollError::InvalidSyntax(
                        "FLAC predictor order exceeds block_size",
                    ))?
            } else {
                part_samples
                    .checked_sub(predictor_order)
                    .ok_or(RagdollError::InvalidSyntax(
                        "FLAC predictor order exceeds first partition",
                    ))?
            }
        } else {
            part_samples
        };
        if idx.saturating_add(count) > dst.len() {
            return Err(RagdollError::InvalidSyntax(
                "FLAC residual partition exceeds subframe block_size",
            ));
        }

        let rice_param = r.read_bits(param_bits)? as u32;
        if rice_param == escape_code {
            let raw_bits = r.read_bits(5)? as u32;
            if raw_bits == 0 {
                for s in &mut dst[idx..idx + count] {
                    *s = 0;
                }
            } else {
                for s in &mut dst[idx..idx + count] {
                    *s = r.read_signed_bits(raw_bits)? as i32;
                }
            }
        } else {
            for s in &mut dst[idx..idx + count] {
                *s = r.read_rice_signed(rice_param)? as i32;
            }
        }
        idx += count;
    }
    Ok(())
}

/// Reconstructs a FLAC Fixed predictor subframe (`order 0..=4`) in-place from warm-up samples + residuals.
///
/// Uses 64-bit intermediate arithmetic (`libavcodec/flacdec.c:decode_subframe_fixed`) to prevent overflow on 24-bit/Side channels.
pub fn reconstruct_fixed_subframe(order: usize, samples: &mut [i32]) -> Result<()> {
    let n = samples.len();
    match order {
        0 => {}
        1 => {
            for i in 1..n {
                samples[i] = (samples[i] as i64 + samples[i - 1] as i64) as i32;
            }
        }
        2 => {
            for i in 2..n {
                let pred = 2 * (samples[i - 1] as i64) - (samples[i - 2] as i64);
                samples[i] = (samples[i] as i64 + pred) as i32;
            }
        }
        3 => {
            for i in 3..n {
                let pred = 3 * (samples[i - 1] as i64) - 3 * (samples[i - 2] as i64)
                    + (samples[i - 3] as i64);
                samples[i] = (samples[i] as i64 + pred) as i32;
            }
        }
        4 => {
            for i in 4..n {
                let pred = 4 * (samples[i - 1] as i64) - 6 * (samples[i - 2] as i64)
                    + 4 * (samples[i - 3] as i64)
                    - (samples[i - 4] as i64);
                samples[i] = (samples[i] as i64 + pred) as i32;
            }
        }
        _ => {
            return Err(RagdollError::InvalidSyntax(
                "invalid FLAC fixed predictor order > 4",
            ));
        }
    }
    Ok(())
}

/// Reconstructs a FLAC FIR LPC subframe (`order 1..=32`) in-place using 64-bit accumulator arithmetic
/// (`libavcodec/flacdsp_lpc_template.c:flac_lpc_32_c`).
pub fn reconstruct_lpc_subframe(coeffs: &[i32], qlp_shift: i32, samples: &mut [i32]) -> Result<()> {
    let order = coeffs.len();
    if qlp_shift < 0 {
        return Err(RagdollError::InvalidSyntax(
            "negative FLAC LPC shift is invalid per RFC 9639",
        ));
    }
    let shift = qlp_shift as u32;
    for i in order..samples.len() {
        let mut sum = 0i64;
        for (j, &c) in coeffs.iter().enumerate() {
            sum += (c as i64) * (samples[i - 1 - j] as i64);
        }
        samples[i] = (samples[i] as i64 + (sum >> shift)) as i32;
    }
    Ok(())
}

/// Decodes a single FLAC subframe of `block_size` samples at `subframe_bps` bits per sample.
fn decode_subframe(
    r: &mut BitReader<'_>,
    block_size: usize,
    subframe_bps: u8,
    dst: &mut [i32],
) -> Result<()> {
    if r.read_bit()? {
        return Err(RagdollError::InvalidSyntax(
            "FLAC subframe header zero-padding bit must be 0",
        ));
    }
    let type_code = r.read_bits(6)? as u8;
    let has_wasted = r.read_bit()?;
    let wasted_bits = if has_wasted {
        (r.read_unary_zeros()? as u8).saturating_add(1)
    } else {
        0
    };
    if wasted_bits >= subframe_bps {
        return Err(RagdollError::InvalidSyntax(
            "FLAC wasted_bits >= subframe bits_per_sample",
        ));
    }
    let bps = (subframe_bps - wasted_bits) as u32;

    match type_code {
        0 => {
            // Constant subframe
            let val = r.read_signed_bits(bps)? as i32;
            dst[..block_size].fill(val);
        }
        1 => {
            // Verbatim subframe
            for s in &mut dst[..block_size] {
                *s = r.read_signed_bits(bps)? as i32;
            }
        }
        8..=12 => {
            // Fixed predictor subframe (order 0..=4)
            let order = (type_code & 7) as usize;
            if order > block_size {
                return Err(RagdollError::InvalidSyntax(
                    "FLAC fixed predictor order exceeds block_size",
                ));
            }
            for s in &mut dst[..order] {
                *s = r.read_signed_bits(bps)? as i32;
            }
            decode_residual_partitions(r, order, block_size, &mut dst[..block_size])?;
            reconstruct_fixed_subframe(order, &mut dst[..block_size])?;
        }
        32..=63 => {
            // FIR LPC subframe (order 1..=32)
            let order = ((type_code & 0x1F) as usize) + 1;
            if order > block_size {
                return Err(RagdollError::InvalidSyntax(
                    "FLAC LPC predictor order exceeds block_size",
                ));
            }
            for s in &mut dst[..order] {
                *s = r.read_signed_bits(bps)? as i32;
            }
            let qlp_prec = (r.read_bits(4)? as u32) + 1;
            if qlp_prec == 16 {
                return Err(RagdollError::InvalidSyntax(
                    "reserved FLAC LPC coefficient precision 1111",
                ));
            }
            let qlp_shift = r.read_signed_bits(5)? as i32;
            let mut coeffs = [0i32; 32];
            for c in &mut coeffs[..order] {
                *c = r.read_signed_bits(qlp_prec)? as i32;
            }
            decode_residual_partitions(r, order, block_size, &mut dst[..block_size])?;
            reconstruct_lpc_subframe(&coeffs[..order], qlp_shift, &mut dst[..block_size])?;
        }
        _ => {
            return Err(RagdollError::InvalidSyntax(
                "reserved FLAC subframe type code",
            ));
        }
    }

    if wasted_bits > 0 {
        for s in &mut dst[..block_size] {
            *s = s.wrapping_shl(wasted_bits as u32);
        }
    }
    Ok(())
}

/// Pure-Rust FLAC decoder supporting single-frame and full-stream decoding.
#[derive(Clone, Debug, Default)]
pub struct FlacDecoder {
    pub stream_info: Option<FlacStreamInfo>,
}

impl FlacDecoder {
    pub fn new(stream_info: Option<FlacStreamInfo>) -> Self {
        Self { stream_info }
    }

    /// Decodes a single FLAC frame from `data` (starting at the `0xFF 0xF8`/`0xFF 0xF9` sync code),
    /// returning `(FlacFrameHeader, planar_channels, bytes_consumed)`.
    ///
    /// Verifies both the 8-bit frame header CRC (`crc8_flac`) and the 16-bit frame footer CRC (`crc16_flac`).
    pub fn decode_frame_planar(
        &self,
        data: &[u8],
    ) -> Result<(FlacFrameHeader, Vec<Vec<i32>>, usize)> {
        let header = FlacFrameHeader::parse(data, self.stream_info.as_ref())?;
        let channels = header.channel_mode.channels() as usize;
        let block_size = header.block_size;

        let mut r = BitReader::new(&data[header.header_len_bytes..]);
        let mut subframes = vec![vec![0i32; block_size]; channels];
        for (c, sf) in subframes.iter_mut().enumerate() {
            let sf_bps = header.channel_mode.subframe_bps(header.bits_per_sample, c);
            decode_subframe(&mut r, block_size, sf_bps, sf)?;
        }

        // Byte-align with zero padding before reading the 16-bit footer CRC
        r.align_to_byte();
        let body_bytes = r.bits_read() / 8;
        let footer_offset = header.header_len_bytes + body_bytes;
        if footer_offset + 2 > data.len() {
            return Err(RagdollError::BitstreamEndOfData);
        }
        let expected_crc16 = r.read_bits(16)? as u16;
        let actual_crc16 = crc16_flac(&data[..footer_offset]);
        if expected_crc16 != actual_crc16 {
            return Err(RagdollError::ChecksumMismatch {
                expected: expected_crc16 as u32,
                actual: actual_crc16 as u32,
            });
        }

        // In-place stereo decorrelation reconstruction with zero heap allocations
        match header.channel_mode {
            FlacChannelMode::Independent(_) => {}
            FlacChannelMode::LeftSide => {
                let (left_slice, right_slice) = subframes.split_at_mut(1);
                let left = &left_slice[0];
                let side = &mut right_slice[0];
                for i in 0..block_size {
                    side[i] = left[i].wrapping_sub(side[i]);
                }
            }
            FlacChannelMode::RightSide => {
                let (left_slice, right_slice) = subframes.split_at_mut(1);
                let side = &mut left_slice[0];
                let right = &right_slice[0];
                for i in 0..block_size {
                    side[i] = side[i].wrapping_add(right[i]);
                }
            }
            FlacChannelMode::MidSide => {
                let (left_slice, right_slice) = subframes.split_at_mut(1);
                let mid = &mut left_slice[0];
                let side = &mut right_slice[0];
                for i in 0..block_size {
                    let m = mid[i] as i64;
                    let s = side[i] as i64;
                    let m2 = (m << 1) | (s & 1);
                    mid[i] = ((m2 + s) >> 1) as i32;
                    side[i] = ((m2 - s) >> 1) as i32;
                }
            }
        }
        let _ = (
            reconstruct_left_side_i32,
            reconstruct_right_side_i32,
            reconstruct_mid_side_i32,
        );

        Ok((header, subframes, footer_offset + 2))
    }

    /// Decodes a single FLAC frame directly into pre-allocated planar channel slices (`dst_channels[c][..block_size]`),
    /// returning `(FlacFrameHeader, bytes_consumed)` with zero heap allocations.
    pub fn decode_frame_into(
        &self,
        data: &[u8],
        dst_channels: &mut [Vec<i32>],
        sample_offset: usize,
    ) -> Result<(FlacFrameHeader, usize)> {
        let header = FlacFrameHeader::parse(data, self.stream_info.as_ref())?;
        let channels = header.channel_mode.channels() as usize;
        let block_size = header.block_size;

        let mut r = BitReader::new(&data[header.header_len_bytes..]);
        for (c, ch_vec) in dst_channels.iter_mut().enumerate().take(channels) {
            if ch_vec.len() < sample_offset + block_size {
                ch_vec.resize(sample_offset + block_size, 0);
            }
            let sf_bps = header.channel_mode.subframe_bps(header.bits_per_sample, c);
            decode_subframe(
                &mut r,
                block_size,
                sf_bps,
                &mut ch_vec[sample_offset..sample_offset + block_size],
            )?;
        }

        r.align_to_byte();
        let body_bytes = r.bits_read() / 8;
        let footer_offset = header.header_len_bytes + body_bytes;
        if footer_offset + 2 > data.len() {
            return Err(RagdollError::BitstreamEndOfData);
        }
        let expected_crc16 = r.read_bits(16)? as u16;
        let actual_crc16 = crc16_flac(&data[..footer_offset]);
        if expected_crc16 != actual_crc16 {
            return Err(RagdollError::ChecksumMismatch {
                expected: expected_crc16 as u32,
                actual: actual_crc16 as u32,
            });
        }

        match header.channel_mode {
            FlacChannelMode::Independent(_) => {}
            FlacChannelMode::LeftSide => {
                let (l_part, r_part) = dst_channels.split_at_mut(1);
                let left = &l_part[0][sample_offset..sample_offset + block_size];
                let side = &mut r_part[0][sample_offset..sample_offset + block_size];
                for i in 0..block_size {
                    side[i] = left[i].wrapping_sub(side[i]);
                }
            }
            FlacChannelMode::RightSide => {
                let (l_part, r_part) = dst_channels.split_at_mut(1);
                let side = &mut l_part[0][sample_offset..sample_offset + block_size];
                let right = &r_part[0][sample_offset..sample_offset + block_size];
                for i in 0..block_size {
                    side[i] = side[i].wrapping_add(right[i]);
                }
            }
            FlacChannelMode::MidSide => {
                let (l_part, r_part) = dst_channels.split_at_mut(1);
                let mid = &mut l_part[0][sample_offset..sample_offset + block_size];
                let side = &mut r_part[0][sample_offset..sample_offset + block_size];
                for i in 0..block_size {
                    let m = mid[i] as i64;
                    let s = side[i] as i64;
                    let m2 = (m << 1) | (s & 1);
                    mid[i] = ((m2 + s) >> 1) as i32;
                    side[i] = ((m2 - s) >> 1) as i32;
                }
            }
        }

        Ok((header, footer_offset + 2))
    }

    /// Decodes a single FLAC frame into an interleaved [`AudioFrame`], returning `(AudioFrame, bytes_consumed)`.
    pub fn decode_frame(&self, data: &[u8]) -> Result<(AudioFrame, usize)> {
        let (header, planar, consumed) = self.decode_frame_planar(data)?;
        let frame = planar_i32_to_audio_frame(&planar, header.sample_rate, header.bits_per_sample)?;
        Ok((frame, consumed))
    }

    /// Decodes an entire `.flac` file or stream into a single contiguous [`AudioFrame`]
    /// with pre-allocated planar buffers (zero per-frame heap allocations).
    pub fn decode_stream(data: &[u8]) -> Result<(FlacStreamInfo, AudioFrame)> {
        let (info, mut pos) = parse_flac_stream_header(data)?;
        let dec = Self::new(Some(info.clone()));
        let init_cap = (info.total_samples as usize).min(data.len().saturating_mul(8));
        let mut all_planar = vec![vec![0i32; init_cap]; info.channels as usize];
        let mut sample_offset = 0usize;

        while pos < data.len() {
            // Fast check for trailing zero padding at EOF
            if data[pos] == 0 && data[pos..].iter().all(|&b| b == 0) {
                break;
            }
            let (hdr, consumed) =
                dec.decode_frame_into(&data[pos..], &mut all_planar, sample_offset)?;
            sample_offset += hdr.block_size;
            pos += consumed;
        }
        for ch in &mut all_planar {
            ch.truncate(sample_offset);
        }

        let audio = planar_i32_to_audio_frame(&all_planar, info.sample_rate, info.bits_per_sample)?;
        if info.md5_signature != [0u8; 16] {
            let actual_md5 = compute_flac_pcm_md5_planar(&all_planar, info.bits_per_sample);
            if actual_md5 != info.md5_signature {
                return Err(RagdollError::ChecksumMismatch {
                    expected: u32::from_be_bytes([
                        info.md5_signature[0],
                        info.md5_signature[1],
                        info.md5_signature[2],
                        info.md5_signature[3],
                    ]),
                    actual: u32::from_be_bytes([
                        actual_md5[0],
                        actual_md5[1],
                        actual_md5[2],
                        actual_md5[3],
                    ]),
                });
            }
        }
        Ok((info, audio))
    }
}

/// Pure-Rust RFC 1321 MD5 hasher for FLAC `STREAMINFO` audio payload signatures (RFC 9639 Section 8.2).
pub fn md5_digest(data: &[u8]) -> [u8; 16] {
    const K: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
        0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
        0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
        0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
        0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
        0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
        0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
        0xeb86d391,
    ];
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];

    let mut a0 = 0x67452301u32;
    let mut b0 = 0xefcdab89u32;
    let mut c0 = 0x98badcfeu32;
    let mut d0 = 0x10325476u32;

    let process_block =
        |block: &[u8; 64], a0: &mut u32, b0: &mut u32, c0: &mut u32, d0: &mut u32| {
            let mut m = [0u32; 16];
            for (i, slot) in m.iter_mut().enumerate() {
                let o = i * 4;
                *slot = u32::from_le_bytes([block[o], block[o + 1], block[o + 2], block[o + 3]]);
            }
            let mut a = *a0;
            let mut b = *b0;
            let mut c = *c0;
            let mut d = *d0;
            for i in 0..64 {
                let (f, g) = if i < 16 {
                    ((b & c) | (!b & d), i)
                } else if i < 32 {
                    ((d & b) | (!d & c), (5 * i + 1) & 15)
                } else if i < 48 {
                    (b ^ c ^ d, (3 * i + 5) & 15)
                } else {
                    (c ^ (b | !d), (7 * i) & 15)
                };
                let temp = d;
                d = c;
                c = b;
                b = b.wrapping_add(
                    a.wrapping_add(f)
                        .wrapping_add(K[i])
                        .wrapping_add(m[g])
                        .rotate_left(S[i]),
                );
                a = temp;
            }
            *a0 = a0.wrapping_add(a);
            *b0 = b0.wrapping_add(b);
            *c0 = c0.wrapping_add(c);
            *d0 = d0.wrapping_add(d);
        };

    let full_blocks = data.len() / 64;
    for b_idx in 0..full_blocks {
        let blk: &[u8; 64] = data[b_idx * 64..(b_idx + 1) * 64].try_into().unwrap();
        process_block(blk, &mut a0, &mut b0, &mut c0, &mut d0);
    }

    let rem = &data[full_blocks * 64..];
    let mut tail = [0u8; 128];
    tail[..rem.len()].copy_from_slice(rem);
    tail[rem.len()] = 0x80;
    let total_bits = (data.len() as u64).wrapping_mul(8);
    let pad_len = if rem.len() < 56 { 64 } else { 128 };
    tail[pad_len - 8..pad_len].copy_from_slice(&total_bits.to_le_bytes());

    let blk0: &[u8; 64] = tail[..64].try_into().unwrap();
    process_block(blk0, &mut a0, &mut b0, &mut c0, &mut d0);
    if pad_len == 128 {
        let blk1: &[u8; 64] = tail[64..128].try_into().unwrap();
        process_block(blk1, &mut a0, &mut b0, &mut c0, &mut d0);
    }

    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&a0.to_le_bytes());
    out[4..8].copy_from_slice(&b0.to_le_bytes());
    out[8..12].copy_from_slice(&c0.to_le_bytes());
    out[12..16].copy_from_slice(&d0.to_le_bytes());
    out
}

/// Computes the RFC 9639 Section 8.2 MD5 signature over planar `i32` audio channels at `bits_per_sample`.
pub fn compute_flac_pcm_md5_planar(planar: &[Vec<i32>], bits_per_sample: u8) -> [u8; 16] {
    if planar.is_empty() {
        return [0u8; 16];
    }
    let n = planar[0].len();
    let ch = planar.len();
    let bytes_per_sample = (bits_per_sample as usize).div_ceil(8);
    let mut raw = Vec::with_capacity(n * ch * bytes_per_sample);

    match bytes_per_sample {
        1 => {
            for i in 0..n {
                for c in planar {
                    raw.push(c[i] as i8 as u8);
                }
            }
        }
        2 => {
            for i in 0..n {
                for c in planar {
                    raw.extend_from_slice(&(c[i] as i16).to_le_bytes());
                }
            }
        }
        3 => {
            for i in 0..n {
                for c in planar {
                    let le = c[i].to_le_bytes();
                    raw.extend_from_slice(&[le[0], le[1], le[2]]);
                }
            }
        }
        _ => {
            for i in 0..n {
                for c in planar {
                    raw.extend_from_slice(&c[i].to_le_bytes());
                }
            }
        }
    }
    md5_digest(&raw)
}

/// Computes Fixed predictor residuals (`order 0..=4`) for `samples` into `residuals[order..]`.
pub fn compute_fixed_residuals(order: usize, samples: &[i32], residuals: &mut [i32]) {
    let n = samples.len();
    residuals[..order].copy_from_slice(&samples[..order]);
    match order {
        0 => residuals.copy_from_slice(samples),
        1 => {
            for i in 1..n {
                residuals[i] = (samples[i] as i64 - samples[i - 1] as i64) as i32;
            }
        }
        2 => {
            for i in 2..n {
                let pred = 2 * (samples[i - 1] as i64) - (samples[i - 2] as i64);
                residuals[i] = (samples[i] as i64 - pred) as i32;
            }
        }
        3 => {
            for i in 3..n {
                let pred = 3 * (samples[i - 1] as i64) - 3 * (samples[i - 2] as i64)
                    + (samples[i - 3] as i64);
                residuals[i] = (samples[i] as i64 - pred) as i32;
            }
        }
        4 => {
            for i in 4..n {
                let pred = 4 * (samples[i - 1] as i64) - 6 * (samples[i - 2] as i64)
                    + 4 * (samples[i - 3] as i64)
                    - (samples[i - 4] as i64);
                residuals[i] = (samples[i] as i64 - pred) as i32;
            }
        }
        _ => {}
    }
}

/// Computes autocorrelation lags `r[0..=max_order]` in a single pass and solves the Yule-Walker
/// equations via Levinson-Durbin recursion (`libavcodec/lpc.h:compute_lpc_coefs`), returning
/// quantized 12-bit QLP integer coefficients, shift, forward LPC residuals, and L1 residual sum.
///
/// Uses early MSE rejection against `best_fixed_l1` to skip forward residual synthesis whenever
/// the Fixed predictor is already superior.
fn try_compute_lpc8_subframe(
    samples: &[i32],
    best_fixed_l1: u64,
) -> Option<(Vec<i32>, i32, Vec<i32>, u64)> {
    const ORDER: usize = 8;
    let n = samples.len();
    if n <= ORDER * 4 || best_fixed_l1 == 0 {
        return None;
    }

    // Single-pass autocorrelation R[0..=8]: reads `samples` from L1 cache once instead of 9 times
    let mut r = [0.0f64; ORDER + 1];
    for i in 0..ORDER {
        let si = samples[i] as f64;
        for lag in 0..=i {
            r[lag] += si * (samples[i - lag] as f64);
        }
    }
    let mut r0 = r[0];
    let mut r1 = r[1];
    let mut r2 = r[2];
    let mut r3 = r[3];
    let mut r4 = r[4];
    let mut r5 = r[5];
    let mut r6 = r[6];
    let mut r7 = r[7];
    let mut r8 = r[8];

    for i in ORDER..n {
        let s = samples[i] as f64;
        r0 += s * s;
        r1 += s * (samples[i - 1] as f64);
        r2 += s * (samples[i - 2] as f64);
        r3 += s * (samples[i - 3] as f64);
        r4 += s * (samples[i - 4] as f64);
        r5 += s * (samples[i - 5] as f64);
        r6 += s * (samples[i - 6] as f64);
        r7 += s * (samples[i - 7] as f64);
        r8 += s * (samples[i - 8] as f64);
    }
    r = [r0, r1, r2, r3, r4, r5, r6, r7, r8];

    if r[0] <= 1.0 {
        return None;
    }

    // Levinson-Durbin recursion
    let mut lpc = [0.0f64; ORDER];
    let mut prev = [0.0f64; ORDER];
    let mut err = r[0];
    for m in 0..ORDER {
        let mut num = r[m + 1];
        for j in 0..m {
            num -= prev[j] * r[m - j];
        }
        let k = num / err;
        if !k.is_finite() || k.abs() >= 0.9999 {
            return None;
        }
        lpc[m] = k;
        for j in 0..m {
            lpc[j] = prev[j] - k * prev[m - 1 - j];
        }
        prev[..=m].copy_from_slice(&lpc[..=m]);
        err *= 1.0 - k * k;
        if err <= 1e-9 {
            break;
        }
    }

    // Early rejection: if theoretical LPC L1 lower bound exceeds `best_fixed_l1`, skip residual pass
    let est_l1 = (err.max(0.0) * (n as f64) * 0.60).sqrt();
    if est_l1 >= (best_fixed_l1 as f64) {
        return None;
    }

    // Quantize floating-point LPC coefficients to 12-bit signed QLP integers (`qlp_prec = 12`, max mag = 2047)
    let max_abs = lpc.iter().fold(0.0f64, |acc, &c| acc.max(c.abs()));
    if max_abs <= 1e-6 || max_abs > 16.0 {
        return None;
    }
    let mut qlp_shift = 10i32;
    while qlp_shift > 0 && (max_abs * ((1 << qlp_shift) as f64)) > 1800.0 {
        qlp_shift -= 1;
    }
    let scale = (1 << qlp_shift) as f64;
    let mut q_coeffs = [0i32; ORDER];
    for i in 0..ORDER {
        q_coeffs[i] = (lpc[i] * scale).round().clamp(-2048.0, 2047.0) as i32;
    }
    if q_coeffs.iter().all(|&c| c == 0) {
        return None;
    }

    // Unrolled 8-tap forward residual synthesis with fused L1 accumulation
    let c0 = q_coeffs[0] as i64;
    let c1 = q_coeffs[1] as i64;
    let c2 = q_coeffs[2] as i64;
    let c3 = q_coeffs[3] as i64;
    let c4 = q_coeffs[4] as i64;
    let c5 = q_coeffs[5] as i64;
    let c6 = q_coeffs[6] as i64;
    let c7 = q_coeffs[7] as i64;

    let mut residuals = vec![0i32; n];
    residuals[..ORDER].copy_from_slice(&samples[..ORDER]);
    let mut l1_sum = 0u64;

    for i in ORDER..n {
        let sum = c0 * (samples[i - 1] as i64)
            + c1 * (samples[i - 2] as i64)
            + c2 * (samples[i - 3] as i64)
            + c3 * (samples[i - 4] as i64)
            + c4 * (samples[i - 5] as i64)
            + c5 * (samples[i - 6] as i64)
            + c6 * (samples[i - 7] as i64)
            + c7 * (samples[i - 8] as i64);
        let pred = sum >> qlp_shift;
        let diff = (samples[i] as i64) - pred;
        if diff < (i32::MIN as i64 / 2) || diff > (i32::MAX as i64 / 2) {
            return None;
        }
        residuals[i] = diff as i32;
        l1_sum += diff.unsigned_abs();
    }

    Some((q_coeffs.to_vec(), qlp_shift, residuals, l1_sum))
}

/// Solves for optimal Rice parameter `k` (`0..=14`) given folded unsigned residual sum `sum` and `n` samples
/// (`libavcodec/flacenc.c:find_optimal_param`).
#[inline]
fn find_optimal_rice_param(sum: u64, n: usize) -> u32 {
    let half_n = (n >> 1) as u64;
    if sum <= half_n || n == 0 {
        return 0;
    }
    let mean = (sum - half_n) / (n as u64);
    if mean == 0 {
        0
    } else {
        (63 - mean.leading_zeros()).min(14)
    }
}

/// Analytical bit count for a Rice partition (`libavcodec/flacenc.c:rice_encode_count`).
#[inline]
fn rice_encode_count(sum: u64, n: usize, k: u32) -> usize {
    let half_n = (n >> 1) as u64;
    n * ((k as usize) + 1) + ((sum.saturating_sub(half_n) >> k) as usize)
}

/// Evaluates adaptive multi-partition Rice coding (`partition_order = 0..=4`, up to 16 partitions)
/// in a single leaf pass + binary tree reduction (`libavcodec/flacenc.c:calc_rice_params`).
///
/// Returns `(best_porder, rice_params[0..16], total_residual_bits)`.
fn find_best_rice_partitions(order: usize, residuals: &[i32]) -> (usize, [u32; 16], usize) {
    let n = residuals.len();
    // Determine highest valid partition order `pmax <= 4` such that `n % (1 << pmax) == 0`
    // and `(n >> pmax) > order` and `(n >> pmax) >= 16`.
    let mut pmax = (n.trailing_zeros() as usize).min(4);
    while pmax > 0 && ((n >> pmax) <= order || (n >> pmax) < 16) {
        pmax -= 1;
    }

    // 1. Single pass over `residuals[order..n]` to populate leaf sums at level `pmax`
    let mut sums = [[0u64; 16]; 5];
    let top_parts = 1usize << pmax;
    let top_psize = n >> pmax;
    let mut idx = order;
    for (p, sum_slot) in sums[pmax].iter_mut().enumerate().take(top_parts) {
        let part_end = (p + 1) * top_psize;
        let mut s = 0u64;
        for &r in &residuals[idx..part_end] {
            let v = r as i64;
            s += ((v << 1) ^ (v >> 63)) as u64;
        }
        *sum_slot = s;
        idx = part_end;
    }

    // 2. Binary tree reduction from `pmax` down to `0` (at most 15 additions total)
    for level in (0..pmax).rev() {
        let parts = 1usize << level;
        for i in 0..parts {
            sums[level][i] = sums[level + 1][2 * i] + sums[level + 1][2 * i + 1];
        }
    }

    // 3. Evaluate bit cost for each partition order `porder in 0..=pmax`
    let mut best_porder = 0usize;
    let mut best_params = [0u32; 16];
    let mut best_bits = usize::MAX;

    for (porder, level_sums) in sums.iter().enumerate().take(pmax + 1) {
        let parts = 1usize << porder;
        let psize = n >> porder;
        let mut total_bits = 2 + 4 + 4 * parts; // coding_method (2) + porder (4) + 4 bits per partition param
        let mut cand_params = [0u32; 16];

        for p in 0..parts {
            let cnt = if p == 0 { psize - order } else { psize };
            let sum = level_sums[p];
            let k = find_optimal_rice_param(sum, cnt);
            cand_params[p] = k;
            total_bits += rice_encode_count(sum, cnt, k);
        }

        if total_bits < best_bits {
            best_bits = total_bits;
            best_porder = porder;
            best_params = cand_params;
        }
    }

    (best_porder, best_params, best_bits)
}

/// Writes multi-partition Rice-coded residuals (`partition_order = 0..=4`) into `w`.
#[inline]
fn write_residual_partitions(
    w: &mut BitWriter,
    order: usize,
    porder: usize,
    params: &[u32; 16],
    residuals: &[i32],
) {
    let n = residuals.len();
    let parts = 1usize << porder;
    let psize = n >> porder;

    w.write_bits(0, 2); // coding_method = 0 (4-bit Rice parameter)
    w.write_bits(porder as u64, 4);

    let mut idx = order;
    for (p, &k) in params.iter().enumerate().take(parts) {
        let part_end = (p + 1) * psize;
        w.write_bits(k as u64, 4);
        for &r in &residuals[idx..part_end] {
            w.write_rice_signed(r as i64, k);
        }
        idx = part_end;
    }
}

/// Encodes a single FLAC subframe (choosing Constant, Fixed 0..=4, FIR LPC-8, or Verbatim) into `w`
/// in a single bitstream pass without temporary `BitWriter` allocations.
fn encode_subframe(w: &mut BitWriter, raw_samples: &[i32], subframe_bps: u8) {
    let n = raw_samples.len();
    assert!(n > 0);

    // 1. Constant check + wasted_bits detection in a single pass
    let first = raw_samples[0];
    let mut is_constant = true;
    let mut or_mask = 0u32;
    for &s in raw_samples {
        if s != first {
            is_constant = false;
        }
        or_mask |= s as u32;
    }
    if is_constant {
        w.write_bit(false); // zero pad
        w.write_bits(0, 6); // Constant (000000)
        w.write_bit(false); // no wasted bits
        w.write_signed_bits(first as i64, subframe_bps as u32);
        return;
    }

    let wasted_bits = if or_mask == 0 {
        0
    } else {
        (or_mask.trailing_zeros() as u8).min(subframe_bps.saturating_sub(2))
    };
    let effective_bps = subframe_bps - wasted_bits;
    let shifted_buf: Vec<i32>;
    let samples: &[i32] = if wasted_bits > 0 {
        shifted_buf = raw_samples.iter().map(|&s| s >> wasted_bits).collect();
        &shifted_buf
    } else {
        raw_samples
    };

    // 2. Evaluate all 5 Fixed predictor orders (0..=4) in a SINGLE fused NEON pass (`libFLAC/fixed.c`)
    let max_fixed_order = 4.min(n.saturating_sub(1));
    let l1_orders = flac_eval_fixed_orders_l1(samples);
    let mut best_fixed_order = 0usize;
    let mut best_l1 = l1_orders[0];
    for (order, &l1) in l1_orders
        .iter()
        .enumerate()
        .take(max_fixed_order + 1)
        .skip(1)
    {
        if l1 < best_l1 {
            best_l1 = l1;
            best_fixed_order = order;
        }
    }

    // 3. Evaluate FIR LPC (order 8) on blocks >= 64 samples when Fixed order >= 2 is chosen
    let mut best_lpc: Option<(Vec<i32>, i32, Vec<i32>)> = None;
    if n >= 64
        && best_fixed_order >= 2
        && let Some((coeffs, shift, lpc_res, lpc_l1)) = try_compute_lpc8_subframe(samples, best_l1)
        && lpc_l1 + (lpc_l1 / 20) < best_l1
    {
        best_lpc = Some((coeffs, shift, lpc_res));
    }

    // 4. Compute multi-partition Rice parameters for the winning predictor vs Verbatim and emit once into `w`
    let wasted_flag_bits = if wasted_bits > 0 {
        1 + (wasted_bits as usize)
    } else {
        1
    };
    let verbatim_bits = 7 + wasted_flag_bits + n * (effective_bps as usize);

    if let Some((coeffs, shift, lpc_res)) = best_lpc {
        let order = coeffs.len();
        let (porder, params, r_bits) = find_best_rice_partitions(order, &lpc_res);
        let lpc_bits =
            7 + wasted_flag_bits + order * (effective_bps as usize) + 4 + 5 + order * 12 + r_bits;
        if lpc_bits <= verbatim_bits {
            w.write_bit(false);
            w.write_bits((32 + order - 1) as u64, 6);
            write_wasted_bits_flag(w, wasted_bits);
            for &s in &samples[..order] {
                w.write_signed_bits(s as i64, effective_bps as u32);
            }
            w.write_bits(12 - 1, 4);
            w.write_signed_bits(shift as i64, 5);
            for &c in &coeffs {
                w.write_signed_bits(c as i64, 12);
            }
            write_residual_partitions(w, order, porder, &params, &lpc_res);
            return;
        }
    } else {
        let mut best_residuals = vec![0i32; n];
        compute_fixed_residuals(best_fixed_order, samples, &mut best_residuals);
        let (porder, params, r_bits) = find_best_rice_partitions(best_fixed_order, &best_residuals);
        let fixed_bits =
            7 + wasted_flag_bits + best_fixed_order * (effective_bps as usize) + r_bits;
        if fixed_bits <= verbatim_bits {
            w.write_bit(false);
            w.write_bits((8 + best_fixed_order) as u64, 6);
            write_wasted_bits_flag(w, wasted_bits);
            for &s in &samples[..best_fixed_order] {
                w.write_signed_bits(s as i64, effective_bps as u32);
            }
            write_residual_partitions(w, best_fixed_order, porder, &params, &best_residuals);
            return;
        }
    }

    // Verbatim fallback when incompressible noise exceeds raw PCM size
    w.write_bit(false);
    w.write_bits(1, 6);
    write_wasted_bits_flag(w, wasted_bits);
    for &s in samples {
        w.write_signed_bits(s as i64, effective_bps as u32);
    }
}

#[inline]
fn write_wasted_bits_flag(w: &mut BitWriter, wasted_bits: u8) {
    if wasted_bits > 0 {
        w.write_bit(true);
        for _ in 0..(wasted_bits - 1) {
            w.write_bit(false);
        }
        w.write_bit(true);
    } else {
        w.write_bit(false);
    }
}

/// Estimates the optimal stereo decorrelation mode (`Independent`, `LeftSide`, `RightSide`, `MidSide`)
/// using 2nd-order residual Rice bit-cost (`libavcodec/flacenc.c:estimate_stereo_mode`).
#[inline]
fn estimate_stereo_mode(left: &[i32], right: &[i32]) -> FlacChannelMode {
    let n = left.len();
    if n < 3 {
        return FlacChannelMode::Independent(2);
    }
    let mut sum = [0u64; 4]; // [left, right, mid, side]
    for i in 2..n {
        let lt = (left[i] as i64) - 2 * (left[i - 1] as i64) + (left[i - 2] as i64);
        let rt = (right[i] as i64) - 2 * (right[i - 1] as i64) + (right[i - 2] as i64);
        sum[0] += lt.unsigned_abs();
        sum[1] += rt.unsigned_abs();
        sum[2] += ((lt + rt) >> 1).unsigned_abs();
        sum[3] += (lt - rt).unsigned_abs();
    }

    let mut bits = [0usize; 4];
    for i in 0..4 {
        let s2 = sum[i].saturating_mul(2);
        let k = find_optimal_rice_param(s2, n);
        bits[i] = rice_encode_count(s2, n, k);
    }

    let scores = [
        bits[0] + bits[1], // Independent (Left + Right)
        bits[0] + bits[3], // LeftSide    (Left + Side)
        bits[3] + bits[1], // RightSide   (Side + Right)
        bits[2] + bits[3], // MidSide     (Mid + Side)
    ];

    let mut best = 0usize;
    for i in 1..4 {
        if scores[i] < scores[best] {
            best = i;
        }
    }
    match best {
        1 => FlacChannelMode::LeftSide,
        2 => FlacChannelMode::RightSide,
        3 => FlacChannelMode::MidSide,
        _ => FlacChannelMode::Independent(2),
    }
}

/// Pure-Rust lossless FLAC encoder.
#[derive(Clone, Debug)]
pub struct FlacEncoder {
    pub block_size: usize,
    pub sample_rate: u32,
    pub channels: u8,
    pub bits_per_sample: u8,
    pub compute_md5: bool,
}

impl FlacEncoder {
    pub fn new(sample_rate: u32, channels: u8, bits_per_sample: u8) -> Self {
        Self {
            block_size: 4096,
            sample_rate,
            channels,
            bits_per_sample,
            compute_md5: false,
        }
    }

    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size.clamp(16, 65535);
        self
    }

    /// Enables or disables RFC 9639 `STREAMINFO` MD5 audio payload signature computation.
    pub fn with_md5(mut self, enable: bool) -> Self {
        self.compute_md5 = enable;
        self
    }

    /// Encodes a single FLAC frame from channel slices `&[&[i32]]` without cloning independent channel vectors.
    pub fn encode_frame_slices(&self, planar: &[&[i32]], frame_number: u64) -> Result<Vec<u8>> {
        if planar.is_empty() || planar.len() != self.channels as usize {
            return Err(RagdollError::InvalidData(
                "FLAC encode_frame_slices channel count mismatch".into(),
            ));
        }
        let block_size = planar[0].len();
        if block_size == 0 {
            return Err(RagdollError::InvalidData(
                "FLAC encode_frame_slices empty block".into(),
            ));
        }

        let channel_mode = if self.channels == 2 {
            estimate_stereo_mode(planar[0], planar[1])
        } else {
            FlacChannelMode::Independent(self.channels)
        };

        let header = FlacFrameHeader {
            variable_block_size: false,
            block_size,
            sample_rate: self.sample_rate,
            channel_mode,
            bits_per_sample: self.bits_per_sample,
            frame_or_sample_num: frame_number,
            header_crc8: 0,
            header_len_bytes: 0,
        };

        let mut w = BitWriter::with_capacity(block_size * (self.channels as usize) * 2 + 32);
        header.write(&mut w);

        match channel_mode {
            FlacChannelMode::Independent(_) => {
                for (c, &sf) in planar.iter().enumerate() {
                    let sf_bps = channel_mode.subframe_bps(self.bits_per_sample, c);
                    encode_subframe(&mut w, sf, sf_bps);
                }
            }
            FlacChannelMode::LeftSide => {
                let mut side = vec![0i32; block_size];
                for i in 0..block_size {
                    side[i] = planar[0][i] - planar[1][i];
                }
                encode_subframe(
                    &mut w,
                    planar[0],
                    channel_mode.subframe_bps(self.bits_per_sample, 0),
                );
                encode_subframe(
                    &mut w,
                    &side,
                    channel_mode.subframe_bps(self.bits_per_sample, 1),
                );
            }
            FlacChannelMode::RightSide => {
                let mut side = vec![0i32; block_size];
                for i in 0..block_size {
                    side[i] = planar[0][i] - planar[1][i];
                }
                encode_subframe(
                    &mut w,
                    &side,
                    channel_mode.subframe_bps(self.bits_per_sample, 0),
                );
                encode_subframe(
                    &mut w,
                    planar[1],
                    channel_mode.subframe_bps(self.bits_per_sample, 1),
                );
            }
            FlacChannelMode::MidSide => {
                let mut mid = vec![0i32; block_size];
                let mut side = vec![0i32; block_size];
                decorrelate_mid_side_i32(planar[0], planar[1], &mut mid, &mut side);
                encode_subframe(
                    &mut w,
                    &mid,
                    channel_mode.subframe_bps(self.bits_per_sample, 0),
                );
                encode_subframe(
                    &mut w,
                    &side,
                    channel_mode.subframe_bps(self.bits_per_sample, 1),
                );
            }
        }

        w.align_to_byte();
        let mut frame_bytes = w.finish();
        let crc16 = crc16_flac(&frame_bytes);
        frame_bytes.extend_from_slice(&crc16.to_be_bytes());
        Ok(frame_bytes)
    }

    /// Encodes a single FLAC frame from `planar_channels` (all channels having `block_size` samples).
    pub fn encode_frame_planar(&self, planar: &[Vec<i32>], frame_number: u64) -> Result<Vec<u8>> {
        let refs: Vec<&[i32]> = planar.iter().map(|v| v.as_slice()).collect();
        self.encode_frame_slices(&refs, frame_number)
    }

    /// Encodes an [`AudioFrame`] into `(FlacStreamInfo, Vec<(frame_bytes, frame_samples)>)` for container muxers (`.flac`, `.ogg`, `.mkv`, `.mp4`).
    #[allow(clippy::type_complexity)]
    pub fn encode_frames(
        &self,
        audio: &AudioFrame,
    ) -> Result<(FlacStreamInfo, Vec<(Vec<u8>, usize)>)> {
        let (planar, detected_bps) = audio_frame_to_planar_i32(audio)?;
        let bps = if self.bits_per_sample != 0 {
            self.bits_per_sample
        } else {
            detected_bps
        };
        let enc = Self {
            block_size: self.block_size,
            sample_rate: audio.sample_rate,
            channels: audio.channels,
            bits_per_sample: bps,
            compute_md5: self.compute_md5,
        };

        let total_samples = audio.num_samples;
        let num_frames = if total_samples == 0 {
            0
        } else {
            total_samples.div_ceil(enc.block_size)
        };
        let mut packets = Vec::with_capacity(num_frames);
        let mut min_frame_size = u32::MAX;
        let mut max_frame_size = 0u32;

        if num_frames >= 8 {
            let workers = ragdoll_core::CpuTopology::detect()
                .optimal_threads(ragdoll_core::global_power_mode(), 0)
                .min(num_frames);
            let next_idx = std::sync::atomic::AtomicUsize::new(0);
            let slots: Vec<std::sync::Mutex<Option<Result<(Vec<u8>, usize)>>>> = (0..num_frames)
                .map(|_| std::sync::Mutex::new(None))
                .collect();

            std::thread::scope(|s| {
                for _ in 0..workers {
                    s.spawn(|| {
                        ragdoll_core::apply_current_power_mode_to_thread();
                        let mut slice_refs = Vec::with_capacity(planar.len());
                        loop {
                            let f_idx = next_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if f_idx >= num_frames {
                                break;
                            }
                            let offset = f_idx * enc.block_size;
                            let count = (total_samples - offset).min(enc.block_size);
                            slice_refs.clear();
                            for ch in &planar {
                                slice_refs.push(&ch[offset..offset + count]);
                            }
                            let res = enc
                                .encode_frame_slices(&slice_refs, f_idx as u64)
                                .map(|b| (b, count));
                            if let Ok(mut g) = slots[f_idx].lock() {
                                *g = Some(res);
                            }
                        }
                    });
                }
            });

            for slot in slots {
                let (packet, count) = slot.into_inner().unwrap().unwrap()?;
                let pkt_len = packet.len() as u32;
                min_frame_size = min_frame_size.min(pkt_len);
                max_frame_size = max_frame_size.max(pkt_len);
                packets.push((packet, count));
            }
        } else {
            let mut offset = 0usize;
            let mut frame_num = 0u64;
            let mut slice_refs = Vec::with_capacity(planar.len());
            while offset < total_samples {
                let count = (total_samples - offset).min(enc.block_size);
                slice_refs.clear();
                for ch in &planar {
                    slice_refs.push(&ch[offset..offset + count]);
                }
                let packet = enc.encode_frame_slices(&slice_refs, frame_num)?;
                let pkt_len = packet.len() as u32;
                min_frame_size = min_frame_size.min(pkt_len);
                max_frame_size = max_frame_size.max(pkt_len);
                packets.push((packet, count));
                offset += count;
                frame_num += 1;
            }
        }

        if total_samples == 0 {
            min_frame_size = 0;
        }
        let nominal_bs = (enc.block_size as u16).max(16);
        let md5_signature = if enc.compute_md5 {
            if audio.format == ragdoll_core::SampleFormat::S16le && bps == 16 {
                let byte_len = total_samples * (audio.channels as usize) * 2;
                md5_digest(&audio.data.as_slice()[..byte_len])
            } else {
                compute_flac_pcm_md5_planar(&planar, bps)
            }
        } else {
            [0u8; 16]
        };

        let stream_info = FlacStreamInfo {
            min_block_size: nominal_bs,
            max_block_size: nominal_bs,
            min_frame_size,
            max_frame_size,
            sample_rate: enc.sample_rate,
            channels: enc.channels,
            bits_per_sample: enc.bits_per_sample,
            total_samples: total_samples as u64,
            md5_signature,
        };
        Ok((stream_info, packets))
    }

    /// Encodes a complete `.flac` file (`fLaC` magic + `STREAMINFO` + framed audio packets) from an [`AudioFrame`].
    pub fn encode_stream(&self, audio: &AudioFrame) -> Result<Vec<u8>> {
        let (planar, detected_bps) = audio_frame_to_planar_i32(audio)?;
        let bps = if self.bits_per_sample != 0 {
            self.bits_per_sample
        } else {
            detected_bps
        };
        let enc = Self {
            block_size: self.block_size,
            sample_rate: audio.sample_rate,
            channels: audio.channels,
            bits_per_sample: bps,
            compute_md5: self.compute_md5,
        };

        let total_samples = audio.num_samples;
        let num_frames = if total_samples == 0 {
            0
        } else {
            total_samples.div_ceil(enc.block_size)
        };
        let mut frames_encoded = Vec::new();
        let mut min_frame_size = u32::MAX;
        let mut max_frame_size = 0u32;

        if num_frames >= 8 {
            let workers = ragdoll_core::CpuTopology::detect()
                .optimal_threads(ragdoll_core::global_power_mode(), 0)
                .min(num_frames);
            let next_idx = std::sync::atomic::AtomicUsize::new(0);
            let slots: Vec<std::sync::Mutex<Option<Result<Vec<u8>>>>> = (0..num_frames)
                .map(|_| std::sync::Mutex::new(None))
                .collect();

            std::thread::scope(|s| {
                for _ in 0..workers {
                    s.spawn(|| {
                        ragdoll_core::apply_current_power_mode_to_thread();
                        let mut slice_refs = Vec::with_capacity(planar.len());
                        loop {
                            let f_idx = next_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if f_idx >= num_frames {
                                break;
                            }
                            let offset = f_idx * enc.block_size;
                            let count = (total_samples - offset).min(enc.block_size);
                            slice_refs.clear();
                            for ch in &planar {
                                slice_refs.push(&ch[offset..offset + count]);
                            }
                            let res = enc.encode_frame_slices(&slice_refs, f_idx as u64);
                            if let Ok(mut g) = slots[f_idx].lock() {
                                *g = Some(res);
                            }
                        }
                    });
                }
            });

            for slot in slots {
                let packet = slot.into_inner().unwrap().unwrap()?;
                let pkt_len = packet.len() as u32;
                min_frame_size = min_frame_size.min(pkt_len);
                max_frame_size = max_frame_size.max(pkt_len);
                frames_encoded.extend_from_slice(&packet);
            }
        } else {
            let mut offset = 0usize;
            let mut frame_num = 0u64;
            let mut slice_refs = Vec::with_capacity(planar.len());
            while offset < total_samples {
                let count = (total_samples - offset).min(enc.block_size);
                slice_refs.clear();
                for ch in &planar {
                    slice_refs.push(&ch[offset..offset + count]);
                }
                let packet = enc.encode_frame_slices(&slice_refs, frame_num)?;
                let pkt_len = packet.len() as u32;
                min_frame_size = min_frame_size.min(pkt_len);
                max_frame_size = max_frame_size.max(pkt_len);
                frames_encoded.extend_from_slice(&packet);
                offset += count;
                frame_num += 1;
            }
        }

        if total_samples == 0 {
            min_frame_size = 0;
        }
        let nominal_bs = (enc.block_size as u16).max(16);

        let md5_signature = if enc.compute_md5 {
            if audio.format == ragdoll_core::SampleFormat::S16le && bps == 16 {
                let byte_len = total_samples * (audio.channels as usize) * 2;
                md5_digest(&audio.data.as_slice()[..byte_len])
            } else {
                compute_flac_pcm_md5_planar(&planar, bps)
            }
        } else {
            [0u8; 16]
        };

        let stream_info = FlacStreamInfo {
            min_block_size: nominal_bs,
            max_block_size: nominal_bs,
            min_frame_size,
            max_frame_size,
            sample_rate: enc.sample_rate,
            channels: enc.channels,
            bits_per_sample: enc.bits_per_sample,
            total_samples: total_samples as u64,
            md5_signature,
        };

        let mut out = Vec::with_capacity(42 + frames_encoded.len());
        out.extend_from_slice(FLAC_STREAM_MARKER);
        // Metadata block header: is_last=1 (0x80) | type=0 (STREAMINFO), length = 34
        out.extend_from_slice(&[0x80, 0x00, 0x00, 34]);
        out.extend_from_slice(&stream_info.to_bytes());
        out.extend_from_slice(&frames_encoded);
        Ok(out)
    }
}
