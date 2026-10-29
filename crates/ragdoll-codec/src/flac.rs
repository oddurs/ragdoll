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

use crate::pcm::{audio_frame_to_planar_i32, decorrelate_mid_side_i32, planar_i32_to_audio_frame};
use ragdoll_core::{
    AlignedBuffer, AudioFrame, BitReader, BitWriter, RagdollError, Result, SampleFormat, crc8_flac,
    crc16_flac,
};
use ragdoll_simd::{
    FLAC_RICE_ESCAPED, flac_autocorr_f64, flac_eval_fixed_orders_l1, flac_lpc_residual_i32,
    flac_rice_partition_costs, flac_rice_partition_costs_scalar, flac_rice_partition_sums,
};

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

/// MSB-first bit cache over one FLAC frame body for the subframe and residual hot loops.
///
/// Keeps up to 63 look-ahead bits left-aligned in a `u64` and refills with a single unaligned
/// big-endian 8-byte load (`buf |= load >> avail; next += (63 - avail) >> 3; avail |= 56`), so a
/// Rice code decodes with one `leading_zeros` and three shifts. Reads past the end of `data` see
/// zero bits; callers check [`FrameBits::check`] at partition and subframe boundaries instead of
/// on every read. The type is `Copy` so hot loops can work on a register-resident copy.
#[derive(Clone, Copy)]
struct FrameBits<'a> {
    data: &'a [u8],
    /// Next byte of `data` to load into the cache.
    next: usize,
    /// Left-aligned bit cache; bits below `avail` are zero or the true stream bits that follow.
    buf: u64,
    /// Number of valid bits at the top of `buf` (`<= 63`).
    avail: u32,
}

/// Big-endian 8-byte load at `data[next..]`, zero-padded past the end.
#[cold]
#[inline(never)]
fn load_tail(data: &[u8], next: usize) -> u64 {
    let mut b = [0u8; 8];
    if let Some(rem) = data.get(next..) {
        b[..rem.len()].copy_from_slice(rem);
    }
    u64::from_be_bytes(b)
}

impl<'a> FrameBits<'a> {
    #[inline]
    fn new(data: &'a [u8]) -> Self {
        let mut r = Self {
            data,
            next: 0,
            buf: 0,
            avail: 0,
        };
        r.refill();
        r
    }

    /// Tops the cache up to `56..=63` valid bits.
    #[inline(always)]
    fn refill(&mut self) {
        let word = match self.data.get(self.next..self.next + 8) {
            Some(b) => u64::from_be_bytes(b.try_into().unwrap()),
            None => load_tail(self.data, self.next),
        };
        self.buf |= word >> self.avail;
        self.next += ((63 - self.avail) >> 3) as usize;
        self.avail |= 56;
    }

    /// Bits consumed from the start of `data`.
    #[inline(always)]
    fn consumed_bits(&self) -> usize {
        self.next * 8 - self.avail as usize
    }

    /// Fails with [`RagdollError::BitstreamEndOfData`] once reads have run past the end of `data`.
    #[inline(always)]
    fn check(&self) -> Result<()> {
        if self.consumed_bits() > self.data.len() * 8 {
            Err(RagdollError::BitstreamEndOfData)
        } else {
            Ok(())
        }
    }

    /// Reads `n` unsigned bits (`1 <= n <= 56`).
    #[inline(always)]
    fn bits(&mut self, n: u32) -> u64 {
        debug_assert!((1..=56).contains(&n));
        if self.avail < n {
            self.refill();
        }
        let v = self.buf >> (64 - n);
        self.buf <<= n;
        self.avail -= n;
        v
    }

    /// Reads `n` two's-complement signed bits (`1 <= n <= 56`).
    #[inline(always)]
    fn signed(&mut self, n: u32) -> i64 {
        let shift = 64 - n;
        ((self.bits(n) << shift) as i64) >> shift
    }

    #[inline(always)]
    fn bit(&mut self) -> bool {
        self.bits(1) != 0
    }

    /// Counts zero bits up to and including the terminating `1` bit.
    fn unary(&mut self) -> Result<u64> {
        let mut count = 0u64;
        loop {
            if self.avail < 56 {
                self.refill();
            }
            let lz = self.buf.leading_zeros();
            if lz < self.avail {
                self.buf <<= lz;
                self.buf <<= 1;
                self.avail -= lz + 1;
                return Ok(count + lz as u64);
            }
            count += self.avail as u64;
            self.buf = 0;
            self.avail = 0;
            self.check()?;
        }
    }

    /// Decodes one zig-zag folded Rice residual with parameter `k <= 30`, truncated to `i32`, when
    /// the whole code sits in the cache; `None` leaves the reader untouched for [`Self::rice_slow`].
    #[inline(always)]
    fn rice_fast(&mut self, k: u32) -> Option<i32> {
        if self.avail < 32 {
            self.refill();
        }
        let q = self.buf.leading_zeros();
        let len = q + 1 + k;
        if len > self.avail {
            return None;
        }
        // `t` starts with the stop bit, so `t >> (63 - k)` is `(1 << k) | low` and
        // `(q << k) | low == ((q - 1) << k) + (t >> (63 - k))`.
        let t = self.buf << q;
        self.buf = t << (k + 1);
        self.avail -= len;
        let u = ((q as u64).wrapping_sub(1) << k).wrapping_add(t >> (63 - k));
        Some(((u >> 1) as i32) ^ -((u & 1) as i32))
    }

    /// General Rice decode (unary runs of any length), matching `BitReader::read_rice_signed`.
    #[cold]
    #[inline(never)]
    fn rice_slow(&mut self, k: u32) -> Result<i32> {
        let q = self.unary()?;
        let low = if k > 0 { self.bits(k) } else { 0 };
        let u = (q << k) | low;
        Ok((((u >> 1) as i64) ^ -((u & 1) as i64)) as i32)
    }
}

/// Partition layout of a FLAC Rice or Rice2 residual (RFC 9639 Section 9.2.7).
struct Partitions {
    param_bits: u32,
    escape_code: u32,
    count: usize,
    samples: usize,
}

impl Partitions {
    /// Reads the residual coding method and partition order, validating them against the
    /// predictor order and the `block_size <= dst_len` samples available.
    fn read(
        r: &mut FrameBits<'_>,
        predictor_order: usize,
        block_size: usize,
        dst_len: usize,
    ) -> Result<Self> {
        let (param_bits, escape_code): (u32, u32) = match r.bits(2) {
            0 => (4, 0b1111),
            1 => (5, 0b11111),
            _ => {
                return Err(RagdollError::InvalidSyntax(
                    "reserved FLAC residual coding method >= 2",
                ));
            }
        };
        let partition_order = r.bits(4) as u32;
        let count = 1usize << partition_order;
        if !block_size.is_multiple_of(count) {
            return Err(RagdollError::InvalidSyntax(
                "FLAC block_size not divisible by num_partitions",
            ));
        }
        let samples = block_size >> partition_order;
        if samples < predictor_order && count > 1 {
            return Err(RagdollError::InvalidSyntax(
                "FLAC partition size smaller than predictor order",
            ));
        }
        if block_size < predictor_order || block_size > dst_len {
            return Err(RagdollError::InvalidSyntax(
                "FLAC predictor order exceeds block_size",
            ));
        }
        Ok(Self {
            param_bits,
            escape_code,
            count,
            samples,
        })
    }
}

/// Decodes a FLAC Rice or Rice2 partitioned residual block into `dst[predictor_order..block_size]`.
#[inline(always)]
fn decode_residual(
    r: &mut FrameBits<'_>,
    predictor_order: usize,
    block_size: usize,
    dst: &mut [i32],
) -> Result<()> {
    decode_predicted::<0, true>(r, &[], 0, predictor_order, block_size, dst)
}

/// Prediction `res + (sum(c[j] * hist(j)) >> shift)` with `hist(j) = s[i - 1 - j]`, in wrapping
/// `i32` (`NARROW`) or in `i64` truncated to `i32` (`libavcodec/flacdsp_lpc_template.c`).
#[inline(always)]
fn predict<const N: usize, const NARROW: bool>(
    c: &[i32; N],
    hist: impl Fn(usize) -> i32,
    shift: u32,
    res: i32,
) -> i32 {
    // Oldest tap first, so the newest sample joins the dependency chain last.
    if NARROW {
        let mut sum = 0i32;
        for j in (0..N).rev() {
            sum = sum.wrapping_add(c[j].wrapping_mul(hist(j)));
        }
        res.wrapping_add(sum >> shift)
    } else {
        let mut sum = 0i64;
        for j in (0..N).rev() {
            sum += c[j] as i64 * hist(j) as i64;
        }
        (res as i64 + (sum >> shift)) as i32
    }
}

/// Decodes the partitioned residual of an `N`-tap predicted subframe (`N <= 12`) whose `N` warm-up
/// samples are already in `dst`, restoring each sample as soon as its residual is read.
///
/// Fusing the two loops lets the bit-position dependency chain of the Rice decoder and the
/// sample-to-sample chain of the predictor overlap. Runs of `N` samples are unrolled over a ring of
/// the last `N` samples with compile-time slots, so the history stays in registers without being
/// shifted. `N == 0` decodes plain residuals for `predictor_order` warm-up samples.
#[inline(always)]
fn decode_predicted<const N: usize, const NARROW: bool>(
    r: &mut FrameBits<'_>,
    coeffs: &[i32; N],
    shift: u32,
    predictor_order: usize,
    block_size: usize,
    dst: &mut [i32],
) -> Result<()> {
    const { assert!(N <= 12) };
    let order = if N == 0 { predictor_order } else { N };
    let parts = Partitions::read(r, order, block_size, dst.len())?;
    // `h[j] = s[i - 1 - j]` between runs.
    let mut h: [i32; N] = std::array::from_fn(|j| dst[N - 1 - j]);
    let push = |h: &mut [i32; N], res: i32| -> i32 {
        if N == 0 {
            return res;
        }
        let v = predict::<N, NARROW>(coeffs, |j| h[j], shift, res);
        *h = std::array::from_fn(|j| if j == 0 { v } else { h[j - 1] });
        v
    };

    let mut idx = order;
    for p in 0..parts.count {
        let end = (p + 1) * parts.samples;
        let part = &mut dst[idx..end];
        let k = r.bits(parts.param_bits) as u32;
        if k == parts.escape_code {
            let raw_bits = r.bits(5) as u32;
            for s in part {
                let res = if raw_bits == 0 {
                    0
                } else {
                    r.signed(raw_bits) as i32
                };
                *s = push(&mut h, res);
            }
        } else {
            let mut c = *r;
            macro_rules! rice {
                () => {
                    match c.rice_fast(k) {
                        Some(v) => v,
                        None => {
                            *r = c;
                            let v = r.rice_slow(k)?;
                            c = *r;
                            v
                        }
                    }
                };
            }
            let rest = if N > 0 {
                // `ring[m]` holds the latest sample at a position `== m (mod N)` from the run start.
                let mut ring: [i32; N] = std::array::from_fn(|m| h[N - 1 - m]);
                let (runs, rest) = part.as_chunks_mut::<N>();
                for run in runs {
                    macro_rules! step {
                        ($($t:literal)*) => {$(
                            if $t < N {
                                let res = rice!();
                                let v = predict::<N, NARROW>(
                                    coeffs,
                                    |j| ring[($t + N - 1 - j) % N],
                                    shift,
                                    res,
                                );
                                ring[$t % N] = v;
                                run[$t % N] = v;
                            }
                        )*};
                    }
                    step!(0 1 2 3 4 5 6 7 8 9 10 11);
                }
                h = std::array::from_fn(|j| ring[N - 1 - j]);
                rest
            } else {
                part
            };
            for s in rest {
                let res = rice!();
                *s = push(&mut h, res);
            }
            *r = c;
        }
        r.check()?;
        idx = end;
    }
    Ok(())
}

/// Decodes the residual of a Fixed predictor subframe (`order 0..=4`) over the warm-up samples in
/// `dst` and restores it.
///
/// The fixed predictors are LPC with coefficients `[1]`, `[2, -1]`, `[3, -3, 1]`, `[4, -6, 4, -1]`
/// and no shift, so wrapping `i32` arithmetic is bit-exact with the 64-bit reference
/// [`reconstruct_fixed_subframe`]: without a shift the result modulo `2^32` does not depend on the
/// intermediate width.
#[inline(always)]
fn decode_fixed(
    r: &mut FrameBits<'_>,
    order: usize,
    block_size: usize,
    dst: &mut [i32],
) -> Result<()> {
    match order {
        0 => decode_residual(r, 0, block_size, dst),
        1 => decode_predicted::<1, true>(r, &[1], 0, 1, block_size, dst),
        2 => decode_predicted::<2, true>(r, &[2, -1], 0, 2, block_size, dst),
        3 => decode_predicted::<3, true>(r, &[3, -3, 1], 0, 3, block_size, dst),
        4 => decode_predicted::<4, true>(r, &[4, -6, 4, -1], 0, 4, block_size, dst),
        _ => Err(RagdollError::InvalidSyntax(
            "invalid FLAC fixed predictor order > 4",
        )),
    }
}

/// Recomputes `s[first..]` with 64-bit sums after the 32-bit pass saw a sample outside the
/// subframe's bit depth at `first - 1` (only possible in a non-conforming stream).
///
/// The residual of each later sample is recovered from the 32-bit pass's own output and history,
/// so the result matches [`reconstruct_lpc_subframe`] exactly.
#[cold]
#[inline(never)]
fn lpc_redo_i64(coeffs: &[i32], shift: u32, s: &mut [i32], first: usize) {
    let wrapped = s.to_vec();
    for i in first.max(coeffs.len())..s.len() {
        let mut sum32 = 0i32;
        let mut sum64 = 0i64;
        for (j, &c) in coeffs.iter().enumerate() {
            sum32 = sum32.wrapping_add(c.wrapping_mul(wrapped[i - 1 - j]));
            sum64 += c as i64 * s[i - 1 - j] as i64;
        }
        let residual = wrapped[i].wrapping_sub(sum32 >> shift);
        s[i] = (residual as i64 + (sum64 >> shift)) as i32;
    }
}

/// Decodes the residual of an FIR LPC subframe over the warm-up samples in `dst` and restores it,
/// bit-exact with the 64-bit reference [`reconstruct_lpc_subframe`].
///
/// Orders `1..=12` (everything `libFLAC` and FFmpeg emit by default) run order-specialised fused
/// loops. When `sum(|coeff|) * 2^(bps - 1) < 2^31` the sums run in 32 bits (`libavcodec/flacdec.c`
/// picks `lpc16` on a similar bound); that bound assumes every sample fits in `bps` bits, which a
/// conforming stream guarantees and a single vectorised pass afterwards verifies.
#[inline(always)]
fn decode_lpc(
    r: &mut FrameBits<'_>,
    coeffs: &[i32],
    shift: u32,
    bps: u32,
    block_size: usize,
    dst: &mut [i32],
) -> Result<()> {
    let order = coeffs.len();
    let abs_sum: u64 = coeffs.iter().map(|&c| c.unsigned_abs() as u64).sum();
    let narrow = bps <= 31 && (abs_sum << (bps - 1)) <= i32::MAX as u64;

    macro_rules! specialised {
        ($($n:literal)*) => {
            match order {
                $($n => {
                    let c: &[i32; $n] = coeffs.try_into().unwrap();
                    if narrow {
                        decode_predicted::<$n, true>(r, c, shift, $n, block_size, dst)?
                    } else {
                        decode_predicted::<$n, false>(r, c, shift, $n, block_size, dst)?
                    }
                })*
                _ => {
                    decode_residual(r, order, block_size, dst)?;
                    let s = &mut dst[..block_size];
                    for i in order..block_size {
                        let mut sum = 0i64;
                        for (j, &c) in coeffs.iter().enumerate() {
                            sum += c as i64 * s[i - 1 - j] as i64;
                        }
                        s[i] = (s[i] as i64 + (sum >> shift)) as i32;
                    }
                    return Ok(());
                }
            }
        };
    }
    specialised!(1 2 3 4 5 6 7 8 9 10 11 12);

    if narrow {
        // Every sample fits in `bps` bits iff its bits above `bps - 1` all equal its sign bit.
        let s = &mut dst[..block_size];
        let hi = bps - 1;
        let escaped = s[order..]
            .iter()
            .fold(0i32, |acc, &v| acc | ((v >> hi) ^ (v >> 31)));
        if escaped != 0 {
            let bad = order
                + s[order..]
                    .iter()
                    .position(|&v| (v >> hi) != (v >> 31))
                    .unwrap();
            lpc_redo_i64(coeffs, shift, s, bad + 1);
        }
    }
    Ok(())
}

/// Reconstructs a FLAC Fixed predictor subframe (`order 0..=4`) in-place from warm-up samples + residuals.
///
/// Uses 64-bit intermediate arithmetic (`libavcodec/flacdec.c:decode_subframe_fixed`) to prevent overflow on 24-bit/Side channels.
/// This is the scalar reference the decoder's fused Fixed path is tested against.
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
/// This is the scalar reference the decoder's fused LPC path is tested against.
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
///
/// On x86_64 the AVX2 dispatch level runs the same code compiled with BMI2 and LZCNT (which every
/// AVX2 CPU has): flag-free variable shifts and `lzcnt` shorten the Rice decoder's critical path.
fn decode_subframe(
    r: &mut FrameBits<'_>,
    block_size: usize,
    subframe_bps: u8,
    dst: &mut [i32],
) -> Result<()> {
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2()
        && std::is_x86_feature_detected!("bmi2")
        && std::is_x86_feature_detected!("lzcnt")
    {
        // Safety: the CPU supports BMI2 and LZCNT (checked just above).
        return unsafe { decode_subframe_bmi2(r, block_size, subframe_bps, dst) };
    }
    decode_subframe_impl(r, block_size, subframe_bps, dst)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi1,bmi2,lzcnt")]
unsafe fn decode_subframe_bmi2(
    r: &mut FrameBits<'_>,
    block_size: usize,
    subframe_bps: u8,
    dst: &mut [i32],
) -> Result<()> {
    decode_subframe_impl(r, block_size, subframe_bps, dst)
}

#[inline(always)]
fn decode_subframe_impl(
    r: &mut FrameBits<'_>,
    block_size: usize,
    subframe_bps: u8,
    dst: &mut [i32],
) -> Result<()> {
    let dst = &mut dst[..block_size];
    if r.bit() {
        return Err(RagdollError::InvalidSyntax(
            "FLAC subframe header zero-padding bit must be 0",
        ));
    }
    let type_code = r.bits(6) as u8;
    let has_wasted = r.bit();
    let wasted_bits = if has_wasted {
        (r.unary()? as u8).saturating_add(1)
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
            let val = r.signed(bps) as i32;
            dst.fill(val);
        }
        1 => {
            // Verbatim subframe
            for s in dst.iter_mut() {
                *s = r.signed(bps) as i32;
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
                *s = r.signed(bps) as i32;
            }
            decode_fixed(r, order, block_size, dst)?;
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
                *s = r.signed(bps) as i32;
            }
            let qlp_prec = (r.bits(4) as u32) + 1;
            if qlp_prec == 16 {
                return Err(RagdollError::InvalidSyntax(
                    "reserved FLAC LPC coefficient precision 1111",
                ));
            }
            let qlp_shift = r.signed(5) as i32;
            if qlp_shift < 0 {
                return Err(RagdollError::InvalidSyntax(
                    "negative FLAC LPC shift is invalid per RFC 9639",
                ));
            }
            let mut coeffs = [0i32; 32];
            for c in &mut coeffs[..order] {
                *c = r.signed(qlp_prec) as i32;
            }
            decode_lpc(r, &coeffs[..order], qlp_shift as u32, bps, block_size, dst)?;
        }
        _ => {
            return Err(RagdollError::InvalidSyntax(
                "reserved FLAC subframe type code",
            ));
        }
    }
    r.check()?;

    if wasted_bits > 0 {
        for s in dst.iter_mut() {
            *s = s.wrapping_shl(wasted_bits as u32);
        }
    }
    Ok(())
}

/// Undoes Left-Side, Right-Side or Mid-Side stereo decorrelation in place on
/// `channels[0][range]` and `channels[1][range]`.
///
/// Mid-Side uses `right = mid - (side >> 1); left = right + side` in wrapping `i32`, which equals the
/// RFC 9639 form `((mid << 1 | side & 1) ± side) >> 1` evaluated in 64 bits and truncated.
fn undo_stereo_decorrelation(
    mode: FlacChannelMode,
    channels: &mut [Vec<i32>],
    range: std::ops::Range<usize>,
) {
    if matches!(mode, FlacChannelMode::Independent(_)) || channels.len() < 2 {
        return;
    }
    let (first, rest) = channels.split_at_mut(1);
    let a = &mut first[0][range.clone()];
    let b = &mut rest[0][range];
    match mode {
        FlacChannelMode::LeftSide => {
            for (l, s) in a.iter().zip(b.iter_mut()) {
                *s = l.wrapping_sub(*s);
            }
        }
        FlacChannelMode::RightSide => {
            for (s, r) in a.iter_mut().zip(b.iter()) {
                *s = s.wrapping_add(*r);
            }
        }
        FlacChannelMode::MidSide => {
            for (m, s) in a.iter_mut().zip(b.iter_mut()) {
                let right = m.wrapping_sub(*s >> 1);
                *m = right.wrapping_add(*s);
                *s = right;
            }
        }
        FlacChannelMode::Independent(_) => {}
    }
}

/// Undoes stereo decorrelation (`MODE`: 0 Independent, 1 Left-Side, 2 Right-Side, 3 Mid-Side) and
/// writes interleaved `S16le` samples (`(v << shift)` clamped to `i16`, as
/// [`planar_i32_to_audio_frame`] does) in a single pass, with AVX2 / NEON kernels.
fn write_stereo_s16<const MODE: u8>(a: &[i32], b: &[i32], shift: u32, out: &mut [u8]) {
    let n = a.len().min(b.len()).min(out.len() / 4);
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        let done = unsafe { write_stereo_s16_avx2::<MODE>(&a[..n], &b[..n], shift, out) };
        return write_stereo_s16_scalar::<MODE>(
            &a[done..n],
            &b[done..n],
            shift,
            &mut out[done * 4..],
        );
    }
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: NEON is baseline on aarch64.
        let done = unsafe { write_stereo_s16_neon::<MODE>(&a[..n], &b[..n], shift, out) };
        return write_stereo_s16_scalar::<MODE>(
            &a[done..n],
            &b[done..n],
            shift,
            &mut out[done * 4..],
        );
    }
    write_stereo_s16_scalar::<MODE>(a, b, shift, out)
}

/// AVX2 body of [`write_stereo_s16`] over whole groups of 8 samples; returns the samples written.
/// `packs_epi32` saturates exactly like the scalar clamp.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn write_stereo_s16_avx2<const MODE: u8>(
    a: &[i32],
    b: &[i32],
    shift: u32,
    out: &mut [u8],
) -> usize {
    use std::arch::x86_64::*;
    let n = a.len() & !7;
    debug_assert!(b.len() >= n && out.len() >= n * 4);
    let count = _mm_cvtsi32_si128(shift as i32);
    let mut i = 0;
    while i < n {
        // Safety: `i + 8 <= n <= a.len(), b.len()` and `out` holds `4 * n` bytes.
        unsafe {
            let x = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let y = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let (l, r) = match MODE {
                1 => (x, _mm256_sub_epi32(x, y)),
                2 => (_mm256_add_epi32(x, y), y),
                3 => {
                    let right = _mm256_sub_epi32(x, _mm256_srai_epi32(y, 1));
                    (_mm256_add_epi32(right, y), right)
                }
                _ => (x, y),
            };
            let l = _mm256_sll_epi32(l, count);
            let r = _mm256_sll_epi32(r, count);
            // Per 128-bit lane: [l0..l3 l0..l3] and [r0..r3 r0..r3], interleaved to l0 r0 .. l3 r3.
            let lr = _mm256_unpacklo_epi16(_mm256_packs_epi32(l, l), _mm256_packs_epi32(r, r));
            _mm256_storeu_si256(out.as_mut_ptr().add(i * 4) as *mut __m256i, lr);
        }
        i += 8;
    }
    n
}

/// NEON body of [`write_stereo_s16`] over whole groups of 4 samples; returns the samples written.
/// `vqmovn_s32` saturates exactly like the scalar clamp.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn write_stereo_s16_neon<const MODE: u8>(
    a: &[i32],
    b: &[i32],
    shift: u32,
    out: &mut [u8],
) -> usize {
    use std::arch::aarch64::*;
    let n = a.len() & !3;
    debug_assert!(b.len() >= n && out.len() >= n * 4);
    let count = vdupq_n_s32(shift as i32);
    let mut i = 0;
    while i < n {
        // Safety: `i + 4 <= n <= a.len(), b.len()` and `out` holds `4 * n` bytes.
        unsafe {
            let x = vld1q_s32(a.as_ptr().add(i));
            let y = vld1q_s32(b.as_ptr().add(i));
            let (l, r) = match MODE {
                1 => (x, vsubq_s32(x, y)),
                2 => (vaddq_s32(x, y), y),
                3 => {
                    let right = vsubq_s32(x, vshrq_n_s32::<1>(y));
                    (vaddq_s32(right, y), right)
                }
                _ => (x, y),
            };
            let l = vqmovn_s32(vshlq_s32(l, count));
            let r = vqmovn_s32(vshlq_s32(r, count));
            vst2_s16(out.as_mut_ptr().add(i * 4) as *mut i16, int16x4x2_t(l, r));
        }
        i += 4;
    }
    n
}

/// Scalar reference for [`write_stereo_s16`].
fn write_stereo_s16_scalar<const MODE: u8>(a: &[i32], b: &[i32], shift: u32, out: &mut [u8]) {
    for ((&x, &y), o) in a.iter().zip(b).zip(out.as_chunks_mut::<4>().0) {
        let (l, r) = match MODE {
            1 => (x, x.wrapping_sub(y)),
            2 => (x.wrapping_add(y), y),
            3 => {
                let right = x.wrapping_sub(y >> 1);
                (right.wrapping_add(y), right)
            }
            _ => (x, y),
        };
        let l = (l << shift).clamp(i16::MIN as i32, i16::MAX as i32) as u16 as u32;
        let r = (r << shift).clamp(i16::MIN as i32, i16::MAX as i32) as u16 as u32;
        *o = (l | (r << 16)).to_le_bytes();
    }
}

/// Writes one decoded frame (`channels[c][..n]`, already decorrelated unless `mode` is stereo and
/// `format` is `S16le`) into its interleaved slot `out` of an [`AudioFrame`] buffer, using the same
/// sample conversion as [`planar_i32_to_audio_frame`].
fn write_interleaved(
    mode: FlacChannelMode,
    channels: &mut [Vec<i32>],
    n: usize,
    format: SampleFormat,
    bits_per_sample: u8,
    out: &mut [u8],
) {
    let ch = channels.len();
    if format == SampleFormat::S16le && ch == 2 {
        let shift = 16u32.saturating_sub(bits_per_sample as u32);
        let (a, b) = (&channels[0][..n], &channels[1][..n]);
        match mode {
            FlacChannelMode::LeftSide => write_stereo_s16::<1>(a, b, shift, out),
            FlacChannelMode::RightSide => write_stereo_s16::<2>(a, b, shift, out),
            FlacChannelMode::MidSide => write_stereo_s16::<3>(a, b, shift, out),
            FlacChannelMode::Independent(_) => write_stereo_s16::<0>(a, b, shift, out),
        }
        return;
    }
    undo_stereo_decorrelation(mode, channels, 0..n);
    let bytes = format.bytes_per_sample();
    for (c, chan) in channels.iter().enumerate() {
        let samples = chan[..n].iter();
        let slots = out[c * bytes..].chunks_mut(ch * bytes);
        match format {
            SampleFormat::U8 => {
                let shift = 8u32.saturating_sub(bits_per_sample as u32);
                for (&v, o) in samples.zip(slots) {
                    o[0] = ((v << shift).clamp(-128, 127) + 128) as u8;
                }
            }
            SampleFormat::S16le => {
                let shift = 16u32.saturating_sub(bits_per_sample as u32);
                for (&v, o) in samples.zip(slots) {
                    let s = (v << shift).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    o[..2].copy_from_slice(&s.to_le_bytes());
                }
            }
            SampleFormat::S24le => {
                let shift = 24u32.saturating_sub(bits_per_sample as u32);
                for (&v, o) in samples.zip(slots) {
                    let s = (v << shift).clamp(-8_388_608, 8_388_607);
                    o[..3].copy_from_slice(&s.to_le_bytes()[..3]);
                }
            }
            SampleFormat::S32le | SampleFormat::F32le => {
                let shift = 32u32.saturating_sub(bits_per_sample as u32);
                for (&v, o) in samples.zip(slots) {
                    o[..4].copy_from_slice(&v.wrapping_shl(shift).to_le_bytes());
                }
            }
        }
    }
}

/// Appends the RFC 9639 Section 8.2 MD5 input for one decoded frame (`channels[c][..n]`,
/// interleaved, `ceil(bps / 8)` little-endian bytes per sample) to `md5`, staging through `scratch`.
fn md5_update_planar(
    md5: &mut Md5,
    channels: &[Vec<i32>],
    n: usize,
    bits_per_sample: u8,
    scratch: &mut Vec<u8>,
) {
    let bytes = (bits_per_sample as usize).div_ceil(8);
    scratch.clear();
    for i in 0..n {
        for chan in channels {
            scratch.extend_from_slice(&chan[i].to_le_bytes()[..bytes]);
        }
    }
    md5.update(scratch);
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

    /// Decodes the subframes of one frame into `channels[c][..block_size]` (growing them as needed)
    /// and verifies the CRC-16 footer, leaving stereo decorrelation undone.
    ///
    /// Returns `(header, bytes_consumed)`.
    fn decode_frame_raw(
        &self,
        data: &[u8],
        channels: &mut [Vec<i32>],
        sample_offset: usize,
    ) -> Result<(FlacFrameHeader, usize)> {
        let header = FlacFrameHeader::parse(data, self.stream_info.as_ref())?;
        let block_size = header.block_size;

        let mut r = FrameBits::new(&data[header.header_len_bytes..]);
        let count = (header.channel_mode.channels() as usize).min(channels.len());
        for (c, ch_vec) in channels.iter_mut().enumerate().take(count) {
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

        // Byte-align with zero padding before reading the 16-bit footer CRC
        let body_bytes = r.consumed_bits().div_ceil(8);
        let footer_offset = header.header_len_bytes + body_bytes;
        if footer_offset + 2 > data.len() {
            return Err(RagdollError::BitstreamEndOfData);
        }
        let expected_crc16 = u16::from_be_bytes([data[footer_offset], data[footer_offset + 1]]);
        let actual_crc16 = crc16_flac(&data[..footer_offset]);
        if expected_crc16 != actual_crc16 {
            return Err(RagdollError::ChecksumMismatch {
                expected: expected_crc16 as u32,
                actual: actual_crc16 as u32,
            });
        }
        Ok((header, footer_offset + 2))
    }

    /// Decodes a single FLAC frame from `data` (starting at the `0xFF 0xF8`/`0xFF 0xF9` sync code),
    /// returning `(FlacFrameHeader, planar_channels, bytes_consumed)`.
    ///
    /// Verifies both the 8-bit frame header CRC (`crc8_flac`) and the 16-bit frame footer CRC (`crc16_flac`).
    pub fn decode_frame_planar(
        &self,
        data: &[u8],
    ) -> Result<(FlacFrameHeader, Vec<Vec<i32>>, usize)> {
        let mut subframes = vec![Vec::new(); 8];
        let (header, consumed) = self.decode_frame_raw(data, &mut subframes, 0)?;
        subframes.truncate(header.channel_mode.channels() as usize);
        undo_stereo_decorrelation(header.channel_mode, &mut subframes, 0..header.block_size);
        Ok((header, subframes, consumed))
    }

    /// Decodes a single FLAC frame directly into pre-allocated planar channel slices (`dst_channels[c][..block_size]`),
    /// returning `(FlacFrameHeader, bytes_consumed)` with zero heap allocations.
    pub fn decode_frame_into(
        &self,
        data: &[u8],
        dst_channels: &mut [Vec<i32>],
        sample_offset: usize,
    ) -> Result<(FlacFrameHeader, usize)> {
        let (header, consumed) = self.decode_frame_raw(data, dst_channels, sample_offset)?;
        undo_stereo_decorrelation(
            header.channel_mode,
            dst_channels,
            sample_offset..sample_offset + header.block_size,
        );
        Ok((header, consumed))
    }

    /// Decodes a single FLAC frame into an interleaved [`AudioFrame`], returning `(AudioFrame, bytes_consumed)`.
    pub fn decode_frame(&self, data: &[u8]) -> Result<(AudioFrame, usize)> {
        let (header, planar, consumed) = self.decode_frame_planar(data)?;
        let frame = planar_i32_to_audio_frame(&planar, header.sample_rate, header.bits_per_sample)?;
        Ok((frame, consumed))
    }

    /// Decodes an entire `.flac` file or stream into a single contiguous interleaved [`AudioFrame`],
    /// verifying every frame's CRC-8 and CRC-16 and, when `STREAMINFO` carries one, the MD5
    /// signature of the decoded audio ([`RagdollError::ChecksumMismatch`] on any mismatch).
    pub fn decode_stream(data: &[u8]) -> Result<(FlacStreamInfo, AudioFrame)> {
        Self::decode_stream_impl(data, true)
    }

    /// Like [`FlacDecoder::decode_stream`], but skips the `STREAMINFO` MD5 check (frame CRCs are
    /// still verified), matching FFmpeg's default FLAC decode behaviour.
    pub fn decode_stream_unverified(data: &[u8]) -> Result<(FlacStreamInfo, AudioFrame)> {
        Self::decode_stream_impl(data, false)
    }

    fn decode_stream_impl(data: &[u8], verify_md5: bool) -> Result<(FlacStreamInfo, AudioFrame)> {
        let (info, mut pos) = parse_flac_stream_header(data)?;
        let dec = Self::new(Some(info.clone()));
        let ch = info.channels as usize;
        let format = match info.bits_per_sample {
            1..=8 => SampleFormat::U8,
            9..=16 => SampleFormat::S16le,
            17..=24 => SampleFormat::S24le,
            _ => SampleFormat::S32le,
        };
        let frame_bytes = ch * format.bytes_per_sample();
        // The MD5 input is the output bytes themselves for whole-byte bit depths.
        let md5_from_output = matches!(info.bits_per_sample, 16 | 24 | 32);
        let mut md5 = (verify_md5 && info.md5_signature != [0u8; 16]).then(Md5::new);
        let mut md5_scratch = Vec::new();

        let mut cap = (info.total_samples as usize).min(data.len().saturating_mul(8));
        let mut out = AlignedBuffer::new_zeroed(cap * frame_bytes);
        // Room for the 8 channels a frame header can announce, so a mismatch is reported as such.
        let mut planar = vec![Vec::new(); 8];
        let mut sample_offset = 0usize;

        while pos < data.len() {
            // Fast check for trailing zero padding at EOF
            if data[pos] == 0 && data[pos..].iter().all(|&b| b == 0) {
                break;
            }
            let (hdr, consumed) = dec.decode_frame_raw(&data[pos..], &mut planar, 0)?;
            if hdr.channel_mode.channels() as usize != ch {
                return Err(RagdollError::InvalidSyntax(
                    "FLAC frame channel count differs from STREAMINFO",
                ));
            }
            let planar = &mut planar[..ch];
            let n = hdr.block_size;
            let mut mode = hdr.channel_mode;
            if md5.is_some() && !md5_from_output {
                undo_stereo_decorrelation(mode, planar, 0..n);
                mode = FlacChannelMode::Independent(ch as u8);
            }
            if sample_offset + n > cap {
                cap = (sample_offset + n).max(cap.saturating_mul(2));
                let mut grown = AlignedBuffer::new_zeroed(cap * frame_bytes);
                let used = sample_offset * frame_bytes;
                grown.as_mut_slice()[..used].copy_from_slice(&out.as_slice()[..used]);
                out = grown;
            }
            let dst = &mut out.as_mut_slice()
                [sample_offset * frame_bytes..(sample_offset + n) * frame_bytes];
            write_interleaved(mode, planar, n, format, info.bits_per_sample, dst);
            if let Some(md5) = md5.as_mut() {
                if md5_from_output {
                    md5.update(dst);
                } else {
                    md5_update_planar(md5, planar, n, info.bits_per_sample, &mut md5_scratch);
                }
            }
            sample_offset += n;
            pos += consumed;
        }

        if sample_offset != cap {
            out = AlignedBuffer::from_slice(&out.as_slice()[..sample_offset * frame_bytes]);
        }
        let mut audio = AudioFrame::new(info.sample_rate, info.channels, format, 0);
        audio.data = out;
        audio.num_samples = sample_offset;

        if let Some(md5) = md5 {
            let actual_md5 = md5.finalize();
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

/// Incremental RFC 1321 MD5 hasher for FLAC `STREAMINFO` audio payload signatures (RFC 9639 Section 8.2).
#[derive(Clone, Debug)]
pub struct Md5 {
    state: [u32; 4],
    block: [u8; 64],
    block_len: usize,
    total_len: u64,
}

impl Default for Md5 {
    fn default() -> Self {
        Self::new()
    }
}

impl Md5 {
    pub const fn new() -> Self {
        Self {
            state: [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476],
            block: [0u8; 64],
            block_len: 0,
            total_len: 0,
        }
    }

    /// Absorbs `data`, hashing every completed 64-byte block straight from the input slice.
    pub fn update(&mut self, mut data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);
        if self.block_len > 0 {
            let take = (64 - self.block_len).min(data.len());
            self.block[self.block_len..self.block_len + take].copy_from_slice(&data[..take]);
            self.block_len += take;
            data = &data[take..];
            if self.block_len < 64 {
                return;
            }
            let block = self.block;
            md5_compress(&mut self.state, std::slice::from_ref(&block));
            self.block_len = 0;
        }
        let (blocks, rest) = data.as_chunks::<64>();
        md5_compress(&mut self.state, blocks);
        self.block[..rest.len()].copy_from_slice(rest);
        self.block_len = rest.len();
    }

    /// Applies RFC 1321 padding and returns the 16-byte digest.
    pub fn finalize(mut self) -> [u8; 16] {
        let total_bits = self.total_len.wrapping_mul(8);
        let mut tail = [0u8; 128];
        tail[..self.block_len].copy_from_slice(&self.block[..self.block_len]);
        tail[self.block_len] = 0x80;
        let pad_len = if self.block_len < 56 { 64 } else { 128 };
        tail[pad_len - 8..pad_len].copy_from_slice(&total_bits.to_le_bytes());
        let (blocks, _) = tail[..pad_len].as_chunks::<64>();
        md5_compress(&mut self.state, blocks);

        let mut out = [0u8; 16];
        for (o, s) in out.as_chunks_mut::<4>().0.iter_mut().zip(self.state) {
            *o = s.to_le_bytes();
        }
        out
    }
}

/// RFC 1321 per-step additive constants `K[i] = floor(2^32 * |sin(i + 1)|)`.
const MD5_K: [u32; 64] = [
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
];

/// RFC 1321 message word index used by each of the 64 steps.
const MD5_G: [usize; 64] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, //
    1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, //
    5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, //
    0, 7, 14, 5, 12, 3, 10, 1, 8, 15, 6, 13, 4, 11, 2, 9,
];

/// RFC 1321 MD5 compression over whole 64-byte blocks, fully unrolled.
///
/// Each step adds `w[i] = m[g(i)] + K[i]`, precomputed for the block, and splits the round function
/// into a part that does not depend on the previous step's output `b` (added early, off the
/// critical path) and the part that does: `F = d ^ (b & (c ^ d))`, `G = (c & !d) + (b & d)`
/// (disjoint bits, so `+` equals `|`), `H = b ^ (c ^ d)`, `I = c ^ (b | !d)`. That leaves four or
/// five dependent operations per step.
fn md5_compress(state: &mut [u32; 4], blocks: &[[u8; 64]]) {
    let [mut a0, mut b0, mut c0, mut d0] = *state;
    for block in blocks {
        let words = block.as_chunks::<4>().0;
        let w: [u32; 64] =
            std::array::from_fn(|i| u32::from_le_bytes(words[MD5_G[i]]).wrapping_add(MD5_K[i]));
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);

        macro_rules! step {
            ($f:expr, $a:ident, $b:ident, $c:ident, $d:ident, $i:expr, $s:literal) => {
                let (early, late): (u32, u32) = $f($b, $c, $d);
                $a = $a
                    .wrapping_add(w[$i])
                    .wrapping_add(early)
                    .wrapping_add(late)
                    .rotate_left($s)
                    .wrapping_add($b);
            };
        }
        macro_rules! round {
            ($f:expr, $i:literal, $s0:literal, $s1:literal, $s2:literal, $s3:literal) => {
                step!($f, a, b, c, d, $i, $s0);
                step!($f, d, a, b, c, $i + 1, $s1);
                step!($f, c, d, a, b, $i + 2, $s2);
                step!($f, b, c, d, a, $i + 3, $s3);
            };
        }
        let f = |b: u32, c: u32, d: u32| (0, d ^ (b & (c ^ d)));
        let g = |b: u32, c: u32, d: u32| (c & !d, b & d);
        let h = |b: u32, c: u32, d: u32| (0, b ^ (c ^ d));
        let i = |b: u32, c: u32, d: u32| (0, c ^ (b | !d));

        round!(f, 0, 7, 12, 17, 22);
        round!(f, 4, 7, 12, 17, 22);
        round!(f, 8, 7, 12, 17, 22);
        round!(f, 12, 7, 12, 17, 22);
        round!(g, 16, 5, 9, 14, 20);
        round!(g, 20, 5, 9, 14, 20);
        round!(g, 24, 5, 9, 14, 20);
        round!(g, 28, 5, 9, 14, 20);
        round!(h, 32, 4, 11, 16, 23);
        round!(h, 36, 4, 11, 16, 23);
        round!(h, 40, 4, 11, 16, 23);
        round!(h, 44, 4, 11, 16, 23);
        round!(i, 48, 6, 10, 15, 21);
        round!(i, 52, 6, 10, 15, 21);
        round!(i, 56, 6, 10, 15, 21);
        round!(i, 60, 6, 10, 15, 21);

        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }
    *state = [a0, b0, c0, d0];
}

/// Pure-Rust RFC 1321 MD5 digest for FLAC `STREAMINFO` audio payload signatures (RFC 9639 Section 8.2).
pub fn md5_digest(data: &[u8]) -> [u8; 16] {
    let mut md5 = Md5::new();
    md5.update(data);
    md5.finalize()
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

/// Default block size: what `libFLAC -5` uses; `libavcodec` level 5 takes 4608.
const DEFAULT_BLOCK_SIZE: usize = 4096;
/// Highest LPC order the encoder searches (`libavcodec` level 5 and `libFLAC -5` stop at 8).
const MAX_LPC_ORDER: usize = 12;
/// Quantized LPC coefficient precision in bits, the most FLAC allows. Highly predictable material
/// gains several percent over 12 bits; the extra header bits are negligible at 4096 samples.
const LPC_PRECISION: u32 = 15;
/// Highest Rice partition order searched (as `libavcodec` from level 5 up).
const MAX_PARTITION_ORDER: usize = 8;
const MAX_PARTITIONS: usize = 1 << MAX_PARTITION_ORDER;
/// Fixed predictors `0..=4` as LPC taps (newest sample first) with no shift.
const FIXED_TAPS: [&[i32]; 5] = [&[], &[1], &[2, -1], &[3, -3, 1], &[4, -6, 4, -1]];
/// The fixed predictor is coded exactly when its single-partition estimate is within this factor of
/// the best LPC coding's single-partition size: fixed residuals often gain much more from
/// partitioning (speech with pauses), but rarely 30 %.
const FIXED_TRY_MARGIN: f64 = 1.3;
/// Stereo pairs whose estimate is within this factor of the best pair's are planned exactly. Only
/// the best (and any tie): 1.01 saves another 0.5 % on music for 10 % more time, and planning all
/// four channels (`libFLAC`'s exhaustive search) 1 % for 40 %.
const STEREO_TRY_MARGIN: f64 = 1.0;
/// Taper of the Tukey apodization window, `libFLAC`'s default `tukey(5e-1)`.
const TUKEY_TAPER: f64 = 0.5;

/// How a subframe predicts its samples.
#[derive(Clone, Copy, Debug)]
enum Predictor {
    Constant(i32),
    Verbatim,
    Fixed(usize),
    Lpc {
        order: usize,
        precision: u32,
        shift: u32,
        coeffs: [i32; MAX_LPC_ORDER],
    },
}

/// Partitioned Rice coding of one residual.
#[derive(Clone, Copy, Debug)]
struct RicePlan {
    porder: usize,
    /// 5-bit parameters (`PARTITIONED_EXP_GOLOMB2`) instead of 4-bit ones.
    wide: bool,
    /// Rice parameter of each partition, or `FLAC_RICE_ESCAPED`.
    params: [u8; MAX_PARTITIONS],
    /// Raw two's-complement width of each escaped partition's residuals.
    raw_bits: [u8; MAX_PARTITIONS],
}

impl Default for RicePlan {
    fn default() -> Self {
        Self {
            porder: 0,
            wide: false,
            params: [0; MAX_PARTITIONS],
            raw_bits: [0; MAX_PARTITIONS],
        }
    }
}

/// The cheapest coding found for one subframe, and its size in bits (exact but for the Rice
/// estimate of each partition, see [`flac_rice_partition_costs`]).
#[derive(Clone, Copy, Debug)]
struct SubframePlan {
    predictor: Predictor,
    wasted: u32,
    bits: usize,
    rice: RicePlan,
}

/// Estimated Rice bits for `cnt` residuals whose folded values sum to `sum`, at the best parameter
/// (see [`flac_rice_partition_costs`]).
fn rice_bits_estimate(sum: u64, cnt: usize) -> usize {
    let (mut bits, mut param) = ([0u32], [0u8]);
    let cnt = cnt.min(u16::MAX as usize) as u32;
    flac_rice_partition_costs_scalar(&[sum], &[u32::MAX], cnt, cnt, &mut bits, &mut param);
    bits[0] as usize
}

/// Partition tree of one residual: node `2^p - 1 + i` is partition `i` at partition order `p`.
struct RiceTree {
    sums: [u64; 2 * MAX_PARTITIONS],
    ors: [u32; 2 * MAX_PARTITIONS],
    bits: [u32; 2 * MAX_PARTITIONS],
    params: [u8; 2 * MAX_PARTITIONS],
}

impl Default for RiceTree {
    fn default() -> Self {
        Self {
            sums: [0; 2 * MAX_PARTITIONS],
            ors: [0; 2 * MAX_PARTITIONS],
            bits: [0; 2 * MAX_PARTITIONS],
            params: [0; 2 * MAX_PARTITIONS],
        }
    }
}

/// Finds the partition order and parameters that code `res[order..]` in the fewest bits, and
/// returns that count (the residual coding header included) with the bits at partition order 0.
#[inline(always)]
fn plan_rice(res: &[i32], order: usize, tree: &mut RiceTree) -> (usize, usize, RicePlan) {
    let n = res.len();
    let mut pmax = (n.trailing_zeros() as usize).min(MAX_PARTITION_ORDER);
    while pmax > 0 && (n >> pmax) <= order {
        pmax -= 1;
    }
    let leaves = 1usize << pmax;
    let RiceTree {
        sums,
        ors,
        bits,
        params,
    } = tree;
    flac_rice_partition_sums(
        res,
        order,
        n >> pmax,
        &mut sums[leaves - 1..2 * leaves - 1],
        &mut ors[leaves - 1..2 * leaves - 1],
    );
    for p in (0..pmax).rev() {
        let (b, c) = ((1usize << p) - 1, (2usize << p) - 1);
        for i in 0..1usize << p {
            sums[b + i] = sums[c + 2 * i] + sums[c + 2 * i + 1];
            ors[b + i] = ors[c + 2 * i] | ors[c + 2 * i + 1];
        }
    }

    // Every partition is costed once; a partition order then costs the sum of its level.
    let mut best = (usize::MAX, 0, false);
    let mut level0 = 0;
    for p in 0..=pmax {
        let (b, parts, psize) = ((1usize << p) - 1, 1usize << p, (n >> p) as u32);
        let level = b..b + parts;
        flac_rice_partition_costs(
            &sums[level.clone()],
            &ors[level.clone()],
            psize,
            psize - order as u32,
            &mut bits[level.clone()],
            &mut params[level.clone()],
        );
        let wide = params[level.clone()]
            .iter()
            .any(|&k| k != FLAC_RICE_ESCAPED && k > 14);
        let total = 2
            + 4
            + parts * if wide { 5 } else { 4 }
            + bits[level].iter().map(|&b| b as usize).sum::<usize>();
        if p == 0 {
            level0 = total;
        }
        if total < best.0 {
            best = (total, p, wide);
        }
    }

    let (total, porder, wide) = best;
    let mut plan = RicePlan {
        porder,
        wide,
        ..RicePlan::default()
    };
    let b = (1usize << porder) - 1;
    plan.params[..1 << porder].copy_from_slice(&params[b..2 * b + 1]);
    for (raw, &or) in plan.raw_bits.iter_mut().zip(&ors[b..2 * b + 1]) {
        *raw = (32 - or.leading_zeros()) as u8;
    }
    (total, level0, plan)
}

/// Writes the partitioned Rice coding of `res[order..]` described by `plan`.
#[inline(always)]
fn write_residual(w: &mut BitWriter, res: &[i32], order: usize, plan: &RicePlan) {
    let (param_bits, escape) = if plan.wide { (5, 31) } else { (4, 15) };
    w.write_bits(u64::from(plan.wide), 2);
    w.write_bits(plan.porder as u64, 4);
    let psize = res.len() >> plan.porder;
    for p in 0..1usize << plan.porder {
        let part = &res[(p * psize).max(order)..(p + 1) * psize];
        let k = plan.params[p];
        if k == FLAC_RICE_ESCAPED {
            let raw = plan.raw_bits[p] as u32;
            w.write_bits(escape, param_bits);
            w.write_bits(raw as u64, 5);
            if raw > 0 {
                for &r in part {
                    w.write_signed_bits(r as i64, raw);
                }
            }
        } else {
            w.write_bits(k as u64, param_bits);
            w.write_rice_signed_i32(part, k as u32);
        }
    }
}

/// Per-thread buffers of the subframe search, reused from frame to frame.
#[derive(Default)]
struct EncoderWorkspace {
    /// Apodization window for blocks of `window.len()` samples.
    window: Vec<f64>,
    /// Mean of the squared window, the share of the energy it keeps.
    window_power: f64,
    windowed: Vec<f64>,
    shifted: Vec<i32>,
    candidate: Vec<i32>,
    tree: Box<RiceTree>,
    side: Vec<i32>,
    mid: Vec<i32>,
    /// Winning residual (warm-up samples first) of each analysed channel.
    residuals: [Vec<i32>; 4],
    /// The block's channels, deinterleaved from 16-bit input.
    input: Vec<Vec<i32>>,
}

impl EncoderWorkspace {
    fn prepare(&mut self, n: usize) {
        if self.window.len() != n {
            self.window = apodization_window(n);
            self.window_power = self.window.iter().map(|w| w * w).sum::<f64>() / n.max(1) as f64;
        }
        for v in [
            &mut self.shifted,
            &mut self.candidate,
            &mut self.side,
            &mut self.mid,
        ] {
            v.resize(n, 0);
        }
        for v in &mut self.residuals {
            v.resize(n, 0);
        }
        self.windowed.resize(n, 0.0);
    }
}

/// Tukey window of `n` samples: flat, with raised-cosine tapers over [`TUKEY_TAPER`] of the block.
/// (It beat `libavcodec`'s Welch window on music, speech and tones alike.)
fn apodization_window(n: usize) -> Vec<f64> {
    let mut w = vec![1.0f64; n];
    let taper = (TUKEY_TAPER / 2.0 * n as f64) as usize;
    for i in 0..taper {
        let v = 0.5 - 0.5 * (std::f64::consts::PI * i as f64 / taper as f64).cos();
        w[i] = v;
        w[n - 1 - i] = v;
    }
    w
}

/// Levinson-Durbin recursion on autocorrelation `r`: `lpc[m - 1][..m]` predicts
/// `x[i] ≈ Σ lpc[m - 1][j] · x[i - 1 - j]` with error energy `err[m]`. Returns the highest order solved.
fn levinson_durbin(
    r: &[f64],
    max_order: usize,
    lpc: &mut [[f64; MAX_LPC_ORDER]; MAX_LPC_ORDER],
    err: &mut [f64; MAX_LPC_ORDER + 1],
) -> usize {
    let mut a = [0.0f64; MAX_LPC_ORDER];
    let mut e = r[0];
    err[0] = e;
    for m in 0..max_order {
        let mut acc = r[m + 1];
        for j in 0..m {
            acc -= a[j] * r[m - j];
        }
        let k = acc / e;
        if !k.is_finite() || k.abs() >= 1.0 {
            return m;
        }
        let prev = a;
        a[m] = k;
        for j in 0..m {
            a[j] = prev[j] - k * prev[m - 1 - j];
        }
        e *= 1.0 - k * k;
        lpc[m] = a;
        err[m + 1] = e;
        if e <= 0.0 {
            return m + 1;
        }
    }
    max_order
}

/// Quantizes `a` to `precision`-bit coefficients with the largest shift (`<= 15`) that fits,
/// carrying each rounding error into the next tap (`libavcodec/lpc.c:quantize_lpc_coefs`).
fn quantize_lpc(a: &[f64], precision: u32, out: &mut [i32]) -> Option<u32> {
    let qmax = (1i32 << (precision - 1)) - 1;
    let cmax = a.iter().fold(0.0f64, |m, &c| m.max(c.abs()));
    if cmax <= 0.0 || !cmax.is_finite() {
        return None;
    }
    let mut shift = 15i32;
    while shift > 0 && cmax * f64::from(1u32 << shift) > qmax as f64 {
        shift -= 1;
    }
    let scale = if cmax * f64::from(1u32 << shift) > qmax as f64 {
        qmax as f64 / cmax
    } else {
        f64::from(1u32 << shift)
    };
    let mut error = 0.0f64;
    let mut any = false;
    for (o, &c) in out.iter_mut().zip(a) {
        error += c * scale;
        let q = (error.round() as i32).clamp(-qmax, qmax);
        error -= q as f64;
        *o = q;
        any |= q != 0;
    }
    any.then_some(shift as u32)
}

/// Expected Rice bits per residual whose folded values average `mean`, at the best parameter.
fn rice_bits_per_sample(mean: f64) -> f64 {
    let t = mean * std::f64::consts::LN_2;
    let k = if t < 1.0 { 0.0 } else { t.log2().floor() };
    let cost = |k: f64| k + 1.0 + mean / k.exp2();
    cost(k).min(cost(k + 1.0))
}

/// Picks the cheapest coding of one subframe of `bps`-bit samples `raw`, leaving its residual
/// (warm-up samples first, or the samples themselves for Verbatim) in `out`.
#[inline(always)]
fn plan_subframe(
    raw: &[i32],
    bps: u32,
    ws: &mut EncoderWorkspace,
    out: &mut Vec<i32>,
    l1_hint: Option<&[u64; 5]>,
) -> SubframePlan {
    let n = raw.len();
    let first = raw[0];
    // `mag` ORs `|s|` rounded down to odd, so `bitlen(mag)` bounds every magnitude.
    let (mut or, mut diff, mut mag) = (0u32, 0u32, 0u32);
    for &s in raw {
        or |= s as u32;
        diff |= (s ^ first) as u32;
        mag |= (s ^ (s >> 31)) as u32;
    }
    let mut plan = SubframePlan {
        predictor: Predictor::Constant(first),
        wasted: 0,
        bits: 8 + bps as usize,
        rice: RicePlan::default(),
    };
    if diff == 0 {
        return plan;
    }

    let wasted = or.trailing_zeros().min(bps - 1);
    let x: &[i32] = if wasted > 0 {
        for (d, &s) in ws.shifted.iter_mut().zip(raw) {
            *d = s >> wasted;
        }
        &ws.shifted
    } else {
        raw
    };
    let ebps = bps - wasted;
    // `s ^ (s >> 31)` is `|s|` or `|s| - 1`, so every `|x|` is at most `2^bitlen(mag >> wasted)`.
    let max_abs = 1u64 << (32 - (mag >> wasted).leading_zeros());
    let head = 8 + wasted as usize;
    plan.wasted = wasted;
    plan.predictor = Predictor::Verbatim;
    plan.bits = head + n * ebps as usize;

    // Fixed predictor with the smallest L1 residual, and its single-partition cost estimate.
    let fixed = (ebps <= 26 && n > 4).then(|| {
        let l1 = match l1_hint {
            Some(l1) if wasted == 0 => *l1,
            _ => flac_eval_fixed_orders_l1(x),
        };
        let order = (0..5).min_by_key(|&o| l1[o]).unwrap_or(0);
        // Folded residuals sum to about `2 · L1 - cnt / 2` (half of them negative).
        let cnt = n - order;
        let folded = (2 * l1[order]).saturating_sub(cnt as u64 / 2);
        let est = head + order * ebps as usize + 2 + 4 + 4 + rice_bits_estimate(folded, cnt);
        (order, est, l1[order])
    });
    // Size of the best predicted coding with its residual in a single partition, to compare
    // with the fixed predictor's estimate.
    let mut single_partition = usize::MAX;

    // LPC from the windowed autocorrelation, unless a fixed predictor leaves almost nothing.
    let max_order = MAX_LPC_ORDER.min(n / 4);
    if max_order > 0 && fixed.is_none_or(|f| f.2 > n as u64) {
        for ((d, &s), &w) in ws.windowed.iter_mut().zip(x).zip(&ws.window) {
            *d = s as f64 * w;
        }
        let mut r = [0.0f64; MAX_LPC_ORDER + 1];
        flac_autocorr_f64(&ws.windowed, &mut r[..=max_order]);
        if r[0] > 0.0 {
            let mut lpc = [[0.0f64; MAX_LPC_ORDER]; MAX_LPC_ORDER];
            let mut err = [0.0f64; MAX_LPC_ORDER + 1];
            let solved = levinson_durbin(&r, max_order, &mut lpc, &mut err);
            let precision = LPC_PRECISION;
            // Only the order of smallest expected size is coded: its header, and Rice codes for
            // residuals of the energy the recursion predicts (the window keeps `window_power` of
            // the energy, and folded Laplacian residuals average `sqrt(2 · variance)`). The
            // estimate ranks orders well: also coding the runner-up bought 0.1 % for 15 % more time.
            let scale = 2.0 / (n as f64 * ws.window_power);
            let expected = |m: usize| {
                let bits = rice_bits_per_sample((err[m].max(0.0) * scale).sqrt());
                bits * (n - m) as f64 + (m * (precision + ebps) as usize) as f64
            };
            let order = (1..=solved).min_by(|&a, &b| expected(a).total_cmp(&expected(b)));
            let mut coeffs = [0i32; MAX_LPC_ORDER];
            if let Some(order) = order
                && let Some(shift) =
                    quantize_lpc(&lpc[order - 1][..order], precision, &mut coeffs[..order])
                && {
                    let c = &coeffs[..order];
                    let abs_sum: u64 = c.iter().map(|&v| v.unsigned_abs() as u64).sum();
                    if (abs_sum + 1) * max_abs < 1 << 31 {
                        flac_lpc_residual_i32(x, c, shift, &mut ws.candidate);
                        true
                    } else {
                        lpc_residual_i64(x, c, shift, &mut ws.candidate)
                    }
                }
            {
                let (rbits, rbits0, rice) = plan_rice(&ws.candidate, order, &mut ws.tree);
                let lpc_head = head + order * (ebps + precision) as usize + 4 + 5;
                single_partition = lpc_head + rbits0;
                if lpc_head + rbits < plan.bits {
                    plan.predictor = Predictor::Lpc {
                        order,
                        precision,
                        shift,
                        coeffs,
                    };
                    plan.bits = lpc_head + rbits;
                    plan.rice = rice;
                    std::mem::swap(out, &mut ws.candidate);
                }
            }
        }
    }

    // The fixed predictor, when its estimate (which ignores partitioning) comes near the best.
    if let Some((order, est, _)) = fixed
        && (est as f64) < single_partition.min(plan.bits) as f64 * FIXED_TRY_MARGIN
    {
        flac_lpc_residual_i32(x, FIXED_TAPS[order], 0, &mut ws.candidate);
        let (rbits, _, rice) = plan_rice(&ws.candidate, order, &mut ws.tree);
        let bits = head + order * ebps as usize + rbits;
        if bits < plan.bits {
            plan.predictor = Predictor::Fixed(order);
            plan.bits = bits;
            plan.rice = rice;
            std::mem::swap(out, &mut ws.candidate);
        }
    }

    if matches!(plan.predictor, Predictor::Verbatim) {
        out[..n].copy_from_slice(x);
    }
    plan
}

/// 64-bit LPC residual for coefficient sets whose sums may overflow 32 bits; `false` when a
/// residual does not fit in `i32`.
fn lpc_residual_i64(x: &[i32], coeffs: &[i32], shift: u32, res: &mut [i32]) -> bool {
    let order = coeffs.len();
    res[..order].copy_from_slice(&x[..order]);
    for i in order..x.len() {
        let mut sum = 0i64;
        for (j, &c) in coeffs.iter().enumerate() {
            sum += c as i64 * x[i - 1 - j] as i64;
        }
        let r = x[i] as i64 - (sum >> shift);
        if r != r as i32 as i64 {
            return false;
        }
        res[i] = r as i32;
    }
    true
}

/// Writes one subframe of `bps`-bit samples planned by [`plan_subframe`] with residual `res`.
#[inline(always)]
fn write_subframe(w: &mut BitWriter, plan: &SubframePlan, res: &[i32], bps: u32) {
    let ebps = bps - plan.wasted;
    let type_code = match plan.predictor {
        Predictor::Constant(_) => 0,
        Predictor::Verbatim => 1,
        Predictor::Fixed(order) => 8 + order as u64,
        Predictor::Lpc { order, .. } => 32 + order as u64 - 1,
    };
    w.write_bits(type_code, 7); // zero pad bit + 6-bit type
    write_wasted_bits_flag(w, plan.wasted as u8);
    match plan.predictor {
        Predictor::Constant(v) => w.write_signed_bits(v as i64, bps),
        Predictor::Verbatim => {
            for &s in res {
                w.write_signed_bits(s as i64, ebps);
            }
        }
        Predictor::Fixed(order) => {
            for &s in &res[..order] {
                w.write_signed_bits(s as i64, ebps);
            }
            write_residual(w, res, order, &plan.rice);
        }
        Predictor::Lpc {
            order,
            precision,
            shift,
            coeffs,
        } => {
            for &s in &res[..order] {
                w.write_signed_bits(s as i64, ebps);
            }
            w.write_bits(precision as u64 - 1, 4);
            w.write_signed_bits(shift as i64, 5);
            for &c in &coeffs[..order] {
                w.write_signed_bits(c as i64, precision);
            }
            write_residual(w, res, order, &plan.rice);
        }
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
            block_size: DEFAULT_BLOCK_SIZE,
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
        self.encode_frame_with(planar, frame_number, &mut EncoderWorkspace::default())
    }

    /// Encodes one frame. On x86_64 the AVX2 dispatch level runs the same code compiled for
    /// AVX2, BMI2 and LZCNT (which every AVX2 CPU has): the plain loops of the search vectorize
    /// 8 lanes wide and the bit twiddling gets flag-free shifts and `lzcnt`. Integer results and
    /// unfused float operations are identical either way, so both write the same stream.
    fn encode_frame_with(
        &self,
        planar: &[&[i32]],
        frame_number: u64,
        ws: &mut EncoderWorkspace,
    ) -> Result<Vec<u8>> {
        #[cfg(target_arch = "x86_64")]
        if ragdoll_core::dispatch::avx2()
            && std::is_x86_feature_detected!("bmi2")
            && std::is_x86_feature_detected!("lzcnt")
        {
            // Safety: the CPU supports AVX2, BMI2 and LZCNT (checked just above).
            return unsafe { self.encode_frame_avx2(planar, frame_number, ws) };
        }
        self.encode_frame_impl(planar, frame_number, ws)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    unsafe fn encode_frame_avx2(
        &self,
        planar: &[&[i32]],
        frame_number: u64,
        ws: &mut EncoderWorkspace,
    ) -> Result<Vec<u8>> {
        self.encode_frame_impl(planar, frame_number, ws)
    }

    #[inline(always)]
    fn encode_frame_impl(
        &self,
        planar: &[&[i32]],
        frame_number: u64,
        ws: &mut EncoderWorkspace,
    ) -> Result<Vec<u8>> {
        if planar.is_empty() || planar.len() != self.channels as usize {
            return Err(RagdollError::InvalidData(
                "FLAC encode_frame_slices channel count mismatch".into(),
            ));
        }
        let block_size = planar[0].len();
        if block_size == 0 || planar.iter().any(|c| c.len() != block_size) {
            return Err(RagdollError::InvalidData(
                "FLAC encode_frame_slices empty or ragged block".into(),
            ));
        }
        ws.prepare(block_size);
        let bps = self.bits_per_sample as u32;

        // Stereo: of left (0), right (1), side (2, `bps + 1` bits) and mid (3), rank the four
        // channel pairs by each channel's best fixed predictor (as `libavcodec` does), plan the
        // channels of the pairs within `STEREO_TRY_MARGIN` of the best exactly, and keep the
        // cheapest of those pairs.
        const PAIRS: [([usize; 2], FlacChannelMode); 4] = [
            ([0, 1], FlacChannelMode::Independent(2)),
            ([0, 2], FlacChannelMode::LeftSide),
            ([2, 1], FlacChannelMode::RightSide),
            ([3, 2], FlacChannelMode::MidSide),
        ];
        let mut plans = [None; 4];
        let mut stereo_pair = None;
        let channel_mode = if self.channels == 2 && bps < 32 {
            let (l, r) = (planar[0], planar[1]);
            let (mut side, mut mid) = (std::mem::take(&mut ws.side), std::mem::take(&mut ws.mid));
            decorrelate_mid_side_i32(l, r, &mut mid, &mut side);
            let chans = [l, r, &side[..], &mid[..]];
            let l1s = chans.map(flac_eval_fixed_orders_l1);
            // Each channel costs at most its Verbatim size (which makes side dearer on noise).
            let est: [usize; 4] = std::array::from_fn(|c| {
                let c_bps = bps as usize + usize::from(c == 2);
                (0..5)
                    .map(|o| rice_bits_estimate(2 * l1s[c][o], block_size - o) + o * c_bps)
                    .fold(block_size * c_bps, usize::min)
            });
            let pair_est = |p: [usize; 2]| est[p[0]].saturating_add(est[p[1]]);
            let best = PAIRS.iter().map(|&(p, _)| pair_est(p)).min().unwrap_or(0);
            let mut wanted = [false; 4];
            for (p, _) in PAIRS {
                if pair_est(p) as f64 <= best as f64 * STEREO_TRY_MARGIN {
                    (wanted[p[0]], wanted[p[1]]) = (true, true);
                }
            }
            let mut res = std::mem::take(&mut ws.residuals);
            for (c, x) in chans.into_iter().enumerate() {
                if wanted[c] {
                    let sub_bps = bps + u32::from(c == 2);
                    plans[c] = Some(plan_subframe(x, sub_bps, ws, &mut res[c], Some(&l1s[c])));
                }
            }
            (ws.residuals, ws.side, ws.mid) = (res, side, mid);
            let bits = |c: usize| plans[c].map_or(usize::MAX, |p: SubframePlan| p.bits);
            let (pair, mode) = *PAIRS
                .iter()
                .min_by_key(|(p, _)| bits(p[0]).saturating_add(bits(p[1])))
                .unwrap();
            stereo_pair = Some(pair);
            mode
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

        if let Some(pair) = stereo_pair {
            for c in pair {
                let plan = plans[c].as_ref().unwrap();
                write_subframe(&mut w, plan, &ws.residuals[c], bps + u32::from(c == 2));
            }
        } else {
            for &x in planar {
                let mut res = std::mem::take(&mut ws.residuals[0]);
                let plan = plan_subframe(x, bps, ws, &mut res, None);
                write_subframe(&mut w, &plan, &res, bps);
                ws.residuals[0] = res;
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
        let (stream_info, packets) = self.encode_packets(audio)?;
        let bs = self.block_size;
        let total = stream_info.total_samples as usize;
        let framed = packets
            .into_iter()
            .enumerate()
            .map(|(i, p)| (p, (total - i * bs).min(bs)))
            .collect();
        Ok((stream_info, framed))
    }

    /// Encodes every block of `audio`, on all cores when there are enough blocks, and computes
    /// `STREAMINFO` (the MD5 alongside the frames).
    fn encode_packets(&self, audio: &AudioFrame) -> Result<(FlacStreamInfo, Vec<Vec<u8>>)> {
        let ch = audio.channels as usize;
        if ch == 0 || ch > 8 {
            return Err(RagdollError::InvalidData(
                "FLAC encodes 1 to 8 channels".into(),
            ));
        }
        // 16-bit input stored as 16-bit samples is deinterleaved block by block inside the
        // workers and hashed as it is; other formats are converted up front.
        let s16 = (audio.format == SampleFormat::S16le
            && matches!(self.bits_per_sample, 0 | 9..=16))
        .then(|| &audio.as_i16_slice()[..audio.num_samples * ch]);
        let (planar, bps) = match s16 {
            Some(_) if self.bits_per_sample == 0 => (Vec::new(), 16),
            Some(_) => (Vec::new(), self.bits_per_sample),
            None => {
                let (planar, detected) = audio_frame_to_planar_i32(audio)?;
                let bps = if self.bits_per_sample != 0 {
                    self.bits_per_sample
                } else {
                    detected
                };
                (planar, bps)
            }
        };
        let enc = Self {
            block_size: self.block_size,
            sample_rate: audio.sample_rate,
            channels: audio.channels,
            bits_per_sample: bps,
            compute_md5: self.compute_md5,
        };
        let total_samples = audio.num_samples;
        let bs = enc.block_size;
        let num_frames = total_samples.div_ceil(bs);
        let encode_one = |f_idx: usize, ws: &mut EncoderWorkspace| {
            let offset = f_idx * bs;
            let count = (total_samples - offset).min(bs);
            let mut input = std::mem::take(&mut ws.input);
            let mut refs: [&[i32]; 8] = [&[]; 8];
            if let Some(pcm) = s16 {
                input.resize_with(ch, Vec::new);
                let src = &pcm[offset * ch..(offset + count) * ch];
                if let [l, r] = &mut input[..] {
                    l.resize(count, 0);
                    r.resize(count, 0);
                    for ((l, r), s) in l.iter_mut().zip(r.iter_mut()).zip(src.as_chunks::<2>().0) {
                        (*l, *r) = (s[0] as i32, s[1] as i32);
                    }
                } else {
                    for (c, dst) in input.iter_mut().enumerate() {
                        dst.clear();
                        dst.extend(src[c..].iter().step_by(ch).map(|&s| s as i32));
                    }
                }
                for (r, c) in refs.iter_mut().zip(&input) {
                    *r = c;
                }
            } else {
                for (r, c) in refs.iter_mut().zip(&planar) {
                    *r = &c[offset..offset + count];
                }
            }
            let packet = enc.encode_frame_with(&refs[..ch], f_idx as u64, ws);
            ws.input = input;
            packet
        };
        let md5 = || {
            if !enc.compute_md5 {
                [0u8; 16]
            } else if s16.is_some() {
                // FLAC hashes 9- to 16-bit samples as 16-bit little-endian words: the input bytes.
                md5_digest(&audio.data.as_slice()[..total_samples * ch * 2])
            } else {
                compute_flac_pcm_md5_planar(&planar, bps)
            }
        };

        let mut packets = Vec::with_capacity(num_frames);
        let md5_signature;
        if num_frames >= 8 {
            let workers = ragdoll_core::CpuTopology::detect()
                .optimal_threads(ragdoll_core::global_power_mode(), 0)
                .min(num_frames);
            let next_idx = std::sync::atomic::AtomicUsize::new(0);
            let slots: Vec<std::sync::Mutex<Option<Result<Vec<u8>>>>> = (0..num_frames)
                .map(|_| std::sync::Mutex::new(None))
                .collect();
            let work = || {
                ragdoll_core::apply_current_power_mode_to_thread();
                let mut ws = EncoderWorkspace::default();
                loop {
                    let f_idx = next_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if f_idx >= num_frames {
                        break;
                    }
                    let res = encode_one(f_idx, &mut ws);
                    if let Ok(mut g) = slots[f_idx].lock() {
                        *g = Some(res);
                    }
                }
            };
            md5_signature = std::thread::scope(|s| {
                for _ in 1..workers {
                    s.spawn(work);
                }
                // The calling thread hashes the PCM first, then joins the frame workers.
                let digest = md5();
                work();
                digest
            });
            for slot in slots {
                packets.push(slot.into_inner().unwrap().unwrap()?);
            }
        } else {
            let mut ws = EncoderWorkspace::default();
            for f_idx in 0..num_frames {
                packets.push(encode_one(f_idx, &mut ws)?);
            }
            md5_signature = md5();
        }

        let (min_frame_size, max_frame_size) =
            packets.iter().fold((u32::MAX, 0u32), |(lo, hi), p| {
                (lo.min(p.len() as u32), hi.max(p.len() as u32))
            });
        let nominal_bs = (bs as u16).max(16);
        let stream_info = FlacStreamInfo {
            min_block_size: nominal_bs,
            max_block_size: nominal_bs,
            min_frame_size: if packets.is_empty() {
                0
            } else {
                min_frame_size
            },
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
        let (stream_info, packets) = self.encode_packets(audio)?;
        let mut out = Vec::with_capacity(42 + packets.iter().map(Vec::len).sum::<usize>());
        out.extend_from_slice(FLAC_STREAM_MARKER);
        // Metadata block header: is_last=1 (0x80) | type=0 (STREAMINFO), length = 34
        out.extend_from_slice(&[0x80, 0x00, 0x00, 34]);
        out.extend_from_slice(&stream_info.to_bytes());
        for p in &packets {
            out.extend_from_slice(p);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ragdoll_core::BitReader;

    /// Deterministic xorshift64* stream for the differential tests.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }

        /// Uniform signed value of `bits` bits (`1..=32`).
        fn signed(&mut self, bits: u32) -> i32 {
            let shift = 64 - bits;
            (((self.next() << shift) as i64) >> shift) as i32
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
            let count =
                if p == 0 {
                    if num_partitions == 1 {
                        block_size.checked_sub(predictor_order).ok_or(
                            RagdollError::InvalidSyntax("FLAC predictor order exceeds block_size"),
                        )?
                    } else {
                        part_samples.checked_sub(predictor_order).ok_or(
                            RagdollError::InvalidSyntax(
                                "FLAC predictor order exceeds first partition",
                            ),
                        )?
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

    /// Decodes a single FLAC subframe of `block_size` samples at `subframe_bps` bits per sample.
    fn reference_decode_subframe(
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

    /// Reference stream decoder: the `BitReader` subframe path, 64-bit stereo reconstruction and
    /// [`planar_i32_to_audio_frame`], with no MD5 check.
    fn reference_decode_stream(data: &[u8]) -> AudioFrame {
        let (info, mut pos) = parse_flac_stream_header(data).unwrap();
        let mut planar = vec![Vec::new(); info.channels as usize];
        while pos < data.len() {
            if data[pos..].iter().all(|&b| b == 0) {
                break;
            }
            let frame = &data[pos..];
            let header = FlacFrameHeader::parse(frame, Some(&info)).unwrap();
            let n = header.block_size;
            let mut r = BitReader::new(&frame[header.header_len_bytes..]);
            let mut subframes = vec![vec![0i32; n]; planar.len()];
            for (c, sf) in subframes.iter_mut().enumerate() {
                let bps = header.channel_mode.subframe_bps(header.bits_per_sample, c);
                reference_decode_subframe(&mut r, n, bps, sf).unwrap();
            }
            r.align_to_byte();
            pos += header.header_len_bytes + r.bits_read() / 8 + 2;
            if planar.len() == 2 {
                let (a, b) = subframes.split_at_mut(1);
                for (x, y) in a[0].iter_mut().zip(b[0].iter_mut()) {
                    let (l, rr) = reference_stereo(header.channel_mode, *x, *y);
                    (*x, *y) = (l, rr);
                }
            }
            for (dst, src) in planar.iter_mut().zip(&subframes) {
                dst.extend_from_slice(src);
            }
        }
        planar_i32_to_audio_frame(&planar, info.sample_rate, info.bits_per_sample).unwrap()
    }

    /// RFC 9639 stereo reconstruction in 64-bit arithmetic, truncated to `i32`.
    fn reference_stereo(mode: FlacChannelMode, a: i32, b: i32) -> (i32, i32) {
        match mode {
            FlacChannelMode::LeftSide => (a, a.wrapping_sub(b)),
            FlacChannelMode::RightSide => (a.wrapping_add(b), b),
            FlacChannelMode::MidSide => {
                let (m, s) = (a as i64, b as i64);
                let m2 = (m << 1) | (s & 1);
                (((m2 + s) >> 1) as i32, ((m2 - s) >> 1) as i32)
            }
            FlacChannelMode::Independent(_) => (a, b),
        }
    }

    #[test]
    fn rice_residuals_match_bitreader_reference() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for case in 0..400 {
            let method = (case & 1) as u64;
            let max_param = if method == 0 { 14 } else { 30 };
            let escape = if method == 0 { 15 } else { 31 };
            let porder = rng.below(5) as u32;
            let block_size = [4096usize, 1152, 192, 4608, 16][rng.below(5) as usize];
            let block_size = if block_size.is_multiple_of(1 << porder) {
                block_size
            } else {
                block_size << porder
            };
            let part = block_size >> porder;
            let order = rng.below(part.min(33) as u64) as usize;

            let mut w = BitWriter::new();
            let lead = rng.below(8) as u32;
            if lead > 0 {
                w.write_bits(rng.below(1 << lead), lead);
            }
            w.write_bits(method, 2);
            w.write_bits(porder as u64, 4);
            for p in 0..(1usize << porder) {
                let count = if p == 0 { part - order } else { part };
                if rng.below(8) == 0 {
                    let raw_bits = rng.below(32) as u32;
                    w.write_bits(escape, if method == 0 { 4 } else { 5 });
                    w.write_bits(raw_bits as u64, 5);
                    for _ in 0..count {
                        if raw_bits > 0 {
                            w.write_signed_bits(rng.signed(raw_bits) as i64, raw_bits);
                        }
                    }
                    continue;
                }
                let k = rng.below(max_param as u64 + 1) as u32;
                w.write_bits(k as u64, if method == 0 { 4 } else { 5 });
                for _ in 0..count {
                    // Mostly short unary prefixes, with occasional runs well past 64 bits.
                    let q = match rng.below(64) {
                        0 => 60 + rng.below(200),
                        1..=8 => rng.below(40),
                        _ => rng.below(4),
                    };
                    let folded = (q << k) | (rng.next() & ((1u64 << k) - 1));
                    let val = ((folded >> 1) as i64) ^ -((folded & 1) as i64);
                    w.write_rice_signed(val, k);
                }
            }
            w.write_bits(0xA5A5, 16);
            w.align_to_byte();
            let bytes = w.finish();

            let mut fast = vec![0i32; block_size];
            let mut fr = FrameBits::new(&bytes);
            if lead > 0 {
                fr.bits(lead);
            }
            decode_residual(&mut fr, order, block_size, &mut fast).unwrap();

            let mut slow = vec![0i32; block_size];
            let mut br = BitReader::new(&bytes);
            br.skip_bits(lead as usize).unwrap();
            decode_residual_partitions(&mut br, order, block_size, &mut slow).unwrap();

            assert_eq!(fast, slow, "case {case}");
            assert_eq!(fr.consumed_bits(), br.bits_read(), "case {case}");
            assert_eq!(fr.bits(16), 0xA5A5, "case {case}");
        }
    }

    #[test]
    fn truncated_residual_reports_end_of_data() {
        let mut w = BitWriter::new();
        w.write_bits(0, 2);
        w.write_bits(0, 4);
        w.write_bits(3, 4);
        for _ in 0..100 {
            w.write_rice_signed(-77, 3);
        }
        let bytes = w.finish();
        let mut dst = vec![0i32; 4096];
        let mut fr = FrameBits::new(&bytes);
        assert!(decode_residual(&mut fr, 0, 4096, &mut dst).is_err());
    }

    /// Writes `res[order..]` as a Rice2 residual (random partition order, per-partition parameter
    /// sized to the partition's largest value) followed by a 16-bit marker.
    fn encode_residual(res: &[i32], order: usize, rng: &mut Rng) -> Vec<u8> {
        let n = res.len();
        let mut porder = rng.below(4) as u32;
        while porder > 0 && (!n.is_multiple_of(1 << porder) || (n >> porder) < order) {
            porder -= 1;
        }
        let part = n >> porder;
        let mut w = BitWriter::new();
        w.write_bits(1, 2);
        w.write_bits(porder as u64, 4);
        let mut idx = order;
        for p in 0..(1usize << porder) {
            let end = (p + 1) * part;
            let max_fold = res[idx..end]
                .iter()
                .map(|&v| ((v << 1) ^ (v >> 31)) as u32)
                .max()
                .unwrap_or(0);
            let k = (32 - max_fold.leading_zeros()).saturating_sub(rng.below(7) as u32);
            let k = k.min(30);
            w.write_bits(k as u64, 5);
            for &v in &res[idx..end] {
                w.write_rice_signed(v as i64, k);
            }
            idx = end;
        }
        w.write_bits(0xA5A5, 16);
        w.align_to_byte();
        w.finish()
    }

    #[test]
    fn fused_lpc_matches_reference() {
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        for case in 0..3000 {
            let order = 1 + rng.below(32) as usize;
            let bps = [4u32, 8, 12, 16, 17, 20, 24, 25, 31, 32, 33][rng.below(11) as usize];
            let prec = 1 + rng.below(15) as u32;
            let shift = rng.below(16) as u32;
            let n = [order, order + 1, order + 7, 37.max(order), 4096][rng.below(5) as usize];
            let coeffs: Vec<i32> = (0..order).map(|_| rng.signed(prec)).collect();
            // Tame cases keep residuals small; wild ones push samples past `bps` bits.
            let wild = rng.below(4) == 0;
            let input: Vec<i32> = (0..n)
                .map(|i| {
                    if i < order {
                        rng.signed(bps.min(32))
                    } else if wild {
                        rng.signed(32)
                    } else {
                        rng.signed(bps.min(8))
                    }
                })
                .collect();
            let mut expected = input.clone();
            reconstruct_lpc_subframe(&coeffs, shift as i32, &mut expected).unwrap();

            let bytes = encode_residual(&input, order, &mut rng);
            let mut got = vec![0i32; n];
            got[..order].copy_from_slice(&input[..order]);
            let mut r = FrameBits::new(&bytes);
            decode_lpc(&mut r, &coeffs, shift, bps, n, &mut got).unwrap();
            assert_eq!(
                got, expected,
                "case {case}: order {order} bps {bps} prec {prec}"
            );
            assert_eq!(r.bits(16), 0xA5A5, "case {case}");
        }
    }

    #[test]
    fn fused_lpc_recovers_from_out_of_range_history() {
        // Narrow-path coefficients whose first restored sample overflows 16 bits.
        let coeffs = [16383i32, 16383];
        let input = vec![32767i32, 32767, 32767, 5, -9, 12, 1 << 20, -3, 7];
        let mut expected = input.clone();
        reconstruct_lpc_subframe(&coeffs, 0, &mut expected).unwrap();
        let bytes = encode_residual(&input, 2, &mut Rng(1));
        let mut got = input.clone();
        decode_lpc(
            &mut FrameBits::new(&bytes),
            &coeffs,
            0,
            16,
            input.len(),
            &mut got,
        )
        .unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn fused_fixed_matches_reference() {
        let mut rng = Rng(0x0123_4567_89AB_CDEF);
        for case in 0..500 {
            let order = rng.below(5) as usize;
            let n = order + rng.below(300) as usize;
            let extreme = [i32::MIN, i32::MAX, -1, 0, 1];
            let input: Vec<i32> = (0..n)
                .map(|_| match rng.below(4) {
                    0 => extreme[rng.below(5) as usize],
                    1 => rng.signed(32),
                    _ => rng.signed(17),
                })
                .collect();
            let mut expected = input.clone();
            reconstruct_fixed_subframe(order, &mut expected).unwrap();

            let bytes = encode_residual(&input, order, &mut rng);
            let mut got = vec![0i32; n];
            got[..order].copy_from_slice(&input[..order]);
            let mut r = FrameBits::new(&bytes);
            decode_fixed(&mut r, order, n, &mut got).unwrap();
            assert_eq!(got, expected, "case {case}: order {order}");
            assert_eq!(r.bits(16), 0xA5A5, "case {case}");
        }
    }

    #[test]
    fn interleaved_writer_matches_planar_conversion() {
        let mut rng = Rng(0x5555_AAAA_1234_5678);
        let modes = [
            FlacChannelMode::Independent(2),
            FlacChannelMode::LeftSide,
            FlacChannelMode::RightSide,
            FlacChannelMode::MidSide,
        ];
        for case in 0..200 {
            let bps = 1 + rng.below(32) as u8;
            let mode = modes[rng.below(4) as usize];
            let n = 1 + rng.below(70) as usize;
            let gen_sample = |rng: &mut Rng| match rng.below(3) {
                0 => [i32::MIN, i32::MAX, -1, 0][rng.below(4) as usize],
                1 => rng.signed(32),
                _ => rng.signed(bps as u32),
            };
            let a: Vec<i32> = (0..n).map(|_| gen_sample(&mut rng)).collect();
            let b: Vec<i32> = (0..n).map(|_| gen_sample(&mut rng)).collect();

            let (l, r): (Vec<i32>, Vec<i32>) = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| reference_stereo(mode, x, y))
                .unzip();
            let expected = planar_i32_to_audio_frame(&[l, r], 44_100, bps).unwrap();

            let mut planar = vec![a.clone(), b.clone()];
            let mut out = vec![0u8; expected.data.len()];
            write_interleaved(mode, &mut planar, n, expected.format, bps, &mut out);
            assert_eq!(
                out,
                expected.data.as_slice(),
                "case {case}: bps {bps} {mode:?}"
            );
        }
    }

    #[test]
    fn stereo_s16_kernel_matches_scalar() {
        let mut rng = Rng(0x0F0F_1234_ABCD_0001);
        for len in 0..70usize {
            let extreme = [i32::MIN, i32::MAX, -1, 0, 1, 32767, -32768, 65535, -65536];
            let sample = |rng: &mut Rng| match rng.below(3) {
                0 => extreme[rng.below(extreme.len() as u64) as usize],
                1 => rng.signed(32),
                _ => rng.signed(17),
            };
            let a: Vec<i32> = (0..len).map(|_| sample(&mut rng)).collect();
            let b: Vec<i32> = (0..len).map(|_| sample(&mut rng)).collect();
            for shift in [0u32, 1, 7] {
                macro_rules! check {
                    ($($mode:literal)*) => {$(
                        let mut want = vec![0u8; len * 4];
                        let mut got = vec![0u8; len * 4];
                        write_stereo_s16_scalar::<$mode>(&a, &b, shift, &mut want);
                        write_stereo_s16::<$mode>(&a, &b, shift, &mut got);
                        assert_eq!(got, want, "len {len} shift {shift} mode {}", $mode);
                    )*};
                }
                check!(0 1 2 3);
            }
        }
    }

    #[test]
    fn md5_matches_rfc1321_vectors_incrementally() {
        let vectors: [(&[u8], &str); 7] = [
            (b"", "d41d8cd98f00b204e9800998ecf8427e"),
            (b"a", "0cc175b9c0f1b6a831c399e269772661"),
            (b"abc", "900150983cd24fb0d6963f7d28e17f72"),
            (b"message digest", "f96b697d7cb7938d525a2f31aaf161d0"),
            (
                b"abcdefghijklmnopqrstuvwxyz",
                "c3fcd3d76192e4007dfb496cca67e13b",
            ),
            (
                b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
                "d174ab98d277d9f5a5611c2c9f419d9f",
            ),
            (
                b"12345678901234567890123456789012345678901234567890123456789012345678901234567890",
                "57edf4a22be3c955ac49da2e2107b67a",
            ),
        ];
        let hex = |d: [u8; 16]| d.iter().map(|b| format!("{b:02x}")).collect::<String>();
        for (msg, want) in vectors {
            assert_eq!(hex(md5_digest(msg)), want);
            for chunk in [1usize, 3, 63, 64, 65] {
                let mut md5 = Md5::new();
                for part in msg.chunks(chunk) {
                    md5.update(part);
                }
                assert_eq!(hex(md5.finalize()), want, "chunk {chunk}");
            }
        }
        let long: Vec<u8> = (0..10_000u32).map(|i| (i * 7 + i / 13) as u8).collect();
        let mut md5 = Md5::new();
        for part in long.chunks(777) {
            md5.update(part);
        }
        assert_eq!(md5.finalize(), md5_digest(&long));
    }

    #[test]
    fn decode_stream_matches_reference_decoder() {
        let mut rng = Rng(0x7777_1111_3333_9999);
        for (channels, bps, block, n) in [
            (2u8, 16u8, 4096usize, 10_000usize),
            (2, 16, 1152, 5000),
            (1, 16, 4096, 9000),
            (2, 24, 4096, 9000),
            (2, 8, 576, 3000),
            (2, 12, 1024, 3000),
            (2, 20, 2048, 5000),
            (3, 16, 512, 2000),
        ] {
            // The encoder takes samples unshifted in the smallest container that holds `bps` bits.
            let container_bps = match bps {
                8 => 8,
                9..=16 => 16,
                _ => 24,
            };
            let mut planar = vec![vec![0i32; n]; channels as usize];
            for (c, chan) in planar.iter_mut().enumerate() {
                let mut phase = 0.0f64;
                for (i, s) in chan.iter_mut().enumerate() {
                    phase += 0.01 + 0.003 * c as f64;
                    let amp = ((1i64 << (bps - 1)) - 1) as f64 * 0.7;
                    let noise = rng.signed(bps.saturating_sub(6).max(1) as u32) as f64;
                    let v = (phase.sin() * amp + noise) as i32;
                    // A silent stretch exercises Constant subframes and wasted bits.
                    *s = if (i / 700) % 5 == 4 { 0 } else { v };
                }
            }
            let audio = planar_i32_to_audio_frame(&planar, 48_000, container_bps).unwrap();

            for md5 in [false, true] {
                let enc = FlacEncoder::new(48_000, channels, bps)
                    .with_block_size(block)
                    .with_md5(md5);
                let bytes = enc.encode_stream(&audio).unwrap();
                let (_, verified) = FlacDecoder::decode_stream(&bytes).unwrap();
                let (_, fast) = FlacDecoder::decode_stream_unverified(&bytes).unwrap();
                let reference = reference_decode_stream(&bytes);
                assert_eq!(verified.data.as_slice(), reference.data.as_slice());
                assert_eq!(fast.data.as_slice(), reference.data.as_slice());
                assert_eq!(fast.num_samples, n);

                if md5 {
                    let mut corrupted = bytes.clone();
                    corrupted[8 + 33] ^= 0x01;
                    assert!(FlacDecoder::decode_stream(&corrupted).is_err());
                    assert!(FlacDecoder::decode_stream_unverified(&corrupted).is_ok());
                }
            }
        }
    }

    #[test]
    fn corrupted_streams_fail_cleanly() {
        let mut rng = Rng(0xBAD_5EED);
        let mut audio = AudioFrame::new(44_100, 2, SampleFormat::S16le, 6000);
        for (i, s) in audio.as_i16_mut_slice().iter_mut().enumerate() {
            *s = ((i as f64 * 0.013).sin() * 12_000.0) as i16 ^ (i as i16 & 7);
        }
        let bytes = FlacEncoder::new(44_100, 2, 16)
            .with_block_size(1024)
            .with_md5(true)
            .encode_stream(&audio)
            .unwrap();
        for _ in 0..400 {
            let mut bad = bytes.clone();
            for _ in 0..1 + rng.below(4) {
                let at = 42 + rng.below((bad.len() - 42) as u64) as usize;
                bad[at] ^= 1 << rng.below(8);
            }
            if rng.below(4) == 0 {
                bad.truncate(42 + rng.below((bad.len() - 42) as u64) as usize);
            }
            // Errors are expected; panics (including debug overflow checks) are not.
            let _ = FlacDecoder::decode_stream(&bad);
            let _ = FlacDecoder::decode_stream_unverified(&bad);
        }
    }

    #[test]
    fn decode_stream_grows_when_total_samples_is_unknown() {
        let mut audio = AudioFrame::new(44_100, 2, SampleFormat::S16le, 20_000);
        for (i, s) in audio.as_i16_mut_slice().iter_mut().enumerate() {
            *s = ((i as i32 * 37) % 20_000 - 10_000) as i16;
        }
        let mut bytes = FlacEncoder::new(44_100, 2, 16)
            .with_block_size(1024)
            .encode_stream(&audio)
            .unwrap();
        // Zero the 36-bit STREAMINFO total_samples field (bytes 21..26 of the file, low nibble first).
        bytes[21] &= 0xF0;
        bytes[22..26].fill(0);
        let (_, dec) = FlacDecoder::decode_stream(&bytes).unwrap();
        assert_eq!(dec.num_samples, 20_000);
        assert_eq!(dec.data.len(), 80_000);
        assert_eq!(dec.as_i16_slice(), audio.as_i16_slice());
    }

    #[test]
    fn frame_apis_match_stream_decode() {
        let mut audio = AudioFrame::new(44_100, 2, SampleFormat::S16le, 3000);
        for (i, s) in audio.as_i16_mut_slice().iter_mut().enumerate() {
            *s = (((i * 7919) % 65_536) as i32 - 32_768) as i16 / 3;
        }
        let bytes = FlacEncoder::new(44_100, 2, 16)
            .with_block_size(1024)
            .encode_stream(&audio)
            .unwrap();
        let (info, mut pos) = parse_flac_stream_header(&bytes).unwrap();
        let dec = FlacDecoder::new(Some(info));
        let mut into = vec![Vec::new(), Vec::new()];
        let mut offset = 0;
        let mut planar_all = vec![Vec::new(), Vec::new()];
        while pos < bytes.len() {
            let (hdr, planar, used) = dec.decode_frame_planar(&bytes[pos..]).unwrap();
            let (hdr2, used2) = dec
                .decode_frame_into(&bytes[pos..], &mut into, offset)
                .unwrap();
            assert_eq!((hdr.block_size, used), (hdr2.block_size, used2));
            for (all, p) in planar_all.iter_mut().zip(planar) {
                all.extend(p);
            }
            offset += hdr.block_size;
            pos += used;
        }
        assert_eq!(into, planar_all);
        let pcm: Vec<i16> = (0..3000)
            .flat_map(|i| [into[0][i] as i16, into[1][i] as i16])
            .collect();
        assert_eq!(pcm, audio.as_i16_slice());
    }

    /// 44.1 kHz 16-bit stereo test signals: `(name, interleaved samples)`.
    fn encoder_test_signals(n: usize) -> Vec<(&'static str, Vec<i16>)> {
        use std::f64::consts::TAU;
        let mut rng = Rng(0x0F1A_C5EE_D000_0091);
        let sr = 44_100.0;
        let mut out = Vec::new();

        // Music: notes of a few harmonics each with plucked envelopes, mixed differently per
        // channel, over a little noise.
        let notes = [220.0, 277.18, 329.63, 440.0, 554.37, 164.81];
        let mut music = Vec::with_capacity(2 * n);
        for i in 0..n {
            let t = i as f64 / sr;
            let (mut l, mut r) = (0.0, 0.0);
            for (j, &f) in notes.iter().enumerate() {
                let start = j as f64 * 0.31;
                let age = (t - start).rem_euclid(1.7);
                let env = (-3.0 * age).exp();
                let v: f64 = (1..5)
                    .map(|h| (TAU * f * h as f64 * t).sin() / (h * h) as f64)
                    .sum::<f64>()
                    * env;
                l += v * (0.3 + 0.1 * j as f64);
                r += v * (0.8 - 0.1 * j as f64);
            }
            let noise = rng.signed(3) as f64;
            music.push((l * 5000.0 + noise) as i16);
            music.push((r * 5000.0 + noise * 0.5) as i16);
        }
        out.push(("music", music));

        // Speech: a glottal pulse train of drifting pitch through three formant resonators,
        // in syllables separated by digital silence; the right channel is delayed and quieter.
        let mut speech_mono = vec![0.0f64; n];
        let formants = [(700.0, 0.97), (1200.0, 0.96), (2600.0, 0.94)];
        let mut state = [[0.0f64; 2]; 3];
        let mut phase = 0.0f64;
        for (i, s) in speech_mono.iter_mut().enumerate() {
            let t = i as f64 / sr;
            let f0 = 110.0 + 25.0 * (TAU * 0.9 * t).sin();
            phase = (phase + f0 / sr).fract();
            let voiced = (TAU * 3.7 * t).sin() * 0.5 + 0.5;
            let gate = (TAU * 0.4 * t).sin() > -0.6;
            let excitation = (2.0 * phase - 1.0).powi(3) * voiced + rng.signed(4) as f64 * 0.01;
            let mut y = 0.0;
            for ((f, rad), st) in formants.iter().zip(state.iter_mut()) {
                let c = 2.0 * rad * (TAU * f / sr).cos();
                let v = excitation + c * st[0] - rad * rad * st[1];
                (st[1], st[0]) = (st[0], v);
                y += v;
            }
            *s = if gate { y * 400.0 } else { 0.0 };
        }
        let speech = (0..n)
            .flat_map(|i| {
                let l = speech_mono[i].clamp(-32768.0, 32767.0) as i16;
                let r = (speech_mono[i.saturating_sub(3)] * 0.8).clamp(-32768.0, 32767.0) as i16;
                [l, r]
            })
            .collect();
        out.push(("speech", speech));

        // Tones: steady and decaying sinusoids at a low level.
        let tonal = (0..n)
            .flat_map(|i| {
                let t = i as f64 / sr;
                let l = 0.15 * (TAU * 220.0 * t).sin() * (-(t % 2.0)).exp()
                    + 0.1 * (TAU * 329.63 * t).sin() * (-((t + 1.0) % 2.0)).exp()
                    + 0.05 * (TAU * 440.0 * t + 2.0 * (TAU * 5.0 * t).sin()).sin();
                let r = 0.15 * (TAU * 220.0 * t + 0.3).sin() * (-(t % 2.0)).exp()
                    + 0.04 * (TAU * 880.0 * t).sin();
                [(l * 32767.0) as i16, (r * 32767.0) as i16]
            })
            .collect();
        out.push(("tonal", tonal));

        // Full-scale white noise: nothing to predict, Verbatim or escaped partitions.
        out.push(("noise", (0..2 * n).map(|_| rng.signed(16) as i16).collect()));
        out
    }

    /// The encoder's streams decode bit-exactly (MD5 checked) in Ragdoll's decoder and FFmpeg's,
    /// and come out smaller than FFmpeg's `-compression_level 5` on music, speech and tones.
    #[test]
    fn encoder_round_trips_and_beats_ffmpeg_level_5() {
        use std::process::Command;
        let n = 2 * 44_100;
        let dir = std::env::temp_dir();
        for (name, pcm) in encoder_test_signals(n) {
            let mut audio = AudioFrame::new(44_100, 2, SampleFormat::S16le, n);
            audio.as_i16_mut_slice().copy_from_slice(&pcm);
            let ours = FlacEncoder::new(44_100, 2, 16)
                .with_md5(true)
                .encode_stream(&audio)
                .unwrap();
            let (_, decoded) = FlacDecoder::decode_stream(&ours).unwrap();
            assert_eq!(decoded.as_i16_slice(), &pcm[..], "{name}: Ragdoll decode");

            let tag = format!("ragdoll_flacenc_{}_{name}", std::process::id());
            let (raw, ours_path, ff_path, dec_path) = (
                dir.join(format!("{tag}.raw")),
                dir.join(format!("{tag}_ours.flac")),
                dir.join(format!("{tag}_ff.flac")),
                dir.join(format!("{tag}_dec.raw")),
            );
            std::fs::write(&raw, audio.data.as_slice()).unwrap();
            std::fs::write(&ours_path, &ours).unwrap();
            let raw_in = ["-f", "s16le", "-ar", "44100", "-ac", "2", "-i"];
            let encoded = Command::new("ffmpeg")
                .args(["-y", "-v", "error"])
                .args(raw_in)
                .arg(&raw)
                .args(["-c:a", "flac", "-compression_level", "5"])
                .arg(&ff_path)
                .status();
            if ragdoll_core::testing::tool_succeeded(encoded, "ffmpeg FLAC encode") {
                let status = Command::new("ffmpeg")
                    .args(["-y", "-v", "error", "-i"])
                    .arg(&ours_path)
                    .args(["-f", "s16le"])
                    .arg(&dec_path)
                    .status()
                    .unwrap();
                assert!(status.success(), "{name}: FFmpeg decode");
                assert_eq!(
                    std::fs::read(&dec_path).unwrap(),
                    audio.data.as_slice(),
                    "{name}: FFmpeg decode"
                );
                // Compare the audio frames, without FFmpeg's extra metadata blocks.
                let ff = std::fs::read(&ff_path).unwrap();
                let ff_frames = ff.len() - parse_flac_stream_header(&ff).unwrap().1;
                let our_frames = ours.len() - parse_flac_stream_header(&ours).unwrap().1;
                let ratio = our_frames as f64 / ff_frames as f64;
                eprintln!("{name}: {our_frames} bytes vs FFmpeg level 5 {ff_frames} ({ratio:.4})");
                assert!(ratio < 1.0, "{name}: {our_frames} vs {ff_frames} bytes");
            }
            for p in [raw, ours_path, ff_path, dec_path] {
                let _ = std::fs::remove_file(p);
            }
        }
    }
}
