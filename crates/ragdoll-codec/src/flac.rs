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
}
