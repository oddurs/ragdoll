//! Opus (RFC 6716) and its Ogg/Matroska mapping (RFC 7845).
//!
//! - **Decoder** ([`decoder::OpusPacketDecoder`], wrapped by [`OpusDecoder`]): a port of
//!   fixed-point libopus 1.3.1 (SILK, CELT, hybrid, mode transitions, PLC, in-band FEC, DTX),
//!   sample-exact with it at every output rate (cairn `0077`). Modules: `range` (entropy coder),
//!   `celt`/`bands`/`vq`/`rate`/`energy`/`mdct`/`plc` (CELT), `silk`/`silk_dsp` (SILK and its
//!   resampler), `math` (fixed-point helpers), `*_tables` (tables generated from libopus).
//!   Being a port, it carries libopus's BSD-3-Clause notice: see `COPYING.libopus`.
//! - **Encoder** ([`OpusEncoder`], `OpusRangeEncoder`, PVQ helpers): writes packets no other
//!   decoder can read; being replaced (cairn `BUG-0078`).
//! - **`OpusHead` and `OpusTags`**: header parser and serializer for Ogg Opus and Matroska/WebM
//!   `CodecPrivate`.

#![allow(clippy::needless_range_loop)]

mod bands;
mod celt;
mod celt_tables;
pub mod decoder;
mod energy;
mod math;
mod mdct;
mod plc;
mod range;
mod rate;
mod silk;
mod silk_dsp;
mod silk_tables;
mod vq;

use ragdoll_core::{
    AudioFrame, ChannelLayout, RagdollError, Rational64, Result, SampleFormat, Timestamp,
};
use ragdoll_simd::{MdctPlan, sine_window_f32};

use crate::pcm::convert_audio_frame;

/// RFC 6716 CELT 21 Bark-scale band boundaries for a 120-bin (2.5 ms) base block;
/// multiply by `4` for a 480-bin (10 ms, `N = 960` MDCT) CELT block.
pub const CELT_EBAND_BASE: [u16; 22] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 120,
];

/// RFC 7845 Section 5.1 `OpusHead` identification header (19 bytes for `mapping_family = 0`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpusHead {
    pub version: u8,
    pub channels: u8,
    pub pre_skip: u16,
    pub input_sample_rate: u32,
    pub output_gain: i16,
    pub mapping_family: u8,
}

impl OpusHead {
    pub fn new(channels: u8, input_sample_rate: u32) -> Self {
        Self {
            version: 1,
            channels: channels.clamp(1, 2),
            pre_skip: 312,
            input_sample_rate,
            output_gain: 0,
            mapping_family: 0,
        }
    }

    /// Serializes the 19-byte `OpusHead` packet for Ogg BOS page or Matroska/WebM `CodecPrivate`.
    pub fn to_bytes(&self) -> [u8; 19] {
        let mut out = [0u8; 19];
        out[0..8].copy_from_slice(b"OpusHead");
        out[8] = self.version;
        out[9] = self.channels;
        out[10..12].copy_from_slice(&self.pre_skip.to_le_bytes());
        out[12..16].copy_from_slice(&self.input_sample_rate.to_le_bytes());
        out[16..18].copy_from_slice(&self.output_gain.to_le_bytes());
        out[18] = self.mapping_family;
        out
    }

    /// Parses an `OpusHead` identification header.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 19 || &bytes[0..8] != b"OpusHead" {
            return Err(RagdollError::InvalidData("invalid OpusHead magic".into()));
        }
        let version = bytes[8];
        let channels = bytes[9].max(1);
        let pre_skip = u16::from_le_bytes([bytes[10], bytes[11]]);
        let input_sample_rate = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        let output_gain = i16::from_le_bytes([bytes[16], bytes[17]]);
        let mapping_family = bytes[18];
        Ok(Self {
            version,
            channels,
            pre_skip,
            input_sample_rate: if input_sample_rate == 0 {
                48_000
            } else {
                input_sample_rate
            },
            output_gain,
            mapping_family,
        })
    }
}

/// Builds an RFC 7845 `OpusTags` packet with vendor string and `(KEY, VALUE)` Vorbis Comment tags.
pub fn build_opus_tags(vendor: &str, comments: &[(String, String)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(64);
    out.extend_from_slice(b"OpusTags");
    let v_bytes = vendor.as_bytes();
    out.extend_from_slice(&(v_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(v_bytes);
    out.extend_from_slice(&(comments.len() as u32).to_le_bytes());
    for (k, v) in comments {
        let kv = format!("{k}={v}");
        let kv_b = kv.as_bytes();
        out.extend_from_slice(&(kv_b.len() as u32).to_le_bytes());
        out.extend_from_slice(kv_b);
    }
    out
}

/// Parses an RFC 7845 `OpusTags` packet into `(vendor, Vec<(key, value)>)`.
pub fn parse_opus_tags(bytes: &[u8]) -> Result<(String, Vec<(String, String)>)> {
    if bytes.len() < 16 || &bytes[0..8] != b"OpusTags" {
        return Err(RagdollError::InvalidData("invalid OpusTags magic".into()));
    }
    let mut pos = 8usize;
    let v_len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    if pos + v_len + 4 > bytes.len() {
        return Err(RagdollError::BitstreamEndOfData);
    }
    let vendor = String::from_utf8_lossy(&bytes[pos..pos + v_len]).into_owned();
    pos += v_len;
    let count = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    let mut tags = Vec::with_capacity(count.min(256));
    for _ in 0..count {
        if pos + 4 > bytes.len() {
            break;
        }
        let c_len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if pos + c_len > bytes.len() {
            break;
        }
        let s = String::from_utf8_lossy(&bytes[pos..pos + c_len]);
        if let Some((k, v)) = s.split_once('=') {
            tags.push((k.to_string(), v.to_string()));
        }
        pos += c_len;
    }
    Ok((vendor, tags))
}

/// Parsed RFC 6716 Section 3.1 Opus TOC (Table of Contents) byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpusToc {
    pub config: u8,
    pub stereo: bool,
    pub frame_count_code: u8,
    /// Frame size in 48 kHz samples (`120, 240, 480, 960, 1920, 2880`).
    pub frame_samples_48k: usize,
}

impl OpusToc {
    pub fn parse(toc: u8) -> Self {
        let config = toc >> 3;
        let stereo = ((toc >> 2) & 1) != 0;
        let frame_count_code = toc & 0x03;
        let frame_samples_48k = match config {
            // SILK-only (10, 20, 40, 60 ms)
            0..=11 => match config & 3 {
                0 => 480,
                1 => 960,
                2 => 1920,
                _ => 2880,
            },
            // Hybrid (10, 20 ms)
            12..=15 => {
                if (config & 1) == 0 {
                    480
                } else {
                    960
                }
            }
            // CELT-only (2.5, 5, 10, 20 ms)
            _ => match config & 3 {
                0 => 120,
                1 => 240,
                2 => 480,
                _ => 960,
            },
        };
        Self {
            config,
            stereo,
            frame_count_code,
            frame_samples_48k,
        }
    }

    /// Encodes a TOC byte for CELT Fullband 20 ms (`config = 31`, `960` samples at 48 kHz).
    #[inline]
    pub const fn celt_fb_20ms(stereo: bool) -> u8 {
        (31 << 3) | (if stereo { 1 << 2 } else { 0 })
    }
}

// ============================================================================
// RFC 6716 Section 4.1 Range Coder (OpusRangeEncoder / OpusRangeDecoder)
// ============================================================================

const EC_SYM_BITS: u32 = 8;
const EC_CODE_BITS: u32 = 32;
const EC_SYM_MAX: u32 = (1 << EC_SYM_BITS) - 1;
const EC_CODE_SHIFT: u32 = EC_CODE_BITS - EC_SYM_BITS - 1;
const EC_CODE_TOP: u32 = 1 << (EC_CODE_BITS - 1);
const EC_CODE_BOT: u32 = EC_CODE_TOP >> EC_SYM_BITS;

/// RFC 6716 Section 4.1 Arithmetic Range Encoder (`ec_enc`).
#[derive(Clone, Debug)]
pub struct OpusRangeEncoder {
    buf: Vec<u8>,
    val: u32,
    rng: u32,
    rem: i32,
    ext: usize,
}

impl OpusRangeEncoder {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
            val: 0,
            rng: EC_CODE_TOP,
            rem: -1,
            ext: 0,
        }
    }

    #[inline]
    fn carry_out(&mut self, c: i32) {
        if c != EC_SYM_MAX as i32 {
            let carry = c >> EC_SYM_BITS;
            if self.rem >= 0 {
                self.buf.push((self.rem + carry) as u8);
            }
            while self.ext > 0 {
                self.buf.push(((EC_SYM_MAX as i32 + carry) & 0xFF) as u8);
                self.ext -= 1;
            }
            self.rem = c & (EC_SYM_MAX as i32);
        } else {
            self.ext += 1;
        }
    }

    #[inline]
    fn normalize(&mut self) {
        while self.rng <= EC_CODE_BOT {
            self.carry_out((self.val >> EC_CODE_SHIFT) as i32);
            self.val = (self.val << EC_SYM_BITS) & (EC_CODE_TOP - 1);
            self.rng <<= EC_SYM_BITS;
        }
    }

    /// Encodes a symbol with cumulative frequency interval `[fl, fh)` out of total `ft`.
    pub fn encode(&mut self, fl: u32, fh: u32, ft: u32) {
        let r = self.rng / ft;
        if fl > 0 {
            self.val += self.rng - r * (ft - fl);
            self.rng = r * (fh - fl);
        } else {
            self.rng -= r * (ft - fh);
        }
        self.normalize();
    }

    /// Encodes a uniformly distributed integer `val` in `0..ft` (`ft <= 65536`).
    pub fn encode_uint(&mut self, val: u32, ft: u32) {
        assert!(val < ft && ft > 0);
        if ft > 1 {
            self.encode(val, val + 1, ft);
        }
    }

    /// Encodes a binary symbol `bit` where $P(\text{bit} = 1) = 2^{-\text{logp}}$.
    pub fn encode_bit_logp(&mut self, bit: bool, logp: u32) {
        let r = self.rng >> logp;
        if bit {
            self.val += self.rng - r;
            self.rng = r;
        } else {
            self.rng -= r;
        }
        self.normalize();
    }

    /// Finalizes the RFC 6716 range coder stream into bytes.
    pub fn finish(mut self) -> Vec<u8> {
        let mut l = EC_CODE_BITS - (32 - self.rng.leading_zeros());
        let mut msk = (EC_CODE_TOP - 1) >> l;
        let mut end = (self.val + msk) & !msk;
        if (end | msk) >= self.val + self.rng {
            l += 1;
            msk >>= 1;
            end = (self.val + msk) & !msk;
        }
        while l > 0 {
            self.carry_out((end >> EC_CODE_SHIFT) as i32);
            end = (end << EC_SYM_BITS) & (EC_CODE_TOP - 1);
            l = l.saturating_sub(EC_SYM_BITS);
        }
        if self.rem >= 0 || self.ext > 0 {
            self.carry_out(0);
        }
        self.buf
    }
}

/// Greedily projects unit-norm band vector `x` onto the integer $L_1$ pyramid $\sum_i |y_i| = k_pulses$,
/// maximizing $\frac{x \cdot y}{\sqrt{y \cdot y}}$ per RFC 6716 Section 4.3.4 (`alg_quant`).
pub fn pvq_search_pulses(x: &[f32], k_pulses: usize, y_out: &mut [i16]) {
    let n = x.len().min(y_out.len());
    y_out[..n].fill(0);
    if n == 0 || k_pulses == 0 {
        return;
    }
    let mut rxy = 0.0f32;
    let mut ryy = 0.0f32;
    for _ in 0..k_pulses {
        let mut best_idx = 0usize;
        let mut best_score = -1.0e30f32;
        let mut best_sign = 1i16;
        for i in 0..n {
            let s = if x[i] >= 0.0 { 1.0f32 } else { -1.0f32 };
            let si = if x[i] >= 0.0 { 1i16 } else { -1i16 };
            let new_rxy = rxy + s * x[i];
            let new_ryy = ryy + 2.0 * (s * (y_out[i] as f32)) + 1.0;
            // Maximize new_rxy / sqrt(new_ryy) <-> sign(new_rxy) * new_rxy^2 / new_ryy
            let score = if new_rxy >= 0.0 {
                (new_rxy * new_rxy) / new_ryy.max(1.0)
            } else {
                -(new_rxy * new_rxy) / new_ryy.max(1.0)
            };
            if score > best_score {
                best_score = score;
                best_idx = i;
                best_sign = si;
            }
        }
        rxy += (best_sign as f32) * x[best_idx];
        ryy += 2.0 * ((best_sign as f32) * (y_out[best_idx] as f32)) + 1.0;
        y_out[best_idx] += best_sign;
    }
}

/// Normalizes integer PVQ pulse vector `y` onto the $L_2$ sphere and scales by `band_gain`.
pub fn pvq_reconstruct_band(y: &[i16], band_gain: f32, x_out: &mut [f32]) {
    let n = y.len().min(x_out.len());
    let mut sum_sq = 0.0f32;
    for i in 0..n {
        let v = y[i] as f32;
        sum_sq += v * v;
    }
    if sum_sq <= 1e-12 {
        x_out[..n].fill(0.0);
        return;
    }
    let scale = band_gain / sum_sq.sqrt();
    for i in 0..n {
        x_out[i] = (y[i] as f32) * scale;
    }
}

/// Pure-Rust Opus Encoder (`48 kHz`, 20 ms = `960` samples per frame, CELT MDCT + Range Coder + PVQ).
#[derive(Clone, Debug)]
pub struct OpusEncoder {
    pub head: OpusHead,
    pub bitrate_bps: u32,
    mdct_960: MdctPlan,
    win_960: Vec<f32>,
    history: [[f32; 480]; 2],
}

impl OpusEncoder {
    /// Creates a new `OpusEncoder` for `sample_rate` (resampled/mapped to 48 kHz Opus clock), `channels` (`1` or `2`), and `bitrate_bps`.
    pub fn new(sample_rate: u32, channels: u8, bitrate_bps: u32) -> Self {
        Self {
            head: OpusHead::new(channels, sample_rate),
            bitrate_bps: bitrate_bps.max(32_000),
            mdct_960: MdctPlan::new(960),
            win_960: sine_window_f32(960),
            history: [[0.0; 480]; 2],
        }
    }

    /// Returns the 19-byte `OpusHead` (`CodecPrivate` for `.mkv`/`.webm` and BOS packet for `.ogg`).
    pub fn extra_data(&self) -> [u8; 19] {
        self.head.to_bytes()
    }

    #[inline]
    fn pulses_for_band(band: usize, bitrate_bps: u32) -> usize {
        let base = match band {
            0..=7 => 10,
            8..=12 => 18,
            13..=16 => 24,
            17..=19 => 20,
            _ => 12,
        };
        if bitrate_bps >= 128_000 {
            base * 2
        } else {
            base
        }
    }

    /// Encodes a 20 ms (`960`-sample at 48 kHz) [`AudioFrame`] into an RFC 6716 CELT Opus packet.
    pub fn encode_frame(&mut self, frame: &AudioFrame) -> Result<Vec<u8>> {
        let f32_frame = if frame.format == SampleFormat::F32le {
            frame.clone()
        } else {
            convert_audio_frame(frame, SampleFormat::F32le)?
        };
        let ch_count = (self.head.channels as usize).clamp(1, 2);
        let src = f32_frame.as_f32_slice();
        let src_ch = (f32_frame.channels as usize).max(1);
        let in_samples = f32_frame.num_samples.min(960);

        let mut re = OpusRangeEncoder::with_capacity(256);
        // Header flags: silence = 0 (logp 15), postfilter = 0 (logp 1), transient = 0 (logp 3), intra = 1 (logp 3)
        re.encode_bit_logp(false, 15);
        re.encode_bit_logp(false, 1);
        re.encode_bit_logp(false, 3);
        re.encode_bit_logp(true, 3);

        // Two 10 ms (480-sample) MDCT sub-frames per 20 ms (960-sample) Opus frame
        for sub in 0..2 {
            let sub_base = sub * 480;
            for c in 0..ch_count {
                let mut cur_480 = [0.0f32; 480];
                for i in 0..480 {
                    let idx = sub_base + i;
                    if idx < in_samples {
                        cur_480[i] = src[idx * src_ch + c.min(src_ch - 1)];
                    }
                }
                let mut win_in = [0.0f32; 960];
                for i in 0..480 {
                    win_in[i] = self.history[c][i] * self.win_960[i];
                    win_in[480 + i] = cur_480[i] * self.win_960[480 + i];
                }
                self.history[c] = cur_480;

                let mut coeffs = [0.0f32; 480];
                self.mdct_960.mdct(&win_in, &mut coeffs);

                for b in 0..21 {
                    let start = (CELT_EBAND_BASE[b] as usize) * 4;
                    let end = (CELT_EBAND_BASE[b + 1] as usize) * 4;
                    let band_slice = &coeffs[start..end];
                    let norm = band_slice.iter().map(|&x| x * x).sum::<f32>().sqrt();
                    if norm < 1e-5 {
                        re.encode_bit_logp(false, 2); // band silent
                    } else {
                        re.encode_bit_logp(true, 2); // band active
                        // Quantize log2(norm) in [-16.0, +16.0] to 11-bit unsigned [0..2047]
                        let q_e =
                            (((norm.log2() + 16.0) * 64.0).round() as i32).clamp(0, 2047) as u32;
                        re.encode_uint(q_e, 2048);

                        let k_pulses = Self::pulses_for_band(b, self.bitrate_bps);
                        let mut pulses = vec![0i16; end - start];
                        pvq_search_pulses(band_slice, k_pulses, &mut pulses);
                        for &p in &pulses {
                            if p == 0 {
                                re.encode_bit_logp(false, 2);
                            } else {
                                re.encode_bit_logp(true, 2);
                                re.encode_bit_logp(p < 0, 1);
                                let mag = (p.unsigned_abs() as u32).min(63);
                                re.encode_uint(mag - 1, 63);
                            }
                        }
                    }
                }
            }
        }

        let range_bytes = re.finish();
        let toc = OpusToc::celt_fb_20ms(ch_count == 2);
        let mut pkt = Vec::with_capacity(1 + range_bytes.len());
        pkt.push(toc);
        pkt.extend_from_slice(&range_bytes);
        Ok(pkt)
    }

    /// Encodes an arbitrary-length `AudioFrame` into 20 ms (`960`-sample) Opus packets.
    pub fn encode_stream_frames(
        &mut self,
        frame: &AudioFrame,
    ) -> Result<Vec<(Vec<u8>, Timestamp, i64)>> {
        let f32_frame = if frame.format == SampleFormat::F32le {
            frame.clone()
        } else {
            convert_audio_frame(frame, SampleFormat::F32le)?
        };
        let ch = f32_frame.channels as usize;
        let total = f32_frame.num_samples;
        let src = f32_frame.as_f32_slice();
        let tb = Rational64::new(1, 48_000);
        let base_pts = frame.pts.rescale_to(tb).ticks;

        let mut out = Vec::new();
        let mut off = 0usize;
        while off < total {
            let take = (total - off).min(960);
            let mut sub = AudioFrame::new(48_000, f32_frame.channels, SampleFormat::F32le, 960);
            sub.as_f32_mut_slice()[..take * ch].copy_from_slice(&src[off * ch..(off + take) * ch]);
            let pkt = self.encode_frame(&sub)?;
            let pts = Timestamp::new(base_pts + (off as i64), tb);
            out.push((pkt, pts, 960));
            off += 960;
        }
        Ok(out)
    }
}

/// Opus decoder for one Ogg/Matroska stream (RFC 7845 channel mapping family 0): wraps the
/// sample-exact [`decoder::OpusPacketDecoder`] at 48 kHz and applies the `OpusHead` output gain
/// the way libopus `OPUS_SET_GAIN` does. Pre-skip is left to the caller ([`Self::pre_skip`]).
#[derive(Clone)]
pub struct OpusDecoder {
    pub head: OpusHead,
    inner: decoder::OpusPacketDecoder,
    pcm: Vec<i16>,
}

impl std::fmt::Debug for OpusDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpusDecoder")
            .field("head", &self.head)
            .finish_non_exhaustive()
    }
}

/// Largest Opus packet duration (120 ms at 48 kHz), in samples per channel.
const MAX_FRAME_48K: usize = 5760;

impl OpusDecoder {
    pub fn new(channels: u8) -> Self {
        let head = OpusHead::new(channels, 48_000);
        Self {
            inner: Self::packet_decoder(&head),
            head,
            pcm: vec![0; MAX_FRAME_48K * 2],
        }
    }

    fn packet_decoder(head: &OpusHead) -> decoder::OpusPacketDecoder {
        let mut d = decoder::OpusPacketDecoder::new(48_000, head.channels.clamp(1, 2) as usize)
            .expect("48 kHz with one or two channels is always valid");
        d.decode_gain = head.output_gain as i32;
        d
    }

    /// Configures the decoder from an `OpusHead` (Ogg BOS packet or Matroska `CodecPrivate`).
    pub fn with_extra_data(mut self, extra_data: &[u8]) -> Result<Self> {
        if extra_data.len() >= 19 && &extra_data[0..8] == b"OpusHead" {
            let head = OpusHead::parse(extra_data)?;
            if head.mapping_family != 0 || head.channels > 2 {
                return Err(RagdollError::Unsupported(format!(
                    "Opus channel mapping family {} with {} channels",
                    head.mapping_family, head.channels
                )));
            }
            self.inner = Self::packet_decoder(&head);
            self.head = head;
        }
        Ok(self)
    }

    /// Samples to drop from the start of the decoded stream (RFC 7845 section 4.2).
    pub fn pre_skip(&self) -> usize {
        self.head.pre_skip as usize
    }

    /// Final range coder state of the last packet (`OPUS_GET_FINAL_RANGE`), equal to the
    /// encoder's for a correctly decoded packet.
    pub fn final_range(&self) -> u32 {
        self.inner.final_range()
    }

    /// Decodes one packet into interleaved 16-bit samples, sample-exact with fixed-point libopus.
    /// `None` (or an empty packet) conceals a lost packet of the previous packet's duration.
    pub fn decode_i16(&mut self, pkt: Option<&[u8]>) -> Result<&[i16]> {
        let ch = self.head.channels.clamp(1, 2) as usize;
        let lost = pkt.is_none_or(|p| p.is_empty());
        let frame_size = if lost {
            (self.inner.last_packet_duration().max(0) as usize).clamp(120, MAX_FRAME_48K)
        } else {
            MAX_FRAME_48K
        };
        let n = self
            .inner
            .decode(pkt, &mut self.pcm, frame_size, false)
            .map_err(|e| RagdollError::InvalidData(format!("Opus: {e}")))?;
        Ok(&self.pcm[..n * ch])
    }

    /// Decodes one packet into a 48 kHz [`AudioFrame`] of `S16le` samples.
    pub fn decode_packet_s16(&mut self, pkt: &[u8]) -> Result<AudioFrame> {
        let ch = self.head.channels.clamp(1, 2);
        let s = self.decode_i16(Some(pkt))?;
        let mut frame = AudioFrame::new(48_000, ch, SampleFormat::S16le, s.len() / ch as usize);
        frame.layout = ChannelLayout::from_channels(ch);
        frame.as_i16_mut_slice().copy_from_slice(s);
        Ok(frame)
    }

    /// Decodes one packet into a 48 kHz [`AudioFrame`] of `F32le` samples (`s16 / 32768`, as
    /// `opus_decode_float` does in fixed-point builds).
    pub fn decode_packet(&mut self, pkt: &[u8]) -> Result<AudioFrame> {
        if pkt.is_empty() {
            return Err(RagdollError::BitstreamEndOfData);
        }
        let ch = self.head.channels.clamp(1, 2);
        let s = self.decode_i16(Some(pkt))?;
        let mut frame = AudioFrame::new(48_000, ch, SampleFormat::F32le, s.len() / ch as usize);
        frame.layout = ChannelLayout::from_channels(ch);
        for (d, &v) in frame.as_f32_mut_slice().iter_mut().zip(s) {
            *d = v as f32 * (1.0 / 32768.0);
        }
        Ok(frame)
    }
}
