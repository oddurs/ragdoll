//! Opus (RFC 6716) and its Ogg/Matroska mapping (RFC 7845).
//!
//! - **Decoder** ([`decoder::OpusPacketDecoder`], wrapped by [`OpusDecoder`]): a port of
//!   fixed-point libopus 1.3.1 (SILK, CELT, hybrid, mode transitions, PLC, in-band FEC, DTX),
//!   sample-exact with it at every output rate (cairn `0077`). Modules: `range` (entropy coder),
//!   `celt`/`bands`/`vq`/`rate`/`energy`/`mdct`/`plc` (CELT), `silk`/`silk_dsp` (SILK and its
//!   resampler), `math` (fixed-point helpers), `*_tables` (tables generated from libopus).
//!   Being a port, it carries libopus's BSD-3-Clause notice: see `COPYING.libopus`.
//! - **Encoder** ([`encoder::OpusPacketEncoder`], wrapped by [`OpusEncoder`]): a port of the
//!   fixed-point libopus 1.3.1 CELT encoder (`celt_enc`) and the CELT-only paths of its Opus
//!   layer, byte-identical with libopus in the restricted low-delay application at complexity
//!   0 to 9 (cairn `BUG-0078`).
//! - **`OpusHead` and `OpusTags`**: header parser and serializer for Ogg Opus and Matroska/WebM
//!   `CodecPrivate`.

#![allow(clippy::needless_range_loop)]

mod bands;
mod celt;
mod celt_enc;
mod celt_tables;
pub mod decoder;
pub mod encoder;
mod energy;
mod math;
mod mdct;
mod plc;
mod range;
mod rate;
mod silk;
mod silk_dsp;
mod silk_tables;
mod simd;
mod simd_enc;
mod vq;

use ragdoll_core::{
    AudioFrame, ChannelLayout, RagdollError, Rational64, Result, SampleFormat, Timestamp,
};

use crate::pcm::convert_audio_frame;

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
}

/// Opus encoder producing one Ogg/Matroska stream: 20 ms CELT packets at 48 kHz, byte-identical
/// with fixed-point libopus 1.3.1 in the restricted low-delay application at complexity 9
/// (constrained VBR by default). See [`encoder::OpusPacketEncoder`] for the full control set.
#[derive(Clone)]
pub struct OpusEncoder {
    pub head: OpusHead,
    pub bitrate_bps: u32,
    inner: encoder::OpusPacketEncoder,
    packet: Vec<u8>,
}

impl std::fmt::Debug for OpusEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpusEncoder")
            .field("head", &self.head)
            .field("bitrate_bps", &self.bitrate_bps)
            .finish_non_exhaustive()
    }
}

/// Samples per channel of the 20 ms frames the encoder codes.
const ENC_FRAME: usize = 960;

impl OpusEncoder {
    /// Creates an encoder for 48 kHz input with `channels` (1 or 2) at `bitrate_bps`;
    /// `sample_rate` is recorded in the `OpusHead` as the original input rate.
    pub fn new(sample_rate: u32, channels: u8, bitrate_bps: u32) -> Self {
        let channels = channels.clamp(1, 2);
        let mut inner = encoder::OpusPacketEncoder::new(channels as usize)
            .expect("one or two channels are always valid");
        inner.set_bitrate(bitrate_bps.min(i32::MAX as u32) as i32);
        let mut head = OpusHead::new(channels, sample_rate);
        head.pre_skip = inner.lookahead() as u16;
        Self {
            head,
            bitrate_bps,
            inner,
            packet: vec![0; 1500],
        }
    }

    /// Returns the 19-byte `OpusHead` (`CodecPrivate` for `.mkv`/`.webm` and BOS packet for `.ogg`).
    pub fn extra_data(&self) -> [u8; 19] {
        self.head.to_bytes()
    }

    /// The underlying packet encoder, for VBR, complexity and other controls.
    pub fn packet_encoder(&mut self) -> &mut encoder::OpusPacketEncoder {
        &mut self.inner
    }

    /// Final range coder state of the last packet (`OPUS_GET_FINAL_RANGE`); a decoder that
    /// reads the packet correctly ends in the same state.
    pub fn final_range(&self) -> u32 {
        self.inner.final_range()
    }

    /// Encodes 20 ms (960 samples per channel at 48 kHz) of interleaved 16-bit samples.
    pub fn encode_i16(&mut self, pcm: &[i16]) -> Result<Vec<u8>> {
        let n = self
            .inner
            .encode(pcm, ENC_FRAME, &mut self.packet)
            .map_err(|e| RagdollError::InvalidData(format!("Opus encode: {e}")))?;
        Ok(self.packet[..n].to_vec())
    }

    /// Encodes a 20 ms (960-sample at 48 kHz) [`AudioFrame`] into one Opus packet. Shorter frames
    /// are padded with silence; floating-point samples are converted as `opus_encode_float` does.
    pub fn encode_frame(&mut self, frame: &AudioFrame) -> Result<Vec<u8>> {
        let ch = self.head.channels as usize;
        let pcm = frame_to_i16(frame, ch, ENC_FRAME)?;
        self.encode_i16(&pcm)
    }

    /// Encodes a whole [`AudioFrame`] into 20 ms packets, adding enough silence at the end that
    /// every input sample comes out of a decoder after the `OpusHead` pre-skip.
    pub fn encode_stream_frames(
        &mut self,
        frame: &AudioFrame,
    ) -> Result<Vec<(Vec<u8>, Timestamp, i64)>> {
        let ch = self.head.channels as usize;
        let total = frame.num_samples;
        let padded = (total + self.head.pre_skip as usize).div_ceil(ENC_FRAME) * ENC_FRAME;
        let pcm = frame_to_i16(frame, ch, padded)?;
        let tb = Rational64::new(1, 48_000);
        let base_pts = frame.pts.rescale_to(tb).ticks;
        let mut out = Vec::with_capacity(padded / ENC_FRAME);
        for (k, chunk) in pcm.chunks_exact(ENC_FRAME * ch).enumerate() {
            let pkt = self.encode_i16(chunk)?;
            let pts = Timestamp::new(base_pts + (k * ENC_FRAME) as i64, tb);
            out.push((pkt, pts, ENC_FRAME as i64));
        }
        Ok(out)
    }
}

/// Interleaved 16-bit samples of `frame` mapped to `ch` channels and zero-padded to `len`
/// samples per channel. Float input is scaled by 32768, clamped and rounded to nearest-even,
/// like libopus's `FLOAT2INT16`.
fn frame_to_i16(frame: &AudioFrame, ch: usize, len: usize) -> Result<Vec<i16>> {
    let src_ch = (frame.channels as usize).max(1);
    let n = frame.num_samples.min(len);
    let mut out = vec![0i16; len * ch];
    let get: Box<dyn Fn(usize) -> i16> = match frame.format {
        SampleFormat::S16le => {
            let s = frame.as_i16_slice();
            Box::new(move |i| s[i])
        }
        SampleFormat::F32le => {
            let s = frame.as_f32_slice();
            Box::new(move |i| (s[i] * 32768.0).clamp(-32768.0, 32767.0).round_ties_even() as i16)
        }
        _ => {
            let f = convert_audio_frame(frame, SampleFormat::F32le)?;
            let v: Vec<i16> = f
                .as_f32_slice()
                .iter()
                .map(|&x| (x * 32768.0).clamp(-32768.0, 32767.0).round_ties_even() as i16)
                .collect();
            Box::new(move |i| v[i])
        }
    };
    for i in 0..n {
        for c in 0..ch {
            out[i * ch + c] = get(i * src_ch + c.min(src_ch - 1));
        }
    }
    Ok(out)
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
