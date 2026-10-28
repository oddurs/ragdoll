//! Opus decoder (RFC 6716 section 4, libopus `opus_decoder.c`, fixed point): packet parsing,
//! SILK/CELT/hybrid frame decoding, mode transitions with redundant CELT frames, and loss
//! concealment.

use super::celt::CeltDecoder;
use super::celt_tables::WINDOW;
use super::math::{mac16, mul16, mul16_q15, sat16};
use super::range::Ec;
use super::silk::{DecControl, SilkDecoder};

pub const MODE_SILK_ONLY: i32 = 1000;
pub const MODE_HYBRID: i32 = 1001;
pub const MODE_CELT_ONLY: i32 = 1002;

pub const BANDWIDTH_NARROWBAND: i32 = 1101;
pub const BANDWIDTH_MEDIUMBAND: i32 = 1102;
pub const BANDWIDTH_WIDEBAND: i32 = 1103;
pub const BANDWIDTH_SUPERWIDEBAND: i32 = 1104;
pub const BANDWIDTH_FULLBAND: i32 = 1105;

/// Errors of [`OpusPacketDecoder::decode`], as libopus reports them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusError {
    BadArg,
    BufferTooSmall,
    InternalError,
    InvalidPacket,
}

impl std::fmt::Display for OpusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::BadArg => "invalid argument",
            Self::BufferTooSmall => "buffer too small",
            Self::InternalError => "internal error",
            Self::InvalidPacket => "corrupted stream",
        })
    }
}

fn packet_mode(toc: u8) -> i32 {
    if toc & 0x80 != 0 {
        MODE_CELT_ONLY
    } else if toc & 0x60 == 0x60 {
        MODE_HYBRID
    } else {
        MODE_SILK_ONLY
    }
}

/// `opus_packet_get_bandwidth`.
pub fn packet_bandwidth(toc: u8) -> i32 {
    if toc & 0x80 != 0 {
        let b = BANDWIDTH_MEDIUMBAND + ((toc >> 5) & 0x3) as i32;
        if b == BANDWIDTH_MEDIUMBAND {
            BANDWIDTH_NARROWBAND
        } else {
            b
        }
    } else if toc & 0x60 == 0x60 {
        if toc & 0x10 != 0 {
            BANDWIDTH_FULLBAND
        } else {
            BANDWIDTH_SUPERWIDEBAND
        }
    } else {
        BANDWIDTH_NARROWBAND + ((toc >> 5) & 0x3) as i32
    }
}

/// `opus_packet_get_samples_per_frame`.
pub fn packet_samples_per_frame(toc: u8, fs: i32) -> i32 {
    if toc & 0x80 != 0 {
        let a = ((toc >> 3) & 0x3) as i32;
        (fs << a) / 400
    } else if toc & 0x60 == 0x60 {
        if toc & 0x08 != 0 { fs / 50 } else { fs / 100 }
    } else {
        let a = ((toc >> 3) & 0x3) as i32;
        if a == 3 {
            fs * 60 / 1000
        } else {
            (fs << a) / 100
        }
    }
}

fn parse_size(data: &[u8]) -> Option<(usize, usize)> {
    match data {
        [] => None,
        [b0, ..] if *b0 < 252 => Some((*b0 as usize, 1)),
        [_] => None,
        [b0, b1, ..] => Some((4 * *b1 as usize + *b0 as usize, 2)),
    }
}

/// A parsed packet: TOC byte, payload offset and frame sizes (`opus_packet_parse`).
pub struct ParsedPacket {
    pub toc: u8,
    pub frames: Vec<(usize, usize)>,
}

/// `opus_packet_parse_impl` without self-delimiting framing.
pub fn parse_packet(data: &[u8]) -> Result<ParsedPacket, OpusError> {
    if data.is_empty() {
        return Err(OpusError::InvalidPacket);
    }
    let framesize = packet_samples_per_frame(data[0], 48000);
    let toc = data[0];
    let mut pos = 1usize;
    let mut len = data.len() as i32 - 1;
    let mut last_size = len;
    let mut sizes: Vec<i32> = Vec::new();
    let count;
    match toc & 0x3 {
        0 => count = 1,
        1 => {
            count = 2;
            if len & 1 != 0 {
                return Err(OpusError::InvalidPacket);
            }
            last_size = len / 2;
            sizes.push(last_size);
        }
        2 => {
            count = 2;
            let (s, bytes) = parse_size(&data[pos..]).ok_or(OpusError::InvalidPacket)?;
            len -= bytes as i32;
            if s as i32 > len {
                return Err(OpusError::InvalidPacket);
            }
            pos += bytes;
            sizes.push(s as i32);
            last_size = len - s as i32;
        }
        _ => {
            if len < 1 {
                return Err(OpusError::InvalidPacket);
            }
            let ch = data[pos];
            pos += 1;
            count = (ch & 0x3f) as usize;
            if count == 0 || framesize * count as i32 > 5760 {
                return Err(OpusError::InvalidPacket);
            }
            len -= 1;
            if ch & 0x40 != 0 {
                loop {
                    if len <= 0 {
                        return Err(OpusError::InvalidPacket);
                    }
                    let p = data[pos];
                    pos += 1;
                    len -= 1;
                    let tmp = if p == 255 { 254 } else { p as i32 };
                    len -= tmp;
                    if p != 255 {
                        break;
                    }
                }
            }
            if len < 0 {
                return Err(OpusError::InvalidPacket);
            }
            let cbr = ch & 0x80 == 0;
            if !cbr {
                last_size = len;
                for _ in 0..count - 1 {
                    let (s, bytes) = parse_size(&data[pos..]).ok_or(OpusError::InvalidPacket)?;
                    len -= bytes as i32;
                    if s as i32 > len {
                        return Err(OpusError::InvalidPacket);
                    }
                    pos += bytes;
                    last_size -= bytes as i32 + s as i32;
                    sizes.push(s as i32);
                }
                if last_size < 0 {
                    return Err(OpusError::InvalidPacket);
                }
            } else {
                last_size = len / count as i32;
                if last_size * count as i32 != len {
                    return Err(OpusError::InvalidPacket);
                }
                for _ in 0..count - 1 {
                    sizes.push(last_size);
                }
            }
        }
    }
    if last_size > 1275 {
        return Err(OpusError::InvalidPacket);
    }
    sizes.push(last_size);
    let mut frames = Vec::with_capacity(count);
    for &s in &sizes {
        frames.push((pos, s as usize));
        pos += s as usize;
    }
    if pos > data.len() {
        return Err(OpusError::InvalidPacket);
    }
    Ok(ParsedPacket { toc, frames })
}

/// One Opus decoder (`OpusDecoder`): 16-bit output at 48, 24, 16, 12 or 8 kHz, mono or stereo.
#[derive(Clone)]
pub struct OpusPacketDecoder {
    pub channels: usize,
    pub fs: i32,
    celt: CeltDecoder,
    silk: SilkDecoder,
    dec_control: DecControl,
    pub decode_gain: i32,
    stream_channels: usize,
    bandwidth: i32,
    mode: i32,
    prev_mode: i32,
    frame_size: i32,
    prev_redundancy: bool,
    last_packet_duration: i32,
    range_final: u32,
}

/// `smooth_fade`: cross-fade from `in1` to `in2` with the squared CELT window.
#[allow(clippy::too_many_arguments)]
fn smooth_fade(
    in1: &[i16],
    in2: &[i16],
    out: &mut [i16],
    overlap: usize,
    channels: usize,
    fs: i32,
) {
    let inc = (48000 / fs) as usize;
    for c in 0..channels {
        for i in 0..overlap {
            let w = mul16_q15(WINDOW[i * inc] as i32, WINDOW[i * inc] as i32);
            let k = i * channels + c;
            out[k] = (mac16(mul16(w, in2[k] as i32), 32767 - w, in1[k] as i32) >> 15) as i16;
        }
    }
}

impl OpusPacketDecoder {
    /// `opus_decoder_create`.
    pub fn new(fs: i32, channels: usize) -> Result<Self, OpusError> {
        if !matches!(fs, 48000 | 24000 | 16000 | 12000 | 8000) || !(1..=2).contains(&channels) {
            return Err(OpusError::BadArg);
        }
        let dec_control = DecControl {
            n_channels_api: channels as i32,
            n_channels_internal: 0,
            api_sample_rate: fs,
            internal_sample_rate: 0,
            payload_size_ms: 0,
            prev_pitch_lag: 0,
        };
        Ok(Self {
            channels,
            fs,
            celt: CeltDecoder::new(fs as u32, channels),
            silk: SilkDecoder::new(),
            dec_control,
            decode_gain: 0,
            stream_channels: channels,
            bandwidth: 0,
            mode: 0,
            prev_mode: 0,
            frame_size: fs / 400,
            prev_redundancy: false,
            last_packet_duration: 0,
            range_final: 0,
        })
    }

    /// `OPUS_GET_FINAL_RANGE`: final range coder state of the last packet.
    pub fn final_range(&self) -> u32 {
        self.range_final
    }

    /// `OPUS_GET_LAST_PACKET_DURATION`.
    pub fn last_packet_duration(&self) -> i32 {
        self.last_packet_duration
    }

    /// `opus_decode`: decodes `packet` (or conceals a loss for `None`) into interleaved `pcm`,
    /// returning samples per channel. `frame_size` is the capacity of `pcm` per channel.
    pub fn decode(
        &mut self,
        packet: Option<&[u8]>,
        pcm: &mut [i16],
        frame_size: usize,
        decode_fec: bool,
    ) -> Result<usize, OpusError> {
        let frame_size = frame_size as i32;
        if frame_size <= 0 || pcm.len() < frame_size as usize * self.channels {
            return Err(OpusError::BadArg);
        }
        let data = packet.filter(|d| !d.is_empty());
        if (decode_fec || data.is_none()) && frame_size % (self.fs / 400) != 0 {
            return Err(OpusError::BadArg);
        }
        let Some(data) = data else {
            let mut pcm_count = 0i32;
            while pcm_count < frame_size {
                let o = pcm_count as usize * self.channels;
                let ret = self.decode_frame(None, &mut pcm[o..], frame_size - pcm_count, false)?;
                pcm_count += ret;
            }
            self.last_packet_duration = pcm_count;
            return Ok(pcm_count as usize);
        };
        let toc = data[0];
        let packet_mode = packet_mode(toc);
        let packet_bandwidth = packet_bandwidth(toc);
        let packet_frame_size = packet_samples_per_frame(toc, self.fs);
        let packet_stream_channels = if toc & 0x4 != 0 { 2 } else { 1 };
        let parsed = parse_packet(data)?;
        if decode_fec {
            if frame_size < packet_frame_size
                || packet_mode == MODE_CELT_ONLY
                || self.mode == MODE_CELT_ONLY
            {
                return self.decode(None, pcm, frame_size as usize, false);
            }
            let duration_copy = self.last_packet_duration;
            if frame_size - packet_frame_size != 0
                && let Err(e) =
                    self.decode(None, pcm, (frame_size - packet_frame_size) as usize, false)
            {
                self.last_packet_duration = duration_copy;
                return Err(e);
            }
            self.mode = packet_mode;
            self.bandwidth = packet_bandwidth;
            self.frame_size = packet_frame_size;
            self.stream_channels = packet_stream_channels;
            let (off, size) = parsed.frames[0];
            let o = (frame_size - packet_frame_size) as usize * self.channels;
            self.decode_frame(
                Some(&data[off..off + size]),
                &mut pcm[o..],
                packet_frame_size,
                true,
            )?;
            self.last_packet_duration = frame_size;
            return Ok(frame_size as usize);
        }
        if parsed.frames.len() as i32 * packet_frame_size > frame_size {
            return Err(OpusError::BufferTooSmall);
        }
        self.mode = packet_mode;
        self.bandwidth = packet_bandwidth;
        self.frame_size = packet_frame_size;
        self.stream_channels = packet_stream_channels;
        let mut nb_samples = 0i32;
        for &(off, size) in &parsed.frames {
            let o = nb_samples as usize * self.channels;
            let ret = self.decode_frame(
                Some(&data[off..off + size]),
                &mut pcm[o..],
                frame_size - nb_samples,
                false,
            )?;
            nb_samples += ret;
        }
        self.last_packet_duration = nb_samples;
        Ok(nb_samples as usize)
    }

    /// `opus_decode_frame`.
    fn decode_frame(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i16],
        frame_size: i32,
        decode_fec: bool,
    ) -> Result<i32, OpusError> {
        let ch = self.channels;
        let f20 = self.fs / 50;
        let f10 = f20 >> 1;
        let f5 = f10 >> 1;
        let f2_5 = f5 >> 1;
        if frame_size < f2_5 {
            return Err(OpusError::BufferTooSmall);
        }
        let mut frame_size = frame_size.min(self.fs / 25 * 3);
        let data = match data {
            Some(d) if d.len() > 1 => Some(d),
            _ => {
                frame_size = frame_size.min(self.frame_size);
                None
            }
        };
        let mut len = data.map_or(0, |d| d.len() as i32);
        let mut audiosize;
        let mode;
        let bandwidth;
        let mut dec_storage: Option<Ec> = None;
        if let Some(d) = data {
            audiosize = self.frame_size;
            mode = self.mode;
            bandwidth = self.bandwidth;
            dec_storage = Some(Ec::decoder(d));
        } else {
            audiosize = frame_size;
            mode = self.prev_mode;
            bandwidth = 0;
            if mode == 0 {
                pcm[..audiosize as usize * ch].fill(0);
                return Ok(audiosize);
            }
            if audiosize > f20 {
                let mut o = 0usize;
                loop {
                    let ret = self.decode_frame(None, &mut pcm[o..], audiosize.min(f20), false)?;
                    o += ret as usize * ch;
                    audiosize -= ret;
                    if audiosize <= 0 {
                        break;
                    }
                }
                return Ok(frame_size);
            } else if audiosize < f20 {
                if audiosize > f10 {
                    audiosize = f10;
                } else if mode != MODE_SILK_ONLY && audiosize > f5 && audiosize < f10 {
                    audiosize = f5;
                }
            }
        }
        let celt_accum = mode != MODE_CELT_ONLY && frame_size >= f10;
        let mut transition = false;
        if data.is_some()
            && self.prev_mode > 0
            && ((mode == MODE_CELT_ONLY
                && self.prev_mode != MODE_CELT_ONLY
                && !self.prev_redundancy)
                || (mode != MODE_CELT_ONLY && self.prev_mode == MODE_CELT_ONLY))
        {
            transition = true;
        }
        let mut pcm_transition: Vec<i16> = Vec::new();
        if transition && mode == MODE_CELT_ONLY {
            pcm_transition = vec![0i16; f5 as usize * ch];
            self.decode_frame(None, &mut pcm_transition, f5.min(audiosize), false)?;
        }
        if audiosize > frame_size {
            return Err(OpusError::BadArg);
        }
        frame_size = audiosize;
        let n = frame_size as usize * ch;
        let mut pcm_silk: Vec<i16> = if mode != MODE_CELT_ONLY && !celt_accum {
            vec![0i16; f10.max(frame_size) as usize * ch]
        } else {
            Vec::new()
        };
        if mode != MODE_CELT_ONLY {
            if self.prev_mode == MODE_CELT_ONLY {
                self.silk.reset();
            }
            self.dec_control.payload_size_ms = 10.max(1000 * audiosize / self.fs);
            if data.is_some() {
                self.dec_control.n_channels_internal = self.stream_channels as i32;
                self.dec_control.internal_sample_rate = if mode == MODE_SILK_ONLY {
                    match bandwidth {
                        BANDWIDTH_NARROWBAND => 8000,
                        BANDWIDTH_MEDIUMBAND => 12000,
                        _ => 16000,
                    }
                } else {
                    16000
                };
            }
            let lost_flag = if data.is_none() {
                1
            } else {
                2 * decode_fec as i32
            };
            let mut decoded = 0i32;
            let mut dummy = Ec::decoder(&[]);
            loop {
                let first_frame = decoded == 0;
                let target: &mut [i16] = if celt_accum {
                    &mut pcm[decoded as usize * ch..]
                } else {
                    &mut pcm_silk[decoded as usize * ch..]
                };
                let dec = dec_storage.as_mut().unwrap_or(&mut dummy);
                let r =
                    self.silk
                        .decode(&mut self.dec_control, lost_flag, first_frame, dec, target);
                let silk_frame_size = match r {
                    Ok(s) => s as i32,
                    Err(_) => {
                        if lost_flag != 0 {
                            target[..frame_size as usize * ch].fill(0);
                            frame_size
                        } else {
                            return Err(OpusError::InternalError);
                        }
                    }
                };
                decoded += silk_frame_size;
                if decoded >= frame_size {
                    break;
                }
            }
        }
        let mut start_band = 0;
        let mut redundancy = false;
        let mut redundancy_bytes = 0i32;
        let mut celt_to_silk = false;
        if !decode_fec
            && mode != MODE_CELT_ONLY
            && let Some(dec) = dec_storage.as_mut()
            && dec.tell() + 17 + 20 * (self.mode == MODE_HYBRID) as i32 <= 8 * len
        {
            redundancy = if mode == MODE_HYBRID {
                dec.dec_bit_logp(12)
            } else {
                true
            };
            if redundancy {
                celt_to_silk = dec.dec_bit_logp(1);
                redundancy_bytes = if mode == MODE_HYBRID {
                    dec.dec_uint(256) as i32 + 2
                } else {
                    len - ((dec.tell() + 7) >> 3)
                };
                len -= redundancy_bytes;
                if len * 8 < dec.tell() {
                    len = 0;
                    redundancy_bytes = 0;
                    redundancy = false;
                }
                dec.storage = (dec.storage as i32 - redundancy_bytes) as u32;
            }
        }
        if mode != MODE_CELT_ONLY {
            start_band = 17;
        }
        if redundancy {
            transition = false;
        }
        if transition && mode != MODE_CELT_ONLY {
            pcm_transition = vec![0i16; f5 as usize * ch];
            self.decode_frame(None, &mut pcm_transition, f5.min(audiosize), false)?;
        }
        if bandwidth != 0 {
            self.celt.end = match bandwidth {
                BANDWIDTH_NARROWBAND => 13,
                BANDWIDTH_MEDIUMBAND | BANDWIDTH_WIDEBAND => 17,
                BANDWIDTH_SUPERWIDEBAND => 19,
                _ => 21,
            };
        }
        self.celt.stream_channels = self.stream_channels;
        let mut redundant_audio: Vec<i16> = if redundancy {
            vec![0i16; f5 as usize * ch]
        } else {
            Vec::new()
        };
        let mut redundant_rng = 0u32;
        let redundant_data = data.map(|d| {
            let s = len.max(0) as usize;
            &d[s.min(d.len())..(s + redundancy_bytes.max(0) as usize).min(d.len())]
        });
        if redundancy && celt_to_silk {
            self.celt.start = 0;
            let _ = self.celt.decode(
                redundant_data,
                &mut redundant_audio,
                f5 as usize,
                None,
                false,
            );
            redundant_rng = self.celt.rng;
        }
        self.celt.start = start_band;
        let mut celt_ret: Result<usize, super::celt::CeltError> = Ok(0);
        if mode != MODE_SILK_ONLY {
            let celt_frame_size = f20.min(frame_size);
            if mode != self.prev_mode && self.prev_mode > 0 && !self.prev_redundancy {
                self.celt.reset();
            }
            let payload = if decode_fec {
                None
            } else {
                data.map(|d| &d[..len.max(0) as usize])
            };
            celt_ret = self.celt.decode(
                payload,
                pcm,
                celt_frame_size as usize,
                dec_storage.as_mut(),
                celt_accum,
            );
        } else {
            let silence = [0xFFu8, 0xFF];
            if !celt_accum {
                pcm[..n].fill(0);
            }
            if self.prev_mode == MODE_HYBRID
                && !(redundancy && celt_to_silk && self.prev_redundancy)
            {
                self.celt.start = 0;
                let _ = self
                    .celt
                    .decode(Some(&silence), pcm, f2_5 as usize, None, celt_accum);
            }
        }
        if mode != MODE_CELT_ONLY && !celt_accum {
            for i in 0..n {
                pcm[i] = sat16(pcm[i] as i32 + pcm_silk[i] as i32);
            }
        }
        if redundancy && !celt_to_silk {
            self.celt.reset();
            self.celt.start = 0;
            let _ = self.celt.decode(
                redundant_data,
                &mut redundant_audio,
                f5 as usize,
                None,
                false,
            );
            redundant_rng = self.celt.rng;
            let o = ch * (frame_size - f2_5) as usize;
            let tail: Vec<i16> = pcm[o..o + f2_5 as usize * ch].to_vec();
            smooth_fade(
                &tail,
                &redundant_audio[ch * f2_5 as usize..],
                &mut pcm[o..],
                f2_5 as usize,
                ch,
                self.fs,
            );
        }
        if redundancy && celt_to_silk {
            for c in 0..ch {
                for i in 0..f2_5 as usize {
                    pcm[ch * i + c] = redundant_audio[ch * i + c];
                }
            }
            let o = ch * f2_5 as usize;
            let cur: Vec<i16> = pcm[o..o + f2_5 as usize * ch].to_vec();
            smooth_fade(
                &redundant_audio[o..],
                &cur,
                &mut pcm[o..],
                f2_5 as usize,
                ch,
                self.fs,
            );
        }
        if transition {
            if audiosize >= f5 {
                let k = ch * f2_5 as usize;
                pcm[..k].copy_from_slice(&pcm_transition[..k]);
                let cur: Vec<i16> = pcm[k..2 * k].to_vec();
                smooth_fade(
                    &pcm_transition[k..],
                    &cur,
                    &mut pcm[k..],
                    f2_5 as usize,
                    ch,
                    self.fs,
                );
            } else {
                let cur: Vec<i16> = pcm[..ch * f2_5 as usize].to_vec();
                smooth_fade(&pcm_transition, &cur, pcm, f2_5 as usize, ch, self.fs);
            }
        }
        if self.decode_gain != 0 {
            let gain = super::math::exp2(super::math::mul16_p15(21771, self.decode_gain));
            for v in pcm[..n].iter_mut() {
                let x = super::math::mul16_32_p16(*v as i32, gain);
                *v = super::math::saturate(x, 32767) as i16;
            }
        }
        self.range_final = if len <= 1 {
            0
        } else {
            dec_storage.as_ref().map_or(0, |d| d.rng) ^ redundant_rng
        };
        self.prev_mode = mode;
        self.prev_redundancy = redundancy && !celt_to_silk;
        match celt_ret {
            Ok(_) => Ok(audiosize),
            Err(_) => Err(OpusError::InternalError),
        }
    }
}
