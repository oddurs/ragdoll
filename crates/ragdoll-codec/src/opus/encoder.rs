//! Opus encoder (libopus 1.3.1 `opus_encoder.c` and `repacketizer.c`, fixed point) for the
//! `OPUS_APPLICATION_RESTRICTED_LOWDELAY` application: every frame is CELT-only, so the packets
//! are byte-identical to libopus configured the same way at complexities up to 9 (complexity 10
//! adds libopus's floating-point signal analysis, which is not ported).

use super::celt_enc::{BITRATE_MAX, CeltEncoder};
use super::celt_tables::WINDOW;
use super::decoder::{
    BANDWIDTH_FULLBAND, BANDWIDTH_MEDIUMBAND, BANDWIDTH_NARROWBAND, BANDWIDTH_SUPERWIDEBAND,
    BANDWIDTH_WIDEBAND, MODE_CELT_ONLY, OpusError, parse_packet,
};
use super::math::{ilog2, mac16, mul16, mul16_q15, pshr32, saturate};
use super::range::Ec;

/// `OPUS_AUTO`.
pub const AUTO: i32 = -1000;
const FS: i32 = 48000;
const MONO_MUSIC_BW: [i32; 8] = [9000, 700, 9000, 700, 11000, 1000, 12000, 2000];
const MONO_VOICE_BW: [i32; 8] = [9000, 700, 9000, 700, 13500, 1000, 14000, 2000];
const STEREO_VOICE_THRESHOLD: i32 = 19000;
const STEREO_MUSIC_THRESHOLD: i32 = 17000;

/// Opus encoder state (`OpusEncoder`), restricted low-delay application at 48 kHz.
#[derive(Clone)]
pub struct OpusPacketEncoder {
    celt: CeltEncoder,
    channels: usize,
    use_vbr: bool,
    vbr_constraint: bool,
    user_bitrate_bps: i32,
    bitrate_bps: i32,
    complexity: i32,
    packet_loss_perc: i32,
    lsb_depth: i32,
    force_channels: i32,
    user_bandwidth: i32,
    max_bandwidth: i32,
    user_forced_mode: i32,
    stream_channels: usize,
    mode: i32,
    prev_mode: i32,
    prev_channels: usize,
    bandwidth: i32,
    auto_bandwidth: i32,
    first: bool,
    hp_mem: [i32; 4],
    hybrid_stereo_width_q14: i32,
    stereo_width_q14: i32,
    range_final: u32,
    /// The DC-rejected input of a frame, kept to reuse its allocation.
    pcm_buf: Vec<i16>,
    /// The range coder's buffer, kept to reuse its allocation.
    ec_buf: Vec<u8>,
}

impl OpusPacketEncoder {
    /// `opus_encoder_init(Fs = 48000, channels, OPUS_APPLICATION_RESTRICTED_LOWDELAY)`.
    pub fn new(channels: usize) -> Result<Self, OpusError> {
        if !(1..=2).contains(&channels) {
            return Err(OpusError::BadArg);
        }
        let mut celt = CeltEncoder::new(channels);
        celt.complexity = 9;
        Ok(Self {
            celt,
            channels,
            use_vbr: true,
            vbr_constraint: true,
            user_bitrate_bps: AUTO,
            bitrate_bps: 3000 + FS * channels as i32,
            complexity: 9,
            packet_loss_perc: 0,
            lsb_depth: 24,
            force_channels: AUTO,
            user_bandwidth: AUTO,
            max_bandwidth: BANDWIDTH_FULLBAND,
            user_forced_mode: AUTO,
            stream_channels: channels,
            mode: super::decoder::MODE_HYBRID,
            prev_mode: 0,
            prev_channels: 0,
            bandwidth: BANDWIDTH_FULLBAND,
            auto_bandwidth: 0,
            first: true,
            hp_mem: [0; 4],
            hybrid_stereo_width_q14: 1 << 14,
            stereo_width_q14: 0,
            range_final: 0,
            pcm_buf: Vec::new(),
            ec_buf: Vec::new(),
        })
    }

    /// Samples of look-ahead (`OPUS_GET_LOOKAHEAD`): the `OpusHead` pre-skip.
    pub fn lookahead(&self) -> usize {
        (FS / 400) as usize
    }

    /// `OPUS_SET_BITRATE`; `AUTO` or `BITRATE_MAX` are accepted as in libopus.
    pub fn set_bitrate(&mut self, value: i32) {
        self.user_bitrate_bps = if value == AUTO || value == BITRATE_MAX {
            value
        } else if value <= 500 {
            500
        } else {
            value.min(300000 * self.channels as i32)
        };
    }

    /// `OPUS_SET_VBR`.
    pub fn set_vbr(&mut self, vbr: bool) {
        self.use_vbr = vbr;
    }

    /// `OPUS_SET_VBR_CONSTRAINT`.
    pub fn set_vbr_constraint(&mut self, constrained: bool) {
        self.vbr_constraint = constrained;
    }

    /// `OPUS_SET_COMPLEXITY` (0 to 10; 10 behaves as 9 here, without signal analysis).
    pub fn set_complexity(&mut self, complexity: i32) {
        self.complexity = complexity.clamp(0, 10);
        self.celt.complexity = self.complexity;
    }

    /// `OPUS_SET_PACKET_LOSS_PERC`.
    pub fn set_packet_loss_perc(&mut self, perc: i32) {
        self.packet_loss_perc = perc.clamp(0, 100);
        self.celt.loss_rate = self.packet_loss_perc;
    }

    /// `OPUS_SET_LSB_DEPTH`.
    pub fn set_lsb_depth(&mut self, depth: i32) {
        self.lsb_depth = depth.clamp(8, 24);
    }

    /// `OPUS_SET_FORCE_CHANNELS` (`AUTO`, 1 or 2).
    pub fn set_force_channels(&mut self, channels: i32) {
        self.force_channels = channels;
    }

    /// `OPUS_SET_BANDWIDTH` (`AUTO` or a `BANDWIDTH_*` value).
    pub fn set_bandwidth(&mut self, bandwidth: i32) {
        self.user_bandwidth = bandwidth;
    }

    /// `OPUS_GET_FINAL_RANGE`.
    pub fn final_range(&self) -> u32 {
        self.range_final
    }

    /// `opus_encode`: encodes `frame_size` samples per channel of interleaved 16-bit `pcm`
    /// (2.5, 5, 10, 20, 40, 60, 80, 100 or 120 ms) into `out` (at most `out.len()` bytes,
    /// 1276 being enough), returning the packet length.
    pub fn encode(
        &mut self,
        pcm: &[i16],
        frame_size: usize,
        out: &mut [u8],
    ) -> Result<usize, OpusError> {
        let fs = frame_size as i32;
        let valid = [400, 200, 100, 50, 25].iter().any(|&r| r * fs == FS)
            || [3, 4, 5, 6].iter().any(|&k| 50 * fs == k * FS);
        if !valid || pcm.len() < frame_size * self.channels {
            return Err(OpusError::BadArg);
        }
        self.encode_native(pcm, frame_size, out, 16)
    }

    fn encode_native(
        &mut self,
        pcm: &[i16],
        frame_size: usize,
        data: &mut [u8],
        lsb_depth: i32,
    ) -> Result<usize, OpusError> {
        let out_data_bytes = data.len() as i32;
        let mut max_data_bytes = out_data_bytes.min(1276);
        self.range_final = 0;
        if frame_size == 0 || max_data_bytes <= 0 {
            return Err(OpusError::BadArg);
        }
        let fsz = frame_size as i32;
        if max_data_bytes == 1 && FS == fsz * 10 {
            return Err(OpusError::BufferTooSmall);
        }
        let lsb_depth = lsb_depth.min(self.lsb_depth);
        let ch = self.channels as i32;
        self.bitrate_bps = self.user_bitrate_to_bitrate(fsz, max_data_bytes);
        let mut frame_rate = FS / fsz;
        if !self.use_vbr {
            let frame_rate12 = 12 * FS / fsz;
            let cbr_bytes =
                ((12 * self.bitrate_bps / 8 + frame_rate12 / 2) / frame_rate12).min(max_data_bytes);
            self.bitrate_bps = cbr_bytes * frame_rate12 * 8 / 12;
            max_data_bytes = 1.max(cbr_bytes);
        }
        if max_data_bytes < 3
            || self.bitrate_bps < 3 * frame_rate * 8
            || (frame_rate < 50 && (max_data_bytes * frame_rate < 300 || self.bitrate_bps < 2400))
        {
            // Too few bits for audio: a TOC-only packet that decodes as packet loss.
            let mut tocmode = self.mode;
            let mut bw = if self.bandwidth == 0 {
                BANDWIDTH_NARROWBAND
            } else {
                self.bandwidth
            };
            let mut packet_code = 0u8;
            let mut num_multiframes = 0;
            if tocmode == 0 {
                tocmode = super::decoder::MODE_SILK_ONLY;
            }
            if frame_rate > 100 {
                tocmode = MODE_CELT_ONLY;
            }
            if frame_rate == 25 && tocmode != super::decoder::MODE_SILK_ONLY {
                frame_rate = 50;
                packet_code = 1;
            }
            if frame_rate <= 16 {
                if out_data_bytes == 1
                    || (tocmode == super::decoder::MODE_SILK_ONLY && frame_rate != 10)
                {
                    tocmode = super::decoder::MODE_SILK_ONLY;
                    packet_code = (frame_rate <= 12) as u8;
                    frame_rate = if frame_rate == 12 { 25 } else { 16 };
                } else {
                    num_multiframes = 50 / frame_rate;
                    frame_rate = 50;
                    packet_code = 3;
                }
            }
            if tocmode == super::decoder::MODE_SILK_ONLY && bw > BANDWIDTH_WIDEBAND {
                bw = BANDWIDTH_WIDEBAND;
            } else if tocmode == MODE_CELT_ONLY && bw == BANDWIDTH_MEDIUMBAND {
                bw = BANDWIDTH_NARROWBAND;
            } else if tocmode == super::decoder::MODE_HYBRID && bw <= BANDWIDTH_SUPERWIDEBAND {
                bw = BANDWIDTH_SUPERWIDEBAND;
            }
            data[0] = gen_toc(tocmode, frame_rate, bw, self.stream_channels) | packet_code;
            let mut ret = if packet_code <= 1 { 1 } else { 2 };
            max_data_bytes = max_data_bytes.max(ret as i32);
            if packet_code == 3 {
                data[1] = num_multiframes as u8;
            }
            if !self.use_vbr {
                packet_pad(data, ret, max_data_bytes as usize)?;
                ret = max_data_bytes as usize;
            }
            return Ok(ret);
        }
        let mut equiv_rate = compute_equiv_rate(
            self.bitrate_bps,
            ch,
            FS / fsz,
            self.use_vbr,
            0,
            self.complexity,
            self.packet_loss_perc,
        );
        // No signal analysis and the restricted low-delay application: the voice estimate is
        // the libopus default for non-VoIP applications.
        let voice_est = 48;
        if self.force_channels != AUTO && self.channels == 2 {
            self.stream_channels = self.force_channels as usize;
        } else if self.channels == 2 {
            let mut stereo_threshold = STEREO_MUSIC_THRESHOLD
                + ((voice_est * voice_est * (STEREO_VOICE_THRESHOLD - STEREO_MUSIC_THRESHOLD))
                    >> 14);
            if self.stream_channels == 2 {
                stereo_threshold -= 1000;
            } else {
                stereo_threshold += 1000;
            }
            self.stream_channels = if equiv_rate > stereo_threshold { 2 } else { 1 };
        } else {
            self.stream_channels = self.channels;
        }
        self.mode = MODE_CELT_ONLY;
        equiv_rate = compute_equiv_rate(
            self.bitrate_bps,
            self.stream_channels as i32,
            FS / fsz,
            self.use_vbr,
            self.mode,
            self.complexity,
            self.packet_loss_perc,
        );
        {
            // Bandwidth decision (always run in CELT mode).
            let mut thresholds = [0i32; 8];
            for i in 0..8 {
                thresholds[i] = MONO_MUSIC_BW[i]
                    + ((voice_est * voice_est * (MONO_VOICE_BW[i] - MONO_MUSIC_BW[i])) >> 14);
            }
            let mut bandwidth = BANDWIDTH_FULLBAND;
            loop {
                let idx = 2 * (bandwidth - BANDWIDTH_MEDIUMBAND) as usize;
                let mut threshold = thresholds[idx];
                let hysteresis = thresholds[idx + 1];
                if !self.first {
                    if self.auto_bandwidth >= bandwidth {
                        threshold -= hysteresis;
                    } else {
                        threshold += hysteresis;
                    }
                }
                if equiv_rate >= threshold {
                    break;
                }
                bandwidth -= 1;
                if bandwidth <= BANDWIDTH_NARROWBAND {
                    break;
                }
            }
            if bandwidth == BANDWIDTH_MEDIUMBAND {
                bandwidth = BANDWIDTH_WIDEBAND;
            }
            self.bandwidth = bandwidth;
            self.auto_bandwidth = bandwidth;
        }
        if self.bandwidth > self.max_bandwidth {
            self.bandwidth = self.max_bandwidth;
        }
        if self.user_bandwidth != AUTO {
            self.bandwidth = self.user_bandwidth;
        }
        self.celt.lsb_depth = lsb_depth;
        if self.bandwidth == BANDWIDTH_MEDIUMBAND {
            self.bandwidth = BANDWIDTH_WIDEBAND;
        }
        let curr_bandwidth = self.bandwidth;
        if fsz > FS / 50 {
            let enc_frame_size = (FS / 50) as usize;
            let nb_frames = frame_size / enc_frame_size;
            return self.encode_multiframe(pcm, nb_frames, enc_frame_size, data);
        }
        let mut pcm_buf = std::mem::take(&mut self.pcm_buf);
        pcm_buf.resize(frame_size * self.channels, 0);
        dc_reject(
            &pcm[..frame_size * self.channels],
            &mut pcm_buf,
            &mut self.hp_mem,
            frame_size,
            self.channels,
        );
        let endband = match curr_bandwidth {
            BANDWIDTH_NARROWBAND => 13,
            BANDWIDTH_MEDIUMBAND | BANDWIDTH_WIDEBAND => 17,
            BANDWIDTH_SUPERWIDEBAND => 19,
            _ => 21,
        };
        self.celt.end = endband;
        self.celt.stream_channels = self.stream_channels;
        self.celt.set_bitrate(BITRATE_MAX);
        self.celt.vbr = false;
        self.celt.disable_pf = false;
        self.celt.force_intra = false;
        if self.use_vbr {
            self.celt.vbr = true;
            self.celt.constrained_vbr = self.vbr_constraint;
            self.celt.set_bitrate(self.bitrate_bps);
        }
        self.stereo_width_q14 = if equiv_rate > 32000 {
            16384
        } else if equiv_rate < 16000 {
            0
        } else {
            16384 - 2048 * (32000 - equiv_rate) / (equiv_rate - 14000)
        };
        if self.channels == 2
            && (self.hybrid_stereo_width_q14 < (1 << 14) || self.stereo_width_q14 < (1 << 14))
        {
            let g1 = if self.hybrid_stereo_width_q14 == 16384 {
                32767
            } else {
                self.hybrid_stereo_width_q14 << 1
            };
            let g2 = if self.stereo_width_q14 == 16384 {
                32767
            } else {
                self.stereo_width_q14 << 1
            };
            stereo_fade(&mut pcm_buf, g1, g2, frame_size);
            self.hybrid_stereo_width_q14 = self.stereo_width_q14;
        }
        let nb_compr_bytes = (max_data_bytes - 1) as usize;
        let mut enc = Ec::encoder_with(std::mem::take(&mut self.ec_buf), nb_compr_bytes);
        self.celt.start = 0;
        let mut ret = 0usize;
        if enc.tell() <= 8 * nb_compr_bytes as i32 {
            self.celt.vbr = self.use_vbr;
            let r = self
                .celt
                .encode_with_ec(&pcm_buf, frame_size, nb_compr_bytes, &mut enc);
            self.pcm_buf = pcm_buf;
            ret = r.map_err(|_| OpusError::InternalError)?;
        } else {
            self.pcm_buf = pcm_buf;
        }
        data[0] = gen_toc(self.mode, FS / fsz, curr_bandwidth, self.stream_channels);
        self.range_final = enc.rng;
        data[1..1 + ret].copy_from_slice(&enc.buf[..ret]);
        self.prev_mode = self.mode;
        self.prev_channels = self.stream_channels;
        self.first = false;
        if enc.tell() > (max_data_bytes - 1) * 8 {
            if max_data_bytes < 2 {
                return Err(OpusError::BufferTooSmall);
            }
            data[1] = 0;
            ret = 1;
            self.range_final = 0;
        }
        self.ec_buf = enc.buf;
        ret += 1;
        if !self.use_vbr {
            packet_pad(data, ret, max_data_bytes as usize)?;
            ret = max_data_bytes as usize;
        }
        Ok(ret)
    }

    /// `encode_multiframe_packet`: codes 20 ms frames and joins them into one packet.
    fn encode_multiframe(
        &mut self,
        pcm: &[i16],
        nb_frames: usize,
        frame_size: usize,
        data: &mut [u8],
    ) -> Result<usize, OpusError> {
        let out_data_bytes = data.len() as i32;
        let max_header_bytes = if nb_frames == 2 {
            3
        } else {
            2 + (nb_frames as i32 - 1) * 2
        };
        let repacketize_len = if self.use_vbr || self.user_bitrate_bps == BITRATE_MAX {
            out_data_bytes
        } else {
            let cbr_bytes = 3 * self.bitrate_bps / (3 * 8 * FS / (frame_size * nb_frames) as i32);
            cbr_bytes.min(out_data_bytes)
        };
        let bytes_per_frame = 1276.min(1 + (repacketize_len - max_header_bytes) / nb_frames as i32);
        let (bak_mode, bak_bandwidth, bak_channels) = (
            self.user_forced_mode,
            self.user_bandwidth,
            self.force_channels,
        );
        self.user_forced_mode = self.mode;
        self.user_bandwidth = self.bandwidth;
        self.force_channels = self.stream_channels as i32;
        self.prev_channels = self.stream_channels;
        let mut frames: Vec<Vec<u8>> = Vec::with_capacity(nb_frames);
        for i in 0..nb_frames {
            let mut tmp = vec![0u8; bytes_per_frame.max(0) as usize];
            let off = i * self.channels * frame_size;
            let len = self.encode_native(&pcm[off..], frame_size, &mut tmp, 16)?;
            tmp.truncate(len);
            frames.push(tmp);
        }
        let mut payloads: Vec<&[u8]> = Vec::new();
        let toc = frames[0][0];
        for f in &frames {
            let parsed = parse_packet(f).map_err(|_| OpusError::InternalError)?;
            for &(o, l) in parsed.frames() {
                payloads.push(&f[o..o + l]);
            }
        }
        let ret = repacketize(
            toc,
            &payloads,
            &mut data[..repacketize_len as usize],
            !self.use_vbr,
        )?;
        self.user_forced_mode = bak_mode;
        self.user_bandwidth = bak_bandwidth;
        self.force_channels = bak_channels;
        Ok(ret)
    }

    fn user_bitrate_to_bitrate(&self, frame_size: i32, max_data_bytes: i32) -> i32 {
        if self.user_bitrate_bps == AUTO {
            60 * FS / frame_size + FS * self.channels as i32
        } else if self.user_bitrate_bps == BITRATE_MAX {
            max_data_bytes * 8 * FS / frame_size
        } else {
            self.user_bitrate_bps
        }
    }
}

fn compute_equiv_rate(
    bitrate: i32,
    channels: i32,
    frame_rate: i32,
    vbr: bool,
    mode: i32,
    complexity: i32,
    loss: i32,
) -> i32 {
    let mut equiv = bitrate;
    if frame_rate > 50 {
        equiv -= (40 * channels + 20) * (frame_rate - 50);
    }
    if !vbr {
        equiv -= equiv / 12;
    }
    equiv = equiv * (90 + complexity) / 100;
    if mode == super::decoder::MODE_SILK_ONLY || mode == super::decoder::MODE_HYBRID {
        if complexity < 2 {
            equiv = equiv * 4 / 5;
        }
        equiv -= equiv * loss / (6 * loss + 10);
    } else if mode == MODE_CELT_ONLY {
        if complexity < 5 {
            equiv = equiv * 9 / 10;
        }
    } else {
        equiv -= equiv * loss / (12 * loss + 20);
    }
    equiv
}

fn gen_toc(mode: i32, mut framerate: i32, bandwidth: i32, channels: usize) -> u8 {
    let mut period = 0;
    while framerate < 400 {
        framerate <<= 1;
        period += 1;
    }
    let mut toc = if mode == super::decoder::MODE_SILK_ONLY {
        (((bandwidth - BANDWIDTH_NARROWBAND) << 5) | ((period - 2) << 3)) as u8
    } else if mode == MODE_CELT_ONLY {
        let tmp = (bandwidth - BANDWIDTH_MEDIUMBAND).max(0);
        (0x80 | (tmp << 5) | (period << 3)) as u8
    } else {
        (0x60 | ((bandwidth - BANDWIDTH_SUPERWIDEBAND) << 4) | ((period - 2) << 3)) as u8
    };
    toc |= ((channels == 2) as u8) << 2;
    toc
}

/// `dc_reject` (fixed point): a one-pole 3 Hz high-pass.
fn dc_reject(input: &[i16], out: &mut [i16], hp_mem: &mut [i32; 4], len: usize, channels: usize) {
    let shift = ilog2(FS / (3 * 4));
    // One sample through a channel's filter, whose state is `m`. The state is a serial
    // recursion, so the channels of a frame advance side by side.
    let step = |v: i16, m: &mut i32| {
        let x = (v as i32) << 14;
        let y = x.wrapping_sub(*m);
        *m = m.wrapping_add(pshr32(x.wrapping_sub(*m), shift));
        saturate(pshr32(y, 14), 32767) as i16
    };
    let (input, out) = (&input[..channels * len], &mut out[..channels * len]);
    if channels == 2 {
        let (mut m0, mut m1) = (hp_mem[0], hp_mem[2]);
        for (o, x) in out
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(input.as_chunks::<2>().0)
        {
            o[0] = step(x[0], &mut m0);
            o[1] = step(x[1], &mut m1);
        }
        (hp_mem[0], hp_mem[2]) = (m0, m1);
    } else {
        let mut m = hp_mem[0];
        for (o, &x) in out.iter_mut().zip(input) {
            *o = step(x, &mut m);
        }
        hp_mem[0] = m;
    }
}

/// `stereo_fade`: narrows the stereo image from gain `g1` to `g2` over the overlap window.
fn stereo_fade(buf: &mut [i16], g1: i32, g2: i32, frame_size: usize) {
    let overlap = 120usize;
    let g1 = (32767 - g1) as i16 as i32;
    let g2 = (32767 - g2) as i16 as i32;
    for i in 0..frame_size {
        let g = if i < overlap {
            let w = mul16_q15(WINDOW[i] as i32, WINDOW[i] as i32);
            mac16(mul16(w, g2), 32767 - w, g1) >> 15
        } else {
            g2
        };
        let diff = ((buf[2 * i] as i32 - buf[2 * i + 1] as i32) >> 1) as i16 as i32;
        let diff = mul16_q15(g, diff);
        buf[2 * i] = (buf[2 * i] as i32 - diff) as i16;
        buf[2 * i + 1] = (buf[2 * i + 1] as i32 + diff) as i16;
    }
}

fn encode_size(size: usize, out: &mut [u8]) -> usize {
    if size < 252 {
        out[0] = size as u8;
        1
    } else {
        out[0] = (252 + (size & 3)) as u8;
        out[1] = ((size - out[0] as usize) >> 2) as u8;
        2
    }
}

/// `opus_repacketizer_out_range_impl` (not self-delimited): writes `frames` (sharing `toc`) as
/// one packet into `data`, padding it to `data.len()` when `pad` is set.
fn repacketize(toc: u8, frames: &[&[u8]], data: &mut [u8], pad: bool) -> Result<usize, OpusError> {
    let maxlen = data.len();
    let count = frames.len();
    let len: Vec<usize> = frames.iter().map(|f| f.len()).collect();
    let mut tot_size;
    let mut ptr = 0usize;
    let mut header = Vec::new();
    if count == 1 {
        tot_size = len[0] + 1;
        if tot_size > maxlen {
            return Err(OpusError::BufferTooSmall);
        }
        header.push(toc & 0xfc);
    } else if count == 2 {
        if len[1] == len[0] {
            tot_size = 2 * len[0] + 1;
            if tot_size > maxlen {
                return Err(OpusError::BufferTooSmall);
            }
            header.push((toc & 0xfc) | 1);
        } else {
            tot_size = len[0] + len[1] + 2 + (len[0] >= 252) as usize;
            if tot_size > maxlen {
                return Err(OpusError::BufferTooSmall);
            }
            header.push((toc & 0xfc) | 2);
            let mut sz = [0u8; 2];
            let k = encode_size(len[0], &mut sz);
            header.extend_from_slice(&sz[..k]);
        }
    } else {
        tot_size = 0;
    }
    if count > 2 || (pad && tot_size < maxlen) {
        header.clear();
        let vbr = len.iter().any(|&l| l != len[0]);
        if vbr {
            tot_size = 2;
            for &l in &len[..count - 1] {
                tot_size += 1 + (l >= 252) as usize + l;
            }
            tot_size += len[count - 1];
            if tot_size > maxlen {
                return Err(OpusError::BufferTooSmall);
            }
            header.push((toc & 0xfc) | 3);
            header.push(count as u8 | 0x80);
        } else {
            tot_size = count * len[0] + 2;
            if tot_size > maxlen {
                return Err(OpusError::BufferTooSmall);
            }
            header.push((toc & 0xfc) | 3);
            header.push(count as u8);
        }
        let pad_amount = if pad { maxlen - tot_size } else { 0 };
        if pad_amount != 0 {
            header[1] |= 0x40;
            let nb_255s = (pad_amount - 1) / 255;
            header.extend(std::iter::repeat_n(255u8, nb_255s));
            header.push((pad_amount - 255 * nb_255s - 1) as u8);
            tot_size += pad_amount;
        }
        if vbr {
            for &l in &len[..count - 1] {
                let mut sz = [0u8; 2];
                let k = encode_size(l, &mut sz);
                header.extend_from_slice(&sz[..k]);
            }
        }
    }
    // Frames may live in `data` itself (opus_packet_pad), so assemble before writing.
    let mut out = header;
    for f in frames {
        out.extend_from_slice(f);
    }
    data[..out.len()].copy_from_slice(&out);
    ptr += out.len();
    if pad {
        data[ptr..maxlen].fill(0);
    }
    Ok(tot_size)
}

/// `opus_packet_pad`: grows the `len`-byte packet at the start of `data` to `new_len` bytes.
fn packet_pad(data: &mut [u8], len: usize, new_len: usize) -> Result<(), OpusError> {
    if len == new_len {
        return Ok(());
    }
    if len > new_len || len < 1 {
        return Err(OpusError::BadArg);
    }
    let packet = data[..len].to_vec();
    let parsed = parse_packet(&packet).map_err(|_| OpusError::InvalidPacket)?;
    let frames: Vec<&[u8]> = parsed
        .frames()
        .iter()
        .map(|&(o, l)| &packet[o..o + l])
        .collect();
    repacketize(packet[0], &frames, &mut data[..new_len], true)?;
    Ok(())
}
