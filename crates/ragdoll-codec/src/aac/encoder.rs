//! AAC-LC encoder with rate control (`BUG-0082`).
//!
//! Long blocks only (`ONLY_LONG`, KBD window). Per frame:
//! 1. MDCT analysis, and per-band mid/side for stereo where it lowers `E₀ · E₁`.
//! 2. Scale factors from an energy model: band `b` gets `sf = g + shape(b)` with
//!    `shape(b) = 2 · β · log2(rms_b)` (`β = NOISE_SHAPING`), so the quantization step grows
//!    slowly with the band level: louder bands take more noise, quiet ones less, a middle road
//!    between equal noise everywhere (the most SNR per bit) and equal SNR per band (noise
//!    following the masking of the band's own energy). Bands too quiet to rise above the step
//!    quantize to zero and cost nothing.
//! 3. Rate control: the global offset `g` is searched (bisection on exact bit counts) so the
//!    frame lands inside the window the bit reservoir allows: at most the average plus what the
//!    reservoir holds (and never more than 6144 bits per channel, the decoder's input buffer),
//!    at least what keeps the reservoir from overflowing. Inside that window the frame keeps the
//!    offset carried from the previous frame (near-constant quality, the reservoir absorbing
//!    frames that need more or fewer bits); the carried offset then moves so that active frames
//!    spend the average plus a share of the reservoir's excess.
//! 4. Codebooks by dynamic programming over the bands (section headers included), scale factor
//!    differences within `±60`, and quantized values within `8191`.
//!
//! When even the finest quantization cannot use the bits that keep the average bitrate (digital
//! silence, a lone tone), fill elements pad the frame, as constant-bitrate encoders do.

use ragdoll_core::{AudioFrame, BitWriter, Rational64, Result, SampleFormat, Timestamp};
use ragdoll_simd::{MdctPlan, kbd_window_f32};

use super::{
    AdtsHeader, AudioSpecificConfig, HCB_1_BITS, HCB_1_CODES, HCB_2_BITS, HCB_2_CODES, HCB_3_BITS,
    HCB_3_CODES, HCB_4_BITS, HCB_4_CODES, HCB_5_BITS, HCB_5_CODES, HCB_6_BITS, HCB_6_CODES,
    HCB_7_BITS, HCB_7_CODES, HCB_8_BITS, HCB_8_CODES, HCB_9_BITS, HCB_9_CODES, HCB_10_BITS,
    HCB_10_CODES, HCB_11_BITS, HCB_11_CODES, HCB_SF_BITS, HCB_SF_CODES, swb_offsets_long,
};
use crate::pcm::convert_audio_frame;

/// Zero-point scale factor offset matching `MdctPlan` (`2048`-sample MDCT with `2/1024` IMDCT
/// normalization) and `libavcodec`'s `POW_SF2_ZERO = 200` (`200 - 36 = 164`): a quantized value
/// `q` in a band of scale factor `sf` decodes to `q^(4/3) · 2^((sf - 164) / 4)`.
const SF_OFFSET: i32 = 164;
/// Largest quantized magnitude the escape codebook carries.
const MAX_QUANT: i32 = 8191;
/// Rounding offset of the AAC quantizer `floor(x^(3/4) + 0.4054)` (ISO/IEC 14496-3 informative
/// encoder, `libavcodec`).
const ROUNDING: f32 = 0.4054;
/// Bits one channel may occupy in the decoder's input buffer (ISO/IEC 14496-3 4.5.3.2).
const MAX_CHANNEL_BITS: usize = 6144;
/// Exponent of the band level in the quantization step (see the module documentation).
const NOISE_SHAPING: f32 = 0.25;
/// Maximum scale-factor bands of a long window (`32 kHz`).
const MAX_BANDS: usize = 51;
/// Largest absolute value of each spectral codebook (`11` escapes up to `8191`).
const CODEBOOK_LAV: [i32; 12] = [0, 1, 1, 2, 2, 4, 4, 7, 7, 12, 12, MAX_QUANT];

/// Per-channel analysis of one frame.
struct Channel {
    coeffs: [f32; 1024],
    /// `|x|^(3/4)`, the quantizer input before the band's step.
    xp: [f32; 1024],
    /// Scale-factor shape `2 · β · log2(rms)` per band; `None` for silent bands.
    shape: [Option<i32>; MAX_BANDS],
    /// Smallest scale factor that keeps the band's largest value within `MAX_QUANT`.
    sf_floor: [i32; MAX_BANDS],
}

/// One channel's coding at a given offset.
#[derive(Clone)]
struct Coding {
    sf: [i32; MAX_BANDS],
    cb: [u8; MAX_BANDS],
    q: [i32; 1024],
    bits: usize,
}

/// Pure-Rust ISO/IEC 14496-3 AAC-LC Encoder (`1024` samples per frame) with rate control.
#[derive(Clone, Debug)]
pub struct AacEncoder {
    pub asc: AudioSpecificConfig,
    pub bitrate_bps: u32,
    mdct_long: MdctPlan,
    win_kbd_long: Vec<f32>,
    history: [[f32; 1024]; 2],
    /// Bits saved by earlier frames below the average (the bit reservoir's fill).
    reservoir: i64,
    /// Global scale-factor offset carried from frame to frame (constant quality).
    offset: f64,
}

impl AacEncoder {
    /// Creates a new `AacEncoder` for `sample_rate`, `channels` (`1` or `2`), and `bitrate_bps`
    /// (e.g. `128_000`), clamped to `8 kbit/s` per channel and to the `6144` bits per channel and
    /// frame the decoder's buffer holds (`288 kbit/s` per channel at 48 kHz).
    pub fn new(sample_rate: u32, channels: u8, bitrate_bps: u32) -> Self {
        let asc = AudioSpecificConfig::new_aac_lc(sample_rate, channels);
        let ch = asc.channel_configuration.clamp(1, 2) as u64;
        let ceiling = (MAX_CHANNEL_BITS as u64 * ch * asc.sample_rate as u64 / 1024) as u32;
        Self {
            asc,
            bitrate_bps: bitrate_bps.clamp(8_000 * ch as u32, ceiling),
            mdct_long: MdctPlan::new(2048),
            win_kbd_long: kbd_window_f32(2048, 4.0),
            history: [[0.0; 1024]; 2],
            reservoir: 0,
            offset: 100.0,
        }
    }

    /// Returns the 2-byte `AudioSpecificConfig` (`esds` / `CodecPrivate`) for MP4/MKV muxers.
    pub fn extra_data(&self) -> [u8; 2] {
        self.asc.to_bytes()
    }

    fn channels(&self) -> usize {
        (self.asc.channel_configuration as usize).clamp(1, 2)
    }

    /// Average bits per frame at the target bitrate.
    fn mean_frame_bits(&self) -> i64 {
        (self.bitrate_bps as u64 * 1024 / self.asc.sample_rate as u64) as i64
    }

    /// Encodes a 1024-sample [`AudioFrame`] into a raw ISO/IEC 14496-3 AAC-LC access unit (for MP4/MKV).
    pub fn encode_raw_frame(&mut self, frame: &AudioFrame) -> Result<Vec<u8>> {
        self.encode_frame(frame, 0)
    }

    /// Encodes a 1024-sample [`AudioFrame`] into a 7-byte ADTS-framed AAC-LC packet (for MPEG-TS / `.aac`).
    pub fn encode_adts_frame(&mut self, frame: &AudioFrame) -> Result<Vec<u8>> {
        let raw = self.encode_frame(frame, 7 * 8)?;
        let hdr = AdtsHeader::new(&self.asc, raw.len());
        let mut out = Vec::with_capacity(7 + raw.len());
        out.extend_from_slice(&hdr.to_bytes());
        out.extend_from_slice(&raw);
        Ok(out)
    }

    /// Encodes an arbitrary-length `AudioFrame` (splitting/padding into 1024-sample AAC frames) into raw AAC access units.
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
        let total_samples = f32_frame.num_samples;
        let src = f32_frame.as_f32_slice();
        let tb = Rational64::new(1, self.asc.sample_rate as i64);
        let base_pts = frame.pts.rescale_to(tb).ticks;

        let mut packets = Vec::new();
        let mut offset = 0usize;
        while offset < total_samples {
            let take = (total_samples - offset).min(1024);
            let mut sub = AudioFrame::new(
                self.asc.sample_rate,
                f32_frame.channels,
                SampleFormat::F32le,
                1024,
            );
            sub.as_f32_mut_slice()[..take * ch]
                .copy_from_slice(&src[offset * ch..(offset + take) * ch]);
            let raw = self.encode_raw_frame(&sub)?;
            let pts = Timestamp::new(base_pts + (offset as i64), tb);
            packets.push((raw, pts, 1024));
            offset += 1024;
        }
        Ok(packets)
    }

    /// Encodes one frame whose container adds `header_bits` per frame (counted in the bitrate).
    fn encode_frame(&mut self, frame: &AudioFrame, header_bits: usize) -> Result<Vec<u8>> {
        let f32_frame = if frame.format == SampleFormat::F32le {
            frame.clone()
        } else {
            convert_audio_frame(frame, SampleFormat::F32le)?
        };
        let ch_count = self.channels();
        let src = f32_frame.as_f32_slice();
        let in_samples = f32_frame.num_samples.min(1024);
        let src_ch = (f32_frame.channels as usize).max(1);

        let mut coeffs = [[0.0f32; 1024]; 2];
        for c in 0..ch_count {
            let mut cur = [0.0f32; 1024];
            for i in 0..in_samples {
                cur[i] = src[i * src_ch + c.min(src_ch - 1)];
            }
            let mut win = [0.0f32; 2048];
            for i in 0..1024 {
                win[i] = self.history[c][i] * self.win_kbd_long[i];
                win[1024 + i] = cur[i] * self.win_kbd_long[1024 + i];
            }
            self.history[c] = cur;
            self.mdct_long.mdct(&win, &mut coeffs[c]);
        }

        let swb = swb_offsets_long(self.asc.sampling_frequency_index);
        let bands = (swb.len() - 1).min(MAX_BANDS);

        // Mid/side per band where it lowers the product of the two channels' energies (which
        // the bits of two quantized channels roughly follow); then `L = M + S`, `R = M - S`.
        let mut ms = [false; MAX_BANDS];
        if ch_count == 2 {
            let [l, r] = &mut coeffs;
            for b in 0..bands {
                let band = swb[b] as usize..swb[b + 1] as usize;
                let (mut el, mut er, mut em, mut es) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for i in band.clone() {
                    let (x, y) = (l[i] as f64, r[i] as f64);
                    el += x * x;
                    er += y * y;
                    em += (x + y) * (x + y) * 0.25;
                    es += (x - y) * (x - y) * 0.25;
                }
                if em * es < el * er * 0.9 {
                    ms[b] = true;
                    for i in band {
                        (l[i], r[i]) = ((l[i] + r[i]) * 0.5, (l[i] - r[i]) * 0.5);
                    }
                }
            }
        }
        let chans: Vec<Channel> = coeffs[..ch_count]
            .iter()
            .map(|c| analyse(c, swb, bands))
            .collect();

        // Fixed bits: elements, ics_info, M/S mask, the end element and byte alignment.
        let ms_present = ms[..bands].iter().any(|&m| m);
        let fixed_bits = header_bits
            + if ch_count == 1 {
                3 + 4 + 11
            } else {
                3 + 4 + 1 + 11 + 2 + if ms_present { bands } else { 0 }
            }
            + 3
            + 7;

        // Rate control.
        let mean = self.mean_frame_bits();
        let max_frame = (MAX_CHANNEL_BITS * ch_count) as i64;
        let max_reservoir = (max_frame - mean).max(0);
        let hi = (mean + self.reservoir).min(max_frame);
        let lo = (mean + self.reservoir - max_reservoir).max(0);
        let eval = |g: i32| -> (usize, Vec<Coding>) {
            let codings: Vec<Coding> = chans
                .iter()
                .map(|c| code_channel(c, swb, bands, g))
                .collect();
            (
                fixed_bits + codings.iter().map(|c| c.bits).sum::<usize>(),
                codings,
            )
        };
        // Bits fall as the offset grows (coarser steps): search from the carried offset, with
        // steps doubling away from it and then bisection.
        const G_MIN: i32 = -120;
        const G_MAX: i32 = 255 + 120;
        let g0 = (self.offset.round() as i32).clamp(G_MIN, G_MAX);
        let mut pick = (g0, eval(g0));
        let start_bits = pick.1.0 as i64;
        if pick.1.0 as i64 > hi {
            // The coarsest offset known to overflow, then the finest that fits.
            let (mut over, mut step) = (g0, 1);
            loop {
                let g = (over + step).min(G_MAX);
                let r = eval(g);
                if r.0 as i64 <= hi || g == G_MAX {
                    pick = (g, r);
                    break;
                }
                (over, step) = (g, step * 2);
            }
            while pick.0 - over > 1 {
                let g = (over + pick.0) / 2;
                let r = eval(g);
                if r.0 as i64 <= hi {
                    pick = (g, r);
                } else {
                    over = g;
                }
            }
        } else if (pick.1.0 as i64) < lo {
            // The finest offset known to spend too few bits (it always fits), then the coarsest
            // that spends enough; taken if it also fits.
            let mut enough = None;
            let mut step = 1;
            while pick.0 > G_MIN {
                let g = (pick.0 - step).max(G_MIN);
                let r = eval(g);
                if r.0 as i64 >= lo {
                    enough = Some((g, r));
                    break;
                }
                (pick, step) = ((g, r), step * 2);
            }
            if let Some(mut e) = enough {
                while pick.0 - e.0 > 1 {
                    let g = (e.0 + pick.0) / 2;
                    let r = eval(g);
                    if r.0 as i64 >= lo {
                        e = (g, r);
                    } else {
                        pick = (g, r);
                    }
                }
                if e.1.0 as i64 <= hi {
                    pick = e;
                }
            }
        }
        let (g, (bits, codings)) = pick;

        // Pad with fill elements what even the finest coding leaves the reservoir unable to hold.
        let fill_bits = (lo - bits as i64).max(0) as usize;
        let mut bw = BitWriter::with_capacity((bits + fill_bits) / 8 + 16);
        if ch_count == 1 {
            bw.write_bits(0, 3); // ID_SCE
            bw.write_bits(0, 4); // element_instance_tag
            write_ics(&mut bw, &codings[0], swb, bands, true);
        } else {
            bw.write_bits(1, 3); // ID_CPE
            bw.write_bits(0, 4); // element_instance_tag
            bw.write_bits(1, 1); // common_window
            write_ics_info(&mut bw, bands);
            if ms_present {
                bw.write_bits(1, 2); // ms_mask_present: per band
                for &m in &ms[..bands] {
                    bw.write_bit(m);
                }
            } else {
                bw.write_bits(0, 2);
            }
            write_ics(&mut bw, &codings[0], swb, bands, false);
            write_ics(&mut bw, &codings[1], swb, bands, false);
        }
        write_fill(&mut bw, fill_bits);
        bw.write_bits(7, 3); // ID_END
        bw.align_to_byte();
        let out = bw.finish();
        let used = out.len() * 8 + header_bits;
        debug_assert!(used <= bits + fill_bits + 16 && used <= MAX_CHANNEL_BITS * ch_count);
        // The next frame starts from this frame's offset, corrected towards spending the average
        // plus an eighth of what the reservoir holds above half full (frame bits roughly double
        // for every 6 steps finer; half of that correction is applied). Frames that need little
        // even at the carried offset (silence, quiet passages) leave it alone: they cannot show
        // what the signal needs, and following them would bank fine steps that the next loud
        // frame then drains the whole reservoir with.
        if start_bits >= mean / 4 {
            let target = (mean + (self.reservoir - max_reservoir / 2) / 8).max(mean / 4);
            self.offset = g as f64 + 3.0 * (used as f64 / target as f64).log2();
        }
        self.reservoir += mean - used as i64;
        Ok(out)
    }
}

/// `|x|^(3/4)`, band shapes and step floors of one channel's spectrum.
fn analyse(coeffs: &[f32; 1024], swb: &[u16], bands: usize) -> Channel {
    let mut ch = Channel {
        coeffs: *coeffs,
        xp: [0.0; 1024],
        shape: [None; MAX_BANDS],
        sf_floor: [0; MAX_BANDS],
    };
    for (p, &x) in ch.xp.iter_mut().zip(coeffs.iter()) {
        *p = x.abs().powf(0.75);
    }
    for b in 0..bands {
        let band = swb[b] as usize..swb[b + 1] as usize;
        let (mut energy, mut peak) = (0.0f64, 0.0f32);
        for i in band.clone() {
            energy += (coeffs[i] as f64).powi(2);
            peak = peak.max(ch.xp[i]);
        }
        let rms = (energy / band.len() as f64).sqrt();
        if peak > 0.0 && rms > 1e-30 {
            ch.shape[b] = Some((2.0 * NOISE_SHAPING * rms.log2() as f32).round() as i32);
            // `peak · 2^(-(sf - 164) · 3/16) + ROUNDING <= MAX_QUANT + 1`
            let limit = (MAX_QUANT as f32 + 1.0 - ROUNDING).log2();
            ch.sf_floor[b] = SF_OFFSET + (16.0 / 3.0 * (peak.log2() - limit)).ceil() as i32;
        }
    }
    ch
}

/// Quantizes and codes one channel at global offset `g`.
fn code_channel(ch: &Channel, swb: &[u16], bands: usize, g: i32) -> Coding {
    let mut c = Coding {
        sf: [0; MAX_BANDS],
        cb: [0; MAX_BANDS],
        q: [0; 1024],
        bits: 8 + 3, // global_gain; pulse, TNS and gain-control flags
    };
    // Scale factors within 0..=255 and the step floors, then raised where needed so that coded
    // neighbours differ by at most 60 (which raising a band can only help, unless it empties
    // the band and brings two others together: then again).
    let mut sf = [None::<i32>; MAX_BANDS];
    for b in 0..bands {
        sf[b] = ch.shape[b].map(|shape| (g + shape).max(ch.sf_floor[b]).clamp(0, 255));
    }
    let mut peaks = [0i32; MAX_BANDS];
    loop {
        let coded: Vec<usize> = (0..bands).filter(|&b| sf[b].is_some()).collect();
        for w in coded.windows(2) {
            let (a, b) = (sf[w[0]].unwrap(), sf[w[1]].unwrap());
            sf[w[1]] = Some(b.max(a - 60));
        }
        for w in coded.windows(2).rev() {
            let (a, b) = (sf[w[0]].unwrap(), sf[w[1]].unwrap());
            sf[w[0]] = Some(a.max(b - 60));
        }
        let mut emptied = false;
        for &b in &coded {
            let s = sf[b].unwrap();
            let band = swb[b] as usize..swb[b + 1] as usize;
            let inv = (-(s - SF_OFFSET) as f32 * 3.0 / 16.0).exp2();
            let mut peak = 0;
            for i in band {
                let q = ((ch.xp[i] * inv + ROUNDING) as i32).min(MAX_QUANT);
                peak = peak.max(q);
                c.q[i] = if ch.coeffs[i] < 0.0 { -q } else { q };
            }
            peaks[b] = peak;
            if peak == 0 {
                sf[b] = None;
                emptied = true;
            }
        }
        if !emptied {
            break;
        }
    }
    let mut band_bits = [[u32::MAX; 12]; MAX_BANDS];
    for b in 0..bands {
        let Some(s) = sf[b] else {
            continue;
        };
        c.sf[b] = s;
        let band = swb[b] as usize..swb[b + 1] as usize;
        let lo_cb = CODEBOOK_LAV
            .iter()
            .position(|&lav| lav >= peaks[b])
            .unwrap();
        for cb in lo_cb..12 {
            // Two classes of codebooks (four books) cover the useful choices.
            if cb >= lo_cb + 4 && cb != 11 {
                continue;
            }
            band_bits[b][cb] = spectral_bits(cb as u8, &c.q[band.clone()]);
        }
    }
    // Codebooks: dynamic programming over the bands, a new section costing 4 + 5 bits (and 5
    // more per 31 bands), codebook 0 free for empty bands.
    let mut cost = [[u32::MAX; 12]; MAX_BANDS + 1];
    let mut from = [[0u8; 12]; MAX_BANDS + 1];
    let mut run = [[0u32; 12]; MAX_BANDS + 1];
    for b in 0..bands {
        let empty = sf[b].is_none();
        for cb in 0..12usize {
            let own = if cb == 0 {
                if empty { 0 } else { continue }
            } else if empty || band_bits[b][cb] == u32::MAX {
                continue;
            } else {
                band_bits[b][cb]
            };
            let (best, best_from, best_run) = if b == 0 {
                (9, 0u8, 1)
            } else {
                let mut best = (u32::MAX, 0u8, 1u32);
                for p in 0..12usize {
                    if cost[b - 1][p] == u32::MAX {
                        continue;
                    }
                    let (extra, r) = if p == cb {
                        let r = run[b - 1][p] + 1;
                        (if r % 31 == 0 { 5 } else { 0 }, r)
                    } else {
                        (9, 1)
                    };
                    if cost[b - 1][p] + extra < best.0 {
                        best = (cost[b - 1][p] + extra, p as u8, r);
                    }
                }
                best
            };
            if best == u32::MAX {
                continue;
            }
            cost[b][cb] = best + own;
            from[b][cb] = best_from;
            run[b][cb] = best_run;
        }
    }
    let (mut cb, mut total) = (0usize, u32::MAX);
    for (k, &v) in cost[bands - 1].iter().enumerate() {
        if v < total {
            (cb, total) = (k, v);
        }
    }
    for b in (0..bands).rev() {
        c.cb[b] = cb as u8;
        cb = from[b][cb] as usize;
    }
    c.bits += total as usize;
    // Scale factor differences.
    let mut last = c.cb.iter().position(|&cb| cb != 0).map_or(0, |b| c.sf[b]);
    for b in 0..bands {
        if c.cb[b] != 0 {
            c.bits += HCB_SF_BITS[(c.sf[b] - last + 60) as usize] as usize;
            last = c.sf[b];
        }
    }
    c
}

/// Spectral bits of the quantized values `q` (one band) in codebook `cb`, sign and escape bits
/// included; `u32::MAX` when a value exceeds the book.
fn spectral_bits(cb: u8, q: &[i32]) -> u32 {
    let mut bits = 0u32;
    match cb {
        1..=4 => {
            for t in q.as_chunks::<4>().0 {
                bits += match cb {
                    1 => HCB_1_BITS[quad_signed(t)] as u32,
                    2 => HCB_2_BITS[quad_signed(t)] as u32,
                    3 => HCB_3_BITS[quad_unsigned(t)] as u32 + nonzero(t),
                    _ => HCB_4_BITS[quad_unsigned(t)] as u32 + nonzero(t),
                };
            }
        }
        5 | 6 => {
            for t in q.as_chunks::<2>().0 {
                let i = ((t[0] + 4) * 9 + t[1] + 4) as usize;
                bits += if cb == 5 {
                    HCB_5_BITS[i]
                } else {
                    HCB_6_BITS[i]
                } as u32;
            }
        }
        _ => {
            for t in q.as_chunks::<2>().0 {
                let (a, b) = (t[0].unsigned_abs(), t[1].unsigned_abs());
                bits += nonzero(t)
                    + match cb {
                        7 => HCB_7_BITS[(a * 8 + b) as usize],
                        8 => HCB_8_BITS[(a * 8 + b) as usize],
                        9 => HCB_9_BITS[(a * 13 + b) as usize],
                        10 => HCB_10_BITS[(a * 13 + b) as usize],
                        _ => HCB_11_BITS[(a.min(16) * 17 + b.min(16)) as usize],
                    } as u32;
                if cb == 11 {
                    bits += escape_bits(a) + escape_bits(b);
                }
            }
        }
    }
    bits
}

fn quad_signed(t: &[i32]) -> usize {
    ((t[0] + 1) * 27 + (t[1] + 1) * 9 + (t[2] + 1) * 3 + t[3] + 1) as usize
}

fn quad_unsigned(t: &[i32]) -> usize {
    (t[0].abs() * 27 + t[1].abs() * 9 + t[2].abs() * 3 + t[3].abs()) as usize
}

fn nonzero(t: &[i32]) -> u32 {
    t.iter().filter(|&&v| v != 0).count() as u32
}

/// Escape sequence length of magnitude `a` in codebook 11: `N` ones, a zero and `N + 4` bits for
/// `2^(N + 4) <= a < 2^(N + 5)`.
fn escape_bits(a: u32) -> u32 {
    if a < 16 {
        0
    } else {
        2 * (31 - a.leading_zeros() - 4) + 5
    }
}

fn write_ics_info(bw: &mut BitWriter, bands: usize) {
    bw.write_bits(0, 1); // ics_reserved_bit
    bw.write_bits(0, 2); // ONLY_LONG_SEQUENCE
    bw.write_bits(1, 1); // window_shape = KBD
    bw.write_bits(bands as u64, 6); // max_sfb
    bw.write_bits(0, 1); // predictor_data_present
}

/// Writes one `individual_channel_stream()`.
fn write_ics(bw: &mut BitWriter, c: &Coding, swb: &[u16], bands: usize, ics_info: bool) {
    let global_gain = c.cb.iter().position(|&cb| cb != 0).map_or(100, |b| c.sf[b]);
    bw.write_bits(global_gain as u64, 8);
    if ics_info {
        write_ics_info(bw, bands);
    }
    // section_data()
    let mut b = 0;
    while b < bands {
        let cb = c.cb[b];
        let mut len = 1;
        while b + len < bands && c.cb[b + len] == cb {
            len += 1;
        }
        bw.write_bits(cb as u64, 4);
        let mut rem = len;
        while rem >= 31 {
            bw.write_bits(31, 5);
            rem -= 31;
        }
        bw.write_bits(rem as u64, 5);
        b += len;
    }
    // scale_factor_data()
    let mut last = global_gain;
    for b in 0..bands {
        if c.cb[b] != 0 {
            let i = (c.sf[b] - last + 60) as usize;
            bw.write_bits(HCB_SF_CODES[i] as u64, HCB_SF_BITS[i] as u32);
            last = c.sf[b];
        }
    }
    bw.write_bits(0, 3); // pulse_data_present, tns_data_present, gain_control_data_present
    // spectral_data()
    for b in 0..bands {
        let band = &c.q[swb[b] as usize..swb[b + 1] as usize];
        match c.cb[b] {
            0 => {}
            cb @ 1..=4 => {
                for t in band.as_chunks::<4>().0 {
                    if cb <= 2 {
                        let i = quad_signed(t);
                        let (code, len) = if cb == 1 {
                            (HCB_1_CODES[i], HCB_1_BITS[i])
                        } else {
                            (HCB_2_CODES[i], HCB_2_BITS[i])
                        };
                        bw.write_bits(code as u64, len as u32);
                    } else {
                        let i = quad_unsigned(t);
                        let (code, len) = if cb == 3 {
                            (HCB_3_CODES[i], HCB_3_BITS[i])
                        } else {
                            (HCB_4_CODES[i], HCB_4_BITS[i])
                        };
                        bw.write_bits(code as u64, len as u32);
                        write_signs(bw, t);
                    }
                }
            }
            cb @ (5 | 6) => {
                for t in band.as_chunks::<2>().0 {
                    let i = ((t[0] + 4) * 9 + t[1] + 4) as usize;
                    let (code, len) = if cb == 5 {
                        (HCB_5_CODES[i], HCB_5_BITS[i])
                    } else {
                        (HCB_6_CODES[i], HCB_6_BITS[i])
                    };
                    bw.write_bits(code as u64, len as u32);
                }
            }
            cb => {
                for t in band.as_chunks::<2>().0 {
                    let (a, b) = (t[0].unsigned_abs(), t[1].unsigned_abs());
                    let (code, len) = match cb {
                        7 => (
                            HCB_7_CODES[(a * 8 + b) as usize],
                            HCB_7_BITS[(a * 8 + b) as usize],
                        ),
                        8 => (
                            HCB_8_CODES[(a * 8 + b) as usize],
                            HCB_8_BITS[(a * 8 + b) as usize],
                        ),
                        9 => (
                            HCB_9_CODES[(a * 13 + b) as usize],
                            HCB_9_BITS[(a * 13 + b) as usize],
                        ),
                        10 => (
                            HCB_10_CODES[(a * 13 + b) as usize],
                            HCB_10_BITS[(a * 13 + b) as usize],
                        ),
                        _ => {
                            let i = (a.min(16) * 17 + b.min(16)) as usize;
                            (HCB_11_CODES[i], HCB_11_BITS[i])
                        }
                    };
                    bw.write_bits(code as u64, len as u32);
                    write_signs(bw, t);
                    if cb == 11 {
                        for v in [a, b] {
                            write_escape(bw, v);
                        }
                    }
                }
            }
        }
    }
}

fn write_signs(bw: &mut BitWriter, t: &[i32]) {
    for &v in t {
        if v != 0 {
            bw.write_bit(v < 0);
        }
    }
}

fn write_escape(bw: &mut BitWriter, a: u32) {
    if a < 16 {
        return;
    }
    let n = 31 - a.leading_zeros() - 4;
    bw.write_bits((1u64 << n) - 1, n); // N ones
    bw.write_bits(0, 1);
    bw.write_bits((a - (1 << (n + 4))) as u64, n + 4);
}

/// Writes fill elements (`ID_FIL`, `EXT_FILL`) of at least `bits` bits.
fn write_fill(bw: &mut BitWriter, mut bits: usize) {
    while bits > 0 {
        // ID_FIL (3) + count (4) [+ esc_count (8)] + count bytes: one byte of extension_type
        // EXT_FILL and fill_nibble, then fill bytes.
        let bytes = bits.saturating_sub(7).div_ceil(8).clamp(1, 14 + 255);
        bw.write_bits(6, 3);
        if bytes < 15 {
            bw.write_bits(bytes as u64, 4);
        } else {
            bw.write_bits(15, 4);
            bw.write_bits((bytes - 15 + 1) as u64, 8);
        }
        bw.write_bits(0, 8); // extension_type EXT_FILL, fill_nibble 0000
        for _ in 1..bytes {
            bw.write_bits(0xA5, 8);
        }
        bits = bits.saturating_sub(7 + if bytes < 15 { 0 } else { 8 } + 8 * bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aac::AacDecoder;
    use std::f64::consts::TAU;

    const SR: u32 = 48_000;

    /// Three seconds of stereo test signals at 48 kHz, interleaved `f32`.
    fn signals(n: usize) -> Vec<(&'static str, Vec<f32>)> {
        let mut seed = 0x0AAC_0082_u64;
        let mut noise = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let t = |i: usize| i as f64 / SR as f64;
        // BUG-0082's reproduction: tones and a chirp (35 kbit/s for a 128 kbit/s request).
        let synth = (0..n)
            .flat_map(|i| {
                let t = t(i);
                let l = 0.3 * (TAU * 440.0 * t).sin()
                    + 0.2 * (TAU * (200.0 + 1500.0 * t) * t).sin()
                    + 0.05 * (TAU * 9000.0 * t).sin();
                [l as f32, (0.3 * (TAU * 660.0 * t).sin()) as f32]
            })
            .collect();
        // Music: plucked harmonic notes panned apart, over a little noise.
        let notes = [220.0, 277.18, 329.63, 440.0, 554.37];
        let music = (0..n)
            .flat_map(|i| {
                let t = t(i);
                let (mut l, mut r) = (0.0, 0.0);
                for (j, &f) in notes.iter().enumerate() {
                    let env = (-3.0 * (t - j as f64 * 0.37).rem_euclid(1.5)).exp();
                    let v: f64 = (1..8)
                        .map(|h| (TAU * f * h as f64 * t).sin() / h as f64)
                        .sum::<f64>()
                        * env
                        * 0.08;
                    l += v * (0.4 + 0.15 * j as f64);
                    r += v * (1.0 - 0.15 * j as f64);
                }
                let e = noise() * 0.003;
                [(l + e) as f32, (r - e) as f32]
            })
            .collect();
        // Speech-like: a buzzy pulse train of drifting pitch, gated into syllables and pauses.
        let mut phase = 0.0f64;
        let speech = (0..n)
            .flat_map(|i| {
                let t = t(i);
                phase = (phase + (120.0 + 30.0 * (TAU * 0.8 * t).sin()) / SR as f64).fract();
                let gate =
                    ((TAU * 3.0 * t).sin() * 0.5 + 0.5) * f64::from((TAU * 0.5 * t).sin() > -0.5);
                let v = ((2.0 * phase - 1.0).powi(5) * 0.6 + noise() * 0.02) * gate;
                [v as f32, (v * 0.7) as f32]
            })
            .collect();
        vec![("synthetic", synth), ("music", music), ("speech", speech)]
    }

    /// Encodes `pcm` as ADTS, checks every frame against the 6144-bit-per-channel buffer and the
    /// Ragdoll decoder, and returns `(kbit/s, SNR in dB)`.
    fn encode(pcm: &[f32], bitrate: u32) -> (f64, f64, Vec<u8>) {
        let mut enc = AacEncoder::new(SR, 2, bitrate);
        let mut dec = AacDecoder::new(SR, 2);
        let (mut stream, mut out) = (Vec::new(), Vec::new());
        for chunk in pcm.chunks(2048) {
            let mut frame = AudioFrame::new(SR, 2, SampleFormat::F32le, 1024);
            frame.as_f32_mut_slice()[..chunk.len()].copy_from_slice(chunk);
            let pkt = enc.encode_adts_frame(&frame).unwrap();
            assert!(
                pkt.len() * 8 <= 2 * MAX_CHANNEL_BITS,
                "{} bits",
                pkt.len() * 8
            );
            out.extend_from_slice(dec.decode_packet(&pkt).unwrap().as_f32_slice());
            stream.extend_from_slice(&pkt);
        }
        let kbps = stream.len() as f64 * 8.0 / (pcm.len() / 2) as f64 * SR as f64 / 1000.0;
        // The decoder runs one frame (1024 samples) behind; skip the first and last frames.
        let (mut sig, mut err) = (0.0f64, 0.0f64);
        for i in 2048..pcm.len() - 2048 {
            let (x, y) = (pcm[i] as f64, out[i + 2048] as f64);
            sig += x * x;
            err += (x - y) * (x - y);
        }
        (kbps, 10.0 * (sig / err).log10(), stream)
    }

    /// BUG-0082: the average bitrate follows the target from 32 to 320 kbit/s on synthetic,
    /// music and speech signals (it used to follow the signal: 35 kbit/s for 128 kbit/s), no
    /// frame exceeds the decoder's buffer, quality rises with the bitrate, and FFmpeg decodes it.
    #[test]
    fn bitrate_tracks_target() {
        let n = 3 * SR as usize;
        for (name, pcm) in signals(n) {
            let mut last_snr = f64::NEG_INFINITY;
            for kbps in [32u32, 64, 128, 192, 320] {
                let (got, snr, stream) = encode(&pcm, kbps * 1000);
                eprintln!("{name} {kbps} kbit/s: {got:.1} kbit/s, SNR {snr:.1} dB");
                assert!(
                    (got / kbps as f64 - 1.0).abs() <= 0.1,
                    "{name}: {got:.1} kbit/s for {kbps} kbit/s"
                );
                assert!(snr > last_snr, "{name} {kbps} kbit/s: SNR {snr:.1} dB");
                last_snr = snr;
                if kbps == 128 {
                    assert!(snr > 12.0, "{name}: SNR {snr:.1} dB at 128 kbit/s");
                    let dir = std::env::temp_dir();
                    let tag = format!("ragdoll_bug0082_{}_{name}", std::process::id());
                    let (aac, raw) = (
                        dir.join(format!("{tag}.aac")),
                        dir.join(format!("{tag}.raw")),
                    );
                    std::fs::write(&aac, &stream).unwrap();
                    let status = std::process::Command::new("ffmpeg")
                        .args(["-y", "-v", "error", "-i"])
                        .arg(&aac)
                        .args(["-f", "f32le"])
                        .arg(&raw)
                        .status();
                    if ragdoll_core::testing::tool_succeeded(status, "ffmpeg AAC decode") {
                        let len = std::fs::metadata(&raw).unwrap().len() as usize;
                        assert!(
                            len >= (n - 1024) * 2 * 4,
                            "{name}: FFmpeg decoded {len} bytes"
                        );
                    }
                    let _ = std::fs::remove_file(aac);
                    let _ = std::fs::remove_file(raw);
                }
            }
        }
    }
}
