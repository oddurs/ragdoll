//! AAC-LC decoder written from ISO/IEC 14496-3 subpart 4 (general audio coding): the
//! `raw_data_block` syntax (SCE, CPE, LFE, DSE, PCE, FIL, END), noiseless decoding, inverse
//! quantisation, pulse data, M/S and intensity stereo, perceptual noise substitution, temporal
//! noise shaping, and the filterbank with all four window sequences and both window shapes.
//! Arithmetic is `f64`; output is `F32le`, scaled to ±1.0 like FFmpeg's float decoder.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::OnceLock;

use ragdoll_core::{AudioFrame, BitReader, ChannelLayout, RagdollError, Result, SampleFormat};

use super::*;

const ONLY_LONG: u8 = 0;
const LONG_START: u8 = 1;
const EIGHT_SHORT: u8 = 2;
const LONG_STOP: u8 = 3;
const ZERO_HCB: u8 = 0;
const ESC_HCB: u8 = 11;
const NOISE_HCB: u8 = 13;
const INTENSITY_HCB2: u8 = 14;
const INTENSITY_HCB: u8 = 15;

/// Scale-factor band offsets of long windows at 64 kHz (47 bands).
const SWB_OFFSET_1024_64: [u16; 48] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 100, 112, 124, 140,
    156, 172, 192, 216, 240, 268, 304, 344, 384, 424, 464, 504, 544, 584, 624, 664, 704, 744, 784,
    824, 864, 904, 944, 984, 1024,
];
/// Scale-factor band offsets of long windows at 8 kHz (40 bands).
const SWB_OFFSET_1024_8: [u16; 41] = [
    0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 172, 188, 204, 220, 236, 252, 268,
    288, 308, 328, 348, 372, 396, 420, 448, 476, 508, 544, 580, 620, 664, 712, 764, 820, 880, 944,
    1024,
];
/// Scale-factor band offsets of short windows at 16, 12 and 11.025 kHz (15 bands).
const SWB_OFFSET_128_16: [u16; 16] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 60, 72, 88, 108, 128,
];
/// Scale-factor band offsets of short windows at 8 kHz (15 bands).
const SWB_OFFSET_128_8: [u16; 16] = [
    0, 4, 8, 12, 16, 20, 24, 28, 36, 44, 52, 60, 72, 88, 108, 128,
];

/// `TNS_MAX_BANDS` for AAC-LC long and short windows, by sampling frequency index.
const TNS_MAX_BANDS_LONG: [usize; 13] = [31, 31, 34, 40, 42, 51, 46, 46, 42, 42, 42, 39, 39];
const TNS_MAX_BANDS_SHORT: [usize; 13] = [9, 9, 10, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14];

fn swb_long(sr_idx: u8) -> &'static [u16] {
    match sr_idx {
        0 | 1 => &SWB_OFFSET_1024_96,
        2 => &SWB_OFFSET_1024_64,
        3 | 4 => &SWB_OFFSET_1024_48,
        5 => &SWB_OFFSET_1024_32,
        6 | 7 => &SWB_OFFSET_1024_24,
        8..=10 => &SWB_OFFSET_1024_16,
        _ => &SWB_OFFSET_1024_8,
    }
}

fn swb_short(sr_idx: u8) -> &'static [u16] {
    match sr_idx {
        0..=2 => &SWB_OFFSET_128_96,
        3..=5 => &SWB_OFFSET_128_48,
        6 | 7 => &SWB_OFFSET_128_24,
        8..=10 => &SWB_OFFSET_128_16,
        _ => &SWB_OFFSET_128_8,
    }
}

/// A Huffman codebook as a `(length, code) -> symbol` map.
struct Codebook {
    map: HashMap<u32, u16>,
    max_len: u32,
}

impl Codebook {
    fn new(codes: &[u32], bits: &[u8]) -> Self {
        let mut map = HashMap::with_capacity(codes.len());
        let mut max_len = 0;
        for (i, (&c, &b)) in codes.iter().zip(bits).enumerate() {
            map.insert(((b as u32) << 24) | c, i as u16);
            max_len = max_len.max(b as u32);
        }
        Codebook { map, max_len }
    }

    fn decode(&self, br: &mut BitReader) -> Result<usize> {
        let mut code = 0u32;
        for len in 1..=self.max_len {
            code = (code << 1) | br.read_bits(1)? as u32;
            if let Some(&s) = self.map.get(&((len << 24) | code)) {
                return Ok(s as usize);
            }
        }
        Err(RagdollError::InvalidData(
            "invalid AAC Huffman codeword".into(),
        ))
    }
}

/// Codebook 0 is the scale-factor codebook; 1 to 11 the spectral codebooks.
fn codebooks() -> &'static [Codebook; 12] {
    static BOOKS: OnceLock<[Codebook; 12]> = OnceLock::new();
    BOOKS.get_or_init(|| {
        let w = |c: &[u16]| c.iter().map(|&v| v as u32).collect::<Vec<u32>>();
        [
            Codebook::new(&HCB_SF_CODES, &HCB_SF_BITS),
            Codebook::new(&w(&HCB_1_CODES), &HCB_1_BITS),
            Codebook::new(&w(&HCB_2_CODES), &HCB_2_BITS),
            Codebook::new(&w(&HCB_3_CODES), &HCB_3_BITS),
            Codebook::new(&w(&HCB_4_CODES), &HCB_4_BITS),
            Codebook::new(&w(&HCB_5_CODES), &HCB_5_BITS),
            Codebook::new(&w(&HCB_6_CODES), &HCB_6_BITS),
            Codebook::new(&w(&HCB_7_CODES), &HCB_7_BITS),
            Codebook::new(&w(&HCB_8_CODES), &HCB_8_BITS),
            Codebook::new(&w(&HCB_9_CODES), &HCB_9_BITS),
            Codebook::new(&w(&HCB_10_CODES), &HCB_10_BITS),
            Codebook::new(&w(&HCB_11_CODES), &HCB_11_BITS),
        ]
    })
}

fn bad(msg: &str) -> RagdollError {
    RagdollError::InvalidData(format!("AAC: {msg}"))
}

#[derive(Clone, Default)]
struct IcsInfo {
    window_sequence: u8,
    window_shape: u8,
    max_sfb: usize,
    num_windows: usize,
    num_window_groups: usize,
    group_len: [usize; 8],
}

impl IcsInfo {
    fn parse(br: &mut BitReader, sr_idx: u8) -> Result<Self> {
        if br.read_bits(1)? != 0 {
            return Err(bad("ics_reserved_bit set"));
        }
        let window_sequence = br.read_bits(2)? as u8;
        let window_shape = br.read_bits(1)? as u8;
        let mut ics = IcsInfo {
            window_sequence,
            window_shape,
            ..Default::default()
        };
        if window_sequence == EIGHT_SHORT {
            ics.max_sfb = br.read_bits(4)? as usize;
            let grouping = br.read_bits(7)? as u32;
            ics.num_windows = 8;
            ics.num_window_groups = 1;
            ics.group_len[0] = 1;
            for i in 0..7 {
                if grouping & (1 << (6 - i)) != 0 {
                    ics.group_len[ics.num_window_groups - 1] += 1;
                } else {
                    ics.num_window_groups += 1;
                    ics.group_len[ics.num_window_groups - 1] = 1;
                }
            }
            if ics.max_sfb > swb_short(sr_idx).len() - 1 {
                return Err(bad("max_sfb too large"));
            }
        } else {
            ics.max_sfb = br.read_bits(6)? as usize;
            ics.num_windows = 1;
            ics.num_window_groups = 1;
            ics.group_len[0] = 1;
            if br.read_bits(1)? != 0 {
                return Err(RagdollError::Unsupported(
                    "AAC prediction (Main/LTP profile) is not supported".into(),
                ));
            }
            if ics.max_sfb > swb_long(sr_idx).len() - 1 {
                return Err(bad("max_sfb too large"));
            }
        }
        Ok(ics)
    }

    fn swb(&self, sr_idx: u8) -> &'static [u16] {
        if self.window_sequence == EIGHT_SHORT {
            swb_short(sr_idx)
        } else {
            swb_long(sr_idx)
        }
    }
}

#[derive(Clone, Default)]
struct TnsFilter {
    length: usize,
    order: usize,
    direction: bool,
    coef: [f64; 32],
}

#[derive(Clone)]
struct Channel {
    ics: IcsInfo,
    band_type: [[u8; 64]; 8],
    sf: [[i32; 64]; 8],
    tns: Vec<Vec<TnsFilter>>,
    /// Dequantised spectrum in window order (`window * 128 + bin` for short windows).
    spec: [f64; 1024],
}

impl Default for Channel {
    fn default() -> Self {
        Channel {
            ics: IcsInfo::default(),
            band_type: [[0; 64]; 8],
            sf: [[0; 64]; 8],
            tns: Vec::new(),
            spec: [0.0; 1024],
        }
    }
}

/// Per-output-channel filterbank state.
#[derive(Clone)]
struct Synth {
    overlap: [f64; 1024],
    prev_shape: u8,
}

impl Default for Synth {
    fn default() -> Self {
        Synth {
            overlap: [0.0; 1024],
            prev_shape: 0,
        }
    }
}

/// Window halves (rising part, `N/2` samples) for the long and short sine and KBD windows.
struct Windows {
    long: [Vec<f64>; 2],
    short: [Vec<f64>; 2],
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let q = x * x / 4.0;
    for k in 1..100 {
        term *= q / (k * k) as f64;
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

fn sine_half(n: usize) -> Vec<f64> {
    (0..n / 2)
        .map(|i| (PI / n as f64 * (i as f64 + 0.5)).sin())
        .collect()
}

fn kbd_half(n: usize, alpha: f64) -> Vec<f64> {
    let half = n / 2;
    let w: Vec<f64> = (0..=half)
        .map(|k| {
            let r = (k as f64 - half as f64 / 2.0) / (half as f64 / 2.0);
            bessel_i0(PI * alpha * (1.0 - r * r).max(0.0).sqrt())
        })
        .collect();
    let total: f64 = w.iter().sum();
    let mut acc = 0.0;
    (0..half)
        .map(|k| {
            acc += w[k];
            (acc / total).sqrt()
        })
        .collect()
}

fn windows() -> &'static Windows {
    static W: OnceLock<Windows> = OnceLock::new();
    W.get_or_init(|| Windows {
        long: [sine_half(2048), kbd_half(2048, 4.0)],
        short: [sine_half(256), kbd_half(256, 6.0)],
    })
}

/// IMDCT of `m` coefficients into `2m` samples:
/// `x[n] = 2/N * sum_k X[k] cos(2*pi/N * (n + n0) * (k + 1/2))`, `N = 2m`, `n0 = (m + 1) / 2`.
fn imdct(spec: &[f64], out: &mut [f64]) {
    let m = spec.len();
    let y = dct4(spec);
    let scale = 1.0 / m as f64;
    let h = m / 2;
    for n in 0..h {
        out[n] = y[n + h] * scale;
    }
    for n in h..3 * h {
        out[n] = -y[3 * h - 1 - n] * scale;
    }
    for n in 3 * h..2 * m {
        out[n] = -y[n - 3 * h] * scale;
    }
}

/// `y[n] = sum_k x[k] cos(pi/M (n + 1/2)(k + 1/2))` through an `M/2`-point complex FFT.
fn dct4(x: &[f64]) -> Vec<f64> {
    let m = x.len();
    let h = m / 2;
    let mut re = vec![0.0; h];
    let mut im = vec![0.0; h];
    for k in 0..h {
        let (a, b) = (x[2 * k], x[m - 1 - 2 * k]);
        let t = -PI * (4 * k + 1) as f64 / (4 * m) as f64;
        let (s, c) = t.sin_cos();
        re[k] = a * c - b * s;
        im[k] = a * s + b * c;
    }
    fft(&mut re, &mut im);
    let mut y = vec![0.0; m];
    for j in 0..h {
        let t = -PI * j as f64 / m as f64;
        let (s, c) = t.sin_cos();
        let ur = re[j] * c - im[j] * s;
        let ui = re[j] * s + im[j] * c;
        y[2 * j] = ur;
        y[m - 1 - 2 * j] = -ui;
    }
    y
}

/// In-place radix-2 forward FFT (`exp(-2*pi*i*jk/n)`).
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * PI / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (s, c) = (ang * k as f64).sin_cos();
                let (a, b) = (start + k, start + k + len / 2);
                let tr = re[b] * c - im[b] * s;
                let ti = re[b] * s + im[b] * c;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len <<= 1;
    }
}

/// ISO/IEC 14496-3 AAC-LC decoder (1024 samples per frame, up to 8 channels).
#[derive(Clone)]
pub struct AacDecoder {
    pub asc: AudioSpecificConfig,
    synth: Vec<Synth>,
    noise_state: u32,
}

impl std::fmt::Debug for AacDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AacDecoder")
            .field("asc", &self.asc)
            .finish_non_exhaustive()
    }
}

impl AacDecoder {
    /// Creates a decoder for `sample_rate` and `channels` (the ADTS header or
    /// `AudioSpecificConfig` of the stream overrides both).
    pub fn new(sample_rate: u32, channels: u8) -> Self {
        Self::from_asc(AudioSpecificConfig::new_aac_lc(sample_rate, channels))
    }

    /// Creates a decoder from a parsed [`AudioSpecificConfig`].
    pub fn from_asc(asc: AudioSpecificConfig) -> Self {
        Self {
            asc,
            synth: Vec::new(),
            noise_state: 0x1f2e_3d4c,
        }
    }

    /// Configures the decoder from an `AudioSpecificConfig` (`esds` / `CodecPrivate`).
    pub fn with_extra_data(mut self, extra_data: &[u8]) -> Result<Self> {
        if extra_data.len() >= 2 {
            self.asc = AudioSpecificConfig::parse(extra_data)?;
        }
        Ok(self)
    }

    /// Decodes one AAC access unit (raw, or with an ADTS header) into 1024 samples per channel.
    pub fn decode_packet(&mut self, pkt: &[u8]) -> Result<AudioFrame> {
        if pkt.is_empty() {
            return Err(RagdollError::BitstreamEndOfData);
        }
        let payload = if pkt.len() >= 7 && pkt[0] == 0xFF && (pkt[1] & 0xF6) == 0xF0 {
            let adts = AdtsHeader::parse(pkt)?;
            if adts.sampling_frequency_index as usize >= AAC_SAMPLE_RATES.len() {
                return Err(bad("bad sampling frequency index"));
            }
            self.asc = AudioSpecificConfig {
                audio_object_type: adts.profile + 1,
                sampling_frequency_index: adts.sampling_frequency_index,
                sample_rate: AAC_SAMPLE_RATES[adts.sampling_frequency_index as usize],
                channel_configuration: adts.channel_configuration,
            };
            let end = (adts.frame_length as usize).min(pkt.len());
            &pkt[adts.header_len.min(end)..end]
        } else {
            pkt
        };
        if self.asc.sampling_frequency_index as usize >= AAC_SAMPLE_RATES.len() {
            return Err(bad("bad sampling frequency index"));
        }
        let (channels, elements) = self.decode_raw(payload)?;
        let n = channels.len();
        // The default arrangements of channel configurations 3 to 7, reordered from bitstream
        // order (C, L, R, ...) to FL, FR, FC, LFE, BL, BR, SL, SR as FFmpeg and WAV use.
        let order: Vec<usize> = match elements.as_str() {
            "SC" => vec![1, 2, 0],
            "SCS" => vec![1, 2, 0, 3],
            "SCC" => vec![1, 2, 0, 3, 4],
            "SCCL" => vec![1, 2, 0, 5, 3, 4],
            "SCCCL" => vec![1, 2, 0, 7, 5, 6, 3, 4],
            _ => (0..n).collect(),
        };
        let mut frame = AudioFrame::new(self.asc.sample_rate, n as u8, SampleFormat::F32le, 1024);
        frame.layout = ChannelLayout::from_channels(n as u8);
        let dst = frame.as_f32_mut_slice();
        for (o, &src) in order.iter().enumerate() {
            for i in 0..1024 {
                dst[i * n + o] = (channels[src][i] / 32768.0) as f32;
            }
        }
        Ok(frame)
    }

    /// Decodes a `raw_data_block` into PCM channels in bitstream order (16-bit scale), with the
    /// sequence of audio elements (`S`CE, `C`PE, `L`FE).
    fn decode_raw(&mut self, payload: &[u8]) -> Result<(Vec<Vec<f64>>, String)> {
        let sr_idx = self.asc.sampling_frequency_index;
        let mut br = BitReader::new(payload);
        let mut out: Vec<Vec<f64>> = Vec::new();
        let mut elements = String::new();
        loop {
            let id = br.read_bits(3)? as u8;
            match id {
                0 | 3 => {
                    // SCE, LFE
                    let _tag = br.read_bits(4)?;
                    let mut ch = Channel::default();
                    self.decode_ics(&mut br, &mut ch, None, sr_idx)?;
                    apply_tns(&mut ch, sr_idx, self.asc.audio_object_type);
                    let idx = out.len();
                    out.push(self.synthesize(idx, &ch));
                    elements.push(if id == 0 { 'S' } else { 'L' });
                }
                1 => {
                    // CPE
                    let _tag = br.read_bits(4)?;
                    let common_window = br.read_bits(1)? != 0;
                    let mut l = Channel::default();
                    let mut r = Channel::default();
                    let mut ms_used = [[false; 64]; 8];
                    if common_window {
                        let ics = IcsInfo::parse(&mut br, sr_idx)?;
                        let ms_mask_present = br.read_bits(2)?;
                        if ms_mask_present == 3 {
                            return Err(bad("reserved ms_mask_present"));
                        }
                        if ms_mask_present == 1 {
                            for g in 0..ics.num_window_groups {
                                for sfb in 0..ics.max_sfb {
                                    ms_used[g][sfb] = br.read_bits(1)? != 0;
                                }
                            }
                        } else if ms_mask_present == 2 {
                            for row in ms_used.iter_mut() {
                                row.fill(true);
                            }
                        }
                        l.ics = ics.clone();
                        r.ics = ics;
                    }
                    let shared = common_window.then(|| l.ics.clone());
                    self.decode_ics(&mut br, &mut l, shared.as_ref(), sr_idx)?;
                    self.decode_ics(&mut br, &mut r, shared.as_ref(), sr_idx)?;
                    if common_window {
                        stereo_tools(&mut l, &mut r, &ms_used, sr_idx);
                    }
                    apply_tns(&mut l, sr_idx, self.asc.audio_object_type);
                    apply_tns(&mut r, sr_idx, self.asc.audio_object_type);
                    let idx = out.len();
                    out.push(self.synthesize(idx, &l));
                    out.push(self.synthesize(idx + 1, &r));
                    elements.push('C');
                }
                2 => {
                    return Err(RagdollError::Unsupported(
                        "AAC coupling channel elements are not supported".into(),
                    ));
                }
                4 => {
                    // DSE
                    let _tag = br.read_bits(4)?;
                    let align = br.read_bits(1)? != 0;
                    let mut count = br.read_bits(8)? as usize;
                    if count == 255 {
                        count += br.read_bits(8)? as usize;
                    }
                    if align {
                        br.align_to_byte();
                    }
                    br.skip_bits(8 * count)?;
                }
                5 => skip_pce(&mut br)?,
                6 => {
                    // FIL: extension payloads (SBR, DRC, fill) are skipped.
                    let mut count = br.read_bits(4)? as usize;
                    if count == 15 {
                        count += br.read_bits(8)? as usize;
                        count -= 1;
                    }
                    br.skip_bits(8 * count)?;
                }
                _ => break,
            }
            if out.len() > 8 {
                return Err(bad("too many channels"));
            }
        }
        if out.is_empty() {
            return Err(bad("no audio channel elements"));
        }
        Ok((out, elements))
    }

    fn decode_ics(
        &mut self,
        br: &mut BitReader,
        ch: &mut Channel,
        common: Option<&IcsInfo>,
        sr_idx: u8,
    ) -> Result<()> {
        let global_gain = br.read_bits(8)? as i32;
        ch.ics = match common {
            Some(ics) => ics.clone(),
            None => IcsInfo::parse(br, sr_idx)?,
        };
        let ics = ch.ics.clone();
        let short = ics.window_sequence == EIGHT_SHORT;
        let swb = ics.swb(sr_idx);
        let books = codebooks();

        // section_data
        let (sect_bits, sect_esc) = if short { (3, 7) } else { (5, 31) };
        for g in 0..ics.num_window_groups {
            let mut k = 0;
            while k < ics.max_sfb {
                let cb = br.read_bits(4)? as u8;
                if cb == 12 {
                    return Err(bad("reserved codebook 12"));
                }
                let mut len = 0usize;
                loop {
                    let incr = br.read_bits(sect_bits)? as usize;
                    len += incr;
                    if incr != sect_esc {
                        break;
                    }
                }
                if k + len > ics.max_sfb {
                    return Err(bad("section past max_sfb"));
                }
                for sfb in k..k + len {
                    ch.band_type[g][sfb] = cb;
                }
                k += len;
            }
        }

        // scale_factor_data
        let mut sf = global_gain;
        let mut is_pos = 0i32;
        let mut noise_nrg = global_gain - 90;
        let mut noise_first = true;
        for g in 0..ics.num_window_groups {
            for sfb in 0..ics.max_sfb {
                match ch.band_type[g][sfb] {
                    ZERO_HCB => ch.sf[g][sfb] = 0,
                    INTENSITY_HCB | INTENSITY_HCB2 => {
                        is_pos += books[0].decode(br)? as i32 - 60;
                        ch.sf[g][sfb] = is_pos;
                    }
                    NOISE_HCB => {
                        if noise_first {
                            noise_first = false;
                            noise_nrg += br.read_bits(9)? as i32 - 256;
                        } else {
                            noise_nrg += books[0].decode(br)? as i32 - 60;
                        }
                        ch.sf[g][sfb] = noise_nrg;
                    }
                    _ => {
                        sf += books[0].decode(br)? as i32 - 60;
                        if !(0..=255).contains(&sf) {
                            return Err(bad("scalefactor out of range"));
                        }
                        ch.sf[g][sfb] = sf;
                    }
                }
            }
        }

        // pulse_data
        let mut pulses: Vec<(usize, i32)> = Vec::new();
        if br.read_bits(1)? != 0 {
            if short {
                return Err(bad("pulse data in a short block"));
            }
            let num_pulse = br.read_bits(2)? as usize + 1;
            let start_sfb = br.read_bits(6)? as usize;
            if start_sfb >= swb.len() - 1 {
                return Err(bad("pulse_start_sfb out of range"));
            }
            let mut k = swb[start_sfb] as usize;
            for _ in 0..num_pulse {
                k += br.read_bits(5)? as usize;
                let amp = br.read_bits(4)? as i32;
                if k >= 1024 {
                    return Err(bad("pulse position out of range"));
                }
                pulses.push((k, amp));
            }
        }

        // tns_data
        ch.tns = vec![Vec::new(); ics.num_windows];
        if br.read_bits(1)? != 0 {
            let (nf_bits, len_bits, ord_bits) = if short { (1, 4, 3) } else { (2, 6, 5) };
            for w in 0..ics.num_windows {
                let n_filt = br.read_bits(nf_bits)? as usize;
                if n_filt == 0 {
                    continue;
                }
                let coef_res = br.read_bits(1)? as u32 + 3;
                for _ in 0..n_filt {
                    let mut f = TnsFilter {
                        length: br.read_bits(len_bits)? as usize,
                        order: br.read_bits(ord_bits)? as usize,
                        ..Default::default()
                    };
                    if f.order > 0 {
                        f.direction = br.read_bits(1)? != 0;
                        let compress = br.read_bits(1)? as u32;
                        let bits = coef_res - compress;
                        let iqfac = ((1 << (coef_res - 1)) as f64 - 0.5) / (PI / 2.0);
                        let iqfac_m = ((1 << (coef_res - 1)) as f64 + 0.5) / (PI / 2.0);
                        let mut refl = [0.0f64; 32];
                        for r in refl.iter_mut().take(f.order) {
                            let raw = br.read_bits(bits)? as i32;
                            let v = (raw << (32 - bits)) >> (32 - bits);
                            *r = (v as f64 / if v >= 0 { iqfac } else { iqfac_m }).sin();
                        }
                        // Reflection coefficients to an LPC polynomial.
                        let mut a = [0.0f64; 33];
                        a[0] = 1.0;
                        for mm in 1..=f.order {
                            let mut b = a;
                            for i in 1..mm {
                                b[i] = a[i] + refl[mm - 1] * a[mm - i];
                            }
                            a[1..mm].copy_from_slice(&b[1..mm]);
                            a[mm] = refl[mm - 1];
                        }
                        f.coef[..f.order].copy_from_slice(&a[1..=f.order]);
                    }
                    ch.tns[w].push(f);
                }
            }
        }

        // gain_control_data
        if br.read_bits(1)? != 0 {
            return Err(RagdollError::Unsupported(
                "AAC SSR gain control is not supported".into(),
            ));
        }

        // spectral_data, into the grouped (interleaved) order, then dequantised per window.
        let mut q = [0i32; 1024];
        let mut win = 0usize;
        for g in 0..ics.num_window_groups {
            let glen = ics.group_len[g];
            for sfb in 0..ics.max_sfb {
                let cb = ch.band_type[g][sfb];
                let (lo, hi) = (swb[sfb] as usize, swb[sfb + 1] as usize);
                if cb == ZERO_HCB || cb >= NOISE_HCB {
                    continue;
                }
                let book = &books[cb as usize];
                for w in 0..glen {
                    let base = (win + w) * 128;
                    let mut k = lo;
                    while k < hi {
                        let s = book.decode(br)? as i32;
                        let vals: [i32; 4];
                        let dim;
                        match cb {
                            1 | 2 => {
                                vals = [s / 27 - 1, (s / 9) % 3 - 1, (s / 3) % 3 - 1, s % 3 - 1];
                                dim = 4;
                            }
                            3 | 4 => {
                                vals = [s / 27, (s / 9) % 3, (s / 3) % 3, s % 3];
                                dim = 4;
                            }
                            5 | 6 => {
                                vals = [s / 9 - 4, s % 9 - 4, 0, 0];
                                dim = 2;
                            }
                            7 | 8 => {
                                vals = [s / 8, s % 8, 0, 0];
                                dim = 2;
                            }
                            9 | 10 => {
                                vals = [s / 13, s % 13, 0, 0];
                                dim = 2;
                            }
                            _ => {
                                vals = [s / 17, s % 17, 0, 0];
                                dim = 2;
                            }
                        }
                        let mut v = vals;
                        let unsigned = matches!(cb, 3 | 4 | 7..=11);
                        if unsigned {
                            for x in v.iter_mut().take(dim) {
                                if *x != 0 && br.read_bits(1)? != 0 {
                                    *x = -*x;
                                }
                            }
                        }
                        if cb == ESC_HCB {
                            for x in v.iter_mut().take(2) {
                                if x.abs() == 16 {
                                    let mut n = 4;
                                    while br.read_bits(1)? != 0 {
                                        n += 1;
                                        if n > 12 {
                                            return Err(bad("escape sequence too long"));
                                        }
                                    }
                                    let mag = (1 << n) + br.read_bits(n)? as i32;
                                    *x = if *x < 0 { -mag } else { mag };
                                }
                            }
                        }
                        let off = if short { base + k } else { k };
                        q[off..off + dim].copy_from_slice(&v[..dim]);
                        k += dim;
                    }
                }
            }
            win += glen;
        }
        for &(k, amp) in &pulses {
            q[k] = if q[k] > 0 { q[k] + amp } else { q[k] - amp };
        }

        // Inverse quantisation and noise substitution.
        ch.spec = [0.0; 1024];
        let mut win = 0usize;
        for g in 0..ics.num_window_groups {
            for w in win..win + ics.group_len[g] {
                let base = if short { w * 128 } else { 0 };
                for sfb in 0..ics.max_sfb {
                    let (lo, hi) = (base + swb[sfb] as usize, base + swb[sfb + 1] as usize);
                    match ch.band_type[g][sfb] {
                        ZERO_HCB | INTENSITY_HCB | INTENSITY_HCB2 => {}
                        NOISE_HCB => {
                            let mut energy = 0.0;
                            for k in lo..hi {
                                self.noise_state = self
                                    .noise_state
                                    .wrapping_mul(1_664_525)
                                    .wrapping_add(1_013_904_223);
                                let r = self.noise_state as i32 as f64;
                                ch.spec[k] = r;
                                energy += r * r;
                            }
                            let scale = 2f64.powf(0.25 * ch.sf[g][sfb] as f64) / energy.sqrt();
                            for v in &mut ch.spec[lo..hi] {
                                *v *= scale;
                            }
                        }
                        _ => {
                            let gain = 2f64.powf(0.25 * (ch.sf[g][sfb] - 100) as f64);
                            for k in lo..hi {
                                let x = q[k];
                                if x != 0 {
                                    let m = (x.unsigned_abs() as f64).powf(4.0 / 3.0) * gain;
                                    ch.spec[k] = if x < 0 { -m } else { m };
                                }
                            }
                        }
                    }
                }
            }
            win += ics.group_len[g];
        }
        Ok(())
    }

    /// Filterbank: IMDCT, windowing and overlap-add for output channel `idx`.
    fn synthesize(&mut self, idx: usize, ch: &Channel) -> Vec<f64> {
        if self.synth.len() <= idx {
            self.synth.resize(idx + 1, Synth::default());
        }
        let w = windows();
        let st = &mut self.synth[idx];
        let cur = ch.ics.window_shape as usize;
        let prev = st.prev_shape as usize;
        let mut z = [0.0f64; 2048];
        match ch.ics.window_sequence {
            EIGHT_SHORT => {
                let mut x = [0.0f64; 256];
                for win in 0..8 {
                    imdct(&ch.spec[win * 128..win * 128 + 128], &mut x);
                    let left = if win == 0 {
                        &w.short[prev]
                    } else {
                        &w.short[cur]
                    };
                    let right = &w.short[cur];
                    let base = 448 + 128 * win;
                    for n in 0..128 {
                        z[base + n] += x[n] * left[n];
                        z[base + 128 + n] += x[128 + n] * right[127 - n];
                    }
                }
            }
            seq => {
                let mut x = [0.0f64; 2048];
                imdct(&ch.spec, &mut x);
                if seq == LONG_STOP {
                    for n in 448..576 {
                        z[n] = x[n] * w.short[prev][n - 448];
                    }
                    z[576..1024].copy_from_slice(&x[576..1024]);
                } else {
                    for n in 0..1024 {
                        z[n] = x[n] * w.long[prev][n];
                    }
                }
                if seq == LONG_START {
                    z[1024..1472].copy_from_slice(&x[1024..1472]);
                    for n in 1472..1600 {
                        z[n] = x[n] * w.short[cur][127 - (n - 1472)];
                    }
                } else {
                    for n in 1024..2048 {
                        z[n] = x[n] * w.long[cur][2047 - n];
                    }
                }
                debug_assert!(seq == ONLY_LONG || seq == LONG_START || seq == LONG_STOP);
            }
        }
        let mut out = vec![0.0f64; 1024];
        for n in 0..1024 {
            out[n] = z[n] + st.overlap[n];
        }
        st.overlap.copy_from_slice(&z[1024..]);
        st.prev_shape = ch.ics.window_shape;
        out
    }
}

/// M/S and intensity stereo for a common-window channel pair.
fn stereo_tools(l: &mut Channel, r: &mut Channel, ms_used: &[[bool; 64]; 8], sr_idx: u8) {
    let ics = l.ics.clone();
    let short = ics.window_sequence == EIGHT_SHORT;
    let swb = ics.swb(sr_idx);
    let mut win = 0usize;
    for g in 0..ics.num_window_groups {
        for w in win..win + ics.group_len[g] {
            let base = if short { w * 128 } else { 0 };
            for sfb in 0..ics.max_sfb {
                let (lo, hi) = (base + swb[sfb] as usize, base + swb[sfb + 1] as usize);
                let (bl, brt) = (l.band_type[g][sfb], r.band_type[g][sfb]);
                if brt == INTENSITY_HCB || brt == INTENSITY_HCB2 {
                    let mut sign = if brt == INTENSITY_HCB { 1.0 } else { -1.0 };
                    // FFmpeg inverts wherever M/S is signalled, `ms_mask_present` 2 included (its
                    // encoder writes that mode over intensity bands); the ISO/IEC 14496-3
                    // `invert_intensity()` reads `ms_used` only for `ms_mask_present` 1.
                    if ms_used[g][sfb] {
                        sign = -sign;
                    }
                    let scale = sign * 0.5f64.powf(0.25 * r.sf[g][sfb] as f64);
                    for k in lo..hi {
                        r.spec[k] = l.spec[k] * scale;
                    }
                } else if ms_used[g][sfb] && bl < NOISE_HCB && brt < NOISE_HCB {
                    for k in lo..hi {
                        let (m, s) = (l.spec[k], r.spec[k]);
                        l.spec[k] = m + s;
                        r.spec[k] = m - s;
                    }
                }
            }
        }
        win += ics.group_len[g];
    }
}

/// Temporal noise shaping: all-pole filtering of the spectrum per window.
fn apply_tns(ch: &mut Channel, sr_idx: u8, _aot: u8) {
    let short = ch.ics.window_sequence == EIGHT_SHORT;
    let swb = ch.ics.swb(sr_idx);
    let num_swb = swb.len() - 1;
    let (max_bands, max_order) = if short {
        (TNS_MAX_BANDS_SHORT[sr_idx as usize], 7)
    } else {
        (TNS_MAX_BANDS_LONG[sr_idx as usize], 12)
    };
    let limit = max_bands.min(ch.ics.max_sfb);
    for w in 0..ch.tns.len() {
        let base = if short { w * 128 } else { 0 };
        let mut bottom = num_swb;
        for f in &ch.tns[w] {
            let top = bottom;
            bottom = top.saturating_sub(f.length);
            let order = f.order.min(max_order);
            if order == 0 {
                continue;
            }
            let start = swb[bottom.min(limit)] as usize;
            let end = swb[top.min(limit)] as usize;
            if end <= start {
                continue;
            }
            let x = &mut ch.spec[base..base + 1024.min(if short { 128 } else { 1024 })];
            let mut state = [0.0f64; 32];
            let n = end - start;
            for i in 0..n {
                let k = if f.direction { end - 1 - i } else { start + i };
                let mut y = x[k];
                for j in 0..order {
                    y -= f.coef[j] * state[j];
                }
                for j in (1..order).rev() {
                    state[j] = state[j - 1];
                }
                state[0] = y;
                x[k] = y;
            }
        }
    }
}

/// Parses (and discards) a `program_config_element`.
fn skip_pce(br: &mut BitReader) -> Result<()> {
    let _tag = br.read_bits(4)?;
    let _object_type = br.read_bits(2)?;
    let _sf_index = br.read_bits(4)?;
    let front = br.read_bits(4)? as usize;
    let side = br.read_bits(4)? as usize;
    let back = br.read_bits(4)? as usize;
    let lfe = br.read_bits(2)? as usize;
    let assoc = br.read_bits(3)? as usize;
    let cc = br.read_bits(4)? as usize;
    if br.read_bits(1)? != 0 {
        br.read_bits(4)?;
    }
    if br.read_bits(1)? != 0 {
        br.read_bits(4)?;
    }
    if br.read_bits(1)? != 0 {
        br.read_bits(3)?;
    }
    br.skip_bits((front + side + back) * 5 + lfe * 4 + assoc * 4 + cc * 5)?;
    br.align_to_byte();
    let comment = br.read_bits(8)? as usize;
    br.skip_bits(8 * comment)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imdct_matches_definition() {
        for m in [64usize, 1024] {
            let spec: Vec<f64> = (0..m).map(|k| ((k * 7919) % 101) as f64 - 50.0).collect();
            let mut fast = vec![0.0; 2 * m];
            imdct(&spec, &mut fast);
            let n = 2 * m;
            let n0 = (m as f64 + 1.0) / 2.0;
            for (i, &v) in fast.iter().enumerate() {
                let direct: f64 = spec
                    .iter()
                    .enumerate()
                    .map(|(k, &x)| {
                        x * (2.0 * PI / n as f64 * (i as f64 + n0) * (k as f64 + 0.5)).cos()
                    })
                    .sum::<f64>()
                    * 2.0
                    / n as f64;
                assert!((v - direct).abs() < 1e-9, "m={m} n={i}: {v} vs {direct}");
            }
        }
    }
}
