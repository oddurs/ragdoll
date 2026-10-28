//! AAC-LC decoder written from ISO/IEC 14496-3 subpart 4 (general audio coding): the
//! `raw_data_block` syntax (SCE, CPE, LFE, DSE, PCE, FIL, END), noiseless decoding, inverse
//! quantisation, pulse data, M/S and intensity stereo, perceptual noise substitution, temporal
//! noise shaping, and the filterbank with all four window sequences and both window shapes.
//! Arithmetic is `f32` like FFmpeg's float decoder, and output is `F32le` scaled to ±1.0.
//!
//! Everything that does not depend on the stream is computed once ([`tables`]): two-level Huffman
//! lookup tables whose codewords carry their sign bits and yield rows of dequantised values,
//! `x^(4/3)` and `2^(sf/4)` tables, the windows, and the DCT-IV plans of the IMDCT
//! ([`Dct4Plan`] and [`imdct_window`], with AVX2 and NEON paths). The spectral data loop reads
//! through a register bit cache ([`Cache`]).

use std::f64::consts::PI;
use std::sync::OnceLock;

use ragdoll_core::{AudioFrame, ChannelLayout, RagdollError, Result, SampleFormat};
use ragdoll_simd::{Dct4Plan, imdct_window};

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

fn bad(msg: &str) -> RagdollError {
    RagdollError::InvalidData(format!("AAC: {msg}"))
}

/// Size of the payload buffer: a power of two above the largest AAC access unit (6144 bits per
/// channel), so that reads can wrap around it instead of being bounds-checked.
const BUF_LEN: usize = 1 << 14;

/// A payload copied to the start of a [`BUF_LEN`] buffer, followed by zeros and 8 more bytes for
/// the last load.
type Buf = [u8; BUF_LEN + 8];

/// MSB-first reader over a [`Buf`]. A read never fails: past the end of the payload it returns
/// padding (or, far past it, stale bytes), and the decoder checks [`Bits::check`] after every
/// element and in the one loop a corrupt stream could otherwise keep spinning (section data).
#[derive(Clone, Copy)]
struct Bits<'a> {
    buf: &'a Buf,
    pos: usize,
    end: usize,
}

impl<'a> Bits<'a> {
    fn new(buf: &'a Buf, len: usize) -> Self {
        Bits {
            buf,
            pos: 0,
            end: len * 8,
        }
    }

    /// The next 64 bits, left-aligned; at least 57 of them are stream bits (or padding).
    #[inline(always)]
    fn peek(&self) -> u64 {
        let i = (self.pos >> 3) & (BUF_LEN - 1);
        let w = u64::from_be_bytes(self.buf[i..i + 8].try_into().unwrap());
        w << (self.pos & 7)
    }

    /// Reads `n` bits, `1 <= n <= 32`.
    #[inline(always)]
    fn read(&mut self, n: u32) -> u32 {
        debug_assert!((1..=32).contains(&n));
        let v = (self.peek() >> (64 - n)) as u32;
        self.pos += n as usize;
        v
    }

    #[inline(always)]
    fn bit(&mut self) -> bool {
        self.read(1) != 0
    }

    fn skip(&mut self, n: usize) {
        self.pos = self.pos.saturating_add(n);
    }

    fn align(&mut self) {
        self.pos = (self.pos + 7) & !7;
    }

    fn check(&self) -> Result<()> {
        if self.pos > self.end {
            Err(RagdollError::BitstreamEndOfData)
        } else {
            Ok(())
        }
    }
}

/// What [`Vlc::decode`] and [`escape`] read: a window on the next bits, and consuming them.
trait BitSource {
    /// The next bits, left-aligned; at least 32 of them are valid.
    fn peek(&self) -> u64;
    /// Consumes `n <= 32` bits.
    fn consume(&mut self, n: u32);
}

impl BitSource for Bits<'_> {
    #[inline(always)]
    fn peek(&self) -> u64 {
        Bits::peek(self)
    }

    #[inline(always)]
    fn consume(&mut self, n: u32) {
        self.pos += n as usize;
    }
}

/// The spectral data reader: [`Bits`] with its next 32 to 64 bits held in a register. Each
/// codeword then costs a shift instead of a dependent load, and the 32-bit refills read from an
/// address that does not depend on the codewords.
struct Cache<'a> {
    buf: &'a Buf,
    /// `avail` unread bits, left-aligned.
    word: u64,
    avail: u32,
    /// Byte offset of the next refill.
    next: usize,
}

impl<'a> Cache<'a> {
    fn new(br: &Bits<'a>) -> Self {
        Cache {
            buf: br.buf,
            word: br.peek(),
            avail: 64 - (br.pos & 7) as u32,
            next: (br.pos >> 3) + 8,
        }
    }

    /// The bit position reached, for [`Bits`].
    fn pos(&self) -> usize {
        self.next * 8 - self.avail as usize
    }
}

impl BitSource for Cache<'_> {
    #[inline(always)]
    fn peek(&self) -> u64 {
        self.word
    }

    #[inline(always)]
    fn consume(&mut self, n: u32) {
        self.word <<= n;
        self.avail -= n;
        if self.avail < 32 {
            let i = self.next & (BUF_LEN - 1);
            let w = u32::from_be_bytes(self.buf[i..i + 4].try_into().unwrap());
            self.word |= (w as u64) << (32 - self.avail);
            self.avail += 32;
            self.next += 4;
        }
    }
}

/// Index bits of a Huffman table's first level.
const VLC_BITS: u32 = 9;

/// A two-level Huffman lookup table. An entry is `payload << 8 | length` for a codeword, or
/// `offset << 8 | 0x80 | bits` for a second-level table of `bits` index bits at `offset` in
/// `second`, where a codeword's length counts only the bits after the first [`VLC_BITS`].
struct Vlc {
    first: Box<[u32; 1 << VLC_BITS]>,
    second: Vec<u32>,
}

impl Vlc {
    /// Builds the table from `(code, length, payload)` triples of a complete prefix code.
    ///
    /// # Panics
    ///
    /// When the code is not complete, so that no bit pattern can miss the table.
    fn new(codes: &[(u32, u32, u32)]) -> Self {
        let kraft: u64 = codes.iter().map(|&(_, len, _)| 1u64 << (32 - len)).sum();
        assert_eq!(kraft, 1 << 32, "incomplete Huffman code");
        let mut first = Box::new([0u32; 1 << VLC_BITS]);
        let mut second = Vec::new();
        // Codewords longer than the first level, by their first VLC_BITS bits.
        let mut long: Vec<Vec<(u32, u32, u32)>> = vec![Vec::new(); 1 << VLC_BITS];
        for &(code, len, payload) in codes {
            if len <= VLC_BITS {
                let start = (code << (VLC_BITS - len)) as usize;
                first[start..start + (1 << (VLC_BITS - len))].fill(payload << 8 | len);
            } else {
                long[(code >> (len - VLC_BITS)) as usize].push((code, len - VLC_BITS, payload));
            }
        }
        for (prefix, codes) in long.iter().enumerate() {
            let Some(bits) = codes.iter().map(|&(_, rest, _)| rest).max() else {
                continue;
            };
            let offset = second.len();
            second.resize(offset + (1 << bits), 0);
            first[prefix] = (offset as u32) << 8 | 0x80 | bits;
            for &(code, rest, payload) in codes {
                let low = (code & ((1 << rest) - 1)) as usize;
                let start = offset + (low << (bits - rest));
                second[start..start + (1 << (bits - rest))].fill(payload << 8 | rest);
            }
        }
        Vlc { first, second }
    }

    /// Decodes one codeword and returns its payload. Every codebook is complete, so every bit
    /// pattern is some codeword.
    #[inline(always)]
    fn decode(&self, br: &mut impl BitSource) -> u32 {
        let p = br.peek();
        let mut e = self.first[(p >> (64 - VLC_BITS)) as usize];
        let mut used = 0;
        if e & 0x80 != 0 {
            let bits = e & 0x1F;
            e = self.second[(e >> 8) as usize + ((p << VLC_BITS) >> (64 - bits)) as usize];
            used = VLC_BITS;
        }
        br.consume(used + (e & 0x1F));
        e >> 8
    }
}

/// Largest `|q|` the `x^(4/3)` table covers: an escape value (8191) plus four pulses of 15.
const POW43_LEN: usize = 8192 + 64;

/// Payload flag of an escape-codebook codeword with a value of magnitude 16.
const ESC_FLAG: u32 = 1 << 12;

/// Payload of a 4-tuple: its row in [`Tables::quads`], `sum (v[i] + 2) 5^(3 - i)`.
fn quad_row(v: [i32; 4]) -> u32 {
    v.iter().fold(0, |row, &x| row * 5 + (x + 2) as u32)
}

/// Payload of a pair: its row in [`Tables::pairs`], `(v[0] + 16) 33 + v[1] + 16`.
fn pair_row(v: [i32; 2]) -> u32 {
    (v[0] + 16) as u32 * 33 + (v[1] + 16) as u32
}

/// The codewords of spectral codebook `cb` as `(code, length, payload)`, ready for [`Vlc::new`].
///
/// The sign bits that follow a codeword of the unsigned codebooks (3, 4 and 7 to 11, one per
/// nonzero value) become part of it, so that one lookup yields signed values. The payload is the
/// row of the values in [`Tables::quads`] (codebooks 1 to 4) or [`Tables::pairs`], plus
/// [`ESC_FLAG`] in the escape codebook.
fn spectral_codewords(cb: usize, codes: &[u16], bits: &[u8]) -> Vec<(u32, u32, u32)> {
    let mut out = Vec::new();
    for (s, (&code, &len)) in codes.iter().zip(bits).enumerate() {
        let s = s as i32;
        let (values, dim, signed) = match cb {
            1 | 2 => (
                [s / 27 - 1, s / 9 % 3 - 1, s / 3 % 3 - 1, s % 3 - 1],
                4,
                true,
            ),
            3 | 4 => ([s / 27, s / 9 % 3, s / 3 % 3, s % 3], 4, false),
            5 | 6 => ([s / 9 - 4, s % 9 - 4, 0, 0], 2, true),
            7 | 8 => ([s / 8, s % 8, 0, 0], 2, false),
            9 | 10 => ([s / 13, s % 13, 0, 0], 2, false),
            _ => ([s / 17, s % 17, 0, 0], 2, false),
        };
        let nz = if signed {
            0
        } else {
            values.iter().filter(|&&x| x != 0).count() as u32
        };
        for signs in 0..1u32 << nz {
            let mut bit = nz;
            let mut v = values;
            for x in v.iter_mut().filter(|x| !signed && **x != 0) {
                bit -= 1;
                if signs >> bit & 1 != 0 {
                    *x = -*x;
                }
            }
            let payload = if dim == 4 {
                quad_row(v)
            } else if v[0].abs() == 16 || v[1].abs() == 16 {
                pair_row([v[0], v[1]]) | ESC_FLAG
            } else {
                pair_row([v[0], v[1]])
            };
            out.push(((code as u32) << nz | signs, len as u32 + nz, payload));
        }
    }
    out
}

/// Window halves of one window shape over a whole long-window half, so that every long window
/// sequence is a multiply: the rising half for `ONLY_LONG`/`LONG_START` and for `LONG_STOP`, and
/// the falling half for `ONLY_LONG`/`LONG_STOP` and for `LONG_START`.
struct Shape {
    long_rise: [f32; 1024],
    stop_rise: [f32; 1024],
    long_fall: [f32; 1024],
    start_fall: [f32; 1024],
    short_rise: [f32; 128],
    short_fall: [f32; 128],
}

struct Tables {
    /// Codebook 0 is the scale-factor codebook (payload: index); 1 to 11 the spectral codebooks
    /// (payload: signed values, see [`spectral_codewords`]).
    books: [Vlc; 12],
    pow43: Vec<f32>,
    /// `sign(v) |v|^(4/3)` of every 4-tuple with `|v| <= 2` (rows of [`quad_row`]) and of every
    /// pair with `|v| <= 16` (rows of [`pair_row`]).
    quads: Box<[[f32; 4]; 1024]>,
    pairs: Box<[[f32; 2]; 2048]>,
    /// `2^((sf - 100) / 4)` for `sf` in `0..=255`.
    gain: [f32; 256],
    shapes: [Shape; 2],
    long: Dct4Plan,
    short: Dct4Plan,
}

impl Tables {
    /// `sign(x) |x|^(4/3) gain`.
    #[inline(always)]
    fn dequantise(&self, x: i32, gain: f32) -> f32 {
        let m = self.pow43[(x.unsigned_abs() as usize).min(POW43_LEN - 1)];
        f32::from_bits(m.to_bits() | (x as u32 & 0x8000_0000)) * gain
    }

    /// Adds a pulse of amplitude `amp` to the quantised value `q` behind `x = dequantise(q,
    /// gain)`: `q + amp` if `q > 0`, else `q - amp`. `q` comes back from `x`, exactly: `|q|` is at
    /// most 8251, and the two roundings to `f32` move `(|x| / gain)^(3/4)` by less than 0.002.
    fn add_pulse(&self, x: f32, amp: i32, gain: f32) -> f32 {
        let mag = (x.abs() as f64 / gain as f64).powf(0.75).round() as i32;
        let q = if x > 0.0 { mag + amp } else { -mag - amp };
        self.dequantise(q, gain)
    }
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

/// Rising half (`n / 2` samples) of the sine window of length `n`.
fn sine_half(n: usize) -> Vec<f64> {
    (0..n / 2)
        .map(|i| (PI / n as f64 * (i as f64 + 0.5)).sin())
        .collect()
}

/// Rising half (`n / 2` samples) of the Kaiser-Bessel-derived window of length `n`.
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

fn shape(long: &[f64], short: &[f64]) -> Shape {
    let mut s = Shape {
        long_rise: [0.0; 1024],
        stop_rise: [0.0; 1024],
        long_fall: [0.0; 1024],
        start_fall: [0.0; 1024],
        short_rise: [0.0; 128],
        short_fall: [0.0; 128],
    };
    for n in 0..1024 {
        s.long_rise[n] = long[n] as f32;
        s.long_fall[n] = long[1023 - n] as f32;
        s.stop_rise[n] = match n {
            0..448 => 0.0,
            448..576 => short[n - 448] as f32,
            _ => 1.0,
        };
        s.start_fall[n] = match n {
            0..448 => 1.0,
            448..576 => short[127 - (n - 448)] as f32,
            _ => 0.0,
        };
    }
    for n in 0..128 {
        s.short_rise[n] = short[n] as f32;
        s.short_fall[n] = short[127 - n] as f32;
    }
    s
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let sf: Vec<(u32, u32, u32)> = (0..HCB_SF_CODES.len())
            .map(|i| (HCB_SF_CODES[i], HCB_SF_BITS[i] as u32, i as u32))
            .collect();
        let spectral = |cb, codes: &[u16], bits| Vlc::new(&spectral_codewords(cb, codes, bits));
        let pow43: Vec<f32> = (0..POW43_LEN)
            .map(|i| (i as f64).powf(4.0 / 3.0) as f32)
            .collect();
        let signed = |v: i32| pow43[v.unsigned_abs() as usize].copysign(v as f32);
        let mut quads = Box::new([[0.0; 4]; 1024]);
        for (r, q) in quads.iter_mut().enumerate().take(625) {
            *q = [125, 25, 5, 1].map(|d| signed(r as i32 / d % 5 - 2));
        }
        let mut pairs = Box::new([[0.0; 2]; 2048]);
        for (r, p) in pairs.iter_mut().enumerate().take(33 * 33) {
            *p = [signed(r as i32 / 33 - 16), signed(r as i32 % 33 - 16)];
        }
        // 32768 maps the 16-bit scale of the spectrum to ±1.0; 1/M is the IMDCT's own gain.
        Tables {
            books: [
                Vlc::new(&sf),
                spectral(1, &HCB_1_CODES, &HCB_1_BITS),
                spectral(2, &HCB_2_CODES, &HCB_2_BITS),
                spectral(3, &HCB_3_CODES, &HCB_3_BITS),
                spectral(4, &HCB_4_CODES, &HCB_4_BITS),
                spectral(5, &HCB_5_CODES, &HCB_5_BITS),
                spectral(6, &HCB_6_CODES, &HCB_6_BITS),
                spectral(7, &HCB_7_CODES, &HCB_7_BITS),
                spectral(8, &HCB_8_CODES, &HCB_8_BITS),
                spectral(9, &HCB_9_CODES, &HCB_9_BITS),
                spectral(10, &HCB_10_CODES, &HCB_10_BITS),
                spectral(11, &HCB_11_CODES, &HCB_11_BITS),
            ],
            pow43,
            quads,
            pairs,
            gain: std::array::from_fn(|sf| (0.25 * (sf as f64 - 100.0)).exp2() as f32),
            shapes: [
                shape(&sine_half(2048), &sine_half(256)),
                shape(&kbd_half(2048, 4.0), &kbd_half(256, 6.0)),
            ],
            long: Dct4Plan::new(1024, 1.0 / (1024.0 * 32768.0)),
            short: Dct4Plan::new(128, 1.0 / (128.0 * 32768.0)),
        }
    })
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
    fn parse(br: &mut Bits, sr_idx: u8) -> Result<Self> {
        if br.bit() {
            return Err(bad("ics_reserved_bit set"));
        }
        let window_sequence = br.read(2) as u8;
        let window_shape = br.read(1) as u8;
        let mut ics = IcsInfo {
            window_sequence,
            window_shape,
            ..Default::default()
        };
        if window_sequence == EIGHT_SHORT {
            ics.max_sfb = br.read(4) as usize;
            let grouping = br.read(7);
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
            ics.max_sfb = br.read(6) as usize;
            ics.num_windows = 1;
            ics.num_window_groups = 1;
            ics.group_len[0] = 1;
            if br.bit() {
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

#[derive(Clone, Copy, Default)]
struct TnsFilter {
    length: usize,
    order: usize,
    direction: bool,
    coef: [f32; 32],
}

#[derive(Clone)]
struct Channel {
    ics: IcsInfo,
    band_type: [[u8; 64]; 8],
    sf: [[i32; 64]; 8],
    /// TNS filters per window (at most three in a long window, one in a short one).
    tns: [[TnsFilter; 3]; 8],
    tns_count: [usize; 8],
    /// Dequantised spectrum in window order (`window * 128 + bin` for short windows).
    spec: [f32; 1024],
}

impl Default for Channel {
    fn default() -> Self {
        Channel {
            ics: IcsInfo::default(),
            band_type: [[0; 64]; 8],
            sf: [[0; 64]; 8],
            tns: [[TnsFilter::default(); 3]; 8],
            tns_count: [0; 8],
            spec: [0.0; 1024],
        }
    }
}

/// Per-output-channel filterbank state.
#[derive(Clone)]
struct Synth {
    overlap: [f32; 1024],
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

/// ISO/IEC 14496-3 AAC-LC decoder (1024 samples per frame, up to 8 channels).
#[derive(Clone)]
pub struct AacDecoder {
    pub asc: AudioSpecificConfig,
    synth: Vec<Synth>,
    noise_state: u32,
    /// Scratch: the two channels of an element, a DCT-IV output, the padded payload and the
    /// output channels.
    chans: Box<[Channel; 2]>,
    y: Box<[f32; 1024]>,
    buf: Option<Box<Buf>>,
    pcm: Vec<[f32; 1024]>,
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
            chans: Box::default(),
            y: Box::new([0.0; 1024]),
            buf: None,
            pcm: Vec::new(),
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
        if payload.len() > BUF_LEN - 8 {
            return Err(bad("access unit too large"));
        }
        let mut buf = self
            .buf
            .take()
            .unwrap_or_else(|| Box::new([0; BUF_LEN + 8]));
        buf[..payload.len()].copy_from_slice(payload);
        buf[payload.len()..payload.len() + 8].fill(0);
        let decoded = self.decode_raw(&buf, payload.len());
        self.buf = Some(buf);
        let (elements, count, channels) = decoded?;
        // The default arrangements of channel configurations 3 to 7, reordered from bitstream
        // order (C, L, R, ...) to FL, FR, FC, LFE, BL, BR, SL, SR as FFmpeg and WAV use.
        let order: &[usize] = match &elements[..count] {
            b"SC" => &[1, 2, 0],
            b"SCS" => &[1, 2, 0, 3],
            b"SCC" => &[1, 2, 0, 3, 4],
            b"SCCL" => &[1, 2, 0, 5, 3, 4],
            b"SCCCL" => &[1, 2, 0, 7, 5, 6, 3, 4],
            _ => &[0, 1, 2, 3, 4, 5, 6, 7][..channels],
        };
        let n = order.len();
        let mut frame = AudioFrame::new(self.asc.sample_rate, n as u8, SampleFormat::F32le, 1024);
        frame.layout = ChannelLayout::from_channels(n as u8);
        let dst = frame.as_f32_mut_slice();
        if n == 2 {
            let (l, r) = (&self.pcm[0], &self.pcm[1]);
            for (i, d) in dst.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                *d = [l[i], r[i]];
            }
        } else {
            for (o, &src) in order.iter().enumerate() {
                for (i, &v) in self.pcm[src].iter().enumerate() {
                    dst[i * n + o] = v;
                }
            }
        }
        Ok(frame)
    }
}

impl AacDecoder {
    /// Decodes a `raw_data_block` from `buf` (`len` payload bytes, then zero padding) into
    /// `self.pcm` in bitstream order. Returns the sequence of audio elements (`S`CE, `C`PE, `L`FE),
    /// its length and the number of channels.
    fn decode_raw(&mut self, buf: &Buf, len: usize) -> Result<([u8; 8], usize, usize)> {
        let sr_idx = self.asc.sampling_frequency_index;
        let t = tables();
        let mut br = Bits::new(buf, len);
        let mut elements = [0u8; 8];
        let (mut count, mut channels) = (0, 0);
        loop {
            br.check()?;
            let id = br.read(3) as u8;
            let needed = match id {
                0 | 3 => 1,
                1 => 2,
                _ => 0,
            };
            if channels + needed > 8 {
                return Err(bad("too many channels"));
            }
            if self.pcm.len() < channels + needed {
                self.pcm.resize(channels + needed, [0.0; 1024]);
                self.synth.resize(channels + needed, Synth::default());
            }
            match id {
                0 | 3 => {
                    // SCE, LFE
                    let _tag = br.read(4);
                    let ch = &mut self.chans[0];
                    decode_ics(&mut br, ch, None, sr_idx, &mut self.noise_state, t)?;
                    br.check()?;
                    apply_tns(ch, sr_idx);
                    let (st, out) = (&mut self.synth[channels], &mut self.pcm[channels]);
                    synthesize(t, st, ch, out, &mut self.y);
                    elements[count] = if id == 0 { b'S' } else { b'L' };
                }
                1 => {
                    // CPE
                    let _tag = br.read(4);
                    let common_window = br.bit();
                    let mut ms_used = [[false; 64]; 8];
                    let [l, r] = &mut *self.chans;
                    let mut shared = None;
                    if common_window {
                        let ics = IcsInfo::parse(&mut br, sr_idx)?;
                        let ms_mask_present = br.read(2);
                        if ms_mask_present == 3 {
                            return Err(bad("reserved ms_mask_present"));
                        }
                        if ms_mask_present == 1 {
                            for g in 0..ics.num_window_groups {
                                for sfb in 0..ics.max_sfb {
                                    ms_used[g][sfb] = br.bit();
                                }
                            }
                        } else if ms_mask_present == 2 {
                            for row in ms_used.iter_mut() {
                                row.fill(true);
                            }
                        }
                        shared = Some(ics);
                    }
                    let noise = &mut self.noise_state;
                    decode_ics(&mut br, l, shared.as_ref(), sr_idx, noise, t)?;
                    decode_ics(&mut br, r, shared.as_ref(), sr_idx, noise, t)?;
                    br.check()?;
                    if common_window {
                        stereo_tools(l, r, &ms_used, sr_idx);
                    }
                    apply_tns(l, sr_idx);
                    apply_tns(r, sr_idx);
                    let (sl, sr) = self.synth[channels..].split_at_mut(1);
                    let (pl, pr) = self.pcm[channels..].split_at_mut(1);
                    synthesize(t, &mut sl[0], l, &mut pl[0], &mut self.y);
                    synthesize(t, &mut sr[0], r, &mut pr[0], &mut self.y);
                    elements[count] = b'C';
                }
                2 => {
                    return Err(RagdollError::Unsupported(
                        "AAC coupling channel elements are not supported".into(),
                    ));
                }
                4 => {
                    // DSE
                    let _tag = br.read(4);
                    let align = br.bit();
                    let mut count = br.read(8) as usize;
                    if count == 255 {
                        count += br.read(8) as usize;
                    }
                    if align {
                        br.align();
                    }
                    br.skip(8 * count);
                }
                5 => skip_pce(&mut br),
                6 => {
                    // FIL: extension payloads (SBR, DRC, fill) are skipped.
                    let mut count = br.read(4) as usize;
                    if count == 15 {
                        count += br.read(8) as usize;
                        count -= 1;
                    }
                    br.skip(8 * count);
                }
                _ => break,
            }
            if needed > 0 {
                channels += needed;
                count += 1;
            }
        }
        if channels == 0 {
            return Err(bad("no audio channel elements"));
        }
        Ok((elements, count, channels))
    }
}

/// Reads the escape sequence of an escape-codebook value of magnitude 16.
#[inline]
fn escape(br: &mut impl BitSource, x: i32) -> Result<i32> {
    let p = br.peek();
    let ones = (!p).leading_zeros();
    if ones > 8 {
        return Err(bad("escape sequence too long"));
    }
    let n = 4 + ones;
    let mag = (1 << n) + ((p << (ones + 1)) >> (64 - n)) as i32;
    br.consume(2 * ones + 5);
    Ok(if x < 0 { -mag } else { mag })
}

/// Parses an `individual_channel_stream` into the dequantised spectrum `ch.spec` (noise
/// substitution included, drawing from `noise_state`).
fn decode_ics(
    br: &mut Bits,
    ch: &mut Channel,
    common: Option<&IcsInfo>,
    sr_idx: u8,
    noise_state: &mut u32,
    t: &Tables,
) -> Result<()> {
    let global_gain = br.read(8) as i32;
    ch.ics = match common {
        Some(ics) => ics.clone(),
        None => IcsInfo::parse(br, sr_idx)?,
    };
    let ics = ch.ics.clone();
    let short = ics.window_sequence == EIGHT_SHORT;
    let swb = ics.swb(sr_idx);

    // section_data
    let (sect_bits, sect_esc) = if short { (3, 7) } else { (5, 31) };
    for g in 0..ics.num_window_groups {
        let mut k = 0;
        while k < ics.max_sfb {
            br.check()?;
            let cb = br.read(4) as u8;
            if cb == 12 {
                return Err(bad("reserved codebook 12"));
            }
            let mut len = 0usize;
            loop {
                let incr = br.read(sect_bits) as usize;
                len += incr;
                if incr != sect_esc {
                    break;
                }
                br.check()?;
            }
            if k + len > ics.max_sfb {
                return Err(bad("section past max_sfb"));
            }
            ch.band_type[g][k..k + len].fill(cb);
            k += len;
        }
    }

    // scale_factor_data
    let sf_book = &t.books[0];
    let mut sf = global_gain;
    let mut is_pos = 0i32;
    let mut noise_nrg = global_gain - 90;
    let mut noise_first = true;
    for g in 0..ics.num_window_groups {
        for sfb in 0..ics.max_sfb {
            ch.sf[g][sfb] = match ch.band_type[g][sfb] {
                ZERO_HCB => 0,
                INTENSITY_HCB | INTENSITY_HCB2 => {
                    is_pos += sf_book.decode(br) as i32 - 60;
                    is_pos
                }
                NOISE_HCB => {
                    if noise_first {
                        noise_first = false;
                        noise_nrg += br.read(9) as i32 - 256;
                    } else {
                        noise_nrg += sf_book.decode(br) as i32 - 60;
                    }
                    noise_nrg
                }
                _ => {
                    sf += sf_book.decode(br) as i32 - 60;
                    if !(0..=255).contains(&sf) {
                        return Err(bad("scalefactor out of range"));
                    }
                    sf
                }
            };
        }
    }

    // pulse_data
    let mut pulses = [(0usize, 0i32); 4];
    let mut num_pulse = 0;
    if br.bit() {
        if short {
            return Err(bad("pulse data in a short block"));
        }
        num_pulse = br.read(2) as usize + 1;
        let start_sfb = br.read(6) as usize;
        if start_sfb >= swb.len() - 1 {
            return Err(bad("pulse_start_sfb out of range"));
        }
        let mut k = swb[start_sfb] as usize;
        for p in pulses.iter_mut().take(num_pulse) {
            k += br.read(5) as usize;
            let amp = br.read(4) as i32;
            if k >= 1024 {
                return Err(bad("pulse position out of range"));
            }
            *p = (k, amp);
        }
    }

    // tns_data
    ch.tns_count = [0; 8];
    if br.bit() {
        let (nf_bits, len_bits, ord_bits) = if short { (1, 4, 3) } else { (2, 6, 5) };
        for w in 0..ics.num_windows {
            let n_filt = br.read(nf_bits) as usize;
            ch.tns_count[w] = n_filt;
            if n_filt == 0 {
                continue;
            }
            let coef_res = br.read(1) + 3;
            for f in ch.tns[w].iter_mut().take(n_filt) {
                f.length = br.read(len_bits) as usize;
                f.order = br.read(ord_bits) as usize;
                if f.order > 0 {
                    f.direction = br.bit();
                    let compress = br.read(1);
                    let bits = coef_res - compress;
                    let iqfac = ((1 << (coef_res - 1)) as f64 - 0.5) / (PI / 2.0);
                    let iqfac_m = ((1 << (coef_res - 1)) as f64 + 0.5) / (PI / 2.0);
                    let mut refl = [0.0f32; 32];
                    for r in refl.iter_mut().take(f.order) {
                        let raw = br.read(bits) as i32;
                        let v = (raw << (32 - bits)) >> (32 - bits);
                        *r = (v as f64 / if v >= 0 { iqfac } else { iqfac_m }).sin() as f32;
                    }
                    // Reflection coefficients to an LPC polynomial, in `f32` as FFmpeg does.
                    let a = &mut f.coef;
                    for mm in 0..f.order {
                        let prev = *a;
                        for i in 0..mm {
                            a[i] = prev[i] + refl[mm] * prev[mm - 1 - i];
                        }
                        a[mm] = refl[mm];
                    }
                }
            }
        }
    }

    // gain_control_data
    if br.bit() {
        return Err(RagdollError::Unsupported(
            "AAC SSR gain control is not supported".into(),
        ));
    }

    // spectral_data, from the grouped (interleaved) order into window order, dequantised on the
    // way: `sign(q) |q|^(4/3) 2^((sf - 100) / 4)`. Noise bands are filled afterwards.
    let spec = &mut ch.spec;
    let mut noise = false;
    let mut bits = Cache::new(br);
    let mut win = 0usize;
    for g in 0..ics.num_window_groups {
        let glen = ics.group_len[g];
        for sfb in 0..ics.max_sfb {
            let cb = ch.band_type[g][sfb];
            let (lo, hi) = (swb[sfb] as usize, swb[sfb + 1] as usize);
            if cb == ZERO_HCB || cb >= NOISE_HCB {
                noise |= cb == NOISE_HCB;
                for w in win..win + glen {
                    let base = if short { w * 128 } else { 0 };
                    spec[base + lo..base + hi].fill(0.0);
                }
                continue;
            }
            let book = &t.books[cb as usize];
            let gain = t.gain[ch.sf[g][sfb] as usize];
            for w in win..win + glen {
                let base = if short { w * 128 } else { 0 };
                let band = &mut spec[base + lo..base + hi];
                // Payloads index the value tables; the masks only spare the bounds checks.
                if cb <= 4 {
                    for s in band.as_chunks_mut::<4>().0 {
                        *s = t.quads[book.decode(&mut bits) as usize & 1023].map(|x| x * gain);
                    }
                } else if cb < ESC_HCB {
                    for s in band.as_chunks_mut::<2>().0 {
                        *s = t.pairs[book.decode(&mut bits) as usize & 2047].map(|x| x * gain);
                    }
                } else {
                    for s in band.as_chunks_mut::<2>().0 {
                        let p = book.decode(&mut bits);
                        if p & ESC_FLAG == 0 {
                            *s = t.pairs[p as usize & 2047].map(|x| x * gain);
                            continue;
                        }
                        let p = (p & !ESC_FLAG) as i32;
                        for (s, v) in s.iter_mut().zip([p / 33 - 16, p % 33 - 16]) {
                            let v = if v.abs() == 16 {
                                escape(&mut bits, v)?
                            } else {
                                v
                            };
                            *s = t.dequantise(v, gain);
                        }
                    }
                }
            }
        }
        win += glen;
    }
    br.pos = bits.pos();
    let top = swb[ics.max_sfb] as usize;
    for w in 0..ics.num_windows {
        let base = if short { w * 128 } else { 0 };
        let len = if short { 128 } else { 1024 };
        spec[base + top..base + len].fill(0.0);
    }

    // Pulses (long windows only) amend coefficients of spectrally coded bands.
    for &(k, amp) in &pulses[..num_pulse] {
        let sfb = swb.partition_point(|&o| o as usize <= k) - 1;
        if sfb < ics.max_sfb && matches!(ch.band_type[0][sfb], 1..=ESC_HCB) {
            spec[k] = t.add_pulse(spec[k], amp, t.gain[ch.sf[0][sfb] as usize]);
        }
    }

    // Noise substitution.
    if noise {
        let mut win = 0usize;
        for g in 0..ics.num_window_groups {
            for w in win..win + ics.group_len[g] {
                let base = if short { w * 128 } else { 0 };
                for sfb in 0..ics.max_sfb {
                    if ch.band_type[g][sfb] != NOISE_HCB {
                        continue;
                    }
                    let (lo, hi) = (base + swb[sfb] as usize, base + swb[sfb + 1] as usize);
                    let mut rand = [0.0f64; 128];
                    let rand = &mut rand[..hi - lo];
                    let mut energy = 0.0;
                    for r in rand.iter_mut() {
                        *noise_state = noise_state
                            .wrapping_mul(1_664_525)
                            .wrapping_add(1_013_904_223);
                        *r = *noise_state as i32 as f64;
                        energy += *r * *r;
                    }
                    let scale = (0.25 * ch.sf[g][sfb] as f64).exp2() / energy.sqrt();
                    for (s, &r) in spec[lo..hi].iter_mut().zip(rand.iter()) {
                        *s = (r * scale) as f32;
                    }
                }
            }
            win += ics.group_len[g];
        }
    }
    Ok(())
}

/// M/S and intensity stereo for a common-window channel pair.
fn stereo_tools(l: &mut Channel, r: &mut Channel, ms_used: &[[bool; 64]; 8], sr_idx: u8) {
    let ics = &l.ics;
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
                    let scale = (sign * (-0.25 * r.sf[g][sfb] as f64).exp2()) as f32;
                    for (d, &s) in r.spec[lo..hi].iter_mut().zip(&l.spec[lo..hi]) {
                        *d = s * scale;
                    }
                } else if ms_used[g][sfb] && bl < NOISE_HCB && brt < NOISE_HCB {
                    for (a, b) in l.spec[lo..hi].iter_mut().zip(&mut r.spec[lo..hi]) {
                        let (m, s) = (*a, *b);
                        *a = m + s;
                        *b = m - s;
                    }
                }
            }
        }
        win += ics.group_len[g];
    }
}

/// Temporal noise shaping: all-pole filtering of the spectrum per window.
fn apply_tns(ch: &mut Channel, sr_idx: u8) {
    let short = ch.ics.window_sequence == EIGHT_SHORT;
    let swb = ch.ics.swb(sr_idx);
    let num_swb = swb.len() - 1;
    let (max_bands, max_order) = if short {
        (TNS_MAX_BANDS_SHORT[sr_idx as usize], 7)
    } else {
        (TNS_MAX_BANDS_LONG[sr_idx as usize], 12)
    };
    let limit = max_bands.min(ch.ics.max_sfb);
    for w in 0..ch.ics.num_windows {
        let base = if short { w * 128 } else { 0 };
        let mut bottom = num_swb;
        for f in &ch.tns[w][..ch.tns_count[w]] {
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
            let x = &mut ch.spec[base..];
            let mut state = [0.0f32; 32];
            for i in 0..end - start {
                let k = if f.direction { end - 1 - i } else { start + i };
                let mut y = x[k];
                for j in 0..order {
                    y -= f.coef[j] * state[j];
                }
                state.copy_within(0..order - 1, 1);
                state[0] = y;
                x[k] = y;
            }
        }
    }
}

/// Filterbank: IMDCT, windowing and overlap-add of one channel into `out`.
fn synthesize(
    t: &Tables,
    st: &mut Synth,
    ch: &Channel,
    out: &mut [f32; 1024],
    y: &mut [f32; 1024],
) {
    let cur = &t.shapes[ch.ics.window_shape as usize];
    let prev = &t.shapes[st.prev_shape as usize];
    let ov = &mut st.overlap;
    if ch.ics.window_sequence == EIGHT_SHORT {
        // The eight short blocks overlap one another at 448 + 128 w within the frame's 2048
        // samples, chained through `tail`.
        let mut z = [0.0f32; 2048];
        let mut tail = [0.0f32; 128];
        let y = &mut y[..128];
        for w in 0..8 {
            t.short.dct4(&ch.spec[w * 128..w * 128 + 128], y);
            let left = if w == 0 {
                &prev.short_rise
            } else {
                &cur.short_rise
            };
            let z = &mut z[448 + 128 * w..576 + 128 * w];
            imdct_window(y, left, &cur.short_fall, &mut tail, z);
        }
        z[1472..1600].copy_from_slice(&tail);
        for n in 0..1024 {
            out[n] = z[n] + ov[n];
        }
        ov.copy_from_slice(&z[1024..]);
    } else {
        t.long.dct4(&ch.spec, y);
        let rise = if ch.ics.window_sequence == LONG_STOP {
            &prev.stop_rise
        } else {
            &prev.long_rise
        };
        let fall = if ch.ics.window_sequence == LONG_START {
            &cur.start_fall
        } else {
            &cur.long_fall
        };
        debug_assert!(matches!(
            ch.ics.window_sequence,
            ONLY_LONG | LONG_START | LONG_STOP
        ));
        imdct_window(y, rise, fall, ov, out);
    }
    st.prev_shape = ch.ics.window_shape;
}

/// Parses (and discards) a `program_config_element`.
fn skip_pce(br: &mut Bits) {
    let _tag = br.read(4);
    let _object_type = br.read(2);
    let _sf_index = br.read(4);
    let front = br.read(4) as usize;
    let side = br.read(4) as usize;
    let back = br.read(4) as usize;
    let lfe = br.read(2) as usize;
    let assoc = br.read(3) as usize;
    let cc = br.read(4) as usize;
    if br.bit() {
        br.read(4);
    }
    if br.bit() {
        br.read(4);
    }
    if br.bit() {
        br.read(3);
    }
    br.skip((front + side + back) * 5 + lfe * 4 + assoc * 4 + cc * 5);
    br.align();
    let comment = br.read(8) as usize;
    br.skip(8 * comment);
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPECTRAL: [(&[u16], &[u8]); 11] = [
        (&HCB_1_CODES, &HCB_1_BITS),
        (&HCB_2_CODES, &HCB_2_BITS),
        (&HCB_3_CODES, &HCB_3_BITS),
        (&HCB_4_CODES, &HCB_4_BITS),
        (&HCB_5_CODES, &HCB_5_BITS),
        (&HCB_6_CODES, &HCB_6_BITS),
        (&HCB_7_CODES, &HCB_7_BITS),
        (&HCB_8_CODES, &HCB_8_BITS),
        (&HCB_9_CODES, &HCB_9_BITS),
        (&HCB_10_CODES, &HCB_10_BITS),
        (&HCB_11_CODES, &HCB_11_BITS),
    ];

    /// A buffer holding `bits` (a string of `0` and `1`), zero padded.
    fn bitstring(bits: &str) -> Box<Buf> {
        let mut buf = Box::new([0u8; BUF_LEN + 8]);
        for (i, c) in bits.bytes().enumerate() {
            buf[i / 8] |= ((c == b'1') as u8) << (7 - i % 8);
        }
        buf
    }

    /// Every codeword (sign bits included) decodes to its own payload and length, whatever
    /// follows it.
    #[test]
    fn huffman_tables_decode_every_codeword() {
        let t = tables();
        for cb in 0..12 {
            let codes: Vec<(u32, u32, u32)> = if cb == 0 {
                (0..HCB_SF_CODES.len())
                    .map(|i| (HCB_SF_CODES[i], HCB_SF_BITS[i] as u32, i as u32))
                    .collect()
            } else {
                let (c, b) = SPECTRAL[cb - 1];
                spectral_codewords(cb, c, b)
            };
            for &(code, len, payload) in &codes {
                for tail in [0u64, u64::MAX] {
                    let word = (code as u64) << (64 - len) | (tail >> len);
                    let mut buf = Box::new([0xA5; BUF_LEN + 8]);
                    buf[..8].copy_from_slice(&word.to_be_bytes());
                    let mut br = Bits::new(&buf, 8);
                    let p = t.books[cb].decode(&mut br);
                    assert_eq!(
                        (p, br.pos),
                        (payload, len as usize),
                        "codebook {cb} code {code}"
                    );
                }
            }
        }
    }

    #[test]
    fn sign_bits_and_escape_sequence() {
        let t = tables();
        let code = |cb: usize, s: usize| {
            let (c, b) = SPECTRAL[cb - 1];
            format!("{:0w$b}", c[s], w = b[s] as usize)
        };
        // Codebook 3, (1, 0, 2, 0): one sign bit per nonzero value, in order (+1, -2).
        let bits = code(3, 27 + 6) + "01";
        let buf = bitstring(&bits);
        let mut br = Bits::new(&buf, bits.len().div_ceil(8));
        let p = t.books[3].decode(&mut br);
        assert_eq!((p, br.pos), (quad_row([1, 0, -2, 0]), bits.len()));

        // Codebook 11, (16, 3): signs (-, +), then the escape of the 16: two ones, a zero and
        // 4 + 2 bits (2^6 + 42).
        let bits = code(11, 16 * 17 + 3) + "10" + "110" + "101010";
        let buf = bitstring(&bits);
        let mut br = Bits::new(&buf, bits.len().div_ceil(8));
        let p = t.books[11].decode(&mut br);
        assert_eq!(p, pair_row([-16, 3]) | ESC_FLAG);
        assert_eq!(escape(&mut br, -16).unwrap(), -(64 + 42));
        assert_eq!(br.pos, bits.len());
        br.check().unwrap();

        // Nine ones is one too many.
        let buf = bitstring("1111111110");
        let mut br = Bits::new(&buf, 2);
        assert!(escape(&mut br, 16).is_err());
    }

    #[test]
    fn value_tables_match_dequantisation() {
        let t = tables();
        let deq = |v: &[i32]| {
            v.iter()
                .map(|&x| t.dequantise(x, 0.75).to_bits())
                .collect::<Vec<_>>()
        };
        for a in -16..=16 {
            for b in -16..=16 {
                let row = t.pairs[pair_row([a, b]) as usize].map(|x| (x * 0.75).to_bits());
                assert_eq!(row.to_vec(), deq(&[a, b]));
                if a.abs() <= 2 && b.abs() <= 2 {
                    let v = [a, b, b, a];
                    let row = t.quads[quad_row(v) as usize].map(|x| (x * 0.75).to_bits());
                    assert_eq!(row.to_vec(), deq(&v));
                }
            }
        }
    }

    /// Pulses amend the quantised value as if it were still at hand, for every value an escape
    /// and up to three earlier pulses can reach, at the extreme scale factors.
    #[test]
    fn pulses_amend_the_quantised_value() {
        let t = tables();
        for sf in [0, 1, 2, 3, 99, 100, 101, 252, 253, 254, 255] {
            let gain = t.gain[sf];
            for q in -(8191 + 45)..=8191 + 45 {
                for amp in [0, 1, 15] {
                    let want = t.dequantise(if q > 0 { q + amp } else { q - amp }, gain);
                    let got = t.add_pulse(t.dequantise(q, gain), amp, gain);
                    assert_eq!(got.to_bits(), want.to_bits(), "sf {sf} q {q} amp {amp}");
                }
            }
        }
    }
}
