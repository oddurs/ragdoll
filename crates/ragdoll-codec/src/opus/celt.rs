//! CELT decoder (libopus `celt_decoder.c`, fixed point) for the 48 kHz mode with 2.5-20 ms
//! frames, including the pitch-based and noise-based packet-loss concealment.

use super::bands::{AllBands, anti_collapse, denormalise_bands, quant_all_bands};
use super::celt_tables::{EBANDS, WINDOW};
use super::energy::{DB_SHIFT, unquant_coarse, unquant_finalise, unquant_fine};
use super::math::{
    frac_div32, lcg_rand, maxabs16, mul16, mul16_32_q15, mul16_q15, round16, sat16, saturate,
    shl32, sig2word16, sqrt, zlog2,
};
use super::mdct::mdct_backward;
use super::plc::{
    COMBFILTER_MINPERIOD, Comb, LPC_ORDER, SIG_SAT, SIG_SHIFT, autocorr, comb_filter_inplace,
    comb_filter_to, fir, iir, lpc_from_autocorr, pitch_downsample, pitch_search,
};
use super::range::{BITRES, Ec};
use super::rate::{NB_EBANDS, compute_allocation, init_caps};
use super::vq::{SPREAD_NORMAL, renormalise_vector};

pub const DECODE_BUFFER_SIZE: usize = 2048;
const OVERLAP: usize = 120;
const SHORT_MDCT: usize = 120;
const MAX_LM: i32 = 3;
const MAX_PERIOD: usize = 1024;
const PLC_PITCH_LAG_MAX: usize = 720;
const PLC_PITCH_LAG_MIN: usize = 100;
const PREEMPH: i32 = 27853;
const STRIDE: usize = DECODE_BUFFER_SIZE + OVERLAP;

pub(super) const TF_SELECT_TABLE: [[i8; 8]; 4] = [
    [0, -1, 0, -1, 0, -1, 0, -1],
    [0, -1, 0, -2, 1, 0, 1, -1],
    [0, -2, 0, -3, 2, 0, 1, -1],
    [0, -2, 0, -3, 3, 0, 1, -1],
];
pub(super) const TRIM_ICDF: [u8; 11] = [126, 124, 119, 109, 87, 41, 19, 9, 4, 2, 0];
pub(super) const SPREAD_ICDF: [u8; 4] = [25, 23, 2, 0];
pub(super) const TAPSET_ICDF: [u8; 3] = [2, 1, 0];

/// The decoder's error for a frame it cannot handle (bad size or internal overrun).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CeltError {
    BadArg,
    Internal,
}

/// CELT decoder state (`OpusCustomDecoder` with the 48000/960 mode).
#[derive(Clone)]
pub struct CeltDecoder {
    pub channels: usize,
    pub stream_channels: usize,
    pub downsample: usize,
    pub start: usize,
    pub end: usize,
    pub disable_inv: bool,
    pub rng: u32,
    pub error: bool,
    last_pitch_index: usize,
    loss_count: i32,
    skip_plc: bool,
    postfilter_period: i32,
    postfilter_period_old: i32,
    postfilter_gain: i32,
    postfilter_gain_old: i32,
    postfilter_tapset: usize,
    postfilter_tapset_old: usize,
    preemph_mem: [i32; 2],
    decode_mem: Vec<i32>,
    /// Per-frame working memory, kept to spare clearing it every frame: the normalised
    /// coefficients (`2 * 960`) then the folding spectrum (`2 * 8 * 78`).
    scratch: Vec<i16>,
    /// The denormalised spectrum of each channel (`2 * 960`).
    freq_scratch: Vec<i32>,
    lpc: [i16; 2 * LPC_ORDER],
    old_band_e: [i16; 2 * NB_EBANDS],
    old_log_e: [i16; 2 * NB_EBANDS],
    old_log_e2: [i16; 2 * NB_EBANDS],
    background_log_e: [i16; 2 * NB_EBANDS],
}

/// `resampling_factor`.
pub fn resampling_factor(rate: u32) -> usize {
    match rate {
        48000 => 1,
        24000 => 2,
        16000 => 3,
        12000 => 4,
        8000 => 6,
        _ => 0,
    }
}

impl CeltDecoder {
    /// `celt_decoder_init` at output rate `rate` (48, 24, 16, 12 or 8 kHz).
    pub fn new(rate: u32, channels: usize) -> Self {
        let mut d = Self {
            channels,
            stream_channels: channels,
            downsample: resampling_factor(rate).max(1),
            start: 0,
            end: NB_EBANDS,
            disable_inv: channels == 1,
            rng: 0,
            error: false,
            last_pitch_index: 0,
            loss_count: 0,
            skip_plc: false,
            postfilter_period: 0,
            postfilter_period_old: 0,
            postfilter_gain: 0,
            postfilter_gain_old: 0,
            postfilter_tapset: 0,
            postfilter_tapset_old: 0,
            preemph_mem: [0; 2],
            decode_mem: vec![0; channels * STRIDE],
            scratch: vec![0; 2 * 960 + 2 * 8 * 78],
            freq_scratch: vec![0; 2 * 960],
            lpc: [0; 2 * LPC_ORDER],
            old_band_e: [0; 2 * NB_EBANDS],
            old_log_e: [0; 2 * NB_EBANDS],
            old_log_e2: [0; 2 * NB_EBANDS],
            background_log_e: [0; 2 * NB_EBANDS],
        };
        d.reset();
        d
    }

    /// `OPUS_RESET_STATE`.
    pub fn reset(&mut self) {
        self.rng = 0;
        self.error = false;
        self.last_pitch_index = 0;
        self.loss_count = 0;
        self.postfilter_period = 0;
        self.postfilter_period_old = 0;
        self.postfilter_gain = 0;
        self.postfilter_gain_old = 0;
        self.postfilter_tapset = 0;
        self.postfilter_tapset_old = 0;
        self.preemph_mem = [0; 2];
        self.decode_mem.fill(0);
        self.lpc = [0; 2 * LPC_ORDER];
        self.old_band_e = [0; 2 * NB_EBANDS];
        self.old_log_e = [-(28 << DB_SHIFT); 2 * NB_EBANDS];
        self.old_log_e2 = [-(28 << DB_SHIFT); 2 * NB_EBANDS];
        self.background_log_e = [0; 2 * NB_EBANDS];
        self.skip_plc = true;
    }

    /// `celt_decode_with_ec`: decodes one frame of `frame_size` samples (at the output rate) into
    /// interleaved `pcm`. `data` of `None` or one byte conceals a lost frame. With `ec`, the
    /// frame continues an existing range decoder (hybrid and CELT-only Opus frames).
    pub fn decode(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i16],
        frame_size: usize,
        ec: Option<&mut Ec>,
        accum: bool,
    ) -> Result<usize, CeltError> {
        let cc = self.channels;
        let frame_size = frame_size * self.downsample;
        let mut lm = 0i32;
        while lm <= MAX_LM && (SHORT_MDCT << lm) != frame_size {
            lm += 1;
        }
        if lm > MAX_LM {
            return Err(CeltError::BadArg);
        }
        let m = 1usize << lm;
        let len = data.map_or(0, |d| d.len());
        if len > 1275 {
            return Err(CeltError::BadArg);
        }
        let n = m * SHORT_MDCT;
        let end = self.end;
        let eff_end = end.min(NB_EBANDS);
        let data = match data {
            Some(d) if d.len() > 1 => d,
            _ => {
                self.decode_lost(n, lm);
                self.deemphasis(pcm, n, accum);
                return Ok(frame_size / self.downsample);
            }
        };
        self.skip_plc = self.loss_count != 0;
        let mut own;
        let dec: &mut Ec = match ec {
            Some(e) => e,
            None => {
                own = Ec::decoder(data);
                &mut own
            }
        };
        let c = self.stream_channels;
        let start = self.start;
        if c == 1 {
            for i in 0..NB_EBANDS {
                self.old_band_e[i] = self.old_band_e[i].max(self.old_band_e[NB_EBANDS + i]);
            }
        }
        let mut total_bits = len as i32 * 8;
        let mut tell = dec.tell();
        let silence = if tell >= total_bits {
            true
        } else if tell == 1 {
            dec.dec_bit_logp(15)
        } else {
            false
        };
        if silence {
            tell = len as i32 * 8;
            dec.nbits_total += tell - dec.tell();
        }
        let mut postfilter_gain = 0i32;
        let mut postfilter_pitch = 0i32;
        let mut postfilter_tapset = 0usize;
        if start == 0 && tell + 16 <= total_bits {
            if dec.dec_bit_logp(1) {
                let octave = dec.dec_uint(6) as i32;
                postfilter_pitch = (16 << octave) + dec.dec_bits((4 + octave) as u32) as i32 - 1;
                let qg = dec.dec_bits(3) as i32;
                if dec.tell() + 2 <= total_bits {
                    postfilter_tapset = dec.dec_icdf(&TAPSET_ICDF, 2) as usize;
                }
                postfilter_gain = 3072 * (qg + 1);
            }
            tell = dec.tell();
        }
        let is_transient = if lm > 0 && tell + 3 <= total_bits {
            let t = dec.dec_bit_logp(3);
            tell = dec.tell();
            t
        } else {
            false
        };
        let short_blocks = is_transient;
        let intra_ener = if tell + 3 <= total_bits {
            dec.dec_bit_logp(3)
        } else {
            false
        };
        unquant_coarse(start, end, &mut self.old_band_e, intra_ener, dec, c, lm);
        let mut tf_res = [0i32; NB_EBANDS];
        tf_decode(start, end, is_transient, &mut tf_res, lm, dec);
        tell = dec.tell();
        let mut spread_decision = SPREAD_NORMAL;
        if tell + 4 <= total_bits {
            spread_decision = dec.dec_icdf(&SPREAD_ICDF, 5);
        }
        let mut cap = [0i32; NB_EBANDS];
        init_caps(&mut cap, lm, c as i32);
        let mut offsets = [0i32; NB_EBANDS];
        let mut dynalloc_logp = 6;
        total_bits <<= BITRES;
        let mut tellf = dec.tell_frac() as i32;
        for i in start..end {
            let width = (c as i32 * (EBANDS[i + 1] - EBANDS[i]) as i32) << lm;
            let quanta = (width << BITRES).min((6 << BITRES).max(width));
            let mut loop_logp = dynalloc_logp;
            let mut boost = 0;
            while tellf + (loop_logp << BITRES) < total_bits && boost < cap[i] {
                let flag = dec.dec_bit_logp(loop_logp as u32);
                tellf = dec.tell_frac() as i32;
                if !flag {
                    break;
                }
                boost += quanta;
                total_bits -= quanta;
                loop_logp = 1;
            }
            offsets[i] = boost;
            if boost > 0 {
                dynalloc_logp = 2.max(dynalloc_logp - 1);
            }
        }
        let alloc_trim = if tellf + (6 << BITRES) <= total_bits {
            dec.dec_icdf(&TRIM_ICDF, 7)
        } else {
            5
        };
        let mut bits = ((len as i32 * 8) << BITRES) - dec.tell_frac() as i32 - 1;
        let anti_collapse_rsv = if is_transient && lm >= 2 && bits >= (lm + 2) << BITRES {
            1 << BITRES
        } else {
            0
        };
        bits -= anti_collapse_rsv;
        let mut pulses = [0i32; NB_EBANDS];
        let mut fine_quant = [0i32; NB_EBANDS];
        let mut fine_priority = [0i32; NB_EBANDS];
        let mut intensity = 0i32;
        let mut dual_stereo = false;
        let alloc = compute_allocation(
            start,
            end,
            &offsets,
            &cap,
            alloc_trim,
            &mut intensity,
            &mut dual_stereo,
            bits,
            &mut pulses,
            &mut fine_quant,
            &mut fine_priority,
            c as i32,
            lm,
            dec,
            false,
            0,
            0,
        );
        unquant_fine(start, end, &mut self.old_band_e, &fine_quant, dec, c);
        for ch in 0..cc {
            let o = ch * STRIDE;
            self.decode_mem
                .copy_within(o + n..o + DECODE_BUFFER_SIZE + OVERLAP / 2, o);
        }
        let mut collapse_masks = [0u8; 2 * NB_EBANDS];
        let collapse_masks = &mut collapse_masks[..c * NB_EBANDS];
        // Every value read below is written first, so the scratch need not be cleared.
        let mut scratch = std::mem::take(&mut self.scratch);
        let (x, norm) = scratch.split_at_mut(2 * 960);
        let x = &mut x[..c * n];
        quant_all_bands(
            AllBands {
                complexity: 0,
                encode: false,
                start,
                end,
                collapse_masks,
                band_e: &[],
                pulses: &pulses,
                short_blocks,
                spread: spread_decision,
                dual_stereo: alloc.dual_stereo,
                intensity: alloc.intensity,
                tf_res: &tf_res,
                total_bits: (len as i32 * (8 << BITRES)) - anti_collapse_rsv,
                balance: alloc.balance,
                lm,
                coded_bands: alloc.coded_bands,
                disable_inv: self.disable_inv,
                norm,
            },
            x,
            c,
            dec,
            &mut self.rng,
        );
        let mut anti_collapse_on = false;
        if anti_collapse_rsv > 0 {
            anti_collapse_on = dec.dec_bits(1) != 0;
        }
        unquant_finalise(
            start,
            end,
            &mut self.old_band_e,
            &fine_quant,
            &fine_priority,
            len as i32 * 8 - dec.tell(),
            dec,
            c,
        );
        if anti_collapse_on {
            anti_collapse(
                x,
                collapse_masks,
                lm,
                c,
                n,
                start,
                end,
                &self.old_band_e,
                &self.old_log_e,
                &self.old_log_e2,
                &pulses,
                self.rng,
            );
        }
        if silence {
            for v in self.old_band_e[..c * NB_EBANDS].iter_mut() {
                *v = -(28 << DB_SHIFT);
            }
        }
        self.synthesis(x, start, eff_end, c, is_transient, lm, silence);
        self.scratch = scratch;
        for ch in 0..cc {
            self.postfilter_period = self.postfilter_period.max(COMBFILTER_MINPERIOD);
            self.postfilter_period_old = self.postfilter_period_old.max(COMBFILTER_MINPERIOD);
            let base = ch * STRIDE;
            let off = base + DECODE_BUFFER_SIZE - n;
            comb_filter_inplace(
                &mut self.decode_mem,
                off,
                SHORT_MDCT,
                Comb {
                    t0: self.postfilter_period_old,
                    t1: self.postfilter_period,
                    g0: self.postfilter_gain_old,
                    g1: self.postfilter_gain,
                    tapset0: self.postfilter_tapset_old,
                    tapset1: self.postfilter_tapset,
                },
                &WINDOW,
                OVERLAP,
            );
            if lm != 0 {
                comb_filter_inplace(
                    &mut self.decode_mem,
                    off + SHORT_MDCT,
                    n - SHORT_MDCT,
                    Comb {
                        t0: self.postfilter_period,
                        t1: postfilter_pitch,
                        g0: self.postfilter_gain,
                        g1: postfilter_gain,
                        tapset0: self.postfilter_tapset,
                        tapset1: postfilter_tapset,
                    },
                    &WINDOW,
                    OVERLAP,
                );
            }
        }
        self.postfilter_period_old = self.postfilter_period;
        self.postfilter_gain_old = self.postfilter_gain;
        self.postfilter_tapset_old = self.postfilter_tapset;
        self.postfilter_period = postfilter_pitch;
        self.postfilter_gain = postfilter_gain;
        self.postfilter_tapset = postfilter_tapset;
        if lm != 0 {
            self.postfilter_period_old = self.postfilter_period;
            self.postfilter_gain_old = self.postfilter_gain;
            self.postfilter_tapset_old = self.postfilter_tapset;
        }
        if c == 1 {
            let (a, b) = self.old_band_e.split_at_mut(NB_EBANDS);
            b.copy_from_slice(a);
        }
        if !is_transient {
            self.old_log_e2 = self.old_log_e;
            self.old_log_e = self.old_band_e;
            let max_background_increase: i32 = if self.loss_count < 10 {
                // M * QCONST16(0.001, DB_SHIFT), which is M * 1.
                m as i32
            } else {
                1 << DB_SHIFT
            };
            for i in 0..2 * NB_EBANDS {
                self.background_log_e[i] =
                    (self.background_log_e[i] as i32 + max_background_increase)
                        .min(self.old_band_e[i] as i32) as i16;
            }
        } else {
            for i in 0..2 * NB_EBANDS {
                self.old_log_e[i] = self.old_log_e[i].min(self.old_band_e[i]);
            }
        }
        for ch in 0..2 {
            for i in (0..start).chain(end..NB_EBANDS) {
                let k = ch * NB_EBANDS + i;
                self.old_band_e[k] = 0;
                self.old_log_e[k] = -(28 << DB_SHIFT);
                self.old_log_e2[k] = -(28 << DB_SHIFT);
            }
        }
        self.rng = dec.rng;
        self.deemphasis(pcm, n, accum);
        self.loss_count = 0;
        if dec.tell() > 8 * len as i32 {
            return Err(CeltError::Internal);
        }
        if dec.error != 0 {
            self.error = true;
        }
        Ok(frame_size / self.downsample)
    }

    /// `celt_synthesis`: denormalisation and inverse MDCT into the decode buffer.
    #[allow(clippy::too_many_arguments)]
    fn synthesis(
        &mut self,
        x: &[i16],
        start: usize,
        eff_end: usize,
        c: usize,
        is_transient: bool,
        lm: i32,
        silence: bool,
    ) {
        let cc = self.channels;
        let m = 1usize << lm;
        let n = m * SHORT_MDCT;
        let (b_cnt, nb, shift) = if is_transient {
            (m, SHORT_MDCT, MAX_LM as usize)
        } else {
            (1, n, (MAX_LM - lm) as usize)
        };
        let ds = self.downsample;
        let mut fs = std::mem::take(&mut self.freq_scratch);
        let (freq, freq2) = fs.split_at_mut(960);
        let freq = &mut freq[..n];
        let out_base = DECODE_BUFFER_SIZE - n;
        let backward = |mem: &mut [i32], ch: usize, f: &[i32]| {
            for b in 0..b_cnt {
                let o = ch * STRIDE + out_base + nb * b;
                mdct_backward(
                    &f[b..],
                    &mut mem[o..o + nb + OVERLAP],
                    &WINDOW,
                    OVERLAP,
                    shift,
                    b_cnt,
                );
            }
        };
        if cc == 2 && c == 1 {
            denormalise_bands(
                x,
                freq,
                &self.old_band_e,
                start,
                eff_end,
                m,
                ds,
                silence,
            );
            backward(&mut self.decode_mem, 0, freq);
            backward(&mut self.decode_mem, 1, freq);
        } else if cc == 1 && c == 2 {
            let freq2 = &mut freq2[..n];
            denormalise_bands(
                x,
                freq,
                &self.old_band_e,
                start,
                eff_end,
                m,
                ds,
                silence,
            );
            denormalise_bands(
                &x[n..],
                freq2,
                &self.old_band_e[NB_EBANDS..],
                start,
                eff_end,
                m,
                ds,
                silence,
            );
            for i in 0..n {
                freq[i] = (freq[i] >> 1).wrapping_add(freq2[i] >> 1);
            }
            backward(&mut self.decode_mem, 0, freq);
        } else {
            for ch in 0..cc {
                denormalise_bands(
                    &x[ch * n..],
                    freq,
                    &self.old_band_e[ch * NB_EBANDS..],
                    start,
                    eff_end,
                    m,
                    ds,
                    silence,
                );
                backward(&mut self.decode_mem, ch, freq);
            }
        }
        for ch in 0..cc {
            let o = ch * STRIDE + out_base;
            saturate_slice(&mut self.decode_mem[o..o + n]);
        }
        self.freq_scratch = fs;
    }

    /// `deemphasis`: de-emphasis, downsampling and conversion to 16-bit `pcm` of the last `n`
    /// samples of the decode buffer.
    fn deemphasis(&mut self, pcm: &mut [i16], n: usize, accum: bool) {
        let cc = self.channels;
        let ds = self.downsample;
        let nd = n / ds;
        let mem = &self.decode_mem;
        let x = |ch: usize| &mem[ch * STRIDE + DECODE_BUFFER_SIZE - n..ch * STRIDE + DECODE_BUFFER_SIZE];
        let step = |mm: &mut i32, v: i32| {
            let tmp = v.wrapping_add(*mm);
            *mm = mul16_32_q15(PREEMPH, tmp);
            tmp
        };
        if cc == 2 && ds == 1 && !accum {
            // The common case: both channels at once so their recurrences overlap, with the
            // conversion to 16 bits (off the recurrence) done a block at a time.
            const BLOCK: usize = 64;
            let [mut m0, mut m1] = self.preemph_mem;
            let mut tmp = [0i32; 2 * BLOCK];
            for ((o, a), b) in pcm[..2 * n]
                .chunks_mut(2 * BLOCK)
                .zip(x(0).chunks(BLOCK))
                .zip(x(1).chunks(BLOCK))
            {
                let t = &mut tmp[..o.len()];
                // The sums stay in 32 bits in practice (|x| <= SIG_SAT keeps them below
                // 2.1e9), so the recurrence runs in 64 bits without the sign extension on its
                // chain, checking each sum; one that wraps leaves for the exact loop.
                let (mut f0, mut f1) = (m0 as i64, m1 as i64);
                let mut k = 0;
                for ((t, &a), &b) in t.chunks_exact_mut(2).zip(a).zip(b) {
                    let (s0, s1) = (a as i64 + f0, b as i64 + f1);
                    if s0 != s0 as i32 as i64 || s1 != s1 as i32 as i64 {
                        break;
                    }
                    t[0] = s0 as i32;
                    t[1] = s1 as i32;
                    f0 = (PREEMPH as i64 * s0) >> 15;
                    f1 = (PREEMPH as i64 * s1) >> 15;
                    k += 1;
                }
                (m0, m1) = (f0 as i32, f1 as i32);
                for ((t, &a), &b) in t.chunks_exact_mut(2).zip(a).zip(b).skip(k) {
                    t[0] = step(&mut m0, a);
                    t[1] = step(&mut m1, b);
                }
                sig2word16_slice(o, t);
            }
            self.preemph_mem = [m0, m1];
            return;
        }
        for ch in 0..cc {
            let mut mm = self.preemph_mem[ch];
            let out = pcm[ch..].iter_mut().step_by(cc).take(nd);
            for (o, xs) in out.zip(x(ch).chunks_exact(ds)) {
                // Only every ds-th sample is kept, but the filter runs on all of them.
                let v = sig2word16(step(&mut mm, xs[0]));
                for &xv in &xs[1..] {
                    step(&mut mm, xv);
                }
                *o = if accum { sat16(*o as i32 + v as i32) } else { v };
            }
            for &xv in &x(ch)[nd * ds..] {
                step(&mut mm, xv);
            }
            self.preemph_mem[ch] = mm;
        }
    }

    /// `celt_plc_pitch_search`.
    fn plc_pitch_search(&self) -> usize {
        let c = self.channels;
        let mut lp = vec![0i16; DECODE_BUFFER_SIZE >> 1];
        let chans: Vec<&[i32]> = (0..c)
            .map(|ch| &self.decode_mem[ch * STRIDE..ch * STRIDE + DECODE_BUFFER_SIZE])
            .collect();
        pitch_downsample(&chans, &mut lp, DECODE_BUFFER_SIZE);
        let p = pitch_search(
            &lp[PLC_PITCH_LAG_MAX >> 1..],
            &lp,
            DECODE_BUFFER_SIZE - PLC_PITCH_LAG_MAX,
            PLC_PITCH_LAG_MAX - PLC_PITCH_LAG_MIN,
        );
        PLC_PITCH_LAG_MAX - p
    }

    /// `celt_decode_lost`: conceals a lost frame of `n` samples.
    fn decode_lost(&mut self, n: usize, lm: i32) {
        let c = self.channels;
        let start = self.start;
        let loss_count = self.loss_count;
        let noise_based = loss_count >= 5 || start != 0 || self.skip_plc;
        if noise_based {
            let end = self.end;
            let eff_end = start.max(end.min(NB_EBANDS));
            let mut x = vec![0i16; c * n];
            let decay: i32 = if loss_count == 0 { 1536 } else { 512 };
            for ch in 0..c {
                for i in start..end {
                    let k = ch * NB_EBANDS + i;
                    self.old_band_e[k] = (self.background_log_e[k] as i32)
                        .max(self.old_band_e[k] as i32 - decay)
                        as i16;
                }
            }
            let mut seed = self.rng;
            for ch in 0..c {
                for i in start..eff_end {
                    let boffs = n * ch + ((EBANDS[i] as usize) << lm);
                    let blen = ((EBANDS[i + 1] - EBANDS[i]) as usize) << lm;
                    for j in 0..blen {
                        seed = lcg_rand(seed);
                        x[boffs + j] = ((seed as i32) >> 20) as i16;
                    }
                    renormalise_vector(&mut x[boffs..], blen, 32767);
                }
            }
            self.rng = seed;
            for ch in 0..c {
                let o = ch * STRIDE;
                self.decode_mem
                    .copy_within(o + n..o + DECODE_BUFFER_SIZE + (OVERLAP >> 1), o);
            }
            self.synthesis(&x, start, eff_end, c, false, lm, false);
        } else {
            let mut fade = 32767i32;
            let pitch_index = if loss_count == 0 {
                let p = self.plc_pitch_search();
                self.last_pitch_index = p;
                p
            } else {
                fade = 26214;
                self.last_pitch_index
            };
            let exc_length = (2 * pitch_index).min(MAX_PERIOD);
            let mut etmp = vec![0i32; OVERLAP];
            let mut exc_buf = vec![0i16; MAX_PERIOD + LPC_ORDER];
            let mut fir_tmp = vec![0i16; exc_length];
            for ch in 0..c {
                let base = ch * STRIDE;
                for i in 0..MAX_PERIOD + LPC_ORDER {
                    exc_buf[i] = round16(
                        self.decode_mem[base + DECODE_BUFFER_SIZE - MAX_PERIOD - LPC_ORDER + i],
                        SIG_SHIFT,
                    );
                }
                let exc_off = LPC_ORDER;
                let lpc = &mut self.lpc[ch * LPC_ORDER..(ch + 1) * LPC_ORDER];
                if loss_count == 0 {
                    let mut ac = [0i32; LPC_ORDER + 1];
                    autocorr(
                        &exc_buf[exc_off..],
                        &mut ac,
                        Some(&WINDOW),
                        OVERLAP,
                        LPC_ORDER,
                        MAX_PERIOD,
                    );
                    ac[0] = ac[0].wrapping_add(ac[0] >> 13);
                    for (i, a) in ac.iter_mut().enumerate().skip(1) {
                        *a = a.wrapping_sub(mul16_32_q15((2 * i * i) as i32, *a));
                    }
                    lpc_from_autocorr(lpc, &ac, LPC_ORDER);
                    loop {
                        let mut tmp = 32767i32;
                        let mut sum = 1 << SIG_SHIFT;
                        for &v in lpc.iter() {
                            sum += (v as i32).abs();
                        }
                        if sum < 65535 {
                            break;
                        }
                        for v in lpc.iter_mut() {
                            tmp = mul16_q15(32440, tmp);
                            *v = mul16_q15(*v as i32, tmp) as i16;
                        }
                    }
                }
                {
                    let fo = exc_off + MAX_PERIOD - exc_length;
                    fir(&exc_buf, fo, lpc, &mut fir_tmp, exc_length, LPC_ORDER);
                    exc_buf[fo..fo + exc_length].copy_from_slice(&fir_tmp);
                }
                let exc = |i: usize| exc_buf[exc_off + i] as i32;
                let decay;
                {
                    let mut e1 = 1i32;
                    let mut e2 = 1i32;
                    let shift = (2 * zlog2(maxabs16(
                        &exc_buf[exc_off + MAX_PERIOD - exc_length..exc_off + MAX_PERIOD],
                    )) - 20)
                        .max(0);
                    let decay_length = exc_length >> 1;
                    for i in 0..decay_length {
                        let e = exc(MAX_PERIOD - decay_length + i);
                        e1 = e1.wrapping_add(mul16(e, e) >> shift);
                        let e = exc(MAX_PERIOD - 2 * decay_length + i);
                        e2 = e2.wrapping_add(mul16(e, e) >> shift);
                    }
                    e1 = e1.min(e2);
                    decay = sqrt(frac_div32(e1 >> 1, e2)) as i16 as i32;
                }
                self.decode_mem
                    .copy_within(base + n..base + DECODE_BUFFER_SIZE, base);
                let extrapolation_offset = MAX_PERIOD - pitch_index;
                let extrapolation_len = n + OVERLAP;
                let mut attenuation = mul16_q15(fade, decay) as i16 as i32;
                let mut s1 = 0i32;
                let mut j = 0usize;
                for i in 0..extrapolation_len {
                    if j >= pitch_index {
                        j -= pitch_index;
                        attenuation = mul16_q15(attenuation, decay) as i16 as i32;
                    }
                    self.decode_mem[base + DECODE_BUFFER_SIZE - n + i] = shl32(
                        mul16_q15(attenuation, exc(extrapolation_offset + j)) as i16 as i32,
                        SIG_SHIFT,
                    );
                    let tmp = round16(
                        self.decode_mem
                            [base + DECODE_BUFFER_SIZE - MAX_PERIOD - n + extrapolation_offset + j],
                        SIG_SHIFT,
                    ) as i32;
                    s1 = s1.wrapping_add(mul16(tmp, tmp) >> 10);
                    j += 1;
                }
                {
                    let mut lpc_mem = [0i16; LPC_ORDER];
                    for (i, v) in lpc_mem.iter_mut().enumerate() {
                        *v = round16(
                            self.decode_mem[base + DECODE_BUFFER_SIZE - n - 1 - i],
                            SIG_SHIFT,
                        );
                    }
                    let o = base + DECODE_BUFFER_SIZE - n;
                    let lpc = &self.lpc[ch * LPC_ORDER..(ch + 1) * LPC_ORDER];
                    iir(
                        &mut self.decode_mem[o..o + extrapolation_len],
                        lpc,
                        extrapolation_len,
                        LPC_ORDER,
                        &mut lpc_mem,
                    );
                    for v in self.decode_mem[o..o + extrapolation_len].iter_mut() {
                        *v = saturate(*v, SIG_SAT);
                    }
                }
                {
                    let o = base + DECODE_BUFFER_SIZE - n;
                    let mut s2 = 0i32;
                    for i in 0..extrapolation_len {
                        let tmp = round16(self.decode_mem[o + i], SIG_SHIFT) as i32;
                        s2 = s2.wrapping_add(mul16(tmp, tmp) >> 10);
                    }
                    if s1 <= (s2 >> 2) {
                        self.decode_mem[o..o + extrapolation_len].fill(0);
                    } else if s1 < s2 {
                        let ratio = sqrt(frac_div32((s1 >> 1) + 1, s2 + 1)) as i16 as i32;
                        for i in 0..OVERLAP {
                            let tmp_g = 32767 - mul16_q15(WINDOW[i] as i32, 32767 - ratio);
                            self.decode_mem[o + i] = mul16_32_q15(tmp_g, self.decode_mem[o + i]);
                        }
                        for i in OVERLAP..extrapolation_len {
                            self.decode_mem[o + i] = mul16_32_q15(ratio, self.decode_mem[o + i]);
                        }
                    }
                }
                comb_filter_to(
                    &mut etmp,
                    &self.decode_mem,
                    base + DECODE_BUFFER_SIZE,
                    OVERLAP,
                    Comb {
                        t0: self.postfilter_period,
                        t1: self.postfilter_period,
                        g0: -self.postfilter_gain,
                        g1: -self.postfilter_gain,
                        tapset0: self.postfilter_tapset,
                        tapset1: self.postfilter_tapset,
                    },
                    &WINDOW,
                    0,
                );
                for i in 0..OVERLAP / 2 {
                    self.decode_mem[base + DECODE_BUFFER_SIZE + i] =
                        mul16_32_q15(WINDOW[i] as i32, etmp[OVERLAP - 1 - i])
                            .wrapping_add(mul16_32_q15(WINDOW[OVERLAP - i - 1] as i32, etmp[i]));
                }
            }
        }
        self.loss_count = loss_count + 1;
    }
}

/// `SATURATE(x, SIG_SAT)` in place.
fn saturate_slice(x: &mut [i32]) {
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        unsafe { super::simd::saturate_avx2(x, SIG_SAT) };
        return;
    }
    for v in x.iter_mut() {
        *v = saturate(*v, SIG_SAT);
    }
}

/// `out[i] = SIG2WORD16(x[i])`.
fn sig2word16_slice(out: &mut [i16], x: &[i32]) {
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        let done = unsafe { super::simd::sig2word16_avx2(out, x) };
        sig2word16_slice_scalar(&mut out[done..], &x[done..]);
        return;
    }
    sig2word16_slice_scalar(out, x);
}

/// Scalar reference of [`sig2word16_slice`].
pub fn sig2word16_slice_scalar(out: &mut [i16], x: &[i32]) {
    for (o, &v) in out.iter_mut().zip(x) {
        *o = sig2word16(v);
    }
}

/// `tf_decode`.
fn tf_decode(
    start: usize,
    end: usize,
    is_transient: bool,
    tf_res: &mut [i32],
    lm: i32,
    dec: &mut Ec,
) {
    let mut budget = dec.storage * 8;
    let mut tell = dec.tell() as u32;
    let mut logp: u32 = if is_transient { 2 } else { 4 };
    let tf_select_rsv = lm > 0 && tell + logp < budget;
    budget -= tf_select_rsv as u32;
    let mut tf_changed = 0i32;
    let mut curr = 0i32;
    for i in start..end {
        if tell + logp <= budget {
            curr ^= dec.dec_bit_logp(logp) as i32;
            tell = dec.tell() as u32;
            tf_changed |= curr;
        }
        tf_res[i] = curr;
        logp = if is_transient { 4 } else { 5 };
    }
    let mut tf_select = 0usize;
    let ti = 4 * is_transient as usize;
    let row = &TF_SELECT_TABLE[lm as usize];
    if tf_select_rsv && row[ti + tf_changed as usize] != row[ti + 2 + tf_changed as usize] {
        tf_select = dec.dec_bit_logp(1) as usize;
    }
    for v in tf_res[start..end].iter_mut() {
        *v = row[ti + 2 * tf_select + *v as usize] as i32;
    }
}
