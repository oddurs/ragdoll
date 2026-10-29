//! CELT encoder, a port of fixed-point libopus 1.3.1 `celt_encoder.c` (with `spreading_decision`
//! and `hysteresis_decision` from `bands.c` and `remove_doubling` from `pitch.c`) for 48 kHz
//! input. The analysis-driven paths of libopus (tonality analysis, surround masking, LFE, hybrid
//! mode) are not used by the Opus layer here and are left out.

use super::bands::{AllBands, compute_band_energies, haar1, normalise_bands, quant_all_bands};
use super::celt::{SPREAD_ICDF, TAPSET_ICDF, TF_SELECT_TABLE, TRIM_ICDF};
use super::celt_tables::{E_MEANS, EBANDS, LOG_N, WINDOW};
use super::energy::{DB_SHIFT, amp2log2, quant_coarse, quant_finalise, quant_fine};
use super::math::{
    add16, celt_udiv, exp2, frac_div32, ilog2, inner_prod, log2, maxabs16, mul16, mul16_32_q15,
    mul16_q14, mul16_q15, pshr32, rcp, rsqrt_norm, shl16, shl32, sqrt, sround16, sub16, vshr32,
};
use super::mdct::mdct_forward;
use super::plc::{
    COMBFILTER_MINPERIOD, Comb, SIG_SHIFT, comb_filter_to, dot16, pitch_downsample, pitch_search,
};
use super::range::{BITRES, Ec, ilog};
use super::rate::{NB_EBANDS, compute_allocation, init_caps};
use super::vq::{SPREAD_AGGRESSIVE, SPREAD_LIGHT, SPREAD_NONE, SPREAD_NORMAL};

const OVERLAP: usize = 120;
const MAXPERIOD: usize = 1024;
const MINPERIOD: usize = COMBFILTER_MINPERIOD as usize;
const PREEMPH: i32 = 27853;
const EPSILON: i32 = 1;
/// `OPUS_BITRATE_MAX`.
pub const BITRATE_MAX: i32 = -1;

#[inline]
fn eb(i: usize) -> i32 {
    EBANDS[i] as i32
}

/// `OpusCustomEncoder` for the 48 kHz / 960-sample mode.
#[derive(Clone)]
pub struct CeltEncoder {
    channels: usize,
    pub stream_channels: usize,
    pub force_intra: bool,
    pub disable_pf: bool,
    pub complexity: i32,
    pub start: usize,
    pub end: usize,
    pub bitrate: i32,
    pub vbr: bool,
    pub constrained_vbr: bool,
    pub loss_rate: i32,
    pub lsb_depth: i32,
    pub disable_inv: bool,
    pub rng: u32,
    spread_decision: i32,
    delayed_intra: i32,
    tonal_average: i32,
    last_coded_bands: i32,
    hf_average: i32,
    tapset_decision: i32,
    prefilter_period: i32,
    prefilter_gain: i32,
    prefilter_tapset: i32,
    consec_transient: i32,
    preemph_mem_e: [i32; 2],
    vbr_reservoir: i32,
    vbr_drift: i32,
    vbr_offset: i32,
    vbr_count: i32,
    overlap_max: i32,
    stereo_saving: i32,
    intensity: i32,
    spec_avg: i32,
    in_mem: Vec<i32>,
    prefilter_mem: Vec<i32>,
    old_band_e: [i16; 2 * NB_EBANDS],
    old_log_e: [i16; 2 * NB_EBANDS],
    old_log_e2: [i16; 2 * NB_EBANDS],
    energy_error: [i16; 2 * NB_EBANDS],
}

impl CeltEncoder {
    /// `celt_encoder_init(st, 48000, channels)`.
    pub fn new(channels: usize) -> Self {
        let mut s = CeltEncoder {
            channels,
            stream_channels: channels,
            force_intra: false,
            disable_pf: false,
            complexity: 5,
            start: 0,
            end: NB_EBANDS,
            bitrate: BITRATE_MAX,
            vbr: false,
            constrained_vbr: true,
            loss_rate: 0,
            lsb_depth: 24,
            disable_inv: false,
            rng: 0,
            spread_decision: 0,
            delayed_intra: 0,
            tonal_average: 0,
            last_coded_bands: 0,
            hf_average: 0,
            tapset_decision: 0,
            prefilter_period: 0,
            prefilter_gain: 0,
            prefilter_tapset: 0,
            consec_transient: 0,
            preemph_mem_e: [0; 2],
            vbr_reservoir: 0,
            vbr_drift: 0,
            vbr_offset: 0,
            vbr_count: 0,
            overlap_max: 0,
            stereo_saving: 0,
            intensity: 0,
            spec_avg: 0,
            in_mem: vec![0; channels * OVERLAP],
            prefilter_mem: vec![0; channels * MAXPERIOD],
            old_band_e: [0; 2 * NB_EBANDS],
            old_log_e: [0; 2 * NB_EBANDS],
            old_log_e2: [0; 2 * NB_EBANDS],
            energy_error: [0; 2 * NB_EBANDS],
        };
        s.reset();
        s
    }

    /// `OPUS_RESET_STATE`: clears everything from `rng` on.
    pub fn reset(&mut self) {
        let cc = self.channels;
        *self = CeltEncoder {
            rng: 0,
            spread_decision: SPREAD_NORMAL,
            delayed_intra: 1,
            tonal_average: 256,
            last_coded_bands: 0,
            hf_average: 0,
            tapset_decision: 0,
            prefilter_period: 0,
            prefilter_gain: 0,
            prefilter_tapset: 0,
            consec_transient: 0,
            preemph_mem_e: [0; 2],
            vbr_reservoir: 0,
            vbr_drift: 0,
            vbr_offset: 0,
            vbr_count: 0,
            overlap_max: 0,
            stereo_saving: 0,
            intensity: 0,
            spec_avg: 0,
            in_mem: vec![0; cc * OVERLAP],
            prefilter_mem: vec![0; cc * MAXPERIOD],
            old_band_e: [0; 2 * NB_EBANDS],
            old_log_e: [-(28 << DB_SHIFT); 2 * NB_EBANDS],
            old_log_e2: [-(28 << DB_SHIFT); 2 * NB_EBANDS],
            energy_error: [0; 2 * NB_EBANDS],
            ..self.clone()
        };
    }

    /// `OPUS_SET_BITRATE` on the CELT encoder.
    pub fn set_bitrate(&mut self, value: i32) {
        self.bitrate = if value == BITRATE_MAX {
            value
        } else {
            value.min(260000 * self.channels as i32)
        };
    }

    /// `celt_encode_with_ec` with an existing range coder: encodes `frame_size` samples of
    /// interleaved `pcm` (`channels` per sample) into at most `nb_compressed_bytes` and returns
    /// the number of bytes used.
    pub fn encode_with_ec(
        &mut self,
        pcm: &[i16],
        frame_size: usize,
        nb_compressed_bytes: usize,
        enc: &mut Ec,
    ) -> Result<usize, ()> {
        let cc = self.channels;
        let c = self.stream_channels;
        let start = self.start;
        let end = self.end;
        let mut tf_estimate = 0i32;
        if nb_compressed_bytes < 2 {
            return Err(());
        }
        let lm: i32 = (0..=3).find(|&l| 120 << l == frame_size).ok_or(())?;
        let m = 1usize << lm;
        let n = m * 120;
        let mut tell = enc.tell();
        let nb_filled_bytes = (tell + 4) >> 3;
        let mut nb_compressed_bytes = nb_compressed_bytes.min(1275) as i32;
        let mut nb_available_bytes = nb_compressed_bytes - nb_filled_bytes;
        let vbr_rate;
        let mut effective_bytes;
        if self.vbr && self.bitrate != BITRATE_MAX {
            let den = 48000 >> BITRES;
            vbr_rate = (self.bitrate * frame_size as i32 + (den >> 1)) / den;
            effective_bytes = vbr_rate >> (3 + BITRES);
        } else {
            vbr_rate = 0;
            let mut tmp = self.bitrate.wrapping_mul(frame_size as i32);
            if tell > 1 {
                tmp += tell;
            }
            if self.bitrate != BITRATE_MAX {
                nb_compressed_bytes =
                    2.max(nb_compressed_bytes.min((tmp + 4 * 48000) / (8 * 48000)));
            }
            effective_bytes = nb_compressed_bytes - nb_filled_bytes;
        }
        let ci = c as i32;
        let mut equiv_rate =
            ((nb_compressed_bytes * 8 * 50) >> (3 - lm)) - (40 * ci + 20) * ((400 >> lm) - 50);
        if self.bitrate != BITRATE_MAX {
            equiv_rate = equiv_rate.min(self.bitrate - (40 * ci + 20) * ((400 >> lm) - 50));
        }
        if vbr_rate > 0 && self.constrained_vbr {
            let vbr_bound = vbr_rate;
            let max_allowed = (if tell == 1 { 2 } else { 0 })
                .max((vbr_rate + vbr_bound - self.vbr_reservoir) >> (BITRES + 3))
                .min(nb_available_bytes);
            if max_allowed < nb_available_bytes {
                nb_compressed_bytes = nb_filled_bytes + max_allowed;
                nb_available_bytes = max_allowed;
                enc.enc_shrink(nb_compressed_bytes as u32);
            }
        }
        let mut total_bits = nb_compressed_bytes * 8;
        let eff_end = end.min(NB_EBANDS);
        let mut inb = vec![0i32; cc * (n + OVERLAP)];
        let mut sample_max = self.overlap_max.max(maxabs16(&pcm[..c * (n - OVERLAP)]));
        self.overlap_max = maxabs16(&pcm[c * (n - OVERLAP)..c * n]);
        sample_max = sample_max.max(self.overlap_max);
        let mut silence = sample_max == 0;
        if tell == 1 {
            enc.enc_bit_logp(silence, 15);
        } else {
            silence = false;
        }
        if silence {
            if vbr_rate > 0 {
                nb_compressed_bytes = nb_compressed_bytes.min(nb_filled_bytes + 2);
                effective_bytes = nb_compressed_bytes;
                total_bits = nb_compressed_bytes * 8;
                nb_available_bytes = 2;
                enc.enc_shrink(nb_compressed_bytes as u32);
            }
            tell = nb_compressed_bytes * 8;
            enc.nbits_total += tell - enc.tell();
        }
        for ch in 0..cc {
            // celt_preemphasis, fast path (no clipping, no upsampling).
            let mut mm = self.preemph_mem_e[ch];
            let base = ch * (n + OVERLAP) + OVERLAP;
            for i in 0..n {
                let x = pcm[cc * i + ch] as i32;
                inb[base + i] = shl32(x, SIG_SHIFT).wrapping_sub(mm);
                mm = mul16(PREEMPH, x) >> (15 - SIG_SHIFT);
            }
            self.preemph_mem_e[ch] = mm;
        }
        let mut pitch_index = MINPERIOD as i32;
        let mut gain1 = 0i32;
        let prefilter_tapset;
        let pf_on;
        {
            let enabled = nb_available_bytes > 12 * ci
                && !silence
                && !self.disable_pf
                && self.complexity >= 5;
            prefilter_tapset = self.tapset_decision;
            let mut qg = 0;
            pf_on = self.run_prefilter(
                &mut inb,
                n,
                prefilter_tapset,
                &mut pitch_index,
                &mut gain1,
                &mut qg,
                enabled,
                nb_available_bytes,
            );
            if !pf_on {
                if tell + 16 <= total_bits {
                    enc.enc_bit_logp(false, 1);
                }
            } else {
                enc.enc_bit_logp(true, 1);
                pitch_index += 1;
                let octave = ilog(pitch_index as u32) - 5;
                enc.enc_uint(octave as u32, 6);
                enc.enc_bits((pitch_index - (16 << octave)) as u32, (4 + octave) as u32);
                pitch_index -= 1;
                enc.enc_bits(qg as u32, 3);
                enc.enc_icdf(prefilter_tapset as usize, &TAPSET_ICDF, 2);
            }
        }
        let mut is_transient = false;
        let mut short_blocks = 0usize;
        let mut tf_chan = 0usize;
        if self.complexity >= 1 {
            is_transient =
                transient_analysis(&inb, n + OVERLAP, cc, &mut tf_estimate, &mut tf_chan);
        }
        let mut transient_got_disabled = false;
        if lm > 0 && enc.tell() + 3 <= total_bits {
            if is_transient {
                short_blocks = m;
            }
        } else {
            is_transient = false;
            transient_got_disabled = true;
        }
        let mut freq = vec![0i32; cc * n];
        let mut band_e = [0i32; 2 * NB_EBANDS];
        let mut band_log_e = [0i16; 2 * NB_EBANDS];
        let mut band_log_e2 = [0i16; 2 * NB_EBANDS];
        let second_mdct = short_blocks != 0 && self.complexity >= 8;
        if second_mdct {
            compute_mdcts(0, &inb, &mut freq, c, cc, lm);
            compute_band_energies(&freq, &mut band_e, eff_end, c, lm);
            amp2log2(eff_end, end, &band_e, &mut band_log_e2, c);
            for v in band_log_e2[..c * NB_EBANDS].iter_mut() {
                *v = add16(*v as i32, shl16(lm, DB_SHIFT) >> 1) as i16;
            }
        }
        compute_mdcts(short_blocks, &inb, &mut freq, c, cc, lm);
        if cc == 2 && c == 1 {
            tf_chan = 0;
        }
        compute_band_energies(&freq, &mut band_e, eff_end, c, lm);
        amp2log2(eff_end, end, &band_e, &mut band_log_e, c);
        let temporal_vbr;
        {
            let mut follow = -(10 << DB_SHIFT);
            let mut frame_avg = 0i32;
            let offset = if short_blocks != 0 {
                shl16(lm, DB_SHIFT) >> 1
            } else {
                0
            };
            for i in start..end {
                follow =
                    ((follow - (1 << DB_SHIFT)).max(band_log_e[i] as i32 - offset)) as i16 as i32;
                if c == 2 {
                    follow = follow.max(band_log_e[i + NB_EBANDS] as i32 - offset) as i16 as i32;
                }
                frame_avg += follow;
            }
            frame_avg /= (end - start) as i32;
            let tv = sub16(frame_avg, self.spec_avg) as i16 as i32;
            temporal_vbr = (3 << DB_SHIFT).min((-1536).max(tv));
            self.spec_avg = add16(self.spec_avg, mul16_q15(655, temporal_vbr));
        }
        if !second_mdct {
            band_log_e2[..c * NB_EBANDS].copy_from_slice(&band_log_e[..c * NB_EBANDS]);
        }
        if lm > 0
            && enc.tell() + 3 <= total_bits
            && !is_transient
            && self.complexity >= 5
            && patch_transient_decision(&band_log_e, &self.old_band_e, start, end, c)
        {
            is_transient = true;
            short_blocks = m;
            compute_mdcts(short_blocks, &inb, &mut freq, c, cc, lm);
            compute_band_energies(&freq, &mut band_e, eff_end, c, lm);
            amp2log2(eff_end, end, &band_e, &mut band_log_e, c);
            for v in band_log_e2[..c * NB_EBANDS].iter_mut() {
                *v = add16(*v as i32, shl16(lm, DB_SHIFT) >> 1) as i16;
            }
            tf_estimate = 3277;
        }
        if lm > 0 && enc.tell() + 3 <= total_bits {
            enc.enc_bit_logp(is_transient, 3);
        }
        let mut x = vec![0i16; c * n];
        normalise_bands(&freq, &mut x, &band_e, eff_end, c, m);
        let enable_tf_analysis = effective_bytes >= 15 * ci && self.complexity >= 2;
        let mut offsets = [0i32; NB_EBANDS];
        let mut importance = [0i32; NB_EBANDS];
        let mut spread_weight = [0i32; NB_EBANDS];
        let mut tot_boost = 0i32;
        let max_depth = dynalloc_analysis(
            &band_log_e,
            &band_log_e2,
            start,
            end,
            c,
            &mut offsets,
            self.lsb_depth,
            is_transient,
            self.vbr,
            self.constrained_vbr,
            lm,
            effective_bytes,
            &mut tot_boost,
            &mut importance,
            &mut spread_weight,
        );
        let mut tf_res = [0i32; NB_EBANDS];
        let tf_select;
        if enable_tf_analysis {
            let lambda = 80.max(20480 / effective_bytes + 2);
            tf_select = tf_analysis(
                eff_end,
                is_transient,
                &mut tf_res,
                lambda,
                &x,
                n,
                lm,
                tf_estimate,
                tf_chan,
                &importance,
            );
            for i in eff_end..end {
                tf_res[i] = tf_res[eff_end - 1];
            }
        } else {
            for v in tf_res[..end].iter_mut() {
                *v = is_transient as i32;
            }
            tf_select = 0;
        }
        let mut error = [0i16; 2 * NB_EBANDS];
        for ch in 0..c {
            for i in start..end {
                let k = i + ch * NB_EBANDS;
                if (band_log_e[k] as i32 - self.old_band_e[k] as i32).abs() < (2 << DB_SHIFT) {
                    band_log_e[k] = (band_log_e[k] as i32
                        - mul16_q15(self.energy_error[k] as i32, 8192))
                        as i16;
                }
            }
        }
        quant_coarse(
            start,
            end,
            eff_end,
            &band_log_e,
            &mut self.old_band_e,
            total_bits as u32,
            &mut error,
            enc,
            c,
            lm,
            nb_available_bytes,
            self.force_intra,
            &mut self.delayed_intra,
            self.complexity >= 4,
            self.loss_rate,
        );
        tf_encode(start, end, is_transient, &mut tf_res, lm, tf_select, enc);
        if enc.tell() + 4 <= total_bits {
            if short_blocks != 0 || self.complexity < 3 || nb_available_bytes < 10 * ci {
                self.spread_decision = if self.complexity == 0 {
                    SPREAD_NONE
                } else {
                    SPREAD_NORMAL
                };
            } else {
                self.spread_decision = spreading_decision(
                    &x,
                    &mut self.tonal_average,
                    self.spread_decision,
                    &mut self.hf_average,
                    &mut self.tapset_decision,
                    pf_on && short_blocks == 0,
                    eff_end,
                    c,
                    m,
                    &spread_weight,
                );
            }
            enc.enc_icdf(self.spread_decision as usize, &SPREAD_ICDF, 5);
        }
        let mut cap = [0i32; NB_EBANDS];
        init_caps(&mut cap, lm, ci);
        let mut dynalloc_logp = 6;
        total_bits <<= BITRES;
        let mut total_boost = 0i32;
        let mut tellf = enc.tell_frac() as i32;
        for i in start..end {
            let width = (ci * (eb(i + 1) - eb(i))) << lm;
            let quanta = (width << BITRES).min((6 << BITRES).max(width));
            let mut loop_logp = dynalloc_logp;
            let mut boost = 0;
            let mut j = 0;
            while tellf + (loop_logp << BITRES) < total_bits - total_boost && boost < cap[i] {
                let flag = j < offsets[i];
                enc.enc_bit_logp(flag, loop_logp as u32);
                tellf = enc.tell_frac() as i32;
                if !flag {
                    break;
                }
                boost += quanta;
                total_boost += quanta;
                loop_logp = 1;
                j += 1;
            }
            if j > 0 {
                dynalloc_logp = 2.max(dynalloc_logp - 1);
            }
            offsets[i] = boost;
        }
        let mut dual_stereo = false;
        if c == 2 {
            const THRESH: [i32; 21] = [
                1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 36, 44, 50, 56, 62, 67, 72, 79, 88, 106, 134,
            ];
            const HYST: [i32; 21] = [
                1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 5, 6, 8, 8,
            ];
            if lm != 0 {
                dual_stereo = stereo_analysis(&x, lm, n);
            }
            self.intensity = hysteresis_decision(
                (equiv_rate / 1000) as i16 as i32,
                &THRESH,
                &HYST,
                self.intensity,
            );
            self.intensity = (end as i32).min((start as i32).max(self.intensity));
        }
        let mut alloc_trim = 5;
        if tellf + (6 << BITRES) <= total_bits - total_boost {
            alloc_trim = alloc_trim_analysis(
                &x,
                &band_log_e,
                end,
                lm,
                c,
                n,
                &mut self.stereo_saving,
                tf_estimate,
                self.intensity,
                equiv_rate,
            );
            enc.enc_icdf(alloc_trim as usize, &TRIM_ICDF, 7);
            tellf = enc.tell_frac() as i32;
        }
        if vbr_rate > 0 {
            let lm_diff = 3 - lm;
            nb_compressed_bytes = nb_compressed_bytes.min(1275 >> (3 - lm));
            let mut base_target = vbr_rate - ((40 * ci + 20) << BITRES);
            if self.constrained_vbr {
                base_target += self.vbr_offset >> lm_diff;
            }
            let mut target = compute_vbr(
                base_target,
                lm,
                equiv_rate,
                self.last_coded_bands,
                c,
                self.intensity,
                self.constrained_vbr,
                self.stereo_saving,
                tot_boost,
                tf_estimate,
                max_depth,
                temporal_vbr,
            );
            target += tellf;
            let min_allowed = ((tellf + total_boost + (1 << (BITRES + 3)) - 1) >> (BITRES + 3)) + 2;
            nb_available_bytes = (target + (1 << (BITRES + 2))) >> (BITRES + 3);
            nb_available_bytes = min_allowed.max(nb_available_bytes);
            nb_available_bytes = nb_compressed_bytes.min(nb_available_bytes);
            let mut delta = target - vbr_rate;
            target = nb_available_bytes << (BITRES + 3);
            if silence {
                nb_available_bytes = 2;
                target = (2 * 8) << BITRES;
                delta = 0;
            }
            let alpha = if self.vbr_count < 970 {
                self.vbr_count += 1;
                rcp(shl32(self.vbr_count + 20, 16)) as i16 as i32
            } else {
                33
            };
            if self.constrained_vbr {
                self.vbr_reservoir += target - vbr_rate;
                self.vbr_drift += mul16_32_q15(
                    alpha,
                    (delta * (1 << lm_diff)) - self.vbr_offset - self.vbr_drift,
                );
                self.vbr_offset = -self.vbr_drift;
                if self.vbr_reservoir < 0 {
                    let adjust = (-self.vbr_reservoir) / (8 << BITRES);
                    nb_available_bytes += if silence { 0 } else { adjust };
                    self.vbr_reservoir = 0;
                }
            }
            nb_compressed_bytes = nb_compressed_bytes.min(nb_available_bytes);
            enc.enc_shrink(nb_compressed_bytes as u32);
        }
        let mut fine_quant = [0i32; NB_EBANDS];
        let mut pulses = [0i32; NB_EBANDS];
        let mut fine_priority = [0i32; NB_EBANDS];
        let mut bits = ((nb_compressed_bytes * 8) << BITRES) - enc.tell_frac() as i32 - 1;
        let anti_collapse_rsv = if is_transient && lm >= 2 && bits >= (lm + 2) << BITRES {
            1 << BITRES
        } else {
            0
        };
        bits -= anti_collapse_rsv;
        let signal_bandwidth = end as i32 - 1;
        let mut intensity = self.intensity;
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
            ci,
            lm,
            enc,
            true,
            self.last_coded_bands,
            signal_bandwidth,
        );
        self.intensity = alloc.intensity;
        let dual_stereo = alloc.dual_stereo;
        let coded_bands = alloc.coded_bands;
        let balance = alloc.balance;
        if self.last_coded_bands != 0 {
            self.last_coded_bands =
                (self.last_coded_bands + 1).min((self.last_coded_bands - 1).max(coded_bands));
        } else {
            self.last_coded_bands = coded_bands;
        }
        quant_fine(
            start,
            end,
            &mut self.old_band_e,
            &mut error,
            &fine_quant,
            enc,
            c,
        );
        let mut collapse_masks = [0u8; 2 * NB_EBANDS];
        let mut seed = self.rng;
        quant_all_bands(
            AllBands {
                encode: true,
                start,
                end,
                collapse_masks: &mut collapse_masks[..c * NB_EBANDS],
                band_e: &band_e,
                pulses: &pulses,
                short_blocks: short_blocks != 0,
                spread: self.spread_decision,
                dual_stereo,
                intensity: self.intensity,
                tf_res: &tf_res,
                total_bits: nb_compressed_bytes * (8 << BITRES) - anti_collapse_rsv,
                balance,
                lm,
                coded_bands,
                disable_inv: self.disable_inv,
                complexity: self.complexity,
                norm: &mut [0; 2 * 8 * 78],
            },
            &mut x,
            c,
            enc,
            &mut seed,
        );
        if anti_collapse_rsv > 0 {
            let anti_collapse_on = self.consec_transient < 2;
            enc.enc_bits(anti_collapse_on as u32, 1);
        }
        let bits_left = nb_compressed_bytes * 8 - enc.tell();
        quant_finalise(
            start,
            end,
            &mut self.old_band_e,
            &mut error,
            &fine_quant,
            &fine_priority,
            bits_left,
            enc,
            c,
        );
        self.energy_error = [0; 2 * NB_EBANDS];
        for ch in 0..c {
            for i in start..end {
                let k = i + ch * NB_EBANDS;
                self.energy_error[k] = (error[k] as i32).clamp(-16384, 16384) as i16;
            }
        }
        if silence {
            for v in self.old_band_e[..c * NB_EBANDS].iter_mut() {
                *v = -(28 << DB_SHIFT);
            }
        }
        self.prefilter_period = pitch_index;
        self.prefilter_gain = gain1;
        self.prefilter_tapset = prefilter_tapset;
        if cc == 2 && c == 1 {
            let (a, b) = self.old_band_e.split_at_mut(NB_EBANDS);
            b.copy_from_slice(a);
        }
        let ccn = cc * NB_EBANDS;
        if !is_transient {
            self.old_log_e2[..ccn].copy_from_slice(&self.old_log_e[..ccn]);
            self.old_log_e[..ccn].copy_from_slice(&self.old_band_e[..ccn]);
        } else {
            for i in 0..ccn {
                self.old_log_e[i] = self.old_log_e[i].min(self.old_band_e[i]);
            }
        }
        for ch in 0..cc {
            for i in (0..start).chain(end..NB_EBANDS) {
                let k = ch * NB_EBANDS + i;
                self.old_band_e[k] = 0;
                self.old_log_e[k] = -(28 << DB_SHIFT);
                self.old_log_e2[k] = -(28 << DB_SHIFT);
            }
        }
        if is_transient || transient_got_disabled {
            self.consec_transient += 1;
        } else {
            self.consec_transient = 0;
        }
        self.rng = enc.rng;
        enc.enc_done();
        if enc.error != 0 {
            return Err(());
        }
        Ok(nb_compressed_bytes as usize)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_prefilter(
        &mut self,
        inb: &mut [i32],
        n: usize,
        prefilter_tapset: i32,
        pitch: &mut i32,
        gain: &mut i32,
        qgain: &mut i32,
        enabled: bool,
        nb_available_bytes: i32,
    ) -> bool {
        let cc = self.channels;
        let mut pre = vec![vec![0i32; n + MAXPERIOD]; cc];
        for (ch, p) in pre.iter_mut().enumerate() {
            p[..MAXPERIOD]
                .copy_from_slice(&self.prefilter_mem[ch * MAXPERIOD..(ch + 1) * MAXPERIOD]);
            let base = ch * (n + OVERLAP) + OVERLAP;
            p[MAXPERIOD..].copy_from_slice(&inb[base..base + n]);
        }
        let mut pitch_index;
        let mut gain1;
        if enabled {
            let mut pitch_buf = vec![0i16; (MAXPERIOD + n) >> 1];
            let refs: Vec<&[i32]> = pre.iter().map(|p| p.as_slice()).collect();
            pitch_downsample(&refs, &mut pitch_buf, MAXPERIOD + n);
            pitch_index = pitch_search(
                &pitch_buf[MAXPERIOD >> 1..],
                &pitch_buf,
                n,
                MAXPERIOD - 3 * MINPERIOD,
            ) as i32;
            pitch_index = MAXPERIOD as i32 - pitch_index;
            gain1 = remove_doubling(
                &pitch_buf,
                MAXPERIOD,
                MINPERIOD,
                n,
                &mut pitch_index,
                self.prefilter_period,
                self.prefilter_gain,
            );
            if pitch_index > MAXPERIOD as i32 - 2 {
                pitch_index = MAXPERIOD as i32 - 2;
            }
            gain1 = mul16_q15(22938, gain1);
            if self.loss_rate > 2 {
                gain1 >>= 1;
            }
            if self.loss_rate > 4 {
                gain1 >>= 1;
            }
            if self.loss_rate > 8 {
                gain1 = 0;
            }
        } else {
            gain1 = 0;
            pitch_index = MINPERIOD as i32;
        }
        let mut pf_threshold = 6554;
        if (pitch_index - self.prefilter_period).abs() * 10 > pitch_index {
            pf_threshold += 6554;
        }
        if nb_available_bytes < 25 {
            pf_threshold += 3277;
        }
        if nb_available_bytes < 35 {
            pf_threshold += 3277;
        }
        if self.prefilter_gain > 13107 {
            pf_threshold -= 3277;
        }
        if self.prefilter_gain > 18022 {
            pf_threshold -= 3277;
        }
        pf_threshold = pf_threshold.max(6554);
        let pf_on;
        let qg;
        if gain1 < pf_threshold {
            gain1 = 0;
            pf_on = false;
            qg = 0;
        } else {
            if (gain1 - self.prefilter_gain).abs() < 3277 {
                gain1 = self.prefilter_gain;
            }
            qg = ((((gain1 + 1536) >> 10) / 3) - 1).clamp(0, 7);
            gain1 = 3072 * (qg + 1);
            pf_on = true;
        }
        for (ch, p) in pre.iter().enumerate() {
            self.prefilter_period = self.prefilter_period.max(COMBFILTER_MINPERIOD);
            let base = ch * (n + OVERLAP);
            inb[base..base + OVERLAP]
                .copy_from_slice(&self.in_mem[ch * OVERLAP..(ch + 1) * OVERLAP]);
            comb_filter_to(
                &mut inb[base + OVERLAP..base + OVERLAP + n],
                p,
                MAXPERIOD,
                n,
                Comb {
                    t0: self.prefilter_period,
                    t1: pitch_index,
                    g0: -self.prefilter_gain,
                    g1: -gain1,
                    tapset0: self.prefilter_tapset as usize,
                    tapset1: prefilter_tapset as usize,
                },
                &WINDOW,
                OVERLAP,
            );
            self.in_mem[ch * OVERLAP..(ch + 1) * OVERLAP]
                .copy_from_slice(&inb[base + n..base + n + OVERLAP]);
            let mem = &mut self.prefilter_mem[ch * MAXPERIOD..(ch + 1) * MAXPERIOD];
            if n > MAXPERIOD {
                mem.copy_from_slice(&p[n..n + MAXPERIOD]);
            } else {
                mem.copy_within(n.., 0);
                mem[MAXPERIOD - n..].copy_from_slice(&p[MAXPERIOD..MAXPERIOD + n]);
            }
        }
        *gain = gain1;
        *pitch = pitch_index;
        *qgain = qg;
        pf_on
    }
}

fn compute_mdcts(short_blocks: usize, inb: &[i32], out: &mut [i32], c: usize, cc: usize, lm: i32) {
    let (b, n, shift) = if short_blocks != 0 {
        (short_blocks, 120usize, 3usize)
    } else {
        (1, 120usize << lm, (3 - lm) as usize)
    };
    for ch in 0..cc {
        for blk in 0..b {
            let ib = ch * (b * n + OVERLAP) + blk * n;
            let ob = blk + ch * n * b;
            mdct_forward(&inb[ib..], &mut out[ob..], &WINDOW, OVERLAP, shift, b);
        }
    }
    if cc == 2 && c == 1 {
        for i in 0..b * n {
            out[i] = (out[i] >> 1).wrapping_add(out[b * n + i] >> 1);
        }
    }
}

fn transient_analysis(
    inb: &[i32],
    len: usize,
    c: usize,
    tf_estimate: &mut i32,
    tf_chan: &mut usize,
) -> bool {
    const INV_TABLE: [u8; 128] = [
        255, 255, 156, 110, 86, 70, 59, 51, 45, 40, 37, 33, 31, 28, 26, 25, 23, 22, 21, 20, 19, 18,
        17, 16, 16, 15, 15, 14, 13, 13, 12, 12, 12, 12, 11, 11, 11, 10, 10, 10, 9, 9, 9, 9, 9, 9,
        8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2,
    ];
    let mut tmp = vec![0i16; len];
    let forward_shift = 4;
    let len2 = len / 2;
    let mut mask_metric = 0i32;
    for ch in 0..c {
        let mut mem0 = 0i32;
        let mut mem1 = 0i32;
        for i in 0..len {
            let x = inb[i + ch * len] >> SIG_SHIFT;
            let y = mem0.wrapping_add(x);
            mem0 = mem1.wrapping_add(y).wrapping_sub(shl32(x, 1));
            mem1 = x.wrapping_sub(y >> 1);
            tmp[i] = sround16(y, 2);
        }
        tmp[..12].fill(0);
        let shift = 14 - ilog2(1.max(maxabs16(&tmp)));
        if shift != 0 {
            for v in tmp.iter_mut() {
                *v = shl16(*v as i32, shift) as i16;
            }
        }
        let mut mean = 0i32;
        mem0 = 0;
        for i in 0..len2 {
            let x2 = pshr32(
                mul16(tmp[2 * i] as i32, tmp[2 * i] as i32)
                    + mul16(tmp[2 * i + 1] as i32, tmp[2 * i + 1] as i32),
                16,
            ) as i16 as i32;
            mean += x2;
            tmp[i] = (mem0 + pshr32(x2 - mem0, forward_shift)) as i16;
            mem0 = tmp[i] as i32;
        }
        mem0 = 0;
        let mut max_e = 0i32;
        for i in (0..len2).rev() {
            tmp[i] = (mem0 + pshr32(tmp[i] as i32 - mem0, 3)) as i16;
            mem0 = tmp[i] as i32;
            max_e = max_e.max(mem0);
        }
        let mean = mul16(sqrt(mean), sqrt(mul16(max_e, (len2 >> 1) as i32)));
        let norm = shl32(len2 as i32, 6 + 14) / (EPSILON + (mean >> 1));
        let mut unmask = 0i32;
        let mut i = 12;
        while i < len2 - 5 {
            let id = 0.max(127.min(mul16_32_q15(tmp[i] as i32 + EPSILON, norm)));
            unmask += INV_TABLE[id as usize] as i32;
            i += 4;
        }
        let unmask = 64 * unmask * 4 / (6 * (len2 as i32 - 17));
        if unmask > mask_metric {
            *tf_chan = ch;
            mask_metric = unmask;
        }
    }
    let is_transient = mask_metric > 200;
    let tf_max = 0.max(sqrt(27 * mask_metric) - 42) as i16 as i32;
    *tf_estimate = sqrt(0.max(shl32(mul16(113, 163.min(tf_max)), 14) - 37312528)) as i16 as i32;
    is_transient
}

fn patch_transient_decision(
    new_e: &[i16],
    old_e: &[i16],
    start: usize,
    end: usize,
    c: usize,
) -> bool {
    let mut spread_old = [0i32; 26];
    if c == 1 {
        spread_old[start] = old_e[start] as i32;
        for i in start + 1..end {
            spread_old[i] = ((spread_old[i - 1] - 1024).max(old_e[i] as i32)) as i16 as i32;
        }
    } else {
        spread_old[start] = (old_e[start] as i32).max(old_e[start + NB_EBANDS] as i32);
        for i in start + 1..end {
            spread_old[i] = ((spread_old[i - 1] - 1024)
                .max((old_e[i] as i32).max(old_e[i + NB_EBANDS] as i32)))
                as i16 as i32;
        }
    }
    for i in (start..end - 1).rev() {
        spread_old[i] = spread_old[i].max(spread_old[i + 1] - 1024) as i16 as i32;
    }
    let mut mean_diff = 0i32;
    for ch in 0..c {
        for i in 2.max(start)..end - 1 {
            let x1 = 0.max(new_e[i + ch * NB_EBANDS] as i32);
            let x2 = 0.max(spread_old[i]);
            mean_diff += 0.max(sub16(x1, x2));
        }
    }
    mean_diff /= (c * (end - 1 - 2.max(start))) as i32;
    mean_diff > 1024
}

fn l1_metric(tmp: &[i16], lm: i32, bias: i32) -> i32 {
    let l1: i32 = tmp.iter().map(|&v| (v as i32).abs()).sum();
    l1.wrapping_add(mul16_32_q15(lm * bias, l1))
}

#[allow(clippy::too_many_arguments)]
fn tf_analysis(
    len: usize,
    is_transient: bool,
    tf_res: &mut [i32],
    lambda: i32,
    x: &[i16],
    n0: usize,
    lm: i32,
    tf_estimate: i32,
    tf_chan: usize,
    importance: &[i32],
) -> i32 {
    let it = is_transient as usize;
    let bias = mul16_q14(1311, (-4096).max(8192 - tf_estimate));
    let mut metric = [0i32; NB_EBANDS];
    let mut path0 = [0i32; NB_EBANDS];
    let mut path1 = [0i32; NB_EBANDS];
    for i in 0..len {
        let n = ((eb(i + 1) - eb(i)) << lm) as usize;
        let narrow = eb(i + 1) - eb(i) == 1;
        let off = tf_chan * n0 + ((eb(i) << lm) as usize);
        let mut tmp = x[off..off + n].to_vec();
        let mut l1 = l1_metric(&tmp, if is_transient { lm } else { 0 }, bias);
        let mut best_l1 = l1;
        let mut best_level = 0;
        if is_transient && !narrow {
            let mut tmp1 = tmp.clone();
            haar1(&mut tmp1, n >> lm, 1 << lm);
            l1 = l1_metric(&tmp1, lm + 1, bias);
            if l1 < best_l1 {
                best_l1 = l1;
                best_level = -1;
            }
        }
        let kmax = lm + !(is_transient || narrow) as i32;
        for k in 0..kmax {
            let b = if is_transient { lm - k - 1 } else { k + 1 };
            haar1(&mut tmp, n >> k, 1 << k);
            l1 = l1_metric(&tmp, b, bias);
            if l1 < best_l1 {
                best_l1 = l1;
                best_level = k + 1;
            }
        }
        metric[i] = if is_transient {
            2 * best_level
        } else {
            -2 * best_level
        };
        if narrow && (metric[i] == 0 || metric[i] == -2 * lm) {
            metric[i] -= 1;
        }
    }
    let tbl = &TF_SELECT_TABLE[lm as usize];
    let sel_val = |sel: usize, v: usize| 2 * tbl[4 * it + 2 * sel + v] as i32;
    let mut selcost = [0i32; 2];
    for (sel, cost) in selcost.iter_mut().enumerate() {
        let mut cost0 = importance[0] * (metric[0] - sel_val(sel, 0)).abs();
        let mut cost1 = importance[0] * (metric[0] - sel_val(sel, 1)).abs()
            + if is_transient { 0 } else { lambda };
        for i in 1..len {
            let curr0 = cost0.min(cost1 + lambda);
            let curr1 = (cost0 + lambda).min(cost1);
            cost0 = curr0 + importance[i] * (metric[i] - sel_val(sel, 0)).abs();
            cost1 = curr1 + importance[i] * (metric[i] - sel_val(sel, 1)).abs();
        }
        *cost = cost0.min(cost1);
    }
    let tf_select = (selcost[1] < selcost[0] && is_transient) as usize;
    let mut cost0 = importance[0] * (metric[0] - sel_val(tf_select, 0)).abs();
    let mut cost1 = importance[0] * (metric[0] - sel_val(tf_select, 1)).abs()
        + if is_transient { 0 } else { lambda };
    for i in 1..len {
        let (curr0, curr1);
        let (from0, from1) = (cost0, cost1 + lambda);
        if from0 < from1 {
            curr0 = from0;
            path0[i] = 0;
        } else {
            curr0 = from1;
            path0[i] = 1;
        }
        let (from0, from1) = (cost0 + lambda, cost1);
        if from0 < from1 {
            curr1 = from0;
            path1[i] = 0;
        } else {
            curr1 = from1;
            path1[i] = 1;
        }
        cost0 = curr0 + importance[i] * (metric[i] - sel_val(tf_select, 0)).abs();
        cost1 = curr1 + importance[i] * (metric[i] - sel_val(tf_select, 1)).abs();
    }
    tf_res[len - 1] = if cost0 < cost1 { 0 } else { 1 };
    for i in (0..len - 1).rev() {
        tf_res[i] = if tf_res[i + 1] == 1 {
            path1[i + 1]
        } else {
            path0[i + 1]
        };
    }
    tf_select as i32
}

fn tf_encode(
    start: usize,
    end: usize,
    is_transient: bool,
    tf_res: &mut [i32],
    lm: i32,
    tf_select: i32,
    enc: &mut Ec,
) {
    let mut budget = enc.storage * 8;
    let mut tell = enc.tell() as u32;
    let mut logp: u32 = if is_transient { 2 } else { 4 };
    let tf_select_rsv = lm > 0 && tell + logp < budget;
    budget -= tf_select_rsv as u32;
    let mut curr = 0;
    let mut tf_changed = 0;
    for v in tf_res[start..end].iter_mut() {
        if tell + logp <= budget {
            enc.enc_bit_logp((*v ^ curr) != 0, logp);
            tell = enc.tell() as u32;
            curr = *v;
            tf_changed |= curr;
        } else {
            *v = curr;
        }
        logp = if is_transient { 4 } else { 5 };
    }
    let it = 4 * is_transient as usize;
    let tbl = &TF_SELECT_TABLE[lm as usize];
    let tf_select =
        if tf_select_rsv && tbl[it + tf_changed as usize] != tbl[it + 2 + tf_changed as usize] {
            enc.enc_bit_logp(tf_select != 0, 1);
            tf_select
        } else {
            0
        };
    for v in tf_res[start..end].iter_mut() {
        *v = tbl[it + 2 * tf_select as usize + *v as usize] as i32;
    }
}

#[allow(clippy::too_many_arguments)]
fn alloc_trim_analysis(
    x: &[i16],
    band_log_e: &[i16],
    end: usize,
    lm: i32,
    c: usize,
    n0: usize,
    stereo_saving: &mut i32,
    tf_estimate: i32,
    intensity: i32,
    equiv_rate: i32,
) -> i32 {
    let mut trim: i32 = 1280;
    if equiv_rate < 64000 {
        trim = 1024;
    } else if equiv_rate < 80000 {
        let frac = (equiv_rate - 64000) >> 10;
        trim = 1024 + 16 * frac;
    }
    let band = |i: usize| {
        let lo = (eb(i) << lm) as usize;
        let hi = (eb(i + 1) << lm) as usize;
        inner_prod(&x[lo..hi], &x[n0 + lo..n0 + hi])
    };
    if c == 2 {
        let mut sum = 0i32;
        for i in 0..8 {
            sum = add16(sum, band(i) >> 18);
        }
        sum = mul16_q15(4096, sum);
        sum = 1024.min((sum as i16 as i32).abs());
        let mut min_xc = sum;
        for i in 8..intensity.max(0) as usize {
            min_xc = min_xc.min(((band(i) >> 18) as i16 as i32).abs());
        }
        min_xc = 1024.min(min_xc.abs());
        let mut log_xc = log2(1049625 - mul16(sum, sum));
        let log_xc2 = (log_xc >> 1).max(log2(1049625 - mul16(min_xc, min_xc)));
        log_xc = pshr32(log_xc - 6144, DB_SHIFT - 8) as i16 as i32;
        let log_xc2 = pshr32(log_xc2 - 6144, DB_SHIFT - 8) as i16 as i32;
        trim = add16(trim, (-1024).max(mul16_q15(24576, log_xc)));
        *stereo_saving = (*stereo_saving + 64).min(-(log_xc2 >> 1)) as i16 as i32;
    }
    let mut diff = 0i32;
    for ch in 0..c {
        for i in 0..end - 1 {
            diff += band_log_e[i + ch * NB_EBANDS] as i32 * (2 + 2 * i as i32 - end as i32);
        }
    }
    diff /= (c * (end - 1)) as i32;
    trim = (trim - (-512).max(512.min(((diff + 1024) >> (DB_SHIFT - 8)) / 6))) as i16 as i32;
    trim = (trim - 2 * (tf_estimate >> (14 - 8))) as i16 as i32;
    let trim_index = pshr32(trim, 8);
    trim_index.clamp(0, 10)
}

fn stereo_analysis(x: &[i16], lm: i32, n0: usize) -> bool {
    let mut sum_lr = EPSILON;
    let mut sum_ms = EPSILON;
    for i in 0..13 {
        for j in (eb(i) << lm) as usize..(eb(i + 1) << lm) as usize {
            let l = x[j] as i32;
            let r = x[n0 + j] as i32;
            let m = l + r;
            let s = l - r;
            sum_lr = sum_lr.wrapping_add(l.abs() + r.abs());
            sum_ms = sum_ms.wrapping_add(m.abs() + s.abs());
        }
    }
    sum_ms = mul16_32_q15(23170, sum_ms);
    let mut thetas = 13;
    if lm <= 1 {
        thetas -= 8;
    }
    mul16_32_q15((eb(13) << (lm + 1)) + thetas, sum_ms) > mul16_32_q15(eb(13) << (lm + 1), sum_lr)
}

fn median_of_5(x: &[i16]) -> i32 {
    let x: [i32; 5] = [
        x[0] as i32,
        x[1] as i32,
        x[2] as i32,
        x[3] as i32,
        x[4] as i32,
    ];
    let t2 = x[2];
    let (mut t0, mut t1) = if x[0] > x[1] {
        (x[1], x[0])
    } else {
        (x[0], x[1])
    };
    let (mut t3, mut t4) = if x[3] > x[4] {
        (x[4], x[3])
    } else {
        (x[3], x[4])
    };
    if t0 > t3 {
        std::mem::swap(&mut t0, &mut t3);
        std::mem::swap(&mut t1, &mut t4);
    }
    if t2 > t1 {
        if t1 < t3 { t2.min(t3) } else { t4.min(t1) }
    } else if t2 < t3 {
        t1.min(t3)
    } else {
        t2.min(t4)
    }
}

fn median_of_3(x: &[i16]) -> i32 {
    let (t0, t1) = if x[0] > x[1] {
        (x[1] as i32, x[0] as i32)
    } else {
        (x[0] as i32, x[1] as i32)
    };
    let t2 = x[2] as i32;
    if t1 < t2 {
        t1
    } else if t0 < t2 {
        t2
    } else {
        t0
    }
}

#[allow(clippy::too_many_arguments)]
fn dynalloc_analysis(
    band_log_e: &[i16],
    band_log_e2: &[i16],
    start: usize,
    end: usize,
    c: usize,
    offsets: &mut [i32],
    lsb_depth: i32,
    is_transient: bool,
    vbr: bool,
    constrained_vbr: bool,
    lm: i32,
    effective_bytes: i32,
    tot_boost_out: &mut i32,
    importance: &mut [i32],
    spread_weight: &mut [i32],
) -> i32 {
    let mut tot_boost = 0i32;
    let mut follower = [0i16; 2 * NB_EBANDS];
    let mut noise_floor = [0i16; NB_EBANDS];
    offsets[..NB_EBANDS].fill(0);
    let mut max_depth = -32666i32;
    for i in 0..end {
        noise_floor[i] = (mul16(64, LOG_N[i] as i32) + 512 + shl16(9 - lsb_depth, DB_SHIFT)
            - shl16(E_MEANS[i] as i32, 6)
            + mul16(6, ((i + 5) * (i + 5)) as i32)) as i16;
    }
    for ch in 0..c {
        for i in 0..end {
            max_depth = max_depth.max(band_log_e[ch * NB_EBANDS + i] as i32 - noise_floor[i] as i32)
                as i16 as i32;
        }
    }
    {
        let mut mask = [0i16; NB_EBANDS];
        let mut sig = [0i16; NB_EBANDS];
        for i in 0..end {
            mask[i] = (band_log_e[i] as i32 - noise_floor[i] as i32) as i16;
        }
        if c == 2 {
            for i in 0..end {
                mask[i] = (mask[i] as i32)
                    .max(band_log_e[NB_EBANDS + i] as i32 - noise_floor[i] as i32)
                    as i16;
            }
        }
        sig[..end].copy_from_slice(&mask[..end]);
        for i in 1..end {
            mask[i] = (mask[i] as i32).max(mask[i - 1] as i32 - 2048) as i16;
        }
        for i in (0..end - 1).rev() {
            mask[i] = (mask[i] as i32).max(mask[i + 1] as i32 - 3072) as i16;
        }
        for i in 0..end {
            let smr = (sig[i] as i32 - 0.max(max_depth - 12288).max(mask[i] as i32)) as i16 as i32;
            let shift = -pshr32((-5120).max(0.min(smr)), DB_SHIFT);
            spread_weight[i] = 32 >> shift;
        }
    }
    if effective_bytes > 50 && lm >= 1 {
        let mut last = 0usize;
        for ch in 0..c {
            let f = &mut follower[ch * NB_EBANDS..(ch + 1) * NB_EBANDS];
            let e2 = &band_log_e2[ch * NB_EBANDS..(ch + 1) * NB_EBANDS];
            f[0] = e2[0];
            for i in 1..end {
                if e2[i] as i32 > e2[i - 1] as i32 + 512 {
                    last = i;
                }
                f[i] = (f[i - 1] as i32 + 1536).min(e2[i] as i32) as i16;
            }
            for i in (0..last).rev() {
                f[i] = (f[i] as i32).min((f[i + 1] as i32 + 2048).min(e2[i] as i32)) as i16;
            }
            let offset = 1024;
            for i in 2..end - 2 {
                f[i] = (f[i] as i32).max(median_of_5(&e2[i - 2..]) - offset) as i16;
            }
            let tmp = median_of_3(&e2[0..]) - offset;
            f[0] = (f[0] as i32).max(tmp) as i16;
            f[1] = (f[1] as i32).max(tmp) as i16;
            let tmp = median_of_3(&e2[end - 3..]) - offset;
            f[end - 2] = (f[end - 2] as i32).max(tmp) as i16;
            f[end - 1] = (f[end - 1] as i32).max(tmp) as i16;
            for i in 0..end {
                f[i] = f[i].max(noise_floor[i]);
            }
        }
        if c == 2 {
            for i in start..end {
                follower[NB_EBANDS + i] =
                    (follower[NB_EBANDS + i] as i32).max(follower[i] as i32 - 4096) as i16;
                follower[i] =
                    (follower[i] as i32).max(follower[NB_EBANDS + i] as i32 - 4096) as i16;
                follower[i] = ((0.max(band_log_e[i] as i32 - follower[i] as i32)
                    + 0.max(band_log_e[NB_EBANDS + i] as i32 - follower[NB_EBANDS + i] as i32))
                    >> 1) as i16;
            }
        } else {
            for i in start..end {
                follower[i] = 0.max(band_log_e[i] as i32 - follower[i] as i32) as i16;
            }
        }
        for i in start..end {
            importance[i] = pshr32(13 * exp2((follower[i] as i32).min(4096)), 16);
        }
        if (!vbr || constrained_vbr) && !is_transient {
            for v in follower[start..end].iter_mut() {
                *v >>= 1;
            }
        }
        for i in start..end {
            if i < 8 {
                follower[i] = follower[i].wrapping_mul(2);
            }
            if i >= 12 {
                follower[i] >>= 1;
            }
        }
        for i in start..end {
            follower[i] = follower[i].min(4 << DB_SHIFT);
            let width = (c as i32 * (eb(i + 1) - eb(i))) << lm;
            let (boost, boost_bits);
            if width < 6 {
                boost = (follower[i] as i32) >> DB_SHIFT;
                boost_bits = (boost * width) << BITRES;
            } else if width > 48 {
                boost = (follower[i] as i32 * 8) >> DB_SHIFT;
                boost_bits = ((boost * width) << BITRES) / 8;
            } else {
                boost = (follower[i] as i32 * width / 6) >> DB_SHIFT;
                boost_bits = (boost * 6) << BITRES;
            }
            if (!vbr || (constrained_vbr && !is_transient))
                && (tot_boost + boost_bits) >> BITRES >> 3 > 2 * effective_bytes / 3
            {
                let cap = (2 * effective_bytes / 3) << BITRES << 3;
                offsets[i] = cap - tot_boost;
                tot_boost = cap;
                break;
            } else {
                offsets[i] = boost;
                tot_boost += boost_bits;
            }
        }
    } else {
        for v in importance[start..end].iter_mut() {
            *v = 13;
        }
    }
    *tot_boost_out = tot_boost;
    max_depth
}

#[allow(clippy::too_many_arguments)]
fn compute_vbr(
    base_target: i32,
    lm: i32,
    bitrate: i32,
    last_coded_bands: i32,
    c: usize,
    intensity: i32,
    constrained_vbr: bool,
    stereo_saving: i32,
    tot_boost: i32,
    tf_estimate: i32,
    max_depth: i32,
    temporal_vbr: i32,
) -> i32 {
    let coded_bands = if last_coded_bands != 0 {
        last_coded_bands as usize
    } else {
        NB_EBANDS
    };
    let mut coded_bins = eb(coded_bands) << lm;
    if c == 2 {
        coded_bins += eb(intensity.min(coded_bands as i32) as usize) << lm;
    }
    let mut target = base_target;
    if c == 2 {
        let coded_stereo_bands = intensity.min(coded_bands as i32);
        let coded_stereo_dof = (eb(coded_stereo_bands as usize) << lm) - coded_stereo_bands;
        let max_frac = (mul16(26214, coded_stereo_dof) / (coded_bins as i16 as i32)) as i16 as i32;
        let stereo_saving = stereo_saving.min(256);
        target -= mul16_32_q15(max_frac, target)
            .min(mul16(stereo_saving - 26, coded_stereo_dof << BITRES) >> 8);
    }
    target += tot_boost - (19 << lm);
    let tf_calibration = 721;
    target += shl32(mul16_32_q15(tf_estimate - tf_calibration, target), 1);
    {
        let bins = eb(NB_EBANDS - 2) << lm;
        let mut floor_depth = mul16((c as i32 * bins) << BITRES, max_depth) >> DB_SHIFT;
        floor_depth = floor_depth.max(target >> 2);
        target = target.min(floor_depth);
    }
    if constrained_vbr {
        target = base_target + mul16_32_q15(21955, target - base_target);
    }
    if tf_estimate < 3277 {
        let amount = mul16_q15(3329, 0.max(32000.min(96000 - bitrate)));
        let tvbr_factor = (mul16(temporal_vbr, amount) >> DB_SHIFT) as i16 as i32;
        target += mul16_32_q15(tvbr_factor, target);
    }
    target.min(2 * base_target)
}

/// `hysteresis_decision`.
fn hysteresis_decision(val: i32, thresholds: &[i32], hysteresis: &[i32], prev: i32) -> i32 {
    let n = thresholds.len();
    let mut i = 0;
    while i < n {
        if val < thresholds[i] {
            break;
        }
        i += 1;
    }
    let mut i = i as i32;
    let p = prev as usize;
    if i > prev && val < thresholds[p] + hysteresis[p] {
        i = prev;
    }
    if i < prev && val > thresholds[p - 1] - hysteresis[p - 1] {
        i = prev;
    }
    i
}

/// `spreading_decision`.
#[allow(clippy::too_many_arguments)]
fn spreading_decision(
    x: &[i16],
    average: &mut i32,
    last_decision: i32,
    hf_average: &mut i32,
    tapset_decision: &mut i32,
    update_hf: bool,
    end: usize,
    c: usize,
    m: usize,
    spread_weight: &[i32],
) -> i32 {
    let n0 = m * 120;
    let mut sum = 0i32;
    let mut nb_bands = 0i32;
    let mut hf_sum = 0i32;
    if m as i32 * (eb(end) - eb(end - 1)) <= 8 {
        return SPREAD_NONE;
    }
    for ch in 0..c {
        for i in 0..end {
            let n = m * (eb(i + 1) - eb(i)) as usize;
            if n <= 8 {
                continue;
            }
            let xs = &x[m * EBANDS[i] as usize + ch * n0..];
            let mut tcount = [0i32; 3];
            for &v in &xs[..n] {
                let x2n = mul16(mul16_q15(v as i32, v as i32), n as i32);
                if x2n < 2048 {
                    tcount[0] += 1;
                }
                if x2n < 512 {
                    tcount[1] += 1;
                }
                if x2n < 128 {
                    tcount[2] += 1;
                }
            }
            if i > NB_EBANDS - 4 {
                hf_sum += celt_udiv((32 * (tcount[1] + tcount[0])) as u32, n as u32) as i32;
            }
            let tmp = (2 * tcount[2] >= n as i32) as i32
                + (2 * tcount[1] >= n as i32) as i32
                + (2 * tcount[0] >= n as i32) as i32;
            sum += tmp * spread_weight[i];
            nb_bands += spread_weight[i];
        }
    }
    if update_hf {
        if hf_sum != 0 {
            hf_sum = celt_udiv(
                hf_sum as u32,
                (c as i32 * (4 - NB_EBANDS as i32 + end as i32)) as u32,
            ) as i32;
        }
        *hf_average = (*hf_average + hf_sum) >> 1;
        hf_sum = *hf_average;
        if *tapset_decision == 2 {
            hf_sum += 4;
        } else if *tapset_decision == 0 {
            hf_sum -= 4;
        }
        *tapset_decision = if hf_sum > 22 {
            2
        } else if hf_sum > 18 {
            1
        } else {
            0
        };
    }
    let mut sum = celt_udiv((sum << 8) as u32, nb_bands as u32) as i32;
    sum = (sum + *average) >> 1;
    *average = sum;
    sum = (3 * sum + (((3 - last_decision) << 7) + 64) + 2) >> 2;
    if sum < 80 {
        SPREAD_AGGRESSIVE
    } else if sum < 256 {
        SPREAD_NORMAL
    } else if sum < 384 {
        SPREAD_LIGHT
    } else {
        SPREAD_NONE
    }
}

fn compute_pitch_gain(xy: i32, xx: i32, yy: i32) -> i32 {
    if xy == 0 || xx == 0 || yy == 0 {
        return 0;
    }
    let sx = ilog2(xx) - 14;
    let sy = ilog2(yy) - 14;
    let mut shift = sx + sy;
    let mut x2y2 = mul16(vshr32(xx, sx), vshr32(yy, sy)) >> 14;
    if shift & 1 != 0 {
        if x2y2 < 32768 {
            x2y2 <<= 1;
            shift -= 1;
        } else {
            x2y2 >>= 1;
            shift += 1;
        }
    }
    let den = rsqrt_norm(x2y2);
    let g = mul16_32_q15(den, xy);
    let g = vshr32(g, (shift >> 1) - 1);
    g.min(32767) as i16 as i32
}

/// `remove_doubling`: refines the pitch period `t0` (at full rate) found on the 2x-decimated
/// `x` and returns the pitch gain.
fn remove_doubling(
    x: &[i16],
    maxperiod: usize,
    minperiod: usize,
    n: usize,
    t0_: &mut i32,
    prev_period: i32,
    prev_gain: i32,
) -> i32 {
    const SECOND_CHECK: [i32; 16] = [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2];
    let minperiod0 = minperiod as i32;
    let maxperiod = (maxperiod / 2) as i32;
    let minperiod = (minperiod / 2) as i32;
    *t0_ /= 2;
    let prev_period = prev_period / 2;
    let n = n / 2;
    let xo = maxperiod as usize;
    // Correlation of the `n` values from `xo + a` with those from `xo + b`.
    let dot = |a: i32, b: i32| -> i32 {
        let (a, b) = ((xo as i32 + a) as usize, (xo as i32 + b) as usize);
        dot16(&x[a..a + n], &x[b..b + n])
    };
    if *t0_ >= maxperiod {
        *t0_ = maxperiod - 1;
    }
    let t0 = *t0_;
    let mut t = t0;
    let xx = dot(0, 0);
    let mut xy = dot(0, -t0);
    let mut yy_lookup = [0i32; MAXPERIOD / 2 + 1];
    let yy_lookup = &mut yy_lookup[..maxperiod as usize + 1];
    yy_lookup[0] = xx;
    let mut yy = xx;
    // yy_lookup[i] = sum of x[xo - i + j]^2 for j < n: one value in, one out per step.
    let (head, tail) = (&x[..xo], &x[..xo + n]);
    for (i, (&a, &b)) in head.iter().rev().zip(tail.iter().rev()).enumerate() {
        let (a, b) = (a as i32, b as i32);
        yy = yy.wrapping_add(a * a).wrapping_sub(b * b);
        yy_lookup[i + 1] = 0.max(yy);
    }
    yy = yy_lookup[t0 as usize];
    let mut best_xy = xy;
    let mut best_yy = yy;
    let g0 = compute_pitch_gain(xy, xx, yy);
    let mut g = g0;
    for k in 2..=15i32 {
        let t1 = celt_udiv((2 * t0 + k) as u32, (2 * k) as u32) as i32;
        if t1 < minperiod {
            break;
        }
        let t1b = if k == 2 {
            if t1 + t0 > maxperiod { t0 } else { t0 + t1 }
        } else {
            celt_udiv(
                (2 * SECOND_CHECK[k as usize] * t0 + k) as u32,
                (2 * k) as u32,
            ) as i32
        };
        xy = dot(0, -t1);
        let xy2 = dot(0, -t1b);
        xy = (xy.wrapping_add(xy2)) >> 1;
        yy = (yy_lookup[t1 as usize].wrapping_add(yy_lookup[t1b as usize])) >> 1;
        let g1 = compute_pitch_gain(xy, xx, yy);
        let cont = if (t1 - prev_period).abs() <= 1 {
            prev_gain
        } else if (t1 - prev_period).abs() <= 2 && 5 * k * k < t0 {
            prev_gain >> 1
        } else {
            0
        };
        let mut thresh = 9830i32.max(mul16_q15(22938, g0) - cont) as i16 as i32;
        if t1 < 3 * minperiod {
            thresh = 13107i32.max(mul16_q15(27853, g0) - cont) as i16 as i32;
        } else if t1 < 2 * minperiod {
            thresh = 16384i32.max(mul16_q15(29491, g0) - cont) as i16 as i32;
        }
        if g1 > thresh {
            best_xy = xy;
            best_yy = yy;
            t = t1;
            g = g1;
        }
    }
    best_xy = 0.max(best_xy);
    let mut pg = if best_yy <= best_xy {
        32767
    } else {
        (frac_div32(best_xy, best_yy + 1) >> 16) as i16 as i32
    };
    let mut xcorr = [0i32; 3];
    for (k, v) in xcorr.iter_mut().enumerate() {
        *v = dot(0, -(t + k as i32 - 1));
    }
    let offset = if (xcorr[2] - xcorr[0]) > mul16_32_q15(22938, xcorr[1] - xcorr[0]) {
        1
    } else if (xcorr[0] - xcorr[2]) > mul16_32_q15(22938, xcorr[1] - xcorr[2]) {
        -1
    } else {
        0
    };
    if pg > g {
        pg = g;
    }
    *t0_ = 2 * t + offset;
    if *t0_ < minperiod0 {
        *t0_ = minperiod0;
    }
    pg
}
