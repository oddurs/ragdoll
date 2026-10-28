//! CELT band energy quantisation (libopus `quant_bands.c`, fixed point). Energies are Q10
//! (`DB_SHIFT`) base-2 logarithms.

use super::celt_tables::{E_MEANS, E_PROB_MODEL};
use super::math::{log2, mac16, mul16, mul16_32_q15, mul16_q15, pshr32, shl16, shl32};
use super::range::Ec;
use super::rate::{MAX_FINE_BITS, NB_EBANDS};

pub const DB_SHIFT: i32 = 10;
const PRED_COEF: [i32; 4] = [29440, 26112, 21248, 16384];
const BETA_COEF: [i32; 4] = [30147, 22282, 12124, 6554];
const BETA_INTRA: i32 = 4915;
const SMALL_ENERGY_ICDF: [u8; 3] = [2, 1, 0];

fn prob_model(lm: i32, intra: bool) -> &'static [u8] {
    let o = (lm as usize * 2 + intra as usize) * 42;
    &E_PROB_MODEL[o..o + 42]
}

/// `unquant_coarse_energy`.
pub fn unquant_coarse(
    start: usize,
    end: usize,
    old: &mut [i16],
    intra: bool,
    dec: &mut Ec,
    c: usize,
    lm: i32,
) {
    let model = prob_model(lm, intra);
    let mut prev = [0i32; 2];
    let (coef, beta) = if intra {
        (0, BETA_INTRA)
    } else {
        (PRED_COEF[lm as usize], BETA_COEF[lm as usize])
    };
    let budget = dec.storage as i32 * 8;
    for i in start..end {
        for ch in 0..c {
            let tell = dec.tell();
            let qi = if budget - tell >= 15 {
                let pi = 2 * i.min(20);
                dec.laplace_decode((model[pi] as u32) << 7, (model[pi + 1] as i32) << 6)
            } else if budget - tell >= 2 {
                let q = dec.dec_icdf(&SMALL_ENERGY_ICDF, 2);
                (q >> 1) ^ -(q & 1)
            } else if budget - tell >= 1 {
                -(dec.dec_bit_logp(1) as i32)
            } else {
                -1
            };
            let q = shl32(qi, DB_SHIFT);
            let k = i + ch * NB_EBANDS;
            old[k] = old[k].max(-(9 << DB_SHIFT));
            let mut tmp = pshr32(mul16(coef, old[k] as i32), 8)
                .wrapping_add(prev[ch])
                .wrapping_add(shl32(q, 7));
            tmp = tmp.max(-(28 << (DB_SHIFT + 7)));
            old[k] = pshr32(tmp, 7) as i16;
            prev[ch] = prev[ch]
                .wrapping_add(shl32(q, 7))
                .wrapping_sub(mul16(beta, pshr32(q, 8)));
        }
    }
}

/// `unquant_fine_energy`.
pub fn unquant_fine(
    start: usize,
    end: usize,
    old: &mut [i16],
    fine_quant: &[i32],
    dec: &mut Ec,
    c: usize,
) {
    for i in start..end {
        if fine_quant[i] <= 0 {
            continue;
        }
        for ch in 0..c {
            let q2 = dec.dec_bits(fine_quant[i] as u32) as i32;
            let offset = ((shl32(q2, DB_SHIFT) + 512) >> fine_quant[i]) - 512;
            let k = i + ch * NB_EBANDS;
            old[k] = (old[k] as i32 + offset) as i16;
        }
    }
}

/// `unquant_energy_finalise`.
#[allow(clippy::too_many_arguments)]
pub fn unquant_finalise(
    start: usize,
    end: usize,
    old: &mut [i16],
    fine_quant: &[i32],
    fine_priority: &[i32],
    mut bits_left: i32,
    dec: &mut Ec,
    c: usize,
) {
    for prio in 0..2 {
        let mut i = start;
        while i < end && bits_left >= c as i32 {
            if fine_quant[i] >= MAX_FINE_BITS || fine_priority[i] != prio {
                i += 1;
                continue;
            }
            for ch in 0..c {
                let q2 = dec.dec_bits(1) as i32;
                let offset = (shl16(q2, DB_SHIFT) - 512) >> (fine_quant[i] + 1);
                let k = i + ch * NB_EBANDS;
                old[k] = (old[k] as i32 + offset) as i16;
                bits_left -= 1;
            }
            i += 1;
        }
    }
}

/// `loss_distortion`.
fn loss_distortion(e: &[i16], old: &[i16], start: usize, end: usize, c: usize) -> i32 {
    let mut dist = 0i32;
    for ch in 0..c {
        for i in start..end {
            let k = i + ch * NB_EBANDS;
            let d = ((e[k] >> 3) as i32 - (old[k] >> 3) as i32) as i16 as i32;
            dist = mac16(dist, d, d);
        }
    }
    200.min(dist >> (2 * DB_SHIFT - 6))
}

#[allow(clippy::too_many_arguments)]
fn quant_coarse_impl(
    start: usize,
    end: usize,
    e: &[i16],
    old: &mut [i16],
    budget: i32,
    error: &mut [i16],
    enc: &mut Ec,
    c: usize,
    lm: i32,
    intra: bool,
    max_decay: i32,
) -> i32 {
    let model = prob_model(lm, intra);
    let mut badness = 0;
    let mut prev = [0i32; 2];
    if enc.tell() + 3 <= budget {
        enc.enc_bit_logp(intra, 3);
    }
    let (coef, beta) = if intra {
        (0, BETA_INTRA)
    } else {
        (PRED_COEF[lm as usize], BETA_COEF[lm as usize])
    };
    for i in start..end {
        for ch in 0..c {
            let k = i + ch * NB_EBANDS;
            let x = e[k] as i32;
            let old_e = (old[k] as i32).max(-(9 << DB_SHIFT));
            let f = shl32(x, 7)
                .wrapping_sub(pshr32(mul16(coef, old_e), 8))
                .wrapping_sub(prev[ch]);
            let mut qi = (f + (1 << (DB_SHIFT + 6))) >> (DB_SHIFT + 7);
            let decay_bound = (-(28 << DB_SHIFT)).max(old[k] as i32 - max_decay) as i16 as i32;
            if qi < 0 && x < decay_bound {
                qi += (decay_bound - x) >> DB_SHIFT;
                if qi > 0 {
                    qi = 0;
                }
            }
            let qi0 = qi;
            let tell = enc.tell();
            let bits_left = budget - tell - 3 * c as i32 * (end - i) as i32;
            if i != start && bits_left < 30 {
                if bits_left < 24 {
                    qi = qi.min(1);
                }
                if bits_left < 16 {
                    qi = qi.max(-1);
                }
            }
            if budget - tell >= 15 {
                let pi = 2 * i.min(20);
                enc.laplace_encode(
                    &mut qi,
                    (model[pi] as u32) << 7,
                    (model[pi + 1] as i32) << 6,
                );
            } else if budget - tell >= 2 {
                qi = qi.clamp(-1, 1);
                enc.enc_icdf(
                    ((2 * qi) ^ -((qi < 0) as i32)) as usize,
                    &SMALL_ENERGY_ICDF,
                    2,
                );
            } else if budget - tell >= 1 {
                qi = qi.min(0);
                enc.enc_bit_logp(qi != 0, 1);
            } else {
                qi = -1;
            }
            error[k] = (pshr32(f, 7) - shl16(qi, DB_SHIFT)) as i16;
            badness += (qi0 - qi).abs();
            let q = shl32(qi, DB_SHIFT);
            let mut tmp = pshr32(mul16(coef, old_e), 8)
                .wrapping_add(prev[ch])
                .wrapping_add(shl32(q, 7));
            tmp = tmp.max(-(28 << (DB_SHIFT + 7)));
            old[k] = pshr32(tmp, 7) as i16;
            prev[ch] = prev[ch]
                .wrapping_add(shl32(q, 7))
                .wrapping_sub(mul16(beta, pshr32(q, 8)));
        }
    }
    badness
}

/// `quant_coarse_energy`: codes the coarse energies, choosing intra or inter prediction.
#[allow(clippy::too_many_arguments)]
pub fn quant_coarse(
    start: usize,
    end: usize,
    eff_end: usize,
    e: &[i16],
    old: &mut [i16],
    budget: u32,
    error: &mut [i16],
    enc: &mut Ec,
    c: usize,
    lm: i32,
    nb_available_bytes: i32,
    force_intra: bool,
    delayed_intra: &mut i32,
    two_pass: bool,
    loss_rate: i32,
) {
    let intra_bias = (budget
        .wrapping_mul(*delayed_intra as u32)
        .wrapping_mul(loss_rate as u32)
        / (c as u32 * 512)) as i32;
    let budget = budget as i32;
    let mut intra = force_intra
        || (!two_pass
            && *delayed_intra > 2 * c as i32 * (end - start) as i32
            && nb_available_bytes > (end - start) as i32 * c as i32);
    let new_distortion = loss_distortion(e, old, start, eff_end, c);
    let mut two_pass = two_pass;
    if enc.tell() + 3 > budget {
        two_pass = false;
        intra = false;
    }
    let mut max_decay = 16 << DB_SHIFT;
    if end - start > 10 {
        max_decay = max_decay.min(shl32(nb_available_bytes, DB_SHIFT - 3));
    }
    let start_state = enc.clone();
    let mut old_intra = old.to_vec();
    let mut error_intra = error.to_vec();
    let mut badness1 = 0;
    if two_pass || intra {
        badness1 = quant_coarse_impl(
            start,
            end,
            e,
            &mut old_intra,
            budget,
            &mut error_intra,
            enc,
            c,
            lm,
            true,
            max_decay,
        );
    }
    if !intra {
        let tell_intra = enc.tell_frac() as i32;
        let intra_state = enc.clone();
        *enc = start_state;
        let badness2 = quant_coarse_impl(
            start, end, e, old, budget, error, enc, c, lm, false, max_decay,
        );
        if two_pass
            && (badness1 < badness2
                || (badness1 == badness2 && enc.tell_frac() as i32 + intra_bias > tell_intra))
        {
            *enc = intra_state;
            old.copy_from_slice(&old_intra);
            error.copy_from_slice(&error_intra);
            intra = true;
        }
    } else {
        old.copy_from_slice(&old_intra);
        error.copy_from_slice(&error_intra);
    }
    if intra {
        *delayed_intra = new_distortion;
    } else {
        let p = PRED_COEF[lm as usize];
        *delayed_intra = mul16_32_q15(mul16_q15(p, p), *delayed_intra).wrapping_add(new_distortion);
    }
}

/// `quant_fine_energy`.
pub fn quant_fine(
    start: usize,
    end: usize,
    old: &mut [i16],
    error: &mut [i16],
    fine_quant: &[i32],
    enc: &mut Ec,
    c: usize,
) {
    for i in start..end {
        if fine_quant[i] <= 0 {
            continue;
        }
        let frac = 1i32 << fine_quant[i];
        for ch in 0..c {
            let k = i + ch * NB_EBANDS;
            let mut q2 = (error[k] as i32 + 512) >> (DB_SHIFT - fine_quant[i]);
            q2 = q2.clamp(0, frac - 1);
            enc.enc_bits(q2 as u32, fine_quant[i] as u32);
            let offset = ((shl32(q2, DB_SHIFT) + 512) >> fine_quant[i]) - 512;
            old[k] = (old[k] as i32 + offset) as i16;
            error[k] = (error[k] as i32 - offset) as i16;
        }
    }
}

/// `quant_energy_finalise`.
#[allow(clippy::too_many_arguments)]
pub fn quant_finalise(
    start: usize,
    end: usize,
    old: &mut [i16],
    error: &mut [i16],
    fine_quant: &[i32],
    fine_priority: &[i32],
    mut bits_left: i32,
    enc: &mut Ec,
    c: usize,
) {
    for prio in 0..2 {
        let mut i = start;
        while i < end && bits_left >= c as i32 {
            if fine_quant[i] >= MAX_FINE_BITS || fine_priority[i] != prio {
                i += 1;
                continue;
            }
            for ch in 0..c {
                let k = i + ch * NB_EBANDS;
                let q2 = (error[k] >= 0) as i32;
                enc.enc_bits(q2 as u32, 1);
                let offset = (shl16(q2, DB_SHIFT) - 512) >> (fine_quant[i] + 1);
                old[k] = (old[k] as i32 + offset) as i16;
                error[k] = (error[k] as i32 - offset) as i16;
                bits_left -= 1;
            }
            i += 1;
        }
    }
}

/// `amp2Log2`: band amplitudes to Q10 log energies relative to the band means.
pub fn amp2log2(eff_end: usize, end: usize, band_e: &[i32], log_e: &mut [i16], c: usize) {
    for ch in 0..c {
        for i in 0..eff_end {
            let k = i + ch * NB_EBANDS;
            log_e[k] = (log2(band_e[k]) - shl16(E_MEANS[i] as i32, 6) + (2 << DB_SHIFT)) as i16;
        }
        for i in eff_end..end {
            log_e[ch * NB_EBANDS + i] = -(14 << DB_SHIFT);
        }
    }
}
