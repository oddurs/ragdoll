//! CELT bit allocation (libopus `rate.c`, `rate.h`).

use super::celt_tables::{BAND_ALLOCATION, CACHE_BITS, CACHE_CAPS, CACHE_INDEX, EBANDS, LOG_N};
use super::range::{BITRES, Ec};

pub const NB_EBANDS: usize = 21;
const NB_ALLOC_VECTORS: usize = 11;
const LOG_MAX_PSEUDO: usize = 6;
pub const MAX_FINE_BITS: i32 = 8;
const FINE_OFFSET: i32 = 21;
pub const QTHETA_OFFSET: i32 = 4;
pub const QTHETA_OFFSET_TWOPHASE: i32 = 16;
const ALLOC_STEPS: i32 = 6;

const LOG2_FRAC_TABLE: [u8; 24] = [
    0, 8, 13, 16, 19, 21, 23, 24, 26, 27, 28, 29, 30, 31, 32, 32, 33, 34, 34, 35, 36, 36, 37, 37,
];

#[inline]
pub fn get_pulses(i: i32) -> i32 {
    if i < 8 {
        i
    } else {
        (8 + (i & 7)) << ((i >> 3) - 1)
    }
}

#[inline]
fn cache(band: usize, lm: i32) -> &'static [u8] {
    let idx = CACHE_INDEX[(lm + 1) as usize * NB_EBANDS + band];
    &CACHE_BITS[idx as usize..]
}

/// `bits2pulses`.
pub fn bits2pulses(band: usize, lm: i32, bits: i32) -> i32 {
    let c = cache(band, lm);
    let mut lo = 0i32;
    let mut hi = c[0] as i32;
    let bits = bits - 1;
    for _ in 0..LOG_MAX_PSEUDO {
        let mid = (lo + hi + 1) >> 1;
        if c[mid as usize] as i32 >= bits {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let lo_bits = if lo == 0 { -1 } else { c[lo as usize] as i32 };
    if bits - lo_bits <= c[hi as usize] as i32 - bits {
        lo
    } else {
        hi
    }
}

/// `pulses2bits`.
pub fn pulses2bits(band: usize, lm: i32, pulses: i32) -> i32 {
    if pulses == 0 {
        0
    } else {
        cache(band, lm)[pulses as usize] as i32 + 1
    }
}

/// `init_caps`.
pub fn init_caps(cap: &mut [i32], lm: i32, c: i32) {
    for (i, slot) in cap.iter_mut().enumerate().take(NB_EBANDS) {
        let n = ((EBANDS[i + 1] - EBANDS[i]) as i32) << lm;
        *slot = ((CACHE_CAPS[NB_EBANDS * (2 * lm + c - 1) as usize + i] as i32 + 64) * c * n) >> 2;
    }
}

/// Outputs of [`compute_allocation`].
pub struct Allocation {
    pub coded_bands: i32,
    pub intensity: i32,
    pub dual_stereo: bool,
    pub balance: i32,
}

/// `interp_bits2pulses`.
#[allow(clippy::too_many_arguments)]
fn interp_bits2pulses(
    start: usize,
    end: usize,
    skip_start: usize,
    bits1: &[i32],
    bits2: &[i32],
    thresh: &[i32],
    cap: &[i32],
    mut total: i32,
    skip_rsv: i32,
    intensity: &mut i32,
    mut intensity_rsv: i32,
    dual_stereo: &mut bool,
    mut dual_stereo_rsv: i32,
    bits: &mut [i32],
    ebits: &mut [i32],
    fine_priority: &mut [i32],
    c: i32,
    lm: i32,
    ec: &mut Ec,
    encode: bool,
    prev: i32,
    signal_bandwidth: i32,
) -> (i32, i32) {
    let eb = |j: usize| EBANDS[j] as i32;
    let alloc_floor = c << BITRES;
    let stereo = (c > 1) as i32;
    let log_m = lm << BITRES;
    let mut lo = 0i32;
    let mut hi = 1i32 << ALLOC_STEPS;
    for _ in 0..ALLOC_STEPS {
        let mid = (lo + hi) >> 1;
        let mut psum = 0i32;
        let mut done = false;
        for j in (start..end).rev() {
            let tmp = bits1[j] + ((mid * bits2[j]) >> ALLOC_STEPS);
            if tmp >= thresh[j] || done {
                done = true;
                psum += tmp.min(cap[j]);
            } else if tmp >= alloc_floor {
                psum += alloc_floor;
            }
        }
        if psum > total {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let mut psum = 0i32;
    let mut done = false;
    for j in (start..end).rev() {
        let mut tmp = bits1[j] + ((lo * bits2[j]) >> ALLOC_STEPS);
        if tmp < thresh[j] && !done {
            tmp = if tmp >= alloc_floor { alloc_floor } else { 0 };
        } else {
            done = true;
        }
        tmp = tmp.min(cap[j]);
        bits[j] = tmp;
        psum += tmp;
    }
    let mut coded_bands = end as i32;
    loop {
        let j = coded_bands as usize - 1;
        if j <= skip_start {
            total += skip_rsv;
            break;
        }
        let mut left = total - psum;
        let percoeff = (left as u32 / (eb(coded_bands as usize) - eb(start)) as u32) as i32;
        left -= (eb(coded_bands as usize) - eb(start)) * percoeff;
        let rem = (left - (eb(j) - eb(start))).max(0);
        let band_width = eb(coded_bands as usize) - eb(j);
        let mut band_bits = bits[j] + percoeff * band_width + rem;
        if band_bits >= thresh[j].max(alloc_floor + (1 << BITRES)) {
            if encode {
                let depth_threshold = if coded_bands > 17 {
                    if (j as i32) < prev { 7 } else { 9 }
                } else {
                    0
                };
                if coded_bands <= start as i32 + 2
                    || (band_bits > ((depth_threshold * band_width) << lm << BITRES) >> 4
                        && j as i32 <= signal_bandwidth)
                {
                    ec.enc_bit_logp(true, 1);
                    break;
                }
                ec.enc_bit_logp(false, 1);
            } else if ec.dec_bit_logp(1) {
                break;
            }
            psum += 1 << BITRES;
            band_bits -= 1 << BITRES;
        }
        psum -= bits[j] + intensity_rsv;
        if intensity_rsv > 0 {
            intensity_rsv = LOG2_FRAC_TABLE[j - start] as i32;
        }
        psum += intensity_rsv;
        if band_bits >= alloc_floor {
            psum += alloc_floor;
            bits[j] = alloc_floor;
        } else {
            bits[j] = 0;
        }
        coded_bands -= 1;
    }
    if intensity_rsv > 0 {
        if encode {
            *intensity = (*intensity).min(coded_bands);
            ec.enc_uint(
                (*intensity - start as i32) as u32,
                (coded_bands + 1 - start as i32) as u32,
            );
        } else {
            *intensity = start as i32 + ec.dec_uint((coded_bands + 1 - start as i32) as u32) as i32;
        }
    } else {
        *intensity = 0;
    }
    if *intensity <= start as i32 {
        total += dual_stereo_rsv;
        dual_stereo_rsv = 0;
    }
    if dual_stereo_rsv > 0 {
        if encode {
            ec.enc_bit_logp(*dual_stereo, 1);
        } else {
            *dual_stereo = ec.dec_bit_logp(1);
        }
    } else {
        *dual_stereo = false;
    }
    let cb = coded_bands as usize;
    let mut left = total - psum;
    let percoeff = (left as u32 / (eb(cb) - eb(start)) as u32) as i32;
    left -= (eb(cb) - eb(start)) * percoeff;
    for j in start..cb {
        bits[j] += percoeff * (eb(j + 1) - eb(j));
    }
    for j in start..cb {
        let tmp = left.min(eb(j + 1) - eb(j));
        bits[j] += tmp;
        left -= tmp;
    }
    let mut balance = 0i32;
    let mut j = start;
    while j < cb {
        let n0 = eb(j + 1) - eb(j);
        let n = n0 << lm;
        let bit = bits[j] + balance;
        let mut excess;
        if n > 1 {
            excess = (bit - cap[j]).max(0);
            bits[j] = bit - excess;
            let den = c * n + (c == 2 && n > 2 && !*dual_stereo && (j as i32) < *intensity) as i32;
            let nclogn = den * (LOG_N[j] as i32 + log_m);
            let mut offset = (nclogn >> 1) - den * FINE_OFFSET;
            if n == 2 {
                offset += (den << BITRES) >> 2;
            }
            if bits[j] + offset < (den * 2) << BITRES {
                offset += nclogn >> 2;
            } else if bits[j] + offset < (den * 3) << BITRES {
                offset += nclogn >> 3;
            }
            ebits[j] = (bits[j] + offset + (den << (BITRES - 1))).max(0);
            ebits[j] = ((ebits[j] as u32 / den as u32) as i32) >> BITRES;
            if c * ebits[j] > (bits[j] >> BITRES) {
                ebits[j] = bits[j] >> stereo >> BITRES;
            }
            ebits[j] = ebits[j].min(MAX_FINE_BITS);
            fine_priority[j] = (ebits[j] * (den << BITRES) >= bits[j] + offset) as i32;
            bits[j] -= (c * ebits[j]) << BITRES;
        } else {
            excess = (bit - (c << BITRES)).max(0);
            bits[j] = bit - excess;
            ebits[j] = 0;
            fine_priority[j] = 1;
        }
        if excess > 0 {
            let extra_fine = (excess >> (stereo + BITRES)).min(MAX_FINE_BITS - ebits[j]);
            ebits[j] += extra_fine;
            let extra_bits = (extra_fine * c) << BITRES;
            fine_priority[j] = (extra_bits >= excess - balance) as i32;
            excess -= extra_bits;
        }
        balance = excess;
        j += 1;
    }
    while j < end {
        ebits[j] = bits[j] >> stereo >> BITRES;
        bits[j] = 0;
        fine_priority[j] = (ebits[j] < 1) as i32;
        j += 1;
    }
    (coded_bands, balance)
}

/// `clt_compute_allocation`: pulses, fine energy bits and priorities per band.
#[allow(clippy::too_many_arguments)]
pub fn compute_allocation(
    start: usize,
    end: usize,
    offsets: &[i32],
    cap: &[i32],
    alloc_trim: i32,
    intensity: &mut i32,
    dual_stereo: &mut bool,
    total: i32,
    pulses: &mut [i32],
    ebits: &mut [i32],
    fine_priority: &mut [i32],
    c: i32,
    lm: i32,
    ec: &mut Ec,
    encode: bool,
    prev: i32,
    signal_bandwidth: i32,
) -> Allocation {
    let eb = |j: usize| EBANDS[j] as i32;
    let mut total = total.max(0);
    let len = NB_EBANDS;
    let mut skip_start = start;
    let skip_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
    total -= skip_rsv;
    let mut intensity_rsv = 0;
    let mut dual_stereo_rsv = 0;
    if c == 2 {
        intensity_rsv = LOG2_FRAC_TABLE[end - start] as i32;
        if intensity_rsv > total {
            intensity_rsv = 0;
        } else {
            total -= intensity_rsv;
            dual_stereo_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
            total -= dual_stereo_rsv;
        }
    }
    let mut bits1 = [0i32; NB_EBANDS];
    let mut bits2 = [0i32; NB_EBANDS];
    let mut thresh = [0i32; NB_EBANDS];
    let mut trim_offset = [0i32; NB_EBANDS];
    for j in start..end {
        let w = eb(j + 1) - eb(j);
        thresh[j] = (c << BITRES).max(((3 * w) << lm << BITRES) >> 4);
        trim_offset[j] =
            (c * w * (alloc_trim - 5 - lm) * (end as i32 - j as i32 - 1) * (1 << (lm + BITRES)))
                >> 6;
        if w << lm == 1 {
            trim_offset[j] -= c << BITRES;
        }
    }
    let mut lo = 1i32;
    let mut hi = NB_ALLOC_VECTORS as i32 - 1;
    loop {
        let mut done = false;
        let mut psum = 0i32;
        let mid = (lo + hi) >> 1;
        for j in (start..end).rev() {
            let n = eb(j + 1) - eb(j);
            let mut bitsj = (c * n * BAND_ALLOCATION[mid as usize * len + j] as i32) << lm >> 2;
            if bitsj > 0 {
                bitsj = (bitsj + trim_offset[j]).max(0);
            }
            bitsj += offsets[j];
            if bitsj >= thresh[j] || done {
                done = true;
                psum += bitsj.min(cap[j]);
            } else if bitsj >= c << BITRES {
                psum += c << BITRES;
            }
        }
        if psum > total {
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
        if lo > hi {
            break;
        }
    }
    hi = lo;
    lo -= 1;
    for j in start..end {
        let n = eb(j + 1) - eb(j);
        let mut bits1j = (c * n * BAND_ALLOCATION[lo as usize * len + j] as i32) << lm >> 2;
        let mut bits2j = if hi >= NB_ALLOC_VECTORS as i32 {
            cap[j]
        } else {
            (c * n * BAND_ALLOCATION[hi as usize * len + j] as i32) << lm >> 2
        };
        if bits1j > 0 {
            bits1j = (bits1j + trim_offset[j]).max(0);
        }
        if bits2j > 0 {
            bits2j = (bits2j + trim_offset[j]).max(0);
        }
        if lo > 0 {
            bits1j += offsets[j];
        }
        bits2j += offsets[j];
        if offsets[j] > 0 {
            skip_start = j;
        }
        bits2j = (bits2j - bits1j).max(0);
        bits1[j] = bits1j;
        bits2[j] = bits2j;
    }
    let (coded_bands, balance) = interp_bits2pulses(
        start,
        end,
        skip_start,
        &bits1,
        &bits2,
        &thresh,
        cap,
        total,
        skip_rsv,
        intensity,
        intensity_rsv,
        dual_stereo,
        dual_stereo_rsv,
        pulses,
        ebits,
        fine_priority,
        c,
        lm,
        ec,
        encode,
        prev,
        signal_bandwidth,
    );
    Allocation {
        coded_bands,
        intensity: *intensity,
        dual_stereo: *dual_stereo,
        balance,
    }
}
