//! CELT band processing (libopus `bands.c`, fixed point): energy normalisation, band splitting
//! and quantisation shared by the decoder and encoder, folding, stereo and anti-collapse.
//!
//! Where libopus aliases the folding source (`lowband`) with scratch space and modifies it in
//! place, this port works on a copy; the in-place changes are never read back in libopus either.

use super::celt_tables::{E_MEANS, EBANDS, LOG_N};
use super::energy::DB_SHIFT;
use super::math::{
    add16, celt_udiv, exp2, exp2_frac, frac_mul16, ilog2, inner_prod, lcg_rand, mac16, maxabs32,
    mul16, mul16_32_q15, mul16_p15, mul16_q14, mul16_q15, pshr32, rcp, rsqrt_norm, sat16_i32,
    shl16, shl32, sqrt, sub16, vshr32, zlog2,
};
use super::range::{BITRES, Ec, EcSave, ilog};
use super::rate::{
    NB_EBANDS, QTHETA_OFFSET, QTHETA_OFFSET_TWOPHASE, bits2pulses, get_pulses, pulses2bits,
};
use super::vq::{SPREAD_AGGRESSIVE, alg_quant, alg_unquant, renormalise_vector, stereo_itheta};

pub const NORM_SCALING: i32 = 16384;
/// Widest band: the last one at LM 3 (`8 * (100 - 78)` coefficients).
pub const MAX_BAND: usize = 176;
const Q15ONE: i32 = 32767;

#[inline]
fn eb(i: usize) -> i32 {
    EBANDS[i] as i32
}

/// `bitexact_cos`.
const fn bitexact_cos(x: i32) -> i32 {
    let tmp = (4096 + (x as i16 as i32) * (x as i16 as i32)) >> 13;
    let x2 = tmp as i16 as i32;
    let x2 = (32767 - x2) + frac_mul16(x2, -7651 + frac_mul16(x2, 8277 + frac_mul16(-626, x2)));
    1 + x2 as i16 as i32
}

/// `bitexact_log2tan`.
const fn bitexact_log2tan(isin: i32, icos: i32) -> i32 {
    let lc = ilog(icos as u32);
    let ls = ilog(isin as u32);
    let icos = icos << (15 - lc);
    let isin = isin << (15 - ls);
    (ls - lc) * (1 << 11) + frac_mul16(isin, frac_mul16(isin, -2597) + 7932)
        - frac_mul16(icos, frac_mul16(icos, -2597) + 7932)
}

/// The bits above which `quant_partition` splits band `i` at `lm` (by `(lm + 1) * 21 + i`):
/// the most its pulse cache holds, plus 12. Bands with no cache never split.
static SPLIT_LIMIT: [i32; 105] = {
    use super::celt_tables::{CACHE_BITS, CACHE_INDEX};
    let mut t = [i32::MAX; 105];
    let mut j = 0;
    while j < t.len() {
        let idx = CACHE_INDEX[j];
        if idx >= 0 {
            let c = idx as usize;
            t[j] = CACHE_BITS[c + CACHE_BITS[c] as usize] as i32 + 12;
        }
        j += 1;
    }
    t
};

/// `isqrt32(8 x + 1)` for every `x` the triangular theta pdf (at most 129 steps a side)
/// takes it of, so the decoding of a split's angle need not wait on a square root.
static TRIANGLE_ISQRT: [u16; 8385] = {
    let mut t = [0u16; 8385];
    let mut x = 0;
    let mut r = 1u32;
    while x < t.len() {
        let v = 8 * x as u32 + 1;
        while (r + 1) * (r + 1) <= v {
            r += 1;
        }
        t[x] = r as u16;
        x += 1;
    }
    t
};

/// `isqrt32(8 x + 1)`.
#[inline]
fn triangle_isqrt(x: u32) -> i32 {
    match TRIANGLE_ISQRT.get(x as usize) {
        Some(&r) => r as i32,
        None => super::math::isqrt32(8 * x + 1) as i32,
    }
}

/// The split angles strictly between 0 and 16384 that a coded theta can take (`qn <= 256`
/// steps) lie in `THETA_MIN..=THETA_MAX`, where both cosines stay below 32768.
const THETA_MIN: usize = 64;
const THETA_MAX: usize = 16384 - THETA_MIN;

/// `bitexact_cos(t)` for `t` in `THETA_MIN..=THETA_MAX`, at `t - THETA_MIN`.
static THETA_COS: [i16; THETA_MAX - THETA_MIN + 1] = {
    let mut c = [0i16; THETA_MAX - THETA_MIN + 1];
    let mut t = THETA_MIN;
    while t <= THETA_MAX {
        c[t - THETA_MIN] = bitexact_cos(t as i32) as i16;
        t += 1;
    }
    c
};

/// `bitexact_log2tan(bitexact_cos(16384 - t), bitexact_cos(t))` for `t` in
/// `THETA_MIN..=THETA_MAX`, at `t - THETA_MIN`.
static THETA_LOG2TAN: [i16; THETA_MAX - THETA_MIN + 1] = {
    let mut l = [0i16; THETA_MAX - THETA_MIN + 1];
    let mut t = THETA_MIN;
    while t <= THETA_MAX {
        let (imid, iside) = (bitexact_cos(t as i32), bitexact_cos(16384 - t as i32));
        l[t - THETA_MIN] = bitexact_log2tan(iside, imid) as i16;
        t += 1;
    }
    l
};

/// The mid and side gains of the split angle `itheta` (strictly between 0 and 16384) and
/// their `bitexact_log2tan`. Every split's bit division waits on these polynomials, so the
/// angles a coded theta can take come from tables.
#[inline]
fn theta_gains(itheta: i32) -> (i32, i32, i32) {
    match itheta as usize {
        t @ THETA_MIN..=THETA_MAX => (
            THETA_COS[t - THETA_MIN] as i32,
            THETA_COS[THETA_MAX - t] as i32,
            THETA_LOG2TAN[t - THETA_MIN] as i32,
        ),
        _ => {
            let (imid, iside) = (bitexact_cos(itheta), bitexact_cos(16384 - itheta));
            (imid, iside, bitexact_log2tan(iside, imid))
        }
    }
}

/// `compute_band_energies`: Q`SIG_SHIFT`-free band amplitudes of the MDCT output.
pub fn compute_band_energies(x: &[i32], band_e: &mut [i32], end: usize, c: usize, lm: i32) {
    let n = 120usize << lm;
    for ch in 0..c {
        for i in 0..end {
            let lo = ch * n + ((EBANDS[i] as usize) << lm);
            let hi = ch * n + ((EBANDS[i + 1] as usize) << lm);
            let maxval = maxabs32(&x[lo..hi]);
            band_e[i + ch * NB_EBANDS] = if maxval > 0 {
                let shift = ilog2(maxval) - 14 + (((LOG_N[i] as i32 >> BITRES) + lm + 1) >> 1);
                let mut sum = 0i32;
                for &v in &x[lo..hi] {
                    let s = if shift > 0 {
                        v >> shift
                    } else {
                        shl32(v, -shift)
                    } as i16 as i32;
                    sum = mac16(sum, s, s);
                }
                1i32.wrapping_add(vshr32(sqrt(sum), -shift))
            } else {
                1
            };
        }
    }
}

/// `normalise_bands`.
pub fn normalise_bands(
    freq: &[i32],
    x: &mut [i16],
    band_e: &[i32],
    end: usize,
    c: usize,
    m: usize,
) {
    let n = m * 120;
    for ch in 0..c {
        for i in 0..end {
            let be = band_e[i + ch * NB_EBANDS];
            let shift = zlog2(be) - 13;
            let e = vshr32(be, shift) as i16 as i32;
            let g = rcp(shl32(e, 3)) as i16 as i32;
            for j in m * EBANDS[i] as usize..m * EBANDS[i + 1] as usize {
                x[j + ch * n] = mul16_q15(vshr32(freq[j + ch * n], shift - 1), g) as i16;
            }
        }
    }
}

/// `denormalise_bands`: scales the unit-norm bands by their energies into `freq`.
#[allow(clippy::too_many_arguments)]
pub fn denormalise_bands(
    x: &[i16],
    freq: &mut [i32],
    band_log_e: &[i16],
    mut start: usize,
    mut end: usize,
    m: usize,
    downsample: usize,
    silence: bool,
) {
    let n = m * 120;
    let mut bound = m * EBANDS[end] as usize;
    if downsample != 1 {
        bound = bound.min(n / downsample);
    }
    if silence {
        bound = 0;
        start = 0;
        end = 0;
    }
    let s0 = m * EBANDS[start] as usize;
    freq[..s0].fill(0);
    let mut f = s0;
    let mut xi = s0;
    for i in start..end {
        let band_end = m * EBANDS[i + 1] as usize;
        let lg = sat16_i32((band_log_e[i] as i32).wrapping_add(shl32(E_MEANS[i] as i32, 6)));
        let mut shift = 16 - (lg >> DB_SHIFT);
        let mut g;
        if shift > 31 {
            shift = 0;
            g = 0;
        } else {
            g = exp2_frac(lg & ((1 << DB_SHIFT) - 1)) as i16 as i32;
        }
        let len = band_end - m * EBANDS[i] as usize;
        let (out, inp) = (&mut freq[f..f + len], &x[xi..xi + len]);
        if shift <= -2 {
            g = 16384;
            shift = -2;
        }
        denormalise_band(out, inp, g, shift);
        f += len;
        xi += len;
    }
    freq[bound..n].fill(0);
}

/// One band of [`denormalise_bands`]: `out[i] = MULT16_16(x[i], g)` shifted right by `shift`
/// (left for a negative one).
pub(super) fn denormalise_band(out: &mut [i32], x: &[i16], g: i32, shift: i32) {
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        let done = unsafe { super::simd::denormalise_avx2(out, x, g, shift) };
        denormalise_band_scalar(&mut out[done..], &x[done..], g, shift);
        return;
    }
    denormalise_band_scalar(out, x, g, shift);
}

/// Scalar reference of [`denormalise_band`].
pub fn denormalise_band_scalar(out: &mut [i32], x: &[i16], g: i32, shift: i32) {
    if shift < 0 {
        for (o, &v) in out.iter_mut().zip(x) {
            *o = shl32(mul16(v as i32, g), -shift);
        }
    } else {
        for (o, &v) in out.iter_mut().zip(x) {
            *o = mul16(v as i32, g) >> shift;
        }
    }
}

/// `anti_collapse`: fills collapsed short blocks with noise.
#[allow(clippy::too_many_arguments)]
pub fn anti_collapse(
    x_: &mut [i16],
    collapse_masks: &[u8],
    lm: i32,
    c: usize,
    size: usize,
    start: usize,
    end: usize,
    log_e: &[i16],
    prev1_log_e: &[i16],
    prev2_log_e: &[i16],
    pulses: &[i32],
    mut seed: u32,
) {
    for i in start..end {
        let n0 = (EBANDS[i + 1] - EBANDS[i]) as i32;
        let depth = (celt_udiv((1 + pulses[i]) as u32, n0 as u32) as i32) >> lm;
        let thresh32 = exp2(-shl16(depth, 10 - BITRES)) >> 1;
        let thresh = mul16_32_q15(16384, thresh32.min(32767)) as i16 as i32;
        let sqrt_1;
        let shift;
        {
            let mut t = n0 << lm;
            shift = ilog2(t) >> 1;
            t = shl32(t, (7 - shift) << 1);
            sqrt_1 = rsqrt_norm(t) as i16 as i32;
        }
        for ch in 0..c {
            let mut prev1 = prev1_log_e[ch * NB_EBANDS + i] as i32;
            let mut prev2 = prev2_log_e[ch * NB_EBANDS + i] as i32;
            if c == 1 {
                prev1 = prev1.max(prev1_log_e[NB_EBANDS + i] as i32);
                prev2 = prev2.max(prev2_log_e[NB_EBANDS + i] as i32);
            }
            let ediff = (log_e[ch * NB_EBANDS + i] as i32 - prev1.min(prev2)).max(0);
            let mut r: i32 = if ediff < 16384 {
                let r32 = exp2(-(ediff as i16 as i32)) >> 1;
                (2 * r32.min(16383)) as i16 as i32
            } else {
                0
            };
            if lm == 3 {
                r = mul16_q14(23170, r.min(23169)) as i16 as i32;
            }
            r = (thresh.min(r)) >> 1;
            r = (mul16_q15(sqrt_1, r) >> shift) as i16 as i32;
            let xo = ch * size + ((EBANDS[i] as usize) << lm);
            let mut renormalize = false;
            for k in 0..1usize << lm {
                if collapse_masks[i * c + ch] & (1 << k) == 0 {
                    for j in 0..n0 as usize {
                        seed = lcg_rand(seed);
                        x_[xo + (j << lm) + k] = if seed & 0x8000 != 0 { r } else { -r } as i16;
                    }
                    renormalize = true;
                }
            }
            if renormalize {
                renormalise_vector(&mut x_[xo..], (n0 << lm) as usize, Q15ONE);
            }
        }
    }
}

/// `out[i] = MULT16_16_Q15(g, x[i])` truncated to 16 bits.
pub(super) fn mul_q15(out: &mut [i16], x: &[i16], g: i32) {
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        if unsafe { super::simd::mul_q15_avx2(out, x, g) } {
            return;
        }
    }
    mul_q15_scalar(out, x, g);
}

/// Scalar reference of [`mul_q15`].
pub fn mul_q15_scalar(out: &mut [i16], x: &[i16], g: i32) {
    for (o, &v) in out.iter_mut().zip(x) {
        *o = mul16_q15(g, v as i32) as i16;
    }
}

fn compute_channel_weights(ex: i32, ey: i32) -> [i32; 2] {
    let min_e = ex.min(ey);
    let ex = ex.wrapping_add(min_e / 3);
    let ey = ey.wrapping_add(min_e / 3);
    let shift = ilog2(1i32.wrapping_add(ex.max(ey))) - 14;
    [
        vshr32(ex, shift) as i16 as i32,
        vshr32(ey, shift) as i16 as i32,
    ]
}

fn intensity_stereo(x: &mut [i16], y: &[i16], band_e: &[i32], i: usize, n: usize) {
    let shift = zlog2(band_e[i].max(band_e[i + NB_EBANDS])) - 13;
    let left = vshr32(band_e[i], shift) as i16 as i32;
    let right = vshr32(band_e[i + NB_EBANDS], shift) as i16 as i32;
    let norm = (1 + sqrt(1 + mul16(left, left) + mul16(right, right))) as i16 as i32;
    let a1 = (shl32(left, 14) / norm) as i16 as i32;
    let a2 = (shl32(right, 14) / norm) as i16 as i32;
    for j in 0..n {
        let (l, r) = (x[j] as i32, y[j] as i32);
        x[j] = (mac16(mul16(a1, l), a2, r) >> 14) as i16;
    }
}

fn stereo_split(x: &mut [i16], y: &mut [i16], n: usize) {
    for j in 0..n {
        let l = mul16(23170, x[j] as i32);
        let r = mul16(23170, y[j] as i32);
        x[j] = ((l + r) >> 15) as i16;
        y[j] = ((r - l) >> 15) as i16;
    }
}

fn stereo_merge(x: &mut [i16], y: &mut [i16], mid: i32, n: usize) {
    let mut xp = inner_prod(&y[..n], &x[..n]);
    let side = inner_prod(&y[..n], &y[..n]);
    xp = mul16_32_q15(mid, xp);
    let mid2 = (mid as i16 as i32) >> 1;
    let el = mul16(mid2, mid2).wrapping_add(side).wrapping_sub(2 * xp);
    let er = mul16(mid2, mid2).wrapping_add(side).wrapping_add(2 * xp);
    if er < 161061 || el < 161061 {
        y[..n].copy_from_slice(&x[..n]);
        return;
    }
    let mut kl = ilog2(el) >> 1;
    let mut kr = ilog2(er) >> 1;
    let t = vshr32(el, (kl - 7) << 1);
    let lgain = rsqrt_norm(t);
    let t = vshr32(er, (kr - 7) << 1);
    let rgain = rsqrt_norm(t);
    if kl < 7 {
        kl = 7;
    }
    if kr < 7 {
        kr = 7;
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        if unsafe {
            super::simd::stereo_merge_avx2(
                &mut x[..n],
                &mut y[..n],
                mid,
                lgain,
                kl + 1,
                rgain,
                kr + 1,
            )
        } {
            return;
        }
    }
    stereo_merge_scalar(&mut x[..n], &mut y[..n], mid, lgain, kl + 1, rgain, kr + 1);
}

/// Scalar reference of the final loop of `stereo_merge`.
pub fn stereo_merge_scalar(
    x: &mut [i16],
    y: &mut [i16],
    mid: i32,
    lgain: i32,
    lsh: i32,
    rgain: i32,
    rsh: i32,
) {
    for (xv, yv) in x.iter_mut().zip(y.iter_mut()) {
        let l = mul16_p15(mid, *xv as i32) as i16 as i32;
        let r = *yv as i32;
        *xv = pshr32(mul16(lgain, sub16(l, r)), lsh) as i16;
        *yv = pshr32(mul16(rgain, add16(l, r)), rsh) as i16;
    }
}

const ORDERY_TABLE: [usize; 30] = [
    1, 0, 3, 0, 2, 1, 7, 0, 4, 3, 6, 1, 5, 2, 15, 0, 8, 7, 12, 3, 11, 4, 14, 1, 9, 6, 13, 2, 10, 5,
];

fn deinterleave_hadamard(x: &mut [i16], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    let mut tmp = [0i16; MAX_BAND];
    let tmp = &mut tmp[..n];
    if hadamard {
        let ordery = &ORDERY_TABLE[stride - 2..];
        for i in 0..stride {
            for j in 0..n0 {
                tmp[ordery[i] * n0 + j] = x[j * stride + i];
            }
        }
    } else {
        for i in 0..stride {
            for j in 0..n0 {
                tmp[i * n0 + j] = x[j * stride + i];
            }
        }
    }
    x[..n].copy_from_slice(tmp);
}

fn interleave_hadamard(x: &mut [i16], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    let mut tmp = [0i16; MAX_BAND];
    let tmp = &mut tmp[..n];
    if hadamard {
        let ordery = &ORDERY_TABLE[stride - 2..];
        for i in 0..stride {
            for j in 0..n0 {
                tmp[j * stride + i] = x[ordery[i] * n0 + j];
            }
        }
    } else {
        for i in 0..stride {
            for j in 0..n0 {
                tmp[j * stride + i] = x[i * n0 + j];
            }
        }
    }
    x[..n].copy_from_slice(tmp);
}

/// `haar1`.
pub fn haar1(x: &mut [i16], n0: usize, stride: usize) {
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        let done = unsafe { super::simd::haar1_avx2(x, n0, stride) };
        haar1_scalar(x, n0, stride, done / (2 * stride));
        return;
    }
    haar1_scalar(x, n0, stride, 0);
}

/// Scalar reference of [`haar1`], from pair `j0` of every run on.
pub fn haar1_scalar(x: &mut [i16], n0: usize, stride: usize, j0: usize) {
    let n0 = n0 >> 1;
    for i in 0..stride {
        for j in j0..n0 {
            let a = stride * 2 * j + i;
            let b = stride * (2 * j + 1) + i;
            let tmp1 = mul16(23170, x[a] as i32);
            let tmp2 = mul16(23170, x[b] as i32);
            x[a] = pshr32(tmp1 + tmp2, 15) as i16;
            x[b] = pshr32(tmp1 - tmp2, 15) as i16;
        }
    }
}

fn compute_qn(n: i32, b: i32, offset: i32, pulse_cap: i32, stereo: bool) -> i32 {
    const EXP2_TABLE8: [i32; 8] = [16384, 17866, 19483, 21247, 23170, 25267, 27554, 30048];
    let mut n2 = 2 * n - 1;
    if stereo && n == 2 {
        n2 -= 1;
    }
    let mut qb = (b + n2 * offset) / n2;
    qb = qb.min(b - pulse_cap - (4 << BITRES));
    qb = qb.min(8 << BITRES);
    if qb < (1 << BITRES >> 1) {
        1
    } else {
        let qn = EXP2_TABLE8[(qb & 0x7) as usize] >> (14 - (qb >> BITRES));
        ((qn + 1) >> 1) << 1
    }
}

/// State shared by the band quantisation of one frame (`struct band_ctx`).
pub struct BandCtx<'a> {
    pub encode: bool,
    pub resynth: bool,
    pub i: usize,
    pub intensity: i32,
    pub spread: i32,
    pub tf_change: i32,
    pub ec: &'a mut Ec,
    pub remaining_bits: i32,
    pub band_e: &'a [i32],
    pub seed: u32,
    pub theta_round: i32,
    pub disable_inv: bool,
    pub avoid_split_noise: bool,
}

struct Split {
    inv: bool,
    imid: i32,
    iside: i32,
    delta: i32,
    itheta: i32,
    qalloc: i32,
}

#[allow(clippy::too_many_arguments)]
fn compute_theta(
    ctx: &mut BandCtx<'_>,
    x: &mut [i16],
    y: &mut [i16],
    n: usize,
    b: &mut i32,
    bb: i32,
    b0: i32,
    lm: i32,
    stereo: bool,
    fill: &mut i32,
) -> Split {
    let i = ctx.i;
    let ni = n as i32;
    let pulse_cap = LOG_N[i] as i32 + lm * (1 << BITRES);
    let offset = (pulse_cap >> 1)
        - if stereo && n == 2 {
            QTHETA_OFFSET_TWOPHASE
        } else {
            QTHETA_OFFSET
        };
    let mut qn = compute_qn(ni, *b, offset, pulse_cap, stereo);
    if stereo && i as i32 >= ctx.intensity {
        qn = 1;
    }
    let mut itheta = 0i32;
    if ctx.encode {
        itheta = stereo_itheta(x, y, stereo, n);
    }
    let tell = ctx.ec.tell_frac() as i32;
    let mut inv = false;
    if qn != 1 {
        if ctx.encode {
            if !stereo || ctx.theta_round == 0 {
                itheta = (itheta * qn + 8192) >> 14;
                if !stereo && ctx.avoid_split_noise && itheta > 0 && itheta < qn {
                    let unquantized = celt_udiv((itheta * 16384) as u32, qn as u32) as i32;
                    let imid = bitexact_cos(unquantized);
                    let iside = bitexact_cos(16384 - unquantized);
                    let delta = frac_mul16((ni - 1) << 7, bitexact_log2tan(iside, imid));
                    if delta > *b {
                        itheta = qn;
                    } else if delta < -*b {
                        itheta = 0;
                    }
                }
            } else {
                let bias = if itheta > 8192 {
                    32767 / qn
                } else {
                    -32767 / qn
                };
                let down = (qn - 1).min(0.max((itheta * qn + bias) >> 14));
                itheta = if ctx.theta_round < 0 { down } else { down + 1 };
            }
        }
        if stereo && n > 2 {
            let p0 = 3;
            let x0 = qn / 2;
            let ft = (p0 * (x0 + 1) + x0) as u32;
            let mut xv = itheta;
            if ctx.encode {
                let (fl, fh) = if xv <= x0 {
                    (p0 * xv, p0 * (xv + 1))
                } else {
                    ((xv - 1 - x0) + (x0 + 1) * p0, (xv - x0) + (x0 + 1) * p0)
                };
                ctx.ec.encode(fl as u32, fh as u32, ft);
            } else {
                let fs = ctx.ec.decode(ft) as i32;
                xv = if fs < (x0 + 1) * p0 {
                    fs / p0
                } else {
                    x0 + 1 + (fs - (x0 + 1) * p0)
                };
                let (fl, fh) = if xv <= x0 {
                    (p0 * xv, p0 * (xv + 1))
                } else {
                    ((xv - 1 - x0) + (x0 + 1) * p0, (xv - x0) + (x0 + 1) * p0)
                };
                ctx.ec.dec_update(fl as u32, fh as u32, ft);
                itheta = xv;
            }
        } else if b0 > 1 || stereo {
            if ctx.encode {
                ctx.ec.enc_uint(itheta as u32, (qn + 1) as u32);
            } else {
                itheta = ctx.ec.dec_uint((qn + 1) as u32) as i32;
            }
        } else {
            let ft = ((qn >> 1) + 1) * ((qn >> 1) + 1);
            if ctx.encode {
                let fs = if itheta <= (qn >> 1) {
                    itheta + 1
                } else {
                    qn + 1 - itheta
                };
                let fl = if itheta <= (qn >> 1) {
                    (itheta * (itheta + 1)) >> 1
                } else {
                    ft - (((qn + 1 - itheta) * (qn + 2 - itheta)) >> 1)
                };
                ctx.ec.encode(fl as u32, (fl + fs) as u32, ft as u32);
            } else {
                let fm = ctx.ec.decode(ft as u32) as i32;
                let (fs, fl);
                if fm < (((qn >> 1) * ((qn >> 1) + 1)) >> 1) {
                    itheta = (triangle_isqrt(fm as u32) - 1) >> 1;
                    fs = itheta + 1;
                    fl = (itheta * (itheta + 1)) >> 1;
                } else {
                    itheta = (2 * (qn + 1) - triangle_isqrt((ft - fm - 1) as u32)) >> 1;
                    fs = qn + 1 - itheta;
                    fl = ft - (((qn + 1 - itheta) * (qn + 2 - itheta)) >> 1);
                }
                ctx.ec.dec_update(fl as u32, (fl + fs) as u32, ft as u32);
            }
        }
        itheta = celt_udiv((itheta * 16384) as u32, qn as u32) as i32;
        if ctx.encode && stereo {
            if itheta == 0 {
                intensity_stereo(x, y, ctx.band_e, i, n);
            } else {
                stereo_split(x, y, n);
            }
        }
    } else if stereo {
        if ctx.encode {
            inv = itheta > 8192 && !ctx.disable_inv;
            if inv {
                for v in y[..n].iter_mut() {
                    *v = v.wrapping_neg();
                }
            }
            intensity_stereo(x, y, ctx.band_e, i, n);
        }
        if *b > 2 << BITRES && ctx.remaining_bits > 2 << BITRES {
            if ctx.encode {
                ctx.ec.enc_bit_logp(inv, 2);
            } else {
                inv = ctx.ec.dec_bit_logp(2);
            }
        } else {
            inv = false;
        }
        if ctx.disable_inv {
            inv = false;
        }
        itheta = 0;
    }
    let qalloc = ctx.ec.tell_frac() as i32 - tell;
    *b -= qalloc;
    let (imid, iside, delta);
    if itheta == 0 {
        imid = 32767;
        iside = 0;
        *fill &= (1 << bb) - 1;
        delta = -16384;
    } else if itheta == 16384 {
        imid = 0;
        iside = 32767;
        *fill &= ((1 << bb) - 1) << bb;
        delta = 16384;
    } else {
        let l2t;
        (imid, iside, l2t) = theta_gains(itheta);
        delta = frac_mul16((ni - 1) << 7, l2t);
    }
    Split {
        inv,
        imid,
        iside,
        delta,
        itheta,
        qalloc,
    }
}

fn quant_band_n1(
    ctx: &mut BandCtx<'_>,
    x: &mut [i16],
    mut y: Option<&mut [i16]>,
    lowband_out: Option<&mut [i16]>,
) -> u32 {
    let stereo = y.is_some();
    for c in 0..1 + stereo as usize {
        let v: &mut [i16] = if c == 0 {
            &mut *x
        } else {
            y.as_deref_mut().expect("stereo")
        };
        let mut sign = 0u32;
        if ctx.remaining_bits >= 1 << BITRES {
            if ctx.encode {
                sign = (v[0] < 0) as u32;
                ctx.ec.enc_bits(sign, 1);
            } else {
                sign = ctx.ec.dec_bits(1);
            }
            ctx.remaining_bits -= 1 << BITRES;
        }
        if ctx.resynth {
            v[0] = if sign != 0 {
                -NORM_SCALING as i16
            } else {
                NORM_SCALING as i16
            };
        }
    }
    if let Some(lo) = lowband_out {
        lo[0] = x[0] >> 4;
    }
    1
}

#[allow(clippy::too_many_arguments)]
fn quant_partition(
    ctx: &mut BandCtx<'_>,
    x: &mut [i16],
    mut n: usize,
    mut b: i32,
    mut bb: i32,
    lowband: Option<&[i16]>,
    mut lm: i32,
    gain: i32,
    mut fill: i32,
) -> u32 {
    let i = ctx.i;
    let b0 = bb;
    let mut cm = 0u32;
    if lm != -1 && b > SPLIT_LIMIT[((lm + 1) as usize) * NB_EBANDS + i] && n > 2 {
        n >>= 1;
        let (xh, yh) = x.split_at_mut(n);
        lm -= 1;
        if bb == 1 {
            fill = (fill & 1) | (fill << 1);
        }
        bb = (bb + 1) >> 1;
        let sctx = compute_theta(ctx, xh, yh, n, &mut b, bb, b0, lm, false, &mut fill);
        let (imid, iside, itheta, qalloc) = (sctx.imid, sctx.iside, sctx.itheta, sctx.qalloc);
        let mut delta = sctx.delta;
        let mid = imid;
        let side = iside;
        if b0 > 1 && (itheta & 0x3fff) != 0 {
            if itheta > 8192 {
                delta -= delta >> (4 - lm);
            } else {
                delta = 0.min(delta + ((n as i32) << BITRES >> (5 - lm)));
            }
        }
        let mut mbits = 0.max(b.min((b - delta) / 2));
        let mut sbits = b - mbits;
        ctx.remaining_bits -= qalloc;
        let (lb1, lb2) = match lowband {
            Some(l) => (Some(&l[..n]), Some(&l[n..])),
            None => (None, None),
        };
        let mut rebalance = ctx.remaining_bits;
        if mbits >= sbits {
            cm = quant_partition(ctx, xh, n, mbits, bb, lb1, lm, mul16_p15(gain, mid), fill);
            rebalance = mbits - (rebalance - ctx.remaining_bits);
            if rebalance > 3 << BITRES && itheta != 0 {
                sbits += rebalance - (3 << BITRES);
            }
            cm |= quant_partition(
                ctx,
                yh,
                n,
                sbits,
                bb,
                lb2,
                lm,
                mul16_p15(gain, side),
                fill >> bb,
            ) << (b0 >> 1);
        } else {
            cm = quant_partition(
                ctx,
                yh,
                n,
                sbits,
                bb,
                lb2,
                lm,
                mul16_p15(gain, side),
                fill >> bb,
            ) << (b0 >> 1);
            rebalance = sbits - (rebalance - ctx.remaining_bits);
            if rebalance > 3 << BITRES && itheta != 16384 {
                mbits += rebalance - (3 << BITRES);
            }
            cm |= quant_partition(ctx, xh, n, mbits, bb, lb1, lm, mul16_p15(gain, mid), fill);
        }
    } else {
        let mut q = bits2pulses(i, lm, b);
        let mut curr_bits = pulses2bits(i, lm, q);
        ctx.remaining_bits -= curr_bits;
        while ctx.remaining_bits < 0 && q > 0 {
            ctx.remaining_bits += curr_bits;
            q -= 1;
            curr_bits = pulses2bits(i, lm, q);
            ctx.remaining_bits -= curr_bits;
        }
        if q != 0 {
            let k = get_pulses(q) as usize;
            if ctx.encode {
                cm = alg_quant(x, n, k, ctx.spread, bb as usize, ctx.ec, gain, ctx.resynth);
            } else {
                cm = alg_unquant(x, n, k, ctx.spread, bb as usize, ctx.ec, gain);
            }
        } else if ctx.resynth {
            let cm_mask = ((1u64 << bb) - 1) as u32;
            fill &= cm_mask as i32;
            if fill == 0 {
                x[..n].fill(0);
            } else {
                match lowband {
                    None => {
                        for v in x[..n].iter_mut() {
                            ctx.seed = lcg_rand(ctx.seed);
                            *v = ((ctx.seed as i32) >> 20) as i16;
                        }
                        cm = cm_mask;
                    }
                    Some(lb) => {
                        for j in 0..n {
                            ctx.seed = lcg_rand(ctx.seed);
                            let tmp: i32 = if ctx.seed & 0x8000 != 0 { 4 } else { -4 };
                            x[j] = (lb[j] as i32 + tmp) as i16;
                        }
                        cm = fill as u32;
                    }
                }
                renormalise_vector(x, n, gain);
            }
        }
    }
    cm
}

#[allow(clippy::too_many_arguments)]
fn quant_band(
    ctx: &mut BandCtx<'_>,
    x: &mut [i16],
    n: usize,
    b: i32,
    bb: i32,
    lowband: Option<&[i16]>,
    lm: i32,
    lowband_out: Option<&mut [i16]>,
    gain: i32,
    mut fill: i32,
) -> u32 {
    let n0 = n;
    let mut n_b = n;
    let mut b_cnt = bb;
    let b0_orig = bb;
    let mut time_divide = 0;
    let mut recombine = 0;
    let long_blocks = b0_orig == 1;
    let encode = ctx.encode;
    let mut tf_change = ctx.tf_change;
    n_b /= b_cnt as usize;
    if n == 1 {
        return quant_band_n1(ctx, x, None, lowband_out);
    }
    if tf_change > 0 {
        recombine = tf_change;
    }
    // The folding source is transformed along with the band only when the band is; only
    // then does it need a copy (as in libopus).
    let transformed = recombine > 0 || ((n_b & 1) == 0 && tf_change < 0) || b0_orig > 1;
    let mut lb_store: [i16; MAX_BAND];
    let (mut lb, lb_direct): (Option<&mut [i16]>, Option<&[i16]>) = match lowband {
        Some(l) if transformed => {
            lb_store = [0; MAX_BAND];
            lb_store[..n].copy_from_slice(&l[..n]);
            (Some(&mut lb_store[..n]), None)
        }
        l => (None, l),
    };
    const BIT_INTERLEAVE_TABLE: [i32; 16] = [0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3];
    for k in 0..recombine {
        if encode {
            haar1(x, n >> k, 1 << k);
        }
        if let Some(l) = lb.as_deref_mut() {
            haar1(l, n >> k, 1 << k);
        }
        fill = BIT_INTERLEAVE_TABLE[(fill & 0xF) as usize]
            | (BIT_INTERLEAVE_TABLE[(fill >> 4) as usize] << 2);
    }
    b_cnt >>= recombine;
    n_b <<= recombine;
    while (n_b & 1) == 0 && tf_change < 0 {
        if encode {
            haar1(x, n_b, b_cnt as usize);
        }
        if let Some(l) = lb.as_deref_mut() {
            haar1(l, n_b, b_cnt as usize);
        }
        fill |= fill << b_cnt;
        b_cnt <<= 1;
        n_b >>= 1;
        time_divide += 1;
        tf_change += 1;
    }
    let b0 = b_cnt;
    let n_b0 = n_b;
    if b0 > 1 {
        if encode {
            deinterleave_hadamard(x, n_b >> recombine, (b0 << recombine) as usize, long_blocks);
        }
        if let Some(l) = lb.as_deref_mut() {
            deinterleave_hadamard(l, n_b >> recombine, (b0 << recombine) as usize, long_blocks);
        }
    }
    let lowband = lb.as_deref().or(lb_direct);
    let mut cm = quant_partition(ctx, x, n, b, b_cnt, lowband, lm, gain, fill);
    if ctx.resynth {
        if b0 > 1 {
            interleave_hadamard(x, n_b >> recombine, (b0 << recombine) as usize, long_blocks);
        }
        n_b = n_b0;
        b_cnt = b0;
        for _ in 0..time_divide {
            b_cnt >>= 1;
            n_b <<= 1;
            cm |= cm >> b_cnt;
            haar1(x, n_b, b_cnt as usize);
        }
        const BIT_DEINTERLEAVE_TABLE: [u32; 16] = [
            0x00, 0x03, 0x0C, 0x0F, 0x30, 0x33, 0x3C, 0x3F, 0xC0, 0xC3, 0xCC, 0xCF, 0xF0, 0xF3,
            0xFC, 0xFF,
        ];
        for k in 0..recombine {
            cm = BIT_DEINTERLEAVE_TABLE[cm as usize];
            haar1(x, n0 >> k, 1 << k);
        }
        b_cnt <<= recombine;
        if let Some(lo) = lowband_out {
            let nn = sqrt(shl32(n0 as i32, 22));
            mul_q15(&mut lo[..n0], &x[..n0], nn);
        }
        cm &= ((1u64 << b_cnt) - 1) as u32;
    }
    cm
}

#[allow(clippy::too_many_arguments)]
fn quant_band_stereo(
    ctx: &mut BandCtx<'_>,
    x: &mut [i16],
    y: &mut [i16],
    n: usize,
    mut b: i32,
    bb: i32,
    lowband: Option<&[i16]>,
    lm: i32,
    lowband_out: Option<&mut [i16]>,
    fill: i32,
) -> u32 {
    if n == 1 {
        return quant_band_n1(ctx, x, Some(y), lowband_out);
    }
    let orig_fill = fill;
    let mut fill = fill;
    let sctx = compute_theta(ctx, x, y, n, &mut b, bb, bb, lm, true, &mut fill);
    let (inv, imid, iside, delta, itheta, qalloc) = (
        sctx.inv,
        sctx.imid,
        sctx.iside,
        sctx.delta,
        sctx.itheta,
        sctx.qalloc,
    );
    let mid = imid;
    let side = iside;
    let mut cm;
    if n == 2 {
        let mut mbits = b;
        let mut sbits = 0;
        if itheta != 0 && itheta != 16384 {
            sbits = 1 << BITRES;
        }
        mbits -= sbits;
        let c = itheta > 8192;
        ctx.remaining_bits -= qalloc + sbits;
        let (x2, y2): (&mut [i16], &mut [i16]) = if c { (y, x) } else { (x, y) };
        let mut sign = 0i32;
        if sbits != 0 {
            if ctx.encode {
                sign = ((x2[0] as i32 * y2[1] as i32 - x2[1] as i32 * y2[0] as i32) < 0) as i32;
                ctx.ec.enc_bits(sign as u32, 1);
            } else {
                sign = ctx.ec.dec_bits(1) as i32;
            }
        }
        sign = 1 - 2 * sign;
        cm = quant_band(
            ctx,
            x2,
            n,
            mbits,
            bb,
            lowband,
            lm,
            lowband_out,
            Q15ONE,
            orig_fill,
        );
        y2[0] = (-sign * x2[1] as i32) as i16;
        y2[1] = (sign * x2[0] as i32) as i16;
        if ctx.resynth {
            let (xx, yy): (&mut [i16], &mut [i16]) = if c { (y2, x2) } else { (x2, y2) };
            xx[0] = mul16_q15(mid, xx[0] as i32) as i16;
            xx[1] = mul16_q15(mid, xx[1] as i32) as i16;
            yy[0] = mul16_q15(side, yy[0] as i32) as i16;
            yy[1] = mul16_q15(side, yy[1] as i32) as i16;
            let tmp = xx[0] as i32;
            xx[0] = sub16(tmp, yy[0] as i32) as i16;
            yy[0] = add16(tmp, yy[0] as i32) as i16;
            let tmp = xx[1] as i32;
            xx[1] = sub16(tmp, yy[1] as i32) as i16;
            yy[1] = add16(tmp, yy[1] as i32) as i16;
            if inv {
                for v in yy[..n].iter_mut() {
                    *v = v.wrapping_neg();
                }
            }
        }
        return cm;
    }
    let mut mbits = 0.max(b.min((b - delta) / 2));
    let mut sbits = b - mbits;
    ctx.remaining_bits -= qalloc;
    let mut rebalance = ctx.remaining_bits;
    if mbits >= sbits {
        cm = quant_band(ctx, x, n, mbits, bb, lowband, lm, lowband_out, Q15ONE, fill);
        rebalance = mbits - (rebalance - ctx.remaining_bits);
        if rebalance > 3 << BITRES && itheta != 0 {
            sbits += rebalance - (3 << BITRES);
        }
        cm |= quant_band(ctx, y, n, sbits, bb, None, lm, None, side, fill >> bb);
    } else {
        cm = quant_band(ctx, y, n, sbits, bb, None, lm, None, side, fill >> bb);
        rebalance = sbits - (rebalance - ctx.remaining_bits);
        if rebalance > 3 << BITRES && itheta != 16384 {
            mbits += rebalance - (3 << BITRES);
        }
        cm |= quant_band(ctx, x, n, mbits, bb, lowband, lm, lowband_out, Q15ONE, fill);
    }
    if ctx.resynth {
        stereo_merge(x, y, mid, n);
        if inv {
            for v in y[..n].iter_mut() {
                *v = v.wrapping_neg();
            }
        }
    }
    cm
}

fn special_hybrid_folding(norm: &mut [i16], norm2_off: usize, start: usize, m: usize, dual: bool) {
    let n1 = m * (EBANDS[start + 1] - EBANDS[start]) as usize;
    let n2 = m * (EBANDS[start + 2] - EBANDS[start + 1]) as usize;
    if n2 > n1 {
        norm.copy_within(2 * n1 - n2..n1, n1);
        if dual {
            norm.copy_within(norm2_off + 2 * n1 - n2..norm2_off + n1, norm2_off + n1);
        }
    }
}

/// Where a band's folding source (`lowband`, at `eff` unless that is -1) and its
/// `lowband_out` (at `out_off`, unless it is the last band) lie in the normalised spectrum.
struct Lowband {
    eff: i32,
    out_off: usize,
    n: usize,
    last: bool,
}

impl Lowband {
    /// The two regions of `norm[base..]`, which libopus aliases. The source ends at or before
    /// `lowband_out` unless it starts at 0 in a band wider than its offset; only then is it
    /// copied into `scratch`.
    fn split<'a>(
        &self,
        norm: &'a mut [i16],
        base: usize,
        scratch: &'a mut [i16; MAX_BAND],
    ) -> (Option<&'a [i16]>, Option<&'a mut [i16]>) {
        let n = self.n;
        let lo = base + self.out_off;
        if self.eff < 0 {
            return (None, (!self.last).then(|| &mut norm[lo..lo + n]));
        }
        let ls = base + self.eff as usize;
        if self.last {
            return (Some(&norm[ls..ls + n]), None);
        }
        if ls + n <= lo {
            let (head, tail) = norm.split_at_mut(lo);
            (Some(&head[ls..ls + n]), Some(&mut tail[..n]))
        } else {
            scratch[..n].copy_from_slice(&norm[ls..ls + n]);
            (Some(&scratch[..n]), Some(&mut norm[lo..lo + n]))
        }
    }
}

/// Inputs of [`quant_all_bands`] besides the coefficient buffer.
pub struct AllBands<'a> {
    pub encode: bool,
    pub start: usize,
    pub end: usize,
    pub collapse_masks: &'a mut [u8],
    pub band_e: &'a [i32],
    pub pulses: &'a [i32],
    pub short_blocks: bool,
    pub spread: i32,
    pub dual_stereo: bool,
    pub intensity: i32,
    pub tf_res: &'a [i32],
    pub total_bits: i32,
    pub balance: i32,
    pub lm: i32,
    pub coded_bands: i32,
    pub disable_inv: bool,
    /// Encoder complexity; 8 and above enables the stereo theta RDO search.
    pub complexity: i32,
    /// Working memory for the normalised spectrum that later bands fold from, at least
    /// `C * 8 * 78` values. It needs no clearing: every value read is written first.
    pub norm: &'a mut [i16],
}

/// `quant_all_bands`: codes every band of `x` (`C` channels of `120 << LM` coefficients).
pub fn quant_all_bands(p: AllBands<'_>, x: &mut [i16], c: usize, ec: &mut Ec, seed: &mut u32) {
    let AllBands {
        encode,
        start,
        end,
        collapse_masks,
        band_e,
        pulses,
        short_blocks,
        spread,
        mut dual_stereo,
        intensity,
        tf_res,
        total_bits,
        mut balance,
        lm,
        coded_bands,
        disable_inv,
        complexity,
        norm: norm_scratch,
    } = p;
    let m = 1usize << lm;
    let bb: i32 = if short_blocks { m as i32 } else { 1 };
    let norm_offset = m * EBANDS[start] as usize;
    let nsize = m * EBANDS[NB_EBANDS - 1] as usize - norm_offset;
    let norm = &mut norm_scratch[..c * nsize];
    let norm2_off = nsize;
    let theta_rdo = encode && c == 2 && !dual_stereo && complexity >= 8;
    let resynth = !encode || theta_rdo;
    let mut lowband_offset = 0usize;
    let mut update_lowband = true;
    let mut lb_scratch = [0i16; MAX_BAND];
    let frame_n = 120usize << lm;
    let (x_buf, y_buf) = x.split_at_mut(frame_n);
    let mut ctx = BandCtx {
        encode,
        resynth,
        i: 0,
        intensity,
        spread,
        tf_change: 0,
        ec,
        remaining_bits: 0,
        band_e,
        seed: *seed,
        theta_round: 0,
        disable_inv,
        avoid_split_noise: bb > 1,
    };
    for i in start..end {
        ctx.i = i;
        let last = i == end - 1;
        let xs = m * EBANDS[i] as usize;
        let n = m * EBANDS[i + 1] as usize - xs;
        let tell = ctx.ec.tell_frac() as i32;
        if i != start {
            balance -= tell;
        }
        let remaining_bits = total_bits - tell - 1;
        ctx.remaining_bits = remaining_bits;
        let b = if (i as i32) < coded_bands {
            let curr_balance = balance / 3.min(coded_bands - i as i32);
            0.max(16383.min((remaining_bits + 1).min(pulses[i] + curr_balance)))
        } else {
            0
        };
        if resynth
            && (m as i32 * eb(i) - n as i32 >= m as i32 * eb(start) || i == start + 1)
            && (update_lowband || lowband_offset == 0)
        {
            lowband_offset = i;
        }
        if i == start + 1 {
            special_hybrid_folding(norm, norm2_off, start, m, dual_stereo);
        }
        let tf_change = tf_res[i];
        ctx.tf_change = tf_change;
        let mut effective_lowband: i32 = -1;
        let (mut x_cm, mut y_cm);
        if lowband_offset != 0 && (spread != SPREAD_AGGRESSIVE || bb > 1 || tf_change < 0) {
            effective_lowband =
                0.max(m as i32 * eb(lowband_offset) - norm_offset as i32 - n as i32);
            let mut fold_start = lowband_offset;
            loop {
                fold_start -= 1;
                if m as i32 * eb(fold_start) <= effective_lowband + norm_offset as i32 {
                    break;
                }
            }
            let mut fold_end = lowband_offset - 1;
            loop {
                fold_end += 1;
                if !(fold_end < i
                    && (m as i32 * eb(fold_end))
                        < effective_lowband + norm_offset as i32 + n as i32)
                {
                    break;
                }
            }
            x_cm = 0u32;
            y_cm = 0u32;
            let mut fold_i = fold_start;
            loop {
                x_cm |= collapse_masks[fold_i * c] as u32;
                y_cm |= collapse_masks[fold_i * c + c - 1] as u32;
                fold_i += 1;
                if fold_i >= fold_end {
                    break;
                }
            }
        } else {
            x_cm = ((1u64 << bb) - 1) as u32;
            y_cm = x_cm;
        }
        if dual_stereo && i as i32 == intensity {
            dual_stereo = false;
            if resynth {
                for j in 0..m * EBANDS[i] as usize - norm_offset {
                    norm[j] = ((norm[j] as i32 + norm[norm2_off + j] as i32) >> 1) as i16;
                }
            }
        }
        let out_off = xs - norm_offset;
        let lbs = Lowband {
            eff: effective_lowband,
            out_off,
            n,
            last,
        };
        if dual_stereo {
            let (lb, lo) = lbs.split(norm, 0, &mut lb_scratch);
            x_cm = quant_band(
                &mut ctx,
                &mut x_buf[xs..xs + n],
                n,
                b / 2,
                bb,
                lb,
                lm,
                lo,
                Q15ONE,
                x_cm as i32,
            );
            let (lb2, lo2) = lbs.split(norm, norm2_off, &mut lb_scratch);
            y_cm = quant_band(
                &mut ctx,
                &mut y_buf[xs..xs + n],
                n,
                b / 2,
                bb,
                lb2,
                lm,
                lo2,
                Q15ONE,
                y_cm as i32,
            );
        } else {
            if c == 2 && theta_rdo && (i as i32) < intensity {
                // The folding source, copied as it may overlap this band's `lowband_out`.
                let lb = (effective_lowband != -1).then(|| {
                    let o = effective_lowband as usize;
                    let mut a = [0i16; MAX_BAND];
                    a[..n].copy_from_slice(&norm[o..o + n]);
                    a
                });
                // Code the band with theta rounded down and up; keep the less distorted.
                let w = compute_channel_weights(band_e[i], band_e[i + NB_EBANDS]);
                let cm = (x_cm | y_cm) as i32;
                let mut ec_save = EcSave::default();
                ctx.ec.save(&mut ec_save);
                let (seed_save, rem_save) = (ctx.seed, ctx.remaining_bits);
                let (mut xb, mut yb) = ([0i16; MAX_BAND], [0i16; MAX_BAND]);
                let (x_save, y_save) = (&mut xb[..n], &mut yb[..n]);
                x_save.copy_from_slice(&x_buf[xs..xs + n]);
                y_save.copy_from_slice(&y_buf[xs..xs + n]);
                ctx.theta_round = -1;
                x_cm = quant_band_stereo(
                    &mut ctx,
                    &mut x_buf[xs..xs + n],
                    &mut y_buf[xs..xs + n],
                    n,
                    b,
                    bb,
                    lb.as_ref().map(|a| &a[..n]),
                    lm,
                    (!last).then(|| &mut norm[out_off..out_off + n]),
                    cm,
                );
                let dist0 = mul16_32_q15(w[0], inner_prod(x_save, &x_buf[xs..xs + n]))
                    + mul16_32_q15(w[1], inner_prod(y_save, &y_buf[xs..xs + n]));
                let cm2 = x_cm;
                let mut ec_save2 = EcSave::default();
                ctx.ec.save(&mut ec_save2);
                let (seed_save2, rem_save2) = (ctx.seed, ctx.remaining_bits);
                let (mut xb2, mut yb2, mut nb2) =
                    ([0i16; MAX_BAND], [0i16; MAX_BAND], [0i16; MAX_BAND]);
                let (x_save2, y_save2) = (&mut xb2[..n], &mut yb2[..n]);
                x_save2.copy_from_slice(&x_buf[xs..xs + n]);
                y_save2.copy_from_slice(&y_buf[xs..xs + n]);
                if !last {
                    nb2[..n].copy_from_slice(&norm[out_off..out_off + n]);
                }
                ctx.ec.restore(&ec_save);
                ctx.seed = seed_save;
                ctx.remaining_bits = rem_save;
                x_buf[xs..xs + n].copy_from_slice(x_save);
                y_buf[xs..xs + n].copy_from_slice(y_save);
                if i == start + 1 {
                    special_hybrid_folding(norm, norm2_off, start, m, dual_stereo);
                }
                ctx.theta_round = 1;
                x_cm = quant_band_stereo(
                    &mut ctx,
                    &mut x_buf[xs..xs + n],
                    &mut y_buf[xs..xs + n],
                    n,
                    b,
                    bb,
                    lb.as_ref().map(|a| &a[..n]),
                    lm,
                    (!last).then(|| &mut norm[out_off..out_off + n]),
                    cm,
                );
                let dist1 = mul16_32_q15(w[0], inner_prod(x_save, &x_buf[xs..xs + n]))
                    + mul16_32_q15(w[1], inner_prod(y_save, &y_buf[xs..xs + n]));
                if dist0 >= dist1 {
                    x_cm = cm2;
                    ctx.ec.restore(&ec_save2);
                    ctx.seed = seed_save2;
                    ctx.remaining_bits = rem_save2;
                    ctx.theta_round = -1;
                    x_buf[xs..xs + n].copy_from_slice(x_save2);
                    y_buf[xs..xs + n].copy_from_slice(y_save2);
                    if !last {
                        norm[out_off..out_off + n].copy_from_slice(&nb2[..n]);
                    }
                }
            } else if c == 2 {
                ctx.theta_round = 0;
                let (lb, lo) = lbs.split(norm, 0, &mut lb_scratch);
                x_cm = quant_band_stereo(
                    &mut ctx,
                    &mut x_buf[xs..xs + n],
                    &mut y_buf[xs..xs + n],
                    n,
                    b,
                    bb,
                    lb,
                    lm,
                    lo,
                    (x_cm | y_cm) as i32,
                );
            } else {
                let (lb, lo) = lbs.split(norm, 0, &mut lb_scratch);
                x_cm = quant_band(
                    &mut ctx,
                    &mut x_buf[xs..xs + n],
                    n,
                    b,
                    bb,
                    lb,
                    lm,
                    lo,
                    Q15ONE,
                    (x_cm | y_cm) as i32,
                );
            }
            y_cm = x_cm;
        }
        collapse_masks[i * c] = x_cm as u8;
        collapse_masks[i * c + c - 1] = y_cm as u8;
        balance += pulses[i] + tell;
        update_lowband = b > (n as i32) << BITRES;
        ctx.avoid_split_noise = false;
    }
    *seed = ctx.seed;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_tables_match() {
        use super::super::celt_tables::{CACHE_BITS, CACHE_INDEX};
        for (j, &idx) in CACHE_INDEX.iter().enumerate() {
            if idx >= 0 {
                let c = &CACHE_BITS[idx as usize..];
                assert_eq!(SPLIT_LIMIT[j], c[c[0] as usize] as i32 + 12, "{j}");
            }
        }
        for x in 0..TRIANGLE_ISQRT.len() as u32 + 10 {
            assert_eq!(
                triangle_isqrt(x),
                super::super::math::isqrt32(8 * x + 1) as i32
            );
        }
    }

    #[test]
    fn theta_gains_match_polynomials() {
        for t in THETA_MIN as i32..=THETA_MAX as i32 {
            let (imid, iside) = (bitexact_cos(t), bitexact_cos(16384 - t));
            assert_eq!(
                theta_gains(t),
                (imid, iside, bitexact_log2tan(iside, imid)),
                "{t}"
            );
        }
        // Every angle a coded theta takes is either an edge or in the tables' range.
        for qn in 2..=256 {
            for q in 1..qn {
                let t = (q * 16384 / qn) as usize;
                assert!((THETA_MIN..=THETA_MAX).contains(&t), "{q} / {qn}");
            }
        }
    }
}
