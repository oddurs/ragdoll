//! SILK fixed-point primitives, LPC helpers and the decoder-side resampler, ported from
//! libopus `silk/` (`SigProc_FIX.h`, `macros.h`, `Inlines.h`, `NLSF2A.c`, `LPC_fit.c`,
//! `LPC_inv_pred_gain.c`, `NLSF_stabilize.c`, `bwexpander*.c`, `resampler*.c`). The 64-bit
//! forms of the `SMULW*`/`SMLAW*` macros are used, matching `OPUS_FAST_INT64` builds.

use super::silk_tables::*;

#[inline]
pub fn smulwb(a: i32, b: i32) -> i32 {
    ((a as i64 * (b as i16) as i64) >> 16) as i32
}

#[inline]
pub fn smlawb(a: i32, b: i32, c: i32) -> i32 {
    (a as i64 + ((b as i64 * (c as i16) as i64) >> 16)) as i32
}

#[inline]
pub fn smulww(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> 16) as i32
}

#[inline]
pub fn smlaww(a: i32, b: i32, c: i32) -> i32 {
    (a as i64 + ((b as i64 * c as i64) >> 16)) as i32
}

#[inline]
pub fn smulbb(a: i32, b: i32) -> i32 {
    (a as i16 as i32) * (b as i16 as i32)
}

#[inline]
pub fn smlabb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(smulbb(b, c))
}

#[inline]
pub fn smultt(a: i32, b: i32) -> i32 {
    (a >> 16) * (b >> 16)
}

#[inline]
pub fn smmul(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> 32) as i32
}

#[inline]
pub fn rshift_round(a: i32, s: i32) -> i32 {
    if s == 1 {
        (a >> 1) + (a & 1)
    } else {
        ((a >> (s - 1)) + 1) >> 1
    }
}

#[inline]
pub fn rshift_round64(a: i64, s: i32) -> i64 {
    if s == 1 {
        (a >> 1) + (a & 1)
    } else {
        ((a >> (s - 1)) + 1) >> 1
    }
}

#[inline]
pub fn sat16(a: i32) -> i32 {
    a.clamp(i16::MIN as i32, i16::MAX as i32)
}

/// `silk_LIMIT`: the bounds may come in either order.
#[inline]
pub fn limit(a: i32, l1: i32, l2: i32) -> i32 {
    if l1 > l2 {
        if a > l1 {
            l1
        } else if a < l2 {
            l2
        } else {
            a
        }
    } else if a > l2 {
        l2
    } else if a < l1 {
        l1
    } else {
        a
    }
}

#[inline]
pub fn lshift_sat32(a: i32, s: i32) -> i32 {
    limit(a, i32::MIN >> s, i32::MAX >> s).wrapping_shl(s as u32)
}

#[inline]
pub fn clz32(x: i32) -> i32 {
    (x as u32).leading_zeros() as i32
}

#[inline]
pub fn ror32(a: i32, rot: i32) -> i32 {
    (a as u32).rotate_right(rot.rem_euclid(32) as u32) as i32
}

#[inline]
pub fn silk_rand(seed: i32) -> i32 {
    907_633_515i32.wrapping_add(seed.wrapping_mul(196_314_165))
}

#[inline]
fn clz_frac(x: i32) -> (i32, i32) {
    let lz = clz32(x);
    (lz, ror32(x, 24 - lz) & 0x7f)
}

pub fn sqrt_approx(x: i32) -> i32 {
    if x <= 0 {
        return 0;
    }
    let (lz, frac_q7) = clz_frac(x);
    let mut y = if lz & 1 != 0 { 32768 } else { 46214 };
    y >>= lz >> 1;
    smlawb(y, y, smulbb(213, frac_q7))
}

pub fn div32_varq(a32: i32, b32: i32, qres: i32) -> i32 {
    let a_headrm = clz32(a32.wrapping_abs()) - 1;
    let mut a32_nrm = a32.wrapping_shl(a_headrm as u32);
    let b_headrm = clz32(b32.wrapping_abs()) - 1;
    let b32_nrm = b32.wrapping_shl(b_headrm as u32);
    let b32_inv = (i32::MAX >> 2) / (b32_nrm >> 16);
    let mut result = smulwb(a32_nrm, b32_inv);
    a32_nrm = a32_nrm.wrapping_sub(smmul(b32_nrm, result).wrapping_shl(3));
    result = smlawb(result, a32_nrm, b32_inv);
    let lshift = 29 + a_headrm - b_headrm - qres;
    if lshift < 0 {
        lshift_sat32(result, -lshift)
    } else if lshift < 32 {
        result >> lshift
    } else {
        0
    }
}

pub fn inverse32_varq(b32: i32, qres: i32) -> i32 {
    let b_headrm = clz32(b32.wrapping_abs()) - 1;
    let b32_nrm = b32.wrapping_shl(b_headrm as u32);
    let b32_inv = (i32::MAX >> 2) / (b32_nrm >> 16);
    let mut result = b32_inv.wrapping_shl(16);
    let err_q32 = ((1i32 << 29) - smulwb(b32_nrm, b32_inv)).wrapping_shl(3);
    result = smlaww(result, err_q32, b32_inv);
    let lshift = 61 - b_headrm - qres;
    if lshift <= 0 {
        lshift_sat32(result, -lshift)
    } else if lshift < 32 {
        result >> lshift
    } else {
        0
    }
}

pub fn log2lin(in_log_q7: i32) -> i32 {
    if in_log_q7 < 0 {
        return 0;
    } else if in_log_q7 >= 3967 {
        return i32::MAX;
    }
    let out = 1i32 << (in_log_q7 >> 7);
    let frac_q7 = in_log_q7 & 0x7f;
    let poly = smlawb(frac_q7, smulbb(frac_q7, 128 - frac_q7), -174);
    if in_log_q7 < 2048 {
        out + ((out * poly) >> 7)
    } else {
        out + (out >> 7) * poly
    }
}

/// `silk_sum_sqr_shift`: energy of `x` with a shift that keeps it in 31 bits.
pub fn sum_sqr_shift(x: &[i16]) -> (i32, i32) {
    let len = x.len();
    let pass = |shft: i32, init: i32| -> i32 {
        let mut nrg = init;
        let mut i = 0;
        while i + 1 < len {
            let t = (smulbb(x[i] as i32, x[i] as i32) as u32)
                .wrapping_add(smulbb(x[i + 1] as i32, x[i + 1] as i32) as u32);
            nrg = (nrg as u32).wrapping_add(t >> shft) as i32;
            i += 2;
        }
        if i < len {
            let t = smulbb(x[i] as i32, x[i] as i32) as u32;
            nrg = (nrg as u32).wrapping_add(t >> shft) as i32;
        }
        nrg
    };
    let mut shft = 31 - clz32(len as i32);
    let nrg = pass(shft, len as i32);
    shft = 0.max(shft + 3 - clz32(nrg));
    (pass(shft, 0), shft)
}

pub fn bwexpander(ar: &mut [i16], mut chirp_q16: i32) {
    let d = ar.len();
    let chirp_minus_one_q16 = chirp_q16 - 65536;
    for v in ar[..d - 1].iter_mut() {
        *v = rshift_round(chirp_q16.wrapping_mul(*v as i32), 16) as i16;
        chirp_q16 += rshift_round(chirp_q16.wrapping_mul(chirp_minus_one_q16), 16);
    }
    ar[d - 1] = rshift_round(chirp_q16.wrapping_mul(ar[d - 1] as i32), 16) as i16;
}

pub fn bwexpander_32(ar: &mut [i32], mut chirp_q16: i32) {
    let d = ar.len();
    let chirp_minus_one_q16 = chirp_q16 - 65536;
    for v in ar[..d - 1].iter_mut() {
        *v = smulww(chirp_q16, *v);
        chirp_q16 += rshift_round(chirp_q16.wrapping_mul(chirp_minus_one_q16), 16);
    }
    ar[d - 1] = smulww(chirp_q16, ar[d - 1]);
}

/// `silk_LPC_analysis_filter`: the first `d` outputs are zero.
pub fn lpc_analysis_filter(out: &mut [i16], input: &[i16], b: &[i16], len: usize, d: usize) {
    for ix in d..len {
        let mut acc = smulbb(input[ix - 1] as i32, b[0] as i32);
        for j in 1..d {
            acc = smlabb(acc, input[ix - 1 - j] as i32, b[j] as i32);
        }
        let acc = ((input[ix] as i32) << 12).wrapping_sub(acc);
        out[ix] = sat16(rshift_round(acc, 12)) as i16;
    }
    out[..d].fill(0);
}

const QA_INV: i32 = 24;
const A_LIMIT: i32 = 16_773_022;

fn lpc_inverse_pred_gain_qa(a_qa: &mut [i32], order: usize) -> i32 {
    let mut inv_gain_q30 = 1i32 << 30;
    let mut k = order - 1;
    while k > 0 {
        if a_qa[k] > A_LIMIT || a_qa[k] < -A_LIMIT {
            return 0;
        }
        let rc_q31 = -(a_qa[k] << (31 - QA_INV));
        let rc_mult1_q30 = (1i32 << 30) - smmul(rc_q31, rc_q31);
        inv_gain_q30 = smmul(inv_gain_q30, rc_mult1_q30) << 2;
        if inv_gain_q30 < 107_374 {
            return 0;
        }
        let mult2q = 32 - clz32(rc_mult1_q30.abs());
        let rc_mult2 = inverse32_varq(rc_mult1_q30, mult2q + 30);
        for n in 0..(k + 1) >> 1 {
            let tmp1 = a_qa[n];
            let tmp2 = a_qa[k - n - 1];
            let frac = |a: i32| rshift_round64(a as i64 * rc_q31 as i64, 31) as i32;
            let t = rshift_round64(
                tmp1.saturating_sub(frac(tmp2)) as i64 * rc_mult2 as i64,
                mult2q,
            );
            if t > i32::MAX as i64 || t < i32::MIN as i64 {
                return 0;
            }
            a_qa[n] = t as i32;
            let t = rshift_round64(
                tmp2.saturating_sub(frac(tmp1)) as i64 * rc_mult2 as i64,
                mult2q,
            );
            if t > i32::MAX as i64 || t < i32::MIN as i64 {
                return 0;
            }
            a_qa[k - n - 1] = t as i32;
        }
        k -= 1;
    }
    if a_qa[0] > A_LIMIT || a_qa[0] < -A_LIMIT {
        return 0;
    }
    let rc_q31 = -(a_qa[0] << (31 - QA_INV));
    let rc_mult1_q30 = (1i32 << 30) - smmul(rc_q31, rc_q31);
    inv_gain_q30 = smmul(inv_gain_q30, rc_mult1_q30) << 2;
    if inv_gain_q30 < 107_374 {
        return 0;
    }
    inv_gain_q30
}

/// `silk_LPC_inverse_pred_gain`: 0 when the filter is unstable.
pub fn lpc_inverse_pred_gain(a_q12: &[i16]) -> i32 {
    let mut a_qa = [0i32; 24];
    let mut dc_resp = 0i32;
    for (k, &a) in a_q12.iter().enumerate() {
        dc_resp += a as i32;
        a_qa[k] = (a as i32) << (QA_INV - 12);
    }
    if dc_resp >= 4096 {
        return 0;
    }
    lpc_inverse_pred_gain_qa(&mut a_qa, a_q12.len())
}

fn lpc_fit(a_qout: &mut [i16], a_qin: &mut [i32], qout: i32, qin: i32, d: usize) {
    let mut i = 0;
    while i < 10 {
        let mut maxabs = 0i32;
        let mut idx = 0usize;
        for (k, &v) in a_qin[..d].iter().enumerate() {
            let absval = v.abs();
            if absval > maxabs {
                maxabs = absval;
                idx = k;
            }
        }
        maxabs = rshift_round(maxabs, qin - qout);
        if maxabs > i16::MAX as i32 {
            maxabs = maxabs.min(163_838);
            let chirp_q16 =
                65470 - ((maxabs - i16::MAX as i32) << 14) / ((maxabs * (idx as i32 + 1)) >> 2);
            bwexpander_32(&mut a_qin[..d], chirp_q16);
        } else {
            break;
        }
        i += 1;
    }
    if i == 10 {
        for k in 0..d {
            a_qout[k] = sat16(rshift_round(a_qin[k], qin - qout)) as i16;
            a_qin[k] = (a_qout[k] as i32) << (qin - qout);
        }
    } else {
        for k in 0..d {
            a_qout[k] = rshift_round(a_qin[k], qin - qout) as i16;
        }
    }
}

const QA: i32 = 16;

fn nlsf2a_find_poly(out: &mut [i32], c_lsf: &[i32], off: usize, dd: usize) {
    out[0] = 1 << QA;
    out[1] = -c_lsf[off];
    for k in 1..dd {
        let ftmp = c_lsf[off + 2 * k] as i64;
        out[k + 1] = (out[k - 1] << 1) - rshift_round64(ftmp * out[k] as i64, QA) as i32;
        let mut n = k;
        while n > 1 {
            out[n] += out[n - 2] - rshift_round64(ftmp * out[n - 1] as i64, QA) as i32;
            n -= 1;
        }
        out[1] -= ftmp as i32;
    }
}

/// `silk_NLSF2A`: NLSFs (Q15) to stable LPC coefficients (Q12).
pub fn nlsf2a(a_q12: &mut [i16], nlsf: &[i16], d: usize) {
    const ORDERING16: [usize; 16] = [0, 15, 8, 7, 4, 11, 12, 3, 2, 13, 10, 5, 6, 9, 14, 1];
    const ORDERING10: [usize; 10] = [0, 9, 6, 3, 4, 5, 8, 1, 2, 7];
    let ordering: &[usize] = if d == 16 { &ORDERING16 } else { &ORDERING10 };
    let mut cos_lsf_qa = [0i32; 24];
    for k in 0..d {
        let f_int = (nlsf[k] as i32) >> (15 - 7);
        let f_frac = nlsf[k] as i32 - (f_int << (15 - 7));
        let cos_val = LSFCOSTAB_FIX_Q12[f_int as usize] as i32;
        let delta = LSFCOSTAB_FIX_Q12[f_int as usize + 1] as i32 - cos_val;
        cos_lsf_qa[ordering[k]] = rshift_round((cos_val << 8) + delta * f_frac, 20 - QA);
    }
    let dd = d >> 1;
    let mut p = [0i32; 13];
    let mut q = [0i32; 13];
    nlsf2a_find_poly(&mut p, &cos_lsf_qa, 0, dd);
    nlsf2a_find_poly(&mut q, &cos_lsf_qa, 1, dd);
    let mut a32_qa1 = [0i32; 24];
    for k in 0..dd {
        let ptmp = p[k + 1] + p[k];
        let qtmp = q[k + 1] - q[k];
        a32_qa1[k] = -qtmp - ptmp;
        a32_qa1[d - k - 1] = qtmp - ptmp;
    }
    lpc_fit(a_q12, &mut a32_qa1, 12, QA + 1, d);
    let mut i = 0;
    while lpc_inverse_pred_gain(&a_q12[..d]) == 0 && i < 16 {
        bwexpander_32(&mut a32_qa1[..d], 65536 - (2 << i));
        for k in 0..d {
            a_q12[k] = rshift_round(a32_qa1[k], QA + 1 - 12) as i16;
        }
        i += 1;
    }
}

/// `silk_NLSF_stabilize`: enforce minimum spacing between NLSFs.
pub fn nlsf_stabilize(nlsf: &mut [i16], delta_min: &[i16], l: usize) {
    let mut loops = 0;
    while loops < 20 {
        let mut min_diff = nlsf[0] as i32 - delta_min[0] as i32;
        let mut ii = 0usize;
        for i in 1..l {
            let diff = nlsf[i] as i32 - (nlsf[i - 1] as i32 + delta_min[i] as i32);
            if diff < min_diff {
                min_diff = diff;
                ii = i;
            }
        }
        let diff = (1 << 15) - (nlsf[l - 1] as i32 + delta_min[l] as i32);
        if diff < min_diff {
            min_diff = diff;
            ii = l;
        }
        if min_diff >= 0 {
            return;
        }
        if ii == 0 {
            nlsf[0] = delta_min[0];
        } else if ii == l {
            nlsf[l - 1] = ((1 << 15) - delta_min[l] as i32) as i16;
        } else {
            let mut min_center = 0i32;
            for k in 0..ii {
                min_center += delta_min[k] as i32;
            }
            min_center += (delta_min[ii] as i32) >> 1;
            let mut max_center = 1i32 << 15;
            for k in (ii + 1..=l).rev() {
                max_center -= delta_min[k] as i32;
            }
            max_center -= (delta_min[ii] as i32) >> 1;
            let center = limit(
                rshift_round(nlsf[ii - 1] as i32 + nlsf[ii] as i32, 1),
                min_center,
                max_center,
            ) as i16;
            nlsf[ii - 1] = (center as i32 - ((delta_min[ii] as i32) >> 1)) as i16;
            nlsf[ii] = (nlsf[ii - 1] as i32 + delta_min[ii] as i32) as i16;
        }
        loops += 1;
    }
    // Fallback after MAX_LOOPS: sort, then clamp from both ends.
    for i in 1..l {
        let value = nlsf[i];
        let mut j = i as isize - 1;
        while j >= 0 && value < nlsf[j as usize] {
            nlsf[j as usize + 1] = nlsf[j as usize];
            j -= 1;
        }
        nlsf[(j + 1) as usize] = value;
    }
    nlsf[0] = nlsf[0].max(delta_min[0]);
    for i in 1..l {
        let floor = sat16(nlsf[i - 1] as i32 + delta_min[i] as i32) as i16;
        nlsf[i] = nlsf[i].max(floor);
    }
    nlsf[l - 1] = (nlsf[l - 1] as i32).min((1 << 15) - delta_min[l] as i32) as i16;
    for i in (0..l - 1).rev() {
        nlsf[i] = (nlsf[i] as i32).min(nlsf[i + 1] as i32 - delta_min[i + 1] as i32) as i16;
    }
}

const RESAMPLER_ORDER_FIR_12: usize = 8;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum ResamplerKind {
    #[default]
    Copy,
    Up2Hq,
    IirFir,
    DownFir,
}

/// `silk_resampler_state_struct` (decoder direction only).
#[derive(Clone, Debug)]
pub struct Resampler {
    s_iir: [i32; 6],
    s_fir_i32: [i32; 36],
    s_fir_i16: [i16; 8],
    delay_buf: [i16; 48],
    kind: ResamplerKind,
    batch_size: usize,
    inv_ratio_q16: i32,
    fir_order: usize,
    fir_fracs: i32,
    fs_in_khz: usize,
    fs_out_khz: usize,
    input_delay: usize,
    coefs: &'static [i16],
}

impl Default for Resampler {
    fn default() -> Self {
        Resampler {
            s_iir: [0; 6],
            s_fir_i32: [0; 36],
            s_fir_i16: [0; 8],
            delay_buf: [0; 48],
            kind: ResamplerKind::Copy,
            batch_size: 0,
            inv_ratio_q16: 0,
            fir_order: 0,
            fir_fracs: 0,
            fs_in_khz: 0,
            fs_out_khz: 0,
            input_delay: 0,
            coefs: &[],
        }
    }
}

const DELAY_MATRIX_DEC: [[u8; 5]; 3] = [[4, 0, 2, 0, 0], [0, 9, 4, 7, 4], [0, 3, 12, 7, 7]];

fn rate_id(r: i32) -> usize {
    ((((r >> 12) - (r > 16000) as i32) >> (r > 24000) as i32) - 1) as usize
}

impl Resampler {
    /// `silk_resampler_init(.., forEnc = 0)`.
    pub fn new(fs_in: i32, fs_out: i32) -> Self {
        let mut s = Resampler {
            input_delay: DELAY_MATRIX_DEC[rate_id(fs_in)][rate_id(fs_out)] as usize,
            fs_in_khz: (fs_in / 1000) as usize,
            fs_out_khz: (fs_out / 1000) as usize,
            ..Default::default()
        };
        s.batch_size = s.fs_in_khz * 10;
        let mut up2x = 0;
        if fs_out > fs_in {
            if fs_out == fs_in * 2 {
                s.kind = ResamplerKind::Up2Hq;
            } else {
                s.kind = ResamplerKind::IirFir;
                up2x = 1;
            }
        } else if fs_out < fs_in {
            s.kind = ResamplerKind::DownFir;
            let (fracs, order, coefs): (i32, usize, &'static [i16]) = if fs_out * 4 == fs_in * 3 {
                (3, 18, &RESAMPLER_3_4_COEFS)
            } else if fs_out * 3 == fs_in * 2 {
                (2, 18, &RESAMPLER_2_3_COEFS)
            } else if fs_out * 2 == fs_in {
                (1, 24, &RESAMPLER_1_2_COEFS)
            } else if fs_out * 3 == fs_in {
                (1, 36, &RESAMPLER_1_3_COEFS)
            } else if fs_out * 4 == fs_in {
                (1, 36, &RESAMPLER_1_4_COEFS)
            } else {
                (1, 36, &RESAMPLER_1_6_COEFS)
            };
            s.fir_fracs = fracs;
            s.fir_order = order;
            s.coefs = coefs;
        }
        s.inv_ratio_q16 = ((fs_in << (14 + up2x)) / fs_out) << 2;
        while smulww(s.inv_ratio_q16, fs_out) < (fs_in << up2x) {
            s.inv_ratio_q16 += 1;
        }
        s
    }

    /// `silk_resampler`: `input.len()` must be at least 1 ms of input.
    pub fn process(&mut self, out: &mut [i16], input: &[i16]) {
        let in_len = input.len();
        let n_samples = self.fs_in_khz - self.input_delay;
        let fin = self.fs_in_khz;
        let fout = self.fs_out_khz;
        self.delay_buf[self.input_delay..fin].copy_from_slice(&input[..n_samples]);
        let delay: [i16; 48] = self.delay_buf;
        match self.kind {
            ResamplerKind::Up2Hq => {
                up2_hq(&mut self.s_iir, out, &delay[..fin]);
                up2_hq(
                    &mut self.s_iir,
                    &mut out[fout..],
                    &input[n_samples..n_samples + in_len - fin],
                );
            }
            ResamplerKind::IirFir => {
                self.iir_fir(out, &delay[..fin]);
                self.iir_fir(
                    &mut out[fout..],
                    &input[n_samples..n_samples + in_len - fin],
                );
            }
            ResamplerKind::DownFir => {
                self.down_fir(out, &delay[..fin]);
                self.down_fir(
                    &mut out[fout..],
                    &input[n_samples..n_samples + in_len - fin],
                );
            }
            ResamplerKind::Copy => {
                out[..fin].copy_from_slice(&delay[..fin]);
                out[fout..fout + in_len - fin]
                    .copy_from_slice(&input[n_samples..n_samples + in_len - fin]);
            }
        }
        let d = self.input_delay;
        self.delay_buf[..d].copy_from_slice(&input[in_len - d..]);
    }

    fn iir_fir(&mut self, out: &mut [i16], mut input: &[i16]) {
        let mut buf = vec![0i16; 2 * self.batch_size + RESAMPLER_ORDER_FIR_12];
        buf[..RESAMPLER_ORDER_FIR_12].copy_from_slice(&self.s_fir_i16);
        let inc = self.inv_ratio_q16;
        let mut o = 0usize;
        let mut n_in;
        loop {
            n_in = input.len().min(self.batch_size);
            up2_hq(
                &mut self.s_iir,
                &mut buf[RESAMPLER_ORDER_FIR_12..],
                &input[..n_in],
            );
            let max_index_q16 = (n_in as i32) << (16 + 1);
            let mut index_q16 = 0i32;
            while index_q16 < max_index_q16 {
                let ti = smulwb(index_q16 & 0xffff, 12) as usize;
                let bp = &buf[(index_q16 >> 16) as usize..];
                let f = |r: usize, c: usize| RESAMPLER_FRAC_FIR_12[r * 4 + c] as i32;
                let mut res = smulbb(bp[0] as i32, f(ti, 0));
                res = smlabb(res, bp[1] as i32, f(ti, 1));
                res = smlabb(res, bp[2] as i32, f(ti, 2));
                res = smlabb(res, bp[3] as i32, f(ti, 3));
                res = smlabb(res, bp[4] as i32, f(11 - ti, 3));
                res = smlabb(res, bp[5] as i32, f(11 - ti, 2));
                res = smlabb(res, bp[6] as i32, f(11 - ti, 1));
                res = smlabb(res, bp[7] as i32, f(11 - ti, 0));
                out[o] = sat16(rshift_round(res, 15)) as i16;
                o += 1;
                index_q16 += inc;
            }
            input = &input[n_in..];
            if !input.is_empty() {
                buf.copy_within(n_in << 1..(n_in << 1) + RESAMPLER_ORDER_FIR_12, 0);
            } else {
                break;
            }
        }
        self.s_fir_i16
            .copy_from_slice(&buf[n_in << 1..(n_in << 1) + RESAMPLER_ORDER_FIR_12]);
    }

    fn down_fir(&mut self, out: &mut [i16], mut input: &[i16]) {
        let order = self.fir_order;
        let mut buf = vec![0i32; self.batch_size + order];
        buf[..order].copy_from_slice(&self.s_fir_i32[..order]);
        let coefs = &self.coefs[2..];
        let inc = self.inv_ratio_q16;
        let mut o = 0usize;
        let mut n_in;
        loop {
            n_in = input.len().min(self.batch_size);
            ar2(
                &mut self.s_iir,
                &mut buf[order..],
                &input[..n_in],
                self.coefs,
            );
            let max_index_q16 = (n_in as i32) << 16;
            let mut index_q16 = 0i32;
            while index_q16 < max_index_q16 {
                let bp = &buf[(index_q16 >> 16) as usize..];
                let mut res;
                if order == 18 {
                    let ii = smulwb(index_q16 & 0xffff, self.fir_fracs) as usize;
                    let ip = &coefs[9 * ii..];
                    res = smulwb(bp[0], ip[0] as i32);
                    for k in 1..9 {
                        res = smlawb(res, bp[k], ip[k] as i32);
                    }
                    let ip = &coefs[9 * (self.fir_fracs as usize - 1 - ii)..];
                    for k in 0..9 {
                        res = smlawb(res, bp[17 - k], ip[k] as i32);
                    }
                } else {
                    let half = order / 2;
                    res = smulwb(bp[0].wrapping_add(bp[order - 1]), coefs[0] as i32);
                    for k in 1..half {
                        res = smlawb(res, bp[k].wrapping_add(bp[order - 1 - k]), coefs[k] as i32);
                    }
                }
                out[o] = sat16(rshift_round(res, 6)) as i16;
                o += 1;
                index_q16 += inc;
            }
            input = &input[n_in..];
            if input.len() > 1 {
                buf.copy_within(n_in..n_in + order, 0);
            } else {
                break;
            }
        }
        self.s_fir_i32[..order].copy_from_slice(&buf[n_in..n_in + order]);
    }
}

const UP2_HQ_0: [i32; 3] = [1746, 14986, 39083 - 65536];
const UP2_HQ_1: [i32; 3] = [6854, 25769, 55542 - 65536];

fn up2_hq(s: &mut [i32; 6], out: &mut [i16], input: &[i16]) {
    for (k, &x) in input.iter().enumerate() {
        let in32 = (x as i32) << 10;
        let y = in32 - s[0];
        let xx = smulwb(y, UP2_HQ_0[0]);
        let out32_1 = s[0] + xx;
        s[0] = in32 + xx;
        let y = out32_1 - s[1];
        let xx = smulwb(y, UP2_HQ_0[1]);
        let out32_2 = s[1] + xx;
        s[1] = out32_1 + xx;
        let y = out32_2 - s[2];
        let xx = smlawb(y, y, UP2_HQ_0[2]);
        let out32_1 = s[2] + xx;
        s[2] = out32_2 + xx;
        out[2 * k] = sat16(rshift_round(out32_1, 10)) as i16;

        let y = in32 - s[3];
        let xx = smulwb(y, UP2_HQ_1[0]);
        let out32_1 = s[3] + xx;
        s[3] = in32 + xx;
        let y = out32_1 - s[4];
        let xx = smulwb(y, UP2_HQ_1[1]);
        let out32_2 = s[4] + xx;
        s[4] = out32_1 + xx;
        let y = out32_2 - s[5];
        let xx = smlawb(y, y, UP2_HQ_1[2]);
        let out32_1 = s[5] + xx;
        s[5] = out32_2 + xx;
        out[2 * k + 1] = sat16(rshift_round(out32_1, 10)) as i16;
    }
}

fn ar2(s: &mut [i32; 6], out_q8: &mut [i32], input: &[i16], a_q14: &[i16]) {
    for (k, &x) in input.iter().enumerate() {
        let out32 = s[0].wrapping_add((x as i32) << 8);
        out_q8[k] = out32;
        let out32 = out32 << 2;
        s[0] = smlawb(s[1], out32, a_q14[0] as i32);
        s[1] = smulwb(out32, a_q14[1] as i32);
    }
}
