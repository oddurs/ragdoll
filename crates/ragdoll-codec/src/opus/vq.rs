//! Pyramid vector quantisation: codeword enumeration (`cwrs.c`) and the PVQ search, spreading
//! rotation and renormalisation (`vq.c`).

use std::sync::OnceLock;

use super::math::{
    add16, atan2p, cos_norm, div, ilog2, inner_prod, mac16, mul16, mul16_32_q16, mul16_p15,
    mul16_q15, pshr32, rcp, rsqrt_norm, sqrt, vshr32,
};
use super::range::Ec;

pub const SPREAD_NONE: i32 = 0;
#[allow(dead_code)] // Encoder side, used by the CELT encoder (BUG-0078).
pub const SPREAD_LIGHT: i32 = 1;
pub const SPREAD_NORMAL: i32 = 2;
pub const SPREAD_AGGRESSIVE: i32 = 3;

/// Columns of the `U(n, k)` table; rows are indexed by `min(n, k) <= 14`.
const U_COLS: usize = 210;

/// `CELT_PVQ_U_ROW`: U(n, k), the number of n-dimensional codewords with k pulses and a positive
/// first element, by `[min(n, k)][max(n, k)]`. Built from the recurrence
/// U(n, k) = U(n-1, k) + U(n, k-1) + U(n-1, k-1); entries libopus uses fit 32 bits.
fn u_table() -> &'static [[u32; U_COLS]; 15] {
    static T: OnceLock<Box<[[u32; U_COLS]; 15]>> = OnceLock::new();
    T.get_or_init(|| {
        // Full symmetric table u[n][k] for n < 15, k < U_COLS, in 64 bits (overflowing entries
        // are never read by valid codebooks).
        let mut u = vec![[0u64; U_COLS]; 15];
        u[0][0] = 1;
        for n in 1..15 {
            for k in 1..U_COLS {
                u[n][k] = u[n - 1][k]
                    .wrapping_add(u[n][k - 1])
                    .wrapping_add(u[n - 1][k - 1]);
            }
        }
        let mut t = Box::new([[0u32; U_COLS]; 15]);
        for (row, src) in t.iter_mut().zip(&u) {
            for (d, s) in row.iter_mut().zip(src) {
                *d = *s as u32;
            }
        }
        t
    })
}

#[inline]
fn pvq_u(n: usize, k: usize) -> u32 {
    u_table()[n.min(k)][n.max(k)]
}

/// `CELT_PVQ_V`: number of codewords with `k` pulses in `n` dimensions.
#[inline]
pub fn pvq_v(n: usize, k: usize) -> u32 {
    pvq_u(n, k).wrapping_add(pvq_u(n, k + 1))
}

/// `icwrs`: index of the codeword `y`.
fn icwrs(n: usize, y: &[i32]) -> u32 {
    let mut j = n - 1;
    let mut i = (y[j] < 0) as u32;
    let mut k = y[j].unsigned_abs() as usize;
    loop {
        j -= 1;
        i = i.wrapping_add(pvq_u(n - j, k));
        k += y[j].unsigned_abs() as usize;
        if y[j] < 0 {
            i = i.wrapping_add(pvq_u(n - j, k + 1));
        }
        if j == 0 {
            break;
        }
    }
    i
}

pub fn encode_pulses(y: &[i32], n: usize, k: usize, enc: &mut Ec) {
    enc.enc_uint(icwrs(n, y), pvq_v(n, k));
}

/// `cwrsi`: codeword of index `i`; returns its squared norm.
fn cwrsi(mut n: usize, mut k: usize, mut i: u32, y: &mut [i32]) -> i32 {
    let rows = u_table();
    let mut yy = 0i32;
    let mut yi = 0usize;
    while n > 2 {
        if k >= n {
            let row = &rows[n];
            let mut p = row[k + 1];
            let s = -((i >= p) as i32);
            i -= p & s as u32;
            let k0 = k;
            let q = row[n];
            if q > i {
                k = n;
                loop {
                    k -= 1;
                    p = rows[k][n];
                    if p <= i {
                        break;
                    }
                }
            } else {
                p = row[k];
                while p > i {
                    k -= 1;
                    p = row[k];
                }
            }
            i -= p;
            let val = ((k0 as i32 - k as i32 + s) ^ s) as i16;
            y[yi] = val as i32;
            yi += 1;
            yy = mac16(yy, val as i32, val as i32);
        } else {
            let p = rows[k][n];
            let q = rows[k + 1][n];
            if p <= i && i < q {
                i -= p;
                y[yi] = 0;
                yi += 1;
            } else {
                let s = -((i >= q) as i32);
                i -= q & s as u32;
                let k0 = k;
                let mut p;
                loop {
                    k -= 1;
                    p = rows[k][n];
                    if p <= i {
                        break;
                    }
                }
                i -= p;
                let val = ((k0 as i32 - k as i32 + s) ^ s) as i16;
                y[yi] = val as i32;
                yi += 1;
                yy = mac16(yy, val as i32, val as i32);
            }
        }
        n -= 1;
    }
    // n == 2
    let p = 2 * k as u32 + 1;
    let s = -((i >= p) as i32);
    i -= p & s as u32;
    let k0 = k;
    k = ((i + 1) >> 1) as usize;
    if k != 0 {
        i -= 2 * k as u32 - 1;
    }
    let val = ((k0 as i32 - k as i32 + s) ^ s) as i16;
    y[yi] = val as i32;
    yy = mac16(yy, val as i32, val as i32);
    let s = -(i as i32);
    let val = ((k as i32 + s) ^ s) as i16;
    y[yi + 1] = val as i32;
    mac16(yy, val as i32, val as i32)
}

pub fn decode_pulses(y: &mut [i32], n: usize, k: usize, dec: &mut Ec) -> i32 {
    let v = pvq_v(n, k);
    let i = dec.dec_uint(v);
    cwrsi(n, k, i, y)
}

fn exp_rotation1(x: &mut [i16], len: usize, stride: usize, c: i32, s: i32) {
    let ms = -s;
    for i in 0..len - stride {
        let x1 = x[i] as i32;
        let x2 = x[i + stride] as i32;
        x[i + stride] = pshr32(mac16(mul16(c, x2), s, x1), 15) as i16;
        x[i] = pshr32(mac16(mul16(c, x1), ms, x2), 15) as i16;
    }
    if len > 2 * stride {
        for i in (0..=len - 2 * stride - 1).rev() {
            let x1 = x[i] as i32;
            let x2 = x[i + stride] as i32;
            x[i + stride] = pshr32(mac16(mul16(c, x2), s, x1), 15) as i16;
            x[i] = pshr32(mac16(mul16(c, x1), ms, x2), 15) as i16;
        }
    }
}

/// `exp_rotation`: spreading rotation (`dir` 1 before quantisation, -1 after).
pub fn exp_rotation(x: &mut [i16], len: usize, dir: i32, stride: usize, k: usize, spread: i32) {
    const SPREAD_FACTOR: [i32; 3] = [15, 10, 5];
    if 2 * k >= len || spread == SPREAD_NONE {
        return;
    }
    let factor = SPREAD_FACTOR[(spread - 1) as usize];
    let gain = div(mul16(32767, len as i32), len as i32 + factor * k as i32) as i16 as i32;
    let theta = (mul16_q15(gain, gain) as i16 as i32) >> 1;
    let c = cos_norm(theta);
    let s = cos_norm(32767 - theta);
    let mut stride2 = 0usize;
    if len >= 8 * stride {
        stride2 = 1;
        while (stride2 * stride2 + stride2) * stride + (stride >> 2) < len {
            stride2 += 1;
        }
    }
    let len = len / stride;
    for i in 0..stride {
        let xs = &mut x[i * len..(i + 1) * len];
        if dir < 0 {
            if stride2 != 0 {
                exp_rotation1(xs, len, stride2, s, c);
            }
            exp_rotation1(xs, len, 1, c, s);
        } else {
            exp_rotation1(xs, len, 1, c, -s);
            if stride2 != 0 {
                exp_rotation1(xs, len, stride2, s, -c);
            }
        }
    }
}

fn normalise_residual(iy: &[i32], x: &mut [i16], n: usize, ryy: i32, gain: i32) {
    let k = ilog2(ryy) >> 1;
    let t = vshr32(ryy, 2 * (k - 7));
    let g = mul16_p15(rsqrt_norm(t), gain);
    for i in 0..n {
        x[i] = pshr32(mul16(g, iy[i]), k + 1) as i16;
    }
}

fn extract_collapse_mask(iy: &[i32], n: usize, b: usize) -> u32 {
    if b <= 1 {
        return 1;
    }
    let n0 = n / b;
    let mut mask = 0u32;
    for i in 0..b {
        let tmp = iy[i * n0..(i + 1) * n0].iter().fold(0i32, |a, &v| a | v);
        mask |= ((tmp != 0) as u32) << i;
    }
    mask
}

/// `op_pvq_search`: best `k`-pulse codeword for `x`; `x` is made non-negative in place.
fn pvq_search(x: &mut [i16], iy: &mut [i32], k: i32, n: usize) -> i32 {
    let mut y = vec![0i16; n];
    let mut signx = vec![0i32; n];
    for j in 0..n {
        signx[j] = (x[j] < 0) as i32;
        x[j] = x[j].wrapping_abs();
        iy[j] = 0;
        y[j] = 0;
    }
    let mut xy = 0i32;
    let mut yy: i16 = 0;
    let mut pulses_left = k;
    if k > (n as i32 >> 1) {
        let mut sum = 0i32;
        for &v in x.iter() {
            sum += v as i32;
        }
        if sum <= k {
            x[0] = 16384;
            for v in x[1..].iter_mut() {
                *v = 0;
            }
            sum = 16384;
        }
        let r = mul16_32_q16(k, rcp(sum)) as i16 as i32;
        for j in 0..n {
            iy[j] = mul16_q15(x[j] as i32, r);
            y[j] = iy[j] as i16;
            yy = mac16(yy as i32, y[j] as i32, y[j] as i32) as i16;
            xy = mac16(xy, x[j] as i32, y[j] as i32);
            y[j] = y[j].wrapping_mul(2);
            pulses_left -= iy[j];
        }
    }
    if pulses_left > n as i32 + 3 {
        let tmp = pulses_left;
        yy = mac16(yy as i32, tmp, tmp) as i16;
        yy = mac16(yy as i32, tmp, y[0] as i32) as i16;
        iy[0] += pulses_left;
        pulses_left = 0;
    }
    for i in 0..pulses_left {
        let rshift = 1 + ilog2(k - pulses_left + i + 1);
        let mut best_id = 0usize;
        yy = add16(yy as i32, 1) as i16;
        let mut rxy = ((xy.wrapping_add(x[0] as i32)) >> rshift) as i16 as i32;
        let mut ryy = add16(yy as i32, y[0] as i32);
        rxy = mul16_q15(rxy, rxy) as i16 as i32;
        let mut best_den = ryy;
        let mut best_num = rxy;
        for j in 1..n {
            rxy = ((xy.wrapping_add(x[j] as i32)) >> rshift) as i16 as i32;
            ryy = add16(yy as i32, y[j] as i32);
            rxy = mul16_q15(rxy, rxy) as i16 as i32;
            if mul16(best_den, rxy) > mul16(ryy, best_num) {
                best_den = ryy;
                best_num = rxy;
                best_id = j;
            }
        }
        xy = xy.wrapping_add(x[best_id] as i32);
        yy = add16(yy as i32, y[best_id] as i32) as i16;
        y[best_id] = y[best_id].wrapping_add(2);
        iy[best_id] += 1;
    }
    for j in 0..n {
        iy[j] = (iy[j] ^ -signx[j]) + signx[j];
    }
    yy as i32
}

/// `alg_quant`: quantises `x` with `k` pulses; returns the collapse mask.
#[allow(clippy::too_many_arguments)]
pub fn alg_quant(
    x: &mut [i16],
    n: usize,
    k: usize,
    spread: i32,
    b: usize,
    enc: &mut Ec,
    gain: i32,
    resynth: bool,
) -> u32 {
    let mut iy = vec![0i32; n + 3];
    exp_rotation(x, n, 1, b, k, spread);
    let yy = pvq_search(x, &mut iy, k as i32, n);
    encode_pulses(&iy, n, k, enc);
    if resynth {
        normalise_residual(&iy, x, n, yy, gain);
        exp_rotation(x, n, -1, b, k, spread);
    }
    extract_collapse_mask(&iy, n, b)
}

/// `alg_unquant`: decodes `k` pulses into `x`; returns the collapse mask.
pub fn alg_unquant(
    x: &mut [i16],
    n: usize,
    k: usize,
    spread: i32,
    b: usize,
    dec: &mut Ec,
    gain: i32,
) -> u32 {
    let mut iy = vec![0i32; n];
    let ryy = decode_pulses(&mut iy, n, k, dec);
    normalise_residual(&iy, x, n, ryy, gain);
    exp_rotation(x, n, -1, b, k, spread);
    extract_collapse_mask(&iy, n, b)
}

/// `renormalise_vector`.
pub fn renormalise_vector(x: &mut [i16], n: usize, gain: i32) {
    let e = 1i32.wrapping_add(inner_prod(&x[..n], &x[..n]));
    let k = ilog2(e) >> 1;
    let t = vshr32(e, 2 * (k - 7));
    let g = mul16_p15(rsqrt_norm(t), gain);
    for v in x[..n].iter_mut() {
        *v = pshr32(mul16(g, *v as i32), k + 1) as i16;
    }
}

/// `stereo_itheta`.
pub fn stereo_itheta(x: &[i16], y: &[i16], stereo: bool, n: usize) -> i32 {
    let mut emid = 1i32;
    let mut eside = 1i32;
    if stereo {
        for i in 0..n {
            let m = add16((x[i] >> 1) as i32, (y[i] >> 1) as i32);
            let s = ((x[i] >> 1) as i32 - (y[i] >> 1) as i32) as i16 as i32;
            emid = mac16(emid, m, m);
            eside = mac16(eside, s, s);
        }
    } else {
        emid = emid.wrapping_add(inner_prod(&x[..n], &x[..n]));
        eside = eside.wrapping_add(inner_prod(&y[..n], &y[..n]));
    }
    let mid = sqrt(emid);
    let side = sqrt(eside);
    mul16_q15(20861, atan2p(side, mid))
}
