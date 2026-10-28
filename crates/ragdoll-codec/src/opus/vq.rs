//! Pyramid vector quantisation: codeword enumeration (`cwrs.c`) and the PVQ search, spreading
//! rotation and renormalisation (`vq.c`).

use std::sync::OnceLock;

use super::bands::MAX_BAND;
use super::math::{
    add16, atan2p, cos_norm, div, ilog2, inner_prod, mac16, mul16, mul16_32_q16, mul16_p15,
    mul16_q15, pshr32, rcp, rsqrt_norm, sqrt, vshr32,
};
use super::range::Ec;

pub const SPREAD_NONE: i32 = 0;
pub const SPREAD_LIGHT: i32 = 1;
pub const SPREAD_NORMAL: i32 = 2;
pub const SPREAD_AGGRESSIVE: i32 = 3;

/// Columns of the `U(n, k)` table; rows are indexed by `min(n, k) <= 14`.
const U_COLS: usize = 210;
/// Zero rows and columns ahead of the table: U(n, j) reads as 0 for `-U_PAD <= j < 0`, so the
/// pulse-count windows of [`cwrsi`] need no tests at the low end.
const U_PAD: usize = 8;
/// Row length of the padded table.
const U_STRIDE: usize = U_PAD + U_COLS;

/// `CELT_PVQ_U_ROW`: U(n, k), the number of n-dimensional codewords with k pulses and a positive
/// first element, at `[U_PAD + min(n, k)][U_PAD + max(n, k)]`. Built from the recurrence
/// U(n, k) = U(n-1, k) + U(n, k-1) + U(n-1, k-1); entries libopus uses fit 32 bits.
fn u_table() -> &'static [[u32; U_STRIDE]; U_PAD + 15] {
    static T: OnceLock<Box<[[u32; U_STRIDE]; U_PAD + 15]>> = OnceLock::new();
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
        let mut t = Box::new([[0u32; U_STRIDE]; U_PAD + 15]);
        for (row, src) in t[U_PAD..].iter_mut().zip(&u) {
            for (d, s) in row[U_PAD..].iter_mut().zip(src) {
                *d = *s as u32;
            }
        }
        t
    })
}

#[inline]
fn pvq_u(n: usize, k: usize) -> u32 {
    u_table()[U_PAD + n.min(k)][U_PAD + n.max(k)]
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

/// The pulse count left after a dimension of `cwrsi`: the largest `j <= k` with
/// `U(n, j) <= i`, where U(n, j) is at `u + j * S`. The `W` thresholds below `k` are compared
/// without branches (U grows with j); only a count past them takes the loop.
///
/// # Safety
///
/// `u + j * S` must be readable for `-W < j <= k`, and must hold U(n, j) (0 for `j < 0`), with
/// U(n, 0) = 0, for `j <= k`.
#[inline(always)]
unsafe fn pulse_count<const W: usize, const S: usize>(u: *const u32, k: usize, i: u32) -> usize {
    // Safety: the caller's contract; `kk` stays within `0..=k` because U(n, 0) = 0 <= i.
    unsafe {
        let at = |j: usize| *u.add(j * S);
        let top = u.add(k * S);
        let mut d = 0usize;
        for w in 0..W {
            d += (*top.sub(w * S) > i) as usize;
        }
        if d < W {
            return k - d;
        }
        // All W thresholds exceed `i`, so `k - W + 1 >= 1` (U(n, j) <= 0 below that).
        let mut kk = k - W;
        while at(kk) > i {
            kk -= 1;
        }
        kk
    }
}

/// `cwrsi`: codeword of index `i`; returns its squared norm.
///
/// Each dimension takes the sign, then the pulse count as the largest `k' <= k` with
/// U(n, k') <= i, which [`pulse_count`] finds from a window of thresholds rather than the
/// one-step-at-a-time scan of libopus: the scan's exits are data-dependent branches the CPU
/// mostly mispredicts. The result is the same codeword (`cwrsi_reference` is the libopus scan).
fn cwrsi(mut n: usize, mut k: usize, mut i: u32, y: &mut [i16]) -> i32 {
    // Every table read is of U(n', j) with n' <= n and j <= k + 1, so these bound them all.
    assert!(n >= 2 && n.min(k + 1) < 15 && n.max(k + 1) < U_COLS && y.len() >= n);
    let t = u_table().as_flattened().as_ptr();
    let mut yy = 0i32;
    let mut yi = 0usize;
    while n > 2 {
        let s;
        let kn;
        let p;
        // Safety: by the assert, U(n, j) for j <= k + 1 lies in the table: in row n when
        // k >= n (then n < 15), in column n otherwise (then j <= k + 1 < 15); the U_PAD zero
        // rows and columns ahead cover the windows' j > -U_PAD.
        unsafe {
            if k >= n {
                let u = t.add((U_PAD + n) * U_STRIDE + U_PAD);
                let q = *u.add(k + 1);
                s = -((i >= q) as i32);
                i -= q & s as u32;
                kn = pulse_count::<8, 1>(u, k, i);
                p = *u.add(kn);
            } else {
                let u = t.add(U_PAD * U_STRIDE + U_PAD + n);
                let q = *u.add((k + 1) * U_STRIDE);
                s = -((i >= q) as i32);
                i -= q & s as u32;
                kn = pulse_count::<3, U_STRIDE>(u, k, i);
                p = *u.add(kn * U_STRIDE);
            }
        }
        i -= p;
        let val = ((k as i32 - kn as i32 + s) ^ s) as i16;
        y[yi] = val;
        yi += 1;
        yy = mac16(yy, val as i32, val as i32);
        k = kn;
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
    y[yi] = val;
    yy = mac16(yy, val as i32, val as i32);
    let s = -(i as i32);
    let val = ((k as i32 + s) ^ s) as i16;
    y[yi + 1] = val;
    mac16(yy, val as i32, val as i32)
}

/// `cwrsi` as libopus writes it, the reference of [`cwrsi`].
#[cfg(test)]
fn cwrsi_reference(mut n: usize, mut k: usize, mut i: u32, y: &mut [i16]) -> i32 {
    let rows = |a: usize| &u_table()[U_PAD + a][U_PAD..];
    let mut yy = 0i32;
    let mut yi = 0usize;
    while n > 2 {
        if k >= n {
            let row = rows(n);
            let mut p = row[k + 1];
            let s = -((i >= p) as i32);
            i -= p & s as u32;
            let k0 = k;
            let q = row[n];
            if q > i {
                k = n;
                loop {
                    k -= 1;
                    p = rows(k)[n];
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
            y[yi] = val;
            yi += 1;
            yy = mac16(yy, val as i32, val as i32);
        } else {
            let p = rows(k)[n];
            let q = rows(k + 1)[n];
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
                    p = rows(k)[n];
                    if p <= i {
                        break;
                    }
                }
                i -= p;
                let val = ((k0 as i32 - k as i32 + s) ^ s) as i16;
                y[yi] = val;
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
    y[yi] = val;
    yy = mac16(yy, val as i32, val as i32);
    let s = -(i as i32);
    let val = ((k as i32 + s) ^ s) as i16;
    y[yi + 1] = val;
    mac16(yy, val as i32, val as i32)
}

pub fn decode_pulses(y: &mut [i16], n: usize, k: usize, dec: &mut Ec) -> i32 {
    let v = pvq_v(n, k);
    let i = dec.dec_uint(v);
    cwrsi(n, k, i, y)
}

fn exp_rotation1(x: &mut [i16], len: usize, stride: usize, c: i32, s: i32) {
    let x = &mut x[..len];
    match stride {
        1 if len >= 2 => exp_rotation1_serial(x, c, s),
        1 => {}
        2 => rotate_chains::<2>(x, c, s),
        3 => rotate_chains::<3>(x, c, s),
        4 => rotate_chains::<4>(x, c, s),
        5 => rotate_chains::<5>(x, c, s),
        6 => rotate_chains::<6>(x, c, s),
        7 => rotate_chains::<7>(x, c, s),
        8 => rotate_chains::<8>(x, c, s),
        _ => rotate_strided(x, stride, c, s),
    }
}

/// [`exp_rotation1`] with a stride above 1, as libopus writes it.
fn rotate_strided(x: &mut [i16], stride: usize, c: i32, s: i32) {
    let len = x.len();
    let ms = -s;
    // The rotated pair (x1, x2) -> (x1', x2').
    let rot = |x1: i32, x2: i32| {
        (
            pshr32(mac16(mul16(c, x1), ms, x2), 15) as i16,
            pshr32(mac16(mul16(c, x2), s, x1), 15) as i16,
        )
    };
    for i in 0..len.saturating_sub(stride) {
        let (a, b) = rot(x[i] as i32, x[i + stride] as i32);
        x[i + stride] = b;
        x[i] = a;
    }
    if len > 2 * stride {
        for i in (0..=len - 2 * stride - 1).rev() {
            let (a, b) = rot(x[i] as i32, x[i + stride] as i32);
            x[i + stride] = b;
            x[i] = a;
        }
    }
}

/// [`rotate_strided`] with stride `S`. Each pass is `S` independent chains (the coefficients
/// `r, r + S, r + 2S, ...`), every step rotating the value the chain's previous step produced.
/// Here each chain carries that value in a register rather than through memory, where every
/// step waited on the previous step's store, and the `S` chains advance in lockstep.
fn rotate_chains<const S: usize>(x: &mut [i16], c: i32, s: i32) {
    let len = x.len();
    if len <= S {
        return;
    }
    let (c, s, ms) = (c as i16 as i32, s as i16 as i32, (-s) as i16 as i32);
    // The rotated pair (x1, x2) -> (x1', x2').
    let a = |x1: i32, x2: i32| ((c * x1).wrapping_add(ms * x2).wrapping_add(16384) >> 15) as i16;
    let b = |x1: i32, x2: i32| ((c * x2).wrapping_add(s * x1).wrapping_add(16384) >> 15) as i16;
    // Forward, steps i < len - S: (x[i], x[i + S]) with x[i] carried, x[i + S] the next one.
    let steps = len - S;
    let (full, rem) = (steps / S, steps % S);
    let mut v: [i32; S] = std::array::from_fn(|r| x[r] as i32);
    for t in 0..full {
        let (cur, next) = x[t * S..t * S + 2 * S].split_at_mut(S);
        for r in 0..S {
            let x2 = next[r] as i32;
            cur[r] = a(v[r], x2);
            v[r] = b(v[r], x2) as i32;
        }
    }
    // Chains r < rem take one more step; then each lands its value.
    let o = full * S;
    for r in 0..S {
        if r < rem {
            let x2 = x[o + S + r] as i32;
            x[o + r] = a(v[r], x2);
            x[o + S + r] = b(v[r], x2);
        } else {
            x[o + r] = v[r] as i16;
        }
    }
    if len <= 2 * S {
        return;
    }
    // Backward, steps i from len - 2S - 1 down: (x[i], x[i + S]) with x[i + S] carried.
    let steps = len - 2 * S;
    let (full, rem) = (steps / S, steps % S);
    let o = full * S;
    // Chains r < rem start one step higher, at i = o + r.
    for r in 0..S {
        v[r] = x[o + r] as i32;
    }
    for r in 0..rem {
        let x1 = x[o + r] as i32;
        let x2 = x[o + S + r] as i32;
        x[o + S + r] = b(x1, x2);
        v[r] = a(x1, x2) as i32;
    }
    for t in (0..full).rev() {
        let (cur, next) = x[t * S..t * S + 2 * S].split_at_mut(S);
        for r in 0..S {
            let x1 = cur[r] as i32;
            next[r] = b(x1, v[r]);
            v[r] = a(x1, v[r]) as i32;
        }
    }
    for r in 0..S {
        x[r] = v[r] as i16;
    }
}

/// [`exp_rotation1`] with stride 1, where every step rotates the previous step's output: a
/// serial chain, so it is kept short. The carried value stays in a register and skips the
/// 16-bit truncation while it is in range (then the truncation changes nothing); the first
/// value out of range, which a normalised band never produces, leaves for the exact loop.
/// The `black_box` keeps the rounding constant and the other product off the chain.
fn exp_rotation1_serial(x: &mut [i16], c: i32, s: i32) {
    use std::hint::black_box;
    let len = x.len();
    let ms = -s;
    let in_range = |v: i32| (v + 32768) as u32 <= 65535;
    // Forward: (x[i], x[i + 1]) -> (a, b), with b carried as the next x[i].
    let mut x1 = x[0] as i32;
    let mut i = 0;
    while i < len - 1 {
        let x2 = x[i + 1] as i32;
        let u = black_box(c * x2 + 16384);
        x[i] = ((c * x1 + ms * x2 + 16384) >> 15) as i16;
        x1 = (s * x1 + u) >> 15;
        i += 1;
        if !in_range(x1) {
            break;
        }
    }
    x1 = x1 as i16 as i32;
    while i < len - 1 {
        let x2 = x[i + 1] as i32;
        x[i] = ((c * x1 + ms * x2 + 16384) >> 15) as i16;
        x1 = ((s * x1 + c * x2 + 16384) >> 15) as i16 as i32;
        i += 1;
    }
    x[len - 1] = x1 as i16;
    if len <= 2 {
        return;
    }
    // Backward: (x[i], x[i + 1]) -> (a, b), with a carried as the next x[i + 1].
    let mut x2 = x[len - 2] as i32;
    let mut i = len - 2;
    while i > 0 {
        i -= 1;
        let x1 = x[i] as i32;
        let u = black_box(c * x1 + 16384);
        x[i + 1] = ((c * x2 + s * x1 + 16384) >> 15) as i16;
        x2 = (ms * x2 + u) >> 15;
        if !in_range(x2) {
            break;
        }
    }
    x2 = x2 as i16 as i32;
    while i > 0 {
        i -= 1;
        let x1 = x[i] as i32;
        x[i + 1] = ((c * x2 + s * x1 + 16384) >> 15) as i16;
        x2 = ((ms * x2 + c * x1 + 16384) >> 15) as i16 as i32;
    }
    x[0] = x2 as i16;
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
    let (g, sh) = pulses_gain(ryy, gain);
    for (o, &v) in x[..n].iter_mut().zip(&iy[..n]) {
        *o = pshr32(mul16(g, v), sh) as i16;
    }
}

fn extract_collapse_mask<T: Copy + Into<i32>>(iy: &[T], n: usize, b: usize) -> u32 {
    if b <= 1 {
        return 1;
    }
    let n0 = n / b;
    let mut mask = 0u32;
    for i in 0..b {
        let tmp = iy[i * n0..(i + 1) * n0]
            .iter()
            .fold(0i32, |a, &v| a | v.into());
        mask |= ((tmp != 0) as u32) << i;
    }
    mask
}

/// `op_pvq_search`: best `k`-pulse codeword for `x`; `x` is made non-negative in place.
fn pvq_search(x: &mut [i16], iy: &mut [i32], k: i32, n: usize) -> i32 {
    let mut y = [0i16; MAX_BAND];
    let mut signx = [0i32; MAX_BAND];
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
    let mut iy = [0i32; MAX_BAND + 3];
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
    // The pulses fit 16 bits, so they are decoded straight into `x` and scaled in place.
    let x = &mut x[..n];
    let ryy = decode_pulses(x, n, k, dec);
    let cm = extract_collapse_mask(x, n, b);
    let (g, sh) = pulses_gain(ryy, gain);
    scale_shr(x, g, sh);
    exp_rotation(x, n, -1, b, k, spread);
    cm
}

/// The reciprocal square root `normalise_residual` takes of the squared norm `ryy` of a
/// pulse vector.
const fn pulses_rsqrt(ryy: i32) -> i32 {
    let k = ilog2(ryy) >> 1;
    rsqrt_norm(vshr32(ryy, 2 * (k - 7)))
}

/// [`pulses_rsqrt`] for the small squared norms nearly every pulse vector has: its polynomial
/// is a serial chain of seven multiplies between a vector's decoding and its scaling.
static RSQRT_PULSES: [i16; 4096] = {
    let mut t = [0i16; 4096];
    let mut r = 1;
    while r < t.len() {
        t[r] = pulses_rsqrt(r as i32) as i16;
        r += 1;
    }
    t
};

/// The gain and shift of `normalise_residual` for pulses of squared norm `ryy`.
fn pulses_gain(ryy: i32, gain: i32) -> (i32, i32) {
    let r = match RSQRT_PULSES.get(ryy as usize) {
        Some(&r) => r as i32,
        None => pulses_rsqrt(ryy),
    };
    (mul16_p15(r, gain), (ilog2(ryy) >> 1) + 1)
}

/// `renormalise_vector`.
pub fn renormalise_vector(x: &mut [i16], n: usize, gain: i32) {
    let e = 1i32.wrapping_add(inner_prod(&x[..n], &x[..n]));
    let k = ilog2(e) >> 1;
    let t = vshr32(e, 2 * (k - 7));
    let g = mul16_p15(rsqrt_norm(t), gain);
    scale_shr(&mut x[..n], g, k + 1);
}

/// `x[i] = PSHR32(MULT16_16(g, x[i]), sh)` truncated to 16 bits: the scaling of
/// `normalise_residual` and `renormalise_vector`.
pub fn scale_shr(x: &mut [i16], g: i32, sh: i32) {
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        if unsafe { super::simd::scale_shr_avx2(x, g, sh) } {
            return;
        }
    }
    scale_shr_scalar(x, g, sh);
}

/// Scalar reference of [`scale_shr`].
pub fn scale_shr_scalar(x: &mut [i16], g: i32, sh: i32) {
    for v in x.iter_mut() {
        *v = pshr32(mul16(g, *v as i32), sh) as i16;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotate_chains_matches_strided() {
        let mut seed = 9u32;
        let mut rnd = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed
        };
        for trial in 0..20000 {
            let len = 1 + (rnd() % 90) as usize;
            let stride = 2 + trial % 7;
            let (c, s) = match trial % 5 {
                0 => (32767, 32767),
                1 => (-32767, 32767),
                _ => (rnd() as i16 as i32 / 2 + 16384, rnd() as i16 as i32),
            };
            let x: Vec<i16> = (0..len)
                .map(|k| match (trial + k) % 11 {
                    0 => i16::MIN,
                    1 => i16::MAX,
                    _ => rnd() as i16,
                })
                .collect();
            let (mut a, mut b) = (x.clone(), x.clone());
            exp_rotation1(&mut a, len, stride, c, s);
            rotate_strided(&mut b, stride, c, s);
            assert_eq!(a, b, "len {len} stride {stride} c {c} s {s}");
        }
    }

    /// Every codebook whose size fits 32 bits (those libopus can code): the windowed decode
    /// matches the libopus scan, and `icwrs` inverts it.
    #[test]
    fn cwrsi_matches_reference() {
        let mut u = vec![[0u128; U_COLS + 1]; 200];
        u[0][0] = 1;
        for n in 1..200 {
            for k in 1..=U_COLS {
                u[n][k] = (u[n - 1][k] + u[n][k - 1] + u[n - 1][k - 1]).min(1 << 64);
            }
        }
        let (mut ya, mut yb) = ([0i16; 200], [0i16; 200]);
        let mut seed = 1u32;
        let mut pairs = 0;
        for n in 2..=176usize {
            for k in 1..U_COLS - 1 {
                let v = u[n][k] + u[n][k + 1];
                if n.min(k + 1) >= 15 || v >= 1 << 32 {
                    continue;
                }
                pairs += 1;
                assert_eq!(pvq_v(n, k) as u128, v);
                let v = v as u32;
                let up = u[n][k + 1] as u32;
                let mut check = |i: u32| {
                    let a = cwrsi(n, k, i, &mut ya);
                    let b = cwrsi_reference(n, k, i, &mut yb);
                    assert_eq!((a, &ya[..n]), (b, &yb[..n]), "n {n} k {k} i {i}");
                    let y: Vec<i32> = ya[..n].iter().map(|&x| x as i32).collect();
                    assert_eq!(icwrs(n, &y), i, "n {n} k {k}");
                };
                if v <= 2048 {
                    (0..v).for_each(&mut check);
                } else {
                    for i in [0, 1, up - 1, up, up + 1, v - 2, v - 1] {
                        check(i);
                    }
                    for _ in 0..48 {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        check(((seed as u64 * v as u64) >> 32) as u32);
                    }
                }
            }
        }
        assert_eq!(pairs, 1979);
    }
}
