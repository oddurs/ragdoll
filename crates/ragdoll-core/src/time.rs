use crate::error::{RagdollError, Result};
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Sentinel timestamp value indicating "no timestamp" (`AV_NOPTS_VALUE`).
pub const NOPTS_VALUE: i64 = i64::MIN;

/// Exact signed 64-bit rational number `num / den` (`den > 0`) for drift-free
/// media timebases, framerates, and sample aspect ratios.
///
/// Uses continued-fraction convergent reduction (`reduce_with_max`, modeled on
/// FFmpeg's `av_reduce`) and 128-bit (`i128`) intermediate arithmetic for
/// overflow-free rescaling and comparisons.
#[derive(Clone, Copy, Debug, Eq)]
pub struct Rational64 {
    pub num: i64,
    pub den: i64,
}

/// Rounding modes for 128-bit rational timestamp rescaling (`a * b / c`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Rounding {
    /// Round toward zero (`AV_ROUND_ZERO`).
    Zero,
    /// Round away from zero (`AV_ROUND_INF`).
    Inf,
    /// Round toward $-\infty$ (`AV_ROUND_DOWN`).
    Down,
    /// Round toward $+\infty$ (`AV_ROUND_UP`).
    Up,
    /// Round to nearest; half-way cases round away from zero (`AV_ROUND_NEAR_INF`).
    #[default]
    NearInf,
}

impl Rational64 {
    pub const ZERO: Self = Self { num: 0, den: 1 };
    pub const ONE: Self = Self { num: 1, den: 1 };

    /// Nanosecond timebase ($1 / 10^9$).
    pub const TIMEBASE_NANOS: Self = Self {
        num: 1,
        den: 1_000_000_000,
    };
    /// Microsecond timebase ($1 / 10^6$).
    pub const TIMEBASE_MICROS: Self = Self {
        num: 1,
        den: 1_000_000,
    };
    /// Millisecond timebase ($1 / 10^3$).
    pub const TIMEBASE_MILLIS: Self = Self { num: 1, den: 1_000 };
    /// MPEG-TS / H.264 90 kHz presentation clock ($1 / 90\,000$).
    pub const TIMEBASE_MPEG_90K: Self = Self {
        num: 1,
        den: 90_000,
    };

    /// NTSC Film framerate ($24\,000 / 1001 \approx 23.976$ fps).
    pub const FPS_23_976: Self = Self {
        num: 24_000,
        den: 1001,
    };
    /// Cinema framerate ($24 / 1$ fps).
    pub const FPS_24: Self = Self { num: 24, den: 1 };
    /// PAL broadcast framerate ($25 / 1$ fps).
    pub const FPS_25: Self = Self { num: 25, den: 1 };
    /// NTSC broadcast framerate ($30\,000 / 1001 \approx 29.97$ fps).
    pub const FPS_29_97: Self = Self {
        num: 30_000,
        den: 1001,
    };
    /// 30 fps ($30 / 1$).
    pub const FPS_30: Self = Self { num: 30, den: 1 };
    /// NTSC high framerate ($60\,000 / 1001 \approx 59.94$ fps).
    pub const FPS_59_94: Self = Self {
        num: 60_000,
        den: 1001,
    };
    /// 60 fps ($60 / 1$).
    pub const FPS_60: Self = Self { num: 60, den: 1 };
    /// 120 fps ($120 / 1$).
    pub const FPS_120: Self = Self { num: 120, den: 1 };

    /// Standard broadcast & cinema framerate table for nearest-framerate snapping.
    pub const STANDARD_FRAMERATES: &'static [Self] = &[
        Self::FPS_23_976,
        Self::FPS_24,
        Self::FPS_25,
        Self::FPS_29_97,
        Self::FPS_30,
        Self {
            num: 48_000,
            den: 1001,
        },
        Self { num: 48, den: 1 },
        Self::FPS_59_94,
        Self::FPS_60,
        Self::FPS_120,
    ];

    /// Creates a reduced `Rational64` with positive denominator.
    /// Panics if `den == 0`.
    #[inline]
    pub const fn new(num: i64, den: i64) -> Self {
        assert!(den != 0, "Rational64 denominator must not be zero");
        let (r, _) = Self::reduce_with_max(num as i128, den as i128, i64::MAX as u64);
        r
    }

    /// Fallible constructor returning `Err` when `den == 0`.
    #[inline]
    pub const fn checked_new(num: i64, den: i64) -> Result<Self> {
        if den == 0 {
            return Err(RagdollError::InvalidTimestamp(
                "rational denominator is zero",
            ));
        }
        let (r, _) = Self::reduce_with_max(num as i128, den as i128, i64::MAX as u64);
        Ok(r)
    }

    /// Creates a `Rational64` without GCD reduction (normalizing denominator sign).
    #[inline]
    pub const fn new_raw(num: i64, den: i64) -> Self {
        assert!(den != 0, "Rational64 denominator must not be zero");
        if den < 0 {
            Self {
                num: num.saturating_neg(),
                den: den.saturating_neg(),
            }
        } else {
            Self { num, den }
        }
    }

    #[inline]
    pub const fn from_int(n: i64) -> Self {
        Self { num: n, den: 1 }
    }

    /// Reduces `num / den` to the best rational approximation `p / q` with `|p| <= max` and `1 <= q <= max`
    /// using continued-fraction convergents and semiconvergent clamping (matching FFmpeg `av_reduce`).
    ///
    /// Returns `(reduced_rational, is_exact)`.
    pub const fn reduce_with_max(raw_num: i128, raw_den: i128, max: u64) -> (Self, bool) {
        if raw_den == 0 {
            return (Self { num: 0, den: 1 }, false);
        }
        let sign = (raw_num < 0) ^ (raw_den < 0);
        let mut num = raw_num.unsigned_abs();
        let mut den = raw_den.unsigned_abs();

        let gcd = gcd_u128(num, den);
        if let (Some(n), Some(d)) = (num.checked_div(gcd), den.checked_div(gcd)) {
            num = n;
            den = d;
        }
        if num == 0 {
            return (Self { num: 0, den: 1 }, true);
        }

        let max_u128 = if max > i64::MAX as u64 {
            i64::MAX as u128
        } else {
            max as u128
        };

        if num <= max_u128 && den <= max_u128 {
            let signed_num = if sign { -(num as i64) } else { num as i64 };
            return (
                Self {
                    num: signed_num,
                    den: den as i64,
                },
                true,
            );
        }

        // Continued fraction convergents: a0 = 0/1, a1 = 1/0
        let mut a0_num: u128 = 0;
        let mut a0_den: u128 = 1;
        let mut a1_num: u128 = 1;
        let mut a1_den: u128 = 0;

        while den != 0 {
            let mut x = num / den;
            let next_den = num % den;
            let a2_num = x.saturating_mul(a1_num).saturating_add(a0_num);
            let a2_den = x.saturating_mul(a1_den).saturating_add(a0_den);

            if a2_num > max_u128 || a2_den > max_u128 {
                if let Some(x_num) = (max_u128 - a0_num).checked_div(a1_num) {
                    x = x_num;
                }
                if let Some(x_den) = (max_u128 - a0_den).checked_div(a1_den)
                    && x_den < x
                {
                    x = x_den;
                }
                // Semiconvergent optimality test: is (x * a1 + a0) closer to num/den than a1?
                if den.saturating_mul(2 * x * a1_den + a0_den) > num.saturating_mul(a1_den) {
                    a1_num = x * a1_num + a0_num;
                    a1_den = x * a1_den + a0_den;
                }
                break;
            }

            a0_num = a1_num;
            a0_den = a1_den;
            a1_num = a2_num;
            a1_den = a2_den;
            num = den;
            den = next_den;
        }

        if a1_den == 0 {
            a1_den = 1;
        }
        let signed_num = if sign {
            -(a1_num as i64)
        } else {
            a1_num as i64
        };
        (
            Self {
                num: signed_num,
                den: a1_den as i64,
            },
            den == 0,
        )
    }

    /// Converts a floating-point number `d` to its closest `Rational64` with denominator/numerator $\le \text{max}$
    /// using binary exponent scaling + continued-fraction reduction (modeled on FFmpeg `av_d2q`).
    ///
    /// Example: `Rational64::from_f64(29.97002997, 100_000)` produces `30000/1001`.
    pub fn from_f64(d: f64, max: u64) -> Self {
        if !d.is_finite() || d == 0.0 {
            return Self::ZERO;
        }
        let abs_d = d.abs();
        let exp = abs_d.log2().floor() as i32 + 1;
        let shift = (61 - exp.max(0)).clamp(0, 61);
        let den = 1i128 << shift;
        let num = (d * (den as f64)).round() as i128;
        let (r, _) = Self::reduce_with_max(num, den, max);
        r
    }

    /// Compares which of `q1` or `q2` is closer to `self` (`av_nearer_q`).
    /// Returns `Ordering::Greater` if `q1` is nearer to `self` than `q2`,
    /// `Ordering::Less` if `q2` is nearer, or `Ordering::Equal` if equidistant.
    pub fn nearer(self, q1: Self, q2: Self) -> Ordering {
        let a = (q1.num as i128) * (q2.den as i128) + (q2.num as i128) * (q1.den as i128);
        let b = 2 * (q1.den as i128) * (q2.den as i128);

        let ad = a * (self.den as i128);
        let x_up = div_round_i128(ad, b, Rounding::Up);
        let x_down = div_round_i128(ad, b, Rounding::Down);
        let n = self.num as i128;

        let dir = (i32::from(x_up > n) - i32::from(x_down < n))
            * match q2.cmp(&q1) {
                Ordering::Greater => 1,
                Ordering::Less => -1,
                Ordering::Equal => 0,
            };
        dir.cmp(&0)
    }

    /// Finds the index of the nearest rational in `candidates` (`av_find_nearest_q_idx`).
    pub fn find_nearest(self, candidates: &[Self]) -> Option<usize> {
        if candidates.is_empty() {
            return None;
        }
        let mut best = 0;
        for (i, &cand) in candidates.iter().enumerate().skip(1) {
            if self.nearer(cand, candidates[best]) == Ordering::Greater {
                best = i;
            }
        }
        Some(best)
    }

    /// Reciprocal `den / num`.
    #[inline]
    pub const fn inv(self) -> Self {
        Self::new(self.den, self.num)
    }

    #[inline]
    pub fn as_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    #[inline]
    pub const fn is_zero(self) -> bool {
        self.num == 0
    }

    #[inline]
    pub const fn is_positive(self) -> bool {
        self.num > 0
    }
}

/// Stein's binary GCD algorithm on 128-bit unsigned integers (`av_gcd` extended to `u128`).
#[inline]
pub const fn gcd_u128(mut u: u128, mut v: u128) -> u128 {
    if u == 0 {
        return v;
    }
    if v == 0 {
        return u;
    }
    let shift = (u | v).trailing_zeros();
    u >>= u.trailing_zeros();
    loop {
        v >>= v.trailing_zeros();
        if u > v {
            let tmp = u;
            u = v;
            v = tmp;
        }
        v -= u;
        if v == 0 {
            break;
        }
    }
    u << shift
}

#[inline]
const fn div_round_i128(num: i128, den: i128, rnd: Rounding) -> i128 {
    assert!(den > 0, "divisor must be positive");
    if num >= 0 {
        let r = match rnd {
            Rounding::Zero | Rounding::Down => 0,
            Rounding::Inf | Rounding::Up => den - 1,
            Rounding::NearInf => den / 2,
        };
        (num + r) / den
    } else {
        let abs_num = -num;
        let r = match rnd {
            Rounding::Zero | Rounding::Up => 0,
            Rounding::Inf | Rounding::Down => den - 1,
            Rounding::NearInf => den / 2,
        };
        -((abs_num + r) / den)
    }
}

impl Default for Rational64 {
    fn default() -> Self {
        Self::ONE
    }
}

impl PartialEq for Rational64 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        (self.num as i128) * (other.den as i128) == (other.num as i128) * (self.den as i128)
    }
}

impl PartialOrd for Rational64 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rational64 {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        let lhs = (self.num as i128) * (other.den as i128);
        let rhs = (other.num as i128) * (self.den as i128);
        lhs.cmp(&rhs)
    }
}

impl Add for Rational64 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let n = (self.num as i128) * (rhs.den as i128) + (rhs.num as i128) * (self.den as i128);
        let d = (self.den as i128) * (rhs.den as i128);
        Self::reduce_with_max(n, d, i64::MAX as u64).0
    }
}

impl Sub for Rational64 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let n = (self.num as i128) * (rhs.den as i128) - (rhs.num as i128) * (self.den as i128);
        let d = (self.den as i128) * (rhs.den as i128);
        Self::reduce_with_max(n, d, i64::MAX as u64).0
    }
}

impl Mul for Rational64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let n = (self.num as i128) * (rhs.num as i128);
        let d = (self.den as i128) * (rhs.den as i128);
        Self::reduce_with_max(n, d, i64::MAX as u64).0
    }
}

impl Div for Rational64 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        assert!(rhs.num != 0, "division by zero Rational64");
        let n = (self.num as i128) * (rhs.den as i128);
        let d = (self.den as i128) * (rhs.num as i128);
        Self::reduce_with_max(n, d, i64::MAX as u64).0
    }
}

impl Neg for Rational64 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            num: self.num.saturating_neg(),
            den: self.den,
        }
    }
}

impl fmt::Display for Rational64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.num, self.den)
    }
}

/// Computes `round(a * b / c)` using exact 128-bit intermediate arithmetic.
///
/// Preserves `NOPTS_VALUE` (`i64::MIN`) and `i64::MAX` (`AV_ROUND_PASS_MINMAX` semantics).
#[inline]
pub const fn rescale_rnd(a: i64, b: i64, c: i64, rnd: Rounding) -> i64 {
    assert!(c > 0, "rescale_rnd divisor c must be positive");
    if a == i64::MIN || a == i64::MAX {
        return a;
    }
    let num = (a as i128) * (b as i128);
    let den = c as i128;
    let res = div_round_i128(num, den, rnd);
    if res > i64::MAX as i128 {
        i64::MAX
    } else if res <= i64::MIN as i128 {
        i64::MIN + 1
    } else {
        res as i64
    }
}

/// Rescales timestamp `val` from timebase `bq` to timebase `cq` (`val * bq / cq`).
#[inline]
pub const fn rescale_q_rnd(val: i64, bq: Rational64, cq: Rational64, rnd: Rounding) -> i64 {
    if val == i64::MIN || val == i64::MAX {
        return val;
    }
    let b = (bq.num as i128) * (cq.den as i128);
    let c = (cq.num as i128) * (bq.den as i128);
    assert!(c > 0, "target timebase product must be positive");

    // Reduce (b, c) by their GCD when val * b might approach i128 limits
    let g = gcd_u128(b.unsigned_abs(), c as u128) as i128;
    let b_red = if g > 1 { b / g } else { b };
    let c_red = if g > 1 { c / g } else { c };

    let num = (val as i128) * b_red;
    let res = div_round_i128(num, c_red, rnd);
    if res > i64::MAX as i128 {
        i64::MAX
    } else if res <= i64::MIN as i128 {
        i64::MIN + 1
    } else {
        res as i64
    }
}

/// Rescales `val` from `bq` to `cq` using `Rounding::NearInf`.
#[inline]
pub const fn rescale_q(val: i64, bq: Rational64, cq: Rational64) -> i64 {
    rescale_q_rnd(val, bq, cq, Rounding::NearInf)
}

/// Exact comparison of two timestamps in potentially different timebases (`av_compare_ts`).
#[inline]
pub fn compare_ts(ts_a: i64, tb_a: Rational64, ts_b: i64, tb_b: Rational64) -> Ordering {
    let a = (tb_a.num as i128) * (tb_b.den as i128);
    let b = (tb_b.num as i128) * (tb_a.den as i128);
    if ts_a.unsigned_abs() <= i32::MAX as u64
        && ts_b.unsigned_abs() <= i32::MAX as u64
        && a.unsigned_abs() <= i32::MAX as u128
        && b.unsigned_abs() <= i32::MAX as u128
    {
        return ((ts_a as i128) * a).cmp(&((ts_b as i128) * b));
    }
    if div_round_i128((ts_a as i128) * a, b, Rounding::Down) < (ts_b as i128) {
        return Ordering::Less;
    }
    if div_round_i128((ts_b as i128) * b, a, Rounding::Down) < (ts_a as i128) {
        return Ordering::Greater;
    }
    Ordering::Equal
}

/// Stateful audio packet timestamp rescaler (`av_rescale_delta`).
///
/// Rescales `in_ts` from coarse input container timebase `in_tb` (e.g., 1/1000 ms in MKV)
/// to sample-accurate timebase `fs_tb` (e.g., 1/48000) and `out_tb`, eliminating 1-sample
/// rounding jitter across consecutive audio frames of length `duration`.
pub fn rescale_delta(
    in_tb: Rational64,
    in_ts: i64,
    fs_tb: Rational64,
    duration: i64,
    last: &mut i64,
    out_tb: Rational64,
) -> i64 {
    if in_ts == NOPTS_VALUE {
        return NOPTS_VALUE;
    }
    let in_step = (in_tb.num as i128) * (out_tb.den as i128);
    let out_step = (out_tb.num as i128) * (in_tb.den as i128);
    if *last == NOPTS_VALUE || duration <= 0 || in_step <= out_step {
        *last = rescale_q(in_ts, in_tb, fs_tb).saturating_add(duration);
        return rescale_q(in_ts, in_tb, out_tb);
    }

    let a = rescale_q_rnd(
        in_ts.saturating_mul(2).saturating_sub(1),
        in_tb,
        fs_tb,
        Rounding::Down,
    ) >> 1;
    let b = (rescale_q_rnd(
        in_ts.saturating_mul(2).saturating_add(1),
        in_tb,
        fs_tb,
        Rounding::Up,
    )
    .saturating_add(1))
        >> 1;

    if *last < 2 * a - b || *last > 2 * b - a {
        *last = rescale_q(in_ts, in_tb, fs_tb).saturating_add(duration);
        return rescale_q(in_ts, in_tb, out_tb);
    }

    let this_ts = (*last).clamp(a, b);
    *last = this_ts.saturating_add(duration);
    rescale_q(this_ts, fs_tb, out_tb)
}

/// 0th-order modified Bessel function of the first kind $I_0(x)$
/// using Blair & Edwards' minimax rational approximation (matching FFmpeg `av_bessel_i0`).
/// Used for Kaiser-windowed polyphase audio and video resampling filters.
#[allow(clippy::excessive_precision)]
pub fn bessel_i0(mut x: f64) -> f64 {
    const P1: [f64; 15] = [
        -2.2335582639474375249e+15,
        -5.5050369673018427753e+14,
        -3.2940087627407749166e+13,
        -8.4925101247114157499e+11,
        -1.1912746104985237192e+10,
        -1.0313066708737980747e+08,
        -5.9545626019847898221e+05,
        -2.4125195876041896775e+03,
        -7.0935347449210549190e+00,
        -1.5453977791786851041e-02,
        -2.5172644670688975051e-05,
        -3.0517226450451067446e-08,
        -2.6843448573468483278e-11,
        -1.5982226675653184646e-14,
        -5.2487866627945699800e-18,
    ];
    const Q1: [f64; 6] = [
        -2.2335582639474375245e+15,
        7.8858692566751002988e+12,
        -1.2207067397808979846e+10,
        1.0377081058062166144e+07,
        -4.8527560179962773045e+03,
        1.0,
    ];
    const P2: [f64; 7] = [
        -2.2210262233306573296e-04,
        1.3067392038106924055e-02,
        -4.4700805721174453923e-01,
        5.5674518371240761397e+00,
        -2.3517945679239481621e+01,
        3.1611322818701131207e+01,
        -9.6090021968656180000e+00,
    ];
    const Q2: [f64; 8] = [
        -5.5194330231005480228e-04,
        3.2547697594819615062e-02,
        -1.1151759188741312645e+00,
        1.3982595353892851542e+01,
        -6.0228002066743340583e+01,
        8.5539563258012929600e+01,
        -3.1446690275135491500e+01,
        1.0,
    ];

    #[inline]
    fn eval_poly(coeff: &[f64], x: f64) -> f64 {
        let mut sum = *coeff.last().unwrap();
        for &c in coeff.iter().rev().skip(1) {
            sum = sum.mul_add(x, c);
        }
        sum
    }

    if x == 0.0 {
        return 1.0;
    }
    x = x.abs();
    if x <= 15.0 {
        let y = x * x;
        eval_poly(&P1, y) / eval_poly(&Q1, y)
    } else {
        let y = 1.0 / x - (1.0 / 15.0);
        let r = eval_poly(&P2, y) / eval_poly(&Q2, y);
        (x.exp() / x.sqrt()) * r
    }
}

/// A strongly-typed media timestamp coupling integer `ticks` with its exact `Rational64` timebase.
#[derive(Clone, Copy, Debug)]
pub struct Timestamp {
    pub ticks: i64,
    pub timebase: Rational64,
}

impl Timestamp {
    pub const NONE_TICKS: i64 = NOPTS_VALUE;

    pub const NONE: Self = Self {
        ticks: Self::NONE_TICKS,
        timebase: Rational64::TIMEBASE_NANOS,
    };

    #[inline]
    pub const fn new(ticks: i64, timebase: Rational64) -> Self {
        Self { ticks, timebase }
    }

    #[inline]
    pub const fn from_nanos(nanos: i64) -> Self {
        Self {
            ticks: nanos,
            timebase: Rational64::TIMEBASE_NANOS,
        }
    }

    #[inline]
    pub const fn is_valid(self) -> bool {
        self.ticks != Self::NONE_TICKS
    }

    /// Converts this timestamp into `target_tb` using `Rounding::NearInf`.
    #[inline]
    pub const fn rescale_to(self, target_tb: Rational64) -> Self {
        self.rescale_to_rnd(target_tb, Rounding::NearInf)
    }

    /// Converts this timestamp into `target_tb` with an explicit rounding mode.
    #[inline]
    pub const fn rescale_to_rnd(self, target_tb: Rational64, rnd: Rounding) -> Self {
        if !self.is_valid() {
            return Self {
                ticks: Self::NONE_TICKS,
                timebase: target_tb,
            };
        }
        if self.timebase.num == target_tb.num && self.timebase.den == target_tb.den {
            return self;
        }
        Self {
            ticks: rescale_q_rnd(self.ticks, self.timebase, target_tb, rnd),
            timebase: target_tb,
        }
    }

    #[inline]
    pub const fn to_nanos(self) -> Option<i64> {
        if !self.is_valid() {
            None
        } else {
            Some(rescale_q(
                self.ticks,
                self.timebase,
                Rational64::TIMEBASE_NANOS,
            ))
        }
    }

    #[inline]
    pub fn to_seconds_f64(self) -> Option<f64> {
        if !self.is_valid() {
            None
        } else {
            Some((self.ticks as f64) * self.timebase.as_f64())
        }
    }

    /// Formats the timestamp as `HH:MM:SS.mmm`.
    pub fn format_hms(self) -> String {
        let Some(nanos) = self.to_nanos() else {
            return "N/A".to_string();
        };
        let sign = if nanos < 0 { "-" } else { "" };
        let total_ms = nanos.unsigned_abs() / 1_000_000;
        let ms = total_ms % 1_000;
        let total_secs = total_ms / 1_000;
        let secs = total_secs % 60;
        let total_mins = total_secs / 60;
        let mins = total_mins % 60;
        let hours = total_mins / 60;
        format!("{sign}{hours:02}:{mins:02}:{secs:02}.{ms:03}")
    }

    /// Parses `HH:MM:SS.mmm`, `MM:SS.mmm`, or `SS.mmm` into a nanosecond-timebase `Timestamp`.
    pub fn parse_hms(s: &str) -> Result<Self> {
        let s = s.trim();
        if s.is_empty() {
            return Err(RagdollError::InvalidTimestamp("empty timestamp string"));
        }
        let parts: Vec<&str> = s.split(':').collect();
        let (hours, mins, secs_str) = match parts.as_slice() {
            [ss] => (0u64, 0u64, *ss),
            [mm, ss] => (
                0u64,
                mm.parse::<u64>()
                    .map_err(|_| RagdollError::InvalidTimestamp("invalid minutes"))?,
                *ss,
            ),
            [hh, mm, ss] => (
                hh.parse::<u64>()
                    .map_err(|_| RagdollError::InvalidTimestamp("invalid hours"))?,
                mm.parse::<u64>()
                    .map_err(|_| RagdollError::InvalidTimestamp("invalid minutes"))?,
                *ss,
            ),
            _ => return Err(RagdollError::InvalidTimestamp("too many ':' separators")),
        };
        let secs_f64: f64 = secs_str
            .parse()
            .map_err(|_| RagdollError::InvalidTimestamp("invalid seconds component"))?;
        if secs_f64 < 0.0 || !secs_f64.is_finite() {
            return Err(RagdollError::InvalidTimestamp(
                "negative or non-finite seconds",
            ));
        }
        let total_secs = (hours * 3600 + mins * 60) as f64 + secs_f64;
        let nanos = (total_secs * 1_000_000_000.0).round() as i64;
        Ok(Self::from_nanos(nanos))
    }
}

impl Default for Timestamp {
    fn default() -> Self {
        Self::new(0, Rational64::TIMEBASE_NANOS)
    }
}

impl PartialEq for Timestamp {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if !self.is_valid() && !other.is_valid() {
            return true;
        }
        if !self.is_valid() || !other.is_valid() {
            return false;
        }
        compare_ts(self.ticks, self.timebase, other.ticks, other.timebase) == Ordering::Equal
    }
}

impl Eq for Timestamp {}

impl PartialOrd for Timestamp {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Timestamp {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.is_valid(), other.is_valid()) {
            (false, false) => Ordering::Equal,
            (false, true) => Ordering::Less,
            (true, false) => Ordering::Greater,
            (true, true) => compare_ts(self.ticks, self.timebase, other.ticks, other.timebase),
        }
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.is_valid() {
            write!(f, "PTS:NONE")
        } else {
            write!(
                f,
                "{} ({} @ {})",
                self.format_hms(),
                self.ticks,
                self.timebase
            )
        }
    }
}
