//! Fixed-point arithmetic of the libopus fixed-point build (`fixed_generic.h`, `mathops.c`),
//! with the 64-bit multiply variants libopus selects on 64-bit targets.
//!
//! The helpers take `i32` and truncate their 16-bit operands exactly where the C macros cast to
//! `opus_val16`, so call sites read like the reference code. Additions wrap instead of
//! overflowing, matching the reference build on two's-complement targets.

#![allow(dead_code)]

/// `MULT16_16`: product of the low 16 bits of `a` and `b`.
#[inline(always)]
pub const fn mul16(a: i32, b: i32) -> i32 {
    (a as i16 as i32) * (b as i16 as i32)
}

/// `MULT16_16SU`: signed 16-bit by unsigned 16-bit.
#[inline(always)]
pub fn mul16su(a: i32, b: i32) -> i32 {
    (a as i16 as i32).wrapping_mul(b as u16 as i32)
}

#[inline(always)]
pub const fn mul16_q15(a: i32, b: i32) -> i32 {
    mul16(a, b) >> 15
}

#[inline(always)]
pub fn mul16_q14(a: i32, b: i32) -> i32 {
    mul16(a, b) >> 14
}

#[inline(always)]
pub fn mul16_q13(a: i32, b: i32) -> i32 {
    mul16(a, b) >> 13
}

#[inline(always)]
pub fn mul16_q11(a: i32, b: i32) -> i32 {
    mul16(a, b) >> 11
}

#[inline(always)]
pub fn mul16_p15(a: i32, b: i32) -> i32 {
    (16384 + mul16(a, b)) >> 15
}

#[inline(always)]
pub fn mul16_p14(a: i32, b: i32) -> i32 {
    (8192 + mul16(a, b)) >> 14
}

#[inline(always)]
pub fn mul16_p13(a: i32, b: i32) -> i32 {
    (4096 + mul16(a, b)) >> 13
}

/// `MULT16_32_Q15`.
#[inline(always)]
pub fn mul16_32_q15(a: i32, b: i32) -> i32 {
    (((a as i16 as i64) * (b as i64)) >> 15) as i32
}

/// `MULT16_32_Q16`.
#[inline(always)]
pub fn mul16_32_q16(a: i32, b: i32) -> i32 {
    (((a as i16 as i64) * (b as i64)) >> 16) as i32
}

/// `MULT16_32_P16`.
#[inline(always)]
pub fn mul16_32_p16(a: i32, b: i32) -> i32 {
    (((a as i16 as i64) * (b as i64) + 32768) >> 16) as i32
}

/// `MULT32_32_Q31`.
#[inline(always)]
pub fn mul32_q31(a: i32, b: i32) -> i32 {
    (((a as i64) * (b as i64)) >> 31) as i32
}

/// `MAC16_16`.
#[inline(always)]
pub fn mac16(c: i32, a: i32, b: i32) -> i32 {
    c.wrapping_add(mul16(a, b))
}

/// `MAC16_32_Q15`.
#[inline(always)]
pub fn mac16_32_q15(c: i32, a: i32, b: i32) -> i32 {
    c.wrapping_add(mul16(a, b >> 15).wrapping_add(mul16(a, b & 0x7fff) >> 15))
}

/// `MAC16_32_Q16`.
#[inline(always)]
pub fn mac16_32_q16(c: i32, a: i32, b: i32) -> i32 {
    c.wrapping_add(mul16(a, b >> 16).wrapping_add(mul16su(a, b & 0xffff) >> 16))
}

/// `SHL32`: shift through `u32` so bits shifted out are discarded.
#[inline(always)]
pub const fn shl32(a: i32, s: i32) -> i32 {
    ((a as u32) << s) as i32
}

/// `VSHR32`: right shift, or left shift for a negative count.
#[inline(always)]
pub const fn vshr32(a: i32, s: i32) -> i32 {
    if s > 0 { a >> s } else { shl32(a, -s) }
}

/// `PSHR32`: rounding right shift.
#[inline(always)]
pub fn pshr32(a: i32, s: i32) -> i32 {
    a.wrapping_add((1i32 << s) >> 1) >> s
}

/// `ROUND16`.
#[inline(always)]
pub fn round16(x: i32, s: i32) -> i16 {
    pshr32(x, s) as i16
}

/// `SATURATE(x, a)`.
#[inline(always)]
pub fn saturate(x: i32, a: i32) -> i32 {
    if x > a {
        a
    } else if x < -a {
        -a
    } else {
        x
    }
}

/// `SROUND16`: rounding shift saturated to `±32767`.
#[inline(always)]
pub fn sround16(x: i32, s: i32) -> i16 {
    saturate(pshr32(x, s), 32767) as i16
}

/// `SAT16`.
#[inline(always)]
pub fn sat16(x: i32) -> i16 {
    x.clamp(-32768, 32767) as i16
}

/// `SATURATE16` as an `i32`.
#[inline(always)]
pub fn sat16_i32(x: i32) -> i32 {
    x.clamp(-32768, 32767)
}

/// `celt_udiv`.
#[inline(always)]
pub fn celt_udiv(n: u32, d: u32) -> u32 {
    n / d
}

/// `ADD16`: 16-bit wrapping addition.
#[inline(always)]
pub const fn add16(a: i32, b: i32) -> i32 {
    (a as i16).wrapping_add(b as i16) as i32
}

/// `SUB16`: difference of the 16-bit operands (not truncated).
#[inline(always)]
pub const fn sub16(a: i32, b: i32) -> i32 {
    (a as i16 as i32) - (b as i16 as i32)
}

/// `SHL16`.
#[inline(always)]
pub const fn shl16(a: i32, s: i32) -> i32 {
    ((a as u16) << s) as i16 as i32
}

/// `FRAC_MUL16`.
#[inline(always)]
pub fn frac_mul16(a: i32, b: i32) -> i32 {
    (16384 + (a as i16 as i32) * (b as i16 as i32)) >> 15
}

/// `SIG2WORD16`.
#[inline(always)]
pub fn sig2word16(x: i32) -> i16 {
    pshr32(x, 12).clamp(-32768, 32767) as i16
}

/// `celt_ilog2`: floor(log2(x)) for x > 0.
#[inline(always)]
pub const fn ilog2(x: i32) -> i32 {
    31 - (x as u32).leading_zeros() as i32
}

/// `celt_zlog2`.
#[inline(always)]
pub fn zlog2(x: i32) -> i32 {
    if x <= 0 { 0 } else { ilog2(x) }
}

/// `celt_maxabs16`.
pub fn maxabs16(x: &[i16]) -> i32 {
    let mut maxval = 0i16;
    let mut minval = 0i16;
    for &v in x {
        maxval = maxval.max(v);
        minval = minval.min(v);
    }
    (maxval as i32).max(-(minval as i32))
}

/// `celt_maxabs32`.
pub fn maxabs32(x: &[i32]) -> i32 {
    let mut maxval = 0i32;
    let mut minval = 0i32;
    for &v in x {
        maxval = maxval.max(v);
        minval = minval.min(v);
    }
    maxval.max(minval.wrapping_neg())
}

/// `isqrt32`: `floor(sqrt(val))`, exactly. The square root of an `f64` is correctly rounded, so
/// for integers below 2^52 its integer part is the integer square root.
#[inline]
pub fn isqrt32(val: u32) -> u32 {
    (val as f64).sqrt() as u32
}

/// The bit-by-bit `isqrt32` of libopus, the reference for [`isqrt32`].
#[cfg(test)]
pub fn isqrt32_bitwise(mut val: u32) -> u32 {
    let mut g = 0u32;
    let mut bshift = (super::range::ilog(val) - 1) >> 1;
    let mut b = 1u32 << bshift;
    loop {
        let t = ((g << 1) + b) << bshift;
        if t <= val {
            g += b;
            val -= t;
        }
        b >>= 1;
        bshift -= 1;
        if bshift < 0 {
            break;
        }
    }
    g
}

/// `celt_rcp`: 1/x in Q(29 - ilog2(x)) form, as libopus returns it.
pub fn rcp(x: i32) -> i32 {
    let i = ilog2(x);
    let n = (vshr32(x, i - 15) - 32768) as i16 as i32;
    let mut r = add16(30840, mul16_q15(-15420, n));
    r = sub16(r, mul16_q15(r, add16(mul16_q15(r, n), add16(r, -32768)))) as i16 as i32;
    r = sub16(
        r,
        add16(1, mul16_q15(r, add16(mul16_q15(r, n), add16(r, -32768)))),
    ) as i16 as i32;
    vshr32(r, i - 16)
}

/// `celt_div`.
#[inline]
pub fn div(a: i32, b: i32) -> i32 {
    mul32_q31(a, rcp(b))
}

/// `frac_div32`.
pub fn frac_div32(a: i32, b: i32) -> i32 {
    let shift = ilog2(b) - 29;
    let a = vshr32(a, shift);
    let b = vshr32(b, shift);
    let r = round16(rcp(round16(b, 16) as i32), 3) as i32;
    let mut result = mul16_32_q15(r, a);
    let rem = pshr32(a, 2).wrapping_sub(mul32_q31(result, b));
    result = result.wrapping_add(shl32(mul16_32_q15(r, rem), 2));
    if result >= 536870912 {
        2147483647
    } else if result <= -536870912 {
        -2147483647
    } else {
        shl32(result, 2)
    }
}

/// `celt_rsqrt_norm`.
pub const fn rsqrt_norm(x: i32) -> i32 {
    let n = (x - 32768) as i16 as i32;
    let r = add16(23557, mul16_q15(n, add16(-13490, mul16_q15(n, 6713))));
    let r2 = mul16_q15(r, r) as i16 as i32;
    let y = shl16(sub16(add16(mul16_q15(r2, n), r2), 16384), 1);
    add16(
        r,
        mul16_q15(r, mul16_q15(y, sub16(mul16_q15(y, 12288), 16384))),
    )
}

/// `celt_sqrt`.
pub fn sqrt(x: i32) -> i32 {
    const C: [i32; 5] = [23175, 11561, -3011, 1699, -664];
    if x == 0 {
        return 0;
    } else if x >= 1073741824 {
        return 32767;
    }
    let k = (ilog2(x) >> 1) - 7;
    let x = vshr32(x, 2 * k);
    let n = (x - 32768) as i16 as i32;
    let rt = add16(
        C[0],
        mul16_q15(
            n,
            add16(
                C[1],
                mul16_q15(
                    n,
                    add16(C[2], mul16_q15(n, add16(C[3], mul16_q15(n, C[4])))),
                ),
            ),
        ),
    );
    vshr32(rt, 7 - k)
}

fn cos_pi_2(x: i32) -> i32 {
    let x2 = mul16_p15(x, x);
    add16(
        1,
        (32766).min(
            sub16(32767, x2) + mul16_p15(x2, -7651 + mul16_p15(x2, 8277 + mul16_p15(-626, x2))),
        ),
    )
}

/// `celt_cos_norm`.
pub fn cos_norm(x: i32) -> i32 {
    let mut x = x & 0x0001ffff;
    if x > (1 << 16) {
        x = (1 << 17) - x;
    }
    if x & 0x00007fff != 0 {
        if x < (1 << 15) {
            cos_pi_2(x as i16 as i32)
        } else {
            -cos_pi_2((65536 - x) as i16 as i32)
        }
    } else if x & 0x0000ffff != 0 {
        0
    } else if x & 0x0001ffff != 0 {
        -32767
    } else {
        32767
    }
}

/// `celt_log2`: Q`DB_SHIFT` base-2 logarithm.
pub fn log2(x: i32) -> i32 {
    const C: [i32; 5] = [-6801 + (1 << 3), 15746, -5217, 2545, -1401];
    if x == 0 {
        return -32767;
    }
    let i = ilog2(x);
    let n = (vshr32(x, i - 15) - 32768 - 16384) as i16 as i32;
    let frac = add16(
        C[0],
        mul16_q15(
            n,
            add16(
                C[1],
                mul16_q15(
                    n,
                    add16(C[2], mul16_q15(n, add16(C[3], mul16_q15(n, C[4])))),
                ),
            ),
        ),
    );
    (shl16(i - 13, 10) + (frac >> 4)) as i16 as i32
}

/// `celt_exp2_frac`.
#[inline]
pub fn exp2_frac(x: i32) -> i32 {
    let frac = shl16(x, 4);
    add16(
        16383,
        mul16_q15(
            frac,
            add16(22804, mul16_q15(frac, add16(14819, mul16_q15(10204, frac)))),
        ),
    )
}

/// `celt_exp2`: input Q10, output Q16.
pub fn exp2(x: i32) -> i32 {
    let integer = (x as i16 as i32) >> 10;
    if integer > 14 {
        return 0x7f000000;
    } else if integer < -15 {
        return 0;
    }
    let frac = exp2_frac(x - shl16(integer, 10)) as i16 as i32;
    vshr32(frac, -integer - 2)
}

fn atan01(x: i32) -> i32 {
    mul16_p15(
        x,
        32767 + mul16_p15(x, -21 + mul16_p15(x, -11943 + mul16_p15(4936, x))),
    )
}

/// `celt_atan2p`.
pub fn atan2p(y: i32, x: i32) -> i32 {
    if y < x {
        let arg = div(shl32(y, 15), x).min(32767);
        (atan01(arg as i16 as i32) as i16 as i32) >> 1
    } else {
        let arg = div(shl32(x, 15), y).min(32767);
        25736 - ((atan01(arg as i16 as i32) as i16 as i32) >> 1)
    }
}

/// `celt_inner_prod`.
#[inline]
pub fn inner_prod(x: &[i16], y: &[i16]) -> i32 {
    x.iter()
        .zip(y)
        .fold(0i32, |acc, (&a, &b)| acc.wrapping_add(a as i32 * b as i32))
}

/// `celt_lcg_rand`.
#[inline]
pub fn lcg_rand(seed: u32) -> u32 {
    1664525u32.wrapping_mul(seed).wrapping_add(1013904223)
}

#[cfg(test)]
mod tests {
    #[test]
    fn isqrt32_matches_bitwise() {
        let mut v = 1u32;
        while v < u32::MAX / 3 {
            for d in [0u32, 1, 2] {
                let x = v - d.min(v - 1);
                assert_eq!(super::isqrt32(x), super::isqrt32_bitwise(x), "{x}");
                let s = super::isqrt32_bitwise(x) + 1;
                for y in [s * s - 1, s * s, s * s + 1] {
                    assert_eq!(super::isqrt32(y), super::isqrt32_bitwise(y), "{y}");
                }
            }
            v = v + v / 7 + 1;
        }
        for x in 1..200_000u32 {
            assert_eq!(super::isqrt32(x), super::isqrt32_bitwise(x), "{x}");
        }
        assert_eq!(super::isqrt32(u32::MAX), super::isqrt32_bitwise(u32::MAX));
    }
}
