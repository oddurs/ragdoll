//! FLAC encoder analysis kernels: windowed autocorrelation, LPC residuals and Rice partition sums.
//!
//! Every SIMD path is bit-exact with its `*_scalar` reference. The integer kernels are exact by
//! construction; the autocorrelation accumulates its products in one fixed order (four interleaved
//! lanes `i ≡ j (mod 4)`, then `((l0 + l1) + (l2 + l3)) + tail`) with fused multiply-adds
//! everywhere (`f64::mul_add`, `vfmadd`, `fmla`), so the scalar, AVX2 and NEON paths round
//! identically and the encoder writes the same stream on every level.

/// Autocorrelation `out[l] = Σ x[i] · x[i + l]` of `x` for every lag `l < out.len()`.
///
/// Lags at or beyond `x.len()` are `0`.
#[inline]
pub fn flac_autocorr_f64(x: &[f64], out: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() && std::is_x86_feature_detected!("fma") {
        // Safety: the CPU supports AVX2 and FMA.
        return unsafe { flac_autocorr_f64_avx2(x, out) };
    }
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: NEON is baseline on every aarch64 CPU.
        return unsafe { flac_autocorr_f64_neon(x, out) };
    }
    flac_autocorr_f64_scalar(x, out)
}

/// Scalar reference for [`flac_autocorr_f64`], defining the summation order.
pub fn flac_autocorr_f64_scalar(x: &[f64], out: &mut [f64]) {
    for (lag, o) in out.iter_mut().enumerate() {
        *o = autocorr_lag(x, lag);
    }
}

/// One lag of [`flac_autocorr_f64_scalar`].
fn autocorr_lag(x: &[f64], lag: usize) -> f64 {
    let n = x.len();
    if lag >= n {
        return 0.0;
    }
    let body = (n - lag) & !3;
    let mut acc = [0.0f64; 4];
    for i in (0..body).step_by(4) {
        for (j, a) in acc.iter_mut().enumerate() {
            *a = x[i + j].mul_add(x[i + j + lag], *a);
        }
    }
    autocorr_finish(x, lag, body, acc)
}

/// Folds the four lanes of lag `lag` and adds the products after `body`.
#[inline(always)]
fn autocorr_finish(x: &[f64], lag: usize, body: usize, acc: [f64; 4]) -> f64 {
    let mut tail = 0.0f64;
    for i in body..x.len() - lag {
        tail = x[i].mul_add(x[i + lag], tail);
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + tail
}

/// Lags per pass of the SIMD autocorrelation: one load of `x[i..i + 4]` feeds this many lanes of
/// independent accumulators, which also hides the add latency.
const AUTOCORR_GROUP: usize = 8;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn flac_autocorr_f64_avx2(x: &[f64], out: &mut [f64]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let lags = out.len();
    let p = x.as_ptr();
    let mut l0 = 0;
    while l0 < lags {
        let g = (lags - l0).min(AUTOCORR_GROUP);
        if l0 + g > n {
            // Lags reaching past the block: the reference handles the short and empty cases.
            for (k, o) in out[l0..].iter_mut().enumerate() {
                *o = autocorr_lag(x, l0 + k);
            }
            return;
        }
        let mut acc = [_mm256_setzero_pd(); AUTOCORR_GROUP];
        // Every lag of the group shares `i < common`; each then runs on alone up to its own body.
        let common = (n - (l0 + g - 1)) & !3;
        let mut i = 0;
        // Safety: `i + 4 <= common <= n - lag` for every lag of the group.
        unsafe {
            while i < common {
                let a = _mm256_loadu_pd(p.add(i));
                for (k, acc_k) in acc.iter_mut().enumerate().take(g) {
                    let b = _mm256_loadu_pd(p.add(i + l0 + k));
                    *acc_k = _mm256_fmadd_pd(a, b, *acc_k);
                }
                i += 4;
            }
            for (k, acc_k) in acc.iter_mut().enumerate().take(g) {
                let lag = l0 + k;
                let body = (n - lag) & !3;
                let mut j = common;
                while j < body {
                    let a = _mm256_loadu_pd(p.add(j));
                    let b = _mm256_loadu_pd(p.add(j + lag));
                    *acc_k = _mm256_fmadd_pd(a, b, *acc_k);
                    j += 4;
                }
                let mut lanes = [0.0f64; 4];
                _mm256_storeu_pd(lanes.as_mut_ptr(), *acc_k);
                out[lag] = autocorr_finish(x, lag, body, lanes);
            }
        }
        l0 += g;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn flac_autocorr_f64_neon(x: &[f64], out: &mut [f64]) {
    use std::arch::aarch64::*;
    let n = x.len();
    let lags = out.len();
    let p = x.as_ptr();
    let mut l0 = 0;
    while l0 < lags {
        let g = (lags - l0).min(AUTOCORR_GROUP);
        if l0 + g > n {
            // Lags reaching past the block: the reference handles the short and empty cases.
            for (k, o) in out[l0..].iter_mut().enumerate() {
                *o = autocorr_lag(x, l0 + k);
            }
            return;
        }
        // Lanes 0-1 and 2-3 of the reference order live in two `float64x2_t` accumulators.
        let mut lo = [vdupq_n_f64(0.0); AUTOCORR_GROUP];
        let mut hi = [vdupq_n_f64(0.0); AUTOCORR_GROUP];
        let common = (n - (l0 + g - 1)) & !3;
        let mut i = 0;
        // Safety: `i + 4 <= common <= n - lag` for every lag of the group.
        unsafe {
            while i < common {
                let a0 = vld1q_f64(p.add(i));
                let a1 = vld1q_f64(p.add(i + 2));
                for k in 0..g {
                    let b0 = vld1q_f64(p.add(i + l0 + k));
                    let b1 = vld1q_f64(p.add(i + l0 + k + 2));
                    lo[k] = vfmaq_f64(lo[k], a0, b0);
                    hi[k] = vfmaq_f64(hi[k], a1, b1);
                }
                i += 4;
            }
            for k in 0..g {
                let lag = l0 + k;
                let body = (n - lag) & !3;
                let mut j = common;
                while j < body {
                    let a0 = vld1q_f64(p.add(j));
                    let a1 = vld1q_f64(p.add(j + 2));
                    let b0 = vld1q_f64(p.add(j + lag));
                    let b1 = vld1q_f64(p.add(j + lag + 2));
                    lo[k] = vfmaq_f64(lo[k], a0, b0);
                    hi[k] = vfmaq_f64(hi[k], a1, b1);
                    j += 4;
                }
                let lanes = [
                    vgetq_lane_f64::<0>(lo[k]),
                    vgetq_lane_f64::<1>(lo[k]),
                    vgetq_lane_f64::<0>(hi[k]),
                    vgetq_lane_f64::<1>(hi[k]),
                ];
                out[lag] = autocorr_finish(x, lag, body, lanes);
            }
        }
        l0 += g;
    }
}

/// LPC residual `res[i] = x[i] - ((Σ_j coeffs[j] · x[i - 1 - j]) >> shift)` for `i >= order`, with
/// `res[..order] = x[..order]` (the warm-up samples), all in wrapping `i32` arithmetic.
///
/// The result is the exact residual whenever `Σ |coeffs[j]| · max |x| < 2^31`; the caller checks
/// that bound. `coeffs` holds at most 32 taps and `res` is as long as `x`.
#[inline]
pub fn flac_lpc_residual_i32(x: &[i32], coeffs: &[i32], shift: u32, res: &mut [i32]) {
    assert!(coeffs.len() <= 32 && res.len() == x.len() && shift < 32);
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        return unsafe { flac_lpc_residual_i32_avx2(x, coeffs, shift, res) };
    }
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: NEON is baseline on every aarch64 CPU.
        return unsafe { flac_lpc_residual_i32_neon(x, coeffs, shift, res) };
    }
    flac_lpc_residual_i32_scalar(x, coeffs, shift, res)
}

/// Scalar reference for [`flac_lpc_residual_i32`].
pub fn flac_lpc_residual_i32_scalar(x: &[i32], coeffs: &[i32], shift: u32, res: &mut [i32]) {
    let order = coeffs.len().min(x.len());
    res[..order].copy_from_slice(&x[..order]);
    lpc_residual_tail(x, coeffs, shift, res, order);
}

/// [`flac_lpc_residual_i32_scalar`] from sample `from` on.
#[inline(always)]
fn lpc_residual_tail(x: &[i32], coeffs: &[i32], shift: u32, res: &mut [i32], from: usize) {
    let order = coeffs.len();
    for i in from.max(order)..x.len() {
        let mut sum = 0i32;
        for (j, &c) in coeffs.iter().enumerate() {
            sum = sum.wrapping_add(c.wrapping_mul(x[i - 1 - j]));
        }
        res[i] = x[i].wrapping_sub(sum >> shift);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn flac_lpc_residual_i32_avx2(x: &[i32], coeffs: &[i32], shift: u32, res: &mut [i32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let order = coeffs.len().min(n);
    res[..order].copy_from_slice(&x[..order]);
    let mut c = [_mm256_setzero_si256(); 32];
    for (cv, &cj) in c.iter_mut().zip(coeffs) {
        *cv = _mm256_set1_epi32(cj);
    }
    let sh = _mm_cvtsi32_si128(shift as i32);
    let p = x.as_ptr();
    let q = res.as_mut_ptr();
    let mut i = order;
    // Safety: every load reads `x[i - 1 - j .. i + 15 - j]` with `j < order <= i` and `i + 16 <= n`.
    unsafe {
        while i + 16 <= n {
            let mut acc0 = _mm256_setzero_si256();
            let mut acc1 = _mm256_setzero_si256();
            for (j, &cj) in c.iter().enumerate().take(order) {
                let s0 = _mm256_loadu_si256(p.add(i - 1 - j) as *const __m256i);
                let s1 = _mm256_loadu_si256(p.add(i + 7 - j) as *const __m256i);
                acc0 = _mm256_add_epi32(acc0, _mm256_mullo_epi32(cj, s0));
                acc1 = _mm256_add_epi32(acc1, _mm256_mullo_epi32(cj, s1));
            }
            let x0 = _mm256_loadu_si256(p.add(i) as *const __m256i);
            let x1 = _mm256_loadu_si256(p.add(i + 8) as *const __m256i);
            let r0 = _mm256_sub_epi32(x0, _mm256_sra_epi32(acc0, sh));
            let r1 = _mm256_sub_epi32(x1, _mm256_sra_epi32(acc1, sh));
            _mm256_storeu_si256(q.add(i) as *mut __m256i, r0);
            _mm256_storeu_si256(q.add(i + 8) as *mut __m256i, r1);
            i += 16;
        }
    }
    lpc_residual_tail(x, &coeffs[..order], shift, res, i);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn flac_lpc_residual_i32_neon(x: &[i32], coeffs: &[i32], shift: u32, res: &mut [i32]) {
    use std::arch::aarch64::*;
    let n = x.len();
    let order = coeffs.len().min(n);
    res[..order].copy_from_slice(&x[..order]);
    let neg_shift = vdupq_n_s32(-(shift as i32));
    let p = x.as_ptr();
    let q = res.as_mut_ptr();
    let mut i = order;
    // Safety: every load reads `x[i - 1 - j .. i + 7 - j]` with `j < order <= i` and `i + 8 <= n`.
    unsafe {
        while i + 8 <= n {
            let mut acc0 = vdupq_n_s32(0);
            let mut acc1 = vdupq_n_s32(0);
            for (j, &cj) in coeffs.iter().enumerate().take(order) {
                let s0 = vld1q_s32(p.add(i - 1 - j));
                let s1 = vld1q_s32(p.add(i + 3 - j));
                acc0 = vmlaq_n_s32(acc0, s0, cj);
                acc1 = vmlaq_n_s32(acc1, s1, cj);
            }
            let x0 = vld1q_s32(p.add(i));
            let x1 = vld1q_s32(p.add(i + 4));
            vst1q_s32(q.add(i), vsubq_s32(x0, vshlq_s32(acc0, neg_shift)));
            vst1q_s32(q.add(i + 4), vsubq_s32(x1, vshlq_s32(acc1, neg_shift)));
            i += 8;
        }
    }
    lpc_residual_tail(x, &coeffs[..order], shift, res, i);
}

/// Sums the zig-zag folded residuals `u = (r << 1) ^ (r >> 31)` of each Rice partition and ORs
/// them together (whose bit length is the raw width an escaped partition needs).
///
/// Partition `p` covers `res[p * part_len .. (p + 1) * part_len]`, except that the first `skip`
/// samples (the predictor's warm-up) belong to no partition. `sums` and `ors` hold one entry per
/// partition, `res.len() / part_len` of them.
#[inline]
pub fn flac_rice_partition_sums(
    res: &[i32],
    skip: usize,
    part_len: usize,
    sums: &mut [u64],
    ors: &mut [u32],
) {
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        return unsafe { flac_rice_partition_sums_avx2(res, skip, part_len, sums, ors) };
    }
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: NEON is baseline on every aarch64 CPU.
        return unsafe { flac_rice_partition_sums_neon(res, skip, part_len, sums, ors) };
    }
    flac_rice_partition_sums_scalar(res, skip, part_len, sums, ors)
}

#[inline(always)]
fn fold(r: i32) -> u32 {
    ((r << 1) ^ (r >> 31)) as u32
}

/// Scalar reference for [`flac_rice_partition_sums`].
pub fn flac_rice_partition_sums_scalar(
    res: &[i32],
    skip: usize,
    part_len: usize,
    sums: &mut [u64],
    ors: &mut [u32],
) {
    for (p, (sum, or)) in sums.iter_mut().zip(ors.iter_mut()).enumerate() {
        let start = (p * part_len).max(skip);
        let end = (p + 1) * part_len;
        let (mut s, mut o) = (0u64, 0u32);
        for &r in &res[start.min(end)..end] {
            let u = fold(r);
            s += u as u64;
            o |= u;
        }
        *sum = s;
        *or = o;
    }
}

/// AVX2 [`flac_rice_partition_sums`]: partitions of a multiple of 8 samples go eight at a time,
/// summed in 32-bit lanes and reduced together; the warm-up partition, the stragglers and any
/// group whose 32-bit sums could overflow take the one-partition-at-a-time path.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn flac_rice_partition_sums_avx2(
    res: &[i32],
    skip: usize,
    part_len: usize,
    sums: &mut [u64],
    ors: &mut [u32],
) {
    let parts = sums.len();
    assert!(ors.len() == parts && parts * part_len <= res.len());
    let mut g = 1;
    if part_len.is_multiple_of(8) && part_len > 0 {
        while g + 8 <= parts {
            // Safety: the eight partitions lie within `res` (asserted above).
            let ok = unsafe {
                partition_sums_x8_avx2(
                    res.as_ptr().add(g * part_len),
                    part_len,
                    &mut sums[g..g + 8],
                    &mut ors[g..g + 8],
                )
            };
            if !ok {
                // Safety: AVX2 is enabled for this function.
                unsafe {
                    partition_sums_each_avx2(
                        res,
                        skip,
                        part_len,
                        g,
                        &mut sums[..g + 8],
                        &mut ors[..g + 8],
                    )
                };
            }
            g += 8;
        }
    }
    // Safety: AVX2 is enabled for this function.
    unsafe {
        partition_sums_each_avx2(res, skip, part_len, 0, &mut sums[..1], &mut ors[..1]);
        if g < parts {
            partition_sums_each_avx2(res, skip, part_len, g.max(1), sums, ors);
        }
    }
}

/// Sums and ORs of eight consecutive partitions of `part_len` (a multiple of 8) samples from `p`;
/// `false` (and nothing written) when the 32-bit sums could have overflowed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn partition_sums_x8_avx2(
    p: *const i32,
    part_len: usize,
    sums: &mut [u64],
    ors: &mut [u32],
) -> bool {
    use std::arch::x86_64::*;
    let mut acc = [_mm256_setzero_si256(); 8];
    let mut ov = [_mm256_setzero_si256(); 8];
    for (l, (a, o)) in acc.iter_mut().zip(ov.iter_mut()).enumerate() {
        let mut j = 0;
        while j < part_len {
            // Safety: the caller guarantees `8 * part_len` readable samples from `p`.
            let r = unsafe { _mm256_loadu_si256(p.add(l * part_len + j) as *const __m256i) };
            let u = _mm256_xor_si256(_mm256_slli_epi32::<1>(r), _mm256_srai_epi32::<31>(r));
            *a = _mm256_add_epi32(*a, u);
            *o = _mm256_or_si256(*o, u);
            j += 8;
        }
    }
    // Reduces eight vectors with `op` to one whose lane `l` combines all of vector `l`: pairs
    // `[x0 ∘ x1, x2 ∘ x3, y0 ∘ y1, y2 ∘ y3]` per 128-bit half (what `vphaddd` does for `+`), twice,
    // then the two halves.
    macro_rules! reduce {
        ($v:expr, $op:ident) => {{
            let pair = |x: __m256i, y: __m256i| {
                let (x, y) = (_mm256_castsi256_ps(x), _mm256_castsi256_ps(y));
                $op(
                    _mm256_castps_si256(_mm256_shuffle_ps::<0b10_00_10_00>(x, y)),
                    _mm256_castps_si256(_mm256_shuffle_ps::<0b11_01_11_01>(x, y)),
                )
            };
            let v = $v;
            let q0 = pair(pair(v[0], v[1]), pair(v[2], v[3]));
            let q1 = pair(pair(v[4], v[5]), pair(v[6], v[7]));
            $op(
                _mm256_permute2x128_si256::<0x20>(q0, q1),
                _mm256_permute2x128_si256::<0x31>(q0, q1),
            )
        }};
    }
    let s = reduce!(acc, _mm256_add_epi32);
    let o = reduce!(ov, _mm256_or_si256);
    let mut lanes = [0u32; 8];
    // Safety: `lanes` holds 32 bytes.
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, o) };
    let all = lanes.iter().fold(0, |a, &b| a | b);
    // Every sum is below `part_len · 2^bits`.
    if (32 - all.leading_zeros()) + part_len.next_power_of_two().trailing_zeros() > 32 {
        return false;
    }
    ors.copy_from_slice(&lanes);
    // Safety: `sums` holds eight `u64`.
    unsafe {
        let q = sums.as_mut_ptr() as *mut __m256i;
        _mm256_storeu_si256(q, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(s)));
        _mm256_storeu_si256(
            q.add(1),
            _mm256_cvtepu32_epi64(_mm256_extracti128_si256::<1>(s)),
        );
    }
    true
}

/// One partition at a time, for partitions `from..sums.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn partition_sums_each_avx2(
    res: &[i32],
    skip: usize,
    part_len: usize,
    from: usize,
    sums: &mut [u64],
    ors: &mut [u32],
) {
    use std::arch::x86_64::*;
    let p = res.as_ptr();
    for (pi, (sum, or)) in sums.iter_mut().zip(ors.iter_mut()).enumerate().skip(from) {
        let mut start = (pi * part_len).max(skip);
        let end = (pi + 1) * part_len;
        assert!(end <= res.len());
        let (mut s, mut o) = (0u64, 0u32);
        // Scalar head up to a multiple of 8 from the partition start.
        while start < end && !(start - pi * part_len).is_multiple_of(8) {
            let u = fold(res[start]);
            s += u as u64;
            o |= u;
            start += 1;
        }
        let mut acc = _mm256_setzero_si256();
        let mut ov = _mm256_setzero_si256();
        let mut i = start;
        // Safety: `i + 8 <= end <= res.len()`.
        unsafe {
            while i + 8 <= end {
                let r = _mm256_loadu_si256(p.add(i) as *const __m256i);
                let u = _mm256_xor_si256(_mm256_slli_epi32::<1>(r), _mm256_srai_epi32::<31>(r));
                ov = _mm256_or_si256(ov, u);
                let lo = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(u));
                let hi = _mm256_cvtepu32_epi64(_mm256_extracti128_si256::<1>(u));
                acc = _mm256_add_epi64(acc, _mm256_add_epi64(lo, hi));
                i += 8;
            }
            let mut a = [0u64; 4];
            _mm256_storeu_si256(a.as_mut_ptr() as *mut __m256i, acc);
            let mut b = [0u32; 8];
            _mm256_storeu_si256(b.as_mut_ptr() as *mut __m256i, ov);
            s += a.iter().sum::<u64>();
            o |= b.iter().fold(0, |x, &y| x | y);
        }
        for &r in &res[i..end] {
            let u = fold(r);
            s += u as u64;
            o |= u;
        }
        *sum = s;
        *or = o;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn flac_rice_partition_sums_neon(
    res: &[i32],
    skip: usize,
    part_len: usize,
    sums: &mut [u64],
    ors: &mut [u32],
) {
    use std::arch::aarch64::*;
    let p = res.as_ptr();
    for (pi, (sum, or)) in sums.iter_mut().zip(ors.iter_mut()).enumerate() {
        let start = (pi * part_len).max(skip);
        let end = (pi + 1) * part_len;
        assert!(end <= res.len());
        let mut acc = vdupq_n_u64(0);
        let mut ov = vdupq_n_u32(0);
        let mut i = start.min(end);
        // Safety: `i + 4 <= end <= res.len()`.
        unsafe {
            while i + 4 <= end {
                let r = vld1q_s32(p.add(i));
                let u = vreinterpretq_u32_s32(veorq_s32(vshlq_n_s32::<1>(r), vshrq_n_s32::<31>(r)));
                ov = vorrq_u32(ov, u);
                acc = vpadalq_u32(acc, u);
                i += 4;
            }
        }
        let mut s = vaddvq_u64(acc);
        let mut o = vgetq_lane_u32::<0>(ov)
            | vgetq_lane_u32::<1>(ov)
            | vgetq_lane_u32::<2>(ov)
            | vgetq_lane_u32::<3>(ov);
        for &r in &res[i..end] {
            let u = fold(r);
            s += u as u64;
            o |= u;
        }
        *sum = s;
        *or = o;
    }
}

/// Rice parameter marking an escaped partition in [`flac_rice_partition_costs`].
pub const FLAC_RICE_ESCAPED: u8 = u8::MAX;

/// Estimated cost in bits of Rice-coding each partition, and its best parameter.
///
/// Partition `i` holds `cnt` residuals (`first_cnt` for `i == 0`) whose zig-zag folded values sum
/// to `sums[i]` (below `2^50`) and OR to `ors[i]`. Its parameter `k` is the one that minimizes the
/// estimate `cnt · (k + 1) + ((sum - cnt · (2^k - 1) / 2) >> k)` (exact for `k = 0`), which is
/// `floor(log2((2 · sum + cnt) / (2 · cnt)))` capped at 30, unless escaping the partition to raw
/// `bitlen(or)`-bit samples (`5 + cnt · bitlen(or)` bits) is cheaper; then `params[i]` is
/// [`FLAC_RICE_ESCAPED`]. `bits` excludes the parameter field itself. `cnt` and `first_cnt` are
/// below `2^16`.
#[inline]
pub fn flac_rice_partition_costs(
    sums: &[u64],
    ors: &[u32],
    cnt: u32,
    first_cnt: u32,
    bits: &mut [u32],
    params: &mut [u8],
) {
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        return unsafe { flac_rice_partition_costs_avx2(sums, ors, cnt, first_cnt, bits, params) };
    }
    flac_rice_partition_costs_scalar(sums, ors, cnt, first_cnt, bits, params)
}

/// One partition of [`flac_rice_partition_costs`]: `(bits, parameter)`.
#[inline(always)]
fn rice_partition_cost(sum: u64, or: u32, cnt: u32) -> (u32, u8) {
    let cnt = cnt as u64;
    let (a, d) = (2 * sum + cnt, 2 * cnt.max(1));
    let k = if a < d {
        0
    } else {
        let k = d.leading_zeros() - a.leading_zeros();
        (k - u32::from((d << k) > a)).min(30)
    };
    let rice = cnt * (k as u64 + 1) + (sum.saturating_sub(cnt * ((1u64 << k) - 1) / 2) >> k);
    let raw = 32 - or.leading_zeros();
    let esc = 5 + cnt * raw as u64;
    if raw < 32 && esc < rice {
        (esc as u32, FLAC_RICE_ESCAPED)
    } else {
        (rice.min(u32::MAX as u64) as u32, k as u8)
    }
}

/// Scalar reference for [`flac_rice_partition_costs`].
pub fn flac_rice_partition_costs_scalar(
    sums: &[u64],
    ors: &[u32],
    cnt: u32,
    first_cnt: u32,
    bits: &mut [u32],
    params: &mut [u8],
) {
    for (i, ((&s, &o), (b, p))) in sums
        .iter()
        .zip(ors)
        .zip(bits.iter_mut().zip(params.iter_mut()))
        .enumerate()
    {
        (*b, *p) = rice_partition_cost(s, o, if i == 0 { first_cnt } else { cnt });
    }
}

/// AVX2 [`flac_rice_partition_costs`], four partitions per step in 64-bit lanes. `floor(log2)`
/// comes from the exponent of the exact `f64` of each value (all below `2^52`), built by OR-ing it
/// into the mantissa of `2^52`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn flac_rice_partition_costs_avx2(
    sums: &[u64],
    ors: &[u32],
    cnt: u32,
    first_cnt: u32,
    bits: &mut [u32],
    params: &mut [u8],
) {
    use std::arch::x86_64::*;
    let m = sums.len();
    assert!(ors.len() >= m && bits.len() >= m && params.len() >= m);
    if m == 0 {
        return;
    }
    (bits[0], params[0]) = rice_partition_cost(sums[0], ors[0], first_cnt);
    // floor(log2(v)) for 0 < v < 2^52 in each 64-bit lane (garbage for v = 0).
    let magic = _mm256_set1_epi64x(0x4330_0000_0000_0000);
    let ilog2 = |v: __m256i| {
        let f = _mm256_sub_pd(
            _mm256_castsi256_pd(_mm256_or_si256(v, magic)),
            _mm256_castsi256_pd(magic),
        );
        _mm256_sub_epi64(
            _mm256_srli_epi64::<52>(_mm256_castpd_si256(f)),
            _mm256_set1_epi64x(1023),
        )
    };
    let c = cnt as u64;
    let vc = _mm256_set1_epi64x(c as i64);
    let d = 2 * c.max(1);
    let vd = _mm256_set1_epi64x(d as i64);
    let log_d = _mm256_set1_epi64x((63 - d.leading_zeros()) as i64);
    let one = _mm256_set1_epi64x(1);
    let zero = _mm256_setzero_si256();
    let thirty = _mm256_set1_epi64x(30);
    let mut i = 1;
    while i + 4 <= m {
        // Safety: `i + 4 <= m` for the loads of `sums` and `ors`.
        let (s, o) = unsafe {
            (
                _mm256_loadu_si256(sums.as_ptr().add(i) as *const __m256i),
                _mm256_cvtepu32_epi64(_mm_loadu_si128(ors.as_ptr().add(i) as *const __m128i)),
            )
        };
        let a = _mm256_add_epi64(_mm256_add_epi64(s, s), vc);
        // k = floor(log2(a)) - floor(log2(d)), minus one if d << k overshoots a; 0 when a < d.
        let mut k = _mm256_sub_epi64(ilog2(a), log_d);
        let below = _mm256_cmpgt_epi64(vd, a);
        k = _mm256_andnot_si256(below, k);
        let over = _mm256_andnot_si256(below, _mm256_cmpgt_epi64(_mm256_sllv_epi64(vd, k), a));
        k = _mm256_add_epi64(k, over);
        k = _mm256_blendv_epi8(k, thirty, _mm256_cmpgt_epi64(k, thirty));
        // rice = cnt · (k + 1) + (sat(sum - cnt · (2^k - 1) / 2) >> k)
        let low = _mm256_srli_epi64::<1>(_mm256_mul_epu32(
            vc,
            _mm256_sub_epi64(_mm256_sllv_epi64(one, k), one),
        ));
        let rem = _mm256_andnot_si256(_mm256_cmpgt_epi64(low, s), _mm256_sub_epi64(s, low));
        let rice = _mm256_add_epi64(
            _mm256_mul_epu32(vc, _mm256_add_epi64(k, one)),
            _mm256_srlv_epi64(rem, k),
        );
        // raw = bitlen(or); escape when raw < 32 and 5 + cnt · raw < rice.
        let raw = _mm256_andnot_si256(_mm256_cmpeq_epi64(o, zero), _mm256_add_epi64(ilog2(o), one));
        let esc = _mm256_add_epi64(_mm256_mul_epu32(vc, raw), _mm256_set1_epi64x(5));
        let use_esc = _mm256_andnot_si256(
            _mm256_cmpgt_epi64(raw, _mm256_set1_epi64x(31)),
            _mm256_cmpgt_epi64(rice, esc),
        );
        let b = _mm256_blendv_epi8(rice, esc, use_esc);
        let p = _mm256_or_si256(k, use_esc);
        let mut bl = [0u64; 4];
        let mut pl = [0u64; 4];
        // Safety: both arrays hold 32 bytes.
        unsafe {
            _mm256_storeu_si256(bl.as_mut_ptr() as *mut __m256i, b);
            _mm256_storeu_si256(pl.as_mut_ptr() as *mut __m256i, p);
        }
        for j in 0..4 {
            bits[i + j] = bl[j].min(u32::MAX as u64) as u32;
            params[i + j] = pl[j] as u8;
        }
        i += 4;
    }
    for j in i..m {
        (bits[j], params[j]) = rice_partition_cost(sums[j], ors[j], cnt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn signed(&mut self, bits: u32) -> i32 {
            ((self.next() >> 11) as i64 as i32) >> (32 - bits)
        }
    }

    #[test]
    fn autocorr_matches_scalar_bit_exactly() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        for n in [
            0usize, 1, 2, 3, 5, 7, 8, 13, 16, 31, 33, 64, 191, 1152, 4095, 4096, 4608,
        ] {
            for bits in [8u32, 17, 25] {
                let x: Vec<f64> = (0..n)
                    .map(|i| {
                        let w =
                            1.0 - ((i as f64 - n as f64 / 2.0) / (n as f64 / 2.0 + 1.0)).powi(2);
                        rng.signed(bits) as f64 * w
                    })
                    .collect();
                for lags in [1usize, 5, 9, 13, 17, 33] {
                    let mut a = vec![f64::NAN; lags];
                    let mut b = vec![f64::NAN; lags];
                    flac_autocorr_f64(&x, &mut a);
                    flac_autocorr_f64_scalar(&x, &mut b);
                    let (a, b): (Vec<u64>, Vec<u64>) = (
                        a.iter().map(|v| v.to_bits()).collect(),
                        b.iter().map(|v| v.to_bits()).collect(),
                    );
                    assert_eq!(a, b, "n={n} bits={bits} lags={lags}");
                }
            }
        }
    }

    #[test]
    fn lpc_residual_matches_scalar() {
        let mut rng = Rng(0xFEED_0123_4567_89AB);
        for n in [1usize, 2, 7, 9, 16, 17, 33, 100, 4096, 4607] {
            for order in [1usize, 2, 3, 4, 7, 8, 12, 13, 32] {
                for (xbits, cbits) in [(16u32, 15u32), (25, 6), (32, 15)] {
                    let x: Vec<i32> = (0..n).map(|_| rng.signed(xbits)).collect();
                    let c: Vec<i32> = (0..order).map(|_| rng.signed(cbits)).collect();
                    for shift in [0u32, 9, 15, 31] {
                        let mut a = vec![0x5555_5555; n];
                        let mut b = vec![0x2222_2222; n];
                        flac_lpc_residual_i32(&x, &c, shift, &mut a);
                        flac_lpc_residual_i32_scalar(&x, &c, shift, &mut b);
                        assert_eq!(a, b, "n={n} order={order} shift={shift}");
                    }
                }
            }
        }
    }

    #[test]
    fn rice_partition_sums_match_scalar() {
        let mut rng = Rng(0x0BAD_CAFE_F00D_1234);
        for (n, part_len) in [
            (16usize, 16usize),
            (4096, 16),
            (4096, 4096),
            (4608, 18),
            (4608, 9),
            (1152, 72),
            (37, 37),
        ] {
            for bits in [1u32, 9, 17, 32] {
                let res: Vec<i32> = (0..n).map(|_| rng.signed(bits)).collect();
                let parts = n / part_len;
                for skip in [0usize, 1, 4, 12, 15] {
                    let skip = skip.min(part_len);
                    let (mut s0, mut o0) = (vec![7u64; parts], vec![7u32; parts]);
                    let (mut s1, mut o1) = (vec![9u64; parts], vec![9u32; parts]);
                    flac_rice_partition_sums(&res, skip, part_len, &mut s0, &mut o0);
                    flac_rice_partition_sums_scalar(&res, skip, part_len, &mut s1, &mut o1);
                    assert_eq!((s0, o0), (s1, o1), "n={n} part_len={part_len} skip={skip}");
                }
            }
        }
        // Extremal residuals fold to `u32::MAX` and `u32::MAX - 1` without overflowing the sums.
        let res = vec![
            i32::MIN,
            i32::MAX,
            -1,
            0,
            i32::MIN,
            i32::MIN,
            i32::MAX,
            1,
            i32::MIN,
        ];
        let (mut s0, mut o0) = (vec![0u64; 1], vec![0u32; 1]);
        let (mut s1, mut o1) = (vec![0u64; 1], vec![0u32; 1]);
        flac_rice_partition_sums(&res, 0, 9, &mut s0, &mut o0);
        flac_rice_partition_sums_scalar(&res, 0, 9, &mut s1, &mut o1);
        assert_eq!((s0, o0), (s1, o1));
    }
    #[test]
    fn rice_partition_costs_match_scalar() {
        let mut rng = Rng(0x5151_7777_ABCD_0001);
        for m in [1usize, 2, 4, 5, 8, 9, 31, 64, 255, 256] {
            for cnt in [1u32, 2, 9, 16, 18, 64, 4096, 65535] {
                for bits in [0u32, 1, 5, 12, 20, 32] {
                    let ors: Vec<u32> = (0..m)
                        .map(|_| {
                            if bits == 0 {
                                0
                            } else {
                                (rng.next() as u32) >> (32 - bits)
                            }
                        })
                        .collect();
                    // Sums from all-zero partitions up to every value at its OR.
                    let sums: Vec<u64> = ors
                        .iter()
                        .map(|&o| match rng.next() % 4 {
                            0 => 0,
                            1 => o as u64 * cnt as u64,
                            _ => (rng.next() % (o as u64 + 1)) * cnt as u64 / 2,
                        })
                        .collect();
                    for first in [cnt, cnt.saturating_sub(12).max(1), 1] {
                        let (mut b0, mut p0) = (vec![0u32; m], vec![0u8; m]);
                        let (mut b1, mut p1) = (vec![1u32; m], vec![1u8; m]);
                        flac_rice_partition_costs(&sums, &ors, cnt, first, &mut b0, &mut p0);
                        flac_rice_partition_costs_scalar(&sums, &ors, cnt, first, &mut b1, &mut p1);
                        assert_eq!((b0, p0), (b1, p1), "m={m} cnt={cnt} bits={bits}");
                    }
                }
            }
        }
    }
}
