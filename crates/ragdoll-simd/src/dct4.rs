//! Power-of-two DCT-IV through an `M/2`-point complex FFT with every twiddle precomputed: the core
//! of the AAC-LC synthesis filterbank (`M = 1024` for long windows, `M = 128` for short ones).
//!
//! `y[n] = scale * sum_k x[k] cos(pi/M (n + 1/2)(k + 1/2))`. The pre-twiddle reads its input in
//! bit-reversed order (gathers), so the FFT is decimation in time with a natural-order result: one
//! radix-8 pass over every block of eight points, radix-2 passes for the rest, then a post-twiddle
//! that also carries `scale`. The AVX2 and NEON paths perform exactly the scalar reference's
//! operations in the same order (separate multiplies and adds, no fused multiply-add), so all three
//! agree to the bit.

use std::f64::consts::PI;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use std::mem::MaybeUninit;

/// Largest supported complex FFT (`M = 1024`).
const MAX_POINTS: usize = 512;

const R: f32 = std::f32::consts::FRAC_1_SQRT_2;
/// `exp(-2 pi i j / 8)` for `j = 0..4`: the twiddles of the last radix-2 stage inside a radix-8 block.
const W8_C: [f32; 4] = [1.0, R, 0.0, -R];
const W8_S: [f32; 4] = [0.0, -R, -1.0, -R];

/// Precomputed tables for a DCT-IV of `m` points (`16 <= m <= 1024`, a power of two).
#[derive(Clone, Debug)]
pub struct Dct4Plan {
    m: usize,
    /// Input index of each bit-reversed FFT point's real part (`2 rev(i)`) and imaginary part
    /// (`m - 1 - 2 rev(i)`).
    src_re: Vec<i32>,
    src_im: Vec<i32>,
    /// Pre-twiddle `exp(-i pi (4n + 1) / 4m)`, `n = rev(i)`.
    pre_c: Vec<f32>,
    pre_s: Vec<f32>,
    /// Radix-2 stage twiddles `exp(-i pi j / h)` at index `h + j`, for half-sizes `h >= 8`.
    tw_c: Vec<f32>,
    tw_s: Vec<f32>,
    /// Post-twiddle `scale * exp(-i pi j / m)`.
    post_c: Vec<f32>,
    post_s: Vec<f32>,
}

impl Dct4Plan {
    /// Creates a plan for `m` points whose outputs are multiplied by `scale`.
    ///
    /// # Panics
    ///
    /// When `m` is not a power of two in `16..=1024`.
    pub fn new(m: usize, scale: f64) -> Self {
        assert!(
            m.is_power_of_two() && (16..=2 * MAX_POINTS).contains(&m),
            "DCT-IV size {m}"
        );
        let k = m / 2;
        let bits = k.trailing_zeros();
        let rev = |i: usize| i.reverse_bits() >> (usize::BITS - bits);
        let (mut src_re, mut src_im) = (Vec::with_capacity(k), Vec::with_capacity(k));
        let (mut pre_c, mut pre_s) = (Vec::with_capacity(k), Vec::with_capacity(k));
        for i in 0..k {
            let n = rev(i);
            src_re.push((2 * n) as i32);
            src_im.push((m - 1 - 2 * n) as i32);
            let t = -PI * (4 * n + 1) as f64 / (4 * m) as f64;
            pre_c.push(t.cos() as f32);
            pre_s.push(t.sin() as f32);
        }
        let (mut tw_c, mut tw_s) = (vec![0.0; k], vec![0.0; k]);
        let mut h = 8;
        while h < k {
            for j in 0..h {
                let t = -PI * j as f64 / h as f64;
                tw_c[h + j] = t.cos() as f32;
                tw_s[h + j] = t.sin() as f32;
            }
            h *= 2;
        }
        let (post_c, post_s) = (0..k)
            .map(|j| {
                let t = -PI * j as f64 / m as f64;
                ((t.cos() * scale) as f32, (t.sin() * scale) as f32)
            })
            .unzip();
        Self {
            m,
            src_re,
            src_im,
            pre_c,
            pre_s,
            tw_c,
            tw_s,
            post_c,
            post_s,
        }
    }

    /// Number of points `M`.
    pub fn len(&self) -> usize {
        self.m
    }

    /// Always `false`: a plan has at least 16 points.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Transforms `x[..M]` into `y[..M]`.
    pub fn dct4(&self, x: &[f32], y: &mut [f32]) {
        assert!(x.len() >= self.m && y.len() >= self.m);
        #[cfg(target_arch = "x86_64")]
        if ragdoll_core::dispatch::avx2() {
            // Safety: avx2() is true only when the CPU supports AVX2.
            return unsafe { dct4_avx2(self, x, y) };
        }
        #[cfg(target_arch = "aarch64")]
        if ragdoll_core::dispatch::neon() {
            // Safety: NEON is baseline on every aarch64 CPU.
            return unsafe { dct4_neon(self, x, y) };
        }
        self.dct4_scalar(x, y);
    }

    /// Scalar reference for [`Self::dct4`].
    pub fn dct4_scalar(&self, x: &[f32], y: &mut [f32]) {
        assert!(x.len() >= self.m && y.len() >= self.m);
        let (m, k) = (self.m, self.m / 2);
        let mut re = [0.0f32; MAX_POINTS];
        let mut im = [0.0f32; MAX_POINTS];
        for i in 0..k {
            let a = x[self.src_re[i] as usize];
            let b = x[self.src_im[i] as usize];
            let (c, s) = (self.pre_c[i], self.pre_s[i]);
            re[i] = a * c - b * s;
            im[i] = a * s + b * c;
        }
        for (r, i) in re[..k]
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .zip(im[..k].as_chunks_mut::<8>().0)
        {
            fft8_scalar(r, i);
        }
        let mut h = 8;
        while h < k {
            for base in (0..k).step_by(2 * h) {
                for j in 0..h {
                    let (wr, wi) = (self.tw_c[h + j], self.tw_s[h + j]);
                    let (a, b) = (base + j, base + h + j);
                    let tr = re[b] * wr - im[b] * wi;
                    let ti = re[b] * wi + im[b] * wr;
                    let (ar, ai) = (re[a], im[a]);
                    re[a] = ar + tr;
                    im[a] = ai + ti;
                    re[b] = ar - tr;
                    im[b] = ai - ti;
                }
            }
            h *= 2;
        }
        for j in 0..k {
            let (c, s) = (self.post_c[j], self.post_s[j]);
            y[2 * j] = re[j] * c - im[j] * s;
            y[m - 1 - 2 * j] = -(re[j] * s + im[j] * c);
        }
    }
}

/// Windowing and overlap-add of one IMDCT block given as its DCT-IV `y` (`M` points, `M` a
/// multiple of 16). The IMDCT output `x` (`2M` samples) is, by quarter with `h = M / 2`,
/// `y[h + n]`, `-y[M - 1 - n]`, `-y[h - 1 - n]` and `-y[n]`; this computes
/// `out = x[..M] * rise + overlap` and then `overlap = x[M..] * fall`.
pub fn imdct_window(y: &[f32], rise: &[f32], fall: &[f32], overlap: &mut [f32], out: &mut [f32]) {
    let m = y.len();
    assert!(m.is_multiple_of(16) && rise.len() == m && fall.len() == m);
    assert!(overlap.len() == m && out.len() == m);
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        return unsafe { imdct_window_avx2(y, rise, fall, overlap, out) };
    }
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: NEON is baseline on every aarch64 CPU.
        return unsafe { imdct_window_neon(y, rise, fall, overlap, out) };
    }
    imdct_window_scalar(y, rise, fall, overlap, out);
}

/// Scalar reference for [`imdct_window`].
pub fn imdct_window_scalar(
    y: &[f32],
    rise: &[f32],
    fall: &[f32],
    overlap: &mut [f32],
    out: &mut [f32],
) {
    let m = y.len();
    let h = m / 2;
    assert!(rise.len() == m && fall.len() == m && overlap.len() == m && out.len() == m);
    for n in 0..h {
        out[n] = y[h + n] * rise[n] + overlap[n];
        out[h + n] = -y[m - 1 - n] * rise[h + n] + overlap[h + n];
    }
    for n in 0..h {
        overlap[n] = -y[h - 1 - n] * fall[n];
        overlap[h + n] = -y[n] * fall[h + n];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn imdct_window_avx2(
    y: &[f32],
    rise: &[f32],
    fall: &[f32],
    overlap: &mut [f32],
    out: &mut [f32],
) {
    use std::arch::x86_64::*;
    let m = y.len();
    let h = m / 2;
    let neg = _mm256_set1_ps(-0.0);
    let rev = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    let (yp, rp, fp) = (y.as_ptr(), rise.as_ptr(), fall.as_ptr());
    let (op, sp) = (overlap.as_mut_ptr(), out.as_mut_ptr());
    // Safety: the caller checked that every slice holds `m` elements, and every access is below.
    unsafe {
        let mut n = 0;
        while n < h {
            let a = _mm256_loadu_ps(yp.add(h + n));
            let o = _mm256_loadu_ps(op.add(n));
            _mm256_storeu_ps(
                sp.add(n),
                _mm256_add_ps(_mm256_mul_ps(a, _mm256_loadu_ps(rp.add(n))), o),
            );
            let b = _mm256_permutevar8x32_ps(_mm256_loadu_ps(yp.add(m - 8 - n)), rev);
            let b = _mm256_mul_ps(_mm256_xor_ps(b, neg), _mm256_loadu_ps(rp.add(h + n)));
            let o = _mm256_loadu_ps(op.add(h + n));
            _mm256_storeu_ps(sp.add(h + n), _mm256_add_ps(b, o));
            n += 8;
        }
        n = 0;
        while n < h {
            let c = _mm256_permutevar8x32_ps(_mm256_loadu_ps(yp.add(h - 8 - n)), rev);
            let c = _mm256_mul_ps(_mm256_xor_ps(c, neg), _mm256_loadu_ps(fp.add(n)));
            _mm256_storeu_ps(op.add(n), c);
            let d = _mm256_xor_ps(_mm256_loadu_ps(yp.add(n)), neg);
            _mm256_storeu_ps(
                op.add(h + n),
                _mm256_mul_ps(d, _mm256_loadu_ps(fp.add(h + n))),
            );
            n += 8;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn imdct_window_neon(
    y: &[f32],
    rise: &[f32],
    fall: &[f32],
    overlap: &mut [f32],
    out: &mut [f32],
) {
    use std::arch::aarch64::*;
    let m = y.len();
    let h = m / 2;
    let (yp, rp, fp) = (y.as_ptr(), rise.as_ptr(), fall.as_ptr());
    let (op, sp) = (overlap.as_mut_ptr(), out.as_mut_ptr());
    // Safety: the caller checked that every slice holds `m` elements, and every access is below.
    unsafe {
        let rev = |v: float32x4_t| {
            let r = vrev64q_f32(v);
            vextq_f32::<2>(r, r)
        };
        let mut n = 0;
        while n < h {
            let a = vld1q_f32(yp.add(h + n));
            let o = vld1q_f32(op.add(n));
            vst1q_f32(sp.add(n), vaddq_f32(vmulq_f32(a, vld1q_f32(rp.add(n))), o));
            let b = vnegq_f32(rev(vld1q_f32(yp.add(m - 4 - n))));
            let o = vld1q_f32(op.add(h + n));
            vst1q_f32(
                sp.add(h + n),
                vaddq_f32(vmulq_f32(b, vld1q_f32(rp.add(h + n))), o),
            );
            n += 4;
        }
        n = 0;
        while n < h {
            let c = vnegq_f32(rev(vld1q_f32(yp.add(h - 4 - n))));
            vst1q_f32(op.add(n), vmulq_f32(c, vld1q_f32(fp.add(n))));
            let d = vnegq_f32(vld1q_f32(yp.add(n)));
            vst1q_f32(op.add(h + n), vmulq_f32(d, vld1q_f32(fp.add(h + n))));
            n += 4;
        }
    }
}

/// The first three decimation-in-time stages (half-sizes 1, 2 and 4) on one block of eight points.
#[inline(always)]
fn fft8_scalar(re: &mut [f32; 8], im: &mut [f32; 8]) {
    for p in (0..8).step_by(2) {
        let (ar, ai, br, bi) = (re[p], im[p], re[p + 1], im[p + 1]);
        re[p] = ar + br;
        im[p] = ai + bi;
        re[p + 1] = ar - br;
        im[p + 1] = ai - bi;
    }
    for q in [0, 4] {
        for j in 0..2 {
            let (a, b) = (q + j, q + j + 2);
            // Twiddle 1 for j = 0, -i for j = 1.
            let (tr, ti) = if j == 0 {
                (re[b], im[b])
            } else {
                (im[b], -re[b])
            };
            let (ar, ai) = (re[a], im[a]);
            re[a] = ar + tr;
            im[a] = ai + ti;
            re[b] = ar - tr;
            im[b] = ai - ti;
        }
    }
    for j in 0..4 {
        let (c, s) = (W8_C[j], W8_S[j]);
        let tr = re[4 + j] * c - im[4 + j] * s;
        let ti = re[4 + j] * s + im[4 + j] * c;
        let (ar, ai) = (re[j], im[j]);
        re[j] = ar + tr;
        im[j] = ai + ti;
        re[4 + j] = ar - tr;
        im[4 + j] = ai - ti;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dct4_avx2(p: &Dct4Plan, x: &[f32], y: &mut [f32]) {
    use std::arch::x86_64::*;
    let k = p.m / 2;
    // The pre-twiddle writes every point before anything reads it.
    let mut re = MaybeUninit::<[f32; MAX_POINTS]>::uninit();
    let mut im = MaybeUninit::<[f32; MAX_POINTS]>::uninit();
    let (rp, ip) = (re.as_mut_ptr() as *mut f32, im.as_mut_ptr() as *mut f32);
    let neg = _mm256_set1_ps(-0.0);
    // Safety: every index is below `k` or `m`, which the caller checked against both slices;
    // gather indices are below `m <= x.len()`.
    unsafe {
        let mut i = 0;
        while i < k {
            let ia = _mm256_loadu_si256(p.src_re.as_ptr().add(i) as *const __m256i);
            let ib = _mm256_loadu_si256(p.src_im.as_ptr().add(i) as *const __m256i);
            let a = _mm256_i32gather_ps::<4>(x.as_ptr(), ia);
            let b = _mm256_i32gather_ps::<4>(x.as_ptr(), ib);
            let c = _mm256_loadu_ps(p.pre_c.as_ptr().add(i));
            let s = _mm256_loadu_ps(p.pre_s.as_ptr().add(i));
            let r = _mm256_sub_ps(_mm256_mul_ps(a, c), _mm256_mul_ps(b, s));
            let q = _mm256_add_ps(_mm256_mul_ps(a, s), _mm256_mul_ps(b, c));
            _mm256_storeu_ps(rp.add(i), r);
            _mm256_storeu_ps(ip.add(i), q);
            i += 8;
        }

        let sign1 = _mm256_setr_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
        let sign2 = _mm256_setr_ps(0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0);
        let sign4 = _mm256_setr_ps(0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0);
        let c8 = _mm256_setr_ps(0.0, 0.0, 0.0, 0.0, W8_C[0], W8_C[1], W8_C[2], W8_C[3]);
        let s8 = _mm256_setr_ps(0.0, 0.0, 0.0, 0.0, W8_S[0], W8_S[1], W8_S[2], W8_S[3]);
        let mut i = 0;
        while i < k {
            let mut r = _mm256_loadu_ps(rp.add(i));
            let mut q = _mm256_loadu_ps(ip.add(i));
            // Half-size 1: (a, b) -> (a + b, a - b) on lane pairs.
            r = _mm256_add_ps(_mm256_xor_ps(r, sign1), _mm256_permute_ps::<0xB1>(r));
            q = _mm256_add_ps(_mm256_xor_ps(q, sign1), _mm256_permute_ps::<0xB1>(q));
            // Half-size 2: lanes 3 and 7 take twiddle -i, (re, im) -> (im, -re).
            let tr = _mm256_blend_ps::<0x88>(r, q);
            let tq = _mm256_blend_ps::<0x88>(q, _mm256_xor_ps(r, neg));
            r = _mm256_add_ps(_mm256_xor_ps(tr, sign2), _mm256_permute_ps::<0x4E>(tr));
            q = _mm256_add_ps(_mm256_xor_ps(tq, sign2), _mm256_permute_ps::<0x4E>(tq));
            // Half-size 4: the upper four lanes take twiddles exp(-2 pi i j / 8).
            let hr = _mm256_sub_ps(_mm256_mul_ps(r, c8), _mm256_mul_ps(q, s8));
            let hq = _mm256_add_ps(_mm256_mul_ps(r, s8), _mm256_mul_ps(q, c8));
            let tr = _mm256_blend_ps::<0xF0>(r, hr);
            let tq = _mm256_blend_ps::<0xF0>(q, hq);
            r = _mm256_add_ps(
                _mm256_xor_ps(tr, sign4),
                _mm256_permute2f128_ps::<0x01>(tr, tr),
            );
            q = _mm256_add_ps(
                _mm256_xor_ps(tq, sign4),
                _mm256_permute2f128_ps::<0x01>(tq, tq),
            );
            _mm256_storeu_ps(rp.add(i), r);
            _mm256_storeu_ps(ip.add(i), q);
            i += 8;
        }

        // Radix-2 stages, two at a time where possible (half-sizes h and 2h on the points
        // base + j + {0, h, 2h, 3h}); each point sees the scalar reference's operations.
        macro_rules! butterfly {
            ($ar:ident, $ai:ident, $br:ident, $bi:ident, $t:expr) => {
                let wr = _mm256_loadu_ps(p.tw_c.as_ptr().add($t));
                let wi = _mm256_loadu_ps(p.tw_s.as_ptr().add($t));
                let tr = _mm256_sub_ps(_mm256_mul_ps($br, wr), _mm256_mul_ps($bi, wi));
                let ti = _mm256_add_ps(_mm256_mul_ps($br, wi), _mm256_mul_ps($bi, wr));
                ($ar, $ai, $br, $bi) = (
                    _mm256_add_ps($ar, tr),
                    _mm256_add_ps($ai, ti),
                    _mm256_sub_ps($ar, tr),
                    _mm256_sub_ps($ai, ti),
                );
            };
        }
        let mut h = 8;
        while h < k {
            let fused = 2 * h < k;
            let span = if fused { 4 * h } else { 2 * h };
            let mut base = 0;
            while base < k {
                let mut j = 0;
                while j < h {
                    let i0 = base + j;
                    let (mut r0, mut q0) =
                        (_mm256_loadu_ps(rp.add(i0)), _mm256_loadu_ps(ip.add(i0)));
                    let i1 = i0 + h;
                    let (mut r1, mut q1) =
                        (_mm256_loadu_ps(rp.add(i1)), _mm256_loadu_ps(ip.add(i1)));
                    if fused {
                        let (i2, i3) = (i0 + 2 * h, i0 + 3 * h);
                        let mut r2 = _mm256_loadu_ps(rp.add(i2));
                        let mut q2 = _mm256_loadu_ps(ip.add(i2));
                        let mut r3 = _mm256_loadu_ps(rp.add(i3));
                        let mut q3 = _mm256_loadu_ps(ip.add(i3));
                        butterfly!(r0, q0, r1, q1, h + j);
                        butterfly!(r2, q2, r3, q3, h + j);
                        butterfly!(r0, q0, r2, q2, 2 * h + j);
                        butterfly!(r1, q1, r3, q3, 3 * h + j);
                        _mm256_storeu_ps(rp.add(i2), r2);
                        _mm256_storeu_ps(ip.add(i2), q2);
                        _mm256_storeu_ps(rp.add(i3), r3);
                        _mm256_storeu_ps(ip.add(i3), q3);
                    } else {
                        butterfly!(r0, q0, r1, q1, h + j);
                    }
                    _mm256_storeu_ps(rp.add(i0), r0);
                    _mm256_storeu_ps(ip.add(i0), q0);
                    _mm256_storeu_ps(rp.add(i1), r1);
                    _mm256_storeu_ps(ip.add(i1), q1);
                    j += 8;
                }
                base += span;
            }
            h = if fused { 4 * h } else { 2 * h };
        }

        // y[2j] = ur[j] and y[2j + 1] = -ui[k - 1 - j]: blocks pair up with their mirror image.
        let rev = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        let yp = y.as_mut_ptr();
        let blocks = k / 8;
        for ba in 0..blocks.div_ceil(2) {
            let bb = blocks - 1 - ba;
            let mut ur = [_mm256_setzero_ps(); 2];
            let mut nui = [_mm256_setzero_ps(); 2];
            for (t, j) in [8 * ba, 8 * bb].into_iter().enumerate() {
                let c = _mm256_loadu_ps(p.post_c.as_ptr().add(j));
                let s = _mm256_loadu_ps(p.post_s.as_ptr().add(j));
                let zr = _mm256_loadu_ps(rp.add(j));
                let zi = _mm256_loadu_ps(ip.add(j));
                ur[t] = _mm256_sub_ps(_mm256_mul_ps(zr, c), _mm256_mul_ps(zi, s));
                let ui = _mm256_add_ps(_mm256_mul_ps(zr, s), _mm256_mul_ps(zi, c));
                nui[t] = _mm256_xor_ps(_mm256_permutevar8x32_ps(ui, rev), neg);
            }
            for (t, j) in [8 * ba, 8 * bb].into_iter().enumerate() {
                if t == 1 && bb == ba {
                    break;
                }
                let lo = _mm256_unpacklo_ps(ur[t], nui[1 - t]);
                let hi = _mm256_unpackhi_ps(ur[t], nui[1 - t]);
                _mm256_storeu_ps(yp.add(2 * j), _mm256_permute2f128_ps::<0x20>(lo, hi));
                _mm256_storeu_ps(yp.add(2 * j + 8), _mm256_permute2f128_ps::<0x31>(lo, hi));
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dct4_neon(p: &Dct4Plan, x: &[f32], y: &mut [f32]) {
    use std::arch::aarch64::*;
    let k = p.m / 2;
    // The pre-twiddle writes every point before anything reads it.
    let mut re = MaybeUninit::<[f32; MAX_POINTS]>::uninit();
    let mut im = MaybeUninit::<[f32; MAX_POINTS]>::uninit();
    let (rp, ip) = (re.as_mut_ptr() as *mut f32, im.as_mut_ptr() as *mut f32);
    // Safety: every index is below `k` or `m`, which the caller checked against both slices.
    unsafe {
        let mut i = 0;
        while i < k {
            let mut a = [0.0f32; 4];
            let mut b = [0.0f32; 4];
            for l in 0..4 {
                a[l] = *x.get_unchecked(*p.src_re.get_unchecked(i + l) as usize);
                b[l] = *x.get_unchecked(*p.src_im.get_unchecked(i + l) as usize);
            }
            let (a, b) = (vld1q_f32(a.as_ptr()), vld1q_f32(b.as_ptr()));
            let c = vld1q_f32(p.pre_c.as_ptr().add(i));
            let s = vld1q_f32(p.pre_s.as_ptr().add(i));
            vst1q_f32(rp.add(i), vsubq_f32(vmulq_f32(a, c), vmulq_f32(b, s)));
            vst1q_f32(ip.add(i), vaddq_f32(vmulq_f32(a, s), vmulq_f32(b, c)));
            i += 4;
        }

        let sign1 = vld1q_u32([0, 0x8000_0000, 0, 0x8000_0000].as_ptr());
        let sign2 = vld1q_u32([0, 0, 0x8000_0000, 0x8000_0000].as_ptr());
        let lane3 = vld1q_u32([0, 0, 0, u32::MAX].as_ptr());
        let c8 = vld1q_f32(W8_C.as_ptr());
        let s8 = vld1q_f32(W8_S.as_ptr());
        let mut i = 0;
        while i < k {
            let mut v = [
                vld1q_f32(rp.add(i)),
                vld1q_f32(rp.add(i + 4)),
                vld1q_f32(ip.add(i)),
                vld1q_f32(ip.add(i + 4)),
            ];
            // Half-size 1: (a, b) -> (a + b, a - b) on lane pairs.
            for r in v.iter_mut() {
                *r = vaddq_f32(xor_neon(*r, sign1), vrev64q_f32(*r));
            }
            // Half-size 2: lane 3 takes twiddle -i, (re, im) -> (im, -re).
            for h in 0..2 {
                let (r, q) = (v[h], v[2 + h]);
                let tr = vbslq_f32(lane3, q, r);
                let tq = vbslq_f32(lane3, vnegq_f32(r), q);
                v[h] = vaddq_f32(xor_neon(tr, sign2), vextq_f32::<2>(tr, tr));
                v[2 + h] = vaddq_f32(xor_neon(tq, sign2), vextq_f32::<2>(tq, tq));
            }
            // Half-size 4: the upper register takes twiddles exp(-2 pi i j / 8).
            let tr = vsubq_f32(vmulq_f32(v[1], c8), vmulq_f32(v[3], s8));
            let tq = vaddq_f32(vmulq_f32(v[1], s8), vmulq_f32(v[3], c8));
            vst1q_f32(rp.add(i), vaddq_f32(v[0], tr));
            vst1q_f32(rp.add(i + 4), vsubq_f32(v[0], tr));
            vst1q_f32(ip.add(i), vaddq_f32(v[2], tq));
            vst1q_f32(ip.add(i + 4), vsubq_f32(v[2], tq));
            i += 8;
        }

        // Radix-2 stages, two at a time where possible (half-sizes h and 2h on the points
        // base + j + {0, h, 2h, 3h}); each point sees the scalar reference's operations.
        macro_rules! butterfly {
            ($ar:ident, $ai:ident, $br:ident, $bi:ident, $t:expr) => {
                let wr = vld1q_f32(p.tw_c.as_ptr().add($t));
                let wi = vld1q_f32(p.tw_s.as_ptr().add($t));
                let tr = vsubq_f32(vmulq_f32($br, wr), vmulq_f32($bi, wi));
                let ti = vaddq_f32(vmulq_f32($br, wi), vmulq_f32($bi, wr));
                ($ar, $ai, $br, $bi) = (
                    vaddq_f32($ar, tr),
                    vaddq_f32($ai, ti),
                    vsubq_f32($ar, tr),
                    vsubq_f32($ai, ti),
                );
            };
        }
        let mut h = 8;
        while h < k {
            let fused = 2 * h < k;
            let span = if fused { 4 * h } else { 2 * h };
            let mut base = 0;
            while base < k {
                let mut j = 0;
                while j < h {
                    let i0 = base + j;
                    let (mut r0, mut q0) = (vld1q_f32(rp.add(i0)), vld1q_f32(ip.add(i0)));
                    let i1 = i0 + h;
                    let (mut r1, mut q1) = (vld1q_f32(rp.add(i1)), vld1q_f32(ip.add(i1)));
                    if fused {
                        let (i2, i3) = (i0 + 2 * h, i0 + 3 * h);
                        let (mut r2, mut q2) = (vld1q_f32(rp.add(i2)), vld1q_f32(ip.add(i2)));
                        let (mut r3, mut q3) = (vld1q_f32(rp.add(i3)), vld1q_f32(ip.add(i3)));
                        butterfly!(r0, q0, r1, q1, h + j);
                        butterfly!(r2, q2, r3, q3, h + j);
                        butterfly!(r0, q0, r2, q2, 2 * h + j);
                        butterfly!(r1, q1, r3, q3, 3 * h + j);
                        vst1q_f32(rp.add(i2), r2);
                        vst1q_f32(ip.add(i2), q2);
                        vst1q_f32(rp.add(i3), r3);
                        vst1q_f32(ip.add(i3), q3);
                    } else {
                        butterfly!(r0, q0, r1, q1, h + j);
                    }
                    vst1q_f32(rp.add(i0), r0);
                    vst1q_f32(ip.add(i0), q0);
                    vst1q_f32(rp.add(i1), r1);
                    vst1q_f32(ip.add(i1), q1);
                    j += 4;
                }
                base += span;
            }
            h = if fused { 4 * h } else { 2 * h };
        }

        // y[2j] = ur[j] and y[2j + 1] = -ui[k - 1 - j]: blocks pair up with their mirror image.
        let yp = y.as_mut_ptr();
        let blocks = k / 4;
        for ba in 0..blocks.div_ceil(2) {
            let bb = blocks - 1 - ba;
            let mut ur = [vdupq_n_f32(0.0); 2];
            let mut nui = [vdupq_n_f32(0.0); 2];
            for (t, j) in [4 * ba, 4 * bb].into_iter().enumerate() {
                let c = vld1q_f32(p.post_c.as_ptr().add(j));
                let s = vld1q_f32(p.post_s.as_ptr().add(j));
                let zr = vld1q_f32(rp.add(j));
                let zi = vld1q_f32(ip.add(j));
                ur[t] = vsubq_f32(vmulq_f32(zr, c), vmulq_f32(zi, s));
                let r = vrev64q_f32(vaddq_f32(vmulq_f32(zr, s), vmulq_f32(zi, c)));
                nui[t] = vnegq_f32(vextq_f32::<2>(r, r));
            }
            vst2q_f32(yp.add(8 * ba), float32x4x2_t(ur[0], nui[1]));
            if bb != ba {
                vst2q_f32(yp.add(8 * bb), float32x4x2_t(ur[1], nui[0]));
            }
        }
    }
}

/// Flips the sign of the lanes whose `mask` is `0x8000_0000`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn xor_neon(
    v: std::arch::aarch64::float32x4_t,
    mask: std::arch::aarch64::uint32x4_t,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    // Safety: NEON is baseline on every aarch64 CPU.
    unsafe { vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v), mask)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct(x: &[f32], scale: f64) -> Vec<f64> {
        let m = x.len();
        (0..m)
            .map(|n| {
                x.iter()
                    .enumerate()
                    .map(|(k, &v)| {
                        v as f64 * (PI / m as f64 * (n as f64 + 0.5) * (k as f64 + 0.5)).cos()
                    })
                    .sum::<f64>()
                    * scale
            })
            .collect()
    }

    fn input(m: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..m)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s as i32 >> 8) as f32 / 65536.0
            })
            .collect()
    }

    #[test]
    fn dct4_matches_definition() {
        for m in [16usize, 32, 64, 128, 256, 512, 1024] {
            let scale = 1.0 / m as f64;
            let plan = Dct4Plan::new(m, scale);
            let x = input(m, m as u32);
            let want = direct(&x, scale);
            let peak = want.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
            let mut y = vec![0.0f32; m];
            plan.dct4_scalar(&x, &mut y);
            for (n, (&g, &w)) in y.iter().zip(&want).enumerate() {
                assert!(
                    (g as f64 - w).abs() <= peak * 4e-6,
                    "m={m} n={n}: {g} vs {w}"
                );
            }
        }
    }

    /// Runs the SIMD path this CPU has (if any) and returns its output.
    fn simd(plan: &Dct4Plan, x: &[f32], y: &mut [f32]) -> bool {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            unsafe { dct4_avx2(plan, x, y) };
            return true;
        }
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { dct4_neon(plan, x, y) };
            return true;
        }
        #[allow(unreachable_code)]
        {
            let _ = (plan, x, y);
            false
        }
    }

    #[test]
    fn imdct_window_simd_is_bit_exact() {
        for m in [16usize, 32, 128, 1024] {
            let y = input(m, 3);
            let rise: Vec<f32> = (0..m).map(|i| (i as f32 * 0.01).sin()).collect();
            let fall: Vec<f32> = (0..m)
                .map(|i| {
                    if i % 5 == 0 {
                        0.0
                    } else {
                        1.0 - i as f32 / m as f32
                    }
                })
                .collect();
            let mut ov_want: Vec<f32> = input(m, 4).iter().map(|v| v * 1.0e20).collect();
            let mut ov_got = ov_want.clone();
            let mut want = vec![0.0f32; m];
            let mut got = vec![0.0f32; m];
            imdct_window_scalar(&y, &rise, &fall, &mut ov_want, &mut want);
            #[cfg(target_arch = "x86_64")]
            if !std::is_x86_feature_detected!("avx2") {
                return;
            }
            #[cfg(target_arch = "x86_64")]
            unsafe {
                imdct_window_avx2(&y, &rise, &fall, &mut ov_got, &mut got)
            };
            #[cfg(target_arch = "aarch64")]
            unsafe {
                imdct_window_neon(&y, &rise, &fall, &mut ov_got, &mut got)
            };
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            return;
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&want), "m={m} output");
            assert_eq!(bits(&ov_got), bits(&ov_want), "m={m} overlap");
        }
    }

    #[test]
    fn dct4_simd_is_bit_exact() {
        for m in [16usize, 32, 64, 128, 256, 512, 1024] {
            for scale in [1.0, 1.0 / (m as f64 * 32768.0), 0.3] {
                let plan = Dct4Plan::new(m, scale);
                let mut inputs = vec![
                    input(m, 7 * m as u32 + 1),
                    vec![0.0; m],
                    (0..m)
                        .map(|i| if i % 3 == 0 { -0.0 } else { 0.0 })
                        .collect(),
                    (0..m)
                        .map(|i| if i % 2 == 0 { 3.0e30 } else { -1.0e-30 })
                        .collect(),
                    (0..m).map(|i| (i as f32 * 0.37).sin() * 8.0e6).collect(),
                ];
                let mut impulse = vec![0.0; m];
                impulse[m - 1] = 1.0;
                inputs.push(impulse);
                for (t, x) in inputs.iter().enumerate() {
                    // Longer buffers than the plan must leave the tail untouched.
                    let mut want = vec![9.0f32; m + 3];
                    let mut got = vec![9.0f32; m + 3];
                    plan.dct4_scalar(x, &mut want);
                    if !simd(&plan, x, &mut got) {
                        return;
                    }
                    for n in 0..m + 3 {
                        assert_eq!(
                            got[n].to_bits(),
                            want[n].to_bits(),
                            "m={m} scale={scale} input {t} n={n}: {} vs {}",
                            got[n],
                            want[n]
                        );
                    }
                }
            }
        }
    }
}
