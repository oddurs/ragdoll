//! SIMD YUV $\leftrightarrow$ RGB / Oklab color matrix and chroma subsampling engine.
//!
//! Modeled after `libswscale/aarch64/yuv2rgb_neon.S` and `rgb2rgb_neon.S`, upgraded with:
//! - Exact Q14 fixed-point ITU-R BT.601, BT.709, and BT.2020 matrices in both studio (`Limited`)
//!   and full (`Full`) ranges.
//! - 2-row $\times$ 16-pixel `aarch64` NEON `YUV420p` $\leftrightarrow$ `RGB24` / `RGBA8` kernels using
//!   hardware 3- and 4-element interleaving/deinterleaving (`vld3q_u8`/`vst3q_u8`, `vld4q_u8`/`vst4q_u8`).
//! - Chroma siting (`Center` vs `TopLeft`) aware 4:2:0 $\leftrightarrow$ 4:4:4 upsampling/downsampling.
//! - Perceptual `Oklab` scanline conversion for tone-mapping and perceptual delta-E analysis.

use ragdoll_core::{
    ChromaLocation, ColorRange, Colorimetry, MatrixCoefficients, PixelFormat, Plane, VideoFrame,
};
use ragdoll_simd::{Q14_HALF, Q14_ONE, clamp_u8};

/// Precomputed Q14 ($2^{14} = 16\,384$) fixed-point coefficients for YUV $\leftrightarrow$ RGB conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct YuvRgbMatrix {
    /// Luma offset subtracted before scaling (`16` for Limited range, `0` for Full range).
    pub y_offset: i32,
    /// Luma scale factor in Q14 (`19077` for Limited `255/219`, `16384` for Full).
    pub y_coeff: i32,
    /// $V \to R$ weight in Q14.
    pub v_to_r: i32,
    /// $U \to G$ weight in Q14 (negative).
    pub u_to_g: i32,
    /// $V \to G$ weight in Q14 (negative).
    pub v_to_g: i32,
    /// $U \to B$ weight in Q14.
    pub u_to_b: i32,

    /// $R \to Y$ weight in Q14.
    pub r_to_y: i32,
    /// $G \to Y$ weight in Q14.
    pub g_to_y: i32,
    /// $B \to Y$ weight in Q14.
    pub b_to_y: i32,

    /// $R \to U$ weight in Q14.
    pub r_to_u: i32,
    /// $G \to U$ weight in Q14.
    pub g_to_u: i32,
    /// $B \to U$ weight in Q14.
    pub b_to_u: i32,

    /// $R \to V$ weight in Q14.
    pub r_to_v: i32,
    /// $G \to V$ weight in Q14.
    pub g_to_v: i32,
    /// $B \to V$ weight in Q14.
    pub b_to_v: i32,
}

impl YuvRgbMatrix {
    /// Builds a Q14 fixed-point conversion matrix from ITU-T H.273 `MatrixCoefficients` and `ColorRange`.
    pub fn from_colorimetry(matrix: MatrixCoefficients, range: ColorRange) -> Self {
        let (kr, kg, kb) = matrix.luma_coefficients();
        let (y_offset, y_span, c_span) = match range {
            ColorRange::Limited => (16, 219.0, 224.0),
            ColorRange::Full => (0, 255.0, 255.0),
        };

        let y_scale = 255.0 / y_span;
        let c_scale = 255.0 / c_span;

        let v_to_r_f = 2.0 * (1.0 - kr) * c_scale;
        let u_to_b_f = 2.0 * (1.0 - kb) * c_scale;
        let u_to_g_f = -2.0 * kb * (1.0 - kb) / kg * c_scale;
        let v_to_g_f = -2.0 * kr * (1.0 - kr) / kg * c_scale;

        let q14 = Q14_ONE as f64;
        let y_coeff = (y_scale * q14).round() as i32;
        let v_to_r = (v_to_r_f * q14).round() as i32;
        let u_to_g = (u_to_g_f * q14).round() as i32;
        let v_to_g = (v_to_g_f * q14).round() as i32;
        let u_to_b = (u_to_b_f * q14).round() as i32;

        // Forward RGB -> YUV weights (exact sum preservation)
        let y_inv = y_span / 255.0;
        let c_inv = c_span / 255.0;
        let r_to_y = (kr * y_inv * q14).round() as i32;
        let b_to_y = (kb * y_inv * q14).round() as i32;
        let g_to_y = ((y_inv * q14).round() as i32) - r_to_y - b_to_y;

        let b_to_u = (0.5 * c_inv * q14).round() as i32;
        let r_to_u = (-0.5 * kr / (1.0 - kb) * c_inv * q14).round() as i32;
        let g_to_u = -b_to_u - r_to_u; // Ensures U(R=G=B) == 128 exactly!

        let r_to_v = (0.5 * c_inv * q14).round() as i32;
        let b_to_v = (-0.5 * kb / (1.0 - kr) * c_inv * q14).round() as i32;
        let g_to_v = -r_to_v - b_to_v; // Ensures V(R=G=B) == 128 exactly!

        Self {
            y_offset,
            y_coeff,
            v_to_r,
            u_to_g,
            v_to_g,
            u_to_b,
            r_to_y,
            g_to_y,
            b_to_y,
            r_to_u,
            g_to_u,
            b_to_u,
            r_to_v,
            g_to_v,
            b_to_v,
        }
    }

    #[inline(always)]
    pub const fn yuv_to_rgb_pixel(&self, y: u8, u: u8, v: u8) -> (u8, u8, u8) {
        let y_adj = (y as i32) - self.y_offset;
        let u_adj = (u as i32) - 128;
        let v_adj = (v as i32) - 128;

        let y_base = y_adj * self.y_coeff + Q14_HALF;
        let r = clamp_u8((y_base + v_adj * self.v_to_r) >> 14);
        let g = clamp_u8((y_base + u_adj * self.u_to_g + v_adj * self.v_to_g) >> 14);
        let b = clamp_u8((y_base + u_adj * self.u_to_b) >> 14);
        (r, g, b)
    }

    #[inline(always)]
    pub const fn rgb_to_yuv_pixel(&self, r: u8, g: u8, b: u8) -> (u8, u8, u8) {
        let ri = r as i32;
        let gi = g as i32;
        let bi = b as i32;

        let y = clamp_u8(
            ((ri * self.r_to_y + gi * self.g_to_y + bi * self.b_to_y + Q14_HALF) >> 14)
                + self.y_offset,
        );
        let u = clamp_u8(
            ((ri * self.r_to_u + gi * self.g_to_u + bi * self.b_to_u + Q14_HALF) >> 14) + 128,
        );
        let v = clamp_u8(
            ((ri * self.r_to_v + gi * self.g_to_v + bi * self.b_to_v + Q14_HALF) >> 14) + 128,
        );
        (y, u, v)
    }
}

/// Converts a planar `YUV420p` image into packed `RGB24` using the given `YuvRgbMatrix`.
///
/// Dispatches to a 2-row $\times$ 16-pixel `aarch64` NEON `vst3q_u8` kernel on Apple Silicon.
#[allow(clippy::too_many_arguments)]
pub fn yuv420p_to_rgb24(
    y_plane: &[u8],
    y_stride: usize,
    u_plane: &[u8],
    u_stride: usize,
    v_plane: &[u8],
    v_stride: usize,
    rgb_dst: &mut [u8],
    rgb_stride: usize,
    width: usize,
    height: usize,
    mat: &YuvRgbMatrix,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            yuv420p_to_rgb24_neon(
                y_plane, y_stride, u_plane, u_stride, v_plane, v_stride, rgb_dst, rgb_stride,
                width, height, mat,
            );
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                yuv420p_to_rgb24_avx2(
                    y_plane, y_stride, u_plane, u_stride, v_plane, v_stride, rgb_dst, rgb_stride,
                    width, height, mat,
                );
            }
        } else {
            yuv420p_to_rgb24_scalar(
                y_plane, y_stride, u_plane, u_stride, v_plane, v_stride, rgb_dst, rgb_stride,
                width, height, mat,
            );
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        yuv420p_to_rgb24_scalar(
            y_plane, y_stride, u_plane, u_stride, v_plane, v_stride, rgb_dst, rgb_stride, width,
            height, mat,
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn yuv420p_to_rgb24_avx2(
    y_plane: &[u8],
    y_stride: usize,
    u_plane: &[u8],
    u_stride: usize,
    v_plane: &[u8],
    v_stride: usize,
    rgb_dst: &mut [u8],
    rgb_stride: usize,
    width: usize,
    height: usize,
    mat: &YuvRgbMatrix,
) {
    use std::arch::x86_64::*;
    let v_y_off = _mm256_set1_epi32(mat.y_offset);
    let v_y_c = _mm256_set1_epi32(mat.y_coeff);
    let v_bias = _mm256_set1_epi32(Q14_HALF);
    for y in 0..height {
        let y_row = &y_plane[y * y_stride..y * y_stride + width];
        let cy = y >> 1;
        let u_row = &u_plane[cy * u_stride..];
        let v_row = &v_plane[cy * v_stride..];
        let dst_row = &mut rgb_dst[y * rgb_stride..y * rgb_stride + width * 3];
        let mut x = 0usize;
        while x + 8 <= width {
            let y8 = unsafe { _mm_loadl_epi64(y_row.as_ptr().add(x) as *const __m128i) };
            let y32 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(y8), v_y_off);
            let y_base = _mm256_add_epi32(_mm256_mullo_epi32(y32, v_y_c), v_bias);
            let mut yb = [0i32; 8];
            unsafe { _mm256_storeu_si256(yb.as_mut_ptr() as *mut __m256i, y_base) };
            for k in 0..8 {
                let px = x + k;
                let u_adj = (u_row[px >> 1] as i32) - 128;
                let v_adj = (v_row[px >> 1] as i32) - 128;
                let d = px * 3;
                dst_row[d] = clamp_u8((yb[k] + v_adj * mat.v_to_r) >> 14);
                dst_row[d + 1] = clamp_u8((yb[k] + u_adj * mat.u_to_g + v_adj * mat.v_to_g) >> 14);
                dst_row[d + 2] = clamp_u8((yb[k] + u_adj * mat.u_to_b) >> 14);
            }
            x += 8;
        }
        while x < width {
            let (r, g, b) = mat.yuv_to_rgb_pixel(y_row[x], u_row[x >> 1], v_row[x >> 1]);
            let d = x * 3;
            dst_row[d] = r;
            dst_row[d + 1] = g;
            dst_row[d + 2] = b;
            x += 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn yuv420p_to_rgb24_scalar(
    y_plane: &[u8],
    y_stride: usize,
    u_plane: &[u8],
    u_stride: usize,
    v_plane: &[u8],
    v_stride: usize,
    rgb_dst: &mut [u8],
    rgb_stride: usize,
    width: usize,
    height: usize,
    mat: &YuvRgbMatrix,
) {
    for y in 0..height {
        let y_row = &y_plane[y * y_stride..y * y_stride + width];
        let cy = y >> 1;
        let u_row = &u_plane[cy * u_stride..];
        let v_row = &v_plane[cy * v_stride..];
        let dst_row = &mut rgb_dst[y * rgb_stride..y * rgb_stride + width * 3];

        for x in 0..width {
            let (r, g, b) = mat.yuv_to_rgb_pixel(y_row[x], u_row[x >> 1], v_row[x >> 1]);
            let d = x * 3;
            dst_row[d] = r;
            dst_row[d + 1] = g;
            dst_row[d + 2] = b;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn yuv420p_to_rgb24_neon(
    y_plane: &[u8],
    y_stride: usize,
    u_plane: &[u8],
    u_stride: usize,
    v_plane: &[u8],
    v_stride: usize,
    rgb_dst: &mut [u8],
    rgb_stride: usize,
    width: usize,
    height: usize,
    mat: &YuvRgbMatrix,
) {
    use std::arch::aarch64::*;
    let v_y_off = vdupq_n_s16(mat.y_offset as i16);
    let v_128 = vdupq_n_s16(128);
    let v_bias = vdupq_n_s32(Q14_HALF);

    let mut y = 0usize;
    // Process 2 rows at a time so chroma (U, V) is loaded and multiplied only once per 2 luma rows
    while y + 2 <= height {
        let cy = y >> 1;
        let y_row0 = &y_plane[y * y_stride..y * y_stride + width];
        let y_row1 = &y_plane[(y + 1) * y_stride..(y + 1) * y_stride + width];
        let u_row = &u_plane[cy * u_stride..];
        let v_row = &v_plane[cy * v_stride..];

        let (top_half, bottom_half) = rgb_dst.split_at_mut((y + 1) * rgb_stride);
        let dst0 = &mut top_half[y * rgb_stride..y * rgb_stride + width * 3];
        let dst1 = &mut bottom_half[..width * 3];

        let mut x = 0usize;
        while x + 16 <= width {
            let cx = x >> 1;
            let u8_vec = unsafe { vld1_u8(u_plane.as_ptr().add(cy * u_stride + cx)) };
            let v8_vec = unsafe { vld1_u8(v_plane.as_ptr().add(cy * v_stride + cx)) };

            let u16_adj = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u8_vec)), v_128);
            let v16_adj = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v8_vec)), v_128);

            // Compute 8 chroma R, G, B offsets in 32-bit Q14 (using vmlaq_n_s32 so coefficients > 32767 like BT.601 u_to_b=33050 never overflow i16)
            let u_lo = vmovl_s16(vget_low_s16(u16_adj));
            let u_hi = vmovl_high_s16(u16_adj);
            let v_lo = vmovl_s16(vget_low_s16(v16_adj));
            let v_hi = vmovl_high_s16(v16_adj);

            let cr_lo = vmlaq_n_s32(v_bias, v_lo, mat.v_to_r);
            let cr_hi = vmlaq_n_s32(v_bias, v_hi, mat.v_to_r);

            let cg_lo = vmlaq_n_s32(vmlaq_n_s32(v_bias, u_lo, mat.u_to_g), v_lo, mat.v_to_g);
            let cg_hi = vmlaq_n_s32(vmlaq_n_s32(v_bias, u_hi, mat.u_to_g), v_hi, mat.v_to_g);

            let cb_lo = vmlaq_n_s32(v_bias, u_lo, mat.u_to_b);
            let cb_hi = vmlaq_n_s32(v_bias, u_hi, mat.u_to_b);

            // Duplicate each chroma sample horizontally ([c0, c1, c2, c3] -> [c0, c0, c1, c1], [c2, c2, c3, c3]) via vzip1q_s32 / vzip2q_s32
            let cr_0 = vzip1q_s32(cr_lo, cr_lo);
            let cr_1 = vzip2q_s32(cr_lo, cr_lo);
            let cr_2 = vzip1q_s32(cr_hi, cr_hi);
            let cr_3 = vzip2q_s32(cr_hi, cr_hi);

            let cg_0 = vzip1q_s32(cg_lo, cg_lo);
            let cg_1 = vzip2q_s32(cg_lo, cg_lo);
            let cg_2 = vzip1q_s32(cg_hi, cg_hi);
            let cg_3 = vzip2q_s32(cg_hi, cg_hi);

            let cb_0 = vzip1q_s32(cb_lo, cb_lo);
            let cb_1 = vzip2q_s32(cb_lo, cb_lo);
            let cb_2 = vzip1q_s32(cb_hi, cb_hi);
            let cb_3 = vzip2q_s32(cb_hi, cb_hi);

            // Helper closure to convert one 16-pixel luma row and store 48 RGB bytes via vst3q_u8
            let emit_row = |y_ptr: *const u8, dst_ptr: *mut u8| unsafe {
                let y_raw = vld1q_u8(y_ptr);
                let y_lo = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_raw))), v_y_off);
                let y_hi = vsubq_s16(vreinterpretq_s16_u16(vmovl_high_u8(y_raw)), v_y_off);

                let y0 = vmull_n_s16(vget_low_s16(y_lo), mat.y_coeff as i16);
                let y1 = vmull_high_n_s16(y_lo, mat.y_coeff as i16);
                let y2 = vmull_n_s16(vget_low_s16(y_hi), mat.y_coeff as i16);
                let y3 = vmull_high_n_s16(y_hi, mat.y_coeff as i16);

                let r01 = vshrn_high_n_s32::<14>(
                    vshrn_n_s32::<14>(vaddq_s32(y0, cr_0)),
                    vaddq_s32(y1, cr_1),
                );
                let r23 = vshrn_high_n_s32::<14>(
                    vshrn_n_s32::<14>(vaddq_s32(y2, cr_2)),
                    vaddq_s32(y3, cr_3),
                );
                let g01 = vshrn_high_n_s32::<14>(
                    vshrn_n_s32::<14>(vaddq_s32(y0, cg_0)),
                    vaddq_s32(y1, cg_1),
                );
                let g23 = vshrn_high_n_s32::<14>(
                    vshrn_n_s32::<14>(vaddq_s32(y2, cg_2)),
                    vaddq_s32(y3, cg_3),
                );
                let b01 = vshrn_high_n_s32::<14>(
                    vshrn_n_s32::<14>(vaddq_s32(y0, cb_0)),
                    vaddq_s32(y1, cb_1),
                );
                let b23 = vshrn_high_n_s32::<14>(
                    vshrn_n_s32::<14>(vaddq_s32(y2, cb_2)),
                    vaddq_s32(y3, cb_3),
                );

                let r8 = vqmovun_high_s16(vqmovun_s16(r01), r23);
                let g8 = vqmovun_high_s16(vqmovun_s16(g01), g23);
                let b8 = vqmovun_high_s16(vqmovun_s16(b01), b23);

                vst3q_u8(dst_ptr, uint8x16x3_t(r8, g8, b8));
            };

            unsafe {
                emit_row(y_row0.as_ptr().add(x), dst0.as_mut_ptr().add(x * 3));
                emit_row(y_row1.as_ptr().add(x), dst1.as_mut_ptr().add(x * 3));
            }
            x += 16;
        }

        while x < width {
            let u = u_row[x >> 1];
            let v = v_row[x >> 1];
            let (r0, g0, b0) = mat.yuv_to_rgb_pixel(y_row0[x], u, v);
            let (r1, g1, b1) = mat.yuv_to_rgb_pixel(y_row1[x], u, v);
            let d = x * 3;
            dst0[d] = r0;
            dst0[d + 1] = g0;
            dst0[d + 2] = b0;
            dst1[d] = r1;
            dst1[d + 1] = g1;
            dst1[d + 2] = b1;
            x += 1;
        }

        y += 2;
    }

    if y < height {
        yuv420p_to_rgb24_scalar(
            &y_plane[y * y_stride..],
            y_stride,
            &u_plane[(y >> 1) * u_stride..],
            u_stride,
            &v_plane[(y >> 1) * v_stride..],
            v_stride,
            &mut rgb_dst[y * rgb_stride..],
            rgb_stride,
            width,
            height - y,
            mat,
        );
    }
}

/// Converts a planar `YUV420p` image into packed `RGBA8` (`alpha = 255`).
///
/// Dispatches to a 2-row $\times$ 16-pixel `aarch64` NEON `vst4q_u8` kernel on Apple Silicon
/// (matching `libswscale/aarch64/yuv2rgb_neon.S`).
#[allow(clippy::too_many_arguments)]
pub fn yuv420p_to_rgba8(
    y_plane: &[u8],
    y_stride: usize,
    u_plane: &[u8],
    u_stride: usize,
    v_plane: &[u8],
    v_stride: usize,
    rgba_dst: &mut [u8],
    rgba_stride: usize,
    width: usize,
    height: usize,
    mat: &YuvRgbMatrix,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            yuv420p_to_rgba_or_bgra_neon::<false>(
                y_plane,
                y_stride,
                u_plane,
                u_stride,
                v_plane,
                v_stride,
                rgba_dst,
                rgba_stride,
                width,
                height,
                mat,
            );
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        yuv420p_to_rgba8_scalar(
            y_plane,
            y_stride,
            u_plane,
            u_stride,
            v_plane,
            v_stride,
            rgba_dst,
            rgba_stride,
            width,
            height,
            mat,
        );
    }
}

/// Converts a planar `YUV420p` image into packed `BGRA8` (`alpha = 255`).
#[allow(clippy::too_many_arguments)]
pub fn yuv420p_to_bgra8(
    y_plane: &[u8],
    y_stride: usize,
    u_plane: &[u8],
    u_stride: usize,
    v_plane: &[u8],
    v_stride: usize,
    bgra_dst: &mut [u8],
    bgra_stride: usize,
    width: usize,
    height: usize,
    mat: &YuvRgbMatrix,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            yuv420p_to_rgba_or_bgra_neon::<true>(
                y_plane,
                y_stride,
                u_plane,
                u_stride,
                v_plane,
                v_stride,
                bgra_dst,
                bgra_stride,
                width,
                height,
                mat,
            );
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        yuv420p_to_bgra8_scalar(
            y_plane,
            y_stride,
            u_plane,
            u_stride,
            v_plane,
            v_stride,
            bgra_dst,
            bgra_stride,
            width,
            height,
            mat,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn yuv420p_to_rgba8_scalar(
    y_plane: &[u8],
    y_stride: usize,
    u_plane: &[u8],
    u_stride: usize,
    v_plane: &[u8],
    v_stride: usize,
    rgba_dst: &mut [u8],
    rgba_stride: usize,
    width: usize,
    height: usize,
    mat: &YuvRgbMatrix,
) {
    for y in 0..height {
        let y_row = &y_plane[y * y_stride..y * y_stride + width];
        let cy = y >> 1;
        let u_row = &u_plane[cy * u_stride..];
        let v_row = &v_plane[cy * v_stride..];
        let dst_row = &mut rgba_dst[y * rgba_stride..y * rgba_stride + width * 4];
        for x in 0..width {
            let (r, g, b) = mat.yuv_to_rgb_pixel(y_row[x], u_row[x >> 1], v_row[x >> 1]);
            let d = x * 4;
            dst_row[d] = r;
            dst_row[d + 1] = g;
            dst_row[d + 2] = b;
            dst_row[d + 3] = 255;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn yuv420p_to_bgra8_scalar(
    y_plane: &[u8],
    y_stride: usize,
    u_plane: &[u8],
    u_stride: usize,
    v_plane: &[u8],
    v_stride: usize,
    bgra_dst: &mut [u8],
    bgra_stride: usize,
    width: usize,
    height: usize,
    mat: &YuvRgbMatrix,
) {
    for y in 0..height {
        let y_row = &y_plane[y * y_stride..y * y_stride + width];
        let cy = y >> 1;
        let u_row = &u_plane[cy * u_stride..];
        let v_row = &v_plane[cy * v_stride..];
        let dst_row = &mut bgra_dst[y * bgra_stride..y * bgra_stride + width * 4];
        for x in 0..width {
            let (r, g, b) = mat.yuv_to_rgb_pixel(y_row[x], u_row[x >> 1], v_row[x >> 1]);
            let d = x * 4;
            dst_row[d] = b;
            dst_row[d + 1] = g;
            dst_row[d + 2] = r;
            dst_row[d + 3] = 255;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn yuv420p_to_rgba_or_bgra_neon<const BGRA: bool>(
    y_plane: &[u8],
    y_stride: usize,
    u_plane: &[u8],
    u_stride: usize,
    v_plane: &[u8],
    v_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    width: usize,
    height: usize,
    mat: &YuvRgbMatrix,
) {
    use std::arch::aarch64::*;

    let v_y_off = vdupq_n_s16(mat.y_offset as i16);
    let v_128 = vdupq_n_s16(128);
    let v_bias = vdupq_n_s32(Q14_HALF);
    let v_alpha = vdupq_n_u8(255);

    let mut y = 0usize;
    while y + 2 <= height {
        let cy = y >> 1;
        let y_row0 = &y_plane[y * y_stride..y * y_stride + width];
        let y_row1 = &y_plane[(y + 1) * y_stride..(y + 1) * y_stride + width];
        let u_row = &u_plane[cy * u_stride..];
        let v_row = &v_plane[cy * v_stride..];

        let (top_half, bot_half) = dst.split_at_mut((y + 1) * dst_stride);
        let dst0 = &mut top_half[y * dst_stride..y * dst_stride + width * 4];
        let dst1 = &mut bot_half[..width * 4];

        let mut x = 0usize;
        while x + 16 <= width {
            let cx = x >> 1;
            let u8_vec = unsafe { vld1_u8(u_row.as_ptr().add(cx)) };
            let v8_vec = unsafe { vld1_u8(v_row.as_ptr().add(cx)) };

            let u16_adj = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u8_vec)), v_128);
            let v16_adj = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v8_vec)), v_128);

            let u_lo = vmovl_s16(vget_low_s16(u16_adj));
            let u_hi = vmovl_high_s16(u16_adj);
            let v_lo = vmovl_s16(vget_low_s16(v16_adj));
            let v_hi = vmovl_high_s16(v16_adj);

            let cr_lo = vmlaq_n_s32(v_bias, v_lo, mat.v_to_r);
            let cr_hi = vmlaq_n_s32(v_bias, v_hi, mat.v_to_r);
            let cg_lo = vmlaq_n_s32(vmlaq_n_s32(v_bias, u_lo, mat.u_to_g), v_lo, mat.v_to_g);
            let cg_hi = vmlaq_n_s32(vmlaq_n_s32(v_bias, u_hi, mat.u_to_g), v_hi, mat.v_to_g);
            let cb_lo = vmlaq_n_s32(v_bias, u_lo, mat.u_to_b);
            let cb_hi = vmlaq_n_s32(v_bias, u_hi, mat.u_to_b);

            let cr_0 = vzip1q_s32(cr_lo, cr_lo);
            let cr_1 = vzip2q_s32(cr_lo, cr_lo);
            let cr_2 = vzip1q_s32(cr_hi, cr_hi);
            let cr_3 = vzip2q_s32(cr_hi, cr_hi);

            let cg_0 = vzip1q_s32(cg_lo, cg_lo);
            let cg_1 = vzip2q_s32(cg_lo, cg_lo);
            let cg_2 = vzip1q_s32(cg_hi, cg_hi);
            let cg_3 = vzip2q_s32(cg_hi, cg_hi);

            let cb_0 = vzip1q_s32(cb_lo, cb_lo);
            let cb_1 = vzip2q_s32(cb_lo, cb_lo);
            let cb_2 = vzip1q_s32(cb_hi, cb_hi);
            let cb_3 = vzip2q_s32(cb_hi, cb_hi);

            let emit_row = |y_ptr: *const u8, dst_ptr: *mut u8| unsafe {
                let y_raw = vld1q_u8(y_ptr);
                let y_lo = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_raw))), v_y_off);
                let y_hi = vsubq_s16(vreinterpretq_s16_u16(vmovl_high_u8(y_raw)), v_y_off);

                let y0 = vmull_n_s16(vget_low_s16(y_lo), mat.y_coeff as i16);
                let y1 = vmull_high_n_s16(y_lo, mat.y_coeff as i16);
                let y2 = vmull_n_s16(vget_low_s16(y_hi), mat.y_coeff as i16);
                let y3 = vmull_high_n_s16(y_hi, mat.y_coeff as i16);

                let r01 = vshrn_high_n_s32::<14>(
                    vshrn_n_s32::<14>(vaddq_s32(y0, cr_0)),
                    vaddq_s32(y1, cr_1),
                );
                let r23 = vshrn_high_n_s32::<14>(
                    vshrn_n_s32::<14>(vaddq_s32(y2, cr_2)),
                    vaddq_s32(y3, cr_3),
                );
                let g01 = vshrn_high_n_s32::<14>(
                    vshrn_n_s32::<14>(vaddq_s32(y0, cg_0)),
                    vaddq_s32(y1, cg_1),
                );
                let g23 = vshrn_high_n_s32::<14>(
                    vshrn_n_s32::<14>(vaddq_s32(y2, cg_2)),
                    vaddq_s32(y3, cg_3),
                );
                let b01 = vshrn_high_n_s32::<14>(
                    vshrn_n_s32::<14>(vaddq_s32(y0, cb_0)),
                    vaddq_s32(y1, cb_1),
                );
                let b23 = vshrn_high_n_s32::<14>(
                    vshrn_n_s32::<14>(vaddq_s32(y2, cb_2)),
                    vaddq_s32(y3, cb_3),
                );

                let r8 = vqmovun_high_s16(vqmovun_s16(r01), r23);
                let g8 = vqmovun_high_s16(vqmovun_s16(g01), g23);
                let b8 = vqmovun_high_s16(vqmovun_s16(b01), b23);

                if BGRA {
                    vst4q_u8(dst_ptr, uint8x16x4_t(b8, g8, r8, v_alpha));
                } else {
                    vst4q_u8(dst_ptr, uint8x16x4_t(r8, g8, b8, v_alpha));
                }
            };

            unsafe {
                emit_row(y_row0.as_ptr().add(x), dst0.as_mut_ptr().add(x * 4));
                emit_row(y_row1.as_ptr().add(x), dst1.as_mut_ptr().add(x * 4));
            }
            x += 16;
        }

        while x < width {
            let u = u_row[x >> 1];
            let v = v_row[x >> 1];
            let (r0, g0, b0) = mat.yuv_to_rgb_pixel(y_row0[x], u, v);
            let (r1, g1, b1) = mat.yuv_to_rgb_pixel(y_row1[x], u, v);
            let d = x * 4;
            if BGRA {
                dst0[d] = b0;
                dst0[d + 1] = g0;
                dst0[d + 2] = r0;
                dst0[d + 3] = 255;
                dst1[d] = b1;
                dst1[d + 1] = g1;
                dst1[d + 2] = r1;
                dst1[d + 3] = 255;
            } else {
                dst0[d] = r0;
                dst0[d + 1] = g0;
                dst0[d + 2] = b0;
                dst0[d + 3] = 255;
                dst1[d] = r1;
                dst1[d + 1] = g1;
                dst1[d + 2] = b1;
                dst1[d + 3] = 255;
            }
            x += 1;
        }

        y += 2;
    }

    if y < height {
        if BGRA {
            yuv420p_to_bgra8_scalar(
                &y_plane[y * y_stride..],
                y_stride,
                &u_plane[(y >> 1) * u_stride..],
                u_stride,
                &v_plane[(y >> 1) * v_stride..],
                v_stride,
                &mut dst[y * dst_stride..],
                dst_stride,
                width,
                height - y,
                mat,
            );
        } else {
            yuv420p_to_rgba8_scalar(
                &y_plane[y * y_stride..],
                y_stride,
                &u_plane[(y >> 1) * u_stride..],
                u_stride,
                &v_plane[(y >> 1) * v_stride..],
                v_stride,
                &mut dst[y * dst_stride..],
                dst_stride,
                width,
                height - y,
                mat,
            );
        }
    }
}

/// Converts semi-planar `NV12` (`Y` plane + interleaved `UV` plane) into packed `RGB24`.
///
/// Uses `vld2_u8` on `aarch64` NEON to deinterleave 8 `U` and 8 `V` samples in a single instruction
/// (matching `libswscale/aarch64/yuv2rgb_neon.S` `load_chroma_nv12`).
#[allow(clippy::too_many_arguments)]
pub fn nv12_to_rgb24(
    y_plane: &[u8],
    y_stride: usize,
    uv_plane: &[u8],
    uv_stride: usize,
    rgb_dst: &mut [u8],
    rgb_stride: usize,
    width: usize,
    height: usize,
    mat: &YuvRgbMatrix,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            nv12_to_rgb24_neon(
                y_plane, y_stride, uv_plane, uv_stride, rgb_dst, rgb_stride, width, height, mat,
            );
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        nv12_to_rgb24_scalar(
            y_plane, y_stride, uv_plane, uv_stride, rgb_dst, rgb_stride, width, height, mat,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn nv12_to_rgb24_scalar(
    y_plane: &[u8],
    y_stride: usize,
    uv_plane: &[u8],
    uv_stride: usize,
    rgb_dst: &mut [u8],
    rgb_stride: usize,
    width: usize,
    height: usize,
    mat: &YuvRgbMatrix,
) {
    for y in 0..height {
        let y_row = &y_plane[y * y_stride..y * y_stride + width];
        let cy = y >> 1;
        let uv_row = &uv_plane[cy * uv_stride..];
        let dst_row = &mut rgb_dst[y * rgb_stride..y * rgb_stride + width * 3];
        for (x, &y_val) in y_row.iter().enumerate() {
            let cx = (x >> 1) * 2;
            let (r, g, b) = mat.yuv_to_rgb_pixel(y_val, uv_row[cx], uv_row[cx + 1]);
            let d = x * 3;
            dst_row[d] = r;
            dst_row[d + 1] = g;
            dst_row[d + 2] = b;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn nv12_to_rgb24_neon(
    y_plane: &[u8],
    y_stride: usize,
    uv_plane: &[u8],
    uv_stride: usize,
    rgb_dst: &mut [u8],
    rgb_stride: usize,
    width: usize,
    height: usize,
    mat: &YuvRgbMatrix,
) {
    use std::arch::aarch64::*;

    let v_y_off = vdupq_n_s16(mat.y_offset as i16);
    let v_128 = vdupq_n_s16(128);
    let v_bias = vdupq_n_s32(Q14_HALF);

    let mut y = 0usize;
    while y + 2 <= height {
        let cy = y >> 1;
        let y_row0 = &y_plane[y * y_stride..y * y_stride + width];
        let y_row1 = &y_plane[(y + 1) * y_stride..(y + 1) * y_stride + width];
        let uv_row = &uv_plane[cy * uv_stride..];

        let (top_half, bot_half) = rgb_dst.split_at_mut((y + 1) * rgb_stride);
        let dst0 = &mut top_half[y * rgb_stride..y * rgb_stride + width * 3];
        let dst1 = &mut bot_half[..width * 3];

        let mut x = 0usize;
        while x + 16 <= width {
            // Single-instruction deinterleave of 8 (U, V) pairs (`ld2 {v16.8b, v17.8b}`)
            let uv8 = unsafe { vld2_u8(uv_row.as_ptr().add(x)) };
            let u16_adj = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(uv8.0)), v_128);
            let v16_adj = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(uv8.1)), v_128);

            let u_lo = vmovl_s16(vget_low_s16(u16_adj));
            let u_hi = vmovl_high_s16(u16_adj);
            let v_lo = vmovl_s16(vget_low_s16(v16_adj));
            let v_hi = vmovl_high_s16(v16_adj);

            let cr_lo = vmlaq_n_s32(v_bias, v_lo, mat.v_to_r);
            let cr_hi = vmlaq_n_s32(v_bias, v_hi, mat.v_to_r);
            let cg_lo = vmlaq_n_s32(vmlaq_n_s32(v_bias, u_lo, mat.u_to_g), v_lo, mat.v_to_g);
            let cg_hi = vmlaq_n_s32(vmlaq_n_s32(v_bias, u_hi, mat.u_to_g), v_hi, mat.v_to_g);
            let cb_lo = vmlaq_n_s32(v_bias, u_lo, mat.u_to_b);
            let cb_hi = vmlaq_n_s32(v_bias, u_hi, mat.u_to_b);

            let cr_0 = vzip1q_s32(cr_lo, cr_lo);
            let cr_1 = vzip2q_s32(cr_lo, cr_lo);
            let cr_2 = vzip1q_s32(cr_hi, cr_hi);
            let cr_3 = vzip2q_s32(cr_hi, cr_hi);

            let cg_0 = vzip1q_s32(cg_lo, cg_lo);
            let cg_1 = vzip2q_s32(cg_lo, cg_lo);
            let cg_2 = vzip1q_s32(cg_hi, cg_hi);
            let cg_3 = vzip2q_s32(cg_hi, cg_hi);

            let cb_0 = vzip1q_s32(cb_lo, cb_lo);
            let cb_1 = vzip2q_s32(cb_lo, cb_lo);
            let cb_2 = vzip1q_s32(cb_hi, cb_hi);
            let cb_3 = vzip2q_s32(cb_hi, cb_hi);

            let emit_row = |y_ptr: *const u8, dst_ptr: *mut u8| unsafe {
                let y_raw = vld1q_u8(y_ptr);
                let y_lo = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_raw))), v_y_off);
                let y_hi = vsubq_s16(vreinterpretq_s16_u16(vmovl_high_u8(y_raw)), v_y_off);

                let y0 = vmull_n_s16(vget_low_s16(y_lo), mat.y_coeff as i16);
                let y1 = vmull_high_n_s16(y_lo, mat.y_coeff as i16);
                let y2 = vmull_n_s16(vget_low_s16(y_hi), mat.y_coeff as i16);
                let y3 = vmull_high_n_s16(y_hi, mat.y_coeff as i16);

                let r01 = vshrn_high_n_s32::<14>(
                    vshrn_n_s32::<14>(vaddq_s32(y0, cr_0)),
                    vaddq_s32(y1, cr_1),
                );
                let r23 = vshrn_high_n_s32::<14>(
                    vshrn_n_s32::<14>(vaddq_s32(y2, cr_2)),
                    vaddq_s32(y3, cr_3),
                );
                let g01 = vshrn_high_n_s32::<14>(
                    vshrn_n_s32::<14>(vaddq_s32(y0, cg_0)),
                    vaddq_s32(y1, cg_1),
                );
                let g23 = vshrn_high_n_s32::<14>(
                    vshrn_n_s32::<14>(vaddq_s32(y2, cg_2)),
                    vaddq_s32(y3, cg_3),
                );
                let b01 = vshrn_high_n_s32::<14>(
                    vshrn_n_s32::<14>(vaddq_s32(y0, cb_0)),
                    vaddq_s32(y1, cb_1),
                );
                let b23 = vshrn_high_n_s32::<14>(
                    vshrn_n_s32::<14>(vaddq_s32(y2, cb_2)),
                    vaddq_s32(y3, cb_3),
                );

                let r8 = vqmovun_high_s16(vqmovun_s16(r01), r23);
                let g8 = vqmovun_high_s16(vqmovun_s16(g01), g23);
                let b8 = vqmovun_high_s16(vqmovun_s16(b01), b23);

                vst3q_u8(dst_ptr, uint8x16x3_t(r8, g8, b8));
            };

            unsafe {
                emit_row(y_row0.as_ptr().add(x), dst0.as_mut_ptr().add(x * 3));
                emit_row(y_row1.as_ptr().add(x), dst1.as_mut_ptr().add(x * 3));
            }
            x += 16;
        }

        while x < width {
            let cx = (x >> 1) * 2;
            let u = uv_row[cx];
            let v = uv_row[cx + 1];
            let (r0, g0, b0) = mat.yuv_to_rgb_pixel(y_row0[x], u, v);
            let (r1, g1, b1) = mat.yuv_to_rgb_pixel(y_row1[x], u, v);
            let d = x * 3;
            dst0[d] = r0;
            dst0[d + 1] = g0;
            dst0[d + 2] = b0;
            dst1[d] = r1;
            dst1[d + 1] = g1;
            dst1[d + 2] = b1;
            x += 1;
        }

        y += 2;
    }

    if y < height {
        nv12_to_rgb24_scalar(
            &y_plane[y * y_stride..],
            y_stride,
            &uv_plane[(y >> 1) * uv_stride..],
            uv_stride,
            &mut rgb_dst[y * rgb_stride..],
            rgb_stride,
            width,
            height - y,
            mat,
        );
    }
}

/// Converts packed `RGB24` into planar `YUV420p` with $2\times2$ box-filtered chroma downsampling.
///
/// Dispatches to a 2-row $\times$ 16-pixel `aarch64` NEON `vld3q_u8` kernel on Apple Silicon.
#[allow(clippy::too_many_arguments)]
pub fn rgb24_to_yuv420p(
    rgb_src: &[u8],
    rgb_stride: usize,
    y_plane: &mut [u8],
    y_stride: usize,
    u_plane: &mut [u8],
    u_stride: usize,
    v_plane: &mut [u8],
    v_stride: usize,
    width: usize,
    height: usize,
    mat: &YuvRgbMatrix,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            rgb24_to_yuv420p_neon(
                rgb_src, rgb_stride, y_plane, y_stride, u_plane, u_stride, v_plane, v_stride,
                width, height, mat,
            );
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        rgb24_to_yuv420p_scalar(
            rgb_src, rgb_stride, y_plane, y_stride, u_plane, u_stride, v_plane, v_stride, width,
            height, mat,
        );
    }
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn rgb24_to_yuv420p_scalar(
    rgb_src: &[u8],
    rgb_stride: usize,
    y_plane: &mut [u8],
    y_stride: usize,
    u_plane: &mut [u8],
    u_stride: usize,
    v_plane: &mut [u8],
    v_stride: usize,
    width: usize,
    height: usize,
    mat: &YuvRgbMatrix,
) {
    // 1. Luma plane
    for y in 0..height {
        let src_row = &rgb_src[y * rgb_stride..y * rgb_stride + width * 3];
        let y_row = &mut y_plane[y * y_stride..y * y_stride + width];
        for x in 0..width {
            let s = x * 3;
            let (yv, _, _) = mat.rgb_to_yuv_pixel(src_row[s], src_row[s + 1], src_row[s + 2]);
            y_row[x] = yv;
        }
    }

    // 2. 2x2 box-averaged Chroma planes
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    for cy in 0..ch {
        let y0 = cy * 2;
        let y1 = (y0 + 1).min(height - 1);
        let row0 = &rgb_src[y0 * rgb_stride..y0 * rgb_stride + width * 3];
        let row1 = &rgb_src[y1 * rgb_stride..y1 * rgb_stride + width * 3];
        let u_row = &mut u_plane[cy * u_stride..cy * u_stride + cw];
        let v_row = &mut v_plane[cy * v_stride..cy * v_stride + cw];

        for cx in 0..cw {
            let x0 = cx * 2;
            let x1 = (x0 + 1).min(width - 1);
            let s00 = x0 * 3;
            let s01 = x1 * 3;

            let r_avg =
                ((row0[s00] as u16 + row0[s01] as u16 + row1[s00] as u16 + row1[s01] as u16 + 2)
                    >> 2) as u8;
            let g_avg = ((row0[s00 + 1] as u16
                + row0[s01 + 1] as u16
                + row1[s00 + 1] as u16
                + row1[s01 + 1] as u16
                + 2)
                >> 2) as u8;
            let b_avg = ((row0[s00 + 2] as u16
                + row0[s01 + 2] as u16
                + row1[s00 + 2] as u16
                + row1[s01 + 2] as u16
                + 2)
                >> 2) as u8;

            let (_, u, v) = mat.rgb_to_yuv_pixel(r_avg, g_avg, b_avg);
            u_row[cx] = u;
            v_row[cx] = v;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn rgb24_to_yuv420p_neon(
    rgb_src: &[u8],
    rgb_stride: usize,
    y_plane: &mut [u8],
    y_stride: usize,
    u_plane: &mut [u8],
    u_stride: usize,
    v_plane: &mut [u8],
    v_stride: usize,
    width: usize,
    height: usize,
    mat: &YuvRgbMatrix,
) {
    use std::arch::aarch64::*;
    let v_y_bias = vdupq_n_s32(Q14_HALF + (mat.y_offset << 14));
    let v_uv_bias = vdupq_n_s32(Q14_HALF + (128 << 14));

    let mut y = 0usize;
    while y + 2 <= height {
        let cy = y >> 1;
        let src0 = &rgb_src[y * rgb_stride..y * rgb_stride + width * 3];
        let src1 = &rgb_src[(y + 1) * rgb_stride..(y + 1) * rgb_stride + width * 3];

        let (y_top, y_bot) = y_plane.split_at_mut((y + 1) * y_stride);
        let y_row0 = &mut y_top[y * y_stride..y * y_stride + width];
        let y_row1 = &mut y_bot[..width];

        let u_row = &mut u_plane[cy * u_stride..];
        let v_row = &mut v_plane[cy * v_stride..];

        let mut x = 0usize;
        while x + 16 <= width {
            let rgb0 = unsafe { vld3q_u8(src0.as_ptr().add(x * 3)) };
            let rgb1 = unsafe { vld3q_u8(src1.as_ptr().add(x * 3)) };

            // Helper to compute 16 luma bytes from deinterleaved uint8x16x3_t
            let compute_y16 = |rgb: uint8x16x3_t| -> uint8x16_t {
                let r_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgb.0)));
                let r_hi = vreinterpretq_s16_u16(vmovl_high_u8(rgb.0));
                let g_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgb.1)));
                let g_hi = vreinterpretq_s16_u16(vmovl_high_u8(rgb.1));
                let b_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgb.2)));
                let b_hi = vreinterpretq_s16_u16(vmovl_high_u8(rgb.2));

                let mut y0 = vmlal_n_s16(v_y_bias, vget_low_s16(r_lo), mat.r_to_y as i16);
                y0 = vmlal_n_s16(y0, vget_low_s16(g_lo), mat.g_to_y as i16);
                y0 = vmlal_n_s16(y0, vget_low_s16(b_lo), mat.b_to_y as i16);

                let mut y1 = vmlal_high_n_s16(v_y_bias, r_lo, mat.r_to_y as i16);
                y1 = vmlal_high_n_s16(y1, g_lo, mat.g_to_y as i16);
                y1 = vmlal_high_n_s16(y1, b_lo, mat.b_to_y as i16);

                let mut y2 = vmlal_n_s16(v_y_bias, vget_low_s16(r_hi), mat.r_to_y as i16);
                y2 = vmlal_n_s16(y2, vget_low_s16(g_hi), mat.g_to_y as i16);
                y2 = vmlal_n_s16(y2, vget_low_s16(b_hi), mat.b_to_y as i16);

                let mut y3 = vmlal_high_n_s16(v_y_bias, r_hi, mat.r_to_y as i16);
                y3 = vmlal_high_n_s16(y3, g_hi, mat.g_to_y as i16);
                y3 = vmlal_high_n_s16(y3, b_hi, mat.b_to_y as i16);

                let y01 = vcombine_s16(vshrn_n_s32::<14>(y0), vshrn_n_s32::<14>(y1));
                let y23 = vcombine_s16(vshrn_n_s32::<14>(y2), vshrn_n_s32::<14>(y3));
                vcombine_u8(vqmovun_s16(y01), vqmovun_s16(y23))
            };

            let y_out0 = compute_y16(rgb0);
            let y_out1 = compute_y16(rgb1);
            unsafe {
                vst1q_u8(y_row0.as_mut_ptr().add(x), y_out0);
                vst1q_u8(y_row1.as_mut_ptr().add(x), y_out1);
            }

            // 2x2 box average of R, G, B via pairwise widening add `vpaddlq_u8` + `vpadalq_u8` + `vrshrn_n_u16::<2>`
            let r_sum = vpadalq_u8(vpaddlq_u8(rgb0.0), rgb1.0);
            let g_sum = vpadalq_u8(vpaddlq_u8(rgb0.1), rgb1.1);
            let b_sum = vpadalq_u8(vpaddlq_u8(rgb0.2), rgb1.2);

            let r_avg8 = vrshrn_n_u16::<2>(r_sum);
            let g_avg8 = vrshrn_n_u16::<2>(g_sum);
            let b_avg8 = vrshrn_n_u16::<2>(b_sum);

            let r16 = vreinterpretq_s16_u16(vmovl_u8(r_avg8));
            let g16 = vreinterpretq_s16_u16(vmovl_u8(g_avg8));
            let b16 = vreinterpretq_s16_u16(vmovl_u8(b_avg8));

            let mut u_lo = vmlal_n_s16(v_uv_bias, vget_low_s16(r16), mat.r_to_u as i16);
            u_lo = vmlal_n_s16(u_lo, vget_low_s16(g16), mat.g_to_u as i16);
            u_lo = vmlal_n_s16(u_lo, vget_low_s16(b16), mat.b_to_u as i16);

            let mut u_hi = vmlal_high_n_s16(v_uv_bias, r16, mat.r_to_u as i16);
            u_hi = vmlal_high_n_s16(u_hi, g16, mat.g_to_u as i16);
            u_hi = vmlal_high_n_s16(u_hi, b16, mat.b_to_u as i16);

            let mut v_lo = vmlal_n_s16(v_uv_bias, vget_low_s16(r16), mat.r_to_v as i16);
            v_lo = vmlal_n_s16(v_lo, vget_low_s16(g16), mat.g_to_v as i16);
            v_lo = vmlal_n_s16(v_lo, vget_low_s16(b16), mat.b_to_v as i16);

            let mut v_hi = vmlal_high_n_s16(v_uv_bias, r16, mat.r_to_v as i16);
            v_hi = vmlal_high_n_s16(v_hi, g16, mat.g_to_v as i16);
            v_hi = vmlal_high_n_s16(v_hi, b16, mat.b_to_v as i16);

            let u16_out = vcombine_s16(vshrn_n_s32::<14>(u_lo), vshrn_n_s32::<14>(u_hi));
            let v16_out = vcombine_s16(vshrn_n_s32::<14>(v_lo), vshrn_n_s32::<14>(v_hi));

            let cx = x >> 1;
            unsafe {
                vst1_u8(u_row.as_mut_ptr().add(cx), vqmovun_s16(u16_out));
                vst1_u8(v_row.as_mut_ptr().add(cx), vqmovun_s16(v16_out));
            }

            x += 16;
        }

        // Scalar tail for odd widths not a multiple of 16
        if x < width {
            for tx in x..width {
                let s0 = tx * 3;
                let (yv0, _, _) = mat.rgb_to_yuv_pixel(src0[s0], src0[s0 + 1], src0[s0 + 2]);
                let (yv1, _, _) = mat.rgb_to_yuv_pixel(src1[s0], src1[s0 + 1], src1[s0 + 2]);
                y_row0[tx] = yv0;
                y_row1[tx] = yv1;
            }
            let cw = width.div_ceil(2);
            for cx in (x >> 1)..cw {
                let x0 = cx * 2;
                let x1 = (x0 + 1).min(width - 1);
                let s00 = x0 * 3;
                let s01 = x1 * 3;
                let r_avg = ((src0[s00] as u16
                    + src0[s01] as u16
                    + src1[s00] as u16
                    + src1[s01] as u16
                    + 2)
                    >> 2) as u8;
                let g_avg = ((src0[s00 + 1] as u16
                    + src0[s01 + 1] as u16
                    + src1[s00 + 1] as u16
                    + src1[s01 + 1] as u16
                    + 2)
                    >> 2) as u8;
                let b_avg = ((src0[s00 + 2] as u16
                    + src0[s01 + 2] as u16
                    + src1[s00 + 2] as u16
                    + src1[s01 + 2] as u16
                    + 2)
                    >> 2) as u8;
                let (_, u, v) = mat.rgb_to_yuv_pixel(r_avg, g_avg, b_avg);
                u_row[cx] = u;
                v_row[cx] = v;
            }
        }

        y += 2;
    }

    if y < height {
        rgb24_to_yuv420p_scalar(
            &rgb_src[y * rgb_stride..],
            rgb_stride,
            &mut y_plane[y * y_stride..],
            y_stride,
            &mut u_plane[(y >> 1) * u_stride..],
            u_stride,
            &mut v_plane[(y >> 1) * v_stride..],
            v_stride,
            width,
            height - y,
            mat,
        );
    }
}

/// Upsamples a 4:2:0 chroma plane (`cw x ch`) to full 4:4:4 resolution (`width x height`)
/// using bilinear interpolation respecting `ChromaLocation` (`Center` vs `TopLeft`).
#[allow(clippy::needless_range_loop)]
pub fn upsample_chroma_420_to_444(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    width: usize,
    height: usize,
    loc: ChromaLocation,
) {
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);

    for y in 0..height {
        let cy0 = (y >> 1).min(ch - 1);
        let cy1 = ((y + 1) >> 1).min(ch - 1);
        let row0 = &src[cy0 * src_stride..cy0 * src_stride + cw];
        let row1 = &src[cy1 * src_stride..cy1 * src_stride + cw];
        let out_row = &mut dst[y * dst_stride..y * dst_stride + width];

        for x in 0..width {
            let cx0 = (x >> 1).min(cw - 1);
            let cx1 = ((x + 1) >> 1).min(cw - 1);
            out_row[x] = match loc {
                ChromaLocation::TopLeft | ChromaLocation::Left | ChromaLocation::BottomLeft => {
                    // Co-sited at even pixels (2k); odd pixels (2k+1) are 50/50 midpoint
                    let p00 = row0[cx0] as u16;
                    let p01 = row0[cx1] as u16;
                    let p10 = row1[cx0] as u16;
                    let p11 = row1[cx1] as u16;
                    let h0 = if (x & 1) == 0 {
                        p00
                    } else {
                        (p00 + p01 + 1) >> 1
                    };
                    let h1 = if (x & 1) == 0 {
                        p10
                    } else {
                        (p10 + p11 + 1) >> 1
                    };
                    if (y & 1) == 0 {
                        h0 as u8
                    } else {
                        ((h0 + h1 + 1) >> 1) as u8
                    }
                }
                ChromaLocation::Center | ChromaLocation::Top | ChromaLocation::Bottom => {
                    // 3:1 / 1:3 bilinear weights for centered MPEG-1/JPEG chroma siting
                    let cx_near = (x >> 1).min(cw - 1);
                    let cx_far = if (x & 1) == 0 {
                        cx_near.saturating_sub(1)
                    } else {
                        (cx_near + 1).min(cw - 1)
                    };
                    let cy_near = (y >> 1).min(ch - 1);
                    let cy_far = if (y & 1) == 0 {
                        cy_near.saturating_sub(1)
                    } else {
                        (cy_near + 1).min(ch - 1)
                    };
                    let r_near = &src[cy_near * src_stride..cy_near * src_stride + cw];
                    let r_far = &src[cy_far * src_stride..cy_far * src_stride + cw];
                    let v = 9 * (r_near[cx_near] as u16)
                        + 3 * (r_near[cx_far] as u16)
                        + 3 * (r_far[cx_near] as u16)
                        + (r_far[cx_far] as u16);
                    ((v + 8) >> 4) as u8
                }
            };
        }
    }
}

/// Perceptual Oklab color coordinate ($L \in [0, 1]$, $a, b \in [-0.4, 0.4]$)
/// using Björn Ottosson's LMS cube-root cone response model.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct OklabColor {
    pub l: f32,
    pub a: f32,
    pub b: f32,
}

impl OklabColor {
    #[inline]
    fn srgb_to_linear(c: u8) -> f32 {
        let u = (c as f32) / 255.0;
        if u <= 0.04045 {
            u / 12.92
        } else {
            ((u + 0.055) / 1.055).powf(2.4)
        }
    }

    #[inline]
    fn linear_to_srgb(u: f32) -> u8 {
        let v = u.clamp(0.0, 1.0);
        let s = if v <= 0.003_130_8 {
            12.92 * v
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0 + 0.5) as u8
    }

    #[inline]
    pub fn from_srgb8(r: u8, g: u8, b: u8) -> Self {
        let lr = Self::srgb_to_linear(r);
        let lg = Self::srgb_to_linear(g);
        let lb = Self::srgb_to_linear(b);

        let l = (0.412_221_46 * lr + 0.536_332_55 * lg + 0.051_445_995 * lb).cbrt();
        let m = (0.211_903_5 * lr + 0.680_699_5 * lg + 0.107_396_96 * lb).cbrt();
        let s = (0.088_302_46 * lr + 0.281_718_85 * lg + 0.629_978_7 * lb).cbrt();

        Self {
            l: 0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
            a: 1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
            b: 0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
        }
    }

    #[inline]
    pub fn to_srgb8(self) -> (u8, u8, u8) {
        let l_ = self.l + 0.396_337_78 * self.a + 0.215_803_76 * self.b;
        let m_ = self.l - 0.105_561_346 * self.a - 0.063_854_17 * self.b;
        let s_ = self.l - 0.089_484_18 * self.a - 1.291_485_5 * self.b;

        let l = l_ * l_ * l_;
        let m = m_ * m_ * m_;
        let s = s_ * s_ * s_;

        let lr = 4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s;
        let lg = -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s;
        let lb = -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s;

        (
            Self::linear_to_srgb(lr),
            Self::linear_to_srgb(lg),
            Self::linear_to_srgb(lb),
        )
    }

    /// Euclidean perceptual color difference $\Delta E_{\text{OK}}$ in Oklab space.
    #[inline]
    pub fn delta_e(self, other: Self) -> f32 {
        let dl = self.l - other.l;
        let da = self.a - other.a;
        let db = self.b - other.b;
        (dl * dl + da * da + db * db).sqrt()
    }
}

/// Converts a row of `RGB24` pixels into perceptual `OklabColor` (`[f32; 3]` per pixel).
pub fn rgb24_row_to_oklab(rgb_row: &[u8], dst: &mut [OklabColor]) {
    let (pixels, _) = rgb_row.as_chunks::<3>();
    for (px, out) in pixels.iter().zip(dst.iter_mut()) {
        *out = OklabColor::from_srgb8(px[0], px[1], px[2]);
    }
}

/// Converts a row of perceptual `OklabColor` values back into `RGB24` (`3` bytes per pixel).
pub fn oklab_row_to_rgb24(oklab_row: &[OklabColor], dst_rgb: &mut [u8]) {
    let (pixels, _) = dst_rgb.as_chunks_mut::<3>();
    for (ok, px) in oklab_row.iter().zip(pixels.iter_mut()) {
        let (r, g, b) = ok.to_srgb8();
        *px = [r, g, b];
    }
}

/// $16\times16$ ordered Bayer dither matrix (`0..=255`) from `zimg/src/zimg/depth/dither.cpp`.
pub const BAYER_TABLE_16X16: [[u8; 16]; 16] = [
    [
        0, 192, 48, 240, 12, 204, 60, 252, 3, 195, 51, 243, 15, 207, 63, 255,
    ],
    [
        128, 64, 176, 112, 140, 76, 188, 124, 131, 67, 179, 115, 143, 79, 191, 127,
    ],
    [
        32, 224, 16, 208, 44, 236, 28, 220, 35, 227, 19, 211, 47, 239, 31, 223,
    ],
    [
        160, 96, 144, 80, 172, 108, 156, 92, 163, 99, 147, 83, 175, 111, 159, 95,
    ],
    [
        8, 200, 56, 248, 4, 196, 52, 244, 11, 203, 59, 251, 7, 199, 55, 247,
    ],
    [
        136, 72, 184, 120, 132, 68, 180, 116, 139, 75, 187, 123, 135, 71, 183, 119,
    ],
    [
        40, 232, 24, 216, 36, 228, 20, 212, 43, 235, 27, 219, 39, 231, 23, 215,
    ],
    [
        168, 104, 152, 88, 164, 100, 148, 84, 171, 107, 155, 91, 167, 103, 151, 87,
    ],
    [
        2, 194, 50, 242, 14, 206, 62, 254, 1, 193, 49, 241, 13, 205, 61, 253,
    ],
    [
        130, 66, 178, 114, 142, 78, 190, 126, 129, 65, 177, 113, 141, 77, 189, 125,
    ],
    [
        34, 226, 18, 210, 46, 238, 30, 222, 33, 225, 17, 209, 45, 237, 29, 221,
    ],
    [
        162, 98, 146, 82, 174, 110, 158, 94, 161, 97, 145, 81, 173, 109, 157, 93,
    ],
    [
        10, 202, 58, 250, 6, 198, 54, 246, 9, 201, 57, 249, 5, 197, 53, 245,
    ],
    [
        138, 74, 186, 122, 134, 70, 182, 118, 137, 73, 185, 121, 133, 69, 181, 117,
    ],
    [
        42, 234, 26, 218, 38, 230, 22, 214, 41, 233, 25, 217, 37, 229, 21, 213,
    ],
    [
        170, 106, 154, 90, 166, 102, 150, 86, 169, 105, 153, 89, 165, 101, 149, 85,
    ],
];

/// Converts a 10-bit or 12-bit little-endian `u16` plane (`src_bits` in `9..=16`) to an 8-bit `u8`
/// plane using $16\times16$ ordered Bayer dithering (`zimg` table) to prevent Mach-band contouring.
pub fn dither_plane_u16le_to_u8_bayer(
    src_u16le: &[u8],
    src_stride_bytes: usize,
    dst_u8: &mut [u8],
    dst_stride: usize,
    width: usize,
    height: usize,
    src_bits: u32,
) {
    assert!((9..=16).contains(&src_bits));
    #[cfg(target_arch = "aarch64")]
    {
        if src_bits == 10 {
            unsafe {
                dither_plane_u10le_to_u8_bayer_neon(
                    src_u16le,
                    src_stride_bytes,
                    dst_u8,
                    dst_stride,
                    width,
                    height,
                );
            }
            return;
        }
    }
    dither_plane_u16le_to_u8_bayer_scalar(
        src_u16le,
        src_stride_bytes,
        dst_u8,
        dst_stride,
        width,
        height,
        src_bits,
    );
}

#[allow(clippy::needless_range_loop)]
pub fn dither_plane_u16le_to_u8_bayer_scalar(
    src_u16le: &[u8],
    src_stride_bytes: usize,
    dst_u8: &mut [u8],
    dst_stride: usize,
    width: usize,
    height: usize,
    src_bits: u32,
) {
    let shift = src_bits - 8;
    let max_val = (1u32 << src_bits) - 1;
    for y in 0..height {
        let bayer_row = &BAYER_TABLE_16X16[y & 15];
        let src_row = &src_u16le[y * src_stride_bytes..y * src_stride_bytes + width * 2];
        let dst_row = &mut dst_u8[y * dst_stride..y * dst_stride + width];
        for x in 0..width {
            let s =
                u16::from_le_bytes([src_row[x * 2], src_row[x * 2 + 1]]).min(max_val as u16) as u32;
            let bayer = bayer_row[x & 15] as u32;
            let dithered = ((s << 8) + (bayer << shift)) >> (shift + 8);
            dst_row[x] = dithered.min(255) as u8;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dither_plane_u10le_to_u8_bayer_neon(
    src_u16le: &[u8],
    src_stride_bytes: usize,
    dst_u8: &mut [u8],
    dst_stride: usize,
    width: usize,
    height: usize,
) {
    use std::arch::aarch64::*;
    let v_max10 = vdupq_n_u16(1023);

    for y in 0..height {
        let bayer_row = &BAYER_TABLE_16X16[y & 15];
        let v_bayer = unsafe { vld1q_u8(bayer_row.as_ptr()) };
        let bayer_lo = vmovl_u8(vget_low_u8(v_bayer));
        let bayer_hi = vmovl_high_u8(v_bayer);

        let src_row = &src_u16le[y * src_stride_bytes..y * src_stride_bytes + width * 2];
        let dst_row = &mut dst_u8[y * dst_stride..y * dst_stride + width];

        let mut x = 0usize;
        while x + 16 <= width {
            let s_lo = vminq_u16(
                unsafe { vld1q_u16(src_row.as_ptr().add(x * 2) as *const u16) },
                v_max10,
            );
            let s_hi = vminq_u16(
                unsafe { vld1q_u16(src_row.as_ptr().add((x + 8) * 2) as *const u16) },
                v_max10,
            );

            // (s << 6) + bayer -> >> 8 == ((s << 8) + (bayer << 2)) >> 10
            let q_lo = vqaddq_u16(vshlq_n_u16::<6>(s_lo), bayer_lo);
            let q_hi = vqaddq_u16(vshlq_n_u16::<6>(s_hi), bayer_hi);

            let out8 = vcombine_u8(vshrn_n_u16::<8>(q_lo), vshrn_n_u16::<8>(q_hi));
            unsafe {
                vst1q_u8(dst_row.as_mut_ptr().add(x), out8);
            }
            x += 16;
        }

        while x < width {
            let s = u16::from_le_bytes([src_row[x * 2], src_row[x * 2 + 1]]).min(1023) as u32;
            let bayer = bayer_row[x & 15] as u32;
            let dithered = ((s << 6) + bayer) >> 8;
            dst_row[x] = dithered.min(255) as u8;
            x += 1;
        }
    }
}

/// Converts a `VideoFrame` into `dst_format` and `dst_colorimetry`.
pub fn convert_frame(
    src: &VideoFrame,
    dst_format: PixelFormat,
    dst_colorimetry: Colorimetry,
) -> VideoFrame {
    let mut dst = VideoFrame::new(src.width, src.height, dst_format);
    dst.colorimetry = dst_colorimetry;
    dst.pts = src.pts;
    dst.duration = src.duration;
    dst.keyframe = src.keyframe;
    dst.pict_type = src.pict_type;
    convert_frame_into(src, &mut dst);
    dst
}

/// Zero-allocation in-place frame format and color-space converter.
pub fn convert_frame_into(src: &VideoFrame, dst: &mut VideoFrame) {
    assert_eq!(src.width, dst.width);
    assert_eq!(src.height, dst.height);
    let width = src.width;
    let height = src.height;

    match (src.format, dst.format) {
        (PixelFormat::Yuv420p, PixelFormat::Rgb24) => {
            let mat = YuvRgbMatrix::from_colorimetry(src.colorimetry.matrix, src.colorimetry.range);
            let y_p: &Plane = src.plane(0);
            let u_p: &Plane = src.plane(1);
            let v_p: &Plane = src.plane(2);
            let rgb_stride = dst.plane(0).stride();
            let rgb_buf = dst.plane_mut(0).data_mut();
            yuv420p_to_rgb24(
                y_p.data(),
                y_p.stride(),
                u_p.data(),
                u_p.stride(),
                v_p.data(),
                v_p.stride(),
                rgb_buf,
                rgb_stride,
                width,
                height,
                &mat,
            );
        }
        (PixelFormat::Yuv420p, PixelFormat::Rgba8) => {
            let mat = YuvRgbMatrix::from_colorimetry(src.colorimetry.matrix, src.colorimetry.range);
            let y_p = src.plane(0);
            let u_p = src.plane(1);
            let v_p = src.plane(2);
            let rgba_stride = dst.plane(0).stride();
            let rgba_buf = dst.plane_mut(0).data_mut();
            yuv420p_to_rgba8(
                y_p.data(),
                y_p.stride(),
                u_p.data(),
                u_p.stride(),
                v_p.data(),
                v_p.stride(),
                rgba_buf,
                rgba_stride,
                width,
                height,
                &mat,
            );
        }
        (PixelFormat::Yuv420p, PixelFormat::Bgra8) => {
            let mat = YuvRgbMatrix::from_colorimetry(src.colorimetry.matrix, src.colorimetry.range);
            let y_p = src.plane(0);
            let u_p = src.plane(1);
            let v_p = src.plane(2);
            let bgra_stride = dst.plane(0).stride();
            let bgra_buf = dst.plane_mut(0).data_mut();
            yuv420p_to_bgra8(
                y_p.data(),
                y_p.stride(),
                u_p.data(),
                u_p.stride(),
                v_p.data(),
                v_p.stride(),
                bgra_buf,
                bgra_stride,
                width,
                height,
                &mat,
            );
        }
        (PixelFormat::Nv12, PixelFormat::Rgb24) => {
            let mat = YuvRgbMatrix::from_colorimetry(src.colorimetry.matrix, src.colorimetry.range);
            let y_p = src.plane(0);
            let uv_p = src.plane(1);
            let rgb_stride = dst.plane(0).stride();
            let rgb_buf = dst.plane_mut(0).data_mut();
            nv12_to_rgb24(
                y_p.data(),
                y_p.stride(),
                uv_p.data(),
                uv_p.stride(),
                rgb_buf,
                rgb_stride,
                width,
                height,
                &mat,
            );
        }
        (PixelFormat::Yuv420p10le, PixelFormat::Yuv420p) => {
            for p in 0..3 {
                let sp = src.plane(p);
                let pw = sp.width();
                let ph = sp.height();
                let s_stride = sp.stride();
                let d_stride = dst.plane(p).stride();
                dither_plane_u16le_to_u8_bayer(
                    sp.data(),
                    s_stride,
                    dst.plane_mut(p).data_mut(),
                    d_stride,
                    pw,
                    ph,
                    10,
                );
            }
        }
        (PixelFormat::Rgb24, PixelFormat::Yuv420p) => {
            let mat = YuvRgbMatrix::from_colorimetry(dst.colorimetry.matrix, dst.colorimetry.range);
            let rgb_p = src.plane(0);
            let y_stride = dst.plane(0).stride();
            let u_stride = dst.plane(1).stride();
            let v_stride = dst.plane(2).stride();
            let [y_plane, u_plane, v_plane] = &mut dst.planes[0..3] else {
                unreachable!()
            };
            rgb24_to_yuv420p(
                rgb_p.data(),
                rgb_p.stride(),
                y_plane.data_mut(),
                y_stride,
                u_plane.data_mut(),
                u_stride,
                v_plane.data_mut(),
                v_stride,
                width,
                height,
                &mat,
            );
        }
        _ => {
            // Generic fallback through RGB24
            let mut tmp_rgb = VideoFrame::new(width, height, PixelFormat::Rgb24);
            if src.format != PixelFormat::Rgb24 {
                convert_frame_into(src, &mut tmp_rgb);
                convert_frame_into(&tmp_rgb, dst);
            }
        }
    }
}
