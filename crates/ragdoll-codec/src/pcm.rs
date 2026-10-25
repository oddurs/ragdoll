//! Vectorized PCM sample format conversion, channel interleaving/deinterleaving,
//! and lossless stereo decorrelation (`Left-Side`, `Right-Side`, `Mid-Side`).
//!
//! Includes `aarch64` NEON vector kernels (`vcvtq_f32_s32`, `vqmovn_s32`, `vst2q_s32`, `vld2q_s32`)
//! paired with bit-exact scalar reference functions for differential testing.

use ragdoll_core::{AudioFrame, RagdollError, Result, SampleFormat};

/// Normalization scale for converting signed 16-bit PCM `[-32768, 32767]` to `f32` `[-1.0, 1.0)`.
pub const S16_TO_F32_SCALE: f32 = 1.0 / 32768.0;

/// Scale for converting `f32` `[-1.0, 1.0]` to signed 16-bit PCM `[-32768, 32767]`.
pub const F32_TO_S16_SCALE: f32 = 32768.0;

/// Scalar reference: converts `s16` samples to `f32` in `[-1.0, 1.0)`.
pub fn pcm_s16_to_f32_scalar(src: &[i16], dst: &mut [f32]) {
    let n = src.len().min(dst.len());
    for i in 0..n {
        dst[i] = (src[i] as f32) * S16_TO_F32_SCALE;
    }
}

/// Converts `s16` samples to `f32` in `[-1.0, 1.0)`, using `aarch64` NEON when available.
pub fn pcm_s16_to_f32(src: &[i16], dst: &mut [f32]) {
    let n = src.len().min(dst.len());
    let mut i = 0usize;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let v_scale = vdupq_n_f32(S16_TO_F32_SCALE);
        while i + 8 <= n {
            let s16x8 = vld1q_s16(src.as_ptr().add(i));
            let lo_s32 = vmovl_s16(vget_low_s16(s16x8));
            let hi_s32 = vmovl_high_s16(s16x8);
            let lo_f32 = vmulq_f32(vcvtq_f32_s32(lo_s32), v_scale);
            let hi_f32 = vmulq_f32(vcvtq_f32_s32(hi_s32), v_scale);
            vst1q_f32(dst.as_mut_ptr().add(i), lo_f32);
            vst1q_f32(dst.as_mut_ptr().add(i + 4), hi_f32);
            i += 8;
        }
    }

    if i < n {
        pcm_s16_to_f32_scalar(&src[i..n], &mut dst[i..n]);
    }
}

/// Scalar reference: converts `f32` samples in `[-1.0, 1.0]` to saturating `i16` with round-to-nearest-away.
pub fn pcm_f32_to_s16_scalar(src: &[f32], dst: &mut [i16]) {
    let n = src.len().min(dst.len());
    for i in 0..n {
        let scaled = (src[i] * F32_TO_S16_SCALE).round();
        dst[i] = scaled.clamp(i16::MIN as f32, i16::MAX as f32) as i16;
    }
}

/// Converts `f32` samples in `[-1.0, 1.0]` to saturating `i16`, using `aarch64` NEON (`vcvtaq_s32_f32` + `vqmovn_s32`).
pub fn pcm_f32_to_s16(src: &[f32], dst: &mut [i16]) {
    let n = src.len().min(dst.len());
    let mut i = 0usize;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let v_scale = vdupq_n_f32(F32_TO_S16_SCALE);
        let v_min = vdupq_n_f32(-32768.0);
        let v_max = vdupq_n_f32(32767.0);
        while i + 8 <= n {
            let f_lo = vld1q_f32(src.as_ptr().add(i));
            let f_hi = vld1q_f32(src.as_ptr().add(i + 4));
            let s_lo = vminq_f32(vmaxq_f32(vmulq_f32(f_lo, v_scale), v_min), v_max);
            let s_hi = vminq_f32(vmaxq_f32(vmulq_f32(f_hi, v_scale), v_min), v_max);
            let i32_lo = vcvtaq_s32_f32(s_lo);
            let i32_hi = vcvtaq_s32_f32(s_hi);
            let i16_lo = vqmovn_s32(i32_lo);
            let i16_hi = vqmovn_s32(i32_hi);
            let combined = vcombine_s16(i16_lo, i16_hi);
            vst1q_s16(dst.as_mut_ptr().add(i), combined);
            i += 8;
        }
    }

    if i < n {
        pcm_f32_to_s16_scalar(&src[i..n], &mut dst[i..n]);
    }
}

/// Unpacks 24-bit signed little-endian PCM bytes (`3 * num_samples`) into sign-extended `i32` samples (`[-8388608, 8388607]`).
pub fn pcm_s24le_to_i32(src_bytes: &[u8], dst: &mut [i32]) {
    let n = (src_bytes.len() / 3).min(dst.len());
    for i in 0..n {
        let b0 = src_bytes[i * 3] as u32;
        let b1 = (src_bytes[i * 3 + 1] as u32) << 8;
        let b2 = (src_bytes[i * 3 + 2] as u32) << 16;
        let raw = (b0 | b1 | b2) as i32;
        dst[i] = (raw << 8) >> 8;
    }
}

/// Packs signed `i32` samples (`[-8388608, 8388607]`) into 24-bit little-endian PCM bytes (`3 * src.len()`).
pub fn pcm_i32_to_s24le(src: &[i32], dst_bytes: &mut [u8]) {
    let n = src.len().min(dst_bytes.len() / 3);
    for i in 0..n {
        let clamped = src[i].clamp(-8_388_608, 8_388_607);
        let bytes = clamped.to_le_bytes();
        dst_bytes[i * 3] = bytes[0];
        dst_bytes[i * 3 + 1] = bytes[1];
        dst_bytes[i * 3 + 2] = bytes[2];
    }
}

/// Scalar reference: deinterleaves stereo `i32` samples `[L0, R0, L1, R1, ...]` into planar `left` and `right`.
pub fn deinterleave_stereo_i32_scalar(interleaved: &[i32], left: &mut [i32], right: &mut [i32]) {
    let frames = (interleaved.len() / 2).min(left.len()).min(right.len());
    for i in 0..frames {
        left[i] = interleaved[i * 2];
        right[i] = interleaved[i * 2 + 1];
    }
}

/// Deinterleaves stereo `i32` samples into planar `left` and `right`, using `aarch64` NEON `vld2q_s32`.
pub fn deinterleave_stereo_i32(interleaved: &[i32], left: &mut [i32], right: &mut [i32]) {
    let frames = (interleaved.len() / 2).min(left.len()).min(right.len());
    let mut i = 0usize;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        while i + 4 <= frames {
            let pair = vld2q_s32(interleaved.as_ptr().add(i * 2));
            vst1q_s32(left.as_mut_ptr().add(i), pair.0);
            vst1q_s32(right.as_mut_ptr().add(i), pair.1);
            i += 4;
        }
    }

    if i < frames {
        deinterleave_stereo_i32_scalar(
            &interleaved[i * 2..frames * 2],
            &mut left[i..frames],
            &mut right[i..frames],
        );
    }
}

/// Scalar reference: interleaves planar `left` and `right` `i32` samples into `[L0, R0, L1, R1, ...]`.
pub fn interleave_stereo_i32_scalar(left: &[i32], right: &[i32], dst: &mut [i32]) {
    let frames = left.len().min(right.len()).min(dst.len() / 2);
    for i in 0..frames {
        dst[i * 2] = left[i];
        dst[i * 2 + 1] = right[i];
    }
}

/// Interleaves planar `left` and `right` `i32` samples into `dst`, using `aarch64` NEON `vst2q_s32`.
pub fn interleave_stereo_i32(left: &[i32], right: &[i32], dst: &mut [i32]) {
    let frames = left.len().min(right.len()).min(dst.len() / 2);
    let mut i = 0usize;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        while i + 4 <= frames {
            let l = vld1q_s32(left.as_ptr().add(i));
            let r = vld1q_s32(right.as_ptr().add(i));
            vst2q_s32(dst.as_mut_ptr().add(i * 2), int32x4x2_t(l, r));
            i += 4;
        }
    }

    if i < frames {
        interleave_stereo_i32_scalar(
            &left[i..frames],
            &right[i..frames],
            &mut dst[i * 2..frames * 2],
        );
    }
}

/// Scalar reference: FLAC Mid-Side stereo decorrelation (`mid = (left + right) >> 1`, `side = left - right`).
///
/// Matches `libavcodec/flacenc.c` (`channel_decorrelation`) and RFC 9639 Section 9.1.3.
pub fn decorrelate_mid_side_i32_scalar(
    left: &[i32],
    right: &[i32],
    mid: &mut [i32],
    side: &mut [i32],
) {
    let n = left.len().min(right.len()).min(mid.len()).min(side.len());
    for i in 0..n {
        let l = left[i] as i64;
        let r = right[i] as i64;
        mid[i] = ((l + r) >> 1) as i32;
        side[i] = (l - r) as i32;
    }
}

/// Vectorized FLAC Mid-Side stereo decorrelation (`mid = (left + right) >> 1`, `side = left - right`).
///
/// Uses `vhaddq_s32` (`(l + r) >> 1` without 32-bit overflow!) and `vsubq_s32` on `aarch64` NEON.
pub fn decorrelate_mid_side_i32(left: &[i32], right: &[i32], mid: &mut [i32], side: &mut [i32]) {
    let n = left.len().min(right.len()).min(mid.len()).min(side.len());
    let mut i = 0usize;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        while i + 4 <= n {
            let l = vld1q_s32(left.as_ptr().add(i));
            let r = vld1q_s32(right.as_ptr().add(i));
            let m = vhaddq_s32(l, r);
            let s = vsubq_s32(l, r);
            vst1q_s32(mid.as_mut_ptr().add(i), m);
            vst1q_s32(side.as_mut_ptr().add(i), s);
            i += 4;
        }
    }

    if i < n {
        decorrelate_mid_side_i32_scalar(&left[i..n], &right[i..n], &mut mid[i..n], &mut side[i..n]);
    }
}

/// Scalar reference: FLAC Mid-Side stereo reconstruction (`right = mid - (side >> 1)`, `left = right + side`).
///
/// Matches `libavcodec/flacdsp_template.c` (`flac_decorrelate_ms_c`).
pub fn reconstruct_mid_side_i32_scalar(
    mid: &[i32],
    side: &[i32],
    left: &mut [i32],
    right: &mut [i32],
) {
    let n = mid.len().min(side.len()).min(left.len()).min(right.len());
    for i in 0..n {
        let m = mid[i];
        let s = side[i];
        let r = m.wrapping_sub(s >> 1);
        let l = r.wrapping_add(s);
        left[i] = l;
        right[i] = r;
    }
}

/// Vectorized FLAC Mid-Side stereo reconstruction (`right = mid - (side >> 1)`, `left = right + side`).
pub fn reconstruct_mid_side_i32(mid: &[i32], side: &[i32], left: &mut [i32], right: &mut [i32]) {
    let n = mid.len().min(side.len()).min(left.len()).min(right.len());
    let mut i = 0usize;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        while i + 4 <= n {
            let m = vld1q_s32(mid.as_ptr().add(i));
            let s = vld1q_s32(side.as_ptr().add(i));
            let s_half = vshrq_n_s32(s, 1);
            let r = vsubq_s32(m, s_half);
            let l = vaddq_s32(r, s);
            vst1q_s32(left.as_mut_ptr().add(i), l);
            vst1q_s32(right.as_mut_ptr().add(i), r);
            i += 4;
        }
    }

    if i < n {
        reconstruct_mid_side_i32_scalar(&mid[i..n], &side[i..n], &mut left[i..n], &mut right[i..n]);
    }
}

/// FLAC Left-Side stereo reconstruction (`left = ch0`, `right = ch0 - ch1`).
pub fn reconstruct_left_side_i32(ch0: &[i32], ch1: &[i32], left: &mut [i32], right: &mut [i32]) {
    let n = ch0.len().min(ch1.len()).min(left.len()).min(right.len());
    for i in 0..n {
        left[i] = ch0[i];
        right[i] = ch0[i].wrapping_sub(ch1[i]);
    }
}

/// FLAC Right-Side stereo reconstruction (`left = ch1 + ch0`, `right = ch1`).
pub fn reconstruct_right_side_i32(ch0: &[i32], ch1: &[i32], left: &mut [i32], right: &mut [i32]) {
    let n = ch0.len().min(ch1.len()).min(left.len()).min(right.len());
    for i in 0..n {
        left[i] = ch1[i].wrapping_add(ch0[i]);
        right[i] = ch1[i];
    }
}

/// Extracts planar `i32` channel buffers (each scaled to the native bit-depth integer range) from an [`AudioFrame`].
///
/// Returns `(planar_channels, bits_per_sample)`.
pub fn audio_frame_to_planar_i32(frame: &AudioFrame) -> Result<(Vec<Vec<i32>>, u8)> {
    let ch = frame.channels as usize;
    let n = frame.num_samples;
    if ch == 0 {
        return Err(RagdollError::InvalidData(
            "audio frame has 0 channels".into(),
        ));
    }
    let total = n * ch;
    let mut interleaved = vec![0i32; total];
    let bps = match frame.format {
        SampleFormat::U8 => {
            for (i, &b) in frame.data.as_slice().iter().take(total).enumerate() {
                interleaved[i] = (b as i32) - 128;
            }
            8
        }
        SampleFormat::S16le => {
            for (i, &s) in frame.as_i16_slice().iter().take(total).enumerate() {
                interleaved[i] = s as i32;
            }
            16
        }
        SampleFormat::S24le => {
            pcm_s24le_to_i32(frame.data.as_slice(), &mut interleaved);
            24
        }
        SampleFormat::S32le => {
            let bytes = frame.data.as_slice();
            for i in 0..total.min(bytes.len() / 4) {
                let b: [u8; 4] = bytes[i * 4..i * 4 + 4].try_into().unwrap();
                // Shift 32-bit PCM down to 24-bit range if needed, or preserve 24-bit dynamic range
                interleaved[i] = i32::from_le_bytes(b) >> 8;
            }
            24
        }
        SampleFormat::F32le => {
            let mut s16 = vec![0i16; total];
            pcm_f32_to_s16(frame.as_f32_slice(), &mut s16);
            for i in 0..total {
                interleaved[i] = s16[i] as i32;
            }
            16
        }
    };

    let mut planar = vec![vec![0i32; n]; ch];
    if ch == 1 {
        planar[0].copy_from_slice(&interleaved);
    } else if ch == 2 {
        let (left_slice, right_slice) = planar.split_at_mut(1);
        deinterleave_stereo_i32(&interleaved, &mut left_slice[0], &mut right_slice[0]);
    } else {
        for i in 0..n {
            for c in 0..ch {
                planar[c][i] = interleaved[i * ch + c];
            }
        }
    }
    Ok((planar, bps))
}

/// Packs planar `i32` channel buffers with `bits_per_sample` (`8`, `16`, `24`, or `32`) into an [`AudioFrame`].
pub fn planar_i32_to_audio_frame(
    planar: &[Vec<i32>],
    sample_rate: u32,
    bits_per_sample: u8,
) -> Result<AudioFrame> {
    if planar.is_empty() {
        return Err(RagdollError::InvalidData(
            "empty planar channel list".into(),
        ));
    }
    let ch = planar.len();
    let n = planar[0].len();
    for chan in planar {
        if chan.len() != n {
            return Err(RagdollError::InvalidData(
                "mismatched channel sample counts".into(),
            ));
        }
    }

    let total = n * ch;
    let mut interleaved = vec![0i32; total];
    if ch == 1 {
        interleaved.copy_from_slice(&planar[0]);
    } else if ch == 2 {
        interleave_stereo_i32(&planar[0], &planar[1], &mut interleaved);
    } else {
        for i in 0..n {
            for (c, chan) in planar.iter().enumerate() {
                interleaved[i * ch + c] = chan[i];
            }
        }
    }

    let format = match bits_per_sample {
        1..=8 => SampleFormat::U8,
        9..=16 => SampleFormat::S16le,
        17..=24 => SampleFormat::S24le,
        _ => SampleFormat::S32le,
    };
    let mut frame = AudioFrame::new(sample_rate, ch as u8, format, n);
    match format {
        SampleFormat::U8 => {
            let shift = 8u8.saturating_sub(bits_per_sample);
            let dst = frame.data.as_mut_slice();
            for i in 0..total {
                let s = (interleaved[i] << shift).clamp(-128, 127);
                dst[i] = (s + 128) as u8;
            }
        }
        SampleFormat::S16le => {
            let shift = 16u8.saturating_sub(bits_per_sample);
            let dst = frame.as_i16_mut_slice();
            for i in 0..total {
                dst[i] = (interleaved[i] << shift).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            }
        }
        SampleFormat::S24le => {
            let shift = 24u8.saturating_sub(bits_per_sample);
            if shift > 0 {
                for s in &mut interleaved {
                    *s <<= shift;
                }
            }
            pcm_i32_to_s24le(&interleaved, frame.data.as_mut_slice());
        }
        SampleFormat::S32le | SampleFormat::F32le => {
            let shift = 32u8.saturating_sub(bits_per_sample);
            let dst = frame.data.as_mut_slice();
            for i in 0..total {
                let val = interleaved[i].wrapping_shl(shift as u32);
                dst[i * 4..i * 4 + 4].copy_from_slice(&val.to_le_bytes());
            }
        }
    }
    Ok(frame)
}

/// Converts an [`AudioFrame`] between any two [`SampleFormat`] representations.
pub fn convert_audio_frame(src: &AudioFrame, dst_format: SampleFormat) -> Result<AudioFrame> {
    if src.format == dst_format {
        return Ok(src.clone());
    }
    let ch = src.channels;
    let n = src.num_samples;
    let total = n * (ch as usize);
    let mut f32_buf = vec![0.0f32; total];

    // Step 1: decode source into normalized f32 [-1.0, 1.0]
    match src.format {
        SampleFormat::U8 => {
            for (i, &b) in src.data.as_slice().iter().take(total).enumerate() {
                f32_buf[i] = ((b as f32) - 128.0) / 128.0;
            }
        }
        SampleFormat::S16le => {
            pcm_s16_to_f32(src.as_i16_slice(), &mut f32_buf);
        }
        SampleFormat::S24le => {
            let mut i32_buf = vec![0i32; total];
            pcm_s24le_to_i32(src.data.as_slice(), &mut i32_buf);
            for i in 0..total {
                f32_buf[i] = (i32_buf[i] as f32) / 8_388_608.0;
            }
        }
        SampleFormat::S32le => {
            let bytes = src.data.as_slice();
            for i in 0..total.min(bytes.len() / 4) {
                let b: [u8; 4] = bytes[i * 4..i * 4 + 4].try_into().unwrap();
                f32_buf[i] = (i32::from_le_bytes(b) as f64 / 2_147_483_648.0) as f32;
            }
        }
        SampleFormat::F32le => {
            f32_buf.copy_from_slice(&src.as_f32_slice()[..total]);
        }
    }

    // Step 2: encode normalized f32 into dst_format
    let mut dst = AudioFrame::new(src.sample_rate, ch, dst_format, n);
    dst.layout = src.layout;
    dst.pts = src.pts;
    match dst_format {
        SampleFormat::U8 => {
            let out = dst.data.as_mut_slice();
            for i in 0..total {
                let s = (f32_buf[i] * 128.0 + 128.0).round().clamp(0.0, 255.0) as u8;
                out[i] = s;
            }
        }
        SampleFormat::S16le => {
            pcm_f32_to_s16(&f32_buf, dst.as_i16_mut_slice());
        }
        SampleFormat::S24le => {
            let mut i32_buf = vec![0i32; total];
            for i in 0..total {
                i32_buf[i] = (f32_buf[i] * 8_388_608.0)
                    .round()
                    .clamp(-8_388_608.0, 8_388_607.0) as i32;
            }
            pcm_i32_to_s24le(&i32_buf, dst.data.as_mut_slice());
        }
        SampleFormat::S32le => {
            let out = dst.data.as_mut_slice();
            for i in 0..total {
                let v = ((f32_buf[i] as f64) * 2_147_483_648.0)
                    .round()
                    .clamp(i32::MIN as f64, i32::MAX as f64) as i32;
                out[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        SampleFormat::F32le => {
            dst.as_f32_mut_slice().copy_from_slice(&f32_buf);
        }
    }
    Ok(dst)
}
