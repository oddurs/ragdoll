//! **`SIMD-0042`**: Windowed-Sinc Kaiser (`bessel_i0`) Polyphase Audio Resampler &
//! ITU-R BS.775 Channel Matrix Mixer (`crates/ragdoll-codec/src/resample.rs`).
//!
//! - Exact rational phase accumulator $(n \cdot F_{\text{in}}') \pmod{F_{\text{out}}'}$ with zero float drift.
//! - Kaiser-windowed sinc filter bank ($\beta = 9.0$, $>90\text{ dB}$ stopband attenuation) with per-phase DC-unity normalization.
//! - Zero heap allocations in the per-frame resampling hot loop (`resample_planar_f32_into`, `resample_interleaved_f32_into`, `resample_planar_i16_into`).
//! - ITU-R BS.775 channel matrix mixer (`Mono <-> Stereo <-> 5.1 <-> 7.1`).

#![allow(clippy::needless_range_loop, clippy::manual_div_ceil)]

use std::f64::consts::PI;

use ragdoll_core::{
    AudioFrame, ChannelLayout, Rational64, Result, SampleFormat, Timestamp, bessel_i0, rescale_q,
};
use ragdoll_simd::{
    ITU_MIX_MINUS_3DB, polyphase_dot_f32, polyphase_dot_f32_scalar, polyphase_dot16_q15_i16,
    polyphase_dot16_q15_i16_scalar, polyphase_dot32_q15_i16, polyphase_dot32_q15_i16_scalar,
    rematrix_5_1_to_stereo_f32, rematrix_mono_to_stereo_f32, rematrix_nxm_f32,
    rematrix_stereo_to_mono_f32,
};

use crate::pcm::{convert_audio_frame, pcm_f32_to_s16, pcm_s16_to_f32};

#[inline]
fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.max(1)
}

#[inline]
fn sinc_norm(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let px = PI * x;
        px.sin() / px
    }
}

/// Stateful, zero-allocation Windowed-Sinc Kaiser (`bessel_i0`) Polyphase Audio Resampler.
#[derive(Clone, Debug)]
pub struct PolyphaseResampler {
    pub in_rate: u32,
    pub out_rate: u32,
    pub channels: usize,
    pub taps_per_phase: usize,
    pub num_phases: usize,
    pub beta: f64,
    in_rate_red: usize,
    out_rate_red: usize,
    step_int: usize,
    step_rem: usize,
    direct_phase_map: bool,
    /// Flattened `[phase * taps_per_phase + k]` `f32` filter table with DC-unity per phase.
    filter_f32: Vec<f32>,
    /// Flattened `[phase * taps_per_phase + k]` `Q15` (`32768 = 1.0`) `i16` filter table with DC-unity per phase.
    filter_q15: Vec<i16>,
    /// Current integer rational phase remainder in `0..out_rate_red`.
    phase_rem: usize,
    /// Per-channel ring/delay history buffer (`taps_per_phase - 1` samples per channel) + scratch workspace
    /// pre-allocated at construction time so `resample_*_into` performs **zero heap allocations**.
    history_f32: Vec<Vec<f32>>,
    work_f32: Vec<Vec<f32>>,
    history_i16: Vec<Vec<i16>>,
    work_i16: Vec<Vec<i16>>,
}

impl PolyphaseResampler {
    /// Constructs a new Kaiser-windowed polyphase resampler (`taps_per_phase` $\in \{16, 32, 64\}$, default $\beta = 9.0$).
    pub fn new(in_rate: u32, out_rate: u32, channels: usize, taps_per_phase: usize) -> Self {
        Self::with_kaiser_beta(in_rate, out_rate, channels, taps_per_phase, 9.0)
    }

    /// Constructs a new Kaiser-windowed polyphase resampler with explicit Kaiser $\beta$ parameter.
    pub fn with_kaiser_beta(
        in_rate: u32,
        out_rate: u32,
        channels: usize,
        taps_per_phase: usize,
        beta: f64,
    ) -> Self {
        assert!(in_rate > 0 && out_rate > 0 && channels > 0);
        assert!(
            taps_per_phase == 16 || taps_per_phase == 32 || taps_per_phase == 64,
            "taps_per_phase must be 16, 32, or 64"
        );
        let g = gcd_u32(in_rate, out_rate);
        let in_rate_red = (in_rate / g) as usize;
        let out_rate_red = (out_rate / g) as usize;
        let step_int = in_rate_red / out_rate_red;
        let step_rem = in_rate_red % out_rate_red;

        // Use exact reduced denominator as phase count when <= 1024 (covers 44.1/48/88.2/96/192 kHz with 0 division in inner loop)
        let (num_phases, direct_phase_map) = if out_rate_red <= 1024 {
            (out_rate_red, true)
        } else {
            (256, false)
        };

        let ratio = (out_rate as f64) / (in_rate as f64);
        // Anti-aliasing cutoff when downsampling, 0.95 Nyquist transition margin
        let fc = ratio.min(1.0) * 0.95;
        let half_t = (taps_per_phase as f64) * 0.5;
        let inv_i0_beta = 1.0 / bessel_i0(beta);

        let mut filter_f32 = vec![0.0f32; num_phases * taps_per_phase];
        let mut filter_q15 = vec![0i16; num_phases * taps_per_phase];
        let mut raw_taps = vec![0.0f64; taps_per_phase];

        for p in 0..num_phases {
            let frac = (p as f64) / (num_phases as f64);
            let mut sum = 0.0f64;
            for (k, tap_val) in raw_taps.iter_mut().enumerate() {
                // Sample position relative to filter center (causal delay = T/2 - 1 + frac)
                let x = (k as f64) - (half_t - 1.0) - frac;
                let u = (x / half_t).clamp(-1.0, 1.0);
                let win = bessel_i0(beta * (1.0 - u * u).max(0.0).sqrt()) * inv_i0_beta;
                let h = sinc_norm(fc * x) * win;
                *tap_val = h;
                sum += h;
            }
            let inv_sum = if sum.abs() > 1e-15 { 1.0 / sum } else { 1.0 };
            let row_f32 = &mut filter_f32[p * taps_per_phase..(p + 1) * taps_per_phase];
            let row_q15 = &mut filter_q15[p * taps_per_phase..(p + 1) * taps_per_phase];
            let mut q15_sum = 0i32;
            for k in 0..taps_per_phase {
                let norm_h = raw_taps[k] * inv_sum;
                row_f32[k] = norm_h as f32;
                let q = (norm_h * 32768.0).round().clamp(-32768.0, 32767.0) as i16;
                row_q15[k] = q;
                q15_sum += q as i32;
            }
            // Enforce exact Q15 DC-unity (sum == 32768) on the center tap
            let center_idx = taps_per_phase / 2 - 1;
            let diff = 32768 - q15_sum;
            row_q15[center_idx] =
                (row_q15[center_idx] as i32 + diff).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }

        let hist_len = taps_per_phase - 1;
        let prealloc_cap = hist_len + 8192;
        let history_f32 = (0..channels).map(|_| vec![0.0f32; hist_len]).collect();
        let work_f32 = (0..channels)
            .map(|_| Vec::with_capacity(prealloc_cap))
            .collect();
        let history_i16 = (0..channels).map(|_| vec![0i16; hist_len]).collect();
        let work_i16 = (0..channels)
            .map(|_| Vec::with_capacity(prealloc_cap))
            .collect();

        Self {
            in_rate,
            out_rate,
            channels,
            taps_per_phase,
            num_phases,
            beta,
            in_rate_red,
            out_rate_red,
            step_int,
            step_rem,
            direct_phase_map,
            filter_f32,
            filter_q15,
            phase_rem: 0,
            history_f32,
            work_f32,
            history_i16,
            work_i16,
        }
    }

    /// Resets the phase accumulator and delay lines to zero.
    pub fn reset(&mut self) {
        self.phase_rem = 0;
        for h in &mut self.history_f32 {
            h.fill(0.0);
        }
        for h in &mut self.history_i16 {
            h.fill(0);
        }
    }

    /// Returns the exact number of output samples that will be produced by the next call
    /// with `in_samples` input samples per channel.
    #[inline]
    pub fn output_frames_for(&self, in_samples: usize) -> usize {
        if in_samples == 0 {
            return 0;
        }
        let total_num = in_samples * self.out_rate_red;
        if total_num <= self.phase_rem {
            return 0;
        }
        (total_num - self.phase_rem + self.in_rate_red - 1) / self.in_rate_red
    }

    /// Ensures the internal work buffers have enough capacity for `max_in_samples` without reallocating in the hot loop.
    pub fn reserve_input_capacity(&mut self, max_in_samples: usize) {
        let needed = (self.taps_per_phase - 1) + max_in_samples;
        for w in &mut self.work_f32 {
            if w.capacity() < needed {
                w.reserve(needed - w.capacity());
            }
        }
        for w in &mut self.work_i16 {
            if w.capacity() < needed {
                w.reserve(needed - w.capacity());
            }
        }
    }

    /// Resamples planar `f32` channels (`inputs[ch]` of length `in_samples`) into pre-allocated `outputs[ch]`.
    /// Performs **zero heap allocations** when `max_in_samples` fits within the pre-allocated work buffer.
    /// Returns the number of output samples written per channel.
    pub fn resample_planar_f32_into(
        &mut self,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> usize {
        self.resample_planar_f32_impl(inputs, outputs, true)
    }

    /// Scalar reference version of [`Self::resample_planar_f32_into`] for differential testing.
    pub fn resample_planar_f32_scalar_into(
        &mut self,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> usize {
        self.resample_planar_f32_impl(inputs, outputs, false)
    }

    fn resample_planar_f32_impl(
        &mut self,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
        use_simd: bool,
    ) -> usize {
        assert_eq!(inputs.len(), self.channels);
        assert_eq!(outputs.len(), self.channels);
        let in_samples = inputs[0].len();
        if in_samples == 0 {
            return 0;
        }
        let out_samples = self.output_frames_for(in_samples);
        let hist_len = self.taps_per_phase - 1;

        for ch in 0..self.channels {
            assert_eq!(inputs[ch].len(), in_samples);
            assert!(outputs[ch].len() >= out_samples);
            let w = &mut self.work_f32[ch];
            w.clear();
            w.extend_from_slice(&self.history_f32[ch]);
            w.extend_from_slice(inputs[ch]);
        }

        let taps = self.taps_per_phase;
        let out_rate_red = self.out_rate_red;
        let step_int = self.step_int;
        let step_rem = self.step_rem;
        let num_phases = self.num_phases;
        let direct = self.direct_phase_map;

        #[cfg(target_arch = "aarch64")]
        if use_simd && taps == 16 && self.channels == 2 && direct {
            unsafe {
                use std::arch::aarch64::*;
                let w0 = self.work_f32[0].as_ptr();
                let w1 = self.work_f32[1].as_ptr();
                let dst0 = outputs[0].as_mut_ptr();
                let dst1 = outputs[1].as_mut_ptr();
                let filt = self.filter_f32.as_ptr();

                let mut src_idx = 0usize;
                let mut rem = self.phase_rem;

                for out_idx in 0..out_samples {
                    let tp = filt.add(rem * 16);
                    let t0 = vld1q_f32(tp);
                    let t1 = vld1q_f32(tp.add(4));
                    let t2 = vld1q_f32(tp.add(8));
                    let t3 = vld1q_f32(tp.add(12));

                    let s0_p = w0.add(src_idx);
                    let mut acc_l0 = vmulq_f32(vld1q_f32(s0_p), t0);
                    let mut acc_l1 = vmulq_f32(vld1q_f32(s0_p.add(4)), t1);
                    acc_l0 = vfmaq_f32(acc_l0, vld1q_f32(s0_p.add(8)), t2);
                    acc_l1 = vfmaq_f32(acc_l1, vld1q_f32(s0_p.add(12)), t3);
                    *dst0.add(out_idx) = vaddvq_f32(vaddq_f32(acc_l0, acc_l1));

                    let s1_p = w1.add(src_idx);
                    let mut acc_r0 = vmulq_f32(vld1q_f32(s1_p), t0);
                    let mut acc_r1 = vmulq_f32(vld1q_f32(s1_p.add(4)), t1);
                    acc_r0 = vfmaq_f32(acc_r0, vld1q_f32(s1_p.add(8)), t2);
                    acc_r1 = vfmaq_f32(acc_r1, vld1q_f32(s1_p.add(12)), t3);
                    *dst1.add(out_idx) = vaddvq_f32(vaddq_f32(acc_r0, acc_r1));

                    src_idx += step_int;
                    rem += step_rem;
                    if rem >= out_rate_red {
                        rem -= out_rate_red;
                        src_idx += 1;
                    }
                }
                self.phase_rem = rem;
                for ch in 0..2 {
                    let w = &self.work_f32[ch];
                    self.history_f32[ch].copy_from_slice(&w[in_samples..in_samples + hist_len]);
                }
                return out_samples;
            }
        }

        let mut src_idx = 0usize;
        let mut rem = self.phase_rem;

        for out_idx in 0..out_samples {
            let phase_idx = if direct {
                rem
            } else {
                (rem * num_phases) / out_rate_red
            };
            let phase_taps = &self.filter_f32[phase_idx * taps..(phase_idx + 1) * taps];
            for ch in 0..self.channels {
                let win = &self.work_f32[ch][src_idx..src_idx + taps];
                outputs[ch][out_idx] = if use_simd {
                    polyphase_dot_f32(win, phase_taps)
                } else {
                    polyphase_dot_f32_scalar(win, phase_taps)
                };
            }
            src_idx += step_int;
            rem += step_rem;
            if rem >= out_rate_red {
                rem -= out_rate_red;
                src_idx += 1;
            }
        }

        self.phase_rem = rem;
        for ch in 0..self.channels {
            let w = &self.work_f32[ch];
            self.history_f32[ch].copy_from_slice(&w[in_samples..in_samples + hist_len]);
        }
        out_samples
    }

    /// Resamples planar `i16` channels using Q15 fixed-point polyphase filter bank into pre-allocated `outputs[ch]`.
    pub fn resample_planar_i16_into(
        &mut self,
        inputs: &[&[i16]],
        outputs: &mut [&mut [i16]],
    ) -> usize {
        self.resample_planar_i16_impl(inputs, outputs, true)
    }

    /// Scalar reference version of [`Self::resample_planar_i16_into`] for 0-LSB differential testing.
    pub fn resample_planar_i16_scalar_into(
        &mut self,
        inputs: &[&[i16]],
        outputs: &mut [&mut [i16]],
    ) -> usize {
        self.resample_planar_i16_impl(inputs, outputs, false)
    }

    fn resample_planar_i16_impl(
        &mut self,
        inputs: &[&[i16]],
        outputs: &mut [&mut [i16]],
        use_simd: bool,
    ) -> usize {
        assert_eq!(inputs.len(), self.channels);
        assert_eq!(outputs.len(), self.channels);
        let in_samples = inputs[0].len();
        if in_samples == 0 {
            return 0;
        }
        let out_samples = self.output_frames_for(in_samples);
        let hist_len = self.taps_per_phase - 1;

        for ch in 0..self.channels {
            assert_eq!(inputs[ch].len(), in_samples);
            assert!(outputs[ch].len() >= out_samples);
            let w = &mut self.work_i16[ch];
            w.clear();
            w.extend_from_slice(&self.history_i16[ch]);
            w.extend_from_slice(inputs[ch]);
        }

        let taps = self.taps_per_phase;
        let out_rate_red = self.out_rate_red;
        let step_int = self.step_int;
        let step_rem = self.step_rem;
        let num_phases = self.num_phases;
        let direct = self.direct_phase_map;

        let mut src_idx = 0usize;
        let mut rem = self.phase_rem;

        for out_idx in 0..out_samples {
            let phase_idx = if direct {
                rem
            } else {
                (rem * num_phases) / out_rate_red
            };
            let phase_taps = &self.filter_q15[phase_idx * taps..(phase_idx + 1) * taps];
            for ch in 0..self.channels {
                let win = &self.work_i16[ch][src_idx..src_idx + taps];
                outputs[ch][out_idx] = if taps == 16 {
                    let t16: &[i16; 16] = phase_taps.try_into().unwrap();
                    if use_simd {
                        polyphase_dot16_q15_i16(win, t16)
                    } else {
                        polyphase_dot16_q15_i16_scalar(win, t16)
                    }
                } else {
                    let t32: &[i16; 32] = phase_taps[..32].try_into().unwrap();
                    if use_simd {
                        polyphase_dot32_q15_i16(win, t32)
                    } else {
                        polyphase_dot32_q15_i16_scalar(win, t32)
                    }
                };
            }
            src_idx += step_int;
            rem += step_rem;
            if rem >= out_rate_red {
                rem -= out_rate_red;
                src_idx += 1;
            }
        }

        self.phase_rem = rem;
        for ch in 0..self.channels {
            let w = &self.work_i16[ch];
            self.history_i16[ch].copy_from_slice(&w[in_samples..in_samples + hist_len]);
        }
        out_samples
    }

    /// Resamples an `AudioFrame` into a new `AudioFrame` at `self.out_rate`, preserving `frame.format` and rescaling `pts`.
    pub fn resample_audio_frame(&mut self, frame: &AudioFrame) -> Result<AudioFrame> {
        if frame.sample_rate == self.out_rate {
            return Ok(frame.clone());
        }
        let orig_fmt = frame.format;
        let f32_frame = if orig_fmt == SampleFormat::F32le {
            frame.clone()
        } else {
            convert_audio_frame(frame, SampleFormat::F32le)?
        };
        let in_samples = f32_frame.num_samples;
        let out_samples = self.output_frames_for(in_samples);
        let ch_count = self.channels;
        assert_eq!(f32_frame.channels as usize, ch_count);

        let src_interleaved = f32_frame.as_f32_slice();
        let mut in_planar = vec![vec![0.0f32; in_samples]; ch_count];
        for i in 0..in_samples {
            for ch in 0..ch_count {
                in_planar[ch][i] = src_interleaved[i * ch_count + ch];
            }
        }
        let mut out_planar = vec![vec![0.0f32; out_samples]; ch_count];
        {
            let in_refs: Vec<&[f32]> = in_planar.iter().map(|v| v.as_slice()).collect();
            let mut out_refs: Vec<&mut [f32]> =
                out_planar.iter_mut().map(|v| v.as_mut_slice()).collect();
            self.resample_planar_f32_into(&in_refs, &mut out_refs);
        }

        let mut out_f32 = AudioFrame::new(
            self.out_rate,
            ch_count as u8,
            SampleFormat::F32le,
            out_samples,
        );
        out_f32.layout = frame.layout;
        let out_tb = Rational64::new(1, self.out_rate as i64);
        out_f32.pts = Timestamp::new(
            rescale_q(frame.pts.ticks, frame.pts.timebase, out_tb),
            out_tb,
        );
        let dst_interleaved = out_f32.as_f32_mut_slice();
        for i in 0..out_samples {
            for ch in 0..ch_count {
                dst_interleaved[i * ch_count + ch] = out_planar[ch][i];
            }
        }

        if orig_fmt == SampleFormat::F32le {
            Ok(out_f32)
        } else {
            convert_audio_frame(&out_f32, orig_fmt)
        }
    }
}

/// ITU-R BS.775 / `libswresample` Channel Matrix Mixer supporting `Mono <-> Stereo <-> 5.1 <-> 7.1`.
#[derive(Clone, Debug)]
pub struct ChannelRematrixer {
    pub in_layout: ChannelLayout,
    pub out_layout: ChannelLayout,
    pub in_channels: usize,
    pub out_channels: usize,
    /// Row-major `out_channels * in_channels` mixing matrix.
    pub matrix: Vec<f32>,
}

impl ChannelRematrixer {
    /// Builds an ITU-R BS.775 channel mixing matrix between `in_layout` and `out_layout`.
    pub fn new(in_layout: ChannelLayout, out_layout: ChannelLayout, normalize: bool) -> Self {
        let in_ch = in_layout.channels() as usize;
        let out_ch = out_layout.channels() as usize;
        assert!(in_ch > 0 && out_ch > 0);
        let mut matrix = vec![0.0f32; out_ch * in_ch];
        let c = ITU_MIX_MINUS_3DB;

        match (in_ch, out_ch) {
            (n, m) if n == m => {
                for i in 0..n {
                    matrix[i * n + i] = 1.0;
                }
            }
            // Stereo -> Mono
            (2, 1) => {
                let g = if normalize { 0.5 } else { c };
                matrix[0] = g;
                matrix[1] = g;
            }
            // Mono -> Stereo
            (1, 2) => {
                let g = if normalize { c } else { 1.0 };
                matrix[0] = g;
                matrix[1] = g;
            }
            // 5.1 (L, R, C, LFE, Ls, Rs) -> Stereo (L, R)
            (6, 2) => {
                let norm = if normalize {
                    1.0 / (1.0 + 2.0 * c)
                } else {
                    1.0
                };
                // Left = L + 0.7071 C + 0.7071 Ls
                matrix[0] = norm;
                matrix[2] = c * norm;
                matrix[4] = c * norm;
                // Right = R + 0.7071 C + 0.7071 Rs
                matrix[6 + 1] = norm;
                matrix[6 + 2] = c * norm;
                matrix[6 + 5] = c * norm;
            }
            // 5.1 -> Mono
            (6, 1) => {
                let norm = if normalize {
                    1.0 / (2.0 + 2.0 * c + 2.0 * c)
                } else {
                    0.5
                };
                matrix[0] = norm;
                matrix[1] = norm;
                matrix[2] = 2.0 * c * norm;
                matrix[4] = c * norm;
                matrix[5] = c * norm;
            }
            // Stereo -> 5.1 (L, R mapped directly, C/LFE/Ls/Rs zeroed)
            (2, 6) => {
                matrix[0] = 1.0; // L -> L
                matrix[2 + 1] = 1.0; // R -> R
            }
            // 7.1 (L, R, C, LFE, Lb, Rb, Ls, Rs) -> 5.1 (L, R, C, LFE, Ls', Rs')
            (8, 6) => {
                let s_norm = if normalize { 1.0 / (1.0 + c) } else { 1.0 };
                matrix[0] = 1.0; // L
                matrix[8 + 1] = 1.0; // R
                matrix[16 + 2] = 1.0; // C
                matrix[24 + 3] = 1.0; // LFE
                // Ls' = Ls + 0.7071 Lb
                matrix[32 + 4] = c * s_norm;
                matrix[32 + 6] = s_norm;
                // Rs' = Rs + 0.7071 Rb
                matrix[40 + 5] = c * s_norm;
                matrix[40 + 7] = s_norm;
            }
            // 7.1 -> Stereo
            (8, 2) => {
                let norm = if normalize {
                    1.0 / (1.0 + 3.0 * c)
                } else {
                    1.0
                };
                matrix[0] = norm; // L
                matrix[2] = c * norm; // C
                matrix[4] = c * norm; // Lb
                matrix[6] = c * norm; // Ls
                matrix[8 + 1] = norm; // R
                matrix[8 + 2] = c * norm; // C
                matrix[8 + 5] = c * norm; // Rb
                matrix[8 + 7] = c * norm; // Rs
            }
            _ => {
                for oc in 0..out_ch {
                    matrix[oc * in_ch + (oc % in_ch)] = 1.0;
                }
            }
        }

        Self {
            in_layout,
            out_layout,
            in_channels: in_ch,
            out_channels: out_ch,
            matrix,
        }
    }

    /// Mixes interleaved `src` frames (`num_samples * in_channels`) into `dst` (`num_samples * out_channels`).
    pub fn process_interleaved_f32(&self, src: &[f32], dst: &mut [f32]) {
        if self.in_channels == 2 && self.out_channels == 1 && self.matrix[0] == self.matrix[1] {
            rematrix_stereo_to_mono_f32(src, dst, self.matrix[0]);
            return;
        }
        if self.in_channels == 1 && self.out_channels == 2 && self.matrix[0] == self.matrix[1] {
            rematrix_mono_to_stereo_f32(src, dst, self.matrix[0]);
            return;
        }
        if self.in_channels == 6 && self.out_channels == 2 {
            rematrix_5_1_to_stereo_f32(
                src,
                dst,
                ITU_MIX_MINUS_3DB,
                ITU_MIX_MINUS_3DB,
                0.0,
                (self.matrix[0] - 1.0).abs() > 1e-4,
            );
            return;
        }
        rematrix_nxm_f32(src, dst, self.in_channels, self.out_channels, &self.matrix);
    }

    /// Rematrixes an `AudioFrame` to `self.out_layout`, preserving its sample format and timestamp.
    pub fn rematrix_audio_frame(&self, frame: &AudioFrame) -> Result<AudioFrame> {
        if frame.channels as usize == self.out_channels {
            return Ok(frame.clone());
        }
        let orig_fmt = frame.format;
        let f32_in = if orig_fmt == SampleFormat::F32le {
            frame.clone()
        } else {
            convert_audio_frame(frame, SampleFormat::F32le)?
        };
        let mut f32_out = AudioFrame::new(
            frame.sample_rate,
            self.out_channels as u8,
            SampleFormat::F32le,
            frame.num_samples,
        );
        f32_out.layout = self.out_layout;
        f32_out.pts = frame.pts;
        self.process_interleaved_f32(f32_in.as_f32_slice(), f32_out.as_f32_mut_slice());
        if orig_fmt == SampleFormat::F32le {
            Ok(f32_out)
        } else {
            convert_audio_frame(&f32_out, orig_fmt)
        }
    }
}

/// Convenience helper converting sample format, channel layout, and sample rate in one pipeline stage.
pub fn resample_and_rematrix_frame(
    frame: &AudioFrame,
    resampler: Option<&mut PolyphaseResampler>,
    rematrixer: Option<&ChannelRematrixer>,
    target_format: SampleFormat,
) -> Result<AudioFrame> {
    let mut cur = if frame.format == SampleFormat::F32le {
        frame.clone()
    } else if frame.format == SampleFormat::S16le {
        let mut f = AudioFrame::new(
            frame.sample_rate,
            frame.channels,
            SampleFormat::F32le,
            frame.num_samples,
        );
        f.layout = frame.layout;
        f.pts = frame.pts;
        pcm_s16_to_f32(frame.as_i16_slice(), f.as_f32_mut_slice());
        f
    } else {
        convert_audio_frame(frame, SampleFormat::F32le)?
    };

    if let Some(rm) = rematrixer
        && cur.channels as usize != rm.out_channels
    {
        cur = rm.rematrix_audio_frame(&cur)?;
    }
    if let Some(rs) = resampler
        && cur.sample_rate != rs.out_rate
    {
        cur = rs.resample_audio_frame(&cur)?;
    }

    if target_format == SampleFormat::F32le {
        Ok(cur)
    } else if target_format == SampleFormat::S16le {
        let mut out = AudioFrame::new(
            cur.sample_rate,
            cur.channels,
            SampleFormat::S16le,
            cur.num_samples,
        );
        out.layout = cur.layout;
        out.pts = cur.pts;
        pcm_f32_to_s16(cur.as_f32_slice(), out.as_i16_mut_slice());
        Ok(out)
    } else {
        convert_audio_frame(&cur, target_format)
    }
}
