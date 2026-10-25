//! **`SIMD-0043`**: ITU-R BS.1770-4 / EBU R128 (`LUFS`) K-Weighting Filter,
//! Gated Integrated/Short-Term/Momentary Loudness, Loudness Range (`LRA`), and
//! 4x Oversampled Polyphase True-Peak (`dBTP`) Detector & Normalizer.
//!
//! - Analytical bilinear transform K-weighting biquad cascade for any sample rate $f_s$.
//! - $75\%$ overlapping $400\text{ ms}$ gating blocks ($\Gamma_a = -70\text{ LUFS}$, $\Gamma_r = L_{\text{abs}} - 10\text{ LU}$).
//! - EBU Tech 3342 Loudness Range (`LRA`) over $3\text{ s}$ short-term blocks ($10\text{th}\text{–}95\text{th}$ percentile).
//! - 4x oversampled 12-tap polyphase FIR True-Peak (`dBTP`) detector and limiter (`--loudness -14lufs --true-peak -1.0`).
//! - Zero heap allocations in the hot loop and 100% bit-exact NEON vs. scalar differential execution.

#![allow(clippy::needless_range_loop)]

use std::f64::consts::PI;

use ragdoll_core::{AudioFrame, Result, SampleFormat};
use ragdoll_simd::{
    BiquadCoeffs, apply_gain_true_peak_limit_f32, biquad_process_4ch_f32,
    biquad_process_4ch_f32_scalar, true_peak_4x_12tap_sample_f32,
    true_peak_4x_12tap_sample_f32_scalar,
};

use crate::pcm::convert_audio_frame;

/// Analytical ITU-R BS.1770-4 Stage 1 (acoustic head high-shelf) and Stage 2 (RLB high-pass) biquads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KWeightingFilterBank {
    pub stage1: BiquadCoeffs,
    pub stage2: BiquadCoeffs,
}

impl KWeightingFilterBank {
    /// Computes the exact ITU-R BS.1770-4 K-weighting biquad cascade for any sample rate `sample_rate`.
    /// At `48_000 Hz`, reproduces the standard Table 1 coefficients to full `f32` precision.
    pub fn for_sample_rate(sample_rate: u32) -> Self {
        let fs = sample_rate.max(8000) as f64;

        // Stage 1: Acoustic head high-shelf filter
        // Analog prototype parameters from ITU-R BS.1770-4:
        let f0_1 = 1681.974450955533_f64;
        let g_db_1 = 3.999843853973347_f64;
        let q_1 = 0.7071752369554196_f64;
        let k1 = (PI * f0_1 / fs).tan();
        let vh = 10.0_f64.powf(g_db_1 / 20.0);
        let vb = vh.powf(0.4996667741545416);
        let a0_1 = 1.0 + k1 / q_1 + k1 * k1;
        let stage1 = BiquadCoeffs {
            b0: ((vh + vb * k1 / q_1 + k1 * k1) / a0_1) as f32,
            b1: ((2.0 * (k1 * k1 - vh)) / a0_1) as f32,
            b2: ((vh - vb * k1 / q_1 + k1 * k1) / a0_1) as f32,
            a1: ((2.0 * (k1 * k1 - 1.0)) / a0_1) as f32,
            a2: ((1.0 - k1 / q_1 + k1 * k1) / a0_1) as f32,
        };

        // Stage 2: RLB (Revised Low-frequency B-curve) high-pass filter
        // Per ITU-R BS.1770-4 Table 1, numerator is unity (1.0, -2.0, 1.0)
        let f0_2 = 38.13547087602444_f64;
        let q_2 = 0.5003270373238773_f64;
        let k2 = (PI * f0_2 / fs).tan();
        let a0_2 = 1.0 + k2 / q_2 + k2 * k2;
        let stage2 = BiquadCoeffs {
            b0: 1.0,
            b1: -2.0,
            b2: 1.0,
            a1: ((2.0 * (k2 * k2 - 1.0)) / a0_2) as f32,
            a2: ((1.0 - k2 / q_2 + k2 * k2) / a0_2) as f32,
        };

        Self { stage1, stage2 }
    }
}

/// Complete EBU R128 & ITU-R BS.1770-4 loudness and True-Peak measurement summary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoudnessReport {
    /// Gated integrated loudness in `LUFS` (EBU R128 `-0.691 + 10 log10(z)`).
    pub integrated_lufs: f64,
    /// Maximum momentary ($400\text{ ms}$) loudness in `LUFS`.
    pub momentary_max_lufs: f64,
    /// Maximum short-term ($3.0\text{ s}$) loudness in `LUFS`.
    pub short_term_max_lufs: f64,
    /// EBU Tech 3342 Loudness Range (`LRA`) in `LU`.
    pub loudness_range_lu: f64,
    /// Maximum 4x oversampled inter-sample True-Peak amplitude (linear `1.0 = 0.0 dBTP`).
    pub true_peak_linear: f32,
    /// Maximum 4x oversampled inter-sample True-Peak in `dBTP` (`20 log10(true_peak_linear)`).
    pub true_peak_dbtp: f64,
}

/// Stateful, zero-allocation EBU R128 (`LUFS`) + 4x True-Peak (`dBTP`) analyzer.
#[derive(Clone, Debug)]
pub struct EbuR128Meter {
    pub sample_rate: u32,
    pub channels: usize,
    pub filters: KWeightingFilterBank,
    channel_weights: [f32; 8],
    // Biquad states grouped in 4-channel SIMD banks (supports up to 8 channels = 2 banks)
    s1_z1: [[f32; 4]; 2],
    s1_z2: [[f32; 4]; 2],
    s2_z1: [[f32; 4]; 2],
    s2_z2: [[f32; 4]; 2],
    // 12-tap True-Peak delay line per channel (circular/shifted 12-sample window)
    tp_hist: [[f32; 12]; 8],
    max_true_peak: f32,
    // 100 ms sub-block accumulator (4 sub-blocks = 400 ms momentary window; 30 sub-blocks = 3 s short-term window)
    subblock_samples_target: usize,
    subblock_samples_cur: usize,
    subblock_weighted_energy_sum: f64,
    /// Circular ring of the last 30 `100 ms` sub-block mean-square energies $z_{\text{sub}}$.
    subblock_ring: [f64; 30],
    subblock_count: usize,
    /// Pre-allocated history of $400\text{ ms}$ momentary block mean-square energies $z_j$ (10 blocks/sec).
    momentary_blocks: Vec<f64>,
    /// Pre-allocated history of $3.0\text{ s}$ short-term block mean-square energies $z_{\text{st}, j}$.
    short_term_blocks: Vec<f64>,
    /// Total weighted energy across all samples (fallback for very short clips $< 400\text{ ms}$).
    total_weighted_energy: f64,
    total_frames: u64,
}

impl EbuR128Meter {
    /// Creates a new `EbuR128Meter` for `sample_rate` and `channels` (`1..=8`), pre-allocating
    /// block storage for up to 2 hours of audio so the measurement hot loop performs **zero heap allocations**.
    pub fn new(sample_rate: u32, channels: usize) -> Self {
        assert!(sample_rate >= 8000 && (1..=8).contains(&channels));
        let filters = KWeightingFilterBank::for_sample_rate(sample_rate);
        // ITU-R BS.1770-4 channel weights:
        // L=1.0, R=1.0, C=1.0, LFE=0.0, Ls=1.41, Rs=1.41, Lb=1.41, Rb=1.41
        let mut channel_weights = [1.0f32; 8];
        if channels >= 6 {
            channel_weights[3] = 0.0; // LFE excluded
            channel_weights[4] = 1.4125375; // +1.5 dB surround
            channel_weights[5] = 1.4125375;
            if channels >= 8 {
                channel_weights[6] = 1.4125375;
                channel_weights[7] = 1.4125375;
            }
        }
        let subblock_samples_target = ((sample_rate as usize) / 10).max(1);

        Self {
            sample_rate,
            channels,
            filters,
            channel_weights,
            s1_z1: [[0.0; 4]; 2],
            s1_z2: [[0.0; 4]; 2],
            s2_z1: [[0.0; 4]; 2],
            s2_z2: [[0.0; 4]; 2],
            tp_hist: [[0.0; 12]; 8],
            max_true_peak: 0.0,
            subblock_samples_target,
            subblock_samples_cur: 0,
            subblock_weighted_energy_sum: 0.0,
            subblock_ring: [0.0; 30],
            subblock_count: 0,
            momentary_blocks: Vec::with_capacity(72_000),
            short_term_blocks: Vec::with_capacity(72_000),
            total_weighted_energy: 0.0,
            total_frames: 0,
        }
    }

    /// Resets the meter state without deallocating internal block buffers.
    pub fn reset(&mut self) {
        self.s1_z1 = [[0.0; 4]; 2];
        self.s1_z2 = [[0.0; 4]; 2];
        self.s2_z1 = [[0.0; 4]; 2];
        self.s2_z2 = [[0.0; 4]; 2];
        self.tp_hist = [[0.0; 12]; 8];
        self.max_true_peak = 0.0;
        self.subblock_samples_cur = 0;
        self.subblock_weighted_energy_sum = 0.0;
        self.subblock_ring = [0.0; 30];
        self.subblock_count = 0;
        self.momentary_blocks.clear();
        self.short_term_blocks.clear();
        self.total_weighted_energy = 0.0;
        self.total_frames = 0;
    }

    /// Processes interleaved `f32` audio samples using NEON SIMD biquad & 4x True-Peak kernels.
    /// Zero heap allocations in the hot loop.
    pub fn process_interleaved_f32(&mut self, interleaved: &[f32]) {
        self.process_interleaved_f32_impl(interleaved, true);
    }

    /// Scalar reference version of [`Self::process_interleaved_f32`] for bit-exact differential verification.
    pub fn process_interleaved_f32_scalar(&mut self, interleaved: &[f32]) {
        self.process_interleaved_f32_impl(interleaved, false);
    }

    fn process_interleaved_f32_impl(&mut self, interleaved: &[f32], use_simd: bool) {
        let ch = self.channels;
        let frames = interleaved.len() / ch;
        let stage1 = self.filters.stage1;
        let stage2 = self.filters.stage2;

        for f in 0..frames {
            let base = f * ch;
            let mut x_bank0 = [0.0f32; 4];
            let n0 = ch.min(4);
            for c in 0..n0 {
                let s = interleaved[base + c];
                x_bank0[c] = s;
                // Update 12-tap True-Peak delay line and evaluate 4x oversampled phases
                let h = &mut self.tp_hist[c];
                h.copy_within(1..12, 0);
                h[11] = s;
                let pk = if use_simd {
                    true_peak_4x_12tap_sample_f32(h)
                } else {
                    true_peak_4x_12tap_sample_f32_scalar(h)
                };
                if pk > self.max_true_peak {
                    self.max_true_peak = pk;
                }
            }

            let mut mid0 = [0.0f32; 4];
            let mut y0 = [0.0f32; 4];
            if use_simd {
                biquad_process_4ch_f32(
                    &stage1,
                    &x_bank0,
                    &mut self.s1_z1[0],
                    &mut self.s1_z2[0],
                    &mut mid0,
                );
                biquad_process_4ch_f32(
                    &stage2,
                    &mid0,
                    &mut self.s2_z1[0],
                    &mut self.s2_z2[0],
                    &mut y0,
                );
            } else {
                biquad_process_4ch_f32_scalar(
                    &stage1,
                    &x_bank0,
                    &mut self.s1_z1[0],
                    &mut self.s1_z2[0],
                    &mut mid0,
                );
                biquad_process_4ch_f32_scalar(
                    &stage2,
                    &mid0,
                    &mut self.s2_z1[0],
                    &mut self.s2_z2[0],
                    &mut y0,
                );
            }

            let mut frame_weighted_sq = 0.0f64;
            for c in 0..n0 {
                let w = self.channel_weights[c] as f64;
                let y = y0[c] as f64;
                frame_weighted_sq += w * (y * y);
            }

            if ch > 4 {
                let mut x_bank1 = [0.0f32; 4];
                let n1 = ch - 4;
                for c in 0..n1 {
                    let s = interleaved[base + 4 + c];
                    x_bank1[c] = s;
                    let h = &mut self.tp_hist[4 + c];
                    h.copy_within(1..12, 0);
                    h[11] = s;
                    let pk = if use_simd {
                        true_peak_4x_12tap_sample_f32(h)
                    } else {
                        true_peak_4x_12tap_sample_f32_scalar(h)
                    };
                    if pk > self.max_true_peak {
                        self.max_true_peak = pk;
                    }
                }
                let mut mid1 = [0.0f32; 4];
                let mut y1 = [0.0f32; 4];
                if use_simd {
                    biquad_process_4ch_f32(
                        &stage1,
                        &x_bank1,
                        &mut self.s1_z1[1],
                        &mut self.s1_z2[1],
                        &mut mid1,
                    );
                    biquad_process_4ch_f32(
                        &stage2,
                        &mid1,
                        &mut self.s2_z1[1],
                        &mut self.s2_z2[1],
                        &mut y1,
                    );
                } else {
                    biquad_process_4ch_f32_scalar(
                        &stage1,
                        &x_bank1,
                        &mut self.s1_z1[1],
                        &mut self.s1_z2[1],
                        &mut mid1,
                    );
                    biquad_process_4ch_f32_scalar(
                        &stage2,
                        &mid1,
                        &mut self.s2_z1[1],
                        &mut self.s2_z2[1],
                        &mut y1,
                    );
                }
                for c in 0..n1 {
                    let w = self.channel_weights[4 + c] as f64;
                    let y = y1[c] as f64;
                    frame_weighted_sq += w * (y * y);
                }
            }

            self.subblock_weighted_energy_sum += frame_weighted_sq;
            self.total_weighted_energy += frame_weighted_sq;
            self.total_frames += 1;
            self.subblock_samples_cur += 1;

            if self.subblock_samples_cur == self.subblock_samples_target {
                let z_sub =
                    self.subblock_weighted_energy_sum / (self.subblock_samples_target as f64);
                self.subblock_ring[self.subblock_count % 30] = z_sub;
                self.subblock_count += 1;
                self.subblock_samples_cur = 0;
                self.subblock_weighted_energy_sum = 0.0;

                // 400 ms momentary gating block (4 x 100 ms sub-blocks, 75% overlap)
                if self.subblock_count >= 4 {
                    let mut sum4 = 0.0f64;
                    for k in 0..4 {
                        sum4 += self.subblock_ring[(self.subblock_count - 1 - k) % 30];
                    }
                    let z_mom = sum4 * 0.25;
                    if self.momentary_blocks.len() < self.momentary_blocks.capacity() {
                        self.momentary_blocks.push(z_mom);
                    }
                }

                // 3.0 s short-term block (30 x 100 ms sub-blocks)
                if self.subblock_count >= 30 {
                    let mut sum30 = 0.0f64;
                    for &v in &self.subblock_ring {
                        sum30 += v;
                    }
                    let z_st = sum30 / 30.0;
                    if self.short_term_blocks.len() < self.short_term_blocks.capacity() {
                        self.short_term_blocks.push(z_st);
                    }
                }
            }
        }
    }

    /// Processes an `AudioFrame` through the EBU R128 + True-Peak meter.
    pub fn process_audio_frame(&mut self, frame: &AudioFrame) -> Result<()> {
        if frame.format == SampleFormat::F32le {
            self.process_interleaved_f32(frame.as_f32_slice());
        } else {
            let f32_frame = convert_audio_frame(frame, SampleFormat::F32le)?;
            self.process_interleaved_f32(f32_frame.as_f32_slice());
        }
        Ok(())
    }

    #[inline]
    fn energy_to_lufs(z: f64) -> f64 {
        if z <= 1e-20 {
            -120.0
        } else {
            -0.691 + 10.0 * z.log10()
        }
    }

    /// Finalizes and returns the [`LoudnessReport`] (`integrated_lufs`, `momentary_max_lufs`,
    /// `short_term_max_lufs`, `loudness_range_lu`, `true_peak_linear`, `true_peak_dbtp`).
    pub fn finalize(&self) -> LoudnessReport {
        let true_peak_linear = self.max_true_peak;
        let true_peak_dbtp = if true_peak_linear <= 1e-10 {
            -120.0
        } else {
            20.0 * (true_peak_linear as f64).log10()
        };

        // 1. Momentary max
        let momentary_max_lufs = if self.momentary_blocks.is_empty() {
            if self.total_frames == 0 {
                -120.0
            } else {
                Self::energy_to_lufs(self.total_weighted_energy / (self.total_frames as f64))
            }
        } else {
            let max_z = self.momentary_blocks.iter().copied().fold(0.0f64, f64::max);
            Self::energy_to_lufs(max_z)
        };

        // 2. Short-term max
        let short_term_max_lufs = if self.short_term_blocks.is_empty() {
            momentary_max_lufs
        } else {
            let max_z = self
                .short_term_blocks
                .iter()
                .copied()
                .fold(0.0f64, f64::max);
            Self::energy_to_lufs(max_z)
        };

        // 3. Integrated Gated Loudness (ITU-R BS.1770-4: Gamma_a = -70 LUFS, Gamma_r = L_abs - 10 LU)
        let abs_gate_z = 10.0_f64.powf((-70.0 + 0.691) / 10.0);
        let integrated_lufs = if self.momentary_blocks.is_empty() {
            momentary_max_lufs
        } else {
            let mut abs_sum = 0.0f64;
            let mut abs_cnt = 0usize;
            for &z in &self.momentary_blocks {
                if z > abs_gate_z {
                    abs_sum += z;
                    abs_cnt += 1;
                }
            }
            if abs_cnt == 0 {
                -120.0
            } else {
                let l_abs = Self::energy_to_lufs(abs_sum / (abs_cnt as f64));
                let gamma_r = l_abs - 10.0;
                let rel_gate_z = 10.0_f64.powf((gamma_r + 0.691) / 10.0);
                let mut rel_sum = 0.0f64;
                let mut rel_cnt = 0usize;
                for &z in &self.momentary_blocks {
                    if z > rel_gate_z {
                        rel_sum += z;
                        rel_cnt += 1;
                    }
                }
                if rel_cnt == 0 {
                    l_abs
                } else {
                    Self::energy_to_lufs(rel_sum / (rel_cnt as f64))
                }
            }
        };

        // 4. EBU Tech 3342 Loudness Range (LRA) over short-term blocks (or momentary blocks if < 3s)
        let lra_source = if !self.short_term_blocks.is_empty() {
            &self.short_term_blocks
        } else {
            &self.momentary_blocks
        };
        let loudness_range_lu = {
            let mut abs_sum = 0.0f64;
            let mut abs_cnt = 0usize;
            for &z in lra_source {
                if z > abs_gate_z {
                    abs_sum += z;
                    abs_cnt += 1;
                }
            }
            if abs_cnt < 2 {
                0.0
            } else {
                let l_abs = Self::energy_to_lufs(abs_sum / (abs_cnt as f64));
                let rel_gate_z = 10.0_f64.powf((l_abs - 20.0 + 0.691) / 10.0);
                let mut gated_lufs: Vec<f64> = lra_source
                    .iter()
                    .copied()
                    .filter(|&z| z >= rel_gate_z)
                    .map(Self::energy_to_lufs)
                    .collect();
                if gated_lufs.len() < 2 {
                    0.0
                } else {
                    gated_lufs
                        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let n = gated_lufs.len();
                    let p10 = gated_lufs[((n - 1) * 10) / 100];
                    let p95 = gated_lufs[((n - 1) * 95) / 100];
                    (p95 - p10).max(0.0)
                }
            }
        };

        LoudnessReport {
            integrated_lufs,
            momentary_max_lufs,
            short_term_max_lufs,
            loudness_range_lu,
            true_peak_linear,
            true_peak_dbtp,
        }
    }

    /// Computes the linear normalization gain that brings the measured signal to `target_lufs`
    /// while ensuring the 4x oversampled True-Peak does not exceed `max_true_peak_dbtp` (e.g. `-1.0 dBTP`).
    pub fn compute_normalization_gain(
        &self,
        target_lufs: f64,
        max_true_peak_dbtp: f64,
    ) -> (f32, LoudnessReport) {
        let report = self.finalize();
        if report.integrated_lufs <= -100.0 {
            return (1.0, report);
        }
        let desired_gain_db = target_lufs - report.integrated_lufs;
        let max_allowed_gain_db = max_true_peak_dbtp - report.true_peak_dbtp;
        let applied_gain_db = desired_gain_db.min(max_allowed_gain_db);
        let linear_gain = 10.0_f64.powf(applied_gain_db / 20.0) as f32;
        (linear_gain, report)
    }
}

/// Applies EBU R128 loudness normalization gain and True-Peak ceiling limiting to an `AudioFrame`.
pub fn normalize_audio_frame(
    frame: &AudioFrame,
    linear_gain: f32,
    max_true_peak_dbtp: f64,
) -> Result<AudioFrame> {
    let ceiling = 10.0_f64.powf(max_true_peak_dbtp / 20.0) as f32;
    let orig_fmt = frame.format;
    let mut f32_frame = if orig_fmt == SampleFormat::F32le {
        frame.clone()
    } else {
        convert_audio_frame(frame, SampleFormat::F32le)?
    };
    apply_gain_true_peak_limit_f32(f32_frame.as_f32_mut_slice(), linear_gain, ceiling);
    if orig_fmt == SampleFormat::F32le {
        Ok(f32_frame)
    } else {
        convert_audio_frame(&f32_frame, orig_fmt)
    }
}
