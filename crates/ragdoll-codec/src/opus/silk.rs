//! SILK decoder, a port of the fixed-point libopus 1.3.1 `silk/` decoder (`dec_API.c`,
//! `decode_frame.c`, `decode_indices.c`, `decode_pulses.c`, `decode_parameters.c`,
//! `decode_core.c`, `PLC.c`, `CNG.c`, `stereo_*.c`). Output is sample-exact with libopus.

use super::range::Ec;
use super::silk_dsp::*;
use super::silk_tables::*;

const MAX_LPC_ORDER: usize = 16;
const MAX_NB_SUBFR: usize = 4;
const MAX_FRAME_LENGTH: usize = 320;
const LTP_ORDER: usize = 5;
const MAX_SUBFR_LENGTH: usize = 80;
const TYPE_NO_VOICE_ACTIVITY: i32 = 0;
const TYPE_VOICED: i32 = 2;
const CODE_INDEPENDENTLY: i32 = 0;
const CODE_INDEPENDENTLY_NO_LTP_SCALING: i32 = 1;
const CODE_CONDITIONALLY: i32 = 2;
const FLAG_DECODE_NORMAL: i32 = 0;
const FLAG_PACKET_LOST: i32 = 1;
const FLAG_DECODE_LBRR: i32 = 2;
const RAND_BUF_SIZE: usize = 128;
const RAND_BUF_MASK: i32 = 127;
const CNG_BUF_MASK_MAX: usize = 255;

/// The parts of `silk_DecControlStruct` the Opus layer drives.
#[derive(Clone, Debug, Default)]
pub struct DecControl {
    pub n_channels_api: i32,
    pub n_channels_internal: i32,
    pub api_sample_rate: i32,
    pub internal_sample_rate: i32,
    pub payload_size_ms: i32,
    pub prev_pitch_lag: i32,
}

/// `silk_NLSF_CB_struct`.
#[derive(Debug)]
pub struct NlsfCb {
    n_vectors: usize,
    order: usize,
    quant_step_size_q16: i32,
    cb1_nlsf_q8: &'static [u8],
    cb1_wght_q9: &'static [i16],
    cb1_icdf: &'static [u8],
    pred_q8: &'static [u8],
    ec_sel: &'static [u8],
    ec_icdf: &'static [u8],
    delta_min_q15: &'static [i16],
}

static NLSF_CB_NB_MB: NlsfCb = NlsfCb {
    n_vectors: 32,
    order: 10,
    quant_step_size_q16: 11796,
    cb1_nlsf_q8: &NLSF_CB1_NB_MB_Q8,
    cb1_wght_q9: &NLSF_CB1_WGHT_Q9,
    cb1_icdf: &NLSF_CB1_ICDF_NB_MB,
    pred_q8: &NLSF_PRED_NB_MB_Q8,
    ec_sel: &NLSF_CB2_SELECT_NB_MB,
    ec_icdf: &NLSF_CB2_ICDF_NB_MB,
    delta_min_q15: &NLSF_DELTA_MIN_NB_MB_Q15,
};

static NLSF_CB_WB: NlsfCb = NlsfCb {
    n_vectors: 32,
    order: 16,
    quant_step_size_q16: 9830,
    cb1_nlsf_q8: &NLSF_CB1_WB_Q8,
    cb1_wght_q9: &NLSF_CB1_WB_WGHT_Q9,
    cb1_icdf: &NLSF_CB1_ICDF_WB,
    pred_q8: &NLSF_PRED_WB_Q8,
    ec_sel: &NLSF_CB2_SELECT_WB,
    ec_icdf: &NLSF_CB2_ICDF_WB,
    delta_min_q15: &NLSF_DELTA_MIN_WB_Q15,
};

#[derive(Clone, Debug, Default)]
struct Indices {
    gains: [i32; MAX_NB_SUBFR],
    ltp: [i32; MAX_NB_SUBFR],
    nlsf: [i32; MAX_LPC_ORDER + 1],
    lag: i32,
    contour: i32,
    signal_type: i32,
    quant_offset_type: i32,
    nlsf_interp_coef_q2: i32,
    per: i32,
    ltp_scale: i32,
    seed: i32,
}

#[derive(Clone, Debug, Default)]
struct Plc {
    pitch_l_q8: i32,
    ltp_coef_q14: [i16; LTP_ORDER],
    prev_lpc_q12: [i16; MAX_LPC_ORDER],
    last_frame_lost: bool,
    rand_seed: i32,
    rand_scale_q14: i16,
    conc_energy: i32,
    conc_energy_shift: i32,
    prev_ltp_scale_q14: i16,
    prev_gain_q16: [i32; 2],
    fs_khz: i32,
    nb_subfr: usize,
    subfr_length: usize,
}

#[derive(Clone, Debug)]
struct Cng {
    exc_buf_q14: [i32; MAX_FRAME_LENGTH],
    smth_nlsf_q15: [i16; MAX_LPC_ORDER],
    synth_state: [i32; MAX_LPC_ORDER],
    smth_gain_q16: i32,
    rand_seed: i32,
    fs_khz: i32,
}

impl Default for Cng {
    fn default() -> Self {
        Cng {
            exc_buf_q14: [0; MAX_FRAME_LENGTH],
            smth_nlsf_q15: [0; MAX_LPC_ORDER],
            synth_state: [0; MAX_LPC_ORDER],
            smth_gain_q16: 0,
            rand_seed: 0,
            fs_khz: 0,
        }
    }
}

/// `silk_decoder_control`.
#[derive(Clone, Debug, Default)]
struct Ctrl {
    pitch_l: [i32; MAX_NB_SUBFR],
    gains_q16: [i32; MAX_NB_SUBFR],
    pred_coef_q12: [[i16; MAX_LPC_ORDER]; 2],
    ltp_coef_q14: [i16; LTP_ORDER * MAX_NB_SUBFR],
    ltp_scale_q14: i32,
}

/// `silk_decoder_state`.
#[derive(Clone, Debug)]
struct ChannelState {
    prev_gain_q16: i32,
    exc_q14: [i32; MAX_FRAME_LENGTH],
    slpc_q14_buf: [i32; MAX_LPC_ORDER],
    out_buf: [i16; MAX_FRAME_LENGTH + 2 * 80],
    lag_prev: i32,
    last_gain_index: i32,
    fs_khz: i32,
    fs_api_hz: i32,
    nb_subfr: usize,
    frame_length: usize,
    subfr_length: usize,
    ltp_mem_length: usize,
    lpc_order: usize,
    prev_nlsf_q15: [i16; MAX_LPC_ORDER],
    first_frame_after_reset: bool,
    pitch_lag_low_bits_icdf: &'static [u8],
    pitch_contour_icdf: &'static [u8],
    n_frames_decoded: usize,
    n_frames_per_packet: usize,
    ec_prev_signal_type: i32,
    ec_prev_lag_index: i32,
    vad_flags: [bool; 3],
    lbrr_flag: bool,
    lbrr_flags: [bool; 3],
    resampler: Resampler,
    nlsf_cb: &'static NlsfCb,
    indices: Indices,
    cng: Cng,
    loss_cnt: i32,
    prev_signal_type: i32,
    plc: Plc,
}

impl ChannelState {
    /// `silk_init_decoder`.
    fn new() -> Self {
        let mut s = ChannelState {
            prev_gain_q16: 65536,
            exc_q14: [0; MAX_FRAME_LENGTH],
            slpc_q14_buf: [0; MAX_LPC_ORDER],
            out_buf: [0; MAX_FRAME_LENGTH + 2 * 80],
            lag_prev: 0,
            last_gain_index: 0,
            fs_khz: 0,
            fs_api_hz: 0,
            nb_subfr: 0,
            frame_length: 0,
            subfr_length: 0,
            ltp_mem_length: 0,
            lpc_order: 0,
            prev_nlsf_q15: [0; MAX_LPC_ORDER],
            first_frame_after_reset: true,
            pitch_lag_low_bits_icdf: &[],
            pitch_contour_icdf: &[],
            n_frames_decoded: 0,
            n_frames_per_packet: 0,
            ec_prev_signal_type: 0,
            ec_prev_lag_index: 0,
            vad_flags: [false; 3],
            lbrr_flag: false,
            lbrr_flags: [false; 3],
            resampler: Resampler::default(),
            nlsf_cb: &NLSF_CB_NB_MB,
            indices: Indices::default(),
            cng: Cng::default(),
            loss_cnt: 0,
            prev_signal_type: 0,
            plc: Plc::default(),
        };
        s.cng_reset();
        s.plc_reset();
        s
    }

    fn set_fs(&mut self, fs_khz: i32, fs_api_hz: i32) {
        self.subfr_length = 5 * fs_khz as usize;
        let frame_length = self.nb_subfr * self.subfr_length;
        if self.fs_khz != fs_khz || self.fs_api_hz != fs_api_hz {
            self.resampler = Resampler::new(fs_khz * 1000, fs_api_hz);
            self.fs_api_hz = fs_api_hz;
        }
        if self.fs_khz != fs_khz || frame_length != self.frame_length {
            let full = self.nb_subfr == MAX_NB_SUBFR;
            self.pitch_contour_icdf = match (fs_khz == 8, full) {
                (true, true) => &PITCH_CONTOUR_NB_ICDF,
                (true, false) => &PITCH_CONTOUR_10_MS_NB_ICDF,
                (false, true) => &PITCH_CONTOUR_ICDF,
                (false, false) => &PITCH_CONTOUR_10_MS_ICDF,
            };
            if self.fs_khz != fs_khz {
                self.ltp_mem_length = 20 * fs_khz as usize;
                if fs_khz == 8 || fs_khz == 12 {
                    self.lpc_order = 10;
                    self.nlsf_cb = &NLSF_CB_NB_MB;
                } else {
                    self.lpc_order = 16;
                    self.nlsf_cb = &NLSF_CB_WB;
                }
                self.pitch_lag_low_bits_icdf = match fs_khz {
                    16 => &UNIFORM8_ICDF,
                    12 => &UNIFORM6_ICDF,
                    _ => &UNIFORM4_ICDF,
                };
                self.first_frame_after_reset = true;
                self.lag_prev = 100;
                self.last_gain_index = 10;
                self.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
                self.out_buf.fill(0);
                self.slpc_q14_buf.fill(0);
            }
            self.fs_khz = fs_khz;
            self.frame_length = frame_length;
        }
    }

    fn decode_indices(&mut self, ec: &mut Ec, frame_index: usize, decode_lbrr: bool, cond: i32) {
        let ix = if decode_lbrr || self.vad_flags[frame_index] {
            ec.dec_icdf(&TYPE_OFFSET_VAD_ICDF, 8) + 2
        } else {
            ec.dec_icdf(&TYPE_OFFSET_NO_VAD_ICDF, 8)
        };
        let ind = &mut self.indices;
        ind.signal_type = ix >> 1;
        ind.quant_offset_type = ix & 1;
        if cond == CODE_CONDITIONALLY {
            ind.gains[0] = ec.dec_icdf(&DELTA_GAIN_ICDF, 8);
        } else {
            let st = ind.signal_type as usize;
            ind.gains[0] = ec.dec_icdf(&GAIN_ICDF[st * 8..st * 8 + 8], 8) << 3;
            ind.gains[0] += ec.dec_icdf(&UNIFORM8_ICDF, 8);
        }
        for i in 1..self.nb_subfr {
            ind.gains[i] = ec.dec_icdf(&DELTA_GAIN_ICDF, 8);
        }
        let cb = self.nlsf_cb;
        let off = (ind.signal_type >> 1) as usize * cb.n_vectors;
        ind.nlsf[0] = ec.dec_icdf(&cb.cb1_icdf[off..], 8);
        let mut ec_ix = [0i16; MAX_LPC_ORDER];
        let mut pred_q8 = [0u8; MAX_LPC_ORDER];
        nlsf_unpack(&mut ec_ix, &mut pred_q8, cb, ind.nlsf[0] as usize);
        for i in 0..cb.order {
            let mut ix = ec.dec_icdf(&cb.ec_icdf[ec_ix[i] as usize..], 8);
            if ix == 0 {
                ix -= ec.dec_icdf(&NLSF_EXT_ICDF, 8);
            } else if ix == 8 {
                ix += ec.dec_icdf(&NLSF_EXT_ICDF, 8);
            }
            ind.nlsf[i + 1] = ix - 4;
        }
        ind.nlsf_interp_coef_q2 = if self.nb_subfr == MAX_NB_SUBFR {
            ec.dec_icdf(&NLSF_INTERPOLATION_FACTOR_ICDF, 8)
        } else {
            4
        };
        if ind.signal_type == TYPE_VOICED {
            let mut absolute = true;
            if cond == CODE_CONDITIONALLY && self.ec_prev_signal_type == TYPE_VOICED {
                let delta = ec.dec_icdf(&PITCH_DELTA_ICDF, 8);
                if delta > 0 {
                    ind.lag = (self.ec_prev_lag_index + delta - 9) as i16 as i32;
                    absolute = false;
                }
            }
            if absolute {
                ind.lag = ec.dec_icdf(&PITCH_LAG_ICDF, 8) * (self.fs_khz >> 1);
                ind.lag += ec.dec_icdf(self.pitch_lag_low_bits_icdf, 8);
            }
            self.ec_prev_lag_index = ind.lag;
            ind.contour = ec.dec_icdf(self.pitch_contour_icdf, 8);
            ind.per = ec.dec_icdf(&LTP_PER_INDEX_ICDF, 8);
            let icdf: &[u8] = match ind.per {
                0 => &LTP_GAIN_ICDF_0,
                1 => &LTP_GAIN_ICDF_1,
                _ => &LTP_GAIN_ICDF_2,
            };
            for k in 0..self.nb_subfr {
                ind.ltp[k] = ec.dec_icdf(icdf, 8);
            }
            ind.ltp_scale = if cond == CODE_INDEPENDENTLY {
                ec.dec_icdf(&LTPSCALE_ICDF, 8)
            } else {
                0
            };
        }
        self.ec_prev_signal_type = ind.signal_type;
        ind.seed = ec.dec_icdf(&UNIFORM4_ICDF, 8);
    }

    fn decode_parameters(&mut self, ctrl: &mut Ctrl, cond: i32) {
        gains_dequant(
            &mut ctrl.gains_q16,
            &self.indices.gains,
            &mut self.last_gain_index,
            cond == CODE_CONDITIONALLY,
            self.nb_subfr,
        );
        let order = self.lpc_order;
        let mut nlsf_q15 = [0i16; MAX_LPC_ORDER];
        nlsf_decode(&mut nlsf_q15, &self.indices.nlsf, self.nlsf_cb);
        nlsf2a(&mut ctrl.pred_coef_q12[1], &nlsf_q15, order);
        if self.first_frame_after_reset {
            self.indices.nlsf_interp_coef_q2 = 4;
        }
        if self.indices.nlsf_interp_coef_q2 < 4 {
            let mut nlsf0 = [0i16; MAX_LPC_ORDER];
            for i in 0..order {
                let p = self.prev_nlsf_q15[i] as i32;
                nlsf0[i] = (p
                    + ((self.indices.nlsf_interp_coef_q2 * (nlsf_q15[i] as i32 - p)) >> 2))
                    as i16;
            }
            nlsf2a(&mut ctrl.pred_coef_q12[0], &nlsf0, order);
        } else {
            ctrl.pred_coef_q12[0] = ctrl.pred_coef_q12[1];
        }
        self.prev_nlsf_q15[..order].copy_from_slice(&nlsf_q15[..order]);
        if self.loss_cnt != 0 {
            bwexpander(&mut ctrl.pred_coef_q12[0][..order], 63570);
            bwexpander(&mut ctrl.pred_coef_q12[1][..order], 63570);
        }
        if self.indices.signal_type == TYPE_VOICED {
            decode_pitch(
                self.indices.lag,
                self.indices.contour as usize,
                &mut ctrl.pitch_l,
                self.fs_khz,
                self.nb_subfr,
            );
            let cbk: &[i8] = match self.indices.per {
                0 => &LTP_GAIN_VQ_0,
                1 => &LTP_GAIN_VQ_1,
                _ => &LTP_GAIN_VQ_2,
            };
            for k in 0..self.nb_subfr {
                let ix = self.indices.ltp[k] as usize;
                for i in 0..LTP_ORDER {
                    ctrl.ltp_coef_q14[k * LTP_ORDER + i] =
                        ((cbk[ix * LTP_ORDER + i] as i32) << 7) as i16;
                }
            }
            ctrl.ltp_scale_q14 = LTPSCALES_TABLE_Q14[self.indices.ltp_scale as usize] as i32;
        } else {
            ctrl.pitch_l[..self.nb_subfr].fill(0);
            ctrl.ltp_coef_q14[..LTP_ORDER * self.nb_subfr].fill(0);
            self.indices.per = 0;
            ctrl.ltp_scale_q14 = 0;
        }
    }

    fn decode_core(&mut self, ctrl: &mut Ctrl, xq: &mut [i16], pulses: &[i16]) {
        let ltp_mem = self.ltp_mem_length;
        let mut s_ltp = [0i16; MAX_FRAME_LENGTH];
        let s_ltp = &mut s_ltp[..ltp_mem];
        let mut s_ltp_q15 = [0i32; 2 * MAX_FRAME_LENGTH];
        let s_ltp_q15 = &mut s_ltp_q15[..ltp_mem + self.frame_length];
        let mut res_q14 = [0i32; MAX_SUBFR_LENGTH];
        let res_q14 = &mut res_q14[..self.subfr_length];
        let mut s_lpc_q14 = [0i32; MAX_SUBFR_LENGTH + MAX_LPC_ORDER];
        let s_lpc_q14 = &mut s_lpc_q14[..self.subfr_length + MAX_LPC_ORDER];
        let st = self.indices.signal_type;
        let offset_q10 = QUANTIZATION_OFFSETS_Q10
            [((st >> 1) * 2 + self.indices.quant_offset_type) as usize]
            as i32;
        let nlsf_interpolation_flag = self.indices.nlsf_interp_coef_q2 < 4;
        let mut rand_seed = self.indices.seed;
        for i in 0..self.frame_length {
            rand_seed = silk_rand(rand_seed);
            let mut e = (pulses[i] as i32) << 14;
            if e > 0 {
                e -= 80 << 4;
            } else if e < 0 {
                e += 80 << 4;
            }
            e += offset_q10 << 4;
            if rand_seed < 0 {
                e = -e;
            }
            self.exc_q14[i] = e;
            rand_seed = rand_seed.wrapping_add(pulses[i] as i32);
        }
        s_lpc_q14[..MAX_LPC_ORDER].copy_from_slice(&self.slpc_q14_buf);
        let mut pexc = 0usize;
        let mut pxq = 0usize;
        let mut s_ltp_buf_idx = ltp_mem;
        let order = self.lpc_order;
        let sub = self.subfr_length;
        let mut lag = 0i32;
        for k in 0..self.nb_subfr {
            let a_q12 = ctrl.pred_coef_q12[k >> 1];
            let mut signal_type = st;
            let gain_q10 = ctrl.gains_q16[k] >> 6;
            let mut inv_gain_q31 = inverse32_varq(ctrl.gains_q16[k], 47);
            let gain_adj_q16 = if ctrl.gains_q16[k] != self.prev_gain_q16 {
                let g = div32_varq(self.prev_gain_q16, ctrl.gains_q16[k], 16);
                for v in s_lpc_q14[..MAX_LPC_ORDER].iter_mut() {
                    *v = smulww(g, *v);
                }
                g
            } else {
                1 << 16
            };
            self.prev_gain_q16 = ctrl.gains_q16[k];
            if self.loss_cnt != 0
                && self.prev_signal_type == TYPE_VOICED
                && self.indices.signal_type != TYPE_VOICED
                && k < MAX_NB_SUBFR / 2
            {
                let b = &mut ctrl.ltp_coef_q14[k * LTP_ORDER..k * LTP_ORDER + LTP_ORDER];
                b.fill(0);
                b[LTP_ORDER / 2] = 4096;
                signal_type = TYPE_VOICED;
                ctrl.pitch_l[k] = self.lag_prev;
            }
            if signal_type == TYPE_VOICED {
                lag = ctrl.pitch_l[k];
                if k == 0 || (k == 2 && nlsf_interpolation_flag) {
                    let start_idx = ltp_mem - lag as usize - order - LTP_ORDER / 2;
                    if k == 2 {
                        self.out_buf[ltp_mem..ltp_mem + 2 * sub].copy_from_slice(&xq[..2 * sub]);
                    }
                    lpc_analysis_filter(
                        &mut s_ltp[start_idx..],
                        &self.out_buf[start_idx + k * sub..],
                        &a_q12,
                        ltp_mem - start_idx,
                        order,
                    );
                    if k == 0 {
                        inv_gain_q31 = smulwb(inv_gain_q31, ctrl.ltp_scale_q14) << 2;
                    }
                    for i in 0..(lag as usize + LTP_ORDER / 2) {
                        s_ltp_q15[s_ltp_buf_idx - i - 1] =
                            smulwb(inv_gain_q31, s_ltp[ltp_mem - i - 1] as i32);
                    }
                } else if gain_adj_q16 != 1 << 16 {
                    for i in 0..(lag as usize + LTP_ORDER / 2) {
                        let j = s_ltp_buf_idx - i - 1;
                        s_ltp_q15[j] = smulww(gain_adj_q16, s_ltp_q15[j]);
                    }
                }
            }
            let use_res = signal_type == TYPE_VOICED;
            if use_res {
                let b = &ctrl.ltp_coef_q14[k * LTP_ORDER..k * LTP_ORDER + LTP_ORDER];
                let pl0 = s_ltp_buf_idx + LTP_ORDER / 2 - lag as usize;
                for (i, pl) in (pl0..pl0 + sub).enumerate() {
                    let h = &s_ltp_q15[pl - 4..=pl];
                    let mut p = 2;
                    p = smlawb(p, h[4], b[0] as i32);
                    p = smlawb(p, h[3], b[1] as i32);
                    p = smlawb(p, h[2], b[2] as i32);
                    p = smlawb(p, h[1], b[3] as i32);
                    p = smlawb(p, h[0], b[4] as i32);
                    res_q14[i] = self.exc_q14[pexc + i].wrapping_add(p << 1);
                    s_ltp_q15[s_ltp_buf_idx] = res_q14[i] << 1;
                    s_ltp_buf_idx += 1;
                }
            }
            let r: &[i32] = if use_res {
                res_q14
            } else {
                &self.exc_q14[pexc..pexc + sub]
            };
            let xo = &mut xq[pxq..pxq + sub];
            if order == 16 {
                lpc_synthesis::<16>(s_lpc_q14, &a_q12, r, xo, gain_q10);
            } else {
                lpc_synthesis::<10>(s_lpc_q14, &a_q12, r, xo, gain_q10);
            }
            s_lpc_q14.copy_within(sub..sub + MAX_LPC_ORDER, 0);
            pexc += sub;
            pxq += sub;
        }
        self.slpc_q14_buf
            .copy_from_slice(&s_lpc_q14[..MAX_LPC_ORDER]);
    }

    fn plc_reset(&mut self) {
        self.plc.pitch_l_q8 = (self.frame_length as i32) << 7;
        self.plc.prev_gain_q16 = [65536, 65536];
        self.plc.subfr_length = 20;
        self.plc.nb_subfr = 2;
    }

    fn plc(&mut self, ctrl: &mut Ctrl, frame: &mut [i16], lost: bool) {
        if self.fs_khz != self.plc.fs_khz {
            self.plc_reset();
            self.plc.fs_khz = self.fs_khz;
        }
        if lost {
            self.plc_conceal(ctrl, frame);
            self.loss_cnt += 1;
        } else {
            self.plc_update(ctrl);
        }
    }

    fn plc_update(&mut self, ctrl: &Ctrl) {
        let nb = self.nb_subfr;
        self.prev_signal_type = self.indices.signal_type;
        let mut ltp_gain_q14 = 0i32;
        let plc = &mut self.plc;
        if self.indices.signal_type == TYPE_VOICED {
            let mut j = 0;
            while (j * self.subfr_length) < ctrl.pitch_l[nb - 1] as usize {
                if j == nb {
                    break;
                }
                let base = (nb - 1 - j) * LTP_ORDER;
                let temp: i32 = ctrl.ltp_coef_q14[base..base + LTP_ORDER]
                    .iter()
                    .map(|&v| v as i32)
                    .sum();
                if temp > ltp_gain_q14 {
                    ltp_gain_q14 = temp;
                    plc.ltp_coef_q14
                        .copy_from_slice(&ctrl.ltp_coef_q14[base..base + LTP_ORDER]);
                    plc.pitch_l_q8 = ctrl.pitch_l[nb - 1 - j] << 8;
                }
                j += 1;
            }
            plc.ltp_coef_q14 = [0; LTP_ORDER];
            plc.ltp_coef_q14[LTP_ORDER / 2] = ltp_gain_q14 as i16;
            if ltp_gain_q14 < 11469 {
                let scale_q10 = (11469 << 10) / ltp_gain_q14.max(1);
                for v in plc.ltp_coef_q14.iter_mut() {
                    *v = (smulbb(*v as i32, scale_q10) >> 10) as i16;
                }
            } else if ltp_gain_q14 > 15565 {
                let scale_q14 = (15565 << 14) / ltp_gain_q14.max(1);
                for v in plc.ltp_coef_q14.iter_mut() {
                    *v = (smulbb(*v as i32, scale_q14) >> 14) as i16;
                }
            }
        } else {
            plc.pitch_l_q8 = smulbb(self.fs_khz, 18) << 8;
            plc.ltp_coef_q14 = [0; LTP_ORDER];
        }
        let order = self.lpc_order;
        plc.prev_lpc_q12[..order].copy_from_slice(&ctrl.pred_coef_q12[1][..order]);
        plc.prev_ltp_scale_q14 = ctrl.ltp_scale_q14 as i16;
        plc.prev_gain_q16
            .copy_from_slice(&ctrl.gains_q16[nb - 2..nb]);
        plc.subfr_length = self.subfr_length;
        plc.nb_subfr = nb;
    }

    fn plc_conceal(&mut self, ctrl: &mut Ctrl, frame: &mut [i16]) {
        const HARM_ATT_Q15: [i32; 2] = [32440, 31130];
        const RAND_ATT_V_Q15: [i32; 2] = [31130, 26214];
        const RAND_ATT_UV_Q15: [i32; 2] = [32440, 29491];
        let ltp_mem = self.ltp_mem_length;
        let order = self.lpc_order;
        let mut s_ltp_q14 = [0i32; 2 * MAX_FRAME_LENGTH];
        let s_ltp_q14 = &mut s_ltp_q14[..ltp_mem + self.frame_length];
        let mut s_ltp = [0i16; MAX_FRAME_LENGTH];
        let s_ltp = &mut s_ltp[..ltp_mem];
        let prev_gain_q10 = [
            self.plc.prev_gain_q16[0] >> 6,
            self.plc.prev_gain_q16[1] >> 6,
        ];
        if self.first_frame_after_reset {
            self.plc.prev_lpc_q12 = [0; MAX_LPC_ORDER];
        }
        // silk_PLC_energy
        let sub = self.subfr_length;
        let mut exc_buf = [0i16; 2 * MAX_SUBFR_LENGTH];
        let exc_buf = &mut exc_buf[..2 * sub];
        for k in 0..2 {
            for i in 0..sub {
                exc_buf[k * sub + i] = sat16(
                    smulww(
                        self.exc_q14[i + (k + self.nb_subfr - 2) * sub],
                        prev_gain_q10[k],
                    ) >> 8,
                ) as i16;
            }
        }
        let (energy1, shift1) = sum_sqr_shift(&exc_buf[..sub]);
        let (energy2, shift2) = sum_sqr_shift(&exc_buf[sub..]);
        let rand_off = if (energy1 >> shift2) < (energy2 >> shift1) {
            0.max(
                (self.plc.nb_subfr as i32 - 1) * self.plc.subfr_length as i32
                    - RAND_BUF_SIZE as i32,
            )
        } else {
            0.max(self.plc.nb_subfr as i32 * self.plc.subfr_length as i32 - RAND_BUF_SIZE as i32)
        } as usize;
        let mut b_q14 = self.plc.ltp_coef_q14;
        let mut rand_scale_q14 = self.plc.rand_scale_q14;
        let att = (self.loss_cnt.min(1)) as usize;
        let harm_gain_q15 = HARM_ATT_Q15[att];
        let mut rand_gain_q15 = if self.prev_signal_type == TYPE_VOICED {
            RAND_ATT_V_Q15[att]
        } else {
            RAND_ATT_UV_Q15[att]
        };
        bwexpander(&mut self.plc.prev_lpc_q12[..order], 64881);
        let a_q12 = self.plc.prev_lpc_q12;
        if self.loss_cnt == 0 {
            rand_scale_q14 = 1 << 14;
            if self.prev_signal_type == TYPE_VOICED {
                for &b in b_q14.iter() {
                    rand_scale_q14 = (rand_scale_q14 as i32 - b as i32) as i16;
                }
                rand_scale_q14 = rand_scale_q14.max(3277);
                rand_scale_q14 = (smulbb(rand_scale_q14 as i32, self.plc.prev_ltp_scale_q14 as i32)
                    >> 14) as i16;
            } else {
                let inv_gain_q30 = lpc_inverse_pred_gain(&self.plc.prev_lpc_q12[..order]);
                let mut down = ((1i32 << 30) >> 3).min(inv_gain_q30);
                down = ((1i32 << 30) >> 8).max(down);
                down <<= 3;
                rand_gain_q15 = smulwb(down, rand_gain_q15) >> 14;
            }
        }
        let mut rand_seed = self.plc.rand_seed;
        let mut lag = rshift_round(self.plc.pitch_l_q8, 8);
        let mut s_ltp_buf_idx = ltp_mem;
        let idx = ltp_mem - lag as usize - order - LTP_ORDER / 2;
        lpc_analysis_filter(
            &mut s_ltp[idx..],
            &self.out_buf[idx..],
            &a_q12,
            ltp_mem - idx,
            order,
        );
        let mut inv_gain_q30 = inverse32_varq(self.plc.prev_gain_q16[1], 46);
        inv_gain_q30 = inv_gain_q30.min(i32::MAX >> 1);
        for i in idx + order..ltp_mem {
            s_ltp_q14[i] = smulwb(inv_gain_q30, s_ltp[i] as i32);
        }
        for _k in 0..self.nb_subfr {
            let pl0 = s_ltp_buf_idx + LTP_ORDER / 2 - lag as usize;
            for pl in pl0..pl0 + sub {
                let mut p = 2;
                p = smlawb(p, s_ltp_q14[pl], b_q14[0] as i32);
                p = smlawb(p, s_ltp_q14[pl - 1], b_q14[1] as i32);
                p = smlawb(p, s_ltp_q14[pl - 2], b_q14[2] as i32);
                p = smlawb(p, s_ltp_q14[pl - 3], b_q14[3] as i32);
                p = smlawb(p, s_ltp_q14[pl - 4], b_q14[4] as i32);
                rand_seed = silk_rand(rand_seed);
                let ri = ((rand_seed >> 25) & RAND_BUF_MASK) as usize;
                s_ltp_q14[s_ltp_buf_idx] =
                    smlawb(p, self.exc_q14[rand_off + ri], rand_scale_q14 as i32).wrapping_shl(2);
                s_ltp_buf_idx += 1;
            }
            for b in b_q14.iter_mut() {
                *b = (smulbb(harm_gain_q15, *b as i32) >> 15) as i16;
            }
            if self.indices.signal_type != TYPE_NO_VOICE_ACTIVITY {
                rand_scale_q14 = (smulbb(rand_scale_q14 as i32, rand_gain_q15) >> 15) as i16;
            }
            self.plc.pitch_l_q8 = smlawb(self.plc.pitch_l_q8, self.plc.pitch_l_q8, 655);
            self.plc.pitch_l_q8 = self.plc.pitch_l_q8.min(smulbb(18, self.fs_khz) << 8);
            lag = rshift_round(self.plc.pitch_l_q8, 8);
        }
        let base = ltp_mem - MAX_LPC_ORDER;
        s_ltp_q14[base..ltp_mem].copy_from_slice(&self.slpc_q14_buf);
        for i in 0..self.frame_length {
            let mut p = (order >> 1) as i32;
            for j in 0..order {
                p = smlawb(
                    p,
                    s_ltp_q14[base + MAX_LPC_ORDER + i - j - 1],
                    a_q12[j] as i32,
                );
            }
            let v = s_ltp_q14[base + MAX_LPC_ORDER + i].saturating_add(lshift_sat32(p, 4));
            s_ltp_q14[base + MAX_LPC_ORDER + i] = v;
            frame[i] = sat16(rshift_round(smulww(v, prev_gain_q10[1]), 8)) as i16;
        }
        let fl = self.frame_length;
        self.slpc_q14_buf
            .copy_from_slice(&s_ltp_q14[base + fl..base + fl + MAX_LPC_ORDER]);
        self.plc.ltp_coef_q14 = b_q14;
        self.plc.rand_seed = rand_seed;
        self.plc.rand_scale_q14 = rand_scale_q14;
        ctrl.pitch_l = [lag; MAX_NB_SUBFR];
    }

    fn plc_glue_frames(&mut self, frame: &mut [i16]) {
        let plc = &mut self.plc;
        if self.loss_cnt != 0 {
            let (e, s) = sum_sqr_shift(frame);
            plc.conc_energy = e;
            plc.conc_energy_shift = s;
            plc.last_frame_lost = true;
        } else {
            if plc.last_frame_lost {
                let (mut energy, energy_shift) = sum_sqr_shift(frame);
                if energy_shift > plc.conc_energy_shift {
                    plc.conc_energy >>= energy_shift - plc.conc_energy_shift;
                } else if energy_shift < plc.conc_energy_shift {
                    energy >>= plc.conc_energy_shift - energy_shift;
                }
                if energy > plc.conc_energy {
                    let lz = clz32(plc.conc_energy) - 1;
                    plc.conc_energy <<= lz;
                    energy >>= 0.max(24 - lz);
                    let frac_q24 = plc.conc_energy / energy.max(1);
                    let mut gain_q16 = sqrt_approx(frac_q24) << 4;
                    let mut slope_q16 = ((1i32 << 16) - gain_q16) / frame.len() as i32;
                    slope_q16 <<= 2;
                    for v in frame.iter_mut() {
                        *v = smulwb(gain_q16, *v as i32) as i16;
                        gain_q16 += slope_q16;
                        if gain_q16 > 1 << 16 {
                            break;
                        }
                    }
                }
            }
            plc.last_frame_lost = false;
        }
    }

    fn cng_reset(&mut self) {
        let step = i16::MAX as i32 / (self.lpc_order as i32 + 1);
        let mut acc = 0;
        for i in 0..self.lpc_order {
            acc += step;
            self.cng.smth_nlsf_q15[i] = acc as i16;
        }
        self.cng.smth_gain_q16 = 0;
        self.cng.rand_seed = 3_176_576;
    }

    fn cng(&mut self, ctrl: &Ctrl, frame: &mut [i16]) {
        let length = frame.len();
        if self.fs_khz != self.cng.fs_khz {
            self.cng_reset();
            self.cng.fs_khz = self.fs_khz;
        }
        let order = self.lpc_order;
        let sub = self.subfr_length;
        if self.loss_cnt == 0 && self.prev_signal_type == TYPE_NO_VOICE_ACTIVITY {
            for i in 0..order {
                let d = self.prev_nlsf_q15[i] as i32 - self.cng.smth_nlsf_q15[i] as i32;
                self.cng.smth_nlsf_q15[i] =
                    (self.cng.smth_nlsf_q15[i] as i32 + smulwb(d, 16348)) as i16;
            }
            let mut max_gain_q16 = 0;
            let mut subfr = 0;
            for i in 0..self.nb_subfr {
                if ctrl.gains_q16[i] > max_gain_q16 {
                    max_gain_q16 = ctrl.gains_q16[i];
                    subfr = i;
                }
            }
            self.cng
                .exc_buf_q14
                .copy_within(0..(self.nb_subfr - 1) * sub, sub);
            self.cng.exc_buf_q14[..sub]
                .copy_from_slice(&self.exc_q14[subfr * sub..subfr * sub + sub]);
            for i in 0..self.nb_subfr {
                self.cng.smth_gain_q16 += smulwb(ctrl.gains_q16[i] - self.cng.smth_gain_q16, 4634);
            }
        }
        if self.loss_cnt != 0 {
            let mut sig = [0i32; MAX_FRAME_LENGTH + MAX_LPC_ORDER];
            let sig = &mut sig[..length + MAX_LPC_ORDER];
            let mut gain_q16 = smulww(self.plc.rand_scale_q14 as i32, self.plc.prev_gain_q16[1]);
            let sg = self.cng.smth_gain_q16;
            if gain_q16 >= (1 << 21) || sg > (1 << 23) {
                gain_q16 = smultt(gain_q16, gain_q16);
                gain_q16 = smultt(sg, sg).wrapping_sub(gain_q16.wrapping_shl(5));
                gain_q16 = sqrt_approx(gain_q16).wrapping_shl(16);
            } else {
                gain_q16 = smulww(gain_q16, gain_q16);
                gain_q16 = smulww(sg, sg).wrapping_sub(gain_q16.wrapping_shl(5));
                gain_q16 = sqrt_approx(gain_q16).wrapping_shl(8);
            }
            let gain_q10 = gain_q16 >> 6;
            // silk_CNG_exc
            let mut exc_mask = CNG_BUF_MASK_MAX;
            while exc_mask > length {
                exc_mask >>= 1;
            }
            let mut seed = self.cng.rand_seed;
            for i in 0..length {
                seed = silk_rand(seed);
                let idx = ((seed >> 24) as usize) & exc_mask;
                sig[MAX_LPC_ORDER + i] = self.cng.exc_buf_q14[idx];
            }
            self.cng.rand_seed = seed;
            let mut a_q12 = [0i16; MAX_LPC_ORDER];
            nlsf2a(&mut a_q12, &self.cng.smth_nlsf_q15, order);
            sig[..MAX_LPC_ORDER].copy_from_slice(&self.cng.synth_state);
            for i in 0..length {
                let mut p = (order >> 1) as i32;
                for j in 0..order {
                    p = smlawb(p, sig[MAX_LPC_ORDER + i - j - 1], a_q12[j] as i32);
                }
                let v = sig[MAX_LPC_ORDER + i].saturating_add(lshift_sat32(p, 4));
                sig[MAX_LPC_ORDER + i] = v;
                frame[i] =
                    sat16(frame[i] as i32 + sat16(rshift_round(smulww(v, gain_q10), 8))) as i16;
            }
            self.cng
                .synth_state
                .copy_from_slice(&sig[length..length + MAX_LPC_ORDER]);
        } else {
            self.cng.synth_state[..order].fill(0);
        }
    }

    /// `silk_decode_frame`: writes `frame_length` samples to `out`.
    fn decode_frame(&mut self, ec: &mut Ec, out: &mut [i16], lost_flag: i32, cond: i32) -> usize {
        let l = self.frame_length;
        let mut ctrl = Ctrl::default();
        if lost_flag == FLAG_DECODE_NORMAL
            || (lost_flag == FLAG_DECODE_LBRR && self.lbrr_flags[self.n_frames_decoded])
        {
            let mut pulses = [0i16; MAX_FRAME_LENGTH];
            self.decode_indices(ec, self.n_frames_decoded, lost_flag != 0, cond);
            decode_pulses(
                ec,
                &mut pulses,
                self.indices.signal_type,
                self.indices.quant_offset_type,
                l,
            );
            self.decode_parameters(&mut ctrl, cond);
            self.decode_core(&mut ctrl, out, &pulses);
            self.plc(&mut ctrl, out, false);
            self.loss_cnt = 0;
            self.prev_signal_type = self.indices.signal_type;
            self.first_frame_after_reset = false;
        } else {
            self.indices.signal_type = self.prev_signal_type;
            self.plc(&mut ctrl, out, true);
        }
        let mv_len = self.ltp_mem_length - l;
        self.out_buf.copy_within(l..l + mv_len, 0);
        self.out_buf[mv_len..mv_len + l].copy_from_slice(&out[..l]);
        self.cng(&ctrl, &mut out[..l]);
        self.plc_glue_frames(&mut out[..l]);
        self.lag_prev = ctrl.pitch_l[self.nb_subfr - 1];
        l
    }
}

/// The short-term synthesis of `silk_decode_core` for one subframe: `s_lpc` holds the
/// `MAX_LPC_ORDER` previous outputs (Q14) followed by room for the `r.len()` new ones.
#[inline(always)]
pub(super) fn lpc_synthesis<const ORDER: usize>(
    s_lpc: &mut [i32],
    a_q12: &[i16; MAX_LPC_ORDER],
    r: &[i32],
    xq: &mut [i16],
    gain_q10: i32,
) {
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        let done = unsafe { super::simd::lpc_synthesis_avx2(s_lpc, a_q12, ORDER, r, xq, gain_q10) };
        lpc_synthesis_scalar::<ORDER>(s_lpc, a_q12, r, xq, gain_q10, done);
        return;
    }
    lpc_synthesis_scalar::<ORDER>(s_lpc, a_q12, r, xq, gain_q10, 0);
}

/// Scalar reference of [`lpc_synthesis`], from output `i0` on.
pub(super) fn lpc_synthesis_scalar<const ORDER: usize>(
    s_lpc: &mut [i32],
    a_q12: &[i16; MAX_LPC_ORDER],
    r: &[i32],
    xq: &mut [i16],
    gain_q10: i32,
    i0: usize,
) {
    let n = r.len();
    let s_lpc = &mut s_lpc[..MAX_LPC_ORDER + n];
    let xq = &mut xq[..n];
    // SMLAWB terms wrap like 32-bit adds, so the sum can take the newest output last: it is
    // the only term on the sample-to-sample dependency chain, and kept in a register.
    let mut prev = s_lpc[MAX_LPC_ORDER - 1 + i0];
    for i in i0..n {
        let h: &[i32; MAX_LPC_ORDER] = s_lpc[i..i + MAX_LPC_ORDER].try_into().unwrap();
        let mut p = (ORDER >> 1) as i32;
        for j in 1..ORDER {
            p = smlawb(p, h[MAX_LPC_ORDER - 1 - j], a_q12[j] as i32);
        }
        p = smlawb(p, prev, a_q12[0] as i32);
        let v = r[i].saturating_add(lshift_sat32(p, 4));
        s_lpc[MAX_LPC_ORDER + i] = v;
        prev = v;
        xq[i] = sat16(rshift_round(smulww(v, gain_q10), 8)) as i16;
    }
}

fn nlsf_unpack(ec_ix: &mut [i16], pred_q8: &mut [u8], cb: &NlsfCb, cb1_index: usize) {
    let sel = &cb.ec_sel[cb1_index * cb.order / 2..];
    for (n, i) in (0..cb.order).step_by(2).enumerate() {
        let entry = sel[n] as i32;
        ec_ix[i] = smulbb((entry >> 1) & 7, 9) as i16;
        pred_q8[i] = cb.pred_q8[i + (entry & 1) as usize * (cb.order - 1)];
        ec_ix[i + 1] = smulbb((entry >> 5) & 7, 9) as i16;
        pred_q8[i + 1] = cb.pred_q8[i + ((entry >> 4) & 1) as usize * (cb.order - 1) + 1];
    }
}

fn nlsf_decode(nlsf_q15: &mut [i16], indices: &[i32], cb: &NlsfCb) {
    let mut ec_ix = [0i16; MAX_LPC_ORDER];
    let mut pred_q8 = [0u8; MAX_LPC_ORDER];
    nlsf_unpack(&mut ec_ix, &mut pred_q8, cb, indices[0] as usize);
    let order = cb.order;
    let mut res_q10 = [0i16; MAX_LPC_ORDER];
    let mut out_q10 = 0i32;
    for i in (0..order).rev() {
        let pred_q10 = smulbb(out_q10, pred_q8[i] as i32) >> 8;
        out_q10 = indices[i + 1] << 10;
        if out_q10 > 0 {
            out_q10 -= 102;
        } else if out_q10 < 0 {
            out_q10 += 102;
        }
        out_q10 = smlawb(pred_q10, out_q10, cb.quant_step_size_q16);
        res_q10[i] = out_q10 as i16;
    }
    let base = indices[0] as usize * order;
    for i in 0..order {
        let w = cb.cb1_wght_q9[base + i] as i32;
        let tmp = (((res_q10[i] as i32) << 14) / w) + ((cb.cb1_nlsf_q8[base + i] as i32) << 7);
        nlsf_q15[i] = tmp.clamp(0, 32767) as i16;
    }
    nlsf_stabilize(&mut nlsf_q15[..order], cb.delta_min_q15, order);
}

fn gains_dequant(
    gain_q16: &mut [i32],
    ind: &[i32],
    prev_ind: &mut i32,
    conditional: bool,
    nb: usize,
) {
    for k in 0..nb {
        if k == 0 && !conditional {
            *prev_ind = ind[k].max(*prev_ind - 16);
        } else {
            let ind_tmp = ind[k] - 4;
            let thr = 2 * 36 - 64 + *prev_ind;
            if ind_tmp > thr {
                *prev_ind += (ind_tmp << 1) - thr;
            } else {
                *prev_ind += ind_tmp;
            }
        }
        *prev_ind = (*prev_ind).clamp(0, 63);
        gain_q16[k] = log2lin((smulwb(1_907_825, *prev_ind) + 2090).min(3967));
    }
}

fn decode_pitch(lag_index: i32, contour: usize, pitch_lags: &mut [i32], fs_khz: i32, nb: usize) {
    let (cb, size): (&[i8], usize) = if fs_khz == 8 {
        if nb == 4 {
            (&CB_LAGS_STAGE2, 11)
        } else {
            (&CB_LAGS_STAGE2_10_MS, 3)
        }
    } else if nb == 4 {
        (&CB_LAGS_STAGE3, 34)
    } else {
        (&CB_LAGS_STAGE3_10_MS, 12)
    };
    let min_lag = 2 * fs_khz;
    let max_lag = 18 * fs_khz;
    let lag = min_lag + lag_index;
    for k in 0..nb {
        pitch_lags[k] = (lag + cb[k * size + contour] as i32).clamp(min_lag, max_lag);
    }
}

fn shell_split(ec: &mut Ec, p: i32, table: &[u8]) -> (i16, i16) {
    if p > 0 {
        let c1 = ec.dec_icdf(&table[SHELL_CODE_TABLE_OFFSETS[p as usize] as usize..], 8);
        (c1 as i16, (p - c1) as i16)
    } else {
        (0, 0)
    }
}

fn shell_decoder(out: &mut [i16], ec: &mut Ec, pulses4: i32) {
    let (t0, t1, t2, t3) = (
        &SHELL_CODE_TABLE0,
        &SHELL_CODE_TABLE1,
        &SHELL_CODE_TABLE2,
        &SHELL_CODE_TABLE3,
    );
    let p3 = shell_split(ec, pulses4, t3);
    let mut p2 = [0i16; 4];
    let mut p1 = [0i16; 8];
    (p2[0], p2[1]) = shell_split(ec, p3.0 as i32, t2);
    (p1[0], p1[1]) = shell_split(ec, p2[0] as i32, t1);
    (out[0], out[1]) = shell_split(ec, p1[0] as i32, t0);
    (out[2], out[3]) = shell_split(ec, p1[1] as i32, t0);
    (p1[2], p1[3]) = shell_split(ec, p2[1] as i32, t1);
    (out[4], out[5]) = shell_split(ec, p1[2] as i32, t0);
    (out[6], out[7]) = shell_split(ec, p1[3] as i32, t0);
    (p2[2], p2[3]) = shell_split(ec, p3.1 as i32, t2);
    (p1[4], p1[5]) = shell_split(ec, p2[2] as i32, t1);
    (out[8], out[9]) = shell_split(ec, p1[4] as i32, t0);
    (out[10], out[11]) = shell_split(ec, p1[5] as i32, t0);
    (p1[6], p1[7]) = shell_split(ec, p2[3] as i32, t1);
    (out[12], out[13]) = shell_split(ec, p1[6] as i32, t0);
    (out[14], out[15]) = shell_split(ec, p1[7] as i32, t0);
}

fn decode_pulses(ec: &mut Ec, pulses: &mut [i16], signal_type: i32, qot: i32, frame_length: usize) {
    let rli = ec.dec_icdf(&RATE_LEVELS_ICDF[(signal_type >> 1) as usize * 9..], 8) as usize;
    let mut iter = frame_length >> 4;
    if iter * 16 < frame_length {
        iter += 1;
    }
    let mut sum_pulses = [0i32; 20];
    let mut n_lshifts = [0i32; 20];
    let cdf = &PULSES_PER_BLOCK_ICDF[rli * 18..rli * 18 + 18];
    for i in 0..iter {
        sum_pulses[i] = ec.dec_icdf(cdf, 8);
        while sum_pulses[i] == 17 {
            n_lshifts[i] += 1;
            let off = 9 * 18 + (n_lshifts[i] == 10) as usize;
            sum_pulses[i] = ec.dec_icdf(&PULSES_PER_BLOCK_ICDF[off..], 8);
        }
    }
    for i in 0..iter {
        let blk = &mut pulses[i * 16..i * 16 + 16];
        if sum_pulses[i] > 0 {
            shell_decoder(blk, ec, sum_pulses[i]);
        } else {
            blk.fill(0);
        }
    }
    for i in 0..iter {
        if n_lshifts[i] > 0 {
            let nls = n_lshifts[i];
            for k in 0..16 {
                let mut abs_q = pulses[i * 16 + k] as i32;
                for _ in 0..nls {
                    abs_q <<= 1;
                    abs_q += ec.dec_icdf(&LSB_ICDF, 8);
                }
                pulses[i * 16 + k] = abs_q as i16;
            }
            sum_pulses[i] |= nls << 5;
        }
    }
    // silk_decode_signs
    let mut icdf = [0u8, 0u8];
    let sign = &SIGN_ICDF[(7 * (qot + (signal_type << 1))) as usize..];
    let blocks = (frame_length + 8) >> 4;
    for i in 0..blocks {
        let p = sum_pulses[i];
        if p > 0 {
            icdf[0] = sign[(p & 0x1f).min(6) as usize];
            for j in 0..16 {
                let q = &mut pulses[i * 16 + j];
                if *q > 0 {
                    // ec_dec_icdf returns 0 or 1; silk_dec_map maps them to -1 and +1.
                    *q = (*q as i32 * ((ec.dec_icdf(&icdf, 8) << 1) - 1)) as i16;
                }
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
struct StereoDec {
    pred_prev_q13: [i16; 2],
    s_mid: [i16; 2],
    s_side: [i16; 2],
}

fn stereo_decode_pred(ec: &mut Ec, pred_q13: &mut [i32; 2]) {
    let n = ec.dec_icdf(&STEREO_PRED_JOINT_ICDF, 8);
    let mut ix = [[0i32; 3]; 2];
    ix[0][2] = n / 5;
    ix[1][2] = n - 5 * ix[0][2];
    for row in ix.iter_mut() {
        row[0] = ec.dec_icdf(&UNIFORM3_ICDF, 8);
        row[1] = ec.dec_icdf(&UNIFORM5_ICDF, 8);
    }
    for n in 0..2 {
        ix[n][0] += 3 * ix[n][2];
        let low_q13 = STEREO_PRED_QUANT_Q13[ix[n][0] as usize] as i32;
        let step_q13 = smulwb(
            STEREO_PRED_QUANT_Q13[ix[n][0] as usize + 1] as i32 - low_q13,
            6554,
        );
        pred_q13[n] = smlabb(low_q13, step_q13, 2 * ix[n][1] + 1);
    }
    pred_q13[0] -= pred_q13[1];
}

fn stereo_ms_to_lr(
    state: &mut StereoDec,
    x1: &mut [i16],
    x2: &mut [i16],
    pred_q13: &[i32; 2],
    fs_khz: i32,
    frame_length: usize,
) {
    x1[..2].copy_from_slice(&state.s_mid);
    x2[..2].copy_from_slice(&state.s_side);
    state
        .s_mid
        .copy_from_slice(&x1[frame_length..frame_length + 2]);
    state
        .s_side
        .copy_from_slice(&x2[frame_length..frame_length + 2]);
    let mut pred0 = state.pred_prev_q13[0] as i32;
    let mut pred1 = state.pred_prev_q13[1] as i32;
    let interp = (8 * fs_khz) as usize;
    let denom_q16 = (1i32 << 16) / (8 * fs_khz);
    let delta0 = rshift_round(
        smulbb(pred_q13[0] - state.pred_prev_q13[0] as i32, denom_q16),
        16,
    );
    let delta1 = rshift_round(
        smulbb(pred_q13[1] - state.pred_prev_q13[1] as i32, denom_q16),
        16,
    );
    for n in 0..frame_length {
        if n < interp {
            pred0 += delta0;
            pred1 += delta1;
        } else if n == interp {
            pred0 = pred_q13[0];
            pred1 = pred_q13[1];
        }
        let mut sum = ((x1[n] as i32 + x1[n + 2] as i32) + ((x1[n + 1] as i32) << 1)) << 9;
        sum = smlawb((x2[n + 1] as i32) << 8, sum, pred0);
        sum = smlawb(sum, (x1[n + 1] as i32) << 11, pred1);
        x2[n + 1] = sat16(rshift_round(sum, 8)) as i16;
    }
    state.pred_prev_q13 = [pred_q13[0] as i16, pred_q13[1] as i16];
    for n in 0..frame_length {
        let sum = x1[n + 1] as i32 + x2[n + 1] as i32;
        let diff = x1[n + 1] as i32 - x2[n + 1] as i32;
        x1[n + 1] = sat16(sum) as i16;
        x2[n + 1] = sat16(diff) as i16;
    }
}

/// `silk_decoder` (both channel states plus the stereo unmixing state).
#[derive(Clone, Debug)]
pub struct SilkDecoder {
    ch: Box<[ChannelState; 2]>,
    stereo: StereoDec,
    n_channels_api: i32,
    n_channels_internal: i32,
    prev_decode_only_middle: bool,
}

impl Default for SilkDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl SilkDecoder {
    pub fn new() -> Self {
        SilkDecoder {
            ch: Box::new([ChannelState::new(), ChannelState::new()]),
            stereo: StereoDec::default(),
            n_channels_api: 0,
            n_channels_internal: 0,
            prev_decode_only_middle: false,
        }
    }

    /// `silk_InitDecoder`: keeps the API/internal channel counts, as libopus does.
    pub fn reset(&mut self) {
        self.ch[0] = ChannelState::new();
        self.ch[1] = ChannelState::new();
        self.stereo = StereoDec::default();
        self.prev_decode_only_middle = false;
    }

    /// `silk_Decode`: decodes one SILK frame (10 or 20 ms) into `out` (interleaved at the API
    /// rate and channel count) and returns the number of samples per channel.
    pub fn decode(
        &mut self,
        ctl: &mut DecControl,
        lost_flag: i32,
        new_packet: bool,
        ec: &mut Ec,
        out: &mut [i16],
    ) -> Result<usize, ()> {
        let nci = ctl.n_channels_internal as usize;
        if new_packet {
            for n in 0..nci {
                self.ch[n].n_frames_decoded = 0;
            }
        }
        if ctl.n_channels_internal > self.n_channels_internal {
            self.ch[1] = ChannelState::new();
        }
        let stereo_to_mono = ctl.n_channels_internal == 1
            && self.n_channels_internal == 2
            && ctl.internal_sample_rate == 1000 * self.ch[0].fs_khz;
        if self.ch[0].n_frames_decoded == 0 {
            for n in 0..nci {
                let (fpp, nb) = match ctl.payload_size_ms {
                    0 | 10 => (1, 2),
                    20 => (1, 4),
                    40 => (2, 4),
                    60 => (3, 4),
                    _ => return Err(()),
                };
                self.ch[n].n_frames_per_packet = fpp;
                self.ch[n].nb_subfr = nb;
                let fs_khz_dec = (ctl.internal_sample_rate >> 10) + 1;
                if fs_khz_dec != 8 && fs_khz_dec != 12 && fs_khz_dec != 16 {
                    return Err(());
                }
                self.ch[n].set_fs(fs_khz_dec, ctl.api_sample_rate);
            }
        }
        if ctl.n_channels_api == 2
            && ctl.n_channels_internal == 2
            && (self.n_channels_api == 1 || self.n_channels_internal == 1)
        {
            self.stereo.pred_prev_q13 = [0; 2];
            self.stereo.s_side = [0; 2];
            self.ch[1].resampler = self.ch[0].resampler.clone();
        }
        self.n_channels_api = ctl.n_channels_api;
        self.n_channels_internal = ctl.n_channels_internal;
        if ctl.api_sample_rate > 48000 || ctl.api_sample_rate < 8000 {
            return Err(());
        }
        let mut ms_pred_q13 = [0i32; 2];
        let mut decode_only_middle = false;
        if lost_flag != FLAG_PACKET_LOST && self.ch[0].n_frames_decoded == 0 {
            for n in 0..nci {
                for i in 0..self.ch[n].n_frames_per_packet {
                    self.ch[n].vad_flags[i] = ec.dec_bit_logp(1);
                }
                self.ch[n].lbrr_flag = ec.dec_bit_logp(1);
            }
            for n in 0..nci {
                let c = &mut self.ch[n];
                c.lbrr_flags = [false; 3];
                if c.lbrr_flag {
                    if c.n_frames_per_packet == 1 {
                        c.lbrr_flags[0] = true;
                    } else {
                        let icdf: &[u8] = if c.n_frames_per_packet == 2 {
                            &LBRR_FLAGS_2_ICDF
                        } else {
                            &LBRR_FLAGS_3_ICDF
                        };
                        let sym = ec.dec_icdf(icdf, 8) + 1;
                        for i in 0..c.n_frames_per_packet {
                            c.lbrr_flags[i] = (sym >> i) & 1 != 0;
                        }
                    }
                }
            }
            if lost_flag == FLAG_DECODE_NORMAL {
                // Skip over the LBRR data of this packet.
                for i in 0..self.ch[0].n_frames_per_packet {
                    for n in 0..nci {
                        if self.ch[n].lbrr_flags[i] {
                            let mut pulses = [0i16; MAX_FRAME_LENGTH];
                            if nci == 2 && n == 0 {
                                stereo_decode_pred(ec, &mut ms_pred_q13);
                                if !self.ch[1].lbrr_flags[i] {
                                    decode_only_middle =
                                        ec.dec_icdf(&STEREO_ONLY_CODE_MID_ICDF, 8) != 0;
                                }
                            }
                            let cond = if i > 0 && self.ch[n].lbrr_flags[i - 1] {
                                CODE_CONDITIONALLY
                            } else {
                                CODE_INDEPENDENTLY
                            };
                            let c = &mut self.ch[n];
                            c.decode_indices(ec, i, true, cond);
                            decode_pulses(
                                ec,
                                &mut pulses,
                                c.indices.signal_type,
                                c.indices.quant_offset_type,
                                c.frame_length,
                            );
                        }
                    }
                }
            }
        }
        if nci == 2 {
            let fd = self.ch[0].n_frames_decoded;
            if lost_flag == FLAG_DECODE_NORMAL
                || (lost_flag == FLAG_DECODE_LBRR && self.ch[0].lbrr_flags[fd])
            {
                stereo_decode_pred(ec, &mut ms_pred_q13);
                if (lost_flag == FLAG_DECODE_NORMAL && !self.ch[1].vad_flags[fd])
                    || (lost_flag == FLAG_DECODE_LBRR && !self.ch[1].lbrr_flags[fd])
                {
                    decode_only_middle = ec.dec_icdf(&STEREO_ONLY_CODE_MID_ICDF, 8) != 0;
                } else {
                    decode_only_middle = false;
                }
            } else {
                ms_pred_q13 = [
                    self.stereo.pred_prev_q13[0] as i32,
                    self.stereo.pred_prev_q13[1] as i32,
                ];
            }
        }
        if nci == 2 && !decode_only_middle && self.prev_decode_only_middle {
            let c = &mut self.ch[1];
            c.out_buf.fill(0);
            c.slpc_q14_buf.fill(0);
            c.lag_prev = 100;
            c.last_gain_index = 10;
            c.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
            c.first_frame_after_reset = true;
        }
        let mut tmp = [[0i16; MAX_FRAME_LENGTH + 2]; 2];
        let has_side = if lost_flag == FLAG_DECODE_NORMAL {
            !decode_only_middle
        } else {
            !self.prev_decode_only_middle
                || (nci == 2
                    && lost_flag == FLAG_DECODE_LBRR
                    && self.ch[1].lbrr_flags[self.ch[1].n_frames_decoded])
        };
        let mut n_samples_out_dec = 0usize;
        for n in 0..nci {
            if n == 0 || has_side {
                let frame_index = self.ch[0].n_frames_decoded as i32 - n as i32;
                let cond = if frame_index <= 0 {
                    CODE_INDEPENDENTLY
                } else if lost_flag == FLAG_DECODE_LBRR {
                    if self.ch[n].lbrr_flags[frame_index as usize - 1] {
                        CODE_CONDITIONALLY
                    } else {
                        CODE_INDEPENDENTLY
                    }
                } else if n > 0 && self.prev_decode_only_middle {
                    CODE_INDEPENDENTLY_NO_LTP_SCALING
                } else {
                    CODE_CONDITIONALLY
                };
                n_samples_out_dec = self.ch[n].decode_frame(ec, &mut tmp[n][2..], lost_flag, cond);
            } else {
                tmp[n][2..2 + n_samples_out_dec].fill(0);
            }
            self.ch[n].n_frames_decoded += 1;
        }
        if ctl.n_channels_api == 2 && nci == 2 {
            let (a, b) = tmp.split_at_mut(1);
            stereo_ms_to_lr(
                &mut self.stereo,
                &mut a[0],
                &mut b[0],
                &ms_pred_q13,
                self.ch[0].fs_khz,
                n_samples_out_dec,
            );
        } else {
            tmp[0][..2].copy_from_slice(&self.stereo.s_mid);
            self.stereo
                .s_mid
                .copy_from_slice(&tmp[0][n_samples_out_dec..n_samples_out_dec + 2]);
        }
        let n_out =
            n_samples_out_dec * ctl.api_sample_rate as usize / (self.ch[0].fs_khz as usize * 1000);
        let api2 = ctl.n_channels_api == 2;
        // 20 ms at 48 kHz.
        let mut rs_buf = [0i16; 960];
        let rs_out = &mut rs_buf[..n_out];
        for n in 0..(ctl.n_channels_api as usize).min(nci) {
            let input = &tmp[n][1..1 + n_samples_out_dec];
            if api2 {
                self.ch[n].resampler.process(rs_out, input);
                for (o, &v) in out[n..].iter_mut().step_by(2).zip(rs_out.iter()) {
                    *o = v;
                }
            } else {
                self.ch[n].resampler.process(&mut out[..n_out], input);
            }
        }
        if api2 && nci == 1 {
            if stereo_to_mono {
                self.ch[1]
                    .resampler
                    .process(rs_out, &tmp[0][1..1 + n_samples_out_dec]);
                for (o, &v) in out[1..].iter_mut().step_by(2).zip(rs_out.iter()) {
                    *o = v;
                }
            } else {
                for o in out[..2 * n_out].as_chunks_mut::<2>().0 {
                    o[1] = o[0];
                }
            }
        }
        if self.ch[0].prev_signal_type == TYPE_VOICED {
            let mult = [6, 4, 3][((self.ch[0].fs_khz - 8) >> 2) as usize];
            ctl.prev_pitch_lag = self.ch[0].lag_prev * mult;
        } else {
            ctl.prev_pitch_lag = 0;
        }
        if lost_flag == FLAG_PACKET_LOST {
            for i in 0..self.n_channels_internal as usize {
                self.ch[i].last_gain_index = 10;
            }
        } else {
            self.prev_decode_only_middle = decode_only_middle;
        }
        Ok(n_out)
    }
}
