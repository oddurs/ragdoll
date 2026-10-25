//! **`0044`**: Pure-Rust ISO/IEC 14496-3 AAC-LC (`mp4a.40.2`) Audio Decoder & Encoder (`crates/ragdoll-codec/src/aac.rs`).
//!
//! - Parses and generates 2-byte `AudioSpecificConfig` (`esds` / MP4 / MKV) and 7-byte ADTS (`0xFFF` / MPEG-TS / `.aac`) headers.
//! - Complete ISO/IEC 14496-3 AAC-LC syntax: `ID_SCE`, `ID_CPE` (`common_window`, `ms_mask_present`), `ID_FIL`, `ID_DSE`, `ID_END`,
//!   `ics_info` (`ONLY_LONG`, `LONG_START`, `EIGHT_SHORT`, `LONG_STOP`, Sine & KBD windows), `section_data`, `scale_factor_data`,
//!   `pulse_data`, `tns_data`, and all 11 Noiseless Huffman Spectral Codebooks (`1..=11`, including `ESC_HCB` escape sequences).
//! - Synthesizes and analyzes 1024-sample frames via `ragdoll_simd::MdctPlan` (`N = 2048` long / `N = 256` short) and
//!   vectorized 50% TDAC overlap-add (`window_overlap_add_f32`).

#![allow(clippy::needless_range_loop)]

use ragdoll_core::{
    AudioFrame, BitReader, BitWriter, ChannelLayout, RagdollError, Rational64, Result,
    SampleFormat, Timestamp,
};
use ragdoll_simd::{MdctPlan, kbd_window_f32, sine_window_f32, window_overlap_add_f32};

use crate::pcm::convert_audio_frame;

/// Standard ISO/IEC 14496-3 sampling frequencies indexed by `samplingFrequencyIndex` (`0..=12`).
pub const AAC_SAMPLE_RATES: [u32; 13] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350,
];

/// Scale-factor band (`swb`) offsets for 1024-bin long windows at 48 kHz / 44.1 kHz (49 bands).
pub const SWB_OFFSET_1024_48: [u16; 50] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 108, 120, 132, 144, 160,
    176, 196, 216, 240, 264, 292, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704,
    736, 768, 800, 832, 864, 896, 928, 1024,
];

/// Scale-factor band (`swb`) offsets for 1024-bin long windows at 96 kHz / 88.2 kHz (41 bands).
pub const SWB_OFFSET_1024_96: [u16; 42] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 96, 108, 120, 132,
    144, 156, 172, 188, 212, 240, 276, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024,
];

/// Scale-factor band (`swb`) offsets for 1024-bin long windows at 32 kHz (51 bands).
pub const SWB_OFFSET_1024_32: [u16; 52] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 108, 120, 132, 144, 160,
    176, 196, 216, 240, 264, 292, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704,
    736, 768, 800, 832, 864, 896, 928, 960, 992, 1024,
];

/// Scale-factor band (`swb`) offsets for 1024-bin long windows at 24 kHz / 22.05 kHz (47 bands).
pub const SWB_OFFSET_1024_24: [u16; 48] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124, 136,
    148, 160, 172, 188, 204, 220, 240, 260, 284, 308, 336, 364, 396, 432, 468, 508, 552, 600, 652,
    704, 768, 832, 896, 960, 1024,
];

/// Scale-factor band (`swb`) offsets for 1024-bin long windows at 16 kHz / 12 kHz / 11.025 kHz (43 bands).
pub const SWB_OFFSET_1024_16: [u16; 44] = [
    0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 100, 112, 124, 136, 148, 160, 172, 184, 196, 212,
    228, 244, 260, 280, 300, 320, 344, 368, 396, 424, 456, 492, 532, 572, 616, 664, 716, 772, 832,
    896, 960, 1024,
];

/// Scale-factor band (`swb`) offsets for 128-bin short windows at 48 kHz / 44.1 kHz / 32 kHz (14 bands).
pub const SWB_OFFSET_128_48: [u16; 15] =
    [0, 4, 8, 12, 16, 20, 28, 36, 44, 56, 68, 80, 96, 112, 128];

/// Scale-factor band (`swb`) offsets for 128-bin short windows at 96 kHz / 88.2 kHz / 64 kHz (12 bands).
pub const SWB_OFFSET_128_96: [u16; 13] = [0, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 92, 128];

/// Scale-factor band (`swb`) offsets for 128-bin short windows at 24 kHz / 22.05 kHz (15 bands).
pub const SWB_OFFSET_128_24: [u16; 16] = [
    0, 4, 8, 12, 16, 20, 24, 28, 36, 44, 52, 64, 76, 92, 108, 128,
];

#[inline]
pub fn swb_offsets_long(sr_idx: u8) -> &'static [u16] {
    match sr_idx {
        0..=1 => &SWB_OFFSET_1024_96,
        2..=4 => &SWB_OFFSET_1024_48,
        5 => &SWB_OFFSET_1024_32,
        6..=7 => &SWB_OFFSET_1024_24,
        _ => &SWB_OFFSET_1024_16,
    }
}

#[inline]
pub fn swb_offsets_short(sr_idx: u8) -> &'static [u16] {
    match sr_idx {
        0..=2 => &SWB_OFFSET_128_96,
        3..=5 => &SWB_OFFSET_128_48,
        _ => &SWB_OFFSET_128_24,
    }
}

/// Returns the closest `samplingFrequencyIndex` (`0..=12`) for `sample_rate`.
pub fn sample_rate_to_index(sample_rate: u32) -> u8 {
    let mut best_idx = 3u8;
    let mut best_diff = u32::MAX;
    for (idx, &sr) in AAC_SAMPLE_RATES.iter().enumerate() {
        let diff = sample_rate.abs_diff(sr);
        if diff < best_diff {
            best_diff = diff;
            best_idx = idx as u8;
        }
    }
    best_idx
}

/// ISO/IEC 14496-3 `AudioSpecificConfig` (2 bytes for AAC-LC).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioSpecificConfig {
    /// Audio Object Type (`2` = AAC-LC).
    pub audio_object_type: u8,
    /// Sampling frequency index (`0..=12`, e.g. `3` = 48 kHz, `4` = 44.1 kHz).
    pub sampling_frequency_index: u8,
    /// Sample rate in Hz (`48_000`, `44_100`, etc.).
    pub sample_rate: u32,
    /// Channel configuration (`1` = Mono, `2` = Stereo, `6` = 5.1).
    pub channel_configuration: u8,
}

impl AudioSpecificConfig {
    pub fn new_aac_lc(sample_rate: u32, channels: u8) -> Self {
        let idx = sample_rate_to_index(sample_rate);
        Self {
            audio_object_type: 2,
            sampling_frequency_index: idx,
            sample_rate: AAC_SAMPLE_RATES[idx as usize],
            channel_configuration: channels.clamp(1, 2),
        }
    }

    /// Serializes the 2-byte `AudioSpecificConfig` for `esds` / MP4 / Matroska `CodecPrivate`.
    pub fn to_bytes(&self) -> [u8; 2] {
        let b0 =
            ((self.audio_object_type & 0x1F) << 3) | ((self.sampling_frequency_index >> 1) & 0x07);
        let b1 = ((self.sampling_frequency_index & 0x01) << 7)
            | ((self.channel_configuration & 0x0F) << 3);
        [b0, b1]
    }

    /// Parses an `AudioSpecificConfig` from `bytes`.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 2 {
            return Err(RagdollError::BitstreamEndOfData);
        }
        let mut br = BitReader::new(bytes);
        let mut aot = br.read_bits(5)? as u8;
        if aot == 31 {
            aot = 32 + (br.read_bits(6)? as u8);
        }
        let sr_idx = br.read_bits(4)? as u8;
        let sample_rate = if sr_idx == 15 {
            br.read_bits(24)? as u32
        } else if (sr_idx as usize) < AAC_SAMPLE_RATES.len() {
            AAC_SAMPLE_RATES[sr_idx as usize]
        } else {
            48_000
        };
        let ch_cfg = br.read_bits(4)? as u8;
        Ok(Self {
            audio_object_type: aot,
            sampling_frequency_index: sr_idx.min(12),
            sample_rate,
            channel_configuration: ch_cfg.max(1),
        })
    }
}

/// 7-byte (or 9-byte with CRC) ISO/IEC 13818-7 ADTS frame header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdtsHeader {
    pub profile: u8,
    pub sampling_frequency_index: u8,
    pub channel_configuration: u8,
    pub frame_length: u16,
    pub header_len: usize,
}

impl AdtsHeader {
    pub fn new(asc: &AudioSpecificConfig, raw_payload_len: usize) -> Self {
        let frame_length = (7 + raw_payload_len) as u16;
        Self {
            profile: asc.audio_object_type.saturating_sub(1),
            sampling_frequency_index: asc.sampling_frequency_index,
            channel_configuration: asc.channel_configuration,
            frame_length,
            header_len: 7,
        }
    }

    pub fn to_bytes(&self) -> [u8; 7] {
        let mut out = [0u8; 7];
        let fl = self.frame_length & 0x1FFF;
        out[0] = 0xFF;
        out[1] = 0xF1; // MPEG-4, Layer 0, protection_absent = 1
        out[2] = ((self.profile & 0x03) << 6)
            | ((self.sampling_frequency_index & 0x0F) << 2)
            | ((self.channel_configuration >> 2) & 0x01);
        out[3] = ((self.channel_configuration & 0x03) << 6) | ((fl >> 11) as u8 & 0x03);
        out[4] = ((fl >> 3) & 0xFF) as u8;
        out[5] = (((fl & 0x07) as u8) << 5) | 0x1F;
        out[6] = 0xFC;
        out
    }

    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 7 || bytes[0] != 0xFF || (bytes[1] & 0xF0) != 0xF0 {
            return Err(RagdollError::InvalidData("invalid ADTS syncword".into()));
        }
        let protection_absent = (bytes[1] & 0x01) != 0;
        let profile = (bytes[2] >> 6) & 0x03;
        let sr_idx = (bytes[2] >> 2) & 0x0F;
        let ch_cfg = ((bytes[2] & 0x01) << 2) | ((bytes[3] >> 6) & 0x03);
        let frame_length = (((bytes[3] as u16) & 0x03) << 11)
            | ((bytes[4] as u16) << 3)
            | (((bytes[5] as u16) >> 5) & 0x07);
        let header_len = if protection_absent { 7 } else { 9 };
        Ok(Self {
            profile,
            sampling_frequency_index: sr_idx.min(12),
            channel_configuration: ch_cfg.max(1),
            frame_length,
            header_len,
        })
    }
}
pub const HCB_SF_CODES: [u32; 121] = [
    262120, 262118, 262119, 262117, 524277, 524273, 524269, 524278, 524270, 524271, 524272, 524284,
    524285, 524287, 524286, 524279, 524280, 524283, 524281, 262116, 524282, 262115, 131055, 131056,
    65525, 131054, 65522, 65523, 65524, 65521, 32758, 32759, 16377, 16373, 16375, 16371, 16374,
    16370, 8183, 8181, 4089, 4087, 4086, 2041, 4084, 2040, 1017, 1015, 1013, 504, 503, 250, 248,
    246, 121, 58, 56, 26, 11, 4, 0, 10, 12, 27, 57, 59, 120, 122, 247, 249, 502, 505, 1012, 1014,
    1016, 2037, 2036, 2038, 2039, 4085, 4088, 8180, 8182, 8184, 16376, 16372, 65520, 32756, 65526,
    32757, 262114, 524249, 524250, 524251, 524252, 524253, 524254, 524248, 524242, 524243, 524244,
    524245, 524246, 524274, 524255, 524263, 524264, 524265, 524266, 524267, 524262, 524256, 524257,
    524258, 524259, 524260, 524261, 524247, 524268, 524276, 524275,
];
pub const HCB_SF_BITS: [u8; 121] = [
    18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 18, 19, 18, 17, 17,
    16, 17, 16, 16, 16, 16, 15, 15, 14, 14, 14, 14, 14, 14, 13, 13, 12, 12, 12, 11, 12, 11, 10, 10,
    10, 9, 9, 8, 8, 8, 7, 6, 6, 5, 4, 3, 1, 4, 4, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 10, 11, 11,
    11, 11, 12, 12, 13, 13, 13, 14, 14, 16, 15, 16, 15, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
];
pub const HCB_1_CODES: [u16; 81] = [
    2040, 497, 2045, 1013, 104, 1008, 2039, 492, 2037, 1009, 114, 1012, 116, 17, 118, 491, 108,
    1014, 2044, 481, 2033, 496, 97, 502, 2034, 490, 2043, 498, 105, 493, 119, 23, 111, 486, 100,
    485, 103, 21, 98, 18, 0, 20, 101, 22, 109, 489, 99, 484, 107, 19, 113, 483, 112, 499, 2046,
    487, 2035, 495, 96, 494, 2032, 482, 2042, 1011, 106, 488, 117, 16, 115, 500, 110, 1015, 2038,
    480, 2041, 1010, 102, 501, 2047, 503, 2036,
];
pub const HCB_1_BITS: [u8; 81] = [
    11, 9, 11, 10, 7, 10, 11, 9, 11, 10, 7, 10, 7, 5, 7, 9, 7, 10, 11, 9, 11, 9, 7, 9, 11, 9, 11,
    9, 7, 9, 7, 5, 7, 9, 7, 9, 7, 5, 7, 5, 1, 5, 7, 5, 7, 9, 7, 9, 7, 5, 7, 9, 7, 9, 11, 9, 11, 9,
    7, 9, 11, 9, 11, 10, 7, 9, 7, 5, 7, 9, 7, 10, 11, 9, 11, 10, 7, 9, 11, 9, 11,
];
pub const HCB_2_CODES: [u16; 81] = [
    499, 111, 509, 235, 35, 234, 503, 232, 506, 242, 45, 112, 32, 6, 43, 110, 40, 233, 505, 102,
    248, 231, 27, 241, 500, 107, 501, 236, 42, 108, 44, 10, 39, 103, 26, 245, 36, 8, 31, 9, 0, 7,
    29, 11, 48, 239, 28, 100, 30, 12, 41, 243, 47, 240, 508, 113, 498, 244, 33, 230, 247, 104, 504,
    238, 34, 101, 49, 2, 38, 237, 37, 106, 507, 114, 510, 105, 46, 246, 511, 109, 502,
];
pub const HCB_2_BITS: [u8; 81] = [
    9, 7, 9, 8, 6, 8, 9, 8, 9, 8, 6, 7, 6, 5, 6, 7, 6, 8, 9, 7, 8, 8, 6, 8, 9, 7, 9, 8, 6, 7, 6, 5,
    6, 7, 6, 8, 6, 5, 6, 5, 3, 5, 6, 5, 6, 8, 6, 7, 6, 5, 6, 8, 6, 8, 9, 7, 9, 8, 6, 8, 8, 7, 9, 8,
    6, 7, 6, 4, 6, 8, 6, 7, 9, 7, 9, 7, 6, 8, 9, 7, 9,
];
pub const HCB_3_CODES: [u16; 81] = [
    0, 9, 239, 11, 25, 240, 491, 486, 1010, 10, 53, 495, 52, 55, 489, 493, 487, 1011, 494, 1005,
    8186, 492, 498, 2041, 2040, 1016, 4088, 8, 56, 1014, 54, 117, 1009, 1003, 1004, 4084, 24, 118,
    2036, 57, 116, 1007, 499, 500, 2038, 488, 1002, 8188, 242, 497, 4091, 1013, 2035, 4092, 238,
    1015, 32766, 496, 2037, 32765, 8187, 16378, 65535, 241, 1008, 16380, 490, 1006, 16379, 4086,
    4090, 32764, 2034, 4085, 65534, 1012, 2039, 32763, 4087, 4089, 32762,
];
pub const HCB_3_BITS: [u8; 81] = [
    1, 4, 8, 4, 5, 8, 9, 9, 10, 4, 6, 9, 6, 6, 9, 9, 9, 10, 9, 10, 13, 9, 9, 11, 11, 10, 12, 4, 6,
    10, 6, 7, 10, 10, 10, 12, 5, 7, 11, 6, 7, 10, 9, 9, 11, 9, 10, 13, 8, 9, 12, 10, 11, 12, 8, 10,
    15, 9, 11, 15, 13, 14, 16, 8, 10, 14, 9, 10, 14, 12, 12, 15, 11, 12, 16, 10, 11, 15, 12, 12,
    15,
];
pub const HCB_4_CODES: [u16; 81] = [
    7, 22, 246, 24, 8, 239, 495, 243, 2040, 25, 23, 237, 21, 1, 226, 240, 112, 1008, 494, 241,
    2042, 238, 228, 1010, 2038, 1007, 2045, 5, 20, 242, 9, 4, 229, 244, 232, 1012, 6, 2, 231, 3, 0,
    107, 227, 105, 499, 235, 230, 1014, 110, 106, 500, 1004, 496, 1017, 245, 236, 2043, 234, 111,
    1015, 2041, 1011, 4095, 233, 109, 1016, 108, 104, 501, 1006, 498, 2036, 2039, 1009, 4094, 1005,
    497, 2037, 2046, 1013, 2044,
];
pub const HCB_4_BITS: [u8; 81] = [
    4, 5, 8, 5, 4, 8, 9, 8, 11, 5, 5, 8, 5, 4, 8, 8, 7, 10, 9, 8, 11, 8, 8, 10, 11, 10, 11, 4, 5,
    8, 4, 4, 8, 8, 8, 10, 4, 4, 8, 4, 4, 7, 8, 7, 9, 8, 8, 10, 7, 7, 9, 10, 9, 10, 8, 8, 11, 8, 7,
    10, 11, 10, 12, 8, 7, 10, 7, 7, 9, 10, 9, 11, 11, 10, 12, 10, 9, 11, 11, 10, 11,
];
pub const HCB_5_CODES: [u16; 81] = [
    8191, 4087, 2036, 2024, 1009, 2030, 2041, 4088, 8189, 4093, 2033, 1000, 488, 240, 492, 1006,
    2034, 4090, 4084, 1007, 498, 232, 112, 236, 496, 1002, 2035, 2027, 491, 234, 26, 8, 25, 238,
    495, 2029, 1008, 242, 115, 11, 0, 10, 113, 243, 2025, 2031, 494, 239, 24, 9, 27, 235, 489,
    2028, 2038, 1003, 499, 237, 114, 233, 497, 1005, 2039, 4086, 2032, 1001, 493, 241, 490, 1004,
    2040, 4089, 8188, 4092, 4085, 2026, 1011, 1010, 2037, 4091, 8190,
];
pub const HCB_5_BITS: [u8; 81] = [
    13, 12, 11, 11, 10, 11, 11, 12, 13, 12, 11, 10, 9, 8, 9, 10, 11, 12, 12, 10, 9, 8, 7, 8, 9, 10,
    11, 11, 9, 8, 5, 4, 5, 8, 9, 11, 10, 8, 7, 4, 1, 4, 7, 8, 11, 11, 9, 8, 5, 4, 5, 8, 9, 11, 11,
    10, 9, 8, 7, 8, 9, 10, 11, 12, 11, 10, 9, 8, 9, 10, 11, 12, 13, 12, 12, 11, 10, 10, 11, 12, 13,
];
pub const HCB_6_CODES: [u16; 81] = [
    2046, 1021, 497, 491, 500, 490, 496, 1020, 2045, 1014, 485, 234, 108, 113, 104, 240, 486, 1015,
    499, 239, 50, 39, 40, 38, 49, 235, 503, 488, 111, 46, 8, 4, 6, 41, 107, 494, 495, 114, 45, 2,
    0, 3, 47, 115, 506, 487, 110, 43, 7, 1, 5, 44, 109, 492, 505, 238, 48, 36, 42, 37, 51, 236,
    498, 1016, 484, 237, 106, 112, 105, 116, 241, 1018, 2047, 1017, 502, 493, 504, 489, 501, 1019,
    2044,
];
pub const HCB_6_BITS: [u8; 81] = [
    11, 10, 9, 9, 9, 9, 9, 10, 11, 10, 9, 8, 7, 7, 7, 8, 9, 10, 9, 8, 6, 6, 6, 6, 6, 8, 9, 9, 7, 6,
    4, 4, 4, 6, 7, 9, 9, 7, 6, 4, 4, 4, 6, 7, 9, 9, 7, 6, 4, 4, 4, 6, 7, 9, 9, 8, 6, 6, 6, 6, 6, 8,
    9, 10, 9, 8, 7, 7, 7, 7, 8, 10, 11, 10, 9, 9, 9, 9, 9, 10, 11,
];
pub const HCB_7_CODES: [u16; 64] = [
    0, 5, 55, 116, 242, 491, 1005, 2039, 4, 12, 53, 113, 236, 238, 494, 501, 54, 52, 114, 234, 241,
    489, 499, 1013, 115, 112, 235, 240, 497, 496, 1004, 1018, 243, 237, 488, 495, 1007, 1009, 1017,
    2043, 493, 239, 490, 498, 1011, 1016, 2041, 2044, 1006, 492, 500, 1012, 1015, 2040, 4093, 4094,
    2038, 1008, 1010, 1014, 2042, 2045, 4092, 4095,
];
pub const HCB_7_BITS: [u8; 64] = [
    1, 3, 6, 7, 8, 9, 10, 11, 3, 4, 6, 7, 8, 8, 9, 9, 6, 6, 7, 8, 8, 9, 9, 10, 7, 7, 8, 8, 9, 9,
    10, 10, 8, 8, 9, 9, 10, 10, 10, 11, 9, 8, 9, 9, 10, 10, 11, 11, 10, 9, 9, 10, 10, 11, 12, 12,
    11, 10, 10, 10, 11, 11, 12, 12,
];
pub const HCB_8_CODES: [u16; 64] = [
    14, 5, 16, 48, 111, 241, 506, 1022, 3, 0, 4, 18, 44, 106, 117, 248, 15, 2, 6, 20, 46, 105, 114,
    245, 47, 17, 19, 42, 50, 108, 236, 250, 113, 43, 45, 49, 109, 112, 242, 505, 239, 104, 51, 107,
    110, 238, 249, 1020, 504, 116, 115, 237, 240, 246, 502, 509, 1021, 243, 244, 247, 503, 507,
    508, 1023,
];
pub const HCB_8_BITS: [u8; 64] = [
    5, 4, 5, 6, 7, 8, 9, 10, 4, 3, 4, 5, 6, 7, 7, 8, 5, 4, 4, 5, 6, 7, 7, 8, 6, 5, 5, 6, 6, 7, 8,
    8, 7, 6, 6, 6, 7, 7, 8, 9, 8, 7, 6, 7, 7, 8, 8, 10, 9, 7, 7, 8, 8, 8, 9, 9, 10, 8, 8, 8, 9, 9,
    9, 10,
];
pub const HCB_9_CODES: [u16; 169] = [
    0, 5, 55, 231, 478, 974, 985, 1992, 1997, 4040, 4061, 8164, 8172, 4, 12, 53, 114, 234, 237,
    482, 977, 979, 992, 2008, 4047, 4053, 54, 52, 113, 232, 236, 481, 975, 989, 987, 2000, 4039,
    4052, 4068, 230, 112, 233, 477, 483, 978, 988, 1996, 1994, 2014, 4056, 4074, 8155, 479, 235,
    476, 486, 981, 990, 1995, 2013, 2012, 4045, 4066, 4071, 8161, 976, 480, 484, 982, 1989, 2001,
    2011, 4050, 2016, 4057, 4075, 8163, 8169, 1988, 485, 983, 1990, 1999, 2010, 4043, 4058, 4067,
    4073, 8166, 8179, 8183, 2003, 984, 993, 2004, 2009, 4051, 4062, 8157, 8153, 8162, 8170, 8177,
    8182, 2002, 980, 986, 1991, 2007, 2018, 4046, 4059, 8152, 8174, 16368, 8180, 16370, 2017, 991,
    1993, 2006, 4042, 4048, 4069, 4070, 8171, 8175, 16371, 16372, 16373, 4064, 1998, 2005, 4038,
    4049, 4065, 8160, 8168, 8176, 16369, 16376, 16374, 32764, 4072, 2015, 4041, 4055, 4060, 8156,
    8159, 8173, 8181, 16377, 16379, 32765, 32766, 8167, 4044, 4054, 4063, 8158, 8154, 8165, 8178,
    16378, 16375, 16380, 16381, 32767,
];
pub const HCB_9_BITS: [u8; 169] = [
    1, 3, 6, 8, 9, 10, 10, 11, 11, 12, 12, 13, 13, 3, 4, 6, 7, 8, 8, 9, 10, 10, 10, 11, 12, 12, 6,
    6, 7, 8, 8, 9, 10, 10, 10, 11, 12, 12, 12, 8, 7, 8, 9, 9, 10, 10, 11, 11, 11, 12, 12, 13, 9, 8,
    9, 9, 10, 10, 11, 11, 11, 12, 12, 12, 13, 10, 9, 9, 10, 11, 11, 11, 12, 11, 12, 12, 13, 13, 11,
    9, 10, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 11, 10, 10, 11, 11, 12, 12, 13, 13, 13, 13, 13,
    13, 11, 10, 10, 11, 11, 11, 12, 12, 13, 13, 14, 13, 14, 11, 10, 11, 11, 12, 12, 12, 12, 13, 13,
    14, 14, 14, 12, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 12, 11, 12, 12, 12, 13, 13, 13,
    13, 14, 14, 15, 15, 13, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15,
];
pub const HCB_10_CODES: [u16; 169] = [
    34, 8, 29, 38, 95, 211, 463, 976, 983, 1005, 2032, 2038, 4093, 7, 0, 1, 9, 32, 84, 96, 213,
    220, 468, 973, 990, 2023, 28, 2, 6, 12, 30, 40, 91, 205, 217, 462, 476, 985, 1009, 37, 11, 10,
    13, 36, 87, 97, 204, 221, 460, 478, 979, 999, 93, 33, 31, 35, 39, 89, 100, 216, 223, 466, 482,
    989, 1006, 209, 85, 41, 86, 88, 98, 206, 224, 226, 474, 980, 995, 2027, 457, 94, 90, 92, 99,
    202, 218, 455, 458, 480, 987, 1000, 2028, 483, 210, 203, 208, 215, 219, 454, 469, 472, 970,
    986, 2026, 2033, 481, 212, 207, 214, 222, 225, 464, 470, 977, 981, 1010, 2030, 2043, 1001, 461,
    456, 459, 465, 471, 479, 975, 992, 1007, 2022, 2040, 4090, 1003, 477, 467, 473, 475, 978, 972,
    988, 1002, 2029, 2035, 2041, 4089, 2034, 974, 484, 971, 984, 982, 994, 997, 2024, 2036, 2037,
    2039, 4091, 2042, 1004, 991, 993, 996, 998, 1008, 2025, 2031, 4088, 4094, 4092, 4095,
];
pub const HCB_10_BITS: [u8; 169] = [
    6, 5, 6, 6, 7, 8, 9, 10, 10, 10, 11, 11, 12, 5, 4, 4, 5, 6, 7, 7, 8, 8, 9, 10, 10, 11, 6, 4, 5,
    5, 6, 6, 7, 8, 8, 9, 9, 10, 10, 6, 5, 5, 5, 6, 7, 7, 8, 8, 9, 9, 10, 10, 7, 6, 6, 6, 6, 7, 7,
    8, 8, 9, 9, 10, 10, 8, 7, 6, 7, 7, 7, 8, 8, 8, 9, 10, 10, 11, 9, 7, 7, 7, 7, 8, 8, 9, 9, 9, 10,
    10, 11, 9, 8, 8, 8, 8, 8, 9, 9, 9, 10, 10, 11, 11, 9, 8, 8, 8, 8, 8, 9, 9, 10, 10, 10, 11, 11,
    10, 9, 9, 9, 9, 9, 9, 10, 10, 10, 11, 11, 12, 10, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 12,
    11, 10, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 12, 11, 10, 10, 10, 10, 10, 10, 11, 11, 12, 12,
    12, 12,
];
pub const HCB_11_CODES: [u16; 289] = [
    0, 6, 25, 61, 156, 198, 423, 912, 962, 991, 2022, 2035, 4091, 2028, 4090, 4094, 910, 5, 1, 8,
    20, 55, 66, 146, 175, 401, 421, 437, 926, 960, 930, 973, 2006, 174, 23, 7, 9, 24, 57, 64, 142,
    163, 184, 409, 428, 449, 945, 918, 958, 970, 157, 60, 21, 22, 26, 59, 68, 145, 165, 190, 406,
    430, 441, 929, 913, 933, 981, 148, 154, 54, 56, 58, 65, 140, 155, 176, 195, 414, 427, 444, 927,
    911, 937, 975, 147, 191, 62, 63, 67, 69, 158, 167, 185, 404, 418, 442, 451, 934, 935, 955, 980,
    159, 416, 143, 141, 144, 152, 166, 182, 196, 415, 431, 447, 921, 959, 948, 969, 999, 168, 438,
    171, 164, 170, 178, 194, 197, 408, 420, 440, 908, 932, 964, 966, 989, 1000, 173, 943, 402, 189,
    188, 398, 407, 410, 419, 433, 909, 920, 951, 979, 977, 987, 2013, 180, 990, 425, 411, 412, 417,
    426, 429, 435, 907, 946, 952, 974, 993, 992, 2002, 2021, 183, 2019, 443, 424, 422, 432, 434,
    439, 923, 922, 954, 949, 982, 2007, 996, 2008, 2026, 186, 2024, 928, 445, 436, 906, 452, 914,
    938, 944, 956, 983, 2004, 2012, 2011, 2005, 2032, 193, 2043, 968, 931, 917, 925, 940, 942, 965,
    984, 994, 998, 2020, 2023, 2016, 2025, 2039, 400, 2034, 915, 446, 448, 916, 919, 941, 963, 961,
    978, 2010, 2009, 2015, 2027, 2036, 2042, 405, 2040, 957, 924, 939, 936, 947, 953, 976, 995,
    997, 2018, 2014, 2029, 2033, 2041, 2044, 403, 4093, 988, 950, 967, 972, 971, 985, 986, 2003,
    2017, 2030, 2031, 2037, 2038, 4092, 4095, 413, 450, 181, 161, 150, 151, 149, 153, 160, 162,
    172, 169, 177, 179, 187, 192, 399, 4,
];
pub const HCB_11_BITS: [u8; 289] = [
    4, 5, 6, 7, 8, 8, 9, 10, 10, 10, 11, 11, 12, 11, 12, 12, 10, 5, 4, 5, 6, 7, 7, 8, 8, 9, 9, 9,
    10, 10, 10, 10, 11, 8, 6, 5, 5, 6, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 8, 7, 6, 6, 6, 7, 7,
    8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 8, 8, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 8, 8,
    7, 7, 7, 7, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 8, 9, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 10, 10,
    10, 10, 10, 8, 9, 8, 8, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 10, 10, 8, 10, 9, 8, 8, 9, 9, 9,
    9, 9, 10, 10, 10, 10, 10, 10, 11, 8, 10, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 11, 11,
    8, 11, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 10, 11, 11, 8, 11, 10, 9, 9, 10, 9, 10, 10,
    10, 10, 10, 11, 11, 11, 11, 11, 8, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
    11, 9, 11, 10, 9, 9, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 9, 11, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 9, 12, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
    11, 12, 12, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 5,
];

/// Zero-point scale factor offset matching `MdctPlan` (`2048`-sample MDCT with `2/1024` IMDCT normalization)
/// and `libavcodec`'s `POW_SF2_ZERO = 200` (`200 - 36 = 164` for long blocks, `200 - 24 = 176` for short blocks).
const AAC_SF_OFFSET_LONG: f32 = 164.0;
const AAC_SF_OFFSET_SHORT: f32 = 176.0;

#[inline]
fn decode_huffman_u32(br: &mut BitReader, codes: &[u32], bits: &[u8]) -> Result<usize> {
    let mut acc = 0u32;
    for len in 1..=19u8 {
        acc = (acc << 1) | (br.read_bits(1)? as u32);
        for (idx, (&c, &b)) in codes.iter().zip(bits.iter()).enumerate() {
            if b == len && c == acc {
                return Ok(idx);
            }
        }
    }
    Err(RagdollError::InvalidData(
        "invalid AAC scalefactor Huffman codeword".into(),
    ))
}

#[inline]
fn decode_huffman_u16(br: &mut BitReader, codes: &[u16], bits: &[u8]) -> Result<usize> {
    let mut acc = 0u16;
    for len in 1..=16u8 {
        acc = (acc << 1) | (br.read_bits(1)? as u16);
        for (idx, (&c, &b)) in codes.iter().zip(bits.iter()).enumerate() {
            if b == len && c == acc {
                return Ok(idx);
            }
        }
    }
    Err(RagdollError::InvalidData(
        "invalid AAC spectral Huffman codeword".into(),
    ))
}

#[inline]
fn inv_quant(q: i32, gain: f32) -> f32 {
    if q == 0 {
        0.0
    } else {
        let a = (q.abs() as f32).powf(4.0 / 3.0) * gain;
        if q < 0 { -a } else { a }
    }
}

#[derive(Clone, Debug)]
struct IcsInfo {
    window_sequence: u8,
    window_shape: u8,
    max_sfb: usize,
    num_windows: usize,
    num_window_groups: usize,
    group_len: [u8; 8],
}

impl Default for IcsInfo {
    fn default() -> Self {
        Self {
            window_sequence: 0,
            window_shape: 0,
            max_sfb: 0,
            num_windows: 1,
            num_window_groups: 1,
            group_len: [1, 0, 0, 0, 0, 0, 0, 0],
        }
    }
}

impl IcsInfo {
    fn parse(br: &mut BitReader) -> Result<Self> {
        let _reserved = br.read_bits(1)?;
        let window_sequence = br.read_bits(2)? as u8;
        let window_shape = br.read_bits(1)? as u8;
        if window_sequence == 2 {
            // EIGHT_SHORT_SEQUENCE
            let max_sfb = br.read_bits(4)? as usize;
            let grouping = br.read_bits(7)? as u8;
            let mut num_window_groups = 1usize;
            let mut group_len = [0u8; 8];
            group_len[0] = 1;
            for i in 0..7 {
                if ((grouping >> (6 - i)) & 1) != 0 {
                    group_len[num_window_groups - 1] += 1;
                } else {
                    num_window_groups += 1;
                    group_len[num_window_groups - 1] = 1;
                }
            }
            Ok(Self {
                window_sequence,
                window_shape,
                max_sfb,
                num_windows: 8,
                num_window_groups,
                group_len,
            })
        } else {
            let max_sfb = br.read_bits(6)? as usize;
            let predictor_data_present = br.read_bits(1)?;
            if predictor_data_present != 0 {
                // Skip AAC Main predictor reset/flags if ever encountered
                let reset = br.read_bits(1)?;
                if reset != 0 {
                    let _ = br.read_bits(5)?;
                }
                let pred_max = max_sfb.min(40);
                for _ in 0..pred_max {
                    let _ = br.read_bits(1)?;
                }
            }
            Ok(Self {
                window_sequence,
                window_shape,
                max_sfb,
                num_windows: 1,
                num_window_groups: 1,
                group_len: [1, 0, 0, 0, 0, 0, 0, 0],
            })
        }
    }
}

#[derive(Clone, Debug)]
struct ChannelData {
    ics: IcsInfo,
    sfb_cb: [[u8; 64]; 8],
    sf: [[i32; 64]; 8],
    coeffs: [f32; 1024],
}

impl Default for ChannelData {
    fn default() -> Self {
        Self {
            ics: IcsInfo::default(),
            sfb_cb: [[0; 64]; 8],
            sf: [[0; 64]; 8],
            coeffs: [0.0; 1024],
        }
    }
}

/// Pure-Rust ISO/IEC 14496-3 AAC-LC Decoder (`1024` samples per frame).
#[derive(Clone, Debug)]
pub struct AacDecoder {
    pub asc: AudioSpecificConfig,
    mdct_long: MdctPlan,
    mdct_short: MdctPlan,
    win_sine_long: Vec<f32>,
    win_kbd_long: Vec<f32>,
    win_sine_short: Vec<f32>,
    win_kbd_short: Vec<f32>,
    overlap: [[f32; 1024]; 2],
    prev_win_shape: [u8; 2],
    rng_state: u32,
}

impl AacDecoder {
    /// Creates an `AacDecoder` initialized for `sample_rate` and `channels` (`1` or `2`).
    pub fn new(sample_rate: u32, channels: u8) -> Self {
        Self::from_asc(AudioSpecificConfig::new_aac_lc(sample_rate, channels))
    }

    /// Creates an `AacDecoder` from a parsed [`AudioSpecificConfig`].
    pub fn from_asc(asc: AudioSpecificConfig) -> Self {
        Self {
            asc,
            mdct_long: MdctPlan::new(2048),
            mdct_short: MdctPlan::new(256),
            win_sine_long: sine_window_f32(2048),
            win_kbd_long: kbd_window_f32(2048, 4.0),
            win_sine_short: sine_window_f32(256),
            win_kbd_short: kbd_window_f32(256, 6.0),
            overlap: [[0.0; 1024]; 2],
            prev_win_shape: [0; 2],
            rng_state: 0x1234_5678,
        }
    }

    /// Configures the decoder from a 2-byte `AudioSpecificConfig` (`esds` / `CodecPrivate`).
    pub fn with_extra_data(mut self, extra_data: &[u8]) -> Result<Self> {
        if extra_data.len() >= 2 {
            self.asc = AudioSpecificConfig::parse(extra_data)?;
        }
        Ok(self)
    }

    fn decode_ics(
        &mut self,
        br: &mut BitReader,
        common_window: bool,
        shared_ics: Option<&IcsInfo>,
    ) -> Result<ChannelData> {
        let mut ch = ChannelData::default();
        let global_gain = br.read_bits(8)? as i32;
        ch.ics = if common_window {
            shared_ics.cloned().unwrap_or_default()
        } else {
            IcsInfo::parse(br)?
        };

        let is_short = ch.ics.window_sequence == 2;
        let sect_bits = if is_short { 3u32 } else { 5u32 };
        let sect_esc = (1u32 << sect_bits) - 1;
        let max_sfb = ch.ics.max_sfb;

        // 1. section_data()
        for g in 0..ch.ics.num_window_groups {
            let mut k = 0usize;
            while k < max_sfb {
                let sect_cb = br.read_bits(4)? as u8;
                let mut sect_len = 0usize;
                loop {
                    let incr = br.read_bits(sect_bits)?;
                    sect_len += incr as usize;
                    if (incr as u32) < sect_esc || k + sect_len >= max_sfb {
                        break;
                    }
                }
                let end = (k + sect_len).min(max_sfb).min(64);
                for sfb in k..end {
                    ch.sfb_cb[g][sfb] = sect_cb;
                }
                k = end;
            }
        }

        // 2. scale_factor_data()
        let mut sf = global_gain;
        let mut is_pos = 0i32;
        let mut noise_nrg = global_gain - 90;
        let mut noise_pcm = true;

        for g in 0..ch.ics.num_window_groups {
            for sfb in 0..max_sfb.min(64) {
                match ch.sfb_cb[g][sfb] {
                    0 => {
                        ch.sf[g][sfb] = 0;
                    }
                    1..=11 => {
                        let diff = decode_huffman_u32(br, &HCB_SF_CODES, &HCB_SF_BITS)? as i32 - 60;
                        sf += diff;
                        ch.sf[g][sfb] = sf;
                    }
                    13 => {
                        if noise_pcm {
                            noise_pcm = false;
                            noise_nrg += (br.read_bits(9)? as i32) - 256;
                        } else {
                            noise_nrg +=
                                decode_huffman_u32(br, &HCB_SF_CODES, &HCB_SF_BITS)? as i32 - 60;
                        }
                        ch.sf[g][sfb] = noise_nrg;
                    }
                    14 | 15 => {
                        is_pos += decode_huffman_u32(br, &HCB_SF_CODES, &HCB_SF_BITS)? as i32 - 60;
                        ch.sf[g][sfb] = is_pos;
                    }
                    _ => {}
                }
            }
        }

        // 3. pulse_data()
        let mut pulses: Vec<(usize, i32)> = Vec::new();
        let pulse_present = br.read_bits(1)? != 0;
        if pulse_present {
            let num_pulse = br.read_bits(2)? as usize;
            let pulse_start_sfb = br.read_bits(6)? as usize;
            let swb = swb_offsets_long(self.asc.sampling_frequency_index);
            let mut k = if pulse_start_sfb < swb.len() {
                swb[pulse_start_sfb] as usize
            } else {
                0
            };
            for _ in 0..=num_pulse {
                let off = br.read_bits(5)? as usize;
                let amp = br.read_bits(4)? as i32;
                k += off;
                pulses.push((k.min(1023), amp));
            }
        }

        // 4. tns_data()
        let tns_present = br.read_bits(1)? != 0;
        if tns_present {
            for _w in 0..ch.ics.num_windows {
                let n_filt_bits = if is_short { 1 } else { 2 };
                let len_bits = if is_short { 4 } else { 6 };
                let ord_bits = if is_short { 3 } else { 5 };
                let n_filt = br.read_bits(n_filt_bits)? as usize;
                if n_filt > 0 {
                    let coef_res = br.read_bits(1)? as u32;
                    for _ in 0..n_filt {
                        let _length = br.read_bits(len_bits)?;
                        let order = br.read_bits(ord_bits)? as usize;
                        if order > 0 {
                            let _direction = br.read_bits(1)?;
                            let coef_compress = br.read_bits(1)? as u32;
                            let coef_bits = coef_res + 3 - coef_compress;
                            for _ in 0..order {
                                let _ = br.read_bits(coef_bits)?;
                            }
                        }
                    }
                }
            }
        }

        // 5. gain_control_data()
        let gain_present = br.read_bits(1)? != 0;
        if gain_present {
            return Err(RagdollError::Unsupported(
                "AAC SSR gain_control_data not supported in AAC-LC".into(),
            ));
        }

        // 6. spectral_data()
        let swb = if is_short {
            swb_offsets_short(self.asc.sampling_frequency_index)
        } else {
            swb_offsets_long(self.asc.sampling_frequency_index)
        };
        let sf_offset = if is_short {
            AAC_SF_OFFSET_SHORT
        } else {
            AAC_SF_OFFSET_LONG
        };

        let mut win_base = 0usize;
        for g in 0..ch.ics.num_window_groups {
            let glen = ch.ics.group_len[g] as usize;
            for sfb in 0..max_sfb.min(swb.len().saturating_sub(1)) {
                let cb = ch.sfb_cb[g][sfb];
                let start = swb[sfb] as usize;
                let end = swb[sfb + 1] as usize;
                let gain = 2.0_f32.powf(((ch.sf[g][sfb] as f32) - sf_offset) * 0.25);

                if cb == 13 {
                    // Perceptual Noise Substitution (PNS)
                    let mut energy = 0.0f32;
                    let start_idx = win_base * 128 + start * glen;
                    let count = (end - start) * glen;
                    for k in 0..count {
                        self.rng_state = self
                            .rng_state
                            .wrapping_mul(1_664_525)
                            .wrapping_add(1_013_904_223);
                        let r = (self.rng_state as i32 as f32) / (i32::MAX as f32);
                        ch.coeffs[start_idx + k] = r;
                        energy += r * r;
                    }
                    let scale = if energy > 1e-12 {
                        gain / energy.sqrt()
                    } else {
                        0.0
                    };
                    for k in 0..count {
                        ch.coeffs[start_idx + k] *= scale;
                    }
                    continue;
                }

                if !(1..=11).contains(&cb) {
                    continue;
                }

                for w in 0..glen {
                    let w_off = (win_base + w) * (if is_short { 128 } else { 1024 });
                    let mut idx = start;
                    while idx < end {
                        match cb {
                            1 | 2 => {
                                let sym = if cb == 1 {
                                    decode_huffman_u16(br, &HCB_1_CODES, &HCB_1_BITS)?
                                } else {
                                    decode_huffman_u16(br, &HCB_2_CODES, &HCB_2_BITS)?
                                };
                                let q0 = (sym / 27) as i32 - 1;
                                let q1 = ((sym / 9) % 3) as i32 - 1;
                                let q2 = ((sym / 3) % 3) as i32 - 1;
                                let q3 = (sym % 3) as i32 - 1;
                                ch.coeffs[w_off + idx] = inv_quant(q0, gain);
                                ch.coeffs[w_off + idx + 1] = inv_quant(q1, gain);
                                ch.coeffs[w_off + idx + 2] = inv_quant(q2, gain);
                                ch.coeffs[w_off + idx + 3] = inv_quant(q3, gain);
                                idx += 4;
                            }
                            3 | 4 => {
                                let sym = if cb == 3 {
                                    decode_huffman_u16(br, &HCB_3_CODES, &HCB_3_BITS)?
                                } else {
                                    decode_huffman_u16(br, &HCB_4_CODES, &HCB_4_BITS)?
                                };
                                let mut q = [
                                    (sym / 27) as i32,
                                    ((sym / 9) % 3) as i32,
                                    ((sym / 3) % 3) as i32,
                                    (sym % 3) as i32,
                                ];
                                for v in &mut q {
                                    if *v != 0 && br.read_bits(1)? != 0 {
                                        *v = -*v;
                                    }
                                }
                                for k in 0..4 {
                                    ch.coeffs[w_off + idx + k] = inv_quant(q[k], gain);
                                }
                                idx += 4;
                            }
                            5 | 6 => {
                                let sym = if cb == 5 {
                                    decode_huffman_u16(br, &HCB_5_CODES, &HCB_5_BITS)?
                                } else {
                                    decode_huffman_u16(br, &HCB_6_CODES, &HCB_6_BITS)?
                                };
                                let q0 = (sym / 9) as i32 - 4;
                                let q1 = (sym % 9) as i32 - 4;
                                ch.coeffs[w_off + idx] = inv_quant(q0, gain);
                                ch.coeffs[w_off + idx + 1] = inv_quant(q1, gain);
                                idx += 2;
                            }
                            7 | 8 => {
                                let sym = if cb == 7 {
                                    decode_huffman_u16(br, &HCB_7_CODES, &HCB_7_BITS)?
                                } else {
                                    decode_huffman_u16(br, &HCB_8_CODES, &HCB_8_BITS)?
                                };
                                let mut q = [(sym / 8) as i32, (sym % 8) as i32];
                                for v in &mut q {
                                    if *v != 0 && br.read_bits(1)? != 0 {
                                        *v = -*v;
                                    }
                                }
                                ch.coeffs[w_off + idx] = inv_quant(q[0], gain);
                                ch.coeffs[w_off + idx + 1] = inv_quant(q[1], gain);
                                idx += 2;
                            }
                            9 | 10 => {
                                let sym = if cb == 9 {
                                    decode_huffman_u16(br, &HCB_9_CODES, &HCB_9_BITS)?
                                } else {
                                    decode_huffman_u16(br, &HCB_10_CODES, &HCB_10_BITS)?
                                };
                                let mut q = [(sym / 13) as i32, (sym % 13) as i32];
                                for v in &mut q {
                                    if *v != 0 && br.read_bits(1)? != 0 {
                                        *v = -*v;
                                    }
                                }
                                ch.coeffs[w_off + idx] = inv_quant(q[0], gain);
                                ch.coeffs[w_off + idx + 1] = inv_quant(q[1], gain);
                                idx += 2;
                            }
                            11 => {
                                let sym = decode_huffman_u16(br, &HCB_11_CODES, &HCB_11_BITS)?;
                                let mut q = [(sym / 17) as i32, (sym % 17) as i32];
                                let mut signs = [1i32; 2];
                                for k in 0..2 {
                                    if q[k] != 0 && br.read_bits(1)? != 0 {
                                        signs[k] = -1;
                                    }
                                }
                                for k in 0..2 {
                                    if q[k] == 16 {
                                        let mut n = 0u32;
                                        while br.read_bits(1)? != 0 {
                                            n += 1;
                                            if n > 12 {
                                                break;
                                            }
                                        }
                                        let esc_word = br.read_bits(n + 4)? as i32;
                                        q[k] = (1i32 << (n + 4)) + esc_word;
                                    }
                                    q[k] *= signs[k];
                                }
                                ch.coeffs[w_off + idx] = inv_quant(q[0], gain);
                                ch.coeffs[w_off + idx + 1] = inv_quant(q[1], gain);
                                idx += 2;
                            }
                            _ => {
                                idx += 2;
                            }
                        }
                    }
                }
            }
            win_base += glen;
        }

        // Apply pulse additions if present
        if !pulses.is_empty() && !is_short {
            for &(k, amp) in &pulses {
                let sfb = swb
                    .iter()
                    .rposition(|&off| (off as usize) <= k)
                    .unwrap_or(0)
                    .min(63);
                let gain = 2.0_f32.powf(((ch.sf[0][sfb] as f32) - sf_offset) * 0.25);
                let base_q = (ch.coeffs[k].abs() / gain.max(1e-20)).powf(0.75).round() as i32;
                let new_q = if ch.coeffs[k] >= 0.0 {
                    base_q + amp
                } else {
                    -(base_q + amp)
                };
                ch.coeffs[k] = inv_quant(new_q, gain);
            }
        }

        Ok(ch)
    }

    fn synthesize_channel(&mut self, ch_idx: usize, ch_data: &ChannelData, pcm_out: &mut [f32]) {
        let cur_shape = ch_data.ics.window_shape;
        let prev_shape = self.prev_win_shape[ch_idx];
        self.prev_win_shape[ch_idx] = cur_shape;

        if ch_data.ics.window_sequence != 2 {
            let mut imdct_buf = [0.0f32; 2048];
            self.mdct_long.imdct(&ch_data.coeffs, &mut imdct_buf);
            let win_left = if prev_shape == 1 {
                &self.win_kbd_long
            } else {
                &self.win_sine_long
            };
            let win_right = if cur_shape == 1 {
                &self.win_kbd_long
            } else {
                &self.win_sine_long
            };
            if prev_shape == cur_shape {
                window_overlap_add_f32(&imdct_buf, win_right, &mut self.overlap[ch_idx], pcm_out);
            } else {
                for i in 0..1024 {
                    pcm_out[i] = self.overlap[ch_idx][i] + imdct_buf[i] * win_left[i];
                    self.overlap[ch_idx][i] = imdct_buf[1024 + i] * win_right[1024 + i];
                }
            }
        } else {
            // EIGHT_SHORT_SEQUENCE: 8 x 256-sample IMDCTs overlapped at 128-sample intervals
            let win_short = if cur_shape == 1 {
                &self.win_kbd_short
            } else {
                &self.win_sine_short
            };
            let mut full_2048 = [0.0f32; 2048];
            let mut short_out = [0.0f32; 256];
            for w in 0..8 {
                let bins = &ch_data.coeffs[w * 128..(w + 1) * 128];
                self.mdct_short.imdct(bins, &mut short_out);
                let base = 448 + w * 128;
                for i in 0..256 {
                    full_2048[base + i] += short_out[i] * win_short[i];
                }
            }
            for i in 0..1024 {
                pcm_out[i] = self.overlap[ch_idx][i] + full_2048[i];
                self.overlap[ch_idx][i] = full_2048[1024 + i];
            }
        }
    }

    /// Decodes a single AAC-LC packet (either raw access unit or ADTS-prefixed frame) into a 1024-sample [`AudioFrame`].
    pub fn decode_packet(&mut self, pkt: &[u8]) -> Result<AudioFrame> {
        if pkt.is_empty() {
            return Err(RagdollError::BitstreamEndOfData);
        }
        let raw_payload = if pkt.len() >= 7 && pkt[0] == 0xFF && (pkt[1] & 0xF0) == 0xF0 {
            let adts = AdtsHeader::parse(pkt)?;
            self.asc = AudioSpecificConfig {
                audio_object_type: adts.profile + 1,
                sampling_frequency_index: adts.sampling_frequency_index,
                sample_rate: AAC_SAMPLE_RATES[adts.sampling_frequency_index as usize],
                channel_configuration: adts.channel_configuration,
            };
            let end = (adts.frame_length as usize).min(pkt.len());
            &pkt[adts.header_len.min(end)..end]
        } else {
            pkt
        };

        let mut br = BitReader::new(raw_payload);
        let mut pcm_ch = [[0.0f32; 1024]; 2];
        let mut decoded_channels = 0usize;

        while br.bits_remaining() >= 3 {
            let id_syn_ele = br.read_bits(3)? as u8;
            match id_syn_ele {
                // ID_SCE (Single Channel Element)
                0 => {
                    let _tag = br.read_bits(4)?;
                    let ch0 = self.decode_ics(&mut br, false, None)?;
                    let idx = decoded_channels.min(1);
                    self.synthesize_channel(idx, &ch0, &mut pcm_ch[idx]);
                    decoded_channels = (decoded_channels + 1).min(2);
                }
                // ID_CPE (Channel Pair Element)
                1 => {
                    let _tag = br.read_bits(4)?;
                    let common_window = br.read_bits(1)? != 0;
                    let mut shared_ics = None;
                    let mut ms_used = [[false; 64]; 8];
                    if common_window {
                        let ics = IcsInfo::parse(&mut br)?;
                        let ms_mask_present = br.read_bits(2)? as u8;
                        if ms_mask_present == 1 {
                            for g in 0..ics.num_window_groups {
                                for sfb in 0..ics.max_sfb.min(64) {
                                    ms_used[g][sfb] = br.read_bits(1)? != 0;
                                }
                            }
                        } else if ms_mask_present == 2 {
                            for g in 0..ics.num_window_groups {
                                for sfb in 0..ics.max_sfb.min(64) {
                                    ms_used[g][sfb] = true;
                                }
                            }
                        }
                        shared_ics = Some(ics);
                    }
                    let mut ch0 = self.decode_ics(&mut br, common_window, shared_ics.as_ref())?;
                    let mut ch1 = self.decode_ics(&mut br, common_window, shared_ics.as_ref())?;

                    // Apply Mid/Side and Intensity Stereo decoding when common_window is active
                    if common_window {
                        let is_short = ch0.ics.window_sequence == 2;
                        let swb = if is_short {
                            swb_offsets_short(self.asc.sampling_frequency_index)
                        } else {
                            swb_offsets_long(self.asc.sampling_frequency_index)
                        };
                        let mut win_base = 0usize;
                        for g in 0..ch0.ics.num_window_groups {
                            let glen = ch0.ics.group_len[g] as usize;
                            for sfb in 0..ch0.ics.max_sfb.min(swb.len().saturating_sub(1)) {
                                let start = swb[sfb] as usize;
                                let end = swb[sfb + 1] as usize;
                                let cb1 = ch1.sfb_cb[g][sfb];
                                if cb1 == 14 || cb1 == 15 {
                                    // Intensity Stereo
                                    let sign = if cb1 == 14 { 1.0f32 } else { -1.0f32 };
                                    let sign = if ms_used[g][sfb] { -sign } else { sign };
                                    let scale = sign * 0.5_f32.powf(0.25 * (ch1.sf[g][sfb] as f32));
                                    for w in 0..glen {
                                        let w_off =
                                            (win_base + w) * (if is_short { 128 } else { 1024 });
                                        for k in start..end {
                                            ch1.coeffs[w_off + k] = ch0.coeffs[w_off + k] * scale;
                                        }
                                    }
                                } else if ms_used[g][sfb] && ch0.sfb_cb[g][sfb] < 13 && cb1 < 13 {
                                    // Mid/Side Stereo
                                    for w in 0..glen {
                                        let w_off =
                                            (win_base + w) * (if is_short { 128 } else { 1024 });
                                        for k in start..end {
                                            let m = ch0.coeffs[w_off + k];
                                            let s = ch1.coeffs[w_off + k];
                                            ch0.coeffs[w_off + k] = m + s;
                                            ch1.coeffs[w_off + k] = m - s;
                                        }
                                    }
                                }
                            }
                            win_base += glen;
                        }
                    }

                    self.synthesize_channel(0, &ch0, &mut pcm_ch[0]);
                    self.synthesize_channel(1, &ch1, &mut pcm_ch[1]);
                    decoded_channels = 2;
                }
                // ID_DSE (Data Stream Element)
                4 => {
                    let _tag = br.read_bits(4)?;
                    let align = br.read_bits(1)? != 0;
                    let mut count = br.read_bits(8)? as usize;
                    if count == 255 {
                        count += br.read_bits(8)? as usize;
                    }
                    if align {
                        br.align_to_byte();
                    }
                    for _ in 0..count {
                        let _ = br.read_bits(8)?;
                    }
                }
                // ID_FIL (Fill Element)
                6 => {
                    let mut cnt = br.read_bits(4)? as usize;
                    if cnt == 15 {
                        cnt += (br.read_bits(8)? as usize).saturating_sub(1);
                    }
                    for _ in 0..(cnt * 8) {
                        if br.bits_remaining() == 0 {
                            break;
                        }
                        let _ = br.read_bits(1)?;
                    }
                }
                // ID_END
                7 => {
                    break;
                }
                _ => {
                    break;
                }
            }
        }

        let out_ch = decoded_channels
            .max(self.asc.channel_configuration as usize)
            .clamp(1, 2);
        let mut frame = AudioFrame::new(
            self.asc.sample_rate,
            out_ch as u8,
            SampleFormat::F32le,
            1024,
        );
        frame.layout = ChannelLayout::from_channels(out_ch as u8);
        let dst = frame.as_f32_mut_slice();
        if out_ch == 1 {
            dst.copy_from_slice(&pcm_ch[0]);
        } else {
            for i in 0..1024 {
                dst[i * 2] = pcm_ch[0][i];
                dst[i * 2 + 1] = pcm_ch[1][i];
            }
        }
        Ok(frame)
    }
}

/// Pure-Rust ISO/IEC 14496-3 AAC-LC Encoder (`1024` samples per frame).
#[derive(Clone, Debug)]
pub struct AacEncoder {
    pub asc: AudioSpecificConfig,
    pub bitrate_bps: u32,
    mdct_long: MdctPlan,
    win_kbd_long: Vec<f32>,
    history: [[f32; 1024]; 2],
}

impl AacEncoder {
    /// Creates a new `AacEncoder` for `sample_rate`, `channels` (`1` or `2`), and `bitrate_bps` (e.g. `128_000`).
    pub fn new(sample_rate: u32, channels: u8, bitrate_bps: u32) -> Self {
        let asc = AudioSpecificConfig::new_aac_lc(sample_rate, channels);
        Self {
            asc,
            bitrate_bps: bitrate_bps.max(32_000),
            mdct_long: MdctPlan::new(2048),
            win_kbd_long: kbd_window_f32(2048, 4.0),
            history: [[0.0; 1024]; 2],
        }
    }

    /// Returns the 2-byte `AudioSpecificConfig` (`esds` / `CodecPrivate`) for MP4/MKV muxers.
    pub fn extra_data(&self) -> [u8; 2] {
        self.asc.to_bytes()
    }

    fn encode_ics_channel(&self, bw: &mut BitWriter, coeffs: &[f32; 1024], write_ics_info: bool) {
        let swb = swb_offsets_long(self.asc.sampling_frequency_index);
        let max_sfb = (swb.len() - 1).min(49);

        // Compute global_gain so max quantized magnitude fits comfortably in [0..2047]
        let max_abs = coeffs.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        // Quality offset biased by target bitrate (higher bitrate -> smaller sf -> finer quantization)
        let quality_bias = if self.bitrate_bps >= 192_000 {
            -10
        } else if self.bitrate_bps >= 128_000 {
            -6
        } else {
            0
        };
        let global_gain = if max_abs < 1e-6 {
            100u8
        } else {
            // Solve max_abs * 2^((164 - sf)/4) <= 256^(4/3) ~ 1625 -> sf >= 164 + 4 * log2(max_abs / 1625)
            let min_sf =
                (AAC_SF_OFFSET_LONG + 4.0 * (max_abs / 512.0).max(1e-9).log2()).ceil() as i32;
            (min_sf + quality_bias).clamp(60, 230) as u8
        };

        let inv_gain = 2.0_f32.powf((AAC_SF_OFFSET_LONG - (global_gain as f32)) * 0.25);
        let mut q_bins = [0i32; 1024];
        for i in 0..1024 {
            let scaled = coeffs[i].abs() * inv_gain;
            if scaled > 0.35 {
                // Magic 0.4054 dead-zone rounding from ISO/IEC 14496-3 / libavcodec
                let q = (scaled.powf(0.75) + 0.4054).floor() as i32;
                let q_clamped = q.min(4095);
                q_bins[i] = if coeffs[i] < 0.0 {
                    -q_clamped
                } else {
                    q_clamped
                };
            }
        }

        // Determine per-sfb codebook: 0 (ZERO_HCB) if all zeros, 11 (ESC_HCB) if non-zero
        let mut sfb_cb = vec![0u8; max_sfb];
        for sfb in 0..max_sfb {
            let s = swb[sfb] as usize;
            let e = swb[sfb + 1] as usize;
            if q_bins[s..e].iter().any(|&q| q != 0) {
                sfb_cb[sfb] = 11;
            }
        }

        // 1. global_gain (8 bits)
        bw.write_bits(global_gain as u64, 8);

        // 2. ics_info (if not common_window)
        if write_ics_info {
            bw.write_bits(0, 1); // ics_reserved_bit
            bw.write_bits(0, 2); // ONLY_LONG_SEQUENCE
            bw.write_bits(1, 1); // window_shape = 1 (KBD)
            bw.write_bits(max_sfb as u64, 6);
            bw.write_bits(0, 1); // predictor_data_present = 0
        }

        // 3. section_data()
        let mut k = 0usize;
        while k < max_sfb {
            let cb = sfb_cb[k];
            let mut run = 1usize;
            while k + run < max_sfb && sfb_cb[k + run] == cb {
                run += 1;
            }
            bw.write_bits(cb as u64, 4);
            let mut rem = run;
            while rem >= 31 {
                bw.write_bits(31, 5);
                rem -= 31;
            }
            bw.write_bits(rem as u64, 5);
            k += run;
        }

        // 4. scale_factor_data(): all active bands use sf == global_gain -> diff = 0 -> HCB_SF[60] = 1-bit `0`!
        for &cb in &sfb_cb {
            if cb != 0 {
                bw.write_bits(HCB_SF_CODES[60] as u64, HCB_SF_BITS[60] as u32);
            }
        }

        // 5. pulse_data_present (0), tns_data_present (0), gain_control_data_present (0)
        bw.write_bits(0, 3);

        // 6. spectral_data() using HCB_11 (ESC_HCB)
        for sfb in 0..max_sfb {
            if sfb_cb[sfb] != 11 {
                continue;
            }
            let s = swb[sfb] as usize;
            let e = swb[sfb + 1] as usize;
            for idx in (s..e).step_by(2) {
                let q0 = q_bins[idx];
                let q1 = q_bins[idx + 1];
                let a0 = q0.unsigned_abs();
                let a1 = q1.unsigned_abs();
                let c0 = a0.min(16) as usize;
                let c1 = a1.min(16) as usize;
                let sym = c0 * 17 + c1;
                bw.write_bits(HCB_11_CODES[sym] as u64, HCB_11_BITS[sym] as u32);
                if a0 > 0 {
                    bw.write_bits(u64::from(q0 < 0), 1);
                }
                if a1 > 0 {
                    bw.write_bits(u64::from(q1 < 0), 1);
                }
                for &a in &[a0, a1] {
                    if a >= 16 {
                        // Escape sequence: find N >= 0 such that 16 * 2^N <= a < 16 * 2^(N+1)
                        let mut n = 0u32;
                        while (1u32 << (n + 5)) <= a {
                            n += 1;
                        }
                        // Write N ones followed by 0
                        if n > 0 {
                            bw.write_bits((1u64 << n) - 1, n);
                        }
                        bw.write_bits(0, 1);
                        let suffix = a - (1u32 << (n + 4));
                        bw.write_bits(suffix as u64, n + 4);
                    }
                }
            }
        }
    }

    /// Encodes a 1024-sample [`AudioFrame`] into a raw ISO/IEC 14496-3 AAC-LC access unit (for MP4/MKV).
    pub fn encode_raw_frame(&mut self, frame: &AudioFrame) -> Result<Vec<u8>> {
        let f32_frame = if frame.format == SampleFormat::F32le {
            frame.clone()
        } else {
            convert_audio_frame(frame, SampleFormat::F32le)?
        };
        let ch_count = (self.asc.channel_configuration as usize).clamp(1, 2);
        let src = f32_frame.as_f32_slice();
        let in_samples = f32_frame.num_samples.min(1024);
        let src_ch = (f32_frame.channels as usize).max(1);

        let mut coeffs = [[0.0f32; 1024]; 2];
        for c in 0..ch_count {
            let mut cur_1024 = [0.0f32; 1024];
            for i in 0..in_samples {
                cur_1024[i] = src[i * src_ch + c.min(src_ch - 1)];
            }
            let mut win_2048 = [0.0f32; 2048];
            for i in 0..1024 {
                win_2048[i] = self.history[c][i] * self.win_kbd_long[i];
                win_2048[1024 + i] = cur_1024[i] * self.win_kbd_long[1024 + i];
            }
            self.history[c] = cur_1024;
            self.mdct_long.mdct(&win_2048, &mut coeffs[c]);
        }

        let mut bw = BitWriter::with_capacity(512);
        let swb = swb_offsets_long(self.asc.sampling_frequency_index);
        let max_sfb = (swb.len() - 1).min(49);

        if ch_count == 1 {
            bw.write_bits(0, 3); // ID_SCE = 0
            bw.write_bits(0, 4); // element_instance_tag = 0
            self.encode_ics_channel(&mut bw, &coeffs[0], true);
        } else {
            bw.write_bits(1, 3); // ID_CPE = 1
            bw.write_bits(0, 4); // element_instance_tag = 0
            bw.write_bits(1, 1); // common_window = 1
            // Shared ics_info
            bw.write_bits(0, 1); // ics_reserved_bit
            bw.write_bits(0, 2); // ONLY_LONG_SEQUENCE
            bw.write_bits(1, 1); // window_shape = 1 (KBD)
            bw.write_bits(max_sfb as u64, 6);
            bw.write_bits(0, 1); // predictor_data_present = 0
            bw.write_bits(0, 2); // ms_mask_present = 0
            self.encode_ics_channel(&mut bw, &coeffs[0], false);
            self.encode_ics_channel(&mut bw, &coeffs[1], false);
        }

        bw.write_bits(7, 3); // ID_END = 7
        bw.align_to_byte();
        Ok(bw.finish())
    }

    /// Encodes a 1024-sample [`AudioFrame`] into a 7-byte ADTS-framed AAC-LC packet (for MPEG-TS / `.aac`).
    pub fn encode_adts_frame(&mut self, frame: &AudioFrame) -> Result<Vec<u8>> {
        let raw = self.encode_raw_frame(frame)?;
        let hdr = AdtsHeader::new(&self.asc, raw.len());
        let mut out = Vec::with_capacity(7 + raw.len());
        out.extend_from_slice(&hdr.to_bytes());
        out.extend_from_slice(&raw);
        Ok(out)
    }

    /// Encodes an arbitrary-length `AudioFrame` (splitting/padding into 1024-sample AAC frames) into raw AAC access units.
    pub fn encode_stream_frames(
        &mut self,
        frame: &AudioFrame,
    ) -> Result<Vec<(Vec<u8>, Timestamp, i64)>> {
        let f32_frame = if frame.format == SampleFormat::F32le {
            frame.clone()
        } else {
            convert_audio_frame(frame, SampleFormat::F32le)?
        };
        let ch = f32_frame.channels as usize;
        let total_samples = f32_frame.num_samples;
        let src = f32_frame.as_f32_slice();
        let tb = Rational64::new(1, self.asc.sample_rate as i64);
        let base_pts = frame.pts.rescale_to(tb).ticks;

        let mut packets = Vec::new();
        let mut offset = 0usize;
        while offset < total_samples {
            let take = (total_samples - offset).min(1024);
            let mut sub = AudioFrame::new(
                self.asc.sample_rate,
                f32_frame.channels,
                SampleFormat::F32le,
                1024,
            );
            sub.as_f32_mut_slice()[..take * ch]
                .copy_from_slice(&src[offset * ch..(offset + take) * ch]);
            let raw = self.encode_raw_frame(&sub)?;
            let pts = Timestamp::new(base_pts + (offset as i64), tb);
            packets.push((raw, pts, 1024));
            offset += 1024;
        }
        Ok(packets)
    }
}
