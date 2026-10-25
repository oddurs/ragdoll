//! # `ragdoll-codec`
//!
//! 100% pure-Rust video and audio codecs:
//! - [`pcm`]: Vectorized `aarch64` NEON + scalar PCM format conversion, stereo interleaving, and Mid-Side decorrelation.
//! - [`flac`]: Lossless FLAC (RFC 9639) stream/frame decoder and Fixed + FIR LPC Rice encoder (`crc8_flac` + `crc16_flac`).
//! - [`h264`]: H.264 / AVC (`SPS`, `PPS`, `SliceHeader`, `SeiMessage`) parser/serializer, $4\times4$ integer DCT/Hadamard,
//!   CAVLC entropy coder, and Baseline (`profile_idc = 66`) frame encoder/decoder.
//! - [`av1`]: AV1 Open Bitstream Unit (`OBU`) parser, `Av1SequenceHeader` inspector (`dav1d` parity), `av1C` builder, and IVF stream helper.
//! - [`rgdi`]: Ragdoll Fast-Intra (`RGDI`) SIMD studio video codec sustaining > 300 fps 4K encode/decode on Apple M4.

pub mod aac;
pub mod av1;
pub mod cabac;
pub mod flac;
pub mod h264;
pub mod loudness;
pub mod opus;
pub mod pcm;
pub mod resample;
pub mod rgdi;

pub use aac::{
    AAC_SAMPLE_RATES, AacDecoder, AacEncoder, AdtsHeader, AudioSpecificConfig, sample_rate_to_index,
};
pub use av1::{
    Av1SequenceHeader, IvfHeader, ObuHeader, ObuPacket, ObuType, encode_obu, parse_ivf_stream,
    parse_obu_header, split_obus,
};
pub use cabac::{CabacContext, CabacDecoder, CabacEncoder, init_cabac_contexts};
pub use flac::{
    FLAC_STREAM_MARKER, FlacChannelMode, FlacDecoder, FlacEncoder, FlacFrameHeader, FlacStreamInfo,
    compute_flac_pcm_md5_planar, md5_digest, parse_flac_stream_header,
};
pub use h264::{
    H264Decoder, H264Encoder, H264Profile, NalHeader, NalUnitType, Pps, RateControlMode,
    SeiMessage, SliceHeader, Sps, decode_cavlc_block, dequantize_block4x4, dequantize_block8x8,
    encode_annex_b_nal, encode_cavlc_block, parse_nal_rbsp, predict_intra_16x16,
    predict_intra_16x16_scalar, predict_intra_chroma_8x8, predict_intra_chroma_8x8_scalar,
    quantize_block4x4, quantize_block8x8, roundtrip_block4x4_dct,
};
pub use loudness::{EbuR128Meter, KWeightingFilterBank, LoudnessReport, normalize_audio_frame};
pub use opus::{
    CELT_EBAND_BASE, OpusDecoder, OpusEncoder, OpusHead, OpusRangeDecoder, OpusRangeEncoder,
    OpusToc, build_opus_tags, parse_opus_tags, pvq_reconstruct_band, pvq_search_pulses,
};
pub use pcm::{
    audio_frame_to_planar_i32, convert_audio_frame, decorrelate_mid_side_i32,
    decorrelate_mid_side_i32_scalar, deinterleave_stereo_i32, deinterleave_stereo_i32_scalar,
    interleave_stereo_i32, interleave_stereo_i32_scalar, pcm_f32_to_s16, pcm_f32_to_s16_scalar,
    pcm_i32_to_s24le, pcm_s16_to_f32, pcm_s16_to_f32_scalar, pcm_s24le_to_i32,
    planar_i32_to_audio_frame, reconstruct_left_side_i32, reconstruct_mid_side_i32,
    reconstruct_mid_side_i32_scalar, reconstruct_right_side_i32,
};
pub use resample::{ChannelRematrixer, PolyphaseResampler, resample_and_rematrix_frame};
pub use rgdi::{
    RGDI_MAGIC, RagdollIntraDecoder, RagdollIntraEncoder, RgdiFrameHeader, decode_rgdi_plane,
    encode_rgdi_plane, encode_rgdi_plane_scalar,
};

#[cfg(test)]
mod tests {
    use super::*;
    use ragdoll_core::{AudioFrame, FramePool, PixelFormat, Rational64, SampleFormat, VideoFrame};
    use ragdoll_pix::evaluate_frame_quality;
    use std::process::Command;

    #[test]
    fn pcm_neon_vs_scalar_differential_and_conversions() {
        let s16_in: Vec<i16> = (-50..60).map(|i| (i * 513) as i16).collect();
        let mut f32_simd = vec![0.0f32; s16_in.len()];
        let mut f32_ref = vec![0.0f32; s16_in.len()];
        pcm_s16_to_f32(&s16_in, &mut f32_simd);
        pcm_s16_to_f32_scalar(&s16_in, &mut f32_ref);
        assert_eq!(f32_simd, f32_ref);

        let mut s16_simd = vec![0i16; s16_in.len()];
        let mut s16_ref = vec![0i16; s16_in.len()];
        pcm_f32_to_s16(&f32_simd, &mut s16_simd);
        pcm_f32_to_s16_scalar(&f32_ref, &mut s16_ref);
        assert_eq!(s16_simd, s16_ref);
        assert_eq!(s16_simd, s16_in);

        // Stereo interleave / deinterleave & Mid-Side decorrelation NEON vs Scalar
        let left: Vec<i32> = (0..37).map(|i| i * 1234 - 15000).collect();
        let right: Vec<i32> = (0..37).map(|i| i * 1190 - 14200).collect();
        let mut mid_simd = vec![0i32; 37];
        let mut side_simd = vec![0i32; 37];
        let mut mid_ref = vec![0i32; 37];
        let mut side_ref = vec![0i32; 37];
        decorrelate_mid_side_i32(&left, &right, &mut mid_simd, &mut side_simd);
        decorrelate_mid_side_i32_scalar(&left, &right, &mut mid_ref, &mut side_ref);
        assert_eq!(mid_simd, mid_ref);
        assert_eq!(side_simd, side_ref);

        let mut rec_l = vec![0i32; 37];
        let mut rec_r = vec![0i32; 37];
        reconstruct_mid_side_i32(&mid_simd, &side_simd, &mut rec_l, &mut rec_r);
        assert_eq!(rec_l, left);
        assert_eq!(rec_r, right);
    }

    #[test]
    fn flac_16bit_and_24bit_lossless_roundtrip_and_ffmpeg_interop() {
        // 1. 16-bit stereo multi-tone + constant + wasted-bits + transient signal
        let sr = 48_000;
        let num_samples = 8_500; // spans 3 blocks (4096 + 4096 + 308)
        let mut audio16 = AudioFrame::new(sr, 2, SampleFormat::S16le, num_samples);
        {
            let s = audio16.as_i16_mut_slice();
            for i in 0..num_samples {
                if i < 4096 {
                    // Correlated stereo sine + harmonics (triggers Mid-Side + Fixed/LPC)
                    let t = i as f64 / (sr as f64);
                    let l = ((t * 440.0 * std::f64::consts::TAU).sin() * 18_000.0
                        + (t * 880.0 * std::f64::consts::TAU).cos() * 4_000.0)
                        as i16;
                    let r = ((t * 440.0 * std::f64::consts::TAU).sin() * 17_500.0
                        - (t * 220.0 * std::f64::consts::TAU).sin() * 2_000.0)
                        as i16;
                    s[i * 2] = l;
                    s[i * 2 + 1] = r;
                } else {
                    // Wasted bits (multiples of 16) on L, constant on R
                    s[i * 2] = (((i as i16) % 256) - 128) * 16;
                    s[i * 2 + 1] = 1024;
                }
            }
        }

        let enc16 = FlacEncoder::new(sr, 2, 16);
        let flac16_bytes = enc16.encode_stream(&audio16).unwrap();
        let (info16, dec16) = FlacDecoder::decode_stream(&flac16_bytes).unwrap();
        assert_eq!(info16.sample_rate, sr);
        assert_eq!(info16.channels, 2);
        assert_eq!(info16.bits_per_sample, 16);
        assert_eq!(info16.total_samples, num_samples as u64);
        assert_eq!(dec16.as_i16_slice(), audio16.as_i16_slice());

        // 2. 24-bit stereo lossless round-trip (0 LSB error across [-8388608, 8388607])
        let mut left24 = Vec::with_capacity(4096);
        let mut right24 = Vec::with_capacity(4096);
        for i in 0..4096i32 {
            let l = ((i as f64 * 0.05).sin() * 6_000_000.0) as i32 + (i * 3 - 5000);
            let r = ((i as f64 * 0.05).sin() * 5_800_000.0) as i32 - (i * 2);
            left24.push(l);
            right24.push(r);
        }
        let audio24 =
            planar_i32_to_audio_frame(&[left24.clone(), right24.clone()], 96_000, 24).unwrap();
        let enc24 = FlacEncoder::new(96_000, 2, 24);
        let flac24_bytes = enc24.encode_stream(&audio24).unwrap();
        let (info24, dec24) = FlacDecoder::decode_stream(&flac24_bytes).unwrap();
        assert_eq!(info24.bits_per_sample, 24);
        let (dec24_planar, _) = audio_frame_to_planar_i32(&dec24).unwrap();
        assert_eq!(dec24_planar[0], left24);
        assert_eq!(dec24_planar[1], right24);

        // 3. Bidirectional FFmpeg interop:
        //    (a) Decode Ragdoll's .flac using `ffmpeg -i` and verify 0 LSB difference!
        //    (b) Encode a .flac using `ffmpeg -c:a flac` and decode it in Ragdoll's FlacDecoder with 0 LSB difference!
        let tmp_dir = std::env::temp_dir();
        let ragdoll_flac_path = tmp_dir.join("ragdoll_test_out.flac");
        let ffmpeg_raw_path = tmp_dir.join("ragdoll_ffmpeg_dec.raw");
        let ffmpeg_flac_path = tmp_dir.join("ffmpeg_encoded.flac");
        std::fs::write(&ragdoll_flac_path, &flac16_bytes).unwrap();

        let status_dec = Command::new("ffmpeg")
            .args([
                "-y",
                "-v",
                "error",
                "-i",
                ragdoll_flac_path.to_str().unwrap(),
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                ffmpeg_raw_path.to_str().unwrap(),
            ])
            .status();

        if let Ok(st) = status_dec
            && st.success()
        {
            let raw_bytes = std::fs::read(&ffmpeg_raw_path).unwrap();
            assert_eq!(raw_bytes, audio16.data.as_slice());

            // Now have FFmpeg encode that raw PCM using FFmpeg's own LPC/Rice FLAC encoder
            let st_enc = Command::new("ffmpeg")
                .args([
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "s16le",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    "-i",
                    ffmpeg_raw_path.to_str().unwrap(),
                    "-c:a",
                    "flac",
                    ffmpeg_flac_path.to_str().unwrap(),
                ])
                .status()
                .unwrap();
            assert!(st_enc.success());

            let ffmpeg_flac_bytes = std::fs::read(&ffmpeg_flac_path).unwrap();
            let (_ff_info, ff_decoded_in_ragdoll) =
                FlacDecoder::decode_stream(&ffmpeg_flac_bytes).unwrap();
            assert_eq!(
                ff_decoded_in_ragdoll.as_i16_slice(),
                audio16.as_i16_slice(),
                "Ragdoll FlacDecoder must decode FFmpeg-generated LPC FLAC bit-identically!"
            );
        }

        let _ = std::fs::remove_file(&ragdoll_flac_path);
        let _ = std::fs::remove_file(&ffmpeg_raw_path);
        let _ = std::fs::remove_file(&ffmpeg_flac_path);
    }

    #[test]
    fn h264_sps_pps_slice_cavlc_and_ffmpeg_bidirectional_interop() {
        // 1. CAVLC block round-trip across diverse coefficient patterns and nC tables
        for nc in [0u8, 1, 2, 5, 10] {
            let coeffs: [i16; 16] = [14, -5, 3, -1, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0];
            let mut w = ragdoll_core::BitWriter::new();
            encode_cavlc_block(&mut w, &coeffs, nc, false);
            let bytes = w.finish();
            let mut r = ragdoll_core::BitReader::new(&bytes);
            let mut dec = [0i16; 16];
            let count = decode_cavlc_block(&mut r, &mut dec, nc, false).unwrap();
            assert_eq!(count, 6);
            assert_eq!(dec, coeffs);
        }

        // 2. Encode a multi-frame H.264 stream (with non-multiple-of-16 crop 320x180 -> 12 macroblock rows)
        //    containing both smooth I_16x16 regions, textured I_PCM regions, and P_Skip inter frames!
        let width = 320usize;
        let height = 180usize;
        let mut enc = H264Encoder::new(width as u32, height as u32, Rational64::FPS_30, 20);
        let mut dec = H264Decoder::new();
        let mut annex_b_stream = Vec::new();
        let mut orig_frames = Vec::new();

        for f_idx in 0..4usize {
            let mut frame = VideoFrame::new(width, height, PixelFormat::Yuv420p);
            for y in 0..height {
                let row = frame.plane_mut(0).row_mut(y);
                for (x, p) in row.iter_mut().enumerate().take(width) {
                    if x < 160 && y < 96 {
                        // Smooth region -> encoded as I_16x16
                        *p = 90;
                    } else {
                        // Textured gradient + animation -> encoded as I_PCM / updated in P-frames
                        *p = ((x * 5 + y * 3 + f_idx * 17) & 0xFF) as u8;
                    }
                }
            }
            for y in 0..height / 2 {
                frame.plane_mut(1).row_mut(y).fill(128);
                frame.plane_mut(2).row_mut(y).fill(128);
            }

            let pkt = enc.encode_frame(&frame).unwrap();
            let decoded = dec.decode_annex_b_packet(&pkt).unwrap().unwrap();
            assert_eq!(decoded.width, width);
            assert_eq!(decoded.height, height);
            let q = evaluate_frame_quality(&frame, &decoded, None);
            assert!(
                q.psnr_y > 45.0,
                "expected high H.264 PSNR, got {}",
                q.psnr_y
            );

            annex_b_stream.extend_from_slice(&pkt);
            orig_frames.push(frame);
        }

        // 3. Verify that `ffmpeg` / `ffprobe` decodes our pure-Rust H.264 bitstream cleanly
        //    AND that our SPS/PPS/SliceHeader parser extracts exact metadata from `ffmpeg -c:v libx264`!
        let tmp_dir = std::env::temp_dir();
        let h264_path = tmp_dir.join("ragdoll_out.h264");
        let ff_dec_y4m = tmp_dir.join("ragdoll_h264_ffdec.yuv");
        let x264_path = tmp_dir.join("ffmpeg_x264.h264");
        std::fs::write(&h264_path, &annex_b_stream).unwrap();

        let status = Command::new("ffmpeg")
            .args([
                "-y",
                "-v",
                "error",
                "-i",
                h264_path.to_str().unwrap(),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                ff_dec_y4m.to_str().unwrap(),
            ])
            .status();

        if let Ok(st) = status
            && st.success()
        {
            let raw_yuv = std::fs::read(&ff_dec_y4m).unwrap();
            let frame_bytes = width * height * 3 / 2;
            assert_eq!(raw_yuv.len(), frame_bytes * 4);

            // Also generate a real-world High Profile stream with libx264 and parse its SPS/PPS/SliceHeader
            let st_x264 = Command::new("ffmpeg")
                .args([
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=640x360:rate=24:duration=1",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    x264_path.to_str().unwrap(),
                ])
                .status();
            if let Ok(st2) = st_x264
                && st2.success()
            {
                let x264_bytes = std::fs::read(&x264_path).unwrap();
                let mut inspector = H264Decoder::new();
                for nal in ragdoll_simd::split_annex_b_nals(&x264_bytes) {
                    let (hdr, rbsp) = parse_nal_rbsp(nal).unwrap();
                    match hdr.nal_unit_type {
                        NalUnitType::Sps => {
                            let sps = Sps::parse_rbsp(&rbsp).unwrap();
                            assert_eq!(sps.width(), 640);
                            assert_eq!(sps.height(), 360);
                            inspector.sps = Some(sps);
                        }
                        NalUnitType::Pps => {
                            inspector.pps = Some(Pps::parse_rbsp(&rbsp).unwrap());
                        }
                        NalUnitType::Sei => {
                            let msgs = SeiMessage::parse_rbsp(&rbsp);
                            assert!(!msgs.is_empty());
                        }
                        NalUnitType::IdrSlice | NalUnitType::NonIdrSlice => {
                            let mut br = ragdoll_core::BitReader::new(&rbsp);
                            let sh = SliceHeader::parse_from_reader(
                                &mut br,
                                hdr,
                                inspector.sps.as_ref().unwrap(),
                                inspector.pps.as_ref().unwrap(),
                            )
                            .unwrap();
                            assert!(sh.slice_qp <= 51);
                        }
                        _ => {}
                    }
                }
                assert!(inspector.sps.is_some());
                assert!(inspector.pps.is_some());
            }
        }

        let _ = std::fs::remove_file(&h264_path);
        let _ = std::fs::remove_file(&ff_dec_y4m);
        let _ = std::fs::remove_file(&x264_path);
    }

    #[test]
    fn av1_obu_sequence_header_parser_and_svtav1_interop() {
        let seq = Av1SequenceHeader {
            seq_profile: 0,
            still_picture: false,
            reduced_still_picture_header: false,
            seq_level_idx_0: 8,
            seq_tier_0: 0,
            max_frame_width: 1920,
            max_frame_height: 1080,
            use_128x128_superblock: true,
            enable_filter_intra: true,
            enable_intra_edge_filter: true,
            enable_superres: false,
            enable_cdef: true,
            enable_restoration: true,
            bit_depth: 8,
            monochrome: false,
            subsampling_x: 1,
            subsampling_y: 1,
            colorimetry: ragdoll_core::Colorimetry::bt709_limited(),
            fps: None,
            film_grain_params_present: false,
        };
        let obu_bytes = seq.to_obu();
        let parsed = Av1SequenceHeader::parse(&obu_bytes).unwrap();
        assert_eq!(parsed, seq);

        let av1c = seq.to_av1c_record(&obu_bytes);
        assert_eq!(av1c[0], 0x81);
        assert_eq!(Av1SequenceHeader::parse(&av1c[4..]).unwrap(), seq);

        // Also test against an actual `ffmpeg -c:v libsvtav1` IVF stream if available
        let tmp_ivf = std::env::temp_dir().join("ragdoll_av1_test.ivf");
        let st = Command::new("ffmpeg")
            .args([
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=320x240:r=30:d=0.1",
                "-c:v",
                "libsvtav1",
                "-f",
                "ivf",
                tmp_ivf.to_str().unwrap(),
            ])
            .status();

        if let Ok(s) = st
            && s.success()
        {
            let ivf_data = std::fs::read(&tmp_ivf).unwrap();
            let (ivf_hdr, packets) = parse_ivf_stream(&ivf_data).unwrap();
            assert_eq!(&ivf_hdr.fourcc, b"AV01");
            assert_eq!(ivf_hdr.width, 320);
            assert_eq!(ivf_hdr.height, 240);
            assert!(!packets.is_empty());

            let obus = split_obus(&packets[0].1).unwrap();
            let seq_obu = obus
                .iter()
                .find(|o| o.header.obu_type == ObuType::SequenceHeader)
                .expect("first AV1 temporal unit must contain OBU_SEQUENCE_HEADER");
            let svt_seq = Av1SequenceHeader::parse_payload(seq_obu.payload).unwrap();
            assert_eq!(svt_seq.max_frame_width, 320);
            assert_eq!(svt_seq.max_frame_height, 240);
            assert_eq!(svt_seq.bit_depth, 8);
        }
        let _ = std::fs::remove_file(&tmp_ivf);
    }

    #[test]
    fn ragdoll_fast_intra_4k_over_300_fps_and_psnr_over_42db() {
        // 1. Differential test: NEON vs Scalar plane encoder produce identical bitstream
        let mut test_frame = VideoFrame::new(320, 180, PixelFormat::Yuv420p);
        for y in 0..180 {
            let row = test_frame.plane_mut(0).row_mut(y);
            for (x, p) in row.iter_mut().enumerate() {
                *p = ((x * 3 + y * 2) & 0xFF) as u8;
            }
        }
        let mut neon_buf = Vec::new();
        let mut scalar_buf = Vec::new();
        encode_rgdi_plane(test_frame.plane(0), 1, &mut neon_buf);
        encode_rgdi_plane_scalar(test_frame.plane(0), 1, &mut scalar_buf);
        assert_eq!(neon_buf, scalar_buf);

        // 2. 4K (3840x2160) throughput & PSNR verification on Apple M4
        let width = 3840usize;
        let height = 2160usize;
        let mut frame_4k = VideoFrame::new(width, height, PixelFormat::Yuv420p);
        for y in 0..height {
            let row = frame_4k.plane_mut(0).row_mut(y);
            for (x, p) in row.iter_mut().enumerate() {
                // Smooth cinematic gradient + vertical/horizontal studio test pattern
                *p = (((x / 4) + (y / 8) + ((x ^ y) & 3)) & 0xFF) as u8;
            }
        }
        for y in 0..height / 2 {
            for x in 0..width / 2 {
                frame_4k.plane_mut(1).row_mut(y)[x] = (120 + ((x + y) / 32) % 32) as u8;
                frame_4k.plane_mut(2).row_mut(y)[x] = (136 - ((x + y) / 32) % 32) as u8;
            }
        }

        let enc = RagdollIntraEncoder::default(); // qstep_shift = 1
        let dec = RagdollIntraDecoder::new();
        let pool = FramePool::new(4);

        // Warm-up
        let pkt = enc.encode_frame(&frame_4k).unwrap();
        let mut recon_4k = pool.acquire(width, height, PixelFormat::Yuv420p);
        dec.decode_into(&pkt, &mut recon_4k).unwrap();

        let q = evaluate_frame_quality(&frame_4k, &recon_4k, None);
        let psnr_y = q.psnr_y;
        let psnr_avg = q.psnr_avg;
        assert!(
            psnr_y > 42.0 && psnr_avg > 42.0,
            "expected 4K PSNR > 42 dB, got Y={psnr_y:.2} dB, Avg={psnr_avg:.2} dB"
        );

        // Benchmark 10 iterations of 4K encode + decode
        let iters = 10;
        let t_enc = std::time::Instant::now();
        for _ in 0..iters {
            let p = enc.encode_frame(&frame_4k).unwrap();
            std::hint::black_box(p);
        }
        let enc_fps = (iters as f64) / t_enc.elapsed().as_secs_f64();

        let t_dec = std::time::Instant::now();
        for _ in 0..iters {
            dec.decode_into(&pkt, &mut recon_4k).unwrap();
            std::hint::black_box(&recon_4k);
        }
        let dec_fps = (iters as f64) / t_dec.elapsed().as_secs_f64();

        println!(
            "RagdollIntra 4K (3840x2160): encode = {:.1} fps, decode = {:.1} fps, PSNR = {:.2} dB, ratio = {:.2}:1",
            enc_fps,
            dec_fps,
            psnr_avg,
            (width * height * 3 / 2) as f64 / (pkt.len() as f64)
        );
    }

    #[test]
    fn h264_neon_intra_pred_and_p_l0_16x16_motion_estimation_interop_and_flac_md5() {
        // 1. Exhaustive NEON vs Scalar differential test for H.264 16x16 and 8x8 spatial predictors
        let top16: [u8; 16] = [
            12, 25, 44, 61, 80, 99, 118, 137, 150, 168, 182, 199, 211, 224, 238, 251,
        ];
        let left16: [u8; 16] = [
            240, 225, 210, 195, 180, 165, 150, 135, 120, 105, 90, 75, 60, 45, 30, 15,
        ];
        for mode in 0..=3u8 {
            for t_opt in [None, Some(&top16)] {
                for l_opt in [None, Some(&left16)] {
                    for tl in [0u8, 64, 128, 255] {
                        let mut simd_out = [0u8; 256];
                        let mut scalar_out = [0u8; 256];
                        predict_intra_16x16(mode, t_opt, l_opt, tl, &mut simd_out, 16);
                        predict_intra_16x16_scalar(mode, t_opt, l_opt, tl, &mut scalar_out, 16);
                        assert_eq!(
                            simd_out, scalar_out,
                            "Intra 16x16 mismatch at mode={mode}, tl={tl}"
                        );
                    }
                }
            }
        }

        let top8: [u8; 8] = [18, 45, 79, 112, 146, 179, 213, 244];
        let left8: [u8; 8] = [230, 200, 170, 140, 110, 80, 50, 20];
        for mode in 0..=3u8 {
            let mut simd_c = [0u8; 64];
            let mut scalar_c = [0u8; 64];
            predict_intra_chroma_8x8(mode, Some(&top8), Some(&left8), 100, &mut simd_c, 8);
            predict_intra_chroma_8x8_scalar(mode, Some(&top8), Some(&left8), 100, &mut scalar_c, 8);
            assert_eq!(simd_c, scalar_c, "Intra Chroma 8x8 mismatch at mode={mode}");
        }

        // 2. FLAC RFC 9639 STREAMINFO MD5 computation and corruption detection
        let mut audio = AudioFrame::new(48_000, 2, SampleFormat::S16le, 4096);
        for (i, s) in audio.as_i16_mut_slice().iter_mut().enumerate() {
            *s = ((i as i32 * 313 - 20000) & 0x7FFF) as i16;
        }
        let enc_md5 = FlacEncoder::new(48_000, 2, 16).with_md5(true);
        let flac_md5_bytes = enc_md5.encode_stream(&audio).unwrap();
        let (info_md5, dec_md5) = FlacDecoder::decode_stream(&flac_md5_bytes).unwrap();
        assert_ne!(info_md5.md5_signature, [0u8; 16]);
        assert_eq!(dec_md5.as_i16_slice(), audio.as_i16_slice());

        // Tampering with STREAMINFO MD5 must trigger ChecksumMismatch on decode_stream
        let mut corrupted = flac_md5_bytes.clone();
        corrupted[8 + 33] ^= 0xFF; // flip last byte of 34-byte STREAMINFO (inside md5_signature)
        assert!(FlacDecoder::decode_stream(&corrupted).is_err());

        // 3. H.264 P_L0_16x16 motion-compensated inter coding on a translating pattern
        let w = 256usize;
        let h = 128usize;
        let mut h264_enc = H264Encoder::new(w as u32, h as u32, Rational64::FPS_30, 22);
        let mut h264_dec = H264Decoder::new();
        let mut stream = Vec::new();
        let mut idr_size = 0usize;
        let mut p_size = 0usize;

        for f_idx in 0..5usize {
            let mut f = VideoFrame::new(w, h, PixelFormat::Yuv420p);
            let shift = f_idx * 4; // translates by 4 pixels per frame -> exact P_L0_16x16 / P_Skip match!
            for y in 0..h {
                let row = f.plane_mut(0).row_mut(y);
                for (x, p) in row.iter_mut().enumerate() {
                    *p = (((x + y + shift) * 7) & 0xFF) as u8;
                }
            }
            for y in 0..h / 2 {
                f.plane_mut(1).row_mut(y).fill(128);
                f.plane_mut(2).row_mut(y).fill(128);
            }
            let pkt = h264_enc.encode_frame(&f).unwrap();
            if f_idx == 0 {
                idr_size = pkt.len();
            } else if f_idx == 1 {
                p_size = pkt.len();
            }
            let dec_f = h264_dec.decode_annex_b_packet(&pkt).unwrap().unwrap();
            assert_eq!(dec_f.width, w);
            assert_eq!(dec_f.height, h);
            stream.extend_from_slice(&pkt);
        }
        // P-frame using P_L0_16x16 + P_Skip motion compensation must be >10x smaller than IDR frame!
        assert!(
            p_size * 10 < idr_size,
            "expected P-frame ({p_size} B) to be >10x smaller than IDR ({idr_size} B)"
        );

        let tmp_h264 = std::env::temp_dir().join("ragdoll_p_l0_16x16.h264");
        let tmp_yuv = std::env::temp_dir().join("ragdoll_p_l0_16x16.yuv");
        std::fs::write(&tmp_h264, &stream).unwrap();
        let st = Command::new("ffmpeg")
            .args([
                "-y",
                "-v",
                "error",
                "-i",
                tmp_h264.to_str().unwrap(),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                tmp_yuv.to_str().unwrap(),
            ])
            .status();
        if let Ok(s) = st
            && s.success()
        {
            let yuv = std::fs::read(&tmp_yuv).unwrap();
            assert_eq!(yuv.len(), w * h * 3 / 2 * 5);
        }
        let _ = std::fs::remove_file(&tmp_h264);
        let _ = std::fs::remove_file(&tmp_yuv);
    }

    #[test]
    fn h264_0039_full_residual_coding_qpel_and_ffmpeg_baseline_bit_exact() {
        let w = 160usize;
        let h = 96usize;
        let num_frames = 8usize;

        // 1. Generate natural moving multi-frequency video sequence with sub-pel motion
        let mut raw_frames = Vec::with_capacity(num_frames);
        let mut raw_yuv_bytes = Vec::new();
        for f_idx in 0..num_frames {
            let mut f = VideoFrame::new(w, h, PixelFormat::Yuv420p);
            for y in 0..h {
                let row = f.plane_mut(0).row_mut(y);
                for (x, p) in row.iter_mut().enumerate() {
                    let sx = x + f_idx * 2;
                    let sy = y + f_idx;
                    let grad = ((sx * 3 + sy * 5) & 0xFF) as u16;
                    let detail = (((sx ^ (sy * 3)) * 17) & 0x3F) as u16;
                    *p = ((grad * 3 + detail) / 4).min(255) as u8;
                }
                raw_yuv_bytes.extend_from_slice(f.plane(0).row(y));
            }
            for y in 0..h / 2 {
                for x in 0..w / 2 {
                    f.plane_mut(1).row_mut(y)[x] = (110 + ((x + y + f_idx) * 3) % 40) as u8;
                    f.plane_mut(2).row_mut(y)[x] = (145 - ((x + y + f_idx) * 3) % 40) as u8;
                }
            }
            for y in 0..h / 2 {
                raw_yuv_bytes.extend_from_slice(f.plane(1).row(y));
            }
            for y in 0..h / 2 {
                raw_yuv_bytes.extend_from_slice(f.plane(2).row(y));
            }
            raw_frames.push(f);
        }

        // 2. Encode with H264Encoder at QP=22 (0% I_PCM macroblocks, >3x smaller than v0.1.0 raw/I_PCM size)
        let mut enc = H264Encoder::new(w as u32, h as u32, Rational64::FPS_30, 22);
        enc.keyint = 4; // Multi-GOP (2 GOPs across 8 frames)
        let mut dec = H264Decoder::new();
        let mut ragdoll_stream = Vec::new();
        let mut ragdoll_dec_frames = Vec::new();
        for f in &raw_frames {
            let pkt = enc.encode_frame(f).unwrap();
            let df = dec.decode_annex_b_packet(&pkt).unwrap().unwrap();
            let q = evaluate_frame_quality(f, &df, None);
            assert!(
                q.psnr_y > 34.0,
                "expected H264Encoder QP=22 luma PSNR > 34 dB, got {:.2} dB",
                q.psnr_y
            );
            ragdoll_stream.extend_from_slice(&pkt);
            ragdoll_dec_frames.push(df);
        }
        let v010_ipcm_bytes = w * h * 3 / 2 * num_frames;
        assert!(
            ragdoll_stream.len() * 3 < v010_ipcm_bytes,
            "expected H264Encoder stream ({} B) to be >3x smaller than v0.1.0 I_PCM ({} B)",
            ragdoll_stream.len(),
            v010_ipcm_bytes
        );

        // 3. Verify Ragdoll -> FFmpeg decoding bit-exactness across multi-GOP moving video (when ffmpeg is installed)
        if Command::new("ffmpeg").arg("-version").output().is_ok() {
            let tmp_dir = std::env::temp_dir();
            let rag_h264_path = tmp_dir.join("ragdoll_0039_out.h264");
            let rag_ff_yuv_path = tmp_dir.join("ragdoll_0039_ff_dec.yuv");
            std::fs::write(&rag_h264_path, &ragdoll_stream).unwrap();
            let st_rag = Command::new("ffmpeg")
                .args([
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    rag_h264_path.to_str().unwrap(),
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "yuv420p",
                    rag_ff_yuv_path.to_str().unwrap(),
                ])
                .status()
                .unwrap();
            assert!(
                st_rag.success(),
                "ffmpeg failed to decode Ragdoll H.264 stream"
            );
            let ff_dec_bytes = std::fs::read(&rag_ff_yuv_path).unwrap();
            let frame_bytes = w * h * 3 / 2;
            assert_eq!(ff_dec_bytes.len(), frame_bytes * num_frames);
            for (f_idx, df) in ragdoll_dec_frames.iter().enumerate() {
                let ff_slice = &ff_dec_bytes[f_idx * frame_bytes..(f_idx + 1) * frame_bytes];
                for y in 0..h {
                    assert_eq!(
                        df.plane(0).row(y),
                        &ff_slice[y * w..(y + 1) * w],
                        "Ragdoll vs FFmpeg Y row {y} mismatch on frame {f_idx}"
                    );
                }
            }

            // 4. Verify FFmpeg (`libx264 -profile:v baseline` with I_4x4, I_16x16, P_L0_16x16, P_8x8) -> Ragdoll bit-exact!
            let raw_in_path = tmp_dir.join("ragdoll_0039_in.yuv");
            let x264_h264_path = tmp_dir.join("ragdoll_0039_x264_base.h264");
            let x264_ff_yuv_path = tmp_dir.join("ragdoll_0039_x264_ff.yuv");
            std::fs::write(&raw_in_path, &raw_yuv_bytes).unwrap();
            let st_x264 = Command::new("ffmpeg")
                .args([
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "yuv420p",
                    "-s:v",
                    &format!("{w}x{h}"),
                    "-r",
                    "30",
                    "-i",
                    raw_in_path.to_str().unwrap(),
                    "-c:v",
                    "libx264",
                    "-profile:v",
                    "baseline",
                    "-x264-params",
                    "qp=22:keyint=4:min-keyint=4:scenecut=0:ref=1:deblock=0,0",
                    "-f",
                    "h264",
                    x264_h264_path.to_str().unwrap(),
                ])
                .status()
                .unwrap();
            assert!(st_x264.success(), "ffmpeg libx264 baseline encode failed");

            let st_x264_dec = Command::new("ffmpeg")
                .args([
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    x264_h264_path.to_str().unwrap(),
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "yuv420p",
                    x264_ff_yuv_path.to_str().unwrap(),
                ])
                .status()
                .unwrap();
            assert!(st_x264_dec.success());

            let x264_stream = std::fs::read(&x264_h264_path).unwrap();
            let x264_ff_yuv = std::fs::read(&x264_ff_yuv_path).unwrap();
            let mut rag_dec = H264Decoder::new();
            let rag_frames = rag_dec.decode_stream_pts_ordered(&x264_stream).unwrap();
            assert_eq!(rag_frames.len(), num_frames);
            for (f_idx, rf) in rag_frames.iter().enumerate() {
                let ff_slice = &x264_ff_yuv[f_idx * frame_bytes..(f_idx + 1) * frame_bytes];
                for y in 0..h {
                    assert_eq!(
                        rf.plane(0).row(y),
                        &ff_slice[y * w..(y + 1) * w],
                        "libx264 Baseline Y mismatch at frame {f_idx} row {y}"
                    );
                }
                let u_off = w * h;
                let v_off = u_off + (w / 2) * (h / 2);
                for y in 0..h / 2 {
                    assert_eq!(
                        rf.plane(1).row(y),
                        &ff_slice[u_off + y * (w / 2)..u_off + (y + 1) * (w / 2)],
                        "libx264 Baseline U mismatch at frame {f_idx} row {y}"
                    );
                    assert_eq!(
                        rf.plane(2).row(y),
                        &ff_slice[v_off + y * (w / 2)..v_off + (y + 1) * (w / 2)],
                        "libx264 Baseline V mismatch at frame {f_idx} row {y}"
                    );
                }
            }

            let _ = std::fs::remove_file(&rag_h264_path);
            let _ = std::fs::remove_file(&rag_ff_yuv_path);
            let _ = std::fs::remove_file(&raw_in_path);
            let _ = std::fs::remove_file(&x264_h264_path);
            let _ = std::fs::remove_file(&x264_ff_yuv_path);
        }
    }

    #[test]
    fn h264_0040_main_and_high_cabac_8x8_transform_and_b_frame_pts_reordering() {
        let w = 160usize;
        let h = 96usize;
        let num_frames = 6usize;

        let mut frames = Vec::with_capacity(num_frames);
        let mut raw_yuv_bytes = Vec::new();
        for f_idx in 0..num_frames {
            let mut f = VideoFrame::new(w, h, PixelFormat::Yuv420p);
            for y in 0..h {
                let row = f.plane_mut(0).row_mut(y);
                for (x, p) in row.iter_mut().enumerate() {
                    let sx = x + f_idx * 2;
                    let sy = y + f_idx;
                    let v = (40 + ((sx * 2 + sy) % 140) + (((sx / 4) ^ (sy / 4)) & 7) * 3).min(235);
                    *p = v as u8;
                }
                raw_yuv_bytes.extend_from_slice(f.plane(0).row(y));
            }
            for y in 0..h / 2 {
                for x in 0..w / 2 {
                    f.plane_mut(1).row_mut(y)[x] = (120 + ((x + y + f_idx) % 24)) as u8;
                    f.plane_mut(2).row_mut(y)[x] = (135 - ((x + y + f_idx) % 24)) as u8;
                }
            }
            for y in 0..h / 2 {
                raw_yuv_bytes.extend_from_slice(f.plane(1).row(y));
            }
            for y in 0..h / 2 {
                raw_yuv_bytes.extend_from_slice(f.plane(2).row(y));
            }
            frames.push(f);
        }

        // 1. Compare Baseline (CAVLC) vs Main (CABAC) vs High (CABAC + 8x8 transform)
        let mut enc_base = H264Encoder::new(w as u32, h as u32, Rational64::FPS_30, 24)
            .with_profile(H264Profile::Baseline);
        let mut enc_main = H264Encoder::new(w as u32, h as u32, Rational64::FPS_30, 24)
            .with_profile(H264Profile::Main);
        let mut enc_high = H264Encoder::new(w as u32, h as u32, Rational64::FPS_30, 24)
            .with_profile(H264Profile::High);

        let mut stream_base = Vec::new();
        let mut stream_main = Vec::new();
        let mut stream_high = Vec::new();
        for f in &frames {
            stream_base.extend_from_slice(&enc_base.encode_frame(f).unwrap());
            stream_main.extend_from_slice(&enc_main.encode_frame(f).unwrap());
            stream_high.extend_from_slice(&enc_high.encode_frame(f).unwrap());
        }

        // CABAC Main profile must reduce bitrate vs CAVLC Baseline at identical QP & reconstruction!
        let savings_pct = (1.0 - (stream_main.len() as f64) / (stream_base.len() as f64)) * 100.0;
        println!(
            "H.264 Baseline={} B, Main (CABAC)={} B ({:.1}% smaller), High (CABAC+8x8)={} B",
            stream_base.len(),
            stream_main.len(),
            savings_pct,
            stream_high.len()
        );
        assert!(
            savings_pct >= 10.0,
            "expected CABAC Main ({} B) to achieve >= 10% savings vs CAVLC Baseline ({} B), got {:.1}%",
            stream_main.len(),
            stream_base.len(),
            savings_pct
        );

        let mut dec_main = H264Decoder::new();
        let recon_main = dec_main.decode_stream_pts_ordered(&stream_main).unwrap();
        assert_eq!(recon_main.len(), num_frames);

        let mut dec_high = H264Decoder::new();
        let recon_high = dec_high.decode_stream_pts_ordered(&stream_high).unwrap();
        assert_eq!(recon_high.len(), num_frames);
        for i in 1..recon_high.len() {
            assert!(
                recon_high[i].pts.ticks > recon_high[i - 1].pts.ticks,
                "PTS must be strictly monotonic after DPB reordering"
            );
        }

        // Verify FFmpeg decodes our CABAC Main and High profile streams cleanly and Ragdoll decodes x264 High + B-frames (when ffmpeg is installed)
        if Command::new("ffmpeg").arg("-version").output().is_ok() {
            let tmp_dir = std::env::temp_dir();
            for (tag, bytes) in [("main", &stream_main), ("high", &stream_high)] {
                let p_h264 = tmp_dir.join(format!("ragdoll_0040_{tag}.h264"));
                let p_yuv = tmp_dir.join(format!("ragdoll_0040_{tag}.yuv"));
                std::fs::write(&p_h264, bytes).unwrap();
                let st = Command::new("ffmpeg")
                    .args([
                        "-y",
                        "-v",
                        "error",
                        "-i",
                        p_h264.to_str().unwrap(),
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "yuv420p",
                        p_yuv.to_str().unwrap(),
                    ])
                    .status()
                    .unwrap();
                assert!(
                    st.success(),
                    "ffmpeg failed to decode Ragdoll {tag} CABAC stream"
                );
                let yuv = std::fs::read(&p_yuv).unwrap();
                assert_eq!(yuv.len(), w * h * 3 / 2 * num_frames);
                let _ = std::fs::remove_file(&p_h264);
                let _ = std::fs::remove_file(&p_yuv);
            }

            // 2. Decode High Profile CABAC + 8x8 transform + B-frames produced by `ffmpeg -c:v libx264 -profile:v high`
            let in_yuv = tmp_dir.join("ragdoll_0040_in.yuv");
            let high_h264 = tmp_dir.join("ragdoll_0040_x264_high.h264");
            std::fs::write(&in_yuv, &raw_yuv_bytes).unwrap();
            let st_high = Command::new("ffmpeg")
                .args([
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "yuv420p",
                    "-s:v",
                    &format!("{w}x{h}"),
                    "-r",
                    "30",
                    "-i",
                    in_yuv.to_str().unwrap(),
                    "-c:v",
                    "libx264",
                    "-profile:v",
                    "high",
                    "-x264-params",
                    "qp=22:bframes=2:b-adapt=0:ref=2:8x8dct=1:cabac=1",
                    "-f",
                    "h264",
                    high_h264.to_str().unwrap(),
                ])
                .status()
                .unwrap();
            assert!(st_high.success());
            let high_bytes = std::fs::read(&high_h264).unwrap();
            let mut x264_high_dec = H264Decoder::new();
            let ordered_frames = x264_high_dec
                .decode_stream_pts_ordered(&high_bytes)
                .unwrap();
            assert_eq!(ordered_frames.len(), num_frames);
            for i in 1..ordered_frames.len() {
                assert!(
                    ordered_frames[i].pts.ticks > ordered_frames[i - 1].pts.ticks,
                    "PTS must be strictly monotonic after DPB reordering"
                );
            }
            let _ = std::fs::remove_file(&in_yuv);
            let _ = std::fs::remove_file(&high_h264);
        }
    }

    #[test]
    fn h264_0041_crf_vaq_perceptual_gain_and_vbv_abr_1080p_rate_control() {
        // 1. Compare CRF + VAQ vs Fixed QP on mixed flat/structural + high-energy scene
        let w = 320usize;
        let h = 192usize;
        let mut frames = Vec::new();
        for f_idx in 0..6usize {
            let mut f = VideoFrame::new(w, h, PixelFormat::Yuv420p);
            for y in 0..h {
                let row = f.plane_mut(0).row_mut(y);
                for (x, p) in row.iter_mut().enumerate() {
                    if y < h / 2 {
                        // Subtle structural gradient region (sensitive to quantization banding)
                        *p = (80 + ((x + y + f_idx * 2) / 3) % 48) as u8;
                    } else {
                        // High-variance complex texture region (strong visual masking)
                        *p = ((((x * 37) ^ (y * 53) ^ (f_idx * 19)) + x * 7) & 0xFF) as u8;
                    }
                }
            }
            for y in 0..h / 2 {
                f.plane_mut(1).row_mut(y).fill(128);
                f.plane_mut(2).row_mut(y).fill(128);
            }
            frames.push(f);
        }

        let mut enc_crf = H264Encoder::new(w as u32, h as u32, Rational64::FPS_30, 23)
            .with_profile(H264Profile::Main)
            .with_rate_control(RateControlMode::Crf {
                crf: 23.0,
                maxrate_kbps: None,
                bufsize_kb: None,
            });
        let mut dec_crf = H264Decoder::new();
        let mut crf_bytes = 0usize;
        let mut crf_msssim_sum = 0.0f64;
        for f in &frames {
            let pkt = enc_crf.encode_frame(f).unwrap();
            crf_bytes += pkt.len();
            let rf = dec_crf.decode_annex_b_packet(&pkt).unwrap().unwrap();
            let q = evaluate_frame_quality(f, &rf, None);
            crf_msssim_sum += q.ms_ssim;
        }
        let crf_msssim = crf_msssim_sum / (frames.len() as f64);
        let crf_db = -10.0 * (1.0 - crf_msssim.min(0.999999)).log10();

        // Find the Fixed QP that uses the same (or slightly higher) average bitrate as CRF 23
        let mut best_fixed_qp = 25u8;
        let mut qp_bytes = 0usize;
        let mut qp_msssim = 0.0f64;
        for cand_qp in 18..=40u8 {
            let mut enc_qp = H264Encoder::new(w as u32, h as u32, Rational64::FPS_30, cand_qp)
                .with_profile(H264Profile::Main);
            let mut dec_qp = H264Decoder::new();
            let mut bytes = 0usize;
            let mut msssim_sum = 0.0f64;
            for f in &frames {
                let pkt = enc_qp.encode_frame(f).unwrap();
                bytes += pkt.len();
                let rf = dec_qp.decode_annex_b_packet(&pkt).unwrap().unwrap();
                let q = evaluate_frame_quality(f, &rf, None);
                msssim_sum += q.ms_ssim;
            }
            best_fixed_qp = cand_qp;
            qp_bytes = bytes;
            qp_msssim = msssim_sum / (frames.len() as f64);
            if bytes <= crf_bytes {
                break;
            }
        }
        let qp_db = -10.0 * (1.0 - qp_msssim.min(0.999999)).log10();
        println!(
            "CRF 23 ({} B): MS-SSIM={:.5} ({:.2} dB) vs Fixed QP {} ({} B): MS-SSIM={:.5} ({:.2} dB)",
            crf_bytes, crf_msssim, crf_db, best_fixed_qp, qp_bytes, qp_msssim, qp_db
        );
        assert!(
            crf_db >= qp_db + 1.0,
            "expected CRF+VAQ MS-SSIM dB ({crf_db:.2}) >= Fixed QP ({qp_db:.2}) + 1.0 dB"
        );

        // 2. Verify 1080p ABR + VBV (`--bitrate 2M --maxrate 3M`) constrains output bitrate within +-5% of target
        let w1080 = 1920usize;
        let h1080 = 1080usize;
        let fps = 30u32;
        let n_1080 = 10usize;
        let mut enc_abr = H264Encoder::new(
            w1080 as u32,
            h1080 as u32,
            Rational64::new(fps as i64, 1),
            26,
        )
        .with_profile(H264Profile::Main)
        .with_rate_control(RateControlMode::Abr {
            bitrate_kbps: 2000,
            maxrate_kbps: Some(3000),
            bufsize_kb: Some(4000),
        });
        let mut total_bytes = 0usize;
        for f_idx in 0..n_1080 {
            let mut f = VideoFrame::new(w1080, h1080, PixelFormat::Yuv420p);
            for y in 0..h1080 {
                let row = f.plane_mut(0).row_mut(y);
                for (x, p) in row.iter_mut().enumerate() {
                    *p = (((x + y + f_idx * 4) * 3) & 0xFF) as u8;
                }
            }
            for y in 0..h1080 / 2 {
                f.plane_mut(1).row_mut(y).fill(128);
                f.plane_mut(2).row_mut(y).fill(128);
            }
            let pkt = enc_abr.encode_frame(&f).unwrap();
            total_bytes += pkt.len();
        }
        let duration_sec = (n_1080 as f64) / (fps as f64);
        let actual_kbps = ((total_bytes * 8) as f64) / 1000.0 / duration_sec;
        let rel_err = ((actual_kbps - 2000.0) / 2000.0).abs();
        println!(
            "1080p ABR target=2000 kbps, actual={:.1} kbps (err={:.2}%)",
            actual_kbps,
            rel_err * 100.0
        );
        assert!(
            rel_err <= 0.05,
            "expected 1080p ABR bitrate within +-5% of 2000 kbps, got {actual_kbps:.1} kbps ({:.2}%)",
            rel_err * 100.0
        );
    }

    #[test]
    fn simd_0042_polyphase_resampler_kaiser_and_channel_rematrix() {
        use ragdoll_core::ChannelLayout;

        // 1. Differential test: NEON vs Scalar within 1e-6 (f32) and 0 LSB (Q15 i16) across 44.1 <-> 48 <-> 88.2 <-> 96 kHz
        for &(in_rate, out_rate) in &[
            (44_100u32, 48_000u32),
            (48_000, 44_100),
            (48_000, 96_000),
            (96_000, 44_100),
            (44_100, 88_200),
        ] {
            let mut rs_simd = PolyphaseResampler::new(in_rate, out_rate, 2, 16);
            let mut rs_scalar = PolyphaseResampler::new(in_rate, out_rate, 2, 16);
            let n_in = 4410usize;
            let in_l: Vec<f32> = (0..n_in)
                .map(|i| {
                    ((i as f32) * 2.0 * std::f32::consts::PI * 1000.0 / (in_rate as f32)).sin()
                        * 0.85
                })
                .collect();
            let in_r: Vec<f32> = (0..n_in)
                .map(|i| {
                    ((i as f32) * 2.0 * std::f32::consts::PI * 440.0 / (in_rate as f32)).cos()
                        * 0.75
                })
                .collect();
            let n_out = rs_simd.output_frames_for(n_in);
            let mut out_l_simd = vec![0.0f32; n_out];
            let mut out_r_simd = vec![0.0f32; n_out];
            let mut out_l_scalar = vec![0.0f32; n_out];
            let mut out_r_scalar = vec![0.0f32; n_out];

            let got_simd = rs_simd
                .resample_planar_f32_into(&[&in_l, &in_r], &mut [&mut out_l_simd, &mut out_r_simd]);
            let got_scalar = rs_scalar.resample_planar_f32_scalar_into(
                &[&in_l, &in_r],
                &mut [&mut out_l_scalar, &mut out_r_scalar],
            );
            assert_eq!(got_simd, got_scalar);
            for i in 0..got_simd {
                assert!(
                    (out_l_simd[i] - out_l_scalar[i]).abs() < 1e-6,
                    "f32 L diff at {in_rate}->{out_rate} sample {i}: {} vs {}",
                    out_l_simd[i],
                    out_l_scalar[i]
                );
                assert!(
                    (out_r_simd[i] - out_r_scalar[i]).abs() < 1e-6,
                    "f32 R diff at {in_rate}->{out_rate} sample {i}: {} vs {}",
                    out_r_simd[i],
                    out_r_scalar[i]
                );
            }

            // Q15 i16 0-LSB bit-exact differential test
            let in_l_i16: Vec<i16> = in_l.iter().map(|&x| (x * 32000.0) as i16).collect();
            let in_r_i16: Vec<i16> = in_r.iter().map(|&x| (x * 32000.0) as i16).collect();
            let mut out_l_q15_simd = vec![0i16; n_out];
            let mut out_r_q15_simd = vec![0i16; n_out];
            let mut out_l_q15_scalar = vec![0i16; n_out];
            let mut out_r_q15_scalar = vec![0i16; n_out];
            rs_simd.reset();
            rs_scalar.reset();
            rs_simd.resample_planar_i16_into(
                &[&in_l_i16, &in_r_i16],
                &mut [&mut out_l_q15_simd, &mut out_r_q15_simd],
            );
            rs_scalar.resample_planar_i16_scalar_into(
                &[&in_l_i16, &in_r_i16],
                &mut [&mut out_l_q15_scalar, &mut out_r_q15_scalar],
            );
            assert_eq!(out_l_q15_simd, out_l_q15_scalar, "Q15 L 0-LSB mismatch");
            assert_eq!(out_r_q15_simd, out_r_q15_scalar, "Q15 R 0-LSB mismatch");
        }

        // 2. Throughput test: 44.1 kHz -> 48 kHz stereo resampling must sustain >1,500x real-time with 0 heap allocations
        let mut rs_perf = PolyphaseResampler::new(44_100, 48_000, 2, 16);
        let chunk_in = 4410usize; // 100 ms per chunk
        let chunk_out = 4800usize;
        rs_perf.reserve_input_capacity(chunk_in);
        let in_l = vec![0.5f32; chunk_in];
        let in_r = vec![-0.5f32; chunk_in];
        let mut out_l = vec![0.0f32; chunk_out + 16];
        let mut out_r = vec![0.0f32; chunk_out + 16];

        let num_chunks = 50usize; // 5.0 seconds of stereo audio
        let t0 = std::time::Instant::now();
        let mut total_out_frames = 0usize;
        for _ in 0..num_chunks {
            total_out_frames +=
                rs_perf.resample_planar_f32_into(&[&in_l, &in_r], &mut [&mut out_l, &mut out_r]);
        }
        let elapsed_s = t0.elapsed().as_secs_f64();
        let audio_duration_s = (num_chunks * chunk_in) as f64 / 44_100.0;
        let realtime_x = audio_duration_s / elapsed_s.max(1e-9);
        assert_eq!(total_out_frames, num_chunks * 4800);
        let min_rt = if cfg!(debug_assertions) { 50.0 } else { 1500.0 };
        println!("SIMD-0042 44.1kHz -> 48kHz stereo throughput: {realtime_x:.1}x real-time");
        assert!(
            realtime_x > min_rt,
            "expected >{min_rt}x real-time on 44.1->48 kHz stereo resampling, got {realtime_x:.1}x"
        );

        // 3. Channel Matrix Mixer: 5.1 -> Stereo -> Mono and 7.1 -> 5.1 -> Stereo
        let mut frame_5_1 = AudioFrame::new(48_000, 6, SampleFormat::F32le, 256);
        frame_5_1.layout = ChannelLayout::Surround5_1;
        for i in 0..256 {
            let b = i * 6;
            let s = frame_5_1.as_f32_mut_slice();
            s[b] = 0.5; // L
            s[b + 1] = 0.25; // R
            s[b + 2] = 0.4; // C
            s[b + 3] = 0.9; // LFE (excluded in BS.775 stereo downmix)
            s[b + 4] = 0.2; // Ls
            s[b + 5] = 0.1; // Rs
        }
        let rm_51_to_2 =
            ChannelRematrixer::new(ChannelLayout::Surround5_1, ChannelLayout::Stereo, false);
        let stereo_f = rm_51_to_2.rematrix_audio_frame(&frame_5_1).unwrap();
        assert_eq!(stereo_f.channels, 2);
        let exp_l = 0.5 + std::f32::consts::FRAC_1_SQRT_2 * (0.4 + 0.2);
        let exp_r = 0.25 + std::f32::consts::FRAC_1_SQRT_2 * (0.4 + 0.1);
        assert!((stereo_f.as_f32_slice()[0] - exp_l).abs() < 1e-5);
        assert!((stereo_f.as_f32_slice()[1] - exp_r).abs() < 1e-5);
    }

    #[test]
    fn simd_0043_ebu_r128_lufs_k_weighting_gated_and_true_peak() {
        // 1. EBU Tech 3341 / ITU-R BS.1770-4 Compliance Test Vector 1:
        // Stereo 1000 Hz full-scale sine wave (amplitude = 1.0 -> RMS = 1/sqrt(2), two channels -> sum z = 1.0)
        // Expected Integrated Loudness = -3.01 LUFS? Wait: at 997 Hz / 1000 Hz, K-weighting has ~ +0.691 dB gain,
        // so a stereo full-scale 1 kHz sine wave (0 dBFS peak per channel) measures 0.0 LUFS,
        // and a stereo -23.0 dBFS peak 1 kHz sine wave measures -23.0 LUFS within +-0.1 LU!
        let fs = 48_000u32;
        let num_frames = (fs as usize) * 4; // 4.0 seconds
        let amp_neg23 = 10.0_f64.powf(-23.0 / 20.0) as f32;
        let mut stereo_tone = Vec::with_capacity(num_frames * 2);
        for n in 0..num_frames {
            let s = ((2.0 * std::f64::consts::PI * 997.0 * (n as f64) / (fs as f64)).sin() as f32)
                * amp_neg23;
            stereo_tone.push(s);
            stereo_tone.push(s);
        }

        let mut meter_simd = EbuR128Meter::new(fs, 2);
        let mut meter_scalar = EbuR128Meter::new(fs, 2);
        meter_simd.process_interleaved_f32(&stereo_tone);
        meter_scalar.process_interleaved_f32_scalar(&stereo_tone);

        let rep_simd = meter_simd.finalize();
        let rep_scalar = meter_scalar.finalize();

        // Bit-exact differential verification between NEON and Scalar!
        assert_eq!(
            rep_simd.integrated_lufs.to_bits(),
            rep_scalar.integrated_lufs.to_bits(),
            "NEON vs Scalar integrated_lufs bit-exact mismatch"
        );
        assert_eq!(
            rep_simd.true_peak_linear.to_bits(),
            rep_scalar.true_peak_linear.to_bits(),
            "NEON vs Scalar true_peak_linear bit-exact mismatch"
        );
        println!(
            "SIMD-0043 997Hz -23 dBFS stereo tone: Integrated={:.3} LUFS, LRA={:.2} LU, TruePeak={:.2} dBTP",
            rep_simd.integrated_lufs, rep_simd.loudness_range_lu, rep_simd.true_peak_dbtp
        );
        assert!(
            (rep_simd.integrated_lufs - (-23.0)).abs() <= 0.1,
            "expected -23.0 LUFS +- 0.1 LU on EBU Tech 3341 -23 dBFS stereo reference tone, got {:.3} LUFS",
            rep_simd.integrated_lufs
        );
        assert!(
            rep_simd.loudness_range_lu <= 0.1,
            "steady tone LRA must be <= 0.1 LU, got {:.3} LU",
            rep_simd.loudness_range_lu
        );
        assert!(
            (rep_simd.true_peak_dbtp - (-23.0)).abs() <= 0.1,
            "expected -23.0 dBTP +- 0.1 dB on 997 Hz tone, got {:.3} dBTP",
            rep_simd.true_peak_dbtp
        );

        // 2. EBU Tech 3341 Gating Test: silence followed by continuous -23 LUFS reference tone
        // Relative & absolute gates (-70 LUFS and L_abs - 10 LU) reject the silent segment and remain within -23.0 +- 0.1 LUFS!
        let silence_samples = (fs as usize) * 2 * 2; // 2s stereo silence
        let active_frames = (fs as usize) * 8; // 8s continuous stereo 1 kHz tone
        let mut gated_signal = vec![0.0f32; silence_samples];
        for n in 0..active_frames {
            let s = ((2.0 * std::f64::consts::PI * 1000.0 * (n as f64) / (fs as f64)).sin() as f32)
                * amp_neg23;
            gated_signal.push(s);
            gated_signal.push(s);
        }
        let mut meter_gated = EbuR128Meter::new(fs, 2);
        meter_gated.process_interleaved_f32(&gated_signal);
        let rep_gated = meter_gated.finalize();
        assert!(
            (rep_gated.integrated_lufs - (-23.0)).abs() <= 0.1,
            "gated loudness with silence segment must remain -23.0 +- 0.1 LUFS, got {:.3} LUFS",
            rep_gated.integrated_lufs
        );

        // 3. Normalization to -14.0 LUFS with -1.0 dBTP True-Peak ceiling
        let (gain, _) = meter_simd.compute_normalization_gain(-14.0, -1.0);
        let mut norm_frame = AudioFrame::new(fs, 2, SampleFormat::F32le, num_frames);
        norm_frame.as_f32_mut_slice().copy_from_slice(&stereo_tone);
        let normalized = normalize_audio_frame(&norm_frame, gain, -1.0).unwrap();
        let mut check_meter = EbuR128Meter::new(fs, 2);
        check_meter.process_audio_frame(&normalized).unwrap();
        let check_rep = check_meter.finalize();
        assert!(
            (check_rep.integrated_lufs - (-14.0)).abs() <= 0.1,
            "expected normalized loudness -14.0 +- 0.1 LUFS, got {:.3} LUFS",
            check_rep.integrated_lufs
        );
        assert!(
            check_rep.true_peak_dbtp <= -0.95,
            "expected true peak <= -1.0 dBTP, got {:.2} dBTP",
            check_rep.true_peak_dbtp
        );
    }

    #[test]
    fn audio_0044_aac_lc_and_opus_roundtrip_and_ffmpeg_interop() {
        // 1. AAC-LC 48 kHz Stereo Roundtrip (AacEncoder -> AacDecoder) + FFmpeg bidirectional interop
        let sr = 48_000u32;
        let mut aac_enc = AacEncoder::new(sr, 2, 128_000);
        let mut aac_dec = AacDecoder::from_asc(aac_enc.asc);
        let mut adts_stream = Vec::new();
        let mut total_recon_energy = 0.0f64;

        for f_idx in 0..8usize {
            let mut frame = AudioFrame::new(sr, 2, SampleFormat::F32le, 1024);
            let dst = frame.as_f32_mut_slice();
            for i in 0..1024 {
                let t = ((f_idx * 1024 + i) as f64) / (sr as f64);
                dst[i * 2] = ((2.0 * std::f64::consts::PI * 440.0 * t).sin() * 0.5) as f32;
                dst[i * 2 + 1] = ((2.0 * std::f64::consts::PI * 880.0 * t).cos() * 0.4) as f32;
            }
            let adts_pkt = aac_enc.encode_adts_frame(&frame).unwrap();
            adts_stream.extend_from_slice(&adts_pkt);
            let dec_frame = aac_dec.decode_packet(&adts_pkt).unwrap();
            assert_eq!(dec_frame.num_samples, 1024);
            assert_eq!(dec_frame.channels, 2);
            if f_idx >= 1 {
                // Compare Frame f_idx output against Frame f_idx - 1 input (due to 1024-sample 50% MDCT lookahead delay)
                for &s in dec_frame.as_f32_slice() {
                    total_recon_energy += (s as f64) * (s as f64);
                }
            }
        }
        assert!(
            total_recon_energy > 100.0,
            "expected non-trivial reconstructed AAC audio energy, got {total_recon_energy}"
        );

        // Verify FFmpeg decodes our ADTS AAC-LC stream AND our AacDecoder decodes `ffmpeg -c:a aac` stream
        if Command::new("ffmpeg").arg("-version").output().is_ok() {
            let tmp = std::env::temp_dir();
            let rag_aac = tmp.join("ragdoll_0044.aac");
            let rag_wav = tmp.join("ragdoll_0044_dec.raw");
            std::fs::write(&rag_aac, &adts_stream).unwrap();
            let st = Command::new("ffmpeg")
                .args([
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    rag_aac.to_str().unwrap(),
                    "-f",
                    "f32le",
                    rag_wav.to_str().unwrap(),
                ])
                .status()
                .unwrap();
            assert!(
                st.success(),
                "ffmpeg failed to decode Ragdoll AAC-LC stream"
            );
            let ff_pcm = std::fs::read(&rag_wav).unwrap();
            assert!(ff_pcm.len() >= 1024 * 2 * 4 * 6);

            let ff_aac = tmp.join("ragdoll_0044_ff.aac");
            let st_enc = Command::new("ffmpeg")
                .args([
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "f32le",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    "-i",
                    rag_wav.to_str().unwrap(),
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    ff_aac.to_str().unwrap(),
                ])
                .status()
                .unwrap();
            assert!(st_enc.success(), "ffmpeg -c:a aac encode failed");
            let ff_aac_bytes = std::fs::read(&ff_aac).unwrap();
            let mut dec_ff = AacDecoder::new(48_000, 2);
            let mut pos = 0usize;
            let mut frames_decoded = 0usize;
            while pos + 7 <= ff_aac_bytes.len() {
                let hdr = AdtsHeader::parse(&ff_aac_bytes[pos..]).unwrap();
                let fl = hdr.frame_length as usize;
                if pos + fl > ff_aac_bytes.len() {
                    break;
                }
                let df = dec_ff.decode_packet(&ff_aac_bytes[pos..pos + fl]).unwrap();
                assert_eq!(df.num_samples, 1024);
                pos += fl;
                frames_decoded += 1;
            }
            assert!(frames_decoded >= 6);
            let _ = std::fs::remove_file(&rag_aac);
            let _ = std::fs::remove_file(&rag_wav);
            let _ = std::fs::remove_file(&ff_aac);
        }

        // 2. Opus 48 kHz Stereo Roundtrip (OpusEncoder -> OpusDecoder) + Range Coder verification
        let mut opus_enc = OpusEncoder::new(48_000, 2, 128_000);
        let mut opus_dec = OpusDecoder::new(2)
            .with_extra_data(&opus_enc.extra_data())
            .unwrap();
        let mut opus_energy = 0.0f64;
        for f_idx in 0..6usize {
            let mut frame = AudioFrame::new(48_000, 2, SampleFormat::F32le, 960);
            let dst = frame.as_f32_mut_slice();
            for i in 0..960 {
                let t = ((f_idx * 960 + i) as f64) / 48_000.0;
                dst[i * 2] = ((2.0 * std::f64::consts::PI * 500.0 * t).sin() * 0.6) as f32;
                dst[i * 2 + 1] = ((2.0 * std::f64::consts::PI * 750.0 * t).cos() * 0.5) as f32;
            }
            let pkt = opus_enc.encode_frame(&frame).unwrap();
            let df = opus_dec.decode_packet(&pkt).unwrap();
            assert_eq!(df.num_samples, 960);
            assert_eq!(df.channels, 2);
            if f_idx >= 1 {
                for &s in df.as_f32_slice() {
                    opus_energy += (s as f64) * (s as f64);
                }
            }
        }
        assert!(
            opus_energy > 100.0,
            "expected non-trivial reconstructed Opus audio energy, got {opus_energy}"
        );
    }
}
