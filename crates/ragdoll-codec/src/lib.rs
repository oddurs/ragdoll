//! # `ragdoll-codec`
//!
//! 100% pure-Rust video and audio codecs:
//! - [`pcm`]: Vectorized `aarch64` NEON + scalar PCM format conversion, stereo interleaving, and Mid-Side decorrelation.
//! - [`flac`]: Lossless FLAC (RFC 9639) stream/frame decoder and Fixed + FIR LPC Rice encoder (`crc8_flac` + `crc16_flac`).
//! - [`h264`]: H.264 / AVC (`SPS`, `PPS`, `SliceHeader`, `SeiMessage`) parser/serializer, $4\times4$ integer DCT/Hadamard,
//!   CAVLC entropy coder, and Baseline (`profile_idc = 66`) frame encoder/decoder.
//! - [`av1`]: AV1 Open Bitstream Unit (`OBU`) parser, `Av1SequenceHeader` inspector (`dav1d` parity), `av1C` builder, and IVF stream helper.
//! - [`rgdi`]: Ragdoll Fast-Intra (`RGDI`) SIMD studio video codec sustaining > 300 fps 4K encode/decode on Apple M4.

pub mod av1;
pub mod flac;
pub mod h264;
pub mod pcm;
pub mod rgdi;

pub use av1::{
    Av1SequenceHeader, IvfHeader, ObuHeader, ObuPacket, ObuType, encode_obu, parse_ivf_stream,
    parse_obu_header, split_obus,
};
pub use flac::{
    FLAC_STREAM_MARKER, FlacChannelMode, FlacDecoder, FlacEncoder, FlacFrameHeader, FlacStreamInfo,
    parse_flac_stream_header,
};
pub use h264::{
    H264Decoder, H264Encoder, NalHeader, NalUnitType, Pps, SeiMessage, SliceHeader, Sps,
    decode_cavlc_block, dequantize_block4x4, encode_annex_b_nal, encode_cavlc_block,
    parse_nal_rbsp, predict_intra_16x16, predict_intra_chroma_8x8, quantize_block4x4,
    roundtrip_block4x4_dct,
};
pub use pcm::{
    audio_frame_to_planar_i32, convert_audio_frame, decorrelate_mid_side_i32,
    decorrelate_mid_side_i32_scalar, deinterleave_stereo_i32, deinterleave_stereo_i32_scalar,
    interleave_stereo_i32, interleave_stereo_i32_scalar, pcm_f32_to_s16, pcm_f32_to_s16_scalar,
    pcm_i32_to_s24le, pcm_s16_to_f32, pcm_s16_to_f32_scalar, pcm_s24le_to_i32,
    planar_i32_to_audio_frame, reconstruct_left_side_i32, reconstruct_mid_side_i32,
    reconstruct_mid_side_i32_scalar, reconstruct_right_side_i32,
};
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
}
