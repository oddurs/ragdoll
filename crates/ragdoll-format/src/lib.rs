//! # `ragdoll-format`
//!
//! Zero-copy memory-mapped container demuxers (`PacketSlice<'mmap>`) and vectored I/O (`writev`)
//! muxers for Ragdoll:
//! - [`io`]: Read-only OS `mmap` wrapper ([`MmapSource`]) and vectored writer ([`VectoredWriter`]).
//! - [`mp4`]: ISOBMFF / MP4 / MOV zero-copy demuxer with flat 24-byte [`ragdoll_core::SampleEntry`]
//!   tables, $O(\log N)$ keyframe seeking, and single-pass Fast-Start (`moov` before `mdat`) muxing.
//! - [`mkv`]: Matroska (`.mkv`) and WebM (`.webm`) zero-copy demuxer and muxer with branchless
//!   `CLZ` EBML VINT parsing and `Cues` keyframe index generation.
//! - [`mpegts`]: SIMD-synced 188-byte MPEG-2 Transport Stream (`.ts`) demuxer and muxer with
//!   hardware Slicing-by-8 `crc32_mpeg2` PAT/PMT tables and automatic AVCC $\leftrightarrow$ Annex-B framing.
//! - [`y4m`]: Zero-copy `YUV4MPEG2` (`.y4m`) uncompressed planar video reader and writer.
//! - [`ppm`]: Binary Netpbm `P6` (`Rgb24`) and `P5` (`Gray8`) image I/O and multi-frame contact-sheet composer.
//! - [`wav`]: Zero-copy RIFF WAVE (`.wav`) PCM & IEEE-Float audio demuxer and muxer.

pub mod io;
pub mod mkv;
pub mod mp4;
pub mod mpegts;
pub mod ppm;
pub mod wav;
pub mod y4m;

pub use io::{MmapSource, VectoredWriter};
pub use mkv::{MkvDemuxer, MkvMuxer};
pub use mp4::{FourCC, Mp4Demuxer, Mp4Muxer, Mp4TrackIndex};
pub use mpegts::{
    FEC_PACKET_SIZE, M2TS_PACKET_SIZE, MpegTsDemuxer, MpegTsMuxer, PID_BASE_ES, PID_DEFAULT_PMT,
    PID_PAT, PID_SDT, TS_PACKET_SIZE, TS_SYNC_BYTE, aac_raw_to_adts, build_avcc_from_sps_pps,
    decode_pes_timestamp, detect_mpegts_stride_and_sync, encode_pes_timestamp,
    h264_annex_b_to_avcc_au, h264_avcc_to_annex_b_au, parse_adts_header,
};
pub use ppm::{compose_contact_sheet_rgb24, read_pnm_image, write_pnm_image};
pub use wav::{WavDemuxer, WavMuxer};
pub use y4m::{Y4mDemuxer, Y4mMuxer};

use ragdoll_core::{AlignedBuffer, CodecId, PacketSlice, RagdollError, Result, TrackInfo};
use std::io::Write;

/// Container format family detected by magic bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ContainerFormat {
    Mp4,
    Matroska,
    WebM,
    MpegTs,
    Y4m,
    Ppm,
    Pgm,
    Wav,
}

impl std::fmt::Display for ContainerFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Mp4 => "mp4/isobmff",
            Self::Matroska => "matroska",
            Self::WebM => "webm",
            Self::MpegTs => "mpegts",
            Self::Y4m => "yuv4mpeg2",
            Self::Ppm => "ppm",
            Self::Pgm => "pgm",
            Self::Wav => "wav",
        };
        write!(f, "{s}")
    }
}

/// Detects the container format of `data` from its leading magic bytes or sync markers.
pub fn detect_container_format(data: &[u8]) -> Option<ContainerFormat> {
    if data.len() >= 12 && &data[4..8] == b"ftyp" {
        return Some(ContainerFormat::Mp4);
    }
    if data.len() >= 8
        && (&data[4..8] == b"moov"
            || &data[4..8] == b"moof"
            || &data[4..8] == b"mdat"
            || &data[4..8] == b"wide")
    {
        return Some(ContainerFormat::Mp4);
    }
    if data.len() >= 4 && data[0..4] == [0x1A, 0x45, 0xDF, 0xA3] {
        // Check if header mentions "webm"
        let head = &data[..data.len().min(64)];
        if head.windows(4).any(|w| w == b"webm") {
            return Some(ContainerFormat::WebM);
        }
        return Some(ContainerFormat::Matroska);
    }
    if data.len() >= 10 && &data[0..10] == b"YUV4MPEG2 " {
        return Some(ContainerFormat::Y4m);
    }
    if data.len() >= 12
        && (&data[0..4] == b"RIFF" || &data[0..4] == b"RF64" || &data[0..4] == b"BW64")
        && &data[8..12] == b"WAVE"
    {
        return Some(ContainerFormat::Wav);
    }
    if data.len() >= 3 && &data[0..2] == b"P6" && data[2].is_ascii_whitespace() {
        return Some(ContainerFormat::Ppm);
    }
    if data.len() >= 3 && &data[0..2] == b"P5" && data[2].is_ascii_whitespace() {
        return Some(ContainerFormat::Pgm);
    }
    if detect_mpegts_stride_and_sync(data).is_some() {
        return Some(ContainerFormat::MpegTs);
    }
    None
}

/// Probes any supported container in `data` and returns its `(ContainerFormat, Vec<TrackInfo>)`.
pub fn probe_container(data: &[u8]) -> Result<(ContainerFormat, Vec<TrackInfo>)> {
    let Some(fmt) = detect_container_format(data) else {
        return Err(RagdollError::InvalidHeader(
            "unrecognized container magic bytes".to_string(),
        ));
    };
    match fmt {
        ContainerFormat::Mp4 => {
            let demux = Mp4Demuxer::open(data)?;
            let tracks = demux.tracks().iter().map(|t| t.info.clone()).collect();
            Ok((fmt, tracks))
        }
        ContainerFormat::Matroska | ContainerFormat::WebM => {
            let demux = MkvDemuxer::open(data)?;
            Ok((fmt, demux.tracks().to_vec()))
        }
        ContainerFormat::MpegTs => {
            let demux = MpegTsDemuxer::open(data)?;
            Ok((fmt, demux.tracks().to_vec()))
        }
        ContainerFormat::Y4m => {
            let demux = Y4mDemuxer::open(data)?;
            Ok((fmt, vec![demux.track().clone()]))
        }
        ContainerFormat::Wav => {
            let demux = WavDemuxer::open(data)?;
            Ok((fmt, vec![demux.track().clone()]))
        }
        ContainerFormat::Ppm | ContainerFormat::Pgm => {
            let frame = read_pnm_image(data)?;
            let info = TrackInfo {
                index: 0,
                track_type: ragdoll_core::TrackType::Video,
                codec_id: ragdoll_core::CodecId::RawVideo,
                timebase: ragdoll_core::Rational64::ONE,
                duration_ticks: Some(1),
                extra_data: Vec::new(),
                video: Some(ragdoll_core::VideoTrackParams {
                    width: frame.width as u32,
                    height: frame.height as u32,
                    pixel_format: frame.format,
                    ..Default::default()
                }),
                audio: None,
                sample_count: 1,
                total_bytes: data.len() as u64,
            };
            Ok((fmt, vec![info]))
        }
    }
}

/// Zero-copy transmuxer from MP4/ISOBMFF (`mp4_bytes`) into Matroska (`.mkv`).
pub fn transmux_mp4_to_mkv<W: Write>(mp4_bytes: &[u8], writer: W) -> Result<u64> {
    let mut demux = Mp4Demuxer::open(mp4_bytes)?;
    let mut mux = MkvMuxer::new_matroska();
    for t in demux.tracks() {
        mux.add_track(t.info.clone());
    }
    while let Some(pkt) = demux.next_packet()? {
        mux.write_packet(pkt)?;
    }
    mux.finish(writer)
}

/// Zero-copy transmuxer from MP4/ISOBMFF (`mp4_bytes`) into MPEG-2 Transport Stream (`.ts`).
pub fn transmux_mp4_to_mpegts<W: Write>(mp4_bytes: &[u8], writer: W) -> Result<u64> {
    let mut demux = Mp4Demuxer::open(mp4_bytes)?;
    let mut mux = MpegTsMuxer::new(writer);
    for t in demux.tracks() {
        mux.add_track(t.info.clone());
    }
    while let Some(pkt) = demux.next_packet()? {
        mux.write_packet(&pkt)?;
    }
    mux.finish()
}

/// Zero-copy transmuxer from Matroska/WebM (`mkv_bytes`) into Fast-Start MP4 (`.mp4`).
pub fn transmux_mkv_to_mp4<W: Write>(mkv_bytes: &[u8], writer: W) -> Result<u64> {
    let mut demux = MkvDemuxer::open(mkv_bytes)?;
    let mut mux = Mp4Muxer::new();
    for t in demux.tracks() {
        mux.add_track(t.clone());
    }
    while let Some(pkt) = demux.next_packet()? {
        mux.write_packet(pkt)?;
    }
    mux.finish(writer)
}

/// Transmuxes an MPEG-2 Transport Stream (`ts_bytes`, 188B TS / 192B M2TS / 204B FEC) into a
/// Fast-Start MP4 (`.mp4`), automatically converting H.264 Annex-B NAL units into 4-byte
/// length-prefixed AVCC samples and stripping ADTS headers into raw AAC + `AudioSpecificConfig`.
pub fn transmux_mpegts_to_mp4<W: Write>(ts_bytes: &[u8], writer: W) -> Result<u64> {
    let mut demux = MpegTsDemuxer::open(ts_bytes)?;
    let tracks: Vec<TrackInfo> = demux.tracks().to_vec();
    let mut mux = Mp4Muxer::new();
    for t in &tracks {
        mux.add_track(t.clone());
    }
    while let Some(pkt) = demux.next_packet()? {
        let track = &tracks[pkt.track_index as usize];
        if track.codec_id == CodecId::H264 {
            let avcc_au = h264_annex_b_to_avcc_au(pkt.data());
            let mut owned = PacketSlice::from_owned(
                AlignedBuffer::from_slice(&avcc_au),
                pkt.track_index,
                pkt.pts,
                pkt.dts,
                pkt.duration,
                pkt.flags,
            );
            owned.file_offset = pkt.file_offset;
            mux.write_packet(owned)?;
        } else {
            mux.write_packet(pkt)?;
        }
    }
    mux.finish(writer)
}

/// Transmuxes an MPEG-2 Transport Stream (`ts_bytes`, 188B TS / 192B M2TS / 204B FEC) into a
/// Matroska (`.mkv`) container with `SeekHead` and `Cues`.
pub fn transmux_mpegts_to_mkv<W: Write>(ts_bytes: &[u8], writer: W) -> Result<u64> {
    let mut demux = MpegTsDemuxer::open(ts_bytes)?;
    let tracks: Vec<TrackInfo> = demux.tracks().to_vec();
    let mut mux = MkvMuxer::new_matroska();
    for t in &tracks {
        mux.add_track(t.clone());
    }
    while let Some(pkt) = demux.next_packet()? {
        let track = &tracks[pkt.track_index as usize];
        if track.codec_id == CodecId::H264 {
            let avcc_au = h264_annex_b_to_avcc_au(pkt.data());
            let mut owned = PacketSlice::from_owned(
                AlignedBuffer::from_slice(&avcc_au),
                pkt.track_index,
                pkt.pts,
                pkt.dts,
                pkt.duration,
                pkt.flags,
            );
            owned.file_offset = pkt.file_offset;
            mux.write_packet(owned)?;
        } else {
            mux.write_packet(pkt)?;
        }
    }
    mux.finish(writer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ragdoll_core::{
        AudioTrackParams, ChannelLayout, CodecId, Colorimetry, PacketFlags, PacketSlice,
        PixelFormat, Rational64, SampleFormat, Timestamp, TrackInfo, TrackType, VideoFrame,
        VideoTrackParams,
    };

    /// Synthesizes a valid minimal H.264 Baseline `SPS` + `PPS` + `IDR` / `P` frame bitstream in AVCC format
    /// so `ffprobe` / `ffmpeg` can genuinely parse and decode the video stream!
    fn make_valid_h264_avcc_stream() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        // Minimal valid 16x16 H.264 Baseline Profile SPS (NAL type 0x67) and PPS (NAL type 0x68)
        let sps: &[u8] = &[
            0x67, 0x42, 0xC0, 0x0A, 0xD9, 0x01, 0x00, 0x4E, 0xFF, 0x00, 0x01, 0x00, 0x01, 0xE2,
            0x44, 0x54,
        ];
        let pps: &[u8] = &[0x68, 0xCE, 0x3C, 0x80];
        let avcc = build_avcc_from_sps_pps(sps, pps);

        // IDR slice NAL (type 5) with 4-byte AVCC length prefix
        let idr_nal: &[u8] = &[0x65, 0x88, 0x84, 0x21, 0xA0, 0x24, 0x05, 0x3E];
        let mut idr_avcc = Vec::with_capacity(4 + idr_nal.len());
        idr_avcc.extend_from_slice(&(idr_nal.len() as u32).to_be_bytes());
        idr_avcc.extend_from_slice(idr_nal);

        // Non-IDR P-slice NAL (type 1) with 4-byte AVCC length prefix
        let p_nal: &[u8] = &[0x41, 0x9A, 0x24, 0x6C, 0x42, 0x10];
        let mut p_avcc = Vec::with_capacity(4 + p_nal.len());
        p_avcc.extend_from_slice(&(p_nal.len() as u32).to_be_bytes());
        p_avcc.extend_from_slice(p_nal);

        (avcc, idr_avcc, p_avcc)
    }

    #[test]
    fn mp4_faststart_zero_copy_demux_and_log_n_keyframe_seeking() {
        let (avcc, idr_pkt, p_pkt) = make_valid_h264_avcc_stream();
        let tb = Rational64::new(1, 30_000);
        let frame_dur = 1001i64; // 29.97 fps

        let mut mux = Mp4Muxer::new();
        let v_track = mux.add_track(TrackInfo {
            index: 0,
            track_type: TrackType::Video,
            codec_id: CodecId::H264,
            timebase: tb,
            duration_ticks: Some(300 * frame_dur),
            extra_data: avcc.clone(),
            video: Some(VideoTrackParams {
                width: 1920,
                height: 1080,
                pixel_format: PixelFormat::Yuv420p,
                colorimetry: Colorimetry::bt709_limited(),
                frame_rate: Rational64::FPS_29_97,
                bit_depth: 8,
                profile: None,
                level: None,
            }),
            audio: None,
            sample_count: 300,
            total_bytes: 0,
        });

        // Emit 300 frames (10 GOPs of 30 frames each, with B-frame CTS offsets on non-keyframes)
        for i in 0..300usize {
            let is_kf = (i % 30) == 0;
            let dts = (i as i64) * frame_dur;
            // Alternate positive and negative CTS offsets to test version-1 signed ctts
            let cts_off = if is_kf {
                0
            } else if i % 2 == 1 {
                frame_dur
            } else {
                -(frame_dur / 2)
            };
            let pts = dts + cts_off;
            let bytes = if is_kf { &idr_pkt[..] } else { &p_pkt[..] };
            let flags = if is_kf {
                PacketFlags::KEYFRAME
            } else {
                PacketFlags::NONE
            };
            mux.write_packet(PacketSlice::from_mmap(
                bytes,
                v_track,
                Timestamp::new(pts, tb),
                Timestamp::new(dts, tb),
                frame_dur,
                flags,
                0,
            ))
            .unwrap();
        }

        let mut mp4_buf = Vec::new();
        mux.finish(&mut mp4_buf).unwrap();

        // Wrap in MmapSource and demux zero-copy
        let mmap = MmapSource::from_slice(&mp4_buf);
        let mmap_range = mmap.as_slice().as_ptr_range();
        let mut demux = Mp4Demuxer::open(&mmap).unwrap();

        assert!(
            demux.is_faststart(),
            "moov must precede mdat for fast-start"
        );
        assert_eq!(demux.tracks().len(), 1);
        assert_eq!(demux.tracks()[0].samples.len(), 300);
        assert_eq!(demux.tracks()[0].info.extra_data, avcc);

        // Verify O(log N) keyframe seeking across GOP boundaries
        // Frame 77 is in GOP #2 (keyframes at 0, 30, 60, 90, ... -> must land on sample 60!)
        let target_ts = Timestamp::new(77 * frame_dur, tb);
        let (kf_idx, kf_entry) = demux.seek_to_timestamp(0, target_ts).unwrap();
        assert_eq!(kf_idx, 60);
        assert!(kf_entry.is_keyframe());
        assert_eq!(kf_entry.pts, 60 * frame_dur);

        // Verify zero-copy PacketSlice emission (every slice pointer points inside `mmap`)
        let first_after_seek = demux.next_packet().unwrap().unwrap();
        assert!(first_after_seek.payload.is_zero_copy());
        assert!(mmap_range.contains(&first_after_seek.data().as_ptr()));
        assert!(first_after_seek.is_keyframe());
        assert_eq!(first_after_seek.pts.ticks, 60 * frame_dur);
        assert_eq!(first_after_seek.data(), &idr_pkt[..]);
    }

    #[test]
    fn zero_copy_transmux_mp4_to_mkv_and_mpegts_roundtrip() {
        let (avcc, idr_pkt, p_pkt) = make_valid_h264_avcc_stream();
        let tb = Rational64::new(1, 1000);

        let mut mp4_mux = Mp4Muxer::new();
        mp4_mux.add_track(TrackInfo {
            index: 0,
            track_type: TrackType::Video,
            codec_id: CodecId::H264,
            timebase: tb,
            duration_ticks: Some(2000),
            extra_data: avcc.clone(),
            video: Some(VideoTrackParams {
                width: 1280,
                height: 720,
                pixel_format: PixelFormat::Yuv420p,
                colorimetry: Colorimetry::bt709_limited(),
                frame_rate: Rational64::FPS_30,
                bit_depth: 8,
                profile: None,
                level: None,
            }),
            audio: None,
            sample_count: 60,
            total_bytes: 0,
        });

        for i in 0..60usize {
            let is_kf = (i % 15) == 0;
            let ts = Timestamp::new((i * 33) as i64, tb);
            let payload = if is_kf { &idr_pkt[..] } else { &p_pkt[..] };
            let flags = if is_kf {
                PacketFlags::KEYFRAME
            } else {
                PacketFlags::NONE
            };
            mp4_mux
                .write_packet(PacketSlice::from_mmap(payload, 0, ts, ts, 33, flags, 0))
                .unwrap();
        }

        let mut mp4_bytes = Vec::new();
        mp4_mux.finish(&mut mp4_bytes).unwrap();

        // 1. Transmux MP4 -> MKV
        let mut mkv_bytes = Vec::new();
        transmux_mp4_to_mkv(&mp4_bytes, &mut mkv_bytes).unwrap();
        assert_eq!(
            detect_container_format(&mkv_bytes),
            Some(ContainerFormat::Matroska)
        );

        let mut mkv_demux = MkvDemuxer::open(&mkv_bytes).unwrap();
        assert_eq!(mkv_demux.tracks().len(), 1);
        assert_eq!(mkv_demux.tracks()[0].codec_id, CodecId::H264);
        assert_eq!(mkv_demux.tracks()[0].extra_data, avcc);
        assert_eq!(mkv_demux.total_packets(), 60);

        let mkv_ptr_range = mkv_bytes.as_slice().as_ptr_range();
        let mut mkv_cnt = 0usize;
        while let Some(pkt) = mkv_demux.next_packet().unwrap() {
            assert!(pkt.payload.is_zero_copy());
            assert!(mkv_ptr_range.contains(&pkt.data().as_ptr()));
            assert_eq!(pkt.is_keyframe(), mkv_cnt.is_multiple_of(15));
            mkv_cnt += 1;
        }
        assert_eq!(mkv_cnt, 60);

        // 2. Transmux MKV -> Fast-Start MP4
        let mut roundtrip_mp4 = Vec::new();
        transmux_mkv_to_mp4(&mkv_bytes, &mut roundtrip_mp4).unwrap();
        let rt_demux = Mp4Demuxer::open(&roundtrip_mp4).unwrap();
        assert!(rt_demux.is_faststart());
        assert_eq!(rt_demux.total_packets(), 60);

        // 3. Transmux MP4 -> MPEG-TS (.ts)
        let mut ts_bytes = Vec::new();
        transmux_mp4_to_mpegts(&mp4_bytes, &mut ts_bytes).unwrap();
        assert_eq!(ts_bytes.len() % TS_PACKET_SIZE, 0);
        assert_eq!(
            detect_container_format(&ts_bytes),
            Some(ContainerFormat::MpegTs)
        );

        let mut ts_demux = MpegTsDemuxer::open(&ts_bytes).unwrap();
        assert_eq!(ts_demux.tracks().len(), 1);
        assert_eq!(ts_demux.tracks()[0].codec_id, CodecId::H264);
        assert!(!ts_demux.tracks()[0].extra_data.is_empty());
        assert_eq!(ts_demux.total_packets(), 60);

        let mut ts_cnt = 0usize;
        while let Some(pkt) = ts_demux.next_packet().unwrap() {
            if ts_cnt.is_multiple_of(15) {
                assert!(pkt.is_keyframe());
            }
            ts_cnt += 1;
        }
        assert_eq!(ts_cnt, 60);
    }

    #[test]
    fn y4m_ppm_contact_sheet_and_wav_studio_containers_roundtrip() {
        // 1. Y4M (YUV4MPEG2) 4:2:0 round-trip
        let mut f0 = VideoFrame::new(32, 24, PixelFormat::Yuv420p);
        let mut f1 = VideoFrame::new(32, 24, PixelFormat::Yuv420p);
        for (i, frame) in [&mut f0, &mut f1].into_iter().enumerate() {
            for (p_idx, plane) in frame.planes.iter_mut().enumerate() {
                for y in 0..plane.height() {
                    let row = plane.row_mut(y);
                    for (x, sample) in row.iter_mut().enumerate() {
                        *sample = ((i * 53 + p_idx * 29 + y * 7 + x) & 0xFF) as u8;
                    }
                }
            }
        }

        let mut y4m_buf = Vec::new();
        let mut y4m_mux = Y4mMuxer::new(
            &mut y4m_buf,
            32,
            24,
            PixelFormat::Yuv420p,
            Rational64::FPS_60,
            Colorimetry::bt709_limited(),
        )
        .unwrap();
        y4m_mux.write_frame(&f0).unwrap();
        y4m_mux.write_frame(&f1).unwrap();
        y4m_mux.finish().unwrap();

        let mut y4m_demux = Y4mDemuxer::open(&y4m_buf).unwrap();
        assert_eq!(y4m_demux.frame_count(), 2);
        assert_eq!(
            y4m_demux.track().video.as_ref().unwrap().frame_rate,
            Rational64::FPS_60
        );

        let mut dec0 = VideoFrame::new(1, 1, PixelFormat::Yuv420p);
        let mut dec1 = VideoFrame::new(1, 1, PixelFormat::Yuv420p);
        assert!(y4m_demux.read_frame_into(&mut dec0).unwrap());
        assert!(y4m_demux.read_frame_into(&mut dec1).unwrap());
        assert!(!y4m_demux.read_frame_into(&mut dec0).unwrap());

        for p in 0..3 {
            for y in 0..f0.planes[p].height() {
                assert_eq!(dec0.planes[p].row(y), f0.planes[p].row(y));
                assert_eq!(dec1.planes[p].row(y), f1.planes[p].row(y));
            }
        }

        // 2. PPM (P6) + Contact-Sheet Grid Composer round-trip
        let mut rgb_a = VideoFrame::new(16, 12, PixelFormat::Rgb24);
        let mut rgb_b = VideoFrame::new(16, 12, PixelFormat::Rgb24);
        for y in 0..12 {
            for x in 0..16 * 3 {
                rgb_a.planes[0].row_mut(y)[x] = ((y * 19 + x * 3) & 0xFF) as u8;
                rgb_b.planes[0].row_mut(y)[x] = ((y * 11 + x * 7 + 100) & 0xFF) as u8;
            }
        }
        let grid =
            compose_contact_sheet_rgb24(&[&rgb_a, &rgb_b, &rgb_b, &rgb_a], 2, 4, [16, 24, 32])
                .unwrap();
        assert_eq!(grid.width, 2 * 16 + 3 * 4);
        assert_eq!(grid.height, 2 * 12 + 3 * 4);

        let mut ppm_bytes = Vec::new();
        write_pnm_image(&mut ppm_bytes, &grid).unwrap();
        let decoded_grid = read_pnm_image(&ppm_bytes).unwrap();
        assert_eq!(decoded_grid.width, grid.width);
        assert_eq!(decoded_grid.height, grid.height);
        for y in 0..grid.height {
            assert_eq!(decoded_grid.planes[0].row(y), grid.planes[0].row(y));
        }

        // 3. RIFF WAVE (s16le & f32le) zero-copy round-trip
        let pcm_samples: Vec<u8> = (0..4096).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
        let audio_params = AudioTrackParams {
            sample_rate: 48_000,
            channels: 2,
            channel_layout: ChannelLayout::Stereo,
            sample_format: SampleFormat::S16le,
            bits_per_coded_sample: 16,
        };
        let mut wav_mux = WavMuxer::new(audio_params);
        wav_mux.write_pcm_bytes(&pcm_samples[..2048]);
        wav_mux.write_pcm_bytes(&pcm_samples[2048..]);
        let mut wav_bytes = Vec::new();
        wav_mux.finish(&mut wav_bytes).unwrap();

        let mut wav_demux = WavDemuxer::open(&wav_bytes).unwrap();
        assert_eq!(wav_demux.pcm_data(), &pcm_samples[..]);
        let pkt = wav_demux.next_packet().unwrap().unwrap();
        assert!(pkt.payload.is_zero_copy());
        assert_eq!(pkt.data(), &pcm_samples[..]);
    }

    #[test]
    fn ffprobe_and_ffmpeg_bidirectional_interop() {
        use std::process::Command;

        // Check if ffprobe and ffmpeg are available on the host
        let has_ffprobe = Command::new("ffprobe")
            .arg("-version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if !has_ffprobe {
            return;
        }

        let tmp_dir = std::env::temp_dir().join(format!("ragdoll_m3_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp_dir);

        // 1. Write Y4M, PPM, and WAV using Ragdoll and verify with `ffprobe -v error`
        let y4m_path = tmp_dir.join("test.y4m");
        {
            let mut f = VideoFrame::new(64, 48, PixelFormat::Yuv420p);
            for (p_idx, plane) in f.planes.iter_mut().enumerate() {
                for y in 0..plane.height() {
                    for (x, s) in plane.row_mut(y).iter_mut().enumerate() {
                        *s = (16 + ((x + y + p_idx * 30) % 200)) as u8;
                    }
                }
            }
            let file = std::fs::File::create(&y4m_path).unwrap();
            let mut mux = Y4mMuxer::new(
                file,
                64,
                48,
                PixelFormat::Yuv420p,
                Rational64::FPS_30,
                Colorimetry::bt709_limited(),
            )
            .unwrap();
            for _ in 0..10 {
                mux.write_frame(&f).unwrap();
            }
            mux.finish().unwrap();
        }

        let ppm_path = tmp_dir.join("test.ppm");
        {
            let f = VideoFrame::new(64, 48, PixelFormat::Rgb24);
            let file = std::fs::File::create(&ppm_path).unwrap();
            write_pnm_image(file, &f).unwrap();
        }

        let wav_path = tmp_dir.join("test.wav");
        {
            let pcm = vec![0u8; 48_000 * 2 * 2 / 10]; // 100ms stereo s16le
            let mut mux = WavMuxer::new(AudioTrackParams {
                sample_rate: 48_000,
                channels: 2,
                channel_layout: ChannelLayout::Stereo,
                sample_format: SampleFormat::S16le,
                bits_per_coded_sample: 16,
            });
            mux.write_pcm_bytes(&pcm);
            let file = std::fs::File::create(&wav_path).unwrap();
            mux.finish(file).unwrap();
        }

        for p in [&y4m_path, &ppm_path, &wav_path] {
            let out = Command::new("ffprobe")
                .args([
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=codec_name,width,height,sample_rate,channels",
                    "-of",
                    "default=noprint_wrappers=1",
                ])
                .arg(p)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "ffprobe failed on {:?}: {}",
                p,
                String::from_utf8_lossy(&out.stderr)
            );
        }

        // 2. Use ffmpeg to encode our Ragdoll-generated Y4M + WAV into a real H.264 + AAC MP4,
        //    then demux it via Ragdoll's MmapSource + Mp4Demuxer, transmux it to Fast-Start MP4,
        //    MKV, and MPEG-TS using Ragdoll, and verify all 3 outputs with `ffprobe -v error`!
        let ffmpeg_mp4_path = tmp_dir.join("ffmpeg_ref.mp4");
        let enc_status = Command::new("ffmpeg")
            .args([
                "-y",
                "-v",
                "error",
                "-i",
                y4m_path.to_str().unwrap(),
                "-i",
                wav_path.to_str().unwrap(),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-g",
                "5",
                "-c:a",
                "aac",
                "-shortest",
                ffmpeg_mp4_path.to_str().unwrap(),
            ])
            .status();

        if let Ok(st) = enc_status
            && st.success()
        {
            let mmap = MmapSource::open(&ffmpeg_mp4_path).unwrap();
            let mut demux = Mp4Demuxer::open(&mmap).unwrap();
            assert!(demux.tracks().len() >= 2);

            // Rewrite as Fast-Start MP4 via Ragdoll Mp4Muxer
            let ragdoll_mp4_path = tmp_dir.join("ragdoll_faststart.mp4");
            {
                let mut mux = Mp4Muxer::new();
                for t in demux.tracks() {
                    mux.add_track(t.info.clone());
                }
                while let Some(pkt) = demux.next_packet().unwrap() {
                    assert!(pkt.payload.is_zero_copy());
                    mux.write_packet(pkt).unwrap();
                }
                let f = std::fs::File::create(&ragdoll_mp4_path).unwrap();
                mux.finish(f).unwrap();
            }

            // Verify fast-start property on Ragdoll's MP4
            let rg_mmap = MmapSource::open(&ragdoll_mp4_path).unwrap();
            let rg_demux = Mp4Demuxer::open(&rg_mmap).unwrap();
            assert!(rg_demux.is_faststart());

            // Transmux to MKV and MPEG-TS via Ragdoll
            let ragdoll_mkv_path = tmp_dir.join("ragdoll.mkv");
            {
                let f = std::fs::File::create(&ragdoll_mkv_path).unwrap();
                transmux_mp4_to_mkv(&rg_mmap, f).unwrap();
            }

            let ragdoll_ts_path = tmp_dir.join("ragdoll.ts");
            {
                let f = std::fs::File::create(&ragdoll_ts_path).unwrap();
                transmux_mp4_to_mpegts(&rg_mmap, f).unwrap();
            }

            // Write Fragmented MP4 (`fMP4`) via Ragdoll `finish_fragmented`
            let ragdoll_fmp4_path = tmp_dir.join("ragdoll_fmp4.mp4");
            {
                let mut r_demux = Mp4Demuxer::open(&rg_mmap).unwrap();
                let mut f_mux = Mp4Muxer::new();
                for t in r_demux.tracks() {
                    f_mux.add_track(t.info.clone());
                }
                while let Some(pkt) = r_demux.next_packet().unwrap() {
                    f_mux.write_packet(pkt).unwrap();
                }
                let f = std::fs::File::create(&ragdoll_fmp4_path).unwrap();
                f_mux.finish_fragmented(f, 5).unwrap();
            }

            // Write RF64 64-bit WAVE via Ragdoll `finish_rf64`
            let ragdoll_rf64_path = tmp_dir.join("ragdoll_rf64.wav");
            {
                let pcm = vec![0u8; 4800 * 4];
                let mut w_mux = WavMuxer::new(AudioTrackParams {
                    sample_rate: 48_000,
                    channels: 2,
                    channel_layout: ChannelLayout::Stereo,
                    sample_format: SampleFormat::F32le,
                    bits_per_coded_sample: 32,
                });
                w_mux.write_pcm_bytes(&pcm);
                let f = std::fs::File::create(&ragdoll_rf64_path).unwrap();
                w_mux.finish_rf64(f).unwrap();
            }

            // Transmux MPEG-TS back to MP4 via Ragdoll `transmux_mpegts_to_mp4`
            let ts_mmap = MmapSource::open(&ragdoll_ts_path).unwrap();
            let ragdoll_ts_to_mp4_path = tmp_dir.join("ragdoll_ts_to_mp4.mp4");
            {
                let f = std::fs::File::create(&ragdoll_ts_to_mp4_path).unwrap();
                transmux_mpegts_to_mp4(&ts_mmap, f).unwrap();
            }

            for p in [
                &ragdoll_mp4_path,
                &ragdoll_fmp4_path,
                &ragdoll_mkv_path,
                &ragdoll_ts_path,
                &ragdoll_rf64_path,
                &ragdoll_ts_to_mp4_path,
            ] {
                let out = Command::new("ffprobe")
                    .args([
                        "-v",
                        "error",
                        "-show_entries",
                        "format=format_name:stream=codec_name",
                        "-of",
                        "default=noprint_wrappers=1",
                    ])
                    .arg(p)
                    .output()
                    .unwrap();
                assert!(
                    out.status.success(),
                    "ffprobe failed on Ragdoll container {:?}: {}",
                    p,
                    String::from_utf8_lossy(&out.stderr)
                );
            }

            // Also generate an FFmpeg-authored Fragmented MP4 (`-movflags frag_keyframe+empty_moov`)
            // and verify Ragdoll's Mp4Demuxer parses all `moof`/`traf`/`trun` fragments zero-copy!
            let ffmpeg_fmp4_path = tmp_dir.join("ffmpeg_fmp4.mp4");
            let fmp4_status = Command::new("ffmpeg")
                .args([
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    ffmpeg_mp4_path.to_str().unwrap(),
                    "-c",
                    "copy",
                    "-movflags",
                    "frag_keyframe+empty_moov",
                    ffmpeg_fmp4_path.to_str().unwrap(),
                ])
                .status();
            if let Ok(fst) = fmp4_status
                && fst.success()
            {
                let fmp4_mmap = MmapSource::open(&ffmpeg_fmp4_path).unwrap();
                let mut fmp4_demux = Mp4Demuxer::open(&fmp4_mmap).unwrap();
                assert!(fmp4_demux.is_fragmented());
                assert_eq!(fmp4_demux.total_packets(), demux.total_packets());
                let first_pkt = fmp4_demux.next_packet().unwrap().unwrap();
                assert!(first_pkt.payload.is_zero_copy());
            }
        }

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn fmp4_fragmented_mux_demux_and_mkv_seekhead_log_n_seeking() {
        let (avcc, idr_pkt, p_pkt) = make_valid_h264_avcc_stream();
        let tb = Rational64::new(1, 30_000);
        let frame_dur = 1001i64;

        let mut mux = Mp4Muxer::new();
        let v_track = mux.add_track(TrackInfo {
            index: 0,
            track_type: TrackType::Video,
            codec_id: CodecId::H264,
            timebase: tb,
            duration_ticks: Some(120 * frame_dur),
            extra_data: avcc.clone(),
            video: Some(VideoTrackParams {
                width: 1920,
                height: 1080,
                pixel_format: PixelFormat::Yuv420p,
                colorimetry: Colorimetry::bt709_limited(),
                frame_rate: Rational64::FPS_29_97,
                bit_depth: 8,
                profile: None,
                level: None,
            }),
            audio: None,
            sample_count: 120,
            total_bytes: 0,
        });

        for i in 0..120usize {
            let is_kf = i.is_multiple_of(15);
            let dts = (i as i64) * frame_dur;
            let cts = if is_kf { 0 } else { frame_dur };
            let pts = dts + cts;
            let payload = if is_kf { &idr_pkt[..] } else { &p_pkt[..] };
            let flags = if is_kf {
                PacketFlags::KEYFRAME
            } else {
                PacketFlags::NONE
            };
            mux.write_packet(PacketSlice::from_mmap(
                payload,
                v_track,
                Timestamp::new(pts, tb),
                Timestamp::new(dts, tb),
                frame_dur,
                flags,
                0,
            ))
            .unwrap();
        }

        let mut fmp4_bytes = Vec::new();
        mux.finish_fragmented(&mut fmp4_bytes, 15).unwrap();

        let mut fmp4_demux = Mp4Demuxer::open(&fmp4_bytes).unwrap();
        assert!(fmp4_demux.is_fragmented());
        assert_eq!(fmp4_demux.total_packets(), 120);

        let (kf_idx, kf_entry) = fmp4_demux
            .seek_to_timestamp(0, Timestamp::new(50 * frame_dur, tb))
            .unwrap();
        assert_eq!(kf_idx, 45);
        assert!(kf_entry.is_keyframe());

        // Transmux fMP4 -> MKV and verify SeekHead + Cues + O(log N) MkvDemuxer::seek_to_timestamp
        let mut mkv_bytes = Vec::new();
        transmux_mp4_to_mkv(&fmp4_bytes, &mut mkv_bytes).unwrap();

        let mut mkv_demux = MkvDemuxer::open(&mkv_bytes).unwrap();
        assert!(mkv_demux.has_seek_head());
        assert_eq!(mkv_demux.cues_count(), 8); // 120 frames / 15-frame GOP = 8 cue points
        let (mkv_kf_idx, mkv_kf_entry) = mkv_demux
            .seek_to_timestamp(0, Timestamp::new(50 * frame_dur, tb))
            .unwrap();
        assert_eq!(mkv_kf_idx, 45);
        assert!(mkv_kf_entry.is_keyframe());
    }

    #[test]
    fn m2ts_192_byte_sdt_adts_aac_and_rf64_wave_roundtrip() {
        let (avcc, idr_pkt, p_pkt) = make_valid_h264_avcc_stream();
        let tb = Rational64::new(1, 90_000);

        // 1. 192-byte Blu-ray M2TS with H.264 + AAC + DVB SDT
        let mut m2ts_bytes = Vec::new();
        {
            let mut mux = MpegTsMuxer::new_m2ts(&mut m2ts_bytes);
            let v_idx = mux.add_track(TrackInfo {
                index: 0,
                track_type: TrackType::Video,
                codec_id: CodecId::H264,
                timebase: tb,
                duration_ticks: None,
                extra_data: avcc.clone(),
                video: Some(VideoTrackParams::default()),
                audio: None,
                sample_count: 0,
                total_bytes: 0,
            });
            let a_idx = mux.add_track(TrackInfo {
                index: 1,
                track_type: TrackType::Audio,
                codec_id: CodecId::Aac,
                timebase: tb,
                duration_ticks: None,
                extra_data: vec![0x11, 0x90], // AAC-LC 48kHz stereo
                video: None,
                audio: Some(AudioTrackParams {
                    sample_rate: 48_000,
                    channels: 2,
                    channel_layout: ChannelLayout::Stereo,
                    sample_format: SampleFormat::F32le,
                    bits_per_coded_sample: 16,
                }),
                sample_count: 0,
                total_bytes: 0,
            });

            let raw_aac_frame = [0x21u8, 0x10, 0x04, 0x60, 0x8C, 0x1C];
            for i in 0..10usize {
                let is_kf = i.is_multiple_of(5);
                let ts = Timestamp::new((i * 3000) as i64, tb);
                let v_pay = if is_kf { &idr_pkt[..] } else { &p_pkt[..] };
                let v_flags = if is_kf {
                    PacketFlags::KEYFRAME
                } else {
                    PacketFlags::NONE
                };
                mux.write_packet(&PacketSlice::from_mmap(
                    v_pay, v_idx, ts, ts, 3000, v_flags, 0,
                ))
                .unwrap();
                mux.write_packet(&PacketSlice::from_mmap(
                    &raw_aac_frame,
                    a_idx,
                    ts,
                    ts,
                    1920,
                    PacketFlags::KEYFRAME,
                    0,
                ))
                .unwrap();
            }
            mux.finish().unwrap();
        }

        assert_eq!(m2ts_bytes.len() % M2TS_PACKET_SIZE, 0);
        assert_eq!(
            detect_container_format(&m2ts_bytes),
            Some(ContainerFormat::MpegTs)
        );
        let m2ts_demux = MpegTsDemuxer::open(&m2ts_bytes).unwrap();
        assert_eq!(m2ts_demux.packet_stride(), M2TS_PACKET_SIZE);
        assert!(m2ts_demux.has_sdt());
        assert_eq!(m2ts_demux.tracks().len(), 2);
        assert_eq!(m2ts_demux.tracks()[1].codec_id, CodecId::Aac);
        assert_eq!(m2ts_demux.tracks()[1].extra_data, vec![0x11, 0x90]);

        // Transmux M2TS -> MP4 and M2TS -> MKV
        let mut mp4_from_m2ts = Vec::new();
        transmux_mpegts_to_mp4(&m2ts_bytes, &mut mp4_from_m2ts).unwrap();
        let mp4_demux = Mp4Demuxer::open(&mp4_from_m2ts).unwrap();
        assert_eq!(mp4_demux.tracks().len(), 2);
        assert_eq!(mp4_demux.total_packets(), 20);

        let mut mkv_from_m2ts = Vec::new();
        transmux_mpegts_to_mkv(&m2ts_bytes, &mut mkv_from_m2ts).unwrap();
        let mkv_demux = MkvDemuxer::open(&mkv_from_m2ts).unwrap();
        assert_eq!(mkv_demux.tracks().len(), 2);
        assert_eq!(mkv_demux.total_packets(), 20);

        // 2. RF64 64-bit WAVE + IEEE Float `fact` chunk round-trip
        let f32_pcm: Vec<u8> = (0..1024).map(|i| (i & 0xFF) as u8).collect();
        let mut rf64_mux = WavMuxer::new(AudioTrackParams {
            sample_rate: 48_000,
            channels: 2,
            channel_layout: ChannelLayout::Stereo,
            sample_format: SampleFormat::F32le,
            bits_per_coded_sample: 32,
        });
        rf64_mux.write_pcm_bytes(&f32_pcm);
        let mut rf64_bytes = Vec::new();
        rf64_mux.finish_rf64(&mut rf64_bytes).unwrap();
        assert_eq!(&rf64_bytes[0..4], b"RF64");
        assert_eq!(
            detect_container_format(&rf64_bytes),
            Some(ContainerFormat::Wav)
        );
        let rf64_demux = WavDemuxer::open(&rf64_bytes).unwrap();
        assert!(rf64_demux.is_rf64());
        assert_eq!(rf64_demux.track().codec_id, CodecId::PcmF32Le);
        assert_eq!(rf64_demux.pcm_data(), &f32_pcm[..]);
    }
}
