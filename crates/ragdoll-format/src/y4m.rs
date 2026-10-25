use crate::io::VectoredWriter;
use ragdoll_core::{
    ChromaLocation, CodecId, ColorRange, Colorimetry, PacketFlags, PacketSlice, PixelFormat,
    RagdollError, Rational64, Result, SampleEntry, Timestamp, TrackInfo, TrackType, VideoFrame,
    VideoTrackParams,
};
use std::io::Write;

/// Zero-copy `YUV4MPEG2` (`.y4m`) Demuxer (`libavformat/yuv4mpegdec.c` compatible).
pub struct Y4mDemuxer<'a> {
    data: &'a [u8],
    track: TrackInfo,
    aspect_ratio: Rational64,
    frames: Vec<SampleEntry>,
    cursor: usize,
}

impl<'a> Y4mDemuxer<'a> {
    pub fn open(data: &'a [u8]) -> Result<Self> {
        const MAGIC: &[u8] = b"YUV4MPEG2 ";
        if data.len() < MAGIC.len() || &data[..MAGIC.len()] != MAGIC {
            return Err(RagdollError::InvalidHeader(
                "missing 'YUV4MPEG2 ' stream header".to_string(),
            ));
        }

        let Some(nl_rel) = data[MAGIC.len()..].iter().position(|&b| b == b'\n') else {
            return Err(RagdollError::InvalidHeader(
                "unterminated YUV4MPEG2 stream header".to_string(),
            ));
        };
        let hdr_end = MAGIC.len() + nl_rel;
        let hdr_str = std::str::from_utf8(&data[MAGIC.len()..hdr_end])
            .map_err(|_| RagdollError::InvalidHeader("non-UTF8 YUV4MPEG2 header".to_string()))?;

        let mut width = 0u32;
        let mut height = 0u32;
        let mut frame_rate = Rational64::FPS_30;
        let mut aspect_ratio = Rational64::ONE;
        let mut pixel_format = PixelFormat::Yuv420p;
        let mut colorimetry = Colorimetry::bt709_limited();
        let mut bit_depth = 8u8;

        for token in hdr_str.split_ascii_whitespace() {
            if let Some(w_str) = token.strip_prefix('W') {
                width = w_str.parse::<u32>().unwrap_or(0);
            } else if let Some(h_str) = token.strip_prefix('H') {
                height = h_str.parse::<u32>().unwrap_or(0);
            } else if let Some(f_str) = token.strip_prefix('F') {
                if let Some((n, d)) = f_str.split_once(':') {
                    let num = n.parse::<i64>().unwrap_or(30);
                    let den = d.parse::<i64>().unwrap_or(1);
                    if num > 0 && den > 0 {
                        frame_rate = Rational64::new(num, den);
                    }
                }
            } else if let Some(a_str) = token.strip_prefix('A') {
                if let Some((n, d)) = a_str.split_once(':') {
                    let num = n.parse::<i64>().unwrap_or(1);
                    let den = d.parse::<i64>().unwrap_or(1);
                    if num > 0 && den > 0 {
                        aspect_ratio = Rational64::new(num, den);
                    }
                }
            } else if let Some(c_str) = token.strip_prefix('C') {
                match c_str {
                    "420" | "420mpeg2" => {
                        pixel_format = PixelFormat::Yuv420p;
                        colorimetry.chroma_location = ChromaLocation::Left;
                    }
                    "420jpeg" => {
                        pixel_format = PixelFormat::Yuv420p;
                        colorimetry.chroma_location = ChromaLocation::Center;
                    }
                    "420paldv" => {
                        pixel_format = PixelFormat::Yuv420p;
                        colorimetry.chroma_location = ChromaLocation::TopLeft;
                    }
                    "422" => {
                        pixel_format = PixelFormat::Yuv422p;
                    }
                    "444" => {
                        pixel_format = PixelFormat::Yuv444p;
                    }
                    "420p10" => {
                        pixel_format = PixelFormat::Yuv420p10le;
                        bit_depth = 10;
                    }
                    "mono" => {
                        pixel_format = PixelFormat::Gray8;
                    }
                    _ => {
                        return Err(RagdollError::Unsupported(format!(
                            "unsupported Y4M colorspace tag C{c_str}"
                        )));
                    }
                }
            } else if let Some(range_val) = token.strip_prefix("XCOLORRANGE=") {
                if range_val.eq_ignore_ascii_case("FULL") {
                    colorimetry.range = ColorRange::Full;
                } else if range_val.eq_ignore_ascii_case("LIMITED") {
                    colorimetry.range = ColorRange::Limited;
                }
            } else if let Some(xyscss) = token.strip_prefix("XYSCSS=") {
                match xyscss.to_ascii_lowercase().as_str() {
                    "420" | "420mpeg2" => {
                        pixel_format = PixelFormat::Yuv420p;
                        colorimetry.chroma_location = ChromaLocation::Left;
                    }
                    "420jpeg" => {
                        pixel_format = PixelFormat::Yuv420p;
                        colorimetry.chroma_location = ChromaLocation::Center;
                    }
                    "420paldv" => {
                        pixel_format = PixelFormat::Yuv420p;
                        colorimetry.chroma_location = ChromaLocation::TopLeft;
                    }
                    "422" => {
                        pixel_format = PixelFormat::Yuv422p;
                    }
                    "444" => {
                        pixel_format = PixelFormat::Yuv444p;
                    }
                    "420p10" => {
                        pixel_format = PixelFormat::Yuv420p10le;
                        bit_depth = 10;
                    }
                    _ => {}
                }
            }
        }

        if width == 0 || height == 0 || width > 16384 || height > 16384 {
            return Err(RagdollError::InvalidHeader(
                "Y4M header missing valid W or H dimensions".to_string(),
            ));
        }

        let frame_bytes = pixel_format.packed_frame_bytes(width as usize, height as usize);
        let mut frames = Vec::new();
        let mut pos = hdr_end + 1;

        while pos < data.len() {
            if data[pos..].len() < 6 || &data[pos..pos + 5] != b"FRAME" {
                return Err(RagdollError::CorruptBitstream(
                    "expected 'FRAME' marker in YUV4MPEG2 stream",
                ));
            }
            let Some(frame_nl) = data[pos..].iter().position(|&b| b == b'\n') else {
                return Err(RagdollError::CorruptBitstream(
                    "unterminated 'FRAME' line in YUV4MPEG2 stream",
                ));
            };
            let payload_start = pos + frame_nl + 1;
            let payload_end = payload_start + frame_bytes;
            if payload_end > data.len() {
                return Err(RagdollError::CorruptBitstream(
                    "truncated frame payload in YUV4MPEG2 stream",
                ));
            }
            let frame_idx = frames.len() as i64;
            frames.push(SampleEntry::new(
                payload_start as u64,
                frame_idx,
                frame_bytes as u32,
                0,
                true,
            ));
            pos = payload_end;
        }

        // Timebase is 1 / frame_rate = (den / num) seconds per frame
        let timebase = frame_rate.recip();
        let total_bytes = (frames.len() * frame_bytes) as u64;
        let track = TrackInfo {
            index: 0,
            track_type: TrackType::Video,
            codec_id: CodecId::RawVideo,
            timebase,
            duration_ticks: Some(frames.len() as i64),
            extra_data: Vec::new(),
            video: Some(VideoTrackParams {
                width,
                height,
                pixel_format,
                colorimetry,
                frame_rate,
                bit_depth,
                profile: None,
                level: None,
            }),
            audio: None,
            sample_count: frames.len(),
            total_bytes,
        };

        Ok(Self {
            data,
            track,
            aspect_ratio,
            frames,
            cursor: 0,
        })
    }

    #[inline]
    pub const fn track(&self) -> &TrackInfo {
        &self.track
    }

    #[inline]
    pub const fn aspect_ratio(&self) -> Rational64 {
        self.aspect_ratio
    }

    #[inline]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Emits the next raw frame as a zero-copy [`PacketSlice<'a>`] borrowing directly from the memory-mapped file.
    #[inline]
    pub fn next_packet(&mut self) -> Result<Option<PacketSlice<'a>>> {
        if self.cursor >= self.frames.len() {
            return Ok(None);
        }
        let entry = self.frames[self.cursor];
        self.cursor += 1;
        let start = entry.offset as usize;
        let end = start + entry.size as usize;
        let ts = Timestamp::new(entry.pts, self.track.timebase);
        Ok(Some(PacketSlice::from_mmap(
            &self.data[start..end],
            0,
            ts,
            ts,
            1,
            PacketFlags::KEYFRAME,
            entry.offset,
        )))
    }

    /// Reads the next Y4M frame directly into a 64-byte cache-line-aligned [`VideoFrame`].
    pub fn read_frame_into(&mut self, dst: &mut VideoFrame) -> Result<bool> {
        let Some(pkt) = self.next_packet()? else {
            return Ok(false);
        };
        let vp = self.track.video.as_ref().unwrap();
        if dst.width != vp.width as usize
            || dst.height != vp.height as usize
            || dst.format != vp.pixel_format
        {
            *dst = VideoFrame::new(vp.width as usize, vp.height as usize, vp.pixel_format);
        }
        dst.pts = pkt.pts;
        dst.colorimetry = vp.colorimetry;

        let raw = pkt.data();
        let mut src_off = 0usize;
        for plane in &mut dst.planes {
            let row_bytes = plane.row_bytes();
            let h = plane.height();
            for y in 0..h {
                plane
                    .row_mut(y)
                    .copy_from_slice(&raw[src_off..src_off + row_bytes]);
                src_off += row_bytes;
            }
        }
        Ok(true)
    }
}

/// Vectored `YUV4MPEG2` (`.y4m`) Muxer.
pub struct Y4mMuxer<W: Write> {
    writer: VectoredWriter<W>,
    width: usize,
    height: usize,
    format: PixelFormat,
}

impl<W: Write> Y4mMuxer<W> {
    pub fn new(
        writer: W,
        width: usize,
        height: usize,
        format: PixelFormat,
        frame_rate: Rational64,
        colorimetry: Colorimetry,
    ) -> Result<Self> {
        let c_tag = match format {
            PixelFormat::Yuv420p => match colorimetry.chroma_location {
                ChromaLocation::Center => "420jpeg",
                ChromaLocation::TopLeft => "420paldv",
                _ => "420mpeg2",
            },
            PixelFormat::Yuv422p => "422",
            PixelFormat::Yuv444p => "444",
            PixelFormat::Yuv420p10le => "420p10",
            PixelFormat::Gray8 => "mono",
            other => {
                return Err(RagdollError::Unsupported(format!(
                    "Y4mMuxer does not support pixel format {other}"
                )));
            }
        };
        let range_tag = match colorimetry.range {
            ColorRange::Full => "FULL",
            ColorRange::Limited => "LIMITED",
        };

        let hdr = format!(
            "YUV4MPEG2 W{} H{} F{}:{} Ip A1:1 C{} XCOLORRANGE={}\n",
            width,
            height,
            frame_rate.num.max(1),
            frame_rate.den.max(1),
            c_tag,
            range_tag
        );

        let mut vw = VectoredWriter::new(writer);
        vw.write_all_counted(hdr.as_bytes())?;

        Ok(Self {
            writer: vw,
            width,
            height,
            format,
        })
    }

    /// Writes a [`VideoFrame`] into the Y4M stream using vectored I/O.
    pub fn write_frame(&mut self, frame: &VideoFrame) -> Result<()> {
        if frame.width != self.width || frame.height != self.height || frame.format != self.format {
            return Err(RagdollError::DimensionMismatch {
                expected_w: self.width,
                expected_h: self.height,
                actual_w: frame.width,
                actual_h: frame.height,
            });
        }
        self.writer.write_all_counted(b"FRAME\n")?;
        for plane in &frame.planes {
            let row_bytes = plane.row_bytes();
            let stride = plane.stride();
            if row_bytes == stride {
                self.writer
                    .write_all_counted(&plane.buffer().as_slice()[..row_bytes * plane.height()])?;
            } else {
                for y in 0..plane.height() {
                    self.writer.write_all_counted(plane.row(y))?;
                }
            }
        }
        Ok(())
    }

    /// Writes a packed raw frame payload directly after a `FRAME\n` marker via `writev`.
    pub fn write_raw_packet(&mut self, raw_payload: &[u8]) -> Result<()> {
        let expected = self.format.packed_frame_bytes(self.width, self.height);
        if raw_payload.len() != expected {
            return Err(RagdollError::CorruptBitstream(
                "Y4mMuxer::write_raw_packet payload size mismatch",
            ));
        }
        self.writer.write_vectored_all(&[b"FRAME\n", raw_payload])
    }

    pub fn finish(mut self) -> Result<u64> {
        self.writer.flush()?;
        Ok(self.writer.bytes_written())
    }
}
