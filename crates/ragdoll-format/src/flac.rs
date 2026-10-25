//! **`0045`**: Native Standalone `.flac` Container Demuxer & Muxer (`crates/ragdoll-format/src/flac.rs`)
//! supporting RFC 9639 metadata blocks (`STREAMINFO`, `SEEKTABLE` with 18-byte seekpoints,
//! `VORBIS_COMMENT` metadata tags, `PADDING`, `PICTURE`), `crc8_flac` + `crc16_flac` frame boundary
//! indexing, and $O(\log N)$ sample seeking.

use std::io::Write;

use ragdoll_core::{
    AudioTrackParams, ChannelLayout, CodecId, PacketFlags, PacketSlice, RagdollError, Rational64,
    Result, SampleEntry, SampleFormat, Timestamp, TrackInfo, TrackType, crc8_flac, crc16_flac,
};

/// RFC 9639 18-byte seek point inside a FLAC `SEEKTABLE` metadata block (`type = 3`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlacSeekPoint {
    /// Sample number of the first sample in the target frame (`0xFFFFFFFFFFFFFFFF` for placeholder).
    pub sample_number: u64,
    /// Byte offset from the first byte of the first frame header to the target frame header.
    pub stream_offset: u64,
    /// Number of samples in the target frame.
    pub frame_samples: u16,
}

impl FlacSeekPoint {
    pub const PLACEHOLDER_SAMPLE: u64 = 0xFFFF_FFFF_FFFF_FFFF;

    pub fn to_bytes(&self) -> [u8; 18] {
        let mut out = [0u8; 18];
        out[0..8].copy_from_slice(&self.sample_number.to_be_bytes());
        out[8..16].copy_from_slice(&self.stream_offset.to_be_bytes());
        out[16..18].copy_from_slice(&self.frame_samples.to_be_bytes());
        out
    }

    pub fn from_bytes(b: &[u8]) -> Self {
        let sample_number = u64::from_be_bytes(b[0..8].try_into().unwrap());
        let stream_offset = u64::from_be_bytes(b[8..16].try_into().unwrap());
        let frame_samples = u16::from_be_bytes(b[16..18].try_into().unwrap());
        Self {
            sample_number,
            stream_offset,
            frame_samples,
        }
    }
}

/// Vorbis Comment metadata block (`VORBIS_COMMENT`, `type = 4` in FLAC and `OpusTags` in Ogg).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VorbisComments {
    pub vendor: String,
    pub tags: Vec<(String, String)>,
}

impl VorbisComments {
    pub fn new(vendor: impl Into<String>) -> Self {
        Self {
            vendor: vendor.into(),
            tags: Vec::new(),
        }
    }

    pub fn set_tag(&mut self, key: impl Into<String>, value: impl Into<String>) {
        let k = key.into();
        let v = value.into();
        if let Some(existing) = self
            .tags
            .iter_mut()
            .find(|(ek, _)| ek.eq_ignore_ascii_case(&k))
        {
            existing.1 = v;
        } else {
            self.tags.push((k, v));
        }
    }

    pub fn get_tag(&self, key: &str) -> Option<&str> {
        self.tags
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(key))
            .map(|(_, v)| v.as_str())
    }

    /// Serializes the raw Vorbis Comment payload (little-endian vendor length + vendor + count + `KEY=VALUE` entries).
    pub fn to_bytes(&self) -> Vec<u8> {
        let vendor = if self.vendor.is_empty() {
            "ragdoll 0.1.0"
        } else {
            &self.vendor
        };
        let v_bytes = vendor.as_bytes();
        let mut out = Vec::with_capacity(32 + v_bytes.len() + self.tags.len() * 24);
        out.extend_from_slice(&(v_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(v_bytes);
        out.extend_from_slice(&(self.tags.len() as u32).to_le_bytes());
        for (k, v) in &self.tags {
            let entry = format!("{k}={v}");
            let eb = entry.as_bytes();
            out.extend_from_slice(&(eb.len() as u32).to_le_bytes());
            out.extend_from_slice(eb);
        }
        out
    }

    /// Parses a raw Vorbis Comment payload (`VORBIS_COMMENT` metadata block body).
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(RagdollError::BitstreamEndOfData);
        }
        let v_len = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        if 4 + v_len + 4 > bytes.len() {
            return Err(RagdollError::BitstreamEndOfData);
        }
        let vendor = String::from_utf8_lossy(&bytes[4..4 + v_len]).into_owned();
        let mut pos = 4 + v_len;
        let count = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let mut tags = Vec::with_capacity(count.min(256));
        for _ in 0..count {
            if pos + 4 > bytes.len() {
                break;
            }
            let c_len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + c_len > bytes.len() {
                break;
            }
            let s = String::from_utf8_lossy(&bytes[pos..pos + c_len]);
            if let Some((k, v)) = s.split_once('=') {
                tags.push((k.to_string(), v.to_string()));
            }
            pos += c_len;
        }
        Ok(Self { vendor, tags })
    }
}

/// Validates and parses the header of a FLAC audio frame at `buf`, returning `(header_len, block_samples)`.
pub fn parse_flac_frame_header_len_and_samples(
    buf: &[u8],
    streaminfo_min_block: u16,
    streaminfo_max_block: u16,
) -> Option<(usize, u32)> {
    if buf.len() < 6 || buf[0] != 0xFF || (buf[1] & 0xFE) != 0xF8 {
        return None;
    }
    let blocking_strategy_var = (buf[1] & 0x01) != 0;
    let bs_code = buf[2] >> 4;
    let sr_code = buf[2] & 0x0F;
    let ch_code = buf[3] >> 4;
    let bps_code = (buf[3] >> 1) & 0x07;
    if bs_code == 0
        || sr_code == 15
        || ch_code > 10
        || bps_code == 3
        || bps_code == 7
        || (buf[3] & 1) != 0
    {
        return None;
    }

    // Parse UTF-8 coded frame or sample number (1..=7 bytes) at buf[4..]
    let b0 = buf[4];
    let utf8_len = if (b0 & 0x80) == 0 {
        1usize
    } else if (b0 & 0xE0) == 0xC0 {
        2
    } else if (b0 & 0xF0) == 0xE0 {
        3
    } else if (b0 & 0xF8) == 0xF0 {
        4
    } else if (b0 & 0xFC) == 0xF8 {
        5
    } else if (b0 & 0xFE) == 0xFC {
        6
    } else if b0 == 0xFE && blocking_strategy_var {
        7
    } else {
        return None;
    };
    if buf.len() < 4 + utf8_len + 1 {
        return None;
    }
    for i in 1..utf8_len {
        if (buf[4 + i] & 0xC0) != 0x80 {
            return None;
        }
    }
    let mut pos = 4 + utf8_len;

    let block_samples = match bs_code {
        1 => 192u32,
        2..=5 => 576u32 << (bs_code - 2),
        6 => {
            if pos + 1 >= buf.len() {
                return None;
            }
            let v = buf[pos] as u32 + 1;
            pos += 1;
            v
        }
        7 => {
            if pos + 2 >= buf.len() {
                return None;
            }
            let v = u16::from_be_bytes([buf[pos], buf[pos + 1]]) as u32 + 1;
            pos += 2;
            v
        }
        8..=15 => 256u32 << (bs_code - 8),
        _ => {
            let _ = (streaminfo_min_block, streaminfo_max_block);
            return None;
        }
    };

    match sr_code {
        12 => {
            if pos + 1 >= buf.len() {
                return None;
            }
            pos += 1;
        }
        13 | 14 => {
            if pos + 2 >= buf.len() {
                return None;
            }
            pos += 2;
        }
        _ => {}
    }

    if pos >= buf.len() {
        return None;
    }
    let expected_crc8 = buf[pos];
    if crc8_flac(&buf[..pos]) != expected_crc8 {
        return None;
    }
    Some((pos + 1, block_samples))
}

/// Zero-copy Native Standalone `.flac` Container Demuxer with `SEEKTABLE`, `VORBIS_COMMENT`, and `SampleEntry` index.
pub struct FlacContainerDemuxer<'a> {
    data: &'a [u8],
    track: TrackInfo,
    pub seek_table: Vec<FlacSeekPoint>,
    pub comments: VorbisComments,
    pub samples: Vec<SampleEntry>,
    pub first_frame_offset: u64,
    cursor: usize,
}

impl<'a> FlacContainerDemuxer<'a> {
    /// Opens and indexes a standalone `.flac` container from `data`.
    pub fn open(data: &'a [u8]) -> Result<Self> {
        if data.len() < 42 || &data[0..4] != b"fLaC" {
            return Err(RagdollError::InvalidHeader(
                "missing 'fLaC' container magic".into(),
            ));
        }

        let mut pos = 4usize;
        let mut streaminfo_raw = [0u8; 34];
        let mut have_streaminfo = false;
        let mut seek_table = Vec::new();
        let mut comments = VorbisComments::default();

        while pos + 4 <= data.len() {
            let is_last = (data[pos] & 0x80) != 0;
            let block_type = data[pos] & 0x7F;
            let block_len = ((data[pos + 1] as usize) << 16)
                | ((data[pos + 2] as usize) << 8)
                | (data[pos + 3] as usize);
            pos += 4;
            if pos + block_len > data.len() {
                return Err(RagdollError::BitstreamEndOfData);
            }
            let body = &data[pos..pos + block_len];
            match block_type {
                // STREAMINFO
                0 if block_len >= 34 => {
                    streaminfo_raw.copy_from_slice(&body[0..34]);
                    have_streaminfo = true;
                }
                // SEEKTABLE
                3 => {
                    for chunk in body.as_chunks::<18>().0 {
                        let sp = FlacSeekPoint::from_bytes(chunk);
                        if sp.sample_number != FlacSeekPoint::PLACEHOLDER_SAMPLE {
                            seek_table.push(sp);
                        }
                    }
                }
                // VORBIS_COMMENT
                4 => {
                    if let Ok(vc) = VorbisComments::parse(body) {
                        comments = vc;
                    }
                }
                _ => {}
            }
            pos += block_len;
            if is_last {
                break;
            }
        }

        if !have_streaminfo {
            return Err(RagdollError::InvalidHeader(
                "FLAC container missing STREAMINFO metadata block".into(),
            ));
        }

        let min_block = u16::from_be_bytes([streaminfo_raw[0], streaminfo_raw[1]]);
        let max_block = u16::from_be_bytes([streaminfo_raw[2], streaminfo_raw[3]]);
        let sr = ((streaminfo_raw[10] as u32) << 12)
            | ((streaminfo_raw[11] as u32) << 4)
            | ((streaminfo_raw[12] as u32) >> 4);
        let channels = ((streaminfo_raw[12] >> 1) & 0x07) + 1;
        let bps = (((streaminfo_raw[12] & 0x01) << 4) | (streaminfo_raw[13] >> 4)) + 1;
        let total_samples_si = (((streaminfo_raw[13] & 0x0F) as u64) << 32)
            | (u32::from_be_bytes(streaminfo_raw[14..18].try_into().unwrap()) as u64);

        let first_frame_offset = pos as u64;
        let mut samples = Vec::new();
        let mut cur_pts = 0i64;
        let mut frame_start = pos;

        while frame_start + 6 <= data.len() {
            let Some((hdr_len, block_samples)) =
                parse_flac_frame_header_len_and_samples(&data[frame_start..], min_block, max_block)
            else {
                break;
            };
            // Scan forward for end of frame verified by CRC-16 (and next frame's CRC-8 header if not at EOF)
            let mut search = frame_start + hdr_len + 2;
            let mut frame_end = data.len();
            while search < data.len() {
                if data[search] == 0xFF
                    && search + 6 <= data.len()
                    && (data[search + 1] & 0xFE) == 0xF8
                    && parse_flac_frame_header_len_and_samples(
                        &data[search..],
                        min_block,
                        max_block,
                    )
                    .is_some()
                {
                    let expected_crc16 = u16::from_be_bytes([data[search - 2], data[search - 1]]);
                    if crc16_flac(&data[frame_start..search - 2]) == expected_crc16 {
                        frame_end = search;
                        break;
                    }
                }
                search += 1;
            }
            let size = (frame_end - frame_start) as u32;
            samples.push(SampleEntry::new(frame_start as u64, cur_pts, size, 0, true));
            cur_pts += block_samples as i64;
            frame_start = frame_end;
        }

        let duration_ticks = if total_samples_si > 0 {
            Some(total_samples_si as i64)
        } else {
            Some(cur_pts)
        };
        let sample_format = match bps {
            1..=8 => SampleFormat::U8,
            9..=16 => SampleFormat::S16le,
            17..=24 => SampleFormat::S24le,
            _ => SampleFormat::S32le,
        };
        let total_bytes: u64 = samples.iter().map(|s| s.size as u64).sum();
        let track = TrackInfo {
            index: 0,
            track_type: TrackType::Audio,
            codec_id: CodecId::Flac,
            timebase: Rational64::new(1, sr.max(1) as i64),
            duration_ticks,
            extra_data: streaminfo_raw.to_vec(),
            video: None,
            audio: Some(AudioTrackParams {
                sample_rate: sr,
                channels,
                channel_layout: ChannelLayout::from_channels(channels),
                sample_format,
                bits_per_coded_sample: bps,
            }),
            sample_count: samples.len(),
            total_bytes,
        };

        Ok(Self {
            data,
            track,
            seek_table,
            comments,
            samples,
            first_frame_offset,
            cursor: 0,
        })
    }

    #[inline]
    pub fn track(&self) -> &TrackInfo {
        &self.track
    }

    #[inline]
    pub fn comments(&self) -> &VorbisComments {
        &self.comments
    }

    #[inline]
    pub fn seek_table(&self) -> &[FlacSeekPoint] {
        &self.seek_table
    }

    /// Performs $O(\log N)$ binary search over the `SampleEntry` table to seek to the frame containing `target_sample`.
    pub fn seek_to_sample(&mut self, target_sample: i64) -> usize {
        let idx = self
            .samples
            .partition_point(|s| s.pts <= target_sample)
            .saturating_sub(1);
        self.cursor = idx;
        idx
    }

    /// Reads the next zero-copy FLAC frame packet (`PacketSlice<'a>`).
    pub fn next_packet(&mut self) -> Result<Option<PacketSlice<'a>>> {
        if self.cursor >= self.samples.len() {
            return Ok(None);
        }
        let entry = self.samples[self.cursor];
        let next_pts = if self.cursor + 1 < self.samples.len() {
            self.samples[self.cursor + 1].pts
        } else {
            self.track.duration_ticks.unwrap_or(entry.pts + 4096)
        };
        self.cursor += 1;
        let start = entry.offset as usize;
        let end = start + (entry.size as usize);
        let ts = Timestamp::new(entry.pts, self.track.timebase);
        Ok(Some(PacketSlice::from_mmap(
            &self.data[start..end],
            0,
            ts,
            ts,
            (next_pts - entry.pts).max(1),
            PacketFlags::KEYFRAME,
            entry.offset,
        )))
    }
}

/// Native Standalone `.flac` Container Muxer writing `STREAMINFO`, `SEEKTABLE` (18-byte seekpoints),
/// `VORBIS_COMMENT` metadata tags, and FLAC audio frames.
pub struct FlacContainerMuxer {
    track: Option<TrackInfo>,
    comments: VorbisComments,
    frames: Vec<(Vec<u8>, u64, u16)>,
    total_samples: u64,
    min_frame_size: u32,
    max_frame_size: u32,
    min_block_size: u16,
    max_block_size: u16,
}

impl FlacContainerMuxer {
    pub fn new() -> Self {
        Self {
            track: None,
            comments: VorbisComments::new("ragdoll 0.1.0"),
            frames: Vec::new(),
            total_samples: 0,
            min_frame_size: u32::MAX,
            max_frame_size: 0,
            min_block_size: u16::MAX,
            max_block_size: 0,
        }
    }

    pub fn set_comments(&mut self, comments: VorbisComments) {
        self.comments = comments;
    }

    pub fn comments_mut(&mut self) -> &mut VorbisComments {
        &mut self.comments
    }

    pub fn add_track(&mut self, track: TrackInfo) -> u32 {
        self.track = Some(track);
        0
    }

    pub fn write_packet(&mut self, pkt: &PacketSlice<'_>) -> Result<()> {
        let data = pkt.data();
        if data.is_empty() {
            return Ok(());
        }
        // Strip leading fLaC + STREAMINFO if caller passed a stream-header-prefixed packet
        let frame_slice = if data.len() > 42 && &data[0..4] == b"fLaC" {
            let mut p = 4usize;
            while p + 4 <= data.len() {
                let last = (data[p] & 0x80) != 0;
                let blen = ((data[p + 1] as usize) << 16)
                    | ((data[p + 2] as usize) << 8)
                    | (data[p + 3] as usize);
                p += 4 + blen;
                if last {
                    break;
                }
            }
            &data[p.min(data.len())..]
        } else {
            data
        };
        if frame_slice.is_empty() {
            return Ok(());
        }

        let block_samples = if let Some((_, bs)) =
            parse_flac_frame_header_len_and_samples(frame_slice, 16, 65535)
        {
            bs as u16
        } else {
            (pkt.duration as u16).max(4096)
        };

        self.write_framed_packet(frame_slice.to_vec(), block_samples);
        Ok(())
    }

    pub fn write_framed_packet(&mut self, frame_bytes: Vec<u8>, block_samples: u16) {
        if frame_bytes.is_empty() {
            return;
        }
        let sample_num = self.total_samples;
        self.total_samples += block_samples as u64;
        let flen = frame_bytes.len() as u32;
        self.min_frame_size = self.min_frame_size.min(flen);
        self.max_frame_size = self.max_frame_size.max(flen);
        self.min_block_size = self.min_block_size.min(block_samples);
        self.max_block_size = self.max_block_size.max(block_samples);
        self.frames.push((frame_bytes, sample_num, block_samples));
    }

    /// Finalizes and writes the complete `.flac` container (`fLaC` + `STREAMINFO` + `SEEKTABLE` + `VORBIS_COMMENT` + frames).
    pub fn finish<W: Write>(self, mut writer: W) -> Result<u64> {
        let track = self.track.ok_or_else(|| {
            RagdollError::InvalidData("FlacContainerMuxer has no audio track".into())
        })?;
        let audio = track.audio.unwrap_or_default();

        // 1. Build 34-byte STREAMINFO
        let mut si = [0u8; 34];
        if track.extra_data.len() >= 34 {
            si.copy_from_slice(&track.extra_data[0..34]);
        } else if track.extra_data.len() >= 42 && &track.extra_data[0..4] == b"fLaC" {
            si.copy_from_slice(&track.extra_data[8..42]);
        } else {
            let min_b = if self.min_block_size == u16::MAX {
                4096
            } else {
                self.min_block_size
            };
            let max_b = if self.max_block_size == 0 {
                4096
            } else {
                self.max_block_size
            };
            si[0..2].copy_from_slice(&min_b.to_be_bytes());
            si[2..4].copy_from_slice(&max_b.to_be_bytes());
            let sr = audio.sample_rate.max(8000);
            let ch = audio.channels.clamp(1, 8) - 1;
            let bps = audio.bits_per_coded_sample.clamp(8, 32) - 1;
            si[10] = (sr >> 12) as u8;
            si[11] = (sr >> 4) as u8;
            si[12] = (((sr & 0x0F) as u8) << 4) | ((ch & 0x07) << 1) | ((bps >> 4) & 0x01);
            si[13] = ((bps & 0x0F) << 4) | (((self.total_samples >> 32) as u8) & 0x0F);
            si[14..18].copy_from_slice(&(self.total_samples as u32).to_be_bytes());
        }
        // Update min/max frame size and total_samples in STREAMINFO
        if self.min_frame_size != u32::MAX {
            si[4] = (self.min_frame_size >> 16) as u8;
            si[5] = (self.min_frame_size >> 8) as u8;
            si[6] = self.min_frame_size as u8;
        }
        if self.max_frame_size != 0 {
            si[7] = (self.max_frame_size >> 16) as u8;
            si[8] = (self.max_frame_size >> 8) as u8;
            si[9] = self.max_frame_size as u8;
        }
        if self.total_samples > 0 {
            si[13] = (si[13] & 0xF0) | (((self.total_samples >> 32) as u8) & 0x0F);
            si[14..18].copy_from_slice(&(self.total_samples as u32).to_be_bytes());
        }

        // 2. Build SEEKTABLE (type = 3) with 18-byte seekpoints
        let mut seek_bytes = Vec::with_capacity(self.frames.len() * 18);
        let mut running_offset = 0u64;
        for (f_bytes, sample_num, frame_samples) in &self.frames {
            let sp = FlacSeekPoint {
                sample_number: *sample_num,
                stream_offset: running_offset,
                frame_samples: *frame_samples,
            };
            seek_bytes.extend_from_slice(&sp.to_bytes());
            running_offset += f_bytes.len() as u64;
        }

        // 3. Build VORBIS_COMMENT (type = 4, is_last = 1)
        let vc_bytes = self.comments.to_bytes();

        let mut written = 0u64;
        writer.write_all(b"fLaC")?;
        written += 4;

        // Block 0: STREAMINFO (is_last = 0, type = 0, len = 34)
        writer.write_all(&[0x00, 0x00, 0x00, 34])?;
        writer.write_all(&si)?;
        written += 38;

        // Block 1: SEEKTABLE (is_last = 0, type = 3)
        let st_len = seek_bytes.len();
        writer.write_all(&[
            0x03,
            ((st_len >> 16) & 0xFF) as u8,
            ((st_len >> 8) & 0xFF) as u8,
            (st_len & 0xFF) as u8,
        ])?;
        writer.write_all(&seek_bytes)?;
        written += 4 + (st_len as u64);

        // Block 2: VORBIS_COMMENT (is_last = 1 -> 0x84)
        let vc_len = vc_bytes.len();
        writer.write_all(&[
            0x84,
            ((vc_len >> 16) & 0xFF) as u8,
            ((vc_len >> 8) & 0xFF) as u8,
            (vc_len & 0xFF) as u8,
        ])?;
        writer.write_all(&vc_bytes)?;
        written += 4 + (vc_len as u64);

        // Write audio frames
        for (f_bytes, _, _) in &self.frames {
            writer.write_all(f_bytes)?;
            written += f_bytes.len() as u64;
        }
        writer.flush()?;
        Ok(written)
    }
}

impl Default for FlacContainerMuxer {
    fn default() -> Self {
        Self::new()
    }
}
