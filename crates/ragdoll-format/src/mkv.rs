use crate::io::VectoredWriter;
use ragdoll_core::{
    AudioTrackParams, ChannelLayout, CodecId, ColorPrimaries, ColorRange, Colorimetry,
    MatrixCoefficients, PacketFlags, PacketSlice, PixelFormat, RagdollError, Rational64, Result,
    SampleEntry, SampleFormat, Timestamp, TrackInfo, TrackType, TransferCharacteristics,
    VideoTrackParams, read_ebml_vint, rescale_q, write_ebml_vint,
};
use std::io::Write;

// EBML & Matroska Element IDs (RFC 8794 / IETF Matroska)
const ID_EBML: u32 = 0x1A45_DFA3;
const ID_DOC_TYPE: u32 = 0x4282;
const ID_DOC_TYPE_VERSION: u32 = 0x4287;
const ID_DOC_TYPE_READ_VERSION: u32 = 0x4285;

const ID_SEGMENT: u32 = 0x1853_8067;
const ID_SEEK_HEAD: u32 = 0x114D_9B74;
const ID_SEEK: u32 = 0x4DBB;
const ID_SEEK_ID: u32 = 0x53AB;
const ID_SEEK_POSITION: u32 = 0x53AC;

const ID_INFO: u32 = 0x1549_A966;
const ID_TIMESTAMP_SCALE: u32 = 0x002A_D7B1;
const ID_DURATION: u32 = 0x4489;
const ID_MUXING_APP: u32 = 0x4D80;
const ID_WRITING_APP: u32 = 0x5741;

const ID_TRACKS: u32 = 0x1654_AE6B;
const ID_TRACK_ENTRY: u32 = 0xAE;
const ID_TRACK_NUMBER: u32 = 0xD7;
const ID_TRACK_UID: u32 = 0x73C5;
const ID_TRACK_TYPE: u32 = 0x83;
const ID_DEFAULT_DURATION: u32 = 0x0023_E383;
const ID_CODEC_ID: u32 = 0x86;
const ID_CODEC_PRIVATE: u32 = 0x63A2;
const ID_CODEC_DELAY: u32 = 0x56AA;
const ID_SEEK_PRE_ROLL: u32 = 0x56BB;

const ID_VIDEO: u32 = 0xE0;
const ID_PIXEL_WIDTH: u32 = 0xB0;
const ID_PIXEL_HEIGHT: u32 = 0xBA;
const ID_COLOUR: u32 = 0x55B0;
const ID_MATRIX_COEFFICIENTS: u32 = 0x55B1;
const ID_BITS_PER_CHANNEL: u32 = 0x55B2;
const ID_RANGE: u32 = 0x55B9;
const ID_TRANSFER_CHARACTERISTICS: u32 = 0x55BA;
const ID_PRIMARIES: u32 = 0x55BB;

const ID_AUDIO: u32 = 0xE1;
const ID_SAMPLING_FREQUENCY: u32 = 0xB5;
const ID_CHANNELS: u32 = 0x9F;
const ID_BIT_DEPTH: u32 = 0x6264;

const ID_CLUSTER: u32 = 0x1F43_B675;
const ID_TIMESTAMP: u32 = 0xE7;
const ID_SIMPLE_BLOCK: u32 = 0xA3;
const ID_BLOCK_GROUP: u32 = 0xA0;
const ID_BLOCK: u32 = 0xA1;
const ID_BLOCK_DURATION: u32 = 0x9B;
const ID_REFERENCE_BLOCK: u32 = 0xFB;

const ID_CUES: u32 = 0x1C53_BB6B;
const ID_CUE_POINT: u32 = 0xBB;
const ID_CUE_TIME: u32 = 0xB3;
const ID_CUE_TRACK_POSITIONS: u32 = 0xB7;
const ID_CUE_TRACK: u32 = 0xF7;
const ID_CUE_CLUSTER_POSITION: u32 = 0xF1;

/// Reads an EBML element header `(id, payload_slice, total_bytes_consumed)`.
/// Properly handles unknown-size (`0xFF` / all-ones VINT) streaming `Segment` and `Cluster` elements.
fn next_ebml_element(data: &[u8]) -> Result<Option<(u32, &[u8], usize)>> {
    if data.is_empty() {
        return Ok(None);
    }
    let (raw_id, _, id_len) = read_ebml_vint(data)?;
    if id_len > 4 || id_len >= data.len() {
        return Err(RagdollError::CorruptBitstream(
            "invalid or truncated EBML Element ID",
        ));
    }
    let (_raw_sz, stripped_sz, sz_len) = read_ebml_vint(&data[id_len..])?;
    let header_len = id_len + sz_len;

    // All-ones payload bits in an EBML size VINT denote an unknown/streaming length
    let unknown_marker = (1u64 << (7 * sz_len as u32)) - 1;
    let payload_len = if stripped_sz == unknown_marker {
        data.len() - header_len
    } else {
        stripped_sz as usize
    };

    if header_len.saturating_add(payload_len) > data.len() {
        return Err(RagdollError::CorruptBitstream(
            "EBML element payload exceeds parent slice bounds",
        ));
    }

    let payload = &data[header_len..header_len + payload_len];
    Ok(Some((raw_id as u32, payload, header_len + payload_len)))
}

#[inline]
fn read_ebml_uint(bytes: &[u8]) -> u64 {
    let mut v = 0u64;
    for &b in bytes.iter().take(8) {
        v = (v << 8) | (b as u64);
    }
    v
}

#[inline]
fn read_ebml_int(bytes: &[u8]) -> i64 {
    if bytes.is_empty() {
        return 0;
    }
    let mut v = if (bytes[0] & 0x80) != 0 { -1i64 } else { 0i64 };
    for &b in bytes.iter().take(8) {
        v = (v << 8) | (b as i64);
    }
    v
}

#[inline]
fn read_ebml_float(bytes: &[u8]) -> f64 {
    match bytes.len() {
        4 => f32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64,
        8 => f64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
        _ => 0.0,
    }
}

/// Zero-copy Matroska (`.mkv`) and WebM (`.webm`) demuxer operating over a memory-mapped slice `&'a [u8]`.
pub struct MkvDemuxer<'a> {
    data: &'a [u8],
    doc_type: String,
    timestamp_scale_ns: u64,
    tracks: Vec<TrackInfo>,
    default_durations_ticks: Vec<i64>,
    packets: Vec<(u32, SampleEntry, i64)>,
    cursor: usize,
    has_seek_head: bool,
    cues_count: usize,
}

impl<'a> MkvDemuxer<'a> {
    /// Parses a Matroska / WebM memory-mapped slice and indexes all `SimpleBlock` / `Block` payloads zero-copy.
    pub fn open(data: &'a [u8]) -> Result<Self> {
        let mut pos = 0usize;
        let mut doc_type = String::from("matroska");
        let mut timestamp_scale_ns = 1_000_000u64; // Default 1 ms
        let mut duration_float: Option<f64> = None;
        let mut tracks = Vec::new();
        let mut track_numbers = Vec::new();
        let mut default_durations_ns = Vec::new();
        let mut packets: Vec<(u32, SampleEntry, i64)> = Vec::new();
        let mut has_seek_head = false;
        let mut cues_count = 0usize;

        while pos < data.len() {
            let Some((id, payload, consumed)) = next_ebml_element(&data[pos..])? else {
                break;
            };
            let payload_base = pos + (consumed - payload.len());
            pos += consumed;

            match id {
                ID_EBML => {
                    let mut p = 0usize;
                    while p < payload.len() {
                        let Some((cid, cpay, ccons)) = next_ebml_element(&payload[p..])? else {
                            break;
                        };
                        p += ccons;
                        if cid == ID_DOC_TYPE {
                            let s = std::str::from_utf8(cpay)
                                .unwrap_or("matroska")
                                .trim_end_matches('\0');
                            doc_type = s.to_string();
                        }
                    }
                }
                ID_SEGMENT => {
                    Self::parse_segment(
                        data,
                        payload_base,
                        payload.len(),
                        &mut timestamp_scale_ns,
                        &mut duration_float,
                        &mut tracks,
                        &mut track_numbers,
                        &mut default_durations_ns,
                        &mut packets,
                        &mut has_seek_head,
                        &mut cues_count,
                    )?;
                }
                _ => {}
            }
        }

        if tracks.is_empty() {
            return Err(RagdollError::InvalidHeader(
                "no valid media tracks found in Matroska/WebM Segment".to_string(),
            ));
        }

        let seg_timebase = Rational64::new(timestamp_scale_ns as i64, 1_000_000_000);
        let seg_dur_ticks = duration_float.map(|d| d.round() as i64);
        let mut default_durations_ticks = Vec::with_capacity(tracks.len());

        for (i, track) in tracks.iter_mut().enumerate() {
            track.timebase = seg_timebase;
            track.duration_ticks = seg_dur_ticks;
            let def_ns = default_durations_ns[i];
            let def_ticks = if def_ns > 0 {
                (def_ns / timestamp_scale_ns.max(1)).max(1) as i64
            } else {
                1
            };
            default_durations_ticks.push(def_ticks);
            if let Some(vp) = &mut track.video
                && def_ns > 0
            {
                vp.frame_rate = Rational64::new(1_000_000_000, def_ns as i64);
            }
        }

        // Update per-track sample counts and byte totals
        for &(t_idx, entry, _) in &packets {
            if let Some(t) = tracks.get_mut(t_idx as usize) {
                t.sample_count += 1;
                t.total_bytes += entry.size as u64;
            }
        }

        Ok(Self {
            data,
            doc_type,
            timestamp_scale_ns,
            tracks,
            default_durations_ticks,
            packets,
            cursor: 0,
            has_seek_head,
            cues_count,
        })
    }

    #[inline]
    pub fn doc_type(&self) -> &str {
        &self.doc_type
    }

    #[inline]
    pub const fn timestamp_scale_ns(&self) -> u64 {
        self.timestamp_scale_ns
    }

    /// Returns `true` if the `Segment` contains a `SeekHead` (`0x114D9B74`) meta-index.
    #[inline]
    pub const fn has_seek_head(&self) -> bool {
        self.has_seek_head
    }

    /// Number of `CuePoint` entries parsed from the `Cues` (`0x1C53BB6B`) element.
    #[inline]
    pub const fn cues_count(&self) -> usize {
        self.cues_count
    }

    #[inline]
    pub fn tracks(&self) -> &[TrackInfo] {
        &self.tracks
    }

    #[inline]
    pub fn total_packets(&self) -> usize {
        self.packets.len()
    }

    /// Seeks to the nearest keyframe at or before `target` on `track_pos` in $O(\log N)$ time,
    /// repositioning the demuxer's sequential cursor and returning `(packet_cursor_idx, SampleEntry)`.
    pub fn seek_to_timestamp(
        &mut self,
        track_pos: usize,
        target: Timestamp,
    ) -> Result<(usize, SampleEntry)> {
        let track = self
            .tracks
            .get(track_pos)
            .ok_or(RagdollError::CorruptBitstream("invalid track index"))?;
        let target_ticks = target.rescale_to(track.timebase).ticks;
        let t_u32 = track_pos as u32;

        // Binary search to the last packet at or before `target_ticks`
        let upper = self
            .packets
            .partition_point(|&(_, entry, _)| entry.pts <= target_ticks);
        let search_end = if upper == 0 {
            self.packets.len()
        } else {
            upper
        };

        // Walk backward from `search_end` to find the nearest keyframe on `track_pos`
        for idx in (0..search_end).rev() {
            let (pkt_track, entry, _) = self.packets[idx];
            if pkt_track == t_u32 && entry.is_keyframe() {
                self.cursor = idx;
                return Ok((idx, entry));
            }
        }

        // Fallback: find the first keyframe on `track_pos` in the entire stream
        for (idx, &(pkt_track, entry, _)) in self.packets.iter().enumerate() {
            if pkt_track == t_u32 && entry.is_keyframe() {
                self.cursor = idx;
                return Ok((idx, entry));
            }
        }

        Err(RagdollError::CorruptBitstream(
            "no keyframe found on target Matroska/WebM track",
        ))
    }

    /// Emits the next packet as a zero-copy [`PacketSlice<'a>`] borrowing directly from the memory-mapped file.
    #[inline]
    pub fn next_packet(&mut self) -> Result<Option<PacketSlice<'a>>> {
        if self.cursor >= self.packets.len() {
            return Ok(None);
        }
        let (track_index, entry, explicit_dur) = self.packets[self.cursor];
        self.cursor += 1;

        let start = entry.offset as usize;
        let end = start + entry.size as usize;
        if end > self.data.len() {
            return Err(RagdollError::CorruptBitstream(
                "Matroska block payload out of bounds",
            ));
        }

        let tb = self.tracks[track_index as usize].timebase;
        let ts = Timestamp::new(entry.pts, tb);
        let duration = if explicit_dur > 0 {
            explicit_dur
        } else {
            self.default_durations_ticks[track_index as usize]
        };
        let flags = if entry.is_keyframe() {
            PacketFlags::KEYFRAME
        } else {
            PacketFlags::NONE
        };

        Ok(Some(PacketSlice::from_mmap(
            &self.data[start..end],
            track_index,
            ts,
            ts,
            duration,
            flags,
            entry.offset,
        )))
    }

    #[allow(clippy::too_many_arguments)]
    fn parse_segment(
        file_data: &'a [u8],
        seg_start: usize,
        seg_len: usize,
        timestamp_scale_ns: &mut u64,
        duration_float: &mut Option<f64>,
        tracks: &mut Vec<TrackInfo>,
        track_numbers: &mut Vec<u64>,
        default_durations_ns: &mut Vec<u64>,
        packets: &mut Vec<(u32, SampleEntry, i64)>,
        has_seek_head: &mut bool,
        cues_count: &mut usize,
    ) -> Result<()> {
        let seg_end = seg_start + seg_len;
        let mut pos = seg_start;

        while pos < seg_end {
            let Some((id, payload, consumed)) = next_ebml_element(&file_data[pos..seg_end])? else {
                break;
            };
            let payload_base = pos + (consumed - payload.len());
            pos += consumed;

            match id {
                ID_SEEK_HEAD => {
                    *has_seek_head = true;
                }
                ID_INFO => {
                    let mut p = 0usize;
                    while p < payload.len() {
                        let Some((cid, cpay, ccons)) = next_ebml_element(&payload[p..])? else {
                            break;
                        };
                        p += ccons;
                        match cid {
                            ID_TIMESTAMP_SCALE => {
                                let ts = read_ebml_uint(cpay);
                                if ts > 0 {
                                    *timestamp_scale_ns = ts;
                                }
                            }
                            ID_DURATION => {
                                *duration_float = Some(read_ebml_float(cpay));
                            }
                            _ => {}
                        }
                    }
                }
                ID_TRACKS => {
                    let mut p = 0usize;
                    while p < payload.len() {
                        let Some((cid, cpay, ccons)) = next_ebml_element(&payload[p..])? else {
                            break;
                        };
                        p += ccons;
                        if cid == ID_TRACK_ENTRY {
                            let idx = tracks.len() as u32;
                            if let Some((info, tnum, def_dur)) = Self::parse_track_entry(cpay, idx)?
                            {
                                tracks.push(info);
                                track_numbers.push(tnum);
                                default_durations_ns.push(def_dur);
                            }
                        }
                    }
                }
                ID_CLUSTER => {
                    Self::parse_cluster(
                        file_data,
                        payload_base,
                        payload.len(),
                        track_numbers,
                        packets,
                    )?;
                }
                ID_CUES => {
                    let mut p = 0usize;
                    while p < payload.len() {
                        let Some((cid, _cpay, ccons)) = next_ebml_element(&payload[p..])? else {
                            break;
                        };
                        p += ccons;
                        if cid == ID_CUE_POINT {
                            *cues_count += 1;
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn parse_track_entry(
        entry_data: &[u8],
        track_index: u32,
    ) -> Result<Option<(TrackInfo, u64, u64)>> {
        let mut track_num = (track_index + 1) as u64;
        let mut track_type = TrackType::Data;
        let mut default_dur_ns = 0u64;
        let mut codec_id_str = String::new();
        let mut extra_data = Vec::new();
        let mut video = VideoTrackParams::default();
        let mut audio = AudioTrackParams::default();

        let mut p = 0usize;
        while p < entry_data.len() {
            let Some((id, pay, cons)) = next_ebml_element(&entry_data[p..])? else {
                break;
            };
            p += cons;
            match id {
                ID_TRACK_NUMBER => track_num = read_ebml_uint(pay),
                ID_TRACK_TYPE => {
                    track_type = match read_ebml_uint(pay) {
                        1 => TrackType::Video,
                        2 => TrackType::Audio,
                        17 => TrackType::Subtitle,
                        _ => TrackType::Data,
                    };
                }
                ID_DEFAULT_DURATION => default_dur_ns = read_ebml_uint(pay),
                ID_CODEC_ID => {
                    codec_id_str = std::str::from_utf8(pay)
                        .unwrap_or("")
                        .trim_end_matches('\0')
                        .to_string();
                }
                ID_CODEC_PRIVATE => {
                    extra_data.clear();
                    extra_data.extend_from_slice(pay);
                }
                ID_VIDEO => {
                    let mut vp = 0usize;
                    while vp < pay.len() {
                        let Some((vid, vpay, vcons)) = next_ebml_element(&pay[vp..])? else {
                            break;
                        };
                        vp += vcons;
                        match vid {
                            ID_PIXEL_WIDTH => video.width = read_ebml_uint(vpay) as u32,
                            ID_PIXEL_HEIGHT => video.height = read_ebml_uint(vpay) as u32,
                            ID_COLOUR => {
                                let mut cp = 0usize;
                                let mut col = Colorimetry::default();
                                while cp < vpay.len() {
                                    let Some((col_id, col_pay, col_cons)) =
                                        next_ebml_element(&vpay[cp..])?
                                    else {
                                        break;
                                    };
                                    cp += col_cons;
                                    match col_id {
                                        ID_MATRIX_COEFFICIENTS => {
                                            col.matrix = MatrixCoefficients::from_cicp(
                                                read_ebml_uint(col_pay) as u8,
                                            );
                                        }
                                        ID_TRANSFER_CHARACTERISTICS => {
                                            col.transfer = TransferCharacteristics::from_cicp(
                                                read_ebml_uint(col_pay) as u8,
                                            );
                                        }
                                        ID_PRIMARIES => {
                                            col.primaries = ColorPrimaries::from_cicp(
                                                read_ebml_uint(col_pay) as u8,
                                            );
                                        }
                                        ID_RANGE => {
                                            col.range = match read_ebml_uint(col_pay) {
                                                2 => ColorRange::Full,
                                                _ => ColorRange::Limited,
                                            };
                                        }
                                        ID_BITS_PER_CHANNEL => {
                                            let bpc = read_ebml_uint(col_pay) as u8;
                                            if bpc == 10 {
                                                video.bit_depth = 10;
                                                video.pixel_format = PixelFormat::Yuv420p10le;
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                                video.colorimetry = col;
                            }
                            _ => {}
                        }
                    }
                }
                ID_AUDIO => {
                    let mut ap = 0usize;
                    while ap < pay.len() {
                        let Some((aid, apay, acons)) = next_ebml_element(&pay[ap..])? else {
                            break;
                        };
                        ap += acons;
                        match aid {
                            ID_SAMPLING_FREQUENCY => {
                                let sr = read_ebml_float(apay).round() as u32;
                                if sr > 0 {
                                    audio.sample_rate = sr;
                                }
                            }
                            ID_CHANNELS => {
                                let ch = read_ebml_uint(apay) as u8;
                                if ch > 0 {
                                    audio.channels = ch;
                                    audio.channel_layout = ChannelLayout::from_channels(ch);
                                }
                            }
                            ID_BIT_DEPTH => {
                                let bd = read_ebml_uint(apay) as u8;
                                if bd > 0 {
                                    audio.bits_per_coded_sample = bd;
                                    audio.sample_format = match bd {
                                        8 => SampleFormat::U8,
                                        24 => SampleFormat::S24le,
                                        32 => SampleFormat::S32le,
                                        _ => SampleFormat::S16le,
                                    };
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }

        let codec_id = match codec_id_str.as_str() {
            "V_MPEG4/ISO/AVC" => CodecId::H264,
            "V_MPEGH/ISO/HEVC" => CodecId::Hevc,
            "V_AV1" => CodecId::Av1,
            "V_VP9" => CodecId::Vp9,
            "V_RAGDOLL/INTRA" => CodecId::RagdollIntra,
            "V_UNCOMPRESSED" => CodecId::RawVideo,
            "A_FLAC" => {
                // Strip "fLaC" + 4-byte block header if present so extra_data is the 34-byte STREAMINFO
                if extra_data.len() >= 42 && &extra_data[0..4] == b"fLaC" {
                    let streaminfo = extra_data[8..42].to_vec();
                    extra_data = streaminfo;
                }
                CodecId::Flac
            }
            s if s.starts_with("A_AAC") => CodecId::Aac,
            "A_OPUS" => CodecId::Opus,
            "A_PCM/FLOAT/IEEE" => {
                audio.sample_format = SampleFormat::F32le;
                CodecId::PcmF32Le
            }
            "A_PCM/INT/LIT" => match audio.bits_per_coded_sample {
                8 => CodecId::PcmU8,
                24 => CodecId::PcmS24Le,
                32 => CodecId::PcmS32Le,
                _ => CodecId::PcmS16Le,
            },
            _ => return Ok(None),
        };

        let info = TrackInfo {
            index: track_index,
            track_type,
            codec_id,
            timebase: Rational64::new(1, 1000),
            duration_ticks: None,
            extra_data,
            video: (track_type == TrackType::Video).then_some(video),
            audio: (track_type == TrackType::Audio).then_some(audio),
            sample_count: 0,
            total_bytes: 0,
        };

        Ok(Some((info, track_num, default_dur_ns)))
    }

    fn parse_cluster(
        file_data: &'a [u8],
        cluster_start: usize,
        cluster_len: usize,
        track_numbers: &[u64],
        packets: &mut Vec<(u32, SampleEntry, i64)>,
    ) -> Result<()> {
        let cluster_end = cluster_start + cluster_len;
        let mut pos = cluster_start;
        let mut cluster_ts = 0i64;

        while pos < cluster_end {
            let Some((id, payload, consumed)) = next_ebml_element(&file_data[pos..cluster_end])?
            else {
                break;
            };
            let payload_base = pos + (consumed - payload.len());
            pos += consumed;

            match id {
                ID_TIMESTAMP => {
                    cluster_ts = read_ebml_uint(payload) as i64;
                }
                ID_SIMPLE_BLOCK => {
                    Self::parse_block(
                        payload,
                        payload_base,
                        cluster_ts,
                        true,
                        false,
                        0,
                        track_numbers,
                        packets,
                    )?;
                }
                ID_BLOCK_GROUP => {
                    let mut bg_pos = 0usize;
                    let mut block_slice: Option<(&[u8], usize)> = None;
                    let mut block_dur = 0i64;
                    let mut has_reference = false;

                    while bg_pos < payload.len() {
                        let Some((bid, bpay, bcons)) = next_ebml_element(&payload[bg_pos..])?
                        else {
                            break;
                        };
                        let bpay_base = payload_base + bg_pos + (bcons - bpay.len());
                        bg_pos += bcons;
                        match bid {
                            ID_BLOCK => block_slice = Some((bpay, bpay_base)),
                            ID_BLOCK_DURATION => block_dur = read_ebml_uint(bpay) as i64,
                            ID_REFERENCE_BLOCK => {
                                let _ = read_ebml_int(bpay);
                                has_reference = true;
                            }
                            _ => {}
                        }
                    }

                    if let Some((bs, bbase)) = block_slice {
                        Self::parse_block(
                            bs,
                            bbase,
                            cluster_ts,
                            false,
                            !has_reference,
                            block_dur,
                            track_numbers,
                            packets,
                        )?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn parse_block(
        block: &[u8],
        block_base_offset: usize,
        cluster_ts: i64,
        is_simple_block: bool,
        block_group_keyframe: bool,
        explicit_dur: i64,
        track_numbers: &[u64],
        packets: &mut Vec<(u32, SampleEntry, i64)>,
    ) -> Result<()> {
        let (_raw, track_num, vlen) = read_ebml_vint(block)?;
        if vlen + 3 > block.len() {
            return Err(RagdollError::CorruptBitstream(
                "truncated Matroska SimpleBlock/Block header",
            ));
        }
        let Some(track_pos) = track_numbers.iter().position(|&n| n == track_num) else {
            return Ok(());
        };
        let rel_ts = i16::from_be_bytes([block[vlen], block[vlen + 1]]) as i64;
        let flags = block[vlen + 2];
        let is_keyframe = if is_simple_block {
            (flags & 0x80) != 0
        } else {
            block_group_keyframe
        };
        let lacing = (flags & 0x06) >> 1;
        let hdr_len = vlen + 3;
        let pts = cluster_ts + rel_ts;

        if lacing == 0 {
            let file_offset = (block_base_offset + hdr_len) as u64;
            let size = (block.len() - hdr_len) as u32;
            packets.push((
                track_pos as u32,
                SampleEntry::new(file_offset, pts, size, 0, is_keyframe),
                explicit_dur,
            ));
        } else {
            // Laced block (1 = Xiph, 2 = EBML, 3 = Fixed-size)
            if hdr_len >= block.len() {
                return Ok(());
            }
            let num_frames = (block[hdr_len] as usize) + 1;
            let mut cur = hdr_len + 1;
            let mut frame_sizes = Vec::with_capacity(num_frames);

            match lacing {
                3 => {
                    // Fixed-size lacing
                    let rem = block.len().saturating_sub(cur);
                    let fsz = rem / num_frames.max(1);
                    frame_sizes.resize(num_frames, fsz);
                }
                1 => {
                    // Xiph 255-continuing lacing
                    let mut sum = 0usize;
                    for _ in 0..num_frames.saturating_sub(1) {
                        let mut sz = 0usize;
                        while cur < block.len() {
                            let b = block[cur] as usize;
                            cur += 1;
                            sz += b;
                            if b < 255 {
                                break;
                            }
                        }
                        sum += sz;
                        frame_sizes.push(sz);
                    }
                    frame_sizes.push(block.len().saturating_sub(cur + sum));
                }
                2 => {
                    // EBML delta lacing
                    let mut sum = 0usize;
                    let mut prev_sz = 0i64;
                    for i in 0..num_frames.saturating_sub(1) {
                        let (_r, val, w) = read_ebml_vint(&block[cur..])?;
                        cur += w;
                        let sz = if i == 0 {
                            val as i64
                        } else {
                            let bias = (1i64 << (7 * w as u32 - 1)) - 1;
                            prev_sz + (val as i64 - bias)
                        };
                        prev_sz = sz;
                        let usz = sz.max(0) as usize;
                        sum += usz;
                        frame_sizes.push(usz);
                    }
                    frame_sizes.push(block.len().saturating_sub(cur + sum));
                }
                _ => {}
            }

            let mut data_off = block_base_offset + cur;
            for fsz in frame_sizes {
                packets.push((
                    track_pos as u32,
                    SampleEntry::new(data_off as u64, pts, fsz as u32, 0, is_keyframe),
                    explicit_dur,
                ));
                data_off += fsz;
            }
        }
        Ok(())
    }
}

/// Zero-copy Matroska (`.mkv`) and WebM (`.webm`) Muxer with `Cues` keyframe index generation.
pub struct MkvMuxer<'a> {
    webm_mode: bool,
    tracks: Vec<TrackInfo>,
    packets: Vec<PacketSlice<'a>>,
}

impl<'a> Default for MkvMuxer<'a> {
    fn default() -> Self {
        Self::new_matroska()
    }
}

impl<'a> MkvMuxer<'a> {
    #[inline]
    pub const fn new_matroska() -> Self {
        Self {
            webm_mode: false,
            tracks: Vec::new(),
            packets: Vec::new(),
        }
    }

    #[inline]
    pub const fn new_webm() -> Self {
        Self {
            webm_mode: true,
            tracks: Vec::new(),
            packets: Vec::new(),
        }
    }

    pub fn add_track(&mut self, mut info: TrackInfo) -> u32 {
        let idx = self.tracks.len() as u32;
        info.index = idx;
        self.tracks.push(info);
        idx
    }

    pub fn write_packet(&mut self, packet: PacketSlice<'a>) -> Result<()> {
        if (packet.track_index as usize) >= self.tracks.len() {
            return Err(RagdollError::CorruptBitstream(
                "MkvMuxer::write_packet track_index out of bounds",
            ));
        }
        self.packets.push(packet);
        Ok(())
    }

    /// Writes a complete spec-compliant `.mkv` or `.webm` container with `Info`, `Tracks`,
    /// `Cluster` (`SimpleBlock`), and `Cues` keyframe seek table.
    pub fn finish<W: Write>(self, writer: W) -> Result<u64> {
        let mut vw = VectoredWriter::new(writer);
        let ms_tb = Rational64::new(1, 1000); // TimestampScale = 1_000_000 ns (1 ms)

        // 1. Build `EBML` header
        let doc_type = if self.webm_mode {
            b"webm" as &[u8]
        } else {
            b"matroska" as &[u8]
        };
        let mut ebml_body = Vec::with_capacity(48);
        Self::push_element(&mut ebml_body, ID_DOC_TYPE, doc_type);
        Self::push_uint(&mut ebml_body, ID_DOC_TYPE_VERSION, 4);
        Self::push_uint(&mut ebml_body, ID_DOC_TYPE_READ_VERSION, 2);
        let ebml_header = Self::wrap_element(ID_EBML, &ebml_body);

        // 2. Compute total duration in milliseconds
        let mut max_end_ms = 0i64;
        for pkt in &self.packets {
            let pts_ms = pkt.pts.rescale_to(ms_tb).ticks;
            let dur_ms = rescale_q(pkt.duration.max(1), pkt.pts.timebase, ms_tb).max(1);
            max_end_ms = max_end_ms.max(pts_ms + dur_ms);
        }

        // 3. Build `Info` element
        let mut info_body = Vec::with_capacity(64);
        Self::push_uint(&mut info_body, ID_TIMESTAMP_SCALE, 1_000_000);
        if max_end_ms > 0 {
            Self::push_f64(&mut info_body, ID_DURATION, max_end_ms as f64);
        }
        Self::push_element(&mut info_body, ID_MUXING_APP, b"Ragdoll v0.1.0");
        Self::push_element(&mut info_body, ID_WRITING_APP, b"Ragdoll v0.1.0");
        let info_elem = Self::wrap_element(ID_INFO, &info_body);

        // 4. Build `Tracks` element
        let tracks_elem = self.build_tracks_element()?;

        // 5. Group packets into `Cluster`s (start a new cluster on video keyframes or every 15s)
        struct ClusterPlan<'p> {
            cluster_ts_ms: i64,
            blocks: Vec<(u64, i16, u8, &'p [u8], bool)>,
            header_bytes: Vec<u8>,
            total_cluster_bytes: usize,
        }

        let mut clusters: Vec<ClusterPlan<'_>> = Vec::new();
        for pkt in &self.packets {
            let pts_ms = pkt.pts.rescale_to(ms_tb).ticks.max(0);
            let is_video = self.tracks[pkt.track_index as usize].track_type == TrackType::Video;
            let need_new_cluster = match clusters.last() {
                None => true,
                Some(c) => {
                    let rel = pts_ms - c.cluster_ts_ms;
                    !(-30_000..=30_000).contains(&rel)
                        || (is_video && pkt.is_keyframe() && !c.blocks.is_empty())
                }
            };
            if need_new_cluster {
                clusters.push(ClusterPlan {
                    cluster_ts_ms: pts_ms,
                    blocks: Vec::new(),
                    header_bytes: Vec::new(),
                    total_cluster_bytes: 0,
                });
            }
            let cur_cluster = clusters.last_mut().unwrap();
            let rel_ts =
                (pts_ms - cur_cluster.cluster_ts_ms).clamp(i16::MIN as i64, i16::MAX as i64) as i16;
            let flags = if pkt.is_keyframe() { 0x80u8 } else { 0x00u8 };
            let track_num = (pkt.track_index + 1) as u64;
            cur_cluster.blocks.push((
                track_num,
                rel_ts,
                flags,
                pkt.data(),
                is_video && pkt.is_keyframe(),
            ));
        }

        // Precompute whether any cluster contains a video keyframe so we know if `Cues` will be emitted,
        // and build a dummy `SeekHead` (using fixed 8-byte SeekPosition ints so byte length is invariant!)
        let has_cues = clusters.iter().any(|c| c.blocks.iter().any(|b| b.4));
        let dummy_seek_head = Self::build_seek_head_element(0, 0, has_cues.then_some(0));
        let seek_head_len = dummy_seek_head.len() as u64;
        let info_rel_offset = seek_head_len;
        let tracks_rel_offset = info_rel_offset + (info_elem.len() as u64);

        // Precompute exact byte size of every Cluster and record Cues positions relative to Segment payload start
        let mut cues_body = Vec::new();
        let mut cluster_rel_offset = tracks_rel_offset + (tracks_elem.len() as u64);

        for c in &mut clusters {
            let mut ts_elem = Vec::with_capacity(12);
            Self::push_uint(&mut ts_elem, ID_TIMESTAMP, c.cluster_ts_ms as u64);

            let mut cluster_payload_len = ts_elem.len();
            let mut logged_cue = false;

            for &(track_num, _rel_ts, _flags, data, is_video_kf) in &c.blocks {
                let mut tn_vint = Vec::with_capacity(8);
                write_ebml_vint(track_num, &mut tn_vint);
                let sb_payload_len = tn_vint.len() + 3 + data.len();
                let mut sb_sz_vint = Vec::with_capacity(8);
                write_ebml_vint(sb_payload_len as u64, &mut sb_sz_vint);
                cluster_payload_len += 1 + sb_sz_vint.len() + sb_payload_len;

                if is_video_kf && !logged_cue {
                    logged_cue = true;
                    let mut cue_pos = Vec::with_capacity(24);
                    Self::push_uint(&mut cue_pos, ID_CUE_TRACK, track_num);
                    Self::push_uint(&mut cue_pos, ID_CUE_CLUSTER_POSITION, cluster_rel_offset);
                    let cue_pos_elem = Self::wrap_element(ID_CUE_TRACK_POSITIONS, &cue_pos);

                    let mut cue_pt = Vec::with_capacity(12 + cue_pos_elem.len());
                    Self::push_uint(&mut cue_pt, ID_CUE_TIME, c.cluster_ts_ms as u64);
                    cue_pt.extend_from_slice(&cue_pos_elem);
                    let cue_pt_elem = Self::wrap_element(ID_CUE_POINT, &cue_pt);
                    cues_body.extend_from_slice(&cue_pt_elem);
                }
            }

            let mut cluster_hdr = Vec::with_capacity(12 + ts_elem.len());
            Self::push_id(&mut cluster_hdr, ID_CLUSTER);
            write_ebml_vint(cluster_payload_len as u64, &mut cluster_hdr);
            cluster_hdr.extend_from_slice(&ts_elem);

            c.total_cluster_bytes = (cluster_hdr.len() - ts_elem.len()) + cluster_payload_len;
            c.header_bytes = cluster_hdr;
            cluster_rel_offset += c.total_cluster_bytes as u64;
        }

        let cues_elem = if cues_body.is_empty() {
            Vec::new()
        } else {
            Self::wrap_element(ID_CUES, &cues_body)
        };

        let seek_head_elem = Self::build_seek_head_element(
            info_rel_offset,
            tracks_rel_offset,
            (!cues_elem.is_empty()).then_some(cluster_rel_offset),
        );
        debug_assert_eq!(dummy_seek_head.len(), seek_head_elem.len());

        let segment_payload_len = cluster_rel_offset + (cues_elem.len() as u64);
        let mut seg_hdr = Vec::with_capacity(12);
        Self::push_id(&mut seg_hdr, ID_SEGMENT);
        write_ebml_vint(segment_payload_len, &mut seg_hdr);

        // 6. Write EBML Header + Segment Header + SeekHead + Info + Tracks via vectored I/O
        vw.write_vectored_all(&[
            &ebml_header,
            &seg_hdr,
            &seek_head_elem,
            &info_elem,
            &tracks_elem,
        ])?;

        // 7. Stream each Cluster and its SimpleBlock payloads zero-copy via vectored I/O
        let mut sb_hdr = Vec::with_capacity(24);
        for c in &clusters {
            vw.write_all_counted(&c.header_bytes)?;
            for &(track_num, rel_ts, flags, data, _) in &c.blocks {
                sb_hdr.clear();
                Self::push_id(&mut sb_hdr, ID_SIMPLE_BLOCK);
                let mut tn_vint = [0u8; 8];
                let mut tn_vec = Vec::with_capacity(8);
                write_ebml_vint(track_num, &mut tn_vec);
                tn_vint[..tn_vec.len()].copy_from_slice(&tn_vec);

                let sb_payload_len = tn_vec.len() + 3 + data.len();
                write_ebml_vint(sb_payload_len as u64, &mut sb_hdr);
                sb_hdr.extend_from_slice(&tn_vint[..tn_vec.len()]);
                sb_hdr.extend_from_slice(&rel_ts.to_be_bytes());
                sb_hdr.push(flags);

                vw.write_vectored_all(&[&sb_hdr, data])?;
            }
        }

        if !cues_elem.is_empty() {
            vw.write_all_counted(&cues_elem)?;
        }

        vw.flush()?;
        Ok(vw.bytes_written())
    }

    /// Builds a deterministic-length `SeekHead` (`0x114D9B74`) indexing `Info`, `Tracks`, and `Cues`
    /// relative to the `Segment` payload start (using fixed 8-byte `SeekPosition` integers so its byte
    /// length is invariant before and after computing cluster positions).
    fn build_seek_head_element(info_pos: u64, tracks_pos: u64, cues_pos: Option<u64>) -> Vec<u8> {
        let mut sh_body = Vec::with_capacity(72);
        for &(target_id, pos) in &[
            (ID_INFO, Some(info_pos)),
            (ID_TRACKS, Some(tracks_pos)),
            (ID_CUES, cues_pos),
        ] {
            if let Some(p) = pos {
                let mut seek_entry = Vec::with_capacity(20);
                Self::push_element(&mut seek_entry, ID_SEEK_ID, &target_id.to_be_bytes());
                Self::push_element(&mut seek_entry, ID_SEEK_POSITION, &p.to_be_bytes());
                sh_body.extend_from_slice(&Self::wrap_element(ID_SEEK, &seek_entry));
            }
        }
        Self::wrap_element(ID_SEEK_HEAD, &sh_body)
    }

    fn build_tracks_element(&self) -> Result<Vec<u8>> {
        let mut tracks_body = Vec::new();
        for (i, t) in self.tracks.iter().enumerate() {
            let track_num = (i + 1) as u64;
            let mut te = Vec::with_capacity(128);
            Self::push_uint(&mut te, ID_TRACK_NUMBER, track_num);
            Self::push_uint(&mut te, ID_TRACK_UID, 1000 + track_num);
            let ttype_code = match t.track_type {
                TrackType::Video => 1,
                TrackType::Audio => 2,
                TrackType::Subtitle => 17,
                TrackType::Data => 0x20,
            };
            Self::push_uint(&mut te, ID_TRACK_TYPE, ttype_code);

            let codec_str = match t.codec_id {
                CodecId::H264 => "V_MPEG4/ISO/AVC",
                CodecId::Hevc => "V_MPEGH/ISO/HEVC",
                CodecId::Av1 => "V_AV1",
                CodecId::Vp9 => "V_VP9",
                CodecId::RagdollIntra => "V_RAGDOLL/INTRA",
                CodecId::RawVideo => "V_UNCOMPRESSED",
                CodecId::Flac => "A_FLAC",
                CodecId::Aac => "A_AAC",
                CodecId::Opus => "A_OPUS",
                CodecId::PcmF32Le => "A_PCM/FLOAT/IEEE",
                CodecId::PcmU8 | CodecId::PcmS16Le | CodecId::PcmS24Le | CodecId::PcmS32Le => {
                    "A_PCM/INT/LIT"
                }
            };
            Self::push_element(&mut te, ID_CODEC_ID, codec_str.as_bytes());

            if !t.extra_data.is_empty() {
                if t.codec_id == CodecId::Flac && t.extra_data.len() == 34 {
                    // Matroska A_FLAC CodecPrivate requires "fLaC" + 4-byte metadata block header + 34-byte STREAMINFO
                    let mut flac_priv = Vec::with_capacity(42);
                    flac_priv.extend_from_slice(b"fLaC");
                    flac_priv.push(0x80);
                    flac_priv.extend_from_slice(&[0x00, 0x00, 34]);
                    flac_priv.extend_from_slice(&t.extra_data);
                    Self::push_element(&mut te, ID_CODEC_PRIVATE, &flac_priv);
                } else {
                    Self::push_element(&mut te, ID_CODEC_PRIVATE, &t.extra_data);
                }
            }

            if t.codec_id == CodecId::Opus {
                // RFC 7845 / WebM Opus TrackEntry requirements:
                // CodecDelay (`0x56AA`) in nanoseconds (from OpusHead pre-skip @ 48 kHz, default 312 samples = 6_500_000 ns)
                // SeekPreRoll (`0x56BB`) = 80_000_000 ns (80 ms)
                let preskip_samples =
                    if t.extra_data.len() >= 12 && &t.extra_data[0..8] == b"OpusHead" {
                        u16::from_le_bytes([t.extra_data[10], t.extra_data[11]]) as u64
                    } else {
                        312
                    };
                let codec_delay_ns = (preskip_samples * 1_000_000_000) / 48_000;
                Self::push_uint(&mut te, ID_CODEC_DELAY, codec_delay_ns);
                Self::push_uint(&mut te, ID_SEEK_PRE_ROLL, 80_000_000);
            }

            if let Some(vp) = &t.video {
                if vp.frame_rate.num > 0 && vp.frame_rate.den > 0 {
                    let def_dur_ns = (1_000_000_000i128 * vp.frame_rate.den as i128
                        / vp.frame_rate.num as i128)
                        .max(1) as u64;
                    Self::push_uint(&mut te, ID_DEFAULT_DURATION, def_dur_ns);
                }
                let mut vbody = Vec::with_capacity(64);
                Self::push_uint(&mut vbody, ID_PIXEL_WIDTH, vp.width.max(1) as u64);
                Self::push_uint(&mut vbody, ID_PIXEL_HEIGHT, vp.height.max(1) as u64);

                let mut col_body = Vec::with_capacity(32);
                Self::push_uint(
                    &mut col_body,
                    ID_MATRIX_COEFFICIENTS,
                    vp.colorimetry.matrix as u64,
                );
                Self::push_uint(
                    &mut col_body,
                    ID_TRANSFER_CHARACTERISTICS,
                    vp.colorimetry.transfer as u64,
                );
                Self::push_uint(&mut col_body, ID_PRIMARIES, vp.colorimetry.primaries as u64);
                Self::push_uint(
                    &mut col_body,
                    ID_RANGE,
                    if vp.colorimetry.range == ColorRange::Full {
                        2
                    } else {
                        1
                    },
                );
                Self::push_uint(&mut col_body, ID_BITS_PER_CHANNEL, vp.bit_depth as u64);
                vbody.extend_from_slice(&Self::wrap_element(ID_COLOUR, &col_body));

                te.extend_from_slice(&Self::wrap_element(ID_VIDEO, &vbody));
            }

            if let Some(ap) = &t.audio {
                let mut abody = Vec::with_capacity(32);
                Self::push_f64(&mut abody, ID_SAMPLING_FREQUENCY, ap.sample_rate as f64);
                Self::push_uint(&mut abody, ID_CHANNELS, ap.channels.max(1) as u64);
                Self::push_uint(
                    &mut abody,
                    ID_BIT_DEPTH,
                    ap.bits_per_coded_sample.max(16) as u64,
                );
                te.extend_from_slice(&Self::wrap_element(ID_AUDIO, &abody));
            }

            tracks_body.extend_from_slice(&Self::wrap_element(ID_TRACK_ENTRY, &te));
        }
        Ok(Self::wrap_element(ID_TRACKS, &tracks_body))
    }

    #[inline]
    fn push_id(out: &mut Vec<u8>, id: u32) {
        if id > 0xFF_FFFF {
            out.extend_from_slice(&id.to_be_bytes());
        } else if id > 0xFFFF {
            let b = id.to_be_bytes();
            out.extend_from_slice(&b[1..4]);
        } else if id > 0xFF {
            let b = id.to_be_bytes();
            out.extend_from_slice(&b[2..4]);
        } else {
            out.push(id as u8);
        }
    }

    #[inline]
    fn wrap_element(id: u32, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(12 + payload.len());
        Self::push_id(&mut out, id);
        write_ebml_vint(payload.len() as u64, &mut out);
        out.extend_from_slice(payload);
        out
    }

    #[inline]
    fn push_element(out: &mut Vec<u8>, id: u32, payload: &[u8]) {
        Self::push_id(out, id);
        write_ebml_vint(payload.len() as u64, out);
        out.extend_from_slice(payload);
    }

    #[inline]
    fn push_uint(out: &mut Vec<u8>, id: u32, val: u64) {
        let be = val.to_be_bytes();
        let first_nonzero = be.iter().position(|&b| b != 0).unwrap_or(7);
        Self::push_element(out, id, &be[first_nonzero..]);
    }

    #[inline]
    fn push_f64(out: &mut Vec<u8>, id: u32, val: f64) {
        Self::push_element(out, id, &val.to_be_bytes());
    }
}
