//! **`0045`**: Native Standalone `.ogg` Container Demuxer & Muxer (`crates/ragdoll-format/src/ogg.rs`)
//! using `ragdoll_core::crc32_ogg` (`0x04C11DB7` Slicing-by-8 CRC-32) for RFC 3533 `OggS` page framing,
//! supporting both **Ogg Opus** (RFC 7845 `OpusHead` + `OpusTags`) and **Ogg FLAC** (RFC 9639 `\x7FFLAC` 1.0 mapping + `VORBIS_COMMENT`),
//! with `SampleEntry` indexing and $O(\log N)$ sample seeking.

use std::io::Write;

use ragdoll_core::{
    AlignedBuffer, AudioTrackParams, ChannelLayout, CodecId, PacketFlags, PacketSlice,
    RagdollError, Rational64, Result, SampleEntry, SampleFormat, Timestamp, TrackInfo, TrackType,
    crc32_ogg,
};

use crate::flac::{VorbisComments, parse_flac_frame_header_len_and_samples};

/// Builds a single RFC 3533 `OggS` page with Slicing-by-8 `crc32_ogg` checksum.
pub fn build_ogg_page(
    header_type: u8,
    granule_position: i64,
    serial_number: u32,
    page_sequence: u32,
    lacing_segments: &[u8],
    payload: &[u8],
) -> Vec<u8> {
    assert!(lacing_segments.len() <= 255);
    let hdr_len = 27 + lacing_segments.len();
    let mut page = Vec::with_capacity(hdr_len + payload.len());
    page.extend_from_slice(b"OggS");
    page.push(0); // stream_structure_version = 0
    page.push(header_type);
    page.extend_from_slice(&granule_position.to_le_bytes());
    page.extend_from_slice(&serial_number.to_le_bytes());
    page.extend_from_slice(&page_sequence.to_le_bytes());
    page.extend_from_slice(&[0u8; 4]); // CRC placeholder at 22..26
    page.push(lacing_segments.len() as u8);
    page.extend_from_slice(lacing_segments);
    page.extend_from_slice(payload);

    let crc = crc32_ogg(&page);
    page[22..26].copy_from_slice(&crc.to_le_bytes());
    page
}

#[derive(Clone, Debug)]
struct IndexedOggPacket {
    entry: SampleEntry,
    duration: i64,
    /// `Some((start, end))` if the packet resides in a single contiguous slice of `data`,
    /// or `None` if it was reassembled across multiple laced Ogg pages.
    contiguous_range: Option<(usize, usize)>,
    reassembled: Vec<u8>,
}

/// Native `.ogg` (Ogg Opus & Ogg FLAC) Demuxer with `crc32_ogg` verification, `VorbisComments`, and `SampleEntry` index.
pub struct OggDemuxer<'a> {
    data: &'a [u8],
    track: TrackInfo,
    pub comments: VorbisComments,
    pub samples: Vec<SampleEntry>,
    packets: Vec<IndexedOggPacket>,
    cursor: usize,
    final_granule: Option<i64>,
}

impl<'a> OggDemuxer<'a> {
    /// Opens and indexes an `.ogg` container (`OggS`) from `data`.
    pub fn open(data: &'a [u8]) -> Result<Self> {
        if data.len() < 27 || &data[0..4] != b"OggS" {
            return Err(RagdollError::InvalidHeader(
                "missing 'OggS' capture pattern".into(),
            ));
        }

        type RawOggPacket = (Vec<u8>, Option<(usize, usize)>, i64, u64);
        let mut pos = 0usize;
        let mut raw_packets: Vec<RawOggPacket> = Vec::new();
        let mut cur_pkt = Vec::new();
        let mut cur_contig: Option<(usize, usize)> = None;
        let mut cur_file_off = 0u64;
        let mut final_granule = None;

        while pos + 27 <= data.len() {
            if &data[pos..pos + 4] != b"OggS" {
                break;
            }
            let _header_type = data[pos + 5];
            let granule_pos = i64::from_le_bytes(data[pos + 6..pos + 14].try_into().unwrap());
            let expected_crc = u32::from_le_bytes(data[pos + 22..pos + 26].try_into().unwrap());
            let num_segs = data[pos + 26] as usize;
            let hdr_end = pos + 27 + num_segs;
            if hdr_end > data.len() {
                return Err(RagdollError::BitstreamEndOfData);
            }
            let seg_table = &data[pos + 27..hdr_end];
            let payload_len: usize = seg_table.iter().map(|&s| s as usize).sum();
            let page_end = hdr_end + payload_len;
            if page_end > data.len() {
                return Err(RagdollError::BitstreamEndOfData);
            }

            // Validate Ogg page CRC-32 (`0x04C11DB7`)
            let mut page_copy = data[pos..page_end].to_vec();
            page_copy[22..26].fill(0);
            let actual_crc = crc32_ogg(&page_copy);
            if actual_crc != expected_crc {
                return Err(RagdollError::InvalidData(format!(
                    "Ogg page CRC-32 mismatch at offset {pos}: expected {expected_crc:#010x}, got {actual_crc:#010x}"
                )));
            }

            if granule_pos >= 0 {
                final_granule = Some(granule_pos);
            }
            let mut seg_off = hdr_end;
            for &lacing in seg_table {
                let slen = lacing as usize;
                let seg_slice = &data[seg_off..seg_off + slen];
                if cur_pkt.is_empty() {
                    cur_file_off = seg_off as u64;
                    cur_contig = Some((seg_off, seg_off + slen));
                } else if let Some((s, e)) = cur_contig {
                    if e == seg_off {
                        cur_contig = Some((s, seg_off + slen));
                    } else {
                        cur_contig = None;
                    }
                }
                cur_pkt.extend_from_slice(seg_slice);
                seg_off += slen;

                if lacing < 255 {
                    raw_packets.push((
                        std::mem::take(&mut cur_pkt),
                        cur_contig.take(),
                        granule_pos,
                        cur_file_off,
                    ));
                }
            }
            pos = page_end;
        }

        if raw_packets.is_empty() {
            return Err(RagdollError::InvalidHeader(
                "Ogg container contains no packets".into(),
            ));
        }

        let first_pkt = &raw_packets[0].0;
        let mut codec_id = CodecId::Opus;
        let mut sample_rate = 48_000u32;
        let mut channels = 2u8;
        let mut bps = 16u8;
        let mut sample_format = SampleFormat::F32le;
        let mut extra_data = Vec::new();
        let mut comments = VorbisComments::default();
        let mut audio_start_idx = 1usize;

        if first_pkt.len() >= 19 && &first_pkt[0..8] == b"OpusHead" {
            codec_id = CodecId::Opus;
            channels = first_pkt[9].max(1);
            sample_rate = 48_000;
            bps = 32;
            sample_format = SampleFormat::F32le;
            extra_data = first_pkt.clone();
            if raw_packets.len() >= 2 && raw_packets[1].0.starts_with(b"OpusTags") {
                if let Ok(vc) = VorbisComments::parse(&raw_packets[1].0[8..]) {
                    comments = vc;
                }
                audio_start_idx = 2;
            }
        } else if first_pkt.len() >= 51 && first_pkt[0] == 0x7F && &first_pkt[1..5] == b"FLAC" {
            // Ogg FLAC 1.0 mapping: [0x7F, 'F','L','A','C', 1, 0, num_hdr_hi, num_hdr_lo, 'f','L','a','C', 0,0,0,34, STREAMINFO(34)]
            codec_id = CodecId::Flac;
            let si = &first_pkt[17..51];
            extra_data = si.to_vec();
            sample_rate = ((si[10] as u32) << 12) | ((si[11] as u32) << 4) | ((si[12] as u32) >> 4);
            channels = ((si[12] >> 1) & 0x07) + 1;
            bps = (((si[12] & 0x01) << 4) | (si[13] >> 4)) + 1;
            sample_format = match bps {
                1..=8 => SampleFormat::U8,
                9..=16 => SampleFormat::S16le,
                17..=24 => SampleFormat::S24le,
                _ => SampleFormat::S32le,
            };
            while audio_start_idx < raw_packets.len() {
                let p = &raw_packets[audio_start_idx].0;
                if p.len() >= 2 && p[0] == 0xFF && (p[1] & 0xFE) == 0xF8 {
                    break;
                }
                if p.len() > 4
                    && (p[0] & 0x7F) == 4
                    && let Ok(vc) = VorbisComments::parse(&p[4..])
                {
                    comments = vc;
                }
                audio_start_idx += 1;
            }
        }

        let mut samples = Vec::new();
        let mut packets = Vec::new();
        let mut cur_pts = 0i64;
        let mut total_bytes = 0u64;

        for (pkt_bytes, contig, _granule, file_off) in raw_packets.into_iter().skip(audio_start_idx)
        {
            if pkt_bytes.is_empty() {
                continue;
            }
            let dur = if codec_id == CodecId::Opus {
                // RFC 6716 section 3.1: frame duration from the TOC config, times the frame
                // count from the TOC code (code 3 carries the count in the second byte).
                let cfg = pkt_bytes[0] >> 3;
                let frames = match pkt_bytes[0] & 3 {
                    0 => 1,
                    1 | 2 => 2,
                    _ => pkt_bytes.get(1).map_or(0, |b| (b & 0x3f) as i64),
                };
                frames
                    * match cfg {
                        0..=11 => match cfg & 3 {
                            0 => 480i64,
                            1 => 960,
                            2 => 1920,
                            _ => 2880,
                        },
                        12..=15 => {
                            if (cfg & 1) == 0 {
                                480
                            } else {
                                960
                            }
                        }
                        _ => match cfg & 3 {
                            0 => 120,
                            1 => 240,
                            2 => 480,
                            _ => 960,
                        },
                    }
            } else if let Some((_, bs)) =
                parse_flac_frame_header_len_and_samples(&pkt_bytes, 16, 65535)
            {
                bs as i64
            } else {
                4096
            };

            let sz = pkt_bytes.len() as u32;
            total_bytes += sz as u64;
            let entry = SampleEntry::new(file_off, cur_pts, sz, 0, true);
            samples.push(entry);
            packets.push(IndexedOggPacket {
                entry,
                duration: dur,
                contiguous_range: contig,
                reassembled: if contig.is_some() {
                    Vec::new()
                } else {
                    pkt_bytes
                },
            });
            cur_pts += dur;
        }

        // RFC 7845 section 4.4: an Opus stream ends at its final granule position, which counts
        // the pre-skip; the last packet usually decodes past it.
        let duration = match final_granule {
            Some(g) if codec_id == CodecId::Opus && extra_data.len() >= 12 => {
                let pre_skip = u16::from_le_bytes([extra_data[10], extra_data[11]]) as i64;
                (g - pre_skip).clamp(0, cur_pts)
            }
            _ => cur_pts,
        };
        let track = TrackInfo {
            index: 0,
            track_type: TrackType::Audio,
            codec_id,
            timebase: Rational64::new(1, sample_rate.max(1) as i64),
            duration_ticks: Some(duration),
            extra_data,
            video: None,
            audio: Some(AudioTrackParams {
                sample_rate,
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
            comments,
            samples,
            packets,
            cursor: 0,
            final_granule,
        })
    }

    /// Granule position of the last page that has one: for Opus, the end of the audio counted
    /// from the start of decoding (pre-skip included); for FLAC, the total sample count.
    #[inline]
    pub fn final_granule(&self) -> Option<i64> {
        self.final_granule
    }

    #[inline]
    pub fn track(&self) -> &TrackInfo {
        &self.track
    }

    #[inline]
    pub fn comments(&self) -> &VorbisComments {
        &self.comments
    }

    /// Performs $O(\log N)$ binary search over the `SampleEntry` table to seek to `target_sample`.
    pub fn seek_to_sample(&mut self, target_sample: i64) -> usize {
        let idx = self
            .samples
            .partition_point(|s| s.pts <= target_sample)
            .saturating_sub(1);
        self.cursor = idx;
        idx
    }

    /// Reads the next audio packet from the `.ogg` container.
    pub fn next_packet(&mut self) -> Result<Option<PacketSlice<'a>>> {
        if self.cursor >= self.packets.len() {
            return Ok(None);
        }
        let item = &self.packets[self.cursor];
        self.cursor += 1;
        let ts = Timestamp::new(item.entry.pts, self.track.timebase);
        if let Some((start, end)) = item.contiguous_range {
            Ok(Some(PacketSlice::from_mmap(
                &self.data[start..end],
                0,
                ts,
                ts,
                item.duration,
                PacketFlags::KEYFRAME,
                item.entry.offset,
            )))
        } else {
            let mut pkt = PacketSlice::from_owned(
                AlignedBuffer::from_slice(&item.reassembled),
                0,
                ts,
                ts,
                item.duration,
                PacketFlags::KEYFRAME,
            );
            pkt.file_offset = item.entry.offset;
            Ok(Some(pkt))
        }
    }
}

/// Pages are flushed once they hold this many payload bytes (or one second of audio).
const OGG_PAGE_TARGET_BYTES: usize = 4096;

/// Accumulates lacing segments into pages for [`OggMuxer`].
struct Pager {
    serial: u32,
    seq: u32,
    segs: Vec<u8>,
    data: Vec<u8>,
    /// Granule position of the last packet completed on this page, or -1.
    granule: i64,
    continued: bool,
    written: u64,
}

impl Pager {
    fn flush<W: Write>(&mut self, writer: &mut W, extra_flags: u8) -> Result<()> {
        if self.segs.is_empty() && extra_flags == 0 {
            return Ok(());
        }
        let htype = if self.continued { 0x01 } else { 0x00 } | extra_flags;
        let page = build_ogg_page(
            htype,
            self.granule,
            self.serial,
            self.seq,
            &self.segs,
            &self.data,
        );
        writer.write_all(&page)?;
        self.written += page.len() as u64;
        self.seq += 1;
        self.segs.clear();
        self.data.clear();
        self.granule = -1;
        self.continued = false;
        Ok(())
    }
}

/// Native `.ogg` Container Muxer (`OggS`) supporting Ogg Opus (`OpusHead` + `OpusTags`) and
/// Ogg FLAC (`\x7FFLAC` 1.0 mapping + `VORBIS_COMMENT`) with `crc32_ogg` page checksums.
pub struct OggMuxer {
    track: Option<TrackInfo>,
    comments: VorbisComments,
    serial_number: u32,
    packets: Vec<(Vec<u8>, i64)>,
}

impl OggMuxer {
    pub fn new() -> Self {
        Self {
            track: None,
            comments: VorbisComments::new("ragdoll 0.1.0"),
            serial_number: 0x5241_4744, // "RAGD"
            packets: Vec::new(),
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
        let dur = pkt.duration.max(1);
        self.packets.push((data.to_vec(), dur));
        Ok(())
    }

    fn write_packet_pages<W: Write>(
        writer: &mut W,
        header_type_first: u8,
        header_type_last: u8,
        granule_position: i64,
        serial_number: u32,
        page_seq: &mut u32,
        pkt_bytes: &[u8],
    ) -> Result<u64> {
        let mut all_lacing = Vec::with_capacity(pkt_bytes.len() / 255 + 1);
        let mut rem = pkt_bytes.len();
        while rem >= 255 {
            all_lacing.push(255u8);
            rem -= 255;
        }
        all_lacing.push(rem as u8);

        let mut written = 0u64;
        let mut seg_idx = 0usize;
        let mut byte_off = 0usize;
        let total_segs = all_lacing.len();

        while seg_idx < total_segs {
            let take_segs = (total_segs - seg_idx).min(255);
            let seg_slice = &all_lacing[seg_idx..seg_idx + take_segs];
            let chunk_bytes: usize = seg_slice.iter().map(|&s| s as usize).sum();
            let is_first = seg_idx == 0;
            let is_last = seg_idx + take_segs == total_segs;

            let mut htype = if is_first { header_type_first } else { 0x01 };
            if is_last {
                htype |= header_type_last;
            }
            let gpos = if is_last { granule_position } else { -1 };

            let page = build_ogg_page(
                htype,
                gpos,
                serial_number,
                *page_seq,
                seg_slice,
                &pkt_bytes[byte_off..byte_off + chunk_bytes],
            );
            *page_seq += 1;
            writer.write_all(&page)?;
            written += page.len() as u64;

            seg_idx += take_segs;
            byte_off += chunk_bytes;
        }
        Ok(written)
    }

    /// Finalizes and writes the complete `.ogg` container with `crc32_ogg` pages.
    pub fn finish<W: Write>(self, mut writer: W) -> Result<u64> {
        let track = self
            .track
            .ok_or_else(|| RagdollError::InvalidData("OggMuxer has no audio track".into()))?;
        let audio = track.audio.unwrap_or_default();
        let mut written = 0u64;
        let mut page_seq = 0u32;

        if track.codec_id == CodecId::Flac {
            // 1. Ogg FLAC 1.0 BOS identification packet (51 bytes)
            let mut bos = Vec::with_capacity(51);
            bos.push(0x7F);
            bos.extend_from_slice(b"FLAC");
            bos.push(1); // major version 1
            bos.push(0); // minor version 0
            bos.extend_from_slice(&1u16.to_be_bytes()); // 1 non-audio header packet follows (VORBIS_COMMENT)
            bos.extend_from_slice(b"fLaC");
            bos.extend_from_slice(&[0x00, 0x00, 0x00, 34]); // STREAMINFO block header
            if track.extra_data.len() >= 34 {
                bos.extend_from_slice(&track.extra_data[0..34]);
            } else {
                let mut si = [0u8; 34];
                si[0..2].copy_from_slice(&4096u16.to_be_bytes());
                si[2..4].copy_from_slice(&4096u16.to_be_bytes());
                let sr = audio.sample_rate.max(8000);
                let ch = audio.channels.clamp(1, 8) - 1;
                let bps = audio.bits_per_coded_sample.clamp(8, 32) - 1;
                si[10] = (sr >> 12) as u8;
                si[11] = (sr >> 4) as u8;
                si[12] = (((sr & 0x0F) as u8) << 4) | ((ch & 0x07) << 1) | ((bps >> 4) & 0x01);
                si[13] = (bps & 0x0F) << 4;
                bos.extend_from_slice(&si);
            }
            written += Self::write_packet_pages(
                &mut writer,
                0x02,
                0x00,
                0,
                self.serial_number,
                &mut page_seq,
                &bos,
            )?;

            // 2. Ogg FLAC VORBIS_COMMENT packet (0x84 + 24-bit BE length + Vorbis Comment body)
            let vc_body = self.comments.to_bytes();
            let vc_len = vc_body.len();
            let mut vc_pkt = Vec::with_capacity(4 + vc_len);
            vc_pkt.push(0x84);
            vc_pkt.push(((vc_len >> 16) & 0xFF) as u8);
            vc_pkt.push(((vc_len >> 8) & 0xFF) as u8);
            vc_pkt.push((vc_len & 0xFF) as u8);
            vc_pkt.extend_from_slice(&vc_body);
            written += Self::write_packet_pages(
                &mut writer,
                0x00,
                0x00,
                0,
                self.serial_number,
                &mut page_seq,
                &vc_pkt,
            )?;
        } else {
            // 1. Ogg Opus BOS identification packet (19-byte OpusHead)
            let bos = if track.extra_data.len() >= 19 && &track.extra_data[0..8] == b"OpusHead" {
                track.extra_data.clone()
            } else {
                let mut head = [0u8; 19];
                head[0..8].copy_from_slice(b"OpusHead");
                head[8] = 1;
                head[9] = audio.channels.clamp(1, 2);
                // pre_skip = 0 so granule positions match sample count directly
                head[10..12].copy_from_slice(&0u16.to_le_bytes());
                head[12..16].copy_from_slice(&audio.sample_rate.max(48_000).to_le_bytes());
                head[16..18].copy_from_slice(&0i16.to_le_bytes());
                head[18] = 0;
                head.to_vec()
            };
            written += Self::write_packet_pages(
                &mut writer,
                0x02,
                0x00,
                0,
                self.serial_number,
                &mut page_seq,
                &bos,
            )?;

            // 2. Ogg Opus OpusTags packet ("OpusTags" + Vorbis Comment body)
            let vc_body = self.comments.to_bytes();
            let mut tags_pkt = Vec::with_capacity(8 + vc_body.len());
            tags_pkt.extend_from_slice(b"OpusTags");
            tags_pkt.extend_from_slice(&vc_body);
            written += Self::write_packet_pages(
                &mut writer,
                0x00,
                0x00,
                0,
                self.serial_number,
                &mut page_seq,
                &tags_pkt,
            )?;
        }

        // 3. Audio packets, several per page. The stream's final granule position is the real
        // end of the audio when the track duration says the packets run past it (Opus end
        // padding, RFC 7845 section 4.4).
        let rate = if track.codec_id == CodecId::Opus {
            48_000
        } else {
            audio.sample_rate.max(1) as i64
        };
        let total: i64 = self.packets.iter().map(|(_, d)| *d).sum();
        let pre_skip = if track.codec_id == CodecId::Opus
            && track.extra_data.len() >= 12
            && &track.extra_data[0..8] == b"OpusHead"
        {
            u16::from_le_bytes([track.extra_data[10], track.extra_data[11]]) as i64
        } else {
            0
        };
        let end_granule = match track.duration_ticks {
            Some(d) if d > 0 && track.codec_id == CodecId::Opus => total.min(pre_skip + d),
            _ => total,
        };
        let mut pager = Pager {
            serial: self.serial_number,
            seq: page_seq,
            segs: Vec::new(),
            data: Vec::new(),
            granule: -1,
            continued: false,
            written: 0,
        };
        let num_pkts = self.packets.len();
        let mut granule = 0i64;
        let mut page_start = 0i64;
        for (idx, (pkt_bytes, dur)) in self.packets.into_iter().enumerate() {
            granule += dur;
            let last = idx + 1 == num_pkts;
            let pkt_granule = if last { end_granule } else { granule };
            let mut rem = pkt_bytes.len();
            let mut off = 0usize;
            loop {
                if pager.segs.len() == 255 {
                    pager.flush(&mut writer, 0)?;
                    // The next page continues this packet unless it starts here.
                    pager.continued = off > 0;
                    page_start = granule - dur;
                }
                let seg = rem.min(255);
                pager.segs.push(seg as u8);
                pager.data.extend_from_slice(&pkt_bytes[off..off + seg]);
                off += seg;
                rem -= seg;
                if seg < 255 {
                    break;
                }
            }
            pager.granule = pkt_granule;
            if last {
                pager.flush(&mut writer, 0x04)?;
            } else if pager.data.len() >= OGG_PAGE_TARGET_BYTES || granule - page_start >= rate {
                pager.flush(&mut writer, 0)?;
                page_start = granule;
            }
        }
        written += pager.written;
        writer.flush()?;
        Ok(written)
    }
}

impl Default for OggMuxer {
    fn default() -> Self {
        Self::new()
    }
}
