use crate::io::VectoredWriter;
use ragdoll_core::{
    AlignedBuffer, AudioTrackParams, CodecId, PacketFlags, PacketSlice, RagdollError, Rational64,
    Result, Timestamp, TrackInfo, TrackType, VideoTrackParams, crc32_mpeg2, rescale_q,
};
use ragdoll_simd::{find_mpeg_ts_sync, split_annex_b_nals};
use std::io::Write;

pub const TS_PACKET_SIZE: usize = 188;
pub const TS_SYNC_BYTE: u8 = 0x47;
pub const PID_PAT: u16 = 0x0000;
pub const PID_DEFAULT_PMT: u16 = 0x1000;
pub const PID_BASE_ES: u16 = 0x0100;

/// Encodes a 33-bit 90 kHz timestamp into the 5-byte MPEG-2 PES marker-bit format.
#[inline]
pub fn encode_pes_timestamp(prefix_4bits: u8, ts_33bit: i64) -> [u8; 5] {
    let t = (ts_33bit as u64) & 0x1_FFFF_FFFF;
    let b0 = ((prefix_4bits & 0x0F) << 4) | ((((t >> 30) & 0x07) as u8) << 1) | 1;
    let b1 = ((t >> 22) & 0xFF) as u8;
    let b2 = ((((t >> 15) & 0x7F) as u8) << 1) | 1;
    let b3 = ((t >> 7) & 0xFF) as u8;
    let b4 = (((t & 0x7F) as u8) << 1) | 1;
    [b0, b1, b2, b3, b4]
}

/// Decodes a 33-bit 90 kHz timestamp from a 5-byte MPEG-2 PES timestamp field.
#[inline]
pub fn decode_pes_timestamp(b: &[u8; 5]) -> i64 {
    ((((b[0] >> 1) & 0x07) as i64) << 30)
        | ((b[1] as i64) << 22)
        | ((((b[2] >> 1) & 0x7F) as i64) << 15)
        | ((b[3] as i64) << 7)
        | (((b[4] >> 1) & 0x7F) as i64)
}

/// Converts an ISO `avcC` configuration box + AVCC (4-byte length-prefixed) H.264 access unit
/// into an Annex-B byte stream (`00 00 00 01` start codes, including `AUD` + `SPS`/`PPS` on keyframes).
pub fn h264_avcc_to_annex_b_au(
    avcc_extra: &[u8],
    avcc_packet: &[u8],
    is_keyframe: bool,
) -> Vec<u8> {
    // Check if packet already starts with Annex-B 00 00 01 or 00 00 00 01
    if (avcc_packet.len() >= 4 && avcc_packet[0..4] == [0, 0, 0, 1])
        || (avcc_packet.len() >= 3 && avcc_packet[0..3] == [0, 0, 1])
    {
        return avcc_packet.to_vec();
    }

    let mut out = Vec::with_capacity(avcc_packet.len() + avcc_extra.len() + 32);
    // Emit H.264 Access Unit Delimiter (AUD, NAL type 9) required by MPEG-TS broadcast decoders
    out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x09, 0xF0]);

    let mut nal_len_size = 4usize;
    if avcc_extra.len() >= 7 && avcc_extra[0] == 1 {
        nal_len_size = ((avcc_extra[4] & 0x03) + 1) as usize;
        if is_keyframe {
            let num_sps = (avcc_extra[5] & 0x1F) as usize;
            let mut pos = 6usize;
            for _ in 0..num_sps {
                if pos + 2 > avcc_extra.len() {
                    break;
                }
                let sps_len = u16::from_be_bytes([avcc_extra[pos], avcc_extra[pos + 1]]) as usize;
                pos += 2;
                if pos + sps_len > avcc_extra.len() {
                    break;
                }
                out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
                out.extend_from_slice(&avcc_extra[pos..pos + sps_len]);
                pos += sps_len;
            }
            if pos < avcc_extra.len() {
                let num_pps = avcc_extra[pos] as usize;
                pos += 1;
                for _ in 0..num_pps {
                    if pos + 2 > avcc_extra.len() {
                        break;
                    }
                    let pps_len =
                        u16::from_be_bytes([avcc_extra[pos], avcc_extra[pos + 1]]) as usize;
                    pos += 2;
                    if pos + pps_len > avcc_extra.len() {
                        break;
                    }
                    out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
                    out.extend_from_slice(&avcc_extra[pos..pos + pps_len]);
                    pos += pps_len;
                }
            }
        }
    }

    let mut pos = 0usize;
    while pos + nal_len_size <= avcc_packet.len() {
        let mut nal_len = 0usize;
        for i in 0..nal_len_size {
            nal_len = (nal_len << 8) | (avcc_packet[pos + i] as usize);
        }
        pos += nal_len_size;
        if nal_len == 0 || pos + nal_len > avcc_packet.len() {
            break;
        }
        out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        out.extend_from_slice(&avcc_packet[pos..pos + nal_len]);
        pos += nal_len;
    }

    if out.len() <= 6 && !avcc_packet.is_empty() {
        // Fallback if input wasn't length-prefixed
        out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        out.extend_from_slice(avcc_packet);
    }
    out
}

/// Synthesizes an ISO `avcC` decoder configuration record from Annex-B `SPS` and `PPS` NAL units.
pub fn build_avcc_from_sps_pps(sps: &[u8], pps: &[u8]) -> Vec<u8> {
    if sps.len() < 4 {
        return Vec::new();
    }
    let mut avcc = Vec::with_capacity(11 + sps.len() + pps.len());
    avcc.push(1); // configurationVersion = 1
    avcc.push(sps[1]); // AVCProfileIndication
    avcc.push(sps[2]); // profile_compatibility
    avcc.push(sps[3]); // AVCLevelIndication
    avcc.push(0xFF); // 6 bits reserved (111111) + lengthSizeMinusOne = 3 (4 bytes)
    avcc.push(0xE1); // 3 bits reserved (111) + numOfSequenceParameterSets = 1
    avcc.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    avcc.extend_from_slice(sps);
    avcc.push(1); // numOfPictureParameterSets = 1
    avcc.extend_from_slice(&(pps.len() as u16).to_be_bytes());
    avcc.extend_from_slice(pps);
    avcc
}

enum TsSliceStorage<'a> {
    ZeroCopy(&'a [u8]),
    Reassembled(AlignedBuffer),
}

struct DemuxedPesPacket<'a> {
    storage: TsSliceStorage<'a>,
    track_index: u32,
    pts: i64,
    dts: i64,
    duration: i64,
    is_keyframe: bool,
    file_offset: u64,
}

/// SIMD-accelerated MPEG-2 Transport Stream (`.ts`) Demuxer (`ISO/IEC 13818-1`).
///
/// Uses `ragdoll-simd`'s NEON `0x47` stride-188 sync scanner to lock onto transport packets,
/// verifies `PAT`/`PMT` PSI tables via hardware Slicing-by-8 `crc32_mpeg2`, and emits zero-copy
/// [`PacketSlice<'a>`] views for single-packet access units or 64-byte-aligned reassembled buffers
/// for multi-packet PES frames.
pub struct MpegTsDemuxer<'a> {
    tracks: Vec<TrackInfo>,
    packets: Vec<DemuxedPesPacket<'a>>,
    cursor: usize,
}

impl<'a> MpegTsDemuxer<'a> {
    pub fn open(data: &'a [u8]) -> Result<Self> {
        let Some(sync_start) = find_mpeg_ts_sync(data) else {
            return Err(RagdollError::InvalidHeader(
                "could not lock onto 188-byte 0x47 MPEG-TS sync byte".to_string(),
            ));
        };

        let mut pmt_pid: Option<u16> = None;
        let mut tracks: Vec<TrackInfo> = Vec::new();
        let mut pid_to_track: Vec<(u16, u32)> = Vec::new();

        struct PesAccum<'p> {
            pts: i64,
            dts: i64,
            file_offset: u64,
            first_slice: Option<&'p [u8]>,
            reassembled: Vec<u8>,
        }

        let mut accumulators: Vec<Option<PesAccum<'a>>> = Vec::new();
        let mut packets: Vec<DemuxedPesPacket<'a>> = Vec::new();

        let flush_accum = |track_idx: usize,
                           acc: PesAccum<'a>,
                           tracks: &mut [TrackInfo],
                           packets: &mut Vec<DemuxedPesPacket<'a>>| {
            let (storage, bytes_ref): (TsSliceStorage<'a>, &[u8]) = if acc.reassembled.is_empty() {
                if let Some(s) = acc.first_slice {
                    (TsSliceStorage::ZeroCopy(s), s)
                } else {
                    return;
                }
            } else {
                let buf = AlignedBuffer::from_slice(&acc.reassembled);
                (TsSliceStorage::Reassembled(buf), &[])
            };

            let payload_slice: &[u8] = match &storage {
                TsSliceStorage::ZeroCopy(s) => s,
                TsSliceStorage::Reassembled(b) => b.as_slice(),
            };
            let _ = bytes_ref;
            if payload_slice.is_empty() {
                return;
            }

            let track = &mut tracks[track_idx];
            let mut is_keyframe = track.track_type == TrackType::Audio;
            if track.codec_id == CodecId::H264 {
                let nals = split_annex_b_nals(payload_slice);
                let mut sps: Option<&[u8]> = None;
                let mut pps: Option<&[u8]> = None;
                for nal in nals {
                    if !nal.is_empty() {
                        let nal_type = nal[0] & 0x1F;
                        if nal_type == 5 {
                            is_keyframe = true;
                        } else if nal_type == 7 {
                            sps = Some(nal);
                            is_keyframe = true;
                        } else if nal_type == 8 {
                            pps = Some(nal);
                        }
                    }
                }
                if track.extra_data.is_empty()
                    && let (Some(s), Some(p)) = (sps, pps)
                {
                    track.extra_data = build_avcc_from_sps_pps(s, p);
                }
            } else if track.track_type == TrackType::Video && track.sample_count == 0 {
                is_keyframe = true;
            }

            track.sample_count += 1;
            track.total_bytes += payload_slice.len() as u64;

            packets.push(DemuxedPesPacket {
                storage,
                track_index: track_idx as u32,
                pts: acc.pts,
                dts: acc.dts,
                duration: 3000, // default 30fps in 90kHz ticks; refined below
                is_keyframe,
                file_offset: acc.file_offset,
            });
        };

        let mut pos = sync_start;
        while pos + TS_PACKET_SIZE <= data.len() {
            if data[pos] != TS_SYNC_BYTE {
                if let Some(rel) = find_mpeg_ts_sync(&data[pos..]) {
                    pos += rel;
                    continue;
                } else {
                    break;
                }
            }

            let pkt = &data[pos..pos + TS_PACKET_SIZE];
            let pusi = (pkt[1] & 0x40) != 0;
            let pid = (((pkt[1] & 0x1F) as u16) << 8) | (pkt[2] as u16);
            let afc = (pkt[3] >> 4) & 0x03;

            let payload_offset = match afc {
                1 => 4usize,
                3 => {
                    let adapt_len = pkt[4] as usize;
                    5 + adapt_len
                }
                _ => TS_PACKET_SIZE,
            };

            if payload_offset < TS_PACKET_SIZE {
                let payload = &pkt[payload_offset..];
                if pid == PID_PAT {
                    if let Some(parsed_pmt) = Self::parse_pat_section(payload, pusi) {
                        pmt_pid = Some(parsed_pmt);
                    }
                } else if Some(pid) == pmt_pid {
                    Self::parse_pmt_section(payload, pusi, &mut tracks, &mut pid_to_track)?;
                    if accumulators.len() < tracks.len() {
                        accumulators.resize_with(tracks.len(), || None);
                    }
                } else if let Some(&(_, track_idx)) = pid_to_track.iter().find(|&&(p, _)| p == pid)
                {
                    let t_idx = track_idx as usize;
                    if pusi {
                        if let Some(prev_acc) = accumulators[t_idx].take() {
                            flush_accum(t_idx, prev_acc, &mut tracks, &mut packets);
                        }
                        if payload.len() >= 9 && payload[0..3] == [0x00, 0x00, 0x01] {
                            let pts_dts_flags = (payload[7] >> 6) & 0x03;
                            let pes_hdr_len = payload[8] as usize;
                            let es_start = 9 + pes_hdr_len;
                            let mut pts = 0i64;
                            let mut dts = 0i64;
                            if pts_dts_flags >= 2 && payload.len() >= 14 {
                                let pts_bytes: [u8; 5] =
                                    payload[9..14].try_into().unwrap_or([0; 5]);
                                pts = decode_pes_timestamp(&pts_bytes);
                                dts = pts;
                                if pts_dts_flags == 3 && payload.len() >= 19 {
                                    let dts_bytes: [u8; 5] =
                                        payload[14..19].try_into().unwrap_or([0; 5]);
                                    dts = decode_pes_timestamp(&dts_bytes);
                                }
                            }
                            if es_start <= payload.len() {
                                let es_slice = &payload[es_start..];
                                let file_off = (pos + payload_offset + es_start) as u64;
                                accumulators[t_idx] = Some(PesAccum {
                                    pts,
                                    dts,
                                    file_offset: file_off,
                                    first_slice: Some(es_slice),
                                    reassembled: Vec::new(),
                                });
                            }
                        }
                    } else if let Some(acc) = &mut accumulators[t_idx] {
                        if acc.reassembled.is_empty()
                            && let Some(first) = acc.first_slice.take()
                        {
                            acc.reassembled.reserve(first.len() + payload.len() * 4);
                            acc.reassembled.extend_from_slice(first);
                        }
                        acc.reassembled.extend_from_slice(payload);
                    }
                }
            }

            pos += TS_PACKET_SIZE;
        }

        for (t_idx, slot) in accumulators.into_iter().enumerate() {
            if let Some(acc) = slot {
                flush_accum(t_idx, acc, &mut tracks, &mut packets);
            }
        }

        // Refine packet durations from consecutive DTS deltas per track
        for t_idx in 0..tracks.len() {
            let mut prev_pos: Option<usize> = None;
            let mut last_dur = 3000i64;
            for i in 0..packets.len() {
                if packets[i].track_index as usize == t_idx {
                    if let Some(p) = prev_pos {
                        let d = packets[i].dts - packets[p].dts;
                        if d > 0 {
                            packets[p].duration = d;
                            last_dur = d;
                        }
                    }
                    prev_pos = Some(i);
                }
            }
            if let Some(p) = prev_pos {
                packets[p].duration = last_dur;
            }
        }

        Ok(Self {
            tracks,
            packets,
            cursor: 0,
        })
    }

    #[inline]
    pub fn tracks(&self) -> &[TrackInfo] {
        &self.tracks
    }

    #[inline]
    pub fn total_packets(&self) -> usize {
        self.packets.len()
    }

    #[inline]
    pub fn next_packet(&mut self) -> Result<Option<PacketSlice<'a>>> {
        if self.cursor >= self.packets.len() {
            return Ok(None);
        }
        let pkt = &self.packets[self.cursor];
        self.cursor += 1;

        let tb = Rational64::TIMEBASE_MPEG_90K;
        let pts = Timestamp::new(pkt.pts, tb);
        let dts = Timestamp::new(pkt.dts, tb);
        let flags = if pkt.is_keyframe {
            PacketFlags::KEYFRAME
        } else {
            PacketFlags::NONE
        };

        let slice = match &pkt.storage {
            TsSliceStorage::ZeroCopy(s) => PacketSlice::from_mmap(
                s,
                pkt.track_index,
                pts,
                dts,
                pkt.duration,
                flags,
                pkt.file_offset,
            ),
            TsSliceStorage::Reassembled(buf) => {
                let mut ps = PacketSlice::from_owned(
                    buf.clone(),
                    pkt.track_index,
                    pts,
                    dts,
                    pkt.duration,
                    flags,
                );
                ps.file_offset = pkt.file_offset;
                ps
            }
        };

        Ok(Some(slice))
    }

    fn parse_pat_section(payload: &[u8], pusi: bool) -> Option<u16> {
        let start = if pusi {
            if payload.is_empty() {
                return None;
            }
            1 + (payload[0] as usize)
        } else {
            0
        };
        if start + 12 > payload.len() {
            return None;
        }
        let sec = &payload[start..];
        if sec[0] != 0x00 {
            return None;
        }
        let sec_len = (((sec[1] & 0x0F) as usize) << 8) | (sec[2] as usize);
        let total_sec = 3 + sec_len;
        if total_sec < 12 || total_sec > sec.len() {
            return None;
        }
        // Verify CRC-32/MPEG-2 over section including 4-byte CRC (remainder must be 0)
        if crc32_mpeg2(&sec[..total_sec]) != 0 {
            return None;
        }
        let mut pos = 8usize;
        while pos + 4 <= total_sec - 4 {
            let prog_num = u16::from_be_bytes([sec[pos], sec[pos + 1]]);
            let pid = u16::from_be_bytes([sec[pos + 2], sec[pos + 3]]) & 0x1FFF;
            if prog_num != 0 {
                return Some(pid);
            }
            pos += 4;
        }
        None
    }

    fn parse_pmt_section(
        payload: &[u8],
        pusi: bool,
        tracks: &mut Vec<TrackInfo>,
        pid_to_track: &mut Vec<(u16, u32)>,
    ) -> Result<()> {
        let start = if pusi {
            if payload.is_empty() {
                return Ok(());
            }
            1 + (payload[0] as usize)
        } else {
            0
        };
        if start + 16 > payload.len() {
            return Ok(());
        }
        let sec = &payload[start..];
        if sec[0] != 0x02 {
            return Ok(());
        }
        let sec_len = (((sec[1] & 0x0F) as usize) << 8) | (sec[2] as usize);
        let total_sec = 3 + sec_len;
        if total_sec < 16 || total_sec > sec.len() {
            return Ok(());
        }
        if crc32_mpeg2(&sec[..total_sec]) != 0 {
            return Err(RagdollError::ChecksumMismatch {
                expected: 0,
                actual: crc32_mpeg2(&sec[..total_sec]),
            });
        }
        let prog_info_len = (((sec[10] & 0x0F) as usize) << 8) | (sec[11] as usize);
        let mut pos = 12 + prog_info_len;
        while pos + 5 <= total_sec - 4 {
            let stream_type = sec[pos];
            let es_pid = u16::from_be_bytes([sec[pos + 1], sec[pos + 2]]) & 0x1FFF;
            let es_info_len = (((sec[pos + 3] & 0x0F) as usize) << 8) | (sec[pos + 4] as usize);
            pos += 5 + es_info_len;

            if pid_to_track.iter().any(|&(p, _)| p == es_pid) {
                continue;
            }

            let (track_type, codec_id) = match stream_type {
                0x1B => (TrackType::Video, CodecId::H264),
                0x24 => (TrackType::Video, CodecId::Hevc),
                0x0F => (TrackType::Audio, CodecId::Aac),
                0x81 | 0x06 => (TrackType::Audio, CodecId::Flac),
                _ => continue,
            };

            let idx = tracks.len() as u32;
            tracks.push(TrackInfo {
                index: idx,
                track_type,
                codec_id,
                timebase: Rational64::TIMEBASE_MPEG_90K,
                duration_ticks: None,
                extra_data: Vec::new(),
                video: (track_type == TrackType::Video).then(|| VideoTrackParams {
                    width: 1920,
                    height: 1080,
                    ..Default::default()
                }),
                audio: (track_type == TrackType::Audio).then(AudioTrackParams::default),
                sample_count: 0,
                total_bytes: 0,
            });
            pid_to_track.push((es_pid, idx));
        }
        Ok(())
    }
}

/// Spec-compliant MPEG-2 Transport Stream (`.ts`) Muxer (`ISO/IEC 13818-1`).
///
/// Emits 188-byte packets with hardware Slicing-by-8 `crc32_mpeg2` PAT/PMT tables,
/// 27 MHz / 90 kHz PCR clock references in keyframe adaptation fields, and automatic
/// H.264 AVCC $\to$ Annex-B NAL framing.
pub struct MpegTsMuxer<W: Write> {
    writer: VectoredWriter<W>,
    tracks: Vec<TrackInfo>,
    cc_pat: u8,
    cc_pmt: u8,
    cc_tracks: Vec<u8>,
    packets_since_psi: usize,
}

impl<W: Write> MpegTsMuxer<W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer: VectoredWriter::new(writer),
            tracks: Vec::new(),
            cc_pat: 0,
            cc_pmt: 0,
            cc_tracks: Vec::new(),
            packets_since_psi: usize::MAX,
        }
    }

    pub fn add_track(&mut self, mut info: TrackInfo) -> u32 {
        let idx = self.tracks.len() as u32;
        info.index = idx;
        self.tracks.push(info);
        self.cc_tracks.push(0);
        idx
    }

    pub fn write_packet(&mut self, packet: &PacketSlice<'_>) -> Result<()> {
        let t_idx = packet.track_index as usize;
        if t_idx >= self.tracks.len() {
            return Err(RagdollError::CorruptBitstream(
                "MpegTsMuxer::write_packet track_index out of bounds",
            ));
        }

        // Emit PAT + PMT at stream start, on every video keyframe, or every 32 packets
        if self.packets_since_psi >= 32 || packet.is_keyframe() {
            self.write_pat_packet()?;
            self.write_pmt_packet()?;
            self.packets_since_psi = 0;
        }
        self.packets_since_psi += 1;

        let track = &self.tracks[t_idx];
        let pid = PID_BASE_ES + (t_idx as u16);
        let tb_90k = Rational64::TIMEBASE_MPEG_90K;
        let pts_90k = rescale_q(packet.pts.ticks, packet.pts.timebase, tb_90k).max(0);
        let dts_90k = rescale_q(packet.dts.ticks, packet.dts.timebase, tb_90k).max(0);

        // Convert H.264 AVCC to Annex-B if needed
        let converted_annex_b: Option<Vec<u8>> = if track.codec_id == CodecId::H264 {
            Some(h264_avcc_to_annex_b_au(
                &track.extra_data,
                packet.data(),
                packet.is_keyframe(),
            ))
        } else {
            None
        };
        let es_payload: &[u8] = converted_annex_b
            .as_deref()
            .unwrap_or_else(|| packet.data());

        // Build PES header (`00 00 01 <stream_id>`)
        let stream_id = if track.track_type == TrackType::Video {
            0xE0u8
        } else {
            0xC0u8
        };
        let has_distinct_dts = pts_90k != dts_90k;
        let pts_dts_flags = if has_distinct_dts { 0xC0u8 } else { 0x80u8 };
        let pes_hdr_data_len = if has_distinct_dts { 10u8 } else { 5u8 };
        let total_pes_len = 3 + (pes_hdr_data_len as usize) + es_payload.len();
        // For video streams, PES_packet_length can be 0 (unbounded) if > 65535
        let pes_len_field = if track.track_type == TrackType::Video && total_pes_len > 65535 {
            0u16
        } else {
            total_pes_len.min(65535) as u16
        };

        let mut pes_hdr = Vec::with_capacity(19);
        pes_hdr.extend_from_slice(&[0x00, 0x00, 0x01, stream_id]);
        pes_hdr.extend_from_slice(&pes_len_field.to_be_bytes());
        pes_hdr.push(0x84); // '10', data_alignment_indicator = 1
        pes_hdr.push(pts_dts_flags);
        pes_hdr.push(pes_hdr_data_len);
        if has_distinct_dts {
            pes_hdr.extend_from_slice(&encode_pes_timestamp(0x03, pts_90k));
            pes_hdr.extend_from_slice(&encode_pes_timestamp(0x01, dts_90k));
        } else {
            pes_hdr.extend_from_slice(&encode_pes_timestamp(0x02, pts_90k));
        }

        // Packetize `pes_hdr + es_payload` into 188-byte TS packets
        let mut first_ts = true;
        let mut es_offset = 0usize;

        while first_ts || es_offset < es_payload.len() {
            let mut ts_pkt = [0xFFu8; TS_PACKET_SIZE];
            let cc = self.cc_tracks[t_idx] & 0x0F;
            self.cc_tracks[t_idx] = self.cc_tracks[t_idx].wrapping_add(1);

            let pusi_bit = if first_ts { 0x40u8 } else { 0x00u8 };
            ts_pkt[0] = TS_SYNC_BYTE;
            ts_pkt[1] = pusi_bit | (((pid >> 8) & 0x1F) as u8);
            ts_pkt[2] = (pid & 0xFF) as u8;

            // Include 6-byte PCR in adaptation field on the first TS packet of a keyframe
            let write_pcr = first_ts && packet.is_keyframe();
            let hdr_prefix = if first_ts { &pes_hdr[..] } else { &[] };
            let rem_es = es_payload.len() - es_offset;
            let total_rem = hdr_prefix.len() + rem_es;

            let mut payload_start = 4usize;
            if write_pcr {
                // Adaptation field with 6-byte PCR (minimum 8 bytes including adapt_len & flags)
                let max_payload_with_pcr = TS_PACKET_SIZE - 4 - 8;
                let actual_payload = total_rem.min(max_payload_with_pcr);
                let stuff_len = max_payload_with_pcr - actual_payload;
                let adapt_len = 7 + stuff_len;
                ts_pkt[3] = 0x30 | cc; // adaptation + payload
                ts_pkt[4] = adapt_len as u8;
                ts_pkt[5] = if packet.is_keyframe() { 0x50 } else { 0x10 }; // RAI + PCR_flag
                // 33-bit PCR base (90 kHz) + 6 reserved '1' bits + 9-bit PCR ext (0)
                let pcr_base = (dts_90k as u64) & 0x1_FFFF_FFFF;
                ts_pkt[6] = ((pcr_base >> 25) & 0xFF) as u8;
                ts_pkt[7] = ((pcr_base >> 17) & 0xFF) as u8;
                ts_pkt[8] = ((pcr_base >> 9) & 0xFF) as u8;
                ts_pkt[9] = ((pcr_base >> 1) & 0xFF) as u8;
                ts_pkt[10] = (((pcr_base & 1) as u8) << 7) | 0x7E;
                ts_pkt[11] = 0x00;
                payload_start = 5 + adapt_len;
            } else if total_rem < TS_PACKET_SIZE - 4 {
                let stuff_total = (TS_PACKET_SIZE - 4) - total_rem;
                ts_pkt[3] = 0x30 | cc; // adaptation + payload
                if stuff_total == 1 {
                    ts_pkt[4] = 0; // adaptation_field_length = 0 (1 byte total)
                } else {
                    ts_pkt[4] = (stuff_total - 1) as u8;
                    ts_pkt[5] = 0x00; // flags = 0, remaining bytes already 0xFF
                }
                payload_start = 4 + stuff_total;
            } else {
                ts_pkt[3] = 0x10 | cc; // payload only
            }

            let mut dst_pos = payload_start;
            if !hdr_prefix.is_empty() {
                ts_pkt[dst_pos..dst_pos + hdr_prefix.len()].copy_from_slice(hdr_prefix);
                dst_pos += hdr_prefix.len();
            }
            let es_chunk = TS_PACKET_SIZE - dst_pos;
            ts_pkt[dst_pos..TS_PACKET_SIZE]
                .copy_from_slice(&es_payload[es_offset..es_offset + es_chunk]);
            es_offset += es_chunk;
            first_ts = false;

            self.writer.write_all_counted(&ts_pkt)?;
        }

        Ok(())
    }

    pub fn finish(mut self) -> Result<u64> {
        self.writer.flush()?;
        Ok(self.writer.bytes_written())
    }

    fn write_pat_packet(&mut self) -> Result<()> {
        let mut pkt = [0xFFu8; TS_PACKET_SIZE];
        let cc = self.cc_pat & 0x0F;
        self.cc_pat = self.cc_pat.wrapping_add(1);

        pkt[0] = TS_SYNC_BYTE;
        pkt[1] = 0x40; // PUSI = 1, PID = 0x0000
        pkt[2] = 0x00;
        pkt[3] = 0x10 | cc;
        pkt[4] = 0x00; // pointer_field = 0

        let sec = &mut pkt[5..17];
        sec[0] = 0x00; // table_id = 0x00 (PAT)
        sec[1] = 0xB0; // section_syntax_indicator = 1
        sec[2] = 13; // section_length = 13 bytes
        sec[3] = 0x00;
        sec[4] = 0x01; // transport_stream_id = 1
        sec[5] = 0xC1; // version 0, current_next_indicator = 1
        sec[6] = 0x00; // section_number = 0
        sec[7] = 0x00; // last_section_number = 0
        sec[8] = 0x00;
        sec[9] = 0x01; // program_number = 1
        let pmt_word = 0xE000u16 | PID_DEFAULT_PMT;
        sec[10..12].copy_from_slice(&pmt_word.to_be_bytes());

        let crc = crc32_mpeg2(&pkt[5..17]);
        pkt[17..21].copy_from_slice(&crc.to_be_bytes());

        self.writer.write_all_counted(&pkt)
    }

    fn write_pmt_packet(&mut self) -> Result<()> {
        let mut pkt = [0xFFu8; TS_PACKET_SIZE];
        let cc = self.cc_pmt & 0x0F;
        self.cc_pmt = self.cc_pmt.wrapping_add(1);

        pkt[0] = TS_SYNC_BYTE;
        pkt[1] = 0x40 | (((PID_DEFAULT_PMT >> 8) & 0x1F) as u8);
        pkt[2] = (PID_DEFAULT_PMT & 0xFF) as u8;
        pkt[3] = 0x10 | cc;
        pkt[4] = 0x00; // pointer_field = 0

        let es_loop_len = self.tracks.len() * 5;
        let sec_len = 9 + es_loop_len + 4;

        let mut pos = 5usize;
        pkt[pos] = 0x02; // table_id = 0x02 (PMT)
        pkt[pos + 1] = 0xB0 | (((sec_len >> 8) & 0x0F) as u8);
        pkt[pos + 2] = (sec_len & 0xFF) as u8;
        pkt[pos + 3] = 0x00;
        pkt[pos + 4] = 0x01; // program_number = 1
        pkt[pos + 5] = 0xC1; // version 0, current_next = 1
        pkt[pos + 6] = 0x00;
        pkt[pos + 7] = 0x00;
        let pcr_word = 0xE000u16 | PID_BASE_ES;
        pkt[pos + 8..pos + 10].copy_from_slice(&pcr_word.to_be_bytes());
        pkt[pos + 10] = 0xF0;
        pkt[pos + 11] = 0x00; // program_info_length = 0
        pos += 12;

        for (i, t) in self.tracks.iter().enumerate() {
            let stream_type = match t.codec_id {
                CodecId::H264 => 0x1Bu8,
                CodecId::Hevc => 0x24u8,
                CodecId::Aac => 0x0Fu8,
                _ => 0x06u8,
            };
            let es_pid = PID_BASE_ES + (i as u16);
            pkt[pos] = stream_type;
            let pid_word = 0xE000u16 | es_pid;
            pkt[pos + 1..pos + 3].copy_from_slice(&pid_word.to_be_bytes());
            pkt[pos + 3] = 0xF0;
            pkt[pos + 4] = 0x00; // ES_info_length = 0
            pos += 5;
        }

        let crc = crc32_mpeg2(&pkt[5..pos]);
        pkt[pos..pos + 4].copy_from_slice(&crc.to_be_bytes());

        self.writer.write_all_counted(&pkt)
    }
}
