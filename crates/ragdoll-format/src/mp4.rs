use crate::io::VectoredWriter;
use ragdoll_core::{
    AudioTrackParams, ChannelLayout, CodecId, ColorPrimaries, ColorRange, Colorimetry,
    MatrixCoefficients, PacketFlags, PacketSlice, PixelFormat, RagdollError, Rational64, Result,
    SampleEntry, SampleFormat, Timestamp, TrackInfo, TrackType, TransferCharacteristics,
    VideoTrackParams, rescale_q,
};
use std::io::Write;

/// Four-character ISOBMFF box type (`FourCC`).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FourCC(pub [u8; 4]);

impl FourCC {
    #[inline]
    pub const fn new(bytes: &[u8; 4]) -> Self {
        Self(*bytes)
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.0).unwrap_or("????")
    }
}

impl std::fmt::Debug for FourCC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FourCC(\"{}\")", self.as_str())
    }
}

impl std::fmt::Display for FourCC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Parsed ISOBMFF box view borrowing directly from the memory-mapped slice.
#[derive(Clone, Copy, Debug)]
struct BoxHeader<'a> {
    fourcc: FourCC,
    total_len: usize,
    payload: &'a [u8],
}

/// Iterates over sibling ISOBMFF boxes in a byte slice.
struct BoxIter<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> BoxIter<'a> {
    #[inline]
    const fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }
}

impl<'a> Iterator for BoxIter<'a> {
    type Item = Result<BoxHeader<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        let rem = &self.data[self.offset..];
        if rem.len() < 8 {
            self.offset = self.data.len();
            return None;
        }
        let size32 = u32::from_be_bytes([rem[0], rem[1], rem[2], rem[3]]) as usize;
        let fourcc = FourCC([rem[4], rem[5], rem[6], rem[7]]);

        let (header_len, total_len) = if size32 == 1 {
            if rem.len() < 16 {
                self.offset = self.data.len();
                return Some(Err(RagdollError::CorruptBitstream(
                    "truncated 64-bit ISOBMFF box header",
                )));
            }
            let size64 = u64::from_be_bytes([
                rem[8], rem[9], rem[10], rem[11], rem[12], rem[13], rem[14], rem[15],
            ]) as usize;
            (16usize, size64)
        } else if size32 == 0 {
            (8usize, rem.len())
        } else {
            (8usize, size32)
        };

        if total_len < header_len || total_len > rem.len() {
            self.offset = self.data.len();
            return Some(Err(RagdollError::CorruptBitstream(
                "ISOBMFF box size exceeds parent bounds",
            )));
        }

        let payload = &rem[header_len..total_len];
        self.offset += total_len;
        Some(Ok(BoxHeader {
            fourcc,
            total_len,
            payload,
        }))
    }
}

#[derive(Default)]
struct RawStblTables {
    stts: Vec<(u32, u32)>,
    ctts: Vec<(u32, i32)>,
    stsc: Vec<(u32, u32, u32)>,
    stsz_default: u32,
    stsz_count: usize,
    stsz_entries: Vec<u32>,
    chunk_offsets: Vec<u64>,
    stss: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Default)]
struct TrexDefaults {
    track_id: u32,
    default_duration: u32,
    default_size: u32,
    default_flags: u32,
}

/// Per-track compiled sample index with 24-byte cache-packed [`SampleEntry`] items.
#[derive(Clone, Debug)]
pub struct Mp4TrackIndex {
    /// ISOBMFF `tkhd` 1-based track ID.
    pub track_id: u32,
    pub info: TrackInfo,
    /// Flat 24-byte-per-sample index (2.66 entries per 64-byte CPU cache line).
    pub samples: Vec<SampleEntry>,
    /// Default or average sample duration in track timebase ticks.
    pub default_sample_duration: i64,
}

impl Mp4TrackIndex {
    /// Performs $O(\log N)$ binary search over the flat 24-byte [`SampleEntry`] table to locate
    /// the keyframe (`is_keyframe() == true`) at or immediately preceding `target_pts`.
    ///
    /// Returns the `0`-based sample index within `self.samples`.
    pub fn seek_keyframe_index(&self, target_pts: i64) -> Option<usize> {
        if self.samples.is_empty() {
            return None;
        }
        // Binary search in O(log N) for the rightmost sample with pts <= target_pts
        let upper = self.samples.partition_point(|s| s.pts <= target_pts);
        let start_idx = if upper == 0 { 0 } else { upper - 1 };

        // Walk backward to the nearest sync/IDR keyframe
        for idx in (0..=start_idx).rev() {
            if self.samples[idx].is_keyframe() {
                return Some(idx);
            }
        }
        // If no earlier keyframe exists before start_idx, find the first keyframe in the track
        self.samples.iter().position(|s| s.is_keyframe())
    }

    /// Computes sample duration in track timebase ticks for sample `sample_idx`.
    #[inline]
    pub fn sample_duration(&self, sample_idx: usize) -> i64 {
        if sample_idx + 1 < self.samples.len() {
            let d = self.samples[sample_idx + 1].dts() - self.samples[sample_idx].dts();
            if d > 0 {
                return d;
            }
        }
        self.default_sample_duration.max(1)
    }
}

#[derive(Clone, Copy, Debug)]
struct PacketRef {
    track_pos: usize,
    sample_idx: usize,
    sort_dts_ns: i64,
    file_offset: u64,
}

/// Zero-copy ISOBMFF / MP4 / MOV demuxer operating over a memory-mapped slice `&'a [u8]`.
///
/// Compiles fragmented `stbl` tables (`stts`, `ctts`, `stsc`, `stsz`, `stco`/`co64`, `stss`)
/// into flat 24-byte [`SampleEntry`] vectors and emits [`PacketSlice<'a>`] views directly into
/// `mdat` with **zero heap allocations per packet**.
pub struct Mp4Demuxer<'a> {
    data: &'a [u8],
    major_brand: FourCC,
    tracks: Vec<Mp4TrackIndex>,
    schedule: Vec<PacketRef>,
    cursor: usize,
    moov_before_mdat: bool,
    fragmented: bool,
}

impl<'a> Mp4Demuxer<'a> {
    /// Parses an MP4 / ISOBMFF (including Fragmented `fMP4` `moof`+`mdat`) memory-mapped byte slice
    /// and compiles flat 24-byte sample indices.
    pub fn open(data: &'a [u8]) -> Result<Self> {
        let mut major_brand = FourCC(*b"isom");
        let mut tracks = Vec::new();
        let mut trex_list = Vec::new();
        let mut moov_offset: Option<usize> = None;
        let mut mdat_offset: Option<usize> = None;
        let mut fragmented = false;

        let mut iter = BoxIter::new(data);
        while let Some(box_res) = iter.next() {
            let b = box_res?;
            let box_start = iter.offset - b.total_len;
            match &b.fourcc.0 {
                b"ftyp" if b.payload.len() >= 4 => {
                    major_brand = FourCC([b.payload[0], b.payload[1], b.payload[2], b.payload[3]]);
                }
                b"moov" => {
                    moov_offset = Some(box_start);
                    Self::parse_moov(b.payload, data.len(), &mut tracks, &mut trex_list)?;
                }
                b"moof" => {
                    fragmented = true;
                    Self::parse_moof(
                        b.payload,
                        box_start as u64,
                        data.len(),
                        &mut tracks,
                        &trex_list,
                    )?;
                }
                b"mdat" if mdat_offset.is_none() => {
                    mdat_offset = Some(box_start);
                }
                _ => {}
            }
        }

        if tracks.is_empty() && moov_offset.is_none() {
            return Err(RagdollError::InvalidHeader(
                "missing 'moov' box in MP4/ISOBMFF container".to_string(),
            ));
        }

        let moov_before_mdat = match (moov_offset, mdat_offset) {
            (Some(m), Some(d)) => m < d,
            _ => false,
        };

        // Build global interleaved packet schedule across all tracks
        let total_packets: usize = tracks.iter().map(|t| t.samples.len()).sum();
        let mut schedule = Vec::with_capacity(total_packets);
        for (track_pos, track) in tracks.iter().enumerate() {
            let tb = track.info.timebase;
            for (sample_idx, sample) in track.samples.iter().enumerate() {
                let sort_dts_ns = rescale_q(sample.dts(), tb, Rational64::TIMEBASE_NANOS);
                schedule.push(PacketRef {
                    track_pos,
                    sample_idx,
                    sort_dts_ns,
                    file_offset: sample.offset,
                });
            }
        }
        schedule.sort_by(|a, b| {
            a.sort_dts_ns
                .cmp(&b.sort_dts_ns)
                .then_with(|| a.file_offset.cmp(&b.file_offset))
        });

        Ok(Self {
            data,
            major_brand,
            tracks,
            schedule,
            cursor: 0,
            moov_before_mdat,
            fragmented,
        })
    }

    /// Major brand from the `ftyp` box (e.g., `isom`, `mp42`, `iso5`).
    #[inline]
    pub const fn major_brand(&self) -> FourCC {
        self.major_brand
    }

    /// Returns `true` if the `moov` header appears before `mdat` (web fast-start).
    #[inline]
    pub const fn is_faststart(&self) -> bool {
        self.moov_before_mdat
    }

    /// Returns `true` if the file contains `moof` (`MovieFragmentBox`) streaming fragments.
    #[inline]
    pub const fn is_fragmented(&self) -> bool {
        self.fragmented
    }

    /// Returns the compiled track indices (`TrackInfo` + flat `Vec<SampleEntry>`).
    #[inline]
    pub fn tracks(&self) -> &[Mp4TrackIndex] {
        &self.tracks
    }

    /// Total number of packets across all tracks.
    #[inline]
    pub fn total_packets(&self) -> usize {
        self.schedule.len()
    }

    /// Emits the next interleaved packet as a zero-copy [`PacketSlice<'a>`] borrowing directly from `data`.
    #[inline]
    pub fn next_packet(&mut self) -> Result<Option<PacketSlice<'a>>> {
        while self.cursor < self.schedule.len() {
            let pref = self.schedule[self.cursor];
            self.cursor += 1;
            let pkt = self.packet_at(pref.track_pos, pref.sample_idx)?;
            // Skip ISO/IEC 14496-30 `vtte` EmptyCueBox gap fillers on subtitle tracks
            if self.tracks[pref.track_pos].info.track_type == TrackType::Subtitle
                && pkt.data().len() >= 8
                && &pkt.data()[4..8] == b"vtte"
            {
                continue;
            }
            return Ok(Some(pkt));
        }
        Ok(None)
    }

    /// Fetches a specific `(track_pos, sample_idx)` packet in $O(1)$ without moving the sequential cursor.
    #[inline]
    pub fn packet_at(&self, track_pos: usize, sample_idx: usize) -> Result<PacketSlice<'a>> {
        let track = self
            .tracks
            .get(track_pos)
            .ok_or(RagdollError::CorruptBitstream("invalid track index"))?;
        let entry = track
            .samples
            .get(sample_idx)
            .ok_or(RagdollError::CorruptBitstream("invalid sample index"))?;

        let start = entry.offset as usize;
        let end = start
            .checked_add(entry.size as usize)
            .ok_or(RagdollError::CorruptBitstream("sample offset overflow"))?;
        if end > self.data.len() {
            return Err(RagdollError::CorruptBitstream(
                "MP4 sample offset/size out of bounds of file",
            ));
        }

        let payload = &self.data[start..end];
        let tb = track.info.timebase;
        let pts = Timestamp::new(entry.pts, tb);
        let dts = Timestamp::new(entry.dts(), tb);
        let duration = track.sample_duration(sample_idx);
        let flags = if entry.is_keyframe() {
            PacketFlags::KEYFRAME
        } else {
            PacketFlags::NONE
        };

        Ok(PacketSlice::from_mmap(
            payload,
            track.info.index,
            pts,
            dts,
            duration,
            flags,
            entry.offset,
        ))
    }

    /// Seeks to the nearest keyframe at or before `target` on `track_pos` in $O(\log N)$ time,
    /// repositioning the demuxer's sequential cursor and returning the `(sample_idx, SampleEntry)`.
    pub fn seek_to_timestamp(
        &mut self,
        track_pos: usize,
        target: Timestamp,
    ) -> Result<(usize, SampleEntry)> {
        let track = self
            .tracks
            .get(track_pos)
            .ok_or(RagdollError::CorruptBitstream("invalid track index"))?;
        let target_ticks = target.rescale_to(track.info.timebase).ticks;
        let kf_sample_idx = track
            .seek_keyframe_index(target_ticks)
            .ok_or(RagdollError::CorruptBitstream("track contains no samples"))?;
        let kf_entry = track.samples[kf_sample_idx];

        // Reposition the global schedule cursor via binary search on sort_dts_ns
        let kf_dts_ns = rescale_q(
            kf_entry.dts(),
            track.info.timebase,
            Rational64::TIMEBASE_NANOS,
        );
        let sched_start = self.schedule.partition_point(|p| p.sort_dts_ns < kf_dts_ns);
        // Find exact match for (track_pos, kf_sample_idx) near sched_start
        self.cursor = self.schedule[sched_start..]
            .iter()
            .position(|p| p.track_pos == track_pos && p.sample_idx == kf_sample_idx)
            .map(|rel| sched_start + rel)
            .unwrap_or(sched_start);

        Ok((kf_sample_idx, kf_entry))
    }

    fn parse_moov(
        moov_data: &'a [u8],
        file_len: usize,
        tracks: &mut Vec<Mp4TrackIndex>,
        trex_list: &mut Vec<TrexDefaults>,
    ) -> Result<()> {
        let mut movie_timescale = 1000u32;
        for box_res in BoxIter::new(moov_data) {
            let b = box_res?;
            match &b.fourcc.0 {
                b"mvhd" if b.payload.len() >= 20 => {
                    let version = b.payload[0];
                    if version == 0 && b.payload.len() >= 16 {
                        let ts = u32::from_be_bytes([
                            b.payload[12],
                            b.payload[13],
                            b.payload[14],
                            b.payload[15],
                        ]);
                        if ts > 0 {
                            movie_timescale = ts;
                        }
                    } else if version == 1 && b.payload.len() >= 24 {
                        let ts = u32::from_be_bytes([
                            b.payload[20],
                            b.payload[21],
                            b.payload[22],
                            b.payload[23],
                        ]);
                        if ts > 0 {
                            movie_timescale = ts;
                        }
                    }
                }
                b"trak" => {
                    let track_index = tracks.len() as u32;
                    if let Some(track) =
                        Self::parse_trak(b.payload, track_index, movie_timescale, file_len)?
                    {
                        tracks.push(track);
                    }
                }
                b"mvex" => {
                    for mvex_res in BoxIter::new(b.payload) {
                        let mb = mvex_res?;
                        if &mb.fourcc.0 == b"trex" && mb.payload.len() >= 24 {
                            let p = mb.payload;
                            let track_id = u32::from_be_bytes([p[4], p[5], p[6], p[7]]);
                            let default_duration = u32::from_be_bytes([p[12], p[13], p[14], p[15]]);
                            let default_size = u32::from_be_bytes([p[16], p[17], p[18], p[19]]);
                            let default_flags = u32::from_be_bytes([p[20], p[21], p[22], p[23]]);
                            trex_list.push(TrexDefaults {
                                track_id,
                                default_duration,
                                default_size,
                                default_flags,
                            });
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Parses an ISO/IEC 14496-12 `moof` (`MovieFragmentBox`) and appends fragment samples
    /// (`traf` -> `tfhd`, `tfdt`, `trun`) directly into the matching track's flat `SampleEntry` table.
    fn parse_moof(
        moof_data: &'a [u8],
        moof_offset: u64,
        file_len: usize,
        tracks: &mut [Mp4TrackIndex],
        trex_list: &[TrexDefaults],
    ) -> Result<()> {
        let mut prev_data_end = moof_offset;

        for box_res in BoxIter::new(moof_data) {
            let b = box_res?;
            if &b.fourcc.0 != b"traf" {
                continue;
            }

            let mut track_id = 1u32;
            let mut tfhd_flags = 0u32;
            let mut base_data_offset: Option<u64> = None;
            let mut default_duration: Option<u32> = None;
            let mut default_size: Option<u32> = None;
            let mut default_flags: Option<u32> = None;
            let mut tfdt_base_dts: Option<i64> = None;

            // Pass 1: parse `tfhd` and `tfdt` inside `traf`
            for traf_res in BoxIter::new(b.payload) {
                let tb = traf_res?;
                match &tb.fourcc.0 {
                    b"tfhd" if tb.payload.len() >= 8 => {
                        let p = tb.payload;
                        tfhd_flags = u32::from_be_bytes([0, p[1], p[2], p[3]]);
                        track_id = u32::from_be_bytes([p[4], p[5], p[6], p[7]]);
                        let mut pos = 8usize;
                        if (tfhd_flags & 0x0000_0001) != 0 && pos + 8 <= p.len() {
                            base_data_offset = Some(u64::from_be_bytes([
                                p[pos],
                                p[pos + 1],
                                p[pos + 2],
                                p[pos + 3],
                                p[pos + 4],
                                p[pos + 5],
                                p[pos + 6],
                                p[pos + 7],
                            ]));
                            pos += 8;
                        }
                        if (tfhd_flags & 0x0000_0002) != 0 && pos + 4 <= p.len() {
                            pos += 4; // sample_description_index
                        }
                        if (tfhd_flags & 0x0000_0008) != 0 && pos + 4 <= p.len() {
                            default_duration = Some(u32::from_be_bytes([
                                p[pos],
                                p[pos + 1],
                                p[pos + 2],
                                p[pos + 3],
                            ]));
                            pos += 4;
                        }
                        if (tfhd_flags & 0x0000_0010) != 0 && pos + 4 <= p.len() {
                            default_size = Some(u32::from_be_bytes([
                                p[pos],
                                p[pos + 1],
                                p[pos + 2],
                                p[pos + 3],
                            ]));
                            pos += 4;
                        }
                        if (tfhd_flags & 0x0000_0020) != 0 && pos + 4 <= p.len() {
                            default_flags = Some(u32::from_be_bytes([
                                p[pos],
                                p[pos + 1],
                                p[pos + 2],
                                p[pos + 3],
                            ]));
                        }
                    }
                    b"tfdt" if tb.payload.len() >= 8 => {
                        let p = tb.payload;
                        let version = p[0];
                        if version == 1 && p.len() >= 12 {
                            tfdt_base_dts = Some(u64::from_be_bytes([
                                p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11],
                            ]) as i64);
                        } else {
                            tfdt_base_dts =
                                Some(u32::from_be_bytes([p[4], p[5], p[6], p[7]]) as i64);
                        }
                    }
                    _ => {}
                }
            }

            // Locate target track (matching `track_id`, or fallback to first track if only 1 exists)
            let track_pos = tracks
                .iter()
                .position(|t| t.track_id == track_id)
                .or_else(|| (tracks.len() == 1).then_some(0));
            let Some(t_pos) = track_pos else {
                continue;
            };

            let trex = trex_list
                .iter()
                .find(|t| t.track_id == track_id)
                .copied()
                .unwrap_or(TrexDefaults {
                    track_id,
                    default_duration: 1,
                    default_size: 0,
                    default_flags: 0,
                });

            let eff_default_dur = default_duration.unwrap_or(trex.default_duration).max(1);
            let eff_default_size = default_size.unwrap_or(trex.default_size);
            let eff_default_flags = default_flags.unwrap_or(trex.default_flags);

            // Base data offset per ISO 14496-12 Section 8.8.7 (`default-base-is-moof` = 0x020000)
            let traf_base_offset = if let Some(bdo) = base_data_offset {
                bdo
            } else if (tfhd_flags & 0x0002_0000) != 0 {
                moof_offset
            } else {
                prev_data_end
            };

            let track = &mut tracks[t_pos];
            if track.default_sample_duration <= 1 && eff_default_dur > 1 {
                track.default_sample_duration = eff_default_dur as i64;
            }
            let mut cur_dts = tfdt_base_dts.unwrap_or_else(|| {
                track
                    .samples
                    .last()
                    .map(|s| s.dts() + eff_default_dur as i64)
                    .unwrap_or(0)
            });
            let mut cur_data_offset = traf_base_offset;

            // Pass 2: parse all `trun` boxes inside `traf`
            for traf_res in BoxIter::new(b.payload) {
                let tb = traf_res?;
                if &tb.fourcc.0 != b"trun" || tb.payload.len() < 8 {
                    continue;
                }
                let p = tb.payload;
                let trun_version = p[0];
                let trun_flags = u32::from_be_bytes([0, p[1], p[2], p[3]]);
                let sample_count = u32::from_be_bytes([p[4], p[5], p[6], p[7]]) as usize;
                if sample_count > file_len.max(65536) {
                    return Err(RagdollError::CorruptBitstream(
                        "fMP4 trun sample_count exceeds file bounds",
                    ));
                }
                let mut pos = 8usize;

                if (trun_flags & 0x0000_0001) != 0 && pos + 4 <= p.len() {
                    let data_offset =
                        i32::from_be_bytes([p[pos], p[pos + 1], p[pos + 2], p[pos + 3]]);
                    cur_data_offset = (traf_base_offset as i64 + data_offset as i64).max(0) as u64;
                    pos += 4;
                }
                let mut first_sample_flags = None;
                if (trun_flags & 0x0000_0004) != 0 && pos + 4 <= p.len() {
                    first_sample_flags = Some(u32::from_be_bytes([
                        p[pos],
                        p[pos + 1],
                        p[pos + 2],
                        p[pos + 3],
                    ]));
                    pos += 4;
                }

                track.samples.reserve(sample_count.min(65536));
                for i in 0..sample_count {
                    let s_dur = if (trun_flags & 0x0000_0100) != 0 && pos + 4 <= p.len() {
                        let d = u32::from_be_bytes([p[pos], p[pos + 1], p[pos + 2], p[pos + 3]]);
                        pos += 4;
                        d
                    } else {
                        eff_default_dur
                    };

                    let s_size = if (trun_flags & 0x0000_0200) != 0 && pos + 4 <= p.len() {
                        let sz = u32::from_be_bytes([p[pos], p[pos + 1], p[pos + 2], p[pos + 3]]);
                        pos += 4;
                        sz
                    } else {
                        eff_default_size
                    };

                    let s_flags = if (trun_flags & 0x0000_0400) != 0 && pos + 4 <= p.len() {
                        let fl = u32::from_be_bytes([p[pos], p[pos + 1], p[pos + 2], p[pos + 3]]);
                        pos += 4;
                        fl
                    } else if i == 0
                        && let Some(fsf) = first_sample_flags
                    {
                        fsf
                    } else {
                        eff_default_flags
                    };

                    let s_cts_offset = if (trun_flags & 0x0000_0800) != 0 && pos + 4 <= p.len() {
                        let raw = [p[pos], p[pos + 1], p[pos + 2], p[pos + 3]];
                        pos += 4;
                        if trun_version == 0 {
                            u32::from_be_bytes(raw) as i64 as i32
                        } else {
                            i32::from_be_bytes(raw)
                        }
                    } else {
                        0
                    };

                    if cur_data_offset.saturating_add(s_size as u64) > file_len as u64 {
                        return Err(RagdollError::CorruptBitstream(
                            "fMP4 trun sample offset/size exceeds file bounds",
                        ));
                    }

                    // ISO 14496-12 sample_flags:
                    // bit 16 (0x0001_0000): sample_is_non_sync_sample
                    // bits 24..25 (0x0300_0000): sample_depends_on (1 = depends on others / non-I)
                    let is_keyframe = (s_flags & (0x0001_0000 | 0x0100_0000)) == 0;
                    let pts = cur_dts + s_cts_offset as i64;
                    track.samples.push(SampleEntry::new(
                        cur_data_offset,
                        pts,
                        s_size,
                        s_cts_offset,
                        is_keyframe,
                    ));
                    track.info.total_bytes += s_size as u64;
                    cur_data_offset += s_size as u64;
                    cur_dts += s_dur.max(1) as i64;
                }
            }

            prev_data_end = cur_data_offset;
            track.info.sample_count = track.samples.len();
            track.info.duration_ticks = Some(cur_dts);
            if track.info.track_type == TrackType::Video
                && !track.samples.is_empty()
                && cur_dts > 0
                && let Some(vp) = &mut track.info.video
            {
                vp.frame_rate = Rational64::new(
                    track.samples.len() as i64 * track.info.timebase.den.max(1),
                    cur_dts,
                );
            }
        }

        Ok(())
    }

    fn parse_trak(
        trak_data: &'a [u8],
        track_index: u32,
        movie_timescale: u32,
        file_len: usize,
    ) -> Result<Option<Mp4TrackIndex>> {
        let mut tkhd_track_id = track_index + 1;
        let mut tkhd_width = 0u32;
        let mut tkhd_height = 0u32;
        let mut elst_media_time = 0i64;
        let mut mdia_slice: Option<&'a [u8]> = None;

        for box_res in BoxIter::new(trak_data) {
            let b = box_res?;
            match &b.fourcc.0 {
                b"tkhd" => {
                    let p = b.payload;
                    if !p.is_empty() {
                        let version = p[0];
                        let id_offset = if version == 1 { 20 } else { 12 };
                        if p.len() >= id_offset + 4 {
                            let tid = u32::from_be_bytes([
                                p[id_offset],
                                p[id_offset + 1],
                                p[id_offset + 2],
                                p[id_offset + 3],
                            ]);
                            if tid > 0 {
                                tkhd_track_id = tid;
                            }
                        }
                        let wh_offset = if version == 1 { 88 } else { 76 };
                        if p.len() >= wh_offset + 8 {
                            tkhd_width = u32::from_be_bytes([
                                p[wh_offset],
                                p[wh_offset + 1],
                                p[wh_offset + 2],
                                p[wh_offset + 3],
                            ]) >> 16;
                            tkhd_height = u32::from_be_bytes([
                                p[wh_offset + 4],
                                p[wh_offset + 5],
                                p[wh_offset + 6],
                                p[wh_offset + 7],
                            ]) >> 16;
                        }
                    }
                }
                b"edts" => {
                    for edts_box in BoxIter::new(b.payload) {
                        let eb = edts_box?;
                        if &eb.fourcc.0 == b"elst" && eb.payload.len() >= 8 {
                            let version = eb.payload[0];
                            let entry_count = u32::from_be_bytes([
                                eb.payload[4],
                                eb.payload[5],
                                eb.payload[6],
                                eb.payload[7],
                            ]) as usize;
                            let mut pos = 8usize;
                            for _ in 0..entry_count {
                                if version == 1 && pos + 20 <= eb.payload.len() {
                                    let media_time = i64::from_be_bytes([
                                        eb.payload[pos + 8],
                                        eb.payload[pos + 9],
                                        eb.payload[pos + 10],
                                        eb.payload[pos + 11],
                                        eb.payload[pos + 12],
                                        eb.payload[pos + 13],
                                        eb.payload[pos + 14],
                                        eb.payload[pos + 15],
                                    ]);
                                    if media_time > 0 {
                                        elst_media_time = media_time;
                                        break;
                                    }
                                    pos += 20;
                                } else if version == 0 && pos + 12 <= eb.payload.len() {
                                    let media_time = i32::from_be_bytes([
                                        eb.payload[pos + 4],
                                        eb.payload[pos + 5],
                                        eb.payload[pos + 6],
                                        eb.payload[pos + 7],
                                    ]) as i64;
                                    if media_time > 0 {
                                        elst_media_time = media_time;
                                        break;
                                    }
                                    pos += 12;
                                } else {
                                    break;
                                }
                            }
                        }
                    }
                }
                b"mdia" => {
                    mdia_slice = Some(b.payload);
                }
                _ => {}
            }
        }

        let Some(mdia) = mdia_slice else {
            return Ok(None);
        };

        let mut timescale = movie_timescale.max(1);
        let mut duration_ticks: Option<i64> = None;
        let mut track_type = TrackType::Data;
        let mut minf_slice: Option<&'a [u8]> = None;

        for box_res in BoxIter::new(mdia) {
            let b = box_res?;
            match &b.fourcc.0 {
                b"mdhd" => {
                    let p = b.payload;
                    if !p.is_empty() {
                        let version = p[0];
                        if version == 0 && p.len() >= 20 {
                            let ts = u32::from_be_bytes([p[12], p[13], p[14], p[15]]);
                            let dur = u32::from_be_bytes([p[16], p[17], p[18], p[19]]);
                            if ts > 0 {
                                timescale = ts;
                            }
                            if dur != u32::MAX {
                                duration_ticks = Some(dur as i64);
                            }
                        } else if version == 1 && p.len() >= 32 {
                            let ts = u32::from_be_bytes([p[20], p[21], p[22], p[23]]);
                            let dur = u64::from_be_bytes([
                                p[24], p[25], p[26], p[27], p[28], p[29], p[30], p[31],
                            ]);
                            if ts > 0 {
                                timescale = ts;
                            }
                            if dur != u64::MAX {
                                duration_ticks = Some(dur as i64);
                            }
                        }
                    }
                }
                b"hdlr" if b.payload.len() >= 12 => {
                    track_type = match &b.payload[8..12] {
                        b"vide" => TrackType::Video,
                        b"soun" => TrackType::Audio,
                        b"subt" | b"sbtl" | b"text" => TrackType::Subtitle,
                        _ => TrackType::Data,
                    };
                }
                b"minf" => {
                    minf_slice = Some(b.payload);
                }
                _ => {}
            }
        }

        if track_type == TrackType::Data {
            return Ok(None);
        }

        let Some(minf) = minf_slice else {
            return Ok(None);
        };

        let mut stbl_slice: Option<&'a [u8]> = None;
        for box_res in BoxIter::new(minf) {
            let b = box_res?;
            if &b.fourcc.0 == b"stbl" {
                stbl_slice = Some(b.payload);
                break;
            }
        }

        let Some(stbl) = stbl_slice else {
            return Ok(None);
        };

        let mut codec_id: Option<CodecId> = None;
        let mut extra_data = Vec::new();
        let mut video_params = VideoTrackParams {
            width: tkhd_width,
            height: tkhd_height,
            ..Default::default()
        };
        let mut audio_params = AudioTrackParams::default();
        let mut tables = RawStblTables::default();

        for box_res in BoxIter::new(stbl) {
            let b = box_res?;
            match &b.fourcc.0 {
                b"stsd" => {
                    Self::parse_stsd(
                        b.payload,
                        track_type,
                        &mut codec_id,
                        &mut extra_data,
                        &mut video_params,
                        &mut audio_params,
                    )?;
                }
                b"stts" if b.payload.len() >= 8 => {
                    let count = u32::from_be_bytes([
                        b.payload[4],
                        b.payload[5],
                        b.payload[6],
                        b.payload[7],
                    ]) as usize;
                    let mut pos = 8usize;
                    tables.stts.reserve(count.min(65536));
                    for _ in 0..count {
                        if pos + 8 > b.payload.len() {
                            break;
                        }
                        let sc = u32::from_be_bytes([
                            b.payload[pos],
                            b.payload[pos + 1],
                            b.payload[pos + 2],
                            b.payload[pos + 3],
                        ]);
                        let sd = u32::from_be_bytes([
                            b.payload[pos + 4],
                            b.payload[pos + 5],
                            b.payload[pos + 6],
                            b.payload[pos + 7],
                        ]);
                        tables.stts.push((sc, sd));
                        pos += 8;
                    }
                }
                b"ctts" if b.payload.len() >= 8 => {
                    let count = u32::from_be_bytes([
                        b.payload[4],
                        b.payload[5],
                        b.payload[6],
                        b.payload[7],
                    ]) as usize;
                    let mut pos = 8usize;
                    tables.ctts.reserve(count.min(65536));
                    for _ in 0..count {
                        if pos + 8 > b.payload.len() {
                            break;
                        }
                        let sc = u32::from_be_bytes([
                            b.payload[pos],
                            b.payload[pos + 1],
                            b.payload[pos + 2],
                            b.payload[pos + 3],
                        ]);
                        let so = i32::from_be_bytes([
                            b.payload[pos + 4],
                            b.payload[pos + 5],
                            b.payload[pos + 6],
                            b.payload[pos + 7],
                        ]);
                        tables.ctts.push((sc, so));
                        pos += 8;
                    }
                }
                b"stsc" if b.payload.len() >= 8 => {
                    let count = u32::from_be_bytes([
                        b.payload[4],
                        b.payload[5],
                        b.payload[6],
                        b.payload[7],
                    ]) as usize;
                    let mut pos = 8usize;
                    tables.stsc.reserve(count.min(65536));
                    for _ in 0..count {
                        if pos + 12 > b.payload.len() {
                            break;
                        }
                        let first_chunk = u32::from_be_bytes([
                            b.payload[pos],
                            b.payload[pos + 1],
                            b.payload[pos + 2],
                            b.payload[pos + 3],
                        ]);
                        let spc = u32::from_be_bytes([
                            b.payload[pos + 4],
                            b.payload[pos + 5],
                            b.payload[pos + 6],
                            b.payload[pos + 7],
                        ]);
                        let sdi = u32::from_be_bytes([
                            b.payload[pos + 8],
                            b.payload[pos + 9],
                            b.payload[pos + 10],
                            b.payload[pos + 11],
                        ]);
                        tables.stsc.push((first_chunk, spc, sdi));
                        pos += 12;
                    }
                }
                b"stsz" if b.payload.len() >= 12 => {
                    tables.stsz_default = u32::from_be_bytes([
                        b.payload[4],
                        b.payload[5],
                        b.payload[6],
                        b.payload[7],
                    ]);
                    tables.stsz_count = u32::from_be_bytes([
                        b.payload[8],
                        b.payload[9],
                        b.payload[10],
                        b.payload[11],
                    ]) as usize;
                    if tables.stsz_default == 0 {
                        let mut pos = 12usize;
                        tables.stsz_entries.reserve(tables.stsz_count.min(262_144));
                        for _ in 0..tables.stsz_count {
                            if pos + 4 > b.payload.len() {
                                break;
                            }
                            let sz = u32::from_be_bytes([
                                b.payload[pos],
                                b.payload[pos + 1],
                                b.payload[pos + 2],
                                b.payload[pos + 3],
                            ]);
                            tables.stsz_entries.push(sz);
                            pos += 4;
                        }
                    }
                }
                b"stz2" if b.payload.len() >= 12 => {
                    // ISO 14496-12 Section 8.7.3.3 CompactSampleSizeBox (`stz2`):
                    // bytes 0..4: version/flags, 4..7: reserved, 7: field_size (4, 8, 16, or 32), 8..12: sample_count
                    let field_size = b.payload[7];
                    tables.stsz_default = 0;
                    tables.stsz_count = u32::from_be_bytes([
                        b.payload[8],
                        b.payload[9],
                        b.payload[10],
                        b.payload[11],
                    ]) as usize;
                    let entries_buf = &b.payload[12..];
                    tables.stsz_entries.reserve(tables.stsz_count.min(262_144));
                    match field_size {
                        4 => {
                            for i in 0..tables.stsz_count {
                                let byte_idx = i / 2;
                                if byte_idx >= entries_buf.len() {
                                    break;
                                }
                                let nib = if i.is_multiple_of(2) {
                                    (entries_buf[byte_idx] >> 4) & 0x0F
                                } else {
                                    entries_buf[byte_idx] & 0x0F
                                };
                                tables.stsz_entries.push(nib as u32);
                            }
                        }
                        8 => {
                            for &b_sz in entries_buf.iter().take(tables.stsz_count) {
                                tables.stsz_entries.push(b_sz as u32);
                            }
                        }
                        16 => {
                            for pair in entries_buf
                                .as_chunks::<2>()
                                .0
                                .iter()
                                .take(tables.stsz_count)
                            {
                                tables.stsz_entries.push(u16::from_be_bytes(*pair) as u32);
                            }
                        }
                        32 => {
                            for quad in entries_buf
                                .as_chunks::<4>()
                                .0
                                .iter()
                                .take(tables.stsz_count)
                            {
                                tables.stsz_entries.push(u32::from_be_bytes(*quad));
                            }
                        }
                        _ => {}
                    }
                }
                b"stco" if b.payload.len() >= 8 => {
                    let count = u32::from_be_bytes([
                        b.payload[4],
                        b.payload[5],
                        b.payload[6],
                        b.payload[7],
                    ]) as usize;
                    let mut pos = 8usize;
                    tables.chunk_offsets.reserve(count.min(65536));
                    for _ in 0..count {
                        if pos + 4 > b.payload.len() {
                            break;
                        }
                        let off = u32::from_be_bytes([
                            b.payload[pos],
                            b.payload[pos + 1],
                            b.payload[pos + 2],
                            b.payload[pos + 3],
                        ]) as u64;
                        tables.chunk_offsets.push(off);
                        pos += 4;
                    }
                }
                b"co64" if b.payload.len() >= 8 => {
                    let count = u32::from_be_bytes([
                        b.payload[4],
                        b.payload[5],
                        b.payload[6],
                        b.payload[7],
                    ]) as usize;
                    let mut pos = 8usize;
                    tables.chunk_offsets.reserve(count.min(65536));
                    for _ in 0..count {
                        if pos + 8 > b.payload.len() {
                            break;
                        }
                        let off = u64::from_be_bytes([
                            b.payload[pos],
                            b.payload[pos + 1],
                            b.payload[pos + 2],
                            b.payload[pos + 3],
                            b.payload[pos + 4],
                            b.payload[pos + 5],
                            b.payload[pos + 6],
                            b.payload[pos + 7],
                        ]);
                        tables.chunk_offsets.push(off);
                        pos += 8;
                    }
                }
                b"stss" if b.payload.len() >= 8 => {
                    let count = u32::from_be_bytes([
                        b.payload[4],
                        b.payload[5],
                        b.payload[6],
                        b.payload[7],
                    ]) as usize;
                    let mut pos = 8usize;
                    tables.stss.reserve(count.min(65536));
                    for _ in 0..count {
                        if pos + 4 > b.payload.len() {
                            break;
                        }
                        let sync_idx = u32::from_be_bytes([
                            b.payload[pos],
                            b.payload[pos + 1],
                            b.payload[pos + 2],
                            b.payload[pos + 3],
                        ]);
                        tables.stss.push(sync_idx);
                        pos += 4;
                    }
                }
                _ => {}
            }
        }

        let Some(cid) = codec_id else {
            return Ok(None);
        };

        let (samples, total_bytes, total_duration_ticks, default_sample_duration) =
            Self::compile_sample_table(&tables, elst_media_time, file_len)?;

        let timebase = Rational64::new(1, timescale as i64);
        if track_type == TrackType::Video && !samples.is_empty() && total_duration_ticks > 0 {
            video_params.frame_rate = Rational64::new(
                samples.len() as i64 * timescale as i64,
                total_duration_ticks,
            );
        }

        let info = TrackInfo {
            index: track_index,
            track_type,
            codec_id: cid,
            timebase,
            duration_ticks: duration_ticks.or(Some(total_duration_ticks)),
            extra_data,
            video: (track_type == TrackType::Video).then_some(video_params),
            audio: (track_type == TrackType::Audio).then_some(audio_params),
            sample_count: samples.len(),
            total_bytes,
        };

        Ok(Some(Mp4TrackIndex {
            track_id: tkhd_track_id,
            info,
            samples,
            default_sample_duration,
        }))
    }

    fn parse_stsd(
        stsd_payload: &'a [u8],
        track_type: TrackType,
        codec_id: &mut Option<CodecId>,
        extra_data: &mut Vec<u8>,
        video: &mut VideoTrackParams,
        audio: &mut AudioTrackParams,
    ) -> Result<()> {
        if stsd_payload.len() < 8 {
            return Ok(());
        }
        for entry_res in BoxIter::new(&stsd_payload[8..]) {
            let entry = entry_res?;
            match track_type {
                TrackType::Video => {
                    *codec_id = match &entry.fourcc.0 {
                        b"avc1" | b"avc3" => Some(CodecId::H264),
                        b"hvc1" | b"hev1" => Some(CodecId::Hevc),
                        b"av01" => Some(CodecId::Av1),
                        b"vp09" => Some(CodecId::Vp9),
                        b"rgdi" => Some(CodecId::RagdollIntra),
                        b"raw " | b"v210" => Some(CodecId::RawVideo),
                        _ => None,
                    };
                    // VisualSampleEntry has 78 bytes of fixed fields before child boxes
                    if entry.payload.len() >= 78 {
                        let w = u16::from_be_bytes([entry.payload[24], entry.payload[25]]) as u32;
                        let h = u16::from_be_bytes([entry.payload[26], entry.payload[27]]) as u32;
                        let depth = u16::from_be_bytes([entry.payload[74], entry.payload[75]]);
                        if w > 0 {
                            video.width = w;
                        }
                        if h > 0 {
                            video.height = h;
                        }
                        if depth == 0x001E || depth == 10 {
                            video.bit_depth = 10;
                            video.pixel_format = PixelFormat::Yuv420p10le;
                        }

                        for child_res in BoxIter::new(&entry.payload[78..]) {
                            let child = child_res?;
                            match &child.fourcc.0 {
                                b"avcC" | b"hvcC" | b"av1C" | b"vpcC" | b"rgdC" => {
                                    extra_data.clear();
                                    extra_data.extend_from_slice(child.payload);
                                }
                                b"colr"
                                    if child.payload.len() >= 11
                                        && &child.payload[0..4] == b"nclx" =>
                                {
                                    let prim =
                                        u16::from_be_bytes([child.payload[4], child.payload[5]])
                                            as u8;
                                    let trc =
                                        u16::from_be_bytes([child.payload[6], child.payload[7]])
                                            as u8;
                                    let mat =
                                        u16::from_be_bytes([child.payload[8], child.payload[9]])
                                            as u8;
                                    let full_range = (child.payload[10] & 0x80) != 0;
                                    video.colorimetry = Colorimetry {
                                        primaries: ColorPrimaries::from_cicp(prim),
                                        transfer: TransferCharacteristics::from_cicp(trc),
                                        matrix: MatrixCoefficients::from_cicp(mat),
                                        range: if full_range {
                                            ColorRange::Full
                                        } else {
                                            ColorRange::Limited
                                        },
                                        ..Default::default()
                                    };
                                }
                                _ => {}
                            }
                        }
                    }
                }
                TrackType::Audio => {
                    *codec_id = match &entry.fourcc.0 {
                        b"mp4a" => Some(CodecId::Aac),
                        b"fLaC" => Some(CodecId::Flac),
                        b"Opus" => Some(CodecId::Opus),
                        b"sowt" | b"lpcm" | b"twos" => Some(CodecId::PcmS16Le),
                        b"in24" => Some(CodecId::PcmS24Le),
                        b"in32" => Some(CodecId::PcmS32Le),
                        b"fl32" => Some(CodecId::PcmF32Le),
                        b"raw " => Some(CodecId::PcmU8),
                        _ => None,
                    };
                    // AudioSampleEntry v0 is 28 bytes before child boxes; v1 is 44 bytes; v2 is 64 bytes
                    if entry.payload.len() >= 28 {
                        let version = u16::from_be_bytes([entry.payload[8], entry.payload[9]]);
                        let ch = u16::from_be_bytes([entry.payload[16], entry.payload[17]]) as u8;
                        let sample_size =
                            u16::from_be_bytes([entry.payload[18], entry.payload[19]]) as u8;
                        let sr_fixed = u32::from_be_bytes([
                            entry.payload[24],
                            entry.payload[25],
                            entry.payload[26],
                            entry.payload[27],
                        ]);
                        let sr = sr_fixed >> 16;
                        if ch > 0 {
                            audio.channels = ch;
                            audio.channel_layout = ChannelLayout::from_channels(ch);
                        }
                        if sample_size > 0 {
                            audio.bits_per_coded_sample = sample_size;
                            audio.sample_format = match sample_size {
                                8 => SampleFormat::U8,
                                24 => SampleFormat::S24le,
                                32 if *codec_id == Some(CodecId::PcmF32Le) => SampleFormat::F32le,
                                32 => SampleFormat::S32le,
                                _ => SampleFormat::S16le,
                            };
                        }
                        if sr > 0 {
                            audio.sample_rate = sr;
                        }
                        let child_start = match version {
                            1 => 44usize,
                            2 => 64usize,
                            _ => 28usize,
                        };
                        if entry.payload.len() > child_start {
                            for child_res in BoxIter::new(&entry.payload[child_start..]) {
                                let child = child_res?;
                                match &child.fourcc.0 {
                                    b"dfLa" if child.payload.len() >= 8 => {
                                        // ISO/IEC 14496-12 FLACSpecificBox: 4-byte version/flags followed by FLAC metadata blocks
                                        let blocks = &child.payload[4..];
                                        // First block is STREAMINFO (4-byte block header + 34-byte STREAMINFO)
                                        if blocks.len() >= 38 && (blocks[0] & 0x7F) == 0 {
                                            extra_data.clear();
                                            extra_data.extend_from_slice(&blocks[4..38]);
                                            // Parse exact 20-bit sample rate, 3-bit channels-1, 5-bit bps-1 from STREAMINFO
                                            let s = &blocks[4..38];
                                            let flac_sr = ((s[10] as u32) << 12)
                                                | ((s[11] as u32) << 4)
                                                | ((s[12] as u32) >> 4);
                                            let flac_ch = ((s[12] >> 1) & 0x07) + 1;
                                            let flac_bps =
                                                (((s[12] & 0x01) << 4) | (s[13] >> 4)) + 1;
                                            if flac_sr > 0 {
                                                audio.sample_rate = flac_sr;
                                            }
                                            audio.channels = flac_ch;
                                            audio.channel_layout =
                                                ChannelLayout::from_channels(flac_ch);
                                            audio.bits_per_coded_sample = flac_bps;
                                        } else {
                                            extra_data.clear();
                                            extra_data.extend_from_slice(blocks);
                                        }
                                    }
                                    b"esds" => {
                                        if let Some(asc) =
                                            Self::extract_esds_dec_specific_info(child.payload)
                                        {
                                            extra_data.clear();
                                            extra_data.extend_from_slice(asc);
                                            if let Some((asc_sr, asc_ch)) = Self::parse_aac_asc(asc)
                                            {
                                                if asc_sr > 0 {
                                                    audio.sample_rate = asc_sr;
                                                }
                                                if asc_ch > 0 {
                                                    audio.channels = asc_ch;
                                                    audio.channel_layout =
                                                        ChannelLayout::from_channels(asc_ch);
                                                }
                                            }
                                        }
                                    }
                                    b"dOps" => {
                                        extra_data.clear();
                                        extra_data.extend_from_slice(child.payload);
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                TrackType::Subtitle => {
                    *codec_id = match &entry.fourcc.0 {
                        b"wvtt" => Some(CodecId::WebVtt),
                        b"tx3g" | b"text" | b"stpp" => Some(CodecId::SubRip),
                        _ => None,
                    };
                    if entry.payload.len() >= 8 {
                        for child_res in BoxIter::new(&entry.payload[8..]) {
                            if let Ok(child) = child_res
                                && &child.fourcc.0 == b"vttC"
                            {
                                extra_data.clear();
                                extra_data.extend_from_slice(child.payload);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Parses MPEG-4 `esds` (Elementary Stream Descriptor) to extract the `DecSpecificInfo` (`tag 0x05`) payload.
    fn extract_esds_dec_specific_info(esds: &'a [u8]) -> Option<&'a [u8]> {
        if esds.len() < 6 {
            return None;
        }
        // Skip 4-byte FullBox version/flags
        let mut i = 4usize;
        while i < esds.len() {
            let tag = esds[i];
            i += 1;
            let mut len = 0usize;
            for _ in 0..4 {
                if i >= esds.len() {
                    return None;
                }
                let b = esds[i];
                i += 1;
                len = (len << 7) | ((b & 0x7F) as usize);
                if (b & 0x80) == 0 {
                    break;
                }
            }
            match tag {
                0x03 => {
                    // ES_Descr: 2-byte ES_ID + 1-byte flags (+ optional dependsOn/URL/OCR)
                    if i + 3 > esds.len() {
                        return None;
                    }
                    let flags = esds[i + 2];
                    i += 3;
                    if (flags & 0x80) != 0 {
                        i += 2;
                    }
                    if (flags & 0x40) != 0 && i < esds.len() {
                        let url_len = esds[i] as usize;
                        i += 1 + url_len;
                    }
                    if (flags & 0x20) != 0 {
                        i += 2;
                    }
                }
                0x04 => {
                    // DecoderConfigDescr: 13 bytes of objectTypeIndication/streamType/bufferSizeDB/maxBitrate/avgBitrate
                    i += 13;
                }
                0x05 => {
                    // DecSpecificInfo: AudioSpecificConfig!
                    if i + len <= esds.len() {
                        return Some(&esds[i..i + len]);
                    }
                    return None;
                }
                _ => {
                    i += len;
                }
            }
        }
        None
    }

    fn parse_aac_asc(asc: &[u8]) -> Option<(u32, u8)> {
        if asc.len() < 2 {
            return None;
        }
        const AAC_SR_TABLE: [u32; 13] = [
            96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025,
            8_000, 7_350,
        ];
        let b0 = asc[0];
        let b1 = asc[1];
        let mut aot = b0 >> 3;
        if aot == 31 && asc.len() >= 3 {
            aot = 32 + (((b0 & 0x07) << 3) | (b1 >> 5));
        }
        let _ = aot;
        let sr_idx = (((b0 & 0x07) << 1) | (b1 >> 7)) as usize;
        let ch_cfg = (b1 >> 3) & 0x0F;
        let sr = *AAC_SR_TABLE.get(sr_idx).unwrap_or(&0);
        Some((sr, ch_cfg))
    }

    /// Compiles fragmented ISOBMFF sample tables (`stsc`, `stco`/`co64`, `stsz`, `stts`, `ctts`, `stss`)
    /// into a single flat 24-byte-per-sample `Vec<SampleEntry>` in $O(N)$ linear time.
    fn compile_sample_table(
        tables: &RawStblTables,
        elst_media_time: i64,
        file_len: usize,
    ) -> Result<(Vec<SampleEntry>, u64, i64, i64)> {
        let sample_count = if tables.stsz_default == 0 {
            tables.stsz_entries.len()
        } else {
            tables.stsz_count
        };
        if sample_count > file_len.max(65536) {
            return Err(RagdollError::CorruptBitstream(
                "MP4 stsz sample_count exceeds file bounds",
            ));
        }

        if sample_count == 0 || tables.chunk_offsets.is_empty() || tables.stsc.is_empty() {
            return Ok((Vec::new(), 0, 0, 1));
        }

        let mut samples = Vec::with_capacity(sample_count.min(65536));
        let mut total_bytes = 0u64;

        // Cursors for run-length tables (stts, ctts, stss)
        let mut stts_idx = 0usize;
        let mut stts_rem = tables.stts.first().map(|e| e.0).unwrap_or(0);
        let mut cur_dts = 0i64;
        let mut first_delta = 1i64;

        let mut ctts_idx = 0usize;
        let mut ctts_rem = tables.ctts.first().map(|e| e.0).unwrap_or(0);

        let mut stss_cursor = 0usize;
        let all_keyframes = tables.stss.is_empty();

        let num_chunks = tables.chunk_offsets.len();
        let mut sample_idx = 0usize;

        for (stsc_i, &(first_chunk_1b, samples_per_chunk, _sdi)) in tables.stsc.iter().enumerate() {
            if samples_per_chunk == 0 || first_chunk_1b == 0 {
                continue;
            }
            let start_chunk = ((first_chunk_1b - 1) as usize).min(num_chunks);
            let end_chunk = if stsc_i + 1 < tables.stsc.len() {
                ((tables.stsc[stsc_i + 1].0.saturating_sub(1)) as usize).min(num_chunks)
            } else {
                num_chunks
            };
            if start_chunk >= end_chunk {
                continue;
            }

            for &chunk_base_offset in &tables.chunk_offsets[start_chunk..end_chunk] {
                let mut cur_offset = chunk_base_offset;
                for _ in 0..samples_per_chunk {
                    if sample_idx >= sample_count {
                        break;
                    }
                    let size = if tables.stsz_default != 0 {
                        tables.stsz_default
                    } else {
                        tables.stsz_entries[sample_idx]
                    };

                    if (cur_offset as usize).saturating_add(size as usize) > file_len {
                        return Err(RagdollError::CorruptBitstream(
                            "MP4 sample offset + size exceeds file size",
                        ));
                    }

                    // Advance stts run-length cursor
                    while stts_rem == 0 && stts_idx + 1 < tables.stts.len() {
                        stts_idx += 1;
                        stts_rem = tables.stts[stts_idx].0;
                    }
                    let delta = tables
                        .stts
                        .get(stts_idx)
                        .map(|e| e.1 as i64)
                        .unwrap_or(first_delta);
                    if sample_idx == 0 && delta > 0 {
                        first_delta = delta;
                    }
                    stts_rem = stts_rem.saturating_sub(1);

                    // Advance ctts run-length cursor
                    let cts_delta = if tables.ctts.is_empty() {
                        0i32
                    } else {
                        while ctts_rem == 0 && ctts_idx + 1 < tables.ctts.len() {
                            ctts_idx += 1;
                            ctts_rem = tables.ctts[ctts_idx].0;
                        }
                        let off = tables.ctts.get(ctts_idx).map(|e| e.1).unwrap_or(0);
                        ctts_rem = ctts_rem.saturating_sub(1);
                        off
                    };

                    // Check stss keyframe in O(1) amortized time
                    let sample_1based = (sample_idx + 1) as u32;
                    while stss_cursor < tables.stss.len()
                        && tables.stss[stss_cursor] < sample_1based
                    {
                        stss_cursor += 1;
                    }
                    let is_keyframe = all_keyframes
                        || (stss_cursor < tables.stss.len()
                            && tables.stss[stss_cursor] == sample_1based);

                    let dts_adjusted = cur_dts - elst_media_time;
                    let pts_adjusted = dts_adjusted + (cts_delta as i64);

                    samples.push(SampleEntry::new(
                        cur_offset,
                        pts_adjusted,
                        size,
                        cts_delta,
                        is_keyframe,
                    ));

                    cur_offset += size as u64;
                    total_bytes += size as u64;
                    cur_dts += delta;
                    sample_idx += 1;
                }
                if sample_idx >= sample_count {
                    break;
                }
            }
            if sample_idx >= sample_count {
                break;
            }
        }

        let total_duration = (cur_dts - elst_media_time).max(0);
        Ok((samples, total_bytes, total_duration, first_delta))
    }
}

struct MuxTrackPacket<'a> {
    payload: PacketSlice<'a>,
}

struct MuxTrackState<'a> {
    info: TrackInfo,
    packets: Vec<MuxTrackPacket<'a>>,
}

/// Fast-Start ISOBMFF / MP4 Muxer (`moov` written before `mdat`).
///
/// Accumulates zero-copy [`PacketSlice<'a>`] references, builds run-length-compressed
/// `stts`/`ctts`/`stsc`/`stsz`/`stco`/`co64`/`stss` boxes in `moov`, computes the exact
/// `ftyp + moov + mdat_header` offset prior to payload emission, and streams packet payloads
/// via [`VectoredWriter`] with zero payload copies.
pub struct Mp4Muxer<'a> {
    tracks: Vec<MuxTrackState<'a>>,
}

impl<'a> Default for Mp4Muxer<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Mp4Muxer<'a> {
    #[inline]
    pub const fn new() -> Self {
        Self { tracks: Vec::new() }
    }

    /// Registers a media track and returns its 0-based `track_index`.
    pub fn add_track(&mut self, mut info: TrackInfo) -> u32 {
        let idx = self.tracks.len() as u32;
        info.index = idx;
        self.tracks.push(MuxTrackState {
            info,
            packets: Vec::new(),
        });
        idx
    }

    /// Queues a zero-copy [`PacketSlice<'a>`] for emission.
    pub fn write_packet(&mut self, packet: PacketSlice<'a>) -> Result<()> {
        let track = self.tracks.get_mut(packet.track_index as usize).ok_or(
            RagdollError::CorruptBitstream("Mp4Muxer::write_packet track_index out of bounds"),
        )?;
        let packet = if track.info.track_type == TrackType::Subtitle {
            let tb = track.info.timebase;
            let cur_start = packet.pts.rescale_to(tb).ticks.max(0);
            let cur_dur = rescale_q(packet.duration.max(1), packet.pts.timebase, tb).max(1);
            let prev_end = if let Some(last) = track.packets.last() {
                let last_start = last.payload.pts.rescale_to(tb).ticks.max(0);
                let last_dur =
                    rescale_q(last.payload.duration.max(1), last.payload.pts.timebase, tb).max(1);
                last_start + last_dur
            } else {
                0
            };
            if cur_start > prev_end {
                let vtte: [u8; 8] = [0, 0, 0, 8, b'v', b't', b't', b'e'];
                let gap_ts = Timestamp::new(prev_end, tb);
                let gap_pkt = PacketSlice::from_owned(
                    ragdoll_core::AlignedBuffer::from_slice(&vtte),
                    packet.track_index,
                    gap_ts,
                    gap_ts,
                    cur_start - prev_end,
                    PacketFlags::KEYFRAME,
                );
                track.packets.push(MuxTrackPacket { payload: gap_pkt });
            }
            let wrapped = crate::subtitle::wrap_mp4_wvtt_sample(packet.data(), "");
            let ts = Timestamp::new(cur_start, tb);
            let mut owned = PacketSlice::from_owned(
                ragdoll_core::AlignedBuffer::from_slice(&wrapped),
                packet.track_index,
                ts,
                ts,
                cur_dur,
                PacketFlags::KEYFRAME,
            );
            owned.file_offset = packet.file_offset;
            owned
        } else {
            packet
        };
        track.packets.push(MuxTrackPacket { payload: packet });
        Ok(())
    }

    /// Finalizes and writes a spec-compliant Fast-Start MP4 (`ftyp` + `moov` + `mdat`)
    /// to `writer` using vectored I/O.
    pub fn finish<W: Write>(self, writer: W) -> Result<u64> {
        let mut vw = VectoredWriter::new(writer);

        // 1. Compute relative byte offsets of every packet within the `mdat` payload body
        //    We interleave chunks by track/packet order and compute total mdat payload size.
        let mut track_rel_offsets: Vec<Vec<u64>> = self
            .tracks
            .iter()
            .map(|t| Vec::with_capacity(t.packets.len()))
            .collect();

        let mut mdat_payload_len = 0u64;
        for (t_idx, track) in self.tracks.iter().enumerate() {
            for pkt in &track.packets {
                track_rel_offsets[t_idx].push(mdat_payload_len);
                mdat_payload_len += pkt.payload.len() as u64;
            }
        }

        // 2. Build `ftyp` box
        let ftyp_box = Self::build_ftyp_box();

        // 3. Determine whether 64-bit `co64` offsets are needed and build a dry-run `moov`
        //    with `base_offset = 0` to learn exact `moov_box.len()`.
        let use_co64 = mdat_payload_len > (u32::MAX as u64 - 16 * 1024 * 1024);
        let mdat_header_len = if mdat_payload_len + 8 > u32::MAX as u64 {
            16u64
        } else {
            8u64
        };

        let dummy_moov = self.build_moov_box(&track_rel_offsets, 0, use_co64)?;
        let mdat_data_start = (ftyp_box.len() as u64) + (dummy_moov.len() as u64) + mdat_header_len;

        // 4. Rebuild `moov` with exact `mdat_data_start` offsets (`moov` byte length is invariant!)
        let moov_box = self.build_moov_box(&track_rel_offsets, mdat_data_start, use_co64)?;
        debug_assert_eq!(dummy_moov.len(), moov_box.len());

        // 5. Build `mdat` header (8-byte or 16-byte 64-bit largesize)
        let mut mdat_header = Vec::with_capacity(16);
        if mdat_header_len == 16 {
            mdat_header.extend_from_slice(&1u32.to_be_bytes());
            mdat_header.extend_from_slice(b"mdat");
            mdat_header.extend_from_slice(&(mdat_payload_len + 16).to_be_bytes());
        } else {
            mdat_header.extend_from_slice(&((mdat_payload_len + 8) as u32).to_be_bytes());
            mdat_header.extend_from_slice(b"mdat");
        }

        // 6. Write `ftyp` + `moov` + `mdat` header in a single vectored write
        vw.write_vectored_all(&[&ftyp_box, &moov_box, &mdat_header])?;

        // 7. Stream all packet payloads into `mdat` in batches of 8 via `writev`
        let mut slice_batch: Vec<&[u8]> = Vec::with_capacity(8);
        for track in &self.tracks {
            for pkt in &track.packets {
                slice_batch.push(pkt.payload.data());
                if slice_batch.len() == 8 {
                    vw.write_vectored_all(&slice_batch)?;
                    slice_batch.clear();
                }
            }
        }
        if !slice_batch.is_empty() {
            vw.write_vectored_all(&slice_batch)?;
        }

        vw.flush()?;
        Ok(vw.bytes_written())
    }

    /// Writes a standalone CMAF (`init.mp4`) initialization segment (`ftyp` + `moov`/`mvex`/`trex`).
    pub fn write_cmaf_init_segment<W: Write>(&self, writer: W) -> Result<u64> {
        let mut vw = VectoredWriter::new(writer);
        let mut ftyp_payload = Vec::with_capacity(28);
        ftyp_payload.extend_from_slice(b"iso5");
        ftyp_payload.extend_from_slice(&512u32.to_be_bytes());
        ftyp_payload.extend_from_slice(b"iso5");
        ftyp_payload.extend_from_slice(b"iso6");
        ftyp_payload.extend_from_slice(b"cmfc");
        ftyp_payload.extend_from_slice(b"avc1");
        ftyp_payload.extend_from_slice(b"mp41");
        let ftyp_box = Self::wrap_box(b"ftyp", &ftyp_payload);
        let init_moov = self.build_fragmented_init_moov()?;
        vw.write_vectored_all(&[&ftyp_box, &init_moov])?;
        vw.flush()?;
        Ok(vw.bytes_written())
    }

    /// Writes a standalone CMAF (`.m4s`) media segment (`styp` + `sidx` + `moof`/`tfdt`/`trun` + `mdat`).
    pub fn write_cmaf_media_segment<W: Write>(
        &self,
        writer: W,
        seq_num: u32,
        track_id: u32,
        packets: &[PacketSlice<'_>],
    ) -> Result<u64> {
        if packets.is_empty() {
            return Ok(0);
        }
        let mut vw = VectoredWriter::new(writer);
        let t_idx = (track_id.saturating_sub(1) as usize).min(self.tracks.len().saturating_sub(1));
        let tb = self
            .tracks
            .get(t_idx)
            .map_or(Rational64::new(1, 90_000), |t| t.info.timebase);
        let track_timescale = if tb.den > 0 { tb.den as u32 } else { 90_000 };
        let target_tb = Rational64::new(1, track_timescale as i64);

        let base_dts = packets[0].dts.rescale_to(target_tb).ticks.max(0) as u64;
        let earliest_pts = packets[0].pts.rescale_to(target_tb).ticks.max(0) as u64;

        let mut trun_entries = Vec::with_capacity(packets.len());
        let mut frag_payload_len = 0u64;
        let mut subsegment_duration = 0u32;

        for (i, pkt) in packets.iter().enumerate() {
            let pts_ticks = pkt.pts.rescale_to(target_tb).ticks;
            let dts_ticks = pkt.dts.rescale_to(target_tb).ticks;
            let cts_offset = (pts_ticks - dts_ticks) as i32;
            let dur = if i + 1 < packets.len() {
                let next_dts = packets[i + 1].dts.rescale_to(target_tb).ticks;
                (next_dts - dts_ticks).max(1) as u32
            } else {
                rescale_q(pkt.duration.max(1), pkt.dts.timebase, target_tb).max(1) as u32
            };
            let size = pkt.len() as u32;
            let flags = if pkt.is_keyframe() {
                0x0200_0000u32
            } else {
                0x0101_0000u32
            };
            trun_entries.push((dur, size, flags, cts_offset));
            frag_payload_len += size as u64;
            subsegment_duration = subsegment_duration.saturating_add(dur);
        }

        let dummy_moof = Self::build_moof_box(seq_num, track_id, base_dts, 0, &trun_entries);
        let data_offset = (dummy_moof.len() as i32) + 8;
        let moof_box =
            Self::build_moof_box(seq_num, track_id, base_dts, data_offset, &trun_entries);

        let mdat_total_len = (frag_payload_len + 8) as u32;
        let mut mdat_header = Vec::with_capacity(8);
        mdat_header.extend_from_slice(&mdat_total_len.to_be_bytes());
        mdat_header.extend_from_slice(b"mdat");

        // `styp` box
        let mut styp_payload = Vec::with_capacity(24);
        styp_payload.extend_from_slice(b"msdh");
        styp_payload.extend_from_slice(&0u32.to_be_bytes());
        styp_payload.extend_from_slice(b"msdh");
        styp_payload.extend_from_slice(b"msix");
        styp_payload.extend_from_slice(b"cmfs");
        styp_payload.extend_from_slice(b"iso6");
        let styp_box = Self::wrap_box(b"styp", &styp_payload);

        // `sidx` (SegmentIndexBox, version 1 for 64-bit earliest_presentation_time)
        let ref_size = (moof_box.len() as u32) + mdat_total_len;
        let mut sidx_payload = Vec::with_capacity(44);
        sidx_payload.extend_from_slice(&0x0100_0000u32.to_be_bytes()); // version 1, flags 0
        sidx_payload.extend_from_slice(&track_id.to_be_bytes());
        sidx_payload.extend_from_slice(&track_timescale.to_be_bytes());
        sidx_payload.extend_from_slice(&earliest_pts.to_be_bytes());
        sidx_payload.extend_from_slice(&0u64.to_be_bytes()); // first_offset = 0
        sidx_payload.extend_from_slice(&0u16.to_be_bytes()); // reserved = 0
        sidx_payload.extend_from_slice(&1u16.to_be_bytes()); // reference_count = 1
        sidx_payload.extend_from_slice(&(ref_size & 0x7FFF_FFFF).to_be_bytes()); // reference_type=0, size
        sidx_payload.extend_from_slice(&subsegment_duration.to_be_bytes());
        // starts_with_SAP = 1 (bit 31), SAP_type = 1 (bits 28..30), SAP_delta_time = 0
        sidx_payload.extend_from_slice(&0x9000_0000u32.to_be_bytes());
        let sidx_box = Self::wrap_box(b"sidx", &sidx_payload);

        vw.write_vectored_all(&[&styp_box, &sidx_box, &moof_box, &mdat_header])?;
        for pkt in packets {
            vw.write_all_counted(pkt.data())?;
        }
        vw.flush()?;
        Ok(vw.bytes_written())
    }

    /// Finalizes and writes an ISO/IEC 14496-12 / CMAF / DASH / MSE Fragmented MP4 (`fMP4`):
    /// - `ftyp` (`iso5` / `iso6` / `mp41`)
    /// - Initialization `moov` (empty `stbl` tables + `mvex`/`trex` track extenders)
    /// - Streaming `moof` (`mfhd` + `traf` (`tfhd`, `tfdt`, `trun`)) + `mdat` fragments
    pub fn finish_fragmented<W: Write>(
        self,
        writer: W,
        max_samples_per_fragment: usize,
    ) -> Result<u64> {
        let mut vw = VectoredWriter::new(writer);
        let max_frag = max_samples_per_fragment.max(1);

        // 1. Build `ftyp` (`iso5`) and initialization `moov` (empty `stbl` + `mvex`/`trex`)
        let mut ftyp_payload = Vec::with_capacity(24);
        ftyp_payload.extend_from_slice(b"iso5");
        ftyp_payload.extend_from_slice(&512u32.to_be_bytes());
        ftyp_payload.extend_from_slice(b"iso5");
        ftyp_payload.extend_from_slice(b"iso6");
        ftyp_payload.extend_from_slice(b"avc1");
        ftyp_payload.extend_from_slice(b"mp41");
        let ftyp_box = Self::wrap_box(b"ftyp", &ftyp_payload);

        let init_moov = self.build_fragmented_init_moov()?;
        vw.write_vectored_all(&[&ftyp_box, &init_moov])?;

        // 2. Emit `moof` + `mdat` fragments per track
        let mut seq_num = 1u32;
        for (t_idx, track) in self.tracks.iter().enumerate() {
            if track.packets.is_empty() {
                continue;
            }
            let track_id = (t_idx + 1) as u32;
            let tb = track.info.timebase;
            let track_timescale = if tb.den > 0 { tb.den as u32 } else { 90_000 };
            let target_tb = Rational64::new(1, track_timescale as i64);

            let mut start = 0usize;
            while start < track.packets.len() {
                let mut end = (start + max_frag).min(track.packets.len());
                // If video track, split at an earlier keyframe boundary if one exists inside `(start + 1)..end`
                if track.info.track_type == TrackType::Video {
                    for idx in (start + 1)..end {
                        if track.packets[idx].payload.is_keyframe() {
                            end = idx;
                            break;
                        }
                    }
                }

                let frag_slice = &track.packets[start..end];
                let base_dts = frag_slice[0].payload.dts.rescale_to(target_tb).ticks.max(0) as u64;

                let mut trun_entries = Vec::with_capacity(frag_slice.len());
                let mut frag_payload_len = 0u64;

                for (local_i, pkt_wrap) in frag_slice.iter().enumerate() {
                    let global_i = start + local_i;
                    let pkt = &pkt_wrap.payload;
                    let pts_ticks = pkt.pts.rescale_to(target_tb).ticks;
                    let dts_ticks = pkt.dts.rescale_to(target_tb).ticks;
                    let cts_offset = (pts_ticks - dts_ticks) as i32;

                    let dur = if global_i + 1 < track.packets.len() {
                        let next_dts = track.packets[global_i + 1]
                            .payload
                            .dts
                            .rescale_to(target_tb)
                            .ticks;
                        (next_dts - dts_ticks).max(1) as u32
                    } else {
                        rescale_q(pkt.duration.max(1), pkt.dts.timebase, target_tb).max(1) as u32
                    };
                    let size = pkt.len() as u32;
                    // ISO 14496-12 sample_flags:
                    // Sync/IDR keyframe: `0x0200_0000` (depends_on = 2: does not depend on others)
                    // Non-sync sample:   `0x0101_0000` (depends_on = 1, is_non_sync = 1)
                    let flags = if pkt.is_keyframe() {
                        0x0200_0000u32
                    } else {
                        0x0101_0000u32
                    };
                    trun_entries.push((dur, size, flags, cts_offset));
                    frag_payload_len += size as u64;
                }

                // Dry-run build `moof` with `data_offset = 0` to learn exact `moof` byte size
                let dummy_moof =
                    Self::build_moof_box(seq_num, track_id, base_dts, 0, &trun_entries);
                let data_offset = (dummy_moof.len() as i32) + 8;
                let moof_box =
                    Self::build_moof_box(seq_num, track_id, base_dts, data_offset, &trun_entries);
                debug_assert_eq!(dummy_moof.len(), moof_box.len());

                let mut mdat_header = Vec::with_capacity(8);
                mdat_header.extend_from_slice(&((frag_payload_len + 8) as u32).to_be_bytes());
                mdat_header.extend_from_slice(b"mdat");

                vw.write_vectored_all(&[&moof_box, &mdat_header])?;

                let mut slice_batch: Vec<&[u8]> = Vec::with_capacity(8);
                for pkt_wrap in frag_slice {
                    slice_batch.push(pkt_wrap.payload.data());
                    if slice_batch.len() == 8 {
                        vw.write_vectored_all(&slice_batch)?;
                        slice_batch.clear();
                    }
                }
                if !slice_batch.is_empty() {
                    vw.write_vectored_all(&slice_batch)?;
                }

                seq_num += 1;
                start = end;
            }
        }

        vw.flush()?;
        Ok(vw.bytes_written())
    }

    fn build_fragmented_init_moov(&self) -> Result<Vec<u8>> {
        let movie_timescale = 1000u32;
        let mut trak_boxes = Vec::with_capacity(self.tracks.len());
        let mut trex_boxes = Vec::with_capacity(self.tracks.len());

        for (t_idx, track) in self.tracks.iter().enumerate() {
            let track_id = (t_idx + 1) as u32;
            let empty_track = MuxTrackState {
                info: track.info.clone(),
                packets: Vec::new(),
            };
            let (trak_box, _) =
                self.build_trak_box(track_id, &empty_track, &[], 0, movie_timescale, false)?;
            trak_boxes.push(trak_box);

            let mut trex = Vec::with_capacity(24);
            trex.extend_from_slice(&0u32.to_be_bytes()); // version/flags
            trex.extend_from_slice(&track_id.to_be_bytes());
            trex.extend_from_slice(&1u32.to_be_bytes()); // default_sample_description_index = 1
            trex.extend_from_slice(&1u32.to_be_bytes()); // default_sample_duration = 1
            trex.extend_from_slice(&0u32.to_be_bytes()); // default_sample_size = 0
            trex.extend_from_slice(&0u32.to_be_bytes()); // default_sample_flags = 0
            trex_boxes.push(Self::wrap_box(b"trex", &trex));
        }

        let mvhd_box = Self::build_mvhd_box(movie_timescale, 0, (self.tracks.len() + 1) as u32);
        let mut mvex_payload = Vec::new();
        for tb in &trex_boxes {
            mvex_payload.extend_from_slice(tb);
        }
        let mvex_box = Self::wrap_box(b"mvex", &mvex_payload);

        let mut moov_payload = Vec::new();
        moov_payload.extend_from_slice(&mvhd_box);
        for tb in &trak_boxes {
            moov_payload.extend_from_slice(tb);
        }
        moov_payload.extend_from_slice(&mvex_box);
        Ok(Self::wrap_box(b"moov", &moov_payload))
    }

    fn build_moof_box(
        seq_num: u32,
        track_id: u32,
        base_dts: u64,
        data_offset: i32,
        trun_entries: &[(u32, u32, u32, i32)],
    ) -> Vec<u8> {
        let mut mfhd = Vec::with_capacity(8);
        mfhd.extend_from_slice(&0u32.to_be_bytes()); // version/flags
        mfhd.extend_from_slice(&seq_num.to_be_bytes());
        let mfhd_box = Self::wrap_box(b"mfhd", &mfhd);

        // `tfhd`: version 0, flags = 0x020020 (`default-base-is-moof | default-sample-flags-present`)
        let mut tfhd = Vec::with_capacity(12);
        tfhd.extend_from_slice(&0x0002_0020u32.to_be_bytes());
        tfhd.extend_from_slice(&track_id.to_be_bytes());
        tfhd.extend_from_slice(&0x0101_0000u32.to_be_bytes());
        let tfhd_box = Self::wrap_box(b"tfhd", &tfhd);

        // `tfdt`: version 1 (64-bit `baseMediaDecodeTime`)
        let mut tfdt = Vec::with_capacity(12);
        tfdt.extend_from_slice(&0x0100_0000u32.to_be_bytes());
        tfdt.extend_from_slice(&base_dts.to_be_bytes());
        let tfdt_box = Self::wrap_box(b"tfdt", &tfdt);

        // `trun`: version 1, flags = 0x000F01 (data-offset + duration + size + flags + cts-offset)
        let mut trun = Vec::with_capacity(12 + trun_entries.len() * 16);
        trun.extend_from_slice(&0x0100_0F01u32.to_be_bytes());
        trun.extend_from_slice(&(trun_entries.len() as u32).to_be_bytes());
        trun.extend_from_slice(&data_offset.to_be_bytes());
        for &(dur, size, flags, cts) in trun_entries {
            trun.extend_from_slice(&dur.to_be_bytes());
            trun.extend_from_slice(&size.to_be_bytes());
            trun.extend_from_slice(&flags.to_be_bytes());
            trun.extend_from_slice(&cts.to_be_bytes());
        }
        let trun_box = Self::wrap_box(b"trun", &trun);

        let mut traf_payload = Vec::with_capacity(tfhd_box.len() + tfdt_box.len() + trun_box.len());
        traf_payload.extend_from_slice(&tfhd_box);
        traf_payload.extend_from_slice(&tfdt_box);
        traf_payload.extend_from_slice(&trun_box);
        let traf_box = Self::wrap_box(b"traf", &traf_payload);

        let mut moof_payload = Vec::with_capacity(mfhd_box.len() + traf_box.len());
        moof_payload.extend_from_slice(&mfhd_box);
        moof_payload.extend_from_slice(&traf_box);
        Self::wrap_box(b"moof", &moof_payload)
    }

    fn build_ftyp_box() -> Vec<u8> {
        let mut payload = Vec::with_capacity(24);
        payload.extend_from_slice(b"isom");
        payload.extend_from_slice(&512u32.to_be_bytes());
        payload.extend_from_slice(b"isom");
        payload.extend_from_slice(b"iso2");
        payload.extend_from_slice(b"avc1");
        payload.extend_from_slice(b"mp41");
        Self::wrap_box(b"ftyp", &payload)
    }

    fn build_moov_box(
        &self,
        track_rel_offsets: &[Vec<u64>],
        mdat_data_start: u64,
        use_co64: bool,
    ) -> Result<Vec<u8>> {
        let movie_timescale = 1000u32;
        let mut max_movie_duration = 0u64;
        let mut trak_boxes = Vec::with_capacity(self.tracks.len());

        for (t_idx, track) in self.tracks.iter().enumerate() {
            let (trak_box, track_dur_movie_tb) = self.build_trak_box(
                (t_idx + 1) as u32,
                track,
                &track_rel_offsets[t_idx],
                mdat_data_start,
                movie_timescale,
                use_co64,
            )?;
            max_movie_duration = max_movie_duration.max(track_dur_movie_tb);
            trak_boxes.push(trak_box);
        }

        let mvhd_box = Self::build_mvhd_box(
            movie_timescale,
            max_movie_duration,
            (self.tracks.len() + 1) as u32,
        );

        let mut moov_payload =
            Vec::with_capacity(mvhd_box.len() + trak_boxes.iter().map(|b| b.len()).sum::<usize>());
        moov_payload.extend_from_slice(&mvhd_box);
        for tb in &trak_boxes {
            moov_payload.extend_from_slice(tb);
        }
        Ok(Self::wrap_box(b"moov", &moov_payload))
    }

    fn build_mvhd_box(timescale: u32, duration: u64, next_track_id: u32) -> Vec<u8> {
        let mut p = Vec::with_capacity(100);
        p.extend_from_slice(&0u32.to_be_bytes()); // version 0, flags 0
        p.extend_from_slice(&0u32.to_be_bytes()); // creation_time
        p.extend_from_slice(&0u32.to_be_bytes()); // modification_time
        p.extend_from_slice(&timescale.to_be_bytes());
        p.extend_from_slice(&(duration.min(u32::MAX as u64) as u32).to_be_bytes());
        p.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // rate 1.0
        p.extend_from_slice(&0x0100u16.to_be_bytes()); // volume 1.0
        p.extend_from_slice(&[0u8; 10]); // reserved
        // Unity 3x3 matrix
        for &m in &[0x0001_0000u32, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000] {
            p.extend_from_slice(&m.to_be_bytes());
        }
        p.extend_from_slice(&[0u8; 24]); // pre_defined
        p.extend_from_slice(&next_track_id.to_be_bytes());
        Self::wrap_box(b"mvhd", &p)
    }

    fn build_trak_box(
        &self,
        track_id: u32,
        track: &MuxTrackState<'a>,
        rel_offsets: &[u64],
        mdat_data_start: u64,
        movie_timescale: u32,
        use_co64: bool,
    ) -> Result<(Vec<u8>, u64)> {
        let tb = track.info.timebase;
        let track_timescale = if tb.den > 0 { tb.den as u32 } else { 90_000 };
        let target_tb = Rational64::new(1, track_timescale as i64);

        // Compute per-packet durations and CTS deltas in `target_tb`
        let mut stts_runs: Vec<(u32, u32)> = Vec::new();
        let mut ctts_runs: Vec<(u32, i32)> = Vec::new();
        let mut has_nonzero_cts = false;
        let mut stsz_entries: Vec<u32> = Vec::with_capacity(track.packets.len());
        let mut stss_entries: Vec<u32> = Vec::new();
        let mut all_keyframes = true;
        let mut total_track_dur = 0u64;

        for (i, pkt_wrap) in track.packets.iter().enumerate() {
            let pkt = &pkt_wrap.payload;
            let pts_ticks = pkt.pts.rescale_to(target_tb).ticks;
            let dts_ticks = pkt.dts.rescale_to(target_tb).ticks;
            let cts_delta = (pts_ticks - dts_ticks) as i32;
            if cts_delta != 0 {
                has_nonzero_cts = true;
            }
            if let Some(last) = ctts_runs.last_mut()
                && last.1 == cts_delta
            {
                last.0 += 1;
            } else {
                ctts_runs.push((1, cts_delta));
            }

            let dur = if i + 1 < track.packets.len() {
                let next_dts = track.packets[i + 1].payload.dts.rescale_to(target_tb).ticks;
                (next_dts - dts_ticks).max(1) as u32
            } else {
                rescale_q(pkt.duration.max(1), pkt.dts.timebase, target_tb).max(1) as u32
            };
            total_track_dur += dur as u64;
            if let Some(last) = stts_runs.last_mut()
                && last.1 == dur
            {
                last.0 += 1;
            } else {
                stts_runs.push((1, dur));
            }

            stsz_entries.push(pkt.len() as u32);
            if pkt.is_keyframe() {
                stss_entries.push((i + 1) as u32);
            } else {
                all_keyframes = false;
            }
        }

        let track_dur_movie_tb = rescale_q(
            total_track_dur as i64,
            target_tb,
            Rational64::new(1, movie_timescale as i64),
        )
        .max(0) as u64;

        let (width_fixed, height_fixed, volume_fixed) = match track.info.track_type {
            TrackType::Video => {
                let vp = track.info.video.as_ref().cloned().unwrap_or_default();
                (vp.width << 16, vp.height << 16, 0u16)
            }
            TrackType::Audio => (0, 0, 0x0100u16),
            _ => (0, 0, 0),
        };

        // `tkhd` (version 0, flags = 0x000003: Track_enabled | Track_in_movie)
        let mut tkhd = Vec::with_capacity(84);
        tkhd.extend_from_slice(&0x0000_0003u32.to_be_bytes());
        tkhd.extend_from_slice(&0u32.to_be_bytes()); // creation_time
        tkhd.extend_from_slice(&0u32.to_be_bytes()); // modification_time
        tkhd.extend_from_slice(&track_id.to_be_bytes());
        tkhd.extend_from_slice(&0u32.to_be_bytes()); // reserved
        tkhd.extend_from_slice(&(track_dur_movie_tb.min(u32::MAX as u64) as u32).to_be_bytes());
        tkhd.extend_from_slice(&[0u8; 8]); // reserved
        tkhd.extend_from_slice(&0u16.to_be_bytes()); // layer
        tkhd.extend_from_slice(&0u16.to_be_bytes()); // alternate_group
        tkhd.extend_from_slice(&volume_fixed.to_be_bytes());
        tkhd.extend_from_slice(&0u16.to_be_bytes()); // reserved
        for &m in &[0x0001_0000u32, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000] {
            tkhd.extend_from_slice(&m.to_be_bytes());
        }
        tkhd.extend_from_slice(&width_fixed.to_be_bytes());
        tkhd.extend_from_slice(&height_fixed.to_be_bytes());
        let tkhd_box = Self::wrap_box(b"tkhd", &tkhd);

        // `mdhd` (version 0)
        let mut mdhd = Vec::with_capacity(24);
        mdhd.extend_from_slice(&0u32.to_be_bytes());
        mdhd.extend_from_slice(&0u32.to_be_bytes());
        mdhd.extend_from_slice(&0u32.to_be_bytes());
        mdhd.extend_from_slice(&track_timescale.to_be_bytes());
        mdhd.extend_from_slice(&(total_track_dur.min(u32::MAX as u64) as u32).to_be_bytes());
        mdhd.extend_from_slice(&0x55C4u16.to_be_bytes()); // 'und' ISO-639-2 language
        mdhd.extend_from_slice(&0u16.to_be_bytes());
        let mdhd_box = Self::wrap_box(b"mdhd", &mdhd);

        // `hdlr`
        let (handler_fourcc, handler_name) = match track.info.track_type {
            TrackType::Video => (b"vide", b"Ragdoll Video Handler\0" as &[u8]),
            TrackType::Audio => (b"soun", b"Ragdoll Audio Handler\0" as &[u8]),
            TrackType::Subtitle => (b"text", b"Ragdoll Subtitle Handler\0" as &[u8]),
            _ => (b"data", b"Ragdoll Data Handler\0" as &[u8]),
        };
        let mut hdlr = Vec::with_capacity(24 + handler_name.len());
        hdlr.extend_from_slice(&0u32.to_be_bytes()); // version/flags
        hdlr.extend_from_slice(&0u32.to_be_bytes()); // pre_defined
        hdlr.extend_from_slice(handler_fourcc);
        hdlr.extend_from_slice(&[0u8; 12]); // reserved
        hdlr.extend_from_slice(handler_name);
        let hdlr_box = Self::wrap_box(b"hdlr", &hdlr);

        // `vmhd`, `smhd`, or `nmhd`
        let media_header_box = match track.info.track_type {
            TrackType::Video => {
                let mut vmhd = Vec::with_capacity(12);
                vmhd.extend_from_slice(&0x0000_0001u32.to_be_bytes()); // version 0, flags 1
                vmhd.extend_from_slice(&[0u8; 8]); // graphicsmode + opcolor
                Self::wrap_box(b"vmhd", &vmhd)
            }
            TrackType::Subtitle => Self::wrap_box(b"nmhd", &0u32.to_be_bytes()),
            _ => {
                let mut smhd = Vec::with_capacity(8);
                smhd.extend_from_slice(&0u32.to_be_bytes());
                smhd.extend_from_slice(&0u32.to_be_bytes()); // balance + reserved
                Self::wrap_box(b"smhd", &smhd)
            }
        };

        // `dinf` -> `dref` -> `url ` (self-contained flag = 1)
        let url_box = Self::wrap_box(b"url ", &0x0000_0001u32.to_be_bytes());
        let mut dref = Vec::with_capacity(8 + url_box.len());
        dref.extend_from_slice(&0u32.to_be_bytes());
        dref.extend_from_slice(&1u32.to_be_bytes());
        dref.extend_from_slice(&url_box);
        let dinf_box = Self::wrap_box(b"dinf", &Self::wrap_box(b"dref", &dref));

        // `stbl` child boxes
        let stsd_box = Self::build_stsd_box(&track.info)?;

        let mut stts = Vec::with_capacity(8 + stts_runs.len() * 8);
        stts.extend_from_slice(&0u32.to_be_bytes());
        stts.extend_from_slice(&(stts_runs.len() as u32).to_be_bytes());
        for &(cnt, delta) in &stts_runs {
            stts.extend_from_slice(&cnt.to_be_bytes());
            stts.extend_from_slice(&delta.to_be_bytes());
        }
        let stts_box = Self::wrap_box(b"stts", &stts);

        let ctts_box = if has_nonzero_cts {
            let has_negative = ctts_runs.iter().any(|&(_, off)| off < 0);
            let version_flags = if has_negative { 0x0100_0000u32 } else { 0u32 };
            let mut ctts = Vec::with_capacity(8 + ctts_runs.len() * 8);
            ctts.extend_from_slice(&version_flags.to_be_bytes());
            ctts.extend_from_slice(&(ctts_runs.len() as u32).to_be_bytes());
            for &(cnt, off) in &ctts_runs {
                ctts.extend_from_slice(&cnt.to_be_bytes());
                ctts.extend_from_slice(&off.to_be_bytes());
            }
            Some(Self::wrap_box(b"ctts", &ctts))
        } else {
            None
        };

        // Since all packets of a track are written contiguously in `mdat` in our faststart builder
        // (or 1 sample per chunk when interleaved), writing 1 sample per chunk (`samples_per_chunk = 1`)
        // works identically for both contiguous and arbitrarily interleaved streams with a single
        // 12-byte `stsc` entry!
        let mut stsc = Vec::with_capacity(20);
        stsc.extend_from_slice(&0u32.to_be_bytes());
        stsc.extend_from_slice(&1u32.to_be_bytes()); // 1 entry
        stsc.extend_from_slice(&1u32.to_be_bytes()); // first_chunk = 1
        stsc.extend_from_slice(&1u32.to_be_bytes()); // samples_per_chunk = 1
        stsc.extend_from_slice(&1u32.to_be_bytes()); // sample_description_index = 1
        let stsc_box = Self::wrap_box(b"stsc", &stsc);

        let mut stsz = Vec::with_capacity(12 + stsz_entries.len() * 4);
        stsz.extend_from_slice(&0u32.to_be_bytes());
        stsz.extend_from_slice(&0u32.to_be_bytes()); // sample_size = 0 (variable)
        stsz.extend_from_slice(&(stsz_entries.len() as u32).to_be_bytes());
        for &sz in &stsz_entries {
            stsz.extend_from_slice(&sz.to_be_bytes());
        }
        let stsz_box = Self::wrap_box(b"stsz", &stsz);

        let stco_or_co64_box = if use_co64 {
            let mut co64 = Vec::with_capacity(8 + rel_offsets.len() * 8);
            co64.extend_from_slice(&0u32.to_be_bytes());
            co64.extend_from_slice(&(rel_offsets.len() as u32).to_be_bytes());
            for &rel in rel_offsets {
                co64.extend_from_slice(&(mdat_data_start + rel).to_be_bytes());
            }
            Self::wrap_box(b"co64", &co64)
        } else {
            let mut stco = Vec::with_capacity(8 + rel_offsets.len() * 4);
            stco.extend_from_slice(&0u32.to_be_bytes());
            stco.extend_from_slice(&(rel_offsets.len() as u32).to_be_bytes());
            for &rel in rel_offsets {
                stco.extend_from_slice(&((mdat_data_start + rel) as u32).to_be_bytes());
            }
            Self::wrap_box(b"stco", &stco)
        };

        let stss_box = if track.info.track_type == TrackType::Video && !all_keyframes {
            let mut stss = Vec::with_capacity(8 + stss_entries.len() * 4);
            stss.extend_from_slice(&0u32.to_be_bytes());
            stss.extend_from_slice(&(stss_entries.len() as u32).to_be_bytes());
            for &idx in &stss_entries {
                stss.extend_from_slice(&idx.to_be_bytes());
            }
            Some(Self::wrap_box(b"stss", &stss))
        } else {
            None
        };

        let mut stbl_payload = Vec::new();
        stbl_payload.extend_from_slice(&stsd_box);
        stbl_payload.extend_from_slice(&stts_box);
        if let Some(c) = &ctts_box {
            stbl_payload.extend_from_slice(c);
        }
        stbl_payload.extend_from_slice(&stsc_box);
        stbl_payload.extend_from_slice(&stsz_box);
        stbl_payload.extend_from_slice(&stco_or_co64_box);
        if let Some(s) = &stss_box {
            stbl_payload.extend_from_slice(s);
        }
        let stbl_box = Self::wrap_box(b"stbl", &stbl_payload);

        let mut minf_payload =
            Vec::with_capacity(media_header_box.len() + dinf_box.len() + stbl_box.len());
        minf_payload.extend_from_slice(&media_header_box);
        minf_payload.extend_from_slice(&dinf_box);
        minf_payload.extend_from_slice(&stbl_box);
        let minf_box = Self::wrap_box(b"minf", &minf_payload);

        let mut mdia_payload = Vec::with_capacity(mdhd_box.len() + hdlr_box.len() + minf_box.len());
        mdia_payload.extend_from_slice(&mdhd_box);
        mdia_payload.extend_from_slice(&hdlr_box);
        mdia_payload.extend_from_slice(&minf_box);
        let mdia_box = Self::wrap_box(b"mdia", &mdia_payload);

        let mut trak_payload = Vec::with_capacity(tkhd_box.len() + mdia_box.len());
        trak_payload.extend_from_slice(&tkhd_box);
        trak_payload.extend_from_slice(&mdia_box);

        Ok((Self::wrap_box(b"trak", &trak_payload), track_dur_movie_tb))
    }

    fn build_stsd_box(info: &TrackInfo) -> Result<Vec<u8>> {
        let mut stsd = Vec::with_capacity(128);
        stsd.extend_from_slice(&0u32.to_be_bytes()); // version/flags
        stsd.extend_from_slice(&1u32.to_be_bytes()); // entry_count = 1

        match info.track_type {
            TrackType::Video => {
                let vp = info.video.as_ref().cloned().unwrap_or_default();
                let (entry_fourcc, cfg_fourcc) = match info.codec_id {
                    CodecId::H264 => (b"avc1", Some(b"avcC")),
                    CodecId::Hevc => (b"hvc1", Some(b"hvcC")),
                    CodecId::Av1 => (b"av01", Some(b"av1C")),
                    CodecId::Vp9 => (b"vp09", Some(b"vpcC")),
                    CodecId::RagdollIntra => (b"rgdi", Some(b"rgdC")),
                    _ => (b"raw ", None),
                };

                let mut vse = Vec::with_capacity(128);
                vse.extend_from_slice(&[0u8; 6]); // reserved
                vse.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index = 1
                vse.extend_from_slice(&[0u8; 16]); // pre_defined + reserved
                vse.extend_from_slice(&(vp.width.max(1) as u16).to_be_bytes());
                vse.extend_from_slice(&(vp.height.max(1) as u16).to_be_bytes());
                vse.extend_from_slice(&0x0048_0000u32.to_be_bytes()); // horizresolution 72 dpi
                vse.extend_from_slice(&0x0048_0000u32.to_be_bytes()); // vertresolution 72 dpi
                vse.extend_from_slice(&0u32.to_be_bytes()); // reserved
                vse.extend_from_slice(&1u16.to_be_bytes()); // frame_count = 1
                vse.extend_from_slice(&[0u8; 32]); // compressorname
                vse.extend_from_slice(&0x0018u16.to_be_bytes()); // depth = 24
                vse.extend_from_slice(&(-1i16).to_be_bytes()); // pre_defined = -1

                if let Some(cfg_tag) = cfg_fourcc
                    && !info.extra_data.is_empty()
                {
                    vse.extend_from_slice(&Self::wrap_box(cfg_tag, &info.extra_data));
                }

                // Append `colr` (`nclx`) CICP colorimetry box
                let mut colr = Vec::with_capacity(11);
                colr.extend_from_slice(b"nclx");
                colr.extend_from_slice(&(vp.colorimetry.primaries as u16).to_be_bytes());
                colr.extend_from_slice(&(vp.colorimetry.transfer as u16).to_be_bytes());
                colr.extend_from_slice(&(vp.colorimetry.matrix as u16).to_be_bytes());
                colr.push(if vp.colorimetry.range == ColorRange::Full {
                    0x80
                } else {
                    0x00
                });
                vse.extend_from_slice(&Self::wrap_box(b"colr", &colr));

                // Append `pasp` 1:1 pixel aspect ratio box
                let mut pasp = Vec::with_capacity(8);
                pasp.extend_from_slice(&1u32.to_be_bytes());
                pasp.extend_from_slice(&1u32.to_be_bytes());
                vse.extend_from_slice(&Self::wrap_box(b"pasp", &pasp));

                stsd.extend_from_slice(&Self::wrap_box(entry_fourcc, &vse));
            }
            TrackType::Audio => {
                let ap = info.audio.as_ref().cloned().unwrap_or_default();
                let entry_fourcc = match info.codec_id {
                    CodecId::Aac => b"mp4a",
                    CodecId::Flac => b"fLaC",
                    CodecId::Opus => b"Opus",
                    CodecId::PcmS16Le => b"sowt",
                    CodecId::PcmS24Le => b"in24",
                    CodecId::PcmS32Le => b"in32",
                    CodecId::PcmF32Le => b"fl32",
                    _ => b"sowt",
                };

                let mut ase = Vec::with_capacity(64);
                ase.extend_from_slice(&[0u8; 6]); // reserved
                ase.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index = 1
                ase.extend_from_slice(&[0u8; 8]); // version 0, revision 0, vendor 0
                ase.extend_from_slice(&(ap.channels.max(1) as u16).to_be_bytes());
                ase.extend_from_slice(&(ap.bits_per_coded_sample.max(16) as u16).to_be_bytes());
                ase.extend_from_slice(&0u16.to_be_bytes()); // compression_id
                ase.extend_from_slice(&0u16.to_be_bytes()); // packet_size
                ase.extend_from_slice(&((ap.sample_rate.min(65535)) << 16).to_be_bytes());

                match info.codec_id {
                    CodecId::Flac if !info.extra_data.is_empty() => {
                        // `dfLa`: 4-byte FullBox version/flags + 4-byte metadata block header (last=1, type=0, len=34) + STREAMINFO
                        let mut dfla = Vec::with_capacity(8 + info.extra_data.len());
                        dfla.extend_from_slice(&0u32.to_be_bytes());
                        if info.extra_data.len() == 34 {
                            dfla.push(0x80); // is_last = 1, block_type = 0 (STREAMINFO)
                            dfla.extend_from_slice(&[0x00, 0x00, 34]);
                        }
                        dfla.extend_from_slice(&info.extra_data);
                        ase.extend_from_slice(&Self::wrap_box(b"dfLa", &dfla));
                    }
                    CodecId::Aac => {
                        let asc = if info.extra_data.is_empty() {
                            // Default AAC-LC 48kHz stereo AudioSpecificConfig: [0x11, 0x90]
                            vec![0x11u8, 0x90]
                        } else {
                            info.extra_data.clone()
                        };
                        let esds = Self::build_esds_payload(&asc);
                        ase.extend_from_slice(&Self::wrap_box(b"esds", &esds));
                    }
                    CodecId::Opus if !info.extra_data.is_empty() => {
                        ase.extend_from_slice(&Self::wrap_box(b"dOps", &info.extra_data));
                    }
                    _ => {}
                }

                stsd.extend_from_slice(&Self::wrap_box(entry_fourcc, &ase));
            }
            TrackType::Subtitle => {
                // ISO/IEC 14496-30 WebVTTSampleEntry (`wvtt`):
                // 6 reserved bytes + 2-byte data_reference_index = 1 + `vttC` WebVTTConfigurationBox
                let mut wvtt = Vec::with_capacity(32);
                wvtt.extend_from_slice(&[0u8; 6]);
                wvtt.extend_from_slice(&1u16.to_be_bytes());
                let vttc_cfg = if info.extra_data.is_empty() {
                    b"WEBVTT\n" as &[u8]
                } else {
                    info.extra_data.as_slice()
                };
                wvtt.extend_from_slice(&Self::wrap_box(b"vttC", vttc_cfg));
                stsd.extend_from_slice(&Self::wrap_box(b"wvtt", &wvtt));
            }
            _ => {}
        }

        Ok(Self::wrap_box(b"stsd", &stsd))
    }

    fn build_esds_payload(asc: &[u8]) -> Vec<u8> {
        let asc_len = asc.len() as u8;
        let dec_cfg_len = 13 + 2 + asc_len;
        let es_descr_len = 3 + 2 + dec_cfg_len + 3;
        let mut p = Vec::with_capacity(6 + es_descr_len as usize);
        p.extend_from_slice(&0u32.to_be_bytes()); // FullBox version/flags
        // ES_Descr (tag 0x03)
        p.push(0x03);
        p.push(es_descr_len);
        p.extend_from_slice(&1u16.to_be_bytes()); // ES_ID = 1
        p.push(0x00); // flags = 0
        // DecoderConfigDescr (tag 0x04)
        p.push(0x04);
        p.push(dec_cfg_len);
        p.push(0x40); // objectTypeIndication = 0x40 (Audio ISO/IEC 14496-3)
        p.push(0x15); // streamType = 0x05 (AudioStream << 2 | 1)
        p.extend_from_slice(&[0x00, 0x00, 0x00]); // bufferSizeDB
        p.extend_from_slice(&128_000u32.to_be_bytes()); // maxBitrate
        p.extend_from_slice(&128_000u32.to_be_bytes()); // avgBitrate
        // DecSpecificInfo (tag 0x05)
        p.push(0x05);
        p.push(asc_len);
        p.extend_from_slice(asc);
        // SLConfigDescr (tag 0x06)
        p.extend_from_slice(&[0x06, 0x01, 0x02]);
        p
    }

    #[inline]
    fn wrap_box(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let total = 8 + payload.len();
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&(total as u32).to_be_bytes());
        out.extend_from_slice(fourcc);
        out.extend_from_slice(payload);
        out
    }
}
