use crate::io::VectoredWriter;
use ragdoll_core::{
    AudioTrackParams, ChannelLayout, CodecId, PacketFlags, PacketSlice, RagdollError, Rational64,
    Result, SampleFormat, Timestamp, TrackInfo, TrackType,
};
use std::io::Write;

const WAVE_FORMAT_PCM: u16 = 0x0001;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 0x0003;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// Zero-copy RIFF / RF64 / BW64 WAVE (`.wav`) PCM Demuxer (`libavformat/wavdec.c` compatible).
pub struct WavDemuxer<'a> {
    data: &'a [u8],
    track: TrackInfo,
    data_offset: usize,
    data_len: usize,
    block_align: usize,
    frames_per_packet: usize,
    cursor_bytes: usize,
    rf64: bool,
}

impl<'a> WavDemuxer<'a> {
    pub fn open(data: &'a [u8]) -> Result<Self> {
        Self::open_with_packet_frames(data, 1024)
    }

    pub fn open_with_packet_frames(data: &'a [u8], frames_per_packet: usize) -> Result<Self> {
        let is_riff = data.len() >= 12 && &data[0..4] == b"RIFF" && &data[8..12] == b"WAVE";
        let is_rf64 = data.len() >= 12
            && (&data[0..4] == b"RF64" || &data[0..4] == b"BW64")
            && &data[8..12] == b"WAVE";
        if !is_riff && !is_rf64 {
            return Err(RagdollError::InvalidHeader(
                "missing 'RIFF/RF64/BW64....WAVE' header".to_string(),
            ));
        }

        let mut pos = 12usize;
        let mut ds64_data_size: Option<u64> = None;
        let mut parsed_fmt: Option<(CodecId, AudioTrackParams, usize)> = None;
        let mut data_range: Option<(usize, usize)> = None;

        while pos + 8 <= data.len() {
            let chunk_id = &data[pos..pos + 4];
            let chunk_size_u32 =
                u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]);
            let chunk_size = chunk_size_u32 as usize;
            let payload_start = pos + 8;
            let clamped_size = chunk_size.min(data.len().saturating_sub(payload_start));
            let payload = &data[payload_start..payload_start + clamped_size];

            match chunk_id {
                b"ds64" if payload.len() >= 24 => {
                    // EBU Tech 3306 / ITU-R BS.2088 `ds64` chunk:
                    // bytes 0..8: riff_size (u64 LE), 8..16: data_size (u64 LE), 16..24: sample_count (u64 LE)
                    let dsize = u64::from_le_bytes([
                        payload[8],
                        payload[9],
                        payload[10],
                        payload[11],
                        payload[12],
                        payload[13],
                        payload[14],
                        payload[15],
                    ]);
                    ds64_data_size = Some(dsize);
                }
                b"fmt " => {
                    if payload.len() < 16 {
                        return Err(RagdollError::CorruptBitstream(
                            "truncated WAVE 'fmt ' chunk",
                        ));
                    }
                    let mut fmt_tag = u16::from_le_bytes([payload[0], payload[1]]);
                    let channels = u16::from_le_bytes([payload[2], payload[3]]) as u8;
                    let sample_rate =
                        u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]);
                    let block_align =
                        u16::from_le_bytes([payload[12], payload[13]]).max(1) as usize;
                    let bits_per_sample = u16::from_le_bytes([payload[14], payload[15]]) as u8;

                    if fmt_tag == WAVE_FORMAT_EXTENSIBLE && payload.len() >= 40 {
                        // SubFormat GUID first 2 bytes hold the underlying WAVE_FORMAT_PCM / IEEE_FLOAT tag
                        fmt_tag = u16::from_le_bytes([payload[24], payload[25]]);
                    }

                    let (codec_id, sample_format) = match (fmt_tag, bits_per_sample) {
                        (WAVE_FORMAT_PCM, 8) => (CodecId::PcmU8, SampleFormat::U8),
                        (WAVE_FORMAT_PCM, 16) => (CodecId::PcmS16Le, SampleFormat::S16le),
                        (WAVE_FORMAT_PCM, 24) => (CodecId::PcmS24Le, SampleFormat::S24le),
                        (WAVE_FORMAT_PCM, 32) => (CodecId::PcmS32Le, SampleFormat::S32le),
                        (WAVE_FORMAT_IEEE_FLOAT, 32) => (CodecId::PcmF32Le, SampleFormat::F32le),
                        _ => {
                            return Err(RagdollError::Unsupported(format!(
                                "unsupported WAVE format tag 0x{fmt_tag:04X} with {bits_per_sample} bits/sample"
                            )));
                        }
                    };

                    parsed_fmt = Some((
                        codec_id,
                        AudioTrackParams {
                            sample_rate: sample_rate.max(1),
                            channels: channels.max(1),
                            channel_layout: ChannelLayout::from_channels(channels.max(1)),
                            sample_format,
                            bits_per_coded_sample: bits_per_sample,
                        },
                        block_align,
                    ));
                }
                b"data" => {
                    let eff_size = if chunk_size_u32 == u32::MAX
                        && let Some(ds64_sz) = ds64_data_size
                    {
                        (ds64_sz as usize).min(data.len().saturating_sub(payload_start))
                    } else {
                        clamped_size
                    };
                    data_range = Some((payload_start, eff_size));
                    break;
                }
                _ => {}
            }

            // RIFF chunks are word-aligned (padded to even byte boundary)
            let padded = chunk_size + (chunk_size & 1);
            pos = payload_start.saturating_add(padded);
        }

        let Some((codec_id, audio_params, block_align)) = parsed_fmt else {
            return Err(RagdollError::InvalidHeader(
                "WAVE file missing 'fmt ' chunk".to_string(),
            ));
        };
        let Some((data_offset, raw_data_len)) = data_range else {
            return Err(RagdollError::InvalidHeader(
                "WAVE file missing 'data' chunk".to_string(),
            ));
        };

        let aligned_data_len = (raw_data_len / block_align) * block_align;
        let total_sample_frames = aligned_data_len / block_align;
        let fpp = frames_per_packet.max(1);
        let packet_count = total_sample_frames.div_ceil(fpp);

        let track = TrackInfo {
            index: 0,
            track_type: TrackType::Audio,
            codec_id,
            timebase: Rational64::new(1, audio_params.sample_rate as i64),
            duration_ticks: Some(total_sample_frames as i64),
            extra_data: Vec::new(),
            video: None,
            audio: Some(audio_params),
            sample_count: packet_count,
            total_bytes: aligned_data_len as u64,
        };

        Ok(Self {
            data,
            track,
            data_offset,
            data_len: aligned_data_len,
            block_align,
            frames_per_packet: fpp,
            cursor_bytes: 0,
            rf64: is_rf64,
        })
    }

    /// Returns `true` if the container uses an EBU Tech 3306 / ITU-R BS.2088 64-bit `RF64`/`BW64` header.
    #[inline]
    pub const fn is_rf64(&self) -> bool {
        self.rf64
    }

    #[inline]
    pub const fn track(&self) -> &TrackInfo {
        &self.track
    }

    /// Returns the entire raw PCM payload slice zero-copy.
    #[inline]
    pub fn pcm_data(&self) -> &'a [u8] {
        &self.data[self.data_offset..self.data_offset + self.data_len]
    }

    /// Emits the next chunk of PCM sample frames as a zero-copy [`PacketSlice<'a>`].
    #[inline]
    pub fn next_packet(&mut self) -> Result<Option<PacketSlice<'a>>> {
        if self.cursor_bytes >= self.data_len {
            return Ok(None);
        }
        let max_chunk = self.frames_per_packet * self.block_align;
        let chunk_bytes = (self.data_len - self.cursor_bytes).min(max_chunk);
        let start = self.data_offset + self.cursor_bytes;
        let pts_frames = (self.cursor_bytes / self.block_align) as i64;
        let dur_frames = (chunk_bytes / self.block_align) as i64;
        self.cursor_bytes += chunk_bytes;

        let ts = Timestamp::new(pts_frames, self.track.timebase);
        Ok(Some(PacketSlice::from_mmap(
            &self.data[start..start + chunk_bytes],
            0,
            ts,
            ts,
            dur_frames,
            PacketFlags::KEYFRAME,
            start as u64,
        )))
    }
}

/// Zero-copy / Vectored RIFF & RF64 WAVE (`.wav`) Muxer (`WAVE_FORMAT_PCM` & `WAVE_FORMAT_IEEE_FLOAT`).
pub struct WavMuxer<'a> {
    params: AudioTrackParams,
    chunks: Vec<&'a [u8]>,
    total_pcm_bytes: usize,
}

impl<'a> WavMuxer<'a> {
    pub fn new(params: AudioTrackParams) -> Self {
        Self {
            params,
            chunks: Vec::new(),
            total_pcm_bytes: 0,
        }
    }

    pub fn write_pcm_bytes(&mut self, pcm: &'a [u8]) {
        if !pcm.is_empty() {
            self.chunks.push(pcm);
            self.total_pcm_bytes += pcm.len();
        }
    }

    pub fn finish<W: Write>(self, writer: W) -> Result<u64> {
        if self.total_pcm_bytes as u64 > (u32::MAX as u64 - 64) {
            return self.finish_rf64(writer);
        }
        let mut vw = VectoredWriter::new(writer);
        let channels = self.params.channels.max(1) as u16;
        let sample_rate = self.params.sample_rate.max(1);
        let bytes_per_sample = self.params.sample_format.bytes_per_sample() as u16;
        let bits_per_sample = bytes_per_sample * 8;
        let block_align = channels * bytes_per_sample;
        let byte_rate = sample_rate * (block_align as u32);
        let is_float = self.params.sample_format.is_float();
        let fmt_tag = if is_float {
            WAVE_FORMAT_IEEE_FLOAT
        } else {
            WAVE_FORMAT_PCM
        };

        let data_size = self.total_pcm_bytes as u32;
        let pad_byte = (self.total_pcm_bytes & 1) as u32;
        let fact_len = if is_float { 12u32 } else { 0u32 };
        let riff_size = 4 + (8 + 16) + fact_len + (8 + data_size + pad_byte);
        let sample_frames = (self.total_pcm_bytes / (block_align.max(1) as usize)) as u32;

        let mut hdr = Vec::with_capacity(56);
        hdr.extend_from_slice(b"RIFF");
        hdr.extend_from_slice(&riff_size.to_le_bytes());
        hdr.extend_from_slice(b"WAVE");
        hdr.extend_from_slice(b"fmt ");
        hdr.extend_from_slice(&16u32.to_le_bytes());
        hdr.extend_from_slice(&fmt_tag.to_le_bytes());
        hdr.extend_from_slice(&channels.to_le_bytes());
        hdr.extend_from_slice(&sample_rate.to_le_bytes());
        hdr.extend_from_slice(&byte_rate.to_le_bytes());
        hdr.extend_from_slice(&block_align.to_le_bytes());
        hdr.extend_from_slice(&bits_per_sample.to_le_bytes());
        if is_float {
            // Mandatory 12-byte `fact` chunk for non-PCM WAVE_FORMAT_IEEE_FLOAT per Microsoft WAVE spec & libavformat/wavenc.c
            hdr.extend_from_slice(b"fact");
            hdr.extend_from_slice(&4u32.to_le_bytes());
            hdr.extend_from_slice(&sample_frames.to_le_bytes());
        }
        hdr.extend_from_slice(b"data");
        hdr.extend_from_slice(&data_size.to_le_bytes());

        vw.write_all_counted(&hdr)?;
        for batch in self.chunks.chunks(8) {
            vw.write_vectored_all(batch)?;
        }
        if pad_byte != 0 {
            vw.write_all_counted(&[0u8])?;
        }
        vw.flush()?;
        Ok(vw.bytes_written())
    }

    /// Writes an EBU Tech 3306 / ITU-R BS.2088 64-bit `RF64` WAVE container with a 28-byte `ds64` chunk.
    pub fn finish_rf64<W: Write>(self, writer: W) -> Result<u64> {
        let mut vw = VectoredWriter::new(writer);
        let channels = self.params.channels.max(1) as u16;
        let sample_rate = self.params.sample_rate.max(1);
        let bytes_per_sample = self.params.sample_format.bytes_per_sample() as u16;
        let bits_per_sample = bytes_per_sample * 8;
        let block_align = channels * bytes_per_sample;
        let byte_rate = sample_rate * (block_align as u32);
        let is_float = self.params.sample_format.is_float();
        let fmt_tag = if is_float {
            WAVE_FORMAT_IEEE_FLOAT
        } else {
            WAVE_FORMAT_PCM
        };

        let data_size_u64 = self.total_pcm_bytes as u64;
        let pad_byte = (self.total_pcm_bytes & 1) as u64;
        let sample_frames_u64 = (self.total_pcm_bytes / (block_align.max(1) as usize)) as u64;
        let fact_len = if is_float { 12u64 } else { 0u64 };
        // RF64 size = 4 ("WAVE") + 36 (8 + 28 ds64) + 24 (8 + 16 fmt) + fact_len + 8 + data_size + pad
        let rf64_size = 4 + 36 + 24 + fact_len + 8 + data_size_u64 + pad_byte;

        let mut hdr = Vec::with_capacity(92);
        hdr.extend_from_slice(b"RF64");
        hdr.extend_from_slice(&u32::MAX.to_le_bytes()); // 0xFFFF_FFFF triggers ds64 lookup
        hdr.extend_from_slice(b"WAVE");
        // `ds64` chunk (28 bytes payload)
        hdr.extend_from_slice(b"ds64");
        hdr.extend_from_slice(&28u32.to_le_bytes());
        hdr.extend_from_slice(&rf64_size.to_le_bytes());
        hdr.extend_from_slice(&data_size_u64.to_le_bytes());
        hdr.extend_from_slice(&sample_frames_u64.to_le_bytes());
        hdr.extend_from_slice(&0u32.to_le_bytes()); // table_length = 0
        // `fmt ` chunk
        hdr.extend_from_slice(b"fmt ");
        hdr.extend_from_slice(&16u32.to_le_bytes());
        hdr.extend_from_slice(&fmt_tag.to_le_bytes());
        hdr.extend_from_slice(&channels.to_le_bytes());
        hdr.extend_from_slice(&sample_rate.to_le_bytes());
        hdr.extend_from_slice(&byte_rate.to_le_bytes());
        hdr.extend_from_slice(&block_align.to_le_bytes());
        hdr.extend_from_slice(&bits_per_sample.to_le_bytes());
        if is_float {
            hdr.extend_from_slice(b"fact");
            hdr.extend_from_slice(&4u32.to_le_bytes());
            hdr.extend_from_slice(&(sample_frames_u64.min(u32::MAX as u64) as u32).to_le_bytes());
        }
        // `data` chunk with 0xFFFF_FFFF 32-bit size sentinel
        hdr.extend_from_slice(b"data");
        hdr.extend_from_slice(&u32::MAX.to_le_bytes());

        vw.write_all_counted(&hdr)?;
        for batch in self.chunks.chunks(8) {
            vw.write_vectored_all(batch)?;
        }
        if pad_byte != 0 {
            vw.write_all_counted(&[0u8])?;
        }
        vw.flush()?;
        Ok(vw.bytes_written())
    }
}
