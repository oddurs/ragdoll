use crate::color::Colorimetry;
use crate::format::{ChannelLayout, PixelFormat, SampleFormat};
use crate::frame::AlignedBuffer;
use crate::time::{Rational64, Timestamp};
use std::fmt;

/// Compact 24-byte (`#[repr(C)]`) sample index entry used by Ragdoll container demuxers.
///
/// Packing offset, PTS, byte size, composition time offset (CTS), and keyframe flag
/// into 24 bytes allows 2.66 samples per 64-byte CPU cache line—enabling sub-microsecond
/// $O(\log N)$ keyframe binary search across multi-hour 4K movies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct SampleEntry {
    /// Exact byte offset of the sample payload in the memory-mapped container file.
    pub offset: u64,
    /// Decoding or presentation timestamp in the track's native timebase.
    pub pts: i64,
    /// Payload size in bytes.
    pub size: u32,
    /// Bit 31: `is_keyframe` (1 = sync/IDR sample).
    /// Bits 0..31: signed 31-bit composition time offset (`cts_delta = pts - dts`).
    pub cts_and_flags: u32,
}

const _: () = assert!(
    std::mem::size_of::<SampleEntry>() == 24,
    "SampleEntry must be exactly 24 bytes"
);

impl SampleEntry {
    const KEYFRAME_BIT: u32 = 1 << 31;
    const CTS_SIGN_BIT: u32 = 1 << 30;
    const CTS_MASK: u32 = 0x7FFF_FFFF;

    #[inline]
    pub const fn new(offset: u64, pts: i64, size: u32, cts_delta: i32, is_keyframe: bool) -> Self {
        let masked_cts = (cts_delta as u32) & Self::CTS_MASK;
        let flags = if is_keyframe { Self::KEYFRAME_BIT } else { 0 };
        Self {
            offset,
            pts,
            size,
            cts_and_flags: flags | masked_cts,
        }
    }

    #[inline]
    pub const fn is_keyframe(self) -> bool {
        (self.cts_and_flags & Self::KEYFRAME_BIT) != 0
    }

    /// Extracts the sign-extended 31-bit composition time offset (`PTS - DTS`).
    #[inline]
    pub const fn cts_delta(self) -> i32 {
        let raw = self.cts_and_flags & Self::CTS_MASK;
        if (raw & Self::CTS_SIGN_BIT) != 0 {
            (raw | Self::KEYFRAME_BIT) as i32
        } else {
            raw as i32
        }
    }

    #[inline]
    pub const fn dts(self) -> i64 {
        self.pts.saturating_sub(self.cts_delta() as i64)
    }
}

/// Packet metadata bitflags.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PacketFlags(pub u8);

impl PacketFlags {
    pub const NONE: Self = Self(0);
    pub const KEYFRAME: Self = Self(1 << 0);
    pub const CORRUPT: Self = Self(1 << 1);
    pub const DISCARD: Self = Self(1 << 2);
    pub const PARAMETER_SET: Self = Self(1 << 3);

    #[inline]
    pub const fn is_keyframe(self) -> bool {
        (self.0 & Self::KEYFRAME.0) != 0
    }

    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

/// Zero-copy or aligned-owned packet payload storage.
#[derive(Clone, Debug)]
pub enum PacketPayload<'a> {
    /// Zero-copy slice borrowing directly from a memory-mapped file (`MmapSource`).
    Borrowed(&'a [u8]),
    /// Owned 64-byte-aligned buffer for newly encoded packets.
    Owned(AlignedBuffer),
}

impl<'a> PacketPayload<'a> {
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::Borrowed(slice) => slice,
            Self::Owned(buf) => buf.as_slice(),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    #[inline]
    pub const fn is_zero_copy(&self) -> bool {
        matches!(self, Self::Borrowed(_))
    }

    pub fn into_owned(self) -> PacketPayload<'static> {
        match self {
            Self::Borrowed(s) => PacketPayload::Owned(AlignedBuffer::from_slice(s)),
            Self::Owned(b) => PacketPayload::Owned(b),
        }
    }
}

impl<'a> AsRef<[u8]> for PacketPayload<'a> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

/// A compressed media packet with zero-copy lifetime `'a` into the source container.
#[derive(Clone, Debug)]
pub struct PacketSlice<'a> {
    pub payload: PacketPayload<'a>,
    pub track_index: u32,
    pub pts: Timestamp,
    pub dts: Timestamp,
    pub duration: i64,
    pub flags: PacketFlags,
    pub file_offset: u64,
}

impl<'a> PacketSlice<'a> {
    #[inline]
    pub const fn from_mmap(
        bytes: &'a [u8],
        track_index: u32,
        pts: Timestamp,
        dts: Timestamp,
        duration: i64,
        flags: PacketFlags,
        file_offset: u64,
    ) -> Self {
        Self {
            payload: PacketPayload::Borrowed(bytes),
            track_index,
            pts,
            dts,
            duration,
            flags,
            file_offset,
        }
    }

    #[inline]
    pub fn from_owned(
        buf: AlignedBuffer,
        track_index: u32,
        pts: Timestamp,
        dts: Timestamp,
        duration: i64,
        flags: PacketFlags,
    ) -> Self {
        Self {
            payload: PacketPayload::Owned(buf),
            track_index,
            pts,
            dts,
            duration,
            flags,
            file_offset: 0,
        }
    }

    #[inline]
    pub fn data(&self) -> &[u8] {
        self.payload.as_slice()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.payload.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.payload.is_empty()
    }

    #[inline]
    pub const fn is_keyframe(&self) -> bool {
        self.flags.is_keyframe()
    }
}

/// Codec identifiers supported by Ragdoll's container and codec engines.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CodecId {
    // Video
    H264,
    Hevc,
    Av1,
    Vp9,
    RagdollIntra,
    RawVideo,
    Png,
    Jpeg,
    // Audio
    Flac,
    Aac,
    Opus,
    PcmU8,
    PcmS16Le,
    PcmS24Le,
    PcmS32Le,
    PcmF32Le,
    // Subtitles
    WebVtt,
    SubRip,
}

impl CodecId {
    #[inline]
    pub const fn is_video(self) -> bool {
        matches!(
            self,
            Self::H264
                | Self::Hevc
                | Self::Av1
                | Self::Vp9
                | Self::RagdollIntra
                | Self::RawVideo
                | Self::Png
                | Self::Jpeg
        )
    }

    #[inline]
    pub const fn is_audio(self) -> bool {
        matches!(
            self,
            Self::Flac
                | Self::Aac
                | Self::Opus
                | Self::PcmU8
                | Self::PcmS16Le
                | Self::PcmS24Le
                | Self::PcmS32Le
                | Self::PcmF32Le
        )
    }

    #[inline]
    pub const fn is_subtitle(self) -> bool {
        matches!(self, Self::WebVtt | Self::SubRip)
    }

    #[inline]
    pub const fn short_name(self) -> &'static str {
        match self {
            Self::H264 => "h264",
            Self::Hevc => "hevc",
            Self::Av1 => "av1",
            Self::Vp9 => "vp9",
            Self::RagdollIntra => "rgdi",
            Self::RawVideo => "rawvideo",
            Self::Png => "png",
            Self::Jpeg => "mjpeg",
            Self::Flac => "flac",
            Self::Aac => "aac",
            Self::Opus => "opus",
            Self::PcmU8 => "pcm_u8",
            Self::PcmS16Le => "pcm_s16le",
            Self::PcmS24Le => "pcm_s24le",
            Self::PcmS32Le => "pcm_s32le",
            Self::PcmF32Le => "pcm_f32le",
            Self::WebVtt => "webvtt",
            Self::SubRip => "subrip",
        }
    }
}

impl fmt::Display for CodecId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.short_name())
    }
}

/// Stream media type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TrackType {
    Video,
    Audio,
    Subtitle,
    Data,
}

/// Video stream parameters discovered by a container demuxer or codec parser.
#[derive(Clone, Debug)]
pub struct VideoTrackParams {
    pub width: u32,
    pub height: u32,
    pub pixel_format: PixelFormat,
    pub colorimetry: Colorimetry,
    pub frame_rate: Rational64,
    pub bit_depth: u8,
    pub profile: Option<String>,
    pub level: Option<String>,
}

impl Default for VideoTrackParams {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            pixel_format: PixelFormat::Yuv420p,
            colorimetry: Colorimetry::default(),
            frame_rate: Rational64::FPS_30,
            bit_depth: 8,
            profile: None,
            level: None,
        }
    }
}

/// Audio stream parameters discovered by a container demuxer or codec parser.
#[derive(Clone, Debug)]
pub struct AudioTrackParams {
    pub sample_rate: u32,
    pub channels: u8,
    pub channel_layout: ChannelLayout,
    pub sample_format: SampleFormat,
    pub bits_per_coded_sample: u8,
}

impl Default for AudioTrackParams {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            channels: 2,
            channel_layout: ChannelLayout::Stereo,
            sample_format: SampleFormat::S16le,
            bits_per_coded_sample: 16,
        }
    }
}

/// Complete track metadata and codec configuration (`avcC`, `av1C`, `STREAMINFO`, etc.).
#[derive(Clone, Debug)]
pub struct TrackInfo {
    pub index: u32,
    pub track_type: TrackType,
    pub codec_id: CodecId,
    pub timebase: Rational64,
    pub duration_ticks: Option<i64>,
    pub extra_data: Vec<u8>,
    pub video: Option<VideoTrackParams>,
    pub audio: Option<AudioTrackParams>,
    pub sample_count: usize,
    pub total_bytes: u64,
}
