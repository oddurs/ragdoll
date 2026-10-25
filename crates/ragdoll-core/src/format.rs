use std::fmt;

/// Pixel memory layout and component representation for video frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum PixelFormat {
    /// Planar 8-bit YUV 4:2:0 (3 planes: Y, U, V).
    #[default]
    Yuv420p,
    /// Planar 8-bit YUV 4:2:2 (3 planes: Y, U, V).
    Yuv422p,
    /// Planar 8-bit YUV 4:4:4 (3 planes: Y, U, V).
    Yuv444p,
    /// Semi-planar 8-bit YUV 4:2:0 (2 planes: Y, interleaved UV). Hardware decoder native format.
    Nv12,
    /// Planar 10-bit little-endian YUV 4:2:0 stored in 16-bit words (3 planes: Y, U, V).
    Yuv420p10le,
    /// Packed 8-bit RGB (1 plane, 3 bytes per pixel: R, G, B).
    Rgb24,
    /// Packed 8-bit RGBA (1 plane, 4 bytes per pixel: R, G, B, A).
    Rgba8,
    /// Packed 8-bit BGRA (1 plane, 4 bytes per pixel: B, G, R, A).
    Bgra8,
    /// Planar 8-bit Grayscale / Luma only (1 plane, 1 byte per pixel).
    Gray8,
    /// Packed 32-bit floating-point linear RGB (1 plane, 12 bytes per pixel).
    RgbF32,
    /// Packed 32-bit floating-point perceptual Oklab (1 plane, 12 bytes per pixel: L, a, b).
    OklabF32,
}

impl PixelFormat {
    #[inline]
    pub const fn num_planes(self) -> usize {
        match self {
            Self::Yuv420p | Self::Yuv422p | Self::Yuv444p | Self::Yuv420p10le => 3,
            Self::Nv12 => 2,
            Self::Rgb24
            | Self::Rgba8
            | Self::Bgra8
            | Self::Gray8
            | Self::RgbF32
            | Self::OklabF32 => 1,
        }
    }

    #[inline]
    pub const fn bit_depth(self) -> u8 {
        match self {
            Self::Yuv420p10le => 10,
            Self::RgbF32 | Self::OklabF32 => 32,
            _ => 8,
        }
    }

    /// Returns `(sub_x, sub_y)` log2 chroma shift for chroma planes.
    #[inline]
    pub const fn chroma_shift(self) -> (u8, u8) {
        match self {
            Self::Yuv420p | Self::Nv12 | Self::Yuv420p10le => (1, 1),
            Self::Yuv422p => (1, 0),
            Self::Yuv444p
            | Self::Rgb24
            | Self::Rgba8
            | Self::Bgra8
            | Self::Gray8
            | Self::RgbF32
            | Self::OklabF32 => (0, 0),
        }
    }

    /// Returns `(plane_width_samples, plane_height_rows)` for a given `plane_index`.
    #[inline]
    pub const fn plane_dimensions(
        self,
        width: usize,
        height: usize,
        plane_index: usize,
    ) -> (usize, usize) {
        if plane_index == 0 {
            return (width, height);
        }
        let (sx, sy) = self.chroma_shift();
        let cw = (width + ((1 << sx) - 1)) >> sx;
        let ch = (height + ((1 << sy) - 1)) >> sy;
        (cw, ch)
    }

    /// Bytes per horizontal sample unit in `plane_index`.
    /// Note: For `Nv12` plane 1, each UV pair is 2 bytes (`cw` pairs -> `2 * cw` bytes).
    #[inline]
    pub const fn plane_bytes_per_sample(self, plane_index: usize) -> usize {
        match self {
            Self::Yuv420p | Self::Yuv422p | Self::Yuv444p | Self::Gray8 => 1,
            Self::Nv12 => {
                if plane_index == 0 {
                    1
                } else {
                    2
                }
            }
            Self::Yuv420p10le => 2,
            Self::Rgb24 => 3,
            Self::Rgba8 | Self::Bgra8 => 4,
            Self::RgbF32 | Self::OklabF32 => 12,
        }
    }

    /// Unpadded active byte width of a single scanline in `plane_index`.
    #[inline]
    pub const fn plane_row_bytes(self, width: usize, plane_index: usize) -> usize {
        let (pw, _) = self.plane_dimensions(width, 1, plane_index);
        pw * self.plane_bytes_per_sample(plane_index)
    }

    /// Tightly-packed total byte size of a frame (ignoring SIMD stride padding).
    #[inline]
    pub const fn packed_frame_bytes(self, width: usize, height: usize) -> usize {
        let mut total = 0;
        let mut p = 0;
        let n = self.num_planes();
        while p < n {
            let (pw, ph) = self.plane_dimensions(width, height, p);
            total += pw * ph * self.plane_bytes_per_sample(p);
            p += 1;
        }
        total
    }
}

impl fmt::Display for PixelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Yuv420p => "yuv420p",
            Self::Yuv422p => "yuv422p",
            Self::Yuv444p => "yuv444p",
            Self::Nv12 => "nv12",
            Self::Yuv420p10le => "yuv420p10le",
            Self::Rgb24 => "rgb24",
            Self::Rgba8 => "rgba8",
            Self::Bgra8 => "bgra8",
            Self::Gray8 => "gray8",
            Self::RgbF32 => "rgbf32",
            Self::OklabF32 => "oklabf32",
        };
        write!(f, "{name}")
    }
}

/// Audio PCM sample format (little-endian native).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum SampleFormat {
    /// Unsigned 8-bit PCM (bias 128).
    U8,
    /// Signed 16-bit little-endian PCM.
    #[default]
    S16le,
    /// Signed 24-bit packed little-endian PCM (3 bytes per sample).
    S24le,
    /// Signed 32-bit little-endian PCM.
    S32le,
    /// 32-bit IEEE 754 floating-point PCM (`[-1.0, 1.0]`).
    F32le,
}

impl SampleFormat {
    #[inline]
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::S16le => 2,
            Self::S24le => 3,
            Self::S32le | Self::F32le => 4,
        }
    }

    #[inline]
    pub const fn bits_per_sample(self) -> u8 {
        (self.bytes_per_sample() * 8) as u8
    }

    #[inline]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32le)
    }
}

impl fmt::Display for SampleFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::U8 => "u8",
            Self::S16le => "s16le",
            Self::S24le => "s24le",
            Self::S32le => "s32le",
            Self::F32le => "f32le",
        };
        write!(f, "{s}")
    }
}

/// Audio speaker / channel layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ChannelLayout {
    Mono,
    #[default]
    Stereo,
    Surround2_1,
    Quad,
    Surround5_0,
    Surround5_1,
    Surround7_1,
    Custom(u8),
}

impl ChannelLayout {
    #[inline]
    pub const fn channels(self) -> u8 {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Surround2_1 => 3,
            Self::Quad => 4,
            Self::Surround5_0 => 5,
            Self::Surround5_1 => 6,
            Self::Surround7_1 => 8,
            Self::Custom(n) => n,
        }
    }

    #[inline]
    pub const fn from_channels(channels: u8) -> Self {
        match channels {
            1 => Self::Mono,
            2 => Self::Stereo,
            3 => Self::Surround2_1,
            4 => Self::Quad,
            5 => Self::Surround5_0,
            6 => Self::Surround5_1,
            8 => Self::Surround7_1,
            n => Self::Custom(n),
        }
    }
}
