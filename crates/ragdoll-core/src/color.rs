use std::fmt;

/// CIE 1931 $(x, y)$ chromaticity coordinates for R, G, B primaries and reference white point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Chromaticities {
    pub red: (f64, f64),
    pub green: (f64, f64),
    pub blue: (f64, f64),
    pub white: (f64, f64),
}

/// ITU-T H.273 / ISO/IEC 23091-2 (CICP) Color Primaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum ColorPrimaries {
    Reserved0 = 0,
    #[default]
    BT709 = 1,
    Unspecified = 2,
    BT470M = 4,
    BT470BG = 5,
    BT601 = 6,
    SMPTE240M = 7,
    Film = 8,
    BT2020 = 9,
    SMPTE428 = 10,
    SMPTE431 = 11,
    DisplayP3 = 12,
}

impl ColorPrimaries {
    #[inline]
    pub const fn from_cicp(code: u8) -> Self {
        match code {
            1 => Self::BT709,
            4 => Self::BT470M,
            5 => Self::BT470BG,
            6 => Self::BT601,
            7 => Self::SMPTE240M,
            8 => Self::Film,
            9 => Self::BT2020,
            10 => Self::SMPTE428,
            11 => Self::SMPTE431,
            12 => Self::DisplayP3,
            _ => Self::Unspecified,
        }
    }

    #[inline]
    pub const fn cicp_code(self) -> u8 {
        self as u8
    }

    #[inline]
    pub const fn chromaticities(self) -> Chromaticities {
        match self {
            Self::BT709 | Self::Unspecified | Self::Reserved0 => Chromaticities {
                red: (0.640, 0.330),
                green: (0.300, 0.600),
                blue: (0.150, 0.060),
                white: (0.3127, 0.3290), // D65
            },
            Self::BT601 | Self::SMPTE240M => Chromaticities {
                red: (0.630, 0.340),
                green: (0.310, 0.595),
                blue: (0.155, 0.070),
                white: (0.3127, 0.3290), // D65
            },
            Self::BT470BG => Chromaticities {
                red: (0.640, 0.330),
                green: (0.290, 0.600),
                blue: (0.150, 0.060),
                white: (0.3127, 0.3290),
            },
            Self::BT470M => Chromaticities {
                red: (0.670, 0.330),
                green: (0.210, 0.710),
                blue: (0.140, 0.080),
                white: (0.310, 0.316), // Illuminant C
            },
            Self::Film => Chromaticities {
                red: (0.681, 0.319),
                green: (0.243, 0.692),
                blue: (0.145, 0.049),
                white: (0.310, 0.316),
            },
            Self::BT2020 => Chromaticities {
                red: (0.708, 0.292),
                green: (0.170, 0.797),
                blue: (0.131, 0.046),
                white: (0.3127, 0.3290), // D65
            },
            Self::DisplayP3 => Chromaticities {
                red: (0.680, 0.320),
                green: (0.265, 0.690),
                blue: (0.150, 0.060),
                white: (0.3127, 0.3290), // D65
            },
            Self::SMPTE431 => Chromaticities {
                red: (0.680, 0.320),
                green: (0.265, 0.690),
                blue: (0.150, 0.060),
                white: (0.314, 0.351), // DCI P3 theater white
            },
            Self::SMPTE428 => Chromaticities {
                red: (1.0, 0.0),
                green: (0.0, 1.0),
                blue: (0.0, 0.0),
                white: (1.0 / 3.0, 1.0 / 3.0),
            },
        }
    }
}

/// ITU-T H.273 / ISO/IEC 23091-2 (CICP) Opto-Electronic Transfer Characteristics (EOTF / OETF).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum TransferCharacteristics {
    #[default]
    BT709 = 1,
    Unspecified = 2,
    Gamma22 = 4,
    Gamma28 = 5,
    BT601 = 6,
    SMPTE240M = 7,
    Linear = 8,
    SRGB = 13,
    BT2020_10 = 14,
    BT2020_12 = 15,
    Smpte2084PQ = 16,
    Smpte428 = 17,
    ArbStdB67HLG = 18,
}

impl TransferCharacteristics {
    #[inline]
    pub const fn from_cicp(code: u8) -> Self {
        match code {
            1 => Self::BT709,
            4 => Self::Gamma22,
            5 => Self::Gamma28,
            6 => Self::BT601,
            7 => Self::SMPTE240M,
            8 => Self::Linear,
            13 => Self::SRGB,
            14 => Self::BT2020_10,
            15 => Self::BT2020_12,
            16 => Self::Smpte2084PQ,
            17 => Self::Smpte428,
            18 => Self::ArbStdB67HLG,
            _ => Self::Unspecified,
        }
    }

    #[inline]
    pub const fn is_hdr(self) -> bool {
        matches!(self, Self::Smpte2084PQ | Self::ArbStdB67HLG)
    }
}

/// ITU-T H.273 / ISO/IEC 23091-2 (CICP) Matrix Coefficients + perceptual Oklab extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum MatrixCoefficients {
    Identity = 0,
    #[default]
    BT709 = 1,
    Unspecified = 2,
    Fcc = 4,
    BT470BG = 5,
    BT601 = 6,
    SMPTE240M = 7,
    YCgCo = 8,
    BT2020NCL = 9,
    BT2020CL = 10,
    ICtCp = 14,
    /// Ragdoll perceptual Oklab color space extension.
    Oklab = 250,
}

impl MatrixCoefficients {
    #[inline]
    pub const fn from_cicp(code: u8) -> Self {
        match code {
            0 => Self::Identity,
            1 => Self::BT709,
            4 => Self::Fcc,
            5 => Self::BT470BG,
            6 => Self::BT601,
            7 => Self::SMPTE240M,
            8 => Self::YCgCo,
            9 => Self::BT2020NCL,
            10 => Self::BT2020CL,
            14 => Self::ICtCp,
            250 => Self::Oklab,
            _ => Self::Unspecified,
        }
    }

    /// Returns exact ITU luma weights $(K_r, K_g, K_b)$ such that $K_r + K_g + K_b = 1.0$.
    #[inline]
    pub const fn luma_coefficients(self) -> (f64, f64, f64) {
        match self {
            Self::BT709 | Self::Unspecified | Self::Identity | Self::YCgCo | Self::Oklab => {
                (0.2126, 0.7152, 0.0722)
            }
            Self::BT601 | Self::BT470BG => (0.299, 0.587, 0.114),
            Self::Fcc => (0.30, 0.59, 0.11),
            Self::SMPTE240M => (0.212, 0.701, 0.087),
            Self::BT2020NCL | Self::BT2020CL | Self::ICtCp => (0.2627, 0.6780, 0.0593),
        }
    }
}

/// Quantization range for luma and chroma samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ColorRange {
    /// Studio / Broadcast range (8-bit Y: 16..=235, UV: 16..=240; 10-bit Y: 64..=940, UV: 64..=960).
    #[default]
    Limited,
    /// Full digital range (8-bit: 0..=255; 10-bit: 0..=1023).
    Full,
}

/// Chroma sample siting relative to the luma grid in 4:2:0 / 4:2:2 formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum ChromaLocation {
    /// Horizontally co-sited with even luma columns, vertically centered between rows (MPEG-2/4, H.264 default).
    #[default]
    Left = 0,
    /// Centered horizontally and vertically between 2x2 luma samples (MPEG-1, JPEG/JFIF).
    Center = 1,
    /// Co-sited with top-left luma sample (BT.2020 / UHD Blu-ray / HDR10 required siting).
    TopLeft = 2,
    Top = 3,
    BottomLeft = 4,
    Bottom = 5,
}

/// Complete colorimetry descriptor carried by every `VideoFrame` and video track.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Colorimetry {
    pub primaries: ColorPrimaries,
    pub transfer: TransferCharacteristics,
    pub matrix: MatrixCoefficients,
    pub range: ColorRange,
    pub chroma_location: ChromaLocation,
}

impl Colorimetry {
    /// Standard HD broadcast & web video (Rec. 709, Limited range, Left chroma siting).
    #[inline]
    pub const fn bt709_limited() -> Self {
        Self {
            primaries: ColorPrimaries::BT709,
            transfer: TransferCharacteristics::BT709,
            matrix: MatrixCoefficients::BT709,
            range: ColorRange::Limited,
            chroma_location: ChromaLocation::Left,
        }
    }

    /// Full-range Rec. 709.
    #[inline]
    pub const fn bt709_full() -> Self {
        Self {
            primaries: ColorPrimaries::BT709,
            transfer: TransferCharacteristics::BT709,
            matrix: MatrixCoefficients::BT709,
            range: ColorRange::Full,
            chroma_location: ChromaLocation::Center,
        }
    }

    /// Standard Definition broadcast video (Rec. 601 / SMPTE 170M, Limited range).
    #[inline]
    pub const fn bt601_limited() -> Self {
        Self {
            primaries: ColorPrimaries::BT601,
            transfer: TransferCharacteristics::BT601,
            matrix: MatrixCoefficients::BT601,
            range: ColorRange::Limited,
            chroma_location: ChromaLocation::Left,
        }
    }

    /// UHD 4K HDR10 (Rec. 2020 primaries, SMPTE ST 2084 PQ transfer, Non-Constant Luminance, TopLeft chroma siting).
    #[inline]
    pub const fn bt2020_pq_hdr10() -> Self {
        Self {
            primaries: ColorPrimaries::BT2020,
            transfer: TransferCharacteristics::Smpte2084PQ,
            matrix: MatrixCoefficients::BT2020NCL,
            range: ColorRange::Limited,
            chroma_location: ChromaLocation::TopLeft,
        }
    }

    /// UHD 4K SDR (Rec. 2020 NCL, Limited range).
    #[inline]
    pub const fn bt2020_sdr() -> Self {
        Self {
            primaries: ColorPrimaries::BT2020,
            transfer: TransferCharacteristics::BT2020_10,
            matrix: MatrixCoefficients::BT2020NCL,
            range: ColorRange::Limited,
            chroma_location: ChromaLocation::TopLeft,
        }
    }

    /// Standard full-range sRGB (Identity RGB matrix).
    #[inline]
    pub const fn srgb() -> Self {
        Self {
            primaries: ColorPrimaries::BT709,
            transfer: TransferCharacteristics::SRGB,
            matrix: MatrixCoefficients::Identity,
            range: ColorRange::Full,
            chroma_location: ChromaLocation::Center,
        }
    }

    /// Perceptual Oklab color space.
    #[inline]
    pub const fn oklab() -> Self {
        Self {
            primaries: ColorPrimaries::BT709,
            transfer: TransferCharacteristics::Linear,
            matrix: MatrixCoefficients::Oklab,
            range: ColorRange::Full,
            chroma_location: ChromaLocation::Center,
        }
    }
}

impl Default for Colorimetry {
    fn default() -> Self {
        Self::bt709_limited()
    }
}

impl fmt::Display for Colorimetry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?}/{:?}/{:?} ({:?}, chroma={:?})",
            self.primaries, self.transfer, self.matrix, self.range, self.chroma_location
        )
    }
}
