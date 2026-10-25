use std::fmt;
use std::io;

pub type Result<T> = std::result::Result<T, RagdollError>;

#[derive(Debug)]
pub enum RagdollError {
    BitstreamEndOfData,
    InvalidSyntax(&'static str),
    CorruptBitstream(&'static str),
    InvalidData(String),
    InvalidHeader(String),
    UnsupportedFeature(&'static str),
    Unsupported(String),
    InvalidDimensions {
        width: usize,
        height: usize,
    },
    DimensionMismatch {
        expected_w: usize,
        expected_h: usize,
        actual_w: usize,
        actual_h: usize,
    },
    InvalidTimestamp(&'static str),
    ChecksumMismatch {
        expected: u32,
        actual: u32,
    },
    Io(io::Error),
}

impl fmt::Display for RagdollError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BitstreamEndOfData => write!(f, "unexpected end of bitstream"),
            Self::InvalidSyntax(msg) => write!(f, "invalid bitstream syntax: {msg}"),
            Self::CorruptBitstream(msg) => write!(f, "corrupt bitstream or container: {msg}"),
            Self::InvalidData(msg) => write!(f, "invalid media data: {msg}"),
            Self::InvalidHeader(msg) => write!(f, "invalid container header: {msg}"),
            Self::UnsupportedFeature(msg) => write!(f, "unsupported media feature: {msg}"),
            Self::Unsupported(msg) => write!(f, "unsupported media format: {msg}"),
            Self::InvalidDimensions { width, height } => {
                write!(f, "invalid frame dimensions: {width}x{height}")
            }
            Self::DimensionMismatch {
                expected_w,
                expected_h,
                actual_w,
                actual_h,
            } => write!(
                f,
                "dimension mismatch (expected {expected_w}x{expected_h}, got {actual_w}x{actual_h})"
            ),
            Self::InvalidTimestamp(msg) => write!(f, "invalid timestamp or timebase: {msg}"),
            Self::ChecksumMismatch { expected, actual } => write!(
                f,
                "checksum mismatch (expected 0x{expected:08x}, got 0x{actual:08x})"
            ),
            Self::Io(err) => write!(f, "I/O error: {err}"),
        }
    }
}

impl std::error::Error for RagdollError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for RagdollError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}
