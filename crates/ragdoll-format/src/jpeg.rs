//! **`0047`**: Pure-Rust Baseline DCT (`SOF0`) JPEG (`.jpg` / `.jpeg`, ITU-T T.81 / JFIF 1.01)
//! Decoder & Encoder with $8\times8$ DCT/IDCT, ITU-T T.81 Annex K Huffman & Quantization tables,
//! `0xFF 0x00` byte stuffing, and full-range BT.601 (ITU-T T.871) YCbCr $\leftrightarrow$ RGB24 conversion.

#![allow(
    clippy::needless_range_loop,
    clippy::approx_constant,
    clippy::excessive_precision
)]

use std::io::Write;

use crate::png::ensure_rgb24;
use ragdoll_core::{Colorimetry, PixelFormat, RagdollError, Result, VideoFrame};

/// Returns `true` if `data` begins with a JPEG SOI marker (`0xFF 0xD8 0xFF`).
#[inline]
pub fn is_jpeg_bytes(data: &[u8]) -> bool {
    data.len() >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF
}

/// Zig-zag scan order mapping $8\times8$ natural raster index `(y * 8 + x)` from zig-zag index `0..64`:
/// `natural_idx = ZIGZAG_TO_NATURAL[zz_idx]`.
pub const ZIGZAG_TO_NATURAL: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// ITU-T T.81 Annex K Table K.1 Luminance quantization table (natural $8\times8$ raster order).
pub const STD_LUMA_QUANT_NATURAL: [u8; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

/// ITU-T T.81 Annex K Table K.2 Chrominance quantization table (natural $8\times8$ raster order).
pub const STD_CHROMA_QUANT_NATURAL: [u8; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

// ITU-T T.81 Annex K Huffman tables:
const STD_DC_LUMA_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const STD_DC_LUMA_VALS: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

const STD_DC_CHROMA_BITS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
const STD_DC_CHROMA_VALS: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

const STD_AC_LUMA_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];
const STD_AC_LUMA_VALS: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];

const STD_AC_CHROMA_BITS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
const STD_AC_CHROMA_VALS: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];

#[derive(Clone)]
struct JpegHuffDecTable {
    maxcode: [i32; 17],
    valoffset: [i32; 17],
    vals: Vec<u8>,
}

impl JpegHuffDecTable {
    fn from_bits_vals(bits: &[u8; 16], vals: &[u8]) -> Self {
        let mut maxcode = [-1i32; 17];
        let mut valoffset = [0i32; 17];
        let mut code = 0i32;
        let mut k = 0i32;
        for l in 1..=16 {
            let count = bits[l - 1] as i32;
            if count == 0 {
                maxcode[l] = -1;
            } else {
                valoffset[l] = k - code;
                code += count;
                maxcode[l] = code - 1;
                k += count;
            }
            code <<= 1;
        }
        Self {
            maxcode,
            valoffset,
            vals: vals.to_vec(),
        }
    }
}

#[derive(Clone)]
struct JpegHuffEncTable {
    codes: [u16; 256],
    sizes: [u8; 256],
}

impl JpegHuffEncTable {
    fn from_bits_vals(bits: &[u8; 16], vals: &[u8]) -> Self {
        let mut codes = [0u16; 256];
        let mut sizes = [0u8; 256];
        let mut code = 0u16;
        let mut k = 0usize;
        for l in 1..=16 {
            let count = bits[l - 1] as usize;
            for _ in 0..count {
                let sym = vals[k] as usize;
                codes[sym] = code;
                sizes[sym] = l as u8;
                code += 1;
                k += 1;
            }
            code <<= 1;
        }
        Self { codes, sizes }
    }
}

struct JpegBitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_buf: u32,
    bit_count: u8,
    hit_marker: Option<u8>,
}

impl<'a> JpegBitReader<'a> {
    fn new(data: &'a [u8], pos: usize) -> Self {
        Self {
            data,
            pos,
            bit_buf: 0,
            bit_count: 0,
            hit_marker: None,
        }
    }

    #[inline]
    fn fill_byte(&mut self) -> u8 {
        if self.hit_marker.is_some() || self.pos >= self.data.len() {
            return 0;
        }
        let b = self.data[self.pos];
        self.pos += 1;
        if b == 0xFF {
            while self.pos < self.data.len() && self.data[self.pos] == 0xFF {
                self.pos += 1;
            }
            if self.pos < self.data.len() {
                let m = self.data[self.pos];
                self.pos += 1;
                if m == 0x00 {
                    // Stuffed 0xFF byte
                    return 0xFF;
                } else {
                    self.hit_marker = Some(m);
                    return 0;
                }
            }
        }
        b
    }

    #[inline]
    fn ensure_bits(&mut self, n: u8) {
        while self.bit_count < n {
            let b = self.fill_byte() as u32;
            self.bit_buf = (self.bit_buf << 8) | b;
            self.bit_count += 8;
        }
    }

    #[inline]
    fn get_bits(&mut self, n: u8) -> i32 {
        if n == 0 {
            return 0;
        }
        self.ensure_bits(n);
        self.bit_count -= n;
        ((self.bit_buf >> self.bit_count) & ((1u32 << n) - 1)) as i32
    }

    #[inline]
    fn decode_huff(&mut self, table: &JpegHuffDecTable) -> Result<u8> {
        let mut code = 0i32;
        for l in 1..=16 {
            let bit = self.get_bits(1);
            code = (code << 1) | bit;
            if code <= table.maxcode[l] {
                let idx = (code + table.valoffset[l]) as usize;
                return table
                    .vals
                    .get(idx)
                    .copied()
                    .ok_or(RagdollError::CorruptBitstream("JPEG Huffman index OOB"));
            }
        }
        Err(RagdollError::CorruptBitstream("invalid JPEG Huffman code"))
    }

    #[inline]
    fn receive_extend(&mut self, ssss: u8) -> i32 {
        if ssss == 0 {
            return 0;
        }
        let v = self.get_bits(ssss);
        let vt = 1i32 << (ssss - 1);
        if v < vt { v + ((-1i32) << ssss) + 1 } else { v }
    }

    fn consume_restart_marker(&mut self) {
        self.bit_buf = 0;
        self.bit_count = 0;
        if let Some(m) = self.hit_marker.take()
            && (0xD0..=0xD7).contains(&m)
        {
            return;
        }
        while self.pos + 1 < self.data.len() {
            if self.data[self.pos] == 0xFF {
                let m = self.data[self.pos + 1];
                if (0xD0..=0xD7).contains(&m) {
                    self.pos += 2;
                    return;
                }
            }
            self.pos += 1;
        }
    }
}

struct JpegBitWriter {
    bytes: Vec<u8>,
    bit_buf: u32,
    bit_count: u8,
}

impl JpegBitWriter {
    fn new(capacity: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(capacity),
            bit_buf: 0,
            bit_count: 0,
        }
    }

    #[inline]
    fn emit_byte(&mut self, b: u8) {
        self.bytes.push(b);
        if b == 0xFF {
            self.bytes.push(0x00); // Byte stuffing
        }
    }

    #[inline]
    fn put_bits(&mut self, val: u16, n: u8) {
        if n == 0 {
            return;
        }
        let masked = (val as u32) & ((1u32 << n) - 1);
        self.bit_buf = (self.bit_buf << n) | masked;
        self.bit_count += n;
        while self.bit_count >= 8 {
            self.bit_count -= 8;
            let b = ((self.bit_buf >> self.bit_count) & 0xFF) as u8;
            self.emit_byte(b);
        }
    }

    fn flush_pad_ones(&mut self) {
        if self.bit_count > 0 {
            let pad = 8 - self.bit_count;
            self.put_bits((1u16 << pad) - 1, pad);
        }
    }
}

/// Separable $8\times8$ 2D Inverse DCT (ITU-T T.81 Section A.3.3, level-shifted `+128` and clamped to `0..=255`).
pub fn idct_8x8(coeffs: &[i32; 64], out: &mut [u8; 64]) {
    const COS_TABLE: [[f32; 8]; 8] = {
        // Precomputed C(u) * cos((2x + 1) * u * pi / 16)
        [
            [
                0.70710678, 0.70710678, 0.70710678, 0.70710678, 0.70710678, 0.70710678, 0.70710678,
                0.70710678,
            ],
            [
                0.98078528,
                0.83146961,
                0.55557023,
                0.19509032,
                -0.19509032,
                -0.55557023,
                -0.83146961,
                -0.98078528,
            ],
            [
                0.92387953,
                0.38268343,
                -0.38268343,
                -0.92387953,
                -0.92387953,
                -0.38268343,
                0.38268343,
                0.92387953,
            ],
            [
                0.83146961,
                -0.19509032,
                -0.98078528,
                -0.55557023,
                0.55557023,
                0.98078528,
                0.19509032,
                -0.83146961,
            ],
            [
                0.70710678,
                -0.70710678,
                -0.70710678,
                0.70710678,
                0.70710678,
                -0.70710678,
                -0.70710678,
                0.70710678,
            ],
            [
                0.55557023,
                -0.98078528,
                0.19509032,
                0.83146961,
                -0.83146961,
                -0.19509032,
                0.98078528,
                -0.55557023,
            ],
            [
                0.38268343,
                -0.92387953,
                0.92387953,
                -0.38268343,
                -0.38268343,
                0.92387953,
                -0.92387953,
                0.38268343,
            ],
            [
                0.19509032,
                -0.55557023,
                0.83146961,
                -0.98078528,
                0.98078528,
                -0.83146961,
                0.55557023,
                -0.19509032,
            ],
        ]
    };

    let mut tmp = [0.0f32; 64];
    // 1D IDCT along rows
    for y in 0..8 {
        for x in 0..8 {
            let mut sum = 0.0f32;
            for u in 0..8 {
                sum += (coeffs[y * 8 + u] as f32) * COS_TABLE[u][x];
            }
            tmp[y * 8 + x] = sum * 0.5;
        }
    }
    // 1D IDCT along columns
    for x in 0..8 {
        for y in 0..8 {
            let mut sum = 0.0f32;
            for v in 0..8 {
                sum += tmp[v * 8 + x] * COS_TABLE[v][y];
            }
            let val = (sum * 0.5 + 128.0).round() as i32;
            out[y * 8 + x] = val.clamp(0, 255) as u8;
        }
    }
}

/// Separable $8\times8$ 2D Forward DCT (level-shifted `-128`).
pub fn fdct_8x8(block: &[u8; 64], out: &mut [f32; 64]) {
    const COS_TABLE: [[f32; 8]; 8] = [
        [
            0.70710678, 0.70710678, 0.70710678, 0.70710678, 0.70710678, 0.70710678, 0.70710678,
            0.70710678,
        ],
        [
            0.98078528,
            0.83146961,
            0.55557023,
            0.19509032,
            -0.19509032,
            -0.55557023,
            -0.83146961,
            -0.98078528,
        ],
        [
            0.92387953,
            0.38268343,
            -0.38268343,
            -0.92387953,
            -0.92387953,
            -0.38268343,
            0.38268343,
            0.92387953,
        ],
        [
            0.83146961,
            -0.19509032,
            -0.98078528,
            -0.55557023,
            0.55557023,
            0.98078528,
            0.19509032,
            -0.83146961,
        ],
        [
            0.70710678,
            -0.70710678,
            -0.70710678,
            0.70710678,
            0.70710678,
            -0.70710678,
            -0.70710678,
            0.70710678,
        ],
        [
            0.55557023,
            -0.98078528,
            0.19509032,
            0.83146961,
            -0.83146961,
            -0.19509032,
            0.98078528,
            -0.55557023,
        ],
        [
            0.38268343,
            -0.92387953,
            0.92387953,
            -0.38268343,
            -0.38268343,
            0.92387953,
            -0.92387953,
            0.38268343,
        ],
        [
            0.19509032,
            -0.55557023,
            0.83146961,
            -0.98078528,
            0.98078528,
            -0.83146961,
            0.55557023,
            -0.19509032,
        ],
    ];

    let mut tmp = [0.0f32; 64];
    for y in 0..8 {
        for u in 0..8 {
            let mut sum = 0.0f32;
            for x in 0..8 {
                let s = (block[y * 8 + x] as f32) - 128.0;
                sum += s * COS_TABLE[u][x];
            }
            tmp[y * 8 + u] = sum * 0.5;
        }
    }
    for u in 0..8 {
        for v in 0..8 {
            let mut sum = 0.0f32;
            for y in 0..8 {
                sum += tmp[y * 8 + u] * COS_TABLE[v][y];
            }
            out[v * 8 + u] = sum * 0.5;
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct JpegCompInfo {
    id: u8,
    h_samp: usize,
    v_samp: usize,
    quant_id: usize,
    dc_table_id: usize,
    ac_table_id: usize,
}

/// Probes the dimensions and pixel format of a `.jpg` bitstream without decoding entropy data.
pub fn probe_jpeg_header(data: &[u8]) -> Result<(u32, u32, PixelFormat)> {
    if !is_jpeg_bytes(data) {
        return Err(RagdollError::InvalidHeader(
            "missing JPEG SOI marker".into(),
        ));
    }
    let mut pos = 2usize;
    while pos + 4 <= data.len() {
        if data[pos] != 0xFF {
            pos += 1;
            continue;
        }
        while pos < data.len() && data[pos] == 0xFF {
            pos += 1;
        }
        if pos >= data.len() {
            break;
        }
        let marker = data[pos];
        pos += 1;
        if marker == 0xD8 || marker == 0x01 || (0xD0..=0xD7).contains(&marker) {
            continue;
        }
        if marker == 0xD9 {
            break;
        }
        if pos + 2 > data.len() {
            break;
        }
        let seg_len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
        if seg_len < 2 || pos + seg_len > data.len() {
            break;
        }
        let body = &data[pos + 2..pos + seg_len];
        if marker == 0xC0 || marker == 0xC1 || marker == 0xC2 {
            if body.len() < 6 {
                return Err(RagdollError::InvalidHeader("truncated JPEG SOF".into()));
            }
            let height = u16::from_be_bytes([body[1], body[2]]) as u32;
            let width = u16::from_be_bytes([body[3], body[4]]) as u32;
            let ncomp = body[5];
            let fmt = if ncomp == 1 {
                PixelFormat::Gray8
            } else {
                PixelFormat::Rgb24
            };
            return Ok((width, height, fmt));
        }
        pos += seg_len;
    }
    Err(RagdollError::InvalidHeader(
        "JPEG missing SOF marker".into(),
    ))
}

/// Decodes a Baseline DCT (`SOF0`) `.jpg`/`.jpeg` image into a [`VideoFrame`] (`Rgb24` or `Gray8`).
pub fn read_jpeg_image(data: &[u8]) -> Result<VideoFrame> {
    if !is_jpeg_bytes(data) {
        return Err(RagdollError::InvalidHeader(
            "missing JPEG SOI marker".into(),
        ));
    }

    let mut qtables = [[1i32; 64]; 4];
    let mut dc_tables: [Option<JpegHuffDecTable>; 4] = [
        Some(JpegHuffDecTable::from_bits_vals(
            &STD_DC_LUMA_BITS,
            &STD_DC_LUMA_VALS,
        )),
        Some(JpegHuffDecTable::from_bits_vals(
            &STD_DC_CHROMA_BITS,
            &STD_DC_CHROMA_VALS,
        )),
        None,
        None,
    ];
    let mut ac_tables: [Option<JpegHuffDecTable>; 4] = [
        Some(JpegHuffDecTable::from_bits_vals(
            &STD_AC_LUMA_BITS,
            &STD_AC_LUMA_VALS,
        )),
        Some(JpegHuffDecTable::from_bits_vals(
            &STD_AC_CHROMA_BITS,
            &STD_AC_CHROMA_VALS,
        )),
        None,
        None,
    ];

    let mut width = 0usize;
    let mut height = 0usize;
    let mut comps: Vec<JpegCompInfo> = Vec::new();
    let mut restart_interval = 0usize;
    let mut sos_data_offset = None;

    let mut pos = 2usize;
    while pos + 2 <= data.len() {
        if data[pos] != 0xFF {
            pos += 1;
            continue;
        }
        while pos < data.len() && data[pos] == 0xFF {
            pos += 1;
        }
        if pos >= data.len() {
            break;
        }
        let marker = data[pos];
        pos += 1;
        if marker == 0xD8 || marker == 0x01 || (0xD0..=0xD7).contains(&marker) {
            continue;
        }
        if marker == 0xD9 {
            break;
        }
        if pos + 2 > data.len() {
            return Err(RagdollError::BitstreamEndOfData);
        }
        let seg_len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
        if seg_len < 2 || pos + seg_len > data.len() {
            return Err(RagdollError::BitstreamEndOfData);
        }
        let body = &data[pos + 2..pos + seg_len];
        pos += seg_len;

        match marker {
            // DQT
            0xDB => {
                let mut i = 0usize;
                while i < body.len() {
                    let pq_tq = body[i];
                    i += 1;
                    let pq = pq_tq >> 4;
                    let tq = (pq_tq & 0x0F) as usize;
                    if tq >= 4 {
                        return Err(RagdollError::InvalidHeader(
                            "JPEG DQT table index >= 4".into(),
                        ));
                    }
                    if pq == 0 {
                        if i + 64 > body.len() {
                            return Err(RagdollError::BitstreamEndOfData);
                        }
                        for zz in 0..64 {
                            qtables[tq][ZIGZAG_TO_NATURAL[zz]] = body[i + zz] as i32;
                        }
                        i += 64;
                    } else {
                        if i + 128 > body.len() {
                            return Err(RagdollError::BitstreamEndOfData);
                        }
                        for zz in 0..64 {
                            let q = u16::from_be_bytes([body[i + zz * 2], body[i + zz * 2 + 1]]);
                            qtables[tq][ZIGZAG_TO_NATURAL[zz]] = q as i32;
                        }
                        i += 128;
                    }
                }
            }
            // SOF0 / SOF1 (Baseline / Extended Sequential DCT)
            0xC0 | 0xC1 => {
                if body.len() < 6 {
                    return Err(RagdollError::InvalidHeader("JPEG SOF0 too short".into()));
                }
                let precision = body[0];
                if precision != 8 {
                    return Err(RagdollError::Unsupported(format!(
                        "unsupported JPEG sample precision {precision}"
                    )));
                }
                height = u16::from_be_bytes([body[1], body[2]]) as usize;
                width = u16::from_be_bytes([body[3], body[4]]) as usize;
                let ncomp = body[5] as usize;
                if width == 0
                    || height == 0
                    || (ncomp != 1 && ncomp != 3)
                    || body.len() < 6 + ncomp * 3
                {
                    return Err(RagdollError::InvalidHeader(
                        "unsupported JPEG SOF0 dimensions/components".into(),
                    ));
                }
                comps.clear();
                for c in 0..ncomp {
                    let id = body[6 + c * 3];
                    let hv = body[6 + c * 3 + 1];
                    let tq = body[6 + c * 3 + 2] as usize;
                    let h_samp = (hv >> 4) as usize;
                    let v_samp = (hv & 0x0F) as usize;
                    if h_samp == 0 || v_samp == 0 || h_samp > 2 || v_samp > 2 || tq >= 4 {
                        return Err(RagdollError::Unsupported(
                            "unsupported JPEG sampling factor or quant table".into(),
                        ));
                    }
                    comps.push(JpegCompInfo {
                        id,
                        h_samp,
                        v_samp,
                        quant_id: tq,
                        dc_table_id: if c == 0 { 0 } else { 1 },
                        ac_table_id: if c == 0 { 0 } else { 1 },
                    });
                }
            }
            // DHT
            0xC4 => {
                let mut i = 0usize;
                while i + 17 <= body.len() {
                    let tc_th = body[i];
                    i += 1;
                    let tc = tc_th >> 4;
                    let th = (tc_th & 0x0F) as usize;
                    if th >= 4 {
                        return Err(RagdollError::InvalidHeader("JPEG DHT th >= 4".into()));
                    }
                    let mut bits = [0u8; 16];
                    bits.copy_from_slice(&body[i..i + 16]);
                    i += 16;
                    let total_vals: usize = bits.iter().map(|&b| b as usize).sum();
                    if i + total_vals > body.len() {
                        return Err(RagdollError::BitstreamEndOfData);
                    }
                    let vals = &body[i..i + total_vals];
                    i += total_vals;
                    let table = JpegHuffDecTable::from_bits_vals(&bits, vals);
                    if tc == 0 {
                        dc_tables[th] = Some(table);
                    } else {
                        ac_tables[th] = Some(table);
                    }
                }
            }
            // DRI
            0xDD => {
                if body.len() >= 2 {
                    restart_interval = u16::from_be_bytes([body[0], body[1]]) as usize;
                }
            }
            // SOS
            0xDA => {
                if body.is_empty() {
                    return Err(RagdollError::InvalidHeader("JPEG empty SOS".into()));
                }
                let ns = body[0] as usize;
                if body.len() < 1 + ns * 2 + 3 {
                    return Err(RagdollError::InvalidHeader("JPEG SOS too short".into()));
                }
                for s in 0..ns {
                    let cs = body[1 + s * 2];
                    let td_ta = body[1 + s * 2 + 1];
                    let td = (td_ta >> 4) as usize;
                    let ta = (td_ta & 0x0F) as usize;
                    if let Some(comp) = comps.iter_mut().find(|c| c.id == cs) {
                        comp.dc_table_id = td.min(3);
                        comp.ac_table_id = ta.min(3);
                    }
                }
                sos_data_offset = Some(pos);
                break;
            }
            _ => {}
        }
    }

    let Some(scan_start) = sos_data_offset else {
        return Err(RagdollError::InvalidHeader(
            "JPEG missing SOS marker".into(),
        ));
    };
    if comps.is_empty() {
        return Err(RagdollError::InvalidHeader(
            "JPEG SOS before SOF0 marker".into(),
        ));
    }

    let max_h = comps.iter().map(|c| c.h_samp).max().unwrap_or(1);
    let max_v = comps.iter().map(|c| c.v_samp).max().unwrap_or(1);
    let mcu_w = max_h * 8;
    let mcu_h = max_v * 8;
    let mcu_cols = width.div_ceil(mcu_w);
    let mcu_rows = height.div_ceil(mcu_h);

    let mut comp_planes: Vec<Vec<u8>> = comps
        .iter()
        .map(|c| vec![128u8; (mcu_cols * c.h_samp * 8) * (mcu_rows * c.v_samp * 8)])
        .collect();
    let comp_strides: Vec<usize> = comps.iter().map(|c| mcu_cols * c.h_samp * 8).collect();

    let mut br = JpegBitReader::new(data, scan_start);
    let mut dc_pred = [0i32; 4];
    let mut mcu_counter = 0usize;

    for mcu_y in 0..mcu_rows {
        for mcu_x in 0..mcu_cols {
            if restart_interval > 0
                && mcu_counter > 0
                && mcu_counter.is_multiple_of(restart_interval)
            {
                br.consume_restart_marker();
                dc_pred = [0; 4];
            }
            mcu_counter += 1;

            for (c_idx, comp) in comps.iter().enumerate() {
                let dc_tab = dc_tables[comp.dc_table_id].as_ref().ok_or_else(|| {
                    RagdollError::InvalidHeader("missing JPEG DC Huffman table".into())
                })?;
                let ac_tab = ac_tables[comp.ac_table_id].as_ref().ok_or_else(|| {
                    RagdollError::InvalidHeader("missing JPEG AC Huffman table".into())
                })?;
                let qtab = &qtables[comp.quant_id];
                let stride = comp_strides[c_idx];

                for by in 0..comp.v_samp {
                    for bx in 0..comp.h_samp {
                        let mut coeffs = [0i32; 64];
                        let ssss = br.decode_huff(dc_tab)?;
                        let diff = br.receive_extend(ssss);
                        dc_pred[c_idx] += diff;
                        coeffs[0] = dc_pred[c_idx] * qtab[0];

                        let mut k = 1usize;
                        while k < 64 {
                            let rs = br.decode_huff(ac_tab)?;
                            let r = (rs >> 4) as usize;
                            let s = rs & 0x0F;
                            if s == 0 {
                                if r == 15 {
                                    k += 16;
                                } else {
                                    break; // EOB
                                }
                            } else {
                                k += r;
                                if k >= 64 {
                                    break;
                                }
                                let ac_val = br.receive_extend(s);
                                let nat = ZIGZAG_TO_NATURAL[k];
                                coeffs[nat] = ac_val * qtab[nat];
                                k += 1;
                            }
                        }

                        let mut pixels = [0u8; 64];
                        idct_8x8(&coeffs, &mut pixels);

                        let dst_x0 = (mcu_x * comp.h_samp + bx) * 8;
                        let dst_y0 = (mcu_y * comp.v_samp + by) * 8;
                        let plane = &mut comp_planes[c_idx];
                        for py in 0..8 {
                            let row_off = (dst_y0 + py) * stride + dst_x0;
                            plane[row_off..row_off + 8]
                                .copy_from_slice(&pixels[py * 8..py * 8 + 8]);
                        }
                    }
                }
            }
        }
    }

    if comps.len() == 1 {
        let mut frame = VideoFrame::new(width, height, PixelFormat::Gray8);
        frame.colorimetry = Colorimetry::srgb();
        let stride = comp_strides[0];
        for y in 0..height {
            frame.plane_mut(0).row_mut(y)[..width]
                .copy_from_slice(&comp_planes[0][y * stride..y * stride + width]);
        }
        return Ok(frame);
    }

    // 3-component YCbCr -> Rgb24 (ITU-T T.871 / JFIF full-range BT.601)
    let mut frame = VideoFrame::new(width, height, PixelFormat::Rgb24);
    frame.colorimetry = Colorimetry::srgb();
    let y_stride = comp_strides[0];
    let cb_stride = comp_strides[1];
    let cr_stride = comp_strides[2];
    let cb_h_shift = if max_h == 2 && comps[1].h_samp == 1 {
        1
    } else {
        0
    };
    let cb_v_shift = if max_v == 2 && comps[1].v_samp == 1 {
        1
    } else {
        0
    };
    let cr_h_shift = if max_h == 2 && comps[2].h_samp == 1 {
        1
    } else {
        0
    };
    let cr_v_shift = if max_v == 2 && comps[2].v_samp == 1 {
        1
    } else {
        0
    };

    for y in 0..height {
        let dst = frame.plane_mut(0).row_mut(y);
        let y_row = &comp_planes[0][y * y_stride..];
        let cb_row = &comp_planes[1][(y >> cb_v_shift) * cb_stride..];
        let cr_row = &comp_planes[2][(y >> cr_v_shift) * cr_stride..];
        for x in 0..width {
            let yv = y_row[x] as f32;
            let cb = (cb_row[x >> cb_h_shift] as f32) - 128.0;
            let cr = (cr_row[x >> cr_h_shift] as f32) - 128.0;
            let r = (yv + 1.402 * cr).round().clamp(0.0, 255.0) as u8;
            let g = (yv - 0.344136 * cb - 0.714136 * cr)
                .round()
                .clamp(0.0, 255.0) as u8;
            let b = (yv + 1.772 * cb).round().clamp(0.0, 255.0) as u8;
            dst[x * 3] = r;
            dst[x * 3 + 1] = g;
            dst[x * 3 + 2] = b;
        }
    }

    Ok(frame)
}

#[inline]
fn scale_quant_table_natural(base: &[u8; 64], quality: u8) -> [u8; 64] {
    let q = quality.clamp(1, 100) as i32;
    let scale = if q < 50 { 5000 / q } else { 200 - q * 2 };
    let mut out = [1u8; 64];
    for i in 0..64 {
        let v = ((base[i] as i32) * scale + 50) / 100;
        out[i] = v.clamp(1, 255) as u8;
    }
    out
}

#[inline]
fn encode_magnitude(val: i32) -> (u8, u16) {
    if val == 0 {
        return (0, 0);
    }
    let abs_v = val.unsigned_abs();
    let ssss = (32 - abs_v.leading_zeros()) as u8;
    let bits = if val > 0 {
        val as u16
    } else {
        (val + (1 << ssss) - 1) as u16
    };
    (ssss, bits)
}

fn encode_block_8x8(
    block: &[u8; 64],
    qtab_natural: &[u8; 64],
    prev_dc: &mut i32,
    dc_enc: &JpegHuffEncTable,
    ac_enc: &JpegHuffEncTable,
    bw: &mut JpegBitWriter,
) {
    let mut dct = [0.0f32; 64];
    fdct_8x8(block, &mut dct);

    let mut zz_coeffs = [0i32; 64];
    for zz in 0..64 {
        let nat = ZIGZAG_TO_NATURAL[zz];
        let q = qtab_natural[nat] as f32;
        zz_coeffs[zz] = (dct[nat] / q).round() as i32;
    }

    // Encode DC coefficient difference
    let dc_diff = zz_coeffs[0] - *prev_dc;
    *prev_dc = zz_coeffs[0];
    let (ssss, mag_bits) = encode_magnitude(dc_diff);
    bw.put_bits(dc_enc.codes[ssss as usize], dc_enc.sizes[ssss as usize]);
    if ssss > 0 {
        bw.put_bits(mag_bits, ssss);
    }

    // Encode AC coefficients in zig-zag order
    let mut zero_run = 0usize;
    for k in 1..64 {
        let ac = zz_coeffs[k];
        if ac == 0 {
            zero_run += 1;
        } else {
            while zero_run >= 16 {
                // ZRL (0xF0)
                bw.put_bits(ac_enc.codes[0xF0], ac_enc.sizes[0xF0]);
                zero_run -= 16;
            }
            let (ssss, mag_bits) = encode_magnitude(ac);
            let rs = ((zero_run as u8) << 4) | ssss;
            bw.put_bits(ac_enc.codes[rs as usize], ac_enc.sizes[rs as usize]);
            bw.put_bits(mag_bits, ssss);
            zero_run = 0;
        }
    }
    if zero_run > 0 {
        // EOB (0x00)
        bw.put_bits(ac_enc.codes[0x00], ac_enc.sizes[0x00]);
    }
}

/// Encodes a [`VideoFrame`] into a Baseline DCT (`SOF0`) JFIF `.jpg` bitstream at `quality` (`1..=100`).
pub fn write_jpeg_image<W: Write>(mut writer: W, frame: &VideoFrame, quality: u8) -> Result<u64> {
    let converted;
    let rgb = if frame.format == PixelFormat::Rgb24 {
        frame
    } else {
        converted = ensure_rgb24(frame);
        &converted
    };

    let width = rgb.width;
    let height = rgb.height;
    let luma_q = scale_quant_table_natural(&STD_LUMA_QUANT_NATURAL, quality);
    let chroma_q = scale_quant_table_natural(&STD_CHROMA_QUANT_NATURAL, quality);

    let dc_luma_enc = JpegHuffEncTable::from_bits_vals(&STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS);
    let dc_chroma_enc = JpegHuffEncTable::from_bits_vals(&STD_DC_CHROMA_BITS, &STD_DC_CHROMA_VALS);
    let ac_luma_enc = JpegHuffEncTable::from_bits_vals(&STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS);
    let ac_chroma_enc = JpegHuffEncTable::from_bits_vals(&STD_AC_CHROMA_BITS, &STD_AC_CHROMA_VALS);

    let mut out = Vec::with_capacity(width * height + 1024);

    // 1. SOI
    out.extend_from_slice(&[0xFF, 0xD8]);

    // 2. APP0 JFIF 1.01
    out.extend_from_slice(&[
        0xFF, 0xE0, 0x00, 0x10, b'J', b'F', b'I', b'F', 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00,
        0x01, 0x00, 0x00,
    ]);

    // 3. DQT (table 0 = luma, table 1 = chroma, stored in zig-zag order)
    for (tq, qtab) in [(0u8, &luma_q), (1u8, &chroma_q)] {
        out.extend_from_slice(&[0xFF, 0xDB, 0x00, 67, tq]);
        for zz in 0..64 {
            out.push(qtab[ZIGZAG_TO_NATURAL[zz]]);
        }
    }

    // 4. SOF0 (Baseline DCT, 3 components, 4:4:4 H1V1 for high chroma sharpness)
    out.extend_from_slice(&[
        0xFF,
        0xC0,
        0x00,
        17,
        8, // precision
        (height >> 8) as u8,
        (height & 0xFF) as u8,
        (width >> 8) as u8,
        (width & 0xFF) as u8,
        3, // 3 components (Y, Cb, Cr)
        1,
        0x11,
        0, // Y: H1V1, qtable 0
        2,
        0x11,
        1, // Cb: H1V1, qtable 1
        3,
        0x11,
        1, // Cr: H1V1, qtable 1
    ]);

    // 5. DHT (4 tables: DC0, AC0, DC1, AC1)
    let huff_defs: [(u8, &[u8; 16], &[u8]); 4] = [
        (0x00, &STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS),
        (0x10, &STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS),
        (0x01, &STD_DC_CHROMA_BITS, &STD_DC_CHROMA_VALS),
        (0x11, &STD_AC_CHROMA_BITS, &STD_AC_CHROMA_VALS),
    ];
    for (tc_th, bits, vals) in huff_defs {
        let seg_len = (2 + 1 + 16 + vals.len()) as u16;
        out.extend_from_slice(&[0xFF, 0xC4]);
        out.extend_from_slice(&seg_len.to_be_bytes());
        out.push(tc_th);
        out.extend_from_slice(bits);
        out.extend_from_slice(vals);
    }

    // 6. SOS
    out.extend_from_slice(&[
        0xFF, 0xDA, 0x00, 12, 3, // 3 components
        1, 0x00, // Y -> DC0, AC0
        2, 0x11, // Cb -> DC1, AC1
        3, 0x11, // Cr -> DC1, AC1
        0x00, 0x3F, 0x00, // Ss=0, Se=63, Ah/Al=0
    ]);

    let mcu_cols = width.div_ceil(8);
    let mcu_rows = height.div_ceil(8);
    let mut bw = JpegBitWriter::new(width * height / 2);
    let mut prev_dc_y = 0i32;
    let mut prev_dc_cb = 0i32;
    let mut prev_dc_cr = 0i32;

    for mcu_y in 0..mcu_rows {
        for mcu_x in 0..mcu_cols {
            let mut y_blk = [0u8; 64];
            let mut cb_blk = [128u8; 64];
            let mut cr_blk = [128u8; 64];

            for py in 0..8 {
                let sy = (mcu_y * 8 + py).min(height - 1);
                let row = rgb.plane(0).row(sy);
                for px in 0..8 {
                    let sx = (mcu_x * 8 + px).min(width - 1);
                    let r = row[sx * 3] as f32;
                    let g = row[sx * 3 + 1] as f32;
                    let b = row[sx * 3 + 2] as f32;
                    let yv = (0.299f32 * r + 0.587f32 * g + 0.114f32 * b)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                    let cb = (-0.168736f32 * r - 0.331264f32 * g + 0.5f32 * b + 128.0f32)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                    let cr = (0.5f32 * r - 0.418688f32 * g - 0.081312f32 * b + 128.0f32)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                    y_blk[py * 8 + px] = yv;
                    cb_blk[py * 8 + px] = cb;
                    cr_blk[py * 8 + px] = cr;
                }
            }

            encode_block_8x8(
                &y_blk,
                &luma_q,
                &mut prev_dc_y,
                &dc_luma_enc,
                &ac_luma_enc,
                &mut bw,
            );
            encode_block_8x8(
                &cb_blk,
                &chroma_q,
                &mut prev_dc_cb,
                &dc_chroma_enc,
                &ac_chroma_enc,
                &mut bw,
            );
            encode_block_8x8(
                &cr_blk,
                &chroma_q,
                &mut prev_dc_cr,
                &dc_chroma_enc,
                &ac_chroma_enc,
                &mut bw,
            );
        }
    }

    bw.flush_pad_ones();
    out.extend_from_slice(&bw.bytes);

    // 7. EOI
    out.extend_from_slice(&[0xFF, 0xD9]);

    writer.write_all(&out)?;
    writer.flush()?;
    Ok(out.len() as u64)
}
