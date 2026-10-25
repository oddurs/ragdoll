//! **`0047`**: Pure-Rust PNG (`.png`, RFC 2083) Decoder & Encoder with RFC 1950/1951 Zlib/DEFLATE
//! compression/decompression, `crc32_png` + `adler32` verification, and adaptive Paeth scanline filtering.

#![allow(clippy::needless_range_loop)]

use std::io::Write;

use ragdoll_core::{
    Colorimetry, PixelFormat, RagdollError, Result, VideoFrame, adler32, crc32_png,
};

pub(crate) fn ensure_rgb24(frame: &VideoFrame) -> VideoFrame {
    let w = frame.width;
    let h = frame.height;
    let mut out = VideoFrame::new(w, h, PixelFormat::Rgb24);
    out.colorimetry = frame.colorimetry;
    match frame.format {
        PixelFormat::Rgb24 => {
            for y in 0..h {
                out.plane_mut(0).row_mut(y)[..w * 3]
                    .copy_from_slice(&frame.plane(0).row(y)[..w * 3]);
            }
        }
        PixelFormat::Rgba8 => {
            for y in 0..h {
                let src = frame.plane(0).row(y);
                let dst = out.plane_mut(0).row_mut(y);
                for x in 0..w {
                    dst[x * 3] = src[x * 4];
                    dst[x * 3 + 1] = src[x * 4 + 1];
                    dst[x * 3 + 2] = src[x * 4 + 2];
                }
            }
        }
        PixelFormat::Bgra8 => {
            for y in 0..h {
                let src = frame.plane(0).row(y);
                let dst = out.plane_mut(0).row_mut(y);
                for x in 0..w {
                    dst[x * 3] = src[x * 4 + 2];
                    dst[x * 3 + 1] = src[x * 4 + 1];
                    dst[x * 3 + 2] = src[x * 4];
                }
            }
        }
        PixelFormat::Gray8 => {
            for y in 0..h {
                let src = frame.plane(0).row(y);
                let dst = out.plane_mut(0).row_mut(y);
                for x in 0..w {
                    let g = src[x];
                    dst[x * 3] = g;
                    dst[x * 3 + 1] = g;
                    dst[x * 3 + 2] = g;
                }
            }
        }
        _ => {
            // Yuv420p or planar YUV -> BT.709 / BT.601 RGB24
            let has_chroma = frame.planes.len() >= 3;
            for y in 0..h {
                let y_row = frame.plane(0).row(y);
                let dst = out.plane_mut(0).row_mut(y);
                for x in 0..w {
                    let yv = y_row[x] as f32;
                    let (u, v) = if has_chroma {
                        let cw = frame.plane(1).width().max(1);
                        let ch = frame.plane(1).height().max(1);
                        let cx = (x * cw / w.max(1)).min(cw - 1);
                        let cy = (y * ch / h.max(1)).min(ch - 1);
                        (
                            frame.plane(1).row(cy)[cx] as f32 - 128.0,
                            frame.plane(2).row(cy)[cx] as f32 - 128.0,
                        )
                    } else {
                        (0.0, 0.0)
                    };
                    let r = (yv + 1.402 * v).round().clamp(0.0, 255.0) as u8;
                    let g = (yv - 0.344136 * u - 0.714136 * v).round().clamp(0.0, 255.0) as u8;
                    let b = (yv + 1.772 * u).round().clamp(0.0, 255.0) as u8;
                    dst[x * 3] = r;
                    dst[x * 3 + 1] = g;
                    dst[x * 3 + 2] = b;
                }
            }
        }
    }
    out
}

/// 8-byte PNG file signature (`\x89PNG\r\n\x1a\n`).
pub const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

/// Returns `true` if `data` begins with the 8-byte PNG signature.
#[inline]
pub fn is_png_bytes(data: &[u8]) -> bool {
    data.len() >= 8 && data[0..8] == PNG_SIGNATURE
}

/// Probes the `IHDR` dimensions and pixel format of a `.png` file in $O(1)$ without decompressing `IDAT`.
pub fn probe_png_header(data: &[u8]) -> Result<(u32, u32, PixelFormat)> {
    if !is_png_bytes(data) || data.len() < 33 || &data[12..16] != b"IHDR" {
        return Err(RagdollError::InvalidHeader(
            "invalid or missing PNG IHDR chunk".into(),
        ));
    }
    let width = u32::from_be_bytes([data[16], data[17], data[18], data[19]]);
    let height = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);
    let color_type = data[25];
    let fmt = match color_type {
        0 => PixelFormat::Gray8,
        2 | 3 => PixelFormat::Rgb24,
        4 | 6 => PixelFormat::Rgba8,
        other => {
            return Err(RagdollError::Unsupported(format!(
                "unsupported PNG color_type {other}"
            )));
        }
    };
    Ok((width, height, fmt))
}

/// RFC 2083 Section 6.6 Paeth predictor function.
#[inline]
pub fn paeth_predictor(a: u8, b: u8, c: u8) -> u8 {
    let ia = a as i32;
    let ib = b as i32;
    let ic = c as i32;
    let p = ia + ib - ic;
    let pa = (p - ia).abs();
    let pb = (p - ib).abs();
    let pc = (p - ic).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Unfilters a single PNG scanline in-place (`filter_type` in `0..=4`, `bpp` bytes per pixel).
pub fn unfilter_png_scanline(
    filter_type: u8,
    cur: &mut [u8],
    prev: &[u8],
    bpp: usize,
) -> Result<()> {
    let n = cur.len();
    match filter_type {
        // 0: None
        0 => {}
        // 1: Sub
        1 => {
            for i in bpp..n {
                cur[i] = cur[i].wrapping_add(cur[i - bpp]);
            }
        }
        // 2: Up
        2 => {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::*;
                let mut i = 0usize;
                while i + 16 <= n {
                    let vc = vld1q_u8(cur.as_ptr().add(i));
                    let vp = vld1q_u8(prev.as_ptr().add(i));
                    vst1q_u8(cur.as_mut_ptr().add(i), vaddq_u8(vc, vp));
                    i += 16;
                }
                while i < n {
                    cur[i] = cur[i].wrapping_add(prev[i]);
                    i += 1;
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            for i in 0..n {
                cur[i] = cur[i].wrapping_add(prev[i]);
            }
        }
        // 3: Average
        3 => {
            for i in 0..bpp.min(n) {
                cur[i] = cur[i].wrapping_add(prev[i] >> 1);
            }
            for i in bpp..n {
                let avg = (((cur[i - bpp] as u16) + (prev[i] as u16)) >> 1) as u8;
                cur[i] = cur[i].wrapping_add(avg);
            }
        }
        // 4: Paeth
        4 => {
            for i in 0..bpp.min(n) {
                cur[i] = cur[i].wrapping_add(paeth_predictor(0, prev[i], 0));
            }
            for i in bpp..n {
                let pred = paeth_predictor(cur[i - bpp], prev[i], prev[i - bpp]);
                cur[i] = cur[i].wrapping_add(pred);
            }
        }
        other => {
            return Err(RagdollError::InvalidData(format!(
                "unsupported PNG scanline filter type {other}"
            )));
        }
    }
    Ok(())
}

/// Applies adaptive PNG scanline filtering (`Sub`, `Up`, `Paeth`) and appends `(filter_byte + filtered_row)` to `out`.
pub fn filter_png_scanline_adaptive(cur: &[u8], prev: &[u8], bpp: usize, out: &mut Vec<u8>) {
    let n = cur.len();
    let mut cand_sub = vec![0u8; n];
    let mut cand_up = vec![0u8; n];
    let mut cand_paeth = vec![0u8; n];

    let mut cost_sub = 0u64;
    let mut cost_up = 0u64;
    let mut cost_paeth = 0u64;

    for i in 0..n {
        let a = if i >= bpp { cur[i - bpp] } else { 0 };
        let b = prev[i];
        let c = if i >= bpp { prev[i - bpp] } else { 0 };

        let ds = cur[i].wrapping_sub(a);
        let du = cur[i].wrapping_sub(b);
        let dp = cur[i].wrapping_sub(paeth_predictor(a, b, c));

        cand_sub[i] = ds;
        cand_up[i] = du;
        cand_paeth[i] = dp;

        cost_sub += (ds as i8 as i32).unsigned_abs() as u64;
        cost_up += (du as i8 as i32).unsigned_abs() as u64;
        cost_paeth += (dp as i8 as i32).unsigned_abs() as u64;
    }

    if cost_sub <= cost_up && cost_sub <= cost_paeth {
        out.push(1);
        out.extend_from_slice(&cand_sub);
    } else if cost_up <= cost_paeth {
        out.push(2);
        out.extend_from_slice(&cand_up);
    } else {
        out.push(4);
        out.extend_from_slice(&cand_paeth);
    }
}

// ---------------------------------------------------------------------------
// RFC 1951 DEFLATE + RFC 1950 Zlib Decompressor & Compressor
// ---------------------------------------------------------------------------

struct DeflateBitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_buf: u64,
    bit_cnt: u32,
}

impl<'a> DeflateBitReader<'a> {
    #[inline]
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_buf: 0,
            bit_cnt: 0,
        }
    }

    #[inline]
    fn refill(&mut self) {
        while self.bit_cnt <= 56 && self.byte_pos < self.data.len() {
            self.bit_buf |= (self.data[self.byte_pos] as u64) << self.bit_cnt;
            self.byte_pos += 1;
            self.bit_cnt += 8;
        }
    }

    #[inline]
    fn read_bits(&mut self, n: u32) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        if self.bit_cnt < n {
            self.refill();
            if self.bit_cnt < n {
                return Err(RagdollError::BitstreamEndOfData);
            }
        }
        let val = (self.bit_buf & ((1u64 << n) - 1)) as u32;
        self.bit_buf >>= n;
        self.bit_cnt -= n;
        Ok(val)
    }

    #[inline]
    fn align_to_byte(&mut self) {
        let drop_bits = self.bit_cnt & 7;
        self.bit_buf >>= drop_bits;
        self.bit_cnt -= drop_bits;
    }

    #[inline]
    fn read_u16_le_aligned(&mut self) -> Result<u16> {
        self.align_to_byte();
        let lo = self.read_bits(8)? as u16;
        let hi = self.read_bits(8)? as u16;
        Ok(lo | (hi << 8))
    }
}

#[derive(Clone)]
struct DeflateHuffmanTable {
    counts: [u16; 16],
    symbols: Vec<u16>,
}

impl DeflateHuffmanTable {
    fn from_code_lengths(lengths: &[u8]) -> Result<Self> {
        let mut counts = [0u16; 16];
        for &l in lengths {
            if l > 15 {
                return Err(RagdollError::InvalidData(
                    "DEFLATE Huffman code length > 15".into(),
                ));
            }
            if l > 0 {
                counts[l as usize] += 1;
            }
        }
        let mut offsets = [0u16; 16];
        let mut total = 0u16;
        for bits in 1..=15 {
            offsets[bits] = total;
            total += counts[bits];
        }
        let mut symbols = vec![0u16; total as usize];
        for (sym, &l) in lengths.iter().enumerate() {
            if l > 0 {
                let idx = offsets[l as usize] as usize;
                offsets[l as usize] += 1;
                symbols[idx] = sym as u16;
            }
        }
        Ok(Self { counts, symbols })
    }

    #[inline]
    fn decode(&self, br: &mut DeflateBitReader<'_>) -> Result<u16> {
        let mut code = 0i32;
        let mut first = 0i32;
        let mut index = 0i32;
        for len in 1..=15 {
            code |= br.read_bits(1)? as i32;
            let count = self.counts[len] as i32;
            if code - count < first {
                let sym_idx = index.saturating_add(code - first);
                if sym_idx >= 0
                    && let Some(&sym) = self.symbols.get(sym_idx as usize)
                {
                    return Ok(sym);
                }
                return Err(RagdollError::InvalidData(
                    "DEFLATE Huffman symbol index out of bounds".into(),
                ));
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        Err(RagdollError::InvalidData(
            "invalid DEFLATE Huffman code".into(),
        ))
    }
}

fn fixed_huffman_tables() -> (DeflateHuffmanTable, DeflateHuffmanTable) {
    let mut lit_lengths = [0u8; 288];
    for i in 0..=143 {
        lit_lengths[i] = 8;
    }
    for i in 144..=255 {
        lit_lengths[i] = 9;
    }
    for i in 256..=279 {
        lit_lengths[i] = 7;
    }
    for i in 280..=287 {
        lit_lengths[i] = 8;
    }
    let dist_lengths = [5u8; 32];
    (
        DeflateHuffmanTable::from_code_lengths(&lit_lengths).unwrap(),
        DeflateHuffmanTable::from_code_lengths(&dist_lengths).unwrap(),
    )
}

const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
const CL_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// Decompresses an RFC 1950 Zlib-wrapped DEFLATE stream and verifies its trailing Adler-32 checksum.
pub fn zlib_decompress(zlib_data: &[u8]) -> Result<Vec<u8>> {
    if zlib_data.len() < 6 {
        return Err(RagdollError::BitstreamEndOfData);
    }
    let cmf = zlib_data[0];
    let flg = zlib_data[1];
    if (cmf & 0x0F) != 8 || !(((cmf as u16) << 8) | (flg as u16)).is_multiple_of(31) {
        return Err(RagdollError::InvalidHeader("invalid Zlib header".into()));
    }
    let fdict = (flg & 0x20) != 0;
    let deflate_start = if fdict { 6 } else { 2 };
    if zlib_data.len() < deflate_start + 4 {
        return Err(RagdollError::BitstreamEndOfData);
    }

    let mut br = DeflateBitReader::new(&zlib_data[deflate_start..]);
    let mut out = Vec::with_capacity(zlib_data.len() * 3);
    let (fixed_lit, fixed_dist) = fixed_huffman_tables();

    loop {
        let bfinal = br.read_bits(1)?;
        let btype = br.read_bits(2)?;
        match btype {
            // Uncompressed stored block
            0 => {
                let len = br.read_u16_le_aligned()? as usize;
                let nlen = br.read_u16_le_aligned()?;
                if (len as u16) != !nlen {
                    return Err(RagdollError::InvalidData(
                        "DEFLATE stored block LEN != ~NLEN".into(),
                    ));
                }
                for _ in 0..len {
                    out.push(br.read_bits(8)? as u8);
                }
            }
            // Fixed or Dynamic Huffman block
            1 | 2 => {
                let (lit_table, dist_table) = if btype == 1 {
                    (fixed_lit.clone(), fixed_dist.clone())
                } else {
                    let hlit = (br.read_bits(5)? as usize) + 257;
                    let hdist = (br.read_bits(5)? as usize) + 1;
                    let hclen = (br.read_bits(4)? as usize) + 4;
                    let mut cl_lengths = [0u8; 19];
                    for i in 0..hclen {
                        cl_lengths[CL_ORDER[i]] = br.read_bits(3)? as u8;
                    }
                    let cl_table = DeflateHuffmanTable::from_code_lengths(&cl_lengths)?;
                    let total_codes = hlit + hdist;
                    let mut lengths = Vec::with_capacity(total_codes);
                    while lengths.len() < total_codes {
                        let sym = cl_table.decode(&mut br)?;
                        match sym {
                            0..=15 => lengths.push(sym as u8),
                            16 => {
                                let prev = *lengths.last().ok_or_else(|| {
                                    RagdollError::InvalidData(
                                        "DEFLATE code 16 with no previous length".into(),
                                    )
                                })?;
                                let rep = (br.read_bits(2)? as usize) + 3;
                                for _ in 0..rep {
                                    if lengths.len() < total_codes {
                                        lengths.push(prev);
                                    }
                                }
                            }
                            17 => {
                                let rep = (br.read_bits(3)? as usize) + 3;
                                for _ in 0..rep {
                                    if lengths.len() < total_codes {
                                        lengths.push(0);
                                    }
                                }
                            }
                            18 => {
                                let rep = (br.read_bits(7)? as usize) + 11;
                                for _ in 0..rep {
                                    if lengths.len() < total_codes {
                                        lengths.push(0);
                                    }
                                }
                            }
                            _ => {
                                return Err(RagdollError::InvalidData(
                                    "invalid DEFLATE code-length symbol".into(),
                                ));
                            }
                        }
                    }
                    let lt = DeflateHuffmanTable::from_code_lengths(&lengths[..hlit])?;
                    let dt = DeflateHuffmanTable::from_code_lengths(&lengths[hlit..])?;
                    (lt, dt)
                };

                loop {
                    let sym = lit_table.decode(&mut br)?;
                    if sym < 256 {
                        out.push(sym as u8);
                    } else if sym == 256 {
                        break;
                    } else {
                        let l_idx = (sym as usize) - 257;
                        if l_idx >= LENGTH_BASE.len() {
                            return Err(RagdollError::InvalidData(
                                "DEFLATE invalid length symbol".into(),
                            ));
                        }
                        let length = (LENGTH_BASE[l_idx] as usize)
                            + (br.read_bits(LENGTH_EXTRA[l_idx] as u32)? as usize);
                        let d_sym = dist_table.decode(&mut br)? as usize;
                        if d_sym >= DIST_BASE.len() {
                            return Err(RagdollError::InvalidData(
                                "DEFLATE invalid distance symbol".into(),
                            ));
                        }
                        let distance = (DIST_BASE[d_sym] as usize)
                            + (br.read_bits(DIST_EXTRA[d_sym] as u32)? as usize);
                        if distance == 0 || distance > out.len() {
                            return Err(RagdollError::InvalidData(
                                "DEFLATE LZ77 distance out of bounds".into(),
                            ));
                        }
                        let start = out.len() - distance;
                        for i in 0..length {
                            let b = out[start + i];
                            out.push(b);
                        }
                    }
                }
            }
            _ => {
                return Err(RagdollError::InvalidData(
                    "reserved DEFLATE block type 3".into(),
                ));
            }
        }
        if bfinal != 0 {
            break;
        }
    }

    // Verify trailing 4-byte big-endian Adler-32
    br.align_to_byte();
    if let (Ok(b0), Ok(b1), Ok(b2), Ok(b3)) = (
        br.read_bits(8),
        br.read_bits(8),
        br.read_bits(8),
        br.read_bits(8),
    ) {
        let expected_adler = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
        let actual_adler = adler32(&out);
        if expected_adler != actual_adler {
            return Err(RagdollError::InvalidData(format!(
                "Zlib Adler-32 mismatch: expected {expected_adler:#010x}, got {actual_adler:#010x}"
            )));
        }
    }
    Ok(out)
}

struct DeflateBitWriter {
    buf: Vec<u8>,
    bit_buf: u64,
    bit_cnt: u32,
}

impl DeflateBitWriter {
    fn with_capacity(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
            bit_buf: 0,
            bit_cnt: 0,
        }
    }

    #[inline]
    fn write_bits_lsb(&mut self, val: u32, n: u32) {
        self.bit_buf |= ((val as u64) & ((1u64 << n) - 1)) << self.bit_cnt;
        self.bit_cnt += n;
        while self.bit_cnt >= 8 {
            self.buf.push((self.bit_buf & 0xFF) as u8);
            self.bit_buf >>= 8;
            self.bit_cnt -= 8;
        }
    }

    /// Writes an MSB-first Huffman code in DEFLATE bit-reversed order.
    #[inline]
    fn write_huffman_msb(&mut self, code: u16, len: u32) {
        let rev = code.reverse_bits() >> (16 - len);
        self.write_bits_lsb(rev as u32, len);
    }

    #[inline]
    fn write_fixed_sym(&mut self, sym: u16) {
        match sym {
            0..=143 => self.write_huffman_msb(0x30 + sym, 8),
            144..=255 => self.write_huffman_msb(0x190 + (sym - 144), 9),
            256..=279 => self.write_huffman_msb(sym - 256, 7),
            _ => self.write_huffman_msb(0xC0 + (sym - 280), 8),
        }
    }

    fn flush_align(&mut self) {
        if self.bit_cnt > 0 {
            self.buf.push((self.bit_buf & 0xFF) as u8);
            self.bit_buf = 0;
            self.bit_cnt = 0;
        }
    }
}

#[inline]
fn length_to_code(len: usize) -> (u16, u8, u32) {
    for i in (0..LENGTH_BASE.len()).rev() {
        let base = LENGTH_BASE[i] as usize;
        if len >= base {
            return (257 + (i as u16), LENGTH_EXTRA[i], (len - base) as u32);
        }
    }
    (257, 0, 0)
}

#[inline]
fn dist_to_code(dist: usize) -> (u16, u8, u32) {
    for i in (0..DIST_BASE.len()).rev() {
        let base = DIST_BASE[i] as usize;
        if dist >= base {
            return (i as u16, DIST_EXTRA[i], (dist - base) as u32);
        }
    }
    (0, 0, 0)
}

/// Compresses `raw` into an RFC 1950 Zlib-wrapped DEFLATE stream using hash-chain LZ77 + Fixed Huffman coding.
pub fn zlib_compress(raw: &[u8]) -> Vec<u8> {
    let mut bw = DeflateBitWriter::with_capacity(raw.len() / 2 + 64);
    // RFC 1950 Zlib header: CMF = 0x78 (Deflate, 32 KiB window), FLG = 0x9C (default compression, (0x789C % 31) == 0)
    bw.buf.push(0x78);
    bw.buf.push(0x9C);

    // Final Fixed Huffman block: BFINAL = 1, BTYPE = 01 (Fixed Huffman)
    bw.write_bits_lsb(1, 1);
    bw.write_bits_lsb(1, 2);

    const HASH_MASK: usize = 0x3FFF;
    let mut head = vec![usize::MAX; HASH_MASK + 1];
    let mut pos = 0usize;
    let n = raw.len();

    while pos < n {
        if pos + 3 <= n {
            let h = (((raw[pos] as usize) << 10)
                ^ ((raw[pos + 1] as usize) << 5)
                ^ (raw[pos + 2] as usize))
                & HASH_MASK;
            let prev = head[h];
            head[h] = pos;

            if prev != usize::MAX && pos > prev && (pos - prev) <= 32_768 {
                let dist = pos - prev;
                let max_len = (n - pos).min(258);
                let mut mlen = 0usize;
                while mlen < max_len && raw[prev + mlen] == raw[pos + mlen] {
                    mlen += 1;
                }
                if mlen >= 3 {
                    let (l_sym, l_eb, l_ev) = length_to_code(mlen);
                    bw.write_fixed_sym(l_sym);
                    if l_eb > 0 {
                        bw.write_bits_lsb(l_ev, l_eb as u32);
                    }
                    let (d_sym, d_eb, d_ev) = dist_to_code(dist);
                    bw.write_huffman_msb(d_sym, 5);
                    if d_eb > 0 {
                        bw.write_bits_lsb(d_ev, d_eb as u32);
                    }
                    for step in 1..mlen {
                        if pos + step + 3 <= n {
                            let sh = (((raw[pos + step] as usize) << 10)
                                ^ ((raw[pos + step + 1] as usize) << 5)
                                ^ (raw[pos + step + 2] as usize))
                                & HASH_MASK;
                            head[sh] = pos + step;
                        }
                    }
                    pos += mlen;
                    continue;
                }
            }
        }
        bw.write_fixed_sym(raw[pos] as u16);
        pos += 1;
    }

    // End-of-block symbol 256
    bw.write_fixed_sym(256);
    bw.flush_align();

    let adler = adler32(raw);
    bw.buf.extend_from_slice(&adler.to_be_bytes());
    bw.buf
}

fn write_png_chunk<W: Write>(writer: &mut W, chunk_type: &[u8; 4], payload: &[u8]) -> Result<u64> {
    let len = payload.len() as u32;
    writer.write_all(&len.to_be_bytes())?;
    writer.write_all(chunk_type)?;
    writer.write_all(payload)?;
    let mut crc_in = Vec::with_capacity(4 + payload.len());
    crc_in.extend_from_slice(chunk_type);
    crc_in.extend_from_slice(payload);
    let crc = crc32_png(&crc_in);
    writer.write_all(&crc.to_be_bytes())?;
    Ok(12 + payload.len() as u64)
}

/// Decodes a `.png` image (`RGB24`, `RGBA8`, `Gray8`, or `Palette`) into a [`VideoFrame`].
pub fn read_png_image(data: &[u8]) -> Result<VideoFrame> {
    if !is_png_bytes(data) {
        return Err(RagdollError::InvalidHeader(
            "missing 8-byte PNG signature".into(),
        ));
    }

    let mut pos = 8usize;
    let mut width = 0usize;
    let mut height = 0usize;
    let mut color_type = 2u8;
    let mut have_ihdr = false;
    let mut palette: Vec<[u8; 3]> = Vec::new();
    let mut trns: Vec<u8> = Vec::new();
    let mut idat_compressed = Vec::new();

    while pos + 12 <= data.len() {
        let len = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        let ctype = &data[pos + 4..pos + 8];
        if pos + 12 + len > data.len() {
            return Err(RagdollError::BitstreamEndOfData);
        }
        let body = &data[pos + 8..pos + 8 + len];
        let expected_crc =
            u32::from_be_bytes(data[pos + 8 + len..pos + 12 + len].try_into().unwrap());
        let actual_crc = crc32_png(&data[pos + 4..pos + 8 + len]);
        if actual_crc != expected_crc {
            return Err(RagdollError::InvalidData(format!(
                "PNG chunk {:?} CRC-32 mismatch: expected {expected_crc:#010x}, got {actual_crc:#010x}",
                String::from_utf8_lossy(ctype)
            )));
        }
        pos += 12 + len;

        match ctype {
            b"IHDR" => {
                if len < 13 {
                    return Err(RagdollError::InvalidHeader("PNG IHDR too short".into()));
                }
                width = u32::from_be_bytes(body[0..4].try_into().unwrap()) as usize;
                height = u32::from_be_bytes(body[4..8].try_into().unwrap()) as usize;
                let bit_depth = body[8];
                color_type = body[9];
                let interlace = body[12];
                if width == 0
                    || height == 0
                    || width > 16384
                    || height > 16384
                    || bit_depth != 8
                    || interlace != 0
                {
                    return Err(RagdollError::Unsupported(format!(
                        "PNG IHDR unsupported parameters: {width}x{height} depth={bit_depth} interlace={interlace}"
                    )));
                }
                have_ihdr = true;
            }
            b"PLTE" => {
                for &rgb in body.as_chunks::<3>().0 {
                    palette.push(rgb);
                }
            }
            b"tRNS" => {
                trns = body.to_vec();
            }
            b"IDAT" => {
                idat_compressed.extend_from_slice(body);
            }
            b"IEND" => break,
            _ => {}
        }
    }

    if !have_ihdr {
        return Err(RagdollError::InvalidHeader("PNG missing IHDR chunk".into()));
    }

    let bpp = match color_type {
        0 => 1usize, // Gray8
        2 => 3,      // RGB24
        3 => 1,      // Indexed Palette
        4 => 2,      // Gray8 + Alpha8
        6 => 4,      // RGBA8
        other => {
            return Err(RagdollError::Unsupported(format!(
                "unsupported PNG color_type {other}"
            )));
        }
    };

    let raw_stream = zlib_decompress(&idat_compressed)?;
    let row_bytes = width * bpp;
    let stride_with_filter = 1 + row_bytes;
    if raw_stream.len() < height * stride_with_filter {
        return Err(RagdollError::BitstreamEndOfData);
    }

    let mut prev_row = vec![0u8; row_bytes];
    let mut unfiltered_rows = Vec::with_capacity(height * row_bytes);

    for y in 0..height {
        let off = y * stride_with_filter;
        let ftype = raw_stream[off];
        let mut cur_row = raw_stream[off + 1..off + stride_with_filter].to_vec();
        unfilter_png_scanline(ftype, &mut cur_row, &prev_row, bpp)?;
        unfiltered_rows.extend_from_slice(&cur_row);
        prev_row = cur_row;
    }

    match color_type {
        0 => {
            let mut frame = VideoFrame::new(width, height, PixelFormat::Gray8);
            frame.colorimetry = Colorimetry::srgb();
            for y in 0..height {
                frame.plane_mut(0).row_mut(y)[..width]
                    .copy_from_slice(&unfiltered_rows[y * width..(y + 1) * width]);
            }
            Ok(frame)
        }
        2 => {
            let mut frame = VideoFrame::new(width, height, PixelFormat::Rgb24);
            frame.colorimetry = Colorimetry::srgb();
            for y in 0..height {
                frame.plane_mut(0).row_mut(y)[..width * 3]
                    .copy_from_slice(&unfiltered_rows[y * width * 3..(y + 1) * width * 3]);
            }
            Ok(frame)
        }
        6 => {
            let mut frame = VideoFrame::new(width, height, PixelFormat::Rgba8);
            frame.colorimetry = Colorimetry::srgb();
            for y in 0..height {
                frame.plane_mut(0).row_mut(y)[..width * 4]
                    .copy_from_slice(&unfiltered_rows[y * width * 4..(y + 1) * width * 4]);
            }
            Ok(frame)
        }
        4 => {
            // Gray + Alpha -> RGBA8
            let mut frame = VideoFrame::new(width, height, PixelFormat::Rgba8);
            frame.colorimetry = Colorimetry::srgb();
            for y in 0..height {
                let dst = frame.plane_mut(0).row_mut(y);
                let src = &unfiltered_rows[y * width * 2..(y + 1) * width * 2];
                for x in 0..width {
                    let g = src[x * 2];
                    let a = src[x * 2 + 1];
                    dst[x * 4] = g;
                    dst[x * 4 + 1] = g;
                    dst[x * 4 + 2] = g;
                    dst[x * 4 + 3] = a;
                }
            }
            Ok(frame)
        }
        3 => {
            // Palette -> RGBA8 (if tRNS present) or RGB24
            if !trns.is_empty() {
                let mut frame = VideoFrame::new(width, height, PixelFormat::Rgba8);
                frame.colorimetry = Colorimetry::srgb();
                for y in 0..height {
                    let dst = frame.plane_mut(0).row_mut(y);
                    let src = &unfiltered_rows[y * width..(y + 1) * width];
                    for x in 0..width {
                        let idx = src[x] as usize;
                        let rgb = palette.get(idx).copied().unwrap_or([0, 0, 0]);
                        let a = trns.get(idx).copied().unwrap_or(255);
                        dst[x * 4] = rgb[0];
                        dst[x * 4 + 1] = rgb[1];
                        dst[x * 4 + 2] = rgb[2];
                        dst[x * 4 + 3] = a;
                    }
                }
                Ok(frame)
            } else {
                let mut frame = VideoFrame::new(width, height, PixelFormat::Rgb24);
                frame.colorimetry = Colorimetry::srgb();
                for y in 0..height {
                    let dst = frame.plane_mut(0).row_mut(y);
                    let src = &unfiltered_rows[y * width..(y + 1) * width];
                    for x in 0..width {
                        let idx = src[x] as usize;
                        let rgb = palette.get(idx).copied().unwrap_or([0, 0, 0]);
                        dst[x * 3] = rgb[0];
                        dst[x * 3 + 1] = rgb[1];
                        dst[x * 3 + 2] = rgb[2];
                    }
                }
                Ok(frame)
            }
        }
        _ => unreachable!(),
    }
}

/// Encodes a [`VideoFrame`] (`Rgb24`, `Rgba8`, `Gray8`, or any YUV format automatically converted to `Rgb24`)
/// into a spec-compliant `.png` image stream.
pub fn write_png_image<W: Write>(mut writer: W, frame: &VideoFrame) -> Result<u64> {
    let converted;
    let src = match frame.format {
        PixelFormat::Rgb24 | PixelFormat::Rgba8 | PixelFormat::Gray8 => frame,
        _ => {
            converted = ensure_rgb24(frame);
            &converted
        }
    };

    let (color_type, bpp) = match src.format {
        PixelFormat::Gray8 => (0u8, 1usize),
        PixelFormat::Rgb24 => (2u8, 3usize),
        PixelFormat::Rgba8 => (6u8, 4usize),
        _ => unreachable!(),
    };

    let width = src.width;
    let height = src.height;
    let row_bytes = width * bpp;

    let mut filtered_buf = Vec::with_capacity(height * (1 + row_bytes));
    let zero_prev = vec![0u8; row_bytes];
    for y in 0..height {
        let cur = &src.plane(0).row(y)[..row_bytes];
        let prev = if y == 0 {
            &zero_prev[..]
        } else {
            &src.plane(0).row(y - 1)[..row_bytes]
        };
        filter_png_scanline_adaptive(cur, prev, bpp, &mut filtered_buf);
    }

    let idat_zlib = zlib_compress(&filtered_buf);

    let mut written = 0u64;
    writer.write_all(&PNG_SIGNATURE)?;
    written += 8;

    let mut ihdr = [0u8; 13];
    ihdr[0..4].copy_from_slice(&(width as u32).to_be_bytes());
    ihdr[4..8].copy_from_slice(&(height as u32).to_be_bytes());
    ihdr[8] = 8; // bit_depth = 8
    ihdr[9] = color_type;
    ihdr[10] = 0; // compression = 0 (Deflate)
    ihdr[11] = 0; // filter = 0 (adaptive 5-type)
    ihdr[12] = 0; // interlace = 0 (None)

    written += write_png_chunk(&mut writer, b"IHDR", &ihdr)?;
    written += write_png_chunk(&mut writer, b"IDAT", &idat_zlib)?;
    written += write_png_chunk(&mut writer, b"IEND", &[])?;
    writer.flush()?;
    Ok(written)
}

/// Returns `true` if `path_str` is an image-sequence printf pattern (e.g. `frame_%04d.png` or `img_%d.jpg`).
pub fn is_image_sequence_pattern(path_str: &str) -> bool {
    let lower = path_str.to_ascii_lowercase();
    let is_img_ext = lower.ends_with(".png")
        || lower.ends_with(".jpg")
        || lower.ends_with(".jpeg")
        || lower.ends_with(".ppm")
        || lower.ends_with(".pgm");
    if !is_img_ext {
        return false;
    }
    parse_printf_index_spec(path_str).is_some()
}

fn parse_printf_index_spec(pattern: &str) -> Option<(usize, usize, usize)> {
    let bytes = pattern.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let start = i;
            i += 1;
            if i < bytes.len() && bytes[i] == b'%' {
                i += 1;
                continue;
            }
            let mut pad_width = 0usize;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                pad_width = pad_width * 10 + (bytes[i] - b'0') as usize;
                i += 1;
            }
            if i < bytes.len() && bytes[i] == b'd' {
                return Some((start, i + 1, pad_width.max(1)));
            }
        } else {
            i += 1;
        }
    }
    None
}

/// Formats an image-sequence pattern (`frame_%04d.png`) for 1-based frame number `number` (`1, 2, 3, ...`).
pub fn format_image_sequence_path(pattern: &str, number: usize) -> String {
    if let Some((start, end, width)) = parse_printf_index_spec(pattern) {
        format!(
            "{}{:0width$}{}",
            &pattern[..start],
            number,
            &pattern[end..],
            width = width
        )
    } else {
        pattern.to_string()
    }
}

/// Expands an input image-sequence pattern (`frame_%04d.png`) into existing numbered files on disk.
pub fn expand_image_sequence_inputs(pattern: &str) -> Result<Vec<std::path::PathBuf>> {
    let mut paths = Vec::new();
    let start_num = if std::path::Path::new(&format_image_sequence_path(pattern, 0)).exists() {
        0usize
    } else {
        1usize
    };
    let mut n = start_num;
    loop {
        let candidate = std::path::PathBuf::from(format_image_sequence_path(pattern, n));
        if candidate.exists() {
            paths.push(candidate);
            n += 1;
        } else {
            break;
        }
    }
    if paths.is_empty() {
        return Err(RagdollError::InvalidHeader(format!(
            "no matching files found for image sequence pattern '{pattern}'"
        )));
    }
    Ok(paths)
}
