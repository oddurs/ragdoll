//! Zero-overhead compile-time CRC kernels for container and codec frame verification.
//!
//! Modeled after `libFLAC/src/libFLAC/crc.c` and FFmpeg `libavutil/crc.c`:
//! - **CRC-8 (`0x07`)**: Table-driven 8-bit polynomial $x^8 + x^2 + x + 1$ for FLAC frame headers.
//! - **CRC-16 (`0x8005`)**: Slicing-by-8 (`8 x 256` lookup table) 16-bit polynomial $x^{16} + x^{15} + x^2 + 1$
//!   processing 8 bytes per loop iteration with independent table lookups for superscalar execution,
//!   used by FLAC frame footers.
//! - **CRC-32 (`0x04C11DB7`)**: Slicing-by-8 (`8 x 256` lookup table) 32-bit IEEE 802.3 MSB-first
//!   polynomial for MPEG-TS PSI tables (`PAT`/`PMT`, `init = 0xFFFF_FFFF`) and Ogg pages (`init = 0`).

const FLAC_CRC8_POLY: u8 = 0x07;
const FLAC_CRC16_POLY: u16 = 0x8005;
const MPEG2_CRC32_POLY: u32 = 0x04C1_1DB7;

const CRC8_TABLE: [u8; 256] = {
    let mut table = [0u8; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = i as u8;
        let mut bit = 0;
        while bit < 8 {
            crc = if (crc & 0x80) != 0 {
                (crc << 1) ^ FLAC_CRC8_POLY
            } else {
                crc << 1
            };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

const CRC16_SLICE8_TABLE: [[u16; 256]; 8] = {
    let mut table = [[0u16; 256]; 8];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = (i as u16) << 8;
        let mut bit = 0;
        while bit < 8 {
            crc = if (crc & 0x8000) != 0 {
                (crc << 1) ^ FLAC_CRC16_POLY
            } else {
                crc << 1
            };
            bit += 1;
        }
        table[0][i] = crc;
        i += 1;
    }
    let mut t = 1usize;
    while t < 8 {
        let mut j = 0usize;
        while j < 256 {
            let prev = table[t - 1][j];
            table[t][j] = (prev << 8) ^ table[0][(prev >> 8) as usize];
            j += 1;
        }
        t += 1;
    }
    table
};

const CRC32_SLICE8_TABLE: [[u32; 256]; 8] = {
    let mut table = [[0u32; 256]; 8];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = (i as u32) << 24;
        let mut bit = 0;
        while bit < 8 {
            crc = if (crc & 0x8000_0000) != 0 {
                (crc << 1) ^ MPEG2_CRC32_POLY
            } else {
                crc << 1
            };
            bit += 1;
        }
        table[0][i] = crc;
        i += 1;
    }
    let mut t = 1usize;
    while t < 8 {
        let mut j = 0usize;
        while j < 256 {
            let prev = table[t - 1][j];
            table[t][j] = (prev << 8) ^ table[0][(prev >> 24) as usize];
            j += 1;
        }
        t += 1;
    }
    table
};

/// Updates a running FLAC CRC-8 (`poly = 0x07`, MSB-first) over `data`.
#[inline]
pub fn crc8_flac_update(mut crc: u8, data: &[u8]) -> u8 {
    for &b in data {
        crc = CRC8_TABLE[(crc ^ b) as usize];
    }
    crc
}

/// Computes the FLAC frame header CRC-8 (`poly = 0x07`, `init = 0x00`).
#[inline]
pub fn crc8_flac(data: &[u8]) -> u8 {
    crc8_flac_update(0, data)
}

/// Updates a running FLAC CRC-16 (`poly = 0x8005`, MSB-first) using Slicing-by-8 (`libFLAC` algorithm).
#[inline]
pub fn crc16_flac_update(mut crc: u16, data: &[u8]) -> u16 {
    let (chunks, remainder) = data.as_chunks::<8>();
    for chunk in chunks {
        crc ^= ((chunk[0] as u16) << 8) | (chunk[1] as u16);
        crc = CRC16_SLICE8_TABLE[7][(crc >> 8) as usize]
            ^ CRC16_SLICE8_TABLE[6][(crc & 0xFF) as usize]
            ^ CRC16_SLICE8_TABLE[5][chunk[2] as usize]
            ^ CRC16_SLICE8_TABLE[4][chunk[3] as usize]
            ^ CRC16_SLICE8_TABLE[3][chunk[4] as usize]
            ^ CRC16_SLICE8_TABLE[2][chunk[5] as usize]
            ^ CRC16_SLICE8_TABLE[1][chunk[6] as usize]
            ^ CRC16_SLICE8_TABLE[0][chunk[7] as usize];
    }
    for &b in remainder {
        crc = (crc << 8) ^ CRC16_SLICE8_TABLE[0][((crc >> 8) as u8 ^ b) as usize];
    }
    crc
}

/// Computes the FLAC frame footer CRC-16 (`poly = 0x8005`, `init = 0x0000`) via Slicing-by-8.
#[inline]
pub fn crc16_flac(data: &[u8]) -> u16 {
    crc16_flac_update(0, data)
}

/// Updates a running MSB-first CRC-32 (`poly = 0x04C11DB7`) using Slicing-by-8.
#[inline]
pub fn crc32_msb_update(mut crc: u32, data: &[u8]) -> u32 {
    let (chunks, remainder) = data.as_chunks::<8>();
    for chunk in chunks {
        let w0 = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) ^ crc;
        let w1 = u32::from_be_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        crc = CRC32_SLICE8_TABLE[7][(w0 >> 24) as usize]
            ^ CRC32_SLICE8_TABLE[6][((w0 >> 16) & 0xFF) as usize]
            ^ CRC32_SLICE8_TABLE[5][((w0 >> 8) & 0xFF) as usize]
            ^ CRC32_SLICE8_TABLE[4][(w0 & 0xFF) as usize]
            ^ CRC32_SLICE8_TABLE[3][(w1 >> 24) as usize]
            ^ CRC32_SLICE8_TABLE[2][((w1 >> 16) & 0xFF) as usize]
            ^ CRC32_SLICE8_TABLE[1][((w1 >> 8) & 0xFF) as usize]
            ^ CRC32_SLICE8_TABLE[0][(w1 & 0xFF) as usize];
    }
    for &b in remainder {
        crc = (crc << 8) ^ CRC32_SLICE8_TABLE[0][((crc >> 24) as u8 ^ b) as usize];
    }
    crc
}

/// Computes the MPEG-2 / MPEG-TS PSI (`PAT`/`PMT`) CRC-32 (`poly = 0x04C11DB7`, `init = 0xFFFF_FFFF`).
#[inline]
pub fn crc32_mpeg2(data: &[u8]) -> u32 {
    crc32_msb_update(0xFFFF_FFFF, data)
}

/// Computes the Ogg container page CRC-32 (`poly = 0x04C11DB7`, `init = 0x0000_0000`).
#[inline]
pub fn crc32_ogg(data: &[u8]) -> u32 {
    crc32_msb_update(0, data)
}

const PNG_CRC32_POLY_REFLECTED: u32 = 0xEDB8_8320;

const CRC32_LSB_SLICE8_TABLE: [[u32; 256]; 8] = {
    let mut table = [[0u32; 256]; 8];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if (crc & 1) != 0 {
                (crc >> 1) ^ PNG_CRC32_POLY_REFLECTED
            } else {
                crc >> 1
            };
            bit += 1;
        }
        table[0][i] = crc;
        i += 1;
    }
    let mut t = 1usize;
    while t < 8 {
        let mut j = 0usize;
        while j < 256 {
            let prev = table[t - 1][j];
            table[t][j] = (prev >> 8) ^ table[0][(prev & 0xFF) as usize];
            j += 1;
        }
        t += 1;
    }
    table
};

/// Updates a running LSB-first reflected IEEE 802.3 CRC-32 (`poly = 0xEDB88320`, PNG / Gzip) using Slicing-by-8.
#[inline]
pub fn crc32_lsb_update(mut crc: u32, data: &[u8]) -> u32 {
    let (chunks, remainder) = data.as_chunks::<8>();
    for chunk in chunks {
        let w0 = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) ^ crc;
        let w1 = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        crc = CRC32_LSB_SLICE8_TABLE[7][(w0 & 0xFF) as usize]
            ^ CRC32_LSB_SLICE8_TABLE[6][((w0 >> 8) & 0xFF) as usize]
            ^ CRC32_LSB_SLICE8_TABLE[5][((w0 >> 16) & 0xFF) as usize]
            ^ CRC32_LSB_SLICE8_TABLE[4][(w0 >> 24) as usize]
            ^ CRC32_LSB_SLICE8_TABLE[3][(w1 & 0xFF) as usize]
            ^ CRC32_LSB_SLICE8_TABLE[2][((w1 >> 8) & 0xFF) as usize]
            ^ CRC32_LSB_SLICE8_TABLE[1][((w1 >> 16) & 0xFF) as usize]
            ^ CRC32_LSB_SLICE8_TABLE[0][(w1 >> 24) as usize];
    }
    for &b in remainder {
        crc = (crc >> 8) ^ CRC32_LSB_SLICE8_TABLE[0][((crc as u8) ^ b) as usize];
    }
    crc
}

/// Computes the IEEE 802.3 / PNG chunk CRC-32 (`poly = 0xEDB88320`, `init = 0xFFFF_FFFF`, `xorout = 0xFFFF_FFFF`).
#[inline]
pub fn crc32_png(data: &[u8]) -> u32 {
    !crc32_lsb_update(0xFFFF_FFFF, data)
}

/// Computes the RFC 1950 Zlib Adler-32 checksum (`MOD_ADLER = 65521`, 5552-byte NMAX deferred-modulo blocks).
#[inline]
pub fn adler32(data: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65_521;
    let mut s1 = 1u32;
    let mut s2 = 0u32;
    for chunk in data.chunks(5552) {
        for &b in chunk {
            s1 += b as u32;
            s2 += s1;
        }
        s1 %= MOD_ADLER;
        s2 %= MOD_ADLER;
    }
    (s2 << 16) | s1
}
