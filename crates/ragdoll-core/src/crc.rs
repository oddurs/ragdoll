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

/// Updates a running FLAC CRC-16 (`poly = 0x8005`, MSB-first).
///
/// Inputs of 64 bytes or more fold 16 bytes per carry-less multiply on CPUs with `PCLMULQDQ`
/// (x86_64 AVX2 dispatch level) or `PMULL` (aarch64 NEON level with the AES extension); everything
/// else uses Slicing-by-8 ([`crc16_flac_update_scalar`]).
#[inline]
pub fn crc16_flac_update(crc: u16, data: &[u8]) -> u16 {
    if data.len() >= 64 {
        #[cfg(target_arch = "x86_64")]
        if crate::dispatch::avx2() && std::is_x86_feature_detected!("pclmulqdq") {
            // Safety: AVX2 implies SSSE3/SSE4.1, and PCLMULQDQ was detected just above.
            return unsafe { crc16_flac_update_clmul(crc, data) };
        }
        #[cfg(target_arch = "aarch64")]
        if crate::dispatch::neon() && std::arch::is_aarch64_feature_detected!("aes") {
            // Safety: NEON is baseline on aarch64 and the AES extension (PMULL) was detected.
            return unsafe { crc16_flac_update_pmull(crc, data) };
        }
    }
    crc16_flac_update_scalar(crc, data)
}

/// `x^n mod P(x)` for the FLAC CRC-16 polynomial `P = x^16 + x^15 + x^2 + 1`.
const fn crc16_flac_xpow_mod(n: u32) -> u64 {
    let mut r = 1u32;
    let mut i = 0;
    while i < n {
        r <<= 1;
        if r & 0x1_0000 != 0 {
            r ^= 0x1_8005;
        }
        i += 1;
    }
    r as u64
}

/// Folding constants: a 128-bit block `X = hi * x^64 + lo` followed by 128 more message bits is
/// congruent to `hi * (x^192 mod P) + lo * (x^128 mod P)` modulo `P`.
const CRC16_FLAC_FOLD_HI: u64 = crc16_flac_xpow_mod(192);
const CRC16_FLAC_FOLD_LO: u64 = crc16_flac_xpow_mod(128);

/// Carry-less folding CRC-16 (Intel, "Fast CRC Computation for Generic Polynomials Using
/// PCLMULQDQ"): the message is reduced 16 bytes at a time to a congruent 16-byte remainder, whose
/// CRC (followed by the unaligned tail) is the CRC of the whole message. The running `crc` enters
/// by XOR into the first two message bytes, as the CRC has no reflection or output XOR.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq,ssse3,sse4.1")]
unsafe fn crc16_flac_update_clmul(crc: u16, data: &[u8]) -> u16 {
    use std::arch::x86_64::*;
    let (blocks, tail) = data.as_chunks::<16>();
    let reverse = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    // Message byte 0 becomes the most significant byte of the 128-bit polynomial.
    let load = |b: &[u8; 16]| unsafe {
        _mm_shuffle_epi8(_mm_loadu_si128(b.as_ptr() as *const __m128i), reverse)
    };
    let k = _mm_set_epi64x(CRC16_FLAC_FOLD_HI as i64, CRC16_FLAC_FOLD_LO as i64);
    let mut x = _mm_xor_si128(
        load(&blocks[0]),
        _mm_set_epi64x(((crc as u64) << 48) as i64, 0),
    );
    for b in &blocks[1..] {
        let hi = _mm_clmulepi64_si128(x, k, 0x11);
        let lo = _mm_clmulepi64_si128(x, k, 0x00);
        x = _mm_xor_si128(_mm_xor_si128(hi, lo), load(b));
    }
    let mut rem = [0u8; 16];
    unsafe {
        _mm_storeu_si128(
            rem.as_mut_ptr() as *mut __m128i,
            _mm_shuffle_epi8(x, reverse),
        )
    };
    crc16_flac_update_scalar(crc16_flac_update_scalar(0, &rem), tail)
}

/// aarch64 `PMULL` twin of [`crc16_flac_update_clmul`].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,aes")]
unsafe fn crc16_flac_update_pmull(crc: u16, data: &[u8]) -> u16 {
    use std::arch::aarch64::*;
    let (blocks, tail) = data.as_chunks::<16>();
    // Message byte 0 becomes the most significant byte of the 128-bit polynomial.
    let load = |b: &[u8; 16]| -> u128 { u128::from_be_bytes(*b) };
    let mut x = load(&blocks[0]) ^ ((crc as u128) << 112);
    for b in &blocks[1..] {
        let hi = vmull_p64((x >> 64) as u64, CRC16_FLAC_FOLD_HI);
        let lo = vmull_p64(x as u64, CRC16_FLAC_FOLD_LO);
        x = hi ^ lo ^ load(b);
    }
    crc16_flac_update_scalar(crc16_flac_update_scalar(0, &x.to_be_bytes()), tail)
}

/// Updates a running FLAC CRC-16 (`poly = 0x8005`, MSB-first) using Slicing-by-8 (`libFLAC` algorithm).
#[inline]
pub fn crc16_flac_update_scalar(mut crc: u16, data: &[u8]) -> u16 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc16_flac_dispatch_matches_scalar() {
        // Odd lengths around the 64-byte dispatch threshold and the 16-byte fold, running CRCs and
        // extremal bytes.
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut data = vec![0u8; 5000];
        for b in &mut data {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            *b = seed as u8;
        }
        for len in (0..300).chain([1023, 1024, 1025, 4096, 4999, 5000]) {
            for init in [0u16, 1, 0x8005, 0xFFFF] {
                let msg = &data[..len];
                assert_eq!(
                    crc16_flac_update(init, msg),
                    crc16_flac_update_scalar(init, msg),
                    "len {len} init {init:#06x}"
                );
            }
        }
        for fill in [0x00u8, 0xFF] {
            let msg = vec![fill; 777];
            assert_eq!(crc16_flac(&msg), crc16_flac_update_scalar(0, &msg));
        }
    }
}
