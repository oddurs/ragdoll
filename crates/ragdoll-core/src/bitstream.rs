use crate::error::{RagdollError, Result};

/// High-throughput, branchless 64-bit MSB-first bitstream reader.
///
/// Maintains a left-justified 64-bit shift register (`bit_buf`) where bit 63 is
/// always the next bit in the stream. Variable-length codes (Exp-Golomb `ue(v)`/`se(v)`
/// and Golomb-Rice) decode directly via hardware `leading_zeros()` (`CLZ` on ARM64 / `LZCNT` on x86_64).
#[derive(Clone, Debug)]
pub struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_buf: u64,
    bits_in_buf: u32,
}

impl<'a> BitReader<'a> {
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        let mut reader = Self {
            data,
            byte_pos: 0,
            bit_buf: 0,
            bits_in_buf: 0,
        };
        reader.refill();
        reader
    }

    /// Refills the 64-bit shift register so `bits_in_buf >= 56` whenever input bytes remain.
    ///
    /// Bits of `bit_buf` below the `bits_in_buf` valid ones are either zero or the stream's
    /// next bits, so loading a byte again over them is harmless.
    #[inline(always)]
    pub fn refill(&mut self) {
        if self.byte_pos + 8 <= self.data.len() {
            if self.bits_in_buf < 64 {
                let chunk: [u8; 8] = self.data[self.byte_pos..self.byte_pos + 8]
                    .try_into()
                    .unwrap();
                self.bit_buf |= u64::from_be_bytes(chunk) >> self.bits_in_buf;
                let bytes = (63 - self.bits_in_buf) >> 3;
                self.byte_pos += bytes as usize;
                self.bits_in_buf += bytes * 8;
            }
            return;
        }
        // By value, so a reader copied into locals can stay in registers.
        (self.bit_buf, self.bits_in_buf, self.byte_pos) =
            refill_tail(self.data, self.bit_buf, self.bits_in_buf, self.byte_pos);
    }

    /// Runs `f` on a copy of the reader in locals and keeps its progress: a reader behind a
    /// reference is reloaded from and stored to memory around every element, a local copy can
    /// stay in registers across a tight decoding loop.
    #[inline(always)]
    pub fn with_local<R>(&mut self, f: impl FnOnce(&mut BitReader<'a>) -> R) -> R {
        let mut local = self.clone();
        let res = f(&mut local);
        self.byte_pos = local.byte_pos;
        self.bit_buf = local.bit_buf;
        self.bits_in_buf = local.bits_in_buf;
        res
    }

    /// Total number of unread bits remaining in the slice.
    #[inline]
    pub const fn bits_remaining(&self) -> usize {
        (self.data.len() - self.byte_pos) * 8 + (self.bits_in_buf as usize)
    }

    /// Total number of bits consumed from the start of the slice.
    #[inline]
    pub const fn bits_read(&self) -> usize {
        self.data.len() * 8 - self.bits_remaining()
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.bits_remaining() == 0
    }

    #[inline]
    pub const fn is_byte_aligned(&self) -> bool {
        (self.bits_in_buf & 7) == 0
    }

    /// Peeks `n` bits (`1 <= n <= 56`) without advancing the stream.
    #[inline]
    pub fn peek_bits(&mut self, n: u32) -> Result<u64> {
        if n == 0 {
            return Ok(0);
        }
        debug_assert!(n <= 56);
        if self.bits_in_buf < n {
            self.refill();
            if self.bits_in_buf < n {
                return Err(RagdollError::BitstreamEndOfData);
            }
        }
        Ok(self.bit_buf >> (64 - n))
    }

    /// The next 32 bits without consuming them; bits past the end of the data read as zero.
    #[inline(always)]
    pub fn peek32(&mut self) -> u32 {
        if self.bits_in_buf < 32 {
            self.refill();
        }
        (self.bit_buf >> 32) as u32
    }

    /// The next 32 bits without consuming them, of which the first `need` (`0..=32`) are valid
    /// until the end of the data (zero past it); the rest may read as zero. Refilling only when
    /// fewer than `need` bits are buffered refills less often than [`Self::peek32`].
    #[inline(always)]
    pub fn peek32_min(&mut self, need: u32) -> u32 {
        if self.bits_in_buf < need {
            self.refill();
        }
        (self.bit_buf >> 32) as u32
    }

    /// Consumes `n` (`0..=32`) bits seen by [`Self::peek32`]; fails if they run past the end of
    /// the data.
    #[inline(always)]
    pub fn consume_peeked(&mut self, n: u32) -> Result<()> {
        if n > self.bits_in_buf {
            return Err(RagdollError::BitstreamEndOfData);
        }
        self.bit_buf <<= n & 63;
        self.bits_in_buf -= n;
        Ok(())
    }

    /// Consumes `n` (`0..=32`) bits seen by [`Self::peek32`] (or valid ones of
    /// [`Self::peek32_min`]) without checking for the end of the data. A decoder that consumes
    /// a run of codes this way calls [`Self::check_skipped`] before it trusts what it read and
    /// before any other method.
    #[inline(always)]
    pub fn skip_peeked(&mut self, n: u32) {
        self.bit_buf <<= n & 63;
        self.bits_in_buf = self.bits_in_buf.wrapping_sub(n);
    }

    /// Fails if [`Self::skip_peeked`] went past the end of the data, leaving the reader at its
    /// end.
    #[inline(always)]
    pub fn check_skipped(&mut self) -> Result<()> {
        if self.bits_in_buf > 64 {
            self.bit_buf = 0;
            self.bits_in_buf = 0;
            return Err(RagdollError::BitstreamEndOfData);
        }
        Ok(())
    }

    /// Advances the stream by `n` bits (`n <= bits_in_buf`).
    #[inline(always)]
    pub fn consume_buffered(&mut self, n: u32) {
        debug_assert!(n <= self.bits_in_buf);
        if n == 64 {
            self.bit_buf = 0;
        } else {
            self.bit_buf <<= n;
        }
        self.bits_in_buf -= n;
    }

    /// Reads a single bit as a `bool`.
    #[inline]
    pub fn read_bit(&mut self) -> Result<bool> {
        if self.bits_in_buf == 0 {
            self.refill();
            if self.bits_in_buf == 0 {
                return Err(RagdollError::BitstreamEndOfData);
            }
        }
        let bit = (self.bit_buf >> 63) != 0;
        self.bit_buf <<= 1;
        self.bits_in_buf -= 1;
        Ok(bit)
    }

    /// Reads `n` unsigned bits (`0 <= n <= 64`) in MSB-first order.
    #[inline]
    pub fn read_bits(&mut self, n: u32) -> Result<u64> {
        if n == 0 {
            return Ok(0);
        }
        if n > 64 {
            return Err(RagdollError::InvalidSyntax("read_bits width > 64"));
        }
        if n <= 56 {
            if self.bits_in_buf < n {
                self.refill();
                if self.bits_in_buf < n {
                    return Err(RagdollError::BitstreamEndOfData);
                }
            }
            let val = self.bit_buf >> (64 - n);
            self.bit_buf <<= n;
            self.bits_in_buf -= n;
            return Ok(val);
        }
        // 57..=64 bits: split into 32 + (n - 32)
        let hi = self.read_bits(32)?;
        let lo_bits = n - 32;
        let lo = self.read_bits(lo_bits)?;
        Ok((hi << lo_bits) | lo)
    }

    /// Reads `n` two's-complement signed bits (`1 <= n <= 64`).
    #[inline]
    pub fn read_signed_bits(&mut self, n: u32) -> Result<i64> {
        if n == 0 || n > 64 {
            return Err(RagdollError::InvalidSyntax(
                "read_signed_bits width must be 1..=64",
            ));
        }
        let raw = self.read_bits(n)?;
        let shift = 64 - n;
        Ok(((raw << shift) as i64) >> shift)
    }

    /// Skips `n` bits forward.
    pub fn skip_bits(&mut self, mut n: usize) -> Result<()> {
        while n > 0 {
            let step = n.min(32) as u32;
            self.read_bits(step)?;
            n -= step as usize;
        }
        Ok(())
    }

    /// Aligns the reader to the next byte boundary, discarding `0..=7` padding bits.
    #[inline]
    pub fn align_to_byte(&mut self) {
        let rem = self.bits_in_buf & 7;
        if rem != 0 {
            self.bit_buf <<= rem;
            self.bits_in_buf -= rem;
        }
    }

    /// Decodes an H.264 / HEVC Unsigned Exp-Golomb code `ue(v)` using hardware `CLZ`.
    #[inline]
    pub fn read_ue(&mut self) -> Result<u64> {
        // A code of up to 31 bits (a value below 65535) already in the register: its top
        // `2 * lz + 1` bits are `value + 1`. Small enough to inline everywhere.
        let lz = self.bit_buf.leading_zeros();
        let n = 2 * lz + 1;
        if lz < 16 && n <= self.bits_in_buf {
            let code = self.bit_buf >> (64 - n);
            self.bit_buf <<= n;
            self.bits_in_buf -= n;
            return Ok(code - 1);
        }
        self.read_ue_refill()
    }

    /// [`Self::read_ue`] after a refill, for long codes and the end of the data.
    #[inline(never)]
    fn read_ue_refill(&mut self) -> Result<u64> {
        if self.bits_in_buf < 32 {
            self.refill();
        }
        let lz = self.bit_buf.leading_zeros();
        let n = 2 * lz + 1;
        if lz < 16 && n <= self.bits_in_buf {
            let code = self.bit_buf >> (64 - n);
            self.bit_buf <<= n;
            self.bits_in_buf -= n;
            return Ok(code - 1);
        }
        if self.bits_in_buf == 0 || lz >= self.bits_in_buf {
            return Err(RagdollError::BitstreamEndOfData);
        }
        if lz > 32 {
            return Err(RagdollError::InvalidSyntax(
                "Exp-Golomb leading zeros exceed 32 bits",
            ));
        }
        self.consume_buffered(lz + 1);
        let info = self.read_bits(lz)?;
        Ok(((1u64 << lz) - 1) + info)
    }

    /// Decodes an H.264 / HEVC Signed Exp-Golomb code `se(v)` using hardware `CLZ` + `CSNEG`.
    #[inline]
    pub fn read_se(&mut self) -> Result<i64> {
        let code_num = self.read_ue()?;
        let mag = ((code_num + 1) >> 1) as i64;
        if (code_num & 1) != 0 {
            Ok(mag)
        } else {
            Ok(-mag)
        }
    }

    /// Decodes an H.264 Truncated Exp-Golomb code `te(v)` with syntax upper bound `max` (`get_te0_golomb`).
    ///
    /// When `max == 1`, a single bit is read and inverted (`1` -> `0`, `0` -> `1`).
    /// When `max > 1`, decodes as standard `ue(v)` and validates `v <= max`.
    #[inline]
    pub fn read_te(&mut self, max: u64) -> Result<u64> {
        if max == 0 {
            return Ok(0);
        }
        if max == 1 {
            let bit = self.read_bit()?;
            return Ok(u64::from(!bit));
        }
        let val = self.read_ue()?;
        if val > max {
            return Err(RagdollError::InvalidSyntax(
                "truncated Exp-Golomb te(v) exceeds maximum",
            ));
        }
        Ok(val)
    }

    /// Decodes a $k$-th order Unsigned Exp-Golomb code `ue(k)` (used in HEVC, AV1, and FFV1).
    #[inline]
    pub fn read_ue_k(&mut self, k: u32) -> Result<u64> {
        let q = self.read_ue()?;
        if k == 0 {
            Ok(q)
        } else {
            let r = self.read_bits(k)?;
            Ok((q << k) | r)
        }
    }

    /// Decodes a $k$-th order Signed Exp-Golomb code `se(k)`.
    #[inline]
    pub fn read_se_k(&mut self, k: u32) -> Result<i64> {
        let code_num = self.read_ue_k(k)?;
        let mag = ((code_num + 1) >> 1) as i64;
        if (code_num & 1) != 0 {
            Ok(mag)
        } else {
            Ok(-mag)
        }
    }

    /// Decodes an AV1 non-power-of-two uniform unsigned integer in `0..max` (`ns(n)` / `dav1d_get_uniform`).
    #[inline]
    pub fn read_uniform(&mut self, max: u32) -> Result<u32> {
        if max <= 1 {
            return Ok(0);
        }
        let l = 32 - max.leading_zeros();
        let m = (1u32 << l) - max;
        let v = self.read_bits(l - 1)? as u32;
        if v < m {
            Ok(v)
        } else {
            let extra_bit = u32::from(self.read_bit()?);
            Ok((v << 1) - m + extra_bit)
        }
    }

    /// Decodes an AV1 subexponentially coded value centered around `ref_val` in `0..(1 << n)` (`dav1d_get_bits_subexp`).
    pub fn read_subexp(&mut self, ref_val: u32, n: u32) -> Result<u32> {
        let mut i = 0u32;
        let mut mk = 0u32;
        let k = 3u32;
        let v = loop {
            let b = if i != 0 { k + i - 1 } else { k };
            if b >= n {
                break self.read_uniform((1u32 << n) - mk)? + mk;
            }
            if !self.read_bit()? {
                break (self.read_bits(b)? as u32) + mk;
            }
            mk += 1u32 << b;
            i += 1;
        };
        // Recenter non-negative around ref_val within [0, 1 << n)
        let max = 1u32 << n;
        let d = if (ref_val << 1) <= max {
            let inv = if (v & 1) != 0 {
                ref_val.wrapping_sub((v + 1) >> 1)
            } else {
                ref_val.wrapping_add(v >> 1)
            };
            if v <= (ref_val << 1) { inv } else { v }
        } else {
            let inv_ref = max - 1 - ref_val;
            let mapped = if v <= (inv_ref << 1) {
                if (v & 1) != 0 {
                    inv_ref.wrapping_sub((v + 1) >> 1)
                } else {
                    inv_ref.wrapping_add(v >> 1)
                }
            } else {
                v
            };
            max - 1 - mapped
        };
        Ok(d)
    }

    /// ITU-T H.264 / HEVC `more_rbsp_data()`: returns `true` if more syntax bits remain
    /// before the terminating `rbsp_stop_one_bit` (`1` followed only by `0` bits to the end of the slice).
    ///
    /// The stop bit is the last `1` bit of the payload; trailing zero bytes (`cabac_zero_word`,
    /// `trailing_zero_8bits`) are ignored.
    #[inline]
    pub fn more_rbsp_data(&self) -> bool {
        let Some(last) = self.data.iter().rposition(|&b| b != 0) else {
            return false;
        };
        let stop_bit = last * 8 + 7 - self.data[last].trailing_zeros() as usize;
        self.bits_read() < stop_bit
    }

    /// Counts consecutive `0` bits up to and consuming the terminating `1` stop bit.
    /// Used by FLAC and Ragdoll Fast-Intra Rice entropy decoders.
    #[inline]
    pub fn read_unary_zeros(&mut self) -> Result<u64> {
        let mut count = 0u64;
        loop {
            if self.bits_in_buf == 0 {
                self.refill();
                if self.bits_in_buf == 0 {
                    return Err(RagdollError::BitstreamEndOfData);
                }
            }
            let lz = self.bit_buf.leading_zeros();
            if lz < self.bits_in_buf {
                count += lz as u64;
                self.consume_buffered(lz + 1);
                return Ok(count);
            }
            count += self.bits_in_buf as u64;
            self.bit_buf = 0;
            self.bits_in_buf = 0;
        }
    }

    /// Decodes an unsigned Golomb-Rice value with parameter `k` (`(q << k) | r`).
    #[inline]
    pub fn read_rice_unsigned(&mut self, k: u32) -> Result<u64> {
        let q = self.read_unary_zeros()?;
        let r = if k > 0 { self.read_bits(k)? } else { 0 };
        Ok((q << k) | r)
    }

    /// Decodes a zig-zag folded signed Golomb-Rice residual (FLAC & Ragdoll Fast-Intra convention).
    #[inline]
    pub fn read_rice_signed(&mut self, k: u32) -> Result<i64> {
        let u = self.read_rice_unsigned(k)?;
        Ok(((u >> 1) as i64) ^ -((u & 1) as i64))
    }
}

/// [`BitReader::refill`] within the last 8 bytes of the data: `(bit_buf, bits_in_buf, byte_pos)`.
#[inline(never)]
fn refill_tail(data: &[u8], mut buf: u64, mut bits: u32, mut pos: usize) -> (u64, u32, usize) {
    while bits <= 56 && pos < data.len() {
        buf |= (data[pos] as u64) << (56 - bits);
        bits += 8;
        pos += 1;
    }
    (buf, bits, pos)
}

/// High-throughput 64-bit accumulator bitstream writer.
///
/// Accumulates bits in a 64-bit register and flushes 8-byte big-endian words directly into a `Vec<u8>`.
#[derive(Clone, Debug, Default)]
pub struct BitWriter {
    out: Vec<u8>,
    bit_buf: u64,
    bits_in_buf: u32,
}

impl BitWriter {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn with_capacity(bytes: usize) -> Self {
        Self {
            out: Vec::with_capacity(bytes),
            bit_buf: 0,
            bits_in_buf: 0,
        }
    }

    /// Total bits written so far.
    #[inline]
    pub const fn bits_written(&self) -> usize {
        self.out.len() * 8 + (self.bits_in_buf as usize)
    }

    /// Writes a single boolean bit (`1` or `0`).
    #[inline]
    pub fn write_bit(&mut self, bit: bool) {
        self.write_bits(u64::from(bit), 1);
    }

    /// Writes the lowest `n` bits (`0 <= n <= 64`) of `val` in MSB-first order.
    #[inline]
    pub fn write_bits(&mut self, val: u64, n: u32) {
        if n == 0 {
            return;
        }
        debug_assert!(n <= 64);
        let masked = if n == 64 {
            val
        } else {
            val & ((1u64 << n) - 1)
        };

        let free_bits = 64 - self.bits_in_buf;
        if n < free_bits {
            self.bit_buf |= masked << (free_bits - n);
            self.bits_in_buf += n;
        } else if n == free_bits {
            self.bit_buf |= masked;
            self.out.extend_from_slice(&self.bit_buf.to_be_bytes());
            self.bit_buf = 0;
            self.bits_in_buf = 0;
        } else {
            // Split across the current 64-bit word and the next
            let hi_bits = free_bits;
            let lo_bits = n - hi_bits;
            self.bit_buf |= masked >> lo_bits;
            self.out.extend_from_slice(&self.bit_buf.to_be_bytes());
            self.bit_buf = masked << (64 - lo_bits);
            self.bits_in_buf = lo_bits;
        }
    }

    /// Writes a two's-complement signed integer in `n` bits (`1 <= n <= 64`).
    #[inline]
    pub fn write_signed_bits(&mut self, val: i64, n: u32) {
        self.write_bits(val as u64, n);
    }

    /// Encodes an Unsigned Exp-Golomb value `ue(v)`.
    #[inline]
    pub fn write_ue(&mut self, val: u64) {
        let code = val + 1;
        let lz = 63 - code.leading_zeros();
        self.write_bits(0, lz);
        self.write_bits(code, lz + 1);
    }

    /// Encodes a Signed Exp-Golomb value `se(v)`.
    #[inline]
    pub fn write_se(&mut self, val: i64) {
        let code_num = if val <= 0 {
            ((-val) as u64) << 1
        } else {
            ((val as u64) << 1) - 1
        };
        self.write_ue(code_num);
    }

    /// Encodes an H.264 Truncated Exp-Golomb value `te(v)` with syntax upper bound `max`.
    #[inline]
    pub fn write_te(&mut self, val: u64, max: u64) {
        debug_assert!(val <= max);
        if max == 0 {
            return;
        }
        if max == 1 {
            self.write_bit(val == 0);
        } else {
            self.write_ue(val);
        }
    }

    /// Encodes a $k$-th order Unsigned Exp-Golomb value `ue(k)`.
    #[inline]
    pub fn write_ue_k(&mut self, val: u64, k: u32) {
        self.write_ue(val >> k);
        if k > 0 {
            self.write_bits(val & ((1u64 << k) - 1), k);
        }
    }

    /// Encodes a $k$-th order Signed Exp-Golomb value `se(k)`.
    #[inline]
    pub fn write_se_k(&mut self, val: i64, k: u32) {
        let code_num = if val <= 0 {
            ((-val) as u64) << 1
        } else {
            ((val as u64) << 1) - 1
        };
        self.write_ue_k(code_num, k);
    }

    /// Encodes an AV1 non-power-of-two uniform unsigned integer in `0..max` (`ns(n)`).
    #[inline]
    pub fn write_uniform(&mut self, val: u32, max: u32) {
        debug_assert!(max == 0 || val < max);
        if max <= 1 {
            return;
        }
        let l = 32 - max.leading_zeros();
        let m = (1u32 << l) - max;
        if val < m {
            self.write_bits(val as u64, l - 1);
        } else {
            self.write_bits((val + m) as u64, l);
        }
    }

    /// Encodes an unsigned Golomb-Rice value with parameter `k`.
    #[inline]
    pub fn write_rice_unsigned(&mut self, val: u64, k: u32) {
        let mut q = val >> k;
        while q >= 32 {
            self.write_bits(0, 32);
            q -= 32;
        }
        if q > 0 {
            self.write_bits(0, q as u32);
        }
        self.write_bit(true);
        if k > 0 {
            self.write_bits(val & ((1u64 << k) - 1), k);
        }
    }

    /// Encodes a signed residual using zig-zag folding + Golomb-Rice parameter `k`.
    #[inline]
    pub fn write_rice_signed(&mut self, val: i64, k: u32) {
        let folded = ((val << 1) ^ (val >> 63)) as u64;
        self.write_rice_unsigned(folded, k);
    }

    /// Writes each of `vals` as a zig-zag folded Golomb-Rice code with parameter `k` (`k <= 30`),
    /// as FLAC residual partitions store them.
    ///
    /// On x86_64 the AVX2 dispatch level runs the same code compiled with BMI2 (which every AVX2
    /// CPU has), whose flag-free variable shifts shorten the per-code work.
    pub fn write_rice_signed_i32(&mut self, vals: &[i32], k: u32) {
        debug_assert!(k <= 30);
        #[cfg(target_arch = "x86_64")]
        if crate::dispatch::avx2() && std::is_x86_feature_detected!("bmi2") {
            // Safety: the CPU supports BMI2 (checked just above).
            return unsafe { self.write_rice_signed_i32_bmi2(vals, k) };
        }
        self.write_rice_signed_i32_impl(vals, k)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "bmi1,bmi2")]
    unsafe fn write_rice_signed_i32_bmi2(&mut self, vals: &[i32], k: u32) {
        self.write_rice_signed_i32_impl(vals, k)
    }

    /// Codes are joined four at a time into one field of up to 64 bits, appended with a single
    /// (mostly unpredictable) buffer-full test instead of one per code; groups too long for one
    /// field, and codes of 63 or more bits, go one at a time.
    #[inline(always)]
    fn write_rice_signed_i32_impl(&mut self, vals: &[i32], k: u32) {
        let mask = (1u64 << k) - 1;
        let stop = 1u64 << k;
        let mut buf = self.bit_buf;
        let mut free = 64 - self.bits_in_buf;
        let mut pos = self.out.len();
        let mut cap = self.out.capacity();
        let mut base = self.out.as_mut_ptr();
        // `q` zeros, the stop bit and the `k` low bits form one field of `q + 1 + k` bits.
        let code_len = |v: i32| {
            let u = ((v << 1) ^ (v >> 31)) as u32 as u64;
            (stop | (u & mask), (u >> k) + k as u64 + 1)
        };
        // Appends the low `len <= 64` bits of `field`.
        macro_rules! append {
            ($field:expr, $len:expr) => {{
                let (field, len): (u64, u32) = ($field, $len);
                if len < free {
                    free -= len;
                    buf |= field << free;
                } else {
                    if pos + 8 > cap {
                        // Safety: bytes up to `pos` were written below.
                        unsafe { self.out.set_len(pos) };
                        self.out.reserve(vals.len() + 64);
                        (cap, base) = (self.out.capacity(), self.out.as_mut_ptr());
                    }
                    let rest = len - free;
                    // Safety: `pos + 8 <= cap`.
                    unsafe {
                        (base.add(pos) as *mut [u8; 8])
                            .write_unaligned((buf | (field >> rest)).to_be_bytes())
                    };
                    pos += 8;
                    free = 64 - rest;
                    // `free` may be 64 here, where a single shift would not clear the word.
                    buf = (field << 1) << (free - 1);
                }
            }};
        }
        let (quads, tail) = vals.as_chunks::<4>();
        for quad in quads.iter().map(|q| &q[..]).chain(std::iter::once(tail)) {
            if quad.len() == 4 {
                let [(c0, l0), (c1, l1), (c2, l2), (c3, l3)] =
                    [0, 1, 2, 3].map(|i| code_len(quad[i]));
                let total = l0 + l1 + l2 + l3;
                if total <= 64 {
                    let field = (((((c0 << l1) | c1) << l2) | c2) << l3) | c3;
                    append!(field, total as u32);
                    continue;
                }
            }
            for &v in quad {
                let (code, len) = code_len(v);
                if len < 64 {
                    append!(code, len as u32);
                } else {
                    // Safety: bytes up to `pos` were written above.
                    unsafe { self.out.set_len(pos) };
                    (self.bit_buf, self.bits_in_buf) = (buf, 64 - free);
                    let mut zeros = len - 1 - k as u64;
                    while zeros >= 32 {
                        self.write_bits(0, 32);
                        zeros -= 32;
                    }
                    self.write_bits(0, zeros as u32);
                    self.write_bits(code, k + 1);
                    (buf, free) = (self.bit_buf, 64 - self.bits_in_buf);
                    (pos, cap, base) = (self.out.len(), self.out.capacity(), self.out.as_mut_ptr());
                }
            }
        }
        // Safety: bytes up to `pos` were written above.
        unsafe { self.out.set_len(pos) };
        (self.bit_buf, self.bits_in_buf) = (buf, 64 - free);
    }

    /// Pads with zero bits to the next byte boundary.
    #[inline]
    pub fn align_to_byte(&mut self) {
        let rem = self.bits_in_buf & 7;
        if rem != 0 {
            self.write_bits(0, 8 - rem);
        }
    }

    /// Writes H.264/HEVC RBSP trailing bits (`1` stop bit followed by `0` alignment bits).
    #[inline]
    pub fn write_rbsp_trailing_bits(&mut self) {
        self.write_bit(true);
        self.align_to_byte();
    }

    /// Flushes any partial bytes (zero-padded on the right) and returns the finished byte buffer.
    pub fn finish(mut self) -> Vec<u8> {
        self.align_to_byte();
        while self.bits_in_buf >= 8 {
            let b = (self.bit_buf >> 56) as u8;
            self.out.push(b);
            self.bit_buf <<= 8;
            self.bits_in_buf -= 8;
        }
        self.out
    }
}

/// Removes H.264 / HEVC RBSP emulation prevention bytes (`0x00 0x00 0x03` -> `0x00 0x00`).
///
/// Uses an `aarch64` NEON 16-byte adjacent-zero pair detector (`vceqq_u8` + `vextq_u8` + `vandq_u8`,
/// modeled after FFmpeg `vc1dsp_neon.S` & `h2645_parse.c`) that stays in the 16-byte SIMD fast-path
/// across isolated single `0x00` bytes and immediately re-enters SIMD after handling any `0x00 0x00 0x03` triad.
pub fn unescape_rbsp(nal_payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(nal_payload.len());
    unescape_rbsp_into(nal_payload, &mut out);
    out
}

pub fn unescape_rbsp_into(src: &[u8], dst: &mut Vec<u8>) {
    let len = src.len();
    if len < 3 {
        dst.extend_from_slice(src);
        return;
    }

    let mut i = 0;
    let mut zeros = 0usize;

    #[cfg(target_arch = "aarch64")]
    if crate::dispatch::neon() {
        use std::arch::aarch64::*;
        unsafe {
            let v_zero = vdupq_n_u8(0);
            while i + 16 <= len {
                // Fast NEON path when no two consecutive zeros exist across or inside the 16-byte block
                if zeros == 0 || (zeros == 1 && src[i] != 0) {
                    let chunk = vld1q_u8(src.as_ptr().add(i));
                    let eq_zero = vceqq_u8(chunk, v_zero);
                    let shifted = vextq_u8(eq_zero, v_zero, 1);
                    let adj_zeros = vandq_u8(eq_zero, shifted);
                    if vmaxvq_u8(adj_zeros) == 0 {
                        dst.extend_from_slice(&src[i..i + 16]);
                        zeros = usize::from(src[i + 15] == 0);
                        i += 16;
                        continue;
                    }
                }
                // Process this 16-byte block in scalar and re-enter NEON on the next 16-byte block
                let block_end = i + 16;
                while i < block_end {
                    let b = src[i];
                    if zeros >= 2 && b == 0x03 {
                        zeros = 0;
                        i += 1;
                        continue;
                    }
                    dst.push(b);
                    if b == 0x00 {
                        zeros += 1;
                    } else {
                        zeros = 0;
                    }
                    i += 1;
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if crate::dispatch::avx2() {
            #[target_feature(enable = "avx2")]
            unsafe fn unescape_rbsp_avx2(
                src: &[u8],
                dst: &mut Vec<u8>,
                i: &mut usize,
                zeros: &mut usize,
            ) {
                use std::arch::x86_64::*;
                let len = src.len();
                let v_zero = _mm256_setzero_si256();
                while *i + 32 <= len {
                    if *zeros == 0 || (*zeros == 1 && src[*i] != 0) {
                        let chunk =
                            unsafe { _mm256_loadu_si256(src.as_ptr().add(*i) as *const __m256i) };
                        let eq_zero = _mm256_cmpeq_epi8(chunk, v_zero);
                        let zmask = _mm256_movemask_epi8(eq_zero) as u32;
                        if (zmask & (zmask >> 1)) == 0 {
                            dst.extend_from_slice(&src[*i..*i + 32]);
                            *zeros = usize::from(src[*i + 31] == 0);
                            *i += 32;
                            continue;
                        }
                    }
                    let block_end = *i + 32;
                    while *i < block_end {
                        let b = src[*i];
                        if *zeros >= 2 && b == 0x03 {
                            *zeros = 0;
                            *i += 1;
                            continue;
                        }
                        dst.push(b);
                        if b == 0x00 {
                            *zeros += 1;
                        } else {
                            *zeros = 0;
                        }
                        *i += 1;
                    }
                }
            }
            unsafe {
                unescape_rbsp_avx2(src, dst, &mut i, &mut zeros);
            }
        }
    }

    while i < len {
        let b = src[i];
        if zeros >= 2 && b == 0x03 {
            // Skip emulation prevention byte 0x03
            zeros = 0;
            i += 1;
            continue;
        }
        dst.push(b);
        if b == 0x00 {
            zeros += 1;
        } else {
            zeros = 0;
        }
        i += 1;
    }
}

/// Reference scalar RBSP unescaper for differential verification against SIMD.
pub fn unescape_rbsp_scalar(src: &[u8]) -> Vec<u8> {
    let mut dst = Vec::with_capacity(src.len());
    let mut zeros = 0usize;
    for &b in src {
        if zeros >= 2 && b == 0x03 {
            zeros = 0;
            continue;
        }
        dst.push(b);
        if b == 0x00 {
            zeros += 1;
        } else {
            zeros = 0;
        }
    }
    dst
}

/// Inserts H.264 / HEVC RBSP emulation prevention bytes (`0x00 0x00 [0x00..=0x03]` -> `0x00 0x00 0x03 ...`).
pub fn escape_rbsp(rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len() + 16);
    let mut zeros = 0usize;
    for &b in rbsp {
        if zeros == 2 && b <= 0x03 {
            out.push(0x03);
            zeros = 0;
        }
        out.push(b);
        if b == 0x00 {
            zeros += 1;
        } else {
            zeros = 0;
        }
    }
    out
}

/// Decodes an AV1 unsigned LEB128 variable-length integer, returning `(value, bytes_consumed)`.
#[inline]
pub fn read_leb128(data: &[u8]) -> Result<(u64, usize)> {
    let mut val = 0u64;
    for (i, &b) in data.iter().take(8).enumerate() {
        val |= ((b & 0x7F) as u64) << (i * 7);
        if (b & 0x80) == 0 {
            return Ok((val, i + 1));
        }
    }
    Err(RagdollError::InvalidSyntax(
        "truncated or overflowing LEB128 integer",
    ))
}

/// Encodes an unsigned integer as AV1 LEB128 into `dst`.
#[inline]
pub fn write_leb128(mut val: u64, dst: &mut Vec<u8>) {
    loop {
        let mut byte = (val & 0x7F) as u8;
        val >>= 7;
        if val != 0 {
            byte |= 0x80;
        }
        dst.push(byte);
        if val == 0 {
            break;
        }
    }
}

/// Decodes a Matroska / WebM EBML variable-length integer (VINT) using hardware `leading_zeros()`.
/// Returns `(raw_element_id_with_marker, stripped_data_value, byte_width)`.
#[inline]
pub fn read_ebml_vint(data: &[u8]) -> Result<(u64, u64, usize)> {
    let Some(&first) = data.first() else {
        return Err(RagdollError::BitstreamEndOfData);
    };
    if first == 0 {
        return Err(RagdollError::InvalidSyntax(
            "invalid EBML VINT leading 0x00",
        ));
    }
    let width = (first.leading_zeros() as usize) + 1;
    if data.len() < width {
        return Err(RagdollError::BitstreamEndOfData);
    }
    let mut raw = first as u64;
    let mask = (1u64 << (8 - width)) - 1;
    let mut stripped = raw & mask;
    for &b in &data[1..width] {
        raw = (raw << 8) | (b as u64);
        stripped = (stripped << 8) | (b as u64);
    }
    Ok((raw, stripped, width))
}

/// Encodes an unsigned data length as an EBML VINT into `dst`.
#[inline]
pub fn write_ebml_vint(val: u64, dst: &mut Vec<u8>) {
    let width = if val < (1 << 7) - 1 {
        1
    } else if val < (1 << 14) - 1 {
        2
    } else if val < (1 << 21) - 1 {
        3
    } else if val < (1 << 28) - 1 {
        4
    } else if val < (1 << 35) - 1 {
        5
    } else if val < (1 << 42) - 1 {
        6
    } else if val < (1 << 49) - 1 {
        7
    } else {
        8
    };
    let marker = 1u64 << (7 * width);
    let encoded = val | marker;
    for i in (0..width).rev() {
        dst.push((encoded >> (i * 8)) as u8);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The block Rice writer (both dispatch levels) matches one `write_rice_signed` per value,
    /// from every starting bit offset, across long codes and extremal residuals.
    #[test]
    fn rice_block_writer_matches_per_value_writer() {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for k in [0u32, 1, 3, 7, 14, 15, 30] {
            // Values of up to `k + 12` bits keep the unary parts short enough to test quickly;
            // 1000 and -777 still take the long-code path at small `k`.
            for bits in [1u32, 4, 9, 17, 32].map(|b: u32| b.min(k + 12)) {
                let mut vals: Vec<i32> = (0..300).map(|_| (next() as i32) >> (32 - bits)).collect();
                vals.extend([0, -1, 1000, -777]);
                if k == 30 {
                    vals.extend([i32::MIN, i32::MAX, 1 << 20, -(1 << 25)]);
                }
                for lead in [0u32, 1, 7, 31, 32, 33, 63] {
                    let mut a = BitWriter::new();
                    let mut b = BitWriter::new();
                    for _ in 0..lead {
                        a.write_bit(true);
                        b.write_bit(true);
                    }
                    a.write_rice_signed_i32(&vals[..150], k);
                    a.write_rice_signed_i32(&vals[150..], k);
                    for &v in &vals {
                        b.write_rice_signed(v as i64, k);
                    }
                    a.write_bits(0x5, 3);
                    b.write_bits(0x5, 3);
                    assert_eq!(
                        a.bits_written(),
                        b.bits_written(),
                        "k={k} bits={bits} lead={lead}"
                    );
                    assert_eq!(a.finish(), b.finish(), "k={k} bits={bits} lead={lead}");
                }
            }
        }
    }
}
