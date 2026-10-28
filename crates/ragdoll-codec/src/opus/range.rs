//! Range coder (RFC 6716 section 4.1, section 5.1).
//!
//! One context type serves both directions, as in libopus `entcode.h`: the decoder reads from the
//! front of the buffer and raw bits from the back; the encoder writes the same way.

const SYM_BITS: u32 = 8;
const CODE_BITS: u32 = 32;
const SYM_MAX: u32 = (1 << SYM_BITS) - 1;
const CODE_SHIFT: u32 = CODE_BITS - SYM_BITS - 1;
const CODE_TOP: u32 = 1 << (CODE_BITS - 1);
const CODE_BOT: u32 = CODE_TOP >> SYM_BITS;
const CODE_EXTRA: u32 = (CODE_BITS - 2) % SYM_BITS + 1;
const WINDOW_SIZE: u32 = 32;
const UINT_BITS: u32 = 8;
/// Fractional bits of [`Ec::tell_frac`].
pub const BITRES: i32 = 3;

/// Number of significant bits of `v` (`EC_ILOG`), 0 for 0.
#[inline]
pub fn ilog(v: u32) -> i32 {
    (32 - v.leading_zeros()) as i32
}

/// Range coder state (`ec_ctx`).
#[derive(Clone, Debug)]
pub struct Ec {
    pub buf: Vec<u8>,
    pub storage: u32,
    end_offs: u32,
    end_window: u32,
    nend_bits: i32,
    pub nbits_total: i32,
    pub offs: u32,
    pub rng: u32,
    val: u32,
    ext: u32,
    rem: i32,
    pub error: i32,
}

impl Ec {
    /// Starts decoding `data` (`ec_dec_init`).
    pub fn decoder(data: &[u8]) -> Self {
        let mut d = Self {
            buf: data.to_vec(),
            storage: data.len() as u32,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            nbits_total: (CODE_BITS + 1 - ((CODE_BITS - CODE_EXTRA) / SYM_BITS) * SYM_BITS) as i32,
            offs: 0,
            rng: 1 << CODE_EXTRA,
            val: 0,
            ext: 0,
            rem: 0,
            error: 0,
        };
        d.rem = d.read_byte();
        d.val = d.rng - 1 - ((d.rem as u32) >> (SYM_BITS - CODE_EXTRA));
        d.dec_normalize();
        d
    }

    /// Starts encoding into a buffer of `size` bytes (`ec_enc_init`).
    pub fn encoder(size: usize) -> Self {
        Self {
            buf: vec![0; size],
            storage: size as u32,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            nbits_total: (CODE_BITS + 1) as i32,
            offs: 0,
            rng: CODE_TOP,
            val: 0,
            ext: 0,
            rem: -1,
            error: 0,
        }
    }

    /// Bits used so far, rounded up (`ec_tell`).
    #[inline]
    pub fn tell(&self) -> i32 {
        self.nbits_total - ilog(self.rng)
    }

    /// Bits used so far in 1/8 bit units (`ec_tell_frac`).
    pub fn tell_frac(&self) -> u32 {
        const CORRECTION: [u32; 8] = [35733, 38967, 42495, 46340, 50535, 55109, 60097, 65535];
        let nbits = (self.nbits_total as u32) << BITRES;
        let mut l = ilog(self.rng);
        let r = self.rng >> (l - 16);
        let mut b = (r >> 12) - 8;
        b += (r > CORRECTION[b as usize]) as u32;
        l = (l << 3) + b as i32;
        nbits.wrapping_sub(l as u32)
    }

    // ----- decoding --------------------------------------------------------------------------

    fn read_byte(&mut self) -> i32 {
        if self.offs < self.storage {
            let b = self.buf[self.offs as usize];
            self.offs += 1;
            b as i32
        } else {
            0
        }
    }

    fn read_byte_from_end(&mut self) -> u32 {
        if self.end_offs < self.storage {
            self.end_offs += 1;
            self.buf[(self.storage - self.end_offs) as usize] as u32
        } else {
            0
        }
    }

    fn dec_normalize(&mut self) {
        while self.rng <= CODE_BOT {
            self.nbits_total += SYM_BITS as i32;
            self.rng <<= SYM_BITS;
            let mut sym = self.rem;
            self.rem = self.read_byte();
            sym = ((sym << SYM_BITS) | self.rem) >> (SYM_BITS - CODE_EXTRA);
            self.val =
                ((self.val << SYM_BITS).wrapping_add(SYM_MAX & !(sym as u32))) & (CODE_TOP - 1);
        }
    }

    /// `ec_decode`.
    pub fn decode(&mut self, ft: u32) -> u32 {
        self.ext = self.rng / ft;
        let s = self.val / self.ext;
        ft - (s + 1).min(ft)
    }

    /// `ec_decode_bin`.
    pub fn decode_bin(&mut self, bits: u32) -> u32 {
        self.ext = self.rng >> bits;
        let s = self.val / self.ext;
        (1 << bits) - (s + 1).min(1 << bits)
    }

    /// `ec_dec_update`.
    pub fn dec_update(&mut self, fl: u32, fh: u32, ft: u32) {
        let s = self.ext.wrapping_mul(ft - fh);
        self.val = self.val.wrapping_sub(s);
        self.rng = if fl > 0 {
            self.ext.wrapping_mul(fh - fl)
        } else {
            self.rng.wrapping_sub(s)
        };
        self.dec_normalize();
    }

    /// `ec_dec_bit_logp`.
    pub fn dec_bit_logp(&mut self, logp: u32) -> bool {
        let r = self.rng;
        let d = self.val;
        let s = r >> logp;
        let ret = d < s;
        if !ret {
            self.val = d - s;
        }
        self.rng = if ret { s } else { r - s };
        self.dec_normalize();
        ret
    }

    /// `ec_dec_icdf`.
    pub fn dec_icdf(&mut self, icdf: &[u8], ftb: u32) -> i32 {
        let mut s = self.rng;
        let d = self.val;
        let r = s >> ftb;
        let mut ret = -1i32;
        let mut t;
        loop {
            t = s;
            ret += 1;
            s = r.wrapping_mul(icdf[ret as usize] as u32);
            if d >= s {
                break;
            }
        }
        self.val = d - s;
        self.rng = t - s;
        self.dec_normalize();
        ret
    }

    /// `ec_dec_uint`: a value uniformly distributed in `0..ft`.
    pub fn dec_uint(&mut self, ft: u32) -> u32 {
        let ft1 = ft - 1;
        let mut ftb = ilog(ft1);
        if ftb > UINT_BITS as i32 {
            ftb -= UINT_BITS as i32;
            let f = (ft1 >> ftb) + 1;
            let s = self.decode(f);
            self.dec_update(s, s + 1, f);
            let t = (s << ftb) | self.dec_bits(ftb as u32);
            if t <= ft1 {
                return t;
            }
            self.error = 1;
            ft1
        } else {
            let s = self.decode(ft);
            self.dec_update(s, s + 1, ft);
            s
        }
    }

    /// `ec_dec_bits`: raw bits from the end of the buffer.
    pub fn dec_bits(&mut self, bits: u32) -> u32 {
        let mut window = self.end_window;
        let mut available = self.nend_bits;
        if (available as u32) < bits {
            loop {
                window |= self.read_byte_from_end() << available;
                available += SYM_BITS as i32;
                if available > (WINDOW_SIZE - SYM_BITS) as i32 {
                    break;
                }
            }
        }
        let ret = window & ((1u64 << bits) - 1) as u32;
        window = if bits >= 32 { 0 } else { window >> bits };
        available -= bits as i32;
        self.end_window = window;
        self.nend_bits = available;
        self.nbits_total += bits as i32;
        ret
    }

    /// `ec_laplace_decode`.
    pub fn laplace_decode(&mut self, fs: u32, decay: i32) -> i32 {
        const MINP: u32 = 1;
        const NMIN: u32 = 16;
        let mut val = 0i32;
        let fm = self.decode_bin(15);
        let mut fl = 0u32;
        let mut fs = fs;
        if fm >= fs {
            val += 1;
            fl = fs;
            fs = (((32768 - MINP * (2 * NMIN) - fs) as i32 * (16384 - decay)) >> 15) as u32 + MINP;
            while fs > MINP && fm >= fl + 2 * fs {
                fs *= 2;
                fl += fs;
                fs = (((fs - 2 * MINP) as i32 * decay) >> 15) as u32;
                fs += MINP;
                val += 1;
            }
            if fs <= MINP {
                let di = (fm - fl) >> 1;
                val += di as i32;
                fl += 2 * di * MINP;
            }
            if fm < fl + fs {
                val = -val;
            } else {
                fl += fs;
            }
        }
        self.dec_update(fl, (fl + fs).min(32768), 32768);
        val
    }

    // ----- encoding --------------------------------------------------------------------------

    fn write_byte(&mut self, v: u32) -> i32 {
        if self.offs + self.end_offs >= self.storage {
            return -1;
        }
        self.buf[self.offs as usize] = v as u8;
        self.offs += 1;
        0
    }

    fn write_byte_at_end(&mut self, v: u32) -> i32 {
        if self.offs + self.end_offs >= self.storage {
            return -1;
        }
        self.end_offs += 1;
        self.buf[(self.storage - self.end_offs) as usize] = v as u8;
        0
    }

    fn carry_out(&mut self, c: i32) {
        if c as u32 != SYM_MAX {
            let carry = c >> SYM_BITS;
            if self.rem >= 0 {
                self.error |= self.write_byte((self.rem + carry) as u32);
            }
            if self.ext > 0 {
                let sym = (SYM_MAX.wrapping_add(carry as u32)) & SYM_MAX;
                loop {
                    self.error |= self.write_byte(sym);
                    self.ext -= 1;
                    if self.ext == 0 {
                        break;
                    }
                }
            }
            self.rem = c & SYM_MAX as i32;
        } else {
            self.ext += 1;
        }
    }

    fn enc_normalize(&mut self) {
        while self.rng <= CODE_BOT {
            self.carry_out((self.val >> CODE_SHIFT) as i32);
            self.val = (self.val << SYM_BITS) & (CODE_TOP - 1);
            self.rng <<= SYM_BITS;
            self.nbits_total += SYM_BITS as i32;
        }
    }

    /// `ec_encode`.
    pub fn encode(&mut self, fl: u32, fh: u32, ft: u32) {
        let r = self.rng / ft;
        if fl > 0 {
            self.val = self
                .val
                .wrapping_add(self.rng.wrapping_sub(r.wrapping_mul(ft - fl)));
            self.rng = r.wrapping_mul(fh - fl);
        } else {
            self.rng = self.rng.wrapping_sub(r.wrapping_mul(ft - fh));
        }
        self.enc_normalize();
    }

    /// `ec_encode_bin`.
    pub fn encode_bin(&mut self, fl: u32, fh: u32, bits: u32) {
        let r = self.rng >> bits;
        if fl > 0 {
            self.val = self
                .val
                .wrapping_add(self.rng.wrapping_sub(r.wrapping_mul((1 << bits) - fl)));
            self.rng = r.wrapping_mul(fh - fl);
        } else {
            self.rng = self.rng.wrapping_sub(r.wrapping_mul((1 << bits) - fh));
        }
        self.enc_normalize();
    }

    /// `ec_enc_bit_logp`.
    pub fn enc_bit_logp(&mut self, val: bool, logp: u32) {
        let mut r = self.rng;
        let l = self.val;
        let s = r >> logp;
        r -= s;
        if val {
            self.val = l.wrapping_add(r);
        }
        self.rng = if val { s } else { r };
        self.enc_normalize();
    }

    /// `ec_enc_icdf`.
    pub fn enc_icdf(&mut self, s: usize, icdf: &[u8], ftb: u32) {
        let r = self.rng >> ftb;
        if s > 0 {
            self.val = self
                .val
                .wrapping_add(self.rng.wrapping_sub(r.wrapping_mul(icdf[s - 1] as u32)));
            self.rng = r.wrapping_mul((icdf[s - 1] - icdf[s]) as u32);
        } else {
            self.rng = self.rng.wrapping_sub(r.wrapping_mul(icdf[s] as u32));
        }
        self.enc_normalize();
    }

    /// `ec_enc_uint`.
    pub fn enc_uint(&mut self, fl: u32, ft: u32) {
        let ft1 = ft - 1;
        let mut ftb = ilog(ft1);
        if ftb > UINT_BITS as i32 {
            ftb -= UINT_BITS as i32;
            let f = (ft1 >> ftb) + 1;
            let v = fl >> ftb;
            self.encode(v, v + 1, f);
            self.enc_bits(fl & ((1u32 << ftb) - 1), ftb as u32);
        } else {
            self.encode(fl, fl + 1, ft);
        }
    }

    /// `ec_enc_bits`: raw bits at the end of the buffer.
    pub fn enc_bits(&mut self, fl: u32, bits: u32) {
        let mut window = self.end_window;
        let mut used = self.nend_bits;
        if used + bits as i32 > WINDOW_SIZE as i32 {
            loop {
                self.error |= self.write_byte_at_end(window & SYM_MAX);
                window >>= SYM_BITS;
                used -= SYM_BITS as i32;
                if used < SYM_BITS as i32 {
                    break;
                }
            }
        }
        window |= fl << used;
        used += bits as i32;
        self.end_window = window;
        self.nend_bits = used;
        self.nbits_total += bits as i32;
    }

    /// `ec_laplace_encode`; `value` may be clamped to what the model can code.
    pub fn laplace_encode(&mut self, value: &mut i32, fs: u32, decay: i32) {
        const MINP: u32 = 1;
        const NMIN: u32 = 16;
        let mut fl = 0u32;
        let mut fs = fs;
        let mut val = *value;
        if val != 0 {
            let s = -((val < 0) as i32);
            val = (val + s) ^ s;
            fl = fs;
            fs = (((32768 - MINP * (2 * NMIN) - fs) as i32 * (16384 - decay)) >> 15) as u32;
            let mut i = 1;
            while fs > 0 && i < val {
                fs *= 2;
                fl += fs + 2 * MINP;
                fs = ((fs as i32 * decay) >> 15) as u32;
                i += 1;
            }
            if fs == 0 {
                let mut ndi_max = (32768 - fl + MINP - 1) as i32;
                ndi_max = (ndi_max - s) >> 1;
                let di = (val - i).min(ndi_max - 1);
                fl = (fl as i32 + (2 * di + 1 + s) * MINP as i32) as u32;
                fs = MINP.min(32768 - fl);
                *value = (i + di + s) ^ s;
            } else {
                fs += MINP;
                fl += fs & !(s as u32);
            }
        }
        self.encode_bin(fl, fl + fs, 15);
    }

    /// `ec_enc_shrink`.
    pub fn enc_shrink(&mut self, size: u32) {
        let src = (self.storage - self.end_offs) as usize;
        let dst = (size - self.end_offs) as usize;
        let n = self.end_offs as usize;
        self.buf.copy_within(src..src + n, dst);
        self.storage = size;
    }

    /// `ec_enc_done`: flushes the coder; the packet is `buf[..storage]`.
    pub fn enc_done(&mut self) {
        let mut l = CODE_BITS as i32 - ilog(self.rng);
        let mut msk = (CODE_TOP - 1) >> l;
        let mut end = self.val.wrapping_add(msk) & !msk;
        if (end | msk) >= self.val.wrapping_add(self.rng) {
            l += 1;
            msk >>= 1;
            end = self.val.wrapping_add(msk) & !msk;
        }
        while l > 0 {
            self.carry_out((end >> CODE_SHIFT) as i32);
            end = (end << SYM_BITS) & (CODE_TOP - 1);
            l -= SYM_BITS as i32;
        }
        if self.rem >= 0 || self.ext > 0 {
            self.carry_out(0);
        }
        let mut window = self.end_window;
        let mut used = self.nend_bits;
        while used >= SYM_BITS as i32 {
            self.error |= self.write_byte_at_end(window & SYM_MAX);
            window >>= SYM_BITS;
            used -= SYM_BITS as i32;
        }
        if self.error == 0 {
            let a = self.offs as usize;
            let b = (self.storage - self.end_offs) as usize;
            self.buf[a..b].fill(0);
            if used > 0 {
                if self.end_offs >= self.storage {
                    self.error = -1;
                } else {
                    l = -l;
                    if self.offs + self.end_offs >= self.storage && l < used {
                        window &= (1 << l) - 1;
                        self.error = -1;
                    }
                    let i = (self.storage - self.end_offs - 1) as usize;
                    self.buf[i] |= window as u8;
                }
            }
        }
    }
}
