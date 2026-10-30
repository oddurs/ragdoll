//! CABAC syntax elements not covered by [`crate::cabac::CabacDecoder`]'s helpers (ITU-T H.264 9.3).

use crate::cabac::{CabacContext, CabacDecoder};

/// `mb_type` in a B slice (Table 9-37, ctxIdxOffset 27 prefix, 32 intra suffix).
/// Returns `0..=22` for inter types and `23 + I mb_type` for intra.
pub fn b_mb_type(cd: &mut CabacDecoder<'_>, ctxs: &mut [CabacContext], inc: usize) -> u32 {
    if cd.decode_decision(&mut ctxs[27 + inc]) == 0 {
        return 0;
    }
    if cd.decode_decision(&mut ctxs[27 + 3]) == 0 {
        return 1 + cd.decode_decision(&mut ctxs[27 + 5]) as u32;
    }
    let mut bits = (cd.decode_decision(&mut ctxs[27 + 4]) as u32) << 3;
    bits |= (cd.decode_decision(&mut ctxs[27 + 5]) as u32) << 2;
    bits |= (cd.decode_decision(&mut ctxs[27 + 5]) as u32) << 1;
    bits |= cd.decode_decision(&mut ctxs[27 + 5]) as u32;
    if bits < 8 {
        return bits + 3;
    }
    match bits {
        13 => 23 + cd.decode_intra_mb_type(ctxs, 32, 0, 0),
        14 => 11,
        15 => 22,
        _ => ((bits << 1) | cd.decode_decision(&mut ctxs[27 + 5]) as u32) - 4,
    }
}

/// `transform_size_8x8_flag` (ctxIdx 399..401).
pub fn transform_8x8(cd: &mut CabacDecoder<'_>, ctxs: &mut [CabacContext], inc: usize) -> bool {
    cd.decode_decision(&mut ctxs[399 + inc]) != 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::{CabacEncoder, init_cabac_contexts};

    /// Every B `mb_type` but I_PCM, as the encoder writes it, reads back (Table 9-37).
    #[test]
    fn b_mb_types_from_the_encoder_read_back() {
        let conds = |t: u32| (t.is_multiple_of(3) as u8, (t % 2) as u8);
        let mut e = CabacEncoder::new();
        let mut ctxs = init_cabac_contexts(26, false, 0);
        for t in 0..48 {
            let (a, b) = conds(t);
            e.encode_b_mb_type(&mut ctxs, a, b, t);
        }
        e.encode_terminate(1);
        let bytes = e.finish();
        let mut d = CabacDecoder::new(&bytes).unwrap();
        let mut ctxs = init_cabac_contexts(26, false, 0);
        for t in 0..48 {
            let (a, b) = conds(t);
            assert_eq!(b_mb_type(&mut d, &mut ctxs, (a + b) as usize), t);
        }
    }
}
