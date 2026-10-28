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
