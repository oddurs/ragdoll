/// Finds the first H.264 / HEVC Annex-B start code (`0x00 0x00 0x01` or `0x00 0x00 0x00 0x01`)
/// in `data`, returning `Some((offset_of_start_code, start_code_len))` where `start_code_len` is 3 or 4.
#[inline]
pub fn find_annex_b_start_code(data: &[u8]) -> Option<(usize, usize)> {
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        // Safety: NEON is baseline on all aarch64 CPUs (including Apple Silicon M-series).
        return unsafe { find_annex_b_start_code_neon(data) };
    }
    #[cfg(target_arch = "x86_64")]
    if ragdoll_core::dispatch::avx2() {
        // Safety: avx2() is true only when the CPU supports AVX2.
        return unsafe { find_annex_b_start_code_avx2(data) };
    }
    find_annex_b_start_code_scalar(data)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn find_annex_b_start_code_avx2(data: &[u8]) -> Option<(usize, usize)> {
    use std::arch::x86_64::*;
    let len = data.len();
    if len < 3 {
        return None;
    }
    let mut i = 0usize;
    let v_zero = _mm256_setzero_si256();

    while i + 2 < len {
        if i == 0 || unsafe { *data.get_unchecked(i - 1) } != 0 {
            while i + 32 <= len {
                let chunk = unsafe { _mm256_loadu_si256(data.as_ptr().add(i) as *const __m256i) };
                let eq_zero = _mm256_cmpeq_epi8(chunk, v_zero);
                let zmask = _mm256_movemask_epi8(eq_zero) as u32;
                if zmask == 0 {
                    i += 32;
                    continue;
                }
                if (zmask & (zmask >> 1)) == 0 && unsafe { *data.get_unchecked(i + 31) } != 0 {
                    i += 32;
                } else {
                    break;
                }
            }
            if i + 2 >= len {
                break;
            }
        }

        if data[i + 1] != 0 {
            i += 2;
            continue;
        }
        if data[i] == 0 {
            if data[i + 2] == 1 {
                if i > 0 && data[i - 1] == 0 {
                    return Some((i - 1, 4));
                }
                return Some((i, 3));
            }
            if data[i + 2] == 0 && i + 3 < len && data[i + 3] == 1 {
                return Some((i, 4));
            }
        }
        i += 1;
    }
    None
}

/// Scalar reference implementation of Annex-B start-code search (`00 00 01` or `00 00 00 01`).
pub fn find_annex_b_start_code_scalar(data: &[u8]) -> Option<(usize, usize)> {
    let len = data.len();
    if len < 3 {
        return None;
    }
    let mut i = 0;
    while i + 2 < len {
        if data[i + 1] != 0 {
            i += 2;
            continue;
        }
        if data[i] == 0 {
            if data[i + 2] == 1 {
                if i > 0 && data[i - 1] == 0 {
                    return Some((i - 1, 4));
                }
                return Some((i, 3));
            }
            if data[i + 2] == 0 && i + 3 < len && data[i + 3] == 1 {
                return Some((i, 4));
            }
        }
        i += 1;
    }
    None
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn find_annex_b_start_code_neon(data: &[u8]) -> Option<(usize, usize)> {
    use std::arch::aarch64::*;
    let len = data.len();
    if len < 3 {
        return None;
    }
    let mut i = 0usize;
    let v_zero = vdupq_n_u8(0);

    while i + 2 < len {
        // Fast NEON 64-byte / 16-byte skip when no trailing 0x00 precedes `i`
        if i == 0 || unsafe { *data.get_unchecked(i - 1) } != 0 {
            while i + 64 <= len {
                let ptr = unsafe { data.as_ptr().add(i) };
                let c0 = unsafe { vld1q_u8(ptr) };
                let c1 = unsafe { vld1q_u8(ptr.add(16)) };
                let c2 = unsafe { vld1q_u8(ptr.add(32)) };
                let c3 = unsafe { vld1q_u8(ptr.add(48)) };
                let z0 = vceqq_u8(c0, v_zero);
                let z1 = vceqq_u8(c1, v_zero);
                let z2 = vceqq_u8(c2, v_zero);
                let z3 = vceqq_u8(c3, v_zero);
                let any01 = vorrq_u8(z0, z1);
                let any23 = vorrq_u8(z2, z3);
                if vmaxvq_u8(vorrq_u8(any01, any23)) == 0 {
                    i += 64;
                } else {
                    break;
                }
            }
            while i + 16 <= len {
                let chunk = unsafe { vld1q_u8(data.as_ptr().add(i)) };
                let eq_zero = vceqq_u8(chunk, v_zero);
                if vmaxvq_u8(eq_zero) == 0 {
                    i += 16;
                    continue;
                }
                // Even if isolated 0x00 bytes exist in the 16-byte chunk, a start code requires
                // at least two consecutive 0x00 bytes (`00 00 01`). Use `vextq_u8` + `vandq_u8`
                // to check for adjacent zero pairs inside the 16-byte vector.
                let shifted = vextq_u8(eq_zero, v_zero, 1);
                let adj_zeros = vandq_u8(eq_zero, shifted);
                if vmaxvq_u8(adj_zeros) == 0 && unsafe { *data.get_unchecked(i + 15) } != 0 {
                    i += 16;
                } else {
                    break;
                }
            }
            if i + 2 >= len {
                break;
            }
        }

        if data[i + 1] != 0 {
            i += 2;
            continue;
        }
        if data[i] == 0 {
            if data[i + 2] == 1 {
                if i > 0 && data[i - 1] == 0 {
                    return Some((i - 1, 4));
                }
                return Some((i, 3));
            }
            if data[i + 2] == 0 && i + 3 < len && data[i + 3] == 1 {
                return Some((i, 4));
            }
        }
        i += 1;
    }
    None
}

/// Zero-copy splitter that yields all NAL unit payloads from an Annex-B byte stream.
pub fn split_annex_b_nals(data: &[u8]) -> Vec<&[u8]> {
    let mut nals = Vec::new();
    let Some((first_sc, first_len)) = find_annex_b_start_code(data) else {
        return nals;
    };
    let mut pos = first_sc + first_len;
    while pos < data.len() {
        if let Some((next_sc, next_len)) = find_annex_b_start_code(&data[pos..]) {
            let nal_end = pos + next_sc;
            if nal_end > pos {
                nals.push(&data[pos..nal_end]);
            }
            pos = nal_end + next_len;
        } else {
            nals.push(&data[pos..]);
            break;
        }
    }
    nals
}

/// Scans `data` for an MPEG-TS `0x47` sync byte locked to 188-byte packet stride.
#[inline]
pub fn find_mpeg_ts_sync(data: &[u8]) -> Option<usize> {
    #[cfg(target_arch = "aarch64")]
    if ragdoll_core::dispatch::neon() {
        return unsafe { find_mpeg_ts_sync_neon(data) };
    }
    find_mpeg_ts_sync_scalar(data)
}

pub fn find_mpeg_ts_sync_scalar(data: &[u8]) -> Option<usize> {
    let len = data.len();
    (0..len).find(|&i| data[i] == 0x47 && (i + 188 >= len || data[i + 188] == 0x47))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn find_mpeg_ts_sync_neon(data: &[u8]) -> Option<usize> {
    use std::arch::aarch64::*;
    let len = data.len();
    let mut i = 0usize;
    let v_sync = vdupq_n_u8(0x47);

    while i + 16 <= len {
        let chunk = unsafe { vld1q_u8(data.as_ptr().add(i)) };
        let eq = vceqq_u8(chunk, v_sync);
        if vmaxvq_u8(eq) == 0 {
            i += 16;
            continue;
        }
        for k in 0..16 {
            let pos = i + k;
            if data[pos] == 0x47 && (pos + 188 >= len || data[pos + 188] == 0x47) {
                return Some(pos);
            }
        }
        i += 16;
    }

    while i < len {
        if data[i] == 0x47 && (i + 188 >= len || data[i + 188] == 0x47) {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Converts MP4 / Matroska length-prefixed AVCC/HVCC NAL units (`nal_length_size` in `{1, 2, 4}`)
/// into an Annex-B byte stream (`0x00 0x00 0x00 0x01` start codes) in a single pass.
pub fn avcc_to_annex_b(avcc: &[u8], nal_length_size: usize, dst: &mut Vec<u8>) -> bool {
    if !matches!(nal_length_size, 1 | 2 | 4) {
        return false;
    }
    let mut pos = 0usize;
    while pos + nal_length_size <= avcc.len() {
        let nal_len = match nal_length_size {
            1 => avcc[pos] as usize,
            2 => u16::from_be_bytes([avcc[pos], avcc[pos + 1]]) as usize,
            4 => u32::from_be_bytes([avcc[pos], avcc[pos + 1], avcc[pos + 2], avcc[pos + 3]])
                as usize,
            _ => unreachable!(),
        };
        pos += nal_length_size;
        if pos + nal_len > avcc.len() {
            return false;
        }
        if nal_len > 0 {
            dst.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
            dst.extend_from_slice(&avcc[pos..pos + nal_len]);
        }
        pos += nal_len;
    }
    pos == avcc.len()
}

/// Converts an Annex-B byte stream (`00 00 01` / `00 00 00 01`) into 4-byte length-prefixed AVCC/HVCC format.
pub fn annex_b_to_avcc(annex_b: &[u8], dst: &mut Vec<u8>) {
    for nal in split_annex_b_nals(annex_b) {
        let len = nal.len() as u32;
        dst.extend_from_slice(&len.to_be_bytes());
        dst.extend_from_slice(nal);
    }
}
