"""Generates src/opus/celt_tables.rs from a libopus 1.3.1 source tree.

Usage: python3 celt_tables.py LIBOPUS > ../../src/opus/celt_tables.rs && rustfmt --edition 2024 ../../src/opus/celt_tables.rs
"""
import re, sys
src = sys.argv[1]
h = open(src + "/celt/static_modes_fixed.h").read()
qb = open(src + "/celt/quant_bands.c").read()
modes = open(src + "/celt/modes.c").read()
def arr(text, name):
    m = re.search(r"%s(?:\s*\[[^\]]*\])+\s*=\s*\{(.*?)\};" % re.escape(name), text, re.S)
    body = re.sub(r"/\*.*?\*/", "", m.group(1), flags=re.S)
    return [int(v) for v in re.findall(r"-?\d+", body)]
out = []
def emit(name, ty, vals, per=16, doc=None):
    if doc: out.append(f"/// {doc}")
    out.append(f"pub static {name}: [{ty}; {len(vals)}] = [")
    for i in range(0, len(vals), per):
        out.append("    " + ", ".join(str(v) for v in vals[i:i+per]) + ",")
    out.append("];")
    out.append("")
emit("EBANDS", "i16", arr(modes, "eband5ms"), doc="Band edges in units of 8 MDCT bins at LM 0 (`eband5ms`).")
emit("BAND_ALLOCATION", "u8", arr(modes, "band_allocation"), 21, doc="Static bit allocation vectors, 11 x 21 (`band_allocation`).")
emit("WINDOW", "i16", arr(h, "window120"), 12, doc="Q15 MDCT overlap window (`window120`).")
emit("LOG_N", "i16", arr(h, "logN400"), doc="log2 of band widths in 1/8 bits (`logN400`).")
emit("CACHE_INDEX", "i16", arr(h, "cache_index50"), 21, doc="Pulse cache index by (LM + 1, band) (`cache_index50`).")
emit("CACHE_BITS", "u8", arr(h, "cache_bits50"), doc="Pulse cache (`cache_bits50`).")
emit("CACHE_CAPS", "u8", arr(h, "cache_caps50"), 21, doc="Band bit caps by (LM, C, band) (`cache_caps50`).")
tw = arr(h, "fft_twiddles48000_960")
out.append("/// FFT twiddles `(r, i)` for the 480-point FFT (`fft_twiddles48000_960`).")
out.append(f"pub static FFT_TWIDDLES: [(i16, i16); {len(tw)//2}] = [")
for i in range(0, len(tw), 16):
    out.append("    " + ", ".join(f"({tw[j]}, {tw[j+1]})" for j in range(i, min(i+16, len(tw)), 2)) + ",")
out.append("];\n")
for n in (480, 240, 120, 60):
    emit(f"FFT_BITREV{n}", "i16", arr(h, f"fft_bitrev{n}"))
emit("MDCT_TWIDDLES", "i16", arr(h, "mdct_twiddles960"), doc="MDCT twiddles for N = 1920, 960, 480, 240 (`mdct_twiddles960`).")
emit("E_MEANS", "i8", arr(qb, "eMeans")[:25], doc="Mean band energies in Q4 (`eMeans`, fixed point).")
ep = arr(qb, "e_prob_model")
emit("E_PROB_MODEL", "u8", ep, 14, doc="Laplace parameters for coarse energy by [LM][intra][band] (`e_prob_model`).")
print("//! CELT mode tables for 48 kHz / 960 samples, from libopus `static_modes_fixed.h`,")
print("//! `modes.c` and `quant_bands.c` (BSD-3-Clause, Xiph.Org Foundation and contributors) by")
print("//! `tools/opus_tables/celt_tables.py`.")
print("")
print("\n".join(out))
