#!/usr/bin/env python3
"""Generates src/opus/silk_tables.rs from a libopus 1.3.1 source tree.

Usage: python3 silk_tables.py LIBOPUS/silk > ../../src/opus/silk_tables.rs && rustfmt --edition 2024 ../../src/opus/silk_tables.rs
"""
import re, sys, os
d = sys.argv[1]
files = ["tables_LTP.c", "tables_NLSF_CB_NB_MB.c", "tables_NLSF_CB_WB.c", "tables_gain.c",
         "tables_other.c", "tables_pitch_lag.c", "tables_pulses_per_block.c", "table_LSF_cos.c",
         "pitch_est_tables.c", "resampler_rom.c"]
macros = {"OFFSET_UVL_Q10": 100, "OFFSET_UVH_Q10": 240, "OFFSET_VL_Q10": 32, "OFFSET_VH_Q10": 100,
          "PE_NB_CBKS_STAGE3_MIN": 16, "PE_NB_CBKS_STAGE3_MID": 24, "PE_NB_CBKS_STAGE3_MAX": 34}
tys = {"opus_uint8": "u8", "opus_int8": "i8", "opus_int16": "i16", "opus_int32": "i32"}
out = ["//! SILK tables, flattened from libopus `silk/tables_*.c` and `resampler_rom.c`",
       "//! (BSD-3-Clause, Xiph.Org Foundation and contributors) by `tools/opus_tables/silk_tables.py`.",
       "//! Multi-dimensional arrays are stored row-major.", "",
       "#![allow(dead_code)]", ""]
def ev(t):
    t = t.strip()
    m = re.fullmatch(r"SILK_FIX_CONST\(\s*([-0-9.e]+)\s*,\s*(\d+)\s*\)", t)
    if m:
        return int(float(m.group(1)) * (1 << int(m.group(2))) + 0.5)
    if t in macros:
        return macros[t]
    return int(t, 0)
for f in files:
    s = open(os.path.join(d, f)).read()
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"//[^\n]*", "", s)
    for m in re.finditer(r"(?:static\s+)?(?:silk_DWORD_ALIGN\s+)?const\s+(opus_u?int\d+)\s+(\w+)((?:\s*\[[^\]]*\])+)\s*=\s*\{(.*?)\};", s, re.S):
        ty, name, dims, body = m.groups()
        body = body.replace("{", " ").replace("}", " ")
        vals = [ev(x) for x in body.split(",") if x.strip()]
        rn = re.sub(r"^silk_", "", name).upper()
        dimtxt = re.sub(r"\s+", "", dims)
        out.append(f"/// `{name}{dimtxt}`")
        out.append(f"pub const {rn}: [{tys[ty]}; {len(vals)}] = [")
        for i in range(0, len(vals), 16):
            out.append("    " + ", ".join(str(v) for v in vals[i:i+16]) + ",")
        out.append("];")
        out.append("")
print("\n".join(out))
