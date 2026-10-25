---
id: 48
uid: 7ce470c8-6f7b-499c-8924-0a8af7b76148
title: 'Fused L1 Compositional Video Filters: crop (--auto-crop), pad, rotate, fps, overlay & Oklab HDR-to-SDR tonemap'
type: kernel
status: done
milestone: m9-edit-image-package
assignee: Oddur Sigurdsson
depends_on:
- 47
created: 2026-10-25
updated: 2026-10-25
closed_at: 2026-10-25
priority: p1
area: pix
effort: l
---

## Mathematical Specification

1. **Spatial Geometry (`crop`, `--auto-crop`, `pad`, `rotate`/`transpose`, `flip`)**:
   - Zero-copy plane view sub-windowing for even-aligned `crop`; SIMD luma energy scanner across sample frames for `--auto-crop` black-bar letterbox/pillarbox detection (`libavfilter/vf_cropdetect.c`).
   - Block-transposed $8\times8$ / $16\times16$ NEON `vtrn` cache-oblivious $90^\circ/270^\circ$ rotation and $180^\circ$/horizontal/vertical flip.
2. **Alpha-Blended Watermark / PiP (`overlay`)**:
   - Pre-multiplied / straight `RGBA8` over `YUV420p` blending: $Y' = ((255 - \alpha) Y + \alpha Y_{\text{ovl}} + 127) \gg 8$ in NEON `vmull_u8` / `vmlal_u8`.
3. **Perceptual Oklab HDR10 (`BT.2020 PQ / HLG`) $\to$ SDR (`BT.709`) Tone Mapping (`tonemap`)**:
   - Evaluate SMPTE ST 2084 (PQ) / ARIB STD-B67 (HLG) EOTF via Q15 LUT + piecewise rational polynomial, map gamut and luminance in perceptual **Oklab** ($L, a, b$) preserving hue and chroma saturation without RGB channel clipping, and encode BT.709 OETF.

## Register & Cache Layout (NEON / AVX2)

- Execute `crop -> scale -> tonemap -> overlay` inside the fused L1 scanline pipeline (`ragdoll-pix`) without full-frame intermediate heap allocations.

## Acceptance criteria

- [x] Scalar reference and NEON/AVX2 SIMD implementations pass differential tests across `crop`, `rotate`, `overlay`, and `tonemap`.
- [x] `--auto-crop` accurately detects letterbox dimensions on synthetic and real widescreen inputs.
- [x] Zero heap allocations in the per-scanline filter loop.

## 2026-10-25

Implemented crop_frame, detect_auto_crop (--auto-crop), pad_frame, rotate_frame (90/180/270/hflip/vflip), NEON-accelerated RGBA8 overlay_rgba_onto_frame, and NEON-accelerated Oklab HDR-to-SDR tonemap_hdr_to_sdr_oklab in ragdoll-pix and wired into ragdoll-pipeline and ragdoll-cli.
