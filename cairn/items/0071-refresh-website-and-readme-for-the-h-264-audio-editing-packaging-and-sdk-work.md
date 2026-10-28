---
id: 71
uid: 84c0492e-dd8a-468d-99fe-36d0d32e9db9
title: Refresh website and README for the H.264, audio, editing, packaging and SDK work
type: docs
status: planned
milestone: m11-validation
created: 2026-10-25
updated: 2026-10-28
priority: p1
effort: s
area: arch
---

## Problem

Since the site launched, H.264 Main/High with CABAC, AAC-LC and Opus, PNG and JPEG, cut and concat, packaging, AVX2 kernels, the SDK and strict option parsing have merged. The site still lists several of these as planned or missing; for example, the roadmap page says no AVX2 kernels exist. Content must be re-verified against the code before it is republished.

Two findings change what the pages should say:

- BUG-0060 is fixed: Ragdoll-encoded H.264 (Baseline, Main and High, in MP4, MKV and TS) decodes byte-identically to FFmpeg. Pages must not describe a decoder mismatch, and any quality figures produced through `transcode`, `extract` or `metrics` on multi-GOP H.264 before the fix are wrong and must be re-measured.
- The roadmap page's statement that decoding is not bounded is still true: BUG-0072 shows `transcode`, `metrics`, `concat` and thumbnail extraction decode the whole input. Keep it as a known limitation until BUG-0072 lands, and do not describe memory as bounded independent of input length.

## Acceptance criteria

- [ ] Every page re-checked against `main`; new subcommands and codecs documented with real output.
- [ ] Roadmap and known limitations updated: no H.264 decoder mismatch is described, and unbounded decode memory is listed as a known limitation until BUG-0072 is fixed.
- [ ] Quality and performance figures for H.264 are re-measured on code that includes the BUG-0060 fix.
- [ ] No internal process references reintroduced.
