---
id: 71
uid: 84c0492e-dd8a-468d-99fe-36d0d32e9db9
title: Refresh website and README for the H.264, audio, editing, packaging and SDK work
type: docs
status: planned
milestone: m11-validation
created: 2026-10-25
updated: 2026-10-25
priority: p1
effort: s
area: arch
---

## Problem

Since the site launched, H.264 Main/High with CABAC, AAC-LC and Opus, PNG and JPEG, cut and concat, packaging, AVX2 kernels, the SDK and strict option parsing have merged. The site still lists several of these as planned or missing; for example, the roadmap page says no AVX2 kernels exist and that decoding is not bounded. Content must be re-verified against the code before it is republished; in particular the H.264 decoder bug in this milestone must be reflected.

## Acceptance criteria

- [ ] Every page re-checked against `main`; new subcommands and codecs documented with real output.
- [ ] Roadmap and known limitations updated, including the H.264 decoder mismatch until it is fixed.
- [ ] No internal process references reintroduced.
