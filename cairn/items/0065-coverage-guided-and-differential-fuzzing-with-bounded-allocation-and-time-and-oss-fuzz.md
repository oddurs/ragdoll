---
id: 65
uid: 72b6310a-cfe6-4ab3-8852-6a8b3f058e3a
title: Coverage-guided and differential fuzzing with bounded allocation and time, and OSS-Fuzz
type: feature
status: backlog
milestone: m11-validation
depends_on:
- 62
created: 2026-10-25
updated: 2026-10-25
priority: p1
effort: l
area: arch
---

## Problem

The existing malformed-bitstream test applies a fixed set of deterministic mutations. That is a good regression floor, but it does not explore the input space the way coverage-guided fuzzing does, and it cannot detect allocation bombs or pathological slowdowns.

## Architecture & SIMD Design

One libFuzzer target per demuxer, parser and decoder in the `qa/` workspace, seeded from corpus tiers 2–4. Each target asserts: no panic, total allocation below a cap (a box that declares 4 GB must be rejected, not allocated), and bounded time per input. A differential mode runs our demuxer and FFmpeg's on the same mutated input and flags disagreements on well-formed prefixes. Crashes are minimized into corpus tier 4 automatically.

## Acceptance criteria

- [ ] Targets exist for every container demuxer, H.264 / AAC / Opus / FLAC decoder, and the PNG / JPEG decoders.
- [ ] Nightly CI fuzzes each target for at least one hour; new crashes open a cairn bug with the minimized input.
- [ ] An OSS-Fuzz integration is submitted.
