---
id: 61
uid: f3a6391d-99e9-4c14-b579-49b741518793
title: 'Stop false passes in CI: fail-loud tool checks, pinned FFmpeg, forced-scalar dispatch, determinism and CLI snapshots'
type: chore
status: planned
milestone: m11-validation
created: 2026-10-25
updated: 2026-10-25
priority: p0
effort: m
area: arch
---

## Problem

Interop tests (for example `ffprobe_and_ffmpeg_bidirectional_interop`) `return` early when `ffprobe` / `ffmpeg` are missing, and Light CI does not install them, so they pass without testing anything. There is no way to force the scalar path on SIMD hardware, no determinism test across thread counts, and no tests for CLI output, JSON shape or exit codes.

## Acceptance criteria

- [ ] Missing external tools fail the test unless an explicit opt-out environment variable is set; CI never sets it.
- [ ] Light CI installs a pinned FFmpeg build and the interop tests run on every PR.
- [ ] `RAGDOLL_SIMD=scalar|neon|avx2` forces dispatch; CI runs the differential suite under each available level.
- [ ] Transcode output is byte-identical for `--threads 1` and `--threads N`, and for performance vs efficiency power modes.
- [ ] Golden snapshots cover text output, a JSON schema covers every `--json` output, and exit codes are tested for each subcommand.
