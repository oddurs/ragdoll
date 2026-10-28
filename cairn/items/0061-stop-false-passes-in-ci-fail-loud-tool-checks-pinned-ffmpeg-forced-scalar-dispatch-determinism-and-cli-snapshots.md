---
id: 61
uid: f3a6391d-99e9-4c14-b579-49b741518793
title: 'Stop false passes in CI: fail-loud tool checks, pinned FFmpeg, forced-scalar dispatch, determinism and CLI snapshots'
type: chore
status: done
milestone: m11-validation
assignee: Claude
created: 2026-10-25
updated: 2026-10-28
closed_at: 2026-10-28
priority: p0
effort: m
area: arch
---

## Problem

Interop tests (for example `ffprobe_and_ffmpeg_bidirectional_interop`) `return` early when `ffprobe` / `ffmpeg` are missing, and Light CI does not install them, so they pass without testing anything. There is no way to force the scalar path on SIMD hardware, no determinism test across thread counts, and no tests for CLI output, JSON shape or exit codes.

## Acceptance criteria

- [x] Missing external tools fail the test unless an explicit opt-out environment variable is set; CI never sets it.
- [x] Light CI installs a pinned FFmpeg build and the interop tests run on every PR.
- [x] `RAGDOLL_SIMD=scalar|neon|avx2` forces dispatch; CI runs the differential suite under each available level.
- [x] Transcode output is byte-identical for `--threads 1` and `--threads N`, and for performance vs efficiency power modes.
- [x] Golden snapshots cover text output, a JSON schema covers every `--json` output, and exit codes are tested for each subcommand.

## 2026-10-28

1. ragdoll_core::testing::{require_tool, tool_succeeded} replace every silent skip in the interop tests (14 guards in ragdoll-format, ragdoll-codec and ragdoll-pipeline: '-version' probes that returned early and 'if let Ok(st) = status && st.success()' blocks that skipped when FFmpeg failed or lacked an encoder). A missing tool or failed reference command panics unless RAGDOLL_ALLOW_MISSING_TOOLS=1. Checked: with ffprobe off PATH the format interop test fails ('ffprobe is required by this test'); with the variable set it skips. 2. ci.yml installs ffmpeg=7:6.1.1-3ubuntu5 (Ubuntu 24.04 release pocket, libx264/libsvtav1 included) on both jobs and asserts the version string; the x86_64 job moved from ubuntu-latest to ubuntu-24.04 so the package pin holds, and now runs the whole workspace instead of three crates. 3. ragdoll_core::dispatch (SimdDispatch::active, neon(), avx2()) reads RAGDOLL_SIMD=scalar|neon|avx2 once (relaxed atomic after the first call; empty = unset; unknown or unsupported values panic). All 61 runtime dispatch sites in ragdoll-simd (31), ragdoll-pix (15), ragdoll-codec, ragdoll-core and ragdoll-format (15) consult it; scalar fallbacks are now plain code reachable on every target. cargo test --workspace passes with and without RAGDOLL_SIMD=scalar (83 tests), and clippy -D warnings (1.99.1) is clean for x86_64 and aarch64-unknown-linux-gnu. CI runs the suite twice per architecture (NEON/AVX2, then scalar).

## 2026-10-28

4. GopRingPipeline GOP length is now a fixed GOP_FRAMES = 15 (was min(15, ceil(frames / workers)), so inputs under 15 frames per worker encoded differently per thread count and in efficiency mode). tests/determinism.rs transcodes a 45-frame input with 1, 2 and 4 threads and in performance and efficiency modes (MP4 High, and MKV Baseline with scaling) and requires identical bytes; it fails with the old rule ('threads 4 differs from threads 1'). Outputs of inputs with fewer than 15 frames per worker change once with this. 5. crates/ragdoll-cli/tests/cli.rs: 13 golden snapshots (help, version, probe, probe --gop-map, remux, transcode video and audio, cut, concat, package, extract, metrics, synth) with timings, rates, machine thread counts and the SIMD/CPU lines masked, identical under RAGDOLL_SIMD=scalar; a strict schema (exact keys and types, hand-rolled JSON parser since the workspace has no dependencies) for the --json output of probe (video and audio input), remux, transcode (video and audio), cut, concat, package, extract, metrics and synth; and the exit status of every subcommand. The CLI now has CliError: usage errors (unknown subcommand or option, missing arguments) exit 2 with a usage hint and no longer print 'invalid bitstream syntax: ...'; runtime failures exit 1; bench and self-test reject unknown arguments. self-test's success path is an #[ignore] test (about a minute in debug) that the x86_64 CI job runs.

## 2026-10-28

bench --json is covered too: an #[ignore] schema test (bench_json_matches_schema, about 135 s in debug) runs with self_test_passes in the x86_64 job's slow step.

## Result

Interop tests fail instead of passing when FFmpeg is missing or fails (opt-out RAGDOLL_ALLOW_MISSING_TOOLS=1, never set in CI); CI installs pinned FFmpeg 6.1.1 and runs the whole suite on aarch64 and x86_64 twice, the second time with RAGDOLL_SIMD=scalar, which every SIMD dispatch site honours; transcode output no longer depends on threads or power mode; ragdoll-cli tests pin text output (snapshots), JSON shape (schemas, bench included) and exit status (0 / 1 runtime / 2 usage).
