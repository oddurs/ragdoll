---
id: 5
uid: ff8653f8-9929-4b76-9308-4e4a7af47168
key: m5-cli-bench
title: Phase 5 — GOP-Parallel Pipeline, Modern CLI & Marketing Benchmark Suite
type: milestone
status: planned
created: 2026-10-24
updated: 2026-10-24
priority: p0
area: cli
---

Unify all engines under a work-stealing GOP-parallel pipeline scheduler, declarative TOML recipe runner, ergonomic CLI (`probe`, `remux`, `transcode`, `extract`, `metrics`, `synth`, `bench`), and a self-contained head-to-head benchmark harness against FFmpeg 9.0.2 on Apple M4.

## Release gate

All CLI subcommands and the 6-workload head-to-head FFmpeg benchmark suite (`ragdoll bench --compare-ffmpeg`) run end-to-end with verified benchmark results recorded in Cairn.
