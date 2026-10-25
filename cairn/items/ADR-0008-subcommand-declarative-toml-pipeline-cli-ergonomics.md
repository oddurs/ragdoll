---
id: 8
uid: 8309a8e8-8020-4135-880d-21068d45805b
title: Subcommand & Declarative TOML Pipeline CLI Ergonomics
type: decision
status: done
milestone: m1-foundation
assignee: antigravity
created: 2026-10-24
updated: 2026-10-24
closed_at: 2026-10-24
priority: p1
area: cli
effort: s
---

## Context

FFmpeg's CLI flag order sensitivity (`-ss` before vs. after `-i`), cryptic filtergraph strings (`-filter_complex "[0:v]scale=1920:1080:flags=lanczos[v]"`), and silent colorimetry drops make everyday media engineering error-prone.

## Options & Trade-offs

1. **Legacy `-i` Flag Soup**: Familiar to existing scripts, but perpetuates ambiguous option scoping.
2. **Structured Subcommands + Declarative `.toml` Recipes (Selected)**: Dedicated subcommands (`ragdoll probe`, `ragdoll remux`, `ragdoll transcode`, `ragdoll extract`, `ragdoll metrics`, `ragdoll synth`, `ragdoll bench`) with typed flags, ASCII GOP waterfall visualizations, and reproducible `.toml` pipeline definitions.

## Decision

Design the `ragdoll` CLI around explicit subcommands, rich terminal telemetry (GOP waterfalls, bitrate sparklines, SIMD throughput), and declarative TOML pipeline recipes.

## Acceptance criteria

- [x] The architectural invariant, its rationale, and its consequences are recorded.

## 2026-10-24

Defined subcommand CLI architecture (probe, remux, transcode, extract, metrics, synth, bench) and declarative TOML pipeline recipes.

## Result

The ragdoll CLI exposes structured subcommands (probe, remux, transcode, extract, metrics, synth, bench), ASCII GOP waterfall & bitrate visualizations, --json machine output, and declarative .toml pipeline recipes.
