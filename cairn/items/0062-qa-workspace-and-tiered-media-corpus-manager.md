---
id: 62
uid: 7976ce3f-d062-405d-bb60-7ab2c3d73cef
title: qa/ workspace and tiered media corpus manager
type: feature
status: done
milestone: m11-validation
assignee: Claude
created: 2026-10-25
updated: 2026-10-28
closed_at: 2026-10-28
priority: p1
effort: m
area: arch
---

## Problem

Validation tooling needs third-party crates (libFuzzer bindings, a test runner, a report writer) that must not enter the library's zero-dependency `Cargo.lock`, and it needs a reproducible media corpus far larger than belongs in git.

## Architecture & SIMD Design

A separate `qa/` Cargo workspace with a `ragdoll-qa` binary (`qa corpus`, `qa conformance`, `qa interop`, `qa fuzz`, `qa rd`, `qa perf`, `qa report`). The corpus has four tiers: (1) synthesized media from `synth`, regenerated deterministically and kept in git; (2) conformance suites, downloaded by manifest with pinned SHA-256s and cached in CI; (3) a real-world zoo of phone captures, OBS recordings, broadcast TS and unusual encoders; (4) minimized regression files from every bug and fuzz crash, kept in git forever.

## Acceptance criteria

- [x] `qa/` builds independently; the root `Cargo.lock` still lists only the seven `ragdoll-*` packages.
- [x] `qa corpus fetch` downloads and verifies every manifest entry, and `qa corpus verify` fails on any hash mismatch.
- [x] CI caches tier 2 and 3 by manifest hash.

## 2026-10-28

qa/ is its own Cargo workspace (package ragdoll-qa, binary qa; root Cargo.toml excludes it) using only std plus path dependencies on the ragdoll crates: SHA-256 is implemented in qa/src/sha256.rs (FIPS 180 vectors tested, including 1,000,000 'a' in uneven chunks) and downloads use curl, so neither lock file gains a third-party crate; the root Cargo.lock still lists exactly the seven ragdoll-* packages (checked in CI). Only 'qa corpus' exists so far; conformance, interop, fuzz, rd, perf and report are items 0063-0069. Manifest qa/corpus/manifest.toml (a [[file]] TOML subset parsed by hand, unknown keys and duplicate names rejected): tier 1, five clips made by ragdoll synth (MP4, MKV, Y4M, WAV, FLAC; 100 KB) in git under qa/corpus/synth, which 'qa corpus synth' regenerates and verifies; tier 2, the RFC 8251 Opus vectors, five JVT H.264 conformance streams and three ISO AAC conformance streams with their reference PCM (FATE suite); tier 3, seven FATE real-world files (TS, MKV with Opus codec delay, MOV/M4A with AAC priming, MP4 edit lists, 7.1 Opus in TS); tier 4, the 43 regression fixtures already in git under crates/*/tests/data. Tiers 2 and 3 total 85 MB. Evidence: a cold 'qa corpus fetch' took 39 s and 'verify' passed 67/67; flipping one byte of BA1_Sony_D.jsv made verify fail (exit 1, reporting the hash); fetch replaced it and verify passed again. CI (.github/workflows/qa.yml) caches qa/corpus/cache with key qa-corpus-${hashFiles(manifest)}, runs fmt/clippy/tests for qa, fetch, verify and synth, and asserts the root lock file's package list.

## Result

qa/ is its own workspace (zero third-party deps): qa corpus list|fetch|verify|synth over a 4-tier SHA-256-pinned manifest (67 entries); tiers 2/3 download to qa/corpus/cache or $RAGDOLL_QA_CACHE, cached in CI by manifest hash (.github/workflows/qa.yml).
