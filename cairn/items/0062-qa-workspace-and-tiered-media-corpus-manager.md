---
id: 62
uid: 7976ce3f-d062-405d-bb60-7ab2c3d73cef
title: qa/ workspace and tiered media corpus manager
type: feature
status: planned
milestone: m11-validation
created: 2026-10-25
updated: 2026-10-25
priority: p1
effort: m
area: arch
---

## Problem

Validation tooling needs third-party crates (libFuzzer bindings, a test runner, a report writer) that must not enter the library's zero-dependency `Cargo.lock`, and it needs a reproducible media corpus far larger than belongs in git.

## Architecture & SIMD Design

A separate `qa/` Cargo workspace with a `ragdoll-qa` binary (`qa corpus`, `qa conformance`, `qa interop`, `qa fuzz`, `qa rd`, `qa perf`, `qa report`). The corpus has four tiers: (1) synthesized media from `synth`, regenerated deterministically and kept in git; (2) conformance suites, downloaded by manifest with pinned SHA-256s and cached in CI; (3) a real-world zoo of phone captures, OBS recordings, broadcast TS and unusual encoders; (4) minimized regression files from every bug and fuzz crash, kept in git forever.

## Acceptance criteria

- [ ] `qa/` builds independently; the root `Cargo.lock` still lists only the seven `ragdoll-*` packages.
- [ ] `qa corpus fetch` downloads and verifies every manifest entry, and `qa corpus verify` fails on any hash mismatch.
- [ ] CI caches tier 2 and 3 by manifest hash.
