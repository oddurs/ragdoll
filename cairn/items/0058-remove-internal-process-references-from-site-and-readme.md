---
id: 58
uid: 93693db2-612a-4a34-98e1-48825a2270f4
title: Remove internal process references from site and README
type: docs
status: done
milestone: m10-production-v1
assignee: Oddur Sigurdsson
created: 2026-10-25
updated: 2026-10-25
closed_at: 2026-10-25
priority: p2
effort: s
area: arch
---

## Problem

The public site and README expose internal tracking vocabulary: milestone codes (M1-M10), cairn item and ADR identifiers, and the cairn workflow. Readers need product language: what works today, what is in development, what is planned, and known limitations.

## Acceptance criteria

- [x] No milestone codes, cairn item IDs, ADR IDs or cairn commands appear on the site or in the README.
- [x] Status and roadmap are expressed as available / in development / planned, plus known limitations.
- [x] CLI user-facing output no longer prints milestone codes, and docs show the refreshed output.
- [x] Site builds; Rust gate and `cairn check --render --strict` pass.
