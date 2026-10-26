---
id: 69
uid: 5e885595-8acb-4118-8659-4a543f6f9f88
title: QA report, generated support matrix and public scorecard, with tiered CI
type: feature
status: backlog
milestone: m11-validation
depends_on:
- 63
- 64
created: 2026-10-25
updated: 2026-10-25
priority: p1
effort: m
area: arch
---

## Problem

The website's Formats & codecs page is hand-written and has already drifted from the code more than once, in both directions. The support matrix should be a build artifact of the QA results.

## Architecture & SIMD Design

`qa report` merges conformance, interop, fuzzing, determinism, rate-distortion and performance results into one versioned JSON document. The site build reads it to render the Formats & codecs matrix (a feature is "Yes" only when its conformance and interop checks pass on `main`) and a public scorecard page. CI is tiered: PR (under 5 minutes), merge (about 20 minutes, publishes the report), nightly (fuzzing, rate-distortion, performance, soak) and release (full suite, signed report attached to the release).

## Acceptance criteria

- [ ] The Formats & codecs page is generated from the report; hand edits to support values are impossible.
- [ ] A scorecard page on the site shows per-feature pass rates and the commit they were measured at.
- [ ] All four CI tiers exist and run on schedule.
