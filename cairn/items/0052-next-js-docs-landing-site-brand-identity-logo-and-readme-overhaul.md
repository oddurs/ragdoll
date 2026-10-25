---
id: 52
uid: 58ee07b7-9efb-48ee-9147-46b8b1b052fe
title: Next.js docs & landing site, brand identity, logo and README overhaul
type: docs
status: done
milestone: m10-production-v1
assignee: Oddur Sigurdsson
created: 2026-10-25
updated: 2026-10-25
closed_at: 2026-10-25
priority: p2
effort: l
area: arch
---

## Problem

The project has no website or long-form documentation, and the README reads as marketing copy rather than an engineering project. The identity (name usage, mark, palette, typography) is inconsistent across README and assets.

## Scope

- Static Next.js site under `site/` (no backend, static export) with a landing page and a docs section.
- Docs written against the actual code: installation, CLI reference, crate-by-crate architecture, container/codec support matrix, benchmarking methodology, roadmap, contributing.
- A new mark and wordmark (SVG), favicon and Open Graph image, plus a short brand guide.
- README rewritten to a sober, scannable open-source format.

## Acceptance criteria

- [x] `site/` builds with `next build` as a fully static export with no server runtime.
- [x] Docs cover install, quickstart, every CLI subcommand, every crate, format/codec support and benchmark methodology, and match the code on main.
- [x] New logo mark, wordmark, favicon and social card committed under `assets/` and used by the site and README.
- [x] README rewritten: one-paragraph pitch, status, install, usage, crate map, support matrix, links to docs.
- [x] Light CI unaffected (Rust gate still passes) and `cairn check --render --strict` passes.

## 2026-10-25

Site is Next.js 16.4.3 (App Router, MDX via @next/mdx, rehype-pretty-code/shiki dual themes) with output: export; 33 static pages, no server runtime. Pinned next to 16.4.3 because npm audit flags 16.0.0-16.4.0 (GHSA-2464-fhr4-w9xw, GHSA-p3m5-59mp-jv5r, GHSA-673q-mjm6-r5xr); those are server features a static export does not ship, but dev server is exposed. Every doc claim was checked against crates/ on main and the Rust snippets were compiled against the workspace in a scratch crate. Docs deliberately state gaps found while writing: no AVX2 kernels exist (README claimed them), transcode/extract/metrics decode the whole input before the bounded GOP ring, H.264 encoder leans on I_PCM, recipe start/duration/sample_rate and remux --faststart are no-ops, FFmpeg kernel-level numbers are not reproducible in-tree, probe on piped TS reports wrong dimensions. Filed as BUG-0053, 0054, SIMD-0055, BENCH-0056, 0057. New mark: an R built on a 3x3 macroblock grid, first block in tally red #FF4F2E; assets/brand/generate.py regenerates every SVG and the social card from Geist outlines (no font dependency on GitHub). Old zig-zag logo removed: it leaned on FFmpeg's own mark. GitHub Pages deploy in .github/workflows/site.yml needs Pages source set to GitHub Actions.
