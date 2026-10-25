---
id: 57
uid: 5bc84baf-5622-42dd-9749-86cf818f6634
title: 'CLI: reject unknown options and remove or implement no-op flags'
type: chore
status: done
milestone: m10-production-v1
assignee: Oddur Sigurdsson
created: 2026-10-25
updated: 2026-10-25
closed_at: 2026-10-25
priority: p2
effort: s
area: cli
---

## Problem

The hand-rolled argument parser in `ragdoll-cli` silently ignores unknown options, so typos have no effect and no error. Several accepted flags do nothing: `remux --faststart` (MP4 is always fast-start), `transcode --color`, and the recipe fields `start`, `duration`, `audio.codec`, `audio.sample_rate` and `output.faststart` are parsed but never applied by `execute_recipe`. The docs (0052) describe these as no-ops.

## Acceptance criteria

- [x] Unknown options produce `error: unknown option` and exit status 1.
- [x] Every documented flag and recipe key either has an effect or is removed.
- [x] Docs pages for remux, transcode and recipes updated to match.
