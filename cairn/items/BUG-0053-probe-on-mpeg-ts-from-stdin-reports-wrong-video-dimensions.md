---
id: 53
uid: 52fbc6f9-499b-4eca-a181-7e93b5552a1c
title: probe on MPEG-TS from stdin reports wrong video dimensions
type: bug
status: done
milestone: m10-production-v1
assignee: Oddur Sigurdsson
created: 2026-10-25
updated: 2026-10-25
closed_at: 2026-10-25
priority: p2
effort: s
area: format
---

## Problem

`ragdoll transcode sample.mp4 - --format ts | ragdoll probe -` reports `1920x1080` for a 1280x720 source. Probing the same stream from a file path is correct, so the stdin path (`MmapSource::from_reader`) or the TS track-parameter discovery on a non-mapped buffer is not reading the SPS.

Found while writing the CLI docs (0052); documented there as a known issue.

## Acceptance criteria

- [x] A regression test pipes a 1280x720 MPEG-TS through the stdin path and asserts the probed dimensions.
- [x] `probe -` and `probe file.ts` report identical track parameters for the same bytes.
- [x] Known-issue callout removed from `site/app/docs/cli/probe/page.mdx`.
