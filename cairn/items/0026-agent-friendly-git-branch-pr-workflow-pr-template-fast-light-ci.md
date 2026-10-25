---
id: 26
uid: 0d243653-ddd3-401f-8961-1e7018c7618b
title: Agent-Friendly Git Branch/PR Workflow, PR Template & Fast Light CI
type: chore
status: done
milestone: m1-foundation
assignee: antigravity
created: 2026-10-24
updated: 2026-10-24
closed_at: 2026-10-24
priority: p0
area: arch
effort: s
---

Establish an agent-friendly Git branch and Pull Request workflow integrated with Cairn item IDs (`<type>/<cairn-id>-<slug>`), configure a lightweight fast GitHub Actions CI pipeline (`.github/workflows/ci.yml` on native `ubuntu-24.04-arm` NEON with concurrency cancellation), add `.github/pull_request_template.md` and `CONTRIBUTING.md`, document the PR loop in `AGENTS.md`, and enable auto-delete-branch squash PR settings on `oddurs/ragdoll`.

## Acceptance criteria

- [x] Light CI workflow (`.github/workflows/ci.yml`) runs fmt, clippy, and unit/SIMD tests on fast `ubuntu-24.04-arm` (`aarch64` NEON) with concurrency cancellation.
- [x] Structured `.github/pull_request_template.md` and `CONTRIBUTING.md` align PRs with Cairn item IDs, SIMD differential verification, and `cairn check`.
- [x] `AGENTS.md` includes the Git branch & PR workflow outside the `cairn:begin`/`cairn:end` block so `cairn agent --write AGENTS.md` preserves it.
- [x] GitHub repository `oddurs/ragdoll` configured with squash-merge and automatic branch deletion, and verified via a real Pull Request.

## 2026-10-24

Configured fast Light CI on ubuntu-24.04-arm (native aarch64 NEON) with concurrency cancel-in-progress, created .github/pull_request_template.md and CONTRIBUTING.md, added Git/PR workflow to AGENTS.md below cairn:end, and enabled squash-merge + auto-branch-deletion on oddurs/ragdoll.

## Result

Established agent- and contributor-friendly Git branch/PR workflow (<type>/<ID>-<slug>), PR template, CONTRIBUTING.md, AGENTS.md workflow docs, and fast Light CI on ubuntu-24.04-arm.
