---
id: 68
uid: c702b059-e8fc-40ef-befe-b31c92f87dcb
title: 'Performance and memory gates: instruction-count budgets, allocation counting and soak tests'
type: benchmark
status: backlog
milestone: m11-validation
created: 2026-10-25
updated: 2026-10-25
priority: p2
effort: m
area: bench
---

## Workload & Fair Comparison Protocol

Wall time is too noisy for CI gates. Per-kernel and per-workload retired-instruction counts (PMU on Apple Silicon, `perf stat` on Linux) are stable enough to gate on. A counting global allocator in tests enforces zero allocations inside hot loops. A soak test streams four hours of synthesized 4K through `transcode` and samples RSS.

## Target Metrics vs FFmpeg

Budgets are stored in-tree per kernel. A change of more than 2% in instruction count fails, unless the budget is updated in the same PR with a justification.

## Acceptance criteria

- [ ] Deterministic test media synthesis and head-to-head execution in `ragdoll bench --compare-ffmpeg`.
- [ ] Measured wall-clock time, throughput (GB/s or fps), and peak RSS recorded in `## Result`.
- [ ] The soak test shows flat RSS (bounded by budget plus fixed overhead) across the full run.
