---
id: 98
uid: 415ddbad-d116-476e-b4cc-09fc4381b6e7
title: CLI text snapshots fail when a masked timing changes width (probe latency of 1000 us or more)
type: bug
status: done
milestone: m11-validation
created: 2026-10-29
updated: 2026-10-29
closed_at: 2026-10-29
priority: p1
---

## What happens

`text_output_matches_golden_snapshots` (`crates/ragdoll-cli/tests/cli.rs`) masks numbers followed by a time unit, but not the padding before them. `probe` prints `Probe Latency : {:>8.2} µs`, so a latency of 1000 µs or more (a loaded machine) has one space fewer before the masked value and the `probe` / `probe_gop_map` snapshots differ.

## What should happen

The snapshot comparison does not depend on how long probing took.

## Reproduction

1. Run `cargo test -p ragdoll-cli --test cli text_output_matches_golden_snapshots` while the machine is busy enough that probing takes over 1 ms (seen during a parallel workspace build).

## Acceptance criteria

- [x] A regression test reproduces the failure and passes with the fix.

## 2026-10-29

The normaliser now masks the run of spaces before a masked timing together with it (one space is kept); probe.txt and probe_gop_map.txt re-blessed, only their latency lines changed. A 1234.56 us latency and a 12.34 us one now normalise to the same line.
