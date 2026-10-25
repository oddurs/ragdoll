---
id: 46
uid: deba5776-f406-44bb-acba-6c39ec4a270b
title: Smart-GOP Frame-Accurate ragdoll cut & ragdoll concat Non-Linear Editing Engine
type: feature
status: done
milestone: m9-edit-image-package
assignee: Oddur Sigurdsson
depends_on:
- 37
- 39
created: 2026-10-25
updated: 2026-10-25
closed_at: 2026-10-25
priority: p0
area: pipeline
effort: l
---

## Problem

In `ffmpeg`, cutting a video at arbitrary timestamps (`-ss 00:01:12.400 -to 00:03:05.100`) forces an agonizing choice: `-c copy` snaps to the nearest keyframe (producing frozen/corrupt frames or multi-second A/V desync at the start), while re-encoding the entire file wastes 98% of CPU cycles on interior GOPs that did not change. Similarly, concatenating files in `ffmpeg` requires fragile external `concat.txt` demuxer scripts.

## Architecture & SIMD Design

1. Implement **Smart-GOP Trimming (`ragdoll cut <IN> <OUT> --from <TIME> --to <TIME> [--keyframes-only]`)**:
   - Use `Mp4Demuxer` / `MkvDemuxer` $O(\log N)$ keyframe index to locate the exact GOPs containing `--from` and `--to`.
   - If `--from` lands mid-GOP, decode that single boundary GOP, drop frames prior to `--from`, and re-encode the remaining frames of that GOP starting with a fresh IDR keyframe matching the stream's `SPS`/`PPS` profile.
   - Zero-copy `mmap` stream-copy all untouched interior GOPs at `>1 GB/s`, and trim/re-encode the trailing boundary GOP if needed, while slicing audio packets to the exact sample/frame boundary.
2. Implement **Zero-Copy & Smart-Normalized `ragdoll concat <IN1> <IN2> ... -o <OUT>`**:
   - When input streams share compatible codec parameters (`SPS`/`PPS`/resolution), zero-copy splice packets with continuous monotonic 128-bit PTS/DTS offset adjustment. When resolution or framerate differs, automatically normalize mismatched inputs via the fused L1 resampler.

## Acceptance criteria

- [x] `ragdoll cut` performs frame-exact cuts on mid-GOP timestamps in `< 50 ms` on a 10-minute video by re-encoding only the boundary GOP and zero-copy copying interior GOPs.
- [x] `ragdoll concat` seamlessly joins multiple `.mp4`, `.mkv`, or `.ts` clips with monotonic timestamps and zero A/V drift.

## 2026-10-25

Implemented Smart-GOP frame-accurate cut_media (ragdoll cut) re-encoding only boundary GOPs and zero-copy copying interior GOPs, plus zero-copy/normalizing concat_media (ragdoll concat).
