---
id: 33
uid: 582637f1-6eca-4c5a-9252-cf3f75bc12df
key: m9-edit-image-package
title: Phase 9 — Smart-GOP Editing (cut/concat), Spatial Filters, PNG/JPEG & ABR Packaging (package)
type: milestone
status: done
assignee: Oddur Sigurdsson
created: 2026-10-25
updated: 2026-10-25
closed_at: 2026-10-25
priority: p0
area: cli
---

Introduce Ragdoll's opinionated workflow upgrades over FFmpeg: frame-accurate Smart-GOP `ragdoll cut` and `ragdoll concat` (re-encoding only boundary GOPs while zero-copy `mmap` stream-copying interior GOPs), pure-Rust `.png` and `.jpg` image I/O, fused L1 compositional filters (`crop`, `--auto-crop`, `pad`, `rotate`, `fps`, `overlay`, Oklab HDR-to-SDR `tonemap`), and one-command multi-rendition CMAF HLS & DASH ABR packaging (`ragdoll package`).

## Release gate

`ragdoll cut`, `ragdoll concat`, `ragdoll package`, `.png`/`.jpg` thumbnail/contact-sheet extraction, and fused spatial/HDR filters execute end-to-end with zero temporary files and verified player compatibility (`ffplay`/HLS/DASH).

## 2026-10-25

Completed Milestone 9 (m9-edit-image-package): 0046, 0047, SIMD-0048, and 0049.
