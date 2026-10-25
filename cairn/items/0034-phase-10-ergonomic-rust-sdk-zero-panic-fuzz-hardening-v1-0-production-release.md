---
id: 34
uid: 70b9ea07-3206-4cb2-941a-8ad596b90873
key: m10-production-v1
title: Phase 10 — Ergonomic Rust SDK, Zero-Panic Fuzz Hardening & v1.0 Production Release
type: milestone
status: planned
created: 2026-10-25
updated: 2026-10-25
priority: p0
area: arch
---

Seal Ragdoll `v1.0` as a production replacement for FFmpeg: expose an ergonomic, zero-unsafe-required Rust library SDK (`ragdoll` crate / builder API), enforce a Zero-Panic Malformed Bitstream Guarantee across all container demuxers and codec decoders via deterministic mutation/fuzz corpus tests, and validate speed, perceptual quality, and E-core battery efficiency across a 12-workload benchmark suite against FFmpeg 9.0.2.

## Release gate

All 12 flagship workloads in `ragdoll bench --compare-ffmpeg`, the malformed-bitstream fuzz corpus, and the public Rust SDK examples pass with zero warnings, zero panics on corrupted inputs, and documented speed + battery-efficiency wins over FFmpeg 9.0.2.
