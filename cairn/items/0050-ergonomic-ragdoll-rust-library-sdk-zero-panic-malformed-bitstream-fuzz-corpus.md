---
id: 50
uid: 5edc67ba-080b-471e-8407-0d0e24cff444
title: Ergonomic ragdoll Rust Library SDK & Zero-Panic Malformed Bitstream Fuzz Corpus
type: feature
status: done
milestone: m10-production-v1
assignee: Oddur Sigurdsson
depends_on:
- 40
- 44
- 46
- 49
created: 2026-10-25
updated: 2026-10-25
closed_at: 2026-10-25
priority: p0
area: arch
effort: l
---

## Problem

1. Rust developers currently have to choose between unsafe C bindings (`ffmpeg-next`) or spawning `ffmpeg` child processes. Ragdoll should offer a clean, high-level Rust builder SDK (`ragdoll_pipeline::TranscodeJob`, `ProbeJob`, `CutJob`, `PackageJob`) with zero `unsafe` required by callers.
2. Production media servers ingest untrusted user uploads; no truncated, corrupted, or adversarial container/bitstream byte sequence may ever trigger a panic (`index out of bounds`, `unwrap()`, divide-by-zero, or OOM allocation).

## Architecture & SIMD Design

1. Audit all `.unwrap()` / slice indexing paths in `ragdoll-format` and `ragdoll-codec`, replacing any fallible input assumptions with checked bounds and `RagdollError::{CorruptBitstream, UnexpectedEof, Unsupported}` plus sanity caps on untrusted container box/allocation sizes.
2. Build a deterministic in-tree **Malformed Bitstream & Mutation Fuzz Harness** (`tests::malformed_bitstream_zero_panic_corpus`) that subjects `Mp4Demuxer`, `MkvDemuxer`, `MpegTsDemuxer`, `Y4mDemuxer`, `WavDemuxer`, `H264Decoder`, `FlacDecoder`, `AacDecoder`, `OpusDecoder`, `Av1Parser`, and `Png/Jpeg` readers to bit-flips, random truncations, zeroed headers, and oversized length fields.
3. Expose a clean, documented Rust SDK in `ragdoll-pipeline` (and workspace docs) for programmatic probing, transcoding, smart-cutting, frame extraction, and ABR packaging.

## Acceptance criteria

- [x] 10,000+ deterministic bit-flip, truncation, and corrupted-length mutations across all container demuxers and codec decoders complete with `0` panics.
- [x] Ergonomic Rust builder API (`TranscodeBuilder`, `CutBuilder`, `PackageBuilder`) is documented and tested in `ragdoll-pipeline`.

## 2026-10-25

Implemented ProbeBuilder/TranscodeBuilder/CutBuilder/PackageBuilder (and ProbeJob/TranscodeJob/CutJob/PackageJob) in ragdoll-pipeline, hardened untrusted length/dimension/overflow checks across ragdoll-format and ragdoll-codec, and verified tests::malformed_bitstream_zero_panic_corpus across 10,000+ deterministic mutations with 0 panics.
