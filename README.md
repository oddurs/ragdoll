<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/wordmark-dark-bg.svg">
    <img src="assets/brand/wordmark-light-bg.svg" alt="ragdoll" height="64">
  </picture>
</p>

<p align="center">
  <strong>Media, from first principles.</strong><br>
  Containers, codecs, scaling, color and quality metrics in pure Rust, with hand-written SIMD and no C libraries.
</p>

<p align="center">
  <a href="https://github.com/oddurs/ragdoll/actions/workflows/ci.yml"><img src="https://github.com/oddurs/ragdoll/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-0E0F12" alt="License: MIT OR Apache-2.0"></a>
  <img src="https://img.shields.io/badge/status-pre--release-FF4F2E" alt="Status: pre-release">
</p>

<p align="center">
  <a href="https://oddurs.github.io/ragdoll/docs/">Documentation</a> ·
  <a href="https://oddurs.github.io/ragdoll/docs/quickstart/">Quickstart</a> ·
  <a href="https://oddurs.github.io/ragdoll/docs/formats/">Formats &amp; codecs</a> ·
  <a href="https://oddurs.github.io/ragdoll/docs/benchmarks/">Benchmarks</a> ·
  <a href="ROADMAP.md">Roadmap</a>
</p>

---

Ragdoll is a media framework and command-line tool written entirely in Rust. It reads and writes containers, decodes and encodes video and audio, converts color, scales images and measures quality. Every component is built from the bitstream up:

- **No C libraries and no third-party crates.** `Cargo.lock` lists seven packages, all of them in this repository.
- **Explicit SIMD, verified.** Hot loops are hand-written NEON intrinsics. Each kernel has a scalar reference, and the tests require the two to match bit for bit.
- **Zero-copy by default.** Demuxers return packets that borrow from a memory-mapped file; muxers write with vectored I/O; frames come from aligned, pooled buffers.
- **A CLI with verbs.** `probe`, `remux`, `transcode`, `extract`, `metrics`. Named options, JSON output, Unix pipes, and TOML recipes.

> [!NOTE]
> Ragdoll is **pre-1.0**. Milestones M1–M6 are complete and production H.264 (M7) is in progress. The H.264 encoder is Constrained Baseline and does not yet compress competitively, `x86_64` runs scalar code paths because AVX2 kernels have not been written yet, and APIs will change. See [Known gaps](https://oddurs.github.io/ragdoll/docs/roadmap/#known-gaps).

## Install

Requires Rust 1.85 or newer. There are no system dependencies.

```sh
cargo install --git https://github.com/oddurs/ragdoll ragdoll-cli
ragdoll --version
```

Or build from a clone and check every SIMD kernel against its reference on your machine:

```sh
git clone https://github.com/oddurs/ragdoll.git && cd ragdoll
cargo build --release
cargo run --release -- self-test
```

## Usage

```sh
ragdoll synth sample.mp4 --resolution 1280x720 --duration 2   # generate test media
ragdoll probe sample.mp4 --gop-map                             # inspect, no decoding
ragdoll remux sample.mp4 sample.mkv                            # change container
ragdoll transcode sample.mp4 small.mp4 --scale 640x360         # scale and re-encode
ragdoll transcode master.wav master.flac                       # lossless audio
ragdoll extract sample.mp4 -o sheet.ppm --thumbnails 16        # contact sheet
ragdoll metrics sample.mp4 small.mp4                           # PSNR, SSIM, MS-SSIM
ragdoll transcode in.mp4 - --format ts | ragdoll probe -       # pipes
```

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ RAGDOLL PROBE : sample.mp4                                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│ Container     : mp4/isobmff        File Size :      7844139 B (7.48 MB)      │
│ Probe Latency :    42.75 µs         GOP Count :           10                 │
│ H.264 Profile : Constrained Baseline (level 4.0)                             │
├──────────────────────────────────────────────────────────────────────────────┤
│ Track #0  [Video] codec=h264 1280x720 @ 30.000 fps (30/1) fmt=yuv420p        │
│   Colorimetry : primaries=BT709, transfer=BT709, matrix=BT709, range=Limited
│   Samples     : 60 packets (7842908 bytes)
└──────────────────────────────────────────────────────────────────────────────┘
```

The [CLI reference](https://oddurs.github.io/ragdoll/docs/cli/) covers every subcommand and option.

## What it supports

| | Read / decode | Write / encode |
| :-- | :-- | :-- |
| **Containers** | MP4 / MOV, Matroska, WebM, MPEG-TS (188/192/204), Y4M, WAV / RF64, FLAC, Netpbm, WebVTT, SRT | MP4 (fast-start and fragmented), Matroska, MPEG-TS, Y4M, WAV, FLAC, Netpbm, WebVTT, SRT |
| **Video** | H.264 Constrained Baseline, RGDI | H.264 Constrained Baseline, RGDI |
| **Audio** | FLAC, PCM (u8 / s16 / s24 / s32 / f32) | FLAC, PCM |
| **Pass-through** | HEVC, VP9, AV1, AAC, Opus are carried through containers without decoding | |
| **Pixels** | YUV 4:2:0 / 4:2:2 / 4:4:4, NV12, 10-bit, RGB24, RGBA, BGRA, Oklab; BT.601 / 709 / 2020; Lanczos-3, Spline36, Mitchell, bilinear scaling | |
| **Metrics** | PSNR, SSIM, MS-SSIM, VMAF-Lite | |

The [full matrix](https://oddurs.github.io/ragdoll/docs/formats/) lists every limitation.

## Workspace

| Crate | Role |
| :-- | :-- |
| [`ragdoll-core`](crates/ragdoll-core) | 128-bit timebases, CICP colorimetry, aligned frames and pools, packets, bit I/O, CRCs, power modes |
| [`ragdoll-simd`](crates/ragdoll-simd) | NEON kernels with scalar twins: start-code scanning, SAD / SSE / SATD / SSIM, integer transforms |
| [`ragdoll-pix`](crates/ragdoll-pix) | Color conversion, polyphase scaling, dithering, quality metrics |
| [`ragdoll-format`](crates/ragdoll-format) | Memory-mapped demuxers and vectored muxers |
| [`ragdoll-codec`](crates/ragdoll-codec) | H.264, FLAC, PCM, RGDI; AV1 OBU inspection |
| [`ragdoll-pipeline`](crates/ragdoll-pipeline) | GOP-parallel scheduling, bounded-memory encoding, A/V sync, TOML recipes |
| [`ragdoll-cli`](crates/ragdoll-cli) | The `ragdoll` binary |

Read the [architecture overview](https://oddurs.github.io/ragdoll/docs/architecture/) for how they fit together.

## Performance

Selected single-threaded kernels on an Apple M4, against FFmpeg 9.0.2 built with `clang -O3 -mcpu=native`:

| Kernel | FFmpeg 9.0.2 | Ragdoll | |
| :-- | --: | --: | --: |
| Annex-B start-code scan | 7.21 GB/s | 32.61 GB/s | 4.52× |
| 4K → 1080p Lanczos-3 | 214.8 fps | 506.5 fps | 2.36× |
| H.264 RBSP unescape | 5.14 GB/s | 10.65 GB/s | 2.07× |
| 4K plane SSIM | 5.04 Gpx/s | 10.06 Gpx/s | 2.00× |
| H.264 4×4 DCT + IDCT | 125.5 Mblk/s | 191.0 Mblk/s | 1.52× |

These figures come from one machine. The Ragdoll column is reproducible with `ragdoll self-test`; the FFmpeg column was measured out of tree and its harness is not yet in the repository. End-to-end comparisons run with `ragdoll bench --compare-ffmpeg`. Read the [methodology and caveats](https://oddurs.github.io/ragdoll/docs/benchmarks/) before quoting any of them.

## Roadmap

| Milestone | Status | |
| :-- | :-- | :-- |
| M1 Foundation · M2 Pixels · M3 Containers · M4 Codecs · M5 CLI & bench · M6 Streaming & sync | Done | |
| M7 Production video | In progress | Full H.264 residuals, quarter-pel, deblocking, CABAC, B-frames, CRF |
| M8 Professional audio | Planned | AAC-LC, Opus, resampling, EBU R128 |
| M9 Edit & package | Planned | Smart-GOP cut / concat, PNG / JPEG, filters, HLS / DASH |
| M10 1.0 | Planned | Stable SDK, fuzzing, release benchmarks |

Work is tracked as Markdown files in [`cairn/items`](cairn/items), and [`ROADMAP.md`](ROADMAP.md) is generated from them.

## Contributing

Contributions are welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) first. The short version:

- No new dependencies and no C. Write it in Rust, in the crate where it belongs.
- Every optimized kernel needs a scalar reference and a differential test.
- `cargo fmt`, `cargo clippy -D warnings`, `cargo test --workspace` and `cairn check --render --strict` must pass.

AVX2 ports of the existing NEON kernels are a good place to start: each one already has a reference and tests to match.

The website and docs live in [`site/`](site) (Next.js, static export): `cd site && npm install && npm run dev`.

## License

Dual-licensed under either of

- MIT License ([`LICENSE-MIT`](LICENSE-MIT))
- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE))

at your option. Unless you explicitly state otherwise, any contribution you submit for inclusion is dual-licensed as above, without additional terms or conditions.
