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
  <a href="https://oddurs.github.io/ragdoll/docs/roadmap/">Roadmap</a>
</p>

---

Ragdoll is a media framework and command-line tool written entirely in Rust. It reads and writes containers, decodes and encodes video and audio, converts color, scales images and measures quality. Every component is built from the bitstream up:

- **No C libraries and no third-party crates.** `Cargo.lock` lists seven packages, all of them in this repository.
- **Explicit SIMD, verified.** Hot loops are hand-written NEON intrinsics on `aarch64`, with AVX2 versions of the hottest on `x86_64`. Each kernel has a scalar reference, and the tests require the two to match bit for bit.
- **Correct before fast.** The H.264 decoder is bit-exact with the JM reference and FFmpeg, and the Opus decoder sample-exact with libopus. Most codecs are still slower than FFmpeg's; the [benchmarks](https://oddurs.github.io/ragdoll/docs/benchmarks/) say by how much.
- **Zero-copy by default.** Demuxers return packets that borrow from a memory-mapped file; muxers write with vectored I/O; frames come from aligned, pooled buffers; errors come back as `Result<_, RagdollError>`.
- **A CLI with verbs and a Rust library.** `probe`, `remux`, `transcode`, `cut`, `concat`, `package`, `extract`, `metrics`, plus `ProbeBuilder`, `TranscodeBuilder`, `CutBuilder` and `PackageBuilder`. Named options, JSON output, Unix pipes, and TOML recipes.

## Install

Requires Rust 1.88 or newer. There are no system dependencies.

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
ragdoll transcode sample.mp4 q30.mp4 --qp 30                   # re-encode at a fixed QP
ragdoll transcode master.wav master.flac                       # lossless audio
ragdoll extract sample.mp4 -o sheet.ppm --thumbnails 16        # contact sheet
ragdoll metrics sample.mp4 q30.mp4                             # PSNR, SSIM, MS-SSIM
ragdoll transcode in.mp4 - --format ts | ragdoll probe -       # pipes
```

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ RAGDOLL PROBE : sample.mp4                                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│ Container     : mp4/isobmff        File Size :       137757 B (0.13 MB)      │
│ Probe Latency :   223.93 µs         GOP Count :            4                 │
│ H.264 Profile : Constrained Baseline (level 4.0)                             │
├──────────────────────────────────────────────────────────────────────────────┤
│ Track #0  [Video] codec=h264 1280x720 @ 30.000 fps (30/1) fmt=yuv420p        │
│   Colorimetry : primaries=BT709, transfer=BT709, matrix=BT709, range=Limited
│   Samples     : 60 packets (136550 bytes)
├──────────────────────────────────────────────────────────────────────────────┤
│ Bitrate Spark : ▆▂▁▁▂▁▁▁▁▁▁▁▁▁▆▂▁▁▂▁▁▁▁▁▁▁▂▁▇▃▂▁▁▁▂▁▁▂▁▁▁▁█▂▂▂▁▁▁▁▁▁▂▁▁▁     │
└──────────────────────────────────────────────────────────────────────────────┘
```

The [CLI reference](https://oddurs.github.io/ragdoll/docs/cli/) covers every subcommand and option.

## What it supports

| | Read / decode | Write / encode |
| :-- | :-- | :-- |
| **Containers** | MP4 / MOV, Matroska, WebM, MPEG-TS (188/192/204), Ogg, Y4M, WAV / RF64, FLAC, PNG, JPEG, Netpbm, WebVTT, SRT | MP4 (fast-start, fragmented, and CMAF `fMP4`), HLS (`master.m3u8`), DASH (`manifest.mpd`), Matroska, MPEG-TS, Ogg, Y4M, WAV, FLAC, PNG, JPEG, Netpbm, WebVTT, SRT |
| **Video** | H.264 High / Main / Baseline, bit-exact with the reference decoders (CABAC & CAVLC, I/P/B, $8\times8$ transform, weighted prediction, interlaced PAFF / MBAFF, 8-bit 4:2:0 and 4:0:0), RGDI | H.264 High / Main / Constrained Baseline (CABAC & CAVLC, I/P frames, fixed QP / CRF / ABR / VBV), RGDI |
| **Audio** | AAC-LC (mono to 5.1), Opus (CELT, SILK and hybrid; sample-exact with libopus), FLAC, PCM (u8 / s16 / s24 / s32 / f32) | AAC-LC (no rate control yet), Opus (CELT-only), FLAC, PCM; with polyphase resampling, channel remixing and EBU R128 loudness normalization |
| **Pass-through** | HEVC, VP9, AV1 are carried through containers without decoding; AV1 OBU / IVF streams can be inspected | |
| **Pixels & Filters** | YUV 4:2:0 / 4:2:2 / 4:4:4, NV12, 10-bit (`Yuv420p10le`), RGB24, RGBA, BGRA, Oklab; BT.601 / 709 / 2020; Lanczos-3, Spline36, Mitchell, bilinear scaling; crop / auto-crop, pad, rotate/flip, RGBA overlay, perceptual Oklab HDR10/HLG $\to$ SDR tonemap | |
| **Metrics** | PSNR, SSIM, MS-SSIM, VMAF-Lite | |

The [full matrix](https://oddurs.github.io/ragdoll/docs/formats/) lists every limitation. Video is decoded, filtered and encoded in batches bounded by a memory budget; audio that is re-encoded or processed is still decoded whole into memory.

## Rust Library SDK (`ragdoll_pipeline`)

```rust
use ragdoll_pipeline::{
    CutBuilder, H264Profile, PackageBuilder, ProbeBuilder, ScaleFilter, TranscodeBuilder,
};

let info = ProbeBuilder::new("input.mp4").run()?;
let report = TranscodeBuilder::new("input.mp4", "output.mp4")
    .video_codec("h264")
    .profile(H264Profile::High)
    .crf(23.0)
    .vbv_constraints(4000, 8000)
    .scale(1280, 720, ScaleFilter::Lanczos3)
    .tonemap(true)
    .audio_codec("aac")
    .audio_sample_rate(48_000)
    .normalize_lufs(-14.0)
    .memory_budget_bytes(64 << 20)
    .run()?;
let cut = CutBuilder::new("output.mp4", "clip.mp4").from_seconds(0.5).to_seconds(5.0).run()?;
let abr = PackageBuilder::new("clip.mp4", "abr_out").ladder("720p:2400k,360p:800k").run()?;
```

## Workspace

| Crate | Role |
| :-- | :-- |
| [`ragdoll-core`](crates/ragdoll-core) | 128-bit timebases, CICP colorimetry, aligned frames and pools, packets, bit I/O, CRCs, power modes |
| [`ragdoll-simd`](crates/ragdoll-simd) | NEON and AVX2 kernels with scalar twins: start-code scanning, SAD / SSE / SATD / SSIM, integer transforms |
| [`ragdoll-pix`](crates/ragdoll-pix) | Color conversion, polyphase scaling, dithering, quality metrics |
| [`ragdoll-format`](crates/ragdoll-format) | Memory-mapped demuxers and vectored muxers |
| [`ragdoll-codec`](crates/ragdoll-codec) | H.264, AAC-LC, Opus, FLAC, PCM, RGDI; resampling and loudness; AV1 OBU inspection |
| [`ragdoll-pipeline`](crates/ragdoll-pipeline) | GOP-parallel scheduling, bounded-memory encoding, A/V sync, TOML recipes |
| [`ragdoll-cli`](crates/ragdoll-cli) | The `ragdoll` binary |

Read the [architecture overview](https://oddurs.github.io/ragdoll/docs/architecture/) for how they fit together.

## Performance

Ragdoll is correct first and not yet fast everywhere. Single-threaded, in-process, codec calls only, against libavcodec 6.1.1, x264 and libopus on one x86_64 AVX2 machine (about ±15% noise):

| Workload | Ragdoll | Reference | Ratio |
| :-- | --: | --: | --: |
| H.264 decode, 1080p High / CABAC | 29 fps | FFmpeg 181 fps | 0.16× |
| H.264 decode, 720p Baseline | 77 fps | FFmpeg 834 fps | 0.09× |
| H.264 encode, 720p QP 26 (similar size and PSNR to `ultrafast`) | 59 fps | x264 `ultrafast` 489 fps | 0.12× |
| AAC-LC decode | 176× real time | FFmpeg 2367× | 0.07× |
| Opus decode / encode | — | libopus | about 0.8× / 1× |
| FLAC encode (files ~11% larger) / decode | — | FFmpeg | 1.5× / 0.4× |

`ragdoll self-test` checks every SIMD kernel against its scalar reference and shows the gain on your machine. `ragdoll bench` times twelve end-to-end workloads in-process; its `--compare-ffmpeg` column runs FFmpeg as a separate process, so its speedups mostly measure FFmpeg's start-up and are not a codec comparison, and its energy table is an estimate, not a measurement. Read the [methodology and caveats](https://oddurs.github.io/ragdoll/docs/benchmarks/) before quoting any of these.

## Roadmap

- **Available today:** the containers, codecs, filters and metrics above; frame-accurate cut and concat; HLS and DASH packaging; a GOP-parallel pipeline with pipes, multi-track A/V sync and bounded-memory video; TOML recipes; the Rust builders.
- **In development:** decoder speed, starting with SIMD macroblock reconstruction for H.264, then AAC-LC, FLAC and Opus.
- **Planned:** conformance and interop test runners, coverage-guided fuzzing, performance and memory regression gates, AAC rate control, bounded-memory audio, multithreaded H.264 decoding, and Python bindings.

The [roadmap page](https://oddurs.github.io/ragdoll/docs/roadmap/) has details and known limitations.

## Contributing

Contributions are welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) first. The short version:

- No new dependencies and no C. Write it in Rust, in the crate where it belongs.
- Every optimized kernel needs a scalar reference and a differential test.
- `cargo fmt`, `cargo clippy -D warnings` and `cargo test --workspace` must pass.

AVX2 ports of the NEON kernels that still run their scalar reference on `x86_64` are a good place to start: each one already has a reference and tests to match.

The website and docs live in [`site/`](site) (Next.js, static export): `cd site && npm install && npm run dev`.

## License

Dual-licensed under either of

- MIT License ([`LICENSE-MIT`](LICENSE-MIT))
- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE))

at your option. Unless you explicitly state otherwise, any contribution you submit for inclusion is dual-licensed as above, without additional terms or conditions.
