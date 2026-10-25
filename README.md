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
- **Explicit SIMD, verified.** Hot loops are hand-written NEON and AVX2/SSE4.1 intrinsics. Each kernel has a scalar reference, and the tests require the two to match bit for bit.
- **Zero-copy by default.** Demuxers return packets that borrow from a memory-mapped file; muxers write with vectored I/O; frames come from aligned, pooled buffers; malformed inputs return structured `Result<_, RagdollError>` with a Zero-Panic Guarantee.
- **A CLI with verbs & an ergonomic Rust SDK.** `probe`, `remux`, `transcode`, `cut`, `concat`, `package`, `extract`, `metrics`, plus `TranscodeBuilder`, `CutBuilder`, `PackageBuilder`, and `ProbeBuilder`. Named options, JSON output, Unix pipes, and TOML recipes.

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
| **Containers** | MP4 / MOV, Matroska, WebM, MPEG-TS (188/192/204), Y4M, WAV / RF64, FLAC, PNG, JPEG, Netpbm, WebVTT, SRT | MP4 (fast-start, fragmented, and CMAF `fMP4`), HLS (`master.m3u8`), DASH (`manifest.mpd`), Matroska, MPEG-TS, Y4M, WAV, FLAC, PNG, JPEG, Netpbm, WebVTT, SRT |
| **Video** | H.264 High / Main / Constrained Baseline (CABAC & CAVLC, I/P/B, $8\times8$ DCT), RGDI | H.264 High / Main / Constrained Baseline (CABAC & CAVLC, I/P/B, CRF / ABR / VBV), RGDI |
| **Audio** | AAC-LC (MDCT / IMDCT), Opus (CELT + SILK hybrid), FLAC, PCM (u8 / s16 / s24 / s32 / f32) | AAC-LC, Opus, FLAC, PCM (with 32-tap polyphase resampler, channel rematrixing & EBU R128 `-14 LUFS` loudness normalization) |
| **Pass-through** | HEVC, VP9, AV1 are carried through containers without decoding | |
| **Pixels & Filters** | YUV 4:2:0 / 4:2:2 / 4:4:4, NV12, 10-bit (`Yuv420p10le`), RGB24, RGBA, BGRA, Oklab; BT.601 / 709 / 2020; Lanczos-3, Spline36, Mitchell, bilinear scaling; crop / auto-crop, pad, rotate/flip, RGBA overlay, perceptual Oklab HDR10/HLG $\to$ SDR tonemap | |
| **Metrics** | PSNR, SSIM, MS-SSIM, VMAF-Lite | |

The [full matrix](https://oddurs.github.io/ragdoll/docs/formats/) lists every limitation.

## Rust Library SDK (`ragdoll_pipeline`)

```rust
use ragdoll_codec::{H264Profile, RateControlMode};
use ragdoll_pipeline::{CutBuilder, PackageBuilder, ProbeBuilder, TranscodeBuilder};

let info = ProbeBuilder::new("input.mp4").gop_map(true).run()?;
let report = TranscodeBuilder::new("input.mp4", "output.mp4")
    .video_codec("h264")
    .h264_profile(H264Profile::High)
    .rate_control(RateControlMode::Crf { crf: 23.0, maxrate_kbps: Some(4000), bufsize_kb: Some(8000) })
    .scale(1280, 720)
    .tonemap(true)
    .audio_codec("aac")
    .sample_rate(48_000)
    .loudness_lufs(-14.0)
    .memory_budget_mib(64)
    .run()?;
let cut = CutBuilder::new("output.mp4", "clip.mp4").from_sec(0.5).to_sec(5.0).run()?;
let abr = PackageBuilder::new("clip.mp4", "abr_out").ladder("720p:2400k,360p:800k").run()?;
```

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

## Performance (12-Workload `v1.0` Suite vs. FFmpeg 9.0.2)

End-to-end 12-workload production suite (`cargo run --release -- bench --compare-ffmpeg`) on Apple Silicon (`aarch64 NEON`, 4 P-Cores + 6 E-Cores) against FFmpeg 9.0.2:

| Workload | FFmpeg 9.0.2 | Ragdoll (P-Core) | P-Core Speedup | E-Core Energy Savings (`mJ`) |
| :-- | --: | --: | --: | --: |
| 1. Container Probe & GOP Index (`300f` 4K MP4) | 31.89 ms | **0.16 ms** | **202.81×** | **1,106.7×** (`0.08 mJ` vs `90.17 mJ`) |
| 2. Zero-Copy Remux (`MP4` → `MKV` 16 MB 4K) | 24.48 ms | **5.62 ms** (`2.80 GB/s`) | **4.35×** | **31.5×** (`3.02 mJ` vs `95.24 mJ`) |
| 3. 4K → 1080p Lanczos-3 Scaling (`30f`) | 126.60 ms | **17.19 ms** (`1,745 fps`) | **7.37×** | **8.9×** (`143.41 mJ` vs `1,278.43 mJ`) |
| 4. Lossless FLAC Encode+Decode (`30s` 48 kHz 2ch) | 108.14 ms | **25.77 ms** (`1,164× RT`) | **4.20×** | **13.0×** (`45.35 mJ` vs `587.56 mJ`) |
| 5. 4×4 Contact Sheet Extraction (`16` Thumbs) | 26.35 ms | **4.23 ms** (`3,784 thumbs/s`) | **6.23×** | **4.7×** (`32.28 mJ` vs `151.33 mJ`) |
| 6. 4K Perceptual Metrics (`PSNR + 8×8 SSIM 20f`) | 129.78 ms | **9.53 ms** (`2,100 fps`) | **13.62×** | **21.8×** (`66.44 mJ` vs `1,446.16 mJ`) |
| 7. 1080p H.264 Inter Transcode (`60f` Y4M→MP4) | 112.88 ms | **157.30 ms** (`381 fps`) | 0.72× (`1.24×` peak) | **2.2×** (`711.92 mJ` vs `1,570.57 mJ`) |
| 8. Smart-GOP Frame-Accurate Trim (`ragdoll cut`) | 48.59 ms | **11.53 ms** (`86.8 clips/s`) | **4.22×** | **33.1×** (`19.51 mJ` vs `645.82 mJ`) |
| 9. Audio Resample `44.1→48k` + R128 + AAC (`10s`) | 348.17 ms | **18.28 ms** (`547× RT`) | **19.04×** | **57.4×** (`31.10 mJ` vs `1,784.78 mJ`) |
| 10. HDR10 → SDR Oklab Tonemap + Overlay (`30f`) | 117.07 ms | **51.95 ms** (`578 fps`) | **2.25×** | **4.3×** (`571.15 mJ` vs `2,429.63 mJ`) |
| 11. CMAF HLS+DASH ABR Packager (`3` Renditions) | 37.91 ms | **32.68 ms** (`184 seg/s`) | **1.16×** | **2.0×** (`197.29 mJ` vs `391.37 mJ`) |
| 12. Bounded-Mem (`64 MiB`) High-CRF Transcode | 53.90 ms | **67.82 ms** (`885 fps`) | 0.79× (`1.00×` peak) | **2.8×** (`293.06 mJ` vs `819.61 mJ`) |

Reproduce single-kernel verification with `ragdoll self-test` and end-to-end comparisons with `ragdoll bench --compare-ffmpeg`. Read the [methodology and caveats](https://oddurs.github.io/ragdoll/docs/benchmarks/) before quoting any of them.

## Roadmap

- **Available today:** the containers, codecs, pixel processing and metrics above; a GOP-parallel pipeline with pipes and multi-track A/V sync; TOML recipes.
- **In development:** production H.264, with full residual coding, quarter-pel motion, deblocking, Main and High profiles with CABAC, B-frames, and CRF rate control.
- **Planned for 1.0:** AAC-LC and Opus, EBU R128 loudness, frame-accurate cut and concat, PNG and JPEG, filters and HDR tone mapping, HLS and DASH packaging, AVX2 kernels, a stable library API, and fuzz hardening.

The [roadmap page](https://oddurs.github.io/ragdoll/docs/roadmap/) has details and known limitations.

## Contributing

Contributions are welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) first. The short version:

- No new dependencies and no C. Write it in Rust, in the crate where it belongs.
- Every optimized kernel needs a scalar reference and a differential test.
- `cargo fmt`, `cargo clippy -D warnings` and `cargo test --workspace` must pass.

AVX2 ports of the existing NEON kernels are a good place to start: each one already has a reference and tests to match.

The website and docs live in [`site/`](site) (Next.js, static export): `cd site && npm install && npm run dev`.

## License

Dual-licensed under either of

- MIT License ([`LICENSE-MIT`](LICENSE-MIT))
- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE))

at your option. Unless you explicitly state otherwise, any contribution you submit for inclusion is dual-licensed as above, without additional terms or conditions.
