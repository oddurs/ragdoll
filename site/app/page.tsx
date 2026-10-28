import Link from "next/link";
import { site } from "@/lib/site";

const probeOutput = `┌──────────────────────────────────────────────────────────┐
│ RAGDOLL PROBE : sample.mp4                               │
├──────────────────────────────────────────────────────────┤
│ Container     : mp4/isobmff      File Size : 0.13 MB     │
│ Probe Latency : 223.93 µs        GOP Count : 4           │
│ H.264 Profile : Constrained Baseline (level 4.0)         │
├──────────────────────────────────────────────────────────┤
│ Track #0 [Video] h264 1280x720 @ 30.000 fps  yuv420p     │
│   Colorimetry : BT709 / BT709 / BT709, Limited           │
│   Samples     : 60 packets (136550 bytes)                │
├──────────────────────────────────────────────────────────┤
│ Bitrate Spark : ▆▂▁▁▂▁▁▁▁▁▁▁▁▁▆▂▁▁▂▁▁▁▁▁▁▁▂▁▇▃▂▁▁▁▂▁▁▂▁▁ │
└──────────────────────────────────────────────────────────┘`;

const principles = [
  {
    title: "No C, no third-party crates",
    body: "Every parser, transform, entropy coder and muxer is written in Rust inside this workspace. The dependency graph is seven crates deep and ends there.",
    code: "Cargo.lock → 7 packages, all ragdoll-*",
  },
  {
    title: "Explicit SIMD, proven against scalar",
    body: "Hot loops are hand-written NEON intrinsics behind safe slice APIs, with AVX2 versions of the hottest on x86_64. Each one ships with a scalar reference and a differential test that must match bit for bit.",
    code: "sad_plane_u8 ≡ sad_plane_u8_scalar",
  },
  {
    title: "Zero-copy by default",
    body: "Demuxers hand out packets that borrow straight from a memory-mapped file. Frames live in 64-byte-aligned pooled buffers with strides chosen to avoid cache-set aliasing.",
    code: "PacketSlice<'mmap>",
  },
  {
    title: "A command line with verbs",
    body: "probe, remux, transcode, cut, concat, package, extract, metrics. Each task is a subcommand with named options, JSON output, and a declarative TOML form for repeatable pipelines.",
    code: "ragdoll transcode in.mp4 out.mkv --scale 1280x720",
  },
];

const stages = [
  { k: "Read", crate: "ragdoll-format", body: "mmap demux of MP4, Matroska/WebM, MPEG-TS, Ogg, Y4M, WAV, FLAC, PNG, JPEG, WebVTT and SRT." },
  { k: "Decode", crate: "ragdoll-codec", body: "H.264, AAC-LC, Opus, FLAC, PCM and the RGDI intra codec. AV1 OBU inspection." },
  { k: "Transform", crate: "ragdoll-pix", body: "Color matrices, polyphase scaling, filters, tone mapping, PSNR / SSIM / MS-SSIM." },
  { k: "Encode", crate: "ragdoll-codec", body: "GOP-parallel H.264 and RGDI encode; AAC-LC, Opus, FLAC and PCM." },
  { k: "Write", crate: "ragdoll-format", body: "Vectored writev muxers: fast-start and fragmented MP4, Cues-indexed Matroska, MPEG-TS, Ogg." },
];

const commands = [
  { name: "probe", href: "/docs/cli/probe/", body: "Inspect containers, tracks, colorimetry and GOP structure in microseconds." },
  { name: "remux", href: "/docs/cli/remux/", body: "Move streams between MP4, Matroska and MPEG-TS without touching a pixel." },
  { name: "transcode", href: "/docs/cli/transcode/", body: "Scale and re-encode video across GOP-parallel workers, keeping audio and subtitles in sync." },
  { name: "extract", href: "/docs/cli/extract/", body: "Pull a frame at a timestamp or tile a contact sheet of thumbnails." },
  { name: "metrics", href: "/docs/cli/metrics/", body: "Score a distorted file against a reference: PSNR, SSIM, MS-SSIM, VMAF-Lite." },
  { name: "bench", href: "/docs/cli/bench/", body: "Time twelve end-to-end workloads in-process on this machine." },
];

const bench = [
  { name: "H.264 decode, 1080p High (CABAC)", ours: "29 fps", theirs: "FFmpeg 181 fps", x: 0.16 },
  { name: "H.264 decode, 720p Baseline", ours: "77 fps", theirs: "FFmpeg 834 fps", x: 0.09 },
  { name: "H.264 encode, 720p QP 26", ours: "59 fps", theirs: "x264 ultrafast 489 fps", x: 0.12 },
  { name: "AAC-LC decode", ours: "176× real time", theirs: "FFmpeg 2367×", x: 0.07 },
];

const status = [
  {
    s: "done",
    title: "Available today",
    items: [
      "MP4, Matroska / WebM, MPEG-TS, Ogg, WAV, PNG, JPEG",
      "H.264 Baseline / Main / High decode and encode",
      "AAC-LC, Opus, FLAC, PCM; EBU R128 loudness",
      "Cut, concat, HLS and DASH packaging",
      "Filters, HDR tone mapping, quality metrics",
    ],
  },
  {
    s: "doing",
    title: "In development",
    items: [
      "H.264 decoder speed: SIMD reconstruction",
      "Partition-sized motion compensation",
      "AAC-LC decoder speed",
      "FLAC and Opus decoder speed",
    ],
  },
  {
    s: "planned",
    title: "Planned",
    items: [
      "Conformance and interop runners",
      "Coverage-guided fuzzing",
      "Performance and memory regression gates",
      "AAC rate control, bounded-memory audio",
      "Multithreaded H.264 decoding",
    ],
  },
];

export default function Home() {
  return (
    <main>
      <section className="hero">
        <div className="wrap hero__inner">
          <div>
            <p className="eyebrow">Pre-release · v{site.version} · MIT / Apache-2.0</p>
            <h1>Media, from first&nbsp;principles.</h1>
            <p className="hero__lede">
              Ragdoll is a media framework and command-line tool written entirely in Rust. Containers,
              codecs, scaling, color and quality metrics are built from the bitstream up, with
              hand-written SIMD in the hot loops and no C libraries anywhere.
            </p>
            <div className="hero__actions">
              <Link className="button button--primary" href="/docs/quickstart/">
                Get started
              </Link>
              <Link className="button" href="/docs/architecture/">
                Read the architecture
              </Link>
            </div>
            <div className="install" aria-label="Install command">
              <span>$</span>
              cargo install --git {site.repo} ragdoll-cli
            </div>
          </div>
          <div className="terminal" aria-label="Example output of ragdoll probe">
            <div className="terminal__bar">
              <i />
              <i />
              <i />
              <b>~/media</b>
            </div>
            <pre>
              <span className="p">$ </span>
              <span className="c">ragdoll probe sample.mp4</span>
              {"\n"}
              <span className="d">{probeOutput}</span>
              {"\n\n"}
              <span className="p">$ </span>
              <span className="c">ragdoll remux sample.mp4 sample.mkv</span>
              {"\n"}
              <span className="k">[ragdoll remux]</span>
              <span className="d"> sample.mp4 -&gt; sample.mkv | frames=60 | 0.32 ms </span>
              <span className="g">(0.865 GB/s)</span>
            </pre>
          </div>
        </div>
      </section>

      <div className="wrap">
        <div className="facts">
          <div className="fact">
            <strong>0</strong>
            <span>C libraries linked</span>
          </div>
          <div className="fact">
            <strong>0</strong>
            <span>third-party crates</span>
          </div>
          <div className="fact">
            <strong>3</strong>
            <span>SIMD levels: scalar, NEON, AVX2</span>
          </div>
          <div className="fact">
            <strong>1:1</strong>
            <span>scalar reference for every kernel</span>
          </div>
        </div>

        <section className="section">
          <div className="section__head">
            <p className="eyebrow">Design</p>
            <h2>Four rules, applied everywhere.</h2>
            <p>
              Media toolkits accumulate decades of wrappers, build flags and platform glue. Ragdoll
              starts from a clean sheet and holds every component to the same constraints.
            </p>
          </div>
          <div className="principles">
            {principles.map((p, i) => (
              <div className="principle" key={p.title}>
                <span className="principle__num">0{i + 1}</span>
                <h3>{p.title}</h3>
                <p>{p.body}</p>
                <code>{p.code}</code>
              </div>
            ))}
          </div>
        </section>

        <section className="section">
          <div className="section__head">
            <p className="eyebrow">Architecture</p>
            <h2>One pipeline, seven crates.</h2>
            <p>
              Each stage of the media path is its own crate with a narrow API. Use the CLI, or depend
              on just the pieces you need.
            </p>
          </div>
          <div className="flow">
            {stages.map((s) => (
              <div className="flow__stage" key={s.k}>
                <small>{s.k}</small>
                <strong>{s.crate}</strong>
                <p>{s.body}</p>
              </div>
            ))}
            <div className="flow__base">
              <span>
                <strong>ragdoll-pipeline</strong> schedules GOPs across cores
              </span>
              <span>
                <strong>ragdoll-simd</strong> NEON and AVX2 kernels with scalar twins
              </span>
              <span>
                <strong>ragdoll-core</strong> time, color, frames, packets, bits
              </span>
            </div>
          </div>
        </section>

        <section className="section">
          <div className="split">
            <div className="section__head" style={{ marginBottom: 0 }}>
              <p className="eyebrow">Command line</p>
              <h2>Say what you mean.</h2>
              <p>
                One binary, one verb per task. Options are named, defaults are sensible, and every
                command can emit JSON for scripts. Input and output can be files or Unix pipes.
              </p>
              <p style={{ marginTop: 24 }}>
                <Link className="button" href="/docs/cli/">
                  CLI reference
                </Link>
              </p>
            </div>
            <div className="cmd-list">
              {commands.map((c) => (
                <Link className="cmd" href={c.href} key={c.name}>
                  <code>{c.name}</code>
                  <span>{c.body}</span>
                </Link>
              ))}
            </div>
          </div>
        </section>

        <section className="section">
          <div className="section__head">
            <p className="eyebrow">Performance</p>
            <h2>Measured, not promised.</h2>
            <p>
              Single-threaded codec throughput against libavcodec 6.1.1 and x264, both called
              in-process on the same inputs, on an x86_64 AVX2 Xeon at 2.1 GHz. The H.264 decoder
              matches the reference bit for bit; it does not yet match FFmpeg&apos;s speed.
            </p>
          </div>
          <div className="table-wrap" style={{ border: 0 }}>
            <table className="bench-table">
              <thead>
                <tr>
                  <th>Codec</th>
                  <th>Ragdoll</th>
                  <th>Reference</th>
                  <th>Ratio</th>
                </tr>
              </thead>
              <tbody>
                {bench.map((b) => (
                  <tr key={b.name}>
                    <td>{b.name}</td>
                    <td className="num">{b.ours}</td>
                    <td className="num" style={{ color: "var(--fg-3)" }}>
                      {b.theirs}
                    </td>
                    <td>
                      <div className="bench-bar">
                        <i style={{ width: `${b.x * 120}px` }} />
                        <b>{b.x.toFixed(2)}×</b>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="footnote">
            One machine, ±15% run to run; a ratio of 1.00× is parity. Opus decodes at about 0.8× and
            encodes at about 1× libopus; FLAC encodes 1.5× faster than FFmpeg (level 5) with ~11%
            larger files and decodes 2.5× slower. Read the{" "}
            <Link href="/docs/benchmarks/">methodology and caveats</Link> before quoting them.
          </p>
        </section>

        <section className="section">
          <div className="section__head">
            <p className="eyebrow">Status</p>
            <h2>Where things stand.</h2>
            <p>
              Ragdoll is pre-1.0. The codecs and tools below work and are checked against reference
              implementations; current work is on speed and on validation.
            </p>
          </div>
          <div className="status">
            {status.map((col) => (
              <div className="status__col" key={col.title}>
                <h3>
                  <span className="dot" data-s={col.s} aria-hidden />
                  {col.title}
                </h3>
                <ul>
                  {col.items.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
          <p className="footnote">
            See the <Link href="/docs/roadmap/">roadmap</Link> for details and known limitations.
          </p>
        </section>

        <section className="cta">
          <div>
            <h2>Build it, run it, read it.</h2>
            <p>
              Clone the repository, run the differential self-test on your own machine, and see every
              kernel checked against its reference.
            </p>
          </div>
          <div className="hero__actions" style={{ margin: 0 }}>
            <Link className="button button--primary" href="/docs/installation/">
              Install
            </Link>
            <a className="button" href={site.repo}>
              View source
            </a>
          </div>
        </section>
      </div>
    </main>
  );
}
