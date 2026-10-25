import Link from "next/link";
import { site } from "@/lib/site";

const probeOutput = `┌──────────────────────────────────────────────────────────┐
│ RAGDOLL PROBE : sample.mp4                               │
├──────────────────────────────────────────────────────────┤
│ Container     : mp4/isobmff      File Size : 7.48 MB     │
│ Probe Latency :  42.75 µs        GOP Count : 10          │
│ H.264 Profile : Constrained Baseline (level 4.0)         │
├──────────────────────────────────────────────────────────┤
│ Track #0 [Video] h264 1280x720 @ 30.000 fps  yuv420p     │
│   Colorimetry : BT709 / BT709 / BT709, Limited           │
│   Samples     : 60 packets (7842908 bytes)               │
├──────────────────────────────────────────────────────────┤
│ Bitrate Spark : ▇▁▁▁▁▁▆▁▁▁▁▁▇▁▂▁▁▇▁▁▁▂▁▇▁▁▁▁▆▁▁▁▁▁▇▁▁▂▁▁ │
└──────────────────────────────────────────────────────────┘`;

const principles = [
  {
    title: "No C, no third-party crates",
    body: "Every parser, transform, entropy coder and muxer is written in Rust inside this workspace. The dependency graph is seven crates deep and ends there.",
    code: "Cargo.lock → 7 packages, all ragdoll-*",
  },
  {
    title: "Explicit SIMD, proven against scalar",
    body: "Hot loops are hand-written NEON intrinsics behind safe slice APIs. Each one ships with a scalar reference and a differential test that must match bit for bit.",
    code: "sad_plane_u8 ≡ sad_plane_u8_scalar",
  },
  {
    title: "Zero-copy by default",
    body: "Demuxers hand out packets that borrow straight from a memory-mapped file. Frames live in 64-byte-aligned pooled buffers with strides chosen to avoid cache-set aliasing.",
    code: "PacketSlice<'mmap>",
  },
  {
    title: "A command line with verbs",
    body: "probe, remux, transcode, extract, metrics. Each task is a subcommand with named options, JSON output, and a declarative TOML form for repeatable pipelines.",
    code: "ragdoll transcode in.mp4 out.mkv --scale 1280x720",
  },
];

const stages = [
  { k: "Read", crate: "ragdoll-format", body: "mmap demux of MP4, Matroska/WebM, MPEG-TS, Y4M, WAV, FLAC, WebVTT and SRT." },
  { k: "Decode", crate: "ragdoll-codec", body: "H.264, FLAC, PCM and the RGDI intra codec. AV1 OBU inspection." },
  { k: "Transform", crate: "ragdoll-pix", body: "Color matrices, polyphase scaling, dithering, PSNR / SSIM / MS-SSIM." },
  { k: "Encode", crate: "ragdoll-codec", body: "GOP-parallel H.264 and RGDI encode, FLAC encode, PCM conversion." },
  { k: "Write", crate: "ragdoll-format", body: "Vectored writev muxers with fast-start MP4 and Cues-indexed Matroska." },
];

const commands = [
  { name: "probe", href: "/docs/cli/probe/", body: "Inspect containers, tracks, colorimetry and GOP structure in microseconds." },
  { name: "remux", href: "/docs/cli/remux/", body: "Move streams between MP4, Matroska and MPEG-TS without touching a pixel." },
  { name: "transcode", href: "/docs/cli/transcode/", body: "Scale and re-encode video across GOP-parallel workers, keeping audio and subtitles in sync." },
  { name: "extract", href: "/docs/cli/extract/", body: "Pull a frame at a timestamp or tile a contact sheet of thumbnails." },
  { name: "metrics", href: "/docs/cli/metrics/", body: "Score a distorted file against a reference: PSNR, SSIM, MS-SSIM, VMAF-Lite." },
  { name: "bench", href: "/docs/cli/bench/", body: "Run the end-to-end benchmark suite, optionally head to head with a local FFmpeg." },
];

const bench = [
  { name: "Annex-B start-code scan", ours: "32.61 GB/s", theirs: "7.21 GB/s", x: 4.52 },
  { name: "MPEG-TS CRC-32", ours: "2.56 GB/s", theirs: "0.59 GB/s", x: 4.34 },
  { name: "4K → 1080p Lanczos-3 scale", ours: "506.5 fps", theirs: "214.8 fps", x: 2.36 },
  { name: "H.264 RBSP unescape", ours: "10.65 GB/s", theirs: "5.14 GB/s", x: 2.07 },
  { name: "4K plane SSIM", ours: "10.06 Gpx/s", theirs: "5.04 Gpx/s", x: 2.0 },
  { name: "H.264 4×4 DCT + IDCT", ours: "191.0 Mblk/s", theirs: "125.5 Mblk/s", x: 1.52 },
  { name: "1080p RGB24 → YUV420p", ours: "6.02 Gpx/s", theirs: "4.03 Gpx/s", x: 1.49 },
];

const milestones = [
  { id: "m1", s: "done", title: "Foundation", body: "Timebases, bitstreams, CRCs, aligned frame pools, core SIMD." },
  { id: "m2", s: "done", title: "Pixels", body: "Color conversion, scaling, dithering, quality metrics." },
  { id: "m3", s: "done", title: "Containers", body: "MP4, Matroska/WebM, MPEG-TS, Y4M, Netpbm, WAV." },
  { id: "m4", s: "done", title: "Codecs", body: "FLAC, H.264 Baseline, AV1 OBU inspection, RGDI." },
  { id: "m5", s: "done", title: "CLI & bench", body: "GOP-parallel pipeline, seven subcommands, benchmark suite." },
  { id: "m6", s: "done", title: "Streaming & sync", body: "Bounded-memory GOP ring, pipes, multi-track A/V and subtitles." },
  { id: "m7", s: "doing", title: "Production video", body: "Full residual coding, quarter-pel, deblocking, CABAC, CRF." },
  { id: "m8", s: "planned", title: "Pro audio", body: "AAC-LC, Opus, resampling, EBU R128 loudness." },
  { id: "m9", s: "planned", title: "Edit & package", body: "Smart-GOP cut and concat, PNG/JPEG, filters, HLS and DASH." },
  { id: "m10", s: "planned", title: "1.0", body: "Stable library SDK, fuzzing, release benchmark suite." },
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
              hand-written SIMD in every hot loop and no C libraries anywhere.
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
              <span className="d"> sample.mp4 -&gt; sample.mkv | frames=60 | 1.10 ms </span>
              <span className="g">(14.2 GB/s)</span>
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
            <strong>28</strong>
            <span>hand-written NEON kernels</span>
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
                <strong>ragdoll-simd</strong> NEON kernels with scalar twins
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
              Kernel throughput on an Apple M4, against FFmpeg 9.0.2 built with clang -O3
              -mcpu=native, on identical 4K planes and 16 MB bitstreams. Every Ragdoll result is
              checked against its scalar reference.
            </p>
          </div>
          <div className="table-wrap" style={{ border: 0 }}>
            <table className="bench-table">
              <thead>
                <tr>
                  <th>Kernel</th>
                  <th>Ragdoll</th>
                  <th>FFmpeg 9.0.2</th>
                  <th>Speedup</th>
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
                        <i style={{ width: `${(b.x / 4.52) * 120}px` }} />
                        <b>{b.x.toFixed(2)}×</b>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="footnote">
            One machine. Ragdoll figures are reproducible with <code>ragdoll self-test</code>; read
            the <Link href="/docs/benchmarks/">methodology and caveats</Link> before quoting them.
          </p>
        </section>

        <section className="section">
          <div className="section__head">
            <p className="eyebrow">Status</p>
            <h2>Built in the open, one milestone at a time.</h2>
            <p>
              Ragdoll is pre-1.0. Six milestones are done; production-grade H.264 is in progress. The
              roadmap lives in the repository as plain Markdown.
            </p>
          </div>
          <ol className="milestones">
            {milestones.map((m) => (
              <li className="milestone" key={m.id}>
                <div className="milestone__id">
                  <span>{m.id}</span>
                  <span className="dot" data-s={m.s} aria-label={m.s} />
                </div>
                <strong>{m.title}</strong>
                <p>{m.body}</p>
              </li>
            ))}
          </ol>
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
