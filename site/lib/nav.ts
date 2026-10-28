export type NavItem = { title: string; href: string };
export type NavSection = { title: string; items: NavItem[] };

export const docsNav: NavSection[] = [
  {
    title: "Getting started",
    items: [
      { title: "Introduction", href: "/docs/" },
      { title: "Installation", href: "/docs/installation/" },
      { title: "Quickstart", href: "/docs/quickstart/" },
      { title: "Coming from FFmpeg", href: "/docs/from-ffmpeg/" },
    ],
  },
  {
    title: "Command line",
    items: [
      { title: "Overview", href: "/docs/cli/" },
      { title: "probe", href: "/docs/cli/probe/" },
      { title: "remux", href: "/docs/cli/remux/" },
      { title: "transcode", href: "/docs/cli/transcode/" },
      { title: "cut", href: "/docs/cli/cut/" },
      { title: "concat", href: "/docs/cli/concat/" },
      { title: "package", href: "/docs/cli/package/" },
      { title: "extract", href: "/docs/cli/extract/" },
      { title: "metrics", href: "/docs/cli/metrics/" },
      { title: "synth", href: "/docs/cli/synth/" },
      { title: "bench", href: "/docs/cli/bench/" },
      { title: "self-test", href: "/docs/cli/self-test/" },
      { title: "Pipeline recipes", href: "/docs/cli/recipes/" },
    ],
  },
  {
    title: "Reference",
    items: [
      { title: "Formats & codecs", href: "/docs/formats/" },
      { title: "Power modes", href: "/docs/power/" },
      { title: "Library usage", href: "/docs/library/" },
    ],
  },
  {
    title: "Architecture",
    items: [
      { title: "Overview", href: "/docs/architecture/" },
      { title: "SIMD & verification", href: "/docs/architecture/simd/" },
      { title: "Memory model", href: "/docs/architecture/memory/" },
      { title: "ragdoll-core", href: "/docs/crates/ragdoll-core/" },
      { title: "ragdoll-simd", href: "/docs/crates/ragdoll-simd/" },
      { title: "ragdoll-pix", href: "/docs/crates/ragdoll-pix/" },
      { title: "ragdoll-format", href: "/docs/crates/ragdoll-format/" },
      { title: "ragdoll-codec", href: "/docs/crates/ragdoll-codec/" },
      { title: "ragdoll-pipeline", href: "/docs/crates/ragdoll-pipeline/" },
    ],
  },
  {
    title: "Project",
    items: [
      { title: "Benchmarks", href: "/docs/benchmarks/" },
      { title: "Roadmap", href: "/docs/roadmap/" },
      { title: "Contributing", href: "/docs/contributing/" },
      { title: "Brand", href: "/docs/brand/" },
    ],
  },
];

export const flatNav: NavItem[] = docsNav.flatMap((s) => s.items);
