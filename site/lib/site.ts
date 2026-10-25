export const site = {
  name: "Ragdoll",
  tagline: "Media, from first principles.",
  description:
    "Ragdoll is a media framework and command-line tool written entirely in Rust: containers, codecs, scaling, color and quality metrics, with hand-written SIMD and no C libraries.",
  repo: "https://github.com/oddurs/ragdoll",
  // Canonical origin for Open Graph URLs; override with NEXT_PUBLIC_SITE_URL.
  url: process.env.NEXT_PUBLIC_SITE_URL ?? "https://oddurs.github.io/ragdoll",
  version: "0.1.0",
  license: "MIT OR Apache-2.0",
};

export const repoFile = (path: string) => `${site.repo}/blob/main/${path}`;
export const repoTree = (path: string) => `${site.repo}/tree/main/${path}`;

export const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
export const asset = (path: string) => `${basePath}${path}`;
