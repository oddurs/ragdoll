import Link from "next/link";
import { Glyph } from "./Logo";
import { site, repoFile } from "@/lib/site";

export function Footer() {
  return (
    <footer className="site-footer">
      <div className="site-footer__inner">
        <div className="site-footer__brand">
          <Glyph size={20} title="" />
          <p>
            Ragdoll is free software, dual-licensed under{" "}
            <a href={repoFile("LICENSE-MIT")}>MIT</a> or{" "}
            <a href={repoFile("LICENSE-APACHE")}>Apache-2.0</a>.
          </p>
        </div>
        <nav className="site-footer__cols" aria-label="Footer">
          <div>
            <h3>Docs</h3>
            <Link href="/docs/quickstart/">Quickstart</Link>
            <Link href="/docs/cli/">Command line</Link>
            <Link href="/docs/formats/">Formats &amp; codecs</Link>
            <Link href="/docs/library/">Library usage</Link>
          </div>
          <div>
            <h3>Project</h3>
            <Link href="/docs/architecture/">Architecture</Link>
            <Link href="/docs/benchmarks/">Benchmarks</Link>
            <Link href="/docs/roadmap/">Roadmap</Link>
            <Link href="/docs/brand/">Brand</Link>
          </div>
          <div>
            <h3>Source</h3>
            <a href={site.repo}>GitHub</a>
            <a href={repoFile("CONTRIBUTING.md")}>Contributing</a>
            <a href={repoFile("ROADMAP.md")}>ROADMAP.md</a>
            <a href={`${site.repo}/issues`}>Issues</a>
          </div>
        </nav>
      </div>
    </footer>
  );
}
