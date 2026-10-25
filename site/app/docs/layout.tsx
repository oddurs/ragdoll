import { Pager, Sidebar, Toc } from "@/components/DocsNav";
import { repoTree } from "@/lib/site";

export default function DocsLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="docs">
      <Sidebar />
      <main className="docs-main">
        <article className="prose">{children}</article>
        <Pager />
        <p className="docs-edit">
          Found something wrong? <a href={repoTree("site/app/docs")}>Edit these docs on GitHub</a>.
        </p>
      </main>
      <Toc />
    </div>
  );
}
