"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { docsNav, flatNav } from "@/lib/nav";

const normalize = (p: string) => (p.endsWith("/") ? p : `${p}/`);

export function Sidebar() {
  const path = normalize(usePathname());
  const [open, setOpen] = useState(false);
  useEffect(() => setOpen(false), [path]);
  const current = flatNav.find((i) => i.href === path);

  return (
    <aside className="docs-sidebar" data-open={open}>
      <button
        type="button"
        className="docs-sidebar__toggle"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        <span>{current?.title ?? "Menu"}</span>
        <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden>
          <path d="M2 4.5 6 8.5l4-4" fill="none" stroke="currentColor" strokeWidth="1.5" />
        </svg>
      </button>
      <nav aria-label="Documentation">
        {docsNav.map((section) => (
          <div key={section.title} className="docs-sidebar__section">
            <h4>{section.title}</h4>
            <ul>
              {section.items.map((item) => (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    aria-current={item.href === path ? "page" : undefined}
                    className={item.title === item.title.toLowerCase() ? "mono" : undefined}
                  >
                    {item.title}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>
    </aside>
  );
}

export function Pager() {
  const path = normalize(usePathname());
  const i = flatNav.findIndex((n) => n.href === path);
  if (i < 0) return null;
  const prev = flatNav[i - 1];
  const next = flatNav[i + 1];
  return (
    <nav className="docs-pager" aria-label="Pagination">
      {prev ? (
        <Link href={prev.href} className="docs-pager__link">
          <span>Previous</span>
          {prev.title}
        </Link>
      ) : (
        <span />
      )}
      {next ? (
        <Link href={next.href} className="docs-pager__link docs-pager__link--next">
          <span>Next</span>
          {next.title}
        </Link>
      ) : (
        <span />
      )}
    </nav>
  );
}

type Heading = { id: string; text: string; level: number };

export function Toc() {
  const path = usePathname();
  const [headings, setHeadings] = useState<Heading[]>([]);
  const [active, setActive] = useState<string>("");

  useEffect(() => {
    const els = Array.from(document.querySelectorAll<HTMLElement>(".prose h2[id], .prose h3[id]"));
    setHeadings(
      els.map((el) => ({ id: el.id, text: el.textContent ?? "", level: el.tagName === "H2" ? 2 : 3 })),
    );
    const io = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting);
        if (visible[0]) setActive(visible[0].target.id);
      },
      { rootMargin: "-80px 0px -70% 0px" },
    );
    els.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, [path]);

  if (headings.length < 2) return <aside className="docs-toc" />;
  return (
    <aside className="docs-toc">
      <h4>On this page</h4>
      <ul>
        {headings.map((h) => (
          <li key={h.id} data-level={h.level}>
            <a href={`#${h.id}`} aria-current={active === h.id ? "location" : undefined}>
              {h.text}
            </a>
          </li>
        ))}
      </ul>
    </aside>
  );
}
