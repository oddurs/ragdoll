import Link from "next/link";
import type { ReactNode } from "react";
import { asset } from "@/lib/site";

type CalloutKind = "note" | "status" | "warning";

export function Callout({
  kind = "note",
  title,
  children,
}: {
  kind?: CalloutKind;
  title?: string;
  children: ReactNode;
}) {
  return (
    <div className="callout" data-kind={kind}>
      {title ? <p className="callout__title">{title}</p> : null}
      <div className="callout__body">{children}</div>
    </div>
  );
}

/** A labelled support badge for matrices: yes / partial / no / planned. */
export function S({ v }: { v: "yes" | "partial" | "no" | "planned" | "inspect" | "passthrough" }) {
  const label = {
    yes: "Yes",
    partial: "Partial",
    no: "—",
    planned: "Planned",
    inspect: "Inspect",
    passthrough: "Copy",
  }[v];
  return (
    <span className="support" data-v={v}>
      {label}
    </span>
  );
}

export function Cards({ children }: { children: ReactNode }) {
  return <div className="cards">{children}</div>;
}

export function Card({ title, href, children }: { title: string; href: string; children: ReactNode }) {
  return (
    <Link className="card" href={href}>
      <strong>{title}</strong>
      <span>{children}</span>
    </Link>
  );
}

/** An image from /public that respects the configured base path. */
export function Img({ src, alt }: { src: string; alt: string }) {
  return <img src={asset(src)} alt={alt} />;
}
