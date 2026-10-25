import Link from "next/link";
import type { MDXComponents } from "mdx/types";
import type { ComponentProps } from "react";
import { Pre } from "@/components/CodeBlock";
import { Callout, Card, Cards, Img, S } from "@/components/Prose";

function A({ href = "", ...rest }: ComponentProps<"a">) {
  if (href.startsWith("/")) return <Link href={href} {...rest} />;
  if (href.startsWith("#")) return <a href={href} {...rest} />;
  return <a href={href} rel="noreferrer" {...rest} />;
}

function anchored(Tag: "h2" | "h3") {
  return function Heading({ id, children, ...rest }: ComponentProps<"h2">) {
    return (
      <Tag id={id} {...rest}>
        {id ? (
          <a href={`#${id}`} className="anchor">
            {children}
          </a>
        ) : (
          children
        )}
      </Tag>
    );
  };
}

function Table(props: ComponentProps<"table">) {
  return (
    <div className="table-wrap">
      <table {...props} />
    </div>
  );
}

const components: MDXComponents = {
  a: A,
  h2: anchored("h2"),
  h3: anchored("h3"),
  pre: Pre,
  table: Table,
  Callout,
  Cards,
  Card,
  S,
  Img,
};

export function useMDXComponents(): MDXComponents {
  return components;
}
