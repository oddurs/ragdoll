type Props = { size?: number; className?: string; title?: string };

/** The Ragdoll glyph: an R assembled from 12-unit transform blocks. The first block is always tally red. */
export function Glyph({ size = 28, className, title = "Ragdoll" }: Props) {
  return (
    <svg
      viewBox="8 8 48 48"
      width={size}
      height={size}
      className={className}
      {...(title ? { role: "img", "aria-label": title } : { "aria-hidden": true })}
    >
      <g fill="currentColor">
        <rect x="11" y="26" width="12" height="12" rx="1.5" />
        <rect x="11" y="41" width="12" height="12" rx="1.5" />
        <rect x="26" y="11" width="12" height="12" rx="1.5" />
        <rect x="26" y="26" width="12" height="12" rx="1.5" />
        <path d="M41 11h.5a13.5 13.5 0 0 1 0 27H41Z" />
        <path d="M30 41h12l11 12H41Z" />
      </g>
      <rect x="11" y="11" width="12" height="12" rx="1.5" fill="var(--tally)" />
    </svg>
  );
}

export function Wordmark({ size = 24 }: { size?: number }) {
  return (
    <span className="wordmark" style={{ fontSize: size * 0.82 }}>
      <Glyph size={size} title="" />
      <span>ragdoll</span>
    </span>
  );
}
