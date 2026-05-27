export function PlaceholderCard({
  title,
  hint,
}: {
  title: string;
  hint: string;
}): JSX.Element {
  return (
    <section className="relative flex flex-col rounded border border-zinc-800 bg-zinc-900">
      <header className="flex items-center justify-between border-b border-zinc-800 px-3 py-1.5">
        <span className="text-xs uppercase tracking-wide text-zinc-400">
          {title}
        </span>
        <span className="rounded bg-amber-900/40 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-amber-300">
          v2
        </span>
      </header>
      <div className="flex flex-1 min-h-0 flex-col items-start justify-end p-3">
        <SparkPlaceholder />
        <p className="pt-2 text-[11px] leading-snug text-zinc-500">{hint}</p>
      </div>
    </section>
  );
}

function SparkPlaceholder(): JSX.Element {
  // Static decorative path so the card has visual weight without faking data.
  return (
    <svg
      viewBox="0 0 200 40"
      preserveAspectRatio="none"
      className="h-10 w-full text-zinc-700"
    >
      <path
        d="M0,30 L40,28 L80,32 L120,26 L160,30 L200,28"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeDasharray="2 3"
      />
    </svg>
  );
}
