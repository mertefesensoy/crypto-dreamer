import { useState } from "react";

import { ArchitectureGraph } from "@/components/internals/ArchitectureGraph";
import { DreamPlayer } from "@/components/internals/DreamPlayer";
import { ObsWindowHeatmap } from "@/components/internals/ObsWindowHeatmap";
import { PortfolioState } from "@/components/internals/PortfolioState";
import { PlaceholderCard } from "@/components/training/PlaceholderCard";

const V2_PLACEHOLDERS = [
  {
    title: "Encoder attention",
    hint: "iTransformer per-feature attention map",
  },
  {
    title: "Latent UMAP",
    hint: "Stochastic state z_t projected to 2D, coloured by reward",
  },
  {
    title: "Live activations",
    hint: "Per-layer norms streamed from the actor head",
  },
];

type SubTab = "live" | "dream";

export function Internals(): JSX.Element {
  const [sub, setSub] = useState<SubTab>("live");
  return (
    <div className="flex h-full flex-col bg-zinc-950">
      <div className="flex items-center gap-1 border-b border-zinc-800 bg-zinc-900 px-3 py-1">
        <SubTabButton active={sub === "live"} onClick={() => setSub("live")}>
          Live internals
        </SubTabButton>
        <SubTabButton active={sub === "dream"} onClick={() => setSub("dream")}>
          Dream Player
        </SubTabButton>
      </div>
      <div className="flex-1 min-h-0">
        {sub === "live" ? <LiveInternals /> : <DreamPlayer />}
      </div>
    </div>
  );
}

function SubTabButton({
  active, onClick, children,
}: { active: boolean; onClick: () => void; children: React.ReactNode }): JSX.Element {
  return (
    <button
      onClick={onClick}
      className={
        "px-3 py-1 text-[11px] uppercase tracking-wide transition-colors " +
        (active ? "bg-zinc-800 text-zinc-100" : "text-zinc-400 hover:text-zinc-200")
      }
    >
      {children}
    </button>
  );
}

function LiveInternals(): JSX.Element {
  return (
    <div className="grid h-full grid-cols-[1fr_320px] grid-rows-1 gap-3 p-3">
      <div className="grid grid-rows-[3fr_2fr_2fr] gap-3 min-h-0">
        <Panel title="Architecture · live data flow" subtitle="edges pulse on each step">
          <ArchitectureGraph />
        </Panel>
        <Panel
          title="Obs window heatmap"
          subtitle="16 timesteps × 12 features (decimated, last row = current)"
        >
          <ObsWindowHeatmap />
        </Panel>
        <Panel title="Portfolio state" subtitle="cash% / btc% over the episode">
          <PortfolioState />
        </Panel>
      </div>
      <aside className="grid grid-rows-3 gap-3 min-h-0">
        {V2_PLACEHOLDERS.map((p) => (
          <PlaceholderCard key={p.title} title={p.title} hint={p.hint} />
        ))}
      </aside>
    </div>
  );
}

function Panel({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}): JSX.Element {
  return (
    <section className="flex min-h-0 flex-col rounded border border-zinc-800 bg-zinc-900">
      <header className="flex items-baseline justify-between border-b border-zinc-800 px-3 py-1.5">
        <span className="text-xs uppercase tracking-wide text-zinc-400">
          {title}
        </span>
        {subtitle && <span className="text-[10px] text-zinc-500">{subtitle}</span>}
      </header>
      <div className="flex-1 min-h-0 overflow-hidden p-2">{children}</div>
    </section>
  );
}
