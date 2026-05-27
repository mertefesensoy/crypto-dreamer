import { useDreamerStore } from "@/store/useDreamerStore";
import { EpisodeSummaryTable } from "@/components/training/EpisodeSummaryTable";
import { RewardDecomposition } from "@/components/training/RewardDecomposition";
import { ActionHistogram } from "@/components/training/ActionHistogram";
import { TurnoverEntropy } from "@/components/training/TurnoverEntropy";
import { PlaceholderCard } from "@/components/training/PlaceholderCard";

const V2_PLACEHOLDERS = [
  { title: "World model loss", hint: "Recon + dynamics + reward + continue" },
  { title: "KL divergence", hint: "Posterior to prior, free-bit floor" },
  { title: "Actor loss", hint: "Reinforce + entropy bonus" },
  { title: "Critic loss", hint: "IQN quantile regression" },
];

export function Training(): JSX.Element {
  const episodes = useDreamerStore((s) => s.episodes);
  const order = useDreamerStore((s) => s.episodeOrder);
  const active = useDreamerStore((s) => s.activeEpisode);
  const setActive = useDreamerStore((s) => s.setActiveEpisode);

  const activeBucket = active !== null ? episodes[active] ?? null : null;

  return (
    <div className="grid h-full grid-rows-[auto_1fr_auto] gap-3 overflow-auto bg-zinc-950 p-3">
      <Panel title="Episodes">
        <EpisodeSummaryTable
          episodes={episodes}
          order={order}
          activeEpisode={active}
          onSelect={setActive}
        />
      </Panel>

      <div className="grid grid-cols-3 grid-rows-1 gap-3 min-h-0">
        <Panel title="Reward decomposition" subtitle="log-return vs turnover penalty">
          <RewardDecomposition bucket={activeBucket} />
        </Panel>
        <Panel title="Action histogram" subtitle="counts per allocation bucket">
          <ActionHistogram bucket={activeBucket} />
        </Panel>
        <Panel title="Turnover & policy entropy" subtitle="per-step">
          <TurnoverEntropy bucket={activeBucket} />
        </Panel>
      </div>

      <div className="grid grid-cols-4 grid-rows-1 gap-3">
        {V2_PLACEHOLDERS.map((p) => (
          <PlaceholderCard key={p.title} title={p.title} hint={p.hint} />
        ))}
      </div>
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
        {subtitle && (
          <span className="text-[10px] text-zinc-500">{subtitle}</span>
        )}
      </header>
      <div className="flex-1 min-h-0 overflow-hidden p-2">{children}</div>
    </section>
  );
}
