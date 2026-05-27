import { useDreamerStore, selectActiveBucket } from "@/store/useDreamerStore";
import { PriceChart } from "@/components/live/PriceChart";
import { EquityChart } from "@/components/live/EquityChart";
import { ActionProbs } from "@/components/live/ActionProbs";
import { QuantileFan } from "@/components/live/QuantileFan";
import { KillSwitch } from "@/components/live/KillSwitch";
import { fmtUsd, fmtPct } from "@/lib/format";

export function Live(): JSX.Element {
  const bucket = useDreamerStore(selectActiveBucket);
  const last = bucket && bucket.steps.length > 0
    ? bucket.steps[bucket.steps.length - 1]
    : null;

  return (
    <div className="grid h-full grid-cols-[1fr_320px] grid-rows-1 gap-3 bg-zinc-950 p-3">
      <div className="grid grid-rows-[3fr_2fr_auto] gap-3 min-h-0">
        <Panel title="BTC / USDT  ·  allocation overlay">
          <PriceChart bucket={bucket} />
        </Panel>
        <Panel title="Equity">
          <EquityChart bucket={bucket} />
        </Panel>
        <KpiStrip last={last} />
      </div>
      <aside className="grid grid-rows-[auto_auto_auto_1fr] gap-3 min-h-0">
        <Panel title="Action probabilities">
          <ActionProbs probs={last?.action_probs ?? null} action={last?.action ?? null} />
        </Panel>
        <Panel title="Critic quantile fan" tag="v2">
          <QuantileFan />
        </Panel>
        <Panel title="Kill switch">
          <KillSwitch />
        </Panel>
        <Panel title="Episode">
          <EpisodeMeta />
        </Panel>
      </aside>
    </div>
  );
}

function Panel({
  title,
  tag,
  children,
}: {
  title: string;
  tag?: string;
  children: React.ReactNode;
}): JSX.Element {
  return (
    <section className="flex min-h-0 flex-col rounded border border-zinc-800 bg-zinc-900">
      <header className="flex items-center justify-between border-b border-zinc-800 px-3 py-1.5">
        <span className="text-xs uppercase tracking-wide text-zinc-400">
          {title}
        </span>
        {tag && (
          <span className="rounded bg-amber-900/40 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-amber-300">
            {tag}
          </span>
        )}
      </header>
      <div className="flex-1 min-h-0 p-2">{children}</div>
    </section>
  );
}

function KpiStrip({
  last,
}: {
  last: import("@/types").StepInfo | null;
}): JSX.Element {
  return (
    <div className="grid grid-cols-6 gap-3 text-xs">
      <Kpi label="price" value={last ? "$" + fmtUsd(last.price, 2) : "—"} />
      <Kpi
        label="alloc"
        value={last ? fmtPct(last.realized_alloc) : "—"}
      />
      <Kpi
        label="equity"
        value={last ? "$" + fmtUsd(last.equity, 2) : "—"}
      />
      <Kpi
        label="cash"
        value={last ? "$" + fmtUsd(last.cash, 2) : "—"}
      />
      <Kpi
        label="btc"
        value={last ? last.btc.toFixed(5) : "—"}
      />
      <Kpi
        label="reward"
        value={last ? last.reward.toFixed(5) : "—"}
        signed={last?.reward}
      />
    </div>
  );
}

function Kpi({
  label,
  value,
  signed,
}: {
  label: string;
  value: string;
  signed?: number | undefined;
}): JSX.Element {
  const tone =
    signed === undefined
      ? "text-zinc-100"
      : signed > 0
        ? "text-lime-400"
        : signed < 0
          ? "text-red-400"
          : "text-zinc-100";
  return (
    <div className="rounded border border-zinc-800 bg-zinc-900 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wide text-zinc-500">
        {label}
      </div>
      <div className={"font-semibold tabular-nums " + tone}>{value}</div>
    </div>
  );
}

function EpisodeMeta(): JSX.Element {
  const bucket = useDreamerStore(selectActiveBucket);
  if (!bucket) {
    return (
      <p className="text-xs text-zinc-500">
        Waiting for stream…
      </p>
    );
  }
  const first = bucket.steps[0];
  const last = bucket.steps[bucket.steps.length - 1];
  return (
    <dl className="grid grid-cols-[80px_1fr] gap-x-3 gap-y-1 text-xs">
      <dt className="text-zinc-500">episode</dt>
      <dd className="text-zinc-100 tabular-nums">{bucket.episode}</dd>
      <dt className="text-zinc-500">steps</dt>
      <dd className="text-zinc-100 tabular-nums">{bucket.steps.length}</dd>
      <dt className="text-zinc-500">start</dt>
      <dd className="text-zinc-100">{first ? first.ts.slice(11, 16) : "—"}</dd>
      <dt className="text-zinc-500">latest</dt>
      <dd className="text-zinc-100">{last ? last.ts.slice(11, 16) : "—"}</dd>
      {bucket.summary && (
        <>
          <dt className="text-zinc-500">total r</dt>
          <dd className="text-zinc-100 tabular-nums">
            {bucket.summary.total_reward.toFixed(3)}
          </dd>
          <dt className="text-zinc-500">final eq</dt>
          <dd className="text-zinc-100 tabular-nums">
            ${fmtUsd(bucket.summary.final_equity, 2)}
          </dd>
        </>
      )}
    </dl>
  );
}
