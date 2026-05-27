import type { ReactNode } from "react";
import { useDreamerStore } from "@/store/useDreamerStore";

export type TabId = "live" | "training" | "internals";

const TABS: { id: TabId; label: string }[] = [
  { id: "live", label: "Live" },
  { id: "training", label: "Training" },
  { id: "internals", label: "Internals" },
];

export function Shell({
  active,
  onTab,
  children,
}: {
  active: TabId;
  onTab: (t: TabId) => void;
  children: ReactNode;
}): JSX.Element {
  const connection = useDreamerStore((s) => s.connection);
  const totalSteps = useDreamerStore((s) => s.totalSteps);
  const activeEp = useDreamerStore((s) => s.activeEpisode);
  const episodeOrder = useDreamerStore((s) => s.episodeOrder);

  return (
    <div className="flex h-full flex-col font-mono text-sm">
      <header className="flex items-center justify-between border-b border-zinc-800 bg-zinc-900 px-4 py-2">
        <div className="flex items-center gap-6">
          <div className="text-zinc-100">
            <span className="font-semibold tracking-tight">crypto-dreamer</span>
            <span className="ml-2 text-xs text-zinc-500">v1 scaffold</span>
          </div>
          <nav className="flex gap-1">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => onTab(t.id)}
                className={
                  "px-3 py-1 text-xs uppercase tracking-wide transition-colors " +
                  (active === t.id
                    ? "bg-zinc-800 text-zinc-100"
                    : "text-zinc-400 hover:text-zinc-200")
                }
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>
        <div className="flex items-center gap-6 text-xs text-zinc-400">
          <ConnectionDot state={connection} />
          <span>
            steps&nbsp;
            <span className="text-zinc-200 tabular-nums">
              {totalSteps.toLocaleString()}
            </span>
          </span>
          <span>
            episodes&nbsp;
            <span className="text-zinc-200 tabular-nums">
              {episodeOrder.length}
            </span>
          </span>
          <span>
            active&nbsp;
            <span className="text-zinc-200 tabular-nums">
              {activeEp ?? "—"}
            </span>
          </span>
        </div>
      </header>
      <main className="flex-1 overflow-hidden">{children}</main>
    </div>
  );
}

function ConnectionDot({
  state,
}: {
  state: ReturnType<typeof useDreamerStore.getState>["connection"];
}): JSX.Element {
  const cls =
    state === "open"
      ? "bg-lime-400"
      : state === "connecting"
        ? "bg-amber-400 animate-pulse"
        : state === "error"
          ? "bg-red-500"
          : "bg-zinc-600";
  return (
    <span className="flex items-center gap-2">
      <span className={"h-2 w-2 rounded-full " + cls} />
      <span className="capitalize">{state}</span>
    </span>
  );
}
