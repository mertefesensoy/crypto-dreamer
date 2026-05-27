import { useMemo } from "react";
import type { EpisodeBucket } from "@/store/useDreamerStore";
import { computeEpisodeStats } from "@/lib/stats";
import { fmtUsd, fmtPct, fmtNum } from "@/lib/format";

type Row = ReturnType<typeof computeEpisodeStats> & {
  agentId: string;
};

export function EpisodeSummaryTable({
  episodes,
  order,
  activeEpisode,
  onSelect,
}: {
  episodes: Record<number, EpisodeBucket>;
  order: number[];
  activeEpisode: number | null;
  onSelect: (ep: number) => void;
}): JSX.Element {
  const rows = useMemo<Row[]>(() => {
    return order.map((ep) => {
      const bucket = episodes[ep]!;
      const stats = computeEpisodeStats(bucket.steps, ep);
      const agentId = bucket.steps[0]?.agent_id ?? bucket.summary?.agent_id ?? "—";
      // If summary disagrees with locally-computed totals (history truncation),
      // prefer the summary for steps + total reward + final equity.
      if (bucket.summary) {
        return {
          ...stats,
          steps: bucket.summary.steps,
          totalReward: bucket.summary.total_reward,
          finalEquity: bucket.summary.final_equity,
          agentId,
        };
      }
      return { ...stats, agentId };
    });
  }, [episodes, order]);

  if (rows.length === 0) {
    return (
      <p className="px-2 py-3 text-xs text-zinc-500">
        No episodes yet. Run the agent to populate.
      </p>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-xs tabular-nums">
        <thead>
          <tr className="border-b border-zinc-800 text-left text-zinc-500">
            <Th>ep</Th>
            <Th>agent</Th>
            <Th align="right">steps</Th>
            <Th align="right">total r</Th>
            <Th align="right">final equity</Th>
            <Th align="right">max DD</Th>
            <Th align="right">mean turnover</Th>
            <Th align="right">mean |log_ret|</Th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const isActive = activeEpisode === r.episode;
            return (
              <tr
                key={r.episode}
                onClick={() => onSelect(r.episode)}
                className={
                  "cursor-pointer border-b border-zinc-800/60 transition-colors " +
                  (isActive
                    ? "bg-zinc-800/60 text-zinc-100"
                    : "text-zinc-300 hover:bg-zinc-900/80")
                }
              >
                <Td>{r.episode}</Td>
                <Td>{r.agentId}</Td>
                <Td align="right">{r.steps.toLocaleString()}</Td>
                <Td align="right" tone={r.totalReward}>
                  {fmtNum(r.totalReward, 3)}
                </Td>
                <Td align="right">${fmtUsd(r.finalEquity, 2)}</Td>
                <Td align="right" tone={-r.maxDrawdown}>
                  {fmtPct(r.maxDrawdown, 2)}
                </Td>
                <Td align="right">{fmtPct(r.meanTurnover, 2)}</Td>
                <Td align="right">{fmtNum(r.meanAbsLogRet, 5)}</Td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Th({
  children,
  align = "left",
}: {
  children: React.ReactNode;
  align?: "left" | "right";
}): JSX.Element {
  return (
    <th
      className={
        "px-2 py-1 text-[10px] uppercase tracking-wide " +
        (align === "right" ? "text-right" : "text-left")
      }
    >
      {children}
    </th>
  );
}

function Td({
  children,
  align = "left",
  tone,
}: {
  children: React.ReactNode;
  align?: "left" | "right";
  tone?: number;
}): JSX.Element {
  const toneClass =
    tone === undefined
      ? ""
      : tone > 0
        ? "text-lime-400"
        : tone < 0
          ? "text-red-400"
          : "";
  return (
    <td
      className={
        "px-2 py-1 " +
        (align === "right" ? "text-right" : "text-left") +
        " " +
        toneClass
      }
    >
      {children}
    </td>
  );
}
