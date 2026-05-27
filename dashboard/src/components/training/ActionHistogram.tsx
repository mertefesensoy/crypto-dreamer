import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { EpisodeBucket } from "@/store/useDreamerStore";
import { actionHistogram } from "@/lib/stats";
import { ACTION_COLORS, ACTION_LABELS } from "@/lib/format";

export function ActionHistogram({
  bucket,
}: {
  bucket: EpisodeBucket | null;
}): JSX.Element {
  const rows = useMemo(() => {
    const counts = actionHistogram(bucket?.steps ?? [], 5);
    const total = counts.reduce((a, b) => a + b, 0);
    return counts.map((count, i) => ({
      label: ACTION_LABELS[i] ?? "?",
      count,
      pct: total === 0 ? 0 : count / total,
      color: ACTION_COLORS[i] ?? "#71717a",
    }));
  }, [bucket]);

  const total = rows.reduce((a, r) => a + r.count, 0);
  if (total === 0) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-zinc-500">
        Waiting for stream…
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#27272a" vertical={false} />
            <XAxis
              dataKey="label"
              stroke="#52525b"
              tick={{ fontSize: 10, fill: "#a1a1aa" }}
            />
            <YAxis
              stroke="#52525b"
              tick={{ fontSize: 10, fill: "#a1a1aa" }}
              width={40}
            />
            <Tooltip
              contentStyle={{
                background: "#18181b",
                border: "1px solid #3f3f46",
                fontSize: 11,
              }}
              formatter={(value: number, _name: string, p: { payload?: { pct: number } }) => [
                `${value} (${((p.payload?.pct ?? 0) * 100).toFixed(1)}%)`,
                "count",
              ]}
            />
            <Bar dataKey="count" isAnimationActive={false}>
              {rows.map((r, i) => (
                <Cell key={i} fill={r.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="px-1 pt-1 text-[10px] leading-snug text-zinc-500">
        {total.toLocaleString()} steps in this episode. Uniform-random ~20%
        each.
      </p>
    </div>
  );
}
