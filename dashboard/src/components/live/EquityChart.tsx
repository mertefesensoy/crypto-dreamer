import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { EpisodeBucket } from "@/store/useDreamerStore";
import { fmtTime, fmtUsd } from "@/lib/format";

const VISIBLE = 600;

export function EquityChart({
  bucket,
}: {
  bucket: EpisodeBucket | null;
}): JSX.Element {
  const rows = useMemo(() => {
    if (!bucket || bucket.steps.length === 0) return [];
    const all = bucket.steps;
    const start = Math.max(0, all.length - VISIBLE);
    return all.slice(start).map((s) => ({
      ts: s.ts,
      equity: s.equity,
    }));
  }, [bucket]);

  if (rows.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-zinc-500">
        Waiting for stream…
      </div>
    );
  }

  const min = Math.min(...rows.map((r) => r.equity));
  const max = Math.max(...rows.map((r) => r.equity));
  const pad = (max - min) * 0.05 || 1;

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={rows} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
        <CartesianGrid stroke="#27272a" vertical={false} />
        <XAxis
          dataKey="ts"
          stroke="#52525b"
          tick={{ fontSize: 10, fill: "#a1a1aa" }}
          tickFormatter={fmtTime}
          interval="preserveStartEnd"
          minTickGap={48}
        />
        <YAxis
          domain={[min - pad, max + pad]}
          stroke="#52525b"
          tick={{ fontSize: 10, fill: "#a1a1aa" }}
          tickFormatter={(v: number) => "$" + fmtUsd(v, 0)}
          width={64}
        />
        <ReferenceLine y={10000} stroke="#3f3f46" strokeDasharray="2 4" />
        <Tooltip
          contentStyle={{
            background: "#18181b",
            border: "1px solid #3f3f46",
            fontSize: 11,
          }}
          labelFormatter={(label: string) => fmtTime(label)}
          formatter={(value: number) => ["$" + fmtUsd(value, 2), "equity"]}
        />
        <Line
          type="monotone"
          dataKey="equity"
          stroke="#a3e635"
          strokeWidth={1.5}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
