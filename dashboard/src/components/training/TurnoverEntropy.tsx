import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { EpisodeBucket } from "@/store/useDreamerStore";
import { entropy } from "@/lib/stats";
import { fmtNum } from "@/lib/format";

const VISIBLE = 600;

type Row = { step: number; turnover: number; entropy: number };

export function TurnoverEntropy({
  bucket,
}: {
  bucket: EpisodeBucket | null;
}): JSX.Element {
  const rows = useMemo<Row[]>(() => {
    if (!bucket) return [];
    const all = bucket.steps;
    const start = Math.max(0, all.length - VISIBLE);
    return all.slice(start).map((s) => ({
      step: s.step,
      turnover: s.turnover,
      entropy: entropy(s.action_probs),
    }));
  }, [bucket]);

  if (rows.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-zinc-500">
        Waiting for stream…
      </div>
    );
  }

  const log5 = Math.log(5);

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
            <CartesianGrid stroke="#27272a" vertical={false} />
            <XAxis
              dataKey="step"
              stroke="#52525b"
              tick={{ fontSize: 10, fill: "#a1a1aa" }}
              minTickGap={48}
            />
            <YAxis
              yAxisId="turn"
              domain={[0, 1]}
              stroke="#52525b"
              tick={{ fontSize: 10, fill: "#a1a1aa" }}
              tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
              width={40}
            />
            <YAxis
              yAxisId="ent"
              orientation="right"
              domain={[0, log5 * 1.05]}
              stroke="#52525b"
              tick={{ fontSize: 10, fill: "#a1a1aa" }}
              tickFormatter={(v: number) => fmtNum(v, 2)}
              width={36}
            />
            <Tooltip
              contentStyle={{
                background: "#18181b",
                border: "1px solid #3f3f46",
                fontSize: 11,
              }}
              formatter={(value: number, name: string) => {
                if (name === "turnover")
                  return [`${(value * 100).toFixed(1)}%`, name];
                return [fmtNum(value, 4), name];
              }}
            />
            <Line
              yAxisId="turn"
              type="monotone"
              dataKey="turnover"
              stroke="#fbbf24"
              strokeWidth={1.2}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              yAxisId="ent"
              type="monotone"
              dataKey="entropy"
              stroke="#22d3ee"
              strokeWidth={1.2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="px-1 pt-1 text-[10px] leading-snug text-zinc-500">
        amber: turnover (left axis) · cyan: entropy (right axis,
        log 5 ≈ {log5.toFixed(3)} ceiling)
      </p>
    </div>
  );
}
