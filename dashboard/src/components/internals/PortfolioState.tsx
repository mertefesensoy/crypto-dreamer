import { useMemo } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useDreamerStore, selectActiveBucket } from "@/store/useDreamerStore";

const VISIBLE = 600;

type Row = { step: number; cashPct: number; btcPct: number };

export function PortfolioState(): JSX.Element {
  const bucket = useDreamerStore(selectActiveBucket);

  const rows = useMemo<Row[]>(() => {
    if (!bucket) return [];
    const all = bucket.steps;
    const start = Math.max(0, all.length - VISIBLE);
    return all.slice(start).map((s) => {
      const eq = Math.max(s.equity, 1e-9);
      const btcPct = (s.btc * s.price) / eq;
      const cashPct = s.cash / eq;
      return { step: s.step, cashPct, btcPct };
    });
  }, [bucket]);

  if (rows.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-zinc-500">
        Waiting for stream…
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={rows} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
        <CartesianGrid stroke="#27272a" vertical={false} />
        <XAxis
          dataKey="step"
          stroke="#52525b"
          tick={{ fontSize: 10, fill: "#a1a1aa" }}
          minTickGap={48}
        />
        <YAxis
          stroke="#52525b"
          tick={{ fontSize: 10, fill: "#a1a1aa" }}
          tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
          domain={[0, "dataMax"]}
          width={40}
        />
        <Tooltip
          contentStyle={{
            background: "#18181b",
            border: "1px solid #3f3f46",
            fontSize: 11,
          }}
          formatter={(value: number, name: string) => [
            `${(value * 100).toFixed(1)}%`,
            name,
          ]}
        />
        <Legend
          wrapperStyle={{ fontSize: 10, color: "#a1a1aa" }}
          iconType="square"
          iconSize={8}
        />
        <Area
          type="monotone"
          dataKey="cashPct"
          stackId="1"
          name="cash"
          stroke="#94a3b8"
          fill="#94a3b855"
          isAnimationActive={false}
        />
        <Area
          type="monotone"
          dataKey="btcPct"
          stackId="1"
          name="btc"
          stroke="#a3e635"
          fill="#a3e63555"
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
