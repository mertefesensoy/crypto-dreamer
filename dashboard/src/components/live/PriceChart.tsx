import { useMemo } from "react";
import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { EpisodeBucket } from "@/store/useDreamerStore";
import { ACTION_COLORS, fmtTime, fmtUsd } from "@/lib/format";

type Row = {
  step: number;
  ts: string;
  price: number;
  alloc: number;
  action: number;
  marker: "buy" | "sell" | null;
};

const VISIBLE = 600;

export function PriceChart({
  bucket,
}: {
  bucket: EpisodeBucket | null;
}): JSX.Element {
  const rows = useMemo<Row[]>(() => {
    if (!bucket || bucket.steps.length === 0) return [];
    const all = bucket.steps;
    const start = Math.max(0, all.length - VISIBLE);
    const window = all.slice(start);
    return window.map((s, i) => {
      const prev = i === 0 ? null : window[i - 1];
      let marker: Row["marker"] = null;
      if (prev) {
        if (s.action > prev.action) marker = "buy";
        else if (s.action < prev.action) marker = "sell";
      }
      return {
        step: s.step,
        ts: s.ts,
        price: s.price,
        alloc: s.realized_alloc,
        action: s.action,
        marker,
      };
    });
  }, [bucket]);

  if (rows.length === 0) {
    return <Empty />;
  }

  const buyMarkers = rows.filter((r) => r.marker === "buy");
  const sellMarkers = rows.filter((r) => r.marker === "sell");
  const minPrice = Math.min(...rows.map((r) => r.price));
  const maxPrice = Math.max(...rows.map((r) => r.price));
  const pad = (maxPrice - minPrice) * 0.05 || 1;

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={rows} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
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
          yAxisId="price"
          domain={[minPrice - pad, maxPrice + pad]}
          stroke="#52525b"
          tick={{ fontSize: 10, fill: "#a1a1aa" }}
          tickFormatter={(v: number) => fmtUsd(v, 0)}
          width={64}
          orientation="left"
        />
        <YAxis
          yAxisId="alloc"
          domain={[0, 1]}
          stroke="#52525b"
          tick={{ fontSize: 10, fill: "#a1a1aa" }}
          tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
          width={36}
          orientation="right"
        />
        <Tooltip
          contentStyle={{
            background: "#18181b",
            border: "1px solid #3f3f46",
            fontSize: 11,
          }}
          labelFormatter={(label: string) => fmtTime(label)}
          formatter={(value: number, name: string) => {
            if (name === "price") return ["$" + fmtUsd(value, 2), "price"];
            if (name === "alloc") return [`${(value * 100).toFixed(1)}%`, "alloc"];
            return [value, name];
          }}
        />
        <Bar
          yAxisId="alloc"
          dataKey="alloc"
          isAnimationActive={false}
          fill="#a3e63540"
          stroke="none"
        />
        <Line
          yAxisId="price"
          type="monotone"
          dataKey="price"
          stroke="#e4e4e7"
          strokeWidth={1.5}
          dot={false}
          isAnimationActive={false}
        />
        {buyMarkers.map((r) => (
          <ReferenceDot
            yAxisId="price"
            key={`b-${r.step}`}
            x={r.ts}
            y={r.price}
            r={2.5}
            fill={ACTION_COLORS[r.action]}
            stroke="#0a0a0a"
            strokeWidth={0.5}
            isFront
          />
        ))}
        {sellMarkers.map((r) => (
          <ReferenceDot
            yAxisId="price"
            key={`s-${r.step}`}
            x={r.ts}
            y={r.price}
            r={2.5}
            fill="#f87171"
            stroke="#0a0a0a"
            strokeWidth={0.5}
            isFront
          />
        ))}
      </ComposedChart>
    </ResponsiveContainer>
  );
}

function Empty(): JSX.Element {
  return (
    <div className="flex h-full items-center justify-center text-xs text-zinc-500">
      Waiting for stream…
    </div>
  );
}
