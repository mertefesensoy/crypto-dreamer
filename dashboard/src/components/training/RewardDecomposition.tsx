import { useMemo } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { EpisodeBucket } from "@/store/useDreamerStore";
import { decomposeReward } from "@/lib/stats";
import { fmtNum } from "@/lib/format";

const VISIBLE = 600;

type Row = {
  step: number;
  logRet: number;
  penalty: number;
  total: number;
};

export function RewardDecomposition({
  bucket,
}: {
  bucket: EpisodeBucket | null;
}): JSX.Element {
  const rows = useMemo<Row[]>(() => {
    if (!bucket) return [];
    const all = bucket.steps;
    const start = Math.max(0, all.length - VISIBLE);
    return all.slice(start).map((s) => {
      const { logRet, penalty } = decomposeReward(s);
      return {
        step: s.step,
        logRet,
        penalty,
        total: s.reward,
      };
    });
  }, [bucket]);

  const ratio = useMemo(() => {
    if (rows.length === 0) return null;
    let sumAbsLog = 0;
    let sumPenalty = 0;
    for (const r of rows) {
      sumAbsLog += Math.abs(r.logRet);
      sumPenalty += Math.abs(r.penalty);
    }
    return sumAbsLog === 0 ? null : sumPenalty / sumAbsLog;
  }, [rows]);

  if (rows.length === 0) {
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
          <ComposedChart
            data={rows}
            stackOffset="sign"
            margin={{ top: 4, right: 8, bottom: 0, left: 8 }}
          >
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
              tickFormatter={(v: number) => fmtNum(v, 3)}
              width={56}
            />
            <ReferenceLine y={0} stroke="#3f3f46" />
            <Tooltip
              contentStyle={{
                background: "#18181b",
                border: "1px solid #3f3f46",
                fontSize: 11,
              }}
              formatter={(value: number, name: string) => [
                fmtNum(value, 5),
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
              dataKey="logRet"
              stackId="1"
              name="log-return"
              stroke="#a3e635"
              fill="#a3e63555"
              isAnimationActive={false}
            />
            <Area
              type="monotone"
              dataKey="penalty"
              stackId="1"
              name="turnover penalty"
              stroke="#f87171"
              fill="#f8717155"
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <p className="px-1 pt-1 text-[10px] leading-snug text-zinc-500">
        ratio Σ|penalty| / Σ|log_ret| ={" "}
        <span className="text-zinc-200 tabular-nums">
          {ratio === null ? "—" : ratio.toFixed(2) + "×"}
        </span>
        . Random agent should sit near 10×.
      </p>
    </div>
  );
}
