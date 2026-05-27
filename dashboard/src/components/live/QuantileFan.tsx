import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";

const FLAT = Array.from({ length: 24 }, (_, i) => ({ x: i, y: 0 }));

export function QuantileFan(): JSX.Element {
  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={FLAT} margin={{ top: 6, right: 6, bottom: 0, left: 6 }}>
            <CartesianGrid stroke="#27272a" vertical={false} />
            <XAxis dataKey="x" hide />
            <YAxis domain={[-1, 1]} hide />
            <Line
              type="monotone"
              dataKey="y"
              stroke="#52525b"
              strokeDasharray="2 3"
              dot={false}
              strokeWidth={1}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="px-1 pt-1 text-[10px] leading-snug text-zinc-500">
        IQN 9-quantile distributional return estimate. Wired in v2 once the
        critic exists.
      </p>
    </div>
  );
}
