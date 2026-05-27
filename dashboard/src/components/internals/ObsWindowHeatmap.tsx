import { useMemo } from "react";
import { scaleDiverging } from "d3-scale";
import { interpolateRdBu } from "d3-scale-chromatic";
import { useDreamerStore, selectActiveBucket } from "@/store/useDreamerStore";
import { decodeFeaturesB64 } from "@/lib/decode";
import {
  FEATURE_COLS,
  FEATURE_NAMES,
  FEATURE_TRANSPORT_ROWS,
} from "@/lib/constants";

// Negative -> blue, zero -> off-white, positive -> red.
// Inverting interpolateRdBu(t) (which is red at 0, blue at 1).
const COLOR = scaleDiverging<string>((t) => interpolateRdBu(1 - t)).domain([
  -3, 0, 3,
]);

export function ObsWindowHeatmap(): JSX.Element {
  const bucket = useDreamerStore(selectActiveBucket);
  const last = bucket && bucket.steps.length > 0
    ? bucket.steps[bucket.steps.length - 1]!
    : null;

  const matrix = useMemo(
    () =>
      last
        ? decodeFeaturesB64(last.features_b64, FEATURE_TRANSPORT_ROWS, FEATURE_COLS)
        : null,
    [last],
  );

  if (!matrix) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-zinc-500">
        Waiting for stream…
      </div>
    );
  }

  return (
    <div className="grid h-full grid-cols-[1fr_120px] gap-3">
      <Heatmap matrix={matrix} />
      <FeatureLegend matrix={matrix} />
    </div>
  );
}

function Heatmap({
  matrix,
}: {
  matrix: ReturnType<typeof decodeFeaturesB64>;
}): JSX.Element {
  const { rows, cols, data } = matrix;
  // Find a nice symmetric domain for the color scale, capped at the
  // diverging extremes so a single outlier doesn't wash out the rest.
  const stats = useMemo(() => {
    let max = 0;
    for (let i = 0; i < data.length; i++) {
      const a = Math.abs(data[i]!);
      if (a > max) max = a;
    }
    return { max: Math.min(Math.max(max, 0.1), 5) };
  }, [data]);
  COLOR.domain([-stats.max, 0, stats.max]);

  return (
    <svg
      viewBox={`0 0 ${cols} ${rows}`}
      preserveAspectRatio="none"
      className="h-full w-full"
      shapeRendering="crispEdges"
      role="img"
      aria-label="observation window heatmap"
    >
      {Array.from({ length: rows }).map((_, r) =>
        Array.from({ length: cols }).map((__, c) => {
          const v = data[r * cols + c]!;
          return (
            <rect
              key={`${r}-${c}`}
              x={c}
              y={r}
              width={1}
              height={1}
              fill={COLOR(v)}
            >
              <title>
                t-{rows - 1 - r} · {FEATURE_NAMES[c] ?? "?"} · {v.toFixed(3)}
              </title>
            </rect>
          );
        }),
      )}
    </svg>
  );
}

function FeatureLegend({
  matrix,
}: {
  matrix: ReturnType<typeof decodeFeaturesB64>;
}): JSX.Element {
  // Show the latest (last row) values to the right of the heatmap so
  // viewers can map columns to values without hovering.
  const { cols, rows, data } = matrix;
  const lastRowOffset = (rows - 1) * cols;
  return (
    <ul className="flex flex-col justify-around overflow-hidden text-[10px] tabular-nums text-zinc-400">
      {Array.from({ length: cols }).map((_, c) => {
        const v = data[lastRowOffset + c]!;
        return (
          <li key={c} className="flex items-center justify-between gap-1">
            <span className="truncate text-zinc-500">{FEATURE_NAMES[c] ?? "?"}</span>
            <span
              className={
                v > 0
                  ? "text-zinc-100"
                  : v < 0
                    ? "text-zinc-300"
                    : "text-zinc-500"
              }
            >
              {v.toFixed(2)}
            </span>
          </li>
        );
      })}
    </ul>
  );
}
