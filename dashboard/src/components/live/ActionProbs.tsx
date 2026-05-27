import { ACTION_COLORS, ACTION_LABELS } from "@/lib/format";

export function ActionProbs({
  probs,
  action,
}: {
  probs: number[] | null;
  action: number | null;
}): JSX.Element {
  const safe = probs ?? [0.2, 0.2, 0.2, 0.2, 0.2];
  return (
    <div className="flex flex-col justify-around gap-2 py-1">
      {safe.map((p, i) => {
        const isActive = action === i;
        const pct = Math.max(0, Math.min(1, p));
        return (
          <div key={i} className="flex items-center gap-2 text-xs">
            <span
              className={
                "w-9 text-right tabular-nums " +
                (isActive ? "text-zinc-100" : "text-zinc-500")
              }
            >
              {ACTION_LABELS[i]}
            </span>
            <div className="relative h-3 flex-1 overflow-hidden rounded-sm bg-zinc-800">
              <div
                className="absolute inset-y-0 left-0 transition-[width] duration-100"
                style={{
                  width: `${pct * 100}%`,
                  backgroundColor: ACTION_COLORS[i],
                  opacity: isActive ? 1 : 0.6,
                }}
              />
            </div>
            <span
              className={
                "w-10 text-right tabular-nums " +
                (isActive ? "text-zinc-100" : "text-zinc-500")
              }
            >
              {(pct * 100).toFixed(0)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}
