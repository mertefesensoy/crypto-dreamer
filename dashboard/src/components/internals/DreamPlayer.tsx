/**
 * DreamPlayer — Phase 5.4 sub-tab in Internals.
 *
 * Pick an episode + a starting step, edit a 50-step action sequence,
 * POST to /dream, and render a fan chart of imagined log-return
 * trajectories vs. the real subsequent path.
 *
 * Notes:
 * - Default action sequence is the realized_alloc bucket at start_step
 *   repeated 50× (zero-knowledge baseline: "do nothing different").
 * - Fan chart uses sample percentiles (p25/p50/p75) computed
 *   client-side from the n_samples × H × 15 tensor.
 * - When the world model checkpoint isn't loaded yet, the API returns
 *   503 and we surface a friendly message rather than break the tab.
 */
import { useEffect, useMemo, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { listDreamEpisodes, postDream } from "@/lib/dreamApi";
import type { DreamEpisode, DreamResponse } from "@/types";

const HORIZON = 50;
const ACTIONS = [0, 1, 2, 3, 4]; // 0%, 25%, 50%, 75%, 100% allocation
const ALLOC_LABELS = ["0%", "25%", "50%", "75%", "100%"];
const N_SAMPLES = 32;

type FanRow = {
  step: number;
  predLogRet_p25: number;
  predLogRet_p50: number;
  predLogRet_p75: number;
  predLogRetSpread: number; // p75 - p25 (recharts Area trick)
  realLogRet: number | null;
  predEquity_p25: number;
  predEquity_p50: number;
  predEquity_p75: number;
  predEquitySpread: number;
  realEquity: number | null;
};

function percentile(arr: number[], p: number): number {
  const sorted = [...arr].sort((a, b) => a - b);
  if (sorted.length === 0) return 0;
  const i = Math.min(sorted.length - 1, Math.floor((p / 100) * (sorted.length - 1)));
  return sorted[i];
}

/** Convert (n_samples, H, 15) predicted obs + (H,) real into fan rows. */
function buildFanData(
  resp: DreamResponse,
  H: number,
): FanRow[] {
  const n = resp.predicted_obs.length;

  // Per-step log_ret across samples.
  const logRetPerStep: number[][] = [];
  // Cumulative log_ret per sample → equity proxy.
  const cumLogRetPerSample: number[][] = Array.from({ length: n }, () => Array(H).fill(0));

  for (let s = 0; s < n; s++) {
    let cum = 0;
    for (let t = 0; t < H; t++) {
      const lr = resp.predicted_obs[s][t][0]; // feature 0 = log_ret
      cum += lr;
      cumLogRetPerSample[s][t] = cum;
    }
  }
  for (let t = 0; t < H; t++) {
    const slice = resp.predicted_obs.map((s) => s[t][0]);
    logRetPerStep.push(slice);
  }

  let realCum = 0;
  return Array.from({ length: H }, (_, t) => {
    const samples = logRetPerStep[t];
    const cumSamples = cumLogRetPerSample.map((s) => s[t]);
    const p25 = percentile(samples, 25);
    const p50 = percentile(samples, 50);
    const p75 = percentile(samples, 75);
    const cp25 = percentile(cumSamples, 25);
    const cp50 = percentile(cumSamples, 50);
    const cp75 = percentile(cumSamples, 75);

    let realLR: number | null = null;
    let realEq: number | null = null;
    if (t < resp.n_steps) {
      realLR = resp.real_subsequent_obs[t][0];
      realCum += realLR;
      realEq = realCum;
    }

    return {
      step: t + 1,
      predLogRet_p25: p25,
      predLogRet_p50: p50,
      predLogRet_p75: p75,
      predLogRetSpread: p75 - p25,
      realLogRet: realLR,
      predEquity_p25: cp25,
      predEquity_p50: cp50,
      predEquity_p75: cp75,
      predEquitySpread: cp75 - cp25,
      realEquity: realEq,
    };
  });
}

export function DreamPlayer(): JSX.Element {
  const [episodes, setEpisodes] = useState<DreamEpisode[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [startStep, setStartStep] = useState<number>(256);
  const [actions, setActions] = useState<number[]>(Array(HORIZON).fill(2));
  const [running, setRunning] = useState(false);
  const [resp, setResp] = useState<DreamResponse | null>(null);

  const selected = useMemo(
    () => episodes?.find((e) => e.episode_id === selectedId),
    [episodes, selectedId],
  );

  useEffect(() => {
    listDreamEpisodes()
      .then((eps) => {
        setEpisodes(eps);
        if (eps.length > 0) setSelectedId(eps[0].episode_id);
      })
      .catch((err) => setError(err.message ?? String(err)));
  }, []);

  // When episode changes, snap startStep into a sensible range.
  useEffect(() => {
    if (selected) {
      const minStart = 1;
      const maxStart = Math.max(minStart, selected.n_steps - 1);
      setStartStep((s) => Math.min(Math.max(s, minStart), maxStart));
    }
  }, [selected]);

  const fanData = useMemo(
    () => (resp ? buildFanData(resp, actions.length) : null),
    [resp, actions.length],
  );

  async function runDream() {
    if (!selectedId) return;
    setRunning(true);
    setError(null);
    try {
      const r = await postDream({
        episode_id: selectedId,
        start_step: startStep,
        action_sequence: actions,
        n_samples: N_SAMPLES,
      });
      setResp(r);
    } catch (err) {
      setError((err as Error).message ?? String(err));
    } finally {
      setRunning(false);
    }
  }

  if (error && !episodes) {
    return (
      <div className="p-4 text-xs text-amber-300">
        Dream Player unavailable: {error}
      </div>
    );
  }
  if (!episodes) {
    return <div className="p-4 text-xs text-zinc-500">Loading episodes…</div>;
  }

  return (
    <div className="flex h-full flex-col gap-3 overflow-auto bg-zinc-950 p-3">
      <header className="flex flex-wrap items-end gap-3 rounded border border-zinc-800 bg-zinc-900 p-3">
        <label className="flex flex-col text-xs text-zinc-400">
          Episode
          <select
            value={selectedId ?? ""}
            onChange={(e) => setSelectedId(e.target.value)}
            className="mt-1 rounded bg-zinc-800 px-2 py-1 text-zinc-100"
          >
            {episodes.map((ep) => (
              <option key={ep.episode_id} value={ep.episode_id}>
                {ep.episode_id} — {ep.start_ts.slice(0, 16)} ({ep.n_steps} steps)
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col text-xs text-zinc-400">
          Start step
          <input
            type="number"
            min={1}
            max={selected?.n_steps ?? 1}
            value={startStep}
            onChange={(e) => setStartStep(Number(e.target.value))}
            className="mt-1 w-24 rounded bg-zinc-800 px-2 py-1 text-zinc-100"
          />
          <span className="text-[10px] text-zinc-600">
            of {selected?.n_steps ?? "—"}
          </span>
        </label>

        <button
          onClick={() => setActions(Array(HORIZON).fill(2))}
          className="rounded bg-zinc-800 px-3 py-1 text-xs text-zinc-200 hover:bg-zinc-700"
          title="Reset action sequence to '50% allocation for 50 steps'"
        >
          Reset actions (50%)
        </button>

        <button
          disabled={running || !selectedId}
          onClick={runDream}
          className="rounded bg-lime-600 px-3 py-1 text-xs text-zinc-900 hover:bg-lime-500 disabled:opacity-40"
        >
          {running ? "Dreaming…" : "Run dream"}
        </button>

        {error && <span className="text-xs text-amber-300">{error}</span>}
      </header>

      <ActionEditor actions={actions} onChange={setActions} />

      {fanData ? (
        <>
          <FanPanel
            title="Predicted log-return per step"
            data={fanData}
            yKeys={{
              p25: "predLogRet_p25",
              spread: "predLogRetSpread",
              p50: "predLogRet_p50",
              real: "realLogRet",
            }}
            yFormatter={(v: number) => v.toFixed(4)}
          />
          <FanPanel
            title="Cumulative log-return (compounded equity proxy)"
            data={fanData}
            yKeys={{
              p25: "predEquity_p25",
              spread: "predEquitySpread",
              p50: "predEquity_p50",
              real: "realEquity",
            }}
            yFormatter={(v: number) => v.toFixed(3)}
          />
        </>
      ) : (
        <div className="rounded border border-zinc-800 bg-zinc-900 p-4 text-xs text-zinc-500">
          Pick an episode, set a start step, and click <em>Run dream</em>.
        </div>
      )}
    </div>
  );
}

function ActionEditor({
  actions,
  onChange,
}: {
  actions: number[];
  onChange: (a: number[]) => void;
}): JSX.Element {
  return (
    <section className="rounded border border-zinc-800 bg-zinc-900 p-3">
      <header className="mb-2 text-xs uppercase tracking-wide text-zinc-400">
        Action sequence ({actions.length} steps · 0=0% → 4=100% alloc)
      </header>
      <div className="flex flex-wrap gap-1">
        {actions.map((a, i) => (
          <button
            key={i}
            title={`Step ${i + 1}: ${ALLOC_LABELS[a]}`}
            onClick={() => {
              const next = [...actions];
              next[i] = (a + 1) % ACTIONS.length;
              onChange(next);
            }}
            className="h-6 w-6 rounded text-[10px] tabular-nums text-zinc-100"
            style={{
              backgroundColor: `hsl(${100 + (a * 40)} 50% ${30 + a * 8}%)`,
            }}
          >
            {a}
          </button>
        ))}
      </div>
      <p className="mt-1 text-[10px] text-zinc-600">
        Click a cell to cycle the action (0 → 1 → 2 → 3 → 4 → 0).
      </p>
    </section>
  );
}

function FanPanel({
  title,
  data,
  yKeys,
  yFormatter,
}: {
  title: string;
  data: FanRow[];
  yKeys: { p25: string; spread: string; p50: string; real: string };
  yFormatter: (v: number) => string;
}): JSX.Element {
  return (
    <section className="rounded border border-zinc-800 bg-zinc-900 p-3">
      <header className="mb-2 flex items-baseline justify-between">
        <span className="text-xs uppercase tracking-wide text-zinc-400">
          {title}
        </span>
        <span className="text-[10px] text-zinc-500">
          shaded = p25..p75 | line = median | white = real
        </span>
      </header>
      <div className="h-56 w-full">
        <ResponsiveContainer>
          <ComposedChart data={data} margin={{ top: 4, right: 8, left: 8, bottom: 4 }}>
            <CartesianGrid strokeDasharray="2 2" stroke="#3f3f46" />
            <XAxis dataKey="step" stroke="#a1a1aa" tick={{ fontSize: 10 }} />
            <YAxis
              stroke="#a1a1aa"
              tick={{ fontSize: 10 }}
              tickFormatter={yFormatter}
            />
            <Tooltip
              contentStyle={{ backgroundColor: "#18181b", border: "1px solid #3f3f46" }}
              labelStyle={{ color: "#e4e4e7" }}
              formatter={yFormatter}
            />
            <Legend wrapperStyle={{ fontSize: 10, color: "#a1a1aa" }} />
            {/* Stacked Areas trick for IQR band: invisible base at p25 + visible spread */}
            <Area
              dataKey={yKeys.p25}
              stackId="iqr"
              stroke="none"
              fill="transparent"
              name=""
            />
            <Area
              dataKey={yKeys.spread}
              stackId="iqr"
              stroke="none"
              fill="#60a5fa"
              fillOpacity={0.3}
              name="IQR"
            />
            <Line
              dataKey={yKeys.p50}
              stroke="#60a5fa"
              strokeWidth={1.5}
              dot={false}
              name="median"
            />
            <Line
              dataKey={yKeys.real}
              stroke="#fafafa"
              strokeWidth={1.5}
              dot={false}
              name="real"
              connectNulls={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
