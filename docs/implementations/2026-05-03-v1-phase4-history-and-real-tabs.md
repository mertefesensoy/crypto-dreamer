# 2026-05-03 — v1 Phase 4: history, observation publishing, real Training & Internals content

## Problem / Motivation

Phase 2's dashboard worked for live tail, but a refresh during an
episode wiped the chart. Training and Internals tabs were placeholders.
Phase 4 closes both gaps:

- Replay history via Redis Streams + DuckDB step_log so the dashboard
  can hydrate on mount and survive reconnects.
- Real content for Training (episode summary, reward decomposition,
  action histogram, turnover and entropy) and Internals (obs window
  heatmap, portfolio state stacked area, verified architecture pulse).

The reward decomposition is also the headline diagnostic the user asked
for: with the random agent it should show penalty area dominating
log-return area, confirming that turnover penalty is doing its work.

## What Changed

| File | Description |
| --- | --- |
| `envs/spot_btc.py` | `StepInfo` gains `features_b64: str` field (appended). `encode_features` helper takes the 256x12 obs window, picks 16 evenly-spaced rows via `np.linspace(0, WINDOW-1, 16, dtype=int)` (last row is current), casts to float16, base64-encodes the row-major byte buffer (~512 chars per step). `step()` snapshots `decision_window = self._features[t-WINDOW:t]` before mutation so the published features match what the agent saw. |
| `agents/run_random.py` | Dual-write: `r.publish` (live tail) plus `r.xadd("dreamer:steps:hist", maxlen=5_000_000, approximate=True)` and `r.xadd("dreamer:episodes:hist", ...)`. Adds `agent_id` to step + episode payloads (default `f"random:{seed}"`, `--agent-id` flag overrides). DuckDB writer opened after env construction (env closes its read-only handle); `step_log` schema: `(ts, episode, step, action, target_alloc, realized_alloc, equity, reward, turnover, fee_paid, agent_id)`. |
| `serve/api.py` | Two new GET endpoints: `/episodes` (XRANGE on `dreamer:episodes:hist`, returns oldest-first list of summaries) and `/history?episode=N&count=M&before_step=K` (single XREVRANGE call sized to `min(max(count*4, 5000), 250000)`, filtered by episode in Python). Single-pass scan replaces a multi-page cursor walk after we found Redis 5.0 does not accept the `(` exclusive ID prefix that 6.2 added. |
| `tests/test_env_smoke.py` | Asserts `step.features_b64` is non-empty, decodes back to a `(16*12,)` float16 array with all-finite values. |
| `dashboard/package.json` | `d3-scale@^4`, `d3-scale-chromatic@^3.1` added to dependencies (≈30 KB total, no ML libs). `@types/d3-scale*` to dev deps. |
| `dashboard/src/types.ts` | `StepInfo` mirrors the new Python field order: `features_b64` appended, `agent_id` appended after `action_probs`. `EpisodeSummary` gains optional `agent_id`. |
| `dashboard/src/store/useDreamerStore.ts` | Ring buffer raised from 1500 to 50_000. New `seedHistory(episode, steps)` and `seedEpisodeSummaries(summaries)` actions. `pushStep` deduplicates against history-seeded steps by step-number so pre-fill plus live-tail does not double-count. New `hydrated` boolean flag. |
| `dashboard/src/hooks/useHistoryHydration.ts` (new) | On mount, fetches `/episodes`, seeds summaries; picks the latest episode and fetches `/history?episode=N&count=2000`, seeds steps and sets it active. Uses `AbortController` so React Strict Mode's double-mount cleanup can cancel the first run cleanly. Exports `loadEarlier(episode, beforeStep, count)` for the future "Load earlier" button. |
| `dashboard/src/lib/decode.ts` (new) | `decodeFeaturesB64(b64, rows, cols)` returns a `Float32Array`. Uses `Float16Array` directly when available (Chrome ≥ 118, Firefox ≥ 122); otherwise falls back to a 14-line manual `halfToFloat` decoder following the IEEE 754 binary16 spec. |
| `dashboard/src/lib/stats.ts` (new) | `decomposeReward(s)` -> `{logRet, penalty}`, `entropy(probs)` (Shannon, nats), `computeEpisodeStats(steps, episode)` (steps / total reward / final equity / max drawdown / mean turnover / mean abs log-return), `actionHistogram(steps, n=5)`. Uses `TURNOVER_PENALTY = 0.05` from the new `lib/constants.ts`. |
| `dashboard/src/lib/constants.ts` (new) | `TURNOVER_PENALTY`, `FEATURE_TRANSPORT_ROWS`, `FEATURE_COLS`, `FEATURE_NAMES`. Mirrors the Python env. |
| `dashboard/src/App.tsx` | Mounts `useHistoryHydration()` once at the top, before the WS hook. |
| `dashboard/src/components/Training.tsx` | Rewritten. Top: episode summary table. Middle: 3-up grid (reward decomposition / action histogram / turnover & entropy). Bottom: 4 v2 placeholders (world model loss, KL divergence, actor loss, critic loss). The mock Replay donut is removed — fake metrics are misleading. |
| `dashboard/src/components/training/EpisodeSummaryTable.tsx` (new) | Clickable rows; clicking sets the active episode. Columns: ep, agent, steps, total r, final equity, max DD, mean turnover, mean \|log_ret\|. Tone-coloured cells for total reward (lime/red) and max DD (red). |
| `dashboard/src/components/training/RewardDecomposition.tsx` (new) | Recharts `ComposedChart` with `stackOffset="sign"` so positive log-return area stacks above zero and negative turnover-penalty area stacks below. Footer prints the live ratio `Σ\|penalty\| / Σ\|log_ret\|` over the visible window. |
| `dashboard/src/components/training/ActionHistogram.tsx` (new) | 5 bars, one per allocation bucket, coloured with `ACTION_COLORS`. |
| `dashboard/src/components/training/TurnoverEntropy.tsx` (new) | Two-axis line chart: amber turnover (left, 0–100%), cyan entropy (right, 0–log(5) ≈ 1.609). Random agent's entropy stays flat at log(5); future stochastic policies will dip below. |
| `dashboard/src/components/Internals.tsx` | Two-column layout. Left column: Architecture diagram (top), obs window heatmap (middle), portfolio state stacked area (bottom). Right column: 3 v2 placeholders (encoder attention, latent UMAP, live activations). |
| `dashboard/src/components/internals/ObsWindowHeatmap.tsx` (new) | Decodes `features_b64` to a 16×12 matrix; renders one `<rect>` per cell with `d3-scale.scaleDiverging(t => interpolateRdBu(1-t)).domain([-max, 0, max])` so negative is blue, positive is red, white at zero. Domain bounds re-derive on each step from the matrix's max absolute value, capped to [0.1, 5] so a single outlier does not wash out the rest. Right-side legend shows the latest (current) row's values per feature. |
| `dashboard/src/components/internals/PortfolioState.tsx` (new) | Stacked area, cash% (slate) and btc% (lime), 600-step window. Always sums to 100%. |
| `dashboard/src/components/internals/ArchitectureGraph.tsx` | Unchanged. Verified pulse works: `pulseTick` subscription increments on every live step, useEffect re-fires `setPulsing(true)` and schedules a 700 ms `setPulsing(false)`. Edges use `animated: pulsing` and `stroke: pulsing ? "#a3e635" : "#52525b"` so they switch to lime + marching-ants while live frames arrive. End-to-end check: ran agent at speed=8 for 180 s, observed `pulseTick = 683` over the run. |
| `dashboard/src/index.css` | `@import "reactflow/dist/style.css";` moved to the top of the file (must precede `@tailwind` directives per CSS spec). |
| `dashboard/src/components/training/ReplayDonut.tsx` | Deleted. Mock data was misleading; replaced by the real episode summary table. |

## Implementation Approach

**Dual-write history.** Live pub/sub stays as the low-latency tail.
Redis Streams hold replay-able history (capped at 5M entries, ~28 hours
of 50 sps). DuckDB `step_log` is the durable column store for offline
analysis. The agent writes to all three; the API only knows about
streams (the dashboard's needs); DuckDB is for Python notebooks and
the v2 trainer's experience replay.

**Hydration pattern.** On dashboard mount, fetch `/episodes` (cheap)
then `/history?episode=latest&count=2000` (full episode for a 24h run
at 1m bars). Seed the store. The WebSocket hook then takes over for
live tail. `pushStep` deduplicates against the last seeded step so the
overlap window between hydration and first WS frame doesn't double-count.

**Strict-Mode-safe abort.** React 18 Strict Mode mounts effects twice
in dev. Without an `AbortController`, the first mount's in-flight
fetches log "Failed to fetch" errors when their cleanup tears them
down. Hook now wraps both fetches in the same controller; cleanup
calls `ctrl.abort()`; the catch swallows AbortError and any TypeError
where `signal.aborted` is true.

**Float16 transport.** 16 timesteps × 12 features × 2 bytes = 384 raw
bytes ≈ 512 base64 chars per step. At 50 sps that's 25 KB/s extra over
the WebSocket. Decoder uses native `Float16Array` if present, else a
manual halfToFloat. The decimation index set
`np.linspace(0, WINDOW-1, 16, dtype=int)` always includes index 0 and
WINDOW-1 so the last row of the heatmap is always the current bar's
feature row.

**Heatmap colormap.** `scaleDiverging(t => interpolateRdBu(1 - t))` so
the conventional "negative = cool blue, positive = warm red" reading
holds. d3 ships RdBu in the opposite orientation, so we invert `t`.
Domain re-derives per render from the matrix's max absolute value so
the colormap stretches to whatever range the data uses, capped to
[0.1, 5] so a single outlier doesn't wash out the rest.

## Mathematical / Statistical Details

**Reward decomposition.** Given `reward = log_ret − 0.05 × turnover`,
the dashboard computes the inverse client-side: `penalty = −0.05 × turnover`,
`log_ret = reward − penalty`. The chart stacks positive log-return area
above zero and the negative penalty area below zero
(`stackOffset="sign"`). The displayed ratio is
`Σ|penalty| / Σ|log_ret|` over the visible window. Random-agent runs
sit around 30–40× — penalty dominates by an order of magnitude or more,
which is the diagnostic confirmation that the penalty is sized correctly
(not silently zero, not silently overwhelming).

**Policy entropy** in nats: `H = −Σ p_i log p_i`. Random agent emits
uniform `[0.2, 0.2, 0.2, 0.2, 0.2]` so entropy is constant
`log 5 ≈ 1.609`. Future stochastic policies will dip below; v2's actor
will land somewhere between log(5) (max exploration) and 0 (degenerate).

**Max drawdown.** Online: track running `peak` of equity, dd_t =
`(peak − equity_t) / peak`. Max DD over the episode = max of all dd_t.

**Action histogram.** Plain bin counts over the 5 buckets, with the
fraction reported in the tooltip (`count (xx.x%)`).

## Design Decisions

- **One global stream, not per-episode.** Per-episode streams would
  speed up `/history` queries but the user spec said one stream with
  MAXLEN. With random-agent throughput a single XREVRANGE pass is fast
  enough; we'll revisit when stream length becomes a bottleneck.
- **Single-pass /history, not paginated.** Redis 5.0.14 (the portable
  Windows zip we use) doesn't accept the `(` exclusive ID prefix that
  6.2 added, so cursor walks error out. v1 sizes one XREVRANGE call to
  `min(count*4, 250_000)` and trusts that's enough for the requested
  count. Spec'd ceiling of 250k means even pathological cases finish
  in ~100 ms.
- **Decoder fallback.** Some users will be on browsers without
  Float16Array. The 14-line halfToFloat keeps the dashboard working
  without a polyfill dep.
- **Heatmap with raw SVG, scale from d3.** d3-scale + d3-scale-chromatic
  give us the diverging colormap with proper perceptual uniformity, but
  the heatmap itself is plain SVG `<rect>`s — d3 selection / data join
  would be overkill for 192 cells.
- **Replay donut removed.** It showed mock data with a "mock" badge,
  which is the kind of fake metric that misleads at a glance. The real
  episode summary table replaces it cleanly.
- **Activate latest historical episode on hydrate.** When the dashboard
  reconnects mid-run, the user usually wants to see the run that's
  going. We pick `summaries[-1]`. If a fresh agent then publishes a
  *new* episode, `pushStep` does not auto-flip the active episode
  (would be jarring); the user clicks the row in the summary table.
- **Architecture pulse left as-is.** Verified working: pulseTick
  increments per live frame, useEffect re-arms the 700 ms timeout, edges
  go lime + animated. The static-screenshot perception was a session
  artefact.

## Verification

1. `uv run pytest -q` -> 10 passed (smoke + features still green
   with `features_b64` round-trip assertion added).
2. `npx tsc --noEmit` -> 0 errors with `strict`,
   `exactOptionalPropertyTypes`, `noUnusedLocals`, `noUnusedParameters`.
3. Run agent: `uv run python -m agents.run_random --episode-hours 1 --speed 200 --episodes 2 --seed 17`. Confirm:
   - `XLEN dreamer:steps:hist` = 120, `XLEN dreamer:episodes:hist` = 2.
   - `SELECT COUNT(*) FROM step_log` = 120, agent_id = `random:17`.
   - `curl http://127.0.0.1:8000/episodes` returns the two summaries.
   - `curl http://127.0.0.1:8000/history?episode=0&count=10` returns
     10 step events including `features_b64` and `agent_id`.
4. Hydration mid-episode: open dashboard, header shows
   `episodes 2 active 1` (latest hydrated), Live tab populated from
   history. Start a fresh agent at speed=8: header step counter starts
   incrementing; Internals architecture diagram pulses lime.
5. Reward-decomp diagnostic: with the random agent, the visible-window
   ratio `Σ|penalty| / Σ|log_ret|` reads 30-40x. Penalty area swamps
   log-return area exactly as expected.
6. Obs heatmap: 16x12 cells, last row = current bar. Colours read
   blue/white/red across the diverging domain. The right-side legend
   matches the bottom row.
7. Portfolio state: stacked area sums to 100%, oscillates between cash
   and BTC as the random agent flips allocation each minute.

## Bugs found and fixed during this phase

1. **Redis 5 cursor pagination.** `XREVRANGE stream (id - count N` is
   syntactically invalid on Redis 5; only Redis 6.2+ accepts the
   exclusive `(` prefix. Discovered when `/history?count=2000` returned
   500. Fixed by replacing the multi-page walk with a single
   appropriately-sized XREVRANGE.
2. **Hydration TypeError: seedSummaries is not a function.** Pulled the
   store actions via `useDreamerStore.getState().seedEpisodeSummaries`
   at hook-call time, then captured them in the useEffect closure. HMR
   replaced the store module mid-flight, leaving a stale reference. Fix:
   re-read getState() inside the async body, after each await.
3. **AbortError swallowed as TypeError "Failed to fetch".** Some browsers
   surface the same condition under both names. Hook now also checks
   `ctrl.signal.aborted` so the catch is quiet for any cleanup-driven
   error, regardless of which name the browser used.
4. **CSS `@import` order.** `@import "reactflow/dist/style.css"` placed
   after `@tailwind base;` triggered a build warning and prevented HMR
   on the CSS file. Moved to the top of `index.css`.

## Related Docs

- Phase 1: `docs/implementations/2026-05-03-v1-phase1-backend.md`
- Phase 2: `docs/implementations/2026-05-03-v1-phase2-dashboard.md`
- Phase 3: `docs/implementations/2026-05-03-v1-phase3-glue.md`
- Plan: `~/.claude/plans/i-m-building-v1-of-crystalline-ladybug.md`
