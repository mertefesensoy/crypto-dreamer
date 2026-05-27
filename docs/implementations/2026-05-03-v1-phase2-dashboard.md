# 2026-05-03 — v1 Phase 2: React dashboard

## Problem / Motivation

Phase 1 produced live `StepInfo` frames over a WebSocket but no UI to
visualise them. Phase 2 builds the dashboard the user spec'd: three tabs
(Live / Training / Internals), single WebSocket connection feeding a
Zustand store keyed by episode, dark theme, information-dense layout, no
emojis. Training and Internals are scaffolding for v2 with explicit `v2`
badges; Internals' architecture diagram is real and pulses on every step.

## What Changed

| File | Description |
| --- | --- |
| `dashboard/package.json` (new) | Vite + React 18 + TS + Tailwind 3 + Zustand 5 + Recharts 2 + reactflow 11 + `@types/node`. |
| `dashboard/index.html` (new) | Sets `class="dark"`, `bg-zinc-950 text-zinc-200` on body. |
| `dashboard/tsconfig.json` (new) | Strict mode + `exactOptionalPropertyTypes` + `noUnusedLocals` + `noUnusedParameters`. Path alias `@/*` → `src/*`. Single config (collapsed the project-references setup that fights `noEmit`). |
| `dashboard/vite.config.ts` (new) | Server bound to `127.0.0.1:5173` (matches the API's CORS allowlist). |
| `dashboard/tailwind.config.ts` (new) | Class-based dark mode. Custom `action.0..4` palette mapped to the five allocation buckets (neutral grey at 0%, lime at 100%) for chart consistency. |
| `dashboard/postcss.config.cjs` (new) | Tailwind + Autoprefixer. |
| `dashboard/src/index.css` (new) | Tailwind base + reactflow base styles. `tnum 1` for tabular numerals on prices. |
| `dashboard/src/types.ts` (new) | TypeScript mirror of the Python `StepInfo` dataclass plus `EpisodeSummary` and `WsMessage` envelope. Stable contract — fields added, not renamed. |
| `dashboard/src/hooks/useDreamerStream.ts` (new) | Single-WS hook. Owns connection lifecycle, auto-reconnects on close after 1.5s, pushes parsed frames into the store. Mounted once at the App root; components subscribe to the store, not the socket. |
| `dashboard/src/store/useDreamerStore.ts` (new) | Zustand store. State keyed by episode in `episodes: Record<number, EpisodeBucket>`. Each bucket holds a bounded ring buffer (1500 most recent steps). Increments `pulseTick` on every step — drives the Internals diagram pulse. Exposes `selectActiveBucket` selector so charts subscribe to slice instead of the whole store. |
| `dashboard/src/lib/format.ts` (new) | Number/time formatters and the shared `ACTION_COLORS` / `ACTION_LABELS` arrays. |
| `dashboard/src/main.tsx` (new) | React 18 root. |
| `dashboard/src/App.tsx` (new) | Mounts the WS hook once, renders the active tab. |
| `dashboard/src/components/Shell.tsx` (new) | Header with tab bar + connection-state dot + step/episode counters. |
| `dashboard/src/components/Live.tsx` (new) | Two-column layout: main column has price chart, equity chart, KPI strip; sidebar has action probabilities, critic-quantile-fan placeholder, kill switch, episode meta. |
| `dashboard/src/components/live/PriceChart.tsx` (new) | Recharts `ComposedChart`: green `Bar` for realized allocation on a 0–100 % right axis; white `Line` for price on the left axis; coloured `ReferenceDot`s for buy markers (action up, action-coloured fill) and red dots for sell markers (action down). Caps history at the most recent 600 bars to keep frame rate sane. |
| `dashboard/src/components/live/EquityChart.tsx` (new) | Lime-on-zinc line chart with a dashed reference line at the $10,000 starting equity. |
| `dashboard/src/components/live/ActionProbs.tsx` (new) | Five horizontal bars, one per allocation bucket. Active action highlighted by row colour intensity. Falls back to uniform 20 % when no probs published. |
| `dashboard/src/components/live/QuantileFan.tsx` (new) | Flat dashed line, "v2" badge in panel header, caption explaining the IQN 9-quantile estimate is wired in v2. |
| `dashboard/src/components/live/KillSwitch.tsx` (new) | Two-stage UI: arm checkbox, then kill button. No remote-control channel exists yet so it's a no-op stub for v1. |
| `dashboard/src/components/Training.tsx` (new) | 3×3 grid: 5 placeholder cards (world model loss, KL, actor, critic, entropy) plus the replay donut. |
| `dashboard/src/components/training/PlaceholderCard.tsx` (new) | Reusable card with `v2` badge and a static decorative spark line. Not faking data — the line is a dashed pattern, not a fake metric. |
| `dashboard/src/components/training/ReplayDonut.tsx` (new) | Recharts `PieChart` showing mock replay-buffer composition (uniform / TD-error / novelty / on-policy). Explicit `mock` badge. |
| `dashboard/src/components/Internals.tsx` (new) | Two-column: architecture graph on the left, three placeholder cards on the right. |
| `dashboard/src/components/internals/ArchitectureGraph.tsx` (new) | reactflow diagram. Six nodes — Obs → Encoder → RSSM → {Actor, Critic} → SpotBTCEnv → (back to Obs). Edges animated and lime-coloured for ~700ms after every `pulseTick` change. |
| `envs/spot_btc.py` (edited) | One-line addition: `df["ts"] = pd.to_datetime(df["ts"], utc=True)` after loading the kline frame. Without this the JS side parsed naive ISO strings as local time and chart axes drifted by the user's UTC offset. |
| `.claude/launch.json` (new) | Registers the dashboard server with the Claude Code preview harness. |

## Implementation Approach

**Single WebSocket, fan-out via store.** A common mistake is one
`useEffect` with `new WebSocket(...)` per chart. We do exactly the
opposite: `useDreamerStream` is mounted once in `App`, owns the lifecycle
(open / message / error / close), and pushes frames into the Zustand
store. Charts read from the store via selectors. Result: one socket per
browser tab, charts only re-render on the slice they subscribe to.

**Bounded state.** Each episode bucket holds a ring buffer of 1500 most
recent steps. At 50 steps/sec for 24 h that's 4.3M frames, which would
crater the browser. Keeping the most recent 1500 covers the visible
chart window (600 bars cap) plus headroom and discards older steps. v2
will revisit this when scrubbable replay matters.

**Pulse mechanism.** Every `pushStep` increments `pulseTick`. The
architecture graph subscribes to `pulseTick` and re-derives its `edges`
array with `animated: pulsing` + a lime stroke for 700 ms after each
change. This is the only "real" thing on the Internals tab in v1; the
three placeholder cards are honest about being v2.

**Layout.** Tailwind dark theme. `font-mono` body so prices align in
columns; `tabular-nums` on numeric KPIs. Panels use `border + bg-zinc-900`,
no rounded-3xl-on-everything aesthetic. `grid-rows-1` on the outer two-
column grids so the children inherit full height instead of collapsing
to content (this was the bug surfaced during preview verification — a
single-row implicit grid sized rows by content, not by available height).

**Strict TypeScript.** `exactOptionalPropertyTypes: true` caught one
real type error during the typecheck (`signed?: number` rejecting
`undefined`); fixed with the explicit `number | undefined` annotation.
No `any` anywhere; every component is typed.

## Mathematical / Statistical Details

The dashboard does no math of its own beyond simple aggregations
(`Math.min`/`Math.max` for chart y-domain padding). The reward formula
shown in the KPI strip is the same `r = log_ret − 0.05 × turnover`
computed by the Python env (Phase 1 doc).

Buy/sell marker rule: `action[t] > action[t-1]` → buy (target alloc
went up), `action[t] < action[t-1]` → sell. Marker fill colour is the
new action's colour from `ACTION_COLORS`, so a buy from 25 % → 100 %
shows a near-white dot, while 25 % → 50 % shows mid-green. Sells are
always red.

Allocation bar height equals the realized allocation
`(_btc * next_price) / equity` in [0, 1], plotted on the right axis
0–100 %. The bar is drawn first, then the price line on top, then the
markers — so the price line is never occluded.

## Design Decisions

- **Recharts over D3 / Visx / nivo.** The user spec asked for Recharts
  by name and Recharts is good enough for static-shape time series.
  D3 would give pixel-level control we don't need in v1. Recharts
  doesn't support per-point Area opacity natively, hence the bar
  approximation for the allocation overlay.
- **`reactflow@^11.11.4` over `@xyflow/react@12`.** v11 is the most
  widely documented version and the API is stable. The newer xyflow
  rename is identical conceptually but has breaking imports.
- **Tailwind 3.4, not 4.** Tailwind 4 introduced CSS-first config that
  diverges from the rest of the React ecosystem. Stable `tailwind.config.ts`
  in v3 is enough for v1; we can move later.
- **Zustand v5.** Brought structural sharing and a leaner API. The
  store has no middleware (no devtools, no persist) — keeping it a
  scaffold per the user's "no premature abstractions" rule.
- **Single tsconfig, not project references.** Vite ships its config in
  TS and a referenced `tsconfig.node.json` fights with `noEmit: true`
  on the app config. Collapsed to one tsconfig that includes both
  `src` and `vite.config.ts`. Adds `@types/node` for `node:path` /
  `__dirname`.
- **`grid-rows-1` quirk.** A `grid grid-cols-X` without an explicit
  `grid-template-rows` gets a single auto row, so children collapse
  to content height. Fix is `grid-rows-1` (`repeat(1, minmax(0, 1fr))`)
  on the parent. Caught by inspecting the live preview, not by tests.
- **Timezone fix in env, not dashboard.** Naive timestamps would have
  required every chart to reparse with `+ 'Z'`. Localizing once at the
  env source means the JSON contract on the wire is correct UTC ISO,
  and any future client (Python notebook, mobile app) gets the right
  timestamp.

## Verification

1. `cd dashboard && npm install` — clean.
2. `npx tsc -p tsconfig.json --noEmit` — 0 errors.
3. `npm run dev` (via the preview harness) — Vite ready in 341 ms.
4. Curling `/` returns the index, `/src/main.tsx` returns transformed
   JSX with the React refresh runtime injected.
5. Browser console: only React DevTools info and Vite HMR debug. No
   errors, no warnings.
6. `redis-cli INFO stats` during an active run: input_kbps ≈ 4–32
   depending on agent speed, `pubsub_channels = 2` while the dashboard
   is open.
7. Visual: Live tab shows BTC price line, allocation bars, buy/sell
   markers, equity decline; sidebar shows action probability bars
   highlighted on the active action; episode meta updates per step.
   Training tab shows 5 v2 placeholders + replay donut. Internals tab
   shows the six-node architecture graph with edges pulsing lime on
   every published step.
8. Tab switching, kill-switch arm/disarm, kill-and-reset all
   interactive without console errors.
9. Reward signed colouring works: KPI `reward` cell goes red when
   negative, lime when positive (random agent stays mostly red).

## Related Docs

- Phase 1: `docs/implementations/2026-05-03-v1-phase1-backend.md`
- Plan: `~/.claude/plans/i-m-building-v1-of-crystalline-ladybug.md`
