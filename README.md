# crypto-dreamer

Spot BTC/USDT reinforcement-learning trading scaffold.
v1 ships environment, random agent, Redis bus, FastAPI WebSocket bridge,
and a React dashboard. v2 will add a DreamerV3-style world model with an
RSSM, an actor, an IQN critic, and an iTransformer encoder.

## Stack

Python 3.11+, PyTorch (placeholder for v2), Gymnasium, DuckDB,
python-binance, Redis 5+, FastAPI, uvicorn. Frontend: Vite, React 18,
TypeScript strict, Tailwind 3, Zustand 5, Recharts, reactflow 11.

## Layout

```
crypto-dreamer/
  data/         kline ingestion (Binance to DuckDB)
  envs/         Gymnasium spot env, fee + slippage model, feature pipeline
  agents/       random agent now, Dreamer in v2
  serve/        FastAPI WebSocket bridge
  tests/        env smoke + feature pipeline tests
  dashboard/    Vite + React 18 SPA
  docs/         implementation docs (one per phase)
  .tools/       portable Redis Windows binary (gitignored)
  Dockerfile, docker-compose.yml, justfile, Makefile, scripts/dev.ps1
```

## Quick start (Docker)

Bring up redis + api + ui together:

```
docker compose up -d
```

In a separate terminal, run the agent against historical data:

```
python -m data.ingest --years 2          # one time, ~10-30 min
python -m agents.run_random --episode-hours 24 --speed 50 --episodes 1
```

Open the dashboard at http://127.0.0.1:5173.

## Quick start (no Docker, native dev)

If Docker is unavailable, the repo includes a portable Redis Windows
binary fetcher path; otherwise install Memurai or any Redis 5+.

```
just redis             # starts portable redis on :6379 (or docker fallback)
just ingest             # 2 years of 1m BTCUSDT klines into data/market.duckdb
just api                # uvicorn on :8000
just ui                 # vite on :5173
just agent              # 1x 24h episode at speed=20
```

`make <target>` works as a shim. Native PowerShell users:
`pwsh -File scripts/dev.ps1` runs redis, api, and ui in parallel jobs.

## Action space

Discrete, 5 buckets: target BTC allocation in {0%, 25%, 50%, 75%, 100%}
of equity. The env computes the trade needed to reach the target,
applies a 0.1% taker fee and a 2 bps linear slippage model. Long-only.
No leverage. Cash plus BTC.

## Reward

```
r_t = log(equity_t / equity_{t-1}) - 0.05 * turnover_t
```

`turnover_t` is fraction of equity swapped this bar. The 0.05
coefficient is roughly 10x the taker fee, sized to discourage churn
without crushing exploration. Fees and slippage are already accounted
for inside `equity` via the cash and BTC ledger.

## Redis schema

Two channels carry the contract between agent and dashboard. Field
names are stable. Add fields, never rename.

`dreamer:steps` (per env step):

```
{
  ts, price, action, target_alloc, realized_alloc,
  cash, btc, equity, turnover, fee_paid, reward,
  episode, step, action_probs
}
```

`dreamer:episodes` (per episode end):

```
{ episode, steps, total_reward, final_equity }
```

The FastAPI bridge subscribes to both and forwards JSON frames to any
WebSocket client connected at `ws://localhost:8000/ws`.

## Where to look in v2

- `envs/spot_btc.py` will gain a live-mode flag for Binance WebSocket
  streaming and a microstructure-calibrated slippage model.
- `agents/` will add `dreamer.py` (RSSM + actor + IQN critic + iTransformer
  encoder) replacing `run_random.py` as the default.
- `serve/api.py` will add a `/control` POST endpoint so the dashboard
  kill switch becomes a real signal back to the trainer.
- The dashboard's Training and Internals tabs hold placeholders tagged
  `v2`; replace with real loss curves, KL, entropy, attention maps.

## Verification

```
uv run pytest -q       # 10 tests, env smoke + feature pipeline
just typecheck          # tsc strict mode
docker compose up       # redis + api + ui green, agent run from host
```

See `docs/implementations/2026-05-03-v1-phase1-backend.md` and
`-phase2-dashboard.md` for the full per-file rationale.
