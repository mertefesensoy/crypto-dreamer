# 2026-05-03 — v1 Phase 1: backend bring-up

## Problem / Motivation

Six starter files existed at the project root (`ingest.py`, `spot_btc.py`,
`run_random.py`, `api.py`, `pyproject.toml`, `README.md`) but had never
been run. Imports referenced packages (`from envs.spot_btc ...`) that did
not exist on disk, the reward formula in code disagreed with the README,
the FastAPI bridge used a deprecated `pubsub.close()` call, and the
DuckDB path was implicitly cwd-relative.

Goal of this phase: get the loop ingest → env → random agent → Redis →
FastAPI WebSocket emitting real `StepInfo` JSON frames, with smoke tests
in place so v2 work doesn't regress the env contract.

## What Changed

| File | Description |
| --- | --- |
| `data/__init__.py` (new) | Package marker for ingest module. |
| `data/ingest.py` (moved + edited) | Was at root. DB path now anchored from `__file__` and overridable via `DREAMER_DATA` env var. |
| `envs/__init__.py` (new) | Re-exports `SpotBTCEnv` and `StepInfo`. |
| `envs/spot_btc.py` (moved + edited) | Was at root. Default `db_path` now resolves from a project-root anchor. Reward formula corrected (see Math section). |
| `agents/__init__.py` (new) | Package marker for agent module. |
| `agents/run_random.py` (moved) | Was at root. Imports unchanged; the `from envs.spot_btc import SpotBTCEnv` now resolves correctly because of the package layout. |
| `serve/__init__.py` (new) | Package marker for the API. |
| `serve/api.py` (moved + rewritten) | Was at root. `pubsub.close()` → `pubsub.aclose()`, lifespan `redis.close()` → `aclose()`, `send_json` wrapped in defensive try/except so a half-closed socket cannot leak the pubsub subscription. Redis host/port now overridable via `DREAMER_REDIS_HOST`/`DREAMER_REDIS_PORT`. |
| `tests/__init__.py`, `tests/conftest.py` (new) | `synthetic_db` fixture builds a 5000-row 1-min GBM DuckDB file in `tmp_path`. |
| `tests/test_env_smoke.py` (new) | reset → 100 steps, asserts no NaN/Inf in obs or reward, verifies `StepInfo` shape, verifies obs window ends at `t-1` (causality), verifies truncation at `episode_steps`. |
| `tests/test_features.py` (new) | Direct tests of `_rsi`, `_macd`, `_zscore`, `_sum_window` on random and constant series; full feature block check on the synthetic DB. |
| `pyproject.toml` (rewritten) | Added `[build-system]`, `[tool.setuptools.packages.find]`, `[tool.pytest.ini_options]` with `asyncio_mode = "auto"`. Bumped versions to ones that ship Python 3.13 wheels (`numpy>=2.0`, `pandas>=2.2.3`, `duckdb>=1.0`, `torch>=2.5`, `redis>=5.0.1` for `aclose`, `fastapi>=0.115`). Added dev deps `pytest-asyncio`, `httpx`, `websockets`. |
| `.tools/redis/*` (new, gitignored later) | Portable Redis 5.0.14.1 Windows binary. Fallback for this dev machine where Docker, WSL, and Memurai-via-winget all fail (Memurai's MSI bombs because Windows Firewall service is stopped). |

## Implementation Approach

**Package layout.** Standard Python packaging with setuptools `find` set
to the four runtime packages (`data`, `envs`, `agents`, `serve`) and an
explicit exclude on `tests*`/`dashboard*`. `pip install -e ".[dev]"`
exposes the runtime packages on the path so `python -m data.ingest`,
`python -m agents.run_random`, and `uvicorn serve.api:app` all resolve.

**DB path resolution.** Both ingest and env compute
`PROJECT_ROOT = Path(__file__).resolve().parents[1]` and respect a
`DREAMER_DATA` env var override. Three lines per file, no helper module
— v1 is a scaffold and a `data.paths` module would be premature.

**WebSocket safety.** The pub/sub listener loop now wraps `send_json` in
`try/except (WebSocketDisconnect, RuntimeError, ConnectionError)` and
breaks on any of them; the `finally` block always calls
`pubsub.unsubscribe(...)` then `pubsub.aclose()` so subscriptions cannot
leak on the Redis side regardless of how the WS client died.

**Test fixture strategy.** Synthesise OHLCV with geometric Brownian
motion (μ=0, σ=0.001 per minute) into a real DuckDB at `tmp_path`. The
env opens it read-only just like in production. No mocking of the DB
or DataFrame — the tests exercise the same code path the agent does.

## Mathematical / Statistical Details

**Reward.**

```
r_t = log(equity_t / equity_{t-1})  -  k_turnover * turnover_t
```

where `k_turnover = 0.05` and
`turnover_t = |delta_value_t| / equity_{t-1}` is the fraction of equity
swapped this bar.

The starter code had `k_turnover * TAKER_FEE * turnover` with
`TAKER_FEE = 0.001`, giving an effective coefficient of 0.0005 — a token
nudge dwarfed by the actual fee already baked into `equity` via the
`_cash -= delta_value + fee_paid` accounting. The README claimed 0.5,
which would crush exploration. We picked `k_turnover = 0.05` (≈10× the
taker fee) to discourage churn without preventing the policy from
trading at all.

Sanity check from the demo run: random agent over 1440 1-minute bars
ended at $5528.11 from $10000 with cumulative reward −28.46. Decomposing,
log(5528/10000) ≈ −0.593 of the −28.46 came from realized log-returns
and ~−27.87 from turnover penalty. Average per-step turnover ≈
27.87 / (1440 × 0.05) ≈ 0.387, consistent with a uniform-random policy
flipping among 5 targets each minute.

**Observation timing (causality).** The window at step `t` is
`features[t-256:t]` (last index `t-1`). Trade fills at `close[t]`.
Reward is measured at `close[t+1]`. Decision uses bar-close information
from one bar ago and execution happens one bar later — the standard
next-bar-fill pattern, no look-ahead. `test_observation_window_excludes_current_bar`
asserts this and will fail if a future refactor accidentally widens the
slice.

**Feature normalization.** All 12 features are clipped to `[-10, 10]`
via `np.nan_to_num(..., posinf=10, neginf=-10)` so the env's
observation-space `Box(-10, 10, ...)` declaration is honoured.

## Design Decisions

- **Inline feature helpers vs. `data/features.py`.** Kept the four
  helpers (`_rsi`, `_macd`, `_zscore`, `_sum_window`) inside
  `envs/spot_btc.py`. Only one caller exists; extracting them is
  abstraction we don't need in v1.
- **Portable Redis zip vs. Docker/Memurai.** Docker isn't installed,
  Memurai's installer fails on a stopped Windows Firewall service which
  we cannot start without admin. Portable zip is reversible and
  self-contained. Phase 3's `docker-compose.yml` will still use the
  official `redis:7-alpine` image; the zip is a dev-machine workaround.
- **Stopping ingest at 145k rows.** Full 2-year ingest takes ~30+
  minutes. Stopped at 145k rows (≈3.3 months of 1m bars,
  2024-05-03 → 2024-08-11) which is 444× the env minimum
  `WINDOW + episode_steps + 10`. Sufficient for v1 demo; user can
  resume the full ingest later via the same `python -m data.ingest`
  command (INSERT OR IGNORE makes it idempotent on overlap).

## Verification

1. `uv run pytest -v` → 10 passed (env smoke, observation causality,
   truncation, feature helpers).
2. `redis-cli ping` → PONG.
3. `python -m data.ingest --years 2` ran for ~10 minutes, committed
   145,000 rows covering 2024-05-03 to 2024-08-11.
4. `uvicorn serve.api:app` started, `curl /health` → `{"ok": true}`.
5. Random agent for 1 × 24h episode, speed=20 steps/sec.
   Final equity 5528.11, 1440 steps.
6. Live WS sniffer at `ws://127.0.0.1:8000/ws` captured the first 6
   frames of the episode — every field of `StepInfo` arrives intact:

```
STEP ep= 0 step=   1 price=  52961.88 action=4 alloc=1.001 equity=   9961.43 reward=-0.053864
STEP ep= 0 step=   2 price=  53229.99 action=3 alloc=0.751 equity=   9996.25 reward=-0.009061
STEP ep= 0 step=   3 price=  53017.24 action=3 alloc=0.749 equity=   9966.27 reward=-0.003060
STEP ep= 0 step=   4 price=  52926.15 action=4 alloc=1.000 equity=   9946.15 reward=-0.014559
STEP ep= 0 step=   5 price=  52744.13 action=2 alloc=0.499 equity=   9923.08 reward=-0.027335
STEP ep= 0 step=   6 price=  52802.01 action=3 alloc=0.750 equity=   9928.26 reward=-0.012011
```

## Related Docs

- Plan: `~/.claude/plans/i-m-building-v1-of-crystalline-ladybug.md`
- Project README: `README.md` (will be rewritten in Phase 3)
