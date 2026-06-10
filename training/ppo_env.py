"""Env construction for the ADR-007 model-free PPO baseline.

Implements exactly the env-interface additions sanctioned by ADR-007
(docs/design/ARCHITECTURE.md Section 12):

- deterministic episode-start selection through ``reset(options={"start_row": int})``
  for evaluation;
- partition-aware start sampling for training (episode start ``s`` valid iff
  ``s >= 256`` and every bar in ``[s, s + episode_steps]`` is train-pure under
  the ``(day - 1) / days_in_month < 0.85`` UTC rule);
- ``episode_steps`` as a constructor argument (already existed on SpotBTCEnv;
  the B&H comparator uses 4320).

``step()`` is inherited from ``envs.spot_btc.SpotBTCEnv`` UNTOUCHED · fees,
slippage, reward, termination are byte-identical. The only divergence from
the parent ``__init__`` is that market data (klines df + feature block) is
injected from a shared, once-loaded ``MarketData`` instead of re-reading
DuckDB per env instance; Track B asserts the injected features equal a
pristine ``SpotBTCEnv``'s own computation.

Data binding per ADR-007 (A)/(G)(vi): everything here reads the frozen
snapshot ``data/market_ro.duckdb``. ``assert_snapshot_binding()`` refuses to
proceed if the ``DREAMER_DATA`` environment variable points anywhere else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import duckdb
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from envs.spot_btc import (
    INITIAL_CASH,
    TARGETS,
    WINDOW,
    SpotBTCEnv,
    compute_feature_block,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DB = PROJECT_ROOT / "data" / "market_ro.duckdb"
VAL_MONTH_FRAC = 0.85  # training/datamodule.py:396 · the single split rule

# Pinned snapshot identity (ADR-007 (A) / (G)(vi)).
EXPECTED_ROWS = 1_051_201


def assert_snapshot_binding() -> None:
    """HALT if DREAMER_DATA would redirect envs away from the frozen snapshot."""
    override = os.environ.get("DREAMER_DATA")
    if override is not None and Path(override).resolve() != SNAPSHOT_DB.resolve():
        raise RuntimeError(
            f"ADR-007 data binding violated: DREAMER_DATA={override!r} does not "
            f"point at the frozen snapshot {SNAPSHOT_DB}. Unset it or set it to "
            "the snapshot path."
        )


@dataclass
class MarketData:
    """Once-loaded snapshot market data shared across env instances.

    Invariants: rows ordered by ts ascending, gap-free 1-min bars,
    ``features = compute_feature_block(df)`` (the canonical pipeline),
    ``val_bar[i]`` True iff bar i falls on a UTC val day-of-month.
    """

    df: pd.DataFrame
    features: np.ndarray  # (N, 12) float32
    ts_seconds: np.ndarray  # (N,) int64
    val_bar: np.ndarray  # (N,) bool


def load_market_data(db_path: str | Path = SNAPSHOT_DB) -> MarketData:
    """Load klines once and precompute features + the val-partition bar mask."""
    assert_snapshot_binding()
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute(
        "SELECT ts, open, high, low, close, volume FROM klines "
        "WHERE symbol = 'BTCUSDT' AND interval = '1m' ORDER BY ts"
    ).df()
    con.close()
    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(
            f"snapshot row count {len(df)} != pinned {EXPECTED_ROWS} (ADR-007 (A))"
        )
    df = df.reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    features = compute_feature_block(df)

    ts_idx = pd.DatetimeIndex(df["ts"])
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    ts_seconds = ((ts_idx - epoch) // pd.Timedelta("1s")).to_numpy().astype(np.int64)
    day = ts_idx.day.to_numpy()
    dim = ts_idx.daysinmonth.to_numpy()
    val_bar = ((day - 1) / dim) >= VAL_MONTH_FRAC

    return MarketData(df=df, features=features, ts_seconds=ts_seconds, val_bar=val_bar)


def valid_train_start_rows(market: MarketData, episode_steps: int = 1440) -> np.ndarray:
    """Rows s where s >= WINDOW and every bar in [s, s + episode_steps] is train-pure.

    Mechanical form of the ADR-007 training-data constraint. Inclusive upper
    bar: step() marks equity at s + episode_steps, so that bar must be
    train-pure too.
    """
    n = len(market.val_bar)
    win = episode_steps + 1  # bars s .. s + episode_steps inclusive
    c = np.concatenate([[0], np.cumsum(market.val_bar.astype(np.int64))])
    # val-bar count inside [s, s + episode_steps] for each feasible s
    counts = c[win:] - c[: n - win + 1]  # index s in [0, n - win]
    valid = np.flatnonzero(counts == 0)
    valid = valid[valid >= WINDOW]
    return valid


class BaselineSpotEnv(SpotBTCEnv):
    """SpotBTCEnv with injected shared market data and deterministic starts.

    ``step()``, ``_obs()`` and all trading mechanics are inherited unchanged.
    ``reset`` adds two sanctioned behaviors: ``options={"start_row": int}``
    pins the episode start (evaluation), and ``start_rows`` restricts random
    sampling to a precomputed set (partition-aware training). With neither,
    it falls back to the parent's uniform rule.
    """

    def __init__(
        self,
        market: MarketData,
        episode_steps: int = 1440,
        slippage_bps: float = 2.0,
        seed: int | None = None,
        start_rows: np.ndarray | None = None,
    ):
        # Deliberately NOT calling SpotBTCEnv.__init__ (it re-reads DuckDB);
        # every attribute it sets is replicated here from the shared data.
        gym.Env.__init__(self)
        self.symbol = "BTCUSDT"
        self.episode_steps = episode_steps
        self.slippage = slippage_bps / 10_000.0
        self.df = market.df
        self._features = market.features
        self.observation_space = spaces.Dict(
            {
                "window": spaces.Box(-10, 10, shape=(WINDOW, 12), dtype=np.float32),
                "portfolio": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(len(TARGETS))
        self._rng = np.random.default_rng(seed)
        self._start_rows = start_rows

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        max_start = len(self.df) - self.episode_steps - 1
        if options is not None and "start_row" in options:
            start = int(options["start_row"])
            if not (WINDOW <= start <= max_start):
                raise ValueError(
                    f"start_row {start} outside feasible range [{WINDOW}, {max_start}]"
                )
        elif self._start_rows is not None:
            start = int(self._start_rows[self._rng.integers(0, len(self._start_rows))])
        else:
            start = int(self._rng.integers(WINDOW, max_start))

        self._t = start
        self._t0 = start
        self._cash = INITIAL_CASH
        self._btc = 0.0
        self._equity_prev = INITIAL_CASH
        self._equity_peak = INITIAL_CASH
        return self._obs(), {}


def make_training_envs(
    market: MarketData,
    n_envs: int,
    seed: int,
    episode_steps: int = 1440,
) -> list[BaselineSpotEnv]:
    """n_envs partition-aware training envs with independent spawned RNG seeds."""
    starts = valid_train_start_rows(market, episode_steps)
    if len(starts) == 0:
        raise RuntimeError("no valid train-pure episode starts found")
    seeds = np.random.SeedSequence(seed).spawn(n_envs)
    return [
        BaselineSpotEnv(
            market,
            episode_steps=episode_steps,
            seed=int(s.generate_state(1)[0]),
            start_rows=starts,
        )
        for s in seeds
    ]


def make_eval_env(market: MarketData, episode_steps: int = 1440) -> BaselineSpotEnv:
    """Deterministic-start eval env (use reset(options={'start_row': ...}))."""
    return BaselineSpotEnv(market, episode_steps=episode_steps, seed=0)
