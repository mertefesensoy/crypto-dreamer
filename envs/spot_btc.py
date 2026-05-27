"""
Spot BTC/USDT trading environment.

Discrete action space: target allocation in {0%, 25%, 50%, 75%, 100%}.
Long-only. No leverage. Cash + BTC only.

Observation: 256-step window of normalized features + portfolio state.
Reward: log-return of equity minus turnover penalty.

This v1 uses a simple linear slippage model. We'll calibrate it against
real Binance L2 data in v2.
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = str(
    Path(os.environ.get("DREAMER_DATA", PROJECT_ROOT / "data" / "market.duckdb"))
)

WINDOW = 256
TARGETS = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
TAKER_FEE = 0.001  # 0.1%, Binance spot taker
# Penalty coefficient applied directly to fraction-of-equity turnover.
# Sized at ~10x the taker fee so it discourages churn without crushing exploration.
TURNOVER_PENALTY = 0.05
INITIAL_CASH = 10_000.0
# Decimated transport size for the obs window heatmap. 16 evenly-spaced rows
# from the 256-row window, including the most recent row.
FEATURE_TRANSPORT_ROWS = 16
FEATURE_TRANSPORT_INDICES = np.linspace(0, WINDOW - 1, FEATURE_TRANSPORT_ROWS, dtype=int)


@dataclass
class StepInfo:
    """Diagnostics emitted each step for the dashboard.

    Field order is the wire contract with the frontend. Add new fields at
    the end. Do not reorder or rename.
    """
    ts: pd.Timestamp
    price: float
    action: int
    target_alloc: float
    realized_alloc: float
    cash: float
    btc: float
    equity: float
    turnover: float
    fee_paid: float
    reward: float
    # Decimated obs window the agent saw at this step. base64(float16,
    # row-major, shape = (FEATURE_TRANSPORT_ROWS, 12)). Decoded by the
    # dashboard for the Internals heatmap.
    features_b64: str = field(default="")


def encode_features(window: np.ndarray) -> str:
    """Encode a (WINDOW, 12) feature window as decimated float16 base64."""
    decimated = window[FEATURE_TRANSPORT_INDICES].astype(np.float16, copy=False)
    return base64.b64encode(decimated.tobytes()).decode("ascii")


class SpotBTCEnv(gym.Env):
    """Single-asset spot environment over preloaded klines."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        episode_steps: int = 1440,  # 24h of 1m bars
        slippage_bps: float = 2.0,
        seed: int | None = None,
    ):
        super().__init__()
        self.symbol = symbol
        self.episode_steps = episode_steps
        self.slippage = slippage_bps / 10_000.0

        con = duckdb.connect(db_path, read_only=True)
        df = con.execute(
            "SELECT ts, open, high, low, close, volume FROM klines "
            "WHERE symbol = ? AND interval = ? ORDER BY ts",
            [symbol, interval],
        ).df()
        con.close()
        if len(df) < WINDOW + episode_steps + 10:
            raise ValueError(f"Not enough rows: {len(df)}. Run ingest first.")
        df = df.reset_index(drop=True)
        # DuckDB returns naive Timestamps, but the values are UTC-encoded by the
        # ingest. Localize so isoformat() emits a "+00:00" suffix and JS Date()
        # parses it as UTC instead of local time.
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        self.df = df
        self._precompute_features()

        # 12 features in window + 3 portfolio scalars
        self.observation_space = spaces.Dict(
            {
                "window": spaces.Box(-10, 10, shape=(WINDOW, 12), dtype=np.float32),
                "portfolio": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(len(TARGETS))
        self._rng = np.random.default_rng(seed)

    def _precompute_features(self) -> None:
        """Compute features once for the whole dataset. Indexed by row."""
        self._features = compute_feature_block(self.df)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        max_start = len(self.df) - self.episode_steps - 1
        self._t = int(self._rng.integers(WINDOW, max_start))
        self._t0 = self._t
        self._cash = INITIAL_CASH
        self._btc = 0.0
        self._equity_prev = INITIAL_CASH
        self._equity_peak = INITIAL_CASH
        return self._obs(), {}

    def step(self, action: int):
        target_alloc = float(TARGETS[action])
        price = float(self.df["close"].iloc[self._t])
        # Snapshot the features the agent saw before deciding (matches _obs slicing).
        decision_window = self._features[self._t - WINDOW : self._t]
        equity_before = self._cash + self._btc * price

        # Compute trade to reach target
        target_btc_value = target_alloc * equity_before
        delta_value = target_btc_value - self._btc * price
        turnover = abs(delta_value) / max(equity_before, 1e-9)

        fee_paid = 0.0
        if abs(delta_value) > 1e-6:
            # Slippage: pay slightly more on buys, receive slightly less on sells
            slip = self.slippage if delta_value > 0 else -self.slippage
            exec_price = price * (1 + slip)
            delta_btc = delta_value / exec_price
            fee_paid = abs(delta_value) * TAKER_FEE
            self._btc += delta_btc
            self._cash -= delta_value + fee_paid

        # Advance to next bar, mark equity at new close
        self._t += 1
        next_price = float(self.df["close"].iloc[self._t])
        equity = self._cash + self._btc * next_price
        self._equity_peak = max(self._equity_peak, equity)

        log_ret = np.log(max(equity, 1e-9) / max(self._equity_prev, 1e-9))
        reward = log_ret - TURNOVER_PENALTY * turnover
        self._equity_prev = equity

        terminated = False
        truncated = (self._t - self._t0) >= self.episode_steps
        # Episode ends if we lose 50% — defensive guardrail
        if equity < 0.5 * INITIAL_CASH:
            terminated = True

        info = StepInfo(
            ts=self.df["ts"].iloc[self._t],
            price=next_price,
            action=int(action),
            target_alloc=target_alloc,
            realized_alloc=(self._btc * next_price) / max(equity, 1e-9),
            cash=self._cash,
            btc=self._btc,
            equity=equity,
            turnover=turnover,
            fee_paid=fee_paid,
            reward=reward,
            features_b64=encode_features(decision_window),
        )
        return self._obs(), float(reward), terminated, truncated, {"step": info}

    def _obs(self) -> dict:
        window = self._features[self._t - WINDOW : self._t]
        price = float(self.df["close"].iloc[self._t])
        equity = self._cash + self._btc * price
        portfolio = np.array(
            [
                (self._btc * price) / max(equity, 1e-9),  # alloc
                self._cash / max(equity, 1e-9),           # cash ratio
                np.log(max(equity, 1e-9) / INITIAL_CASH), # log equity vs start
            ],
            dtype=np.float32,
        )
        return {"window": window, "portfolio": portfolio}


# --- feature helpers -------------------------------------------------------


def compute_feature_block(df: pd.DataFrame) -> np.ndarray:
    """Compute the 12 market features over a klines DataFrame.

    Returns shape (N, 12) float32, with NaN/Inf clipped to [-10, 10].
    Column order matches the wire contract — see the dict below.

    Used by SpotBTCEnv during reset and by Phase 5+ training code so a
    single source of truth determines what the agent and the world model
    see for any given (close, volume, high, low) row.
    """
    c = df["close"].to_numpy()
    v = df["volume"].to_numpy()
    log_ret = np.zeros_like(c)
    log_ret[1:] = np.log(c[1:] / c[:-1])

    def rolling_std(x: np.ndarray, w: int) -> np.ndarray:
        return pd.Series(x).rolling(w, min_periods=1).std().fillna(0).to_numpy()

    feats = {
        "log_ret": log_ret,
        "vol_5": rolling_std(log_ret, 5),
        "vol_15": rolling_std(log_ret, 15),
        "vol_60": rolling_std(log_ret, 60),
        "rsi_14": _rsi(c, 14),
        "macd": _macd(c),
        "vol_z": _zscore(v, 60),
        "hl_range": (df["high"] - df["low"]).to_numpy() / c,
        "close_norm": _zscore(c, 60),
        "ret_5": _sum_window(log_ret, 5),
        "ret_15": _sum_window(log_ret, 15),
        "ret_60": _sum_window(log_ret, 60),
    }
    block = np.stack([feats[k] for k in feats], axis=1).astype(np.float32)
    return np.nan_to_num(block, nan=0.0, posinf=10.0, neginf=-10.0)


# Stable feature column order — append-only, used by training pipelines
# that need to identify columns by name without re-reading this file.
FEATURE_NAMES = (
    "log_ret", "vol_5", "vol_15", "vol_60",
    "rsi_14", "macd", "vol_z", "hl_range",
    "close_norm", "ret_5", "ret_15", "ret_60",
)


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    diff = np.diff(close, prepend=close[0])
    up = np.where(diff > 0, diff, 0.0)
    down = np.where(diff < 0, -diff, 0.0)
    roll_up = pd.Series(up).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    roll_down = pd.Series(down).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    rs = roll_up / np.where(roll_down == 0, 1e-9, roll_down)
    rsi = 100 - 100 / (1 + rs)
    return (rsi - 50) / 50  # center on 0


def _macd(close: np.ndarray) -> np.ndarray:
    s = pd.Series(close)
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26).to_numpy()
    return macd / np.where(close == 0, 1, close)


def _zscore(x: np.ndarray, w: int) -> np.ndarray:
    s = pd.Series(x)
    m = s.rolling(w, min_periods=1).mean()
    sd = s.rolling(w, min_periods=1).std().fillna(1e-9)
    return ((s - m) / sd).fillna(0).to_numpy()


def _sum_window(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=1).sum().fillna(0).to_numpy()
