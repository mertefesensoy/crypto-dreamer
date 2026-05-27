"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_db(tmp_path: Path) -> str:
    """Build a small DuckDB with ~5000 synthetic 1m BTCUSDT klines.

    Enough rows for WINDOW=256 + a 200-step episode + buffer. Geometric
    Brownian motion at ~1% daily vol so returns are realistic.
    """
    db_path = tmp_path / "test.duckdb"
    n = 5000
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    log_ret = rng.normal(0.0, 0.001, n)
    close = 50_000.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    spread = rng.uniform(0.0, 0.002, n)
    high = np.maximum(close, open_) * (1 + spread)
    low = np.minimum(close, open_) * (1 - spread)
    volume = rng.uniform(10.0, 100.0, n)

    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE klines (
            symbol VARCHAR,
            interval VARCHAR,
            ts TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            quote_volume DOUBLE,
            trades INTEGER,
            PRIMARY KEY (symbol, interval, ts)
        )
        """
    )
    rows = [
        (
            "BTCUSDT",
            "1m",
            t.to_pydatetime(),
            float(o),
            float(h),
            float(l),
            float(c),
            float(v),
            float(v * c),
            100,
        )
        for t, o, h, l, c, v in zip(ts, open_, high, low, close, volume, strict=True)
    ]
    con.executemany(
        "INSERT INTO klines VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    con.close()
    return str(db_path)


@pytest.fixture
def synthetic_db_with_steps(synthetic_db: str) -> str:
    """Extends `synthetic_db` with a tiny `step_log` so the Phase 5.1
    datamodule has something to consume.

    Two synthetic episodes from agent `random:test`:
      - episode 0: 200 steps, truncated normally (final equity stays > 5000)
      - episode 1: 120 steps, early-terminated (final equity < 5000)

    Step timestamps land inside the synthetic kline window so feature
    lookups succeed.
    """
    import duckdb as _duck
    import numpy as _np
    import pandas as _pd

    rng = _np.random.default_rng(7)
    con = _duck.connect(synthetic_db)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS step_log (
            ts TIMESTAMP,
            episode INTEGER,
            step INTEGER,
            action INTEGER,
            target_alloc DOUBLE,
            realized_alloc DOUBLE,
            equity DOUBLE,
            reward DOUBLE,
            turnover DOUBLE,
            fee_paid DOUBLE,
            agent_id VARCHAR
        )
        """
    )

    klines_ts = con.execute("SELECT ts FROM klines ORDER BY ts").df()["ts"].to_list()
    # Episode 0: starts well after the WINDOW=256 warmup, runs 200 steps (truncated)
    ep0_start_idx = 300
    ep0_len = 200
    # Episode 1: starts later, runs 120 steps (early terminated)
    ep1_start_idx = 700
    ep1_len = 120

    rows = []
    for ep_idx, (start, length, terminate_low) in enumerate(
        [(ep0_start_idx, ep0_len, False), (ep1_start_idx, ep1_len, True)]
    ):
        equity = 10_000.0
        for step in range(1, length + 1):
            ts = klines_ts[start + step - 1]
            action = int(rng.integers(0, 5))
            realized = float(action) / 4.0  # 0.0, 0.25, 0.5, 0.75, 1.0
            reward = float(rng.normal(0.0, 0.005))
            equity *= float(_np.exp(reward))
            if terminate_low and step == length:
                equity = 4900.0  # force early-termination signal
            rows.append((
                ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                ep_idx, step, action, realized, realized, equity, reward,
                0.05, 0.001, "random:test",
            ))
    con.executemany(
        "INSERT INTO step_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows,
    )
    con.close()
    return synthetic_db
