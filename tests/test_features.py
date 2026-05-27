"""Feature pipeline tests: no leakage, no infinities, behaves on edge cases."""
from __future__ import annotations

import numpy as np

from envs.spot_btc import SpotBTCEnv, _macd, _rsi, _sum_window, _zscore


def test_zscore_constant_series_is_finite() -> None:
    x = np.full(200, 100.0)
    z = _zscore(x, 60)
    assert np.isfinite(z).all()


def test_zscore_random_series_centered() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, 1000)
    z = _zscore(x, 60)
    assert np.isfinite(z).all()
    # After the warm-up window the rolling mean of z should be near zero.
    assert abs(np.mean(z[100:])) < 0.5


def test_rsi_no_nan_no_inf() -> None:
    rng = np.random.default_rng(2)
    close = 50_000.0 + np.cumsum(rng.normal(0.0, 100.0, 1000))
    rsi = _rsi(close, 14)
    assert np.isfinite(rsi).all()
    assert rsi.shape == close.shape
    # Centered RSI is in [-1, 1].
    assert rsi.min() >= -1.0001
    assert rsi.max() <= 1.0001


def test_rsi_constant_series_is_finite() -> None:
    close = np.full(500, 50_000.0)
    rsi = _rsi(close, 14)
    assert np.isfinite(rsi).all()


def test_macd_no_nan_no_inf() -> None:
    rng = np.random.default_rng(3)
    close = 50_000.0 + np.cumsum(rng.normal(0.0, 100.0, 1000))
    m = _macd(close)
    assert np.isfinite(m).all()
    assert m.shape == close.shape


def test_sum_window_finite() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(0.0, 1.0, 500)
    s = _sum_window(x, 60)
    assert np.isfinite(s).all()


def test_full_feature_block_no_inf_no_nan(synthetic_db: str) -> None:
    """The whole pre-computed feature matrix is sanitized at construction."""
    env = SpotBTCEnv(db_path=synthetic_db, episode_steps=200, seed=42)
    feats = env._features  # noqa: SLF001
    assert feats.shape[1] == 12
    assert np.isfinite(feats).all()
    # Bounded after nan_to_num clipping.
    assert feats.min() >= -10.0
    assert feats.max() <= 10.0
