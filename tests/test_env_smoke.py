"""Smoke test: env can reset and step many times without producing NaN/Inf."""
from __future__ import annotations

import base64

import numpy as np

from envs.spot_btc import (
    FEATURE_TRANSPORT_ROWS,
    WINDOW,
    SpotBTCEnv,
    StepInfo,
)


def test_reset_then_100_steps(synthetic_db: str) -> None:
    env = SpotBTCEnv(db_path=synthetic_db, episode_steps=200, seed=42)
    obs, _ = env.reset(seed=42)

    assert obs["window"].shape == (WINDOW, 12)
    assert obs["portfolio"].shape == (3,)
    assert np.isfinite(obs["window"]).all()
    assert np.isfinite(obs["portfolio"]).all()

    rewards: list[float] = []
    for i in range(100):
        action = i % env.action_space.n
        obs, reward, terminated, truncated, info = env.step(int(action))
        assert np.isfinite(reward), f"non-finite reward at step {i}: {reward}"
        assert np.isfinite(obs["window"]).all(), f"NaN/Inf in window at step {i}"
        assert np.isfinite(obs["portfolio"]).all(), f"NaN/Inf in portfolio at step {i}"

        step = info["step"]
        assert isinstance(step, StepInfo)
        assert np.isfinite(step.equity)
        assert np.isfinite(step.cash)
        assert np.isfinite(step.btc)
        assert np.isfinite(step.realized_alloc)
        assert 0.0 <= step.target_alloc <= 1.0
        # features_b64 wire payload must decode to (FEATURE_TRANSPORT_ROWS, 12) float16.
        assert step.features_b64
        decoded = np.frombuffer(
            base64.b64decode(step.features_b64), dtype=np.float16
        )
        assert decoded.shape == (FEATURE_TRANSPORT_ROWS * 12,)
        assert np.isfinite(decoded.astype(np.float32)).all()
        rewards.append(reward)

        if terminated or truncated:
            break

    assert len(rewards) == 100, "episode should not terminate within 100 steps"
    # No reward should explode; bound generously.
    assert max(abs(r) for r in rewards) < 1.0


def test_observation_window_excludes_current_bar(synthetic_db: str) -> None:
    """Causality check: at step t, the obs window must end at feature[t-1].

    If a future refactor accidentally includes feature[t] in the window,
    the agent gets to see the close it's about to trade at — that's
    look-ahead. Guard against the regression here.
    """
    env = SpotBTCEnv(db_path=synthetic_db, episode_steps=200, seed=42)
    env.reset(seed=42)
    t = env._t  # noqa: SLF001 — white-box assertion is the point of this test
    obs, _ = env._obs(), None  # noqa: SLF001
    last_window_row = obs["window"][-1]
    np.testing.assert_array_equal(last_window_row, env._features[t - 1])
    # And the feature at the current bar must NOT equal the last window row,
    # except by coincidence on a constant series. We check that they differ
    # at least somewhere across the 12 channels.
    feature_at_t = env._features[t]
    assert not np.array_equal(last_window_row, feature_at_t)


def test_episode_truncation(synthetic_db: str) -> None:
    env = SpotBTCEnv(db_path=synthetic_db, episode_steps=50, seed=42)
    env.reset(seed=42)
    truncated = False
    for _ in range(60):
        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            break
    assert truncated, "episode should truncate at episode_steps=50"
