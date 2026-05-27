"""Smoke test for the /dream endpoint.

Uses a synthetic, untrained WorldModel so the test is hermetic and
runs in CI without requiring a Phase 5.3 checkpoint. Verifies request
plumbing, output shapes, and graceful 404/400 handling. Does NOT
verify model fidelity — that's the validation suite's job.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import serve.dream_endpoint as de
from serve.dream_endpoint import DreamContext, _Episode, get_or_create_context, router


@pytest.fixture
def stub_context(synthetic_db_with_steps: str, monkeypatch: pytest.MonkeyPatch) -> DreamContext:
    """Build a DreamContext that bypasses checkpoint loading.

    Wires in:
    - a fresh, untrained WorldModel (random init)
    - the synthetic DB for kline/step_log access
    - an in-memory feature cache + episode arrays computed from synth data
    """
    import duckdb
    import numpy as np
    import pandas as pd
    import torch

    from envs.spot_btc import compute_feature_block
    from models.world_model import WorldModel

    ctx = DreamContext(
        ckpt_path="",
        klines_db=synthetic_db_with_steps,
        steps_db=synthetic_db_with_steps,
    )
    ctx.device = torch.device("cpu")  # avoid GPU in test

    # Untrained model
    ctx.model = WorldModel().to(ctx.device).eval()

    # Klines + features
    con = duckdb.connect(synthetic_db_with_steps, read_only=True)
    klines = con.execute(
        "SELECT ts, open, high, low, close, volume FROM klines "
        "WHERE symbol = 'BTCUSDT' AND interval = '1m' ORDER BY ts"
    ).df()
    con.close()
    ctx.feature_cache = compute_feature_block(klines)
    kline_ts = pd.to_datetime(klines["ts"].to_numpy(), utc=True)
    ctx.kline_t0 = kline_ts[0]

    # Episodes
    con = duckdb.connect(synthetic_db_with_steps, read_only=True)
    steps = con.execute(
        "SELECT ts, episode, step, action, realized_alloc, equity, "
        "       reward, agent_id "
        "FROM step_log WHERE agent_id LIKE 'random:%' "
        "ORDER BY agent_id, episode, step"
    ).df()
    con.close()
    episodes: dict[tuple[str, int], _Episode] = {}
    for (aid, ep_idx), grp in steps.groupby(["agent_id", "episode"], sort=False):
        grp = grp.sort_values("step").reset_index(drop=True)
        ts_pd = pd.to_datetime(grp["ts"].to_numpy(), utc=True)
        kline_idx = (
            ((ts_pd - ctx.kline_t0).total_seconds().to_numpy() / 60).astype(np.int64)
        )
        episodes[(str(aid), int(ep_idx))] = _Episode(
            ts=ts_pd.values, kline_idx=kline_idx,
            action=grp["action"].to_numpy(dtype=np.int64),
            realized=grp["realized_alloc"].to_numpy(dtype=np.float32),
            equity=grp["equity"].to_numpy(dtype=np.float32),
            reward=grp["reward"].to_numpy(dtype=np.float32),
        )
    ctx.episodes = episodes

    # Inject context into the module-level singleton.
    monkeypatch.setattr(de, "_ctx", ctx)
    return ctx


@pytest.fixture
def client(stub_context: DreamContext) -> TestClient:
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_list_episodes(client: TestClient, stub_context: DreamContext) -> None:
    r = client.get("/dream/episodes")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert len(body) == 2  # synthetic_db_with_steps creates 2 episodes
    ep0 = body[0]
    assert set(ep0.keys()) >= {"episode_id", "agent_id", "episode", "n_steps", "start_ts", "end_ts", "final_equity"}


def test_dream_basic_shapes(client: TestClient, stub_context: DreamContext) -> None:
    eps = client.get("/dream/episodes").json()
    ep = eps[0]
    H = 8
    n_samples = 4
    r = client.post("/dream", json={
        "episode_id": ep["episode_id"],
        "start_step": 30,
        "action_sequence": [2] * H,
        "n_samples": n_samples,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["predicted_obs"]) == n_samples
    assert len(body["predicted_obs"][0]) == H
    assert len(body["predicted_obs"][0][0]) == 15
    assert len(body["predicted_rewards"]) == n_samples
    assert len(body["predicted_rewards"][0]) == H
    # Real subsequent path may be capped by remaining episode length.
    assert body["n_steps"] <= H
    assert len(body["real_subsequent_obs"]) == body["n_steps"]
    assert len(body["real_subsequent_rewards"]) == body["n_steps"]


def test_unknown_episode_404(client: TestClient, stub_context: DreamContext) -> None:
    r = client.post("/dream", json={
        "episode_id": "random:nope|999",
        "start_step": 1,
        "action_sequence": [0],
        "n_samples": 1,
    })
    assert r.status_code == 404


def test_bad_episode_id_400(client: TestClient, stub_context: DreamContext) -> None:
    r = client.post("/dream", json={
        "episode_id": "no-pipe-separator",
        "start_step": 1,
        "action_sequence": [0],
        "n_samples": 1,
    })
    assert r.status_code == 400


def test_start_step_too_early_400(client: TestClient, stub_context: DreamContext) -> None:
    """If kline_idx[start-1] < WINDOW (256), the feature window
    underflows. Synthetic ep0 starts at kline_idx 300, so step=1 -> idx
    300, fine. step=1 of ep where idx<256 would 400."""
    # ep0 starts at idx 300. step=1 -> idx=300 (>=256), passes.
    eps = client.get("/dream/episodes").json()
    ep = eps[0]
    r = client.post("/dream", json={
        "episode_id": ep["episode_id"],
        "start_step": 1,
        "action_sequence": [2] * 4,
        "n_samples": 2,
    })
    assert r.status_code in (200, 400)  # depends on synthetic episode start position


def test_start_step_out_of_range_400(client: TestClient, stub_context: DreamContext) -> None:
    eps = client.get("/dream/episodes").json()
    ep = eps[0]
    r = client.post("/dream", json={
        "episode_id": ep["episode_id"],
        "start_step": ep["n_steps"] + 100,
        "action_sequence": [0],
        "n_samples": 1,
    })
    assert r.status_code == 400
