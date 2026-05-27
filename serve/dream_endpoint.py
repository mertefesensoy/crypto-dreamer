"""POST /dream — imagine n_samples trajectories from a real episode start.

Request body:
    episode_id:       "<agent_id>|<episode_int>"  e.g. "random:1|42"
    start_step:       int   (1-indexed step within the episode)
    action_sequence:  list[int]   length 50, each ∈ [0,4]
    n_samples:        int   default 32

Response body:
    predicted_obs:        float[n_samples][50][15]
    predicted_rewards:    float[n_samples][50]
    real_subsequent_obs:  float[H][15]   (H ≤ 50, may be shorter if the
                                          chosen episode runs out)
    real_subsequent_rewards: float[H]
    n_steps:              int   = min(50, episode length - start_step)

The world model + episode arrays + feature cache are loaded once at
process start (or first request) into a `DreamContext` cached at module
level. The checkpoint path is read from `DREAMER_WORLD_MODEL_CKPT` env
var, defaulting to `checkpoints/world_model_full_best.ckpt`.

Phase 5.3 note: this module is *not* loaded against in-training
checkpoints. It boots cleanly without any model loaded if the env var
points to a missing file — the /dream route returns 503 in that case so
the rest of the API stays responsive.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from envs.spot_btc import INITIAL_CASH, WINDOW, compute_feature_block
from models.world_model import WorldModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)

DEFAULT_CKPT = str(PROJECT_ROOT / "checkpoints" / "world_model_full_best.ckpt")
DEFAULT_KLINES_DB = str(PROJECT_ROOT / "data" / "market_ro.duckdb")
DEFAULT_STEPS_DB = str(PROJECT_ROOT / "data" / "market.duckdb")


# ---------------- Pydantic schemas ----------------


class DreamRequest(BaseModel):
    episode_id: str
    start_step: int = Field(ge=1)
    action_sequence: list[int] = Field(min_length=1, max_length=200)
    n_samples: int = Field(default=32, ge=1, le=128)


class DreamResponse(BaseModel):
    predicted_obs: list[list[list[float]]]       # (n_samples, H, 15)
    predicted_rewards: list[list[float]]          # (n_samples, H)
    real_subsequent_obs: list[list[float]]        # (H_real, 15)
    real_subsequent_rewards: list[float]          # (H_real,)
    n_steps: int


# ---------------- Context (model + episodes + features) ----------------


@dataclass(slots=True)
class _Episode:
    ts: np.ndarray
    kline_idx: np.ndarray
    action: np.ndarray
    realized: np.ndarray
    equity: np.ndarray
    reward: np.ndarray


class DreamContext:
    """Singleton holding the loaded world model, feature cache, and per-episode arrays."""

    def __init__(self, ckpt_path: str, klines_db: str, steps_db: str):
        self.ckpt_path = ckpt_path
        self.klines_db = klines_db
        self.steps_db = steps_db
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: WorldModel | None = None
        self.feature_cache: np.ndarray | None = None
        self.episodes: dict[tuple[str, int], _Episode] | None = None
        self.kline_t0 = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def ensure_loaded(self) -> None:
        if self.is_loaded():
            return
        self._load()

    def _load(self) -> None:
        ckpt = Path(self.ckpt_path)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"world model checkpoint not found: {ckpt} "
                f"(set DREAMER_WORLD_MODEL_CKPT to override)"
            )

        log.info("Loading world model from %s", ckpt)
        if ckpt.suffix == ".ckpt":
            model = WorldModel.load_from_checkpoint(str(ckpt), map_location=self.device)
        else:
            model = WorldModel()
            state = torch.load(str(ckpt), map_location=self.device, weights_only=True)
            model.load_state_dict(state)
            model = model.to(self.device)
        model.eval()
        self.model = model

        log.info("Loading klines + computing features")
        con = duckdb.connect(self.klines_db, read_only=True)
        klines = con.execute(
            "SELECT ts, open, high, low, close, volume FROM klines "
            "WHERE symbol = 'BTCUSDT' AND interval = '1m' ORDER BY ts"
        ).df()
        con.close()
        self.feature_cache = compute_feature_block(klines)
        import pandas as pd
        kline_ts = pd.to_datetime(klines["ts"].to_numpy(), utc=True)
        self.kline_t0 = kline_ts[0]

        log.info("Loading step_log")
        con = duckdb.connect(self.steps_db, read_only=True)
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
                ((ts_pd - self.kline_t0).total_seconds().to_numpy() / 60)
                .astype(np.int64)
            )
            episodes[(str(aid), int(ep_idx))] = _Episode(
                ts=ts_pd.values,
                kline_idx=kline_idx,
                action=grp["action"].to_numpy(dtype=np.int64),
                realized=grp["realized_alloc"].to_numpy(dtype=np.float32),
                equity=grp["equity"].to_numpy(dtype=np.float32),
                reward=grp["reward"].to_numpy(dtype=np.float32),
            )
        self.episodes = episodes
        log.info("DreamContext ready: %d episodes", len(episodes))

    def list_episodes(self) -> list[dict[str, Any]]:
        if self.episodes is None:
            self.ensure_loaded()
        out = []
        assert self.episodes is not None
        for (aid, ep_idx), ep in sorted(self.episodes.items()):
            out.append({
                "episode_id": f"{aid}|{ep_idx}",
                "agent_id": aid,
                "episode": ep_idx,
                "n_steps": int(len(ep.kline_idx)),
                "start_ts": str(ep.ts[0]),
                "end_ts": str(ep.ts[-1]),
                "final_equity": float(ep.equity[-1]),
            })
        return out

    @torch.no_grad()
    def dream(
        self, episode_id: str, start_step: int,
        action_sequence: list[int], n_samples: int,
    ) -> DreamResponse:
        if self.episodes is None or self.model is None or self.feature_cache is None:
            self.ensure_loaded()
        assert self.episodes is not None
        assert self.model is not None
        assert self.feature_cache is not None

        try:
            aid, ep_str = episode_id.split("|", 1)
            ep_idx = int(ep_str)
        except (ValueError, IndexError):
            raise HTTPException(400, f"bad episode_id: {episode_id!r}")
        key = (aid, ep_idx)
        if key not in self.episodes:
            raise HTTPException(404, f"episode not found: {episode_id}")
        ep = self.episodes[key]
        n_ep = len(ep.kline_idx)
        if start_step < 1 or start_step > n_ep:
            raise HTTPException(400, f"start_step {start_step} out of range 1..{n_ep}")

        H = len(action_sequence)
        H_real = min(H, n_ep - start_step)  # we have ground truth for this many

        device = self.device
        model = self.model

        # Build the (256, 15) obs window at start_step.
        idx = int(ep.kline_idx[start_step - 1])
        if idx < WINDOW:
            raise HTTPException(400, "start_step too early — feature window underflows")
        market = self.feature_cache[idx - WINDOW : idx]                # (256, 12)
        realized_at_start = float(ep.realized[start_step - 1])
        equity_at_start = float(ep.equity[start_step - 1])
        portfolio = np.array(
            [
                realized_at_start,
                1.0 - realized_at_start,
                np.log(max(equity_at_start, 1e-9) / INITIAL_CASH),
            ],
            dtype=np.float32,
        )
        portfolio_b = np.broadcast_to(portfolio[None, :], (WINDOW, 3))
        obs_full = np.concatenate([market, portfolio_b], axis=-1)      # (256, 15)
        obs_t = torch.from_numpy(obs_full.copy()).to(device).unsqueeze(0)  # (1, 256, 15)

        # Encode + posterior at start.
        x_0 = model.encode_obs(obs_t)                                   # (1, d_model)
        prev_h, prev_z = model.rssm.initial_state(1, device)
        a0 = torch.zeros(1, dtype=torch.long, device=device)
        a_emb = model.action_embed(a0)
        h_0, z_0, _, _ = model.rssm.step(
            prev_z, prev_h, a_emb, x_0,
            is_first=torch.ones(1, dtype=torch.bool, device=device),
        )

        # Replicate to n_samples for parallel rollout.
        h = h_0.repeat(n_samples, 1)
        z = z_0.repeat(n_samples, 1)

        actions = torch.tensor(action_sequence, dtype=torch.long, device=device)
        # Imagine forward H steps using the prior; sample stochastically.
        pred_obs = torch.zeros(n_samples, H, 15, device=device)
        pred_rew = torch.zeros(n_samples, H, device=device)
        for t in range(H):
            a_t = actions[t].expand(n_samples)
            a_emb = model.action_embed(a_t)
            gru_in = model.rssm.pre_gru(torch.cat([z, a_emb], dim=-1))
            h = model.rssm.gru(gru_in, h)
            prior_logits = model.rssm.prior_head(h).reshape(
                -1, model.rssm.n_latents, model.rssm.n_classes,
            )
            z = model.rssm.sample_st(prior_logits).reshape(-1, model.rssm.z_dim)
            feat = torch.cat([h, z], dim=-1)
            pred_obs[:, t] = model.decoder_head(feat)
            pred_rew[:, t] = model.reward_head.predict(model.reward_head(feat))

        # Real subsequent path: pull obs/reward from feature_cache + step_log.
        real_obs = np.zeros((H_real, 15), dtype=np.float32)
        real_rewards = np.zeros(H_real, dtype=np.float32)
        for j in range(H_real):
            step_idx = start_step - 1 + j  # next step's index
            kidx = int(ep.kline_idx[step_idx])
            row_market = self.feature_cache[kidx - 1]                # current bar = last row of window
            r = float(ep.realized[step_idx])
            eq = float(ep.equity[step_idx])
            real_obs[j, :12] = row_market
            real_obs[j, 12] = r
            real_obs[j, 13] = 1.0 - r
            real_obs[j, 14] = float(np.log(max(eq, 1e-9) / INITIAL_CASH))
            real_rewards[j] = float(ep.reward[step_idx])

        return DreamResponse(
            predicted_obs=pred_obs.float().cpu().tolist(),
            predicted_rewards=pred_rew.float().cpu().tolist(),
            real_subsequent_obs=real_obs.tolist(),
            real_subsequent_rewards=real_rewards.tolist(),
            n_steps=H_real,
        )


# ---------------- FastAPI router ----------------

_ctx: DreamContext | None = None


def get_or_create_context() -> DreamContext:
    global _ctx
    if _ctx is None:
        _ctx = DreamContext(
            ckpt_path=os.environ.get("DREAMER_WORLD_MODEL_CKPT", DEFAULT_CKPT),
            klines_db=os.environ.get("DREAMER_KLINES_DB", DEFAULT_KLINES_DB),
            steps_db=os.environ.get("DREAMER_STEPS_DB", DEFAULT_STEPS_DB),
        )
    return _ctx


router = APIRouter()


@router.get("/dream/episodes")
def list_dream_episodes() -> list[dict[str, Any]]:
    ctx = get_or_create_context()
    try:
        ctx.ensure_loaded()
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    return ctx.list_episodes()


@router.post("/dream", response_model=DreamResponse)
def post_dream(req: DreamRequest) -> DreamResponse:
    ctx = get_or_create_context()
    try:
        ctx.ensure_loaded()
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    return ctx.dream(req.episode_id, req.start_step,
                     req.action_sequence, req.n_samples)
