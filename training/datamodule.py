"""Episode-aware Lightning DataModule for the Phase 5.2+ world model.

Reads:
    klines from `data/market_ro.duckdb` (snapshot)
    step_log from `data/market.duckdb` (live, no writers expected)

Builds a single in-memory feature cache (~50 MB fp32, full 2y of 12
market features) via the canonical `envs.spot_btc.compute_feature_block`,
groups `step_log` rows into per-episode arrays, splits each step into
train/val by month-fraction (last 15% of each calendar month → val),
and yields trajectory batches of shape:

    obs_window       (B, T, 256, 15)   T historical 256-bar windows + portfolio
    next_obs_window  (B, T, 256, 15)   shifted by 1
    action           (B, T)            int64
    reward           (B, T)            fp32
    continue_flag    (B, T)            bool — False at the LAST step of an
                                       early-terminated episode only
    is_first         (B, T)            bool — True only at episode-step 1

A subsequence cannot cross the train/val boundary; a (B,T) batch draws
B independent subsequences from possibly different episodes/agents,
weighted to give every calendar month roughly equal expected presence
per epoch (so 2024-05 isn't drowned out by 2026-04).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from envs.spot_btc import INITIAL_CASH, WINDOW, compute_feature_block

log = logging.getLogger(__name__)


@dataclass(slots=True)
class _EpisodeArrays:
    ts: np.ndarray            # datetime64[ns, UTC]
    kline_idx: np.ndarray     # int64, 1-step-aligned to feature cache rows
    action: np.ndarray        # int64
    realized: np.ndarray      # float32 (alloc fraction)
    equity: np.ndarray        # float32
    reward: np.ndarray        # float32
    cont: np.ndarray          # bool — continue flag
    is_first: np.ndarray      # bool


class TrajectoryDataset(Dataset):
    """Yields one trajectory subsequence per __getitem__.

    Each item produces tensors with the leading T dim = `T`. Batching
    via the default collate stacks them into (B, T, ...).
    """

    def __init__(
        self,
        starts: list[tuple[str, int, int]],
        episodes: dict[tuple[str, int], _EpisodeArrays],
        feature_cache: np.ndarray,
        seq_len: int,
        T: int,
    ):
        self.starts = starts
        self.episodes = episodes
        self.feature_cache = feature_cache
        self.seq_len = seq_len
        self.T = T

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        agent_id, episode, start = self.starts[i]
        ep = self.episodes[(agent_id, episode)]
        # Slice T+1 raw step_log rows starting at `start` (0-based).
        sl = slice(start, start + self.T + 1)
        kidx = ep.kline_idx[sl]                              # (T+1,)

        # Per-step 256-bar market window.
        market = np.empty((self.T + 1, self.seq_len, 12), dtype=np.float32)
        for j, k in enumerate(kidx):
            market[j] = self.feature_cache[k - self.seq_len : k]

        # Per-step portfolio (3 scalars), broadcast across the 256-row window.
        realized = ep.realized[sl].astype(np.float32)
        equity = ep.equity[sl].astype(np.float32)
        portfolio = np.stack(
            [
                realized,
                1.0 - realized,
                np.log(np.maximum(equity, 1e-9) / INITIAL_CASH).astype(np.float32),
            ],
            axis=-1,
        )                                                    # (T+1, 3)
        portfolio_b = np.broadcast_to(
            portfolio[:, None, :], (self.T + 1, self.seq_len, 3)
        )

        full = np.concatenate([market, portfolio_b], axis=-1)  # (T+1, 256, 15)

        return {
            "obs_window":      torch.from_numpy(full[: self.T].copy()),
            "next_obs_window": torch.from_numpy(full[1 : self.T + 1].copy()),
            "action":          torch.from_numpy(ep.action[sl][: self.T].astype(np.int64)),
            "reward":          torch.from_numpy(ep.reward[sl][: self.T].astype(np.float32)),
            "continue_flag":   torch.from_numpy(ep.cont[sl][: self.T].astype(bool)),
            "is_first":        torch.from_numpy(ep.is_first[sl][: self.T].astype(bool)),
        }


class SpotBTCDataModule(L.LightningDataModule):
    def __init__(
        self,
        klines_db: str | Path,
        steps_db: str | Path,
        batch_size: int = 16,
        T: int = 64,
        seq_len: int = WINDOW,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        val_month_frac: float = 0.85,
        agent_id_pattern: str = "random:%",
        num_workers: int = 0,
        seed: int = 42,
        max_episodes: int | None = None,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.klines_db = str(klines_db)
        self.steps_db = str(steps_db)
        self.batch_size = batch_size
        self.T = T
        self.seq_len = seq_len
        self.symbol = symbol
        self.interval = interval
        self.val_month_frac = val_month_frac
        self.agent_id_pattern = agent_id_pattern
        self.num_workers = num_workers
        self.seed = seed
        self.max_episodes = max_episodes
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0

        self._train_ds: TrajectoryDataset | None = None
        self._val_ds: TrajectoryDataset | None = None
        self._train_weights: np.ndarray | None = None

        # Diagnostics for STOP POINT 3.
        self.month_summary: pd.DataFrame | None = None
        self.episode_count: int = 0
        self.terminated_count: int = 0

    def setup(self, stage: str | None = None) -> None:
        # --- klines + feature cache --------------------------------------
        con = duckdb.connect(self.klines_db, read_only=True)
        klines = con.execute(
            "SELECT ts, open, high, low, close, volume FROM klines "
            "WHERE symbol = ? AND interval = ? ORDER BY ts",
            [self.symbol, self.interval],
        ).df()
        con.close()

        feats = compute_feature_block(klines)                # (N, 12)
        kline_ts = pd.to_datetime(klines["ts"].to_numpy(), utc=True)
        kline_t0 = kline_ts[0]
        log.info(
            "Klines: %d rows, span %s -> %s, features %s",
            len(klines), kline_ts[0], kline_ts[-1], feats.shape,
        )

        # --- step_log + per-episode arrays -------------------------------
        con = duckdb.connect(self.steps_db, read_only=True)
        steps = con.execute(
            "SELECT ts, episode, step, action, realized_alloc, equity, "
            "       reward, agent_id "
            "FROM step_log "
            "WHERE agent_id LIKE ? "
            "ORDER BY agent_id, episode, step",
            [self.agent_id_pattern],
        ).df()
        con.close()
        log.info(
            "Step log: %d rows, %d agents, %d (agent,episode) pairs",
            len(steps), steps["agent_id"].nunique(),
            steps.groupby(["agent_id", "episode"]).ngroups,
        )

        # Optionally cap to first N episodes (for smoke runs).
        if self.max_episodes is not None:
            unique_pairs = (
                steps[["agent_id", "episode"]]
                .drop_duplicates()
                .head(self.max_episodes)
            )
            keep = steps.merge(unique_pairs, on=["agent_id", "episode"])
            steps = keep
            log.info("max_episodes=%d -> %d step_log rows kept",
                     self.max_episodes, len(steps))

        episodes: dict[tuple[str, int], _EpisodeArrays] = {}
        terminated_count = 0
        for (aid, ep_idx), grp in steps.groupby(["agent_id", "episode"], sort=False):
            grp = grp.sort_values("step").reset_index(drop=True)
            n = len(grp)
            ts_pd = pd.to_datetime(grp["ts"].to_numpy(), utc=True)
            # Convert ts -> kline row index; klines are gap-free 1-min,
            # so offset/60s is the exact row index.
            kline_idx = (
                ((ts_pd - kline_t0).total_seconds().to_numpy() / 60).astype(np.int64)
            )

            cont = np.ones(n, dtype=bool)
            if float(grp["equity"].iloc[-1]) < 0.5 * INITIAL_CASH:
                cont[-1] = False
                terminated_count += 1
            is_first = np.zeros(n, dtype=bool)
            if int(grp["step"].iloc[0]) == 1:
                is_first[0] = True

            episodes[(aid, ep_idx)] = _EpisodeArrays(
                ts=ts_pd.values,
                kline_idx=kline_idx,
                action=grp["action"].to_numpy(dtype=np.int64),
                realized=grp["realized_alloc"].to_numpy(dtype=np.float32),
                equity=grp["equity"].to_numpy(dtype=np.float32),
                reward=grp["reward"].to_numpy(dtype=np.float32),
                cont=cont,
                is_first=is_first,
            )
        self.episode_count = len(episodes)
        self.terminated_count = terminated_count

        # --- candidate subsequence starts, partitioned by month ----------
        train_starts: list[tuple[str, int, int]] = []
        val_starts: list[tuple[str, int, int]] = []
        train_ym: list[str] = []
        val_ym: list[str] = []

        for (aid, ep_idx), ep in episodes.items():
            ts = pd.DatetimeIndex(ep.ts)
            day = ts.day.to_numpy()
            dim = ts.daysinmonth.to_numpy()
            partition = ((day - 1) / dim >= self.val_month_frac).astype(np.int8)
            ym = ts.strftime("%Y-%m").to_numpy()
            n = len(ts)
            # Need T+1 raw rows; also feature window must fit (kline_idx >= seq_len).
            for start in range(0, n - self.T):
                window_partition = partition[start : start + self.T + 1]
                if window_partition.min() != window_partition.max():
                    continue                         # straddles boundary
                if ep.kline_idx[start] < self.seq_len:
                    continue                         # not enough history for the window
                p = int(window_partition[0])
                if p == 0:
                    train_starts.append((aid, ep_idx, start))
                    train_ym.append(ym[start])
                else:
                    val_starts.append((aid, ep_idx, start))
                    val_ym.append(ym[start])

        log.info(
            "Subsequence starts: train=%d, val=%d", len(train_starts), len(val_starts)
        )

        # Month summary table for STOP POINT 3 reporting.
        s_train = pd.Series(train_ym).value_counts().to_frame(name="train_starts")
        s_val = pd.Series(val_ym).value_counts().to_frame(name="val_starts")
        self.month_summary = (
            s_train.join(s_val, how="outer")
            .fillna(0).astype(int).sort_index()
            .rename_axis("year_month")
        )

        # Per-month inverse-frequency weights so each month is equally
        # likely per epoch in expectation.
        if len(train_starts) > 0:
            month_size = pd.Series(train_ym).value_counts().to_dict()
            w = np.array(
                [1.0 / month_size[m] for m in train_ym], dtype=np.float64
            )
            w /= w.sum()
            self._train_weights = w

        self._train_ds = TrajectoryDataset(
            train_starts, episodes, feats, self.seq_len, self.T,
        )
        self._val_ds = TrajectoryDataset(
            val_starts, episodes, feats, self.seq_len, self.T,
        )

    def train_dataloader(self) -> DataLoader:
        if self._train_ds is None:
            raise RuntimeError("setup() must be called before train_dataloader()")
        sampler = None
        if self._train_weights is not None and len(self._train_ds) > 0:
            g = torch.Generator()
            g.manual_seed(self.seed)
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(self._train_weights).double(),
                num_samples=len(self._train_ds),
                replacement=True,
                generator=g,
            )
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_ds is None:
            raise RuntimeError("setup() must be called before val_dataloader()")
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
