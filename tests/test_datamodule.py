"""Smoke tests for SpotBTCDataModule + encoder pretrain checkpoint loading.

The synthetic-DB test runs hermetically. The real-DB test is gated on
`data/market_ro.duckdb` and `data/market.duckdb` existing — skipped if
they don't, so CI on a fresh machine still passes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from models.encoder import iTransformerEncoder
from training.datamodule import SpotBTCDataModule

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIVE_DB = PROJECT_ROOT / "data" / "market.duckdb"
SNAPSHOT_DB = PROJECT_ROOT / "data" / "market_ro.duckdb"
PRETRAIN_CKPT = PROJECT_ROOT / "checkpoints" / "encoder_mae_full_raw.pt"


def test_synthetic_datamodule_shapes(synthetic_db_with_steps: str) -> None:
    dm = SpotBTCDataModule(
        klines_db=synthetic_db_with_steps,
        steps_db=synthetic_db_with_steps,
        batch_size=2,
        T=8,
        agent_id_pattern="random:%",
    )
    dm.setup()

    # Episode bookkeeping
    assert dm.episode_count == 2
    assert dm.terminated_count == 1, "exactly 1 of the 2 synthetic eps is early-term"

    # At least some train candidates (val pool may be empty on a tiny synthetic ts range)
    train_dl = dm.train_dataloader()
    batch = next(iter(train_dl))

    assert batch["obs_window"].shape == (2, 8, 256, 15)
    assert batch["next_obs_window"].shape == (2, 8, 256, 15)
    assert batch["action"].shape == (2, 8)
    assert batch["reward"].shape == (2, 8)
    assert batch["continue_flag"].shape == (2, 8)
    assert batch["is_first"].shape == (2, 8)

    assert batch["action"].dtype == torch.int64
    assert batch["reward"].dtype == torch.float32
    assert batch["continue_flag"].dtype == torch.bool
    assert batch["is_first"].dtype == torch.bool

    assert torch.isfinite(batch["obs_window"]).all()
    assert torch.isfinite(batch["next_obs_window"]).all()
    assert torch.isfinite(batch["reward"]).all()

    # Portfolio cols are constant across the 256-row window dim.
    pf = batch["obs_window"][..., 12:15]
    assert torch.equal(pf[:, :, 0, :], pf[:, :, 100, :])

    # Action range
    assert batch["action"].min() >= 0
    assert batch["action"].max() <= 4


def test_synthetic_next_obs_is_one_step_shift(synthetic_db_with_steps: str) -> None:
    dm = SpotBTCDataModule(
        klines_db=synthetic_db_with_steps,
        steps_db=synthetic_db_with_steps,
        batch_size=1,
        T=4,
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    # next_obs_window[:, t] should equal obs_window[:, t+1] for t < T-1.
    obs = batch["obs_window"][0]          # (T, 256, 15)
    nxt = batch["next_obs_window"][0]     # (T, 256, 15)
    for t in range(obs.shape[0] - 1):
        assert torch.equal(nxt[t], obs[t + 1]), f"mismatch at t={t}"


@pytest.mark.skipif(
    not LIVE_DB.exists() or not SNAPSHOT_DB.exists(),
    reason="real DuckDBs not present; run after Phase 5.0c",
)
def test_real_dm_encoder_forward() -> None:
    """End-to-end: real datamodule -> encoder forward (incl. pretrain ckpt load)."""
    dm = SpotBTCDataModule(
        klines_db=str(SNAPSHOT_DB),
        steps_db=str(LIVE_DB),
        batch_size=4,
        T=16,
    )
    dm.setup()

    assert dm.episode_count > 0
    assert dm.month_summary is not None and len(dm.month_summary) > 0

    batch = next(iter(dm.train_dataloader()))
    assert batch["obs_window"].shape == (4, 16, 256, 15)
    assert torch.isfinite(batch["obs_window"]).all()

    enc = iTransformerEncoder(
        n_vars=15, seq_len=256, d_model=128, n_layers=4, n_heads=4,
        mae_checkpoint=str(PRETRAIN_CKPT) if PRETRAIN_CKPT.exists() else None,
    )
    enc.eval()
    # Collapse trajectory dim into batch: (B*T, 256, 15).
    flat = batch["obs_window"].reshape(-1, 256, 15)
    with torch.no_grad():
        out = enc(flat)
    assert out.shape == (4 * 16, 15, 128)
    assert torch.isfinite(out).all()


@pytest.mark.skipif(
    not PRETRAIN_CKPT.exists(),
    reason="encoder pretrain checkpoint not present; run Phase 5.0.5 first",
)
def test_pretrain_checkpoint_partial_copy() -> None:
    """Verify mae_checkpoint actually copies the first 12 var embeddings
    and leaves embeddings 12..14 at random init."""
    state = torch.load(str(PRETRAIN_CKPT), map_location="cpu", weights_only=True)
    pretrained_var = state["var_embed.weight"]
    assert pretrained_var.shape == (12, 128)

    enc = iTransformerEncoder(
        n_vars=15, seq_len=256, d_model=128, n_layers=4, n_heads=4,
        mae_checkpoint=str(PRETRAIN_CKPT),
    )
    # First 12 should match exactly.
    assert torch.allclose(enc.var_embed.weight[:12], pretrained_var)
    # Embeddings 12..14 should NOT be all zeros (random init).
    new_rows = enc.var_embed.weight[12:]
    assert new_rows.shape == (3, 128)
    assert (new_rows.abs().sum(dim=-1) > 0).all(), "portfolio embeddings unexpectedly zero"

    # input_proj should match exactly.
    assert torch.allclose(enc.input_proj.weight, state["input_proj.weight"])
    assert torch.allclose(enc.input_proj.bias, state["input_proj.bias"])
