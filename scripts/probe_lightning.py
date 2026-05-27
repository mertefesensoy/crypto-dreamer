"""Lightning-overhead probe — 200 training steps, no logger/ckpt/val.

Mirrors the production WorldModel + datamodule + autocast config but
runs Trainer in `barebones=True` mode so all bookkeeping that doesn't
affect the gradient is off. Gives a clean number for "Lightning's
floor cost" to compare against the hand-rolled bench.

Run:
    python -m scripts.probe_lightning
"""
from __future__ import annotations

import time
from pathlib import Path

import lightning as L
import torch

from models.world_model import WorldModel
from training.datamodule import SpotBTCDataModule

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    L.seed_everything(42, workers=True)

    dm = SpotBTCDataModule(
        klines_db=str(PROJECT_ROOT / "data" / "market_ro.duckdb"),
        steps_db=str(PROJECT_ROOT / "data" / "market.duckdb"),
        T=48, batch_size=32, num_workers=4,
        pin_memory=True, persistent_workers=True,
    )
    model = WorldModel(
        mae_checkpoint=str(PROJECT_ROOT / "checkpoints" / "encoder_mae_full_raw.pt"),
    )

    t0 = time.time()
    trainer = L.Trainer(
        max_steps=200,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        gradient_clip_val=1000.0,
        barebones=True,           # disables: progress bar, model summary,
                                  # checkpointing, logger, sanity val, profiler
    )
    trainer.fit(model, dm)
    elapsed = time.time() - t0

    # Trainer.fit() includes setup time (datamodule, model construction,
    # mae checkpoint load). Subtract a conservative 30s for setup to get
    # a steady-state estimate. We also want strictly per-step timing,
    # which we can't get from barebones alone — so report both.
    print(f"\n=== Lightning barebones probe ===")
    print(f"Wall clock total : {elapsed:.1f}s")
    print(f"Steps            : 200")
    print(f"ms/step (raw)    : {elapsed / 200 * 1000:.1f}")
    print(f"ms/step (-30s setup) : {(elapsed - 30) / 200 * 1000:.1f}")


if __name__ == "__main__":
    main()
