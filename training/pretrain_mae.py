"""TS-MAE pretraining for the iTransformer encoder.

Masked-timestep reconstruction on raw 2-year BTCUSDT 1m kline features
(12 market features only — no portfolio). Mask 40% of timesteps per
sample, MSE loss on masked positions only. The encoder weights are
saved at the end; the MAE decoder is throwaway.

Run:
    # Tiny verify (200 steps, 5-episode subset) — must show >=20% MSE drop
    # at step 200 vs step 10, else halt.
    python -m training.pretrain_mae mode=tiny train.max_steps=200 \
        train.max_hours=0 train.val_check_interval=100

    # Full 2h pretrain
    python -m training.pretrain_mae

Stop rules:
    Tiny run:  step 200 MSE >= 0.8 * step 10 MSE -> exit non-zero, do not
               proceed to the 2h run.
    Full run:  at the 30-min mark, val MSE has not dropped >=10% vs
               random init -> abort, fall back to random encoder init in
               Phase 5.1.

Determinism:
    seed_everything(42) at script entry. DataLoader uses a torch.Generator
    seeded from the same value. cuDNN is left in its default
    nondeterministic mode for speed; this is disclosed in the impl doc.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from envs.spot_btc import compute_feature_block
from models.encoder import iTransformerEncoder
from models.mae_decoder import MAEDecoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class KlineWindowDataset(Dataset):
    """Random 256-bar windows over a precomputed feature matrix.

    The feature matrix is held in CPU memory (~50 MB fp32 for 2y of 1m
    bars) and shared by reference; __getitem__ slices a (T, F) view.
    """

    def __init__(self, features: np.ndarray, indices: np.ndarray, seq_len: int):
        self.features = features
        self.indices = indices
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> torch.Tensor:
        end = int(self.indices[i])
        win = self.features[end - self.seq_len : end]
        return torch.from_numpy(win.copy())


def build_split_indices(ts_array: np.ndarray, val_month: str, seq_len: int):
    """Return (train_idx, val_idx) of right-edge indices.

    A window's right edge is the index of its last (most recent) row.
    Train pool: right-edge ts < val_month_start.
    Val pool:   right-edge ts in [val_month_start, val_month_end).
    """
    ts_pd = pd.to_datetime(ts_array, utc=True)
    val_start = pd.Timestamp(f"{val_month}-01", tz="UTC")
    if val_start.month == 12:
        val_end = pd.Timestamp(f"{val_start.year + 1}-01-01", tz="UTC")
    else:
        val_end = pd.Timestamp(
            f"{val_start.year}-{val_start.month + 1:02d}-01", tz="UTC"
        )

    eligible = np.arange(seq_len, len(ts_array))
    edges = ts_pd[eligible]
    is_val = (edges >= val_start) & (edges < val_end)
    return eligible[~is_val], eligible[is_val]


class MAEModule(L.LightningModule):
    def __init__(
        self,
        n_vars: int,
        seq_len: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dim_ff: int,
        dropout: float,
        mask_ratio: float,
        lr: float,
        weight_decay: float,
        warmup_steps: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = iTransformerEncoder(
            n_vars=n_vars, seq_len=seq_len, d_model=d_model,
            n_layers=n_layers, n_heads=n_heads, dim_ff=dim_ff, dropout=dropout,
        )
        self.decoder = MAEDecoder(seq_len=seq_len, d_model=d_model, hidden=d_model)
        self.mask_ratio = mask_ratio
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

    def forward(self, x: torch.Tensor):
        # x: (B, T, F)
        B, T, _ = x.shape
        mask = torch.rand(B, T, device=x.device) < self.mask_ratio
        x_in = x.masked_fill(mask.unsqueeze(-1), 0.0)
        tokens = self.encoder(x_in)        # (B, F, d_model)
        recon = self.decoder(tokens)       # (B, T, F)
        return recon, mask

    def _step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        recon, mask = self(batch)
        diff = (recon - batch) ** 2                 # (B, T, F)
        per_step_mse = diff.mean(dim=-1)            # (B, T)
        denom = mask.sum().clamp(min=1)
        loss = (per_step_mse * mask).sum() / denom
        self.log(
            f"{stage}/mse", loss,
            on_step=(stage == "train"), on_epoch=True, prog_bar=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return (step + 1) / self.warmup_steps
            return 1.0

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return [opt], [{"scheduler": sched, "interval": "step"}]


class TimeBudgetCallback(L.Callback):
    """Stop training when wall-clock budget is exceeded."""

    def __init__(self, max_seconds: float):
        self.max_seconds = max_seconds
        self._start: float | None = None

    def on_train_start(self, trainer, pl_module):
        self._start = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._start is not None and (time.time() - self._start) > self.max_seconds:
            print(f"[TIME-BUDGET] Hit {self.max_seconds:.0f}s wall-clock limit; stopping.")
            trainer.should_stop = True


class TinyVerifyCallback(L.Callback):
    """Tiny-mode pass criterion: train MSE at step `final` must be
    <= 0.8 * train MSE at step `early`. If violated, exit non-zero so
    the calling shell can halt before the 2h run."""

    def __init__(self, early: int = 10, final: int = 200, drop_ratio: float = 0.20):
        self.early = early
        self.final = final
        self.drop_ratio = drop_ratio
        self.early_mse: float | None = None
        self.final_mse: float | None = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = float(outputs["loss"]) if isinstance(outputs, dict) else float(outputs)
        step = trainer.global_step
        if step == self.early:
            self.early_mse = loss
            print(f"[TINY-VERIFY] step={step} train MSE = {loss:.6f}")
        elif step == self.final:
            self.final_mse = loss
            print(f"[TINY-VERIFY] step={step} train MSE = {loss:.6f}")

    def on_train_end(self, trainer, pl_module):
        if self.early_mse is None or self.final_mse is None:
            print("[TINY-VERIFY] Did not capture both checkpoints — exiting non-zero.")
            sys.exit(1)
        threshold = (1.0 - self.drop_ratio) * self.early_mse
        passed = self.final_mse <= threshold
        drop_pct = 100.0 * (1.0 - self.final_mse / self.early_mse)
        print(
            f"[TINY-VERIFY] step{self.early} MSE={self.early_mse:.6f} "
            f"step{self.final} MSE={self.final_mse:.6f} "
            f"drop={drop_pct:.1f}% (need >={self.drop_ratio*100:.0f}%) "
            f"-> {'PASS' if passed else 'FAIL'}"
        )
        if not passed:
            sys.exit(2)


@hydra.main(
    version_base=None,
    config_path=str(PROJECT_ROOT / "configs"),
    config_name="pretrain_mae",
)
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed, workers=True)
    print(OmegaConf.to_yaml(cfg))

    db_path = str(PROJECT_ROOT / cfg.data.db_path)
    print(f"Loading klines from {db_path} ...")
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(
        "SELECT ts, open, high, low, close, volume FROM klines "
        "WHERE symbol = ? AND interval = ? ORDER BY ts",
        [cfg.data.symbol, cfg.data.interval],
    ).df()
    con.close()
    print(f"Klines: {len(df):,} rows, {df.ts.min()} -> {df.ts.max()}")

    print("Computing features ...")
    feats = compute_feature_block(df)
    print(f"Features: shape={feats.shape}, dtype={feats.dtype}, mb={feats.nbytes / 1024**2:.1f}")

    ts_array = df["ts"].to_numpy()
    train_idx, val_idx = build_split_indices(ts_array, cfg.data.val_month, cfg.model.seq_len)
    print(f"Train windows: {len(train_idx):,} | Val windows: {len(val_idx):,}")

    if cfg.mode == "tiny":
        rng = np.random.default_rng(cfg.seed)
        train_idx = rng.choice(train_idx, size=min(7200, len(train_idx)), replace=False)
        val_idx = rng.choice(val_idx, size=min(720, len(val_idx)), replace=False)
        print(f"TINY MODE — train={len(train_idx)}, val={len(val_idx)}")

    train_ds = KlineWindowDataset(feats, train_idx, cfg.model.seq_len)
    val_ds = KlineWindowDataset(feats, val_idx, cfg.model.seq_len)

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    common_kw = {
        "batch_size": cfg.train.batch_size,
        "num_workers": cfg.train.num_workers,
        "drop_last": True,
    }
    train_dl = DataLoader(train_ds, shuffle=True, generator=g, **common_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **{**common_kw, "drop_last": False})

    model = MAEModule(
        n_vars=cfg.model.n_vars,
        seq_len=cfg.model.seq_len,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        dim_ff=cfg.model.dim_ff,
        dropout=cfg.model.dropout,
        mask_ratio=cfg.train.mask_ratio,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        warmup_steps=cfg.train.warmup_steps,
    )

    wandb_logger = L.pytorch.loggers.WandbLogger(
        project=cfg.wandb.project,
        name=f"{cfg.wandb.run_name}-{cfg.mode}" if cfg.mode == "tiny" else cfg.wandb.run_name,
        mode=cfg.wandb.mode,
        save_dir=str(PROJECT_ROOT),
    )
    wandb_logger.log_hyperparams({
        "seed": cfg.seed,
        "shuffle_seed": cfg.seed,
        "mode": cfg.mode,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    })

    callbacks: list[L.Callback] = []
    if cfg.train.max_hours and cfg.train.max_hours > 0:
        callbacks.append(TimeBudgetCallback(cfg.train.max_hours * 3600))

    if cfg.mode == "tiny":
        callbacks.append(TinyVerifyCallback(early=10, final=cfg.train.max_steps, drop_ratio=0.20))

    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_cb = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"encoder_mae_{cfg.mode}_{{step}}",
        save_top_k=1,
        monitor="val/mse",
        mode="min",
        save_last=True,
    )
    callbacks.append(ckpt_cb)

    trainer = L.Trainer(
        max_steps=cfg.train.max_steps,
        accelerator="gpu",
        devices=1,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.grad_clip,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        val_check_interval=cfg.train.val_check_interval,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_dl, val_dl)

    out = ckpt_dir / f"encoder_mae_{cfg.mode}_raw.pt"
    torch.save(model.encoder.state_dict(), out)
    print(f"Saved encoder state_dict: {out}")


if __name__ == "__main__":
    main()
