"""Hydra entrypoint for the Phase 5.2+ world-model trainer.

Run:
    # smoke (100 steps, 5 episodes, all 4 loss components logged)
    python -m training.train_world_model

    # full (Phase 5.3) — overrides via CLI:
    python -m training.train_world_model mode=full \
        train.max_steps=200000 train.max_hours=4 \
        wandb.run_name=phase5.3-rssm-full \
        data.max_episodes=null
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from models.world_model import WorldModel
from training.datamodule import SpotBTCDataModule

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TimeBudgetCallback(L.Callback):
    def __init__(self, max_seconds: float):
        self.max_seconds = max_seconds
        self._start: float | None = None

    def on_train_start(self, trainer, pl_module):
        self._start = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._start is None:
            return
        if (time.time() - self._start) > self.max_seconds:
            print(f"[TIME-BUDGET] {self.max_seconds:.0f}s elapsed; stopping.")
            trainer.should_stop = True


class HeartbeatCallback(L.Callback):
    """Append a one-line heartbeat to a log file every N batches.

    Independent of W&B — even if wandb sync stalls, the heartbeat tells
    us exactly when training paused. If the file's mtime hasn't moved
    in >5min while the process is alive, we know to investigate.
    Cheap insurance for unattended overnight runs on a Windows laptop.
    """

    def __init__(self, path: str | Path, every_n: int = 100):
        self.path = Path(path)
        self.every_n = every_n
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_train_start(self, trainer, pl_module):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} TRAIN_START\n")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.every_n != 0:
            return
        loss = None
        if isinstance(outputs, dict):
            loss_t = outputs.get("loss")
            if loss_t is not None:
                loss = float(loss_t.detach())
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(
                f"{datetime.now().isoformat()} step={trainer.global_step} "
                f"loss={loss if loss is None else f'{loss:.4f}'}\n"
            )

    def on_train_end(self, trainer, pl_module):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} TRAIN_END step={trainer.global_step}\n")


@hydra.main(
    version_base=None,
    config_path=str(PROJECT_ROOT / "configs"),
    config_name="world_model",
)
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed, workers=True)
    print(OmegaConf.to_yaml(cfg))

    # Smoke-mode auto-overrides.
    max_eps = cfg.data.max_episodes
    if cfg.mode == "smoke" and max_eps is None:
        max_eps = 5

    dm = SpotBTCDataModule(
        klines_db=str(PROJECT_ROOT / cfg.data.klines_db),
        steps_db=str(PROJECT_ROOT / cfg.data.steps_db),
        symbol=cfg.data.symbol,
        interval=cfg.data.interval,
        T=cfg.data.T,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.get("pin_memory", False),
        persistent_workers=cfg.data.get("persistent_workers", False),
        seed=cfg.seed,
        max_episodes=max_eps,
    )

    model = WorldModel(
        n_vars=cfg.model.n_vars,
        seq_len=cfg.model.seq_len,
        d_model=cfg.model.d_model,
        encoder_layers=cfg.model.encoder_layers,
        encoder_heads=cfg.model.encoder_heads,
        encoder_ff=cfg.model.encoder_ff,
        encoder_dropout=cfg.model.encoder_dropout,
        mae_checkpoint=str(PROJECT_ROOT / cfg.model.mae_checkpoint)
        if cfg.model.mae_checkpoint
        else None,
        hidden_dim=cfg.model.hidden_dim,
        n_latents=cfg.model.n_latents,
        n_classes=cfg.model.n_classes,
        rssm_mlp_hidden=cfg.model.rssm_mlp_hidden,
        unimix=cfg.model.unimix,
        n_actions=cfg.model.n_actions,
        action_emb_dim=cfg.model.action_emb_dim,
        head_hidden=cfg.model.head_hidden,
        reward_n_bins=cfg.model.reward_n_bins,
        reward_low=cfg.model.reward_low,
        reward_high=cfg.model.reward_high,
        # Forward-distribution head config (PR 4). list(...) materialises the
        # OmegaConf ListConfig into the plain lists the head expects.
        forward_horizons=list(cfg.model.forward_horizons),
        forward_bins=cfg.model.forward_bins,
        forward_ranges=list(cfg.model.forward_ranges),
        coef_dyn=cfg.model.coef_dyn,
        coef_rep=cfg.model.coef_rep,
        free_bits=cfg.model.free_bits,
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
        warmup_steps=cfg.model.warmup_steps,
        burn_in=cfg.model.burn_in,
    )

    if cfg.train.get("compile", False):
        try:
            mode = cfg.train.get("compile_mode", "reduce-overhead")
            model = torch.compile(model, mode=mode)
            print(f"[compile] applied mode={mode}")
        except Exception as e:
            print(f"[compile] FAILED ({type(e).__name__}: {e}); proceeding without compile")

    # init_timeout=300: wandb's default 90s is sometimes too short on
    # this machine when the wandb CDN is slow. Same fix used in 5.0.5.
    import wandb as _wandb_pkg  # local import: settings only matters here

    wandb_logger = L.pytorch.loggers.WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        mode=cfg.wandb.mode,
        save_dir=str(PROJECT_ROOT),
        settings=_wandb_pkg.Settings(init_timeout=300),
    )
    wandb_logger.log_hyperparams(
        {
            "seed": cfg.seed,
            "shuffle_seed": cfg.seed,
            "mode": cfg.mode,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
    )

    callbacks: list[L.Callback] = []
    if cfg.train.max_hours and cfg.train.max_hours > 0:
        callbacks.append(TimeBudgetCallback(cfg.train.max_hours * 3600))
    heartbeat_path = cfg.train.get("heartbeat_path", None)
    if heartbeat_path:
        callbacks.append(
            HeartbeatCallback(
                path=str(PROJECT_ROOT / heartbeat_path),
                every_n=int(cfg.train.get("heartbeat_every_n", 100)),
            )
        )

    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_cb = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"world_model_{cfg.mode}_{{step}}",
        save_top_k=1,
        monitor="val/loss_reward",  # gate spec: best by val reward NLL
        mode="min",
        save_last=True,
        every_n_train_steps=cfg.train.ckpt_every_n_steps,
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
        log_every_n_steps=cfg.train.log_every_n_steps,
        val_check_interval=cfg.train.val_check_interval,
        limit_val_batches=cfg.train.get("limit_val_batches", 1.0),
        enable_progress_bar=False,
        detect_anomaly=False,
    )
    # Resume from a previous Lightning checkpoint if one is configured.
    # Pass via `+resume_ckpt=path/to/ckpt` on the CLI.
    resume_ckpt = cfg.get("resume_ckpt", None)
    if resume_ckpt:
        resume_path = str(PROJECT_ROOT / resume_ckpt)
        print(f"[resume] from {resume_path}")
        trainer.fit(model, dm, ckpt_path=resume_path)
    else:
        trainer.fit(model, dm)

    # Save raw world-model state_dict (no Lightning wrapper).
    out = ckpt_dir / f"world_model_{cfg.mode}_raw.pt"
    torch.save(model.state_dict(), out)
    print(f"Saved world-model state_dict: {out}")


if __name__ == "__main__":
    main()
