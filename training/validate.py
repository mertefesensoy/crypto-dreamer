"""Phase 5.4 validation gate computation.

Runs three classes of metric on the held-out val pool:

1. **One-step reward MAE** vs predict-zero baseline — gate ≥30% improvement.
2. **Decoder R² per feature** on the 15-dim current-bar reconstruction —
   gate ≥0.5 on fast/return features (log_ret, vol_5, vol_15, ret_5,
   ret_15, ret_60), ≥0.7 on slow features (vol_60, close_norm, rsi_14,
   macd, vol_z, hl_range, plus the 3 portfolio scalars).
3. **Multi-step rollout KL** at horizons h ∈ {5, 15, 30} between the
   imagined prior at step h and the posterior at step h conditioned on
   real observations. Gate <0.5 / 1.2 / 2.0 nats respectively.

Outputs:
- Console gate-status table with PASS/FAIL per metric.
- Markdown report at `docs/implementations/phase5-4-validation-report.md`
  (overwritten each run).
- Returns process exit code 0 if all gates pass, 1 otherwise.

Designed to be run *after* Phase 5.3 completes against the final
best-by-val-reward-NLL checkpoint. Does not load in-training
checkpoints — accepts an explicit `--checkpoint` path.

Run:
    python -m training.validate --checkpoint checkpoints/world_model_full_best.ckpt
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F

from envs.spot_btc import FEATURE_NAMES
from models.world_model import WorldModel
from models.rssm import RSSM
from training.datamodule import SpotBTCDataModule

PROJECT_ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)

# Per-feature gate (R²). Cols 0-11 follow envs.spot_btc.FEATURE_NAMES;
# cols 12-14 are portfolio scalars. Slow features get the harder bar.
FAST_FEATURES = {"log_ret", "vol_5", "vol_15", "ret_5", "ret_15", "ret_60"}
SLOW_FEATURES = {"vol_60", "rsi_14", "macd", "vol_z", "hl_range", "close_norm"}
PORTFOLIO_NAMES = ("alloc_ratio", "cash_ratio", "log_equity_ratio")
ALL_FEATURE_NAMES = tuple(FEATURE_NAMES) + PORTFOLIO_NAMES  # 15

REWARD_MAE_GATE = 0.30          # ≥30% improvement vs predict-zero
DECODER_R2_FAST = 0.50
DECODER_R2_SLOW = 0.70
MULTI_STEP_KL_GATES = {5: 0.5, 15: 1.2, 30: 2.0}


def _r2(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Per-feature 1D R² (treating each (B,T) pair as a sample)."""
    y_true_mean = y_true.mean()
    ss_res = float(((y_pred - y_true) ** 2).sum())
    ss_tot = float(((y_true - y_true_mean) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


@torch.no_grad()
def collect_one_step_metrics(
    model: WorldModel, dl, device: torch.device, max_batches: int | None = None,
) -> dict:
    """Run posterior path, accumulate decoder + reward predictions and
    targets across the val pool."""
    model.eval()

    abs_err_sum = 0.0
    abs_target_sum = 0.0
    n_reward = 0

    feat_pred_chunks: list[np.ndarray] = []
    feat_true_chunks: list[np.ndarray] = []

    for batch_idx, batch in enumerate(dl):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        obs = batch["obs_window"]
        action = batch["action"]
        reward = batch["reward"]
        is_first = batch["is_first"]

        B, T, S, F_dim = obs.shape

        # Encode all T windows.
        x = model.encode_obs(obs.reshape(B * T, S, F_dim)).reshape(B, T, -1)
        dec_target = obs[:, :, -1, :]  # (B, T, 15)

        prev_h, prev_z = model.rssm.initial_state(B, device)
        prev_a = torch.zeros(B, dtype=torch.long, device=device)

        feat_pred_steps = []
        rew_pred_steps = []

        for t in range(T):
            reset = is_first[:, t].clone()
            if t == 0:
                reset = torch.ones_like(reset)
            a_emb = model.action_embed(prev_a)
            h, z, _, _ = model.rssm.step(
                prev_z, prev_h, a_emb, x[:, t], is_first=reset,
            )
            feat = torch.cat([h, z], dim=-1)
            feat_pred_steps.append(model.decoder_head(feat))
            rew_pred_steps.append(model.reward_head.predict(model.reward_head(feat)))
            prev_h, prev_z = h, z
            prev_a = action[:, t]

        # Skip the first burn_in steps so we mirror training conditions.
        bi = model.burn_in
        feat_pred = torch.stack(feat_pred_steps[bi:], dim=1)         # (B, T-bi, 15)
        feat_true = dec_target[:, bi:]                               # (B, T-bi, 15)
        rew_pred = torch.stack(rew_pred_steps[bi:], dim=1)           # (B, T-bi)
        rew_true = reward[:, bi:]                                    # (B, T-bi)

        abs_err_sum += float((rew_pred - rew_true).abs().sum())
        abs_target_sum += float(rew_true.abs().sum())
        n_reward += rew_true.numel()

        feat_pred_chunks.append(feat_pred.float().cpu().numpy())
        feat_true_chunks.append(feat_true.float().cpu().numpy())

    feat_pred_all = np.concatenate(feat_pred_chunks, axis=0)
    feat_true_all = np.concatenate(feat_true_chunks, axis=0)

    # Per-feature R² across all (B, T-bi) pairs.
    r2_per_feature = {}
    for f in range(feat_pred_all.shape[-1]):
        r2_per_feature[ALL_FEATURE_NAMES[f]] = _r2(
            feat_pred_all[..., f].reshape(-1),
            feat_true_all[..., f].reshape(-1),
        )

    model_mae = abs_err_sum / max(n_reward, 1)
    baseline_mae = abs_target_sum / max(n_reward, 1)  # predict-zero
    return {
        "reward_mae_model": model_mae,
        "reward_mae_baseline": baseline_mae,
        "reward_mae_improvement": (baseline_mae - model_mae) / max(baseline_mae, 1e-12),
        "r2_per_feature": r2_per_feature,
        "n_reward_samples": n_reward,
    }


@torch.no_grad()
def collect_multi_step_kl(
    model: WorldModel, dl, device: torch.device,
    horizons: tuple[int, ...] = (5, 15, 30),
    max_batches: int | None = None,
) -> dict[int, float]:
    """At each horizon h, KL(posterior_real_h || prior_imagined_h),
    averaged over (batch, latents)."""
    model.eval()
    max_h = max(horizons)
    kl_acc: dict[int, list[float]] = defaultdict(list)

    for batch_idx, batch in enumerate(dl):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        obs = batch["obs_window"]
        action = batch["action"]
        is_first = batch["is_first"]
        B, T, S, F_dim = obs.shape
        if T <= max_h:
            continue

        x = model.encode_obs(obs.reshape(B * T, S, F_dim)).reshape(B, T, -1)

        # 1) Real (posterior) path through step max_h. Save (h_t, z_t,
        #    posterior_logits) at every step.
        prev_h, prev_z = model.rssm.initial_state(B, device)
        prev_a = torch.zeros(B, dtype=torch.long, device=device)
        post_logits_at_step = {}
        h_at_step = {}
        z_at_step = {}
        for t in range(max_h + 1):
            reset = is_first[:, t].clone()
            if t == 0:
                reset = torch.ones_like(reset)
            a_emb = model.action_embed(prev_a)
            h, z, _, post_logits = model.rssm.step(
                prev_z, prev_h, a_emb, x[:, t], is_first=reset,
            )
            post_logits_at_step[t] = post_logits
            h_at_step[t] = h
            z_at_step[t] = z
            prev_h, prev_z = h, z
            prev_a = action[:, t]

        # 2) For each horizon, imagine forward from (h_0, z_0) using
        #    prior + real actions, then KL against posterior at step h.
        for h in horizons:
            imag_h = h_at_step[0]
            imag_z = z_at_step[0]
            for t in range(1, h + 1):
                a_emb = model.action_embed(action[:, t - 1])
                # Re-use the rssm.step path but discard posterior side
                # by passing x_t equal to anything (we only use prior).
                # Cleanest: replicate the GRU + prior_head computation.
                gru_in = model.rssm.pre_gru(torch.cat([imag_z, a_emb], dim=-1))
                imag_h = model.rssm.gru(gru_in, imag_h)
                prior_logits = model.rssm.prior_head(imag_h).reshape(
                    -1, model.rssm.n_latents, model.rssm.n_classes,
                )
                imag_z = model.rssm.sample_st(prior_logits).reshape(
                    -1, model.rssm.z_dim,
                )

            # KL(posterior_real_h || prior_imagined_h), per-latent, mean.
            post_h = post_logits_at_step[h]
            kl_per_dim = RSSM.categorical_kl(
                post_h, prior_logits, unimix=model.unimix,
            )                                              # (B, n_latents)
            kl_acc[h].append(float(kl_per_dim.mean().cpu()))

    return {h: float(np.mean(v)) if v else float("nan") for h, v in kl_acc.items()}


def _evaluate_gates(metrics: dict, multi_step: dict[int, float]) -> tuple[bool, str]:
    lines: list[str] = []
    all_pass = True

    # Reward MAE
    impr = metrics["reward_mae_improvement"]
    p_mae = impr >= REWARD_MAE_GATE
    all_pass = all_pass and p_mae
    lines.append(
        f"| reward MAE improvement | {impr:.3f} | ≥{REWARD_MAE_GATE:.2f} | "
        f"{'PASS' if p_mae else 'FAIL'} |"
    )

    # Decoder R²
    for fname in ALL_FEATURE_NAMES:
        r2 = metrics["r2_per_feature"][fname]
        gate = DECODER_R2_FAST if fname in FAST_FEATURES else DECODER_R2_SLOW
        p = r2 >= gate
        all_pass = all_pass and p
        lines.append(
            f"| R² {fname:<18} | {r2:.3f} | ≥{gate:.2f} | "
            f"{'PASS' if p else 'FAIL'} |"
        )

    # Multi-step KL
    for h in sorted(multi_step):
        kl = multi_step[h]
        gate = MULTI_STEP_KL_GATES[h]
        p = kl < gate
        all_pass = all_pass and p
        lines.append(
            f"| multi-step KL h={h:<2} | {kl:.3f} nats | <{gate:.2f} | "
            f"{'PASS' if p else 'FAIL'} |"
        )

    table = (
        "| metric | value | gate | status |\n"
        "|---|---|---|---|\n" + "\n".join(lines) + "\n"
    )
    return all_pass, table


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to world_model checkpoint (.ckpt or _raw.pt)")
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--klines-db", default=str(PROJECT_ROOT / "data" / "market_ro.duckdb"))
    p.add_argument("--steps-db", default=str(PROJECT_ROOT / "data" / "market.duckdb"))
    p.add_argument("--output", default=str(PROJECT_ROOT / "docs" / "implementations" / "phase5-4-validation-report.md"))
    args = p.parse_args()

    L.seed_everything(42, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2

    # Lightning .ckpt: load via LightningModule. _raw.pt: bare state_dict.
    if ckpt_path.suffix == ".ckpt":
        model = WorldModel.load_from_checkpoint(str(ckpt_path), map_location=device)
    else:
        model = WorldModel()
        state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        model.load_state_dict(state)
        model = model.to(device)
    model.eval()

    dm = SpotBTCDataModule(
        klines_db=args.klines_db, steps_db=args.steps_db,
        T=model.hparams.get("seq_len", 256) and 48,  # use cfg T from training; default 48
        batch_size=32, num_workers=2, pin_memory=True, persistent_workers=True,
    )
    # Override: the datamodule expects T via constructor arg; pull from
    # model hparams if available.
    dm.T = int(getattr(model, "hparams", {}).get("T", 48)) if hasattr(model, "hparams") else 48
    dm.setup()

    print(f"Validating with {len(dm._val_ds):,} val subsequences "
          f"(T={dm.T}, B={dm.batch_size})")

    one_step = collect_one_step_metrics(model, dm.val_dataloader(), device, max_batches=args.max_batches)
    multi_step = collect_multi_step_kl(model, dm.val_dataloader(), device,
                                        horizons=(5, 15, 30),
                                        max_batches=args.max_batches)

    all_pass, table = _evaluate_gates(one_step, multi_step)
    print("\n=== Phase 5.4 validation gates ===\n" + table)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"# Phase 5.4 validation report\n\n")
        f.write(f"Checkpoint: `{ckpt_path}`\n\n")
        f.write(f"Reward MAE: model={one_step['reward_mae_model']:.5f}, "
                f"baseline (predict-zero)={one_step['reward_mae_baseline']:.5f}\n\n")
        f.write(table)
        f.write(f"\n\nOverall: **{'PASS' if all_pass else 'FAIL'}**\n")
    print(f"Report written: {out}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
