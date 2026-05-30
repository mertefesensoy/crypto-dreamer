"""Gate-eval reader for the world-model diagnostic · reads the three gates
from a CHECKPOINT, never from W&B.

Rationale · W&B logging silently failed mid-run on the Phase 5.4 diagnostic
(run `1rq8d8u5` synced only to step ~612 of 30000 while training completed
correctly), so the charts cannot be trusted. This script is the authoritative
gate read · it loads a trained WorldModel checkpoint, rebuilds the validation
dataloader, runs the model's own `_step(..., stage="val", collect_trace=True)`
over N val batches, and averages the returned `info` components.

Promoted from the repo-root `_gate.py` (ADR-006 Section 5) and parameterized
by checkpoint path so it is reusable across runs. The evaluation methodology
is intentionally identical to `_gate.py` so the numbers reproduce; the only
addition is a fixed torch seed so the straight-through categorical sampling in
the RSSM is deterministic across invocations.

Gates (ADR-003 / ADR-006):
- Gate 1 · `kl_unclipped` vs the 32-nat free-bits floor · release => > 32.
- Gate 2 · per-horizon `loss_forward` summed vs the 8.8632 marginal baseline ·
  pass => < 8.85, 8.85-8.86 inconclusive.
- Gate 3 · `loss_reward` vs ~0.48 · integrity check, expected stable.

Usage:
    uv run python -m scripts.eval_gates \
        --ckpt checkpoints/world_model_diagnostic_step=30000-v1.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from models.world_model import WorldModel
from training.datamodule import SpotBTCDataModule

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GATE1_KL_FLOOR = 32.0  # nats · posterior-prior KL must exceed this to "release"
GATE2_BASELINE = 8.8632  # marginal forward-NLL baseline (docs/findings/2026-05-27)
GATE2_PASS = 8.85  # forward sum below this is a clean Gate 2 pass
GATE3_REWARD = 0.48  # reward-NLL integrity target


def _resolve(path: str) -> str:
    """Return `path` as-is if absolute, else resolved against the project root."""
    p = Path(path)
    return str(p) if p.is_absolute() else str(PROJECT_ROOT / p)


def build_model(cfg: OmegaConf, sd: dict[str, torch.Tensor], device: str) -> WorldModel:
    """Construct a WorldModel from `cfg` and load a prefix-stripped state_dict.

    Args:
        cfg: the OmegaConf config (expects the `model` group).
        sd: state_dict whose keys may carry a `model.` prefix (Lightning
            wrapper); the prefix is stripped by the caller before this point.
        device: torch device string, e.g. "cuda" or "cpu".

    Returns:
        A WorldModel in eval mode with weights loaded (strict=False, to
        tolerate buffers that may be absent from a raw state_dict dump).
    """
    m = WorldModel(
        n_vars=cfg.model.n_vars,
        seq_len=cfg.model.seq_len,
        d_model=cfg.model.d_model,
        encoder_layers=cfg.model.encoder_layers,
        encoder_heads=cfg.model.encoder_heads,
        encoder_ff=cfg.model.encoder_ff,
        encoder_dropout=cfg.model.encoder_dropout,
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
        coef_dyn=cfg.model.coef_dyn,
        coef_rep=cfg.model.coef_rep,
        free_bits=cfg.model.free_bits,
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
        warmup_steps=cfg.model.warmup_steps,
        burn_in=cfg.model.burn_in,
        forward_horizons=list(cfg.model.forward_horizons),
        forward_bins=cfg.model.forward_bins,
        forward_ranges=list(cfg.model.forward_ranges),
    ).to(device)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Read the world-model gates from a checkpoint (not W&B)."
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/world_model_diagnostic_step=30000-v1.ckpt",
        help="checkpoint path · absolute, or relative to the project root",
    )
    ap.add_argument("--config", type=str, default="configs/world_model.yaml")
    ap.add_argument("--n-batches", type=int, default=40, help="val batches to average")
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seeds straight-through sampling for reproducible gate reads",
    )
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    cfg = OmegaConf.load(_resolve(args.config))
    ckpt_path = _resolve(args.ckpt)

    ckpt = torch.load(ckpt_path, map_location=args.device)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}

    m = build_model(cfg, sd, args.device)

    d = cfg.data
    dm = SpotBTCDataModule(
        klines_db=str(PROJECT_ROOT / d.klines_db),
        steps_db=str(PROJECT_ROOT / d.steps_db),
        symbol=d.symbol,
        interval=d.interval,
        T=d.T,
        batch_size=d.batch_size,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        max_episodes=d.get("max_episodes", None),
        seed=cfg.seed,
    )
    dm.setup("fit")
    vl = dm.val_dataloader()

    keys = [
        "loss",
        "loss_forward",
        "loss_reward",
        "loss_continue",
        "loss_dyn",
        "loss_rep",
        "kl_unclipped",
    ]
    agg = {k: 0.0 for k in keys}
    ph: torch.Tensor | None = None
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(vl):
            if i >= args.n_batches:
                break
            batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            _, info = m._step(batch, stage="val", collect_trace=True)
            for k in keys:
                agg[k] += info[k].item()
            p = info["loss_forward_per_horizon"].detach().cpu()
            ph = p if ph is None else ph + p
            n += 1

    if n == 0 or ph is None:
        raise RuntimeError("no validation batches produced; cannot read gates")

    print(f"\n=== val metrics over {n} batches · ckpt {Path(ckpt_path).name} ===")
    for k in keys:
        print(f"{k:16s} {agg[k] / n:.4f}")

    ph = ph / n
    hz = list(cfg.model.forward_horizons)
    fwd_sum = float(ph.sum())
    print("\nper-horizon forward loss:")
    for hi, h in enumerate(hz):
        print(f"  h={h:2d}: {ph[hi]:.4f}")
    print(
        f"  sum: {fwd_sum:.4f}   (Gate 2 baseline {GATE2_BASELINE}; "
        f"<{GATE2_PASS} pass, {GATE2_PASS}-{GATE2_BASELINE} inconclusive)"
    )

    kl = agg["kl_unclipped"] / n
    rew = agg["loss_reward"] / n
    g1 = "RELEASE" if kl > GATE1_KL_FLOOR else "collapsed (no release)"
    g2 = "PASS" if fwd_sum < GATE2_PASS else "FAIL"
    g3 = "ok" if abs(rew - GATE3_REWARD) < 0.05 else "DRIFT"
    print(f"\nGATE 1  kl_unclipped {kl:.3f}  vs {GATE1_KL_FLOOR} floor  -> {g1}")
    print(f"GATE 2  forward sum  {fwd_sum:.4f}  vs {GATE2_BASELINE} baseline  -> {g2}")
    print(f"GATE 3  loss_reward  {rew:.4f}  vs ~{GATE3_REWARD}  -> {g3}")


if __name__ == "__main__":
    main()
