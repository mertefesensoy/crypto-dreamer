"""Phase 5.3 diagnostic-run report puller.

Pulls W&B history for a run id and renders the curves the user asked for:
- total train loss
- per-component losses
- val/loss_reward
- KL prior↔posterior trajectory
- gradient norm
- GPU temp + step-time

Saves PNG to `docs/implementations/phase5-3-diag-report/<run_id>.png`
plus a Markdown summary with key step/loss numbers.

Run:
    python -m scripts.diag_report <run_id>          # e.g. lleske3b
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTITY = "sensoymertefe-ted-niversitesi"
PROJECT = "crypto-dreamer"


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m scripts.diag_report <run_id>", file=sys.stderr)
        return 1
    run_id = sys.argv[1]

    api = wandb.Api(timeout=60)
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    print(f"Pulling history for {run.name} (id={run.id}, state={run.state})")

    # Pull all available history (no explicit keys — wandb returns empty
    # if any requested key is missing, which is the case for `lr-AdamW`
    # since LR isn't logged via the standard hook).
    df = run.history(samples=10000, pandas=True)
    print(f"Loaded {len(df)} datapoints, columns: {list(df.columns)}")

    out_dir = PROJECT_ROOT / "docs" / "implementations" / "phase5-3-diag-report"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    x = df["trainer/global_step"]

    def plot_or_skip(ax, key, label, **kw):
        if key in df.columns:
            mask = df[key].notna()
            ax.plot(x[mask], df[key][mask], label=label, **kw)

    # 1. Total loss + components
    ax = axes[0, 0]
    plot_or_skip(ax, "train/loss_step", "total", color="black", linewidth=1.2)
    plot_or_skip(ax, "train/loss_decoder_step", "decoder", alpha=0.7)
    plot_or_skip(ax, "train/loss_reward_step", "reward", alpha=0.7)
    plot_or_skip(ax, "train/loss_continue_step", "continue", alpha=0.7)
    ax.set_title("train losses")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 2. KL trajectory
    ax = axes[0, 1]
    plot_or_skip(ax, "train/loss_dyn_step", "loss_dyn (clipped)", color="tab:red")
    plot_or_skip(ax, "train/loss_rep_step", "loss_rep (clipped)", color="tab:blue")
    plot_or_skip(ax, "train/kl_unclipped_step", "kl unclipped (raw)", color="tab:green", linestyle="--")
    plot_or_skip(ax, "train/kl_clip_excess_step", "clip excess (free-bits floor)", color="tab:orange", alpha=0.5)
    ax.set_title("KL prior↔posterior")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 3. Val reward NLL
    ax = axes[1, 0]
    plot_or_skip(ax, "val/loss_reward", "val/loss_reward", color="tab:purple", marker="o", markersize=3)
    ax.set_title("val reward NLL")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 4. LR
    ax = axes[1, 1]
    plot_or_skip(ax, "lr-AdamW", "lr", color="tab:gray")
    ax.set_title("learning rate")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_yscale("log")

    # 5. Step time (derived from _runtime / global_step)
    ax = axes[2, 0]
    if "_runtime" in df.columns:
        rt = df["_runtime"].dropna()
        steps = df.loc[rt.index, "trainer/global_step"]
        # Rolling: last 200 steps
        diffs_t = rt.diff()
        diffs_s = steps.diff()
        ms_per_step = (diffs_t / diffs_s.where(diffs_s > 0)) * 1000
        ax.plot(steps, ms_per_step.rolling(20).mean(), color="tab:cyan")
        ax.set_title("ms / step (20-pt rolling mean)")
        ax.set_ylim(0, max(2000, ms_per_step.median() * 3 if ms_per_step.notna().any() else 2000))
    ax.grid(alpha=0.3)

    # 6. GPU temp — not currently logged via WandbLogger. Pull from
    # wandb's system metrics if available.
    ax = axes[2, 1]
    try:
        sys_df = run.history(stream="system", samples=2000, pandas=True)
        if "system.gpu.0.temp" in sys_df.columns:
            ax.plot(sys_df["_runtime"] / 60.0, sys_df["system.gpu.0.temp"], color="tab:red")
            ax.set_title("GPU temp (°C) vs runtime (min)")
        else:
            ax.text(0.5, 0.5, "no GPU temp in system metrics", ha="center", va="center", transform=ax.transAxes)
    except Exception as e:
        ax.text(0.5, 0.5, f"system metrics: {type(e).__name__}", ha="center", va="center", transform=ax.transAxes)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Phase 5.3 diagnostic — {run.name} ({run_id})", fontsize=12)
    fig.tight_layout()
    out = out_dir / f"{run_id}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"Saved: {out}")

    # Quick summary numbers
    summary = run.summary
    md = out_dir / f"{run_id}.md"
    with open(md, "w") as f:
        f.write(f"# Phase 5.3 diagnostic summary — {run.name}\n\n")
        f.write(f"- W&B run: https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run_id}\n")
        f.write(f"- State: `{run.state}`\n")
        f.write(f"- Runtime: {summary.get('_runtime', 0):.1f}s\n")
        f.write(f"- Final global_step: {summary.get('trainer/global_step', '?')}\n\n")
        f.write("## Final metrics\n\n")
        for k in [
            "train/loss_epoch", "train/loss_decoder_epoch", "train/loss_reward_epoch",
            "train/loss_continue_epoch", "train/loss_dyn_epoch", "train/loss_rep_epoch",
            "train/kl_unclipped_epoch", "train/kl_clip_excess_epoch",
            "val/loss", "val/loss_reward", "val/loss_decoder",
            "val/loss_dyn", "val/loss_rep",
        ]:
            v = summary.get(k)
            if v is not None:
                f.write(f"- `{k}`: {v}\n")
    print(f"Saved: {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
