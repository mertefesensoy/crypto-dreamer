"""Generate Phase 5.0.5 qualitative reconstruction sanity plot.

Loads the final encoder + best decoder, picks a held-out window from
val month 2026-04, masks 40% of timesteps, and overlays
reconstructed-vs-original for all 12 features.

Run:
    python -m scripts.mae_qualitative
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from envs.spot_btc import FEATURE_NAMES, compute_feature_block
from models.encoder import iTransformerEncoder
from models.mae_decoder import MAEDecoder
from training.pretrain_mae import MAEModule


def main() -> None:
    L.seed_everything(7, workers=True)
    ckpt_dir = PROJECT_ROOT / "checkpoints"

    # Load best Lightning checkpoint (has both encoder and decoder).
    ckpts = sorted(ckpt_dir.glob("encoder_mae_full_step=*.ckpt"))
    best_ckpt = ckpts[-1]
    print(f"Loading {best_ckpt.name}")
    module = MAEModule.load_from_checkpoint(str(best_ckpt), map_location="cpu")
    module.eval()

    db_path = PROJECT_ROOT / "data" / "market_ro.duckdb"
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute(
        "SELECT ts, open, high, low, close, volume FROM klines "
        "WHERE symbol = 'BTCUSDT' AND interval = '1m' "
        "  AND ts >= '2026-04-15 00:00:00' AND ts < '2026-04-30 00:00:00' "
        "ORDER BY ts"
    ).df()
    con.close()
    print(f"Held-out window source: {len(df)} rows from {df.ts.min()} to {df.ts.max()}")

    feats = compute_feature_block(df)
    end = 256 + 5000
    window = feats[end - 256 : end]  # (T, F)
    x = torch.from_numpy(window.copy()).unsqueeze(0)  # (1, T, F)

    rng = np.random.default_rng(7)
    mask_np = rng.random(256) < 0.4
    mask = torch.from_numpy(mask_np).unsqueeze(0)  # (1, T)

    with torch.no_grad():
        x_in = x.masked_fill(mask.unsqueeze(-1), 0.0)
        tokens = module.encoder(x_in)
        recon = module.decoder(tokens)  # (1, T, F)

    x_np = x.squeeze(0).numpy()
    recon_np = recon.squeeze(0).numpy()
    masked_idx = np.where(mask_np)[0]
    print(
        f"Per-channel masked-position MSE: "
        f"{((recon_np - x_np)[masked_idx] ** 2).mean(axis=0)}"
    )

    fig, axes = plt.subplots(4, 3, figsize=(18, 14), sharex=True)
    axes = axes.flatten()
    t = np.arange(256)
    for f, name in enumerate(FEATURE_NAMES):
        ax = axes[f]
        ax.plot(t, x_np[:, f], color="black", linewidth=1.0, label="orig", alpha=0.85)
        ax.plot(t, recon_np[:, f], color="tab:blue", linewidth=0.9, label="recon", alpha=0.85)
        ax.scatter(
            masked_idx, x_np[masked_idx, f],
            s=8, color="tab:red", alpha=0.5, label="masked",
        )
        ax.set_title(f"{f:02d} {name}", fontsize=10)
        ax.grid(alpha=0.3)
        if f == 0:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Phase 5.0.5 TS-MAE qualitative reconstruction (held-out 2026-04 window)",
        fontsize=12,
    )
    fig.tight_layout()
    out = PROJECT_ROOT / "checkpoints" / "mae_qualitative.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
