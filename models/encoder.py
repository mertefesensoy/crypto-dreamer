"""iTransformer encoder for time-series features.

Per-variable tokenization: each input variable's full T-step series is
projected to a single d_model token, then transformer attention runs
across variables (not across time). This captures inter-variable
structure with cost that scales with the number of variables, not with
sequence length — the inverse of the standard time-axis transformer.

Reference: Liu et al., "iTransformer: Inverted Transformers Are Effective
for Time Series Forecasting" (ICLR 2024).

Two consumers:
- Phase 5.0.5 TS-MAE pretraining (n_vars=12, market features only)
- Phase 5.2+ world model (n_vars=15, market + 3 portfolio scalars)

When the world-model encoder is constructed with `mae_checkpoint` set
and the file exists, the shared input projection and all transformer
layer weights are copied verbatim from the pretrained encoder. The
first 12 (market) variable embeddings are also copied; the remaining
embeddings (e.g. 3 portfolio scalars at indices 12..14) are left at
random init.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange

log = logging.getLogger(__name__)


class iTransformerEncoder(nn.Module):
    def __init__(
        self,
        n_vars: int,
        seq_len: int = 256,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        dim_ff: int = 512,
        dropout: float = 0.1,
        mae_checkpoint: str | Path | None = None,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.seq_len = seq_len
        self.d_model = d_model

        # Per-variable input projection: maps a (seq_len,) time series to
        # (d_model,). Shared across variables (iTransformer convention).
        self.input_proj = nn.Linear(seq_len, d_model)

        # Learnable per-variable embedding added to each token.
        self.var_embed = nn.Embedding(n_vars, d_model)

        # Pre-norm transformer for stability under bf16.
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        if mae_checkpoint is not None:
            self._load_mae_checkpoint(Path(mae_checkpoint))

    def _load_mae_checkpoint(self, path: Path) -> None:
        if not path.exists():
            log.warning(
                "mae_checkpoint=%s not found; encoder stays at random init", path
            )
            return
        state = torch.load(path, map_location="cpu", weights_only=True)
        copied: list[str] = []
        skipped: list[str] = []
        with torch.no_grad():
            # Shared input projection — copy verbatim.
            if "input_proj.weight" in state and "input_proj.bias" in state:
                self.input_proj.weight.copy_(state["input_proj.weight"])
                self.input_proj.bias.copy_(state["input_proj.bias"])
                copied.append("input_proj")
            # Variable embedding — copy first min(self.n_vars, ckpt_n_vars) rows.
            if "var_embed.weight" in state:
                ckpt_var = state["var_embed.weight"]
                n_copy = min(self.n_vars, ckpt_var.shape[0])
                self.var_embed.weight[:n_copy].copy_(ckpt_var[:n_copy])
                copied.append(f"var_embed[:{n_copy}]")
                if n_copy < self.n_vars:
                    skipped.append(
                        f"var_embed[{n_copy}:{self.n_vars}] (random init)"
                    )
            # All transformer encoder layer weights (shape-identical).
            ckpt_enc_keys = [k for k in state if k.startswith("encoder.")]
            local_enc = self.encoder.state_dict()
            tx_copied = 0
            for k in ckpt_enc_keys:
                local_key = k[len("encoder.") :]
                if local_key in local_enc and local_enc[local_key].shape == state[k].shape:
                    local_enc[local_key].copy_(state[k])
                    tx_copied += 1
            if tx_copied > 0:
                copied.append(f"encoder.* ({tx_copied} tensors)")
        log.info(
            "Loaded MAE pretrain from %s — copied: %s; skipped: %s",
            path.name, ", ".join(copied) or "none", ", ".join(skipped) or "none",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) with T == seq_len, F == n_vars.
        x = rearrange(x, "b t f -> b f t")
        tokens = self.input_proj(x)                 # (B, F, d_model)
        var_idx = torch.arange(self.n_vars, device=x.device)
        tokens = tokens + self.var_embed(var_idx)   # broadcast over B
        return self.encoder(tokens)                 # (B, F, d_model)
