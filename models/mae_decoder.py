"""Lightweight MLP decoder for TS-MAE pretraining.

Maps per-variable encoder tokens (B, F, d_model) back to the full
(B, T, F) feature matrix. Shared MLP across variables — every variable
uses the same Linear weights to project its token to a T-length series.

Throwaway: only the encoder weights are kept after pretraining.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class MAEDecoder(nn.Module):
    def __init__(
        self,
        seq_len: int = 256,
        d_model: int = 128,
        hidden: int = 128,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, seq_len),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, F, d_model) -> (B, F, T) -> (B, T, F)
        out = self.mlp(tokens)
        return rearrange(out, "b f t -> b t f")
