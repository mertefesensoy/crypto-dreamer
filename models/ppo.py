"""ActorCritic network for the ADR-007 model-free PPO baseline (Track A).

Contract:
    forward(window, portfolio) -> (logits, value)
      window:    (B, 256, 12) float32 - the env's feature window (time, feature)
      portfolio: (B, 3)       float32 - [alloc, cash_ratio, log_equity_vs_start]
      logits:    (B, n_actions)       - unnormalized action preferences
      value:     (B,)                 - state-value estimate

Encoder: a small Conv1d stack over the TIME axis (in_channels = 12 features):
12->32 k8 s4, 32->64 k4 s2, 64->64 k4 s2, GELU between, flatten, Linear to
``hidden``. The 3 portfolio scalars are concatenated post-encoder and fused
by one more Linear+GELU before the policy and value heads. Sized for a 4070
laptop GPU: ~0.33M parameters, well under the 2M budget.

Initialization (PPO convention): orthogonal everywhere - gain sqrt(2) for
hidden conv/linear layers, 0.01 for the policy head (near-uniform initial
policy so early exploration is unbiased), 1.0 for the value head; all biases
zero.

Reconstruction contract: EVERY constructor kwarg is recorded verbatim in
``self.ctor_kwargs`` so a checkpoint's ``arch`` entry suffices to rebuild
the exact architecture (``ActorCritic(**ckpt["arch"])``).
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _conv_out_len(length: int, kernel: int, stride: int) -> int:
    """Output length of an unpadded 1-d convolution (floor convention)."""
    return (length - kernel) // stride + 1


class ActorCritic(nn.Module):
    """Shared-encoder discrete actor-critic for the (window, portfolio) obs."""

    def __init__(
        self,
        n_actions: int = 5,
        hidden: int = 256,
        window_len: int = 256,
        n_features: int = 12,
        portfolio_dim: int = 3,
    ):
        super().__init__()
        # Recorded for checkpoint-driven reconstruction (ADR-007 contract).
        self.ctor_kwargs = dict(
            n_actions=n_actions,
            hidden=hidden,
            window_len=window_len,
            n_features=n_features,
            portfolio_dim=portfolio_dim,
        )

        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=8, stride=4),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=4, stride=2),
            nn.GELU(),
        )
        length = _conv_out_len(window_len, 8, 4)
        length = _conv_out_len(length, 4, 2)
        length = _conv_out_len(length, 4, 2)
        if length <= 0:
            raise ValueError(f"window_len {window_len} too short for conv stack")
        flat = 64 * length  # 896 for window_len=256

        self.proj = nn.Sequential(nn.Linear(flat, hidden), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(hidden + portfolio_dim, hidden), nn.GELU())
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal init: sqrt(2) hidden, 0.01 policy head, 1.0 value head."""
        for module in [*self.conv, *self.proj, *self.fuse]:
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(
        self, window: torch.Tensor, portfolio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, 256, 12), (B, 3) -> logits (B, n_actions), value (B,)."""
        x = self.conv(window.transpose(1, 2))  # (B, 12, T) -> (B, 64, L)
        x = self.proj(x.flatten(1))            # (B, hidden)
        h = self.fuse(torch.cat([x, portfolio], dim=1))
        return self.policy_head(h), self.value_head(h).squeeze(-1)
