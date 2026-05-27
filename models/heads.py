"""World-model output heads.

All three heads share input shape `(B, hidden_dim + n_latents *
n_classes)` — the concatenation of the RSSM's deterministic state
`h` and the flattened stochastic state `z`.

- DecoderHead: MLP -> (B, n_features). Loss is mean MSE; equivalent
  up to constants to a Gaussian NLL with unit variance.
- RewardHead: MLP -> (B, n_bins) logits. The target reward is encoded
  as a two-hot vector over `n_bins` evenly-spaced bin centers in
  `[low, high]`, and the loss is cross-entropy. Inference returns the
  expected value under the predicted distribution.
- ContinueHead: MLP -> (B,) logit. Binary cross-entropy against a
  bool target.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderHead(nn.Module):
    def __init__(self, in_dim: int, n_features: int = 15, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @staticmethod
    def loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


class RewardHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_bins: int = 41,
        low: float = -0.2,
        high: float = 0.2,
        hidden: int = 256,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.low = low
        self.high = high
        self.bin_width = (high - low) / (n_bins - 1)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_bins),
        )
        self.register_buffer(
            "bin_centers", torch.linspace(low, high, n_bins), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def two_hot_encode(self, r: torch.Tensor) -> torch.Tensor:
        """r: (B,) -> (B, n_bins) probability vector summing to 1."""
        r = r.clamp(self.low, self.high)
        # Position in normalized bin units: 0..(n_bins-1)
        pos = (r - self.low) / self.bin_width
        idx_lo = pos.floor().long().clamp(0, self.n_bins - 2)
        w_hi = pos - idx_lo.to(pos.dtype)        # in [0, 1]
        w_lo = 1.0 - w_hi
        target = torch.zeros(r.shape[0], self.n_bins, device=r.device, dtype=pos.dtype)
        target.scatter_(1, idx_lo.unsqueeze(-1), w_lo.unsqueeze(-1))
        target.scatter_add_(1, (idx_lo + 1).unsqueeze(-1), w_hi.unsqueeze(-1))
        return target

    def loss(self, logits: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        target = self.two_hot_encode(r)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(target * log_probs).sum(dim=-1).mean()

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        return (probs * self.bin_centers).sum(dim=-1)


class ContinueHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @staticmethod
    def loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype))
