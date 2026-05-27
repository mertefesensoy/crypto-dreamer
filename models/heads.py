"""World-model output heads.

All heads share input shape `(B, hidden_dim + n_latents *
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
- ForwardDistributionHead: MLP -> (B, H, n_bins) logits over H
  forward-return horizons. Two-hot targets, cross-entropy loss
  summed across horizons.
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
        self.register_buffer("bin_centers", torch.linspace(low, high, n_bins), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def two_hot_encode(self, r: torch.Tensor) -> torch.Tensor:
        """r: (B,) -> (B, n_bins) probability vector summing to 1."""
        r = r.clamp(self.low, self.high)
        # Position in normalized bin units: 0..(n_bins-1)
        pos = (r - self.low) / self.bin_width
        idx_lo = pos.floor().long().clamp(0, self.n_bins - 2)
        w_hi = pos - idx_lo.to(pos.dtype)  # in [0, 1]
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


class ForwardDistributionHead(nn.Module):
    """Forward-return distribution prediction head for the Phase 5.4
    pivot. Predicts categorical distributions over discretized
    log-returns at H horizons given the world-model state
    ``feat = [h_t, z_t]``.

    Mirrors RewardHead's two-hot encoding and cross-entropy loss
    pattern, generalized to per-horizon symmetric bin ranges.

    Reference: docs/design/ARCHITECTURE.md Section 6 + ADR-002.
    """

    def __init__(
        self,
        in_dim: int,
        horizons: list[int],
        n_bins: int,
        ranges: list[float],
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if len(horizons) != len(ranges):
            raise ValueError(
                f"horizons and ranges must have the same length, "
                f"got {len(horizons)} and {len(ranges)}"
            )
        if any(r <= 0 for r in ranges):
            raise ValueError(f"all ranges must be positive, got {ranges}")
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")

        self.horizons = horizons
        self.n_bins = n_bins
        H = len(horizons)

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.proj = nn.Linear(hidden_dim, H * n_bins)

        ranges_t = torch.tensor(ranges, dtype=torch.float32)
        bin_widths = 2 * ranges_t / (n_bins - 1)
        centers = torch.stack([torch.linspace(-r, r, n_bins) for r in ranges])
        self.register_buffer("ranges", ranges_t, persistent=False)
        self.register_buffer("bin_widths", bin_widths, persistent=False)
        self.register_buffer("bin_centers", centers, persistent=False)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, in_dim) -> logits: (B, H, n_bins)."""
        h = self.trunk(feat)
        return self.proj(h).view(feat.shape[0], len(self.horizons), self.n_bins)

    def two_hot_encode(self, targets: torch.Tensor) -> torch.Tensor:
        """targets: (B, H) -> (B, H, n_bins) probability vectors summing to 1."""
        B, H = targets.shape
        clamped = targets.clamp(-self.ranges, self.ranges)
        pos = (clamped + self.ranges) / self.bin_widths
        idx_lo = pos.floor().long().clamp(0, self.n_bins - 2)
        w_hi = pos - idx_lo.to(pos.dtype)
        w_lo = 1.0 - w_hi
        target = torch.zeros(B, H, self.n_bins, device=targets.device, dtype=pos.dtype)
        target.scatter_(2, idx_lo.unsqueeze(-1), w_lo.unsqueeze(-1))
        target.scatter_add_(2, (idx_lo + 1).unsqueeze(-1), w_hi.unsqueeze(-1))
        return target

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss summed across horizons, averaged over batch.

        logits: (B, H, n_bins), targets: (B, H) -> scalar.
        """
        target_oh = self.two_hot_encode(targets)
        log_probs = F.log_softmax(logits, dim=-1)
        ce_per_horizon = -(target_oh * log_probs).sum(dim=-1)  # (B, H)
        return ce_per_horizon.sum(dim=-1).mean()

    def per_horizon_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Per-horizon mean CE, no summation across horizons.

        logits: (B, H, n_bins), targets: (B, H) -> (H,).
        """
        target_oh = self.two_hot_encode(targets)
        log_probs = F.log_softmax(logits, dim=-1)
        ce_per_horizon = -(target_oh * log_probs).sum(dim=-1)  # (B, H)
        return ce_per_horizon.mean(dim=0)

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Expected log-return per horizon. logits: (B, H, n_bins) -> (B, H)."""
        probs = F.softmax(logits, dim=-1)
        return (probs * self.bin_centers).sum(dim=-1)
