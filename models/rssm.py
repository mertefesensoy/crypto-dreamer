"""DreamerV3-style Recurrent State-Space Model (RSSM).

Each step's state is the pair (h, z):
- h ∈ ℝ^hidden_dim is the deterministic state (GRU hidden).
- z ∈ {0,1}^(n_latents × n_classes) is the stochastic state — a flat
  one-hot of `n_latents` independent categoricals over `n_classes`
  outcomes each, sampled with straight-through gradients.

Two paths produce z at each step:
- Prior     p(z_t | h_t)        from h_t alone (used during imagination)
- Posterior q(z_t | h_t, x_t)   from h_t and the encoded observation

During world-model training, z_t is sampled from the posterior; the KL
between posterior and prior is the dynamics-vs-representation
regularizer that lets the prior eventually carry observation
information forward without re-encoding.

Unimix noise (1%) clamps each categorical's probability mass away from
0 so KL never explodes to inf.

Reference: Hafner et al., "Mastering Diverse Domains through World
Models" (DreamerV3, 2023).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSM(nn.Module):
    def __init__(
        self,
        action_emb_dim: int,
        x_dim: int,
        hidden_dim: int = 256,
        n_latents: int = 32,
        n_classes: int = 32,
        mlp_hidden: int = 256,
        unimix: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_latents = n_latents
        self.n_classes = n_classes
        self.z_dim = n_latents * n_classes
        self.unimix = unimix

        # Pre-GRU MLP: projects [z_{t-1}, action_emb_{t-1}] to a single vector.
        self.pre_gru = nn.Sequential(
            nn.Linear(self.z_dim + action_emb_dim, mlp_hidden),
            nn.GELU(),
        )
        self.gru = nn.GRUCell(mlp_hidden, hidden_dim)

        # ADR-006 linear-prior disambiguation (2026-05-31): the prior is
        # restricted to a bare affine map h_t -> z-logits. The hidden layer
        # and its GELU (the prior's expressive capacity) are removed · that
        # is the single experimental variable. The original prior_head had
        # no normalization (no LayerNorm on the input, between layers, or on
        # the output), so per ADR-006 Section 2.2 the no-norm replacement is
        # a plain Linear with nothing to preserve. posterior_head below is a
        # separate module and is left unchanged.
        self.prior_head = nn.Linear(hidden_dim, self.z_dim)
        self.posterior_head = nn.Sequential(
            nn.Linear(hidden_dim + x_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, self.z_dim),
        )

    def initial_state(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(batch_size, self.z_dim, device=device)
        return h, z

    def step(
        self,
        prev_z: torch.Tensor,
        prev_h: torch.Tensor,
        action_emb: torch.Tensor,
        x_t: torch.Tensor,
        is_first: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One RSSM update.

        Returns (h, z, prior_logits, posterior_logits) where logits are
        shape (B, n_latents, n_classes).
        """
        if is_first is not None:
            keep = (~is_first).float().unsqueeze(-1)
            prev_z = prev_z * keep
            prev_h = prev_h * keep

        gru_in = self.pre_gru(torch.cat([prev_z, action_emb], dim=-1))
        h = self.gru(gru_in, prev_h)

        prior = self.prior_head(h).reshape(-1, self.n_latents, self.n_classes)
        post = self.posterior_head(torch.cat([h, x_t], dim=-1)).reshape(
            -1, self.n_latents, self.n_classes
        )
        z = self.sample_st(post).reshape(-1, self.z_dim)
        return h, z, prior, post

    def sample_st(self, logits: torch.Tensor) -> torch.Tensor:
        """Categorical sample with straight-through gradients.

        Forward: returns the one-hot of the multinomial draw.
        Backward: gradient flows through (softmax) probs as identity.
        Includes unimix noise so probabilities never reach exactly 0.
        """
        probs = F.softmax(logits, dim=-1)
        if self.unimix > 0:
            probs = (1.0 - self.unimix) * probs + self.unimix / self.n_classes
        flat = probs.reshape(-1, self.n_classes)
        idx = torch.multinomial(flat, num_samples=1).reshape(probs.shape[:-1])
        one_hot = F.one_hot(idx, self.n_classes).to(probs.dtype)
        return one_hot + probs - probs.detach()

    @staticmethod
    def categorical_kl(
        q_logits: torch.Tensor,
        p_logits: torch.Tensor,
        unimix: float = 0.01,
    ) -> torch.Tensor:
        """Per-latent KL(q || p), summed over classes, returning shape
        (B, n_latents) in nats. Both logit tensors are (B, n_latents,
        n_classes). Unimix is applied to both distributions for a fair
        comparison and to prevent log(0)."""
        n_classes = q_logits.shape[-1]
        q = F.softmax(q_logits, dim=-1)
        p = F.softmax(p_logits, dim=-1)
        if unimix > 0:
            q = (1.0 - unimix) * q + unimix / n_classes
            p = (1.0 - unimix) * p + unimix / n_classes
        log_q = torch.log(q)
        log_p = torch.log(p)
        return (q * (log_q - log_p)).sum(dim=-1)

    @staticmethod
    def free_bits_kl(
        kl_per_dim: torch.Tensor,
        free_bits: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the per-latent free-bits clip and return
        (loss_clipped, raw_avg).

        - kl_per_dim: shape (B, n_latents) — per-sample, per-latent KL.
        - Reduce over batch first, then clip per latent at >=free_bits,
          then sum over latents. Matches the user spec wording.
        - Returns (clipped_loss, unclipped_total_per_dim_summed) for
          logging.
        """
        avg = kl_per_dim.mean(dim=0)  # (n_latents,)
        clipped = avg.clamp(min=free_bits)  # per-latent floor
        return clipped.sum(), avg.sum()
