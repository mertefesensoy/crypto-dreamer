"""DreamerV3-style world model — Lightning module wrapping encoder +
RSSM + heads + action embedding.

Per-step prediction targets at trajectory step t:
    decoder  → obs_window[:, t, -1, :]    (15-dim current-bar feature row)
    reward   → reward[:, t]
    continue → continue_flag[:, t]

Per-step KL between posterior(h_t, x_t) and prior(h_t).

Loss aggregation (per spec):
    L_pred  = NLL_decoder + NLL_reward + NLL_continue       (each coef 1.0)
    L_dyn   = KL(stop_grad(post) || prior)   with per-latent free bits
    L_rep   = KL(post || stop_grad(prior))   with per-latent free bits
    L_total = L_pred + 0.5 · L_dyn + 0.1 · L_rep

Burn-in: the first `burn_in` steps' losses are excluded from gradients
because the RSSM hidden state is freshly reset and predictions there
are dominated by the init, not the dynamics.
"""
from __future__ import annotations

import logging

import lightning as L
import torch
import torch.nn as nn

from envs.spot_btc import WINDOW
from models.action_embed import ActionEmbed
from models.encoder import iTransformerEncoder
from models.heads import ContinueHead, DecoderHead, RewardHead
from models.rssm import RSSM

log = logging.getLogger(__name__)


class WorldModel(L.LightningModule):
    def __init__(
        self,
        # Encoder
        n_vars: int = 15,
        seq_len: int = WINDOW,
        d_model: int = 128,
        encoder_layers: int = 4,
        encoder_heads: int = 4,
        encoder_ff: int = 512,
        encoder_dropout: float = 0.1,
        mae_checkpoint: str | None = None,
        # RSSM
        hidden_dim: int = 256,
        n_latents: int = 32,
        n_classes: int = 32,
        rssm_mlp_hidden: int = 256,
        unimix: float = 0.01,
        # Action
        n_actions: int = 5,
        action_emb_dim: int = 32,
        # Heads
        head_hidden: int = 256,
        reward_n_bins: int = 41,
        reward_low: float = -0.2,
        reward_high: float = 0.2,
        # Loss coefficients
        coef_dyn: float = 0.5,
        coef_rep: float = 0.1,
        free_bits: float = 1.0,
        # Optimization
        lr: float = 1e-4,
        weight_decay: float = 1e-6,
        warmup_steps: int = 1000,
        # Burn-in
        burn_in: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = iTransformerEncoder(
            n_vars=n_vars, seq_len=seq_len, d_model=d_model,
            n_layers=encoder_layers, n_heads=encoder_heads, dim_ff=encoder_ff,
            dropout=encoder_dropout, mae_checkpoint=mae_checkpoint,
        )
        # Per-variable tokens are mean-pooled to a single x_t per step.
        self.x_dim = d_model

        self.action_embed = ActionEmbed(n_actions, action_emb_dim)

        self.rssm = RSSM(
            action_emb_dim=action_emb_dim,
            x_dim=self.x_dim,
            hidden_dim=hidden_dim,
            n_latents=n_latents,
            n_classes=n_classes,
            mlp_hidden=rssm_mlp_hidden,
            unimix=unimix,
        )

        feat_dim = hidden_dim + n_latents * n_classes
        self.decoder_head = DecoderHead(feat_dim, n_features=n_vars, hidden=head_hidden)
        self.reward_head = RewardHead(
            feat_dim, n_bins=reward_n_bins, low=reward_low, high=reward_high,
            hidden=head_hidden,
        )
        self.continue_head = ContinueHead(feat_dim, hidden=head_hidden)

        self.coef_dyn = coef_dyn
        self.coef_rep = coef_rep
        self.free_bits = free_bits
        self.unimix = unimix
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.burn_in = burn_in

    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B*T, seq_len, n_vars) -> (B*T, d_model)."""
        tokens = self.encoder(obs)            # (B*T, n_vars, d_model)
        return tokens.mean(dim=1)             # mean-pool variables

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        obs = batch["obs_window"]              # (B, T, S, F)
        action = batch["action"]               # (B, T)
        reward = batch["reward"]               # (B, T)
        cont = batch["continue_flag"]          # (B, T) bool
        is_first = batch["is_first"]           # (B, T) bool

        B, T, S, F = obs.shape

        # Encode all T observations in one big batch.
        obs_flat = obs.reshape(B * T, S, F)
        x_flat = self.encode_obs(obs_flat)     # (B*T, d_model)
        x = x_flat.reshape(B, T, -1)           # (B, T, d_model)

        # Per-step decoder targets: 15-dim row at the current bar (last
        # row of the 256-bar window).
        dec_target = obs[:, :, -1, :]          # (B, T, F)

        # Initial state.
        device = obs.device
        prev_h, prev_z = self.rssm.initial_state(B, device)
        prev_a = torch.zeros(B, dtype=torch.long, device=device)

        # Accumulators.
        loss_dec_sum = torch.tensor(0.0, device=device)
        loss_rew_sum = torch.tensor(0.0, device=device)
        loss_cont_sum = torch.tensor(0.0, device=device)
        kl_dyn_sum = torch.tensor(0.0, device=device)
        kl_rep_sum = torch.tensor(0.0, device=device)
        kl_unclipped_sum = torch.tensor(0.0, device=device)
        n_loss_steps = 0

        for t in range(T):
            # is_first reset (or t==0 always)
            reset = is_first[:, t].clone()
            if t == 0:
                reset = torch.ones_like(reset)

            a_emb = self.action_embed(prev_a)
            h, z, prior_logits, post_logits = self.rssm.step(
                prev_z, prev_h, a_emb, x[:, t], is_first=reset,
            )

            if t >= self.burn_in:
                feat = torch.cat([h, z], dim=-1)
                dec_pred = self.decoder_head(feat)
                rew_logits = self.reward_head(feat)
                cont_logits = self.continue_head(feat)

                loss_dec_sum = loss_dec_sum + DecoderHead.loss(
                    dec_pred, dec_target[:, t]
                )
                loss_rew_sum = loss_rew_sum + self.reward_head.loss(
                    rew_logits, reward[:, t]
                )
                loss_cont_sum = loss_cont_sum + ContinueHead.loss(
                    cont_logits, cont[:, t]
                )

                # KL: compute once, use both directions via stop-grad.
                kl_per_dim = self.rssm.categorical_kl(
                    post_logits, prior_logits, unimix=self.unimix,
                )                                              # (B, n_latents)
                kl_dyn_per_dim = self.rssm.categorical_kl(
                    post_logits.detach(), prior_logits, unimix=self.unimix,
                )
                kl_rep_per_dim = self.rssm.categorical_kl(
                    post_logits, prior_logits.detach(), unimix=self.unimix,
                )
                clip_dyn, raw_dyn = self.rssm.free_bits_kl(
                    kl_dyn_per_dim, free_bits=self.free_bits,
                )
                clip_rep, _ = self.rssm.free_bits_kl(
                    kl_rep_per_dim, free_bits=self.free_bits,
                )
                kl_dyn_sum = kl_dyn_sum + clip_dyn
                kl_rep_sum = kl_rep_sum + clip_rep
                kl_unclipped_sum = kl_unclipped_sum + raw_dyn
                n_loss_steps += 1

            prev_h = h
            prev_z = z
            prev_a = action[:, t]

        # Average across loss-active steps (so coef weights don't drift
        # with T-or-burn_in changes).
        denom = max(n_loss_steps, 1)
        loss_dec = loss_dec_sum / denom
        loss_rew = loss_rew_sum / denom
        loss_cont = loss_cont_sum / denom
        loss_dyn = kl_dyn_sum / denom
        loss_rep = kl_rep_sum / denom
        kl_unclipped = kl_unclipped_sum / denom

        loss_pred = loss_dec + loss_rew + loss_cont
        loss = loss_pred + self.coef_dyn * loss_dyn + self.coef_rep * loss_rep

        on_step = stage == "train"
        log_kw = dict(on_step=on_step, on_epoch=True, batch_size=B)
        # `loss` MUST stay as the live autograd tensor so Lightning can
        # backprop. Components are detached and pre-materialised so we
        # don't create a fresh autograd graph node inside `self.log()`
        # args — that pattern keeps the loss graph alive longer than
        # necessary and adds a per-step CUDA op for nothing.
        kl_clip_excess = (loss_dyn - kl_unclipped).detach().clamp_(min=0)
        self.log(f"{stage}/loss", loss, prog_bar=True, **log_kw)
        self.log(f"{stage}/loss_decoder", loss_dec.detach(), **log_kw)
        self.log(f"{stage}/loss_reward", loss_rew.detach(), **log_kw)
        self.log(f"{stage}/loss_continue", loss_cont.detach(), **log_kw)
        self.log(f"{stage}/loss_dyn", loss_dyn.detach(), **log_kw)
        self.log(f"{stage}/loss_rep", loss_rep.detach(), **log_kw)
        self.log(f"{stage}/kl_unclipped", kl_unclipped.detach(), **log_kw)
        self.log(f"{stage}/kl_clip_excess", kl_clip_excess, **log_kw)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )

        def lr_lambda(step: int) -> float:
            if self.warmup_steps <= 0:
                return 1.0
            if step < self.warmup_steps:
                return (step + 1) / self.warmup_steps
            return 1.0

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return [opt], [{"scheduler": sched, "interval": "step"}]
