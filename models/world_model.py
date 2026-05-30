"""DreamerV3-style world model - Lightning module wrapping encoder +
RSSM + heads + action embedding.

Per-step prediction targets at trajectory step t:
    forward  -> forward_returns[:, t]   (B, H) log-returns at H horizons,
                masked by forward_valid[:, t] so series-end placeholders
                contribute zero loss and zero gradient
    reward   -> reward[:, t]
    continue -> continue_flag[:, t]

Per-step KL between posterior(h_t, x_t) and prior(h_t).

Loss aggregation (ARCHITECTURE Section 9 · Phase 5.4 forward-distribution
pivot):
    L_pred  = L_forward + NLL_reward + NLL_continue         (each coef 1.0)
    L_dyn   = KL(stop_grad(post) || prior)   with per-latent free bits
    L_rep   = KL(post || stop_grad(prior))   with per-latent free bits
    L_total = L_pred + coef_dyn * L_dyn + coef_rep * L_rep

L_forward is the two-hot cross-entropy of the ForwardDistributionHead,
summed across horizons (L_forward = CE_1 + CE_5 + CE_15 + CE_30) and
averaged over valid (step, batch) positions per horizon.

The old feature-reconstruction DecoderHead is SEVERED from this path as
of PR 4: `_step` never calls it, never forms a decoder target, and no
decoder term enters the loss or the autograd graph. The module is still
instantiated only for external visualization/diagnostic consumers that
read `model.decoder_head` (see __init__); its parameters receive no
gradient and never update.

Burn-in: the first `burn_in` steps' losses are excluded from gradients
because the RSSM hidden state is freshly reset and predictions there
are dominated by the init, not the dynamics.
"""

from __future__ import annotations

import logging

import lightning as L
import torch

from envs.spot_btc import WINDOW
from models.action_embed import ActionEmbed
from models.encoder import iTransformerEncoder
from models.heads import (
    ContinueHead,
    DecoderHead,
    ForwardDistributionHead,
    RewardHead,
)
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
        # Forward-distribution head (Phase 5.4 · ADR-002). Defaults mirror
        # configs/world_model.yaml so a config-less construction (tests,
        # dream endpoint) gets the canonical horizons/bins/ranges.
        forward_horizons: tuple[int, ...] = (1, 5, 15, 30),
        forward_bins: int = 41,
        forward_ranges: tuple[float, ...] = (0.005, 0.010, 0.018, 0.025),
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
            n_vars=n_vars,
            seq_len=seq_len,
            d_model=d_model,
            n_layers=encoder_layers,
            n_heads=encoder_heads,
            dim_ff=encoder_ff,
            dropout=encoder_dropout,
            mae_checkpoint=mae_checkpoint,
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
        # DecoderHead is retained (instantiated) ONLY for external consumers
        # that still read `model.decoder_head` for visualization/diagnostics
        # (`serve/dream_endpoint.py`, `training/validate.py`). As of PR 4 it is
        # SEVERED from the training path: `_step` never calls it, never forms a
        # decoder target, and no decoder term enters the loss or the autograd
        # graph (brief 3.2 · ADR-002). Its parameters therefore receive no
        # gradient and never update. A future PR re-points those consumers at
        # the forward-distribution head and removes this module.
        self.decoder_head = DecoderHead(feat_dim, n_features=n_vars, hidden=head_hidden)
        self.reward_head = RewardHead(
            feat_dim,
            n_bins=reward_n_bins,
            low=reward_low,
            high=reward_high,
            hidden=head_hidden,
        )
        self.continue_head = ContinueHead(feat_dim, hidden=head_hidden)
        # Forward-distribution head · the Phase 5.4 pivot target. Consumes the
        # same feat = [h_t, z_t] as the reward/continue heads and predicts a
        # categorical distribution over discretized forward log-returns at each
        # horizon (ARCHITECTURE Section 6 · ADR-002).
        self.forward_head = ForwardDistributionHead(
            in_dim=feat_dim,
            horizons=list(forward_horizons),
            n_bins=forward_bins,
            ranges=list(forward_ranges),
            hidden_dim=head_hidden,
        )

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
        tokens = self.encoder(obs)  # (B*T, n_vars, d_model)
        return tokens.mean(dim=1)  # mean-pool variables

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        stage: str,
        collect_trace: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """One training/validation step over a (B, T) trajectory batch.

        Args:
            batch: datamodule batch. Requires `obs_window` (B, T, S, F),
                `action` (B, T), `reward` (B, T), `continue_flag` (B, T),
                `is_first` (B, T). The production datamodule also supplies
                `forward_returns` (B, T, H) and `forward_valid` (B, T, H);
                when both are present the forward-distribution loss is
                formed, otherwise it is skipped (legacy synthetic batches).
            stage: "train" or "val" · controls on_step logging.
            collect_trace: when True, also return a diagnostics dict with
                per-active-step tensors and the materialised loss
                components (used by the step-alignment trace test). The
                default path returns only the scalar loss so training and
                validation are unaffected.

        Returns:
            The scalar training loss, or `(loss, info)` when `collect_trace`.
        """
        obs = batch["obs_window"]  # (B, T, S, F)
        action = batch["action"]  # (B, T)
        reward = batch["reward"]  # (B, T)
        cont = batch["continue_flag"]  # (B, T) bool
        is_first = batch["is_first"]  # (B, T) bool

        # Forward-distribution targets (PR 3). The production datamodule
        # always supplies these; a few legacy synthetic batches (e.g. the
        # RSSM smoke test) omit them, in which case the forward loss is
        # simply not formed · the real training path never hits that branch.
        forward_returns = batch.get("forward_returns")  # (B, T, H) or None
        forward_valid = batch.get("forward_valid")  # (B, T, H) bool or None
        has_forward = forward_returns is not None and forward_valid is not None

        B, T, S, F = obs.shape

        # Encode all T observations in one big batch.
        obs_flat = obs.reshape(B * T, S, F)
        x_flat = self.encode_obs(obs_flat)  # (B*T, d_model)
        x = x_flat.reshape(B, T, -1)  # (B, T, d_model)

        H = len(self.forward_head.horizons)

        # Initial state.
        device = obs.device
        prev_h, prev_z = self.rssm.initial_state(B, device)
        prev_a = torch.zeros(B, dtype=torch.long, device=device)

        # Accumulators.
        loss_rew_sum = torch.tensor(0.0, device=device)
        loss_cont_sum = torch.tensor(0.0, device=device)
        kl_dyn_sum = torch.tensor(0.0, device=device)
        kl_rep_sum = torch.tensor(0.0, device=device)
        kl_unclipped_sum = torch.tensor(0.0, device=device)
        # Forward loss is accumulated as a per-horizon masked cross-entropy
        # numerator plus the count of valid (step, batch) positions per
        # horizon. Normalizing each horizon by its OWN valid count keeps
        # series-end placeholders (0.0, forward_valid=False) out of the
        # signal entirely and makes the per-horizon losses sum to the total
        # (ARCHITECTURE Section 6 · brief 3.3).
        fwd_numer = torch.zeros(H, device=device)
        fwd_denom = torch.zeros(H, device=device)
        n_loss_steps = 0

        trace: list[dict] | None = [] if collect_trace else None

        for t in range(T):
            # is_first reset (or t==0 always)
            reset = is_first[:, t].clone()
            if t == 0:
                reset = torch.ones_like(reset)

            a_emb = self.action_embed(prev_a)
            h, z, prior_logits, post_logits = self.rssm.step(
                prev_z,
                prev_h,
                a_emb,
                x[:, t],
                is_first=reset,
            )

            if t >= self.burn_in:
                # feat is THIS step's belief: the RSSM state produced after
                # ingesting x[:, t] (the encoding of obs_window[:, t], whose
                # 256-bar window ends just before kline k = kline_idx[t]).
                feat = torch.cat([h, z], dim=-1)
                rew_logits = self.reward_head(feat)
                cont_logits = self.continue_head(feat)

                loss_rew_sum = loss_rew_sum + self.reward_head.loss(rew_logits, reward[:, t])
                loss_cont_sum = loss_cont_sum + ContinueHead.loss(cont_logits, cont[:, t])

                fwd_logits = None
                fwd_targets = None
                fwd_valid_t = None
                if has_forward:
                    # Forward head consumes feat at step t; its target is the
                    # forward return anchored on the SAME kline k as step t's
                    # observation (forward_returns[:, t]). Pairing feat[t] with
                    # forward_returns[:, t-1] or [:, t+1] would be a silent
                    # temporal leak/lag · locked by the alignment-trace test.
                    fwd_logits = self.forward_head(feat)  # (B, H, n_bins)
                    fwd_targets = forward_returns[:, t]  # (B, H)
                    fwd_valid_t = forward_valid[:, t]  # (B, H) bool
                    fwd_mask = fwd_valid_t.to(fwd_logits.dtype)  # (B, H)
                    # Mask-aware per-(step, horizon) cross-entropy at the
                    # wiring level (heads.py is decoder-removal-only scope).
                    # Reuse the head's two-hot encoding; the mask zeroes the
                    # placeholder positions before they inject a spurious
                    # "zero-return" signal or any gradient.
                    target_oh = self.forward_head.two_hot_encode(fwd_targets)
                    log_probs = torch.log_softmax(fwd_logits, dim=-1)
                    ce_bh = -(target_oh * log_probs).sum(dim=-1)  # (B, H)
                    ce_bh = ce_bh * fwd_mask
                    fwd_numer = fwd_numer + ce_bh.sum(dim=0)  # (H,)
                    fwd_denom = fwd_denom + fwd_mask.sum(dim=0)  # (H,)

                # KL: compute each direction once via stop-grad.
                kl_dyn_per_dim = self.rssm.categorical_kl(
                    post_logits.detach(),
                    prior_logits,
                    unimix=self.unimix,
                )
                kl_rep_per_dim = self.rssm.categorical_kl(
                    post_logits,
                    prior_logits.detach(),
                    unimix=self.unimix,
                )
                clip_dyn, raw_dyn = self.rssm.free_bits_kl(
                    kl_dyn_per_dim,
                    free_bits=self.free_bits,
                )
                clip_rep, _ = self.rssm.free_bits_kl(
                    kl_rep_per_dim,
                    free_bits=self.free_bits,
                )
                kl_dyn_sum = kl_dyn_sum + clip_dyn
                kl_rep_sum = kl_rep_sum + clip_rep
                kl_unclipped_sum = kl_unclipped_sum + raw_dyn
                n_loss_steps += 1

                if collect_trace:
                    trace.append(
                        {
                            "t": t,
                            "feat": feat.detach(),
                            "rssm_h": h.detach(),
                            "rssm_z": z.detach(),
                            "x_used": x[:, t].detach(),
                            "forward_logits": (
                                fwd_logits.detach() if fwd_logits is not None else None
                            ),
                            "forward_targets": (
                                fwd_targets.detach() if fwd_targets is not None else None
                            ),
                            "forward_valid": (
                                fwd_valid_t.detach() if fwd_valid_t is not None else None
                            ),
                        }
                    )

            prev_h = h
            prev_z = z
            prev_a = action[:, t]

        # Average across loss-active steps (so coef weights don't drift
        # with T-or-burn_in changes).
        denom = max(n_loss_steps, 1)
        loss_rew = loss_rew_sum / denom
        loss_cont = loss_cont_sum / denom
        loss_dyn = kl_dyn_sum / denom
        loss_rep = kl_rep_sum / denom
        kl_unclipped = kl_unclipped_sum / denom

        # Per-horizon masked mean CE; total is the sum across horizons
        # (L_forward = CE_1 + CE_5 + CE_15 + CE_30). fwd_denom carries no
        # gradient (it is a count), so dividing by it does not distort grads.
        if has_forward:
            loss_forward_per_h = fwd_numer / fwd_denom.clamp(min=1.0)  # (H,)
            loss_forward = loss_forward_per_h.sum()
        else:
            loss_forward_per_h = torch.zeros(H, device=device)
            loss_forward = torch.zeros((), device=device)

        loss = (
            loss_forward
            + loss_rew
            + loss_cont
            + self.coef_dyn * loss_dyn
            + self.coef_rep * loss_rep
        )

        on_step = stage == "train"
        log_kw = dict(on_step=on_step, on_epoch=True, batch_size=B)
        # `loss` MUST stay as the live autograd tensor so Lightning can
        # backprop. Components are detached and pre-materialised so we
        # don't create a fresh autograd graph node inside `self.log()`
        # args - that pattern keeps the loss graph alive longer than
        # necessary and adds a per-step CUDA op for nothing.
        kl_clip_excess = (loss_dyn - kl_unclipped).detach().clamp_(min=0)
        self.log(f"{stage}/loss", loss, prog_bar=True, **log_kw)
        self.log(f"{stage}/loss_forward", loss_forward.detach(), **log_kw)
        self.log(f"{stage}/loss_reward", loss_rew.detach(), **log_kw)
        self.log(f"{stage}/loss_continue", loss_cont.detach(), **log_kw)
        self.log(f"{stage}/loss_dyn", loss_dyn.detach(), **log_kw)
        self.log(f"{stage}/loss_rep", loss_rep.detach(), **log_kw)
        self.log(f"{stage}/kl_unclipped", kl_unclipped.detach(), **log_kw)
        self.log(f"{stage}/kl_clip_excess", kl_clip_excess, **log_kw)
        # Per-horizon forward losses (ADR-001) · separate W&B series so the
        # post-diagnostic reweighting decision has data.
        for h_i, hz in enumerate(self.forward_head.horizons):
            self.log(f"{stage}/loss_forward_{hz}", loss_forward_per_h[h_i].detach(), **log_kw)

        if collect_trace:
            info = {
                "loss": loss,
                "loss_forward": loss_forward,
                "loss_forward_per_horizon": loss_forward_per_h,
                "loss_reward": loss_rew,
                "loss_continue": loss_cont,
                "loss_dyn": loss_dyn,
                "loss_rep": loss_rep,
                "kl_unclipped": kl_unclipped,
                "n_loss_steps": n_loss_steps,
                "fwd_denom": fwd_denom,
                "horizons": list(self.forward_head.horizons),
                "trace": trace,
            }
            return loss, info
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        def lr_lambda(step: int) -> float:
            if self.warmup_steps <= 0:
                return 1.0
            if step < self.warmup_steps:
                return (step + 1) / self.warmup_steps
            return 1.0

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return [opt], [{"scheduler": sched, "interval": "step"}]
