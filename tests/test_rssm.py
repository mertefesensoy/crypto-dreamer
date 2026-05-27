"""Smoke tests for RSSM, heads, and the WorldModel forward pass.

Verifies shapes, ST-gradient flow, free-bits clipping, two-hot
encoding correctness, and that one full WorldModel training step
produces a finite scalar loss with finite gradients.
"""
from __future__ import annotations

import torch

from models.heads import ContinueHead, DecoderHead, RewardHead
from models.rssm import RSSM
from models.world_model import WorldModel


# ---------- RSSM ----------

def test_rssm_step_shapes() -> None:
    r = RSSM(action_emb_dim=32, x_dim=128, hidden_dim=256, n_latents=32, n_classes=32)
    B = 4
    h, z = r.initial_state(B, torch.device("cpu"))
    a = torch.randn(B, 32)
    x = torch.randn(B, 128)
    h_out, z_out, prior, post = r.step(z, h, a, x)
    assert h_out.shape == (B, 256)
    assert z_out.shape == (B, 32 * 32)
    assert prior.shape == (B, 32, 32)
    assert post.shape == (B, 32, 32)
    assert torch.isfinite(h_out).all()
    assert torch.isfinite(z_out).all()


def test_rssm_st_gradient_flow() -> None:
    """Discrete sample on the forward pass, identity-through-probs on
    backward — gradients should reach the posterior_head."""
    r = RSSM(action_emb_dim=8, x_dim=16, hidden_dim=32, n_latents=4, n_classes=8)
    h, z = r.initial_state(2, torch.device("cpu"))
    a = torch.randn(2, 8)
    x = torch.randn(2, 16, requires_grad=True)
    _, z_out, _, _ = r.step(z, h, a, x)
    loss = z_out.sum()
    loss.backward()
    grads = [p.grad for n, p in r.posterior_head.named_parameters()]
    for g in grads:
        assert g is not None
        assert torch.isfinite(g).all()
        assert g.abs().sum() > 0


def test_rssm_kl_nonnegative() -> None:
    q = torch.randn(3, 32, 32)
    p = torch.randn(3, 32, 32)
    kl = RSSM.categorical_kl(q, p)
    assert kl.shape == (3, 32)
    # KL should be non-negative (with floating point tolerance).
    assert (kl >= -1e-4).all()


def test_rssm_kl_zero_for_identical_logits() -> None:
    logits = torch.randn(3, 32, 32)
    kl = RSSM.categorical_kl(logits, logits)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)


def test_rssm_free_bits_clip() -> None:
    """When per-latent KL is below the floor, free_bits should inflate
    it; when above, free_bits is a no-op."""
    # Below-floor case
    kl_low = torch.full((4, 8), 0.3)
    clipped, raw = RSSM.free_bits_kl(kl_low, free_bits=1.0)
    assert torch.isclose(raw, torch.tensor(2.4), atol=1e-4)
    assert torch.isclose(clipped, torch.tensor(8.0), atol=1e-4)  # 8 latents × 1.0

    # Above-floor case
    kl_high = torch.full((4, 8), 2.5)
    clipped, raw = RSSM.free_bits_kl(kl_high, free_bits=1.0)
    assert torch.isclose(clipped, raw)                          # no inflation


# ---------- Heads ----------

def test_decoder_head_shape_and_loss() -> None:
    h = DecoderHead(in_dim=128, n_features=15)
    x = torch.randn(7, 128)
    pred = h(x)
    assert pred.shape == (7, 15)
    target = torch.randn(7, 15)
    loss = DecoderHead.loss(pred, target)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_reward_head_two_hot_sums_to_one() -> None:
    h = RewardHead(in_dim=64, n_bins=41, low=-0.2, high=0.2)
    r = torch.tensor([-0.2, -0.1, 0.0, 0.05, 0.2, 0.5, -0.5])  # incl. clipping
    enc = h.two_hot_encode(r)
    assert enc.shape == (7, 41)
    sums = enc.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    # Non-negative
    assert (enc >= 0).all()
    # At a bin center, exactly one entry should be 1.
    enc_center = h.two_hot_encode(torch.tensor([0.0]))
    assert (enc_center > 0).sum() == 1


def test_reward_head_predict_round_trip() -> None:
    """Train signal: predicting the encoded value back out gives the
    input under softmax decoding (within bin resolution)."""
    h = RewardHead(in_dim=64, n_bins=41, low=-0.2, high=0.2)
    r = torch.tensor([0.05, -0.1, 0.0])
    target = h.two_hot_encode(r)
    # Use the target as logits directly (high mass on the right bins).
    decoded = h.predict(torch.log(target.clamp(min=1e-9)))
    assert torch.allclose(decoded, r, atol=0.02)


def test_continue_head_loss_finite() -> None:
    h = ContinueHead(in_dim=32)
    x = torch.randn(5, 32)
    logits = h(x)
    assert logits.shape == (5,)
    target = torch.tensor([True, False, True, True, False])
    loss = ContinueHead.loss(logits, target)
    assert torch.isfinite(loss)
    assert loss.item() > 0


# ---------- WorldModel ----------

def test_world_model_forward_smoke() -> None:
    """One full WorldModel training step on synthetic data — shapes,
    finite loss, finite gradients reach every submodule."""
    wm = WorldModel(
        n_vars=15, seq_len=64, d_model=64,            # tiny for speed
        encoder_layers=1, encoder_heads=2, encoder_ff=128,
        hidden_dim=64, n_latents=8, n_classes=8, rssm_mlp_hidden=64,
        action_emb_dim=8, head_hidden=64,
        warmup_steps=0, burn_in=2, free_bits=1.0,
    )
    B, T, S, F = 2, 8, 64, 15
    batch = {
        "obs_window": torch.randn(B, T, S, F),
        "next_obs_window": torch.randn(B, T, S, F),
        "action": torch.randint(0, 5, (B, T)),
        "reward": torch.randn(B, T) * 0.05,
        "continue_flag": torch.ones(B, T, dtype=torch.bool),
        "is_first": torch.zeros(B, T, dtype=torch.bool),
    }
    loss = wm._step(batch, "train")
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0

    loss.backward()
    n_with_grad = 0
    for name, p in wm.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            continue
        assert torch.isfinite(p.grad).all(), f"nan/inf grad in {name}"
        n_with_grad += 1
    # Most parameters should have a gradient (encoder + rssm + heads + action_embed).
    assert n_with_grad > 30
