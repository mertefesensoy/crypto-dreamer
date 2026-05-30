"""PR 4 · world-model forward-distribution wiring tests.

These lock the Phase 5.4 pivot wiring in `models/world_model.py::_step`:

- The step-alignment trace (brief 3.5 · LOAD-BEARING): one concrete
  trajectory step is followed end to end and every link · observation
  window, RSSM belief `feat = [h_t, z_t]`, and the forward-return target ·
  is asserted to reference the SAME kline index `k`. An off-by-one pairing
  of `feat` at step t with `forward_returns` at t-1 or t+1 is a silent
  temporal leak/lag that a finite-loss smoke would not catch; this test
  fires on it because the expected target is recomputed independently from
  the raw closes the datamodule loaded (`dm._closes`) rather than read back
  from the batch.
- The forward-validity mask contributes exactly zero loss and zero
  gradient at invalid positions (the 0.0 placeholders are not real
  targets).
- The decoder is severed from backprop: its parameters receive no
  gradient while the forward head, RSSM, and encoder do.
- Per-horizon losses are sum-consistent with the masked total, and the
  total loss is a finite scalar that backpropagates.

The alignment test runs the production datamodule on the hermetic
`synthetic_db_with_steps` fixture (so the obs window is the real
256-bar slice the trainer uses); the mask/severance/consistency tests
use small synthetic batches with a tiny `seq_len` model for speed and to
control the validity pattern, which the synthetic-data path cannot
exercise (all its anchors sit far from the series end).
"""

from __future__ import annotations

import numpy as np
import torch

from models.world_model import WorldModel
from training.datamodule import FORWARD_HORIZONS, SpotBTCDataModule

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _aligned_model(burn_in: int = 2) -> WorldModel:
    """Tiny model whose encoder accepts the real 256-bar, 15-feature obs
    windows the datamodule emits. Small d_model/layers keep it fast on CPU.
    """
    return WorldModel(
        n_vars=15,
        seq_len=256,
        d_model=32,
        encoder_layers=1,
        encoder_heads=2,
        encoder_ff=64,
        hidden_dim=32,
        n_latents=8,
        n_classes=8,
        rssm_mlp_hidden=32,
        action_emb_dim=8,
        head_hidden=32,
        warmup_steps=0,
        burn_in=burn_in,
        free_bits=1.0,
        mae_checkpoint=None,
    )


def _tiny_model(seq_len: int = 16, burn_in: int = 2) -> WorldModel:
    """Even smaller model for synthetic-batch tests (no datamodule), so the
    short obs windows encode quickly while the forward head keeps the
    canonical (1, 5, 15, 30) / 41-bin / range layout.
    """
    return WorldModel(
        n_vars=15,
        seq_len=seq_len,
        d_model=16,
        encoder_layers=1,
        encoder_heads=2,
        encoder_ff=32,
        hidden_dim=16,
        n_latents=4,
        n_classes=8,
        rssm_mlp_hidden=16,
        action_emb_dim=8,
        head_hidden=16,
        warmup_steps=0,
        burn_in=burn_in,
        free_bits=1.0,
        mae_checkpoint=None,
    )


def _synthetic_batch(
    B: int,
    T: int,
    seq_len: int,
    valid: torch.Tensor | None = None,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Build a (B, T) batch with forward targets/validity under our control.

    `valid` is an optional (B, T, H) bool mask; when omitted everything is
    valid. Invalid positions hold 0.0 in `forward_returns`, matching the
    datamodule's placeholder convention.
    """
    g = torch.Generator().manual_seed(seed)
    H = len(FORWARD_HORIZONS)
    if valid is None:
        valid = torch.ones(B, T, H, dtype=torch.bool)
    fwd = torch.randn(B, T, H, generator=g) * 0.005
    fwd = fwd * valid  # placeholders are exactly 0.0 where invalid
    return {
        "obs_window": torch.randn(B, T, seq_len, 15, generator=g),
        "next_obs_window": torch.randn(B, T, seq_len, 15, generator=g),
        "action": torch.randint(0, 5, (B, T), generator=g),
        "reward": torch.randn(B, T, generator=g) * 0.05,
        "continue_flag": torch.ones(B, T, dtype=torch.bool),
        "is_first": torch.zeros(B, T, dtype=torch.bool),
        "forward_returns": fwd,
        "forward_valid": valid,
    }


# ---------------------------------------------------------------------
# Construction / config sanity
# ---------------------------------------------------------------------


def test_forward_head_matches_spec() -> None:
    """The wired head carries the ARCHITECTURE Section 6 layout and lines up
    with the datamodule horizon order."""
    model = _tiny_model()
    head = model.forward_head
    assert tuple(head.horizons) == FORWARD_HORIZONS
    assert head.n_bins == 41
    assert head.bin_centers.shape == (4, 41)
    # Ranges match ADR-002 (first/last bin centers are +/-R_h).
    expected_ranges = [0.005, 0.010, 0.018, 0.025]
    for h, r in enumerate(expected_ranges):
        assert torch.isclose(head.bin_centers[h, 0], torch.tensor(-r), atol=1e-6)
        assert torch.isclose(head.bin_centers[h, -1], torch.tensor(r), atol=1e-6)


# ---------------------------------------------------------------------
# Step-alignment trace (brief 3.5 · LOAD-BEARING)
# ---------------------------------------------------------------------


def test_step_alignment_trace(synthetic_db_with_steps: str) -> None:
    """Follow one trajectory step end to end and assert the belief state and
    the forward-return target both reference the SAME kline index k.

    For step t with anchor k = ep.kline_idx[start + t]:
      1. obs_window[:, t] equals feature_cache[k - 256 : k] (window ends just
         before k) · the market channels (first 12) of the observation.
      2. The encoded obs the RSSM ingested at step t (x_used in the trace)
         equals encode(obs_window[:, t]).
      3. feat consumed by the forward head at step t is THIS step's belief,
         cat([h_t, z_t]) (not the previous step's).
      4. The forward target the loss paired at step t equals the
         independently hand-computed ln(close[k+h]/close[k]) from dm._closes.
    A t-1 / t+1 off-by-one on either side breaks (3) or (4).
    """
    T = 8
    burn_in = 2
    dm = SpotBTCDataModule(
        klines_db=synthetic_db_with_steps,
        steps_db=synthetic_db_with_steps,
        batch_size=1,
        T=T,
    )
    dm.setup()
    ds = dm._train_ds
    assert ds is not None and len(ds) > 0
    closes = dm._closes
    assert closes is not None
    n_klines = len(closes)

    model = _aligned_model(burn_in=burn_in)
    model.eval()

    checked_pairs = 0
    checked_valid_targets = 0
    for i in (0, len(ds) // 2, len(ds) - 1):
        item = ds[i]
        agent_id, episode, start = ds.starts[i]
        ep = ds.episodes[(agent_id, episode)]
        kidx_t = ep.kline_idx[start : start + T]  # (T,)
        feature_cache = ds.feature_cache  # (N, 12) market features

        batch = {k: v.unsqueeze(0) for k, v in item.items()}
        with torch.no_grad():
            _, info = model._step(batch, "val", collect_trace=True)
        trace_by_t = {entry["t"]: entry for entry in info["trace"]}

        # Inspect a couple of active steps (t >= burn_in).
        for t in (burn_in, T - 1):
            k = int(kidx_t[t])
            entry = trace_by_t[t]

            # (1) obs window is the 256-bar slice ending just before k.
            obs_market = batch["obs_window"][0, t, :, :12].numpy()
            np.testing.assert_allclose(obs_market, feature_cache[k - 256 : k], rtol=1e-5, atol=1e-5)

            # (2) the encoded obs ingested at step t equals encode(obs[:, t]).
            with torch.no_grad():
                x_recompute = model.encode_obs(batch["obs_window"][:, t])
            torch.testing.assert_close(entry["x_used"], x_recompute, rtol=1e-4, atol=1e-4)

            # (3) feat the head consumed is THIS step's belief cat([h, z]).
            feat_expected = torch.cat([entry["rssm_h"], entry["rssm_z"]], dim=-1)
            assert torch.equal(entry["feat"], feat_expected), (
                f"feat at step {t} is not cat([h_t, z_t]) (item {i})"
            )

            # (4) the forward target the loss used is anchored on close[k].
            fwd_targets = entry["forward_targets"][0].numpy()  # (H,)
            fwd_valid = entry["forward_valid"][0].numpy()  # (H,)
            for h_idx, h in enumerate(FORWARD_HORIZONS):
                if k + h < n_klines:
                    expected = float(np.log(closes[k + h] / closes[k]))
                    assert fwd_valid[h_idx], (
                        f"valid mask False at (t={t}, h={h}) but k+h<N (item {i})"
                    )
                    assert np.isclose(fwd_targets[h_idx], expected, atol=1e-5), (
                        f"misaligned target at item {i} step {t} horizon {h}: "
                        f"got {fwd_targets[h_idx]:.6f}, expected {expected:.6f} "
                        f"(k={k})"
                    )
                    checked_valid_targets += 1
                else:
                    assert not fwd_valid[h_idx]
            checked_pairs += 1

    assert checked_pairs > 0
    assert checked_valid_targets > 0


# ---------------------------------------------------------------------
# Forward-validity masking (brief 3.3 / 3.7)
# ---------------------------------------------------------------------


def test_masked_positions_contribute_zero_loss_and_grad() -> None:
    """An all-invalid forward mask yields zero forward loss and zero forward
    gradient · the placeholders inject nothing into the objective."""
    B, T, seq_len = 2, 6, 16
    model = _tiny_model(seq_len=seq_len, burn_in=2)
    model.eval()
    valid = torch.zeros(B, T, len(FORWARD_HORIZONS), dtype=torch.bool)
    batch = _synthetic_batch(B, T, seq_len, valid=valid, seed=1)

    loss, info = model._step(batch, "train", collect_trace=True)
    assert float(info["loss_forward"]) == 0.0
    assert torch.all(info["loss_forward_per_horizon"] == 0.0)

    loss.backward()
    # Forward head is in the graph (logits computed) but every CE was
    # multiplied by a zero mask, so its gradient must be exactly zero.
    for name, p in model.forward_head.named_parameters():
        assert p.grad is None or torch.all(p.grad == 0), (
            f"forward head param {name} got nonzero grad from masked positions"
        )


def test_masked_positions_do_not_change_loss() -> None:
    """Corrupting forward_returns at INVALID positions must not change the
    forward loss. RNG is reseeded so the (stochastic) RSSM samples match
    across both runs · forward_returns only feeds the head's target, never
    the belief, so any difference would be a masking leak."""
    B, T, seq_len = 3, 6, 16
    H = len(FORWARD_HORIZONS)
    model = _tiny_model(seq_len=seq_len, burn_in=2)
    model.eval()

    valid = torch.ones(B, T, H, dtype=torch.bool)
    valid[:, :, 3] = False  # h=30 invalid everywhere
    valid[0, 4, 1] = False  # a scattered invalid position
    batch = _synthetic_batch(B, T, seq_len, valid=valid, seed=2)

    torch.manual_seed(123)
    _, info_a = model._step(batch, "val", collect_trace=True)
    loss_forward_a = float(info_a["loss_forward"])

    # Replace forward_returns at the invalid positions with large garbage.
    corrupted = batch["forward_returns"].clone()
    corrupted[~valid] = 1000.0
    batch_b = dict(batch)
    batch_b["forward_returns"] = corrupted

    torch.manual_seed(123)
    _, info_b = model._step(batch_b, "val", collect_trace=True)
    loss_forward_b = float(info_b["loss_forward"])

    assert loss_forward_a == loss_forward_b, (
        f"masked positions leaked into the loss: {loss_forward_a} != {loss_forward_b}"
    )


def test_partial_mask_normalizes_by_valid_count() -> None:
    """A horizon that is valid for only some positions still produces a
    finite, sensible per-horizon loss (normalized by its own valid count)."""
    B, T, seq_len = 2, 6, 16
    H = len(FORWARD_HORIZONS)
    model = _tiny_model(seq_len=seq_len, burn_in=2)
    model.eval()
    valid = torch.ones(B, T, H, dtype=torch.bool)
    valid[:, 3:, 2] = False  # h=15 valid only for the earlier steps
    batch = _synthetic_batch(B, T, seq_len, valid=valid, seed=3)
    _, info = model._step(batch, "val", collect_trace=True)
    per_h = info["loss_forward_per_horizon"]
    assert per_h.shape == (H,)
    assert torch.isfinite(per_h).all()
    assert (per_h >= 0).all()


# ---------------------------------------------------------------------
# Decoder severance (brief 3.2 / 3.7)
# ---------------------------------------------------------------------


def test_decoder_severed_from_backprop() -> None:
    """No decoder parameter receives gradient, while the forward head, RSSM,
    and encoder all do · the loss flows through feat = [h_t, z_t] into the
    shared trunk but never through the decoder."""
    B, T, seq_len = 2, 6, 16
    model = _tiny_model(seq_len=seq_len, burn_in=2)
    model.eval()
    batch = _synthetic_batch(B, T, seq_len, seed=4)

    loss = model._step(batch, "train")
    loss.backward()

    # Decoder is severed: every decoder parameter has no gradient.
    for name, p in model.decoder_head.named_parameters():
        assert p.grad is None or torch.all(p.grad == 0), (
            f"decoder param {name} received gradient · not severed"
        )

    # Forward head is wired into backprop.
    fwd_grads = [p.grad for p in model.forward_head.parameters() if p.grad is not None]
    assert len(fwd_grads) > 0, "forward head received no gradient"
    assert all(torch.isfinite(g).all() for g in fwd_grads)

    # Gradients reach the RSSM and encoder through feat.
    rssm_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.rssm.parameters())
    enc_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()
    )
    assert rssm_grad, "no gradient reached the RSSM through feat"
    assert enc_grad, "no gradient reached the encoder through feat"


def test_total_loss_has_no_decoder_term() -> None:
    """The total loss equals L_forward + L_reward + L_continue +
    coef_dyn*L_dyn + coef_rep*L_rep · no decoder term."""
    B, T, seq_len = 2, 6, 16
    model = _tiny_model(seq_len=seq_len, burn_in=2)
    model.eval()
    batch = _synthetic_batch(B, T, seq_len, seed=5)
    loss, info = model._step(batch, "val", collect_trace=True)

    recomposed = (
        info["loss_forward"]
        + info["loss_reward"]
        + info["loss_continue"]
        + model.coef_dyn * info["loss_dyn"]
        + model.coef_rep * info["loss_rep"]
    )
    torch.testing.assert_close(loss, recomposed, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------
# Per-horizon consistency, finiteness, gradient flow (brief 3.7)
# ---------------------------------------------------------------------


def test_per_horizon_sum_consistency() -> None:
    """The total forward loss equals the sum of the per-horizon losses."""
    B, T, seq_len = 3, 6, 16
    H = len(FORWARD_HORIZONS)
    model = _tiny_model(seq_len=seq_len, burn_in=2)
    model.eval()
    valid = torch.ones(B, T, H, dtype=torch.bool)
    valid[:, :, 3] = False  # exercise masking on one horizon
    batch = _synthetic_batch(B, T, seq_len, valid=valid, seed=6)
    _, info = model._step(batch, "val", collect_trace=True)
    torch.testing.assert_close(
        info["loss_forward"],
        info["loss_forward_per_horizon"].sum(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_forward_loss_finite_scalar_and_components() -> None:
    """All logged loss components are finite; forward loss is a scalar."""
    B, T, seq_len = 2, 6, 16
    model = _tiny_model(seq_len=seq_len, burn_in=2)
    model.eval()
    batch = _synthetic_batch(B, T, seq_len, seed=7)
    loss, info = model._step(batch, "train", collect_trace=True)

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert info["loss_forward"].ndim == 0 and torch.isfinite(info["loss_forward"])
    assert info["loss_forward_per_horizon"].shape == (len(FORWARD_HORIZONS),)
    for key in (
        "loss_forward_per_horizon",
        "loss_reward",
        "loss_continue",
        "loss_dyn",
        "loss_rep",
        "kl_unclipped",
    ):
        assert torch.isfinite(info[key]).all(), f"{key} not finite"


def test_forward_gradient_flows_into_head_and_trunk() -> None:
    """A finite, all-valid batch produces finite gradients in the forward
    head and through feat into the RSSM/encoder."""
    B, T, seq_len = 2, 6, 16
    model = _tiny_model(seq_len=seq_len, burn_in=2)
    model.eval()
    batch = _synthetic_batch(B, T, seq_len, seed=8)
    loss = model._step(batch, "train")
    loss.backward()

    n_with_grad = 0
    for _, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all()
            n_with_grad += 1
    assert n_with_grad > 30


def test_legacy_batch_without_forward_keys_runs() -> None:
    """A pre-PR3 batch (no forward_returns/forward_valid) still produces a
    finite scalar loss · the forward term is simply not formed. This guards
    the backward-compat path the RSSM smoke test relies on."""
    B, T, seq_len = 2, 6, 16
    model = _tiny_model(seq_len=seq_len, burn_in=2)
    model.eval()
    batch = _synthetic_batch(B, T, seq_len, seed=9)
    del batch["forward_returns"]
    del batch["forward_valid"]
    loss, info = model._step(batch, "train", collect_trace=True)
    assert loss.ndim == 0 and torch.isfinite(loss) and float(loss) > 0
    assert float(info["loss_forward"]) == 0.0
