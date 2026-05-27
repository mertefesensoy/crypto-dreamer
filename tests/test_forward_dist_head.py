"""Unit tests for ForwardDistributionHead.

Verifies bin-edge computation, two-hot encoding correctness,
loss gradient flow, and shape contracts per the Phase 5.4 pivot
specification (docs/design/ARCHITECTURE.md Section 6).
"""

from __future__ import annotations

import pytest
import torch

from models.heads import ForwardDistributionHead

HORIZONS = [1, 5, 15, 30]
N_BINS = 41
RANGES = [0.005, 0.010, 0.018, 0.025]
IN_DIM = 1280


def _make_head(in_dim: int = IN_DIM) -> ForwardDistributionHead:
    return ForwardDistributionHead(
        in_dim=in_dim,
        horizons=HORIZONS,
        n_bins=N_BINS,
        ranges=RANGES,
    )


def test_init_validates_inputs() -> None:
    """Mismatched lengths, zero/negative range, and n_bins < 2 raise ValueError."""
    with pytest.raises(ValueError, match="same length"):
        ForwardDistributionHead(IN_DIM, [1, 5], N_BINS, [0.005])
    with pytest.raises(ValueError, match="positive"):
        ForwardDistributionHead(IN_DIM, [1], N_BINS, [0.0])
    with pytest.raises(ValueError, match="positive"):
        ForwardDistributionHead(IN_DIM, [1], N_BINS, [-0.005])
    with pytest.raises(ValueError, match="n_bins"):
        ForwardDistributionHead(IN_DIM, [1], 1, [0.005])


def test_bin_centers_shape_and_values() -> None:
    """bin_centers has shape (H, n_bins) with correct edge and center values."""
    head = _make_head()
    assert head.bin_centers.shape == (4, N_BINS), f"got {head.bin_centers.shape}"
    assert head.bin_widths.shape == (4,), f"got {head.bin_widths.shape}"

    for h, R in enumerate(RANGES):
        centers = head.bin_centers[h]
        assert torch.isclose(centers[0], torch.tensor(-R), atol=1e-6), (
            f"horizon {h}: first center {centers[0]} != -{R}"
        )
        assert torch.isclose(centers[-1], torch.tensor(R), atol=1e-6), (
            f"horizon {h}: last center {centers[-1]} != {R}"
        )
        assert torch.isclose(centers[20], torch.tensor(0.0), atol=1e-6), (
            f"horizon {h}: center bin {centers[20]} != 0.0"
        )

        expected_width = 2 * R / (N_BINS - 1)
        assert torch.isclose(head.bin_widths[h], torch.tensor(expected_width), atol=1e-8), (
            f"horizon {h}: bin_width {head.bin_widths[h]} != {expected_width}"
        )


def test_forward_output_shape() -> None:
    """Forward pass produces (B, H, n_bins) logits, all finite."""
    head = _make_head()
    feat = torch.randn(8, IN_DIM)
    logits = head(feat)
    assert logits.shape == (8, 4, N_BINS), f"got {logits.shape}"
    assert torch.isfinite(logits).all(), "non-finite logits"


def test_two_hot_encode_at_exact_center() -> None:
    """Target at an exact bin center produces near-one-hot at that bin."""
    head = _make_head()
    center_val = head.bin_centers[0, 10].item()
    targets = torch.zeros(1, 4)
    targets[0, 0] = center_val

    encoded = head.two_hot_encode(targets)
    row = encoded[0, 0]

    assert torch.isclose(row.sum(), torch.tensor(1.0), atol=1e-5), f"sum {row.sum()} != 1.0"
    nonzero = row.nonzero(as_tuple=True)[0]
    assert len(nonzero) <= 2, f"expected at most 2 nonzero, got {len(nonzero)}"
    assert 10 in nonzero.tolist(), f"expected bin 10 in nonzero indices {nonzero.tolist()}"


def test_two_hot_encode_sums_to_one() -> None:
    """Encoded output sums to 1.0 along bin dim for every (batch, horizon)."""
    head = _make_head()
    targets = torch.randn(16, 4) * 0.01
    encoded = head.two_hot_encode(targets)
    sums = encoded.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), (
        f"max deviation from 1.0: {(sums - 1.0).abs().max()}"
    )


def test_two_hot_encode_clamps_out_of_range() -> None:
    """Targets beyond +/-range clip to the edge bins."""
    head = _make_head()
    targets = torch.zeros(1, 4)

    # Above +range for horizon 0
    targets[0, 0] = 1.0
    encoded = head.two_hot_encode(targets)
    row = encoded[0, 0]
    assert torch.isclose(row[N_BINS - 1], torch.tensor(1.0), atol=1e-5), (
        f"expected all mass at last bin, got row[-1]={row[-1]}"
    )
    assert torch.isclose(row[: N_BINS - 1].sum(), torch.tensor(0.0), atol=1e-5)

    # Below -range for horizon 0
    targets[0, 0] = -1.0
    encoded = head.two_hot_encode(targets)
    row = encoded[0, 0]
    assert torch.isclose(row[0], torch.tensor(1.0), atol=1e-5), (
        f"expected all mass at first bin, got row[0]={row[0]}"
    )
    assert torch.isclose(row[1:].sum(), torch.tensor(0.0), atol=1e-5)


def test_two_hot_encode_mid_bin() -> None:
    """Target at midpoint between adjacent bins produces w_lo == w_hi == 0.5."""
    head = _make_head()
    # Horizon 0: range 0.005, bin_width 0.00025
    # bin_centers[0, 5] = -0.00375, bin_centers[0, 6] = -0.00350
    # Midpoint = -0.003625
    targets = torch.zeros(1, 4)
    targets[0, 0] = -0.003625

    encoded = head.two_hot_encode(targets)
    row = encoded[0, 0]

    assert torch.isclose(row[5], torch.tensor(0.5), atol=1e-6), (
        f"expected w_lo=0.5 at bin 5, got {row[5]}"
    )
    assert torch.isclose(row[6], torch.tensor(0.5), atol=1e-6), (
        f"expected w_hi=0.5 at bin 6, got {row[6]}"
    )


def test_loss_gradient_flows() -> None:
    """Loss backward produces finite non-None grads for all parameters."""
    head = _make_head(in_dim=64)
    feat = torch.randn(4, 64)
    targets = torch.randn(4, 4) * 0.005

    logits = head(feat)
    loss = head.loss(logits, targets)
    loss.backward()

    assert loss.shape == (), f"loss should be scalar, got {loss.shape}"
    for name, p in head.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient in {name}"


def test_per_horizon_loss_consistency() -> None:
    """per_horizon_loss summed across H equals loss()."""
    head = _make_head(in_dim=64)
    feat = torch.randn(8, 64)
    targets = torch.randn(8, 4) * 0.005

    logits = head(feat)
    total = head.loss(logits, targets)
    per_h = head.per_horizon_loss(logits, targets)

    assert torch.isclose(per_h.sum(), total, atol=1e-5), (
        f"sum of per_horizon {per_h.sum()} != total loss {total}"
    )


def test_predict_inside_range() -> None:
    """predict() returns (B, H) with values within each horizon's range."""
    head = _make_head(in_dim=64)
    feat = torch.randn(8, 64)
    logits = head(feat)
    preds = head.predict(logits)

    assert preds.shape == (8, 4), f"got {preds.shape}"
    for h, R in enumerate(RANGES):
        assert (preds[:, h] >= -R).all(), f"horizon {h}: prediction below -{R}"
        assert (preds[:, h] <= R).all(), f"horizon {h}: prediction above {R}"


def test_per_horizon_loss_shape() -> None:
    """per_horizon_loss returns shape (H,) with all finite entries."""
    head = _make_head(in_dim=64)
    feat = torch.randn(4, 64)
    targets = torch.randn(4, 4) * 0.005

    logits = head(feat)
    per_h = head.per_horizon_loss(logits, targets)

    assert per_h.shape == (4,), f"got {per_h.shape}"
    assert torch.isfinite(per_h).all(), "non-finite per-horizon loss"


def test_independence_across_batch() -> None:
    """Different targets produce different encodings, each a valid distribution."""
    head = _make_head()
    targets = torch.tensor(
        [
            [0.001, 0.005, 0.010, 0.015],
            [-0.003, -0.008, -0.015, -0.020],
        ]
    )

    encoded = head.two_hot_encode(targets)

    for h in range(4):
        row_0 = encoded[0, h]
        row_1 = encoded[1, h]
        assert not torch.equal(row_0, row_1), (
            f"horizon {h}: identical encodings for different targets"
        )
        assert torch.isclose(row_0.sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.isclose(row_1.sum(), torch.tensor(1.0), atol=1e-6)
