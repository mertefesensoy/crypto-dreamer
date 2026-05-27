"""Smoke tests for the iTransformer encoder + MAE decoder.

Verifies output shapes, finite outputs on random input, and that the
underlying multi-head attention produces probabilities that sum to 1
along the key axis (catches a softmax-axis mistake that would otherwise
silently corrupt training).
"""
from __future__ import annotations

import torch

from models.encoder import iTransformerEncoder
from models.mae_decoder import MAEDecoder


def test_encoder_shape_12_vars() -> None:
    enc = iTransformerEncoder(n_vars=12, seq_len=256, d_model=128, n_layers=2, n_heads=4)
    x = torch.randn(4, 256, 12)
    out = enc(x)
    assert out.shape == (4, 12, 128), f"got {out.shape}"
    assert torch.isfinite(out).all()


def test_encoder_shape_15_vars() -> None:
    enc = iTransformerEncoder(n_vars=15, seq_len=256, d_model=128, n_layers=2, n_heads=4)
    x = torch.randn(4, 256, 15)
    out = enc(x)
    assert out.shape == (4, 15, 128)
    assert torch.isfinite(out).all()


def test_decoder_shape() -> None:
    dec = MAEDecoder(seq_len=256, d_model=128, hidden=128)
    tokens = torch.randn(4, 12, 128)
    out = dec(tokens)
    assert out.shape == (4, 256, 12)
    assert torch.isfinite(out).all()


def test_encoder_decoder_roundtrip() -> None:
    enc = iTransformerEncoder(n_vars=12, seq_len=256, d_model=128, n_layers=2, n_heads=4)
    dec = MAEDecoder(seq_len=256, d_model=128, hidden=128)
    x = torch.randn(2, 256, 12)
    out = dec(enc(x))
    assert out.shape == x.shape


def test_attention_weights_sum_to_one() -> None:
    """Pull attention weights from the underlying MHA and verify softmax property.

    Catches the common bug of softmaxing along the wrong axis: weights
    must sum to 1.0 across the key (variable) dimension.
    """
    enc = iTransformerEncoder(
        n_vars=12, seq_len=256, d_model=128, n_layers=1, n_heads=4, dropout=0.0,
    )
    enc.eval()
    x = torch.randn(2, 256, 12)

    # Replicate the encoder's per-variable token construction so we can
    # call self_attn directly with need_weights=True.
    x_t = x.transpose(1, 2)
    tokens = enc.input_proj(x_t)
    var_idx = torch.arange(12)
    tokens = tokens + enc.var_embed(var_idx)

    layer = enc.encoder.layers[0]
    src = layer.norm1(tokens)  # pre-norm
    _, weights = layer.self_attn(
        src, src, src, need_weights=True, average_attn_weights=False,
    )
    assert weights.shape == (2, 4, 12, 12), f"got {weights.shape}"
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), (
        f"attention weights should sum to 1.0 along key axis, "
        f"got range {sums.min().item():.6f}..{sums.max().item():.6f}"
    )


def test_encoder_gradients_flow() -> None:
    enc = iTransformerEncoder(n_vars=12, seq_len=256, d_model=128, n_layers=2, n_heads=4)
    x = torch.randn(2, 256, 12, requires_grad=False)
    out = enc(x).sum()
    out.backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient in {name}"
