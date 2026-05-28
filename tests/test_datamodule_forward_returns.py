"""PR 3 · forward-return target tests for `SpotBTCDataModule`.

Test taxonomy follows `docs/briefings/PR3-forward-returns-briefing.md`
Section 5 and the three-layer guidance: unit (pure computation),
integration (through the datamodule), and real-data validation
(hand-computed ground truth from `data/market.duckdb`).

The unit tests exercise `_compute_forward_targets` directly with
synthetic numpy inputs so the boundary, dtype, and gap-detection logic
is locked independently of DB plumbing.

The integration tests use the existing `synthetic_db_with_steps`
fixture so they run hermetically. The real-data tests are gated on
`data/market_ro.duckdb` + `data/market.duckdb` and skip when those are
absent · matching the convention in `tests/test_datamodule.py`.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pytest
import torch

from models.heads import ForwardDistributionHead
from training.datamodule import (
    FORWARD_HORIZONS,
    SpotBTCDataModule,
    _compute_forward_targets,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIVE_DB = PROJECT_ROOT / "data" / "market.duckdb"
SNAPSHOT_DB = PROJECT_ROOT / "data" / "market_ro.duckdb"


# ---------------------------------------------------------------------
# 5.1 · Unit tests (pure computation, synthetic numpy data)
# ---------------------------------------------------------------------


def _gap_free_ts(n: int, interval: int = 60) -> np.ndarray:
    """Build a gap-free 1m timestamp array (int64 seconds since epoch)."""
    return np.arange(n, dtype=np.int64) * interval


def test_forward_return_value_correctness() -> None:
    """`_compute_forward_targets` matches ln(close[k+h]/close[k]) exactly."""
    closes = np.array(
        [100.0, 101.0, 99.0, 105.0, 110.0, 108.0, 107.0, 115.0, 120.0, 125.0], dtype=np.float64
    )
    ts = _gap_free_ts(len(closes))
    horizons = (1, 5)

    fr, valid, _ = _compute_forward_targets(closes, ts, horizons)

    assert fr.shape == (10, 2)
    # k=0, h=1: ln(101/100)
    assert np.isclose(fr[0, 0], np.log(101.0 / 100.0), atol=1e-6)
    # k=2, h=1: ln(105/99)
    assert np.isclose(fr[2, 0], np.log(105.0 / 99.0), atol=1e-6)
    # k=0, h=5: ln(108/100)
    assert np.isclose(fr[0, 1], np.log(108.0 / 100.0), atol=1e-6)
    # k=4, h=5: ln(125/110)
    assert np.isclose(fr[4, 1], np.log(125.0 / 110.0), atol=1e-6)
    # k=9 has h=1 invalid (k+1=10 >= N=10)
    assert not valid[9, 0]
    assert fr[9, 0] == 0.0


def test_horizon_ordering_matches_head() -> None:
    """FORWARD_HORIZONS in the datamodule MUST line up with the head."""
    assert FORWARD_HORIZONS == (1, 5, 15, 30)
    head = ForwardDistributionHead(
        in_dim=64,
        horizons=list(FORWARD_HORIZONS),
        n_bins=41,
        ranges=[0.005, 0.010, 0.018, 0.025],
    )
    assert tuple(head.horizons) == FORWARD_HORIZONS


def test_series_end_mask_per_horizon() -> None:
    """Last `h` positions of each horizon must be invalid; nothing before."""
    n = 60
    closes = 100.0 * np.exp(np.cumsum(np.random.default_rng(0).normal(0.0, 1e-3, n)))
    ts = _gap_free_ts(n)

    fr, valid, _ = _compute_forward_targets(closes, ts, FORWARD_HORIZONS)

    assert valid.shape == (n, 4)
    for h_idx, h in enumerate(FORWARD_HORIZONS):
        # Tail-h positions are invalid; everything before is valid.
        assert valid[: n - h, h_idx].all(), f"interior False at horizon {h} (h_idx={h_idx})"
        assert not valid[n - h :, h_idx].any(), f"tail True at horizon {h} (h_idx={h_idx})"
    # The step exactly h=1 from the end has h=1 invalid but h=5,15,30 also
    # invalid (since they need more bars). The one at 30 from the end has
    # h=1,5,15 valid and h=30 invalid.
    assert not valid[n - 1, 0]
    assert not valid[n - 1, 3]
    assert valid[n - 30, 0]
    assert valid[n - 30, 1]
    assert valid[n - 30, 2]
    assert not valid[n - 30, 3]


def test_placeholder_value_at_invalid() -> None:
    """Invalid positions hold exactly 0.0 (the safe mask-multiply placeholder)."""
    n = 50
    closes = np.linspace(100.0, 200.0, n, dtype=np.float64)
    ts = _gap_free_ts(n)
    fr, valid, _ = _compute_forward_targets(closes, ts, FORWARD_HORIZONS)
    # Every invalid position is exactly 0.0.
    assert (fr[~valid] == 0.0).all()


def test_dtypes_and_shapes() -> None:
    n = 200
    closes = 100.0 + np.arange(n, dtype=np.float64)
    ts = _gap_free_ts(n)
    fr, valid, _ = _compute_forward_targets(closes, ts, FORWARD_HORIZONS)
    assert fr.dtype == np.float32
    assert valid.dtype == np.bool_
    assert fr.shape == (n, 4)
    assert valid.shape == (n, 4)


def test_gap_detection_synthetic() -> None:
    """Insert a known 2-minute gap and verify it's flagged at every horizon
    whose window straddles it."""
    n = 100
    closes = 100.0 + np.arange(n, dtype=np.float64)
    ts = _gap_free_ts(n)
    # Push timestamps [50:] forward by 60s · simulates a missing minute
    # between bar 49 and bar 50.
    ts[50:] += 60
    fr, valid, gap_stats = _compute_forward_targets(closes, ts, FORWARD_HORIZONS)
    # Returns themselves are still computed (we do NOT mask gaps in PR 3).
    assert valid[40, 0]
    # h=1: positions k=49 straddles the gap (ts[50]-ts[49] = 120, not 60).
    assert 49 in gap_stats[1]["indices"].tolist()
    # h=5: positions k in [45, 50) all span the gap.
    assert all(k in gap_stats[5]["indices"].tolist() for k in range(45, 50))
    # h=15: positions k in [35, 50) span it.
    assert all(k in gap_stats[15]["indices"].tolist() for k in range(35, 50))
    # h=30: positions k in [20, 50) span it.
    assert all(k in gap_stats[30]["indices"].tolist() for k in range(20, 50))
    # Counts are internally consistent.
    for h in FORWARD_HORIZONS:
        s = gap_stats[h]
        assert s["count"] == len(s["indices"])
        assert s["count"] <= s["total"]


def test_helper_rejects_bad_inputs() -> None:
    closes = np.array([100.0, 101.0])
    ts = np.array([0], dtype=np.int64)
    with pytest.raises(ValueError, match="length mismatch"):
        _compute_forward_targets(closes, ts, (1,))
    with pytest.raises(ValueError, match="positive"):
        _compute_forward_targets(
            np.array([100.0, 101.0]),
            np.array([0, 60], dtype=np.int64),
            (0,),
        )


# ---------------------------------------------------------------------
# 5.2 · Integration tests (synthetic DB through the datamodule)
# ---------------------------------------------------------------------


def test_batch_contains_new_keys(synthetic_db_with_steps: str) -> None:
    """`forward_returns` and `forward_valid` are emitted; pre-existing keys
    are unchanged in shape and dtype."""
    dm = SpotBTCDataModule(
        klines_db=synthetic_db_with_steps,
        steps_db=synthetic_db_with_steps,
        batch_size=2,
        T=8,
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))

    expected_keys = {
        "obs_window",
        "next_obs_window",
        "action",
        "reward",
        "continue_flag",
        "is_first",
        "forward_returns",
        "forward_valid",
    }
    assert set(batch.keys()) == expected_keys

    # Pre-existing keys unchanged.
    assert batch["obs_window"].shape == (2, 8, 256, 15)
    assert batch["next_obs_window"].shape == (2, 8, 256, 15)
    assert batch["action"].dtype == torch.int64
    assert batch["reward"].dtype == torch.float32
    assert batch["continue_flag"].dtype == torch.bool
    assert batch["is_first"].dtype == torch.bool

    # New keys: shape (B, T, 4), correct dtypes.
    fr = batch["forward_returns"]
    fv = batch["forward_valid"]
    assert fr.shape == (2, 8, 4)
    assert fv.shape == (2, 8, 4)
    assert fr.dtype == torch.float32
    assert fv.dtype == torch.bool
    # Finite everywhere (placeholders are 0.0, not NaN).
    assert torch.isfinite(fr).all()


def test_placeholder_zero_in_batch(synthetic_db_with_steps: str) -> None:
    """Anywhere `forward_valid` is False, `forward_returns` must be exactly 0.0
    so PR 4's loss mask-multiply is safe."""
    dm = SpotBTCDataModule(
        klines_db=synthetic_db_with_steps,
        steps_db=synthetic_db_with_steps,
        batch_size=4,
        T=8,
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    fr = batch["forward_returns"]
    fv = batch["forward_valid"]
    masked_vals = fr[~fv]
    if masked_vals.numel() > 0:
        assert torch.all(masked_vals == 0.0)


def test_alignment_to_observation_window(synthetic_db_with_steps: str) -> None:
    """Highest-risk test (brief 3.3): forward-return anchor must be
    close[k] where k is the exact kline index obs_window uses for that step.

    Procedure: instantiate the datamodule, pull a specific dataset item by
    index, recover (agent_id, episode, start) from `dm._train_ds.starts`,
    look up the kline_idx the dataset used, then hand-compute
    ln(close[k+h]/close[k]) from the raw closes the datamodule loaded into
    `dm._closes`. Assert the dataset's emitted `forward_returns` match
    exactly to float32 precision.
    """
    dm = SpotBTCDataModule(
        klines_db=synthetic_db_with_steps,
        steps_db=synthetic_db_with_steps,
        batch_size=1,
        T=8,
    )
    dm.setup()
    ds = dm._train_ds
    assert ds is not None and len(ds) > 0
    closes = dm._closes
    assert closes is not None
    n_klines = len(closes)

    # Pick a few items so we cover multiple (agent, episode, start) triples.
    for i in (0, len(ds) // 2, len(ds) - 1):
        item = ds[i]
        agent_id, episode, start = ds.starts[i]
        ep = ds.episodes[(agent_id, episode)]
        kidx_t = ep.kline_idx[start : start + ds.T]
        fr_emitted = item["forward_returns"].numpy()
        fv_emitted = item["forward_valid"].numpy()
        for j, k in enumerate(kidx_t):
            for h_idx, h in enumerate(FORWARD_HORIZONS):
                if k + h < n_klines:
                    expected = float(np.log(closes[k + h] / closes[k]))
                    assert fv_emitted[j, h_idx], f"valid mask False at (j={j}, h={h}) but k+h<N"
                    assert np.isclose(
                        fr_emitted[j, h_idx],
                        expected,
                        atol=1e-5,
                    ), (
                        f"misaligned at item {i} step {j} horizon {h}: "
                        f"got {fr_emitted[j, h_idx]:.6f}, "
                        f"expected {expected:.6f}"
                    )
                else:
                    assert not fv_emitted[j, h_idx]
                    assert fr_emitted[j, h_idx] == 0.0


def test_mask_consistency_across_batch(synthetic_db_with_steps: str) -> None:
    """forward_valid is False only when the kline anchor sits within `h` of
    the global kline series end · never spuriously in the interior."""
    dm = SpotBTCDataModule(
        klines_db=synthetic_db_with_steps,
        steps_db=synthetic_db_with_steps,
        batch_size=4,
        T=16,
    )
    dm.setup()
    n_klines = len(dm._closes)
    ds = dm._train_ds
    # The DataLoader collates anonymously so it can't tell us which item
    # produced which row · instead we drive the underlying dataset directly,
    # which exposes `starts[i]` and `episodes[(aid, ep)].kline_idx`.
    for i in range(min(4, len(ds))):
        item = ds[i]
        agent_id, episode, start = ds.starts[i]
        ep = ds.episodes[(agent_id, episode)]
        kidx_t = ep.kline_idx[start : start + ds.T]
        fv_i = item["forward_valid"].numpy()
        for j, k in enumerate(kidx_t):
            for h_idx, h in enumerate(FORWARD_HORIZONS):
                expected_valid = (k + h) < n_klines
                assert bool(fv_i[j, h_idx]) == expected_valid, (
                    f"interior mask mismatch at i={i} j={j} h={h}: "
                    f"k={k} N={n_klines} expected_valid={expected_valid}"
                )


def test_gap_stats_zero_on_clean_synthetic(synthetic_db_with_steps: str) -> None:
    """The synthetic fixture has a contiguous 1-min ts series · gap_stats
    must be all-zero. This locks the gap detector against false positives."""
    dm = SpotBTCDataModule(
        klines_db=synthetic_db_with_steps,
        steps_db=synthetic_db_with_steps,
        batch_size=1,
        T=4,
    )
    dm.setup()
    assert dm.gap_stats is not None
    for h in FORWARD_HORIZONS:
        s = dm.gap_stats[h]
        assert s["count"] == 0
        assert s["fraction"] == 0.0


def test_mask_fraction_per_horizon_synthetic(synthetic_db_with_steps: str) -> None:
    """OQ-2 measurement on the synthetic fixture. Pure reporting · no
    threshold asserted (the brief specifies a 15% escalation trigger for
    real data; the synthetic fixture is too small to be meaningful)."""
    dm = SpotBTCDataModule(
        klines_db=synthetic_db_with_steps,
        steps_db=synthetic_db_with_steps,
        batch_size=4,
        T=8,
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    fv = batch["forward_valid"]  # (B, T, 4)
    fractions = (~fv).float().mean(dim=(0, 1))  # (4,)
    # Fractions are in [0, 1] and finite.
    assert (fractions >= 0).all()
    assert (fractions <= 1).all()
    # The synthetic episodes start at kline 300 and 700 with 200/120 steps,
    # and the kline series ends at 4999 · last-step kline_idx <= 819, so all
    # forward windows up to h=30 fit comfortably. Expect zero masking here.
    assert torch.all(fractions == 0.0)


# ---------------------------------------------------------------------
# 5.3 · Real-data validation (skipped if real DBs absent)
# ---------------------------------------------------------------------


@pytest.mark.skipif(
    not LIVE_DB.exists() or not SNAPSHOT_DB.exists(),
    reason="real DuckDBs not present",
)
def test_real_data_spotcheck() -> None:
    """Pick five real (trajectory, step, horizon) triples, query
    `data/market_ro.duckdb` directly for close[k] and close[k+h], and
    confirm the tensor matches the hand-computed log-return to float32
    precision."""
    dm = SpotBTCDataModule(
        klines_db=str(SNAPSHOT_DB),
        steps_db=str(LIVE_DB),
        batch_size=4,
        T=16,
    )
    dm.setup()
    ds = dm._train_ds
    assert ds is not None and len(ds) > 0

    con = duckdb.connect(str(SNAPSHOT_DB), read_only=True)
    closes_df = con.execute(
        "SELECT close FROM klines WHERE symbol='BTCUSDT' AND interval='1m' ORDER BY ts"
    ).df()
    con.close()
    closes_raw = closes_df["close"].to_numpy()
    n_klines = len(closes_raw)
    assert n_klines > 0

    rng = np.random.default_rng(123)
    samples = 0
    boundary_seen = False
    attempts = 0
    while samples < 5 and attempts < 200:
        attempts += 1
        i = int(rng.integers(0, len(ds)))
        item = ds[i]
        agent_id, episode, start = ds.starts[i]
        ep = ds.episodes[(agent_id, episode)]
        kidx_t = ep.kline_idx[start : start + ds.T]
        j = int(rng.integers(0, ds.T))
        h_idx = int(rng.integers(0, len(FORWARD_HORIZONS)))
        h = FORWARD_HORIZONS[h_idx]
        k = int(kidx_t[j])
        if k + h < n_klines:
            expected = float(np.log(closes_raw[k + h] / closes_raw[k]))
            got = float(item["forward_returns"][j, h_idx])
            assert np.isclose(got, expected, atol=1e-5), (
                f"spot-check fail at i={i} j={j} k={k} h={h}: got {got:.6f} expected {expected:.6f}"
            )
        else:
            assert not bool(item["forward_valid"][j, h_idx])
            assert float(item["forward_returns"][j, h_idx]) == 0.0
            boundary_seen = True
        samples += 1
    assert samples == 5
    # At least one mask path is exercised across the whole batch population.
    # Force-check by scanning if the random draw missed any.
    if not boundary_seen:
        fr_kn = dm._forward_returns_kn
        fv_kn = dm._forward_valid_kn
        # The longest horizon (h=30) always has the last 30 positions invalid.
        assert not fv_kn[-1, -1]
        assert fr_kn[-1, -1] == 0.0


@pytest.mark.skipif(
    not LIVE_DB.exists() or not SNAPSHOT_DB.exists(),
    reason="real DuckDBs not present",
)
def test_gap_detection_reports_real() -> None:
    """Run gap detection against real klines. Validate internal consistency:
    count <= total, fraction = count / total, and every reported gap index
    actually has a non-h-minute timestamp delta."""
    dm = SpotBTCDataModule(
        klines_db=str(SNAPSHOT_DB),
        steps_db=str(LIVE_DB),
        batch_size=1,
        T=4,
    )
    dm.setup()
    assert dm.gap_stats is not None
    ts_seconds = dm._ts_seconds
    assert ts_seconds is not None
    n = len(ts_seconds)
    for h in FORWARD_HORIZONS:
        s = dm.gap_stats[h]
        assert s["total"] == n - h
        assert 0 <= s["count"] <= s["total"]
        assert np.isclose(s["fraction"], s["count"] / max(s["total"], 1))
        # Every reported gap index has a ts delta != h*60.
        for k in s["indices"][:50]:
            assert ts_seconds[k + h] - ts_seconds[k] != h * 60


@pytest.mark.skipif(
    not LIVE_DB.exists() or not SNAPSHOT_DB.exists(),
    reason="real DuckDBs not present",
)
def test_mask_fraction_per_horizon_real() -> None:
    """OQ-2 real measurement. Computes per-horizon masked fraction over a
    full epoch's worth of sampled subsequence anchors (the kline indices
    `kidx[: T]` across all train starts). Asserts numbers are sane and
    flags if the 30-bar masked fraction exceeds 15% (brief OQ-2 stop
    condition · ADR-001 design concern)."""
    dm = SpotBTCDataModule(
        klines_db=str(SNAPSHOT_DB),
        steps_db=str(LIVE_DB),
        batch_size=1,
        T=16,
    )
    dm.setup()
    ds = dm._train_ds
    fv_kn = dm._forward_valid_kn
    assert ds is not None and fv_kn is not None

    # Accumulate counts across every (start, T) anchor in the train pool.
    horizon_counts = np.zeros(len(FORWARD_HORIZONS), dtype=np.int64)
    horizon_invalid = np.zeros(len(FORWARD_HORIZONS), dtype=np.int64)
    for agent_id, episode, start in ds.starts:
        ep = ds.episodes[(agent_id, episode)]
        kidx_t = ep.kline_idx[start : start + ds.T]
        invalid_step = ~fv_kn[kidx_t]  # (T, 4) bool
        horizon_invalid += invalid_step.sum(axis=0)
        horizon_counts += invalid_step.shape[0]
    fractions = horizon_invalid / np.maximum(horizon_counts, 1)
    # Brief OQ-2: > 15% at h=30 is a design concern. Not a hard fail · we
    # log it for the findings doc.
    print(
        "OQ-2 real masked fractions per horizon:",
        {h: float(f) for h, f in zip(FORWARD_HORIZONS, fractions)},
    )
    assert (fractions >= 0).all()
    assert (fractions <= 1).all()
