"""Compute the Gate 2 marginal baseline for the Phase 5.4 diagnostic.

Reproduces the exact train/val split from training/datamodule.py
(last 15% of each calendar month by day-of-month -> val) and computes
the cross-entropy baseline for the constant-predictor that always
outputs the training-set marginal forward-return histogram.

See docs/design/ARCHITECTURE.md Section 11 (Gate 2) for the formal
definition of the baseline.

Dependencies: numpy, duckdb. No pandas, no torch.

Usage:
    python scripts/compute_marginal_baseline.py
"""
from __future__ import annotations

import calendar
from pathlib import Path

import duckdb
import numpy as np

HORIZONS = [1, 5, 15, 30]
N_BINS = 41
RANGES = {1: 0.005, 5: 0.010, 15: 0.018, 30: 0.025}
VAL_MONTH_FRAC = 0.85
EPS = 1e-9
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "market.duckdb"


def two_hot_histogram(values: np.ndarray, half_range: float) -> np.ndarray:
    """Two-hot encode all values and return the summed (unnormalized) histogram."""
    bin_width = 2 * half_range / (N_BINS - 1)
    v = np.clip(values, -half_range, half_range)
    pos = (v + half_range) / bin_width
    idx_lo = np.floor(pos).astype(np.int64).clip(0, N_BINS - 2)
    w_hi = pos - idx_lo.astype(np.float64)
    w_lo = 1.0 - w_hi

    hist = np.zeros(N_BINS, dtype=np.float64)
    np.add.at(hist, idx_lo, w_lo)
    np.add.at(hist, idx_lo + 1, w_hi)
    return hist


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    data = con.execute(
        "SELECT ts, close FROM klines "
        "WHERE symbol='BTCUSDT' AND interval='1m' "
        "ORDER BY ts"
    ).fetchnumpy()
    con.close()

    close = data["close"].astype(np.float64)
    ts = data["ts"]
    N = len(close)

    # ---- Train/val partition (matches training/datamodule.py:252) ----
    # Rule: (day_of_month - 1) / days_in_month >= 0.85 -> validation
    dates = ts.astype("datetime64[D]")
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    months = dates.astype("datetime64[M]").astype(int) % 12 + 1
    days = (dates - dates.astype("datetime64[M]")).astype(int) + 1

    ym_keys = years * 100 + months
    unique_ym = np.unique(ym_keys)
    dim_lookup = {
        int(ym): calendar.monthrange(*divmod(int(ym), 100))[1]
        for ym in unique_ym
    }
    dims = np.array([dim_lookup[int(k)] for k in ym_keys], dtype=np.float64)

    is_val = ((days - 1) / dims) >= VAL_MONTH_FRAC
    is_train = ~is_val

    n_train_total = int(is_train.sum())
    n_val_total = int(is_val.sum())

    print(f"Total klines: {N:,}")
    print(f"Train: {n_train_total:,}  Val: {n_val_total:,}")
    print()

    total_baseline = 0.0
    total_train_entropy = 0.0
    total_val_entropy = 0.0
    results = []

    for h in HORIZONS:
        R_h = RANGES[h]

        fwd_ret = np.log(close[h:] / close[:-h])
        valid = np.isfinite(fwd_ret)
        train_mask = is_train[: N - h] & valid
        val_mask = is_val[: N - h] & valid

        train_ret = fwd_ret[train_mask]
        val_ret = fwd_ret[val_mask]

        p_hist = two_hot_histogram(train_ret, R_h)
        p = p_hist / p_hist.sum()

        q_hist = two_hot_histogram(val_ret, R_h)
        q = q_hist / q_hist.sum()

        cross_ent = float(-np.sum(q * np.log(p + EPS)))
        train_ent = float(-np.sum(p * np.log(p + EPS)))
        val_ent = float(-np.sum(q * np.log(q + EPS)))

        gibbs_gap = cross_ent - val_ent
        if gibbs_gap < -1e-6:
            raise ValueError(
                f"Gibbs inequality violated at horizon {h}: "
                f"H(q,p)={cross_ent:.6f} < H(q)={val_ent:.6f}, "
                f"gap={gibbs_gap:.6e}"
            )

        total_baseline += cross_ent
        total_train_entropy += train_ent
        total_val_entropy += val_ent

        results.append({
            "horizon": h,
            "range": R_h,
            "cross_ent": cross_ent,
            "train_ent": train_ent,
            "val_ent": val_ent,
            "drift": gibbs_gap,
            "train_n": int(train_mask.sum()),
            "val_n": int(val_mask.sum()),
        })

        print(
            f"Horizon {h:>2} bar  |  R=+/-{R_h:.3f}  |  "
            f"H(q,p)={cross_ent:.4f}  H(q)={val_ent:.4f}  H(p)={train_ent:.4f}  |  "
            f"drift={gibbs_gap:.4f}  |  "
            f"train={int(train_mask.sum()):>9,}  val={int(val_mask.sum()):>8,}"
        )

    print()
    print(f"Total marginal baseline H(q,p) (sum): {total_baseline:.4f}")
    print(f"Total val entropy H(q) (sum):         {total_val_entropy:.4f}")
    print(f"Total train entropy H(p) (sum):       {total_train_entropy:.4f}")

    # ---- Write markdown report ----
    report_dir = Path(__file__).resolve().parent.parent / "docs" / "findings"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "2026-05-27-marginal-baseline.md"

    lines = [
        "# 2026-05-27 · Gate 2 Marginal Baseline",
        "",
        "## Reproduction",
        "",
        "```",
        "python scripts/compute_marginal_baseline.py",
        "```",
        "",
        f"Data: `data/market.duckdb`, BTCUSDT 1m, {N:,} rows.",
        (
            "Split: last 15% of each calendar month by day-of-month to "
            "validation (matching `training/datamodule.py:252`, "
            "`val_month_frac=0.85`)."
        ),
        f"Train: {n_train_total:,} kline rows · Val: {n_val_total:,} kline rows.",
        "",
        "## Per-Horizon Results",
        "",
        "| Horizon | Range | H(p) train entropy | H(q) val entropy | H(q,p) cross-entropy | Drift H(q,p)-H(q) | Train N | Val N |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r['horizon']} bar | +/-{r['range']:.3f} | "
            f"{r['train_ent']:.4f} | {r['val_ent']:.4f} | "
            f"{r['cross_ent']:.4f} | {r['drift']:.4f} | "
            f"{r['train_n']:,} | {r['val_n']:,} |"
        )
    lines += [
        "",
        f"**Total marginal baseline H(q,p): {total_baseline:.4f}**",
        "",
        f"Total val entropy H(q): {total_val_entropy:.4f}",
        "",
        f"Total train entropy H(p): {total_train_entropy:.4f}",
        "",
        "## Interpretation",
        "",
    ]

    drift_summary = ", ".join(
        f"{r['horizon']}-bar {r['drift']:.4f}" for r in results
    )
    max_drift = max(r["drift"] for r in results)

    interp = (
        f"Gate 2 of the Phase 5.4 diagnostic requires the model's "
        f"val/loss_forward_dist at step 20k to be strictly less than "
        f"**{total_baseline:.4f}**. "
        f"The val entropy H(q) = {total_val_entropy:.4f} is the irreducible "
        f"lower bound: no predictor can achieve a per-horizon loss below H(q) "
        f"on the validation set. "
        f"The drift H(q,p) - H(q) per horizon measures distribution shift "
        f"between train and val partitions: {drift_summary}."
    )

    if max_drift < 0.01:
        interp += (
            " All gaps are below 0.01, confirming the forward-return "
            "distribution is effectively stationary across the month-based "
            "train/val split. The baseline is a clean target."
        )
    else:
        interp += (
            f" The maximum per-horizon drift is {max_drift:.4f}, which is "
            "non-negligible. A model could partially beat the baseline by "
            "exploiting the train-val distribution mismatch rather than "
            "learning genuine conditional structure. The diagnostic "
            "interpretation should account for this: a model that beats "
            "the baseline by less than the drift margin has weak evidence "
            "of conditional learning."
        )

    lines.append(interp)
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
