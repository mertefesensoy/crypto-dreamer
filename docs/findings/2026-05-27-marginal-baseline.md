# 2026-05-27 · Gate 2 Marginal Baseline

## Reproduction

```
python scripts/compute_marginal_baseline.py
```

Data: `data/market.duckdb`, BTCUSDT 1m, 1,051,201 rows.
Split: last 15% of each calendar month by day-of-month to validation (matching `training/datamodule.py:252`, `val_month_frac=0.85`).
Train: 912,961 kline rows · Val: 138,240 kline rows.

## Per-Horizon Results

| Horizon | Range | H(p) train entropy | H(q) val entropy | H(q,p) cross-entropy | Drift H(q,p)-H(q) | Train N | Val N |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 bar | +/-0.005 | 2.2290 | 2.1366 | 2.1402 | 0.0036 | 912,960 | 138,240 |
| 5 bar | +/-0.010 | 2.3488 | 2.2632 | 2.2665 | 0.0033 | 912,956 | 138,240 |
| 15 bar | +/-0.018 | 2.3081 | 2.2236 | 2.2271 | 0.0035 | 912,946 | 138,240 |
| 30 bar | +/-0.025 | 2.3178 | 2.2250 | 2.2294 | 0.0044 | 912,931 | 138,240 |

**Total marginal baseline H(q,p): 8.8632**

Total val entropy H(q): 8.8484

Total train entropy H(p): 9.2037

## Interpretation

Gate 2 of the Phase 5.4 diagnostic requires the model's val/loss_forward_dist at step 20k to be strictly less than **8.8632**. The val entropy H(q) = 8.8484 is the irreducible lower bound: no predictor can achieve a per-horizon loss below H(q) on the validation set. The drift H(q,p) - H(q) per horizon measures distribution shift between train and val partitions: 1-bar 0.0036, 5-bar 0.0033, 15-bar 0.0035, 30-bar 0.0044. All gaps are below 0.01, confirming the forward-return distribution is effectively stationary across the month-based train/val split. The baseline is a clean target.
