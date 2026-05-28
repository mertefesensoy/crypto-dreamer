# 2026-05-28 · Forward-return target data quality

This finding is the quantitative payload of PR 3. It tabulates the masked-fraction-per-horizon and the mid-series gap prevalence over the production kline snapshot, and notes what those numbers mean for ADR-001 (equal horizon weighting) and the deferred gap-masking backlog item.

## Sources

- Klines: `data/market_ro.duckdb`, 1,051,201 rows of 1-minute BTCUSDT bars spanning 2024-05-03 to 2026-04-30.
- Step log: `data/market.duckdb`, 504 episodes of `random:%` agents, total trajectory anchors: 608,160 train starts at the PR-3-default T=16 (= 9,730,560 (start, step) anchors).
- Measurement code: `tests/test_datamodule_forward_returns.py::test_mask_fraction_per_horizon_real` and the one-off probe documented in the implementation doc Verification step 6.

## OQ-2 · Masked fraction per horizon

`forward_valid` is False iff the anchor's kline index `k` satisfies `k + h >= N_klines` (series-end invalidity, brief 3.1). The fraction is computed over every (start, step) tuple in the train dataset · the same anchors the trainer will iterate.

| Horizon | Invalid anchors | Total anchors | Masked fraction |
| ------- | --------------: | ------------: | --------------: |
| 1       | 0               | 9,730,560     | 0.00%           |
| 5       | 0               | 9,730,560     | 0.00%           |
| 15      | 0               | 9,730,560     | 0.00%           |
| 30      | 0               | 9,730,560     | 0.00%           |

Interpretation. Brief OQ-2 set 15% at h=30 as the threshold above which the model would see materially less 30-bar signal than 1-bar signal, which would interact with ADR-001's equal-weighting choice. The observed fraction is 0% at every horizon · no escalation needed. ADR-001 is not at risk from boundary masking in the current snapshot.

The 0% result reflects an interaction between two facts: the env's `max_start = len(df) - episode_steps - 1` constraint keeps the highest reachable `_t` bounded away from the global series end by `episode_steps + 1 = 1441` bars, and the random-agent runs in this snapshot did not happen to land within 30 bars of even that bounded ceiling. If future runs of `random:%` or other agents push trajectories closer to the kline tail, this fraction will rise; the test will keep tracking it.

## OQ-3 · Mid-series gap prevalence

Gap detection: for each kline position `k` and each horizon `h`, flag as gap-affected iff `ts[k + h] - ts[k] != h * 60` seconds. Counts are over the full kline series, not over trajectory anchors.

| Horizon | Gap count | Total (k in [0, N-h)) | Gap fraction |
| ------- | --------: | --------------------: | -----------: |
| 1       | 0         | 1,051,200             | 0.0000%      |
| 5       | 0         | 1,051,196             | 0.0000%      |
| 15      | 0         | 1,051,186             | 0.0000%      |
| 30      | 0         | 1,051,171             | 0.0000%      |

Interpretation. The Phase 5.0b "gap-free 1-min" claim (`docs/implementations/2026-05-04-phase5-1-datamodule.md`) holds for the full 2-year snapshot at the precision of this detector. Brief OQ-3 set 5% at any horizon as the threshold for "raise the priority of the deferred masking work"; the observed value is 0% · no escalation needed. The detector itself is validated by `test_gap_detection_synthetic` (which inserts a known 1-minute gap and confirms the detector flags every horizon-window that spans it) and by `test_gap_detection_reports_real` (which asserts internal consistency: count <= total, fraction = count / total, every reported index has a non-h-minute delta).

If a future ingestion surfaces fresh klines with exchange-downtime gaps, the same code path will report them. The deferred masking policy decision can be made then with concrete numbers, not before.

## Implications

1. ADR-001 (equal horizon weighting) is not at risk from boundary masking on this snapshot. Per-horizon loss curves are still the right post-30k-diagnostic check (per the ADR), but they will not be confounded by horizon-asymmetric mask densities.

2. The deferred gap-masking backlog item (`docs/planning/BACKLOG.md` Operational) is currently small-effort precautionary work, not blocking. If a Phase 6 live-trading pipeline ingests bars with real exchange-downtime gaps, revisit · the gap-detector vectorization in `_compute_forward_targets` is already in place to surface those statistics at setup() time.

3. The `forward_valid` mask in the batch is still operationally required even though it never fires on this snapshot · PR 4 multiplies it into the loss unconditionally, so the contract stands regardless of current prevalence.

## Reproducibility

Re-run the probe (requires the production DuckDBs to be present):

```
.venv\Scripts\python.exe -c "from training.datamodule import SpotBTCDataModule, FORWARD_HORIZONS; \
dm = SpotBTCDataModule(klines_db='data/market_ro.duckdb', steps_db='data/market.duckdb', batch_size=1, T=16); \
dm.setup(); \
print({h: dm.gap_stats[h] for h in FORWARD_HORIZONS})"
```

The exact probe script that produced the tables above is reproduced inline in the implementation doc Verification step 6.
