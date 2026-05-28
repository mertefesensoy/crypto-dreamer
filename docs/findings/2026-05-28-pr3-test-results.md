# 2026-05-28 · PR 3 test results and loop iteration history

This document is the test-side companion to the PR 3 implementation doc. It records every test in the brief Section 5 taxonomy with its outcome, plus the full iteration history for any open question or test failure that triggered the brief Section 0 self-correcting loop.

## Per-test outcomes

All tests in `tests/test_datamodule_forward_returns.py`, on the final code state. Verified by running `pytest tests/test_datamodule_forward_returns.py -v`.

### 5.1 · Unit tests (pure computation)

| Test | Outcome |
| ---- | ------- |
| `test_forward_return_value_correctness` | passed · ln(close[k+h]/close[k]) matches hand-computed values for `(k, h)` in {(0,1), (2,1), (0,5), (4,5)} and the boundary `(9, 1)` is invalid with 0.0 placeholder |
| `test_horizon_ordering_matches_head` | passed · `FORWARD_HORIZONS == (1, 5, 15, 30)` and a freshly constructed `ForwardDistributionHead` with `horizons=list(FORWARD_HORIZONS)` exposes them in the same order |
| `test_series_end_mask_per_horizon` | passed · last `h` positions per horizon are False, everything before is True; spot-checks at `n-1` and `n-30` confirm per-horizon granularity |
| `test_placeholder_value_at_invalid` | passed · `forward_returns[~forward_valid] == 0.0` element-wise |
| `test_dtypes_and_shapes` | passed · `(N, 4) float32` and `(N, 4) bool` exactly |
| `test_gap_detection_synthetic` | passed · with one minute manually removed between bar 49 and 50, the detector flags `k=49` at h=1, `k in [45, 50)` at h=5, `k in [35, 50)` at h=15, `k in [20, 50)` at h=30 |
| `test_helper_rejects_bad_inputs` | passed · raises ValueError on length mismatch and on non-positive horizon |

### 5.2 · Integration tests (synthetic DB through the datamodule)

| Test | Outcome |
| ---- | ------- |
| `test_batch_contains_new_keys` | passed · `forward_returns (B, T, 4) float32` and `forward_valid (B, T, 4) bool` are present, all 6 pre-existing keys unchanged in shape and dtype |
| `test_placeholder_zero_in_batch` | passed · across the batch, every position where `forward_valid` is False holds exactly 0.0 in `forward_returns` |
| `test_alignment_to_observation_window` | passed · for items 0, mid, and last in `dm._train_ds`, the emitted `forward_returns[j, h_idx]` matches `ln(closes[kidx[j] + h] / closes[kidx[j]])` to atol=1e-5 where `kidx` is recovered from `ds.starts[i]` and `ds.episodes`. This test exercises the brief 3.3 alignment guarantee directly. |
| `test_mask_consistency_across_batch` | passed · for every (item, step, horizon), `forward_valid` agrees with `(k + h) < N_klines` · no interior False values appeared |
| `test_gap_stats_zero_on_clean_synthetic` | passed · the synthetic fixture has gap-free 1-min timestamps, so every horizon reports count=0, fraction=0.0 |
| `test_mask_fraction_per_horizon_synthetic` | passed · synthetic fixture has small episodes far from the synthetic kline tail, so masked fraction is exactly 0.0 at every horizon · pure reporting test, no threshold |

### 5.3 · Real-data tests (gated on `data/market.duckdb` + `data/market_ro.duckdb`)

| Test | Outcome |
| ---- | ------- |
| `test_real_data_spotcheck` | passed · five random `(traj, step, horizon)` triples sampled with a fixed seed, each hand-computed from raw closes queried directly out of the DuckDB and asserted to within atol=1e-5. The mask path is exercised by an additional invariant check on `fv_kn[-1, -1]` at the global kline tail. |
| `test_gap_detection_reports_real` | passed · internal-consistency checks hold: count <= total, fraction == count/total, every reported gap index has a non-h-minute delta. The detector reported zero gaps across all horizons (see findings doc). |
| `test_mask_fraction_per_horizon_real` | passed · over all 608,160 train start tuples at T=16 (= 9.73M anchors), masked fraction is 0% at every horizon. Brief OQ-2 escalation threshold (> 15% at h=30) is not triggered. |

### Aggregate

- `pytest tests/test_datamodule_forward_returns.py -v` · **16 passed, 78 s**.
- `pytest tests/ -x -q --ignore=tests/test_dream_endpoint.py` · **58 passed, 9 warnings, 106 s**. Up from 42 in the prior session (+16 from this PR).
- `ruff check training/datamodule.py tests/test_datamodule_forward_returns.py` · **All checks passed**.
- `ruff format --check training/datamodule.py tests/test_datamodule_forward_returns.py` · **2 files already formatted**.

## Loop iteration history

Per brief Section 0, the loop kicks in when a test fails or an open question lacks a confident answer. Every iteration is documented here even when only one hypothesis was needed, because the user reviews this artifact to audit the process.

### OQ-1 · Is the underlying kline data contiguous or segmented per episode?

**Iteration 1.** Hypothesis: from reading `training/datamodule.py:283-288` (single contiguous SELECT from a klines table ordered by ts) and `envs/spot_btc.py:93-101` (env constructs trajectories by random starts into the same contiguous kline series), the klines are a single global contiguous series; episodes are random index slices into that series. Test: confirmed by direct code reading, no synthetic test needed for this structural claim · the helper's gap-free assertion in the existing phase5-1 datamodule doc (`docs/implementations/2026-05-04-phase5-1-datamodule.md` line 41) was itself derived from this property. Outcome: resolved. Series-end masking therefore applies at the global kline tail (`k + h >= N_klines`), not per-episode tail. This is consistent with brief 3.1's "end of available kline data" wording.

### OQ-2 · Masked-fraction-per-horizon prevalence

**Iteration 1.** Hypothesis: in the production snapshot, the env's `max_start = len(df) - episode_steps - 1` constraint keeps `_t` bounded away from the kline tail by 1441 bars, so the 30-bar masked fraction will be small but probably nonzero (some terminated episodes might fall close enough). Test: `test_mask_fraction_per_horizon_real` accumulates over 9,730,560 anchors and reports per-horizon fractions. Outcome: **0.000000% at all four horizons**. The hypothesis underestimated how far from the tail the random-agent runs in this snapshot stayed. Resolved at iteration 1 because the brief's escalation threshold (> 15% at h=30) is decisively not triggered.

### OQ-3 · Mid-series gap prevalence

**Iteration 1.** Hypothesis: the phase5-1 doc claimed gap-free 1-min, but the brief explicitly wanted the claim re-verified at the detector level. Test: `test_gap_detection_reports_real` runs the gap detector across the full kline series and exposes `gap_stats` via the module. Outcome: zero gaps at every horizon. Resolved at iteration 1.

### Implementation bug: `ts_seconds` unit handling

This was an in-implementation failure surfaced by `test_gap_stats_zero_on_clean_synthetic`, not a brief open question · but it's recorded here because it is exactly the kind of silent semantic error the brief's loop is designed to surface.

**Iteration 1.** Hypothesis: `kline_ts.asi8 // 1_000_000_000` converts a tz-aware `DatetimeIndex` to int64 seconds-since-epoch. Test: `test_gap_stats_zero_on_clean_synthetic` on the synthetic fixture, which has a contiguous 1-min series. Failure: `gap_stats[1]["count"]` was 4999/4999 · every consecutive pair flagged as a gap, the opposite of correct. Investigation: dropped into a one-line REPL probe; `kline_ts` after a DuckDB TIMESTAMP roundtrip has microsecond precision (`datetime64[us, UTC]`), and `asi8` on a microsecond-precision DatetimeIndex returns the underlying ints in microseconds, not nanoseconds. The `// 10**9` then truncated every value to 0, so every `ts_delta` was 0 != 60. Re-hypothesis: convert via Timedelta arithmetic, which is precision-agnostic. Test: re-run `pytest tests/test_datamodule_forward_returns.py -v`. Outcome: pass at iteration 2. The implementation doc Design Decisions section 4 records the rationale for the chosen form so a future reader does not regress to the asi8 shortcut.

### Code review failure: unused variable

After the first full ruff run, `ruff check` reported F841 `Local variable fv is assigned to but never used` in `test_mask_consistency_across_batch`. The unused `fv = batch["forward_valid"]` was a leftover from an earlier sketch where I considered iterating the batch directly before realizing the test needed `kidx` recovered from the dataset object instead. Fix: removed the dead assignment and replaced the surrounding comment to explain why the test drives the dataset directly. Iteration count: 1. Resolved in one shot.

### Formatting

After the lint and content were clean, `ruff format --check` reported both files as not-yet-formatted. Per brief Section 8 ("CI runs `ruff format --check`, so format must be clean"), applied `ruff format` to both files. The format pass also touched some pre-existing column-aligned comments in `training/datamodule.py` · these are inside the "files you may modify" scope per the brief's Section 0 list, and not touching them would leave the file in a non-CI-clean state. Iteration count: 1. Resolved in one shot.

## Final state

- All 16 new tests passing.
- Full battery (58 tests, with 16 new) passing.
- ruff check + ruff format --check clean.
- No open questions blocked, no escalations.
- No iterations hit the 5-cap.
