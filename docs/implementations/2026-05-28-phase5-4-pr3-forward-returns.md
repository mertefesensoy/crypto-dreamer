# 2026-05-28 · Phase 5.4 PR 3 · Datamodule forward-return targets

## Problem / Motivation

PR 2 of the Phase 5.4 pivot shipped `ForwardDistributionHead` (`models/heads.py:112`), which predicts categorical distributions over discretized log-returns at four horizons {1, 5, 15, 30} bars. The head exists and is unit-tested but unused · nothing yet feeds it real targets. PR 3 is the data-side counterpart: surface, for every trajectory step, the actual forward log-returns the head will be trained against, in a shape and ordering that PR 4 can multiply straight into the loss.

The single highest-risk part of this work is alignment (brief Section 3.3). A wrong off-by-one anchors the targets on a bar that is *not* the bar the obs window, reward, and continue signals are aligned with for that step. Every downstream test could pass while the targets are silently wrong. The implementation doc states the resolved indexing explicitly so a future reader can audit it.

## What Changed

| File | Description |
| --- | --- |
| `training/datamodule.py` | New module constant `FORWARD_HORIZONS = (1, 5, 15, 30)` matching `ForwardDistributionHead.horizons`. New helper `_compute_forward_targets(closes, ts_seconds, horizons)` that returns `(forward_returns_kn (N, 4) float32, forward_valid_kn (N, 4) bool, gap_stats dict)` in a single pass over the full kline series. `SpotBTCDataModule.setup` computes these arrays once and exposes `self.gap_stats`, `self._closes`, `self._ts_seconds`, `self._forward_returns_kn`, `self._forward_valid_kn`. `TrajectoryDataset` accepts the precomputed arrays and emits two new batch keys per item: `forward_returns: (T, 4) float32` and `forward_valid: (T, 4) bool`. All pre-existing keys (`obs_window`, `next_obs_window`, `action`, `reward`, `continue_flag`, `is_first`) are unchanged in shape, dtype, and value. |
| `tests/test_datamodule_forward_returns.py` (new) | Sixteen tests across three layers: 7 unit tests against `_compute_forward_targets` with synthetic numpy inputs, 6 integration tests against `SpotBTCDataModule` on the `synthetic_db_with_steps` fixture, and 3 real-data tests gated on `data/market_ro.duckdb` + `data/market.duckdb` (skipped on machines without the snapshots). Includes the highest-risk alignment test `test_alignment_to_observation_window` that recovers the kline anchor independently and asserts the emitted targets match a hand-computation. |
| `docs/findings/2026-05-28-forward-returns-data-quality.md` (new) | OQ-2 masked-fraction-per-horizon table and OQ-3 gap-prevalence statistics over the full 2-year snapshot. |
| `docs/findings/2026-05-28-pr3-test-results.md` (new) | Per-test outcomes and the iteration history of the one self-correcting-loop trigger (`asi8` precision bug). |
| `docs/planning/BACKLOG.md` | New Operational item for deferred gap masking, sized small per the OQ-3 statistics. |

No changes to `models/`, `configs/`, `envs/`, `serve/`, or `dashboard/`. The brief constraints were followed: no model code, no config, no env edits.

## Implementation Approach

The computation is split between a pure helper that owns the numerics and a thin integration layer that wires the result into the existing dataset.

`_compute_forward_targets` takes a flat `(N,)` close-price array and a flat `(N,)` int64 timestamps-in-seconds array and produces three things in one pass: the `(N, 4)` forward-return matrix, the `(N, 4)` series-end validity mask, and the per-horizon gap statistics dict. The pure-helper form makes the math testable in isolation · the unit tests construct synthetic close arrays where the expected log-returns are known by hand.

For each horizon `h`, the helper computes `ln(close[k+h]) - ln(close[k])` for `k in [0, N - h)` via vectorized slicing on the log-close array. Series-end positions (`k + h >= N`) get a 0.0 placeholder in the returns and `False` in the validity mask. Gap detection runs on the same slice: `ts_seconds[k+h] - ts_seconds[k]` must equal `h * 60`; anything else is flagged in `gap_stats[h]["indices"]`. Gaps are detected but never masked · per brief 3.2, gap masking is a deferred policy decision and the `forward_valid_kn` mask in PR 3 reflects ONLY series-end invalidity.

In `SpotBTCDataModule.setup`, the helper is called once on the full kline series (about 1.05M rows for the production snapshot) after the existing `compute_feature_block` call. The two `(N, 4)` arrays are then passed into both `TrajectoryDataset` constructors (train and val). The per-step lookup in `TrajectoryDataset.__getitem__` is a single fancy-index `forward_returns_kn[kidx_t]` where `kidx_t = kidx[: T]` is the same kline-anchor slice already used for obs windows · so the targets reuse the alignment by construction, not by re-derivation.

The conversion from `kline_ts` (a pandas DatetimeIndex) to int64 seconds-since-epoch is precision-agnostic: `((kline_ts - epoch_UTC) // pd.Timedelta("1s")).to_numpy().astype(np.int64)`. The naive shortcut `kline_ts.asi8 // 10**9` returns the wrong unit when the DatetimeIndex carries microsecond precision (which it does after a DuckDB TIMESTAMP roundtrip), and the gap detector fires on every consecutive pair. See the test-results doc for the iteration that surfaced this.

## Mathematical / Statistical Details

### Forward log-return at horizon h

For each kline index `k` and horizon `h`:

```
forward_returns_kn[k, h_idx] = ln(close[k + h] / close[k])
```

This is the change in log-price over `h` 1-minute bars, anchored on the close of bar `k`. For `k + h >= N_klines`, the right-hand side is undefined; the placeholder `0.0` is stored with `forward_valid_kn[k, h_idx] = False`. Per brief 3.1, the placeholder is never used in the loss · PR 4 will multiply the per-(step, horizon) cross-entropy by `forward_valid` before averaging.

### Series-end validity mask

```
forward_valid_kn[k, h_idx] = (k + h < N_klines)
```

For horizon 30 and a series of N rows, positions `[N-30, N)` are invalid; for horizon 1 only position `N-1` is invalid. Mask state at intermediate `k` is True for all four horizons.

### Gap detector

For each horizon `h`, position `k` is flagged as gap-affected iff:

```
ts_seconds[k + h] - ts_seconds[k] != h * 60
```

That is, the elapsed wall-clock time across `h` bar steps does not equal `h` minutes. This catches missing-minute klines that survived ingestion (where bars `k` and `k + h` exist but at least one bar between them does not). The detector does NOT alter `forward_valid_kn`; it only accumulates counts and indices for the findings doc.

### Empirical findings on the production snapshot

Run output from the production snapshot (`N_klines = 1,051,201`):

| Horizon | Gap count | Total | Gap fraction |
| ------- | --------: | ----: | -----------: |
| 1       | 0         | 1,051,200 | 0.0000% |
| 5       | 0         | 1,051,196 | 0.0000% |
| 15      | 0         | 1,051,186 | 0.0000% |
| 30      | 0         | 1,051,171 | 0.0000% |

Zero gaps confirm the Phase 5.0b "gap-free 1-min" claim across the full 2-year window. The full findings doc records this with the masked-fraction-per-horizon table.

## Design Decisions

### 1 · Precompute once over the full series, not per-`__getitem__`

The forward returns and validity mask are computed once at `setup()` time over the full `N_klines` array and stored on the dataset. Total memory: about 16 MB for `(N, 4) float32` + 4 MB for `(N, 4) bool` on a 1.05M-row snapshot · trivial alongside the 50 MB feature cache. The per-step lookup is then a single fancy-index op (O(T) memory copy, no log/exp/arithmetic). Considered and rejected: computing per `__getitem__` from a slice of closes. Rejected because the gap-detector vectorization is cleaner over the full series, and the helper becomes naturally pure (no per-call dataset state).

### 2 · Anchor on `close[k]` where `k = ep.kline_idx[step]`

This is the brief 3.3 resolution. The reasoning, traceable through the env and datamodule code:

`SpotBTCEnv.step` (`envs/spot_btc.py:138-189`) writes `StepInfo.ts = self.df["ts"].iloc[self._t]` *after* the `self._t += 1` on line 161. The kline index recorded in `step_log` for that step therefore points at the bar immediately following the agent's action. The agent's observation at that step was built from `decision_window = self._features[self._t - WINDOW : self._t]` (line 142), evaluated *before* the `+=1` · so the window is 256 bars *ending at the bar just prior to* the stored `ts`.

In the datamodule, `kline_idx = (ts - kline_t0).total_seconds() / 60` (`training/datamodule.py:357`) reproduces that same `_t`-after-increment. The obs window at trajectory step `j` is `feature_cache[kidx[j] - seq_len : kidx[j]]`, which equals `feature_cache[kidx[j] - 256 : kidx[j]]` · bars `kidx[j] - 256` through `kidx[j] - 1`, NOT including bar `kidx[j]`.

So `close[kidx[j]]` is the close of the first bar *not yet in the obs window* · which is exactly the "current price" the agent stands on after its action. Predicting `ln(close[kidx[j] + h] / close[kidx[j]])` is predicting future price change starting from that current bar's close. This matches the brief's `forward_return[k, h] = ln(close[k + h] / close[k])` definition with `k = kidx[j]`. Reward and continue at step `j` reflect the transition that resulted in arriving at bar `kidx[j]`, so the anchor is consistent across all four signals.

`test_alignment_to_observation_window` locks this in: it recovers `kidx[j]` from the dataset's own `starts` and `episodes` structures, looks up the closes the datamodule loaded into `dm._closes`, hand-computes `ln(close[k+h] / close[k])`, and asserts the emitted `forward_returns` match to float32 precision. A bug that anchored on `close[k-1]` or `close[k+1]` would produce different values and the test would fire.

### 3 · Series-end mask only · gaps detected but not masked

Per brief 3.1, `forward_valid` flips to False only when `k + h >= N_klines`. Per brief 3.2, mid-series gaps (where klines `k` and `k + h` exist but are not actually `h` minutes apart) are detected, counted, and logged but NOT masked. The scope boundary is explicit both in the helper's docstring (`forward_valid_kn ... Reflects series-end invalidity ONLY`) and in the datamodule's module docstring (`Does NOT mask mid-series gaps`).

Rationale: masking gaps correctly requires a policy decision (skip the affected step, interpolate, re-anchor on the next contiguous bar) and that decision should be made with the gap statistics in hand. PR 3's job is to quantify the problem, not solve it · the deferred masking work is filed in `docs/planning/BACKLOG.md` Operational section.

The empirical finding for this snapshot is zero gaps (see findings doc), so the deferred work is currently small-effort precautionary; if a future ingestion of fresh klines surfaces gaps, the prevalence statistics tell us the priority.

### 4 · Precision-agnostic ts-to-seconds conversion

`((kline_ts - epoch_UTC) // pd.Timedelta("1s"))` instead of `kline_ts.asi8 // 10**9`. The asi8 form returns int64 in whatever resolution the DatetimeIndex carries · ns when constructed purely in pandas, but microseconds after a DuckDB TIMESTAMP roundtrip. Dividing microseconds by 10^9 truncates everything to 0 and the gap detector flags every consecutive pair. The Timedelta-floor-division form is invariant under precision changes. The test-results doc records this iteration explicitly.

### 5 · 0.0 placeholder rather than NaN

NaN would also satisfy "the value is never used because the mask hides it" but would propagate through any downstream arithmetic that forgets the mask (e.g., a mean) into NaN gradients. 0.0 is benign in additions and the test `test_placeholder_value_at_invalid` asserts the invariant explicitly for PR 4's mask-multiply.

## Verification

1. **Unit tests** · `pytest tests/test_datamodule_forward_returns.py::test_forward_return_value_correctness tests/test_datamodule_forward_returns.py::test_horizon_ordering_matches_head tests/test_datamodule_forward_returns.py::test_series_end_mask_per_horizon tests/test_datamodule_forward_returns.py::test_placeholder_value_at_invalid tests/test_datamodule_forward_returns.py::test_dtypes_and_shapes tests/test_datamodule_forward_returns.py::test_gap_detection_synthetic tests/test_datamodule_forward_returns.py::test_helper_rejects_bad_inputs -v` · 7 passed.

2. **Integration tests** · `pytest tests/test_datamodule_forward_returns.py::test_batch_contains_new_keys tests/test_datamodule_forward_returns.py::test_placeholder_zero_in_batch tests/test_datamodule_forward_returns.py::test_alignment_to_observation_window tests/test_datamodule_forward_returns.py::test_mask_consistency_across_batch tests/test_datamodule_forward_returns.py::test_gap_stats_zero_on_clean_synthetic tests/test_datamodule_forward_returns.py::test_mask_fraction_per_horizon_synthetic -v` · 6 passed.

3. **Real-data tests** · `pytest tests/test_datamodule_forward_returns.py::test_real_data_spotcheck tests/test_datamodule_forward_returns.py::test_gap_detection_reports_real tests/test_datamodule_forward_returns.py::test_mask_fraction_per_horizon_real -v` · 3 passed.

4. **Full battery** · `pytest tests/ -x -q --ignore=tests/test_dream_endpoint.py` · 58 passed (was 42, +16 from this PR).

5. **Lint and format** · `ruff check training/datamodule.py tests/test_datamodule_forward_returns.py` · all checks passed. `ruff format --check training/datamodule.py tests/test_datamodule_forward_returns.py` · already formatted.

6. **Real-data probe** · over the production snapshot (`data/market_ro.duckdb`, 1,051,201 klines, 608,160 train start tuples at T=16): zero gap-affected anchors at any horizon, zero series-end-masked anchors at any horizon. See findings doc for the table.

## Related Docs

- Architecture reference · `docs/design/ARCHITECTURE.md` Sections 3, 6.
- Phase 5.4 pivot changelog · `docs/implementations/2026-05-27-phase5-4-pivot-forward-distribution.md`.
- Brief that governed this PR · `docs/briefings/PR3-forward-returns-briefing.md`.
- Findings · `docs/findings/2026-05-28-forward-returns-data-quality.md`.
- Test results · `docs/findings/2026-05-28-pr3-test-results.md`.
- ADRs · `docs/design/ARCHITECTURE.md` Section 12, ADR-001 (equal horizon weighting), ADR-002 (forward-distribution architecture).
- Roadmap entry for the next PR · `docs/planning/ROADMAP.md` Section 1, PR 4.
