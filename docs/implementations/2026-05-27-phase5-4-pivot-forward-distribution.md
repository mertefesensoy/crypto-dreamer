# 2026-05-27 · Phase 5.4 · Forward-Return-Distribution Pivot

## Problem / Motivation

Phase 5.3 ran a 30k-step diagnostic of the DreamerV3-style world model with a feature-reconstruction decoder (W&B run `kk3mzb3k`). The result was hard posterior collapse: kl_unclipped finished at 25.7 nat, below the 32-nat free-bits floor, with val/loss_decoder at 0.0021. The deterministic path h_t reconstructed the 15-dim feature target via MSE without engaging the stochastic latent z_t. Three weeks of pause followed.

Re-diagnosis on 2026-05-27 identified the root cause: the feature-reconstruction target has insufficient stochastic structure for z_t. The 12 market features are deterministic functions of the 256-bar price/volume window that h_t already encodes through the GRU's sequential processing. The 3 portfolio features are deterministic functions of the action history. There is no residual conditional entropy for z_t to capture. The reconstruction loss converged to near-zero because the problem is effectively deterministic given h_t.

The pivot decision: replace the feature-reconstruction decoder entirely with a forward-return-distribution head that predicts categorical distributions over discretized log-returns at four horizons {1, 5, 15, 30} bars. This gives z_t a genuinely stochastic job · the shape of the forward return distribution depends on the current volatility regime, which is not fully determined by history. The empirical evidence supporting this design is that BTC 1-min return standard deviations scale as sqrt(t) (verified: 1-bar std 0.00067, 5-bar 0.00148, 15-bar 0.00254, 30-bar 0.00357 over 1.05M rows), confirming near-zero conditional mean structure but leaving regime-dependent variance as the exploitable signal.

## What Changed

This document lands before code. The table below lists files that will change in subsequent PRs.

| File | Description |
| --- | --- |
| `models/heads.py::DecoderHead` (delete) | Remove the feature-reconstruction decoder class (lines 23-39) and its MSE loss. |
| `models/heads.py::ForwardDistributionHead` (new) | Forward-distribution head: 4 horizons x 41 bins, per-horizon symmetric ranges, two-hot targets, cross-entropy loss summed across horizons with equal weighting. |
| `models/world_model.py` (edit) | Remove `dec_target` computation (line 136), `loss_dec_sum` accumulation (lines 169-171), and `DecoderHead` instantiation (line 99). Add `ForwardDistributionHead` instantiation and wiring in `_step`. Add per-horizon loss logging. |
| `training/datamodule.py` (edit) | Add `forward_returns: (B, T, 4)` tensor to batch output · log-return of close price from bar t to bar t+h for each horizon h in {1, 5, 15, 30}. |
| `configs/world_model.yaml` (edit) | Add `forward_horizons: [1, 5, 15, 30]`, `forward_bins: 41`, `forward_ranges: [0.005, 0.010, 0.018, 0.025]`. Remove decoder-specific config if any. |
| `tests/test_forward_dist_head.py` (new) | Unit tests for `ForwardDistributionHead`: bin-edge computation correctness, two-hot encoding round-trip, loss gradient flow, output shape checks across all four horizons. |
| `models/mae_decoder.py` (keep) | Used only in Phase 5.0.5 MAE pretraining pipeline. Unrelated to the world-model decoder. Unchanged. |
| `models/encoder.py` (keep) | iTransformer encoder. Unchanged. |
| `models/rssm.py` (keep) | RSSM core. Unchanged. |
| `models/heads.py::RewardHead` (keep) | Reward head. Unchanged. |
| `models/heads.py::ContinueHead` (keep) | Continue head. Unchanged. |

## Implementation Approach

The pivot is split into independently reviewable PRs in the following order.

**PR 1 · Architecture doc and pivot doc.** This PR. Documentation only, no code changes. Establishes the architectural reference (`docs/design/ARCHITECTURE.md`) and the pivot changelog (this file).

**PR 2 · ForwardDistributionHead + unit test.** Implement the head class in `models/heads.py` alongside the existing `RewardHead` and `ContinueHead`. Add `tests/test_forward_dist_head.py` covering: bin-edge computation matches the formula in `docs/design/ARCHITECTURE.md` Section 6, two-hot encoding produces valid probability vectors for edge cases (exact bin centers, boundary values, mid-bin values), loss produces finite gradients, output shapes are correct for all four horizons.

**PR 3 · Datamodule change + test.** Add `forward_returns` computation to `training/datamodule.py`. For each trajectory step at kline index k and each horizon h, the target is ln(close[k+h] / close[k]). Verify that the tensor shape is (B, T, 4) and values match manual computation from raw kline data. Handle boundary conditions: if k+h exceeds the available data, the trajectory is truncated or the return is clamped.

**PR 4 · World-model wiring + 100-step smoke.** Delete `DecoderHead` from `models/heads.py`. Wire `ForwardDistributionHead` into `models/world_model.py::_step` (lines 120-233): replace decoder target/loss with forward-distribution target/loss, update loss aggregation to use L_forward instead of L_decoder. Update `configs/world_model.yaml` with the new parameters. Run a 100-step smoke test to verify all loss components are finite and gradients flow through the forward-distribution path.

**PR 5 · 30k diagnostic.** Run the full 30k diagnostic with the new architecture and evaluate against the validation gates defined in `docs/design/ARCHITECTURE.md` Section 11. This is the decision point: if the gates pass, advance to Phase 5.5 (100k full run). If they fail, consult ADR-003's contingency plan.

Each PR after the first depends on the previous one. PRs 2 and 3 are structurally independent (head implementation vs. data pipeline) but are sequenced to allow unit testing of the head in isolation before it needs real data.

## Mathematical / Statistical Details

### Empirical return distributions

Standard deviations of BTCUSDT 1m log-returns at each horizon, computed over 1.05M rows via `_quantile_check.py`:

| Horizon | Std | sqrt(h) x std_1 | Ratio |
| --- | --- | --- | --- |
| 1 bar | 0.00067 | — | — |
| 5 bar | 0.00148 | 0.00150 | 0.99 |
| 15 bar | 0.00254 | 0.00259 | 0.98 |
| 30 bar | 0.00357 | 0.00367 | 0.97 |

The near-perfect sqrt(h) scaling confirms the random-walk character of the conditional mean. The conditional variance, however, exhibits strong clustering (GARCH-like behavior), which is the regime-dependent structure that z_t is designed to capture.

### Bin-edge derivation

For horizon h with half-range R_h and n_bins = 41:

```
bin_centers_h[k] = -R_h + k * (2 * R_h / 40)    for k = 0, ..., 40
bin_width_h      = 2 * R_h / 40
```

| Horizon | R_h | Bin width |
| --- | --- | --- |
| 1 bar | 0.005 | 0.000250 |
| 5 bar | 0.010 | 0.000500 |
| 15 bar | 0.018 | 0.000900 |
| 30 bar | 0.025 | 0.001250 |

The ranges are rounded from the raw 99.9% empirical quantiles to clean decimal numbers. They provide 1.16x to 1.29x margin over the 99.9% quantile (q999: 1-bar 0.0039, 5-bar 0.0087, 15-bar 0.0149, 30-bar 0.0206), meaning roughly 0.1% to 0.2% of target mass per horizon clips into the edge bins · the model learns "extreme move" as the edge-bin category rather than wasting bin resolution on tail magnitude. The ranges do not follow exact sqrt(h) scaling (e.g., sqrt(5) x 0.005 = 0.0112 vs. the chosen 0.010, and sqrt(30) x 0.005 = 0.0274 vs. the chosen 0.025) because they are sized from the empirical distribution (which has fat tails that don't scale exactly as sqrt(h)) and then rounded to clean numbers.

### Two-hot encoding

The two-hot encoding from `RewardHead` (`models/heads.py:70-81`) generalizes directly to per-horizon ranges. For a target log-return v at horizon h:

1. Clamp: v' = clamp(v, -R_h, +R_h).
2. Normalized position: pos = (v' + R_h) / bin_width_h, yielding a value in [0, 40].
3. Lower bin: idx_lo = floor(pos), clamped to [0, 39].
4. Weights: w_lo = 1 - frac(pos), w_hi = frac(pos).
5. Target vector: t[idx_lo] = w_lo, t[idx_lo + 1] = w_hi, all others 0.

This is equivalent to linear interpolation between adjacent one-hot vectors. The resulting target is a valid probability vector (sums to 1.0), and the cross-entropy loss against it produces smooth gradients for the predicted distribution even when the target falls between bin centers.

### Loss aggregation

The forward-distribution loss replaces the decoder MSE loss in the total:

```
L = L_forward + L_reward + L_continue + 0.5 * L_dyn + 0.1 * L_rep
```

where L_forward = CE_1 + CE_5 + CE_15 + CE_30, with each CE_h being the per-sample cross-entropy averaged over the batch and over T - burn_in active trajectory steps.

## Design Decisions

All design decisions for this pivot are documented as Architectural Decision Records in `docs/design/ARCHITECTURE.md` Section 12. Summary with cross-references:

**ADR-001** · Equal loss weighting across forward horizons at bootstrap. Let the data show which horizons learn well; reweight after 30k evidence.

**ADR-000** · Feature-reconstruction decoder deleted. Root cause: 15-dim features have near-zero conditional entropy given h_t; z_t had no stochastic job.

**ADR-002** · Forward-distribution head with empirically-sized bins. Four horizons, 41 bins, ranges from 99.9% empirical quantiles.

**ADR-003** · Free-bits floor unchanged at 1.0 nat/latent. The floor was not the cause of Phase 5.3 collapse; the target was. Keep the floor and let the new target drive KL release.

**ADR-004** · MAE encoder pretraining preserved. Encoder input/output shapes are unchanged; retraining would waste compute and confound the diagnostic.

Refer to the ADRs for full context, rationale, and consequences. This document does not duplicate that content.

## Verification

This PR is documentation-only. Verification is a read-through review confirming:

1. All claims about existing code mechanics are traceable to specific files and line numbers in the repo.
2. The ADRs are internally consistent and reference the correct W&B run IDs (kk3mzb3k for Phase 5.3 diagnostic).
3. The file-change table accurately reflects the planned scope of subsequent PRs.
4. The bin-edge formula is consistent with the empirical data produced by `_quantile_check.py` when run against `data/market.duckdb`.

Subsequent PRs have concrete verification gates: unit tests (PR 2 and PR 3), 100-step smoke test with all-finite losses (PR 4), and the full validation gate battery defined in `docs/design/ARCHITECTURE.md` Section 11 (PR 5).

## Related Docs

- Architecture reference: `docs/design/ARCHITECTURE.md`
- Phase 5.3 full training: `docs/implementations/2026-05-04-phase5-3-rssm-full-train.md`
- Phase 5.3 diagnostic report (run kk3mzb3k): `docs/implementations/phase5-3-diag-report/kk3mzb3k.md`
- Phase 5.0.5 encoder pretrain: `docs/implementations/2026-05-03-phase5-0-5-encoder-pretrain.md`
- Phase 5.1 datamodule: `docs/implementations/2026-05-04-phase5-1-datamodule.md`
- Phase 5.5 (not yet written): 100k full training with forward-distribution head, contingent on Phase 5.4 diagnostic gates passing.
