# 2026-05-04 — Phase 5.1: datamodule + encoder integration

## Problem / Motivation

Phase 5.0.5 produced a pretrained iTransformer encoder. Phase 5.2 will
train a DreamerV3-style RSSM world model on top of it. Between those
two phases sits a piece of plumbing: a Lightning DataModule that turns
the agent's `step_log` table into trajectory batches the world model
can consume, and an encoder constructor that knows how to absorb the
pretrained weights when expanded from 12 to 15 input variables.

This phase delivers both, plus the smoke tests that lock the contract
in place — so when 5.2 starts wiring the RSSM, batch shapes, dtypes,
and "what does index 12-14 of obs_window even mean" are no longer
open questions.

## What Changed

| File | Description |
| --- | --- |
| `tests/conftest.py` | Added `synthetic_db_with_steps` fixture: extends the existing `synthetic_db` with a 2-episode `step_log` (one truncated, one early-terminated) so datamodule tests can run hermetically without touching the real DBs. |
| `models/encoder.py` | `iTransformerEncoder.__init__` now accepts `mae_checkpoint: str \| Path \| None`. When provided AND the file exists, copies `input_proj` (shared), all transformer layer weights, and the first `min(n_vars, ckpt_n_vars)` rows of `var_embed`. Remaining variable embeddings (e.g. portfolio rows 12..14 when expanding from 12→15) stay at random init. Logs a single line summarizing what was copied vs. skipped. Missing checkpoint or shape mismatch downgrades to a `WARNING` and full random init — never crashes the training pipeline. |
| `training/datamodule.py` (new) | `SpotBTCDataModule` — Lightning DataModule. `setup()` reads klines from `data/market_ro.duckdb`, computes the 12-feature matrix once via `envs.spot_btc.compute_feature_block`, reads `step_log` from `data/market.duckdb`, and groups rows into per-`(agent_id, episode)` `_EpisodeArrays`. Train/val partitioning is per-step by month-fraction (last 15% of each calendar month → val); subsequences are forbidden from crossing the boundary. A `WeightedRandomSampler` reweights starts by 1/month_size so every calendar month contributes roughly equally per epoch. |
| `tests/test_datamodule.py` (new) | Four tests: (1) batch shapes & dtypes on the synthetic fixture, (2) `next_obs_window[t] == obs_window[t+1]` shift correctness, (3) real-DB end-to-end forward through the encoder (skipped if DBs missing), (4) pretrain-checkpoint partial-copy verification (skipped if checkpoint missing). |

No changes to `envs/`, `agents/`, `serve/`, or `dashboard/`.

## Implementation Approach

**Episode arrays.** `step_log` is grouped by `(agent_id, episode)` and
each group becomes a small dataclass of numpy arrays — `ts`,
`kline_idx`, `action`, `realized`, `equity`, `reward`, `cont`,
`is_first`. The arrays are tightly packed (int64 / float32 / bool) so
504 episodes total ~10 MB RAM. Keyed by tuple in a dict so subsequent
slicing in `__getitem__` is O(1).

**Kline-index pre-resolution.** Because the kline table is gap-free at
1-min resolution (verified at end of 5.0b), the kline row index for any
step's `ts` is exactly `(ts - kline_t0).total_seconds() / 60`. This
runs once per episode at `setup()` time and replaces what would
otherwise be a per-`__getitem__` dict lookup or binary search.

**Subsequence enumeration.** For each episode of length `n`, every
position `start ∈ [0, n - T)` produces a candidate `(agent_id, episode,
start)` tuple, provided:
1. All `T+1` raw rows `[start..start+T]` are in the same train/val
   partition (no boundary-crossing).
2. The kline window for the first row fits — `kline_idx[start] ≥
   seq_len`. This trivially holds for all real episodes (the env
   always starts at `_t ≥ WINDOW`), but the synthetic fixture has
   short klines so the guard matters.

**Obs window construction.** For each of the `T+1` rows:
- `market[j] = feature_cache[kline_idx[j] - seq_len : kline_idx[j]]` →
  shape `(seq_len, 12)`.
- `portfolio[j] = (realized[j], 1 - realized[j], log(equity[j] /
  10000))` → shape `(3,)`.
- The portfolio is broadcast across the seq_len axis so the
  obs_window has shape `(seq_len, 15)` with cols 12..14 *constant*
  along the time dim. The iTransformer's per-variable Linear treats a
  constant series as `Linear(1 → d_model)` for that variable — exactly
  what we want for "current portfolio as side-channel".

`obs_window` returns rows `[0..T-1]`, `next_obs_window` returns rows
`[1..T]`. The world model's training loop will use them as
`(obs_t, action_t) → predict obs_{t+1}, reward_t`.

**Continue / is_first.** `cont[i]` is True everywhere except the *last*
step of an episode whose final equity dropped below `0.5 *
INITIAL_CASH = 5000`. Truncated episodes (those that hit the
1440-step cap) keep `cont = True` throughout, including the last step.
`is_first[i]` is True only at episode-step 1; sub-sequences sampled
mid-episode have `is_first` False everywhere, and the world model
handles the continued hidden-state via burn-in.

**Month-stratified sampler.** With 24 calendar months in the train
pool and counts ranging from 12,163 (2024-05) to 38,627 (2025-05),
uniform sampling over starts would over-represent dense months by
~3×. The `WeightedRandomSampler(replacement=True)` weights each start
by `1 / month_size[start_month]` so each month's *expected* per-epoch
contribution is equal. Replacement is set so a single epoch can
revisit popular starts when the model needs more updates per epoch.

**Encoder pretrain copy.** `_load_mae_checkpoint` is intentionally
permissive: it copies whatever shape-compatible tensors it finds and
logs the rest. So if Phase 5.0.5 had been aborted at the 30-min stop
rule, training still proceeds with a full-random encoder rather than
crashing.

## Mathematical / Statistical Details

**Train/val month-fraction split.** For step at timestamp `ts` with
`day = ts.day`, `D = days_in_month(ts)`:

    month_frac(ts) = (day - 1) / D
    val_partition  = month_frac(ts) >= 0.85

Equivalent at calendar resolution: a date is val iff its day-number
within the month is in the top 15% of that month's day-numbers.
30-day months: days 26-30; 31-day months: days 27-31; 28-day months:
days 25-28; 29-day months: days 26-29.

This is *not* time-localized leakage-safe in the strictest sense —
adjacent train/val days are still close in time. But for evaluating
month-by-month regime understanding (the goal of the validation
gates in 5.4), it's the right granularity.

**Why per-month inverse-frequency weighting.** With months containing
from ~12K to ~39K starts, naive uniform sampling means a model sees
a 2025-05 start ~3× more often than a 2024-05 start. The weights
`w_i = 1 / |month(i)|` then normalized give

    P(sample i) = (1 / |month(i)|) / Σ_j (1 / |month(j)|)
                = 1 / (|month(i)| · M)         where M = number of months

so the marginal probability of any *month* being represented in a
batch is exactly `1/M`, regardless of that month's size. Random
within-month: each start in month m is equally likely conditional on
month m being chosen.

## Design Decisions

- **Snapshot vs live DB for klines.** The 5.0.5 snapshot
  (`data/market_ro.duckdb`) is reused for klines reads in 5.1 and
  beyond — it's frozen at the moment after the 2y backfill, so reads
  are deterministic across phases. `step_log` reads the live
  `data/market.duckdb` because that's where new agent runs would
  write; we accept that running new agents in parallel with training
  would cause lock contention (Windows-exclusive write lock — known
  v1 limitation).
- **Constant-portfolio broadcast vs per-step portfolio history.**
  Considered: at each trajectory step `s`, building col 12-14 from
  the *historical* portfolio trajectory leading up to `s` (not the
  scalar at `s` repeated). Rejected — the env's observation contract
  passes only the current portfolio scalar, and matching that
  contract preserves the "encoder sees what the agent saw" property
  the pretrained weights already understand. The world model can
  recover portfolio history from the action sequence + RSSM state
  anyway.
- **Subsequence sampling within episodes vs across episodes.** Always
  within. A trajectory that splices two different episodes would have
  a discontinuity the RSSM has no way to reset for (we'd need to
  insert a synthetic `is_first=True` mid-trajectory). Within-episode
  sampling keeps the temporal structure clean and `is_first` simple.
- **Subsequence cannot cross train/val boundary.** Plan §6. Enforced
  by checking `partition[start..start+T+1].min() == .max()` before
  enumerating. Episodes that straddle the boundary contribute
  multiple non-overlapping runs (one per partition).
- **WeightedRandomSampler vs custom batch sampler.** Lightning's
  default DataLoader works with a `WeightedRandomSampler` and yields
  the standard `(B, ...)` collate. A custom batch sampler that
  guarantees "at least N distinct months per batch" would be more
  precise but adds complexity for ~no gain — with `M=24` months
  and `B=16`, expected distinct months per batch with the weighted
  sampler is already ~10.

## Verification

1. **Existing tests still pass.**
   ```
   pytest tests/test_env_smoke.py tests/test_features.py tests/test_encoder.py -q
   ```
   `16 passed`.

2. **New datamodule tests.**
   ```
   pytest tests/test_datamodule.py -q
   ```
   `4 passed` — synthetic shapes, next/obs shift correctness, real-DB
   end-to-end forward, pretrain-checkpoint partial-copy.

3. **Real-data smoke** (run from the project root):
   ```python
   from training.datamodule import SpotBTCDataModule
   dm = SpotBTCDataModule(klines_db="data/market_ro.duckdb",
                          steps_db="data/market.duckdb",
                          batch_size=16, T=64)
   dm.setup()
   batch = next(iter(dm.train_dataloader()))
   ```
   Expected output:
   - `dm.episode_count`: 504
   - `dm.terminated_count`: 237 (≈ 47% — random-policy hitting the 50%
     drawdown guardrail roughly half the time, consistent with v1
     phase-1 analysis)
   - Train subsequence starts: **586,278**
   - Val subsequence starts: **93,230** (val_frac = 13.7%, close to
     the 15% target — minor downward drift from subsequences being
     forbidden from crossing the boundary)
   - Setup time: ~6 sec (one-shot kline read + feature compute)
   - Batch construction: ~25 ms (40 batches/sec sustained on CPU,
     `num_workers=0`)

4. **Batch contract** (verified in `dm.train_dataloader()` output):

   | key | shape | dtype |
   |---|---|---|
   | `obs_window` | `[16, 64, 256, 15]` | float32 |
   | `next_obs_window` | `[16, 64, 256, 15]` | float32 |
   | `action` | `[16, 64]` | int64 |
   | `reward` | `[16, 64]` | float32 |
   | `continue_flag` | `[16, 64]` | bool |
   | `is_first` | `[16, 64]` | bool |

5. **Encoder forward** (with pretrain ckpt loaded):
   - Constructor incl. ckpt copy: 0.01 sec
   - Output shape `[1024, 15, 128]` (B·T flattened to leading dim)
   - All-finite
   - GPU forward: ~24 ms/pass at fp32, B·T=1024 — bf16 will improve
     this; well under the per-step budget for the 5.3 RSSM run.

6. **Regime distribution** (`dm.month_summary` printout):

   | month | train_starts | val_starts |
   |---|---|---|
   | 2024-05 | 12,163 | 2,210 |
   | 2024-06 | 19,553 | 4,041 |
   | 2024-07 | 30,633 | 699 |
   | 2024-08 | 18,031 | 3,010 |
   | 2024-09 | 26,796 | 6,022 |
   | 2024-10 | 20,703 | 2,695 |
   | 2024-11 | 27,703 | 536 |
   | 2024-12 | 24,479 | 3,483 |
   | 2025-01 | 19,867 | 2,330 |
   | 2025-02 | 21,674 | 2,550 |
   | 2025-03 | 24,801 | 10,014 |
   | 2025-04 | 27,877 | 3,671 |
   | 2025-05 | 38,627 | 5,053 |
   | 2025-06 | 16,368 | 4,507 |
   | 2025-07 | 34,765 | 4,721 |
   | 2025-08 | 24,201 | 4,408 |
   | 2025-09 | 26,529 | 3,146 |
   | 2025-10 | 14,270 | 3,264 |
   | 2025-11 | 24,938 | 12,223 |
   | 2025-12 | 21,002 | 4,248 |
   | 2026-01 | 24,350 | 5,441 |
   | 2026-02 | 25,798 | 1,119 |
   | 2026-03 | 29,765 | 1,283 |
   | 2026-04 | 31,385 | 2,556 |

   Every month has both train and val starts. Per-month variance is
   high (val_starts range 536 → 12,223) because the random agents
   started at random points in the 2y window — months whose val
   sub-window happened to land between agent runs got fewer val
   starts. This is acceptable; the world model evaluates on
   per-batch loss, not per-month, so as long as every month is
   represented, distribution shift is exposed.

## Reproducibility

- DataModule constructor takes `seed=42`; threaded into the
  `WeightedRandomSampler`'s `torch.Generator`.
- `__getitem__` is deterministic given the same `start` index.
- No cuDNN nondeterminism here — the encoder is invoked at eval
  time only in tests, and at train time the determinism budget is
  inherited from the world-model trainer (Phase 5.2).

## Related Docs

- Plan: `~/.claude/plans/we-re-starting-phase-5-staged-pinwheel.md`
- Phase 5.0.5 encoder pretrain: `2026-05-03-phase5-0-5-encoder-pretrain.md`
- Up next (Phase 5.2): `models/rssm.py`, `models/heads.py`,
  `models/world_model.py`, `training/train_world_model.py`,
  `configs/world_model.yaml`. The DataModule built here is the only
  data input the world model needs.
