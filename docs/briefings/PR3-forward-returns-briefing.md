# PR 3 Briefing · Datamodule `forward_returns` Tensor

**Project:** crypto-dreamer · Phase 5.4 forward-distribution pivot
**PR:** 3 of 7 · datamodule forward-return targets
**Author of brief:** design session, 2026-05-28
**Intended consumer:** autonomous coding agent operating under the self-correcting goal loop defined in Section 0
**Destination in repo:** `docs/briefings/PR3-forward-returns.md`

---

## 0 · Operating Protocol (read first, applies to the whole task)

You are completing PR 3 as a whole · implementation, all tests, and all documentation · under a self-correcting goal loop. The loop governs how you handle every open question and every failure in this brief.

### The loop

For each open question (Section 6) and for any test failure encountered during implementation:

1. **Attempt** · implement the current best hypothesis.
2. **Test** · run the relevant diagnostic for that question (each open question in Section 6 specifies its own test-and-diagnose protocol).
3. **On pass** · document the resolution in the findings doc (Section 7) and move to the next item.
4. **On fail** · do NOT force a green test by weakening the spec. Instead: document the failure (what you tried, what happened, the actual error or wrong value), investigate the root cause, form a new hypothesis, plan the fix in one or two sentences, implement it, and re-test.
5. **Iteration cap** · a maximum of **5 hypothesis-iterations per open question**. If you reach 5 without a clean pass, STOP work on that question, document all 5 attempts with their failure modes and your current best understanding of why it is hard, and escalate to the user in the final report. Do not proceed to dependent work that requires the unresolved question to be answered.

### Hard constraints on the loop

- **Never make a test pass by changing what the test asserts to something weaker than this brief specifies.** If a test is wrong (asserts something this brief did not ask for), that is itself a finding to document, not a silent edit.
- **Never resolve a correctness-of-meaning question by picking whatever makes tests green.** The semantic decisions in Section 3 are already resolved by the user and are not open questions. Do not relitigate them inside the loop.
- **No git commits, pushes, merges, or tags at any point.** Stay uncommitted through every iteration. The user reviews the complete arc and commits once.

### Git boundaries · do NOT cross

Do NOT run any of: `git commit`, `git push`, `git tag`, `git merge`, `gh pr create`, `gh pr merge`, `gh release`. Do not modify `.gitignore`. Do not delete files. You MAY read any file in the repo.

**Files you may modify:** `training/datamodule.py`, `tests/test_datamodule_forward_returns.py` (new). You MAY also create the documentation files listed in Section 7 and append to `docs/planning/BACKLOG.md`.

**Files you must NOT modify:** `models/world_model.py`, `models/heads.py`, `configs/world_model.yaml`, `data/ingest.py`, `envs/spot_btc.py`, or any other model/config/env file. Forward-return *consumption* is PR 4. This PR only *produces* the tensor and proves it correct.

---

## 1 · Where PR 3 Sits

PR 2 shipped `ForwardDistributionHead` (`models/heads.py`), which predicts categorical distributions over discretized log-returns at four horizons {1, 5, 15, 30} bars from the world-model state `feat = [h_t, z_t]`. The head is implemented and unit-tested but unused · nothing yet feeds it real targets.

PR 3 is the data-side counterpart. It must surface, for every trajectory step, the *actual* forward log-returns the head will be trained against. After PR 3, PR 4 wires the head and the targets together in `models/world_model.py::_step` and runs the 100-step smoke.

The authoritative spec for what these targets mean is `docs/design/ARCHITECTURE.md` Section 6 and ADR-002. Read both before writing code. The horizons, bin count, and ranges live there and must not be re-derived here.

---

## 2 · The Core Computation

For each trajectory step at kline index `k`, and for each horizon `h` in `{1, 5, 15, 30}`, the forward log-return target is:

```
forward_return[k, h] = ln(close[k + h] / close[k])
```

where `close[k]` is the close price of the kline at the trajectory step and `close[k + h]` is the close price `h` bars later in the *same contiguous kline series*.

The output tensor the datamodule must add to each batch is:

```
forward_returns: (B, T, 4)   float32   ln-returns, one per (step, horizon)
forward_valid:   (B, T, 4)   bool      True where the return is computable and trustworthy
```

`B` is batch size (32), `T` is trajectory length (48), and 4 is the horizon count. The horizon ordering in the last dim must be exactly `[1, 5, 15, 30]` to match the head's output ordering · verify against `ForwardDistributionHead.horizons`.

These two tensors travel alongside the existing batch outputs (`obs_window`, `actions`, `rewards`, `continues`, and any others the datamodule currently emits). Read `training/datamodule.py` to enumerate the current batch dict keys and add the two new keys without disturbing existing ones.

---

## 3 · Resolved Design Decisions (NOT open questions · do not relitigate)

These were decided by the user. Implement them as specified. They are recorded here so the loop cannot wander into a plausible-but-wrong alternative.

### 3.1 · Boundary handling at series end → MASK

When step `k` is within `h` bars of the end of available kline data, `close[k + h]` does not exist and the forward return is uncomputable. **Do not** clamp to zero (zero is a real signal · "price did not move" · and clamping teaches the model a lie). **Do not** forward-fill the last value. **Do not** silently drop the step.

Instead, set `forward_returns[k, h] = 0.0` as a neutral placeholder AND set `forward_valid[k, h] = False`. The placeholder value is never used in the loss because PR 4 will multiply the per-(step, horizon) loss by `forward_valid`. The placeholder exists only so the tensor has a defined value everywhere; its semantic meaning comes entirely from the mask.

Note that validity is per-horizon, not per-step: at a given `k` near the series end, the 1-bar return may be valid while the 30-bar return is not. The mask must reflect this · each of the four horizons gets its own validity bit.

### 3.2 · Mid-series data gaps → DETECT, LOG, DEFER MASKING

Binance occasionally has missing minutes (exchange downtime, maintenance). If klines `k` and `k + h` both exist in the table but are not actually `h` minutes apart (because minutes are missing between them), then `ln(close[k+h] / close[k])` silently computes a return over the *wrong* horizon.

For PR 3: **detect** these cases by checking that `ts[k + h] - ts[k]` equals exactly `h` minutes (for the relevant interval; the data is 1m so `h` bars should be `h` minutes). **Log** the count and location (timestamp ranges) of every gap-affected (step, horizon) pair. Write a summary to the findings doc: how many target values across the dataset are gap-affected, at which horizons, clustered in which time periods.

**Do NOT mask gap-affected returns in PR 3.** That is deferred to a backlog item you will add (Section 7). The reason for deferral: masking gaps correctly requires deciding policy (skip the step entirely, interpolate, re-anchor) and that is its own design decision the user wants to make with the gap statistics in hand. PR 3's job is to quantify the problem, not solve it. The `forward_valid` mask in PR 3 reflects ONLY series-end boundary invalidity (3.1), not gap invalidity. Make this scope boundary explicit in code comments and in the findings doc so a future reader does not assume gaps are already masked.

### 3.3 · Index alignment → READ THE DATAMODULE, DO NOT INVENT

The forward returns must align to the exact same kline indices `k` that the existing observation windows use for each trajectory step. The datamodule already maps klines into trajectory steps via some indexing scheme (and samples trajectories via a `WeightedRandomSampler` over months · see `docs/design/ARCHITECTURE.md` Section 3 and `docs/implementations/2026-05-04-phase5-1-datamodule.md`).

Before writing any computation, read `training/datamodule.py` in full and determine:

- Does the datamodule operate on a single contiguous kline series, or on pre-segmented episodes? This changes how "series end" (3.1) is defined · per-episode-end if segmented, per-series-end if contiguous.
- What is the exact index `k` (into the underlying kline array) that corresponds to trajectory step `t` in a sampled trajectory? The forward return at step `t` must use the close at that same `k`.
- Are observation windows already offset in any way (e.g. the observation at step `t` is the 256-bar window *ending* at `k`)? The forward return anchor `close[k]` must be the same `close` the step is "standing on," consistent with how `reward` and `continue` are aligned for that step.

If you cannot determine the indexing unambiguously from the code, STOP and report what is ambiguous. Do not guess an indexing scheme · a wrong alignment produces targets that are off-by-one or off-by-window and every downstream test in this brief could still pass while the targets are silently wrong. This is the single highest-risk part of PR 3.

---

## 4 · Implementation Roadmap

Work in this order. Each step has a verification before the next begins.

1. **Read phase.** Read `training/datamodule.py`, `docs/design/ARCHITECTURE.md` Sections 3 and 6, `docs/implementations/2026-05-04-phase5-1-datamodule.md`, and `models/heads.py::ForwardDistributionHead` (for horizon ordering). Produce a short internal note: the current batch dict keys, the kline-to-trajectory-step indexing scheme, whether data is contiguous or segmented, and where in the datamodule the new tensors should be computed. Resolve Section 3.3 here. If 3.3 is ambiguous, stop and escalate before writing code.

2. **Forward-return computation.** Implement the core `ln(close[k+h]/close[k])` computation aligned to the indexing from step 1. Vectorize across horizons and across the trajectory · avoid per-step Python loops where array operations work. Produce `forward_returns` and the series-end `forward_valid` mask (3.1).

3. **Gap detection.** Add the timestamp-delta check (3.2). Accumulate gap statistics. This does NOT alter `forward_valid` · it only feeds the findings doc and logging.

4. **Datamodule wiring.** Add `forward_returns` and `forward_valid` to the batch dict without disturbing existing keys. Confirm tensor dtypes (`float32` for returns, `bool` for valid) and shapes `(B, T, 4)`.

5. **Tests.** Implement the full test taxonomy in Section 5.

6. **Documentation.** Write the findings, test-results, and implementation docs and update the backlog (Section 7).

---

## 5 · Test Taxonomy

Create `tests/test_datamodule_forward_returns.py`. Follow the existing test conventions in `tests/` (read `tests/test_datamodule.py` for fixture and style patterns · note it requires `einops`, which is installed). Use the real `data/market.duckdb` where the existing datamodule tests do; use small synthetic fixtures where a controlled input is needed.

### 5.1 · Unit tests (pure computation, synthetic data)

- **test_forward_return_value_correctness** · construct a tiny synthetic close-price series with known values, compute expected `ln(close[k+h]/close[k])` by hand for several `(k, h)` pairs, assert the implementation matches to float32 tolerance.
- **test_horizon_ordering** · assert the last-dim ordering is exactly `[1, 5, 15, 30]` and matches `ForwardDistributionHead.horizons`.
- **test_series_end_mask_per_horizon** · construct a series where the last 30 steps progressively lose validity per horizon; assert `forward_valid` is True/False in exactly the right per-(step, horizon) pattern. The step at distance 1 from the end should have valid h=1 but invalid h=5,15,30, and so on.
- **test_placeholder_value_at_invalid** · assert that wherever `forward_valid` is False, `forward_returns` is exactly 0.0 (the neutral placeholder), so PR 4's mask-multiply is safe.
- **test_dtypes_and_shapes** · `forward_returns` is float32 `(B, T, 4)`, `forward_valid` is bool `(B, T, 4)`.

### 5.2 · Integration tests (through the datamodule, real data)

- **test_batch_contains_new_keys** · pull one real batch from the datamodule; assert `forward_returns` and `forward_valid` are present and correctly shaped, and that ALL pre-existing batch keys are still present and unchanged (compare key set before/after, and spot-check that an existing tensor like `obs_window` has its expected shape).
- **test_alignment_to_observation_window** · for a sampled trajectory, independently look up the close price at the kline index the step corresponds to (per the indexing resolved in 3.3) and confirm `forward_returns` was anchored on that exact close. This is the test that catches off-by-one/off-by-window alignment errors · design it carefully, it is the most valuable test in the file.
- **test_mask_consistency_across_batch** · across a full real batch, assert that `forward_valid` is False only near series/episode ends (3.1) and never spuriously in the interior (gaps are NOT masked in PR 3, so interior False values would indicate a bug).

### 5.3 · Real-data validation (the "production" check for this PR)

There is no live production system at PR 3 · the datamodule serves nothing yet. The equivalent of a production test here is a **real-data spot-check against hand-computed ground truth**:

- **test_real_data_spotcheck** · pick 5 random `(trajectory, step, horizon)` triples from real sampled batches. For each, query `data/market.duckdb` directly for `close[k]` and `close[k+h]`, compute the log-return by hand, and assert it matches the tensor value to float32 precision. At least one of the 5 should be near a series/episode boundary to exercise the mask path.
- **test_gap_detection_reports** · run gap detection over a known slice of the real data and assert the reported gap statistics are internally consistent (e.g. gap-affected count is less than total target count, timestamps of reported gaps actually have a delta other than `h` minutes).

---

## 6 · Open Questions (each with its own test-and-diagnose protocol)

These are genuine unknowns that the self-correcting loop should resolve. Each has a stopping condition; the 5-iteration cap from Section 0 applies per question.

### OQ-1 · Is the underlying kline data contiguous or segmented per episode?

**Why it matters:** determines whether "series end" boundary masking (3.1) is applied at the end of the whole series or at the end of each episode.

**Diagnose:** read `training/datamodule.py` for how episodes are constructed. If episodes are pre-segmented (separate index ranges per episode), the mask must apply at each episode's tail. If the data is one contiguous series sampled into windows, the mask applies only at the global series tail.

**Test:** `test_series_end_mask_per_horizon` and `test_mask_consistency_across_batch` should reflect whichever answer is correct. If you find episodes ARE segmented, add a test asserting masking at an interior episode boundary, not just the global end.

**Stop condition:** if after reading you cannot tell whether data is segmented, escalate · do not assume contiguous.

### OQ-2 · Does the existing datamodule have spare data past the trajectory window to satisfy forward lookups, or does the lookup itself reduce usable trajectory positions?

**Why it matters:** if the datamodule currently uses klines right up to the end of its sampling range for trajectory steps, then a 30-bar forward lookup at the last step needs 30 klines that may not exist · which is exactly what the mask handles, but it also means the *fraction* of masked targets could be non-trivial and worth quantifying.

**Diagnose:** compute, over a representative set of sampled batches, what fraction of the `(B, T, 4)` targets are masked at each horizon. If the 30-bar horizon is masking more than a few percent of steps, that is a finding worth surfacing · it means the model sees materially less 30-bar signal than 1-bar signal, which interacts with the equal-weighting decision (ADR-001).

**Test:** add `test_mask_fraction_per_horizon` that computes and reports (not asserts a hard threshold · just logs to the findings doc) the masked fraction per horizon over real batches.

**Stop condition:** this is a measurement, not a pass/fail · record the numbers and move on. Only escalate if the 30-bar masked fraction exceeds 15%, which would be a real design concern for ADR-001.

### OQ-3 · How prevalent are mid-series gaps, and at which horizons do they bite hardest?

**Why it matters:** feeds the deferred backlog decision on gap masking. Longer horizons span more minutes and are more likely to straddle a gap.

**Diagnose:** run gap detection (3.2) across the full dataset. Tabulate gap-affected target counts per horizon and the time clustering of gaps.

**Test:** `test_gap_detection_reports` validates the detector's internal consistency. The actual statistics go in the findings doc.

**Stop condition:** record the statistics. If gap-affected targets exceed 5% at any horizon, flag prominently in the findings doc and the backlog item, because it raises the priority of the deferred masking work.

---

## 7 · Documentation Requirements

Produce all of the following. Use the established style conventions (Section 8).

### 7.1 · Implementation doc

`docs/implementations/2026-05-28-phase5-4-pr3-forward-returns.md`, following the established template (Problem/Motivation, What Changed, Implementation Approach, Mathematical Details, Design Decisions with cross-references to ADR-002, Verification, Related Docs). Document the indexing scheme you resolved for 3.3 explicitly · this is the single most important thing for a future reader to understand.

### 7.2 · Findings doc

`docs/findings/2026-05-28-forward-returns-data-quality.md`. Contains: the masked-fraction-per-horizon table (OQ-2), the gap-prevalence statistics (OQ-3), and a short interpretation of what these mean for ADR-001 (equal horizon weighting) and for the deferred gap-masking work. This is the quantitative payload of the PR.

### 7.3 · Test results doc

A section either inside the implementation doc or as `docs/findings/2026-05-28-pr3-test-results.md` recording: every test in Section 5 and its outcome, and · critically · for any open question that triggered the self-correcting loop, the full iteration history (hypothesis, test, failure, re-hypothesis) up to resolution or the 5-iteration cap. The loop history is the most valuable artifact for the user to review; do not summarize it away.

### 7.4 · Backlog update

Append to `docs/planning/BACKLOG.md` under the Operational section: a new item for **gap masking in forward-return targets**, sized by the OQ-3 statistics (small if gaps are rare, medium if prevalent), with a one-line rationale referencing the findings doc. If OQ-2 surfaced a high 30-bar masked fraction, also add a backlog or roadmap note flagging the ADR-001 interaction.

---

## 8 · Style Conventions (locked)

- Middle dot (`·`, U+00B7) for separators. NEVER em dash or en dash.
- ASCII only. No smart quotes.
- File paths in backticks. Code references include line numbers where possible.
- Type hints on all new public functions; docstrings explaining args, returns, and tensor shapes.
- Match the code style of the existing `training/datamodule.py` and `tests/test_datamodule.py`.
- No `print()` in production code. No banner comments.
- Run `ruff check` and `ruff format --check` on every modified file before finishing. CI runs `ruff format --check`, so format must be clean.
- Prose over bullets in docs except for genuine enumerations.

---

## 9 · Final Report

Print at the end of the run:

1. Files modified and created (paths only).
2. Full output of `pytest tests/test_datamodule_forward_returns.py -v`.
3. Full output of `pytest tests/ -x -q --ignore=tests/test_dream_endpoint.py` (confirm the existing 42 still pass · we added tests, so expect more than 42).
4. Output of `ruff check` and `ruff format --check` on modified files.
5. The masked-fraction-per-horizon table (OQ-2) and gap-prevalence summary (OQ-3), inline in the report so they are visible without opening the findings doc.
6. For each open question (OQ-1, OQ-2, OQ-3): resolved / escalated, and if any hit the 5-iteration cap, a clear ESCALATION marker with the blocking issue stated.
7. The resolved answer to 3.3 (the indexing scheme) stated in one or two sentences · the user will sanity-check this against their own understanding of the datamodule.

Do not stage, commit, or push.
