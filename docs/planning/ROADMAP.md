# crypto-dreamer · Project Roadmap

This document is the authoritative ordering of work units through Phase 6. It is a planning artifact; effort estimates are guidance, not commitments. The companion file `docs/planning/BACKLOG.md` holds future-work items not yet on the active roadmap. Update this document as work completes (see Section 7).

**OMI freeze date assumption: 2026-06-15.** All Phase 5.4 implementation and operational PRs must land before this date. If the actual OMI start date differs, adjust the timeline accordingly.

## Current State (2026-05-27)

Phase 5.3 diagnostic completed and failed with a documented root cause: the feature-reconstruction decoder target has insufficient stochastic structure for z_t (hard posterior collapse, kl_unclipped = 25.7 nat against a 32-nat floor, W&B run `kk3mzb3k`). Phase 5.4 documentation is complete: the architecture reference (`docs/design/ARCHITECTURE.md`), the pivot changelog (`docs/implementations/2026-05-27-phase5-4-pivot-forward-distribution.md`), and the marginal baseline findings (`docs/findings/2026-05-27-marginal-baseline.md`, total baseline 8.8632, drift 0.015 nat). Phase 5.4 implementation has not started. The next concrete action is PR 2: implementing `ForwardDistributionHead`.

---

## 1 · Phase 5.4 Implementation

The next 2-3 weeks of work. Six PRs that must land before the OMI freeze. PRs 2-5 are the model pivot. PRs 6-7 are operational infrastructure for the cold period.

### PR 2 · ForwardDistributionHead

**Goal:** Implement the forward-distribution prediction head as a standalone, unit-tested class.

**Scope:**
- Add `ForwardDistributionHead` class to `models/heads.py` alongside `RewardHead` and `ContinueHead`
- Four horizons {1, 5, 15, 30} bars, 41 bins each, per-horizon symmetric ranges ±{0.005, 0.010, 0.018, 0.025}
- Two-hot encoding generalized from `RewardHead.two_hot_encode` (`models/heads.py:70-81`)
- Cross-entropy loss per horizon, summed with equal weighting (ADR-001)
- Add `tests/test_forward_dist_head.py` covering: bin-edge correctness, two-hot encoding for edge cases (exact centers, boundaries, mid-bin), loss produces finite gradients, output shapes for all four horizons

**Out of scope:**
- World-model wiring (PR 4)
- Datamodule changes (PR 3)

**Dependencies:** PR 1 (documentation · already complete).

**Verification gate:** `pytest tests/test_forward_dist_head.py -v` passes with all assertions green. Gradient flow verified by calling `.backward()` on the loss and checking all head parameters have non-zero `.grad`.

**Off-ramp:** If loss produces NaN or infinite gradients, debug the two-hot encoding math and log-softmax numerics before proceeding. Check that bin edges do not produce degenerate one-hot targets at boundary values.

**Effort estimate:** 4-6 hours.

### PR 3 · Datamodule forward_returns tensor

**Goal:** Add forward-return targets to the training data pipeline.

**Scope:**
- Add `forward_returns: (B, T, 4)` tensor to `training/datamodule.py` batch output
- For each trajectory step at kline index k and each horizon h in {1, 5, 15, 30}, compute ln(close[k+h] / close[k])
- Handle boundary conditions: if k+h exceeds available kline data, clamp or truncate
- Add unit test verifying tensor shape and values against manual computation from raw kline data

**Out of scope:**
- Modifying the world-model training loop (PR 4)
- Changing the feature pipeline or observation space

**Dependencies:** None (structurally independent of PR 2, but sequenced after it for clean review).

**Verification gate:** Unit test passes. Manual spot-check: pick 5 random trajectory steps, compute forward returns by hand from `data/market.duckdb`, compare against the tensor values. All must match to float32 precision.

**Off-ramp:** If the kline data has gaps (missing minutes) that cause forward-return computation to silently use wrong bars, add gap detection and skip affected trajectories rather than producing incorrect targets.

**Effort estimate:** 3-5 hours.

### PR 4 · World-model wiring + 100-step smoke

**Goal:** Wire the new head into the world model, remove the old decoder, and verify the full loss pipeline in a short smoke test.

**Scope:**
- Delete `DecoderHead` class from `models/heads.py` (lines 23-39)
- Remove decoder-related machinery from `models/world_model.py`: `dec_target` computation (line 136), `loss_dec_sum` accumulation (lines 169-171), `DecoderHead` instantiation (line 99)
- Add `ForwardDistributionHead` instantiation and wiring in `models/world_model.py::_step`
- Add per-horizon loss logging (`loss_forward_1`, `loss_forward_5`, `loss_forward_15`, `loss_forward_30`)
- Update `configs/world_model.yaml`: add `forward_horizons`, `forward_bins`, `forward_ranges`; remove decoder-specific config if any
- Run 100-step smoke test with `max_episodes=5`

**Out of scope:**
- Hyperparameter tuning
- Running more than 100 steps

**Dependencies:** PR 2 (head class) and PR 3 (forward_returns tensor).

**Verification gate:** 100-step smoke completes without NaN or Inf in any loss component. All six logged losses (forward_1, forward_5, forward_15, forward_30, reward, continue) are finite and decreasing or stable. KL components are finite. Existing test suite still passes: `pytest tests/ -x -q --ignore=tests/test_dream_endpoint.py`.

**Off-ramp:** If losses are finite but forward-distribution losses are not decreasing after 100 steps, this is expected (100 steps is too few for learning signal). If losses are NaN, check the wiring: verify that `forward_returns` tensor aligns with the correct trajectory step, and that the head receives `feat` (h_t concatenated with z_t) not just h_t.

**Effort estimate:** 4-8 hours.

- **Step-alignment trace (load-bearing).** Add a test that follows one
  concrete trajectory step end to end · kline index `k` → observation
  window → RSSM belief state `feat = [h_t, z_t]` → ForwardDistributionHead
  input → `forward_returns[:, t]` target · and asserts every link
  references the same `k`. PR 3 confirmed forward returns anchor on the
  `step_log`-aligned bar (`k_decision + 1`, the bar the agent moved into,
  one past the obs window). PR 4 must pair `feat` at step `t` with
  `forward_returns` at the SAME step `t`; an off-by-one pairing here is a
  silent temporal lag or look-ahead leak that a finite-loss smoke test
  will NOT catch. This trace is the PR 4 analogue of PR 3's
  `test_alignment_to_observation_window` and protects the whole pivot.

### **PR 5 · 30k Diagnostic Run + Gate Evaluation** · CRITICAL PATH

**Goal:** Run the full 30k diagnostic with the forward-distribution head and determine whether the Phase 5.4 hypothesis holds.

**Scope:**
- Configure `configs/world_model.yaml` for diagnostic mode (30k steps, W&B online)
- Apply powercfg mitigations before starting (see Phase 5.3 stall postmortem in `docs/implementations/2026-05-04-phase5-3-rssm-full-train.md`)
- Run 30k steps on the 4070 laptop (~3 hours training wall clock)
- At step 20k, evaluate Gate 1 (KL_unclipped > 32 nat), Gate 2 (val/loss_forward_dist < 8.85), Gate 3 (reward NLL ~0.48)
- Write findings doc at `docs/implementations/2026-MM-DD-phase5-4-diagnostic.md`
- If gates pass, write ADR confirming equal weighting (ADR-001 resolution)
- If gates fail, trigger the decision tree in Section 2 below

**Out of scope:**
- Running past 30k steps (that is Phase 5.5)
- Tuning hyperparameters during the run

**Dependencies:** PR 4 (wired world model with smoke-tested loss pipeline).

**Verification gate:** See Section 2 (The Gate Decision Point). This is the only PR whose verification gate determines the project's future direction.

**Off-ramp:** If training crashes (NaN, OOM, Modern Standby), diagnose and restart from the latest checkpoint. The heartbeat callback provides crash detection. If the run completes but gates fail, follow the decision tree in Section 2.

**Effort estimate:** 2-3 days elapsed (1 day setup and training, 1-2 days analysis and findings doc).

### PR 6 · Daily ingestion service

**Goal:** Ensure `data/market.duckdb` stays current with 1-minute klines during the OMI cold period.

**Scope:**
- Create `scripts/daily_ingest.py`: connects to Binance public API, fetches klines from the last-ingested timestamp to now, appends to `data/market.duckdb`
- Reuse the ingestion logic from `data/ingest.py` (which already handles Binance kline fetching and DuckDB insertion)
- Create a Windows Task Scheduler XML definition for daily execution at a fixed time (e.g. 06:00 UTC)
- Write a setup doc at `docs/operations/daily-ingest-setup.md` with installation steps
- Add logging to a file (`logs/daily_ingest.log`) with rotation

**Out of scope:**
- Live WebSocket streaming (backlog item)
- Ingesting assets other than BTCUSDT (backlog item)

**Dependencies:** None (can be developed in parallel with PRs 2-4).

**Verification gate:** Run the script manually and verify that new rows appear in `data/market.duckdb` for the current date. Check that re-running is idempotent (INSERT OR IGNORE). Verify the Task Scheduler task triggers correctly with `schtasks /run /tn "crypto-dreamer-ingest"`.

**Off-ramp:** If Binance API access is unreliable from the laptop's network, add retry logic with exponential backoff. If the Task Scheduler XML doesn't trigger correctly, fall back to a simple PowerShell scheduled task.

**Effort estimate:** 3-5 hours.

### PR 7 · Weekly health check service

**Goal:** Detect codebase or data rot during the OMI cold period before it compounds.

**Scope:**
- Create `scripts/weekly_health.py`: runs pytest, runs 100-step model smoke, checks `data/market.duckdb` for recent data (freshness gate: last kline within 48 hours), reports results
- Log output to `logs/weekly_health.log` with timestamps
- Create a Windows Task Scheduler XML for weekly execution (e.g. Sunday 03:00 UTC)
- Exit code 0 on all-pass, nonzero on any failure (for Task Scheduler history)

**Out of scope:**
- Sending alerts (email, Slack) · the log file is sufficient for the OMI period
- Model training or checkpoint evaluation beyond the smoke test

**Dependencies:** None (can be developed in parallel with PRs 2-4).

**Verification gate:** Run the script manually and verify it produces a clean log entry with pytest results, smoke test status, and data freshness check. Verify the Task Scheduler task triggers and records exit code.

**Off-ramp:** If the 100-step smoke fails because the model code has changed (e.g. after PR 4 deletes DecoderHead), update the smoke test configuration to match the current model architecture before the OMI freeze.

**Effort estimate:** 3-5 hours.

---

## 2 · The Gate Decision Point

This section describes what happens after PR 5's 30k diagnostic finishes. It is the only place where the project's plan honestly forks.

**IF** kl_unclipped at step 20k > 32 nat **AND** val/loss_forward_dist < 8.85:

Gate 1 and Gate 2 pass cleanly. Phase 5.5 (100k full run) is authorized. Proceed to Section 3. Write a findings doc confirming the gates passed and resolve ADR-001 (equal weighting either confirmed or revised based on per-horizon loss curves). This is the desired outcome.

**IF** kl_unclipped at step 20k > 32 nat **BUT** val/loss_forward_dist is in [8.85, 8.86]:

Gate 1 passes. Gate 2 is ambiguous (within the drift band documented in `docs/findings/2026-05-27-marginal-baseline.md`). Three options:

1. Extend the diagnostic by 10k steps (to 40k total) and re-evaluate. If val/loss_forward_dist drops below 8.85 by step 30k, accept Gate 2 as a slow pass.
2. Accept ambiguous Gate 2 and proceed to Phase 5.5, but log per-horizon forward losses at higher granularity (every 500 steps instead of 2500) to detect whether the model is genuinely learning conditional structure or riding the train-val drift.
3. Pause for a full review of the per-horizon loss curves before deciding.

Write ADR-005 documenting the chosen option and the evidence that informed it. Any of these options is defensible; the key is that the decision is explicit and documented.

**IF** kl_unclipped at step 20k <= 32 nat:

Gate 1 fails. The forward-distribution target did not release KL from the free-bits floor. The Phase 5.4 hypothesis is falsified: giving z_t a stochastic prediction task at the return-distribution level was not sufficient to produce regime encoding.

Trigger ADR-003 contingency review. Write ADR-006 evaluating three options:

(a) Prior capacity restriction · reduce `prior_head` hidden dim (e.g. 256 -> 64) to force the prior to be less expressive, making the posterior's information advantage more valuable. This is the canonical DreamerV3 fix for posterior collapse.

(b) KL warmup schedule · start with zero KL weight and ramp to full over 5k steps, giving the posterior time to find useful structure before the prior learns to match it. This addresses a potential early-training failure mode.

(c) Paradigm change · drop the world model entirely and train a model-free RL agent (e.g. PPO or SAC) on the same observation/reward setup. This concedes that the data does not have exploitable stochastic structure at the regime level that a latent variable model can capture.

Do NOT proceed to Phase 5.5 in any form. Re-plan from here. All roadmap entries past this point become contingent on the ADR-006 decision.

---

## 3 · Phase 5.5 (100k Full Run)

**Goal:** Produce a world-model checkpoint suitable for Phase 6 actor training.

**Pre-conditions:** Gate 1 and Gate 2 pass cleanly from PR 5.

**Approach:** Resume from the Phase 5.4 30k diagnostic checkpoint, extend to 100k total steps. Run on the 4070 laptop with full powercfg mitigations (standby-timeout-ac 0, monitor-timeout-ac 0, lid action disabled). Expected wall clock: 15-20 hours. Plan as 2-3 overnight blocks: start before bed, check heartbeat log in the morning, resume if interrupted.

**Verification gate:** KL_unclipped remains stable above 32 nat across the full 100k (no regression after the diagnostic checkpoint). val/loss_forward_dist is trending downward without plateauing by step 80k. No NaN events in any loss component. Reward NLL stable at ~0.48 throughout. Per-horizon forward losses are all individually below their horizon-specific marginal baselines.

**Effort estimate:** 1-2 weeks elapsed (mostly waiting on training, not implementation). Implementation work is minimal: update config to `mode=full`, `max_steps=100000`, set a new `run_name`, and start the run.

**Output:** `checkpoints/world_model_full_100k.ckpt` + a Phase 5.5 findings doc at `docs/implementations/2026-MM-DD-phase5-5-full-run.md`.

**Contingency note:** If Phase 5.4 diagnostic does not complete until early-to-mid June, Phase 5.5 may not be realistic before the OMI freeze (June 15). In that case, the freeze point is the Phase 5.4 diagnostic checkpoint and Phase 5.5 becomes the first task on thaw. This is acceptable · the diagnostic checkpoint is a known-good state (or a known-failed state with ADR-006), and Phase 5.5 is a pure continuation that requires no design decisions.

---

## 4 · Phase 6 (Actor Training)

**Goal:** Train an RL actor on top of the frozen Phase 5.5 world model.

**Pre-conditions:** Phase 5.5 checkpoint exists and passes its verification gate.

**Open questions:**
- Which actor algorithm: DreamerV3-style imagination training (actor learns entirely within the world model's dream rollouts), real-environment training (actor interacts with the spot_btc env directly), or a hybrid (imagination pre-training followed by real-environment fine-tuning).
- Whether to apply additional reward shaping beyond the current log-return minus turnover penalty formula.
- Whether the 5-action discrete allocation space ({0%, 25%, 50%, 75%, 100%}) is granular enough or should be expanded.
- How to handle the distribution shift between the world model's learned dynamics and the actual market during actor deployment.

**Effort estimate:** 4-8 weeks of focused work, much of it research rather than implementation. The actor algorithm choice alone may require a literature review and a small-scale comparison experiment.

**Out of scope for this roadmap:** Detailed PR breakdown. Phase 6 gets its own design document when Phase 5.5 lands.

---

## 5 · Beyond Phase 6

Long-horizon items that are not planned at PR-level detail. Paper draft documenting the architecture and Phase 5-6 results. Public release of the codebase with a cleaned-up README. Production backtesting on held-out 2026 data that was not seen during training. Multi-asset extension (ETH, SOL alongside BTC) if the single-asset architecture proves sound. These items live in `docs/planning/BACKLOG.md` for further notes.

---

## 6 · OMI Freeze and Thaw Protocol

### Freeze condition

Before the OMI start date (assumed: 2026-06-15), the project must be in one of two well-defined states.

**State A · Diagnostic passed.** Phase 5.4 complete (PRs 2-5 merged). Gate 1 and Gate 2 pass cleanly. Phase 5.5 is either complete (checkpoint exists) or queued as the first thaw task (diagnostic checkpoint is the freeze point). PRs 6 and 7 merged and Task Scheduler entries verified.

**State B · Diagnostic failed.** Phase 5.4 complete (PRs 2-5 merged). Gate 1 failed. ADR-006 written, documenting the decision between prior capacity restriction, KL warmup, or paradigm change. The project is paused with the architectural decision documented for thaw-time review. PRs 6 and 7 still merged (the ingestion and health check run regardless of model state).

Any state other than A or B is a freeze violation. If PRs 2-4 are not merged by the freeze date, merge what is ready, document the incomplete state in this roadmap, and accept that thaw will require re-loading more context.

### Background services during OMI

Two unattended services run on the Windows laptop during the cold period:

1. **Daily ingestion** (PR 6): `scripts/daily_ingest.py` runs via Task Scheduler, fetching new BTCUSDT 1m klines from Binance and appending to `data/market.duckdb`. Logs to `logs/daily_ingest.log`.

2. **Weekly health check** (PR 7): `scripts/weekly_health.py` runs via Task Scheduler, executing pytest, a 100-step model smoke, and a data freshness check. Logs to `logs/weekly_health.log`.

No model training runs during OMI. No model decisions are made. No code changes occur.

### Thaw protocol

On return from OMI, follow this sequence:

1. Read `docs/design/ARCHITECTURE.md` end-to-end (30 minutes). Read this roadmap. Read the latest entries in `docs/findings/` and `docs/implementations/`.
2. Review `logs/weekly_health.log` for the full OMI period. Scan for any test failures, smoke failures, or data freshness warnings. Review `logs/daily_ingest.log` for ingestion errors or gaps.
3. Check `data/market.duckdb` row count and latest timestamp to confirm 6 weeks of new kline data accumulated successfully.
4. Resume based on freeze state: if State A, start or continue Phase 5.5. If State B, work the ADR-006 decision (the architecture doc and the ADR contain enough context to re-engage without re-deriving the problem from scratch).

---

## 7 · How to Update This Document

When a PR lands, mark its entry in Section 1 with a checkmark and the merge date (e.g. "**PR 2 · ForwardDistributionHead** · done 2026-06-02"). When a phase completes, move its detail to a "Completed Phases" appendix at the bottom of this file and expand the next phase's detail in its place. The roadmap is a living document; do not let it go stale by more than one PR cycle. If the plan forks at the Gate Decision Point (Section 2), update the subsequent sections to reflect the chosen path and grey out or delete the unchosen branches.
