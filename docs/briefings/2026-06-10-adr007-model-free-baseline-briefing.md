# crypto-dreamer · operating briefing · ADR-007 and the model-free baseline
Date: 2026-06-10 · Audience: Claude Code agent with full repo access, no prior memory
Machine: Windows · PowerShell · RTX 4070 laptop GPU · Python 3.11

## 0. Role and how to use this document
You are the implementation agent. The operator is the reviewer, decision-maker, and
the ONLY party who runs git operations. This briefing is self-contained: it embeds
the established facts, the ratified decision, the project invariants, four phases
of work, and standing hard rules. Read it fully before acting. Architectural
decisions stated here are already made — you execute them, you do not re-decide,
re-derive, or relitigate them. If the repository ever contradicts this briefing,
HALT and report the discrepancy instead of proceeding or silently reconciling.

## 1. Ground truth (established by two independent read-only audits, 2026-06-10)
Treat everything in this section as fact. Do not re-run the audit.

- The ADR-006 linear-prior disambiguation experiment RAN, COMPLETED, and was
  COMMITTED in `a3941d3` (2026-05-31). It FALSIFIED Hypothesis A.
- The applied change: `prior_head` in `models/rssm.py` is now a bare
  `nn.Linear(256, 1024)`. The original was
  `Linear(256,256) -> GELU -> Linear(256,1024)` with no normalization anywhere, so
  the only removed components were the hidden layer and GELU — the single intended
  experimental variable. Parameter count dropped by exactly 65,792.
  `posterior_head` is untouched.
- Evidence checkpoint: `checkpoints/world_model_diagnostic_step=30000-v2.ckpt`
  (global_step=30000, clean linear-prior state_dict, 0 missing / 0 unexpected keys).
- Gates, re-derived from the CHECKPOINT via `scripts/eval_gates.py`
  (40 val batches, seed 42) — NOT from W&B:
  - Gate 1: kl_unclipped = 26.31 vs 32 floor -> FAIL (no KL release; +0.36 vs the
    MLP prior's 25.95)
  - Gate 2: forward loss sum = 9.6142 vs 8.8632 baseline -> FAIL
    (h1 2.2152 / h5 2.3494 / h15 2.4943 / h30 2.5552)
  - Gate 3: reward NLL = 0.4776 ~= 0.478 -> PASS (wiring and reward head intact)
  - loss_dyn/rep = 32.0055 (free-bits floor-clipped; the latent is pinned)
  - Seed-stable across seeds 42/0/123 (kl 26.31 +/- 0.002). A 3-agent adversarial
    review found no valid refutation.
- W&B run `jnkypsrt` silently desynced at ~step 612 and is UNTRUSTED. This is the
  second consecutive run where W&B desynced. W&B is never a gate source.
- Classification: ADR-006 Outcome 3. Handicapping the prior to a bare affine map
  did not release KL. This falsifies Hypothesis A (over-expressive prior) and
  supports Hypothesis B: BTC 1-min data lacks regime-level stochastic structure
  that a latent-variable world model can exploit in this setup.

## 2. The locked decision (operator-ratified — non-negotiable)
Per the pre-registered ADR-006 decision tree, Outcome 3 selects option (c):

- DROP the world-model / latent paradigm for this line of work.
- STAND UP a model-free RL baseline — PPO preferred, SAC-discrete acceptable —
  trained on the SAME observations and the SAME reward, evaluated on the SAME
  held-out windows, as the honest apples-to-apples comparison.
- Phase 5.5 (world-model scaling) is BLOCKED.
- Option (d) — keep the RSSM and redesign the decoder target to conditional
  volatility/scale — was considered and DEFERRED. It is recorded in ADR-007 as the
  rejected alternative with rationale. Do NOT implement it.
- The c-vs-d fork was the operator's call, ratified at the ADR-007 review gate.
  No agent re-opens it.
- The RSSM and the step-30000 checkpoint are frozen historical evidence. Do not
  retrain, modify, fine-tune, or extend them.

## 3. Project invariants (reuse as-is; deviation invalidates the comparison)
- Environment: the existing Gymnasium env in `envs/`. Discrete 5-action target
  allocation {0, 25, 50, 75, 100}% of equity. 0.1% taker fee plus linear slippage.
  Reward = log-return - 0.05 x turnover. T=48 windows. The Windows WDDM
  kernel-launch tax is the documented reason for T=48; keep it.
- Data and features: DuckDB BTCUSDT 1-min klines (~1.05M rows) consumed through
  the existing iTransformer feature pipeline. The model-free agent must see the
  SAME observations and be evaluated on the SAME held-out windows used by the
  world-model diagnostics. No new features, no changed normalization, no changed
  split boundaries.
- Stack: Python 3.11. PyTorch CUDA build torch 2.6.0+cu124 pinned via uv sources
  with the Windows platform marker. After ANY dependency change, verify that
  `uv sync` has not replaced the CUDA build with CPU torch
  (`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`).
  Lightning where it fits; a plain training loop is acceptable for PPO. W&B for
  tracking only, offline mode, never as a gate source.

## 4. Phase 0 — re-ground (read-only)
Purpose: load the project into your working context and verify the repo matches
this briefing. No file writes, no training, no code changes in this phase.

Read, in this order, extracting the listed items:
1. `ARCHITECTURE.md` Section 12 — the ADR ledger. Extract: the ADR-005 tombstone,
   the full ADR-006 entry including its pre-registered outcome tree, and the
   placement/format convention ADR-007 must follow.
2. `docs/findings/2026-05-31-adr006-linear-prior-results.md` — extract the gate
   numbers, seeds, and Outcome-3 classification; confirm they match Section 1
   of this briefing.
3. The env in `envs/` — extract: action mapping, fee and slippage implementation,
   reward formula, T=48 windowing, and how episodes/windows are constructed.
4. The data/feature pipeline — extract: DuckDB source, iTransformer feature set,
   and exactly where the train/val/held-out window boundaries are defined.
5. `scripts/eval_gates.py` — extract: how gates are computed from a checkpoint,
   seeding, batch counts, and output artifact format. Your Phase 2 harness will
   mirror these conventions.

Deliverable: a re-grounding statement, in your own words, covering (a) what
ADR-006 tested and what it found, (b) why option (c) follows from Outcome 3,
(c) the env/data invariants, (d) where the held-out windows are defined, and
(e) any ambiguity or contradiction you found.

HALT conditions for Phase 0: any file above is missing; any number or fact in the
repo contradicts Section 1; the checkpoint file is absent. On halt, report and wait.

STOP at end of Phase 0. Wait for explicit operator confirmation.

## 5. Phase 1 — write ADR-007, then STOP
No training, no implementation, no dependency changes in this phase. One
deliverable: the ADR-007 entry, in the same location and format as the existing
ADR ledger (per the convention you confirmed in Phase 0).

ADR-007 must contain, as distinct sections:
1. Status and date — Proposed (the operator flips it to Accepted on commit).
2. Context — the ADR-006 result: Hypothesis A falsified, Outcome-3 classification,
   citing commit `a3941d3`, the checkpoint path, and the gate numbers from
   Section 1. State Hypothesis B as the supported interpretation.
3. Decision — option (c): model-free baseline (PPO preferred, SAC-discrete
   acceptable) on the unchanged env, observations, reward, and held-out windows.
   Mark it operator-ratified.
4. Rejected alternative — option (d), recorded as considered-and-DEFERRED (not
   dead), with the rationale for deferring it.
5. Consequences — Phase 5.5 blocked; RSSM and step-30000 checkpoint frozen as
   evidence; the baseline becomes the new reference point.
6. PRE-REGISTERED evaluation gate — the core of this ADR. Propose specific HARD
   numeric thresholds for operator approval. Required shape:
   - Metrics: net-of-fees out-of-sample cumulative log-return AND Sharpe.
   - Comparators: (i) buy-and-hold, (ii) a zero-turnover flat policy. A seeded
     random-action policy is run as a sanity reference but is not a gate.
   - Seeds: at least 3 training seeds; state the aggregation rule explicitly
     (e.g. the median seed must clear every threshold).
   - Turnover: the 0.05 x turnover penalty is already inside the reward; report
     realized turnover separately and include an explicit cap or adjustment so the
     agent cannot "win" by overtrading.
   - Data: the exact held-out windows identified in Phase 0, named explicitly.
   - PASS/FAIL: defined explicitly and exhaustively. Every criterion is binary.
     No partial pass, no judgment calls left for Phase 3 — classification there
     must be purely mechanical.

STOP at end of Phase 1. Present the ADR-007 draft. The operator reviews, possibly
amends thresholds, and commits. Do not begin Phase 2 without explicit approval.

## 6. Phase 2 — implement the baseline (gated on ADR-007 approval)
This phase parallelizes into four tracks. Use fan-out subagents for the tracks
below, then integrate.

Track A — agent implementation:
- PPO on the discrete 5-action space, consuming the existing feature pipeline
  output unchanged. Fall back to SAC-discrete only if PPO is blocked for a
  documented reason, and ask the operator first.
- Modest network sizing appropriate to a 4070 laptop GPU. Deterministic seeding
  throughout. Hyperparameters in a config file, not hardcoded.
- Logging: W&B offline plus local CSV/JSON metrics on disk. Checkpoints to
  `checkpoints/` with seed and step in the filename.

Track B — env-reuse verification:
- A short written verification confirming the env is consumed as-is: action
  mapping, 0.1% fee, linear slippage, reward formula, T=48 windowing.
- An assertion script/test that checks observation shapes, dtypes, and feature
  ordering match what the world-model diagnostics consumed, and that the held-out
  window indices are identical. This is the apples-to-apples guarantee; it must
  pass before any full run.

Track C — evaluation/gates harness:
- A new script (e.g. `scripts/eval_baseline_gates.py`) mirroring the conventions
  of `scripts/eval_gates.py`: loads a policy checkpoint from disk, runs a
  deterministic seeded rollout on the held-out windows, computes net-of-fees
  cumulative log-return, Sharpe, realized turnover, and max drawdown
  (report-only), and writes a JSON + markdown artifact to disk.
- By construction, gate classification reads ONLY these on-disk artifacts.

Track D — comparator policies:
- Buy-and-hold (move to 100% at the first step, hold), flat (0% always), and
  seeded random-action. All three run through the SAME env and SAME harness on
  the SAME held-out windows, results saved as artifacts alongside the agent's.

Integration, after the tracks land:
- Smoke test: ~100 training steps, confirm finite losses, then run the full eval
  path end-to-end on a small window subset and confirm artifacts are written.
- Full run: train per the approved ADR-007 plan across the pre-registered seeds.
  Heartbeat callback on (the heartbeat is the "did it finish" signal); powercfg
  awake mitigations applied; W&B offline.

STOP at end of Phase 2. Report: training completed, seeds run, checkpoint and
artifact paths, any anomalies. Do not evaluate gates yet. Wait for confirmation.

## 7. Phase 3 — evaluate, document, STOP
- Run the Track C harness on the final checkpoint of every seed. Read all numbers
  from the on-disk artifacts — NEVER from W&B.
- Classify each pre-registered ADR-007 criterion PASS or FAIL exactly as written.
  No reinterpretation, no threshold adjustment, no "close enough".
- Write a findings doc to
  `docs/findings/2026-06-XX-adr007-model-free-baseline-results.md` in the same
  register as the ADR-006 findings doc, containing: what ran (config, seeds,
  steps, wall-clock); exact per-seed and aggregate numbers; comparator numbers;
  gate-by-gate classification with the pre-registered threshold beside each
  result; artifact and checkpoint paths; anomalies observed. Observations are
  fine; recommendations for a next branch are not — the next branch is the
  operator's call.
- STOP regardless of outcome. Do not start any follow-on work.

## 8. Standing hard rules (apply to every phase)
- NO git commits, pushes, tags, merges, PRs, or releases. Leave the working tree
  uncommitted; the operator reviews and commits.
- Gates are read from checkpoints / on-disk eval artifacts, never from W&B. W&B's
  only trusted role is the heartbeat "did it finish" signal.
- Gates are HARD thresholds, not soft targets. Never weaken, reinterpret, or
  post-hoc adjust a spec to force a pass.
- Self-correcting loop per problem: attempt -> test -> on failure, document the
  failure, re-hypothesize, re-implement, re-test. MAXIMUM 5 hypothesis-iterations
  per problem, then halt and report. Do not grind past 5.
- Every phase ends in a STOP that requires explicit operator confirmation.
- Locked decisions (option c, the invariants in Section 3, and the ADR-007
  thresholds once approved) are not relitigated without asking the operator.
- Reproducibility: pinned seeds, deterministic eval, powercfg awake mitigations,
  heartbeat callback, W&B offline.
- Docs style: ASCII-only, middle-dot separators, tight technical prose. Findings
  to `docs/findings/`, briefings to `docs/briefings/`.
