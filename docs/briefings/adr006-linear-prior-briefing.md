# Mega-Briefing · ADR-006 Linear-Prior Disambiguation Experiment

**Project:** crypto-dreamer · post-Phase-5.4 contingency (ADR-006)
**Run mode:** supervised · operator present at the machine and watching live
**Brief author:** design session, 2026-05-31
**Destination in repo:** `docs/briefings/adr006-linear-prior.md`

---

## 0 · Operating Protocol (read first)

The operator is present and watching the heartbeat live today. That changes the safety model from yesterday's unattended run: the operator is the real-time fallback. You still run the self-correcting loop and document rigorously, but the elaborate pivot-to-other-work fallback is replaced by a simpler rule · **if blocked, halt and report to the operator, who is watching.**

### Self-correcting loop

For each problem: attempt -> test -> on-fail document and investigate and re-hypothesize and re-implement and re-test. **Maximum 5 hypothesis-iterations per problem.** At the cap: stop, document all 5 attempts and your best understanding of why it is hard, and report to the operator. Do not grind past 5. Do not weaken the spec to force a pass.

### Hard rules (non-negotiable, same as prior runs)

- **No git commits, pushes, tags, merges, or PR/release creation.** Stay uncommitted. The operator reviews and commits.
- **Read the gate from the CHECKPOINT, never from W&B.** W&B logging silently failed mid-run on yesterday's diagnostic (`1rq8d8u5` synced only to step ~612 of 30000 while training completed correctly). That bug is unresolved. The authoritative gate read is the checkpoint-eval path (Section 5); the heartbeat is the authoritative "did it finish" signal. Do NOT trust the charts.
- **Never weaken the spec to make a result look better.** This is a disambiguation experiment · a "bad" result (KL does not release) is a VALID and useful outcome, not a failure to fix. Report what is true.
- **Single experimental variable.** Section 2 is the heart of this task: the ONLY thing that may change in the model is the prior's expressive capacity. Anything else changing confounds the experiment and ruins its interpretability.

### Context you must load first

Read, in parallel (fan-out subagents):
- `docs/design/ADR-006...` (in `ARCHITECTURE.md` Section 12) · the decision this experiment executes, including the three-outcome classification.
- `docs/findings/2026-05-30-phase5-4-diagnostic-results.md` · the falsification this responds to (Gate 1 fail kl 25.95, Gate 2 fail 9.76 vs 8.86 baseline, Gate 3 pass 0.478).
- `models/rssm.py` · the actual `prior_head` definition (Section 2 depends on reading this precisely · the architecture doc's description may be approximate).
- `configs/world_model.yaml` and `training/train_world_model.py` · the diagnostic config and entrypoint (unchanged from yesterday's successful launch).

---

## 1 · Mission

The Phase 5.4 forward-distribution pivot was falsified: the stochastic latent `z_t` collapsed (KL pinned at the free-bits floor) under BOTH a too-easy target (Phase 5.3 reconstruction) and a too-noisy target (Phase 5.4 forward returns). ADR-006 specifies a disambiguating experiment: restrict the prior network to a linear map and re-run the 30k diagnostic, to test whether an over-expressive prior is what collapses the latent (Hypothesis A) versus the data simply lacking exploitable regime structure (Hypothesis B).

Execute that experiment: make the prior-capacity edit, run the 30k diagnostic, read the gate from the checkpoint, and classify the result into one of ADR-006's three outcomes. Then HALT · do not start the downstream work any outcome implies (Section 6).

---

## 2 · The Single Experimental Variable (the heart of this task)

The experimental variable is the prior's **expressive capacity** · its ability to approximate complex nonlinear functions of `h_t`. The hidden layer plus its nonlinearity (GELU) is what provides that capacity. Removing them reduces it, which is the test.

### 2.1 · Read the actual `prior_head`

Read the current `prior_head` definition in `models/rssm.py`. The architecture doc describes it as a two-layer MLP (256 -> 256 -> 1024, GELU), but verify against the real code · do not assume. Note precisely: every layer, every activation, and critically, **any normalization** (LayerNorm, etc.), whether applied to `h_t` on the way in, between layers, or on the output.

### 2.2 · The norm/confound rule (READ CAREFULLY)

Normalization is NOT expressive capacity · it affects scale and optimization, not the function class the prior can represent. Therefore:

- **Remove** the hidden layer and its activation (GELU). This is the capacity reduction · the experimental variable.
- **Preserve** any normalization that exists. If the current `prior_head` applies a LayerNorm, keep it, so the ONLY thing that changes is expressive capacity.
- If the current `prior_head` has NO normalization, the replacement is a bare `nn.Linear(256, 1024)` (or `nn.Linear(hidden_dim, n_latents * n_classes)` in the code's actual symbols).
- If it DOES have normalization, keep the input-side norm and drop only the hidden transformation.

This is the single most important judgment in the experiment. A naive "replace with one Linear" that accidentally also removes a LayerNorm would change TWO things (capacity AND normalization), and then a KL change could not be attributed to capacity · the disambiguation would be confounded and the run wasted. ADR-006's literal wording said "bare linear, no hidden layer, no activation"; this brief refines that to "remove capacity, preserve normalization" precisely to keep the experiment single-variable. Note this refinement in the findings doc.

### 2.3 · Show the diff, then proceed

Before running anything, output: the current `prior_head` definition, your proposed replacement, the diff between them, and an explicit one-paragraph statement of what changed and what was preserved · in particular, name any normalization that existed and what you did with it. Then PROCEED automatically to the precondition gate (Section 3). Do NOT block waiting for a chat confirmation · the operator is watching live and will intervene if the diff is wrong. The diff being visible in your output is the checkpoint; the operator's live presence is the safety.

### 2.4 · Verify single-variable integrity

After editing, confirm nothing else changed: the posterior head, encoder, RSSM recurrence, heads, loss composition, free-bits floor (1.0), coefficients (coef_dyn 0.5, coef_rep 0.1), and all training hyperparameters (T=48, batch 32, lr 1e-4, 30k steps) are identical to yesterday's run. The forward-distribution head and its target are unchanged · this experiment changes ONLY the prior's capacity, on top of the exact Phase 5.4 setup. If the posterior head shares code or structure with the prior head such that editing the prior affects the posterior, STOP and report · that coupling would itself confound the experiment.

**Files you may modify:** `models/rssm.py` (prior_head only), and optionally promote the gate-eval script (Section 5). Plus the findings doc (Section 7).
**Files you must NOT modify:** `configs/world_model.yaml` (the diagnostic config is unchanged · do not alter hyperparameters), `models/heads.py`, `models/encoder.py`, `models/world_model.py`, `training/datamodule.py`, `envs/spot_btc.py`, `data/ingest.py`. If you believe a config change is needed, STOP and report rather than editing.

---

## 3 · Precondition Gate (before the full 30k · same as the prior diagnostic)

All three must pass. If any fails, do NOT launch the 30k; report to the operator.

1. **powercfg + AC.** Apply `powercfg /change standby-timeout-ac 0` and `monitor-timeout-ac 0`; verify both read 0 via `powercfg /query SCHEME_CURRENT`; confirm AC via `Win32_Battery.BatteryStatus = 2`.
2. **CUDA.** `torch.cuda.is_available()` True, `torch.version.cuda` 12.4, torch 2.6.0+cu124.
3. **1000-step pre-flight.** Run the diagnostic config for 1000 steps as a NaN/divergence/throughput check. Require: losses finite throughout, no NaN/Inf, `kl_unclipped` finite, throughput ~400-550 ms/step (no 3x stall). A NOTE specific to this experiment: do NOT interpret the pre-flight's `kl_unclipped` as a gate signal · at 1000 steps (before LR warmup even completes) it tells you nothing about release. It only confirms the run is healthy enough to commit 3 hours to.

---

## 4 · Launch the 30k

Only after all three preconditions pass. Launch the full 30k diagnostic with the same config as yesterday EXCEPT the prior is now linear. Pick a **descriptive W&B run name that distinguishes it from yesterday's falsified run** (`1rq8d8u5` / `phase5.4-diag-30k`) · e.g. `phase5.4-linearprior-30k` or similar of your choosing. Heartbeat to `logs/heartbeat_*.log`. Capture and record the new run id and name. The operator is watching, so the run may stay attached or detached at your discretion · either way, confirm it advances healthily past ~1-2k steps before considering the launch successful.

---

## 5 · Read the Gate from the Checkpoint (NOT W&B)

When the 30k completes (heartbeat shows `TRAIN_END step=30000`), read the gate from the final checkpoint · NOT from W&B, which cannot be trusted (see Section 0).

The working method exists as `_gate.py` in the repo root from yesterday: it loads the final checkpoint into a `WorldModel`, builds the val dataloader, and runs `m._step(batch, stage="val", collect_trace=True)` over ~40 val batches, averaging the `info` dict components (`kl_unclipped`, `loss_forward`, `loss_forward_per_horizon`, `loss_reward`). Reuse that method. Optionally (it is on the backlog) promote it to `scripts/eval_gates.py` parameterized by checkpoint path · a clean reusable gate-reader is worth having for this and future runs. If you promote it, verify it reproduces yesterday's numbers on yesterday's checkpoint as a sanity check before trusting it on the new one.

Report, measured on the new checkpoint over ~40 val batches:
- `kl_unclipped` vs the 32-nat floor (Gate 1)
- per-horizon `loss_forward` and their sum vs the 8.8632 baseline (Gate 2)
- `loss_reward` vs ~0.48 (Gate 3 · integrity check; should stay ~0.48 since only the prior changed)

---

## 6 · Classify the Outcome · then HALT

Classify the result into exactly one of ADR-006's three outcomes:

1. **KL releases (kl_unclipped > 32) AND Gate 2 improves below 8.85** · Hypothesis A confirmed, the prior was the problem, latent is alive and useful.
2. **KL releases BUT Gate 2 still fails (forward sum at or above baseline)** · latent now carries information but the target is wrong; points to volatility-target redesign (ADR-006 option d).
3. **KL still will not release even with a linear prior** · strongest evidence the data lacks regime structure; points to model-free paradigm (ADR-006 option c).

State which outcome obtained, with the measured numbers as evidence. Note where `kl_unclipped` ended relative to its starting ~26 and whether it released, decayed, or stayed flat · the trajectory shape, read from the heartbeat (total loss) and the final checkpoint.

**Then HALT.** Do NOT start the downstream work any outcome implies. Outcome 1's "continue the pivot," outcome 2's "redesign the target to volatility," and outcome 3's "build a model-free agent" are all major architectural decisions reserved for the operator and the design partner · exactly the kind of fork that must not be taken autonomously, the same discipline as halting at the gate yesterday. Your job ends at "here is the outcome classification with evidence."

---

## 7 · Documentation

Write `docs/findings/2026-05-31-adr006-linear-prior-results.md`: the prior-head diff (before/after, and the norm-preservation decision from Section 2.2), the precondition-gate results, the new run id/name, the measured gate metrics from the checkpoint, the outcome classification (1/2/3) with evidence, and a one-paragraph "what this means for the next decision" that maps the outcome to the ADR-006 branch WITHOUT starting that branch. Also record the full self-correcting-loop iteration history for anything that triggered it.

If you promoted `scripts/eval_gates.py`, note it and its sanity-check-against-yesterday result.

---

## 8 · Git Boundaries

No commits, pushes, tags, merges, gh operations, or `.gitignore` edits. No file deletion. Stay uncommitted · the operator reviews the diff and the result and commits. Respect the file scope in Section 2.4.

---

## 9 · Style Conventions

Middle dot (`·`, U+00B7) separators · never em/en dash. ASCII only. File paths in backticks. Type hints and shape-documented docstrings on any new code. Match existing `rssm.py` style for the edit. No `print()` in production code (the eval script may print, that is its purpose). `ruff check` and `ruff format --check` clean on any modified Python file.

---

## 10 · Final Report

1. The `prior_head` diff (before/after) and the explicit single-variable / norm-preservation statement.
2. Single-variable integrity confirmation (what was held fixed).
3. Precondition-gate results (powercfg, CUDA, 1000-step pre-flight).
4. New run id and name; confirmation it advanced healthily.
5. Gate metrics measured FROM THE CHECKPOINT: kl_unclipped, per-horizon forward sum, loss_reward.
6. Outcome classification (1, 2, or 3) with the measured numbers as evidence and the KL trajectory shape.
7. Explicit statement: downstream branch NOT started · reserved for operator decision.
8. Any loop iteration history; the findings doc path. Confirm nothing committed.
