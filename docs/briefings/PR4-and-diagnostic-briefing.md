# Mega-Briefing · PR 4 Wiring + 30k Diagnostic Launch + PR 6/7 Fallback

**Project:** crypto-dreamer · Phase 5.4 forward-distribution pivot
**Run mode:** semi-autonomous (Opus 4.8 ultra-code), operator unavailable (finals), laptop kept awake and on AC all day
**Brief author:** design session, 2026-05-29
**Destination in repo:** `docs/briefings/PR4-and-diagnostic.md`

---

## 0 · Operating Protocol (read first · governs everything)

You run today largely unattended. The operator is studying for finals and may not respond for hours. Every instruction below is written so that the SAFE outcome happens with zero intervention. Do not assume the operator can answer questions, approve actions, or unblock you mid-run. If you would normally pause to ask, instead follow the documented fallback and leave a clear record.

### The self-correcting loop

For each open question and each failure: attempt → test → on-fail document and investigate and re-hypothesize and plan and re-implement and re-test. **Maximum 5 hypothesis-iterations per distinct problem.**

### What happens at the 5-iteration cap (CRITICAL · differs from prior briefs)

When a problem in PR 4 (Phase A below) cannot be resolved in 5 principled iterations, you do NOT keep grinding and you do NOT weaken the spec to force a pass. You:

1. Document all 5 attempts, their failure modes, and your best understanding of why the problem is hard, in the test-results doc.
2. HALT Phase A (PR 4). Do not start the 30k diagnostic · a blocked PR 4 means the wiring is not trustworthy and a diagnostic on untrustworthy wiring is worthless.
3. PIVOT to Phase C (PR 6 and PR 7). These are independent operational tasks that require no model judgment and are on the pre-OMI critical path. Completing them is genuine forward progress and is SAFE to do unattended.
4. Leave the blocked PR 4 fully documented for the operator to resume tonight.

This is the single most important rule in this brief. An unattended agent that loops without bound tends to "fix" errors by quietly degrading the specification · widening tolerances, clamping values that should not be clamped, disabling checks, or altering a loss so a run merely looks healthy. That failure mode has occurred twice this week in minutes. The cap plus pivot-to-safe-work is the guardrail. Respect it absolutely.

### Hard constraints

- **Never weaken the spec to make a test pass.** If a test asserts something this brief did not ask for, that is a finding to document, not a silent edit.
- **Never resolve a correctness-of-meaning question by picking whatever turns a test green.** Locked decisions (Sections 3.2, 4) are not open questions.
- **No git commits, pushes, tags, merges, or PR/release creation at any point.** Stay uncommitted through everything. The operator reviews the full arc tonight and commits.
- **Do not depend on any remote-control or live-monitoring feature.** The run must be correct and safe with zero operator intervention until tonight.

---

## 1 · Today's Mission and Decision Tree

The day has three phases. You always start at Phase A. Where you go next depends on outcomes.

```
PHASE A · PR 4 (wire head, sever decoder loss, smoke, alignment trace)
   |
   |-- PR 4 lands clean (smoke finite, alignment trace green, 58+ tests pass)
   |        |
   |        v
   |   PHASE B · 30k diagnostic launch
   |        |
   |        |-- HARD PRECONDITION GATE passes (powercfg + CUDA + 1000-step pre-flight clean)
   |        |        -> launch full 30k, capture run id, write findings skeleton, HALT AT GATE
   |        |           (do NOT evaluate Gate 1/2/3 · that fork is the operator's tonight)
   |        |
   |        +-- HARD PRECONDITION GATE fails (powercfg wrong, no CUDA, or pre-flight NaN/diverge)
   |                 -> do NOT start the 30k. Document why. PIVOT to PHASE C.
   |
   +-- PR 4 blocks (a problem unsolved after 5 principled iterations)
            -> document the full 5-attempt arc, HALT PR 4, PIVOT to PHASE C.

PHASE C · Fallback: PR 6 (daily ingest) + PR 7 (weekly health check)
   Independent, safe, no model judgment. Complete both if reached.
```

Tonight the operator returns to one of three good states: (1) PR 4 done + 30k running or finished, gate un-evaluated and waiting for joint review; (2) PR 4 done but precondition gate failed, so PR 6/7 shipped instead and the 30k is teed up for a watched start; (3) PR 4 blocked with a full diagnosis, and PR 6/7 shipped while waiting. All three are productive. None requires you to have made the project's forking decision alone.

---

## 2 · Project State

PR 2 (`ForwardDistributionHead` in `models/heads.py`) and PR 3 (`forward_returns` + `forward_valid` tensors in `training/datamodule.py`) are shipped and committed. The environment is hardened (CUDA torch 2.6.0+cu124 pinned via uv source; `uv sync --extra dev` is idempotent). 58 tests pass.

Authoritative references · read before writing:
- `docs/design/ARCHITECTURE.md` · Sections 5 (RSSM), 6 (forward head), 9 (loss), 10 (training), 11 (gates), 12 (ADRs). This is the spec.
- `docs/implementations/2026-05-28-phase5-4-pr3-forward-returns.md` · PR 3's resolved indexing scheme. The forward-return anchor at trajectory step j is `close[k]` where `k = ep.kline_idx[start + j]`, which is the bar immediately following the agent's action (one past the obs window end). PR 4 must preserve this alignment.
- `docs/planning/ROADMAP.md` · PR 4 entry, including the step-alignment-trace verification note added at the end of PR 3.

---

## 3 · PHASE A · PR 4 · Wire the Head

### 3.1 · Goal and file scope

Wire `ForwardDistributionHead` into the world model, sever the old feature-reconstruction decoder from the loss, update config, prove correctness with a 100-step smoke and an end-to-end step-alignment trace.

**Files you MAY modify in Phase A:** `models/world_model.py`, `models/heads.py` (decoder removal only), `configs/world_model.yaml`, `tests/test_world_model_forward_wiring.py` (new), plus Phase A documentation.

**Files you must NOT modify in Phase A:** `training/datamodule.py` (PR 3, done · read only), `data/ingest.py`, `envs/spot_btc.py`, `models/rssm.py`, `models/encoder.py`. Do not retrain or alter the encoder. Do not touch the MAE checkpoint.

### 3.2 · Decoder removal (agent's choice on method · loss severance is mandatory)

The old `DecoderHead` (`models/heads.py:23-39`) and its loss machinery in `world_model.py` (`dec_target` at line 136, `loss_dec_sum` at lines 169-171, `DecoderHead` instantiation at line 99) must be removed from the active training path. You may either delete the code outright or comment/feature-flag it · your judgment based on what you find in the code structure. Git holds the old version either way, so deletion is recoverable.

**Mandatory regardless of method:** the decoder loss must be genuinely severed from the total loss and from backprop · not zeroed-while-still-computed. The decoder forward pass must not run, `dec_target` must not be computed, and no decoder term may enter the autograd graph. A half-removed decoder that still computes its forward pass wastes compute and can leave dead nodes that confuse the diagnostic. Verify by confirming the total-loss expression in `world_model.py` (currently around lines 204-215) contains `L_forward + L_reward + L_continue + coef_dyn*L_dyn + coef_rep*L_rep` and NO decoder term.

### 3.3 · Head wiring in `world_model.py::_step`

Instantiate `ForwardDistributionHead` with the horizons, bin count, and ranges from config (Section 3.4). In the per-step loop, the head consumes `feat = [h_t, z_t]` (the same concatenation the reward and continue heads use · confirm the exact feat construction in the existing code, currently the heads take the 1280-dim state). Compute `L_forward` via the head's `loss(logits, targets)` method where `targets = forward_returns[:, t]` for the matching trajectory step t.

**Apply the validity mask.** PR 3 emits `forward_valid: (B, T, 4)`. The forward loss must be masked: positions where `forward_valid` is False must contribute zero to the loss and zero gradient. The head's `loss` method may not know about masking, so apply the mask at the wiring level · either by masking the per-(step, horizon) cross-entropy before reduction, or by using `per_horizon_loss` and zeroing masked contributions. Whichever you choose, the masked positions (which hold 0.0 placeholders in `forward_returns`) must not inject a spurious "zero return" training signal. This is correctness-critical: the placeholder is not a real target.

**Per-horizon logging.** Log `loss_forward_1`, `loss_forward_5`, `loss_forward_15`, `loss_forward_30` separately to W&B (using the head's `per_horizon_loss`, mask-aware), per ADR-001, so the post-diagnostic weighting decision has data.

### 3.4 · Config update

Add to `configs/world_model.yaml`:
```
forward_horizons: [1, 5, 15, 30]
forward_bins: 41
forward_ranges: [0.005, 0.010, 0.018, 0.025]
```
Remove any decoder-specific config keys if present. Do not change unrelated hyperparameters (T, batch_size, free_bits, coef_dyn, coef_rep, lr, etc.) · those are locked from prior phases.

### 3.5 · The step-alignment trace (LOAD-BEARING · brief 3.3 carryover)

This is the most important test in PR 4. An off-by-one in how `feat` at step t pairs with `forward_returns[:, t]` is a silent temporal lag or look-ahead leak that a finite-loss smoke test will NOT catch.

Add a test (`test_world_model_forward_wiring.py`) that traces ONE concrete trajectory step end to end and asserts every link references the same kline index `k`:

1. Pick a sampled trajectory and a step t.
2. Identify the kline index `k = ep.kline_idx[start + t]` for that step (PR 3's anchor).
3. Confirm the observation window the RSSM ingested for step t corresponds to `feature_cache[k - 256 : k]` (the window ending just before k, per `envs/spot_btc.py` slicing).
4. Confirm the `feat = [h_t, z_t]` the forward head consumes at step t is the belief state produced after ingesting that step's observation.
5. Confirm `forward_returns[:, t]` is the target anchored on `close[k]` (PR 3 guarantees this).
6. Assert: the belief state at step t and the forward-return anchor both correspond to the SAME k. If the wiring pairs `feat` at step t with `forward_returns` at t-1 or t+1, this test must fail.

If you cannot construct this trace because the alignment is genuinely ambiguous in the code, treat it as a Phase A blocker (5-iteration cap, then pivot to Phase C). Do NOT paper over it · this is exactly the kind of correctness-of-meaning question that must not be resolved by making a weaker test pass.

### 3.6 · 100-step smoke

Run a 100-step smoke (`mode` configured for smoke / `max_episodes=5` per existing convention). Verify: all loss components finite (no NaN/Inf), all six logged losses present (forward_1/5/15/30, reward, continue), KL components finite, masked forward positions contributing zero. 100 steps is too few for learning · losses need not decrease, only stay finite and well-formed.

### 3.7 · PR 4 test suite

In `tests/test_world_model_forward_wiring.py`, beyond the alignment trace:
- Forward loss is finite and a scalar; gradients flow into the head and into the RSSM/encoder through `feat`.
- Masked positions contribute exactly zero to loss and gradient (construct a synthetic batch with known-invalid positions; assert masking).
- The total loss expression contains no decoder term (assert the decoder is gone from backprop · e.g. confirm no decoder parameters receive gradient, or that the decoder is not instantiated).
- Per-horizon losses sum-consistency with the masked total.
- Existing suite still green: `pytest tests/ -x -q --ignore=tests/test_dream_endpoint.py` (expect more than 58 with the new tests).
- `ruff check` and `ruff format --check` clean on modified files (CI runs format check).

### 3.8 · PR 4 done criteria

PR 4 is clean and you proceed to Phase B only if ALL hold: smoke finite and well-formed, alignment trace green, masking verified, decoder severed from backprop, full suite green, ruff clean. If any of these cannot be achieved in 5 principled iterations on the blocking issue, HALT and pivot to Phase C.

---

## 4 · PHASE B · 30k Diagnostic Launch (only if PR 4 clean)

### 4.1 · HARD PRECONDITION GATE (all three must pass before launching the full 30k)

This gate is non-negotiable. Phase 5.3 burned a full day on a run that stalled at step 7k due to Modern Standby. The gate prevents repeating that on a finals day.

**Precondition 1 · powercfg.** Apply and then verify the Modern Standby mitigations:
```
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
```
Then confirm via `powercfg /query SCHEME_CURRENT` (or equivalent) that both timeouts read 0. Also confirm AC power: `Get-WmiObject Win32_Battery` should show `BatteryStatus = 2` (on AC). If AC is not confirmed or timeouts are not 0, do NOT launch · document and pivot to Phase C.

**Precondition 2 · CUDA.** Verify:
```
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```
Must print `True 12.4`. If CUDA is unavailable, do NOT launch (CPU training will not finish) · document and pivot to Phase C.

**Precondition 3 · 1000-step pre-flight.** Run the diagnostic config for exactly 1000 steps as a NaN/divergence check before committing to the full ~3-5h. Watch: losses finite throughout, no NaN/Inf, KL components behaving (kl_unclipped a finite positive number, not exploding or zero), forward losses finite and not exploding, throughput in a sane range (~400-550 ms/step per prior benchmarks · if it is 3x slower like the Phase 5.3 stall, something is wrong). If the pre-flight NaNs, diverges, or stalls, do NOT launch the full run · document and pivot to Phase C. Save the pre-flight as a short report.

### 4.2 · Launch the full 30k

Only after all three preconditions pass: launch the full 30k diagnostic (`mode: diagnostic`, `max_steps: 30000`, W&B online, heartbeat callback to `logs/heartbeat_phase5_4_diag.log`). Launch it so it runs to completion independently of your agent session (the run takes hours; your session should not need to babysit it). Confirm the run has started and advanced past its first checkpoint healthily (re-check heartbeat and W&B that it is progressing past ~1000-2000 steps with finite losses). Capture and record the W&B run id and run name.

### 4.3 · HALT AT GATE · do NOT evaluate Gate 1/2/3

This is a locked decision. You launch the diagnostic and confirm it is running healthily. You do NOT evaluate Gate 1 (KL release), Gate 2 (forward loss vs 8.85 baseline), or Gate 3 (reward NLL) · even if the run finishes while you are still active. Gate evaluation determines whether the project advances to Phase 5.5 or enters the ADR-006 contingency (prior restriction vs KL warmup vs paradigm change), and that fork is reserved for the operator and the design partner together tonight. Your job ends at "the diagnostic is running and healthy" or "the diagnostic finished and the raw numbers are recorded for review." Do not interpret, do not decide, do not start Phase 5.5, do not write ADR-006.

### 4.4 · Findings skeleton

Write `docs/findings/2026-05-29-phase5-4-diagnostic-skeleton.md` containing: the W&B run id and name, the precondition-gate results (powercfg state, CUDA confirmation, 1000-step pre-flight summary), the launch timestamp, and a clearly-marked EMPTY section titled "Gate Evaluation · TO BE COMPLETED BY OPERATOR" listing the three gates and their thresholds (Gate 1: kl_unclipped > 32 nat by step 20k; Gate 2: val/loss_forward_dist < 8.85 at step 20k, with the [8.85, 8.86] inconclusive band; Gate 3: reward NLL ~0.48) but with NO conclusions filled in. The operator fills this in tonight.

---

## 5 · PHASE C · Fallback · PR 6 and PR 7 (only if PR 4 blocks or precondition gate fails)

If you reach Phase C, complete both PRs. They are independent of the model wiring and safe to do unattended.

**Files you MAY modify/create in Phase C:** `scripts/daily_ingest.py` (new), `scripts/weekly_health.py` (new), Task Scheduler XML files, `docs/operations/daily-ingest-setup.md` (new), `docs/operations/weekly-health-setup.md` (new). You MAY read `data/ingest.py` to reuse its Binance/DuckDB logic. Do NOT modify `data/ingest.py` itself, and do NOT touch model/config files.

### 5.1 · PR 6 · Daily ingestion service

Create `scripts/daily_ingest.py`: connect to Binance public API, fetch klines from the last-ingested timestamp in `data/market.duckdb` to now, append (idempotent · INSERT OR IGNORE or equivalent), log to `logs/daily_ingest.log`. Reuse `data/ingest.py`'s fetch/insert logic rather than reimplementing. Create a Windows Task Scheduler XML for daily execution (e.g. 06:00 local). Write `docs/operations/daily-ingest-setup.md` with install steps. Verification: run manually, confirm new rows for the current date appear, confirm re-running is idempotent.

### 5.2 · PR 7 · Weekly health check service

Create `scripts/weekly_health.py`: runs `pytest tests/ -x -q --ignore=tests/test_dream_endpoint.py`, runs a 100-step model smoke, checks `data/market.duckdb` freshness (last kline within 48h), logs to `logs/weekly_health.log` with timestamps, exits 0 on all-pass and nonzero on any failure. Create a Task Scheduler XML for weekly execution (e.g. Sunday 03:00 local). Write `docs/operations/weekly-health-setup.md`. Verification: run manually, confirm a clean log entry with pytest result, smoke status, and freshness check.

**Note on PR 7 and the pivot:** if you reached Phase C because PR 4 blocked, the model code is in a partially-modified state. PR 7's smoke test must run against whatever state the model is in. If the smoke fails because of PR 4's incomplete wiring, that is expected · note it in the health-check log and make the script robust to it (the script should report the failure, not crash). Do not attempt to fix PR 4 from within Phase C · PR 4's blocker is documented and reserved for the operator.

---

## 6 · Documentation Requirements

Produce the documents relevant to the path you took.

- **Always:** a test-results record (`docs/findings/2026-05-29-pr4-test-results.md`) with every test outcome and, for any problem that triggered the loop, the full iteration history up to resolution or the 5-cap. The loop history is the highest-value artifact for the operator · do not summarize it away.
- **If PR 4 landed:** an implementation doc (`docs/implementations/2026-05-29-phase5-4-pr4-wiring.md`) following the established template, documenting the decoder-removal method chosen, the masking approach, and the alignment-trace result.
- **If Phase B ran:** the findings skeleton from 4.4.
- **If Phase C ran:** implementation docs for PR 6 and PR 7 and the two operations setup docs; append the daily-ingest and weekly-health items to `docs/planning/ROADMAP.md` as done (do NOT commit · just edit the file), and mark them in the OMI freeze checklist.

---

## 7 · Git Boundaries (apply throughout, all phases)

No `git commit`, `git push`, `git tag`, `git merge`, `gh pr create`, `gh pr merge`, `gh release`. No `.gitignore` edits. No file deletion outside the explicit decoder-removal choice in 3.2 (and even that should prefer comment/flag if deletion feels risky · git has it regardless). Stay uncommitted through every phase. The operator reviews the entire arc tonight and commits in logical units. Per-phase file-scope restrictions are stated in Sections 3.1 and 5 · respect them; a phase must not modify another phase's files.

---

## 8 · Style Conventions (locked)

Middle dot (`·`, U+00B7) for separators · never em or en dash. ASCII only. File paths in backticks; code references include line numbers where possible. Type hints and shape-documented docstrings on new public functions. Match existing code style in `world_model.py`, `heads.py`, and the test files. No `print()` in production code. No banner comments. `ruff check` and `ruff format --check` clean on every modified file. Prose over bullets in docs except for genuine enumerations.

---

## 9 · Final Report (branch-dependent)

State clearly which path the run took, then report accordingly.

**If PR 4 landed and Phase B ran:**
1. Files modified/created (paths).
2. Full output of `pytest tests/test_world_model_forward_wiring.py -v` and of the full suite run.
3. `ruff check` / `ruff format --check` results.
4. Decoder-removal method chosen and confirmation the loss is severed from backprop.
5. Alignment-trace result · stated explicitly (which k, that belief state and target agree).
6. Precondition-gate results: powercfg state, CUDA confirmation, 1000-step pre-flight summary (throughput, finite-loss confirmation).
7. 30k launch confirmation: W&B run id and name, that it advanced healthily past ~1-2k steps.
8. Explicit statement: "Gate NOT evaluated · reserved for operator review tonight."

**If PR 4 landed but the precondition gate failed:**
- Everything from PR 4 above, the specific precondition that failed and why, confirmation the 30k was NOT launched, and the Phase C (PR 6/7) results.

**If PR 4 blocked:**
- The full 5-attempt iteration history for the blocking problem, your best understanding of why it is hard, confirmation PR 4 was halted (not force-passed), and the Phase C (PR 6/7) results.

Across all branches: do not stage, commit, or push. End by stating which of the three good states the operator is returning to.
