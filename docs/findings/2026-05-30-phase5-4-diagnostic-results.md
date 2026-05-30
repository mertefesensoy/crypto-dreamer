# 2026-05-30 · Phase 5.4 · 30k Diagnostic Results · Gate Evaluation

Completes the gate-evaluation section left empty in
`docs/findings/2026-05-29-phase5-4-diagnostic-skeleton.md`. The
autonomous run launched the diagnostic and halted at the gate per
design; this doc records the operator-and-design-partner evaluation
of run `1rq8d8u5` against the three gates in `ARCHITECTURE.md`
Section 11.

## 0 · How these numbers were obtained (and why not from W&B)

**The 30k run completed.** The independent heartbeat
(`logs/heartbeat_phase5_4_diag.log`) records `TRAIN_END step=30000` at
04:32 local, and `checkpoints/world_model_diagnostic_step=30000-v1.ckpt`
is dated to match. `Trainer.fit` logged `max_steps=30000 reached`.

**But W&B stopped logging at step ~612 while training continued to
30k.** The W&B run `1rq8d8u5` shows `state: finished` with summary
`_step: 612`; its history contains only ~600 logged points spanning
steps 0-611. This is a logging/sync failure, NOT a training failure.
The charts and the W&B summary are a truncated VIEW of a complete run.
Had the heartbeat not been independent of W&B (a design choice carried
over from the Phase 5.3 stall postmortem), this run would have been
misread as a crash at step 612. It was not.

**Consequence for methodology:** the gate metrics below were measured
by loading the final 30k checkpoint and running the model's own
`_step(..., collect_trace=True)` path over 40 validation batches
(`_gate.py` in repo root), NOT read from the W&B charts. This is more
authoritative than the charts would have been: it is the exact
training code path evaluated on the final trained weights. The W&B
logging failure is filed as a backlog item (Section 5).

## 1 · Measured gate metrics (40 val batches, 30k checkpoint)

```
loss             29.4346   (total)
loss_forward      9.7564
loss_reward       0.4778
loss_continue     0.0000
loss_dyn         32.0007   (free-bits floor-pinned)
loss_rep         32.0007   (free-bits floor-pinned)
kl_unclipped     25.9494

per-horizon forward loss:
  h= 1: 2.2142
  h= 5: 2.3948
  h=15: 2.5209
  h=30: 2.6264
  sum:  9.7564
```

## 2 · Gate verdicts (severity-tiered)

### CRITICAL · Gate 1 · KL release · FAIL

`kl_unclipped = 25.95`, below the 32-nat free-bits floor. Both
`loss_dyn` and `loss_rep` are pinned at 32.0007 (floor-clipped),
meaning the raw posterior-prior KL never reaches the threshold and
clipping pads it to 32. The posterior carries less information than
the free-bits floor requires; the prior predicts `z_t` essentially as
well as the posterior infers it. The stochastic latent is dead.

This value is flat across the entire run: ~26 at the 1000-step
pre-flight, ~27 at step 600 (the last W&B-logged point), 25.95 at the
30k checkpoint. There was no release and no peak-and-decay · the KL sat
below the floor from initialization through 30k. The brief 4.1
pre-flight already showed `kl_unclipped = 26.0`; it never moved.

### CRITICAL · Gate 2 · Forward loss vs marginal baseline · FAIL

Per-horizon forward-distribution loss sums to 9.7564 against the
marginal baseline of 8.8632 (`2026-05-27-marginal-baseline.md`). The
model predicts forward returns ~0.89 nats WORSE than emitting the
unconditional marginal distribution · it is below the
"predict-nothing" baseline. Well outside the [8.85, 8.86] inconclusive
band; an unambiguous fail.

The per-horizon shape is diagnostic: 2.21 / 2.39 / 2.52 / 2.63,
monotonically rising with horizon. The model is least bad at 1-bar and
worst at 30-bar, and every horizon exceeds its share of baseline. The
W&B validation curves (the portion that logged) showed val forward
loss dipping until ~6k then rising for the rest of training while
train forward loss stayed flat · the signature of overfitting to
training-set forward-return structure that does not generalize. For a
near-random-walk return process this is expected: the head burns
capacity chasing the conditional MEAN (unpredictable, per the sqrt(t)
finding in `ARCHITECTURE.md` Section 2) rather than learning the one
plausibly-learnable signal (conditional variance / volatility
clustering).

### PASS · Gate 3 · Reward NLL stability

`loss_reward = 0.4778`, matching Phase 5.3's 0.478. The reward head
improved over training and lands exactly where the prior architecture
did. This confirms the shared RSSM and reward pathway are intact and
the PR 4 wiring is sound · the Gate 1/2 failures describe the
ARCHITECTURE, not a pipeline bug. The PR 4 step-alignment trace
(mutation-verified discriminating) is what licenses this conclusion:
we know `feat[t]` pairs with the correct target, so the collapse is
real, not an off-by-one artifact.

## 3 · The two-target convergence finding (the important one)

crypto-dreamer has now run two architecturally distinct decoder
targets, and BOTH leave the stochastic latent collapsed:

- **Phase 5.3 · feature reconstruction.** Target too EASY · the
  deterministic state `h_t` reconstructed the 15-dim feature row to
  MSE 0.002 without needing `z_t`. KL floor-pinned, kl_unclipped 25.7.
- **Phase 5.4 · forward-return distribution.** Target too NOISY ·
  nobody solved it (forward loss 9.76, barely below the 9.2 training
  entropy and above the 8.86 marginal baseline), and `z_t` still never
  engaged. KL floor-pinned, kl_unclipped 25.95.

Two failures with opposite mechanisms (trivial target vs unlearnable
target) converging on the same collapse is the central evidence for
the contingency analysis in ADR-006. It is consistent with two
non-exclusive hypotheses: (A) the prior network is expressive enough
to out-predict the posterior regardless of target, collapsing any
latent; and/or (B) BTC 1-min data lacks exploitable conditional
stochastic structure at the regime level for a latent to capture at
all. Gate 2's sub-baseline result leans toward B being at least
partly live, but neither has been tested · no standard collapse
remedy has yet been tried. ADR-006 specifies the disambiguating
experiment.

## 4 · Verdict and posture

The Phase 5.4 hypothesis · "predicting a forward-return distribution
gives `z_t` a genuinely stochastic job" · is **FALSIFIED** as
implemented. Gate 1 and Gate 2 both fail on the completed 30k weights.

Per `ROADMAP.md` Section 2 (the Gate Decision Point) and Section 6,
this is the **State B freeze posture**: diagnostic failed, decision
documented (ADR-006), one contingency experiment queued. Phase 5.5
(100k run) is NOT authorized and will not be until a gate passes.
This is the anticipated contingency branch, not a derailment · the
roadmap explicitly planned for it.

## 5 · Follow-up items (backlog)

- **W&B logging halts mid-run while training continues** · CRITICAL
  for unattended runs. Run `1rq8d8u5` synced only to step ~612 of
  30000; the heartbeat saved the interpretation. Investigate the
  wandb 0.27.0 sync stall (possibly the same family as the Phase 5.3
  `wandb.Settings(init_timeout=300)` issue). Until fixed, ALL gate
  evaluation must read from the checkpoint (`_gate.py`), never the
  charts, and the heartbeat is the authoritative completion signal.
  This is doubly important for the OMI-period unattended runs.
- **Gate evaluation tooling** · `_gate.py` (loads a checkpoint, runs
  `_step` with `collect_trace=True` over N val batches, prints the
  three gates) should be promoted from an ad-hoc root script to
  `scripts/eval_gates.py` so future diagnostics have a one-command
  gate read independent of W&B.

## 6 · Related docs

- `docs/findings/2026-05-29-phase5-4-diagnostic-skeleton.md` · launch
  record this completes.
- `docs/design/ADR-006...` · the contingency decision and experiment.
- `docs/findings/2026-05-27-marginal-baseline.md` · the 8.8632 Gate 2
  baseline.
- `docs/design/ARCHITECTURE.md` Section 11 · gate definitions.
