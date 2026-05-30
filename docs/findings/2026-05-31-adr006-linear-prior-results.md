# 2026-05-31 · ADR-006 · Linear-Prior Disambiguation · Results

Executes the ADR-006 contingency experiment (`docs/design/ARCHITECTURE.md`
Section 12) in response to the Phase 5.4 falsification recorded in
`docs/findings/2026-05-30-phase5-4-diagnostic-results.md` (Gate 1 fail
`kl_unclipped` 25.95 < 32 · Gate 2 fail forward sum 9.76 > 8.86 baseline ·
Gate 3 pass 0.478). The experiment restricts the RSSM prior to a linear map
and re-runs the 30k diagnostic to disambiguate:

- **Hypothesis A · over-expressive prior** · the prior MLP is powerful enough
  to match the posterior from `h_t` alone, so the posterior gains nothing by
  encoding the observation and the KL collapses to the free-bits floor.
- **Hypothesis B · no exploitable signal** · the data lacks regime-level
  stochastic structure a latent-variable model can capture in this setup, so
  the KL stays collapsed regardless of prior capacity.

This run was supervised · the operator watched the heartbeat live.

## 0 · How the gate was read (and why not from W&B)

Same methodology and same reason as yesterday: W&B logging silently failed
mid-run on `1rq8d8u5` (synced only to step ~612 of 30000 while training
completed correctly), so the charts cannot be trusted. The authoritative gate
read loads the final checkpoint and runs the model's own
`_step(..., stage="val", collect_trace=True)` over ~40 validation batches,
averaging the `info` components. The heartbeat
(`logs/heartbeat_*.log`) is the authoritative "did it finish" signal. For this
run the gate reader was promoted to `scripts/eval_gates.py` (Section 8).

## 1 · The single experimental variable · `prior_head` diff

The ONLY model change is the RSSM prior's expressive capacity. In
`models/rssm.py`:

**Before** (two-layer MLP · 256 -> 256 -> 1024, GELU):

```python
self.prior_head = nn.Sequential(
    nn.Linear(hidden_dim, mlp_hidden),   # 256 -> 256
    nn.GELU(),
    nn.Linear(mlp_hidden, self.z_dim),   # 256 -> 1024
)
```

**After** (bare affine map · 256 -> 1024):

```python
self.prior_head = nn.Linear(hidden_dim, self.z_dim)   # 256 -> 1024
```

### Norm-preservation decision (ADR-006 Section 2.2 · the critical judgment)

The actual `prior_head` was read precisely before editing. It contained **no
normalization of any kind** · no `LayerNorm` on the input `h_t`, none between
layers, none on the output. It was purely `Linear -> GELU -> Linear`.

Therefore the **no-norm branch** of the brief applies: the replacement is a
bare `nn.Linear(hidden_dim, z_dim)` = `Linear(256, 1024)`, with **nothing to
preserve**. The only thing removed is the hidden layer (`Linear(256, 256)`)
and its `GELU` activation · i.e. exactly the prior's expressive capacity, and
nothing else. Normalization is not expressive capacity (it affects scale and
optimization, not the function class), so dropping a norm would have been a
second, confounding variable · but there was no norm to drop. Here ADR-006's
literal wording ("256 -> 1024, no hidden layer, no activation") and the brief's
refinement ("remove capacity, preserve normalization") coincide because no
norm existed.

Structural confirmation: the Lightning model summary at launch reports the
`rssm` module at **1.3 M params**, ~66 k fewer than yesterday (the removed
`Linear(256, 256)` hidden layer is 256*256 + 256 = 65,792 params), confirming
the capacity reduction took effect.

## 2 · Single-variable integrity (held fixed)

Everything except the prior's capacity is identical to yesterday's run:

- **`posterior_head`** · unchanged. It is a **separate** `nn.Sequential` with
  its own parameters; the prior and posterior share no code and no weights, so
  editing the prior cannot affect the posterior. No coupling -> no STOP
  condition. (Verified by reading `models/rssm.py`: prior and posterior are
  distinct attributes; the only shared input is `h_t`, which is unchanged.)
- **RSSM recurrence** · `pre_gru` MLP and `GRUCell` unchanged.
- **Encoder, reward/forward/continue heads, loss composition** · untouched
  (those files were not edited).
- **Hyperparameters** · `free_bits=1.0`, `coef_dyn=0.5`, `coef_rep=0.1`,
  `T=48`, `batch=32`, `lr=1e-4`, `max_steps=30000`, `mode=diagnostic`, forward
  head + target · all unchanged in `configs/world_model.yaml` (NOT modified).
  The distinct W&B run name and pre-flight heartbeat path were supplied as
  Hydra CLI overrides at launch, not config edits.

The two extra hunks `ruff format` produced in `models/rssm.py` (a blank line
after the module docstring, and comment spacing in `free_bits_kl`) are
pre-existing cosmetic drift with zero behavioral or capacity impact.

## 3 · Precondition gate

1. **powercfg + AC** · PASS. `standby-timeout-ac = 0x0`, `monitor-timeout-ac =
   0x0` (verified via `powercfg /query SCHEME_CURRENT`); `Win32_Battery.
   BatteryStatus = 2` (on AC).
2. **CUDA** · PASS. torch `2.6.0+cu124`, `torch.cuda.is_available() = True`,
   `torch.version.cuda = 12.4`, device NVIDIA GeForce RTX 4070 Laptop GPU.
3. **1000-step pre-flight** · PASS (on the clean re-run). Losses finite
   throughout (step 100 -> 1000: 38.38 -> 30.64, no NaN/Inf), no errors.
   Steady-state throughput **~400-533 ms/step** (avg ~438), in the 400-550
   band. Per the brief, the pre-flight `kl_unclipped` is NOT read as a gate
   signal. See Section 9 for the two-run history: the FIRST pre-flight showed
   a transient contention spike (~1110 ms/step over steps 0-300 and a second
   ~610-820 ms/step patch over steps 600-800) caused by the Ollama desktop app
   loading/unloading a model; once Ollama was idle the same run recovered to
   410-452 ms/step. The operator terminated Ollama and a CLEAN re-run
   confirmed 400-533 ms/step from the first step with no transient.

## 4 · Launch · 30k diagnostic

- **W&B run name** · `phase5.4-linearprior-30k` (distinct from yesterday's
  `1rq8d8u5` / `phase5.4-diag-30k`).
- **W&B run id** · `jnkypsrt`
  (https://wandb.ai/sensoymertefe-ted-niversitesi/crypto-dreamer/runs/jnkypsrt).
  Reminder: W&B is NOT trusted for the gate (it silently truncated yesterday);
  the gate is read from the checkpoint (Section 5).
- **Console log** · `logs/phase5_4_linearprior_30k.log`
- **Heartbeat** · `logs/heartbeat_phase5_4_linearprior.log` · TRAIN_START
  17:19:10 local, step 100 at 17:20:26 (loss 38.3826, identical to the
  deterministic baseline · healthy), W&B init clean, no errors.
- **Healthy advance past ~1-2k steps** · CONFIRMED. Reached step 1500 with
  finite, decreasing losses (38.38 -> 29.47) and no errors. Throughput over
  steps 100-1500 averaged ~654 ms/step (early contention/warmup ~1030 ms/step
  at steps 100-200, a clean ~380-390 ms/step patch at steps 600-800, then a
  steady ~600-660 ms/step) -- comparable to yesterday's late-run rate, implying
  a similar ~5.4 h completion absent a fully idle machine. Early loss
  trajectory matches yesterday's almost exactly, as expected: both priors sit
  at the free-bits floor during warmup; the disambiguation appears only in the
  final-checkpoint `kl_unclipped`.

## 5 · Gate metrics from the checkpoint (40 val batches)

**Checkpoint** · `checkpoints/world_model_diagnostic_step=30000-v2.ckpt`
(2026-05-30 22:34, the new linear-prior run; Lightning auto-versioned `-v2`
because yesterday's `-v1` and the Phase 5.3 `=30000.ckpt` already existed).
Loaded with **0 missing, 0 unexpected keys** (clean load · see Section 8 for
why this matters). Metrics seed-stable across 3 seeds (42/0/123).

```
loss             29.295   (total · = fwd + rew + 0.5*dyn + 0.1*rep)
loss_forward      9.614
loss_reward       0.478
loss_continue     0.000
loss_dyn         32.006   (free-bits floor-clipped)
loss_rep         32.006   (free-bits floor-clipped)
kl_unclipped     26.31    (raw, unclipped · seeds: 26.312 / 26.315 / 26.313)

per-horizon forward loss:
  h= 1: 2.215
  h= 5: 2.349
  h=15: 2.494
  h=30: 2.555
  sum:  9.61    (seeds: 9.6142 / 9.6209 / 9.6147)
```

**Gate verdicts:**
- **Gate 1 · KL release · FAIL (no release).** `kl_unclipped = 26.31` sits
  ~6 nats BELOW the 32-nat free-bits floor; `loss_dyn`/`loss_rep` are pinned
  at 32.006 (floor-clipped). The latent is collapsed. vs the MLP-prior
  baseline 25.95 -> the linear prior changed kl by only **+0.36**.
- **Gate 2 · forward NLL · FAIL.** Sum 9.61 vs the 8.8632 marginal baseline ·
  still ~0.75 nats WORSE than the constant marginal predictor. vs baseline
  9.76 -> a marginal -0.15 improvement, far from crossing 8.85.
- **Gate 3 · reward NLL · ok (integrity).** 0.478 vs ~0.48, unchanged from
  before the edit · confirms only the prior changed and the model loaded
  correctly.

The KL trajectory is FLAT: it began ~26 (the MLP-prior run, and this run's
own early steps) and ended at 26.31 on the final checkpoint · it never
approached, let alone crossed, the 32-nat release threshold. The total-loss
heartbeat was likewise flat (~28-30) and effectively identical to yesterday's
at every logged step, because both runs sit at the free-bits floor with
`loss_dyn`/`loss_rep` clipped.

## 6 · Outcome classification · OUTCOME 3

**OUTCOME 3 · the KL will NOT release even with a linear prior.**

Evidence:
- `kl_unclipped = 26.31` (seed-stable 26.312/26.315/26.313) is ~5.7 nats BELOW
  the 32-nat release threshold. Outcomes 1 and 2 are both hard-gated on
  `kl_unclipped > 32`; that condition is not met, so neither can apply.
- Restricting the prior from a 2-layer MLP to a bare linear map (the
  maximally-handicapped prior ADR-006 chose for the sharpest read on whether
  the KL CAN release) moved `kl_unclipped` by only +0.36 (25.95 -> 26.31). The
  trajectory is FLAT across all 30k steps · heartbeat total loss ~28-30
  throughout, identical to yesterday at every logged step, ending 28.23. No
  upward drift to extrapolate toward release.
- Gate 2 fails independently: forward sum 9.61 is still WORSE than the 8.8632
  constant-marginal baseline · the latent-variable model does not beat
  ignoring the input.
- Measurement is trustworthy: clean load (0 missing / 0 unexpected), Gate 3
  integrity intact (reward 0.478), KL seed-stable.

This FALSIFIES Hypothesis A (over-expressive prior): if the prior's
expressiveness were what collapsed the latent, handicapping it to linear would
have released the KL · it did not. It SUPPORTS Hypothesis B: the data lacks the
regime-level stochastic structure a latent-variable model can exploit in this
setup.

**Caveat (honest).** The +0.36 KL move and the -0.15 forward improvement are
reproducible across seeds and correctly-signed per Hypothesis A · so the prior
plausibly made a SMALL contribution to the collapse. But both are far from
their thresholds (KL ~5.7 nats short of release; forward still above baseline),
so by ADR-006's pre-registered gates this is unambiguously Outcome 3, not a
nascent Outcome 1/2. The directional move is noted, not dispositive.

**Adversarial verification.** Three independent classifiers (one framed to
seek a release, one probing whether the micro-move is a hidden signal) all
returned Outcome 3 at high confidence; none could produce a valid refutation
(`any_valid_refutation_of_outcome3 = false`). The refutations they did raise
are exactly the caveat above, and all three judged it non-dispositive against
the hard `kl > 32` gate.

## 7 · What this means for the next decision (branch NOT started)

Outcome 3 maps to ADR-006 contingency option (c): the world-model / latent
paradigm has now failed to find exploitable regime structure under BOTH a
too-easy target (Phase 5.3 reconstruction), a too-noisy target (Phase 5.4
forward returns), AND across the full prior-capacity range (2-layer MLP down to
bare linear). The stochastic latent collapses regardless. The indicated next
step is to drop the world model and train a model-free RL agent (PPO/SAC) on
the same observation and reward as the honest baseline, written as its own ADR
with the model-free comparison as the deciding experiment.

A secondary, weaker thread (from the Section 6 caveat and ADR-006's broader
contingency menu): the small reproducible directional KL move leaves a thin
opening to first try a target redesign toward conditional VOLATILITY / scale
(option d), which volatility-clustering makes more genuinely predictable, before
abandoning the latent paradigm entirely. This is a judgement call between option
(c) and option (d).

**This decision is NOT taken here.** Choosing between option (c) (model-free)
and option (d) (volatility-target redesign), or any continuation, is a major
architectural fork reserved for the operator and the design partner. This run's
job ends at the classification above. No downstream branch has been started.

## 8 · `scripts/eval_gates.py` promotion + sanity check

Promoted the repo-root `_gate.py` to `scripts/eval_gates.py`, parameterized by
`--ckpt` (and `--config`, `--n-batches`, `--seed`, `--device`). The evaluation
methodology is intentionally identical to `_gate.py` so numbers reproduce; the
only addition is a fixed `torch.manual_seed` so the RSSM straight-through
sampling is deterministic across invocations.

**Sanity check against yesterday's checkpoint · IMPORTANT methodological
finding.** Running `eval_gates.py` on yesterday's `-v1` checkpoint reproduced
`loss_forward` (9.7578 vs 9.7564) and `loss_reward` (0.4777 vs 0.4778) to 3-4
decimals, but reported `kl_unclipped = 73.49` instead of yesterday's 25.95.
This is NOT a bug · it is an unavoidable consequence of the experiment itself:
`-v1` was trained with the OLD two-layer MLP prior, whose state_dict keys are
`rssm.prior_head.0/.2.{weight,bias}`, but the edited `models/rssm.py` now
builds a bare-linear prior expecting `rssm.prior_head.{weight,bias}`. Loading
`-v1` into the current architecture therefore leaves `prior_head` at random
init (`strict=False` silently skips the 4 mismatched keys · verified: 2
missing, 4 unexpected, ALL `prior_head`), and a random prior inflates the KL.
A direct key inspection confirmed: `-v1` has the 4 MLP keys, `-v2` has the 2
linear keys and loads with 0 missing / 0 unexpected.

Consequence: the brief's "reproduce yesterday's numbers on yesterday's
checkpoint" sanity check is **architecturally impossible once `rssm.py` is
edited** · the old checkpoint cannot be faithfully loaded into the new prior.
The method was instead validated three other ways: (a) it reproduces
`_gate.py`'s forward/reward exactly on `-v1` (the architecture-invariant
metrics), isolating the KL divergence to the known `prior_head` mismatch;
(b) `-v2` loads with 0 missing / 0 unexpected keys; (c) Gate 3 integrity holds
(reward 0.478) and the `-v2` `kl_unclipped` is seed-stable (26.31 +/- 0.002
over seeds 42/0/123), as expected for a fully-loaded model. The earlier
seed-sensitivity was entirely the `-v1` random-`prior_head` artifact.

## 9 · Self-correcting-loop iteration history

**Problem · pre-flight throughput anomaly (resolved, not a model issue).**

- _Iteration 1_ · First pre-flight steps 100->300 ran at ~1110-1140 ms/step,
  ~2.8x yesterday's early ~390 ms/step and ~2x the 400-550 band. Hypothesis:
  external CPU/GPU contention rather than the model edit (GPU util was only
  10% at 12.9 W -> training is CPU/kernel-launch bound, not GPU bound; losses
  were finite and tracked yesterday's almost exactly). `nvidia-smi` showed
  Ollama (`ollama.exe`) holding a GPU context alongside the trainer.
- _Observation_ · The same first run then recovered to 410-452 ms/step over
  steps 300-600 (in band), then slowed again to ~610-820 ms/step over steps
  600-800 (coinciding with the operator terminating + Ollama respawning), then
  settled back to 410-430 ms/step by steps 800-1000. This pattern -- spikes
  that correlate with Ollama model load/unload, healthy throughput when Ollama
  is idle -- confirmed the cause was Ollama, not the linear prior.
- _Iteration 2 (resolution)_ · Operator terminated Ollama; a CLEAN pre-flight
  re-run held **397-533 ms/step across all of steps 100-600** with a fast 42 s
  startup (vs 117 s under contention) and no transient. Precondition #3 passes.

No model-side problem occurred · the edit was correct on the first attempt and
the loss trajectory matched yesterday's at every logged step. The loop was
spent entirely on diagnosing an environment (Ollama) throughput artifact.

**Problem · gate-read sanity-check discrepancy (resolved, not a bug).**

- _Iteration 1_ · The `eval_gates.py` sanity check on `-v1` reported
  `kl_unclipped = 73.49` rather than yesterday's 25.95, while forward/reward
  matched exactly. Per the brief I did NOT trust any KL read until explained.
- _Iteration 2 (resolution)_ · Hypothesised an architecture mismatch: the
  edited linear `prior_head` cannot load `-v1`'s MLP prior weights, leaving it
  random. Confirmed by a direct state_dict key inspection (`-v1`: 4 MLP keys ->
  2 missing + 4 unexpected; `-v2`: 2 linear keys -> 0 missing + 0 unexpected).
  The new-checkpoint read (`-v2`, clean load) is the trustworthy one; verified
  seed-stable (kl 26.31 +/- 0.002 over 3 seeds). See Section 8.

The outcome classification (Section 6) was additionally cross-checked by an
independent adversarial multi-classifier verification (3 agents, each also
attempting to refute Outcome 3).

## 10 · Git status

Nothing committed. Files changed and left uncommitted for operator review:
`models/rssm.py` (prior_head -> linear), `scripts/eval_gates.py` (new), this
findings doc. No commits, pushes, tags, merges, or gh operations.
