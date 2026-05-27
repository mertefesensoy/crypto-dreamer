# 2026-05-04 — Phase 5.3: RSSM full training run (diagnostic 30k → 100k)

## Problem / Motivation

Phase 5.2 produced a 100-step RSSM smoke run with healthy loss components.
Phase 5.3 is the long-running training that produces the world-model
checkpoint Phase 5.4 evaluates. Original spec was a single 200k-step
4 h run on a desktop-class GPU; on the laptop 4070 with WDDM
kernel-launch overhead, that math doesn't hold, so this phase landed
as a **two-stage diagnostic-then-resume** structure:

- **Stage A: 30k diagnostic** — confirm loss curves are healthy,
  free-bits clipping engages somewhere in the 5k–15k window, val
  reward NLL trends down, no NaNs. Not expected to pass 5.4 gates.
- **Stage B: resume from 30k → 100k** — only after Stage A reviewed.
  100k is the level at which the user expects KL dynamics to mature
  enough for 5.4 gates to plausibly pass.

Plus an unplanned third stage:

- **Stage A1 → A2 split**: Stage A interrupted at step 10k by laptop
  Modern Standby. Resumed offline from ckpt-10k targeting the
  remaining 20k steps. Root cause documented below.

## What Changed

| File | Description |
| --- | --- |
| `models/world_model.py` (line 217-230) | `self.log()` cleanup: detached aggregate metrics; precomputed `kl_clip_excess` outside the log call (prevented a fresh autograd graph node every step). Total loss stays as the live tensor for backprop. |
| `configs/world_model.yaml` | `T=48` (down from 64), `B=32`, `num_workers=4`, `pin_memory=true`, `persistent_workers=true`, `compile=false`, `heartbeat_path: logs/heartbeat_phase5_3_diag.log`, `heartbeat_every_n: 100`, `mode: diagnostic`, `max_steps: 30000`, `val_check_interval: 2500`, `limit_val_batches: 100`, `ckpt_every_n_steps: 5000`. |
| `training/train_world_model.py` | Added `HeartbeatCallback` that writes one line every `heartbeat_every_n` train batches to a configurable path (independent of W&B — survives wandb stalls). Added `+resume_ckpt=...` Hydra override that threads through to `Trainer.fit(ckpt_path=...)`. Wired `pin_memory` and `persistent_workers` into `SpotBTCDataModule`. Used `wandb.Settings(init_timeout=300)` (default 90 s timed out twice). |
| `scripts/probe_lightning.py` (new) | Lightning `barebones=True` 200-step probe — measures Trainer floor cost without any logging or callbacks. Confirmed Lightning's overhead alone is ~70 ms/step above a hand-rolled bench. |
| `scripts/bench_smoke.py` (new) | Hand-rolled `model._step()` + opt loop, single-knob (workers, batch, compile flag). Used for the optimization sweep below. |
| `scripts/test_resume_parity.py` (new) | Test 2 of the correctness gate. Loads a Lightning ckpt, runs 5 fwd+bwd steps with `seed=42` from a fresh DataLoader, repeats with a fresh process state, asserts (a) all losses finite, (b) within ±30 % of reference loss, (c) bit-identical between runs (`rel_diff < 1e-3`). |
| `scripts/diagnose_stall.py` (new) | Pulls W&B history with delta-runtime / delta-step rate analysis. Cross-references the largest gap with `data/market.duckdb*` mtimes. Used to diagnose the Stage A1 stall. |
| `backup_powercfg.txt` (new) | Snapshot of the Windows power scheme at the moment of the run, captured by `powercfg /query SCHEME_CURRENT > backup_powercfg.txt`. To revert: `powercfg /change standby-timeout-ac 30; powercfg /change monitor-timeout-ac 10` (or whatever the original values were). |

## Implementation Approach

**T amendment (64 → 48).** Profiling of the 5.2 smoke at T=64
showed 295 ms/step kernel-launch overhead from the per-step Python
unroll (`for t in range(T):`) on Windows under WDDM. iTransformer
encoder forward dominates per-step compute (~40 % of CUDA time);
RSSM unroll, MLPs, and head computations are launch-overhead-bound.
Reducing T from 64 → 48 cut steady-state ms/step from 458 → 343 in
the bench. T=48 = 48 minutes of context, which still spans:
- 30+ minute autocorrelation horizon for `vol_60`,
- multiple action transitions per trajectory,
- a meaningful prior↔posterior dynamic horizon for KL learning.

This is documented as a **deliberate hyperparameter relaxation**
chosen to amortize WDDM kernel-launch tax on the laptop 4070, *not* an
architectural change. v3 may revert to T=64 on a Linux training
environment.

**Self.log audit (line-228 fix).** The 5.2 smoke had:
```python
self.log(f"{stage}/kl_clip_excess",
         (loss_dyn - kl_unclipped).clamp(min=0), **log_kw)
```
which materialized a *new autograd-graph tensor* inside the call args
every step. Rewritten to precompute `kl_clip_excess` once with
`.detach().clamp_(min=0)`. Aggregate components (`loss_decoder`,
`loss_reward`, etc.) also `.detach()`'d before logging. Total `loss`
stays as the live autograd tensor because Lightning needs the graph
for backward. **Effect**: combined Probe 1+2 measurement was 565 ms
gross / ~415 ms steady-state, under the 550 ms gate.

**Heartbeat callback.** Independent log file at
`logs/heartbeat_phase5_3_diag.log` with one line every 100 train
batches:
```
2026-05-04T18:06:35.123456 step=100 loss=21.5432
```
If the file's mtime stops advancing while the process is alive, we
know training stalled (vs. wandb stalled). The Stage A1 stall was
diagnosed *after the fact* by W&B history; this callback gives
real-time visibility for Stage B.

**Resume mechanism.** `+resume_ckpt=<path>` via Hydra threads through
to `Trainer.fit(model, dm, ckpt_path=resume_path)`. Lightning restores
model state, optimizer state (AdamW moments), LR scheduler position,
and global_step. The DataModule's `WeightedRandomSampler` does *not*
preserve its position across resume (no `state_dict` implementation),
so the data sequence post-resume differs from a hypothetical unbroken
run — but determinism within the resumed run is preserved (verified
by Test 2). For our purposes (matching aggregate trajectories, not
exact per-batch losses), this is acceptable.

**Hydra ckpt-path quirk**: Lightning's default
`ModelCheckpoint(filename="world_model_{mode}_{step}")` produces files
named `world_model_diagnostic_step=10000.ckpt`. Hydra's override
grammar parses `=` as a key-value separator, so
`+resume_ckpt=path/with=equals.ckpt` fails to parse. **Workaround**:
copy the file to a name without `=` before passing to Hydra. Future
runs should set the filename template to `world_model_{mode}_{step:d}`
or similar to avoid the `=`.

## Mathematical / Statistical Details

**Free-bits release expectation.** With 32 categorical latents and a
free-bits floor of 1 nat per latent dim, the dynamics + representation
KL losses are clipped at a floor of `32 × 1 = 32 nat` total. Training
sees `loss_dyn = loss_rep = max(KL_per_step, 32)`. As the posterior
learns observation-specific information that the prior cannot
recover from action history alone, raw KL rises; once it crosses 32
nat per step, `kl_clip_excess` (= `loss_dyn − kl_unclipped`) drops to
0, and the per-latent KL floor stops affecting gradients. This is
"free-bits release" — the inflection point at which the prior is
genuinely tracking the posterior.

User's expectation: the release happens between step 5k and 15k. At
step 10k of Stage A1, `kl_unclipped` was 25.6 nats and `clip_excess`
was 6.4 — clipping still fully active. **Release event will land in
Stage A2 if the model is healthy.**

## Design Decisions

- **Two-stage diagnostic + resume vs single 30k overnight run.** The
  user's two-stage structure ensures a bug surfaces at the end of one
  evening (cost: one evening) rather than after a full overnight
  (cost: one night plus the entire next day if we have to restart).
  Worth the extra checkpoint-resume ceremony.
- **Reject `torch.compile`.** Triton not available in the Windows
  venv; the bench-tagged "compile" runs were actually never invoking
  the compiled path (we called `model._step` directly, bypassing
  `__call__`). Lightning's path goes through `__call__` so attempting
  compile crashes on missing Triton. Disabled in config; documented
  the reason inline.
- **wandb.mode=offline for Stage A2.** The original Stage A1 used
  `mode=online` and we suspected the stall was wandb-sync related.
  Diagnosis ruled that out (it was laptop sleep), but Stage A2 still
  uses offline because (a) it's measurably faster on this network
  (saves ~40 ms/step round-trip), (b) it makes the run robust to
  wandb-server transient errors, (c) syncing one big run at the end
  is faster than streaming. Sync after run completes:
  `wandb sync wandb/offline-run-*`.
- **Reject custom training loop B.** Probes 1+2 demonstrated Lightning
  + the line-228 fix can deliver ~415 ms/step steady-state. Custom
  loop would have saved another ~70 ms/step but at the cost of
  test-surface complexity (5 correctness-gate tests per the
  resume/parity/AMP-state checklist). Not worth the risk for a 17 %
  speedup.
- **Heartbeat callback writes to a plain text file, not a database.**
  A database would be more queryable but adds dependency surface.
  Plain text + tail/grep is universal and works while the run is
  alive.

## Verification

1. **5.2 smoke regression** still passes after the line-228 fix:
   `pytest tests/ -x -q --ignore=tests/test_dream_endpoint.py` →
   `30 passed`.
2. **Probe 1 (Lightning barebones)**: 200 steps in 134.8 s wall
   clock → 524 ms/step steady-state (subtracting 30 s setup). Lightning
   floor cost is ~70 ms/step above the hand-rolled bench's 455 ms.
3. **Probe 2 (line-228 fix + production logging)**: 200 steps in
   112.9 s → 565 ms/step gross, ~415 ms/step steady-state. Under the
   550 ms gate.
4. **Test 2 (resume parity)** on `world_model_diagnostic_step=10000.ckpt`:
   PASS. `rel_diff = 0.00e+00` for all 5 paired steps; all losses
   finite and in `[19.67, 19.69]` (reference 19.68 ± 30 %). Lightning
   ckpt round-trip is bit-identical.
5. **Stage A1 progress at step 10k**:
   `train/loss = 19.68`, `train/loss_decoder = 0.009`,
   `train/loss_reward = 0.467`, `train/loss_continue ≈ 0`,
   `kl_unclipped = 25.6 nat`, `kl_clip_excess = 6.4 nat`,
   `val/loss_reward = 0.478`. Free-bits clipping fully active —
   normal early-training, on track for 5k-15k release window.

## Stall postmortem (Stage A1)

**Symptom**: between step 7149 and 7199, training advanced 50 steps
in 8129 s (≈2 h 15 min). Average step rate before and after the gap:
~700 ms/step. Inside the gap: ~162 s/step.

**Triage** (`scripts/diagnose_stall.py`):
- W&B gap is a single contiguous block, not many small gaps. Rules
  out wandb sync queue throttling.
- DuckDB `market.duckdb` mtime predates the run; WAL absent. Rules out
  DB-write contention.
- Windows event log between 19:48 and 22:05 shows:
  - `19:48:32` — System entering Modern Standby. Reason: Austerity
    Battery Drain Budget Exceeded.
  - `20:40:02` — System entering sleep (Hibernate from Sleep —
    Fixed Timeout).
  - `20:40:04` — System resumed from sleep.
  - `22:03:47` — Power source change.
  - `22:03:48` — System entering Modern Standby (lid?).
  - `22:03:49` — System exiting Modern Standby. Reason: Lid.
  - Volsnap shadow-copy abort + NDIS Wi-Fi miniport "failed power
    transition" clustered at 22:03 — wake-up artifacts.

**Root cause**: laptop entered Modern Standby because Windows decided
the battery drain budget was exceeded, even though we believed the
system was on AC. The transition pattern suggests the AC adapter
either lost contact briefly or Windows misread the power state.

**Mitigation applied** before Stage A2 resume (PowerShell, AC verified
plugged in via `Get-WmiObject Win32_Battery → BatteryStatus = 2`):

```
powercfg /query SCHEME_CURRENT > backup_powercfg.txt
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0
powercfg /setdcvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0
powercfg /setacvalueindex SCHEME_CURRENT 2a737441-1930-4402-8d77-b2bebba308a3 48e6b7a6-50f5-4782-a5d4-53bb8f07e226 0
powercfg /setactive SCHEME_CURRENT
```

(`hibernate off` requires elevation; skipped — `standby-timeout-ac=0`
is sufficient.)

**Reverting** in the morning (no admin needed):
```
# Re-read the original from backup_powercfg.txt and apply, or simply:
powercfg /change standby-timeout-ac 30
powercfg /change monitor-timeout-ac 10
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 1
powercfg /setdcvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 1
powercfg /setactive SCHEME_CURRENT
```

**Future-you reading this in six months**: if you're retraining on
this same laptop and the run mysteriously slows after ~2 h, check
Windows event log for Modern Standby entries first. Replicate the
powercfg block above before kicking off any overnight run.

## Reproducibility

- `lightning.pytorch.seed_everything(42, workers=True)` at script entry.
- DataLoader uses `WeightedRandomSampler` seeded from `seed=42`.
- `torch.backends.cudnn.deterministic` left at default (False) for
  speed. Bit-for-bit reproducibility across runs is not guaranteed.
  Test 2 (resume parity) shows that *within* a process, ckpt loading
  is bit-identical — sufficient for our needs.
- Stage A1 → A2 resume: the WeightedRandomSampler restarts from a
  fresh seed at resume; data order post-resume differs from a
  hypothetical unbroken run. This is acceptable because the world
  model trains on a stochastic sampling distribution, not a fixed
  trajectory.

## Wandb sync instructions (run in the morning)

```
cd C:\Users\senso\OneDrive\Masaüstü\crypto-dreamer
.\.venv\Scripts\python.exe -m wandb sync wandb/offline-run-* --project crypto-dreamer
```

This uploads Stage A2's offline run record to wandb.ai. Stage A1's
online run is already on the cloud (run id `lleske3b`). Diagnostic
report (`scripts/diag_report.py <run_id>`) needs to combine both
runs to produce continuous curves.

## Related Docs

- Plan: `~/.claude/plans/we-re-starting-phase-5-staged-pinwheel.md` (§5.3 + throughput investigation)
- Phase 5.0.5 encoder pretrain: `2026-05-03-phase5-0-5-encoder-pretrain.md`
- Phase 5.1 datamodule: `2026-05-04-phase5-1-datamodule.md`
- Phase 5.2 RSSM smoke: `2026-05-04-phase5-2-rssm-smoke.md`
- Up next (Phase 5.4): validation gates against the final 100k checkpoint.
