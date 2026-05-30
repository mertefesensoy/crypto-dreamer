# 2026-05-29 · Phase 5.4 · 30k diagnostic · findings skeleton

This is the launch record and gate-evaluation skeleton for the Phase 5.4
forward-distribution 30k diagnostic. It is written at launch time by the
semi-autonomous run. The gate evaluation (Section 4) is deliberately left
EMPTY · per brief Section 4.3 the Gate 1/2/3 fork (advance to Phase 5.5
vs enter the ADR-006 contingency) is reserved for the operator and the
design partner tonight. This run launched the diagnostic and confirmed it
is healthy; it did NOT interpret the gates.

## 1 · W&B run

- **Run name:** `phase5.4-diag-30k`
- **Run id:** `1rq8d8u5`
- **Project:** `crypto-dreamer`
- **Mode:** online
- **Launch timestamp (local):** 2026-05-29 23:09:04 (process start);
  `TRAIN_START` 23:09:50; detached process PID 33980 (Start-Process,
  hidden), stdout/stderr at `logs/pr4_30k_stdout.log` /
  `logs/pr4_30k_stderr.log`.
- **Config:** `mode=diagnostic`, `max_steps=30000`, T=48, batch_size=32,
  bf16-mixed, lr=1e-4, warmup_steps=1000, free_bits=1.0, coef_dyn=0.5,
  coef_rep=0.1, forward_horizons=[1,5,15,30], forward_bins=41,
  forward_ranges=[0.005, 0.010, 0.018, 0.025].
- **Heartbeat:** `logs/heartbeat_phase5_4_diag.log`
- **Early-progress confirmation:** advanced to step 2000 by 23:22 with
  finite losses at every 100-step heartbeat (38.33 at step 100 declining
  to 28.50 at step 2000, no NaN/Inf), process PID 33980 alive, ~387
  ms/step (steps 100->1600 over 580 s). W&B online run `1rq8d8u5` logging.

## 2 · Precondition gate results (brief 4.1)

### Precondition 1 · powercfg + AC — PASS

- `powercfg /change standby-timeout-ac 0` and `monitor-timeout-ac 0`
  applied. Verified via `powercfg /query SCHEME_CURRENT`:
  - Standby AC Power Setting Index: `0x00000000` (0)
  - Monitor AC Power Setting Index: `0x00000000` (0)
- AC confirmed: `Win32_Battery.BatteryStatus = 2`, `PowerLineStatus =
  Online`.

### Precondition 2 · CUDA — PASS

- `torch.cuda.is_available()` = True, `torch.version.cuda` = `12.4`,
  `torch.__version__` = `2.6.0+cu124`, device = NVIDIA GeForce RTX 4070
  Laptop GPU.
- W&B API key present (the online run will not block on authentication).

### Precondition 3 · 1000-step pre-flight — PASS

Ran the diagnostic config (full data, offline run id `utbnfmdd`) for 1000
steps with W&B offline as a NaN/divergence/throughput check. Component
values read by loading the step-1000 checkpoint
(`world_model_diagnostic_raw.pt`, loaded missing=0/unexpected=0) and
running one `_step` on a real batch.

- Total loss finite throughout: heartbeat every 100 steps shows
  38.33 -> 35.58 -> 29.50 -> 30.21 -> 30.04 -> 31.29 -> 30.25 -> 29.90 ->
  30.22 -> 30.72; no NaN/Inf.
- `kl_unclipped` = 25.999 · finite, positive, near the 32-nat free-bits
  floor (not exploding, not zero). Release above the floor is the Gate 1
  question (Section 4), not evaluated here.
- `loss_forward` = 9.746; per-horizon `loss_forward_1/5/15/30` =
  [2.347, 2.435, 2.422, 2.542] · all finite and sum-consistent with the
  total.
- `loss_reward` = 1.483, `loss_continue` = 0.0003 · finite.
- Composition cross-check: 9.746 + 1.483 + 0.0003 + 0.5*32.05 + 0.1*32.05
  = 30.45 = total loss · the wiring math is exact on real data and carries
  no decoder term.
- Throughput (steps 100->1000, 900 steps over 353.6 s): ~393 ms/step ·
  within the sane range (target ~400-550; faster, nowhere near a 3x
  stall).
- Verdict: PASS.

## 3 · Diagnostic status

Launched and running independently (detached PID 33980, Start-Process
hidden, survives without agent babysitting). Confirmed a healthy advance
past step 2000 with finite losses; at ~387 ms/step the 30000-step run
projects to ~3.2 h (finishing ~02:30 local). Self-sufficiency: heartbeat
`logs/heartbeat_phase5_4_diag.log`, W&B online run `1rq8d8u5`, and
periodic checkpoints every 5000 steps to
`checkpoints/world_model_diagnostic_{step}.ckpt` plus `last.ckpt`.

Note: `checkpoints/world_model_diagnostic_raw.pt` currently holds the
1000-step PRE-FLIGHT state; the entrypoint overwrites it with the final
30k state when the run completes. Track the live run via W&B / heartbeat /
the step-numbered checkpoints, not `*_raw.pt`, until completion.

HALTED AT GATE per brief Section 4.3 · Gate 1/2/3 were NOT evaluated by
this run (Section 4 is intentionally empty and reserved for the operator).

## 4 · Gate Evaluation · TO BE COMPLETED BY OPERATOR

Per brief Section 4.3 this section is intentionally empty. Do NOT fill it
from within the autonomous run. The Gate 1/2/3 outcome determines whether
the project advances to Phase 5.5 or enters the ADR-006 contingency
(prior-capacity restriction vs KL warmup vs paradigm change), and that
fork is reserved for the operator and the design partner. Evaluate after
the 30k run reaches step 20k (and at completion).

| Gate | Threshold (ARCHITECTURE Section 11) | Measurement at step 20k | Verdict |
| ---- | ----------------------------------- | ----------------------- | ------- |
| Gate 1 · KL release | `kl_unclipped > 32 nat` by step 20k (load-bearing: confirms z_t encodes information the prior cannot predict from action history alone) | _to fill_ | _to fill_ |
| Gate 2 · Forward loss vs marginal baseline | `val/loss_forward_dist < 8.85` at step 20k. Marginal baseline = 8.8632 (`docs/findings/2026-05-27-marginal-baseline.md`); the band [8.85, 8.86] is a tie with the baseline and is reported as INCONCLUSIVE, not passing | _to fill_ | _to fill_ |
| Gate 3 · Reward NLL stability | `val/loss_reward` ~ 0.48 (Phase 5.3 achieved 0.478); a significant regression indicates a bug in the shared RSSM/reward pathway | _to fill_ | _to fill_ |

**Decision (operator):** If Gate 1 and Gate 2 pass -> proceed to Phase 5.5
(100k full run). If Gate 1 fails -> write ADR-006 and evaluate the
contingency options. If Gate 2 fails with Gate 1 passing -> the problem is
in the head/target, not the RSSM. If Gate 3 regresses -> debug the shared
components. See ARCHITECTURE Section 11 and ADR-006 (contingency).

## 5 · Notes

- This diagnostic measures the forward-distribution pivot (PR 4 wiring) in
  isolation: the old feature-reconstruction decoder is severed from the
  loss and backprop, so KL release (Gate 1) reflects the forward target's
  stochastic structure, not a reconstruction term.
- The run was launched detached so it completes independently of the
  agent session.
