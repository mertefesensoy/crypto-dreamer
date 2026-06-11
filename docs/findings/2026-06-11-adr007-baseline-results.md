# 2026-06-11 · ADR-007 · Model-free PPO baseline · Results

Executes Phase 3 of the ADR-007 plan (`docs/design/ARCHITECTURE.md`
Section 12, committed `0afe37b`, Accepted; operating brief
`docs/briefings/2026-06-10-adr007-model-free-baseline-briefing.md`):
full-subset evaluation of the run-of-record PPO baseline against the
pre-registered hard gate, with buy-and-hold and flat as comparators and
seeded random-action as a non-gate sanity reference. Classification is
purely mechanical; every criterion is reported with its pre-registered
threshold beside the realized number.

## 0 · How the gate was read

All numbers come from on-disk JSON artifacts in `artifacts/adr007/`
written by `scripts/eval_baseline_gates.py` · NEVER from W&B (offline,
tracking-only, per the standing rule established after two consecutive
desyncs; see the 2026-05-31 findings doc Section 0). The classifier
(`scripts/classify_adr007_gate.py`) recomputes every derived expression
in float64 from the stored primitives (per-episode `r_i`, per-episode
turnover, the 72 B&H interval returns) per gate section (D); markdown
artifacts are display-only and were not read. Before classification it
verified: every artifact's `episodes_sha256` equals the frozen value;
flat and B&H integrity flags true; the three agent checkpoints
pairwise-distinct by SHA-256.

## 1 · What ran

- **Run of record** (operator launch approval 2026-06-11; first
  completed 3-seed run after the freeze): seeds **42, 0, 123**,
  **2,000,000 env steps each** (1000 PPO updates), frozen config
  `configs/ppo_baseline.yaml` · SHA-256
  `d02454548ea55182034fb4f063cbc12832fa11510178591b3ac5a0c3c5288858`.
  Wall-clock 2026-06-10 21:17:57Z -> 22:04:23Z (~46.5 min; per-seed
  1203.3 / 734.4 / 807.0 s). Zero non-finite events; all 4,176 realized
  training episode starts verified train-pure post hoc ((G)(v)).
- **Evaluation set** (frozen pre-launch): `artifacts/adr007/
  eval_episodes.json` · SHA-256
  `1842d800900b871733414e6d71e068f188e6ce8cc0de2a923035cab43d9811f1` ·
  24 monthly val spans (2024-05 .. 2026-04), 3 non-overlapping 1440-step
  episodes per span at +0/+24/+48 h -> 72 episodes, all gap-free with
  >= 256 bars history, partition rule `(day-1)/days_in_month >= 0.85`
  (UTC) identical to `training/datamodule.py:396`.
- **Evaluated checkpoints** (final per seed, the only gate-eligible
  artifacts per amendment A2):
  - `checkpoints/ppo_baseline_seed42_step2000000.ckpt` · sha256
    `68a9c65d44a7ab5d8f2f9ff590baec65af1292a38a0bdc6bd8680f24aba2a626`
  - `checkpoints/ppo_baseline_seed0_step2000000.ckpt` · sha256
    `891f999d00a63ca346419bf5bebd7f1654b9f6c5a127223a0d62ca98b6b38433`
  - `checkpoints/ppo_baseline_seed123_step2000000.ckpt` · sha256
    `d55aa775ce696b0bba0b2dee338ba7d4a2e55c69b0716dd6f384943bfcbe77ac`
- Evaluation rollouts: CPU, deterministic argmax (lowest-index
  tie-break), per-episode reset to 10,000 cash, policy state-free.
  Classification stamped 2026-06-11T07:04:12Z.

## 2 · Integrity preconditions and amendment A3

Flat passed exactly on all 72 episodes (cumulative net log-return 0.0,
turnover 0.0 · exact float equality). The B&H closed-form check of gate
section (C) initially FAILED on span 2025-02 (|diff| 1.2415e-4 > 1e-4;
23/24 passed), which was a HALT: no gate was read, no agent touched the
full eval set. Diagnosis (read-only, hypothesis-iteration 1) and the
operator ruling produced amendment A3; the corrected reference was then
re-verified (iteration 2) and ALL 24 spans pass at the tightened 1e-5
tolerance · worst |diff| 1.2398e-7 (80x margin), including 2025-02. The
env-run B&H gate inputs were bitwise unchanged across the fix
(R_BH = -0.2627548090021279 before and after) · only the verification
reference moved.

### Amendment A3 (verbatim from ADR-007)

> **Amendment A3 · 2026-06-11 · operator-ratified · B&H integrity
> reference corrected, tolerance tightened.** During the Phase-3 full-set
> integrity preconditions (before any gate read), the B&H closed-form
> check of (C) FAILED on exactly one of 24 spans · 2025-02: env cumulative
> -0.1197939 vs closed-form -0.1199181, |diff| = 1.2415e-4 > 1e-4; the
> other 23 spans passed. Verified root cause: the closed-form assumed a
> constant -10 cash balance with the full entry BTC position held, while
> the env's constant-action-4 path settles the fee deficit by selling
> ~$10 of BTC at the SECOND bar of the span (the env's rebalance rule
> gives delta_value = cash exactly), so the reference mispriced the span's
> price move on that $10 exposure · divergence = ~1.001e-3 x span log
> move, and 2025-02 (-11.86%) was the only span beyond +/-10%. The env and
> harness are correct; the reference formula was wrong. Per the operator
> ruling (option (a) ratified; option (b) · tolerance re-budgeting ·
> REJECTED as spec-weakening): the corrected kline-only reference is, for
> span start row `s`,
>
>     btc_ref = 10000 / (close[s] x 1.0002) - 10 / (close[s+1] x 0.9998)
>     ref     = ln((btc_ref x close[s+4320] - 0.01) / 10000)
>
> (entry at the span-start close with the 10.0 taker fee and +2 bps
> slippage; the analytic second-bar dust-settlement sale of 10.0 notional
> at -2 bps slippage with its 0.01 fee leaving cash -0.01; the residual
> sub-cent dust cascade is bounded below ~6e-7 log and absorbed by the
> tolerance), and the tolerance is TIGHTENED from 1e-4 to **1e-5**. For
> the record: (i) the env-run B&H numbers are the gate inputs per
> amendment A1 and are UNCHANGED by this amendment · only the verification
> reference moved; (ii) B&H aggregates were observed before
> classification · inert, since checkpoints and thresholds are frozen and
> nothing adjustable remains downstream. Authorized harness modification
> is narrowly scoped to the closed-form reference function and the
> tolerance constant · nothing in gate computation, episode execution,
> artifact schema, or policy code.

### Harness diff (A3 scope · 2 files, 57 insertions, 30 deletions)

`scripts/eval_baseline_gates.py` (complete substantive hunks):

```diff
-BH_INTEGRITY_TOL = 1e-4
+BH_INTEGRITY_TOL = 1e-5  # tightened 1e-4 -> 1e-5 by amendment A3 (2026-06-11)

         closed_form = bh_closed_form_logret(
-            float(close.iloc[start_row]), float(close.iloc[start_row + steps])
+            float(close.iloc[start_row]),
+            float(close.iloc[start_row + 1]),  # second bar (A3 dust settlement)
+            float(close.iloc[start_row + steps]),
         )
```

(plus the matching docstring update: 1e-4 -> 1e-5, second-bar closes.)

`training/baseline_policies.py` · `bh_closed_form_logret` before/after
(docstring and the function's inline self-test updated to match; full
diff in the working tree):

```diff
-def bh_closed_form_logret(close_start, close_end, fee=0.001,
-                          slippage=0.0002, initial_cash=10000.0):
-    return math.log(
-        (initial_cash * close_end / (close_start * (1.0 + slippage))
-         - initial_cash * fee) / initial_cash
-    )
+def bh_closed_form_logret(close_start, close_second, close_end, fee=0.001,
+                          slippage=0.0002, initial_cash=10000.0):
+    entry_fee = initial_cash * fee
+    btc_ref = initial_cash / (close_start * (1.0 + slippage)) - entry_fee / (
+        close_second * (1.0 - slippage)
+    )
+    return math.log((btc_ref * close_end - entry_fee * fee) / initial_cash)
```

### Per-span verification table (env vs corrected reference · tolerance 1e-5)

| span | env_cumulative | reference | abs_diff | pass |
|---|---|---|---|---|
| 2024-05 | -0.018929376345863533 | -0.018929392963364388 | 1.6617500854521072e-08 | True |
| 2024-06 | -0.0006608839779609722 | -0.0006608827360283688 | 1.241932603415416e-09 | True |
| 2024-07 | -0.02345867830431379 | -0.023458704383696218 | 2.6079382427907083e-08 | True |
| 2024-08 | -0.0479762972926639 | -0.04797634461664924 | 4.7323985341574115e-08 | True |
| 2024-09 | 0.01637845246919644 | 0.016378470186096127 | 1.771689968690926e-08 | True |
| 2024-10 | 0.07014324247028146 | 0.0701433128233993 | 7.035311784531206e-08 | True |
| 2024-11 | 0.06499613111569613 | 0.06499619355017568 | 6.243447954468184e-08 | True |
| 2024-12 | -0.004056353846079447 | -0.004056355593699542 | 1.74762009464563e-09 | True |
| 2025-01 | 0.03225420487786266 | 0.03225423696574511 | 3.208788244835059e-08 | True |
| 2025-02 | -0.1197939103578459 | -0.11979403434156606 | 1.239837201655325e-07 | True |
| 2025-03 | -0.05667616354728854 | -0.056676219314852734 | 5.57675641915667e-08 | True |
| 2025-04 | 0.006252536468198011 | 0.006252544791217257 | 8.323019246025964e-09 | True |
| 2025-05 | -0.047836129225316534 | -0.047836175699369876 | 4.6474053341794e-08 | True |
| 2025-06 | -0.005279343974947854 | -0.005279347022433176 | 3.0474853222536846e-09 | True |
| 2025-07 | -0.014358470923318803 | -0.014358483626160028 | 1.2702841225079031e-08 | True |
| 2025-08 | -0.03507226317409272 | -0.03507229535334158 | 3.217924886278478e-08 | True |
| 2025-09 | 0.043896538879389345 | 0.043896584104357456 | 4.522496811071308e-08 | True |
| 2025-10 | -0.06446474449407652 | -0.06446480879686055 | 6.430278402802525e-08 | True |
| 2025-11 | 0.010660751347010567 | 0.010660764064490048 | 1.2717479480964244e-08 | True |
| 2025-12 | 0.002628174670631728 | 0.0026281795684589405 | 4.8978272123786915e-09 | True |
| 2026-01 | -0.06278822860197292 | -0.06278829206497481 | 6.346300189530307e-08 | True |
| 2026-02 | 0.018405627770577637 | 0.018405649504055408 | 2.1733477770929932e-08 | True |
| 2026-03 | 0.007962914899116116 | 0.0079629247721108 | 9.872994683954306e-09 | True |
| 2026-04 | -0.03498253990434633 | -0.034982573142052026 | 3.323770569885198e-08 | True |

The failed pre-A3 artifact is preserved as
`artifacts/adr007/eval_bh_full_preA3.json` (identical primitives;
integrity flags only).

## 3 · Numbers · per-seed and comparators (72 episodes; net of fees and slippage; the 0.05 x turnover reward shaping is not a cash flow and is excluded)

| policy | R (cum net log-return) | Sharpe (ddof=1, sqrt(365)) | TO (total) | mean ep TO | max ep TO | worst DD | terminated |
|---|---|---|---|---|---|---|---|
| agent seed 42 | -0.1674535432734741 | -4.004367168608533 | 45.838696382173886 | 0.6366485608635262 | 0.834089669425433 | 0.04468751164160867 | 0/72 |
| agent seed 0 | -0.16919859808246518 | -4.03075885136514 | 46.83761509785614 | 0.6505224319146685 | 1.3331014592296953 | 0.045762158884814275 | 0/72 |
| agent seed 123 | -0.1674535432734741 | -4.004367168608533 | 45.838696382173886 | 0.6366485608635262 | 0.834089669425433 | 0.04468751164160867 | 0/72 |
| buy-and-hold | -0.2627548090021279 | -3.1480304464973776 | 24.024049463009693 | 1.0010020609587371 (per span) | 1.0010029201410913 | 0.12732172303557765 | 0/24 spans |
| flat | 0.0 (exact) | null (zero variance) | 0.0 (exact) | 0.0 | 0.0 | 0.0 | 0/72 |
| random seed 7 (report-only) | -49.446761409460365 | -1168.5021621213928 | 41036.14517287506 | 569.9464607343757 | 592.0038749101288 | 0.5022950219521043 | 38/72 |

Max drawdown is report-only per the pre-registration. Random's numbers
are the expected fee-bleed sanity signature: ~570 turnover per episode
at 12 bps round-trip cost destroys ~7% of equity per day, 38 of its 72
episodes hit the 50% guardrail and counted as complete at termination
per amendment A1, and the huge negative annualized Sharpe is the
mechanical consequence of uniformly catastrophic daily returns with
small dispersion. It participates in no criterion.

## 4 · Median designation and gate classification (designated median seed)

Designation by R per gate section (E): sorted by R the seeds are
[0, 42, 123] with middle value -0.1674535432734741, shared exactly by
seeds 42 and 123 (a true float64 tie · see Section 6) -> the tie rule
designates the numerically lowest tied seed number: **seed 42**.

| criterion | pre-registered definition | realized (median seed 42) | threshold | verdict |
|---|---|---|---|---|
| G-BL1 | slippage-stressed profitability · `R - 0.0002 x TO >= 0.010` | lhs = -0.17662128254990886 | >= 0.010 | **FAIL** |
| G-BL2 | return parity with passive · `R >= R_BH` | R = -0.1674535432734741 | >= R_BH = -0.2627548090021279 | **PASS** |
| G-BL3 | risk-adjusted parity · `S >= max(S_BH, 0.0)` | S = -4.004367168608533 | >= max(-3.1480304464973776, 0.0) = 0.0 | **FAIL** |
| G-BL4 | turnover cap · mean ep TO <= 2.0 AND every ep TO <= 10.0 | 0.6366485608635262 / 0.834089669425433 | <= 2.0 / <= 10.0 | **PASS** |

**VERDICT: FAIL** (`PASS = G-BL1 AND G-BL2 AND G-BL3 AND G-BL4`; G-BL1
and G-BL3 are false). Classification artifact:
`artifacts/adr007/gate_classification.json`. Non-designated seeds, for
dispersion: seed 0 fails/passes identically on all four criteria
(G-BL1 lhs -0.1785661211020364; S -4.0308 < 0; turnover within caps);
seed 123 is numerically identical to seed 42.

Per the pre-registered (H) language: this FAIL · combined with the
ADR-006 result · is evidence that this data/feature/cost setup lacks
extractable edge for BOTH paradigms tried, and the next fork is the
operator's call, made outside this ADR. No recommendation is made here.

## 5 · Observations and anomalies (observations only)

1. **All three seeds converged to a near-constant 50% allocation.** The
   post-classification read-only action tally
   (`scripts/adr007_action_distribution.py`, artifact
   `artifacts/adr007/action_distribution.json`) over the 72 eval
   episodes: seeds 42 and 123 emit action 2 (50%) on ALL 103,680 steps;
   seed 0 emits action 2 on 103,677 steps and action 3 (75%) on 3 steps.
   Eval mechanics of a constant-50% policy: ~0.5 entry turnover at each
   episode start plus continuous small rebalances to hold 50% (mean
   0.637 turnover/episode), return ~half the market move minus ~6 bps
   entry cost per episode.
2. **Seeds 42 and 123 are bitwise identical in evaluation** (identical
   R, S, TO, drawdown to all printed digits) despite pairwise-distinct
   checkpoint SHA-256s · two independently trained networks reduced to
   the same constant argmax action on every eval observation, making
   their deterministic rollouts identical. The (E) tie rule resolved the
   resulting exact tie (-> seed 42).
3. **Training-side entropy collapse (carried over from the Phase-2
   report).** Policy entropy fell from ln(5) ~ 1.609 to ~0 within
   ~40-120 updates in every seed and the policies stayed
   near-deterministic for the remaining ~90% of the 2M-step budget; seed
   0 showed two transient entropy revivals (peak 0.39 near update 560,
   smaller near 860) before re-collapsing · consistent with its 3
   stray action-3 steps at eval. Mean per-episode training return
   (shaping included) settled at -0.035 .. -0.046. Occasional transient
   value-loss spikes (max 0.032) were finite and recovered; the NaN
   guard never fired.
4. **B&H entered every span and rode the period's net-down month-ends**
   (R_BH = -0.263 over the 24 evaluated spans with worst span drawdown
   12.7%) · context for G-BL2 passing with both numbers negative.
5. The A3 episode itself (Section 2): one integrity tolerance was
   mis-budgeted in the pre-registration; the env-side gate inputs were
   unaffected and the corrected reference agrees with the env to ~1e-7.

## 6 · Artifact and checkpoint paths

- Eval artifacts (JSON = classification inputs; MD display-only):
  `artifacts/adr007/eval_agent_seed{42,0,123}_full.{json,md}`,
  `eval_bh_full.{json,md}` (+ `eval_bh_full_preA3.{json,md}` archive),
  `eval_flat_full.{json,md}`, `eval_random_full.{json,md}`.
- Classification: `artifacts/adr007/gate_classification.json`.
- Episode set: `artifacts/adr007/eval_episodes.json` + `.sha256`.
- Action tally: `artifacts/adr007/action_distribution.json`.
- Run log (append-only, every harness invocation + freezes + A3 table):
  `artifacts/adr007/run_log.md`.
- Checkpoints: `checkpoints/ppo_baseline_seed{42,0,123}_step{250000..2000000}.ckpt`
  (periodic gate-ineligible per A2; final = gate inputs), training logs
  `logs/ppo_baseline_seed*_metrics.csv`, heartbeats
  `logs/heartbeat_ppo_seed*.log`, realized train starts
  `artifacts/adr007/train_starts_seed{42,0,123}.json`.

## 7 · Self-correcting-loop iteration history

**Problem · B&H integrity reference mismatch (resolved via amendment
A3, 2 of 5 iterations).** Iteration 1: full-set check failed on span
2025-02 only; HALT; read-only diagnosis isolated the constant-cash
assumption vs the env's second-bar dust-settlement sale, verified
across all 24 spans (predicted-vs-measured residual ~1.2e-6).
Iteration 2: operator ratified option (a); corrected reference + 1e-5
tolerance implemented in the narrowly authorized scope; re-run passed
24/24 with worst |diff| 1.2398e-7. No other problems occurred in
Phase 3.

## 8 · Git status

Nothing committed (standing rule). Working-tree changes left for
operator review: ADR-007 amendment A3 + (C) supersession pointer in
`docs/design/ARCHITECTURE.md`; the A3 harness fix in
`training/baseline_policies.py` and `scripts/eval_baseline_gates.py`;
new `scripts/classify_adr007_gate.py` and
`scripts/adr007_action_distribution.py`; this findings doc; the
`artifacts/adr007/` evaluation artifacts and run-log entries.
