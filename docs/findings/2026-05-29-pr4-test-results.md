# 2026-05-29 · PR 4 test results and loop iteration history

Test-side companion to `docs/implementations/2026-05-29-phase5-4-pr4-wiring.md`.
Records every PR 4 test with its outcome plus the full iteration history
for anything that triggered the brief Section 0 self-correcting loop. No
problem reached the 5-iteration cap; PR 4 landed clean and the run
proceeded to Phase B.

## Per-test outcomes

All tests in `tests/test_world_model_forward_wiring.py`, on the final code
state. Verified by `pytest tests/test_world_model_forward_wiring.py -v`.

| Test | Outcome |
| ---- | ------- |
| `test_forward_head_matches_spec` | passed · the wired `forward_head` exposes horizons `(1, 5, 15, 30)`, 41 bins, and bin centers at `+/-{0.005, 0.010, 0.018, 0.025}`, matching `FORWARD_HORIZONS` and ADR-002 |
| `test_step_alignment_trace` | passed · LOAD-BEARING. For items 0/mid/last and steps `t in {burn_in, T-1}`, the obs market channels equal `feature_cache[k-256:k]`, the encoded obs the RSSM ingested equals `encode(obs[:,t])`, `feat` equals `cat([h_t, z_t])`, and the forward target the loss paired equals the independently hand-computed `ln(close[k+h]/close[k])` from `dm._closes` for every valid horizon · all anchored on the same `k = ep.kline_idx[start+t]` |
| `test_masked_positions_contribute_zero_loss_and_grad` | passed · an all-invalid mask gives `loss_forward == 0.0` and every `forward_head` parameter gradient is None or exactly zero |
| `test_masked_positions_do_not_change_loss` | passed · corrupting `forward_returns` at invalid positions to 1000.0 leaves `loss_forward` bit-identical (RNG reseeded so the stochastic RSSM samples match across both runs) |
| `test_partial_mask_normalizes_by_valid_count` | passed · a horizon valid for only some positions yields a finite, non-negative per-horizon loss |
| `test_decoder_severed_from_backprop` | passed · no `decoder_head` parameter receives gradient; the forward head, RSSM, and encoder all do |
| `test_total_loss_has_no_decoder_term` | passed · the returned loss equals `L_forward + L_reward + L_continue + coef_dyn*L_dyn + coef_rep*L_rep` to 1e-6 |
| `test_per_horizon_sum_consistency` | passed · `loss_forward == loss_forward_per_horizon.sum()` to 1e-6 even with one horizon masked |
| `test_forward_loss_finite_scalar_and_components` | passed · loss is a finite scalar; `loss_forward`, the (4,) per-horizon vector, reward, continue, dyn, rep, kl_unclipped are all finite |
| `test_forward_gradient_flows_into_head_and_trunk` | passed · >30 parameters carry finite gradients through `feat` |
| `test_legacy_batch_without_forward_keys_runs` | passed · a batch lacking `forward_returns`/`forward_valid` still yields a finite positive scalar loss with `loss_forward == 0.0` (backward-compat path for the pre-PR3 RSSM smoke) |

## Aggregate

- `pytest tests/test_world_model_forward_wiring.py -v` · **11 passed**, ~8 s.
- `pytest tests/ -x -q --ignore=tests/test_dream_endpoint.py` · **69
  passed, 30 warnings**, ~88 s. Up from 58 (+11 from this PR).
- `ruff check models/world_model.py training/train_world_model.py
  tests/test_world_model_forward_wiring.py` · **All checks passed**.
- `ruff format --check` on the same three files · **already formatted**.
- 100-step smoke (`mode=smoke max_episodes=5`, GPU, bf16-mixed):
  ran to `step=100`, heartbeat `loss=38.3280` (finite), checkpoint saved,
  exit 0. Six logged losses present and finite, confirmed via a real
  Trainer's `callback_metrics` (`train/loss_forward_1/5/15/30`,
  `train/loss_reward`, `train/loss_continue`, plus `loss_forward`,
  `loss_dyn`, `loss_rep`, `kl_unclipped`, `kl_clip_excess`).

## Loop iteration history

Per brief Section 0, the loop is recorded even when a single hypothesis
sufficed, because the operator reviews this artifact to audit the
process.

### Decision · decoder removal method (brief 3.2)

**Iteration 1.** Hypothesis: the brief's example wording ("delete
`DecoderHead`, the instantiation at line 99, ...") could be taken
literally. Investigation before acting: grepped non-test usages of
`decoder_head` and found `serve/dream_endpoint.py:258` and
`training/validate.py:114` read `model.decoder_head`, and
`tests/test_rssm.py` both imports the `DecoderHead` class and unit-tests
it (`test_decoder_head_shape_and_loss`). All three are outside Phase A's
edit scope, and brief 3.7 requires the existing suite stay green.
Deleting the class or the instantiation would break the suite and two
production modules. Re-hypothesis: the brief explicitly delegates the
method "based on what you find in the code structure" and the mandatory
requirement is loss/backprop severance, not class deletion; brief 3.7
accepts "no decoder parameters receive gradient" as the severance proof.
Resolution: keep the class and the `self.decoder_head` instantiation,
remove every decoder reference from `_step`, and prove severance by
gradient absence (`test_decoder_severed_from_backprop`). Resolved at
iteration 1; full rationale in implementation doc 5.1.

### Decision · backward compatibility with the legacy RSSM smoke batch

**Iteration 1.** Hypothesis: the new `_step` should read
`batch["forward_returns"]` unconditionally. Investigation:
`tests/test_rssm.py::test_world_model_forward_smoke` (uneditable, in the
required suite) calls `_step` with a batch that predates PR 3 and has no
`forward_returns`/`forward_valid`; an unconditional read would `KeyError`
and turn the suite red. Re-hypothesis: form the forward loss only when
both keys are present (the production datamodule always supplies them),
otherwise skip it · this changes nothing on the real path. Resolution:
`has_forward` guard with `loss_forward = 0` fallback, covered by
`test_legacy_batch_without_forward_keys_runs`. Resolved at iteration 1.

### Verification · off-by-one sensitivity of the alignment trace

**Iteration 1.** Concern (brief HIGHEST RISK): a passing
`test_step_alignment_trace` is only meaningful if it would FAIL on a real
off-by-one. Test: temporarily mutated the wiring to `fwd_targets =
forward_returns[:, t-1]` and re-ran the file. Outcome: two tests failed ·
`test_step_alignment_trace` (target no longer matches the independently
computed `ln(close[k+h]/close[k])`) and
`test_masked_positions_do_not_change_loss` (the mask at step `t` no longer
aligns with the target it gates), while the other nine passed. The
mutation was reverted and the suite re-confirmed green. The trace is
genuinely discriminating, not a tautology. Resolved at iteration 1.

### Smoke · `val_check_interval` exceeds the capped-smoke batch count

**Iteration 1.** Hypothesis: `python -m training.train_world_model
mode=smoke train.max_steps=100 wandb.mode=disabled` would run the 100-step
smoke directly. Failure: Lightning raised `ValueError: val_check_interval
(2500) must be less than or equal to the number of training batches
(166)` during `setup_data`, before any training step · `max_episodes=5`
caps the train set to 5332 starts (= 166 batches at batch_size 32) while
the diagnostic config's `val_check_interval=2500` is sized for the full
30k run. This is a smoke-config mismatch, not a wiring bug; the model
constructed cleanly with all seven submodules (encoder, action_embed,
rssm, decoder_head, reward_head, continue_head, forward_head). Re-
hypothesis: override the validation cadence for the capped smoke without
touching any locked hyperparameter. Test: re-ran with
`train.val_check_interval=50 train.limit_val_batches=3`. Outcome: ran to
step 100, finite loss 38.3280, validation ran at steps 50 and 100, exit
0. Resolved at iteration 2. No locked hyperparameter (T, batch_size,
free_bits, coef_dyn, coef_rep, lr) was changed; `val_check_interval` is a
per-run operational cadence and the diagnostic config retains 2500 for
the full data path.

## Final state

- All 11 new tests passing; full battery 69 passing.
- ruff check + ruff format --check clean on all modified Python files.
- Decoder severed from backprop (gradient-absence verified); total loss
  carries no decoder term.
- Step-alignment trace green and mutation-verified as discriminating.
- 100-step smoke finite and well-formed; six logged losses present.
- No iteration hit the 5-cap. PR 4 done criteria (brief 3.8) all hold.
