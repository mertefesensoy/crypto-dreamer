# 2026-05-29 · PR 4 · ForwardDistributionHead wiring + decoder severance

Phase 5.4 forward-distribution pivot · world-model wiring step. This doc
records what was changed in `models/world_model.py` to replace the
feature-reconstruction decoder loss with the forward-distribution head,
how the forward-validity mask is applied, the decoder-removal method
chosen, and the result of the load-bearing step-alignment trace. The
companion test-results record is
`docs/findings/2026-05-29-pr4-test-results.md`.

## 1 · Problem / Motivation

PR 2 shipped `ForwardDistributionHead` and PR 3 shipped the
`forward_returns` / `forward_valid` targets in the datamodule, but
nothing consumed them: `models/world_model.py::_step` still trained the
old `DecoderHead` (feature reconstruction) whose MSE loss the Phase 5.3
diagnostic showed could be solved by the deterministic state `h_t`
alone, leaving the stochastic latent `z_t` with no job and the KL pinned
at the free-bits floor (posterior collapse, kl_unclipped 25.7 nat). The
pivot's whole premise (ARCHITECTURE Section 6, ADR-002) is that
predicting a distribution over forward returns gives `z_t` a genuinely
stochastic job. PR 4 wires that head into the loss and severs the
decoder so the 30k diagnostic measures the pivot, not a mix of the
pivot and a dead reconstruction term.

The single highest risk (brief 3.5) is a silent off-by-one: pairing the
belief `feat` at trajectory step `t` with `forward_returns` at `t-1` or
`t+1` is a temporal lag or look-ahead leak that a finite-loss smoke
would not catch. The alignment-trace test exists specifically to make
that failure mode loud.

## 2 · What Changed

| File | Change |
| ---- | ------ |
| `models/world_model.py` | Removed the decoder from the training path of `_step` (no `dec_target`, no decoder forward, no `loss_dec`, no `loss_decoder` log). Instantiated `self.forward_head = ForwardDistributionHead(...)` from new constructor params. Added the masked per-(step, horizon) forward cross-entropy, per-horizon W&B logging (`loss_forward_1/5/15/30`), and the `loss_forward` total. Updated the total-loss expression. Added an optional `collect_trace` return path used by the alignment test. Dropped a dead `kl_per_dim` computation. |
| `configs/world_model.yaml` | Added `forward_horizons: [1, 5, 15, 30]`, `forward_bins: 41`, `forward_ranges: [0.005, 0.010, 0.018, 0.025]` under `model:`. No decoder-specific keys existed to remove. |
| `training/train_world_model.py` | Passed the three new config keys into `WorldModel(...)` so the head is genuinely config-driven. `ruff format` also normalized pre-existing formatting in this file (see Design Decisions 5.4). |
| `tests/test_world_model_forward_wiring.py` (new) | 11 tests: the step-alignment trace, mask-contributes-zero (loss and gradient), masked-position value-invariance, partial-mask normalization, decoder-severed-from-backprop, no-decoder-term-in-total, per-horizon sum-consistency, finiteness, gradient flow, and the legacy-batch backward-compat path. |

`models/heads.py` was intentionally NOT modified (see 5.1).

## 3 · Implementation Approach

### 3.1 · Head wiring

`ForwardDistributionHead` is instantiated with `in_dim = hidden_dim +
n_latents * n_classes` (= feat_dim, the same 1280-dim `feat = [h_t, z_t]`
the reward and continue heads consume in the production config) and the
config horizons/bins/ranges. In the per-step loop, for every active step
`t >= burn_in`, the head consumes `feat` (this step's belief, the RSSM
state produced after ingesting `x[:, t]`) and is trained against
`forward_returns[:, t]` · the forward return anchored on the SAME kline
`k = kline_idx[t]` as step `t`'s observation window.

### 3.2 · Mask-aware loss at the wiring level

The head's `loss` method is mask-unaware, and `models/heads.py` is out of
scope for anything but decoder removal, so the masked cross-entropy is
computed in `world_model.py` using the head's public `two_hot_encode`:

```
target_oh = forward_head.two_hot_encode(forward_returns[:, t])   # (B, H, n_bins)
log_probs = log_softmax(forward_head(feat), dim=-1)              # (B, H, n_bins)
ce_bh     = -(target_oh * log_probs).sum(dim=-1)                 # (B, H)
ce_bh     = ce_bh * forward_valid[:, t]                          # zero at invalid
```

Per horizon `h`, the masked CE is summed over `(step, batch)` into a
numerator and the count of valid positions into a denominator; the
per-horizon loss is `numerator_h / max(denominator_h, 1)` and the total
forward loss is the sum across horizons.

### 3.3 · Total loss

```
L_total = L_forward + L_reward + L_continue + coef_dyn * L_dyn + coef_rep * L_rep
```

with no decoder term · matching ARCHITECTURE Section 9.

## 4 · Mathematical / Statistical Details

**Two-hot cross-entropy (unchanged from the head).** For a target
log-return `v` at horizon `h` with symmetric range `R_h` and 41 bins, the
target is the two-hot vector placing weight `1 - frac(pos)` and
`frac(pos)` on the two bins adjacent to `pos = (clamp(v, -R_h, R_h) +
R_h) / bin_width_h`. The per-(sample, horizon) loss is
`CE = -sum_k target[k] * log_softmax(logits)[k]`.

**Masking and normalization (new, the correctness-critical part).** Let
`m_{b,t,h} in {0,1}` be `forward_valid`. The forward loss is

```
L_forward = sum_h ( sum_{b,t} m_{b,t,h} * CE_{b,t,h} ) / max( sum_{b,t} m_{b,t,h}, 1 )
```

i.e. each horizon is a mean of cross-entropy over only its valid
`(batch, step)` positions, and the total sums across horizons. Three
consequences:

1. Invalid positions (the `forward_valid = False`, `forward_returns =
   0.0` series-end placeholders) contribute exactly zero to the loss and,
   because the per-position CE is multiplied by `m` before any
   reduction, exactly zero gradient. The 0.0 placeholder is never a real
   "zero-return" training signal.
2. The per-horizon losses sum to the total by construction, so the W&B
   series `loss_forward_1/5/15/30` add up to `loss_forward`.
3. Normalizing each horizon by its own valid count makes the total
   comparable to the marginal baseline in
   `docs/findings/2026-05-27-marginal-baseline.md` (8.8632), which is
   itself the sum of per-horizon cross-entropies (Gate 2). `fwd_denom` is
   a count and carries no gradient, so dividing by it does not distort
   the backward pass.

**Burn-in.** The forward loss, like the reward/continue/KL losses, is
formed only for `t >= burn_in` (default 5): the freshly-reset RSSM state
at the first steps is uninformative (ARCHITECTURE Section 9).

## 5 · Design Decisions

### 5.1 · Decoder removal method · keep the class and module, sever the loss

Brief 3.2 delegates the removal method ("delete outright or
comment/feature-flag · your judgment based on what you find in the code
structure"); the mandatory requirement is that the decoder loss is
severed from the total loss and the autograd graph, the decoder forward
does not run in training, and `dec_target` is not computed.

What the code structure showed: `models/heads.py::DecoderHead` is
imported and unit-tested directly by `tests/test_rssm.py`
(`test_decoder_head_shape_and_loss`), and `model.decoder_head` is read by
two production consumers · `serve/dream_endpoint.py:258` and
`training/validate.py:114` · for visualization/diagnostics. None of
those files are in Phase A's edit scope. Deleting the class or the
`self.decoder_head` instantiation would break the existing test suite
(which brief 3.7 requires stays green) and two production modules.

Decision: keep the `DecoderHead` class in `heads.py` (untouched) and keep
`self.decoder_head` instantiated, but remove every decoder reference from
`_step`. The decoder forward never runs during training, no decoder
target is computed, and no decoder term enters the loss or the autograd
graph; its parameters therefore receive no gradient and never update.
This fully satisfies the mandatory severance while keeping the suite and
the external consumers working. Brief 3.7 explicitly allows verifying
severance via "no decoder parameters receive gradient," which the test
`test_decoder_severed_from_backprop` does. A future PR re-points the dream
and validate consumers at the forward head and removes the module
(ADR-002 already flags the dream visualization re-point as future work).

Alternative considered: delete the class and instantiation as the brief's
example wording suggests. Rejected because it would turn the required
test suite red and break two uneditable production modules · the brief's
overriding rule is "never weaken the spec to make a test pass," and a
half-broken repo for an unattended overnight diagnostic is the larger
risk. The instantiated-but-unused module wastes no training compute (its
forward is never called) and adds no autograd nodes, which is the
concern the brief actually raises about half-removed decoders.

### 5.2 · Mask applied at the wiring level, not in the head

Brief 3.3 offers two options. Masking the per-(step, horizon) CE before
reduction (chosen) is exact and keeps `models/heads.py` untouched (its
only permitted change was decoder removal, which 5.1 declined). The head's
public `two_hot_encode` is reused so the encoding is not duplicated; only
the reduction is at the wiring level.

### 5.3 · Per-horizon valid-count normalization

Considered: a single global normalization by total valid `(step, batch,
horizon)` count. Rejected in favor of per-horizon normalization because
the architecture defines `L_forward = CE_1 + CE_5 + CE_15 + CE_30` as a
sum across horizons and the Gate 2 baseline is likewise a per-horizon
sum; per-horizon normalization makes the logged per-horizon series and
the total mutually consistent and directly comparable to that baseline.

### 5.4 · Config-driven head and the entrypoint edit

Brief 3.3/3.4 require the head be instantiated "from config" and the keys
added to `configs/world_model.yaml`. `train_world_model.py` maps each
`cfg.model.*` to a `WorldModel` kwarg explicitly, so honoring "from
config" requires passing the three new keys through the entrypoint;
otherwise the config keys would be inert and editing them tonight would
silently do nothing · itself a footgun. The brief's Phase A "may modify"
list does not name `train_world_model.py`, but the operator's governing
constraint is "a phase must not modify another phase's files," and the
trainer entrypoint belongs to no other phase (Phase B launches runs and
writes findings; Phase C creates `scripts/` and `docs/operations/`). The
edit is three kwargs plus `list(...)` materialization of the OmegaConf
lists. `WorldModel.__init__` also carries defaults that mirror the config
so config-less constructions (tests, the dream endpoint) still get the
canonical layout. `ruff format` reformatted pre-existing lines in
`train_world_model.py` to satisfy brief 8's per-file format-check gate;
this matches the precedent set in the PR 3 test-results doc.

### 5.5 · `collect_trace` on the real `_step`

The alignment test needs to inspect the actual tensors `_step` pairs, not
a re-implementation that could drift from the real path. `_step` takes an
optional `collect_trace` flag (default False, so training/validation are
byte-identical to before) that returns a diagnostics dict with the
per-active-step `feat`, RSSM `h`/`z`, encoded obs, and the exact
`forward_targets`/`forward_valid` the loss used. The test asserts `feat ==
cat(h, z)` (the head consumes this step's belief), `x_used ==
encode(obs[:, t])`, and that the captured target equals the
independently hand-computed `ln(close[k+h]/close[k])`.

## 6 · Verification

- `pytest tests/test_world_model_forward_wiring.py -v` · **11 passed**.
- `pytest tests/ -x -q --ignore=tests/test_dream_endpoint.py` · **69
  passed** (58 baseline + 11 new), ~88 s.
- `ruff check` + `ruff format --check` on `models/world_model.py`,
  `training/train_world_model.py`,
  `tests/test_world_model_forward_wiring.py` · clean.
- **Step-alignment trace:** PASS. Verified directly that the test catches
  the failure mode it guards: temporarily mutating the wiring to
  `forward_returns[:, t-1]` made `test_step_alignment_trace` (and
  `test_masked_positions_do_not_change_loss`) fail while the other nine
  passed; the mutation was reverted.
- **100-step smoke** (`mode=smoke max_episodes=5`, GPU, bf16-mixed):
  PASS. Ran to `step=100` with heartbeat `loss=38.3280` (finite),
  `Trainer.fit stopped: max_steps=100 reached`, checkpoint
  `world_model_smoke_raw.pt` saved, exit 0; validation ran at steps 50 and
  100 without error. The total ~38.3 matches the expected init
  composition: `L_forward` ~15 (four horizons each ~ln(41) at uniform
  logits), `L_reward` ~3.7, `L_continue` ~0.69, `0.5 * L_dyn` ~16 (the
  32-nat free-bits floor), `0.1 * L_rep` ~3.2. All six logged losses
  (`loss_forward_1/5/15/30`, `loss_reward`, `loss_continue`) are present
  and finite, confirmed separately by inspecting a real Trainer's
  `callback_metrics`. The smoke needed two CLI overrides on the validation
  cadence (`val_check_interval=50 limit_val_batches=3`) because
  `max_episodes=5` caps the train set to 166 batches/epoch while the
  diagnostic config's `val_check_interval=2500` is sized for the full run;
  no locked hyperparameter (T, batch_size, free_bits, coef_dyn/rep, lr)
  was changed. See the test-results doc for the iteration history.

See `docs/findings/2026-05-29-pr4-test-results.md` for per-test outcomes
and the full loop iteration history.

## 7 · Related Docs

- `docs/design/ARCHITECTURE.md` · Sections 6 (forward head), 9 (loss), 11
  (gates), 12 (ADR-001, ADR-002).
- `docs/implementations/2026-05-28-phase5-4-pr3-forward-returns.md` · the
  anchor convention this wiring preserves.
- `docs/implementations/2026-05-27-phase5-4-pivot-forward-distribution.md`
  · the pivot rationale.
- `docs/briefings/PR4-and-diagnostic-briefing.md` · the governing brief.
- `docs/findings/2026-05-29-pr4-test-results.md` · test-side companion.
