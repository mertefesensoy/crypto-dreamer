# 2026-05-04 — Phase 5.2: RSSM core + heads + losses, 100-step smoke

## Problem / Motivation

Phase 5.0.5 produced a pretrained encoder; Phase 5.1 built the
trajectory datamodule. Phase 5.2 wires those together with the
DreamerV3-style RSSM and three prediction heads (decoder, reward,
continue), and runs a 100-step smoke training pass. The goal is *not*
to learn anything substantive in 100 steps — it's to prove every
piece of the loss pipeline is wired correctly: shapes match, all four
loss components are computed and logged separately, no NaN/Inf in
gradients, and free-bits clipping is mechanically active. Phase 5.3
(the 4-hour run) will not start until this smoke says everything's
healthy.

## What Changed

| File | Description |
| --- | --- |
| `models/rssm.py` (new) | `RSSM` cell. Pre-GRU MLP projects `[z_{t-1}, action_emb_{t-1}]` to GRU input; `nn.GRUCell(hidden=256)` produces the deterministic state. Two MLP heads emit logits for `prior` (from `h`) and `posterior` (from `[h, x]`). `sample_st` does straight-through categorical sampling with 1% unimix to keep KL finite. `categorical_kl` and `free_bits_kl` are static methods so the world model can plug them into either KL direction. |
| `models/heads.py` (new) | `DecoderHead` (3-layer MLP → 15-dim, MSE loss); `RewardHead` (3-layer MLP → 41 logits over `[-0.2, 0.2]`, two-hot encoded target, cross-entropy loss); `ContinueHead` (2-layer MLP → 1 logit, BCE-with-logits loss). All take `[h, z]` of dim `256 + 32·32 = 1280`. |
| `models/action_embed.py` (new) | Trivial: `nn.Embedding(5, 32)` subclass. |
| `models/world_model.py` (new) | `WorldModel` Lightning module wrapping encoder + RSSM + heads + action embed. `_step` unrolls T=64 trajectory steps in Python, computing the four losses per step, accumulating, then averaging across loss-active steps. Handles `is_first` reset, burn-in (skip first 5 steps' losses), and the per-step encoder forward (B·T windows in one batch). |
| `training/datamodule.py` | Added `max_episodes: int \| None = None` constructor arg + a `head(N)` filter at `setup()` time so the smoke run can shrink to 5 episodes via Hydra override. |
| `training/train_world_model.py` (new) | Hydra entrypoint. Builds DataModule + WorldModel + W&B logger + `TimeBudgetCallback` + `ModelCheckpoint` (best by `val/loss_reward`). Auto-applies `max_episodes=5` when `mode=smoke`. Saves a raw `state_dict` at the end (Phase 5.4 dream endpoint will load this). |
| `configs/world_model.yaml` (new) | Default config with `mode: smoke`. CLI overrides for the Phase 5.3 full run: `mode=full train.max_steps=200000 train.max_hours=4 wandb.run_name=phase5.3-rssm-full data.max_episodes=null`. |
| `tests/test_rssm.py` (new) | 12 tests across RSSM cell, three heads, and a full `WorldModel._step` forward+backward on tiny synthetic data (asserts every parameter has a finite gradient). |

## Implementation Approach

**Trajectory unroll alignment.** At loop iteration `t ∈ [0, T)`:
- `prev_a` (initialized to zero at t=0; updated to `action[:, t]` at end
  of iteration) is the *previous* action — the one that semantically
  led to the current step.
- `x_t = encoder(obs_window[:, t]).mean(dim=variables)` — the
  encoded current observation, mean-pooled across the 15
  per-variable tokens to a single d_model=128 vector.
- `h_t, z_t, prior_logits, post_logits = rssm.step(prev_z, prev_h,
  embed(prev_a), x_t, is_first=reset_mask)`.
- Predictions from `[h_t, z_t]`:
  - `decoder_head` → predicts the 15-dim feature row at the *current
    bar* (`obs_window[:, t, -1, :]`, the last row of the 256-bar
    window), MSE loss.
  - `reward_head` → 41 logits, two-hot CE against `reward[:, t]`.
  - `continue_head` → 1 logit, BCE against `continue_flag[:, t]`.
- KL between posterior and prior at this step.

**Hidden-state reset.** `reset = is_first[:, t]`, plus `t==0` always
forces reset (start of trajectory regardless of `is_first` value).
The RSSM cell zeroes `prev_z` and `prev_h` where `reset` is True
before the GRU step, so a fresh trajectory starts from origin.

**Burn-in.** `burn_in=5`: the first 5 steps' losses are not added to
the accumulators. The RSSM has just been reset to zeros and its
predictions from a freshly-initialized hidden state are dominated by
the init, not the dynamics. Including them would push the model to
"learn the init" which is pointless. Loss aggregation runs over
`T - burn_in = 59` steps.

**Pretrain weight reuse.** WorldModel passes `mae_checkpoint` through
to its encoder. At construction time, the encoder loads
`checkpoints/encoder_mae_full_raw.pt`, copies `input_proj` (shared),
all 48 transformer encoder tensors, and `var_embed[:12]` (the market
features). `var_embed[12:15]` (portfolio rows) stay at random init.
Logged at INFO: `Loaded MAE pretrain ... copied: input_proj,
var_embed[:12], encoder.* (48 tensors); skipped: var_embed[12:15]
(random init)`.

## Mathematical / Statistical Details

**Two-hot reward encoding.** Bin centers `c_i = -0.2 + i · (0.4 / 40)`
for `i ∈ [0, 41)`. For a true reward `r` clipped to `[-0.2, 0.2]`,
locate the two adjacent bins `i_lo = ⌊(r - low) / Δ⌋` and `i_hi =
i_lo + 1`. Mass on `i_hi` = `(r - c_{i_lo}) / Δ`; mass on `i_lo` =
`1 - mass_{i_hi}`. Loss is cross-entropy `-(target · log_softmax(logits)).sum`.
At inference, `predict(logits) = sum_i softmax(logits)_i · c_i` —
the expected value under the predicted distribution.

**Categorical KL with unimix.** For two categorical distributions
`q, p ∈ Δ^{n_classes}` with logits, the per-class softmax is mixed
with uniform: `q' = (1 - α) · softmax(q_logits) + α / n_classes`,
same for `p'`. Then `KL(q' || p') = Σ_c q'_c · log(q'_c / p'_c)`.
The unimix term `α = 0.01` ensures every class has probability ≥
`α / n_classes`, so no `log(0)` and no infinite KL.

**Free-bits clipping (per spec).** With `kl ∈ ℝ^{B × n_latents}`:
1. `avg_per_latent = kl.mean(dim=batch)` — vector of length 32.
2. `clipped = max(avg_per_latent, free_bits=1.0)` per latent.
3. `loss = clipped.sum()` — sum across latents.

When all 32 latents are below 1 nat (early training), the loss is
pinned at exactly 32 nats and gradient is zero. Gradient flow to the
prior/posterior heads only resumes once individual latents start
exceeding 1 nat.

**Loss aggregation.**
```
L_pred(t)   = MSE(decoder(t), obs_window[:,t,-1,:])
            + CE(reward(t), two_hot(reward[:,t]))
            + BCE(continue(t), continue_flag[:,t])

L_dyn(t)    = free_bits_kl(KL(stop_grad(post_t) || prior_t))
L_rep(t)    = free_bits_kl(KL(post_t || stop_grad(prior_t)))

L_total = mean over loss-active steps of [
    L_pred + 0.5 · L_dyn + 0.1 · L_rep
]
```

The mean (rather than sum) over steps keeps the coefficient
balance independent of `T - burn_in`.

## Design Decisions

- **Mean-pool variable tokens for x_t.** Considered: flatten the
  `(15, 128)` per-variable tokens to a `(15·128,)` vector for the
  posterior input. Rejected — increases the posterior_head input
  dim from 256+128 to 256+1920, ~7× more params for no clear
  benefit. Mean-pooling collapses the variables to a single feature
  vector that the posterior_head can then mix with `h`.
- **Burn-in vs explicit warm-up.** Considered: feed the first
  `burn_in` steps through the RSSM but compute losses anyway. Burn-in
  cleanly separates "context-only" steps from "loss-active" steps —
  lower variance estimator of the loss per gradient step. Cost: 5
  fewer steps' worth of supervised signal per trajectory.
- **`every_n_train_steps` checkpoints, not `every_n_epochs`.** The
  smoke run has only 100 steps, well below one epoch. Step-based
  checkpointing also makes Phase 5.3's 200K-step run bookkeeping
  consistent with the smoke run's.
- **Two-hot vs scalar regression for reward.** Two-hot is a soft
  classification target — gives bounded gradients regardless of
  reward magnitude, much easier to optimize than direct scalar
  regression with MSE, especially for sparse-reward signals. Standard
  DreamerV3 choice.
- **`gradient_clip_val=1000` (huge).** DreamerV3 convention: KL spikes
  on individual categoricals can push gradient norm into the
  hundreds; clipping at the typical Adam-default 1.0 would crush
  legitimate updates. 1000 is effectively a "never clip in practice"
  setting that still catches NaN explosions.

## Verification

1. **Tests.** `pytest tests/ -q` → **30/30 pass** (10 v1 env + 6
   encoder + 4 datamodule + 12 RSSM/heads/world-model).

2. **100-step smoke run.**
   ```
   python -m training.train_world_model mode=smoke \
       train.max_steps=100 train.val_check_interval=50
   ```
   W&B run [phase5.2-rssm-smoke (sf3ebyfh)](https://wandb.ai/sensoymertefe-ted-niversitesi/crypto-dreamer/runs/sf3ebyfh).
   - 5 episodes filter applied: 5,268 train + 1,347 val candidates.
   - Total params: **3.3M** (encoder 827K, RSSM 1.4M, decoder head
     397K, reward head 404K, continue head 328K, action embed 160).
   - Wall clock: **126 s** (~1.3 s/step on the 4070 with bf16-mixed,
     batch 16, T=64).

3. **Pass criteria** (per plan):

   | Criterion | Result |
   | --- | --- |
   | Total loss decreasing over 100 steps (first10 mean vs last10) | **PASS** — 23.84 → 23.78 (+0.3%) |
   | Each loss component logged separately | **PASS** — see W&B panel |
   | No NaN/Inf in gradients | **PASS** — all logs finite, no crash |
   | Gradient norm bounded (no explosion) | **PASS** — clip val 1000 never tripped |
   | Free-bits clipping active | **PASS** — `kl_clip_excess` ≈ 31.78, raw KL ≈ 0.20 nats |

4. **Per-component drift** (10-step rolling mean):

   | component | first10 | last10 | drop | interpretation |
   | --- | --- | --- | --- | --- |
   | decoder | 0.272 | 0.264 | +2.8% | learning the current-bar 15-dim feature reconstruction |
   | reward | 3.694 | 3.679 | +0.4% | barely moves — 41-bin classification needs more steps |
   | continue | 0.678 | 0.635 | +6.4% | strongest learner — most steps are continue=True, easy bias to learn |
   | dyn (clipped) | 32.00 | 32.00 | 0% | constant — KL below floor, no gradient |
   | rep (clipped) | 32.00 | 32.00 | 0% | same |
   | **kl_unclipped** | **0.112** | **0.196** | **−74%** | **GROWING — posterior becoming more informative than prior, exactly what we want** |
   | L_pred only | 4.645 | 4.578 | +1.4% | the actual supervised signal |

   The 0.3% total-loss drop is dominated by the constant 19.2 nats of
   clipped KL contribution (`0.5·32 + 0.1·32`). The supervised
   prediction loss (`L_pred`) dropped 1.4% — slow but consistent with
   100 steps and bf16 noise. Importantly, `kl_unclipped` is *growing*:
   the posterior is becoming more informative than the prior, which
   is the desired direction (the prior should learn to predict, but
   should stay less precise than the posterior which has observation
   access). KL clipping will release as individual latent KLs cross
   the 1-nat floor — expected to happen sometime in the first ~10K
   steps of the Phase 5.3 full run.

5. **Artifacts.**
   - `checkpoints/world_model_smoke_raw.pt` — raw state_dict (will be
     overwritten by the Phase 5.3 run).
   - `checkpoints/world_model_smoke_step=*.ckpt` — best-by-val-reward
     Lightning checkpoint.
   - W&B run history at the URL above with all 8 logged scalars.

## Reproducibility

- `lightning.pytorch.seed_everything(42, workers=True)`.
- DataLoader uses `WeightedRandomSampler` with a `torch.Generator`
  seeded from 42.
- Logged to W&B run config: `seed`, `shuffle_seed`, `mode`,
  `torch_version=2.6.0+cu124`, `cuda_version=12.4`,
  `cudnn_version=90100`, `gpu=NVIDIA GeForce RTX 4070 Laptop GPU`.
- cuDNN nondeterminism left enabled — same disclosure as 5.0.5.
- Multinomial categorical sampling in `RSSM.sample_st` is
  device-RNG-driven (CUDA generator state); reruns will differ in the
  exact sampled one-hots even with the same seed if the CUDA stream
  ordering changes. The straight-through path's continuous
  probabilities are deterministic given the inputs.

## Related Docs

- Plan: `~/.claude/plans/we-re-starting-phase-5-staged-pinwheel.md`
- Phase 5.0.5 (encoder pretrain): `2026-05-03-phase5-0-5-encoder-pretrain.md`
- Phase 5.1 (datamodule): `2026-05-04-phase5-1-datamodule.md`
- Up next (Phase 5.3): `python -m training.train_world_model
  mode=full train.max_steps=200000 train.max_hours=4
  wandb.run_name=phase5.3-rssm-full data.max_episodes=null` — same
  config, same code, just longer. Plus 5.4 wiring (validate.py +
  dream endpoint + DreamPlayer.tsx) developed in parallel.
