# crypto-dreamer · System Architecture

This document describes the crypto-dreamer system as it exists after the Phase 5.4 pivot. It is a time-invariant cold-start reference, not a changelog. If you are returning after a long absence, start here.

---

## 1 · Problem Statement

crypto-dreamer is a DreamerV3-derived stochastic-latent world model that learns the joint dynamics of BTC/USDT 1-minute spot prices and a discrete portfolio-allocation agent. The model ingests a 256-bar observation window of 15 engineered features, maintains a recurrent belief state through a Recurrent State-Space Model, and predicts forward return distributions at four horizons (1, 5, 15, and 30 bars), a per-step scalar reward distribution, and an episode-continuation probability. It does not predict point-valued prices, individual trade entries, order-book microstructure, or cross-asset correlations.

## 2 · Why a Stochastic-Latent World Model on Financial Data

The architecture rests on a single empirical observation: BTC 1-minute log-return standard deviations scale as sqrt(t).

| Horizon (bars) | Empirical std | Predicted by sqrt(t) × std_1 | Ratio |
| --- | --- | --- | --- |
| 1 | 0.00067 | — | — |
| 5 | 0.00148 | 0.00150 | 0.99 |
| 15 | 0.00254 | 0.00259 | 0.98 |
| 30 | 0.00357 | 0.00367 | 0.97 |

These numbers come from 1.05M rows of BTCUSDT 1m closes in `data/market.duckdb` (reproducible via `_quantile_check.py` in the repo root). The near-perfect sqrt(t) scaling means the unconditional price process behaves like a random walk in the first moment · the conditional mean of future returns, given past returns, carries negligible predictable structure. Any model that optimizes a mean-prediction loss on this data will converge to predicting zero (or the marginal mean), which is correct but useless.

Predictable structure lives in higher moments. Volatility clusters: high-volatility regimes persist for tens of minutes to hours before reverting. The conditional variance of returns at time t depends on the regime the market is in, not just on the deterministic history of past bars. This is the phenomenon that GARCH-family models exploit, and it is the reason crypto-dreamer uses a stochastic latent z_t alongside a deterministic state h_t.

The deterministic path h_t (a GRU hidden state of dimension 256) captures everything that is a deterministic function of past observations and actions · trend features, rolling statistics, portfolio state. The categorical latent z_t (32 independent categoricals over 32 classes each) captures the stochastic regime · the part of the market state that is not determined by history but that governs the distribution of future returns. The posterior q(z_t | h_t, x_t) infers the current regime from the observation; the prior p(z_t | h_t) predicts it from history alone. The KL divergence between them measures how much regime information the observation carries beyond what history predicts.

If z_t has no stochastic job · if h_t alone can predict the target well enough · then the posterior collapses onto the prior, KL goes to zero (or to the free-bits floor), and the architecture degenerates into a deterministic autoregressive model with a dead latent. This is exactly what happened with the original feature-reconstruction decoder (see ADR-000 in Section 12): 15-dim normalized features at 1-minute resolution have near-zero conditional entropy given a 256-bar history, so h_t reconstructed them without needing z. The forward-distribution head (Section 6) gives z_t a genuinely stochastic job: predicting the shape of a conditional return distribution that varies by regime.

## 3 · Observation Space

Each observation is a 256-bar × 15-feature window. The 256 bars are consecutive 1-minute OHLCV candles from `data/market.duckdb`, which stores raw Binance klines with schema (`symbol`, `interval`, `ts`, `open`, `high`, `low`, `close`, `volume`, `quote_volume`, `trades`). The 15 features are computed on the fly: 12 market features derived from price/volume history, plus 3 portfolio-state scalars broadcast across the window. All values are clipped to [-10, +10] and NaN-filled to 0.

### Market features (12)

Computed in `envs/spot_btc.py` (lines 228-252). Rolling statistics use `pandas.rolling(w, min_periods=1)` with NaN fill to 0.

| # | Name | Definition | Normalization |
| --- | --- | --- | --- |
| 1 | `log_ret` | ln(close_t / close_{t-1}) | Direct (clipped) |
| 2 | `vol_5` | rolling std of `log_ret`, window 5 | Direct |
| 3 | `vol_15` | rolling std of `log_ret`, window 15 | Direct |
| 4 | `vol_60` | rolling std of `log_ret`, window 60 | Direct |
| 5 | `rsi_14` | (RSI_14 - 50) / 50 | EWM with alpha = 1/14, centered |
| 6 | `macd` | (EMA_12 - EMA_26) / close | Price-normalized |
| 7 | `vol_z` | (volume - mean_60) / std_60 | Z-score, 60-bar window |
| 8 | `hl_range` | (high - low) / close | Price-normalized |
| 9 | `close_norm` | (close - mean_60) / std_60 | Z-score, 60-bar window |
| 10 | `ret_5` | rolling sum of `log_ret`, window 5 | Direct cumulative |
| 11 | `ret_15` | rolling sum of `log_ret`, window 15 | Direct cumulative |
| 12 | `ret_60` | rolling sum of `log_ret`, window 60 | Direct cumulative |

### Portfolio features (3)

Appended per-step from the agent's portfolio state and broadcast across the 256-row window.

| # | Name | Definition | Range |
| --- | --- | --- | --- |
| 13 | `realized_alloc` | (BTC_value) / equity | [0, 1] |
| 14 | `cash_ratio` | 1 - realized_alloc | [0, 1] |
| 15 | `log_equity` | ln(equity / 10000) | unbounded |

The combined (256, 15) array is the input tensor for each trajectory step. In training, the datamodule (`training/datamodule.py`) constructs batch tensors of shape (B, T, 256, 15) where B = 32 and T = 48. The datamodule partitions data by calendar month, reserving the last 15% of each month (by day-of-month) for validation. Training uses a `WeightedRandomSampler` with inverse-frequency weights per month to equalize month representation.

## 4 · iTransformer Encoder

The encoder is an inverted transformer (Liu et al., "iTransformer," ICLR 2024) implemented in `models/encoder.py:35-121`. Standard transformers attend across time steps; iTransformer attends across variables. Each of the 15 input variables' full 256-bar series is projected into a single d_model-dimensional token by a shared linear layer (`input_proj`, line 54), then a learnable per-variable embedding is added (`var_embed`, line 57), and the resulting 15 tokens are processed by a standard pre-norm TransformerEncoder (lines 60-69). Attention cost scales with the number of variables (15), not the sequence length (256), making the architecture efficient for long time-series windows.

Architecture parameters: 4 layers, 4 attention heads, d_model = 128, feedforward = 512, dropout = 0.1, GELU activation, batch_first = True, norm_first = True (for bf16 stability).

The encoder forward pass (`forward`, lines 115-121) takes input (B, 256, 15), rearranges to (B, 15, 256), projects to (B, 15, 128) via `input_proj`, adds per-variable embeddings, and produces (B, 15, 128) after the transformer layers. The world model then mean-pools over the variable dimension (`encode_obs`, `models/world_model.py:115-118`) to produce a single x_t in R^128 per observation, which the RSSM posterior conditions on.

### MAE pretraining

The encoder is pretrained with a masked autoencoder strategy in Phase 5.0.5. During pretraining the encoder processes only the 12 market features (no portfolio scalars), and a throwaway MLP decoder (`models/mae_decoder.py:16-36`) reconstructs the masked bars. The decoder is a shared-weight MLP (128 -> 128 -> 128 -> 256, GELU) applied independently per variable, then rearranged from (B, F, T) to (B, T, F). Only the encoder weights are kept.

The pretrained checkpoint lives at `checkpoints/encoder_mae_full_raw.pt`. When loaded into the world model (where n_vars = 15), `_load_mae_checkpoint` (encoder.py:74-113) copies `input_proj` weights, the first 12 rows of `var_embed`, and all transformer encoder layer parameters. The 3 additional portfolio-feature embeddings (indices 12-14) are left at random initialization. The encoder architecture and MAE pretraining are unchanged by the Phase 5.4 pivot (see ADR-004).

## 5 · RSSM Core

The Recurrent State-Space Model (`models/rssm.py:31-154`) is the temporal backbone. At each step t, the model state is a pair (h_t, z_t):

h_t in R^256 is the deterministic state, produced by a GRU cell. It summarizes the entire causal history of observations and actions. z_t in {0,1}^(32x32) is the stochastic state · 32 independent categorical variables, each over 32 classes, represented as a flattened 1024-dim one-hot vector.

The transition at each step proceeds in three stages.

**Stage 1 · GRU update.** The previous stochastic state z_{t-1} and the previous action embedding (from `ActionEmbed`, `models/action_embed.py:11-13`, which maps the 5 discrete portfolio allocation targets {0%, 25%, 50%, 75%, 100%} to 32-dim learned embeddings) are concatenated and projected through `pre_gru` (lines 50-53, a single linear layer with GELU) to produce the GRU input. The GRU cell (line 54) updates the deterministic state: h_t = GRU(pre_gru([z_{t-1}, a_{t-1}]), h_{t-1}). Verified against `models/rssm.py:50-93` on 2026-05-27.

**Stage 2 · Prior and posterior.** Two distributions over z_t are computed from h_t. The prior p(z_t | h_t) is produced by `prior_head` (lines 56-60): a two-layer MLP (256 -> 256 -> 1024, GELU) that maps h_t to logits reshaped to (B, 32, 32). The posterior q(z_t | h_t, x_t) is produced by `posterior_head` (lines 61-65): a two-layer MLP (256 + 128 -> 256 -> 1024, GELU) that maps the concatenation [h_t, x_t] to the same logit shape. During imagination (planning without observations), z_t is sampled from the prior. During training, z_t is sampled from the posterior.

**Stage 3 · Sampling.** Sampling uses straight-through gradients (`sample_st`, lines 102-115). The forward pass computes softmax probabilities, mixes them with a uniform distribution at weight 0.01 (the `unimix` parameter, which prevents any class probability from reaching exactly zero and thus prevents infinite KL), draws a multinomial sample, and converts it to a one-hot vector. The backward pass passes gradients through the softmax probabilities as if sampling were an identity operation: the returned tensor is `one_hot + (probs - probs.detach())`.

KL divergence between posterior and prior is computed per-latent (`categorical_kl`, lines 117-135) as KL(q || p) = sum_k q_k * log(q_k / p_k) for each of the 32 latent dimensions, producing a (B, 32) tensor. Both q and p have unimix applied before the KL computation. Free-bits clipping (`free_bits_kl`, lines 137-153) averages the per-latent KL across the batch, clamps each latent's contribution at a floor of 1.0 nat, and sums over the 32 latents. The total floor is therefore 32 nat.

When the `is_first` flag is set (first step of an episode), both h and z are reset to zeros (`step`, lines 87-90).

## 6 · Forward-Distribution Head

The forward-distribution head replaces the deleted feature-reconstruction decoder (see ADR-000 in Section 12). It predicts categorical probability distributions over discretized log-returns at four horizons: {1, 5, 15, 30} bars into the future. The head will be implemented as `ForwardDistributionHead` in `models/heads.py`.

Input: the concatenation of h_t and z_t, denoted feat in R^1280 (256 + 32 x 32). This is the same input shape used by the reward and continue heads.

For each horizon h, the head produces a 41-dim logit vector. The 41 bins are evenly spaced across a symmetric range [-R_h, +R_h] where R_h is the horizon-specific half-range:

| Horizon (bars) | Half-range R_h | Bin width (2R_h / 40) | Coverage margin over 99.9% quantile |
| --- | --- | --- | --- |
| 1 | 0.005 | 0.000250 | 1.28x |
| 5 | 0.010 | 0.000500 | 1.16x |
| 15 | 0.018 | 0.000900 | 1.20x |
| 30 | 0.025 | 0.001250 | 1.21x |

The ranges are derived from the empirical 99.9% quantiles of BTCUSDT 1m log-returns at each horizon (q999: 1-bar 0.0039, 5-bar 0.0087, 15-bar 0.0149, 30-bar 0.0206; reproducible via `_quantile_check.py`), rounded to clean numbers for engineering convenience. They do not follow exact sqrt(h) scaling because the rounding introduces small deviations, and the empirical tails of BTC returns are fatter than Gaussian at shorter horizons.

The ~1.2x margin over the 99.9% quantile means roughly 0.1% to 0.2% of forward-return mass per horizon falls outside the bin range and clips into the edge bins. This is intentional · the model learns "extreme move" as the edge-bin category rather than wasting bin resolution on the precise magnitude of rare tail events.

Bin-edge formula for horizon h:

```
bin_centers_h[k] = -R_h + k * (2 * R_h / 40)    for k = 0, 1, ..., 40
bin_width_h      = 2 * R_h / 40
```

Targets are encoded using two-hot encoding, the same mechanism implemented in `RewardHead.two_hot_encode` (`models/heads.py:70-81`) generalized to per-horizon ranges. For a target log-return v at horizon h:

1. Clamp: v' = clamp(v, -R_h, +R_h).
2. Normalized position: pos = (v' + R_h) / bin_width_h, yielding a value in [0, 40].
3. Lower bin index: idx_lo = floor(pos), clamped to [0, 39].
4. Interpolation weights: w_lo = 1 - frac(pos), w_hi = frac(pos).
5. Target vector: target[idx_lo] = w_lo, target[idx_lo + 1] = w_hi, all other entries zero.

This produces a probability vector summing to 1.0 that assigns weight to exactly two adjacent bins proportional to the target's position between their centers. Cross-entropy against this target provides smooth gradients even when the target falls between bin centers, unlike a hard one-hot encoding which would produce zero gradient for all but one bin.

The loss for each horizon is the cross-entropy between the predicted categorical distribution (log-softmax of logits) and the two-hot target:

```
CE_h = -sum_k target_h[k] * log_softmax(logits_h)[k]
```

The total forward-distribution loss is the unweighted sum across horizons:

```
L_forward = CE_1 + CE_5 + CE_15 + CE_30
```

Equal weighting is a deliberate bootstrap choice (ADR-001 in Section 12). Per-horizon losses are logged separately to W&B (`loss_forward_1`, `loss_forward_5`, `loss_forward_15`, `loss_forward_30`) to inform potential future reweighting.

The forward-return targets themselves (the log-return of the close price from bar t to bar t+h for each horizon h) will be added to the datamodule output as a `forward_returns: (B, T, 4)` tensor in `training/datamodule.py`.

## 7 · Reward Head

The reward head (`models/heads.py:42-91`) predicts a categorical distribution over scalar per-step rewards. The architecture is a two-layer MLP with GELU activations (1280 -> 256 -> 256 -> 41) that produces 41-bin logits. Bin centers are 41 evenly spaced values from -0.2 to +0.2 (bin width 0.01), registered as a non-persistent buffer (`bin_centers`, lines 63-65).

The reward at each step is computed by the environment (`envs/spot_btc.py`, lines 138-169) as:

```
reward = ln(equity_t / equity_{t-1}) - 0.05 * turnover
```

where turnover = |delta_value| / equity_before, accounting for taker fees (0.1%, `TAKER_FEE = 0.001`) and slippage (2 bps default). Target encoding and loss use the same two-hot / cross-entropy mechanism described in Section 6. The `predict` method (lines 88-90) returns the expected value under the predicted distribution for inference: E[r] = sum_k softmax(logits)[k] * bin_centers[k].

## 8 · Continue Head

The continue head (`models/heads.py:93-107`) predicts whether the episode continues at the next step. It is a two-layer MLP with GELU activation (1280 -> 128 -> 1) that produces a single logit per sample, trained with binary cross-entropy against a boolean target. The episode terminates early when equity drops below 50% of the initial 10,000 USDT (`envs/spot_btc.py`, line 173).

## 9 · Loss Composition

The total training loss combines prediction losses and KL regularization (`models/world_model.py:204-215`):

```
L = L_forward + L_reward + L_continue + 0.5 * L_dyn + 0.1 * L_rep
```

where:

L_forward is the forward-distribution cross-entropy summed over four horizons (Section 6). L_reward is the reward cross-entropy (Section 7). L_continue is the binary cross-entropy (Section 8).

L_dyn = KL(sg(posterior) || prior) is the dynamics loss, which trains the prior to predict the posterior. The stop-gradient (sg) on the posterior means this loss only updates the prior pathway (the `prior_head` MLP and the GRU).

L_rep = KL(posterior || sg(prior)) is the representation loss, which trains the posterior to stay close to the prior. The stop-gradient on the prior means this loss only updates the encoder and the `posterior_head` MLP.

Both KL terms use free-bits clipping with a floor of 1.0 nat per latent dimension (`free_bits_kl`, `models/rssm.py:137-153`). The procedure: average the per-latent KL across the batch to get a (32,) vector, clamp each entry at >= 1.0, then sum over the 32 latents. The total floor is 32 nat. This prevents the KL from collapsing to zero in early training before the posterior has learned useful structure, while allowing it to grow naturally once z_t finds regime information.

The coefficients 0.5 (dynamics) and 0.1 (representation) follow DreamerV3. The asymmetric weighting ensures the prior is updated more aggressively than the posterior, preventing a failure mode where the posterior drifts far from the prior and imagination becomes unreliable.

Per-step losses are averaged over T - burn_in active steps (burn_in = 5 by default). The first 5 trajectory steps are excluded from the loss because the RSSM hidden state is freshly reset and uninformative; including them would inject noise into the gradient.

The raw, unclipped KL (kl_unclipped) is logged separately from the clipped losses. The difference (loss_dyn - kl_unclipped), clamped at zero, is logged as `kl_clip_excess` · when this drops to zero, the free-bits floor is no longer active and the KL is "released."

**The load-bearing prediction of this architecture:** KL_unclipped rises above the 32-nat free-bits floor during training, because z_t has a genuinely stochastic job (predicting forward return distributions conditioned on regime) that h_t alone cannot solve. If KL does not release above the floor, the architecture is wrong for this data and the project should be re-evaluated. This is not a tuning issue · it is a statement about whether the data has exploitable stochastic structure at the regime level.

## 10 · Training Protocol

**Status note (2026-06-10):** this section describes the WORLD-MODEL
training protocol and is retained as historical context. Stage 2 /
Phase 5.5 is BLOCKED per ADR-007 (Section 12) · the active line of work
is the model-free PPO baseline.

Training proceeds in two stages gated by validation diagnostics.

**Stage 1 · Diagnostic (30k steps).** A short run to verify that loss curves are healthy and KL releases from the free-bits floor. Configuration: T = 48 trajectory length, B = 32 batch size, bf16-mixed precision, AdamW optimizer with lr = 1e-4, weight_decay = 1e-6, 1000-step linear LR warmup, gradient clipping at 1000 (DreamerV3-style permissive clipping that catches only catastrophic gradient explosions, not normal training dynamics). Validation runs every 2500 steps over 100 batches (~3200 samples). Checkpoints saved every 5000 steps. T = 48 is a deliberate relaxation from T = 64 to amortize WDDM kernel-launch overhead on Windows; future Linux training may revert to T = 64.

**Stage 2 · Full run (100k steps).** Advanced to only after Stage 1 passes the validation gates (Section 11). Resumes from the Stage 1 checkpoint using Lightning's `Trainer.fit(ckpt_path=...)`, which restores model state, optimizer moments, LR scheduler position, and global step. The DataModule's `WeightedRandomSampler` does not preserve position across resume, so the data sequence post-resume differs from a hypothetical unbroken run · this is acceptable because training samples are drawn from a stochastic distribution.

**Operational practices.** A heartbeat callback writes one timestamped line per 100 training batches to `logs/heartbeat_*.log`, independent of W&B, to provide crash detection without depending on network connectivity. On the 4070 laptop, Modern Standby must be disabled before any overnight run via `powercfg /change standby-timeout-ac 0` to prevent WDDM-triggered hibernation mid-training (documented in the Phase 5.3 stall postmortem, `docs/implementations/2026-05-04-phase5-3-rssm-full-train.md`). torch.compile is disabled because Triton is unavailable in the Windows venv.

All training configuration lives in `configs/world_model.yaml`.

## 11 · Validation Gates

**Status note (2026-06-10):** these are the WORLD-MODEL gates, retained
as historical context. Both 30k diagnostics failed Gate 1 (ADR-006
Outcome 3) and Phase 5.5 is BLOCKED per ADR-007 (Section 12); the
active gate set is ADR-007's pre-registered baseline gate.

The following gates are evaluated after the 30k diagnostic run. All must pass before advancing to the 100k full run.

**Gate 1 · KL release.** KL_unclipped must exceed 32 nat by step 20k. This is the load-bearing gate. It confirms that z_t is encoding information the prior cannot predict from action history alone. If KL remains pinned at or below 32 nat (as happened in Phase 5.3 with the feature-reconstruction decoder, which finished at 25.7 nat), the forward-distribution target has insufficient stochastic structure and the pivot has failed.

**Gate 2 · Forward-distribution loss below marginal baseline.** The marginal baseline at horizon h is the cross-entropy of the constant-predictor that always outputs the empirical marginal forward-return histogram (computed once over the training set), evaluated on the validation set. Formally: let p_h(k) be the empirical frequency of bin k for horizon h over training data, and q_h(k) be the empirical frequency over validation data. Then baseline_h = -sum_k q_h(k) * log(p_h(k) + epsilon) with epsilon = 1e-9 for numerical safety. The total marginal baseline is the sum across the four horizons. The model's val/loss_forward_dist at step 20k of the diagnostic must be strictly less than this baseline. A model that fails to beat it is predicting no conditional structure beyond the unconditional distribution and has effectively learned nothing useful from the encoder or RSSM.

Drift band. The marginal baseline reported in `docs/findings/2026-05-27-marginal-baseline.md` is 8.8632. The drift band · H(q,p) - H(q) summed across horizons · is approximately 0.015 nat. A val/loss_forward_dist value between 8.85 and 8.86 is effectively a tie with the baseline · the model can achieve this band by exploiting the mild train-to-val distribution mismatch without learning conditional structure. Gate 2 is genuinely passed only when val/loss_forward_dist < 8.85 at step 20k of the diagnostic. Values in [8.85, 8.86] are reported as inconclusive rather than passing.

**Gate 3 · Reward NLL stability.** Validation reward NLL should be approximately 0.48, consistent with prior runs (Phase 5.3 achieved val/loss_reward = 0.478). A significant regression would indicate the pivot introduced a bug in the shared RSSM or reward pathway.

If all gates pass, proceed to the 100k full run (Phase 5.5 · now BLOCKED per ADR-007; this branch was never taken). If Gate 1 fails, revisit the architecture per ADR-003's contingency options: prior capacity restriction, KL warmup schedule, or model paradigm change · this is the branch the project took (ADR-006 -> Outcome 3 -> ADR-007). If Gate 2 fails with Gate 1 passing, the problem is in the head architecture or the target computation, not the RSSM. If Gate 3 fails, debug the shared components (encoder, RSSM, optimizer configuration).

## 12 · Architectural Decision Records

### ADR-001 · Equal loss weighting across forward horizons at bootstrap

**Status:** Accepted · revisit post-diagnostic.

**Context:** The forward-distribution head predicts four horizons with potentially different learning dynamics. Shorter horizons have narrower bin ranges and may converge faster; longer horizons may dominate the loss early if their initial cross-entropy is higher due to wider bin ranges.

**Decision:** Weight each horizon's cross-entropy loss at 1.0 in the initial 30k diagnostic. Per-horizon losses are logged separately to W&B (`loss_forward_1`, `loss_forward_5`, `loss_forward_15`, `loss_forward_30`).

**Consequences:** After the 30k run, inspect per-horizon loss curves. If one horizon dominates total loss by more than 3x, write ADR-005 proposing a reweighting scheme with concrete evidence from the loss curves. If all horizons converge at similar rates, accept equal weighting permanently.

### ADR-000 · Feature-reconstruction decoder (superseded)

**Status:** Superseded by ADR-002 as of Phase 5.4 pivot.

**Context:** The original DreamerV3-style `DecoderHead` (`models/heads.py:23-39`) reconstructed the current bar's 15-dim feature row via MSE loss. The Phase 5.3 diagnostic (W&B run `kk3mzb3k`) finished 30k steps with val/loss_decoder at 0.0021, KL pinned at the 32-nat free-bits floor, and kl_unclipped at 25.7 nat (below the floor). The deterministic path h_t reconstructed features without needing z_t.

**Root cause:** 15-dim normalized financial features at 1-minute resolution have near-zero conditional entropy given a 256-bar history. The 12 market features are deterministic functions of the price/volume window that h_t already encodes through the GRU's sequential processing. The 3 portfolio features are deterministic functions of the agent's action history. There is no stochastic residual for z_t to model.

**Decision:** Delete `DecoderHead` and all decoder-related machinery in `models/world_model.py` (target computation at line 136, `loss_dec_sum` accumulation, head instantiation at line 99). Do not attempt to augment the reconstruction target with noise or additional features · the problem is structural, not a matter of target difficulty.

### ADR-002 · Forward-distribution head with empirically-sized bins

**Status:** Accepted.

**Context:** Replacement for the deleted feature-reconstruction decoder. The forward-distribution target gives z_t a stochastic job: predicting the shape of conditional return distributions that vary by volatility regime.

**Decision:** Four horizons {1, 5, 15, 30} bars, 41 bins each, symmetric ranges +/-{0.005, 0.010, 0.018, 0.025} log-return. Ranges are sized from the 99.9% empirical quantile of BTCUSDT 1m closes over 1.05M rows (reproducible via `_quantile_check.py`). Two-hot encoding reuses the mechanics from `RewardHead.two_hot_encode` (`models/heads.py:70-81`), generalized to per-horizon ranges.

**Consequences:** Forward-return targets must be added to the datamodule output as a `forward_returns: (B, T, 4)` tensor in `training/datamodule.py`. The dream rollout visualization in `dashboard/src/components/internals/DreamPlayer.tsx` will need to be re-pointed at distribution outputs in a future phase.

### ADR-003 · Free-bits floor unchanged at 1.0 nat per latent

**Status:** Accepted.

**Context:** The 1.0 nat floor coincided with posterior collapse in Phase 5.3, but the collapse was caused by the target (feature reconstruction) having insufficient stochastic structure, not by the floor value. With the new forward-distribution target, z_t has a genuinely stochastic job, and KL is expected to release naturally above the floor.

**Decision:** Keep free_bits = 1.0 in the initial 30k diagnostic. The floor acts as a regularizer that prevents the posterior from immediately diverging from the prior in early training; once z_t finds useful regime information, KL rises above the floor organically.

**Contingency:** If KL still does not release after 30k steps with the forward-distribution target, write ADR-006 evaluating one of: (a) prior capacity restriction · reduce `prior_head` hidden dim to force the prior to be less expressive, the canonical DreamerV3 fix; (b) KL warmup schedule · start with zero KL weight and ramp to full over 5k steps; (c) fundamental re-evaluation of the model paradigm for this data domain. Do not preemptively add a KL warmup · the diagnostic must first establish whether the target change alone is sufficient.

### ADR-004 · MAE encoder pretraining preserved across pivot

**Status:** Accepted.

**Decision:** Keep `checkpoints/encoder_mae_full_raw.pt` and the Phase 5.0.5 MAE pretraining protocol unchanged. The encoder maps raw features to embeddings; the downstream head it feeds is changing, but the input shape (256 bars x 15 features) and output shape (15 x 128 tokens, mean-pooled to 128-dim) are identical.

**Rationale:** The pretrained encoder already produces good feature representations (Phase 5.0.5 achieved train/mse_step = 0.1007). The pivot changes what consumes those representations, not how they are produced. Retraining the encoder would waste compute and introduce a confounding variable in the diagnostic.

### ADR-005 · Gate 2 inconclusive-band resolution

**Status:** Not instantiated · number intentionally vacant.

Reserved in `ROADMAP.md` Section 2 (the Gate Decision Point) for the
branch where Gate 1 passes AND Gate 2 lands in the [8.85, 8.86]
inconclusive band · the case where KL releases but the forward loss is
a statistical tie with the marginal baseline. That branch did not
occur: the Phase 5.4 30k diagnostic (run `1rq8d8u5`) failed Gate 1
outright (kl_unclipped 25.95 < 32), so the project took the
Gate-1-failure branch (ADR-006) rather than the Gate-2-ambiguous one.
This number is left vacant as a record of the decision tree · the gap
documents which fork the project actually took.

# ADR-006 · Gate 1 Failure Contingency · Prior Capacity Restriction

This ADR belongs in `docs/design/ARCHITECTURE.md` Section 12 alongside
ADR-000 through ADR-004 (paste it there to keep the ADR namespace in
one place, per the convention that ADRs stay inline until the count
exceeds ~10). It is delivered standalone for review.

**Status:** Accepted · executed 2026-05-31 · Outcome 3 obtained
(Hypothesis A falsified) · committed `a3941d3` · results in
`docs/findings/2026-05-31-adr006-linear-prior-results.md` · follow-on
decision recorded in ADR-007.

## Context

The Phase 5.4 forward-distribution pivot was falsified by the 30k
diagnostic (run `1rq8d8u5`, results in
`docs/findings/2026-05-30-phase5-4-diagnostic-results.md`). Measured
on the completed 30k checkpoint over 40 validation batches:

- **Gate 1 (KL release): FAIL.** `kl_unclipped = 25.95` below the
  32-nat free-bits floor; `loss_dyn` and `loss_rep` floor-pinned at
  32.0007. The latent never carried information the prior could not
  already predict. Flat at ~26 from pre-flight through 30k · no
  release.
- **Gate 2 (forward loss vs baseline): FAIL.** Per-horizon sum 9.7564
  vs the 8.8632 marginal baseline · the model predicts forward returns
  worse than the unconditional marginal. Per-horizon losses rise
  monotonically with horizon (2.21 / 2.39 / 2.52 / 2.63); validation
  forward loss rose after ~6k while train stayed flat (overfitting to
  unpredictable conditional means).
- **Gate 3 (reward NLL): PASS.** 0.4778, matching Phase 5.3. Shared
  RSSM and reward pathway intact; the alignment trace confirms the
  failure is architectural, not a wiring bug.

**The decisive context is the two-target convergence.** crypto-dreamer
has now collapsed the latent under two architecturally opposite
targets: feature reconstruction (Phase 5.3, target too easy, `h_t`
solved it alone) and forward-return distribution (Phase 5.4, target
too noisy, nobody solved it). Two opposite failure mechanisms, one
shared outcome. This narrows the cause to two non-exclusive
hypotheses:

- **Hypothesis A · over-expressive prior.** The `prior_head` MLP
  (256 -> 256 -> 1024) is expressive enough to predict the posterior's
  output from `h_t` alone, so the posterior's marginal information
  contribution goes to zero and KL collapses to the floor · regardless
  of target. This is the canonical DreamerV3 posterior-collapse mode.
  The observed KL pattern (a brief spike to ~31 at step 5 then decay to
  ~26 as the prior trains) is consistent with the prior progressively
  out-competing the posterior.
- **Hypothesis B · no exploitable signal.** BTC 1-min returns are
  near-martingale in the first moment (sqrt(t) std scaling,
  `ARCHITECTURE.md` Section 2), so there may be little conditional
  stochastic structure at the regime level for a latent to encode even
  if the prior allowed it. Gate 2's sub-baseline forward loss · the
  model cannot beat the marginal · is direct evidence for this.

A and B are not mutually exclusive and the current data cannot
separate them, because no standard collapse remedy has been tried.

## Decision

**Run a single disambiguating experiment before any further
architectural commitment: restrict the prior to a linear map and
re-run the 30k diagnostic.**

Concretely, change `prior_head` in `models/rssm.py` from the current
two-layer MLP (256 -> 256 -> 1024 with GELU) to a single linear layer
(256 -> 1024, no hidden layer, no activation). Everything else held
fixed · same encoder, posterior, free-bits floor (1.0), coefficients
(coef_dyn 0.5, coef_rep 0.1), forward-distribution head and target,
T=48, batch 32, lr 1e-4, 30k steps. Only the prior's capacity changes.

**Rationale for linear (aggressive) over a milder restriction.** The
goal of this experiment is disambiguation, not tuning. A linear prior
maximally handicaps the prior's ability to out-predict the posterior,
giving the sharpest possible read on whether KL CAN release. A milder
restriction (e.g. 256 -> 64 -> 1024) risks an ambiguous middle result.
A linear prior is the cleanest single-run test of Hypothesis A. If the
latent releases even when the prior is a bare affine map, A is
confirmed and a follow-up run can find the right middle capacity; if
it will not release even then, that is the strongest available
evidence against A and for B.

**Three outcomes, each diagnostic:**

1. **KL releases (kl_unclipped > 32) AND Gate 2 improves below
   baseline.** Hypothesis A was the problem; the latent is alive and
   useful. Continue the forward-distribution pivot with the restricted
   prior; proceed toward a gated Phase 5.5. Write the result as a
   follow-up finding and, if needed, ADR-007 tuning prior capacity.

2. **KL releases BUT Gate 2 still fails (forward loss stays at or
   above baseline).** The latent now carries information, but that
   information does not help predict returns · confirming Hypothesis B
   for the current target. The latent is alive but the TARGET is
   wrong. Next step becomes target redesign: predict conditional
   VOLATILITY / scale (which volatility-clustering makes genuinely
   predictable) rather than a location-bearing return distribution.
   This is contingency option (d), to be written as its own ADR.

3. **KL still will not release even with a linear prior.** Strongest
   available evidence that the data lacks regime-level stochastic
   structure a latent variable model can capture in this setup.
   Escalate to contingency option (c): drop the world-model paradigm
   and train a model-free RL agent (PPO/SAC) on the same observation
   and reward, treated as the honest baseline. To be written as its
   own ADR with the model-free comparison as the deciding experiment.

## Contingency options NOT chosen now (documented for the branch tree)

- **Option (b) · KL warmup schedule** (ramp KL weight from 0 over ~5k
  steps). Deliberately deferred. The diagnostic must first establish
  whether a capacity fix (a) alone releases KL; layering a warmup on
  top now would confound which intervention mattered. Revisit only if
  the linear-prior run is itself ambiguous. Backlog item exists.
- **Option (d) · target redesign to conditional volatility.** The
  likely next step if outcome 2 obtains. Not run now because it is
  premature until we know the latent CAN engage (outcome 1 or 2
  distinguishes this).
- **Option (c) · model-free RL paradigm change.** The honest endpoint
  if outcome 3 obtains. Not run now because declaring the world-model
  paradigm dead without trying the canonical collapse fix would be
  premature.

## Consequences

- **Phase 5.5 (100k full run) is blocked** until a diagnostic passes
  Gate 1 and Gate 2. The roadmap's State B freeze posture holds.
- **OMI freeze handling.** If the linear-prior experiment runs and is
  evaluated before the OMI freeze, the thaw state is "outcome N
  obtained, next experiment is X." If it does not run in time, the
  freeze state is "ADR-006 written, run the linear-prior experiment
  first on thaw" · a complete, decision-ready state either way.
- **Gate reads come from the checkpoint, not W&B.** Per the diagnostic
  results doc Section 5, W&B logging silently failed mid-run on
  `1rq8d8u5`. Tonight's experiment and all unattended runs must read
  gates via the checkpoint-eval path (`_gate.py` / future
  `scripts/eval_gates.py`) and treat the heartbeat as the
  authoritative completion signal. Do not trust the charts.
- **`prior_head` is the only model change.** `models/rssm.py` is
  modified for the prior layer; no change to the encoder, posterior,
  heads, datamodule, or loss composition. This keeps the experiment a
  clean single-variable test.

## Verification (for the experiment when it runs)

Same precondition gate as brief 4.1 (powercfg standby/monitor = 0, AC
confirmed, CUDA available, 1000-step pre-flight finite). Run 30k. Read
`kl_unclipped`, the per-horizon forward sum, and `loss_reward` from the
final checkpoint via the checkpoint-eval path. Classify into outcome
1, 2, or 3 above. `loss_reward` should remain ~0.48 (a regression would
indicate the prior change broke the shared pathway · unlikely, since
only the prior is touched).

---

### ADR-007 · Model-free PPO baseline · world-model paradigm dropped (option c)

**Status:** Proposed · 2026-06-10. The operator flips this to Accepted on
commit. The pre-registered evaluation gate below becomes binding at that
moment; thresholds are not adjusted afterward. Any ambiguity discovered
later is resolved by an operator ruling recorded here as an amendment
BEFORE the gate is read · never after.

**Context.** The ADR-006 linear-prior disambiguation experiment ran to
completion and was committed in `a3941d3` (2026-05-31; results in
`docs/findings/2026-05-31-adr006-linear-prior-results.md`). Four-decimal
values below are quoted from the operating briefing
(`docs/briefings/2026-06-10-adr007-model-free-baseline-briefing.md`
Section 1), which records the same checkpoint eval at full precision; the
findings doc rounds to 3 decimals. Components and sums are independently
rounded, so the per-horizon values below sum to 9.6141 at 4 dp while the
directly measured sum is 9.6142 · a rounding artifact, not an arithmetic
error. Gates were re-derived from the evidence checkpoint
`checkpoints/world_model_diagnostic_step=30000-v2.ckpt` via
`scripts/eval_gates.py` (40 val batches, seed 42, clean load · 0 missing /
0 unexpected keys) · never from W&B (W&B silently desynced at ~step 612 on
run `1rq8d8u5`, per the findings doc Section 0, and is not a gate source):

- Gate 1 · `kl_unclipped` = 26.31 vs the 32-nat free-bits floor -> FAIL.
  No KL release; +0.36 vs the MLP prior's 25.95; seed-stable 26.31 +/-
  0.002 across seeds 42/0/123. `loss_dyn`/`loss_rep` floor-clipped at
  32.0055 · the latent is pinned.
- Gate 2 · forward loss sum = 9.6142 vs the 8.8632 marginal baseline ->
  FAIL (h1 2.2152 / h5 2.3494 / h15 2.4943 / h30 2.5552) · the model
  still predicts forward returns worse than the unconditional marginal.
- Gate 3 · reward NLL = 0.4776 ~= 0.478 -> PASS · wiring and the shared
  reward pathway intact; the failure is architectural, not a bug.

This is **Outcome 3** of ADR-006's pre-registered decision tree: the KL
will not release even against a maximally handicapped (bare affine)
prior. **Hypothesis A (over-expressive prior) is falsified.** The
supported interpretation is **Hypothesis B**: BTC 1-min data lacks
regime-level stochastic structure that a latent-variable world model can
exploit in this setup. The stochastic latent has now collapsed under two
architecturally opposite targets (Phase 5.3 feature reconstruction · too
easy · `h_t` solved it alone; Phase 5.4 forward-return distribution · too
noisy · nobody solved it) and across the full prior-capacity range
(2-layer MLP down to bare linear).

**Decision (operator-ratified).** Adopt ADR-006 contingency option (c):

- DROP the world-model / latent paradigm for this line of work.
- STAND UP a model-free RL baseline · **PPO** on the discrete 5-action
  space (SAC-discrete acceptable only as a documented fallback, operator
  approval required first) · trained on the SAME observations and the
  SAME reward, evaluated on the SAME held-out calendar partition, as the
  honest apples-to-apples comparison.
- The c-vs-d fork was the operator's call, ratified at the ADR-007 review
  gate. No agent re-opens it.

Comparability constraints (what "unchanged" means, binding for the
implementation):

- **Env** · `envs/spot_btc.py` as-is: Discrete(5) target allocation
  {0, 25, 50, 75, 100}% of equity, 0.1% taker fee, linear slippage
  (2 bps default), reward = log-return - 0.05 x turnover, 1440-step
  episodes, 50%-equity termination guardrail. No dynamics, fee, slippage,
  or reward changes. Permitted additive-only interface changes:
  deterministic episode-start selection through `reset(options=...)` for
  evaluation, and partition-aware start sampling for training. Neither
  may alter `step()` semantics. `episode_steps` is an EXISTING
  constructor parameter (`envs/spot_btc.py:84`, default 1440): training
  and the agent/flat/random evaluation use 1440; the B&H comparator alone
  instantiates the env with `episode_steps=4320` for its one per-span
  episode · a constructor argument, not a code or `step()`-semantics
  change.
- **Data source (bound)** · every training AND evaluation env is
  constructed over the frozen snapshot `data/market_ro.duckdb`
  (`db_path` passed explicitly; the `DREAMER_DATA` environment variable
  must be unset or equal to it). The env's default `data/market.duckdb`
  is the live DB and is NOT acceptable for any ADR-007 run. Identity is
  asserted mechanically by Track B (G)(vi), not assumed.
- **Observations** · the env-native dict · `window` (256, 12) float32
  from the canonical `envs.spot_btc.compute_feature_block` (column order
  `FEATURE_NAMES`, append-only) + `portfolio` (3,). Identical feature
  content to what the world-model pipeline consumed (its (256, 15) tensor
  was the same 12 features with the 3 portfolio scalars broadcast · a
  packaging detail, not a feature difference). No new features, no
  changed normalization.
- **Training data** · PPO training rollouts are restricted to the train
  partition: an episode start `s` (kline row index) is valid iff
  `s >= 256` and every bar in `[s, s + 1440]` falls on a UTC
  day-of-month with `(day - 1) / days_in_month < 0.85`. The 256-bar
  observation lookback may cross partition boundaries · identical to the
  world-model convention, where only the trajectory rows were
  partition-pure. Carried over deliberately for comparability. The
  training loop writes every realized episode start index to
  `artifacts/adr007/train_starts_seed<NN>.json`; partition purity is
  verified mechanically both pre-run and post hoc per (G)(v) · it is a
  gate precondition, not a prose promise.
- **Training budget and run of record (pre-registered, anti-gaming)** ·
  2,000,000 env steps per seed, fixed in config before launch. The
  evaluated artifact is the FINAL checkpoint of each seed · no checkpoint
  shopping, no eval-driven early stopping, no train-until-pass · and the
  prohibition covers ACROSS-run shopping too: the SHA-256 of
  `configs/ppo_baseline.yaml` is recorded in
  `artifacts/adr007/run_log.md` before launch and ratified by the
  operator's explicit launch approval; the first completed 3-seed run
  after that freeze is the RUN OF RECORD, and the gate may be read only
  from its final checkpoints. Periodic intermediate checkpoints ARE
  saved during training (seed and step in the filename) for forensics
  and are explicitly GATE-INELIGIBLE; their existence never substitutes
  for a missing final checkpoint (operator amendment A2 · 2026-06-10).
  Any relaunch or config change, for any
  reason (including a crashed seed or a "trial" run), requires an
  operator amendment recorded BEFORE the new launch · GPU
  non-determinism makes even an identical-config relaunch a free reroll
  of the seed triple, so relaunches are never agent-discretionary.
  Before the official gate read, NO policy may be evaluated on any
  enumerated eval episode or any val-partition bar, with one pre-named
  exception: the smoke subset · the first enumerated episode
  (2024-05-28 00:00) plus the comparators on that same episode · whose
  results verify harness plumbing only and select nothing. Every
  invocation of `scripts/eval_baseline_gates.py` appends an entry
  (timestamp, checkpoint hash, episode set, purpose) to
  `artifacts/adr007/run_log.md`, so off-the-books gate reads are
  visible. If the budget proves infeasible on the 4070 (>12 h/seed
  projected), halt and report · do not silently reduce.
- **Stack** · torch 2.6.0+cu124 pinned via uv sources; CUDA-build
  verification after any dependency change. Hyperparameters in
  `configs/ppo_baseline.yaml`, not hardcoded. `wandb.mode: offline` set
  EXPLICITLY in that config (not inherited); W&B's only trusted role is
  the heartbeat "did it finish" signal. Checkpoints to `checkpoints/`
  with seed and step in the filename. Deterministic seeding throughout.
- **T=48 (briefing Section 3 invariant) · proposed ruling for operator
  ratification with this ADR** · T was the world model's BPTT sequence
  length, a supervised-batching parameter whose WDDM
  kernel-launch-amortization rationale is specific to backprop through
  time. PROPOSED: it does not transfer to model-free on-policy rollouts;
  PPO's rollout-segment length is a free hyperparameter recorded in the
  frozen config. If the operator instead rules it carried over, the
  config sets rollout-segment length = 48 before the freeze. Either way
  the ruling is recorded here · the invariant is not silently dropped.

**Rejected alternative · option (d) · considered and DEFERRED (not
dead).** Keep the RSSM and redesign the decoder target to conditional
volatility/scale · which volatility-clustering makes genuinely
predictable, and which the reproducible-but-tiny directional moves in the
ADR-006 result (+0.36 KL, -0.15 forward vs the MLP prior · correctly
signed for Hypothesis A but ~5.7 nats and ~0.75 nats short of their
thresholds) keep faintly alive. Deferred because: (i) the latent has
already collapsed under two opposite targets and the full prior-capacity
range · a third target redesign would be the third consecutive bet on a
paradigm with zero confirmed wins on this data; (ii) the model-free
baseline is prerequisite evidence either way · any future latent-paradigm
revival (including option d) must beat it to justify its complexity;
(iii) sequencing, not killing: the volatility-target idea remains
recorded here and in ADR-006's contingency menu, and re-opens only by
operator decision after the baseline result is in.

**Consequences.**

- Phase 5.5 (world-model scaling / 100k run) is BLOCKED · permanently for
  this line of work unless the operator re-opens it.
- The RSSM and `checkpoints/world_model_diagnostic_step=30000-v2.ckpt`
  are frozen historical evidence · no retraining, modification,
  fine-tuning, or extension.
- The model-free baseline becomes the new reference point: every future
  approach on this data/env/reward must be compared against it.
- A new harness `scripts/eval_baseline_gates.py` mirrors the conventions
  of `scripts/eval_gates.py` (checkpoint-driven, fixed seed, deterministic
  eval, never W&B) and additionally writes JSON + markdown artifacts to
  `artifacts/adr007/` · gate classification reads ONLY those on-disk
  artifacts. (`eval_gates.py`'s stdout-only output is superseded for the
  baseline.)
- An env-reuse assertion suite (Track B) must pass before any full run:
  it verifies the calendar-partition rule and feature-pipeline identity
  (NOT step-log window-index identity, which is a supervised-eval concept
  and does not apply to env rollouts) · see gate Section G below.

**Pre-registered evaluation gate.** Hard thresholds, every criterion
binary, classification in Phase 3 is purely mechanical. PASS requires ALL
of G-BL1..G-BL4 to be true on the designated median seed. Anything else
is FAIL. No partial pass, no rounding (comparisons at float64 precision
of the artifact values), no judgment calls.

**(A) Held-out data · calendar-partition definition.** "Same held-out
data" is defined at the calendar-partition level, not the step-log
window-index level: a UTC timestamp is held-out iff its day-of-month
satisfies `(day - 1) / days_in_month >= 0.85` · the exact rule of
`training/datamodule.py:396` with `val_month_frac = 0.85`, the same rule
behind the world-model diagnostics and the 8.8632 marginal baseline
(`docs/findings/2026-05-27-marginal-baseline.md`). Data span: the
BTCUSDT 1m snapshot `data/market_ro.duckdb`, 1,051,201 rows,
2024-05-03 03:00 -> 2026-05-03 03:00 UTC (gap-free at every horizon per
`docs/findings/2026-05-28-forward-returns-data-quality.md` · note that
doc's Sources line says the span ends 2026-04-30, which is stale prose:
1,051,201 gap-free 1-min rows force exactly 730 days, and the snapshot's
actual MAX(ts) was re-verified read-only as 2026-05-03 03:00 on
2026-06-10; the doc's gap TABLES, not its span prose, are the cited
evidence). Every
evaluation episode start has >= 256 bars of history. Policies are rolled
out IN THE ENV over these spans.

**(B) Evaluation episode set · fixed and enumerated.** 24 monthly val
spans (2024-05 .. 2026-04; 2026-05 has no val region in the snapshot,
which ends 2026-05-03). Per span: 3 non-overlapping 1440-step episodes
starting at span start + 0 h / + 24 h / + 48 h -> 72 episodes, shared
across all seeds and all comparator policies. Span starts at 00:00 UTC of
the first val day:

| Month | First val day | Month | First val day |
|---|---|---|---|
| 2024-05 | 28 | 2025-05 | 28 |
| 2024-06 | 27 | 2025-06 | 27 |
| 2024-07 | 28 | 2025-07 | 28 |
| 2024-08 | 28 | 2025-08 | 28 |
| 2024-09 | 27 | 2025-09 | 27 |
| 2024-10 | 28 | 2025-10 | 28 |
| 2024-11 | 27 | 2025-11 | 27 |
| 2024-12 | 28 | 2025-12 | 28 |
| 2025-01 | 28 | 2026-01 | 28 |
| 2025-02 | 25 | 2026-02 | 25 |
| 2025-03 | 28 | 2026-03 | 28 |
| 2025-04 | 27 | 2026-04 | 27 |

(31-day months -> day 28; 30-day -> 27; 28-day February -> 25; identical
to the datamodule rule. Each evaluation span uses bars from span start
through +4320 minutes inclusive · all inside the val region; the
remaining val day is an unused buffer.) All 24 spans were verified
contiguous against the snapshot on 2026-06-10 (4,321 bars each, exact
1-min spacing). The materialized 72-episode list (timestamps + kline row
indices) is generated deterministically from this rule and frozen as
`artifacts/adr007/eval_episodes.json` BEFORE the first full training
run, with its SHA-256 recorded in `artifacts/adr007/run_log.md` at
freeze time; the harness refuses to run against a hash-mismatching
artifact, and the gate may only be evaluated against that artifact.
Regeneration happens exactly once, at gate time, against the same frozen
snapshot: ANY mismatch with the frozen artifact · including a newly
detected gap in a span verified contiguous on 2026-06-10 · is an
integrity anomaly and a HALT for operator review. No agent-side month
exclusion exists; an exclusion is valid only as an operator amendment
recorded BEFORE the gate read, with thresholds unchanged (NOT rescaled)
and all metric definitions in (D)-(F) ranging over the surviving
enumerated set.

**(C) Policies under evaluation.**

- **Agent** · the final PPO checkpoint of each training seed, evaluated
  deterministically: action = argmax over policy logits, exact logit
  ties resolving to the lowest action index. Eval seed 42 seeds all
  framework RNGs as belt-and-braces · with enumerated starts and argmax
  actions no rollout stochasticity should remain, and (G)(vii) asserts
  bitwise repeatability (evaluation may run on CPU to guarantee it).
  Episodes run in enumerated order; each episode resets to initial cash
  10,000; any policy-internal recurrent/hidden state is re-initialized
  at every episode reset · the per-step information set is exactly the
  env observation (window + portfolio), never cross-episode memory.
- **Buy-and-hold comparator** · per disjoint val span, ONE 4320-step env
  episode with constant action 4 (100%): enters at span start (paying
  taker fee + slippage on the full notional), holds, marks to market at
  span end. Defined per-span · NOT per-episode · so entry costs are paid
  once per span, then aggregated. Its per-interval returns follow the
  exact formula in (D) -> 72 interval returns on the identical time base
  as the agent. (Constant action 4 implies negligible fee-dust
  rebalances after entry through the env's own mechanics · accepted; B&H
  runs through the SAME env and harness as every other policy.)
  Integrity precondition (mirror of flat's): per span, the harness B&H
  cumulative net log-return must match the closed-form kline value
  `ln((10000 x close_end / (close_start x 1.0002) - 10) / 10000)` ·
  entry at the span-start close paying the 0.1% taker fee (10.0 on the
  10,000 notional) and 2 bps slippage, mark at the span-end close ·
  within `|diff| <= 1e-4` (post-entry fee-dust rebalances are bounded
  well below this). Known-and-accepted asymmetry: if a span ever
  breached the 50% guardrail, B&H would stop out and carry forward while
  agent episodes reset daily · academic on this snapshot (no 72 h window
  approaches -50%).
- **Flat comparator** · constant action 0 over the 72 episodes. By
  construction it never trades. Harness integrity precondition, scoped
  precisely: flat's cumulative net log-return == 0.0 exactly AND flat's
  total realized turnover == 0.0 exactly. Flat's Sharpe is undefined by
  construction (zero variance); the harness reports it as null, and it
  is exempt from this precondition and from every criterion. Any
  violation of the scoped precondition means the harness is broken and
  must be fixed before any gate read (integrity precondition, not a
  gate).
- **Random reference** · uniform over the 5 actions, dedicated RNG seed
  7, same 72 episodes. Sanity reference ONLY · reported in full,
  participates in no gate criterion.

**(D) Metrics.** All returns are net of fees and slippage as embodied in
env equity. The 0.05 x turnover term is reward shaping · not a cash
flow · and is EXCLUDED from evaluation metrics (it remains in the
training reward unchanged).

- Per-episode net log-return · `r_i = ln(equity_end_i / 10000)` for
  agent/flat/random (each episode starts fresh; an early-terminated
  episode contributes its at-termination value). For B&H: per span `k`
  let `E_k(t)`, `t = 0..4320`, be its equity curve with
  `E_k(0) := 10000` (initial cash, BEFORE the entry trade · the harness
  prepends it; `StepInfo.equity` starts post-entry) and, after any early
  termination, `E_k(t)` carried forward flat to span end. The three
  interval returns per span are
  `r = ln(E_k(j x 1440) / E_k((j - 1) x 1440))`, `j = 1..3` -> 72
  interval returns. They telescope by construction: each span's three
  returns sum to `ln(E_k(4320) / 10000)`, the span's cumulative net
  log-return, entry cost included, also under carry-forward. `R_BH` has
  exactly ONE definition · the sum of the 72 interval returns; any
  per-span phrasing elsewhere in this ADR is this same number via the
  telescoping identity, never a second definition.
- Early termination (operator amendment A1 · 2026-06-10) · if ANY eval
  episode · agent, B&H, flat, random · terminates on the equity < 50%
  guardrail, the episode counts as COMPLETE with return
  `ln(E_terminal / E_start)`; B&H's span mark uses terminal equity if
  its 4320-step episode terminates early (the carry-forward rule above
  implements exactly this). Same env semantics for every policy · no
  termination-disabled special cases.
- Cumulative net log-return · `R = sum(r_i)` over the 72 episodes.
- Sharpe · `S = mean(r_i) / std(r_i, ddof=1) * sqrt(365)` (daily
  episodes, crypto trades 365 days, risk-free rate 0). The guard uses
  the SAME estimator as the formula: if `std(r_i, ddof=1) < 1e-12`, S is
  undefined and any criterion referencing it FAILs.
- Realized turnover · per-episode sum of per-step turnover (fraction of
  equity, the env's own definition); report mean/median/max across the 72
  episodes; `TO = total turnover summed over all 72 episodes`.
- Max drawdown · report-only, per episode and worst-of-set. Not a gate.
- Artifact authority and precision · the JSON artifact is the SOLE input
  to classification; the markdown artifact is display-only. All floats
  are serialized at full float64 round-trip precision (17 significant
  digits). The Phase-3 classifier recomputes every derived expression in
  G-BL1..G-BL4 (`R - 0.0002 x TO`, `max(S_BH, 0.0)`, the means) in
  float64 from the stored primitives (per-episode `r_i`, per-episode
  turnover, the 72 B&H interval returns) · it never compares against
  pre-rounded display values.

**(E) Seeds and aggregation rule.** Training seeds: **42, 0, 123** (the
project's standing seed triple). All metrics are computed and reported
for every seed. The DESIGNATED MEDIAN SEED is the seed whose cumulative
net log-return R is the middle value of the three (if two or more seeds
share the middle value, the numerically lowest seed number AMONG THE
TIED SEEDS is designated). EVERY gate criterion is evaluated on the
designated median seed only; the other seeds' numbers are reported as
dispersion evidence. A seed that fails to produce its final
step-2,000,000 checkpoint (crash, NaN divergence, power loss) is a
HALT: the gate cannot be read, no substitute seed and no
earlier-checkpoint evaluation is permitted, and the path forward is an
operator amendment recorded before any relaunch.

**(F) Criteria** (designated median seed; each independently binary):

- **G-BL1 · slippage-stressed absolute profitability (vs flat).**
  `R - 0.0002 x TO >= 0.010`. Rationale: flat is exactly 0; +0.010 over
  72 evaluated days (~5% annualized) is the minimum economically nonzero
  bar; the `0.0002 x TO` term re-prices every unit of realized turnover
  under a doubled slippage assumption (the v1 linear 2 bps slippage model
  is uncalibrated), so profits must survive a 2x slippage stress and
  overtrading RAISES the bar an agent must clear.
- **G-BL2 · return parity with passive.** `R >= R_BH`, where `R_BH` is
  the sum of buy-and-hold's 72 interval returns per (D) (equivalently,
  by the telescoping identity, the sum of the 24 per-span cumulative net
  log-returns). Equality passes (measure-zero at float64).
- **G-BL3 · risk-adjusted parity.** `S >= max(S_BH, 0.0)`, with S and
  S_BH per (D) on the identical 72-interval time base. Undefined S ->
  FAIL.
- **G-BL4 · turnover cap (overtrading guard + live-feasibility).** Mean
  per-episode realized turnover `<= 2.0` (at most two full portfolio
  flips per evaluated day on average) AND per-episode realized turnover
  `<= 10.0` for EVERY episode (concentration guard: a mean-only cap
  would allow one ~144-turnover episode amid 71 quiet ones · exactly the
  slippage-model-dominance the cap exists to prevent). One binary
  criterion: both conditions must hold.

**(G) Apples-to-apples preconditions (Track B · must pass before any
full run; failures block, they do not reinterpret).** An assertion
script verifies mechanically:

- (i) the partition rule in the eval-episode artifact reproduces
  `training/datamodule.py:396` month-by-month, and each of the 72 starts
  equals its span start + exactly {0, 24, 48} h for the 24 span starts
  derived from that rule (cross-checked against the table in (B);
  72 = 24 x 3 exactly);
- (ii) feature-pipeline identity · eval observations come from
  `envs.spot_btc.compute_feature_block` itself (single source of truth)
  with unchanged `FEATURE_NAMES` order, shapes (256, 12) + (3,), dtypes
  float32, and sanitization per the code's ACTUAL contract · NaN -> 0.0
  and +/-Inf -> +/-10.0 via `np.nan_to_num`; finite values are NOT
  clipped by the code, so the bounds check is empirical: assert no
  NaN/Inf and all eval-span observation values within the declared Box
  bounds [-10, 10] (max |x| ~ 7.6 over the full snapshot, measured
  2026-06-10);
- (iii) every enumerated episode lies wholly inside the val partition
  with >= 256 bars of history;
- (iv) the flat-policy and B&H integrity preconditions of (C);
- (v) training-partition purity · pre-run, draw >= 10,000 starts from
  the actual training sampler as configured and assert every
  [s, s + 1440] is train-pure under the (A) rule with s >= 256;
  post-run, re-verify the realized start logs
  (`artifacts/adr007/train_starts_seed<NN>.json`) the same way as a
  gate precondition;
- (vi) data-source identity · the env instantiated by the harness
  reports exactly 1,051,201 rows with MIN(ts) = 2024-05-03 03:00 and
  MAX(ts) = 2026-05-03 03:00 UTC, and the kline timestamp at every
  eval-episode row index equals the timestamp stored in the frozen
  artifact;
- (vii) rollout determinism · two consecutive harness invocations on the
  same checkpoint produce bitwise-identical artifact primitives.

The world-model's step-log window indices are NOT asserted · ruled out
by the operator (2026-06-10) as a supervised-eval concept that does not
transfer to env rollouts.

**(H) Verdict.** `PASS = G-BL1 AND G-BL2 AND G-BL3 AND G-BL4` on the
designated median seed. Every criterion is reported beside its
pre-registered threshold and its measured value in the findings doc
(`docs/findings/2026-06-XX-adr007-model-free-baseline-results.md`).
Either verdict is informative: PASS establishes extractable model-free
edge net of costs and a live reference for any future paradigm; FAIL ·
combined with the ADR-006 result · is evidence that this
data/feature/cost setup lacks extractable edge for BOTH paradigms tried,
and the next fork (richer features, different costs, option (d), or
stop) is the operator's call, made outside this ADR.

---

## Cold-Start Checklist

If you are returning to this codebase after a break, follow these steps in order.

1. Read this document end-to-end. Budget 30 minutes.
2. Read the latest implementation doc in `docs/implementations/` to understand what changed most recently and where the project stands in the phase sequence.
3. Run `pytest tests/ -x -q --ignore=tests/test_dream_endpoint.py` to verify the codebase compiles and existing tests pass.
4. Open `configs/world_model.yaml` and scan current hyperparameters, especially `max_steps`, `mode`, and any `forward_*` parameters added by the pivot.
5. Read gates from on-disk checkpoints and eval artifacts, NEVER from W&B: `scripts/eval_gates.py --ckpt <path>` for the (historical) world-model gates, `scripts/eval_baseline_gates.py` + `artifacts/adr007/` for the ADR-007 baseline gate. W&B silently desynced on two consecutive runs; its only trusted role is the heartbeat "did it finish" signal (`logs/heartbeat_*.log`).
6. Verify that `checkpoints/encoder_mae_full_raw.pt` exists. This is the pretrained encoder and cannot be regenerated without rerunning the MAE pretraining pipeline (~2 hours on the 4070).
7. Review the ADRs in Section 12. Any ADR with status "revisit" or referencing a not-yet-written future ADR number is a pending decision that may need your attention before the next training run.
8. If the dashboard's `DreamPlayer` panel shows feature-reconstruction visuals (15-dim feature time series), that visualization is stale · it predates the Phase 5.4 pivot and will be replaced with forward-distribution fan charts as part of Phase 5.5. Do not interpret current dashboard rollouts as model behavior until the dashboard work lands.
