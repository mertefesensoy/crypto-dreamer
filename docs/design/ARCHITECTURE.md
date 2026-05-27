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

Training proceeds in two stages gated by validation diagnostics.

**Stage 1 · Diagnostic (30k steps).** A short run to verify that loss curves are healthy and KL releases from the free-bits floor. Configuration: T = 48 trajectory length, B = 32 batch size, bf16-mixed precision, AdamW optimizer with lr = 1e-4, weight_decay = 1e-6, 1000-step linear LR warmup, gradient clipping at 1000 (DreamerV3-style permissive clipping that catches only catastrophic gradient explosions, not normal training dynamics). Validation runs every 2500 steps over 100 batches (~3200 samples). Checkpoints saved every 5000 steps. T = 48 is a deliberate relaxation from T = 64 to amortize WDDM kernel-launch overhead on Windows; future Linux training may revert to T = 64.

**Stage 2 · Full run (100k steps).** Advanced to only after Stage 1 passes the validation gates (Section 11). Resumes from the Stage 1 checkpoint using Lightning's `Trainer.fit(ckpt_path=...)`, which restores model state, optimizer moments, LR scheduler position, and global step. The DataModule's `WeightedRandomSampler` does not preserve position across resume, so the data sequence post-resume differs from a hypothetical unbroken run · this is acceptable because training samples are drawn from a stochastic distribution.

**Operational practices.** A heartbeat callback writes one timestamped line per 100 training batches to `logs/heartbeat_*.log`, independent of W&B, to provide crash detection without depending on network connectivity. On the 4070 laptop, Modern Standby must be disabled before any overnight run via `powercfg /change standby-timeout-ac 0` to prevent WDDM-triggered hibernation mid-training (documented in the Phase 5.3 stall postmortem, `docs/implementations/2026-05-04-phase5-3-rssm-full-train.md`). torch.compile is disabled because Triton is unavailable in the Windows venv.

All training configuration lives in `configs/world_model.yaml`.

## 11 · Validation Gates

The following gates are evaluated after the 30k diagnostic run. All must pass before advancing to the 100k full run.

**Gate 1 · KL release.** KL_unclipped must exceed 32 nat by step 20k. This is the load-bearing gate. It confirms that z_t is encoding information the prior cannot predict from action history alone. If KL remains pinned at or below 32 nat (as happened in Phase 5.3 with the feature-reconstruction decoder, which finished at 25.7 nat), the forward-distribution target has insufficient stochastic structure and the pivot has failed.

**Gate 2 · Forward-distribution loss below marginal baseline.** The marginal baseline at horizon h is the cross-entropy of the constant-predictor that always outputs the empirical marginal forward-return histogram (computed once over the training set), evaluated on the validation set. Formally: let p_h(k) be the empirical frequency of bin k for horizon h over training data, and q_h(k) be the empirical frequency over validation data. Then baseline_h = -sum_k q_h(k) * log(p_h(k) + epsilon) with epsilon = 1e-9 for numerical safety. The total marginal baseline is the sum across the four horizons. The model's val/loss_forward_dist at step 20k of the diagnostic must be strictly less than this baseline. A model that fails to beat it is predicting no conditional structure beyond the unconditional distribution and has effectively learned nothing useful from the encoder or RSSM.

Drift band. The marginal baseline reported in `docs/findings/2026-05-27-marginal-baseline.md` is 8.8632. The drift band · H(q,p) - H(q) summed across horizons · is approximately 0.015 nat. A val/loss_forward_dist value between 8.85 and 8.86 is effectively a tie with the baseline · the model can achieve this band by exploiting the mild train-to-val distribution mismatch without learning conditional structure. Gate 2 is genuinely passed only when val/loss_forward_dist < 8.85 at step 20k of the diagnostic. Values in [8.85, 8.86] are reported as inconclusive rather than passing.

**Gate 3 · Reward NLL stability.** Validation reward NLL should be approximately 0.48, consistent with prior runs (Phase 5.3 achieved val/loss_reward = 0.478). A significant regression would indicate the pivot introduced a bug in the shared RSSM or reward pathway.

If all gates pass, proceed to the 100k full run (Phase 5.5). If Gate 1 fails, revisit the architecture per ADR-003's contingency options: prior capacity restriction, KL warmup schedule, or model paradigm change. If Gate 2 fails with Gate 1 passing, the problem is in the head architecture or the target computation, not the RSSM. If Gate 3 fails, debug the shared components (encoder, RSSM, optimizer configuration).

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

---

## Cold-Start Checklist

If you are returning to this codebase after a break, follow these steps in order.

1. Read this document end-to-end. Budget 30 minutes.
2. Read the latest implementation doc in `docs/implementations/` to understand what changed most recently and where the project stands in the phase sequence.
3. Run `pytest tests/ -x -q --ignore=tests/test_dream_endpoint.py` to verify the codebase compiles and existing tests pass.
4. Open `configs/world_model.yaml` and scan current hyperparameters, especially `max_steps`, `mode`, and any `forward_*` parameters added by the pivot.
5. Check the project's W&B dashboard (`crypto-dreamer` project) for the latest diagnostic run. Compare `kl_unclipped` against the 32-nat gate threshold.
6. Verify that `checkpoints/encoder_mae_full_raw.pt` exists. This is the pretrained encoder and cannot be regenerated without rerunning the MAE pretraining pipeline (~2 hours on the 4070).
7. Review the ADRs in Section 12. Any ADR with status "revisit" or referencing a not-yet-written future ADR number is a pending decision that may need your attention before the next training run.
8. If the dashboard's `DreamPlayer` panel shows feature-reconstruction visuals (15-dim feature time series), that visualization is stale · it predates the Phase 5.4 pivot and will be replaced with forward-distribution fan charts as part of Phase 5.5. Do not interpret current dashboard rollouts as model behavior until the dashboard work lands.
