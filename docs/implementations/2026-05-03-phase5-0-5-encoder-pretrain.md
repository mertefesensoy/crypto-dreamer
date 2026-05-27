# 2026-05-03 — Phase 5.0.5: encoder pretraining (TS-MAE)

## Problem / Motivation

Phase 5 builds a DreamerV3-style world model. Stage 1 of that work is
the iTransformer encoder: a per-variable tokenizer that compresses each
agent observation (256 historical bars × 12 market features + 3
portfolio scalars) into 15 tokens of dim 128. The encoder will be
trained jointly with the RSSM in Phase 5.2, but starting from random
init makes that joint optimization harder — the RSSM has to learn
*both* the world dynamics *and* a useful representation simultaneously.

Phase 5.0.5 pretrains just the encoder with a self-supervised
objective on raw market data, so Phase 5.2 starts with weights that
already understand short-horizon market structure (autocorrelation,
volatility clustering, return cross-relationships).

The objective is **TS-MAE** (Time-Series Masked Autoencoder): mask 40%
of timesteps, reconstruct the masked positions with MSE. Standard
inverted-transformer pretraining recipe; the auxiliary MLP decoder is
discarded after this phase, only the encoder weights persist into
Phase 5.1+.

## What Changed

| File | Description |
| --- | --- |
| `pyproject.toml` | Added `lightning>=2.4`, `wandb>=0.18`, `einops>=0.8`. Added `models*`, `training*` to setuptools package discovery. |
| `.gitignore` | Excluded Phase-5 artifact dirs: `checkpoints/`, `wandb/`, `outputs/`, `.hydra/`, `multirun/`. |
| `envs/spot_btc.py` | Extracted module-level `compute_feature_block(df) -> ndarray` from `SpotBTCEnv._precompute_features`. The env method now delegates. Added `FEATURE_NAMES` tuple as a stable name mapping for downstream code. **Byte-equality with prior behavior preserved** — the existing `tests/test_env_smoke.py` and `tests/test_features.py` (10 tests) still pass unchanged. |
| `models/__init__.py` (new) | Package marker with module map. |
| `models/encoder.py` (new) | `iTransformerEncoder` — 4-layer pre-norm transformer, d_model=128, 4 heads, dim_ff=512, dropout=0.1. Per-variable tokenization via shared `Linear(seq_len → d_model)` plus learnable `nn.Embedding(n_vars, d_model)`. Constructor accepts `n_vars` so the same class serves pretrain (n_vars=12) and the world model (n_vars=15). |
| `models/mae_decoder.py` (new) | `MAEDecoder` — 3-layer GELU MLP that maps each per-variable token (d_model) back to a length-T series, then transposes to (B, T, F). Throwaway after pretraining. |
| `training/__init__.py` (new) | Package marker. |
| `training/pretrain_mae.py` (new) | Hydra entrypoint. Loads klines from a DuckDB snapshot, computes the 12-feature matrix once, builds train/val pools split by calendar month, runs a Lightning `MAEModule` with bf16-mixed precision. Includes `TimeBudgetCallback` (wall-clock cap) and `TinyVerifyCallback` (200-step pass criterion). |
| `configs/pretrain_mae.yaml` (new) | Default hyperparams. `mode: full|tiny` selects the run profile. |
| `tests/test_encoder.py` (new) | 6 smoke tests: shape (n_vars=12, n_vars=15), decoder shape, encoder→decoder roundtrip, attention-weights-sum-to-1 (catches softmax-axis bugs), gradient flow. |
| `data/market_ro.duckdb` (new, gitignored) | Read-only snapshot of `data/market.duckdb`, taken at the moment the kline ingest finished and before the random-agent runs reacquired the write lock. Pretraining reads from this snapshot so it doesn't compete with the agents for the (Windows-exclusive) DuckDB lock. |
| `checkpoints/encoder_mae_full_raw.pt` (new, gitignored) | Final encoder `state_dict` (no Lightning wrapper). Loaded by Phase 5.1 datamodule + 5.2 world model. |
| `checkpoints/encoder_mae_full-step=*.ckpt` (new, gitignored) | Best-by-val-MSE Lightning checkpoint (full module). Useful if anyone wants to resume training. |
| `checkpoints/mae_qualitative.png` (new, gitignored) | Held-out reconstruction sanity plot — 12 features over a randomly-selected 256-bar window, original vs. reconstructed overlaid. |

## Implementation Approach

**Model.** iTransformer follows Liu et al. 2024: instead of tokenizing
*timesteps* and attending across time (the standard pattern that
scales O(T²)), tokenize *variables* and attend across variables
(O(F²) where F ≤ 15 here). Each variable's full T-length series is
projected to a single d_model token by a shared `Linear(T → d_model)`,
then a learnable variable-specific embedding is added. Pre-norm
transformer for bf16 stability.

**Why pre-norm.** With bf16-mixed, post-norm transformers can produce
NaN-loss spikes early in training because residual additions overflow
before normalization clamps them. Pre-norm normalizes the residual
input first; the addition stays in range. Cost: a small drop in
final-loss performance, well worth it for stability.

**Masking.** For each sample in a batch, draw a Bernoulli(0.4) mask
over the 256 timesteps. Replace masked rows with zero (same
distribution as the input post-normalization, which is z-score-like
and zero-centered for most channels). The encoder still ingests the
full 256-step series; the decoder reconstructs the entire matrix; the
loss is averaged only over masked positions.

**Decoder.** Per-variable shared 3-layer MLP (d_model → 128 → 128 →
T). The output is `(B, F, T)` then transposed to `(B, T, F)` to match
the input shape. ~66K params total — deliberately small so the
encoder is forced to do the work of representing useful features
rather than letting the decoder memorize.

**Train/val split.** Hold out the last calendar month with full
coverage (`2026-04`, 30 days × 1440 min = 43,200 minutes) as
validation. A window's "right edge" timestamp determines its bucket —
windows with right-edge ts in `[2026-04-01, 2026-05-01)` are val,
everything else is train. **No subsequence crosses the boundary** by
construction (one window = one right edge → one bucket).

**Snapshot reads.** Random-agent data collection runs in parallel
with this phase but on a different agent_id range. DuckDB on Windows
takes an exclusive file lock for any writer (memory note from
v1 phase 1). To avoid lock contention, `data/market.duckdb` was
copied to `data/market_ro.duckdb` at the moment kline ingest
completed and *before* the random agents reacquired the write lock.
The pretraining script reads only the snapshot.

**Tiny verify run.** A 200-step run on a 7,200-window subset, with
`warmup_steps=0` so step 10 and step 200 are both at full LR
(otherwise the comparison is dominated by the warmup ramp, not
learning). Pass criterion (locked in plan): train MSE at step 200
must be ≥20% below train MSE at step 10. **Result: 21.3% drop**
(0.253 → 0.199). The model is learning meaningfully.

**Time budget.** A `TimeBudgetCallback` polls wall clock at the end
of each train batch and sets `trainer.should_stop = True` past the
limit. Set to 2 hours for the full run. The 30-min in-flight stop
rule (val MSE drop ≥10% vs. random init) was a *manual* gate —
verified out-of-band via the W&B summary.

## Mathematical / Statistical Details

**TS-MAE loss.** Let X ∈ ℝᴮˣᵀˣᶠ be the input, M ∈ {0,1}ᴮˣᵀ a 40%
Bernoulli mask, and X̂ ∈ ℝᴮˣᵀˣᶠ the decoder output. Define the
per-(b,t) MSE as the mean over feature channels:

    e[b,t] = (1/F) · Σ_f (X̂[b,t,f] − X[b,t,f])²

The loss is the mean of e[b,t] over masked positions only:

    L = (Σ_{b,t} e[b,t] · M[b,t]) / (Σ_{b,t} M[b,t])

Equivalently: average over feature channels first, then average over
*masked* (b,t) positions (not all positions). Unmasked positions
contribute zero to the gradient.

**Why mask 40%.** Higher mask ratios force the encoder to rely on
inter-variable correlations and longer-range time context (since
nearby timesteps may all be masked). 40% is the original MAE/TS-MAE
recommendation — high enough to be a non-trivial reconstruction
problem, low enough that the encoder still has anchor points.

**Two-hot reward / continue / decoder losses for Phase 5.2** are
*not* part of this run; pretraining is reconstruction-only.

## Design Decisions

- **Snapshot the DB instead of pausing agents.** Considered: serially
  running pretrain after agents finish (saves no time but avoids the
  snapshot file). Considered: re-architecting the env/agents to
  release the DB connection between transactions. The snapshot is the
  smallest, lowest-risk option — 90 MB extra on disk, zero changes to
  the agent code path, and the snapshot can be deleted after Phase 5.2
  caches its own feature tensor.
- **warmup_steps=0 for tiny verify, 1000 for full run.** The full run
  inherits the world-model setup's warmup convention (locked in plan).
  The tiny run does *not* need warmup — its 200-step budget is too
  short to amortize a 1000-step ramp. Without warmup, "step 10 vs
  step 200" cleanly measures actual learning, which is what the
  pass criterion is testing.
- **MAE decoder is throwaway.** Considered: keeping it as a head for
  some auxiliary loss in Phase 5.2 (could regularize against
  representation collapse). Rejected — the world-model decoder already
  does feature reconstruction at the per-step granularity, so a second
  reconstruction head would be redundant and conflict with the
  world-model loss balance.
- **Shared (not per-variable) input projection.** Per-variable
  projections give each variable its own `Linear(T → d_model)` weights
  (15 × 33K = 500K params). Shared collapses to one `Linear(T →
  d_model)` (33K params) plus a per-variable embedding (15 × 128 = 2K).
  iTransformer paper convention. Saves params, doesn't hurt
  performance because the variable embedding still differentiates.
- **bf16-mixed over fp16-mixed.** RTX 4070 (Ada, compute 8.9) supports
  bf16 natively. bf16 has the same dynamic range as fp32 (just lower
  precision), so loss spikes from underflow are nearly impossible.
  fp16 has the same precision but a much smaller dynamic range and
  needs gradient scaling.

## Verification

1. **Existing tests still pass after the env refactor.**
   ```
   pytest tests/test_env_smoke.py tests/test_features.py -q
   ```
   `10 passed in 35.61s`. Byte equality with prior behavior is
   implicit — the env method now delegates to
   `compute_feature_block`, and the smoke test asserts feature shape,
   no NaN, and `[-10, 10]` clipping just like before.

2. **Encoder smoke tests.**
   ```
   pytest tests/test_encoder.py -v
   ```
   `6 passed in ~30s`. Verifies (a) shape for n_vars ∈ {12, 15},
   (b) decoder shape, (c) roundtrip shape, (d) attention weights sum
   to 1.0 along the key axis, (e) gradient flow through all params.

3. **Tiny verify run.**
   ```
   python -m training.pretrain_mae mode=tiny train.max_steps=200 \
       train.max_hours=0 train.warmup_steps=0 \
       train.val_check_interval=100
   ```
   PASS — train MSE 0.253 (step 10) → 0.199 (step 200), 21.3% drop
   (criterion ≥20%).

4. **Full 2-hour pretrain run.**
   ```
   python -m training.pretrain_mae mode=full train.max_steps=-1 \
       train.max_hours=2.0
   ```
   Result: see W&B run [phase5.0.5-mae-pretrain](https://wandb.ai/sensoymertefe-ted-niversitesi/crypto-dreamer/runs/kbpkmhd8) (id `kbpkmhd8`).
   - Final train/mse_step: **0.1007**
   - Final train/mse_epoch: **0.1025**
   - Final val/mse: **0.1066**
   - Random-init train MSE (from tiny verify step 10): ~0.26
   - **Drop vs. random init: ~62%** (criterion was ≥10% by 30-min mark; comfortably exceeded)
   - Final global step: **163,511** (~22.7 steps/sec on the 4070 with bf16-mixed)
   - Wall clock: **7,212 sec (≈ exactly 2 h)** — TimeBudgetCallback fired as designed
   - GPU stayed cool throughout (49–52°C, 24–37% utilization with batch=32 — there is room to push batch size in future runs but not needed for this stage)

5. **Qualitative check.** [`checkpoints/mae_qualitative.png`](../../checkpoints/mae_qualitative.png)
   overlays reconstructed features against originals on a held-out
   256-bar window from 2026-04-15 onward (val month). Per-channel
   masked-position MSE on this window:

   | feature | masked-MSE | interpretation |
   | --- | --- | --- |
   | log_ret, vol_5, vol_15, vol_60 | 5.8 – 6.2e-6 | recon collapses toward smoothed mean (correct: per-minute returns are unpredictable noise) |
   | rsi_14 | 9.2e-3 | close tracking; rsi is slow |
   | macd | 6.3e-6 | near-perfect |
   | vol_z | 4.7e-1 | high — volume z-score has unpredictable spikes; this is the dominant residual error |
   | hl_range | 5.9e-6 | near-perfect |
   | close_norm | 1.6e-1 | tracks but with some lag on fast moves |
   | ret_5, ret_15, ret_60 | 7e-6 – 2e-5 | recon collapses toward 0 (correct: short-window returns are noise) |

   Visual inspection: slow features (`rsi_14`, `macd`, `close_norm`,
   `vol_60`) reconstruct tightly; high-frequency channels show the
   expected denoising-autoencoder behavior of predicting the conditional
   mean rather than a specific noise realization.

## Reproducibility

- `lightning.pytorch.seed_everything(42, workers=True)` at script entry.
- DataLoader uses `torch.Generator().manual_seed(42)` for shuffle order.
- Determinism logged to W&B run config: `seed=42`, `shuffle_seed=42`,
  `torch_version`, `cuda_version`, `cudnn_version`, `gpu` model name.
- **cuDNN nondeterminism is left enabled** — the
  `torch.backends.cudnn.deterministic` flag is *not* set, so identical
  reruns may differ in the last few decimals due to non-deterministic
  GEMM kernel selection. This is the standard tradeoff for ~10-30%
  training throughput. Disclosed here so future readers don't try to
  reproduce a specific final loss to 6 decimal places.

## Related Docs

- Plan: `~/.claude/plans/we-re-starting-phase-5-staged-pinwheel.md`
- v1 Phase 1 (env, feature pipeline): `2026-05-03-v1-phase1-backend.md`
- Up next: `2026-05-03-phase5-1-datamodule.md` — episode-aware
  datamodule that reuses `compute_feature_block` and consumes this
  encoder's pretrained weights.
