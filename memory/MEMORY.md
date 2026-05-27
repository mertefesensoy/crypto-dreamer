# memory index

## Phase docs

- [2026-05-03 phase 1 — backend](../docs/implementations/2026-05-03-v1-phase1-backend.md)
- [2026-05-03 phase 2 — dashboard](../docs/implementations/2026-05-03-v1-phase2-dashboard.md)
- [2026-05-03 phase 3 — glue and polish](../docs/implementations/2026-05-03-v1-phase3-glue.md)
- [2026-05-03 phase 4 — history + real Training/Internals tabs](../docs/implementations/2026-05-03-v1-phase4-history-and-real-tabs.md)
- [2026-05-03 phase 5.0.5 — encoder pretraining (TS-MAE)](../docs/implementations/2026-05-03-phase5-0-5-encoder-pretrain.md)
- [2026-05-04 phase 5.1 — datamodule + encoder integration](../docs/implementations/2026-05-04-phase5-1-datamodule.md)
- [2026-05-04 phase 5.2 — RSSM core + heads + losses smoke](../docs/implementations/2026-05-04-phase5-2-rssm-smoke.md)

## Stable contracts

- Action space: discrete `{0%, 25%, 50%, 75%, 100%}` allocation, long-only.
- Reward: `r_t = log(equity_t / equity_{t-1}) - 0.05 * turnover_t`.
  Coefficient is ~10x the 0.001 taker fee, sized to discourage churn
  without crushing exploration.
- Redis channels (live tail): `dreamer:steps`, `dreamer:episodes`.
- Redis streams (replay history): `dreamer:steps:hist` (MAXLEN 5,000,000),
  `dreamer:episodes:hist`. Each entry is a single field `data` with the
  full JSON payload.
- DuckDB table `step_log` mirrors the per-step columns plus `agent_id`
  for offline analysis. v2 trainer's experience replay reads from here.
- StepInfo wire field order (stable): `ts, price, action, target_alloc,
  realized_alloc, cash, btc, equity, turnover, fee_paid, reward,
  features_b64, episode, step, action_probs, agent_id`. Add fields at
  the end, never rename, never reorder.
- `features_b64` is base64(float16, row-major, 16 timesteps × 12 features).
  16 evenly-spaced rows from the 256-row obs window via
  `np.linspace(0, WINDOW-1, 16, dtype=int)`. Last row = current bar.
- API CORS allowlist: `http://localhost:5173` (dashboard origin).

## Non-obvious environment facts

- DuckDB on Windows takes an exclusive file lock; the agent and the
  ingest cannot run concurrently.
- DuckDB stores tz-naive timestamps but the ingest writes UTC values;
  `envs/spot_btc.py` localises with `pd.to_datetime(..., utc=True)` so
  JS `Date()` parses chart axes correctly.
- The portable Redis 5.0.14.1 zip in `.tools/redis/` is a dev-machine
  workaround. Docker, Memurai-via-winget, and WSL were all unavailable
  on the dev machine; phase 3's compose uses the canonical
  `redis:7-alpine` image.
- Memurai install via winget fails when the Windows Firewall service
  (`MpsSvc`) is stopped. Starting the service requires admin.
- Redis 5.0.14 (the portable Windows port we use) does not accept the
  `(` exclusive ID prefix added in Redis 6.2. Cursor-based pagination
  with that syntax errors out. `/history` uses a single sized
  XREVRANGE call as a workaround.

## v2 follow-ups

- Replace `agents/run_random.py` with `agents/dreamer.py` (RSSM + actor +
  IQN critic + iTransformer encoder).
- Add a Binance WebSocket live-mode flag to the env for real-time bars.
- Wire the dashboard kill-switch into a `serve/api.py` `/control` POST.
- Replace dashboard placeholders on Training and Internals tabs with
  real loss curves, KL, entropy, attention maps.
- Calibrate the slippage model against real Binance L2 data.

## Phase 5 contracts

- `envs/spot_btc.py` exposes module-level `compute_feature_block(df) ->
  ndarray` (12-feature matrix) and `FEATURE_NAMES` tuple. Single source
  of truth for what features the agent and the world model see —
  importing this from training code guarantees byte-equality with what
  the env produces.
- DuckDB has two paths: `data/market.duckdb` (live, agents write to
  it) and `data/market_ro.duckdb` (snapshot, training reads from it).
  The snapshot was taken right after the kline ingest finished and is
  identical to the live DB at that moment. Phase 5.1+ should also
  read klines from the snapshot to avoid contention with any future
  agents.
- Pretrained encoder weights: `checkpoints/encoder_mae_full_raw.pt` —
  raw `state_dict` (no Lightning wrapper). Phase 5.1 datamodule + 5.2
  world model load this to initialize the iTransformer's variable
  embeddings 0..11 (market features); embeddings 12..14 (portfolio)
  use random init.
- W&B project: `crypto-dreamer`. Pretrain run id `kbpkmhd8`,
  name `phase5.0.5-mae-pretrain`. Phase 5.3 will use run name
  `phase5.3-rssm-full`.
- 500 random-agent episodes collected across 4 agent_ids:
  `random:1` (114 eps, seed=1), `random:1b` (87 eps, seed=4),
  `random:2` (150 eps, seed=2), `random:3` (150 eps, seed=3).
  Total: 501 useful episodes / ~713K step_log rows over 2024-05 →
  2026-04. Plus 3 dev-leftover eps in `random:17`/`random:42`. The
  Phase 5.1 datamodule includes ALL `random:%` agent_ids (the 3 extra
  eps add 1,560 step_log rows — immaterial relative to the 713K main).
- Phase 5.1 datamodule (`training.datamodule.SpotBTCDataModule`)
  yields trajectory batches: `obs_window (B,T,256,15)`,
  `next_obs_window (B,T,256,15)`, `action (B,T) i64`,
  `reward (B,T) f32`, `continue_flag (B,T) bool`, `is_first (B,T) bool`.
  Default `B=16, T=64`. obs_window cols 0-11 are the 12 market features
  (sliced from a precomputed feature cache); cols 12-14 are the
  portfolio scalar at that step BROADCAST across the 256-row time
  dim. `setup()` produces 586K train + 93K val subsequence starts
  from the 504 episodes. Sampler is `WeightedRandomSampler` weighted
  by 1/month_size for month stratification.
- Phase 5.1 encoder loading: `iTransformerEncoder(n_vars=15,
  mae_checkpoint="checkpoints/encoder_mae_full_raw.pt")` copies
  `input_proj` (shared), all transformer layer weights, and
  `var_embed[:12]` (market). `var_embed[12:15]` (portfolio) stay at
  random init. Missing checkpoint → WARNING + full random init,
  never crashes.
- Survivorship sanity (Phase 5.1 `setup()` output): 237/504 = 47% of
  random-policy episodes early-terminated at the 50% drawdown
  guardrail. Their tails are fully included in the training pool;
  `continue_flag[-1] = False` is the only marker.
- Phase 5.2 world model: `models.world_model.WorldModel` Lightning
  module. 3.3M params total (encoder 827K, RSSM 1.4M, heads ~1.1M,
  action embed 160). Loss aggregation: `L_pred + 0.5·L_dyn + 0.1·L_rep`
  averaged over loss-active steps (T - burn_in). Burn-in = 5 steps.
  At loop step t: `prev_a` (init zero) is the action used in the GRU,
  decoder predicts `obs_window[:, t, -1, :]` (current bar 15-dim row),
  reward and continue heads predict `reward[:, t]` and
  `continue_flag[:, t]`. Hidden state reset at t=0 always; mid-traj on
  `is_first[:, t]`. `mode=smoke` auto-sets `data.max_episodes=5`.
- Phase 5.2 free-bits behavior in early training: per-latent KL stays
  below 1 nat for the first ~10K steps, so dyn/rep losses are
  constant at 32 each (32 latents × 1 nat floor). Total loss is
  dominated by `0.5·32 + 0.1·32 = 19.2`. Real learning visible in
  `L_pred` (decoder + reward + continue). `kl_unclipped` GROWING is a
  desired signal — posterior becoming more informative than prior.
  Don't expect total-loss drops > a few % until clipping releases.
- Phase 5.3 full-run command (when ready):
  `python -m training.train_world_model mode=full train.max_steps=200000
   train.max_hours=4 wandb.run_name=phase5.3-rssm-full
   data.max_episodes=null`
