# crypto-dreamer · Backlog

Future-work items that are not on the active roadmap but should not be forgotten. Each item has a rough effort sense (small/medium/large) and a rationale for deferral.

---

## Dashboard

**DreamPlayer rework for forward-distribution visualization** · medium · The `DreamPlayer` panel in `dashboard/src/components/internals/DreamPlayer.tsx` currently shows feature-reconstruction visuals (15-dim feature time series) from the deleted `DecoderHead`. After the Phase 5.4 pivot, this visualization is stale. Replace with forward-distribution fan charts showing the predicted return distribution at each of the four horizons (1, 5, 15, 30 bars), overlaid with realized returns. Blocked on Phase 5.5 checkpoint existence · the fan chart requires a trained `ForwardDistributionHead` to produce meaningful outputs.

**Live trading monitoring view** · large · A dashboard tab showing real-time portfolio performance, current allocation, P&L decomposition, and model confidence metrics (entropy of the predicted forward-return distributions). Requires Phase 6 actor training to be complete and a live data feed from Binance. Research-track: needs a latency budget analysis to determine whether the 1-minute decision loop is compatible with real-time dashboard updates via the existing WebSocket architecture.

**Observation window heatmap update** · small · The Internals tab heatmap currently displays a 16x12 sparse sampling of the 256x12 market-feature window (sent as `features_b64` in `StepInfo`). After the pivot, consider whether the 3 portfolio scalars should also be visualized, and whether a denser sampling (e.g. 32x15) is worth the bandwidth. Low priority · the current visualization is functional.

---

## Model Architecture (research-track)

**Multi-asset extension (ETH, SOL alongside BTC)** · large · Requires re-derivation of bin ranges per asset (each asset has different return volatility and tail behavior), datamodule rework to handle multiple symbols in the klines table, and a decision about whether the RSSM maintains per-asset latent states or a shared cross-asset latent. The iTransformer encoder's per-variable attention would naturally extend to cross-asset features, but the bin-sizing and two-hot encoding are currently hardcoded to BTC-specific ranges in `docs/design/ARCHITECTURE.md` Section 6. Defer until the single-asset architecture is validated through Phase 6.

**Longer trajectory length T=64 on Linux** · small · `configs/world_model.yaml` documents the T=64 to T=48 reduction as a deliberate hyperparameter relaxation to amortize WDDM kernel-launch overhead on the Windows 4070. A Linux training environment (bare-metal or cloud GPU) would eliminate this overhead and allow reverting to T=64 for longer temporal context. Depends on the "Linux training environment migration" operational item below.

**Prior capacity restriction experiment** · medium · ADR-003 contingency option (a): reduce `prior_head` hidden dim to force the prior to be less expressive. Only relevant if Gate 1 fails in the Phase 5.4 diagnostic. If Gate 1 passes, this experiment becomes a research curiosity rather than a necessity. Keep on backlog as insurance.

**Model-free RL baseline** · medium · Train a PPO or SAC agent on the same observation space (256-bar x 15-feature window) and reward signal (log-return minus turnover penalty) without a world model. Serves as a control experiment regardless of world-model outcome: if the model-free agent achieves comparable or better returns with less complexity, the world-model architecture needs stronger justification. Also listed as ADR-003 contingency option (c). Useful even if Phase 5.4 succeeds, as a benchmark for Phase 6 actor performance.

**KL warmup schedule experiment** · small · ADR-003 contingency option (b): start with zero KL weight and ramp to full over 5k steps. Only relevant if Gate 1 fails. Low implementation effort (a few lines in `models/world_model.py::_step`), but requires a full 30k diagnostic to evaluate. Defer unless needed.

**Continuous action space** · medium · Replace the 5-action discrete allocation ({0%, 25%, 50%, 75%, 100%}) with a continuous [0, 1] target allocation. Would allow finer-grained portfolio control but changes the action embedding (`models/action_embed.py`) from a learned lookup to a linear projection, and requires a different actor architecture in Phase 6. Not on the critical path · the discrete space is sufficient for validating the world model.

---

## Operational

**Linux training environment migration** · medium · Eliminates WDDM kernel-launch overhead (~70ms/step) and Modern Standby risk. Options: (a) personal Linux desktop with a GPU, (b) cloud GPU instance (Lambda, Vast.ai), (c) WSL2 with GPU passthrough (partial fix · WDDM overhead remains). Requires porting the training pipeline and verifying that the Lightning checkpoints are cross-platform compatible. Not urgent while the 4070 laptop is the only available GPU.

**W&B sweep infrastructure** · small · Set up W&B Sweeps for hyperparameter exploration, particularly useful for Phase 6 actor tuning (learning rate, entropy coefficient, imagination horizon). Low effort to configure but not needed until Phase 6 begins.

**Automated Modern Standby check at training-script startup** · small · Add a check at the top of `training/train_world_model.py` that queries `powercfg /query SCHEME_CURRENT` and fails fast if standby-timeout-ac is not set to 0. Prevents a repeat of the Phase 5.3 stall. Could be implemented as a pytest fixture or a pre-training assertion. Low priority since the powercfg mitigation is now documented in the Phase 5.3 implementation doc and the ARCHITECTURE.md training protocol section.

**Live Binance WebSocket ingestion** · medium · `data/ingest.py` is offline-only (REST API batch fetch). A WebSocket worker would provide real-time kline updates for live trading. Referenced in `data/ingest.py` line 5 as a v2 item. Not needed until Phase 6 actor is deployed for live trading.

**Slippage model calibration with L2 data** · large · The current slippage model in `envs/spot_btc.py` uses a fixed 2 bps assumption. Referenced in the environment code as needing calibration against "real Binance L2 data in v2." Requires order-book data ingestion and a statistical model of execution quality as a function of order size and market conditions. Research-track · defer until Phase 6 actor is generating realistic trade sizes.

---

## Documentation

**Phase 6 design document** · large · Written when Phase 5.5 lands. Should cover: actor algorithm selection (DreamerV3 imagination vs. real-environment vs. hybrid), reward shaping decisions, action space refinements, evaluation protocol (backtesting on held-out data), and a new set of ADRs for Phase 6 architectural choices.

**Public-facing README rewrite** · small · The current README was written during Phase 3 (v1 glue) and describes the project as a live-trading dashboard with a random agent. It does not mention the world model, the RSSM, the forward-distribution head, or Phases 5-6. Rewrite to reflect the project's current scope and architecture. Reference `docs/design/ARCHITECTURE.md` for technical detail rather than duplicating it in the README.

**ADR directory extraction** · small · Current ADRs are inline in `docs/design/ARCHITECTURE.md` Section 12. If the count grows past ~10 (currently at 5: ADR-000 through ADR-004), extract them into a `docs/adr/` directory with one file per ADR for easier cross-referencing. Not worth doing until the count justifies the overhead.

**Implementation doc template** · small · The existing implementation docs in `docs/implementations/` follow a consistent structure (Problem/Motivation, What Changed, Implementation Approach, Mathematical Details, Design Decisions, Verification, Related Docs) but there is no `_TEMPLATE.md` file. Creating one would reduce friction for future docs and ensure consistency. Low priority since the convention is well-established by example.
