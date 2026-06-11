# ADR-007 eval - agent - full (display-only)

Display-only artifact. The JSON file with the same basename is the
SOLE classification input (ADR-007 (D) artifact authority).

| field | value |
|---|---|
| policy | agent |
| subset | full |
| purpose | phase3-gate-eval |
| device | cpu |
| episodes_sha256 | 1842d800900b871733414e6d71e068f188e6ce8cc0de2a923035cab43d9811f1 |
| ckpt_path | C:\Users\senso\OneDrive\Masaüstü\crypto-dreamer\checkpoints\ppo_baseline_seed0_step2000000.ckpt |
| ckpt_sha256 | 891f999d00a63ca346419bf5bebd7f1654b9f6c5a127223a0d62ca98b6b38433 |
| seed | 0 |
| generated_utc | 2026-06-11T07:00:56+00:00 |

## Aggregates (computed, not classified here)

| metric | value |
|---|---|
| R (sum r_i) | -0.16919859808246518 |
| sharpe | -4.03075885136514 |
| TO (total turnover) | 46.837615097856144 |
| turnover mean / median / max | 0.6505224319146686 / 0.6353682350583597 / 1.3331014592296953 |
| worst max drawdown | 0.045762158884814275 |
| n returns / n rollouts / n terminated | 72 / 72 / 0 |

## Integrity

| check | result |
|---|---|
| flat_ok | None |
| bh_ok | None |
