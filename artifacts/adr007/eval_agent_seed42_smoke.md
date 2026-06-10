# ADR-007 eval - agent - smoke (display-only)

Display-only artifact. The JSON file with the same basename is the
SOLE classification input (ADR-007 (D) artifact authority).

| field | value |
|---|---|
| policy | agent |
| subset | smoke |
| purpose | smoke-determinism-recheck |
| device | cpu |
| episodes_sha256 | 1842d800900b871733414e6d71e068f188e6ce8cc0de2a923035cab43d9811f1 |
| ckpt_path | C:\Users\senso\OneDrive\Masaüstü\crypto-dreamer\checkpoints\ppo_baseline_seed42_step20000.ckpt |
| ckpt_sha256 | 7ca928c133ce75df3ee67fd9fee640e8bd4baf713eedab4cb16b493235d7d380 |
| seed | 42 |
| generated_utc | 2026-06-10T21:07:29+00:00 |

## Aggregates (computed, not classified here)

| metric | value |
|---|---|
| R (sum r_i) | -0.016692369456567525 |
| sharpe | None |
| TO (total turnover) | 7.160571820492924 |
| turnover mean / median / max | 7.160571820492924 / 7.160571820492924 / 7.160571820492924 |
| worst max drawdown | 0.0247544521297407 |
| n returns / n rollouts / n terminated | 1 / 1 / 0 |

## Integrity

| check | result |
|---|---|
| flat_ok | None |
| bh_ok | None |
