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
| ckpt_path | C:\Users\senso\OneDrive\Masaüstü\crypto-dreamer\checkpoints\ppo_baseline_seed123_step2000000.ckpt |
| ckpt_sha256 | d55aa775ce696b0bba0b2dee338ba7d4a2e55c69b0716dd6f384943bfcbe77ac |
| seed | 123 |
| generated_utc | 2026-06-11T07:02:52+00:00 |

## Aggregates (computed, not classified here)

| metric | value |
|---|---|
| R (sum r_i) | -0.1674535432734741 |
| sharpe | -4.004367168608533 |
| TO (total turnover) | 45.83869638217388 |
| turnover mean / median / max | 0.6366485608635261 / 0.6353682350583597 / 0.834089669425433 |
| worst max drawdown | 0.04468751164160867 |
| n returns / n rollouts / n terminated | 72 / 72 / 0 |

## Integrity

| check | result |
|---|---|
| flat_ok | None |
| bh_ok | None |
