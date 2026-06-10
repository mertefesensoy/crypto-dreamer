# ADR-007 eval - bh - smoke (display-only)

Display-only artifact. The JSON file with the same basename is the
SOLE classification input (ADR-007 (D) artifact authority).

| field | value |
|---|---|
| policy | bh |
| subset | smoke |
| purpose | smoke-selftest |
| device | cpu |
| episodes_sha256 | 1842d800900b871733414e6d71e068f188e6ce8cc0de2a923035cab43d9811f1 |
| ckpt_path | None |
| ckpt_sha256 | None |
| seed | None |
| generated_utc | 2026-06-10T21:02:08+00:00 |

## Aggregates (computed, not classified here)

| metric | value |
|---|---|
| R (sum r_i) | -0.02026494547643154 |
| sharpe | None |
| TO (total turnover) | 1.001002435402735 |
| turnover mean / median / max | 1.001002435402735 / 1.001002435402735 / 1.001002435402735 |
| worst max drawdown | 0.03636781510442688 |
| n returns / n rollouts / n terminated | 1 / 1 / 0 |

## Integrity

| check | result |
|---|---|
| flat_ok | None |
| bh_ok | True |

| span month | env cumulative | closed form | abs diff | pass |
|---|---|---|---|---|
| 2024-05 | -0.02026494547643154 | -0.02028278163789558 | 1.7836161464040295e-05 | True |
