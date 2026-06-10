| 2026-06-10T20:42:25+00:00 | freeze | eval_episodes.json | sha256=1842d800900b871733414e6d71e068f188e6ce8cc0de2a923035cab43d9811f1 | 24 spans / 72 episodes, all contiguous |
| 2026-06-10T20:42:25+00:00 | freeze | eval_episodes.json | sha256=1842d800900b871733414e6d71e068f188e6ce8cc0de2a923035cab43d9811f1 | 24 spans / 72 episodes, all contiguous |
| 2026-06-10T20:53:08+00:00 | preconditions | assert_adr007_preconditions | pass=i,ii,iii,v-pre,vi | seeds=42,0,123 n_envs=16; iv,vii deferred to eval harness |
| 2026-06-10T20:54:04+00:00 | preconditions | assert_adr007_preconditions | pass=i,ii,iii,v-pre,vi | seeds=42,0,123 n_envs=16; iv,vii deferred to eval harness |
| 2026-06-10T21:00:46+00:00 | smoke-selftest | eval_flat_smoke | ckpt_sha256=none | R=0.0 |
| 2026-06-10T21:01:20+00:00 | smoke-selftest | eval_random_smoke | ckpt_sha256=none | R=-0.687055603848776 |
| 2026-06-10T21:01:24+00:00 | smoke-selftest | eval_random_smoke | ckpt_sha256=none | R=-0.687055603848776 |
| 2026-06-10T21:02:08+00:00 | smoke-selftest | eval_bh_smoke | ckpt_sha256=none | R=-0.02026494547643154 |
| 2026-06-10T21:06:53+00:00 | smoke | eval_agent_smoke | ckpt_sha256=7ca928c1 | R=-0.016692369456567525 |
| 2026-06-10T21:06:58+00:00 | smoke-determinism-recheck | eval_agent_smoke | ckpt_sha256=7ca928c1 | R=-0.016692369456567525 |
| 2026-06-10T21:07:29+00:00 | smoke-determinism-recheck | eval_agent_smoke | ckpt_sha256=7ca928c1 | R=-0.016692369456567525 |
| 2026-06-10T21:07:50+00:00 | preconditions | assert_adr007_preconditions | pass=i,ii,iii,v-pre,vi | seeds=42,0,123 n_envs=16; iv,vii deferred to eval harness |
| 2026-06-10T21:08:09+00:00 | config-freeze-proposed | configs/ppo_baseline.yaml | sha256=d02454548ea55182034fb4f063cbc12832fa11510178591b3ac5a0c3c5288858 | binds on operator launch approval; run of record = first completed 3-seed run after approval |
| 2026-06-10T21:17:40+00:00 | run-of-record-launch | train_ppo seeds 42,0,123 | config_sha256=d02454548ea55182034fb4f063cbc12832fa11510178591b3ac5a0c3c5288858 episodes_sha256=1842d800900b871733414e6d71e068f188e6ce8cc0de2a923035cab43d9811f1 | operator launch approval 2026-06-11; freeze binding |
| 2026-06-10T22:05:18+00:00 | preconditions | assert_adr007_preconditions | pass=i,ii,iii,v-pre,vi | seeds=42,0,123 n_envs=16; iv,vii deferred to eval harness |
| 2026-06-10T22:05:39+00:00 | run-of-record-complete | seeds 42,0,123 x 2,000,000 env steps | final ckpts ppo_baseline_seed(42,0,123)_step2000000.ckpt | wall 2026-06-10T21:17:57Z..22:04:23Z; 0 nonfinite; 4176 realized train starts verified train-pure |
