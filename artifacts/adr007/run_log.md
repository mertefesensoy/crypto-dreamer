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
| 2026-06-10T22:11:11+00:00 | phase3-integrity-precondition | eval_flat_full | ckpt_sha256=none | R=0.0 |
| 2026-06-10T22:11:28+00:00 | phase3-integrity-precondition | eval_bh_full | ckpt_sha256=none | R=-0.2627548090021279 |
| 2026-06-11T06:55:59+00:00 | phase3-integrity-precondition-A3 | eval_bh_full | ckpt_sha256=none | R=-0.2627548090021279 |
| 2026-06-11T06:56:21+00:00 | A3-verification | bh per-span env-vs-corrected-reference (tolerance 1e-5) | worst abs_diff=1.239837201655325e-07 | 24/24 PASS |

A3 per-span verification table (env span cumulative vs corrected closed-form reference):

| span | env_cumulative | reference | abs_diff | pass |
|---|---|---|---|---|
| 2024-05 | -0.018929376345863533 | -0.018929392963364388 | 1.6617500854521072e-08 | True |
| 2024-06 | -0.0006608839779609722 | -0.0006608827360283688 | 1.241932603415416e-09 | True |
| 2024-07 | -0.02345867830431379 | -0.023458704383696218 | 2.6079382427907083e-08 | True |
| 2024-08 | -0.0479762972926639 | -0.04797634461664924 | 4.7323985341574115e-08 | True |
| 2024-09 | 0.01637845246919644 | 0.016378470186096127 | 1.771689968690926e-08 | True |
| 2024-10 | 0.07014324247028146 | 0.0701433128233993 | 7.035311784531206e-08 | True |
| 2024-11 | 0.06499613111569613 | 0.06499619355017568 | 6.243447954468184e-08 | True |
| 2024-12 | -0.004056353846079447 | -0.004056355593699542 | 1.74762009464563e-09 | True |
| 2025-01 | 0.03225420487786266 | 0.03225423696574511 | 3.208788244835059e-08 | True |
| 2025-02 | -0.1197939103578459 | -0.11979403434156606 | 1.239837201655325e-07 | True |
| 2025-03 | -0.05667616354728854 | -0.056676219314852734 | 5.57675641915667e-08 | True |
| 2025-04 | 0.006252536468198011 | 0.006252544791217257 | 8.323019246025964e-09 | True |
| 2025-05 | -0.047836129225316534 | -0.047836175699369876 | 4.6474053341794e-08 | True |
| 2025-06 | -0.005279343974947854 | -0.005279347022433176 | 3.0474853222536846e-09 | True |
| 2025-07 | -0.014358470923318803 | -0.014358483626160028 | 1.2702841225079031e-08 | True |
| 2025-08 | -0.03507226317409272 | -0.03507229535334158 | 3.217924886278478e-08 | True |
| 2025-09 | 0.043896538879389345 | 0.043896584104357456 | 4.522496811071308e-08 | True |
| 2025-10 | -0.06446474449407652 | -0.06446480879686055 | 6.430278402802525e-08 | True |
| 2025-11 | 0.010660751347010567 | 0.010660764064490048 | 1.2717479480964244e-08 | True |
| 2025-12 | 0.002628174670631728 | 0.0026281795684589405 | 4.8978272123786915e-09 | True |
| 2026-01 | -0.06278822860197292 | -0.06278829206497481 | 6.346300189530307e-08 | True |
| 2026-02 | 0.018405627770577637 | 0.018405649504055408 | 2.1733477770929932e-08 | True |
| 2026-03 | 0.007962914899116116 | 0.0079629247721108 | 9.872994683954306e-09 | True |
| 2026-04 | -0.03498253990434633 | -0.034982573142052026 | 3.323770569885198e-08 | True |

| 2026-06-11T06:56:56+00:00 | phase3-integrity-precondition-postA3 | eval_flat_full | ckpt_sha256=none | R=0.0 |
| 2026-06-11T06:57:05+00:00 | phase3-comparator | eval_random_full | ckpt_sha256=none | R=-49.446761409460365 |
| 2026-06-11T06:59:01+00:00 | phase3-gate-eval | eval_agent_full | ckpt_sha256=68a9c65d | R=-0.1674535432734741 |
| 2026-06-11T07:00:56+00:00 | phase3-gate-eval | eval_agent_full | ckpt_sha256=891f999d | R=-0.16919859808246518 |
| 2026-06-11T07:02:52+00:00 | phase3-gate-eval | eval_agent_full | ckpt_sha256=d55aa775 | R=-0.1674535432734741 |
| 2026-06-11T07:04:12+00:00 | gate-classification | median seed 42 | G-BL1=False G-BL2=True G-BL3=False G-BL4=True | VERDICT FAIL |
