"""ADR-007 findings support · per-seed action distribution over the eval set.

READ-ONLY diagnostic, run AFTER the Phase-3 gate classification (operator
ruling, 2026-06-11): replays the three final checkpoints deterministically
over the 72 frozen eval episodes (same argmax policy and env construction
as the harness, via the same public APIs) and tallies the action histogram.
Does not touch the gate harness, writes no eval artifact · output goes to
stdout and `artifacts/adr007/action_distribution.json` (findings-doc
support only, not a gate input).

Usage:
    uv run python -m scripts.adr007_action_distribution
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from training.baseline_policies import get_policy
from training.ppo_env import load_market_data, make_eval_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ART = PROJECT_ROOT / "artifacts" / "adr007"
SEEDS = (42, 0, 123)


def main() -> None:
    spec = json.load(open(ART / "eval_episodes.json", encoding="utf-8"))
    market = load_market_data()
    out: dict[str, dict] = {}

    for seed in SEEDS:
        ckpt = PROJECT_ROOT / "checkpoints" / f"ppo_baseline_seed{seed}_step2000000.ckpt"
        policy = get_policy("agent", ckpt_path=str(ckpt), device="cpu")
        env = make_eval_env(market, episode_steps=spec["episode_steps"])
        counts: Counter[int] = Counter()
        per_episode_modes = []
        for ep in spec["episodes"]:
            policy.reset(ep["index"])
            obs, _ = env.reset(options={"start_row": ep["start_row"]})
            ep_counts: Counter[int] = Counter()
            while True:
                a = policy.act(obs)
                ep_counts[a] += 1
                obs, _r, term, trunc, _info = env.step(a)
                if term or trunc:
                    break
            counts.update(ep_counts)
            per_episode_modes.append(ep_counts.most_common(1)[0][0])
        total = sum(counts.values())
        hist = {str(a): counts.get(a, 0) for a in range(5)}
        frac = {str(a): counts.get(a, 0) / total for a in range(5)}
        out[str(seed)] = {
            "steps": total,
            "action_counts": hist,
            "action_fractions": frac,
            "episode_modal_actions": dict(Counter(per_episode_modes)),
        }
        print(f"seed {seed}: steps={total} counts={hist} fractions=" +
              "{" + ", ".join(f"{k}: {v:.4f}" for k, v in frac.items()) + "}")

    payload = {
        "schema": "adr007-action-distribution-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "read-only findings support, generated after gate classification; not a gate input",
        "per_seed": out,
    }
    with open(ART / "action_distribution.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"written: {ART / 'action_distribution.json'}")


if __name__ == "__main__":
    main()
