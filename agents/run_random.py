"""
Run a random agent on the spot environment and publish each step to Redis.

Write paths:
  - PUBLISH dreamer:steps / dreamer:episodes  (live tail for connected clients)
  - XADD    dreamer:steps:hist (MAXLEN ~ 5,000,000) and
            dreamer:episodes:hist (no maxlen) for replay / hydration
  - INSERT  data/market.duckdb table step_log (one row per step)

Run:
    python -m agents.run_random --episode-hours 24 --speed 50
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import duckdb
import numpy as np
import redis

from envs.spot_btc import SpotBTCEnv

CHANNEL_STEPS = "dreamer:steps"
CHANNEL_EPISODES = "dreamer:episodes"
STREAM_STEPS = "dreamer:steps:hist"
STREAM_EPISODES = "dreamer:episodes:hist"
STREAM_MAXLEN = 5_000_000

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = Path(
    os.environ.get("DREAMER_DATA", PROJECT_ROOT / "data" / "market.duckdb")
)


def random_action(rng: np.random.Generator, n: int) -> int:
    """Uniform over the 5 target allocations. The dumbest possible policy."""
    return int(rng.integers(0, n))


def init_step_log(con: duckdb.DuckDBPyConnection) -> None:
    """Create the step_log table if it doesn't exist. Idempotent."""
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS step_log (
            ts TIMESTAMP,
            episode INTEGER,
            step INTEGER,
            action INTEGER,
            target_alloc DOUBLE,
            realized_alloc DOUBLE,
            equity DOUBLE,
            reward DOUBLE,
            turnover DOUBLE,
            fee_paid DOUBLE,
            agent_id VARCHAR
        )
        """
    )


def insert_step(
    con: duckdb.DuckDBPyConnection,
    ts: str,
    episode: int,
    step: int,
    payload: dict,
    agent_id: str,
) -> None:
    con.execute(
        "INSERT INTO step_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ts,
            episode,
            step,
            payload["action"],
            payload["target_alloc"],
            payload["realized_alloc"],
            payload["equity"],
            payload["reward"],
            payload["turnover"],
            payload["fee_paid"],
            agent_id,
        ],
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episode-hours", type=int, default=24)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument(
        "--speed",
        type=float,
        default=50.0,
        help="Steps per second (cap). Set high for fast replay, 1 for realtime.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--agent-id",
        default=None,
        help="Override agent identifier (default: random:<seed>).",
    )
    args = p.parse_args()
    agent_id = args.agent_id or f"random:{args.seed}"

    r = redis.Redis(
        host=os.environ.get("DREAMER_REDIS_HOST", "localhost"),
        port=int(os.environ.get("DREAMER_REDIS_PORT", "6379")),
        decode_responses=True,
    )
    r.ping()

    env = SpotBTCEnv(episode_steps=args.episode_hours * 60, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    n_actions = env.action_space.n
    period = 1.0 / args.speed

    # Open the DuckDB writer after the env has read its klines and closed.
    con = duckdb.connect(str(DEFAULT_DB_PATH))
    init_step_log(con)
    uniform_probs = [1.0 / n_actions] * n_actions

    try:
        for ep in range(args.episodes):
            env.reset()
            done = False
            ep_return = 0.0
            ep_steps = 0
            t_last = time.perf_counter()

            while not done:
                a = random_action(rng, n_actions)
                _, reward, terminated, truncated, info = env.step(a)
                ep_return += reward
                ep_steps += 1

                payload = asdict(info["step"])
                ts_iso = payload["ts"].isoformat()
                payload["ts"] = ts_iso
                payload["episode"] = ep
                payload["step"] = ep_steps
                payload["action_probs"] = uniform_probs
                payload["agent_id"] = agent_id
                payload_json = json.dumps(payload)

                # Live tail
                r.publish(CHANNEL_STEPS, payload_json)
                # History (capped stream)
                r.xadd(
                    STREAM_STEPS,
                    {"data": payload_json},
                    maxlen=STREAM_MAXLEN,
                    approximate=True,
                )
                # Persistent log
                insert_step(con, ts_iso, ep, ep_steps, payload, agent_id)

                done = terminated or truncated
                elapsed = time.perf_counter() - t_last
                if elapsed < period:
                    time.sleep(period - elapsed)
                t_last = time.perf_counter()

            summary = {
                "episode": ep,
                "steps": ep_steps,
                "total_reward": ep_return,
                "final_equity": payload["equity"],
                "agent_id": agent_id,
            }
            summary_json = json.dumps(summary)
            r.publish(CHANNEL_EPISODES, summary_json)
            r.xadd(STREAM_EPISODES, {"data": summary_json})
            print(
                f"ep {ep}: steps={ep_steps} reward={ep_return:.4f} "
                f"final_equity={payload['equity']:.2f}"
            )
    finally:
        con.close()


if __name__ == "__main__":
    main()
