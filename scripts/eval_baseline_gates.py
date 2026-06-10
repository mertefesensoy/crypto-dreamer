"""ADR-007 evaluation/gates harness (Track C).

Loads a policy (final PPO checkpoint or a comparator), rolls it
DETERMINISTICALLY over the frozen eval episode set
(artifacts/adr007/eval_episodes.json), computes metric PRIMITIVES per
ADR-007 Sections (C)/(D), and writes:

- a JSON artifact  artifacts/adr007/eval_<policy>[_seed<NN>]_<subset>.json
  - the SOLE input to Phase-3 gate classification (artifact authority,
  ADR-007 (D)); floats serialized at full float64 round-trip precision
  via json.dump's default float repr;
- a markdown artifact (same basename, .md) - display-only, never a
  classification input;
- one appended line per invocation to artifacts/adr007/run_log.md, so
  off-the-books gate reads are visible (ADR-007 anti-gaming requirement).

This script COMPUTES and RECORDS; Phase 3 CLASSIFIES. It never reads W&B.
It mirrors scripts/eval_gates.py conventions: checkpoint-driven, fixed
seed 42, deterministic eval, on-disk outputs. Default device is cpu
(ADR-007 (C): CPU evaluation guarantees bitwise repeatability, (G)(vii)).

Subsets
-------
- smoke : ONLY the pre-named smoke episode (enumerated index 0,
  2024-05-28 00:00). EVERY policy - including bh - runs a single
  1440-step episode at that episode's start_row. bh's per-span 4320-step
  semantics run only in subset=full: a 4320-step span rollout would touch
  bars beyond the first enumerated episode, while the ADR-007 pre-named
  eval-contact exception covers exactly "the first enumerated episode
  plus the comparators on that same episode". Smoke verifies harness
  plumbing only and selects nothing.
- full  : agent/flat/random run the 72 enumerated 1440-step episodes in
  order; bh runs the 24 per-span 4320-step episodes (constant action 4)
  per ADR-007 (C)/(D). Before the official gate read, full runs are
  prohibited outside sanctioned invocations - every run is logged.

Integrity preconditions (recorded in the artifact; exit 3 on violation)
-----------------------------------------------------------------------
- flat : every per-episode r_i == 0.0 exactly AND every per-episode
  turnover == 0.0 exactly (ADR-007 (C)).
- bh   : per span, |env span cumulative - closed-form kline value| <= 1e-4
  (entry at span-start close paying 0.1% taker fee + 2 bps slippage, mark
  at span-end close). In smoke, the analogous 1440-step check runs
  (closes at start_row and start_row + 1440).

Exit codes: 0 ok - 2 episodes-artifact SHA-256 mismatch - 3 integrity
failure (artifacts and run-log line are still written first).

Usage
-----
    uv run python -m scripts.eval_baseline_gates \
        --policy {agent,bh,flat,random} --subset {smoke,full} \
        --purpose "<free text>" [--ckpt <path>] \
        [--episodes artifacts/adr007/eval_episodes.json] [--device cpu]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from training.baseline_policies import bh_closed_form_logret, get_policy
from training.ppo_env import MarketData, load_market_data, make_eval_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "adr007"
RUN_LOG_PATH = ARTIFACT_DIR / "run_log.md"

SCHEMA = "adr007-eval-v1"
EPISODES_SCHEMA = "adr007-eval-episodes-v1"
INITIAL_CASH = 10_000.0
EPISODE_STEPS = 1440
BH_SPAN_STEPS = 4320
BH_INTERVALS_PER_SPAN = 3
BH_INTEGRITY_TOL = 1e-4
SHARPE_ANNUALIZER = math.sqrt(365.0)  # daily episodes, crypto trades 365 days
SHARPE_STD_GUARD = 1e-12  # std(r_i, ddof=1) below this -> Sharpe undefined (null)
HARNESS_SEED = 42  # belt-and-braces framework seeding (ADR-007 (C))


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _resolve(path: str) -> Path:
    """Absolute paths pass through; relative paths resolve against the repo root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _verify_episodes_hash(episodes_path: Path) -> str:
    """Refuse to run unless the episodes JSON bytes hash to the frozen SHA-256.

    The expected hash lives next to the artifact (eval_episodes.sha256, the
    value also recorded in run_log.md at freeze time). Mismatch -> exit 2
    (ADR-007 (B): the harness refuses a hash-mismatching artifact).
    """
    sha_path = episodes_path.with_suffix(".sha256")
    if not sha_path.exists():
        print(f"ERROR: frozen hash file not found: {sha_path}")
        sys.exit(2)
    expected = sha_path.read_text(encoding="utf-8").split()[0].strip().lower()
    actual = hashlib.sha256(episodes_path.read_bytes()).hexdigest()
    if actual != expected:
        print("ERROR: eval-episodes artifact SHA-256 mismatch (ADR-007 (B)).")
        print(f"  expected {expected}")
        print(f"  actual   {actual}")
        print("Refusing to run. HALT for operator review.")
        sys.exit(2)
    return actual


def _validate_spec(spec: dict) -> None:
    """Sanity-check the frozen episode artifact against its pinned schema."""
    if spec.get("schema") != EPISODES_SCHEMA:
        raise RuntimeError(f"unexpected episodes schema: {spec.get('schema')!r}")
    if spec.get("episode_steps") != EPISODE_STEPS:
        raise RuntimeError(f"episode_steps {spec.get('episode_steps')} != {EPISODE_STEPS}")
    if spec.get("bh_span_steps") != BH_SPAN_STEPS:
        raise RuntimeError(f"bh_span_steps {spec.get('bh_span_steps')} != {BH_SPAN_STEPS}")
    if spec.get("smoke_episode_index") != 0:
        raise RuntimeError("smoke_episode_index != 0 in frozen artifact")
    if len(spec.get("episodes", [])) != 72:
        raise RuntimeError(f"expected 72 episodes, got {len(spec.get('episodes', []))}")
    if len(spec.get("spans", [])) != 24:
        raise RuntimeError(f"expected 24 spans, got {len(spec.get('spans', []))}")
    for i, ep in enumerate(spec["episodes"]):
        if ep["index"] != i:
            raise RuntimeError(f"episode list out of order at position {i}")


def _assert_start_ts(market: MarketData, start_row: int, start_ts: str, label: str) -> None:
    """Belt-and-braces (G)(vi) cross-check: snapshot ts at the row must equal
    the frozen artifact timestamp. A mismatch means snapshot/artifact desync
    and is a HALT, not something to roll over."""
    expected = int(pd.Timestamp(start_ts).value) // 1_000_000_000
    actual = int(market.ts_seconds[start_row])
    if expected != actual:
        raise RuntimeError(
            f"{label}: snapshot ts at row {start_row} != frozen artifact "
            f"start_ts {start_ts} (ADR-007 (G)(vi) identity violated)"
        )


def _read_ckpt_seed(ckpt_path: Path) -> int | None:
    """Pull the training seed out of the agent checkpoint dict (Track A contract)."""
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    seed = ckpt.get("seed") if isinstance(ckpt, dict) else None
    if seed is None:
        print("WARNING: checkpoint carries no 'seed' key; artifact records seed=null")
        return None
    return int(seed)


# --------------------------------------------------------------------------
# rollout mechanics
# --------------------------------------------------------------------------


def _rollout(env, policy, episode_index: int, start_row: int):
    """Roll one episode to terminated/truncated; return per-step equity and
    turnover (the StepInfo fields the primitives are built from).

    Contract: env is a BaselineSpotEnv whose reset(options={'start_row': s})
    pins the start deterministically; policy.reset(episode_index) is called
    exactly once before the episode (RandomPolicy's seed-7 stream continues
    across episodes in enumerated order - reset never reseeds).
    """
    policy.reset(episode_index)
    obs, _ = env.reset(options={"start_row": int(start_row)})
    equities: list[float] = []
    turnovers: list[float] = []
    terminated = truncated = False
    for _ in range(env.episode_steps + 2):  # hard cap: env must truncate itself
        action = policy.act(obs)
        obs, _reward, terminated, truncated, info = env.step(action)
        si = info["step"]
        equities.append(float(si.equity))
        turnovers.append(float(si.turnover))
        if terminated or truncated:
            break
    else:
        raise RuntimeError("env failed to terminate/truncate within episode_steps")
    return equities, turnovers, bool(terminated)


def _max_drawdown(equities: list[float]) -> float:
    """max over t of 1 - E_t/peak_t, with E_0 = 10000.0 prepended (ADR-007 (D))."""
    curve = np.concatenate(
        [[INITIAL_CASH], np.asarray(equities, dtype=np.float64)]
    )
    peak = np.maximum.accumulate(curve)
    return float(np.max(1.0 - curve / peak))


def _sharpe(r: np.ndarray) -> float | None:
    """mean/std(ddof=1)*sqrt(365); std < 1e-12 (or n < 2) -> undefined (null).
    Same estimator in the guard as in the formula (ADR-007 (D))."""
    if len(r) < 2:
        return None
    sd = float(np.std(r, ddof=1))
    if sd < SHARPE_STD_GUARD:
        return None
    return float(np.mean(r) / sd * SHARPE_ANNUALIZER)


def run_standard(policy, market: MarketData, episodes: list[dict]) -> dict:
    """agent/flat/random: 1440-step episodes at the enumerated start rows, in order."""
    env = make_eval_env(market, episode_steps=EPISODE_STEPS)
    r_i, turnover_i, drawdown_i = [], [], []
    terminated_l, steps_l, start_row_l, start_ts_l, index_l = [], [], [], [], []
    for ep in episodes:
        _assert_start_ts(market, ep["start_row"], ep["start_ts"], f"episode {ep['index']}")
        equities, turnovers, terminated = _rollout(env, policy, ep["index"], ep["start_row"])
        # Early termination counts as complete (operator amendment A1).
        r_i.append(math.log(equities[-1] / INITIAL_CASH))
        turnover_i.append(float(np.sum(np.asarray(turnovers, dtype=np.float64))))
        drawdown_i.append(_max_drawdown(equities))
        terminated_l.append(terminated)
        steps_l.append(len(equities))
        start_row_l.append(int(ep["start_row"]))
        start_ts_l.append(str(ep["start_ts"]))
        index_l.append(int(ep["index"]))
    return {
        "r_i_definition": "per-episode net log-return ln(E_end / 10000.0)",
        "r_i": r_i,
        "turnover_i": turnover_i,
        "drawdown_i": drawdown_i,
        "terminated": terminated_l,
        "steps": steps_l,
        "start_row": start_row_l,
        "start_ts": start_ts_l,
        "episode_index": index_l,
    }


def run_bh(policy, market: MarketData, spec: dict, subset: str):
    """Buy-and-hold comparator.

    full : 24 per-span 4320-step episodes (entry costs paid once per span);
           r_i = the 72 interval returns ln(E_k[j*1440]/E_k[(j-1)*1440]),
           j = 1..3 per span; per-span closed-form integrity check.
    smoke: a single 1440-step episode at the pre-named smoke episode's
           start_row - bh's per-span 4320-step semantics run only in
           subset=full, because a 4320-step rollout would contact bars
           beyond the first enumerated episode while the ADR-007 pre-named
           eval-contact exception covers only that episode (plus the
           comparators on it). The analogous 1440-step closed-form
           integrity check runs instead.
    """
    if subset == "smoke":
        ep0 = spec["episodes"][spec["smoke_episode_index"]]
        spans = [
            {
                "month": ep0["month"],
                "span_start_ts": ep0["start_ts"],
                "span_start_row": ep0["start_row"],
            }
        ]
        steps = EPISODE_STEPS
    else:
        spans = spec["spans"]
        steps = BH_SPAN_STEPS

    env = make_eval_env(market, episode_steps=steps)
    close = market.df["close"]

    r_i: list[float] = []  # full: 72 interval returns; smoke: 1 episode return
    turnover_i, drawdown_i = [], []
    terminated_l, steps_l, start_row_l, start_ts_l = [], [], [], []
    span_records = []
    all_pass = True

    for k, span in enumerate(spans):
        start_row = int(span["span_start_row"])
        _assert_start_ts(market, start_row, span["span_start_ts"], f"bh span {k}")
        equities, turnovers, terminated = _rollout(env, policy, k, start_row)

        # Equity curve E_k with E_k(0) := 10000 prepended (initial cash,
        # BEFORE the entry trade); after early termination the terminal
        # equity is carried forward flat so the curve has steps+1 points
        # (ADR-007 (D) / amendment A1 carry-forward rule).
        curve = [INITIAL_CASH] + equities
        while len(curve) < steps + 1:
            curve.append(curve[-1])

        if subset == "smoke":
            r_i.append(math.log(curve[steps] / INITIAL_CASH))
        else:
            for j in range(1, BH_INTERVALS_PER_SPAN + 1):
                r_i.append(math.log(curve[j * EPISODE_STEPS] / curve[(j - 1) * EPISODE_STEPS]))

        span_cumulative = math.log(curve[steps] / INITIAL_CASH)
        closed_form = bh_closed_form_logret(
            float(close.iloc[start_row]), float(close.iloc[start_row + steps])
        )
        abs_diff = abs(span_cumulative - closed_form)
        ok = abs_diff <= BH_INTEGRITY_TOL
        all_pass = all_pass and ok
        if not ok:
            print(
                f"INTEGRITY FAIL: bh span {span.get('month')} env cumulative "
                f"{span_cumulative!r} vs closed-form {closed_form!r} "
                f"(|diff| {abs_diff!r} > {BH_INTEGRITY_TOL})"
            )

        turnover_i.append(float(np.sum(np.asarray(turnovers, dtype=np.float64))))
        drawdown_i.append(_max_drawdown(equities))
        terminated_l.append(terminated)
        steps_l.append(len(equities))
        start_row_l.append(start_row)
        start_ts_l.append(str(span["span_start_ts"]))
        span_records.append(
            {
                "month": str(span.get("month")),
                "span_start_row": start_row,
                "span_start_ts": str(span["span_start_ts"]),
                "steps_contracted": steps,
                "steps_executed": len(equities),
                "terminated": terminated,
                "span_cumulative": span_cumulative,
                "closed_form": closed_form,
                "abs_diff": abs_diff,
                "integrity_pass": ok,
            }
        )

    primitives = {
        "r_i_definition": (
            "smoke: single 1440-step episode return ln(E_end / 10000.0)"
            if subset == "smoke"
            else "72 B&H interval returns ln(E_k[j*1440]/E_k[(j-1)*1440]), j=1..3 per span"
        ),
        "r_i": r_i,
        "turnover_i": turnover_i,  # per span (per smoke episode in smoke)
        "drawdown_i": drawdown_i,
        "terminated": terminated_l,
        "steps": steps_l,
        "start_row": start_row_l,
        "start_ts": start_ts_l,
        "spans": span_records,
    }
    integrity = {
        "flat_ok": None,
        "bh_ok": bool(all_pass),
        "bh_tolerance": BH_INTEGRITY_TOL,
        "bh_spans": [
            {
                "month": rec["month"],
                "span_cumulative": rec["span_cumulative"],
                "closed_form": rec["closed_form"],
                "abs_diff": rec["abs_diff"],
                "integrity_pass": rec["integrity_pass"],
            }
            for rec in span_records
        ],
        "notes": (
            "smoke runs the analogous 1440-step closed-form check"
            if subset == "smoke"
            else "per-span 4320-step closed-form check (ADR-007 (C))"
        ),
    }
    return primitives, integrity


def check_flat_integrity(primitives: dict) -> dict:
    """flat precondition (ADR-007 (C), scoped): every r_i == 0.0 exactly AND
    every per-episode turnover == 0.0 exactly. Violation = broken harness."""
    r_ok = all(v == 0.0 for v in primitives["r_i"])
    to_ok = all(v == 0.0 for v in primitives["turnover_i"])
    ok = r_ok and to_ok
    if not ok:
        print(
            "INTEGRITY FAIL: flat policy must produce exactly-zero returns and "
            f"turnover (returns_zero={r_ok}, turnover_zero={to_ok})"
        )
    return {
        "flat_ok": bool(ok),
        "flat_returns_all_zero": bool(r_ok),
        "flat_turnover_all_zero": bool(to_ok),
        "bh_ok": None,
        "bh_spans": [],
        "notes": "flat integrity precondition, exact == 0.0 comparison",
    }


def compute_aggregates(primitives: dict) -> dict:
    """Aggregates are computed and RECORDED here, never classified - Phase 3
    recomputes every gate expression in float64 from the stored primitives."""
    r = np.asarray(primitives["r_i"], dtype=np.float64)
    to = np.asarray(primitives["turnover_i"], dtype=np.float64)
    dd = np.asarray(primitives["drawdown_i"], dtype=np.float64)
    return {
        "R": float(np.sum(r)),
        "sharpe": _sharpe(r),
        "TO": float(np.sum(to)),
        "turnover_mean": float(np.mean(to)),
        "turnover_median": float(np.median(to)),
        "turnover_max": float(np.max(to)),
        "max_drawdown_worst": float(np.max(dd)),
        "n_returns": int(len(r)),
        "n_rollouts": int(len(to)),
        "n_terminated": int(sum(1 for t in primitives["terminated"] if t)),
    }


# --------------------------------------------------------------------------
# artifacts
# --------------------------------------------------------------------------


def _artifact_basename(policy: str, subset: str, seed: int | None) -> str:
    seed_part = ""
    if policy == "agent":
        seed_part = f"_seed{seed if seed is not None else 'NA'}"
    return f"eval_{policy}{seed_part}_{subset}"


def write_json_artifact(path: Path, payload: dict) -> None:
    # json.dump default float serialization = Python repr round-trip,
    # i.e. full float64 precision (ADR-007 (D) artifact authority).
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def write_markdown_artifact(path: Path, payload: dict) -> None:
    a = payload["aggregates"]
    integ = payload["integrity"]
    lines = [
        f"# ADR-007 eval - {payload['policy']} - {payload['subset']} (display-only)",
        "",
        "Display-only artifact. The JSON file with the same basename is the",
        "SOLE classification input (ADR-007 (D) artifact authority).",
        "",
        "| field | value |",
        "|---|---|",
        f"| policy | {payload['policy']} |",
        f"| subset | {payload['subset']} |",
        f"| purpose | {payload['purpose']} |",
        f"| device | {payload['device']} |",
        f"| episodes_sha256 | {payload['episodes_sha256']} |",
        f"| ckpt_path | {payload['ckpt_path']} |",
        f"| ckpt_sha256 | {payload['ckpt_sha256']} |",
        f"| seed | {payload['seed']} |",
        f"| generated_utc | {payload['generated_utc']} |",
        "",
        "## Aggregates (computed, not classified here)",
        "",
        "| metric | value |",
        "|---|---|",
        f"| R (sum r_i) | {a['R']!r} |",
        f"| sharpe | {a['sharpe']!r} |",
        f"| TO (total turnover) | {a['TO']!r} |",
        f"| turnover mean / median / max | {a['turnover_mean']!r} / "
        f"{a['turnover_median']!r} / {a['turnover_max']!r} |",
        f"| worst max drawdown | {a['max_drawdown_worst']!r} |",
        f"| n returns / n rollouts / n terminated | {a['n_returns']} / "
        f"{a['n_rollouts']} / {a['n_terminated']} |",
        "",
        "## Integrity",
        "",
        "| check | result |",
        "|---|---|",
        f"| flat_ok | {integ.get('flat_ok')} |",
        f"| bh_ok | {integ.get('bh_ok')} |",
    ]
    if integ.get("bh_spans"):
        lines += [
            "",
            "| span month | env cumulative | closed form | abs diff | pass |",
            "|---|---|---|---|---|",
        ]
        for rec in integ["bh_spans"]:
            lines.append(
                f"| {rec['month']} | {rec['span_cumulative']!r} | "
                f"{rec['closed_form']!r} | {rec['abs_diff']!r} | "
                f"{rec['integrity_pass']} |"
            )
    lines.append("")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def append_run_log(purpose: str, policy: str, subset: str, ckpt_sha256: str | None, R: float) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ck = ckpt_sha256[:8] if ckpt_sha256 else "none"
    safe_purpose = purpose.replace("|", "/").replace("\n", " ").replace("\r", " ")
    line = f"| {ts} | {safe_purpose} | eval_{policy}_{subset} | ckpt_sha256={ck} | R={R!r} |\n"
    prefix = ""
    if RUN_LOG_PATH.exists():
        existing = RUN_LOG_PATH.read_text(encoding="utf-8")
        if existing and not existing.endswith("\n"):
            prefix = "\n"
    with open(RUN_LOG_PATH, "a", encoding="utf-8", newline="\n") as f:
        f.write(prefix + line)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="ADR-007 baseline eval harness: computes and records metric "
        "primitives from deterministic env rollouts; Phase 3 classifies."
    )
    ap.add_argument("--policy", required=True, choices=["agent", "bh", "flat", "random"])
    ap.add_argument("--subset", required=True, choices=["smoke", "full"])
    ap.add_argument("--purpose", required=True, help="free text, appended to the run log")
    ap.add_argument("--ckpt", default=None, help="agent checkpoint path (required iff --policy agent)")
    ap.add_argument("--episodes", default="artifacts/adr007/eval_episodes.json")
    ap.add_argument("--device", default="cpu", help="cpu (default) for bitwise repeatability")
    args = ap.parse_args()

    if args.policy == "agent" and args.ckpt is None:
        ap.error("--ckpt is required when --policy agent")
    if args.policy != "agent" and args.ckpt is not None:
        ap.error("--ckpt is only valid with --policy agent")

    # Belt-and-braces determinism (ADR-007 (C)): with enumerated starts and
    # argmax/constant/dedicated-RNG actions no rollout stochasticity should
    # remain; seed framework RNGs anyway.
    torch.manual_seed(HARNESS_SEED)
    np.random.seed(HARNESS_SEED)

    episodes_path = _resolve(args.episodes)
    if not episodes_path.exists():
        print(f"ERROR: episodes artifact not found: {episodes_path}")
        return 2
    episodes_sha256 = _verify_episodes_hash(episodes_path)  # exits 2 on mismatch
    spec = json.loads(episodes_path.read_text(encoding="utf-8"))
    _validate_spec(spec)

    ckpt_path: Path | None = None
    ckpt_sha256: str | None = None
    seed: int | None = None
    if args.policy == "agent":
        ckpt_path = _resolve(args.ckpt)
        if not ckpt_path.exists():
            print(f"ERROR: checkpoint not found: {ckpt_path}")
            return 1
        ckpt_sha256 = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
        seed = _read_ckpt_seed(ckpt_path)

    market = load_market_data()  # frozen snapshot; asserts DREAMER_DATA binding
    policy = get_policy(
        args.policy,
        ckpt_path=str(ckpt_path) if ckpt_path is not None else None,
        device=args.device,
    )

    if args.policy == "bh":
        primitives, integrity = run_bh(policy, market, spec, args.subset)
    else:
        if args.subset == "smoke":
            episodes = [spec["episodes"][spec["smoke_episode_index"]]]
        else:
            episodes = spec["episodes"]
        primitives = run_standard(policy, market, episodes)
        if args.policy == "flat":
            integrity = check_flat_integrity(primitives)
        else:
            integrity = {
                "flat_ok": None,
                "bh_ok": None,
                "bh_spans": [],
                "notes": "no integrity precondition applies to this policy",
            }

    aggregates = compute_aggregates(primitives)

    payload = {
        "schema": SCHEMA,
        "policy": args.policy,
        "subset": args.subset,
        "purpose": args.purpose,
        "episodes_sha256": episodes_sha256,
        "ckpt_path": str(ckpt_path) if ckpt_path is not None else None,
        "ckpt_sha256": ckpt_sha256,
        "seed": seed,
        "device": args.device,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "primitives": primitives,
        "aggregates": aggregates,
        "integrity": integrity,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    base = _artifact_basename(args.policy, args.subset, seed)
    json_path = ARTIFACT_DIR / f"{base}.json"
    md_path = ARTIFACT_DIR / f"{base}.md"
    write_json_artifact(json_path, payload)
    write_markdown_artifact(md_path, payload)
    append_run_log(args.purpose, args.policy, args.subset, ckpt_sha256, aggregates["R"])

    integrity_ok = integrity.get("flat_ok") is not False and integrity.get("bh_ok") is not False
    print(f"policy={args.policy} subset={args.subset} device={args.device}")
    print(f"R={aggregates['R']!r} sharpe={aggregates['sharpe']!r} TO={aggregates['TO']!r}")
    print(f"json artifact     : {json_path}")
    print(f"markdown artifact : {md_path}")
    print(f"run log appended  : {RUN_LOG_PATH}")
    if not integrity_ok:
        print("RESULT: INTEGRITY FAILURE (recorded in artifact) -> exit 3")
        return 3
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
