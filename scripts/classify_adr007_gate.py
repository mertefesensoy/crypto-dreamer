"""ADR-007 Phase-3 mechanical gate classifier.

Reads ONLY the on-disk JSON eval artifacts in `artifacts/adr007/` (never
W&B, never the harness internals) and classifies the pre-registered
criteria G-BL1..G-BL4 of ADR-007 (docs/design/ARCHITECTURE.md Section 12)
exactly as written. Every derived expression is recomputed in float64 from
the stored primitives (per-episode r_i, per-episode turnover, the 72 B&H
interval returns) per gate section (D) · markdown artifacts are
display-only and are not read.

Preconditions enforced before classification:
- every artifact's episodes_sha256 equals the frozen
  `artifacts/adr007/eval_episodes.sha256`;
- flat integrity flags true; bh integrity flags true (amendment A3);
- the three agent checkpoints have pairwise-distinct ckpt_sha256
  (provenance · identical RESULTS are legitimate policy convergence,
  identical CHECKPOINTS would be an invocation error).

Outputs: a console table and `artifacts/adr007/gate_classification.json`;
appends one line to `artifacts/adr007/run_log.md`.

Usage:
    uv run python -m scripts.classify_adr007_gate
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ART = PROJECT_ROOT / "artifacts" / "adr007"

SEEDS = (42, 0, 123)
SHARPE_ANNUALIZER = math.sqrt(365.0)
SHARPE_STD_GUARD = 1e-12

# Pre-registered thresholds (ADR-007 (F), ratified 2026-06-11, frozen).
G_BL1_FLOOR = 0.010
G_BL1_STRESS = 0.0002
G_BL4_MEAN_CAP = 2.0
G_BL4_EPISODE_CAP = 10.0


def load(name: str) -> dict:
    with open(ART / name, encoding="utf-8") as f:
        return json.load(f)


def sharpe(r_i: list[float]) -> float | None:
    n = len(r_i)
    if n < 2:
        return None
    mean = sum(r_i) / n
    var = sum((x - mean) ** 2 for x in r_i) / (n - 1)
    std = math.sqrt(var)
    if std < SHARPE_STD_GUARD:
        return None
    return mean / std * SHARPE_ANNUALIZER


def derive(artifact: dict) -> dict:
    r_i = artifact["primitives"]["r_i"]
    to_i = artifact["primitives"]["turnover_i"]
    dd_i = artifact["primitives"].get("drawdown_i") or artifact["primitives"].get(
        "max_drawdown_i", []
    )
    return {
        "n_episodes": len(r_i),
        "R": sum(r_i),
        "sharpe": sharpe(r_i),
        "TO": sum(to_i),
        "mean_episode_turnover": sum(to_i) / len(to_i),
        "max_episode_turnover": max(to_i),
        "worst_drawdown": max(dd_i) if dd_i else None,
    }


def main() -> int:
    frozen_sha = (ART / "eval_episodes.sha256").read_text(encoding="utf-8").strip()

    agents = {s: load(f"eval_agent_seed{s}_full.json") for s in SEEDS}
    bh = load("eval_bh_full.json")
    flat = load("eval_flat_full.json")
    rnd = load("eval_random_full.json")

    # --- preconditions ------------------------------------------------------
    for name, art in [("flat", flat), ("bh", bh), ("random", rnd)] + [
        (f"agent seed {s}", agents[s]) for s in SEEDS
    ]:
        if art["episodes_sha256"] != frozen_sha:
            print(f"HALT: {name} artifact episodes_sha256 mismatch")
            return 1
    if flat["integrity"]["flat_ok"] is not True:
        print("HALT: flat integrity precondition not satisfied")
        return 1
    if bh["integrity"]["bh_ok"] is not True:
        print("HALT: bh integrity precondition (A3) not satisfied")
        return 1
    shas = {s: agents[s]["ckpt_sha256"] for s in SEEDS}
    if len(set(shas.values())) != len(SEEDS):
        print(f"HALT: agent checkpoints not pairwise distinct: {shas}")
        return 1

    # --- derived quantities (float64, from primitives only) ------------------
    a = {s: derive(agents[s]) for s in SEEDS}
    b = derive(bh)
    r = derive(rnd)
    f = derive(flat)
    R_BH = b["R"]
    S_BH = b["sharpe"]

    # --- designated median seed (ADR-007 (E)) --------------------------------
    by_R = sorted(SEEDS, key=lambda s: a[s]["R"])
    middle_value = a[by_R[1]]["R"]
    tied = [s for s in SEEDS if a[s]["R"] == middle_value]
    median_seed = min(tied)
    m = a[median_seed]

    # --- criteria (each binary, float64, exactly as pre-registered) ----------
    g1_lhs = m["R"] - G_BL1_STRESS * m["TO"]
    g1 = g1_lhs >= G_BL1_FLOOR
    g2 = m["R"] >= R_BH
    g3_rhs = max(S_BH, 0.0) if S_BH is not None else 0.0
    g3 = (m["sharpe"] is not None) and (m["sharpe"] >= g3_rhs)
    g4 = (m["mean_episode_turnover"] <= G_BL4_MEAN_CAP) and (
        m["max_episode_turnover"] <= G_BL4_EPISODE_CAP
    )
    overall = g1 and g2 and g3 and g4

    result = {
        "schema": "adr007-gate-classification-v1",
        "classified_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "episodes_sha256": frozen_sha,
        "agent_ckpt_sha256": shas,
        "per_seed": a,
        "comparators": {"bh": b, "flat": f, "random_report_only": r},
        "median_designation": {
            "seeds_sorted_by_R": by_R,
            "middle_value": middle_value,
            "tied_at_middle": tied,
            "designated_median_seed": median_seed,
        },
        "criteria": {
            "G-BL1": {
                "definition": "R - 0.0002*TO >= 0.010",
                "lhs": g1_lhs,
                "threshold": G_BL1_FLOOR,
                "pass": g1,
            },
            "G-BL2": {
                "definition": "R >= R_BH",
                "lhs": m["R"],
                "threshold": R_BH,
                "pass": g2,
            },
            "G-BL3": {
                "definition": "S >= max(S_BH, 0.0); undefined S -> FAIL",
                "lhs": m["sharpe"],
                "threshold": g3_rhs,
                "pass": g3,
            },
            "G-BL4": {
                "definition": "mean ep turnover <= 2.0 AND every ep <= 10.0",
                "mean_episode_turnover": m["mean_episode_turnover"],
                "max_episode_turnover": m["max_episode_turnover"],
                "thresholds": [G_BL4_MEAN_CAP, G_BL4_EPISODE_CAP],
                "pass": g4,
            },
        },
        "verdict": "PASS" if overall else "FAIL",
    }

    out = ART / "gate_classification.json"
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")

    # --- console table --------------------------------------------------------
    print("per-seed derived quantities (from JSON primitives, float64):")
    for s in SEEDS:
        d = a[s]
        print(
            f"  seed {s:>3}: R={d['R']!r} S={d['sharpe']!r} TO={d['TO']!r} "
            f"meanTO={d['mean_episode_turnover']!r} maxTO={d['max_episode_turnover']!r} "
            f"worstDD={d['worst_drawdown']!r}"
        )
    print(f"  bh     : R={b['R']!r} S={b['sharpe']!r} TO={b['TO']!r}")
    print(f"  flat   : R={f['R']!r} TO={f['TO']!r}")
    print(f"  random : R={r['R']!r} S={r['sharpe']!r} TO={r['TO']!r} (report-only)")
    md = result["median_designation"]
    print(
        f"designated median seed: {md['designated_median_seed']} "
        f"(sorted {md['seeds_sorted_by_R']}, middle value {md['middle_value']!r}, "
        f"tied {md['tied_at_middle']})"
    )
    for k, c in result["criteria"].items():
        print(f"  {k}: {'PASS' if c['pass'] else 'FAIL'} · {c['definition']}")
    print(f"VERDICT: {result['verdict']}")
    print(f"classification artifact: {out}")

    stamp = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    with open(ART / "run_log.md", "a", encoding="utf-8", newline="\n") as fh:
        fh.write(
            f"| {stamp} | gate-classification | median seed {median_seed} | "
            f"G-BL1={g1} G-BL2={g2} G-BL3={g3} G-BL4={g4} | VERDICT {result['verdict']} |\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
