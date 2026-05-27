"""Diagnose the 2 h stall in run lleske3b around step 7199.

Three checks:
1. W&B history gap detection — find points where _runtime advanced
   without trainer/global_step keeping pace. Confirms wandb sync stall
   if the gap is one big bin rather than gradual slowdown.
2. Wall-clock window for the stall (computed from W&B _runtime offsets).
3. Verify no concurrent DuckDB writers during stall window (check
   data/market.duckdb.wal mtime).

Run:
    python -m scripts.diagnose_stall <run_id>
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTITY = "sensoymertefe-ted-niversitesi"
PROJECT = "crypto-dreamer"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_id")
    p.add_argument("--launch-time", default="2026-05-04 18:06:00",
                   help="Wall-clock launch time of the run, used to"
                        " convert _runtime offsets to absolute timestamps")
    args = p.parse_args()

    api = wandb.Api(timeout=60)
    run = api.run(f"{ENTITY}/{PROJECT}/{args.run_id}")
    df = run.history(keys=["_runtime", "trainer/global_step"], samples=5000, pandas=True)
    df = df.dropna(subset=["_runtime", "trainer/global_step"]).sort_values("trainer/global_step").reset_index(drop=True)

    # ------ Check 1: gap detection ------
    print("=== W&B history gap detection ===")
    df["dt"] = df["_runtime"].diff()
    df["ds"] = df["trainer/global_step"].diff()
    df["rate_ms"] = df["dt"] / df["ds"].where(df["ds"] > 0) * 1000
    big_gaps = df[df["dt"] > 60].sort_values("dt", ascending=False).head(10)
    print(f"Top 10 longest inter-log gaps (delta-runtime in seconds):")
    print(big_gaps[["trainer/global_step", "_runtime", "dt", "ds", "rate_ms"]].to_string(index=False))

    # ------ Check 2: wall-clock window ------
    print("\n=== Stall wall-clock window ===")
    launch = datetime.fromisoformat(args.launch_time)
    big_gap = df.loc[df["dt"].idxmax()] if df["dt"].max() > 60 else None
    if big_gap is not None:
        gap_end_ts = launch + timedelta(seconds=big_gap["_runtime"])
        gap_start_ts = gap_end_ts - timedelta(seconds=big_gap["dt"])
        print(f"Largest gap: {big_gap['dt']:.0f}s ({big_gap['dt']/60:.1f}min)")
        print(f"  step at end of gap: {big_gap['trainer/global_step']:.0f}")
        print(f"  start (estimated):  {gap_start_ts}")
        print(f"  end:                {gap_end_ts}")
    else:
        print("No gaps >60s found.")

    # ------ Check 3: DuckDB writers during gap ------
    print("\n=== Check 3: DuckDB writers ===")
    db = PROJECT_ROOT / "data" / "market.duckdb"
    wal = PROJECT_ROOT / "data" / "market.duckdb.wal"
    for f in [db, wal]:
        if f.exists():
            mt = datetime.fromtimestamp(f.stat().st_mtime)
            print(f"  {f.name:30s} mtime={mt} size={f.stat().st_size:,}b")
        else:
            print(f"  {f.name:30s} (does not exist)")

    if big_gap is not None and wal.exists():
        wal_mt = datetime.fromtimestamp(wal.stat().st_mtime)
        if gap_start_ts <= wal_mt <= gap_end_ts:
            print(f"  WARNING: WAL was modified DURING the stall window")
        else:
            print(f"  OK: WAL mtime is outside the stall window")

    return 0


if __name__ == "__main__":
    sys.exit(main())
