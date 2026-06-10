"""Materialize the ADR-007 pre-registered evaluation episode set.

Generates `artifacts/adr007/eval_episodes.json` mechanically from the rule
pre-registered in ADR-007 Section (B) (`docs/design/ARCHITECTURE.md`):

- 24 monthly val spans, 2024-05 .. 2026-04.
- Span start = 00:00 UTC of the first val day per the exact datamodule rule
  `(day - 1) / days_in_month >= 0.85` (`training/datamodule.py:396`).
- Per span: 3 non-overlapping 1440-step episodes at +0 h / +24 h / +48 h.
- Every span must be contiguous (4,321 bars at exact 1-min spacing from
  span start through +4320 minutes inclusive) in `data/market_ro.duckdb`.
- Every episode start must have >= 256 bars of history.

The derived first-val-day per month is cross-checked against the table
printed in ADR-007 (embedded below as `ADR_TABLE`). ANY mismatch is a HALT:
the script exits non-zero without writing the artifact.

The JSON contains no wall-clock or random content, so its SHA-256 is stable
across regenerations on the same snapshot. The SHA-256 is written to
`artifacts/adr007/eval_episodes.sha256` and appended to
`artifacts/adr007/run_log.md` (the run log carries wall time; the hashed
artifact does not).

Usage:
    uv run python -m scripts.make_adr007_eval_episodes
"""

from __future__ import annotations

import calendar
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = PROJECT_ROOT / "data" / "market_ro.duckdb"
OUT_DIR = PROJECT_ROOT / "artifacts" / "adr007"
OUT_JSON = OUT_DIR / "eval_episodes.json"
OUT_SHA = OUT_DIR / "eval_episodes.sha256"
RUN_LOG = OUT_DIR / "run_log.md"

VAL_MONTH_FRAC = 0.85  # training/datamodule.py:396
EPISODE_STEPS = 1440
EPISODES_PER_SPAN = 3
SPAN_BARS = EPISODES_PER_SPAN * EPISODE_STEPS  # 4320; span uses +4320 inclusive
MIN_HISTORY = 256

# The table as printed in ADR-007 Section (B). The derivation below must
# reproduce it exactly · any mismatch is a HALT.
ADR_TABLE: dict[str, int] = {
    "2024-05": 28, "2024-06": 27, "2024-07": 28, "2024-08": 28,
    "2024-09": 27, "2024-10": 28, "2024-11": 27, "2024-12": 28,
    "2025-01": 28, "2025-02": 25, "2025-03": 28, "2025-04": 27,
    "2025-05": 28, "2025-06": 27, "2025-07": 28, "2025-08": 28,
    "2025-09": 27, "2025-10": 28, "2025-11": 27, "2025-12": 28,
    "2026-01": 28, "2026-02": 25, "2026-03": 28, "2026-04": 27,
}

# Snapshot identity pinned by ADR-007 (A) / (G)(vi).
EXPECTED_ROWS = 1_051_201
EXPECTED_MIN_TS = "2024-05-03T03:00:00+00:00"
EXPECTED_MAX_TS = "2026-05-03T03:00:00+00:00"


def first_val_day(year: int, month: int) -> int:
    """First UTC day-of-month in the val partition per the datamodule rule."""
    dim = calendar.monthrange(year, month)[1]
    for day in range(1, dim + 1):
        if (day - 1) / dim >= VAL_MONTH_FRAC:
            return day
    raise RuntimeError(f"no val day found for {year}-{month:02d}")


def halt(msg: str) -> None:
    print(f"\nHALT · {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    # --- load the full ts column once (gap checks are slice arithmetic) ---
    con = duckdb.connect(str(SNAPSHOT), read_only=True)
    ts = con.execute(
        "SELECT ts FROM klines WHERE symbol = 'BTCUSDT' AND interval = '1m' "
        "ORDER BY ts"
    ).fetchnumpy()["ts"]
    con.close()

    ts_sec = ts.astype("datetime64[s]").astype(np.int64)
    n = len(ts_sec)

    def iso(sec: int) -> str:
        return datetime.fromtimestamp(int(sec), tz=timezone.utc).isoformat()

    if n != EXPECTED_ROWS:
        halt(f"snapshot row count {n} != pinned {EXPECTED_ROWS}")
    if iso(ts_sec[0]) != EXPECTED_MIN_TS or iso(ts_sec[-1]) != EXPECTED_MAX_TS:
        halt(
            f"snapshot span {iso(ts_sec[0])} .. {iso(ts_sec[-1])} != pinned "
            f"{EXPECTED_MIN_TS} .. {EXPECTED_MAX_TS}"
        )

    # --- derive months and cross-check the ADR table -----------------------
    months = [(y, m) for y in (2024, 2025, 2026) for m in range(1, 13)]
    months = [(y, m) for (y, m) in months if ("2024-05" <= f"{y}-{m:02d}" <= "2026-04")]
    if len(months) != 24:
        halt(f"derived {len(months)} months, expected 24")

    agreement: list[tuple[str, int, int, bool]] = []
    derived: dict[str, int] = {}
    for y, m in months:
        key = f"{y}-{m:02d}"
        d = first_val_day(y, m)
        derived[key] = d
        agreement.append((key, ADR_TABLE[key], d, ADR_TABLE[key] == d))
    if derived != ADR_TABLE:
        bad = [a for a in agreement if not a[3]]
        halt(f"derived first-val-days disagree with the ADR table: {bad}")

    # --- locate spans, verify contiguity, enumerate episodes ---------------
    spans = []
    episodes = []
    ep_idx = 0
    for key, day in sorted(ADR_TABLE.items()):
        y, m = int(key[:4]), int(key[5:7])
        span_start_sec = int(
            datetime(y, m, day, 0, 0, tzinfo=timezone.utc).timestamp()
        )
        row = int(np.searchsorted(ts_sec, span_start_sec))
        if row >= n or ts_sec[row] != span_start_sec:
            halt(f"{key}: span start {iso(span_start_sec)} not present in snapshot")
        end_row = row + SPAN_BARS  # +4320 minutes inclusive
        if end_row >= n:
            halt(f"{key}: span exceeds snapshot end")
        seg = ts_sec[row : end_row + 1]
        deltas = np.diff(seg)
        if len(seg) != SPAN_BARS + 1 or not np.all(deltas == 60):
            halt(
                f"{key}: span not contiguous · {len(seg)} bars, "
                f"bad deltas at {np.flatnonzero(deltas != 60)[:5]}"
            )
        if row < MIN_HISTORY:
            halt(f"{key}: span start row {row} has < {MIN_HISTORY} bars history")

        spans.append(
            {
                "month": key,
                "first_val_day": day,
                "span_start_ts": iso(span_start_sec),
                "span_start_row": row,
                "span_bars_inclusive": SPAN_BARS + 1,
            }
        )
        for j in range(EPISODES_PER_SPAN):
            start_row = row + j * EPISODE_STEPS
            episodes.append(
                {
                    "index": ep_idx,
                    "month": key,
                    "offset_hours": j * 24,
                    "start_ts": iso(int(ts_sec[start_row])),
                    "start_row": start_row,
                    "steps": EPISODE_STEPS,
                }
            )
            ep_idx += 1

    if len(episodes) != 72 or len(spans) != 24:
        halt(f"enumerated {len(spans)} spans / {len(episodes)} episodes, expected 24/72")

    # --- write the artifact (deterministic content, stable SHA) ------------
    payload = {
        "schema": "adr007-eval-episodes-v1",
        "rule": (
            "val iff (day_of_month - 1) / days_in_month >= 0.85 (UTC); "
            "span start 00:00 UTC of first val day; 3 episodes per span at "
            "+0/+24/+48 h; 1440 steps each; ADR-007 Section (B)"
        ),
        "snapshot": {
            "path": "data/market_ro.duckdb",
            "rows": EXPECTED_ROWS,
            "min_ts": EXPECTED_MIN_TS,
            "max_ts": EXPECTED_MAX_TS,
        },
        "episode_steps": EPISODE_STEPS,
        "bh_span_steps": SPAN_BARS,
        "smoke_episode_index": 0,
        "spans": spans,
        "episodes": episodes,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2) + "\n"
    OUT_JSON.write_text(text, encoding="ascii", newline="\n")
    sha = hashlib.sha256(text.encode("ascii")).hexdigest()
    OUT_SHA.write_text(sha + "\n", encoding="ascii", newline="\n")

    stamp = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    entry = (
        f"| {stamp} | freeze | eval_episodes.json | sha256={sha} | "
        f"24 spans / 72 episodes, all contiguous |\n"
    )
    if not RUN_LOG.exists():
        RUN_LOG.write_text(
            "# ADR-007 run log - append-only\n\n"
            "| utc | purpose | subject | identity | note |\n"
            "|---|---|---|---|---|\n",
            encoding="utf-8",
            newline="\n",
        )
    with RUN_LOG.open("a", encoding="utf-8", newline="\n") as f:
        f.write(entry)

    # --- report -------------------------------------------------------------
    print("ADR table vs derived first-val-day (must all be True):")
    for key, expected, got, ok in agreement:
        print(f"  {key}  adr={expected:2d}  derived={got:2d}  {'OK' if ok else 'MISMATCH'}")
    print(f"\nspans: {len(spans)}  episodes: {len(episodes)}  (all contiguous, "
          f">= {MIN_HISTORY} bars history)")
    print(f"artifact: {OUT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"sha256:   {sha}")


if __name__ == "__main__":
    main()
