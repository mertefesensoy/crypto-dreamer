"""ADR-007 gate section (G) - mechanical pre-training precondition assertions.

Track B of the ADR-007 Phase-2 plan (docs/design/ARCHITECTURE.md Section 12,
"(G) Apples-to-apples preconditions"). Verifies the items that are runnable
BEFORE any training:

- (i)   episode-set integrity: the frozen artifact's SHA-256 matches the
        recorded hash AND the 24 spans / 72 episodes are re-derived from the
        pre-registered rule and match the artifact exactly;
- (ii)  feature-pipeline identity: the shared MarketData feature block equals
        a pristine SpotBTCEnv's own computation bit-for-bit, FEATURE_NAMES is
        the pinned 12-name order, eval-span observations (incl. the 256-bar
        lookback) are finite / in [-10, 10] / float32, and BaselineSpotEnv
        resets at sample eval starts reproduce the declared spaces exactly;
- (iii) containment: every enumerated episode lies wholly inside the val
        partition with >= 256 bars of history;
- (v)   training-partition purity, pre-run half: >= 10,000 resets through the
        actual configured sampler per seed, every start train-pure; plus the
        post-run half against artifacts/adr007/train_starts_seed*.json when
        those files exist (SKIP if not yet written);
- (vi)  data-source identity: snapshot binding, pinned row count and ts span,
        kline ts at every episode start row equals the frozen artifact's ts.

Items (iv) (flat / B&H integrity preconditions) and (vii) (rollout
determinism) require policy rollouts and belong to the eval harness
(Track C, scripts/eval_baseline_gates.py). They are NOT implemented here;
a NOTE line records the deferral.

Partition-rule independence: this script re-derives the val mask directly
from the kline timestamps ((day - 1) / days_in_month >= 0.85, UTC) instead
of trusting MarketData.val_bar, so (iii) and (v) cross-check ppo_env's
implementation rather than restating it.

Eval-episode restriction: this script performs env RESETS ONLY - no policy
is constructed and no step() is taken - so no policy is evaluated on any
enumerated episode. The reset-identity samples at episode indices 0/35/71
are sanctioned by the Track B specification.

Exit code 0 iff every check passes; on full pass a row is appended to
artifacts/adr007/run_log.md. Any FAIL -> exit 1 and no run-log entry.

Usage:
    uv run python -m scripts.assert_adr007_preconditions \
        [--n-envs 16] [--seeds 42 0 123] \
        [--episodes artifacts/adr007/eval_episodes.json]
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from envs.spot_btc import FEATURE_NAMES, WINDOW, SpotBTCEnv
from training.ppo_env import (
    SNAPSHOT_DB,
    MarketData,
    assert_snapshot_binding,
    load_market_data,
    make_eval_env,
    make_training_envs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_LOG = PROJECT_ROOT / "artifacts" / "adr007" / "run_log.md"
TRAIN_STARTS_DIR = PROJECT_ROOT / "artifacts" / "adr007"

VAL_MONTH_FRAC = 0.85  # training/datamodule.py:396 - the single split rule
EPISODE_STEPS = 1440
SPAN_BARS = 4320  # B&H span: +4320 minutes inclusive -> 4321 bars
MONTHS = [
    f"{y}-{m:02d}"
    for y in (2024, 2025, 2026)
    for m in range(1, 13)
    if "2024-05" <= f"{y}-{m:02d}" <= "2026-04"
]

# Pinned identity per ADR-007 (A) / (G)(vi).
EXPECTED_ROWS = 1_051_201
EXPECTED_MIN_TS = "2024-05-03T03:00:00+00:00"
EXPECTED_MAX_TS = "2026-05-03T03:00:00+00:00"

# Pinned feature order per ADR-007 (G)(ii) - the wire contract.
PINNED_FEATURE_NAMES = (
    "log_ret", "vol_5", "vol_15", "vol_60",
    "rsi_14", "macd", "vol_z", "hl_range",
    "close_norm", "ret_5", "ret_15", "ret_60",
)

# Sample eval episode indices for the reset-identity check (reset only,
# never stepped; index 0 is the pre-named smoke episode).
RESET_SAMPLE_INDICES = (0, 35, 71)

MIN_SAMPLER_RESETS = 10_000


def _fail(tag: str, msg: str) -> bool:
    print(f"FAIL ({tag}) {msg}")
    return False


def _pass(tag: str, msg: str) -> bool:
    print(f"PASS ({tag}) {msg}")
    return True


def independent_val_prefix(market: MarketData) -> np.ndarray:
    """Prefix-sum of the val-bar mask re-derived from raw kline timestamps.

    val_count over the INCLUSIVE row range [a, b] = prefix[b + 1] - prefix[a].
    Recomputed here (not taken from MarketData.val_bar) so the partition rule
    is asserted independently of training/ppo_env.py.
    """
    ts_idx = pd.DatetimeIndex(market.df["ts"])
    day = ts_idx.day.to_numpy()
    dim = ts_idx.daysinmonth.to_numpy()
    val = ((day - 1) / dim) >= VAL_MONTH_FRAC
    return np.concatenate([[0], np.cumsum(val.astype(np.int64))])


def first_val_day(year: int, month: int) -> int:
    """First UTC day-of-month in the val partition per the datamodule rule."""
    dim = calendar.monthrange(year, month)[1]
    for day in range(1, dim + 1):
        if (day - 1) / dim >= VAL_MONTH_FRAC:
            return day
    raise RuntimeError(f"no val day found for {year}-{month:02d}")


def iso_to_epoch(iso: str) -> int:
    return int(datetime.fromisoformat(iso).timestamp())


# --------------------------------------------------------------------------
# (i) episode-set integrity
# --------------------------------------------------------------------------

def check_i(
    artifact: dict, artifact_bytes: bytes, sha_path: Path, market: MarketData
) -> bool:
    # 1. hash binding
    if not sha_path.exists():
        return _fail("i", f"sha file missing: {sha_path}")
    recorded = sha_path.read_text(encoding="ascii").split()[0]
    actual = hashlib.sha256(artifact_bytes).hexdigest()
    if actual != recorded:
        return _fail(
            "i", f"sha256 mismatch: artifact={actual} recorded={recorded}"
        )

    # 2. re-derive spans and episodes from the pre-registered rule
    if len(MONTHS) != 24:
        return _fail("i", f"derived {len(MONTHS)} months, expected 24")
    expected_spans = []
    expected_episodes = []
    ep_idx = 0
    for key in MONTHS:
        y, m = int(key[:4]), int(key[5:7])
        day = first_val_day(y, m)
        span_sec = int(datetime(y, m, day, 0, 0, tzinfo=timezone.utc).timestamp())
        row = int(np.searchsorted(market.ts_seconds, span_sec))
        if row >= len(market.ts_seconds) or market.ts_seconds[row] != span_sec:
            return _fail("i", f"{key}: derived span start ts not in snapshot")
        expected_spans.append(
            {
                "month": key,
                "first_val_day": day,
                "span_start_ts": datetime.fromtimestamp(
                    span_sec, tz=timezone.utc
                ).isoformat(),
                "span_start_row": row,
                "span_bars_inclusive": SPAN_BARS + 1,
            }
        )
        for j in range(3):
            off_sec = j * 24 * 3600
            expected_episodes.append(
                {
                    "index": ep_idx,
                    "month": key,
                    "offset_hours": j * 24,
                    "start_ts": datetime.fromtimestamp(
                        span_sec + off_sec, tz=timezone.utc
                    ).isoformat(),
                    "start_row": row + j * EPISODE_STEPS,
                    "steps": EPISODE_STEPS,
                }
            )
            ep_idx += 1

    # 3. exact match against the artifact
    problems = []
    if artifact.get("episode_steps") != EPISODE_STEPS:
        problems.append(f"episode_steps={artifact.get('episode_steps')}")
    if artifact.get("bh_span_steps") != SPAN_BARS:
        problems.append(f"bh_span_steps={artifact.get('bh_span_steps')}")
    if artifact.get("smoke_episode_index") != 0:
        problems.append(f"smoke_episode_index={artifact.get('smoke_episode_index')}")
    snap = artifact.get("snapshot", {})
    if (
        snap.get("rows") != EXPECTED_ROWS
        or snap.get("min_ts") != EXPECTED_MIN_TS
        or snap.get("max_ts") != EXPECTED_MAX_TS
    ):
        problems.append(f"snapshot block mismatch: {snap}")

    spans = artifact.get("spans", [])
    episodes = artifact.get("episodes", [])
    if len(spans) != 24 or len(episodes) != 72 or len(episodes) != 24 * 3:
        problems.append(f"counts: {len(spans)} spans / {len(episodes)} episodes")
    else:
        for got, exp in zip(spans, expected_spans):
            if got != exp:
                problems.append(f"span {exp['month']}: artifact={got} derived={exp}")
        for got, exp in zip(episodes, expected_episodes):
            if got != exp:
                problems.append(f"episode {exp['index']}: artifact={got} derived={exp}")

    if problems:
        return _fail("i", "artifact does not match rule re-derivation: "
                     + "; ".join(problems[:5]))
    return _pass(
        "i",
        f"sha256 {actual[:12]}.. matches; 24 spans / 72 episodes re-derived "
        "from the (day-1)/days_in_month >= 0.85 rule + 0/24/48 h offsets "
        "and match the frozen artifact exactly (72 == 24*3)",
    )


# --------------------------------------------------------------------------
# (ii) feature-pipeline identity
# --------------------------------------------------------------------------

def check_ii(artifact: dict, market: MarketData) -> bool:
    if tuple(FEATURE_NAMES) != PINNED_FEATURE_NAMES:
        return _fail("ii", f"FEATURE_NAMES drifted: {FEATURE_NAMES}")

    print("  (ii) building pristine SpotBTCEnv from the snapshot "
          "(recomputes the full feature block)...")
    pristine = SpotBTCEnv(db_path=str(SNAPSHOT_DB))
    if not np.array_equal(market.features, pristine._features):
        return _fail(
            "ii",
            "MarketData.features != pristine SpotBTCEnv._features (full-array "
            "np.array_equal)",
        )
    if market.features.dtype != np.float32:
        return _fail("ii", f"features dtype {market.features.dtype} != float32")
    if market.features.shape != (EXPECTED_ROWS, 12):
        return _fail("ii", f"features shape {market.features.shape}")

    # eval-span value bounds, lookback included: rows [row-256, row+4320]
    for span in artifact["spans"]:
        row = span["span_start_row"]
        block = market.features[row - WINDOW : row + SPAN_BARS + 1]
        if block.shape[0] != WINDOW + SPAN_BARS + 1:
            return _fail("ii", f"{span['month']}: span slice short: {block.shape}")
        if not np.isfinite(block).all():
            return _fail("ii", f"{span['month']}: NaN/Inf in eval-span features")
        if np.abs(block).max() > 10.0:
            return _fail(
                "ii",
                f"{span['month']}: |feature| max {np.abs(block).max():.4f} > 10",
            )

    # reset-identity at sample eval starts (reset only - no policy, no step)
    env = make_eval_env(market)
    for idx in RESET_SAMPLE_INDICES:
        ep = artifact["episodes"][idx]
        if ep["index"] != idx:
            return _fail("ii", f"episode list not index-ordered at {idx}")
        start = ep["start_row"]
        obs, _ = env.reset(options={"start_row": start})
        win, port = obs["window"], obs["portfolio"]
        if not np.array_equal(win, market.features[start - WINDOW : start]):
            return _fail("ii", f"episode {idx}: obs window != features[s-256:s]")
        if win.shape != (WINDOW, 12) or win.dtype != np.float32:
            return _fail("ii", f"episode {idx}: window {win.shape} {win.dtype}")
        if port.shape != (3,) or port.dtype != np.float32:
            return _fail("ii", f"episode {idx}: portfolio {port.shape} {port.dtype}")
        if not env.observation_space["window"].contains(win):
            return _fail("ii", f"episode {idx}: window outside declared Box")
        if not env.observation_space["portfolio"].contains(port):
            return _fail("ii", f"episode {idx}: portfolio outside declared Box")
        if not np.array_equal(port, np.array([0.0, 1.0, 0.0], dtype=np.float32)):
            return _fail("ii", f"episode {idx}: reset portfolio {port.tolist()}")

    return _pass(
        "ii",
        "feature-pipeline identity: shared features == pristine SpotBTCEnv "
        "computation (bitwise); FEATURE_NAMES pinned; eval spans (+256-bar "
        "lookback) finite, within [-10, 10], float32; reset identity at "
        f"episodes {list(RESET_SAMPLE_INDICES)} incl. portfolio [0, 1, 0]",
    )


# --------------------------------------------------------------------------
# (iii) containment
# --------------------------------------------------------------------------

def check_iii(artifact: dict, val_prefix: np.ndarray) -> bool:
    bad = []
    for ep in artifact["episodes"]:
        s = ep["start_row"]
        n_bars = EPISODE_STEPS + 1  # bars s .. s+1440 inclusive
        val_count = int(val_prefix[s + n_bars] - val_prefix[s])
        if s < WINDOW:
            bad.append(f"episode {ep['index']}: start {s} < {WINDOW}")
        if val_count != n_bars:
            bad.append(
                f"episode {ep['index']}: only {val_count}/{n_bars} bars in val"
            )
    if bad:
        return _fail("iii", "; ".join(bad[:5]))
    return _pass(
        "iii",
        "all 72 episodes lie wholly inside the val partition over "
        "[s, s+1440] inclusive (UTC (day-1)/days_in_month >= 0.85, "
        "independently re-derived) with start_row >= 256",
    )


# --------------------------------------------------------------------------
# (v) training-partition purity - pre-run sampler draw + optional post-run
# --------------------------------------------------------------------------

def _verify_starts(starts: np.ndarray, val_prefix: np.ndarray) -> list[str]:
    bad = []
    n_bars = EPISODE_STEPS + 1
    if (starts < WINDOW).any():
        bad.append(f"{int((starts < WINDOW).sum())} starts < {WINDOW}")
    counts = val_prefix[starts + n_bars] - val_prefix[starts]
    if (counts != 0).any():
        n_dirty = int((counts != 0).sum())
        worst = int(starts[np.argmax(counts)])
        bad.append(f"{n_dirty} starts touch the val partition (e.g. row {worst})")
    return bad


def check_v(
    market: MarketData, seeds: list[int], n_envs: int, val_prefix: np.ndarray
) -> bool:
    per_env = math.ceil(MIN_SAMPLER_RESETS / n_envs)
    for seed in seeds:
        envs = make_training_envs(market, n_envs, seed)
        starts = np.empty(per_env * n_envs, dtype=np.int64)
        k = 0
        for env in envs:
            for _ in range(per_env):
                env.reset()
                starts[k] = env._t0
                k += 1
        bad = _verify_starts(starts, val_prefix)
        if bad:
            return _fail("v", f"seed {seed} pre-run sampler: " + "; ".join(bad))
        print(
            f"  (v) seed {seed}: {k} resets across {n_envs} envs, "
            "all starts train-pure"
        )

    # post-run half: verify realized start logs if any exist yet
    logs = sorted(TRAIN_STARTS_DIR.glob("train_starts_seed*.json"))
    if not logs:
        print("SKIP (v-post) no train_starts files yet")
    else:
        for path in logs:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                logged = payload.get("starts", payload.get("start_rows"))
            else:
                logged = payload
            if logged is None or not isinstance(logged, list):
                return _fail("v", f"{path.name}: unrecognized start-log schema")
            arr = np.asarray(logged, dtype=np.int64)
            bad = _verify_starts(arr, val_prefix)
            if bad:
                return _fail("v", f"{path.name} post-run: " + "; ".join(bad))
            print(f"  (v-post) {path.name}: {len(arr)} logged starts train-pure")

    total = per_env * n_envs
    return _pass(
        "v",
        f"pre-run sampler purity: {total} resets per seed (seeds "
        f"{seeds}) through make_training_envs, every start s >= 256 with "
        "zero val bars in [s, s+1440] (independent rule re-derivation); "
        "post-run start-log verification implemented "
        f"({len(logs)} train_starts file(s) found)",
    )


# --------------------------------------------------------------------------
# (vi) data-source identity
# --------------------------------------------------------------------------

def check_vi(artifact: dict, market: MarketData) -> bool:
    assert_snapshot_binding()  # raises if DREAMER_DATA redirects the env

    n = len(market.df)
    if n != EXPECTED_ROWS:
        return _fail("vi", f"row count {n} != pinned {EXPECTED_ROWS}")

    min_ts = datetime.fromtimestamp(
        int(market.ts_seconds[0]), tz=timezone.utc
    ).isoformat()
    max_ts = datetime.fromtimestamp(
        int(market.ts_seconds[-1]), tz=timezone.utc
    ).isoformat()
    if min_ts != EXPECTED_MIN_TS or max_ts != EXPECTED_MAX_TS:
        return _fail(
            "vi", f"snapshot span {min_ts} .. {max_ts} != pinned "
            f"{EXPECTED_MIN_TS} .. {EXPECTED_MAX_TS}"
        )

    bad = []
    for ep in artifact["episodes"]:
        expected_sec = iso_to_epoch(ep["start_ts"])
        actual_sec = int(market.ts_seconds[ep["start_row"]])
        if actual_sec != expected_sec:
            bad.append(
                f"episode {ep['index']}: kline ts at row {ep['start_row']} = "
                f"{actual_sec} != artifact {expected_sec}"
            )
    if bad:
        return _fail("vi", "; ".join(bad[:5]))

    return _pass(
        "vi",
        f"data-source identity: snapshot binding OK ({SNAPSHOT_DB.name}); "
        f"{EXPECTED_ROWS} rows; span {EXPECTED_MIN_TS} .. {EXPECTED_MAX_TS}; "
        "kline ts at all 72 episode start rows equals the frozen artifact",
    )


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def append_run_log(seeds: list[int], n_envs: int) -> None:
    stamp = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    note = (
        f"seeds={','.join(str(s) for s in seeds)} n_envs={n_envs}; "
        "iv,vii deferred to eval harness"
    )
    entry = (
        f"| {stamp} | preconditions | assert_adr007_preconditions | "
        f"pass=i,ii,iii,v-pre,vi | {note} |\n"
    )
    existing = RUN_LOG.read_bytes() if RUN_LOG.exists() else b""
    prefix = "" if (not existing or existing.endswith(b"\n")) else "\n"
    with RUN_LOG.open("a", encoding="utf-8", newline="\n") as f:
        f.write(prefix + entry)
    print(f"run log appended: {RUN_LOG.relative_to(PROJECT_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ADR-007 (G) pre-training precondition assertions (Track B)"
    )
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 123])
    parser.add_argument(
        "--episodes",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "adr007" / "eval_episodes.json"),
    )
    args = parser.parse_args()

    def rel(p: Path) -> str:
        # Project-relative display path: the absolute prefix may contain
        # non-ASCII characters (user dir), which the console rule forbids.
        try:
            return str(p.relative_to(PROJECT_ROOT))
        except ValueError:
            return p.name

    episodes_path = Path(args.episodes)
    if not episodes_path.is_absolute():
        episodes_path = Path.cwd() / episodes_path
    if not episodes_path.exists():
        print(f"FAIL (i) frozen episode artifact missing: {rel(episodes_path)}")
        return 1
    sha_path = episodes_path.with_suffix(".sha256")

    artifact_bytes = episodes_path.read_bytes()
    artifact = json.loads(artifact_bytes.decode("ascii"))

    print("ADR-007 (G) pre-training preconditions - Track B")
    print(f"  artifact: {rel(episodes_path)}")
    print(f"  snapshot: {rel(SNAPSHOT_DB)}")
    print(f"  seeds={args.seeds}  n_envs={args.n_envs}")
    print("loading market data (snapshot read + feature computation)...")
    market = load_market_data()
    val_prefix = independent_val_prefix(market)

    results: dict[str, bool] = {}
    checks = [
        ("i", lambda: check_i(artifact, artifact_bytes, sha_path, market)),
        ("ii", lambda: check_ii(artifact, market)),
        ("iii", lambda: check_iii(artifact, val_prefix)),
        ("v", lambda: check_v(market, args.seeds, args.n_envs, val_prefix)),
        ("vi", lambda: check_vi(artifact, market)),
    ]
    for tag, fn in checks:
        try:
            results[tag] = fn()
        except Exception as exc:  # a crashed check is a FAIL, not a crash
            results[tag] = _fail(tag, f"exception: {type(exc).__name__}: {exc}")

    print(
        "NOTE (iv) flat/B&H integrity preconditions and (vii) rollout "
        "determinism are deferred to the eval harness "
        "(scripts/eval_baseline_gates.py, Track C)."
    )

    if all(results.values()):
        append_run_log(args.seeds, args.n_envs)
        print("ALL PRECONDITIONS PASS (i, ii, iii, v-pre, vi)")
        return 0
    failed = [t for t, ok in results.items() if not ok]
    print(f"PRECONDITION FAILURES: {failed} - full run is BLOCKED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
