"""
Ingest BTC/USDT 1-minute klines from Binance into DuckDB.

Uses python-binance for bulk historical pulls. For live streaming we'll add
a WebSocket worker in v2; this module is offline-only.

Schema:
    klines(ts TIMESTAMP, open, high, low, close, volume, quote_volume, trades INT)

Run:
    python -m data.ingest --symbol BTCUSDT --interval 1m --years 2
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
from binance.client import Client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.environ.get("DREAMER_DATA", PROJECT_ROOT / "data" / "market.duckdb"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS klines (
            symbol VARCHAR,
            interval VARCHAR,
            ts TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            quote_volume DOUBLE,
            trades INTEGER,
            PRIMARY KEY (symbol, interval, ts)
        )
        """
    )


def fetch_and_store(symbol: str, interval: str, years: int) -> None:
    client = Client()  # public endpoints only, no keys needed for klines
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * years)

    con = duckdb.connect(str(DB_PATH))
    init_schema(con)

    print(f"Fetching {symbol} {interval} from {start.date()} to {end.date()}")
    # generator yields chunks; binance enforces 1000-row limits internally
    rows = client.get_historical_klines_generator(
        symbol, interval, start.strftime("%d %b %Y"), end.strftime("%d %b %Y")
    )

    batch: list[tuple] = []
    inserted = 0
    for k in rows:
        batch.append(
            (
                symbol,
                interval,
                datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
                float(k[1]),
                float(k[2]),
                float(k[3]),
                float(k[4]),
                float(k[5]),
                float(k[7]),
                int(k[8]),
            )
        )
        if len(batch) >= 5_000:
            con.executemany(
                "INSERT OR IGNORE INTO klines VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
            inserted += len(batch)
            batch.clear()
            print(f"  inserted ~{inserted:,}")
    if batch:
        con.executemany(
            "INSERT OR IGNORE INTO klines VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            batch,
        )
        inserted += len(batch)

    total = con.execute(
        "SELECT COUNT(*) FROM klines WHERE symbol = ? AND interval = ?",
        [symbol, interval],
    ).fetchone()[0]
    print(f"Done. {inserted:,} new rows. {total:,} total for {symbol} {interval}.")
    con.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1m")
    p.add_argument("--years", type=int, default=2)
    args = p.parse_args()
    fetch_and_store(args.symbol, args.interval, args.years)
