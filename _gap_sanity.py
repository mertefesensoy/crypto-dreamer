import duckdb
con = duckdb.connect("data/market.duckdb", read_only=True)
print(con.execute("""
    WITH d AS (
        SELECT ts, lag(ts) OVER (ORDER BY ts) AS prev_ts
        FROM klines WHERE symbol='BTCUSDT' AND interval='1m'
    )
    SELECT count(*) AS gap_count
    FROM d
    WHERE prev_ts IS NOT NULL
      AND date_diff('second', prev_ts, ts) <> 60
""").fetchone())
