import duckdb
con = duckdb.connect("data/market.duckdb", read_only=True)

print("\n=== 1-bar log-return quantiles (BTCUSDT 1m) ===")
print(con.execute("""
    WITH r AS (
        SELECT ln(close / lag(close) OVER (ORDER BY ts)) AS lr
        FROM klines WHERE symbol='BTCUSDT' AND interval='1m'
    )
    SELECT
        quantile_cont(lr, 0.001) AS q001,
        quantile_cont(lr, 0.01)  AS q01,
        quantile_cont(lr, 0.05)  AS q05,
        quantile_cont(lr, 0.5)   AS q50,
        quantile_cont(lr, 0.95)  AS q95,
        quantile_cont(lr, 0.99)  AS q99,
        quantile_cont(lr, 0.999) AS q999,
        stddev(lr)               AS std
    FROM r WHERE lr IS NOT NULL
""").fetchone())

for h in [5, 15, 30]:
    print(f"\n=== {h}-bar log-return quantiles ===")
    print(con.execute(f"""
        WITH r AS (
            SELECT ln(close / lag(close, {h}) OVER (ORDER BY ts)) AS lr
            FROM klines WHERE symbol='BTCUSDT' AND interval='1m'
        )
        SELECT
            quantile_cont(lr, 0.001) AS q001,
            quantile_cont(lr, 0.01)  AS q01,
            quantile_cont(lr, 0.05)  AS q05,
            quantile_cont(lr, 0.5)   AS q50,
            quantile_cont(lr, 0.95)  AS q95,
            quantile_cont(lr, 0.99)  AS q99,
            quantile_cont(lr, 0.999) AS q999,
            stddev(lr)               AS std
        FROM r WHERE lr IS NOT NULL
    """).fetchone())

print(f"\n=== row count ===")
print(con.execute("SELECT count(*) FROM klines WHERE symbol='BTCUSDT' AND interval='1m'").fetchone())
