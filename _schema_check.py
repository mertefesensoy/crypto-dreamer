import duckdb
con = duckdb.connect("data/market.duckdb", read_only=True)
tables = con.execute("SHOW TABLES").fetchall()
print("Tables:", tables)
for t in tables:
    cols = con.execute(f"DESCRIBE {t[0]}").fetchall()
    print(f"\n{t[0]} columns:")
    for c in cols:
        print(f"  {c[0]}: {c[1]}")
