import wandb
api = wandb.Api()
run = api.run("sensoymertefe-ted-niversitesi/crypto-dreamer/runs/1rq8d8u5")
print("state:", run.state, "| summary _step:", run.summary.get("_step"))
rows = []
for r in run.scan_history(keys=["_step","train/kl_unclipped_step","train/kl_clip_excess_step"]):
    if r.get("train/kl_unclipped_step") is not None:
        rows.append((r["_step"], r["train/kl_unclipped_step"], r.get("train/kl_clip_excess_step")))
print("rows with kl:", len(rows), "| step range:", rows[0][0], "->", rows[-1][0])
import bisect
steps = [x[0] for x in rows]
for s in [600, 2000, 6000, 10000, 15000, 20000, 25000, 29000]:
    i = min(bisect.bisect_left(steps, s), len(rows)-1)
    print(f"~{s:6d}: step {rows[i][0]:6d}  kl {rows[i][1]:.3f}  excess {rows[i][2]:.3f}")
mx = max(rows, key=lambda x: x[1])
print(f"\nMAX kl_unclipped over FULL run: {mx[1]:.3f} at step {mx[0]}")
print(f"FINAL: step {rows[-1][0]}  kl {rows[-1][1]:.3f}  excess {rows[-1][2]:.3f}")
