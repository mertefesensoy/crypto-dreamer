import wandb
api = wandb.Api()
run = api.run("sensoymertefe-ted-niversitesi/crypto-dreamer/runs/1rq8d8u5")
h = run.history(keys=["train/kl_unclipped_step","train/kl_clip_excess_step"], samples=30000)
h = h.dropna(subset=["train/kl_unclipped_step"])
print("rows:", len(h), "max step:", int(h["_step"].max()))
for s in [600, 2000, 6000, 10000, 15000, 20000, 25000, 29000]:
    row = h.iloc[(h["_step"]-s).abs().argmin()]
    print(f"step {int(row['_step']):6d}  kl_unclipped {row['train/kl_unclipped_step']:.3f}  excess {row['train/kl_clip_excess_step']:.3f}")
print("\nmax kl_unclipped over whole run:", round(h["train/kl_unclipped_step"].max(),3), "at step", int(h.loc[h['train/kl_unclipped_step'].idxmax(),'_step']))
