import wandb
api = wandb.Api()
run = api.run("sensoymertefe-ted-niversitesi/crypto-dreamer/runs/1rq8d8u5")
h = run.history(samples=5000)
kcols = [c for c in h.columns if "kl" in c.lower() or "unclip" in c.lower()]
print("KL columns:", kcols)
sub = h[["_step"] + kcols].dropna(how="all", subset=kcols)
print(sub.iloc[::max(1, len(sub)//25)].to_string())
