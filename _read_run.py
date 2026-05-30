import wandb
api = wandb.Api()
run = api.run("sensoymertefe-ted-niversitesi/crypto-dreamer/runs/1rq8d8u5")
h = run.history(keys=["train/kl_unclipped","train/kl_clip_excess","val/loss_forward_dist","val/loss_reward","train/loss_forward_1","train/loss_forward_5","train/loss_forward_15","train/loss_forward_30","_step"])
print(h.tail(20).to_string())
