import torch
from omegaconf import OmegaConf
from models.world_model import WorldModel
from training.datamodule import SpotBTCDataModule

cfg = OmegaConf.load("configs/world_model.yaml")
ckpt = torch.load("checkpoints/world_model_diagnostic_step=30000-v1.ckpt", map_location="cuda")
sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
sd = {(k[6:] if k.startswith("model.") else k): v for k,v in sd.items()}

m = WorldModel(
    n_vars=cfg.model.n_vars, seq_len=cfg.model.seq_len, d_model=cfg.model.d_model,
    encoder_layers=cfg.model.encoder_layers, encoder_heads=cfg.model.encoder_heads,
    encoder_ff=cfg.model.encoder_ff, encoder_dropout=cfg.model.encoder_dropout,
    hidden_dim=cfg.model.hidden_dim, n_latents=cfg.model.n_latents, n_classes=cfg.model.n_classes,
    rssm_mlp_hidden=cfg.model.rssm_mlp_hidden, unimix=cfg.model.unimix,
    n_actions=cfg.model.n_actions, action_emb_dim=cfg.model.action_emb_dim,
    head_hidden=cfg.model.head_hidden, reward_n_bins=cfg.model.reward_n_bins,
    reward_low=cfg.model.reward_low, reward_high=cfg.model.reward_high,
    coef_dyn=cfg.model.coef_dyn, coef_rep=cfg.model.coef_rep, free_bits=cfg.model.free_bits,
    lr=cfg.model.lr, weight_decay=cfg.model.weight_decay, warmup_steps=cfg.model.warmup_steps,
    burn_in=cfg.model.burn_in,
    forward_horizons=list(cfg.model.forward_horizons), forward_bins=cfg.model.forward_bins,
    forward_ranges=list(cfg.model.forward_ranges),
).cuda()
m.load_state_dict(sd, strict=False)
m.eval()

d = cfg.data
dm = SpotBTCDataModule(
    klines_db=d.klines_db, steps_db=d.steps_db, symbol=d.symbol, interval=d.interval,
    T=d.T, batch_size=d.batch_size, num_workers=0, pin_memory=False,
    persistent_workers=False, max_episodes=d.get("max_episodes", None),
)
dm.setup("fit")
vl = dm.val_dataloader()

b0 = next(iter(vl))
b0 = {k:(v.cuda() if torch.is_tensor(v) else v) for k,v in b0.items()}
with torch.no_grad():
    probe = m._step(b0, stage="val")
print("type(_step output):", type(probe))
if isinstance(probe, tuple):
    print("tuple len:", len(probe), "| elem types:", [type(x).__name__ for x in probe])

agg, n = {}, 0
with torch.no_grad():
    for i, batch in enumerate(vl):
        if i >= 30: break
        batch = {k:(v.cuda() if torch.is_tensor(v) else v) for k,v in batch.items()}
        out = m._step(batch, stage="val")
        logs = None
        if isinstance(out, tuple):
            for el in out:
                if isinstance(el, dict): logs = el
        elif isinstance(out, dict):
            logs = out
        if logs is None:
            print("no dict found in _step return:", out); break
        for k,v in logs.items():
            if torch.is_tensor(v): v = v.item()
            agg[k] = agg.get(k,0)+v
        n += 1
print(f"\n=== val metrics over {n} batches · 30k checkpoint ===")
for k in sorted(agg): print(f"{k:34s} {agg[k]/n:.4f}")
