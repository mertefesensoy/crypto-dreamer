import torch, inspect
from omegaconf import OmegaConf
from models.world_model import WorldModel
from training.datamodule import SpotBTCDataModule

print("=== _step source ===")
print(inspect.getsource(WorldModel._step))
print("=== RSSM.forward / step signatures ===")
from models.rssm import RSSM
for name in ["forward","step","observe","categorical_kl","free_bits_kl"]:
    f = getattr(RSSM, name, None)
    if f: print(name, inspect.signature(f))
print("=== ForwardDistributionHead methods ===")
from models.heads import ForwardDistributionHead
for name in ["forward","loss","per_horizon_loss","two_hot_encode"]:
    f = getattr(ForwardDistributionHead, name, None)
    if f: print(name, inspect.signature(f))
