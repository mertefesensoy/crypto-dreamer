"""Learned 32-dim embedding of the 5 discrete actions.

Subclasses nn.Embedding so it composes naturally with optimizers,
state_dict, and bf16 mixed precision.
"""
from __future__ import annotations

import torch.nn as nn


class ActionEmbed(nn.Embedding):
    def __init__(self, n_actions: int = 5, dim: int = 32):
        super().__init__(n_actions, dim)
