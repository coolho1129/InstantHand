# Copyright (c) 2026 The InstantHand Authors.
# SPDX-License-Identifier: Apache-2.0
"""Building blocks of the appearance head (pre-norm residual MLP and cross-attention)."""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ResidualMLP", "CrossAttentionBlock"]


class ResidualMLP(nn.Module):
    """x + MLP(LayerNorm(x)), hidden width dim * expansion, GELU. [..., dim] -> [..., dim]"""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = dim * expansion
        # NOTE: submodule names and creation order are fixed (checkpoint keys, initialization).
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention from `query` to `memory`, followed by a ResidualMLP.

    query [B,Nq,dim], memory [B,Nm,dim] -> [B,Nq,dim]
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ffn = ResidualMLP(dim=dim, expansion=4, dropout=dropout)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        kv = self.kv_norm(memory)
        attn_out, _ = self.attn(self.q_norm(query), kv, kv, need_weights=False)
        return self.ffn(query + attn_out)