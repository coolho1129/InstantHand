# Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors.
# Copyright (c) 2026 The InstantHand Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from TransformerDecoder (lam/models/transformer.py) and SD3JointTransformerBlock
# (lam/models/transformer_dit.py) in LAM (https://github.com/aigc3d/LAM), licensed under the
# Apache License 2.0. The block follows JointTransformerBlock of diffusers
# (https://github.com/huggingface/diffusers, Apache License 2.0) with the timestep (AdaLN)
# modulation replaced by plain LayerNorm. Modified by the InstantHand authors: only the
# "sd3_cond" decoder used by InstantHand is kept; the other block types, the dual attention
# and the chunked feed-forward are removed.
"""Transformer that updates the MANO anchor tokens with image features (MMDiT joint attention).

    x    [N, L, inner_dim]  anchor (query) tokens
    cond [N, Lc, cond_dim]  image-encoder tokens
      -> [N, L, inner_dim]
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import JointAttnProcessor2_0

__all__ = ["SD3JointTransformerBlock", "TransformerDecoder"]


class SD3JointTransformerBlock(nn.Module):
    """MMDiT block (Stable Diffusion 3, https://arxiv.org/abs/2403.03206) without timestep modulation.

    Anchor tokens and image tokens attend jointly (one attention over the concatenated
    sequence, separate projections per stream); each stream then has its own feed-forward.
    context_pre_only: last block; the image stream is not updated and None is returned for it.
    """

    def __init__(self, dim: int, num_heads: int, eps: float, context_pre_only: bool = False,
                 qk_norm: Optional[str] = None):
        super().__init__()
        head_dim = dim // num_heads
        if head_dim * num_heads != dim:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.context_pre_only = context_pre_only

        # NOTE: submodule names and creation order are fixed (checkpoint keys, initialization).
        self.norm1 = nn.LayerNorm(dim)
        self.norm1_context = nn.LayerNorm(dim)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=head_dim,
            heads=num_heads,
            out_dim=dim,
            context_pre_only=context_pre_only,
            bias=True,
            processor=JointAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=eps,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        if context_pre_only:
            self.norm2_context = None
            self.ff_context = None
        else:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
            self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor,
                temb: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """hidden_states [N,L,D], encoder_hidden_states [N,Lc,D]; temb is accepted and ignored."""
        attn_output, context_attn_output = self.attn(
            hidden_states=self.norm1(hidden_states),
            encoder_hidden_states=self.norm1_context(encoder_hidden_states),
        )
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))

        if self.context_pre_only:
            return hidden_states, None
        encoder_hidden_states = encoder_hidden_states + context_attn_output
        encoder_hidden_states = encoder_hidden_states + self.ff_context(self.norm2_context(encoder_hidden_states))
        return hidden_states, encoder_hidden_states


class TransformerDecoder(nn.Module):
    """Stack of SD3JointTransformerBlock ("sd3_cond"): the image tokens are projected to
    inner_dim, both streams pass through `num_layers` joint blocks, and the anchor tokens get a
    final LayerNorm.

    block_type and mod_dim are kept for config compatibility; only "sd3_cond" without
    modulation is supported.
    """

    def __init__(
        self,
        block_type: str,
        num_layers: int,
        num_heads: int,
        inner_dim: int,
        cond_dim: Optional[int] = None,
        mod_dim: Optional[int] = None,
        gradient_checkpointing: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        if block_type != "sd3_cond":
            raise ValueError(f"Unsupported block_type {block_type!r}; InstantHand uses 'sd3_cond'")
        if cond_dim is None or mod_dim is not None:
            raise ValueError("'sd3_cond' needs cond_dim and takes no modulation (mod_dim=None)")
        self.block_type = block_type
        self.gradient_checkpointing = gradient_checkpointing

        # NOTE: submodule names and creation order are fixed (checkpoint keys, initialization).
        self.layers = nn.ModuleList([
            SD3JointTransformerBlock(inner_dim, num_heads=num_heads, eps=eps,
                                     context_pre_only=(i == num_layers - 1), qk_norm="rms_norm")
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        self.linear_cond_proj = nn.Linear(cond_dim, inner_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mod: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x [N,L,inner_dim], cond [N,Lc,cond_dim] -> [N,L,inner_dim]"""
        if mod is not None:
            raise ValueError("'sd3_cond' takes no modulation input")
        cond = self.linear_cond_proj(cond)
        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                x, cond = torch.utils.checkpoint.checkpoint(layer, x, cond, use_reentrant=False)
            else:
                x, cond = layer(hidden_states=x, encoder_hidden_states=cond)
        return self.norm(x)