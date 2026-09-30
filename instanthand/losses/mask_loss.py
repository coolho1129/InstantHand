# Copyright (c) 2026 The InstantHand Authors.
# SPDX-License-Identifier: Apache-2.0
"""Silhouette loss between the rendered alpha and the ground-truth mask."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = ["MaskLoss"]


class MaskLoss(nn.Module):
    """L1 or binary cross-entropy between the rendered alpha and the GT mask.

    The loss is averaged over the pixels of each image, then over views, then over the batch.

    pred: [B,V,1,H,W] or [B,V,H,W], alpha in [0,1] (probabilities, not logits)
    gt:   [B,V,1,H,W] or [B,V,H,W], mask in [0,1]
    """

    def __init__(self, option: str = "l1", eps: float = 1e-7):
        super().__init__()
        assert option in ("l1", "bce"), f"Unknown mask loss option: {option}"
        self.option = option
        self.eps = eps

    @torch.compile
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if pred.ndim == 4:
            pred = pred.unsqueeze(2)
        if gt.ndim == 4:
            gt = gt.unsqueeze(2)
        assert pred.ndim == 5 and gt.ndim == 5, f"pred {pred.shape}, gt {gt.shape}"
        B, V, C, H, W = pred.shape
        assert C == 1, f"mask channel must be 1, got C={C}"

        pred = pred.clamp(self.eps, 1.0 - self.eps)
        gt = gt.clamp(0.0, 1.0)
        pred_ = rearrange(pred, "b v c h w -> (b v) c h w")
        gt_ = rearrange(gt, "b v c h w -> (b v) c h w")

        if self.option == "l1":
            per_pix = (pred_ - gt_).abs()
        else:
            per_pix = F.binary_cross_entropy(pred_, gt_, reduction="none")

        per_img = per_pix.mean(dim=[1, 2, 3])        # [B*V]
        per_b = per_img.reshape(B, V).mean(dim=1)    # [B]
        return per_b.mean()