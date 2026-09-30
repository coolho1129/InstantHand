# Copyright (c) 2026 The InstantHand Authors.
# SPDX-License-Identifier: Apache-2.0
#
# MaskAwareLPIPSLoss is adapted from LPIPSLoss in LAM (lam/losses/perceptual.py, originally
# from OpenLRM), Copyright (c) 2023-2024, Zexin He, licensed under the Apache License 2.0.
# Modified by the InstantHand authors: mask-aware background compositing and view averaging.
"""Mask-aware image losses.

Tensor layout used throughout (N = batch, M = views):
    pred_rgb, gt_rgb : [N,M,3,H,W] in [0,1]
    masks            : [N,M,1,H,W] or [N,M,H,W] in [0,1]
    bg               : [3], [N,3], [N,M,3] or [N,M,3,H,W] in [0,1]
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

__all__ = ["MaskAwarePixelLoss", "MaskAwareLPIPSLoss", "VisibleMaskLoss"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _to_5d_mask(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """[N,M,H,W] or [N,M,1,H,W] -> [N,M,1,H,W], on ref's device and dtype, clamped to [0,1]."""
    if mask is None:
        return None
    if mask.ndim == 4:
        mask = mask.unsqueeze(2)
    assert mask.ndim == 5, f"mask must be 4D/5D, got {mask.shape}"
    return mask.to(device=ref.device, dtype=ref.dtype).clamp(0.0, 1.0)


def _expand_bg(bg, N, M, H, W, device, dtype) -> torch.Tensor | None:
    """Background color [3], [N,3], [N,M,3] or [N,M,3,H,W] -> [N,M,3,H,W]."""
    if bg is None:
        return None

    bg = torch.as_tensor(bg, device=device, dtype=dtype)

    if bg.ndim == 1 and bg.numel() == 3:
        return bg.view(1, 1, 3, 1, 1).expand(N, M, 3, H, W)

    if bg.ndim == 2 and bg.shape[-1] == 3:  # [N,3]
        assert bg.shape[0] == N, f"bg [N,3] expected N={N}, got {bg.shape}"
        return bg.view(N, 1, 3, 1, 1).expand(N, M, 3, H, W)

    if bg.ndim == 3 and bg.shape[-1] == 3:  # [N,M,3]
        assert bg.shape[0] == N and bg.shape[1] == M, f"bg [N,M,3] expected {(N, M)}, got {bg.shape}"
        return bg.view(N, M, 3, 1, 1).expand(N, M, 3, H, W)

    if bg.ndim == 5:  # [N,M,3,H,W]
        assert bg.shape[:3] == (N, M, 3) and bg.shape[3:] == (H, W), f"bg [N,M,3,H,W] mismatch: {bg.shape}"
        return bg

    raise ValueError(f"Unsupported bg shape: {tuple(bg.shape)}")


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
class MaskAwarePixelLoss(nn.Module):
    """L1 or MSE restricted to the mask.

    Default: errors inside the mask, summed per image and divided by (mask area x C)
    (or averaged over all pixels if normalize_by_mask=False), then averaged over views
    and batch.

    robust_percentile < 1: per batch item, only the smallest `robust_percentile` fraction
    of the in-mask errors is averaged (outlier rejection).

    x, y: [N,M,C,H,W];  mask: [N,M,1,H,W] or [N,M,H,W] (1 = foreground)
    """

    def __init__(
        self,
        option: str = "l1",
        eps: float = 1e-6,
        normalize_by_mask: bool = True,
        robust_percentile: float = 1.0,
    ):
        super().__init__()
        assert option in ("l1", "mse")
        self.option = option
        self.eps = float(eps)
        self.normalize_by_mask = bool(normalize_by_mask)
        self.robust_percentile = float(robust_percentile)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        assert x.ndim == 5 and y.ndim == 5, f"x {x.shape}, y {y.shape}"
        N, M, C, H, W = x.shape
        assert y.shape == x.shape, f"shape mismatch x {x.shape} vs y {y.shape}"

        if mask is None:
            diff = (x - y).abs() if self.option == "l1" else (x - y).pow(2)
            if self.robust_percentile >= 1.0 - 1e-6:
                return diff.mean()
            # robust: mean of the smallest k errors over the whole batch
            flat = diff.view(-1)
            k = int(flat.numel() * self.robust_percentile)
            if k == 0:
                return diff.mean()
            val, _ = torch.topk(flat, k, largest=False)
            return val.mean()

        mask = _to_5d_mask(mask, x)  # [N,M,1,H,W]
        per = (x - y).abs() if self.option == "l1" else (x - y).pow(2)
        per = per * mask  # keep in-mask errors only (broadcast over channels)

        if self.robust_percentile < 1.0 - 1e-6:
            # robust: per batch item, mean of the smallest k in-mask errors
            per_flat = rearrange(per, "n m c h w -> n (m c h w)")
            mask_flat = rearrange(mask, "n m 1 h w -> n (m 1 h w)")
            mask_flat = mask_flat.repeat_interleave(C, dim=1)

            loss_list = []
            for b in range(N):
                valid_errs = per_flat[b][mask_flat[b] > 0.5]
                if valid_errs.numel() == 0:
                    loss_list.append(torch.tensor(0.0, device=x.device))
                    continue
                k = int(valid_errs.numel() * self.robust_percentile)
                if k < 1:
                    loss_list.append(valid_errs.mean())
                else:
                    val, _ = torch.topk(valid_errs, k, largest=False)
                    loss_list.append(val.mean())
            return torch.stack(loss_list).mean()

        per = rearrange(per, "n m c h w -> (n m) c h w")
        m = rearrange(mask, "n m c h w -> (n m) c h w")  # [(N*M),1,H,W]

        if self.normalize_by_mask:
            denom = (m.sum(dim=[1, 2, 3]) * C).clamp_min(self.eps)  # masked pixels x C
            num = per.sum(dim=[1, 2, 3])
            image_loss = num / denom
        else:
            image_loss = per.mean(dim=[1, 2, 3])

        return image_loss.reshape(N, M).mean(dim=1).mean()


class MaskAwareLPIPSLoss(nn.Module):
    """LPIPS on images composited onto a background outside the mask:
        x' = x * mask + bg * (1 - mask),   y' = y * mask + bg * (1 - mask)
    so pixels outside the mask contribute (almost) nothing. Without `bg`, gray (0.5) is used.

    VGG is used in training (is_training=True) and AlexNet otherwise. LPIPS networks are
    frozen and cached per instance. `prefech` loads both networks at construction.

    x, y: [N,M,3,H,W] in [0,1];  mask: [N,M,1,H,W] or [N,M,H,W];  bg: see module docstring
    """

    def __init__(self, device, prefech: bool = False):
        super().__init__()
        self.device = device
        self.cached_models = {}
        if prefech:
            self.prefetch_models()

    def _get_model(self, model_name: str):
        if model_name not in self.cached_models:
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                import lpips

                model = lpips.LPIPS(net=model_name, eval_mode=True, verbose=False).to(self.device)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self.cached_models[model_name] = model
        return self.cached_models[model_name]

    def prefetch_models(self):
        for model_name in ("alex", "vgg"):
            self._get_model(model_name)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor | None = None,
        bg=None,
        is_training: bool = True,
    ) -> torch.Tensor:
        assert x.ndim == 5 and y.ndim == 5, f"x {x.shape}, y {y.shape}"
        N, M, C, H, W = x.shape
        assert y.shape == x.shape, f"shape mismatch x {x.shape} vs y {y.shape}"
        assert C == 3, f"LPIPS expects RGB C=3, got C={C}"

        loss_fn = self._get_model("vgg" if is_training else "alex")

        if mask is not None:
            mask = _to_5d_mask(mask, x)
            bg_img = _expand_bg(bg, N, M, H, W, x.device, x.dtype)
            if bg_img is None:
                bg_img = torch.full((N, M, 3, H, W), 0.5, device=x.device, dtype=x.dtype)
            x = x * mask + bg_img * (1.0 - mask)
            y = y * mask + bg_img * (1.0 - mask)

        x = x.reshape(N * M, C, H, W)
        y = y.reshape(N * M, C, H, W)

        d = loss_fn(x, y, normalize=True)  # [N*M,1,1,1]; normalize: [0,1] -> [-1,1]
        d = d.mean(dim=[1, 2, 3])
        return d.reshape(N, M).mean(dim=1).mean()


class VisibleMaskLoss(nn.Module):
    """Pushes the rendered alpha toward 1 on GT-foreground pixels only.

    Pixels where the GT mask is 0 (background, occluded or missing parts) are not
    penalized. The loss is normalized by the GT-foreground area.

    mode: 'l1'       |1 - pred|
          'bce'      -log(pred)          (target 1)
          'smoothl1' SmoothL1(pred, 1)

    pred_mask: [N,M,1,H,W] or [N,M,H,W] in [0,1] (logits if use_logits=True)
    gt_mask:   [N,M,1,H,W] or [N,M,H,W] in [0,1]
    """

    def __init__(self, mode: str = "bce", eps: float = 1e-6, use_logits: bool = False):
        super().__init__()
        assert mode in ("l1", "bce", "smoothl1")
        self.mode = mode
        self.eps = float(eps)
        self.use_logits = bool(use_logits)
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none") if mode == "smoothl1" else None

    def forward(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        assert pred_mask is not None and gt_mask is not None
        if pred_mask.ndim == 4:
            pred_mask = pred_mask.unsqueeze(2)
        if gt_mask.ndim == 4:
            gt_mask = gt_mask.unsqueeze(2)
        assert pred_mask.ndim == 5 and pred_mask.shape[2] == 1, f"pred_mask {pred_mask.shape}"
        assert gt_mask.ndim == 5 and gt_mask.shape[2] == 1, f"gt_mask {gt_mask.shape}"

        # float32 for the log in 'bce'
        pm = pred_mask.to(dtype=torch.float32)
        gm = gt_mask.to(dtype=torch.float32).to(device=pm.device).clamp(0.0, 1.0)
        fg = (gm > 0.5).to(pm.dtype)  # [N,M,1,H,W]

        if self.use_logits:
            pm = torch.sigmoid(pm)
        pm = pm.clamp(self.eps, 1.0 - self.eps)

        if self.mode == "l1":
            per = (1.0 - pm).abs()
        elif self.mode == "bce":
            per = -torch.log(pm)
        else:
            per = self.smooth_l1(pm, torch.ones_like(pm))

        per = per * fg
        denom = fg.sum().clamp_min(self.eps)
        return per.sum() / denom