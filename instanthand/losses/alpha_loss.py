# Copyright (c) 2026 The InstantHand Authors.
# SPDX-License-Identifier: Apache-2.0
"""Soft Alpha Band (SAB) regularizer on the rendered alpha."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["AlphaLossOutput", "SoftBandAlphaLoss"]


@dataclass
class AlphaLossOutput:
    alpha_out: torch.Tensor        # leakage term L_out
    alpha_in: torch.Tensor         # under-fill term L_in
    outer_band_mean: torch.Tensor  # the rest are for logging
    core_mean: torch.Tensor
    excess_out_mean: torch.Tensor
    lack_in_mean: torch.Tensor
    pred_area_mean: torch.Tensor
    gt_area_mean: torch.Tensor
    alpha_mult: torch.Tensor       # schedule value mu(t)

    def as_dict(self) -> Dict[str, torch.Tensor]:
        # the tensors themselves (dataclasses.asdict would deep-copy them)
        return {f.name: getattr(self, f.name) for f in fields(self)}


class SoftBandAlphaLoss(nn.Module):
    """Soft Alpha Band regularizer.

    From the GT silhouette M and the rendered alpha A (both in [0,1]):
        outer band  B_out = dilate(M, r_out) - M     pixels just outside the hand
        core        B_in  = erode(M, r_core)         hand interior
        L_out = sum h_out(relu(A - tau_out) * B_out) / (sum B_out + 1e-6)   leakage above tau_out
        L_in  = sum h_in (relu(tau_in - A) * B_in)   / (sum B_in  + 1e-6)   under-fill below tau_in
        loss  = mu(t) * (lambda_alpha_out * L_out + lambda_alpha_in * L_in)

    h is the Huber penalty against zero, quadratic below beta and linear above
    (beta_out / beta_in). Dilation and erosion use (2r+1) x (2r+1) max pooling.

    mu(t) is 0 up to ramp_start and 1 from ramp_end on. In between it follows the
    smoothstep s^2 (3 - 2s), with s = (t - ramp_start) / (ramp_end - ramp_start).

    pred_alpha, gt_sil: [B,V,1,H,W] or [B,V,H,W]
    """

    def __init__(
        self,
        lambda_alpha_out: float = 0.25,
        lambda_alpha_in: float = 0.3,
        r_out: int = 2,
        r_core: int = 2,
        tau_out: float = 0.03,
        tau_in: float = 0.60,
        beta_out: float = 0.03,
        beta_in: float = 0.05,
        ramp_start: int = 5000,
        ramp_end: int = 12000,
    ):
        super().__init__()
        self.lambda_alpha_out = float(lambda_alpha_out)
        self.lambda_alpha_in = float(lambda_alpha_in)
        self.r_out = int(r_out)
        self.r_core = int(r_core)
        self.tau_out = float(tau_out)
        self.tau_in = float(tau_in)
        self.beta_out = float(beta_out)
        self.beta_in = float(beta_in)
        self.ramp_start = int(ramp_start)
        self.ramp_end = int(ramp_end)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_bvchw_mask(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            return x
        if x.ndim == 4:
            return x.unsqueeze(2)
        raise ValueError(f"Unexpected mask ndim: {x.ndim}")

    @staticmethod
    def _morphology_mask(mask: torch.Tensor, mode: str, k: int) -> torch.Tensor:
        """Grayscale dilation / erosion with a (2k+1) x (2k+1) square, via max pooling."""
        if k <= 0:
            return mask.clamp(0, 1)

        orig_ndim = mask.ndim
        if orig_ndim == 5:
            b, v, c, h, w = mask.shape
            x = mask.reshape(b * v, c, h, w)
        elif orig_ndim == 4:
            x = mask
        else:
            raise ValueError(f"Unexpected mask ndim: {mask.ndim}")

        kernel = 2 * k + 1
        if mode == "dilate":
            y = F.max_pool2d(x, kernel_size=kernel, stride=1, padding=k)
        elif mode == "erode":
            y = -F.max_pool2d(-x, kernel_size=kernel, stride=1, padding=k)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if orig_ndim == 5:
            y = y.reshape(b, v, c, h, w)
        return y.clamp(0, 1)

    @staticmethod
    def _huber_zero(x: torch.Tensor, beta: float = 0.03) -> torch.Tensor:
        """Huber penalty of x against 0: 0.5 x^2 / beta if |x| < beta, else |x| - 0.5 beta."""
        ax = x.abs()
        beta_t = torch.as_tensor(beta, dtype=x.dtype, device=x.device)
        return torch.where(
            ax < beta_t,
            0.5 * ax.pow(2) / beta_t.clamp_min(1e-12),
            ax - 0.5 * beta_t,
        )

    def _make_outer_band_and_core(self, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """GT mask [B,V,1,H,W] -> (outer band, core), both [B,V,1,H,W]."""
        if mask.ndim != 5:
            raise ValueError(f"Expected [B,V,1,H,W], got {mask.shape}")
        mask = mask.float().clamp(0.0, 1.0)
        outer = (self._morphology_mask(mask, mode="dilate", k=self.r_out) - mask).clamp(0.0, 1.0)
        core = self._morphology_mask(mask, mode="erode", k=self.r_core).clamp(0.0, 1.0)
        return outer, core

    def get_alpha_multiplier(self, step: int, device=None, dtype=None) -> torch.Tensor:
        """Schedule mu(t): 0 before ramp_start, smoothstep up to 1 at ramp_end."""
        if self.ramp_end <= self.ramp_start:
            val = 1.0 if step >= self.ramp_start else 0.0
        elif step <= self.ramp_start:
            val = 0.0
        elif step >= self.ramp_end:
            val = 1.0
        else:
            t = float(step - self.ramp_start) / float(self.ramp_end - self.ramp_start)
            val = t * t * (3.0 - 2.0 * t)
        return torch.tensor(val, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # loss
    # ------------------------------------------------------------------
    def compute_stats(self, pred_alpha: torch.Tensor, gt_sil: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Unweighted L_out, L_in and logging statistics."""
        pred_alpha = self._to_bvchw_mask(pred_alpha).clamp(0.0, 1.0)
        gt_sil = self._to_bvchw_mask(gt_sil).clamp(0.0, 1.0)

        outer_band, core = self._make_outer_band_and_core(gt_sil)

        excess_out = F.relu(pred_alpha - self.tau_out) * outer_band
        lack_in = F.relu(self.tau_in - pred_alpha) * core

        alpha_out = self._huber_zero(excess_out, beta=self.beta_out).sum() / (outer_band.sum() + 1e-6)
        alpha_in = self._huber_zero(lack_in, beta=self.beta_in).sum() / (core.sum() + 1e-6)

        pred_area = pred_alpha.mean(dim=(-1, -2, -3))
        gt_area = gt_sil.mean(dim=(-1, -2, -3))

        return {
            "alpha_out": alpha_out,
            "alpha_in": alpha_in,
            "outer_band_mean": outer_band.mean(),
            "core_mean": core.mean(),
            "excess_out_mean": excess_out.mean(),
            "lack_in_mean": lack_in.mean(),
            "pred_area_mean": pred_area.mean(),
            "gt_area_mean": gt_area.mean(),
        }

    def forward(self, pred_alpha: torch.Tensor, gt_sil: torch.Tensor, global_step: int) -> AlphaLossOutput:
        """Unweighted terms, logging statistics and mu(global_step)."""
        stats = self.compute_stats(pred_alpha, gt_sil)
        alpha_mult = self.get_alpha_multiplier(global_step, device=pred_alpha.device, dtype=pred_alpha.dtype)
        return AlphaLossOutput(**stats, alpha_mult=alpha_mult)

    def weighted_loss(
        self,
        pred_alpha: torch.Tensor,
        gt_sil: torch.Tensor,
        global_step: int,
    ) -> Tuple[torch.Tensor, AlphaLossOutput]:
        """(mu(t) * (lambda_alpha_out * L_out + lambda_alpha_in * L_in), AlphaLossOutput)"""
        out = self.forward(pred_alpha, gt_sil, global_step)
        loss = out.alpha_mult * (
            self.lambda_alpha_out * out.alpha_out +
            self.lambda_alpha_in * out.alpha_in
        )
        return loss, out