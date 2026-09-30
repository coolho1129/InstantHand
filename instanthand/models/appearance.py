# Copyright (c) 2026 The InstantHand Authors.
# SPDX-License-Identifier: Apache-2.0
"""Stage-2 appearance head: reliability-gated transfer of source-image color to the canonical
Gaussians (Stage 2 of the paper).

The renderer poses the Stage-1 Gaussians in the source pose, renders them in the source camera
and calls this head with that render; only the Gaussian colors are changed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .appearance_blocks import CrossAttentionBlock, ResidualMLP
from .encoders.xunet_wrapper import XUNet

__all__ = ["AppearanceHead"]


class AppearanceHead(nn.Module):
    """Reliability-gated appearance transfer (Stage 2).

    Each canonical Gaussian i gets its final color as

        c*_i = clamp(c0_i + g_i * gamma_i * (c_obs_i - c0_i) + Delta_aid, 0, 1)

    c0_i       base color of the Stage-1 avatar
    c_obs_i    color predicted from source-image evidence (observed_head)
    gamma_i    source-observation confidence (compute_render_aware_gs_confidence)
    g_i        learned gate (global_gate_head)
    Delta_aid  a single color shift shared by all Gaussians: the mean over Gaussians of
               delta_i = identity_delta_scale * tanh(identity_delta_head(.))

    Source evidence enters each Gaussian only through the gamma-weighted term, so
    Gaussians that were not observed in the source image keep c0_i plus the shared
    Delta_aid.

    Inputs (dict):
        source_image           [B,3,H,W]    source image in [0,1]
        source_mask            [B,1,H,W]    hand mask of the source image
        source_render_rgb      [B,3,H,W]    Stage-1 avatar rendered in the source view
        posed_base_gs          Gaussians in the source pose; .xyz [B,N,3] or [N,3],
                               .shs base colors ([B,N,3], [B,N,1,3], [N,3] or [N,1,3])
        radii                  [B,N] or [N] screen-space radii in the source view
        contrib                [B,N] or [N] render contribution of each Gaussian in the source view
        source_intrinsic       [B,3,3] or [B,4,4]
        source_c2w             [B,4,4]
        app_token              [B,N,C] or [N,C] Stage-1 appearance token per Gaussian
        source_dino_feat_bchw  [B,Cd,Hd,Wd] image-encoder features of the source image
                               (optional; zeros are used when absent)

    Returns:
        final_rgb              [B,N,3]
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        basis_size: int = 64,
        num_heads: int = 8,
        composer_layers: int = 4,
        evidence_dropout_prob: float = 0.10,
        observed_mask_threshold: float = 0.50,   # tau_m
        global_gate_init_bias: float = -1.5,
        source_encoder_feat_dim: int = 64,
        dino_feat_dim: int = 1024,
        identity_delta_scale: float = 0.1,
        visibility_ratio: float = 0.5,           # kappa
        confidence_quantile: float = 0.95,       # q
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.basis_size = int(basis_size)
        self.evidence_dropout_prob = float(evidence_dropout_prob)
        self.observed_mask_threshold = float(observed_mask_threshold)
        self.source_feat_dim = int(source_encoder_feat_dim)
        self.dino_feat_dim = int(dino_feat_dim)
        self.identity_delta_scale = float(identity_delta_scale)
        self.visibility_ratio = float(visibility_ratio)
        self.confidence_quantile = float(confidence_quantile)

        # NOTE: submodule names and their creation order are fixed. Names must match the
        # released checkpoint, and the order determines random initialization in training.

        # ---- local evidence (source-visible Gaussians only) ----
        self.source_encoder = XUNet(model_name="resnet18", encoder_feat_dim=self.source_feat_dim)

        # source rgb (3), rendered rgb (3), residual (3), XUNet feature (F), confidence (1)
        self.local_evidence_dim = 3 + 3 + 3 + self.source_feat_dim + 1
        self.local_evidence_proj = nn.Sequential(
            nn.Linear(self.local_evidence_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.local_evidence_refine = ResidualMLP(dim=hidden_dim, expansion=4, dropout=evidence_dropout_prob)

        # ---- global source identity ----
        # masked image-encoder feature (Cd), masked XUNet feature (F), source rgb mean/std (6),
        # base color mean/std (6), observed residual mean/std (6), pooled observed evidence (hidden)
        self.identity_input_dim = self.dino_feat_dim + self.source_feat_dim + 6 + 6 + 6 + hidden_dim
        self.source_identity_proj = nn.Sequential(
            nn.Linear(self.identity_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # ---- Stage-1 priors (input sizes are inferred on the first forward pass) ----
        self.app_prior_proj = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.base_color_proj = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # updates the appearance prior with local evidence and the identity token
        self.app_prior_updater = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, num_heads=num_heads, dropout=evidence_dropout_prob),
            CrossAttentionBlock(hidden_dim, num_heads=num_heads, dropout=evidence_dropout_prob),
        ])

        # ---- identity-conditioned basis bank ----
        self.appearance_basis = nn.Parameter(torch.randn(basis_size, hidden_dim) * 0.02)
        self.identity_to_basis = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ---- per-Gaussian query and composer ----
        # query = [local evidence, updated prior, identity, base color token]
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.appearance_composer = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, num_heads=num_heads, dropout=evidence_dropout_prob)
            for _ in range(composer_layers)
        ])

        # ---- output heads ----
        self.observed_head = nn.Sequential(        # c_obs
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.identity_delta_head = nn.Sequential(  # delta_i, averaged into Delta_aid
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.global_gate_head = nn.Sequential(     # g
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.constant_(self.global_gate_head[-1].bias, global_gate_init_bias)

    # ------------------------------------------------------------------
    # Pooling helpers
    # ------------------------------------------------------------------
    @staticmethod
    def weighted_pool(token: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """token [B,N,D], weight [B,N] -> [B,D]"""
        w = weight.unsqueeze(-1)
        return (token * w).sum(dim=1) / w.sum(dim=1).clamp_min(eps)

    @staticmethod
    def weighted_mean_std(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """x [B,N,C], weight [B,N] -> [B,2C] (mean, std)"""
        w = weight.unsqueeze(-1)
        denom = w.sum(dim=1).clamp_min(eps)
        mean = (x * w).sum(dim=1) / denom
        var = (((x - mean.unsqueeze(1)) ** 2) * w).sum(dim=1) / denom
        std = torch.sqrt(var.clamp_min(eps))
        return torch.cat([mean, std], dim=-1)

    @staticmethod
    def masked_image_mean_std(image: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """image [B,C,H,W], mask [B,1,H',W'] -> [B,2C] (mean, std inside the mask)"""
        if mask.shape[-2:] != image.shape[-2:]:
            mask = F.interpolate(mask.float(), size=image.shape[-2:], mode="bilinear", align_corners=False)
        mask = mask.float().clamp(0.0, 1.0)
        denom = mask.sum(dim=(2, 3)).clamp_min(eps)
        mean = (image * mask).sum(dim=(2, 3)) / denom
        var = (((image - mean[:, :, None, None]) ** 2) * mask).sum(dim=(2, 3)) / denom
        std = torch.sqrt(var.clamp_min(eps))
        return torch.cat([mean, std], dim=-1)

    @staticmethod
    def masked_feature_pool(feat: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """feat [B,C,Hf,Wf], mask [B,1,H,W] -> [B,C] (mean inside the mask)"""
        if mask.shape[-2:] != feat.shape[-2:]:
            mask = F.interpolate(mask.float(), size=feat.shape[-2:], mode="bilinear", align_corners=False)
        mask = mask.float().clamp(0.0, 1.0)
        denom = mask.sum(dim=(2, 3)).clamp_min(eps)
        return (feat * mask).sum(dim=(2, 3)) / denom

    # ------------------------------------------------------------------
    # Shape / image helpers
    # ------------------------------------------------------------------
    @staticmethod
    def to_base_color(shs: torch.Tensor) -> torch.Tensor:
        """[B,N,3], [B,N,1,3], [N,3] or [N,1,3] -> [B,N,3]"""
        if shs.ndim == 2:
            shs = shs.unsqueeze(0)
        elif shs.ndim == 3 and shs.shape[-2] == 1 and shs.shape[-1] == 3:
            shs = shs.squeeze(-2).unsqueeze(0)
        elif shs.ndim == 4 and shs.shape[-2] == 1:
            shs = shs.squeeze(-2)
        return shs

    @staticmethod
    def normalize_image(x: torch.Tensor) -> torch.Tensor:
        """ImageNet normalization for the XUNet encoder."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - mean) / std

    # ------------------------------------------------------------------
    # Projection and footprint sampling
    # ------------------------------------------------------------------
    @staticmethod
    def project_to_uv(
        posed_xyz: torch.Tensor,
        source_intrinsic: torch.Tensor,
        source_c2w: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Pinhole projection. posed_xyz [B,N,3], K [B,3,3], c2w [B,4,4] -> pixel uv [B,N,2]"""
        assert posed_xyz.ndim == 3 and posed_xyz.shape[-1] == 3
        assert source_intrinsic.ndim == 3 and source_intrinsic.shape[-2:] == (3, 3)
        assert source_c2w.ndim == 3 and source_c2w.shape[-2:] == (4, 4)

        source_w2c = torch.inverse(source_c2w)
        posed_xyz_h = torch.cat([posed_xyz, torch.ones_like(posed_xyz[..., :1])], dim=-1)
        cam_xyz = torch.matmul(posed_xyz_h, source_w2c.transpose(-1, -2))[..., :3]

        x = cam_xyz[..., 0]
        y = cam_xyz[..., 1]
        z = cam_xyz[..., 2].clamp_min(eps)

        fx = source_intrinsic[:, 0, 0].unsqueeze(1)
        fy = source_intrinsic[:, 1, 1].unsqueeze(1)
        cx = source_intrinsic[:, 0, 2].unsqueeze(1)
        cy = source_intrinsic[:, 1, 2].unsqueeze(1)

        u = fx * (x / z) + cx
        v = fy * (y / z) + cy
        return torch.stack([u, v], dim=-1)

    @staticmethod
    def gaussian_weighted_sample(
        image: torch.Tensor,
        gaussian_uv: torch.Tensor,
        gaussian_radii: torch.Tensor,
        patch_size: int = 7,
        eps: float = 1e-6,
        padding_mode: str = "border",
        mask: torch.Tensor | None = None,
        sigma_mult: float = 0.5,
    ) -> torch.Tensor:
        """Average `image` over each Gaussian's screen footprint.

        A P x P pixel patch around each projected center is weighted by an isotropic
        Gaussian with sigma = sigma_mult * max(radius, 1). With `mask`, only pixels inside
        the mask count: sum(image * w * mask) / sum(w * mask).

        image [B,C,H,W], gaussian_uv [B,N,2], gaussian_radii [B,N], mask [B,1,H,W] -> [B,N,C]
        """
        B, C, H, W = image.shape
        P = patch_size

        offsets = torch.linspace(-(P // 2), (P // 2), P, device=image.device, dtype=image.dtype)
        dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
        dx = dx.view(1, 1, P, P)
        dy = dy.view(1, 1, P, P)

        x = gaussian_uv[..., 0].unsqueeze(-1).unsqueeze(-1) + dx
        y = gaussian_uv[..., 1].unsqueeze(-1).unsqueeze(-1) + dy
        x_norm = (x / max(W - 1, 1)) * 2.0 - 1.0
        y_norm = (y / max(H - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1).view(B, -1, P, 2)

        sampled = F.grid_sample(image.float(), grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)
        sampled = sampled.view(B, C, -1, P, P).permute(0, 2, 1, 3, 4)

        sigma = gaussian_radii.unsqueeze(-1).unsqueeze(-1).clamp(min=1.0) * sigma_mult
        weights = torch.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma ** 2))
        weights = weights / (weights.sum(dim=(-1, -2), keepdim=True) + eps)

        if mask is None:
            return (sampled * weights.unsqueeze(2)).sum(dim=(-1, -2))

        if mask.shape[-2:] != (H, W):
            mask = F.interpolate(mask.float(), size=(H, W), mode="bilinear", align_corners=False)
        mask_sampled = F.grid_sample(mask.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        mask_sampled = mask_sampled.view(B, 1, -1, P, P).permute(0, 2, 1, 3, 4)

        masked_weights = weights.unsqueeze(2) * mask_sampled.clamp(0.0, 1.0)
        denom = masked_weights.sum(dim=(-1, -2)).clamp_min(eps)
        return (sampled * masked_weights).sum(dim=(-1, -2)) / denom

    @staticmethod
    def resize_uv_and_radii(uv: torch.Tensor, radii: torch.Tensor, src_h: int, src_w: int, dst_h: int, dst_w: int):
        """Rescale pixel uv and radii from a (src_h, src_w) image to a (dst_h, dst_w) feature map."""
        scale_x = dst_w / max(src_w, 1)
        scale_y = dst_h / max(src_h, 1)
        feature_uv = uv.clone()
        feature_uv[..., 0] = uv[..., 0] * scale_x
        feature_uv[..., 1] = uv[..., 1] * scale_y
        return feature_uv, radii * (0.5 * (scale_x + scale_y))

    def _get_source_dino_feat(self, inputs: dict, source_image: torch.Tensor) -> torch.Tensor:
        """Image-encoder features of the source image; zeros when the renderer does not provide them."""
        feat = inputs.get("source_dino_feat_bchw")
        if feat is None:
            feat = inputs.get("source_dino_feat_bcwh")  # legacy key (same [B,C,H,W] layout)
        if feat is None:
            B, _, H, W = source_image.shape
            feat = torch.zeros(
                B, self.dino_feat_dim, max(1, H // 16), max(1, W // 16),
                device=source_image.device, dtype=source_image.dtype,
            )
        return feat

    # ------------------------------------------------------------------
    # Source-observation confidence (gamma)
    # ------------------------------------------------------------------
    def compute_mask_ratio(
        self,
        source_mask: torch.Tensor,
        posed_uv: torch.Tensor,
        radii: torch.Tensor,
        patch_size: int = 7,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Overlap m_i of each Gaussian's footprint with the source mask -> [B,N]"""
        sampled_mask = self.gaussian_weighted_sample(
            image=source_mask,
            gaussian_uv=posed_uv,
            gaussian_radii=radii,
            patch_size=patch_size,
            eps=eps,
            padding_mode="zeros",
        )
        return sampled_mask.squeeze(-1)

    def compute_render_aware_gs_confidence(
        self,
        gaussian_contribution: torch.Tensor,
        mask_overlap_ratio: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """gamma_i in [0,1] from render contribution C_i and mask overlap m_i -> [B,N]

        visible   C_i > kappa * median of the nonzero contributions
        observed  visible and m_i > tau_m
        score     sqrt(C_i) * m_i on observed Gaussians, 0 elsewhere
        gamma     score / (q-quantile of the observed scores), clamped to [0,1]
        """
        contrib = gaussian_contribution.clamp_min(0.0)

        nonzero = contrib.masked_fill(~(contrib > 1e-6), float("nan"))
        median = torch.nan_to_num(torch.nanquantile(nonzero, 0.50, dim=1, keepdim=True), nan=0.0)
        visible = contrib > self.visibility_ratio * median
        observed = visible & (mask_overlap_ratio > self.observed_mask_threshold)

        score = torch.sqrt(contrib) * mask_overlap_ratio * observed.float()

        ref = torch.nanquantile(score.masked_fill(~observed, float("nan")), self.confidence_quantile, dim=1)
        ref = torch.where(torch.isnan(ref), torch.ones_like(ref), ref)  # no observed Gaussian

        confidence = (score / (ref.unsqueeze(1) + eps)).clamp(0.0, 1.0)
        return torch.nan_to_num(confidence, nan=0.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, inputs: dict) -> torch.Tensor:
        source_image = inputs["source_image"]
        source_mask = inputs["source_mask"].float()
        source_render_rgb = inputs["source_render_rgb"]
        source_c2w = inputs["source_c2w"]
        source_intrinsic = inputs["source_intrinsic"]
        if source_intrinsic.shape[-2:] == (4, 4):
            source_intrinsic = source_intrinsic[..., :3, :3]

        posed_xyz = inputs["posed_base_gs"].xyz
        if posed_xyz.ndim == 2:
            posed_xyz = posed_xyz.unsqueeze(0)
        base_color = self.to_base_color(inputs["posed_base_gs"].shs)  # c0, [B,N,3]

        app_token = inputs["app_token"]
        radii = inputs["radii"]
        contrib = inputs["contrib"]
        if app_token.ndim == 2:
            app_token = app_token.unsqueeze(0)
        if radii.ndim == 1:
            radii = radii.unsqueeze(0)
        if contrib.ndim == 1:
            contrib = contrib.unsqueeze(0)

        B, N = base_color.shape[:2]

        # ---- source-observation confidence gamma ----
        uv = self.project_to_uv(posed_xyz, source_intrinsic, source_c2w)
        mask_ratio = self.compute_mask_ratio(source_mask, uv, radii)
        confidence = self.compute_render_aware_gs_confidence(contrib, mask_ratio)  # [B,N]

        # ---- local evidence ----
        src_rgb = self.gaussian_weighted_sample(source_image, uv, radii, mask=source_mask)
        rendered_rgb = self.gaussian_weighted_sample(source_render_rgb, uv, radii, mask=source_mask)
        residual = src_rgb - rendered_rgb

        source_feat = self.source_encoder(self.normalize_image(source_image))
        feat_uv, feat_radii = self.resize_uv_and_radii(uv, radii, *source_image.shape[-2:], *source_feat.shape[-2:])
        src_feat = self.gaussian_weighted_sample(source_feat, feat_uv, feat_radii, mask=source_mask)

        local_evidence = torch.cat([src_rgb, rendered_rgb, residual, src_feat, confidence.unsqueeze(-1)], dim=-1)
        local_token = self.local_evidence_refine(self.local_evidence_proj(local_evidence))

        # ---- global source identity (image-encoder features are pooled globally only) ----
        identity_input = torch.cat([
            self.masked_feature_pool(self._get_source_dino_feat(inputs, source_image), source_mask),
            self.masked_feature_pool(source_feat, source_mask),
            self.masked_image_mean_std(source_image, source_mask),
            self.weighted_mean_std(base_color, torch.ones(B, N, device=base_color.device, dtype=base_color.dtype)),
            self.weighted_mean_std(residual, confidence),
            self.weighted_pool(local_token, confidence),
        ], dim=-1)
        identity_token = self.source_identity_proj(identity_input)  # [B,hidden]
        identity_per_gs = identity_token.unsqueeze(1).expand(-1, N, -1)

        # ---- update the Stage-1 appearance prior with the source ----
        app_prior = self.app_prior_proj(app_token)
        base_color_token = self.base_color_proj(base_color)
        prior_memory = torch.cat([local_token, identity_token.unsqueeze(1)], dim=1)
        for blk in self.app_prior_updater:
            app_prior = blk(app_prior, prior_memory)

        basis_bank = self.appearance_basis.unsqueeze(0) + self.identity_to_basis(identity_token).unsqueeze(1)

        # ---- per-Gaussian appearance token ----
        token = self.query_proj(torch.cat([
            local_token,
            app_prior,
            identity_per_gs,
            base_color_token,
        ], dim=-1))
        composer_memory = torch.cat([identity_token.unsqueeze(1), basis_bank], dim=1)
        for blk in self.appearance_composer:
            token = blk(token, composer_memory)

        # ---- compose c* ----
        observed_rgb = torch.sigmoid(self.observed_head(token))    # c_obs
        gate = torch.sigmoid(self.global_gate_head(token))          # g
        local_rgb = base_color + (confidence.unsqueeze(-1) * gate) * (observed_rgb - base_color)

        delta = torch.tanh(self.identity_delta_head(token)) * self.identity_delta_scale
        final_rgb = (local_rgb + delta.mean(dim=1, keepdim=True)).clamp(0.0, 1.0)  # + Delta_aid

        # Per-Gaussian maps from the last call, kept for visualization.
        self.last_observed_confidence = confidence.detach()
        self.last_global_gate = gate.detach().squeeze(-1)
        self.last_evidence_weight = (gate.squeeze(-1) * confidence).detach()

        return final_rgb