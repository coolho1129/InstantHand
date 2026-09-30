# Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors.
# Copyright (c) 2026 The InstantHand Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from DPTHead and Dinov2FusionWrapper in LAM (https://github.com/aigc3d/LAM,
# lam/models/encoders/dinov2_fusion_wrapper.py), licensed under the Apache License 2.0.
# Modified by the InstantHand authors: DINOv3 backbone loaded from local weights; the DINOv2
# backbone, feature modulation and the class-token readout of DPTHead are removed.
#
# The DINOv3 backbone code (./dinov3) and its pretrained weights are released by Meta under
# the DINOv3 License, not under Apache 2.0; see the license file in that directory.
"""Image encoder: frozen DINOv3 ViT + DPT-style fusion of four intermediate blocks.

    image [N,3,H,W] -> tokens [N, (H/16)*(W/16), encoder_feat_dim]
"""
from __future__ import annotations

import logging
from importlib import import_module
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = ["DPTHead", "Dinov3FusionWrapper"]

# Transformer blocks whose patch tokens are fused (four evenly spaced blocks per backbone).
INTERMEDIATE_LAYERS = {
    "dinov3_vits16": [2, 5, 8, 11],
    "dinov3_vits16plus": [2, 5, 8, 11],
    "dinov3_vitb16": [2, 5, 8, 11],
    "dinov3_vitl16": [4, 11, 17, 23],
    "dinov3_vitl16plus": [4, 11, 17, 23],
    "dinov3_vith16plus": [6, 13, 21, 31],
    "dinov3_vit7b16": [9, 19, 29, 39],
}


class DPTHead(nn.Module):
    """Fuse the patch tokens of several blocks: a 1x1 projection per block, channel
    concatenation, and a 1x1 output convolution to `out_channel`."""

    def __init__(self, in_channels: int, inner_channels: Sequence[int], out_channel: int = 1024):
        super().__init__()
        self.projects = nn.ModuleList([nn.Conv2d(in_channels, c, kernel_size=1) for c in inner_channels])
        self.output_conv = nn.Conv2d(sum(inner_channels), out_channel, kernel_size=1)

    def forward(self, features, patch_h: int, patch_w: int) -> torch.Tensor:
        """features: one (patch_tokens [N, h*w, C], class_token [N, C]) pair per block; the class
        token is not used. Returns [N, out_channel, h, w]."""
        out = []
        for project, feat in zip(self.projects, features):
            tokens = feat[0]
            x = tokens.permute(0, 2, 1).reshape(tokens.shape[0], tokens.shape[-1], patch_h, patch_w)
            out.append(project(x))
        return self.output_conv(torch.cat(out, dim=1))


class Dinov3FusionWrapper(nn.Module):
    """DINOv3 backbone (`self.model`, frozen by default) and fusion head (`self.fusion_head`).

    The attribute names are part of the checkpoint keys (encoder.model.*, encoder.fusion_head.*)
    and are used by ModelHAM to freeze the backbone and train the head.
    """

    def __init__(
        self,
        model_name: str,
        freeze: bool = True,
        encoder_feat_dim: int = 384,
        weights_path: Optional[str] = None,
        pretrained: bool = True,
    ):
        super().__init__()
        if model_name not in INTERMEDIATE_LAYERS:
            raise ValueError(f"Unsupported DINOv3 model: {model_name}; choose from {sorted(INTERMEDIATE_LAYERS)}")

        self.model = self._build_backbone(model_name, pretrained=pretrained, weights_path=weights_path)
        self.intermediate_layer_idx = INTERMEDIATE_LAYERS[model_name]
        self.fusion_head = DPTHead(
            in_channels=self.model.embed_dim,
            inner_channels=[self.model.embed_dim] * len(self.intermediate_layer_idx),
            out_channel=encoder_feat_dim,
        )
        if freeze:
            self._freeze()

    def _freeze(self) -> None:
        logger.info("Freezing the DINOv3 backbone")
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @staticmethod
    def _build_backbone(model_name: str, pretrained: bool = True, weights_path: Optional[str] = None) -> nn.Module:
        """Build the DINOv3 backbone. `weights_path` may be a local file or a file:// / https:// URL;
        without it, the backbone is built without pretrained weights (they come from the checkpoint)."""
        model_fn = getattr(import_module(".dinov3.hub.backbones", package=__package__), model_name)
        if not (pretrained and weights_path):
            return model_fn(pretrained=False)

        spec = str(weights_path).strip()
        if not spec.startswith(("file://", "https://")):
            path = Path(spec).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"DINOv3 weights not found: {path}")
            spec = path.resolve().as_uri()
        logger.info("Loading DINOv3 weights from %s", spec)
        return model_fn(pretrained=True, weights=spec)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image: [N,3,H,W], H and W multiples of the patch size (16). Returns [N, h*w, encoder_feat_dim]."""
        patch_h = image.shape[-2] // self.model.patch_size
        patch_w = image.shape[-1] // self.model.patch_size
        features = self.model.get_intermediate_layers(
            image, n=self.intermediate_layer_idx, reshape=False, return_class_token=True, norm=True
        )
        fused = self.fusion_head(features, patch_h, patch_w)  # [N, C, h, w]
        return fused.permute(0, 2, 3, 1).flatten(1, 2)