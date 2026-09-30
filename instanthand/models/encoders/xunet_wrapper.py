# Copyright (c) 2023-2024, Zexin He
# Copyright (c) 2026 The InstantHand Authors.
# SPDX-License-Identifier: Apache-2.0
#
# XUNet is taken from LHM (https://github.com/aigc3d/LHM, LHM/models/encoders/xunet_wrapper.py),
# licensed under the Apache License 2.0. Modified by the InstantHand authors: the unused
# XnetWrapper and the Swin decoder variant are removed, the default backbone is the ResNet-18
# the decoder is built for, and loading the ImageNet weights is optional.
"""Small U-Net on a timm ResNet-18/34, used by the appearance head to encode the source image.

    image [N,3,H,W] (ImageNet-normalized, H and W multiples of 32)
        -> features [N, encoder_feat_dim, H/4, W/4]
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

__all__ = ["XUNet"]

# ResNet-18/34 feature channels at strides 2, 4, 8, 16, 32; the decoder below is built for these.
RESNET_CHANNELS = [64, 64, 128, 256, 512]


class XUNet(nn.Module):
    """ResNet encoder, three transposed-conv upsampling steps with additive skip connections
    (stride 32 -> 16 -> 8 -> 4), and a 1x1 output convolution.

    pretrained: load the timm ImageNet weights (training). The released checkpoint contains
    all XUNet weights, so evaluation and inference can pass False and skip the download.
    """

    def __init__(self, model_name: str = "resnet18", encoder_feat_dim: int = 384, pretrained: bool = True):
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=pretrained)
        channels = [f["num_chs"] for f in self.encoder.feature_info]
        if channels != RESNET_CHANNELS:
            raise ValueError(f"XUNet expects a ResNet-18/34 backbone with channels {RESNET_CHANNELS}; "
                             f"{model_name} has {channels}")
        del self.encoder.global_pool
        del self.encoder.fc

        self.upconv4 = self.upconv_block(512, 256)  # stride 32 -> 16
        self.upconv3 = self.upconv_block(256, 128)  # stride 16 -> 8
        self.upconv2 = self.upconv_block(128, 64)   # stride 8 -> 4
        self.out_conv = nn.Conv2d(64, encoder_feat_dim, kernel_size=1)

    @staticmethod
    def upconv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder.forward_intermediates(x, stop_early=True, intermediates_only=True)  # strides 2..32
        x = self.upconv4(feats[4]) + feats[3]
        x = self.upconv3(x) + feats[2]
        x = self.upconv2(x) + feats[1]
        return self.out_conv(x)