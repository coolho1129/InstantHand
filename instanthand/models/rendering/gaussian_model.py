# Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors.
# Copyright (c) 2026 The InstantHand Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from GaussianModel in LAM (https://github.com/aigc3d/LAM,
# lam/models/rendering/gaussian_model.py), licensed under the Apache License 2.0.
# The PLY layout of save_ply / load_ply follows GaussianModel of 3D Gaussian Splatting
# (https://github.com/graphdeco-inria/gaussian-splatting), distributed under its own license.
# Modified by the InstantHand authors: documentation, unused and debug-only methods removed.
"""Container for the 3D Gaussians of one avatar, with PLY export and import."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from torch import Tensor

from instanthand.models.rendering.utils.sh_utils import RGB2SH, SH2RGB
from instanthand.models.rendering.utils.utils import trunc_exp

__all__ = ["GaussianModel"]


def inverse_sigmoid(x: np.ndarray) -> np.ndarray:
    return np.log(x / (1 - x))


class GaussianModel:
    """The Gaussians of one avatar (N of them), as used by the renderer.

    xyz       [N, 3]     centers
    opacity   [N, 1]     in (0, 1)
    rotation  [N, 4]     unit quaternions (w, x, y, z)
    scaling   [N, 3]     positive scales
    shs       [N, K, 3]  color: K = 1 holds RGB in InstantHand (K > 1: SH coefficients);
                         a flat [N, 3K] layout is also accepted by save_ply
    offset    [N, 3]     offset of each center from its MANO anchor

    ply_path: load the Gaussians from a PLY file instead (see load_ply).
    """

    def __init__(self, xyz: Optional[Tensor] = None, opacity: Optional[Tensor] = None,
                 rotation: Optional[Tensor] = None, scaling: Optional[Tensor] = None,
                 shs: Optional[Tensor] = None, offset: Optional[Tensor] = None,
                 ply_path: Optional[str] = None, sh2rgb: bool = False):
        self.xyz = xyz
        self.opacity = opacity
        self.rotation = rotation
        self.scaling = scaling
        self.shs = shs
        self.offset = offset
        if ply_path is not None:
            self.load_ply(ply_path, sh2rgb=sh2rgb)

    def to(self, device) -> "GaussianModel":
        for name in ("xyz", "opacity", "rotation", "scaling", "shs", "offset"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.to(device))
        return self

    # ------------------------------------------------------------------
    # PLY (3D Gaussian Splatting layout)
    # ------------------------------------------------------------------
    def _color_dc_rest(self):
        """Split the color into the DC term [N,1,3] and the remaining coefficients."""
        if self.shs.dim() == 2:
            return self.shs[:, :3].unsqueeze(1), self.shs[:, 3:].unsqueeze(1)
        return self.shs[:, :1], self.shs[:, 1:]

    def construct_list_of_attributes(self) -> list:
        features_dc, features_rest = self._color_dc_rest()
        names = ["x", "y", "z", "nx", "ny", "nz"]
        names += [f"f_dc_{i}" for i in range(features_dc.shape[1] * features_dc.shape[2])]
        names += [f"f_rest_{i}" for i in range(features_rest.shape[1] * features_rest.shape[2])]
        names.append("opacity")
        names += [f"scale_{i}" for i in range(self.scaling.shape[1])]
        names += [f"rot_{i}" for i in range(self.rotation.shape[1])]
        return names

    def save_ply(self, path: str, rgb2sh: bool = False) -> None:
        """Write the Gaussians as a 3DGS PLY: opacity as a logit, scales as logs.

        rgb2sh: store the RGB color as the SH DC term, as 3DGS viewers expect.
        """
        def to_numpy(t: Tensor) -> np.ndarray:
            return t.detach().float().contiguous().cpu().numpy()

        xyz = to_numpy(self.xyz)
        features_dc, features_rest = self._color_dc_rest()
        f_dc = to_numpy(features_dc.float().flatten(start_dim=1))
        f_rest = to_numpy(features_rest.float().flatten(start_dim=1))
        if rgb2sh:
            f_dc = RGB2SH(f_dc)
        opacities = inverse_sigmoid(to_numpy(torch.clamp(self.opacity, 1e-3, 1 - 1e-3)))
        scale = np.log(to_numpy(self.scaling))
        rotation = to_numpy(self.rotation)

        elements = np.empty(xyz.shape[0], dtype=[(name, "f4") for name in self.construct_list_of_attributes()])
        attributes = np.concatenate((xyz, np.zeros_like(xyz), f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    def load_ply(self, path: str, sh2rgb: bool = False) -> None:
        """Read a 3DGS PLY with RGB / degree-0 color (no f_rest_* fields).

        sh2rgb=False: raw PLY values (SH DC color, opacity logits, log scales).
        sh2rgb=True:  the values the renderer uses (RGB color, opacity, scales), i.e. the inverse
                      of save_ply(rgb2sh=True). The offsets are set to zero.
        """
        vertex = PlyData.read(path).elements[0]

        def fields(prefix: str) -> np.ndarray:
            names = sorted((p.name for p in vertex.properties if p.name.startswith(prefix)),
                           key=lambda name: int(name.split("_")[-1]))
            return np.stack([np.asarray(vertex[name]) for name in names], axis=1)

        if any(p.name.startswith("f_rest_") for p in vertex.properties):
            raise ValueError(f"{path}: only RGB / degree-0 color is supported (found f_rest_* fields)")

        def tensor(a: np.ndarray) -> Tensor:
            return torch.tensor(a, dtype=torch.float)

        xyz = np.stack([np.asarray(vertex[c]) for c in ("x", "y", "z")], axis=1)
        self.xyz = tensor(xyz)
        shs = tensor(fields("f_dc_")).unsqueeze(1)                     # [N,1,3]
        self.shs = SH2RGB(shs) if sh2rgb else shs
        self.opacity = tensor(np.asarray(vertex["opacity"])[:, None])
        self.scaling = tensor(fields("scale_"))
        self.rotation = tensor(fields("rot_"))
        self.offset = torch.zeros_like(self.xyz)
        if sh2rgb:
            self.opacity = torch.sigmoid(self.opacity)
            self.scaling = trunc_exp(self.scaling)
        self.sh_degree = 0