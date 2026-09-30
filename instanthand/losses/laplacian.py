# Copyright (c) 2026 The InstantHand Authors.
# SPDX-License-Identifier: Apache-2.0
"""Mesh Laplacian regularizer."""
import numpy as np
import torch
import torch.nn as nn

__all__ = ["LaplacianReg"]


class LaplacianReg(nn.Module):
    """Uniform graph Laplacian on a triangle mesh.

    For vertex i with neighbors N(i) (sorted by index, at most `neighbor_max_num`),
        L(x)_i = x_i - mean_{j in N(i)} x_j.
    forward returns L(out)^2, or (L(out) - L(target))^2 if a target is given, elementwise.

    Neighbor indices and weights are non-persistent buffers: they follow .to(device) and
    are not stored in checkpoints.
    """

    def __init__(self, vertex_num, face, device=None, neighbor_max_num=20):
        super().__init__()
        self.neighbor_max_num = int(neighbor_max_num)

        idxs, wts = self.get_neighbor(vertex_num, face, neighbor_max_num=self.neighbor_max_num)
        self.register_buffer("neighbor_idxs", idxs if device is None else idxs.to(device), persistent=False)
        self.register_buffer("neighbor_weights", wts if device is None else wts.to(device), persistent=False)

    @staticmethod
    def get_neighbor(vertex_num, face, neighbor_max_num=20):
        """Padded neighbor table [V, K] and weights -1/n ([V, K]; 0 on padding).

        Padding entries point to the vertex itself with weight 0.
        """
        adj = {i: set() for i in range(vertex_num)}
        for tri in face:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            adj[a].update([b, c])
            adj[b].update([a, c])
            adj[c].update([a, b])

        neighbor_idxs = np.tile(np.arange(vertex_num)[:, None], (1, neighbor_max_num))
        neighbor_weights = np.zeros((vertex_num, neighbor_max_num), dtype=np.float32)

        for vid in range(vertex_num):
            neigh = sorted(adj[vid])  # deterministic order
            n = min(len(neigh), neighbor_max_num)
            if n > 0:
                neighbor_idxs[vid, :n] = np.asarray(neigh[:n], dtype=np.int64)
                neighbor_weights[vid, :n] = -1.0 / float(n)

        return torch.from_numpy(neighbor_idxs).long(), torch.from_numpy(neighbor_weights).float()

    def compute_laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,V,3] -> [B,V,3]"""
        w = self.neighbor_weights.to(dtype=x.dtype)  # match x under mixed precision
        return x + (x[:, self.neighbor_idxs] * w[None, :, :, None]).sum(2)

    def forward(self, out: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        """out, target: [B,V,3] -> elementwise squared Laplacian (difference), [B,V,3]"""
        if target is None:
            return self.compute_laplacian(out).pow(2)
        lap_out = self.compute_laplacian(out)
        lap_tgt = self.compute_laplacian(target)
        return (lap_out - lap_tgt).pow(2)