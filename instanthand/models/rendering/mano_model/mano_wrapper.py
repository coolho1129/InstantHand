# Copyright (c) 2023 UC Regents, Georgios Pavlakos
# Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors.
# Copyright (c) 2026 The InstantHand Authors.
# SPDX-License-Identifier: Apache-2.0 AND MIT
#
# MANO is taken from HaMeR (https://github.com/geopavlakos/hamer, hamer/models/mano_wrapper.py),
# released under the MIT License (see LICENSES/HaMeR-MIT.txt).
# MANOHandSubdivided is adapted from FlameHeadSubdivided in LAM (https://github.com/aigc3d/LAM,
# lam/models/rendering/flame_model/flame.py), licensed under the Apache License 2.0; ported
# from FLAME to MANO and modified by the InstantHand authors.
# The smplx package and the MANO model files are not part of this repository; they are
# distributed under their own licenses (https://github.com/vchoutas/smplx,
# https://mano.is.tue.mpg.de).
"""MANO hand model on a subdivided mesh, used to place and pose the Gaussian anchors.

MANOHandSubdivided subdivides the MANO template `subdivide_num` times (midpoint subdivision)
and interpolates the per-vertex model data (skinning weights, shape and pose blend shapes,
joint regressor) to the new vertices. Points on that mesh are posed by linear blend skinning
with the joint transforms of the original 778-vertex MANO model.
"""
from __future__ import annotations

import pickle
from typing import Optional

import smplx
import torch
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.structures import Meshes
from smplx.lbs import batch_rigid_transform, batch_rodrigues, blend_shapes, vertices2joints
from smplx.utils import MANOOutput, to_tensor
from smplx.vertex_ids import vertex_ids

__all__ = ["MANO", "MANOHandSubdivided"]

# 16 MANO joints + 5 fingertip vertices, reordered to the OpenPose hand layout.
MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


class MANO(smplx.MANOLayer):
    """smplx MANO layer that also returns the fingertips, in OpenPose joint order (from HaMeR).

    joint_regressor_extra: optional path to a regressor for additional joints.
    """

    def __init__(self, *args, joint_regressor_extra: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if joint_regressor_extra is not None:
            with open(joint_regressor_extra, "rb") as f:
                regressor = pickle.load(f, encoding="latin1")
            self.register_buffer("joint_regressor_extra", torch.tensor(regressor, dtype=torch.float32))
        # NOTE: both buffers are stored in InstantHand checkpoints (renderer.mano_model.*), keep them.
        self.register_buffer("extra_joints_idxs", to_tensor(list(vertex_ids["mano"].values()), dtype=torch.long))
        self.register_buffer("joint_map", torch.tensor(MANO_TO_OPENPOSE, dtype=torch.long))

    def forward(self, *args, **kwargs) -> MANOOutput:
        mano_output = super().forward(*args, **kwargs)
        extra_joints = torch.index_select(mano_output.vertices, 1, self.extra_joints_idxs)
        joints = torch.cat([mano_output.joints, extra_joints], dim=1)[:, self.joint_map, :]
        if hasattr(self, "joint_regressor_extra"):
            joints = torch.cat([joints, vertices2joints(self.joint_regressor_extra, mano_output.vertices)], dim=1)
        mano_output.joints = joints
        return mano_output


class MANOHandSubdivided(MANO):
    """MANO with a mesh subdivided `subdivide_num` times.

    Buffers of the subdivided mesh (rebuilt from the MANO file, not stored in checkpoints):
        faces_up        [F_up, 3]
        v_template_up   [V_up, 3]
        lbs_weights_up  [V_up, J]
        posedirs_up     [P, V_up*3]
        shapedirs_up    [V_up, 3, S]
        J_regressor_up  [J, V_up]

    Joint transforms always come from the original mesh (v_template, shapedirs, J_regressor);
    only the skinned points live on the subdivided mesh. Poses are axis-angle:
    rotation (global orientation) [B,3], hand_pose [B,45]; shape (betas) [B,S]; translation [B,3].
    """

    def __init__(self, *args, subdivide_num: int = 0, joint_regressor_extra: Optional[str] = None, **kwargs):
        super().__init__(*args, joint_regressor_extra=joint_regressor_extra, **kwargs)
        if subdivide_num < 0:
            raise ValueError(f"subdivide_num must be >= 0, got {subdivide_num}")
        self.subdivide_num = int(subdivide_num)
        self.subdivider_cpu_list = self.get_subdivider_cpu(self.subdivide_num)

        if self.subdivide_num > 0:
            self.face_upsampled = self.subdivider_cpu_list[-1]._subdivided_faces.numpy()
        else:
            self.face_upsampled = self.faces.cpu().numpy() if torch.is_tensor(self.faces) else self.faces
        self.vertex_num_upsampled = int(self.face_upsampled.max() + 1)
        self.vertex_num = int(self.v_template.shape[0])   # 778
        self.joint_num = int(self.J_regressor.shape[0])   # 16

        # Per-vertex model data as [V, D] rows, interpolated to the new vertices like vertex features.
        V = self.vertex_num
        n_shape = self.shapedirs.shape[-1]
        posedirs = self.posedirs
        if posedirs.dim() != 2:
            raise ValueError(f"Unexpected posedirs shape: {tuple(posedirs.shape)}")
        if posedirs.shape[0] == V * 3:        # [V*3, P]
            P = posedirs.shape[1]
            posedirs_rows = posedirs.reshape(V, 3 * P)
        elif posedirs.shape[1] == V * 3:      # [P, V*3] (smplx layout)
            P = posedirs.shape[0]
            posedirs_rows = posedirs.permute(1, 0).reshape(V, 3 * P)
        else:
            raise ValueError(f"Cannot interpret posedirs shape: {tuple(posedirs.shape)}")
        self.P_pose = int(P)

        v_template_up, lbs_weights_up, posedirs_rows_up, shapedirs_rows_up, j_regressor_rows_up = self.upsample_mesh_cpu(
            self.v_template.float(),
            [
                self.lbs_weights.float(),                        # [V, J]
                posedirs_rows,                                   # [V, 3P]
                self.shapedirs.reshape(V, 3 * n_shape),          # [V, 3S]
                self.J_regressor.permute(1, 0).contiguous(),     # [V, J]
            ],
        )
        V_up = self.vertex_num_upsampled
        self.register_buffer("faces_up", torch.from_numpy(self.face_upsampled).long(), persistent=False)
        self.register_buffer("v_template_up", v_template_up.contiguous(), persistent=False)
        self.register_buffer("lbs_weights_up", lbs_weights_up.contiguous(), persistent=False)
        self.register_buffer("posedirs_up", posedirs_rows_up.reshape(V_up * 3, P).permute(1, 0).contiguous(),
                             persistent=False)
        self.register_buffer("shapedirs_up", shapedirs_rows_up.reshape(V_up, 3, n_shape).contiguous(),
                             persistent=False)
        self.register_buffer("J_regressor_up", j_regressor_rows_up.permute(1, 0).contiguous(), persistent=False)

    # ------------------------------------------------------------------
    # Subdivision
    # ------------------------------------------------------------------
    def get_subdivider_cpu(self, subdivide_num: int) -> list:
        """One pytorch3d SubdivideMeshes per level (built on CPU); [mesh] when subdivide_num == 0."""
        mesh = Meshes(self.v_template.float()[None], torch.as_tensor(self.faces, dtype=torch.long)[None])
        if subdivide_num == 0:
            return [mesh]
        subdividers = [SubdivideMeshes(mesh)]
        for _ in range(subdivide_num - 1):
            mesh = subdividers[-1](mesh)
            subdividers.append(SubdivideMeshes(mesh))
        return subdividers

    def upsample_mesh_cpu(self, vert: torch.Tensor, feat_list: Optional[list] = None):
        """Subdivide the MANO mesh with vertices `vert` [V,3], interpolating the per-vertex
        features in `feat_list` ([V, D_k] each). Returns vert_up, or (vert_up, *feats_up)."""
        if self.subdivide_num <= 0:
            return vert if feat_list is None else (vert, *feat_list)

        mesh = Meshes(vert[None], torch.as_tensor(self.faces, dtype=torch.long)[None])
        if feat_list is None:
            for subdivider in self.subdivider_cpu_list:
                mesh = subdivider(mesh)
            return mesh.verts_list()[0]

        feat_dims = [x.shape[1] for x in feat_list]
        feats = torch.cat(feat_list, dim=1)
        for subdivider in self.subdivider_cpu_list:
            mesh, feats = subdivider(mesh, feats)
        return (mesh.verts_list()[0], *torch.split(feats[0], feat_dims, dim=1))

    # ------------------------------------------------------------------
    # Shape, pose and skinning
    # ------------------------------------------------------------------
    def get_cano_verts(self, shape: torch.Tensor) -> torch.Tensor:
        """Canonical (unposed) subdivided vertices for betas `shape` [B,S] -> [B, V_up, 3]."""
        B, device, dtype = shape.shape[0], shape.device, shape.dtype
        v0 = self.v_template_up.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        return v0 + blend_shapes(shape, self.shapedirs_up.to(device=device, dtype=dtype))

    def _pose_feature(self, full_pose: torch.Tensor, dtype, device):
        """full_pose [B, J*3] -> rotation matrices [B,J,3,3] and pose feature [B, (J-1)*9]."""
        B = full_pose.shape[0]
        rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(B, -1, 3, 3)
        ident = torch.eye(3, dtype=dtype, device=device)
        return rot_mats, (rot_mats[:, 1:, :, :] - ident).reshape(B, -1)

    def get_transformed_mat(self, full_pose: torch.Tensor, v_shaped_base: torch.Tensor, dtype):
        """Joint transforms A [B,J,4,4] and posed joints [B,J,3] of the original mesh."""
        rot_mats, _ = self._pose_feature(full_pose, dtype=dtype, device=full_pose.device)
        joints = vertices2joints(self.J_regressor, v_shaped_base)
        joints_posed, A = batch_rigid_transform(rot_mats, joints, self.parents, dtype=dtype)
        return A, joints_posed

    def _joint_transforms(self, shape, rotation, hand_pose):
        """full pose [B, 48], joint transforms A and posed joints J for the given shape and pose."""
        B, device, dtype = shape.shape[0], shape.device, shape.dtype
        full_pose = torch.cat([rotation, hand_pose], dim=1)
        v_shaped_base = self.v_template.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1) \
            + blend_shapes(shape, self.shapedirs.to(device=device, dtype=dtype))
        A, J = self.get_transformed_mat(full_pose, v_shaped_base=v_shaped_base, dtype=dtype)
        return full_pose, A, J

    @staticmethod
    def _vertex_transforms(A, lbs_weights, batch_size, num_joints):
        """Per-vertex blended transforms sum_j w_vj A_j -> [B, V, 4, 4]."""
        W = lbs_weights.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)

    @staticmethod
    def _transform_points(T, points, batch_size, dtype, device):
        homogen = torch.ones((batch_size, points.shape[1], 1), dtype=dtype, device=device)
        return torch.matmul(T, torch.cat([points, homogen], dim=2)[..., None])[:, :, :3, 0]

    def skinning(self, v_posed, A, lbs_weights, batch_size, num_joints, dtype, device):
        T = self._vertex_transforms(A, lbs_weights, batch_size, num_joints)
        return self._transform_points(T, v_posed, batch_size, dtype, device)

    def inverse_skinning(self, v_posed, A, lbs_weights, batch_size, num_joints, dtype, device):
        T = self._vertex_transforms(A, lbs_weights, batch_size, num_joints)
        return self._transform_points(torch.inverse(T), v_posed, batch_size, dtype, device)

    def _apply_pose_blendshape_up(self, v_up: torch.Tensor, full_pose: torch.Tensor) -> torch.Tensor:
        """Add the pose-dependent corrective offsets of the subdivided mesh to `v_up`."""
        B, dtype, device = full_pose.shape[0], v_up.dtype, v_up.device
        _, pose_feature = self._pose_feature(full_pose, dtype=dtype, device=device)
        pose_offsets = torch.matmul(pose_feature, self.posedirs_up.to(device=device, dtype=dtype)).view(B, -1, 3)
        return v_up + pose_offsets

    @staticmethod
    def _to_world(verts, J, translation, zero_centered_at_root_node):
        if zero_centered_at_root_node:
            verts = verts - J[:, [0]]
            J = J - J[:, [0]]
        return verts + translation[:, None, :], J + translation[:, None, :]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def animation_forward(
        self,
        v_cano: torch.Tensor,
        shape: torch.Tensor,
        rotation: torch.Tensor,
        hand_pose: torch.Tensor,
        translation: torch.Tensor,
        zero_centered_at_root_node: bool = False,
        return_verts_cano: bool = False,
        apply_pose_bs: bool = False,
    ) -> dict:
        """Pose canonical points `v_cano` [B, V_up, 3] on the subdivided mesh.

        Returns {"animated": [B, V_up, 3], "joints": [B, J, 3]} (+ "cano": v_cano).
        apply_pose_bs adds the pose correctives before skinning (not used by InstantHand).
        """
        B, device, dtype = shape.shape[0], shape.device, shape.dtype
        full_pose, A, J = self._joint_transforms(shape, rotation, hand_pose)
        v_in = self._apply_pose_blendshape_up(v_cano, full_pose) if apply_pose_bs else v_cano
        verts = self.skinning(v_in, A, self.lbs_weights_up.to(device=device, dtype=dtype),
                              B, self.joint_num, dtype, device)
        verts, J = self._to_world(verts, J, translation, zero_centered_at_root_node)
        ret = {"animated": verts, "joints": J}
        if return_verts_cano:
            ret["cano"] = v_cano
        return ret

    def inverse_animation(
        self,
        v_pose: torch.Tensor,
        shape: torch.Tensor,
        rotation: torch.Tensor,
        hand_pose: torch.Tensor,
        translation: torch.Tensor,
        zero_centered_at_root_node: bool = False,
        return_verts_cano: bool = False,
        apply_pose_bs: bool = False,
    ) -> dict:
        """Map posed points `v_pose` [B, V_up, 3] back to the canonical space.

        Returns {"cano": [B, V_up, 3], "joints": [B, J, 3]} (+ "cano_with_pose": same as "cano").
        """
        B, device, dtype = shape.shape[0], shape.device, shape.dtype
        full_pose, A, J = self._joint_transforms(shape, rotation, hand_pose)
        v_cano = self.inverse_skinning(v_pose - translation[:, None, :], A,
                                       self.lbs_weights_up.to(device=device, dtype=dtype),
                                       B, self.joint_num, dtype, device)
        if apply_pose_bs:
            v_cano = v_cano - self._apply_pose_blendshape_up(torch.zeros_like(v_cano), full_pose)
        if zero_centered_at_root_node:
            v_cano = v_cano - J[:, [0]]
            J = J - J[:, [0]]
        ret = {"cano": v_cano, "joints": J}
        if return_verts_cano:
            ret["cano_with_pose"] = v_cano
        return ret

    def forward(
        self,
        shape: torch.Tensor,
        rotation: torch.Tensor,
        hand_pose: torch.Tensor,
        translation: torch.Tensor,
        zero_centered_at_root_node: bool = False,
        return_verts_cano: bool = False,
        apply_pose_bs: bool = False,
    ) -> dict:
        """Posed subdivided MANO mesh for the given shape and pose.

        Returns {"animated": [B, V_up, 3], "joints": [B, J, 3], "faces": faces_up}
        (+ "cano": the canonical vertices, with pose correctives if apply_pose_bs).
        """
        v_cano = self.get_cano_verts(shape)
        if apply_pose_bs:
            v_cano = self._apply_pose_blendshape_up(v_cano, torch.cat([rotation, hand_pose], dim=1))
        out = self.animation_forward(v_cano, shape, rotation, hand_pose, translation,
                                     zero_centered_at_root_node=zero_centered_at_root_node)
        ret = {"animated": out["animated"], "joints": out["joints"], "faces": self.faces_up}
        if return_verts_cano:
            ret["cano"] = v_cano
        return ret