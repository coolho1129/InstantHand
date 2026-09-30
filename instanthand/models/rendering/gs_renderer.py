# Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors.
# Copyright (c) 2026 The InstantHand Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from the Gaussian renderer of LAM (https://github.com/aigc3d/LAM,
# lam/models/rendering/gs_renderer.py), licensed under the Apache License 2.0.
# The camera / projection setup and the rasterizer call follow the original 3D Gaussian
# Splatting implementation (https://github.com/graphdeco-inria/gaussian-splatting), which is
# distributed under its own license.
# Modified by the InstantHand authors: Gaussians anchored on the subdivided MANO mesh,
# constrained Gaussian parameterization, source-view rendering and the appearance head.
"""3D Gaussian hand renderer: canonical Gaussians on the subdivided MANO mesh, LBS posing
with MANO, and rasterization."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from einops import rearrange
from pytorch3d.transforms import matrix_to_quaternion

from instanthand.models.appearance import AppearanceHead
from instanthand.models.rendering.gaussian_model import GaussianModel
from instanthand.models.rendering.mano_model.mano_wrapper import MANOHandSubdivided
from instanthand.models.rendering.utils.utils import MLP, trunc_exp

APPEARANCE_INPUT_KEYS = ("source_image", "source_mask", "source_intrinsic", "source_c2w")


def inverse_sigmoid(x):
    return np.log(x / (1 - x))


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
def get_projection_matrix(znear, zfar, fov_x, fov_y) -> torch.Tensor:
    """OpenGL-style perspective projection, as in 3D Gaussian Splatting."""
    tan_half_fov_y = math.tan(fov_y / 2)
    tan_half_fov_x = math.tan(fov_x / 2)

    top = tan_half_fov_y * znear
    bottom = -top
    right = tan_half_fov_x * znear
    left = -right

    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def intrinsic_to_fov(intrinsic, w, h):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    fov_x = 2 * torch.arctan2(w, 2 * fx)
    fov_y = 2 * torch.arctan2(h, 2 * fy)
    return fov_x, fov_y


class Camera:
    """Camera in the form the Gaussian rasterizer expects (transposed matrices)."""

    def __init__(self, w2c: torch.Tensor, intrinsic: torch.Tensor, fov_x: torch.Tensor, fov_y: torch.Tensor,
                 height: int, width: int) -> None:
        self.FoVx = fov_x
        self.FoVy = fov_y
        self.height = int(height)
        self.width = int(width)
        self.intrinsic = intrinsic
        self.zfar = 100.0
        self.znear = 0.01

        self.world_view_transform = w2c.transpose(0, 1)
        self.projection_matrix = get_projection_matrix(
            znear=self.znear, zfar=self.zfar, fov_x=self.FoVx, fov_y=self.FoVy,
        ).transpose(0, 1).to(w2c.device)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @staticmethod
    def from_c2w(c2w, intrinsic, height, width) -> "Camera":
        if intrinsic.shape[-2:] == (4, 4):
            intrinsic = intrinsic[:3, :3]
        w2c = torch.inverse(c2w)
        fov_x, fov_y = intrinsic_to_fov(
            intrinsic,
            w=torch.tensor(width, device=w2c.device),
            h=torch.tensor(height, device=w2c.device),
        )
        return Camera(w2c=w2c, intrinsic=intrinsic, fov_x=fov_x, fov_y=fov_y, height=height, width=width)


# ---------------------------------------------------------------------------
# Gaussian attributes and point embedding
# ---------------------------------------------------------------------------
class GSLayer(nn.Module):
    """Per-anchor Gaussian attributes from the tokens.

    Geometry (from the geometry token):
        offset   = (sigmoid(.) - 0.5) * xyz_offset_max_step   (raw if restrict_offset=False)
        xyz      = anchor + offset
        scaling  = clamp(trunc_exp(.), 0.0005, clip_scaling); one value shared by the 3 axes
                   if scale_sphere
        opacity  = 1 if fix_opacity, else sigmoid(.)
        rotation = identity if fix_rotation, else normalized quaternion
    Color (from the appearance token): sigmoid RGB if use_rgb, else SH coefficients.

    All layers except the color layer start at zero weight, so every Gaussian starts at its
    anchor with scale exp(init_scaling).
    """

    GEO_KEYS = ("xyz", "scaling", "opacity", "rotation")
    BASECOLOR_KEYS = ("shs",)

    def __init__(
        self,
        in_channels: int,
        use_rgb: bool,
        clip_scaling: float = 0.2,
        init_scaling: float = -5.0,
        scale_sphere: bool = False,
        init_density: float = 0.1,
        sh_degree: Optional[int] = None,
        restrict_offset: bool = True,
        xyz_offset_max_step: Optional[float] = None,
        fix_opacity: bool = False,
        fix_rotation: bool = False,
    ):
        super().__init__()
        self.clip_scaling = clip_scaling
        self.use_rgb = use_rgb
        self.restrict_offset = restrict_offset
        self.xyz_offset_max_step = xyz_offset_max_step
        self.fix_opacity = fix_opacity
        self.fix_rotation = fix_rotation
        self.scale_sphere = scale_sphere

        # NOTE: layer names and creation order are fixed (checkpoint keys, initialization).
        self.geo_out_layers = nn.ModuleDict()
        self.color_out_layers = nn.ModuleDict()

        xyz_layer = nn.Linear(in_channels, 3)
        nn.init.constant_(xyz_layer.weight, 0.0)
        nn.init.constant_(xyz_layer.bias, 0.0)
        self.geo_out_layers["xyz"] = xyz_layer

        scaling_layer = nn.Linear(in_channels, 1 if scale_sphere else 3)
        nn.init.constant_(scaling_layer.weight, 0.0)
        nn.init.constant_(scaling_layer.bias, init_scaling)
        self.geo_out_layers["scaling"] = scaling_layer

        if fix_opacity:
            self.geo_out_layers["opacity"] = nn.Identity()
        else:
            opacity_layer = nn.Linear(in_channels, 1)
            nn.init.constant_(opacity_layer.weight, 0.0)
            nn.init.constant_(opacity_layer.bias, inverse_sigmoid(init_density))
            self.geo_out_layers["opacity"] = opacity_layer

        if fix_rotation:
            self.geo_out_layers["rotation"] = nn.Identity()
        else:
            rotation_layer = nn.Linear(in_channels, 4)
            nn.init.constant_(rotation_layer.weight, 0.0)
            nn.init.constant_(rotation_layer.bias, 0.0)
            nn.init.constant_(rotation_layer.bias[0], 1.0)
            self.geo_out_layers["rotation"] = rotation_layer

        shs_dim = 3 if use_rgb else (sh_degree + 1) ** 2 * 3
        self.color_out_layers["shs"] = nn.Linear(in_channels, shs_dim)

    def iter_geo_params(self):
        yield from self.geo_out_layers.parameters()

    def iter_basecolor_params(self):
        yield from self.color_out_layers.parameters()

    def forward(self, geo_token: torch.Tensor, app_token: torch.Tensor, pts: torch.Tensor) -> GaussianModel:
        """geo_token, app_token: [N,C]; pts (anchors): [N,3]"""
        assert geo_token.ndim == 2, f"Expected geo_token [N,C], got {geo_token.shape}"
        assert app_token.ndim == 2, f"Expected app_token [N,C], got {app_token.shape}"
        ret = {}

        offset = self.geo_out_layers["xyz"](geo_token)
        if self.restrict_offset:
            offset = (torch.sigmoid(offset) - 0.5) * self.xyz_offset_max_step
        ret["offset"] = offset
        ret["xyz"] = pts + offset

        scaling = trunc_exp(self.geo_out_layers["scaling"](geo_token))
        if self.scale_sphere:
            scaling = torch.cat([scaling, scaling, scaling], dim=-1)
        if self.clip_scaling is not None:
            scaling = torch.clamp(scaling, min=0.0005, max=self.clip_scaling)
        ret["scaling"] = scaling

        if self.fix_opacity:
            ret["opacity"] = torch.ones_like(geo_token[..., :1])
        else:
            ret["opacity"] = torch.sigmoid(self.geo_out_layers["opacity"](geo_token))

        if self.fix_rotation:
            ret["rotation"] = matrix_to_quaternion(
                torch.eye(3, device=geo_token.device, dtype=geo_token.dtype)[None].repeat(geo_token.shape[0], 1, 1)
            )
        else:
            ret["rotation"] = F.normalize(self.geo_out_layers["rotation"](geo_token), dim=-1)

        shs = self.color_out_layers["shs"](app_token)
        if self.use_rgb:
            shs = torch.clamp(torch.sigmoid(shs), 0.0, 1.0)
        ret["shs"] = shs.view(shs.shape[0], -1, 3)

        return GaussianModel(**ret)


class PointEmbed(nn.Module):
    """Fourier-feature embedding of 3D points (as in 3DShape2VecSet): [B,N,3] -> [B,N,dim]."""

    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()
        assert hidden_dim % 6 == 0
        self.embedding_dim = hidden_dim
        k = self.embedding_dim // 6
        e = torch.pow(2, torch.arange(k)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(k), torch.zeros(k)]),
            torch.cat([torch.zeros(k), e, torch.zeros(k)]),
            torch.cat([torch.zeros(k), torch.zeros(k), e]),
        ])
        self.register_buffer("basis", e)
        self.mlp = nn.Linear(self.embedding_dim + 3, dim)
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum("bnd,de->bne", input, basis)
        return torch.cat([projections.sin(), projections.cos()], dim=2)

    def forward(self, input):
        return self.norm(self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2)))


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
class GS3DRenderer(nn.Module):
    """Builds canonical hand Gaussians from the tokens, optionally recolors them with the
    appearance head, poses them with MANO (LBS) and rasterizes them.

    Query points are the canonical vertices of the MANO mesh subdivided `subdivide_num`
    times; each vertex anchors one Gaussian.

    Main entry points
        forward(...)                    training: build, pose and render every target view
        build_gs(...)                   canonical Gaussians only (the avatar)
        render_gs(...)                  pose and render an avatar from build_gs
        get_query_points_from_params()  canonical anchors for the given MANO shape
    """

    def __init__(
        self,
        human_model_path,
        subdivide_num,
        smpl_type,
        query_dim,
        use_rgb,
        sh_degree,
        xyz_offset_max_step,
        mlp_network_config,
        clip_scaling=0.2,
        scale_sphere=False,
        skip_decoder=True,
        fix_opacity=False,
        fix_rotation=False,
        gradient_checkpointing=False,
        use_appearance_head=False,
        appearance_head_cfg=None,
        **kwargs,  # ignored (e.g. render_rgb from older configs)
    ):
        super().__init__()
        assert smpl_type == "mano"
        if not skip_decoder:
            raise NotImplementedError("Only skip_decoder=True (query points = MANO anchors) is supported.")
        self.smpl_type = smpl_type
        self.skip_decoder = skip_decoder
        self.gradient_checkpointing = gradient_checkpointing
        self.current_train_stage = "stage1"
        self.scaling_modifier = 1.0
        self.sh_degree = 0 if use_rgb else sh_degree

        # NOTE: submodule names and creation order are fixed (checkpoint keys, initialization).
        self.mano_model = MANOHandSubdivided(
            model_path=f"{human_model_path}/mano/MANO_RIGHT.pkl",
            use_pca=False,
            num_pca_comps=45,
            flat_hand_mean=True,
            subdivide_num=subdivide_num,
        )
        self.mlp_network_config = mlp_network_config
        self.mlp_net = MLP(query_dim, query_dim, **mlp_network_config) if mlp_network_config is not None else None
        self.gs_net = GSLayer(
            in_channels=query_dim,
            use_rgb=use_rgb,
            sh_degree=self.sh_degree,
            clip_scaling=clip_scaling,
            scale_sphere=scale_sphere,
            init_scaling=-5.0,
            init_density=0.1,
            restrict_offset=True,
            xyz_offset_max_step=xyz_offset_max_step,
            fix_opacity=fix_opacity,
            fix_rotation=fix_rotation,
        )
        self.appearance_head = AppearanceHead(**(appearance_head_cfg or {})) if use_appearance_head else None

    # ------------------------------------------------------------------
    # training stage
    # ------------------------------------------------------------------
    def set_train_stage(self, stage: str) -> None:
        """Renderer-level freezing: stage1 trains gs_net, stage2 its color layer and the
        appearance head. forward / build_gs / render_gs call this on every call; the model
        (ModelHAM.set_train_stage) applies its own, stricter setting on top."""
        stage = str(stage).lower()
        self.current_train_stage = stage
        for p in self.parameters():
            p.requires_grad = False
        if stage == "stage1":
            modules = [self.gs_net]
        elif stage == "stage2":
            modules = [self.gs_net.color_out_layers, self.appearance_head]
        else:
            raise ValueError(f"Unknown train stage: {stage}")
        for module in modules:
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = True

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    # ------------------------------------------------------------------
    # rasterization
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_bg_color(background_color: Optional[torch.Tensor], ref_tensor: torch.Tensor) -> torch.Tensor:
        """Background color as 3 values (black if None)."""
        if background_color is None:
            return torch.zeros(3, device=ref_tensor.device, dtype=ref_tensor.dtype)
        if background_color.ndim > 1:
            background_color = background_color.reshape(-1)
        if background_color.numel() != 3:
            raise ValueError(f"background_color must have 3 values, got shape {tuple(background_color.shape)}")
        return background_color.to(device=ref_tensor.device, dtype=ref_tensor.dtype)

    @staticmethod
    def _normalize_source_bg_color(source_bg_color):
        """Source background color as 3 values (grey levels are repeated); None stays None."""
        if source_bg_color is None or not torch.is_tensor(source_bg_color):
            return source_bg_color
        flat = source_bg_color.reshape(-1)
        if flat.numel() == 0:
            return torch.zeros(3, device=source_bg_color.device, dtype=source_bg_color.dtype)
        if flat.numel() == 1:
            return flat.repeat(3)
        if flat.numel() == 2:
            return torch.cat([flat, flat[-1:].expand(1)])
        return flat[:3]

    def forward_single_view(self, gs: GaussianModel, viewpoint_camera: Camera,
                            background_color: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Rasterize one set of posed Gaussians.

        Returns comp_rgb [H,W,3], comp_mask (alpha) [H,W,1], comp_depth [H,W,1],
        radii [N] (screen-space) and contrib [N] (per-Gaussian contribution to the image).
        """
        device = gs.xyz.device
        background_color = self._normalize_bg_color(background_color, gs.xyz)

        # screen-space means; the rasterizer writes their gradients here
        screenspace_points = torch.zeros_like(gs.xyz, dtype=gs.xyz.dtype, requires_grad=True, device=device) + 0
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
            tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
            bg=background_color.float(),
            scale_modifier=self.scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform.float(),
            sh_degree=self.sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        shs = None
        colors_precomp = None
        if self.gs_net.use_rgb:
            colors_precomp = gs.shs.squeeze(1)
        else:
            shs = gs.shs

        with torch.autocast(device_type=device.type, dtype=torch.float32):
            rendered_image, radii, rendered_depth, rendered_alpha, contrib = rasterizer(
                means3D=gs.xyz.float(),
                means2D=screenspace_points.float(),
                shs=shs.float() if shs is not None else None,
                colors_precomp=colors_precomp.float() if colors_precomp is not None else None,
                opacities=gs.opacity.float(),
                scales=gs.scaling.float(),
                rotations=gs.rotation.float(),
                cov3D_precomp=None,
            )

        return {
            "comp_rgb": rendered_image.permute(1, 2, 0),
            "comp_rgb_bg": background_color,
            "comp_mask": rendered_alpha.permute(1, 2, 0),
            "comp_depth": rendered_depth.permute(1, 2, 0),
            "radii": radii,
            "contrib": contrib,
        }

    def forward_single_batch(self, gs_list: list, c2ws: torch.Tensor, intrinsics: torch.Tensor, height: int, width: int,
                             background_color: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Render view v with gs_list[v]. c2ws, intrinsics: [V,4,4]; background_color: [V,3], [3] or None."""
        out_list = []
        for v_idx, (c2w, intrinsic) in enumerate(zip(c2ws, intrinsics)):
            if background_color is None:
                bg_v = None
            elif background_color.ndim == 1:
                bg_v = background_color if background_color.numel() == 3 else background_color.repeat(3)
            else:
                bg_v = background_color[v_idx]
                if bg_v.numel() == 1:
                    bg_v = bg_v.repeat(3)
            out_list.append(self.forward_single_view(gs_list[v_idx], Camera.from_c2w(c2w, intrinsic, height, width), bg_v))

        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        out = {k: torch.stack(v, dim=0) for k, v in out.items()}
        out["3dgs"] = gs_list
        out["gs_offset"] = torch.stack([g.offset for g in gs_list], dim=0)
        out["gs_scaling"] = torch.stack([g.scaling for g in gs_list], dim=0)
        out["gs_opacity"] = torch.stack([g.opacity for g in gs_list], dim=0)
        return out

    def render_source_view(self, gs_model: GaussianModel, source_c2w: torch.Tensor, source_intrinsic: torch.Tensor,
                           source_image: torch.Tensor, background_color: Optional[torch.Tensor] = None):
        """Render posed Gaussians in the source camera at the source image size.
        comp_* are returned as [1,C,H,W]; radii and contrib as [N]."""
        if source_c2w.ndim == 3:
            source_c2w = source_c2w[0]
        if source_intrinsic.ndim == 3:
            source_intrinsic = source_intrinsic[0]
        if source_intrinsic.shape[-2:] == (4, 4):
            source_intrinsic = source_intrinsic[:3, :3]

        H, W = source_image.shape[-2:]
        cam = Camera.from_c2w(c2w=source_c2w, intrinsic=source_intrinsic, height=H, width=W)
        out = self.forward_single_view(gs=gs_model, viewpoint_camera=cam, background_color=background_color)
        return {
            "comp_rgb": out["comp_rgb"].permute(2, 0, 1).unsqueeze(0),
            "comp_mask": out["comp_mask"].permute(2, 0, 1).unsqueeze(0),
            "comp_depth": out["comp_depth"].permute(2, 0, 1).unsqueeze(0),
            "radii": out["radii"],
            "contrib": out["contrib"],
        }

    # ------------------------------------------------------------------
    # posing
    # ------------------------------------------------------------------
    def animate_gs_model(self, gs_attr: GaussianModel, mano_data, debug=False) -> List[GaussianModel]:
        """Pose the canonical Gaussian centers with MANO LBS, one GaussianModel per view.
        Other attributes are shared. debug=True poses zero-offset, unit-opacity Gaussians."""
        device = gs_attr.xyz.device
        if debug:
            N = gs_attr.xyz.shape[0]
            gs_attr.xyz = torch.zeros_like(gs_attr.xyz)
            gs_attr.rotation = matrix_to_quaternion(torch.eye(3).float()[None].repeat(N, 1, 1)).to(device)
            gs_attr.opacity = torch.ones((N, 1), dtype=torch.float32, device=device)

        with torch.autocast(device_type=device.type, dtype=torch.float32):
            num_view = mano_data["hand_pose"].shape[0]
            mean_3d = gs_attr.xyz.unsqueeze(0).repeat(num_view, 1, 1)
            mean_3d = self.mano_model.animation_forward(
                v_cano=mean_3d,
                shape=mano_data["betas"].repeat(num_view, 1),
                rotation=mano_data["global_orient"],
                hand_pose=mano_data["hand_pose"],
                translation=mano_data["translation"],
                zero_centered_at_root_node=False,
                return_verts_cano=False,
                apply_pose_bs=False,
            )["animated"]

        return [
            GaussianModel(
                xyz=mean_3d[i],
                opacity=gs_attr.opacity,
                rotation=gs_attr.rotation,
                scaling=gs_attr.scaling,
                shs=gs_attr.shs,
                offset=gs_attr.offset,
            )
            for i in range(num_view)
        ]

    # ------------------------------------------------------------------
    # canonical Gaussians
    # ------------------------------------------------------------------
    def build_canonical_base_gs_attr(self, geo_token, app_token, query_points):
        """Stage-1 canonical Gaussians (base colors). The shared MLP is applied to both tokens."""
        if self.mlp_net is not None:
            geo_token = self.mlp_net(geo_token)
            app_token = self.mlp_net(app_token)
        canonical_base_gs = self.gs_net(geo_token=geo_token, app_token=app_token, pts=query_points)
        return canonical_base_gs, geo_token, app_token

    def apply_source_aware_appearance(self, canonical_base_gs: GaussianModel, app_token: torch.Tensor,
                                      geo_token: torch.Tensor, additional_features: dict, source_mano_data: dict):
        """Recolor the canonical Gaussians from the source image with the appearance head.

        The base Gaussians are posed with the source MANO parameters and rendered in the
        source camera; the head uses that render (colors, radii, contributions) together
        with the source image. Only the colors change.
        """
        if self.appearance_head is None:
            return canonical_base_gs
        if not self.gs_net.use_rgb:
            raise NotImplementedError("Appearance head currently assumes use_rgb=True")
        if additional_features is None:
            raise ValueError("additional_features is required for appearance refinement")

        source_image = additional_features.get("source_image", None)
        source_mask = additional_features.get("source_mask", None)
        source_intrinsic = additional_features.get("source_intrinsic", None)
        source_c2w = additional_features.get("source_c2w", None)
        source_bg_color = self._normalize_source_bg_color(additional_features.get("source_bg_color", None))

        for name, value in (("source_image", source_image), ("source_mask", source_mask),
                            ("source_intrinsic", source_intrinsic), ("source_c2w", source_c2w),
                            ("source_mano_data", source_mano_data)):
            if value is None:
                raise ValueError(f"{name} is required")

        if source_image.ndim == 3:
            source_image = source_image.unsqueeze(0)
        if source_mask.ndim == 3:
            source_mask = source_mask.unsqueeze(1)
        app_token = app_token.unsqueeze(0) if app_token.ndim == 2 else app_token

        source_posed_base_gs_list = self.animate_gs_model(canonical_base_gs, source_mano_data, debug=False)
        if len(source_posed_base_gs_list) == 0:
            raise RuntimeError("source_posed_base_gs_list is empty")
        source_posed_base_gs = source_posed_base_gs_list[0]

        source_render_out = self.render_source_view(
            gs_model=source_posed_base_gs,
            source_c2w=source_c2w,
            source_intrinsic=source_intrinsic,
            source_image=source_image,
            background_color=source_bg_color,
        )

        final_rgb = self.appearance_head({
            "source_image": source_image,
            "source_mask": source_mask,
            "posed_base_gs": source_posed_base_gs,
            "source_render_rgb": source_render_out["comp_rgb"],
            "radii": source_render_out["radii"].unsqueeze(0),
            "contrib": source_render_out["contrib"].unsqueeze(0),
            "source_intrinsic": source_intrinsic,
            "source_c2w": source_c2w,
            "app_token": app_token,
            # NOTE: only the training forward provides "source_dino_image_feats_bchw". build_gs
            # callers (evaluation, inference) pass "image_feats_bchw" instead, so there the
            # appearance head receives None here and falls back to zero features.
            "source_dino_feat_bcwh": additional_features.get("source_dino_image_feats_bchw", None),
        })
        final_rgb = torch.clamp(final_rgb.squeeze(0), 0.0, 1.0)

        return GaussianModel(
            xyz=canonical_base_gs.xyz,
            opacity=canonical_base_gs.opacity,
            rotation=canonical_base_gs.rotation,
            scaling=canonical_base_gs.scaling,
            shs=final_rgb.unsqueeze(1),
            offset=canonical_base_gs.offset,
        )

    def _missing_app_inputs(self, additional_features: Optional[dict], source_mano_data: Optional[dict],
                            query_points: Optional[torch.Tensor]) -> List[str]:
        missing = []
        if additional_features is None:
            missing.append("additional_features")
        else:
            missing += [k for k in APPEARANCE_INPUT_KEYS if additional_features.get(k, None) is None]
        if source_mano_data is None:
            missing.append("source_mano_data")
        if query_points is None:
            missing.append("query_points")
        return missing

    def forward_gs_attr(self, geo_token, app_token, query_points, mano_data=None, debug=False,
                        additional_features=None, source_mano_data=None) -> GaussianModel:
        """Canonical Gaussians of one sample; recolored by the appearance head in stage2
        when the source inputs are available (required in training)."""
        canonical_base_gs, geo_token, app_token = self.build_canonical_base_gs_attr(
            geo_token=geo_token, app_token=app_token, query_points=query_points,
        )

        use_app = self.appearance_head is not None and self.current_train_stage == "stage2"
        missing = self._missing_app_inputs(additional_features, source_mano_data, query_points) if use_app else []

        if use_app and not missing:
            return self.apply_source_aware_appearance(
                canonical_base_gs=canonical_base_gs,
                app_token=app_token,
                geo_token=geo_token,
                additional_features=additional_features,
                source_mano_data=source_mano_data,
            )
        if use_app and self.training:
            raise ValueError(f"Missing inputs for stage2 appearance refinement: {missing}")
        return canonical_base_gs

    # ------------------------------------------------------------------
    # batching helpers
    # ------------------------------------------------------------------
    def get_query_points(self, mano_data, device):
        """Canonical anchors (subdivided MANO vertices) for the given shape, without gradients."""
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float32):
                positions = self.mano_model.get_cano_verts(shape=mano_data["betas"])
        return positions, mano_data

    def get_query_points_from_params(self, mano_params: dict, device):
        return self.get_query_points(mano_params, device)

    def query_latent_feat(self, positions, mano_data, latent_feat):
        assert positions is not None
        return latent_feat, positions, mano_data

    def get_single_batch_mano_data(self, mano_data: dict, bidx: int) -> dict:
        out = {"betas": mano_data["betas"][bidx:bidx + 1]}
        for k in ("global_orient", "hand_pose", "translation"):
            if k not in mano_data:
                continue
            v = mano_data[k]
            if torch.is_tensor(v) and v.dim() >= 3:
                out[k] = v[bidx]
            elif torch.is_tensor(v) and v.dim() >= 2:
                out[k] = v[bidx:bidx + 1]
            else:
                out[k] = v[bidx]
        return out

    def get_single_batch_additional_features(self, additional_features: Optional[dict], bidx: int) -> Optional[dict]:
        """Slice every batched tensor to sample `bidx` (tensors with batch size 1 are shared)."""
        if additional_features is None:
            return None
        out = {}
        for k, v in additional_features.items():
            if torch.is_tensor(v) and v.shape[0] != 1 and v.shape[0] > bidx:
                out[k] = v[bidx:bidx + 1]
            else:
                out[k] = v
        return out

    def get_single_view_mano_data(self, mano_data: dict, vidx: int) -> dict:
        out = {"betas": mano_data["betas"]}
        for k in ("global_orient", "hand_pose", "translation"):
            if k not in mano_data:
                continue
            v = mano_data[k]
            if torch.is_tensor(v) and v.dim() >= 2 and v.shape[0] == 1:
                out[k] = v[:, vidx:vidx + 1]
            elif torch.is_tensor(v) and v.dim() >= 1:
                out[k] = v[vidx:vidx + 1]
            else:
                out[k] = v
        return out

    # ------------------------------------------------------------------
    # build / pose / render
    # ------------------------------------------------------------------
    def forward_gs(self, gs_hidden_features, query_points, mano_data, additional_features=None, debug: bool = False, **kwargs):
        """Canonical Gaussians for every sample in the batch.

        Returns (gs_model_list, query_points, mano_data, gs_hidden_features,
                 rgb list, opacity list, scaling list, offset list).
        """
        batch_size = gs_hidden_features["geo"].shape[0]
        gs_hidden_features, query_points, mano_data = self.query_latent_feat(query_points, mano_data, gs_hidden_features)
        source_mano_data = additional_features.get("source_mano_data", None) if additional_features is not None else None

        gs_model_list, all_query_points = [], []
        gs_rgb_list, gs_opacity_list, gs_scaling_list, gs_offset_list = [], [], [], []
        for b in range(batch_size):
            all_query_points.append(query_points[b:b + 1])
            gs_model = self.forward_gs_attr(
                geo_token=gs_hidden_features["geo"][b],
                app_token=gs_hidden_features["app"][b],
                query_points=query_points[b],
                mano_data=None,
                debug=debug,
                additional_features=self.get_single_batch_additional_features(additional_features, b),
                source_mano_data=None if source_mano_data is None else self.get_single_batch_mano_data(source_mano_data, b),
            )
            gs_model_list.append(gs_model)
            gs_rgb_list.append(gs_model.shs.squeeze(1))
            gs_opacity_list.append(gs_model.opacity)
            gs_scaling_list.append(gs_model.scaling)
            gs_offset_list.append(gs_model.offset)

        query_points = torch.cat(all_query_points, dim=0)
        return (gs_model_list, query_points, mano_data, gs_hidden_features,
                gs_rgb_list, gs_opacity_list, gs_scaling_list, gs_offset_list)

    def forward_animate_gs(self, gs_model_list, query_points, mano_data, c2w, intrinsic, height, width,
                           background_color, debug=False) -> Dict[str, torch.Tensor]:
        """Pose each sample's Gaussians for all its views and render them.
        comp_rgb / comp_mask / comp_depth are returned as [B,V,C,H,W]."""
        out_list = []
        for b, gs_model in enumerate(gs_model_list):
            posed_gs_list = self.animate_gs_model(gs_model, self.get_single_batch_mano_data(mano_data, b), debug=debug)
            assert len(posed_gs_list) == c2w.shape[1]
            out_list.append(self.forward_single_batch(
                posed_gs_list, c2w[b], intrinsic[b], height, width,
                background_color[b] if background_color is not None else None,
            ))

        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        for k, v in out.items():
            out[k] = torch.stack(v, dim=0) if isinstance(v[0], torch.Tensor) else v
        for key in ("comp_rgb", "comp_mask", "comp_depth"):
            out[key] = rearrange(out[key], "b v h w c -> b v c h w")
        return out

    def forward(self, gs_hidden_features, query_points, mano_data, c2w, intrinsic, height, width,
                additional_features=None, background_color=None, debug: bool = False, train_stage=None, **kwargs):
        """Training path: build (and recolor) the avatars, then pose and render every target view."""
        if train_stage is not None:
            self.set_train_stage(train_stage)

        gs_model_list, query_points, mano_data, _, gs_rgb_list, gs_opacity_list, gs_scaling_list, gs_offset_list = \
            self.forward_gs(gs_hidden_features, query_points, mano_data=mano_data,
                            additional_features=additional_features, debug=debug)

        out = self.forward_animate_gs(gs_model_list, query_points, mano_data, c2w, intrinsic, height, width,
                                      background_color, debug)

        # canonical attributes [B,N,...] (replace the per-view copies from forward_single_batch)
        if gs_rgb_list:
            out["gs_rgb"] = torch.stack(gs_rgb_list, dim=0)
        if gs_opacity_list:
            out["gs_opacity"] = torch.stack(gs_opacity_list, dim=0)
        if gs_scaling_list:
            out["gs_scaling"] = torch.stack(gs_scaling_list, dim=0)
        if gs_offset_list:
            out["gs_offset"] = torch.stack(gs_offset_list, dim=0)
        return out

    def build_gs(self, gs_hidden_features, query_points, mano_params, additional_features=None, debug=False,
                 train_stage=None):
        """Canonical Gaussians (the avatar); see forward_gs for the returned tuple."""
        if train_stage is not None:
            self.set_train_stage(train_stage)
        return self.forward_gs(gs_hidden_features=gs_hidden_features, query_points=query_points,
                               mano_data=mano_params, additional_features=additional_features, debug=debug)

    def get_view_mano(self, mano_state: dict, view_idx: int) -> dict:
        return self.get_single_view_mano_data(mano_state, view_idx)

    def render_gs(self, gs_models, query_points, view_mano, c2w, intrinsic, height, width, background_color,
                  debug=False, train_stage=None):
        """Pose the avatars from build_gs with `view_mano` and render them."""
        if train_stage is not None:
            self.set_train_stage(train_stage)
        return self.forward_animate_gs(gs_model_list=gs_models, query_points=query_points, mano_data=view_mano,
                                       c2w=c2w, intrinsic=intrinsic, height=height, width=width,
                                       background_color=background_color, debug=debug)

    @torch.no_grad()
    def get_posed_gaussian_centers(self, gs_models, query_points, view_mano, debug: bool = False) -> torch.Tensor:
        """Posed Gaussian centers: [B,V,N,3] for a list of avatars, else [V,N,3]."""
        if isinstance(gs_models, list):
            return torch.stack([
                torch.stack([g.xyz for g in self.animate_gs_model(gs_model, self.get_single_batch_mano_data(view_mano, b), debug=debug)], dim=0)
                for b, gs_model in enumerate(gs_models)
            ], dim=0)
        return torch.stack([g.xyz for g in self.animate_gs_model(gs_models, view_mano, debug=debug)], dim=0)