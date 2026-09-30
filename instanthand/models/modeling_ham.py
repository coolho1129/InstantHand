# Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors.
# Copyright (c) 2026 The InstantHand Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from ModelLAM in LAM (https://github.com/aigc3d/LAM, lam/models/modeling_lam.py),
# licensed under the Apache License 2.0. Modified by the InstantHand authors: MANO-anchored
# hand queries, separate geometry / appearance tokens, two-stage training and the
# appearance head.
"""InstantHand model: one hand image -> animatable 3D Gaussian hand avatar."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange

from .transformer import TransformerDecoder
from instanthand.models.rendering.gs_renderer import GS3DRenderer, PointEmbed


class CrossAttnBlock(nn.Module):
    """Cross-attention from the geometry tokens (queries) to the image features
    (keys / values). Produces the appearance tokens."""

    def __init__(
        self,
        inner_dim: int,
        cond_dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        attn_bias: bool = False,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=inner_dim,
            num_heads=num_heads,
            kdim=cond_dim,
            vdim=cond_dim,
            dropout=attn_drop,
            bias=attn_bias,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.cross_attn(x, cond, cond, need_weights=False)[0]


class ModelHAM(nn.Module):
    """Feed-forward 3D Gaussian hand avatar from a single image.

    image -> DINOv3 encoder -> image features
    posed MANO vertices (query points) -> PointEmbed -> transformer (cross-attends to the
        image features) -> geometry tokens
    geometry tokens -> CrossAttnBlock (cross-attends to the image features) -> appearance tokens
    tokens -> GS3DRenderer: canonical Gaussians on the subdivided MANO mesh, optional
        appearance head (source-image appearance transfer), LBS posing and rasterization

    Training stages (set_train_stage):
        stage1: encoder fusion head, transformer, appearance-token block and Gaussian decoder
                are trained; DINOv3 backbone and appearance head are frozen.
        stage2: only the appearance head is trained; tokens and image features are detached.
        The DINOv3 backbone is frozen in both stages.
    """

    def __init__(
        self,
        transformer_dim: int,
        transformer_layers: int,
        transformer_heads: int,
        transformer_type: str = "sd3_cond",
        tf_grad_ckpt: bool = False,
        encoder_grad_ckpt: bool = False,
        encoder_freeze: bool = True,
        encoder_type: str = "dinov3_fusion",
        encoder_model_name: str = "dinov3_vitl16",
        encoder_feat_dim: int = 1024,
        encoder_weights_path: str | None = None,
        pcl_dim: int = 512,
        mano_model_path: str = "./model_zoo/hand_parametric_models",
        mano_subdivide_num: int = 0,
        gs_use_rgb: bool = False,
        gs_sh: int = 3,
        gs_mlp_network_config=None,
        gs_xyz_offset_max_step: float = 1.8 / 32,
        gs_clip_scaling: float = 0.2,
        fix_opacity: bool = False,
        fix_rotation: bool = False,
        scale_sphere: bool = False,
        render_rgb: bool = True,
        use_appearance_head: bool = False,
        appearance_head_cfg: dict | None = None,
        latent_query_points_type: str = "e2e_mano",
        **kwargs,  # ignored (older configs also pass e.g. num_pcl)
    ):
        super().__init__()
        if not str(latent_query_points_type).startswith("e2e_mano"):
            raise NotImplementedError(f"Unsupported latent_query_points_type: {latent_query_points_type}")
        self.latent_query_points_type = latent_query_points_type

        self.gradient_checkpointing = tf_grad_ckpt
        self.encoder_gradient_checkpointing = encoder_grad_ckpt
        self.encoder_feat_dim = encoder_feat_dim
        self.current_train_stage = "stage1"

        # NOTE: submodule names and creation order are fixed. Names must match the released
        # checkpoint, and the order determines random initialization in training.
        encoder_kwargs = {"model_name": encoder_model_name, "freeze": encoder_freeze, "encoder_feat_dim": encoder_feat_dim}
        if encoder_weights_path:
            encoder_kwargs["weights_path"] = encoder_weights_path
        self.encoder = self._encoder_fn(encoder_type)(**encoder_kwargs)

        self.pcl_embed = PointEmbed(dim=pcl_dim)

        self.transformer = TransformerDecoder(
            block_type=transformer_type,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            inner_dim=transformer_dim,
            cond_dim=encoder_feat_dim,
            mod_dim=None,
            gradient_checkpointing=self.gradient_checkpointing,
        )

        self.app_token_block = CrossAttnBlock(
            inner_dim=transformer_dim,
            cond_dim=encoder_feat_dim,
            num_heads=transformer_heads,
        )

        self.renderer = GS3DRenderer(
            human_model_path=mano_model_path,
            subdivide_num=mano_subdivide_num,
            smpl_type="mano",
            query_dim=transformer_dim,
            use_rgb=gs_use_rgb,
            sh_degree=gs_sh,
            mlp_network_config=gs_mlp_network_config,
            xyz_offset_max_step=gs_xyz_offset_max_step,
            clip_scaling=gs_clip_scaling,
            scale_sphere=scale_sphere,
            skip_decoder=True,
            fix_opacity=fix_opacity,
            fix_rotation=fix_rotation,
            gradient_checkpointing=self.gradient_checkpointing,
            render_rgb=render_rgb,
            use_appearance_head=use_appearance_head,
            appearance_head_cfg=appearance_head_cfg,
        )
        if hasattr(self.renderer, "set_train_stage"):
            self.renderer.set_train_stage(self.current_train_stage)

    @staticmethod
    def _encoder_fn(encoder_type: str):
        if str(encoder_type).lower() in {"dinov3", "dinov3_fusion"}:
            from .encoders.dinov3_fusion_wrapper import Dinov3FusionWrapper

            return Dinov3FusionWrapper
        raise ValueError(f"Unsupported encoder_type for ModelHAM: {encoder_type}")

    def get_last_layer(self):
        return self.renderer.gs_net.color_out_layers["shs"].weight

    # ------------------------------------------------------------------
    # training stage
    # ------------------------------------------------------------------
    @staticmethod
    def _set_requires_grad(module: nn.Module | None, flag: bool) -> None:
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = flag

    def _encoder_backbone_module(self) -> nn.Module | None:
        for name in ("backbone", "model"):
            if hasattr(self.encoder, name):
                return getattr(self.encoder, name)
        return None

    def _encoder_head_module(self) -> nn.Module | None:
        for name in ("head", "fusion_head"):
            if hasattr(self.encoder, name):
                return getattr(self.encoder, name)
        return None

    def _stage_trainable_modules(self, stage: str) -> List[nn.Module]:
        if stage == "stage1":
            modules = [self._encoder_head_module(), self.transformer, self.app_token_block, self.renderer.gs_net]
        elif stage == "stage2":
            modules = [getattr(self.renderer, "appearance_head", None)]
        else:
            raise ValueError(f"Unknown train stage: {stage}")
        return [m for m in modules if m is not None]

    def set_train_stage(self, stage: str) -> None:
        """Freeze everything, unfreeze the modules trained in `stage`, keep the backbone frozen."""
        stage = str(stage).lower()
        self.current_train_stage = stage
        if hasattr(self.renderer, "set_train_stage"):
            self.renderer.set_train_stage(stage)

        for p in self.parameters():
            p.requires_grad = False
        for module in self._stage_trainable_modules(stage):
            self._set_requires_grad(module, True)
        self._set_requires_grad(self._encoder_backbone_module(), False)

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def print_trainable_parameters(self) -> None:
        trainable = [n for n, p in self.named_parameters() if p.requires_grad]
        frozen = [n for n, p in self.named_parameters() if not p.requires_grad]
        print("=" * 80)
        print(f"[Train stage] {self.current_train_stage}")
        print("[Trainable parameters]")
        for n in trainable:
            print("  ", n)
        print("-" * 80)
        print(f"Trainable count: {len(trainable)}")
        print(f"Frozen count   : {len(frozen)}")
        print("=" * 80)

    # ------------------------------------------------------------------
    # tokens
    # ------------------------------------------------------------------
    def forward_encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """image [B,3,H,W] -> image features [B,T,C]"""
        if self.training and self.encoder_gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(self.encoder, image, use_reentrant=False)
        return self.encoder(image)

    def forward_transformer(self, image_feats: torch.Tensor, camera_embeddings, query_points: torch.Tensor) -> torch.Tensor:
        """Query points [B,N,3] -> geometry tokens [B,N,D], cross-attending to the image features."""
        x = self.pcl_embed(query_points).to(image_feats.dtype)
        return self.transformer(x, cond=image_feats, mod=camera_embeddings)

    def forward_latent_points(self, image: torch.Tensor, camera=None, query_points: Optional[torch.Tensor] = None):
        """Returns ({"geo": geometry tokens, "app": appearance tokens}, image features).

        `camera` is not used; the argument is kept for existing callers.
        In stage2 all outputs are detached, so only the appearance head receives gradients.
        """
        image_feats = self.forward_encode_image(image)
        assert image_feats.shape[-1] == self.encoder_feat_dim, (
            f"Feature dim mismatch: {image_feats.shape[-1]} vs {self.encoder_feat_dim}"
        )

        geo_token = self.forward_transformer(image_feats=image_feats, camera_embeddings=None, query_points=query_points)
        app_token = self.app_token_block(geo_token, image_feats)

        if self.current_train_stage == "stage2":
            geo_token = geo_token.detach()
            app_token = app_token.detach()
            image_feats = image_feats.detach()

        return {"geo": geo_token, "app": app_token}, image_feats

    # ------------------------------------------------------------------
    # input helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _render_size(render_full_resolutions: Optional[torch.Tensor], render_intrs: torch.Tensor) -> Tuple[int, int]:
        """(H, W) from render_full_resolutions ([B,V,2] or [B,2]); else twice the principal point."""
        if render_full_resolutions is None:
            return int(render_intrs[0, 0, 1, 2] * 2), int(render_intrs[0, 0, 0, 2] * 2)
        if render_full_resolutions.ndim == 3:
            return int(render_full_resolutions[0, 0, 0].item()), int(render_full_resolutions[0, 0, 1].item())
        if render_full_resolutions.ndim == 2:
            return int(render_full_resolutions[0, 0].item()), int(render_full_resolutions[0, 1].item())
        raise ValueError(f"Unexpected render_full_resolutions shape: {render_full_resolutions.shape}")

    @staticmethod
    def _feats_to_bchw(image_feats: torch.Tensor) -> torch.Tensor:
        """Image tokens [B,T,C] on a square grid -> [B,C,H,W]."""
        side = int(math.sqrt(image_feats.shape[1]))
        return rearrange(image_feats, "b (h w) c -> b c h w", h=side)

    @staticmethod
    def _first_view(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return x[:, 0] if x is not None and x.ndim == 5 else x

    def _build_source_mano_data(self, source_mano_params=None, data=None) -> Optional[Dict[str, torch.Tensor]]:
        """MANO parameters of the source image: `source_mano_params` if given, else the
        source_* entries of `data` (None if incomplete)."""
        if source_mano_params is not None:
            return source_mano_params
        if data is None:
            return None
        if not all(k in data for k in ("source_global_orient", "source_hand_pose", "source_translation")):
            return None

        betas = data["source_betas"] if "source_betas" in data else data.get("betas")
        if betas is None:
            return None

        def _with_view_dim(x):
            return x.unsqueeze(1) if x.ndim == 2 else x

        return {
            "betas": betas,
            "global_orient": _with_view_dim(data["source_global_orient"]),
            "hand_pose": _with_view_dim(data["source_hand_pose"]),
            "translation": _with_view_dim(data["source_translation"]),
        }

    def _extract_source_side_inputs(self, data=None):
        """(source mask, source depth map, source background color) from `data`, first view."""
        source_mask = source_depth_map = source_bg_color = None
        if data is not None:
            if "source_mask" in data:
                source_mask = self._first_view(data["source_mask"])
            if "source_silhouette_mask" in data and source_mask is None:
                source_mask = self._first_view(data["source_silhouette_mask"])
            if "source_depth_map" in data:
                source_depth_map = self._first_view(data["source_depth_map"])
            if "source_bg_color" in data:
                source_bg_color = data["source_bg_color"]
        return source_mask, source_depth_map, source_bg_color

    # ------------------------------------------------------------------
    # forward / inference
    # ------------------------------------------------------------------
    def forward(
        self,
        image,
        source_c2ws,
        source_intrs,
        render_c2ws,
        render_intrs,
        render_bg_colors,
        mano_params,
        source_mano_params=None,
        render_images=None,  # not used; kept for existing callers
        data=None,
        render_full_resolutions=None,
    ) -> Dict[str, Any]:
        """Training forward: build the avatar from image[:, 0] and render all target views."""
        B = image.shape[0]
        assert B == render_c2ws.shape[0]
        assert B == render_bg_colors.shape[0]
        assert B == mano_params["betas"].shape[0]
        assert mano_params["betas"].ndim == 2

        render_h, render_w = self._render_size(render_full_resolutions, render_intrs)
        query_points, mano_params = self.renderer.get_query_points_from_params(mano_params=mano_params, device=image.device)

        token_dict, image_feats = self.forward_latent_points(image=image[:, 0], camera=None, query_points=query_points)
        image_feats_bchw = self._feats_to_bchw(image_feats)

        source_mask, source_depth_map, source_bg_color = self._extract_source_side_inputs(data=data)
        source_mano_data = self._build_source_mano_data(source_mano_params=source_mano_params, data=data)

        render_results = self.renderer(
            gs_hidden_features=token_dict,
            query_points=query_points,
            mano_data=mano_params,
            c2w=render_c2ws,
            intrinsic=render_intrs,
            height=render_h,
            width=render_w,
            background_color=render_bg_colors,
            additional_features={
                "source_dino_image_feats": image_feats,
                "source_dino_image_feats_bchw": image_feats_bchw,
                "source_image": image[:, 0],
                "source_mask": source_mask,
                "source_intrinsic": source_intrs[:, 0] if source_intrs.ndim == 4 else source_intrs,
                "source_c2w": source_c2ws[:, 0] if source_c2ws.ndim == 4 else source_c2ws,
                "source_depth_map": source_depth_map,
                "source_bg_color": source_bg_color,
                "source_mano_data": source_mano_data,
            },
            train_stage=self.current_train_stage,
        )
        return {"latent_points": token_dict, **render_results}

    @torch.no_grad()
    def infer_single_view(
        self,
        image,
        source_c2ws,
        source_intrs,
        render_c2ws,
        render_intrs,
        render_bg_colors,
        mano_params,
        render_full_resolutions=None,
        data=None,
    ) -> Dict[str, Any]:
        """Build the avatar from image[:, 0] (batch size 1) and render each target view.
        comp_rgb / comp_mask / comp_depth are returned as [V,H,W,C]."""
        assert image.shape[0] == 1
        assert mano_params["betas"].ndim == 2

        render_h, render_w = self._render_size(render_full_resolutions, render_intrs)
        num_views = render_c2ws.shape[1]
        query_points, mano_params = self.renderer.get_query_points_from_params(mano_params=mano_params, device=image.device)

        token_dict, image_feats = self.forward_latent_points(image=image[:, 0], camera=None, query_points=query_points)
        image_feats_bchw = self._feats_to_bchw(image_feats)

        source_mask, source_depth_map, source_bg_color = self._extract_source_side_inputs(data=data)
        source_mano_data = self._build_source_mano_data(data=data)
        if source_mano_data is None:
            source_mano_data = self._build_source_mano_data(source_mano_params=mano_params)

        gs_models, query_points, mano_state, *_ = self.renderer.build_gs(
            gs_hidden_features=token_dict,
            query_points=query_points,
            mano_params=mano_params,
            additional_features={
                "image_feats": image_feats,
                "image": image[:, 0],
                "image_feats_bchw": image_feats_bchw,
                "source_image": image[:, 0],
                "source_mask": source_mask,
                "source_intrinsic": source_intrs[:, 0] if source_intrs.ndim == 4 else source_intrs,
                "source_c2w": source_c2ws[:, 0] if source_c2ws.ndim == 4 else source_c2ws,
                "source_depth_map": source_depth_map,
                "source_bg_color": source_bg_color,
                "source_mano_data": source_mano_data,
            },
            train_stage=self.current_train_stage,
        )

        out = defaultdict(list)
        for view_idx in range(num_views):
            render_res = self.renderer.render_gs(
                gs_models=gs_models,
                query_points=query_points,
                view_mano=self.renderer.get_view_mano(mano_state, view_idx),
                c2w=render_c2ws[:, view_idx:view_idx + 1],
                intrinsic=render_intrs[:, view_idx:view_idx + 1],
                height=render_h,
                width=render_w,
                background_color=render_bg_colors[:, view_idx:view_idx + 1],
                train_stage=self.current_train_stage,
            )
            for k, v in render_res.items():
                out[k].append(v)

        for k, v in out.items():
            if isinstance(v[0], torch.Tensor):
                out[k] = torch.concat(v, dim=1)
                if k in ("comp_rgb", "comp_mask", "comp_depth"):
                    out[k] = out[k][0].permute(0, 2, 3, 1)
            else:
                out[k] = v

        out["cano_gs_models"] = gs_models
        return out