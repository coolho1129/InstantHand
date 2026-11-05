# Copyright (c) 2026 The InstantHand Authors.
# SPDX-License-Identifier: Apache-2.0
"""InstantHand demo: one hand image -> animatable 3D Gaussian hand avatar -> videos.

    python demo.py --ids all                                  # every example in demo/input
    python demo.py --image hand.png --mask hand_mask.png --wilor_json hand.json
    python demo.py --ids all --seam_blend                      # optional seam blending (3 mm)
    python demo.py --ids all --k4                              # optional k=4 densification

Inputs per image: the image, a hand mask (or an RGBA image whose alpha is the hand mask) and
the WiLoR estimate of the hand (MANO pose, 2D keypoints, camera) as a json file; see README.
Outputs in <out_dir>/<name>/:
    input_rgba.png              the composed input
    source.png                  the network input (keypoint-centered crop)
    self_render.png             the avatar rendered from the input camera
    canonical_360_{yaw,pitch}.mp4   turntable of the avatar in the rest pose
    pose_anim_view_*.mp4        the avatar driven through the poses of a pose bank
    camera.json                 the render cameras

Resolution: the network was trained on 256x256 crops only, so the network input and every
output image and video are 256x256. There is no resolution option; preprocess.image_res and
videos.res in older configs are ignored.

The video cameras do not depend on the input photo: fixed field of view, looking at the hand
along its anatomical axes (fingers up), starting from the side of the hand that the input
photo shows, at the distance where the hand fills a fixed part of the frame over the whole
video. Only that side and the hand's own size change between subjects.

Optional seam blending (--seam_blend or postprocess.seam_blend: true; not used in the paper):
where Gaussians colored from the input meet Gaussians completed from the prior, a color step can
show as a line in novel views. The step is replaced by a smooth ramp over +-postprocess.seam_blend_mm
(3 mm) around that boundary; geometry and all other colors are unchanged.

Optional k=4 densification (--k4 or densify.k4: true; Appendix, Table S10): the trained k=3
avatar is interpolated onto the anchors of the next MANO subdivision level (4x as many
Gaussians) with all scales multiplied by densify.scale (0.6), then skinned and rendered at k=4.
It reduces the striations at strongly flexed knuckles; the network itself only sees k=3 inputs.
Needs a second copy of the network (k=4 renderer) and more GPU memory while the avatar is built.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file
from scipy.spatial import cKDTree
from tqdm import tqdm

from instanthand.datasets.multi_hand import MultiHandDataset, MultiHandDatasetConfig
from instanthand.models.modeling_ham import ModelHAM
from instanthand.models.checkpoint import load_checkpoint, resolve_encoder_weights

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

# The network was trained on 256x256 crops only: the input crop and every render use this size.
RES = 256


# =============================================================================================
# input: image (+ mask) + WiLoR json -> model inputs (the same crop as in training)
# =============================================================================================
def compose_rgba(image_path: str, mask_path: Optional[str], out_path: str, border_feather: int = 0) -> str:
    """RGBA with the hand mask as alpha. A mask touching the image border is faded out over
    `border_feather` pixels there, so that the crop padding does not leave a hard edge. RGB is
    cleared where alpha is 0, so no background color can leak in through the crop."""
    img = Image.open(image_path)
    if mask_path:
        rgba = np.asarray(img.convert("RGBA")).copy()
        mask = Image.open(mask_path).convert("L")
        if mask.size != img.size:
            mask = mask.resize(img.size, Image.NEAREST)
        rgba[..., 3] = np.asarray(mask)
    else:
        if img.mode != "RGBA":
            raise ValueError(f"{image_path}: no --mask given and the image has no alpha channel")
        rgba = np.asarray(img).copy()
        if rgba[..., 3].min() == 255:
            print(f"[input] warning: the alpha of {image_path} is all 255 (no hand segmentation?)")
    a = rgba[..., 3]
    if border_feather > 0 and (a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any()):
        h, w = a.shape
        yy, xx = np.mgrid[0:h, 0:w]
        d = np.minimum.reduce([yy, xx, h - 1 - yy, w - 1 - xx]).astype(np.float32)
        rgba[..., 3] = (a.astype(np.float32) * np.clip(d / float(border_feather), 0.0, 1.0)).astype(np.uint8)
    rgba[rgba[..., 3] == 0, :3] = 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(out_path)
    return out_path


class FramePreprocessor:
    """The per-frame preprocessing of the training dataset (MultiHandDataset) for one image:
    WiLoR json parsing, keypoint-centered square crop (RES x RES), left-hand mirroring,
    background compositing, crop intrinsics with the principal point moved to the crop center."""

    _load_from_json = MultiHandDataset._load_from_json
    _square_bbox_from_kpts = MultiHandDataset._square_bbox_from_kpts
    _crop_square_with_pad = staticmethod(MultiHandDataset._crop_square_with_pad)
    _process_frame_keypoint_centered = MultiHandDataset._process_frame_keypoint_centered

    def __init__(self, kp_enlarge: float = 1.3, alpha_thr: float = 0.5):
        self.cfg = MultiHandDatasetConfig(kp_enlarge=kp_enlarge, alpha_thr=alpha_thr)

    def __call__(self, rgba_path: str, wilor_json: str, multiply: int, bg_value: float,
                 device: torch.device) -> Dict[str, Any]:
        wilor = self._load_from_json(wilor_json)
        if wilor["focal_full"] is None:
            raise KeyError(f"{wilor_json}: no focal_length")
        rgb, mask, _, intr, cam_t, _ = self._process_frame_keypoint_centered(
            rgba_path, wilor, RES, multiply, bg_value)
        eye = torch.eye(4, dtype=torch.float32, device=device).view(1, 1, 4, 4)
        intr = intr.to(device).view(1, 1, 4, 4)
        return {
            "image": rgb.to(device)[:, None],                                         # [1,1,3,H,W]
            "mano_params": {
                "betas": wilor["betas"].to(device).view(1, 10),
                "global_orient": wilor["global_orient"].to(device).view(1, 1, 3),
                "hand_pose": wilor["hand_pose"].to(device).view(1, 1, 45),
                "translation": cam_t.to(device).view(1, 1, 3),
            },
            "source_c2ws": eye, "source_intrs": intr,
            "render_c2ws": eye.clone(), "render_intrs": intr.clone(),
            "render_bg_colors": torch.full((1, 3), float(bg_value), device=device),
            "data": {"source_mask": mask.to(device)},                                 # [1,1,H,W]
        }


# =============================================================================================
# model
# =============================================================================================
def load_model(model_cfg: Dict[str, Any], ckpt_dir: str, device: torch.device) -> ModelHAM:
    if not model_cfg or "REPLACE_WITH_EXISTING_MODEL_BLOCK" in model_cfg:
        raise SystemExit("configs/demo.yaml: fill in the `model:` block (the same as configs/eval_hanco.yaml)")
    ckpt = Path(ckpt_dir) / "model.safetensors"
    model = ModelHAM(**resolve_encoder_weights(model_cfg, ckpt))
    load_checkpoint(model, ckpt, device)
    model = model.to(device)
    model.set_train_stage("stage2")
    model.eval()
    return model


@torch.no_grad()
def build_avatar(model: ModelHAM, inputs: Dict[str, Any], device: torch.device):
    """One forward pass: input image -> canonical Gaussian avatar (as in the evaluation scripts)."""
    image = inputs["image"][:, 0]
    mano = inputs["mano_params"]
    query_points, mano_params = model.renderer.get_query_points_from_params(mano_params=mano, device=device)
    latent_points, image_feats = model.forward_latent_points(image=image, camera=None, query_points=query_points)
    side = int(math.sqrt(image_feats.shape[1]))
    gs_models, query_points, *_ = model.renderer.build_gs(
        gs_hidden_features=latent_points,
        query_points=query_points,
        mano_params=mano_params,
        additional_features={
            "image_feats": image_feats,
            "image": image,
            "image_feats_bchw": rearrange(image_feats, "b (h w) c -> b c h w", h=side),
            "source_image": image,
            "source_mask": inputs["data"]["source_mask"],
            "source_intrinsic": inputs["source_intrs"][:, 0],
            "source_c2w": inputs["source_c2ws"][:, 0],
            "source_depth_map": None,
            "source_mano_data": {k: mano[k] for k in ("betas", "global_orient", "hand_pose", "translation")},
        },
        train_stage=model.current_train_stage,
    )
    return gs_models, query_points


@torch.no_grad()
def render(model: ModelHAM, avatar, inputs: Dict[str, Any], view_mano: Dict[str, torch.Tensor],
           c2w: Optional[torch.Tensor], K: Optional[torch.Tensor], H: int, W: int) -> np.ndarray:
    """RGB uint8 [H,W,3] of the avatar posed with `view_mano`, seen from c2w / K (4x4; None:
    the input camera)."""
    gs_models, query_points = avatar
    c2ws = inputs["render_c2ws"] if c2w is None else c2w.to(inputs["render_c2ws"]).expand_as(inputs["render_c2ws"])
    intrs = inputs["render_intrs"] if K is None else K.to(inputs["render_intrs"]).expand_as(inputs["render_intrs"])
    out = model.renderer.render_gs(gs_models=gs_models, query_points=query_points, view_mano=view_mano,
                                   c2w=c2ws, intrinsic=intrs, height=H, width=W,
                                   background_color=inputs["render_bg_colors"], debug=False,
                                   train_stage=model.current_train_stage)
    rgb = out["comp_rgb"].reshape(-1, 3, H, W)[0].permute(1, 2, 0).float().clamp(0, 1).cpu().numpy()
    return (rgb * 255.0 + 0.5).astype(np.uint8)


@torch.no_grad()
def posed_points(model: ModelHAM, avatar, view_mano: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Posed Gaussian centers [N,3]."""
    pts = model.renderer.get_posed_gaussian_centers(gs_models=avatar[0], query_points=avatar[1], view_mano=view_mano)
    while pts.ndim > 2:        # [B,V,N,3] -> first sample, first view
        pts = pts[0]
    return pts.float()


# =============================================================================================
# optional seam blending (inference-time post-process of the Gaussian colors; not in the paper)
# =============================================================================================
SEAM_ITERS = 200          # Jacobi iterations of the k-NN mean inside the band
SEAM_OBSERVED = 0.05      # gamma above this counts as observed in the input


@torch.no_grad()
def _knn(x: torch.Tensor, k: int = 8, chunk: int = 2048) -> torch.Tensor:
    """[N,3] -> [N,k] indices of the k nearest other points."""
    out = []
    for s in range(0, x.shape[0], chunk):
        out.append(torch.cdist(x[s:s + chunk], x).topk(k + 1, dim=1, largest=False).indices[:, 1:])
    return torch.cat(out, 0)


@torch.no_grad()
def _dist_to_set(x: torch.Tensor, ref: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    return torch.cat([torch.cdist(x[s:s + chunk], ref).amin(dim=1) for s in range(0, x.shape[0], chunk)])


@torch.no_grad()
def seam_blend(final_rgb: torch.Tensor, base_rgb: torch.Tensor, gamma: torch.Tensor, xyz: torch.Tensor,
               band: float, iters: int = SEAM_ITERS, thr: float = SEAM_OBSERVED) -> torch.Tensor:
    """Replace the color step at the boundary between observed (gamma > thr) and unobserved
    Gaussians by a smooth ramp.

    d = c* - c0 (the color change made by the appearance module) is re-solved as a harmonic
    function inside a band of half-width `band` (MANO units, metres) around that boundary, with
    its values outside the band held fixed (Jacobi iterations of the k-NN mean). Colors away
    from the boundary are unchanged.
    final_rgb, base_rgb [N,3]; gamma [N]; xyz [N,3] (posed Gaussian centers) -> [N,3]"""
    obs = gamma > thr
    if obs.all() or (~obs).all():
        return final_rgb
    nbr = _knn(xyz.float())
    seam = (obs & (~obs[nbr]).any(1)) | (~obs & obs[nbr].any(1))
    in_band = _dist_to_set(xyz.float(), xyz[seam].float()) < band
    d = (final_rgb - base_rgb).float()
    for _ in range(int(iters)):
        d = torch.where(in_band[:, None], d[nbr].mean(1), d)
    print(f"[seam] blended {int(in_band.sum())} of {obs.numel()} Gaussians "
          f"({100.0 * float(in_band.sum()) / obs.numel():.1f}%) within {band * 1000:.1f} mm of the observed boundary")
    return (base_rgb.float() + d).clamp(0.0, 1.0).to(final_rgb.dtype)


def enable_seam_blending(model: ModelHAM, mm: float) -> None:
    """Apply seam_blend() to the output of every appearance head of `model` (the Stage-2
    AppearanceHead, whose forward returns the final Gaussian colors c*)."""
    heads = [m for m in model.modules() if type(m).__name__ == "AppearanceHead"]
    if not heads:
        raise SystemExit("seam blending is on, but the model has no AppearanceHead")
    for head in heads:
        head_forward = head.forward

        def forward(inputs, _head=head, _forward=head_forward):
            final = _forward(inputs)                                   # [B,N,3]
            xyz = inputs["posed_base_gs"].xyz
            xyz = xyz.unsqueeze(0) if xyz.ndim == 2 else xyz
            base = _head.to_base_color(inputs["posed_base_gs"].shs)    # c0, [B,N,3]
            gamma = _head.last_observed_confidence                     # [B,N]
            return torch.stack([seam_blend(final[b], base[b], gamma[b], xyz[b], mm / 1000.0)
                                for b in range(final.shape[0])], 0)

        head.forward = forward
    print(f"[seam] seam blending on: +-{mm} mm around the observed boundary")


# =============================================================================================
# optional k=4 densification (Appendix, Table S10; not used for the paper's main results)
# =============================================================================================
@torch.no_grad()
def midpoint_parents(q3: torch.Tensor, q4: torch.Tensor, knn: int = 8, tol_rel: float = 1e-3):
    """For every k=4 anchor, the two k=3 anchors it is interpolated from: itself twice if it
    coincides with a k=3 anchor, otherwise the endpoints of the k=3 edge whose midpoint it is."""
    a3 = q3.reshape(-1, 3).float().cpu().numpy()
    a4 = q4.reshape(-1, 3).float().cpu().numpy()
    ext = float(np.linalg.norm(a3.max(0) - a3.min(0)))
    tree = cKDTree(a3)
    d0, i0 = tree.query(a4, k=1)
    _, nbr = tree.query(a4, k=knn)                                   # [N4, knn]
    px, py = np.triu_indices(knn, k=1)                               # all pairs of neighbours
    A = a3[nbr]                                                      # [N4, knn, 3]
    err = np.linalg.norm(0.5 * (A[:, px] + A[:, py]) - a4[:, None, :], axis=-1)
    best = err.argmin(1)
    rows = np.arange(len(a4))
    pi, pj, e = nbr[rows, px[best]], nbr[rows, py[best]], err[rows, best]
    same = d0 <= tol_rel * ext
    pi[same] = i0[same]
    pj[same] = i0[same]
    e[same] = 0.0
    info = {"N3": int(len(a3)), "N4": int(len(a4)), "copied": int(same.sum()),
            "interpolated": int((~same).sum()), "max_midpoint_error_rel": float(e.max() / ext)}
    return (torch.from_numpy(pi), torch.from_numpy(pj)), info


def fill_from_k3(gs3, gs4, parents, n3: int, scale_factor: float = 1.0, scale_keys=("scal",)):
    """Overwrite every per-Gaussian tensor of the k=4 container `gs4` with the k=3 values averaged
    over the parents (scales also multiplied by `scale_factor`). Returns (gs4, filled paths)."""
    pi, pj = parents
    filled = []

    def interp(t, factor):
        if t.shape[0] == n3:
            o = 0.5 * (t[pi.to(t.device)] + t[pj.to(t.device)])
            return o * factor if factor != 1.0 else o
        if t.ndim >= 2 and t.shape[0] == 1 and t.shape[1] == n3:
            return interp(t[0], factor).unsqueeze(0)
        return None

    def walk(a, b, path):
        if torch.is_tensor(a) and torch.is_tensor(b):
            scaled = any(k in path.lower() for k in scale_keys)
            new = interp(a, scale_factor if scaled else 1.0)
            if new is not None and new.shape == b.shape:
                filled.append((path, scaled))
                return new.to(b.dtype)
            return b
        if isinstance(a, dict) and isinstance(b, dict):
            return {k: (walk(a[k], b[k], f"{path}.{k}") if k in a else b[k]) for k in b}
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
            return type(b)([walk(x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b))])
        if hasattr(a, "__dict__") and hasattr(b, "__dict__"):
            for k, v in list(vars(b).items()):
                if k in vars(a):
                    try:
                        setattr(b, k, walk(vars(a)[k], v, f"{path}.{k}"))
                    except Exception:
                        pass
            return b
        return b

    return walk(gs3, gs4, "gs"), filled


@torch.no_grad()
def densify_k4(model4: ModelHAM, avatar3, inputs: Dict[str, Any], device: torch.device, scale: float):
    """k=3 avatar -> k=4 avatar (method B of Table S10). The k=4 network pass only provides the
    k=4 Gaussian container and anchors; every per-Gaussian value is replaced by the k=3 prediction."""
    gs3, q3 = avatar3
    gs4, q4 = build_avatar(model4, inputs, device)
    parents, info = midpoint_parents(q3, q4)
    gs4, filled = fill_from_k3(gs3, gs4, parents, q3.shape[-2], scale_factor=scale)
    if not filled:
        raise SystemExit("k=4 densification: no per-Gaussian tensor was filled (Gaussian container changed?)")
    print(f"[k4] {info['N3']} -> {info['N4']} Gaussians ({info['copied']} copied, {info['interpolated']} "
          f"interpolated, max midpoint error {info['max_midpoint_error_rel']:.1e}); scale x{scale}; "
          f"filled {', '.join(p + (' (scaled)' if sc else '') for p, sc in filled)}")
    return gs4, q4


# =============================================================================================
# rotations
# =============================================================================================
_AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}


def _axis(name: str, device) -> torch.Tensor:
    name = name.strip().lower()
    name = "+" + name if name in ("x", "y", "z") else name
    if name not in _AXES:
        raise ValueError(f"axis must be one of {list(_AXES)}, got {name!r}")
    return torch.tensor(_AXES[name], device=device, dtype=torch.float32)


def _rodrigues(axis: torch.Tensor, theta: float) -> torch.Tensor:
    axis = axis / axis.norm().clamp_min(1e-8)
    x, y, z = axis
    K = torch.tensor([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], device=axis.device, dtype=axis.dtype)
    return torch.eye(3, device=axis.device, dtype=axis.dtype) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def _aa_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    aa = aa.reshape(3).float()
    theta = float(aa.norm())
    if theta < 1e-8:
        return torch.eye(3, device=aa.device)
    return _rodrigues(aa, theta)


def _aa_to_quat(aa: torch.Tensor) -> torch.Tensor:
    theta = aa.norm(dim=-1, keepdim=True)
    axis = aa / theta.clamp_min(1e-8)
    return torch.cat([torch.cos(theta * 0.5), axis * torch.sin(theta * 0.5)], dim=-1)


def _quat_to_aa(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    q = torch.where(q[..., :1] < 0, -q, q)
    w = q[..., :1].clamp(-1.0, 1.0)
    s = torch.sqrt((1.0 - w * w).clamp_min(1e-12))
    theta = 2.0 * torch.acos(w)
    axis = torch.where(s < 1e-6, torch.zeros_like(q[..., 1:]), q[..., 1:] / s)
    return axis * theta


def _slerp(q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
    d = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(d < 0, -q1, q1)
    theta = torch.acos(d.abs().clamp(max=1.0 - 1e-7))
    sin_t = torch.sin(theta)
    w0 = torch.sin((1.0 - t) * theta) / sin_t.clamp_min(1e-8)
    w1 = torch.sin(t * theta) / sin_t.clamp_min(1e-8)
    out = torch.where(sin_t < 1e-6, (1.0 - t) * q0 + t * q1, w0 * q0 + w1 * q1)
    return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def mean_rotation(aa: torch.Tensor) -> torch.Tensor:
    """Rotation matrix [3,3] closest to the average of the axis-angle rotations aa [N,3]
    (chordal L2 mean: principal eigenvector of the summed quaternion outer products)."""
    q = _aa_to_quat(aa.reshape(-1, 3).float())
    m = torch.linalg.eigh(q.T @ q)[1][:, -1]            # principal eigenvector: sign-free quaternion mean
    return _aa_to_matrix(_quat_to_aa(m / m.norm().clamp_min(1e-8)))


# =============================================================================================
# cameras
# =============================================================================================
def _revolve(c2w: torch.Tensor, center: torch.Tensor, axis: torch.Tensor, theta: float) -> torch.Tensor:
    """c2w rotated by theta about the line through `center` along `axis`."""
    R = _rodrigues(axis, float(theta))
    M = torch.eye(4, device=c2w.device, dtype=c2w.dtype)
    M[:3, :3] = R
    M[:3, 3] = center - R @ center
    return M @ c2w


def _look_at(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """OpenCV camera (x right, y down, z forward) at `eye` looking at `target`, image-up = `up`."""
    eye, target, up = eye.float(), target.float(), up.float()
    z = target - eye
    z = z / z.norm().clamp_min(1e-8)
    y = -up
    y = y - (y @ z) * z
    if y.norm() < 1e-6:                                   # up parallel to the view direction
        y = torch.tensor([0.0, -1.0, 0.0], device=eye.device)
        y = y - (y @ z) * z
    y = y / y.norm().clamp_min(1e-8)
    x = torch.cross(y, z, dim=0)
    x = x / x.norm().clamp_min(1e-8)
    c2w = torch.eye(4, device=eye.device, dtype=torch.float32)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    return c2w


def orbit(base: torch.Tensor, center: torch.Tensor, n_frames: int, deg: float, axis: str) -> List[torch.Tensor]:
    """Camera path around `center`: a full turn (deg >= 360) or a swing of +-deg/2 and back.
    yaw turns about the image up axis, pitch about the image x axis."""
    base, center = base.detach().float(), center.detach().float().view(3)
    rot_axis = -base[:3, 1] if axis == "yaw" else base[:3, 0]
    n = max(int(n_frames), 1)
    if deg >= 360.0:
        thetas = [2.0 * math.pi * i / n for i in range(n)]
    else:
        h, q = math.radians(deg) / 2.0, max(n // 4, 1)
        thetas = list(np.concatenate([np.linspace(0.0, h, q, endpoint=False),
                                      np.linspace(h, -h, 2 * q, endpoint=False),
                                      np.linspace(-h, 0.0, n - 3 * q, endpoint=True)]))
    return [_revolve(base, center, rot_axis, t) for t in thetas]


def turn_views(base: torch.Tensor, center: torch.Tensor, degs: Sequence[float]) -> List[torch.Tensor]:
    """The base camera turned by each angle about the image up axis through `center`."""
    up = -base[:3, 1]
    return [_revolve(base, center, up, math.radians(float(d))) for d in degs]


# MANO joint order: 0 wrist, 1-3 index, 4-6 middle, 7-9 little, 10-12 ring, 13-15 thumb
_WRIST, _INDEX_MCP, _MIDDLE_MCP, _LITTLE_MCP = 0, 1, 4, 7


@torch.no_grad()
def anatomical_axes(model, mano: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """(dorsal, up) in the MANO frame from the joints of the flat rest hand of this subject:
    up = wrist -> middle-finger base, dorsal = normal of the back of the hand. The hand is a right
    hand (left hands are mirrored in preprocessing), for which index base x little base points out
    of the palm. Independent of how the MANO frame is oriented."""
    betas = mano["betas"].reshape(1, -1)[:, :10].float()
    zeros = lambda n: torch.zeros(1, n, device=betas.device, dtype=betas.dtype)
    _, _, J = model.renderer.mano_model._joint_transforms(betas, zeros(3), zeros(45))
    J = J[0].float()
    palmar = torch.cross(J[_INDEX_MCP] - J[_WRIST], J[_LITTLE_MCP] - J[_WRIST], dim=0)
    dorsal = -palmar / palmar.norm().clamp_min(1e-8)
    up = J[_MIDDLE_MCP] - J[_WRIST]
    up = up - (up @ dorsal) * dorsal
    return dorsal, up / up.norm().clamp_min(1e-8)


def seen_side(mano: Dict[str, torch.Tensor], dorsal: torch.Tensor) -> Tuple[str, float]:
    """Which side of the hand the input photo shows: 'dorsal' or 'palmar', and the cosine between
    the back-of-hand normal and the direction to the input camera (at the origin, looking +z).
    The finger pose is ignored (flat-hand normal rotated by the global orientation)."""
    n = _aa_to_matrix(mano["global_orient"].reshape(-1)[:3]) @ dorsal.float()
    t = mano["translation"].reshape(-1)[:3].float()
    cos = float(n @ (-t / t.norm().clamp_min(1e-8)))
    return ("dorsal" if cos >= 0.0 else "palmar"), cos


def resolve_start_side(cfg_value: str, mano: Dict[str, torch.Tensor], dorsal: torch.Tensor) -> str:
    """camera.start_side: 'auto' (the side the input shows), 'dorsal' or 'palmar'."""
    value = str(cfg_value or "auto").lower()
    if value == "auto":
        side, cos = seen_side(mano, dorsal)
        print(f"[camera] input shows the {side} side (cos {cos:+.2f}); videos start there")
        return side
    if value not in ("dorsal", "palmar"):
        raise ValueError(f"camera.start_side must be auto, dorsal or palmar, got {cfg_value!r}")
    return value


def view_axes(model, mano: Dict[str, torch.Tensor], cam_cfg: Dict[str, Any], orientation: str, side: str,
              rotation: Optional[torch.Tensor], device) -> Tuple[torch.Tensor, torch.Tensor]:
    """(view-from, image-up) directions in the MANO frame.
    'anatomical': from the `side` of the hand, fingers up; `rotation` (3x3) turns these with the
    hand when the rendered hand has a global orientation. 'axes': the fixed MANO-frame axes
    camera.dorsal_axis / camera.finger_axis (the supplementary-video setting)."""
    if orientation == "axes":
        return _axis(cam_cfg["dorsal_axis"], device), _axis(cam_cfg["finger_axis"], device)
    if orientation != "anatomical":
        raise ValueError(f"camera orientation must be 'anatomical' or 'axes', got {orientation!r}")
    dorsal, up = anatomical_axes(model, mano)
    view = dorsal if side == "dorsal" else -dorsal
    if rotation is not None:
        R = rotation.to(view.device).float()
        view, up = R @ view, R @ up
    return view, up


@dataclass
class Camera:
    """The render camera of one video set (written to camera.json)."""
    fov_deg: float
    fill: float
    fit: str
    orientation: str
    start_side: str
    view_dir: List[float]
    up_dir: List[float]
    dist: float
    radius: float
    center: List[float]
    res: int
    intr: List[List[float]]
    base_c2w: List[List[float]]
    extra: Dict[str, Any] = field(default_factory=dict)


def fixed_fov_intrinsics(fov_deg: float, H: int, W: int, device) -> torch.Tensor:
    f = (H / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    K = torch.eye(4, dtype=torch.float32, device=device)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = f, f, W / 2.0, H / 2.0
    return K


def fit_distance(pts: Sequence[torch.Tensor], cam_rots: Sequence[torch.Tensor], center: torch.Tensor,
                 half_tan: float, q: float = 0.998) -> float:
    """Smallest distance from `center` at which every point set in `pts`, seen through every
    camera rotation in `cam_rots` (3x3, columns x right / y down / z forward, looking at
    `center`), projects inside +-half_tan (tangent units) in x and y. A point at camera
    coordinates (x, y, z) relative to the center lies at depth d + z, so it fits when
    d >= max(|x|, |y|) / half_tan - z. The q-quantile per set ignores a few stray Gaussians."""
    need = 0.0
    for R in cam_rots:
        for p in pts:
            rel = (p - center.view(1, 3)) @ R.to(p)
            req = rel[:, :2].abs().amax(dim=1) / half_tan - rel[:, 2]
            need = max(need, float(torch.quantile(req, q)), float(rel[:, 2].neg().max()) + 1e-3)
    return need


def place_camera(pts: Sequence[torch.Tensor], view_dir: torch.Tensor, up_dir: torch.Tensor,
                 cam_cfg: Dict[str, Any], fill: float, path_fn, tag: str, orientation: str, side: str,
                 K: torch.Tensor, res: int) -> Tuple[torch.Tensor, torch.Tensor, Camera]:
    """Base camera looking along -view_dir with up_dir as image up.
    camera.fit 'tight': centered on the middle of the hand as seen by the base camera, at the
    distance where the hand, over all point sets and all cameras of the video (path_fn(base,
    center) -> list of c2w), spans `fill` of the frame at its widest.
    camera.fit 'sphere': centered on the mean point, at the distance where the bounding sphere
    (0.995-quantile radius) spans `fill` of the frame height (the supplementary-video framing).
    camera.dist > 0 fixes the distance. Returns (base c2w, center, Camera)."""
    fov, fit = float(cam_cfg["fov_deg"]), str(cam_cfg.get("fit", "tight"))
    view_dir, up_dir = view_dir.float(), up_dir.float()
    allc = torch.cat(list(pts), 0)
    mean = allc.mean(dim=0)
    radius = max(float(torch.quantile((p - mean.view(1, 3)).norm(dim=-1), 0.995)) for p in pts)
    half_tan = math.tan(math.radians(fov) / 2.0)
    if fit == "sphere":
        center = mean
        dist = radius / (max(fill, 1e-3) * half_tan)
    elif fit == "tight":
        R0 = _look_at(mean + view_dir, mean, up_dir)[:3, :3]
        lo = torch.stack([torch.quantile(p @ R0, 0.002, dim=0) for p in pts]).amin(dim=0)
        hi = torch.stack([torch.quantile(p @ R0, 0.998, dim=0) for p in pts]).amax(dim=0)
        mid = 0.5 * (lo + hi)
        mid[2] = (mean @ R0)[2]
        center = R0 @ mid
        unit = _look_at(center + view_dir, center, up_dir)
        rots = [c[:3, :3] for c in path_fn(unit, center)]
        dist = fit_distance(pts, rots, center, half_tan * max(fill, 1e-3))
    else:
        raise ValueError(f"camera.fit must be 'tight' or 'sphere', got {fit!r}")
    if float(cam_cfg.get("dist", 0.0)) > 0:
        dist = float(cam_cfg["dist"])
    base = _look_at(center + dist * view_dir, center, up_dir)
    cam = Camera(fov_deg=fov, fill=fill, fit=fit, orientation=orientation, start_side=side,
                 view_dir=view_dir.tolist(), up_dir=up_dir.tolist(), dist=dist, radius=radius,
                 center=center.tolist(), res=res, intr=K.detach().float().cpu().tolist(),
                 base_c2w=base.detach().cpu().tolist())
    print(f"[camera:{tag}] {orientation}/{side} fit={fit} fov={fov:.1f} fill={fill:.2f} "
          f"radius={radius:.4f} dist={dist:.4f} res={res}")
    return base, center, cam


# =============================================================================================
# poses
# =============================================================================================
def _clone(mano: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in mano.items()}


def mano_frame_pose(mano: Dict[str, torch.Tensor], rest: bool) -> Dict[str, torch.Tensor]:
    """The input hand in the MANO frame (global orientation zero), in the rest pose (finger
    pose zero) or with its own finger pose."""
    out = _clone(mano)
    if rest:
        out["hand_pose"] = torch.zeros_like(out["hand_pose"])
    out["global_orient"] = torch.zeros_like(out["global_orient"])
    return out


def bank_pose(mano: Dict[str, torch.Tensor], pose48: torch.Tensor, keep_global: bool) -> Dict[str, torch.Tensor]:
    """The input hand (shape, translation) with a pose-bank pose [48] = global orient + fingers.
    keep_global False: global orientation zero (the hand stays in the MANO frame, only the
    fingers move)."""
    out = _clone(mano)
    p = pose48.detach().to(out["hand_pose"].device, out["hand_pose"].dtype).reshape(48)
    out["hand_pose"] = p[3:].reshape(out["hand_pose"].shape)
    out["global_orient"] = p[:3].reshape(out["global_orient"].shape) if keep_global \
        else torch.zeros_like(out["global_orient"])
    return out


def load_pose_bank(path: str) -> Tuple[np.ndarray, List[str]]:
    """[N,48] axis-angle poses (global orient + 15 joints) from .npy / .pt / .json; names from
    a .json next to the .npy / .pt if it lists `gesture_name` or `pose_id` per pose."""
    p = Path(path)
    names = None
    if p.suffix == ".npy":
        arr = np.load(str(p))
    elif p.suffix in (".pt", ".pth"):
        obj = torch.load(str(p), map_location="cpu")
        if isinstance(obj, dict):
            obj = obj["poses"] if "poses" in obj else next(v for v in obj.values() if torch.is_tensor(v))
        arr = obj.detach().cpu().numpy() if torch.is_tensor(obj) else np.asarray(obj)
    elif p.suffix == ".json":
        with open(p) as f:
            j = json.load(f)
        entries = j["poses"] if isinstance(j, dict) else j
        arr = np.asarray([e["pose"] for e in entries], dtype=np.float32)
        names = [str(e.get("gesture_name", e.get("pose_id", i))) for i, e in enumerate(entries)]
    else:
        raise ValueError(f"unsupported pose bank format: {p.suffix}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 48:
        raise ValueError(f"pose bank must be [N,48] axis-angle, got {arr.shape}")
    if names is None and p.with_suffix(".json").exists():
        with open(p.with_suffix(".json")) as f:
            j = json.load(f)
        entries = j["poses"] if isinstance(j, dict) else j
        if len(entries) == len(arr):
            names = [str(e.get("gesture_name", e.get("pose_id", i))) for i, e in enumerate(entries)]
    return arr, names or [f"pose{i:03d}" for i in range(len(arr))]


def pose_sequence(poses48: torch.Tensor, hold: int, trans: int, loop: bool) -> List[torch.Tensor]:
    """Each pose held for `hold` frames, then `trans` frames of per-joint slerp to the next."""
    seq: List[torch.Tensor] = []
    n, hold, trans = poses48.shape[0], max(int(hold), 1), max(int(trans), 0)
    for i in range(n):
        seq.extend([poses48[i]] * hold)
        j = i + 1
        if j >= n:
            if not loop:
                break
            j = 0
        q0, q1 = _aa_to_quat(poses48[i].view(16, 3)), _aa_to_quat(poses48[j].view(16, 3))
        for k in range(trans):
            seq.append(_quat_to_aa(_slerp(q0, q1, (k + 1) / (trans + 1))).reshape(48))
    return seq


# =============================================================================================
# videos
# =============================================================================================
class VideoWriter:
    """mp4 via imageio (or OpenCV); if neither can encode, the frames are kept as PNGs."""

    def __init__(self, path: str, fps: int, frames_dir: Optional[str] = None):
        self.path, self.fps, self.frames_dir = str(path), int(fps), frames_dir
        self.frames: List[np.ndarray] = []
        if frames_dir:
            os.makedirs(frames_dir, exist_ok=True)

    def add(self, frame: np.ndarray) -> None:
        self.frames.append(np.ascontiguousarray(frame))
        if self.frames_dir:
            Image.fromarray(frame).save(os.path.join(self.frames_dir, f"{len(self.frames) - 1:05d}.png"))

    def close(self) -> None:
        if not self.frames:
            return
        try:
            import imageio
            imageio.mimsave(self.path, self.frames, fps=self.fps, quality=8, macro_block_size=1)
            print(f"[video] {self.path} ({len(self.frames)} frames)")
            return
        except Exception as e:
            print(f"[video] imageio failed ({e}); trying OpenCV")
        try:
            import cv2
            h, w = self.frames[0].shape[:2]
            vw = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            for f in self.frames:
                vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            vw.release()
            print(f"[video] {self.path} ({len(self.frames)} frames, OpenCV)")
            return
        except Exception as e:
            print(f"[video] OpenCV failed ({e})")
        d = self.frames_dir or self.path + "_frames"
        if not self.frames_dir:
            os.makedirs(d, exist_ok=True)
            for i, f in enumerate(self.frames):
                Image.fromarray(f).save(os.path.join(d, f"{i:05d}.png"))
        print(f"[video] no encoder; frames in {d} "
              f"(ffmpeg -framerate {self.fps} -i {d}/%05d.png -c:v libx264 -pix_fmt yuv420p {self.path})")


def _even(img: np.ndarray) -> np.ndarray:
    return img[:img.shape[0] - img.shape[0] % 2, :img.shape[1] - img.shape[1] % 2]


@torch.no_grad()
def make_turntables(model, avatar, inputs, save_root: Path, cfg: Dict[str, Any], cams: Dict[str, Camera]) -> None:
    """The avatar in the MANO frame (rest pose or the input finger pose), seen from a full yaw
    turn and a pitch swing, rendered at RES x RES. The start view looks at the side of the hand
    the input photo shows (camera.start_side: auto) with the fingers up
    (camera.canonical_orientation: anatomical)."""
    vcfg, ccfg = cfg["videos"], cfg["camera"]
    res = RES
    dev = inputs["render_intrs"].device
    K = fixed_fov_intrinsics(ccfg["fov_deg"], res, res, dev)
    anchor = inputs["mano_params"]
    mano = mano_frame_pose(anchor, rest=(vcfg["orbit_pose"] == "rest"))
    orientation = ccfg.get("canonical_orientation", "anatomical")
    side = resolve_start_side(ccfg.get("start_side", "auto"), anchor, anatomical_axes(model, anchor)[0]) \
        if orientation == "anatomical" else "dorsal"
    view_dir, up_dir = view_axes(model, anchor, ccfg, orientation, side, None, dev)
    pts = [posed_points(model, avatar, mano)]
    axes = list(vcfg["orbit_axes"])
    degs = {a: (vcfg["yaw_deg"] if a == "yaw" else vcfg["pitch_deg"]) for a in axes}
    n_fit = max(int(vcfg["n_orbit"]) // 4, 8)
    path = lambda b, c: [v for a in axes for v in orbit(b, c, n_fit, degs[a], a)]
    base, center, cams["canonical"] = place_camera(pts, view_dir, up_dir, ccfg, float(ccfg["fill"]), path,
                                                   "canonical", orientation, side, K, res)
    for axis in axes:
        writer = VideoWriter(save_root / f"canonical_360_{axis}.mp4", vcfg["fps"],
                             str(save_root / f"frames_canonical_{axis}") if vcfg["save_frames"] else None)
        for c2w in tqdm(orbit(base, center, vcfg["n_orbit"], degs[axis], axis), desc=f"turntable ({axis})"):
            writer.add(_even(render(model, avatar, inputs, mano, c2w, K, res, res)))
        writer.close()


@torch.no_grad()
def make_pose_animation(model, avatar, inputs, save_root: Path, cfg: Dict[str, Any], cams: Dict[str, Camera]) -> None:
    """The avatar driven through the pose bank (held poses joined by slerp transitions), seen from
    the start camera turned by each of pose_view_degs about the image up axis, rendered at
    RES x RES per view.
    videos.pose_global 'rest': the pose-bank global orientations are replaced by zero, so the
    hand stays in the orientation of the turntable and only the fingers move. 'bank': the
    global orientations are kept and the camera follows their mean."""
    vcfg, ccfg = cfg["videos"], cfg["camera"]
    res = RES
    poses_np, names = load_pose_bank(cfg["pose_bank"])
    ids = list(vcfg["pose_ids"]) if vcfg.get("pose_ids") else list(range(len(poses_np)))
    if vcfg["pose_max"] > 0:
        ids = ids[:vcfg["pose_max"]]
    if not ids:
        raise ValueError("the pose selection is empty (videos.pose_ids / videos.pose_max)")
    poses = torch.from_numpy(poses_np[ids]).float()
    print(f"[pose] {len(ids)} poses: {[names[i] for i in ids][:8]}{' ...' if len(ids) > 8 else ''}")

    pose_global = str(vcfg.get("pose_global", "rest")).lower()
    if pose_global not in ("rest", "bank"):
        raise ValueError(f"videos.pose_global must be 'rest' or 'bank', got {pose_global!r}")
    keep_global = pose_global == "bank"
    anchor = inputs["mano_params"]
    dev = inputs["render_intrs"].device
    K = fixed_fov_intrinsics(ccfg["fov_deg"], res, res, dev)
    orientation = ccfg.get("pose_orientation", "anatomical")
    side = resolve_start_side(ccfg.get("start_side", "auto"), anchor, anatomical_axes(model, anchor)[0]) \
        if orientation == "anatomical" else "dorsal"
    rotation = mean_rotation(poses[:, :3]).to(dev) if keep_global else None
    view_dir, up_dir = view_axes(model, anchor, ccfg, orientation, side, rotation, dev)

    seq = pose_sequence(poses, round(vcfg["pose_hold_sec"] * vcfg["fps"]), round(vcfg["pose_transition_sec"] * vcfg["fps"]),
                        vcfg["pose_loop"])
    view_degs = [float(d) for d in vcfg["pose_view_degs"]]
    if str(ccfg.get("fit", "tight")) == "sphere":        # supplementary-video framing: the bank poses only
        fit_poses = list(poses)
    else:                                                # every pose the video passes through (with transitions)
        fit_poses = seq[::max(len(seq) // int(vcfg.get("pose_fit_frames", 120)), 1)]
    fit_pts = [posed_points(model, avatar, bank_pose(anchor, p, keep_global)) for p in fit_poses]
    fill = float(ccfg.get("pose_fill", ccfg["fill"]))
    base, center, cam = place_camera(fit_pts, view_dir, up_dir, ccfg, fill,
                                     lambda b, c: turn_views(b, c, view_degs),
                                     "pose_anim", orientation, side, K, res)
    cam.extra = {"pose_global": pose_global, "view_degs": view_degs, "n_poses": len(ids)}
    cams["pose_anim"] = cam
    views = turn_views(base, center, view_degs)

    tags = [f"view_{'+' if d >= 0 else '-'}{abs(int(round(d))):03d}" for d in view_degs]
    frames_dir = lambda tag: str(save_root / f"frames_pose_anim_{tag}") if vcfg["save_frames"] else None
    if vcfg["pose_layout"] == "hstack" and len(views) > 1:
        writers = [VideoWriter(save_root / "pose_anim_multiview.mp4", vcfg["fps"], frames_dir("multiview"))]
    else:
        writers = [VideoWriter(save_root / f"pose_anim_{t}.mp4", vcfg["fps"], frames_dir(t)) for t in tags]
    divider = np.full((res - res % 2, 4, 3), 255, dtype=np.uint8)
    for pose48 in tqdm(seq, desc="pose animation"):
        mano = bank_pose(anchor, pose48, keep_global)
        panels = [_even(render(model, avatar, inputs, mano, c2w, K, res, res)) for c2w in views]
        if len(writers) == 1 and len(panels) > 1:
            row = [panels[0]]
            for p in panels[1:]:
                row += [divider, p]
            writers[0].add(np.concatenate(row, axis=1))
        else:
            for w, p in zip(writers, panels):
                w.add(p)
    for w in writers:
        w.close()


# =============================================================================================
# main
# =============================================================================================
def resolve_inputs(args, cfg) -> List[Tuple[str, str, Optional[str], str]]:
    """[(name, image, mask, wilor_json)] from --ids (<input_root>/rgba/<id>.png + json/<id>.json)
    or from --image / --mask / --wilor_json."""
    if args.ids:
        root = Path(cfg["input_root"])
        ids = sorted(p.stem for p in (root / "rgba").glob("*.png")) if args.ids == "all" else \
            [s.strip() for s in args.ids.split(",") if s.strip()]
        jobs = []
        for i in ids:
            img, js = root / "rgba" / f"{i}.png", root / "json" / f"{i}.json"
            if not (img.exists() and js.exists()):
                print(f"[skip] {i}: missing {img if not img.exists() else js}")
                continue
            jobs.append((i, str(img), None, str(js)))
        return jobs
    if not (args.image and args.wilor_json):
        raise SystemExit("give --ids, or --image and --wilor_json (and --mask unless the image has alpha)")
    return [(args.name or Path(args.image).stem, args.image, args.mask, args.wilor_json)]


@torch.no_grad()
def run_one(model, prep: FramePreprocessor, cfg, name: str, image: str, mask: Optional[str], wilor_json: str,
            device, model4: Optional[ModelHAM] = None) -> None:
    save_root = Path(cfg["out_dir"]) / name
    save_root.mkdir(parents=True, exist_ok=True)
    pcfg = cfg["preprocess"]
    rgba = compose_rgba(image, mask, str(save_root / "input_rgba.png"), pcfg["border_feather"])
    inputs = prep(rgba, wilor_json, pcfg["multiply"], pcfg["bg_value"], device)
    got = tuple(inputs["image"].shape[-2:])
    if got != (RES, RES):
        raise RuntimeError(f"{name}: the network input is {got[0]}x{got[1]}, but the model was trained on "
                           f"{RES}x{RES} only (check preprocess.multiply)")
    H = W = RES
    src = inputs["image"][0, 0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    Image.fromarray((src * 255.0 + 0.5).astype(np.uint8)).save(save_root / "source.png")

    avatar = build_avatar(model, inputs, device)
    if model4 is not None:                   # k=4 densification: render every output from the k=4 avatar
        avatar = densify_k4(model4, avatar, inputs, device, float((cfg.get("densify") or {}).get("scale", 0.6)))
        model = model4
    Image.fromarray(render(model, avatar, inputs, inputs["mano_params"], None, None, H, W)).save(save_root / "self_render.png")

    cams: Dict[str, Camera] = {}
    if "canonical" in cfg["videos"]["kinds"]:
        make_turntables(model, avatar, inputs, save_root, cfg, cams)
    if "pose" in cfg["videos"]["kinds"]:
        if cfg.get("pose_bank") and Path(cfg["pose_bank"]).exists():
            make_pose_animation(model, avatar, inputs, save_root, cfg, cams)
        else:
            print(f"[pose] skipped: pose bank not found ({cfg.get('pose_bank')})")
    with open(save_root / "camera.json", "w") as f:
        json.dump({k: vars(v) for k, v in cams.items()}, f, indent=2)
    print(f"[done] {save_root}")


def parse_args():
    p = argparse.ArgumentParser(description="InstantHand demo: hand image -> avatar videos "
                                            f"(all outputs are {RES}x{RES}, the training resolution)")
    p.add_argument("--cfg", default="configs/demo.yaml")
    p.add_argument("--ids", default="", help="comma-separated example ids in <input_root>, or 'all'")
    p.add_argument("--image", default="")
    p.add_argument("--mask", default="", help="hand mask (omit if the image is RGBA with the mask as alpha)")
    p.add_argument("--wilor_json", default="")
    p.add_argument("--name", default="", help="output folder name (default: image file name)")
    p.add_argument("--seam_blend", action="store_true",
                   help="blend the color step at the observed boundary (postprocess.seam_blend_mm, "
                        "3 mm by default; not used in the paper)")
    p.add_argument("--k4", action="store_true",
                   help="render a k=4 densified avatar (densify.scale, 0.6 by default; Appendix, Table S10)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config entry, e.g. --set videos.kinds=[canonical] (repeatable)")
    args = p.parse_args()
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in args.set:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(value)
    for sec, key in (("preprocess", "image_res"), ("videos", "res")):
        if key in (cfg.get(sec) or {}):
            print(f"[res] {sec}.{key} is no longer used: the resolution is fixed at {RES}x{RES}")
            cfg[sec].pop(key)
    if args.seam_blend:
        cfg.setdefault("postprocess", {})["seam_blend"] = True
    if args.k4:
        cfg.setdefault("densify", {})["k4"] = True
    return args, cfg


def main():
    args, cfg = parse_args()
    jobs = resolve_inputs(args, cfg)
    if not jobs:
        raise SystemExit("no inputs")
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = load_model(cfg["model"], cfg["ckpt_dir"], device)
    post = cfg.get("postprocess") or {}
    if post.get("seam_blend", False):
        enable_seam_blending(model, float(post.get("seam_blend_mm", 3.0)))
    model4 = None
    if (cfg.get("densify") or {}).get("k4", False):
        # same weights, k=4 MANO subdivision (anchors and skinning weights of the densified avatar)
        model4 = load_model({**cfg["model"], "mano_subdivide_num": 4}, cfg["ckpt_dir"], device)
        print(f"[k4] densification on: k=3 prediction interpolated onto k=4 anchors, "
              f"scale x{float(cfg['densify'].get('scale', 0.6))}")
    prep = FramePreprocessor(kp_enlarge=cfg["preprocess"]["kp_enlarge"], alpha_thr=cfg["preprocess"]["alpha_thr"])
    print(f"[run] {len(jobs)} input(s): {[j[0] for j in jobs]}; resolution {RES}x{RES}")
    for name, image, mask, wilor_json in jobs:
        try:
            run_one(model, prep, cfg, name, image, mask or None, wilor_json, device, model4)
        except Exception as e:
            if len(jobs) == 1:
                raise
            print(f"[error] {name}: {e}")


if __name__ == "__main__":
    main()
