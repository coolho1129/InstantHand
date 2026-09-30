# Copyright (c) 2026 The InstantHand Authors.
# SPDX-License-Identifier: Apache-2.0
"""Adapter from a HandAvatar evaluation sample to InstantHand model inputs.

Principal point
---------------
The Gaussian rasterizer places the optical axis at the image center ((W-1)/2, (H-1)/2) and
ignores cx, cy, but the calibrated crop cameras of the benchmarks do not have a centered
principal point. With pp_mode="rotate" (default) the camera is rotated about its center so that
its optical axis passes through the crop center. For a pure rotation the two views are related
exactly by a homography (no parallax),

    x_render ~ Hm @ x_crop,     Hm = K_new @ R_rot @ inv(K),

so the model receives the rotated camera (centered principal point) and the source image and
mask warped into it, and every render is mapped back with `unwarp_to_crop(render, Hm)` before
it is compared with the ground truth, which stays in the original crop. HandAvatar and OHTA
render the crop pixels with the full intrinsics, so all methods are scored on the same pixels.

pp_mode="none" passes the calibrated camera unchanged; use it only with a rasterizer that
honours cx, cy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["HA2HAMCfg", "recenter_by_rotation", "warp_with_homography_np", "unwarp_to_crop",
           "handavatar_sample_to_infer_inputs"]


@dataclass
class HA2HAMCfg:
    pp_mode: str = "rotate"           # "rotate" | "none" (see the module docstring)
    bg_value: float = 0.0             # background in [0,1] when the sample has no "bgcolor"
    clamp01: bool = True              # clamp the image to [0,1]
    enable_source_mask: bool = True   # binarized target_alpha_img as the source hand mask
    alpha_thr: float = 0.5


# ---------------------------------------------------------------------------------------------
# camera rotation and homography warps
# ---------------------------------------------------------------------------------------------
def recenter_by_rotation(K33: np.ndarray, R: np.ndarray, T: np.ndarray, H: int, W: int) -> Dict[str, Any]:
    """Rotate the camera [R|T] about its center so that its optical axis passes through the crop
    center. Returns K_new (principal point exactly at ((W-1)/2, (H-1)/2)), R_new, T_new, the
    homography H_crop_to_render (x_render ~ Hm @ x_crop, pixel centers at integers) and the
    rotation angle in degrees."""
    K = np.asarray(K33, np.float64)
    c = np.array([(W - 1) * 0.5, (H - 1) * 0.5, 1.0])
    d = np.linalg.solve(K, c)                 # viewing ray through the crop center (z = 1)
    n = float(np.linalg.norm(d))
    dh = d / n
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(dh, z)
    s, cth = float(np.linalg.norm(v)), float(dh @ z)
    if s < 1e-12:
        R_rot = np.eye(3)
    else:                                     # minimal rotation with R_rot @ dh = z (Rodrigues)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R_rot = np.eye(3) + vx + vx @ vx * ((1.0 - cth) / (s * s))
    K_new = np.array([[K[0, 0] * n, 0.0, c[0]],     # f * n keeps the pixel scale at the crop center
                      [0.0, K[1, 1] * n, c[1]],
                      [0.0, 0.0, 1.0]])
    Hm = K_new @ R_rot @ np.linalg.inv(K)
    Hm = Hm / Hm[2, 2]
    return {
        "K_new": K_new.astype(np.float32),
        "R_new": (R_rot @ np.asarray(R, np.float64)).astype(np.float32),
        "T_new": (R_rot @ np.asarray(T, np.float64).reshape(3)).astype(np.float32),
        "H_crop_to_render": Hm,
        "angle_deg": float(np.degrees(np.arctan2(s, cth))),
    }


def _bilinear_sample_np(img: np.ndarray, xs: np.ndarray, ys: np.ndarray, fill) -> np.ndarray:
    """Sample img (H,W[,C]) at float pixel coordinates (pixel centers at integers); `fill` outside."""
    Hh, Ww = img.shape[:2]
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    wx = (xs - x0)[..., None] if img.ndim == 3 else (xs - x0)
    wy = (ys - y0)[..., None] if img.ndim == 3 else (ys - y0)
    inside = (xs >= 0) & (xs <= Ww - 1) & (ys >= 0) & (ys <= Hh - 1)
    x0c, x1c = np.clip(x0, 0, Ww - 1), np.clip(x0 + 1, 0, Ww - 1)
    y0c, y1c = np.clip(y0, 0, Hh - 1), np.clip(y0 + 1, 0, Hh - 1)
    out = (img[y0c, x0c] * (1 - wx) * (1 - wy) + img[y0c, x1c] * wx * (1 - wy)
           + img[y1c, x0c] * (1 - wx) * wy + img[y1c, x1c] * wx * wy).astype(np.float32)
    out[~inside] = np.asarray(fill, np.float32)
    return out


def warp_with_homography_np(img: np.ndarray, H_dst_to_src: np.ndarray, out_hw: Tuple[int, int], fill=0.0) -> np.ndarray:
    """out(x_dst) = img(H_dst_to_src @ x_dst), bilinear. img (H,W[,C])."""
    Ho, Wo = out_hw
    ys, xs = np.mgrid[0:Ho, 0:Wo].astype(np.float64)
    p = np.stack([xs, ys, np.ones_like(xs)], -1) @ np.asarray(H_dst_to_src, np.float64).T
    return _bilinear_sample_np(img.astype(np.float32), p[..., 0] / p[..., 2], p[..., 1] / p[..., 2], fill)


def unwarp_to_crop(x, H_crop_to_render, fill=0.0, mode: str = "bicubic", clamp: Optional[Tuple[float, float]] = None):
    """Map a render of the rotated camera back to the original crop.

    x     torch tensor [..., H, W] (channel-first: rgb [B,3,H,W], alpha [B,1,H,W]), or numpy
          array (H, W[, C]) (always bilinear; for quick checks).
    fill  value for crop pixels the rotated camera does not see: a scalar or one value per
          channel (pass the background color for RGB, 0 for alpha).
    mode  "bicubic" (use with clamp=(0,1)), "bilinear" or "nearest". Report metrics with bicubic
          or nearest: measured on HanCo renders, one bilinear resampling raises PSNR by about
          0.05 dB (it low-passes the render), bicubic by less than 0.01 dB.
    """
    Hm = np.asarray(_to_numpy(H_crop_to_render), np.float64).reshape(3, 3)
    if not torch.is_tensor(x):
        out = warp_with_homography_np(np.asarray(x, np.float32), Hm, np.asarray(x).shape[:2], fill=fill)
        return np.clip(out, clamp[0], clamp[1]) if clamp is not None else out

    Hh, Ww = int(x.shape[-2]), int(x.shape[-1])
    lead = x.shape[:-2]
    xf = x.reshape(-1, 1, Hh, Ww).float()
    ys, xs = torch.meshgrid(torch.arange(Hh, device=x.device, dtype=torch.float64),
                            torch.arange(Ww, device=x.device, dtype=torch.float64), indexing="ij")
    p = torch.stack([xs, ys, torch.ones_like(xs)], -1) @ torch.as_tensor(Hm, device=x.device).T  # crop -> render pixel
    px, py = p[..., 0] / p[..., 2], p[..., 1] / p[..., 2]
    inside = ((px >= 0) & (px <= Ww - 1) & (py >= 0) & (py <= Hh - 1)).float()
    grid = torch.stack([px * (2.0 / (Ww - 1)) - 1.0, py * (2.0 / (Hh - 1)) - 1.0], -1)   # align_corners=True
    grid = grid.float().unsqueeze(0).expand(xf.shape[0], -1, -1, -1)
    out = F.grid_sample(xf, grid, mode=mode, padding_mode="border", align_corners=True).reshape(*lead, Hh, Ww)

    fill_t = torch.as_tensor(fill, device=x.device, dtype=out.dtype)
    if fill_t.ndim == 1:                                       # one value per channel (dim -3)
        if x.ndim < 3 or fill_t.numel() != x.shape[-3]:
            raise ValueError("per-channel fill needs a channel-first tensor with matching channel count")
        fill_t = fill_t.view(-1, 1, 1)
    out = out * inside + fill_t * (1.0 - inside)
    if clamp is not None:
        out = out.clamp(clamp[0], clamp[1])
    return out.to(x.dtype)


# ---------------------------------------------------------------------------------------------
# adapter
# ---------------------------------------------------------------------------------------------
def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _source_mask_np(ha: Dict[str, Any], H: int, W: int, cfg: HA2HAMCfg) -> Optional[np.ndarray]:
    """Binarized hand mask of the sample (crop space) from target_alpha_img."""
    if not cfg.enable_source_mask:
        return None
    if ha.get("target_alpha_img") is None:
        raise ValueError("the sample needs 'target_alpha_img' for the source mask "
                         "(the ray mask is not a substitute: it is not the hand silhouette)")
    a = _to_numpy(ha["target_alpha_img"]).astype(np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    if a.shape[:2] != (H, W):
        import cv2
        a = cv2.resize(a, (W, H), interpolation=cv2.INTER_NEAREST)
    if a.max() > 2.0:
        a = a / 255.0
    return (a > float(cfg.alpha_thr)).astype(np.float32)


def _w2c_to_c2w(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """HandAvatar convention: x_cam = R @ x_world + T."""
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R.astype(np.float32)
    w2c[:3, 3] = T.astype(np.float32).reshape(3)
    return np.linalg.inv(w2c).astype(np.float32)


@torch.no_grad()
def handavatar_sample_to_infer_inputs(ha: Dict[str, Any], device, cfg: Optional[HA2HAMCfg] = None) -> Dict[str, Any]:
    """Build the model inputs from one HandAvatar sample (single view, batch 1).

    Needs target_img [H,W,3] in [0,1], dst_global_orient [3], dst_posevec [45], dst_shape [10],
    cam_K [3,3], cam_R [3,3], cam_T [3], target_alpha_img [H,W] and optionally bgcolor (RGB,
    0-255 or 0-1). Returns image [1,1,3,H,W], source / render cameras (c2w and 4x4 intrinsics,
    [1,1,4,4]), render_bg_colors [1,3], mano_params, render_full_resolutions [[H,W]] and
    data = {source_mask, target_mask (the model's view), gt_image_crop, gt_mask_crop (untouched
    crop: compute metrics against these), H_crop_to_render, H_render_to_crop (rotate mode), ...}.
    """
    cfg = cfg or HA2HAMCfg()
    for keys in (("target_img",), ("dst_global_orient", "dst_posevec", "dst_shape"), ("cam_K", "cam_R", "cam_T")):
        missing = [k for k in keys if k not in ha]
        if missing:
            raise KeyError(f"HandAvatar sample is missing {missing}")

    img = _to_numpy(ha["target_img"]).astype(np.float32)
    if cfg.clamp01:
        img = np.clip(img, 0.0, 1.0)
    H, W = int(img.shape[0]), int(img.shape[1])

    def param(key, shape):
        return torch.from_numpy(_to_numpy(ha[key]).astype(np.float32)).to(device).view(*shape)

    mano_params = {
        "betas": param("dst_shape", (1, 10)),
        "global_orient": param("dst_global_orient", (1, 1, 3)),
        "hand_pose": param("dst_posevec", (1, 1, 45)),
        "translation": torch.zeros((1, 3), dtype=torch.float32, device=device),
    }
    K33 = _to_numpy(ha["cam_K"]).astype(np.float32)
    R = _to_numpy(ha["cam_R"]).astype(np.float32)
    T = _to_numpy(ha["cam_T"]).astype(np.float32).reshape(3)

    if ha.get("bgcolor") is not None:
        bg = np.array(ha["bgcolor"], dtype=np.float32)
        if bg.max() > 2.0:
            bg = bg / 255.0
    else:
        bg = np.array([cfg.bg_value] * 3, dtype=np.float32)

    mask_np = _source_mask_np(ha, H, W, cfg)
    data: Dict[str, Any] = {
        "pp_mode": cfg.pp_mode,
        # untouched crop-space ground truth; in rotate mode `image` / `target_mask` are warped copies
        "gt_image_crop": torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).unsqueeze(0).to(device),
        "gt_mask_crop": None if mask_np is None else torch.from_numpy(mask_np.copy()).to(device)[None, None],
        "model_input_space": "rotated" if cfg.pp_mode == "rotate" else "crop",
    }

    if cfg.pp_mode == "rotate":
        rc = recenter_by_rotation(K33, R, T, H, W)
        K_use, R_use, T_use = rc["K_new"], rc["R_new"], rc["T_new"]
        Hm = rc["H_crop_to_render"]
        Hinv = np.linalg.inv(Hm)                                  # render pixel -> crop pixel
        img = warp_with_homography_np(img, Hinv, (H, W), fill=bg)  # what the rotated camera sees
        if mask_np is not None:
            mask_np = (warp_with_homography_np(mask_np, Hinv, (H, W), fill=0.0) > 0.5).astype(np.float32)
        data["H_crop_to_render"] = torch.from_numpy(Hm.astype(np.float64)).to(device)
        data["H_render_to_crop"] = torch.from_numpy(Hinv.astype(np.float64)).to(device)
        data["pp_rotation_deg"] = rc["angle_deg"]
    elif cfg.pp_mode == "none":
        K_use, R_use, T_use = K33, R, T
    else:
        raise ValueError(f"unknown pp_mode {cfg.pp_mode!r}; use 'rotate' or 'none'")

    intr4 = np.eye(4, dtype=np.float32)
    intr4[:3, :3] = K_use
    intr4_t = torch.from_numpy(intr4).to(device).view(1, 1, 4, 4)
    c2w_t = torch.from_numpy(_w2c_to_c2w(R_use, T_use)).to(device).view(1, 1, 4, 4)
    source_mask = None if mask_np is None else torch.from_numpy(mask_np).to(device)[None, None]
    data["source_mask"] = source_mask
    data["target_mask"] = source_mask

    return {
        "image": torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None, None].to(device),  # [1,1,3,H,W]
        "source_c2ws": c2w_t,
        "source_intrs": intr4_t,
        "render_c2ws": c2w_t,
        "render_intrs": intr4_t,
        "render_bg_colors": torch.from_numpy(bg)[None].to(device),
        "mano_params": mano_params,
        "render_full_resolutions": torch.tensor([[H, W]], dtype=torch.int32, device=device),
        "data": data,
    }