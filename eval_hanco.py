"""Evaluate InstantHand on HanCo (zero-shot).

The canonical avatar is built once from a single anchor image and then driven to
every frame of the HanCo evaluation sequence. PSNR, SSIM and LPIPS follow the
HandAvatar progress protocol.

Pose strata (optional, `pose_strata.enabled`): the same per-frame PSNR / SSIM / LPIPS,
reported in equal-count bins (Easy / Medium / Extreme) of two pose changes relative to
the anchor frame:
    palm  camera-relative palm-orientation change (geodesic distance between the anchor
          and target global hand rotations, each in its own camera frame)
    art   finger-articulation change (mean per-joint geodesic distance over the 15 finger
          joints, global rotation excluded)

Temporal stability (optional, `temporal.enabled`): per camera, consecutive frame pairs
(t, t+1) of the continuous sequence give
    d_pred = mean |pred(t+1) - pred(t)|,  d_gt = mean |gt(t+1) - gt(t)|
over the union of the two ray masks, channel-averaged, on the same composited 8-bit images
the metrics use. Excess = d_pred - d_gt, reported per camera relative to its mean GT change;
the summary over cameras averages the per-camera magnitudes ("Mean of |.|"), so opposite
biases of different cameras do not cancel.

Self-reconstruction (optional, `self_recon.enabled`, a second pass after the results above):
every evaluation frame serves as its own source. An avatar is built from the frame and
rendered back into the same frame, with the same crop, ray mask and metrics.
    w/o transfer  same checkpoint with the appearance head disabled (Stage-1 base colors)
    Ours          the full model (composed colors)

Usage:
    python eval_hanco.py --cfg configs/eval_hanco.yaml                # main + pose strata + temporal
    python eval_hanco.py --cfg configs/eval_hanco.yaml --self_recon   # + self-reconstruction (slow)
"""
from __future__ import annotations

import sys

# HandAvatar's config module parses the command line when it is imported and rejects flags
# it does not know, so our arguments are set aside and it is imported with an empty one.
_CLI_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]

import argparse
import contextlib
import io
import math
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from tqdm import tqdm

from external.handavatar.third_parties.lpips import LPIPS
from external.handavatar.core.utils.metric_util import ssim_metric

import external.handavatar.configs as ha_configs
from external.handavatar.configs import make_cfg as ha_make_cfg, args as ha_args

from instanthand.datasets.handavatar_to_ham_inputs import HA2HAMCfg, handavatar_sample_to_infer_inputs, unwarp_to_crop
from instanthand.models.modeling_ham import ModelHAM
from instanthand.models.checkpoint import load_checkpoint, resolve_encoder_weights
from instanthand.datasets.handavatar_single import (
    preprocess_single_anchor_like_handavatar,
    ANCHOR_IMAGE_SHAPE,
    ANCHOR_BASE_SCALE,
)


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _quiet(enabled: bool = True):
    """Hide stdout prints and Python warnings emitted while third-party objects are built
    (smplx MANO shape-coefficient notice, LPIPS setup, torchvision and lazy-layer warnings)."""
    if not enabled:
        yield
        return
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        yield


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------
def _to_float_tensor(x: Any, device: torch.device) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif not torch.is_tensor(x):
        x = torch.as_tensor(x)
    return x.to(device).float()


def _to_unit_range(x: torch.Tensor) -> torch.Tensor:
    """Map [0, 255] images to [0, 1]; images already in [0, 1] are only clamped."""
    x = x.float()
    if x.numel() > 0 and x.max() > 2.0:
        x = x / 255.0
    return x.clamp(0, 1)


def _to_nchw(rgb: torch.Tensor) -> torch.Tensor:
    """Rendered RGB ([B,V,H,W,3], [B,H,W,3], [B,3,H,W], [H,W,3] or [3,H,W]) -> [B,3,H,W]."""
    t = rgb[:, 0] if rgb.ndim == 5 else rgb
    if t.ndim == 4:
        if t.shape[1] == 3:
            return t.float().clamp(0, 1)
        if t.shape[-1] == 3:
            return t.permute(0, 3, 1, 2).float().clamp(0, 1)
    if t.ndim == 3:
        if t.shape[0] == 3:
            return t.unsqueeze(0).float().clamp(0, 1)
        if t.shape[-1] == 3:
            return t.permute(2, 0, 1).unsqueeze(0).float().clamp(0, 1)
    raise ValueError(f"Unsupported comp_rgb shape: {tuple(rgb.shape)}")


def _unbatch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the DataLoader batch dimension (batch size 1)."""
    for k, v in list(batch.items()):
        if torch.is_tensor(v) and v.ndim >= 1:
            batch[k] = v[0]
        elif isinstance(v, (list, tuple)) and len(v) > 0:
            batch[k] = v[0]
    return batch


def _image_to_nchw(x: torch.Tensor, name: str = "image") -> torch.Tensor:
    """[H,W,3], [3,H,W], [B,H,W,3], [B,3,H,W], [B,V,H,W,3] or [B,V,3,H,W] -> [1,3,H,W] in [0,1]."""
    x = _to_unit_range(x)
    if x.ndim == 5:
        x = x[:, 0]
    if x.ndim == 3:
        if x.shape[0] == 3:
            x = x.unsqueeze(0)
        elif x.shape[-1] == 3:
            x = x.permute(2, 0, 1).unsqueeze(0).contiguous()
        else:
            raise ValueError(f"Unsupported {name} shape: {tuple(x.shape)}")
    elif x.ndim == 4:
        if x.shape[1] != 3:
            if x.shape[-1] != 3:
                raise ValueError(f"Unsupported {name} shape: {tuple(x.shape)}")
            x = x.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(f"Unsupported {name} shape: {tuple(x.shape)}")
    return x[:1].contiguous()


def _mask_to_nchw(x: torch.Tensor, name: str = "mask") -> torch.Tensor:
    """[H,W], [H,W,1|3], [1,H,W], [B,H,W], [B,1,H,W], [B,H,W,1|3], [B,V,1,H,W], [B,V,H,W,1|3]
    -> [1,1,H,W] in [0,1] (first channel of 3-channel masks)."""
    x = _to_unit_range(x)
    if x.ndim == 5:
        x = x[:, 0]
        if x.shape[1] != 1:
            if x.shape[-1] not in (1, 3):
                raise ValueError(f"Unsupported {name} shape after view squeeze: {tuple(x.shape)}")
            x = x[..., :1].permute(0, 3, 1, 2).contiguous()
    elif x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        if x.shape[0] == 1:
            x = x.unsqueeze(0)
        elif x.shape[-1] in (1, 3):
            x = x[..., :1].permute(2, 0, 1).unsqueeze(0).contiguous()
        else:  # [B,H,W]
            x = x[:1].unsqueeze(1)
    elif x.ndim == 4:
        if x.shape[1] != 1:
            if x.shape[-1] not in (1, 3):
                raise ValueError(f"Unsupported {name} shape: {tuple(x.shape)}")
            x = x[..., :1].permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(f"Unsupported {name} shape: {tuple(x.shape)}")
    return x[:1].contiguous()


def _eval_targets(batch: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ray mask [H*W] (bool) and GT colors of the masked pixels [K,3].

    Uses the loader's ray samples ('ray_mask', 'target_rgbs') when present. Otherwise the
    full target image is used, restricted to the target (or source) hand mask.
    """
    if "ray_mask" in batch and "target_rgbs" in batch:
        ray_mask = batch["ray_mask"]
        if isinstance(ray_mask, np.ndarray):
            ray_mask = torch.from_numpy(ray_mask)
        ray_mask = (ray_mask.to(device) > 0).reshape(-1)
        return ray_mask, _to_unit_range(_to_float_tensor(batch["target_rgbs"], device))

    if "target_img" not in batch:
        raise KeyError("Evaluation needs ('ray_mask', 'target_rgbs') or 'target_img' in each batch.")

    target_img = _image_to_nchw(_to_float_tensor(batch["target_img"], device), name="target_img")
    H, W = target_img.shape[-2:]

    mask = None
    for key in ("target_alpha_img", "target_mask", "source_mask"):
        if key in batch:
            mask = _mask_to_nchw(_to_float_tensor(batch[key], device), name=key)
            break
    if mask is None:
        ray_mask = torch.ones((H * W,), dtype=torch.bool, device=device)
    else:
        if mask.shape[-2:] != (H, W):
            mask = F.interpolate(mask.float(), size=(H, W), mode="nearest")
        ray_mask = mask[0, 0].reshape(-1) > 0.5

    target_rgbs = target_img[0].permute(1, 2, 0).reshape(H * W, 3)[ray_mask]
    return ray_mask, target_rgbs


# ---------------------------------------------------------------------------
# Metrics (HandAvatar progress protocol)
# ---------------------------------------------------------------------------
# Paper tables report PSNR to 2 decimals and SSIM / LPIPS to 3 decimals, rounded directly
# from the unrounded mean (no intermediate 4-decimal step).
PAPER_DIGITS = {"PSNR": 2, "SSIM": 3, "LPIPS": 3}


def paper_round(value: float, digits: int) -> str:
    """Round the unrounded mean to `digits` decimals (e.g. 0.188470 -> 0.188)."""
    return f"{value:.{digits}f}"


@torch.no_grad()
def compute_metrics(
    pred_img01: torch.Tensor,
    gt_target_rgbs: torch.Tensor,
    ray_mask_flat: torch.Tensor,
    H: int,
    W: int,
    bgcolor01: torch.Tensor,
    lpips_model,
    return_images: bool = False,
):
    """PSNR / SSIM / LPIPS over the full H x W image.

    Both the prediction and the ground truth keep their values only inside the ray
    mask and are filled with the background color elsewhere. They are quantized to
    8 bits before the metrics are computed. With return_images, the composited 8-bit
    prediction and ground truth (H x W x 3 numpy) are returned as well.
    """
    device = pred_img01.device
    assert pred_img01.shape[0] == 1
    assert pred_img01.shape[-2:] == (H, W)

    if ray_mask_flat.dtype != torch.bool:
        ray_mask_flat = ray_mask_flat > 0.5
    ray_mask_flat = ray_mask_flat.view(-1)

    bg_full = bgcolor01.view(1, 3).to(device).expand(H * W, 3).contiguous()
    gt_full = bg_full.clone()
    pr_full = bg_full.clone()

    pred_flat = pred_img01[0].permute(1, 2, 0).reshape(H * W, 3)
    gt_full[ray_mask_flat] = gt_target_rgbs.to(device)
    pr_full[ray_mask_flat] = pred_flat[ray_mask_flat]

    gt_uint8 = (gt_full.reshape(H, W, 3).clamp(0, 1) * 255.0 + 0.5).to(torch.uint8)
    pr_uint8 = (pr_full.reshape(H, W, 3).clamp(0, 1) * 255.0 + 0.5).to(torch.uint8)
    gt_img = (gt_uint8.float() / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous()
    pr_img = (pr_uint8.float() / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous()

    mse = ((pr_img - gt_img) ** 2).mean().clamp_min(1e-8)
    psnr = float((20.0 * torch.log10(1.0 / torch.sqrt(mse))).cpu().item())
    ssim = float(ssim_metric(pr_img, gt_img, val_range=1.0).cpu().item())

    pr_lp = (pr_img * 2 - 1).clamp(-1, 1)
    gt_lp = (gt_img * 2 - 1).clamp(-1, 1)
    lpips = float(lpips_model(pr_lp, gt_lp).view(-1)[0].cpu().item())

    if return_images:
        return psnr, ssim, lpips, pr_uint8.cpu().numpy(), gt_uint8.cpu().numpy()
    return psnr, ssim, lpips


# ---------------------------------------------------------------------------
# Pose strata (pose change of each frame relative to the anchor; numpy only)
# ---------------------------------------------------------------------------
POSE_ANALYSES = (  # (measure key, printed title)
    ("palm_deg", "Palm-orientation change vs. anchor"),
    ("art_deg", "Finger-articulation change vs. anchor"),
)


def _flat64(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def aa_to_mat(aa: np.ndarray) -> np.ndarray:
    """[N,3] axis-angle -> [N,3,3] (Rodrigues)."""
    aa = np.asarray(aa, dtype=np.float64).reshape(-1, 3)
    theta = np.linalg.norm(aa, axis=1)
    k = aa / np.maximum(theta, 1e-12)[:, None]
    K = np.zeros((len(aa), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = np.sin(theta)[:, None, None], (1 - np.cos(theta))[:, None, None]
    R = np.eye(3)[None] + s * K + c * (K @ K)
    R[theta < 1e-10] = np.eye(3)
    return R


def _is_rotation(R: np.ndarray, tol: float = 1e-3) -> bool:
    if R.ndim < 2 or R.shape[-2:] != (3, 3):
        return False
    orth = np.abs(R @ np.swapaxes(R, -1, -2) - np.eye(3)).max()
    return orth < tol and np.abs(np.linalg.det(R) - 1.0).max() < tol


def mano_rotations(x: Any, n_joints: int, rot_repr: str = "auto") -> Optional[np.ndarray]:
    """MANO rotation field -> [n_joints,3,3] (first batch/view element).
    Axis-angle (n_joints*3 values) or matrices (n_joints*9); when both layouts divide the
    count, rot_repr decides, otherwise orthonormality does."""
    t = _flat64(x)
    if t is None or t.size == 0 or n_joints <= 0:
        return None
    fits_aa, fits_mat = t.size % (n_joints * 3) == 0, t.size % (n_joints * 9) == 0

    def as_aa():
        return aa_to_mat(t.reshape(-1, n_joints, 3)[0])

    def as_mat():
        return t.reshape(-1, n_joints, 3, 3)[0]

    if rot_repr == "aa" and fits_aa:
        return as_aa()
    if rot_repr == "matrix" and fits_mat:
        return as_mat()
    if fits_aa and not fits_mat:
        return as_aa()
    if fits_mat and not fits_aa:
        return as_mat()
    if fits_mat and _is_rotation(as_mat()):
        return as_mat()
    return as_aa() if fits_aa else None


def _c2w_rotation(x: Any) -> Optional[np.ndarray]:
    """Rotation block of the first camera-to-world matrix (4x4 or 3x4)."""
    t = _flat64(x)
    if t is None:
        return None
    if t.size % 16 == 0:
        return t.reshape(-1, 4, 4)[0, :3, :3]
    if t.size % 12 == 0:
        return t.reshape(-1, 3, 4)[0, :3, :3]
    return None


def geodesic_deg(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    Rrel = np.swapaxes(R1, -1, -2) @ R2
    cos = np.clip((np.trace(Rrel, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def pose_raw(inputs: Dict[str, Any], cam_R: Any = None, rot_repr: str = "auto", n_joints: int = 15) -> Dict[str, Any]:
    """Global rotation, finger rotations and camera rotation of one frame.

    cam_R: world-to-camera rotation of the TRUE camera (the loader's 'cam_R'). When given, the
    camera rotation comes from it (c2w rotation = cam_R^T) instead of render_c2ws, which may be
    the principal-point-centering rotated camera of the adapter.
    """
    mano = inputs.get("mano_params", {}) or {}
    t = _flat64(cam_R)
    R_c2w = t[:9].reshape(3, 3).T if t is not None and t.size >= 9 else _c2w_rotation(inputs.get("render_c2ws"))
    R_glob = mano_rotations(mano.get("global_orient"), 1, rot_repr)
    return {
        "R_glob": None if R_glob is None else R_glob[0],
        "R_art": mano_rotations(mano.get("hand_pose"), n_joints, rot_repr),
        "R_c2w": R_c2w,
    }


def _parse_cam_stem(frame_name: Any) -> Tuple[str, str]:
    """'.../cam400272/image15012(.jpg)' -> ('cam400272', 'image15012'); no cam part -> 'all'."""
    s = (frame_name.decode("utf-8") if isinstance(frame_name, bytes) else str(frame_name)).strip().replace("\\", "/")
    for ext in (".jpg", ".jpeg", ".png"):
        s = s.removesuffix(ext)
    parts = [p for p in s.split("/") if p]
    cam = next((p for p in parts if p.lower().startswith("cam")), "all")
    stem = parts[-1] if parts else "unknown"
    return cam, stem


def detect_global_frame(records: List[Dict[str, Any]], tol_deg: float = 1.0) -> Tuple[str, str]:
    """Is global_orient already camera-relative, or world-frame with per-camera extrinsics?
    Compare the same time frame seen by different cameras."""
    by_stem: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r["_pose"]["R_glob"] is not None:
            by_stem[r["stem"]].append(r)
    glob_d, cam_d = [], []
    for rs in by_stem.values():
        base = rs[0]
        for r in rs[1:]:
            if r["cam"] == base["cam"]:
                continue
            glob_d.append(float(geodesic_deg(base["_pose"]["R_glob"], r["_pose"]["R_glob"])))
            if base["_pose"]["R_c2w"] is not None and r["_pose"]["R_c2w"] is not None:
                cam_d.append(float(geodesic_deg(base["_pose"]["R_c2w"], r["_pose"]["R_c2w"])))
    if not glob_d:
        # no frame is seen by two cameras: identity cameras mean global_orient is already
        # camera-relative, non-identity cameras mean it must be composed
        rots = [r["_pose"]["R_c2w"] for r in records if r["_pose"]["R_c2w"] is not None]
        off = float(np.max([geodesic_deg(np.eye(3), R) for R in rots])) if rots else 0.0
        if off <= tol_deg:
            return "camera", "one camera per frame, identity cameras: global rotations already camera-relative"
        return "world", (f"one camera per frame, non-identity cameras (up to {off:.1f} deg): "
                         f"composed with the world-to-camera rotation")
    g = float(np.median(glob_d))
    c = float(np.median(cam_d)) if cam_d else 0.0
    if g > tol_deg:
        msg = f"global_orient differs across cameras of the same frame (median {g:.1f} deg): already camera-relative"
        if c > tol_deg:
            msg += f"  [warning: cameras also differ (median {c:.1f} deg); check the loader's convention]"
        return "camera", msg
    if c > tol_deg:
        return "world", (f"global_orient identical across cameras (median {g:.2f} deg) while the cameras differ "
                         f"(median {c:.1f} deg): composed with the world-to-camera rotation")
    return "camera", f"global_orient and cameras both identical across views (median {g:.2f} / {c:.2f} deg)"


def compute_pose_measures(rec_pose: Dict[str, Any], src_pose: Dict[str, Any], frame: str) -> Dict[str, float]:
    out = {"palm_deg": float("nan"), "view_hand_deg": float("nan"), "art_deg": float("nan")}

    def cam_rel(p):
        if p["R_glob"] is None:
            return None
        if frame == "world":
            return None if p["R_c2w"] is None else p["R_c2w"].T @ p["R_glob"]
        return p["R_glob"]

    Rs, Rt = cam_rel(src_pose), cam_rel(rec_pose)
    if Rs is not None and Rt is not None:
        out["palm_deg"] = float(geodesic_deg(Rs, Rt))
        # camera viewing direction expressed in the hand frame (roll about the optical axis ignored)
        vs, vt = Rs.T @ np.array([0.0, 0.0, 1.0]), Rt.T @ np.array([0.0, 0.0, 1.0])
        out["view_hand_deg"] = float(np.degrees(np.arccos(np.clip(vs @ vt, -1.0, 1.0))))
    if rec_pose["R_art"] is not None and src_pose["R_art"] is not None:
        out["art_deg"] = float(geodesic_deg(src_pose["R_art"], rec_pose["R_art"]).mean())
    return out


def equal_count_bins(values: np.ndarray, n_bins: int = 3) -> List[np.ndarray]:
    idx = np.where(np.isfinite(values))[0]
    order = idx[np.argsort(values[idx], kind="mergesort")]
    return [np.asarray(s, dtype=int) for s in np.array_split(order, n_bins)]


def bin_names(n_bins: int) -> List[str]:
    return ["Easy", "Medium", "Extreme"] if n_bins == 3 else [f"Bin {i + 1}" for i in range(n_bins)]


def print_pose_strata(rows: List[Tuple[float, float, float]], pose_records: List[Dict[str, Any]],
                      src_pose: Dict[str, Any], pcfg: Dict[str, Any], verbose: bool = False) -> None:
    """PSNR / SSIM / LPIPS of the frames in each equal-count bin of the two pose changes,
    in the layout of the supplementary table (Tercile, Range in degrees, metrics; All last)."""
    n_bins = int(pcfg.get("bins", 3))
    frame = str(pcfg.get("global_frame", "auto"))
    info = f"forced '{frame}'"
    if frame == "auto":
        frame, info = detect_global_frame(pose_records)
    measures = [compute_pose_measures(r["_pose"], src_pose, frame) for r in pose_records]
    metrics = np.array(rows, dtype=np.float64)  # [N, 3] = PSNR, SSIM, LPIPS

    def fmt(m: np.ndarray) -> str:
        psnr, ssim, lpips = m
        return (f"{paper_round(psnr, PAPER_DIGITS['PSNR']):>7s} {paper_round(ssim, PAPER_DIGITS['SSIM']):>7s} "
                f"{paper_round(lpips, PAPER_DIGITS['LPIPS']):>7s}")

    print("-------- pose strata --------")
    if verbose:
        print(f"global rotations: {info}")
    print(f"{'Tercile' if n_bins == 3 else 'Bin':<9s} {'Range':>9s} {'PSNR':>7s} {'SSIM':>7s} {'LPIPS':>7s}"
          + ("    n  view (median)" if verbose else ""))
    for key, title in POSE_ANALYSES:
        vals = np.array([m[key] for m in measures], dtype=np.float64)
        views = np.array([m["view_hand_deg"] for m in measures], dtype=np.float64)
        print(title)
        for name, sel in zip(bin_names(n_bins), equal_count_bins(vals, n_bins)):
            if len(sel) == 0:
                continue
            rng = f"{float(vals[sel].min()):.0f}-{float(vals[sel].max()):.0f}"
            extra = ""
            if verbose:  # frame count; for the palm bins, the median viewing-direction change in the hand frame
                extra = f" {len(sel):4d}" + (f"  {float(np.nanmedian(views[sel])):.0f} deg" if key == "palm_deg" else "")
            print(f"{name:<9s} {rng:>9s} {fmt(metrics[sel].mean(axis=0))}" + extra)
        n_missing = int((~np.isfinite(vals)).sum())
        if n_missing:
            print(f"[warn] {n_missing} frame(s) without a pose measure are left out of the bins")
    print(f"{'All':<9s} {'':>9s} {fmt(metrics.mean(axis=0))}" + (f" {len(metrics):4d}" if verbose else ""))


# ---------------------------------------------------------------------------
# Temporal stability (consecutive frames per camera; numpy only)
# ---------------------------------------------------------------------------
def _stem_key(s: str):
    return (0, int(s), s) if s.isdigit() else (1, 0, s)


def _frame_change(x: np.ndarray, y: np.ndarray, m: np.ndarray) -> float:
    """mean |y - x| over mask m, channel-averaged, in [0,1] intensity."""
    return float(np.abs(y.astype(np.float32) - x.astype(np.float32)).mean(axis=2)[m].mean() / 255.0)


def temporal_pairs(tstore: Dict[str, Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Consecutive pairs per camera, in frame order (sorted here, whatever the loader order)."""
    rows = []
    for cam in sorted(tstore):
        stems = sorted(tstore[cam], key=_stem_key)
        for a, b in zip(stems[:-1], stems[1:]):
            A, B = tstore[cam][a], tstore[cam][b]
            m = A["mask"] | B["mask"]
            if not m.any():
                continue
            rows.append({"cam": cam,
                         "gap": (int(b) - int(a)) if a.isdigit() and b.isdigit() else None,
                         "d_gt": _frame_change(A["gt"], B["gt"], m),
                         "d_pred": _frame_change(A["pred"], B["pred"], m)})
    return rows


FLICKER_STRENGTHS = (0.0, 0.01, 0.02, 0.04)


def flicker_check(tstore: Dict[str, Dict[str, Dict[str, Any]]], strengths=FLICKER_STRENGTHS,
                  seed: int = 0) -> List[Tuple[float, float]]:
    """Calibration: add a per-frame brightness offset drawn uniformly from [-s, s] (inside the
    ray mask, re-quantized to 8 bits) to our renders and recompute the excess / GT change
    pooled over all pairs. It must increase with s. Returns [(s, excess/GT %)]."""
    out = []
    for sgm in strengths:
        rng = np.random.default_rng(seed)
        jit = {}
        for cam in sorted(tstore):
            for st in sorted(tstore[cam], key=_stem_key):
                fr = tstore[cam][st]
                x = fr["pred"].astype(np.float32) / 255.0
                if sgm > 0:
                    x = x + fr["mask"][..., None] * rng.uniform(-sgm, sgm)
                jit[(cam, st)] = (np.clip(x, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        ex, gt = [], []
        for cam in sorted(tstore):
            stems = sorted(tstore[cam], key=_stem_key)
            for a, b in zip(stems[:-1], stems[1:]):
                m = tstore[cam][a]["mask"] | tstore[cam][b]["mask"]
                if not m.any():
                    continue
                dg = _frame_change(tstore[cam][a]["gt"], tstore[cam][b]["gt"], m)
                ex.append(_frame_change(jit[(cam, a)], jit[(cam, b)], m) - dg)
                gt.append(dg)
        out.append((float(sgm), 100.0 * float(np.mean(ex)) / float(np.mean(gt)) if gt else float("nan")))
    return out


def print_temporal(rows: List[Dict[str, Any]], source_cam: str,
                   flicker: Optional[List[Tuple[float, float]]] = None) -> None:
    """Per camera: pairs, mean GT change, mean prediction change, excess / GT change (%);
    then the mean of the per-camera magnitudes."""
    print("-------- temporal stability (consecutive frames per camera) --------")
    if not rows:
        print("no consecutive frame pairs")
        return
    gaps = sorted({r["gap"] for r in rows if r["gap"] is not None})
    if gaps and gaps != [1]:
        print(f"[warn] consecutive evaluation frames are not all one frame apart (steps {gaps})")

    print(f"{'Camera':<20s} {'pairs':>5s} {'GT change':>10s} {'Pred':>8s} {'Excess/GT':>10s}")
    per_cam = []
    for cam in sorted({r["cam"] for r in rows}):
        sel = [r for r in rows if r["cam"] == cam]
        dg = np.array([r["d_gt"] for r in sel], dtype=np.float64)
        dp = np.array([r["d_pred"] for r in sel], dtype=np.float64)
        gt, pred, excess = float(dg.mean()), float(dp.mean()), float((dp - dg).mean())
        ratio = 100.0 * excess / gt if gt > 0 else float("nan")
        per_cam.append({"n": len(sel), "gt": gt, "pred": pred, "ratio": ratio})
        label = f"{cam} (source view)" if cam == source_cam else cam
        print(f"{label:<20s} {len(sel):5d} {gt:10.4f} {pred:8.4f} {ratio:+9.1f}%")

    n = sum(c["n"] for c in per_cam)
    gt = float(np.mean([c["gt"] for c in per_cam]))
    pred = float(np.mean([c["pred"] for c in per_cam]))
    ratio = float(np.mean([abs(c["ratio"]) for c in per_cam]))
    print(f"{'Mean of |.|':<20s} {n:5d} {gt:10.4f} {pred:8.4f} {ratio:9.1f}%")

    dg = np.array([r["d_gt"] for r in rows], dtype=np.float64)
    dp = np.array([r["d_pred"] for r in rows], dtype=np.float64)
    print(f"{'pooled (signed)':<20s} {len(rows):5d} {dg.mean():10.4f} {dp.mean():8.4f} "
          f"{100.0 * (dp - dg).mean() / dg.mean():+9.1f}%")

    if flicker:
        mono = all(b[1] > a[1] for a, b in zip(flicker[:-1], flicker[1:]))
        print("flicker calibration (pooled excess/GT with a per-frame brightness offset of +-s on our renders):")
        print("  " + ", ".join(f"s={sg:g}: {r:+.1f}%" for sg, r in flicker)
              + ("  (increases monotonically)" if mono else "  [warn] NOT monotonic, check the pairing"))


def _camera_of(frame_name: Any) -> str:
    """'0191/cam3/00000000.jpg' or 'cam6_00000075' -> 'cam3' / 'cam6'."""
    m = re.search(r"cam\d+", str(frame_name))
    return m.group(0) if m else "all"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model(model_cfg: Dict[str, Any], ckpt_dir: str, device: torch.device, quiet: bool = True) -> ModelHAM:
    ckpt_path = Path(ckpt_dir) / "model.safetensors"
    with _quiet(quiet):
        model = ModelHAM(**resolve_encoder_weights(model_cfg, ckpt_path))
    load_checkpoint(model, ckpt_path, device)
    model = model.to(device)
    model.set_train_stage("stage2")
    model.eval()
    return model


def canonical_only_model(full_model: ModelHAM, model_cfg: Dict[str, Any], ckpt_dir: str, device: torch.device,
                         train_stage: str, quiet: bool = True) -> ModelHAM:
    """Same checkpoint with the appearance head disabled: renders the Stage-1 base colors.

    Built without the head and filled from the already loaded full model (every tensor except
    the head's), so it works for any checkpoint format the full model loads from.
    """
    cfg = dict(model_cfg)
    cfg["use_appearance_head"] = False
    cfg["appearance_head_cfg"] = None
    with _quiet(quiet):
        model = ModelHAM(**resolve_encoder_weights(cfg, Path(ckpt_dir) / "model.safetensors"))
    head = "renderer.appearance_head."
    state = {k: v for k, v in full_model.state_dict().items() if not k.startswith(head)}
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.set_train_stage(train_stage)
    model.eval()
    return model


@torch.no_grad()
def build_canonical_avatars(models: List[ModelHAM], source_inputs: Dict[str, Any], device: torch.device):
    """Source image -> canonical Gaussian avatar, for each model in `models`.

    The image encoding and the latent points come from the first model and are shared: the
    models given together differ only in the appearance head (the canonical-only model is a
    copy of the full model without it), so the encoder and transformer outputs are identical
    and need to be computed once. Returns [(gs_models, query_points)] in the order of `models`.
    """
    model = models[0]
    image = source_inputs["image"]
    if image.ndim == 5:  # [B, V, 3, H, W] -> first view
        image = image[:, 0]

    mano = source_inputs["mano_params"]
    query_points, mano_params = model.renderer.get_query_points_from_params(
        mano_params=mano,
        device=device,
    )

    latent_points, image_feats = model.forward_latent_points(
        image=image,
        camera=None,
        query_points=query_points,
    )
    side = int(math.sqrt(image_feats.shape[1]))
    if side * side != image_feats.shape[1]:
        raise ValueError(f"image_feats tokens do not form a square grid: {image_feats.shape[1]}")
    image_feats_bchw = rearrange(image_feats, "b (h w) c -> b c h w", h=side)

    data = source_inputs.get("data") or {}
    source_intrs = source_inputs["source_intrs"]
    source_c2ws = source_inputs["source_c2ws"]
    additional_features = {
        "image_feats": image_feats,
        "image": image,
        "image_feats_bchw": image_feats_bchw,
        "source_image": image,
        "source_mask": data.get("source_mask"),
        "source_intrinsic": source_intrs[:, 0] if source_intrs.ndim == 4 else source_intrs,
        "source_c2w": source_c2ws[:, 0] if source_c2ws.ndim == 4 else source_c2ws,
        "source_depth_map": None,
        "source_mano_data": {k: mano[k] for k in ("betas", "global_orient", "hand_pose", "translation")},
    }

    avatars = []
    for m in models:
        gs_models, qp, *_ = m.renderer.build_gs(
            gs_hidden_features=latent_points,
            query_points=query_points,
            mano_params=mano_params,
            additional_features=additional_features,
            train_stage=m.current_train_stage,
        )
        avatars.append((gs_models, qp))
    return avatars


def build_canonical_avatar(model: ModelHAM, source_inputs: Dict[str, Any], device: torch.device):
    """Single forward pass: source image -> canonical Gaussian avatar."""
    return build_canonical_avatars([model], source_inputs, device)[0]


@torch.no_grad()
def render_frame(model: ModelHAM, gs_models, query_points, inputs: Dict[str, Any], H: int, W: int) -> torch.Tensor:
    """Drive the canonical avatar to a target frame's pose and camera -> [1,3,H,W] in [0,1]."""
    out = model.renderer.render_gs(
        gs_models=gs_models,
        query_points=query_points,
        view_mano=inputs["mano_params"],
        c2w=inputs["render_c2ws"],
        intrinsic=inputs["render_intrs"],
        height=H,
        width=W,
        background_color=inputs["render_bg_colors"],
        debug=False,
        train_stage=model.current_train_stage,
    )
    pred = _to_nchw(out["comp_rgb"])
    if tuple(pred.shape[-2:]) != (H, W):
        pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)
    return pred


def to_crop(pred: torch.Tensor, inputs: Dict[str, Any]) -> torch.Tensor:
    """Map a render back to the image crop the ground truth (ray mask, target colors) lives in.

    The adapter renders from a camera rotated so that its principal point is at the image
    center (the rasterizer ignores cx, cy); `H_crop_to_render` maps crop pixels into that
    render. Bicubic resampling, background color where the rotated view has no pixels.
    """
    Hm = inputs["data"].get("H_crop_to_render")
    if Hm is None:  # camera not rotated (pp_mode "none")
        return pred
    bg = inputs["render_bg_colors"].reshape(-1)[:3].float().cpu().tolist()  # [1,3] -> [3]
    return unwarp_to_crop(pred, Hm, fill=bg, mode="bicubic", clamp=(0.0, 1.0))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_eval(cfg: Dict[str, Any]) -> None:
    # HandAvatar modules read a module-level cfg, so set it before creating the dataloader.
    ha_args.cfg = cfg["handavatar_cfg"]
    progress_cfg = ha_make_cfg(ha_args)
    ha_configs.cfg = progress_cfg

    quiet = not cfg.get("verbose", False)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = load_model(cfg["model"], cfg["ckpt_dir"], device, quiet=quiet)
    with _quiet(quiet):
        lpips_model = LPIPS(net="vgg").eval().to(device)

    bg_rgb255 = np.array(progress_cfg.bgcolor, dtype=np.float32)
    bg01 = torch.tensor(bg_rgb255 / 255.0, device=device)

    # Canonical avatar from the anchor image
    anchor = preprocess_single_anchor_like_handavatar(
        anchor_image=cfg["anchor_image"],
        anchor_mask=cfg["anchor_mask"],
        anchor_pkl=cfg["anchor_pkl"],
        bgcolor_rgb255=bg_rgb255,
        input_img_shape=ANCHOR_IMAGE_SHAPE,
        base_scale=ANCHOR_BASE_SCALE,
    )
    # principal point: "rotate" (exact, default) or "none" (only for a rasterizer that honours cx, cy)
    adapter_cfg = HA2HAMCfg(pp_mode=cfg.get("pp_mode", "rotate"))
    anchor_inputs = handavatar_sample_to_infer_inputs(anchor, device, adapter_cfg)
    gs_models, query_points = build_canonical_avatar(model, anchor_inputs, device)

    from external.handavatar.core.data.create_dataset import create_dataloader  # needs ha_configs.cfg

    with _quiet(quiet):
        loader = create_dataloader(data_type="progress")

    pcfg = cfg.get("pose_strata") or {}
    with_pose = bool(pcfg.get("enabled", False))
    src_pose = pose_raw(anchor_inputs, cam_R=anchor.get("cam_R")) if with_pose else None

    with_temporal = bool((cfg.get("temporal") or {}).get("enabled", False))
    tstore: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)  # cam -> frame -> 8-bit images + mask

    max_batches = int(cfg["max_batches"])
    rows = []  # (psnr, ssim, lpips) per frame
    pose_records = []  # pose of each frame, for the pose strata
    for frame_index, batch in enumerate(tqdm(loader, desc="eval")):
        if max_batches > 0 and frame_index >= max_batches:
            break
        batch = _unbatch(batch)
        H, W = int(batch["img_height"]), int(batch["img_width"])

        ray_mask, target_rgbs = _eval_targets(batch, device)

        inputs = handavatar_sample_to_infer_inputs(batch, device, adapter_cfg)
        pred = to_crop(render_frame(model, gs_models, query_points, inputs, H, W), inputs)

        psnr, ssim, lpips, pred_u8, gt_u8 = compute_metrics(pred, target_rgbs, ray_mask, H, W, bg01, lpips_model,
                                                            return_images=True)
        rows.append((psnr, ssim, lpips))

        frame_name = batch.get("frame_name", f"{frame_index:06d}")
        cam, stem = _parse_cam_stem(frame_name)
        if with_pose:
            pose_records.append({"cam": cam, "stem": stem, "_pose": pose_raw(inputs, cam_R=batch.get("cam_R"))})
        if with_temporal:
            if stem in tstore[cam]:
                raise RuntimeError(f"[temporal] frame {cam}/{stem} appears twice in the evaluation set")
            tstore[cam][stem] = {"pred": pred_u8, "gt": gt_u8, "mask": ray_mask.view(H, W).cpu().numpy()}

    # Mean over frames, rounded as in the paper tables: main, pose strata, temporal
    print(f"\n======== HanCo evaluation ({len(rows)} frames) ========")
    print("-------- main --------")
    for col, name in enumerate(("PSNR", "SSIM", "LPIPS")):
        mean = float(np.array([r[col] for r in rows], dtype=np.float64).mean())
        line = f"{name:<5}: {paper_round(mean, PAPER_DIGITS[name])}"
        print(line + (f"  (unrounded {mean:.6f})" if cfg.get("verbose") else ""))

    if with_pose:
        print_pose_strata(rows, pose_records, src_pose, pcfg, verbose=bool(cfg.get("verbose")))

    if with_temporal:
        tcfg = cfg.get("temporal") or {}
        flicker = flicker_check(tstore) if bool(tcfg.get("flicker_check", True)) else None
        print_temporal(temporal_pairs(tstore), source_cam=_camera_of(anchor.get("frame_name", cfg["anchor_image"])),
                       flicker=flicker)
        del tstore

    # Self-reconstruction: a second pass over the test frames, after the results above are printed
    scfg = cfg.get("self_recon") or {}
    if bool(scfg.get("enabled", False)):
        del gs_models, query_points
        sys.stdout.flush()  # show the results above before the second pass starts
        run_self_recon(cfg, scfg, model, lpips_model, bg01, adapter_cfg, device, quiet)


@torch.no_grad()
def run_self_recon(cfg: Dict[str, Any], scfg: Dict[str, Any], model: ModelHAM, lpips_model, bg01: torch.Tensor,
                   adapter_cfg: HA2HAMCfg, device: torch.device, quiet: bool) -> None:
    """Every test frame is its own source: build the avatar from the frame, render it back into
    the same frame. 'w/o transfer' = same checkpoint without the appearance head, 'Ours' = full
    model; one image encoding per frame is shared by the two."""
    from external.handavatar.core.data.create_dataset import create_dataloader  # needs ha_configs.cfg

    canon_model = canonical_only_model(model, cfg["model"], cfg["ckpt_dir"], device,
                                       train_stage=str(scfg.get("canonical_train_stage", "stage2")), quiet=quiet)
    with _quiet(quiet):
        loader = create_dataloader(data_type="progress")

    max_batches = int(cfg["max_batches"])
    self_rows = {"w/o transfer": [], "Ours": []}  # (psnr, ssim, lpips) per frame
    for frame_index, batch in enumerate(tqdm(loader, desc="self-recon")):
        if max_batches > 0 and frame_index >= max_batches:
            break
        batch = _unbatch(batch)
        H, W = int(batch["img_height"]), int(batch["img_width"])
        ray_mask, target_rgbs = _eval_targets(batch, device)

        inputs = handavatar_sample_to_infer_inputs(batch, device, adapter_cfg)
        avatars = build_canonical_avatars([model, canon_model], inputs, device)
        for (setting, m), (gs_self, qp_self) in zip((("Ours", model), ("w/o transfer", canon_model)), avatars):
            pred = to_crop(render_frame(m, gs_self, qp_self, inputs, H, W), inputs)
            self_rows[setting].append(compute_metrics(pred, target_rgbs, ray_mask, H, W, bg01, lpips_model))
        del avatars

    print("-------- self-reconstruction (each frame its own source) --------")
    print(f"{'':<13s} {'PSNR':>7s} {'SSIM':>7s} {'LPIPS':>7s}")
    for setting, r in self_rows.items():
        psnr, ssim, lpips = np.array(r, dtype=np.float64).mean(axis=0)
        print(f"{setting + ':':<13s} {paper_round(psnr, PAPER_DIGITS['PSNR']):>7s} "
              f"{paper_round(ssim, PAPER_DIGITS['SSIM']):>7s} {paper_round(lpips, PAPER_DIGITS['LPIPS']):>7s}")


def parse_args() -> Dict[str, Any]:
    p = argparse.ArgumentParser(description="Evaluate InstantHand on HanCo.")
    p.add_argument("--cfg", default="configs/eval_hanco.yaml")
    p.add_argument("--ckpt_dir", help="override ckpt_dir in the config")
    p.add_argument("--max_batches", type=int, help="override max_batches in the config (-1: all frames)")
    p.add_argument("--pp_mode", choices=["rotate", "none"], help="override pp_mode in the config")
    p.add_argument("--no_pose_strata", action="store_true", help="skip the pose-stratified metrics")
    p.add_argument("--no_temporal", action="store_true", help="skip the temporal stability metrics")
    p.add_argument("--self_recon", action="store_true", help="also run the self-reconstruction metrics (slow)")
    p.add_argument("--no_self_recon", action="store_true", help="skip the self-reconstruction metrics")
    p.add_argument("--verbose", action="store_true", help="show logs and warnings from model, LPIPS and data setup")
    args = p.parse_args(_CLI_ARGV)

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("ckpt_dir", "max_batches", "pp_mode"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.no_pose_strata:
        cfg.setdefault("pose_strata", {})["enabled"] = False
    if args.no_temporal:
        cfg.setdefault("temporal", {})["enabled"] = False
    if args.self_recon:
        cfg.setdefault("self_recon", {})["enabled"] = True
    if args.no_self_recon:
        cfg.setdefault("self_recon", {})["enabled"] = False
    cfg["verbose"] = args.verbose
    return cfg


if __name__ == "__main__":
    run_eval(parse_args())