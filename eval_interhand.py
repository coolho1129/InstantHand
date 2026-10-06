"""Evaluate InstantHand on InterHand2.6M.

The canonical avatar is built once from a single anchor image and then driven to
every frame of the HandAvatar test split. PSNR, SSIM and LPIPS follow the
HandAvatar progress protocol. Boundary metrics (boundary F-score, boundary IoU, alpha
leakage, interior coverage) compare the rendered alpha with the GT hand mask.

Pose strata (optional, `pose_strata.enabled`): the same per-frame PSNR / SSIM / LPIPS,
reported in equal-count bins (Easy / Medium / Extreme) of two pose changes relative to
the anchor frame:
    palm  camera-relative palm-orientation change (geodesic distance between the anchor
          and target global hand rotations, each in its own camera frame)
    art   finger-articulation change (mean per-joint geodesic distance over the 15 finger
          joints, global rotation excluded)

Self-reconstruction (optional, `self_recon.enabled`): every evaluation frame serves as its
own source. An avatar is built from the frame and rendered back into the same frame (its pose
and camera), with the same crop, ray mask and metrics; the frame metrics are averaged.
    w/o transfer  same checkpoint with the appearance head disabled (Stage-1 base colors)
    Ours          the full model (composed colors)
All non-color Gaussian attributes are identical between the two settings.

Usage:
    python eval_interhand.py --cfg configs/eval_interhand.yaml                # main + boundary + pose strata
    python eval_interhand.py --cfg configs/eval_interhand.yaml --self_recon   # + self-reconstruction (slow)
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
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from tqdm import tqdm

from external.handavatar.third_parties.lpips import LPIPS
from external.handavatar.core.utils.metric_util import ssim_metric

import external.handavatar.configs as ha_configs
from external.handavatar.configs import make_cfg as ha_make_cfg, args as ha_args

from instanthand.datasets.handavatar_to_ham_inputs import handavatar_sample_to_infer_inputs, unwarp_to_crop
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


def _alpha_to_nchw(alpha: torch.Tensor) -> torch.Tensor:
    """Rendered alpha ([B,V,H,W,1], [B,V,1,H,W], [B,H,W,1], [B,1,H,W] or [B,H,W]) -> [B,1,H,W]."""
    t = alpha[:, 0] if alpha.ndim == 5 else alpha
    if t.ndim == 4:
        if t.shape[1] == 1:
            return t.float().clamp(0, 1)
        if t.shape[-1] == 1:
            return t.permute(0, 3, 1, 2).float().clamp(0, 1)
    if t.ndim == 3:
        return t.unsqueeze(1).float().clamp(0, 1)
    raise ValueError(f"Unsupported comp_mask shape: {tuple(alpha.shape)}")


def _unbatch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the DataLoader batch dimension (batch size 1)."""
    for k, v in list(batch.items()):
        if torch.is_tensor(v) and v.ndim >= 1:
            batch[k] = v[0]
        elif isinstance(v, (list, tuple)) and len(v) > 0:
            batch[k] = v[0]
    return batch


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
) -> Tuple[float, float, float]:
    """PSNR / SSIM / LPIPS over the full H x W image.

    Both the prediction and the ground truth keep their values only inside the ray
    mask and are filled with the background color elsewhere. They are quantized to
    8 bits before the metrics are computed.
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

    return psnr, ssim, lpips


# ---------------------------------------------------------------------------
# Boundary / alpha metrics (rendered alpha vs GT hand mask)
# ---------------------------------------------------------------------------
BOUNDARY_METRICS = (  # (key, printed name)
    ("boundary_fscore", "Boundary F-score"),
    ("boundary_iou", "Boundary IoU"),
    ("alpha_leakage", "Alpha leakage"),
    ("inside_coverage", "Interior coverage"),
)


def extract_boundary(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """One-pixel inner boundary: mask minus its erosion."""
    mask = mask.astype(bool)
    return mask ^ binary_erosion(mask, iterations=radius)


def boundary_fscore(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance: int = 2) -> Tuple[float, float, float]:
    """(precision, recall, F): boundary pixels within `tolerance` px of the other boundary."""
    pred_b = extract_boundary(pred_mask)
    gt_b = extract_boundary(gt_mask)
    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 1.0, 1.0, 1.0
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return 0.0, 0.0, 0.0
    gt_dist = distance_transform_edt(~gt_b)
    pred_dist = distance_transform_edt(~pred_b)
    precision = float((gt_dist[pred_b] <= tolerance).mean())
    recall = float((pred_dist[gt_b] <= tolerance).mean())
    fscore = float(2.0 * precision * recall / (precision + recall + 1e-8))
    return precision, recall, fscore


def boundary_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, band_radius: int = 3) -> float:
    """IoU of the two boundaries, each dilated by `band_radius` px."""
    pred_band = binary_dilation(extract_boundary(pred_mask), iterations=band_radius)
    gt_band = binary_dilation(extract_boundary(gt_mask), iterations=band_radius)
    inter = np.logical_and(pred_band, gt_band).sum()
    union = np.logical_or(pred_band, gt_band).sum()
    return float(inter / (union + 1e-8))


def alpha_band_metrics(pred_alpha: np.ndarray, gt_mask: np.ndarray, band_radius: int = 5, tau_out: float = 0.1) -> Dict[str, float]:
    """Alpha leakage: mean of max(alpha - tau_out, 0) in the outer band (GT mask dilated by
    `band_radius`, minus the mask). Interior coverage: mean alpha inside the GT mask eroded by
    `band_radius`."""
    gt_mask = gt_mask.astype(bool)
    outer_band = np.logical_and(binary_dilation(gt_mask, iterations=band_radius), ~gt_mask)
    inner_region = binary_erosion(gt_mask, iterations=band_radius)
    alpha_leakage = np.maximum(pred_alpha[outer_band] - tau_out, 0.0).mean() if outer_band.sum() > 0 else 0.0
    inside_coverage = pred_alpha[inner_region].mean() if inner_region.sum() > 0 else 1.0
    return {"alpha_leakage": float(alpha_leakage), "inside_coverage": float(inside_coverage)}


def gt_hand_mask(inputs: Dict[str, Any]) -> Optional[torch.Tensor]:
    """GT hand mask in the image crop, where the renders are compared.

    'gt_mask_crop' is the untouched crop mask; 'target_mask' is the copy the model sees, which
    lives in the rotated view when the adapter rotates the camera (used only if the other is
    missing).
    """
    data = inputs.get("data") or {}
    mask = data.get("gt_mask_crop")
    return data.get("target_mask") if mask is None else mask


@torch.no_grad()
def compute_boundary_metrics(pred_alpha01: torch.Tensor, gt_mask: torch.Tensor, H: int, W: int, bcfg: Dict[str, Any]) -> Dict[str, float]:
    """Boundary / alpha metrics of one frame. pred_alpha01: [1,1,H,W] rendered alpha (crop space)."""
    alpha = pred_alpha01.detach().float().cpu().reshape(-1, *pred_alpha01.shape[-2:])[0]
    if tuple(alpha.shape) != (H, W):
        alpha = F.interpolate(alpha[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    alpha_np = alpha.clamp(0, 1).numpy()
    pred_mask_np = alpha_np > float(bcfg["pred_mask_thresh"])

    gt = torch.as_tensor(gt_mask).detach().float().cpu()
    gt = gt.reshape(-1, *gt.shape[-2:])[0]
    if tuple(gt.shape) != (H, W):
        gt = F.interpolate(gt[None, None], size=(H, W), mode="nearest")[0, 0]
    gt_mask_np = gt.numpy() > 0.5

    _, _, bf = boundary_fscore(pred_mask_np, gt_mask_np, tolerance=int(bcfg["fscore_tolerance"]))
    biou = boundary_iou(pred_mask_np, gt_mask_np, band_radius=int(bcfg["iou_band_radius"]))
    alpha_stats = alpha_band_metrics(alpha_np, gt_mask_np, band_radius=int(bcfg["alpha_band_radius"]),
                                     tau_out=float(bcfg["tau_out"]))
    return {"boundary_fscore": bf, "boundary_iou": biou, **alpha_stats}


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
def render_frame(model: ModelHAM, gs_models, query_points, inputs: Dict[str, Any], H: int, W: int
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drive the canonical avatar to a target frame's pose and camera.
    Returns RGB [1,3,H,W] and alpha [1,1,H,W], both in [0,1]."""
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
    alpha = _alpha_to_nchw(out["comp_mask"])
    if tuple(pred.shape[-2:]) != (H, W):
        pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)
    if tuple(alpha.shape[-2:]) != (H, W):
        alpha = F.interpolate(alpha, size=(H, W), mode="bilinear", align_corners=False)
    return pred, alpha


def to_crop(pred: torch.Tensor, inputs: Dict[str, Any], fill=None) -> torch.Tensor:
    """Map a render back to the image crop the ground truth (ray mask, target colors) lives in.

    The adapter renders from a camera rotated so that its principal point is at the image
    center (the rasterizer ignores cx, cy); `H_crop_to_render` maps crop pixels into that
    render. Bicubic resampling; `fill` where the rotated view has no pixels (default: the
    background color, for RGB; 0 for alpha).
    """
    Hm = inputs["data"].get("H_crop_to_render")
    if Hm is None:  # camera not rotated (pp_mode "none")
        return pred
    if fill is None:
        fill = inputs["render_bg_colors"].reshape(-1)[:3].float().cpu().tolist()  # [1,3] -> [3]
    return unwarp_to_crop(pred, Hm, fill=fill, mode="bicubic", clamp=(0.0, 1.0))


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
    anchor_inputs = handavatar_sample_to_infer_inputs(anchor, device=device)
    gs_models, query_points = build_canonical_avatar(model, anchor_inputs, device)

    from external.handavatar.core.data import create_dataloader  # needs ha_configs.cfg

    with _quiet(quiet):
        loader = create_dataloader(data_type="progress")

    bcfg = cfg.get("boundary") or {}
    with_boundary = bool(bcfg.get("enabled", False))

    pcfg = cfg.get("pose_strata") or {}
    with_pose = bool(pcfg.get("enabled", False))
    src_pose = pose_raw(anchor_inputs, cam_R=anchor.get("cam_R")) if with_pose else None

    max_batches = int(cfg["max_batches"])
    rows = []  # (psnr, ssim, lpips) per frame
    boundary_rows = []  # boundary / alpha metrics per frame
    pose_records = []  # pose of each frame, for the pose strata
    for frame_index, batch in enumerate(tqdm(loader, desc="eval")):
        if max_batches > 0 and frame_index >= max_batches:
            break
        batch = _unbatch(batch)
        H, W = int(batch["img_height"]), int(batch["img_width"])

        ray_mask = batch["ray_mask"]
        if isinstance(ray_mask, np.ndarray):
            ray_mask = torch.from_numpy(ray_mask)
        ray_mask = (ray_mask.to(device) > 0).reshape(-1)
        target_rgbs = _to_unit_range(_to_float_tensor(batch["target_rgbs"], device))

        inputs = handavatar_sample_to_infer_inputs(batch, device=device)
        pred, alpha = render_frame(model, gs_models, query_points, inputs, H, W)
        pred = to_crop(pred, inputs)

        rows.append(compute_metrics(pred, target_rgbs, ray_mask, H, W, bg01, lpips_model))

        if with_boundary:
            gt_mask = gt_hand_mask(inputs)
            if gt_mask is None:
                raise KeyError("Boundary metrics need a GT hand mask ('gt_mask_crop' or 'target_mask' in the "
                               "adapter output); set boundary.enabled: false to skip them.")
            boundary_rows.append(compute_boundary_metrics(to_crop(alpha, inputs, fill=0.0), gt_mask, H, W, bcfg))

        if with_pose:
            cam, stem = _parse_cam_stem(batch.get("frame_name", f"{frame_index:06d}"))
            pose_records.append({"cam": cam, "stem": stem, "_pose": pose_raw(inputs, cam_R=batch.get("cam_R"))})

    # Mean over frames, rounded as in the paper tables: main, boundary / alpha, pose strata
    print(f"\n======== InterHand2.6M evaluation ({len(rows)} frames) ========")
    print("-------- main --------")
    for col, name in enumerate(("PSNR", "SSIM", "LPIPS")):
        mean = float(np.array([r[col] for r in rows], dtype=np.float64).mean())
        line = f"{name:<5}: {paper_round(mean, PAPER_DIGITS[name])}"
        print(line + (f"  (unrounded {mean:.6f})" if cfg.get("verbose") else ""))

    if with_boundary:
        print("-------- boundary / alpha --------")
        for key, name in BOUNDARY_METRICS:
            mean = float(np.array([r[key] for r in boundary_rows], dtype=np.float64).mean())
            line = f"{name:<17}: {paper_round(mean, 3)}"
            print(line + (f"  (unrounded {mean:.6f})" if cfg.get("verbose") else ""))

    if with_pose:
        print_pose_strata(rows, pose_records, src_pose, pcfg, verbose=bool(cfg.get("verbose")))

    # Self-reconstruction: a second pass over the test frames, after the results above are printed
    scfg = cfg.get("self_recon") or {}
    if bool(scfg.get("enabled", False)):
        del gs_models, query_points
        sys.stdout.flush()  # show the results above before the second pass starts
        run_self_recon(cfg, scfg, model, lpips_model, bg01, device, quiet)


@torch.no_grad()
def run_self_recon(cfg: Dict[str, Any], scfg: Dict[str, Any], model: ModelHAM, lpips_model, bg01: torch.Tensor,
                   device: torch.device, quiet: bool) -> None:
    """Every test frame is its own source: build the avatar from the frame, render it back into
    the same frame. 'w/o transfer' = same checkpoint without the appearance head, 'Ours' = full
    model; one image encoding per frame is shared by the two."""
    from external.handavatar.core.data import create_dataloader  # needs ha_configs.cfg

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

        ray_mask = batch["ray_mask"]
        if isinstance(ray_mask, np.ndarray):
            ray_mask = torch.from_numpy(ray_mask)
        ray_mask = (ray_mask.to(device) > 0).reshape(-1)
        target_rgbs = _to_unit_range(_to_float_tensor(batch["target_rgbs"], device))

        inputs = handavatar_sample_to_infer_inputs(batch, device=device)
        avatars = build_canonical_avatars([model, canon_model], inputs, device)
        for (setting, m), (gs_self, qp_self) in zip((("Ours", model), ("w/o transfer", canon_model)), avatars):
            pred, _ = render_frame(m, gs_self, qp_self, inputs, H, W)
            pred = to_crop(pred, inputs)
            self_rows[setting].append(compute_metrics(pred, target_rgbs, ray_mask, H, W, bg01, lpips_model))
        del avatars

    print("-------- self-reconstruction (each frame its own source) --------")
    print(f"{'':<13s} {'PSNR':>7s} {'SSIM':>7s} {'LPIPS':>7s}")
    for setting, r in self_rows.items():
        psnr, ssim, lpips = np.array(r, dtype=np.float64).mean(axis=0)
        print(f"{setting + ':':<13s} {paper_round(psnr, PAPER_DIGITS['PSNR']):>7s} "
              f"{paper_round(ssim, PAPER_DIGITS['SSIM']):>7s} {paper_round(lpips, PAPER_DIGITS['LPIPS']):>7s}")


def parse_args() -> Dict[str, Any]:
    p = argparse.ArgumentParser(description="Evaluate InstantHand on InterHand2.6M.")
    p.add_argument("--cfg", default="configs/eval_interhand.yaml")
    p.add_argument("--ckpt_dir", help="override ckpt_dir in the config")
    p.add_argument("--max_batches", type=int, help="override max_batches in the config (-1: all frames)")
    p.add_argument("--no_boundary", action="store_true", help="skip the boundary / alpha metrics")
    p.add_argument("--no_pose_strata", action="store_true", help="skip the pose-stratified metrics")
    p.add_argument("--self_recon", action="store_true", help="also run the self-reconstruction metrics (slow)")
    p.add_argument("--no_self_recon", action="store_true", help="skip the self-reconstruction metrics")
    p.add_argument("--verbose", action="store_true", help="show logs and warnings from model, LPIPS and data setup")
    args = p.parse_args(_CLI_ARGV)

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("ckpt_dir", "max_batches"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.no_boundary:
        cfg.setdefault("boundary", {})["enabled"] = False
    if args.no_pose_strata:
        cfg.setdefault("pose_strata", {})["enabled"] = False
    if args.self_recon:
        cfg.setdefault("self_recon", {})["enabled"] = True
    if args.no_self_recon:
        cfg.setdefault("self_recon", {})["enabled"] = False
    cfg["verbose"] = args.verbose
    return cfg


if __name__ == "__main__":
    run_eval(parse_args())