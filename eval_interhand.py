"""Evaluate InstantHand on InterHand2.6M.

The canonical avatar is built once from a single anchor image and then driven to
every frame of the HandAvatar test split. PSNR, SSIM and LPIPS follow the
HandAvatar progress protocol. Boundary metrics (boundary F-score, boundary IoU, alpha
leakage, interior coverage) compare the rendered alpha with the GT hand mask.

Usage:
    python eval_interhand.py --cfg configs/eval_interhand.yaml
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
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from safetensors.torch import load_file
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


@torch.no_grad()
def build_canonical_avatar(model: ModelHAM, anchor_inputs: Dict[str, Any], device: torch.device):
    """Single forward pass: anchor image -> canonical Gaussian avatar."""
    image = anchor_inputs["image"]
    if image.ndim == 5:  # [B, V, 3, H, W] -> first view
        image = image[:, 0]

    mano = anchor_inputs["mano_params"]
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

    data = anchor_inputs.get("data") or {}
    source_intrs = anchor_inputs["source_intrs"]
    source_c2ws = anchor_inputs["source_c2ws"]

    gs_models, query_points, *_ = model.renderer.build_gs(
        gs_hidden_features=latent_points,
        query_points=query_points,
        mano_params=mano_params,
        additional_features={
            "image_feats": image_feats,
            "image": image,
            "image_feats_bchw": image_feats_bchw,
            "source_image": image,
            "source_mask": data.get("source_mask"),
            "source_intrinsic": source_intrs[:, 0] if source_intrs.ndim == 4 else source_intrs,
            "source_c2w": source_c2ws[:, 0] if source_c2ws.ndim == 4 else source_c2ws,
            "source_mano_data": {k: mano[k] for k in ("betas", "global_orient", "hand_pose", "translation")},
        },
        train_stage=model.current_train_stage,
    )
    return gs_models, query_points


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

    max_batches = int(cfg["max_batches"])
    rows = []  # (psnr, ssim, lpips) per frame
    boundary_rows = []  # boundary / alpha metrics per frame
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

    # Mean over frames, rounded as in the paper tables
    print(f"\n======== InterHand2.6M evaluation ========\nsamples: {len(rows)}")
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


def parse_args() -> Dict[str, Any]:
    p = argparse.ArgumentParser(description="Evaluate InstantHand on InterHand2.6M.")
    p.add_argument("--cfg", default="configs/eval_interhand.yaml")
    p.add_argument("--ckpt_dir", help="override ckpt_dir in the config")
    p.add_argument("--max_batches", type=int, help="override max_batches in the config (-1: all frames)")
    p.add_argument("--no_boundary", action="store_true", help="skip the boundary / alpha metrics")
    p.add_argument("--verbose", action="store_true", help="show logs and warnings from model, LPIPS and data setup")
    args = p.parse_args(_CLI_ARGV)

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("ckpt_dir", "max_batches"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.no_boundary:
        cfg.setdefault("boundary", {})["enabled"] = False
    cfg["verbose"] = args.verbose
    return cfg


if __name__ == "__main__":
    run_eval(parse_args())