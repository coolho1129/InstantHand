"""Evaluate InstantHand on HanCo (zero-shot).

The canonical avatar is built once from a single anchor image and then driven to
every frame of the HanCo evaluation sequence. PSNR, SSIM and LPIPS follow the
HandAvatar progress protocol.

Usage:
    python eval_hanco.py --cfg configs/eval_hanco.yaml
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
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from safetensors.torch import load_file
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
            "source_depth_map": None,
            "source_mano_data": {k: mano[k] for k in ("betas", "global_orient", "hand_pose", "translation")},
        },
        train_stage=model.current_train_stage,
    )
    return gs_models, query_points


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

    from external.handavatar.core.data.create_dataset2 import create_dataloader  # needs ha_configs.cfg

    with _quiet(quiet):
        loader = create_dataloader(data_type="progress")

    max_batches = int(cfg["max_batches"])
    rows = []  # (psnr, ssim, lpips) per frame
    for frame_index, batch in enumerate(tqdm(loader, desc="eval")):
        if max_batches > 0 and frame_index >= max_batches:
            break
        batch = _unbatch(batch)
        H, W = int(batch["img_height"]), int(batch["img_width"])

        ray_mask, target_rgbs = _eval_targets(batch, device)

        inputs = handavatar_sample_to_infer_inputs(batch, device, adapter_cfg)
        pred = to_crop(render_frame(model, gs_models, query_points, inputs, H, W), inputs)

        rows.append(compute_metrics(pred, target_rgbs, ray_mask, H, W, bg01, lpips_model))

    # Mean over frames, rounded as in the paper tables
    print(f"\n======== HanCo evaluation ========\nsamples: {len(rows)}")
    for col, name in enumerate(("PSNR", "SSIM", "LPIPS")):
        mean = float(np.array([r[col] for r in rows], dtype=np.float64).mean())
        line = f"{name:<5}: {paper_round(mean, PAPER_DIGITS[name])}"
        print(line + (f"  (unrounded {mean:.6f})" if cfg.get("verbose") else ""))


def parse_args() -> Dict[str, Any]:
    p = argparse.ArgumentParser(description="Evaluate InstantHand on HanCo.")
    p.add_argument("--cfg", default="configs/eval_hanco.yaml")
    p.add_argument("--ckpt_dir", help="override ckpt_dir in the config")
    p.add_argument("--max_batches", type=int, help="override max_batches in the config (-1: all frames)")
    p.add_argument("--pp_mode", choices=["rotate", "none"], help="override pp_mode in the config")
    p.add_argument("--verbose", action="store_true", help="show logs and warnings from model, LPIPS and data setup")
    args = p.parse_args(_CLI_ARGV)

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("ckpt_dir", "max_batches", "pp_mode"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    cfg["verbose"] = args.verbose
    return cfg


if __name__ == "__main__":
    run_eval(parse_args())