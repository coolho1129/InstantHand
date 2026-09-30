# Copyright (c) 2026 The InstantHand Authors.
# SPDX-License-Identifier: Apache-2.0
"""Stage 1: train the avatar predictor (geometry and base colors) from multi-view hand videos.

    accelerate launch --num_processes 4 train_stage1.py --cfg configs/train_stage1.yaml
    (options: --resume_dir <checkpoint dir>, --set key.subkey=value)

The DINOv3 backbone is frozen; the appearance head is not built. Losses on the rendered target
views: masked L1 + LPIPS, silhouette L1 and the soft alpha band (SAB) on the rendered alpha,
and Laplacian / L2 regularizers on the Gaussian offsets and a hinge on the Gaussian scales.
"""
from warnings import filterwarnings

filterwarnings("ignore", category=UserWarning)

import os
from typing import Any, Dict, Optional

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader

from instanthand.datasets.multi_hand import MultiHandDataset
from instanthand.losses import LaplacianReg, MaskAwareLPIPSLoss, MaskAwarePixelLoss, MaskLoss, SoftBandAlphaLoss
from instanthand.models.modeling_ham import ModelHAM
from instanthand.training.common import (
    build_dataset_config,
    collate_keep_uid,
    cosine_warmup_hold_decay_scheduler,
    load_config,
    mano_params_from_batch,
    mixed_precision_mode,
    resume_state,
    target_cameras_from_batch,
    to_bvchw,
    train_loop,
)

TRAIN_STAGE = "stage1"
LOSS_KEYS = ["loss", "rgb", "l1", "lpips", "sil", "lap", "off", "scale", "alpha_out", "alpha_in",
             "outer_band_mean", "core_mean", "excess_out_mean", "lack_in_mean", "pred_area_mean",
             "gt_area_mean", "alpha_mult"]


def pack_inputs(batch: Dict[str, Any], device):
    """Model inputs and ground truth (targets [B,V,...]) for one batch."""
    source_rgbs = batch["source_rgbs"].to(device)
    if source_rgbs.ndim == 4:
        source_rgbs = source_rgbs.unsqueeze(1)                      # [B,1,3,H,W]
    c2ws, intrs, bg = target_cameras_from_batch(batch, device)
    gt_rgb = to_bvchw(batch["render_image"].to(device))             # [B,V,3,H,W]
    gt_msk = to_bvchw(batch["render_mask"].to(device))              # [B,V,1,H,W]
    sil = batch.get("render_silhouette_mask")
    gt_sil = to_bvchw(sil.to(device)) if sil is not None else gt_msk

    model_in = dict(
        image=source_rgbs,
        source_c2ws=batch["source_c2ws"].to(device) if "source_c2ws" in batch else c2ws[:, :1],
        source_intrs=batch["source_intrs"].to(device) if "source_intrs" in batch else intrs[:, :1],
        render_c2ws=c2ws,
        render_intrs=intrs,
        render_bg_colors=bg,
        render_full_resolutions=None,
        mano_params=mano_params_from_batch(batch, device),
        source_mano_params=None,
        render_images=None,
        data=None,
    )
    return model_in, dict(render_image=gt_rgb, render_mask=gt_msk, render_silhouette_mask=gt_sil)


def _pred_alpha(out: Dict[str, Any]) -> Optional[torch.Tensor]:
    """Rendered alpha as [B,V,1,H,W] (the renderer's comp_mask)."""
    for k in ("comp_alpha", "comp_mask", "alpha", "mask", "acc_alpha", "accum_alpha", "opacity", "comp_opacity"):
        if out.get(k) is not None:
            return to_bvchw(out[k])
    return None


def _gaussian_attr(out: Dict[str, Any], *keys: str) -> Optional[torch.Tensor]:
    """Per-Gaussian attribute [B,N,C] (view 0 if the output is [B,V,N,C])."""
    for k in keys:
        v = out.get(k)
        if torch.is_tensor(v):
            return v[:, 0] if v.ndim == 4 else v
    return None


def main():
    cfg = load_config("configs/train_stage1.yaml", "InstantHand stage-1 training")
    optim_cfg, loss_cfg, data_cfg, out_cfg = cfg["optim"], cfg["loss"], cfg["data"], cfg["output"]

    set_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    accelerator = Accelerator(
        mixed_precision=mixed_precision_mode(cfg["mixed_precision"]),
        gradient_accumulation_steps=optim_cfg["accum_steps"],
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    device = accelerator.device

    model = ModelHAM(**cfg["model"])
    model.set_train_stage(TRAIN_STAGE)

    pixel_loss = MaskAwarePixelLoss(option="l1", eps=1e-6, normalize_by_mask=True)
    lpips_loss = MaskAwareLPIPSLoss(device=device, prefech=False)
    silhouette_loss = MaskLoss(option="l1")
    alpha_loss = SoftBandAlphaLoss(**loss_cfg["soft_alpha_band"])

    optimizer = AdamW(model.get_trainable_parameters(), lr=optim_cfg["lr"],
                      betas=tuple(optim_cfg["betas"]), weight_decay=optim_cfg["weight_decay"])

    train_ds = MultiHandDataset(build_dataset_config(
        data_cfg, TRAIN_STAGE, optim_cfg["steps_per_epoch"], optim_cfg["batch_size"],
        optim_cfg["accum_steps"], accelerator.num_processes))
    train_loader = DataLoader(train_ds, batch_size=optim_cfg["batch_size"], shuffle=False,
                              num_workers=data_cfg["num_workers"], pin_memory=True, drop_last=True,
                              collate_fn=collate_keep_uid, persistent_workers=data_cfg["num_workers"] > 0)

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    scheduler = cosine_warmup_hold_decay_scheduler(optimizer, **optim_cfg["scheduler"])
    raw_model = accelerator.unwrap_model(model)
    raw_model.set_train_stage(TRAIN_STAGE)

    mano = raw_model.renderer.mano_model
    lap_reg = LaplacianReg(vertex_num=mano.vertex_num_upsampled, face=mano.face_upsampled,
                           device=device, neighbor_max_num=20).to(device)

    start_epoch, global_step = 0, 0
    if cfg["resume_dir"]:
        start_epoch, global_step = resume_state(accelerator, cfg["resume_dir"], optimizer, scheduler,
                                                optim_cfg["steps_per_epoch"], optim_cfg["scheduler"])
        raw_model = accelerator.unwrap_model(model)
        raw_model.set_train_stage(TRAIN_STAGE)

    if accelerator.is_main_process:
        os.makedirs(out_cfg["save_dir"], exist_ok=True)
        os.makedirs(out_cfg["tb_dir"], exist_ok=True)
        raw_model.print_trainable_parameters()
        accelerator.print(f"[Dataset] subjects={train_ds.num_subjects}, frames={train_ds.total_frames}, "
                          f"weights={dict(zip(train_ds.available_datasets, train_ds.available_dataset_weights))}")
        if start_epoch or global_step:
            accelerator.print(f"[Resume] start_epoch={start_epoch}, global_step={global_step}")

    warned = {"no_alpha": False}

    def step_fn(batch: Dict[str, Any], step: int) -> Dict[str, torch.Tensor]:
        model_in, gt = pack_inputs(batch, device)
        out = model(**model_in)
        zero = torch.zeros([], device=device)

        l1 = pixel_loss(out["comp_rgb"], gt["render_image"], mask=gt["render_mask"])
        lp = lpips_loss(out["comp_rgb"], gt["render_image"], mask=gt["render_mask"], is_training=True)
        l_rgb = l1 + loss_cfg["lambda_lpips"] * lp

        pred_alpha = _pred_alpha(out)
        if pred_alpha is None:
            if not warned["no_alpha"] and accelerator.is_main_process:
                warned["no_alpha"] = True
                accelerator.print("[Warn] no rendered alpha in the model output; silhouette and SAB losses are 0")
            l_sil, l_alpha = zero, zero
            stats = {k: zero for k in ("alpha_out", "alpha_in", "outer_band_mean", "core_mean", "excess_out_mean",
                                       "lack_in_mean", "pred_area_mean", "alpha_mult")}
            stats["gt_area_mean"] = gt["render_silhouette_mask"].mean()
        else:
            l_sil = silhouette_loss(pred_alpha, gt["render_silhouette_mask"])
            l_alpha, alpha_out = alpha_loss.weighted_loss(pred_alpha=pred_alpha, gt_sil=gt["render_silhouette_mask"],
                                                          global_step=step)
            stats = alpha_out.as_dict()

        off = _gaussian_attr(out, "offset", "gs_offset")
        l_lap = lap_reg(off).sum(dim=-1).mean() if off is not None else zero
        l_off = (off ** 2).mean() if off is not None else zero
        sc = _gaussian_attr(out, "scaling", "gs_scaling")
        l_scale = torch.relu(sc - loss_cfg["scale_reg_thresh"]).pow(2).mean() if sc is not None else zero

        loss = (loss_cfg["lambda_rgb"] * l_rgb + loss_cfg["lambda_silhouette"] * l_sil
                + loss_cfg["lambda_lap"] * l_lap + loss_cfg["lambda_offset"] * l_off
                + loss_cfg["lambda_scale"] * l_scale + l_alpha)
        return {"loss": loss, "rgb": l_rgb, "l1": l1, "lpips": lp, "sil": l_sil, "lap": l_lap, "off": l_off,
                "scale": l_scale, **{k: stats[k] for k in LOSS_KEYS[8:]}}

    sab = loss_cfg["soft_alpha_band"]

    def postfix(ema: Dict[str, float]) -> Dict[str, str]:
        return {
            "loss": f"{ema['loss']:.4f}",
            "rgb": f"{ema['rgb']:.4f}",
            "l1": f"{ema['l1'] * loss_cfg['lambda_rgb']:.4f}",
            "lp": f"{ema['lpips'] * loss_cfg['lambda_lpips']:.4f}",
            "sil": f"{ema['sil'] * loss_cfg['lambda_silhouette']:.4f}",
            "a_out": f"{ema['alpha_out'] * sab['lambda_alpha_out'] * ema['alpha_mult']:.5f}",
            "a_in": f"{ema['alpha_in'] * sab['lambda_alpha_in'] * ema['alpha_mult']:.5f}",
            "sc": f"{ema['scale'] * loss_cfg['lambda_scale']:.6f}",
            "lap": f"{ema['lap'] * loss_cfg['lambda_lap']:.6f}",
            "off": f"{ema['off'] * loss_cfg['lambda_offset']:.6f}",
        }

    train_loop(cfg=cfg, accelerator=accelerator, model=model, optimizer=optimizer, scheduler=scheduler,
               train_loader=train_loader, step_fn=step_fn, loss_keys=LOSS_KEYS, postfix_fn=postfix,
               train_stage=TRAIN_STAGE, start_epoch=start_epoch, global_step=global_step)


if __name__ == "__main__":
    main()
