# Copyright (c) 2026 The InstantHand Authors.
# SPDX-License-Identifier: Apache-2.0
"""Stage 2: train the appearance head on top of a frozen stage-1 model.

    accelerate launch --num_processes 4 train_stage2.py --cfg configs/train_stage2.yaml
    (options: --resume_dir <checkpoint dir>, --set key.subkey=value)

Only the appearance head is optimized. Losses: masked L1 + LPIPS on the first target view
(the source view itself) and on the remaining (novel) target views, with separate weights.
The feature-statistics term (self_feat_stats) compares frozen image-encoder statistics of the
self-view render and the source image; it is computed without gradients, so it is logged and
added to the total as in the original training but does not change the updates.
"""
from warnings import filterwarnings

filterwarnings("ignore", category=UserWarning)

import os
from typing import Any, Dict, Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader

from instanthand.datasets.multi_hand import MultiHandDataset
from instanthand.losses import MaskAwareLPIPSLoss, MaskAwarePixelLoss
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

TRAIN_STAGE = "stage2"
LOSS_KEYS = ["loss", "self", "self_l1", "self_lpips", "self_feat_stats", "novel", "novel_l1", "novel_lpips"]


def _to_b1hw(x: torch.Tensor) -> torch.Tensor:
    """Mask as [B,1,H,W] from [B,1,1,H,W], [B,C,H,W] (first channel) or [B,H,W]."""
    if x.ndim == 5:
        if x.shape[2] != 1:
            raise ValueError(f"unexpected 5D mask shape {tuple(x.shape)}")
        return x[:, 0]
    if x.ndim == 4:
        return x if x.shape[1] == 1 else x[:, 0:1]
    if x.ndim == 3:
        return x.unsqueeze(1)
    raise ValueError(f"unexpected mask ndim {x.ndim}")


def pack_inputs(batch: Dict[str, Any], device):
    """Model inputs (with the source-side data the appearance head needs) and ground truth."""
    source_rgbs = batch["source_rgbs"].to(device)
    if source_rgbs.ndim == 4:
        source_rgbs = source_rgbs.unsqueeze(1)                      # [B,1,3,H,W]
    c2ws, intrs, bg = target_cameras_from_batch(batch, device)
    gt_rgb = to_bvchw(batch["render_image"].to(device))             # [B,V,3,H,W]
    gt_msk = to_bvchw(batch["render_mask"].to(device))              # [B,V,1,H,W]
    mano_params = mano_params_from_batch(batch, device)

    if "source_mask" in batch:
        source_mask = _to_b1hw(batch["source_mask"].to(device))
    elif "source_silhouette_mask" in batch:
        source_mask = _to_b1hw(batch["source_silhouette_mask"].to(device))
    else:
        source_mask = _to_b1hw(gt_msk[:, 0])

    source_mano_params = None
    if all(batch.get(k) is not None for k in ("source_global_orient", "source_hand_pose", "source_translation")):
        source_betas = batch.get("source_betas")
        if source_betas is None:
            source_betas = mano_params["betas"]
        else:
            source_betas = source_betas.to(device)
            if source_betas.ndim == 1:
                source_betas = source_betas.unsqueeze(0)
        source_mano_params = dict(betas=source_betas,
                                  global_orient=batch["source_global_orient"].to(device),
                                  hand_pose=batch["source_hand_pose"].to(device),
                                  translation=batch["source_translation"].to(device))

    data = dict(batch)
    data.update(source_mask=source_mask, render_images=gt_rgb, render_fg_masks=gt_msk)
    model_in = dict(
        image=source_rgbs,
        source_c2ws=batch["source_c2ws"].to(device) if "source_c2ws" in batch else c2ws[:, :1],
        source_intrs=batch["source_intrs"].to(device) if "source_intrs" in batch else intrs[:, :1],
        render_c2ws=c2ws,
        render_intrs=intrs,
        render_bg_colors=bg,
        render_full_resolutions=None,
        mano_params=mano_params,
        source_mano_params=source_mano_params,
        render_images=gt_rgb,
        data=data,
    )
    gt = dict(render_images=gt_rgb, render_fg_masks=gt_msk, source_image=source_rgbs[:, 0], source_mask=source_mask)
    return model_in, gt


class FeatureMeanStdLoss(nn.Module):
    """|mean_pred - mean_tgt| + |std_pred - std_tgt| of feature maps [B,C,H,W], each inside its own mask."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, feat_pred, feat_tgt, mask_pred, mask_tgt) -> torch.Tensor:
        if feat_pred.shape != feat_tgt.shape or feat_pred.ndim != 4:
            raise ValueError(f"expected two [B,C,H,W] maps of one shape, got {tuple(feat_pred.shape)}, {tuple(feat_tgt.shape)}")
        _, C, H, W = feat_pred.shape

        def stats(feat, mask):
            mask = F.interpolate(mask.float(), size=(H, W), mode="nearest").expand(-1, C, -1, -1)
            denom = mask.sum(dim=(2, 3)).clamp_min(self.eps)
            mean = (feat * mask).sum(dim=(2, 3)) / denom
            var = (((feat - mean[..., None, None]) ** 2) * mask).sum(dim=(2, 3)) / denom
            return mean, torch.sqrt(var.clamp_min(self.eps))

        mean_p, std_p = stats(feat_pred, mask_pred)
        mean_t, std_t = stats(feat_tgt, mask_tgt)
        return (mean_p - mean_t).abs().mean() + (std_p - std_t).abs().mean()


@torch.no_grad()
def encode_image_features_frozen(model: ModelHAM, image_bchw: torch.Tensor) -> torch.Tensor:
    """Image-encoder tokens [B,T,C] of an image, as a [B,C,sqrt(T),sqrt(T)] map (no gradients)."""
    tokens = model.forward_encode_image(image_bchw)
    B, T, C = tokens.shape
    side = int(math.sqrt(T))
    if side * side != T:
        raise ValueError(f"token count {T} is not a square")
    return tokens.transpose(1, 2).reshape(B, C, side, side)


class Stage2Criterion(nn.Module):
    """Masked L1 + LPIPS on the self view (target 0, the source view) and on the novel views, plus
    the feature-statistics term on the self view (no gradient; see the module docstring)."""

    def __init__(self, device, lambda_self_l1: float = 1.0, lambda_self_lpips: float = 0.5,
                 lambda_self_feat_stats: float = 0.05, lambda_novel_l1: float = 1.0, lambda_novel_lpips: float = 0.25):
        super().__init__()
        self.lambda_self_l1 = float(lambda_self_l1)
        self.lambda_self_lpips = float(lambda_self_lpips)
        self.lambda_self_feat_stats = float(lambda_self_feat_stats)
        self.lambda_novel_l1 = float(lambda_novel_l1)
        self.lambda_novel_lpips = float(lambda_novel_lpips)
        self.pixel_l1 = MaskAwarePixelLoss(option="l1", normalize_by_mask=True)
        self.lpips = MaskAwareLPIPSLoss(device=device, prefech=False)
        self.feat_stats = FeatureMeanStdLoss()

    def forward(self, raw_model: ModelHAM, outputs: Dict[str, torch.Tensor],
                gt: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pred, gt_img, gt_mask = outputs["comp_rgb"], gt["render_images"], gt["render_fg_masks"]   # [B,V,...]
        if pred.ndim != 5 or pred.shape[1] < 1:
            raise ValueError(f"expected rendered targets [B,V,3,H,W] with V >= 1, got {tuple(pred.shape)}")

        self_l1 = self.pixel_l1(pred[:, 0:1], gt_img[:, 0:1], gt_mask[:, 0:1])
        self_lpips = self.lpips(pred[:, 0:1], gt_img[:, 0:1], gt_mask[:, 0:1], bg=None, is_training=True)
        if self.lambda_self_feat_stats > 0:
            self_feat_stats = self.feat_stats(
                feat_pred=encode_image_features_frozen(raw_model, pred[:, 0]),
                feat_tgt=encode_image_features_frozen(raw_model, gt["source_image"]),
                mask_pred=gt_mask[:, 0],
                mask_tgt=gt["source_mask"],
            )
        else:
            self_feat_stats = torch.zeros([], device=pred.device)
        loss_self = (self.lambda_self_l1 * self_l1 + self.lambda_self_lpips * self_lpips
                     + self.lambda_self_feat_stats * self_feat_stats)

        if pred.shape[1] > 1:
            novel_l1 = self.pixel_l1(pred[:, 1:], gt_img[:, 1:], gt_mask[:, 1:])
            novel_lpips = self.lpips(pred[:, 1:], gt_img[:, 1:], gt_mask[:, 1:], bg=None, is_training=True)
            loss_novel = self.lambda_novel_l1 * novel_l1 + self.lambda_novel_lpips * novel_lpips
        else:
            novel_l1 = novel_lpips = loss_novel = torch.zeros([], device=pred.device)

        return {"loss": loss_self + loss_novel, "self": loss_self, "self_l1": self_l1, "self_lpips": self_lpips,
                "self_feat_stats": self_feat_stats, "novel": loss_novel, "novel_l1": novel_l1,
                "novel_lpips": novel_lpips}


def load_stage1_weights(model: ModelHAM, ckpt_dir: Optional[str], accelerator) -> None:
    """Load the stage-1 weights (pytorch_model.bin or model.safetensors, strict=False: the
    appearance head is new). stage1_ckpt_dir: null trains from the current initialization."""
    if not ckpt_dir or str(ckpt_dir).lower() == "none":
        accelerator.print("[Stage2] stage1_ckpt_dir is null: training from the current initialization")
        return
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"stage1_ckpt_dir not found: {ckpt_dir} (set it with --set stage1_ckpt_dir=...)")
    for name in ("pytorch_model.bin", "model.safetensors"):
        path = os.path.join(ckpt_dir, name)
        if os.path.isfile(path):
            if name.endswith(".bin"):
                state = torch.load(path, map_location="cpu")
            else:
                from safetensors.torch import load_file
                state = load_file(path)
            missing, unexpected = model.load_state_dict(state, strict=False)
            not_head = [k for k in missing if not k.startswith("renderer.appearance_head.")]
            accelerator.print(f"[Stage2] loaded {path}: {len(missing)} missing keys (appearance head, new), "
                              f"{len(unexpected)} unexpected")
            if not_head or unexpected:   # loading continues (strict=False), as in the original training
                accelerator.print(f"[Stage2][Warn] checkpoint/model key mismatch: missing outside the appearance head "
                                  f"{len(not_head)} {not_head[:5]}, unexpected {len(unexpected)} {list(unexpected)[:5]}")
            return
    raise FileNotFoundError(f"no pytorch_model.bin or model.safetensors in stage1_ckpt_dir {ckpt_dir}")


@torch.no_grad()
def materialize_lazy_modules(model: ModelHAM, device, image_res: int) -> None:
    """Run one dummy forward pass so that the LazyLinear layers of the appearance head get their
    shapes before the parameters are frozen and handed to the optimizer."""
    intr = torch.eye(4, device=device)
    intr[0, 0] = intr[1, 1] = image_res * 1.2
    intr[0, 2] = intr[1, 2] = image_res * 0.5
    zeros_mano = lambda: {"betas": torch.zeros(1, 10, device=device), "global_orient": torch.zeros(1, 3, device=device),
                          "hand_pose": torch.zeros(1, 45, device=device), "translation": torch.zeros(1, 3, device=device)}
    model.eval()
    model(
        image=torch.zeros(1, 1, 3, image_res, image_res, device=device),
        source_c2ws=torch.eye(4, device=device).view(1, 1, 4, 4),
        source_intrs=intr.view(1, 1, 4, 4),
        render_c2ws=torch.eye(4, device=device).view(1, 1, 4, 4),
        render_intrs=intr.view(1, 1, 4, 4),
        render_bg_colors=torch.zeros(1, 1, 3, device=device),
        mano_params=zeros_mano(),
        source_mano_params=zeros_mano(),
        render_images=torch.zeros(1, 1, 3, image_res, image_res, device=device),
        data={"source_mask": torch.ones(1, 1, image_res, image_res, device=device),
              "render_images": torch.zeros(1, 1, 3, image_res, image_res, device=device),
              "render_fg_masks": torch.ones(1, 1, 1, image_res, image_res, device=device),
              "source_bg_color": torch.zeros(1, 3, device=device)},
        render_full_resolutions=None,
    )
    model.train()


def freeze_all_but_appearance_head(model: ModelHAM) -> None:
    if getattr(model.renderer, "appearance_head", None) is None:
        raise RuntimeError("the model has no appearance head (model.use_appearance_head must be true)")
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.renderer.appearance_head.parameters():
        p.requires_grad_(True)


def main():
    cfg = load_config("configs/train_stage2.yaml", "InstantHand stage-2 training", train_stage=TRAIN_STAGE)
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
    model.to(device)
    load_stage1_weights(model, cfg["stage1_ckpt_dir"], accelerator)
    materialize_lazy_modules(model, device, image_res=data_cfg["dataset"]["source_image_res"])
    model.set_train_stage(TRAIN_STAGE)
    freeze_all_but_appearance_head(model)

    criterion = Stage2Criterion(device=device, **loss_cfg)
    loss_weights = {k: getattr(criterion, k) for k in ("lambda_self_l1", "lambda_self_lpips", "lambda_self_feat_stats",
                                                       "lambda_novel_l1", "lambda_novel_lpips")}
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=optim_cfg["lr"],
                      betas=tuple(optim_cfg["betas"]), weight_decay=optim_cfg["weight_decay"])

    train_ds = MultiHandDataset(build_dataset_config(
        data_cfg, TRAIN_STAGE, optim_cfg["steps_per_epoch"], optim_cfg["batch_size"],
        optim_cfg["accum_steps"], accelerator.num_processes))
    train_loader = DataLoader(train_ds, batch_size=optim_cfg["batch_size"], shuffle=False,
                              num_workers=data_cfg["num_workers"], pin_memory=True, drop_last=True,
                              collate_fn=collate_keep_uid, persistent_workers=data_cfg["num_workers"] > 0)

    model, optimizer, train_loader, criterion = accelerator.prepare(model, optimizer, train_loader, criterion)
    scheduler = cosine_warmup_hold_decay_scheduler(optimizer, **optim_cfg["scheduler"])
    raw_model = accelerator.unwrap_model(model)
    raw_model.set_train_stage(TRAIN_STAGE)

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
        accelerator.print("[Loss] " + ", ".join(f"{k}={v}" for k, v in loss_weights.items()))
        if start_epoch or global_step:
            accelerator.print(f"[Resume] start_epoch={start_epoch}, global_step={global_step}")

    def step_fn(batch: Dict[str, Any], step: int) -> Dict[str, torch.Tensor]:
        model_in, gt = pack_inputs(batch, device)
        return criterion(accelerator.unwrap_model(model), model(**model_in), gt)

    def postfix(ema: Dict[str, float]) -> Dict[str, str]:
        return {"loss": f"{ema['loss']:.4f}", "s_l1": f"{ema['self_l1']:.4f}", "s_lp": f"{ema['self_lpips']:.4f}",
                "s_stat": f"{ema['self_feat_stats']:.4f}", "n_l1": f"{ema['novel_l1']:.4f}",
                "n_lp": f"{ema['novel_lpips']:.4f}"}

    train_loop(cfg=cfg, accelerator=accelerator, model=model, optimizer=optimizer, scheduler=scheduler,
               train_loader=train_loader, step_fn=step_fn, loss_keys=LOSS_KEYS, postfix_fn=postfix,
               train_stage=TRAIN_STAGE, start_epoch=start_epoch, global_step=global_step)


if __name__ == "__main__":
    main()
