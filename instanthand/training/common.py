# Copyright (c) 2026 The InstantHand Authors.
# SPDX-License-Identifier: Apache-2.0
"""Training code shared by train_stage1.py and train_stage2.py.

    load_config           YAML config + command-line overrides
    mixed_precision_mode  accelerate mixed-precision setting from the config
    cosine_warmup_hold_decay_scheduler, lr_multiplier
    collate_keep_uid, next_from_loader
    build_dataset_config  MultiHandDatasetConfig from the `data` section
    resume_state          restore accelerate state, step counters and the LR schedule
    train_loop            epochs of gradient-accumulated steps with logging and checkpoints
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import yaml
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data._utils.collate import default_collate
from tqdm.auto import tqdm

from instanthand.datasets.multi_hand import DatasetSamplingPolicy, MultiHandDatasetConfig


# ---------------------------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------------------------
def load_config(default_cfg: str, description: str) -> Dict[str, Any]:
    """Read the YAML config and apply command-line overrides.

        --cfg configs/train_stage1.yaml
        --resume_dir <accelerate checkpoint dir>
        --set optim.lr=1e-4 --set data.root=/data/train      (values are parsed as YAML)
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--cfg", default=default_cfg)
    p.add_argument("--resume_dir", default=None, help="resume from an accelerate checkpoint directory")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config entry, e.g. --set optim.batch_size=4 (repeatable)")
    args = p.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(value)
    if args.resume_dir is not None:
        cfg["resume_dir"] = args.resume_dir
    cfg.setdefault("resume_dir", None)
    return cfg


def mixed_precision_mode(value) -> str:
    """accelerate mixed-precision mode; YAML reads an unquoted `no` as False."""
    return "no" if value in (False, None) else str(value)


def build_dataset_config(data_cfg: Dict[str, Any], train_stage: str, optimizer_steps_per_epoch: int,
                         batch_size: int, accum_steps: int, world_size: int) -> MultiHandDatasetConfig:
    """MultiHandDatasetConfig from the `data` section. One epoch of the (stochastic) dataset holds
    as many samples as the optimizer steps of one epoch consume on all processes."""
    policies = {name: DatasetSamplingPolicy(**p) for name, p in data_cfg["policies"].items()}
    datasets = data_cfg.get("datasets") or list(policies.keys())
    return MultiHandDatasetConfig(
        root_dirs=data_cfg["root"],
        datasets=tuple(datasets),
        dataset_policies=policies,
        train_stage=train_stage,
        steps_per_epoch=int(optimizer_steps_per_epoch) * int(accum_steps) * int(batch_size) * int(world_size),
        **data_cfg["dataset"],
    )


# ---------------------------------------------------------------------------------------------
# learning-rate schedule: cosine warm-up, hold, cosine decay to min_lr_ratio
# ---------------------------------------------------------------------------------------------
def lr_multiplier(step: int, warmup_steps: int, hold_steps: int, decay_steps: int, min_lr_ratio: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return 0.5 * (1.0 - math.cos(math.pi * float(step) / float(max(1, warmup_steps))))
    hold_end = warmup_steps + hold_steps
    if step < hold_end:
        return 1.0
    if decay_steps > 0 and step < hold_end + decay_steps:
        t = float(step - hold_end) / float(max(1, decay_steps))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))
    return min_lr_ratio


def cosine_warmup_hold_decay_scheduler(optimizer, warmup_steps: int, hold_steps: int, decay_steps: int,
                                       min_lr_ratio: float = 0.1) -> LambdaLR:
    if min(warmup_steps, hold_steps, decay_steps) < 0:
        raise ValueError("warmup_steps, hold_steps and decay_steps must be >= 0")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    return LambdaLR(optimizer, lambda step: lr_multiplier(step, warmup_steps, hold_steps, decay_steps, min_lr_ratio))


# ---------------------------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------------------------
def collate_keep_uid(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """default_collate, with the string sample ids kept as a list."""
    uids = [b.get("uid", "") for b in batch_list]
    out = default_collate([{k: v for k, v in b.items() if k != "uid"} for b in batch_list])
    out["uid"] = uids
    return out


def next_from_loader(loader, iterator):
    """Next batch; restarts the iterator when the loader is exhausted."""
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def to_bvchw(x: torch.Tensor) -> torch.Tensor:
    """[B,C,H,W] -> [B,1,C,H,W]; [B,V,C,H,W] unchanged."""
    if x.ndim == 5:
        return x
    if x.ndim == 4:
        return x.unsqueeze(1)
    raise ValueError(f"expected a 4D or 5D tensor, got {x.ndim}D")


def mano_params_from_batch(batch: Dict[str, Any], device) -> Dict[str, torch.Tensor]:
    """Target MANO parameters: betas [B,10], global_orient [B,V,3], hand_pose [B,V,45], translation [B,V,3]."""
    betas = batch["betas"].to(device)
    if betas.ndim == 1:
        betas = betas.unsqueeze(0)
    global_orient = batch["global_orient"].to(device)
    hand_pose = batch["hand_pose"].to(device)
    translation = batch["translation"].to(device)
    if global_orient.ndim == 2:
        global_orient, hand_pose, translation = global_orient[None], hand_pose[None], translation[None]
    return dict(betas=betas, global_orient=global_orient, hand_pose=hand_pose, translation=translation)


def target_cameras_from_batch(batch: Dict[str, Any], device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Target c2w [B,V,4,4], intrinsics [B,V,4,4] and background colors [B,V,3]."""
    c2ws, intrs, bg = batch["c2ws"].to(device), batch["intrs"].to(device), batch["render_bg_colors"].to(device)
    if c2ws.ndim == 3:
        c2ws, intrs, bg = c2ws[None], intrs[None], bg[None]
    return c2ws, intrs, bg


# ---------------------------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------------------------
def load_meta(resume_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(resume_dir, "meta.json")
    if os.path.isfile(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def resume_state(accelerator, resume_dir: str, optimizer, scheduler, steps_per_epoch: int,
                 sched_cfg: Dict[str, Any]) -> Tuple[int, int]:
    """Load the accelerate state and put the LR schedule at the saved step. Returns (start_epoch, global_step)."""
    accelerator.print(f"[Resume] loading state from {resume_dir}")
    accelerator.load_state(resume_dir)
    meta = load_meta(resume_dir)
    if meta is not None:
        global_step = int(meta.get("global_step", 0))
        start_epoch = int(meta.get("epoch", global_step // max(1, steps_per_epoch)))
    else:  # no meta.json: infer from an "epoch_<n>" directory name
        m = re.search(r"epoch_(\d+)", os.path.basename(resume_dir.rstrip("/")))
        start_epoch = max(0, (int(m.group(1)) if m else -1) + 1)
        global_step = start_epoch * int(steps_per_epoch)

    mult = lr_multiplier(global_step, **sched_cfg)
    for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
        group["lr"] = base_lr * mult
    scheduler.last_epoch = global_step
    scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
    return start_epoch, global_step


def save_checkpoint(accelerator, save_dir: str, epoch: int, global_step: int, steps_per_epoch: int,
                    datasets: Optional[Sequence[str]], train_stage: str) -> None:
    path = os.path.join(save_dir, f"step_{global_step:07d}")
    accelerator.save_state(path)
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump({"epoch": epoch, "global_step": global_step, "steps_per_epoch": steps_per_epoch,
                   "datasets": datasets, "train_stage": train_stage}, f)


# ---------------------------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------------------------
def train_loop(
    *,
    cfg: Dict[str, Any],
    accelerator,
    model,
    optimizer,
    scheduler,
    train_loader,
    step_fn: Callable[[Dict[str, Any], int], Dict[str, torch.Tensor]],
    loss_keys: Sequence[str],
    postfix_fn: Callable[[Dict[str, float]], Dict[str, str]],
    train_stage: str,
    start_epoch: int = 0,
    global_step: int = 0,
) -> None:
    """Run the epochs of `cfg["optim"]`.

    step_fn(batch, global_step) runs the forward pass and returns the losses (keys `loss_keys`,
    "loss" is the total that is backpropagated). Gradients are accumulated over accum_steps
    batches; each optimizer step clips the gradient norm, steps the optimizer and the LR schedule
    and increments global_step. Losses are logged to TensorBoard (one run per epoch), shown as
    EMAs in the progress bar and averaged over all processes at the end of each epoch.
    """
    from torch.utils.tensorboard import SummaryWriter

    optim_cfg, out_cfg = cfg["optim"], cfg["output"]
    device = accelerator.device
    steps_per_epoch = int(optim_cfg["steps_per_epoch"])
    epochs = int(optim_cfg["epochs"])
    total_steps = epochs * steps_per_epoch
    ema, ema_inited = defaultdict(float), defaultdict(bool)
    writer = None
    train_iter = iter(train_loader)
    model.train()

    for epoch in range(max(start_epoch, global_step // max(1, steps_per_epoch)), epochs):
        epoch_start, epoch_end = epoch * steps_per_epoch, (epoch + 1) * steps_per_epoch
        if global_step >= epoch_end:
            continue

        pbar = None
        if accelerator.is_main_process:
            if writer is not None:
                writer.close()
            tb_epoch_dir = os.path.join(out_cfg["tb_dir"], f"epoch_{epoch + 1:04d}")
            os.makedirs(tb_epoch_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_epoch_dir)
            pbar = tqdm(total=steps_per_epoch, initial=max(0, global_step - epoch_start),
                        desc=f"Epoch {epoch + 1}/{epochs}", dynamic_ncols=True, leave=True)

        epoch_sum = {k: torch.zeros(1, device=device) for k in loss_keys}
        epoch_cnt = torch.zeros(1, device=device)

        while global_step < epoch_end and global_step < total_steps:
            batch, train_iter = next_from_loader(train_loader, train_iter)
            with accelerator.accumulate(model):
                losses = step_fn(batch, global_step)
                accelerator.backward(losses["loss"])
                did_step = False
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), optim_cfg["clip_grad"])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    did_step = True
            if not did_step:
                continue

            vals_t = {k: losses[k].detach() for k in loss_keys}
            for k in loss_keys:
                epoch_sum[k] += vals_t[k].view(1)
            epoch_cnt += 1.0

            if accelerator.is_main_process:
                vals = {k: float(v.item()) for k, v in vals_t.items()}
                for k, v in vals.items():
                    ema[k] = v if not ema_inited[k] else out_cfg["ema_beta"] * ema[k] + (1.0 - out_cfg["ema_beta"]) * v
                    ema_inited[k] = True
                if global_step % out_cfg["log_every"] == 0:
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                    for k, v in vals.items():
                        writer.add_scalar(f"train/{k}", v, global_step)
                if (global_step - epoch_start) % out_cfg["tqdm_every"] == 0:
                    pbar.set_postfix({"lr": f"{scheduler.get_last_lr()[0]:.2e}", **postfix_fn(ema)})
                pbar.update(1)
                if out_cfg["ckpt_every"] > 0 and global_step % out_cfg["ckpt_every"] == 0:
                    save_checkpoint(accelerator, out_cfg["save_dir"], epoch, global_step, steps_per_epoch,
                                    cfg["data"].get("datasets"), train_stage)

        accelerator.wait_for_everyone()
        if pbar is not None:
            pbar.close()

        total_cnt = accelerator.gather_for_metrics(epoch_cnt).sum().clamp(min=1.0)
        avg = {k: (accelerator.gather_for_metrics(epoch_sum[k]).sum() / total_cnt).item() for k in loss_keys}
        if accelerator.is_main_process:
            lr_now = scheduler.get_last_lr()[0]
            accelerator.print(f"[Epoch {epoch + 1:04d}] {train_stage} | lr={lr_now:.8f} | "
                              + ", ".join(f"{k}={v:.6f}" for k, v in avg.items())
                              + f" (opt_steps={int(total_cnt.item())}, global_step={global_step})")
            writer.add_scalar("epoch_avg/lr", lr_now, epoch + 1)
            for k, v in avg.items():
                writer.add_scalar(f"epoch_avg/{k}", v, epoch + 1)
            writer.flush()

    if accelerator.is_main_process:
        if writer is not None:
            writer.close()
        print("Training done.")
