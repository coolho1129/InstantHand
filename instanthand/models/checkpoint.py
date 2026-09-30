# Copyright (c) 2026 The InstantHand Authors.
# SPDX-License-Identifier: Apache-2.0
"""Loading an InstantHand checkpoint, with or without the frozen DINOv3 backbone (encoder.model.*).

full checkpoint : contains the backbone; the DINOv3 weight file is not needed and not read.
light checkpoint: without the backbone; it is read from encoder_weights_path (the DINOv3 file)."""
from pathlib import Path
from typing import Any, Dict, Union

import torch
from safetensors import safe_open
from safetensors.torch import load_file

BACKBONE_PREFIX = "encoder.model."


def checkpoint_has_backbone(path: Union[str, Path]) -> bool:
    with safe_open(str(path), "pt") as f:
        return any(k.startswith(BACKBONE_PREFIX) for k in f.keys())


def resolve_encoder_weights(model_cfg: Dict[str, Any], ckpt_path: Union[str, Path]) -> Dict[str, Any]:
    """Model config for this checkpoint: the DINOv3 file is read only if the checkpoint lacks the backbone."""
    cfg = dict(model_cfg)
    if checkpoint_has_backbone(ckpt_path):
        cfg["encoder_weights_path"] = None
        return cfg
    p = cfg.get("encoder_weights_path")
    if not p or not Path(p).is_file():
        raise SystemExit(f"{ckpt_path} does not contain the DINOv3 backbone: download the DINOv3 weights to "
                         f"{p} (README, 'DINOv3 weights') or use the checkpoint that includes them")
    return cfg


def load_checkpoint(model: torch.nn.Module, path: Union[str, Path], device) -> None:
    missing, unexpected = model.load_state_dict(load_file(str(path), device=str(device)), strict=False)
    other = [k for k in missing if not k.startswith(BACKBONE_PREFIX)]
    if other or unexpected:
        raise RuntimeError(f"{path} does not match the model: missing {other[:5]}, unexpected {list(unexpected)[:5]}")
    print(f"[ckpt] DINOv3 backbone from {'encoder_weights_path' if missing else 'the checkpoint'}")
