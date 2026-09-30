"""Measure InstantHand's efficiency on HanCo: avatar-creation time, rendering FPS and GPU memory.

  Time (s)  Building the avatar from one image (the network forward pass), averaged over
            `creation_repeats` calls after `creation_warmup` discarded calls. The same
            checkpoint is also timed with the appearance head disabled, which splits the time
            into canonical construction and appearance transfer.
  FPS       1 / mean time of one posed rendering call (MANO-driven deformation and
            rasterization) over `render_passes` passes of the test frames, after
            `render_warmup` discarded frames. Inputs of all frames are prepared before timing.
  Mem.      Peak GPU memory of this process during the rendering loop as reported by NVML
            (as in nvidia-smi, including the CUDA context). Needs `pip install nvidia-ml-py`.

Every timed call is bracketed by CUDA synchronization. Data loading, preprocessing, model
loading and CUDA initialization are not timed. The paper reports the mean of 5 separate
runs of this script on one NVIDIA A100 (80 GB).

Usage:
    python eval_efficient.py --cfg configs/eval_efficient.yaml
"""
from __future__ import annotations

import sys

# HandAvatar's config module parses the command line when it is imported and rejects flags
# it does not know, so our arguments are set aside and it is imported with an empty one.
_CLI_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]

import argparse
import contextlib
import gc
import io
import math
import os
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from einops import rearrange
from safetensors.torch import load_file
from tqdm import tqdm

import external.handavatar.configs as ha_configs
from external.handavatar.configs import make_cfg as ha_make_cfg, args as ha_args

from instanthand.datasets.handavatar_to_ham_inputs import handavatar_sample_to_infer_inputs
from instanthand.models.modeling_ham import ModelHAM
from instanthand.models.checkpoint import BACKBONE_PREFIX, resolve_encoder_weights
from instanthand.datasets.handavatar_single import (
    preprocess_single_anchor_like_handavatar,
    ANCHOR_IMAGE_SHAPE,
    ANCHOR_BASE_SCALE,
)

MIB = 1024.0 ** 2
RENDER_INPUT_KEYS = ("mano_params", "render_c2ws", "render_intrs", "render_bg_colors")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _quiet(enabled: bool = True):
    """Hide stdout prints and Python warnings emitted while third-party objects are built."""
    if not enabled:
        yield
        return
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        yield


def paper_round(value: float, digits: int = 2) -> str:
    """Round the unrounded value to `digits` decimals, as for the paper tables."""
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _unbatch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the DataLoader batch dimension (batch size 1)."""
    for k, v in list(batch.items()):
        if torch.is_tensor(v) and v.ndim >= 1:
            batch[k] = v[0]
        elif isinstance(v, (list, tuple)) and len(v) > 0:
            batch[k] = v[0]
    return batch


def _to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, tuple) and hasattr(x, "_fields"):  # namedtuple
        return type(x)(*[_to_device(v, device) for v in x])
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(v, device) for v in x)
    return x


def _progress_dataloader():
    from external.handavatar.core.data.create_dataset2 import create_dataloader  # needs ha_configs.cfg

    return create_dataloader(data_type="progress")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model(
    model_cfg: Dict[str, Any],
    ckpt_dir: str,
    device: torch.device,
    *,
    appearance_head: bool,
    train_stage: str,
    quiet: bool = True,
) -> ModelHAM:
    """Build the model with or without the appearance head and load the same checkpoint.

    Without the appearance head, the checkpoint's appearance-head tensors are not used;
    every other tensor must match exactly. A light checkpoint has no DINOv3 backbone, which
    is then read from encoder_weights_path; a full checkpoint provides it (the DINOv3 file is
    not read).
    """
    ckpt_path = Path(ckpt_dir) / "model.safetensors"
    kwargs = dict(resolve_encoder_weights(model_cfg, ckpt_path), use_appearance_head=appearance_head)
    if not appearance_head:
        kwargs["appearance_head_cfg"] = None
    with _quiet(quiet):
        model = ModelHAM(**kwargs)

    state = load_file(str(ckpt_path), device=str(device))
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state.keys())
    # a light checkpoint has no DINOv3 backbone; it was read from encoder_weights_path
    missing = sorted(k for k in model_keys - ckpt_keys if not k.startswith(BACKBONE_PREFIX))
    if missing:
        raise RuntimeError(f"Checkpoint lacks {len(missing)} tensors of the model, e.g. {missing[:5]}")
    unused = sorted(ckpt_keys - model_keys)
    if unused:
        if appearance_head:
            raise RuntimeError(f"Checkpoint has {len(unused)} tensors the model does not, e.g. {unused[:5]}")
        if not quiet:
            prefixes = sorted({".".join(k.split(".")[:2]) for k in unused})
            print(f"[canonical only] {len(unused)} checkpoint tensors not used: {prefixes}")
        state = {k: v for k, v in state.items() if k in model_keys}

    model.load_state_dict(state, strict=False)  # every key but the backbone was checked above
    model = model.to(device)
    model.set_train_stage(train_stage)
    model.eval()
    return model


@torch.no_grad()
def build_canonical_avatar(model: ModelHAM, anchor_inputs: Dict[str, Any], device: torch.device):
    """Single forward pass: anchor image -> canonical Gaussian avatar."""
    image = anchor_inputs["image"]
    if image.ndim == 5:  # [B, V, 3, H, W] -> first view
        image = image[:, 0]

    mano = anchor_inputs["mano_params"]
    query_points, mano_params = model.renderer.get_query_points_from_params(mano_params=mano, device=device)

    latent_points, image_feats = model.forward_latent_points(image=image, camera=None, query_points=query_points)
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


def render_frame(model: ModelHAM, gs_models, query_points, inputs: Dict[str, Any], H: int, W: int):
    """The call timed for FPS (the same call the metric evaluation uses)."""
    return model.renderer.render_gs(
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


def count_gaussians(gs_models) -> int:
    total = 0
    for g in gs_models if isinstance(gs_models, list) else [gs_models]:
        for attr in ("offset", "xyz", "scaling"):
            t = getattr(g, attr, None)
            if torch.is_tensor(t) and t.ndim >= 2:
                total += int(t.reshape(-1, t.shape[-1]).shape[0])
                break
    return total


# ---------------------------------------------------------------------------
# Timing and memory
# ---------------------------------------------------------------------------
def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(fn: Callable[[], Any], device: torch.device) -> Tuple[Any, float]:
    """sync -> start -> fn() -> sync -> stop; returns (output, seconds)."""
    _sync(device)
    t0 = time.perf_counter()
    out = fn()
    _sync(device)
    return out, time.perf_counter() - t0


def _bench(fn: Callable[[], Any], device: torch.device, warmup: int, repeats: int, desc: str) -> Tuple[Any, float]:
    """Discard `warmup` calls, then return (last output, mean seconds over `repeats` calls)."""
    for _ in range(max(0, warmup)):
        out = fn()
        del out
    _sync(device)
    times: List[float] = []
    out = None
    for _ in tqdm(range(max(1, repeats)), desc=desc, leave=False):
        out, t = _timed(fn, device)
        times.append(t)
    return out, float(np.mean(times))


def _reset_torch_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


class NvmlPeakSampler:
    """Polls NVML in a background thread and keeps the peak GPU memory of this process
    (as nvidia-smi shows it, including the CUDA context). If per-process numbers are not
    available (common in containers) it falls back to the whole device, which also counts
    other processes on the GPU."""

    def __init__(self, device: torch.device, interval_ms: int):
        self.ok = False
        self.status = "disabled"
        self.interval_sec = max(interval_ms, 1) / 1000.0
        self._peak_proc = 0.0
        self._peak_dev = 0.0
        self._proc_seen = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if device.type != "cuda" or interval_ms <= 0:
            return
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nv = pynvml
            self._handle = self._resolve_handle(device)
            self._pid = os.getpid()
            self.ok = True
            self.status = "ok"
        except Exception as e:
            self.status = f"unavailable ({type(e).__name__}: {e})"

    def _resolve_handle(self, device: torch.device):
        nv = self._nv
        cuda_idx = device.index if device.index is not None else torch.cuda.current_device()
        # 1) match by UUID (robust to CUDA_VISIBLE_DEVICES remapping)
        try:
            uuid = str(getattr(torch.cuda.get_device_properties(cuda_idx), "uuid", "")).lower().replace("gpu-", "")
        except Exception:
            uuid = ""
        if uuid:
            for i in range(nv.nvmlDeviceGetCount()):
                h = nv.nvmlDeviceGetHandleByIndex(i)
                u = nv.nvmlDeviceGetUUID(h)
                u = u.decode() if isinstance(u, bytes) else u
                if u.lower().replace("gpu-", "") == uuid:
                    return h
        # 2) CUDA_VISIBLE_DEVICES mapping
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            try:
                ids = [int(x) for x in visible.split(",") if x.strip()]
                return nv.nvmlDeviceGetHandleByIndex(ids[cuda_idx])
            except Exception:
                pass
        # 3) plain index
        return nv.nvmlDeviceGetHandleByIndex(cuda_idx)

    def _sample(self) -> None:
        dev_used = self._nv.nvmlDeviceGetMemoryInfo(self._handle).used / MIB
        proc_used = None
        try:
            for p in self._nv.nvmlDeviceGetComputeRunningProcesses(self._handle):
                if p.pid == self._pid and p.usedGpuMemory is not None:
                    proc_used = p.usedGpuMemory / MIB
        except Exception:
            pass
        with self._lock:
            self._peak_dev = max(self._peak_dev, dev_used)
            if proc_used is not None:
                self._proc_seen = True
                self._peak_proc = max(self._peak_proc, proc_used)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception:
                pass
            self._stop.wait(self.interval_sec)

    def start(self) -> "NvmlPeakSampler":
        if self.ok and self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def reset(self) -> None:
        with self._lock:
            self._peak_proc = 0.0
            self._peak_dev = 0.0

    def read(self) -> Tuple[Optional[float], bool]:
        """(peak MiB, is_per_process). None if NVML is not available."""
        if not self.ok:
            return None, False
        try:
            self._sample()  # one sample at the phase boundary
        except Exception:
            pass
        with self._lock:
            return (self._peak_proc, True) if self._proc_seen else (self._peak_dev, False)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.ok:
            try:
                self._nv.nvmlShutdown()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_creation(model: ModelHAM, anchor: Dict[str, Any], device: torch.device, timing: Dict[str, Any]):
    """Returns (mean seconds, first-call seconds, gs_models, query_points)."""
    inputs = handavatar_sample_to_infer_inputs(anchor, device=device)  # preprocessing, not timed

    cold, first_call = _timed(lambda: build_canonical_avatar(model, inputs, device), device)
    del cold

    (gs_models, query_points), mean_sec = _bench(
        lambda: build_canonical_avatar(model, inputs, device),
        device, timing["creation_warmup"], timing["creation_repeats"], "avatar creation",
    )
    return mean_sec, first_call, gs_models, query_points


def preload_frames(device: torch.device, num_frames: int) -> List[Tuple[int, int, Dict[str, Any]]]:
    """Convert every test frame once and keep only the render inputs, parked on the CPU.
    The copy to the GPU happens per frame outside the timer."""
    frames = []
    cpu = torch.device("cpu")
    for batch in tqdm(_progress_dataloader(), desc="load frames", leave=False):
        if num_frames > 0 and len(frames) >= num_frames:
            break
        batch = _unbatch(batch)
        inputs = handavatar_sample_to_infer_inputs(batch, device=device)
        frames.append((int(batch["img_height"]), int(batch["img_width"]),
                       _to_device({k: inputs[k] for k in RENDER_INPUT_KEYS}, cpu)))
        del inputs
    if not frames:
        raise RuntimeError("No frames loaded from the progress dataloader.")
    return frames


@torch.no_grad()
def measure_rendering(model: ModelHAM, gs_models, query_points, frames, device: torch.device,
                      timing: Dict[str, Any]) -> Tuple[float, int]:
    """Returns (FPS = 1 / mean seconds per frame, number of timed frames)."""
    for i in range(max(0, timing["render_warmup"])):  # warm-up cycles over real frames, discarded
        H, W, cpu_inputs = frames[i % len(frames)]
        inputs = _to_device(cpu_inputs, device)
        out = render_frame(model, gs_models, query_points, inputs, H, W)
        del out, inputs
    _sync(device)

    times: List[float] = []
    passes = max(1, timing["render_passes"])
    for p in range(passes):
        for H, W, cpu_inputs in tqdm(frames, desc=f"rendering pass {p + 1}/{passes}", leave=False):
            inputs = _to_device(cpu_inputs, device)  # not timed
            out, t = _timed(lambda: render_frame(model, gs_models, query_points, inputs, H, W), device)
            times.append(t)
            del out, inputs
    return 1.0 / float(np.mean(times)), len(times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(cfg: Dict[str, Any]) -> None:
    ha_args.cfg = cfg["handavatar_cfg"]
    progress_cfg = ha_make_cfg(ha_args)
    ha_configs.cfg = progress_cfg

    quiet = not cfg.get("verbose", False)
    timing = cfg["timing"]
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    sampler = NvmlPeakSampler(device, int(timing["nvml_poll_ms"])).start()

    anchor = preprocess_single_anchor_like_handavatar(
        anchor_image=cfg["anchor_image"],
        anchor_mask=cfg["anchor_mask"],
        anchor_pkl=cfg["anchor_pkl"],
        bgcolor_rgb255=np.array(progress_cfg.bgcolor, dtype=np.float32),
        input_img_shape=ANCHOR_IMAGE_SHAPE,
        base_scale=ANCHOR_BASE_SCALE,
    )

    # 1) Canonical avatar only: same checkpoint, appearance head disabled
    model = load_model(cfg["model"], cfg["ckpt_dir"], device, appearance_head=False,
                       train_stage=cfg["canonical_only_train_stage"], quiet=quiet)
    _reset_torch_peak(device)
    canonical_sec, _, gs_models, query_points = measure_creation(model, anchor, device, timing)
    del gs_models, query_points, model
    gc.collect()
    _reset_torch_peak(device)

    # 2) Render inputs of all test frames (data only)
    with _quiet(quiet):
        frames = preload_frames(device, int(timing["render_num_frames"]))

    # 3) Full model: avatar creation, then rendering
    model = load_model(cfg["model"], cfg["ckpt_dir"], device, appearance_head=True,
                       train_stage="stage2", quiet=quiet)

    _reset_torch_peak(device)
    sampler.reset()
    full_sec, first_call_sec, gs_models, query_points = measure_creation(model, anchor, device, timing)
    creation_nvml, creation_per_process = sampler.read()
    creation_alloc = torch.cuda.max_memory_allocated(device) / MIB if device.type == "cuda" else float("nan")
    num_gaussians = count_gaussians(gs_models)

    _reset_torch_peak(device)
    sampler.reset()
    fps, timed_frames = measure_rendering(model, gs_models, query_points, frames, device, timing)
    render_nvml, render_per_process = sampler.read()
    sampler.stop()

    # ---- report ----
    def gb(mib: Optional[float]) -> str:
        return "n/a" if mib is None else paper_round(mib / 1024.0)

    def nvml_note(per_process: bool) -> str:
        if not sampler.ok:
            return f"NVML {sampler.status} (needs a GPU and `pip install nvidia-ml-py`)"
        return "NVML process peak" if per_process else "NVML whole-device peak, includes other processes"

    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    rows = [
        ("Gaussians", str(num_gaussians), ""),
        ("Time (s)", paper_round(full_sec),
         f"avatar creation, mean of {timing['creation_repeats']} after {timing['creation_warmup']} warm-up"),
        ("  canonical only", paper_round(canonical_sec), "appearance head disabled, same checkpoint"),
        ("  appearance transfer", paper_round(full_sec - canonical_sec), "difference of the two"),
        ("  first call", paper_round(first_call_sec), "before warm-up; not a table value"),
        ("FPS", paper_round(fps),
         f"rendering only, {timed_frames} frames ({len(frames)} x {max(1, timing['render_passes'])} passes) "
         f"after {timing['render_warmup']} warm-up"),
        ("Mem. (GB)", gb(render_nvml), f"during rendering; {nvml_note(render_per_process)}, incl. CUDA context"),
        ("  avatar creation", gb(creation_nvml),
         f"{nvml_note(creation_per_process)}; PyTorch allocated {paper_round(creation_alloc / 1024.0)} GB"),
    ]
    print(f"\n======== InstantHand efficiency on HanCo ({gpu}) ========")
    for label, value, note in rows:
        print(f"{label:<22}: {value:<8}  {note}".rstrip())


def parse_args() -> Dict[str, Any]:
    p = argparse.ArgumentParser(description="Measure InstantHand's avatar-creation time, FPS and GPU memory.")
    p.add_argument("--cfg", default="configs/eval_efficient.yaml")
    p.add_argument("--ckpt_dir", help="override ckpt_dir in the config")
    p.add_argument("--verbose", action="store_true", help="show logs and warnings from model and data setup")
    args = p.parse_args(_CLI_ARGV)

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.ckpt_dir is not None:
        cfg["ckpt_dir"] = args.ckpt_dir
    cfg["verbose"] = args.verbose
    return cfg


if __name__ == "__main__":
    run(parse_args())