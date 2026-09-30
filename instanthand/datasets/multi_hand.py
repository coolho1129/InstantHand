# Copyright (c) 2026 The InstantHand Authors.
# SPDX-License-Identifier: Apache-2.0
"""Multi-view hand dataset for training (WiLoR estimates + segmented RGBA frames).

Layout:
    <root>/<dataset>/<subject>/<group>/
        seg_rgba/<frame>.png  (or seg_rgba/sam3/<frame>.png)   RGBA, alpha = hand segmentation
        wilor_out/<frame>.json                                 WiLoR MANO, 2D keypoints, camera
        silhouette/<frame>.png                                 optional hand silhouette
A group is one observation bucket of a subject (a camera, a sequence, a session, ...).

One sample: pick a dataset (by policy weight), a subject (uniformly or by frame count) and a
source frame, then `num_targets` target frames of the same subject: the source itself ("self"),
frames of the source's group ("same") and of other groups ("cross"), falling back to other
buckets when one runs out. Every frame is cropped to a square around its 2D keypoints (enlarged
by kp_enlarge), resized, composited on a random gray background, and left hands are mirrored.
Betas and the focal length of the source are used for all targets.

Cameras: the WiLoR camera is kept in camera space (identity c2w). With force_center_pp the
principal point is moved to the crop center and the translation shifted to compensate at the
hand depth (single-depth approximation; this is what the released models were trained with).
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch3d.transforms import matrix_to_axis_angle
from torch.utils.data import Dataset

__all__ = ["DatasetSamplingPolicy", "MultiHandDatasetConfig", "MultiHandDataset"]

FrameRef = Tuple[int, str]   # (group index within the subject, frame id)


@dataclass
class DatasetSamplingPolicy:
    """Per-dataset sampling: weight of the dataset, stride of the source frames, and how many of
    the num_targets targets are the source itself, from its group and from other groups."""

    dataset_weight: float = 1.0
    source_stride: int = 3
    num_targets: int = 6
    num_self_targets: int = 1
    num_same_group_targets: int = 2
    num_cross_group_targets: int = 3
    allow_cross_group: bool = True

    def normalized_counts(self) -> Tuple[int, int, int]:
        """(self, same, cross) summing to num_targets: cross-group fills or is trimmed first,
        then same-group, then self."""
        n_total = max(1, int(self.num_targets))
        n_self = max(0, int(self.num_self_targets))
        n_same = max(0, int(self.num_same_group_targets))
        n_cross = max(0, int(self.num_cross_group_targets)) if self.allow_cross_group else 0

        s = n_self + n_same + n_cross
        if s < n_total:
            n_cross += n_total - s
        elif s > n_total:
            overflow = s - n_total
            trim = min(overflow, n_cross)
            n_cross -= trim
            overflow -= trim
            trim = min(overflow, n_same)
            n_same -= trim
            overflow -= trim
            n_self = max(0, n_self - overflow)
        return n_self, n_same, n_cross


@dataclass
class MultiHandDatasetConfig:
    root_dirs: Sequence[str] | str = "./data/train"
    datasets: Optional[Tuple[str, ...]] = None          # None: every dataset folder under the roots
    dataset_policies: Dict[str, DatasetSamplingPolicy] = field(default_factory=dict)   # missing: defaults
    subject_sampling: str = "uniform_subject"           # "uniform_subject" | "by_frames"
    steps_per_epoch: int = 2500                          # samples per epoch (the dataset is stochastic)
    train_stage: str = "stage1"                          # "stage1" | "stage2"

    # crops
    render_image_res: int = 256
    source_image_res: int = 256
    render_multiply: int = 16                            # resolution snapped down to a multiple of this
    source_multiply: int = 16
    kp_enlarge: float = 1.3                              # crop side / keypoint box side
    min_side: float = 32.0
    bg_color_choices: Tuple[float, ...] = (0.0, 0.5, 1.0)
    flip_left_hand: bool = True

    # photometric augmentation, one draw shared by the source and all targets (stage 1 only)
    use_photometric_aug: bool = True
    stage1_photo_brightness: float = 0.10
    stage1_photo_contrast: float = 0.10
    stage1_photo_saturation: float = 0.05
    stage1_photo_hue: float = 0.00
    stage1_photo_gamma_min: float = 0.95
    stage1_photo_gamma_max: float = 1.05

    # folders inside a group
    est_dirname: str = "wilor_out"
    rgba_dirname: str = "seg_rgba"
    silhouette_dirname: str = "silhouette"

    # cameras
    use_est_camera: bool = True
    fix_focal_by_source: bool = True                     # targets use the source focal length
    force_center_pp: bool = True
    adjust_cam_t_when_force_center: bool = True

    # masks
    alpha_thr: float = 0.5
    silhouette_thr: float = 0.5
    mask_resize: str = "area"                            # "area" | "nearest"

    # WiLoR json keys
    json_keypoints2d_key: str = "pred_keypoints_2d"
    json_cam_t_key: str = "pred_cam_t"
    json_focal_key: str = "focal_length"
    json_mano_key: str = "pred_mano_params"
    json_hand_label_key: str = "hand_label"

    # retries: frames whose hand mask covers less than min_mask_ratio are skipped
    min_mask_ratio: float = 0.01
    max_resample_trials: int = 20
    max_source_retry_pool: int = 32
    max_target_retry_pool: int = 64
    strict: bool = False                                 # raise on unreadable groups instead of skipping


class MultiHandDataset(Dataset):
    def __init__(self, cfg: MultiHandDatasetConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.subject_sampling in ("uniform_subject", "by_frames")
        assert cfg.mask_resize in ("area", "nearest")
        assert cfg.train_stage in ("stage1", "stage2")

        root_dirs = [cfg.root_dirs] if isinstance(cfg.root_dirs, str) else cfg.root_dirs
        self.root_dirs: List[str] = [str(x) for x in root_dirs]
        for r in self.root_dirs:
            if not os.path.isdir(r):
                raise RuntimeError(f"root_dirs not found: {r}")
        wanted = None if cfg.datasets is None else {str(x).strip() for x in cfg.datasets if str(x).strip()}

        self.subjects: List[Dict[str, Any]] = []
        self.dataset_to_subject_indices: Dict[str, List[int]] = {}
        skipped_groups = skipped_subjects = 0
        subdirs = lambda d: sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))

        for root_dir in self.root_dirs:
            for dname in subdirs(root_dir):
                if wanted is not None and dname not in wanted:
                    continue
                droot = os.path.join(root_dir, dname)
                for subject_id in subdirs(droot):
                    subj_root = os.path.join(droot, subject_id)
                    groups = []
                    for gname in (g for g in subdirs(subj_root) if g != "meta"):
                        group_root = os.path.join(subj_root, gname)
                        rgba_root = os.path.join(group_root, cfg.rgba_dirname)
                        est_root = os.path.join(group_root, cfg.est_dirname)
                        sil_root = os.path.join(group_root, cfg.silhouette_dirname)
                        if not (os.path.isdir(rgba_root) and os.path.isdir(est_root)):
                            continue
                        try:
                            group = self._build_group_meta(dname, subject_id, gname, rgba_root, est_root,
                                                           sil_root if os.path.isdir(sil_root) else None,
                                                           self._get_policy(dname).source_stride)
                        except Exception:
                            if cfg.strict:
                                raise
                            group = None
                        if group is None:
                            skipped_groups += 1
                        else:
                            groups.append(group)
                    if not groups:
                        skipped_subjects += 1
                        continue
                    self.dataset_to_subject_indices.setdefault(dname, []).append(len(self.subjects))
                    self.subjects.append(self._build_subject_meta(dname, subject_id, groups))

        if not self.subjects:
            raise RuntimeError(f"No valid subjects found. roots={self.root_dirs}, "
                               f"requested_datasets={sorted(wanted) if wanted is not None else None}, "
                               f"skipped_subjects={skipped_subjects}, skipped_groups={skipped_groups}")

        self.num_subjects = len(self.subjects)
        self.total_frames = int(sum(len(s["all_frame_refs"]) for s in self.subjects))
        self.available_datasets = sorted(self.dataset_to_subject_indices)
        weights = [max(float(self._get_policy(d).dataset_weight), 1e-8) for d in self.available_datasets]
        self.available_dataset_weights = [w / sum(weights) for w in weights]

    def __len__(self) -> int:
        return int(self.cfg.steps_per_epoch)

    def _get_policy(self, dataset_name: str) -> DatasetSamplingPolicy:
        return self.cfg.dataset_policies.get(dataset_name, DatasetSamplingPolicy())

    # ------------------------------------------------------------------
    # index
    # ------------------------------------------------------------------
    @staticmethod
    def _sort_key_stem(s: str):
        return (0, int(s)) if s.isdigit() else (1, s)

    def _list_stems(self, folder: str, ext: str) -> Set[str]:
        return {os.path.splitext(f)[0] for f in os.listdir(folder) if f.lower().endswith(ext)}

    def _build_group_meta(self, dataset_name, subject_id, group_id, rgba_root, est_root, sil_root,
                          source_stride) -> Optional[Dict[str, Any]]:
        png_ids = self._list_stems(rgba_root, ".png")
        if os.path.isdir(os.path.join(rgba_root, "sam3")):
            png_ids |= self._list_stems(os.path.join(rgba_root, "sam3"), ".png")
        frame_ids = sorted(png_ids & self._list_stems(est_root, ".json"), key=self._sort_key_stem)
        if not frame_ids:
            return None
        stride = max(1, int(source_stride))
        return {"dataset": dataset_name, "subject_id": subject_id, "group_id": group_id,
                "rgba_root": rgba_root, "est_root": est_root, "sil_root": sil_root,
                "frame_ids": frame_ids, "source_frame_ids": frame_ids[0::stride]}

    @staticmethod
    def _build_subject_meta(dataset_name: str, subject_id: str, groups: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_frame_refs = [(gidx, str(fid)) for gidx, g in enumerate(groups) for fid in g["frame_ids"]]
        all_source_refs = [(gidx, str(fid)) for gidx, g in enumerate(groups) for fid in g["source_frame_ids"]]
        return {"dataset": dataset_name, "subject_id": subject_id, "groups": groups,
                "all_frame_refs": all_frame_refs, "all_source_refs": all_source_refs}

    # ------------------------------------------------------------------
    # files
    # ------------------------------------------------------------------
    @staticmethod
    def _find_rgba_path(rgba_root: str, frame_id: str) -> str:
        for p in (os.path.join(rgba_root, f"{frame_id}.png"), os.path.join(rgba_root, "sam3", f"{frame_id}.png")):
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"RGBA not found for frame {frame_id} in {rgba_root} (or its sam3/)")

    @staticmethod
    def _find_json_path(group: Dict[str, Any], frame_id: str) -> str:
        p = os.path.join(group["est_root"], f"{frame_id}.json")
        if os.path.exists(p):
            return p
        raise FileNotFoundError(f"WiLoR json not found: {p}")

    @staticmethod
    def _find_silhouette_path(sil_root: Optional[str], frame_id: str) -> Optional[str]:
        if sil_root is None:
            return None
        p = os.path.join(sil_root, f"{frame_id}.png")
        return p if os.path.exists(p) else None

    def _load_from_json(self, json_path: str) -> Dict[str, Any]:
        """WiLoR output: hand label, betas [10], global orient [3] and hand pose [45] (axis-angle),
        camera translation [3], focal length [2] and 2D keypoints [21,2] of the full image."""
        cfg = self.cfg
        with open(json_path, "r") as f:
            d = json.load(f)

        hand_label = d.get(cfg.json_hand_label_key)
        hand_label = hand_label.lower().strip() if isinstance(hand_label, str) else None

        focal = None
        if cfg.json_focal_key in d:
            fl = np.array(d[cfg.json_focal_key], dtype=np.float32)
            focal = (torch.tensor(fl[0], dtype=torch.float32) if fl.ndim == 2
                     else torch.tensor(fl, dtype=torch.float32).reshape(-1)[:2])

        cam_t = d.get(cfg.json_cam_t_key)
        cam_t = np.array([0.0, 0.0, 0.0] if cam_t is None else cam_t, dtype=np.float32)
        cam_t = torch.tensor(cam_t[0] if cam_t.ndim == 2 else cam_t, dtype=torch.float32).reshape(-1)[:3]

        kpts2d = d.get(cfg.json_keypoints2d_key)
        if kpts2d is None:
            raise KeyError(f"Missing {cfg.json_keypoints2d_key} in {json_path}")
        kpts2d = np.array(kpts2d, dtype=np.float32)
        if kpts2d.ndim == 3:
            kpts2d = kpts2d[0]
        elif kpts2d.ndim != 2:
            raise ValueError(f"Unexpected keypoints2d shape: {kpts2d.shape}")

        pm = d.get(cfg.json_mano_key, {})
        if len(pm) == 0:
            raise KeyError(f"Missing {cfg.json_mano_key} in {json_path}")
        go = torch.tensor(pm["global_orient"], dtype=torch.float32)       # rotation matrices
        hp = torch.tensor(pm["hand_pose"], dtype=torch.float32)
        go_mat = go[0, 0] if go.ndim == 4 else (go[0] if go.ndim == 3 else go)
        if hp.ndim not in (3, 4):
            raise ValueError(f"Unexpected hand_pose shape: {hp.shape}")
        hp_mat = hp[0] if hp.ndim == 4 else hp

        return {
            "hand_label": hand_label,
            "betas": torch.tensor(pm["betas"], dtype=torch.float32).reshape(-1)[:10],
            "global_orient": matrix_to_axis_angle(go_mat),
            "hand_pose": matrix_to_axis_angle(hp_mat).reshape(-1),
            "cam_t_full": cam_t,
            "focal_full": focal,
            "kpts2d_full": kpts2d,
        }

    # ------------------------------------------------------------------
    # preprocessing
    # ------------------------------------------------------------------
    def _pick_bg(self) -> float:
        return float(random.choice(self.cfg.bg_color_choices))

    def _sample_photo_aug_params(self) -> Optional[Dict[str, float]]:
        """One photometric transform shared by the source and all targets of a sample (stage 1)."""
        cfg = self.cfg
        if not cfg.use_photometric_aug or cfg.train_stage == "stage2":
            return None
        b, c, s, h = (float(cfg.stage1_photo_brightness), float(cfg.stage1_photo_contrast),
                      float(cfg.stage1_photo_saturation), float(cfg.stage1_photo_hue))
        gamma_min = max(float(cfg.stage1_photo_gamma_min), 1e-6)
        gamma_max = max(float(cfg.stage1_photo_gamma_max), gamma_min)
        return {
            "brightness": random.uniform(1.0 - b, 1.0 + b),
            "contrast": random.uniform(1.0 - c, 1.0 + c),
            "saturation": random.uniform(1.0 - s, 1.0 + s),
            "hue": random.uniform(-h, h),
            "gamma": random.uniform(gamma_min, gamma_max),
        }

    @staticmethod
    def _apply_photo_aug_np(rgb: np.ndarray, params: Optional[Dict[str, float]]) -> np.ndarray:
        """Brightness, contrast, saturation, hue and gamma on an RGB image in [0,1]."""
        x = rgb.astype(np.float32).clip(0.0, 1.0)
        if params is None:
            return x
        x = x * float(params.get("brightness", 1.0))
        mean = x.mean(axis=(0, 1), keepdims=True)
        x = (x - mean) * float(params.get("contrast", 1.0)) + mean
        gray = 0.299 * x[..., 0:1] + 0.587 * x[..., 1:2] + 0.114 * x[..., 2:3]
        x = ((x - gray) * float(params.get("saturation", 1.0)) + gray).clip(0.0, 1.0)
        hue = float(params.get("hue", 0.0))
        if abs(hue) > 1e-8:
            hsv = cv2.cvtColor((x * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + hue * 179.0) % 180.0
            x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        x = np.power(x.clip(1e-6, 1.0), float(params.get("gamma", 1.0)))
        return x.clip(0.0, 1.0).astype(np.float32)

    def _square_bbox_from_kpts(self, kpts2d: np.ndarray) -> Tuple[float, float, float]:
        """Center and side of the square around the finite keypoints, enlarged by kp_enlarge."""
        cfg = self.cfg
        k = np.asarray(kpts2d, dtype=np.float32)
        finite = np.isfinite(k).all(axis=1)
        if finite.sum() >= 2:
            k = k[finite]
        if k.shape[0] < 2:
            return 0.0, 0.0, float(cfg.min_side)
        x0, x1, y0, y1 = float(k[:, 0].min()), float(k[:, 0].max()), float(k[:, 1].min()), float(k[:, 1].max())
        side = max(max(x1 - x0, y1 - y0) * float(cfg.kp_enlarge), float(cfg.min_side))
        return 0.5 * (x0 + x1), 0.5 * (y0 + y1), side

    @staticmethod
    def _crop_square_with_pad(images: List[np.ndarray], cx: float, cy: float, side: float):
        """Crop the same square from each image (values in [0,1]); outside the image is zero.
        When padding is needed the images are quantized to 8 bit, as in the original pipeline."""
        H, W = images[0].shape[:2]
        half = side / 2.0
        x0, y0 = int(np.floor(cx - half)), int(np.floor(cy - half))
        x1, y1 = int(np.ceil(cx + half)), int(np.ceil(cy + half))
        pad_l, pad_t = max(0, -x0), max(0, -y0)
        pad_r, pad_b = max(0, x1 - W), max(0, y1 - H)
        crops = []
        for img in images:
            if pad_l or pad_t or pad_r or pad_b:
                img = cv2.copyMakeBorder((img * 255.0).clip(0, 255).astype(np.uint8), pad_t, pad_b, pad_l, pad_r,
                                         borderType=cv2.BORDER_CONSTANT, value=0).astype(np.float32) / 255.0
            crops.append(img[y0 + pad_t:y1 + pad_t, x0 + pad_l:x1 + pad_l])
        return crops, {"full_W": W, "full_H": H, "crop_x0": x0, "crop_y0": y0, "crop_side": float(side)}

    def _process_frame_keypoint_centered(self, rgba_path: str, wilor: Dict[str, Any], out_res: int, multiply: int,
                                         bg_color: float, silhouette_path: Optional[str] = None,
                                         photo_aug_params: Optional[Dict[str, float]] = None):
        """Crop, resize and composite one frame.

        Returns rgb [1,3,H,W] on the background, mask [1,1,H,W] (alpha > alpha_thr), silhouette
        mask [1,1,H,W] (the mask when there is no silhouette), intrinsics [4,4], translation [3]
        and {"full_H", "full_W"} of the original frame.
        """
        cfg = self.cfg
        rgba = np.array(Image.open(rgba_path).convert("RGBA")).astype(np.float32) / 255.0
        rgb_full, a_full = rgba[..., :3], rgba[..., 3]
        H, W = rgb_full.shape[:2]

        sil_full = None
        if silhouette_path is not None and os.path.exists(silhouette_path):
            sil_full = np.array(Image.open(silhouette_path).convert("L")).astype(np.float32) / 255.0
            if sil_full.shape[:2] != (H, W):
                sil_full = cv2.resize(sil_full, (W, H), interpolation=cv2.INTER_NEAREST)

        kpts2d = np.array(wilor["kpts2d_full"], dtype=np.float32)
        cam_t_full = wilor["cam_t_full"].clone()
        if cfg.flip_left_hand and wilor.get("hand_label") == "left":        # mirror left hands to right
            rgb_full, a_full = rgb_full[:, ::-1, :].copy(), a_full[:, ::-1].copy()
            if sil_full is not None:
                sil_full = sil_full[:, ::-1].copy()
            kpts2d[:, 0] = (W - 1.0) - kpts2d[:, 0]
            cam_t_full[0] = -cam_t_full[0]

        cx, cy, side = self._square_bbox_from_kpts(kpts2d)
        (rgb_c, a_c), crop = self._crop_square_with_pad([rgb_full, a_full], cx, cy, side)
        sil_c = self._crop_square_with_pad([sil_full], cx, cy, side)[0][0] if sil_full is not None else None

        out_hw = max(int(out_res // multiply) * int(multiply), int(multiply)) if multiply > 1 else int(out_res)
        mask_interp = cv2.INTER_AREA if cfg.mask_resize == "area" else cv2.INTER_NEAREST
        rgb_r = self._apply_photo_aug_np(cv2.resize(rgb_c, (out_hw, out_hw), interpolation=cv2.INTER_AREA),
                                         photo_aug_params)
        mask_r = (cv2.resize(a_c, (out_hw, out_hw), interpolation=mask_interp) > float(cfg.alpha_thr)).astype(np.float32)
        rgb_out = rgb_r * mask_r[..., None] + float(bg_color) * (1.0 - mask_r[..., None])
        if sil_c is not None:
            sil_r = (cv2.resize(sil_c, (out_hw, out_hw), interpolation=mask_interp) > float(cfg.silhouette_thr))
            sil_r = sil_r.astype(np.float32)
        else:
            sil_r = mask_r

        # intrinsics of the crop: full-image principal point at the image center, scaled to out_hw
        focal_full = wilor.get("focal_full")
        if focal_full is None:
            raise KeyError(f"[{rgba_path}] focal_length missing in WiLoR json")
        s = float(out_hw) / max(crop["crop_side"], 1e-9)
        fx, fy = float(focal_full[0]) * s, float(focal_full[1]) * s
        cx_true = (W * 0.5 - float(crop["crop_x0"])) * s
        cy_true = (H * 0.5 - float(crop["crop_y0"])) * s
        cam_t = cam_t_full
        if cfg.force_center_pp:
            cx_pp = cy_pp = float(out_hw) * 0.5
            if cfg.adjust_cam_t_when_force_center:      # shift so that the hand depth projects unchanged
                tx, ty, tz = cam_t_full.tolist()
                cam_t = torch.tensor([tx - (tz / max(fx, 1e-9)) * (cx_pp - cx_true),
                                      ty - (tz / max(fy, 1e-9)) * (cy_pp - cy_true), tz], dtype=torch.float32)
        else:
            cx_pp, cy_pp = cx_true, cy_true
        intr4 = torch.eye(4, dtype=torch.float32)
        intr4[0, 0], intr4[1, 1], intr4[0, 2], intr4[1, 2] = fx, fy, cx_pp, cy_pp

        to_t = lambda a: torch.from_numpy(a).float()
        return (to_t(rgb_out).permute(2, 0, 1)[None], to_t(mask_r)[None, None], to_t(sil_r)[None, None],
                intr4, cam_t, {"full_H": H, "full_W": W})

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    @staticmethod
    def _dedup_keep_order(refs: List[FrameRef]) -> List[FrameRef]:
        seen: Set[FrameRef] = set()
        return [x for x in refs if not (x in seen or seen.add(x))]

    @staticmethod
    def _shuffled(refs: List[FrameRef]) -> List[FrameRef]:
        out = list(refs)
        random.shuffle(out)
        return out

    def _choose_subject(self) -> Dict[str, Any]:
        dataset_name = random.choices(self.available_datasets, weights=self.available_dataset_weights, k=1)[0]
        indices = self.dataset_to_subject_indices[dataset_name]
        if self.cfg.subject_sampling == "uniform_subject":
            return self.subjects[int(random.choice(indices))]
        lens = [len(self.subjects[i]["all_frame_refs"]) for i in indices]
        return self.subjects[int(random.choices(indices, weights=lens, k=1)[0])]

    def _target_candidates(self, subject: Dict[str, Any], src_ref: FrameRef, bucket: str,
                           used: Set[FrameRef]) -> List[FrameRef]:
        """Candidate frames for one target, in order of preference for its bucket."""
        src_gidx, src_fid = src_ref
        refs = subject["all_frame_refs"]
        same = self._shuffled([(g, f) for g, f in refs if g == src_gidx and f != src_fid and (g, f) not in used])
        cross = self._shuffled([(g, f) for g, f in refs if g != src_gidx and (g, f) not in used])
        others = self._shuffled([(g, f) for g, f in refs if (g, f) != (src_gidx, src_fid) and (g, f) not in used])
        if bucket == "self":
            return [src_ref]
        order = (same + cross) if bucket == "same" else (cross + same)
        return self._dedup_keep_order(order + others + [src_ref])[:max(1, self.cfg.max_target_retry_pool)]

    def _load_with_retry(self, subject: Dict[str, Any], candidates: List[FrameRef], out_res: int, multiply: int,
                         focal_override: Optional[torch.Tensor], photo_aug_params: Optional[Dict[str, float]]):
        """First candidate (in random order, at most max_resample_trials tried) whose hand mask covers
        at least min_mask_ratio of the crop; the first candidate if none does.

        Returns (ref, wilor, rgb, mask, silhouette, intrinsics, translation, meta with "bg")."""
        cfg = self.cfg
        cands = self._dedup_keep_order(list(candidates))
        if not cands:
            raise RuntimeError("no candidate frames")
        random.shuffle(cands)

        def load(ref):
            group = subject["groups"][ref[0]]
            rgba_path = self._find_rgba_path(group["rgba_root"], ref[1])
            wilor = self._load_from_json(self._find_json_path(group, ref[1]))
            if focal_override is not None:
                wilor["focal_full"] = focal_override
            sil_path = self._find_silhouette_path(group["sil_root"], ref[1])
            bg = self._pick_bg()
            out = self._process_frame_keypoint_centered(rgba_path, wilor, out_res, multiply, bg, sil_path,
                                                        photo_aug_params)
            out[5]["bg"] = float(bg)
            return (ref, wilor) + out

        for ref in cands[:int(cfg.max_resample_trials)]:
            try:
                result = load(ref)
            except Exception:
                continue
            if float(result[3].mean().item()) >= float(cfg.min_mask_ratio):
                return result
        return load(cands[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cfg = self.cfg
        subject = self._choose_subject()
        dataset_name = subject["dataset"]
        src_candidates = self._shuffled(subject["all_source_refs"])[:max(1, cfg.max_source_retry_pool)]
        photo_aug = self._sample_photo_aug_params()

        (src_ref, src_wilor, src_rgb, src_mask, src_sil, src_intr, src_cam_t, src_meta) = self._load_with_retry(
            subject, src_candidates, cfg.source_image_res, cfg.source_multiply, None, photo_aug)

        focal = None
        if cfg.use_est_camera and cfg.fix_focal_by_source:
            if src_wilor["focal_full"] is None:
                raise KeyError(f"[{dataset_name}/{subject['subject_id']}] focal_length missing in the source json")
            focal = src_wilor["focal_full"].clone()

        n_self, n_same, n_cross = self._get_policy(dataset_name).normalized_counts()
        used: Set[FrameRef] = set()
        targets = []
        for bucket in ["self"] * n_self + ["same"] * n_same + ["cross"] * n_cross:
            candidates = self._target_candidates(subject, src_ref, bucket, used if bucket != "self" else set())
            ref, wilor, rgb, mask, sil, intr, cam_t, meta = self._load_with_retry(
                subject, candidates, cfg.render_image_res, cfg.render_multiply, focal, photo_aug)
            if bucket != "self":
                used.add(ref)
            targets.append((wilor, rgb, mask, sil, intr, cam_t, meta))

        n = len(targets)
        return {
            "uid": f"{dataset_name}::{subject['subject_id']}",
            "dataset": dataset_name,
            "subject_id": subject["subject_id"],
            "train_stage": cfg.train_stage,
            # source view
            "source_c2ws": torch.eye(4, dtype=torch.float32)[None],
            "source_intrs": src_intr[None],
            "source_rgbs": src_rgb.clamp(0, 1),
            "source_mask": src_mask.clamp(0, 1),
            "source_silhouette_mask": src_sil.clamp(0, 1),
            "source_bg_color": torch.tensor([float(src_meta.get("bg", 0.0))] * 3, dtype=torch.float32),
            # target views [N,...]
            "render_image": torch.cat([t[1] for t in targets], dim=0).clamp(0, 1),
            "render_mask": torch.cat([t[2] for t in targets], dim=0).clamp(0, 1),
            "render_silhouette_mask": torch.cat([t[3] for t in targets], dim=0).clamp(0, 1),
            "c2ws": torch.eye(4, dtype=torch.float32)[None].repeat(n, 1, 1),
            "intrs": torch.stack([t[4] for t in targets], dim=0),
            "render_bg_colors": torch.tensor([float(t[6].get("bg", 0.0)) for t in targets],
                                             dtype=torch.float32).unsqueeze(-1).repeat(1, 3),
            "render_full_resolutions": torch.tensor([[float(t[6]["full_H"]), float(t[6]["full_W"])] for t in targets],
                                                    dtype=torch.float32),
            # target MANO (betas of the source for all views)
            "betas": src_wilor["betas"].clone(),
            "global_orient": torch.stack([t[0]["global_orient"] for t in targets], dim=0),
            "hand_pose": torch.stack([t[0]["hand_pose"] for t in targets], dim=0),
            "translation": torch.stack([t[5] for t in targets], dim=0),
            # source MANO
            "source_global_orient": src_wilor["global_orient"],
            "source_hand_pose": src_wilor["hand_pose"],
            "source_translation": src_cam_t,
            "source_betas": src_wilor["betas"].clone(),
        }
