# Copyright (c) 2026 The InstantHand Authors.
# SPDX-License-Identifier: Apache-2.0
#
# The preprocessing reproduces the evaluation pipeline of HandAvatar
# (https://github.com/SeanChenxy/HandAvatar, MIT License) with its own utilities (external/handavatar),
# so that the source frame is cropped exactly like the evaluation frames of the benchmark.
"""Preprocess one source (anchor) frame into a HandAvatar-style evaluation sample."""
from __future__ import annotations

import pickle
from typing import Any, Dict, Tuple

import numpy as np

from external.handavatar.core.utils.augm_util import augmentation, process_bbox, trans_point2d
from external.handavatar.core.utils.camera_util import (
    apply_global_tfm_to_camera,
    get_rays_from_KRT,
    rays_intersect_3d_bbox,
)
from external.handavatar.core.utils.image_util import load_image

__all__ = ["ANCHOR_IMAGE_SHAPE", "ANCHOR_BASE_SCALE", "preprocess_single_anchor_like_handavatar"]

ANCHOR_IMAGE_SHAPE: Tuple[int, int] = (256, 256)
ANCHOR_BASE_SCALE: float = 1.3   # crop side / tight hand bbox side (1.3: 30% margin)


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.array(x)


def _load_anchor_meta(anchor_pkl: str) -> Dict[str, Any]:
    """Read a one-frame annotation [cameras, mesh_infos, bbox_xywh, frame_name].

    cameras:    intrinsics [3,3], extrinsics [4,4], distortions [5] (optional)
    mesh_infos: poses [48], shape [10], Rh [3] / Th [3] (optional), bbox {min_xyz, max_xyz},
                joint_cam [21,3] / joint_valid [21] (optional)
    bbox_xywh:  hand box in the full image
    """
    with open(anchor_pkl, "rb") as f:
        data = pickle.load(f)
    if not (isinstance(data, (list, tuple)) and len(data) == 4
            and isinstance(data[0], dict) and isinstance(data[1], dict)):
        raise TypeError("anchor_pkl must hold [cameras_dict, mesh_infos_dict, bbox_xywh, frame_name]")
    cameras, mesh_infos, bbox_xywh, frame_name = data
    if not isinstance(mesh_infos.get("bbox"), dict):
        raise KeyError("mesh_infos must contain the 3D box 'bbox' = {min_xyz, max_xyz}")

    def f32(x, shape=None):
        a = _to_numpy(x).astype(np.float32)
        return a.reshape(shape) if shape is not None else a

    meta = dict(
        K=f32(cameras["intrinsics"]),
        E=f32(cameras["extrinsics"]),
        D=f32(cameras.get("distortions", np.zeros(5, np.float32))),
        bbox_xywh=f32(bbox_xywh, 4),
        poses=f32(mesh_infos["poses"], -1),
        shape=f32(mesh_infos["shape"], -1),
        Rh=f32(mesh_infos.get("Rh", np.zeros(3, np.float32)), 3),
        Th=f32(mesh_infos.get("Th", np.zeros(3, np.float32)), 3),
        dst_bbox_3d=mesh_infos["bbox"],
        frame_name=str(frame_name),
    )
    if "joint_cam" in mesh_infos:
        meta["joint_cam"] = f32(mesh_infos["joint_cam"])
    if "joint_valid" in mesh_infos:
        meta["joint_valid"] = f32(mesh_infos["joint_valid"])
    return meta


def preprocess_single_anchor_like_handavatar(
    *,
    anchor_image: str,
    anchor_mask: str,
    anchor_pkl: str,
    bgcolor_rgb255: np.ndarray,
    input_img_shape: Tuple[int, int] = ANCHOR_IMAGE_SHAPE,
    base_scale: float = ANCHOR_BASE_SCALE,
) -> Dict[str, Any]:
    """Crop one source frame like HandAvatar's evaluation loader does.

    The image is undistorted, composited on the background color with the hand mask, cropped
    around the hand box and resized to `input_img_shape`; K follows the crop and the global
    hand transform (Rh, Th) is folded into the camera.

    Returns the keys of a HandAvatar evaluation sample: target_img [H,W,3] and target_alpha_img
    [H,W] in [0,1], cam_K / cam_R / cam_T, dst_global_orient, dst_posevec, dst_shape, bgcolor,
    ray_mask [H*W] with target_rgbs / target_alpha (the ray-mask pixels), img_height / img_width,
    dst_bbox, dst_cam_joints, dst_valid_joints and frame_name.
    """
    meta = _load_anchor_meta(anchor_pkl)

    orig_img = np.array(load_image(anchor_image))
    alpha_mask = np.array(load_image(anchor_mask))
    if alpha_mask.ndim == 2:
        alpha_mask = np.repeat(alpha_mask[..., None], 3, axis=-1)
    elif alpha_mask.ndim == 3 and alpha_mask.shape[2] == 1:
        alpha_mask = np.repeat(alpha_mask, 3, axis=-1)
    if orig_img.shape[:2] != alpha_mask.shape[:2]:
        raise ValueError(f"anchor image / mask size mismatch: {orig_img.shape} vs {alpha_mask.shape}")

    if np.any(np.abs(meta["D"]) > 1e-8):
        import cv2
        orig_img = cv2.undistort(orig_img, meta["K"], meta["D"])
        alpha_mask = cv2.undistort(alpha_mask, meta["K"], meta["D"])

    bg = bgcolor_rgb255.astype(np.float32)
    alpha_norm = alpha_mask / 255.0
    img = alpha_norm * orig_img.astype(np.float32) + (1.0 - alpha_norm) * bg[None, None, :]

    bbox_xywh = process_bbox(meta["bbox_xywh"].astype(np.float32), img.shape[1], img.shape[0])
    img, img2bb_trans, _, aug_param, _, _, alpha = augmentation(
        img, bbox_xywh, "eval",
        exclude_flip=True, input_img_shape=input_img_shape, mask=alpha_mask, base_scale=base_scale,
        scale_factor=0.2, rot_factor=0.0, shift_wh=[bbox_xywh[2], bbox_xywh[3]], gaussian_std=3,
        bordervalue=bg.tolist(),
    )
    img = (img / 255.0).astype(np.float32)
    alpha = alpha.astype(np.float32)
    H, W = img.shape[:2]

    # intrinsics of the crop
    K = meta["K"][:3, :3].copy()
    K[:2, 2] = trans_point2d(K[:2, 2], img2bb_trans)
    K[[0, 1], [0, 1]] = K[[0, 1], [0, 1]] * input_img_shape[0] / (bbox_xywh[2] * aug_param[1])

    E = apply_global_tfm_to_camera(E=meta["E"].copy(), Rh=meta["Rh"], Th=meta["Th"])
    R = E[:3, :3].astype(np.float32)
    T = E[:3, 3].astype(np.float32)

    # pixels whose ray hits the 3D hand box (HandAvatar's ray mask, used for the metrics)
    rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T)
    _, _, ray_mask = rays_intersect_3d_bbox(meta["dst_bbox_3d"], rays_o.reshape(-1, 3), rays_d.reshape(-1, 3))

    poses = meta["poses"].astype(np.float32).reshape(-1)
    joint_valid = meta.get("joint_valid", np.ones((21,), np.float32))
    if isinstance(joint_valid, np.ndarray) and joint_valid.ndim == 2 and joint_valid.shape[1] == 1:
        joint_valid = joint_valid[:, 0]

    return {
        "frame_name": meta["frame_name"],
        "target_img": img,
        "target_alpha_img": alpha[..., 0],
        "img_height": H,
        "img_width": W,
        "ray_mask": ray_mask,
        "target_rgbs": img.reshape(-1, 3)[ray_mask].astype(np.float32),
        "target_alpha": alpha.reshape(-1, 3)[ray_mask][..., 0].astype(np.float32),
        "bgcolor": bgcolor_rgb255.astype(np.float32),
        "cam_K": K.astype(np.float32),
        "cam_R": R,
        "cam_T": T,
        "dst_bbox": meta["dst_bbox_3d"],
        "dst_global_orient": poses[:3],
        "dst_posevec": poses[3:] + 1e-2,   # as in HandAvatar's loader
        "dst_shape": meta["shape"].astype(np.float32).reshape(-1),
        "dst_cam_joints": meta.get("joint_cam", np.zeros((21, 3), np.float32)).astype(np.float32),
        "dst_valid_joints": np.asarray(joint_valid, dtype=np.float32),
    }