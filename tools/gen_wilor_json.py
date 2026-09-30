# Copyright (c) 2026 The InstantHand Authors.
# SPDX-License-Identifier: Apache-2.0
"""Export the WiLoR estimate of a hand (MANO pose and shape, camera, 2D keypoints) as the json
file that demo.py reads.

Needs a working WiLoR installation (https://github.com/rolpotamias/WiLoR) with its checkpoints;
run it in the WiLoR environment:

    python tools/gen_wilor_json.py --wilor_root /path/to/WiLoR \
        --images my_hand.jpg --out_dir demo/input/json

--images takes image files and/or folders. For every image, the most confident detection is
estimated and <out_dir>/<image name>.json is written; an overlay of the estimated mesh and a short
log go to <out_dir>/vis/. The hand is always estimated as a right hand, as InstantHand expects:
mirror an image of a left hand horizontally before running this script.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MESH_COLOR = (0.25098039, 0.274117647, 0.65882353)


def project_full_img(points: torch.Tensor, cam_trans: torch.Tensor, focal_length: float,
                     img_wh: Tuple[int, int]) -> np.ndarray:
    """Project camera-space points [N,3] into the full image (principal point at the center)."""
    W, H = float(img_wh[0]), float(img_wh[1])
    K = torch.eye(3)
    K[0, 0] = K[1, 1] = focal_length
    K[0, 2], K[1, 2] = W / 2.0, H / 2.0
    pts = points + cam_trans
    pts = pts / pts[..., -1:]
    return (K @ pts.T).T[..., :-1].cpu().numpy()


def pick_detection(detections, rank: int) -> Optional[Tuple[float, List[float]]]:
    """(confidence, bbox xyxy) of the detection with the given confidence rank (0: most
    confident; falls back to 0 if there are fewer detections), or None."""
    found = []
    for det in detections:
        conf = float(det.boxes.conf.detach().cpu().squeeze().item())
        xyxy = det.boxes.data.detach().cpu().squeeze().numpy()[:4].tolist()
        found.append((conf, xyxy))
    if not found:
        return None
    found.sort(key=lambda x: x[0], reverse=True)
    return found[rank] if rank < len(found) else found[0]


def list_images(paths: List[str]) -> List[Path]:
    out: List[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS)
        elif p.is_file():
            out.append(p)
        else:
            print(f"[skip] not found: {p}")
    return out


@torch.no_grad()
def estimate_one(img_path: Path, out_dir: Path, vis_dir: Path, model, model_cfg, detector, renderer,
                 dataset_cls, cam_crop_to_full, recursive_to, device, args) -> bool:
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"[skip] cannot read {img_path}")
        return False
    H, W = img_bgr.shape[:2]
    name = img_path.stem

    picked = pick_detection(detector(img_bgr, conf=args.conf_threshold, verbose=False)[0], args.detection_rank)
    if picked is None:
        print(f"[no hand] {img_path}")
        return False
    conf, bbox = picked

    right = np.array([1.0], dtype=np.float32)          # always estimated as a right hand
    dataset = dataset_cls(model_cfg, img_bgr, np.array([bbox], dtype=np.float32), right,
                          rescale_factor=args.rescale_factor)
    batch = recursive_to(next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False))), device)
    out = model(batch)

    img_size = batch["img_size"].float()
    focal = (model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()).item()
    cam_t = cam_crop_to_full(out["pred_cam"], batch["box_center"].float(), batch["box_size"].float(),
                             img_size, focal).detach().cpu().numpy()[0]
    verts = out["pred_vertices"][0].detach().cpu().numpy()
    joints = out["pred_keypoints_3d"][0].detach().cpu().numpy()
    t = torch.tensor(cam_t, dtype=torch.float32)
    verts_2d = project_full_img(torch.tensor(verts, dtype=torch.float32), t, focal, (W, H))
    joints_2d = project_full_img(torch.tensor(joints, dtype=torch.float32), t, focal, (W, H))

    mano = out["pred_mano_params"]
    to_list = lambda x: x[0].detach().cpu().numpy().tolist()
    result = {
        "image_path": img_path.name,
        "hand_label": "right",
        "pred_cam": [to_list(out["pred_cam"])],
        "pred_mano_params": {"global_orient": [to_list(mano["global_orient"])],
                             "hand_pose": [to_list(mano["hand_pose"])],
                             "betas": [to_list(mano["betas"])]},
        "pred_cam_t": [cam_t.tolist()],
        "focal_length": [[focal, focal]],
        "pred_keypoints_3d": [joints.tolist()],
        "pred_vertices": [verts.tolist()],
        "pred_vertices_2d": [verts_2d.tolist()],
        "pred_keypoints_2d": [joints_2d.tolist()],
    }
    with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(result, f)

    if args.save_mesh:
        renderer.vertices_to_trimesh(verts, cam_t.copy(), MESH_COLOR, is_right=1.0).export(str(vis_dir / f"{name}.obj"))
    if not args.no_overlay:
        rgba = renderer.render_rgba_multiple([verts], cam_t=[cam_t], render_res=(W, H), is_right=[1.0],
                                             mesh_base_color=MESH_COLOR, scene_bg_color=(1, 1, 1),
                                             focal_length=focal)
        img = img_bgr.astype(np.float32)[:, :, ::-1] / 255.0
        overlay = img * (1 - rgba[:, :, 3:]) + rgba[:, :, :3] * rgba[:, :, 3:]
        cv2.imwrite(str(vis_dir / f"{name}.jpg"), (255 * overlay[:, :, ::-1]).astype(np.uint8))
    (vis_dir / f"{name}_meta.txt").write_text(
        f"image={img_path}\nhand_label=right\nconf={conf}\nbbox_xyxy={bbox}\n", encoding="utf-8")
    print(f"[done] {img_path.name} -> {out_dir / (name + '.json')} (conf {conf:.3f})")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="WiLoR estimate -> InstantHand demo json")
    p.add_argument("--wilor_root", required=True, help="WiLoR repository (with pretrained_models/ and mano_data/)")
    p.add_argument("--images", nargs="+", required=True, help="image files and/or folders")
    p.add_argument("--out_dir", required=True, help="where <image name>.json is written")
    p.add_argument("--conf_threshold", type=float, default=0.3)
    p.add_argument("--detection_rank", type=int, default=0, help="0: most confident detection")
    p.add_argument("--rescale_factor", type=float, default=2.0)
    p.add_argument("--save_mesh", action="store_true", help="also write the estimated mesh (.obj) to vis/")
    p.add_argument("--no_overlay", action="store_true", help="skip the mesh overlay (no rendering needed)")
    p.add_argument("--wilor_ckpt", default="pretrained_models/wilor_final.ckpt", help="relative to --wilor_root")
    p.add_argument("--wilor_cfg", default="pretrained_models/model_config.yaml", help="relative to --wilor_root")
    p.add_argument("--detector_ckpt", default="pretrained_models/detector.pt", help="relative to --wilor_root")
    args = p.parse_args()

    images = [q.resolve() for q in list_images(args.images)]
    if not images:
        raise SystemExit("no images")
    out_dir = Path(args.out_dir).resolve()
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # WiLoR resolves its model config (MANO files etc.) relative to its own root
    root = Path(args.wilor_root).resolve()
    sys.path.insert(0, str(root))
    os.chdir(root)
    from ultralytics import YOLO
    from wilor.models import load_wilor
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils import recursive_to
    from wilor.utils.renderer import Renderer, cam_crop_to_full

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_wilor(checkpoint_path=args.wilor_ckpt, cfg_path=args.wilor_cfg)
    model = model.to(device).eval()
    detector = YOLO(args.detector_ckpt).to(device)
    renderer = None if args.no_overlay and not args.save_mesh else Renderer(model_cfg, faces=model.mano.faces)

    n = sum(estimate_one(img, out_dir, vis_dir, model, model_cfg, detector, renderer, ViTDetDataset,
                         cam_crop_to_full, recursive_to, device, args) for img in images)
    print(f"{n}/{len(images)} images -> {out_dir}")


if __name__ == "__main__":
    main()