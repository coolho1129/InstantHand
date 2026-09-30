# hanco/train.py
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import json
import pickle
import numpy as np
import cv2
import torch

from external.handavatar.core.utils.image_util import load_image
from external.handavatar.core.utils.hand_util import body_pose_to_body_RTs
from external.handavatar.core.utils.camera_util import (
    apply_global_tfm_to_camera,
    get_rays_from_KRT,
    rays_intersect_3d_bbox,
    cam2pixel,
)
from external.handavatar.core.utils.augm_util import process_bbox, augmentation, trans_point2d
from external.handavatar.configs import cfg
import smplx


PAPER_HANCO_CAMERAS = {
    "0191": ["cam3", "cam5", "cam6", "cam7"],
}


def sorted_nicely(xs):
    import re

    def key(s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", s)]

    return sorted(xs, key=key)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_mask_binary(path, mask_thr=0.5):
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)

    if m.ndim == 3:
        m = m[..., 0]

    if m.dtype == np.uint8:
        m = m.astype(np.float32) / 255.0
    else:
        m = m.astype(np.float32)

    return (m >= mask_thr).astype(np.uint8)


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min = float(xs.min())
    x_max = float(xs.max())
    y_min = float(ys.min())
    y_max = float(ys.max())

    return np.array(
        [x_min, y_min, x_max - x_min + 1.0, y_max - y_min + 1.0],
        dtype=np.float32,
    )


def parse_vec61_mano(path):
    vec = np.asarray(load_json(path), dtype=np.float32).reshape(-1)
    if vec.shape[0] != 61:
        raise ValueError(f"Expected 61D MANO vec, got {vec.shape[0]}: {path}")

    poses = vec[:48]
    betas = vec[48:58]
    trans = vec[58:61]
    return poses, betas, trans


def parse_calib_json(path):
    data = load_json(path)
    Ks = data["K"]
    Ms = data["M"]

    camK = {}
    camM = {}
    for i, (K, M) in enumerate(zip(Ks, Ms)):
        cam = f"cam{i}"
        camK[cam] = np.asarray(K, dtype=np.float32).reshape(3, 3)
        camM[cam] = np.asarray(M, dtype=np.float32).reshape(4, 4)

    return camK, camM


def world_joints_to_camera(joints_world, M):
    M = np.asarray(M, dtype=np.float32).reshape(4, 4)
    R = M[:3, :3]
    t = M[:3, 3]
    return (R @ joints_world.T).T + t[None, :]


def project_points(K, pts_cam):
    z = np.clip(pts_cam[:, 2], 1e-8, None)
    u = K[0, 0] * (pts_cam[:, 0] / z) + K[0, 2]
    v = K[1, 1] * (pts_cam[:, 1] / z) + K[1, 2]
    return np.stack([u, v], axis=-1).astype(np.int32)


class Dataset(torch.utils.data.Dataset):

    @torch.no_grad()
    def __init__(
            self,
            dataset_path,
            keyfilter=None,
            maxframes=-1,
            bgcolor=None,
            ray_shoot_mode='image',
            skip=1,
            subject=None,
            **kwargs):

        print('[Dataset Path]', dataset_path)

        # MANO
        self.mano = smplx.create(**cfg.smpl_cfg)
        self.handtype = ('left', 'right')[cfg.smpl_cfg.is_rhand]
        if self.handtype == 'left':
            self.mano.shapedirs[:, 0, :] *= -1

        self.phase = kwargs.get('data_type', 'train')
        if subject is None:
            if self.phase == 'train':
                subject = cfg.subject
            else:
                subject = cfg[kwargs['data_type']].subject

        if isinstance(subject, list):
            subject = subject[0]

        self.sequence = subject
        self.seq_root = os.path.join(dataset_path, self.sequence)
        self.rgb_root = os.path.join(self.seq_root, "rgb")
        self.mask_root = os.path.join(self.seq_root, "mask")
        self.shape_root = os.path.join(self.seq_root, "shape")
        self.xyz_root = os.path.join(self.seq_root, "xyz")
        self.calib_root = os.path.join(self.seq_root, "calib")

        self.image_dir = self.rgb_root

        anno_name = os.path.join(self.seq_root, "pkl", "anno_cam_hanco_scale1.pkl")

        print('Load annotation', anno_name)
        if not os.path.exists(anno_name):
            print('Preprocessing HanCo ...')
            self.preprocess(dataset_path, subject, anno_name, self.phase)

        with open(anno_name, 'rb') as f:
            self.cameras, self.mesh_infos, self.bbox, self.framelist = pickle.load(f)

        # canonical
        mano_res = self.mano(
            torch.zeros(1, 10).float(),
            torch.zeros(1, 3).float(),
            torch.zeros(1, 45).float(),
            return_verts=True
        )
        self.canonical_bbox = mano_res.joints[0].numpy()
        self.canonical_verts = mano_res.vertices[0].numpy()
        self.canonical_bbox = self.skeleton_to_bbox(self.canonical_verts)

        # post process
        self.framelist = self.framelist[::skip]

        exclude = cfg[kwargs['data_type']].get('exclude_idx', None)
        if isinstance(exclude, list):
            sel_idx = list(range(len(self.framelist)))
            self.framelist = [self.framelist[i] for i in sel_idx if i not in exclude]

        if maxframes > 0:
            self.framelist = self.framelist[:maxframes]

        print(f' -- Total Frames: {self.get_total_frames()}')

        self.keyfilter = keyfilter
        self.bgcolor = bgcolor
        self.ray_shoot_mode = ray_shoot_mode

    @torch.no_grad()
    def preprocess(self, dataset_path, subject, anno_name, data_type, with_mask=True):
        th_hands_mean_right = np.array([
            0.1117, -0.0429, 0.4164, 0.1088, 0.0660, 0.7562, -0.0964, 0.0909,
            0.1885, -0.1181, -0.0509, 0.5296, -0.1437, -0.0552, 0.7049, -0.0192,
            0.0923, 0.3379, -0.4570, 0.1963, 0.6255, -0.2147, 0.0660, 0.5069,
            -0.3697, 0.0603, 0.0795, -0.1419, 0.0859, 0.6355, -0.3033, 0.0579,
            0.6314, -0.1761, 0.1321, 0.3734, 0.8510, -0.2769, 0.0915, -0.4998,
            -0.0266, -0.0529, 0.5356, -0.0460, 0.2774,
        ], dtype=np.float32)

        th_hands_mean_left = th_hands_mean_right.copy().reshape(-1, 3)
        th_hands_mean_left[:, 1:] *= -1
        th_hands_mean_left = th_hands_mean_left.reshape(-1)

        seq_root = os.path.join(dataset_path, subject)
        rgb_root = os.path.join(seq_root, "rgb")
        mask_root = os.path.join(seq_root, "mask")
        shape_root = os.path.join(seq_root, "shape")
        xyz_root = os.path.join(seq_root, "xyz")
        calib_root = os.path.join(seq_root, "calib")

        cams = sorted_nicely([
            d for d in os.listdir(rgb_root)
            if d.startswith("cam") and os.path.isdir(os.path.join(rgb_root, d))
        ])

        if subject in PAPER_HANCO_CAMERAS:
            cams = [c for c in cams if c in PAPER_HANCO_CAMERAS[subject]]

        print("[HanCo preprocess] cams:", cams)

        calib_by_frame = {}
        for fname in sorted_nicely(os.listdir(calib_root)):
            if not fname.endswith(".json"):
                continue
            frame_stem = os.path.splitext(fname)[0]
            calib_by_frame[frame_stem] = parse_calib_json(os.path.join(calib_root, fname))

        self.cameras = {}
        self.mesh_infos = {}
        self.bbox = {}
        self.framelist = []

        for cam in cams:
            rgb_dir = os.path.join(rgb_root, cam)
            mask_dir = os.path.join(mask_root, cam)
            shape_dir = os.path.join(shape_root, cam)

            rgb_files = sorted_nicely([
                f for f in os.listdir(rgb_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ])

            for i, rgb_file in enumerate(rgb_files):
                if i % 500 == 0:
                    print(f"[{cam}] {i}/{len(rgb_files)}")

                frame_stem = os.path.splitext(rgb_file)[0]

                img_path = os.path.join(rgb_dir, rgb_file)
                mask_path = os.path.join(mask_dir, frame_stem + ".png")
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(mask_dir, frame_stem + ".jpg")

                shape_path = os.path.join(shape_dir, frame_stem + ".json")
                xyz_path = os.path.join(xyz_root, frame_stem + ".json")

                if not os.path.exists(mask_path):
                    continue
                if not os.path.exists(shape_path):
                    continue
                if not os.path.exists(xyz_path):
                    continue
                if frame_stem not in calib_by_frame:
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    continue

                img_height, img_width = img.shape[:2]

                if data_type != 'infer':
                    if img.max() < 20:
                        continue
                    if (
                        np.allclose(img[..., 0], img[..., 1], atol=1)
                        or np.allclose(img[..., 2], img[..., 1], atol=1)
                        or np.allclose(img[..., 0], img[..., 2], atol=1)
                    ):
                        continue

                mask_bin = load_mask_binary(mask_path)
                mask_sum = mask_bin.astype(bool).sum()
                if data_type != 'infer':
                    if mask_sum < 1000:
                        continue
                    sel_img = img[mask_bin.astype(bool)]
                    if sel_img.size == 0 or sel_img.mean(axis=-1).max() < 20:
                        continue

                bbox_xywh = bbox_from_mask(mask_bin)
                if bbox_xywh is None:
                    continue

                if data_type != 'infer':
                    if (
                        bbox_xywh[0] < 10
                        or bbox_xywh[1] < 10
                        or max(bbox_xywh[2], bbox_xywh[3]) < 80
                        or bbox_xywh[0] + bbox_xywh[2] > img_width - 10
                        or bbox_xywh[1] + bbox_xywh[3] > img_height - 10
                    ):
                        continue

                bbox_xywh = process_bbox(bbox_xywh, img_width, img_height)

                camK_by_name, camM_by_name = calib_by_frame[frame_stem]
                if cam not in camK_by_name or cam not in camM_by_name:
                    continue

                K = camK_by_name[cam].astype(np.float32)
                M = camM_by_name[cam].astype(np.float32)

                E = np.eye(4, dtype=np.float32)

                frame_name = f"{subject}/{cam}/{frame_stem}"

                self.cameras[frame_name] = {
                    'intrinsics': K,
                    'extrinsics': E,
                    'distortions': np.zeros(5, dtype=np.float32),
                }

                # MANO
                poses, betas, trans = parse_vec61_mano(shape_path)

                # HanCo shape/cam json is assumed to already contain camera-frame root rotation.
                poses = poses.astype(np.float32)
                poses[3:] += (th_hands_mean_left, th_hands_mean_right)[self.handtype == 'right']

                tres = self.mano(
                    torch.from_numpy(betas)[None].float(),
                    torch.zeros(1, 3).float(),
                    torch.zeros(1, 45).float(),
                    return_verts=True
                )
                tpose_joints = tres.joints[0].numpy()

                posed_res = self.mano(
                    torch.from_numpy(betas)[None].float(),
                    torch.from_numpy(poses)[None, :3].float(),
                    torch.from_numpy(poses)[None, 3:].float(),
                    return_verts=True
                )

                joints = posed_res.joints[0].numpy()
                verts = posed_res.vertices[0].numpy()

                xyz_world = np.asarray(load_json(xyz_path), dtype=np.float32)
                if xyz_world.ndim != 2 or xyz_world.shape[1] != 3:
                    continue

                joint_cam = world_joints_to_camera(xyz_world, M)

                # scale consistency with MANO
                joint_cam = joint_cam.astype(np.float32) * cfg.smpl_cfg.scale

                # If HanCo xyz is already in MANO-scale meters, set cfg.smpl_cfg.scale=1.
                joint_img = project_points(K, joint_cam)
                joint_valid = np.ones((joint_cam.shape[0], 1), dtype=np.float32)

                self.mesh_infos[frame_name] = {
                    'Rh': np.zeros(3, dtype=np.float32),
                    'Th': (joint_cam[0] - joints[0]).astype(np.float32),
                    'poses': poses.reshape(-1).astype(np.float32),
                    'shape': betas.astype(np.float32),
                    'joints': joints.astype(np.float32),
                    'tpose_joints': tpose_joints.astype(np.float32),
                    'bbox': self.skeleton_to_bbox(verts),
                    'joint_img': joint_img.astype(np.int32),
                    'joint_cam': joint_cam.astype(np.float32),
                    'joint_valid': joint_valid.astype(np.float32),

                    # HanCo path info
                    'rgb_path': img_path,
                    'mask_path': mask_path,
                    'cam_name': cam,
                    'frame_stem': frame_stem,
                }

                self.bbox[frame_name] = bbox_xywh.astype(np.float32)
                self.framelist.append(frame_name)

        if len(self.framelist) == 0:
            raise RuntimeError("No valid HanCo samples after preprocessing.")

        os.makedirs(os.path.dirname(anno_name), exist_ok=True)
        with open(anno_name, 'wb') as f:
            pickle.dump([self.cameras, self.mesh_infos, self.bbox, self.framelist], f)

        print(f"[HanCo preprocess] saved: {anno_name}")
        print(f"[HanCo preprocess] samples: {len(self.framelist)}")

    @staticmethod
    def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - cfg.bbox_offset
        max_xyz = np.max(skeleton, axis=0) + cfg.bbox_offset
        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz,
        }

    def query_dst_skeleton(self, frame_name):
        mi = self.mesh_infos[frame_name]
        return {
            'poses': mi['poses'].astype('float32'),
            'shape': mi['shape'].astype('float32'),
            'dst_tpose_joints': mi['tpose_joints'].astype('float32'),
            'bbox': mi['bbox'].copy(),
            'Rh': mi['Rh'].astype('float32'),
            'Th': mi['Th'].astype('float32'),
            'joint_img': mi['joint_img'].astype('int32'),
            'joint_cam': mi['joint_cam'].astype('float32'),
            'joint_valid': mi['joint_valid'].astype('float32'),
        }

    @staticmethod
    def select_rays(select_inds, rays_o, rays_d, ray_img, ray_alpha, near, far):
        return (
            rays_o[select_inds],
            rays_d[select_inds],
            ray_img[select_inds],
            ray_alpha[select_inds],
            near[select_inds],
            far[select_inds],
        )

    def get_patch_ray_indices(self, N_patch, ray_mask, subject_mask, bbox_mask, patch_size, H, W):
        assert subject_mask.dtype == np.bool_
        assert bbox_mask.dtype == np.bool_

        bbox_exclude_subject_mask = np.bitwise_and(
            bbox_mask,
            np.bitwise_not(subject_mask)
        )

        list_ray_indices = []
        list_mask = []
        list_xy_min = []
        list_xy_max = []

        total_rays = 0
        patch_div_indices = [total_rays]

        for _ in range(N_patch):
            if np.random.rand(1)[0] < cfg.patch.sample_subject_ratio:
                candidate_mask = subject_mask
            else:
                candidate_mask = bbox_exclude_subject_mask

            ray_indices, mask, xy_min, xy_max = self._get_patch_ray_indices(
                ray_mask, candidate_mask, patch_size, H, W
            )

            total_rays += len(ray_indices)
            list_ray_indices.append(ray_indices)
            list_mask.append(mask)
            list_xy_min.append(xy_min)
            list_xy_max.append(xy_max)
            patch_div_indices.append(total_rays)

        select_inds = np.concatenate(list_ray_indices, axis=0)
        patch_info = {
            'mask': np.stack(list_mask, axis=0),
            'xy_min': np.stack(list_xy_min, axis=0),
            'xy_max': np.stack(list_xy_max, axis=0),
        }

        return select_inds, patch_info, np.array(patch_div_indices)

    def _get_patch_ray_indices(self, ray_mask, candidate_mask, patch_size, H, W):
        valid_ys, valid_xs = np.where(candidate_mask)
        select_idx = np.random.choice(valid_ys.shape[0], size=[1], replace=False)[0]
        center_x = valid_xs[select_idx]
        center_y = valid_ys[select_idx]

        half_patch_size = patch_size // 2
        x_min = np.clip(center_x - half_patch_size, 0, W - patch_size)
        x_max = x_min + patch_size
        y_min = np.clip(center_y - half_patch_size, 0, H - patch_size)
        y_max = y_min + patch_size

        sel_ray_mask = np.zeros_like(candidate_mask)
        sel_ray_mask[y_min:y_max, x_min:x_max] = True

        sel_ray_mask = sel_ray_mask.reshape(-1)
        inter_mask = np.bitwise_and(sel_ray_mask, ray_mask)
        select_masked_inds = np.where(inter_mask)

        masked_indices = np.cumsum(ray_mask) - 1
        select_inds = masked_indices[select_masked_inds]

        inter_mask = inter_mask.reshape(H, W)

        return (
            select_inds,
            inter_mask[y_min:y_max, x_min:x_max],
            np.array([x_min, y_min]),
            np.array([x_max, y_max]),
        )

    def load_image(self, frame_name, bg_color, use_mask=True):
        mi = self.mesh_infos[frame_name]
        imagepath = mi['rgb_path']
        orig_img = np.array(load_image(imagepath))

        if use_mask:
            maskpath = mi['mask_path']
            alpha_mask = np.array(load_image(maskpath))
        else:
            alpha_mask = np.ones_like(orig_img) * 255

        if alpha_mask.ndim == 2:
            alpha_mask = np.repeat(alpha_mask[..., None], 3, axis=-1)
        elif alpha_mask.ndim == 3 and alpha_mask.shape[2] == 1:
            alpha_mask = np.repeat(alpha_mask, 3, axis=-1)

        if frame_name in self.cameras and 'distortions' in self.cameras[frame_name]:
            K = self.cameras[frame_name]['intrinsics']
            D = self.cameras[frame_name]['distortions']
            orig_img = cv2.undistort(orig_img, K, D)
            alpha_mask = cv2.undistort(alpha_mask, K, D)

        img = alpha_mask / 255.0 * orig_img + (1.0 - alpha_mask / 255.0) * bg_color[None, None, :]

        if cfg.resize_img_scale != 1.:
            img = cv2.resize(
                img, None,
                fx=cfg.resize_img_scale,
                fy=cfg.resize_img_scale,
                interpolation=cv2.INTER_LANCZOS4,
            )
            alpha_mask = cv2.resize(
                alpha_mask,
                None,
                fx=cfg.resize_img_scale,
                fy=cfg.resize_img_scale,
                interpolation=cv2.INTER_LINEAR,
            )

        return img, alpha_mask

    def get_total_frames(self):
        return len(self.framelist)

    def sample_patch_rays(self, img, alpha, H, W, subject_mask, bbox_mask, ray_mask,
                          rays_o, rays_d, ray_img, ray_alpha, near, far):

        select_inds, patch_info, patch_div_indices = self.get_patch_ray_indices(
            N_patch=cfg.patch.N_patches,
            ray_mask=ray_mask,
            subject_mask=subject_mask,
            bbox_mask=bbox_mask,
            patch_size=cfg.patch.size,
            H=H,
            W=W,
        )

        rays_o, rays_d, ray_img, ray_alpha, near, far = self.select_rays(
            select_inds, rays_o, rays_d, ray_img, ray_alpha, near, far
        )

        targets = []
        targets_alpha = []
        for i in range(cfg.patch.N_patches):
            x_min, y_min = patch_info['xy_min'][i]
            x_max, y_max = patch_info['xy_max'][i]
            targets.append(img[y_min:y_max, x_min:x_max])
            targets_alpha.append(alpha[y_min:y_max, x_min:x_max])

        target_patches = np.stack(targets, axis=0)
        target_alpha_patches = np.stack(targets_alpha, axis=0)
        patch_masks = patch_info['mask']

        return (
            rays_o,
            rays_d,
            ray_img,
            ray_alpha,
            near,
            far,
            target_patches,
            target_alpha_patches,
            patch_masks,
            patch_div_indices,
        )

    def __len__(self):
        return self.get_total_frames()

    def __getitem__(self, idx):
        frame_name = self.framelist[idx]
        results = {'frame_name': frame_name}

        if self.bgcolor is None:
            bgcolor = (np.random.rand(3) * 255.).astype('float32')
        else:
            bgcolor = np.array(self.bgcolor, dtype='float32')

        img, alpha = self.load_image(frame_name, bgcolor, use_mask=(self.phase != 'infer'))

        bbox = self.bbox[frame_name]
        img, img2bb_trans, bb2img_trans, aug_param, do_flip, scale, alpha = augmentation(
            img,
            bbox,
            'eval',
            exclude_flip=True,
            input_img_shape=(256, 256),
            mask=alpha,
            base_scale=1.3,
            scale_factor=0.2,
            rot_factor=0,
            shift_wh=[bbox[2], bbox[3]],
            gaussian_std=3,
            bordervalue=bgcolor.tolist(),
        )

        img = (img / 255.).astype('float32')
        alpha = alpha.astype('float32')
        results['target_alpha_img'] = alpha[..., 0]

        H, W = img.shape[0:2]

        dst_skel_info = self.query_dst_skeleton(frame_name)
        dst_bbox = dst_skel_info['bbox']
        dst_poses = dst_skel_info['poses']
        dst_shape = dst_skel_info['shape']
        dst_tpose_joints = dst_skel_info['dst_tpose_joints']
        dst_cam_joints = dst_skel_info['joint_cam']
        dst_valid_joints = dst_skel_info['joint_valid']

        K = self.cameras[frame_name]['intrinsics'][:3, :3].copy()
        K[:2, 2] = trans_point2d(K[:2, 2], img2bb_trans)
        K[[0, 1], [0, 1]] = K[[0, 1], [0, 1]] * 256 / (bbox[2] * aug_param[1])

        E = self.cameras[frame_name]['extrinsics']
        E = apply_global_tfm_to_camera(
            E=E,
            Rh=dst_skel_info['Rh'],
            Th=dst_skel_info['Th'],
        )
        R = E[:3, :3]
        T = E[:3, 3]

        results.update({
            'cam_K': K,
            'cam_R': R,
            'cam_T': T,
        })

        rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T)
        ray_img = img.reshape(-1, 3)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        ray_alpha = alpha.reshape(-1, 3)

        near, far, ray_mask = rays_intersect_3d_bbox(dst_bbox, rays_o, rays_d)

        rays_o = rays_o[ray_mask]
        rays_d = rays_d[ray_mask]
        ray_img = ray_img[ray_mask]
        ray_alpha = ray_alpha[ray_mask]

        near = near[:, None].astype('float32')
        far = far[:, None].astype('float32')

        if self.ray_shoot_mode == 'image':
            pass
        elif self.ray_shoot_mode == 'patch':
            rays_o, rays_d, ray_img, ray_alpha, near, far, \
            target_patches, target_alpha_patches, patch_masks, patch_div_indices = self.sample_patch_rays(
                img=img,
                alpha=alpha,
                H=H,
                W=W,
                subject_mask=alpha[:, :, 0] > 0.,
                bbox_mask=ray_mask.reshape(H, W),
                ray_mask=ray_mask,
                rays_o=rays_o,
                rays_d=rays_d,
                ray_img=ray_img,
                ray_alpha=ray_alpha,
                near=near,
                far=far,
            )
        else:
            assert False, f"Invalid Ray Shoot Mode: {self.ray_shoot_mode}"

        batch_rays = np.stack([rays_o, rays_d], axis=0)

        if 'rays' in self.keyfilter:
            results.update({
                'img_width': W,
                'img_height': H,
                'ray_mask': ray_mask,
                'rays': batch_rays,
                'near': near,
                'far': far,
                'bgcolor': bgcolor,
            })

            if self.ray_shoot_mode == 'patch':
                results.update({
                    'patch_div_indices': patch_div_indices,
                    'patch_masks': patch_masks,
                    'target_patches': target_patches,
                    'target_alpha_patches': target_alpha_patches[..., 0],
                })

        if 'target_rgbs' in self.keyfilter:
            results['target_rgbs'] = ray_img
            results['target_alpha'] = ray_alpha[..., 0]
            results['target_img'] = img

        if 'motion_bases' in self.keyfilter:
            dst_Rs, dst_Ts = body_pose_to_body_RTs(dst_poses, dst_tpose_joints)
            results.update({
                'dst_Rs': dst_Rs,
                'dst_Ts': dst_Ts,
            })

        if 'dst_posevec_69' in self.keyfilter:
            dst_posevec_69 = dst_poses[3:] + 1e-2
            results.update({
                'dst_posevec': dst_posevec_69,
                'dst_shape': dst_shape,
                'dst_global_orient': dst_poses[:3],
                'dst_cam_joints': dst_cam_joints,
                'dst_valid_joints': dst_valid_joints[:, 0],
            })

     
        return results