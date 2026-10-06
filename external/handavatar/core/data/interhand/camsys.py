# interhand/camsys.py
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
import pickle

import numpy as np
import cv2
import torch.utils.data

from external.handavatar.core.utils.image_util import load_image
from external.handavatar.core.utils.hand_util import \
    body_pose_to_body_RTs, \
    get_canonical_global_tfms, \
    approx_gaussian_bone_volumes
from external.handavatar.core.utils.camera_util import \
    apply_global_tfm_to_camera, \
    get_rays_from_KRT, \
    rays_intersect_3d_bbox

from external.handavatar.core.utils.hand_util import MANOHand, INTERHAND2MANO

import json
from pycocotools.coco import COCO
from external.handavatar.core.utils.augm_util import augmentation, trans_point2d
from external.handavatar.configs import cfg
import smplx


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
        if self.handtype=='left':
            self.mano.shapedirs[:,0,:] *= -1
        
        # annotation
        self.phase = kwargs.get('data_type', 'train')
        if subject is None:
            if self.phase == 'train':
                subject = cfg.subject
            else:
                subject = cfg[kwargs['data_type']].subject
        if isinstance(subject, list):
            subject = subject[0]
        self.image_dir = os.path.join(dataset_path, f'InterHand2.6M_{cfg.interhand.fps}fps_batch1/images')
        anno_name = os.path.join(self.image_dir.replace('images', 'preprocess'), subject, 'anno_cam.pkl')
        print('Load annotation', anno_name)
        with open(anno_name, 'rb') as f:
            self.cameras, self.mesh_infos, self.bbox, self.framelist = pickle.load(f)

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


    @staticmethod
    def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - cfg.bbox_offset
        max_xyz = np.max(skeleton, axis=0) + cfg.bbox_offset

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }


    def query_dst_skeleton(self, frame_name):
        return {
            'poses': self.mesh_infos[frame_name]['poses'].astype('float32'),
            'shape': self.mesh_infos[frame_name]['shape'].astype('float32'),
            'dst_tpose_joints': \
                self.mesh_infos[frame_name]['tpose_joints'].astype('float32'),
            'bbox': self.mesh_infos[frame_name]['bbox'].copy(),
            'Rh': self.mesh_infos[frame_name]['Rh'].astype('float32'),
            'Th': self.mesh_infos[frame_name]['Th'].astype('float32'),
        }

    @staticmethod
    def select_rays(select_inds, rays_o, rays_d, ray_img, ray_alpha, near, far):
        rays_o = rays_o[select_inds]
        rays_d = rays_d[select_inds]
        ray_img = ray_img[select_inds]
        ray_alpha = ray_alpha[select_inds]
        near = near[select_inds]
        far = far[select_inds]
        return rays_o, rays_d, ray_img, ray_alpha, near, far
    
    
    def load_image(self, frame_name, bg_color, use_mask=True):
        imagepath = os.path.join(self.image_dir, frame_name)
        orig_img = np.array(load_image(imagepath))

        if use_mask:
            maskpath = imagepath.replace('images', 'masks_removeblack').replace('.jpg', '.png')
            alpha_mask = np.array(load_image(maskpath))
        else:
            alpha_mask = np.ones_like(orig_img) * 255
        
        # undistort image
        if frame_name in self.cameras and 'distortions' in self.cameras[frame_name]:
            K = self.cameras[frame_name]['intrinsics']
            D = self.cameras[frame_name]['distortions']
            orig_img = cv2.undistort(orig_img, K, D)
            alpha_mask = cv2.undistort(alpha_mask, K, D)

        img = alpha_mask / 255. * orig_img + (1.0 - alpha_mask / 255.) * bg_color[None, None, :]
        if cfg.resize_img_scale != 1.:
            img = cv2.resize(img, None, 
                                fx=cfg.resize_img_scale,
                                fy=cfg.resize_img_scale,
                                interpolation=cv2.INTER_LANCZOS4)
            alpha_mask = cv2.resize(alpha_mask, None,
                                    fx=cfg.resize_img_scale,
                                    fy=cfg.resize_img_scale,
                                    interpolation=cv2.INTER_LINEAR)
                                
        return img, alpha_mask


    def get_total_frames(self):
        return len(self.framelist)

    def __len__(self):
        return self.get_total_frames()

    def __getitem__(self, idx):
        frame_name = self.framelist[idx]
        results = {
            'frame_name': frame_name
        }

        if self.bgcolor is None:
            bgcolor = (np.random.rand(3) * 255.).astype('float32')
        else:
            bgcolor = np.array(self.bgcolor, dtype='float32')

        img, alpha = self.load_image(frame_name, bgcolor, use_mask=(True, False)[self.phase=='infer'])
        bbox = self.bbox[frame_name]
        img, img2bb_trans, bb2img_trans, aug_param, do_flip, scale, alpha = augmentation(img, self.bbox[frame_name], 'eval',
                                                                                     exclude_flip=True,
                                                                                     input_img_shape=(256, 256), mask=alpha,
                                                                                     base_scale=1.3,
                                                                                     scale_factor=0.2,
                                                                                     rot_factor=0,
                                                                                     shift_wh=[bbox[2], bbox[3]],
                                                                                     gaussian_std=3,
                                                                                     bordervalue=bgcolor.tolist())
        



        # cv2.imwrite('test.png', img.astype('uint8'))
        # cv2.waitKey(0)

        img = (img / 255.).astype('float32')
        alpha = alpha.astype('float32')
        results['target_alpha_img'] = alpha[..., 0]

        H, W = img.shape[0:2]

        dst_skel_info = self.query_dst_skeleton(frame_name)
        dst_bbox = dst_skel_info['bbox']
        dst_poses = dst_skel_info['poses']
        dst_shape = dst_skel_info['shape']
        dst_tpose_joints = dst_skel_info['dst_tpose_joints']
        dst_Th = dst_skel_info['Th']
        dst_bbox['min_xyz'] += dst_Th
        dst_bbox['max_xyz'] += dst_Th

        assert frame_name in self.cameras
        K = self.cameras[frame_name]['intrinsics'][:3, :3].copy()
        K[:2, 2] = trans_point2d(K[:2, 2], img2bb_trans)
        K[[0, 1], [0, 1]] = K[[0, 1], [0, 1]] * 256 / (bbox[2]*aug_param[1])

        E = self.cameras[frame_name]['extrinsics']

        E = apply_global_tfm_to_camera(
                E=E, 
                Rh=dst_skel_info['Rh'],
                Th=np.zeros(3))
        R = E[:3, :3]
        T = E[:3, 3]
        results.update({
            'cam_K': K,
            'cam_R': R,
            'cam_T': T,
        })
        rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T)
        ray_img = img.reshape(-1, 3) 
        rays_o = rays_o.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
        rays_d = rays_d.reshape(-1, 3)
        ray_alpha = alpha.reshape(-1, 3) 

        # (selected N_samples, ), (selected N_samples, ), (N_samples, )
        near, far, ray_mask = rays_intersect_3d_bbox(dst_bbox, rays_o, rays_d)
        rays_o = rays_o[ray_mask]
        rays_d = rays_d[ray_mask]
        ray_img = ray_img[ray_mask]
        ray_alpha = ray_alpha[ray_mask]

        near = near[:, None].astype('float32')
        far = far[:, None].astype('float32')
    
        batch_rays = np.stack([rays_o, rays_d], axis=0) 

        if 'rays' in self.keyfilter:
            results.update({
                'img_width': W,
                'img_height': H,
                'ray_mask': ray_mask,
                'rays': batch_rays,
                'near': near,
                'far': far,
                'bgcolor': bgcolor})

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
            dst_Rs, dst_Ts = body_pose_to_body_RTs(
                dst_poses, dst_tpose_joints)
            results.update({
                'dst_Rs': dst_Rs,
                'dst_Ts': dst_Ts,
            })

        if 'dst_posevec_69' in self.keyfilter:
            # 1. ignore global orientation
            # 2. add a small value to avoid all zeros
            dst_posevec_69 = dst_poses[3:] + 1e-2
            results.update({
                'dst_posevec': dst_posevec_69,
                'dst_shape': dst_shape,
                'dst_global_orient': dst_poses[:3],
                'dst_Th': dst_Th,
            })

        return results
