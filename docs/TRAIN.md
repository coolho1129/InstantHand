# Training

[← README](../README.md) · [Installation](INSTALL.md) · [Demo](DEMO.md) · [Evaluation](EVAL.md) · Training

Needs MANO (1), the DINOv3 weights (2) and the training data (5); see
[Files to download](INSTALL.md#files-to-download). Run every command from the repository root.

Training takes two stages: stage 1 learns the avatar predictor (geometry and base colors), stage 2
trains the appearance head on top of the frozen stage-1 model. The released models were trained on
4 RTX A6000 GPUs (about 25 h and 13 h).

```bash
# stage 1
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 train_stage1.py --cfg configs/train_stage1.yaml
# stage 2 (starts from the released stage-1 weights by default, see stage1_ckpt_dir)
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 train_stage2.py --cfg configs/train_stage2.yaml
```

All settings come from the config files; the values in them reproduce the released training
(batch size 6 × 2 accumulation steps for stage 1, 4 for stage 2, per GPU; 10 epochs of 2,000 steps).
Change any entry from the command line with `--set` (values are parsed as YAML):

```bash
accelerate launch --num_processes 4 train_stage1.py --set optim.batch_size=4 --set data.root=/path/to/train
accelerate launch --num_processes 4 train_stage2.py --set stage1_ckpt_dir=./exps/stage1/step_0020000
accelerate launch --num_processes 4 train_stage1.py --resume_dir ./exps/stage1/step_0010000   # resume
```

Checkpoints are written to `output.save_dir` (`exps/stage1/step_XXXXXXX/`, `exps/stage2/step_XXXXXXX/`)
every `output.ckpt_every` optimizer steps, with TensorBoard logs in `output.tb_dir`. Evaluate a
stage-2 checkpoint with `--ckpt_dir exps/stage2/step_XXXXXXX` (see [Evaluation](EVAL.md)).

| Config | Content |
|---|---|
| `configs/train_stage1.yaml`, `configs/train_stage2.yaml` | model, losses, optimizer, schedule, outputs; `data_config` points to the data config |
| `configs/train_data.yaml` | data root, every dataset setting (crops, cameras, masks, file layout, WiLoR keys) and the sampling policies of both stages (`policies.stage1`, `policies.stage2`) |

The data config is checked strictly: a missing or misspelled setting, or a dataset without a policy,
stops the training instead of falling back to a default.

## Training data

The training datasets are not redistributed; obtain each from its official source under its
license. We train on PALM, InterHand2.6M (official training split only), HO3Dv3, Goliath and Hand
Appearance (HARP), 304 subjects and 339,532 frames in total, and add synthetic source–target pairs
generated following Zhao et al. (CVPR 2025) in stage 2. Each dataset is converted to this layout:

```
data/train/
└── <dataset>/                        PALM, InterHand2.6M, HO3D_v3, Goliath, HARP, Synthetic
    └── <subject>/                    one hand of one person (e.g. 0001_R, 0001_L)
        ├── meta/                     optional, ignored
        └── <group>/                  one observation bucket: a camera, a sequence, a session, ...
            ├── seg_rgba/<frame>.png      RGBA image, alpha = hand mask (also looked up in seg_rgba/sam3/)
            ├── wilor_out/<frame>.json    WiLoR estimate of the hand in this frame
            └── silhouette/<frame>.png    optional hand silhouette for the silhouette and SAB losses
                                          (the alpha of seg_rgba is used when it is missing)
```

- **Dataset folder names** must match the keys of `policies.stage1` / `policies.stage2` in
  `configs/train_data.yaml`; only the datasets listed there are used (stage 1 has no `Synthetic`).
  The policy of each dataset sets its sampling weight, the stride of the source frames and the
  6 target frames per sample as self / same group / other groups (Appendix, Table S1).
- **Left and right hands.** A subject is one hand: put the left and the right hand of a person in
  separate subject folders (any names; we use `<id>_L` and `<id>_R`). The folder name is not
  parsed. Whether a frame is mirrored is decided per frame by `hand_label` in its WiLoR json:
  frames labeled `left` are flipped horizontally (image, mask, silhouette, 2D keypoints and the
  x of the camera translation) and handled by the right-hand model, as at inference. All frames of a
  sample come from one subject and share the source betas, so the two hands of a person must not
  be mixed in one subject folder.
- **Frames** are the frame ids present in both `seg_rgba/` and `wilor_out/`; frame ids that are
  integers are ordered numerically. A group without any such frame is skipped.
- **WiLoR json** (one hand per file):

  | Key | Content |
  |---|---|
  | `pred_mano_params` | `global_orient` and `hand_pose` as rotation matrices, `betas` (10) |
  | `pred_keypoints_2d` | 21 2D keypoints in full-image pixels (used for the square crop) |
  | `pred_cam_t` | camera translation |
  | `focal_length` | focal length in pixels (the principal point is the image center) |
  | `hand_label` | `left` or `right`; left hands are mirrored to right hands |

  The key names can be changed in `configs/train_data.yaml` (`json_*_key`).
- Where a dataset provides no masks or MANO/camera annotations, estimate them with
  [SAM 3](https://github.com/facebookresearch/sam3) and [WiLoR](https://github.com/rolpotamias/WiLoR)
  (see [Your own images](DEMO.md#your-own-images)).

Each sample picks a dataset by its weight, a subject uniformly, and a source frame, then crops every
frame to a square around its 2D keypoints (enlarged 1.3×), resizes it to 256 × 256 and composites it
on a random gray background.
