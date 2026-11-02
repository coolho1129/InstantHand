# Evaluation

[← README](../README.md) · [Installation](INSTALL.md) · [Demo](DEMO.md) · Evaluation · [Training](TRAIN.md)

Needs MANO (1), a checkpoint (3) and the evaluation data (4); see
[Files to download](INSTALL.md#files-to-download). Run every command from the repository root.

We follow the single-image protocol of HandAvatar and OHTA: InterHand2.6M
`test/Capture0/ROM03_RT_No_Occlusion` with the source image of the official OHTA configuration,
and HanCo sequence `0191` (cameras 3, 5, 6, 7) with the 76th frame of camera 6 as the source,
evaluated zero-shot.

## Data preparation

The evaluation reads only the files below, all under `data/eval/`. Our annotations (and, for
InterHand2.6M, the hand masks and the source frame) are provided as two small archives; the images
come from the official datasets.

```
data/eval/
├── InterHand/5/InterHand2.6M_5fps_batch1/
│   ├── images/test/Capture0/ROM03_RT_No_Occlusion/cam*/image*.jpg                 official images
│   ├── masks_removeblack/test/Capture0/ROM03_RT_No_Occlusion/cam*/image*.png       our archive
│   ├── preprocess/test/Capture0/ROM03_RT_No_Occlusion/anno_cam_ffgs.pkl            our archive
│   └── anchor/                                                                     our archive
│       ├── img/test_Capture0_ROM03_RT_No_Occlusion_cam400272_image15012.jpg
│       ├── mask/test_Capture0_ROM03_RT_No_Occlusion_cam400272_image15012.jpg
│       └── anno/test_Capture0_ROM03_RT_No_Occlusion_cam400272_image15012_scale1.pkl
└── HanCo/0191/
    ├── rgb/cam{3,5,6,7}/000000{00..75}.jpg                                         official images
    ├── mask/cam{3,5,6,7}/000000{00..75}.jpg                                        official masks
    ├── pkl/anno_cam_hanco_scale1.pkl                                               our archive
    └── anchor/
        ├── rgb/cam6_00000075.jpg                                                   copied from rgb/ (below)
        ├── mask/cam6_00000075.png                                                  converted from the mask (below)
        └── anno/0191_cam6_00000075_scale1.pkl                                      our archive
```

The HanCo annotation file stores the image and mask paths of every evaluation frame relative to the
repository root (`data/eval/HanCo/0191/...`), so the images must be at exactly these paths.

### InterHand2.6M

1. Download the InterHand2.6M images (5 fps, batch 1) from the
   [official site](https://mks0601.github.io/InterHand2.6M/) and place
   `images/test/Capture0/ROM03_RT_No_Occlusion/` under
   `data/eval/InterHand/5/InterHand2.6M_5fps_batch1/` (a symlink is fine).
2. Download `InstantHand_eval_InterHand2.6M.zip` (our hand masks and annotations, and the source
   frame) from the [v1.0 release](https://github.com/coolho1129/InstantHand/releases/tag/v1.0) and
   unzip it in the repository root (it extracts to `data/eval/InterHand/...`):
   ```bash
   wget https://github.com/coolho1129/InstantHand/releases/download/v1.0/InstantHand_eval_InterHand2.6M.zip
   unzip InstantHand_eval_InterHand2.6M.zip
   ```

### HanCo

1. Download the RGB images and hand masks (`HanCo_rgb.zip`, `HanCo_mask_hand.zip`) from the
   [official site](https://lmb.informatik.uni-freiburg.de/resources/datasets/HanCo.en.html) and
   copy sequence `0191` into the layout above:
   ```bash
   H=/path/to/HanCo          # unzipped HanCo_rgb.zip and HanCo_mask_hand.zip
   D=data/eval/HanCo/0191
   for c in 3 5 6 7; do
     mkdir -p $D/rgb/cam$c $D/mask/cam$c
     cp $H/rgb/0191/cam$c/000000{00..75}.jpg $D/rgb/cam$c/
     cp $H/mask_hand/0191/cam$c/000000{00..75}.jpg $D/mask/cam$c/
   done
   mkdir -p $D/anchor/rgb $D/anchor/mask
   cp $D/rgb/cam6/00000075.jpg $D/anchor/rgb/cam6_00000075.jpg
   python -c "from PIL import Image; Image.open('$H/mask_hand/0191/cam6/00000075.jpg').convert('L').save('$D/anchor/mask/cam6_00000075.png')"
   ```
2. Download `InstantHand_eval_HanCo.zip` (our annotations for sequence 0191) from the
   [v1.0 release](https://github.com/coolho1129/InstantHand/releases/tag/v1.0) and unzip it in the
   repository root (it extracts to `data/eval/HanCo/0191/...`):
   ```bash
   wget https://github.com/coolho1129/InstantHand/releases/download/v1.0/InstantHand_eval_HanCo.zip
   unzip InstantHand_eval_HanCo.zip
   ```

HanCo is provided for research purposes only; if you use it, please cite FreiHAND
(Zimmermann et al., ICCV 2019) and HanCo (Zimmermann et al., 2021).

## Running the evaluation

```bash
python eval_interhand.py --cfg configs/eval_interhand.yaml
python eval_hanco.py --cfg configs/eval_hanco.yaml
python eval_efficient.py                                       # avatar-construction time, FPS, memory
```

Each evaluation script builds the avatar once from the source frame and reports, in this order:

| Section | `eval_interhand.py` | `eval_hanco.py` | Paper |
|---|---|---|---|
| main (PSNR / SSIM / LPIPS) | ✓ | ✓ | Tables 1, 2 |
| boundary / alpha (boundary F-score, boundary IoU, alpha leakage, interior coverage) | ✓ | | Table 4 |
| pose strata (palm-orientation and finger-articulation terciles, and All) | ✓ | ✓ | Table S12 |
| temporal stability (excess frame-to-frame variation per camera, flicker calibration) | | ✓ | Table S14 |
| self-reconstruction (`--self_recon`, a second pass; every test frame is its own source) | ✓ | ✓ | Table S2 |

Expected results:

| Benchmark | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| InterHand2.6M | 26.21 | 0.866 | 0.133 |
| HanCo (zero-shot) | 20.18 | 0.836 | 0.188 |

`eval_interhand.py` also reports boundary F-score 0.698, boundary IoU 0.562, alpha leakage 0.234
and interior coverage 0.999. With `--self_recon`, the self-reconstruction rows are
`w/o transfer` (the same checkpoint with the appearance head disabled) 26.58 / 0.869 / 0.133 and
`Ours` 28.54 / 0.903 / 0.108 on InterHand2.6M, and 20.47 / 0.839 / 0.187 and 23.88 / 0.898 / 0.153
on HanCo. Self-reconstruction builds two avatars per test frame and takes several minutes
(about 4 min for InterHand2.6M and 7 min for HanCo on one RTX 6000 Ada).

Options of both scripts:

| Option | Effect |
|---|---|
| `--ckpt_dir <dir>` | evaluate another checkpoint (directory with `model.safetensors`) |
| `--max_batches N` | first N test frames only (quick check) |
| `--self_recon` / `--no_self_recon` | turn the self-reconstruction pass on / off (off by default; `self_recon.enabled` in the config) |
| `--no_pose_strata` | skip the pose strata |
| `--no_boundary` | skip the boundary metrics (`eval_interhand.py`) |
| `--no_temporal` | skip the temporal stability (`eval_hanco.py`) |
| `--verbose` | unrounded means, frame counts and median viewing-direction change per pose tercile, and setup logs |

The metrics are averaged over frames and rounded as in the paper. Results on other GPUs can differ
in the second decimal place.

## Efficiency

`eval_efficient.py` measures avatar construction (canonical Gaussians only, with the
appearance head disabled, and the full model), novel-pose rendering speed and peak GPU memory on the
HanCo frames. The numbers of the paper (Table 2 and Table S3: 2.84 s, of which 1.38 s for the canonical
Gaussians; 161 FPS; 1.79 GB while rendering) were measured on one A100 (80 GB); other GPUs give different absolute
numbers.
