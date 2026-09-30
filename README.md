# InstantHand: Feed-Forward 3D Gaussian Splatting for Animatable Hand Avatar Reconstruction from a Single Image

**NeurIPS 2026**

[Chanho Kim](https://coolho1129.github.io),
[Muhammad Salman Ali](https://salmanali96.github.io/),
[Kyeonghwan Gwak](https://khgwak.github.io/about/),
[Muhammad Shaheryar](https://scholar.google.com/citations?user=v9FnXzwAAAAJ&hl=en),
Incheol Park,
[Jeongwan On](https://on-jungwoan.github.io/),
[Seungryul Baek](https://sites.google.com/site/bsrvision00/)

[UNIST Vision & Learning Lab](https://vision.unist.ac.kr/), UNIST

<!-- Replace "(coming soon)" with a link to the first public version of the paper (OpenReview, arXiv, or proceedings) -->
**[Project Page](https://coolho1129.github.io/InstantHand-page/) | Paper (coming soon) | [Checkpoints & Data](https://drive.google.com/drive/folders/10iqqssndhqTGFWGWVBmpnCxzchryCdUE)**

<p align="center">
  <img src="assets/teaser.webp" width="100%" alt="A single input hand image, one forward pass, and the reconstructed canonical 3D Gaussian hand avatar rotating.">
</p>

From a single hand image, InstantHand builds an animatable 3D Gaussian hand avatar in 2.84 s, with no per-subject optimization.

## Contents

- [Installation](#installation)
- [Demo](#demo)
- [Evaluation](#evaluation)
- [Optional inference-time settings](#optional-inference-time-settings)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Installation

Tested on Ubuntu with Python 3.10, PyTorch 2.3.0 and CUDA 11.8 (RTX 3090, RTX A6000, A100).

```bash
git clone --recursive https://github.com/coolho1129/InstantHand.git
cd InstantHand
# if you cloned without --recursive:
git submodule update --init --recursive

conda create -n instanthand python=3.10 -y
conda activate instanthand
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA extensions; set the compute capability of your GPU
# (8.6: RTX 3090 / A6000, 8.0: A100, 8.9: RTX 4090; +PTX also runs on newer GPUs)
export TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
pip install --no-build-isolation ./diff-gaussian-rasterization
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba"

pip install --no-build-isolation chumpy==0.70
pip install -r requirements.txt
```

`--no-build-isolation` lets the CUDA extensions build against the installed PyTorch. The
nvcc version on your system should match the CUDA version of PyTorch (11.8 above).

`diff-gaussian-rasterization/` is a modified version of the Inria 3D Gaussian Splatting
rasterizer that additionally returns the rendering contribution of every Gaussian, which
Reliability-gated Appearance Transfer uses. The original rasterizer cannot be used in its place.

### Files to download

Everything else is in this repository. The files below come from other sources because of their
licenses or size; place them at these paths. The demo needs (1) and (3), plus (2) if you use the light
checkpoint; the evaluation also needs (4).

```
InstantHand/
├── instanthand/models/rendering/mano_model/mano/
│   ├── MANO_RIGHT.pkl                                  ← (1) MANO
│   └── mano_mean_params.npz                              included (from WiLoR)
├── instanthand/models/encoders/dinov3/weights/
│   └── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth    ← (2) DINOv3 (light checkpoint or training only)
├── exps/neurips2026/stage2_best/
│   └── model.safetensors                               ← (3) InstantHand checkpoint
├── data/                                               ← (4) evaluation data, see Evaluation
│   ├── InterHand/5/InterHand2.6M_5fps_batch1/
│   └── HanCo/0191/
├── diff-gaussian-rasterization/third_party/glm/          git submodule (clone with --recursive)
└── demo/input/, demo/pose_bank/                          included
```

| | Place at | Source |
|---|---|---|
| (1) | `instanthand/models/rendering/mano_model/mano/MANO_RIGHT.pkl` | Register at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de), download `mano_v1_2.zip` and copy `mano_v1_2/models/MANO_RIGHT.pkl`. `MANO_LEFT.pkl` is not needed: left hands are mirrored and handled by the right-hand model. |
| (2) | `instanthand/models/encoders/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | Needed only with the light checkpoint or for training. Meta's DINOv3 download (access request), see [DINOv3 weights](#dinov3-weights) below. |
| (3) | `exps/neurips2026/stage2_best/model.safetensors` | One of the two checkpoints below, saved under this name. Another location works if you change `ckpt_dir` in `configs/demo.yaml`, `configs/eval_interhand.yaml` and `configs/eval_hanco.yaml`. |
| (4) | `data/` | Official InterHand2.6M and HanCo images plus our archives, see [Data preparation](#data-preparation). |

**Checkpoints.** Both are in our [Google Drive folder](https://drive.google.com/drive/folders/10iqqssndhqTGFWGWVBmpnCxzchryCdUE) and give identical results; the full one
includes the frozen DINOv3 ViT-L/16 backbone, so you do not need (2). Save the one you use as
`exps/neurips2026/stage2_best/model.safetensors`:

```bash
mkdir -p exps/neurips2026/stage2_best
mv ~/Downloads/instanthand_full.safetensors exps/neurips2026/stage2_best/model.safetensors
```

| Checkpoint | Size | DINOv3 file (2) | Download |
|---|---|---|---|
| full | 1.4 GB | not needed | [`checkpoints/instanthand_full.safetensors`](https://drive.google.com/drive/folders/10iqqssndhqTGFWGWVBmpnCxzchryCdUE) |
| light | 0.2 GB | needed | [`checkpoints/instanthand_light.safetensors`](https://drive.google.com/drive/folders/10iqqssndhqTGFWGWVBmpnCxzchryCdUE) |

The DINOv3 backbone in the full checkpoint is distributed under the DINOv3 License (a copy is
included with the download). The MANO model is not redistributed here (MANO license);
`mano_mean_params.npz` is included as released by [WiLoR](https://github.com/rolpotamias/WiLoR).

#### DINOv3 weights

Only needed with the light checkpoint or for training. The DINOv3 weights are distributed by Meta
after an access request; approval can take a while, so request it first.

1. Request access on the [DINOv3 download page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
   (linked from the [official repository](https://github.com/facebookresearch/dinov3)). Once
   approved, Meta e-mails a list of download URLs for all DINOv3 models.
2. From that list, download the ViT-L/16 backbone pretrained on LVD-1689M and save it as
   `instanthand/models/encoders/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`
   (the path set by `encoder_weights_path` in the configs; keep the file name):
   ```bash
   mkdir -p instanthand/models/encoders/dinov3/weights
   wget -O instanthand/models/encoders/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
        "<URL of dinov3_vitl16_pretrain_lvd1689m from the e-mail>"
   ls -lh instanthand/models/encoders/dinov3/weights/
   ```

The `facebook/dinov3-vitl16-pretrain-lvd1689m` repository on Hugging Face holds the same model in
the Transformers format, which is not the file this code loads.

## Demo

Each input is a hand image with its hand mask (or an RGBA image whose alpha channel is the
mask) and the [WiLoR](https://github.com/rolpotamias/WiLoR) estimate of the hand (MANO pose and
shape, 2D keypoints, camera) saved as a json file. The examples in `demo/input/`
(`rgba/<id>.png` and `json/<id>.json`) show the format.

```bash
python demo.py --ids all                      # every example in demo/input
python demo.py --ids in_the_wild3             # one example
python demo.py --image hand.png --mask hand_mask.png --wilor_json hand.json
```

Outputs are written to `demo/output/<name>/`:

| File | Content |
|---|---|
| `input_rgba.png` | the composed input |
| `source.png` | the network input (keypoint-centered crop) |
| `self_render.png` | the avatar rendered from the input camera |
| `canonical_360_yaw.mp4`, `canonical_360_pitch.mp4` | turntables of the avatar in the rest pose |
| `pose_anim_view_*.mp4` | the avatar driven through the poses in `demo/pose_bank/` |
| `camera.json` | the render cameras |

The cameras and video settings in `configs/demo.yaml` reproduce the supplementary videos
(fixed 30° field of view, 256 px). Any entry can be changed from the command line, e.g.
`--set "videos.kinds=[canonical]"` for the turntables only or `--set videos.res=512` for larger
videos. `in_the_wild3` reproduces the in-the-wild vitiligo result on the project page.

### Examples

| id | Source | What it shows |
|---|---|---|
| `interhand_test_Capture0_ROM03_cam400272_15012` | InterHand2.6M test subject (unseen) | studio capture |
| `interhand_..._15012_tattoo_tribal` | same, with a tattoo drawn by the authors | black pattern |
| `interhand_..._15012_tattoo_butterfly` | same, with a tattoo drawn by the authors | colored pattern |
| `in_the_wild1` | OHTA in-the-wild image | palm-side input |
| `in_the_wild1_tattoo_spiral` | same, with a tattoo drawn by the authors | pattern on the palm |
| `in_the_wild2` | WiLoR demo image | darker skin tone |
| `in_the_wild3` | OHTA in-the-wild image | vitiligo |
| `in_the_wild3_edit_darkskin` | same, skin tone edited by the authors | skin-tone edit |

Example inputs are taken from InterHand2.6M (CC BY-NC 4.0), OHTA, and WiLoR; the tattoo and
skin-tone examples are edited versions made by the authors. The driving poses in
`demo/pose_bank/` are selected from the InterHand2.6M annotations.

### Your own images

1. **Hand mask.** Segment the hand with [SAM 3](https://github.com/facebookresearch/sam3)
   (e.g. with the text prompt `hand`), keeping only the hand you want to reconstruct.
2. **RGBA input (recommended).** Store the mask as the alpha channel, as in the examples:
   ```bash
   python -c "from PIL import Image; im = Image.open('my_hand.jpg').convert('RGBA'); \
   im.putalpha(Image.open('my_hand_mask.png').convert('L')); im.save('demo/input/rgba/my_hand.png')"
   ```
   `demo.py` also accepts the image and the mask separately (`--image`, `--mask`).
3. **WiLoR estimate.** In an environment with [WiLoR](https://github.com/rolpotamias/WiLoR) and its
   checkpoints installed, run on the original image:
   ```bash
   python tools/gen_wilor_json.py --wilor_root /path/to/WiLoR --images my_hand.jpg --out_dir demo/input/json
   ```
   This writes `demo/input/json/my_hand.json` and a mesh overlay in `demo/input/json/vis/` to check
   the fit. `--wilor_root` is your WiLoR clone, with its checkpoints in `pretrained_models/` and the
   MANO files in `mano_data/` as in the WiLoR instructions. `--images` also takes folders; with
   several hands in an image, `--detection_rank 1` picks the second most confident one, and
   `--no_overlay` skips the rendering (e.g. on a headless server). The hand is estimated as a right
   hand; for a left hand, mirror the image horizontally first (and use the mirrored image in step 2).
4. **Run.** `python demo.py --ids my_hand`

## Evaluation

We follow the single-image protocol of HandAvatar and OHTA: InterHand2.6M
`test/Capture0/ROM03_RT_No_Occlusion` with the source image of the official OHTA configuration,
and HanCo sequence `0191` (cameras 3, 5, 6, 7) with the 76th frame of camera 6 as the source,
evaluated zero-shot.

### Data preparation

The evaluation reads only the files below. Our annotations (and, for InterHand2.6M, the hand masks
and the source frame) are provided as two small archives; the images come from the official datasets.

```
data/
├── InterHand/5/InterHand2.6M_5fps_batch1/
│   ├── images/test/Capture0/ROM03_RT_No_Occlusion/cam*/image*.jpg
│   ├── masks_removeblack/test/Capture0/ROM03_RT_No_Occlusion/cam*/image*.png
│   ├── preprocess/test/Capture0/ROM03_RT_No_Occlusion/anno_cam_ffgs.pkl
│   └── anchor/
│       ├── img/test_Capture0_ROM03_RT_No_Occlusion_cam400272_image15012.jpg
│       ├── mask/test_Capture0_ROM03_RT_No_Occlusion_cam400272_image15012.jpg
│       └── anno/test_Capture0_ROM03_RT_No_Occlusion_cam400272_image15012_scale1.pkl
└── HanCo/0191/
    ├── rgb/cam{3,5,6,7}/000000{00..75}.jpg
    ├── mask/cam{3,5,6,7}/000000{00..75}.jpg
    ├── pkl/anno_cam_hanco_scale1.pkl
    └── anchor/
        ├── rgb/cam6_00000075.jpg
        ├── mask/cam6_00000075.png
        └── anno/0191_cam6_00000075_scale1.pkl
```

**InterHand2.6M.**
1. Download the InterHand2.6M images (5 fps, batch 1) from the
   [official site](https://mks0601.github.io/InterHand2.6M/) and place
   `images/test/Capture0/ROM03_RT_No_Occlusion/` under
   `data/InterHand/5/InterHand2.6M_5fps_batch1/` (a symlink is fine).
2. Download `eval_data/InstantHand_eval_InterHand2.6M.zip` (our hand masks and annotations, and the
   source frame) from our [Google Drive folder](https://drive.google.com/drive/folders/10iqqssndhqTGFWGWVBmpnCxzchryCdUE) and unzip it in the repository root.

**HanCo.**
1. Download the RGB images and hand masks (`HanCo_rgb.zip`, `HanCo_mask_hand.zip`) from the
   [official site](https://lmb.informatik.uni-freiburg.de/resources/datasets/HanCo.en.html) and
   copy sequence `0191` into the layout above:
   ```bash
   H=/path/to/HanCo          # unzipped HanCo_rgb.zip and HanCo_mask_hand.zip
   D=data/HanCo/0191
   for c in 3 5 6 7; do
     mkdir -p $D/rgb/cam$c $D/mask/cam$c
     cp $H/rgb/0191/cam$c/000000{00..75}.jpg $D/rgb/cam$c/
     cp $H/mask_hand/0191/cam$c/000000{00..75}.jpg $D/mask/cam$c/
   done
   mkdir -p $D/anchor/rgb $D/anchor/mask
   cp $D/rgb/cam6/00000075.jpg $D/anchor/rgb/cam6_00000075.jpg
   python -c "from PIL import Image; Image.open('$H/mask_hand/0191/cam6/00000075.jpg').convert('L').save('$D/anchor/mask/cam6_00000075.png')"
   ```
2. Download `eval_data/InstantHand_eval_HanCo.zip` (our annotations for sequence 0191) from our
   [Google Drive folder](https://drive.google.com/drive/folders/10iqqssndhqTGFWGWVBmpnCxzchryCdUE) and unzip it in the repository root.

HanCo is provided for research purposes only; if you use it, please cite FreiHAND
(Zimmermann et al., ICCV 2019) and HanCo (Zimmermann et al., 2021).

### Running the evaluation

```bash
python eval_interhand.py --cfg configs/eval_interhand.yaml
python eval_hanco.py --cfg configs/eval_hanco.yaml
```

Expected results (Tables 1 and 2 of the paper):

| Benchmark | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| InterHand2.6M | 26.21 | 0.866 | 0.133 |
| HanCo (zero-shot) | 20.18 | 0.836 | 0.188 |

`eval_interhand.py` also reports the boundary metrics of Table 3: boundary F-score 0.698,
boundary IoU 0.562, alpha leakage 0.234, interior coverage 0.999 (`--no_boundary` skips them).
`--verbose` prints the unrounded means.

## Optional inference-time settings

Both are off by default, are not used for the main results of the paper, and are available in
`demo.py` only. The evaluation scripts always use the paper setting.

**Seam blending** (`--seam_blend`). Where Gaussians colored from the input meet Gaussians
completed from the prior, a color step can appear as a line in novel views. This replaces the
step by a smooth ramp over ±3 mm around that boundary (`postprocess.seam_blend_mm`); geometry
and all other colors are unchanged. At 3 mm it leaves the benchmarks unchanged or slightly
better (PSNR: InterHand2.6M 26.21 → 26.23 dB, HanCo 20.18 → 20.22 dB). Wider bands remove
stronger seams but can soften fine patterns on the fingers.

**k=4 densification** (`--k4`). The trained avatar is interpolated onto the anchors of the next
MANO subdivision level (4× as many Gaussians) with all scales multiplied by 0.6
(`densify.scale`), which reduces the striations at strongly flexed knuckles (Appendix,
Table S10 and Fig. S11). It needs a second copy of the network and more GPU memory while the
avatar is built; rendering speed is unchanged.

```bash
python demo.py --ids all --seam_blend --k4
```

## License

| Component | License |
|---|---|
| InstantHand code | [Apache-2.0](LICENSE) |
| Portions derived from [LAM](https://github.com/aigc3d/LAM) and [LHM](https://github.com/aigc3d/LHM) | Apache-2.0 |
| `diff-gaussian-rasterization/` (modified) | [Gaussian-Splatting License](diff-gaussian-rasterization/LICENSE.md) (Inria, non-commercial research use) |
| `instanthand/models/encoders/dinov3/` | [DINOv3 License](instanthand/models/encoders/dinov3/LICENSE.md) |
| `external/handavatar/` | [MIT](external/handavatar/LICENSE) |
| `external/handavatar/third_parties/lpips/` | [BSD](external/handavatar/third_parties/lpips/LICENSE) |
| Pretrained checkpoint | non-commercial research use; the DINOv3 backbone in the full checkpoint is under the DINOv3 License |
| MANO | not included; see the [MANO license](https://mano.is.tue.mpg.de/license.html) |

Because the rasterizer and the pretrained model are restricted to non-commercial research use,
the repository as a whole can only be used for non-commercial research.

## Acknowledgements

This codebase builds on [LAM](https://github.com/aigc3d/LAM) and [LHM](https://github.com/aigc3d/LHM),
and we thank their authors for releasing them. We also thank the authors of
[HandAvatar](https://github.com/SeanChenxy/HandAvatar) and [OHTA](https://github.com/zxz267/OHTA)
for their evaluation protocols and code,
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) for the rasterizer,
[DINOv3](https://github.com/facebookresearch/dinov3) for the image encoder, and
[WiLoR](https://github.com/rolpotamias/WiLoR) and [SAM 3](https://github.com/facebookresearch/sam3)
for hand pose and mask estimation.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{kim2026instanthand,
  title     = {{InstantHand}: Feed-Forward {3D} {Gaussian} Splatting for Animatable Hand Avatar Reconstruction from a Single Image},
  author    = {Kim, Chanho and Ali, Muhammad Salman and Gwak, Kyeonghwan and Shaheryar, Muhammad and Park, Incheol and On, Jeongwan and Baek, Seungryul},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```