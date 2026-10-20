# Installation

[← README](../README.md) · Installation · [Demo](DEMO.md) · [Evaluation](EVAL.md) · [Training](TRAIN.md)

Tested on Ubuntu with Python 3.10, PyTorch 2.3.0 and CUDA 11.8 (RTX 3090, RTX A6000, RTX 6000 Ada, A100).

```bash
git clone --recursive https://github.com/coolho1129/InstantHand.git
cd InstantHand
# if you cloned without --recursive:
git submodule update --init --recursive

conda create -n instanthand python=3.10 -y
conda activate instanthand
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA extensions; set the compute capability of your GPU
# (8.6: RTX 3090 / A6000, 8.0: A100, 8.9: RTX 4090 / RTX 6000 Ada; +PTX also runs on newer GPUs)
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

**Run every command from the repository root.** Config files, data paths and the paths stored in
the evaluation annotations are relative to it.

## Files to download

Everything else is in this repository. The files below come from other sources because of their
licenses or size; place them at these paths. The demo needs (1) and (3), plus (2) if you use the light
checkpoint; the evaluation also needs (4); training needs (1), (2) and (5).

```
InstantHand/
├── instanthand/models/rendering/mano_model/mano/
│   ├── MANO_RIGHT.pkl                                  ← (1) MANO
│   └── mano_mean_params.npz                              included (from WiLoR)
├── instanthand/models/encoders/dinov3/weights/
│   └── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth    ← (2) DINOv3 (light checkpoint or training only)
├── exps/neurips2026/stage2_best/
│   └── model.safetensors                               ← (3) InstantHand checkpoint
├── data/
│   ├── eval/                                           ← (4) evaluation data, see Evaluation
│   │   ├── InterHand/5/InterHand2.6M_5fps_batch1/
│   │   └── HanCo/0191/
│   └── train/                                          ← (5) training data, see Training
├── diff-gaussian-rasterization/third_party/glm/          git submodule (clone with --recursive)
└── demo/input/, demo/pose_bank/                          included
```

| | Place at | Source |
|---|---|---|
| (1) | `instanthand/models/rendering/mano_model/mano/MANO_RIGHT.pkl` | Register at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de), download `mano_v1_2.zip` and copy `mano_v1_2/models/MANO_RIGHT.pkl`. `MANO_LEFT.pkl` is not needed: left hands are mirrored and handled by the right-hand model. |
| (2) | `instanthand/models/encoders/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | Needed only with the light checkpoint or for training. Meta's DINOv3 download (access request), see [DINOv3 weights](#dinov3-weights) below. |
| (3) | `exps/neurips2026/stage2_best/model.safetensors` | One of the two checkpoints below, saved under this name. Another location works if you change `ckpt_dir` in `configs/demo.yaml`, `configs/eval_interhand.yaml`, `configs/eval_hanco.yaml` and `configs/eval_efficient.yaml` (or pass `--ckpt_dir`). |
| (4) | `data/eval/` | Official InterHand2.6M and HanCo images plus our archives, see [Data preparation](EVAL.md#data-preparation). |
| (5) | `data/train/` | The training datasets, converted to the layout of [Training data](TRAIN.md#training-data). Not redistributed. |

### Checkpoints

Both are on [Hugging Face](https://huggingface.co/coolho1129/InstantHand) and give identical results; the full one
includes the frozen DINOv3 ViT-L/16 backbone, so you do not need (2). Save the one you use as
`exps/neurips2026/stage2_best/model.safetensors`:

```bash
pip install -U huggingface_hub
hf download coolho1129/InstantHand checkpoints/instanthand_full.safetensors --local-dir .
mkdir -p exps/neurips2026/stage2_best
mv checkpoints/instanthand_full.safetensors exps/neurips2026/stage2_best/model.safetensors
```

For the light checkpoint, replace `instanthand_full` with `instanthand_light` in both commands.

| Checkpoint | Size | DINOv3 file (2) | Download |
|---|---|---|---|
| full | 1.4 GB | not needed | [`checkpoints/instanthand_full.safetensors`](https://huggingface.co/coolho1129/InstantHand/blob/main/checkpoints/instanthand_full.safetensors) |
| light | 0.2 GB | needed | [`checkpoints/instanthand_light.safetensors`](https://huggingface.co/coolho1129/InstantHand/blob/main/checkpoints/instanthand_light.safetensors) |

The DINOv3 backbone in the full checkpoint is distributed under the DINOv3 License (a copy,
[`DINOv3_LICENSE.md`](https://huggingface.co/coolho1129/InstantHand/blob/main/DINOv3_LICENSE.md), is included in the Hugging Face repository). The MANO model is not redistributed here (MANO license);
`mano_mean_params.npz` is included as released by [WiLoR](https://github.com/rolpotamias/WiLoR).

### DINOv3 weights

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

Next: [run the demo](DEMO.md).
