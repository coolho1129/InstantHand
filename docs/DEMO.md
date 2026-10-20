# Demo

[← README](../README.md) · [Installation](INSTALL.md) · Demo · [Evaluation](EVAL.md) · [Training](TRAIN.md)

Needs MANO (1) and a checkpoint (3), plus the DINOv3 weights (2) with the light checkpoint; see
[Files to download](INSTALL.md#files-to-download). Run every command from the repository root.

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

## Examples

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

## Your own images

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
