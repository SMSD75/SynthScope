# B-Free (`bfree`)

**A Bias-Free Training Paradigm for More General AI-generated Image Detection** (Guillaro, Zingarini, Usman, Sud,
Cozzolino, Verdoliva; CVPR 2025) — [paper (arXiv 2412.17671)](https://arxiv.org/abs/2412.17671) ·
[official code](https://github.com/grip-unina/B-Free) · [project page](https://grip-unina.github.io/B-Free/)

Detectors often learn shortcuts: content (what generators like to draw) or format (PNG fakes vs JPEG reals). B-Free
removes them from the training data. Every fake is made from a real image by Stable Diffusion 2.1 self-conditioned
reconstruction or inpainting, so real and fake share content, and heavy augmentation (local inpainting, blending,
resizing, compression) teaches the detector to look at generation traces only. The model is a DINOv2 ViT-B/14 with
registers, fine-tuned end to end on these data.

## Pipeline (official, reproduced exactly)

```
image at its native resolution ─ ToTensor → ImageNet normalization   (no resize, no crop: official get_list_norm('resnet'))
  → patch embedding (14×14) of the whole image
  → five 36×36-patch windows (504×504 pixels: center + four corners); a smaller image has its patch grid tiled up to 36×36
  → DINOv2 ViT-B/14 + 4 registers (timm vit_base_patch14_reg4_dinov2) → one logit per window → mean of the five
  → P(fake) = sigmoid(logit)   (the official scores are the logit; bAcc at logit > 0)
```

Weights: `BFREE_dino2reg4/model_epoch_best.pth` from grip.unina.it (`bfree.pth` in SynthScope), the full model,
loaded strictly.

## SynthScope adaptations

| choice | reason |
|---|---|
| `official_arch.py` = the official `code/networks/wrapper5crops.py`, verbatim | The five-crop wrapper is the model. GRIP license (nonprofit use; notices kept). |
| `SynthScopeBFreeModel` subclasses `Wrapper5crops` and restores `nn.Module.load_state_dict` | The official wrapper splits a checkpoint itself; SynthScope loads the standard state dict, and `registry.remap_bfree` prefixes the timm keys with `model.` (the patch embedding stays `patch_embed.*`). |
| `forward` returns the logit squeezed to `[B]`; `compute_loss` = BCE | `predict_kind="bce"`: P(fake) = sigmoid(logit), so 0.5 is the official logit > 0. |
| registry `native_res=True`, `tf_native(ImageNet)` | The official evaluation feeds each image at its own size, one at a time; any resize would change what the five crops see. |
| `pretrained_backbone: false` | The released checkpoint holds the whole model; `true` starts from the DINOv2 weights for training. |

## Fidelity check (identical to the official model)

The official code (`get_network` + `load_weights` from the B-Free repo, official transform) and SynthScope's model
(registry remap, strict load, registry transform) on the released weights: **177/177 state-dict tensors identical** and
**identical logits** (|diff| 0.0) on seven images: the repo's four demo images and random images of 300×200 (smaller than
a crop), 504×504 and 1100×700. On the demo images the logits also match the authors' published `results.csv`
(-5.9375, -4.4419, +4.4305, +3.8500; difference ≤ 5e-6, CPU vs their GPU run). Script: CPU, login node.

## Results

Paper metric: balanced accuracy at logit 0 (bAcc). SynthScope numbers are the mean per-subset balanced accuracy with
the official checkpoint, scored one image at a time at native resolution.

| Benchmark | Paper | SynthScope_official | SynthScope_train | Status |
|---|---|---|---|---|
| **SynthWildX** (DALL-E 3, Firefly, Midjourney v5; images still online on X) | 95.6 | **95.64** (mAP 99.02, AUROC 98.93) | — | **reproduced** |
| **WildRF** Facebook / Reddit / Twitter | 95.6 / 86.2 / 97.3 (mean 93.0) | **95.31 / 86.47 / 97.66** (mean 93.15; mAP 98.55, AUROC 98.23) | — | **reproduced** (each platform within 0.4 pt) |
| Synthbuster (Midjourney, SDXL, DALL·E 2, DALL·E 3, Firefly) | 99.6 / 99.8 / 95.6 / 98.2 / 99.7 | — | — | missing (RAISE-1k reals not downloaded yet) |
| New generators (FLUX, SD 3.5) | 92.3 / 98.9 | — | — | missing (the paper's own images) |
| GenImage (unbiased: 5k + 5k, fakes JPEG-compressed like the reals) | 89.3 | — | — | missing (the subset is not released) |
| FakeInversion | 86.2 | — | — | missing (not in SynthScope yet) |
| BFree-Online (viral images still online) | — | 91.54 (mAP 98.65, AUROC 96.93) | — | reference (the paper reports the viral set separately) |
| RealChain (7 generators) | — | 67.59 (mAP 74.83, AUROC 70.07) | — | reference (not in the paper) |
| EvalGEN (5 generators; MSCOCO val2017 reals) | — | 95.18 (mAP 99.52, AUROC 98.94) | — | reference (not in the paper) |
| RRDataset (original / transfer / redigital) | — | 72.33 (mAP 80.34, AUROC 74.34) | — | reference (not in the paper) |

Both paper benchmarks we can run are reproduced: SynthWildX 95.64 vs 95.6 and WildRF 95.31 / 86.47 / 97.66 vs 95.6 /
86.2 / 97.3. Outside the paper: EvalGEN Flux 78.50 (fake acc 57.92; GoT, Infinity, NOVA and OmniGen ≥ 98.8),
RRDataset original 81.61 · transfer 73.84 · redigital 61.54. Results: `repro/bench_results/bfree_*.json` (jobs 267861,
268052).

## Training

B-Free's training data (self-conditioned SD 2.1 reconstructions and inpaintings of COCO images) are released in the
official repo (`training_data/`), but the training code is not. SynthScope does not train B-Free yet.

## Usage

```bash
synthscope detect path/to/image.jpg -m bfree
python benchmark/bench_eval.py --model bfree --dataset wildrf --tag bfree_wildrf      # one image at a time, native size
```

## License

The B-Free code and weights are released by GRIP-UNINA for informational and nonprofit purposes only
([license](https://www.grip.unina.it/download/LICENSE_OPEN.txt)); `official_arch.py` keeps the original notice.
