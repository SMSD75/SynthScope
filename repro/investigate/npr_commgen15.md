# Why NPR scores badly on CommGen15 (no implementation bug)

NPR (official ProGAN weights) scored mean AUROC 23 on CommGen15, i.e. worse than chance. We checked whether SynthScope's
NPR is wrong. It is not; the result comes from how CommGen15 is built and NPR's sensitivity to resampling.

**Our NPR equals the official one** (official repo `chuangchuangtan/NPR-DeepfakeDetection`, cloned 2026-10-04):
- Network: identical layers (3x3 stride-2 stem, BN, max-pool, 3+4 Bottleneck blocks, average pool, `Linear(512, 1)`); the
  residual is `x - up(down(x, 0.5))` with nearest-neighbour interpolation, scaled by 2/3, exactly as `networks/resnet.py`.
- Test-time preprocessing: the official `test.py` uses `Resize((256, 256))`, no crop, ImageNet normalization for every
  mixed-size test set; our registered transform is the same.
- Weights: the served `npr.pth` is bit-identical to the official `NPR.pth` (after stripping the `module.` prefix).

**It still works on what it was trained for** (ForenSynths, 100 real + 100 fake each, paper-faithful input):
ProGAN 100.0, StyleGAN2 99.9, CycleGAN 99.4, BigGAN 90.5, GauGAN 90.8 AUROC.

**CommGen15 has a format/size split between classes.** In every subset checked (akool, kling, flux, midjourney, sora,
nano_banana; 40 images per class), all real images are JPEG, 416-640 px (mean luma quantization step about 6-8), and all
fakes are 320x320 PNG. With the paper's square resize to 256, reals are downscaled about 2.5x and fakes about 1.25x; NPR, which
detects resampling traces, then finds the reals more "fake" than the fakes:

| Subset | AUROC, paper-faithful (resize 256) | AUROC, native resolution | AUROC, centre-crop 256 |
| --- | ---: | ---: | ---: |
| akool | 1.1 | 36.8 | 29.6 |
| kling | 11.9 | 84.4 | 70.8 |
| flux | 24.3 | 61.1 | 48.4 |
| midjourney | 48.3 | 82.0 | 73.0 |
| sora | 60.9 | 96.1 | 93.5 |
| nano_banana | 23.5 | 74.1 | 65.1 |

Removing the resize recovers most of the signal, which confirms the cause. JPEG matters too: our own FLUX.2 [klein] images
drop from AUROC 91.7 (PNG) to 74.9 when both classes are re-saved as JPEG quality 95.

**Takeaways.** Keep the paper-faithful default (it reproduces the paper); for datasets whose real and fake images differ in
size or format, also report NPR at native resolution (`bench_eval.py --native-res --bs 1`). A side fix from this check: the
NPR residual now crops odd image sizes by one pixel (as the official code's commented-out lines do), so native-resolution
scoring no longer crashes on odd sizes. The 256x256 path is bit-identical to before.

Script and raw numbers: `npr_commgen15_check.py`, `npr_commgen15_check.json`.
