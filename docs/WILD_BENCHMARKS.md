# Detectors on recent and in-the-wild benchmarks

Every SynthScope image detector, with its default (published) weights, on six benchmarks built from recent
commercial generators or from images collected online. Each cell is **balanced accuracy at a 0.5 threshold /
AUROC**, both the unweighted mean over the dataset's subsets (generators or platforms), in percent; bold marks
a balanced accuracy of 75 or more. `…` means the run has not finished yet.

All six datasets download and lay themselves out on first use, for example
`python benchmark/bench_eval.py --model cure --dataset realchain --root /data/RealChain`.
Each number comes from `repro/bench_results/<model>_<dataset>.json`; regenerate this page with
`python repro/build_wild_table.py`.

| Detector | RealChain | EvalGEN | SynthWildX | WildRF | BFree-Online | RRDataset | Mean acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CuRe | **79.7 / 85.5** | **98.1 / 99.7** | **92.7 / 96.5** | **95.9 / 99.1** | **91.6 / 98.1** | **82.0 / 88.9** | 90.0 |
| DDA | 67.2 / 75.0 | **97.9 / 99.8** | **90.8 / 96.4** | **90.2 / 96.1** | **95.0 / 98.9** | **76.0 / 81.7** | 86.2 |
| SPAI | 61.6 / 69.0 | **93.8 / 98.7** | **76.2 / 84.6** | **80.6 / 88.4** | 59.8 / 60.2 | 64.8 / 72.0 | 72.8 |
| Co-Spy | 71.0 / 76.5 | **96.7 / 99.5** | 64.3 / 69.5 | 61.6 / 71.2 | 56.3 / 73.3 | 66.8 / 73.9 | 69.4 |
| VIB-Net | 65.8 / 74.3 | **94.8 / 99.7** | 58.5 / 73.6 | 56.1 / 71.3 | 51.9 / 53.6 | 60.0 / 66.4 | 64.5 |
| DINOv3 probe | 50.3 / 49.7 | 66.6 / 81.7 | 72.8 / 82.5 | **77.3 / 87.7** | 57.1 / 76.5 | 62.6 / 69.5 | 64.4 |
| AIDE | 55.5 / 72.4 | 59.7 / 92.7 | 57.4 / 65.7 | 63.6 / 68.6 | 52.8 / 63.1 | 61.1 / 65.2 | 58.4 |
| FatFormer | 48.9 / 38.7 | 62.1 / 62.7 | 53.5 / 58.6 | 59.5 / 72.8 | 50.0 / 52.7 | 53.0 / 53.4 | 54.5 |
| SAFE | 49.8 / 53.7 | 50.8 / 60.0 | 50.0 / 51.1 | 58.5 / 58.5 | 50.2 / 48.7 | 62.9 / 65.7 | 53.7 |
| NPR | 47.2 / 51.7 | 52.6 / 50.6 | 50.4 / 51.7 | 65.0 / 70.9 | 44.1 / 32.2 | 61.1 / 64.1 | 53.4 |
| D3QE | 49.5 / 43.5 | 62.2 / 83.9 | 49.6 / 45.5 | 53.0 / 63.9 | 47.0 / 64.0 | 55.0 / 55.1 | 52.7 |
| PGC | 49.9 / 36.0 | 50.2 / 77.7 | 50.9 / 56.3 | 60.1 / 76.8 | 50.0 / 46.5 | 52.3 / 57.8 | 52.2 |
| DINOv2 probe | 47.9 / 38.8 | 52.6 / 57.6 | 51.1 / 49.6 | 58.9 / 63.6 | 50.0 / 19.4 | 50.6 / 43.0 | 51.8 |
| UnivFD / CLIP | 49.2 / 40.9 | 51.8 / 55.9 | 49.3 / 48.1 | 53.4 / 64.8 | 50.4 / 43.9 | 52.7 / 54.7 | 51.1 |
| Adapted DINOv2 | 50.0 / 46.0 | 50.9 / 64.9 | 50.0 / 48.5 | 50.4 / 63.4 | 50.0 / 19.0 | 53.0 / 60.0 | 50.7 |
| CNNDetection | 49.7 / 36.7 | 50.7 / 55.8 | 50.2 / 53.0 | 49.9 / 58.9 | 50.0 / 64.3 | 50.5 / 44.6 | 50.2 |

## The datasets

| Dataset (`config name`) | What it is | Caveat |
| --- | --- | --- |
| RealChain (`realchain`) | 7 recent generators (FLUX.1, SD 3.5, Qwen-Image, Hunyuan 3, NanoBanana, Seedream 4, image-to-image) vs 7000 reals, everything JPEG q60 | the 7000 unpaired reals are split into 7 slices of 1000 |
| EvalGEN (`evalgen`) | Flux, GoT, Infinity, OmniGen, NOVA on GenEval prompts, about 11k fakes each | fakes only; paired with MSCOCO val2017 reals (our choice) |
| SynthWildX (`synthwildx`) | DALL-E 3, Firefly, Midjourney v5 images posted on X, vs real X images | 1713 of 2000 images are still online |
| WildRF (`wildrf`) | real and fake images from Facebook, Reddit and Twitter | full test split, 1251 / 1252 |
| BFree-Online (`bfree_online`) | real and fake images that went viral, several web copies each | 807 of about 1400 images are still online |
| RRDataset (`rrdataset`) | real vs AI images as created, after sharing online and after re-digitization | test archive only; three scenarios, about 8500 / 8500 each |

## Reading the table

- CuRe's paper reports all six benchmarks, and CuRe reproduces it: RealChain 79.69 vs 79.7, EvalGEN 98.14 vs
  98.1, SynthWildX 92.69 vs 92.7, WildRF 95.89 vs 95.8, RRDataset 82.04 vs 81.5, BFree-Online 91.64 vs 90.9 (fewer images; see
  [the CuRe README](../models/image/cure/README.md)).
- DDA's paper reports four of them, and DDA reproduces it: SynthWildX 90.80 vs 90.9, WildRF 90.20 vs 90.3, BFree-Online
  94.98 vs 95.1 (fewer images) and EvalGEN 96.86 vs 97.2 in the paper's metric, the accuracy on the fakes alone (the
  table shows balanced accuracy; see [the DDA README](../models/image/dda/README.md)). Its RealChain and RRDataset rows
  are new measurements.
- D3QE is trained on LlamaGen vs ImageNet only and targets autoregressive generators; on these sets it calls almost
  every fake real (fake recall mostly under 15%), a new measurement that its paper does not report.
- The other detectors were never evaluated on these sets by their authors, so their rows are new measurements, not
  reproductions.
- Balanced accuracy uses a fixed 0.5 threshold. A detector with a high AUROC but a balanced accuracy near 50
  separates real from fake but is miscalibrated for this data (it calls nearly everything one class).
- An AUROC well below 50 means the detector ranks real images as more fake than the generated ones. We traced
  the same effect for NPR on CommGen15 to how the real and fake images were stored (JPEG reals vs PNG fakes,
  different sizes), not to a bug ([write-up](../repro/investigate/npr_commgen15.md)).
