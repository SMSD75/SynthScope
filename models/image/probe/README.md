# PROBE (`probe`, `probe_resnet`)

**Where Detectors Fail: Probing Generative Space for Generalizable AI-Generated Image Detection** (ICML 2026) —
[paper (arXiv 2605.24906)](https://arxiv.org/abs/2605.24906) ·
[official code](https://github.com/Amamiya-C/PROBE-AIGI-Detection) ·
[checkpoints](https://modelscope.cn/models/shuinishaojiu/PROBE-AIGI-Detection)

PROBE (Probing Robustness via Boundary Exploration) is a way to train a detector, not a new architecture. Detectors
fail on generators they have not seen because the training data covers only a small part of what generators can
produce. PROBE uses the detector as a critic: it steers an image generator through small edits of its internal
representations ("manifold-level modifications") towards realistic images that the detector misclassifies, and then
refines the detector on those hard samples. The released detectors are therefore plain classifiers with one logit:

* **PROBE-DINOv2** (`probe`): DINOv2 ViT-L/14 with registers, fine-tuned, with a linear head on the CLS token; trained
  on the paper's *Reconstruction Training Set* (real images reconstructed with SD 2.1) plus PROBE samples. README:
  93.9 average balanced accuracy over 7 benchmarks.
* **PROBE-ResNet50** (`probe_resnet`): a ResNet-50 with a linear head, trained on GenImage SD v1.4 plus PROBE samples
  (78.1 average). Both are rows of the paper's Table 1, so both are registered.

## Pipeline (official, reproduced exactly)

```
image ─ Image.open(path).convert('RGB')
      ─ pad_image to the crop size S, only if a side is shorter than S:
          probe:        S = 336, mode 'repeat' (tile the image with copies of itself until both sides reach 336)
          probe_resnet: S = 224, mode 'pad'    (paste it centered on a black 224×224 square; a longer side is cut)
      ─ ToTensor → ImageNet normalization (on the whole image)
      ─ non-overlapping S×S patches, row-major (unfold, stride S); the strip at the right / bottom that is narrower
        than S is not used                                                   → [N, 3, S, S], N depends on the size
      → each patch: DINOv2-reg ViT-L/14 → CLS → Linear(1024, 1)   |   ResNet-50 → avgpool → Linear(2048, 1)
      → mean of the N patch logits → P(fake) = sigmoid(mean)        (official evaluate_dino.py / evaluate_resnet.py)
```

The crop sizes are the official README commands (`evaluate_dino.py --crop_size 336`, `evaluate_resnet.py --crop_size
224`); the pad modes are the scripts' `data_augment_params['pad_mode']`. Every test-time perturbation of the official
`data_augment` (blur, JPEG, noise, resize, cutout) has probability 0 in the README commands, so it is the identity.

Weights: `DINOv2_best_model_step_34999.pth` (1.22 GB, sha256 `c8caa9ba…450156`) holds the full model (all 440 backbone
tensors + the head) and loads key-for-key as `probe.pth`, so the backbone download (`facebook/dinov2-with-registers-
large`) is not needed for inference; `ResNet50_best_model_step_19343.pth` (94 MB, sha256 `71ef257e…20f0507c`) loads
key-for-key as `probe_resnet.pth`. Both are `{"model_state_dict": ...}` (ModelScope
`shuinishaojiu/PROBE-AIGI-Detection`). Place them under `$SYNTHSCOPE_CKPT_DIR` with those names.

## SynthScope adaptations

| choice | reason |
|---|---|
| `official/dino_classifier.py` and `official/resnet50.py` = the official model files | `SynthScopePROBEModel` subclasses the official `dino_classifier`, `SynthScopePROBEResNetModel` the official `ResNet` (as `resnet50(num_classes=1, classifier_type='linear')`), so the checkpoints load strictly (`ckpt_kind="official_probe"`); attribution and licence status in the headers. `resnet50.py` is unchanged. |
| DINOv2 backbone built from its config (`DINOV2_REG_LARGE`), not `AutoModel.from_pretrained(model_name)` | the checkpoint already holds the backbone weights; `backbone_path` still loads a pretrained HF-format backbone (for training). Both give the same `Dinov2WithRegistersModel` with the default (sdpa) attention. |
| transform steps `probe_pad_repeat_336` / `probe_pad_224` (the official `pad_image`, vendored unchanged in `preprocess.py`), `to_dtype`, `normalize_imagenet`, `sliding_patches_336` / `sliding_patches_224` (the official `generate_sliding_patches`) | the official per-image preprocessing as eval-list steps; the transform returns the image's patch bag `[N, 3, S, S]`. |
| the mean over patches is in `forward` (`[B, N, 3, S, S]` → `[B, 1]`); `native_res=True` scores one image at a time | the official `evaluate` averages the patch logits of each image; N varies with the image size, so bags cannot be stacked across images. |
| patches go through the network in chunks of `patch_chunk_size: 32` | bounds memory on very large images; the official code runs all patches of a batch of images in one forward. Chunking does not change the scores here (max \|diff\| 0.0 against the official batch-1 run, including a 40-patch image). `patch_chunk_size: 0` runs all patches at once. |
| `remap_probe` unwraps `model_state_dict` | the official `checkpoint['model_state_dict']` load (no key renaming). |
| an unreadable image is skipped by `bench_eval` (with a "bad image" print) | the official loader scores it as P(fake) = 0.5 (logit 0); no effect on readable data. |
| image files of every format PIL reads | the official loader lists `.jpg`, `.jpeg`, `.png` only; the pixels of those formats are the same. |
| `compute_loss` = BCE with logits on one patch per image, `outputs` = `[-logit, logit]` | the official fine-tuning and boundary-exploration code is not released (README TODO), so there is no training recipe; this is the repo convention for single-logit detectors. |

## Fidelity check

The official code (`probe_ref` at commit b145f71, unmodified, in a separate process: `evaluate_dino.EvaluateDataset`
/ `evaluate_resnet.EvaluateDataset`, `collate_fn` and `evaluate`, with the README settings, `fake_equal_real=False` so
every file is scored, the backbone path pointed at a local copy of `facebook/dinov2-with-registers-large` @ e4c89a4,
CPU) against SynthScope's full pipeline: raw file → `bench_eval._open_rgb` → registry transform → engine-built model
with the checkpoint loaded by the engine (`_load_checkpoint`, strict) → `engine._fake_prob`, through `bench_eval.score`.

Images: 34 files (16 real, 18 fake; PNG and JPEG incl. one RGBA PNG and one grayscale JPEG; 128×128 to 3000×1999; sides
below, at and just under the crop size, e.g. 335×700 and 336×500): 1 to 40 patches per image for `probe` (179 in
all), 1 to 104 for `probe_resnet` (389).

Results (CPU, torch 2.11, transformers 5.5.4; logs `repro/logs/probepar_cpu_{official_dino,official_resnet}_bs{1,8}.log`
and `probepar_cpu_ours_{dino,resnet}.log`):

| | `probe` (DINOv2, 336) | `probe_resnet` (ResNet-50, 224) |
|---|---|---|
| state dict (strict engine load) | 442/442 tensors identical to the checkpoint and to the official model (sha256 of the bytes); missing 0, unexpected 0 | 320/320 identical; missing 0, unexpected 0 |
| model inputs (patch bags) | 34/34 identical to the official tensors (sha256) | 34/34 identical |
| P(fake), mean logit vs official batch 1 | **max \|diff\| 0.0 / 0.0**, 0 flips | **max \|diff\| 0.0 / 0.0**, 0 flips |
| P(fake), mean logit vs official batch 8 (README) | 2.6e-8 / 1.9e-6, 0 flips | 1.8e-7 / 5.7e-6, 0 flips |

The batch-8 differences are the official code's own batch-size effect (it runs the patches of 8 images in one
forward), not a SynthScope difference. A checkpoint with one tensor removed or one extra key makes the engine raise.
The code tested is commit f01b3ee; `patch_chunk_size` 32 and 0 (all patches at once) give identical scores.

Memory and speed (CPU, fp32, 16 threads on a shared login node, the 34 parity images scored one at a time as the
engine does, second pass timed; logs `repro/logs/probepar_cpu_speed_{probe,probe_resnet,probe_chunk0}.log`):

| | `probe` | `probe_resnet` |
|---|---|---|
| model time per patch | 507 ms (336×336) | 28 ms (224×224) |
| one-patch image (e.g. 500×375) | 0.90 s | 0.13 s |
| largest image (3000×1999) | 40 patches, 16.5 s | 104 patches, 2.0 s |
| mean per image (incl. preprocessing) | 2.68 s | 0.32 s |
| peak RSS (load + inference) | 2.98 GiB (3.19 GiB with `patch_chunk_size: 0`) | 1.24 GiB |

The cost grows with the image area: an image is scored on all its patches.

## Results

**Verdict: faithful port (SynthScope equals the official code on every file compared: the 34 parity images, WildRF and
SynthWildX); the released checkpoints reproduce the
paper's Table 1 on four of the five benchmarks we can run (GenImage, SynthWildX, WildRF, AIGI-Bench) to within 0.9
points per benchmark; Synthbuster uses different reals and is not a like-for-like comparison. Chameleon and
AIGI-Quality-Paradox were not run (no loader), so the paper's 7-benchmark averages (93.9 / 78.1) are not comparable
with the means below.**

Paper = Table 1 of arXiv 2605.24906 (balanced accuracy, %, threshold 0.5, positive = fake; per-benchmark Avg = unweighted
mean of the per-subset bAcc of its Tables 2-4). SynthScope_official = the released checkpoint run through SynthScope
(`benchmark/bench_eval.py --model probe` / `--model probe_resnet`); its `means.acc` is the same quantity (the mean over
subsets of (real accuracy + fake accuracy) / 2). SynthScope_train = "—": the training code (generator probing and
fine-tuning) is not released, so there is no SynthScope-trained model.

| Benchmark | Paper PROBE-DINOv2 | SynthScope_official `probe` | Paper PROBE-ResNet50 | SynthScope_official `probe_resnet` | SynthScope_train |
|---|---:|---:|---:|---:|:-:|
| GenImage (val100k, fakes JPEG q96) | 96.8 | **97.47** (+0.67) | 78.4 | **78.38** (-0.02) | — |
| Synthbuster (**different reals**: 5000 MSCOCO, not RAISE-1k) | 97.5 | 98.20 (not like-for-like) | 79.1 | 83.75 (not like-for-like) | — |
| SynthWildX | 96.4 | **97.30** (+0.90) | 75.6 | **75.20** (-0.40) | — |
| WildRF | 96.9 | **96.92** (+0.02) | 81.3 | **81.63** (+0.33) | — |
| AIGI-Bench (CommunityAI + SocialRF) | 89.9 | **89.88** (-0.02) | 70.4 | **70.44** (+0.04) | — |
| Chameleon | 86.6 | — (no loader; by request only) | 74.0 | — (no loader) | — |
| AIGI-Quality-Paradox | 92.9 | — (no loader) | 87.7 | — (no loader) | — |
| Mean of the 5 benchmarks we ran | 95.50 | 95.95 | 76.96 | 77.88 | — |
| Mean of the 4 like-for-like benchmarks (no Synthbuster) | 95.00 | 95.39 | 76.43 | 76.41 | — |
| Paper's Avg. over all 7 | 93.9 | not comparable | 78.1 | not comparable | — |

The SynthScope columns are `means.acc` of `repro/bench_results/probe_<benchmark>.json` and
`probe_resnet_<benchmark>.json` (tags below). Their mean AP / AUROC (the paper does not report them in Table 1): GenImage
99.77 / 99.80 (DINOv2), 86.69 / 84.81 (ResNet-50); Synthbuster 99.64 / 99.93, 85.23 / 95.88; SynthWildX 98.80 / 99.18,
83.88 / 82.80; WildRF 99.36 / 99.26, 89.45 / 85.76; AIGI-Bench 96.46 / 95.88, 79.80 / 77.55.

### Protocol

* **Metric**: balanced accuracy at threshold 0.5 (mean of real and fake accuracy), positive class = fake; the per-benchmark
  value is the unweighted mean over the paper's subsets. The paper checks out against itself: the mean of its eight
  GenImage DINOv2 values is 96.76 (Table 1: 96.8), of its nine Synthbuster values 97.52 (97.5), of its WildRF values 96.93
  (96.9).
* **Inference**: the SynthScope registry transform of Task 1 (exact parity with the official code), batch size 1 (native
  resolution), fp32, one RTX A6000 per job.
* **GenImage**: the official val split, 100,000 images, 8 generators, **all fakes JPEG-compressed at quality 96, reals
  untouched** (`--jpeg-fakes 96`, the same PIL round trip as the official `util.jpeg_compress`), as the paper does to remove
  GenImage's compression bias. These numbers are for that compressed protocol; they are not comparable with GenImage
  numbers measured on the uncompressed split. The folders hold
  6000 real + 6000 fake images per generator except SD v1.5 with 8000 + 8000 (the paper's Table 12 lists 48k / 48k).
* **Synthbuster**: 9 generators, 1000 fakes each, paired with **5000 MSCOCO val2017 reals** (the `EvalGEN/Flux/0_real`
  folder, the same reals as the MIRROR and B-Free Synthbuster runs). The paper pairs them with the **1000 RAISE-1k reals**
  (its Table 12); RAISE is distributed through a personal registration form that was not filled in, so it is not in
  SynthScope. These rows are therefore a *different-reals* comparison and are not presented as reproducing the paper.
* **SynthWildX**: dalle3 / firefly / midjourney_v5, 413 reals and 392 / 473 / 435 fakes (1713 unique images: 413 reals scored against each of
  the 3 fake subsets; the paper's Table 12 lists 0.5k reals and 1.5k fakes: our copy is the part still online).
* **WildRF**: facebook / reddit / twitter, 2503 files (160 + 160, 750 + 750, 341 + 342); the paper lists 1.25k + 1.25k.
* **AIGI-Bench**: only the two subsets the paper evaluates, **CommunityAI** (6000 real + 6000 fake) and **SocialRF** (3000
  real + 3000 fake), 18,000 files (the paper's 9k / 9k), staged as `aigibench/{CommunityAI,SocialRF}/{0_real,1_fake}`; the
  other subsets of the AIGIBench release (about 330 GB) were not scored.
* **Not run**: Chameleon (available by request only) and AIGI-Quality-Paradox have no loader in SynthScope.
* No JPEG alignment on any benchmark other than GenImage (the paper applies it to GenImage only).

### Per-subset results

balanced accuracy, % (real accuracy / fake accuracy in brackets); `Paper` = Tables 2-4, `Avg` row = Table 1 for the
paper and `means.acc` for SynthScope.

**GenImage (val100k, fakes JPEG q96)**

| subset | Paper DINOv2 | SynthScope_official `probe` (real / fake acc) | Paper ResNet-50 | SynthScope_official `probe_resnet` (real / fake acc) |
|---|---:|---:|---:|---:|
| midjourney | 98.8 | 98.73 (98.57 / 98.9) | 90.7 | 90.73 (98.67 / 82.8) |
| sdv1.4 | 99.4 | 99.41 (98.85 / 99.97) | 99.4 | 99.36 (98.72 / 100.0) |
| sdv1.5 | 99.4 | 99.39 (98.9 / 99.87) | 99.2 | 99.24 (98.6 / 99.89) |
| adm | 91.0 | 93.22 (98.55 / 87.88) | 60.0 | 58.94 (98.52 / 19.37) |
| glide | 92.1 | 93.86 (99.07 / 88.65) | 63.0 | 63.01 (98.78 / 27.23) |
| wukong | 99.5 | 99.33 (98.68 / 99.97) | 99.4 | 99.41 (98.87 / 99.95) |
| vqdm | 98.9 | 98.76 (98.62 / 98.9) | 66.9 | 66.90 (98.72 / 35.08) |
| biggan | 95.0 | 97.07 (98.75 / 95.38) | 49.4 | 49.41 (98.82 / 0.0) |
| **Avg** | **96.8** | **97.47** | **78.4** | **78.38** |

**Synthbuster (5000 MSCOCO reals, not the paper's RAISE-1k)**

| subset | Paper DINOv2 | SynthScope_official `probe` (real / fake acc) | Paper ResNet-50 | SynthScope_official `probe_resnet` (real / fake acc) |
|---|---:|---:|---:|---:|
| dalle2 | 94.6 | 93.47 (99.54 / 87.4) | 48.3 | 52.97 (99.14 / 6.8) |
| dalle3 | 99.4 | 99.57 (99.54 / 99.6) | 90.2 | 94.82 (99.14 / 90.5) |
| firefly | 99.5 | 99.77 (99.54 / 100.0) | 46.9 | 51.57 (99.14 / 4.0) |
| glide | 86.7 | 92.22 (99.54 / 84.9) | 57.4 | 62.07 (99.14 / 25.0) |
| midjourney-v5 | 99.5 | 99.77 (99.54 / 100.0) | 93.2 | 97.82 (99.14 / 96.5) |
| stable-diffusion-1-3 | 99.5 | 99.72 (99.54 / 99.9) | 94.9 | 99.57 (99.14 / 100.0) |
| stable-diffusion-1-4 | 99.5 | 99.77 (99.54 / 100.0) | 94.9 | 99.57 (99.14 / 100.0) |
| stable-diffusion-2 | 99.5 | 99.77 (99.54 / 100.0) | 93.1 | 97.72 (99.14 / 96.3) |
| stable-diffusion-xl | 99.5 | 99.72 (99.54 / 99.9) | 93.0 | 97.67 (99.14 / 96.2) |
| **Avg** | **97.5** | **98.20** | **79.1** | **83.75** |

**SynthWildX**

| subset | Paper DINOv2 | SynthScope_official `probe` (real / fake acc) | Paper ResNet-50 | SynthScope_official `probe_resnet` (real / fake acc) |
|---|---:|---:|---:|---:|
| dalle3 | 96.5 | 97.44 (95.64 / 99.23) | 90.3 | 89.71 (89.1 / 90.31) |
| firefly | 96.4 | 96.98 (95.64 / 98.31) | 56.7 | 56.29 (89.1 / 23.47) |
| midjourney_v5 | 96.5 | 97.48 (95.64 / 99.31) | 79.7 | 79.61 (89.1 / 70.11) |
| **Avg** | **96.4** | **97.30** | **75.6** | **75.20** |

**WildRF**

| subset | Paper DINOv2 | SynthScope_official `probe` (real / fake acc) | Paper ResNet-50 | SynthScope_official `probe_resnet` (real / fake acc) |
|---|---:|---:|---:|---:|
| facebook | 98.4 | 98.75 (98.12 / 99.37) | 87.8 | 87.81 (82.5 / 93.12) |
| reddit | 93.4 | 93.33 (94.4 / 92.27) | 72.9 | 72.87 (89.6 / 56.13) |
| twitter | 99.0 | 98.68 (99.12 / 98.25) | 83.3 | 84.20 (89.74 / 78.65) |
| **Avg** | **96.9** | **96.92** | **81.3** | **81.63** |

**AIGI-Bench (CommunityAI + SocialRF)**

| subset | Paper DINOv2 | SynthScope_official `probe` (real / fake acc) | Paper ResNet-50 | SynthScope_official `probe_resnet` (real / fake acc) |
|---|---:|---:|---:|---:|
| CommunityAI | 89.8 | 90.38 (97.17 / 83.6) | 71.0 | 70.99 (87.1 / 54.88) |
| SocialRF | 89.9 | 89.38 (89.87 / 88.9) | 69.9 | 69.88 (84.2 / 55.57) |
| **Avg** | **89.9** | **89.88** | **70.4** | **70.44** |


Reading the tables:

* **ResNet-50 reproduces the paper** on GenImage (7 of 8 generators within 0.1, ADM 58.94 vs 60.0; average 78.38 vs
  78.4), SynthWildX, WildRF and AIGI-Bench (every subset within 0.9, the largest being WildRF twitter 84.2 vs 83.3). Its
  weakness is real: it detects almost nothing from ADM, GLIDE, VQDM and BigGAN (fake accuracy 0 - 35 %) because it is
  trained on SD v1.4 only.
* **DINOv2**: Stable Diffusion, Wukong, VQDM and Midjourney are within 0.2 of the paper. **ADM, GLIDE and BigGAN are
  about 2 points above it** (+2.2, +1.8, +2.1), which makes the GenImage average 97.47 vs 96.8. SynthScope equals the
  official code wherever we compared the two (the 34 parity images of the fidelity check, and every WildRF and SynthWildX
  file, next section), and the official code was not run on GenImage itself. Since the pipeline is identical, the
  difference points to the data or preprocessing on the paper's side (the exact JPEG-q96 implementation and the exact val
  subset are not specified); we cannot determine it.
* **Synthbuster**: the real-class accuracy is fixed by the reals, so swapping RAISE for MSCOCO moves every subset by the
  same real-accuracy term. Inferred, not measured: for ResNet-50 the paper's bAcc of SD 1.3 / 1.4 (94.9), DALL-E 2 (48.3)
  and DALL-E 3 (90.2) is reproduced from our fake accuracies (100.0, 6.8, 90.5) if the RAISE real accuracy is about 89.8
  (ours on MSCOCO: 99.14); for DINOv2 the same reading gives about 99.0 (ours 99.54), and the per-subset differences are
  small except GLIDE (+5.5) and DALL-E 2 (-1.1). This is why 98.20 and 83.75 must not be quoted as "reproducing 97.5 and
  79.1".

### Official code on our data

To separate port from data, the **unmodified official** `evaluate_dino.py` / `evaluate_resnet.py` code path
(`EvaluateDataset`, `collate_fn`, `evaluate`, README settings, batch size 4; `repro/probe_task2/probe_official_eval.py`)
was run on the same files as SynthScope for WildRF and SynthWildX (job 66753). Per file: **0 flips at 0.5 on every
common file in all four runs; max |dP| = 4.8e-6 (DINOv2) and 1.3e-3 (ResNet-50, GPU convolution algorithms / TF32 at
batch 4 vs 1)**.

| run | SynthScope `means.acc` | official, bAcc on all files it reads | official "Overall Accuracy" (seed-42 class-balanced subsample) | Paper |
|---|---:|---:|---:|---:|
| `probe` WildRF | 96.92 | 96.88 | 96.78 | 96.9 |
| `probe` SynthWildX | 97.30 | 97.30 | 97.42 | 96.4 |
| `probe_resnet` WildRF | 81.63 | 81.65 | 80.99 | 81.3 |
| `probe_resnet` SynthWildX | 75.20 | 75.20 | 75.09 | 75.6 |

* The official script's default `fake_equal_real=True` subsamples the larger class to the smaller one (seed 42) and
  reports pooled accuracy on that sample; it equals bAcc up to sampling noise (at most 2.0 per subset, ResNet-50 on the unbalanced WildRF
  twitter subset: 82.29 vs 84.27; at most 0.66 per benchmark, ResNet-50 WildRF: 80.99 vs 81.65). `means.acc` is the paper's metric.
* **The official file lister keeps `.jpg` / `.jpeg` / `.png` only**, so it silently drops the 262 `.jfif` files of WildRF
  twitter (683 files in SynthScope, 421 in the official run). SynthScope scores all of them. On the 421 common files both
  give the same number (DINOv2 98.57, ResNet-50 84.27); on all 683 SynthScope gives 98.68 / 84.20. Every other subset is
  identical to two decimals.
* An unreadable image is scored P(fake) = 0.5 by the official code and skipped by `bench_eval`; no file in these
  benchmarks was unreadable.

Raw comparison: `repro/probe_task2/official_vs_synthscope.json`, `official_runs_summary.json`,
`paper_vs_synthscope.json`; scripts `repro/probe_task2/`.

### Tags and jobs

DAS6, one RTX A6000 per job, code at commit a6ba2c5 (clean checkout in every job; the checkpoints were verified against
`SHA256SUMS` in every job). Logs: `repro/logs/probebench_das6_<job>.out`; every `repro/bench_results/probe_*.json` matches
its `=== RESULT <tag>` log line and its `repro/MASTER_RESULTS.jsonl` line.

| tags (`probe_<x>`, `probe_resnet_<x>`) | benchmark | job |
|---|---|---|
| `genimage_val100k` | GenImage val100k, fakes JPEG q96 | 66753 (node409) |
| `synthbuster_coco` | Synthbuster vs 5000 MSCOCO reals | 66755 (node412) |
| `synthwildx`, `wildrf` | SynthWildX, WildRF (+ the official-code runs above) | 66753 |
| `aigibench` | CommunityAI + SocialRF | 66790 (node409) |

(Jobs 66754, 66757 and the IVI job 284282 were cancelled and produced no result used here.)

## Licence

The official repository has no LICENSE file and states no licence, and the ModelScope model page has an empty licence
field: the licence is **unclear**. The weights are not redistributed (not in the SynthScope model zoo or on the
Space); download them from ModelScope. See [LICENSES.md](../../../LICENSES.md).
