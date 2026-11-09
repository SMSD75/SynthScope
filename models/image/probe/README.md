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

Target: the paper's Table 1 (balanced accuracy, %, threshold 0.5). SynthScope_official = the released checkpoint run
through SynthScope (`bench_eval.py --model probe` / `--model probe_resnet`); SynthScope_train = "—" (no training code).

| Benchmark | Paper PROBE-DINOv2 | SynthScope_official | Paper PROBE-ResNet50 | SynthScope_official | SynthScope_train |
|---|---:|:-:|---:|:-:|:-:|
| GenImage | 96.8 | pending | 78.4 | pending | — |
| AIGI-Quality-Paradox | 92.9 | — (no loader) | 87.7 | — (no loader) | — |
| Synthbuster | 97.5 | pending | 79.1 | pending | — |
| Chameleon | 86.6 | — (by request only) | 74.0 | — (by request only) | — |
| SynthWildX | 96.4 | pending | 75.6 | pending | — |
| WildRF | 96.9 | pending | 81.3 | pending | — |
| AIGI-Bench | 89.9 | pending | 70.4 | pending | — |
| Avg. (7) | 93.9 | — | 78.1 | — | — |

"pending" = the benchmark reproduction (bench_eval on full data) has not been run yet.

## Licence

The official repository has no LICENSE file and states no licence, and the ModelScope model page has an empty licence
field: the licence is **unclear**. The weights are not redistributed (not in the SynthScope model zoo or on the
Space); download them from ModelScope. See [LICENSES.md](../../../LICENSES.md).
