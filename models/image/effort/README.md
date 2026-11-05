# Effort (`effort`, `effort_chameleon`)

**Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection** (Yan, Wang, Jin, Zhang, Liu, Chen,
Yao, Ding, Wu, Yuan; ICML 2025, oral) — [paper (arXiv 2411.15633)](https://arxiv.org/abs/2411.15633) ·
[official code](https://github.com/YZY-stack/Effort-AIGI-Detection)

A detector fine-tuned on a few generators tends to collapse its features onto those generators' patterns (a low-rank
feature space that does not generalize). Effort keeps CLIP's knowledge instead: every attention projection of the
CLIP ViT-L/14 vision tower is split by SVD into its principal part (the top 1023 singular directions, frozen) and the
remaining direction (trainable), so fine-tuning only moves an orthogonal minor subspace, with losses that keep the two
parts orthogonal and the singular values in place.

## Pipeline (official, reproduced exactly)

```
image ─ cv2 bicubic squash to 224×224 (the whole image, aspect not kept) ─ ToTensor ─ CLIP normalization
      → CLIP ViT-L/14 vision tower, each self-attention q/k/v/out projection = weight_main (rank 1023, frozen)
        + U_residual · diag(S_residual) · V_residual (rank 1, trained)
      → pooler output (1024) → Linear(1024, 2) → P(fake) = softmax(logits)[:, 1]   (the official `prob`)
```

The preprocessing is DeepfakeBench's test loader (`abstract_dataset.load_rgb` + `to_tensor` + `normalize`), which the
GenImage numbers come from; the face demo (`demo.py`) resizes bilinearly and crops faces, so it is not used here.

| key | release | trained on |
|---|---|---|
| `effort` | `effort_clip_L14_trainOn_sdv14.pth` (the paper's GenImage table) | GenImage SD v1.4 |
| `effort_chameleon` | `effort_clip_L14_trainOn_chameleon.pth` | Chameleon |

## SynthScope adaptations

| choice | reason |
|---|---|
| `official_arch.py` = `SVDResidualLinear`, `apply_svd_residual_to_self_attn` and `replace_with_svd_residual` from the official `effort_detector.py`, verbatim | They are the model; CC BY-NC 4.0, attribution in the header. |
| `SynthScopeEffortModel` = the official `EffortDetector` layout (`backbone` = HF CLIP vision tower, `head`) | The released checkpoints load key for key once the DataParallel `module.` prefix is dropped (`registry.remap_effort`, as the official demo does). |
| `pretrained_prefixes` = the principal SVD factors (`U_r`, `S_r`, `V_r`) | They are recomputed from the CLIP weights at construction (as in the official code) and only feed the training-time orthogonality loss; the checkpoints do not store them, so they are the only keys allowed to be missing. Every other tensor loads strictly. |
| `Cv2SquareResize` (`preprocess.py`) | cv2's `INTER_CUBIC`, as DeepfakeBench; PIL's bicubic differs. |

## Training (UniversalFakeDetect / ProGAN recipe)

The authors released no ProGAN checkpoint (their UFD Tables 2-3 model), only their training code
(`UniversalFakeDetect_Benchmark`). `bash experiments/image/effort_ufd.sh` (configs `conf/model/image/effort_ufd.yaml`,
`conf/trainer/image/effort_ufd.yaml`) trains it with that recipe; the served `effort` models are not affected.

| | official (`script/train_svd.sh`, `train.py`, `models/trainer.py`, `data/datasets.py`) | SynthScope |
|---|---|---|
| data | `wang2020` ProGAN train, all 20 classes (720k images) | CNNDetection `train`, all 20 classes |
| train transform | `RandomCrop(224)` of the 256 px image, `RandomHorizontalFlip`, CLIP mean/std (no resize; `--data_aug` is never applied) | the same (`resize_short_256` is the identity on 256 px images) |
| trainable | SVD residual factors of every self-attention projection + `fc` | the same (290 tensors; 197,729 parameters that can move, as the official) |
| optimizer | AdamW lr 2e-4 (constant: the decay call is commented out), betas (0.9, 0.999), weight decay 0 | the same |
| batch / precision | 48, fp32, one GPU, seed 0, no gradient clipping | 48, no accumulation (measured to fit 24 GB: peak 19.8 GiB, 1.22 s per step, smoke job 275708), fp32, one GPU, seed 0, no clipping |
| loss / head | BCE on one logit, `fc = Linear(1024, 1)`, weight N(0, 0.02) | exactly the same: the 2-logit head's row 0 is pinned at zero (`head_single_logit`: zeroed, its gradient zeroed by hooks, so AdamW never moves it), the logits are [0, z] and cross-entropy equals `BCEWithLogits(z)`; row 1 weight N(0, 0.02). The checkpoint keeps the 2-logit format (the served loader reads it unchanged) |
| checkpoints | `model_iters_<N>.pth` at fixed iterations; the test uses 18000 (~1.2 epochs) | `model_step<N>.pth` every 3000 iterations up to 24000 |
| test preprocessing | tile if < 256 px, `CenterCrop(224)`, CLIP mean/std | `eval_transform_list` of `effort_ufd.yaml` (the same, without the tiling) |

## Fidelity check (identical to the official model)

The official `EffortDetector` (its file loaded unchanged; DeepfakeBench's training-only registries and logging stood in
for) and SynthScope's model, both on each released checkpoint: **969/969 state-dict tensors identical**, the registry
transform equal to DeepfakeBench's `load_rgb` pixel for pixel (max |diff| 0.0), and **identical P(fake)** (|diff| 0.0) on
seven GenImage images, for both releases. CPU, job 269032.

## Results

Paper metrics: accuracy at 0.5 (`effort`, GenImage-trained), mean over the 8 GenImage subsets (paper Table 9); mAP and accuracy at the fixed threshold 0.5 over the 19 UniversalFakeDetect subsets (`effort_ufd`, ProGAN-trained, Tables 2-3). `SynthScope_train` is the checkpoint we trained ourselves.

| Benchmark | Paper | SynthScope_official | SynthScope_train | Status |
|---|---|---|---|---|
| **GenImage** (8 subsets, SD v1.4-trained) | 91.1 | **89.83** (mAP 99.37, AUROC 99.48) | — | **reproduced** (1.3 below; SD v1.4 / v1.5 to the decimal) |
| **UniversalFakeDetect** (ProGAN-trained, 19 subsets, Tables 2-3) | 99.41 mAP / 95.19 mAcc | — (no ProGAN checkpoint released) | **99.23 mAP / 94.81 mAcc** (AUROC 99.06) | **reproduced** (0.2 mAP and 0.4 acc below; every subset within 1.7 pt acc) |
| SynthWildX | — | 56.69 (mAP 61.20) | — | reference (not in the paper) |
| WildRF | — | 71.36 (mAP 81.11) | — | reference |
| BFree-Online | — | 67.75 (mAP 96.45) | — | reference |
| EvalGEN | — | 92.16 (mAP 99.55) | — | reference |
| RealChain | — | 50.56 (mAP 60.38) | — | reference |
| RRDataset | — | 64.59 (mAP 69.78) | — | reference |

Per GenImage subset (ours / paper): Midjourney 80.03 / 82.4 · SD v1.4 99.82 / 99.8 · SD v1.5 99.77 / 99.8 · ADM 80.67 / 78.7 ·
GLIDE 96.00 / 93.3 · Wukong 98.07 / 97.4 · VQDM 90.60 / 91.7 · BigGAN 73.70 / 77.6. Real accuracy is about 99.7 on every
subset, so the misses are unseen-generator fakes called real (the AP stays above 98.3 everywhere). Results:
`repro/bench_results/effort_*.json` (job 269039). Trained on SD v1.4 only, the model transfers poorly to in-the-wild
images (SynthWildX, RealChain).

### UniversalFakeDetect: Effort trained in SynthScope (ProGAN)

**Protocol (written down before the runs).** The paper (Section 4.2) reports AP and classification accuracy with the threshold fixed
at 0.5 (not a best threshold), over 19 subsets, and the official `UniversalFakeDetect_Benchmark/dataset_paths.py` lists exactly
them: the 11 CNNDetection test sets ProGAN, CycleGAN, BigGAN, StyleGAN, GauGAN, StarGAN, DeepFakes, SITD (`seeingdark`), SAN, CRN,
IMLE (not StyleGAN2 or WhichFaceIsReal) and the 8 UFD diffusion sets Guided, LDM 200 / 200 w/cfg / 100, Glide 100-27 / 50-27 /
100-10, DALL-E (reals: ImageNet for Guided, LAION for the rest). `validate.py` preprocessing is center crop 224 + CLIP mean/std,
which is `eval_transform_list` of `conf/model/image/effort_ufd.yaml`. We score the same checkpoint with
`benchmark/bench_eval.py --model effort --model-config image/effort_ufd --ckpt <file>`: `--model-config` makes a served detector use another
model config of the same detector (here the recipe's, whose test transform is the crop, not the served DeepfakeBench squash;
it needs `--ckpt`, and the web registry is untouched). The two runs (`--dataset cnndetection --select <the 11>`, `--dataset
universialfakedetection`) are combined by `repro/effort_ufd19.py` into the 19-subset mean. `bench_eval` scores every image of a
subset (real accuracy and fake accuracy averaged into a balanced accuracy, which equals plain accuracy on these balanced sets);
the official `validate.py` draws `--max_sample` 1000 images per class and falls back to **100** when a set has fewer (SAN and
SITD here), so the paper's SAN and SITD columns rest on 200 images each, ours on all of them (438 and 360). The sampling is the
only protocol difference we know of.

**Checkpoint selection.** The official test script uses iteration 18000 (`script/test.sh`), so iteration 18000 is the SynthScope_train
number; the other iterations show the curve (the run is the official recipe, `experiments/image/effort_ufd.sh`, 24000 iterations
of batch 48 on the 720k-image ProGAN set, job 276492, checkpoints every 3000 iterations; `model_final` is the 24000 weights saved again).

| iteration | mAP | mAcc (0.5) | AUROC |
|---:|---:|---:|---:|
| 6000 | 98.44 | 90.43 | 98.18 |
| 12000 | 99.23 | 93.43 | 99.07 |
| **18000** | **99.23** | **94.81** | 99.06 |
| 21000 | 99.02 | 92.38 | 98.79 |
| 24000 | 98.99 | 91.99 | 98.76 |
| final | 98.99 | 91.99 | 98.76 |

Iteration 18000 is fixed by the official script, not picked on these test sets. Accuracy peaks there and then falls while AP
stays near 99.0: real accuracy stays at 100 on the diffusion sets and the fake recall drops on the hard sets (fake accuracy at 24000 vs
18000: DeepFakes 48.2 vs 73.3, SAN 43.4 vs 63.0, Guided 23.9 vs 37.7, Glide 82-85 vs 93-95), the signature of a score threshold that
moves away from 0.5 while the ranking stays good. Per subset at iteration 18000 (ours / paper):

| subset | AP | acc | real acc | fake acc | acc at 24000 |
|---|---:|---:|---:|---:|---:|
| ProGAN | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 | 100.0 | 100.0 |
| CycleGAN | 100.0 / 100.0 | 99.85 / 99.85 | 99.7 | 100.0 | 99.92 |
| BigGAN | 99.99 / 99.99 | 99.5 / 99.6 | 99.45 | 99.55 | 99.55 |
| StyleGAN | 99.86 / 99.77 | 96.23 / 95.05 | 100.0 | 92.46 | 93.0 |
| GauGAN | 100.0 / 100.0 | 99.68 / 99.6 | 99.36 | 100.0 | 99.92 |
| StarGAN | 100.0 / 100.0 | 99.9 / 100.0 | 99.8 | 100.0 | 99.97 |
| DeepFakes | 98.35 / 98.95 | 86.45 / 87.6 | 99.63 | 73.28 | 74.11 |
| SITD (seeingdark) | 98.08 / 97.53 | 90.83 / 92.5 | 86.11 | 95.56 | 88.61 |
| SAN | 96.2 / 97.53 | 81.05 / 81.5 | 99.09 | 63.01 | 71.69 |
| CRN | 99.95 / 100.0 | 97.92 / 98.9 | 95.83 | 100.0 | 99.43 |
| IMLE | 99.98 / 100.0 | 97.91 / 98.9 | 95.83 | 99.98 | 99.55 |
| Guided | 93.47 / 95.39 | 68.85 / 69.15 | 100.0 | 37.7 | 61.95 |
| LDM 200 | 99.99 / 99.99 | 99.25 / 99.3 | 100.0 | 98.5 | 98.3 |
| LDM 200 w/cfg | 99.85 / 99.89 | 96.1 / 96.8 | 100.0 | 92.2 | 92.35 |
| LDM 100 | 100.0 / 100.0 | 99.4 / 99.45 | 100.0 | 98.8 | 98.65 |
| Glide 100-27 | 99.9 / 99.87 | 97.0 / 97.45 | 100.0 | 94.0 | 91.5 |
| Glide 50-27 | 99.93 / 99.92 | 97.35 / 97.8 | 100.0 | 94.7 | 92.25 |
| Glide 100-10 | 99.94 / 99.98 | 96.3 / 97.15 | 100.0 | 92.6 | 90.95 |
| DALL-E | 99.86 / 99.96 | 97.9 / 98.05 | 100.0 | 95.8 | 96.05 |
|**mean**| **99.23 / 99.41** | **94.81 / 95.19** | | | 91.99 |

Results: `repro/bench_results/effort_ufd_<iteration>_{gan11,ufd8,ufd19}.json` (tags `effort_ufd_step<N>_*`, `effort_ufd_final_*`), logs
`repro/logs/effuevl_<job>.out`. Eval jobs 278422 (6000, 18000, GenImage-8), 278423 (12000, 21000), 278424 (24000, final).
The checkpoints (2.0 GB each) stay on the cluster's shared filesystem and are **not published**: Effort is CC BY-NC 4.0.

**GenImage-8 for reference (not the paper's setting).** The same iteration-18000 checkpoint on the 8 GenImage subsets (`effort_ufd_step18000_genimage8`, job 278422):
78.81 acc / 95.27 mAP (AUROC 94.68). This is a different training set (ProGAN, 20 classes) from the served `effort` (GenImage SD v1.4), whose
numbers are 89.83 / 99.37 above, so the two are not comparable as a recipe check. A ProGAN-trained model is expected to transfer worse to
the GenImage generators: acc per subset (ProGAN-trained / SD v1.4-trained) Midjourney 54.10 / 80.03, SD v1.4 90.13 / 99.82, SD v1.5 85.13 / 99.77,
ADM 67.97 / 80.67, GLIDE 89.53 / 96.00, Wukong 81.57 / 98.07, VQDM 77.10 / 90.60, BigGAN 84.93 / 73.70 (BigGAN, an ImageNet GAN, is the one subset the
ProGAN model does better on).

## Usage

```bash
synthscope detect path/to/image.jpg -m effort
python benchmark/bench_eval.py --model effort --benchmark pergen --data <GenImage>/test --tag effort_genimage
```

## License

The Effort code and checkpoints are released under CC BY-NC 4.0 (non-commercial use, with attribution).
