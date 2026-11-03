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

## Fidelity check (identical to the official model)

The official `EffortDetector` (its file loaded unchanged; DeepfakeBench's training-only registries and logging stood in
for) and SynthScope's model, both on each released checkpoint: **969/969 state-dict tensors identical**, the registry
transform equal to DeepfakeBench's `load_rgb` pixel for pixel (max |diff| 0.0), and **identical P(fake)** (|diff| 0.0) on
seven GenImage images, for both releases. CPU, job 269032.

## Results

Paper metric: accuracy at 0.5 (`effort`, GenImage-trained), mean over the 8 GenImage subsets (paper Table 9).

| Benchmark | Paper | SynthScope_official | SynthScope_train | Status |
|---|---|---|---|---|
| **GenImage** (8 subsets, SD v1.4-trained) | 91.1 | **89.83** (mAP 99.37, AUROC 99.48) | — | **reproduced** (1.3 below; SD v1.4 / v1.5 to the decimal) |
| UniversalFakeDetect (ProGAN-trained, Tables 2-3) | — | — | — | missing (the ProGAN checkpoint is not released) |
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

## Usage

```bash
synthscope detect path/to/image.jpg -m effort
python benchmark/bench_eval.py --model effort --benchmark pergen --data <GenImage>/test --tag effort_genimage
```

## License

The Effort code and checkpoints are released under CC BY-NC 4.0 (non-commercial use, with attribution).
