# OmniAID (`omniaid`, `omniaid_v1`, `omniaid_genimage`)

**OmniAID: Decoupling Semantics and Artifacts for Universal AI-Generated Image Detection in the Wild** —
[paper (arXiv 2511.08423)](https://arxiv.org/abs/2511.08423) · [official code](https://github.com/yunncheng/OmniAID) ·
[Mirage-Test](https://huggingface.co/datasets/Yunncheng/Mirage-Test)

A detector trained on one kind of content (say, landscapes) often fails on another (faces), because it mixes up *what*
is shown with *how* it was made. OmniAID splits these apart with a hybrid mixture of experts on CLIP ViT-L/14@336:
semantic experts (human, animal, object, scene, ...) are routed per image, and an artifact expert is always on. The
authors also built **Mirage**, a train/test set of images from recent generators tuned for realism.

## Pipeline (official, reproduced exactly)

```
image ─ Resize((336, 336))   [v2]   |   Resize((512, 512)) → Resize((336, 336))   [v1-era releases]
      ─ ToTensor → CLIP normalization
      ├ frozen CLIP vision tower → pooler output → router → top-k semantic experts
      └ MoE ViT-L/14: every attention projection = shared main weights (SVD) + the routed semantic experts
                      + the always-on artifact expert
      → ln_post(CLS) → Linear(1024, 2) → logits;  P(fake) = softmax(logits)[:, 1]   (the official `prob`)
```

Three official releases (Hugging Face `Yunncheng/OmniAID`, MIT), each a full model loaded with `strict=True` as in the
official eval:

| key | release | trained on | experts × rank, router, top-k | preprocessing |
|---|---|---|---|---|
| `omniaid` | `checkpoint_omniaid_v2.pth` (recommended; served on the demo) | Mirage-Train | 6 × 1, 256, top-2 | single resize to 336 |
| `omniaid_v1` | `checkpoint_omniaid_v1.pth` (the paper's OmniAID-Mirage) | Mirage-Train | 6 × 8, 256, top-2 | 512 → 336 |
| `omniaid_genimage` | `checkpoint_omniaid_genimage_sd14.pth` (the paper's academic tables) | GenImage SD v1.4 | 3 × 4, 64, top-1 | 512 → 336 |

The expert counts, ranks and router widths were read from each checkpoint's tensor shapes (the README's
`config_omniaid_v1/v2.json` are not in the repo); top-k comes from the official configs and `reward/clean_test.py`.
The v1-era double resize is the pipeline the official code says to use for v1 weights; for the GenImage release it is
confirmed by the results below (it reproduces the paper).

## SynthScope adaptations

| choice | reason |
|---|---|
| `official_arch.py` = the official `models/OmniAID.py`, **byte-identical body** | `SynthScopeOmniAIDModel` subclasses the official `OmniAID`, so the checkpoints load key-for-key; MIT attribution in the header. |
| `forward` returns the 2-class logits (`cls`) | SynthScope's `ce` convention gives softmax[:, 1], the official `prob`. |
| one registry key per release | Each release has its own expert configuration and preprocessing. |
| `compute_loss` = cross-entropy on the logits | **The official recipe trains in two stages** (each semantic expert with hard routing, then the router with the artifact expert fixed) with orthogonality, load-balancing and gating losses; that recipe is not reproduced here. |

## Fidelity check (identical to the official model)

The official `OmniAID` (built from the release's config and loaded `strict=True`, as `main_finetune.py` does) and
SynthScope's class, for all three releases on an A5000: **every state-dict tensor identical** (2804 / 2804 / 1940, max
|diff| 0.0), the registry transforms equal the official v2 / v1 pipelines pixel for pixel, and **max |logit diff| 0.0
with 0 verdict flips** on 24 GenImage images (ADM, BigGAN, Midjourney, Wukong; real and fake). Job 265851.

## Results

Paper metric: accuracy at 0.5 per subset, mean over subsets (Tables 2, 3, 6). Every subset used here is balanced except
Mirage-Test's Anime (fakes only), so accuracy equals SynthScope's balanced accuracy.

| Benchmark | Release | Paper | SynthScope_official | SynthScope_train | Status |
|---|---|---|---|---|---|
| **GenImage** (8 subsets) | `omniaid_genimage` | 95.94 | **95.82** (AP 99.02) | — | **reproduced** (every subset within 1.5 pt) |
| **AIGCDetectBenchmark** (17) | `omniaid_genimage` | 88.87 | **88.87** (AP 97.88) | — | **reproduced** exactly |
| **Mirage-Test** (5 categories) | `omniaid_genimage` | 51.10 | **51.10** | — | **reproduced** exactly (every category) |
| **GenImage** (8 subsets) | `omniaid_v1` | 97.24 | **97.16** (AP 99.20) | — | **reproduced** (every subset within 0.6 pt) |
| **AIGCDetectBenchmark** (17) | `omniaid_v1` | 92.88 | **92.88** (AP 98.93) | — | **reproduced** exactly |
| **Mirage-Test** (5 categories) | `omniaid_v1` | 88.39 | **88.39** | — | **reproduced** exactly |
| Chameleon / DRCT-2M | `omniaid_genimage` | 77.35 / 88.21 | — | — | missing (Chameleon is by request; DRCT-2M is not in SynthScope) |

GenImage per subset (SynthScope vs paper): `omniaid_genimage` Midjourney 87.20 vs 85.7 · SD v1.4 98.92 vs 98.9 · SD v1.5
98.77 vs 98.8 · ADM 90.40 vs 91.4 · GLIDE 98.97 vs 98.7 · Wukong 97.50 vs 98.1 · VQDM 96.07 vs 97.3 · BigGAN 98.70 vs
98.7; `omniaid_v1` Midjourney 98.20 vs 98.0 · SD v1.4 98.69 vs 98.7 · SD v1.5 98.63 vs 98.4 · ADM 88.90 vs 89.5 · GLIDE
98.20 vs 98.3 · Wukong 98.43 vs 98.6 · VQDM 98.40 vs 98.4 · BigGAN 97.87 vs 98.1.

**Mirage-Test protocol.** The paper's mean averages the balanced accuracy ("Overall") of Human, Animal, Object and Scene with the fake accuracy of Anime, which has no reals; the loader pairs Anime with the other categories' reals, and SynthScope reports the same mean. `omniaid_genimage`: Human 59.38, Animal 55.90, Object 49.02, Scene 46.52, Anime 44.67 (paper Table 3: 59.38, 55.90, 49.02, 46.53, 44.67); `omniaid_v1`: Human 93.69, Animal 81.37, Object 84.92, Scene 87.07, Anime 94.92 (paper: 93.69, 81.37, 84.92, 87.07, 94.92). The recommended v2 release (`omniaid`, not in the paper) gives 78.46 (AP 96.48): Human 88.82, Animal 77.20, Object 75.09, Scene 70.89, Anime 80.28.

**In the wild (`omniaid`, v2; new measurements, not in the paper)** — balanced accuracy / AP: SynthWildX 86.02 / 95.52,
WildRF 88.36 / 97.28, BFree-Online 70.26 / 91.70, RealChain 61.71 / 84.02, EvalGEN 74.14 / 96.46, RRDataset 71.75 /
84.19. See [docs/WILD_BENCHMARKS.md](../../../docs/WILD_BENCHMARKS.md).

Runs: jobs 265856 (`omniaid_genimage`), 265857 (`omniaid_v1`), 265858 (`omniaid`), 265935-265937 (Mirage-Test); results in
`repro/bench_results/omniaid*.json`.

## Usage

```bash
synthscope detect photo.png --model omniaid             # v2; needs omniaid.pth in $SYNTHSCOPE_CKPT_DIR + HF access for CLIP
python benchmark/bench_eval.py --model omniaid_genimage --dataset genimage_full --root /data/GenImage_full
python benchmark/bench_eval.py --model omniaid_v1 --dataset mirage_test --root /data/MirageTest
```
