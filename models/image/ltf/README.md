# LTF (`ltf`)

**Learning to Falsify: Hierarchical Multiple Instance Learning for High Resolution AI-Generated Image Detection**
(Chen, Gao, Li, Wang, Zhu, Cai, Dong) — [paper (arXiv 2611.04529)](https://arxiv.org/abs/2611.04529) ·
[official code](https://github.com/HuiGuanLab/LTF) · [weights](https://huggingface.co/MuskAI/LTF-v3.3)

Most detectors shrink or crop an image to one fixed-size view, which throws away the fine pixel traces of a
high-resolution fake. LTF keeps the image at its own resolution: it cuts it into 224×224 tiles, adds one resized view
of the whole image, and treats the views as a bag of instances (multiple instance learning). A frozen CLIP ViT-L/14
reads every view; MIL pools the patch tokens inside each tile (L1) and the CLS tokens across views (L2), and a gated
fusion head makes the decision.

## Pipeline (official, reproduced exactly)

```
image (RGB, native size)
  ├ global view: Resize(224) on the short side (bicubic) → CenterCrop(224)
  └ tiles: 224×224 crops of the original pixels; starts 0, 224, 448, ... while a full tile fits, plus one tile aligned
           to the far edge if pixels remain (500 px → 0, 224, 276); a side under 224 px keeps one crop padded with black
  → ToTensor + CLIP normalization → [1 + tiles, 3, 224, 224]
  → CLIP ViT-L/14 (frozen), hidden tokens after block 24, before ln_post / projection → [views, 257, 1024]
  L1: gated-attention MIL over each tile's 256 patch tokens → tile bags → 8-head attention with a learned query → [1024]
  L2: gated-attention MIL over the CLS tokens of all views (global + tiles)                                → [1024]
  gate = sigmoid(MLP(attention-pooled L1 / L2 instance logits))
  logits = MLP([gate · L1, (1 − gate) · L2]) → [2];  P(fake) = softmax(logits)[1]   (the official fusion score_fake)
```

`ltf.pth` is the official `weights/epoch_20.pth` (Hugging Face `MuskAI/LTF-v3.3`, revision `a2a2e29c`, SHA-256
`32e9fa2e…`), unchanged: the GenImage checkpoint, the only one released. It holds the MIL head only (64 tensors). The
frozen CLIP is the pinned `backbones/ViT-L-14.pt` of the same revision (SHA-256 `b8cca3fd…`, byte-identical to OpenAI's
`ViT-L-14.pt`), loaded with the OpenAI `clip` package as the official code does. `resolve_backbone` takes
`ModelConfig.backbone_checkpoint`, else `$SYNTHSCOPE_LTF_BACKBONE`, else an existing OpenAI clip cache copy
(`$SYNTHSCOPE_CLIP_DIR` or `~/.cache/clip`), else downloads the pinned file, and always checks the SHA-256.

## SynthScope adaptations

| choice | reason |
|---|---|
| `official/` = the official `ltf/model.py`, `ltf/token_ops.py`, `ltf/mil/*.py` and `ltf/data/images.py`, **unchanged bodies** | `SynthScopeLTFModel` subclasses the official `LTF`, so the checkpoint loads key-for-key and the tiling is the official code; MIT notice in `official/LICENSE` and the file headers. |
| `remap_ltf` (registry) drops the `baseline_core.*` groups and prefixes `core.` | Same as the official `load_core`: 27 disconnected baseline tensors are excluded by name, the 37 retained MIL / fusion tensors load strictly (the engine's official-weights check raises on any missing or unexpected key). |
| `pretrained_prefixes = ("visual.",)` | The checkpoint does not contain the frozen CLIP; it is built from the pinned backbone file in `__init__`. |
| `forward` returns the fusion logits `[B, 2]` (`predict_kind="ce"`) | The official prediction is `softmax(logits_fusion)[:, 1]`; the L1 / L2 branch logits stay available through the official `LTF.forward`. |
| `ltf_views` transform step + `native_res=True` | The official `prepare_image` (global view + tiles) as one step; the view count depends on the image size, so the benchmarks score one image at a time. |
| `compute_loss` = cross-entropy on the level1, level2 and fusion logits (weight 1 each) | The official training objective. The official training data pipeline (crop / reconstruction pairs, per-step tile schedule) is not ported. |

**Tiling policy.** The official docs (`docs/reproduction.md`) say the release tiling (non-overlapping tiles plus one
boundary-aligned tile) corrected an earlier policy that spread the overlap evenly over all tiles, and that the repo's
historical GenImage number was not recomputed with the corrected tiling. SynthScope uses the current official policy, so
our numbers below are made with the corrected tiling, and they reproduce the paper's values (98.36 vs 98.4). We did not run
the earlier tiling, so this does not show whether it differed.

**Where the paper and the released code differ** (the code is the authority, and SynthScope follows it; from the official
`docs/reproduction.md`):

- Fusion: the paper writes a weighted sum followed by a linear classifier; the checkpoint uses gated concatenation
  followed by an MLP.
- Loss: the paper describes an additional bag loss; the released training path optimizes the three branch
  classification losses and does not include it.
- The paper's ablation table calls the patch-token bags L1 and the view-CLS bag L2 in a different way than the saved code;
  this README and the code use L1 for patch bags and L2 for view CLS tokens.

**Training is not ported.** `ltf` is evaluation-only in SynthScope. `compute_loss` is the official three-branch objective,
but there is no train transform list and the official data pipeline (crop / reconstruction pairs with a per-step tile
schedule) is not reproduced. Calling `compute_loss` on a non-view (4-D) batch raises a `NotImplementedError` that says so; to
train LTF, use the official repository. Only the GenImage checkpoint is released, so there is no `SynthScope_train` number.

## Fidelity check (identical to the official model)

The official `Detector` (official repo, `predict.py` path) and SynthScope's full pipeline (image file → registry
transform → model) with the same checkpoint, on CPU: **333/333 state-dict tensors identical** (37 MIL / fusion + 296
CLIP), the view tensors equal the official `prepare_image` bit for bit, and **max |P(fake) diff| 0.0 and max |logit diff|
0.0** over 27 images (14 real, 13 fake; 100×300 to 1500×1001, including sides under 224, exact multiples of 224 such as
448×224 and 896×1120, and non-multiples such as 500×375).

## Results

Paper: Table 1 (accuracy per subset at threshold 0.5; the Midjourney, SD v1.4, SD v1.5, ADM, GLIDE, Wukong, VQDM and
BigGAN columns; the paper trains every method on SD v1.4) and Table 4 ("Full LTF"). The paper's mAcc is 98.4; the unweighted
mean of its eight Table 1 values is 98.375 (the official README says 98.39).

**Protocol.** GenImage, the 8 subsets, real = 0 and fake = 1, native resolution (global view + 224 tiles, one image at a
time, fp32), the fusion branch, threshold 0.5, as in the official `evaluate.py`. Accuracy is the mean of real and fake
accuracy, which equals the official sklearn accuracy because every subset is balanced; the macro is the unweighted mean over
the 8 subsets. The paper's 100,000 images are the full official GenImage val: 7 subsets of 6000 + 6000, and SD v1.5 with
8000 + 8000.

Three runs, in `repro/bench_results/`:

- `ltf_official_genimage8.json`: the official `ltf_ref/evaluate.py` on `GenImage_full/test`.
- `ltf_synthscope_genimage8.json`: SynthScope (`bench_eval --benchmark pergen --bs 1`) on the same files. `GenImage_full/test`
  has 33,000 images: SD v1.4 in full (6000 + 6000) and the first 1500 files per class, by sorted name, for the other 7 subsets.
- `ltf_synthscope_genimage_val100k.json`: SynthScope on the full official val, 100,000 images. The official code was run on
  the 33k set only, not on the 100k set.

| Subset | Paper acc | SynthScope_official, 33k | SynthScope, 33k | SynthScope, full val (100k) | full val real / fake acc | full val AP |
|---|---:|---:|---:|---:|---:|---:|
| Midjourney | 97.3 | 97.67 | 97.67 | **97.26** | 97.90 / 96.62 | 99.52 |
| SD v1.4 | 98.9 | 98.80 | 98.80 | **98.80** | 97.60 / 100.00 | 100.00 |
| SD v1.5 | 98.6 | 98.47 | 98.47 | **98.57** | 97.25 / 99.89 | 99.99 |
| ADM | 98.4 | 98.63 | 98.63 | **98.44** | 97.67 / 99.22 | 99.86 |
| GLIDE | 98.1 | 98.47 | 98.47 | **98.12** | 97.60 / 98.63 | 99.77 |
| Wukong | 98.9 | 99.00 | 99.00 | **98.89** | 97.78 / 100.00 | 100.00 |
| VQDM | 98.7 | 98.47 | 98.47 | **98.64** | 97.48 / 99.80 | 99.97 |
| BigGAN | 98.1 | 97.53 | 97.53 | **98.15** | 97.65 / 98.65 | 99.78 |
| **macro** | **98.4** (98.375) | 98.38 | 98.38 | **98.36** | 97.62 / 99.10 | 99.86 |

Paper Table 4 ("Full LTF") against our full-val macros:

| | mAcc | real acc | fake acc | AUC | AP |
|---|---:|---:|---:|---:|---:|
| Paper, Table 4 | 98.40 | 97.68 | 99.09 | 99.89 | 99.86 |
| SynthScope, full val (100k) | 98.36 | 97.62 | 99.10 | 99.88 | 99.86 |

| Benchmark | Paper | SynthScope_official | SynthScope_train | Status |
|---|---|---|---|---|
| **GenImage** (8 subsets, accuracy at 0.5) | 98.4 mAcc (Table 1; 98.40 in Table 4) | **98.38** (official code, 33k set) | **98.36** (SynthScope, full val); 98.38 on the 33k set, equal to the official code | **reproduced**; every subset within 0.10 point of Table 1 (mean absolute difference 0.04) |

- **Parity with the official code is exact.** On the same 33,000 files all 8 subsets agree with `ltf_ref/evaluate.py` to the
  second decimal on real accuracy, fake accuracy, accuracy, AP and AUROC. This matches the logit-level result in the
  fidelity check above.
- **Against the paper.** The 33k set differs from the paper by up to 0.57 point per subset (BigGAN), which is sampling noise from
  1500 images per class (standard error about 0.25 point per subset); on the full val every subset is within 0.10.
- The checkpoint is GenImage-trained. Its real-image false-positive rate of about 2 to 3% is a property of the released
  weights, not of the port: the official code gives the same numbers. Several non-GenImage real photos are scored fake by the
  official code and by SynthScope alike, so do not read the GenImage numbers as in-the-wild performance.

Jobs (all on cees, cn029, one RTX A5000):

- SynthScope on the 33k set: 277009 (Midjourney), 277011 (SD v1.4), 277013 (SD v1.5, ADM, GLIDE), 277015 (Wukong, VQDM, BigGAN).
- Official code on the same files: 277010, 277012, 277014, 277016.
- Full val extraction: 277022. SynthScope on the full val: 277066 (Midjourney), 277068 (SD v1.5), 277069 (ADM), 277070 (GLIDE),
  277071 (Wukong), 277072 (VQDM), 277073 (BigGAN).
- The full-val SD v1.4 row reuses job 277011: `GenImage_full/test/sdv1.4` and `GenImage_val_full/test/sdv1.4` are
  identical (6000 + 6000 names and sizes identical, and a 300-file byte sample per class equal; CPU diff job 277197).
- Merged by `repro/ltf_genimage8_merge.py`.

## Usage

```bash
synthscope detect photo.png --model ltf     # needs ltf.pth (the official epoch_20.pth) in $SYNTHSCOPE_CKPT_DIR, and the pinned CLIP
python benchmark/bench_eval.py --model ltf --benchmark pergen --data /data/GenImage_full/test --bs 1
```

Scoring is one image at a time (`native_res`: the number of views depends on the size), about 2.7 images per second at
1024 px (26 views) and 7.4 at 256 px on an RTX A5000.

**Weights.** `ltf.pth` is not in the SynthScope weights zoo and the Space does not serve `ltf`: the Hugging Face weights card
(`MuskAI/LTF-v3.3`) has no license tag; only the code (MIT) is licensed. Until the authors confirm the terms for the
weights, fetch `weights/epoch_20.pth` from the official repository (SHA-256 `32e9fa2e…`, see `fetch_core` in `ltf.py`) and
save it as `ltf.pth` yourself. See [LICENSES.md](../../../LICENSES.md).
