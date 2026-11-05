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

The official docs note that the release tiling policy (non-overlapping tiles + one boundary tile) replaced an earlier
evenly-overlapping one; learned parameters are unchanged, but the repo's historical GenImage number was not recomputed
with the corrected tiling. SynthScope uses the current official policy.

## Fidelity check (identical to the official model)

The official `Detector` (official repo, `predict.py` path) and SynthScope's full pipeline (image file → registry
transform → model) with the same checkpoint, on CPU: **333/333 state-dict tensors identical** (37 MIL / fusion + 296
CLIP), the view tensors equal the official `prepare_image` bit for bit, and **max |P(fake) diff| 0.0 and max |logit diff|
0.0** over 27 images (14 real, 13 fake; 100×300 to 1500×1001, including sides under 224, exact multiples of 224 such as
448×224 and 896×1120, and non-multiples such as 500×375).

## Results

| Benchmark | Paper | SynthScope_official | SynthScope_train | Status |
|---|---|---|---|---|
| **GenImage** (8 subsets, accuracy at 0.5) | 98.39 macro accuracy (official repo, historical evaluation) | pending | — (not trained) | pending |
