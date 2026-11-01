# MoE-JEPA (`moejepa`)

**Unifying Semantic Priors and High-Frequency Traces: Enhancing V-JEPA with Mixture-of-Experts for Robust Synthetic
Image Forensics** (Simone Teglia, Irene Amerini; AI4MFDD workshop @ ECCV 2026) —
[paper (arXiv 2609.16778)](https://arxiv.org/abs/2609.16778) · [official code](https://github.com/ALCOR-Lab-DIAG/MoE-JEPA)

A dual-branch detector for social-media images. The semantic branch is a frozen **V-JEPA 2 ViT-L** video encoder (the
image is fed as a static 4-frame clip); in its last three blocks the MLP becomes a **residual mixture of experts**: a
frozen shared expert (the pretrained MLP) plus the top 2 of 6 trained routed experts. The patch tokens are pooled by
**gated-attention MIL**, which can single out a few manipulated patches. The noise branch is a constrained **BayarConv**
high-pass layer and a small CNN; a learned per-image gate adds its feature to the semantic one. The head classifies
**real / fully synthetic / tampered** (SID-Set's three classes).

## Pipeline (official, reproduced exactly)

```
image ─ V-JEPA 2 video processor: short side 292 (bilinear), center crop 256, ImageNet normalization
      → repeated into a static 4-frame clip
      → V-JEPA 2 ViT-L (frozen); blocks 22-24: MLP = shared expert + top-2-of-6 routed experts
      → patch tokens → gated-attention MIL pooling → LayerNorm ──────────────┐
      → BayarConv → CNN → Linear(128, 1024) → LayerNorm → × gate α(semantic, noise) ─┤ +
      → Linear(1024, 3): real / fully synthetic / tampered
SynthScope P(fake) = P(fully synthetic) + P(tampered) = 1 − P(real)
```

## SynthScope adaptations

| choice | reason |
|---|---|
| `official_arch.py` = `BayarConv2d`, `NoiseStream`, `GatedAttentionPooling`, `Expert`, `MoEJEPALayer` from the official `model/model.py`, verbatim | They are the model; MIT, notice in the header. |
| `SynthScopeMoEJEPAModel` = the official `MoEJEPADeepfakeDetector` without Lightning, same attribute names | The released `moejepa_sidset.safetensors` loads key for key, unchanged (the engine now reads `.safetensors`). |
| `pretrained_prefixes` = the frozen V-JEPA 2 tensors and the shared experts | The checkpoint stores only the trained tensors (routers, routed experts, pooler, noise branch, gate, head); the rest is rebuilt from `facebook/vjepa2-vitl-fpc64-256`, as the official `load_moejepa` does. Every trained tensor loads strictly. |
| `forward` returns [real, logsumexp(fully synthetic, tampered)] | SynthScope is binary; softmax of these two logits gives exactly the official 1 − P(real). `forward_3way` keeps the official three logits. |
| `VJEPA2ImageTransform` calls the official `AutoVideoProcessor` | The pixels are the official ones exactly. |
| fp32 by default | The official test runs in bf16 autocast; both are reported below. |

## Fidelity check (identical to the official model)

The official `MoEJEPADeepfakeDetector` (`model/model.py` unchanged, built by the official `load_moejepa`; Lightning, which
the SynthScope environment does not have, stood in for by a bare `nn.Module`) and SynthScope's model on the released
checkpoint: **703/703 state-dict tensors identical**, the registry transform equal to the official `preprocess_image`
(max |diff| 0), and **identical 3-way logits** (max |diff| 0) on seven RRDataset and SynthWildX images. CPU, job 270454.

## Results

Paper metric (Table 1): three-way accuracy per class (the official ``test.py`` logs each class's recall as its
"Acc"), per-class F1, overall accuracy and macro F1, on the SID-Set test split (60,000 images, 20,000 per class).
`repro/official_evals/moejepa_sidset_eval.py` runs SynthScope's model, identical to the official one, with these
metrics, in bf16 autocast as the official test and in fp32 (job 270455).

| SID-Set test (accuracy / F1) | Paper | SynthScope_official (bf16) | SynthScope_official (fp32) |
|---|---|---|---|
| Real | 91.61 / 93.06 | 94.59 / 94.62 | 94.59 / 94.64 |
| Fully synthetic | 99.98 / 99.67 | 99.95 / 99.74 | 99.97 / 99.73 |
| Tampered | 95.12 / 93.50 | 94.58 / 94.74 | 94.58 / 94.76 |
| **Total** | 95.54 / 94.21 | **96.37 / 96.37** | 96.38 / 96.38 |

- **Reproduced**: the released checkpoint is 0.8 points above the paper's overall accuracy; the classes shift a little
  (real +3.0, tampered −0.5), as expected if the release is not exactly the checkpoint behind Table 1. The paper's
  "Total F1" (94.21) is not the mean of its own per-class F1s (95.41), so the F1 comparison is approximate.
- The paper's RRDataset table (Table 2, 84.11 overall) uses a different model, pre-trained on GenImage SD v1.4 +
  ImageNet and fine-tuned on RRDataset's training split, which is not released: **missing**.

Binary detection in SynthScope (P(fake) = 1 − P(real), threshold 0.5; balanced accuracy, AP, AUROC):

| Benchmark | Status | mAP | AUROC | Acc | Notes |
|---|---|---|---|---|---|
| SID-Set (fully synthetic / tampered vs real) | in-domain | 99.47 | 99.41 | 95.96 | fully synthetic 99.99 AP, tampered 98.95 |
| SynthWildX | reference | 83.81 | 80.35 | 73.66 | DALL-E 3 80.2, Firefly 66.4, Midjourney 74.4 acc |
| WildRF | reference | 77.06 | 73.50 | 68.19 | Facebook 71.3, Reddit 63.5, Twitter 69.9 |
| BFree-Online | reference | 91.79 | 85.22 | 57.99 | real 99.3, fake 16.7 at 0.5 |
| EvalGEN | reference | 92.77 | 86.43 | 70.85 | fake accuracy 90.8 (the EvalGEN metric); MSCOCO reals 50.9 |
| RealChain | reference | 71.74 | 73.22 | 66.39 | FLUX.1 88.7; image-to-image 46.9 |
| RRDataset | reference | 65.09 | 64.17 | 59.67 | original 72.4, transfer 55.8, re-digitized 50.8 (paper's own RRDataset model not released) |

Trained only on SID-Set, the released model is excellent in domain and moderate in the wild; it flags COCO photos
often (EvalGEN reals) and misses most re-digitized fakes. Results: `repro/bench_results/moejepa_*.json`.

## Usage

```bash
synthscope detect path/to/image.jpg -m moejepa
python repro/official_evals/moejepa_sidset_eval.py --root <SIDSet>          # the paper's 3-way SID-Set table
python benchmark/bench_eval.py --model moejepa --dataset sidset --tag moejepa_sidset
```

`--dataset sidset` downloads SID-Set's test split (one 36.5 GB zip from the SIDA authors' Google Drive; the train and
validation splits are on Hugging Face `saberzl/SID_Set`) and lays it out as two binary groups,
`full_synthetic/{0_real,1_fake}` and `tampered/{0_real,1_fake}`, sharing the 20,000 reals.

## License

The MoE-JEPA code and checkpoint are released under the MIT license. V-JEPA 2 (`facebook/vjepa2-vitl-fpc64-256`) is
MIT. SID-Set is CC BY 4.0.
