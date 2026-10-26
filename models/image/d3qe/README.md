# D3QE (`d3qe`)

**D³QE: Learning Discrete Distribution Discrepancy-aware Quantization Error for Autoregressive-Generated Image
Detection** (ICCV 2025) — [paper (arXiv 2510.05891)](https://arxiv.org/abs/2510.05891) ·
[official code](https://github.com/Zhangyr2022/D3QE)

Visual autoregressive (AR) generators such as LlamaGen, VAR or Janus-Pro build an image from discrete codebook tokens.
D3QE passes the image through a frozen VQ tokenizer and reads the **quantization error** (quantized minus continuous
latent) with a transformer whose attention is biased by how often real and fake images use each codebook entry. It
fuses that with frozen CLIP features. The paper introduces **ARForensics** (7 AR generators) and reports 82.11 accuracy /
92.07 AP there, with a detector trained only on LlamaGen vs ImageNet.

## Pipeline (official, reproduced exactly)

```
image ─ Resize(256) on the short side (bilinear) → CenterCrop(256) → ToTensor            (no normalization)
      ├ LlamaGen VQ-16 tokenizer (frozen) on the [0, 1] image → z, z_q          residual = z_q - z   [B, 8, 16, 16]
      │     → Conv + BN + ReLU → D3AT (2 blocks, codebook-frequency attention bias) → pool → MLP      [B, 256]
      └ CenterCrop(224) + CLIP normalization → CLIP ViT-L/14 (frozen) → Linear                       [B, 256]
      → MLP(concat) → logit;  P(fake) = sigmoid(logit)
```

`d3qe.pth` is the official `model_epoch_best.pth` (Hugging Face `Yanran21/D3QE`) unchanged. It holds only the trained
part (81 tensors, 7.7M parameters, plus the codebook counts). As in the official code, the frozen OpenAI CLIP ViT-L/14
is downloaded to `~/.cache/clip` (or `$SYNTHSCOPE_CLIP_DIR`) and the LlamaGen VQ-16 tokenizer from Hugging Face
`FoundationVision/LlamaGen` (or `ModelConfig.vq_checkpoint` / `$SYNTHSCOPE_LLAMAGEN_VQ`).

## SynthScope adaptations

| choice | reason |
|---|---|
| `official/` = the official `networks/D3QE.py`, `vq_model.py` and `clip/model.py`, **byte-identical bodies** | `SynthScopeD3QEModel` subclasses the official `D3QE`, so the checkpoint loads key-for-key; MIT attribution in the headers. |
| `official/clip/clip.py` reduced to the official `load()` path D3QE uses (CPU, non-JIT) | The text tokenizer is not needed; `from pkg_resources import packaging`, which current setuptools no longer provides, became `import packaging.version`. |
| `pretrained_prefixes = ("clip_model.", "vq_model.")` | The checkpoint does not contain the frozen CLIP and VQ weights (the official `eval.py` falls back to `strict=False`). The engine accepts missing keys only under these prefixes; every trained tensor still loads strictly. |
| `compute_loss` = BCE on the logit, labels passed to `forward` | The official objective; the forward pass updates the codebook statistics during training as in the official code. ARForensics training is not reproduced here. |

Two properties of the released code are kept on purpose, because they decide the released scores:
* the VQ tokenizer sees the [0, 1] image: `forward` builds a [-1, 1] copy but encodes the unnormalized one;
* the codebook-frequency bias is **zero at inference**. `freq_log_counter` is a plain attribute, not a saved buffer,
  so a loaded model is below its 200-step warm-up and uses `freq_diff = 0`; the saved codebook counts do not change
  the released checkpoint's test scores.

## Fidelity check (identical to the official model)

The official `D3QE` (from the official repo, loaded as its `eval.py` does) and SynthScope's class with the same
checkpoint, on an A5000: **871/871 state-dict tensors identical** (including the frozen CLIP and VQ; max |diff| 0.0),
the registry transform equals the official `processing()` pixel for pixel, and **max |logit diff| 0.0 with 0 verdict
flips** over 56 ARForensics images (4 real + 4 fake per generator, sizes 256² to 1024², square and non-square).
Job 264794.

## Results

Paper metric: accuracy at 0.5 and AP (fake = positive), mean over subsets. Every subset used here is balanced, so
accuracy equals SynthScope's balanced accuracy.

| Benchmark | Paper (acc / AP) | SynthScope_official (acc / AP) | SynthScope_train | Status |
|---|---|---|---|---|
| **ARForensics** (7 AR generators, all 84,000 test images) | 82.11 / 92.07 | **81.74 / 92.22** (AUROC 91.91) | — | **reproduced** |
| **ForenSynths** (the paper's 7 GANs) | 83.73 / 92.23 | **83.45 / 92.23** (AUROC 90.65) | — | **reproduced** |
| **GenImage** (the paper's 6 subsets) | 78.61 / 89.60 | **79.44 / 90.94** (AUROC 90.49) | — | **reproduced** on the mean; per-subset gaps up to 6.8 pt |

Per generator (SynthScope vs paper, accuracy):

* **ARForensics** — LlamaGen 97.16 vs 97.19 · VAR 84.46 vs 85.33 · Infinity 63.06 vs 62.88 · Janus-Pro 92.39 vs 92.28 ·
  RAR 91.70 vs 91.69 · Switti 74.23 vs 75.31 · Open-MAGVIT2 69.21 vs 70.08 (every generator within 1.1 pt; AP within
  0.9 pt).
* **ForenSynths** — ProGAN 95.10 vs 95.20 · StyleGAN 76.21 vs 77.67 · StyleGAN2 75.08 vs 75.83 · BigGAN 86.07 vs 86.03 ·
  CycleGAN 83.27 vs 82.44 · StarGAN 73.64 vs 74.64 · GauGAN 94.76 vs 94.31 (every GAN within 1.5 pt).
* **GenImage** — ADM 63.67 vs 70.43 · GLIDE 91.13 vs 88.89 · Midjourney 63.80 vs 61.21 · SD-v1.4 82.21 vs 83.33 ·
  SD-v1.5 85.10 vs 83.37 · Wukong 90.73 vs 84.43. The mean matches, but single subsets move by up to 6.8 pt in both
  directions, so the paper's GenImage test sample probably differs from SynthScope's `genimage_full` test release.

**In the wild (new measurements, not in the paper).** D3QE is trained on LlamaGen vs ImageNet only and targets AR
generators; on recent photorealistic fakes it calls nearly everything real: SynthWildX 49.58, WildRF 53.03,
BFree-Online 47.03, RealChain 49.49, EvalGEN 62.21 (balanced; AUROC 83.94), RRDataset 54.95. Fake recall is under 15% on
17 of these 22 subsets. See [docs/WILD_BENCHMARKS.md](../../../docs/WILD_BENCHMARKS.md).

Runs: jobs 264795 (ARForensics, ForenSynths, GenImage) and 264796 (in-the-wild sets); results in
`repro/bench_results/d3qe_*.json`.

## Usage

```bash
synthscope detect photo.png --model d3qe          # needs d3qe.pth (= model_epoch_best.pth) in $SYNTHSCOPE_CKPT_DIR
python benchmark/bench_eval.py --model d3qe --dataset arforensics --root /data/ARForensics
python benchmark/bench_eval.py --model d3qe --dataset cnndetection --root /data/CNNDetection \
    --select progan,stylegan,stylegan2,biggan,cyclegan,stargan,gaugan
```
