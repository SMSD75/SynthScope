# MIRROR (`mirror`)

**MIRROR: Manifold Ideal Reference ReconstructOR for Generalizable AI-Generated Image Detection** (NeurIPS 2026) —
[paper (arXiv 2602.02222)](https://arxiv.org/abs/2602.02222) · [official code](https://github.com/handsome-rich/MIRROR)

MIRROR does not hunt for generator artifacts. It compares the image with what a real image "should" look like: a
memory bank of real-image prototypes rebuilds an *ideal reference* of the input's DINOv3 features by sparse attention,
and the gap between the input and its reference (plus how uncertain the retrieval was, plus the global CLS token)
gives the score. The released detector uses a DINOv3 ViT-H+/16 backbone with LoRA adapters.

## Pipeline (official, reproduced exactly)

```
image ─ short side > 512 ? resize short side to 512 (bicubic) : keep
      ─ center crop 224 (smaller images are zero-padded)
      ─ PNG / BMP / TIFF source ? JPEG round trip at quality 96 ("format-aligned inputs")
      ─ ToTensor → ImageNet normalization (inside the model)
      → DINOv3 ViT-H+/16 + LoRA (r 8, alpha 16, on q/k/v) → CLS + 200 tokens (196 patches + 4 registers)
      → memory bank (4096 prototypes, 8 heads, sparse top-128 attention) → reconstruction + attention map
      → head(perplexity MLP(max, entropy of the attention), residual MLP(mean(tokens − reconstruction)), CLS)
      → logits [B, 2];  P(fake) = softmax(logits)[:, 1]          (official inference.py)
```

Precision: SynthScope runs fp32, the official default: `inference.py`'s `--use_amp` is an opt-in flag, `scripts/eval.sh`
and `main_finetune.py --eval` leave it off, and the released checkpoint was trained with `use_amp=False`. The README's
quick-start command does pass `--use_amp`; `use_amp: true` in `conf/model/image/mirror.yaml` reproduces that (CUDA fp16
autocast at inference, never in training). On the parity images the two precisions differ by at most 8.5e-4 in P(fake)
with no flipped verdict; the benchmark reproduction checks one benchmark under AMP as well.

Weights: the phase-2 checkpoint `checkpoint-h-cur.pth` (authors' Google Drive, 3.45 GB) holds the full model, the
frozen DINOv3 weights included, and loads key-for-key as `mirror.pth`. Neither the gated DINOv3 download nor the
phase-1 file `mirror_phase1.pth` is needed for inference: the official inference loads the phase-1 memory bank first,
but the strict phase-2 load overwrites every one of its tensors.

## SynthScope adaptations

| choice | reason |
|---|---|
| `official_arch.py` = the official `models/mirror.py`, detector code unchanged | `SynthScopeMIRRORModel` subclasses the official `MIRROR_Detector`, so the checkpoint loads strictly; attribution and licence status in the header. |
| peft replaced by a small LoRA module with peft's layout and forward | peft is not a SynthScope dependency; the module names (`base_model.model.*.q_proj.base_layer`, `lora_A.default`, `lora_B.default`) and the forward (input cast, `B(A(x)) * alpha / r`, output cast) are peft 0.21.2's, and the smoke test checks it against peft when peft is installed. |
| backbone built from its config (`DINOV3_VITH16PLUS`), not `from_pretrained` | the checkpoint already holds the backbone weights; `backbone_path` still loads a pretrained HF-format DINOv3 (for training). |
| ViT block prefix renamed on load (`dino.layer` ↔ `dino.model.layer`) | transformers 5 moved the blocks; the official `load_detector_checkpoint` does the same rename. |
| JPEG q96 step decided by the decoded file format, not the file extension | a decoded PIL image has no file name; SynthScope's loaders record the format (`synthscope_src_format`). It equals the official extension test for every correctly named file. |
| `compute_loss` = cross-entropy with label smoothing 0.1 | the official phase-2 loss (timm `LabelSmoothingCrossEntropy`); phase-1 memory-bank training (`mmbank.py`) and the official training data pipeline are not reproduced. |

**Note on the released files.** Observations about the released checkpoint, for anyone comparing it with the paper's
two-phase description:
* The memory bank and its projections in `checkpoint-h-cur.pth` are bit-identical (max |diff| 0.0 on all 6 tensors) to
  the official code's default initialization: `seed_everything(0)` followed by `build_mirror(memory_path=None)`.
* The checkpoint's optimizer state covers 202 tensors: the 192 LoRA tensors and the 10 head tensors. The memory bank
  has `requires_grad=False` in phase 2, so it is not in it.
* The saved training arguments have `memory_path=None`. The current `main_finetune.py` refuses a new phase-2 run without
  `--memory_path`, so this run came from an earlier version of the script without that check.
* `mirror_phase1.pth` is an epoch-2 checkpoint (the README trains phase 1 for 100 epochs), and its memory bank is a
  different one (cosine similarity of the `memory` tensors −0.0007).

SynthScope reproduces the released checkpoint as it is. Scripts and logs: `SynthScope_shared/mirror_provenance/`.

## Fidelity check (identical to the official code)

The official code (`mirror_ref` at commit 7f108fd, with its pinned transformers 4.57.1 and peft 0.21.2, in a separate
process) against SynthScope's full pipeline (raw file → `bench_eval._open_rgb` → registry transform → engine-built
model with `mirror.pth` loaded by the engine) on 24 real and fake images: PNG, JPEG, WebP and BMP; 128×128 to
2048×1365, square and non-square:

* state dict: **all 825 tensors identical** (sha256 of the bytes; the only difference is the block-prefix name),
* model inputs: identical on all 24 images,
* P(fake): **max |diff| 0.0, 0 flips**, in fp32 on CPU and on an RTX 3090, and under fp16 autocast on the RTX 3090.

The official fp32 and fp16-autocast outputs differ from each other by at most 8.5e-4 in P(fake) on these images
(0 flips). Runs: job 279405 (GPU), `repro/logs/mirrorpar_cpu_*_20261106.log` (CPU).

Memory and speed (RTX 3090, 224×224 input, the parity job): fp16 autocast 3.40 GiB peak at batch 1 (19.3 img/s),
6.59 GiB at batch 32 (90.1 img/s); fp32 3.42 GiB at batch 1 (25.7 img/s), 7.88 GiB at batch 32 (34.4 img/s).

## Results

Pending: the paper's benchmarks are reproduced in a follow-up step.
