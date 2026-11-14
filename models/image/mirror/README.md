# MIRROR (`mirror`)

**MIRROR: Manifold Ideal Reference ReconstructOR for Generalizable AI-Generated Image Detection** (NeurIPS 2026) —
[paper (arXiv 2602.02222)](https://arxiv.org/abs/2602.02222) · [official code](https://github.com/handsome-rich/MIRROR)

> **Note (2026-11-14):** the official repository `github.com/handsome-rich/MIRROR` has returned 404 since 2026-11-12
> (no renamed copy found). The authors' Google Drive checkpoint links still work. SynthScope's integration does not
> depend on the repository: the model code it needs is vendored in `official_arch.py` (from commit 7f108fd).

MIRROR does not hunt for generator artifacts. It compares the image with what a real image "should" look like: a
memory bank of real-image prototypes rebuilds an *ideal reference* of the input's DINOv3 features by sparse attention,
and the gap between the input and its reference (plus how uncertain the retrieval was, plus the global CLS token)
gives the score. The released detector uses a DINOv3 ViT-H+/16 backbone with LoRA adapters; its memory bank holds the
code's seeded default initialization, not trained prototypes (see "Note on the released files" below). The training data
of the released checkpoint is unverified here; the paper's setting is in `repro/benchmark_specs/mirror.json`.

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

The released checkpoint nevertheless reproduces the README H+ numbers on 9 of the 10 held benchmarks (see Results), so it
is very likely the model behind those numbers. SynthScope reproduces the released checkpoint as it is. Scripts and
logs: `SynthScope_shared/mirror_provenance/`.

## Fidelity check (identical to the official code)

The official code (`mirror_ref` at commit 7f108fd, with its pinned transformers 4.57.1 and peft 0.21.2, in a separate
process) against SynthScope's full pipeline (raw file → `bench_eval._open_rgb` → registry transform → engine-built
model with `mirror.pth` loaded by the engine) on 24 real and fake images: PNG, JPEG, WebP and BMP; 128×128 to
2048×1365, square and non-square:

* state dict: **all 825 tensors identical** (sha256 of the bytes; the only difference is the block-prefix name),
* model inputs: identical on all 24 images,
* P(fake): **max |diff| 0.0, 0 flips**, in fp32 on CPU and on an RTX 3090, and under fp16 autocast on the RTX 3090.

The official fp32 and fp16-autocast outputs differ from each other by at most 8.5e-4 in P(fake) on these images
(0 flips). Runs: GPU job 279415 on commit b53e5ec (RTX 3090; log `repro/logs/mirrorpar_279415.out`), CPU logs
`repro/logs/mirrorpar_cpu_ours_4d7f84e.log` and `mirrorpar_cpu_*_20261106.log`.

Memory and speed (job 279415, RTX 3090, 224×224 input, peak allocated): fp32 (served) 3.42 GiB at batch 1 (25.7
img/s), 7.88 GiB at batch 32 (34.6 img/s); fp16 autocast 3.40 GiB at batch 1 (19.0 img/s), 6.59 GiB at batch 32
(90.1 img/s).

## Results

The paper's per-subset tables are for DINOv3-L. The released checkpoint is DINOv3-H+, and the only H+ numbers are
the 13 per-benchmark values in the official README ("13 standard + in-the-wild benchmarks", DINOv3-H+ column), so
those are the reproduction target. Metric: balanced accuracy at threshold 0.5, the unweighted mean over a
benchmark's subsets, as the official `inference.py` scores it (the README table's "B.Acc"). SynthScope numbers are the
released checkpoint run through SynthScope's `bench_eval` (SynthScope_official); MIRROR is not trained in SynthScope
(SynthScope_train "—").

| Benchmark | Paper (README H+) | SynthScope_official | SynthScope_train | Status |
|---|---:|---:|---:|---|
| **AIGCDetectBenchmark** (17 subsets, full data, 152,598 images) | 97.3 | **97.39** (mAP 99.49, AUROC 99.51) | — | **reproduced** (+0.09) |
| **GenImage** (8 subsets, full val, 100k images) | 99.8 | **99.78** (mAP 100.00, AUROC 100.00) | — | **reproduced** (-0.02); test release 99.79 |
| **UnivFakeDetect** (official code's 13-subset `UnivFD` list) | 92.4 | **92.44** (mAP 97.04, AUROC 97.24) | — | **reproduced** (+0.04); the paper's Table 7 list (12) gives 93.81 |
| **Synthbuster** (9 generators vs 5000 MSCOCO reals) | 99.2 | **99.28** (mAP 99.92, AUROC 99.96) | — | **reproduced** (+0.08); the served pipeline gives 99.43 |
| **EvalGEN** (5 generators vs MSCOCO reals) | 99.8 | **99.85** (mAP 100.00, AUROC 99.99) | — | **reproduced** (+0.05) |
| DRCT-2M | 93.0 | — | — | missing (not in SynthScope) |
| Chameleon | 94.6 | — | — | missing (available by request only) |
| **SynthWildX** (1713 of 2000 images still online) | 95.1 | **95.08** (mAP 97.71, AUROC 98.33) | — | **reproduced** (-0.02) |
| **WildRF** (official file set, no `.jfif`) | 97.8 | **97.88** (mAP 99.80, AUROC 99.74) | — | **reproduced** (+0.08); all files 97.86 |
| **AIGIBench** (SocialRF + CommunityAI) | 94.9 | **94.98** (mAP 98.06, AUROC 98.66) | — | **reproduced** (+0.08) |
| CO-SPY in-the-wild | 97.4 | — | — | missing (not held; `cospy` is the synthetic Co-Spy-Bench) |
| RRDataset (pooled folder) | 88.3 | 86.91 (mAP 93.86, AUROC 91.98: means of the 3 scenarios) | — | **short** by 1.39 (see below) |
| **BFree-Online** (807 of about 1400 images still online) | 97.6 | **97.40** (mAP 99.78, AUROC 99.50) | — | **reproduced** (-0.20) on the images still online |
| RealChain (7 generators, not in the paper) | — | 81.71 (mAP 91.76, AUROC 89.96) | — | reference |

Human-AIGI has no value in the README table and is not in SynthScope.

**Verdict.** SynthScope's MIRROR is identical to the official code (state dict, inputs and logits, see above), and
the released checkpoint reproduces the README's DINOv3-H+ numbers on **9 of the 10 benchmarks we hold**, each within
0.25 B.Acc. RRDataset is the exception, 1.39 short. DRCT-2M, Chameleon, CO-SPY in-the-wild and Human-AIGI could not
be run (no data), so the README's six-standard-benchmark average and its 16-column in-the-wild score cannot be
computed in full. On what we hold:

* five of the six standard benchmarks (no DRCT-2M): SynthScope 97.75, the README values for the same five 97.70;
* ten of the 16 in-the-wild columns (SynthWildX x3, WildRF x3, AIGIBench x2, RRDataset, BFree-Online; no Chameleon,
  no CO-SPY x5): SynthScope 95.32, the README values for the same ten columns 95.44. The README does not print the
  H+ aggregates; the six-benchmark average (96.92) and the 16-column score (96.00) are derived from its per-benchmark
  values. For DINOv3-L the paper prints 94.0 and 91.2 (Tables 1 and 2).

The aggregates use 99.28 for Synthbuster and the official `UnivFD` list for UFD.

### Protocol

* **Scoring.** Per subset: pred = argmax(logits), i.e. P(fake) > 0.5; B.Acc = (real acc + fake acc) / 2; the
  benchmark score is the unweighted mean over its subsets (the CSV `MEAN` row of the official code). bench_eval uses
  the same 0.5 threshold and the same per-subset balanced accuracy. A benchmark without an entry in the official
  `DATASET_CONFIGS` (RRDataset, B-Free, Chameleon) is scored by the official code as one pooled folder.
* **Synthbuster and EvalGEN ship fakes only.** As in the paper (Table 10 caption), each generator is scored against
  the MSCOCO val2017 reals (5000 images), as a balanced accuracy.
* **UnivFakeDetect.** Table 7 of the paper lists 12 subsets (SITD, SAN, CRN, IMLE, Guided, LDM x3, Glide x3, DALL-E)
  and gives 93.81 here. The official code's `UnivFD` list has 13 (the same plus deepfake)
  and gives 92.44, the README's 92.4, so that list is the headline. The code's `ojha` list (12) gives 95.03. Weak
  subsets: seeingdark 61.39 (fake acc 24.44) and deepfake 76.09.
* **Synthbuster headline = 99.28, the official extension-based rule.** The official `TestDataset` applies the
  format-aligned JPEG q96 step when the file *extension* is .png/.bmp/.tiff; the served SynthScope pipeline keys on
  the *decoded* format. They differ only for files whose extension does not match their content. The Synthbuster
  `firefly` subset is 1000 JPEGs named `.png`: the official rule re-encodes them (firefly B.Acc 94.29, fake acc 88.60)
  and the served pipeline does not (95.64, fake acc 91.30). That moves the Synthbuster mean from 99.43 to 99.28, and
  99.28 is the headline and the value used in the aggregates. A per-model "take the format from the extension when a
  path is known" option for `bench_eval` is a deferred follow-up (the image-loading code is shared with CuRe); it is
  not implemented. The other mismatch found by a pre-run scan of every file, 18 RRDataset PNG files named `.jpg`, has
  no effect (original: 98.06 under both rules).
* **WildRF.** 262 Twitter files are `.jfif` (197 of 341 reals, 65 of 342 fakes). The official loader never reads
  them (its extension list has no `.jfif`); `bench_eval` scores them. The headline uses the official file set, the
  other detectors in this repository use all files: 97.86 (twitter 98.54 against 98.60).
* **RRDataset** (`original`, `redigital`, `transfer`, about 8500 + 8500 images each) is scored pooled, as the official code
  does for a benchmark without a config entry. Pooled B.Acc and the mean of the three scenarios agree (86.91),
  because the per-scenario class counts are nearly equal (8500, 8499 and 8500 per class); the mAP and AUROC in the
  table are the mean over the three scenarios, not pooled.
* **Runs.** `benchmark/bench_eval.py --model mirror --bs 32`, fp32, the released checkpoint as `mirror.pth`,
  code at commit bbe6597 (0 uncommitted files, shown in every log header). The tables are built by
  `repro/mirror_task2_table.py` from `repro/bench_results/mirror_*.json`; logs
  `repro/logs/mirrorbench_<job>.out`.

| jobs | what | tags |
|---|---|---|
| 279513 | WildRF (all files, no `.jfif`, AMP), SynthWildX, BFree-Online, RealChain; the official `inference.py` on WildRF and SynthWildX | `mirror_wildrf`, `_nojfif`, `_amp`, `mirror_synthwildx`, `mirror_bfree_online`, `mirror_realchain` |
| 279527 | Synthbuster, with the extension-rule re-run of firefly | `mirror_synthbuster_coco`, `_firefly_extrule` |
| 279550 | UnivFakeDetect (8 diffusion sets + 5 CNNDetection sets) | `mirror_ufd_diffusion8`, `mirror_ufd_cnndet5` |
| 279551 | EvalGEN | `mirror_evalgen` |
| 279552, 279553 | GenImage full val (100k) and the test release | `mirror_genimage_val100k`, `_full` |
| 279554 | RRDataset (with the extension-rule re-run of `original`), AIGIBench | `mirror_rrdataset`, `_original_extrule`, `mirror_aigibench` |
| 279555 | AIGCDetectBenchmark | `mirror_aigcdetect` |

### Official code on the same files

The unmodified official `inference.py` (fp32, its default, with its pinned library versions) was run on WildRF (the
official file set) and SynthWildX in job 279513, on the same images. **It gives the same B.Acc as SynthScope on every
subset**: WildRF facebook 97.50, reddit 97.53, twitter 98.60 (mean 97.88); SynthWildX dalle3 97.44, firefly 90.21,
midjourney_v5 97.59 (mean 95.08). The CSVs are in `repro/bench_results/mirror_official_code/279513/`.

Two further official-code runs (job 279857; 279825 was a cancelled duplicate) confirm the other two numbers. Synthbuster
firefly against the MSCOCO reals gives B.Acc 94.29 (real 99.98, fake 88.60, AP 99.29, AUROC 99.63), equal to SynthScope
under the official extension rule (the Synthbuster mean is then 99.28). BFree-Online pooled (807 images) gives B.Acc
97.40 (real 98.14, fake 96.65, AP 99.78, AUROC 99.50), equal to SynthScope. The CSVs are in
`repro/bench_results/mirror_official_code/279857/`; `repro/mirror_task2_table.py` fills its "official code on our
data" column from them.

### Variant runs

* **AMP.** WildRF re-run with `use_amp: true` (CUDA fp16 autocast, the README quick-start's `--use_amp`): B.Acc,
  mAP, AUROC and every per-subset real and fake accuracy are identical to fp32 (97.86 / 99.63 / 99.65). So fp32 as
  the served precision does not change any reported metric (`mirror_wildrf_amp`).
* **Extension rule, `.jfif`.** See the protocol above (`mirror_synthbuster_coco_firefly_extrule`,
  `mirror_rrdataset_original_extrule`, `mirror_wildrf_nojfif`).

### Where it is weak

* AIGCDetect: cyclegan 83.65 (fake acc 67.75), whichfaceisreal 87.10, stargan 93.67; every diffusion subset is at
  least 97.35.
* RRDataset: original 98.06, transfer 88.38, **redigital 74.30** (fake acc 53.20, real acc 95.40). Pooled real acc is
  97.50 and pooled fake acc 76.32. This is not a scoring artefact (see above). The causes are unresolved: the
  paper's RRDataset copy or split may differ from ours, or the checkpoint behind the README value may differ from
  the released one. Every other benchmark we hold matches, so this is one open gap.
* RealChain (not in the paper): Flux1 96.60, Hunyuan3 90.90, QwenImage 90.50, SDv3.5 80.05, NanoBanana 78.75,
  Seedream4 69.25, i2i 65.90.
* The released H+ checkpoint is higher than the paper's DINOv3-L tables on most subsets but lower on some:
  UFD seeingdark 61.39 (L: 63.3), the UFD LDM and DALL-E subsets (97.65 against 98.6 to 99.0), and EvalGEN Flux
  (99.43 against 99.9).

## Caveats

* **Licence unclear.** The official README carries an MIT badge, but the repository has no LICENSE file (the badge's
  link returns 404). MIRROR is therefore not redistributed: not in the model zoo, not on the Space. `LICENSES.md`
  records it as "unclear".
* **The checkpoint embeds the gated DINOv3 ViT-H+/16 weights** (`facebook/dinov3-vith16plus-pretrain-lvd1689m`,
  DINOv3 License). Inference needs only `mirror.pth`; the gated download is needed only to train.
* **Precision.** SynthScope serves fp32, the official default. AMP changes P(fake) by at most 8.5e-4 on the parity
  images and changes no reported metric.
* **Memory and speed** (RTX 3090, 224x224): see above. About 3.4 GiB at batch 1 and 7.9 GiB at batch 32 in fp32, and
  34.6 img/s at batch 32; any 24 GB card is plenty. Building the model and loading the 3.45 GB checkpoint takes 8 to
  14 s (the parity job, which held the official and the SynthScope model in two processes, used 15.5 GiB of host
  RAM); on CPU one image takes about 0.5 s.
* **Not reproduced:** DRCT-2M, Chameleon, CO-SPY in-the-wild and Human-AIGI (no data in SynthScope), and the training
  (phase-1 memory-bank pretraining and phase-2 LoRA + head fine-tuning). `compute_loss` is the phase-2 loss only.

## Usage

```bash
synthscope detect photo.png --model mirror        # needs mirror.pth (= checkpoint-h-cur.pth) in $SYNTHSCOPE_CKPT_DIR
python benchmark/bench_eval.py --model mirror --dataset wildrf --root /data/WildRF --bs 32
python repro/mirror_task2_table.py                # the comparison tables above, from repro/bench_results/mirror_*.json
```

The input is resized (short side above 512 down to 512), center-cropped to 224 and, for PNG/BMP/TIFF sources,
JPEG-recompressed at quality 96, exactly as in the official test transform.
