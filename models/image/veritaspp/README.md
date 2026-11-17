# Veritas++ (`veritaspp`)

**Veritas++** (Tan et al., arXiv 2607.27113, July 2026) — [paper](https://arxiv.org/abs/2607.27113) ·
[official code](https://github.com/EricTan7/VeritasPP) (Apache-2.0) ·
[weights](https://huggingface.co/EricTan7/VeritasPP) (Apache-2.0)

A multimodal-LLM detector. The released model is Qwen3-VL-8B fine-tuned with the authors' ms-swift fork (the released
`args.json` records the last stage, a GKD distillation run) to look at an image, write a human-like rationale with self-checks, and end with a verdict in
`<answer> real </answer>` or `<answer> fake </answer>`. It is the first detector of SynthScope's **VLM kind**
(`models/vlm/vlm.py`, a family-adapter design: Veritas++ uses the Qwen-VL family), which later VLM detectors reuse.
Still images only (no video).

**Status.** Inference only, with the official weights (no training path in SynthScope: the official training runs in
the authors' ms-swift fork). Not on the Space (an 8B model does not run on the CPU Space) and not in the model
zoo (a 17.5 GB folder). Fidelity against a direct transformers call with the official request: 15 / 15 identical greedy
tokens and verdicts, max |logit-gap diff| 0.0 ([Fidelity check](#fidelity-check)). Benchmark results on 15 benchmarks, each subsampled
and scored on identical images as 27 other detectors ([Results](#results)): it is among the best on in-the-wild and recent
generators (RRDataset and RealChain first, DiTFake second, EvalGEN third) and weak on older GAN and early-diffusion sets
(ForenSynths 25th of 28). Paper comparison: RealChain 82.74 vs 82.3 (a 1,008-image subset), WildRF 95.65 vs 96.4 (full data;
96.46 with the authors' sampled decoding); EvalGEN is **not** reproduced on full data (a 500-fake subset: 98.20 on fakes only
vs 99.5, 99.10 balanced).

## Pipeline (official request, reproduced)

The official inference is a vLLM OpenAI server (`self_scripts/deploy/deploy_model.sh`: bf16, tensor-parallel 4) queried
by `self_scripts/infer/infer_vllm_single.py`, one image per request (the file's bytes, base64). vLLM is not in the
SynthScope environment, so the same request goes through transformers (`Qwen3VLForConditionalGeneration` + the model's
own `Qwen3VLProcessor`), with vLLM's image loading reproduced:

```
image file ─ decoded without an RGB conversion (media.decode_image_raw)
  → vLLM's loading (vLLM 0.31.0 ImageMediaIO.load_bytes, ported in models/vlm/image.py):
      ImageOps.exif_transpose (pixels upright as displayed), then RGB passes, RGBA is composited on white, other modes
      with transparency (LA, PA, palette tRNS) go through RGBA onto white, everything else is convert("RGB")
  → Qwen3-VL processor at its defaults: resized only to multiples of 32 px within 65,536 - 16,777,216 pixels
    (native resolution for almost every benchmark image), normalized, 16x16 patches, 2x2 merge
  → chat request: system  "You are an expert visual analyst. Please think about the question as if you were a human
                           pondering deeply. It’s encouraged to include self-reflection or verification in the
                           reasoning process. Then, give a final verdict within <answer> </answer> tags."
                  user    [image, "Is this image real or fake?"]
                  chat_template_kwargs {"enable_thinking": False}
  → generate (SynthScope: greedy; the authors' script: temperature 0.7, with top-p 0.8 / top-k 20 from the released generation_config.json)
  → "...rationale (about 400-650 tokens)... <answer> real </answer>"
```

The prompts are copied verbatim from `infer_vllm_single.py` @ 9a1f00b (the apostrophe in "It’s" is U+2019 there too);
`tests/test_veritaspp_smoke.py` compares them with the official file. The release's chat template does not use
`enable_thinking`, so that flag changes nothing; it is passed because the official request passes it.

The image loading equals vLLM 0.31.0's own `ImageMediaIO.load_bytes` on 16 synthetic files that cover every branch
(the eight EXIF orientations, RGBA, LA, palette with and without tRNS, L, CMYK, RGB, RGBA with EXIF):
`repro/vlm_image_vllm_parity.py`, run in a vLLM environment, records 16/16 identical (`repro/vlm_image_vllm_parity.json`).
The authors install the latest vLLM (`vllm>=0.5.1 -U`), so 0.31.0 is the reference. On the held benchmarks this
matters for transparent PNGs only: none of their JPEGs has an EXIF rotation (orientation 2-8; checked over WildRF,
SynthWildX, BFreeOnline, RealChain, RRDataset and AIGIBench), while WildRF has RGBA PNGs (PIL's plain `convert("RGB")`
would show the model the colours under the transparent pixels; vLLM shows white).

**Score.** P(fake) = sigmoid(l(Ġfake) - l(Ġreal)) at the verdict token: the first token after `<answer>` (and before
`</answer>`) that spells fake or real, compared with its same-spelling counterpart (`Ġreal` ↔ `Ġfake`, `ĠReal` ↔ `ĠFake`, ...). `logit` is that
gap, so benchmarks can rank by it. The logits are the raw LM-head outputs of the step that produced the verdict token
(read with a forward hook; identical to transformers' `output_logits`). Because the model has already argued its way to
an answer, the score is near 0 or 1, and the paper reports accuracy: the primary metric is the accuracy of the parsed
verdict, with AUROC / AP as extras. No verdict token (no `<answer>` generated, e.g. a truncated output, or no fake / real
word inside the answer): P(fake) = 0.5, logit 0, `score_source: "unparsed"` (ruling U1). Scoring holds a lock: the
logit recorder hooks the model's shared LM head, so concurrent requests are served one after the other.

**Verdict.** The official repo has no parser or metric script. The verdict is parsed with the rule of the authors'
earlier Veritas (v1) `self_scripts/infer/cal_metrics.py` (vendored in `models/vlm/parsers.py`): the first
`<answer>(.*?)</answer>`, lower-cased, "real" checked before "fake". The official script drops an unparsed answer
from the accuracy; SynthScope reports it as `verdict: "unparsed"` (ruling U1: counted as wrong for accuracy; its score
is the verdict token's gap when one was generated, e.g. a truncated `<answer> fake`, else 0.5).

Weights: `EricTan7/VeritasPP` @ `22ba6c6b9f623755c1a68a9816b83a0540039829` (4 safetensors shards, 750 tensors, 17.5 GB
bf16, plus the processor and tokenizer files), unchanged, as the folder `$SYNTHSCOPE_CKPT_DIR/veritaspp/`:

```bash
hf download EricTan7/VeritasPP --revision 22ba6c6b9f623755c1a68a9816b83a0540039829 --local-dir $SYNTHSCOPE_CKPT_DIR/veritaspp
```

```python
import synthscope
det = synthscope.load("veritaspp")             # or checkpoint="/path/to/VeritasPP-folder"
r = det.predict("photo.jpg")                   # fake_probability, logit, verdict, rationale, response, ...
```

## SynthScope adaptations

| choice | reason |
|---|---|
| transformers (`Qwen3VLForConditionalGeneration`, bf16, SDPA) instead of the vLLM server | vLLM is not in the SynthScope environment. The prompts, chat template and processor are the official ones and the image loading is vLLM's (above); the generated text can still differ from a vLLM run by kernel-level numerics. |
| **greedy decoding** (`do_sample=False`) by default | the authors' inference script (`infer_vllm_single.py`) samples at temperature 0.7 (the paper's text states no sampling temperature; top-p 0.8 and top-k 20 come from the released `generation_config.json`), so its verdicts are not deterministic. The sampled setting stays available (`Engine.vlm_score(..., decoding="official", seed=...)`); the fidelity check below compares the two. |
| `max_new_tokens=2048` | vLLM's default is the rest of the context. The rationales are 400-650 tokens; the bound only stops a run-away generation (it is then reported, `finished: false`, and the answer is unparsed). |
| the Veritas v1 parser; unparsed answers kept and flagged | Veritas++ releases no parser (see Verdict above). |
| P(fake) from the verdict-token logits | the official output is text only; the logit gap is the continuous score that agrees with the text (under greedy decoding the emitted token is the argmax, so the sign of the logit is the verdict). |
| no pixel cap (`max_pixels=None`) | the official processor default (16.8 MP) is used; see Memory below. |
| inference only, still images only, one image per call | no `trainer.py` dispatch (the official training in the ms-swift fork is not reproduced); the engine, the API and the web app refuse videos for this detector (a rationale per frame would take minutes per video); the web app scores it in a worker thread and caps `/api/score_batch` at 20 images. |

## Fidelity check

`repro/veritaspp_fidelity.py` (`repro/veritaspp_fidelity.sbatch`, job 292005 on ivi-cn029, commit bd4d8d2, RTX A5000,
torch 2.11, transformers 5.5.4, bf16, SDPA; results `repro/veritaspp_fidelity.json`, log
`repro/logs/veritaspp_fidelity_292005.out`) compares SynthScope's engine path (`Engine._get_model` + `Engine.vlm_score`
on the file decoded as the API decodes it) with an independent reference: a direct transformers `generate` with the
official prompts (read from `infer_vllm_single.py`), the released processor and vLLM's image loading written out.
The job first checked that every file of the folder came from the pinned revision 22ba6c6b (hf download metadata; the
shard etags equal `SHA256SUMS`) and that all 15 files pass `sha256sum -c`.

Images: 15 (8 real, 7 fake) from GenImage val (the gate check's 3 real + 3 fake, 256x256 to
1024x1024), SynthWildX (3 real + 4 fake, 662x1000 to 2048x2048) and WildRF (2 real RGBA PNGs, transparency composited
on white).

| check | result |
|---|---|
| (a) strict load | `from_pretrained` loading info empty (missing 0, unexpected 0, mismatched 0); 750 / 750 tensors of the four shards equal to the loaded parameters (`torch.equal`), no other parameter |
| request | the engine's rendered prompt == the reference's |
| (b) greedy tokens | **15 / 15 identical** to the reference (341-653 new tokens each) |
| (b) parsed verdict | **15 / 15 identical**; verdict step identical on all 15; 0 unparsed |
| (c) verdict logit gap | **max \|diff\| 0.0** (engine hook vs reference `output_logits`) |
| accuracy, greedy | 15 / 15 correct; P(fake) 0.00016-0.00063 on the 8 reals, 0.99945-0.99997 on the 7 fakes |
| (d) the authors' sampling setting (T 0.7, top-p 0.8, top-k 20, seed 0) | 15 / 15 correct; verdict agreement with greedy **15 / 15** |
| `predict_image` wrapper | same P(fake) and verdict as `vlm_score` |

## Speed and memory (RTX A5000, bf16, one image at a time)

| | measured |
|---|---|
| weights resident | 17.54 GB; engine load 3.9 s (node-local SSD, files already in the page cache from the reference run) |
| 15 fidelity images (up to 2048x2048, about 4.2k visual tokens) | **17.35 s per image (0.0576 images/s)**, 12.2-21.8 s, 29.2 generated tokens/s, 341-653 tokens per answer; peak allocated **18.91 GB** (at 2048x2048) |
| memory probe, WildRF 2448x3264 (8.0 MP) | 22.8 s, 7,826 prompt tokens, peak 20.12 GB |
| memory probe, WildRF 4096x4096 (16.8 MP, the processor's cap) | 37.8 s, 16,458 prompt tokens, peak **22.96 GB**, no OOM |
| memory probe, WildRF 6936x9248 (64 MP; the processor resizes it to 3520x4704) | 38.8 s, 16,244 prompt tokens, peak 22.89 GB, no OOM |

Nothing ran out of memory on the 24 GB card, so `max_pixels` stays at the processor's default (16,777,216 pixels): no
cap and no deviation from the official preprocessing. The cost grows with the image: about 17 s per image up to 4 MP,
about 38 s at the 16.8 MP cap.

One probe image is a model error, not a fidelity issue: the WildRF fake `reddit/1_fake/000325_0325.jpg` (4096x4096) was
called "real" (logit -6.375). The probe images are not part of the parity comparison (engine path only).

## Results

**Inference only, with the official weights. No SynthScope training. Veritas++ is not a model SynthScope trained, so the
comparison below is a robustness and generalisation comparison, not a reproduction of a training recipe.**

### Protocol (subsampled; identical images for all 28 detectors)

A full pass costs about 17-19 s per image, so every benchmark is **subsampled**: per benchmark, `bench_eval --limit L` with
`L = ceil(500 / number of generators)`: the first L sorted reals and the first L sorted fakes of every generator
(`bench_eval.discover` order; the lists with every file's sha1 are `repro/vlm_bench/subsets/<benchmark>.json`). The same
files go to Veritas++ and to the 27 comparison detectors (the field-report detectors, scored with their usual `bench_eval`
settings; staged copies are verified sha1-identical to the originals). 14,937 images are scored per detector (17,935 with the JPEG probe
and the sampled run). Decoding is **greedy**. An answer without a verdict is **counted as wrong** for accuracy (ruling U1;
the authors' script drops it, and the paper-comparison rows below follow the authors) and scores 0.5 for AUROC. There are
**10 unparsed answers in 17,935 images**, all truncations at the 2,048-token limit (repetition loops): GenImage 1, UFD 3
(and 3 more in the JPEG95 run), RRDataset 2, SynthWildX 1. vLLM's unbounded default might parse some of them.

`acc` is `bench_eval`'s balanced accuracy (mean over generators of the mean of real and fake accuracy); AUROC and AP come
from the verdict-token logit gap. Rank = 1 + the number of the 27 comparators that are strictly better. "Best publicly
licensed" = the best comparator that is not in `internal_only` in `repro/vlm_bench/spec.json` (LTF, MIRROR, VINA, PROBE and
PROBE-ResNet have no clear licence and are not redistributed or served; the best overall detector is MIRROR, VINA or PROBE
on 14 of the 15 benchmarks). Ties for best are broken by the alphabetically first detector key, and the tied keys are
listed in `comparison.json`. Every number is in `repro/vlm_bench/comparison.json` (built by `repro/vlm_bench/compare.py` from
`repro/vlm_bench/results/`; logs `repro/logs/vpp_*.out`).

### Accuracy (balanced, %)

| benchmark (images per class and generator L) | Veritas++ acc | best overall | best publicly licensed | median of 27 | rank of 28 | unparsed | mean fake / real acc |
|---|---:|---|---|---:|---:|---:|---:|
| GenImage val (100k) (L=63) | **76.88** | 99.50 (`mirror`) | 96.03 (`omniaid_v1`) | 82.14 | 17 | 1 | 54.2 / 99.6 |
| ForenSynths (13 generators) (L=39) | **63.51** | 90.34 (`fat_former`) | 90.34 (`fat_former`) | 79.29 | 25 | 0 | 37.7 / 89.3 |
| UniversalFakeDetect (8 diffusion sets) (L=63) | **73.12** | 97.92 (`mirror`) | 95.83 (`npr`) | 87.40 | 25 | 3 | 49.2 / 97.0 |
| AIGCDetectBenchmark (17) (L=30) | **78.04** | 96.86 (`mirror`) | 93.73 (`omniaid_v1`) | 83.63 | 19 | 0 | 61.0 / 95.1 |
| Synthbuster (MSCOCO reals) (L=56) | **89.98** | 99.31 (`probe`) | 96.63 (`omniaid_v1`) | 87.80 | 13 | 0 | 80.0 / 100.0 |
| EvalGEN (L=100) | **99.10** | 99.90 (`mirror`) | 98.40 (`cure`) | 83.40 | 3 | 0 | 98.2 / 100.0 |
| SynthWildX (L=167) | **92.11** | 97.70 (`probe`) | 96.31 (`bfree`) | 61.28 | 6 | 1 | 93.8 / 90.4 |
| WildRF (L=167) | **95.64** | 97.47 (`mirror`) | 96.24 (`cure`) | 67.16 | 4 | 0 | 96.8 / 94.5 |
| AIGIBench (25 subsets, full) (L=20) | **82.40** | 89.00 (`mirror`) | 87.80 (`cure` (+1 tied)) | 79.20 | 11 | 0 | 69.6 / 95.2 |
| RRDataset (L=167) | **89.92** | 88.72 (`vina`) | 76.84 (`cure`) | 60.28 | 1 | 2 | 82.0 / 97.8 |
| BFree-Online (L=500) | **79.35** | 97.67 (`mirror`) | 94.93 (`dda`) | 58.03 | 8 | 0 | 70.6 / 88.1 |
| Co-Spy-Bench (110 subsets) (L=5) | **93.82** | 99.73 (`mirror`) | 98.18 (`safe`) | 89.27 | 7 | 0 | 91.6 / 96.0 |
| DiTFake (L=167) | **99.50** | 100.00 (`mirror`) | 99.10 (`safe`) | 94.31 | 2 | 0 | 99.2 / 99.8 |
| ARForensics (L=72) | **74.40** | 99.50 (`mirror`) | 97.72 (`omniaid_v1`) | 83.04 | 19 | 0 | 49.4 / 99.4 |
| RealChain (L=72) | **82.74** | 80.75 (`mirror` (+1 tied)) | 78.17 (`cure`) | 61.71 | 1 | 0 | 69.4 / 96.0 |

The last column is the mean over generators of the fake accuracy and the real accuracy (the "unparsed" column counts the
answers without a verdict). Two sets share one real pool across generators (UFD, Synthbuster): the per-generator real accuracy
is then the same real images counted once per generator; the committed `acc_pooled` in the result JSONs counts them that
way, while the headline `acc` is unaffected.

### AUROC (%)

| benchmark | Veritas++ mAUROC | best overall | best publicly licensed | median of 27 | rank of 28 |
|---|---:|---|---|---:|---:|
| GenImage val (100k) | **90.17** | 100.00 (`mirror`) | 99.83 (`effort`) | 94.25 | 18 |
| ForenSynths (13 generators) | **74.82** | 98.89 (`vina`) | 98.18 (`omniaid_genimage`) | 89.79 | 25 |
| UniversalFakeDetect (8 diffusion sets) | **84.94** | 99.97 (`mirror`) | 99.51 (`omniaid_v1`) | 96.97 | 26 |
| AIGCDetectBenchmark (17) | **89.11** | 99.50 (`ltf`) | 98.64 (`omniaid_genimage`) | 93.92 | 20 |
| Synthbuster (MSCOCO reals) | **98.43** | 100.00 (`mirror` (+1 tied)) | 99.99 (`pgc`) | 96.52 | 12 |
| EvalGEN | **99.95** | 100.00 (`mirror` (+1 tied)) | 99.70 (`dda`) | 95.59 | 3 |
| SynthWildX | **97.34** | 99.55 (`probe`) | 98.96 (`bfree`) | 72.41 | 5 |
| WildRF | **99.25** | 99.79 (`mirror`) | 99.30 (`cure`) | 72.95 | 3 |
| AIGIBench (25 subsets, full) | **90.53** | 98.78 (`mirror`) | 97.54 (`bfree`) | 90.48 | 14 |
| RRDataset | **91.56** | 95.60 (`vina`) | 87.15 (`dinov3`) | 64.62 | 3 |
| BFree-Online | **86.32** | 99.50 (`mirror`) | 98.82 (`dda`) | 76.52 | 11 |
| Co-Spy-Bench (110 subsets) | **98.00** | 100.00 (`mirror`) | 99.85 (`safe`) | 96.47 | 10 |
| DiTFake | **99.90** | 100.00 (`mirror` (+1 tied)) | 100.00 (`safe`) | 99.28 | 5 |
| ARForensics | **84.49** | 99.76 (`mirror`) | 99.70 (`effort`) | 96.12 | 22 |
| RealChain | **85.08** | 89.81 (`omniaid_v1`) | 89.81 (`omniaid_v1`) | 74.21 | 6 |


### What the numbers show (computed from `comparison.json`, `vlm_balance`)

- **Strong on in-the-wild and recent generators.** First among 28 on RRDataset (89.92) and RealChain (82.74), second on
  DiTFake (99.50), third on EvalGEN (99.10), fourth on WildRF (95.64), sixth on SynthWildX (92.11). Against the best publicly
  licensed detector it is ahead on RRDataset (+13.1), RealChain (+4.6), EvalGEN (+0.7) and DiTFake (+0.4), and behind on the
  other 11 benchmarks.
- **Weak on older GAN and early-diffusion sets and on autoregressive generators.** ForenSynths 63.51 (25th), UFD 73.12 (25th),
  ARForensics 74.40 (19th), GenImage 76.88 (17th), AIGCDetect 78.04 (19th). On these the real accuracy stays high (ForenSynths
  89.3, UFD 97.0, ARForensics 99.4, GenImage 99.6, mean over generators) while the fake accuracy is low (37.7, 49.2, 49.4, 54.2):
  fake accuracy is below 50% on 9 of 13 ForenSynths generators (StarGAN 30.8, StyleGAN2 7.7, SeeingDark 0.0), 4 of 8 GenImage
  generators (BigGAN 15.9, ADM 19.1, VQDM 25.4, GLIDE 31.8), 4 of 7 ARForensics generators (VAR 8.3, Open-MAGVIT2 12.5, RAR 15.3,
  LlamaGen 23.6) and 5 of 8 UFD sets. It calls the unfamiliar fake "real" rather than raising false alarms; its AUROC is
  higher than its accuracy on every one of these sets (the logit ranks better than the verdict separates).
- **Weak spots inside strong sets:** RealChain Seedream4 (fake 31.9) and image-to-image (22.2); RRDataset's re-digitised set
  (47.9); EvalGEN NOVA (91.0, the other four 100); Synthbuster Firefly (42.9).
- **Costs about 18 s per image** (16.8-20.6 s per benchmark on an A5000 or A6000; 488-540 generated tokens per image on average), against
  milliseconds for the others.

### Robustness: JPEG quality 95 on every image

All images (reals and fakes) re-encoded at quality 95 before scoring, same subsets and the same transform for the comparators:

| subset | Veritas++ acc clean -> JPEG95 | AUROC change | median of the 27: acc change / AUROC change |
|---|---|---:|---:|
| DiTFake | 99.50 -> 99.60 (**+0.10**, immune) | +0.06 | -2.00 / -1.14 |
| UFD (8 diffusion sets) | 73.12 -> 67.86 (**-5.26**) | -5.68 | -2.08 / -1.59 |

On UFD-8 Veritas++ loses more accuracy than the median detector, mostly on fakes (fake accuracy 49.2 -> 38.5, real accuracy
97.0 -> 97.2; all 8 sets are then below 50% on fakes); on DiTFake it is unaffected. Two sets only; not a general
robustness result.

### Greedy decoding vs the authors' sampled setting (WildRF subset, 988 images)

| | accuracy | AUROC | unparsed |
|---|---:|---:|---:|
| greedy (the matrix setting) | 95.64 | 99.25 | 0 |
| sampled (T 0.7, top-p 0.8, top-k 20, seed 0 per image) | 96.45 | 99.49 | 0 |

The verdicts agree on 96.96% of the images; the difference is +0.81 accuracy, one run, one seed.

### Comparison with the paper (Acc., VERITAS++ row; arXiv 2607.27113 v1)

The authors' metric is accuracy over all images with unparsed answers dropped; there were 0 unparsed in these rows, so
dropping and counting them wrong agree. The paper states no sampling temperature (T 0.7 is the authors' inference script's).

| benchmark (paper table) | paper | SynthScope (greedy) | difference | status |
|---|---:|---|---:|---|
| WildRF (III), **full data**, 2,503 images | 96.4 | **95.65** (subset of 988: 95.65; sampled, the authors' setting, subset: **96.46**) | -0.75 (sampled +0.06) | close; within 1 point, on the full test set |
| RealChain (III), 1,008-image subset (7% of the 14,000) | 82.3 | **82.74** | +0.44 | close, but a subset: not a full-data reproduction |
| EvalGEN (II), 500-fake subset (of 55,298 fakes) | 99.5 | **98.20** accuracy on the fakes only; **99.10** balanced (fakes + MSCOCO reals) | -1.30 fakes only; -0.40 balanced | **not reproduced** on full data (about 55k images at 17 s is not affordable); within about 1 point on the subset |

On the EvalGEN metric: the paper does not say how its EvalGen column is scored; scoring the fakes only is **our inference**,
from DDA's paper row (97.2), which equals DDA's fakes-only EvalGEN accuracy. The fakes-only number is therefore the like-for-like
guess and the balanced one the other reading; NOVA (91.0) is the only weak generator (100 fakes, so about 3 points of
subsampling noise). Never read the EvalGEN row as reproduced. The paper's RealChain and WildRF test sets are assumed to be the
released ones (RealChain 7 generators; WildRF facebook / reddit / twitter test sets); the paper does not give the file lists.

Cost: the WildRF full run took about 13.8 GPU-hours in total (about 8.7 hours beyond the subset; 19.8 s per image), above
the plan's guideline of 12 GPU-hours. That was a judgement call: WildRF is the one paper benchmark where a full-data comparison was
affordable, and the extra hours are what makes the -0.75 a full-data number. The full run is checked against the subset run:
all 988 subset images score identically (logit, verdict, tokens) in both, and the full data is sha1-identical to the cn029
original.

### Reproduce

```bash
hf download EricTan7/VeritasPP --revision 22ba6c6b9f623755c1a68a9816b83a0540039829 --local-dir $SYNTHSCOPE_CKPT_DIR/veritaspp
python benchmark/bench_eval.py --model veritaspp --benchmark pergen --data <subset-root> --limit <L> --vlm-cache <dir> --tag veritaspp_<bench>
python repro/vlm_bench/compare.py          # regenerates repro/vlm_bench/comparison.json from the result JSONs
```

`--vlm-cache` keeps every scored image as a JSON line, so an interrupted run resumes; the cache folder is keyed by a hash of the
weights revision, `max_new_tokens`, the prompts and the VLM code version (`identity.json` in the folder says what it
stands for), so a cache reused after any of them changes starts empty. `repro/vlm_bench/run_units.py` runs the whole matrix
(`vpp_ivi.sbatch` on the IVI cluster).

## Availability

- Not on the Hugging Face Space: an 8B model needs a GPU (17.5 GB of weights in bf16; 18.9 GB peak at 2048x2048, 23.0 GB at the
  16.8 MP processor cap).
- Not in the model zoo (download them from the authors' repo at the pinned revision with the `hf download` command above): the weights are Apache-2.0, so redistribution is allowed, but the release is a 17.5 GB
  multi-file folder that the zoo's one-file-per-key layout does not hold (`NEVER_REDISTRIBUTE` in
  `synthscope/hub.py`); `synthscope.load("veritaspp")` without the folder prints the download command above.

## Licence

Code: Apache-2.0 (EricTan7/VeritasPP, an ms-swift fork). Weights: Apache-2.0 (HF model card; ModelScope
`EricTanh/VeritasPP`). Base model Qwen3-VL-8B: Apache-2.0. The parser rule is from EricTan7/Veritas (Apache-2.0). See
[LICENSES.md](../../../LICENSES.md).
