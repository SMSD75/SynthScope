# Align, Don't Blend: VLM branch (`adb_vlm`)

**Paper:** Chen et al., *Align, Don't Blend: Orthogonal Supervision Disentanglement As the Key to Generalizable
AI-Generated Image Detection*, ACM MM 2026 ([arXiv 2512.06746](https://arxiv.org/abs/2512.06746)).
**Code:** [GzZning/Align-Dont-Blend](https://github.com/GzZning/Align-Dont-Blend) @ `f4b4794` (Apache-2.0 since 2026-11-20,
checked at `5d83e40`).
**Weights:** LoRA adapter [Gaffeyzz/Align-Dont-Blend-VLM](https://huggingface.co/Gaffeyzz/Align-Dont-Blend-VLM) @
`49975ccd` (Apache-2.0: the card's licence tag, added at `ff7973df` with identical adapter files), on the base model [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
@ `cc594898` (Apache-2.0).

**Status.** Inference only, with the official adapter. This is **the VLM branch alone**. The paper's detector ORs it
with a pixel-artifact expert, the released DDA checkpoint (the authors confirmed this on 2026-11-20). That full method is
the detector [`adb_full`](../adb_full/README.md), which carries the paper comparison. The paper prints no VLM-only number
for the released adapter. It is not on the Space, because a 7B VLM needs a GPU, and the adapter is not mirrored to the
model zoo ([Licence](#licence)).

- **Fidelity** against the official ms-swift 4.4.2 inference (pt backend): identical text, tokens and verdicts on 16 / 16
  images ([Fidelity](#fidelity)).
- **15 benchmarks**, each subsampled and scored on the same images as Veritas++ and 27 other detectors
  ([Results](#results)).
  - In the wild it ranks 5th to 8th of 28: WildRF, EvalGEN, BFree-Online, SynthWildX, RRDataset, RealChain.
  - On older GAN and diffusion sets it is near chance: UFD-8 52.88 (last), ForenSynths 55.72.
  - Mean of the 15: 76.58 acc / 84.11 AUROC (Veritas++ 84.70 / 91.33).
- **Full WildRF** (2,503 images): the VLM branch alone gives 93.98 balanced accuracy. The paper's full method prints
  92.9; `adb_full` (VLM OR DDA) gives 91.22 on the same images.

## Method
The paper pairs two detectors and ORs their verdicts: a vision-language model trained to judge semantic plausibility,
and a pixel-artifact expert. This detector is the **VLM branch**: a LoRA (r 16, alpha 32, 196 modules on the language
model's attention and MLP projections) on Qwen2.5-VL-7B-Instruct, trained with DPO on 5,000 real Unsplash images and
5,000 semantically implausible Echo-4o images. It writes a short `<rationale>` and then answers inside `<answer>`:
"This is an authentic image." or "This is an AI-generated image.". The expert is the released DDA checkpoint
([`dda`](../dda/README.md)); the two together, with the official decision rule, are [`adb_full`](../adb_full/README.md).

## Official inference, as SynthScope reproduces it
`predict.py` writes a dataset file and runs ms-swift (`swift infer`; the vLLM backend by default, or the pt backend).
SynthScope sends the same request through transformers + peft (`models/image/adb_vlm/adb_vlm.py`, shared VLM kind
`models/vlm/`). The fidelity check compares it with the **pt backend**; the vLLM backend was not compared, and its
kernels can give a different greedy text.

| step | official (`predict.py` -> ms-swift 4.4.2) | SynthScope |
|---|---|---|
| model | base + `--adapters` (LoRA, unmerged in the pt backend) | `Qwen2_5_VLForConditionalGeneration` from the local base folder, peft `PeftModel` unmerged, as the pt backend runs it |
| image loading | `load_image`: `convert("RGB")`, no EXIF rotation, alpha channel dropped | `swift_rgb` (`models/vlm/image.py`), same |
| resize | `--max_pixels 1003520`: `rescale_image` above the cap (bilinear), then qwen-vl-utils `fetch_image` (multiples of 28, 3,136-1,003,520 px, bicubic); processor with `do_resize=False` | `preprocess` hook `swift_qwen_vl_resize` (ported, pixel-identical in `tests/test_adb_vlm_smoke.py`); processor with `do_resize=False` |
| prompt | `VLM_SYSTEM_PROMPT`; user `"<image>\nWhether the image is authentic or AI-generated?"` | verbatim; the `<image>` tag becomes the image |
| decoding | `--temperature 0 --max_new_tokens 512 --seed 42`; repetition penalty unset in swift, so 1.05 from the base `generation_config.json` | greedy, 512 new tokens, repetition penalty 1.05 (default and "official" decoding are the same) |
| verdict | `parse_vlm_verdict` (raises on anything else; the script stops) | the same function (`models/vlm/parsers.py`); a rejected answer is "unparsed" (counted wrong) |

**Score.** P(fake) = sigmoid(l(ĠAI) - l(Ġauthentic)) at the verdict word the official parser read, from the **raw**
LM-head logits: the first "AI" / "authentic" token after `<answer>` (`score_source: "verdict_token"`), or, when the
model omits the tags and ends with the bare line "This is an AI-generated / authentic image." (which
`parse_vlm_verdict` also accepts), the "AI" / "authentic" token of that line (`"verdict_word"`; the parser-paired hook
`adb_verdict_offset` gives its character offset, and the engine finds the token holding it on the same incremental
decode as the `<answer>` search). A parsed verdict whose word cannot be located would get P(fake) 0.5 and
`score_source: "unscored"`, counted as `n_unscored` in every result JSON; there were none (below). The repetition
penalty is applied only to choose the generated tokens (both words occur in the prompt, so it shrinks both logits). The
penalised gap is recorded in the fidelity JSON for comparison.

**Loading.** Strict and offline: the base loads with no missing or unexpected tensor, and the adapter's tensors must equal
the file's key for key. For Qwen-VL adapters saved before the transformers 4.52 module rename, a family hook maps the
keys to the current names first. An unknown key still fails. A pre-4.52 vision-tower LoRA also fails, because peft
leaves it unloaded. The released adapter already uses the current names.

```python
import synthscope
det = synthscope.load("adb_vlm")               # needs both folders in $SYNTHSCOPE_CKPT_DIR (see Licence)
r = det.predict("photo.jpg")                   # fake_probability, logit, verdict, rationale, response, ...
```

## Fidelity
`repro/adb_vlm_fidelity.py` / `.sbatch`. Results: `repro/adb_vlm_fidelity.json` and `repro/logs/adb_vlm_fidelity_293634.out`
(job 293634, RTX A5000, bf16, SDPA, transformers 5.5.4, peft 0.21.3).

The run used 16 images: the gate's 6 GenImage val images, 7 SynthWildX, 2 RGBA WildRF PNGs and one 8 MP WildRF JPEG.
They range from 256x256 to 3264x2448; 6 are above the 1,003,520-pixel cap.

Three comparisons:
- with the official run (unmodified `predict.py`, ms-swift 4.4.2, pt backend);
- with the same swift command plus `--logprobs`, which gives the tokens and the chosen token's log-probability;
- with a direct transformers + peft call (the reference).

swift exposes only the chosen token's log-probability, not the logit gap. So, against swift, the text, the tokens
and that log-probability are identical. The logit gap is compared between the engine and the direct call.

| check | result |
|---|---|
| strict load | 729/729 base tensors equal to the shards; 392 adapter tensors equal to the file; 196 LoRA modules unmerged |
| vs the official ms-swift 4.4.2 run (unmodified `predict.py`, pt backend) | identical text 16/16, parsed verdict 16/16 |
| vs the same swift command with `--logprobs` | identical tokens 16/16 (1,570 tokens); chosen-token log-prob max diff 0.0 vs the reference |
| vs a direct transformers + peft call | identical tokens 16/16, verdict 16/16, verdict step 16/16 |
| verdict logit gap, engine vs the direct call (not vs swift) | max abs diff 0.0, raw and penalised |
| raw vs penalised gap (repetition penalty 1.05) | differs by 0.24-1.20 logits; same sign 16/16; P(fake) differs by at most 0.0016 |
| speed (engine, bs 1) | 0.204 img/s (4.9 s/img, 20 tok/s); swift 0.196 img/s |
| peak memory (engine) | 17.26 GB allocated (weights 16.76 GB) |
| VLM verdicts | 15/16 correct; the miss is the gate's ADM fake, as in the gate run |

Compared with the 2026-11-16 gate script (which reached 5/6 identical texts), two loading steps now follow swift
exactly:
- swift's `rescale_image` runs before `fetch_image` on images above the cap. This is the gate's mismatched 1024x1024
  Midjourney image.
- swift's `load_image` drops the alpha channel. The gate's ADM fake is an RGBA PNG; `fetch_image` alone composites it
  on white. The verdict did not change, but the P(fake) did: 3e-5 in the gate, and 3.4e-9 here on the image as swift
  prepares it.

## SynthScope adaptations

| choice | reason |
|---|---|
| transformers + peft instead of `swift infer` | ms-swift is not in the SynthScope environment. The request is the official one, and the output matches the pt backend token for token (above). The vLLM backend (the official default) was not compared. |
| P(fake) from the raw verdict-token logits | the official output is text only. The logit gap is the continuous score that agrees with the text, and AUROC needs it. The raw gap does not depend on the repetition penalty. |
| unparsed answers kept and flagged | the official script stops on an answer `parse_vlm_verdict` rejects. SynthScope records it as `verdict: "unparsed"`, counts it wrong and gives it logit 0. There were none in any run below. |
| tag-less answers scored at the parser's word | `parse_vlm_verdict` accepts a final line "This is an AI-generated / authentic image." without tags; 386 of the 19,286 images scored here end that way. The score is read at that line's "AI" / "authentic" token, the word the parser read. (Before the final-review fix these had P(fake) 0.5; the 386 were rescored with identical text, tokens and verdicts, so no accuracy changed.) |
| inference only, still images only, one image per call | the training (DPO in ms-swift) is not ported. |

## Results

**Inference only, with the official adapter, VLM branch only.** SynthScope did not train this model. The tables below
compare robustness and generalisation; they do not reproduce a training recipe or the paper's full method.

### Protocol (subsampled; the same images as Veritas++ and the 27 field-report detectors)

Every benchmark is **subsampled** exactly as for Veritas++ (`models/image/veritaspp/README.md`):
- the same manifests, `repro/vlm_bench/subsets/<benchmark>.json`, with `L = ceil(500 / generators)` reals and fakes per
  generator;
- 14,937 images per detector;
- the official **greedy** decoding;
- an unparsed answer counts as wrong. There were **0 unparsed answers** in every ADB result JSON: the 15 subsets, the
  two JPEG95 probes, the two fake-JPEG runs and the full WildRF run;
- **tag-less answers:** 446 of the 23,981 per-image records (386 of the 19,286 scored images; 0 to 42 per result JSON, at most 4.2%;
  full WildRF 51 of 2,503) end with the bare verdict line instead of `<answer>` tags. They are scored at that line's
  verdict word (`score_source: "verdict_word"`); **0 unscored** answers (`n_unscored`) in every result JSON.

`acc` is `bench_eval`'s balanced accuracy (mean over generators of the mean of real and fake accuracy). AUROC comes from the
verdict-token logit gap (after `<answer>`, or at the tag-less final line's verdict word).

**Ranks.** "Rank of 28" is among the 27 comparators plus ADB (1 = best; the other VLMs are not counted). `rank_all_*` in
`comparison.json` also counts the other VLMs, Veritas++ and FakeVLM. "Best publicly licensed" excludes LTF, MIRROR, VINA, PROBE and PROBE-ResNet.

**Sources.** Every number is in `repro/vlm_bench/comparison.json`, under `vlms.adb_vlm` per subset and `subset_means.adb_vlm`.
It is built by `repro/vlm_bench/compare.py` from `repro/vlm_bench/results/<bench>/adb_vlm*.json`. The logs are
`repro/logs/adb_ivi_*.out` (jobs 293636-293639, 293641, 293642 and 293763; IVI RTX A5000). The final-review rescore of
the 386 tag-less answers is jobs 293800 (ivi-cn026) and 293801 (ivi-cn027): every unit rerun from the per-image caches,
only those images scored again. `repro/vlm_bench/trace_check.py` checks every result JSON and per-image record against
its log.

### Accuracy (balanced, %)

| benchmark (images per class and generator L) | ADB acc | Veritas++ acc | best overall | best publicly licensed | median of 27 | rank of 28 | mean fake / real acc |
|---|---:|---:|---|---|---:|---:|---:|
| GenImage val (100k) (L=63) | **68.05** | 76.88 | 99.50 (`mirror`) | 96.03 (`omniaid_v1`) | 82.14 | 24 | 44.2 / 91.9 |
| ForenSynths (13 generators) (L=39) | **55.72** | 63.51 | 90.34 (`fat_former`) | 90.34 (`fat_former`) | 79.29 | 26 | 39.2 / 72.2 |
| UniversalFakeDetect (8 diffusion sets) (L=63) | **52.88** | 73.12 | 97.92 (`mirror`) | 95.83 (`npr`) | 87.40 | 28 | 41.5 / 64.3 |
| AIGCDetectBenchmark (17) (L=30) | **65.20** | 78.04 | 96.86 (`mirror`) | 93.73 (`omniaid_v1`) | 83.63 | 26 | 41.0 / 89.4 |
| Synthbuster (MSCOCO reals) (L=56) | **80.95** | 89.98 | 99.31 (`probe`) | 96.63 (`omniaid_v1`) | 87.80 | 21 | 63.7 / 98.2 |
| EvalGEN (L=100) | **97.30** | 99.10 | 99.90 (`mirror`) | 98.40 (`cure`) | 83.40 | 5 | 99.6 / 95.0 |
| SynthWildX (L=167) | **89.22** | 92.11 | 97.70 (`probe`) | 96.31 (`bfree`) | 61.28 | 7 | 95.2 / 83.2 |
| WildRF (L=167) | **93.92** | 95.64 | 97.47 (`mirror`) | 96.24 (`cure`) | 67.16 | 5 | 95.1 / 92.7 |
| AIGIBench (25 subsets, full) (L=20) | **76.70** | 82.40 | 89.00 (`mirror`) | 87.80 (`cure` (+1 tied)) | 79.20 | 16 | 62.2 / 91.2 |
| RRDataset (L=167) | **69.86** | 89.92 | 88.72 (`vina`) | 76.84 (`cure`) | 60.28 | 7 | 44.3 / 95.4 |
| BFree-Online (L=500; 269 real + 500 fake) | **87.96** | 79.35 | 97.67 (`mirror`) | 94.93 (`dda`) | 58.03 | 6 | 77.4 / 98.5 |
| Co-Spy-Bench (110 subsets) (L=5) | **78.36** | 93.82 | 99.73 (`mirror`) | 98.18 (`safe`) | 89.27 | 19 | 72.7 / 84.0 |
| DiTFake (L=167) | **95.11** | 99.50 | 100.00 (`mirror`) | 99.10 (`safe`) | 94.31 | 10 | 93.0 / 97.2 |
| ARForensics (L=72) | **64.78** | 74.40 | 99.50 (`mirror`) | 97.72 (`omniaid_v1`) | 83.04 | 23 | 36.9 / 92.7 |
| RealChain (L=72) | **72.72** | 82.74 | 80.75 (`mirror` (+1 tied)) | 78.17 (`cure`) | 61.71 | 8 | 63.7 / 81.8 |
| **mean of 15** | **76.58** | 84.70 | | | | | |

### AUROC (%)

| benchmark | ADB mAUROC | Veritas++ mAUROC | best overall | best publicly licensed | median of 27 | rank of 28 |
|---|---:|---:|---|---|---:|---:|
| GenImage val (100k) | **82.70** | 90.17 | 100.00 (`mirror`) | 99.83 (`effort`) | 94.25 | 24 |
| ForenSynths (13 generators) | **64.55** | 74.82 | 98.89 (`vina`) | 98.18 (`omniaid_genimage`) | 89.79 | 26 |
| UniversalFakeDetect (8 diffusion sets) | **54.65** | 84.94 | 99.97 (`mirror`) | 99.51 (`omniaid_v1`) | 96.97 | 28 |
| AIGCDetectBenchmark (17) | **74.24** | 89.11 | 99.50 (`ltf`) | 98.64 (`omniaid_genimage`) | 93.92 | 26 |
| Synthbuster (MSCOCO reals) | **93.22** | 98.43 | 100.00 (`mirror` (+1 tied)) | 99.99 (`pgc`) | 96.52 | 19 |
| EvalGEN | **99.79** | 99.95 | 100.00 (`mirror` (+1 tied)) | 99.70 (`dda`) | 95.59 | 3 |
| SynthWildX | **97.74** | 97.34 | 99.55 (`probe`) | 98.96 (`bfree`) | 72.41 | 5 |
| WildRF | **98.81** | 99.25 | 99.79 (`mirror`) | 99.30 (`cure`) | 72.95 | 4 |
| AIGIBench (25 subsets, full) | **87.68** | 90.53 | 98.78 (`mirror`) | 97.54 (`bfree`) | 90.48 | 15 |
| RRDataset | **77.42** | 91.56 | 95.60 (`vina`) | 87.15 (`dinov3`) | 64.62 | 6 |
| BFree-Online | **98.36** | 86.32 | 99.50 (`mirror`) | 98.82 (`dda`) | 76.52 | 3 |
| Co-Spy-Bench (110 subsets) | **84.84** | 98.00 | 100.00 (`mirror`) | 99.85 (`safe`) | 96.47 | 20 |
| DiTFake | **98.86** | 99.90 | 100.00 (`mirror` (+1 tied)) | 100.00 (`safe`) | 99.28 | 16 |
| ARForensics | **68.27** | 84.49 | 99.76 (`mirror`) | 99.70 (`effort`) | 96.12 | 26 |
| RealChain | **80.57** | 85.08 | 89.81 (`omniaid_v1`) | 89.81 (`omniaid_v1`) | 74.21 | 10 |
| **mean of 15** | **84.11** | 91.33 | | | | |

### What the numbers show (from `comparison.json`, `vlms.adb_vlm.balance`)

**In the wild and on recent generators it does well**, ranking 5th to 8th of 28:
- WildRF 93.92 (5th);
- EvalGEN 97.30 (5th);
- BFree-Online 87.96 (6th);
- SynthWildX 89.22 (7th);
- RRDataset 69.86 (7th);
- RealChain 72.72 (8th).

It is behind the best publicly licensed detector on all 15 benchmarks; the closest are EvalGEN (-1.10) and WildRF (-2.32).

**Against Veritas++:**
- Accuracy: ADB is ahead only on BFree-Online (87.96 vs 79.35).
- AUROC: ADB is ahead on BFree-Online (98.36 vs 86.32) and on SynthWildX (97.74 vs 97.34).
- Veritas++ is ahead on everything else.

**On older GAN and diffusion sets and on autoregressive generators it is near chance:**

| benchmark | acc | rank of 28 |
|---|---:|---:|
| UFD-8 | 52.88 | 28 (last) |
| ForenSynths | 55.72 | 26 |
| ARForensics | 64.78 | 23 |
| AIGCDetect | 65.20 | 26 |
| GenImage | 68.05 | 24 |

On these sets the real accuracy (64-93%) is higher than the fake accuracy (37-44%, mean over generators). Fake
accuracy is below 50% on:
- 9 of 13 ForenSynths generators (StyleGAN2 7.7, BigGAN 10.3);
- 6 of 8 UFD sets (guided 11.1);
- 4 of 8 GenImage generators (ADM 11.1, BigGAN 15.9, VQDM 20.6, GLIDE 33.3);
- 4 of 7 ARForensics generators (Open-MAGVIT2 1.4, VAR 6.9, RAR 8.3, LlamaGen 15.3).

This fits the method: the VLM branch is trained on semantic implausibility, and the paper leaves pixel artifacts to the
expert (DDA; see `adb_full`). On UFD-8 the real accuracy also drops, to 64.3.

**Weak spots inside strong sets:**
- RRDataset: transfer 13.2 and re-digitised 25.1 fake accuracy (original 94.6);
- RealChain: Seedream4 38.9;
- Synthbuster: Midjourney v5 26.8;
- AIGIBench: the face-swap subsets (SimSwap 0.0, BlendFace 5.0, InSwap 5.0).

**Cost: 4.4-5.1 s per image** (per-benchmark means on an RTX A5000), with 93-100 generated tokens per image on average.
Veritas++ takes 16.8-20.6 s and writes 488-540 tokens per image.

### Robustness: JPEG quality 95 on every image

All images (reals and fakes) are re-encoded at quality 95 before scoring. The subsets are the same, and the comparators
get the same transform:

| subset | ADB acc clean -> JPEG95 | AUROC change | Veritas++ acc change / AUROC change | median of the 27: acc change / AUROC change |
|---|---|---:|---:|---:|
| DiTFake | 95.11 -> 95.41 (**+0.30**) | +0.35 | +0.10 / +0.06 | -2.00 / -1.14 |
| UFD (8 diffusion sets) | 52.88 -> 52.18 (**-0.70**) | -0.84 | -5.26 / -5.68 | -2.08 / -1.59 |

ADB barely changes under JPEG on either set. On UFD-8 it was near chance before the re-encoding, so there was little to
lose. This is two sets only, not a general robustness result. Greedy vs sampled does not apply, because the official
decoding is greedy.

### Comparison with the paper (arXiv 2512.06746 v3; balanced accuracy)

Every AlignGemini number the paper prints is the **full method**, the VLM branch OR the DDA expert. Its reproduction is
in the [`adb_full` README](../adb_full/README.md#results) (`repro/adb_full/paper_comparison.json`). **The paper prints no
VLM-only number for the released adapter**, so for this branch the printed numbers below are context, not targets, and
no row gets a delta. Two other rows are listed because they also concern Qwen2.5-VL-7B, but neither is the released
adapter:
- Table 4's "Qwen2.5-VL-7B" row is ambiguous between the base model and a trained VLM;
- Table 5's "Qwen2.5-VL-7B (Untrained)" row is the base model without the adapter.

| benchmark (ours) | paper, full method (VLM OR expert; context only) | Table 4 / Table 5 Qwen2.5-VL-7B (not the adapter) | SynthScope, VLM branch only (greedy) |
|---|---:|---|---|
| WildRF, **full test split** (2,503 images) | 92.9 (Facebook 92.2 / Reddit 90.9 / Twitter 95.5) | 86.3 / 83.8 | **93.98** (Facebook 93.13 / Reddit 93.06 / Twitter 95.75); AUROC 98.78 |
| BFree-Online, subset of **269 real + 500 fake** (769 images) | 97.5 | 82.9 / 77.6 | **87.96**; AUROC 98.36 |
| AIGI-Bench CommunityAI + SocialRF, **subset slice** (20 real + 20 fake each) | 88.8 (91.9 / 85.6) | 80.4 / 75.7 | **82.50** (85.0 / 80.0) |
| GenImage, **subset** (63 + 63 per generator), fakes JPEG q96 | 91.7 | n/a | **65.97**; AUROC 79.18 (clean: 68.05) |
| AIGCDetectBenchmark, **subset** (30 + 30 per generator), fakes JPEG q96 (**inferred protocol**) | 87.7 | n/a | **63.92**; AUROC 70.55 (clean: 65.20) |
| CO-SPY-Bench in-the-wild | 94.3 | 67.6 / 57.4 | **not held** (only the synthetic Co-Spy-Bench is held: subset acc 78.36 above) |

Notes on the rows:
- **WildRF.** This is the one full-data paper benchmark. The released VLM branch alone gives 93.98; the paper's full
  method prints 92.9, and SynthScope's full method (`adb_full`) gives 91.22 on the same 2,503 images.
- **GenImage.** The paper JPEG-compresses the fakes "following BFree" to match the reals' format, and gives no quality
  factor. We used q96 (`bench_eval --jpeg-fakes 96`, reals untouched); q96 is our choice, following DDA's published
  GenImage protocol.
- **AIGCDetectBenchmark.** The paper states the fake-JPEG step for GenImage only; its AIGCDetectBenchmark table does not
  mention it. Applying it here, at q96, is **our inference**. The clean run is in the subset table.
- **BFree-Online.** The subset has 269 reals because the benchmark folder holds only 269. Accuracy is balanced, so the
  imbalance does not weight it.

Sources: `paper_comparison.adb_vlm` and `paper_note.adb_vlm` in `comparison.json`, built from
`results/wildrf/adb_vlm__full*.json`, `results/{genimage_val100k,aigcdetect17}/adb_vlm__jpegfakes96.json`,
`results/bfree_online/adb_vlm.json` and `results/aigibench25/adb_vlm.json`.

### Reproduce

```bash
python benchmark/bench_eval.py --model adb_vlm --benchmark pergen --data <subset-root> --limit <L> --vlm-cache <dir> --tag adb_vlm_<bench>
#   add --jpeg-fakes 96 for the GenImage / AIGCDetect paper-protocol runs
VLM_MODEL=adb_vlm UNITS=vlm:wildrf,vlm:evalgen sbatch --job-name=adb_ivi repro/vlm_bench/vpp_ivi.sbatch   # run_units.py on IVI
python repro/vlm_bench/compare.py        # regenerates comparison.json from the result JSONs
python repro/vlm_bench/trace_check.py --logs repro/logs   # every result JSON and per-image record against its log
```

## Caveats
- **VLM branch only.** The paper's detector is the VLM branch OR the DDA pixel expert (`adb_full`). No number here is
  the paper's method.
- **Fidelity is against the ms-swift pt backend.** The official default is the vLLM backend, which was not compared.
- **Most scores are near 0 or 1, not all.** On the 16 fidelity images the verdict-token gap is 5.1-25.3 logits in
  magnitude. Over the 19,286 scored benchmark images the median |gap| is 19.6; 512 (2.7%) have |gap| < 2.2, i.e.
  P(fake) between 0.1 and 0.9, and 9 have a gap of exactly 0 (bf16 ties; scored, verdict "real"). No image got
  P(fake) 0.5 for want of a verdict token (0 unscored, 0 unparsed).
- **Subsets.** All 15 benchmarks are subsampled. Only WildRF was also run on its full data.
- **Cache identity (also affects Veritas++).** The `--vlm-cache` folder is keyed by, among other things, the adapter's and
  the base model's revisions, the VLM config (including the resize parameters `preprocess_kwargs` = `max_pixels`
  1,003,520 and the parser's `verdict_offset` hook) and a hash of `models/vlm/*.py`.
  - Every committed ADB result records VLM code version `b490c64477` (`vlm.identity` in each JSON).
  - Any edit to `models/vlm/*.py` re-keys the caches of **every** VLM detector, Veritas++ included. This branch edits
    `models/vlm/vlm.py`, `image.py` and `parsers.py`, so it re-keys them.
  - A rerun after such an edit scores from scratch: it never reuses records written by other code. The one exception
    is an explicit, checked carry-over: the final-review fix (verdict-word hook, identity fields) moved the ADB caches
    with `repro/vlm_bench/migrate_vlm_cache.py`, which kept the 18,900 records the change cannot affect and left out
    the 386 tag-less ones (rescored). Veritas++ caches were not migrated.
  - The FakeVLM integration edits `models/vlm/*.py` again (a new LLaVA family, image hooks and parser that ADB does not
    use), so the caches written by main e5749ae are re-keyed once more. `migrate_vlm_cache.py --model adb_vlm` carries
    them over byte for byte (rule `unchanged`, see its docstring).

## Availability
- Not on the Hugging Face Space: a 7B model needs a GPU (17.26 GB peak).
- Not in the model zoo (`NEVER_REDISTRIBUTE` in `synthscope/hub.py`). `synthscope.load("adb_vlm")` without the folders
  prints the download commands below.

## Licence
The code and the VLM adapter are **Apache-2.0** (since 2026-11-20):
- the code repository has an Apache-2.0 `LICENSE` (commit `4ce147f`; its README: "The code and VLM adapter are licensed
  under Apache-2.0");
- the Hugging Face adapter card has the `license: apache-2.0` tag and a `LICENSE` file since revision `ff7973df`. That
  revision changes only the card and adds the licence: `adapter_model.safetensors` (sha256 `78b127c5c7623c01c25d9f57d039033bee4d6de608b077a507c326d293f11b5f`) and
  `adapter_config.json` are byte-identical to the pinned `49975ccd`, so the pin is kept (it also keys the VLM caches).

The base model Qwen2.5-VL-7B-Instruct is Apache-2.0. Redistribution is therefore allowed, but the adapter is **not
mirrored** to the model zoo: it needs the separate 16 GB base folder, and mirroring it is the maintainers' call (the key
stays in `NEVER_REDISTRIBUTE`). See [LICENSES.md](../../../LICENSES.md). Download both folders yourself into
`$SYNTHSCOPE_CKPT_DIR`:

```
hf download Gaffeyzz/Align-Dont-Blend-VLM --revision 49975ccd8b28eaab8ef14650ffee0c545934d02b --local-dir $SYNTHSCOPE_CKPT_DIR/adb_vlm
hf download Qwen/Qwen2.5-VL-7B-Instruct --revision cc594898137f460bfe9f0759e9844b3ce807cfb5 --local-dir $SYNTHSCOPE_CKPT_DIR/qwen2.5-vl-7b-instruct
pip install ".[vlm]"     # peft 0.21.3, qwen-vl-utils 0.0.14
```
