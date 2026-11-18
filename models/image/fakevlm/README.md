# FakeVLM (`fakevlm`)

**Paper:** Wen et al., *Spot the Fake: Large Multimodal Model-Based Synthetic Image Detection with Artifact
Explanation*, NeurIPS 2025 ([arXiv 2503.14905](https://arxiv.org/abs/2503.14905)).
**Code:** [opendatalab/FakeVLM](https://github.com/opendatalab/FakeVLM) @ `fdcdcf1` (no licence file).
**Weights:** [lingcco/fakeVLM](https://huggingface.co/lingcco/fakeVLM) @ `dce9e88d` (no licence, no model card): a full
LLaVA-1.5-7B in bf16, 14.1 GB. **Processor:** the tokenizer and processor files of
[llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf) @ `a272c74`, as the official `eval.py`
loads them.

**Status.** Inference only, with the official weights. Not on the Space and not in the model zoo: neither the code nor
the weights has a licence, and the base LLaVA-1.5 / Vicuna is under the Llama 2 Community License ([Licence](#licence)).
**GenImage is in its training data** (FakeClue contains GenImage real and fake images), so GenImage is
in-distribution here. The paper's benchmarks (FakeClue, LOKI, DMimage, DD-VQA, FF++) are not held, so **there is no
paper reproduction claim**.

- **Fidelity** (16 images): the first sentence ("This is a real / fake image.") and the parsed verdict equal **all
  five official runs on 16 / 16 images**. These are the batched `eval.py` under transformers 4.45.2 and 5.5.4, whose
  explanations change with the batch size, and the official code run one image at a time. At the token level, the
  greedy text equals the single-image run under the authors' transformers 4.45.2 on 16 / 16 ([Fidelity](#fidelity)).
- **15 benchmarks**, each subsampled and scored on the same images as Veritas++, Align, Don't Blend and 27 other
  detectors ([Results](#results)).
  - **Held-out mean (14 benchmarks, GenImage left out): 84.15 acc / 91.16 AUROC.** With the in-distribution GenImage the
    mean of the 15 is 85.19 / 91.74 (Veritas++ 84.70 / 91.33, Align, Don't Blend 76.58 / 84.11; both means are over all
    15).
  - Strong on GAN, diffusion and autoregressive sets: 2nd of 28 on UFD-8 (97.02), DiTFake (99.30) and ARForensics
    (97.92), each above the best publicly licensed detector.
  - Weak in the wild, because it calls real photos fake: SynthWildX 65.46 and WildRF 64.11, with real accuracy 32.93 and
    33.41.
  - JPEG quality 95 drops UFD-8 from 97.02 to 65.67. The reals stay at 98.61 and the fakes fall from 95.44 to 32.74.

## Method
FakeVLM fine-tunes LLaVA-1.5-7B on FakeClue (over 100,000 real and synthetic images in seven categories, with artifact
explanations in natural language). Asked "Does the image looks real/fake?", it answers "This is a real image." or
"This is a fake image." and then, usually for fakes, explains the artifacts.

## Official inference, as SynthScope reproduces it
`scripts/eval.py` (transformers 4.45.2, batches of 16 in `eval.sh`). SynthScope sends the same request through
transformers 5 (`models/image/fakevlm/fakevlm.py`, shared VLM kind `models/vlm/`, `LlavaFamily`).

| step | official (`scripts/eval.py`, transformers 4.45.2) | SynthScope |
|---|---|---|
| model | `LlavaForConditionalGeneration`, bf16, flash-attention 2 | the same class, bf16, SDPA (flash-attn is not installed); loaded strictly, offline |
| processor | `AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", revision='a272c74')` | the same files from a local folder (`llava-1.5-7b-hf_a272c74_processor/`), plus `patch_size=14`, `vision_feature_select_strategy="default"`, `num_additional_image_tokens=1`, which transformers 5 needs to expand `<image>` to the 576 tokens 4.45.2 merged in the model. The load fails early if any of the three is unset. FakeVLM's own tokenizer appends `</s>` to the prompt (`add_eos_token=true`) and is refused |
| image | `Image.open`; the processor converts to RGB (alpha dropped, no EXIF rotation), resizes the shorter side to 336 (PIL bicubic) and centre-crops 336 | `transformers_rgb` + `clip_resize_center_crop` (transformers 4.45.2's code, ported in `models/vlm/image.py`; pixel values equal to 4.45.2's to 2.4e-7); the processor then only rescales and normalises. transformers 5's own resize uses torchvision and differs by up to 0.03 |
| prompt | the raw FakeClue question `"<image>Does the image looks real/fake?"` (no chat template), `padding="max_length", max_length=1024, truncation=True`, left padding; 4.45.2 expands `<image>` inside the model, so the model sees 1,599 positions | the same prompt; `max_length=1599` after the processor's expansion, i.e. the same 1,599-position layout with 1,013 pads |
| decoding | `generate(max_new_tokens=256)`, the model's generation config (greedy) | greedy, 256 new tokens |
| verdict | the text after the last "?"; "real" in the first sentence, else "fake", else the same on the second sentence, else `random.choice` | the same rule (`parse_fakevlm`, `models/vlm/parsers.py`); where the official code picks at random, "unparsed" (counted wrong) |

**Score.** P(fake) = sigmoid(l(▁fake) - l(▁real)) of the raw LM-head logits at the **first** "▁fake" / "▁real" token
the model generates (the counterpart pair: the vocabulary has one spelling of "fake" and four of "real"). It is
usually the 5th token ("▁▁ This is a" first), sometimes the 4th or 6th. Only spellings with a same-spelling counterpart
are scored. A first verdict word without one, for example a bare "real" without the "▁", would give P(fake) 0.5 and
`score_source: "unscored"` (the parsed verdict still counts for accuracy). This did not happen on any image here (0
unscored in the fidelity run and in every benchmark result).

**Teacher-forced variant (not the default).** `FAKEVLM_TEACHER_FORCED` reads the same gap from one forward pass on the
prompt + "▁▁ This is a", about 9x faster (0.38 vs 3.41 s per image). It failed the rule for a default scorer: the greedy
answer does not always start with those four tokens, and its P(fake) differs from the generated one by more than 1e-3
on some images ([Fidelity](#fidelity)). Results scored with it are not the official method.

```python
import synthscope
det = synthscope.load("fakevlm")              # needs both folders in $SYNTHSCOPE_CKPT_DIR (see Licence)
r = det.predict("photo.jpg")                  # fake_probability, logit, verdict, response (the explanation), ...
```

## Fidelity
`repro/fakevlm_fidelity.py`, `repro/fakevlm_official.py` / `repro/fakevlm_fidelity.sbatch`. Results:
`repro/fakevlm_fidelity.json` and `repro/logs/fakevlm_fidelity_293873.out` (job 293873, RTX A5000, bf16, SDPA,
transformers 5.5.4; the official runs under transformers 4.45.2 from a `pip --target` folder, and under 5.5.4).

The run used 16 images, from 256x256 to 3264x2448: the gate's 6 GenImage val images, 7 SynthWildX images and 3 WildRF
reals (two PNGs and one 8 MP JPEG). Three are RGBA: the GenImage ADM fake and the two WildRF PNGs. The fakes come from
GenImage and SynthWildX; WildRF gives reals only. An **extended set** of 224 images (8 per class folder of GenImage val,
SynthWildX and WildRF) is used only for the teacher-forced check, which needs just the first 8 generated tokens. These
224 have no official-run comparison.

The official script runs in batches (`eval.sh`: 16) and cannot run one image at a time: its `.squeeze()` drops the
batch axis. Its greedy text depends on the batch size: under 4.45.2, batches of 4 and of 2 agree on 14/16 texts, and
batches of 4 and single images on 11/16 (bf16 numerics; the padding and the batch change the reductions). So the
engine (one image at a time) is compared with five official runs:
- `eval.py` unmodified, in batches of 4 and of 2 under 4.45.2, and of 4 under 5.5.4;
- eval.py's own dataset item, model and `generate` call, one image at a time, under 4.45.2 and 5.5.4. This
  single-image mode was written for this check, because the shipped `eval.py` cannot run batches of 1.

`eval.sh`'s default batch of 16 was not run, and every official run used SDPA instead of the hard-coded
flash-attention 2 (not installed).

| check | result |
|---|---|
| strict load | 686/686 tensors equal to the shards; no network access |
| inputs vs the official 4.45.2 processor | pixel values max abs diff 2.4e-7; token ids equal 16/16 (1,599 positions) |
| first sentence and parsed verdict vs all five official runs | **16/16 in each** |
| vs the official code one image at a time, transformers 4.45.2 (the authors' pin) | **identical text 16/16** |
| vs `eval.py` unmodified, 4.45.2, batches of 4 / of 2 | identical text 11/16 / 12/16 |
| vs `eval.py` unmodified, 5.5.4, batches of 4; the code one at a time, 5.5.4 | identical text 10/16 / 13/16 |
| vs a direct transformers call (the reference) | identical tokens 16/16, verdict step 16/16, verdict 16/16 |
| verdict logit gap, engine vs the direct call on the same runtime (transformers 5.5.4, fed 4.45.2's token ids) | max abs diff 0.0 (generated); 0.0 (teacher-forced). This checks the scoring hook and the verdict-step choice, not logits across transformers versions: the official runs return text only |
| official accuracy on the 16 | 13/16 in every official batched run and in the engine (3 reals called fake) |
| speed (engine, bs 1, generated) | 0.294 img/s (3.41 s/img, 29 tok/s): 0.63 s for a bare verdict (9 tokens), 8.3 s for 256 tokens |
| speed (teacher-forced variant) | 2.6 img/s (0.38 s/img) |
| peak memory (engine) | 15.1 GB allocated (weights 14.1 GB) |

Every text difference is in the explanation after the first sentence ("This is a fake / real image."), or in the
whitespace before it (one image under 5.5.4). The token-level match is fragile: the same single-image code under
transformers 5.5.4 matches the 4.45.2 run on 13/16 texts. The verdict-level match holds in every run.

**Teacher-forced variant vs the generated score.** The rule for making it the default: the same verdict-token
position, |P(fake) diff| <= 1e-3 on every image, and every answer starting "This is a". Result: on the 16, the answer
starts "▁▁ This is a" 16/16, but |P diff| > 1e-3 on 2 (max 0.0023). bf16 logit gaps from cached decoding and from one
forward pass differ by one or two bf16 steps (up to 0.25). On the 224 extended images, 219 start with those tokens (the
rest lack the "▁▁" or start with a newline); |P diff| > 1e-3 on 29 (max 0.031, gaps up to 0.75 apart); the verdicts
agree on 223. Rule not met, so the **generated path is the default**.

**Three transformers-version steps** make the request equal to 4.45.2's:
- the processor attributes `patch_size`, `vision_feature_select_strategy`, `num_additional_image_tokens` (otherwise
  transformers 5 fails, or inserts 575 tokens);
- the PIL resize and crop of 4.45.2's CLIP processor (transformers 5's torchvision resize differs by up to 0.03);
- `max_length=1599`: 4.45.2 pads the 1024-token text and then expands `<image>` in the model, so it sees 1,599
  positions (1,013 pads). With `max_length=1024` (438 pads) the text equalled the official single-image 4.45.2 run on
  12/16 (job 293846); with 1599 on 16/16.

## SynthScope adaptations

| choice | reason |
|---|---|
| transformers 5.5.4 instead of 4.45.2 | the SynthScope environment. The three steps above make the request the same; the verdict then equals every official run, and the text equals the official code run one image at a time under 4.45.2 on 16/16 |
| SDPA instead of flash-attention 2 | flash-attn is not installed; the official runs here used SDPA too |
| one image at a time | `eval.py` batches, and its text depends on the batch size (above); a single image is the batch-independent reference |
| P(fake) from the raw logits at the first "▁fake" / "▁real" | the official output is text only; AUROC needs a continuous score that agrees with the text |
| unparsed answers kept and flagged | the official script picks a random label for them; SynthScope counts them wrong (logit 0). None in the fidelity run, 1 in the benchmarks |
| inference only, still images only | the training (LLaVA fine-tuning on FakeClue) is not ported |

## Results

**Inference only, with the official weights.** SynthScope did not train this model. The tables compare robustness and
generalisation. They are not a reproduction of the paper, whose benchmarks are not held.

**In-distribution.** FakeClue, FakeVLM's training set, contains GenImage images, so the GenImage row is not held-out.
It is flagged in `comparison.json` (`in_distribution`, `vlms.fakevlm.in_distribution`), and
`subset_means.fakevlm.held_out` gives the mean without it. The FakeClue test file also lists Chameleon, FF++,
document and satellite images. Whether FakeClue overlaps the other 14 subsets has not been checked, so "held-out" means
"not known to be in the training data".

### Protocol (subsampled; the same images as Veritas++, Align, Don't Blend and 27 other detectors)

Every benchmark is **subsampled** exactly as for Veritas++ (`models/image/veritaspp/README.md`):
- the same manifests, `repro/vlm_bench/subsets/<benchmark>.json`, with `L = ceil(500 / generators)` reals and fakes per
  generator;
- 14,937 images per detector;
- the official **greedy** decoding with 256 new tokens; 3,448 of the 16,947 scored answers (the 15 subsets and the two
  JPEG95 probes) reach that limit and are cut, as in the official run (the verdict is in the first sentence);
- an unparsed answer counts as wrong: **1 unparsed** answer (an ARForensics real) and **0 unscored** answers over the
  16,947.

`acc` is `bench_eval`'s balanced accuracy (mean over generators of the mean of real and fake accuracy). AUROC comes from the
verdict-token logit gap (the first "▁fake" / "▁real" the model writes).

**Ranks.** "Rank of 28" is among the 27 comparators plus FakeVLM (1 = best; the other VLMs are not counted). `rank_all_*`
in `comparison.json` also counts Veritas++ and Align, Don't Blend (ADB). "Best publicly licensed" excludes LTF, MIRROR,
VINA, PROBE and PROBE-ResNet.

**Sources.** Every number is in `repro/vlm_bench/comparison.json`, under `vlms.fakevlm` per subset and
`subset_means.fakevlm`. It is built by `repro/vlm_bench/compare.py` from `repro/vlm_bench/results/<bench>/fakevlm*.json`.
The logs are `repro/logs/fakevlm_bench_293884.out` and `fakevlm_bench_293885.out` (IVI RTX A5000; code 37eae6c), and
`repro/logs/fakevlm_compare_293884_293885.out` (trace check, invariance check and the compare output).
`repro/vlm_bench/trace_check.py` checks every result JSON and per-image record against its log.

### Accuracy (balanced, %)

| benchmark (images per class and generator L) | FakeVLM acc | Veritas++ acc | ADB acc | best overall | best publicly licensed | median of 27 | rank of 28 | mean fake / real acc |
|---|---:|---:|---:|---|---|---:|---:|---:|
| GenImage val (100k) (L=63), **in-distribution** | **99.70** | 76.88 | 68.05 | 99.50 (`mirror`) | 96.03 (`omniaid_v1`) | 82.14 | 1 | 99.8 / 99.6 |
| ForenSynths (13 generators) (L=39) | **76.43** | 63.51 | 55.72 | 90.34 (`fat_former`) | 90.34 (`fat_former`) | 79.29 | 17 | 96.06 / 56.8 |
| UniversalFakeDetect (8 diffusion sets) (L=63) | **97.02** | 73.12 | 52.88 | 97.92 (`mirror`) | 95.83 (`npr`) | 87.40 | 2 | 95.44 / 98.61 |
| AIGCDetectBenchmark (17) (L=30) | **92.45** | 78.04 | 65.20 | 96.86 (`mirror`) | 93.73 (`omniaid_v1`) | 83.63 | 5 | 97.45 / 87.45 |
| Synthbuster (MSCOCO reals) (L=56) | **96.52** | 89.98 | 80.95 | 99.31 (`probe`) | 96.63 (`omniaid_v1`) | 87.80 | 6 | 98.41 / 94.64 |
| EvalGEN (L=100) | **98.10** | 99.10 | 97.30 | 99.90 (`mirror`) | 98.40 (`cure`) | 83.40 | 4 | 99.2 / 97.0 |
| SynthWildX (L=167) | **65.46** | 92.11 | 89.22 | 97.70 (`probe`) | 96.31 (`bfree`) | 61.28 | 13 | 98.0 / 32.93 |
| WildRF (L=167) | **64.11** | 95.64 | 93.92 | 97.47 (`mirror`) | 96.24 (`cure`) | 67.16 | 17 | 94.81 / 33.41 |
| AIGIBench (25 subsets, full) (L=20) | **83.90** | 82.40 | 76.70 | 89.00 (`mirror`) | 87.80 (`cure` (+1 tied)) | 79.20 | 9 | 97.4 / 70.4 |
| RRDataset (L=167) | **62.67** | 89.92 | 69.86 | 88.72 (`vina`) | 76.84 (`cure`) | 60.28 | 14 | 75.25 / 50.1 |
| BFree-Online (L=500; 269 real + 500 fake) | **70.47** | 79.35 | 87.96 | 97.67 (`mirror`) | 94.93 (`dda`) | 58.03 | 9 | 95.6 / 45.35 |
| Co-Spy-Bench (110 subsets) (L=5) | **96.00** | 93.82 | 78.36 | 99.73 (`mirror`) | 98.18 (`safe`) | 89.27 | 6 | 96.0 / 96.0 |
| DiTFake (L=167) | **99.30** | 99.50 | 95.11 | 100.00 (`mirror`) | 99.10 (`safe`) | 94.31 | 2 | 99.8 / 98.8 |
| ARForensics (L=72) | **97.92** | 74.40 | 64.78 | 99.50 (`mirror`) | 97.72 (`omniaid_v1`) | 83.04 | 2 | 96.43 / 99.4 |
| RealChain (L=72) | **77.78** | 82.74 | 72.72 | 80.75 (`mirror` (+1 tied)) | 78.17 (`cure`) | 61.71 | 5 | 84.13 / 71.43 |
| **mean of the 14 held-out (without GenImage)** | **84.15** | | | | | | | |
| mean of 15 (with the in-distribution GenImage) | 85.19 | 84.70 | 76.58 | | | | | |

### AUROC (%)

| benchmark | FakeVLM mAUROC | Veritas++ mAUROC | ADB mAUROC | best overall | best publicly licensed | median of 27 | rank of 28 |
|---|---:|---:|---:|---|---|---:|---:|
| GenImage val (100k), **in-distribution** | **99.99** | 90.17 | 82.70 | 100.00 (`mirror`) | 99.83 (`effort`) | 94.25 | 2 |
| ForenSynths (13 generators) | **88.49** | 74.82 | 64.55 | 98.89 (`vina`) | 98.18 (`omniaid_genimage`) | 89.79 | 16 |
| UniversalFakeDetect (8 diffusion sets) | **99.90** | 84.94 | 54.65 | 99.97 (`mirror`) | 99.51 (`omniaid_v1`) | 96.97 | 2 |
| AIGCDetectBenchmark (17) | **97.55** | 89.11 | 74.24 | 99.50 (`ltf`) | 98.64 (`omniaid_genimage`) | 93.92 | 7 |
| Synthbuster (MSCOCO reals) | **99.85** | 98.43 | 93.22 | 100.00 (`mirror` (+1 tied)) | 99.99 (`pgc`) | 96.52 | 6 |
| EvalGEN | **99.94** | 99.95 | 99.79 | 100.00 (`mirror` (+1 tied)) | 99.70 (`dda`) | 95.59 | 3 |
| SynthWildX | **87.78** | 97.34 | 97.74 | 99.55 (`probe`) | 98.96 (`bfree`) | 72.41 | 9 |
| WildRF | **80.86** | 99.25 | 98.81 | 99.79 (`mirror`) | 99.30 (`cure`) | 72.95 | 12 |
| AIGIBench (25 subsets, full) | **95.38** | 90.53 | 87.68 | 98.78 (`mirror`) | 97.54 (`bfree`) | 90.48 | 6 |
| RRDataset | **74.64** | 91.56 | 77.42 | 95.60 (`vina`) | 87.15 (`dinov3`) | 64.62 | 9 |
| BFree-Online | **70.17** | 86.32 | 98.36 | 99.50 (`mirror`) | 98.82 (`dda`) | 76.52 | 16 |
| Co-Spy-Bench (110 subsets) | **99.02** | 98.00 | 84.84 | 100.00 (`mirror`) | 99.85 (`safe`) | 96.47 | 5 |
| DiTFake | **99.99** | 99.90 | 98.86 | 100.00 (`mirror` (+1 tied)) | 100.00 (`safe`) | 99.28 | 3 |
| ARForensics | **99.84** | 84.49 | 68.27 | 99.76 (`mirror`) | 99.70 (`effort`) | 96.12 | 1 |
| RealChain | **82.77** | 85.08 | 80.57 | 89.81 (`omniaid_v1`) | 89.81 (`omniaid_v1`) | 74.21 | 8 |
| **mean of the 14 held-out (without GenImage)** | **91.16** | | | | | | |
| mean of 15 (with the in-distribution GenImage) | 91.74 | 91.33 | 84.11 | | | | |

### What the numbers show (from `comparison.json`, `vlms.fakevlm.balance`)

**GenImage (99.70, 1st of 28) is in its training data** and says nothing about generalisation.

**On GAN, diffusion and autoregressive sets it is among the best** (held-out):
- UFD-8 97.02, DiTFake 99.30 and ARForensics 97.92, each 2nd of 28 and above the best publicly licensed detector (+1.19,
  +0.20, +0.20); in AUROC it is 1st on ARForensics (99.84);
- EvalGEN 98.10 (4th), AIGCDetect 92.45 (5th), Synthbuster 96.52 (6th), Co-Spy-Bench 96.00 (6th).

**In the wild it calls real photos fake.** Real accuracy drops to 32.93-50.1, while fake accuracy stays at 94.81-98.0
(RRDataset: 75.25):

| benchmark | acc | rank of 28 | real acc | fake acc |
|---|---:|---:|---:|---:|
| SynthWildX | 65.46 | 13 | 32.93 | 98.0 |
| WildRF | 64.11 | 17 | 33.41 | 94.81 |
| BFree-Online | 70.47 | 9 | 45.35 | 95.6 |
| RRDataset | 62.67 | 14 | 50.1 | 75.25 |

ForenSynths (76.43, 17th) shows the same pattern (real 56.8, fake 96.06). This is the opposite of ADB, which is strong in
the wild and near chance on the older sets.

**Against Veritas++:** FakeVLM is ahead in accuracy on 8 of the 15 benchmarks, 7 of them held-out (GenImage is the
eighth): ForenSynths, UFD-8, AIGCDetect, Synthbuster, AIGIBench, Co-Spy-Bench and ARForensics. Veritas++ is ahead on
EvalGEN, DiTFake, RealChain and the four in-the-wild sets above.

**Weak generators inside strong sets** (fake accuracy): RealChain i2i 13.89, RRDataset re-digitised 47.31, ForenSynths
StyleGAN 51.28, AIGCDetect StyleGAN 56.67, and two Co-Spy-Bench Playground-v2 256px subsets at 40.0.

**Cost: 1.90-4.53 s per image** (per-benchmark means on an RTX A5000), with 49.5-134.9 generated tokens per image on average.
ADB takes 4.4-5.1 s, Veritas++ 16.8-20.6 s.

### Robustness: JPEG quality 95 on every image

All images (reals and fakes) are re-encoded at quality 95 before scoring. The subsets are the same, and the comparators
get the same transform:

| subset | FakeVLM acc clean -> JPEG95 | AUROC change | Veritas++ acc / AUROC change | ADB acc / AUROC change | median of the 27: acc / AUROC change |
|---|---|---:|---:|---:|---:|
| DiTFake | 99.30 -> 98.90 (**-0.40**) | -0.04 | +0.10 / +0.06 | +0.30 / +0.35 | -2.00 / -1.14 |
| UFD (8 diffusion sets) | 97.02 -> 65.67 (**-31.35**) | -12.05 | -5.26 / -5.68 | -0.70 / -0.84 | -2.08 / -1.59 |

On UFD-8 the drop is all on the fakes:
- the real accuracy is 98.61 before and after;
- the mean fake accuracy falls from 95.44 to 32.74. After JPEG95, 5 of the 8 sets have fake accuracy below 50% (LDM-200
  4.76, LDM-200-CFG 4.76, Guided 7.94, LDM-100 14.29);
- the rank of 28 falls from 2nd to 21st.

After re-encoding, the UFD fakes are called real. DiTFake under the same transform barely moves. A **hypothesis, not
tested here**: the model has learned a format or compression cue, so JPEG artifacts read as "real". These are two sets
only, not a general robustness result. Greedy vs sampled does not apply, because the official decoding is greedy.

### Comparison with the paper (arXiv 2503.14905)

**No paper benchmark is held, so there is no reproduction claim and no delta.** The paper evaluates on FakeClue (the
authors' own benchmark and training source; its test images are not held here), LOKI, DMimage, DD-VQA and FF++. None of
these is downloaded (`paper_comparison.fakevlm` in `comparison.json`: every entry "not held"). The 15 subsets above use
our own protocol.

### Reproduce

```bash
python benchmark/bench_eval.py --model fakevlm --benchmark pergen --data <subset-root> --limit <L> --vlm-cache <dir> --tag fakevlm_<bench>
#   add --degrade jpeg95 for the JPEG probe
VLM_MODEL=fakevlm UNITS=vlm:wildrf,vlm:ufd8:jpeg95 sbatch --job-name=fakevlm_bench repro/vlm_bench/vpp_ivi.sbatch   # run_units.py on IVI
python repro/vlm_bench/compare.py        # regenerates comparison.json from the result JSONs
python repro/vlm_bench/compare_invariance.py --old <rev>   # the other columns unchanged; rank_all_* +1 per better new VLM
python repro/vlm_bench/trace_check.py --logs repro/logs   # every result JSON and per-image record against its log
```

## Caveats
- **GenImage is in-distribution**, and the overlap of FakeClue with the other 14 subsets has not been checked. The
  held-out mean (84.15 / 91.16) is the fair headline.
- **No paper reproduction.** None of the paper's benchmarks is held.
- **Fidelity is at verdict level against every official run, and at token level against one.** The token-level
  reference is the official code run one image at a time under 4.45.2, a mode written for this check from `eval.py`'s
  own item, model and `generate` call. `eval.sh`'s batch of 16 and flash-attention 2 were not run.
- **The UFD-8 JPEG95 collapse** (-31.35 acc) is measured on one set. Its cause is not established.
- **Subsets.** All 15 benchmarks are subsampled; none was run on full data.
- **Cache identity.** The `--vlm-cache` folder is keyed by the weights revision, `max_new_tokens`, the prompt, the VLM
  config and a hash of `models/vlm/*.py`. The benchmark jobs wrote code version `02852acf62` (`vlm.identity` in each
  result JSON). The later `num_additional_image_tokens` load check changes `models/vlm/vlm.py` (code version
  `4a950958ed`) but not the scoring. `python repro/vlm_bench/migrate_vlm_cache.py --cache-root <dir> --model fakevlm`
  carries those caches over byte for byte, and the same script carries the ADB and Veritas++ caches of main e5749ae.

## Availability
- Not on the Hugging Face Space: no licence, and a 7B model needs a GPU (15.1 GB peak).
- Not in the model zoo (`NEVER_REDISTRIBUTE` in `synthscope/hub.py`). `synthscope.load("fakevlm")` without the folders
  prints the download commands below.

## Licence
- Code: no licence (no LICENSE file; GitHub reports none). SynthScope vendors only the verdict rule of
  `scripts/eval.py` (`parse_fakevlm`) and copies the prompt.
- Weights (`lingcco/fakeVLM`): no licence tag and no model card. The base LLaVA-1.5-7B / Vicuna-7B v1.5 and the
  llava-hf tokenizer are under the Llama 2 Community License. FakeClue (the training data) is Apache-2.0.
- So: not redistributed (`NEVER_REDISTRIBUTE` in `synthscope/hub.py`), not in the model zoo, not on the Space. Download
  both folders yourself at the pinned revisions:

```bash
hf download lingcco/fakeVLM --revision dce9e88df7d1053e72af197c1bd3c61c94089e96 --local-dir $SYNTHSCOPE_CKPT_DIR/fakevlm
hf download llava-hf/llava-1.5-7b-hf --revision a272c74b2481d8aff3aa6fc2c4bf891fe57334fb --include '*.json' \
    --include 'tokenizer.model' --local-dir $SYNTHSCOPE_CKPT_DIR/llava-1.5-7b-hf_a272c74_processor
```
