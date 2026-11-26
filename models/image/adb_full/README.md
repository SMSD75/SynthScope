# Align, Don't Blend: full method, VLM + DDA (`adb_full`)

**Paper:** Chen et al., *Align, Don't Blend: Orthogonal Supervision Disentanglement As the Key to Generalizable
AI-Generated Image Detection*, ACM MM 2026 ([arXiv 2512.06746](https://arxiv.org/abs/2512.06746)).
**Code:** [GzZning/Align-Dont-Blend](https://github.com/GzZning/Align-Dont-Blend) @ `cf168db` (Apache-2.0; first integrated
at `5d83e40`, see [Upstream sync](#upstream-sync-cf168db)).
**Weights:** none of its own. It runs two registered detectors:
- [`adb_vlm`](../adb_vlm/README.md): the VLM branch, the authors' LoRA (HF `Gaffeyzz/Align-Dont-Blend-VLM` @ `49975ccd`) on
  Qwen2.5-VL-7B-Instruct @ `cc594898`;
- [`dda`](../dda/README.md): the pixel-artifact expert, the released DDA checkpoint (HF `Junwei-Xi/Dual-Data-Alignment`
  `DDA_ckpt.pth`, which is `dda.pth`).

**Status.** Inference only, with the official weights and the official decision rule of `predict.py` @ `cf168db`. The
authors confirmed that the expert is the released DDA checkpoint (2026-11-20).
- **Fidelity:** identical final, DDA and VLM decisions and VLM text against the unmodified official `predict.py` @
  `cf168db` on 24 images, in bf16 and in float32 ([Fidelity](#fidelity)).
- **Paper benchmarks**, VLM bf16 greedy on RTX A5000 ([Results](#results)):
  - WildRF, full data: **93.01** vs 92.9 (+0.11): within tolerance on the mean; Twitter is still 2.10 below.
  - AIGI-Bench CommunityAI + SocialRF, full data (18,000): **89.22** vs 88.8 (+0.42): within tolerance.
  - BFree-Online, 807 of the 1,111 images: 96.84 vs 97.5.
  - GenImage, 10,000 images (625 + 625 per generator): 88.86 [88.32, 89.38] vs 91.7; DDA alone 89.35.
  - AIGCDetect, 3,400 images (100 + 100 per generator): 84.76 [83.82, 85.74] vs 87.7; DDA alone 85.94.
- **The upstream sync** (new prompt, decoding and pixel cap) raised the fused accuracy on four of five benchmarks, paired
  on the same images: WildRF +1.79, AIGI-Bench (9,000) +1.97, GenImage +2.16, AIGCDetect (1,020) +0.88 (CI includes 0);
  BFree-Online -0.09 ([Before and after](#before-and-after-the-upstream-sync)).

## Method
The paper keeps two detectors with orthogonal supervision apart and ORs their verdicts:
- a vision-language model trained with DPO to judge **semantic plausibility** (`adb_vlm`);
- a **pixel-artifact** expert: DDA, a DINOv2 ViT-L/14 + LoRA classifier trained on VAE reconstructions aligned with their
  real images.

An image is AI-generated if either detector says so.

## Upstream sync (cf168db)
SynthScope was first built against `5d83e40`. Three later upstream commits change the inference; SynthScope follows all
three (branch `fix/adb-upstream-sync`):

| upstream commit | change | where in SynthScope |
|---|---|---|
| `2f0adbc` "Use the DPO user prompt for inference" | the task prompt is the user message, directly after `<image>`; the system message is "You are a helpful assistant."; `--max_pixels` 1,210,000 (was 1,003,520); `--repetition_penalty 1.0 --top_p 1.0` (before, the base model's 1.05 applied) | `adb_vlm` (`models/image/adb_vlm/adb_vlm.py`); details in its [README](../adb_vlm/README.md#upstream-sync-cf168db) |
| `898b607` "Default unparsed VLM answers to Real before OR fusion" | an answer `parse_vlm_verdict` rejects is Real (`vlm_fake: false`, `vlm_answer_parsed: false`); before, the script stopped | `adb_vlm`'s parser `parse_adb_default_real`; `fused_verdict` reads a None VLM verdict as Real |
| `cf168db` "Add in-memory JPEG96 evaluation preprocessing" | `tools/jpeg96.py` `prepare_image(image, dataset, label)`: RGB; for GenImage and AIGCDetectBenchmark only, the **fake** images re-encoded in memory as PIL JPEG quality 96 (Pillow's default 4:2:0 subsampling) and decoded; real images not recompressed; applied before both branches | `bench_eval --jpeg-fakes 96`: the same PIL save and decode, after the VLM's `swift_rgb` and the expert's RGB conversion (the same `convert("RGB")`). Pixel-identical to `prepare_image` for both branches (`tests/test_adb_full_smoke.py` [F]) |

So the fake-only JPEG q96 step on GenImage and AIGCDetect, which SynthScope had applied before as its own choice (q96
following DDA) and its own inference (AIGCDetect), is now the official protocol. Every earlier GenImage / AIGCDetect row
already used it, so only the VLM request changes between the rows compared below. The other benchmarks get no
preprocessing beyond the RGB conversion.

The VLM cache identity changed with the request (prompt, decoding, pixel cap, parser, and the code version of
`models/vlm/*.py`, now `bbad7ab02d`), so no record written with the old request is reused; `migrate_vlm_cache.py` refuses
to carry any folder whose prompt or decoding differs.

## The official decision rule
`predict.py` runs the VLM (`swift infer`) and the expert command of its README,
`python -m detector.expert --checkpoint weights/expert.pth`, on the same files. It then writes per image:

| field | official definition |
|---|---|
| `vlm_fake` | `parse_vlm_verdict(response)`: the `<answer>` text, or a tag-less final line "This is an AI-generated / authentic image."; **False (Real) when it raises** |
| `vlm_answer_parsed` | whether `parse_vlm_verdict` returned a verdict |
| `expert_fake` | `fake_probability >= expert_threshold` (`read_expert_results`; `--expert-threshold`, default **0.5**; a probability of exactly 0.5 is fake) |
| `fake` | `fuse(vlm_fake, expert_fake)` = `vlm_fake or expert_fake` |

There is no tie rule beyond `>=`, no weighting and no score: the official output is booleans.
`models/image/adb_full/adb_full.py` ports the rule (`fuse`, `expert_decision`, `fused_verdict`). The CPU test checks it
against the official `predict.py` itself, unparsed answers included (below).

## SynthScope's score (for AUROC / AP)
The official output has no continuous score, so SynthScope defines one. At the official threshold 0.5, its decision is
exactly the official one:

```
s_vlm = max(P_vlm, 0.5)          if the VLM's decision is "fake"
      = min(P_vlm, 0.5 - ulp)    otherwise (also an unparsed answer)
score = max(s_vlm, P_dda)        # fake  <=>  score >= 0.5  <=>  vlm_fake or P_dda >= 0.5
```

- `P_vlm` is `adb_vlm`'s P(fake): the sigmoid of the raw AI-minus-authentic logit gap at the verdict word.
- `P_dda` is DDA's sigmoid. Both are unrounded.
- `0.5 - ulp` is the largest float below 0.5.

`P_vlm` is clamped to the side of 0.5 of the VLM decision. The decision follows the text, as in the official code, even
where the raw gap disagrees with it (a bf16 gap of exactly 0 gives P_vlm 0.5 with the verdict "real"). Within each side,
images still rank by the larger of the two probabilities. The result's `fake_probability` is this score rounded to 6
decimals; `verdict` comes from the rule, not from the rounded number. So a "real" image whose score is `0.5 - ulp` shows
`fake_probability` 0.5 with the verdict "real". The unrounded score is `fused_score`.

**Unparsed VLM answers** are Real, as in `predict.py` since `898b607`; the final decision is then the expert's.
`adb_vlm` reports them (`answer_parsed: false`; `bench_eval` counts them in `n_unparsed`) and keeps their P(fake) on the
Real side (logit at most 0) even if the text holds a verdict word. The result JSONs of job 301129 were written before
that count existed (they show them as `n_unscored`); their unparsed answers are listed in `unparsed_answers.json`. None
of the 287 holds a verdict word, so the clamp changes no committed score. Most of
them are a rationale with no verdict line. With the cf168db request, 287 of the 34,710 benchmark answers below are
unparsed (0.8%: WildRF 19, BFree-Online 11, AIGI-Bench 96, GenImage 113, AIGCDetect 48;
`repro/adb_full/vlm_cf168db/unparsed_answers.json`). With the earlier request there was one, a SocialRF fake.

## Inference
| step | official (`predict.py` @ cf168db) | SynthScope (`Engine.predict_image`, `predict_kind="fusion"`) |
|---|---|---|
| evaluation preprocessing | GenImage / AIGCDetectBenchmark: `prepare_image` (fakes JPEG q96 in memory) | `bench_eval --jpeg-fakes 96`, pixel-identical |
| VLM | `swift infer` (ms-swift; vLLM by default, or `--infer-backend pt`) | `adb_vlm` on the decoded file without an RGB conversion (ms-swift's loading; see its README) |
| expert input | `Image.open(path).convert("RGB")`, `CenterCrop(336)`, `ToTensor`, CLIP mean/std, batches of 8 | `media.to_rgb` (the same conversion), the `dda` registry transform (`center_crop_336, to_dtype, normalize_clip`): tensor-equal, one image at a time |
| expert model | `detector/expert.py` `ExpertModel` (`DDAExpert` before `ff5c349`): the vendored DINOv2 ViT-L/14 (img 518, patch 14, offset 0.1) + LoRA (rank 8, alpha 1) + linear head on `x_norm_clstoken`; checkpoint loaded strictly | `SynthScopeDDAModel` (torch.hub DINOv2 + the vendored DDA LoRA): the same 537 tensors, `dda.pth` = `DDA_ckpt.pth` byte for byte |
| expert decision | `sigmoid(logit) >= 0.5` | the same, on the unrounded P(fake) |
| final | `vlm_fake or expert_fake` | the same; plus the score above |

```python
import synthscope
det = synthscope.load("adb_full")          # needs adb_vlm's two folders and dda.pth in $SYNTHSCOPE_CKPT_DIR
r = det.predict("photo.jpg")               # verdict, fake_probability (score), fused_score, decided_by, components
r["components"]["adb_vlm"]["verdict"], r["components"]["dda"]["fake_probability"]
```

Both components stay loaded: 18.6 GB peak in bf16 (the VLM and DDA), about 6.3 s per image on an RTX A6000.
`decided_by` says which component made an image fake: "vlm", "expert", "both" or "neither".

`bench_eval` refuses the key. A benchmark scores the components separately, so the cached `adb_vlm` per-image results are
reused:
- the VLM: `bench_eval --model adb_vlm --vlm-cache`;
- DDA: `bench_eval --model dda --per-image`, on the same files with the same `--jpeg-fakes` / `--limit` / `--select`.

`repro/adb_full/fuse.py` then fuses the per-image records with `fuse_components`.

## Fidelity
**Identity checks (CPU, `tests/test_adb_full_smoke.py`, against the official clone @ cf168db):**
- **Decision rule.** The unmodified official `predict.py` is run with `--vlm-results` / `--expert-results` on mocked
  component outputs:
  - tagged and tag-less answers with both verdicts, and three answers its parser rejects (no answer tag, an unknown
    answer, two answer tags);
  - expert probabilities 0, 0.3, the float just below 0.5, exactly 0.5, the float just above, 0.9 and 1;
  - VLM P(fake) on both sides of 0.5, also against its verdict.

  The final, VLM and expert decisions and `vlm_answer_parsed` are identical in all 49 cases (21 with an unparsed answer).
- **Evaluation preprocessing.** `bench_eval --jpeg-fakes 96` gives each branch the same pixels as the official
  `prepare_image` (RGB, RGBA, palette and greyscale PNGs, a JPEG; GenImage, AIGCDetectBenchmark and another dataset; real
  and fake): 30 cases.
- **Expert preprocessing.** The transform built by the official `PixelArtifactExpert` equals the `dda` registry
  transform tensor for tensor, on RGB, RGBA, palette, greyscale, images smaller than 336 (zero-padded), odd sizes, and
  PNG and JPEG files.
- **Expert architecture.** The official `ExpertModel` loads SynthScope's 537 `dda` tensors strictly and gives the same
  logits (random weights, max |diff| 0.0 on CPU).
- **Expert weights.** The official expert checkpoint is SynthScope's `dda.pth`: sha256 `b27a31d39374803ddeff02bfabb2be76e190b04300490cddfafb24f683f37e3e`, equal to the
  HF LFS hash of `DDA_ckpt.pth`.
- **VLM request.** The prompts equal `detector/core.py` @ cf168db byte for byte; the resize equals ms-swift 4.4.2 +
  qwen-vl-utils pixel for pixel, including an image in the band where swift's token-rounded cap (1,209,712) matters
  (`tests/test_adb_vlm_smoke.py`).

**End to end on a GPU.** `repro/adb_full_fidelity.py` / `.sbatch`, job 301130 on one RTX A6000 (ivi-cn023). Results:
`repro/adb_full_fidelity.json`, the official outputs in `repro/adb_full_fidelity_official/`, the log
`repro/logs/adb_full_fidelity_301130.out`. (The first run, job 296802 against `5d83e40`, is in the git history; it was
identical on the same 24 images too.)

The run used 24 images:
- the 16 of the `adb_vlm` fidelity check (GenImage, SynthWildX, WildRF; RGBA PNGs; one 8 MP JPEG);
- 8 GenImage val images on which the components disagreed under the earlier request. They come from a pool of 43 on
  which a cached `adb_vlm` verdict is wrong, scored by `dda`: the 4 + 4 with (VLM real, DDA fake) and (VLM fake, DDA
  real) closest to 0.5. The selection is checked equal to the committed one.

The official side is the unmodified `predict.py` @ cf168db with the README's expert command, run twice:
- `--infer-backend pt`: ms-swift 4.4.2, the model's own bf16;
- the README's full command: `--torch-dtype float32 --attn-impl eager --device-map auto`.

The official expert was also run alone, for its probabilities. SynthScope's side is `Engine.predict_image` on the
registry config (bf16, SDPA), then with the VLM in float32 + eager attention.

**Environment.** The official VLM runs in the ms-swift reference folder of the `adb_vlm` check (the README's ms-swift
4.4.2, peft 0.19.1 and qwen-vl-utils 0.0.14) with the SynthScope interpreter: Python 3.11, torch 2.11, transformers 5.5.4
and Pillow 12.1.1, where the README pins Python 3.12, torch 2.6, transformers 5.12.1 and Pillow 11.3. `predict.py`'s
default vLLM backend was not compared.

| check (24 images) | bf16 vs official pt backend | float32 eager vs the README command |
|---|---|---|
| final decision | 24/24 identical | 24/24 identical |
| DDA decision | 24/24 | 24/24 |
| VLM decision | 24/24 | 24/24 |
| `vlm_answer_parsed` | 24/24 (no unparsed answer on these images) | 24/24 |
| VLM text | 24/24 identical | 24/24 identical |
| score >= 0.5 exactly when fake | 24/24 | 24/24 |
| DDA P(fake) vs the official expert | max abs diff 2.1e-6 | the same |
| VLM / DDA disagreements (official) | 8: expert-only fake 6, VLM-only fake 2 | 7: 5 and 2 |

- **Speed and memory** (engine, both components resident): 6.26 s per image and 18.58 GB peak in bf16; 7.24 s and
  39.68 GB in float32.
- **The two official runs disagree with each other.** bf16 and the README's float32 command give the same VLM text on
  only 2 of 24 images, the same VLM verdict on 23/24 and the same final decision on 24/24. SynthScope matches the official
  code exactly at a given precision and GPU, but the VLM text itself is not stable across precisions (and, as the earlier
  check showed, across GPU models).
- The 24 images are not a benchmark. The final decision is correct on 21 of 24, the VLM on 15 and DDA on 23.
- **Not covered end to end by this run:** none of the 24 answers is unparsed, no image falls in the band where the
  token-rounded cap (1,209,712) changes the resize, and JPEG96 is not part of `predict.py`. These three paths are checked
  on the CPU against the official code instead: the decision rule on unparsed answers ([A]), `prepare_image` ([F]) and
  the ms-swift resize (`tests/test_adb_vlm_smoke.py` [C]).

## Results
**The paper's benchmarks, full method, official weights** (SynthScope did not train either component). Balanced
accuracy, mean over the paper's subsets. Per image, the `adb_vlm` record (bf16, greedy, the cf168db request) is fused
with the `dda` record of the same file by the official rule (`repro/adb_full/fuse.py`). All VLM records were scored new
on three RTX A5000 of ivi-cn029 (job 301129, `repro/adb_full/upstream_sync_a5000.sbatch`), the GPU model of every
earlier row; every VLM record carries its file's sha1, equal to the DDA record's (34,710 of 34,710). 95% CIs: bootstrap,
2,000 replicates, stratified by subset and class (`repro/adb_full/bootstrap.py`). Sources:
`repro/adb_full/paper_comparison.json`, `repro/adb_full/results/<row>.json`, `repro/adb_full/bootstrap_<row>.json`.

| benchmark (paper table) | data (n images) | paper | SynthScope `adb_full` [95% CI] | delta | AUROC | VLM alone | DDA alone | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| WildRF (Table 1) | **full** test split (2,503) | 92.9 | **93.01** [91.87, 94.17] | +0.11 | 99.47 | 95.96 | 90.20 | within tolerance |
| AIGI-Bench CommunityAI + SocialRF (Table 1) | **full**, 12,000 + 6,000 (18,000) | 88.8 | **89.22** [88.75, 89.69] | +0.42 | 98.03 | 88.81 | 83.18 | within tolerance |
| BFree-Online (Table 1) | all held images: 269 real + 538 fake of the 1,111 listed (807) | 97.5 | **96.84** [95.44, 98.05] | -0.66 | 99.41 | 81.69 | 94.98 | reference |
| GenImage (Table 6), fakes JPEG q96 | **10x sample**, 625 + 625 per generator (10,000) | 91.7 | **88.86** [88.32, 89.38] | -2.84 | 96.47 | 59.50 | 89.35 | reference |
| AIGCDetectBenchmark (Table 8), fakes JPEG q96 | **sample**, 100 + 100 per generator (3,400) | 87.7 | **84.76** [83.82, 85.74] | -2.94 | 92.93 | 59.06 | 85.94 | reference |
| CO-SPY-Bench in-the-wild (Table 1) | not held | 94.3 | — | | | | | |

Per subset:
- WildRF: Facebook 93.44 / Reddit 92.20 / Twitter 93.40 (paper 92.2 / 90.9 / 95.5).
- AIGI-Bench, full: CommunityAI 92.75 / SocialRF 85.68 (paper 91.9 / 85.6).

**Status.**
- **WildRF and AIGI-Bench reproduce** on the paper's full data: the means are at or above the paper's (+0.11, +0.42).
  On WildRF, Facebook and Reddit are above the paper and Twitter is 2.10 below it (93.40 vs 95.5, unchanged by the sync).
- **BFree-Online** is close (-0.66, the paper's value inside the CI), but only 807 of the 1,111 listed images are still
  online, so it is not the paper's data.
- **GenImage and AIGCDetect** are samples (about a tenth and 2% of the data), not reproductions, and both are about 3
  points below the paper. On both, the fused number is still below DDA alone (paired: GenImage -0.49 [-0.73, -0.27],
  AIGCDetect -1.18 [-1.76, -0.59]): the VLM's false positives on reals outweigh the fakes it adds. DDA alone on the full
  data gives 89.37 and 87.84 ([`dda` README](../dda/README.md)), so the samples are representative for DDA; the gap to
  the paper on these two is not explained here.

### Before and after the upstream sync
The same images scored with the earlier VLM request (upstream `f4b4794` / `5d83e40`: the task prompt as the system
message, repetition penalty 1.05, 1,003,520 pixels) and with the cf168db request, fused with the same DDA records. The
change and its CI are paired: one bootstrap resampling of the shared images drives both rows
(`bootstrap_<row>_cf168db__vs__<row>.json`).

| benchmark (images) | before (f4b4794) | after (cf168db) | change [paired 95% CI] | VLM alone: before -> after (change [CI]) | final decisions changed | paper |
|---|---:|---:|---:|---|---:|---:|
| WildRF full (2,503) | 91.22 | **93.01** | **+1.79** [+0.98, +2.65] | 93.98 -> 95.96 (+1.98 [+0.89, +3.12]) | 85 | 92.9 |
| BFree-Online held (807) | 96.93 | **96.84** | -0.09 [-1.02, +0.84] | 87.08 -> 81.69 (-5.39 [-7.34, -3.35]) | 12 | 97.5 |
| AIGI-Bench, SocialRF full + CommunityAI 1,500 + 1,500 (9,000) | 87.52 | **89.49** | **+1.97** [+1.50, +2.47] | 89.65 -> 90.71 (+1.06 [+0.48, +1.67]) | 434 | 88.8 |
| GenImage 625 + 625 per generator (10,000) | 86.70 | **88.86** | **+2.16** [+1.77, +2.53] | 61.05 -> 59.50 (-1.55 [-2.13, -0.99]) | 380 | 91.7 |
| AIGCDetect 30 + 30 per generator (1,020) | 83.82 | **84.71** | +0.88 [-0.29, +2.06] | 63.92 -> 59.61 (-4.31 [-6.18, -2.45]) | 45 | 87.7 |

AUROC of the fused score, after - before: WildRF +0.25 [+0.07, +0.47], BFree-Online +0.29 [-0.45, +1.09], AIGI-Bench
+0.79 [+0.54, +1.04], GenImage +3.60 [+3.08, +4.14], AIGCDetect +2.56 [+1.08, +4.06].

**What changed.** With the new request the VLM calls fewer reals AI-generated. That is what the OR needs: its fused
real accuracy is capped by the VLM's false positives. On GenImage the fused real accuracy is 94.72-97.44 per generator
(before 88.96-93.12), while the VLM alone loses a little fake recall, so the fused number rises although the VLM-alone
number falls. On BFree-Online the VLM alone drops by 5.39 (its fake accuracy falls to 63.75), but DDA catches most of
those fakes, so the fused number does not move. On WildRF and AIGI-Bench both the VLM alone and the fusion improve.

### The earlier rows (f4b4794 request)
Kept in `paper_comparison.json` (`main: false`, `vlm_request: f4b4794`) for the comparison above: WildRF full 91.22
(float32 + eager on an RTX A6000: 91.34), BFree-Online 96.93 (807) and 97.08 (the 769-image subset), AIGI-Bench 87.52
(9,000) and 81.25 (the 80-image slice), GenImage 86.70 (10,000) and 85.12 (the 1,008-image prefix), AIGCDetect 83.82
(1,020). They used the committed `adb_vlm` records of `repro/vlm_bench/results` and the new VLM units of jobs 297122 and
297930; see the git history of this README for their details.

### GenImage per generator (10,000 images, cf168db request)

| generator | paper (Table 6) | `adb_full` | real / fake acc | VLM alone | DDA alone | before (f4b4794) |
|---|---:|---:|---|---:|---:|---:|
| ADM | 77.2 | 84.64 | 96.16 / 73.12 | 49.92 | 85.60 | 81.52 |
| BigGAN | 98.4 | 81.92 | 95.36 / 68.48 | 50.08 | 82.48 | 80.08 |
| GLIDE | 90.3 | 88.40 | 97.44 / 79.36 | 53.36 | 87.84 | 87.60 |
| Midjourney | 89.3 | 95.60 | 94.72 / 96.48 | 82.08 | 95.36 | 92.96 |
| SD v1.4 | 89.6 | 97.76 | 96.16 / 99.36 | 63.76 | 98.72 | 94.32 |
| SD v1.5 | 95.6 | 97.60 | 96.00 / 99.20 | 63.92 | 98.48 | 95.12 |
| VQDM | 98.1 | 67.20 | 96.48 / 37.92 | 50.48 | 67.44 | 66.48 |
| Wukong | 98.3 | 97.76 | 95.52 / 100.00 | 62.40 | 98.88 | 95.52 |
| **mean** | **91.7** | **88.86** [88.32, 89.38] | | 59.50 | 89.35 | 86.70 |

The paper's per-generator AlignGemini values in Tables 6 and 8 share numbers in permuted columns (e.g. GenImage ADM 77.2
and AIGCDetect VQDM 77.2; GenImage VQDM 98.1 and AIGCDetect SD v1.5 98.1), so compare the averages only. The two
generators far below their printed columns, BigGAN and VQDM, are where DDA alone is weak too (82.48, 67.44).

### AIGCDetect per generator (3,400 images, cf168db request)
Fused / paper, balanced accuracy: ADM 76.5 / 89.3, DALLE2 95.5 / 94.9, Glide 84.0 / 90.3, Midjourney 94.5 / 95.6,
VQDM 64.5 / 77.2, BigGAN 87.0 / 90.4, CycleGAN 75.0 / 74.4, GauGAN 92.5 / 92.5, ProGAN 78.0 / 91.0, SDXL 100.0 / 99.1,
SD v1.4 97.5 / 98.3, SD v1.5 97.5 / 98.1, StarGAN 72.0 / 72.9, StyleGAN 84.0 / 86.6, StyleGAN2 90.0 / 89.2,
WhichFaceIsReal 55.5 / 52.6, Wukong 97.0 / 98.4. The 1,020-image subset of the earlier row is a prefix of this sample
(84.71 on it).

### Notes
- **Precision and GPU.** Every row above is bf16 on RTX A5000; the official README command is float32 + eager. On the
  24 fidelity images the two precisions give the same final decision but different text on 22 of 24, so a float32
  rerun would move each benchmark a little (on WildRF with the earlier request: +0.12).
- **The vLLM backend** (`predict.py`'s default) was not compared.
- **Rounded logits.** P_vlm is the sigmoid of the record's logit, which `bench_eval` rounds to 4 decimals. No decision
  changes (the VLM decision is its parsed text, DDA's probability is unrounded); AUROC / AP can change only through ties.
- **DDA records.** The earlier rows' DDA records are reused (the expert did not change). The two new samples have new
  ones: AIGCDetect 100 + 100 per generator (85.94) and AIGI-Bench CommunityAI full (84.62). On the 1,020 AIGCDetect
  images that two DDA runs share, the decisions are identical and the probabilities differ by at most 4.5e-5 (batching).

### Reproduce
```bash
sbatch --nodelist=ivi-cn029 repro/adb_full/upstream_sync_a5000.sbatch   # VLM (cf168db request) on every row + the new DDA records
sbatch --nodelist=ivi-cn023 repro/adb_full_fidelity.sbatch              # 24 images vs the unmodified predict.py @ cf168db
python repro/adb_full/unparsed_answers.py --cache-root /ssdstore/ssalehi/SynthScope/vpp/cache   # on ivi-cn029
python repro/adb_full/fuse.py | tee repro/logs/adb_full_fuse.out
python repro/adb_full/bootstrap.py wildrf_full_cf168db aigibench_full_cf168db genimage_10k_jpegfakes96_cf168db ...   # 95% CIs
python repro/adb_full/bootstrap.py --pair genimage_10k_jpegfakes96 genimage_10k_jpegfakes96_cf168db                  # paired change
python repro/vlm_bench/trace_check.py --logs repro/logs --glob 'adbsync_301129*.out' --results repro/adb_full/vlm_cf168db
python repro/vlm_bench/trace_check.py --logs repro/logs --glob 'adbsync_301129.out,adbfull_bench_*.out,adbfull_gi10k_*.out' --results repro/adb_full/dda
python repro/vlm_bench/trace_check.py --logs repro/logs --glob 'adb_full_fuse.out' --results repro/adb_full
```

## Availability and licence
- Code and VLM adapter: Apache-2.0 (the repository's `LICENSE`, added 2026-11-20; the adapter's HF card at `ff7973df`,
  whose files are byte-identical to the pinned `49975ccd`). DDA: Apache-2.0. Base model: Apache-2.0.
- Not on the Hugging Face Space: a 7B VLM needs a GPU.
- Nothing to mirror: `adb_full` has no weights of its own. The `adb_vlm` adapter is not mirrored to the model zoo (see
  [LICENSES.md](../../../LICENSES.md)).
