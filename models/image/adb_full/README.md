# Align, Don't Blend: full method, VLM + DDA (`adb_full`)

**Paper:** Chen et al., *Align, Don't Blend: Orthogonal Supervision Disentanglement As the Key to Generalizable
AI-Generated Image Detection*, ACM MM 2026 ([arXiv 2512.06746](https://arxiv.org/abs/2512.06746)).
**Code:** [GzZning/Align-Dont-Blend](https://github.com/GzZning/Align-Dont-Blend) @ `5d83e40` (Apache-2.0).
**Weights:** none of its own. It runs two registered detectors:
- [`adb_vlm`](../adb_vlm/README.md): the VLM branch, the authors' LoRA (HF `Gaffeyzz/Align-Dont-Blend-VLM` @ `49975ccd`) on
  Qwen2.5-VL-7B-Instruct @ `cc594898`;
- [`dda`](../dda/README.md): the pixel-artifact expert, the released DDA checkpoint (HF `Junwei-Xi/Dual-Data-Alignment`
  `DDA_ckpt.pth`, which is `dda.pth`).

**Status.** Inference only, with the official weights and the official decision rule. The authors confirmed that the
expert is the released DDA checkpoint (2026-11-20). The official repository now ships the expert's inference
(`detector/expert.py`) and the fused `predict.py`.
- **Fidelity:** identical final, DDA and VLM decisions and VLM text against the unmodified official `predict.py` on 24
  images, in bf16 and in float32 ([Fidelity results](#fidelity-results)).
- **Paper benchmarks** ([Results](#results)):
  - WildRF, full data: 91.22 vs 92.9 (-1.68): below the paper on all three platforms (Twitter -2.10).
  - BFree-Online, 807 of the 1,111 images: 96.93 vs 97.5.
  - AIGI-Bench, 9,000 images: 87.52 vs 88.8.
  - GenImage, 10,000 images (625 + 625 per generator): 86.70 [86.10, 87.29] vs 91.7; DDA alone 89.35.
  - AIGCDetect, subset: 83.82 vs 87.7.
- **Float32 on an RTX A6000 vs bf16 on an RTX A5000** (WildRF): +0.12 accuracy (the precision and the GPU both change).

## Method
The paper keeps two detectors with orthogonal supervision apart and ORs their verdicts:
- a vision-language model trained with DPO to judge **semantic plausibility** (`adb_vlm`);
- a **pixel-artifact** expert: DDA, a DINOv2 ViT-L/14 + LoRA classifier trained on VAE reconstructions aligned with their
  real images.

An image is AI-generated if either detector says so.

## The official decision rule
`predict.py` runs the VLM (`swift infer`) and the expert command of its README,
`python -m detector.expert --checkpoint weights/DDA_ckpt.pth`, on the same files. It then writes per image
(`detector/core.py`):

| field | official definition |
|---|---|
| `vlm_fake` | `parse_vlm_verdict(response)`: the `<answer>` text, or a tag-less final line "This is an AI-generated / authentic image."; anything else raises and stops the script |
| `expert_fake` | `fake_probability >= expert_threshold` (`read_expert_results`; `--expert-threshold`, default **0.5**; a probability of exactly 0.5 is fake) |
| `fake` | `fuse(vlm_fake, expert_fake)` = `vlm_fake or expert_fake` |

There is no tie rule beyond `>=`, no weighting and no score: the official output is three booleans.
`models/image/adb_full/adb_full.py` ports the rule (`fuse`, `expert_decision`, `fused_verdict`). The CPU test checks it
against the official `predict.py` itself (below).

## SynthScope's score (for AUROC / AP)
The official output has no continuous score, so SynthScope defines one. At the official threshold 0.5, its decision is
exactly the official one:

```
s_vlm = max(P_vlm, 0.5)          if the VLM's parsed verdict is "fake"
      = min(P_vlm, 0.5 - ulp)    otherwise
score = max(s_vlm, P_dda)        # fake  <=>  score >= 0.5  <=>  vlm_fake or P_dda >= 0.5
```

- `P_vlm` is `adb_vlm`'s P(fake): the sigmoid of the raw AI-minus-authentic logit gap at the verdict word.
- `P_dda` is DDA's sigmoid. Both are unrounded.
- `0.5 - ulp` is the largest float below 0.5.

`P_vlm` is clamped to the side of 0.5 of its parsed verdict. The decision follows the text, as in the official code, even
where the raw gap disagrees with it: the repetition penalty picks the text, and a bf16 gap of exactly 0 gives P_vlm 0.5
with the verdict "real". Within each side, images still rank by the larger of the two probabilities. The result's
`fake_probability` is this score rounded to 6 decimals; `verdict` comes from the rule, not from the rounded number. So a
"real" image whose score is `0.5 - ulp` shows `fake_probability` 0.5 with the verdict "real". The unrounded score is
`fused_score`.

**Unparsed VLM answers.** The official script stops on an answer `parse_vlm_verdict` rejects, so such an image has no
official decision. SynthScope keeps the image:
- if DDA says fake, the OR is fake whatever the VLM meant, and the verdict is "fake";
- otherwise the verdict is "unparsed" and counts as wrong, as for `adb_vlm`. Its score is `max(0.5 - ulp, P_dda)`.

One answer of the paper benchmarks is unparsed: `SocialRF/1_fake/SocialRF_2365.png` (AIGI-Bench). DDA calls it fake
(P 0.988), so it counts as fake; no other run had one.

## Inference
| step | official (`predict.py`) | SynthScope (`Engine.predict_image`, `predict_kind="fusion"`) |
|---|---|---|
| VLM | `swift infer` (ms-swift; vLLM by default, or `--infer-backend pt`) | `adb_vlm` on the decoded file without an RGB conversion (ms-swift's loading; see its README) |
| expert input | `Image.open(path).convert("RGB")`, `CenterCrop(336)`, `ToTensor`, CLIP mean/std, batches of 8 | `media.to_rgb` (the same conversion), the `dda` registry transform (`center_crop_336, to_dtype, normalize_clip`): tensor-equal, one image at a time |
| expert model | `detector/expert.py` `DDAExpert`: the vendored DINOv2 ViT-L/14 (img 518, patch 14, offset 0.1) + LoRA (rank 8, alpha 1) + linear head on `x_norm_clstoken`; `DDA_ckpt.pth` loaded strictly | `SynthScopeDDAModel` (torch.hub DINOv2 + the vendored DDA LoRA): the same 537 tensors, `dda.pth` = `DDA_ckpt.pth` byte for byte |
| expert decision | `sigmoid(logit) >= 0.5` | the same, on the unrounded P(fake) |
| final | `vlm_fake or expert_fake` | the same; plus the score above |

```python
import synthscope
det = synthscope.load("adb_full")          # needs adb_vlm's two folders and dda.pth in $SYNTHSCOPE_CKPT_DIR
r = det.predict("photo.jpg")               # verdict, fake_probability (score), fused_score, decided_by, components
r["components"]["adb_vlm"]["verdict"], r["components"]["dda"]["fake_probability"]
```

Both components stay loaded: 18.5 GB peak in bf16 (the VLM and DDA), about 5.2 s per image on an RTX A6000.
`decided_by` says which component made an image fake: "vlm", "expert", "both" or "neither".

`bench_eval` refuses the key. A benchmark scores the components separately, so the cached `adb_vlm` per-image results are
reused:
- the VLM: `bench_eval --model adb_vlm --vlm-cache`;
- DDA: `bench_eval --model dda --per-image`, on the same files with the same `--jpeg-fakes` / `--limit` / `--select`.

`repro/adb_full/fuse.py` then fuses the per-image records with `fuse_components`.

## Fidelity
**Identity checks (CPU, `tests/test_adb_full_smoke.py`):**
- **Decision rule.** The unmodified official `predict.py` is run with `--vlm-results` / `--expert-results` on mocked
  component outputs:
  - tagged and tag-less answers with both verdicts;
  - expert probabilities 0, 0.3, the float just below 0.5, exactly 0.5, the float just above, 0.9 and 1;
  - VLM P(fake) on both sides of 0.5, also against its verdict.

  The final, VLM and expert decisions are identical in all 28 cases. The official script stops on an unparsable answer.
- **Expert preprocessing.** The transform built by the official `PixelArtifactExpert` equals the `dda` registry
  transform tensor for tensor, on RGB, RGBA, palette, greyscale, images smaller than 336 (zero-padded), odd sizes, and
  PNG and JPEG files.
- **Expert architecture.** The official `DDAExpert` loads SynthScope's 537 `dda` tensors strictly and gives the same
  logits (random weights, max |diff| 0.0 on CPU).
- **Expert weights.** The official expert checkpoint is SynthScope's `dda.pth`: sha256 `b27a31d39374803ddeff02bfabb2be76e190b04300490cddfafb24f683f37e3e`, equal to the
  HF LFS hash of `DDA_ckpt.pth`.

**End to end on a GPU.** `repro/adb_full_fidelity.py` / `.sbatch`, job 296802 on one RTX A6000. Results:
`repro/adb_full_fidelity.json`, the official outputs in `repro/adb_full_fidelity_official/`, the log
`repro/logs/adb_full_fidelity_296802.out`.

The run used 24 images:
- the 16 of the `adb_vlm` fidelity check (GenImage, SynthWildX, WildRF; RGBA PNGs; one 8 MP JPEG);
- 8 GenImage val images on which the components disagree. They come from a pool of 43 on which the cached `adb_vlm`
  verdict is wrong, scored by `dda`: the 4 + 4 with (VLM real, DDA fake) and (VLM fake, DDA real) closest to 0.5. One
  has DDA P(fake) 0.5116.

The official side is the unmodified `predict.py` with the README's expert command, run twice:
- `--infer-backend pt`: ms-swift 4.4.2, the model's own bf16, as the `adb_vlm` check ran it;
- the README's full command: `--torch-dtype float32 --attn-impl eager --device-map auto`.

The official expert was also run alone, for its probabilities. SynthScope's side is `Engine.predict_image` on the
registry config (bf16, SDPA), then with the VLM in float32 + eager attention.

**Environment.** The official VLM runs in the ms-swift reference folder of the `adb_vlm` check, which has the README's
ms-swift 4.4.2, peft 0.19.1 and qwen-vl-utils 0.0.14. It uses the SynthScope interpreter: Python 3.11, torch 2.11,
transformers 5.5.4 and Pillow 12.1.1, where the README pins Python 3.12, torch 2.6, transformers 5.12.1 and Pillow
11.3. `predict.py`'s default vLLM backend was not compared.

| check (24 images) | bf16 vs official pt backend | float32 eager vs the README command |
|---|---|---|
| final decision | 24/24 identical | 24/24 identical |
| DDA decision | 24/24 | 24/24 |
| VLM decision | 24/24 | 24/24 |
| VLM text | 24/24 identical | 24/24 identical |
| score >= 0.5 exactly when fake | 24/24 | 24/24 |
| DDA P(fake) vs the official expert | max abs diff 2.1e-6 | the same |
| VLM / DDA disagreements (official) | 10: expert-only fake 5, VLM-only fake 5 | 7: 4 and 3 |

- **Speed and memory** (engine, both components resident): 5.21 s per image and 18.49 GB peak in bf16; 5.93 s and
  38.14 GB in float32.
- **The two official runs disagree with each other.** bf16 and the README's float32 command give the same VLM text on
  5 of 24 images, the same VLM verdict on 21/24 and the same final decision on 22/24. The bf16 greedy text also
  depends on the GPU: on the 16 shared images, this A6000 run and the A5000 run of the `adb_vlm` check (job 293634) give
  the same text on 7/16. Two VLM verdicts flip: the Midjourney fake is "AI-generated" on the A5000 and "authentic"
  here; the RGBA Reddit real goes the other way. SynthScope matches the official code exactly at a given precision and
  GPU, but the VLM verdict itself is not stable across precisions or GPU models.
- The 24 images are not a benchmark: 8 were chosen because the VLM is wrong on them. The final decision is correct on
  18 of 24, the VLM on 13 and DDA on 23.

## Results
**The paper's benchmarks, full method, official weights** (SynthScope did not train either component). Balanced
accuracy, mean over the paper's subsets. Per image, the `adb_vlm` record (bf16, greedy) is fused with the `dda` record of
the same file by the official rule (`repro/adb_full/fuse.py`). Every VLM row ran on RTX A5000s, the GPU model of the
committed `adb_vlm` runs. Sources: `repro/adb_full/paper_comparison.json` and `repro/adb_full/results/<row>.json`.

| benchmark (paper table) | data (n images) | paper | SynthScope `adb_full` | delta | AUROC | VLM alone | DDA alone |
|---|---|---:|---:|---:|---:|---:|---:|
| WildRF (Table 1) | **full** test split (2,503) | 92.9 | **91.22** | -1.68 | 99.22 | 93.98 | 90.20 |
| BFree-Online (Table 1) | all held images: 269 real + 538 fake of the 1,111 listed (807) | 97.5 | **96.93** | -0.57 | 99.11 | 87.08 | 94.98 |
| AIGI-Bench CommunityAI + SocialRF (Table 1) | SocialRF **full** (6,000); CommunityAI the first 1,500 + 1,500 of 6,000 + 6,000 (3,000) | 88.8 | **87.52** | -1.28 | 97.35 | 89.65 | 82.41 |
| GenImage (Table 6), fakes JPEG q96 | **10x sample**, 625 + 625 per generator (10,000); [below](#genimage-10x-larger-sample-10000-images) | 91.7 | **86.70** | -5.00 | 92.86 | 61.05 | 89.35 |
| GenImage, the earlier subset | 63 + 63 per generator (1,008; a prefix of the 10,000) | 91.7 | 85.12 | -6.58 | 92.23 | 65.97 | 87.30 |
| AIGCDetectBenchmark (Table 8), fakes JPEG q96 (**inferred protocol**) | **subset**, 30 + 30 per generator (1,020) | 87.7 | **83.82** | -3.88 | 90.05 | 63.92 | 86.86 |
| CO-SPY-Bench in-the-wild (Table 1) | not held | 94.3 | — | | | | |

Per subset:
- WildRF: Facebook 90.94 / Reddit 89.33 / Twitter 93.40 (paper 92.2 / 90.9 / 95.5).
- AIGI-Bench: CommunityAI 91.20 / SocialRF 83.83 (paper 91.9 / 85.6).

**Status.** WildRF is the one like-for-like row. It is **below** the paper: -1.68 on the mean, inside the 2-point
tolerance, but every platform is below (Facebook -1.26, Reddit -1.57, Twitter -2.10). This is labelled "below", as VINA's
WildRF row is. BFree-Online and AIGI-Bench are close (-0.57, -1.28) but not on the paper's exact data:
- BFree-Online: 807 of the 1,111 listed images are still online;
- AIGI-Bench: CommunityAI is a quarter of its images, chosen as the first 1,500 per class.

GenImage is a 10,000-image sample: 625 per class and generator, about a tenth of GenImage val. AIGCDetect is a
30-per-class subset of the VLM benchmark campaign. Neither is a reproduction. On both, DDA alone is above the fused
number (GenImage 89.35 vs 86.70, AIGCDetect 86.86 vs 83.82): the OR adds the VLM's false positives on reals. DDA on the
full data gives 89.37 and 87.84 ([`dda` README](../dda/README.md)).

**What the OR does.** By construction, the fused real accuracy is below either component's and the fused fake accuracy
above it. On full WildRF (all 2,503 images pooled), the real accuracy is 91.4 for the VLM alone, 89.0 for DDA alone and
81.5 fused; the fake accuracy rises to 99.8 fused. The VLM alone is
above the full method on WildRF (93.98) and AIGI-Bench (89.65). DDA alone is above it on GenImage and AIGCDetect.

**Notes.**
- **GenImage.** The paper JPEG-compresses the fakes "following BFree" without a quality factor; q96 is our choice
  (DDA's protocol). The paper's AlignGemini per-generator values in Tables 6 and 8 share numbers in permuted columns, so
  compare the averages only.
- **AIGCDetect.** The paper states the fake-JPEG step for GenImage only; applying it here is our inference.
- **One unparsed VLM answer** (a SocialRF fake): the official script would stop on it. DDA calls it fake, so the OR is
  fake (see [the rule](#synthscopes-score-for-auroc--ap)). No other row has one.
- **File identity.** The two components scored the same files. For new runs the sha1 of each file is checked against
  the VLM record (AIGI-Bench, BFree-Online). For committed runs it is checked against the subset manifests (GenImage,
  AIGCDetect, the 769-image BFree subset). The WildRF headline row uses the committed VLM records, which carry no sha1
  (2,503 unchecked in `file_identity`), so its identity is indirect:
  - the VLM run read a copy of ivi-cn029's `WildRF/test`, sha1-list identical to it when it was staged;
  - the DDA run read another such copy, and its records carry each file's sha1;
  - the float32 run's records also carry sha1s, and all 2,503 equal DDA's (`wildrf_full_fp32.json`).
- **Reused VLM records.** WildRF, GenImage and AIGCDetect use the committed `adb_vlm` per-image records, as they are.
  BFree-Online reused 769 cached records on the same node after `migrate_vlm_cache.py` (rule `unchanged`). They are
  identical to the committed ones, and 38 images were new. AIGI-Bench was scored new; its 80 images shared with the
  committed slice give identical verdicts and logits (A5000 on ivi-cn027 vs ivi-cn026).
- **Rounded logits.** P_vlm is the sigmoid of the record's logit, which `bench_eval` rounds to 4 decimals. No decision
  changes (the VLM decision is its parsed text, DDA's probability is unrounded); AUROC / AP can change only through ties.
- **GPU and precision.** bf16 greedy text depends on the GPU model (A5000 vs A6000: the same text on 7 of 16 fidelity
  images). Every row above is A5000.

### GenImage, 10x larger sample (10,000 images)
GenImage val, the first 625 real + 625 fake per generator (the sorted rule of `bench_eval --limit 625`), fakes JPEG
q96. The 63-per-class subset above is a prefix of this sample: its 1,008 committed VLM records were reused. They were
seeded into the cache by `repro/adb_full/seed_vlm_cache.py`, each file's sha1 checked against the subset manifest. The
other 8,992 images were scored new. Run: job 297930, two RTX A5000 on ivi-cn029, VLM bf16 greedy, 6.0 h. Every record
carries its file's sha1, and all 10,000 VLM and DDA sha1s are equal.

Bootstrap 95% CIs (`repro/adb_full/bootstrap.py`, `bootstrap_genimage_10k_jpegfakes96.json`): 2,000 replicates, stratified
by generator and class, paired across the three detectors.

| detector (10,000 images) | balanced acc [95% CI] | mean AUROC [95% CI] |
|---|---:|---:|
| `adb_full` (VLM OR DDA) | **86.70** [86.10, 87.29] | 92.86 [92.26, 93.47] |
| VLM alone (`adb_vlm`) | 61.05 [60.36, 61.72] | 74.72 [73.76, 75.66] |
| DDA alone (`dda`) | 89.35 [88.85, 89.84] | 98.35 [98.12, 98.57] |
| `adb_full` - DDA (paired) | -2.65 [-3.08, -2.24] | -5.49 [-6.07, -4.93] |
| `adb_full` - VLM (paired) | +25.65 [+24.97, +26.33] | +18.14 [+17.31, +18.97] |

| generator | paper (Table 6) | `adb_full` [95% CI] | real / fake acc | VLM alone | DDA alone |
|---|---:|---:|---|---:|---:|
| ADM | 77.2 | 81.52 [79.52, 83.68] | 88.96 / 74.08 | 48.24 | 85.60 |
| BigGAN | 98.4 | 80.08 [77.92, 82.16] | 89.92 / 70.24 | 49.84 | 82.48 |
| GLIDE | 90.3 | 87.60 [85.68, 89.28] | 93.12 / 82.08 | 56.32 | 87.84 |
| Midjourney | 89.3 | 92.96 [91.52, 94.32] | 89.12 / 96.80 | 78.32 | 95.36 |
| SD v1.4 | 89.6 | 94.32 [93.04, 95.52] | 89.12 / 99.52 | 64.96 | 98.72 |
| SD v1.5 | 95.6 | 95.12 [93.92, 96.24] | 91.04 / 99.20 | 68.72 | 98.48 |
| VQDM | 98.1 | 66.48 [64.16, 68.80] | 90.08 / 42.88 | 52.08 | 67.44 |
| Wukong | 98.3 | 95.52 [94.40, 96.64] | 91.04 / 100.00 | 69.92 | 98.88 |
| **mean** | **91.7** | **86.70** [86.10, 87.29] | | 61.05 | 89.35 |

- **Against the paper:** 86.70 vs 91.7 (-5.00). The paper's value is outside the CI. Four generators are more than 2
  points below their printed column: BigGAN -18.3, VQDM -31.6, Wukong -2.8 and GLIDE -2.7. So the row is "reference"
  (a subsample that does not meet the tolerance). The paper's per-generator AlignGemini values share numbers with its
  AIGCDetect table in permuted columns (see Notes above), so the per-generator comparison is weak evidence; the mean is
  the comparable number.
- **Against the 1,008-image subset:** 85.12 [83.04, 87.10] there, 86.70 [86.10, 87.29] here. The larger sample moves the
  full method by +1.58, inside the subset's CI, and narrows the CI about 3.4x. On the same images the VLM alone falls
  from 65.97 to 61.05 and DDA alone rises from 87.30 to 89.35.
- **DDA for context:** DDA alone gives 89.35 on these 10,000 images and 89.37 on the full GenImage data
  ([`dda` README](../dda/README.md)), so the sample is representative for DDA. The full method is 2.65 points below DDA
  alone (paired CI [-3.08, -2.24]). Of the 5,000 reals, DDA calls 4,897 real; the VLM turns 382 of those into fakes.
  Of the 5,000 fakes, the VLM catches only 117 that DDA misses (VQDM fake accuracy: 36.64 for DDA alone, 42.88 fused).
- No unparsed answers.

### Precision spot check: float32 (the official README's command)
WildRF full data, the VLM in float32 + eager attention on an RTX A6000 (job 297123), fused with the same DDA records:

| VLM precision (GPU) | adb_full acc | AUROC | VLM alone | Facebook / Reddit / Twitter |
|---|---:|---:|---:|---|
| bf16, SDPA (RTX A5000; the registry default) | 91.22 | 99.22 | 93.98 | 90.94 / 89.33 / 93.40 |
| float32, eager (RTX A6000; the README command) | 91.34 | 99.25 | 94.00 | 90.94 / 89.67 / 93.40 |

72 of 2,503 VLM verdicts (2.9%) and 41 final decisions (1.6%) differ between the two runs. The changes nearly cancel:
+0.12 accuracy, +0.03 AUROC. The precision and the GPU change together here, so this measures both.

### Reproduce
```bash
sbatch --nodelist=ivi-cn027 repro/adb_full/bench_a5000.sbatch   # dda --per-image on every row + the new bf16 VLM units
sbatch --nodelist=ivi-cn023 repro/adb_full/fp32_a6000.sbatch    # WildRF full, VLM float32 + eager
sbatch --nodelist=ivi-cn029 repro/adb_full/genimage10k_a5000.sbatch   # GenImage 625 + 625 per generator (seeds the cache)
python repro/adb_full/fuse.py | tee repro/logs/adb_full_fuse.out
python repro/adb_full/bootstrap.py genimage_10k_jpegfakes96 genimage_subset_jpegfakes96   # 95% CIs
python repro/vlm_bench/trace_check.py --logs repro/logs --glob 'adb_ivi_*.out,adbfull_*.out' --results repro/adb_full/vlm
python repro/vlm_bench/trace_check.py --logs repro/logs --glob 'adbfull_bench_*.out,adbfull_gi10k_*.out' --results repro/adb_full/dda
python repro/vlm_bench/trace_check.py --logs repro/logs --glob 'adb_full_fuse.out' --results repro/adb_full
```

## Availability and licence
- Code and VLM adapter: Apache-2.0 (the repository's `LICENSE`, added 2026-11-20; the adapter's HF card at `ff7973df`,
  whose files are byte-identical to the pinned `49975ccd`). DDA: Apache-2.0. Base model: Apache-2.0.
- Not on the Hugging Face Space: a 7B VLM needs a GPU.
- Nothing to mirror: `adb_full` has no weights of its own. The `adb_vlm` adapter is not mirrored to the model zoo (see
  [LICENSES.md](../../../LICENSES.md)).
