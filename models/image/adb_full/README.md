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
  - WildRF, full data: 91.22 vs 92.9 (-1.68, within 2 points).
  - BFree-Online, 807 of the 1,111 images: 96.93 vs 97.5.
  - AIGI-Bench, 9,000 images: 87.52 vs 88.8.
  - GenImage and AIGCDetect, subsets: 85.12 vs 91.7 and 83.82 vs 87.7.
- **Float32 vs bf16** (WildRF): +0.12 accuracy.

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

There were no unparsed answers in any `adb_vlm` run so far.

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
| GenImage (Table 6), fakes JPEG q96 | **subset**, 63 + 63 per generator (1,008) | 91.7 | **85.12** | -6.58 | 92.23 | 65.97 | 87.30 |
| AIGCDetectBenchmark (Table 8), fakes JPEG q96 (**inferred protocol**) | **subset**, 30 + 30 per generator (1,020) | 87.7 | **83.82** | -3.88 | 90.05 | 63.92 | 86.86 |
| CO-SPY-Bench in-the-wild (Table 1) | not held | 94.3 | — | | | | |

Per subset:
- WildRF: Facebook 90.94 / Reddit 89.33 / Twitter 93.40 (paper 92.2 / 90.9 / 95.5).
- AIGI-Bench: CommunityAI 91.20 / SocialRF 83.83 (paper 91.9 / 85.6).

**Status.** WildRF is the one like-for-like row; it is within 2 points (the repository's tolerance), 1.68 below the
paper. BFree-Online and AIGI-Bench are close (-0.57, -1.28) but not on the paper's exact data:
- BFree-Online: 807 of the 1,111 listed images are still online;
- AIGI-Bench: CommunityAI is a quarter of its images, chosen as the first 1,500 per class.

GenImage and AIGCDetect are subsets of the VLM benchmark campaign: 63 or 30 images per class and generator, the
affordable VLM cost. They are not reproductions. On these two sets DDA alone is above the fused number (87.30 vs 85.12,
86.86 vs 83.82): the OR adds the VLM's false positives on reals. DDA on the full data gives 89.37 and 87.84
([`dda` README](../dda/README.md)).

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
  AIGCDetect, the 769-image BFree subset). For WildRF, both runs copied ivi-cn029's `WildRF/test`, each sha1-list
  identical to it.
- **Reused VLM records.** WildRF, GenImage and AIGCDetect use the committed `adb_vlm` per-image records, as they are.
  BFree-Online reused 769 cached records on the same node after `migrate_vlm_cache.py` (rule `unchanged`). They are
  identical to the committed ones, and 38 images were new. AIGI-Bench was scored new; its 80 images shared with the
  committed slice give identical verdicts and logits (A5000 on ivi-cn027 vs ivi-cn026).
- **GPU and precision.** bf16 greedy text depends on the GPU model (A5000 vs A6000: the same text on 7 of 16 fidelity
  images). Every row above is A5000.

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
python repro/adb_full/fuse.py | tee repro/logs/adb_full_fuse.out
python repro/vlm_bench/trace_check.py --logs repro/logs --glob 'adb_ivi_*.out,adbfull_*.out' --results repro/adb_full/vlm
python repro/vlm_bench/trace_check.py --logs repro/logs --glob 'adbfull_bench_*.out' --results repro/adb_full/dda
python repro/vlm_bench/trace_check.py --logs repro/logs --glob 'adb_full_fuse.out' --results repro/adb_full
```

## Availability and licence
- Code and VLM adapter: Apache-2.0 (the repository's `LICENSE`, added 2026-11-20; the adapter's HF card at `ff7973df`,
  whose files are byte-identical to the pinned `49975ccd`). DDA: Apache-2.0. Base model: Apache-2.0.
- Not on the Hugging Face Space: a 7B VLM needs a GPU.
- Nothing to mirror: `adb_full` has no weights of its own. The `adb_vlm` adapter is not mirrored to the model zoo (see
  [LICENSES.md](../../../LICENSES.md)).
