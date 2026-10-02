# CuRe (`cure`)

**Learning Continuous Source Responses for Generalizable AI-Generated Image Detection** —
[paper (arXiv 2609.14316)](https://arxiv.org/abs/2609.14316) · [official code](https://github.com/manic-cui/CuRe)

CuRe reframes backbone adaptation as **regressing real↔generated mixing ratios** instead of binary
classification (finer supervision), with a compact *source-response subspace* that suppresses nuisance
variation and shortcut cues. The released detector is trained **only on GenImage SD‑v1.4** and
reports **89.7 % average balanced accuracy over ten public benchmarks** (+5.2 pts over the second best).

## Pipeline (official, reproduced exactly)

```
image ─ PNG? → JPEG q96 round-trip ─ SmallestMaxSize(390) → CenterCrop(390) → Resize(336)  (cubic)
      ─ JPEG q70 (cv2, on the RGB array) ─ (x − 127.5)/127.5
      ─ PE-Core L/14 @336 (frozen) + Q/V LoRA   → mean of the 576 patch tokens   [B, 1024]
      ─ (feat − mu) @ W                          → [B, 128]    (source-response subspace)
      ─ Linear(128, 2)                           → logits;  P(fake) = softmax(logits)[:, 1]
```

`cure.pth` is the official **25 MB adapter** (`cure_adapter.pt`, SHA-256 checked) unchanged: LoRA
updates, `mu`, `W` and the head. The frozen base (`facebook/PE-Core-L14-336`, 2.7 GB, Apache-2.0) is
fetched from the Hugging Face Hub on first use (or `ModelConfig.base_checkpoint` / `$AGID_PE_CORE_CKPT`).

## AGID adaptations

| choice | reason |
|---|---|
| `official_arch.py` = the official backbone, **byte-identical body** | The adapter loads key-for-key; Apache-2.0 attribution in the header. |
| Preprocessing in cv2/PIL (`preprocess.py`), not albumentations | No new dependency. Equal to the official pipeline up to **1 float32 ulp (5.96e-8)** on 104 PNG/JPEG images, 200 px–4256 px, square and non-square (`repro/cure_preproc_equiv.py`). |
| **Source-format hint** `img.info["agid_src_format"]` | Official PNG sources get an extra JPEG‑q96 round-trip, decided by the file suffix. `bench_eval` and the web decoder now record the original format on the PIL image; without the hint an image is treated as non-PNG. |
| fp16 autocast on CUDA inside `forward` | The released results use fp16 AMP; CPU runs in fp32 (scores differ in the 3rd decimal). |
| `compute_loss` = plain cross-entropy | **The paper's two-stage mixing-ratio-regression training recipe is not in the official release.** `compute_loss` only satisfies AGID's trainer interface (LoRA + head trainable); it does **not** reproduce the paper's training. The AGID-trained row below is therefore empty. |

## Fidelity check (identical to the official model)

`repro/cure_verify_load.py` loads the official `cure` package and AGID's class with the same adapter
and base, on an A5000: all **395 state-dict tensors identical**, all 100 adapter tensors consumed
(`unexpected=0`), **max |logit diff| 4.8e-6**, **max |P(fake) diff| 5.4e-7, 0 verdict flips** over 60
images (official preprocessing vs AGID preprocessing, each end to end). Log: `repro/logs/cure_verify_245812.out`.

## Results (official weights through AGID; balanced accuracy at 0.5 unless noted)

Paper: **89.7 avg over ten benchmarks** — AIGCDetect 89.3 · AIGIBench 92.8 · UnivFD 82.0 · EvalGEN 98.1 ·
SynthWildX 92.7 · WildRF 95.8 · BFree-Online 90.9 · RRDataset 81.5 · RealChain 79.7 · Chameleon 93.8.

| Benchmark (AGID) | generators | acc (mean of per-gen balanced) | mAP | mAUROC | Paper | Status |
|---|---|---:|---:|---:|---|---|
| UFD-8 (diffusion/DALL·E half of UnivFD) | 8 | **93.94** | 99.19 | 98.98 | UnivFD 82.0 | **not comparable** (see below) |
| ForenSynths-13 (GAN half) | 13 | **81.95** | 94.05 | 95.02 | UnivFD 82.0 | **not comparable** |
| ForenSynths-8 ∪ UFD-8 (16-gen "UnivFD-style") | 16 | 89.27 | – | – | UnivFD 82.0 | aggregation differs from the paper |
| GenImage-8 (`--limit 1500`/class; CuRe's *training* set) | 8 | 85.18 | 97.30 | 97.00 | not in the paper's ten | sanity (in-distribution) |
| AIGIBench (all 25 test generators, `--limit 1500`/class) | 25 | **85.95** | 90.72 | 91.43 | 92.8 | **below the paper** (−6.8); see below |
| AIGIBench without the 5 face-swap generators (post-hoc subset) | 20 | 93.13 | 97.14 | 97.59 | 92.8 | observation only — the paper's subset/aggregation is unknown |
| AIGCDetect (17 gens) | 17 | *not run* | | | 89.3 | data lives only on cn023 (its GPUs/CPUs were fully booked) |
| EvalGEN, SynthWildX, WildRF, BFree-Online, RRDataset, RealChain, Chameleon | – | *not in AGID* | | | see above | datasets not available here |

**Honest reading.** The port is verified identical to the official model (above), and the in-distribution
behaviour is exactly what the paper's training implies — SD‑v1.4 **99.73**, SD‑v1.5 **99.80**, Wukong
**99.87** — while other families are weaker at the fixed 0.5 threshold, with high AP/AUROC (e.g. GenImage
ADM/BigGAN/VQDM fake-recall 22–65 %, AUROC 87–99). **However, the paper's headline benchmark numbers are
NOT yet reproduced:** only 3 of its 10 benchmarks exist in AGID, the paper's UnivFD aggregation is not
specified (AGID's UFD-8 and ForenSynths-13 splits give 93.9 and 82.0, the 16-generator union 89.3, versus the
paper's 82.0), AIGIBench (all 25 test generators, 75k images at `--limit 1500`/class) gives **85.95**, i.e. 6.8 pts
below the paper's 92.8; the gap is almost entirely the five face-swap generators (BlendFace 45.9,
FaceSwap 52.2, InSwap 53.6, SimSwap 57.0, E4S 77.4 — real-acc ~90–97 but fake-recall 3–66 %), while the other 20
generators average 93.13 (AP 97.1, AUROC 97.6). That subset was chosen *after* seeing the gap, and the
paper's AIGIBench subset/aggregation is not specified here (AIGIBench's own README lists 23 fake subsets,
AGID's test split has 25), so 93.13 is an observation, not a confirmation. AIGCDetect could not be run (its
data lives only on cn023, whose GPUs/CPUs were fully booked). Treat the numbers above as AGID-protocol
measurements, not confirmation of the paper's table.

Per-generator tables (balanced acc / AP / AUROC / real-acc / fake-acc), from `repro/bench_results/cure_repro_*.json`:

**UFD-8** — dalle 97.45 / 99.73 / 99.66 · glide_100_10 95.25 / 99.30 / 99.10 · glide_100_27 93.20 / 99.17 / 99.01 ·
glide_50_27 92.65 / 98.93 / 98.62 · guided 79.25 / 96.68 / 95.80 · ldm_100 97.55 / 99.89 / 99.89 ·
ldm_200 97.95 / 99.91 / 99.92 · ldm_200_cfg 98.20 / 99.88 / 99.86

**ForenSynths-13** — progan 95.20 · stylegan 87.15 · stylegan2 79.11 · biggan 93.50 · cyclegan 70.36 ·
stargan 92.90 · gaugan 94.21 · deepfake 64.46 · crn 72.34 · imle 74.62 · san 75.11 · seeingdark 72.78 ·
whichfaceisreal 93.60 (the low accuracies are fake-recall: cyclegan 41.8, deepfake 29.7; AP/AUROC are 95–99 on those)

**GenImage-8** — sdv1.4 99.73 · sdv1.5 99.80 · wukong 99.87 · midjourney 85.83 · glide 85.33 · vqdm 82.43 ·
biggan 67.80 · adm 60.63

## Usage

```bash
agid detect photo.png --model cure          # needs cure.pth in $AGID_CKPT_DIR (official adapter) + HF access
python benchmark/bench_eval.py --model cure --benchmark ufd --data <UFD/test> --tag cure_repro_ufd
```

`cure.pth` = copy of the official `weights/cure_adapter.pt`. Use the registered transform (`CuReTransform`):
the PNG→JPEG96 step and the final JPEG‑q70 pass are part of the method.
