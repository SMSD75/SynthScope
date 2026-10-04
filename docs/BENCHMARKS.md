# Setting up and running the benchmarks

SynthScope has **three kinds of benchmark**. They answer different questions, use different commands, and feed different
places in the web page. Pick the one you need:

| I want to... | Run | Result appears in |
| --- | --- | --- |
| **Check a detector against its paper** (does our port reproduce the published numbers?) | [`benchmark/bench_eval.py`](../benchmark/bench_eval.py) (images), [`bench_eval_video.py`](../benchmark/bench_eval_video.py) (video) | the page's **Leaderboard** tab, the tables in each model's `README.md` |
| **See how detectors hold up against a particular image generator** (FLUX, Qwen, Nano Banana, ...) | [`benchmark/generator_bench.py`](../benchmark/generator_bench.py) | the page's **Generator benchmarks** tab |
| **Submit a number to the standardized, versioned leaderboard** | [`benchmark/evaluate.py`](../benchmark/evaluate.py) | [`LEADERBOARD.md`](../LEADERBOARD.md), see [`benchmark/README.md`](../benchmark/README.md) |

Everything below assumes you installed SynthScope (`pip install -e .`, see the [README](../README.md#-install)) and, for the
GPU steps, have a CUDA GPU. Detectors run on CPU too, but the larger ones (AIDE, SPAI, CuRe) are slow there.

---

## 1. Get the detector weights

A detector needs its checkpoint before it can be scored.

```bash
synthscope list                       # the AVAIL column shows which detectors have weights on this machine
export SYNTHSCOPE_CKPT_DIR=/path/to/checkpoints     # where <key>.pth files live (default ~/.cache/synthscope/checkpoints)
```

- **Official weights** (NPR, FatFormer, AIDE, SPAI, PGC, CuRe, ...) come from the authors; each model's
  `models/<modality>/<key>/README.md` says where to download them and what to name the file. Their licenses are your
  responsibility ([`LICENSES.md`](../LICENSES.md)).
- **SynthScope-trained weights**: train with the recipe in the README's *Train your own* section, or fetch released ones with
  `synthscope pull --model <key>` once a model-zoo repo is configured (`SYNTHSCOPE_HUB_REPO`).
- Some detectors need extra backbones: SPAI needs the MFM backbone (`SYNTHSCOPE_MFM_CKPT`), CuRe the Perception-Encoder
  base (`SYNTHSCOPE_PE_CORE_CKPT`, or it downloads `facebook/PE-Core-L14-336`), the DINOv3 probe and STALL the gated DINOv3
  weights (`DINO_V3_WEIGHTS`, `DINO_V3_WEIGHTS_DIR`).

---

## 2. Datasets for the paper benchmarks

Every dataset has a config in [`conf/dataset/`](../conf/dataset/) that records where the data comes from. List them:

```bash
python benchmark/download_dataset.py --list
```

| Dataset (config name) | What it is | Source |
| --- | --- | --- |
| `cnndetection` | ForenSynths: ProGAN training set and the GAN test sets | Google Drive (`source_url` in the yaml) |
| `universialfakedetection` | UnivFD test sets (LDM, Glide, DALL-E, guided) | Google Drive |
| `genimage`, `genimage_full` | GenImage, 8 generators | Hugging Face `jzousz/GenImage` (Google Drive is quota-limited) |
| `aigibench` | AIGIBench | Hugging Face `HorizonTEL/AIGIBench` |
| `aigcdetect` | AIGCDetectBenchmark, 16 generators | ModelScope `aemilia/AIGCDetectionBenchmark` |
| `cospy` | Co-Spy-Bench | Hugging Face `ruojiruoli/Co-Spy-Bench` |
| `synthbuster` | Synthbuster | Zenodo record `10066460` |
| `ditfake`, `commgen15` | DiTFake, CommGen15 | Hugging Face (`lioooox/DiTFake`, `xiaoyuzhou68/CommGen15`) |
| `diffusionforensics`, `diffusion1kstep`, `freqnetdetection` | DiffusionForensics, Diffusion-1kStep, FreqNet sets | see the yamls |
| Video: `genvideo`, `genvidbench`, `evalcrafter`, `videofeedback`, `vidprom`, `comgenvid`, `physics_iq`, `videophy` | the video benchmarks | Hugging Face where a repo is set; otherwise place the files by hand |

**Download one with its built-in loader:**

```bash
python benchmark/download_dataset.py genimage --root /data/GenImage          # download + lay out
python benchmark/download_dataset.py synthbuster --root /data/Synthbuster --dry-run   # only show the source
```

Hosts sometimes rate-limit (Google Drive) or ask you to accept terms first. If a download fails, fetch the data by hand
from the source above and unpack it so it matches this layout, which is what the evaluators read:

```
<root>/
  <generator_a>/0_real/*.png     <generator_a>/1_fake/*.png
  <generator_b>/0_real/*.png     <generator_b>/1_fake/*.png
```

(`real/`+`fake/`, `nature/`+`ai/` and shared real folders are also understood; see the docstring of
[`bench_eval.py`](../benchmark/bench_eval.py).) Datasets are large, so keep them on fast local disk.

---

## 3. Reproduce a paper's numbers

```bash
python benchmark/bench_eval.py --model clip --benchmark forensynths \
    --data /data/CNNDetection/test --tag clip_forensynths
```

`--benchmark` selects how the folders are grouped:

| `--benchmark` | Layout it expects | Typical dataset |
| --- | --- | --- |
| `forensynths` | `<root>/<generator>/.../{0_real,1_fake}`, grouped by the top-level generator | `cnndetection` |
| `ufd` | fakes per generator, reals paired the UnivFD way (guided->imagenet, others->laion) | `universialfakedetection` |
| `pergen` | one `<generator>/{0_real,1_fake}` folder each, optional shared `--reals` | AIGIBench, FreqNet, Co-Spy-Bench |
| `pool` | one group: `<root>/{real,fake}` | GenImage (per subset) |

Useful flags: `--ckpt FILE` to score a different checkpoint (official vs SynthScope-trained side by side), `--select a,b` to
limit the generators, `--limit N` to cap images per class (quick checks), `--native-res` to score at the original resolution,
`--bs` for the batch size. Video: `python benchmark/bench_eval_video.py --model d3 --benchmark genvideo --data ... --tag ...`.

Each run prints per-generator accuracy, AP and AUROC with the mean, writes `repro/bench_results/<tag>.json` and appends a
line to `repro/MASTER_RESULTS.jsonl`, so every reported number traces to a run. `python benchmark/aggregate_results.py`
rebuilds the combined table `repro/RESULTS_TABLE.md`.

**Match the paper's protocol before judging a gap.** Read which subsets the paper evaluates and how it averages them.
For example CuRe reports the 20 full-image-synthesis subsets of AIGIBench, not all 25; scoring all 25 looked like a 7-point
miss until we used the paper's subsets. The input handling also matters: each detector's registered preprocessing mirrors its
paper's evaluation (a different resize or crop can destroy artifact-based detectors).

### How a result reaches the Leaderboard tab

1. The paper's numbers and the benchmark list for a detector live in [`repro/benchmark_specs/<key>.json`](../repro/benchmark_specs/).
   Add a spec file for a new detector (copy `cure.json`).
2. Add your measured numbers to the `VAL` table in [`interface/backend/build_repro_full.py`](../interface/backend/build_repro_full.py).
   Every value must trace to a `repro/bench_results/*.json` file.
3. Rebuild and reload the page:

```bash
python interface/backend/build_repro_full.py          # writes interface/backend/repro_full.json
```

A row's status is computed, not typed: **Reproduced** (our own trained checkpoint is within the tolerance of the paper),
**Official weights match** (the authors' weights run through SynthScope match), **Below paper**, **Reference only** (the paper
reports no comparable number), **Not evaluated**. The tolerance is 2.0 points for accuracy and mAP, 2.5 for AUROC.

---

## 4. Benchmark against an image generator

The *Generator benchmarks* tab compares detectors on images from one generator. The recipe is the same for every generator:

1. Take real photos (people are skipped) and write a prompt for each: a BLIP caption plus a photographic style suffix.
2. Let the generator **create a new image from the prompt alone**, so every pixel is synthesized (no editing of the photo).
3. Score the real and the generated images with every detector and rank by AUROC.

```bash
# the same 100 real images and prompts for every generator, so generators are compared on identical content
python benchmark/generator_bench.py --generator flux2_klein --label "FLUX.2 [klein] 4B" \
    --reuse benchmark/generator_runs/reference/manifest.json --real-root /data/GenImage_full --real-root /data/CNNDetection \
    --out benchmark/generator_runs/flux2_klein_4b
```

The reference set ([`benchmark/generator_runs/reference/manifest.json`](../benchmark/generator_runs/reference)) lists 100 prompts and the
file names of the real images they were written for. The images themselves are third-party data and are not in the repo:
`--real-root` finds them by file name in your own GenImage and CNNDetection downloads (section 2), so your run uses the same
content as ours. Any real images you cannot match are skipped.

- `--generator` is any key from [`interface/backend/generators.py`](../interface/backend/generators.py) that can create an
  image from text (`flux2_klein`, `flux2_klein_9b`, `qwen_image_2_1`, `nanobanana`, `gpt_image`, `grok`, `flux`, `ideogram`).
- **Local open-weight generators** (`flux2_klein`, `flux2_klein_9b`, `qwen_image_2_1`) need a 24 GB GPU and a newer
  `diffusers` than the main environment; the setup (`SYNTHSCOPE_GEN_PYTHONPATH`, gated-model licenses, `HF_TOKEN`) is in
  [`interface/README.md`](../interface/README.md). They run as one batch with a single model load.
- **Hosted generators** need their API key in `.env` (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY`, `FAL_KEY`) and cost
  money per image; the benchmark prints and skips any image the provider refuses.
- **Your own real images** instead of the reference set: `--real-glob "/data/real/*.jpg" --n 100` (repeat the flag for
  several folders). Prompts are then captioned fresh, so numbers are only comparable between runs on the same images.
- `--detectors clip,safe,...` limits the detectors; `--skip-generate` re-scores images that are already there; an interrupted
  run resumes because existing images are skipped.

The run writes `manifest.json`, `results.json` and `RESULTS.md` (commit these) plus `real/` and `fake/` image folders
(git-ignored). The Generator benchmarks tab picks up every folder under `benchmark/generator_runs/` automatically; reload
the page after a run.

**What the numbers mean.** AUROC is threshold-free (50 is chance, 100 perfect). *Fakes caught* and *Reals flagged* use the
detectors' 0.5 threshold, which is not tuned per generator, so accuracy can look poor even when AUROC is high. Real photos
are JPEG-origin and of varying size, while the generated images are 1024x1024 PNGs, which can help size- or
format-sensitive detectors; treat small differences with care.

---

## 5. Time and hardware

Measured on one RTX A5000 (24 GB) with 100 real + 100 generated images:

| Step | Time |
| --- | --- |
| FLUX.2 [klein] 4B, generate 100 images | about 1 hour |
| FLUX.2 [klein] 9B, generate 100 images | see the generator benchmarks page notes |
| Qwen-Image-2.1, generate 100 images | see the generator benchmarks page notes |
| Score all image detectors on 200 images | about 10 minutes |
| A paper benchmark (`bench_eval.py`) | minutes for small sets, hours for GenImage-size sets (use `--limit` to try it) |

Host RAM matters for the local generators: give a job 48-64 GB (the larger models are held in host RAM and streamed to
the GPU). Models are downloaded once into the Hugging Face cache (tens of GB each), so point
`SYNTHSCOPE_GEN_HF_HOME` at a disk with room.

---

## 6. Add your own

- **A detector**: follow [`docs/ADD_DETECTOR.md`](ADD_DETECTOR.md); then add it to the Leaderboard as in section 3.
- **A dataset**: add `conf/dataset/image/<name>.yaml` with the source (`hf_repo_id`, `source_url`, ...), make it follow the
  layout in section 2, and add a profile to `bench_eval.py` if its grouping is new.
- **A generator**: add a function `call(image_png | None, prompt, size=None) -> png bytes` and one entry in `GENERATORS` in
  `interface/backend/generators.py`. `image_png=None` means "create from the prompt alone". Then run section 4 with
  `--generator <your key>`.
- **A number for the standardized leaderboard**: see [`benchmark/README.md`](../benchmark/README.md).
