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

## 2. Datasets: they download and organize themselves

You do not have to find benchmark data or rearrange folders. Each supported dataset has a config in
[`conf/dataset/`](../conf/dataset/) that records its official source, and the dataset's loader **downloads it, unpacks it safely
(zip/tar extraction guards against path tricks) and restructures it into the one layout every SynthScope tool reads**:

```
<root>/
  <generator_a>/0_real/*.png     <generator_a>/1_fake/*.png
  <generator_b>/0_real/*.png     <generator_b>/1_fake/*.png
```

The evaluators, the training loop and the web page's *Browse datasets* tab all rely on that layout, which is why no per-dataset
glue code is needed. We check this end to end: downloading CommGen15 (about 26,000 images, 4 GB) with the command below took
about two minutes, produced `test/<generator>/{0_real,1_fake}` for 15 generators, and `bench_eval.py` scored it straight away.

```bash
python benchmark/download_dataset.py --list                                       # every dataset and where it comes from
python benchmark/bench_eval.py --model npr --dataset commgen15 --root /data/CommGen15   # download if missing, then evaluate
python benchmark/download_dataset.py commgen15 --root /data/CommGen15             # only download + lay out
```

`bench_eval.py --dataset NAME` is the one-step version: it downloads the dataset when no images are under `--root` (or the
`root_dir` in the dataset's yaml), finds the folder to evaluate (`<root>/test`, `<root>/test_full` or the root itself), picks the
grouping profile from the layout (UnivFD and ForenSynths have their own), and runs the evaluation. `--no-download` makes it fail
instead of fetching, and `--dry-run` only shows what it resolved. An incomplete download is not detected: delete the folder to
fetch it again.

Training works the same way: a dataset config with `prefetch: true` fetches its data the first time it is used, into
`DatasetConfig.root_dir`. Downloads can be large (CommGen15 about 10 GB on the Hub, DiTFake about 22 GB, AIGIBench about
220 GB), so check your disk first. Re-running skips data that is already in place.

Datasets with an **Auto** download below are fetched by the loader. A few video sets (GenVideo, VidProM, VideoPhy, Physics-IQ)
have no automatic source: get them from their official sites and place them in the same layout.

| Dataset (config name) | What it is | Source (download) |
| --- | --- | --- |
| `cnndetection` | ForenSynths: ProGAN training set and the GAN test sets | Google Drive (`source_url` in the yaml) |
| `universialfakedetection` | UnivFD test sets (LDM, Glide, DALL-E, guided) | Google Drive |
| `genimage`, `genimage_full` | GenImage, 8 generators | Hugging Face `jzousz/GenImage` (Google Drive is quota-limited) |
| `aigibench` | AIGIBench | Hugging Face `HorizonTEL/AIGIBench` |
| `aigcdetect` | AIGCDetectBenchmark, 17 generators, full test data (76.3k real / 76.3k fake, 34 GB) | ModelScope `aemilia/AIGCDetectionBenchmark` |
| `cospy` | Co-Spy-Bench | Hugging Face `ruojiruoli/Co-Spy-Bench` |
| `synthbuster` | Synthbuster | Zenodo record `10066460` |
| `ditfake`, `commgen15` | DiTFake, CommGen15 | Hugging Face (`lioooox/DiTFake`, `xiaoyuzhou68/CommGen15`) |
| `realchain` | RealChain (REM; used by CuRe): 7 generators × 1000 fakes and 7000 reals, all JPEG q60. CC BY-NC 4.0. The reals are one unpaired pool, so the loader splits them into 7 disjoint slices of 1000 (the mean over generators then equals the pooled balanced accuracy) | Hugging Face `handsomerich/RealChain` |
| `evalgen` | EvalGEN (DDA; used by DDA and CuRe): Flux, GoT, Infinity, OmniGen, NOVA, about 11,060 fakes each, no reals. The loader pairs every generator with MSCOCO val2017 (5000 reals), our choice: CuRe only says "a shared MSCOCO pool". `fake_acc` matches DDA's fakes-only accuracy | Hugging Face `Junwei-Xi/EvalGEN` + `images.cocodataset.org` |
| `synthwildx` | SynthWildX (GRIP; used by DDA and CuRe): 500 real + 500 each DALL-E 3 / Firefly / Midjourney v5, all posted on X. Only URLs are released, so the loader fetches each image from X; posts deleted since then return 404 and are listed in `<root>/missing.txt` (in October 2026, 1713 of 2000 were still online) | `data/synthwildx/list.csv` in `grip-unina/ClipBased-SyntheticImageDetection` |
| `wildrf` | WildRF (LaDeDa paper; used by DDA and CuRe): test split, one folder per platform (Facebook 160/160, Reddit 750/750, Twitter 341/342 real/fake; 262 Twitter files are `.jfif`) | Google Drive file from `barcavia/RealTime-DeepfakeDetection-in-the-RealWorld` |
| `bfree_online` | BFree-Online (B-Free's viral images; used by DDA and CuRe): real and fake images that went viral, several web-scraped copies of each (about 1400 in the paper). Only URLs are released; the loader keeps an image only if its md5 matches, so dead links and placeholders are skipped (`<root>/missing.txt`; in October 2026, 807 of the 1111 listed were still online: 269 real, 538 fake). One group, `test/viral` | `viral_images_dataset/BFree_viral_images.csv` in `grip-unina/B-Free` (nonprofit use only) |
| `rrdataset` | RRDataset (ICCV 2025; used by CuRe): three scenarios, `original`, `transfer` (shared over online platforms) and `redigital` (re-digitized), about 8500 real and 8500 fake each. Only the test archive is used | Zenodo record `14963880` (`RRDataset_test.tar.gz`, 20 GB, CC BY 4.0) |
| `arforensics` | ARForensics (ICCV 2025; used by D3QE): images from 7 visual autoregressive generators, LlamaGen, VAR, Infinity, Janus-Pro, RAR, Switti and Open-MAGVIT2, 6000 fakes and 6000 ImageNet reals each. The corrected test release (MAR replaced by Switti) | HF `Yanran21/ARForensics` (`ARForensics_updated_test_part00{1..4}.zip`, 21 GB, MIT) |
| `diffusionforensics`, `diffusion1kstep`, `freqnetdetection` | DiffusionForensics, Diffusion-1kStep, FreqNet sets | see the yamls |
| `genvideo` | GenVideo-Val (DeMamba; D3 Table 1, STALL): 10 generators vs 10,000 MSR-VTT reals, shared reals hard-linked into every generator | ModelScope `cccnju/Gen-Video` (`GenVideo-Val.zip`, 14 GB) |
| `videophy` | VideoPhy test set (D3 Table 3): 3360 clips from 10 generators, released as URLs; dead links are skipped (both CogVideoX sets were offline in October 2026). Fake-only, so paired with GenVideo-Val's MSR-VTT reals, as D3 does | `videophysics/videophy_test_public` (MIT) + ModelScope for the reals |
| `vidprom` | VidProM (D3 Table 4, ReStraV): ModelScope, Pika, Text2Video-Zero, VideoCrafter2. The full set is terabytes, so the loader streams the first 1,000 clips of tar part 1 per generator (results are on that subsample); paired with MSR-VTT reals. OpenSora and StreamingT2V, in D3's table, are not in the repository | `WenhaoWang/VidProM` (CC BY-NC 4.0) |
| Video: `genvidbench`, `evalcrafter`, `videofeedback`, `comgenvid`, `physics_iq` | the other video benchmarks | Hugging Face where a repo is set; otherwise place the files by hand |

**If a download fails.** Hosts sometimes rate-limit (Google Drive) or ask you to accept terms first. Fetch the data by hand from the
source in the table and unpack it so it matches the layout above (`real/`+`fake/` and `nature/`+`ai/` folders are also
understood; see the docstring of [`bench_eval.py`](../benchmark/bench_eval.py)). Datasets are large, so keep them on fast
local disk.

**Results on the in-the-wild sets.** [WILD_BENCHMARKS.md](WILD_BENCHMARKS.md) has every image detector on `realchain`,
`evalgen`, `synthwildx`, `wildrf`, `bfree_online` and `rrdataset` (built from `repro/bench_results/` by `repro/build_wild_table.py`).

---

## 3. Reproduce a paper's numbers

```bash
python benchmark/bench_eval.py --model clip --dataset cnndetection --root /data/CNNDetection   # download if needed + evaluate
# the explicit form, for data you arranged yourself:
python benchmark/bench_eval.py --model clip --benchmark forensynths --data /data/CNNDetection/test --tag clip_forensynths
```

With `--dataset` the grouping below is chosen for you; with `--data` you pick `--benchmark` yourself:

| `--benchmark` | Layout it expects | Typical dataset |
| --- | --- | --- |
| `forensynths` | `<root>/<generator>/.../{0_real,1_fake}`, grouped by the top-level generator | `cnndetection` |
| `ufd` | fakes per generator, reals paired the UnivFD way (guided->imagenet, others->laion) | `universialfakedetection` |
| `pergen` | one `<generator>/{0_real,1_fake}` folder each, optional shared `--reals` | AIGIBench, FreqNet, Co-Spy-Bench |
| `pool` | one group: `<root>/{real,fake}` | GenImage (per subset) |

Useful flags: `--ckpt FILE` to score a different checkpoint (official vs SynthScope-trained side by side), `--select a,b` to
limit the generators, `--limit N` to cap images per class (quick checks), `--native-res` to score at the original resolution,
`--bs` for the batch size. Video: `python benchmark/bench_eval_video.py --model d3 --dataset genvideo --root /data/GenVideo --limit 1000` downloads
the dataset if needed and evaluates it (`--limit 1000` is D3's official protocol: at most 1000 videos per class per generator;
each generator also reports `ap_real`, D3's AP with real videos as the positive class).

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

Measured on one RTX A5000 (24 GB) with 100 real + 100 generated images at 1024 px:

| Step | Time |
| --- | --- |
| FLUX.2 [klein] 4B, generate 100 images | about 15 minutes |
| FLUX.2 [klein] 9B, generate 100 images | about 40 minutes |
| Qwen-Image-2.1, generate 100 images | about 70 minutes |
| Score all image detectors on the 200 images | 5 to 10 minutes |
| Download a dataset | minutes for CommGen15 (4 GB took about 2 minutes), hours for the largest (AIGIBench is about 220 GB) |
| A paper benchmark (`bench_eval.py`) | minutes for small sets, hours for GenImage-size sets (use `--limit` to try it) |

Generation times are for a GPU that is not shared with other jobs on the same host; they grow when the node is busy.

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
