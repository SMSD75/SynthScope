# SynthScope Web Interface

A web app to run the SynthScope detectors on an uploaded **image or video** and get the probability
that it is **AI-generated (fake)** vs **real**. Pick any detector from a menu; the backend loads
the corresponding model + checkpoint and runs real inference.

```
interface/
├── backend/            FastAPI service (model registry, inference engine, media decoding)
│   ├── app.py          HTTP API + static serving
│   ├── registry.py     per-detector config / preprocessing / checkpoint / output convention
│   ├── engine.py       lazy model build, checkpoint load, LRU cache, predict
│   ├── media.py        image/video decoding (reuses the repo's own decoders)
│   └── config.py       paths, caches, device, limits (env-overridable)
├── frontend/           Vite + React SPA (build where Node is available)
├── static/index.html   no-build React page (CDN) the backend serves by default
├── scripts/            checkpoint collection / training / promotion
├── requirements.txt    web deps (added on top of the `AGID` conda env)
└── run_backend.sh      launch the backend
```

## Quick start (on the GPU node, e.g. cn023)

```bash
# 1. one-time: web deps into the AGID conda env
conda run -n AGID pip install -r interface/requirements.txt

# 2. assemble the central checkpoint dir (symlinks the available official weights)
conda run -n AGID python interface/scripts/collect_checkpoints.py

# 3. run the backend (serves the no-build UI at http://HOST:8000)
bash interface/run_backend.sh
```

Then open `http://<node>:8000` (use SSH port-forwarding from your laptop, e.g.
`ssh -L 8000:cn023:8000 …`). The browser fetches React from a CDN; the model inference happens
on the node.

### Optional: the full Vite/React build
Where Node is available (the cluster nodes don't have it):
```bash
cd interface/frontend
npm install
npm run build        # emits frontend/dist, which the backend serves automatically
# or for live dev with hot reload (proxies /api to the backend on :8000):
npm run dev
```

## How it works

* Every detector is a SynthScope `nn.Module` built exactly like `main.py` does, via
  `config_utils.compose_config(<hydra overrides>)` + the same `model_type` dispatch as
  `trainer._build_model`.
* The checkpoint is loaded into that module. Official weights (FatFormer, AIDE, NPR) are remapped
  to the SynthScope state-dict layout (the verified remaps from `repro/*_eval.py`); trained-from-scratch
  SynthScope checkpoints load directly.
* Preprocessing matches each model's training/eval recipe (see `registry.py`).
* The fake probability is the model's native positive-class score: `softmax(logits)[fake]` for the
  cross-entropy heads, `sigmoid(logit)` for the BCE heads. Label index **1 = fake**
  (`dataset.py:_get_label`).
* **Video**: native video detectors (D3, ReStraV, STALL) decode a clip and score it directly; image
  detectors sample frames (default 16), score each as an image, and report the **mean** P(fake)
  plus a per-frame breakdown.
* Models load lazily on first request and are kept in a small LRU (`AGID_MAX_RESIDENT_MODELS`,
  default 2) so the GPU is not exhausted by the large backbones.

## Detectors (16) & helper scripts

13 image (`clip`, `cnn`, `dinov2`, `dinov3`, `adapted_dino`, `safe`, `fat_former`, `co_spy`, `aide`,
`spai`, `vib_net`, `npr`, `pgc`) + 3 video (`d3`, `restrav`, `stall`). Per-model held-out numbers and the
checkpoint-zoo table are in the top-level [README](../README.md#web-interface--checkpoint-zoo).

| script | purpose |
| --- | --- |
| `scripts/collect_checkpoints.py` | symlink official weights into `$AGID_CKPT_DIR` + write `MANIFEST.md` |
| `scripts/train_model.sh <key> <gpus> <port> [epochs]` | train a SynthScope-class detector (saving on) |
| `scripts/promote_checkpoint.py <key>` | publish the best trained epoch as `<key>.pth` |
| `scripts/verify_model.py <key>` | quick held-out check on ForenSynths |
| `scripts/build_restrav_data.py` | stage GenVideo into the image-style `<split>/<gen>/{0_real,1_fake}` layout |
| `scripts/calibrate_d3.py` | calibrate D3's score→probability |
| `scripts/restrav_repro.py` | **faithful** ReStraV reproduction (in-distribution + cross-generator) |

Notes:
* **STALL** is training-free; it ships the official VATEX calibration npz and can recompute it on your own
  real videos (`ModelConfig.recompute_params=true`). It and the **`dinov3`** probe need the gated **DINOv3
  ViT-L/16** weights (set `DINO_V3_WEIGHTS` / `DINO_V3_WEIGHTS_DIR`).
* **ReStraV** uses raw `[0,1]` input + standardized 21-d features (matching the official repo); see the
  README footnote on in-distribution vs cross-generator results.

## API

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/api/health` | – | status, device, checkpoints dir |
| GET | `/api/models` | – | detectors + availability |
| POST | `/api/predict` | multipart `file`, `model` | `{fake_probability, real_probability, verdict, …}` |
| GET | `/api/datasets` | – | on-disk datasets + per-model training dataset |
| GET | `/api/datasets/{key}/sample` | query `split`, `n`, `seed` | random balanced batch `{items:[{rel,label,modality}]}` |
| GET | `/api/datasets/{key}/media` | query `rel`, `thumb` | the file (or a JPEG thumbnail / video middle-frame) |
| POST | `/api/predict_dataset` | form `dataset`, `rel`, `model` | prediction on a dataset file + `ground_truth`, `correct` |

```bash
curl -s -X POST http://localhost:8000/api/predict \
  -F model=fat_former -F file=@/path/to/image.png | python -m json.tool
```

The UI has two tabs: **Detect** (upload an image/video) and **Data Explorer** (pick a dataset —
the one used to train the chosen detector is starred — sample a random batch of thumbnails, click any
item to run the detector and see prediction vs. ground truth, and "↻ New batch" to resample). Datasets
are auto-discovered from disk (`backend/datasets.py`), so only those present on the node appear.

## Checkpoints

Central directory (env `AGID_CKPT_DIR`, default `/hddstore/ssalehi/AGID/checkpoints`). The menu
shows a model as available only when its checkpoint file is present (D3 is training-free and always
available). See `MANIFEST.md` there for status.

* `collect_checkpoints.py` — symlink available official weights to canonical names + write manifest.
* `train_model.sh <key> <gpus> <port> [epochs]` — train a missing model with saving enabled.
* `promote_checkpoint.py <key>` — publish the best trained epoch as `<key>.pth`.

## Configuration (env vars)

| Var | Default | Meaning |
| --- | --- | --- |
| `AGID_CKPT_DIR` | `/hddstore/ssalehi/AGID/checkpoints` | central checkpoint directory |
| `AGID_CACHE_DIR` | `/hddstore/ssalehi/AGID/.cache` | HF/torch backbone cache |
| `AGID_MAX_RESIDENT_MODELS` | `2` | models kept in GPU memory (LRU) |
| `AGID_VIDEO_IMAGE_FRAMES` | `16` | frames sampled when an image detector runs on a video |
| `AGID_MAX_UPLOAD_MB` | `200` | upload size limit |
| `AGID_PORT` / `AGID_HOST` | `8000` / `0.0.0.0` | backend bind |

## Deployment notes

* Checkpoints, datasets and backbone caches live on **node-local** storage (cn023 `/hddstore`), so
  run the backend on that node, or set `AGID_CKPT_DIR`/`AGID_CACHE_DIR` to wherever the weights are.
* First use of a model downloads its pretrained backbone (CLIP/DINOv2/SigLIP/…) into the cache.
* GPU is auto-detected; falls back to CPU (slow for the large backbones).

## Configuring SOTA generator API keys

The "SOTA vs Detectors" demo calls external image generators (Nano Banana / GPT Image / Grok / FLUX).
Keys are read from environment variables — **never commit them**:

```bash
cp .env.example .env        # at the repo root; .env is gitignored
# edit .env and fill in the keys you have:
#   GEMINI_API_KEY=...   OPENAI_API_KEY=...   XAI_API_KEY=...   FAL_KEY=...
```

The backend auto-loads `.env` at startup (`interface/backend/config.py`). Real shell exports or a
secrets manager override the file. A generator whose key is unset shows greyed-out "(no API key)" in
the UI; the rest of the demo (real-image serving, detector scoring) works without any key. For
production, prefer real env vars / a secrets manager over the `.env` file.

**Local Ideogram 4.0 (open-weight, no paid API key):** needs `Ideogram4Pipeline`, currently only in diffusers `main`:
```bash
pip install "git+https://github.com/huggingface/diffusers.git"
```
and a free `HF_TOKEN` (once) to download the license-gated nf4 weights (accept at hf.co/ideogram-ai/ideogram-4-nf4-diffusers). It then runs locally on a 24 GB GPU.

### Local open-weight generators: FLUX.2 [klein] 4B and Qwen-Image-2.1 (no paid API key)

Two image-**conditioned** generators run locally and edit the uploaded photo into a new image:

| key | model | size | license | notes |
|---|---|---|---|---|
| `flux2_klein` | `black-forest-labs/FLUX.2-klein-4B` | 24 GB | Apache-2.0 | distilled, 4 steps (~1 min incl. load) |
| `qwen_image_2_1` | `Qwen/Qwen-Image-2.1` (7B unified generation + editing) | 33 GB | **Qwen Research License (research use only)** | 40 steps (~1.5 min incl. load) |

They need a newer `diffusers` than the shared env (`QwenImage21Pipeline`), so they run in an **isolated
subprocess** (`interface/backend/local_gen.py`) with their own library path — the shared `AGID` env is left
untouched. One-time setup (no env changes):

```bash
G=/path/to/gen_env && mkdir -p $G
pip install --no-deps --target $G git+https://github.com/huggingface/diffusers
pip install --no-deps --target $G "huggingface_hub>=1.12,<2" hf_xet      # diffusers main needs hub 1.x; hub 2.x is rejected by transformers
export AGID_GEN_PYTHONPATH=$G                                           # prepended to the child's PYTHONPATH
export AGID_GEN_HF_HOME=/node-local/hf_cache                            # 57 GB of weights: keep them off the shared quota
```

Both are ungated, but a `HF_TOKEN` in `.env` is passed through. A generator shows as available only when
`AGID_GEN_PYTHONPATH` points at an existing directory **and** a GPU is visible.

How it fits a 24 GB GPU next to the detectors: the web engine first releases its resident detectors
(`Engine.release_gpu`, they reload lazily); then the child runs one image and exits, freeing everything.
FLUX.2 uses `enable_model_cpu_offload`. Qwen-Image-2.1 (33 GB, would need >28 GB of host RAM that way) is
loaded **in stages** straight to the GPU — text encoder (17 GiB) → dropped → 7B transformer (14 GiB) → moved to
host RAM just before a single-pass VAE decode — so host RAM stays small. (VAE *tiling* is deliberately not used:
tiling the encode of the conditioning image corrupts it, and tiled decode leaves visible seams a detector could
key on.) Generation runs off the event loop, and local runs are serialized by a lock.

Other FLUX variants (`FLUX.2-klein-9B`, `FLUX.1-Kontext-dev`, `FLUX.2-dev`) are gated on the Hub: accept the licence
on the model page with the `HF_TOKEN` account, then add an entry to `MODELS` in `local_gen.py` and `GENERATORS` in
`generators.py`. FLUX 3 *Image* has no public open weights at the time of writing (only FLUX 3 Action does).
