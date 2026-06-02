# AGID Web Interface

A web app to run the AGID detectors on an uploaded **image or video** and get the probability
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

* Every detector is an AGID `nn.Module` built exactly like `main.py` does, via
  `config_utils.compose_config(<hydra overrides>)` + the same `model_type` dispatch as
  `trainer._build_model`.
* The checkpoint is loaded into that module. Official weights (FatFormer, AIDE, NPR) are remapped
  to the AGID state-dict layout (the verified remaps from `repro/*_eval.py`); trained-from-scratch
  AGID checkpoints load directly.
* Preprocessing matches each model's training/eval recipe (see `registry.py`).
* The fake probability is the model's native positive-class score: `softmax(logits)[fake]` for the
  cross-entropy heads, `sigmoid(logit)` for the BCE heads. Label index **1 = fake**
  (`dataset.py:_get_label`).
* **Video**: native video detectors (D3, ReStraV) decode a clip and score it directly; image
  detectors sample frames (default 16), score each as an image, and report the **mean** P(fake)
  plus a per-frame breakdown.
* Models load lazily on first request and are kept in a small LRU (`AGID_MAX_RESIDENT_MODELS`,
  default 2) so the GPU is not exhausted by the large backbones.

## API

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/api/health` | – | status, device, checkpoints dir |
| GET | `/api/models` | – | detectors + availability |
| POST | `/api/predict` | multipart `file`, `model` | `{fake_probability, real_probability, verdict, …}` |

```bash
curl -s -X POST http://localhost:8000/api/predict \
  -F model=fat_former -F file=@/path/to/image.png | python -m json.tool
```

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
