"""FastAPI app for the SynthScope detector web interface.

Endpoints:
  GET  /api/models             -> list detectors + availability
  GET  /api/health             -> liveness + device
  POST /api/predict            -> {file, model} -> fake/real probability

Serves the built Vite frontend from ``frontend/dist`` if present, otherwise the no-build
standalone page in ``static/index.html`` (works without Node).
"""
import asyncio
import json
import tempfile
import threading
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import config, datasets, media
from .engine import get_engine
from .registry import REPRO, REPRO_FULL, get_spec, input_note, list_specs

app = FastAPI(title="SynthScope Detector Interface", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIST = config.INTERFACE_DIR / "frontend" / "dist"
STATIC_DIR = config.INTERFACE_DIR / "static"


@app.get("/api/health")
def health():
    return {"status": "ok", "device": config.device(),
            "checkpoints_dir": str(config.CHECKPOINTS_DIR)}


_TRENDING_PATH = Path(__file__).resolve().parent / "trending.json"


@app.get("/api/trending")
def trending():
    """High-quality trending AIGC-detection papers (image + video). Refreshed daily by a
    scheduled task that updates trending.json; TRENDING.md and the README teaser are regenerated from the same file."""
    try:
        return json.loads(_TRENDING_PATH.read_text())
    except FileNotFoundError:
        return {"updated": None, "image": [], "video": [], "landscape": [], "criteria": ""}


@app.get("/api/models")
def models():
    engine = get_engine()
    out = []
    for spec in list_specs():
        out.append({
            "key": spec.key,
            "name": spec.name,
            "paper": spec.paper,
            "modality": spec.modality,
            "available": engine.is_available(spec),
            "checkpoint": spec.checkpoint,
            "weights": spec.weights,   # "official" | "synthscope_trained" | "training_free"
            "notes": spec.notes,
            "input": input_note(spec),     # {label, explanation}: what the detector does to the image
            "repro": REPRO.get(spec.key),
            "repro_full": REPRO_FULL.get(spec.key),
        })
    return {"models": out, "device": config.device()}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), model: str = Form(...)):
    try:
        spec = get_spec(model)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    engine = get_engine()
    if not engine.is_available(spec):
        raise HTTPException(
            status_code=409,
            detail=f"No checkpoint available for '{spec.key}'. Expected file "
                   f"{spec.checkpoint} under {config.CHECKPOINTS_DIR}.",
        )

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")

    try:
        kind = media.media_kind(file.filename or "", file.content_type)
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e))

    try:
        if kind == "image":
            if spec.modality == "video":
                raise HTTPException(
                    status_code=400,
                    detail=f"'{spec.name}' is a video detector; please upload a video.",
                )
            result = engine.predict_image(spec, media.decode_image(data))
            result["media_type"] = "image"
            return result

        # video upload -> write to a temp file for the decoder
        suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            if spec.modality == "video":
                result = engine.predict_video_clip(spec, tmp.name)
            else:
                frames = media.sample_video_pil(tmp.name, config.VIDEO_IMAGE_FRAMES)
                result = engine.predict_video_frames(spec, frames)
        result["media_type"] = "video"
        return result
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:  # surface model/decoding errors as 500 with a message
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


# --- SOTA generator vs detector comparison demo ------------------------------
# Pick a real image from our datasets -> send it to a current SOTA generator (Nano Banana / GPT Image /
# Grok / FLUX) with "make another image conceptually similar to this" -> score the generated image with
# the SynthScope detectors (best/promoted checkpoints) so the user sees how today's detectors hold up.
import base64 as _b64, glob as _glob, random as _random
from . import generators as _gens

from .datasets import _root_candidates  # node-portable scratch-mount remap (/hddstore<->/ssdstore<->...)

# Base patterns point at cn023 /hddstore; expand each onto the other scratch mounts so the
# real-image pool resolves wherever the data was staged (cees /ssdstore, etc.).
_REAL_POOL_PATTERNS = [
    "/hddstore/ssalehi/SynthScope/GenImage/test/real",
    "/hddstore/ssalehi/SynthScope/CNNDetection/test/*/0_real",
    "/hddstore/ssalehi/SynthScope/GenImage_full/test_full/*/0_real",
]
_REAL_POOLS = [c for p in _REAL_POOL_PATTERNS for c in _root_candidates(p)]


@app.get("/api/sota/generators")
def sota_generators():
    return {"generators": _gens.list_generators(), "default_prompt": _gens.DEFAULT_PROMPT}


@app.get("/api/sota/real_image")
def sota_real_image():
    """Return a random REAL image from our datasets (base64 PNG) for the comparison demo."""
    candidates = []
    for pat in _REAL_POOLS:
        for d in _glob.glob(pat):
            fs = _glob.glob(d + "/*")
            if fs:
                candidates.append(_random.choice(fs)); break
        if len(candidates) >= 3:
            break
    if not candidates:
        raise HTTPException(status_code=404, detail="No real images found in datasets.")
    path = _random.choice(candidates)
    img = media.decode_image(open(path, "rb").read())
    buf = __import__("io").BytesIO(); img.convert("RGB").save(buf, "PNG")
    return {"image_b64": _b64.b64encode(buf.getvalue()).decode(), "source": path.split("/")[-3:]}


_GPU_LOCK = threading.Lock()   # detector scoring and heavy local generation never overlap on the GPU


async def _sota_generate(real_bytes: bytes, generator: str, prompt: str, mode: str = "recreate"):
    """Run one generator on the real image -> (effective prompt, generated PNG bytes); HTTP errors on failure."""
    if not real_bytes:
        raise HTTPException(status_code=400, detail="Empty image.")
    # text-to-image generators left on the old default prompt get a caption-derived one (generators.effective_prompt)
    prompt = _gens.effective_prompt(generator, real_bytes, prompt)
    try:
        # off the event loop: a local generation takes 1-4 minutes and must not freeze the whole UI
        heavy = generator in _gens.GENERATORS and _gens.GENERATORS[generator].heavy

        def run() -> bytes:
            if not heavy:
                return _gens.generate(generator, real_bytes, prompt, mode)
            with _GPU_LOCK:                          # waits for any in-flight scoring to finish first
                get_engine().release_gpu()
                return _gens.generate(generator, real_bytes, prompt, mode)
        return prompt, await asyncio.to_thread(run)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:                       # missing API key
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"generator failed: {type(e).__name__}: {e}")


def _score_one(engine, key: str, real_img, gen_img) -> dict:
    try:
        spec = get_spec(key)
        if spec.modality != "image" or not engine.is_available(spec):
            return {"detector": key, "error": "unavailable / not an image detector"}
        rr, gr = engine.predict_image(spec, real_img), engine.predict_image(spec, gen_img)
        return {"detector": key, "name": spec.name,
                "real_fake_prob": round(rr["fake_probability"], 4),
                "gen_fake_prob": round(gr["fake_probability"], 4),
                "gen_verdict": gr.get("verdict"), "caught": gr["fake_probability"] >= 0.5}
    except Exception as e:
        return {"detector": key, "error": f"{type(e).__name__}: {e}"}


def _sota_score(real_bytes: bytes, gen_bytes: bytes, detectors: str) -> list[dict]:
    """Score the real and the generated image with each listed detector (no generation involved)."""
    real_img, gen_img = media.decode_image(real_bytes), media.decode_image(gen_bytes)
    engine = get_engine()
    results = []
    for key in [d.strip() for d in detectors.split(",") if d.strip()]:
        with _GPU_LOCK:                              # one detector at a time: concurrent requests queue, not race
            results.append(_score_one(engine, key, real_img, gen_img))
    return results


def _layout_similarity(a_bytes: bytes, b_bytes: bytes) -> float:
    """Correlation of the two images' 64x64 grayscale layouts, -1..1. About 0.9+ means the generated image is an edit
    of the source (same composition); low values mean a genuinely new image."""
    import numpy as np
    def thumb(b: bytes):
        im = media.decode_image(b).convert("L").resize((64, 64))
        return np.asarray(im, dtype="float64") - np.asarray(im, dtype="float64").mean()
    a, b = thumb(a_bytes), thumb(b_bytes)
    return float((a * b).sum() / (((a * a).sum() * (b * b).sum()) ** 0.5 + 1e-8))


@app.post("/api/sota/describe")
async def sota_describe(file: UploadFile = File(...)):
    """Caption the source photo and return a text-to-image prompt for it (the user can edit it before generating)."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image.")
    return await asyncio.to_thread(_gens.describe_prompt, data)


@app.post("/api/sota/generate")
async def sota_generate(file: UploadFile = File(...), generator: str = Form(...),
                        prompt: str = Form(_gens.DEFAULT_PROMPT), mode: str = Form("recreate")):
    """real image + generator -> the generated image only. Scoring is a separate call (/api/sota/score), so
    trying another detector never regenerates. mode "recreate" builds a new image from the prompt alone,
    "edit" edits the source photo."""
    real_bytes = await file.read()
    prompt, gen_bytes = await _sota_generate(real_bytes, generator, prompt, mode)
    sim = _layout_similarity(real_bytes, gen_bytes)
    return {"generator": generator, "mode": mode, "prompt": prompt, "similarity": round(sim, 3),
            "near_copy": sim >= 0.8, "generated_b64": _b64.b64encode(gen_bytes).decode()}


@app.post("/api/sota/score")
async def sota_score(real: UploadFile = File(...), generated: UploadFile = File(...), detectors: str = Form(...)):
    """Score an existing real/generated pair with the listed detectors (fast: no generation)."""
    real_bytes, gen_bytes = await real.read(), await generated.read()
    if not real_bytes or not gen_bytes:
        raise HTTPException(status_code=400, detail="Empty image.")
    return {"results": await asyncio.to_thread(_sota_score, real_bytes, gen_bytes, detectors)}


@app.post("/api/sota/compare")
async def sota_compare(file: UploadFile = File(...), generator: str = Form(...),
                       detectors: str = Form(...), prompt: str = Form(_gens.DEFAULT_PROMPT)):
    """Generate + score in one call (kept for API clients; the web UI uses generate then score)."""
    real_bytes = await file.read()
    prompt, gen_bytes = await _sota_generate(real_bytes, generator, prompt, "edit")
    return {"generator": generator, "prompt": prompt, "generated_b64": _b64.b64encode(gen_bytes).decode(),
            "results": await asyncio.to_thread(_sota_score, real_bytes, gen_bytes, detectors)}


# --- Leaderboard: all methods x benchmarks (paper vs SynthScope-trained vs official), by benchmark ------
_METHOD_NAMES = {
    "clip": "CLIP / UnivFD", "cnn": "CNNDetection", "fat_former": "FatFormer", "npr": "NPR",
    "aide": "AIDE", "safe": "SAFE", "co_spy": "Co-Spy", "spai": "SPAI", "vib_net": "VIB-Net",
    "pgc": "PGC", "d3": "D3", "restrav": "ReStraV", "stall": "STALL",
}
_FAMILY = [  # (substring -> benchmark family label); first match wins
    ("genimage", "GenImage"), ("forensynth", "ForenSynths"), ("universalfakedetect", "UniversalFakeDetect"),
    ("ojha", "UniversalFakeDetect"), ("ufd", "UniversalFakeDetect"), ("aigcdetect", "AIGCDetect"),
    ("co-spy-bench", "Co-Spy-Bench"), ("aigibench", "AIGIBench"), ("aigi", "AIGIBench"),
    ("commgen", "CommGen15"), ("genvideo", "GenVideo"), ("evalcrafter", "EvalCrafter"),
    ("videophy", "VideoPhy"), ("vidprom", "VidProM"), ("genvidbench", "GenVidBench"),
    ("ditfake", "DiTFake"), ("diffusionforensics", "DiffusionForensics"),
    ("diffusion1kstep", "Diffusion1kStep"),  # before self-synthesis (NPR Table 5 contains both)
    ("gangen", "GANGen / Self-Synthesis"), ("self-synthesis", "GANGen / Self-Synthesis"),
    ("chameleon", "Chameleon"), ("synthbuster", "Synthbuster"), ("in-the-wild", "In-the-wild"),
    ("videofeedback", "VideoFeedback"), ("comgenvid", "ComGenVid"),
    ("physics-iq", "Physics-IQ"), ("veo3", "Veo3 zero-shot"), ("demamba", "DeMamba"),
    ("gan-source", "GAN-source"), ("diffusion-source", "Diffusion-source"),
]


def _family(name: str) -> str:
    n = (name or "").lower()
    for sub, fam in _FAMILY:
        if sub in n:
            return fam
    return (name.split("(")[0].strip() or name)[:28]


def _lead_num(s: str) -> float:
    import re as _re
    m = _re.search(r"(\d+\.?\d*)", s or "")
    return float(m.group(1)) if m else -1.0


@app.get("/api/leaderboard")
def leaderboard():
    """Every method x benchmark with paper / SynthScope-trained / official-weights numbers, grouped by
    benchmark family so methods that share a benchmark can be compared head-to-head."""
    groups: dict[str, list] = {}
    for key, rows in REPRO_FULL.items():
        for r in rows:
            fam = _family(r["benchmark"])
            groups.setdefault(fam, []).append({
                "method": _METHOD_NAMES.get(key, key), "key": key,
                "benchmark": r["benchmark"], "paper": r["paper"],
                "synthscope": r.get("ours_synthscope", ""), "official": r.get("ours_official", ""),
                "status": r["status"],
                "_sort": max(_lead_num(r.get("ours_synthscope")), _lead_num(r.get("ours_official"))),
            })
    out = []
    for fam, methods in groups.items():
        methods.sort(key=lambda m: m["_sort"], reverse=True)
        for m in methods:
            m.pop("_sort", None)
        out.append({"benchmark": fam, "n_methods": len(methods), "rows": methods})
    out.sort(key=lambda g: (-g["n_methods"], g["benchmark"]))  # most cross-method-compared first
    return {"groups": out}


@app.get("/api/generator_benchmarks")
def generator_benchmarks():
    """Detector-vs-SOTA-generator benchmarks. Scans repro/ideogram_bench* (e.g. 512px and 1024px runs);
    extensible to FLUX / Nano Banana / GPT-Image by dropping in more results.json dirs."""
    import json as _json, glob as _glob2
    out = []
    for d in sorted(_glob2.glob(str(config.REPO_ROOT / "repro" / "ideogram_bench*"))):
        res_f, man_f = Path(d) / "results.json", Path(d) / "manifest.json"
        if not res_f.is_file():
            continue
        rows = _json.loads(res_f.read_text())
        meta = _json.loads(man_f.read_text()) if man_f.is_file() else {}
        out.append({
            "generator": meta.get("label") or "Ideogram 4.0",
            "key": Path(d).name,
            "n_real": meta.get("n"), "n_fake": meta.get("n"), "res": meta.get("res", 0),
            "settings": f"{meta.get('res','?')}px · {meta.get('steps','?')} steps"
                        + (f" · {meta['attempts']} attempts" if meta.get("attempts") else ""),
            "note": "Balanced real (GenImage/CNNDetection) vs Ideogram fakes from BLIP-caption prompts; "
                    "people excluded (Ideogram refuses them). Detectors = promoted SynthScope checkpoints. "
                    "None were trained on Ideogram, so this is cross-generator generalization. Ranked by AUROC.",
            "rows": rows,
        })
    out.sort(key=lambda b: b.get("res", 0), reverse=True)   # highest-res first
    return {"benchmarks": out}


# --- dataset explorer --------------------------------------------------------
@app.get("/api/datasets")
def datasets_list():
    """Datasets present on disk + which dataset each detector was trained on."""
    return {
        "datasets": datasets.list_datasets(),
        "model_train": datasets.MODEL_TRAIN_DATASET,
    }


@app.get("/api/datasets/{key}/sample")
def dataset_sample(key: str, split: str = "test", n: int = 24, seed: int = 0):
    try:
        items = datasets.sample_batch(key, split, n=max(1, min(n, 60)), seed=seed)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"dataset": key, "split": split, "seed": seed, "items": items}


@app.get("/api/datasets/{key}/media")
def dataset_media(key: str, rel: str = Query(...), thumb: int = 0):
    try:
        path = datasets.resolve_media(key, rel)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if thumb:
        modality = datasets.get_def(key).modality
        try:
            return Response(content=media.thumbnail_bytes(path, modality), media_type="image/jpeg")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"thumbnail failed: {e}")
    return FileResponse(str(path))


def _run_on_dataset_item(engine, spec, dataset: str, rel: str) -> dict:
    """Run ``spec`` on an on-disk dataset file ``rel`` and attach ground-truth/correctness.

    Raises HTTPException for caller-facing errors. Shared by /api/predict_dataset (single item,
    full result) and /api/score_batch (many items, scores only).
    """
    path = datasets.resolve_media(dataset, rel)          # may raise Permission/FileNotFound
    d = datasets.get_def(dataset)
    if d.modality == "video":
        if spec.modality == "video":
            result = engine.predict_video_clip(spec, str(path))
        else:
            frames = media.sample_video_pil(str(path), config.VIDEO_IMAGE_FRAMES)
            result = engine.predict_video_frames(spec, frames)
        result["media_type"] = "video"
    else:
        if spec.modality == "video":
            raise HTTPException(status_code=400, detail=f"'{spec.name}' is a video detector; pick a video dataset.")
        with open(path, "rb") as fh:
            result = engine.predict_image(spec, media.decode_image(fh.read()))
        result["media_type"] = "image"
    gt = datasets.label_of(Path(rel))
    result["ground_truth"] = gt
    result["correct"] = (gt is not None and result["verdict"] == gt)
    return result


@app.post("/api/predict_dataset")
def predict_dataset(dataset: str = Form(...), rel: str = Form(...), model: str = Form(...)):
    """Run a detector on a dataset file already on disk (no re-upload) + return ground truth."""
    try:
        spec = get_spec(model)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    engine = get_engine()
    if not engine.is_available(spec):
        raise HTTPException(status_code=409, detail=f"No checkpoint available for '{spec.key}'.")
    try:
        return _run_on_dataset_item(engine, spec, dataset, rel)
    except HTTPException:
        raise
    except (KeyError, PermissionError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.post("/api/score_batch")
def score_batch(payload: dict = Body(...)):
    """Score many dataset items at once (for threshold calibration). Body: {dataset, model, rels:[...]}.

    Returns ``{items:[{rel, fake_probability, ground_truth}]}`` -- per-item errors are reported inline
    (``error`` field) instead of failing the whole batch, so calibration uses whatever decoded."""
    dataset = payload.get("dataset"); model = payload.get("model"); rels = payload.get("rels") or []
    try:
        spec = get_spec(model)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    engine = get_engine()
    if not engine.is_available(spec):
        raise HTTPException(status_code=409, detail=f"No checkpoint available for '{spec.key}'.")
    out = []
    for rel in rels[:200]:
        try:
            r = _run_on_dataset_item(engine, spec, dataset, rel)
            out.append({"rel": rel, "fake_probability": r["fake_probability"], "ground_truth": r["ground_truth"]})
        except Exception as e:
            out.append({"rel": rel, "error": f"{type(e).__name__}: {getattr(e, 'detail', e)}"})
    return {"dataset": dataset, "model": model, "items": out}


# --- static / SPA serving ----------------------------------------------------
if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

# Serve the standalone page's local assets (vendored React/htm) so the UI needs no external CDN.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


_NO_CACHE = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}


@app.get("/")
def index():
    # Serve the UI with no-store so a plain reload always picks up edits (no stale cached page).
    src = FRONTEND_DIST / "index.html" if (FRONTEND_DIST / "index.html").is_file() else STATIC_DIR / "index.html"
    return FileResponse(str(src), headers=_NO_CACHE)
