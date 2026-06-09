"""FastAPI app for the AGID detector web interface.

Endpoints:
  GET  /api/models             -> list detectors + availability
  GET  /api/health             -> liveness + device
  POST /api/predict            -> {file, model} -> fake/real probability

Serves the built Vite frontend from ``frontend/dist`` if present, otherwise the no-build
standalone page in ``static/index.html`` (works without Node).
"""
import tempfile
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import config, datasets, media
from .engine import get_engine
from .registry import REPRO, REPRO_FULL, get_spec, list_specs

app = FastAPI(title="AGID Detector Interface", version="1.0")
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
            "notes": spec.notes,
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
# the AGID detectors (best/promoted checkpoints) so the user sees how today's detectors hold up.
import base64 as _b64, glob as _glob, random as _random
from . import generators as _gens

_REAL_POOLS = [
    "/hddstore/ssalehi/AGID/GenImage/test/real",
    "/hddstore/ssalehi/AGID/CNNDetection/test/*/0_real",
    "/hddstore/ssalehi/AGID/GenImage_full/test_full/*/0_real",
]


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


@app.post("/api/sota/compare")
async def sota_compare(file: UploadFile = File(...), generator: str = Form(...),
                       detectors: str = Form(...), prompt: str = Form(_gens.DEFAULT_PROMPT)):
    """real image + SOTA generator + detector list -> generated image + each detector's score on real & generated."""
    real_bytes = await file.read()
    if not real_bytes:
        raise HTTPException(status_code=400, detail="Empty image.")
    try:
        gen_bytes = _gens.generate(generator, real_bytes, prompt)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:                       # missing API key
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"generator failed: {type(e).__name__}: {e}")

    real_img, gen_img = media.decode_image(real_bytes), media.decode_image(gen_bytes)
    engine = get_engine()
    results = []
    for key in [d.strip() for d in detectors.split(",") if d.strip()]:
        try:
            spec = get_spec(key)
            if spec.modality != "image" or not engine.is_available(spec):
                results.append({"detector": key, "error": "unavailable / not an image detector"}); continue
            rr, gr = engine.predict_image(spec, real_img), engine.predict_image(spec, gen_img)
            results.append({"detector": key, "name": spec.name,
                            "real_fake_prob": round(rr["fake_probability"], 4),
                            "gen_fake_prob": round(gr["fake_probability"], 4),
                            "gen_verdict": gr.get("verdict"), "caught": gr["fake_probability"] >= 0.5})
        except Exception as e:
            results.append({"detector": key, "error": f"{type(e).__name__}: {e}"})
    return {"generator": generator, "prompt": prompt,
            "generated_b64": _b64.b64encode(gen_bytes).decode(), "results": results}


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


@app.get("/")
def index():
    if (FRONTEND_DIST / "index.html").is_file():
        return FileResponse(str(FRONTEND_DIST / "index.html"))
    return FileResponse(str(STATIC_DIR / "index.html"))
