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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config, media
from .engine import get_engine
from .registry import get_spec, list_specs

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


# --- static / SPA serving ----------------------------------------------------
if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")


@app.get("/")
def index():
    if (FRONTEND_DIST / "index.html").is_file():
        return FileResponse(str(FRONTEND_DIST / "index.html"))
    return FileResponse(str(STATIC_DIR / "index.html"))
