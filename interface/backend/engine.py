"""Inference engine: build AGID models from config, load checkpoints, and predict.

Models are built lazily on first use (backbones are large and slow to download/instantiate)
and kept in a small LRU so a shared GPU is not exhausted. All forward passes happen under
``torch.no_grad()`` with the model in fp32/eval mode.
"""
import threading
import time
from collections import OrderedDict

import torch
from PIL import Image

from . import config
from .registry import ModelSpec, get_spec
from . import media


# ---------------------------------------------------------------------------
# Model construction (mirrors trainer.AGIDModelTrainer._build_model dispatch, with lazy
# imports so unused heavy dependencies are never loaded).
# ---------------------------------------------------------------------------
def _build_model(cfg):
    model_type = cfg.ModelConfig.get("model_type", "dino").lower()
    if model_type == "safe":
        from models.image.safe.safe import AGIDSAFEModel
        return AGIDSAFEModel(cfg)
    if model_type == "fat_former":
        from models.image.fat_former.fat_former import AGIDFatFormerModel
        return AGIDFatFormerModel(cfg)
    if model_type == "co_spy":
        from models.image.co_spy.co_spy import AGIDCoSpyModel
        return AGIDCoSpyModel(cfg)
    if model_type == "aide":
        from models.image.aide.aide import AGIDAIDEModel
        return AGIDAIDEModel(cfg)
    if model_type == "spai":
        from models.image.spai.spai import SPAI
        return SPAI(cfg)
    if model_type == "vib_net":
        from models.image.vib_net.vib_net import AGIDVIBNetModel
        return AGIDVIBNetModel(cfg)
    if model_type == "npr":
        from models.image.npr.npr import NPR
        return NPR(cfg)
    if model_type == "pgc":
        from models.image.pgc.pgc import AGIDPGCModel
        return AGIDPGCModel(cfg)
    if model_type == "stall":
        from models.video.stall.stall import AGIDSTALLModel
        return AGIDSTALLModel(cfg)
    if model_type == "d3":
        from models.video.d3.d3 import AGIDD3Model
        return AGIDD3Model(cfg)
    if model_type == "restrav":
        from models.video.restrav.restrav import AGIDReStraVModel
        return AGIDReStraVModel(cfg)
    if model_type == "cnn":
        from models.image.cnn_detection.cnn_detection import AGIDCNNModel
        return AGIDCNNModel(cfg)
    if model_type == "adapted_dino":
        from models.image.adapted_dino.adapted_dino import AGIDAdaptedDINOModel
        return AGIDAdaptedDINOModel(cfg)
    if model_type == "dinov2":
        from models.image.dino_detection.dino_detection import AGIDDINOModel
        return AGIDDINOModel(cfg)
    if model_type == "dinov3":
        from models.image.dino_detection.dino_detection import AGIDDINOv3Model
        return AGIDDINOv3Model(cfg)
    if model_type == "clip" or cfg.ModelConfig.backbone_name.startswith("CLIP:"):
        from models.image.clip_detection.clip_detection import AGIDCLIPModel
        return AGIDCLIPModel(cfg)
    raise ValueError(f"Invalid model_type: {model_type}")


def _load_checkpoint(model, spec: ModelSpec):
    """Load weights for ``spec`` into ``model`` (already constructed). Returns a status string."""
    path = config.CHECKPOINTS_DIR / spec.checkpoint
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Train or place the checkpoint, then retry."
        )
    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    if spec.remap is not None:
        sd = spec.remap(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Buffers that are legitimately reconstructed in __init__ / not in official dumps.
    benign = {"clip_mean", "clip_std"}
    real_missing = [
        k for k in missing
        if not k.startswith("criterion") and k.split(".")[-1] not in benign
    ]
    if spec.ckpt_kind.startswith("official"):
        # Official-weights loads are verified to be exact; flag any genuine mismatch loudly.
        if real_missing or unexpected:
            raise RuntimeError(
                f"{spec.key}: checkpoint mismatch missing={real_missing[:8]} "
                f"unexpected={unexpected[:8]}"
            )
    return (
        f"loaded {len(sd)} tensors from {path.name} "
        f"(missing={len(real_missing)}, unexpected={len(unexpected)})"
    )


class Engine:
    """Lazily builds, caches, and runs the detectors."""

    def __init__(self, device: str | None = None):
        # ``device`` lets a caller (e.g. the ``agid`` Python API) pin a specific device;
        # the web backend constructs it with no argument and auto-detects via config.device().
        self.device = device or config.device()
        self._cache: "OrderedDict[str, torch.nn.Module]" = OrderedDict()
        self._status: dict[str, str] = {}
        self._lock = threading.Lock()

    # -- availability -------------------------------------------------------
    def checkpoint_path(self, spec: ModelSpec):
        if spec.checkpoint is None:
            return None
        return config.CHECKPOINTS_DIR / spec.checkpoint

    def is_available(self, spec: ModelSpec) -> bool:
        if spec.key == "stall":  # training-free, but needs the gated DINOv3 ViT-L/16 weights
            import os
            from .registry import DINOV3_WEIGHTS
            return os.path.exists(os.environ.get("DINO_V3_WEIGHTS") or DINOV3_WEIGHTS)
        if spec.ckpt_kind == "none":  # training-free (D3)
            return True
        p = self.checkpoint_path(spec)
        return bool(p and p.exists())

    # -- model lifecycle ----------------------------------------------------
    def _get_model(self, spec: ModelSpec) -> torch.nn.Module:
        with self._lock:
            if spec.key in self._cache:
                self._cache.move_to_end(spec.key)
                return self._cache[spec.key]

            # Evict LRU entries if at capacity.
            while len(self._cache) >= config.MAX_RESIDENT_MODELS:
                old_key, old_model = self._cache.popitem(last=False)
                del old_model
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            from config_utils import compose_config

            cfg = compose_config(spec.overrides)
            model = _build_model(cfg)
            if spec.ckpt_kind != "none":
                self._status[spec.key] = _load_checkpoint(model, spec)
            else:
                self._status[spec.key] = "training-free (no checkpoint)"
            model = model.to(self.device).float().eval()
            self._cache[spec.key] = model
            return model

    # -- prediction helpers -------------------------------------------------
    @torch.no_grad()
    def _fake_prob(self, model, spec: ModelSpec, x: torch.Tensor) -> torch.Tensor:
        """Return P(fake) per sample for an image batch ``x`` [B,3,H,W]."""
        kind = spec.predict_kind
        if kind == "ce":
            return model(x).softmax(dim=1)[:, 1]
        if kind == "bce":
            return torch.sigmoid(model(x).reshape(x.shape[0]))
        if kind == "bce_npr":
            logits = model.forward(model.NPR_transform(x)).reshape(x.shape[0])
            return torch.sigmoid(logits)
        if kind == "vib":
            out = model(x)
            logit = out[1] if isinstance(out, (tuple, list)) else out
            return torch.sigmoid(logit.reshape(x.shape[0]))
        raise ValueError(f"Unexpected image predict_kind: {kind}")

    def _d3_calibration(self):
        """Load (and cache) the D3 score->probability calibration if present."""
        if not hasattr(self, "_d3_cal"):
            import json
            p = config.CHECKPOINTS_DIR / "d3_calibration.json"
            self._d3_cal = json.loads(p.read_text()) if p.exists() else None
        return self._d3_cal

    @torch.no_grad()
    def _video_fake_prob(self, model, spec: ModelSpec, clip: torch.Tensor) -> torch.Tensor:
        """Return P(fake) for a single normalized clip tensor [T,3,H,W]."""
        frames = clip.unsqueeze(0).to(self.device)  # [1,T,3,H,W]
        if spec.key == "d3":
            cal = self._d3_calibration()
            if cal is not None:
                s = model.score(frames)  # [1] raw volatility score
                z = (s - cal["threshold"]) / cal["scale"]
                if not cal["higher_is_fake"]:
                    z = -z
                return torch.sigmoid(z)
        out = model(frames)  # [1,2]
        if spec.key == "stall":
            # STALL returns [real_prob, fake_prob] (already calibrated percentiles); use directly.
            return out[:, 1]
        return out.softmax(dim=1)[:, 1]

    # -- public API ---------------------------------------------------------
    def predict_image(self, spec: ModelSpec, pil: Image.Image) -> dict:
        model = self._get_model(spec)
        x = spec.transform(pil).unsqueeze(0).to(self.device)
        t0 = time.time()
        p = float(self._fake_prob(model, spec, x).item())
        return self._result(spec, p, time.time() - t0)

    def predict_video_frames(self, spec: ModelSpec, frames: list[Image.Image]) -> dict:
        """Run an *image* detector on sampled video frames and aggregate (mean P(fake))."""
        model = self._get_model(spec)
        batch = torch.stack([spec.transform(f) for f in frames]).to(self.device)
        t0 = time.time()
        per_frame = self._fake_prob(model, spec, batch).detach().float().cpu().tolist()
        elapsed = time.time() - t0
        mean_p = sum(per_frame) / len(per_frame)
        res = self._result(spec, mean_p, elapsed)
        res["per_frame"] = [round(v, 6) for v in per_frame]
        res["aggregation"] = "mean over sampled frames"
        return res

    def predict_video_clip(self, spec: ModelSpec, path) -> dict:
        """Run a native *video* detector on a decoded clip."""
        model = self._get_model(spec)
        clip = media.video_clip_tensor(
            path, num_frames=spec.num_frames, fps=spec.fps, frame_size=spec.frame_size
        )
        mean, std = spec.video_norm
        norm = torch.tensor(mean).view(1, 3, 1, 1)
        sd = torch.tensor(std).view(1, 3, 1, 1)
        clip = (clip - norm) / sd
        t0 = time.time()
        p = float(self._video_fake_prob(model, spec, clip).item())
        res = self._result(spec, p, time.time() - t0)
        res["num_frames"] = spec.num_frames
        return res

    def _result(self, spec: ModelSpec, fake_prob: float, elapsed: float) -> dict:
        fake_prob = max(0.0, min(1.0, fake_prob))
        return {
            "model": spec.key,
            "model_name": spec.name,
            "fake_probability": round(fake_prob, 6),
            "real_probability": round(1.0 - fake_prob, 6),
            "verdict": "fake" if fake_prob >= 0.5 else "real",
            "confidence": round(max(fake_prob, 1.0 - fake_prob), 6),
            "elapsed_ms": round(elapsed * 1000, 1),
            "checkpoint_status": self._status.get(spec.key, ""),
        }


_ENGINE: Engine | None = None


def get_engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = Engine()
    return _ENGINE
