"""STALL integrated into AGID.

Paper: "Training-free Detection of Generated Videos via Spatial-Temporal Likelihoods"
(Ben Hayun et al., CVPR 2026). https://github.com/OmerBenHayun/STALL

STALL is a *training-free* detector. A frozen DINOv3 ViT-L/16 turns each frame into a feature
``F_i``. It scores how anomalous a video's spatial and temporal statistics are relative to a
calibration set of REAL videos (VATEX), under two isotropic-Gaussian likelihood branches:

    F_i            = encoder(frame_i)                                  # [T, D=1024]
    spatial LL     = N(0,I).logpdf( (F_i - mu_s) @ W_s )               # [T]
    T_i            = normalize(F_{i+1} - F_i)                          # [T-1, D]
    temporal LL    = N(0,I).logpdf( (T_i - mu_t) @ W_t )              # [T-1]

where (mu_s, W_s) / (mu_t, W_t) are the mean + PCA-whitening matrix fit on the real
calibration set. Per-video aggregation is max over the spatial LLs and min over the temporal
LLs; each aggregate is converted to a percentile against the calibration LLs and the two are
averaged:

    final = 0.5 * ( pct(max_t spatial_LL) + pct(min_t temporal_LL) )   # higher => more REAL

We map ``final`` to two-class outputs so the AGID metrics work:

    outputs = stack([final, 1 - final], dim=1)        # col 0 = real, col 1 = fake

so ``softmax(outputs)[:, 1]`` ranks AI-generated videos by *low* ``final``. STALL does not
train (threshold-free AP/AUROC are the meaningful metrics); the dummy ``_noop`` parameter
keeps AdamW happy, exactly like the D3 detector.

The whitening transforms and calibration log-likelihoods are the official VATEX/DINOv3 params
in ``stall_params_vatex_dino_v3.npz`` (override with ``ModelConfig.stall_params_path`` to use a
calibration set recomputed via the official ``create_params.py``).

Normalization note
------------------
This model does NOT normalize internally. It expects frames already normalized by the dataset
transform pipeline, with **ImageNet** statistics -- use ``normalize_dinov2`` in the experiment
script / dataset config (the official STALL DINOv3 transform uses ImageNet mean/std, NOT the
DINOv3-paper stats, so ``normalize_dinov2`` is correct here, *not* ``normalize_dinov3``). Frames
are decoded square at ``frame_size=224`` by the video dataset. The frozen backbone -- and the
calibration params, which were fit under this exact preprocessing -- silently degrade on
mismatched normalization.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_PARAMS = _THIS_DIR / "stall_params_vatex_dino_v3.npz"
_AGG = {"mean": np.mean, "max": np.max, "min": np.min}


# ─────────────────────────────────────────────────────────────────────────────
# Pure scoring math (numpy; mirrors the official implementation exactly)
# ─────────────────────────────────────────────────────────────────────────────
def diff_vec(arr: np.ndarray) -> np.ndarray:
    """Consecutive frame differences. [N, T, D] -> [N, T-1, D]."""
    return arr[:, 1:, :] - arr[:, :-1, :]


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.where(norm == 0, 1.0, norm)


def normalized_temporal_diff(arr: np.ndarray) -> np.ndarray:
    """T_i = L2normalize(F_{i+1} - F_i). [N, T, D] -> [N, T-1, D]."""
    return l2_normalize(diff_vec(arr))


def whiten(x: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    """PCA-whitening with pre-fitted params: (x - mu) @ W."""
    return np.matmul(x - mu, W)


def isotropic_gaussian_loglik(x: np.ndarray) -> np.ndarray:
    """Log-likelihood under N(0, I). x[..., D] -> [...]."""
    D = x.shape[-1]
    return -0.5 * (D * np.log(2.0 * np.pi) + (x ** 2).sum(axis=-1))


def percentile_vs_calibration(scores: np.ndarray, sorted_calib: np.ndarray) -> np.ndarray:
    """Fraction of calibration scores <= each score (both 1-D)."""
    return np.searchsorted(sorted_calib, scores, side="right") / len(sorted_calib)


def fit_whitening(X: np.ndarray, n_components: int | None = None, eps: float = 1e-5):
    """Fit a PCA-whitening transform on rows of ``X`` [N, D]. Returns (mu, W) as float32.

    Whitened ``(X - mu) @ W`` has (approximately) identity covariance. Mirrors the official
    STALL ``WhiteningTransform``: eigendecomposition of the covariance, descending order,
    rank-truncated to ``min(N-1, D)`` (or ``n_components``), non-positive eigenvalues dropped,
    ``W = eigvecs @ diag(1/sqrt(eigvals + eps))``. Computed in float64 for stability.
    """
    X = np.asarray(X, dtype=np.float64)
    N, D = X.shape
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)                       # mean-subtracted internally
    evals, evecs = np.linalg.eigh(cov)                  # ascending
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    max_rank = min(N - 1, D)
    evals, evecs = evals[:max_rank], evecs[:, :max_rank]
    if n_components is not None:
        r = min(n_components, max_rank)
        evals, evecs = evals[:r], evecs[:, :r]
    valid = evals > 0
    if not np.any(valid):
        raise ValueError("All eigenvalues are non-positive; cannot whiten.")
    evals, evecs = evals[valid], evecs[:, valid]
    W = evecs @ np.diag(1.0 / np.sqrt(evals + eps))
    return mu.astype(np.float32), W.astype(np.float32)


def fit_stall_params(embs: np.ndarray, spat_agg: str = "max", temp_agg: str = "min") -> dict:
    """Compute STALL calibration params from REAL-video embeddings ``embs`` [N, T, D].

    Mirrors the official ``create_params.build_params``:
      * spatial: fit whitening on 1 random frame per video ([N, D]); calib LLs over all frames.
      * temporal: fit whitening on all L2-normalized consecutive diffs ([(T-1)*N, D]); calib LLs
        over all frame-pairs.
    ``spat_agg`` / ``temp_agg`` are stored only as metadata (aggregation happens at scoring time).
    """
    N, T, D = embs.shape
    rng = np.random.RandomState(42)
    single = embs[np.arange(N), rng.randint(0, T, size=N)]            # [N, D]
    mu_spat, W_spat = fit_whitening(single)
    calib_ll_spat = isotropic_gaussian_loglik(whiten(embs, mu_spat, W_spat))   # [N, T]

    diffs = normalized_temporal_diff(embs)                            # [N, T-1, D]
    flat = diffs.reshape((T - 1) * N, D)
    mu_temp, W_temp = fit_whitening(flat)
    calib_ll_temp = isotropic_gaussian_loglik(whiten(diffs, mu_temp, W_temp))  # [N, T-1]

    return dict(
        mu_spat=mu_spat, W_spat=W_spat, calib_ll_spat=calib_ll_spat,
        mu_temp=mu_temp, W_temp=W_temp, calib_ll_temp=calib_ll_temp,
        spat_agg=np.array(spat_agg), temp_agg=np.array(temp_agg),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Backbone
# ─────────────────────────────────────────────────────────────────────────────
class _Dinov3Encoder(nn.Module):
    """DINOv3 ViT-L/16 -> per-frame pooled feature ``[N, 1024]``.

    ``x`` is ``[N, 3, H, W]`` already normalized (ImageNet stats) by the dataset transform
    pipeline. Loaded from a local clone of the DINOv3 repo plus its gated weights.
    """

    def __init__(self, repo_dir: str, weights: str):
        super().__init__()
        if not repo_dir or not os.path.isdir(repo_dir):
            raise FileNotFoundError(
                f"DINOv3 repo not found at '{repo_dir}'. Clone github.com/facebookresearch/dinov3 "
                f"and set ModelConfig.dino_repo or the DINO_V3_REPO_DIR env var."
            )
        if not weights or not os.path.exists(weights):
            raise FileNotFoundError(
                "DINOv3 ViT-L/16 weights not found. Download "
                "'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth' (gated) and set "
                "ModelConfig.dino_weights or the DINO_V3_WEIGHTS env var."
            )
        sys.path.insert(0, repo_dir)
        try:
            from dinov3.hub.backbones import dinov3_vitl16
            self.model = dinov3_vitl16(weights=weights)
        finally:
            if repo_dir in sys.path:
                sys.path.remove(repo_dir)

    def encode_frames(self, x):
        return self.model(x)  # [N, D] pooled


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────
class AGIDSTALLModel(nn.Module):
    """Training-free spatial-temporal likelihood video detector (STALL)."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        cfg = config.ModelConfig

        self.backbone_name = str(cfg.get("backbone_name", "dinov3_vitl16")).lower()
        self.spat_agg = str(cfg.get("spat_agg", "max")).lower()
        self.temp_agg = str(cfg.get("temp_agg", "min")).lower()
        self.frozen = bool(cfg.get("freeze_backbone", True))

        repo_dir = cfg.get("dino_repo", None) or os.getenv(
            "DINO_V3_REPO_DIR", "/home/ssalehi/.cache/torch/hub/facebookresearch_dinov3"
        )
        weights = cfg.get("dino_weights", None) or os.getenv("DINO_V3_WEIGHTS", "")
        self.backbone = self._build_backbone(self.backbone_name, repo_dir, weights)
        if self.frozen:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # AdamW needs >=1 trainable parameter; this one is a true no-op.
        self._noop = nn.Parameter(torch.zeros(1))

        # Calibration params (real-data statistics). Recompute from a real-video calibration set
        # when asked (or when the file is missing); otherwise load the .npz (default = the
        # VATEX/DINOv3 params shipped next to this model).
        self.params_path = Path(cfg.get("stall_params_path", None) or DEFAULT_PARAMS)
        if bool(cfg.get("recompute_params", False)) or not self.params_path.exists():
            print(f"[STALL] computing calibration params -> {self.params_path}", flush=True)
            params = self.fit_calibration(save_to=self.params_path)
        else:
            params = dict(np.load(self.params_path))
        self._set_params(params)

    def _build_backbone(self, name: str, repo_dir: str, weights: str):
        """Return the visual-encoder module exposing ``encode_frames(x) -> [N, D]``."""
        if name in ("dinov3_vitl16", "dinov3-vitl16", "dinov3_vit_l16", "dinov3", "dino3"):
            return _Dinov3Encoder(repo_dir, weights)
        raise ValueError(
            f"Unknown STALL backbone_name: {name!r} (only dinov3_vitl16 matches the shipped "
            f"calibration params)."
        )

    # ── calibration ─────────────────────────────────────────────────────────────
    def _set_params(self, p: dict):
        """Install whitening + calibration params (from a loaded npz or a fresh fit)."""
        self.mu_spat, self.W_spat = p["mu_spat"], p["W_spat"]
        self.mu_temp, self.W_temp = p["mu_temp"], p["W_temp"]
        self.calib_spat_sorted = np.sort(_AGG[self.spat_agg](p["calib_ll_spat"], axis=1))
        self.calib_temp_sorted = np.sort(_AGG[self.temp_agg](p["calib_ll_temp"], axis=1))

    def fit_calibration(self, loader=None, save_to=None) -> dict:
        """Compute calibration params from a REAL-video set, then ``fit_stall_params``.

        Uses the AGID stack so calibration sees pixels exactly like inference: video loading
        via the dataset layer, ImageNet preprocessing via the transform pipeline, and this
        model's frozen DINOv3 backbone for features. If ``loader`` is None it is built from
        ``self.config`` (``ModelConfig.calib_split``, default 'train'); only real clips
        (label 0) are used unless ``ModelConfig.calib_real_only`` is false. ``ModelConfig.
        calib_max_videos`` caps the set (0 = all). Saves to ``save_to`` when given.
        """
        cfg = self.config.ModelConfig
        if loader is None:
            from dataset import AIGIDatasetModule  # lazy: heavy import, only when recomputing
            dm = AIGIDatasetModule(self.config)
            split = str(cfg.get("calib_split", "train"))
            loader = {"train": dm.train_dataloader, "val": dm.val_dataloader,
                      "test": dm.test_dataloader}[split]()
        real_only = bool(cfg.get("calib_real_only", True))
        max_videos = int(cfg.get("calib_max_videos", 0))

        self.backbone.to(self.device).eval()
        chunks, n = [], 0
        for frames, labels, _folder_ids in loader:
            if real_only:
                frames = frames[labels == 0]
            if frames.shape[0] == 0:
                continue
            chunks.append(self.extract_embeddings(frames.to(self.device)))
            n += chunks[-1].shape[0]
            if max_videos and n >= max_videos:
                break
        if not chunks:
            raise RuntimeError("STALL calibration: no clips collected (check the dataset/split).")
        embs = np.concatenate(chunks, axis=0)
        if max_videos:
            embs = embs[:max_videos]
        params = fit_stall_params(embs, self.spat_agg, self.temp_agg)
        if save_to is not None:
            Path(save_to).parent.mkdir(parents=True, exist_ok=True)
            np.savez(save_to, **params)
            print(f"[STALL] fit calibration on {embs.shape[0]} clips -> {save_to}", flush=True)
        return params

    # ── branch log-likelihoods (numpy) ─────────────────────────────────────────
    def _spatial_ll(self, embs: np.ndarray) -> np.ndarray:
        """[N, T, D] -> [N, T]."""
        return isotropic_gaussian_loglik(whiten(embs, self.mu_spat, self.W_spat))

    def _temporal_ll(self, embs: np.ndarray) -> np.ndarray:
        """[N, T, D] -> [N, T-1]. Zero diffs (identical frames) -> +inf, excluded by min."""
        zero_mask = (np.linalg.norm(diff_vec(embs), axis=-1) == 0)
        ll = isotropic_gaussian_loglik(
            whiten(normalized_temporal_diff(embs), self.mu_temp, self.W_temp)
        )
        ll[zero_mask] = np.inf
        return ll

    def _final_real_score(self, embs: np.ndarray) -> np.ndarray:
        """[N, T, D] -> [N] in [0,1]; higher => more likely real."""
        spat = _AGG[self.spat_agg](self._spatial_ll(embs), axis=1)
        temp = _AGG[self.temp_agg](self._temporal_ll(embs), axis=1)
        spat_pct = percentile_vs_calibration(spat, self.calib_spat_sorted)
        temp_pct = percentile_vs_calibration(temp, self.calib_temp_sorted)
        return 0.5 * (spat_pct + temp_pct)

    @torch.no_grad()
    def extract_embeddings(self, frames: torch.Tensor) -> np.ndarray:
        """``frames`` [B, T, 3, H, W] (normalized by the dataset transform) -> [B, T, D] numpy.

        Reused both for scoring (forward) and for fitting calibration params, so the backbone /
        preprocessing are identical between calibration and inference."""
        b, t = frames.shape[:2]
        flat = frames.reshape(b * t, *frames.shape[2:])
        feats = self.backbone.encode_frames(flat).float()           # [B*T, D]
        return feats.reshape(b, t, -1).cpu().numpy()

    # ── AGID framework interface ────────────────────────────────────────────────
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """``frames`` [B, T, 3, H, W] (normalized by the dataset transform) -> [B, 2] = [real, fake]."""
        embs = self.extract_embeddings(frames)                      # [B, T, D]
        final = self._final_real_score(embs)                        # [B], higher = real
        final = torch.from_numpy(final).to(frames.device).float()
        return torch.stack([final, 1.0 - final], dim=1)

    def compute_loss(self, batch):
        frames, _labels, _folder_ids = batch
        outputs = self.forward(frames)
        loss = (self._noop * 0.0).sum()   # training-free; true no-op for optimizer compat
        return {"loss": loss, "outputs": outputs}
