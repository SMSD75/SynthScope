"""Model zoo: publish SynthScope checkpoints to the Hugging Face Hub and auto-download them.

``synthscope.load(...)`` transparently pulls a detector's weights from the Hub when they are not
already present under ``$SYNTHSCOPE_CKPT_DIR``. Maintainers publish with ``synthscope push``.

License safety (see LICENSES.md): by default ``push`` uploads only **SynthScope-trained** checkpoints,
which are this project's to redistribute. Checkpoints derived from authors' **official released
weights** (``ckpt_kind`` starting with ``official_`` — e.g. FatFormer, AIDE, NPR) are skipped
unless you pass ``--include-official`` AND have confirmed the upstream license permits it.
"""
from __future__ import annotations

import os
from pathlib import Path

from interface.backend import config as _config
from interface.backend.registry import list_specs

# Where the public zoo lives by default; override with $SYNTHSCOPE_HUB_REPO.
DEFAULT_REPO = os.environ.get("SYNTHSCOPE_HUB_REPO", "SMSD75/synthscope-checkpoints")

# Small SynthScope-produced auxiliary files (calibration stats) that some detectors read from the
# checkpoint dir. Safe to redistribute (produced by this project).
AUX_FILES = ["d3_calibration.json", "stall_params_genvideo.npz"]


def _ckpt_dir() -> Path:
    return _config.CHECKPOINTS_DIR


def is_official(spec) -> bool:
    """Whether a spec's checkpoint comes from the authors' official released weights."""
    return str(spec.ckpt_kind).startswith("official")


def repo_filename(key: str) -> str:
    """Canonical name of a detector's checkpoint inside the Hub repo."""
    return f"{key}.pth"


# ---------------------------------------------------------------------------
# Download (used by the public API)
# ---------------------------------------------------------------------------
def download(key: str, repo_id: str | None = None, token: str | None = None) -> Path:
    """Fetch ``<key>.pth`` from the Hub into ``$SYNTHSCOPE_CKPT_DIR`` and return its local path."""
    from huggingface_hub import hf_hub_download

    repo_id = repo_id or DEFAULT_REPO
    dest_dir = _ckpt_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=repo_filename(key),
        repo_type="model",
        local_dir=str(dest_dir),
        token=token,
    )
    # Best-effort: also pull a known aux file for this detector if the repo has one.
    _maybe_download_aux(key, repo_id, token)
    return Path(path)


def _maybe_download_aux(key: str, repo_id: str, token: str | None) -> None:
    from huggingface_hub import hf_hub_download

    wanted = {"d3": "d3_calibration.json", "stall": "stall_params_genvideo.npz"}.get(key)
    if not wanted:
        return
    try:
        hf_hub_download(repo_id=repo_id, filename=wanted, repo_type="model",
                        local_dir=str(_ckpt_dir()), token=token)
    except Exception:  # the aux file is optional — absence is fine
        pass


# ---------------------------------------------------------------------------
# Upload (maintainer side, via `synthscope push`)
# ---------------------------------------------------------------------------
def _select(keys: list[str] | None, include_official: bool):
    """Return (to_upload: list[(spec, resolved_path)], skipped: list[(key, reason)])."""
    ckpt_dir = _ckpt_dir()
    to_upload, skipped = [], []
    specs = [s for s in list_specs() if s.checkpoint]
    if keys:
        wanted = set(keys)
        specs = [s for s in specs if s.key in wanted]
    for s in specs:
        path = ckpt_dir / s.checkpoint
        if not path.exists():
            skipped.append((s.key, "no local checkpoint"))
            continue
        if is_official(s) and not include_official:
            skipped.append((s.key, "official weights (license-restricted; use --include-official)"))
            continue
        to_upload.append((s, path.resolve()))
    return to_upload, skipped


def _manifest_text(uploaded_keys: list[str], repo_id: str) -> str:
    lines = [
        "---", "license: apache-2.0", "library_name: synthscope",
        "tags:", "- ai-generated-image-detection", "- synthetic-image-detection",
        "- deepfake-detection", "---", "",
        "# SynthScope model zoo", "",
        "Checkpoints for [SynthScope](https://github.com/SMSD75/AGID) detectors. Load them with:", "",
        "```python", "import synthscope", f'# export SYNTHSCOPE_HUB_REPO="{repo_id}" (or pass repo=...)',
        'synthscope.load("clip").predict("image.png")  # auto-downloads from this repo', "```", "",
        "## Contents", "",
        "| Detector | File | Provenance |", "| --- | --- | --- |",
    ]
    for s in list_specs():
        if s.key in uploaded_keys:
            prov = "official weights" if is_official(s) else "SynthScope-trained"
            lines.append(f"| {s.key} | `{repo_filename(s.key)}` | {prov} |")
    lines += [
        "", "## License", "",
        "The SynthScope framework is Apache-2.0. Re-implementation-trained weights are released under "
        "Apache-2.0. Any weights derived from third-party official releases retain their upstream "
        "license — see the repo's `LICENSES.md`. Verify terms before downstream use.",
    ]
    return "\n".join(lines) + "\n"


def push(
    repo_id: str | None = None,
    keys: list[str] | None = None,
    include_official: bool = False,
    private: bool = True,
    token: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Publish checkpoints to ``repo_id`` (default ``$SYNTHSCOPE_HUB_REPO``).

    Returns a summary dict. With ``dry_run=True`` nothing is uploaded — it only reports the
    selection (use this to sanity-check before an unattended run).
    """
    repo_id = repo_id or DEFAULT_REPO
    to_upload, skipped = _select(keys, include_official)
    uploaded_keys = [s.key for s, _ in to_upload]

    print(f"Target repo : {repo_id} ({'private' if private else 'PUBLIC'})")
    print(f"Checkpoint dir: {_ckpt_dir()}")
    print(f"Will upload ({len(to_upload)}): {', '.join(uploaded_keys) or '(none)'}")
    for key, reason in skipped:
        print(f"  skip {key}: {reason}")
    aux_present = [a for a in AUX_FILES if (_ckpt_dir() / a).exists()]
    if aux_present:
        print(f"Aux files   : {', '.join(aux_present)}")

    if dry_run:
        print("[dry-run] no upload performed.")
        return {"repo_id": repo_id, "uploaded": uploaded_keys, "skipped": skipped, "dry_run": True}
    if not to_upload:
        print("Nothing to upload.")
        return {"repo_id": repo_id, "uploaded": [], "skipped": skipped, "dry_run": False}

    from huggingface_hub import HfApi, create_repo

    create_repo(repo_id, repo_type="model", private=private, exist_ok=True, token=token)
    api = HfApi(token=token)
    for spec, path in to_upload:
        name = repo_filename(spec.key)
        sz = path.stat().st_size / 1e6
        print(f"  uploading {name}  ({sz:.0f} MB) <- {path}")
        api.upload_file(path_or_fileobj=str(path), path_in_repo=name,
                        repo_id=repo_id, repo_type="model", token=token)
    for a in aux_present:
        print(f"  uploading {a}")
        api.upload_file(path_or_fileobj=str(_ckpt_dir() / a), path_in_repo=a,
                        repo_id=repo_id, repo_type="model", token=token)

    readme = _manifest_text(uploaded_keys, repo_id)
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="model", token=token)
    print(f"Done. https://huggingface.co/{repo_id}")
    return {"repo_id": repo_id, "uploaded": uploaded_keys, "skipped": skipped, "dry_run": False}
