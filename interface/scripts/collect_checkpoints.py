"""Assemble the central checkpoint directory used by the web interface.

- Symlinks the already-available OFFICIAL checkpoints (FatFormer, AIDE, NPR) to the canonical
  filenames the registry expects.
- Writes a MANIFEST.{json,md} summarizing, per model, whether a checkpoint is present and how
  it was obtained (official weights / trained-from-scratch / training-free).

Run:  conda run -n SynthScope python interface/scripts/collect_checkpoints.py
Override the destination with SYNTHSCOPE_CKPT_DIR.  ``--check`` only verifies that the source files exist.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import config            # noqa: E402  (sets paths)
from backend.registry import list_specs  # noqa: E402

# Known source locations of the official weights on this node. Every entry must be the file the registry (and the
# public Space) actually serves under that name:
#   fat_former.pth  official 4-class ProGAN FatFormer release
#   aide.pth        official SD-v1.4 release (sd14_train.pth), not the ProGAN-trained progan_train.pth
#   npr.pth         official NPR.pth with the DataParallel ``module.`` prefix stripped (NPR_table1.pth, linked 2026-06-09),
#                   not the older model_epoch_last_3090.pth
REPRO_CKPTS = Path("/hddstore/ssalehi/SynthScope/repro/ckpts")
AIDE_OFFICIAL = Path("/hddstore/ssalehi/SynthScope/repro/official_ckpts/aide/AIDE_checkpoints")
OFFICIAL_SOURCES = {
    "fat_former.pth": REPRO_CKPTS / "fatformer_official.pth",
    "aide.pth": AIDE_OFFICIAL / "sd14_train.pth",
    "npr.pth": REPRO_CKPTS / "NPR_table1.pth",
}


def missing_sources(sources: dict = None) -> list:
    """Names whose source file does not exist (on a node without /hddstore this is all of them)."""
    sources = OFFICIAL_SOURCES if sources is None else sources
    return [name for name, src in sources.items() if not src.exists()]


def link(dest: Path, src: Path) -> str:
    if not src.exists():
        return f"source missing: {src}"
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    dest.symlink_to(src)
    return f"linked -> {src}"


def main():
    if "--check" in sys.argv[1:]:       # only verify that every source exists; link nothing, write nothing
        missing = missing_sources()
        for name in missing:
            print(f"WARNING: source for {name} is missing: {OFFICIAL_SOURCES[name]}", file=sys.stderr)
        print(f"{len(OFFICIAL_SOURCES) - len(missing)}/{len(OFFICIAL_SOURCES)} official sources present")
        sys.exit(1 if missing else 0)
    for name in missing_sources():
        print(f"WARNING: source for {name} is missing: {OFFICIAL_SOURCES[name]} "
              f"(an existing {name} link is left unchanged)", file=sys.stderr)
    dst = config.CHECKPOINTS_DIR
    dst.mkdir(parents=True, exist_ok=True)
    print(f"checkpoints dir: {dst}")

    for name, src in OFFICIAL_SOURCES.items():
        print(f"  {name:18s} {link(dst / name, src)}")

    manifest = {"checkpoints_dir": str(dst), "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "models": []}
    for spec in list_specs():
        if spec.ckpt_kind == "none":
            status, present = "training-free", True
        else:
            p = dst / spec.checkpoint
            present = p.exists()
            if spec.ckpt_kind.startswith("official"):
                status = "official weights" if present else "MISSING (official)"
            else:
                status = "trained from scratch" if present else "MISSING (needs training)"
        manifest["models"].append({
            "key": spec.key, "name": spec.name, "modality": spec.modality,
            "checkpoint": spec.checkpoint, "present": present, "status": status,
        })

    (dst / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    lines = ["# SynthScope checkpoints manifest", "",
             f"Directory: `{dst}`  ", f"Generated: {manifest['generated']}", "",
             "| Model | Modality | Checkpoint | Present | Status |",
             "| --- | --- | --- | --- | --- |"]
    for m in manifest["models"]:
        lines.append(f"| {m['key']} | {m['modality']} | {m['checkpoint'] or '—'} | "
                     f"{'✅' if m['present'] else '❌'} | {m['status']} |")
    (dst / "MANIFEST.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {dst/'MANIFEST.json'} and {dst/'MANIFEST.md'}")
    present = sum(m["present"] for m in manifest["models"])
    print(f"present: {present}/{len(manifest['models'])} models")


if __name__ == "__main__":
    main()
