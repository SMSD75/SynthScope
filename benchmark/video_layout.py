"""The video benchmark layouts read by ``benchmark/bench_eval_video.py`` (standard library only, so that tools which only
plan or verify a selection, e.g. ``repro/video_agent_bench/make_subsets.py``, run without torch or the engine).

  genvideo   <data>/Fake/<gen>/*.mp4 + <data>/Real/*.mp4 (shared real pool)   [GenVideo-Val]
  fakeonly   <data>/<gen>/*.mp4 (all fake) + a shared real pool              [EvalCrafter]
  pergen     <data>/<gen>/{0_real,1_fake}/*.mp4                                [GenVidBench, the SynthScope loaders]
"""
from __future__ import annotations

from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif", ".m4v"}
REAL_DIRNAMES = ("0_real", "real", "reals", "Real")
FAKE_DIRNAMES = ("1_fake", "fake", "fakes", "Fake")


def _videos(d: Path) -> list[Path]:
    if not d or not d.is_dir():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def _find(parent: Path, names) -> Path | None:
    for n in names:
        if (parent / n).is_dir():
            return parent / n
    return None


def discover(benchmark: str, root: Path, shared_reals: Path | None, select):
    """[(generator, reals, fakes)] in sorted generator order, each list sorted by path."""
    shared = _videos(shared_reals) if shared_reals else []
    out = []
    if benchmark == "genvideo":
        fake_root = _find(root, FAKE_DIRNAMES)
        real_root = _find(root, REAL_DIRNAMES)
        reals = _videos(real_root) or shared
        gens = sorted(p for p in fake_root.iterdir() if p.is_dir()) if fake_root else []
        for g in gens:
            if select and g.name not in select:
                continue
            fk = _videos(g)
            if fk and reals:
                out.append((g.name, reals, fk))
        return out
    if benchmark == "fakeonly":
        gens = sorted(p for p in root.iterdir() if p.is_dir())
        for g in gens:
            if select and g.name not in select:
                continue
            fk = _videos(g)
            if fk and shared:
                out.append((g.name, shared, fk))
        return out
    # pergen
    for g in sorted(p for p in root.iterdir() if p.is_dir()):
        if select and g.name not in select:
            continue
        reals = _videos(_find(g, REAL_DIRNAMES)) or shared
        fakes = _videos(_find(g, FAKE_DIRNAMES))
        if reals and fakes:
            out.append((g.name, reals, fakes))
    return out
