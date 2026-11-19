"""CPU test of the GenBuster-Bench / GenBuster-Bench++ loaders (dataset.py) on toy zips with the released folder layout (no
network: ``hf_hub_download`` is replaced by a function that returns the toy zip).

Checks:
* GenBuster-Bench: ``GenBuster-Bench/{id/real, id/fake/<gen>, ood/fake/<gen>, wild/fake/<gen>}`` becomes
  ``test/<split>_<gen>/1_fake`` with every real hard-linked into every ``test/<split>_<gen>/0_real``, in sorted order;
  non-video files and macOS ``._`` stubs are skipped; ``SOURCE.json`` records the pinned revision, the archive sha256 and
  the counts; the download and staging folders are removed;
* the pinned revision and ``token=False`` are passed to ``hf_hub_download``;
* GenBuster-Bench++: only the video track (``video/{real,fake}``) is laid out (one group ``video``), the image track is not;
* a second call is a no-op (no download), and ``bench_eval_video.discover`` reads the layout as pergen.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "interface"), str(ROOT / "benchmark")]

import huggingface_hub                                                            # noqa: E402

import dataset as ds                                                              # noqa: E402

_spec = importlib.util.spec_from_file_location("video_layout", ROOT / "benchmark" / "video_layout.py")
video_layout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(video_layout)


def make_zip(path: Path, members: dict) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def fake_download(zip_path: Path, calls: list):
    def download(**kw):
        calls.append(kw)
        out = Path(kw["local_dir"]) / kw["filename"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(zip_path.read_bytes())
        return str(out)
    return download


def test_genbuster_bench(tmp: Path) -> None:
    members = {"GenBuster-Bench/": b"", "GenBuster-Bench/id/real/b.mp4": b"rb", "GenBuster-Bench/id/real/a.mp4": b"ra",
               "GenBuster-Bench/id/real/._a.mp4": b"stub", "GenBuster-Bench/id/real/notes.txt": b"x",
               "GenBuster-Bench/id/fake/cogvideox/c2.mp4": b"c2", "GenBuster-Bench/id/fake/cogvideox/c1.mp4": b"c1",
               "GenBuster-Bench/ood/fake/sora/s1.mp4": b"s1", "GenBuster-Bench/wild/fake/veo3.1/v1.mp4": b"v1"}
    zp = tmp / "gb.zip"
    make_zip(zp, members)
    calls = []
    huggingface_hub.hf_hub_download = fake_download(zp, calls)
    root = tmp / "GenBuster-Bench"
    ds.GenBusterBenchDataset.prepare_split(root, "test")
    assert len(calls) == 1 and calls[0]["revision"] == ds.GenBusterBenchDataset.revision and calls[0]["token"] is False
    assert calls[0]["repo_id"] == "l8cv/GenBuster-Bench" and calls[0]["repo_type"] == "dataset"
    test = root / "test"
    assert sorted(p.name for p in test.iterdir()) == ["id_cogvideox", "ood_sora", "wild_veo3.1"]
    for g in ("id_cogvideox", "ood_sora", "wild_veo3.1"):
        reals = sorted(p.name for p in (test / g / "0_real").iterdir())
        assert reals == ["00000_a.mp4", "00001_b.mp4"], reals
        assert (test / g / "0_real" / "00000_a.mp4").read_bytes() == b"ra"
    assert (test / "id_cogvideox" / "0_real" / "00000_a.mp4").stat().st_nlink >= 3          # hard links, not copies
    assert sorted(p.name for p in (test / "id_cogvideox" / "1_fake").iterdir()) == ["00000_c1.mp4", "00001_c2.mp4"]
    src = json.loads((root / "SOURCE.json").read_text())
    assert src["archive_sha256"] == hashlib.sha256(zp.read_bytes()).hexdigest()
    assert src["reals"] == 2 and src["fakes"] == {"id_cogvideox": 2, "ood_sora": 1, "wild_veo3.1": 1}
    assert src["revision"] == "68e41ab55726013f1c7455980180114302196c23"
    assert not (root / "_downloads").exists() and not (root / "_staging").exists()
    ds.GenBusterBenchDataset.prepare_split(root, "test")                       # already there: no download
    assert len(calls) == 1
    gens = video_layout.discover("pergen", test, None, None)
    assert [(n, len(r), len(f)) for n, r, f in gens] == [("id_cogvideox", 2, 2), ("ood_sora", 2, 1), ("wild_veo3.1", 2, 1)]
    print("GenBuster-Bench layout OK")


def test_genbuster_pp(tmp: Path) -> None:
    members = {"image/real/i1.jpg": b"i", "image/fake/i2.jpg": b"j", "video/real/r2.mp4": b"r2", "video/real/r1.mp4": b"r1",
               "video/fake/f1.mp4": b"f1"}
    zp = tmp / "gbpp.zip"
    make_zip(zp, members)
    calls = []
    huggingface_hub.hf_hub_download = fake_download(zp, calls)
    root = tmp / "GenBuster-Bench-plusplus"
    ds.GenBusterBenchPPDataset.prepare_split(root, "test")
    assert calls[0]["repo_id"] == "l8cv/GenBuster-Bench-plusplus" and calls[0]["revision"] == ds.GenBusterBenchPPDataset.revision
    test = root / "test"
    assert sorted(p.name for p in test.iterdir()) == ["video"]
    assert sorted(p.name for p in (test / "video" / "0_real").iterdir()) == ["00000_r1.mp4", "00001_r2.mp4"]
    assert sorted(p.name for p in (test / "video" / "1_fake").iterdir()) == ["00000_f1.mp4"]
    assert not any(p.suffix == ".jpg" for p in test.rglob("*"))
    assert json.loads((root / "SOURCE.json").read_text())["fakes"] == {"video": 1}
    assert "genbuster" in ds.SynthScopeModuleDict and "genbuster_pp" in ds.SynthScopeModuleDict
    print("GenBuster-Bench++ layout OK")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        test_genbuster_bench(Path(d))
        test_genbuster_pp(Path(d))
    print("test_genbuster_loader: all OK")
