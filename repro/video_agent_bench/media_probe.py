"""Media facts behind two statements in the wevis docs, written to repro/video_agent_bench/msrvtt_probe.json:

1. ``--part media`` (run where the held subsets live, ivi-cn029): the frame size, frame rate and frame count of every
   distinct REAL video of the four held subsets (GenVideo-Val, VideoPhy, VidProM, EvalCrafter; all MSR-VTT clips from
   GenVideo-Val), and a summary of the fakes' frame sizes and rates for contrast. Each file is read with OpenCV after its
   sha1 is checked against the committed manifest (subsets/<bench>.json).
2. ``--part audio`` (run on the agent's score caches, which hold the full traces): per subset, how many distinct videos
   have an audio stream (``trace.audio_stream_present``, probed by the official ``core.media.probe_media`` in the runner),
   split by label (from results/<bench>/wevis.json). Every video was run with has_audio=False either way.

    python repro/video_agent_bench/media_probe.py --part media --out repro/video_agent_bench/msrvtt_probe.json
    python repro/video_agent_bench/media_probe.py --part audio --agent-cache DIR --out repro/video_agent_bench/msrvtt_probe.json

Each part replaces its own key in --out and keeps the other.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import statistics
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
HELD = ("genvideo", "videophy", "vidprom", "evalcrafter")
ALL = HELD + ("genbuster", "genbuster_pp")


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(manifest: dict, rel: str, root: str) -> Path:
    if rel.startswith("@reals/"):
        return Path(root) / manifest["reals"] / rel[len("@reals/"):]
    return Path(root) / manifest["data"] / rel


def props(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    out = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
           "fps": round(cap.get(cv2.CAP_PROP_FPS), 3), "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    cap.release()
    return out


def part_media(root: str | None) -> dict:
    reals, fakes = {}, {}
    for bench in HELD:
        m = json.loads((HERE / "subsets" / f"{bench}.json").read_text())
        base = root or m["source_root"]
        for g in m["generators"]:
            for kind, rels in (("real", g["reals"]), ("fake", g["fakes"])):
                for rel in rels:
                    want, _size = m["files"][rel]
                    store = reals if kind == "real" else fakes
                    if want in store:
                        store[want]["subsets"].add(bench)
                        continue
                    path = resolve(m, rel, base)
                    got = sha1(path)
                    if got != want:
                        raise SystemExit(f"{path}: sha1 {got}, manifest {want}")
                    store[want] = {"name": rel.split("/")[-1], "subsets": {bench}, **props(path)}

    def summary(store: dict) -> dict:
        sizes = collections.Counter(f"{v['width']}x{v['height']}" for v in store.values())
        rates = collections.Counter(str(v["fps"]) for v in store.values())
        return {"n_distinct": len(store), "frame_sizes": dict(sizes.most_common()), "fps": dict(rates.most_common()),
                "median_height": statistics.median(v["height"] for v in store.values()),
                "median_frames": statistics.median(v["frames"] for v in store.values())}
    return {
        "what": "distinct videos of the four held subsets, by sha1 (the subsets share the GenVideo-Val MSR-VTT reals); "
                "OpenCV header properties after a sha1 check against subsets/<bench>.json",
        "reals": summary(reals),
        "fakes": summary(fakes),
        "real_files": sorted(({"sha1": k, "name": v["name"], "subsets": sorted(v["subsets"]), "width": v["width"],
                               "height": v["height"], "fps": v["fps"], "frames": v["frames"]}
                              for k, v in reals.items()), key=lambda r: r["name"]),
    }


def part_audio(cache: str) -> dict:
    out = {}
    for bench in ALL:
        labels = {r["k"]: r["label"] for r in json.loads((HERE / "results" / bench / "wevis.json").read_text())["per_video"]}
        seen = {}
        for f in sorted(glob.glob(os.path.join(cache, bench, "traces.*.jsonl"))):
            for line in open(f):
                r = json.loads(line)
                if r["k"] in labels:
                    seen[r["k"]] = bool(r["trace"].get("audio_stream_present"))
        missing = sorted(set(labels) - set(seen))
        out[bench] = {"n_videos": len(labels), "n_traced": len(seen), "missing_traces": len(missing),
                      "audio_stream": {"real": sum(v for k, v in seen.items() if labels[k] == 0),
                                       "fake": sum(v for k, v in seen.items() if labels[k] == 1)},
                      "n_real": sum(1 for k in seen if labels[k] == 0), "n_fake": sum(1 for k in seen if labels[k] == 1)}
    return {"what": "distinct videos with an audio stream (trace.audio_stream_present, the official probe_media in the "
                    "runner), by label; every video was run with has_audio=False", "per_subset": out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["media", "audio"], required=True)
    ap.add_argument("--root", help="media: the folder holding the subsets' data (default: each manifest's source_root)")
    ap.add_argument("--agent-cache", help="audio: the agent's score-cache root (<root>/<bench>/traces.*.jsonl)")
    ap.add_argument("--out", default=str(HERE / "msrvtt_probe.json"))
    args = ap.parse_args()
    out = Path(args.out)
    doc = json.loads(out.read_text()) if out.exists() else {}
    if args.part == "media":
        doc["media"] = part_media(args.root)
    else:
        if not args.agent_cache:
            raise SystemExit("--agent-cache is required for --part audio")
        doc["audio"] = part_audio(args.agent_cache)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: (v.get("reals", v.get("per_subset")) if isinstance(v, dict) else v) for k, v in doc.items()},
                     indent=1)[:3000])


if __name__ == "__main__":
    main()
