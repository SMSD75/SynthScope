"""Seed a VLM per-image cache folder from a COMMITTED bench_eval result JSON, so a larger run reuses the images that run
scored (e.g. the 63-per-class GenImage subset inside the first 625 per class) without scoring them again.

The committed records are the cache content of the run that wrote them (verdict, logit, tokens; traced to its job log by
``repro/vlm_bench/trace_check.py``). A record is written to the cache only when:
* the file it scored is provably the same: the subset manifest's sha1 of its key (``repro/vlm_bench/subsets``) equals the
  sha1 of the file under ``--data`` (the record is stored under that sha1, as ``bench_eval.VLMCache`` checks it);
* the result's VLM identity equals the current one apart from the code version, and that code version change is one
  ``repro/vlm_bench/migrate_vlm_cache.py`` lists as ``unchanged`` for this model (the scoring path is the same).

The target folder is the current identity's (``bench_eval.vlm_variant`` with the result's ``jpeg_fakes`` / degradation);
it gets ``seeded.<result stem>.jsonl`` (each record marked ``seeded_from``) and ``SEED.json``. A folder that already
holds other records is left alone.

    python repro/adb_full/seed_vlm_cache.py --result repro/vlm_bench/results/genimage_val100k/adb_vlm__jpegfakes96.json \
        --manifest genimage_val100k --data /ssdstore/.../GenImage_val_full/test --cache-dir /ssdstore/.../vpp/cache/genimage_val100k
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "interface"), str(REPO / "benchmark")]

import bench_eval                                                    # noqa: E402
from interface.backend.registry import get_spec                     # noqa: E402
from repro.vlm_bench import migrate_vlm_cache as mv                  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--result", required=True)
    ap.add_argument("--manifest", required=True, help="subset manifest name (repro/vlm_bench/subsets/<name>.json)")
    ap.add_argument("--data", required=True, help="the folder the new run reads (keys are relative to it)")
    ap.add_argument("--cache-dir", required=True, help="<cache-root>/<bench>")
    args = ap.parse_args(argv)
    res = json.loads(Path(args.result).read_text())
    model, vlm = res["model"], res["vlm"]
    spec = get_spec(model)
    cur = bench_eval.vlm_identity(spec, vlm["decoding"])
    old = vlm["identity"]
    if {k: v for k, v in old.items() if k != "code_version"} != {k: v for k, v in cur.items() if k != "code_version"}:
        raise SystemExit(f"{args.result}: identity differs from the current one beyond the code version")
    if old["code_version"] != cur["code_version"]:
        ok = any(e["rule"] == "unchanged" and e["identity"] == old and cur["code_version"] in e["to_code_versions"]
                 for e in mv.PREDECESSORS.get(model, []))
        if not ok:
            raise SystemExit(f"no checked 'unchanged' re-key {old['code_version']} -> {cur['code_version']} for {model}")
    files = json.loads((REPO / "repro" / "vlm_bench" / "subsets" / f"{args.manifest}.json").read_text())["files"]
    folder = Path(args.cache_dir) / bench_eval.vlm_variant(model, vlm["decoding"], vlm.get("seed"), res.get("degrade"),
                                                           res.get("jpeg_fakes"), cur)
    stem = Path(args.result).stem
    if folder.exists() and any(f.name != f"seeded.{stem}.jsonl" for f in folder.glob("*.jsonl")):
        print(json.dumps({"folder": str(folder), "status": "holds other records, left alone"}))
        return 0
    data = Path(args.data)
    out, bad = [], []
    for r in res["per_image"]:
        sha1 = hashlib.sha1((data / r["key"]).read_bytes()).hexdigest()
        if files[r["key"]][0] != sha1:
            bad.append(r["key"])
            continue
        out.append({"key": r["key"], "sha1": sha1, "logit": r["logit"], "verdict": r["verdict"],
                    "score_source": r["score_source"], "new_tokens": r["new_tokens"], "finished": r["finished"],
                    "verdict_token": None, "input_size": None, "seconds": r["seconds"], "decoding": vlm["decoding"],
                    "seed": vlm.get("seed"), "response": None, "seeded_from": f"{args.result} ({res['tag']})"})
    if bad:
        raise SystemExit(f"{len(bad)} file(s) differ from the manifest, e.g. {bad[:3]}")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "identity.json").write_text(json.dumps(cur, indent=1) + "\n")
    (folder / f"seeded.{stem}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    (folder / "SEED.json").write_text(json.dumps({
        "from": args.result, "tag": res["tag"], "from_identity": old, "to_identity": cur, "records": len(out),
        "manifest": args.manifest, "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "script": "repro/adb_full/seed_vlm_cache.py"}, indent=1) + "\n")
    print(json.dumps({"folder": str(folder), "records": len(out), "status": "seeded"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
