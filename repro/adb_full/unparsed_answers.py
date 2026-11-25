"""The VLM answers of the cf168db runs that the official parser rejects (``parse_vlm_verdict`` raises; ``predict.py`` since
898b607 decides Real with ``vlm_answer_parsed: false``). The result JSONs keep the decision ("real"), not the text, so this
reads the per-image VLM cache (one JSON line per image, with the response) of the current ``adb_vlm`` identity on the node that
scored them, and checks every record against the result JSONs of ``repro/adb_full/vlm_cf168db`` (same key, sha1, verdict).

    python repro/adb_full/unparsed_answers.py --cache-root /ssdstore/ssalehi/SynthScope/vpp/cache

Output: ``repro/adb_full/vlm_cf168db/unparsed_answers.json``: per benchmark folder, the records, the unparsed ones (key,
label, score source, the end of the response) and how many result records were checked.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO), str(REPO / "interface"), str(REPO / "benchmark")]

import bench_eval                                                    # noqa: E402
from interface.backend.registry import get_spec                     # noqa: E402
from models.vlm.parsers import parse_adb, parse_adb_default_real    # noqa: E402

RESULTS = HERE / "vlm_cf168db"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    args = ap.parse_args(argv)
    identity = bench_eval.vlm_identity(get_spec("adb_vlm"), "greedy")
    out = {"identity": identity, "benchmarks": {}}
    for bench_dir in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
        results = [json.loads(f.read_text()) for f in sorted(bench_dir.glob("adb_vlm*.json"))]
        recs = {}
        for jpeg in sorted({r.get("jpeg_fakes") for r in results}, key=str):
            folder = Path(args.cache_root) / bench_dir.name / bench_eval.vlm_variant("adb_vlm", "greedy", None, None, jpeg,
                                                                                     identity)
            with open("/dev/null", "w") as null:
                stdout, sys.stdout = sys.stdout, null
                try:
                    cache = bench_eval.VLMCache(folder.parent, folder.name, None)
                finally:
                    sys.stdout = stdout
            recs[jpeg] = cache.records
        checked = 0
        for res in results:
            for r in res["per_image"]:
                c = recs[res.get("jpeg_fakes")][r["key"]]
                assert (c["sha1"], c["verdict"]) == (r["sha1"], r["verdict"]), r["key"]
                assert parse_adb_default_real(c["response"]) == c["verdict"], r["key"]
                checked += 1
        labels = {r["key"]: r["label"] for res in results for r in res["per_image"]}
        used = {k: c for d in recs.values() for k, c in d.items() if k in labels}
        bad = [{"key": k, "label": labels[k], "score_source": c["score_source"], "response_end": c["response"][-160:]}
               for k, c in sorted(used.items()) if parse_adb(c["response"]) is None]
        out["benchmarks"][bench_dir.name] = {"records": len(used), "result_records_checked": checked,
                                             "unparsed": len(bad), "unparsed_answers": bad}
        print(f"{bench_dir.name}: {len(used)} records ({checked} result records checked), {len(bad)} unparsed", flush=True)
    (RESULTS / "unparsed_answers.json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
