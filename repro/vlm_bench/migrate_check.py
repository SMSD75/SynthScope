"""Check a ``migrate_vlm_cache`` rule-``unchanged`` migration (e.g. on a COPY of real caches): for every folder it wrote,
the records are the source folder's lines byte for byte (the winning line per key), ``bench_eval.VLMCache`` reads the same
records from it, and its name and identity.json are the current identity, which differs from the source's only in the
code version.

    cp -a <cache-root>/*/adb_vlm__* <copy>/...   (keep the <bench>/ level)
    python repro/vlm_bench/migrate_vlm_cache.py --cache-root <copy> --model adb_vlm
    python repro/vlm_bench/migrate_check.py --cache-root <copy> --model adb_vlm
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "interface"), str(REPO / "benchmark")]

import bench_eval                                                    # noqa: E402
from interface.backend.registry import get_spec                     # noqa: E402
from repro.vlm_bench import migrate_vlm_cache as mv                 # noqa: E402


def digest(lines: dict) -> str:
    return hashlib.sha1("\n".join(lines[k] for k in sorted(lines)).encode()).hexdigest()[:12]


def without_code(identity: dict) -> dict:
    return {k: v for k, v in identity.items() if k != "code_version"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args(argv)
    root, model = Path(args.cache_root), args.model
    total = bad = 0
    for marker in sorted(root.glob(f"*/{model}__*/MIGRATION.json")):
        m = json.loads(marker.read_text())
        if m.get("rule") != "unchanged":         # an earlier migration (e.g. cf179eee30 -> e5749ae) copied along
            continue
        new = marker.parent
        src = new.parent / Path(m["from"]).name
        src_lines, new_lines = {}, {}
        src_recs = mv.read_records(src, src_lines)
        new_recs = mv.read_records(new, new_lines)
        decoding = new.name[len(model) + 2:].split("__")[0]
        decoding = "official" if decoding.startswith("official") else decoding
        cur = bench_eval.vlm_identity(get_spec(model), decoding)
        with contextlib.redirect_stdout(io.StringIO()):
            cache = bench_eval.VLMCache(new.parent, new.name, cur)
        ok = (src_lines == new_lines and src_recs == new_recs and cache.records == src_recs
              and new.name.rpartition("__")[2] == mv.identity_hash(cur)
              and without_code(m["from_identity"]) == without_code(cur))
        print(f"{new.parent.name}/{src.name} -> {new.name}: {len(src_lines)} records, lines byte-equal "
              f"{src_lines == new_lines} (sha1 {digest(src_lines)} / {digest(new_lines)}), VLMCache reads the same records "
              f"{cache.records == src_recs}, code {m['from_identity']['code_version']} -> {cur['code_version']}: "
              f"{'OK' if ok else 'FAIL'}", flush=True)
        total += len(src_lines)
        bad += not ok
    print(f"migrate_check {model}: {total} records in the migrated folders, {bad} folder(s) failing")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
