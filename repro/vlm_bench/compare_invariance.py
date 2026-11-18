"""Check that adding a VLM column to comparison.json left every existing value as it was.

Walks the committed (older) comparison.json and the new one: every value of the old file must be equal in the new one
(subset rows matched by benchmark + variant), except ``models`` and the ``rank_all_*`` ranks of the existing VLMs, which by
definition also count the new VLM; those changes are listed one by one. Each existing VLM's ``rank_all_<metric>`` must
change by exactly the number of new VLMs strictly better than it on that row and metric (+1 or 0 for one new VLM), the
rule compare.py ranks by. Keys only in the new file are summarised.

    python repro/vlm_bench/compare_invariance.py --old <git rev> [--new repro/vlm_bench/comparison.json]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ALLOWED = re.compile(r"^models$|^subsets\[[^\]]+\]\.vlms\.[^.]+\.rank_all_(acc|mauroc)$")


def keyed(doc: dict) -> dict:
    """The subset rows as a dict keyed by benchmark + variant (their order in the list is not a value)."""
    doc = dict(doc)
    doc["subsets"] = {f"{r['benchmark']}:{r['variant']}": r for r in doc["subsets"]}
    return doc


def walk(old, new, path: str, changed: list, added: list) -> None:
    if isinstance(old, dict) and isinstance(new, dict):
        for k in old:
            p = f"{path}[{k}]" if path == "subsets" else (f"{path}.{k}" if path else k)
            if k not in new:
                changed.append((p, old[k], "<missing>"))
            else:
                walk(old[k], new[k], p, changed, added)
        added += [f"{path}.{k}" if path else k for k in new if k not in old]
    elif old != new:
        changed.append((path, old, new))


def rank_all_mismatches(old: dict, new: dict) -> list:
    """(path, old rank, new rank, expected rank) for every existing VLM's ``rank_all_*`` that did not move by the number of
    new VLMs strictly better on that row and metric (``old`` / ``new`` keyed)."""
    added = [m for m in new["models"] if m not in old["models"]]
    bad = []
    for key, orow in old["subsets"].items():
        row = new["subsets"].get(key)
        if row is None:
            continue
        for vlm, ov in orow.get("vlms", {}).items():
            nv = row["vlms"].get(vlm, {})
            for metric in ("acc", "mauroc"):
                k = f"rank_all_{metric}"
                if k not in ov or nv.get(metric) is None:
                    continue
                better = sum(1 for m in added if row["vlms"].get(m, {}).get(metric) is not None
                             and row["vlms"][m][metric] > nv[metric])
                if nv.get(k) != ov[k] + better:
                    bad.append((f"subsets[{key}].vlms.{vlm}.{k}", ov[k], nv.get(k), ov[k] + better))
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--old", required=True, help="git revision of the committed comparison.json")
    ap.add_argument("--new", default=str(HERE / "comparison.json"))
    args = ap.parse_args(argv)
    old = json.loads(subprocess.check_output(["git", "-C", str(REPO), "show", f"{args.old}:repro/vlm_bench/comparison.json"]))
    new = json.loads(Path(args.new).read_text())
    changed, added = [], []
    walk(keyed(old), keyed(new), "", changed, added)
    bad = [c for c in changed if not ALLOWED.match(c[0])]
    for p, o, n in changed:
        print(f"{'ALLOWED' if ALLOWED.match(p) else 'CHANGED'} {p}: {json.dumps(o)} -> {json.dumps(n)}")
    pattern = Counter(re.sub(r"\[[^\]]+\]", "[*]", a) for a in added)
    for p, n in sorted(pattern.items()):
        print(f"added {p} (x{n})")
    ranks = rank_all_mismatches(keyed(old), keyed(new))
    for p, o, n, want in ranks:
        print(f"RANK_ALL {p}: {o} -> {n}, expected {want} (old + the new VLMs strictly better)")
    print(f"compare_invariance vs {args.old}: {len(changed) - len(bad)} allowed change(s) (models, rank_all_*), "
          f"{len(bad)} other value(s) changed, {len(ranks)} rank_all value(s) off the +1-per-better-new-VLM rule")
    return 1 if bad or ranks else 0


if __name__ == "__main__":
    raise SystemExit(main())
