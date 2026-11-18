"""Check that adding a VLM column to comparison.json left every existing value as it was.

Walks the committed (older) comparison.json and the new one: every value of the old file must be equal in the new one
(subset rows matched by benchmark + variant), except ``models`` and the ``rank_all_*`` ranks of the existing VLMs, which by
definition also count the new VLM; those changes are listed one by one. Keys only in the new file are summarised.

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
    print(f"compare_invariance vs {args.old}: {len(changed) - len(bad)} allowed change(s) (models, rank_all_*), "
          f"{len(bad)} other value(s) changed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
