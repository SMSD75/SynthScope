"""Carry VLM per-image cache records over a cache re-key that does not change them (one-time, per known re-key).

bench_eval keys a VLM cache folder (``<cache-root>/<bench>/<model>__<decoding>[__variant]__<hash>``) by a hash of
``vlm_identity``: an edit of ``models/vlm/*.py`` or a new identity field gives a new, empty folder. When the change is
known not to alter some records, this copies them from the older folder into the folder of the current identity and
leaves out the records it may alter, which the next bench_eval run then scores again.

Known re-keys (``PREDECESSORS``):

* adb_vlm, final-review fix wave (2026-11-18): from the identity of the committed results (code version cf179eee30,
  config cf. 65977eff07) to the current one. What changed:
    - ``verdict_offset`` (``parsers.adb_verdict_offset``): an output with no ``<answer>`` tag whose verdict the official
      parser reads from the tag-less final line is now scored at that line's "AI" / "authentic" token; before, it had no
      verdict token (``score_source`` "unparsed" with a parsed verdict, logit 0). Every other output takes the unchanged
      search after ``<answer>``.
    - ``preprocess=swift_qwen_vl_resize`` with ``preprocess_kwargs={"max_pixels": 1003520}``: the same call as the former
      ``adb_preprocess`` wrapper, now visible to the cache identity; ``base_weights_revision`` added to the identity
      (cc594898, the revision every run used); docstrings and the ``PEFT_PREFIX`` move (8e66aee).
  Dropped (rescored): the records with a parsed verdict and no verdict token. Every kept record must have
  ``verdict_offset(response) is None`` (its scoring path is unchanged), else the script stops.

    python repro/vlm_bench/migrate_vlm_cache.py --cache-root /ssdstore/ssalehi/SynthScope/vpp/cache --model adb_vlm [--dry-run]

The new folder gets ``migrated.<old hash>.jsonl`` (the kept records), ``identity.json`` and ``MIGRATION.json`` (source folder,
old identity, counts, dropped keys). A folder already migrated from the same source is left alone. Run it with the code the
scoring jobs will run (a clean checkout of the same commit): the current identity includes the code version.
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

PREDECESSORS = {
    "adb_vlm": [{
        "weights_revision": "49975ccd8b28eaab8ef14650ffee0c545934d02b", "ckpt": None, "max_new_tokens": 512,
        "generation": {"do_sample": False, "max_new_tokens": 512, "repetition_penalty": 1.05},
        "prompt_sha1": "91db63b292", "config_sha1": "65977eff07", "code_version": "cf179eee30",
    }],
}


def identity_hash(identity: dict) -> str:
    return hashlib.sha1(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:10]


def read_records(folder: Path) -> dict:
    """{key: record} as ``bench_eval.VLMCache`` reads a folder (every .jsonl, a later line of a key wins)."""
    records = {}
    for f in sorted(folder.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            records[rec["key"]] = rec
    return records


def unscored(rec: dict) -> bool:
    """A parsed verdict without a verdict token (logit 0, P(fake) 0.5)."""
    return rec["verdict"] is not None and rec["score_source"] in bench_eval.NO_SCORE_SOURCES


def migrate_folder(old: Path, model: str, dry_run: bool) -> dict:
    spec = get_spec(model)
    old_id = json.loads((old / "identity.json").read_text())
    stem, _, old_hash = old.name.rpartition("__")
    if identity_hash(old_id) != old_hash:
        raise SystemExit(f"{old}: identity.json does not hash to the folder name ({identity_hash(old_id)} != {old_hash})")
    decoding = old.name[len(model) + 2:].split("__")[0]
    decoding = "official" if decoding.startswith("official") else decoding
    new_id = bench_eval.vlm_identity(spec, decoding)
    if old_id == new_id:
        return {"folder": str(old), "status": "current"}
    if old_id not in PREDECESSORS.get(model, []):
        return {"folder": str(old), "status": "unknown identity, left alone"}
    new = old.parent / f"{stem}__{identity_hash(new_id)}"
    records = read_records(old)
    dropped = sorted(k for k, r in records.items() if unscored(r))
    kept = [r for r in records.values() if not unscored(r)]
    hook = spec.vlm.verdict_offset
    changed = [r["key"] for r in kept if hook is not None and hook(r["response"]) is not None]
    if changed:
        raise SystemExit(f"{old}: {len(changed)} kept record(s) would be scored at the parser's verdict word, e.g. {changed[:3]}")
    summary = {"folder": str(old), "to": str(new), "records": len(records), "kept": len(kept), "dropped": len(dropped)}
    marker = new / "MIGRATION.json"
    if marker.exists() and json.loads(marker.read_text())["from"] == str(old):
        return {**summary, "status": "migrated earlier"}
    if new.exists() and any(new.glob("*.jsonl")):
        raise SystemExit(f"{new} already holds records not migrated from {old}; not mixing them")
    if dry_run:
        return {**summary, "status": "dry run"}
    new.mkdir(parents=True, exist_ok=True)
    with (new / f"migrated.{old_hash}.jsonl").open("w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    (new / "identity.json").write_text(json.dumps(new_id, indent=1) + "\n")
    marker.write_text(json.dumps({
        "from": str(old), "from_identity": old_id, "to_identity": new_id, "records": len(records), "kept": len(kept),
        "dropped_unscored": dropped, "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "script": "repro/vlm_bench/migrate_vlm_cache.py"}, indent=1) + "\n")
    return {**summary, "status": "migrated"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache-root", required=True, help="the run_units --cache-root (one subfolder per benchmark)")
    ap.add_argument("--model", required=True, choices=sorted(PREDECESSORS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    for ident in sorted(Path(args.cache_root).glob(f"*/{args.model}__*/identity.json")):
        print(json.dumps(migrate_folder(ident.parent, args.model, args.dry_run)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
