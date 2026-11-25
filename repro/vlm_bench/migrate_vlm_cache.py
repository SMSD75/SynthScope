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
* adb_vlm and veritaspp, the FakeVLM branch (detector/fakevlm, 2026-11-18): from the identity of main e5749ae (code
  version b490c64477) to the branch's (02852acf62, then 4a950958ed). Only ``models/vlm/*.py`` changed, and only by
  additions that neither detector uses: ``LlavaFamily``, the ``transformers_rgb`` / ``clip_resize_center_crop`` image
  hooks, ``parse_fakevlm`` and the ``VLMConfig.processor_hf_*`` download-hint fields (not in ``VLM_CONFIG_FIELDS``);
  4a950958ed moves the processor-folder choice into ``VLMFamily.processor_folder`` (same folder) and adds the
  ``num_additional_image_tokens`` load check to ``LlavaFamily``. Every other identity field is equal, so every record is
  kept, byte for byte (rule ``unchanged``). Such an entry names the code versions it may migrate TO
  (``to_code_versions``): after any further edit of ``models/vlm/*.py`` the folder is left alone until someone checks
  that edit and adds its code version.
* adb_vlm, veritaspp and fakevlm, detector/adb-full (2026-11-21): from code version 4a950958ed (main ca433a7) to
  53f470222c. The only change is a docstring line of ``models/vlm/parsers.py`` (Align, Don't Blend's licence is now
  Apache-2.0), so every record is kept (rule ``unchanged``); 53f470222c is also added to the e5749ae entries' targets,
  so a folder not yet migrated goes straight to it.
* fakevlm, the same branch: from the identity its benchmark jobs wrote (code version 02852acf62; folders
  ``fakevlm__greedy__ee254109e9`` and ``fakevlm__greedy__jpeg95__ee254109e9``) to 4a950958ed. The only change for FakeVLM
  is the load check, which its config passes (all three fields are in ``processor_overrides``); rule ``unchanged``.

* veritaspp and fakevlm, fix/adb-upstream-sync (2026-11-25): from code version 53f470222c (main 1b607c1) to bbad7ab02d.
  ``models/vlm/image.py``: ``swift_qwen_vl_resize`` caps ``fetch_image`` at the token-rounded ``max_pixels // 784 * 784``
  (``swift_fetch_max_pixels``), as swift does; ``models/vlm/parsers.py``: ``parse_adb_default_real`` added, docstrings.
  Neither detector uses either function (Veritas++: ``vllm_rgb``; FakeVLM: ``transformers_rgb`` +
  ``clip_resize_center_crop``) and every other identity field is equal, so every record is kept (rule ``unchanged``).
  adb_vlm has NO entry for this re-key: its prompt, decoding, pixel cap and parser changed with the upstream commits
  2f0adbc / 898b607, so its earlier records are not reused (they stay in their old folders; ``migrate_folder`` refuses any
  folder whose prompt or decoding differs from the current identity). bbad7ab02d is also added to the earlier entries'
  targets, so a veritaspp / fakevlm folder not yet migrated goes straight to it.

    python repro/vlm_bench/migrate_vlm_cache.py --cache-root /ssdstore/ssalehi/SynthScope/vpp/cache --model adb_vlm [--dry-run]
    python repro/vlm_bench/migrate_vlm_cache.py --cache-root /ssdstore/ssalehi/SynthScope/vpp/cache --model veritaspp
    python repro/vlm_bench/migrate_vlm_cache.py --cache-root /ssdstore/ssalehi/SynthScope/vpp/cache --model fakevlm

The new folder gets ``migrated.<old hash>.jsonl`` (the kept records, each the source folder's line as written),
``identity.json`` and ``MIGRATION.json`` (source folder, old identity, rule, counts, dropped keys). A folder already
migrated from the same source is left alone, and so is a source folder that an earlier migration carried into another
folder of the same stem (that folder holds its records and is migrated in its place). Run it with the code the scoring
jobs will run (a clean checkout of the same commit): the current identity includes the code version.
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

# main e5749ae -> detector/fakevlm: only the code version differs (see the module doc); identities as bench_eval.vlm_identity
# printed them at e5749ae (adb_vlm's official decoding is greedy too, so it shares the greedy identity)
_MAIN_E5749AE = "b490c64477"
# 53f470222c: detector/adb-full's docstring-only edit; bbad7ab02d: fix/adb-upstream-sync (veritaspp / fakevlm unchanged; an
# adb_vlm folder never matches it: its prompt and decoding changed, which migrate_folder refuses before any rule)
_FAKEVLM_BRANCH = ["02852acf62", "4a950958ed", "53f470222c", "bbad7ab02d"]
_ADB_FULL_BRANCH = ["53f470222c", "bbad7ab02d"]
_ADB_E5749AE = {
    "weights_revision": "49975ccd8b28eaab8ef14650ffee0c545934d02b", "base_weights_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    "ckpt": None, "max_new_tokens": 512, "generation": {"do_sample": False, "max_new_tokens": 512, "repetition_penalty": 1.05},
    "prompt_sha1": "91db63b292", "config_sha1": "21143b5948", "code_version": _MAIN_E5749AE,
}
_VPP_E5749AE = {
    "weights_revision": "22ba6c6b9f623755c1a68a9816b83a0540039829", "base_weights_revision": None, "ckpt": None,
    "max_new_tokens": 2048, "generation": {"do_sample": False, "max_new_tokens": 2048, "repetition_penalty": 1.0},
    "prompt_sha1": "e2e3580107", "config_sha1": "c71ec77e7a", "code_version": _MAIN_E5749AE,
}
_VPP_E5749AE_OFFICIAL = dict(_VPP_E5749AE, generation={"do_sample": True, "max_new_tokens": 2048, "repetition_penalty": 1.0,
                                                       "temperature": 0.7, "top_k": 20, "top_p": 0.8})
# fakevlm as its benchmark jobs (293884 / 293885, code 37eae6c) wrote it; the official decoding is greedy too (one identity)
_FAKEVLM_02852ACF62 = {
    "weights_revision": "dce9e88df7d1053e72af197c1bd3c61c94089e96", "base_weights_revision": None, "ckpt": None,
    "max_new_tokens": 256, "generation": {"do_sample": False, "max_new_tokens": 256}, "prompt_sha1": "98da686e12",
    "config_sha1": "02eebd2c47", "code_version": "02852acf62",
}

# main ca433a7 (code version 4a950958ed): the identities its runs wrote, the same as e5749ae's apart from the code version
_CA433A7 = "4a950958ed"
# main 1b607c1 (code version 53f470222c) -> fix/adb-upstream-sync (bbad7ab02d): veritaspp and fakevlm only (see the module doc)
_MAIN_1B607C1 = "53f470222c"
_ADB_SYNC_BRANCH = ["bbad7ab02d"]
# the code versions under which adb_vlm still sent the f4b4794 request (prompt 91db63b292, repetition penalty 1.05, 1,003,520 px)
_ADB_F4B4794_REQUEST = [_MAIN_E5749AE, "02852acf62", _CA433A7, _MAIN_1B607C1]


def _at(identity: dict, code_version: str) -> dict:
    return dict(identity, code_version=code_version)


# {model: [{"identity": an older cache identity, "rule": how its records carry over, "to_code_versions": (rule "unchanged")
# the code versions it may be migrated to; also checked for the other rules when given}]}. A folder whose prompt or decoding
# differs from the current identity's is never migrated.
PREDECESSORS = {
    "adb_vlm": [
        {"identity": {
            "weights_revision": "49975ccd8b28eaab8ef14650ffee0c545934d02b", "ckpt": None, "max_new_tokens": 512,
            "generation": {"do_sample": False, "max_new_tokens": 512, "repetition_penalty": 1.05},
            "prompt_sha1": "91db63b292", "config_sha1": "65977eff07", "code_version": "cf179eee30",
        }, "rule": "adb_verdict_offset", "to_code_versions": _ADB_F4B4794_REQUEST},
        {"identity": _ADB_E5749AE, "rule": "unchanged", "to_code_versions": _FAKEVLM_BRANCH},
        {"identity": _at(_ADB_E5749AE, _CA433A7), "rule": "unchanged", "to_code_versions": _ADB_FULL_BRANCH},
    ],
    "veritaspp": [
        {"identity": _VPP_E5749AE, "rule": "unchanged", "to_code_versions": _FAKEVLM_BRANCH},
        {"identity": _VPP_E5749AE_OFFICIAL, "rule": "unchanged", "to_code_versions": _FAKEVLM_BRANCH},
        {"identity": _at(_VPP_E5749AE, _CA433A7), "rule": "unchanged", "to_code_versions": _ADB_FULL_BRANCH},
        {"identity": _at(_VPP_E5749AE_OFFICIAL, _CA433A7), "rule": "unchanged", "to_code_versions": _ADB_FULL_BRANCH},
        {"identity": _at(_VPP_E5749AE, _MAIN_1B607C1), "rule": "unchanged", "to_code_versions": _ADB_SYNC_BRANCH},
        {"identity": _at(_VPP_E5749AE_OFFICIAL, _MAIN_1B607C1), "rule": "unchanged", "to_code_versions": _ADB_SYNC_BRANCH},
    ],
    "fakevlm": [
        {"identity": _FAKEVLM_02852ACF62, "rule": "unchanged", "to_code_versions": _FAKEVLM_BRANCH},
        {"identity": _at(_FAKEVLM_02852ACF62, _CA433A7), "rule": "unchanged", "to_code_versions": _ADB_FULL_BRANCH},
        {"identity": _at(_FAKEVLM_02852ACF62, _MAIN_1B607C1), "rule": "unchanged", "to_code_versions": _ADB_SYNC_BRANCH},
    ],
}


def identity_hash(identity: dict) -> str:
    return hashlib.sha1(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:10]


def read_records(folder: Path, lines: dict | None = None) -> dict:
    """{key: record} as ``bench_eval.VLMCache`` reads a folder (every .jsonl, a later line of a key wins); ``lines``, when
    given, is filled with {key: that record's line as written}."""
    records = {}
    for f in sorted(folder.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            records[rec["key"]] = rec
            if lines is not None:
                lines[rec["key"]] = line
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
    entry = next((e for e in PREDECESSORS.get(model, []) if e["identity"] == old_id), None)
    if entry is None:
        return {"folder": str(old), "status": "unknown identity, left alone"}
    if (old_id["prompt_sha1"], old_id["generation"]) != (new_id["prompt_sha1"], new_id["generation"]):
        return {"folder": str(old), "status": "the prompt or the decoding changed: no re-key, left alone"}
    if "to_code_versions" in entry and new_id["code_version"] not in entry["to_code_versions"]:
        return {"folder": str(old), "status": f"no checked re-key to the current identity (code version "
                                              f"{new_id['code_version']}), left alone"}
    if entry["rule"] == "unchanged":
        same = {k: v for k, v in old_id.items() if k != "code_version"} == {k: v for k, v in new_id.items() if k != "code_version"}
        if not same or new_id["code_version"] not in entry["to_code_versions"]:
            return {"folder": str(old), "status": f"no checked re-key to the current identity (code version "
                                                  f"{new_id['code_version']}), left alone"}
    new = old.parent / f"{stem}__{identity_hash(new_id)}"
    carried = [m.parent for m in old.parent.glob(f"{stem}__*/MIGRATION.json")
               if m.parent != new and Path(json.loads(m.read_text())["from"]).name == old.name]   # copied caches too
    if carried:                                 # e.g. cf179eee30 -> e5749ae earlier: that folder is the one to carry on
        return {"folder": str(old), "status": f"carried earlier into {carried[0]}, which is migrated instead"}
    lines = {}
    records = read_records(old, lines)
    if entry["rule"] == "unchanged":            # the scoring path is unchanged: every record as it is
        dropped = []
        kept = list(records.values())
    else:                                       # adb_verdict_offset
        dropped = sorted(k for k, r in records.items() if unscored(r))
        kept = [r for r in records.values() if not unscored(r)]
        hook = spec.vlm.verdict_offset
        changed = [r["key"] for r in kept if hook is not None and hook(r["response"]) is not None]
        if changed:
            raise SystemExit(f"{old}: {len(changed)} kept record(s) would be scored at the parser's verdict word, "
                             f"e.g. {changed[:3]}")
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
            f.write(lines[r["key"]] + "\n")
    (new / "identity.json").write_text(json.dumps(new_id, indent=1) + "\n")
    marker.write_text(json.dumps({
        "from": str(old), "from_identity": old_id, "to_identity": new_id, "rule": entry["rule"], "records": len(records),
        "kept": len(kept),
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
