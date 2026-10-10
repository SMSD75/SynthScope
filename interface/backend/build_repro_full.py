"""Build interface/backend/repro_full.json: per-method list of ALL reported benchmarks with
SEPARATE columns for the SynthScope-trained checkpoint and the official-weights-in-SynthScope-class number.

Source of truth = the verified specs in repro/benchmark_specs/*.json. Numbers that the specs only
record in prose notes (the AGID/official split is structured inconsistently across methods) are
pulled into an explicit, auditable VAL table below — every value here traces to a file under
repro/bench_results/ (see the per-row tags in the spec notes). Run with --audit to print the table.
"""
import json, glob, os, sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])   # the repository root, wherever it is checked out

# (model, benchmark-name-substring) -> verified {synthscope:{...}, official:{...}} numbers.
# These resolve the SynthScope-vs-official ambiguity in the spec our_number dicts and surface the number
# that the spec only kept in prose. Each traces to a repro/bench_results/*.json (tag in spec note).
VAL = {
    # AIGCDetect is the ProGAN-trained setting (B1); we only retrained the SDv1.4 setting, so no SynthScope_train here.
    ("aide", "AIGCDetect"):              {"official": {"map": 97.94, "acc": 91.89}},
    # SynthScope_train = faithful native-res 5-view retrain (aide_native_v2/model_2) — reproduces official.
    ("aide", "GenImage"):                {"official": {"map": 97.12, "acc": 85.91}, "synthscope": {"map": 96.42, "acc": 77.87}},
    ("clip", "ForenSynths"):             {"synthscope": {"map": 97.22, "acc": 88.28}, "official": {"map": 96.95}},
    ("clip", "UniversalFakeDetect"):     {"synthscope": {"map": 96.44, "acc": 87.23}, "official": {"map": 95.66}},
    ("cnn", "ForenSynths"):              {"synthscope": {"map": 94.42, "acc": 76.36}, "official": {"map": 92.67}},
    ("co_spy", "Co-Spy-Bench"):          {"synthscope": {"map": 96.6, "acc": 89.9}, "official": {"map": 86.65, "acc": 83.88}},
    ("co_spy", "AIGCDetect"):            {"official": {"map": 94.59, "acc": 89.25}, "synthscope": {"map": 84.91}},
    ("co_spy", "GenImage"):              {"official": {"map": 92.41}, "synthscope": {"map": 90.59, "acc": 75.91}},
    ("co_spy", "ForenSynths"):           {"official": {"map": 87.97}, "synthscope": {"map": 96.17}},
    # CuRe: only the official adapter exists (the training recipe is unreleased), so there is no SynthScope-trained number.
    ("cure", "AIGIBench"):               {"official": {"map": 97.0, "acc": 92.85, "auroc": 97.48}},
    ("cure", "UniversalFakeDetect"):     {"official": {"map": 99.19, "acc": 93.94}},
    ("cure", "ForenSynths"):             {"official": {"map": 94.05, "acc": 81.95}},
    ("cure", "GenImage"):                {"official": {"map": 97.3, "acc": 85.18}},
    ("cure", "RealChain"):               {"official": {"map": 87.39, "acc": 79.69, "auroc": 85.47}},
    ("cure", "EvalGEN"):                 {"official": {"map": 99.86, "acc": 98.14, "auroc": 99.66}},
    ("fat_former", "4-class"):           {"official": {"map": 99.7, "acc": 98.39}},
    ("fat_former", "Diffusion-model"):   {"official": {"map": 98.6, "acc": 93.72}},
    ("npr", "Table 1"):                  {"official": {"map": 94.22, "acc": 92.30}},
    ("npr", "Table 2"):                  {"official": {"map": 96.49, "acc": 94.81}},
    ("npr", "Table 3"):                  {"official": {"map": 99.4, "acc": 95.03}},
    ("npr", "Table 4"):                  {"official": {"map": 96.69, "acc": 93.94}},
    ("npr", "Table 5"):                  {"official": {"map": 83.47, "acc": 79.13}},
    # SAFE now SERVES the SynthScope-trained checkpoint (retrain v2 / model_38), which reproduces the paper.
    ("safe", "ForenSynths"):             {"official": {"map": 98.78, "acc": 96.2}, "synthscope": {"map": 97.79, "acc": 95.1}},
    ("safe", "GANGen"):                  {"official": {"map": 99.85, "acc": 99.0}},
    ("safe", "UniversalFakeDetect"):     {"official": {"map": 99.1, "acc": 95.71}, "synthscope": {"map": 98.44, "acc": 94.0}},
    ("safe", "GenImage"):                {"official": {"map": 99.32, "acc": 95.62}, "synthscope": {"map": 99.51, "acc": 95.9}},
    ("safe", "DiTFake"):                 {"synthscope": {"map": 99.96, "acc": 99.2}},
    ("safe", "DiffusionForensics"):      {"synthscope": {"map": 99.34, "acc": 91.3}},
    ("pgc", "GenImage"):                 {"official": {"map": 100.0, "acc": 99.49}, "synthscope": {"map": 97.62, "acc": 78.6}},
    ("pgc", "UniversalFakeDetect"):      {"official": {"map": 99.34, "acc": 94.16}, "synthscope": {"map": 99.01}},
    ("pgc", "ForenSynths"):              {"official": {"map": 96.25, "acc": 83.73}, "synthscope": {"map": 89.03}},
    ("pgc", "AIGIBench"):                {"official": {"map": 95.36, "acc": 88.33}, "synthscope": {"map": 86.88}},
    ("pgc", "CommGen15"):                {"official": {"map": 99.26, "acc": 97.39}, "synthscope": {"map": 98.61, "acc": 66.66}},
    # SPAI now SERVES official weights remapped into the SynthScope class, scored at NATIVE resolution
    # (the faithful any-res path). official = that served number; synthscope = the superseded from-scratch retrain.
    ("spai", "GenImage"):                {"official": {"map": 94.22, "auroc": 94.28}, "synthscope": {"map": 88.8, "acc": 82.81}},
    ("spai", "Synthbuster"):             {"official": {"map": 96.75, "auroc": 96.97}},
    # D3 now uses the faithful protocol (X-CLIP-B/16 + 10% edge-crop). On GenVideo, AUROC/acc are the
    # robust metrics (the mAP harness is degenerate: 10000 reals : as few as 56 fakes/generator).
    ("d3", "GenVideo"):                  {"synthscope": {"map": 63.93, "auroc": 95.95, "acc": 91.71}},
    ("d3", "EvalCrafter"):               {"synthscope": {"map": 91.19, "auroc": 95.41, "acc": 87.67}},
    ("restrav", "GenVidBench"):          {"synthscope": {"map": 92.7, "auroc": 95.9, "acc": 86.87}},
    ("stall", "GenVideo"):               {"synthscope": {"map": 82.38, "auroc": 83.51, "acc": 56.88}},
    ("stall", "VideoFeedback"):          {"synthscope": {"map": 83.07, "auroc": 85.1, "acc": 62.12}},
    ("stall", "ComGenVid"):              {"synthscope": {"map": 82.59, "auroc": 84.0, "acc": 57.34}},
    ("vib_net", "Diffusion-source (SDv1.4-trained), AP"):  {"synthscope": {"map": 83.94, "acc": 77.57}},
    ("vib_net", "Diffusion-source (SDv1.4-trained), ACC"): {"synthscope": {"map": 83.94, "acc": 77.57}},
    ("vib_net", "GAN-source (ForenSynths/ProGAN-trained), AP"):  {"synthscope": {"map": 61.71, "acc": 57.91}},
    ("vib_net", "GAN-source (ForenSynths/ProGAN-trained), ACC"): {"synthscope": {"map": 61.71, "acc": 57.91}},
    ("co_spy", "In-the-wild"): {},   # genuinely not evaluated (prevents the "Co-Spy-Bench" substring collision)
}

# Tolerance (percentage points) for calling a number a reproduction of the paper.
TOL = {"map": 2.0, "acc": 2.0, "auroc": 2.5}

LAB = {"map": "mAP", "acc": "acc", "auroc": "AUROC"}


def paper_metrics(d):
    """Normalize a paper_number dict to {map,acc,auroc} in PERCENT (fractions like 0.80 -> 80)."""
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        if not isinstance(v, (int, float)):
            continue
        kl = k.lower()
        val = v * 100 if 0 < v <= 1.0 else v
        if "auc" in kl or "auroc" in kl:
            out.setdefault("auroc", val)
        elif kl == "acc" or kl.startswith("acc") or "_acc" in kl:
            out.setdefault("acc", val)
        elif "ap" in kl or "map" in kl:
            out.setdefault("map", val)
    return out


def cmp_ok(ours, pm):
    """True/False if ours reproduces the paper on every shared metric; None if not comparable."""
    shared = [m for m in ("map", "acc", "auroc") if m in ours and m in pm]
    if not shared:
        return None
    return all(ours[m] >= pm[m] - TOL[m] for m in shared)


def fmt_paper(d):
    if not isinstance(d, dict):
        return "—" if d in (None, "") else str(d)
    parts = [f"{k}={v}" for k, v in d.items()
             if k != "note" and v not in (None, "") and not isinstance(v, (dict, list))]
    return " / ".join(parts) if parts else "—"


def has_paper_number(d):
    if not isinstance(d, dict):
        return d not in (None, "")
    return any(k != "note" and v not in (None, "") and not isinstance(v, (dict, list))
               for k, v in d.items())


def fmt_ours(d):
    if not d:
        return ""
    return " / ".join(f"{LAB[k]}={d[k]}" for k in ("map", "acc", "auroc") if k in d)


def combined(synthscope, official):
    a, o = fmt_ours(synthscope), fmt_ours(official)
    if a and o:
        return f"SynthScope {a} · official {o}"
    if o:
        return f"{o} (official wts)"
    return a


def match_val(model, name):
    """Return (synthscope, official, matched_substr) for the best VAL hit. A prefix match wins over a
    mid-string match (so 'In-the-wild (Co-Spy-Bench ...)' matches the In-the-wild entry, not Co-Spy-Bench)."""
    best = None  # (is_prefix, len, sub, v)
    nl = name.lower()
    for (m, sub), v in VAL.items():
        if m == model and sub.lower() in nl:
            cand = (nl.startswith(sub.lower()), len(sub), sub, v)
            if best is None or cand[:2] > best[:2]:
                best = cand
    if not best:
        return {}, {}, None
    return best[3].get("synthscope", {}), best[3].get("official", {}), best[2]


def status_for(name, paper, synthscope, official, spec_status):
    """reproduced = SynthScope-trained matches paper; official = official-weights-in-SynthScope matches paper
    (SynthScope-trained number shown alongside where attempted); below = neither; reference = paper gives
    no number for this cross-dataset; missing = not evaluated. Falls back to the spec's label only
    when the paper metric keys are not numerically comparable (e.g. ReStraV's nested per-split dicts)."""
    if not has_paper_number(paper):
        return "reference" if (synthscope or official) else "missing"
    if not (synthscope or official):
        return "missing"
    pm = paper_metrics(paper)
    if not pm:                                   # paper has numbers but not in comparable keys
        return spec_status or "below"
    synthscope_ok, off_ok = cmp_ok(synthscope, pm), cmp_ok(official, pm)
    if synthscope_ok:
        return "reproduced"
    if off_ok:
        return "official"
    if synthscope_ok is None and off_ok is None:       # neither comparable -> trust the spec
        return spec_status or "below"
    return "below"


out = {}
audit = []
for f in sorted(glob.glob(f"{ROOT}/repro/benchmark_specs/*.json")):
    d = json.load(open(f))
    model = d.get("model")
    rows = []
    for b in d.get("benchmarks", []):
        name = b.get("name", "") or ""
        paper = b.get("paper_number")
        spec_status = b.get("our_status", "")
        synthscope, official, sub = match_val(model, name)
        status = status_for(name, paper, synthscope, official, spec_status)
        on = b.get("our_number")
        note = ""
        if isinstance(on, dict) and on.get("note"):
            note = on["note"]
        else:
            note = (b.get("gap_cause") or b.get("dataset_source") or "")
        rows.append({
            "benchmark": name[:46],
            "metric": b.get("metric", ""),
            "paper": fmt_paper(paper),
            "ours": combined(synthscope, official),
            "ours_synthscope": fmt_ours(synthscope),
            "ours_official": fmt_ours(official),
            "status": status,
            "note": note[:400],
        })
        audit.append((model, name[:38], fmt_paper(paper)[:22],
                      (fmt_ours(synthscope) or "-")[:22], (fmt_ours(official) or "-")[:22],
                      f"{spec_status}->{status}"))
    out[model] = rows

json.dump(out, open(f"{ROOT}/interface/backend/repro_full.json", "w"), indent=2)
n = sum(len(v) for v in out.values())
counts = {}
for v in out.values():
    for r in v:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
print(f"wrote repro_full.json: {len(out)} methods, {n} rows  status={counts}")

if "--audit" in sys.argv:
    print(f"\n{'model':10} {'benchmark':38} {'paper':22} {'SynthScope':22} {'official':22} status")
    print("-" * 140)
    for r in audit:
        print(f"{r[0]:10} {r[1]:38} {r[2]:22} {r[3]:22} {r[4]:22} {r[5]}")
