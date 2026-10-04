"""Insert the full per-benchmark results (paper vs ours + status) into each model README,
and build the aggregated all-models table for the main README. Source: interface/backend/repro_full.json."""
import json, re, pathlib
ROOT = pathlib.Path("/ivi/zfs/s0/original_homes/ssalehi/SynthScope")
full = json.loads((ROOT / "interface/backend/repro_full.json").read_text())

# model key -> (README path, display name)
PATHS = {
    "clip": ("models/image/clip_detection", "CLIP / UnivFD"), "cnn": ("models/image/cnn_detection", "CNNDetection"),
    "fat_former": ("models/image/fat_former", "FatFormer"), "npr": ("models/image/npr", "NPR"),
    "aide": ("models/image/aide", "AIDE"), "safe": ("models/image/safe", "SAFE"),
    "co_spy": ("models/image/co_spy", "Co-Spy"), "spai": ("models/image/spai", "SPAI"),
    "vib_net": ("models/image/vib_net", "VIB-Net"), "pgc": ("models/image/pgc", "PGC"),
    "d3": ("models/video/d3", "D3"), "restrav": ("models/video/restrav", "ReStraV"),
    "stall": ("models/video/stall", "STALL"),
}
HDR = "## Reproduced benchmark results"
LEGEND = ("**reproduced** = SynthScope-trained checkpoint matches the paper · **official** = official weights "
          "loaded into the SynthScope class match (the SynthScope-trained number is shown alongside where we trained "
          "from scratch) · **below** = gap remains · **reference** = paper reports no comparable number · "
          "**missing** = not yet evaluated.")

def section(model):
    rows = full.get(model, [])
    out = [HDR, "",
           "All benchmarks the paper reports, with our numbers (verified, traced to "
           "`repro/bench_results/` + `repro/benchmark_specs/`).", "",
           "| Benchmark | Paper | SynthScope (trained) | Official wts | Status |", "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['benchmark']} | {r['paper']} | {r.get('ours_synthscope') or '—'} | "
                   f"{r.get('ours_official') or '—'} | {r['status']} |")
    out += ["", LEGEND]
    return "\n".join(out) + "\n"

def upsert(readme: pathlib.Path, sec: str):
    txt = readme.read_text() if readme.exists() else f"# {readme.parent.name}\n"
    # replace an existing "## Reproduced benchmark..." section (to next ## or EOF), else append
    pat = re.compile(r"\n## Reproduced benchmark results.*?(?=\n## |\Z)", re.S)
    new = "\n" + sec
    txt = pat.sub(new, txt) if pat.search(txt) else (txt.rstrip() + "\n\n" + sec)
    readme.write_text(txt)

for model, (d, _name) in PATHS.items():
    rm = ROOT / d / "README.md"
    upsert(rm, section(model))
    print(f"updated {rm.relative_to(ROOT)} ({len(full.get(model,[]))} benchmarks)")

# Aggregated table for the main README
agg = ["## Full-benchmark reproduction (all models × all reported benchmarks)", "",
       "Every benchmark each paper reports, with our verified numbers. Per-model detail in each "
       "`models/*/<name>/README.md`; raw results in `repro/bench_results/`, specs in "
       "`repro/benchmark_specs/`, matrix in `repro/BENCHMARK_COVERAGE_MATRIX.md`.", "",
       LEGEND, "",
       "| Model | Benchmark | Paper | SynthScope (trained) | Official wts | Status |", "|---|---|---|---|---|---|"]
for model, (_d, name) in PATHS.items():
    for r in full.get(model, []):
        agg.append(f"| {name} | {r['benchmark']} | {r['paper']} | {r.get('ours_synthscope') or '—'} | "
                   f"{r.get('ours_official') or '—'} | {r['status']} |")
agg_md = "\n".join(agg) + "\n"
(ROOT / "repro/AGGREGATED_BENCHMARK.md").write_text(agg_md)
print("wrote repro/AGGREGATED_BENCHMARK.md")

# Keep the same aggregated section in the top-level README in sync.
readme = ROOT / "README.md"
rtxt = readme.read_text()
rpat = re.compile(r"## Full-benchmark reproduction \(all models × all reported benchmarks\).*?(?=\n## )", re.S)
if rpat.search(rtxt):
    readme.write_text(rpat.sub(agg_md.rstrip() + "\n", rtxt))
    print("updated README.md aggregated section")
else:
    print("WARN: aggregated section header not found in README.md — left unchanged")
