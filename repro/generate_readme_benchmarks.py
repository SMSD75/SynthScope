"""Insert the full per-benchmark results (paper vs ours + status) into each model README,
and build the aggregated all-models table for the main README. Source: interface/backend/repro_full.json."""
import json, re, pathlib
ROOT = pathlib.Path("/ivi/zfs/s0/original_homes/ssalehi/AGID")
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

def section(model):
    rows = full.get(model, [])
    out = [HDR, "",
           "All benchmarks the paper reports, with our reproduced numbers (verified, traced to "
           "`repro/bench_results/` + `repro/benchmark_specs/`).", "",
           "| Benchmark | Metric | Paper | Ours | Status |", "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['benchmark']} | {r.get('metric','')} | {r['paper']} | {r['ours']} | {r['status']} |")
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
       "Every benchmark each paper reports, reproduced and verified. Per-model detail in each "
       "`models/*/<name>/README.md`; raw results in `repro/bench_results/`, specs in "
       "`repro/benchmark_specs/`, matrix in `repro/BENCHMARK_COVERAGE_MATRIX.md`.", "",
       "| Model | Benchmark | Paper | Ours | Status |", "|---|---|---|---|---|"]
for model, (_d, name) in PATHS.items():
    for r in full.get(model, []):
        agg.append(f"| {name} | {r['benchmark']} | {r['paper']} | {r['ours']} | {r['status']} |")
(ROOT / "repro/AGGREGATED_BENCHMARK.md").write_text("\n".join(agg) + "\n")
print("wrote repro/AGGREGATED_BENCHMARK.md")
