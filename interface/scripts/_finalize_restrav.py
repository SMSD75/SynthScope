"""Read the ReStraV eval log, write the deployed cross-generator result, and refresh the README number."""
import re, sys
from pathlib import Path
log = Path(sys.argv[1]).read_text(errors="ignore")
lines = [l for l in log.splitlines() if l.startswith("restrav: acc@0.5=")]
if not lines:
    print("[finalize] no eval line found"); sys.exit(0)
last = lines[-1]
ap = re.search(r"AP=([\d.]+)", last).group(1)
auc = re.search(r"AUC=([\d.]+)", last).group(1)
res = Path("/hddstore/ssalehi/SynthScope/checkpoints/_train/restrav_exact_RESULT.txt")
res.write_text(f"deployed ReStraV (exact official recipe) cross-generator on full GenVideo-Val:\n  AP {ap} / AUROC {auc}\n  ({last})\n")
readme = Path("README.md")
t = readme.read_text()
t2 = re.sub(r"\*\*AP [\d.]+ / AUROC [\d.]+\*\* \(last run\)", f"**AP {ap} / AUROC {auc}** (last run)", t, count=1)
if t2 != t:
    readme.write_text(t2); print(f"[finalize] README updated -> AP {ap} / AUROC {auc}")
else:
    print("[finalize] README token not found (left unchanged)")
