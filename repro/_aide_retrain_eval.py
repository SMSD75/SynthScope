"""Eval an AGID-TRAINED AIDE checkpoint (the Phase-2 from-scratch retrain) on GenImage/ForenSynths.

The retrained checkpoint is already an AGIDAIDEModel state_dict (keys 'net.*'), so it loads directly
(NOT through the registry's remap_aide, which adds a net. prefix for the official no-prefix ckpts).
Uses the uncapped native-res 5-view transform + bench_eval's discover()/metrics()."""
import argparse, datetime, json, sys
from pathlib import Path
import torch
from PIL import Image, ImageFile
sys.path.insert(0, "/ivi/zfs/s0/original_homes/ssalehi/AGID")
from config_utils import compose_config
from interface.backend.engine import _build_model
from models.image.aide.official_arch import AIDEFiveView
from benchmark.bench_eval import discover, metrics
ImageFile.LOAD_TRUNCATED_IMAGES = True

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--benchmark", default="pergen")
ap.add_argument("--reals", default=None)
ap.add_argument("--limit", type=int, default=300)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--tag", required=True)
args = ap.parse_args()

device = torch.device("cuda")
cfg = compose_config(["model@ModelConfig=image/aide", "ModelConfig.convnext_pretrained=null"])
model = _build_model(cfg)
sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
missing, unexpected = model.load_state_dict(sd, strict=False)
real_missing = [k for k in missing if not k.endswith(("clip_mean", "clip_std", "dinov2_mean", "dinov2_std"))]
print(f"loaded {args.ckpt}: {len(sd)} keys, missing(real)={len(real_missing)} unexpected={len(unexpected)}", flush=True)
model = model.to(device).float().eval()
tf = AIDEFiveView()  # uncapped native-res 5-view (eval fidelity)

@torch.no_grad()
def score(paths):
    probs = []
    for i in range(0, len(paths), args.bs):
        batch = []
        for p in paths[i:i + args.bs]:
            try: batch.append(tf(Image.open(p).convert("RGB")))
            except Exception as e: print(f"  bad {p}: {e}")
        if not batch: continue
        x = torch.stack(batch).to(device)
        probs.append(model(x).softmax(dim=1)[:, 1].float().cpu())
    return torch.cat(probs) if probs else torch.empty(0)

gens = discover(args.benchmark, Path(args.data), Path(args.reals) if args.reals else None, None)
per_gen, agg = {}, {"ap": [], "auroc": [], "acc": []}
for name, reals, fakes in gens:
    reals, fakes = reals[:args.limit], fakes[:args.limit]
    labels = torch.tensor([0] * len(reals) + [1] * len(fakes))
    probs = score(reals + fakes)
    if probs.numel() != labels.numel(): labels = labels[:probs.numel()]
    m = metrics(probs, labels)
    if m is None: continue
    per_gen[name] = m
    for k in agg: agg[k].append(m[k])
    print(f"  {name}: acc={m['acc']} AP={m['ap']} AUROC={m['auroc']}", flush=True)
mean = lambda xs: round(sum(xs) / len(xs), 2)
means = {"acc": mean(agg["acc"]), "map": mean(agg["ap"]), "mauroc": mean(agg["auroc"])}
res = {"tag": args.tag, "model": "aide-retrain", "ckpt": args.ckpt, "benchmark": args.benchmark,
       "data_root": args.data, "n_generators": len(per_gen), "means": means, "per_generator": per_gen,
       "date": datetime.datetime.now().isoformat(timespec="seconds")}
out = Path("/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/bench_results") / f"{args.tag}.json"
out.write_text(json.dumps(res, indent=2) + "\n")
print(f"=== RESULT {args.tag}: mAP={means['map']} mAUROC={means['mauroc']} acc={means['acc']} over {len(per_gen)} gens ===", flush=True)
