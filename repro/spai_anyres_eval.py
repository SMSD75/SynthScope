"""Eval SPAI through the AGID class at NATIVE/arbitrary resolution (the official variant), using the
remapped official weights. Per-image (bs=1) native-res forward -> the class's any-res path (variable L
patches + five_crop fallback). Reports per-generator AUROC (SPAI's headline metric) + AP. Reuses
bench_eval's discover()/metrics() so numbers are comparable."""
import os, sys, json, argparse, datetime
os.environ.setdefault("AGID_MFM_CKPT", "/hddstore/ssalehi/AGID/mfm/mfm_pretrain_vit_base.pth")
sys.path.insert(0, "/ivi/zfs/s0/original_homes/ssalehi/AGID")
import torch
from pathlib import Path
from PIL import Image, ImageFile
import torchvision.transforms as T
from omegaconf import OmegaConf
from benchmark.bench_eval import discover, metrics
from models.image.spai.spai import SPAI
ImageFile.LOAD_TRUNCATED_IMAGES = True

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--ckpt", default="/hddstore/ssalehi/AGID/repro/ckpts/spai_official_agid.pth")
ap.add_argument("--benchmark", default="pergen")
ap.add_argument("--limit", type=int, default=1000)
ap.add_argument("--max_side", type=int, default=1024, help="cap long side to bound memory (official caps too)")
ap.add_argument("--tag", required=True)
args = ap.parse_args()

cfg = OmegaConf.create({"ModelConfig": {"freeze_backbone": True, "projection_dim": 1024,
    "input_size": 224, "mask_radius": 16, "use_sca": True, "patch_size": 224, "patch_stride": 224,
    "num_heads": 12, "attn_embed_dim": 1536, "minimum_patches": 4}})
model = SPAI(cfg).cuda().eval()
sd = torch.load(args.ckpt, map_location="cpu")
msg = model.load_state_dict(sd, strict=False)
print(f"loaded {args.ckpt}: missing(non-bb)={[k for k in msg.missing_keys if not k.startswith('backbone.')]} unexpected={len(msg.unexpected_keys)}", flush=True)
to_tensor = T.ToTensor()

@torch.no_grad()
def fake_prob(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if max(w, h) > args.max_side:  # downscale only very large images (keep aspect)
        s = args.max_side / max(w, h)
        im = im.resize((max(int(w*s), 224), max(int(h*s), 224)))
    x = to_tensor(im).unsqueeze(0).cuda()           # 1 x 3 x H x W, raw [0,1] native
    return torch.sigmoid(model(x)).item()

gens = discover(args.benchmark, Path(args.data), None, None)
per_gen, agg = {}, {"ap": [], "auroc": [], "acc": []}
for name, reals, fakes in gens:
    reals, fakes = reals[:args.limit], fakes[:args.limit]
    probs = []
    for p in reals + fakes:
        try: probs.append(fake_prob(p))
        except Exception as e: probs.append(0.0)
    labels = torch.tensor([0]*len(reals) + [1]*len(fakes))
    m = metrics(torch.tensor(probs), labels)
    if m is None: continue
    per_gen[name] = m
    for k in agg: agg[k].append(m[k])
    print(f"  {name}: AUROC={m['auroc']} AP={m['ap']} acc={m['acc']}", flush=True)

mean = lambda xs: round(sum(xs)/len(xs), 2)
means = {"acc": mean(agg["acc"]), "map": mean(agg["ap"]), "mauroc": mean(agg["auroc"])}
res = {"tag": args.tag, "model": "spai-anyres", "ckpt": args.ckpt, "benchmark": args.benchmark,
       "data_root": args.data, "n_generators": len(per_gen), "means": means, "per_generator": per_gen,
       "date": datetime.datetime.now().isoformat(timespec="seconds"), "agid_version": "spai-anyres-official"}
out = Path("/ivi/zfs/s0/original_homes/ssalehi/AGID/repro/bench_results") / f"{args.tag}.json"
out.write_text(json.dumps(res, indent=2) + "\n")
print(f"=== RESULT {args.tag}: mAUROC={means['mauroc']} mAP={means['map']} acc={means['acc']} over {len(per_gen)} gens ===", flush=True)
