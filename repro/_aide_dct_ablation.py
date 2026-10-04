"""Isolate the AIDE SynthScope_train gap: run the GENUINE official AIDE model + weights, but ablate the
DCT-input resolution. native-res (official) vs resize-256-before-DCT (what the SynthScope class does).
If resize-256 tanks the official weights, the gap is the preprocessing (native-res DCT), not training."""
import argparse, sys
from pathlib import Path
import torch
from PIL import Image, ImageFile
from torchvision import transforms
sys.path.insert(0, "/ivi/zfs/s0/original_homes/ssalehi/AGID")
from repro.official_evals.aide_official_eval import (
    DCT_base_Rec_Module, build_model, discover, metrics, transform_train)
ImageFile.LOAD_TRUNCATED_IMAGES = True

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--benchmark", default="pergen")
ap.add_argument("--limit", type=int, default=200)
ap.add_argument("--mode", choices=["native", "resize256"], required=True)
ap.add_argument("--select", default=None)
args = ap.parse_args()

device = torch.device("cuda")
model = build_model(Path(args.ckpt), device)
dct = DCT_base_Rec_Module()
# native: ToTensor only (official). resize256: Resize([256,256]) BEFORE DCT (mimics SynthScope aide.py:221).
if args.mode == "native":
    before = transforms.Compose([transforms.ToTensor()])
else:
    before = transforms.Compose([transforms.Resize([256, 256]), transforms.ToTensor()])

@torch.no_grad()
def fake_prob(path):
    im = before(Image.open(path).convert("RGB"))
    a, b, c, d = dct(im)
    x0 = transform_train(im)
    stack = torch.stack([transform_train(a), transform_train(b), transform_train(c),
                         transform_train(d), x0], dim=0).unsqueeze(0).to(device)
    return model(stack).softmax(dim=1)[0, 1].item()

sel = set(args.select.split(",")) if args.select else None
gens = discover(args.benchmark, Path(args.data), None, sel)
ap_all, au_all, acc_all = [], [], []
for name, reals, fakes in gens:
    reals, fakes = reals[:args.limit], fakes[:args.limit]
    probs = []
    for p in reals + fakes:
        try: probs.append(fake_prob(p))
        except Exception: probs.append(0.0)
    labels = torch.tensor([0]*len(reals) + [1]*len(fakes))
    m = metrics(torch.tensor(probs), labels)
    if not m: continue
    ap_all.append(m["ap"]); au_all.append(m["auroc"]); acc_all.append(m["acc"])
    print(f"  {name}: AP={m['ap']} AUROC={m['auroc']} acc={m['acc']}", flush=True)
mean = lambda xs: round(sum(xs)/len(xs), 2)
print(f"=== AIDE DCT-ABLATION mode={args.mode}: mAP={mean(ap_all)} mAUROC={mean(au_all)} "
      f"acc={mean(acc_all)} over {len(ap_all)} gens ===", flush=True)
