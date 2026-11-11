"""Debug check (Task 2): why did the stage-3 export check fail for multi_stage23.py's cached head features (r1e0)?
Compares, on the stage-3 export-check images (first 32 reals + first 32 generated of GenImage SD1.4 train), the 1024-d
features from: the cache (head_features.pt), a single encoder on the 64-image batch, a single encoder on LabelledImages
batches of 128, the multi path (all encoders loaded, same order), and the model reloaded from cure_trained.pth.

    python repro/cure_stage1/check_head_features.py --data <GenImage> --out <s23_r1e0 dir> --lora r1e0=<p> --lora r2e1=<p> ...
"""
import argparse

import torch
from torch.utils.data import DataLoader, Subset

from dataset import CuReEndpoints
from models.image.cure.train_stages import LabelledImages, build_model, endpoint_transform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None, help="the s23_<name> dir of the FIRST --lora (omit: no cache / export comparison)")
    ap.add_argument("--lora", action="append", required=True, help="name=path, in the multi run's order; first = under test")
    args = ap.parse_args()
    loras = [l.split("=", 1) for l in args.lora]
    dev = "cuda"
    ep = CuReEndpoints(args.data, "train", endpoint_transform(), None)
    n_r = len(ep.reals)
    idx = list(range(32)) + [n_r + i for i in range(32)]
    if args.out:
        zc = torch.load(f"{args.out}/head_features.pt", map_location="cpu", weights_only=True)["z"][idx]
    else:
        zc = None
    probe = torch.stack([ep.load(ep.reals[i]) for i in range(32)] + [ep.load(ep.fakes[i]) for i in range(32)]).to(dev)

    def d(a, b):
        if a is None or b is None:
            return "n/a"
        return f"max|diff| {float((a.float().cpu() - b.float().cpu()).abs().max()):.3e}"
    single = build_model(loras[0][1], dev)
    with torch.no_grad():
        zs = single.features(probe).float().cpu()
    print("single encoder, 64-image batch vs cache:", d(zs, zc), flush=True)
    # LabelledImages batches of 128 (the cache's batching): items 0..127 are reals; fakes start at n_r
    li = LabelledImages(ep)
    with torch.no_grad():
        xb = torch.stack([li[i][0] for i in range(128)]).to(dev)
        xf = torch.stack([li[n_r + i][0] for i in range(128)]).to(dev)
        zb = torch.cat([single.features(xb)[:32], single.features(xf)[:32]]).float().cpu()
    print("single encoder, 128 batches vs cache:", d(zb, zc), "| vs 64 batch:", d(zb, zs), flush=True)
    print("probe tensors == LabelledImages tensors:", d(torch.cat([xb[:32], xf[:32]]), probe), flush=True)
    models = {k: build_model(p, dev) for k, p in loras}
    with torch.no_grad():
        zm = {}
        for k, m in models.items():
            zm[k] = torch.cat([m.features(xb)[:32], m.features(xf)[:32]]).float().cpu()
    first = loras[0][0]
    print(f"multi path ({first}) vs cache:", d(zm[first], zc), "| vs single:", d(zm[first], zb), flush=True)
    for k in zm:
        print(f"  cache vs multi-path encoder {k}:", d(zm[k], zc), flush=True)
    check = build_model(None, dev)
    if args.out:
        st = torch.load(f"{args.out}/cure_trained.pth", map_location="cpu", weights_only=True)
    else:
        st = {k: v.cpu() for k, v in single.adapter_state_dict().items()}
    print("reload:", check.load_state_dict(st, strict=False).unexpected_keys, flush=True)
    with torch.no_grad():
        zr = check.features(probe).float().cpu()
    print("reloaded model vs cache:", d(zr, zc), "| vs single:", d(zr, zs), flush=True)
    lt = {k: v for k, v in st.items() if "lora_" in k}
    ls = {k: v for k, v in single.state_dict().items() if "lora_" in k}
    print("LoRA tensors in cure_trained.pth vs --lora file:", max(float((lt[k].float() - ls[k].float().cpu()).abs().max()) for k in lt), flush=True)
    if args.out:
        W, mu, hw, hb = st["W"].float(), st["mu"].float(), st["head.weight"].float(), st["head.bias"].float()
        logit = lambda z: ((z - mu) @ W) @ hw.T + hb
        print("logits: cache vs reloaded", d(logit(zc), logit(zr)), "| |logit| max", float(logit(zc).abs().max()),
              "| head |w| max", float(hw.abs().max()), flush=True)
    print("feature scale: |z| max", float(zs.abs().max()), "mean |z|", float(zs.abs().mean()), flush=True)


if __name__ == "__main__":
    main()
