"""CuRe stage-1 investigation, Task 2: stages 2 and 3 for SEVERAL stage-1 encoders with one pass over the images.

The expensive part of ``models/image/cure/train_stages.py`` is decoding / preprocessing the images (20,000 x 3 endpoint
pairs for stage 2, all 324k training images for stage 3), which is identical for every encoder. This driver reads each
image once, runs every encoder on it, and writes for each encoder exactly the caches that ``train_stages.py`` itself would
write (``<out>/trajectories.pt`` with the same pairs, seed and fields; ``<out>/head_features.pt``). It then calls the
UNCHANGED ``train_stages.py stage2`` / ``stage3`` for each encoder, which reuse those caches, so the solve (Eq. 6-11), the
head training and the export / reload check are the recipe's own code with its defaults.

    python repro/cure_stage1/multi_stage23.py --data <GenImage SD1.4 root> --reference-adapter <cure_adapter.pt> \
        --run r2e4=<lora.pth>:<out dir> --run r2e1=<lora.pth>:<out dir> ...
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import CuReEndpoints, cure_mix
from models.image.cure import subspace as S
from models.image.cure.cure import FEATURE_DIM
from models.image.cure.train_stages import (TYPES, LabelledImages, TrajectoryPairs, build_model, draw_pairs,
                                            endpoint_transform, main as train_stages_main)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", action="append", required=True, help="name=lora_path:out_dir")
    ap.add_argument("--reference-adapter", default=None)
    ap.add_argument("--pairs", type=int, default=20000)      # train_stages.py stage2 default
    ap.add_argument("--seed", type=int, default=0)           # train_stages.py default
    ap.add_argument("--batch-pairs", type=int, default=16)   # train_stages.py stage2 default
    ap.add_argument("--encode-batch", type=int, default=128) # train_stages.py stage3 default
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-images", type=int, default=None, help="smoke tests only")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    runs = {}
    for r in args.run:
        name, rest = r.split("=", 1)
        lora, out = rest.rsplit(":", 1)
        runs[name] = (lora, Path(out))
        Path(out).mkdir(parents=True, exist_ok=True)
    dev = args.device
    models = {k: build_model(v[0], dev) for k, v in runs.items()}
    ep = CuReEndpoints(args.data, "train", endpoint_transform("none"), args.max_images)

    # ---- stage 2 trajectories (train_stages.stage2's pair draw, same order and seed)
    if not all((o / "trajectories.pt").exists() for _, o in runs.values()):
        gen = torch.Generator().manual_seed(args.seed)
        n_r, n_f = len(ep.reals), len(ep.fakes)
        pairs = {"RF": draw_pairs(args.pairs, n_r, n_f, False, gen), "RR": draw_pairs(args.pairs, n_r, n_r, True, gen),
                 "FF": draw_pairs(args.pairs, n_f, n_f, True, gen)}
        print(f"[multi] stage-2 trajectories: {n_r} / {n_f} images, {args.pairs} pairs per type, {len(runs)} encoders", flush=True)
        loader = DataLoader(TrajectoryPairs(ep, pairs), batch_size=args.batch_pairs, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
        grid = torch.tensor(S.GRID, device=dev).view(1, -1, 1, 1, 1)
        outs = {k: {f"{a}_{t}": torch.zeros(args.pairs, FEATURE_DIM) for t in TYPES for a in ("b", "zbar")} for k in runs}
        done, t0 = 0, time.time()
        with torch.no_grad():
            for step, (x0, x1, types, idx) in enumerate(loader):
                x0, x1 = x0.to(dev, non_blocking=True), x1.to(dev, non_blocking=True)
                mixed = cure_mix(x0.unsqueeze(1), x1.unsqueeze(1), grid).flatten(0, 1)
                for k, m in models.items():
                    z = m.features(mixed).view(x0.shape[0], len(S.GRID), FEATURE_DIM)
                    b, zbar = S.trajectory_summary(z)
                    for j in range(x0.shape[0]):
                        t = TYPES[int(types[j])]
                        outs[k][f"b_{t}"][int(idx[j])] = b[j].float().cpu()
                        outs[k][f"zbar_{t}"][int(idx[j])] = zbar[j].float().cpu()
                done += x0.shape[0]
                if (step + 1) % 200 == 0:
                    print(f"  trajectories {done}/{3 * args.pairs} ({done / (time.time() - t0):.1f} pairs/s)", flush=True)
        for k, (lora, out) in runs.items():
            traj = outs[k]
            traj["pairs"] = pairs
            traj["meta"] = json.dumps({"lora": lora, "data": args.data, "pairs": args.pairs, "seed": args.seed,
                                       "max_images": args.max_images, "control": "none", "driver": "repro/cure_stage1/multi_stage23.py"})
            torch.save(traj, out / "trajectories.pt")
        del outs

    # ---- stage 3 head features (train_stages.stage3's LabelledImages, all training images)
    if not all((o / "head_features.pt").exists() for _, o in runs.values()):
        print(f"[multi] stage-3 features: {len(ep.reals)} + {len(ep.fakes)} images, {len(runs)} encoders", flush=True)
        loader = DataLoader(LabelledImages(ep), batch_size=args.encode_batch, shuffle=False, num_workers=args.workers,
                            pin_memory=True)
        feats, labels, t0 = {k: [] for k in runs}, [], time.time()
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x = x.to(dev, non_blocking=True)
                for k, m in models.items():
                    feats[k].append(m.features(x).float().cpu())
                labels.append(y)
                if (i + 1) % 100 == 0:
                    n = sum(len(f) for f in labels)
                    print(f"  head features {n}/{len(loader.dataset)} ({n / (time.time() - t0):.0f} images/s)", flush=True)
        y = torch.cat(labels)
        for k, (_, out) in runs.items():
            torch.save({"z": torch.cat(feats[k]), "y": y}, out / "head_features.pt")
        del feats
    del models
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()

    # ---- the recipe's own solve / head / export, reusing the caches
    for k, (lora, out) in runs.items():
        common = ["--lora", lora, "--data", args.data, "--out", str(out), "--seed", str(args.seed), "--device", dev]
        if args.max_images:
            common += ["--max-images", str(args.max_images)]
        a2 = ["stage2", *common, "--pairs", str(args.pairs)]
        if args.reference_adapter:
            a2 += ["--reference-adapter", args.reference_adapter]
        print(f"[multi] {k}: train_stages {' '.join(a2)}", flush=True)
        train_stages_main(a2)
        a3 = ["stage3", *common]
        print(f"[multi] {k}: train_stages {' '.join(a3)}", flush=True)
        train_stages_main(a3)


if __name__ == "__main__":
    main()
