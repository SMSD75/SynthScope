"""CuRe stages 2 and 3 (arXiv 2609.14316, Sec. 3.2-3.3, Appendix A), after stage 1 (experiments/image/cure.sh).

    python -m models.image.cure.train_stages stage2 --lora <stage-1 model_last.pth> --data <GenImage SD v1.4> --out <dir>
    python -m models.image.cure.train_stages stage3 --lora <stage-1 model_last.pth> --data <GenImage SD v1.4> --out <dir>

stage2  freezes the stage-1 encoder, draws ``--pairs`` (20,000) endpoint pairs per trajectory type (RF real->generated,
        RR real->real, FF generated->generated; independent random endpoints, a same-class pair never uses one image
        twice), encodes the six mixtures of each on the grid {0, .2, .4, .6, .8, 1}, keeps the per-trajectory slope and
        mean (Eq. 6) in ``<out>/trajectories.pt`` and solves Eq. 7-11 (``subspace.py``) into ``<out>/subspace.pt``
        (W [1024, 128], mu, the generalised eigenvalues). A rerun with other ``--lambda-s/--lambda-m/--eps`` reuses the
        cached trajectories. ``--reference-adapter`` (the released cure_adapter.pt) prints the overlap of the solved
        subspace with the released W, the distance of mu to the released mu, and the (lambda_s, lambda_m, eps) for which
        the released W is B-orthonormal under our C_S / C_M (a least-squares fit; a check of the unspecified choices).
stage3  encodes the training images once (``<out>/head_features.pt``; all 162k + 162k by default, clean endpoints with
        the same preprocessing), projects them, ``u = W^T (z - mu)``, and trains only Linear(128, 2) with dropout 0.05 on
        u: cross-entropy, AdamW lr 2e-4, weight decay 0.1, batch 256, 5 epochs (constant lr: the paper names no
        schedule). The result is written as ``<out>/cure_trained.pth`` with exactly the released adapter's 100 keys
        (96 LoRA tensors, mu, W, head), loadable as ``cure.pth`` (``bench_eval --model cure --ckpt``); the export is
        checked by reloading it into a fresh model and comparing its logits with the trained head's on cached features.

Both stages run on one GPU (fp16 autocast inside the model, as the released inference). ``--lora`` takes any state dict
holding the 96 LoRA tensors (a stage-1 trainer checkpoint, or the released adapter for the parity check).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from dataset import CuReEndpoints, cure_mix
from transformations import AIGITransformations
from .cure import FEATURE_DIM, SUBSPACE_DIM, SynthScopeCuReModel
from . import subspace as S

ROOT = Path(__file__).resolve().parents[3]
MODEL_YAML = ROOT / "conf" / "model" / "image" / "cure.yaml"
TYPES = ("RF", "RR", "FF")


def build_model(lora_path: str | None, device: str) -> SynthScopeCuReModel:
    """The frozen CuRe encoder (PE Core base + the LoRA tensors of ``lora_path``), eval mode on ``device``."""
    cfg = OmegaConf.create({"ModelConfig": OmegaConf.load(MODEL_YAML), "TrainerConfig": {}})
    model = SynthScopeCuReModel(cfg)
    if lora_path:
        state = torch.load(lora_path, map_location="cpu", weights_only=True, mmap=True)
        state = state.get("model", state) if isinstance(state, dict) else state
        lora = {k: v for k, v in state.items() if "lora_" in k}
        if len(lora) != 96:
            raise RuntimeError(f"{lora_path}: expected 96 LoRA tensors, found {len(lora)}")
        missing, unexpected = model.load_state_dict(lora, strict=False)
        if unexpected or any("lora_" in k for k in missing):
            raise RuntimeError(f"LoRA load mismatch: unexpected={unexpected[:4]}")
    model.requires_grad_(False)
    return model.to(device).eval()


def endpoint_transform() -> AIGITransformations:
    """The stage-1 endpoint preprocessing (the model config's eval list), see ``dataset.CuReEndpoints``."""
    return AIGITransformations(list(OmegaConf.load(MODEL_YAML).eval_transform_list))


def draw_pairs(n_pairs: int, n_a: int, n_b: int, same_class: bool, gen: torch.Generator) -> torch.Tensor:
    """``[n_pairs, 2]`` endpoint indices: independent uniform draws (without replacement while the class is large
    enough); for a same-class pair the second index is uniform over the other images of the class."""
    def draw(n):
        return torch.randperm(n, generator=gen)[:n_pairs] if n_pairs <= n else torch.randint(n, (n_pairs,), generator=gen)
    first = draw(n_a)
    if same_class:
        second = (first + 1 + torch.randint(n_a - 1, (n_pairs,), generator=gen)) % n_a
    else:
        second = draw(n_b)
    return torch.stack([first, second], dim=1)


class TrajectoryPairs(Dataset):
    """Endpoint pairs ``(x_0, x_1, type index, pair index)``; the mixtures are formed on the GPU."""

    def __init__(self, ep: CuReEndpoints, pairs: dict[str, torch.Tensor]):
        self.ep = ep
        self.items = [(t, i, int(a), int(b)) for t in TYPES for i, (a, b) in enumerate(pairs[t].tolist())]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        t, i, a, b = self.items[idx]
        lists = {"RF": (self.ep.reals, self.ep.fakes), "RR": (self.ep.reals, self.ep.reals), "FF": (self.ep.fakes, self.ep.fakes)}[t]
        g = torch.Generator().manual_seed(idx)
        x0 = self.ep.load_any(lists[0], a, lambda: int(torch.randint(len(lists[0]), (), generator=g)))
        x1 = self.ep.load_any(lists[1], b, lambda: int(torch.randint(len(lists[1]), (), generator=g)))
        return x0, x1, TYPES.index(t), i


class LabelledImages(Dataset):
    """All (or the first ``max_per_class``) training images with labels 0 = real, 1 = generated."""

    def __init__(self, ep: CuReEndpoints):
        self.ep = ep
        self.items = [(p, 0) for p in ep.reals] + [(p, 1) for p in ep.fakes]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        paths = self.ep.reals if self.items[idx][1] == 0 else self.ep.fakes
        offset = 0 if self.items[idx][1] == 0 else len(self.ep.reals)
        g = torch.Generator().manual_seed(idx)
        return self.ep.load_any(paths, idx - offset, lambda: int(torch.randint(len(paths), (), generator=g))), self.items[idx][1]


def encode_trajectories(model, loader, n_pairs: int, device: str) -> dict:
    """Slopes and means of every trajectory: ``{"b_RF": [N, D], "zbar_RF": [N, D], ...}`` (float32)."""
    grid = torch.tensor(S.GRID, device=device).view(1, -1, 1, 1, 1)
    out = {f"{k}_{t}": torch.zeros(n_pairs, FEATURE_DIM) for t in TYPES for k in ("b", "zbar")}
    done, start = 0, time.time()
    with torch.no_grad():
        for step, (x0, x1, types, idx) in enumerate(loader):
            x0, x1 = x0.to(device, non_blocking=True), x1.to(device, non_blocking=True)
            mixed = cure_mix(x0.unsqueeze(1), x1.unsqueeze(1), grid)          # [P, 6, 3, 336, 336]
            z = model.features(mixed.flatten(0, 1)).view(x0.shape[0], len(S.GRID), FEATURE_DIM)
            b, zbar = S.trajectory_summary(z)
            for j in range(x0.shape[0]):
                t = TYPES[int(types[j])]
                out[f"b_{t}"][int(idx[j])] = b[j].float().cpu()
                out[f"zbar_{t}"][int(idx[j])] = zbar[j].float().cpu()
            done += x0.shape[0]
            if (step + 1) % 50 == 0:
                rate = done / (time.time() - start)
                print(f"  trajectories {done}/{3 * n_pairs} ({rate:.1f} pairs/s, {6 * rate:.0f} images/s)", flush=True)
    return out


def fit_normalisation(w_ref: torch.Tensor, c_s: torch.Tensor, c_m: torch.Tensor) -> dict:
    """Least squares (lambda_s, lambda_m, eps) with ``W_ref^T (lambda_s C_S + lambda_m C_M + eps I) W_ref = I`` over all
    entries: if the released W is B-orthonormal (our solver's convention) under statistics like ours, this recovers its
    B. ``rel_residual`` = ||fit - I||_F / ||I||_F."""
    w = w_ref.double()
    blocks = [w.T @ c_s.double() @ w, w.T @ c_m.double() @ w, w.T @ w]
    a = torch.stack([blk.flatten() for blk in blocks], dim=1)
    target = torch.eye(w.shape[1], dtype=torch.float64).flatten()
    coef = torch.linalg.lstsq(a, target.unsqueeze(1)).solution.squeeze(1)
    resid = (a @ coef - target).norm() / target.norm()
    return {"lambda_s": float(coef[0]), "lambda_m": float(coef[1]), "eps": float(coef[2]), "rel_residual": float(resid)}


def reference_report(stats: dict, w: torch.Tensor, adapter_path: str) -> dict:
    ref = torch.load(adapter_path, map_location="cpu", weights_only=True)
    w_ref, mu_ref = ref["W"].double(), ref["mu"].double()
    mu = stats["mu"]
    g = w_ref.T @ stats["C_F"] @ w_ref
    report = {
        "overlap_with_released_W": S.subspace_overlap(w, w_ref),
        "overlap_random_baseline": SUBSPACE_DIM / FEATURE_DIM,
        "mu_cosine": float(F.cosine_similarity(mu, mu_ref, dim=0)),
        "mu_rel_dist": float((mu - mu_ref).norm() / mu_ref.norm()),
        "released_W_col_norms": [float(x) for x in w_ref.norm(dim=0)[:4]],
        "solved_W_col_norms": [float(x) for x in w.norm(dim=0)[:4]],
        # released W against our C_F: a generalised eigenbasis makes W^T C_F W diagonal
        "released_WtCfW_offdiag_ratio": float((g - torch.diag(torch.diag(g))).norm() / torch.diag(g).norm()),
        "normalisation_fit": fit_normalisation(w_ref, stats["C_S"], stats["C_M"]),
    }
    return report


def stage2(args) -> None:
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    traj_path = out / "trajectories.pt"
    if traj_path.exists() and not args.recompute:
        print(f"[stage2] reusing {traj_path}", flush=True)
        traj = torch.load(traj_path, map_location="cpu", weights_only=True)
    else:
        ep = CuReEndpoints(args.data, "train", endpoint_transform(), args.max_per_class)
        gen = torch.Generator().manual_seed(args.seed)
        n_r, n_f = len(ep.reals), len(ep.fakes)
        pairs = {"RF": draw_pairs(args.pairs, n_r, n_f, False, gen), "RR": draw_pairs(args.pairs, n_r, n_r, True, gen),
                 "FF": draw_pairs(args.pairs, n_f, n_f, True, gen)}
        print(f"[stage2] {n_r} real / {n_f} generated images, {args.pairs} pairs per type x 6 ratios "
              f"= {18 * args.pairs} forwards", flush=True)
        model = build_model(args.lora, args.device)
        loader = DataLoader(TrajectoryPairs(ep, pairs), batch_size=args.batch_pairs, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
        traj = encode_trajectories(model, loader, args.pairs, args.device)
        traj["pairs"] = pairs
        traj["meta"] = json.dumps({"lora": args.lora, "data": args.data, "pairs": args.pairs, "seed": args.seed,
                                   "max_per_class": args.max_per_class})
        torch.save(traj, traj_path)
        del model
    stats = S.statistics(traj["b_RF"], traj["b_RR"], traj["b_FF"], traj["zbar_RF"])
    w, gammas = S.solve(stats["C_F"], stats["C_M"], stats["C_S"], k=args.k,
                        lambda_s=args.lambda_s, lambda_m=args.lambda_m, eps=args.eps)
    result = {"W": w.float(), "mu": stats["mu"].float(), "gammas": gammas,
              "config": {"k": args.k, "lambda_s": args.lambda_s, "lambda_m": args.lambda_m, "eps": args.eps}}
    print(f"[stage2] solved: k={args.k}, lambda_s={args.lambda_s}, lambda_m={args.lambda_m}, eps={args.eps}; "
          f"top eigenvalues {[round(float(x), 3) for x in gammas[:5]]}, k-th {float(gammas[-1]):.4f}", flush=True)
    if args.reference_adapter:
        report = reference_report(stats, w, args.reference_adapter)
        result["reference_report"] = report
        print("[stage2] vs released adapter: " + json.dumps(report), flush=True)
    torch.save(result, out / args.subspace_name)
    print(f"[stage2] wrote {out / args.subspace_name}", flush=True)


def encode_images(model, loader, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    feats, labels, start = [], [], time.time()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            feats.append(model.features(x.to(device, non_blocking=True)).float().cpu())
            labels.append(y)
            if (i + 1) % 100 == 0:
                n = sum(len(f) for f in feats)
                print(f"  head features {n}/{len(loader.dataset)} ({n / (time.time() - start):.0f} images/s)", flush=True)
    return torch.cat(feats), torch.cat(labels)


def train_head(u: torch.Tensor, y: torch.Tensor, args) -> nn.Linear:
    """Linear(128, 2) on ``u`` with dropout on its input: CE, AdamW, constant lr, shuffled mini-batches."""
    torch.manual_seed(args.seed)
    head = nn.Linear(SUBSPACE_DIM, 2).to(u.device)
    drop = nn.Dropout(args.dropout)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gen = torch.Generator().manual_seed(args.seed)
    for epoch in range(args.epochs):
        head.train(); drop.train()
        order = torch.randperm(len(y), generator=gen).to(u.device)
        total, correct = 0.0, 0
        for start in range(0, len(y), args.batch):
            idx = order[start:start + args.batch]
            logits = head(drop(u[idx]))
            loss = F.cross_entropy(logits, y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss) * len(idx)
            correct += int((logits.argmax(1) == y[idx]).sum())
        head.eval()
        with torch.no_grad():
            pred = head(u).argmax(1)
            real_acc = float((pred[y == 0] == 0).float().mean())
            fake_acc = float((pred[y == 1] == 1).float().mean())
        print(f"[stage3] epoch {epoch}: train loss {total / len(y):.4f} (dropout on), train acc {correct / len(y):.4f}; "
              f"eval-mode acc real {real_acc:.4f} / fake {fake_acc:.4f}, lr {opt.param_groups[0]['lr']:.1e}", flush=True)
    return head.eval()


def stage3(args) -> None:
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sub = torch.load(args.subspace or out / "subspace.pt", map_location="cpu", weights_only=True)
    w, mu = sub["W"].float(), sub["mu"].float()
    feat_path = out / "head_features.pt"
    ep = CuReEndpoints(args.data, "train", endpoint_transform(), args.max_per_class)
    model = build_model(args.lora, args.device)
    if feat_path.exists() and not args.recompute:
        print(f"[stage3] reusing {feat_path}", flush=True)
        cache = torch.load(feat_path, map_location="cpu", weights_only=True)
        z, y = cache["z"], cache["y"]
    else:
        print(f"[stage3] encoding {len(ep.reals)} real + {len(ep.fakes)} generated images", flush=True)
        loader = DataLoader(LabelledImages(ep), batch_size=args.encode_batch, shuffle=False, num_workers=args.workers,
                            pin_memory=True)
        z, y = encode_images(model, loader, args.device)
        torch.save({"z": z, "y": y}, feat_path)
    dev = args.device
    u = ((z.to(dev) - mu.to(dev)) @ w.to(dev)).float()
    head = train_head(u, y.to(dev), args)

    # export in the released adapter's format and check it reproduces the trained head on real inputs
    with torch.no_grad():
        model.mu.copy_(mu.to(dev)); model.W.copy_(w.to(dev))
        model.head.weight.copy_(head.weight); model.head.bias.copy_(head.bias)
    adapter = model.adapter_state_dict()
    torch.save(adapter, out / "cure_trained.pth")
    print(f"[stage3] wrote {out / 'cure_trained.pth'} ({len(adapter)} tensors)", flush=True)
    check = build_model(None, dev)
    missing, unexpected = check.load_state_dict(torch.load(out / "cure_trained.pth", map_location="cpu", weights_only=True),
                                                strict=False)
    stray = [k for k in missing if "lora_" in k or k in ("mu", "W", "head.weight", "head.bias")]
    n = min(64, len(ep.reals))
    probe = [ep.load(ep.reals[i]) for i in range(n // 2)] + [ep.load(ep.fakes[i]) for i in range(n - n // 2)]
    idx = torch.tensor(list(range(n // 2)) + [len(ep.reals) + i for i in range(n - n // 2)])
    with torch.no_grad():
        logits_export = check(torch.stack(probe).to(dev)).float().cpu()
        logits_cached = head(u[idx.to(dev)]).float().cpu()
    diff = float((logits_export - logits_cached).abs().max())
    print(f"[stage3] export check: missing adapter keys {stray}, unexpected {unexpected}; reloaded-model logits vs trained "
          f"head on cached features, max |diff| {diff:.2e} over {n} images", flush=True)
    if stray or unexpected or diff > 1e-2:
        raise RuntimeError("exported adapter does not reproduce the trained head")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    for name in ("stage2", "stage3"):
        p = sub.add_parser(name)
        p.add_argument("--lora", required=True, help="state dict with the 96 stage-1 LoRA tensors (trainer model_last.pth)")
        p.add_argument("--data", required=True, help="GenImage SD v1.4 root (train/real, train/fake)")
        p.add_argument("--out", required=True)
        p.add_argument("--max-per-class", type=int, default=None, help="first N images of each class (smoke runs)")
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--device", default="cuda")
        p.add_argument("--recompute", action="store_true", help="re-encode even if the cached features exist")
    p2 = sub.choices["stage2"]
    p2.add_argument("--pairs", type=int, default=20000, help="endpoint pairs per trajectory type (paper: 20,000)")
    p2.add_argument("--batch-pairs", type=int, default=16, help="pairs per forward (x 6 mixtures)")
    p2.add_argument("--k", type=int, default=SUBSPACE_DIM)
    p2.add_argument("--lambda-s", type=float, default=S.DEFAULT_LAMBDA_S)
    p2.add_argument("--lambda-m", type=float, default=S.DEFAULT_LAMBDA_M)
    p2.add_argument("--eps", type=float, default=S.DEFAULT_EPS)
    p2.add_argument("--reference-adapter", default=None, help="released cure_adapter.pt: compare W / mu with it")
    p2.add_argument("--subspace-name", default="subspace.pt")
    p3 = sub.choices["stage3"]
    p3.add_argument("--subspace", default=None, help="default <out>/subspace.pt")
    p3.add_argument("--epochs", type=int, default=5)
    p3.add_argument("--lr", type=float, default=2e-4)
    p3.add_argument("--weight-decay", type=float, default=0.1)
    p3.add_argument("--dropout", type=float, default=0.05)
    p3.add_argument("--batch", type=int, default=256)
    p3.add_argument("--encode-batch", type=int, default=128)
    args = ap.parse_args(argv)
    print(f"[{args.stage}] args: {vars(args)}", flush=True)
    {"stage2": stage2, "stage3": stage3}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
