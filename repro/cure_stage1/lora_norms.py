"""CuRe stage-1 investigation (Task 1, diagnostic 2c): per-layer LoRA statistics of the released adapter vs our trained
stage-1 LoRA, CPU only.

For every one of the 24 attention layers and both adapted projections (q, v) it reports the Frobenius norms of A and B,
the effective update dW = (alpha / r) * B @ A (alpha / r = 2, as the official forward), its spectral norm and stable
rank, ||dW|| relative to the frozen base projection, the largest |B_ij| and |A_ij|, and the subspace overlap of the
released and our dW (top-8 left singular vectors).

Why the max |B_ij| matters: B starts at 0 and AdamW moves an element by about lr_t per step (|m / sqrt(v)| <= ~1 for
steady gradients; the decoupled weight decay is negligible at lr 1e-5), so max |B_ij| <= sum_t lr_t ~= the "update
budget" of the run. For the paper's schedule (2 warmup epochs linear from 0, cosine over 3 epochs from 1e-5 to 1e-6)
the budget is about 2.65 * lr * S with S optimizer steps per epoch = 162,000 / global batch. A released max |B_ij| far
above our run's budget means the release took more (or larger) optimizer steps than our 843 steps/epoch.

    python repro/cure_stage1/lora_norms.py --released cure_adapter.pt --ours model_last.pth --base PE-Core-L14-336.pt \
        --out repro/cure_stage1/results/lora_norms.json
"""
import argparse
import json
import math

import torch

LAYERS = 24
RANK = 64
SCALE = 2.0            # alpha / r = 128 / 64 (official backbone.py: 2.0 * B(A(x)))
KAIMING_BOUND = 1 / 32  # kaiming_uniform_(a=sqrt(5)) on [64, 1024]: U(-1/sqrt(1024), 1/sqrt(1024))


def lora(state, layer, proj):
    p = f"backbone.transformer.resblocks.{layer}.attn.lora_"
    return state[p + f"A_{proj}"].double(), state[p + f"B_{proj}"].double()


def budget(steps_per_epoch, lr=1e-5, min_lr=1e-6, warm=2, total=5):
    """sum_t lr_t for linear warmup from 0 over ``warm`` epochs + cosine from lr to min_lr over the rest."""
    s_w, s_c = warm * steps_per_epoch, (total - warm) * steps_per_epoch
    warm_sum = sum(lr * i / s_w for i in range(s_w))
    cos_sum = sum(min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * i / s_c)) for i in range(s_c))
    return warm_sum + cos_sum


def stats(A, B, W0):
    dW = SCALE * B @ A
    sv = torch.linalg.svdvals(dW)
    sa = torch.linalg.svdvals(A)
    return {
        "A_fro": float(A.norm()), "B_fro": float(B.norm()), "dW_fro": float(dW.norm()), "dW_spec": float(sv[0]),
        "dW_stable_rank": float((sv ** 2).sum() / sv[0] ** 2), "dW_rel_base": float(dW.norm() / W0.norm()),
        "B_maxabs": float(B.abs().max()), "B_rms": float(B.pow(2).mean().sqrt()), "A_maxabs": float(A.abs().max()),
        "A_sv_max_over_min": float(sa[0] / sa[-1]),
    }


def top_left(dW, k=8):
    return torch.linalg.svd(dW, full_matrices=False)[0][:, :k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--released", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rel = torch.load(args.released, map_location="cpu", weights_only=True)
    ours = torch.load(args.ours, map_location="cpu", weights_only=True, mmap=True)
    ours = ours.get("model", ours)
    base = torch.load(args.base, map_location="cpu", weights_only=True, mmap=True)
    # random reference: a kaiming-uniform A (the standard LoRA init, = SynthScope's _init_trainable) and its stats
    g = torch.Generator().manual_seed(0)
    A_init = (torch.rand(RANK, 1024, generator=g, dtype=torch.float64) * 2 - 1) * KAIMING_BOUND
    init_ref = {"A_fro": float(A_init.norm()), "A_maxabs": float(A_init.abs().max()),
                "A_sv_max_over_min": float((lambda s: s[0] / s[-1])(torch.linalg.svdvals(A_init)))}
    rows = []
    for layer in range(LAYERS):
        W_in = base[f"visual.transformer.resblocks.{layer}.attn.in_proj_weight"].double()
        for proj, sl in (("q", slice(0, 1024)), ("v", slice(2048, 3072))):
            W0 = W_in[sl]
            Ar, Br = lora(rel, layer, proj)
            Ao, Bo = lora(ours, layer, proj)
            r, o = stats(Ar, Br, W0), stats(Ao, Bo, W0)
            Ur, Uo = top_left(SCALE * Br @ Ar), top_left(SCALE * Bo @ Ao)
            overlap = float((Ur.T @ Uo).pow(2).sum() / Ur.shape[1])
            rows.append({"layer": layer, "proj": proj, "released": r, "ours": o, "top8_left_overlap": overlap})
    def agg(who, key, fn=lambda v: sum(v) / len(v)):
        return fn([row[who][key] for row in rows])
    summary = {}
    for key in ("A_fro", "B_fro", "dW_fro", "dW_spec", "dW_stable_rank", "dW_rel_base", "B_maxabs", "B_rms", "A_maxabs",
                "A_sv_max_over_min"):
        summary[key] = {"released_mean": agg("released", key), "ours_mean": agg("ours", key),
                        "released_max": agg("released", key, max), "ours_max": agg("ours", key, max)}
    summary["dW_fro_ratio_released_over_ours_mean"] = sum(r["released"]["dW_fro"] / r["ours"]["dW_fro"] for r in rows) / len(rows)
    summary["top8_left_overlap_mean"] = sum(r["top8_left_overlap"] for r in rows) / len(rows)
    summary["random_top8_overlap"] = 8 / 1024
    budgets = {f"{n}gpu_x64": {"steps_per_epoch": 162000 // (64 * n), "sum_lr": budget(162000 // (64 * n))}
               for n in (1, 2, 3, 4, 8)}
    out = {"kaiming_init_reference": init_ref, "update_budget_sum_lr": budgets, "summary": summary, "per_layer": rows}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"kaiming_init_reference": init_ref, "update_budget_sum_lr": budgets, "summary": summary}, indent=1))
    print("layer proj | rel dW_fro ours dW_fro | rel Bmax ours Bmax | rel Afro ours Afro | overlap")
    for r in rows:
        print(f"{r['layer']:2d} {r['proj']} | {r['released']['dW_fro']:.3f} {r['ours']['dW_fro']:.3f} | "
              f"{r['released']['B_maxabs']:.4f} {r['ours']['B_maxabs']:.4f} | {r['released']['A_fro']:.3f} "
              f"{r['ours']['A_fro']:.3f} | {r['top8_left_overlap']:.3f}")


if __name__ == "__main__":
    main()
