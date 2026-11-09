"""CuRe stage-1 investigation, Task 2: per-epoch selection numbers for the R1 / R2 stage-1 retrains (and reference rows
for the released encoder, the frozen base and our original run), on NON-benchmark data only.

PRE-REGISTERED SELECTION RULE (fixed before any R1 / R2 epoch was scored; plans/2026-11-09-cure-stage1-investigation.md,
Task 2 ruling): for R1, the selected epoch is the one with the highest
    select_score = mean( Synthbuster bacc of the stage-1 ratio head used directly (threshold 0.5),
                         Synthbuster bacc of the SD1.4-fitted linear probe (logistic regression, C = 1, threshold 0.5) )
where "Synthbuster" = the 1287 Synthbuster fakes (143 per generator, every 7th file; not a CuRe paper benchmark) scored
together with the 2000 SD1.4-test reals; ties within 0.5 points go to the EARLIER epoch. R2 uses its final epoch (ruling).
Reported for every epoch but not used in the rule: the LoRA update norm (total ||2 B A||_F, and its ratio to the released
adapter's), the SD1.4 mixing fit (excess BCE over the floor of a fresh linear ratio head and of the run's own ratio head
on the 2000 SD1.4-test "after" mixtures), the SD1.4-test bacc, and WildRF (DIAGNOSTIC ONLY, never used for selection).

    python repro/cure_stage1/epoch_select.py --enc released=<cure_adapter.pt> --enc r1e0=<model_0_lora.pth> ... \
        --train-dir ... --test-dir ... --wild-dir ... --extra-fake-dir ... --out <dir>

Features are cached per encoder in ``<out>/feat_<name>.pt`` (a rerun encodes only new encoders); the numbers go to
``<out>/epoch_select.json``. The image sets and mixing pairs are exactly those of ``feature_diag.py`` (same seed).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from models.image.cure.train_stages import build_model
from repro.cure_stage1.feature_diag import Jobs, list_images
from repro.cure_stage1.feature_analysis import apply_head, binary_metrics, fit_ratio_head, ratio_metrics

SCALE = 2.0
RELEASED_TOTAL_DW = None   # computed from --enc released


def build_sets(args):
    tr_r, tr_f = list_images(Path(args.train_dir) / "real"), list_images(Path(args.train_dir) / "fake")
    te_r, te_f = list_images(Path(args.test_dir) / "real"), list_images(Path(args.test_dir) / "fake")
    wild = [(p, 0 if "0_real" in p.parts else 1) for p in list_images(args.wild_dir)]
    extra = list_images(args.extra_fake_dir)
    g = torch.Generator().manual_seed(1234)          # feature_diag.py's seed and draw order (mix_train, then mix_val)
    meta, jobs = {}, []

    def add(name, js, **cols):
        meta[name] = {"start": len(jobs), "n": len(js), **cols}
        jobs.extend(js)
    add("train_clean", [("clean", p) for p in tr_r + tr_f], label=[0] * len(tr_r) + [1] * len(tr_f))
    add("test_clean", [("clean", p) for p in te_r + te_f], label=[0] * len(te_r) + [1] * len(te_f))
    for split, (R, F_, n) in {"mix_train": (tr_r, tr_f, 4000), "mix_val": (te_r, te_f, 2000)}.items():
        ri = torch.randint(len(R), (n,), generator=g).tolist()
        fi = torch.randint(len(F_), (n,), generator=g).tolist()
        al = torch.rand(n, generator=g, dtype=torch.float64).tolist()
        add(f"{split}_after", [("mix", R[a], F_[b], c, "after") for a, b, c in zip(ri, fi, al)], alpha=al)
    add("wild", [("clean", w[0]) for w in wild], label=[w[1] for w in wild])
    add("synthbuster_fake", [("clean", p) for p in extra], label=[1] * len(extra),
        generator=[p.relative_to(args.extra_fake_dir).parts[0] for p in extra])
    return meta, jobs, len(te_r)


def lora_total_dw(path):
    st = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    st = st.get("model", st)
    tot = 0.0
    for layer in range(24):
        for proj in "qv":
            p = f"backbone.transformer.resblocks.{layer}.attn.lora_"
            tot += float((SCALE * st[p + f"B_{proj}"].double() @ st[p + f"A_{proj}"].double()).norm() ** 2)
    ratio = (st["ratio_head.weight"].float().view(-1), st["ratio_head.bias"].float().view(())) if "ratio_head.weight" in st else None
    return tot ** 0.5, ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", action="append", required=True, help="name=path (LoRA source; 'base' = no LoRA)")
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--wild-dir", required=True)
    ap.add_argument("--extra-fake-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    encs = dict(e.split("=", 1) for e in args.enc)
    meta, jobs, n_te_r = build_sets(args)
    print(f"[sel] {len(jobs)} images: " + ", ".join(f"{k} {v['n']}" for k, v in meta.items()), flush=True)

    todo = [k for k in encs if not (out / f"feat_{k}.pt").is_file()]
    if todo:
        models = {k: build_model(None if encs[k] == "base" else encs[k], "cuda") for k in todo}
        feats = {k: torch.zeros(len(jobs), 1024, dtype=torch.float16) for k in todo}
        loader = DataLoader(Jobs(jobs), batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)
        t0 = time.time()
        with torch.no_grad():
            for step, (x, idx) in enumerate(loader):
                x = x.cuda(non_blocking=True)
                for k, m in models.items():
                    feats[k][idx] = m.features(x).half().cpu()
                if step % 100 == 0:
                    print(f"[sel] {(step + 1) * args.bs}/{len(jobs)} {(step + 1) * args.bs / (time.time() - t0):.1f} img/s", flush=True)
        for k in todo:
            assert torch.isfinite(feats[k].float()).all(), k
            torch.save({"features": feats[k], "n": len(jobs)}, out / f"feat_{k}.pt")
        del models

    def blk(F, name):
        m = meta[name]
        return F[m["start"]:m["start"] + m["n"]].float()
    y_tr = np.asarray(meta["train_clean"]["label"])
    y_te = np.asarray(meta["test_clean"]["label"])
    y_wild = np.asarray(meta["wild"]["label"])
    n_sb = meta["synthbuster_fake"]["n"]
    y_sb = np.r_[np.zeros(n_te_r), np.ones(n_sb)]
    a_tr, a_va = torch.tensor(meta["mix_train_after"]["alpha"]), torch.tensor(meta["mix_val_after"]["alpha"])
    rel_dw = lora_total_dw(encs["released"])[0] if "released" in encs and encs["released"] != "base" else None
    res = {"rule": "select_score = mean(Synthbuster bacc of the stage-1 ratio head direct, Synthbuster bacc of the SD1.4 "
                   "linear probe); argmax over R1 epochs, ties within 0.5 -> earlier epoch; R2: final epoch",
           "released_total_dW": rel_dw, "rows": {}}
    for k, path in encs.items():
        Fk = torch.load(out / f"feat_{k}.pt", weights_only=False)["features"]
        row = {}
        if path != "base":
            dw, ratio = lora_total_dw(path)
            row["total_dW"] = round(dw, 4)
            if rel_dw:
                row["dW_over_released"] = round(dw / rel_dw, 3)
        else:
            ratio = None
        head = fit_ratio_head(blk(Fk, "mix_train_after"), a_tr)
        row["mix_fit_fresh_head"] = ratio_metrics(apply_head(head, blk(Fk, "mix_val_after")), a_va, True)
        sb_X = torch.cat([blk(Fk, "test_clean")[:n_te_r], blk(Fk, "synthbuster_fake")])
        sc = StandardScaler().fit(blk(Fk, "train_clean").numpy())
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(blk(Fk, "train_clean").numpy()), y_tr)
        probe = lambda X: clf.predict_proba(sc.transform(X.numpy()))[:, 1]
        row["probe_sd14_test"] = binary_metrics(probe(blk(Fk, "test_clean")), y_te)
        row["probe_synthbuster"] = binary_metrics(probe(sb_X), y_sb)
        row["probe_wildrf_DIAGNOSTIC"] = binary_metrics(probe(blk(Fk, "wild")), y_wild)
        if ratio is not None:   # the run's own stage-1 ratio head, used directly (paper Table 4 "Direct")
            w, b = ratio
            direct = lambda X: torch.sigmoid(X @ w + b).numpy()
            row["mix_fit_own_head"] = ratio_metrics(blk(Fk, "mix_val_after") @ w + b, a_va, False)
            tag = "own"
        else:                   # released / base have no ratio head: the fresh head (reference rows only)
            direct = lambda X: apply_head(head, X).numpy()
            tag = "fresh"
        row["direct_head"] = tag
        row["direct_sd14_test"] = binary_metrics(direct(blk(Fk, "test_clean")), y_te)
        row["direct_synthbuster"] = binary_metrics(direct(sb_X), y_sb)
        row["direct_wildrf_DIAGNOSTIC"] = binary_metrics(direct(blk(Fk, "wild")), y_wild)
        row["select_score"] = round((row["direct_synthbuster"]["bacc"] + row["probe_synthbuster"]["bacc"]) / 2, 3)
        res["rows"][k] = row
        print(f"[sel] {k:10s} dW {row.get('total_dW', '-')} ({row.get('dW_over_released', '-')}x rel) | mix excess fresh "
              f"{row['mix_fit_fresh_head']['excess_bce']} own {row.get('mix_fit_own_head', {}).get('excess_bce', '-')} | "
              f"SB direct {row['direct_synthbuster']['bacc']} probe {row['probe_synthbuster']['bacc']} -> score "
              f"{row['select_score']} | SD14 direct {row['direct_sd14_test']['bacc']} | WildRF(diag) direct "
              f"{row['direct_wildrf_DIAGNOSTIC']['bacc']} probe {row['probe_wildrf_DIAGNOSTIC']['bacc']}", flush=True)
    r1 = sorted(k for k in res["rows"] if k.startswith("r1e"))
    if r1:
        best = max(res["rows"][k]["select_score"] for k in r1)
        res["r1_selected"] = next(k for k in r1 if res["rows"][k]["select_score"] >= best - 0.5)
        print(f"[sel] R1 selected (pre-registered rule): {res['r1_selected']}", flush=True)
    with open(out / "epoch_select.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
