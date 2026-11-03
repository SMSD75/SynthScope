"""MoE-JEPA on the SID-Set test split, three-way, with the official metrics (paper Table 1).

The official ``train/test.py`` runs the model in Lightning's ``bf16-mixed`` precision on the test split and logs, from
``on_validation_epoch_end``: accuracy over all images ("Total Acc"), the per-class recall ("Acc" of Real / Fully
Synthetic / Tampered), the per-class F1 and the macro F1 ("Total F1"), all via scikit-learn on the argmax of the three
logits. This script computes the same on the SID-Set test images (the ``sidset`` loader keeps the three-class tree in
``<root>/_sidset_test/{real,full_synthetic,tampered}``), with SynthScope's MoE-JEPA, whose weights, preprocessing and
logits are identical to the official model's. Both precisions are reported: ``bf16`` (autocast, as the official test)
and ``fp32`` (SynthScope's default).

    python repro/official_evals/moejepa_sidset_eval.py --root /data/SIDSet [--limit N] [--bs 32]

Output: repro/bench_results/<tag>.json and a line in repro/MASTER_RESULTS.jsonl.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "interface")]
from backend.engine import get_engine        # noqa: E402
from backend.registry import get_spec        # noqa: E402

REPRO = REPO / "repro"
CLASSES = ("real", "full_synthetic", "tampered")          # SID-Set labels 0 / 1 / 2
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class ThreeClassFolder(Dataset):
    def __init__(self, root: Path, transform, limit: int = 0):
        self.items = []
        for label, c in enumerate(CLASSES):
            files = sorted(p for p in (root / c).rglob("*") if p.suffix.lower() in EXTS)
            self.items += [(p, label) for p in (files[:limit] if limit else files)]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, label = self.items[i]
        with Image.open(p) as im:
            return self.transform(im.convert("RGB")), label


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """The official on_validation_epoch_end metrics, in percent."""
    rec = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    out = {c: {"acc": round(float(r) * 100, 2), "f1": round(float(f) * 100, 2)} for c, r, f in zip(CLASSES, rec, f1)}
    out["total"] = {"acc": round(accuracy_score(y_true, y_pred) * 100, 2),
                    "f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="SID-Set root (with _sidset_test/{real,full_synthetic,tampered})")
    ap.add_argument("--limit", type=int, default=0, help="images per class (0 = all)")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="moejepa_sidset_3way")
    args = ap.parse_args(argv)

    engine = get_engine(); spec = get_spec("moejepa"); model = engine._get_model(spec)
    data = ThreeClassFolder(Path(args.root) / "_sidset_test", spec.transform, args.limit)
    counts = {c: sum(1 for _, l in data.items if l == i) for i, c in enumerate(CLASSES)}
    print(f"=== moejepa SID-Set test 3-way | {counts} | device={engine.device} ===", flush=True)
    loader = DataLoader(data, batch_size=args.bs, num_workers=args.workers, shuffle=False)
    labels, preds = [], {"bf16": [], "fp32": []}
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(engine.device)
            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
                preds["bf16"].append(model.forward_3way(x).float().argmax(1).cpu())
            preds["fp32"].append(model.forward_3way(x).argmax(1).cpu())
            labels.append(y)
            if i % 200 == 0:
                print(f"  {i * args.bs}/{len(data)}", flush=True)
    y = torch.cat(labels).numpy()
    result = {"tag": args.tag, "model": "moejepa", "ckpt": "official moejepa_sidset.safetensors",
              "benchmark": "sidset:3way", "data_root": str(args.root), "n_images": counts,
              "bf16": summarize(y, torch.cat(preds["bf16"]).numpy()),
              "fp32": summarize(y, torch.cat(preds["fp32"]).numpy()),
              "date": datetime.datetime.now().isoformat(timespec="seconds")}
    out = REPRO / "bench_results" / f"{args.tag}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    with (REPRO / "MASTER_RESULTS.jsonl").open("a") as f:
        f.write(json.dumps({"date": result["date"], "tag": args.tag, "model": "moejepa", "ckpt": result["ckpt"],
                            "benchmark": result["benchmark"], "data_root": result["data_root"],
                            "means": {"acc_bf16": result["bf16"]["total"]["acc"],
                                      "f1_bf16": result["bf16"]["total"]["f1"],
                                      "acc_fp32": result["fp32"]["total"]["acc"]}}) + "\n")
    for k in ("bf16", "fp32"):
        print(f"=== RESULT {args.tag} [{k}]: " + " | ".join(f"{c} acc {v['acc']} f1 {v['f1']}" for c, v in result[k].items()))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
