"""PROBE official-protocol evaluation (Task 2 parity reference).

Runs the UNMODIFIED official evaluate_dino.py / evaluate_resnet.py code path (EvaluateDataset, collate_fn, evaluate)
on <name>=<real dir>,<fake dir> subsets, one subset at a time as separate `python evaluate_*.py --root_list real fake`
invocations would (global seed 42 re-applied before every subset, so the `fake_equal_real` random subsample is the one
the official script draws). README settings: DINOv2 crop 336 / ResNet crop 224, ImageNet normalisation, pad_mode 'repeat'
(dino) / 'pad' (resnet) as in the official __main__, no augmentation. Only the backbone path (local copy of
facebook/dinov2-with-registers-large, hard-coded to a lab path in the official class) and the batch size are set from outside.

Per subset it scores EVERY file once (fake_equal_real=False dataset, official `evaluate`) and reports
  full      : bAcc = (real_acc + fake_acc)/2 over all files (= the paper's metric and bench_eval's `acc`), AP, pooled accuracy
  official  : the numbers the official script prints ("Overall Accuracy" = pooled accuracy, fake/real acc, AP) on the
              class-balanced random subsample (fake_equal_real=True dataset built by the official class, seed 42);
              computed from the same per-file P with the same sklearn calls as `evaluate`.
and stores the per-file P(fake) so SynthScope's bench_eval scores can be compared file by file.
usage: python probe_official_eval.py {dino|resnet} OUT_JSON BATCH_SIZE NAME=REAL_DIR,FAKE_DIR ...
env: PROBE_REF (official Detector dir), PROBE_CKPTS (dir with DINOv2_best_model_step_34999.pth, ResNet50_best_model_step_19343.pth,
     dinov2-with-registers-large/), PROBE_WORKERS
"""
import json
import os
import random
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score
from torch.utils.data import DataLoader

REF = os.environ["PROBE_REF"]
CK = os.environ["PROBE_CKPTS"]
variant, out_path, bs = sys.argv[1], os.path.abspath(sys.argv[2]), int(sys.argv[3])
subsets = [a.split("=", 1) for a in sys.argv[4:]]
sys.path.insert(0, REF)
os.chdir(REF)
if variant == "dino":
    import evaluate_dino as E
    from model.dino_classifier import dino_classifier
    crop, pad_mode = 336, "repeat"
else:
    import evaluate_resnet as E
    from model.resnet50 import resnet50
    crop, pad_mode = 224, "pad"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
E.device = device
params = dict(blur_prob=0.0, blur_sig_min=0.0, blur_sig_max=3.0, blur_sig=0.0, jpeg_prob=0.0, jpeg_quality_min=60,
              jpeg_quality_max=100, jpeg_quality=100, cutout_prob=0.0, cutout_ratio_min=0.1, cutout_ratio_max=0.5,
              noise_prob=0.0, noise_std_min=0.0, noise_std_max=50.0, noise_std=0.0, resize_prob=0.0,
              resize_scale_min=0.5, resize_scale_max=2.0, resize_scale=1.0, pad_mode=pad_mode)

if variant == "dino":
    det = dino_classifier(model_name=CK + "/dinov2-with-registers-large", classifier_type="linear")
    ckpt = CK + "/DINOv2_best_model_step_34999.pth"
else:
    det = resnet50(num_classes=1, classifier_type="linear")
    ckpt = CK + "/ResNet50_best_model_step_19343.pth"
print(det.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state_dict"], strict=True), flush=True)
det = det.to(device).eval()


def seed_like_fresh_process():          # the module-level seeding of evaluate_*.py, as at the start of a new invocation
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)


def make_ds(real, fake, equal):
    kw = dict(root_list=[real, fake], crop_size=crop, fake_equal_real=equal, data_augment_params=params)
    if variant == "dino":
        kw["normalization"] = "imagenet"
    else:
        kw["model"] = "cnn"
    return E.EvaluateDataset(**kw)


def official_metrics(y_true, y_pred):    # the formulas of evaluate_*.py
    fm, rm = y_true == 1, y_true == 0
    return {"acc": accuracy_score(y_true, y_pred > 0.5) * 100,
            "fake_acc": accuracy_score(y_true[fm], y_pred[fm] > 0.5) * 100 if fm.any() else 0.0,
            "real_acc": accuracy_score(y_true[rm], y_pred[rm] > 0.5) * 100 if rm.any() else 0.0,
            "ap": average_precision_score(y_true, y_pred) * 100 if len(np.unique(y_true)) > 1 else 0.0}


res = {"variant": variant, "batch_size": bs, "ckpt": ckpt, "torch": torch.__version__, "device": str(device),
       "ref": REF, "subsets": {}}
for name, spec in subsets:
    real, fake = spec.split(",")
    seed_like_fresh_process()
    ds_off = make_ds(real, fake, True)                    # what the official script evaluates by default
    off_paths = set(ds_off.image_paths)
    ds_all = make_ds(real, fake, False)                   # every file
    E.dataset = ds_all
    dl = DataLoader(ds_all, batch_size=bs, shuffle=False, num_workers=int(os.environ.get("PROBE_WORKERS", 4)),
                    collate_fn=E.collate_fn)
    t0 = time.time()
    _, _, _, _, y_true, y_pred, _ = E.evaluate(det, dl)   # the official evaluate(), unchanged
    sec = time.time() - t0
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    full = official_metrics(y_true, y_pred)
    full["bacc"] = (full["fake_acc"] + full["real_acc"]) / 2
    sel = np.array([p in off_paths for p in ds_all.image_paths])
    off = official_metrics(y_true[sel], y_pred[sel])
    off["n_real"], off["n_fake"] = int((y_true[sel] == 0).sum()), int((y_true[sel] == 1).sum())
    off["bacc"] = (off["fake_acc"] + off["real_acc"]) / 2
    n_bad = sum(1 for p in y_pred if p == 0.5)
    res["subsets"][name] = {"n_real": int((y_true == 0).sum()), "n_fake": int((y_true == 1).sum()), "full": full,
                            "official": off, "sec": sec, "n_p_exactly_0.5": n_bad,
                            "paths": ds_all.image_paths, "labels": y_true.tolist(), "probs": y_pred.tolist()}
    print(f"\nOFFICIAL {variant} {name}: files {len(y_true)} ({sec:.0f}s) | full bAcc {full['bacc']:.2f} "
          f"(real {full['real_acc']:.2f} fake {full['fake_acc']:.2f} pooled {full['acc']:.2f} AP {full['ap']:.2f}) | "
          f"official-style subsample {off['n_real']}+{off['n_fake']}: Overall {off['acc']:.2f} "
          f"real {off['real_acc']:.2f} fake {off['fake_acc']:.2f} AP {off['ap']:.2f} bAcc {off['bacc']:.2f}", flush=True)
    json.dump(res, open(out_path, "w"))
names = list(res["subsets"])
mean = lambda key, k2: sum(res["subsets"][n][key][k2] for n in names) / len(names)
print(f"OFFICIAL {variant} MEAN over {len(names)} subsets: full bAcc {mean('full', 'bacc'):.2f} | official-style Overall "
      f"{mean('official', 'acc'):.2f} (bAcc {mean('official', 'bacc'):.2f}) -> {out_path}", flush=True)
