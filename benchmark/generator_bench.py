"""Generator benchmark: how well do SynthScope's image detectors catch images from ONE image generator?

For every real image a text prompt is written (BLIP caption + a photographic style suffix), the generator creates a
new image from that prompt ALONE, and every available image detector scores the real and the generated images. The
result is a ranking by AUROC, shown in the web page's "Generator benchmarks" tab.

    # same 100 real images + prompts as the reference run, so generators are compared on identical content
    python benchmark/generator_bench.py --generator flux2_klein --label "FLUX.2 [klein] 4B" \\
        --reuse benchmark/generator_runs/reference/manifest.json --out benchmark/generator_runs/flux2_klein_4b

    # build a fresh image set from your own real images
    python benchmark/generator_bench.py --generator nanobanana --real-glob "/data/real/*.jpg" --n 100 \\
        --out benchmark/generator_runs/nanobanana

``--generator`` is any key of ``interface/backend/generators.py`` that can create an image from text. Local open-weight
generators (FLUX.2 [klein], Qwen-Image-2.1) run in batch with one model load; hosted ones need their API key (``.env``).
Outputs in ``--out``: manifest.json, results.json, RESULTS.md (committed) and real/ + fake/ images (git-ignored).
Re-running skips images that already exist, so an interrupted run can be resumed. See docs/BENCHMARKS.md.
"""
import argparse
import glob
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from interface.backend import generators as G
from interface.backend.registry import list_specs

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")


def log(*a):
    print(*a, flush=True)


def build_pairs(args, real_dir: Path) -> list:
    """[{idx, caption, prompt, real_src}] and the real images copied to ``real_dir`` as 000.png, 001.png, ..."""
    pairs = []
    if args.reuse:
        src = json.load(open(args.reuse))
        src_real = Path(args.reuse).parent / "real"
        for p in src["pairs"][: args.n]:
            png = src_real / f"{p['idx']:03d}.png"
            if png.is_file():
                shutil.copyfile(png, real_dir / f"{p['idx']:03d}.png")
                pairs.append({"idx": p["idx"], "caption": p.get("caption", ""), "prompt": p["prompt"],
                              "real_src": p.get("real_src", "")})
        log(f"[pairs] reusing {len(pairs)} prompts + real images from {args.reuse}")
        return pairs
    if not args.real_glob:
        sys.exit("give --reuse MANIFEST or --real-glob PATTERN (real images to caption)")
    cands = sorted({f for pat in args.real_glob for f in glob.glob(pat) if f.lower().endswith(IMG_EXT)})
    random.Random(args.seed).shuffle(cands)
    log(f"[pairs] {len(cands)} candidate real images; need {args.n} (people are skipped)")
    for src in cands:
        if len(pairs) >= args.n:
            break
        data = open(src, "rb").read()
        cap = G._caption(data)
        if not cap or G._PERSON_RE.search(cap):          # no photos of people in the benchmark
            continue
        i = len(pairs)
        Image.open(src).convert("RGB").save(real_dir / f"{i:03d}.png")
        pairs.append({"idx": i, "caption": cap, "prompt": cap + G.STYLE_SUFFIX, "real_src": src})
    return pairs


def generate(args, pairs: list, real_dir: Path, fake_dir: Path) -> list:
    g = G.GENERATORS[args.generator]
    if "recreate" not in g.modes:
        sys.exit(f"{g.name} cannot create an image from text (modes: {g.modes})")
    if not g.available():
        sys.exit(f"{g.name} is not available (API key / local setup missing, see docs/BENCHMARKS.md)")
    model_key = getattr(g.call, "model_key", None)
    t0 = time.time()
    if model_key:                                          # local open-weight model: one batch, one model load
        items = [{"prompt": p["prompt"], "out": str(fake_dir / f"{p['idx']:03d}.png"), "seed": p["idx"],
                  "size": f"{args.res}x{args.res}"} for p in pairs]
        G.local_batch(model_key, items)
    else:                                                  # hosted generator: one API call per image
        for p in pairs:
            out = fake_dir / f"{p['idx']:03d}.png"
            if out.exists():
                continue
            try:
                png = G.generate(args.generator, open(real_dir / f"{p['idx']:03d}.png", "rb").read(), p["prompt"], "recreate")
                out.write_bytes(png)
                log(f"  [{p['idx']}] {p['caption']!r}")
            except Exception as e:                         # blocked by a safety filter, rate limit, ...
                log(f"  [{p['idx']}] skipped: {type(e).__name__}: {e}")
    kept = [p for p in pairs if (fake_dir / f"{p['idx']:03d}.png").is_file()]
    for p in pairs:                                        # keep real and fake balanced
        if p not in kept and (real_dir / f"{p['idx']:03d}.png").exists():
            (real_dir / f"{p['idx']:03d}.png").unlink()
    log(f"[generate] {len(kept)}/{len(pairs)} fakes ready ({time.time() - t0:.0f}s)")
    return kept


def score(args, real_dir: Path, fake_dir: Path, out: Path) -> list:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from interface.backend.engine import get_engine
    engine = get_engine()
    reals = [Image.open(p).convert("RGB") for p in sorted(glob.glob(str(real_dir / "*.png")))]
    fakes = [Image.open(p).convert("RGB") for p in sorted(glob.glob(str(fake_dir / "*.png")))]
    log(f"[score] {len(reals)} real + {len(fakes)} fake")
    y = np.array([0] * len(reals) + [1] * len(fakes))
    keys = args.detectors.split(",") if args.detectors else None
    rows = []
    for spec in list_specs():
        if spec.modality != "image" or (keys and spec.key not in keys):
            continue
        if not engine.is_available(spec):
            log(f"  {spec.key}: no checkpoint here, skipped")
            continue
        try:
            t0 = time.time()
            probs = np.array([engine.predict_image(spec, im)["fake_probability"] for im in reals + fakes], dtype="float64")
            rows.append({
                "detector": spec.key, "name": spec.name,
                "auroc": round(float(roc_auc_score(y, probs)) * 100, 2),
                "ap": round(float(average_precision_score(y, probs)) * 100, 2),
                "acc": round(float(((probs >= 0.5).astype(int) == y).mean()) * 100, 2),
                "fake_detect_rate": round(float((probs[len(reals):] >= 0.5).mean()) * 100, 2),
                "real_false_pos": round(float((probs[: len(reals)] >= 0.5).mean()) * 100, 2),
                "mean_p_fake_on_fakes": round(float(probs[len(reals):].mean()), 4),
                "mean_p_fake_on_reals": round(float(probs[: len(reals)].mean()), 4)})
            r = rows[-1]
            log(f"  {spec.key:12} AUROC={r['auroc']:6.2f} AP={r['ap']:6.2f} acc={r['acc']:6.2f}  ({time.time() - t0:.0f}s)")
        except Exception as e:
            log(f"  {spec.key}: ERROR {type(e).__name__}: {e}")
    rows.sort(key=lambda r: r["auroc"], reverse=True)
    json.dump(rows, open(out / "results.json", "w"), indent=2)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generator", required=True, help="key from interface/backend/generators.py, e.g. flux2_klein")
    ap.add_argument("--label", help="name shown in the web page (default: the generator's name)")
    ap.add_argument("--out", required=True, help="output directory, e.g. benchmark/generator_runs/flux2_klein_4b")
    ap.add_argument("--reuse", help="manifest.json of an earlier run: reuse its real images + prompts (same content for every generator)")
    ap.add_argument("--real-glob", action="append", help="glob of real images to caption (repeatable); needed without --reuse")
    ap.add_argument("--n", type=int, default=100, help="number of real/fake pairs")
    ap.add_argument("--res", type=int, default=1024, help="output side in pixels for local generators")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--detectors", help="comma-separated detector keys (default: every available image detector)")
    ap.add_argument("--skip-generate", action="store_true", help="only (re)score the images already in --out")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    real_dir, fake_dir = out / "real", out / "fake"
    real_dir.mkdir(parents=True, exist_ok=True); fake_dir.mkdir(parents=True, exist_ok=True)
    g = G.GENERATORS[args.generator]
    manifest_path = out / "manifest.json"
    if manifest_path.exists():                              # resume
        pairs = json.load(open(manifest_path))["pairs"]
        for p in pairs:                                     # real images may have been cleaned up after a partial run
            if not (real_dir / f"{p['idx']:03d}.png").exists() and p.get("real_src") and args.reuse:
                shutil.copyfile(Path(args.reuse).parent / "real" / f"{p['idx']:03d}.png", real_dir / f"{p['idx']:03d}.png")
    else:
        pairs = build_pairs(args, real_dir)
    if not args.skip_generate:
        pairs = generate(args, pairs, real_dir, fake_dir)
    rows = score(args, real_dir, fake_dir, out)
    label = args.label or g.name
    manifest = {"label": label, "generator": args.generator, "n": len(pairs), "res": args.res,
                "settings": f"{args.res}px · text-to-image from a caption", "created": time.strftime("%Y-%m-%d"),
                "reuse_from": args.reuse or "", "pairs": pairs}
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    md = [f"# Detectors vs {label}", "",
          f"{len(pairs)} real images (people excluded) + {len(pairs)} images generated by {label} from a BLIP caption of each "
          f"real image ({args.res}px). Ranked by AUROC; accuracy uses the 0.5 threshold.", "",
          "| Rank | Detector | AUROC | AP | acc@0.5 | fake detect % | real FP % |", "|---|---|---|---|---|---|---|"]
    md += [f"| {i} | {r['name']} | {r['auroc']} | {r['ap']} | {r['acc']} | {r['fake_detect_rate']} | {r['real_false_pos']} |"
           for i, r in enumerate(rows, 1)]
    (out / "RESULTS.md").write_text("\n".join(md) + "\n")
    log("\n".join(md))


if __name__ == "__main__":
    main()
