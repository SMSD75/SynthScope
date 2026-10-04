"""Build a small 'can today's detectors catch Ideogram 4.0?' benchmark and rank our detectors on it.

Phase 1 — dataset: sample real images, BLIP-caption each (the demo's "conceptually similar" prompt),
skip people (Ideogram refuses photorealistic humans) and any safety-blocked output, generate with the
local Ideogram nf4 pipeline until N real + N Ideogram-fake images are collected.
Phase 2 — score: run every available SynthScope image detector (promoted checkpoints + the web UI's exact
preprocessing) over all 2N images and rank by AUROC / AP / accuracy@0.5.

Run with CUDA_VISIBLE_DEVICES=7. Outputs under repro/ideogram_bench/.
"""
import os, sys, io, glob, json, random, time
sys.path.insert(0, os.path.abspath("/ivi/zfs/s0/original_homes/ssalehi/SynthScope"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

from interface.backend import config, generators as G
from PIL import Image
import numpy as np

N = int(os.environ.get("IDEO_N", "100"))
STEPS = int(os.environ.get("IDEO_STEPS", "28"))
RES = int(os.environ.get("IDEO_RES", "512"))
LABEL = os.environ.get("IDEO_LABEL", f"Ideogram 4.0 ({RES}px)")
OUT = os.environ.get("IDEO_OUT", "/ivi/zfs/s0/original_homes/ssalehi/SynthScope/repro/ideogram_bench")
# Optional: reuse the prompts + real images from a prior run (isolates a single variable, e.g. resolution).
REUSE_FROM = os.environ.get("IDEO_REUSE_FROM", "")
REAL_DIR, FAKE_DIR = os.path.join(OUT, "real"), os.path.join(OUT, "fake")
for d in (REAL_DIR, FAKE_DIR):
    os.makedirs(d, exist_ok=True)
REAL_POOLS = ["/hddstore/ssalehi/SynthScope/GenImage/test/real/*",
              "/hddstore/ssalehi/SynthScope/CNNDetection/test/*/0_real/*",
              "/hddstore/ssalehi/SynthScope/GenImage_full/test_full/*/0_real/*"]


def log(*a):
    print(*a, flush=True)


def build_from_reuse(manifest_path):
    """Regenerate fakes at the current RES/STEPS from a prior run's prompts + real images (single-variable
    comparison, e.g. resolution). Same concepts/reals; only the fake generator settings change."""
    import shutil, torch
    from diffusers import Ideogram4Pipeline
    src = json.load(open(REUSE_FROM))
    src_dir = os.path.dirname(REUSE_FROM)
    pairs_in = src["pairs"][:N]
    log(f"[dataset] REUSE {len(pairs_in)} prompts from {REUSE_FROM} at {RES}px/{STEPS} steps")
    pipe = Ideogram4Pipeline.from_pretrained(
        "ideogram-ai/ideogram-4-nf4-diffusers", torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN")).to("cuda")
    pairs = []
    for p in pairs_in:
        i, prompt = p["idx"], p["prompt"]
        src_real = os.path.join(src_dir, "real", f"{i:03d}.png")
        if not os.path.isfile(src_real):
            log(f"  [{i}] missing source real, skip"); continue
        if os.path.isfile(os.path.join(FAKE_DIR, f"{i:03d}.png")):     # resume: already generated
            import shutil as _sh
            _sh.copyfile(src_real, os.path.join(REAL_DIR, f"{i:03d}.png"))
            pairs.append({"idx": i, "real_src": p.get("real_src"), "caption": p.get("caption"), "prompt": prompt})
            continue
        img, ok = None, False
        for attempt in range(3):                 # the block has some randomness; retry a couple times
            t0 = time.time()
            img = pipe(prompt=prompt, height=RES, width=RES, num_inference_steps=STEPS,
                       guidance_scale=7.0, guidance_schedule=None).images[0]
            if not G._looks_blocked(img):
                ok = True; break
            log(f"  [{len(pairs)+1}/{len(pairs_in)}] blocked (retry {attempt+1})")
        if not ok:
            log(f"  [{i}] blocked after retries, skip"); continue
        shutil.copyfile(src_real, os.path.join(REAL_DIR, f"{i:03d}.png"))
        img.convert("RGB").save(os.path.join(FAKE_DIR, f"{i:03d}.png"))
        pairs.append({"idx": i, "real_src": p.get("real_src"), "caption": p.get("caption"), "prompt": prompt})
        log(f"  [{len(pairs)}/{len(pairs_in)}] {time.time()-t0:.0f}s  {p.get('caption','')!r}")
    del pipe; torch.cuda.empty_cache()
    man = {"n": len(pairs), "steps": STEPS, "res": RES, "label": LABEL, "reuse_from": REUSE_FROM, "pairs": pairs}
    json.dump(man, open(manifest_path, "w"), indent=2)
    log(f"[dataset] reuse built {len(pairs)} pairs")
    return man


def build_dataset():
    manifest_path = os.path.join(OUT, "manifest.json")
    if os.path.isfile(manifest_path):
        man = json.load(open(manifest_path))
        if len(man.get("pairs", [])) >= N:
            log(f"[dataset] reusing existing {len(man['pairs'])} pairs")
            return man
    if REUSE_FROM:
        return build_from_reuse(manifest_path)
    cands = []
    for pat in REAL_POOLS:
        cands += [f for f in glob.glob(pat) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    random.seed(0); random.shuffle(cands)
    log(f"[dataset] {len(cands)} candidate real images; need {N} non-person, non-blocked")

    import torch
    from diffusers import Ideogram4Pipeline
    pipe = Ideogram4Pipeline.from_pretrained(
        "ideogram-ai/ideogram-4-nf4-diffusers", torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN")).to("cuda")

    pairs, attempts = [], 0
    for src in cands:
        if len(pairs) >= N:
            break
        attempts += 1
        try:
            real_bytes = open(src, "rb").read()
            cap = G._caption(real_bytes)
            if not cap or G._PERSON_RE.search(cap):
                continue  # Ideogram blocks people
            prompt = f"{cap}, photorealistic, high detail"
            t0 = time.time()
            img = pipe(prompt=prompt, height=RES, width=RES, num_inference_steps=STEPS,
                       guidance_scale=7.0, guidance_schedule=None).images[0]
            if G._looks_blocked(img):
                log(f"  [{len(pairs)}/{N}] BLOCKED (skipped): {cap!r}")
                continue
            i = len(pairs)
            Image.open(io.BytesIO(real_bytes)).convert("RGB").save(os.path.join(REAL_DIR, f"{i:03d}.png"))
            img.convert("RGB").save(os.path.join(FAKE_DIR, f"{i:03d}.png"))
            pairs.append({"idx": i, "real_src": src, "caption": cap, "prompt": prompt})
            log(f"  [{len(pairs)}/{N}] {time.time()-t0:.0f}s  {cap!r}")
        except Exception as e:
            log(f"  attempt failed: {type(e).__name__}: {e}")
    del pipe
    torch.cuda.empty_cache()
    man = {"n": len(pairs), "steps": STEPS, "res": RES, "label": LABEL, "attempts": attempts, "pairs": pairs}
    json.dump(man, open(manifest_path, "w"), indent=2)
    log(f"[dataset] built {len(pairs)} pairs in {attempts} attempts")
    return man


def score():
    from interface.backend.engine import get_engine
    from interface.backend.registry import get_spec
    engine = get_engine()
    reals = sorted(glob.glob(os.path.join(REAL_DIR, "*.png")))
    fakes = sorted(glob.glob(os.path.join(FAKE_DIR, "*.png")))
    real_imgs = [Image.open(p).convert("RGB") for p in reals]
    fake_imgs = [Image.open(p).convert("RGB") for p in fakes]
    log(f"[score] {len(real_imgs)} real + {len(fake_imgs)} fake")
    from sklearn.metrics import roc_auc_score, average_precision_score

    DETECTORS = ["clip", "cnn", "fat_former", "npr", "aide", "safe", "co_spy", "spai", "vib_net", "pgc"]
    y = np.array([0] * len(real_imgs) + [1] * len(fake_imgs))
    rows = []
    for key in DETECTORS:
        try:
            spec = get_spec(key)
            if spec.modality != "image" or not engine.is_available(spec):
                log(f"  {key}: unavailable, skip"); continue
            t0 = time.time()
            probs = [engine.predict_image(spec, im)["fake_probability"] for im in real_imgs + fake_imgs]
            probs = np.array(probs, dtype="float64")
            auroc = float(roc_auc_score(y, probs)) * 100
            ap = float(average_precision_score(y, probs)) * 100
            acc = float(((probs >= 0.5).astype(int) == y).mean()) * 100
            det_rate = float((probs[len(real_imgs):] >= 0.5).mean()) * 100   # fakes flagged fake
            fpr = float((probs[:len(real_imgs)] >= 0.5).mean()) * 100        # reals flagged fake
            rows.append({"detector": key, "name": spec.name, "auroc": round(auroc, 2), "ap": round(ap, 2),
                         "acc": round(acc, 2), "fake_detect_rate": round(det_rate, 2),
                         "real_false_pos": round(fpr, 2),
                         "mean_p_fake_on_fakes": round(float(probs[len(real_imgs):].mean()), 4),
                         "mean_p_fake_on_reals": round(float(probs[:len(real_imgs)].mean()), 4)})
            log(f"  {key:11} AUROC={auroc:6.2f} AP={ap:6.2f} acc={acc:6.2f} "
                f"detect={det_rate:5.1f}% FPR={fpr:5.1f}%  ({time.time()-t0:.0f}s)")
        except Exception as e:
            log(f"  {key}: ERROR {type(e).__name__}: {e}")
    rows.sort(key=lambda r: r["auroc"], reverse=True)
    json.dump(rows, open(os.path.join(OUT, "results.json"), "w"), indent=2)

    md = ["# Detectors vs Ideogram 4.0 — ranked", "",
          f"Dataset: {len(real_imgs)} real (sampled from GenImage/CNNDetection reals) + {len(fake_imgs)} "
          f"Ideogram-generated fakes (BLIP-caption prompts, {RES}px, {STEPS} steps; people excluded — "
          "Ideogram refuses them). Detectors use the promoted SynthScope checkpoints + the web UI's preprocessing.", "",
          "| Rank | Detector | AUROC | AP | acc@0.5 | fake detect % | real FP % |", "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        md.append(f"| {i} | {r['name']} | {r['auroc']} | {r['ap']} | {r['acc']} | "
                  f"{r['fake_detect_rate']} | {r['real_false_pos']} |")
    open(os.path.join(OUT, "RESULTS.md"), "w").write("\n".join(md) + "\n")
    log("[score] wrote results.json + RESULTS.md")
    log("\n" + "\n".join(md))


if __name__ == "__main__":
    t = time.time()
    build_dataset()
    score()
    log(f"[done] total {time.time()-t:.0f}s")
