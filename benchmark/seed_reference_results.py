"""Build SynthScope's own leaderboard submissions from the logged evaluations in ``repro/bench_results``.

Each entry below names one committed ``repro/bench_results/*.json`` (written by a logged ``bench_eval.py`` /
``bench_eval_video.py`` run), the generators its track scores, and the job that wrote it. The metrics are read
from that file: the run's own means when the track scores every generator in it, otherwise the unweighted mean
of its per-generator values over the track's generators. No number is typed in by hand, so every SynthScope
row of LEADERBOARD.md can be recomputed from the repo:

    python benchmark/seed_reference_results.py           # rewrite submissions/<slug>.json for these entries
    python benchmark/seed_reference_results.py --check   # fail if a submission no longer matches its evidence
    python benchmark/leaderboard.py                      # then refresh LEADERBOARD.md

Entries with ``verified=False`` have a logged evaluation whose checkpoint cannot be confirmed to fit the track;
their notes say why. Submissions from other people are not touched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
SUB = BENCH_DIR / "submissions"
RESULTS = BENCH_DIR.parent / "repro" / "bench_results"
REPO = "https://github.com/SMSD75/SynthScope"
METRIC_KEYS = ["acc", "map", "ap", "auroc"]

# The ForenSynths track scores the 8 GAN test sets (ForenSynths-8). Runs over all 13 sets are subset to these.
FS8 = ["biggan", "cyclegan", "deepfake", "gaugan", "progan", "stargan", "stylegan", "stylegan2"]
# The GenImage SD-v1.4 track scores the SD-v1.4 subset of the GenImage test set.
SD14 = ["sdv1.4"]
JUNE_LOG = "job log not kept on the shared filesystem; MASTER_RESULTS.jsonl row of the same tag"

# slug -> (method, track, model_type, paper, evaluation, bench_results file, generators (None = all in the
#          file), job, verified, protocol_notes)
ENTRIES = {
    # --- ForenSynths-8 (image) ---
    "fat_former": (
        "FatFormer", "forensynths", "fat_former", "Liu et al., CVPR'24", "official-weights",
        "fatformer_official_forensynths.json", None, JUNE_LOG, True,
        "ForenSynths-8; official 4-class ProGAN weights (the served fat_former.pth)"),
    "co_spy_official": (
        "CO-SPY (official weights)", "forensynths", "co_spy", "Cheng et al., CVPR'25", "official-weights",
        "cospy_official_forensynths.json", FS8, JUNE_LOG, True,
        "ForenSynths-8; official CO-SPY ProGAN checkpoint in the vendored official architecture "
        "(not the served co_spy.pth)"),
    "npr": (
        "NPR", "forensynths", "npr", "Tan et al., CVPR'24", "official-weights",
        "npr_official_nprpth_forensynths8.json", None, "job 265808 (repro/logs/nprevt_265808.out)", True,
        "ForenSynths-8; official ProGAN weights NPR.pth (the served npr.pth)"),
    "aide": (
        "AIDE", "forensynths", "aide", "Yan et al., ICLR'25", "official-weights",
        "aide_official_forensynths.json", FS8, "repro/eval_logs/fixup_aide_gpu7_bs8.log (2026-06-05 rerun)", True,
        "ForenSynths-8; official ProGAN weights progan_train.pth, vendored official eval (native-resolution "
        "5-view DCT); the served aide.pth is the SD-v1.4 release"),
    "clip": (
        "UnivFD / CLIP", "forensynths", "clip", "Ojha et al., CVPR'23", "frozen-probe",
        "clip_synthscope_forensynths.json", FS8, JUNE_LOG, True,
        "ForenSynths-8; frozen CLIP ViT-L/14 + linear head trained in SynthScope on ProGAN (the served clip.pth)"),
    "cnn": (
        "CNNDetection", "forensynths", "cnn", "Wang et al., CVPR'20", "from-scratch",
        "cnn_synthscope_forensynths.json", FS8, JUNE_LOG, True,
        "ForenSynths-8; ImageNet ResNet-50 fine-tuned in SynthScope on ProGAN (the served cnn.pth)"),
    "dinov2": (
        "DINOv2 probe", "forensynths", "dinov2", "Oquab et al., 2023", "frozen-probe",
        "dinov2_synthscope_forensynths.json", FS8, JUNE_LOG, True,
        "ForenSynths-8; frozen DINOv2 ViT-L/14 + linear head trained in SynthScope on ProGAN (the served dinov2.pth)"),
    "safe": (
        "SAFE", "forensynths", "safe", "Li et al., KDD'25", "from-scratch",
        "safe_v2_forensynths8.json", None, "job 186235 (repro/logs/safe_v2_fs8_186235.out)", True,
        "ForenSynths-8; trained in SynthScope on ProGAN, 40-epoch crop recipe (safe_full_v2/model_38, "
        "the served safe.pth)"),
    "co_spy_scratch": (
        "CO-SPY (from scratch)", "forensynths", "co_spy", "Cheng et al., CVPR'25", "from-scratch",
        "cospy_synthscope_forensynths.json", FS8, JUNE_LOG, False,
        "Unverified: ForenSynths-8 score of the co_spy.pth of 2026-06-04, then labelled ProGAN-trained, whose "
        "training run is not recorded; not the served co_spy.pth (a GenImage SD-v1.4 retrain, not scored here)"),
    # --- GenImage SD-v1.4 (image) ---
    "vib_net": (
        "VIB-Net", "genimage-sdv1.4", "vib_net", "Zhang et al., CVPR'25", "from-scratch",
        "vib_net_trained_noresize_genimage7.json", SD14, "eval job 267928 (repro/logs/vibevt_267928.out)", True,
        "GenImage SD-v1.4, 6000 real + 6000 fake; trained in SynthScope on GenImage SD-v1.4 with the "
        "reconstructed recipe (job 267846, the served vib_net.pth); one generator, so mAP = AP"),
    "spai_official": (
        "SPAI (official weights)", "genimage-sdv1.4", "spai", "Karageorgiou et al., CVPR'25", "official-weights",
        "spai_official_genimage8.json", SD14, "job 273444 (repro/logs/spaiev_official_273444.out)", True,
        "GenImage SD-v1.4, 300 real + 300 fake (the run's per-generator cap); official weights remapped into "
        "the SynthScope class, native resolution (the served spai.pth)"),
    "spai_scratch": (
        "SPAI (from scratch)", "genimage-sdv1.4", "spai", "Karageorgiou et al., CVPR'25", "from-scratch",
        "spai_ep8_genimage8.json", SD14, "job 273445 (repro/logs/spaiev_ep8_273445.out)", True,
        "GenImage SD-v1.4, 300 real + 300 fake; trained in SynthScope with the official recipe (LDM vs "
        "COCO/LSUN, job 267192, epoch 8 by validation loss), native resolution"),
    # --- GenVideo-Val (video) ---
    "d3": (
        "D3", "genvideo-val", "d3", "Zheng et al., ICCV'25", "training-free",
        "d3_genvideo_fpr.json", None, "job 270384 (repro/logs/spliteval_270384.out)", True,
        "Training-free X-CLIP 2nd-order feature volatility (official preprocessing)"),
    "split": (
        "SPLIT", "genvideo-val", "split", "Hyun & Kim, ECCV'26", "training-free",
        "split_genvideo_fpr.json", None, "job 270384 (repro/logs/spliteval_270384.out)", True,
        "Training-free X-CLIP patch roughness x motion incoherence; Acc at the real-only FPR-1% calibration"),
    "stall": (
        "STALL", "genvideo-val", "stall", "Ben Hayun et al., CVPR'26", "training-free",
        "stall_genvideo_full.json", None, "job 271899 (repro/logs/stallgv_271899.out)", True,
        "Training-free DINOv3 spatio-temporal likelihood under VATEX real-video statistics"),
    "restrav": (
        "ReStraV", "genvideo-val", "restrav", "Interno et al., 2025", "frozen-probe",
        "restrav_genvideo_full.json", None, "job 271898 (repro/logs/restravgv_271898.out)", True,
        "Frozen DINOv2 trajectory geometry + MLP trained in SynthScope (the served restrav.pth; its training "
        "set is not recorded); does not transfer to GenVideo-Val (AUROC below 50)"),
}


def _mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 2)


def build(slug: str) -> dict:
    """Return the submission for one entry, with metrics read from its bench_results file."""
    method, track, model_type, paper, evaluation, fname, gens, job, verified, notes = ENTRIES[slug]
    res = json.loads((RESULTS / fname).read_text())
    pg = res["per_generator"]
    video = track == "genvideo-val"
    if gens is None:
        gens = sorted(pg)
        means = res["means"]
        metrics = {"acc": means["acc"], "map": means["map"], "auroc": means["mauroc"]}
    else:
        missing = [g for g in gens if g not in pg]
        if missing:
            raise KeyError(f"{fname} has no per-generator result for {missing}")
        metrics = {"acc": _mean([pg[g]["acc"] for g in gens]), "map": _mean([pg[g]["ap"] for g in gens]),
                   "auroc": _mean([pg[g]["auroc"] for g in gens])}
    keep = ("auroc", "ap") if video else ("ap", "auroc", "acc")
    scope = "all generators" if len(gens) == len(pg) else f"{len(gens)} of its {len(pg)} generators"
    return {
        "method": method, "track": track, "modality": "video" if video else "image", "model_type": model_type,
        "paper": paper, "url": REPO, "evaluation": evaluation,
        "metrics": {k: metrics.get(k) for k in METRIC_KEYS},
        "per_generator": {g: {k: pg[g][k] for k in keep} for g in gens},
        "submitted_by": "SynthScope", "date": res["date"][:10], "verified": verified,
        "source": f"repro/bench_results/{fname} (tag {res['tag']}, {scope}); {job}",
        "protocol_notes": notes, "synthscope_version": "0.1.0",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Rebuild SynthScope's leaderboard submissions from repro/bench_results.")
    ap.add_argument("--check", action="store_true", help="fail if a submission differs from its evidence")
    args = ap.parse_args(argv)
    stale = []
    for slug in ENTRIES:
        text = json.dumps(build(slug), indent=2) + "\n"
        path = SUB / f"{slug}.json"
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(path.name)
        else:
            path.write_text(text)
    if args.check:
        if stale:
            print("Out of date with repro/bench_results: " + ", ".join(stale), file=sys.stderr)
            return 1
        print(f"OK: {len(ENTRIES)} SynthScope submissions match their bench_results evidence.")
        return 0
    print(f"Wrote {len(ENTRIES)} SynthScope submissions to {SUB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
