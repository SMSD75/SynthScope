"""Build SynthScope's own leaderboard submissions from the logged evaluations in ``repro/bench_results``.

Each entry below names one committed ``repro/bench_results/*.json`` (written by a logged ``bench_eval.py`` /
``bench_eval_video.py`` run), the generators its track scores, and the job that wrote it. The metrics are read
from that file: the run's own means when the track scores every generator in it, otherwise the unweighted mean
of its per-generator values over the track's generators. No number is typed in by hand, so every SynthScope
row of LEADERBOARD.md can be recomputed from the repo:

    python benchmark/seed_reference_results.py           # rewrite submissions/<slug>.json for these entries
    python benchmark/seed_reference_results.py --check   # fail if a submission no longer matches its evidence,
                                                         # or a SynthScope submission has no entry here
    python benchmark/leaderboard.py                      # then refresh LEADERBOARD.md

Entries with ``verified=False`` have a logged evaluation whose checkpoint cannot be confirmed to fit the track;
their notes say why. Submissions from other people are not touched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

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



class Entry(NamedTuple):
    method: str
    track: str
    model_type: str
    paper: str
    evaluation: str
    results_file: str              # repro/bench_results/<file>
    generators: list[str] | None   # None = every generator in the file
    job: str
    verified: bool
    notes: str


# slug (= submissions/<slug>.json) -> its evidence
ENTRIES = {
    # --- ForenSynths-8 (image) ---
    "fat_former": Entry(
        "FatFormer", "forensynths", "fat_former", "Liu et al., CVPR'24", "official-weights",
        "fatformer_official_forensynths.json", None, JUNE_LOG, True,
        "ForenSynths-8; official 4-class ProGAN weights (the served fat_former.pth)"),
    "co_spy_official": Entry(
        "CO-SPY (official weights)", "forensynths", "co_spy", "Cheng et al., CVPR'25", "official-weights",
        "cospy_official_fs8_nov.json", None, "DAS6 job 66671 (repro/logs/lbfu_cospyoff_das6_66671.out)", True,
        "ForenSynths-8, full test sets; official CO-SPY ProGAN checkpoint in the vendored official architecture "
        "(not the served co_spy.pth)"),
    "npr": Entry(
        "NPR", "forensynths", "npr", "Tan et al., CVPR'24", "official-weights",
        "npr_official_nprpth_forensynths8.json", None, "job 265808 (repro/logs/nprevt_265808.out)", True,
        "ForenSynths-8; official ProGAN weights NPR.pth (the served npr.pth)"),
    "aide": Entry(
        "AIDE", "forensynths", "aide", "Yan et al., ICLR'25", "official-weights",
        "aide_official_forensynths.json", FS8, "repro/eval_logs/fixup_aide_gpu7_bs8.log (2026-06-05 rerun)", True,
        "ForenSynths-8; official ProGAN weights progan_train.pth, vendored official eval (native-resolution "
        "5-view DCT); the served aide.pth is the SD-v1.4 release"),
    "clip": Entry(
        "UnivFD / CLIP", "forensynths", "clip", "Ojha et al., CVPR'23", "frozen-probe",
        "clip_synthscope_forensynths.json", FS8, JUNE_LOG, True,
        "ForenSynths-8; frozen CLIP ViT-L/14 + linear head trained in SynthScope on ProGAN (the served clip.pth)"),
    "cnn": Entry(
        "CNNDetection", "forensynths", "cnn", "Wang et al., CVPR'20", "from-scratch",
        "cnn_synthscope_forensynths.json", FS8, JUNE_LOG, True,
        "ForenSynths-8; ImageNet ResNet-50 fine-tuned in SynthScope on ProGAN (the served cnn.pth)"),
    "dinov2": Entry(
        "DINOv2 probe", "forensynths", "dinov2", "Oquab et al., 2023", "frozen-probe",
        "dinov2_synthscope_forensynths.json", FS8, JUNE_LOG, True,
        "ForenSynths-8; frozen DINOv2 ViT-L/14 + linear head trained in SynthScope on ProGAN (the served dinov2.pth)"),
    "safe": Entry(
        "SAFE", "forensynths", "safe", "Li et al., KDD'25", "from-scratch",
        "safe_v2_forensynths8.json", None, "job 186235 (repro/logs/safe_v2_fs8_186235.out)", True,
        "ForenSynths-8; trained in SynthScope on ProGAN, 40-epoch crop recipe (safe_full_v2/model_38, "
        "the served safe.pth)"),
    "co_spy_scratch": Entry(
        "CO-SPY (from scratch)", "forensynths", "co_spy", "Cheng et al., CVPR'25", "from-scratch",
        "cospy_synthscope_forensynths.json", FS8, JUNE_LOG, False,
        "Unverified: ForenSynths-8 score of the co_spy.pth of 2026-06-04, then labelled ProGAN-trained, whose "
        "training run is not recorded; not the served co_spy.pth (a GenImage SD-v1.4 retrain, not scored here)"),
    # --- GenImage SD-v1.4 (image) ---
    "vib_net": Entry(
        "VIB-Net", "genimage-sdv1.4", "vib_net", "Zhang et al., CVPR'25", "from-scratch",
        "vib_net_trained_noresize_genimage7.json", SD14, "eval job 267928 (repro/logs/vibevt_267928.out)", True,
        "GenImage SD-v1.4, 6000 real + 6000 fake; trained in SynthScope on GenImage SD-v1.4 with the "
        "reconstructed recipe (job 267846, the served vib_net.pth); one generator, so mAP = AP"),
    "spai_official": Entry(
        "SPAI (official weights)", "genimage-sdv1.4", "spai", "Karageorgiou et al., CVPR'25", "official-weights",
        "spai_official_genimage8.json", SD14, "job 273444 (repro/logs/spaiev_official_273444.out)", True,
        "300+300 subsample; trained on latent-diffusion (LDM) data, not SD-v1.4. GenImage SD-v1.4 (the run's "
        "per-generator cap); official weights remapped into the SynthScope class, native resolution (the served spai.pth)"),
    "spai_scratch": Entry(
        "SPAI (from scratch)", "genimage-sdv1.4", "spai", "Karageorgiou et al., CVPR'25", "from-scratch",
        "spai_ep8_genimage8.json", SD14, "job 273445 (repro/logs/spaiev_ep8_273445.out)", True,
        "300+300 subsample; trained on latent-diffusion (LDM) data, not SD-v1.4. GenImage SD-v1.4; trained in "
        "SynthScope with the official recipe (LDM vs COCO/LSUN, job 267192, epoch 8 by validation loss), native resolution"),
    # --- GenVideo-Val (video) ---
    "d3": Entry(
        "D3", "genvideo-val", "d3", "Zheng et al., ICCV'25", "training-free",
        "d3_genvideo_fpr.json", None, "job 270384 (repro/logs/spliteval_270384.out)", True,
        "Training-free X-CLIP 2nd-order feature volatility (official preprocessing). AP here is fake-as-positive "
        "against all 10k reals; the README's 99.0 AP is the paper protocol (real-as-positive, first 1000 clips)"),
    "split": Entry(
        "SPLIT", "genvideo-val", "split", "Hyun & Kim, ECCV'26", "training-free",
        "split_genvideo_fpr.json", None, "job 270384 (repro/logs/spliteval_270384.out)", True,
        "Training-free X-CLIP patch roughness x motion incoherence; Acc at the real-only FPR-1% calibration"),
    "stall": Entry(
        "STALL", "genvideo-val", "stall", "Ben Hayun et al., CVPR'26", "training-free",
        "stall_genvideo_full.json", None, "job 271899 (repro/logs/stallgv_271899.out)", True,
        "Training-free DINOv3 spatio-temporal likelihood under VATEX real-video statistics. Full run; the "
        "README's 83.5 AUROC is the June subsample run"),
    "restrav": Entry(
        "ReStraV", "genvideo-val", "restrav", "Interno et al., 2025", "frozen-probe",
        "restrav_genvideo_full.json", None, "job 271898 (repro/logs/restravgv_271898.out)", True,
        "Frozen DINOv2 trajectory geometry + MLP trained in SynthScope (the served restrav.pth; its training "
        "set is not recorded); does not transfer to GenVideo-Val (AUROC below 50)"),
}


def _mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 2)


def build(slug: str) -> dict:
    """Return the submission for one entry, with metrics read from its bench_results file."""
    e = ENTRIES[slug]
    fname, gens = e.results_file, e.generators
    res = json.loads((RESULTS / fname).read_text())
    pg = res["per_generator"]
    video = e.track == "genvideo-val"
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
        "method": e.method, "track": e.track, "modality": "video" if video else "image", "model_type": e.model_type,
        "paper": e.paper, "url": REPO, "evaluation": e.evaluation,
        "metrics": {k: metrics.get(k) for k in METRIC_KEYS},
        "per_generator": {g: {k: pg[g][k] for k in keep} for g in gens},
        "submitted_by": "SynthScope", "date": res["date"][:10], "verified": e.verified,
        "source": f"repro/bench_results/{fname} (tag {res['tag']}, {scope}); {e.job}",
        "protocol_notes": e.notes, "synthscope_version": "0.1.0",
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
        # A SynthScope-authored submission must have an entry here, or its numbers have no recorded evidence.
        for path in sorted(SUB.glob("*.json")):
            if path.stem not in ENTRIES and json.loads(path.read_text()).get("submitted_by") == "SynthScope":
                stale.append(f"{path.name} (SynthScope submission with no ENTRIES entry)")
        if stale:
            print("Out of date with repro/bench_results: " + ", ".join(stale), file=sys.stderr)
            return 1
        print(f"OK: {len(ENTRIES)} SynthScope submissions match their bench_results evidence.")
        return 0
    print(f"Wrote {len(ENTRIES)} SynthScope submissions to {SUB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
