"""Generate the reference (already-reproduced) leaderboard submissions.

Every number here is **transcribed from the verified reproduction table in the repo README**
(each traces to a logged `Test Metrics` evaluation; see `repro/FINAL_RESULTS.md`). No number is
estimated. Re-run to regenerate `submissions/*.json`:

    python benchmark/seed_reference_results.py
"""
import json
from pathlib import Path

SUB = Path(__file__).resolve().parent / "submissions"
REPO = "https://github.com/SMSD75/SynthScope"
SOURCE = "SynthScope reproduction (logged Test Metrics); see repro/FINAL_RESULTS.md"
DATE = "2026-06-02"

# (slug, method, track, modality, model_type, paper, evaluation, metrics, protocol_notes)
ROWS = [
    # --- ForenSynths (image) ---
    ("fat_former", "FatFormer", "forensynths", "image", "fat_former", "Liu et al., CVPR'24",
     "official-weights", {"acc": 98.39, "map": 99.70}, "ForenSynths-8 subset; official released weights"),
    ("co_spy_official", "CO-SPY (official weights)", "forensynths", "image", "co_spy", "Cheng et al., CVPR'25",
     "official-weights", {"acc": 95.12, "map": 98.71}, "ForenSynths-8; vendored official architecture"),
    ("npr", "NPR", "forensynths", "image", "npr", "Tan et al., CVPR'24",
     "official-weights", {"acc": 91.83, "map": 87.09}, "ForenSynths-8; official ProGAN weights"),
    ("aide", "AIDE", "forensynths", "image", "aide", "Yan et al., ICLR'25",
     "official-weights", {"map": 92.08}, "ForenSynths-8; mAP reproduces, acc@0.5 miscalibrated (single-view vs official 5-view DCT)"),
    ("clip", "UnivFD / CLIP", "forensynths", "image", "clip", "Ojha et al., CVPR'23",
     "frozen-probe", {"map": 97.2, "auroc": 97.0}, "ForenSynths; frozen CLIP ViT-L/14 probe (ProGAN)"),
    ("cnn", "CNNDetection", "forensynths", "image", "cnn", "Wang et al., CVPR'20",
     "from-scratch", {"map": 96.8, "auroc": 96.6}, "ForenSynths; ResNet-50 fine-tune (ProGAN)"),
    ("dinov2", "DINOv2 probe", "forensynths", "image", "dinov2", "Oquab et al., 2023",
     "frozen-probe", {"map": 95.03, "auroc": 94.86}, "ForenSynths; frozen DINOv2 ViT-L/14 probe"),
    ("safe", "SAFE", "forensynths", "image", "safe", "Li et al., KDD'25",
     "from-scratch", {"map": 94.13, "auroc": 93.68}, "ForenSynths; trained from scratch"),
    ("co_spy_scratch", "CO-SPY (from scratch)", "forensynths", "image", "co_spy", "Cheng et al., CVPR'25",
     "from-scratch", {"map": 96.98, "auroc": 96.79}, "ForenSynths; SynthScope re-implementation trained on ProGAN"),
    # --- GenImage SD-v1.4 (image) ---
    ("spai_official", "SPAI (official weights)", "genimage-sdv1.4", "image", "spai", "Karageorgiou et al., CVPR'25",
     "official-weights", {"auroc": 98.45, "ap": 94.01}, "GenImage SD-v1.4; vendored official architecture, fixed-res 2x2x224"),
    ("vib_net", "VIB-Net", "genimage-sdv1.4", "image", "vib_net", "Zhang et al., CVPR'25",
     "from-scratch", {"map": 99.33, "auroc": 99.33}, "GenImage SD-v1.4 single real/fake pool (mAP == AUROC by construction)"),
    ("spai_scratch", "SPAI (from scratch)", "genimage-sdv1.4", "image", "spai", "Karageorgiou et al., CVPR'25",
     "from-scratch", {"auroc": 97.66}, "DMID Latent-Diffusion vs COCO; SynthScope fixed-resolution variant"),
    # --- GenVideo-Val (video) ---
    ("d3", "D3", "genvideo-val", "video", "d3", "ICCV'25",
     "training-free", {"ap": 95.6, "auroc": 93.6}, "Training-free frozen CLIP 2nd-order trajectory"),
    ("restrav", "ReStraV", "genvideo-val", "video", "restrav", "2025",
     "frozen-probe", {"ap": 96.67, "auroc": 97.06}, "Frozen DINOv2 trajectory geometry + MLP"),
]

METRIC_KEYS = ["acc", "map", "ap", "auroc"]


def main() -> None:
    SUB.mkdir(parents=True, exist_ok=True)
    for slug, method, track, modality, model_type, paper, evaluation, metrics, notes in ROWS:
        full_metrics = {k: metrics.get(k) for k in METRIC_KEYS}
        sub = {
            "method": method, "track": track, "modality": modality, "model_type": model_type,
            "paper": paper, "url": REPO, "evaluation": evaluation, "metrics": full_metrics,
            "per_generator": None, "submitted_by": "SynthScope", "date": DATE,
            "verified": True, "source": SOURCE, "protocol_notes": notes, "synthscope_version": "0.1.0",
        }
        (SUB / f"{slug}.json").write_text(json.dumps(sub, indent=2) + "\n")
    print(f"Wrote {len(ROWS)} reference submissions to {SUB}")


if __name__ == "__main__":
    main()
