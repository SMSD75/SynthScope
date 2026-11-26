"""Align, Don't Blend (arXiv 2512.06746, ACM MM 2026), the FULL method: the VLM branch (``adb_vlm``) OR the
pixel-artifact expert (``dda``), with the official final-decision rule.

Official code: github.com/GzZning/Align-Dont-Blend @ cf168db (Apache-2.0; first integrated at 5d83e40). ``predict.py``
runs the VLM (``swift infer``,
as ``adb_vlm`` reproduces it) and the expert command of its README, ``python -m detector.expert --checkpoint
weights/DDA_ckpt.pth`` (DDA's DINOv2 ViT-L/14 + LoRA; CenterCrop(336), ToTensor, CLIP normalisation; P = sigmoid(logit)),
on the same files, and writes per image (``detector/core.py``):

* ``vlm_fake    = parse_vlm_verdict(response)``: True / False; where it raises ValueError, False (Real), with
  ``vlm_answer_parsed: false`` (commit 898b607, "Default unparsed VLM answers to Real before OR fusion"; at 5d83e40 the
  script stopped);
* ``expert_fake = fake_probability >= expert_threshold`` (``read_expert_results``; ``--expert-threshold``, default 0.5,
  so a probability of exactly 0.5 is fake);
* ``fake        = fuse(vlm_fake, expert_fake) = vlm_fake or expert_fake``.

The expert is the released DDA checkpoint (HF Junwei-Xi/Dual-Data-Alignment ``DDA_ckpt.pth``, byte-identical to
SynthScope's ``dda.pth``); the authors confirmed that DDA is the expert (2026-11-20). There are no other weights, no
tie rule and no score: the official output is three booleans.

SynthScope's continuous score (for AUROC / AP), at the official threshold 0.5:

    s_vlm = max(P_vlm, 0.5)          if the VLM verdict is "fake"
          = min(P_vlm, 0.5 - ulp)    otherwise ("real", or an unparsed answer)
    score = max(s_vlm, P_dda)

``P_vlm`` is ``adb_vlm``'s P(fake) (sigmoid of the raw verdict-token logit gap) and ``P_dda`` is DDA's sigmoid, both
unrounded; ``0.5 - ulp`` is the largest float below 0.5. Clamping ``P_vlm`` to the side of its parsed verdict makes
``score >= 0.5`` exactly ``vlm_fake or P_dda >= 0.5``, also where the raw gap and the emitted text disagree (the
repetition penalty picks the text; a bf16 gap of exactly 0 gives P_vlm = 0.5 with the verdict "real"). Within each side
the order of the scores is kept: real-side scores rank by the larger of the two P(fake)s, fake-side ones likewise.

An answer ``parse_vlm_verdict`` rejects is Real on the VLM side (``predict.py`` since 898b607; ``adb_vlm``'s parser,
``parse_adb_default_real``, already returns "real" for it, and a None verdict is read the same way here): the final
decision is the expert's. Its score is ``max(min(P_vlm, 0.5 - ulp), P_dda)``.

Evaluation preprocessing (``tools/jpeg96.py`` @ cf168db, ``prepare_image(image, dataset, label)``): RGB, and for
GenImage and AIGCDetectBenchmark only, the FAKE images (label 1) re-encoded in memory as PIL JPEG quality 96 (Pillow's
default 4:2:0 chroma subsampling) and decoded again, before both branches; real images are not recompressed. This is
``bench_eval --jpeg-fakes 96`` (``_jpeg``: the same PIL save and decode), applied after the VLM's ``swift_rgb`` and the
expert's RGB conversion (the same ``convert("RGB")``).
"""
import math
from dataclasses import dataclass

EXPERT_THRESHOLD = 0.5                       # predict.py --expert-threshold default (read_expert_results: >=)
REAL_CAP = math.nextafter(0.5, 0.0)          # the largest float below 0.5: the top of the VLM's "real" side
VERDICTS = ("fake", "real")


@dataclass(frozen=True)
class FusionConfig:
    """A detector that combines two registered detectors with an official decision rule (``predict_kind="fusion"``)."""
    vlm: str                                 # registry key of the VLM branch (scored by Engine.vlm_score)
    expert: str                              # registry key of the pixel expert (an image detector, sigmoid output)
    expert_threshold: float = EXPERT_THRESHOLD
    code: str = ""                           # the official code the rule is ported from


ADB_FULL = FusionConfig(vlm="adb_vlm", expert="dda", expert_threshold=EXPERT_THRESHOLD,
                        code="github.com/GzZning/Align-Dont-Blend @ cf168db (predict.py, detector/core.py fuse, "
                             "read_expert_results)")
JPEG96_DATASETS = ("GenImage", "AIGCDetectBenchmark")  # tools/jpeg96.py @ cf168db: the fakes of these, JPEG q96 in memory
JPEG96_QUALITY = 96


def fuse(vlm_fake: bool, expert_fake: bool) -> bool:
    """``detector/core.py`` ``fuse`` for two booleans (``predict.py`` passes it no None since 898b607): the OR."""
    return vlm_fake or expert_fake


def expert_decision(probability: float, threshold: float = EXPERT_THRESHOLD) -> bool:
    """The expert's decision as ``read_expert_results`` makes it: ``probability >= threshold``, after the same checks (a
    finite number in [0, 1], not a bool)."""
    if (isinstance(probability, bool) or not isinstance(probability, (int, float)) or not math.isfinite(probability)
            or not 0.0 <= probability <= 1.0):
        raise ValueError(f"invalid expert fake_probability: {probability!r}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Expert threshold must be in [0, 1]")
    return probability >= threshold


def vlm_side_score(p_vlm: float, vlm_verdict) -> float:
    """``P_vlm`` clamped to the side of the VLM decision: >= 0.5 for "fake", < 0.5 for "real" (an unparsed answer is
    Real)."""
    if vlm_verdict == "fake":
        return max(float(p_vlm), 0.5)
    return min(float(p_vlm), REAL_CAP)


def fused_score(p_vlm: float, vlm_verdict, p_expert: float) -> float:
    """SynthScope's continuous score; ``>= 0.5`` exactly when the official decision is fake (threshold 0.5)."""
    expert_decision(p_expert)
    return max(vlm_side_score(p_vlm, vlm_verdict), float(p_expert))


def fused_verdict(vlm_verdict, p_expert: float, threshold: float = EXPERT_THRESHOLD) -> str:
    """The official final decision as "fake" / "real": a VLM verdict other than "fake" (also None, an unparsed answer)
    is Real, as ``predict.py`` decides since 898b607."""
    return "fake" if fuse(vlm_verdict == "fake", expert_decision(p_expert, threshold)) else "real"


def fuse_components(p_vlm: float, vlm_verdict, p_expert: float) -> dict:
    """The fused decision and score from the two components' outputs (unrounded probabilities; ``vlm_verdict`` =
    "fake" / "real" as ``adb_vlm``'s parser returns it; None, an unparsed answer, is Real)."""
    verdict = fused_verdict(vlm_verdict, p_expert)
    score = fused_score(p_vlm, vlm_verdict, p_expert)
    if (score >= 0.5) != (verdict == "fake"):          # cannot happen (see the module docstring); kept as a guard
        raise RuntimeError(f"fused score {score!r} disagrees with the decision {verdict!r}")
    expert_fake = expert_decision(p_expert)
    vlm_fake = vlm_verdict == "fake"
    decided_by = ("both" if vlm_fake and expert_fake else "vlm" if vlm_fake else "expert" if expert_fake else "neither")
    return {"verdict": verdict, "score": score, "vlm_verdict": "fake" if vlm_fake else "real",
            "expert_verdict": "fake" if expert_fake else "real", "decided_by": decided_by}
