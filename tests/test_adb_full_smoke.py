"""Smoke test for Align, Don't Blend's full method (``adb_full``: the VLM branch OR the DDA pixel expert); CPU, no
downloads, no released weights needed.

Checks:
[A] the decision rule == the official one: the UNMODIFIED official ``predict.py`` (github.com/GzZning/Align-Dont-Blend @
    cf168db, $ADB_FULL_OFFICIAL_DIR, default SynthScope_shared/vlm_ref/Align-Dont-Blend-cf168db; skipped without it) is
    run with ``--vlm-results`` / ``--expert-results`` on mocked component outputs: tagged and tag-less answers, both
    verdicts, answers its parser rejects (Real since 898b607, ``vlm_answer_parsed: false``), expert probabilities 0, 0.3, the float just below 0.5, exactly 0.5 (fake: ``>=``), the float just above
    0.5, 0.9 and 1; its final, VLM and expert decisions equal ``fuse_components`` on ``adb_vlm``'s parse of the same
    answers, for VLM P(fake) values on either side of 0.5 and against their verdict (the raw gap is not the decision);
[B] the score: ``score >= 0.5`` exactly when the decision is fake (also at the edges above, and at P_vlm = 0.5 with the
    verdict "real"); the order of the scores within each side; invalid expert probabilities are refused as
    ``read_expert_results`` refuses them; an unparsed answer is Real on the VLM side (the official script decides so
    since 898b607: checked), so the final decision is the expert's;
[C] the expert == the official ``detector.expert``: its transform (built by the official ``PixelArtifactExpert``) equals
    the ``dda`` registry transform on RGB, RGBA, palette, greyscale, small (zero-padded) and odd-sized images, tensor for
    tensor; the official ``ExpertModel`` (``DDAExpert`` before ff5c349; its vendored DINOv2 + LoRA) loads SynthScope's ``dda`` state dict strictly and
    gives the same logits (random weights; skipped without the official clone or the DINOv2 hub code);
[D] the registry spec, the input note, the zoo / Space / licence rows, and the engine path with mocked components: the
    VLM gets the decoded file unconverted (RGBA kept), the expert its RGB conversion; the verdict follows the rule where
    the rounded score shows 0.5; availability needs both components; ``synthscope.load`` decodes without the RGB
    conversion and refuses videos; ``bench_eval`` refuses the key;
[E] the benchmark path: ``bench_eval --per-image`` (stub expert) records each image's key, sha1, label and P(fake);
    ``repro/adb_full/fuse.py`` joins it with VLM records (sha1 checked, labels checked, a missing image refused) and its
    accuracy / AUROC equal a hand computation of the rule; ``run_units`` maps ``limit<N>`` / ``fullfp32`` units to the full data
    with ``--limit N`` / ``--vlm-dtype float32 --vlm-attn eager``; ``trace_check`` checks such results by their RESULT line;
[F] the evaluation preprocessing == the official ``tools/jpeg96.py`` (cf168db) ``prepare_image``: ``bench_eval
    --jpeg-fakes 96`` gives each branch the same pixels as ``prepare_image(image, "GenImage" / "AIGCDetectBenchmark",
    label)`` (fakes: in-memory PIL JPEG q96; reals: the RGB conversion only), on RGB / RGBA / palette / greyscale PNGs and
    a JPEG, for the VLM's input (``swift_rgb`` then ``_jpeg``) and the expert's (``_open_rgb`` then ``_jpeg``); other
    datasets are not recompressed (skipped without the official clone).
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import types
import warnings
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from PIL import Image

os.environ.setdefault("XFORMERS_DISABLED", "1")          # CPU: DINOv2's standard attention (both copies read it)
warnings.filterwarnings("ignore", message="xFormers")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "interface"), str(ROOT / "benchmark")]

from interface.backend import config as backend_config               # noqa: E402
from interface.backend import media                                  # noqa: E402
from interface.backend.engine import Engine                          # noqa: E402
from interface.backend.registry import get_spec, input_note          # noqa: E402
from models.image.adb_full.adb_full import (ADB_FULL, EXPERT_THRESHOLD, JPEG96_DATASETS, JPEG96_QUALITY,  # noqa: E402
                                            REAL_CAP, expert_decision, fuse_components, fused_score, fused_verdict)
from models.image.dda.dda import SynthScopeDDAModel                  # noqa: E402
from models.vlm.parsers import parse_adb, parse_adb_default_real     # noqa: E402
from synthscope import hub                                           # noqa: E402
from synthscope.api import Detector                                  # noqa: E402
import bench_eval                                                    # noqa: E402
from repro.adb_full import fuse                                      # noqa: E402
from repro.vlm_bench import run_units, trace_check                   # noqa: E402

OFFICIAL_DIR = Path(os.environ.get("ADB_FULL_OFFICIAL_DIR",
                                   "/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared/vlm_ref/Align-Dont-Blend-cf168db"))
HAS_OFFICIAL = (OFFICIAL_DIR / "predict.py").exists() and (OFFICIAL_DIR / "detector" / "expert.py").exists()

TAGGED = "<rationale>The hands look fine and the light is coherent.</rationale>\n<answer>\nThis is an {} image.\n</answer>"
TAGLESS = "<rationale>An AI-generated look? No: the reflections agree.</rationale>\nThis is an {} image."
UNPARSABLE = ["<rationale>Hard to say.</rationale>\n<answer>\nMaybe.\n</answer>",
              "<rationale>The image has strong AI-generated characteristics.",                  # truncated, no answer
              "<answer>This is an authentic image.</answer><answer>This is an AI-generated image.</answer>"]
RESPONSES = [TAGGED.format("AI-generated"), TAGGED.format("authentic"), TAGLESS.format("AI-generated"),
             TAGLESS.format("authentic")] + UNPARSABLE
EXPERT_PS = [0.0, 0.3, REAL_CAP, 0.5, math.nextafter(0.5, 1.0), 0.9, 1.0]
VLM_PS = [0.0, 0.2, REAL_CAP, 0.5, 0.97, 1.0]          # also against the verdict: the raw gap can disagree with the text


def run_official(folder: Path, responses: list[str], expert_ps: list[float]) -> subprocess.CompletedProcess:
    """The unmodified official ``predict.py`` on one tiny image per case, with the components' results given as files."""
    images = []
    for i in range(len(responses)):
        path = folder / f"img_{i:03d}.png"
        Image.new("RGB", (8, 8), (i, 0, 0)).save(path)
        images.append(str(path.resolve()))
    (folder / "inputs.txt").write_text("".join(f"{p}\n" for p in images))
    (folder / "vlm.jsonl").write_text("".join(json.dumps({"images": [p], "response": r}) + "\n"
                                              for p, r in zip(images, responses)))
    (folder / "expert.jsonl").write_text("".join(json.dumps({"image": p, "fake_probability": e}) + "\n"
                                                 for p, e in zip(images, expert_ps)))
    return subprocess.run([sys.executable, "predict.py", "--input", str(folder / "inputs.txt"),
                           "--output", str(folder / "out.jsonl"), "--vlm-results", str(folder / "vlm.jsonl"),
                           "--expert-results", str(folder / "expert.jsonl")],
                          cwd=OFFICIAL_DIR, capture_output=True, text=True)


def test_official_rule() -> None:
    if not HAS_OFFICIAL:
        print("[A] official clone not found: skipped")
        return
    print("[A] decisions == the unmodified official predict.py on mocked component outputs ...")
    cases = [(r, e, VLM_PS[(i + j) % len(VLM_PS)]) for i, r in enumerate(RESPONSES) for j, e in enumerate(EXPERT_PS)]
    with tempfile.TemporaryDirectory() as tmp:
        proc = run_official(Path(tmp), [c[0] for c in cases], [c[1] for c in cases])
        assert proc.returncode == 0, proc.stderr[-2000:]
        official = [json.loads(line) for line in (Path(tmp) / "out.jsonl").read_text().splitlines()]
    assert len(official) == len(cases)
    n_fake = 0
    for (response, p_expert, p_vlm), off in zip(cases, official):
        verdict = parse_adb_default_real(response)            # adb_vlm's parser: the official decision
        ours = fuse_components(p_vlm, verdict, p_expert)
        assert fuse_components(p_vlm, parse_adb(response), p_expert) == ours   # a None verdict is read as Real too
        assert off["vlm_response"] == response
        assert off["vlm_answer_parsed"] == (parse_adb(response) is not None), (response, off)
        assert (ours["vlm_verdict"] == "fake") == off["vlm_fake"], (response, off)
        assert (ours["expert_verdict"] == "fake") == off["expert_fake"], (p_expert, off)
        assert (ours["verdict"] == "fake") == off["fake"], (response, p_expert, p_vlm, off)
        assert (ours["score"] >= 0.5) == off["fake"], (response, p_expert, p_vlm, ours["score"])
        n_fake += off["fake"]
    edge = {e: o["expert_fake"] for (_, e, _), o in zip(cases, official)}
    assert edge[0.5] and not edge[REAL_CAP] and edge[math.nextafter(0.5, 1.0)]
    n_unparsed = sum(not o["vlm_answer_parsed"] for o in official)
    print(f"    {len(cases)} cases ({n_fake} fake, {n_unparsed} with an unparsed answer): final, VLM and expert decisions "
          f"and vlm_answer_parsed identical; P(expert) 0.5 is fake, the float below it real; the VLM decision is its "
          f"parsed text (Real when unparsed), not the sign of its gap")


def test_score() -> None:
    print("[B] score, edge cases, unparsed answers ...")
    for vlm_verdict in ("fake", "real", None):
        for p_vlm in VLM_PS:
            for p_expert in EXPERT_PS:
                s = fused_score(p_vlm, vlm_verdict, p_expert)
                v = fused_verdict(vlm_verdict, p_expert)
                assert (s >= 0.5) == (v == "fake"), (vlm_verdict, p_vlm, p_expert, s, v)
                assert v == ("fake" if vlm_verdict == "fake" or p_expert >= 0.5 else "real")
    assert fused_score(0.5, "real", 0.3) == REAL_CAP < 0.5 and round(REAL_CAP, 6) == 0.5    # rounds to 0.5: see [D]
    assert fused_score(0.2, "fake", 0.1) == 0.5 and fused_score(0.97, "real", 0.1) == REAL_CAP
    assert fused_score(0.1, "real", 0.4) == 0.4 and fused_score(0.45, "real", 0.4) == 0.45      # order kept in a side
    assert fused_score(0.99, "fake", 0.6) == 0.99 and fused_score(0.7, "fake", 0.8) == 0.8
    for bad in (float("nan"), float("inf"), -0.1, 1.5, True, "0.7", None):
        try:
            expert_decision(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} accepted")
    assert expert_decision(1) and not expert_decision(0) and EXPERT_THRESHOLD == ADB_FULL.expert_threshold == 0.5
    assert fuse_components(0.5, None, 0.7)["verdict"] == "fake" and fuse_components(0.5, None, 0.3)["verdict"] == "real"
    assert fuse_components(0.5, None, 0.3)["vlm_verdict"] == "real" and fuse_components(0.5, None, 0.3)["score"] < 0.5
    assert [fuse_components(0.9, "fake", e)["decided_by"] for e in (0.9, 0.1)] == ["both", "vlm"]
    assert [fuse_components(0.1, "real", e)["decided_by"] for e in (0.9, 0.1)] == ["expert", "neither"]
    for unparsable in UNPARSABLE:
        assert parse_adb(unparsable) is None and parse_adb_default_real(unparsable) == "real"
    if HAS_OFFICIAL:
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_official(Path(tmp), [UNPARSABLE[0]] * 2, [0.9, 0.1])
            assert proc.returncode == 0, proc.stderr[-500:]
            rows = [json.loads(line) for line in (Path(tmp) / "out.jsonl").read_text().splitlines()]
        assert [(r["vlm_fake"], r["vlm_answer_parsed"], r["fake"]) for r in rows] == [(False, False, True), (False, False, False)]
        print("    the official script decides Real for an unparsable answer (checked); the OR is then the expert's")
    print("    score >= 0.5 <=> fake for every case; 0.5 - ulp rounds to 0.5 (the verdict, not the rounded score, "
          "decides)")


def official_expert_module():
    sys.path.insert(0, str(OFFICIAL_DIR))
    try:
        from detector import expert                                  # the official package (detector/expert.py)
    finally:
        sys.path.remove(str(OFFICIAL_DIR))
    return expert


def official_transform(expert):
    """The transform the official ``PixelArtifactExpert`` builds (on a meta-device model: no weights needed)."""
    with torch.device("meta"):
        state = expert.ExpertModel().state_dict()
    with tempfile.NamedTemporaryFile(suffix=".pth") as f:
        saved = torch.load
        torch.load = lambda *a, **k: state
        try:
            obj = expert.PixelArtifactExpert(Path(f.name), "meta")
        finally:
            torch.load = saved
    return obj.transform


def test_expert() -> None:
    if not HAS_OFFICIAL:
        print("[C] official clone not found: skipped")
        return
    print("[C] the expert == the official detector.expert (transform, architecture) ...")
    expert = official_expert_module()
    official_tf = official_transform(expert)
    ours_tf = get_spec("dda").transform
    g = torch.Generator().manual_seed(0)
    images = []
    for w, h in ((500, 400), (200, 150), (337, 337), (999, 335), (336, 336), (100, 900)):
        arr = (torch.rand(h, w, 3, generator=g) * 255).to(torch.uint8).numpy()
        images.append(Image.fromarray(arr))
    rgba = images[0].convert("RGBA")
    rgba.putalpha(Image.new("L", rgba.size, 7))
    images += [rgba, images[1].convert("P"), images[2].convert("L")]
    for buf_fmt in ("PNG", "JPEG"):                                   # through a file, as both load it
        buf = io.BytesIO()
        images[0].save(buf, buf_fmt)
        images.append(Image.open(io.BytesIO(buf.getvalue())))
    for im in images:
        a = official_tf(im.convert("RGB"))                            # detector/expert.py: image.convert("RGB")
        b = ours_tf(media.to_rgb(im))                                 # the engine's expert input
        assert a.shape == b.shape == (3, 336, 336) and torch.equal(a, b), (im.size, im.mode)
    print(f"    transform: {len(images)} images (RGB, RGBA, P, L, smaller than 336, odd sizes, PNG/JPEG files) "
          "tensor-equal")

    try:
        ours = SynthScopeDDAModel(OmegaConf.create({"ModelConfig": {"model_type": "dda", "pretrained_backbone": False},
                                                    "TrainerConfig": {}})).eval()
    except Exception as e:                                            # no DINOv2 hub code here (offline, no cache)
        print(f"    architecture: DINOv2 hub code unavailable ({type(e).__name__}): skipped")
        return
    torch.manual_seed(0)
    with torch.no_grad():
        for name, p in ours.named_parameters():
            if ".lora.lora_B" in name or name.startswith("base_model.fc."):
                p.normal_(0, 0.02)                                    # LoRA B is zero-initialised: make it count
    off = expert.ExpertModel().eval()
    off.load_state_dict(ours.state_dict(), strict=True)
    assert set(off.state_dict()) == set(ours.state_dict()) and len(ours.state_dict()) == 537
    x = torch.stack([ours_tf(media.to_rgb(im)) for im in images[:2]])
    with torch.no_grad():
        lo, lt = off(x), ours(x)
    diff = float((lo - lt).abs().max())
    assert torch.allclose(lo, lt, atol=1e-5, rtol=0), diff
    print(f"    architecture: the official ExpertModel loads SynthScope's 537 dda tensors strictly; logits max |diff| "
          f"{diff:.2e} (random weights, CPU)")


class _Stub:
    """Records what each component receives; returns fixed outputs."""

    def __init__(self, vlm_out: dict, p_expert: float):
        self.vlm_out, self.p_expert, self.seen = vlm_out, p_expert, {}

    def vlm_score(self, spec, pil, decoding="greedy", seed=None):
        self.seen["vlm"] = (spec.key, pil.mode)
        return dict(self.vlm_out)

    def expert_probability(self, spec, pil):
        self.seen["expert"] = (spec.key, pil.mode)
        return self.p_expert


def vlm_out(verdict, p):
    logit = math.log(p / (1 - p)) if 0 < p < 1 else (0.0 if p == 0.5 else math.copysign(50.0, p - 0.5))
    return {"verdict": verdict, "fake_probability": p, "logit": logit, "score_source": "verdict_token",
            "verdict_token": "ĠAI" if verdict == "fake" else "Ġauthentic", "new_tokens": 42, "finished": True,
            "input_size": [56, 56], "rationale": "stub", "response": "stub response"}


def engine_with(stub: _Stub) -> Engine:
    engine = Engine(device="cpu")
    engine._get_model = lambda spec: None                            # components are mocked
    engine.vlm_score = stub.vlm_score
    engine.expert_probability = stub.expert_probability
    return engine


def test_registry_and_engine() -> None:
    print("[D] registry, engine path with mocked components, availability, API ...")
    spec = get_spec("adb_full")
    assert (spec.predict_kind, spec.ckpt_kind, spec.checkpoint, spec.weights, spec.modality) == \
        ("fusion", "fusion", None, "official", "image")
    assert spec.fusion is ADB_FULL and (ADB_FULL.vlm, ADB_FULL.expert) == ("adb_vlm", "dda") and spec.has_vlm
    assert spec.transform is None and spec.vlm is None and spec.overrides == [] and spec.eval_batch_size(32) == 1
    assert get_spec("adb_vlm").has_vlm and not get_spec("dda").has_vlm
    note = input_note(spec)
    assert "336" in note["label"] and "1,210,000" in note["explanation"]
    assert "adb_full" not in hub.NEVER_REDISTRIBUTE and "adb_vlm" in hub.NEVER_REDISTRIBUTE
    assert "Apache-2.0" in hub.NEVER_REDISTRIBUTE["adb_vlm"]
    licences = (ROOT / "LICENSES.md").read_text().splitlines()
    for key in ("adb_full", "adb_vlm"):
        row = [line for line in licences if line.startswith(f"| `{key}` |")]
        assert len(row) == 1 and "Apache-2.0" in row[0] and "no licence" not in row[0], (key, row)
    fetch = (ROOT / "deploy" / "hf-space" / "fetch_weights.py").read_text()
    assert "adb_full" not in fetch and "adb_vlm" not in fetch

    # Engine: the VLM sees the decoded file (RGBA kept), the expert its RGB conversion; the rule decides the verdict.
    rgba = Image.new("RGBA", (40, 30), (10, 20, 30, 128))
    for (verdict, p_vlm, p_expert), want, by in ((("real", 0.5, 0.3), "real", "neither"),
                                                 (("real", 0.02, 0.5), "fake", "expert"),
                                                 (("fake", 0.3, 0.1), "fake", "vlm"),
                                                 (("fake", 0.99, 0.97), "fake", "both"),
                                                 ((None, 0.5, 0.2), "real", "neither")):
        stub = _Stub(vlm_out(verdict, p_vlm), p_expert)
        res = engine_with(stub).predict_image(spec, rgba)
        assert stub.seen == {"vlm": ("adb_vlm", "RGBA"), "expert": ("dda", "RGB")}, stub.seen
        assert res["verdict"] == want and res["decided_by"] == by, (verdict, p_vlm, p_expert, res["verdict"])
        assert (res["fused_score"] >= 0.5) == (want == "fake") and res["model"] == "adb_full"
        assert res["components"]["dda"]["fake_probability"] == p_expert and res["components"]["dda"]["threshold"] == 0.5
        assert res["components"]["adb_vlm"]["verdict"] == (verdict or "real")
        assert res["response"] == "stub response" and res["orig_size"] == [40, 30]
    stub = _Stub(vlm_out("real", 0.5), 0.3)
    res = engine_with(stub).predict_image(spec, rgba)
    assert res["fake_probability"] == 0.5 and res["verdict"] == "real" and res["fused_score"] == REAL_CAP
    try:
        Engine(device="cpu").predict_video_frames(spec, [rgba.convert("RGB")])
    except ValueError as e:
        assert "still images only" in str(e)
    else:
        raise AssertionError("video frames accepted")

    # Availability: both components' files.
    saved = backend_config.CHECKPOINTS_DIR
    with tempfile.TemporaryDirectory() as tmp:
        backend_config.CHECKPOINTS_DIR = Path(tmp)
        try:
            engine = Engine(device="cpu")
            assert not engine.is_available(spec)
            (Path(tmp) / "dda.pth").write_bytes(b"x")
            assert engine.is_available(get_spec("dda")) and not engine.is_available(spec)
            for folder in ("adb_vlm", get_spec("adb_vlm").vlm.base_model):
                (Path(tmp) / folder).mkdir()
            assert engine.is_available(get_spec("adb_vlm")) and engine.is_available(spec)
            (Path(tmp) / "dda.pth").unlink()
            assert not engine.is_available(spec)
            det = Detector(spec, device="cpu", auto_download=False)
            try:
                det._ensure_weights()
            except FileNotFoundError as e:
                assert "dda" in str(e), e
            else:
                raise AssertionError("missing dda.pth not reported")
        finally:
            backend_config.CHECKPOINTS_DIR = saved

    # Public API: the decoded file without an RGB conversion (as for a VLM); videos refused.
    buf = io.BytesIO()
    rgba.save(buf, "PNG")
    det = Detector(spec, device="cpu")
    kind, pil, _ = det._classify(buf.getvalue())
    assert kind == "image" and pil.mode == "RGBA"
    det._ensure_weights = lambda: None
    try:
        det.predict(Path(tempfile.gettempdir()) / "clip.mp4")
    except ValueError as e:
        assert "still images only" in str(e)
    else:
        raise AssertionError("video accepted")

    # bench_eval refuses the key (its components are scored per image and fused).
    proc = subprocess.run([sys.executable, str(ROOT / "benchmark" / "bench_eval.py"), "--model", "adb_full",
                           "--benchmark", "pergen", "--data", tempfile.gettempdir(), "--tag", "x", "--device", "cpu"],
                          capture_output=True, text=True, cwd=ROOT,
                          env={**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])})
    assert proc.returncode != 0 and "repro/adb_full/fuse.py" in proc.stderr, proc.stderr[-800:]
    print("    spec (fusion of adb_vlm + dda, no weights of its own); VLM gets RGBA, expert RGB; verdict by the rule "
          "(rounded score 0.5 -> real); available only with both; API raw decode, video refused; bench_eval refuses")


class _StubExpertEngine:
    device = "cpu"

    def _get_model(self, spec):
        return None

    def _fake_prob(self, model, spec, x):
        return x.mean(dim=(1, 2, 3))                                  # "P(fake)" = mean pixel in [0, 1]


def _expert_run(tmp: Path) -> dict:
    spec = types.SimpleNamespace(predict_kind="bce", transform=lambda im: torch.from_numpy(
        np.asarray(im, dtype=np.float32) / 255).permute(2, 0, 1), vlm=None, eval_batch_size=lambda b: b)
    det = types.SimpleNamespace(_engine=_StubExpertEngine(), spec=spec, _ensure_weights=lambda: None)
    saved = bench_eval.synthscope
    bench_eval.synthscope = types.SimpleNamespace(load=lambda model, device=None, **kw: det, __version__="stub")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rc = bench_eval.main(["-m", "stubexpert", "--benchmark", "pergen", "--data", str(tmp / "pool"), "--tag", "e",
                                  "--out", str(tmp / "dda.json"), "--results-log", str(tmp / "log.jsonl"), "--per-image",
                                  "--bs", "3"])
    finally:
        bench_eval.synthscope = saved
    assert rc == 0
    return json.loads((tmp / "dda.json").read_text())


def test_benchmark_path() -> None:
    print("[E] bench_eval --per-image, fuse.py, run_units variants, trace_check ...")
    with tempfile.TemporaryDirectory() as tmp_:
        tmp = Path(tmp_)
        levels = {"0_real": [30, 60, 200, 90], "1_fake": [220, 40, 180, 250]}   # P = level / 255
        for sub, vals in levels.items():
            (tmp / "pool" / "g" / sub).mkdir(parents=True)
            for i, v in enumerate(vals):
                Image.new("RGB", (8, 8), (v, v, v)).save(tmp / "pool" / "g" / sub / f"{i}.png")
        d = _expert_run(tmp)
        recs = d["per_image"]
        assert [r["key"] for r in recs] == [f"g/{s}/{i}.png" for s in levels for i in range(4)]
        assert [r["label"] for r in recs] == [0] * 4 + [1] * 4
        for r in recs:
            data = (tmp / "pool" / r["key"]).read_bytes()
            assert r["sha1"] == __import__("hashlib").sha1(data).hexdigest()
        assert [round(r["prob"] * 255) for r in recs] == levels["0_real"] + levels["1_fake"]
        # VLM records: verdicts against labels r: real real fake real | f: real fake real fake
        verdicts = ["real", "real", "fake", "real", "real", "fake", "real", "fake"]
        logits = [-9.0, -3.0, 2.5, -0.5, -4.0, 7.0, -1.0, 12.0]
        vlm = {"gpu": "stub", "jpeg_fakes": None, "means": {}, "per_image": [
            {"generator": "g", "key": r["key"], "label": r["label"], "logit": lg, "verdict": v, "new_tokens": 5,
             "sha1": r["sha1"]} for r, v, lg in zip(recs, verdicts, logits)]}
        (tmp / "vlm.json").write_text(json.dumps(vlm))
        cfg = {"paper": "BFree-Online", "main": True, "data": "stub", "precision": "bf16", "gpu": None,
               "vlm": [str(tmp / "vlm.json")], "dda": [str(tmp / "dda.json")], "generators": None, "manifest": None,
               "vlm_logs": ["stub.out"]}
        res = fuse.fuse_row("stub", cfg)
        want = ["fake" if v == "fake" or r["prob"] >= 0.5 else "real" for r, v in zip(recs, verdicts)]
        assert [x["verdict"] for x in res["per_image"]] == want
        real_acc = 100 * sum(w == "real" for w, r in zip(want, recs) if r["label"] == 0) / 4
        fake_acc = 100 * sum(w == "fake" for w, r in zip(want, recs) if r["label"] == 1) / 4
        m = res["per_generator"]["g"]
        assert m["acc"] == round((real_acc + fake_acc) / 2, 2) and m["n_real"] == m["n_fake"] == 4
        assert m["auroc"] == round(100 * roc_auc_score([r["label"] for r in recs], [x["score"] for x in res["per_image"]]), 2)
        assert res["file_identity"]["sha1_vs_vlm_record"] == 8 and res["vlm_gpu"] == ["stub"]
        assert res["logs"]["vlm"] == ["stub.out"] and all(r.get("vlm_logs") for r in fuse.ROWS.values())
        vlm["per_image"][0]["sha1"] = "0" * 40
        (tmp / "vlm.json").write_text(json.dumps(vlm))
        for bad in ("sha1", "missing"):
            if bad == "missing":
                vlm["per_image"] = vlm["per_image"][1:]
                (tmp / "vlm.json").write_text(json.dumps(vlm))
            try:
                fuse.fuse_row("stub", cfg)
            except SystemExit:
                pass
            else:
                raise AssertionError(f"{bad}: accepted")
    spec = run_units.load_spec()
    args = types.SimpleNamespace(vlm_model="adb_vlm", out_root="/o", subsets_root="/s", python="py", bs=32, seed=0,
                                 cache_root="/c")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"aigibench25": "/full/aigi", "wildrf": "/full/wildrf"}, f)
    args.full_data = f.name
    cmd, out = run_units.command("vlm:aigibench25:limit1500:CommunityAI", spec, args)
    assert out == Path("/o/aigibench25/adb_vlm__limit1500__CommunityAI.json")
    assert cmd[cmd.index("--data") + 1] == "/full/aigi" and cmd[cmd.index("--limit") + 1] == "1500"
    assert cmd[cmd.index("--select") + 1] == "CommunityAI" and cmd[cmd.index("--vlm-cache") + 1] == "/c/aigibench25"
    cmd, out = run_units.command("vlm:wildrf:fullfp32", spec, args)
    assert out.name == "adb_vlm__fullfp32.json" and "--limit" not in cmd
    assert cmd[cmd.index("--vlm-dtype") + 1] == "float32" and cmd[cmd.index("--vlm-attn") + 1] == "eager"
    cmd, _ = run_units.command("vlm:wildrf:full", spec, args)
    assert "--vlm-dtype" not in cmd and "--limit" not in cmd
    with open(args.full_data, "w") as fh:
        json.dump({"aigibench25": "/full/aigi", "wildrf": "/full/wildrf", "genimage_val100k": "/full/gi"}, fh)
    cmd, out = run_units.command("vlm:genimage_val100k:limit625_jpegfakes96:adm", spec, args)
    assert out.name == "adb_vlm__limit625_jpegfakes96__adm.json" and cmd[cmd.index("--data") + 1] == "/full/gi"
    assert cmd[cmd.index("--limit") + 1] == "625" and cmd[cmd.index("--jpeg-fakes") + 1] == "96"
    assert trace_check.canonical("limit625_jpegfakes96") == "jpegfakes96" and trace_check.canonical("limit20") == "clean"
    paper = {"a": 90.0, "b": 80.0}
    res = {"per_generator": {"a": {"acc": 88.5}, "b": {"acc": 77.9}}}
    assert fuse.status({"status": "per_generator_tolerance"}, res, paper) == "reference"            # b: -2.1
    res["per_generator"]["b"]["acc"] = 78.0
    assert fuse.status({"status": "per_generator_tolerance"}, res, paper) == "within tolerance (subsample)"
    assert fuse.status({"status": "below"}, res, paper) == "below"
    os.unlink(f.name)
    assert trace_check.SHARED_CLEAN.fullmatch("limit1500") and trace_check.SHARED_CLEAN.fullmatch("full")
    assert not trace_check.SHARED_CLEAN.fullmatch("fullfp32")
    print("    per-image key / sha1 / P recorded; fused verdicts and AUROC == hand computation; sha1 mismatch and a "
          "missing image refused; limit<N> / fullfp32 units; trace_check variants")


def test_jpeg96() -> None:
    jpeg96 = OFFICIAL_DIR / "tools" / "jpeg96.py"
    if not jpeg96.exists():
        print("[F] official tools/jpeg96.py not found: skipped")
        return
    print("[F] --jpeg-fakes 96 == the official tools/jpeg96.py prepare_image ...")
    spec = importlib.util.spec_from_file_location("adb_official_jpeg96", jpeg96)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    rng = np.random.RandomState(7)
    images = {"rgb": Image.fromarray((rng.rand(97, 131, 3) * 255).astype("uint8")),
              "rgba": Image.fromarray((rng.rand(64, 80, 4) * 255).astype("uint8"), "RGBA"),
              "palette": Image.fromarray((rng.rand(50, 70, 3) * 255).astype("uint8")).convert("P"),
              "gray": Image.fromarray((rng.rand(45, 61) * 255).astype("uint8"), "L")}
    n = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, img in images.items():
            for fmt in ("PNG", "JPEG") if name == "rgb" else ("PNG",):
                path = Path(tmp) / f"{name}.{fmt.lower()}"
                img.save(path, fmt)
                for dataset in ("GenImage", "AIGCDetectBenchmark", "WildRF"):
                    for label in (0, 1):
                        want = np.asarray(official.prepare_image(Image.open(path), dataset, label))
                        jpeg = label == 1 and dataset in JPEG96_DATASETS
                        vlm = get_spec("adb_vlm").vlm.image_loader(media.decode_image_raw(path.read_bytes()))
                        expert = bench_eval._open_rgb(path)
                        if jpeg:
                            vlm, expert = bench_eval._jpeg(vlm, JPEG96_QUALITY), bench_eval._jpeg(expert, JPEG96_QUALITY)
                        assert np.array_equal(np.asarray(vlm), want) and np.array_equal(np.asarray(expert), want), \
                            (name, fmt, dataset, label)
                        n += 1
    print(f"    {n} cases (RGB / RGBA / P / L PNGs, a JPEG; GenImage, AIGCDetectBenchmark, another dataset; real, fake): "
          f"the VLM's and the expert's pixels == prepare_image's")


if __name__ == "__main__":
    with contextlib.suppress(Exception):
        import transformers
        transformers.logging.set_verbosity_error()
    test_official_rule()
    test_score()
    test_expert()
    test_registry_and_engine()
    test_benchmark_path()
    test_jpeg96()
    print("adb_full smoke test passed")
