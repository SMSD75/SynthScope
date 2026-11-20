"""Smoke test for WeVisForensics (``wevis``), the agentic video detector run in its own process; CPU, no downloads, no
GPU, no released files needed (a few checks use them when present).

Checks:
[A] the pinned release (models/video/wevis/manifest.json): code commit 43ae1c0, adapters 9fa025a0, base 851bf6e8, the
    five roles + skill card, the PP-OCRv5 models; with the official clone ($WEVIS_OFFICIAL_DIR, default
    SynthScope_shared/vlm_ref/WeVisForensics) every pinned code file's sha256 matches, the official evaluator passes a
    manifest's ``has_audio`` to ``ForensicAgent.run`` (the field the detector sets to False) and the backend decodes
    greedily (skipped otherwise);
[B] the score (runner side, a stub tokenizer, and the Qwen3.5 tokenizer when the base folder is present): the verdict
    token after the LAST ``"content":"`` of the final turn, l(fake) - l(real) from the recorded rows, the pair's softmax
    mass and argmax; a verdict word without a single-token counterpart is "unscored"; the fake/real ids 29571 / 7700;
[C] runner helpers: OOM classification through the official ``ToolError`` chain, sha256 pinning (a changed file is
    refused, verified files are remembered), strict adapter checks (a missing / unexpected key or a factor count that
    differs from the file is refused), the socket guard (non-loopback refused, loopback allowed);
[D] the registry spec, input note, block list, LICENSES rows, the Space list, the download hint; then the engine with a
    MOCKED runner process ($SYNTHSCOPE_WEVIS_PYTHON = a fake interpreter that speaks the runner protocol): the runner's
    command line, environment (offline, no Hub token, isolated deps first) and config (has_audio False), a scored
    video (P(fake) = sigmoid(logit), the agent's verdict, trace, cost), an unparsed one (verdict "unparsed", P 0.5),
    the typed errors (OOM, tool error, a crashed runner, which is restarted on the next call), release_gpu stopping it;
    a still image is refused as video-only before the weights check;
[E] the runner's process group: every kill path takes the OCR-worker-like child with it; SIGKILL only for a group that
    is still alive; a runner nulled by close() mid-score raises WeVisRunnerError (not AttributeError);
[F] the engine stops an evicted / released agent OUTSIDE its lock: other requests are not blocked while close() waits.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import signal
import subprocess
import math
import os
import socket
import stat
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "interface")]

from interface.backend import config as backend_config                              # noqa: E402
from interface.backend.engine import Engine                                         # noqa: E402
from interface.backend.registry import get_spec, input_note                         # noqa: E402
from models.video.wevis import runner                                               # noqa: E402
from models.video.wevis.wevis import (ENV_CODE, ENV_PADDLEX, ENV_PYTHON, ENV_PYTHONPATH, ENV_WORK_DIR, RUNNER,   # noqa: E402
                                      WEVIS, WeVisClosedError, WeVisOOMError, WeVisRunner, WeVisRunnerError, WeVisToolError, download_hint,
                                      load_manifest, missing, sha256_file)
from models.vlm.vlm import verdict_pair_map                                         # noqa: E402
from synthscope import hub                                                          # noqa: E402
from synthscope.api import Detector                                                 # noqa: E402

SHARED = Path("/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared")
OFFICIAL_DIR = Path(os.environ.get("WEVIS_OFFICIAL_DIR", SHARED / "vlm_ref" / "WeVisForensics"))
BASE_DIR = Path(os.environ.get("WEVIS_BASE_DIR", SHARED / "wevis_gate" / "stage" / "ckpts" / "qwen3.5-4b"))
ROLES = ("main", "physical_consistency", "visual_appearance", "common_sense", "ocr_consistency")
FINAL_TURN = '{"reasoning": "Both inspectors found issues.", "action": {"type": "answer", "content": "fake"}}'

FAKE_RUNNER = r'''#!{python}
import json, os, signal, subprocess, sys, time
args = sys.argv[1:]
assert args[0].endswith("runner.py") and args[1] == "--serve" and args[2] == "--config", args
cfg = json.loads(args[3])
env = {{k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_TOKEN", "PYTHONPATH", "WVF_BASE",
                                      "WVF_ADAPTERS", "WVF_OCR_DEVICE", "PADDLE_PDX_CACHE_HOME")}}
out = sys.stdout
child = None
if os.environ.get("FAKE_WEVIS_CHILD"):              # like the official CPU OCR worker: a long-lived child process
    child = subprocess.Popen(["sleep", "600"]).pid
    signal.signal(signal.SIGTERM, signal.SIG_IGN)    # a runner deaf to SIGTERM: the kill must escalate to SIGKILL
out.write(json.dumps({{"event": "ready", "info": {{"cfg": cfg, "env": env, "pid": os.getpid(), "child": child,
    "release": {{"code_commit": "43ae1c0fe25", "adapters_revision": "9fa025a02d", "base_revision": "851bf6e806"}},
    "adapter_tensors": {{"main": 2}}, "versions": {{"transformers": "5.23.0"}}}}}}) + "\n"); out.flush()
for line in sys.stdin:
    req = json.loads(line)
    if req.get("cmd") == "shutdown":
        break
    name = os.path.basename(req["video"])
    if name == "crash.mp4":
        sys.exit(3)
    if name == "hang.mp4":
        time.sleep(600)
    if name == "slow.mp4":
        time.sleep(2)
    if name in ("oom.mp4", "tool.mp4"):
        kind = "oom" if name == "oom.mp4" else "tool_error"
        out.write(json.dumps({{"id": req["id"], "ok": False, "error_kind": kind, "error": kind + " on " + name}}) + "\n")
        out.flush(); continue
    answered = name != "invalid.mp4"
    gap = 15.25 if answered else 0.0
    res = {{"logit": gap, "fake_probability": 1 / (1 + 2.718281828459045 ** -gap) if answered else 0.5,
           "score_source": "verdict_token" if answered else "unparsed", "verdict_token": "fake" if answered else None,
           "verdict": "fake" if answered else None, "status": "answered" if answered else "invalid_output",
           "trace": {{"tool_calls": [["visual_appearance", [0.0, 3.0]]], "steps": [], "final_text": "x",
                      "has_audio": False, "pid": os.getpid()}},
           "cost": {{"seconds": 1.5, "model_calls": 3}}}}
    out.write(json.dumps({{"id": req["id"], "ok": True, "result": res}}) + "\n"); out.flush()
'''


def expect_raise(fn, exc_type, needle: str = ""):
    try:
        fn()
    except exc_type as e:
        assert needle in str(e), (needle, str(e))
        return str(e)
    raise AssertionError(f"expected {exc_type.__name__}")


class StubTokenizer:
    """Whole-piece tokenizer over a fixed vocabulary (byte-level-BPE-like: decoding concatenates the pieces)."""

    def __init__(self, pieces):
        self.pieces = list(pieces)
        self.vocab = {p: i for i, p in enumerate(self.pieces)}

    def get_vocab(self):
        return dict(self.vocab)

    def convert_tokens_to_string(self, toks):
        return "".join(toks)

    def convert_ids_to_tokens(self, ids):
        return self.pieces[ids] if isinstance(ids, int) else [self.pieces[i] for i in ids]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.pieces[i] for i in ids)

    def encode(self, text):
        out, rest = [], text
        while rest:
            piece = max((p for p in self.pieces if rest.startswith(p)), key=len)
            out.append(self.vocab[piece])
            rest = rest[len(piece):]
        return out


def stub_session(tok) -> runner.Session:
    s = object.__new__(runner.Session)
    s.tok = tok
    s.pair_map = verdict_pair_map(tok, [tuple(p) for p in WEVIS.verdict_pairs])
    s.record_ids = sorted({i for pair in s.pair_map.values() for i in pair})
    s.column = {tid: c for c, tid in enumerate(s.record_ids)}
    return s


def recorded_call(s, gen, seed=0):
    g = torch.Generator().manual_seed(seed)
    full = torch.randn(len(gen), max(max(s.record_ids), max(gen)) + 5, generator=g) * 3
    for t, tid in enumerate(gen):                    # greedy: the emitted token is the argmax
        full[t, tid] = full[t].max() + 2.0
    rows = full[:, s.record_ids]
    return {"_ids": gen, "_rows": rows, "_lse": torch.logsumexp(full, -1), "_argmax": full.argmax(-1)}, full


def test_release() -> None:
    print("[A] pinned release, official code sha256, has_audio manifest field, greedy decoding ...")
    m = load_manifest()
    assert m["code"]["commit"].startswith("43ae1c0") and m["adapters"]["revision"].startswith("9fa025a0")
    assert m["base"]["revision"].startswith("851bf6e8") and m["base"]["repo"] == "Qwen/Qwen3.5-4B"
    assert m["adapters"]["repo"] == "tencent/WeVisForensics"
    for role in ROLES:
        assert f"{role}/adapter_model.safetensors" in m["adapters"]["files"] and f"{role}/adapter_config.json" in m["adapters"]["files"]
    assert "skill_card.json" in m["adapters"]["files"] and "demo/wvf_local.py" in m["code"]["files"]
    assert {f.split("/")[1] for f in m["paddlex"]["files"]} == {"PP-OCRv5_server_det", "PP-OCRv5_server_rec"}
    assert all(len(v) == 64 for part in ("code", "adapters", "base", "paddlex") for v in m[part]["files"].values())
    pins = dict(WEVIS.pins)
    assert WEVIS.has_audio is False and pins["transformers"] == "5.23.0" and pins["paddlepaddle"] == "3.2.2"
    assert pins["flash-linear-attention"] == pins["fla-core"] == "0.6.0" and pins["peft"] == "0.21.3"
    assert pins["torch"] == "2.11.0+cu126"
    deps = SHARED / "vlm_pydeps" / "wevis"
    if deps.is_dir():                                    # the pins are the gate env's versions
        for name in ("transformers", "flash_linear_attention", "fla_core"):
            want = pins[name.replace("_", "-")]
            assert (deps / f"{name}-{want}.dist-info").is_dir(), (name, want)
    if not OFFICIAL_DIR.is_dir():
        print(f"    official clone not found ({OFFICIAL_DIR}): code sha256 skipped")
        return
    assert runner.verify_files(OFFICIAL_DIR, m["code"]["files"], "official code", {}) == len(m["code"]["files"])
    evaluate = (OFFICIAL_DIR / "scripts" / "evaluate.py").read_text()
    assert 'has_audio=row.get("has_audio")' in evaluate                    # the manifest field wevis sets to False
    local = (OFFICIAL_DIR / "demo" / "wvf_local.py").read_text()
    assert "do_sample=False" in local and "repetition_penalty=1.0" in local
    print(f"    {len(m['code']['files'])} official code files match; evaluate.py passes the manifest's has_audio")


def test_score() -> None:
    print("[B] verdict token, logit gap, pair mass, unscored ...")
    pieces = ['{"', 'reasoning', '":', ' "', 'Both', ' inspectors', ' found', ' issues', '."', ',', ' "', 'action',
              '":', ' {"', 'type', 'answer', '",', 'content', 'fake', 'real', '"}}', ' fake', ' real', 'Fake', 'Real',
              'fakery', ': ', ':', '"', ' ', '<|im_end|>']
    tok = StubTokenizer(pieces)
    s = stub_session(tok)
    assert s.pair_map[tok.vocab["fake"]] == (tok.vocab["fake"], tok.vocab["real"])
    assert s.pair_map[tok.vocab["Real"]] == (tok.vocab["Fake"], tok.vocab["Real"])
    gen = tok.encode(FINAL_TURN) + [tok.vocab["<|im_end|>"]]
    call, full = recorded_call(s, gen)
    out = s.verdict_score(call)
    step = out["verdict_step"]
    assert gen[step] == tok.vocab["fake"] and out["score_source"] == "verdict_token" and out["verdict_token"] == "fake"
    want = float(full[step, tok.vocab["fake"]] - full[step, tok.vocab["real"]])
    assert out["logit"] == want and abs(out["fake_probability"] - 1 / (1 + math.exp(-want))) < 1e-12
    mass = float(full[step].softmax(-1)[[tok.vocab["fake"], tok.vocab["real"]]].sum())
    assert abs(out["pair_softmax_mass"] - mass) < 1e-5 and out["argmax_is_emitted"] is True
    # the LAST "content" counts (a reasoning that quotes an earlier action does not)
    text2 = '{"reasoning": "content": "real", "action": {"type": "answer", "content": "Real"}}'
    gen2 = tok.encode(text2)
    call2, full2 = recorded_call(s, gen2, seed=1)
    out2 = s.verdict_score(call2)
    assert gen2[out2["verdict_step"]] == tok.vocab["Real"]
    assert out2["logit"] == float(full2[out2["verdict_step"], tok.vocab["Fake"]] - full2[out2["verdict_step"], tok.vocab["Real"]])
    # a verdict word inside a longer token has no counterpart: unscored (the verdict still counts)
    gen3 = tok.encode('{"action": {"type": "answer", "content": "fakery"}}')
    out3 = s.verdict_score(recorded_call(s, gen3)[0])
    assert out3["score_source"] == "unscored" and out3["logit"] == 0.0 and out3["fake_probability"] == 0.5
    if (BASE_DIR / "tokenizer.json").exists():
        from transformers import AutoTokenizer
        qtok = AutoTokenizer.from_pretrained(str(BASE_DIR))
        qs = stub_session(qtok)
        for word, tid in (("fake", 29571), ("real", 7700)):
            assert qtok.convert_tokens_to_ids(word) == tid and tid in qs.pair_map
        assert qs.pair_map[29571] == (29571, 7700)
        ids = qtok.encode(FINAL_TURN, add_special_tokens=False)
        call, full = recorded_call(qs, ids)
        out = qs.verdict_score(call)
        assert ids[out["verdict_step"]] == 29571 and out["logit"] == float(full[out["verdict_step"], 29571] - full[out["verdict_step"], 7700])
        print(f"    Qwen3.5 tokenizer: verdict token 'fake' (29571) at step {out['verdict_step']} of {len(ids)}")
    else:
        print(f"    no Qwen3.5 tokenizer at {BASE_DIR}: stub tokenizer only")


def test_runner_helpers() -> None:
    print("[C] OOM classification, sha256 pinning, strict adapters, socket guard ...")
    s = object.__new__(runner.Session)
    s.common = type("common", (), {"ToolError": type("ToolError", (RuntimeError,), {})})

    def wrapped(inner, outer):
        try:
            try:
                raise inner
            except Exception as exc:
                raise outer from exc
        except Exception as exc:  # noqa: BLE001
            return exc
    assert s.classify(wrapped(torch.OutOfMemoryError("CUDA out of memory"), s.common.ToolError("main-agent generation failed"))) == "oom"
    assert s.classify(wrapped(RuntimeError("CUDA error: out of memory"), ValueError("x"))) == "oom"
    assert s.classify(s.common.ToolError("could not decode overview frames")) == "tool_error"
    assert s.classify(ValueError("bad")) == "tool_error"          # official evaluate.py: any exception -> tool_error
    expect_raise(lambda: runner.check_versions({"pins": {"torch": "0.0"}}), RuntimeError, "not the pinned runtime")
    if importlib.util.find_spec("fla") is None:        # the SynthScope env has no fla: refused for that reason only
        expect_raise(lambda: runner.check_versions({"pins": {"torch": torch.__version__}}), RuntimeError, "(fla)")
    else:
        assert runner.check_versions({"pins": {"torch": torch.__version__}})["torch"] == torch.__version__
    # each video gets its own frame cache, deleted afterwards unless keep_frames
    with tempfile.TemporaryDirectory() as tmp:
        roots = []

        class Frames:
            def __init__(self, root):
                self.root = Path(root)

        sess = object.__new__(runner.Session)
        sess.paths, sess.n_videos = {"work_dir": Path(tmp)}, 0
        sess.media = type("media", (), {"FrameCache": Frames})
        sess.agent = type("agent", (), {})()

        def fake_score(video, verify_direct):
            root = sess.agent.frames.root
            (root / "frames").mkdir(parents=True)
            (root / "frames" / "f.jpg").write_bytes(b"x")
            roots.append(root)
            if video == "bad":
                raise ValueError("boom")
            return {"ok": video}
        sess._score = fake_score
        sess.cfg = {"keep_frames": False}
        assert sess.score("a") == {"ok": "a"} and not roots[-1].exists()
        expect_raise(lambda: sess.score("bad"), ValueError, "boom")
        assert not roots[-1].exists() and roots[0] != roots[1]
        sess.cfg = {"keep_frames": True}
        sess.score("c")
        assert (roots[-1] / "frames" / "f.jpg").exists()
        assert roots[0].name == f"{socket.gethostname()}_{os.getpid()}_1" and roots[0].parent.name == "videos"
    # a killed runner's frame caches are removed by the next runner on the same host; live / foreign ones are kept
    with tempfile.TemporaryDirectory() as tmp:
        videos = Path(tmp, "videos")
        host = socket.gethostname()
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()                                       # reaped: its pid is free
        names = {"stale": f"{host}_{dead.pid}_3", "own": f"{host}_{os.getpid()}_1", "live": f"{host}_{os.getppid()}_2",
                 "other_host": f"not-{host}_{dead.pid}_1", "legacy": f"{dead.pid}_7", "odd": f"{host}_x_1"}
        for name in names.values():
            (videos / name / "frames").mkdir(parents=True)
        (videos / "note.txt").write_text("x")
        removed = runner.sweep_stale_frames(Path(tmp))
        assert [d.name for d in removed] == [names["stale"]] and not (videos / names["stale"]).exists()
        assert all((videos / names[k]).is_dir() for k in names if k != "stale") and (videos / "note.txt").exists()
        assert runner.sweep_stale_frames(Path(tmp, "absent")) == []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "a.bin").write_bytes(b"abc")
        want = {"a.bin": sha256_file(root / "a.bin")}
        memo = {}
        assert runner.verify_files(root, want, "x", memo) == 1 and len(memo) == 1
        expect_raise(lambda: runner.verify_files(root, {"a.bin": "0" * 64}, "x", {}), RuntimeError, "pinned release")
        expect_raise(lambda: runner.verify_files(root, {"b.bin": "0" * 64}, "x", {}), FileNotFoundError, "missing")
        # strict adapters: two adapted layers, each role's file holds A and B for both
        for role in ROLES:
            (root / role).mkdir()
            save_file({f"base_model.model.l{i}.lora_{ab}.weight": torch.zeros(2, 2) for i in range(2) for ab in "AB"},
                      str(root / role / "adapter_model.safetensors"))
        rt = type("rt", (), {"layers": [(None, None, {r: (0, 0, 2.0) for r in ROLES}) for _ in range(2)]})
        loads = [{"adapter": r, "path": str(root / r), "missing": [], "unexpected": []} for r in ROLES]
        assert runner.check_adapters(rt, loads, ROLES, root) == {r: 4 for r in ROLES}
        bad = [dict(x) for x in loads]
        bad[2]["unexpected"] = ["base_model.model.extra.lora_A.weight"]
        expect_raise(lambda: runner.check_adapters(rt, bad, ROLES, root), RuntimeError, "adapter mismatch")
        expect_raise(lambda: runner.check_adapters(rt, loads[:4], ROLES, root), RuntimeError, "expected")
        rt.layers[1][2].pop("common_sense")
        expect_raise(lambda: runner.check_adapters(rt, loads, ROLES, root), RuntimeError, "common_sense: 4 tensors in the file, 2")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        expect_raise(lambda: runner._guarded_connect(sock, ("10.255.255.1", 9)), OSError, "network access refused")
        assert runner._guarded_connect_ex(sock, ("10.255.255.1", 9)) == 111
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        runner._guarded_connect(sock, server.getsockname())                 # loopback: allowed
    finally:
        sock.close()
        server.close()
    print("    ok")


def test_registry_and_engine() -> None:
    print("[D] registry, licences, Space, hint; engine with a mocked runner process ...")
    spec = get_spec("wevis")
    assert (spec.modality, spec.predict_kind, spec.ckpt_kind, spec.checkpoint, spec.weights) == \
        ("video", "video_agent", "official_wevis", "wevisforensics_lora", "official")
    assert spec.agent is WEVIS and spec.transform is None and spec.overrides == [] and spec.vlm is None
    note = input_note(spec)
    assert "overview" in note["label"] and "Audio is not used" in note["explanation"]
    assert "wevis" in hub.NEVER_REDISTRIBUTE
    from interface.backend.registry import _headline
    card = _headline("wevis")                        # the held-out result, not the in-distribution GenBuster row
    assert card["benchmark"].startswith("4 held subsets") and card["match"] == "partial", card
    assert "acc=85.59" in card["ours"] and "AUROC=90.23" in card["ours"], card
    rows = [line for line in Path(ROOT, "LICENSES.md").read_text().splitlines() if line.startswith("| `wevis` |")]
    assert len(rows) == 1 and "Apache-2.0" in rows[0] and "disabled" in rows[0] and "not on the Space" in rows[0], rows
    assert "`Qwen/Qwen3.5-4B` @ `851bf6e8`" in Path(ROOT, "LICENSES.md").read_text()
    fetch = Path(ROOT, "deploy", "hf-space", "fetch_weights.py").read_text()
    assert '"wevis"' not in fetch and "'wevis'" not in fetch
    hint = download_hint()
    assert "hf download tencent/WeVisForensics --revision 9fa025a02dde4f7f7de8883fe17b6f251f39098e" in hint
    assert "hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a" in hint and "43ae1c0" in hint

    saved_env = {k: os.environ.get(k) for k in (ENV_PYTHONPATH, ENV_PYTHON, ENV_CODE, ENV_PADDLEX, ENV_WORK_DIR, "HF_TOKEN")}
    original = backend_config.CHECKPOINTS_DIR, backend_config.CACHE_DIR
    with tempfile.TemporaryDirectory() as tmp:
        ck, cache = Path(tmp, "ck"), Path(tmp, "cache")
        backend_config.CHECKPOINTS_DIR, backend_config.CACHE_DIR = ck, cache
        for k in saved_env:
            os.environ.pop(k, None)
        try:
            engine = Engine(device="cuda")                    # the device is only checked, the GPU work is in the runner
            assert not engine.is_available(spec) and len(missing(WEVIS, ck, cache)) == 5
            det = Detector(spec, device="cpu", auto_download=True)
            expect_raise(det._ensure_weights, FileNotFoundError, "hf download tencent/WeVisForensics")
            # nothing is installed, yet a still image gets the video-only error (not the missing-weights one)
            expect_raise(lambda: det.predict(Image.new("RGB", (8, 8))), ValueError, "video-only detector")
            for rel in ("wevisforensics_lora/main/adapter_config.json", "qwen3.5-4b/config.json",
                        "wevis_paddlex/official_models/PP-OCRv5_server_det/inference.pdiparams",
                        "WeVisForensics/demo/wvf_local.py"):
                Path(ck, rel).parent.mkdir(parents=True, exist_ok=True)
                Path(ck, rel).write_text("{}")
            assert missing(WEVIS, ck, cache) == [f"${ENV_PYTHONPATH} is not set (the isolated transformers 5.23.0 + "
                                                 "PaddleOCR dependency folders)"]
            os.environ[ENV_PYTHONPATH] = os.pathsep.join(["/deps/wevis", "/deps/wevis_ocr"])
            os.environ["HF_TOKEN"] = "secret"
            assert engine.is_available(spec)
            expect_raise(lambda: Engine(device="cpu")._get_model(spec), RuntimeError, "CUDA GPU")
            # a still image is refused before any runner starts (API and engine)
            img = Image.new("RGB", (64, 64))
            png = Path(tmp, "x.png")
            img.save(png)
            det = Detector(spec, device="cuda", auto_download=False)
            for source in (img, png, png.read_bytes()):
                expect_raise(lambda: det.predict(source), ValueError, "video-only detector")
            expect_raise(lambda: engine.predict_image(spec, img), ValueError, "video-only detector")
            expect_raise(lambda: engine.predict_video_frames(spec, [img]), ValueError, "video file itself")
            assert "wevis" not in engine._cache and "wevis" not in det._engine._cache
            fake = Path(tmp, "fake_python")
            fake.write_text(FAKE_RUNNER.format(python=sys.executable))
            fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
            os.environ[ENV_PYTHON] = str(fake)
            res = engine.predict_video_clip(spec, Path(tmp, "clip.mp4"))
            info = engine._cache["wevis"].info
            cfg, env = info["cfg"], info["env"]
            assert cfg["has_audio"] is False and cfg["paths"]["base"] == str(ck / "qwen3.5-4b")
            assert cfg["paths"]["paddlex"] == str(ck / "wevis_paddlex") and cfg["paths"]["work_dir"] == str(cache / "wevis_work")
            assert env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1" and env["HF_TOKEN"] is None
            assert env["PYTHONPATH"].split(os.pathsep) == ["/deps/wevis", "/deps/wevis_ocr", ROOT]
            assert env["WVF_BASE"] == str(ck / "qwen3.5-4b") and env["WVF_OCR_DEVICE"] == "cpu"
            assert env["PADDLE_PDX_CACHE_HOME"] == str(ck / "wevis_paddlex")
            assert res["verdict"] == "fake" and res["logit"] == 15.25 and res["score_source"] == "verdict_token"
            assert res["fake_probability"] == round(1 / (1 + math.exp(-15.25)), 6) and res["status"] == "answered"
            assert res["trace"]["tool_calls"] == [["visual_appearance", [0.0, 3.0]]] and res["cost"]["model_calls"] == 3
            assert "43ae1c0" in engine._status["wevis"] and "audio off" in engine._status["wevis"]
            pid = res["trace"]["pid"]
            inv = engine.predict_video_clip(spec, Path(tmp, "invalid.mp4"))
            assert inv["verdict"] == "unparsed" and inv["fake_probability"] == 0.5 and inv["score_source"] == "unparsed"
            assert inv["trace"]["pid"] == pid                                     # one process, model loaded once
            expect_raise(lambda: engine.predict_video_clip(spec, Path(tmp, "oom.mp4")), WeVisOOMError, "oom on oom.mp4")
            expect_raise(lambda: engine.predict_video_clip(spec, Path(tmp, "tool.mp4")), WeVisToolError, "tool_error")
            expect_raise(lambda: engine.predict_video_clip(spec, Path(tmp, "crash.mp4")), WeVisRunnerError, "exited (code 3)")
            again = engine.predict_video_clip(spec, Path(tmp, "clip.mp4"))       # restarted
            assert again["verdict"] == "fake" and again["trace"]["pid"] != pid
            proc = engine._cache["wevis"]._proc
            engine.release_gpu()
            assert proc.poll() is not None and "wevis" not in engine._cache
        finally:
            backend_config.CHECKPOINTS_DIR, backend_config.CACHE_DIR = original
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    print("    spec, licences, hint; mocked runner: env/config, score, unparsed, OOM / tool / crash errors, restart, release")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:                                              # a zombie still answers signal 0: check its state
        return Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


def gone(pid: int, seconds: float = 10.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.1)
    return False


def test_process_group() -> None:
    print("[E] runner process group: kill paths take the child (OCR worker) with them; close waits for a video ...")
    saved = os.environ.get("FAKE_WEVIS_CHILD")
    os.environ["FAKE_WEVIS_CHILD"] = "1"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp, "fake_python")
            fake.write_text(FAKE_RUNNER.format(python=sys.executable))
            fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
            cfg = replace(WEVIS, timeout=3.0, kill_grace=1.0)
            env = dict(os.environ, **{ENV_PYTHONPATH: "/deps"})

            def make():
                return WeVisRunner(cfg, Path(tmp, "ck"), Path(tmp, "cache"), command=[str(fake), str(RUNNER), "--serve"],
                                   environ=env)
            # 1. per-video timeout: the runner ignores SIGTERM, so SIGKILL to the group is needed
            r = make()
            info = r.start()
            child, pid = info["child"], info["pid"]
            assert alive(child) and os.getpgid(child) == pid != os.getpgid(0)
            expect_raise(lambda: r.score(Path(tmp, "hang.mp4")), WeVisRunnerError, "no answer within 3 s")
            assert gone(pid) and gone(child), "runner or its child survived the timeout kill"
            # 2. close(): shutdown, then the group (the fake runner leaves its child running on exit)
            r = make()
            child = r.start()["child"]
            r.close()
            assert gone(child), "child survived close()"
            # 3. close() while a video is in progress waits for it, then stops the runner
            r = make()
            child = r.start()["child"]
            box = {}
            t = threading.Thread(target=lambda: box.update(res=r.score(Path(tmp, "slow.mp4"))))
            t.start()
            time.sleep(0.5)
            r.close()
            t.join(10)
            assert box["res"]["verdict"] == "fake" and gone(child) and not r.running
            # 4. close(timeout) on a stuck video gives up waiting and kills the group; the video fails
            r = make()
            child = r.start()["child"]
            box = {}

            def stuck():
                try:
                    r.score(Path(tmp, "hang.mp4"))
                except WeVisRunnerError as e:
                    box["err"] = str(e)
            t = threading.Thread(target=stuck)
            t.start()
            time.sleep(0.5)
            r.close(timeout=0.5)
            t.join(10)
            assert "exited" in box.get("err", "") and gone(child)
            # 5. a runner that crashes: its child is killed with it
            r = make()
            child = r.start()["child"]
            expect_raise(lambda: r.score(Path(tmp, "crash.mp4")), WeVisRunnerError, "exited (code 3)")
            assert gone(child), "child survived the runner's crash"
            # 6. SIGKILL only while the group lives: a runner that ignores SIGTERM gets it ...
            r = make()
            r.start()
            sent = []
            r._signal_group = lambda pgid, sig: (sent.append(sig), WeVisRunner._signal_group(pgid, sig))[1]
            r._kill()
            assert sent == [signal.SIGTERM, signal.SIGKILL], sent
        os.environ.pop("FAKE_WEVIS_CHILD", None)
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp, "fake_python")
            fake.write_text(FAKE_RUNNER.format(python=sys.executable))
            fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
            r = WeVisRunner(replace(WEVIS, kill_grace=5.0), Path(tmp, "ck"), Path(tmp, "cache"),
                            command=[str(fake), str(RUNNER), "--serve"], environ=dict(os.environ, **{ENV_PYTHONPATH: "/deps"}))
            # ... one that exits on SIGTERM (group empty, leader reaped) does not: its pgid may be reused
            r.start()
            sent = []
            r._signal_group = lambda pgid, sig: (sent.append(sig), WeVisRunner._signal_group(pgid, sig))[1]
            r._kill()
            assert sent == [signal.SIGTERM], sent
            # 7. close() nulls _proc while a score is reading (its wait timed out): WeVisRunnerError, not AttributeError
            del r._signal_group
            r.start()
            proc = r._proc
            r._proc = None
            os.killpg(proc.pid, signal.SIGKILL)
            expect_raise(lambda: r._read(10.0, proc), WeVisRunnerError, "exited")
            proc.wait()
            r._lines = __import__("queue").Queue()
            expect_raise(lambda: r._read(10.0), WeVisRunnerError, "exited")      # no runner at all
            # 8. a closed client never starts a new runner (an evicted client still held by a request would leak one)
            pid = r.start()["pid"]
            r.close()
            assert gone(pid) and not r.running
            expect_raise(lambda: r.score(Path(tmp, "clip.mp4")), WeVisClosedError, "closed")
            expect_raise(r.start, WeVisClosedError, "closed")
            assert not r.running and r._proc is None
            # 9. a dropped client is collected (the pump thread holds only the pipe) and __del__ stops its runner
            r = WeVisRunner(replace(WEVIS, kill_grace=5.0), Path(tmp, "ck"), Path(tmp, "cache"),
                            command=[str(fake), str(RUNNER), "--serve"], environ=dict(os.environ, **{ENV_PYTHONPATH: "/deps"}))
            pid = r.start()["pid"]
            assert r.score(Path(tmp, "clip.mp4"))["verdict"] == "fake"
            del r
            gc.collect()
            assert gone(pid), "a dropped runner client left its runner process alive"
    finally:
        if saved is None:
            os.environ.pop("FAKE_WEVIS_CHILD", None)
        else:
            os.environ["FAKE_WEVIS_CHILD"] = saved
    print("    timeout kill (TERM ignored -> KILL), close, close while busy, close timeout, crash: no process left; "
          "no SIGKILL to an empty group; a nulled runner raises WeVisRunnerError; a closed client never restarts; a "
          "dropped client stops its runner")


class BlockingAgent:
    """Stands in for a WeVisRunner whose close() waits for a video in progress."""

    def __init__(self):
        self.entered, self.release, self.closed = threading.Event(), threading.Event(), False

    def close(self):
        self.entered.set()
        assert self.release.wait(10)
        self.closed = True


def test_engine_lock() -> None:
    print("[F] engine: an evicted / released agent is closed outside the engine lock ...")
    saved = backend_config.MAX_RESIDENT_MODELS
    backend_config.MAX_RESIDENT_MODELS = 1
    try:
        engine = Engine(device="cpu")
        loaded = []

        def fake_load(spec):                     # stands in for building a detector (the caller holds the lock)
            assert engine._lock.locked()
            loaded.append(spec.key)
            engine._cache[spec.key] = f"model:{spec.key}"
            return engine._cache[spec.key]
        engine._load = fake_load
        # eviction: loading another detector evicts the agent; while its close() waits, the lock is free
        agent = BlockingAgent()
        engine._cache["wevis"] = agent
        box = {}
        t = threading.Thread(target=lambda: box.update(m=engine._get_model(get_spec("clip"))))
        t.start()
        assert agent.entered.wait(10)
        assert engine._lock.acquire(timeout=2), "engine lock held while the evicted agent closes"
        engine._lock.release()
        assert "wevis" not in engine._cache and not loaded           # evicted first, loaded only after the close
        agent.release.set()
        t.join(10)
        assert agent.closed and box["m"] == "model:clip" and loaded == ["clip"]
        assert engine._get_model(get_spec("clip")) == "model:clip" and loaded == ["clip"]   # cached
        # release_gpu: the cache is cleared at once, the agent closed outside the lock
        agent = BlockingAgent()
        engine._cache.clear()
        engine._cache["wevis"] = agent
        t = threading.Thread(target=engine.release_gpu)
        t.start()
        assert agent.entered.wait(10)
        assert engine._lock.acquire(timeout=2), "engine lock held while release_gpu closes the agent"
        assert not engine._cache
        engine._lock.release()
        agent.release.set()
        t.join(10)
        assert agent.closed and not t.is_alive()
        # a runner client closed between _get_model and score (evicted by another request): fetched again once
        engine._cache.clear()

        class ClosedAgent:
            def close(self):
                pass

            def score(self, path, verify_direct=False):
                engine._cache.pop("wevis", None)          # what an eviction did meanwhile
                raise WeVisClosedError("the WeVisForensics runner client was closed")

        class FreshAgent(ClosedAgent):
            def score(self, path, verify_direct=False):
                return {"verdict": "fake", "path": str(path)}
        engine._cache["wevis"] = ClosedAgent()

        def load_fresh(spec):
            engine._cache[spec.key] = FreshAgent()
            return engine._cache[spec.key]
        engine._load = load_fresh
        assert engine.agent_score(get_spec("wevis"), "v.mp4") == {"verdict": "fake", "path": "v.mp4"}
        assert isinstance(engine._cache["wevis"], FreshAgent)
        stuck = ClosedAgent()                               # the same closed client again: surfaced, no loop
        stuck.score = lambda path, verify_direct=False: (_ for _ in ()).throw(WeVisClosedError("closed"))
        engine._cache["wevis"] = stuck
        expect_raise(lambda: engine.agent_score(get_spec("wevis"), "v.mp4"), WeVisClosedError, "closed")
    finally:
        backend_config.MAX_RESIDENT_MODELS = saved
    print("    eviction and release_gpu close the agent with the engine lock free; load after the close; a closed "
          "client is fetched again once")


if __name__ == "__main__":
    test_release()
    test_score()
    test_runner_helpers()
    test_registry_and_engine()
    test_process_group()
    test_engine_lock()
    print("wevis smoke test passed")
