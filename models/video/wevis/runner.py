"""Isolated runner for WeVisForensics (``wevis``): the official in-process transformers backend
(Tencent/WeVisForensics ``demo/wvf_local.py``, unmodified) in its own process.

Started by ``WeVisRunner`` (models/video/wevis/wevis.py) with the SynthScope interpreter and ``PYTHONPATH`` = the isolated
dependency folders ($SYNTHSCOPE_WEVIS_PYTHONPATH: transformers 5.23.0, flash-linear-attention 0.6.0, huggingface_hub,
tokenizers, safetensors; paddlepaddle 3.2.2 + PaddleOCR 3.7 for the CPU OCR worker) + this repository. Nothing is
installed into the SynthScope environment.

    python models/video/wevis/runner.py --serve --config '<json>'

Protocol (JSON lines; this process writes nothing else to its stdout, its logs go to stderr):
    -> {"event": "ready", "info": {...}}  once loaded, or {"event": "error", "error_kind", "error"}
    <- {"id": N, "video": PATH, "verify_direct": false}
    -> {"id": N, "ok": true, "result": {...}}  or  {"id": N, "ok": false, "error_kind": "oom"|"tool_error", ...}
       (any exception from agent.run that is not out of memory is a tool_error, as in the official evaluate.py)
    <- {"cmd": "shutdown"}

Loading (each step fails loudly):
  1. versions: every pinned package (WeVisConfig.pins: transformers 5.23.0, flash-linear-attention / fla-core 0.6.0,
     peft 0.21.3, torch 2.11.0+cu126, paddlepaddle 3.2.2, paddleocr 3.7.0) at its pinned version, fla importable;
  2. sha256 of every pinned file (models/video/wevis/manifest.json): the official code files (commit 43ae1c0), the five
     adapters + skill card (tencent/WeVisForensics @ 9fa025a0), the base (Qwen/Qwen3.5-4B @ 851bf6e8) and the PP-OCRv5
     models. Weight checks are remembered per (path, size, mtime) in <work_dir>/sha256_ok.json;
  3. no network: Hugging Face offline (also for the OCR worker, which inherits the environment), local folders only,
     and a socket guard in THIS process (not in the OCR worker subprocess, nor in non-Python clients) that refuses
     every non-loopback connection (the OCR worker is reached on 127.0.0.1);
  4. ``import wvf_local`` (the official module builds its Runtime: base + working copy + every role's LoRA factors,
     the agent and the CPU OCR worker); every ``PeftModel.load_adapter`` result is recorded and must have no missing and
     no unexpected adapter key, and each role's LoRA factor count must equal its file's tensor count (strict).

Per video: ``asyncio.run(wvf_local.AGENT.run(video, has_audio=False))``, as ``scripts/evaluate.py`` runs a manifest row
(``duration`` is probed by the official code). ``Runtime.main_generate`` / ``Runtime.chat`` are wrapped to count and
time the model calls (inputs and outputs untouched); during each investigator generation a forward hook on the LM head
keeps the raw logits of the verdict tokens, the log-sum-exp and the argmax of every step. After the run the verdict
token of the final turn is located and l(fake) - l(real) is read there (``models.vlm.vlm`` helpers, as the image VLMs).
"""
import argparse
import asyncio
import gc
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import shutil
import socket
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from models.video.wevis.wevis import VERDICT_RE, load_manifest, sha256_file  # noqa: E402
from models.vlm.vlm import LogitRecorder, gap_fields, offset_step, token_pieces, verdict_pair_map  # noqa: E402

LOOPBACK = {"127.0.0.1", "::1", "localhost"}
REFUSED: list = []
_CONNECT, _CONNECT_EX = socket.socket.connect, socket.socket.connect_ex


def _remote(sock, address) -> bool:
    return sock.family in (socket.AF_INET, socket.AF_INET6) and address[0] not in LOOPBACK


def _guarded_connect(sock, address):
    if _remote(sock, address):
        REFUSED.append(str(address))
        raise OSError(f"wevis runner: network access refused ({address}); every file must be local")
    return _CONNECT(sock, address)


def _guarded_connect_ex(sock, address):
    if _remote(sock, address):
        REFUSED.append(str(address))
        return 111                                           # ECONNREFUSED
    return _CONNECT_EX(sock, address)


class VerdictRecorder(LogitRecorder):
    """``LogitRecorder`` (the raw LM-head logits at ``ids`` of every generated step, = ``output_logits``) that also keeps
    each step's log-sum-exp and argmax over the whole vocabulary."""

    def __enter__(self):
        self.lse, self.argmax = [], []
        return super().__enter__()

    def _hook(self, module, inputs, output):
        super()._hook(module, inputs, output)
        last = output[:, -1].to(torch.float32)
        self.lse.append(torch.logsumexp(last, -1))
        self.argmax.append(last.argmax(-1))

    def extras(self) -> tuple:
        """(log-sum-exp [steps], argmax [steps]) on the CPU."""
        if not self.lse:
            return None, None
        return torch.cat(self.lse).cpu(), torch.cat(self.argmax).cpu()


class AdapterLoads:
    """Records the result of every ``PeftModel.load_adapter`` call (the official Runtime loads all five roles)."""

    def __init__(self):
        self.loads = []

    def install(self):
        from peft import PeftModel
        original = PeftModel.load_adapter

        def load_adapter(model, model_id, adapter_name, *args, **kwargs):
            result = original(model, model_id, adapter_name, *args, **kwargs)
            self.loads.append({"adapter": adapter_name, "path": str(model_id),
                               "missing": list(getattr(result, "missing_keys", [])),
                               "unexpected": list(getattr(result, "unexpected_keys", []))})
            return result

        PeftModel.load_adapter = load_adapter


def emit(stream, msg: dict) -> None:
    stream.write(json.dumps(msg, default=str) + "\n")
    stream.flush()


def protocol_stream():
    """A private copy of stdout for the protocol; fd 1 then points at stderr, so prints of the official code, its
    libraries and child processes cannot corrupt the protocol."""
    stream = os.fdopen(os.dup(1), "w", buffering=1)
    sys.stdout.flush()
    os.dup2(2, 1)
    return stream


def check_versions(cfg: dict) -> dict:
    """The installed version of every pinned package (``WeVisConfig.pins``: the gate / fidelity runtime); any other
    version is refused, since the traces depend on the kernels (transformers, fla, torch) and on the OCR stack."""
    have = {name: importlib.metadata.version(name) for name in cfg["pins"]}
    have["torch"] = torch.__version__
    wrong = {name: (have[name], want) for name, want in cfg["pins"].items() if have[name] != want}
    if wrong:
        raise RuntimeError("not the pinned runtime (installed, pinned): " + json.dumps(wrong) + "; put the isolated "
                           "dependency folders first on $SYNTHSCOPE_WEVIS_PYTHONPATH")
    if importlib.util.find_spec("fla") is None:
        raise RuntimeError("flash-linear-attention (fla) is not importable: the gated-delta kernels the fidelity run used")
    have["causal_conv1d"] = importlib.util.find_spec("causal_conv1d") is not None
    return have


def verify_files(root: Path, files: dict, what: str, memo: dict) -> int:
    """sha256 of every pinned file under ``root``; a file already verified at the same (size, mtime) is skipped."""
    for rel, want in files.items():
        path = root / rel
        if not path.exists():
            raise FileNotFoundError(f"{what}: {path} is missing")
        st = path.stat()
        key = f"{path.resolve()}\0{st.st_size}\0{st.st_mtime_ns}"
        if memo.get(key) == want:
            continue
        got = sha256_file(path)
        if got != want:
            raise RuntimeError(f"{what}: {path} has sha256 {got}, the pinned release has {want}")
        memo[key] = want
    return len(files)


def verify_release(paths: dict) -> dict:
    manifest = load_manifest()
    memo_path = Path(paths["work_dir"]) / "sha256_ok.json"
    try:
        memo = json.loads(memo_path.read_text())
    except (OSError, json.JSONDecodeError):
        memo = {}
    code = manifest["code"]
    out = {"code_commit": code["commit"],
           "code_files": verify_files(Path(paths["code"]), code["files"], "official code", {}),
           "adapters_revision": manifest["adapters"]["revision"],
           "adapter_files": verify_files(Path(paths["adapters"]), manifest["adapters"]["files"], "adapters", memo),
           "base_revision": manifest["base"]["revision"],
           "base_files": verify_files(Path(paths["base"]), manifest["base"]["files"], "base model", memo),
           "paddlex_revision": manifest["paddlex"]["revision"],
           "paddlex_files": verify_files(Path(paths["paddlex"]), manifest["paddlex"]["files"], "PP-OCRv5 models", memo)}
    tmp = memo_path.with_name(f".sha256_ok.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(memo, indent=0))
    os.replace(tmp, memo_path)
    return out


def check_adapters(rt, loads: list, roles: tuple, adapter_root: Path) -> dict:
    """Strict adapter loading: every role loaded once, no missing / unexpected adapter key, and each role's LoRA factors
    in the runtime (A and B per adapted layer) as many as its file's tensors."""
    names = [r["adapter"] for r in loads]
    if sorted(names) != sorted(roles):
        raise RuntimeError(f"adapters loaded {names}, expected {list(roles)}")
    bad = [r for r in loads if r["missing"] or r["unexpected"]]
    if bad:
        raise RuntimeError(f"adapter mismatch (missing / unexpected keys): "
                           f"{[(r['adapter'], r['missing'][:4], r['unexpected'][:4]) for r in bad]}")
    counts = {}
    for role in roles:
        with safe_open(str(adapter_root / role / "adapter_model.safetensors"), "pt") as f:
            n_file = len(list(f.keys()))
        n_loaded = 2 * sum(role in per_role for _, _, per_role in rt.layers)
        if n_file != n_loaded:
            raise RuntimeError(f"adapter {role}: {n_file} tensors in the file, {n_loaded} LoRA factors loaded")
        counts[role] = n_file
    return counts


def frames_dir_name(n: int) -> str:
    """A video's frame-cache folder under ``<work_dir>/videos``: ``<host>_<pid>_<n>``. The host and pid say which runner
    owns it (the work folder may be on storage shared by several nodes)."""
    return f"{socket.gethostname()}_{os.getpid()}_{n}"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def sweep_stale_frames(work_dir: Path) -> list:
    """Remove the frame caches of runners on THIS host that no longer run (a runner killed on a timeout, a crash or the
    OOM killer never reaches its per-video cleanup). A folder of a live process, of another host, or not named by
    ``frames_dir_name`` is kept. Returns the removed folders."""
    videos = Path(work_dir) / "videos"
    if not videos.is_dir():
        return []
    host, removed = socket.gethostname(), []
    for d in sorted(videos.iterdir()):
        parts = d.name.rsplit("_", 2)
        if not d.is_dir() or len(parts) != 3 or parts[0] != host or not (parts[1].isdigit() and parts[2].isdigit()):
            continue
        pid = int(parts[1])
        if pid == os.getpid() or _pid_alive(pid):
            continue
        shutil.rmtree(d, ignore_errors=True)
        removed.append(d)
    return removed


def chain(exc):
    seen = []
    while exc is not None and exc not in seen:
        seen.append(exc)
        exc = exc.__cause__ or exc.__context__
    return seen


class Session:
    """The loaded official backend plus the per-video bookkeeping."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.paths = {k: Path(v) for k, v in cfg["paths"].items()}
        self.calls: list = []
        self.n_videos = 0
        t0 = time.monotonic()
        swept = [] if cfg.get("keep_frames") else sweep_stale_frames(self.paths["work_dir"])
        if swept:
            print(f"[wevis runner] removed {len(swept)} stale frame cache(s) of runners that no longer run",
                  file=sys.stderr, flush=True)
        self.versions = check_versions(cfg)
        self.release = verify_release(self.paths)
        socket.socket.connect, socket.socket.connect_ex = _guarded_connect, _guarded_connect_ex
        os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        code = self.paths["code"]
        sys.path[:0] = [str(code / "demo"), str(code)]
        loads = AdapterLoads()
        loads.install()
        self.wvf = importlib.import_module("wvf_local")         # official: builds RUNTIME + AGENT, starts the OCR worker
        self.parity = importlib.import_module("parity_check")   # official helpers (calls_from_steps)
        self.loop_mod = importlib.import_module("main_agent.loop")
        self.media = importlib.import_module("core.media")
        self.common = importlib.import_module("core.common")
        self.rt, self.agent = self.wvf.RUNTIME, self.wvf.AGENT
        self.adapter_tensors = check_adapters(self.rt, loads.loads, self.wvf.ROLES, self.paths["adapters"])
        self.tok = self.rt.tokenizer
        self.pair_map = verdict_pair_map(self.tok, [tuple(p) for p in cfg["verdict_pairs"]])
        self.record_ids = sorted({i for pair in self.pair_map.values() for i in pair})
        self.column = {tid: c for c, tid in enumerate(self.record_ids)}
        self.head = self.rt.work.get_output_embeddings()
        self._main, self._chat = self.rt.main_generate, self.rt.chat
        self.rt.main_generate, self.rt.chat = self._wrapped_main, self._wrapped_chat
        self.info = {
            "versions": self.versions, "release": self.release, "adapter_tensors": self.adapter_tensors,
            "gpu": torch.cuda.get_device_name(0), "attn": getattr(self.rt.work.config, "_attn_implementation", None),
            "weights_gib": round(torch.cuda.memory_allocated() / 2 ** 30, 2),
            "load_seconds": round(time.monotonic() - t0, 1), "has_audio": bool(cfg["has_audio"]),
            "verdict_pair_ids": sorted({tuple(p) for p in self.pair_map.values()}),
            "network_refused": list(REFUSED)}

    # -- the official Runtime's two entry points, wrapped (inputs and outputs untouched) -----------------------------
    def _wrapped_main(self, prompt_ids, images, max_new_tokens):
        t0 = time.monotonic()
        with VerdictRecorder(self.head, self.record_ids) as rec:
            ids, finish = self._main(prompt_ids, images, max_new_tokens)
        lse, argmax = rec.extras()
        self.calls.append(dict(role="main", seconds=round(time.monotonic() - t0, 3), prompt_tokens=len(prompt_ids),
                               images=len(images or []), completion_tokens=len(ids), finish=finish,
                               _prompt_ids=list(prompt_ids), _images=list(images or []), _ids=list(ids),
                               _max_new_tokens=max_new_tokens, _rows=rec.stacked(), _lse=lse, _argmax=argmax))
        return ids, finish

    def _wrapped_chat(self, adapter, messages, max_new_tokens):
        t0 = time.monotonic()
        result = self._chat(adapter, messages, max_new_tokens)
        n_img = sum(1 for m in messages if not isinstance(m["content"], str) for p in m["content"] if p["type"] == "image")
        self.calls.append(dict(role=adapter or "base(asr)", seconds=round(time.monotonic() - t0, 3), images=n_img,
                               prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
                               finish=result.finish_reason))
        return result

    # -- scoring ---------------------------------------------------------------------------------------------------
    def verdict_score(self, call: dict) -> dict:
        """l(fake) - l(real) at the verdict token of the final investigator turn ``call``."""
        gen = call["_ids"]
        pieces = token_pieces(self.tok, gen)
        text = "".join(pieces)
        match = None
        for match in VERDICT_RE.finditer(text):
            pass
        step = None if match is None else offset_step(gen, pieces, text, match.start(1), self.pair_map)
        if step is None:
            return {"logit": 0.0, "fake_probability": 0.5, "score_source": "unscored", "verdict_step": None,
                    "verdict_token": None}
        fake_id, real_id = self.pair_map[gen[step]]
        row = call["_rows"][step].float()
        gap = float(row[self.column[fake_id]] - row[self.column[real_id]])     # float32, as models.vlm.vlm.verdict_score
        l_fake, l_real = float(row[self.column[fake_id]]), float(row[self.column[real_id]])
        lse = float(call["_lse"][step])
        return {**gap_fields(gap), "score_source": "verdict_token", "verdict_step": step,
                "verdict_token": self.tok.convert_ids_to_tokens(gen[step]),
                "pair_tokens": self.tok.convert_ids_to_tokens([fake_id, real_id]),
                "logit_fake": l_fake, "logit_real": l_real,
                "pair_softmax_mass": math.exp(l_fake - lse) + math.exp(l_real - lse),
                "argmax_is_emitted": int(call["_argmax"][step]) == gen[step]}

    def direct_call(self, call: dict, score: dict) -> dict:
        """The final investigator turn again, as a direct ``generate(..., output_logits=True)`` on the same inputs (the
        input construction of ``Runtime.main_generate``), and the gap at the same step from its logits."""
        rt = self.rt
        ids = torch.tensor([call["_prompt_ids"]], dtype=torch.long)
        inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
        if call["_images"]:
            pixels = rt.processor.image_processor(images=list(call["_images"]), return_tensors="pt")
            inputs.update(pixel_values=pixels["pixel_values"], image_grid_thw=pixels["image_grid_thw"])
            if rt.needs_type_ids:
                inputs["mm_token_type_ids"] = (ids == rt.image_token_id).int()
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with rt.lock, torch.inference_mode():
            rt._activate("main")
            out = rt.work.generate(**inputs, max_new_tokens=call["_max_new_tokens"], do_sample=False, temperature=None,
                                   top_p=None, top_k=None, repetition_penalty=1.0, eos_token_id=sorted(rt.eos),
                                   output_logits=True, return_dict_in_generate=True)
        regen = out.sequences[0, inputs["input_ids"].shape[1]:].tolist()
        res = {"ids_identical": regen == call["_ids"]}
        step = score.get("verdict_step")
        if step is not None and len(out.logits) > step:
            fake_id, real_id = self.pair_map[call["_ids"][step]]
            row = out.logits[step][0].float()
            direct = float(row[fake_id] - row[real_id])
            res.update(logit_direct=direct, logit_recorded=score["logit"], diff=score["logit"] - direct,
                       argmax_is_emitted=int(row.argmax()) == call["_ids"][step])
        del out
        torch.cuda.empty_cache()
        return res

    def score(self, video: str, verify_direct: bool = False) -> dict:
        """One video with its own frame cache (decoded frames + OCR crops), deleted afterwards unless ``keep_frames``:
        the official cache is keyed by path + size + mtime, so web uploads (a new temporary file each) would otherwise
        grow it without bound. The cache is only a cache: frames are decoded deterministically either way. A runner that is
        killed mid-video leaves its folder behind; the next runner on the host removes it (``sweep_stale_frames``, not
        with ``keep_frames``)."""
        self.n_videos += 1
        frames_root = self.paths["work_dir"] / "videos" / frames_dir_name(self.n_videos)
        self.agent.frames = self.media.FrameCache(frames_root)
        try:
            return self._score(video, verify_direct)
        finally:
            if not self.cfg.get("keep_frames"):
                shutil.rmtree(frames_root, ignore_errors=True)

    def _score(self, video: str, verify_direct: bool) -> dict:
        self.calls = []
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        audio_stream = bool(self.media.probe_media(video)[1])
        traj = asyncio.run(self.agent.run(video, has_audio=bool(self.cfg["has_audio"])))
        torch.cuda.synchronize()
        seconds = time.monotonic() - t0
        mains = [c for c in self.calls if c["role"] == "main"]
        if traj.status == "answered" and mains:
            score = self.verdict_score(mains[-1])
        else:
            score = {"logit": 0.0, "fake_probability": 0.5, "score_source": "unparsed", "verdict_step": None,
                     "verdict_token": None}
        public = [{k: v for k, v in c.items() if not k.startswith("_")} for c in self.calls]
        steps = traj.steps
        result = {
            **score, "verdict": traj.answer, "status": traj.status,
            "trace": {"tool_calls": [[skill, list(span)] for skill, span in self.parity.calls_from_steps(steps)],
                      "steps": steps, "final_text": steps[-1]["raw"] if steps else None,
                      "duration": traj.duration, "has_audio": traj.has_audio, "audio_stream_present": audio_stream,
                      "overview_timestamps": traj.overview_timestamps, "error": traj.error},
            "cost": {"seconds": round(seconds, 2), "model_calls": len(public),
                     "main_turns": sum(c["role"] == "main" for c in public),
                     "inspector_calls": sum(c["role"] != "main" for c in public),
                     "generated_tokens": sum(c["completion_tokens"] for c in public),
                     "prompt_tokens": sum(c["prompt_tokens"] for c in public),
                     "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                     "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2 ** 30, 2),
                     "calls": public},
        }
        if verify_direct and mains and traj.status == "answered":
            result["direct"] = self.direct_call(mains[-1], score)
        return result

    def classify(self, exc) -> str:
        errs = chain(exc)
        if any(isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower() for e in errs):
            return "oom"
        return "tool_error"                     # as the official evaluate.py: any exception from agent.run

    def recover(self) -> None:
        """After a failed video: drop its tensors, and force the next role switch to rebuild every adapted layer from
        the read-only base (an error inside ``Runtime._activate`` may have left some layers half switched)."""
        self.calls = []
        self.rt.active = None
        gc.collect()
        torch.cuda.empty_cache()


def serve(cfg: dict) -> int:
    proto = protocol_stream()
    try:
        session = Session(cfg)
    except Exception as exc:  # noqa: BLE001 — reported to the client, which raises it
        traceback.print_exc()
        emit(proto, {"event": "error", "error_kind": "runner", "error": f"{type(exc).__name__}: {exc}"})
        return 1
    print(f"[wevis runner] ready: {json.dumps(session.info, default=str)}", file=sys.stderr, flush=True)
    emit(proto, {"event": "ready", "info": session.info})
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        if req.get("cmd") == "shutdown":
            break
        try:
            result = session.score(req["video"], bool(req.get("verify_direct")))
            msg = {"id": req["id"], "ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001 — one video failed; the runner keeps serving
            kind = session.classify(exc)
            traceback.print_exc()
            session.recover()
            msg = {"id": req["id"], "ok": False, "error_kind": kind, "error": f"{type(exc).__name__}: {exc}"[:2000]}
        emit(proto, msg)
    ocr = getattr(session.agent, "ocr", None)
    if hasattr(ocr, "close"):
        ocr.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serve", action="store_true", required=True)
    ap.add_argument("--config", required=True, help="JSON from WeVisRunner.config_json()")
    args = ap.parse_args(argv)
    return serve(json.loads(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
