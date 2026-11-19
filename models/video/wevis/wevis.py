"""WeVisForensics (``wevis``): an agentic detector of AI-generated videos, run in its own process.

WeVisForensics (Du et al., arXiv 2611.21373; code Tencent/WeVisForensics, weights HF tencent/WeVisForensics) is an
investigator agent: a Qwen3.5-4B with a LoRA (``main``) looks at a 1 fps overview of the video (at most 16 frames),
calls up to four inspector tools (the same base with the ``visual_appearance``, ``physical_consistency``,
``common_sense`` and ``ocr_consistency`` LoRAs; OCR uses PP-OCRv5 on the CPU) and answers ``real`` or ``fake`` as JSON.

SynthScope runs the authors' in-process transformers backend (``demo/wvf_local.py``) unmodified, in a separate process
(``runner.py``) under its own pinned dependencies, because the official code needs transformers >= 5.13 (pinned here to
5.23.0, the version the fidelity run used) and PaddleOCR 3.7 / paddlepaddle 3.2.2, which are not in the SynthScope
environment. This module is the SynthScope side: the configuration, the files a run needs, and ``WeVisRunner``, which
starts the runner, sends it one video at a time as a JSON line and reads one JSON line back. It imports only the
standard library.

Per video the runner returns:
  * ``verdict``: the official answer ("real" / "fake"), or None when the agent did not answer validly (an invalid
    action, too many steps, an inspector that broke its JSON contract): the official evaluator counts those wrong;
  * ``logit`` = l(fake) - l(real) of the investigator's raw LM-head logits at the verdict token of its final turn
    (``..."content":"fake"``), read DURING the official generation (a forward hook on the LM head; the generated text
    is unchanged), and ``fake_probability`` = sigmoid(logit). With no valid answer: logit 0, P 0.5,
    ``score_source="unparsed"``; with an answer whose verdict token has no same-spelling counterpart: P 0.5,
    ``"unscored"`` (the verdict still counts);
  * ``trace``: every turn's raw text, action, reasoning and tool observation, the tool-call list, the final text;
  * ``cost``: seconds, model calls, generated tokens, peak GPU memory.

Audio is off: every video is run with ``has_audio=False``, the official manifest field (``scripts/evaluate.py`` passes
a manifest's ``has_audio`` to ``ForensicAgent.run``). The audio tools need a gated diarization model and non-commercial
voice models, and in our benchmarks only some fakes carry an audio track while every real is silent, so the audio
flag alone would separate them.

Errors: an out-of-memory error raises ``WeVisOOMError`` (retry the video on a larger GPU; it is neither right nor
wrong); an official ``ToolError`` (a decode / OCR failure, which the official evaluator retries and then leaves out)
raises ``WeVisToolError``; a runner that dies or stops answering raises ``WeVisRunnerError``.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "runner.py"
MANIFEST = HERE / "manifest.json"
REPO_ROOT = HERE.parents[2]

# The verdict word in the investigator's final action, as the official parser reads it (``protocol.py``:
# ``str(action["content"]).strip().lower()``): the last ``"content": "<word>`` in the final turn.
VERDICT_RE = re.compile(r'"content"\s*:\s*"\s*(real|fake)', re.IGNORECASE)


class WeVisError(RuntimeError):
    """A video WeVisForensics could not score; ``kind`` says why."""
    kind = "other"


class WeVisOOMError(WeVisError):
    """CUDA out of memory (long high-resolution clips): retry the video on a GPU with more memory."""
    kind = "oom"


class WeVisToolError(WeVisError):
    """The official ``ToolError`` (decode, OCR or engine failure): the official evaluator retries such a video and then
    leaves it out of the accuracy."""
    kind = "tool_error"


class WeVisRunnerError(WeVisError):
    """The runner process failed to start, died, or did not answer in time."""
    kind = "runner"


ERRORS = {"oom": WeVisOOMError, "tool_error": WeVisToolError, "runner": WeVisRunnerError}


@dataclass(frozen=True)
class WeVisConfig:
    """Where the released files live (folder names under ``$SYNTHSCOPE_CKPT_DIR``, each overridable by an environment
    variable) and the pins the runner checks."""
    adapters: str = "wevisforensics_lora"     # HF tencent/WeVisForensics @ 9fa025a0: 5 LoRA folders + skill_card.json
    base: str = "qwen3.5-4b"                  # HF Qwen/Qwen3.5-4B @ 851bf6e8
    paddlex: str = "wevis_paddlex"            # PaddleX cache home holding official_models/PP-OCRv5_server_{det,rec}
    code: str = "WeVisForensics"              # a checkout of Tencent/WeVisForensics @ 43ae1c0
    has_audio: bool = False                   # passed to ForensicAgent.run for every video (see the module docstring)
    transformers_version: str = "5.23.0"      # the isolated runtime the fidelity run used
    paddle_version: str = "3.2.2"             # 3.3.x crashes in oneDNN on the CPU (PaddleOCR worker)
    verdict_pairs: tuple = (("fake", "real"), ("Fake", "Real"), ("FAKE", "REAL"))
    timeout: float = 3600.0                   # seconds per video before the runner is killed
    ready_timeout: float = 3600.0             # seconds to load (weights sha256 on first use, two 4B copies, OCR worker)


WEVIS = WeVisConfig()

ENV_PYTHONPATH = "SYNTHSCOPE_WEVIS_PYTHONPATH"     # the isolated dependency folders (os.pathsep-separated); required
ENV_PYTHON = "SYNTHSCOPE_WEVIS_PYTHON"             # the interpreter (default: this one)
ENV_CODE = "SYNTHSCOPE_WEVIS_CODE"                 # the official checkout (default: $SYNTHSCOPE_CKPT_DIR/WeVisForensics)
ENV_PADDLEX = "SYNTHSCOPE_WEVIS_PADDLEX"           # PaddleX cache home (default: $SYNTHSCOPE_CKPT_DIR/wevis_paddlex)
ENV_WORK_DIR = "SYNTHSCOPE_WEVIS_WORK_DIR"         # frame cache + OCR worker files (default: <cache dir>/wevis_work)


def load_manifest(path: Path = MANIFEST) -> dict:
    """The pinned release: code commit + file sha256s, adapter / base / PaddleOCR revisions + file sha256s."""
    return json.loads(Path(path).read_text())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def runner_paths(cfg: WeVisConfig, ckpt_dir: Path, cache_dir: Path, environ=None) -> dict:
    """{"adapters", "base", "paddlex", "code", "work_dir"} as absolute paths."""
    environ = os.environ if environ is None else environ
    ckpt_dir = Path(ckpt_dir)

    def under(name: str) -> Path:
        return Path(name) if Path(name).is_absolute() else ckpt_dir / name

    return {"adapters": under(cfg.adapters), "base": under(cfg.base),
            "paddlex": Path(environ.get(ENV_PADDLEX) or under(cfg.paddlex)),
            "code": Path(environ.get(ENV_CODE) or under(cfg.code)),
            "work_dir": Path(environ.get(ENV_WORK_DIR) or Path(cache_dir) / "wevis_work")}


def missing(cfg: WeVisConfig, ckpt_dir: Path, cache_dir: Path, environ=None) -> list:
    """Why the detector cannot run here (empty when every file and the isolated runtime are in place). Only the
    presence is checked; the runner verifies the sha256 of every pinned file when it starts."""
    environ = os.environ if environ is None else environ
    p = runner_paths(cfg, ckpt_dir, cache_dir, environ)
    out = []
    if not environ.get(ENV_PYTHONPATH):
        out.append(f"${ENV_PYTHONPATH} is not set (the isolated transformers {cfg.transformers_version} + PaddleOCR "
                   "dependency folders)")
    checks = [("adapters", p["adapters"] / "main" / "adapter_config.json", "the LoRA adapters (HF tencent/WeVisForensics)"),
              ("base", p["base"] / "config.json", "the base model (HF Qwen/Qwen3.5-4B)"),
              ("paddlex", p["paddlex"] / "official_models" / "PP-OCRv5_server_det" / "inference.pdiparams",
               "the PP-OCRv5 models (PaddleX cache)"),
              ("code", p["code"] / "demo" / "wvf_local.py", "the official code (github.com/Tencent/WeVisForensics)")]
    for _, marker, what in checks:
        if not marker.exists():
            out.append(f"no {what}: {marker} not found")
    return out


def download_hint(cfg: WeVisConfig = WEVIS) -> str:
    m = load_manifest()
    return (f"git clone {m['code']['repo']} {cfg.code} && git -C {cfg.code} checkout {m['code']['commit'][:7]}; "
            f"hf download {m['adapters']['repo']} --revision {m['adapters']['revision']} --local-dir {cfg.adapters}; "
            f"hf download {m['base']['repo']} --revision {m['base']['revision']} --local-dir {cfg.base}; "
            f"the PP-OCRv5 server det/rec models into {cfg.paddlex}/official_models (see models/video/wevis/README.md)")


def runner_env(cfg: WeVisConfig, paths: dict, environ=None) -> dict:
    """The runner's environment: the isolated dependency folders first on PYTHONPATH, offline Hugging Face, the local
    PaddleX models, the official backend's settings (local folders, CPU OCR), and no Hub token."""
    environ = dict(os.environ if environ is None else environ)
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "PYTHONPATH"):
        environ.pop(k, None)
    deps = [p for p in environ.get(ENV_PYTHONPATH, "").split(os.pathsep) if p]
    environ["PYTHONPATH"] = os.pathsep.join(deps + [str(REPO_ROOT)])
    environ.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
        "PADDLE_PDX_CACHE_HOME": str(paths["paddlex"]), "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
        "WVF_BASE": str(paths["base"]), "WVF_ADAPTERS": str(paths["adapters"]),
        "WVF_WORK_DIR": str(paths["work_dir"]), "WVF_OCR_DEVICE": "cpu",
    })
    return environ


class WeVisRunner:
    """Client of one long-lived runner process (``runner.py --serve``): the model loads once (about 3-4 min, plus the
    sha256 check of the weights on first use) and then scores one video per ``score`` call. Thread-safe (one video at a
    time). A runner that died is restarted on the next call. ``command`` replaces the runner command line (tests)."""

    def __init__(self, cfg: WeVisConfig, ckpt_dir: Path, cache_dir: Path, *, command: Optional[list] = None,
                 environ: Optional[dict] = None):
        self.cfg = cfg
        self.paths = runner_paths(cfg, ckpt_dir, cache_dir, environ)
        python = (environ or os.environ).get(ENV_PYTHON) or sys.executable
        self.command = command or [python, str(RUNNER), "--serve"]
        self.env = runner_env(cfg, self.paths, environ)
        self.info: Optional[dict] = None
        self._proc: Optional[subprocess.Popen] = None
        self._lines: "queue.Queue" = queue.Queue()
        self._lock = threading.Lock()
        self._n = 0

    def config_json(self) -> str:
        return json.dumps({"paths": {k: str(v) for k, v in self.paths.items()}, "has_audio": self.cfg.has_audio,
                           "transformers_version": self.cfg.transformers_version,
                           "paddle_version": self.cfg.paddle_version,
                           "verdict_pairs": [list(p) for p in self.cfg.verdict_pairs]})

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _pump(self, proc, lines) -> None:
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    def _read(self, timeout: float) -> dict:
        while True:
            try:
                line = self._lines.get(timeout=timeout)
            except queue.Empty:
                self._kill()
                raise WeVisRunnerError(f"the WeVisForensics runner gave no answer within {timeout:.0f} s (killed)")
            if line is None:
                code = self._proc.wait() if self._proc is not None else None
                raise WeVisRunnerError(f"the WeVisForensics runner exited (code {code}); its log is on stderr")
            try:
                return json.loads(line)
            except json.JSONDecodeError:          # the runner writes only JSON to this pipe; be tolerant anyway
                continue

    def start(self) -> dict:
        self._kill()
        self.paths["work_dir"].mkdir(parents=True, exist_ok=True)
        self._lines = queue.Queue()
        self._proc = subprocess.Popen(self.command + ["--config", self.config_json()], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1, env=self.env,
                                      cwd=str(self.paths["work_dir"]))
        threading.Thread(target=self._pump, args=(self._proc, self._lines), daemon=True).start()
        msg = self._read(self.cfg.ready_timeout)
        if msg.get("event") != "ready":
            self._kill()
            raise ERRORS.get(msg.get("error_kind"), WeVisRunnerError)(
                f"the WeVisForensics runner failed to load: {msg.get('error', msg)}")
        self.info = msg.get("info", {})
        return self.info

    def score(self, path, *, verify_direct: bool = False) -> dict:
        """The runner's result for one video (see the module docstring). ``verify_direct`` also re-generates the final
        investigator turn with a direct ``generate(output_logits=True)`` call and reports the gap difference."""
        with self._lock:
            if not self.running:
                self.start()
            self._n += 1
            req = {"id": self._n, "video": str(Path(path).resolve()), "verify_direct": bool(verify_direct)}
            try:
                self._proc.stdin.write(json.dumps(req) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                self._kill()
                raise WeVisRunnerError(f"the WeVisForensics runner is gone: {e}") from e
            msg = self._read(self.cfg.timeout)
            if msg.get("id") != req["id"]:
                self._kill()
                raise WeVisRunnerError(f"protocol error: answer {msg.get('id')} for request {req['id']}")
            if msg.get("ok"):
                return msg["result"]
            raise ERRORS.get(msg.get("error_kind"), WeVisError)(msg.get("error", "unknown error"))

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        proc.kill()
        proc.wait()

    def close(self) -> None:
        """Ask the runner to exit (it stops its OCR worker), then make sure it is gone."""
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                proc.stdin.flush()
                proc.wait(timeout=60)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                pass
        self._kill()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 — interpreter shutdown
            pass
