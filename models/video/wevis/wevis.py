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

Errors (the official ``scripts/evaluate.py`` records ANY exception from ``agent.run`` as a ``tool_error`` row, retries
it, and leaves rows that still fail out of the strict accuracy's denominator, reporting their count):
  * out of memory: ``WeVisOOMError``, kept apart so the video can be retried on a larger GPU (a 48 GB A6000);
  * every other exception from the agent (the official ``ToolError``: decode / OCR / engine failures, and anything
    else): ``WeVisToolError``. Benchmarks treat it as the official evaluator does: retried, then excluded from the
    accuracy denominator, with the number of excluded videos reported next to the accuracy (Task 2 also reports the
    stricter variant that counts them wrong);
  * a runner that dies or stops answering: ``WeVisRunnerError`` (the runner and its OCR worker are killed together).

The runner runs in its own session (process group): every kill path (per-video timeout, load timeout, protocol error,
``close`` timeout) signals the whole group, so the official CPU OCR worker it starts is never left behind.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
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
    # The runtime the fidelity run used (gate env: SynthScope's torch + peft, vlm_pydeps/wevis, vlm_pydeps/wevis_ocr);
    # the runner refuses another version of any of them. paddlepaddle 3.3.x crashes in oneDNN on the CPU (OCR worker).
    pins: tuple = (("transformers", "5.23.0"), ("flash-linear-attention", "0.6.0"), ("fla-core", "0.6.0"),
                   ("peft", "0.21.3"), ("torch", "2.11.0+cu126"), ("paddlepaddle", "3.2.2"), ("paddleocr", "3.7.0"))
    verdict_pairs: tuple = (("fake", "real"), ("Fake", "Real"), ("FAKE", "REAL"))
    timeout: float = 3600.0                   # seconds per video before the runner is killed
    ready_timeout: float = 3600.0             # seconds to load (weights sha256 on first use, two 4B copies, OCR worker)
    close_timeout: float = 3600.0             # close(): seconds to wait for the video in progress
    kill_grace: float = 10.0                  # seconds between SIGTERM and SIGKILL to the runner's process group


WEVIS = WeVisConfig()

ENV_PYTHONPATH = "SYNTHSCOPE_WEVIS_PYTHONPATH"     # the isolated dependency folders (os.pathsep-separated); required
ENV_PYTHON = "SYNTHSCOPE_WEVIS_PYTHON"             # the interpreter (default: this one)
ENV_CODE = "SYNTHSCOPE_WEVIS_CODE"                 # the official checkout (default: $SYNTHSCOPE_CKPT_DIR/WeVisForensics)
ENV_PADDLEX = "SYNTHSCOPE_WEVIS_PADDLEX"           # PaddleX cache home (default: $SYNTHSCOPE_CKPT_DIR/wevis_paddlex)
ENV_WORK_DIR = "SYNTHSCOPE_WEVIS_WORK_DIR"         # frame cache + OCR worker files (default: <cache dir>/wevis_work)
ENV_KEEP_FRAMES = "SYNTHSCOPE_WEVIS_KEEP_FRAMES"   # 1: keep each video's decoded frames / crops (debugging)


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
        out.append(f"${ENV_PYTHONPATH} is not set (the isolated transformers {dict(cfg.pins)['transformers']} + PaddleOCR "
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
                           "pins": dict(self.cfg.pins),
                           "keep_frames": self.env.get(ENV_KEEP_FRAMES, "") not in ("", "0"),
                           "verdict_pairs": [list(p) for p in self.cfg.verdict_pairs]})

    @property
    def running(self) -> bool:
        proc = self._proc                                # one read: close() may null it from another thread
        return proc is not None and proc.poll() is None

    def _pump(self, proc, lines) -> None:
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    def _read(self, timeout: float, proc: Optional[subprocess.Popen] = None) -> dict:
        """The runner's next JSON line. The runner exiting (EOF, or the process gone while a child it started still
        holds the pipe open) and a timeout both kill its process group and raise ``WeVisRunnerError``. ``proc`` is the
        runner the caller is talking to (default: the current one): ``close()`` may set ``self._proc`` to None from
        another thread (after its wait timed out), which must surface as ``WeVisRunnerError``, never AttributeError."""
        proc = self._proc if proc is None else proc
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                self._kill()
                raise WeVisRunnerError(f"the WeVisForensics runner gave no answer within {timeout:.0f} s (killed)")
            try:
                line = self._lines.get(timeout=min(left, 1.0))
            except queue.Empty:
                if (proc is None or proc.poll() is not None) and self._lines.empty():
                    line = None
                else:
                    continue
            if line is None:
                code = proc.wait() if proc is not None else None
                self._kill()                             # a dead runner's children (the OCR worker) go with it
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
                                      cwd=str(self.paths["work_dir"]), start_new_session=True)
        threading.Thread(target=self._pump, args=(self._proc, self._lines), daemon=True).start()
        msg = self._read(self.cfg.ready_timeout, self._proc)
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
            proc = self._proc                            # close() (after its wait timed out) may null self._proc
            if proc is None:
                raise WeVisRunnerError("the WeVisForensics runner was closed")
            self._n += 1
            req = {"id": self._n, "video": str(Path(path).resolve()), "verify_direct": bool(verify_direct)}
            try:
                proc.stdin.write(json.dumps(req) + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as e:      # ValueError: the pipe was closed under us
                self._kill()
                raise WeVisRunnerError(f"the WeVisForensics runner is gone: {e}") from e
            msg = self._read(self.cfg.timeout, proc)
            if msg.get("id") != req["id"]:
                self._kill()
                raise WeVisRunnerError(f"protocol error: answer {msg.get('id')} for request {req['id']}")
            if msg.get("ok"):
                return msg["result"]
            raise ERRORS.get(msg.get("error_kind"), WeVisError)(msg.get("error", "unknown error"))

    @staticmethod
    def _signal_group(pgid: int, sig) -> bool:
        """Send ``sig`` to the process group; False when the group no longer exists."""
        try:
            os.killpg(pgid, sig)
            return True
        except ProcessLookupError:
            return False

    def _kill(self) -> None:
        """Stop the runner AND everything it started (the official CPU OCR worker): SIGTERM to its process group, then
        SIGKILL after ``kill_grace`` seconds. Also run after a normal exit, for children the runner left behind."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        pgid = proc.pid                                  # start_new_session: the runner leads its own group
        if self._signal_group(pgid, signal.SIGTERM):
            deadline = time.monotonic() + self.cfg.kill_grace
            empty = False
            while time.monotonic() < deadline:
                if proc.poll() is not None and not self._group_alive(pgid):
                    empty = True
                    break
                time.sleep(0.1)
            # Only a group still alive gets SIGKILL: once it is empty and the leader reaped, its id could be reused.
            if not empty and self._group_alive(pgid):
                self._signal_group(pgid, signal.SIGKILL)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            pass

    def _group_alive(self, pgid: int) -> bool:
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def close(self, timeout: Optional[float] = None) -> None:
        """Wait for the video in progress (up to ``timeout``, default ``cfg.close_timeout``), ask the runner to exit (it
        stops its OCR worker), then kill its process group, so nothing it started is left behind."""
        got = self._lock.acquire(timeout=self.cfg.close_timeout if timeout is None else timeout)
        try:
            proc = self._proc
            if proc is None:
                return
            if proc.poll() is None and got:
                try:
                    proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                    proc.stdin.flush()
                    proc.wait(timeout=60)
                except (BrokenPipeError, OSError, ValueError, subprocess.TimeoutExpired):
                    pass
            self._kill()
        finally:
            if got:
                self._lock.release()

    def __del__(self):
        try:
            self.close(timeout=0)
        except Exception:  # noqa: BLE001 — interpreter shutdown
            pass
