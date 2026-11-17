"""Checks the VLM path of ``bench_eval`` (CPU only, no model weights: a stub engine stands in for ``Engine.vlm_score``).

* a vlm spec is scored one image at a time through ``vlm_score`` on the raw decoded file (an RGBA PNG reaches it as RGBA),
* metrics: accuracy from the parsed verdict with an unparsed answer counted wrong, the ``*_drop`` fields leaving it out,
  AUROC / AP from the logit (sklearn, rank-based), the unparsed count and the
  mean generated tokens, equal to a hand computation,
* the output JSON carries ``limit``, the ``vlm`` block and ``per_image``,
* ``--vlm-cache``: a second run scores nothing and gives the identical JSON (apart from date and tag); a changed file
  (other sha1) is scored again; records of another decoding / degradation are kept in their own folders,
* ``--degrade``: the model's ``image_loader`` runs first (transparent -> white), the degradation is applied to its RGB output,
* ``--vlm-decoding official`` passes the seed to every call; greedy passes none,
* the non-VLM ``metrics`` keys are unchanged.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "interface"), str(ROOT / "benchmark")]

import bench_eval                                                    # noqa: E402
from models.vlm.image import vllm_rgb                                 # noqa: E402

# per file stem: (logit, verdict, tokens); r* are reals, f* fakes. r2 is unparsed (logit 0), f1 is a wrong "real".
OUTPUTS = {"r0": (-8.0, "real", 400), "r1": (-30.0, "real", 410), "r2": (0.0, None, 2048), "r3": (3.0, "fake", 500),
           "f0": (9.0, "fake", 450), "f1": (-1.0, "real", 460), "f2": (40.0, "fake", 470), "f3": (41.0, "fake", 480)}
STEMS = ["r0", "r1", "r2", "r3", "f0", "f1", "f2", "f3"]
SPEC: dict = {}                                                      # the stub spec of the latest ``patched`` run


class StubEngine:
    def __init__(self):
        self.device = torch.device("cpu")
        self.calls = []

    def _get_model(self, spec):
        return None

    def vlm_score(self, spec, pil, decoding="greedy", seed=None):
        stem = pil.info["stem"]
        self.calls.append({"stem": stem, "mode": pil.mode, "decoding": decoding, "seed": seed, "pixels": np.asarray(pil)})
        logit, verdict, tok = OUTPUTS[stem]
        return {"logit": logit, "verdict": verdict, "score_source": "unparsed" if verdict is None else "answer",
                "new_tokens": tok, "finished": tok < 2048, "verdict_token": None, "input_size": list(pil.size),
                "response": f"<answer> {verdict} </answer>"}


def keep_stem(fn):
    """Wrap an image -> image function so the stub's ``info["stem"]`` survives it (PIL drops ``info`` on convert/JPEG)."""
    def wrapped(image, *a):
        out = fn(image, *a)
        out.info["stem"] = image.info["stem"]
        return out
    return wrapped


@contextlib.contextmanager
def patched(engine: StubEngine, stem_of: dict, loader_modes: list):
    def loader(image):
        loader_modes.append(image.mode)
        return keep_stem(vllm_rgb)(image)
    cfg = types.SimpleNamespace(image_loader=loader, hf_revision=SPEC.get("revision", "rev1"), system_prompt="sys", user_prompt="usr",
                                chat_template=True, chat_template_kwargs={"enable_thinking": False},
                                decoding={"do_sample": False, "max_new_tokens": 2048},
                                official_decoding={"do_sample": True, "temperature": 0.7, "max_new_tokens": 2048})
    spec = types.SimpleNamespace(predict_kind="vlm", transform=None, vlm=cfg, eval_batch_size=lambda b: 1)
    SPEC["spec"] = spec
    det = types.SimpleNamespace(_engine=engine, spec=spec, _ensure_weights=lambda: None)
    real = (bench_eval.synthscope, bench_eval.media, bench_eval._jpeg)
    raw = bench_eval.media.decode_image_raw

    def decode(data):
        image = raw(data)
        image.info["stem"] = stem_of[hashlib.sha1(data).hexdigest()]
        return image
    bench_eval.synthscope = types.SimpleNamespace(load=lambda model, device=None, **kw: det, __version__="stub")
    bench_eval.media = types.SimpleNamespace(decode_image_raw=decode)
    bench_eval._jpeg = keep_stem(real[2])
    try:
        yield
    finally:
        bench_eval.synthscope, bench_eval.media, bench_eval._jpeg = real


def make_pool(root: Path) -> dict:
    rng = np.random.default_rng(0)
    stem_of = {}
    for sub, prefix in (("0_real", "r"), ("1_fake", "f")):
        (root / "gen" / sub).mkdir(parents=True)
        for i in range(4):
            arr = rng.integers(0, 256, (24, 32, 4 if (prefix, i) == ("r", 0) else 3), dtype=np.uint8)
            if arr.shape[2] == 4:
                arr[..., 3] = 0                                       # fully transparent: white after the official loader
            path = root / "gen" / sub / f"{prefix}{i}.png"
            Image.fromarray(arr).save(path)
            stem_of[hashlib.sha1(path.read_bytes()).hexdigest()] = f"{prefix}{i}"
    return stem_of


def run(tmp: Path, stem_of: dict, name: str, extra: list[str]) -> tuple[dict, StubEngine, list]:
    engine, modes, out = StubEngine(), [], tmp / f"{name}.json"
    with patched(engine, stem_of, modes), contextlib.redirect_stdout(io.StringIO()):
        rc = bench_eval.main(["-m", "stubvlm", "--benchmark", "pergen", "--data", str(tmp / "pool"), "--tag", name,
                              "--out", str(out), "--results-log", str(tmp / "log.jsonl"), "--limit", "4"] + extra)
    assert rc == 0
    return json.loads(out.read_text()), engine, modes


def expected() -> dict:
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    logits = [OUTPUTS[s][0] for s in STEMS]
    return {"ap": round(100 * average_precision_score(labels, logits), 2), "auroc": round(100 * roc_auc_score(labels, logits), 2),
            "real_acc": 50.0, "fake_acc": 75.0, "acc": 62.5, "acc_pooled": 62.5,
            "real_acc_drop": round(200 / 3, 2), "fake_acc_drop": 75.0, "acc_drop": round((round(200 / 3, 2) + 75) / 2, 2),
            "acc_pooled_drop": round(500 / 7, 2), "n_real": 4, "n_fake": 4, "n_unparsed": 1, "n_unfinished": 1,
            "n_logit_verdict_disagree": 0, "mean_tokens": round(sum(OUTPUTS[s][2] for s in STEMS) / 8, 1)}


def check_identity(tmp: Path, stem_of: dict) -> None:
    """The cache folder is keyed by the weights revision, max_new_tokens, the prompts and the code version: changing any
    of them (each alone) gives an empty cache, so stale records are never returned; restoring it finds them again."""
    ident = bench_eval.vlm_identity(SPEC["spec"], "greedy")
    assert ident["weights_revision"] == "rev1" and ident["max_new_tokens"] == 2048 and ident["code_version"]
    assert len(ident["prompt_sha1"]) == 10
    variants = {}
    base = bench_eval.vlm_variant("stubvlm", "greedy", None, None, None, ident)
    cfg = SPEC["spec"].vlm
    changes = {
        "weights revision": lambda: setattr(cfg, "hf_revision", "rev2"),
        "max_new_tokens": lambda: cfg.decoding.__setitem__("max_new_tokens", 4096),
        "system prompt": lambda: setattr(cfg, "system_prompt", "sys2"),
        "user prompt": lambda: setattr(cfg, "user_prompt", "usr2"),
        "chat template kwargs": lambda: cfg.chat_template_kwargs.__setitem__("enable_thinking", True),
    }
    for name, change in changes.items():
        saved = (cfg.hf_revision, cfg.system_prompt, cfg.user_prompt, dict(cfg.chat_template_kwargs), dict(cfg.decoding))
        change()
        variants[name] = bench_eval.vlm_variant("stubvlm", "greedy", None, None, None, bench_eval.vlm_identity(SPEC["spec"], "greedy"))
        cfg.hf_revision, cfg.system_prompt, cfg.user_prompt, cfg.chat_template_kwargs, cfg.decoding = saved
    real_version = bench_eval.vlm_code_version
    try:
        bench_eval.vlm_code_version = lambda: "other-code"
        variants["code version"] = bench_eval.vlm_variant("stubvlm", "greedy", None, None, None,
                                                          bench_eval.vlm_identity(SPEC["spec"], "greedy"))
    finally:
        bench_eval.vlm_code_version = real_version
    variants["--ckpt"] = bench_eval.vlm_variant("stubvlm", "greedy", None, None, None,
                                                bench_eval.vlm_identity(SPEC["spec"], "greedy", "/other/weights"))
    assert all(v != base for v in variants.values()) and len(set(variants.values())) == len(variants), variants
    assert bench_eval.vlm_variant("stubvlm", "greedy", None, None, None, bench_eval.vlm_identity(SPEC["spec"], "greedy")) == base
    # end to end: a run after a revision change finds an empty cache folder and scores every image again
    cache = ["--vlm-cache", str(tmp / "cache")]
    _, eng, _ = run(tmp, stem_of, "same", cache)
    assert eng.calls == []                                           # same identity: all from the cache
    SPEC["revision"] = "rev2"
    try:
        _, eng, _ = run(tmp, stem_of, "newrev", cache)
    finally:
        SPEC["revision"] = "rev1"
    assert len(eng.calls) == 8                                       # new weights revision: nothing reused


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_:
        tmp = Path(tmp_)
        stem_of = make_pool(tmp / "pool")
        cache = ["--vlm-cache", str(tmp / "cache")]

        a, eng, modes = run(tmp, stem_of, "a", cache)
        assert [c["stem"] for c in eng.calls] == STEMS
        assert all(c["decoding"] == "greedy" and c["seed"] is None for c in eng.calls)
        assert eng.calls[0]["mode"] == "RGBA" and modes == []          # raw decode; the scorer applies the loader itself
        m = a["per_generator"]["gen"]
        assert m == expected(), (m, expected())
        assert a["means"] == {"acc": 62.5, "map": m["ap"], "mauroc": m["auroc"]}
        assert a["limit"] == 4 and a["vlm"]["n_unparsed"] == 1 and a["vlm"]["n_images"] == 8
        assert a["vlm"]["acc_pooled"] == 62.5 and a["vlm"]["acc_pooled_drop"] == round(500 / 7, 2)
        assert [r["key"] for r in a["per_image"]] == [f"gen/0_real/r{i}.png" for i in range(4)] + \
            [f"gen/1_fake/f{i}.png" for i in range(4)]

        b, eng, _ = run(tmp, stem_of, "b", cache)
        assert eng.calls == []                                           # everything from the cache
        strip = lambda d: {k: v for k, v in d.items() if k not in ("date", "tag")}   # noqa: E731
        assert strip(a) == strip(b)

        f2 = tmp / "pool" / "gen" / "1_fake" / "f2.png"
        Image.fromarray(np.zeros((24, 32, 3), np.uint8)).save(f2)
        stem_of[hashlib.sha1(f2.read_bytes()).hexdigest()] = "f2"
        _, eng, _ = run(tmp, stem_of, "c", cache)
        assert [c["stem"] for c in eng.calls] == ["f2"]                 # changed file (sha1): scored again

        _, eng, _ = run(tmp, stem_of, "d", cache + ["--vlm-decoding", "official", "--vlm-seed", "7"])
        assert len(eng.calls) == 8 and all(c["decoding"] == "official" and c["seed"] == 7 for c in eng.calls)

        e, eng, modes = run(tmp, stem_of, "e", cache + ["--degrade", "jpeg95"])
        assert len(eng.calls) == 8 and modes[0] == "RGBA" and all(c["mode"] == "RGB" for c in eng.calls)
        assert eng.calls[0]["pixels"].min() > 240                        # transparent -> white, then JPEG q95
        assert e["degrade"] == "jpeg95"
        ident = lambda decoding="greedy": bench_eval.vlm_identity(SPEC["spec"], decoding)    # noqa: E731
        assert sorted(p.name for p in (tmp / "cache").iterdir()) == sorted([
            bench_eval.vlm_variant("stubvlm", "greedy", None, None, None, ident()),
            bench_eval.vlm_variant("stubvlm", "greedy", None, "jpeg95", None, ident()),
            bench_eval.vlm_variant("stubvlm", "official", 7, None, None, ident("official"))])
        folder = tmp / "cache" / bench_eval.vlm_variant("stubvlm", "greedy", None, None, None, ident())
        assert json.loads((folder / "identity.json").read_text()) == ident()
        assert a["vlm"]["identity"] == ident() and "once per generator" in a["vlm"]["metric_note"]
        check_identity(tmp, stem_of)

        plain = bench_eval.metrics(torch.tensor([0.1, 0.9]), torch.tensor([0, 1]))
        assert list(plain) == ["ap", "auroc", "real_acc", "fake_acc", "acc"]   # the non-VLM metrics are unchanged
    print("bench_eval VLM path: all checks passed")


def test_bench_eval_vlm_path() -> None:      # pytest entry point; the file also runs as a script
    main()


if __name__ == "__main__":
    main()
