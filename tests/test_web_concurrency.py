"""The web app scores off the event loop without letting concurrent requests break the engine's model cache (CPU, no
downloads; the bit-identical check uses LTF's released weights when ``$SYNTHSCOPE_CKPT_DIR/ltf.pth`` is present).

* ``Engine.pinned``: a pinned detector is never evicted by the LRU or dropped by ``release_gpu``; a load that finds every
  resident detector pinned goes over the limit and the cache is trimmed back when the pin is released; two requests
  for the same cold detector load it once,
* /api/predict: a slow score (mocked, 1.5 s) does not block a second /api/predict on another detector nor /api/health,
  and the detector is pinned while it is scored,
* /api/score_batch: with ``SYNTHSCOPE_MAX_VIEWS`` set, a view-bag detector (LTF, PROBE) scores at most
  ``SCORE_BATCH_CAP_VIEWS`` items per call and the rest come back with a reason; other detectors keep their caps,
* /api/models lists the paper / code links; the LTF score through /api/predict equals ``Engine.predict_image``.
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import threading
import time
import warnings

import httpx
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "interface")]

from interface.backend import app as web_app                     # noqa: E402
from interface.backend import config as backend_config            # noqa: E402
from interface.backend.engine import Engine                       # noqa: E402
from interface.backend.registry import get_spec                   # noqa: E402


def png_bytes(w: int = 300, h: int = 200) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 130, 140)).save(buf, "PNG")
    return buf.getvalue()


class FakeLoads:
    """Stands in for ``Engine._load`` (the caller holds the engine lock): records each load, optionally slowly."""

    def __init__(self, engine: Engine, delay: float = 0.0):
        self.engine, self.delay, self.calls = engine, delay, []

    def __call__(self, spec):
        self.calls.append(spec.key)
        time.sleep(self.delay)
        model = object()
        self.engine._cache[spec.key] = model
        return model


def test_pins() -> None:
    print("[A] Engine.pinned: no eviction / release of a model in use, over-limit load trimmed, single load ...")
    original = backend_config.MAX_RESIDENT_MODELS
    backend_config.MAX_RESIDENT_MODELS = 1
    try:
        engine = Engine(device="cpu")
        engine._load = FakeLoads(engine)
        a, b, c = get_spec("clip"), get_spec("cnn"), get_spec("npr")
        with engine.pinned(a):
            model_a = engine._get_model(a)
            engine._get_model(b)                           # A is in use: B goes over the limit, A stays
            assert list(engine._cache) == ["clip", "cnn"] and engine._cache["clip"] is model_a
            engine.release_gpu()                           # drops B (unused), keeps A (in use)
            assert list(engine._cache) == ["clip"]
            engine._get_model(b)
        assert list(engine._cache) == ["cnn"], list(engine._cache)   # released: trimmed to the limit, LRU A dropped
        with engine.pinned(c), engine.pinned(c):           # pins count
            engine._get_model(c)
        assert not engine._pins, engine._pins
        engine._get_model(a)                               # unpinned: plain LRU eviction as before
        assert list(engine._cache) == ["clip"]

        engine = Engine(device="cpu")                      # two concurrent requests for one cold detector: one load
        engine._load = loads = FakeLoads(engine, delay=0.3)
        got = []
        threads = [threading.Thread(target=lambda: got.append(engine._get_model(a))) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert loads.calls == ["clip"] and got[0] is got[1], (loads.calls, got)
    finally:
        backend_config.MAX_RESIDENT_MODELS = original
    print("    pinned model kept through an over-limit load and release_gpu; trimmed after; one load for two threads")


async def _concurrent(client: httpx.AsyncClient) -> dict:
    t0 = time.perf_counter()
    done = {}

    async def post(model: str):
        r = await client.post("/api/predict", data={"model": model}, files={"file": ("x.png", png_bytes(), "image/png")})
        done[model] = (time.perf_counter() - t0, r.status_code)

    async def health():
        await asyncio.sleep(0.2)
        r = await client.get("/api/health")
        done["health"] = (time.perf_counter() - t0, r.status_code)

    async def fast():
        await asyncio.sleep(0.2)                           # starts while the slow score is running
        await post("cnn")

    await asyncio.gather(post("ltf"), fast(), health())
    return done


def test_event_loop_not_blocked() -> None:
    print("[B] /api/predict: a slow score does not block other requests ...")
    engine = web_app.get_engine()
    original = engine.is_available, engine.predict_image
    pinned_during = {}

    def fake_predict(spec, pil):
        pinned_during[spec.key] = engine._pins[spec.key]
        if spec.key == "ltf":
            time.sleep(1.5)                                # a slow CPU score
        return {"model": spec.key, "fake_probability": 0.25, "verdict": "real"}

    engine.is_available, engine.predict_image = (lambda spec: True), fake_predict
    try:
        transport = httpx.ASGITransport(app=web_app.app)

        async def run():
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await _concurrent(client)

        done = asyncio.run(run())
    finally:
        engine.is_available, engine.predict_image = original
    assert all(code == 200 for _, code in done.values()), done
    assert done["ltf"][0] >= 1.5, done
    assert done["cnn"][0] < 1.0 and done["health"][0] < 1.0, done      # not queued behind the 1.5 s score
    assert pinned_during == {"ltf": 1, "cnn": 1} and not engine._pins, (pinned_during, engine._pins)
    print(f"    slow ltf {done['ltf'][0]:.2f} s; cnn answered at {done['cnn'][0]:.2f} s, health at "
          f"{done['health'][0]:.2f} s; each detector pinned while scored")


def test_score_batch_cap() -> None:
    print("[C] /api/score_batch: view-bag detectors capped when SYNTHSCOPE_MAX_VIEWS is set ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient
    engine = web_app.get_engine()
    original = engine.is_available, web_app._run_on_dataset_item, backend_config.MAX_VIEWS
    scored = []

    def fake_item(engine_, spec, dataset, rel):
        scored.append(rel)
        return {"fake_probability": 0.5, "ground_truth": "real"}

    engine.is_available, web_app._run_on_dataset_item = (lambda spec: True), fake_item
    try:
        backend_config.MAX_VIEWS = 0
        assert web_app.score_batch_cap(get_spec("ltf")) == web_app.SCORE_BATCH_CAP == 200
        backend_config.MAX_VIEWS = 50
        caps = {k: web_app.score_batch_cap(get_spec(k)) for k in ("ltf", "probe", "probe_resnet", "aide", "spai", "clip",
                                                                   "veritaspp")}
        assert caps == {"ltf": 4, "probe": 4, "probe_resnet": 4, "aide": 200, "spai": 200, "clip": 200,
                        "veritaspp": 20}, caps
        client = TestClient(web_app.app)
        rels = [f"r{i}.png" for i in range(6)]
        out = client.post("/api/score_batch", json={"dataset": "d", "model": "ltf", "rels": rels}).json()
        assert scored == rels[:4] and out["cap"] == 4, (scored, out)
        assert [("error" in it) for it in out["items"]] == [False] * 4 + [True] * 2, out["items"]
        assert "at most 4 items" in out["items"][-1]["error"] and "SYNTHSCOPE_MAX_VIEWS" in out["items"][-1]["error"]
        scored.clear()
        out = client.post("/api/score_batch", json={"dataset": "d", "model": "clip", "rels": rels}).json()
        assert scored == rels and all("error" not in it for it in out["items"])
    finally:
        engine.is_available, web_app._run_on_dataset_item, backend_config.MAX_VIEWS = original
    print("    ltf / probe: 4 items per call (the rest explained); aide, spai, clip, VLMs unchanged; no cap when unset")


def test_links_and_identical_score() -> None:
    print("[D] /api/models links; LTF via /api/predict == Engine.predict_image ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient
    client = TestClient(web_app.app)
    models = {m["key"]: m for m in client.get("/api/models").json()["models"]}
    assert models["ltf"]["links"] == {"paper": "https://arxiv.org/abs/2611.04529",
                                      "code": "https://github.com/HuiGuanLab/LTF"}, models["ltf"]["links"]
    assert all(url.startswith("https://") for m in models.values() for url in m["links"].values())
    engine = web_app.get_engine()
    spec = get_spec("ltf")
    if not engine.is_available(spec):
        print("    links ok; identical-score check skipped ($SYNTHSCOPE_CKPT_DIR/ltf.pth not found)")
        return
    data = png_bytes(300, 200)
    via_api = client.post("/api/predict", data={"model": "ltf"}, files={"file": ("x.png", data, "image/png")}).json()
    direct = engine.predict_image(spec, web_app.media.decode_image(data))
    assert via_api["fake_probability"] == direct["fake_probability"], (via_api["fake_probability"], direct)
    print(f"    links ok; LTF P(fake) {direct['fake_probability']} through the worker thread and directly")


if __name__ == "__main__":
    test_pins()
    test_event_loop_not_blocked()
    test_score_batch_cap()
    test_links_and_identical_score()
    print("web concurrency test passed")
