"""The web app scores off the event loop without letting concurrent requests break the engine's model cache (CPU, no
downloads; the bit-identical check uses LTF's released weights when ``$SYNTHSCOPE_CKPT_DIR/ltf.pth`` is present).

* ``Engine.pinned``: a pinned detector is never evicted by the LRU or dropped by ``release_gpu``; a load that finds every
  resident detector pinned goes over the limit and the cache is trimmed back when the pin is released; two requests
  for the same cold detector load it once,
* /api/predict: a slow score (mocked, 1.5 s) does not block a second /api/predict on another detector nor /api/health,
  and the detector is pinned while it is scored,
* /api/score_batch: with ``SYNTHSCOPE_MAX_VIEWS`` set, a view-bag detector (LTF, PROBE) scores at most
  ``SCORE_BATCH_CAP_VIEWS`` items per call and the rest come back with a reason; other detectors keep their caps,
* /api/models lists the paper / code links; the LTF score through /api/predict equals ``Engine.predict_image``,
* ``SYNTHSCOPE_MAX_CONCURRENT_SCORES``: N concurrent requests on N detectors never score more than the limit at once,
* a VLM request waiting for ``_GPU_LOCK`` is not pinned, so a generation's ``release_gpu`` drops the idle VLM,
* a race (many threads x detectors, ``release_gpu`` firing, small ``MAX_RESIDENT_MODELS``, mocked models): no eviction
  mid-use, no unpinned use, no double load, no hang; and a failing ``close()`` while trimming does not fail a request.
"""

from __future__ import annotations

import asyncio
import io
import os
import random
import sys
import threading
import time
import warnings

import httpx
import torch
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
        print(f"    links ok; SKIP identical-score check: LTF is not available here (needs ltf.pth in "
              f"{backend_config.CHECKPOINTS_DIR} and the pinned CLIP backbone; set SYNTHSCOPE_CKPT_DIR to run it)")
        return
    data = png_bytes(300, 200)
    via_api = client.post("/api/predict", data={"model": "ltf"}, files={"file": ("x.png", data, "image/png")}).json()
    direct = engine.predict_image(spec, web_app.media.decode_image(data))
    assert via_api["fake_probability"] == direct["fake_probability"], (via_api["fake_probability"], direct)
    print(f"    links ok; LTF P(fake) {direct['fake_probability']} through the worker thread and directly")


def test_score_slots() -> None:
    print("[E] SYNTHSCOPE_MAX_CONCURRENT_SCORES=2: never more than 2 scores at once ...")
    keys = ["clip", "cnn", "npr", "safe", "aide", "co_spy"]
    engine = web_app.get_engine()
    original = engine.is_available, engine.predict_image, backend_config.MAX_CONCURRENT_SCORES
    state = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def fake_predict(spec, pil):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.3)
        with lock:
            state["now"] -= 1
        return {"model": spec.key, "fake_probability": 0.25, "verdict": "real"}

    engine.is_available, engine.predict_image = (lambda spec: True), fake_predict
    backend_config.MAX_CONCURRENT_SCORES = 2
    try:
        async def run():
            transport = httpx.ASGITransport(app=web_app.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                t0 = time.perf_counter()

                async def post(k):
                    r = await client.post("/api/predict", data={"model": k},
                                          files={"file": ("x.png", png_bytes(), "image/png")})
                    return r.status_code

                async def health():
                    await asyncio.sleep(0.1)
                    r = await client.get("/api/health")
                    return r.status_code, time.perf_counter() - t0
                return await asyncio.gather(*[post(k) for k in keys], health())

        *codes, (health_code, health_at) = asyncio.run(run())
    finally:
        engine.is_available, engine.predict_image, backend_config.MAX_CONCURRENT_SCORES = original
    assert codes == [200] * len(keys) and health_code == 200, codes
    assert state["peak"] == 2, state                      # the limit is reached and never exceeded
    assert health_at < 0.3, health_at                     # waiting requests do not block the page
    print(f"    {len(keys)} requests on {len(keys)} detectors: peak {state['peak']} scoring at once; health at "
          f"{health_at:.2f} s")


def test_vlm_pin_order() -> None:
    print("[F] a VLM waiting for _GPU_LOCK is not pinned; release_gpu drops it ...")
    engine = Engine(device="cpu")
    spec = get_spec("fakevlm")
    engine._cache["fakevlm"] = object()                   # resident from an earlier request
    seen = {}

    def fake_predict(spec_, pil):
        seen["pinned_while_scoring"] = engine._pins["fakevlm"]
        return {"model": "fakevlm", "fake_probability": 0.5, "verdict": "fake"}

    engine.predict_image = fake_predict
    with web_app._GPU_LOCK:                               # a heavy local generation holds the GPU
        worker = threading.Thread(target=web_app._score_upload, args=(engine, spec, "image", png_bytes(), "x.png"))
        worker.start()
        time.sleep(0.3)                                   # the request is now waiting for the lock
        assert worker.is_alive() and not engine._pins["fakevlm"], engine._pins
        engine.release_gpu()                              # the generation frees the GPU first
        assert "fakevlm" not in engine._cache, list(engine._cache)
    worker.join(10)
    assert not worker.is_alive() and seen == {"pinned_while_scoring": 1} and not engine._pins, (seen, engine._pins)
    print("    idle waiting VLM not pinned and dropped by release_gpu; pinned once it holds the lock")


class FakeModel:
    def __init__(self, key):
        self.key = key


class RaceEngine:
    """Mocked loads / forwards on a real Engine (real transforms, predict_image, pins), recording every violation."""

    def __init__(self, engine: Engine):
        self.engine, self.errors, self.uses, self.lock = engine, [], 0, threading.Lock()
        engine._load, engine._fake_prob = self.load, self.prob
        engine.is_available = lambda spec: True

    def load(self, spec):                                 # the caller holds engine._lock
        if spec.key in self.engine._cache:
            self.errors.append(f"double load of {spec.key}")
        time.sleep(0.01 * random.random())
        model = self.engine._cache[spec.key] = FakeModel(spec.key)
        return model

    def prob(self, model, spec, x):
        if self.engine._cache.get(spec.key) is not model:
            self.errors.append(f"{spec.key} evicted before its forward")
        if not self.engine._pins[spec.key]:
            self.errors.append(f"{spec.key} used unpinned")
        time.sleep(0.01 * random.random())
        if self.engine._cache.get(spec.key) is not model:
            self.errors.append(f"{spec.key} evicted during its forward")
        with self.lock:
            self.uses += 1
        return torch.tensor([0.3])


def test_race() -> None:
    print("[G] race: 24 threads x 6 detectors, release_gpu every 5 ms, MAX_RESIDENT_MODELS 1 / 2 / 3 ...")
    keys = ["clip", "cnn", "npr", "ltf", "aide", "safe"]
    img = Image.new("RGB", (300, 200), (10, 20, 30))
    original = backend_config.MAX_RESIDENT_MODELS
    random.seed(0)
    try:
        for limit in (1, 2, 3):
            backend_config.MAX_RESIDENT_MODELS = limit
            engine = Engine(device="cpu")
            race = RaceEngine(engine)
            stop = threading.Event()

            def worker(i):
                rnd = random.Random(i)
                for _ in range(12):
                    spec = get_spec(rnd.choice(keys))
                    with engine.pinned(spec):
                        engine.predict_image(spec, img)

            def releaser():
                while not stop.is_set():
                    engine.release_gpu()
                    time.sleep(0.005)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(24)]
            rel = threading.Thread(target=releaser)
            rel.start()
            for t in threads:
                t.start()
            for t in threads:
                t.join(120)
            stop.set()
            rel.join(10)
            assert not any(t.is_alive() for t in threads), "hung worker"
            assert not race.errors, sorted(set(race.errors))[:5]
            assert race.uses == 24 * 12 and not engine._pins and len(engine._cache) <= limit, \
                (race.uses, dict(engine._pins), list(engine._cache))

        class Runner:                                     # an agent runner whose close() fails (pinned()'s trim)
            closed = []

            def __init__(self, name, fail):
                self.name, self.fail = name, fail

            def close(self):
                Runner.closed.append(self.name)
                if self.fail:
                    raise RuntimeError("runner close failed")

        backend_config.MAX_RESIDENT_MODELS = 1
        engine = Engine(device="cpu")
        a = get_spec("clip")
        with engine.pinned(a):                            # over the limit while pinned: two runners + clip resident
            engine._cache["r1"] = Runner("r1", fail=True)
            engine._cache["r2"] = Runner("r2", fail=False)
            engine._cache.move_to_end("r1", last=False)
            engine._cache.move_to_end("r2", last=False)
            engine._cache["clip"] = object()
        assert sorted(Runner.closed) == ["r1", "r2"] and list(engine._cache) == ["clip"], (Runner.closed,
                                                                                            list(engine._cache))
    finally:
        backend_config.MAX_RESIDENT_MODELS = original
    print("    0 violations, 0 hangs, pins empty and cache within the limit each time; a failing close() is logged "
          "and the other runners are still closed")


if __name__ == "__main__":
    test_pins()
    test_event_loop_not_blocked()
    test_score_batch_cap()
    test_links_and_identical_score()
    test_score_slots()
    test_vlm_pin_order()
    test_race()
    print("web concurrency test passed")
