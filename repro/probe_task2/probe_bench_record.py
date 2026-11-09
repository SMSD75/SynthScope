"""(Copy of vina_bench_record.py for PROBE; the video mode is unused.)
Run SynthScope's benchmark/bench_eval.py or bench_eval_video.py UNCHANGED and also save the per-file P(fake) it
computed, so the scores can be compared file by file with the official evaluate_dino.py / evaluate_resnet.py and the
scores can be compared file by file with the official eval.py (--analysis CSV).

The wrapper only observes: bench_eval's own `score`/`metrics` (image) or the engine's `predict_video_clip` (video) are
wrapped to record their inputs/outputs; bench_eval writes its JSON and MASTER_RESULTS.jsonl line itself.

    python vina_bench_record.py image <scores.json> -- --model probe --benchmark pergen --data D --tag T
    python vina_bench_record.py video <scores.json> -- --model vina_video --benchmark genvideo --data D --tag T
"""
import json
import os
import sys
import time
from pathlib import Path

W = Path(os.environ.get("PROBE_WT", "/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared/probe_wt"))
sys.path[:0] = [str(W / "benchmark"), str(W / "interface"), str(W)]


def run_image(argv, out):
    import bench_eval as be
    gens, cur, rec = [], {}, []
    orig_discover, orig_score, orig_metrics = be.discover, be.score, be.metrics

    def discover(*a, **k):
        res = orig_discover(*a, **k)
        gens.extend(name for name, _, _ in res)
        return res

    def score(det, model, spec, paths, bs, **kw):
        t0 = time.time()
        probs, kept = orig_score(det, model, spec, paths, bs, **kw)
        cur.update(paths=[str(paths[i]) for i in kept], probs=probs.tolist(), sec=time.time() - t0)
        return probs, kept

    def metrics(fake_probs, labels):
        rec.append({"name": gens[len(rec)], "paths": cur["paths"], "probs": cur["probs"],
                    "labels": labels.tolist(), "sec": cur["sec"]})
        return orig_metrics(fake_probs, labels)

    be.discover, be.score, be.metrics = discover, score, metrics
    rc = be.main(argv)
    n = sum(len(r["paths"]) for r in rec)
    sec = sum(r["sec"] for r in rec)
    Path(out).write_text(json.dumps({"argv": argv, "gens": rec}) + "\n")
    print(f"[record] {n} scored images in {sec:.0f} s ({n / max(sec, 1e-9):.2f} img/s incl. decode) -> {out}",
          flush=True)
    return rc


def run_video(argv, out):
    import bench_eval_video as bv
    from backend import engine as engine_mod
    rec = []
    cls = type(engine_mod.get_engine())
    orig, orig_prob = cls.predict_video_clip, cls._video_fake_prob
    last = {}

    def _video_fake_prob(self, *a, **k):          # the unrounded P(fake) (predict_video_clip rounds to 6 decimals)
        p = orig_prob(self, *a, **k)
        last["raw"] = float(p.reshape(-1)[0])
        return p

    def predict_video_clip(self, spec, path, *a, **k):
        t0 = time.time()
        last.clear()
        res = orig(self, spec, path, *a, **k)
        rec.append({"path": str(path), "prob": float(res["fake_probability"]), "prob_raw": last.get("raw"),
                    "sec": time.time() - t0})
        return res

    cls.predict_video_clip, cls._video_fake_prob = predict_video_clip, _video_fake_prob
    rc = bv.main(argv)
    sec = sum(r["sec"] for r in rec)
    Path(out).write_text(json.dumps({"argv": argv, "videos": rec}) + "\n")
    print(f"[record] {len(rec)} scored videos in {sec:.0f} s ({len(rec) / max(sec, 1e-9):.3f} clips/s incl. ffmpeg)"
          f" -> {out}", flush=True)
    return rc


if __name__ == "__main__":
    mode, out, sep, *rest = sys.argv[1:]
    assert sep == "--" and mode in ("image", "video"), __doc__
    sys.exit((run_image if mode == "image" else run_video)(rest, out))
