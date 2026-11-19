"""WeVisForensics (``wevis``) fidelity: SynthScope's engine vs the authors' transformers backend, on the same clips, in
the same isolated runtime, the same way (``has_audio=False``, greedy). Run by repro/wevis_fidelity.sbatch.

    reference  the official backend: ``import wvf_local`` (demo/wvf_local.py, unmodified) and
               ``asyncio.run(wvf_local.AGENT.run(video, has_audio=False))`` per clip, nothing wrapped; runs under the
               isolated dependencies (PYTHONPATH = $SYNTHSCOPE_WEVIS_PYTHONPATH + the official checkout)
    ours       SynthScope: ``Engine.agent_score(spec, video, verify_direct=True)`` per clip (the runner process, its
               sha256 + strict-adapter checks, the verdict logit read during generation, and a DIRECT
               ``generate(output_logits=True)`` call on the final investigator turn's inputs), then
               ``Engine.predict_video_clip`` on the first clip again (repeat determinism + the engine result fields)
    compare    per clip: answer, status, tool-call trace (skill + window), every turn's raw text, final text identical;
               ids of the direct call identical and logit gap difference exactly 0.0; correctness vs the label;
               agreement with the recorded vLLM verdicts of the demo cases; seconds, model calls, peak GPU memory, OOMs
"""
import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
G = Path("/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared")
REF = Path(os.environ.get("SYNTHSCOPE_WEVIS_CODE", G / "vlm_ref" / "WeVisForensics"))
CLIPS = Path(os.environ.get("WEVIS_CLIPS", G / "wevis_gate" / "stage" / "clips"))
CASES = json.loads((REF / "demo" / "examples" / "cases.json").read_text(encoding="utf-8"))
AUDIO_TOOLS = ("voice_profile", "asr_analysis")

# The 6 gate clips (Task 0), the 7 demo cases whose recorded vLLM trajectories use no audio tool, and 5 more GenVideo
# fakes (short HD clips of 4 generators + one 74 s 1080p WildScrape clip with an audio track, the long-clip memory case).
GATE = [
    ("gate", "real", "GenVideo/test/Sora/0_real/00000_msrvtt_1.mp4"),
    ("gate", "real", "GenVideo/test/Sora/0_real/00001_msrvtt_2.mp4"),
    ("gate", "real", "GenVideo/test/Sora/0_real/00005_msrvtt_6.mp4"),
    ("gate", "fake", "GenVideo/test/Sora/1_fake/00000_sora_1.mp4"),
    ("gate", "fake", "VideoPhy/test/Dream-Machine/1_fake/00000_02335_3783b0e2-3546-40a1-be7e-8f44fda5d8ef_raw_video_0.mp4_video0ebc55cd30cbb472aab1a01c62c913daa.mp4"),
    ("gate", "fake", "VidProM/test/Pika/1_fake/00000_pika-001820a9-a6ba-51b0-95bb-1bdfeb0b8f5a.mp4"),
]
EXTRA = [
    ("extra", "fake", "GenVideo/test/Gen2/1_fake/00000_gen2_1.mp4"),
    ("extra", "fake", "GenVideo/test/WildScrape/1_fake/00000_D1.mp4"),
    ("extra", "fake", "GenVideo/test/MoonValley/1_fake/00000_MoonValley_1.mp4"),
    ("extra", "fake", "GenVideo/test/HotShot/1_fake/00000_hotshot_1.mp4"),
    ("extra", "fake", "GenVideo/test/WildScrape/1_fake/00360_D174.mp4"),
]


def clips():
    out = [(s, label, CLIPS / rel) for s, label, rel in GATE]
    out += [("demo", c["label"], REF / "demo" / "examples" / c["video"]) for c in CASES
            if not any(st.get("skill") in AUDIO_TOOLS for st in c["steps"])]
    out += [(s, label, CLIPS / rel) for s, label, rel in EXTRA]
    limit = int(os.environ.get("WEVIS_FID_LIMIT") or 0)     # a spot check: only the first N clips
    return out[:limit] if limit else out


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str) + "\n")


def oom(exc) -> bool:
    seen = []
    while exc is not None and exc not in seen:
        seen.append(exc)
        if "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError":
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def reference(out_path):
    """The official backend, unmodified: every video through wvf_local.AGENT.run(video, has_audio=False). The official
    modules are imported here, not at the top: importing wvf_local loads the model, and only this phase runs with the
    official checkout and the isolated dependencies on the path."""
    sys.path[:0] = [str(REF / "demo"), str(REF)]
    t0 = time.monotonic()
    import wvf_local                                   # official: builds RUNTIME + AGENT at import
    from parity_check import calls_from_steps          # official helper
    import transformers
    env = dict(transformers=transformers.__version__, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
               load_seconds=round(time.monotonic() - t0, 1),
               weights_gib=round(torch.cuda.memory_allocated() / 2 ** 30, 2))
    print("REF ENV", json.dumps(env), flush=True)
    rows = []
    for subset, label, path in clips():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t1 = time.monotonic()
        row = dict(subset=subset, label=label, video=str(path))
        try:
            traj = asyncio.run(wvf_local.AGENT.run(str(path), has_audio=False))
            torch.cuda.synchronize()
            row.update(answer=traj.answer, status=traj.status, tool_calls=calls_from_steps(traj.steps),
                       raws=[s["raw"] for s in traj.steps], steps=traj.steps, duration=traj.duration,
                       has_audio=traj.has_audio)
        except Exception as exc:  # noqa: BLE001 — recorded; OOM or a tool error on this clip
            traceback.print_exc()
            row.update(error=f"{type(exc).__name__}: {exc}"[:1000], oom=oom(exc))
            wvf_local.RUNTIME.active = None             # rebuild every layer on the next switch
            torch.cuda.empty_cache()
        row.update(seconds=round(time.monotonic() - t1, 1),
                   peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
        rows.append(row)
        print("REF", Path(path).name[:60], json.dumps({k: row.get(k) for k in ("label", "answer", "status", "tool_calls",
                                                                              "seconds", "peak_alloc_gib", "error")},
                                                      default=str), flush=True)
        write(out_path, dict(env=env, rows=rows))
    print("REF DONE", flush=True)


def ours(out_path):
    """SynthScope's engine: the runner process, verdict logit during generation, a direct call to compare it with. The
    SynthScope modules are imported here: this phase runs in the plain SynthScope environment (the reference phase
    does not have it on its path)."""
    sys.path[:0] = [str(ROOT), str(ROOT / "interface")]
    from interface.backend.engine import Engine
    from interface.backend.registry import get_spec
    from models.video.wevis.wevis import WeVisError

    engine = Engine(device="cuda")
    spec = get_spec("wevis")
    t0 = time.monotonic()
    engine._get_model(spec)
    info = dict(engine._cache["wevis"].info, engine_load_seconds=round(time.monotonic() - t0, 1),
                status=engine._status["wevis"])
    print("OURS INFO", json.dumps(info, default=str), flush=True)
    rows = []
    for subset, label, path in clips():
        t1 = time.monotonic()
        row = dict(subset=subset, label=label, video=str(path))
        try:
            res = engine.agent_score(spec, path, verify_direct=True)
            row.update(answer=res["verdict"], status=res["status"], tool_calls=res["trace"]["tool_calls"],
                       raws=[s["raw"] for s in res["trace"]["steps"]], result=res)
        except WeVisError as exc:
            row.update(error=f"{type(exc).__name__}: {exc}"[:1000], error_kind=exc.kind)
        row["wall_seconds"] = round(time.monotonic() - t1, 1)
        rows.append(row)
        r = row.get("result") or {}
        print("OURS", Path(path).name[:60], json.dumps(dict(label=label, answer=row.get("answer"), logit=r.get("logit"),
                                                            direct=r.get("direct"), tool_calls=row.get("tool_calls"),
                                                            cost={k: (r.get("cost") or {}).get(k) for k in
                                                                  ("seconds", "model_calls", "peak_alloc_gib")},
                                                            error=row.get("error")), default=str), flush=True)
        write(out_path, dict(info=info, rows=rows))
    first = clips()[0][2]
    repeat = engine.predict_video_clip(spec, first)
    write(out_path, dict(info=info, rows=rows, repeat=dict(video=str(first), result=repeat)))
    engine.release_gpu()
    print("OURS DONE", flush=True)


def compare(ref_path, ours_path, out_path, log_name):
    ref, mine = json.loads(Path(ref_path).read_text()), json.loads(Path(ours_path).read_text())
    case_answer = {c["video"]: c["answer"] for c in CASES}
    rows = []
    for a, b in zip(ref["rows"], mine["rows"]):
        assert a["video"] == b["video"], (a["video"], b["video"])
        r = b.get("result") or {}
        d = r.get("direct") or {}
        cost = r.get("cost") or {}
        tool = lambda calls: [[s, list(w)] for s, w in calls] if calls is not None else None   # noqa: E731
        row = dict(subset=a["subset"], video=Path(a["video"]).name, label=a["label"],
                   reference=dict(answer=a.get("answer"), status=a.get("status"), tool_calls=tool(a.get("tool_calls")),
                                  seconds=a["seconds"], peak_alloc_gib=a["peak_alloc_gib"], error=a.get("error"),
                                  oom=a.get("oom")),
                   synthscope=dict(answer=b.get("answer"), status=b.get("status"), tool_calls=tool(b.get("tool_calls")),
                                   logit=r.get("logit"), fake_probability=r.get("fake_probability"),
                                   score_source=r.get("score_source"), verdict_token=r.get("verdict_token"),
                                   verdict_step=r.get("verdict_step"), pair_softmax_mass=r.get("pair_softmax_mass"),
                                   argmax_is_emitted=r.get("argmax_is_emitted"), seconds=cost.get("seconds"),
                                   wall_seconds=b["wall_seconds"], model_calls=cost.get("model_calls"),
                                   generated_tokens=cost.get("generated_tokens"),
                                   peak_alloc_gib=cost.get("peak_alloc_gib"),
                                   peak_reserved_gib=cost.get("peak_reserved_gib"),
                                   audio_stream_present=(r.get("trace") or {}).get("audio_stream_present"),
                                   error=b.get("error"), error_kind=b.get("error_kind")),
                   direct=d)
        both = "error" not in a and "error" not in b
        row["same_answer"] = both and a["answer"] == b["answer"]
        row["same_status"] = both and a["status"] == b["status"]
        row["same_tool_calls"] = both and tool(a["tool_calls"]) == tool(b["tool_calls"])
        row["same_raw_texts"] = both and a["raws"] == b["raws"]
        row["same_final_text"] = both and bool(a["raws"]) and a["raws"][-1] == b["raws"][-1]
        row["correct"] = b.get("answer") == a["label"] if "error" not in b else None
        if row["video"] in case_answer:
            row["recorded_vllm_answer"] = case_answer[row["video"]]
            row["agrees_with_recorded"] = b.get("answer") == case_answer[row["video"]]
        rows.append(row)
    ok = [r for r in rows if r["reference"]["error"] is None and r["synthscope"]["error"] is None]
    direct = [r["direct"] for r in ok if r["direct"]]
    secs = [r["synthscope"]["seconds"] for r in ok]
    summary = dict(
        clips=len(rows), compared=len(ok),
        identical_answer=sum(r["same_answer"] for r in ok), identical_status=sum(r["same_status"] for r in ok),
        identical_tool_calls=sum(r["same_tool_calls"] for r in ok),
        identical_raw_texts_every_turn=sum(r["same_raw_texts"] for r in ok),
        identical_final_text=sum(r["same_final_text"] for r in ok),
        direct_calls=len(direct), direct_ids_identical=sum(bool(d.get("ids_identical")) for d in direct),
        direct_logit_max_abs_diff=max((abs(d["diff"]) for d in direct if "diff" in d), default=None),
        direct_logit_all_exactly_equal=all(d.get("diff") == 0.0 for d in direct) if direct else None,
        correct=sum(bool(r["correct"]) for r in ok),
        demo_agree_with_recorded_vllm=sum(bool(r.get("agrees_with_recorded")) for r in rows if "recorded_vllm_answer" in r),
        demo_cases=sum("recorded_vllm_answer" in r for r in rows),
        seconds_per_video_mean=round(sum(secs) / len(secs), 1) if secs else None,
        seconds_per_video_mean_excl_first=round(sum(secs[1:]) / len(secs[1:]), 1) if len(secs) > 1 else None,
        seconds_per_video_range=[min(secs), max(secs)] if secs else None,
        reference_seconds_mean=round(sum(r["reference"]["seconds"] for r in ok) / len(ok), 1) if ok else None,
        peak_alloc_gib_max=max((r["synthscope"]["peak_alloc_gib"] for r in ok), default=None),
        peak_reserved_gib_max=max((r["synthscope"]["peak_reserved_gib"] or 0 for r in ok), default=None),
        oom=[r["video"] for r in rows if r["synthscope"].get("error_kind") == "oom" or r["reference"].get("oom")],
        errors=[(r["video"], r["reference"]["error"], r["synthscope"]["error"]) for r in rows
                if r["reference"]["error"] or r["synthscope"]["error"]],
        unparsed=sum(r["synthscope"]["score_source"] == "unparsed" for r in ok),
        unscored=sum(r["synthscope"]["score_source"] == "unscored" for r in ok),
    )
    rep = mine.get("repeat") or {}
    rr = rep.get("result") or {}
    first = next((r for r in mine["rows"] if r["video"] == rep.get("video")), None)
    if first and rr:
        summary["repeat_first_clip_identical"] = (rr.get("logit") == round(first["result"]["logit"], 6)
                                                  and rr["trace"]["tool_calls"] == first["result"]["trace"]["tool_calls"]
                                                  and [s["raw"] for s in rr["trace"]["steps"]] == first["raws"])
    out = dict(
        what="WeVisForensics fidelity: SynthScope engine (models/video/wevis runner) vs the official transformers backend "
             "(demo/wvf_local.py, unmodified), same isolated runtime, same clips, has_audio=False, greedy",
        reference_backend="Tencent/WeVisForensics demo/wvf_local.py @ 43ae1c0 (in-process transformers); the official "
                          "multi-GPU vLLM evaluator is NOT reproduced (no vLLM here)",
        log=log_name, reference_env=ref.get("env"), synthscope_info=mine.get("info"),
        repeat=dict(video=rep.get("video"), engine_result={k: rr.get(k) for k in
                    ("fake_probability", "verdict", "logit", "score_source", "status", "elapsed_ms", "checkpoint_status")}),
        summary=summary, clips=rows)
    write(out_path, out)
    print("SUMMARY", json.dumps(summary, default=str), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["reference", "ours", "compare", "list"])
    ap.add_argument("--out")
    ap.add_argument("--reference")
    ap.add_argument("--ours")
    ap.add_argument("--log", default="")
    args = ap.parse_args()
    if args.phase == "list":
        for subset, label, path in clips():
            print(subset, label, path, path.exists())
    elif args.phase == "reference":
        reference(args.out)
    elif args.phase == "ours":
        ours(args.out)
    else:
        compare(args.reference, args.ours, args.out, args.log)


if __name__ == "__main__":
    main()
