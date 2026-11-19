"""CPU test of repro/video_agent_bench/trace_check.py and compare.py on bench_eval_video outputs made with mock engines
(the agent mock of test_bench_eval_video_agent, the plain detector mock of test_bench_eval_video_cache).

Checks:
* trace_check: the result JSONs written by bench_eval_video trace to the captured log (RESULT line incl. the agent's
  acc_strict / pooled / error counts, and every agent per-video record = its last per-video log line); a JSON whose
  means, or one agent record's logit, was edited no longer traces;
* compare: the agent's acc / acc_strict / AUROC, the comparators' best / median and the agent's ranks
  (1 + the number of comparators strictly better), the internal-only exclusion of the best public comparator, and the
  GenBuster paper block (pooled accuracy, Wilson interval, Table 11 groups, deltas against the paper's numbers).
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "interface"), str(ROOT / "benchmark"), str(ROOT / "tests")]

import test_bench_eval_video_agent as ta                                         # noqa: E402
import test_bench_eval_video_cache as tc                                         # noqa: E402

bev = ta.bev
VAB = ROOT / "repro" / "video_agent_bench"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, VAB / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


trace_check, compare = load("trace_check"), load("compare")


class PlainSpec2(tc.MockSpec):
    key = "mock2"


def run_agent(data: Path, cache: Path, out: Path, tag: str, log: io.StringIO) -> None:
    eng = ta.AgentEngine()
    bev.get_engine, bev.get_spec = (lambda: eng), (lambda key: ta.AgentSpec)
    with contextlib.redirect_stdout(log):
        common = ["-m", "wevis", "--benchmark", "pergen", "--data", str(data), "--cache-dir", str(cache), "--tag", tag]
        for k in range(2):
            assert bev.main(common + ["--shard", f"{k}/2"]) == 0
        assert bev.main(common + ["--cache-only", "--out", str(out)]) == 0


def run_plain(spec, data: Path, out: Path, tag: str, log: io.StringIO, ckpt: Path) -> None:
    eng = tc.MockEngine(ckpt)
    bev.get_engine, bev.get_spec = (lambda: eng), (lambda key: spec)
    with contextlib.redirect_stdout(log):
        assert bev.main(["-m", "waverep", "--benchmark", "pergen", "--data", str(data), "--tag", tag, "--out", str(out),
                         "--per-video"]) == 0


def test_trace_and_compare() -> None:
    saved = bev.get_engine, bev.get_spec, bev.REPRO, bev.RESULTS_LOG
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bev.REPRO, bev.RESULTS_LOG = tmp, tmp / "M.jsonl"
        try:
            res, log = tmp / "results", io.StringIO()
            ta.make_tree(tmp / "gv")
            (res / "genvideo").mkdir(parents=True)
            run_agent(tmp / "gv", tmp / "c_gv", res / "genvideo" / "wevis.json", "vab_genvideo_wevis", log)
            ckpt = tmp / "m.pth"
            ckpt.write_bytes(b"w")
            tc.make_tree(tmp / "plain", tc.MockSpec)
            run_plain(tc.MockSpec, tmp / "plain", res / "genvideo" / "d3.json", "vab_genvideo_d3", log, ckpt)
            run_plain(PlainSpec2, tmp / "plain", res / "genvideo" / "vina_video.json", "vab_genvideo_vina_video", log, ckpt)
            # a GenBuster-like tree: id_ / ood_ / wild_ groups sharing one real pool
            gb = tmp / "gb"
            reals = ["r1__real__-15.0.mp4", "r2__fake__4.0.mp4", "r3__real__-12.0.mp4"]
            for g, fk in (("id_a", ["a1__fake__9.0.mp4", "a2__fake__8.0.mp4"]), ("ood_sora", ["s1__real__-3.0.mp4"]),
                          ("wild_v", ["w1__fake__7.0.mp4"])):
                ta.touch(gb / g / "0_real", reals)
                ta.touch(gb / g / "1_fake", fk)
            (res / "genbuster").mkdir()
            run_agent(gb, tmp / "c_gb", res / "genbuster" / "wevis.json", "vab_genbuster_wevis", log)
            (tmp / "logs").mkdir()
            (tmp / "logs" / "wevis_test_1.out").write_text(log.getvalue())

            with contextlib.redirect_stdout(io.StringIO()) as out:
                rc = trace_check.main(["--logs", str(tmp / "logs"), "--results", str(res)])
            assert rc == 0, out.getvalue()
            assert "4 result JSONs trace to their logs" in out.getvalue(), out.getvalue()

            # comparison
            with contextlib.redirect_stdout(io.StringIO()):
                assert compare.main(["--results", str(res), "--out", str(tmp / "cmp.json")]) == 0
            c = json.loads((tmp / "cmp.json").read_text())
            row = next(r for r in c["subsets"] if r["benchmark"] == "genvideo")
            agent = json.loads((res / "genvideo" / "wevis.json").read_text())
            d3 = json.loads((res / "genvideo" / "d3.json").read_text())
            assert row["wevis"]["acc"] == agent["means"]["acc"] and row["wevis"]["acc_strict"] == agent["means"]["acc_strict"]
            assert row["best_acc"]["value"] == d3["means"]["acc"] and row["best_acc"]["tied"] == ["d3", "vina_video"]
            assert row["best_public_acc"]["detector"] == "d3"                      # vina_video is internal-only
            assert row["median_acc"] == d3["means"]["acc"]
            assert row["wevis"]["rank_acc"] == 1 + (d3["means"]["acc"] > agent["means"]["acc"]) * 2
            assert row["wevis"]["rank_acc_of"] == 3 and "acc_pooled" in row["comparators"]["d3"]
            pc = c["paper_comparison"]["genbuster"]["ours_subsample"]
            gbr = json.loads((res / "genbuster" / "wevis.json").read_text())["means"]["pooled"]
            # 3 reals (r2 called fake) + 4 fakes (s1 called real): 5/7 right
            assert pc["n"] == 7 and pc["acc"] == gbr["acc"] == round(500 / 7, 2)
            assert pc["delta_acc_vs_paper"] == round(gbr["acc"] - 97.7, 2) and len(pc["acc_wilson95"]) == 2
            assert pc["table11"]["id"] == 100.0 and pc["table11"]["ood"] == {"sora": 0.0} and pc["table11"]["wild"] == 100.0
            assert pc["table11"]["real_acc"] == round(200 / 3, 2)

            # a tampered JSON no longer traces
            agent["means"]["acc"] += 1
            (res / "genvideo" / "wevis.json").write_text(json.dumps(agent))
            g = json.loads((res / "genbuster" / "wevis.json").read_text())
            g["per_video"][0]["logit"] += 0.5
            (res / "genbuster" / "wevis.json").write_text(json.dumps(g))
            with contextlib.redirect_stdout(io.StringIO()) as out:
                rc = trace_check.main(["--logs", str(tmp / "logs"), "--results", str(res)])
            assert rc == 1 and out.getvalue().count("MISMATCH") == 2, out.getvalue()
        finally:
            bev.get_engine, bev.get_spec, bev.REPRO, bev.RESULTS_LOG = saved
    print("trace_check + compare OK")


if __name__ == "__main__":
    test_trace_and_compare()
    print("test_video_agent_bench_compare: all OK")
