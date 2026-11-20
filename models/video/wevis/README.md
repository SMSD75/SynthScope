# WeVisForensics (`wevis`)

**Paper:** Du, Li, Zhang, Sun, Li, Lyu, Tai, *WeVisForensics* (agentic AI-generated video detection), arXiv
[2611.21373](https://arxiv.org/abs/2611.21373) (Nanjing University + Tencent WeChat Vision).
**Code:** [Tencent/WeVisForensics](https://github.com/Tencent/WeVisForensics) @ `43ae1c0` (Apache-2.0).
**Weights:** [tencent/WeVisForensics](https://huggingface.co/tencent/WeVisForensics) @ `9fa025a0` (Apache-2.0): five
LoRA adapters (r64) on [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) @ `851bf6e8` (Apache-2.0), plus
`skill_card.json`.

**Status.** Inference only, with the official code and weights.
- **Fidelity:** our engine and the official transformers backend give identical traces, texts and verdicts on 17 / 17
  clips, and the logit gap matches a direct call exactly (difference 0.0). The authors' vLLM engine is not reproduced.
- **Paper reproduction:** on a documented 40 % subsample of every folder, GenBuster-Bench gives **97.30** accuracy
  (95 % Wilson interval 96.25-98.06; paper 97.7) and GenBuster-Bench++ gives **97.88** (96.62-98.67; paper 97.5). Both
  paper values are inside our intervals.
- **Not a held-out result:** GenBuster-Bench / ++ are **in-distribution** for WeVisForensics, because GenBuster-200K is
  in its training data.
- **Held-out headline:** on four held subsets, against 8 video detectors on identical files, it ranks **7th of 9** (mean
  accuracy 85.59, AUROC 90.23). Its rank there comes from the real videos: it calls about 1 in 4 of the low-resolution
  MSR-VTT reals fake.
- **Availability:** not in the model zoo and not on the Space (see Licence).

## Method
An investigator agent decides by gathering evidence. The investigator (Qwen3.5-4B + the `main` LoRA) sees a 1 fps
overview of the video (at most 16 frames) and, turn by turn, writes one JSON action: call an inspector, or answer `real`
/ `fake`. The inspectors are the same base with their own LoRAs: `visual_appearance`, `physical_consistency` and
`common_sense` look at a time window at 2 fps (at most 72 frames); `ocr_consistency` reads one full-resolution frame
with PP-OCRv5 and checks the text. At most 5 turns and 4 tool calls; answering before a successful tool call, repeating
a call, or not answering by turn 5 is invalid. Decoding is greedy everywhere.

## How SynthScope runs it
The official in-process transformers backend (`demo/wvf_local.py`, the backend of the authors' demo) runs **unmodified**
in its own process, `runner.py`, because it needs transformers >= 5.13 (pinned to 5.23.0) and PaddleOCR 3.7 /
paddlepaddle 3.2.2, which are not in the SynthScope environment. `wevis.py` starts the runner once (the model loads in
about 3-4 min) and sends it one video at a time (JSON lines). Per video the runner calls
`wvf_local.AGENT.run(video, has_audio=False)`, as the official evaluator runs a manifest row.

Before loading, the runner refuses to start unless:
- the official code files, the adapters, the base and the PP-OCRv5 models match the sha256 values pinned in
  `manifest.json`;
- every pinned package has its pinned version (the gate / fidelity runtime): transformers 5.23.0,
  flash-linear-attention and fla-core 0.6.0, peft 0.21.3, torch 2.11.0+cu126, paddlepaddle 3.2.2, paddleocr 3.7.0;
- every adapter loads with no missing and no unexpected key, and each role's LoRA factor count equals its file's
  tensor count.

Everything is read from local folders, and Hugging Face is offline (also in the OCR worker, which inherits the
environment). In the runner process itself, a socket guard also refuses every Python connection that is not to the
local machine. The guard does not cover the OCR worker subprocess or non-Python clients; offline settings and
pre-verified local files are what keep those local.

**Process and files.**
- The runner runs in its own process group. Every way of stopping it (per-video timeout, load timeout, protocol error,
  `close()`) sends SIGTERM and then SIGKILL to the whole group, so the official CPU OCR worker it starts is never left
  running.
- `close()` (model eviction, `release_gpu`) first waits for the video in progress. The web engine calls it outside its
  own lock, so other detectors keep serving while an evicted agent finishes its video.
- A closed client is finished: `score()` raises `WeVisClosedError` instead of starting a new runner, and the engine
  fetches the detector again (a fresh runner) once. A client that is dropped is garbage-collected (the reader thread
  holds only the pipe), and its `__del__` stops the runner.
- **Loading holds the engine lock**, as for every detector. For the agent that is the runner's start: about 40 s with
  warm file caches, 3-4 min on a cold node. Every other detector request on the same engine (web app) waits meanwhile.
- SIGKILL is sent only to a group that is still alive (an empty group's id could be reused).
- If `close()` stops the runner while a video is being scored, that video fails with `WeVisRunnerError`.
- Each video gets its own frame cache (decoded frames, OCR crops) in `<work dir>/videos/<host>_<pid>_<n>`. It is deleted
  after the video unless `$SYNTHSCOPE_WEVIS_KEEP_FRAMES=1`. Web uploads would otherwise grow the cache without bound. The
  trace in the result keeps what a benchmark needs.
- A runner that was killed mid-video (timeout, crash, the OOM killer) leaves its folder behind. The next runner on the
  same host removes the folders of processes that no longer run. Folders of other hosts are kept, and nothing is swept
  with `KEEP_FRAMES`.
- **A still image is refused** ("video-only detector") before anything is loaded or checked, in the Python API and in the
  engine.

**Score.** P(fake) = sigmoid(l(fake) - l(real)), from the investigator's raw LM-head logits at the verdict token of its
final turn (`..."content":"fake"`, the single token `fake` 29571 / `real` 7700). The logits are read during the official
generation by a forward hook, so the text does not change. The verdict is the agent's answer. If the agent gives no
valid answer, the result is `unparsed`: counted wrong, logit 0, P 0.5. The score is near 0 or 1 (|logit| 12.5-20.75 in
the fidelity run), so AUROC adds little to accuracy.

**Audio is off.** Every video runs with `has_audio=False`, the official manifest field. The audio tools need a gated
diarization model and non-commercial voice models. Also, in our benchmarks some fakes have an audio track and every
real is silent, so the audio flag alone would give the label away. In the scored subsets, 8 GenVideo-Val fakes have an
audio stream; no real does, and none of the 2,060 GenBuster-Bench / ++ clips does (`repro/video_agent_bench/msrvtt_probe.json`,
`audio`, from the runner's `audio_stream_present` probe).

**Errors.** The official `scripts/evaluate.py` turns *any* exception from `agent.run` into a `tool_error` row. It
retries the row and, if it still fails, leaves it out of the strict accuracy's denominator and reports how many it left
out. SynthScope follows it:
- Out of memory (long HD clips): `WeVisOOMError`. Kept apart so the video can be retried on a 48 GB A6000; it is not
  counted as wrong.
- Any other exception from the agent (the official `ToolError`, a decode / OCR / engine failure, or anything else):
  `WeVisToolError`. Benchmarks retry it, then leave it out of the accuracy and report the count, as the official
  evaluator does. They also report the stricter variant that counts these videos wrong.
- A runner that dies or stops answering: `WeVisRunnerError`. The next call restarts it.

## Setup
1. Isolated dependencies, installed with `pip install --target`, never into the SynthScope environment. Put them on
   `$SYNTHSCOPE_WEVIS_PYTHONPATH` (os.pathsep-separated).
   - transformers 5.23.0, flash-linear-attention 0.6.0, huggingface_hub, tokenizers, safetensors.
   - paddlepaddle 3.2.2 (CPU; 3.3.x crashes in oneDNN), paddleocr 3.7.0, paddlex 3.7.2.
   - opencv-contrib-python-headless 4.10.0.84, plus a metadata-only `opencv_contrib_python` dist-info shim on nodes
     without libGL.
2. Under `$SYNTHSCOPE_CKPT_DIR`:
   - `wevisforensics_lora/` (`hf download tencent/WeVisForensics --revision 9fa025a02dde4f7f7de8883fe17b6f251f39098e`);
   - `qwen3.5-4b/` (`hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`);
   - `wevis_paddlex/official_models/PP-OCRv5_server_{det,rec}/` (PaddleOCR's own download, or HF
     `PaddlePaddle/PP-OCRv5_server_det` @ `ca867c89` / `_rec` @ `b26c3587`; or set `$SYNTHSCOPE_WEVIS_PADDLEX`);
   - `WeVisForensics/` (a checkout of the code at `43ae1c0`; or set `$SYNTHSCOPE_WEVIS_CODE`).
3. Optional:
   - `$SYNTHSCOPE_WEVIS_PYTHON`: the interpreter (default: the current one).
   - `$SYNTHSCOPE_WEVIS_WORK_DIR`: the per-video frame caches, OCR worker files and the sha256 memo (default:
     `<cache dir>/wevis_work`).
   - `$SYNTHSCOPE_WEVIS_KEEP_FRAMES=1`: keep each video's frames, for debugging.

## Fidelity
`repro/wevis_fidelity.json` (job 295351, ivi-cn029, RTX A5000; log `repro/logs/wevis_fidelity_295351.out`). On 18
clips, our engine is compared with the official backend (`demo/wvf_local.py`, unmodified), run the same way
(`has_audio=False`, greedy) in the same runtime (transformers 5.23.0, fla 0.6.0, torch 2.11.0+cu126, bf16, SDPA). The
18 clips are the 6 gate clips, the 7 demo cases whose recorded traces use no audio tool, and 5 GenVideo fakes: 4 short
HD clips and a 74 s 1080p WildScrape clip.

| check | result |
|---|---|
| clips compared (both backends finished) | 17 / 18 |
| same answer and status | 17 / 17 |
| same tool-call trace (skill + window) | 17 / 17 |
| same raw text of every investigator turn, same final text | 17 / 17 |
| verdict logit vs a direct `generate(output_logits=True)` on the final turn's inputs: same ids | 17 / 17 |
| max abs difference of the logit gap | **0.0** (exactly equal on 17 / 17) |
| same video scored again through `Engine.predict_video_clip` | identical trace and logit |
| demo cases: verdict = the recorded vLLM trajectory's | 7 / 7 |
| correct vs the label | 15 / 17 (MSR-VTT real `msrvtt_6` called fake, HotShot `hotshot_1` called real) |
| unparsed / unscored | 0 / 0 |
| verdict pair's softmax mass at the verdict token; argmax = emitted | >= 0.999996; 17 / 17 |
| out of memory | 1: `WildScrape/00360_D174.mp4` (74 s, 1080p) in **both** backends (a 72-frame inspector window; 22.6 GiB allocated when the vision encoder asked for 286 MiB more). The engine raised `WeVisOOMError` and the runner kept serving |

**Cost on an RTX A5000.**
- 46.6 s per video on average, 45.6 s without the first (warm-up) video; range 20.0-70.9 s; 3-8 model calls.
- The official backend took 48.3 s per video.
- Peak GPU memory: 22.38 GiB allocated, 23.13 GiB reserved (a 20 s 720p Sora clip); 19.54 GiB is resident weights.
- The runner loads in about 40 s with warm file caches. The official reference took 299 s when it ran first in the
  job: a cold read of the weights from network storage, plus the OCR worker start and prefetch. Job MaxRSS was
  26.7 GiB.

Longer HD clips need a larger GPU: retry the videos that raise `WeVisOOMError` on a 48 GB A6000 (see Out of memory).

**Spot check after fix round 1** (`repro/wevis_fidelity_spot.json`, job 295556, ivi-cn023, **RTX A6000**; log
`repro/logs/wevis_fidelity_295556.out`). The first 3 clips were rerun with the per-video frame cache and the error
rule changed.
- Identical answers, tool calls and per-turn raw text: 3 / 3.
- Logit-gap difference vs a direct call: 0.0. The repeat of clip 1 was identical.
- Each video's frame cache was removed after scoring.
- The A6000 logits differ from the A5000 run's (msrvtt_1: -18.75 vs -18.625; msrvtt_2: -18.0 vs -20.75), with the same
  tool calls and verdicts. The kernels are numerically different on another GPU type. Identity holds because both
  backends ran on the same GPU in the same job.

The authors' multi-GPU vLLM evaluator is not reproduced (no vLLM here). Against its recorded demo trajectories, the
transformers backend gives the same verdicts (7 / 7) but some tool traces differ (Task 0 gate).

**GPU type.** Every benchmark row below ran on an RTX A6000 (ivi-cn023), so the score pool has one GPU type; each row
records its GPU. The fidelity run was on an RTX A5000. A5000 and A6000 logits differ by up to 2.75 on the same clip, with
the same tool calls and verdicts (msrvtt_2: -20.75 vs -18.0).

## Cost
On an RTX A6000, over 3,208 scored rows (`repro/video_agent_bench/results/*/wevis.json`, per-video `seconds`,
`model_calls`, `peak_alloc_gib`):
- 41.1 s per video on average (median 39.9 s, range 17.6-100.5 s);
- 5.2 model calls per video, with 1-4 tool calls (2 in about 80 % of the videos);
- peak 22.69 GiB allocated;
- 36.6 GPU-h in all, of which 23.6 GPU-h is GenBuster-Bench / ++.

The runner loads in about 40 s with warm file caches; the first load on a node takes 3-4 min. A full GenBuster-Bench +
GenBuster-Bench++ run (5,150 clips) would take about 60 GPU-h, from the pilot's 41.8 s per clip (job 295610).

## Out of memory
The agent holds two 4B copies (19.5 GiB of resident weights). A long HD clip can run out of memory on a 24 GB GPU:
in the fidelity run, `GenVideo/WildScrape/00360_D174.mp4` (74 s, 1080p) did so on an RTX A5000, in both our engine and
the official backend, while the vision encoder read a 72-frame inspector window.
- The runner reports out of memory as `WeVisOOMError` and keeps serving the next video.
- `benchmark/bench_eval_video.py` retries an OOM video in place (`--agent-oom-retries`, default 2) and counts the
  retries per row.
- A video that still runs out of memory becomes a final `oom` row. Like a `tool_error` row, it is left out of the
  official `acc` and counted wrong in `acc_strict`, and its count is reported separately.
- Run long HD clips on a 48 GB GPU. On the RTX A6000 no benchmark video ran out of memory (0 OOM, 0 retries in 3,208
  rows).

## Benchmarks: four held subsets against 8 video detectors
**Protocol** (`repro/video_agent_bench/spec.json`, `make_subsets.py`, committed manifests with a sha1 per file):
- GenVideo-Val, VideoPhy, VidProM and EvalCrafter, subsampled to L = ceil(250 / generators) per class per generator:
  the first L sorted reals and the first L sorted fakes of every generator.
- EvalCrafter has no reals. It is paired with the first L GenVideo-Val MSR-VTT reals, the convention of the merged
  video matrix.
- The agent and every comparator score byte-identical files: `d3`, `split`, `stall`, `restrav`, `nsg_vd`, `waverep`,
  `waverep_g4` and `vina_video`. `nsg_vd` and `vina_video` are not redistributed; "best public" names the best publicly
  licensed comparator.
- Accuracy is `bench_eval_video`'s balanced accuracy: the mean of real and fake accuracy per generator, then the mean
  over generators. For the agent it is computed from its verdict (`unparsed` is wrong for both classes).
- AUROC ranks the verdict-token logit gap. The comparators use their served thresholds.
- There were no tool errors, OOMs, unparsed or unscored answers, so the official and strict accuracies are equal.
- Source: `repro/video_agent_bench/comparison.json`, from `results/<subset>/*.json`. Logs: `repro/logs/wevis_wevis_w0_295637.out`,
  `wevis_wevis_w1_295640.out` and `wevis_wevis_w0_295611.out` (31 cached GenVideo videos).
- `trace_check.py`: all 54 result JSONs and all 3,208 per-video agent records trace to the logs.

| subset (L; clips, of which real) | wevis acc | AUROC | real / fake acc | rank acc / AUROC (of 9) | best comparator acc | median acc | pooled acc (rank) |
|---|---:|---:|---:|---:|---|---:|---:|
| GenVideo-Val (25; 275, 25) | 85.20 | 92.20 | 72.00 / 98.40 | 7 / 6 | SPLIT 100.0 | 88.60 | 96.00 (4) |
| VideoPhy (32; 288, 32) | 86.33 | 90.05 | 75.00 / 97.66 | 7 / 7 | VINA 99.41 (public: SPLIT 98.44) | 92.19 | 95.14 (5) |
| VidProM (63; 315, 63) | 85.52 | 89.11 | 73.02 / 98.02 | 7 / 7 | SPLIT 99.21 | 89.68 | 93.02 (4) |
| EvalCrafter + MSR-VTT reals (18; 270, 18) | 85.32 | 89.57 | 72.22 / 98.41 | 6 / 8 | SPLIT 100.0 | 92.85 | 96.67 (5) |
| **mean of the 4 (headline)** | **85.59** | **90.23** | | **7 / 7** | SPLIT 99.41 (AUROC 100.0) | 91.40 (AUROC 96.31) | |

**Reading.**
- **The real videos drive the rank.** Fake accuracy is 97.7-98.4 on every held subset, but real accuracy is only
  72-75: the agent calls about 1 in 4 MSR-VTT reals fake (37 of the 138 real rows).
- These reals are low-resolution: all 63 distinct MSR-VTT reals of the four subsets are 298x224 at 3 fps (median 38
  frames), while the fakes' median frame height is 512 (`repro/video_agent_bench/msrvtt_probe.json`, `media`, from
  `media_probe.py`; every file sha1-checked against the manifests). On GenBuster's own
  reals the agent does much better (real accuracy 93.25 on GB, 98.25 on GB++; below).
- The fidelity run predicted this: `msrvtt_6` was called fake.
- The real pools are small (18-63 clips per subset), so real accuracy is noisy, about +-9 points at n = 25.
- The pooled one-row-per-video accuracy (the paper's metric) is 93.0-96.7 and ranks 4th-5th. It is dominated by the
  fakes, which outnumber the reals 4:1 to 14:1 here.
- **Near-binary scores:** |logit| is 9.1-24.75 over all 3,208 rows, so AUROC adds little to accuracy.
- **Determinism:** 32 files were scored in more than one subset (136 pairs). All pairs are identical in verdict, logit
  and tool calls.
- **Overlap with the training data:** WeVisForensics' training sources include VBench-2.0, AIGVDBench and RewardData.
  Their overlap with these four subsets was not checked.
- **Comparators:** every comparator video equals the merged video-matrix cache (decision agreement 100 % on the 28
  cells checked; max |dP| 7.3e-5, NSG-VD 0.014; `comparator_consistency.json`). D3 and all GenBuster cells have no
  reference.

`subset_means.all6` (89.29 / 92.61, 2nd of 9) also averages in GenBuster-Bench / ++. Those are in-distribution for
wevis, so it is not a held-out mean.

## GenBuster-Bench and GenBuster-Bench++ (the paper's GB / GB++; IN-DISTRIBUTION)
GenBuster-Bench (HF `l8cv/GenBuster-Bench` @ `68e41ab5`) holds 1,000 reals and 2,150 fakes:
- 1,000 in-domain fakes from 4 open generators;
- 1,000 OOD fakes from 8 commercial generators;
- 150 wild fakes from 11 recent generators.

GenBuster-Bench++ (`l8cv/GenBuster-Bench-plusplus` @ `7f184bef`, video track) holds 1,000 reals and 1,000 fakes.
`dataset.py` loads both (`--dataset genbuster` / `genbuster_pp`).

**Subsample.** The full sets would cost about 60 GPU-h. We scored the first ceil(0.4 n) sorted clips of every folder,
which keeps the real : fake composition. The files are sha256-named, so the sorted order is effectively random. That is
1,260 GB clips (400 real + 860 fake) and 800 GB++ clips (400 + 400).

**Metric.** The paper's metric, one row per clip, as the official `scripts/evaluate.py` computes it: fakes are the
positive class, invalid answers count wrong, and tool errors are excluded (there were none). The paper ran the authors'
vLLM engine; we ran the official transformers backend, greedy, with audio off (none of the 2,060 GenBuster clips has an
audio stream: `msrvtt_probe.json`).

| | SynthScope (40 % subsample) | paper (arXiv 2611.21373) |
|---|---:|---:|
| GB accuracy | **97.30** (95 % Wilson 96.25-98.06), n = 1,260 | 97.7 (Table 4) |
| GB fake recall / F1 | 99.19 / 98.05 | 99.3 / 98.3 (Table 12) |
| GB real accuracy | 93.25 | 94.3 (implied by Acc. and R) |
| GB in-domain / OOD (pooled) / wild fakes | 99.5 / 99.25 / 96.67 (n = 400 / 400 / 60) | 99.7 / 99.6 / 95.3 (Table 11) |
| GB++ accuracy | **97.88** (96.62-98.67), n = 800 | 97.5 (Table 4) |
| GB++ fake recall / F1 | 97.50 / 97.87 | 97.2 / 97.4 (Table 12) |
| GB++ real accuracy | 98.25 | 97.8 (implied) |

**Result.**
- Both accuracies **reproduce the paper**: -0.40 on GB and +0.38 on GB++, each paper value inside our 95 % interval.
- These are 40 % subsamples, not the full sets.
- The paper's OOD mean (99.6) is pooled over clips; our unweighted mean of the 8 OOD generators is 99.22.
- The wild generators have 4-8 fakes each here, so the per-generator rates and the per-generator AP (48.6 on GB) are
  noisy. Quote the pooled numbers.

**In-distribution.** GenBuster-200K, the source of GenBuster-Bench, is in WeVisForensics' training data (paper Sec. 3;
GB is the paper's in-domain benchmark). GB++ comes from the same BusterX family. The 8 comparators were trained on other
data, and their thresholds were calibrated elsewhere (SPLIT and D3 call nearly every GenBuster real fake).
- wevis ranks 1st of 9 on both sets: GB 97.30 vs VINA 89.37 (best public: WaveRep-G4 85.63; median 69.80); GB++ 97.88
  vs VINA 89.00 (WaveRep-G4 80.00; median 52.56).
- **This is not a held-out comparison.** The held-out headline is the four subsets above.

## Licence
- Code and adapters: Apache-2.0 (Tencent preamble + the Qwen3.5-4B attribution). The model card adds: research use,
  not for autonomous high-impact decisions about individuals.
- Base: Apache-2.0. PP-OCRv5: Apache-2.0.
- The audio tools' models (pyannote, gated CC-BY-4.0; audEERING, CC-BY-NC-SA-4.0) are never used.
- See `LICENSES.md`.

**Why it is not on the Space or in the model zoo.** The licence would allow both. But:
- the Space runs on a CPU, and the agent needs a CUDA GPU with about 22 GB (41 s per video on an RTX A6000);
- the release is five LoRA folders on a separate 9.3 GB base folder, plus the OCR models, the official code checkout and
  an isolated transformers 5.23 / PaddleOCR runtime, which the zoo's one-file-per-key layout does not hold.

So it is in `NEVER_REDISTRIBUTE` (`synthscope/hub.py`): users download the pinned revisions themselves (Setup above).
