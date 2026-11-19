# WeVisForensics (`wevis`)

**Paper:** Du, Li, Zhang, Sun, Li, Lyu, Tai, *WeVisForensics* (agentic AI-generated video detection), arXiv
[2611.21373](https://arxiv.org/abs/2611.21373) (Nanjing University + Tencent WeChat Vision).
**Code:** [Tencent/WeVisForensics](https://github.com/Tencent/WeVisForensics) @ `43ae1c0` (Apache-2.0).
**Weights:** [tencent/WeVisForensics](https://huggingface.co/tencent/WeVisForensics) @ `9fa025a0` (Apache-2.0): five
LoRA adapters (r64) on [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) @ `851bf6e8` (Apache-2.0), plus
`skill_card.json`.

**Status.** Inference only, with the official code and weights. Not on the Space (it needs a GPU with about 24 GB) and not
in the model zoo. The benchmark results and the paper reproduction (GenBuster-Bench / ++) are to follow.

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
- transformers is 5.23.0 and paddlepaddle is 3.2.2;
- every adapter loads with no missing and no unexpected key, and each role's LoRA factor count equals its file's
  tensor count.

Everything is read from local folders: Hugging Face is offline, and a socket guard refuses any connection that is not
to the local machine.

**Score.** P(fake) = sigmoid(l(fake) - l(real)), from the investigator's raw LM-head logits at the verdict token of its
final turn (`..."content":"fake"`, the single token `fake` 29571 / `real` 7700). The logits are read during the official
generation by a forward hook, so the text does not change. The verdict is the agent's answer. If the agent gives no
valid answer, the result is `unparsed`: counted wrong, logit 0, P 0.5. The score is near 0 or 1 (|logit| 12.5-20.75 in
the fidelity run), so AUROC adds little to accuracy.

**Audio is off.** Every video runs with `has_audio=False`, the official manifest field. The audio tools need a gated
diarization model and non-commercial voice models. Also, in our benchmarks some fakes have an audio track and every
real is silent, so the audio flag alone would give the label away.

**Errors.**
- Out of memory (long HD clips): `WeVisOOMError`. Retry the video on a larger GPU; it does not count as wrong.
- An official `ToolError` (decode / OCR failure): `WeVisToolError`. The official evaluator retries such videos and then
  leaves them out.
- A runner that dies: `WeVisRunnerError`. The next call restarts it.

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
3. Optional: `$SYNTHSCOPE_WEVIS_PYTHON` (the interpreter, default the current one) and `$SYNTHSCOPE_WEVIS_WORK_DIR` (the
   frame cache and OCR worker files, default `<cache dir>/wevis_work`).

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
- The runner loads in about 40 s with warm file caches (the official reference: 299 s cold, including the
  sha256-checked first read). Job MaxRSS 26.7 GiB.

Longer HD clips need a larger GPU: retry the videos that raise `WeVisOOMError` on a 48 GB A6000.

The authors' multi-GPU vLLM evaluator is not reproduced (no vLLM here). Against its recorded demo trajectories, the
transformers backend gives the same verdicts (7 / 7) but some tool traces differ (Task 0 gate).

## Licence
- Code and adapters: Apache-2.0 (Tencent preamble + the Qwen3.5-4B attribution).
- Base: Apache-2.0. PP-OCRv5: Apache-2.0.
- The audio tools' models (pyannote, gated CC-BY-4.0; audEERING, CC-BY-NC-SA-4.0) are never used.
- See `LICENSES.md`.
