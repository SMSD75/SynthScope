# Working with SynthScope (guide for coding agents)

SynthScope wraps published AI-generated image and video detectors behind one registry, one inference engine and one
benchmark harness. This file tells coding agents (Claude Code, Codex, Copilot, ...) how to use and extend it. Humans:
start with [README.md](README.md).

## Setup

```bash
pip install -e .                 # library + CLI; extras: ".[web]" web demo, ".[train]" training, ".[dev]" tests/lint
```

- Install the PyTorch build that matches the machine's CUDA first. `clip` and `fat_former` also need OpenAI CLIP:
  `pip install git+https://github.com/openai/CLIP.git`.
- Weights are read from `$SYNTHSCOPE_CKPT_DIR/<key>.pth` (default `~/.cache/synthscope/checkpoints`). Each detector's
  `models/<image|video>/<key>/README.md` links the official download. Training-free detectors (`d3`, `split`, `stall`)
  need no checkpoint.

## Run a detector

```python
import synthscope
synthscope.list_models()                      # detector keys + whether their weights are present
det = synthscope.load("npr")                  # any key from list_models()
det.predict("photo.png")                      # {'fake_probability': ..., 'verdict': 'fake' | 'real', ...}
```

```bash
synthscope list
synthscope detect photo.png --model npr --json
```

Pass a file path or bytes rather than an already-decoded image when you can: some detectors' official preprocessing
depends on the source format (e.g. a PNG is re-saved as JPEG before scoring), and the loaders record it.

## Evaluate on a benchmark

```bash
python benchmark/bench_eval.py --model npr --dataset synthwildx      # downloads + organises the dataset if needed
python benchmark/bench_eval.py --help                               # protocols (--benchmark), --ckpt, --model-config, ...
python benchmark/bench_eval_video.py --help                         # video detectors
```

Results go to `repro/bench_results/<tag>.json` and one line in `repro/MASTER_RESULTS.jsonl`. Report numbers only from
those files, and state the protocol (subsets, threshold, metric) with every number.

## Add a detector

Follow [docs/ADD_DETECTOR.md](docs/ADD_DETECTOR.md) exactly. In short:

- `models/<image|video>/<key>/` with the model class (vendor official code with its licence header) and a `README.md`;
- `conf/model/<image|video>/<key>.yaml` whose `eval_transform_list` is the official test-time preprocessing;
- the dispatch in both `trainer.py` and `interface/backend/engine.py` (`_build_model`);
- a `ModelSpec` in `interface/backend/registry.py` using `model_transform()`;
- `tests/test_<key>_smoke.py` and a `LICENSES.md` row.

Prove fidelity before claiming results: identical state dict and identical outputs to the official code on raw files,
then reproduce the paper's tables with `bench_eval.py` on full data.

## Checks before a pull request

```bash
python -m compileall -q synthscope interface models benchmark *.py
ruff check synthscope
python benchmark/leaderboard.py --check
python tests/test_model_transforms.py            # plus tests/test_<key>_smoke.py for any detector you touched
```

## Conventions

- Imports at the top of the file; helpers and classes at module level; short comments in the style of the surrounding
  code.
- Never commit weights, datasets, `.env` or API keys. Third-party detectors, weights and datasets keep their own
  licences ([LICENSES.md](LICENSES.md)); do not redistribute weights whose licence does not allow it.
- Keep the official preprocessing exact: a transform that "looks equivalent" can move a benchmark by several points.

## Citing

If SynthScope helped the user's work, point them to [CITATION.cff](CITATION.cff) (GitHub's "Cite this repository"
button) and to the original papers of the detectors they used.
