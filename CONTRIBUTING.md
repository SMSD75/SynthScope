# Contributing to SynthScope

Thanks for your interest! SynthScope aims to be the **unified, reproducible benchmark** for
AI-generated image & video detection. The most valuable contributions are **new detectors**,
**new datasets/generators**, and **independently reproduced results**.

## Getting set up

```bash
git clone https://github.com/SMSD75/SynthScope.git
cd SynthScope
pip install -e ".[dev]"          # editable install + contributor tooling
# PyTorch: install the build matching your CUDA from https://pytorch.org if needed.
# Some detectors also need OpenAI CLIP:  pip install git+https://github.com/openai/CLIP.git
```

Before opening a PR:

```bash
ruff check synthscope                   # lint the library surface (errors only; see pyproject [tool.ruff])
python -m compileall synthscope interface models *.py
pytest tests/ -k smoke            # where data/weights are available
```

## Adding a new detector

The framework is model-agnostic: the trainer only needs a `compute_loss(batch)` method, where a
batch is `(image, label, folder_id)` and `folder_id` is the generator/source used for
per-generator metrics. To add a method `foo`:

1. **Model** — create `models/<image|video>/foo/foo.py` with an `nn.Module` exposing
   `compute_loss(batch) -> {"loss": ..., "outputs": ...}`. Add a short method `README.md`.
2. **Dispatch** — register `model_type=foo` in `SynthScopeModelTrainer._build_model` (`trainer.py`).
3. **Config** — add `conf/model/<modality>/foo.yaml` and `conf/trainer/<modality>/foo.yaml`.
4. **Recipe** — add `experiments/<modality>/foo.sh` with the env vars + launch command.
5. **(Optional) API/demo** — add a `ModelSpec` to `interface/backend/registry.py` so the detector
   is runnable via `synthscope.load("foo")`, the CLI, and the web demo (preprocessing, checkpoint +
   any official-weights remap, and the forward → P(fake) convention).
6. **Results** — run it and paste the logged `Test Metrics` in the PR. New numbers should trace to
   a real run, never be estimated (see the reproduction table convention in the README).

## Adding a dataset

Add a `dataset` class + data module in `dataset.py` (auto-download → common `real`/`fake` layout)
and a `conf/dataset/<modality>/<name>.yaml`. Keep paths overridable via config/env — no hard-coded
absolute paths.

## Submitting a benchmark result

Add one JSON file under [`benchmark/submissions/`](benchmark/submissions/) (see
[`benchmark/schema.json`](benchmark/schema.json) and the protocol in
[`benchmark/README.md`](benchmark/README.md)). Prefer `verified` results produced by
`benchmark/evaluate.py`; for `external` numbers, set `source`. Regenerate and commit the table:

```bash
python benchmark/evaluate.py --model <key> --track <track> --data <root> --out benchmark/submissions/<key>.json
python benchmark/leaderboard.py        # updates LEADERBOARD.md
python benchmark/leaderboard.py --check  # what CI runs
```

## Style

- Match the surrounding code; keep imports and helper/class definitions at module top level.
- Keep changes focused; one detector/dataset/fix per PR.
- Be honest about faithfulness: if a re-implementation diverges from the paper, document it (as the
  reproduction notes already do).

## Reporting bugs / proposing features

Use the issue templates. For questions, open a Discussion. By contributing you agree your work is
licensed under the project's [Apache-2.0 license](LICENSE).
