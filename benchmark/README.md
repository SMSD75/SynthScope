# SynthScope Benchmark & Leaderboard (v1)

A standardized, **versioned** evaluation protocol for AI-generated image & video detection, so that
methods are comparable under identical data, metrics, and scoring. The leaderboard is generated
from the submission files in [`submissions/`](submissions/) — see [LEADERBOARD.md](../LEADERBOARD.md).

> **Why:** every detection paper ships its own repo, data format, and metric conventions, which
> makes cross-method comparison unreliable. SynthScope fixes the protocol once and evaluates every
> detector through the same pipeline.

## Tracks (v1)

| Track id | Modality | Test set | Train signal | Primary metric |
| --- | --- | --- | --- | --- |
| `forensynths` | image | ForenSynths real/fake, scored **per generator** | ProGAN-trained (cross-generator generalization) | **mAP** (mean AP over generators) |
| `genimage-sdv1.4` | image | GenImage SD-v1.4 real/fake | in-distribution diffusion | **AUROC** |
| `genvideo-val` | video | GenVideo validation split | mixed | **AP** |

Metrics are reported in **%**. `mAP` = mean Average Precision over generators; `AUROC`/`AP` use the
detector's native fake-probability; `Acc` = mean balanced accuracy (mean of real- and fake-accuracy)
over generators at threshold 0.5.

## Protocol rules

1. **No test-set training or tuning.** Thresholds/calibration must be fixed before evaluation.
2. **Report per generator, then average.** A track's headline number is the unweighted mean over
   generators (not a pooled number), which measures generalization rather than rewarding a few
   large generators.
3. **Declare provenance** via `evaluation`: `official-weights`, `from-scratch`, `frozen-probe`,
   `training-free`, or `external` (numbers produced outside SynthScope).
4. **Verified numbers trace to a run.** `verified: true` means the result came from a logged SynthScope
   evaluation (set `source` to the log/run path). Reference seed entries are transcribed from the
   repo's reproduction table.

## Data layout

The evaluator expects the SynthScope common layout — one directory per generator, each split into
`real/` and `fake/`:

```
<benchmark_root>/
  <generator_a>/real/*.png   <generator_a>/fake/*.png
  <generator_b>/real/*.png   <generator_b>/fake/*.png
```

A single pooled set is just one generator directory (or `<root>/real`, `<root>/fake`).

## Evaluate a detector and produce a submission

```bash
python benchmark/evaluate.py \
  --model fat_former --track forensynths \
  --data /path/to/forensynths_test \
  --out benchmark/submissions/fat_former.json
```

This loads the detector via the `synthscope` API, scores every image (batched), computes per-generator +
mean metrics, and writes a schema-valid submission JSON. Then regenerate the table:

```bash
python benchmark/leaderboard.py            # writes LEADERBOARD.md
python benchmark/leaderboard.py --check     # CI: fail if LEADERBOARD.md is stale or a submission is invalid
```

## Submit a result (PR)

1. Add one JSON file under `submissions/` (see [`schema.json`](schema.json) and existing entries).
2. Run `python benchmark/leaderboard.py` and commit the updated `LEADERBOARD.md`.
3. Open a PR. CI validates your submission and that the table is up to date. Prefer `verified`
   results produced by `evaluate.py`; for `external` numbers, link the source.

Versioning: this is **v1**. Protocol-breaking changes bump to v2 and start a new leaderboard so old
numbers stay meaningful.
