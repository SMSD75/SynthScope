# Adding a detector to SynthScope (the repo rules)

This is the checklist to follow (for people and for the daily arXiv agent) to incorporate a new
AI-generated-content detector cleanly and verify that SynthScope reproduces the paper's results. Keep the
code style and structure consistent with the existing detectors (PGC, SAFE, Co-Spy, AIDE, …).

## When to integrate
Only integrate a paper that satisfies **both** rules: (1) a solid public GitHub repo, and
(2) solid results vs previous methods. If the architecture is unclear, weights aren't released, or
the benchmark data can't be obtained, **do not** commit broken code — note "integration pending: <reason>"
for the maintainers and stop.

## Files to add (one detector = key `<k>`, e.g. `pgc`)
1. **`models/<modality>/<k>/<k>.py`** — `class SynthScope<Name>Model(nn.Module)`:
   - `__init__(self, config: DictConfig)` — build the architecture; respect `config.ModelConfig`.
   - `forward(self, images) -> logits` — raw logits (NOT probabilities). Do model-specific
     preprocessing (e.g. residual/DWT/DCT) **inside** forward where the official model does.
   - `compute_loss(self, batch) -> {"loss", "outputs"}` — used by training.
   - Imports + helper classes at module top-level (repo rule: no nested defs/imports).
   - Add `models/<modality>/<k>/__init__.py` and a `README.md` (see structure below).
2. **`conf/model/image/<k>.yaml`** (or `video/`): `defaults: [/model/default, _self_]`, then `model_type: <k>`
   plus any model knobs, and the model's preprocessing as named transform lists:
   - `eval_transform_list` (required for image models): the **official** test-time preprocessing, used for
     inference, benchmarks and the end-of-training test;
   - `train_transform_list`: the training augmentation;
   - `val_transform_list` (optional): validation during training, when it differs from the eval list;
   - `test_transform_list` (optional): replaces the eval list for the end-of-training test only, e.g. a batchable
     stand-in for SPAI's native-resolution eval, or AIDE's five views with the long side capped at 1024 to bound
     dataloader-worker memory; benchmarks and inference still use `eval_transform_list`.

   Each name is a step built by `AIGITransformations` in `transformations.py`; add new step names there.
   The model's lists take precedence over the dataset's: the `train/eval/val_transform_list` of a
   `conf/dataset/...` config are fallbacks, used only for models whose config sets none, and a warning is printed
   when a dataset list is shadowed. Validation and test pipelines run on PIL images like inference, so an
   end-of-training test with `eval_transform_list` preprocesses exactly as the benchmarks do.
   Video models are mostly out of scope for training: their preprocessing is the registry's video settings (frame
   sampler, `num_frames`, window) plus the dataset lists, and most do not use `eval_transform_list`. `waverep` and `nsg_vd` are the
   exceptions: their frames go through a per-frame transform, so they set `eval_transform_list` (WaveRep:
   `center_crop_504`, `to_dtype`, `normalize_imagenet`; NSG-VD: `nsgvd_frame_224`) and the registry builds their
   `transform` from it with `model_transform` like an image detector (`tests/test_model_transforms.py` checks video
   configs that have a list). A model that reads extra files from the checkpoint folder (NSG-VD: its diffusion model and
   reference set) lists them in the spec's `aux_files`; the engine passes their paths as config overrides. Add
   `conf/dataset/...` / `conf/trainer/...` only if a new recipe is needed.
3. **Dispatch** — add a branch in `trainer.py` `_build_model` (the `if model_type == "<k>": return SynthScope<Name>Model(config)` chain, ~line 69+). The interface lazy-imports by `model_type` in `interface/backend/engine.py` — mirror it if needed.
4. **Interface registry** — `interface/backend/registry.py` `_add(ModelSpec(...))`:
   - `key`, `name`, `paper`, `modality` (`image`|`video`), `overrides=["model@ModelConfig=image/<k>"]`,
   - `predict_kind`: `ce` (logits→`softmax[:,1]`), `bce` (`sigmoid`), `bce_npr`, `vib`, or `video`,
   - `checkpoint="<k>.pth"`, `ckpt_kind`: `synthscope` (our trained weights, strict load) **or** `official_<k>` + a `remap=remap_<k>` function (rename official keys → our module names) for the authors' released weights,
   - `transform=model_transform("image/<k>")`: builds the eval preprocessing from the model config's `eval_transform_list` — match the **official** eval exactly there (resize vs crop, norm vs none). (Getting this wrong silently tanks results — see SAFE: a resize destroyed the high-freq signal.)
   - `weights`: `"official"` | `"synthscope_trained"` | `"training_free"` (shown in the web UI badge).
5. **Per-model `README.md`** — mirror `models/image/pgc/README.md`: short method explanation + official-repo/paper links, the pipeline, "SynthScope adaptations", and a results table in the **Paper / SynthScope_official / SynthScope_train** schema.

## Verify reproduction (required)
Run the detector on the **paper's own benchmark(s)** and compare to the paper's reported numbers:
```
python benchmark/bench_eval.py --model <k> --benchmark {forensynths|ufd|pergen|pool} --data <root> \
  --ckpt <official-or-trained.pth> --bs 32 --device cuda --tag <k>_repro
```
- Loading **official weights** with `--ckpt` checks architecture fidelity (should match the paper).
- Mismatched `load_state_dict` keys → write a `remap_<k>`; near-chance/all-fake → check the transform
  (preprocessing) and the head-name mapping first.
- Record per-generator AP/AUROC; update `repro/AGGREGATED_BENCHMARK.md`, `repro/RESULTS_SUMMARY.md`,
  the repo README table, and the per-model README — all in the Paper / SynthScope_official / SynthScope_train schema.

## Git workflow (keep main clean)
- Do the integration on a branch: `git checkout -b detector/<k>`. Commit the model + conf + registry +
  README + repro results. **Push the branch and surface it for review — do NOT auto-merge to `main`.**
- End commit messages with the repo's `Co-Authored-By` trailer.
- Never commit weights (`*.pth`), `.env`, or caches (already gitignored).

## Tests
Add a smoke test under `tests/` like `tests/test_<k>_smoke.py` (mirror `tests/test_aide_smoke.py`):
build the model, run a dummy `forward`, assert output shape + finite logits.
