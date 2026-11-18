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

### VLM detectors (a released vision-language model that answers "real or fake")
A detector that is a released Hugging Face VLM (Veritas++; Align, Don't Blend's VLM branch, a LoRA) uses the `vlm` kind (`models/vlm/vlm.py`) instead of
steps 1-3. It is a family-adapter design:
- a **family** (`VLMFamily`) says how a model family is loaded and fed: the strict `from_pretrained` load into the
  official transformers class (local files only; a LoRA adapter folder, `adapter_config.json`, is kept unmerged on a
  local `base_model` folder named relative to `$SYNTHSCOPE_CKPT_DIR`, needs peft (`pip install ".[vlm]"`: peft 0.21.3, qwen-vl-utils 0.0.14); the processor then comes from the
  base unless `processor` is set; the strict adapter check compares the file's keys after the family's `adapter_key`
  rename, which for Qwen-VL maps keys saved before the transformers 4.52 module rename to the current names; an unknown
  key still fails), the processor (+ `processor_overrides`), the verdict token sequences (default: the
  first verdict pair with a leading space as the tokenizer encodes it, e.g. SentencePiece `["▁", "▁fake"]`; set
  `verdict_sequences` to give token strings), the prompt (chat template on, or the raw prompt with
  `chat_template=False`), the processor call and the input size. `QwenVLFamily` (`QWEN_VL`) is implemented; the base
  class fits processors that take `text=[prompt], images=[image]` (LLaVA-style; `tests/test_veritaspp_smoke.py` plugs a
  dummy one in). A family whose loading or inputs differ (e.g. FakeVLM's processor from another repo) overrides the
  methods it needs;
- `models/image/<k>/<k>.py` holds the detector's `VLMConfig`: the family and transformers class, the official prompts
  (verbatim), the image loader (`models/vlm/image.py`: vLLM's EXIF transpose + white alpha composite, when the official
  inference is a vLLM server; ms-swift's plain RGB conversion, `swift_rgb`, when it is `swift infer`), an optional PIL
  `preprocess` hook (e.g. ms-swift's Qwen-VL resize, `swift_qwen_vl_resize`: `rescale_image` + qwen-vl-utils
  `fetch_image`, ported in `models/vlm/image.py`; with `processor_kwargs={"images_kwargs": {"do_resize": False}}` as
  swift calls the processor), the decoding
  (`decoding` = greedy for benchmarks, `official_decoding` = the authors' setting), the answer marker and end tag (both
  optional), the verdict word pairs, the verdict mode (`generate`, or `teacher_forced`: one forward pass on the prompt +
  `teacher_forced_suffix` + the verdict sequences' shared prefix, read at their first differing token), raw or
  processed logits (processed = after a repetition penalty, what greedy decoding
  picks from), the official verdict parser (`models/vlm/parsers.py`, vendored with attribution) and `max_pixels`
  (None = the processor's default; set a cap only if an image does not fit, and document it). A pre-resize hook takes its
  parameters through `preprocess_kwargs` (so they enter the cache identity). If the official parser can read the verdict
  somewhere other than after the answer marker (Align, Don't Blend: a tag-less final line), pair it with a
  `verdict_offset` hook (response -> character offset of the word it read; `parsers.adb_verdict_offset`), so the score is
  read at that word;
- the spec: `predict_kind="vlm"`, `ckpt_kind="official_vlm"`, `checkpoint="<folder>"`, `vlm=<config>`, `overrides=[]`,
  no `transform`, `native_res=True`; no model config and no `trainer.py` dispatch (inference only);
- score: P(fake) = sigmoid(l(fake) - l(real)) at the verdict token (generate: the emitted verdict token and its
  same-spelling counterpart; teacher-forced: the first differing tokens of the verdict sequences; `logit` = that gap;
  `score_source` "verdict_token", or "verdict_word" at the parser's `verdict_offset`), 0.5 with `score_source="unparsed"`
  when no verdict token was found and the parser found no verdict either, or `"unscored"` when it did (counted as
  `n_unscored` in bench_eval's result JSON: report it with AUROC), plus the parsed verdict and the rationale
  (`Engine.vlm_score`); pass the decoded file without an RGB conversion (`media.decode_image_raw`);
- not supported yet: batching (one image per call), video;
- a multi-GB folder does not fit the one-file zoo: add the key to `NEVER_REDISTRIBUTE` with the reason.
- benchmarking: `bench_eval` scores a `vlm` spec one image at a time through `Engine.vlm_score` (greedy by default;
  `--vlm-decoding official --vlm-seed N` for the authors' sampling). An unparsed answer counts as wrong for accuracy
  (the `*_drop` fields leave it out, as the authors' scripts usually do) and gets logit 0 for AUROC; report the unparsed
  count. A VLM costs seconds per image, so benchmarks are subsampled (`--limit L`, the same files for every detector it is
  compared with; `repro/vlm_bench/` holds the subset manifests, the unit runner and `compare.py`). `--vlm-cache DIR`
  resumes an interrupted run; its folder is keyed by the weights revision (and a LoRA base's), `max_new_tokens`, the
  prompts, the rest of the VLM config (including `preprocess_kwargs` and `verdict_offset`) and the VLM code version, so
  it never returns records written under another setting. The code version hashes every `models/vlm/*.py`, so an edit
  there (even a comment) re-keys the caches of every VLM detector; say so in the merge note. When a re-key provably
  leaves some records unchanged, `repro/vlm_bench/migrate_vlm_cache.py` (one entry per known re-key) carries them into
  the new folder and leaves out the rest, which the next run scores again.
  `compare.py` writes one column per VLM (`vlms.<model>` per subset, with `rank_*` among the comparators plus that VLM
  and `rank_all_*` also counting the other VLMs; `subset_means.<model>`; `paper_comparison.<model>`), and
  `run_units.py` takes `--vlm-model <key>` and the `jpegfakes<Q>` variant (`bench_eval --jpeg-fakes Q`: only the fakes
  JPEG-compressed, for a paper protocol that aligns the fakes' format with the reals');
- documentation: the model README states the official request, the greedy-versus-official decoding choice (say where each
  sampling setting comes from: a script, the model's `generation_config.json` or the paper), the unparsed policy, the
  fidelity check against a direct transformers call (identical tokens and logit gap) and, when the official inference
  code runs here, against it too (Align, Don't Blend: ms-swift's pt backend; name any backend you did not compare), memory
  and speed, and the licence. Compare with the paper only on the paper's own data and metric; a subset or a metric you
  had to guess is "reference", never "reproduced". If the paper's number is for a larger system than the released model
  (e.g. VLM OR an unreleased expert), say that it is not reproducible and list the printed number as context, with no
  delta.

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
