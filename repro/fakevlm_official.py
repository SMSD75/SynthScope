"""FakeVLM's official inference (``scripts/eval.py`` of github.com/opendatalab/FakeVLM @ fdcdcf1, unmodified), for
repro/fakevlm_fidelity.py. Independent of SynthScope's code: only the official script, transformers, PIL and numpy.

Two subcommands, each run under the transformers it is given (``PYTHONPATH``: the authors' pin, transformers 4.45.2
with tokenizers 0.20.3 and huggingface_hub 0.25.2 in a ``pip --target`` folder; or the SynthScope env's 5.5.4):

* ``eval``: ``eval.py``'s ``main()`` on a test JSON in the official format (``image``, ``label`` real=1 / fake=0,
  ``conversations``) with three patches the environment needs, nothing else:
    1. ``AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", revision='a272c74')`` -> the local copy of that
       revision (offline). On transformers >= 4.47, which expands ``<image>`` in the processor, the processor is also
       given ``patch_size=14``, ``vision_feature_select_strategy="default"`` (lingcco/fakeVLM's own processor_config.json)
       and ``num_additional_image_tokens=1`` (576 image tokens, what 4.45's in-model merge gives); 4.45.2 needs none;
    2. ``use_flash_attention_2=True`` -> ``attn_implementation="sdpa"`` (flash-attn is not installed);
    3. ``revision='a272c74'`` dropped for the local weights folder.
  It writes the official output JSON (``image_path``, ``output``) and, next to it, ``<out>.acc.json``: the accuracy dict
  ``eval.py`` prints (``calculate_results_acc``), the transformers version, wall time and peak GPU memory.
* ``inputs``: the official processor call of ``legion_cls_dataset.__getitem__`` (``Image.open``, the raw prompt,
  ``padding="max_length", max_length=1024, truncation=True``) for each image, saved as ``input_ids``,
  ``attention_mask`` and ``pixel_values`` arrays (npz) -- under 4.45.2, the reference inputs of the fidelity check.

    PYTHONPATH=$DEPS/transformers4452 python repro/fakevlm_official.py eval --model M --processor P --data ROOT \
        --test-json T --out OUT.json [--bs 4]
    PYTHONPATH=$DEPS/transformers4452 python repro/fakevlm_official.py inputs --processor P --data ROOT --list L --out X.npz
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image

OFFICIAL_DIR = Path("/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared/vlm_ref/FakeVLM")   # @ fdcdcf1
PROMPT = "<image>Does the image looks real/fake?"          # FakeClue test.json, every row
EXPANDS_IN_PROCESSOR = tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 47)


def load_processor(path: str):
    proc = transformers.AutoProcessor.from_pretrained(path, local_files_only=True)
    if EXPANDS_IN_PROCESSOR:
        proc.patch_size = 14
        proc.vision_feature_select_strategy = "default"
        proc.num_additional_image_tokens = 1
    return proc


def official_module(official_dir: Path):
    spec = importlib.util.spec_from_file_location("fakevlm_official_eval", official_dir / "scripts" / "eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_eval(args) -> int:
    mod = official_module(Path(args.official_dir))
    processor_path, model_path = args.processor, args.model

    class LocalProcessor:                          # patch 1
        @staticmethod
        def from_pretrained(name, revision=None, **kw):
            assert (name, revision) == ("llava-hf/llava-1.5-7b-hf", "a272c74"), (name, revision)
            return load_processor(processor_path)

    class SdpaLlava:                               # patches 2 and 3
        @staticmethod
        def from_pretrained(path, **kw):
            assert path == model_path and kw.pop("use_flash_attention_2") is True and kw.pop("revision") == "a272c74"
            return transformers.LlavaForConditionalGeneration.from_pretrained(path, attn_implementation="sdpa", **kw)

    mod.AutoProcessor = LocalProcessor
    mod.LlavaForConditionalGeneration = SdpaLlava
    printed = {}
    calculate = mod.calculate_results_acc

    def keep(results):
        printed.update(calculate(results))
        return printed

    mod.calculate_results_acc = keep
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.parent / (out.stem + "_cwd")        # eval.py makes ./results in its working directory
    work.mkdir(exist_ok=True)
    os.chdir(work)
    sys.argv = ["eval.py", "--model_path", model_path, "--val_batch_size", str(args.bs), "--workers", "1",
                "--output_path", str(out), "--test_json_file", str(Path(args.test_json).resolve()),
                "--data_base_test", str(Path(args.data).resolve())]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    mod.main()
    seconds = time.time() - t0
    meta = {"transformers": transformers.__version__, "torch": torch.__version__, "val_batch_size": args.bs,
            "wall_seconds_incl_load": round(seconds, 1), "peak_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "device": torch.cuda.get_device_name(0), "accuracy": printed}
    Path(str(out) + ".acc.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"[official eval.py, transformers {transformers.__version__}] {seconds:.1f}s -> {out}", flush=True)
    return 0


def run_inputs(args) -> int:
    proc = load_processor(args.processor)
    rels = [r for r in Path(args.list).read_text().split("\n") if r]
    arrays = {}
    for i, rel in enumerate(rels):
        x = proc(text=PROMPT, images=Image.open(Path(args.data) / rel), return_tensors="pt", padding="max_length",
                 max_length=1024, truncation=True)
        for key in ("input_ids", "attention_mask", "pixel_values"):
            arrays[f"{i}/{key}"] = x[key].numpy()
    arrays["images"] = np.array(rels)
    arrays["transformers"] = np.array(transformers.__version__)
    np.savez(args.out, **arrays)
    print(f"[official inputs, transformers {transformers.__version__}] {len(rels)} images -> {args.out}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--model", required=True)
    e.add_argument("--processor", required=True, help="local llava-hf/llava-1.5-7b-hf @ a272c74 processor folder")
    e.add_argument("--data", required=True)
    e.add_argument("--test-json", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--bs", type=int, default=4, help="eval.py needs > 1 (its .squeeze() breaks a batch of one)")
    e.add_argument("--official-dir", default=str(OFFICIAL_DIR))
    i = sub.add_parser("inputs")
    i.add_argument("--processor", required=True)
    i.add_argument("--data", required=True)
    i.add_argument("--list", required=True, help="image paths relative to --data, one per line")
    i.add_argument("--out", required=True)
    args = ap.parse_args()
    return run_eval(args) if args.cmd == "eval" else run_inputs(args)


if __name__ == "__main__":
    raise SystemExit(main())
