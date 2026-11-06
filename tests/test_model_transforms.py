"""Checks the preprocessing lists of the model configs (no downloads; MoE-JEPA's V-JEPA 2 processor runs only if cached).

* every ``conf/model/image/*.yaml`` has a non-empty ``eval_transform_list``, and every ``*_transform_list`` of every model
  and dataset config builds through ``AIGITransformations``; malformed names raise ``Invalid transform name``,
* each image detector's registry transform is its model config's ``eval_transform_list`` (one source of truth),
* the output shape of every eval pipeline on a 300x200 image,
* eval path == inference path: the test transform ``dataset.resolve_transforms`` builds, on ``dataset._decode_rgb``
  (the training loaders), equals the registry transform on ``bench_eval._open_rgb`` (benchmarks / web UI) exactly, on
  PNG images of several sizes; models with a batchable ``test_transform_list`` compare their eval list through the
  training-path wrapper instead, and their test list must batch across image sizes,
* pixel parity with hand-written torchvision v1 pipelines (the official eval code) for CLIP, NPR, PGC (incl. zero
  padding), OmniAID v1, DDA and SAFE,
* a model's train / val / test outputs have the same per-sample rank,
* ``without_resize`` drops only resize / crop steps and refuses pipelines that resize inside,
* ``to_dtype`` equals ``to_tensor`` on a PIL image and ``ToDtype(float32, scale=True)`` on a uint8 tensor, exactly,
* ``resolve_transforms`` reports a DatasetConfig list shadowed by the model's only when it was overridden (differs from
  the dataset yaml).
"""

from __future__ import annotations

import contextlib
import io
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as v2
from huggingface_hub import try_to_load_from_cache
from omegaconf import OmegaConf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "interface"), str(ROOT / "benchmark")]

from transformations import (AIGITransformations, CLIP_MEAN, CLIP_STD, IMAGENET_MEAN,   # noqa: E402
                             IMAGENET_STD)
from backend.registry import list_specs, model_transform                                 # noqa: E402
from bench_eval import _open_rgb                                                         # noqa: E402
from dataset import _decode_rgb, resolve_transforms                                      # noqa: E402
from models.image.cure.preprocess import SRC_FORMAT_KEY                                  # noqa: E402
from models.image.moejepa.moejepa import BACKBONE as VJEPA2                              # noqa: E402

CONF = ROOT / "conf"
IMAGE_MODELS = sorted((CONF / "model" / "image").glob("*.yaml"))
TRAINING_KEYS = ("train_transform_list", "val_transform_list", "test_transform_list")
# (width, height) of the PNG test images: smaller than every crop, non-square both ways, square, long side > 1024.
SIZES = [(300, 200), (200, 150), (150, 260), (512, 512), (1300, 900)]
# Each image model's eval output on a 300x200 image (bfree / spai: native resolution).
EXPECTED_SHAPES = {
    "adapted_dino": (3, 224, 224), "aide": (5, 3, 256, 256), "bfree": (3, 200, 300), "clip": (3, 224, 224),
    "cnn": (3, 224, 224), "co_spy": (3, 384, 384), "cure": (3, 336, 336), "d3qe": (3, 256, 256),
    "dda": (3, 336, 336), "dinov2": (3, 224, 224), "dinov3": (3, 256, 256), "effort": (3, 224, 224),
    "fat_former": (3, 224, 224), "mirror": (3, 224, 224), "moejepa": (3, 256, 256), "npr": (3, 256, 256),
    "omniaid": (3, 336, 336), "omniaid_genimage": (3, 336, 336), "omniaid_v1": (3, 336, 336), "pgc": (3, 224, 224),
    "safe": (3, 256, 256),
    "spai": (3, 200, 300), "vib_net": (3, 224, 224), "vib_net_official": (3, 224, 224),
}
# The official eval code as torchvision v1 pipelines (PIL in).
GOLDEN = {
    "clip": T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)]),
    "npr": T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
    "pgc": T.Compose([T.CenterCrop(224), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
    "omniaid_v1": T.Compose([T.Resize((512, 512)), T.Resize((336, 336)), T.ToTensor(),
                             T.Normalize(CLIP_MEAN, CLIP_STD)]),
    "dda": T.Compose([T.CenterCrop(336), T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)]),
    "safe": T.Compose([T.CenterCrop(256), T.ToTensor()]),
}
# CuRe and MIRROR re-encode PNG sources as JPEG q96 first. The training loaders decode to tensors and lose the source
# format, so their test path treats every image as non-PNG; the parity check compares against that branch of the inference.
FORMAT_DEPENDENT = {"cure", "mirror"}


def _walk(node, prefix: str, found: dict[str, list[str]]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            name = f"{prefix}{key}"
            if str(key).endswith("_transform_list") and isinstance(value, list):
                found[name] = [str(v) for v in value]
            else:
                _walk(value, name + ".", found)


def transform_lists(path: Path) -> dict[str, list[str]]:
    """Every ``*_transform_list`` in a config file (any nesting depth), unresolved."""
    found = {}
    _walk(OmegaConf.to_container(OmegaConf.load(path), resolve=False), "", found)
    return found


def resolved(path: Path):
    """``(train, val, test)`` of a run with model config ``path`` and a dataset config without transform lists."""
    return resolve_transforms(OmegaConf.create({"ModelConfig": OmegaConf.load(path), "DatasetConfig": {}}))


def noise_pil(size=(300, 200)) -> Image.Image:
    arr = np.random.default_rng(0).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def noise_uint8(h=320, w=400) -> torch.Tensor:
    return torch.randint(0, 256, (3, h, w), generator=torch.Generator().manual_seed(0), dtype=torch.uint8)


def write_pngs(directory: Path) -> list[Path]:
    """Gradients plus noise at ``SIZES``, as PNG (lossless, so both decoders return the same pixels)."""
    paths = []
    for i, (w, h) in enumerate(SIZES):
        yy, xx = np.mgrid[0:h, 0:w]
        base = np.stack([xx * 255 / w, yy * 255 / h, (xx + yy) * 127 / (w + h)], axis=-1)
        arr = np.clip(base + np.random.default_rng(i).normal(0, 20, base.shape), 0, 255).astype(np.uint8)
        paths.append(directory / f"img_{w}x{h}.png")
        Image.fromarray(arr, "RGB").save(paths[-1])
    return paths


def vjepa2_cached() -> bool:
    return isinstance(try_to_load_from_cache(VJEPA2, "video_preprocessor_config.json"), str)


def runnable(path: Path) -> bool:
    if "vjepa2_processor" in transform_lists(path)["eval_transform_list"] and not vjepa2_cached():
        print(f"    {path.stem}: skipped ({VJEPA2} processor not in the HF cache)")
        return False
    return True


def inference_transforms() -> dict[str, AIGITransformations]:
    """Model config stem -> the transform the web UI / benchmarks run (the registry spec's, else ``model_transform``)."""
    out = {path.stem: model_transform(f"image/{path.stem}") for path in IMAGE_MODELS}
    for spec in list_specs():
        configs = [o.split("=", 1)[1] for o in spec.overrides if o.startswith("model@ModelConfig=image/")]
        if configs:
            out[configs[0].split("/", 1)[1]] = spec.transform
    return out


def test_lists_build() -> None:
    print("[A] every transform list builds; malformed names are rejected ...")
    for path in IMAGE_MODELS:
        assert len(transform_lists(path).get("eval_transform_list", [])) > 0, f"{path.name}: no eval_transform_list"
    configs = sorted((CONF / "model").rglob("*.yaml")) + sorted((CONF / "dataset").rglob("*.yaml"))
    n = 0
    for path in configs:
        for names in transform_lists(path).values():
            AIGITransformations(names)
            n += 1
    for name in ("resize_300", "resize_short_224", "center_crop_100", "resize_0256", "resize_", "center_crop_abc",
                 "resize_short_256x", "normalise_clip"):
        try:
            AIGITransformations([name])
        except ValueError as e:
            assert str(e).startswith("Invalid transform name"), (name, e)
        else:
            raise AssertionError(f"{name!r} was accepted")
    print(f"    {n} lists in {len(configs)} configs ({len(IMAGE_MODELS)} image models, all with an eval list)")


def test_registry_matches_config() -> None:
    print("[B] registry transform == the model config's eval_transform_list ...")
    n = 0
    for spec in list_specs():
        configs = [o.split("=", 1)[1] for o in spec.overrides if o.startswith("model@ModelConfig=image/")]
        if not configs:
            continue
        expected = transform_lists(CONF / "model" / f"{configs[0]}.yaml")["eval_transform_list"]
        assert isinstance(spec.transform, AIGITransformations), (spec.key, type(spec.transform))
        assert list(spec.transform.transform_list) == expected, (spec.key, spec.transform.transform_list, expected)
        n += 1
    print(f"    {n} image specs")


def test_eval_shapes() -> None:
    print("[C] eval pipelines on a 300x200 PIL image: float tensors of the expected shape ...")
    assert set(EXPECTED_SHAPES) == {p.stem for p in IMAGE_MODELS}, "add the new model to EXPECTED_SHAPES"
    pil = noise_pil()
    for path in IMAGE_MODELS:
        if not runnable(path):
            continue
        x = AIGITransformations(transform_lists(path)["eval_transform_list"])(pil.copy())
        assert isinstance(x, torch.Tensor) and x.is_floating_point() and torch.isfinite(x).all(), (path.stem, type(x))
        assert tuple(x.shape) == EXPECTED_SHAPES[path.stem], (path.stem, tuple(x.shape))
    print(f"    {len(IMAGE_MODELS)} models")


def test_eval_path_is_inference_path() -> None:
    print("[D] test transform on _decode_rgb == registry transform on _open_rgb (torch.equal) ...")
    inference = inference_transforms()
    with tempfile.TemporaryDirectory() as tmp:
        pngs = write_pngs(Path(tmp))
        for path in IMAGE_MODELS:
            if not runnable(path):
                continue
            lists = transform_lists(path)
            test = resolved(path)[2]
            if "test_transform_list" in lists:
                assert test.transform_list == lists["test_transform_list"] and test.pil_input, (path.stem, test)
                shapes = {tuple(test(_decode_rgb(p)).shape) for p in pngs}
                assert len(shapes) == 1, (path.stem, "test_transform_list does not batch", shapes)
                test = AIGITransformations(lists["eval_transform_list"], pil_input=True)
            for p in pngs:
                image = _open_rgb(p)
                if path.stem in FORMAT_DEPENDENT:
                    del image.info[SRC_FORMAT_KEY]
                got, want = test(_decode_rgb(p)), inference[path.stem](image)
                assert got.shape == want.shape and torch.equal(got, want), \
                    (path.stem, p.name, got.shape, want.shape, (got.float() - want.float()).abs().max().item())
            via = " (eval list via the training path)" if "test_transform_list" in lists else ""
            print(f"    {path.stem}: {len(pngs)} images equal{via}")


def test_golden() -> None:
    print("[E] pixel parity with the official torchvision v1 pipelines ...")
    with tempfile.TemporaryDirectory() as tmp:
        pngs = write_pngs(Path(tmp))
        for stem, golden in GOLDEN.items():
            tf = model_transform(f"image/{stem}")
            for p in pngs:
                image = _open_rgb(p)
                assert torch.equal(tf(image), golden(image)), (stem, p.name)
    print(f"    {', '.join(GOLDEN)}: {len(SIZES)} images each (incl. {SIZES[1][0]}x{SIZES[1][1]}: zero-padded crops)")


def test_split_ranks() -> None:
    print("[F] train / val / test outputs of a model have the same per-sample rank ...")
    random.seed(0)
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        image = _decode_rgb(write_pngs(Path(tmp))[-1])
    for path in IMAGE_MODELS:
        if not runnable(path):
            continue
        ranks = {split: tf(image.clone()).ndim for split, tf in zip(("train", "val", "test"), resolved(path))
                 if tf is not None}
        assert len(set(ranks.values())) == 1, (path.stem, ranks)
    print(f"    {len(IMAGE_MODELS)} models")


def test_training_lists_on_tensor() -> None:
    print("[G] train / val / test lists on a uint8 tensor (val / test through PIL, as resolve_transforms builds them) ...")
    random.seed(0)
    torch.manual_seed(0)
    lists = [(path, key, names) for path in IMAGE_MODELS for key, names in transform_lists(path).items()
             if key in TRAINING_KEYS]
    lists += [(path, key, names) for path in sorted((CONF / "dataset" / "image").glob("*.yaml"))
              for key, names in transform_lists(path).items()]
    for path, key, names in lists:
        x = AIGITransformations(names, pil_input=key != "train_transform_list")(noise_uint8())
        assert isinstance(x, torch.Tensor), (path.stem, key, type(x))
        if "to_dtype" in names:
            assert x.is_floating_point() and torch.isfinite(x).all(), (path.stem, key, x.dtype)
    print(f"    {len(lists)} lists")


def test_without_resize() -> None:
    print("[H] without_resize ...")
    tf = AIGITransformations(["resize_short_256", "center_crop_224", "to_dtype", "normalize_clip"], pil_input=True)
    native = tf.without_resize()
    assert native.transform_list == ["to_dtype", "normalize_clip"] and native.pil_input, native
    assert tuple(native(noise_pil()).shape) == (3, 200, 300)
    for stem in ("spai", "cure", "effort", "aide", "moejepa", "mirror"):
        try:
            model_transform(f"image/{stem}").without_resize()
        except ValueError:
            continue
        raise AssertionError(f"{stem}: without_resize accepted a pipeline that resizes inside")


def test_to_dtype() -> None:
    print("[I] to_dtype == to_tensor (PIL) / ToDtype(float32, scale=True) (uint8 tensor) ...")
    to_dtype = AIGITransformations(["to_dtype"])
    pil = noise_pil()
    assert torch.equal(to_dtype(pil), TF.to_tensor(pil))
    t = noise_uint8()
    assert torch.equal(to_dtype(t), v2.ToDtype(torch.float32, scale=True)(t))


def resolve_output(model: str, dataset: str, **data_overrides) -> str:
    """What ``resolve_transforms`` prints for model config ``model`` on dataset config ``dataset`` with overrides."""
    data = OmegaConf.load(CONF / "dataset" / "image" / f"{dataset}.yaml")
    for key, value in data_overrides.items():
        data[key] = value
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        resolve_transforms(OmegaConf.create({"ModelConfig": OmegaConf.load(CONF / "model" / "image" / f"{model}.yaml"),
                                             "DatasetConfig": data}))
    return out.getvalue()


def test_shadow_warning() -> None:
    print("[J] shadowed DatasetConfig lists are reported only when overridden ...")
    assert resolve_output("npr", "genimage") == ""                       # the yaml's own lists: no report
    changed = ["resize_short_256", "center_crop_224", "to_dtype"]
    lines = resolve_output("npr", "genimage", eval_transform_list=changed).splitlines()
    assert len(lines) == 1 and "DatasetConfig.eval_transform_list is ignored" in lines[0], lines
    assert resolve_output("npr", "genimage", name="no_such_dataset", eval_transform_list=changed) == ""


if __name__ == "__main__":
    os.chdir(ROOT)
    test_lists_build()
    test_registry_matches_config()
    test_eval_shapes()
    test_eval_path_is_inference_path()
    test_golden()
    test_split_ranks()
    test_training_lists_on_tensor()
    test_without_resize()
    test_to_dtype()
    test_shadow_warning()
    print("Model transform checks passed")
