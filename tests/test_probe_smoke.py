"""Smoke test for the PROBE integration (CPU, no downloads).

Checks:
* the registry transforms of ``probe`` (``probe_pad_repeat_336``, ``to_dtype``, ``normalize_imagenet``,
  ``sliding_patches_336``) and ``probe_resnet`` (``probe_pad_224``, ..., ``sliding_patches_224``) equal the official
  evaluate_dino.py / evaluate_resnet.py preprocessing written out independently (repeat-tiling or black-square padding,
  ToTensor + ImageNet Normalize, row-major non-overlapping patches; edge pixels past the last full patch dropped) on
  small, exact-size, non-square and large images, bit for bit,
* a tiny-backbone ``SynthScopePROBEModel``: official key layout (``backbone.embeddings.*``,
  ``backbone.encoder.layer.*``, ``fc.*``), ``forward`` on a [B, N, 3, S, S] patch bag = the mean of the official
  ``dino_classifier`` patch logits, [B, 1], independent of ``patch_chunk_size``; a [B, 3, S, S] batch is one patch
  each; ``compute_loss`` + backward,
* ``SynthScopePROBEResNetModel`` = the official ``resnet50(num_classes=1, classifier_type='linear')`` key for key,
  same patch-mean forward,
* the engine: ``_fake_prob`` = sigmoid(mean patch logit) for both keys, the strict ``official_probe`` load through
  ``remap_probe`` (``{"model_state_dict": ...}``) accepts the exact state dict and raises on a missing or an unexpected
  key, the model-input preview handles a patch bag, the input notes,
* the full-size models (built on the meta device) have exactly the keys and shapes of the released checkpoints, if
  ``$SYNTHSCOPE_CKPT_DIR/probe.pth`` / ``probe_resnet.pth`` exist (skipped otherwise).
"""

from __future__ import annotations

import dataclasses
import os
import sys
import tempfile

import numpy as np
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "interface")]

from backend.engine import Engine, _load_checkpoint                         # noqa: E402
from backend.hf_compat import HF_RENAMES, translate_hf_keys                 # noqa: E402
from backend.registry import get_spec, input_note, remap_probe              # noqa: E402
from models.image.probe.official.dino_classifier import DINOV2_REG_LARGE, dino_classifier   # noqa: E402
from models.image.probe.official.resnet50 import resnet50                   # noqa: E402
from models.image.probe.probe import SynthScopePROBEModel, SynthScopePROBEResNetModel   # noqa: E402

TINY = {"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4}
IMAGENET = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# (width, height): under the crop both ways, exact, one side under, non-square, large; also a grayscale image
SIZES = [(128, 96), (180, 120), (224, 224), (336, 336), (300, 1200), (335, 700), (336, 500), (513, 700), (1258, 833),
         (2048, 1365)]


def known_key(sd: dict, key: str) -> bool:
    """``key`` is in ``sd`` under the naming of either transformers release (attention.attention.query vs attention.q_proj)."""
    if key in sd:
        return True
    return any((a in key and key.replace(a, b, 1) in sd) or (b in key and key.replace(b, a, 1) in sd)
               for a, b in HF_RENAMES)


def other_naming(sd: dict) -> dict:
    """``sd`` with the attention parameters renamed to the other transformers release's names (what a checkpoint saved
    under that release holds)."""
    out = {}
    for k, v in sd.items():
        for a, b in HF_RENAMES:
            if a in k:
                k = k.replace(a, b, 1)
                break
            if b in k:
                k = k.replace(b, a, 1)
                break
        out[k] = v
    return out


def noise_image(size: tuple[int, int], mode: str = "RGB") -> Image.Image:
    gen = torch.Generator().manual_seed(size[0] * 7 + size[1])
    arr = torch.randint(0, 256, (size[1], size[0], 3), generator=gen, dtype=torch.uint8).numpy()
    return Image.fromarray(arr, "RGB").convert(mode).convert("RGB")


def official_patches(img: Image.Image, ps: int, mode: str) -> torch.Tensor:
    """The official test-time preprocessing, written out: pad_image (repeat tiling / black square), ToTensor +
    ImageNet Normalize on the whole image, then the row-major non-overlapping ps x ps patches."""
    w, h = img.size
    if w < ps or h < ps:
        if mode == "repeat":
            arr = np.tile(np.asarray(img), (-(-ps // h), -(-ps // w), 1))
            img = Image.fromarray(arr)
        else:
            canvas = Image.new("RGB", (ps, ps))
            canvas.paste(img, ((ps - w) // 2, (ps - h) // 2))
            img = canvas
    x = IMAGENET(T.ToTensor()(img))
    rows, cols = x.shape[1] // ps, x.shape[2] // ps
    return torch.stack([x[:, r * ps:(r + 1) * ps, c * ps:(c + 1) * ps] for r in range(rows) for c in range(cols)])


def tiny_model(chunk: int = 0) -> SynthScopePROBEModel:
    cfg = {"ModelConfig": {"model_type": "probe", "backbone_config": TINY, "patch_chunk_size": chunk}}
    return SynthScopePROBEModel(OmegaConf.create(cfg)).eval()


def test_transforms() -> None:
    print("[A] registry transforms == the official evaluate_dino.py / evaluate_resnet.py preprocessing ...")
    for key, ps, mode in (("probe", 336, "repeat"), ("probe_resnet", 224, "pad")):
        tf = get_spec(key).transform
        expected = ([f"probe_pad_repeat_{ps}"] if mode == "repeat" else [f"probe_pad_{ps}"]) + \
            ["to_dtype", "normalize_imagenet", f"sliding_patches_{ps}"]
        assert list(tf.transform_list) == expected, (key, tf.transform_list)
        assert get_spec(key).native_res, key
        for size in SIZES:
            for img in (noise_image(size), noise_image(size, "L")):
                x = tf(img)
                want = official_patches(img, ps, mode)
                assert x.shape == want.shape and torch.equal(x, want), (key, size, tuple(x.shape), tuple(want.shape))
                assert x.shape[0] == max(1, size[0] // ps) * max(1, size[1] // ps) or mode == "pad", (key, size)
        # 'pad' with one long side: a single centered patch (the long side is cut)
        assert tf(noise_image((300, 200))).shape[0] == 1
    print(f"    {len(SIZES)} sizes x RGB / grayscale, both variants, bit-exact")


def test_dino_model() -> None:
    print("[B] tiny PROBE-DINOv2: key layout, patch-mean forward, chunking, loss ...")
    torch.manual_seed(0)
    model = tiny_model()
    sd = model.state_dict()
    for k in ("backbone.embeddings.cls_token", "backbone.embeddings.register_tokens", "backbone.embeddings.mask_token",
              "backbone.encoder.layer.0.attention.attention.query.weight",
              "backbone.encoder.layer.1.layer_scale2.lambda1",
              "backbone.layernorm.weight", "fc.weight", "fc.bias"):
        assert known_key(sd, k), k
    assert tuple(sd["fc.weight"].shape) == (1, TINY["hidden_size"])
    # the vendored official class with the same config and weights
    official = dino_classifier(model_name=dict(DINOV2_REG_LARGE, **TINY)).eval()
    official.load_state_dict(sd, strict=True)
    bag = torch.randn(2, 3, 3, 28, 28)
    with torch.no_grad():
        out = model(bag)
        want = official(bag.flatten(0, 1)).view(2, 3, 1).mean(dim=1)
        assert out.shape == (2, 1) and torch.allclose(out, want, atol=1e-6), (out, want)
        for chunk in (1, 2, 5):
            other = tiny_model(chunk)
            other.load_state_dict(sd, strict=True)
            assert torch.allclose(other(bag), out, atol=1e-6), chunk
        single = model(bag[:, 0])
        assert single.shape == (2, 1) and torch.allclose(single, official(bag[:, 0]), atol=1e-6)
    model.train()
    loss = model.compute_loss((torch.randn(2, 3, 28, 28), torch.tensor([0, 1])))
    assert torch.isfinite(loss["loss"]) and loss["outputs"].shape == (2, 2)
    assert torch.equal(loss["outputs"][:, 0], -loss["outputs"][:, 1])
    loss["loss"].backward()
    assert model.fc.weight.grad is not None
    print(f"    {len(sd)} tensors, [B, N, 3, S, S] -> [B, 1] = mean official patch logit, chunk-invariant")


def test_resnet_model() -> None:
    print("[C] PROBE-ResNet50 = the official resnet50(num_classes=1, classifier_type='linear') ...")
    torch.manual_seed(0)
    model = SynthScopePROBEResNetModel(OmegaConf.create({"ModelConfig": {"model_type": "probe_resnet",
                                                                         "patch_chunk_size": 2}})).eval()
    official = resnet50(num_classes=1, classifier_type="linear").eval()
    assert {k: v.shape for k, v in model.state_dict().items()} == {k: v.shape for k, v in official.state_dict().items()}
    official.load_state_dict(model.state_dict(), strict=True)
    bag = torch.randn(1, 3, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model(bag), official(bag[0]).mean(dim=0, keepdim=True), atol=1e-6)
    print(f"    {len(official.state_dict())} tensors, same names and shapes; patch-mean forward")


def test_engine() -> None:
    print("[D] engine: P(fake) = sigmoid(mean patch logit), strict official load, preview, notes ...")
    engine = Engine(device="cpu")
    model = tiny_model()
    spec = get_spec("probe")
    bag = torch.randn(1, 4, 3, 28, 28)
    with torch.no_grad():
        p = engine._fake_prob(model, spec, bag)
        assert p.shape == (1,) and torch.allclose(p, torch.sigmoid(model(bag)[:, 0]))
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    assert remap_probe({"model_state_dict": sd}) is sd and remap_probe(sd) is sd
    with tempfile.TemporaryDirectory() as tmp:
        for case in ("exact", "renamed", "missing", "unexpected"):
            # "renamed": the attention parameters under the other transformers release's names, as a checkpoint saved
            # with that release holds them; the engine maps them onto this release's names (hf_compat) and loads strictly
            ckpt = other_naming(sd) if case == "renamed" else dict(sd)
            if case == "renamed":
                assert set(ckpt) != set(sd)
            if case == "missing":
                ckpt.pop("fc.bias")
            elif case == "unexpected":
                ckpt["head.weight"] = torch.zeros(1)
            path = os.path.join(tmp, f"{case}.pth")
            torch.save({"model_state_dict": ckpt}, path)
            target = tiny_model()
            try:
                status = _load_checkpoint(target, dataclasses.replace(spec, checkpoint=path))
            except RuntimeError as e:
                assert case != "exact" and "checkpoint mismatch" in str(e), (case, e)
                continue
            assert case in ("exact", "renamed") and "missing=0, unexpected=0" in status, (case, status)
            with torch.no_grad():
                assert torch.equal(target(bag), model(bag))
    url = engine._model_input_data_url(spec, spec.transform(noise_image((700, 513))).unsqueeze(0))
    assert url.startswith("data:image/png;base64,")
    for key in ("probe", "probe_resnet"):
        assert "patches" in input_note(get_spec(key))["label"], key
    print("    exact and renamed-key loads pass, missing / unexpected keys raise; preview of a 2-patch bag renders")


def test_released_layout() -> None:
    ckpt_dir = os.environ.get("SYNTHSCOPE_CKPT_DIR", "")
    for key, cls in (("probe", SynthScopePROBEModel), ("probe_resnet", SynthScopePROBEResNetModel)):
        path = os.path.join(ckpt_dir, f"{key}.pth")
        if not os.path.isfile(path):
            print(f"[E] {key}: released checkpoint layout skipped ($SYNTHSCOPE_CKPT_DIR/{key}.pth not found)")
            continue
        print(f"[E] {key}: full-size model (meta device) vs the released checkpoint ...")
        with torch.device("meta"):
            model = cls(OmegaConf.create({"ModelConfig": {"model_type": key}}))
        ours = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        released = remap_probe(torch.load(path, map_location="cpu", weights_only=False, mmap=True))
        released, _ = translate_hf_keys(released, model.state_dict())     # the names of this transformers release
        assert ours == {k: tuple(v.shape) for k, v in released.items()}, key
        print(f"    {len(ours)} tensors: same names and shapes")


if __name__ == "__main__":
    test_transforms()
    test_dino_model()
    test_resnet_model()
    test_engine()
    test_released_layout()
    print("PROBE smoke test passed")
