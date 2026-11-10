"""CPU checks of the CuRe training recipe (no download: the PE Core base is not fetched, the backbone is random).

* [A] stage-1 data: a validation item is exactly ``(1 - a) x_r + a x_g`` of the endpoints it names, each preprocessed by
  the CuRe eval pipeline (a PNG fake gets the JPEG q96 step); training draws (partner, a) differ between epochs and
  between items; a ~ U(0, 1),
* [B] stage-1 model: trainable = the 96 LoRA tensors + the ratio head; the loss is BCE with the soft target a; gradients
  reach the LoRA and the ratio head; the optimizer groups carry weight decay 0.1 (LoRA) / 0.01 (head),
* [C] gradient checkpointing gives the same features and the same LoRA gradients as the plain forward,
* [D] the trainer's timm_cosine run with conf/trainer/image/cure.yaml: the logged lr at every optimizer step equals the
  intended schedule (linear warmup from 0 over 2 epochs to 1e-5, then cosine to 1e-6 over 3 epochs) and the AdamW
  groups keep their own weight decays,
* [E] stage-2 math (Eq. 6-10) on random features against a direct numpy computation (np.polyfit slopes, covariances,
  the eigenvalues of B^-1 C_F) and scipy.linalg.eigh(C_F, B),
* [G] R3 augmentation (models/image/cure/train_augment.py): disabled it equals the eval pipeline bit for bit; enabled it is
  deterministic in the seed, varies with the seed, leaves the caller's global RNGs untouched; in the train dataset the
  pair / ratio draws are those of the recipe and both endpoints use the same seed; validation is not augmented,
* [F] the export has exactly the released adapter's 100 keys / shapes and loads into a fresh model like it; the
  normalisation fit recovers known (lambda_s, lambda_m, eps).
"""
from __future__ import annotations

import contextlib
import io
import math
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
from omegaconf import OmegaConf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

from dataset import CuReEndpoints, CuReMixDataset, cure_mix                  # noqa: E402
from models.image.cure import subspace as S                                  # noqa: E402
from models.image.cure.cure import SynthScopeCuReModel                       # noqa: E402
from models.image.cure.preprocess import SRC_FORMAT_KEY, CuReTransform       # noqa: E402
from models.image.cure.train_augment import CuReDDAEndpoint                  # noqa: E402
from models.image.cure.train_stages import draw_pairs, fit_normalisation     # noqa: E402
from trainer import SynthScopeModelTrainer                                   # noqa: E402

RELEASED_ADAPTER = Path(os.environ.get("SYNTHSCOPE_CURE_ADAPTER",
                                       "/ivi/zfs/s0/original_homes/ssalehi/SynthScope_shared/cure_ref/CuRe/weights/cure_adapter.pt"))


def model_config(**model) -> OmegaConf:
    return OmegaConf.create({"ModelConfig": {"model_type": "cure", "load_base_backbone": False, **model},
                             "TrainerConfig": {}})


def write_genimage(root: Path, n=4) -> None:
    """A tiny GenImage-layout split: JPEG reals, PNG fakes, various sizes."""
    rng = np.random.default_rng(0)
    for split in ("train", "test"):
        for cls, ext in (("real", "JPEG"), ("fake", "png")):
            (root / split / cls).mkdir(parents=True)
            for i in range(n):
                arr = rng.integers(0, 256, (300 + 37 * i, 420 - 23 * i, 3), dtype=np.uint8)
                Image.fromarray(arr).save(root / split / cls / f"{cls}_{i}.{ext}", quality=90)


def test_mix_dataset() -> None:
    print("[A] stage-1 mixing data ...")
    tf = CuReTransform()
    with tempfile.TemporaryDirectory() as tmp:
        write_genimage(Path(tmp))
        ep = CuReEndpoints(tmp, "test", tf)
        val = CuReMixDataset(ep, "val", val_pairs=3, seed=5)
        assert len(val) == 3
        for idx in range(3):
            g = torch.Generator().manual_seed(5 * 1_000_003 + idx)
            fake_idx = int(torch.randint(len(ep.fakes), (), generator=g))
            alpha = float(torch.rand((), generator=g))
            x, a, _ = val[idx]
            real_img = Image.open(ep.reals[idx]).convert("RGB")
            fake_img = Image.open(ep.fakes[fake_idx]).convert("RGB")
            fake_img.info[SRC_FORMAT_KEY] = "PNG"
            expected = cure_mix(tf(real_img), tf(fake_img), alpha)
            assert float(a) == np.float32(alpha) and torch.allclose(x, expected, atol=1e-6), idx
        # the PNG hint changes the fake's preprocessing (the JPEG q96 re-encoding): it must be applied
        plain = tf(Image.open(ep.fakes[0]).convert("RGB"))
        assert not torch.equal(plain, ep.load(ep.fakes[0])), "PNG fakes must get the JPEG q96 step"
        assert torch.equal(tf(Image.open(ep.reals[0]).convert("RGB")), ep.load(ep.reals[0]))
        # the stage-2 parity control (no final JPEG q70) differs from the default
        img = Image.open(ep.reals[0]).convert("RGB")
        ctrl, base = CuReTransform(eval_jpeg_quality=None)(img), tf(img)
        assert ctrl.shape == base.shape and not torch.equal(ctrl, base) and bool(torch.isfinite(ctrl).all())
        assert torch.equal(CuReTransform(eval_jpeg_quality=70)(img), base)

        train = CuReMixDataset(CuReEndpoints(tmp, "train", tf), "train")
        loader = torch.utils.data.DataLoader(train, batch_size=4, shuffle=True, num_workers=2)
        torch.manual_seed(0)
        epochs = [torch.cat([b[1] for b in loader]) for _ in range(2)]
        assert all(len(a) == 4 for a in epochs) and not torch.equal(epochs[0].sort().values, epochs[1].sort().values)
        assert len(set(epochs[0].tolist())) == 4, "every item draws its own ratio"
    big = torch.tensor([float(torch.rand(())) for _ in range(20000)])
    assert big.min() >= 0 and big.max() < 1 and abs(float(big.mean()) - 0.5) < 0.01
    print("    val items = (1-a) x_r + a x_g of the CuRe-preprocessed endpoints (PNG fake -> JPEG q96); "
          "train draws change every epoch")


def test_train_augment() -> None:
    print("[G] R3 DDA-style endpoint augmentation ...")
    tf = CuReTransform()
    with tempfile.TemporaryDirectory() as tmp:
        write_genimage(Path(tmp))
        ep = CuReEndpoints(tmp, "train", tf)
        off, on = CuReDDAEndpoint(augment=False), CuReDDAEndpoint(augment=True)
        for p, fmt in ((ep.reals[1], "JPEG"), (ep.fakes[2], "PNG")):
            img = Image.open(p).convert("RGB"); img.info[SRC_FORMAT_KEY] = fmt
            assert torch.equal(off(img, 7), tf(img)), "augment=False must equal the eval pipeline"
            states = (torch.get_rng_state(), np.random.get_state()[1].copy())
            a, b, c = on(img, 11), on(img, 11), on(img, 12)
            assert torch.equal(torch.get_rng_state(), states[0]) and np.array_equal(np.random.get_state()[1], states[1])
            assert torch.equal(a, b) and not torch.equal(a, c) and a.shape == (3, 336, 336) and bool(torch.isfinite(a).all())
        # dataset: recipe draws unchanged, one seed per pair shared by both endpoints, val not augmented
        seen = []
        class Spy(CuReDDAEndpoint):
            def __call__(self, img, seed=None):
                seen.append(seed); return super().__call__(img, seed)
        aug = CuReMixDataset(CuReEndpoints(tmp, "train", tf, augment=Spy(augment=True)), "train")
        plain = CuReMixDataset(CuReEndpoints(tmp, "train", tf), "train")
        _, a1, _ = aug[0]; _, a0, _ = plain[0]
        assert len(seen) == 2 and seen[0] == seen[1] and seen[0] is not None, seen
        val = CuReMixDataset(CuReEndpoints(tmp, "test", tf, augment=Spy(augment=True)), "val", val_pairs=2, seed=5)
        n = len(seen); val[0]
        assert len(seen) == n, "validation must not be augmented"
        for idx in range(2):
            torch.manual_seed(idx)
            x1 = aug[idx]; torch.manual_seed(idx); x0 = plain[idx]
            assert float(x1[1]) == float(x0[1]), "the augmentation seed is drawn after the recipe's draws"
    print("    off == eval pipeline; seeded, deterministic, RNG-neutral; one shared seed per pair; val un-augmented")


def test_ratio_stage(model: SynthScopeCuReModel) -> None:
    print("[B] stage-1 model: trainable set, BCE soft target, optimizer groups ...")
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert len(trainable) == 98 and sum("lora_" in n for n in trainable) == 96, len(trainable)
    assert {n for n in trainable if "lora_" not in n} == {"ratio_head.weight", "ratio_head.bias"}
    assert not model.head.weight.requires_grad
    groups = model.optimizer_param_groups()
    assert [len(g["params"]) for g in groups] == [96, 2] and [g["weight_decay"] for g in groups] == [0.1, 0.01]
    images = torch.randn(2, 3, 336, 336) * 0.5
    alpha = torch.tensor([0.25, 0.9])
    model.train()
    out = model.compute_loss((images, alpha, torch.zeros(2)))
    logits = model.ratio_head(model.features(images)).squeeze(-1)
    p = torch.sigmoid(logits)
    manual = -(alpha * torch.log(p) + (1 - alpha) * torch.log(1 - p)).mean()
    assert torch.allclose(out["loss"], manual, atol=1e-5) and torch.allclose(out["outputs"], p, atol=1e-6)
    out["loss"].backward()
    attn = model.backbone.transformer.resblocks[0].attn
    assert attn.lora_B_q.grad is not None and attn.lora_B_q.grad.abs().sum() > 0      # B = 0 at init: only B moves first
    assert model.ratio_head.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    print(f"    {len(trainable)} trainable tensors (96 LoRA + ratio head), loss {float(out['loss'].detach()):.4f} = manual BCE")


def test_grad_checkpointing(model: SynthScopeCuReModel) -> None:
    print("[C] gradient checkpointing == plain forward (features and LoRA gradients) ...")
    torch.manual_seed(1)
    for name, p in model.named_parameters():          # non-zero LoRA B so that every LoRA tensor gets a gradient
        if "lora_B" in name:
            p.data.normal_(std=0.01)
    images = torch.randn(1, 3, 336, 336) * 0.5
    model.train()
    results = []
    for ckpt in (False, True):
        model.grad_checkpointing = ckpt
        model.zero_grad(set_to_none=True)
        feats = model.features(images)
        feats.square().sum().backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if "lora_" in n}
        results.append((feats.detach(), grads))
    model.grad_checkpointing = False
    (f0, g0), (f1, g1) = results
    assert torch.equal(f0, f1), float((f0 - f1).abs().max())
    worst = max(float((g0[n] - g1[n]).abs().max() / (g0[n].abs().max() + 1e-12)) for n in g0)
    assert worst < 1e-5, worst
    with torch.no_grad():
        model.eval()
        assert torch.equal(model.features(images), model.backbone(images))
    model.zero_grad(set_to_none=True)
    print(f"    features identical, LoRA gradients equal (max relative diff {worst:.1e}) over 96 tensors")


class TinyRatio(torch.nn.Module):
    """A stand-in with CuRe's interface: optimizer_param_groups with two weight decays, BCE on a soft target."""

    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Linear(4, 4)
        self.ratio_head = torch.nn.Linear(4, 1)

    def optimizer_param_groups(self):
        return [{"params": list(self.lora_A.parameters()), "weight_decay": 0.1},
                {"params": list(self.ratio_head.parameters()), "weight_decay": 0.01}]

    def compute_loss(self, batch):
        x, a, _ = batch
        logits = self.ratio_head(self.lora_A(x)).squeeze(-1)
        return {"loss": torch.nn.functional.binary_cross_entropy_with_logits(logits, a), "outputs": torch.sigmoid(logits)}


class TinyData:
    def train_dataloader(self):
        g = torch.Generator().manual_seed(0)
        return torch.utils.data.DataLoader(list(zip(torch.randn(40, 4, generator=g), torch.rand(40, generator=g),
                                                    torch.zeros(40))), batch_size=4)

    val_dataloader = train_dataloader

    def test_dataloader(self):
        return None


def intended_lr(t: int, n_iter: int, epochs=5, warmup_epochs=2, lr=1e-5, min_lr=1e-6) -> float:
    """The paper's schedule as read here: linear warmup from 0 over 2 epochs, then cosine from lr to min_lr over the rest
    (index t = the optimizer steps done before the update that sets the lr)."""
    w, total = warmup_epochs * n_iter, epochs * n_iter
    if t < w:
        return lr * t / w
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * (t - w) / (total - w)))


def test_trainer_schedule() -> None:
    print("[D] trainer timm_cosine with conf/trainer/image/cure.yaml: lr per step, weight-decay groups ...")
    SynthScopeModelTrainer._build_model = lambda self, config: TinyRatio()
    trainer_cfg = OmegaConf.load(ROOT / "conf" / "trainer" / "image" / "cure.yaml")
    trainer_cfg.pop("defaults")
    with tempfile.TemporaryDirectory() as tmp:
        trainer_cfg.save_path = tmp
        trainer_cfg.timm_cosine.log_every = 1
        cfg = OmegaConf.create({"TrainerConfig": trainer_cfg, "metrics": [], "logger": {"type": None}})
        trainer = SynthScopeModelTrainer(cfg, TinyData())
        assert [g["weight_decay"] for g in trainer.optimizer.param_groups] == [0.1, 0.01]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            trainer.train()
        assert sorted(p.name for p in Path(tmp).glob("model_*.pth")) == \
            ["model_0.pth", "model_1.pth", "model_2.pth", "model_3.pth", "model_4.pth", "model_best.pth", "model_last.pth"]
    lines = re.findall(r"epoch (\d+) step (\d+)/(\d+): loss [\d.]+, lr ([\d.e+-]+)", out.getvalue())
    n_iter = int(lines[0][2])
    assert n_iter == 10 and len(lines) == 50, (n_iter, len(lines))
    worst = 0.0
    for e, k, _, lr in lines:
        t = int(e) * n_iter + int(k) - 1      # the trainer calls step_update(epoch * n_iter + k) after step k + 1
        want = intended_lr(t, n_iter)
        worst = max(worst, abs(float(lr) - want))
        assert abs(float(lr) - want) <= 5e-4 * max(want, 1e-6) + 1e-12, (e, k, lr, want)
    peak = [float(lr) for e, k, _, lr in lines if int(e) * n_iter + int(k) - 1 == 2 * n_iter][0]
    # (the log prints 4 significant digits)
    assert abs(peak - 1e-5) < 1e-9 and abs(float(lines[-1][3]) - intended_lr(5 * n_iter - 1, n_iter)) < 1e-9
    print(f"    50 logged steps match the schedule (max |diff| {worst:.1e}); peak {peak:.3e} after 2 epochs, "
          f"last {float(lines[-1][3]):.3e} (floor 1e-6)")


def test_subspace_math() -> None:
    print("[E] stage-2 math vs numpy / scipy ...")
    rng = np.random.default_rng(0)
    n, d, k = 300, 24, 6
    grid = np.array(S.GRID)
    z = {t: rng.normal(size=(n, 6, d)) + (grid[None, :, None] * rng.normal(size=(n, 1, d)) * (3 if t == "RF" else 1))
         for t in ("RF", "RR", "FF")}
    summ = {t: S.trajectory_summary(torch.from_numpy(z[t])) for t in z}
    for t in z:
        slopes = np.polyfit(grid, z[t].transpose(1, 0, 2).reshape(6, -1), 1)[0].reshape(n, d)
        assert np.allclose(summ[t][0].numpy(), slopes, atol=1e-10) and np.allclose(summ[t][1].numpy(), z[t].mean(1))
    stats = S.statistics(summ["RF"][0], summ["RR"][0], summ["FF"][0], summ["RF"][1])
    b = {t: np.polyfit(grid, z[t].transpose(1, 0, 2).reshape(6, -1), 1)[0].reshape(n, d) for t in z}
    c_f = b["RF"].T @ b["RF"] / n
    c_m = b["RR"].T @ b["RR"] / n + b["FF"].T @ b["FF"] / n
    c_s = np.cov(z["RF"].mean(1), rowvar=False, bias=True)
    for name, ref in (("C_F", c_f), ("C_M", c_m), ("C_S", c_s)):
        assert np.allclose(stats[name].numpy(), ref / np.trace(ref), atol=1e-12), name
    assert np.allclose(stats["mu"].numpy(), z["RF"].reshape(-1, d).mean(0))
    ls, lm, eps = 0.7, 1.3, 1e-3
    w, gam = S.solve(stats["C_F"], stats["C_M"], stats["C_S"], k=k, lambda_s=ls, lambda_m=lm, eps=eps)
    cf, cm, cs = (stats[x].numpy() for x in ("C_F", "C_M", "C_S"))
    bmat = ls * cs + lm * cm + eps * np.eye(d)
    eig = np.sort(np.linalg.eigvals(np.linalg.solve(bmat, cf)).real)[::-1][:k]      # the direct (non-symmetric) solve
    assert np.allclose(gam.numpy(), eig, rtol=1e-8), (gam.numpy(), eig)
    sg, sv = scipy.linalg.eigh(cf, bmat)
    assert np.allclose(gam.numpy(), sg[::-1][:k], rtol=1e-8)
    wn = w.numpy()
    assert np.allclose(cf @ wn, bmat @ wn @ np.diag(gam.numpy()), atol=1e-10)                # Eq. 10
    assert np.allclose(wn.T @ bmat @ wn, np.eye(k), atol=1e-10)                              # B-orthonormal
    assert np.allclose(np.abs(np.sum(wn * sv[:, ::-1][:, :k], axis=0)) / np.linalg.norm(wn, axis=0) /
                       np.linalg.norm(sv[:, ::-1][:, :k], axis=0), 1, atol=1e-8)              # same vectors as scipy
    assert abs(S.subspace_overlap(w, torch.from_numpy(sv[:, ::-1][:, :k].copy())) - 1) < 1e-10
    pairs = draw_pairs(50, 7, 7, True, torch.Generator().manual_seed(0))
    assert (pairs[:, 0] != pairs[:, 1]).all() and pairs.max() < 7
    print(f"    slopes = polyfit, C_F/C_M/C_S/mu exact; top-{k} eigenvalues = numpy eig(B^-1 C_F) = scipy eigh; "
          f"W^T B W = I")


def test_export(model: SynthScopeCuReModel) -> None:
    print("[F] export: the released adapter's keys, loads like it; normalisation fit ...")
    adapter = model.adapter_state_dict()
    assert len(adapter) == 100 and not any(k.startswith("ratio_head") for k in adapter)
    assert adapter["W"].shape == (1024, 128) and adapter["mu"].shape == (1024,) and adapter["head.weight"].shape == (2, 128)
    if RELEASED_ADAPTER.is_file():
        ref = torch.load(RELEASED_ADAPTER, map_location="cpu", weights_only=True)
        assert set(ref) == set(adapter) and all(ref[k].shape == adapter[k].shape and ref[k].dtype == adapter[k].dtype
                                                for k in ref)
        print(f"    same 100 keys, shapes and dtypes as {RELEASED_ADAPTER.name}")
    fresh = SynthScopeCuReModel(model_config())
    missing, unexpected = fresh.load_state_dict(adapter, strict=False)
    assert not unexpected and not any("lora_" in k or k in ("mu", "W") or k.startswith("head.") for k in missing)
    assert all(torch.equal(fresh.state_dict()[k], v) for k, v in adapter.items())
    # (lambda_s, lambda_m, eps) are recovered from a W solved with them
    rng = np.random.default_rng(1)
    mats = [torch.from_numpy(np.cov(rng.normal(size=(400, 40)) * rng.uniform(0.2, 3, 40), rowvar=False)) for _ in range(3)]
    mats = [S.trace_normalised(m) for m in mats]
    w, _ = S.solve(mats[0], mats[1], mats[2], k=40, lambda_s=0.5, lambda_m=2.0, eps=1e-3)
    fit = fit_normalisation(w, mats[2], mats[1])
    assert abs(fit["lambda_s"] - 0.5) < 1e-6 and abs(fit["lambda_m"] - 2.0) < 1e-6 and abs(fit["eps"] - 1e-3) < 1e-8, fit
    print(f"    100 tensors, reload exact; fit recovers lambda_s 0.5 / lambda_m 2.0 / eps 1e-3 ({fit})")


def main() -> None:
    torch.manual_seed(0)
    test_mix_dataset()
    test_train_augment()
    test_subspace_math()
    test_trainer_schedule()
    if os.environ.get("CURE_RECIPE_SKIP_MODEL"):            # the ViT-L parts take a few CPU minutes
        print("model checks skipped (CURE_RECIPE_SKIP_MODEL)")
        return
    model = SynthScopeCuReModel(model_config(training_stage="ratio"))
    test_ratio_stage(model)
    test_grad_checkpointing(model)
    test_export(model)
    print("ALL CURE RECIPE CHECKS PASSED")


if __name__ == "__main__":
    main()
