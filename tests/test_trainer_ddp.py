"""DDP training through the trainer equals single-process training (CPU, gloo, 2 processes; no GPU, no data, no download).

``SynthScopeModelTrainer`` calls ``model.compute_loss(batch)``, which a ``DistributedDataParallel`` wrapper does not have;
``_compute_loss`` runs it through the wrapper's ``forward`` so the gradient reducer fires. Checked here, with a tiny model
(a trainable linear layer behind frozen parameters, like a frozen backbone), on a global batch of 16 samples per step:

* ``train_epoch`` (no accumulation): 2 processes x 8 equal one process x 16, and both ranks hold identical weights,
* ``_train_with_timm_cosine`` (AIDE's path, with accumulation, 2 epochs): 2 processes x micro-batch 4 x accumulation 2 give
  the weights of one process x 16, of one process x 4 x accumulation 4, and of the same DDP run with ``no_sync`` disabled;
  the non-stepping micro-batches run under ``accelerator.no_sync`` (counted), the ranks stay identical,
* ``_train_with_step_budget`` (PGC/DDA path) with a final partial accumulation (5 micro-batches, accumulation 2): the last
  micro-batch syncs, so the ranks agree and match one process x 8,
* evaluation (``torch.no_grad`` + ``eval()``) works through the wrapper, and the inner module's ``forward`` is restored.

Run as a script: python tests/test_trainer_ddp.py
"""

from __future__ import annotations

import contextlib
import os
import socket
import sys
import tempfile
from pathlib import Path

import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
N, FEAT = 32, 4      # samples; one epoch is 2 global steps of 16


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.frozen = torch.nn.Linear(FEAT, FEAT)
        self.frozen.requires_grad_(False)
        self.head = torch.nn.Linear(FEAT, 2)

    def forward(self, x):
        return self.head(torch.relu(self.frozen(x)))

    def compute_loss(self, batch):
        x, y, _ = batch
        out = self(x)
        return {"loss": torch.nn.functional.cross_entropy(out, y), "outputs": out}


class _Logger:
    def log_metrics(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def data():
    torch.manual_seed(0)
    return torch.randn(N, FEAT), torch.randint(0, 2, (N,)), torch.zeros(N, dtype=torch.long)


def build(micro):
    from accelerate import Accelerator
    from trainer import SynthScopeModelTrainer
    x, y, f = data()
    loader = torch.utils.data.DataLoader(list(zip(x, y, f)), batch_size=micro, shuffle=False)
    t = SynthScopeModelTrainer.__new__(SynthScopeModelTrainer)
    t.accelerator = Accelerator(cpu=True)
    t.model = Tiny()
    t.optimizer = torch.optim.SGD([p for p in t.model.parameters() if p.requires_grad], lr=0.1)
    t.scheduler, t.use_metrics, t.global_step, t.logger = None, False, 0, _Logger()
    t.model, t.optimizer, t.train_loader = t.accelerator.prepare(t.model, t.optimizer, loader)
    t.no_sync_calls = 0
    return t


def count_no_sync(t, enabled):
    """Count the accelerator.no_sync calls; with ``enabled=False`` they do nothing (the all-reduce runs on every micro-batch)."""
    real = t.accelerator.no_sync

    def no_sync(model):
        t.no_sync_calls += 1
        return real(model) if enabled else contextlib.nullcontext()

    t.accelerator.no_sync = no_sync


def run(mode, micro, accum, no_sync=True):
    t = build(micro)
    count_no_sync(t, no_sync)
    t.val = lambda: (0.0, {})
    t._save_model = lambda *args: None
    if mode == "train_epoch":
        t.train_epoch()
    elif mode == "timm_cosine":
        t.config = OmegaConf.create({"TrainerConfig": {"epochs": 2, "lr": 0.1, "save": False, "save_path": "unused"}})
        t._train_with_timm_cosine(OmegaConf.create({"warmup_epochs": 0, "warmup_lr": 0.0, "min_lr": 0.01,
                                                    "grad_accum_steps": accum, "log_every": 1000}))
    else:
        t.config = OmegaConf.create({"TrainerConfig": {"epochs": 1, "save_path": "unused"}})
        t._train_with_step_budget(OmegaConf.create({"grad_accum_steps": accum, "max_steps": 5, "save_every": 1000}))
    inner = t.accelerator.unwrap_model(t.model)
    assert "forward" not in inner.__dict__, "the inner forward was not restored"
    t.model.eval()
    x, y, f = data()
    with torch.no_grad():
        t._compute_loss((x[:4], y[:4], f[:4]))
    return inner.head.weight.detach().clone(), inner.head.bias.detach().clone(), t.global_step, t.no_sync_calls


def worker(rank, world, port, mode, micro, accum, no_sync, out):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world), MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    w, b, steps, calls = run(mode, micro, accum, no_sync)
    torch.save({"w": w, "b": b, "steps": steps, "calls": calls}, f"{out}.{rank}")


def free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def ddp(mode, micro, accum, tmp, no_sync=True):
    out = f"{tmp}/{mode}_{micro}_{accum}_{int(no_sync)}"
    mp.spawn(worker, args=(2, free_port(), mode, micro, accum, no_sync, out), nprocs=2, join=True)
    r0, r1 = torch.load(f"{out}.0"), torch.load(f"{out}.1")
    assert torch.equal(r0["w"], r1["w"]) and torch.equal(r0["b"], r1["b"]), f"{mode}: the ranks diverged"
    assert r0["steps"] == r1["steps"] and r0["calls"] == r1["calls"]
    return r0


def close(a, b, what):
    err = max((a["w"] - b["w"]).abs().max().item(), (a["b"] - b["b"]).abs().max().item())
    assert err < 1e-6, f"{what}: max difference {err}"


def single(mode, micro, accum):
    w, b, steps, calls = run(mode, micro, accum)
    return {"w": w, "b": b, "steps": steps, "calls": calls}


def test_ddp_equals_single_process():
    with tempfile.TemporaryDirectory() as tmp:
        # train_epoch, no accumulation: 2 x 8 = 16
        ref = single("train_epoch", 16, 1)
        got = ddp("train_epoch", 8, 1, tmp)
        assert got["steps"] == ref["steps"] == 2
        close(got, ref, "train_epoch")
        # timm_cosine with accumulation, 2 epochs x 2 steps: 2 procs x 4 x 2 = 16
        ref = single("timm_cosine", 16, 1)
        assert ref["steps"] == 4 and ref["calls"] == 0
        assert single("timm_cosine", 4, 4)["steps"] == 4
        close(single("timm_cosine", 4, 4), ref, "timm_cosine, one process x 4 x 4")
        with_sync = ddp("timm_cosine", 4, 2, tmp, no_sync=True)
        without = ddp("timm_cosine", 4, 2, tmp, no_sync=False)
        assert with_sync["steps"] == without["steps"] == 4
        assert with_sync["calls"] == 4 == 2 * 2 * (2 - 1), with_sync["calls"]   # epochs x steps x non-stepping micro-batches
        close(with_sync, ref, "timm_cosine, 2 procs x 4 x 2 vs one process x 16")
        close(with_sync, without, "timm_cosine, with vs without no_sync")
        # step_budget: 5 micro-batches of 8 (a final partial accumulation) = 3 optimizer steps
        ref = single("step_budget", 8, 2)
        got = ddp("step_budget", 4, 2, tmp)
        close(got, ref, "step_budget with a partial last accumulation")


if __name__ == "__main__":
    test_ddp_equals_single_process()
    print("DDP trainer checks passed")
