"""DDP gradients through the trainer equal single-process gradients (CPU, gloo, 2 processes; no GPU, no data, no download).

``SynthScopeModelTrainer`` calls ``model.compute_loss(batch)``, which a ``DistributedDataParallel`` wrapper does not have;
``_compute_loss`` runs it through the wrapper's ``forward`` so the gradient reducer fires. Checked here, with a tiny model
(a trainable linear layer with unused-in-loss frozen parameters, like a frozen backbone):

* one optimizer step over a global batch of 16 (2 processes x micro-batch 4 x accumulation 2) gives the weights of one
  single-process step over the same 16 samples, and both ranks hold identical weights,
* the same with accumulation 1 (2 x 8),
* evaluation (``torch.no_grad`` + ``eval()``) works through the wrapper, and the inner module's ``forward`` is restored.

Run as a script: python tests/test_trainer_ddp.py
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
N, FEAT = 16, 4


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


def data():
    torch.manual_seed(0)
    return torch.randn(N, FEAT), torch.randint(0, 2, (N,)), torch.zeros(N, dtype=torch.long)


def run(micro, accum):
    """Train one epoch with the trainer's own train_epoch on whatever process layout accelerate detects."""
    from accelerate import Accelerator
    from trainer import SynthScopeModelTrainer
    x, y, f = data()
    loader = torch.utils.data.DataLoader(list(zip(x, y, f)), batch_size=micro, shuffle=False)
    t = SynthScopeModelTrainer.__new__(SynthScopeModelTrainer)
    t.accelerator = Accelerator(cpu=True)
    t.model = Tiny()
    t.optimizer = torch.optim.SGD([p for p in t.model.parameters() if p.requires_grad], lr=0.1)
    t.scheduler, t.use_metrics, t.global_step = None, False, 0
    t.model, t.optimizer, t.train_loader = t.accelerator.prepare(t.model, t.optimizer, loader)
    t.train_epoch(accum=accum)
    inner = t.accelerator.unwrap_model(t.model)
    assert "forward" not in inner.__dict__, "the inner forward was not restored"
    t.model.eval()
    with torch.no_grad():
        val = t._compute_loss((x[:4], y[:4], f[:4]))["loss"].item()
    return inner.head.weight.detach().clone(), t.global_step, val


def worker(rank, world, port, micro, accum, out):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world), MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    w, steps, val = run(micro, accum)
    torch.save({"w": w, "steps": steps, "val": val}, f"{out}.{rank}")


def free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def check(micro, accum, tmp):
    ref_w, ref_steps, _ = run(N, 1)                      # single process, one batch of 16
    out = f"{tmp}/r{micro}_{accum}"
    mp.spawn(worker, args=(2, free_port(), micro, accum, out), nprocs=2, join=True)
    r0, r1 = torch.load(f"{out}.0"), torch.load(f"{out}.1")
    assert r0["steps"] == r1["steps"] == ref_steps == 1, (r0["steps"], r1["steps"], ref_steps)
    assert torch.equal(r0["w"], r1["w"]), "the ranks diverged"
    assert (r0["w"] - ref_w).abs().max() < 1e-6, (r0["w"], ref_w)


def test_ddp_equals_single_process():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        check(micro=4, accum=2, tmp=tmp)     # 2 procs x 4 x 2 = 16
        check(micro=8, accum=1, tmp=tmp)     # 2 procs x 8 x 1 = 16


if __name__ == "__main__":
    test_ddp_equals_single_process()
    print("DDP trainer checks passed")
