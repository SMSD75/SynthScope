"""CPU checks of the trainer's ``lr_step`` schedule and gradient accumulation (no GPU, no data, no model download):

* the per-epoch learning rate for ``phase`` 0 (NPR's rule) and ``phase`` 1 (StepLR(10)) with every 10, factor 0.9, 25 epochs,
* accumulating N micro-batches gives exactly the update of one batch of N x micro-batch,
* the shorter last step of an epoch averages over the micro-batches it has (not over ``accum``).

Run as a script (pytest is optional): python tests/test_trainer_lr_step.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trainer import SynthScopeModelTrainer  # noqa: E402


class Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.l = torch.nn.Linear(4, 2)

    def compute_loss(self, batch):
        x, y, _ = batch
        out = self.l(x)
        return {"loss": torch.nn.functional.cross_entropy(out, y), "outputs": out}


def make_trainer(n, bs, lr=0.1):
    torch.manual_seed(0)
    x, y, f = torch.randn(n, 4), torch.randint(0, 2, (n,)), torch.zeros(n, dtype=torch.long)
    loader = torch.utils.data.DataLoader(list(zip(x, y, f)), batch_size=bs, shuffle=False)
    t = SynthScopeModelTrainer.__new__(SynthScopeModelTrainer)
    t.accelerator = Accelerator(cpu=True)
    t.model = Linear()
    t.optimizer = torch.optim.SGD(t.model.parameters(), lr=lr)
    t.scheduler, t.use_metrics, t.global_step = None, False, 0
    t.model, t.optimizer, t.train_loader = t.accelerator.prepare(t.model, t.optimizer, loader)
    return t


def weights_after(n, bs, accum):
    t = make_trainer(n, bs)
    t.train_epoch(accum=accum)
    return t.model.l.weight.detach().clone(), t.global_step


def lr_per_epoch(phase, epochs=25, every=10, factor=0.9, lr0=4e-4):
    """The learning rate each epoch TRAINS with, driving the trainer's own _train_with_lr_step on a stub."""
    t = SynthScopeModelTrainer.__new__(SynthScopeModelTrainer)
    t.accelerator = SimpleNamespace(print=lambda *a, **k: None)
    t.optimizer = SimpleNamespace(param_groups=[{"lr": lr0}])
    t.logger = SimpleNamespace(log_metrics=lambda *a, **k: None, finish=lambda: None)
    t.global_step = 0
    seen = []

    def train_epoch(accum=1, log_every=0):
        seen.append(t.optimizer.param_groups[0]["lr"])
        return 0.0, {}

    t.train_epoch = train_epoch
    t.val = lambda: (0.0, {})
    t._val_accuracy = lambda m: 0.0
    t.config = OmegaConf.create({"TrainerConfig": {"epochs": epochs, "save": False}})
    t._train_with_lr_step(OmegaConf.create({"every": every, "factor": factor, "phase": phase}))
    return seen


def test_lr_schedule():
    for phase, first_drop in ((0, 11), (1, 10)):      # the 0-based epoch that first trains at the decayed rate
        lrs = lr_per_epoch(phase)
        want = [4e-4 * 0.9 ** ((e >= first_drop) + (e >= first_drop + 10)) for e in range(25)]
        assert len(lrs) == 25 and all(abs(a - b) < 1e-12 for a, b in zip(lrs, want)), (phase, lrs)
    assert lr_per_epoch(1)[9] == 4e-4 and abs(lr_per_epoch(1)[10] - 3.6e-4) < 1e-12 and abs(lr_per_epoch(1)[20] - 3.24e-4) < 1e-12
    assert lr_per_epoch(0)[10] == 4e-4 and abs(lr_per_epoch(0)[11] - 3.6e-4) < 1e-12 and abs(lr_per_epoch(0)[21] - 3.24e-4) < 1e-12


def test_accumulation_equals_big_batch():
    big, s_big = weights_after(40, 20, 1)
    acc, s_acc = weights_after(40, 5, 4)
    assert s_big == s_acc == 2 and (big - acc).abs().max() < 1e-7, (big, acc)


def test_partial_last_step_is_a_mean():
    """40 samples, micro-batch 5, accum 3: steps over micro-batches [0-2], [3-5], [6-7]; the last step has 2 and must
    equal one SGD step on the mean over those 10 samples."""
    got, steps = weights_after(40, 5, 3)
    assert steps == 3
    t = make_trainer(40, 10)                     # reference: the same three steps with batches of 15, 15, 10 samples
    ref = Linear(); opt = torch.optim.SGD(ref.parameters(), lr=0.1)
    x = torch.stack([b[0] for b in t.train_loader.dataset]); y = torch.stack([b[1] for b in t.train_loader.dataset])
    for lo, hi in ((0, 15), (15, 30), (30, 40)):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(ref.l(x[lo:hi]), y[lo:hi]).backward()
        opt.step()
    assert (got - ref.l.weight.detach()).abs().max() < 1e-6, (got, ref.l.weight)


if __name__ == "__main__":
    test_lr_schedule()
    test_accumulation_equals_big_batch()
    test_partial_last_step_is_a_mean()
    print("Trainer lr_step / accumulation checks passed")
