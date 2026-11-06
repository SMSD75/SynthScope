"""Backward compatibility of ``SynthScopeModelTrainer._train_with_step_budget`` (no GPU, no download): a tiny model on random data,
counting the checkpoints saved and the loss / lr lines printed.

* ``unit`` unset (every existing recipe: PGC, DDA, ...): ``max_steps`` / ``save_every`` / ``log_every`` count micro-batches, the
  checkpoints are ``model_step<micro-batch>.pth``, ``log_every`` unset logs with each checkpoint (the old behaviour),
* ``unit: optimizer_step`` with 2 accumulated micro-batches: the same counts in optimizer steps, checkpoints named by them,
* the optimizer takes ``max_steps / accum`` steps either way, at the configured lr.
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

from trainer import SynthScopeModelTrainer   # noqa: E402


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def compute_loss(self, batch):
        x, y, _ = batch
        out = self.fc(x)
        return {"loss": nn.functional.cross_entropy(out, y), "outputs": out}


class Data:
    def train_dataloader(self):
        g = torch.Generator().manual_seed(0)
        return torch.utils.data.DataLoader(list(zip(torch.randn(24, 4, generator=g), torch.randint(0, 2, (24,), generator=g), torch.zeros(24, dtype=torch.long))), batch_size=2)

    val_dataloader = test_dataloader = train_dataloader


def run(budget: dict) -> tuple[list[str], list[str], int]:
    SynthScopeModelTrainer._build_model = lambda self, config: Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        cfg = OmegaConf.create({"TrainerConfig": {"lr": 1e-3, "weight_decay": 0.0, "epochs": 1, "save_path": tmp, "warmup": {"mode": "off"},
                                                  "step_budget": budget}, "metrics": [], "logger": {"type": None}})
        trainer = SynthScopeModelTrainer(cfg, Data())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            trainer.train()
        lines = [l for l in out.getvalue().splitlines() if re.match(r"step \d+ \(optimizer step", l)]
        return sorted(p.name for p in Path(tmp).glob("model_step*.pth")), lines, trainer.global_step


def test_micro_batch_unit() -> None:
    print("[A] unit unset: counts in micro-batches, log with each checkpoint ...")
    saved, lines, opt_steps = run({"max_steps": 12, "save_every": 4, "grad_accum_steps": 2, "max_grad_norm": 0.0})
    assert saved == ["model_step12.pth", "model_step4.pth", "model_step8.pth"], saved
    assert [l.split(" ")[1] for l in lines] == ["4", "8", "12"] and opt_steps == 6, (lines, opt_steps)
    saved, lines, _ = run({"max_steps": 12, "save_every": 4, "log_every": 2, "grad_accum_steps": 2, "max_grad_norm": 0.0})
    assert len(saved) == 3 and [l.split(" ")[1] for l in lines] == ["2", "4", "6", "8", "10", "12"], (saved, lines)
    assert all(l.endswith("lr 1.000e-03") for l in lines)
    print(f"    3 checkpoints, 3 log lines (6 with log_every 2), {opt_steps} optimizer steps")


def test_optimizer_step_unit() -> None:
    print("[B] unit: optimizer_step: counts in optimizer steps (2 micro-batches each), checkpoints named by them ...")
    saved, lines, opt_steps = run({"unit": "optimizer_step", "max_steps": 6, "save_every": 2, "log_every": 1, "grad_accum_steps": 2, "max_grad_norm": 0.0})
    assert saved == ["model_step2.pth", "model_step4.pth", "model_step6.pth"], saved
    assert [l.split(" ")[1] for l in lines] == ["1", "2", "3", "4", "5", "6"] and opt_steps == 6, (lines, opt_steps)
    print(f"    3 checkpoints, 6 log lines, {opt_steps} optimizer steps (same update count as [A])")


if __name__ == "__main__":
    os.chdir(ROOT)
    test_micro_batch_unit()
    test_optimizer_step_unit()
    print("Trainer step_budget checks passed")
