"""CPU check of ``trainer.SynthScopeModelTrainer._train_with_timm_cosine`` with ``select: accuracy`` (AIDE's recipe).

A tiny linear model and scripted validation results, no data or GPU:

* ``model_best.pth`` follows the validation ACCURACY (first epoch of the highest value, strict improvement), not the
  validation loss: the scripted losses rise every epoch, so a loss-selecting run would keep only epoch 0,
* ``save_all`` keeps every epoch and ``model_last.pth``; ``val_losses.json`` holds ``val_accuracy`` and the best epoch,
* the learning rate follows the half-cosine from lr to min_lr, with the first step at the base lr,
* the header line reports the effective batch (processes x micro-batch x accumulation), not 0.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer import SynthScopeModelTrainer

ACCURACIES = [0.60, 0.90, 0.90, 0.80, 0.95]       # real = fake = accuracy; the tie at epoch 2 is not an improvement
LOSSES = [0.1, 0.2, 0.3, 0.4, 0.5]                # rising: a loss-selecting run would keep epoch 0 only
MICRO, ACCUM, EPOCHS = 8, 32, len(ACCURACIES)     # effective batch 256 on one process


class _Logger:
    def log_metrics(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def build(save_path: str) -> SynthScopeModelTrainer:
    cfg = OmegaConf.create({"TrainerConfig": {
        "epochs": EPOCHS, "lr": 5e-4, "save": True, "save_path": save_path,
        "timm_cosine": {"warmup_epochs": 0, "warmup_lr": 0.0, "min_lr": 1e-6, "grad_accum_steps": ACCUM,
                        "save_all": True, "select": "accuracy", "log_every": 1}}})
    trainer = SynthScopeModelTrainer.__new__(SynthScopeModelTrainer)
    trainer.config = cfg
    trainer.accelerator = Accelerator(cpu=True)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    data = torch.utils.data.TensorDataset(torch.randn(MICRO * ACCUM * 2, 4))     # 2 optimizer steps per epoch
    loader = torch.utils.data.DataLoader(data, batch_size=MICRO)
    trainer.model, trainer.optimizer, trainer.train_loader = trainer.accelerator.prepare(model, optimizer, loader)
    trainer.global_step, trainer.logger = 0, _Logger()
    trainer.train_step = lambda batch: (trainer.model(batch[0]) ** 2).mean()
    epochs_done = iter(range(EPOCHS))

    def val():
        epoch = next(epochs_done)
        return LOSSES[epoch], {"real_accuracy": ACCURACIES[epoch], "fake_accuracy": ACCURACIES[epoch]}

    trainer.val = val
    trainer.saved = []
    trainer._save_model = lambda tag, path: trainer.saved.append(str(tag))
    return trainer


def main() -> None:
    with tempfile.TemporaryDirectory() as save_path:
        trainer = build(save_path)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            trainer._train_with_timm_cosine(trainer.config.TrainerConfig.timm_cosine)
        log = out.getvalue()
        best = [i for i, tag in enumerate(trainer.saved) if tag == "best"]
        # per epoch the saves are: last, <epoch>, and best when the accuracy improved
        assert trainer.saved == ["last", "0", "best", "last", "1", "best", "last", "2", "last", "3", "last", "4", "best"], trainer.saved
        assert len(best) == 3
        with open(f"{save_path}/val_losses.json") as f:
            record = json.load(f)
        assert record["val_accuracy"] == ACCURACIES and record["best_epoch"] == 4 and record["val_loss"] == LOSSES, record
        with open(f"{save_path}/train_state/progress.json") as f:
            assert json.load(f)["val_accuracy"] == ACCURACIES
        assert "effective batch 256" in log, log
        assert "lr 0.0005 -> 1e-06 over 5 epochs" in log, log
        lrs = [float(line.split("lr ")[1].split()[0]) for line in log.splitlines() if " step " in line and "lr " in line]
        assert abs(lrs[0] - 5e-4) < 5e-7 and lrs == sorted(lrs, reverse=True) and lrs[-1] < 5e-5, lrs   # first step ~ base lr, then down
        print(f"best-model saves follow the accuracy; effective batch and lr logged ({lrs[0]:.3e} -> {lrs[-1]:.3e})")
    print("timm_cosine select-accuracy checks passed")


if __name__ == "__main__":
    main()
