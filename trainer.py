import torch
from torch.nn.parallel import DistributedDataParallel
from models.image.clip_detection.clip_detection import SynthScopeCLIPModel
from models.image.cnn_detection.cnn_detection import SynthScopeCNNModel
from models.image.dino_detection.dino_detection import SynthScopeDINOModel, SynthScopeDINOv3Model
from models.image.adapted_dino.adapted_dino import SynthScopeAdaptedDINOModel
from models.image.safe.safe import SynthScopeSAFEModel
from models.image.fat_former.fat_former import SynthScopeFatFormerModel
from models.image.co_spy.co_spy import SynthScopeCoSpyModel
from models.image.aide.aide import SynthScopeAIDEModel
from models.image.spai.spai import SPAI
from models.image.vib_net.vib_net import SynthScopeVIBNetModel
from models.image.npr.npr import NPR
from models.image.pgc.pgc import SynthScopePGCModel
from models.image.cure.cure import SynthScopeCuReModel
from models.image.dda.dda import SynthScopeDDAModel
from models.image.omniaid.omniaid import SynthScopeOmniAIDModel
from models.image.effort.effort import SynthScopeEffortModel
from models.image.moejepa.moejepa import SynthScopeMoEJEPAModel
from models.image.bfree.bfree import SynthScopeBFreeModel
from models.image.d3qe.d3qe import SynthScopeD3QEModel
from models.video.d3.d3 import SynthScopeD3Model
from models.video.split.split import SynthScopeSPLITModel
from models.video.restrav.restrav import SynthScopeReStraVModel
from models.video.stall.stall import SynthScopeSTALLModel
from metrics import AIGIMetrics
from dataset import AIGIDatasetModule
from omegaconf import DictConfig
from accelerate import Accelerator
from transformers import get_scheduler
from timm.scheduler.cosine_lr import CosineLRScheduler
import json
import os
from logger import build_logger
from tqdm import tqdm

class SynthScopeModelTrainer:
    def __init__(self, config: DictConfig, dataset_module: AIGIDatasetModule):
        self.model = self._build_model(config)
        # TrainerConfig.sync_batchnorm: batch statistics over all processes' micro-batches (a recipe whose official
        # per-GPU batch does not fit one GPU keeps the BatchNorm batch it was tuned with). Only in a multi-process run.
        if config.TrainerConfig.get("sync_batchnorm", False) and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.optimizer = torch.optim.AdamW(
            self._param_groups(config),
            lr=config.TrainerConfig.lr,
            betas=(0.9, 0.999),
            weight_decay=config.TrainerConfig.get("weight_decay", 0.01),
        )
        self.criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=config.TrainerConfig.get("label_smoothing", 0.0)
        )
        self.train_loader = dataset_module.train_dataloader()
        self.val_loader = dataset_module.val_dataloader()
        self.test_loader = dataset_module.test_dataloader()
        self.config = config
        self.train_metrics = AIGIMetrics(config.metrics)
        self.val_metrics = AIGIMetrics(config.metrics)
        self.test_metrics = AIGIMetrics(config.metrics)
        self.accelerator = Accelerator()
        self.use_metrics = len(config.metrics) > 0
        self.scheduler = None
        warmup = config.TrainerConfig.warmup
        # A bare `mode: off` in YAML is the boolean False (and `on` is True), so compare the string form: before this, an
        # unquoted `off` (conf/trainer/default.yaml, aide, restrav, ...) silently enabled a schedule annealing the LR to 0.
        if str(warmup.mode).lower() not in ("off", "false", "none"):
            num_training_steps = config.TrainerConfig.epochs * (len(self.train_loader) // self.accelerator.gradient_accumulation_steps)
            self.scheduler = get_scheduler(warmup.lr_scheduler, optimizer=self.optimizer, num_warmup_steps=int(warmup.warmup_ratio * num_training_steps), num_training_steps=num_training_steps)
        self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader, self.test_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader, self.test_loader)
        self.train_metrics = self.train_metrics.to(self.accelerator.device)
        self.val_metrics = self.val_metrics.to(self.accelerator.device)
        self.test_metrics = self.test_metrics.to(self.accelerator.device)
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_test_loss = float("inf")
        self.logger = build_logger(config, self.accelerator)
        if self.accelerator.is_main_process:    
            self.print_trainable_parameters()
    
    def print_trainable_parameters(self):
        trainable_parameters = [n for (n, p) in self.model.named_parameters() if p.requires_grad]
        count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.accelerator.print(f"Trainable Parameters ({len(trainable_parameters)} tensors, {count:,} values): {trainable_parameters}")

    def _build_model(self, config: DictConfig):
        model_type = config.ModelConfig.get("model_type", "dino").lower()
        if model_type == "safe":
            return SynthScopeSAFEModel(config)
        if model_type == "fat_former":
            return SynthScopeFatFormerModel(config)
        if model_type == "co_spy":
            return SynthScopeCoSpyModel(config)
        if model_type == "aide":
            return SynthScopeAIDEModel(config)
        if model_type == "spai":
            return SPAI(config)
        if model_type == "vib_net":
            return SynthScopeVIBNetModel(config)
        if model_type == "npr":
            return NPR(config)
        if model_type == "pgc":
            return SynthScopePGCModel(config)
        if model_type == "cure":
            return SynthScopeCuReModel(config)
        if model_type == "dda":
            return SynthScopeDDAModel(config)
        if model_type == "omniaid":
            return SynthScopeOmniAIDModel(config)
        if model_type == "effort":
            return SynthScopeEffortModel(config)
        if model_type == "moejepa":
            return SynthScopeMoEJEPAModel(config)
        if model_type == "bfree":
            return SynthScopeBFreeModel(config)
        if model_type == "d3qe":
            return SynthScopeD3QEModel(config)
        if model_type == "d3":
            return SynthScopeD3Model(config)
        if model_type == "split":
            return SynthScopeSPLITModel(config)
        if model_type == "restrav":
            return SynthScopeReStraVModel(config)
        if model_type == "stall":
            return SynthScopeSTALLModel(config)
        if model_type == "cnn":
            return SynthScopeCNNModel(config)
        if model_type == "adapted_dino":
            return SynthScopeAdaptedDINOModel(config)
        if model_type == "dinov2":
            return SynthScopeDINOModel(config)
        if model_type == "dinov3":
            return SynthScopeDINOv3Model(config)
        # Explicit model_type wins above; the CLIP-prefixed backbone is only a last-resort
        # fallback, so e.g. model_type=cnn with the default CLIP:ViT-L/14 backbone is not hijacked.
        if model_type == "clip" or config.ModelConfig.backbone_name.startswith("CLIP:"):
            return SynthScopeCLIPModel(config)
        raise ValueError(f"Invalid model_type: {model_type}")
    
    def _filter_trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _param_groups(self, config):
        """The trainable parameters; with ``TrainerConfig.no_weight_decay_1d`` the biases and 1-D parameters (norms, ...)
        get no weight decay, as in the BEiT/MFM-style fine-tuning optimizers (SPAI's ``get_finetune_param_groups``)."""
        if not config.TrainerConfig.get("no_weight_decay_1d", False):
            return self._filter_trainable_parameters()
        named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        no_decay = lambda n, p: p.ndim <= 1 or n.endswith(".bias")
        return [{"params": [p for n, p in named if not no_decay(n, p)]},
                {"params": [p for n, p in named if no_decay(n, p)], "weight_decay": 0.0}]

    def _compute_loss(self, batch):
        """``model.compute_loss(batch)``. Under multi-GPU DDP the prepared model is a ``DistributedDataParallel``, which has
        no ``compute_loss`` and whose gradient reducer is armed only by its own ``forward``: the loss must be computed
        through the wrapper, so for the duration of the call the inner module's ``forward`` is a dispatcher that runs
        ``compute_loss`` on the wrapper's call (DDP runs ``module(*inputs)``) and the real ``forward`` for the calls
        ``compute_loss`` itself makes. Single-process and DeepSpeed runs (the engine forwards unknown attributes to the module)
        keep the plain call."""
        if not isinstance(self.model, DistributedDataParallel):
            return self.model.compute_loss(batch)
        inner = self.model.module
        previous = inner.__dict__.get("forward")
        original_forward, armed = inner.forward, [True]

        def dispatch(*args, **kwargs):
            if armed[0]:                       # DDP's call: run the loss ...
                armed[0] = False
                return inner.compute_loss(*args, **kwargs)
            return original_forward(*args, **kwargs)   # ... and compute_loss's own self(...) / self.forward(...) calls

        inner.forward = dispatch
        try:
            return self.model(batch)
        finally:
            if previous is None:
                del inner.forward
            else:
                inner.forward = previous

    def train_step(self, batch):
        loss_dict = self._compute_loss(batch)
        loss = loss_dict["loss"]
        outputs = loss_dict["outputs"]
        labels = batch[1]
        folder_ids = batch[2]
        if self.use_metrics:    
            outputs_g, labels_g, folder_ids_g = self.accelerator.gather_for_metrics((outputs.detach(), labels.detach(), folder_ids.detach()))
            self.train_metrics(outputs_g, labels_g, folder_ids_g)  
        return loss

    def val_step(self, batch):
        loss_dict = self._compute_loss(batch)
        loss = loss_dict["loss"]
        outputs = loss_dict["outputs"]
        labels = batch[1]
        folder_ids = batch[2]
        if self.use_metrics:
            outputs_g, labels_g, folder_ids_g = self.accelerator.gather_for_metrics((outputs.detach(), labels.detach(), folder_ids.detach()))
            self.val_metrics(outputs_g, labels_g, folder_ids_g)
        return loss
    
    def test_step(self, batch):
        loss_dict = self._compute_loss(batch)
        loss = loss_dict["loss"]
        outputs = loss_dict["outputs"]
        labels = batch[1]
        folder_ids = batch[2]
        if self.use_metrics:
            outputs_g, labels_g, folder_ids_g = self.accelerator.gather_for_metrics((outputs.detach(), labels.detach(), folder_ids.detach()))
            self.test_metrics(outputs_g, labels_g, folder_ids_g)
        return loss

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in tqdm(self.train_loader, desc="Training"):
            self.optimizer.zero_grad()
            loss  = self.train_step(batch)
            self.accelerator.backward(loss)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            if self.accelerator.sync_gradients:
                self.global_step += 1
            loss_sum = self.accelerator.gather(loss.detach().unsqueeze(0)).mean()
            total_loss += loss_sum.item()
        if self.use_metrics:
            total_metrics_dict = {
                metric: self.train_metrics.metrics_dict[metric].compute().item()
                for metric in self.train_metrics.metrics_dict
            }
            self.train_metrics._reset_metrics_dict()
        else:
            total_metrics_dict = {}
        return total_loss / len(self.train_loader), total_metrics_dict
    
    def val_epoch(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                loss = self.val_step(batch)
                loss_sum = self.accelerator.gather(loss.detach().unsqueeze(0)).mean()
                total_loss += loss_sum.item()
        if self.use_metrics:
            total_metrics_dict = {
                metric: self.val_metrics.metrics_dict[metric].compute().item()
                for metric in self.val_metrics.metrics_dict
            }
            self.val_metrics._reset_metrics_dict()
        else:
            total_metrics_dict = {}
        return total_loss / len(self.val_loader), total_metrics_dict
    
    def test_epoch(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                loss = self.test_step(batch)
                loss_sum = self.accelerator.gather(loss.detach().unsqueeze(0)).mean()
                total_loss += loss_sum.item()
           
        if self.use_metrics:
            total_metrics_dict = {
                metric: self.test_metrics.metrics_dict[metric].compute().item()
                for metric in self.test_metrics.metrics_dict
            }
            self.test_metrics._reset_metrics_dict()
        else:
            total_metrics_dict = {}
        return total_loss / len(self.test_loader), total_metrics_dict


    def _save_model(self, epoch, save_path):
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if self.accelerator.is_main_process:
            os.makedirs(save_path, exist_ok=True)
            self.accelerator.save(unwrapped_model.state_dict(), f"{save_path}/model_{epoch}.pth")

    def _save(self, val_loss, epoch):
        if self.config.TrainerConfig.save_policy == "max":
            if val_loss > self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_model(epoch, self.config.TrainerConfig.save_path)
        elif self.config.TrainerConfig.save_policy == "min":
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_model(epoch, self.config.TrainerConfig.save_path)
        self.accelerator.print(f"Epoch {epoch}, Model Saved")

    @staticmethod
    def _val_accuracy(metrics_dict):
        """Mean of the real and fake accuracies over the validation folders (balanced accuracy; on a balanced
        validation set it equals the plain accuracy the CNNDetection-family training scripts monitor)."""
        real = [v for k, v in metrics_dict.items() if k.startswith("real_accuracy")]
        fake = [v for k, v in metrics_dict.items() if k.startswith("fake_accuracy")]
        if not real or not fake:
            raise ValueError("validation accuracy (the plateau schedule, timm_cosine select: accuracy) needs metrics=[real_accuracy,fake_accuracy]")
        return (sum(real) / len(real) + sum(fake) / len(fake)) / 2

    def _train_with_plateau(self, plateau):
        """The CNNDetection-family schedule (CNNDetection, D3QE, ...; their EarlyStopping + adjust_learning_rate):
        validate after every epoch and keep the checkpoint with the best validation accuracy (``model_best.pth``);
        when it has not improved by at least ``delta`` for ``patience`` epochs, divide the learning rate by
        ``factor`` and restart the tracker with ``delta_after_drop`` (so, as in the official code, the next epoch is
        kept); stop once the learning rate falls below ``min_lr`` or after ``epochs`` epochs."""
        cfg = self.config.TrainerConfig
        best, bad, delta = None, 0, float(plateau.delta)
        for epoch in range(cfg.epochs):
            train_loss, train_metrics_dict = self.train_epoch()
            self.accelerator.print(f"Epoch {epoch}, Train Loss: {train_loss}, Train Metrics: {train_metrics_dict}")
            self.logger.log_metrics({"train/loss": train_loss, **{f"train/{m}": v for m, v in train_metrics_dict.items()}}, self.global_step)
            val_loss, val_metrics_dict = self.val()
            acc = self._val_accuracy(val_metrics_dict)
            lr = self.optimizer.param_groups[0]["lr"]
            self.accelerator.print(f"Epoch {epoch}, Val accuracy {acc:.4f} (lr {lr:.2e}, step {self.global_step})")
            self.logger.log_metrics({"val/accuracy": acc, "lr": lr}, self.global_step)
            if cfg.save:
                self._save_model("last", cfg.save_path)
            if best is None or acc >= best + delta:
                best, bad = acc, 0
                if cfg.save:
                    self._save_model("best", cfg.save_path)
                    self.accelerator.print(f"Epoch {epoch}: best validation accuracy {acc:.4f}, saved model_best.pth")
                continue
            bad += 1
            if bad < int(plateau.patience):
                continue
            for group in self.optimizer.param_groups:
                group["lr"] /= float(plateau.factor)
            new_lr = self.optimizer.param_groups[0]["lr"]
            if new_lr < float(plateau.min_lr):
                self.accelerator.print(f"Epoch {epoch}: learning rate {new_lr:.2e} below min_lr, early stopping")
                break
            self.accelerator.print(f"Epoch {epoch}: no improvement for {bad} epochs, learning rate -> {new_lr:.2e}")
            best, bad, delta = None, 0, float(plateau.get("delta_after_drop", plateau.delta))
        self.logger.finish()

    def _train_with_lr_step(self, step):
        """A fixed step schedule as in NPR's train.py (``delr_freq``): after epoch ``e`` (0-based) with
        ``e % every == 0 and e != 0`` the learning rate is multiplied by ``factor``; validation after every epoch is
        logged (plus ``model_best.pth`` by validation accuracy), and the final model is ``model_last.pth``, which is
        the checkpoint the official code releases."""
        cfg = self.config.TrainerConfig
        best = None
        for epoch in range(cfg.epochs):
            train_loss, train_metrics_dict = self.train_epoch()
            self.accelerator.print(f"Epoch {epoch}, Train Loss: {train_loss}, Train Metrics: {train_metrics_dict}")
            self.logger.log_metrics({"train/loss": train_loss, **{f"train/{m}": v for m, v in train_metrics_dict.items()}}, self.global_step)
            if epoch % int(step.every) == 0 and epoch != 0:
                for group in self.optimizer.param_groups:
                    group["lr"] *= float(step.factor)
                self.accelerator.print(f"Epoch {epoch}: learning rate -> {self.optimizer.param_groups[0]['lr']:.3e}")
            val_loss, val_metrics_dict = self.val()
            acc = self._val_accuracy(val_metrics_dict)
            self.accelerator.print(f"Epoch {epoch}, Val accuracy {acc:.4f} (lr {self.optimizer.param_groups[0]['lr']:.3e}, step {self.global_step})")
            self.logger.log_metrics({"val/accuracy": acc, "lr": self.optimizer.param_groups[0]["lr"]}, self.global_step)
            if cfg.save:
                self._save_model("last", cfg.save_path)
                if best is None or acc > best:
                    best = acc
                    self._save_model("best", cfg.save_path)
        self.logger.finish()

    def _train_with_step_budget(self, budget):
        """A fixed budget of micro-batches, as in PGC's train.py: the loss of each micro-batch is divided by
        ``grad_accum_steps`` and the optimizer steps every ``grad_accum_steps`` micro-batches, after clipping the gradient
        norm to ``max_grad_norm`` (0: no clipping); ``model_step<N>.pth`` is saved every ``save_every`` micro-batches, up
        to ``max_steps``. For detectors whose transfer peaks early (PGC's released checkpoint is ~400 steps in), the
        saved curve is evaluated rather than only the last model."""
        accum = int(budget.get("grad_accum_steps", 1))
        clip = float(budget.get("max_grad_norm", 0.0))
        # max_steps null: `epochs` full passes over the training data
        max_steps = int(budget.get("max_steps") or len(self.train_loader) * int(self.config.TrainerConfig.epochs))
        save_every = int(budget.save_every)
        save_path = self.config.TrainerConfig.save_path
        if budget.get("cosine_t_max"):
            # torch CosineAnnealingLR stepped once per optimizer step (e.g. DDA: T_max=1000, eta_min=1e-3 x lr; it cycles)
            opt = getattr(self.optimizer, "optimizer", self.optimizer)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=int(budget.cosine_t_max),
                eta_min=float(self.config.TrainerConfig.lr) * float(budget.get("eta_min_ratio", 0.0)))
        step, running = 0, 0.0
        self.model.train()
        self.optimizer.zero_grad()
        while step < max_steps:
            for batch in tqdm(self.train_loader, desc="Training"):
                loss = self.train_step(batch)
                self.accelerator.backward(loss / accum)
                running += loss.detach().item()
                step += 1
                if step % accum == 0:
                    if clip > 0:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), clip)
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                if step % save_every == 0:
                    self.accelerator.print(f"step {step} (optimizer step {self.global_step}): mean loss {running / save_every:.4f}, "
                                           f"lr {self.optimizer.param_groups[0]['lr']:.3e}")
                    self.logger.log_metrics({"train/loss": running / save_every}, step)
                    running = 0.0
                    self._save_model(f"step{step}", save_path)
                if step >= max_steps:
                    break
        if step % accum != 0:                     # a partial accumulation at the end (the official finalize_epoch)
            if clip > 0:
                self.accelerator.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
        self._save_model("final", save_path)
        self.logger.finish()

    def _train_with_timm_cosine(self, sched):
        """SPAI's schedule (its lr_scheduler.py + train_one_epoch): timm's CosineLRScheduler stepped after every
        optimizer step, a linear warmup from ``warmup_lr`` over ``warmup_epochs`` and a cosine to ``min_lr`` at the last
        step. ``grad_accum_steps`` splits each official batch into micro-batches (loss divided accordingly) and the
        schedule counts official steps, so a smaller GPU follows the same LR curve. Validation after every epoch; as in
        the official code, a checkpoint is kept whenever the validation loss improves (``model_<epoch>.pth``, the
        selected one also as ``model_best.pth``; ``save_all`` keeps every epoch, as the official ``--save-all``), plus
        ``model_last.pth``; the losses go to ``val_losses.json``. With ``select: accuracy`` the selected epoch is the
        first with the highest validation accuracy instead (AIDE's ``max_accuracy``; needs ``metrics=[real_accuracy,
        fake_accuracy]``, a balanced validation set makes it the plain accuracy), and the accuracies are logged to
        ``val_losses.json`` as ``val_accuracy``. After
        every epoch the full training state (model, optimizer, loss scaler, RNGs) is saved to ``<save_path>/train_state``,
        and a rerun with the same ``save_path`` resumes after the last finished epoch (runs longer than a job limit)."""
        cfg = self.config.TrainerConfig
        accum = int(sched.get("grad_accum_steps", 1))
        n_iter = len(self.train_loader) // accum
        scheduler = CosineLRScheduler(getattr(self.optimizer, "optimizer", self.optimizer), t_initial=int(cfg.epochs * n_iter),
                                      lr_min=float(sched.min_lr), warmup_lr_init=float(sched.warmup_lr),
                                      warmup_t=int(float(sched.warmup_epochs) * n_iter), cycle_limit=1, t_in_epochs=False)
        save_all = bool(sched.get("save_all", False))
        log_every = int(sched.get("log_every", 100))      # optimizer steps between loss / lr lines
        micro = self.train_loader.batch_sampler.batch_size     # DataLoaderShard.batch_size is None after prepare()
        self.accelerator.print(f"timm_cosine: {len(self.train_loader)} micro-batches per process and epoch -> {n_iter} optimizer steps per epoch "
                               f"({self.accelerator.num_processes} processes x micro-batch {micro} x accumulation {accum} "
                               f"= effective batch {self.accelerator.num_processes * micro * accum}), "
                               f"lr {cfg.lr} -> {sched.min_lr} over {cfg.epochs} epochs")
        history, start = [], 0          # validation loss per epoch; None where it was not measured on this validation set
        by_accuracy = str(sched.get("select", "loss")) == "accuracy"
        accuracies = []                 # validation accuracy per epoch (select: accuracy)
        state_dir, log_path = f"{cfg.save_path}/train_state", f"{cfg.save_path}/val_losses.json"
        if cfg.save and os.path.isfile(f"{state_dir}/progress.json"):
            with open(f"{state_dir}/progress.json") as f:
                progress = json.load(f)
            self.accelerator.load_state(state_dir)
            history, start = progress["val_loss"], progress["epoch"] + 1
            accuracies = progress.get("val_accuracy", [])
            self.global_step = start * n_iter
            self.accelerator.print(f"Resuming after epoch {progress['epoch']}")
        measured = lambda: [v for v in history if v is not None]
        best = min(measured(), default=float("inf"))
        best_accuracy = max(accuracies, default=float("-inf"))
        for epoch in range(start, cfg.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            total, k = 0.0, 0
            for idx, batch in enumerate(tqdm(self.train_loader, desc=f"Training epoch {epoch}")):
                loss = self.train_step(batch)
                self.accelerator.backward(loss / accum)
                total += loss.detach().item()
                if (idx + 1) % accum == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    scheduler.step_update(epoch * n_iter + k)
                    k += 1
                    self.global_step += 1
                    if k == 1 or k % log_every == 0:
                        self.accelerator.print(f"epoch {epoch} step {k}/{n_iter}: loss {total / (idx + 1):.4f}, "
                                               f"lr {self.optimizer.param_groups[0]['lr']:.3e}")
                    if k == n_iter:
                        break
            self.logger.log_metrics({"train/loss": total / max(1, k * accum), "lr": self.optimizer.param_groups[0]["lr"]}, self.global_step)
            val_loss, val_metrics_dict = self.val()
            history.append(val_loss)
            best_epoch = history.index(min(measured()))
            self.accelerator.print(f"Epoch {epoch}, Val Loss: {val_loss:.4f} (best {history[best_epoch]:.4f} at epoch {best_epoch}), "
                                   f"Val Metrics: {val_metrics_dict}")
            if by_accuracy:
                accuracies.append(self._val_accuracy(val_metrics_dict))
                improved = accuracies[-1] > best_accuracy        # strictly greater: the earliest of equal epochs, as AIDE
                best_accuracy = max(accuracies)
                best_epoch = accuracies.index(best_accuracy)
                self.accelerator.print(f"Epoch {epoch}, Val Accuracy: {accuracies[-1]:.4f} (best {best_accuracy:.4f} at epoch {best_epoch})")
            else:
                improved = val_loss < best
            if cfg.save:
                self._save_model("last", cfg.save_path)
                if save_all or improved:   # the official --save-all keeps every epoch
                    self._save_model(epoch, cfg.save_path)
                if improved:
                    best = min(best, val_loss)
                    self._save_model("best", cfg.save_path)
                if self.accelerator.is_main_process:
                    with open(log_path, "w") as f:
                        json.dump({"val_loss": history, **({"val_accuracy": accuracies} if by_accuracy else {}),
                                   "best_epoch": best_epoch}, f, indent=1)
                self.accelerator.save_state(state_dir)
                if self.accelerator.is_main_process:
                    with open(f"{state_dir}/progress.json", "w") as f:
                        json.dump({"epoch": epoch, "val_loss": history, "val_accuracy": accuracies}, f)
        self.logger.finish()

    def train(self):
        timm_cosine = self.config.TrainerConfig.get("timm_cosine", None)
        if timm_cosine is not None:
            return self._train_with_timm_cosine(timm_cosine)
        plateau = self.config.TrainerConfig.get("plateau", None)
        if plateau is not None:
            return self._train_with_plateau(plateau)
        lr_step = self.config.TrainerConfig.get("lr_step", None)
        if lr_step is not None:
            return self._train_with_lr_step(lr_step)
        budget = self.config.TrainerConfig.get("step_budget", None)
        if budget is not None:
            return self._train_with_step_budget(budget)
        for epoch in range(self.config.TrainerConfig.epochs):
            if epoch % self.config.TrainerConfig.val_interval == 0:
                val_loss, val_metrics_dict = self.val()
                self.accelerator.print(f"Epoch {epoch}, Val Loss: {val_loss}, Val Metrics: {val_metrics_dict}")
                self.logger.log_metrics({"val/loss": val_loss, **{f"val/{metric}": value for metric, value in val_metrics_dict.items()}}, self.global_step)
            if epoch % self.config.TrainerConfig.test_interval == 0:
                test_loss, test_metrics_dict = self.test()   
                self.accelerator.print(f"Epoch {epoch}, Test Loss: {test_loss}, Test Metrics: {test_metrics_dict}")
                self.logger.log_metrics({"test/loss": test_loss, **{f"test/{metric}": value for metric, value in test_metrics_dict.items()}}, self.global_step)
            train_loss, train_metrics_dict = self.train_epoch()
            self.accelerator.print(f"Epoch {epoch}, Train Loss: {train_loss}, Train Metrics: {train_metrics_dict}")
            self.logger.log_metrics({"train/loss": train_loss, **{f"train/{metric}": value for metric, value in train_metrics_dict.items()}}, self.global_step)
            self.accelerator.print(f"Epoch{epoch}, Training Loss: {train_loss}, Training Metrics: {train_metrics_dict}")
            if epoch % self.config.TrainerConfig.save_interval == 0:
                if self.config.TrainerConfig.save:
                    self._save(val_loss, epoch)
        self.logger.finish()
    
    def test(self):
        test_loss, test_metrics_dict = self.test_epoch()
        self.accelerator.print(f"Test Loss: {test_loss}, Test Metrics: {test_metrics_dict}")
        self.logger.log_metrics({"test/loss": test_loss, **{f"test/{metric}": value for metric, value in test_metrics_dict.items()}}, self.global_step)
        return test_loss, test_metrics_dict
    
    def val(self):
        val_loss, val_metrics_dict = self.val_epoch()
        self.accelerator.print(f"Val Loss: {val_loss}, Val Metrics: {val_metrics_dict}")
        self.logger.log_metrics({"val/loss": val_loss, **{f"val/{metric}": value for metric, value in val_metrics_dict.items()}}, self.global_step)
        return val_loss, val_metrics_dict
