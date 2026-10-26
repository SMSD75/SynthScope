import torch
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
from models.image.d3qe.d3qe import SynthScopeD3QEModel
from models.video.d3.d3 import SynthScopeD3Model
from models.video.restrav.restrav import SynthScopeReStraVModel
from models.video.stall.stall import SynthScopeSTALLModel
from metrics import AIGIMetrics
from dataset import AIGIDatasetModule
from omegaconf import DictConfig
from accelerate import Accelerator
from transformers import get_scheduler
import os
from logger import build_logger
from tqdm import tqdm

class SynthScopeModelTrainer:
    def __init__(self, config: DictConfig, dataset_module: AIGIDatasetModule):
        self.model = self._build_model(config)
        self.optimizer = torch.optim.AdamW(
            self._filter_trainable_parameters(),
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
        if warmup.mode != "off":
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
        self.accelerator.print(f"Trainable Parameters: {trainable_parameters}")

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
        if model_type == "d3qe":
            return SynthScopeD3QEModel(config)
        if model_type == "d3":
            return SynthScopeD3Model(config)
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

    def train_step(self, batch):
        loss_dict = self.model.compute_loss(batch)
        loss = loss_dict["loss"]
        outputs = loss_dict["outputs"]
        labels = batch[1]
        folder_ids = batch[2]
        if self.use_metrics:    
            outputs_g, labels_g, folder_ids_g = self.accelerator.gather_for_metrics((outputs.detach(), labels.detach(), folder_ids.detach()))
            self.train_metrics(outputs_g, labels_g, folder_ids_g)  
        return loss

    def val_step(self, batch):
        loss_dict = self.model.compute_loss(batch)
        loss = loss_dict["loss"]
        outputs = loss_dict["outputs"]
        labels = batch[1]
        folder_ids = batch[2]
        if self.use_metrics:
            outputs_g, labels_g, folder_ids_g = self.accelerator.gather_for_metrics((outputs.detach(), labels.detach(), folder_ids.detach()))
            self.val_metrics(outputs_g, labels_g, folder_ids_g)
        return loss
    
    def test_step(self, batch):
        loss_dict = self.model.compute_loss(batch)
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
            raise ValueError("the plateau schedule needs metrics=[real_accuracy,fake_accuracy]")
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

    def train(self):
        plateau = self.config.TrainerConfig.get("plateau", None)
        if plateau is not None:
            return self._train_with_plateau(plateau)
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
