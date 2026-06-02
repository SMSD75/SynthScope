import torch
from models.image.clip_detection.clip_detection import AGIDCLIPModel
from models.image.cnn_detection.cnn_detection import AGIDCNNModel
from models.image.dino_detection.dino_detection import AGIDDINOModel, AGIDDINOv3Model
from models.image.adapted_dino.adapted_dino import AGIDAdaptedDINOModel
from models.image.safe.safe import AGIDSAFEModel
from models.image.fat_former.fat_former import AGIDFatFormerModel
from models.image.co_spy.co_spy import AGIDCoSpyModel
from models.image.aide.aide import AGIDAIDEModel
from models.image.spai.spai import SPAI
from models.image.vib_net.vib_net import AGIDVIBNetModel
from models.image.npr.npr import NPR
from models.image.pgc.pgc import AGIDPGCModel
from models.video.d3.d3 import AGIDD3Model
from models.video.restrav.restrav import AGIDReStraVModel
from models.video.stall.stall import AGIDSTALLModel
from metrics import AIGIMetrics
from dataset import AIGIDatasetModule
from omegaconf import DictConfig
from accelerate import Accelerator
from transformers import get_scheduler
import os
from logger import build_logger
from tqdm import tqdm

class AGIDModelTrainer:
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
            return AGIDSAFEModel(config)
        if model_type == "fat_former":
            return AGIDFatFormerModel(config)
        if model_type == "co_spy":
            return AGIDCoSpyModel(config)
        if model_type == "aide":
            return AGIDAIDEModel(config)
        if model_type == "spai":
            return SPAI(config)
        if model_type == "vib_net":
            return AGIDVIBNetModel(config)
        if model_type == "npr":
            return NPR(config)
        if model_type == "pgc":
            return AGIDPGCModel(config)
        if model_type == "d3":
            return AGIDD3Model(config)
        if model_type == "restrav":
            return AGIDReStraVModel(config)
        if model_type == "stall":
            return AGIDSTALLModel(config)
        if model_type == "cnn":
            return AGIDCNNModel(config)
        if model_type == "adapted_dino":
            return AGIDAdaptedDINOModel(config)
        if model_type == "dinov2":
            return AGIDDINOModel(config)
        if model_type == "dinov3":
            return AGIDDINOv3Model(config)
        # Explicit model_type wins above; the CLIP-prefixed backbone is only a last-resort
        # fallback, so e.g. model_type=cnn with the default CLIP:ViT-L/14 backbone is not hijacked.
        if model_type == "clip" or config.ModelConfig.backbone_name.startswith("CLIP:"):
            return AGIDCLIPModel(config)
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

    def train(self):
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
