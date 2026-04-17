import torch
from model import AGIDModel
from metrics import AIGIMetrics
from dataset import AIGIDatasetModule
from omegaconf import OmegaConf
from accelerate import Accelerator

class AGIDModelTrainer:
    def __init__(self, config: OmegaConf, dataset_module: AIGIDatasetModule):
        self.model = AGIDModel(config.backbone_name, config.bottleneck_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_loader = dataset_module.train_dataloader()
        self.val_loader = dataset_module.val_dataloader()
        self.test_loader = dataset_module.test_dataloader()
        self.config = config
        self.train_metrics = AIGIMetrics(config.metrics)
        self.val_metrics = AIGIMetrics(config.metrics)
        self.test_metrics = AIGIMetrics(config.metrics)
        self.use_metrics = len(config.metrics) > 0
        self.accelerator = Accelerator()
        self.model, self.optimizer, self.train_loader, self.val_loader, self.test_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader, self.test_loader)

    def train_step(self, batch):
        images, labels, folder_ids = batch
        outputs = self.model(images)      
        loss = self.criterion(outputs, labels)
        if self.use_metrics:
            outputs_g, labels_g, folder_ids_g = self.accelerator.gather_for_metrics((outputs, labels, folder_ids))
            self.train_metrics(outputs_g, labels_g, folder_ids_g)  
        return loss

    def val_step(self, batch):
        images, labels, folder_ids = batch
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        if self.use_metrics:
            outputs_g, labels_g, folder_ids_g = self.accelerator.gather_for_metrics((outputs.detach(), labels.detach(), folder_ids.detach()))
            self.val_metrics(outputs_g, labels_g, folder_ids_g)
        return loss
    
    def test_step(self, batch):
        images, labels, folder_ids = batch
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        if self.use_metrics:
            outputs_g, labels_g, folder_ids_g = self.accelerator.gather_for_metrics((outputs.detach(), labels.detach(), folder_ids.detach()))
            self.test_metrics(outputs_g, labels_g, folder_ids_g)
        return loss

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            loss  = self.train_step(batch)
            self.accelerator.backward(loss)
            self.optimizer.step()
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
            for batch in self.val_loader:
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
            for batch in self.test_loader:
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


    def save_model(self, epoch, save_path):
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if self.accelerator.is_main_process:
            self.accelerator.save(unwrapped_model.state_dict(), f"{save_path}/model_{epoch}.pth")

    def train(self):
        for epoch in range(self.config.epochs):
            train_loss, train_metrics_dict = self.train_epoch()
            self.accelerator.print(f"Epoch {epoch}, Train Loss: {train_loss}, Train Metrics: {train_metrics_dict}")
            if epoch % self.config.val_interval == 0:
                val_loss, val_metrics_dict = self.val_epoch()
                self.accelerator.print(f"Epoch {epoch}, Val Loss: {val_loss}, Val Metrics: {val_metrics_dict}")
            if epoch % self.config.test_interval == 0:
                test_loss, test_metrics_dict = self.test_epoch()
                self.accelerator.print(f"Epoch {epoch}, Test Loss: {test_loss}, Test Metrics: {test_metrics_dict}")
            if epoch % self.config.save_interval == 0:
                self.save_model(epoch, self.config.save_path)
                self.accelerator.print(f"Epoch {epoch}, Model Saved")