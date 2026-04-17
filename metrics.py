from sys import api_version
import torch
from torchmetrics import Accuracy, AveragePrecision, AUROC

class AIGIMetrics(torch.nn.Module):
    def __init__(self, metrics: list[str]):
        super().__init__()
        self.metrics = metrics
        self.metrics_dict = torch.nn.ModuleDict()
        self.factory_dict = {
            "real_accuracy": lambda: Accuracy(task="binary"),
            "fake_accuracy": lambda: Accuracy(task="binary"),
            "AP": lambda: AveragePrecision(task="binary"),
            "AUC": lambda: AUROC(task="binary"),
        }

    def _get_metrics_dict(self):
        return self.metrics_dict

    def _reset_metrics_dict(self):
        for m in self.metrics_dict.values():
            m.reset()

    def __call__(self, outputs, labels, folder_ids):
        for metric in self.metrics:
            model_ids = set(folder_ids.tolist())
            if metric == "real_accuracy":
                for model_id in model_ids:
                    real_outputs = outputs[(folder_ids == model_id) & (labels == 0)]
                    real_labels = labels[(folder_ids == model_id) & (labels == 0)]
                    if f"{metric}_{model_id}" not in self.metrics_dict:
                        self.metrics_dict[f"{metric}_{model_id}"] = self.factory_dict[metric]()
                    self.metrics_dict[f"{metric}_{model_id}"].update(real_outputs, real_labels)
            elif metric == "fake_accuracy":
                for model_id in model_ids:
                    fake_outputs = outputs[(folder_ids == model_id) & (labels == 1)]
                    fake_labels = labels[(folder_ids == model_id) & (labels == 1)]
                    if f"{metric}_{model_id}" not in self.metrics_dict:
                        self.metrics_dict[f"{metric}_{model_id}"] = self.factory_dict[metric]()
                    self.metrics_dict[f"{metric}_{model_id}"].update(fake_outputs, fake_labels)
            elif metric == "AP":
                for model_id in model_ids:
                    ap_outputs = outputs[(folder_ids == model_id)]
                    ap_labels = labels[(folder_ids == model_id)]
                    if f"{metric}_{model_id}" not in self.metrics_dict:
                        self.metrics_dict[f"{metric}_{model_id}"] = self.factory_dict[metric]()
                    self.metrics_dict[f"{metric}_{model_id}"].update(ap_outputs, ap_labels)
            elif metric == "AUC":
                for model_id in model_ids:
                    roc_auc_outputs = outputs[(folder_ids == model_id)]
                    roc_auc_labels = labels[(folder_ids == model_id)]
                    if f"{metric}_{model_id}" not in self.metrics_dict:
                        self.metrics_dict[f"{metric}_{model_id}"] = self.factory_dict[metric]()  
                    self.metrics_dict[f"{metric}_{model_id}"].update(roc_auc_outputs, roc_auc_labels)
            else:
                raise ValueError(f"Invalid metric: {metric}")