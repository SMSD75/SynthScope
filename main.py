import torch
from model import AGIDModel
from dataset import AIGIDatasetModule
from omegaconf import OmegaConf
from trainer import AGIDModelTrainer

if __name__ == "__main__":
    config = OmegaConf.load("config.yaml")
    dataset_module = AIGIDatasetModule(config)
    trainer = AGIDModelTrainer(config, dataset_module)
    trainer.train()