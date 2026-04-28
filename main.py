import torch
from accelerate.utils import set_seed
from model import AGIDDINOModel
from dataset import AIGIBenchDatasetModule
from omegaconf import OmegaConf
from trainer import AGIDModelTrainer

if __name__ == "__main__":
    config = OmegaConf.load("config.yaml")
    set_seed(getattr(config.TrainerConfig, "seed", 42))
    dataset_module = AIGIBenchDatasetModule(config)
    trainer = AGIDModelTrainer(config, dataset_module)
    trainer.train()