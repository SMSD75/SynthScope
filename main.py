import hydra
from accelerate.utils import set_seed
from omegaconf import DictConfig
from dataset import AIGIDatasetModule
from trainer import AGIDModelTrainer


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(getattr(cfg.TrainerConfig, "seed", 42))
    dataset_module = AIGIDatasetModule(cfg)
    trainer = AGIDModelTrainer(cfg, dataset_module)
    trainer.train()


if __name__ == "__main__":
    main()
