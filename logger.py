from abc import ABC, abstractmethod
from typing import Any
from PIL import Image
from omegaconf import DictConfig, OmegaConf
import datetime

class AIGILogger(ABC):
    def __init__(self, config: DictConfig):
        self.config = config

    def log_hyperparameters(self, params: dict[str, Any]) -> None:
        pass
    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass
    @abstractmethod
    def log_images(self, images: list[Image.Image], step: int | None = None, tag: str = "images"):
        pass
    def finish(self) -> None:
        pass
    def __enter__(self):  return self
    def __exit__(self, *exc): self.finish()


class NullLogger(AIGILogger):
    def log_metrics(self, metrics, step=None): pass
    def log_images(self, images, step=None, tag="images"): pass


class AIGILoggerWandb(AIGILogger):
    def __init__(self, config, is_main_process: bool = True):
        super().__init__(config)
        self.enabled = is_main_process
        if not self.enabled:
            return
        import wandb
        self._wandb = wandb
        self.run = wandb.init(
            project=config.logger.project,
            name=config.logger.name + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            tags=list(config.logger.tags),
            config=OmegaConf.to_container(config, resolve=True),
        )

    def log_metrics(self, metrics, step=None):
        if self.enabled:
            self.run.log(metrics, step=step)

    def log_images(self, images, step=None, tag="images"):
        if self.enabled:
            self.run.log({tag: [self._wandb.Image(i) for i in images]}, step=step)

    def finish(self):
        if self.enabled:
            self.run.finish()


def build_logger(config, accelerator=None) -> AIGILogger:
    is_main = accelerator.is_main_process if accelerator is not None else True
    kind = config.logger.type
    if kind == "wandb":   return AIGILoggerWandb(config, is_main_process=is_main)
    if kind in (None, "none", "null"):  return NullLogger(config)
    raise ValueError(f"Unknown logger type: {kind}")