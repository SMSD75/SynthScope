"""Load the Hydra-composed app config from non-Hydra entrypoints (scripts, tests)."""
from pathlib import Path

from omegaconf import DictConfig

_CONF_DIR = Path(__file__).resolve().parent / "conf"


def compose_config(overrides: list[str] | None = None) -> DictConfig:
    """Compose ``conf/config.yaml`` with optional Hydra overrides."""
    from hydra import compose, initialize_config_dir

    overrides = overrides or []
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        return compose(config_name="config", overrides=overrides)
