from __future__ import annotations

import os
from typing import Dict, Optional

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from .config_models import ConfigSchema
from .logging import get_logger


class ConfigLoader:
    """Load and validate project configuration."""

    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict] = None) -> None:
        if config_path:
            base_dir = os.path.dirname(os.path.abspath(config_path))
            config_name = os.path.splitext(os.path.basename(config_path))[0]
            with initialize_config_dir(version_base=None, config_dir=base_dir):
                cfg = compose(config_name=config_name)
            config_dict = OmegaConf.to_container(cfg, resolve=True)
            self.base_dir = base_dir
        elif config is not None:
            config_dict = config
            self.base_dir = os.getcwd()
        else:
            raise ValueError("Either config_path or config must be provided")

        self.config = ConfigSchema(**config_dict)
        self.logger = get_logger(self.__class__.__name__)
        self.logger.debug("Configuration loaded")

    def resolve_path(self, path: str) -> str:
        """Resolve path relative to the config base directory."""
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(self.base_dir, path))

    def get_config(self) -> ConfigSchema:
        """Return the validated configuration object."""
        return self.config
