from __future__ import annotations

import os
from typing import Dict, Optional

import yaml

from .config_models import ConfigSchema
from .logging import get_logger


class ConfigLoader:
    """Load and validate project configuration."""

    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict] = None) -> None:
        if config_path:
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
            self.base_dir = os.path.dirname(os.path.abspath(config_path))
        elif config:
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
