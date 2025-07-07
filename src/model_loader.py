"""Utility for loading pretrained models when training is disabled."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib

from .config_loader import ConfigLoader
from .logging import get_logger


class ModelLoader:
    """Load previously trained models based on configuration.

    The loader validates that the model directory contains the expected
    structure and loads models for inference when training is disabled.
    """

    def __init__(self, config_path: Optional[str] = None, config: Optional[dict] = None) -> None:
        """Create a loader for pretrained models.

        Args:
            config_path: Path to a configuration file.
            config: Configuration dictionary.
        """

        self.logger = get_logger(self.__class__.__name__)
        self.config_loader = ConfigLoader(config_path=config_path, config=config)
        self.config = self.config_loader.get_config()
        self.base_path: Optional[str] = self.config.training.load_model_path

        if not self.config.training.train_enabled:
            if not self.base_path:
                raise ValueError(
                    "Training is disabled but no 'training.load_model_path' is provided in the config."
                )
            self.base_path = self.config_loader.resolve_path(self.base_path)
            self._validate_structure()

    def _validate_structure(self) -> None:
        """Ensure the expected directory hierarchy exists."""

        base = Path(self.base_path)
        if not base.exists():
            raise FileNotFoundError(f"Model directory not found: {base}")
        for model_type in ["propensity", "revenue"]:
            type_dir = base / model_type
            if not type_dir.exists():
                raise FileNotFoundError(f"Expected directory {type_dir} not found")
            for product in self.config.products:
                product_dir = type_dir / product
                if not product_dir.exists():
                    raise FileNotFoundError(
                        f"Expected directory for product '{product}' not found in {type_dir}"
                    )

    def load_model(self, model_type: str, product: str):
        """Return a deserialized model for the given type and product."""
        if not self.base_path:
            raise RuntimeError("Model loading requested without a base path")
        path = Path(self.base_path) / model_type / product
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")
        joblib_files = list(path.glob("*.joblib"))
        if not joblib_files:
            raise FileNotFoundError(f"No model file found in {path}")
        self.logger.info("Loaded model from %s", joblib_files[0])
        return joblib.load(joblib_files[0])
