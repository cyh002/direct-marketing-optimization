"""Inference utilities for propensity and revenue models."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from .config_loader import ConfigLoader
from .model_loader import ModelLoader
from .logging import get_logger


class BaseInference(ABC):
    """Abstract base class for model inference."""

    model_type: str
    output_prefix: str

    def __init__(
        self, config_path: Optional[str] = None, config: Optional[dict] = None
    ) -> None:
        """Instantiate the inference helper.

        Args:
            config_path: Optional configuration file path.
            config: Optional configuration dictionary.
        """

        self.logger = get_logger(self.__class__.__name__)
        self.config_loader = ConfigLoader(config_path=config_path, config=config)
        self.config = self.config_loader.get_config()
        self.model_loader = ModelLoader(config_path=config_path, config=config)
        self.products = self.config.products
        self.output_dir = self.config_loader.resolve_path(
            self.config.inference.output_dir
        )
        os.makedirs(self.output_dir, exist_ok=True)

    @abstractmethod
    def _predict(self, model, X: pd.DataFrame) -> pd.Series:
        """Generate predictions for a single product."""

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict for all configured products."""
        results = pd.DataFrame(index=X.index)
        for product in self.products:
            model = self.model_loader.load_model(self.model_type, product)
            preds = self._predict(model, X)
            results[f"{self.output_prefix}_{product}"] = preds
        return results

    def save(self, df: pd.DataFrame, filename: str) -> str:
        """Persist predictions to CSV with client identifiers."""
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index_label="Client")
        self.logger.info("Saved inference results to %s", path)
        return path


class PropensityInference(BaseInference):
    """Inference for propensity models."""

    model_type = "propensity"
    output_prefix = "probability"

    def _predict(self, model, X: pd.DataFrame) -> pd.Series:
        if hasattr(model, "predict_proba"):
            return pd.Series(model.predict_proba(X)[:, 1], index=X.index)
        return pd.Series(model.predict(X), index=X.index)


class RevenueInference(BaseInference):
    """Inference for revenue models."""

    model_type = "revenue"
    output_prefix = "expected_revenue"

    def _predict(self, model, X: pd.DataFrame) -> pd.Series:
        return pd.Series(model.predict(X), index=X.index)
