"""Factory for creating models from configuration."""
from __future__ import annotations

import importlib
import os
from typing import Any, Dict

import yaml
from sklearn.base import BaseEstimator


class ModelFactory:
    """Create scikit-learn models from config files."""

    MODEL_MAPPING = {
        "LogisticRegression": "sklearn.linear_model.LogisticRegression",
        "RandomForest": "sklearn.ensemble.RandomForestRegressor",
        "RandomForestRegressor": "sklearn.ensemble.RandomForestRegressor",
    }

    @staticmethod
    def _instantiate(model_name: str, params: Dict[str, Any]) -> BaseEstimator:
        if model_name not in ModelFactory.MODEL_MAPPING:
            raise ValueError(f"Unsupported model '{model_name}'")
        module_path, class_name = ModelFactory.MODEL_MAPPING[model_name].rsplit(".", 1)
        module = importlib.import_module(module_path)
        model_cls = getattr(module, class_name)
        return model_cls(**params)

    @staticmethod
    def from_config(model_type: str, model_name: str, base_dir: str) -> BaseEstimator:
        """Instantiate a model based on YAML configuration."""
        subdir = f"{model_type}_model"
        config_path = os.path.join(base_dir, subdir, f"{model_name}.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model config not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        params = config.get("model", {})
        params.pop("name", None)
        return ModelFactory._instantiate(model_name, params)
