"""Utilities for creating models from configuration."""
from __future__ import annotations

from typing import Dict, Type

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator

from .config_models import ModelConfig


_MODEL_REGISTRY: Dict[str, Type[BaseEstimator]] = {
    "LogisticRegression": LogisticRegression,
    "LogisticRegressionRegressor": LogisticRegression,
    "RandomForestRegressor": RandomForestRegressor,
}


def create_model(cfg: ModelConfig) -> BaseEstimator:
    """Instantiate a scikit-learn model from ``ModelConfig``.

    Args:
        cfg: Model configuration with ``name`` and hyperparameters.

    Returns:
        Instantiated scikit-learn estimator.

    Raises:
        KeyError: If ``cfg.name`` is not in the registry.
    """

    params = dict(cfg.model)
    name = params.pop("name", None)
    if not name:
        raise KeyError("Model configuration missing 'name'")
    try:
        model_cls = _MODEL_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown model name '{name}'") from exc
    return model_cls(**params)
