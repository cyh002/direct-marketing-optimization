"""Evaluation utilities for marketing offers."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .config_models import ConfigSchema
from .logging import get_logger
from .metrics import (
    METRIC_REGISTRY,
    Metric,
    ROIMetric,
)


class Evaluator:
    """Compute evaluation metrics for optimized offers."""

    def __init__(
        self,
        config: Optional[Dict | ConfigSchema] = None,
        metrics: Optional[Sequence[str | Metric]] = None,
        cost_per_contact: float = 1.0,
    ) -> None:
        if config is None:
            self.config = None
            config_dict = {}
        elif isinstance(config, ConfigSchema):
            self.config = config
            config_dict = config.model_dump()
        else:
            self.config = ConfigSchema(**config)
            config_dict = config

        metric_names: Iterable[str | Metric]
        if metrics is not None:
            metric_names = metrics
        else:
            metric_names = config_dict.get("evaluation", {}).get("metrics", [])

        self.metrics: List[Metric] = []
        for m in metric_names:
            if isinstance(m, Metric):
                self.metrics.append(m)
            elif isinstance(m, str):
                metric_cls = METRIC_REGISTRY.get(m)
                if metric_cls is None:
                    raise ValueError(f"Unknown metric '{m}'")
                if metric_cls is ROIMetric:
                    self.metrics.append(metric_cls(cost_per_contact))
                else:
                    self.metrics.append(metric_cls())
            else:
                raise TypeError("metrics must be names or Metric instances")

        self.cost_per_contact = cost_per_contact
        self.logger = get_logger(self.__class__.__name__)

    def evaluate(
        self,
        selection: np.ndarray,
        propensity: np.ndarray,
        revenue: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate optimization results.

        Args:
            selection: Binary matrix ``(n_customers, n_products)`` where ``1``
                indicates a customer receives a product offer.
            propensity: Propensity predictions with the same shape as
                ``selection``.
            revenue: Revenue predictions with the same shape as ``selection``.

        Returns:
            Dictionary of metric name to value.
        """
        if selection.shape != propensity.shape or selection.shape != revenue.shape:
            raise ValueError(
                "Selection, propensity and revenue matrices must have the same shape"
            )

        results: Dict[str, float] = {}
        for metric in self.metrics:
            results[metric.name] = metric.compute(selection, propensity, revenue)
        return results


