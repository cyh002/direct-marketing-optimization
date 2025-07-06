"""Metric abstractions for offer evaluation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Type

import numpy as np


class Metric(ABC):
    """Abstract base metric."""

    name: str

    @abstractmethod
    def compute(
        self, selection: np.ndarray, propensity: np.ndarray, revenue: np.ndarray
    ) -> float:
        """Compute metric value."""


class TotalRevenueMetric(Metric):
    """Total expected revenue from selected offers."""

    name = "total_revenue"

    def compute(
        self, selection: np.ndarray, propensity: np.ndarray, revenue: np.ndarray
    ) -> float:
        expected = propensity * revenue
        return float(np.sum(expected * selection))


class RevenuePerContactMetric(Metric):
    """Expected revenue per contacted customer."""

    name = "revenue_per_contact"

    def compute(
        self, selection: np.ndarray, propensity: np.ndarray, revenue: np.ndarray
    ) -> float:
        contacts = max(int(selection.sum()), 1)
        total = TotalRevenueMetric().compute(selection, propensity, revenue)
        return float(total / contacts)


class AcceptanceRateMetric(Metric):
    """Average propensity of contacted customers."""

    name = "acceptance_rate"

    def compute(
        self, selection: np.ndarray, propensity: np.ndarray, revenue: np.ndarray
    ) -> float:
        contacts = max(int(selection.sum()), 1)
        return float(np.sum(propensity * selection) / contacts)


class ROIMetric(Metric):
    """Return on investment given contact cost."""

    name = "roi"

    def __init__(self, cost_per_contact: float = 1.0) -> None:
        self.cost_per_contact = cost_per_contact

    def compute(
        self, selection: np.ndarray, propensity: np.ndarray, revenue: np.ndarray
    ) -> float:
        contacts = max(int(selection.sum()), 1)
        total = TotalRevenueMetric().compute(selection, propensity, revenue)
        return float(total / (self.cost_per_contact * contacts))


METRIC_REGISTRY: Dict[str, Type[Metric]] = {
    TotalRevenueMetric.name: TotalRevenueMetric,
    RevenuePerContactMetric.name: RevenuePerContactMetric,
    AcceptanceRateMetric.name: AcceptanceRateMetric,
    ROIMetric.name: ROIMetric,
}


class ModelMetric(ABC):
    """Base class for model evaluation metrics."""

    name: str

    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute metric value."""


class PrecisionMetric(ModelMetric):
    """Precision for binary classification."""

    name = "precision"

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        from sklearn.metrics import precision_score

        return float(precision_score(y_true, y_pred))


class RecallMetric(ModelMetric):
    """Recall for binary classification."""

    name = "recall"

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        from sklearn.metrics import recall_score

        return float(recall_score(y_true, y_pred))


class F1Metric(ModelMetric):
    """F1 score for binary classification."""

    name = "f1"

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        from sklearn.metrics import f1_score

        return float(f1_score(y_true, y_pred))


class R2Metric(ModelMetric):
    """Coefficient of determination for regression."""

    name = "r2"

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        from sklearn.metrics import r2_score

        return float(r2_score(y_true, y_pred))


class AdjustedR2Metric(ModelMetric):
    """Adjusted :math:`R^2` taking number of features into account."""

    name = "adjusted_r2"

    def __init__(self, n_features: int) -> None:
        self.n_features = n_features

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        from sklearn.metrics import r2_score

        r2 = r2_score(y_true, y_pred)
        n = len(y_true)
        return float(1 - (1 - r2) * (n - 1) / (n - self.n_features - 1))


class RMSEMetric(ModelMetric):
    """Root mean squared error for regression."""

    name = "rmse"

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        from sklearn.metrics import mean_squared_error

        return float(mean_squared_error(y_true, y_pred, squared=False))


MODEL_METRIC_REGISTRY: Dict[str, Type[ModelMetric]] = {
    PrecisionMetric.name: PrecisionMetric,
    RecallMetric.name: RecallMetric,
    F1Metric.name: F1Metric,
    R2Metric.name: R2Metric,
    AdjustedR2Metric.name: AdjustedR2Metric,
    RMSEMetric.name: RMSEMetric,
}
