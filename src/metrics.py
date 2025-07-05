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
