import numpy as np
import pytest
from src.evaluator import Evaluator
from src.metrics import (
    TotalRevenueMetric,
    RevenuePerContactMetric,
    AcceptanceRateMetric,
    ROIMetric,
)
import cvxpy as cp
from src.optimizer import Optimizer


def test_evaluator_metrics():
    """Check that Evaluator computes all metrics correctly."""
    selection = np.array([[1, 0], [0, 1]])
    propensity = np.array([[0.5, 0.2], [0.1, 0.8]])
    revenue = np.array([[100, 50], [200, 150]])

    evaluator = Evaluator(
        config=None,
        metrics=[
            TotalRevenueMetric(),
            RevenuePerContactMetric(),
            AcceptanceRateMetric(),
            ROIMetric(cost_per_contact=2.0),
        ],
        cost_per_contact=2.0,
    )
    results = evaluator.evaluate(selection, propensity, revenue)
    expected_total = (0.5 * 100) + (0.8 * 150)
    assert results["total_revenue"] == pytest.approx(expected_total)
    assert results["revenue_per_contact"] == pytest.approx(expected_total / 2)
    expected_acc = (0.5 + 0.8) / 2
    assert results["acceptance_rate"] == pytest.approx(expected_acc)
    assert results["roi"] == pytest.approx(expected_total / (2 * 2.0))


def test_evaluator_save(tmp_path):
    """Evaluator should persist metrics to the specified CSV file."""
    selection = np.array([[1]])
    propensity = np.array([[0.5]])
    revenue = np.array([[100]])
    evaluator = Evaluator(
        config=None,
        metrics=[TotalRevenueMetric()],
        cost_per_contact=1.0,
    )
    results = evaluator.evaluate(selection, propensity, revenue)
    path = tmp_path / "metrics.csv"
    saved_path = evaluator.save(results, str(path))
    assert saved_path == str(path.resolve())
    saved = np.genfromtxt(saved_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    assert saved["total_revenue"] == pytest.approx(50.0)


@pytest.mark.skipif("ECOS_BB" not in cp.installed_solvers(), reason="ECOS_BB solver not available")
def test_optimizer_simple():
    """Optimizer should respect contact limit and one-off constraints."""
    rev = np.array([[20, 10], [30, 5], [25, 15]])
    opt = Optimizer(contact_limit=2)
    selection = opt.solve(rev)
    assert selection.shape == rev.shape
    assert selection.sum() <= 2
    assert np.all(selection.sum(axis=1) <= 1)

