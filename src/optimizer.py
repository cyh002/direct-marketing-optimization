"""Optimization solver for marketing offers."""
from __future__ import annotations

from typing import Dict, Optional

import cvxpy as cp
import numpy as np


class Optimizer:
    """Solve the customer-product assignment problem using linear programming."""

    def __init__(
        self,
        contact_limit: int,
        max_fraction_per_product: Optional[Dict[int, float]] = None,
        min_expected_revenue: float = 0.0,
    ) -> None:
        self.contact_limit = contact_limit
        self.max_fraction_per_product = max_fraction_per_product or {}
        self.min_expected_revenue = min_expected_revenue

    def solve(self, expected_revenues: np.ndarray) -> np.ndarray:
        """Optimize which product to offer each customer.

        Args:
            expected_revenues: Matrix ``(n_customers, n_products)`` of expected
                revenues per customer-product pair.

        Returns:
            Binary matrix of the same shape indicating selected offers.
        """
        n_customers, n_products = expected_revenues.shape
        x = cp.Variable((n_customers, n_products), boolean=True)

        constraints = [
            cp.sum(x, axis=1) <= 1,
            cp.sum(x) <= self.contact_limit,
        ]

        if self.max_fraction_per_product:
            total_contacts = cp.sum(x)
            for j, fraction in self.max_fraction_per_product.items():
                constraints.append(cp.sum(x[:, j]) <= fraction * total_contacts)

        if self.min_expected_revenue > 0:
            constraints.append(
                cp.multiply(expected_revenues, x) >= self.min_expected_revenue * x
            )

        objective = cp.Maximize(cp.sum(cp.multiply(expected_revenues, x)))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS_BB)

        if x.value is None:
            raise RuntimeError("Optimization failed to find a solution")

        return np.rint(x.value).astype(int)
