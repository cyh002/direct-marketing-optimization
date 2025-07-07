"""Optimization solver for marketing offers."""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, field_validator

import cvxpy as cp
import numpy as np


class OptimizerSettings(BaseModel):
    """Parameters for :class:`Optimizer`."""

    contact_limit: int
    max_fraction_per_product: Optional[Dict[int, float]] = None
    min_expected_revenue: float = 0.0

    @field_validator("contact_limit")
    @classmethod
    def contact_limit_positive(cls, v: int) -> int:
        """Ensure ``contact_limit`` is positive."""
        if v <= 0:
            raise ValueError("contact_limit must be greater than zero")
        return v


class Optimizer:
    """Solve the customer-product assignment problem using linear programming."""

    def __init__(
        self,
        contact_limit: int,
        max_fraction_per_product: Optional[Dict[int, float]] = None,
        min_expected_revenue: float = 0.0,
    ) -> None:
        """Create a new optimizer instance.

        Args:
            contact_limit: Maximum number of customers to contact.
            max_fraction_per_product: Optional cap on the fraction of contacts per product index.
            min_expected_revenue: Minimum expected revenue for any selected offer.
        """

        settings = OptimizerSettings(
            contact_limit=contact_limit,
            max_fraction_per_product=max_fraction_per_product,
            min_expected_revenue=min_expected_revenue,
        )
        self.contact_limit = settings.contact_limit
        self.max_fraction_per_product = settings.max_fraction_per_product or {}
        self.min_expected_revenue = settings.min_expected_revenue

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
