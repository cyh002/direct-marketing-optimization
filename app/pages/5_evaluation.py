"""Evaluation metrics for optimized offers."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import streamlit as st
from utils import load_predictions
from src.evaluator import Evaluator
from src.metrics import (
    AcceptanceRateMetric,
    RevenuePerContactMetric,
    TotalRevenueMetric,
)

run_dir = st.session_state.get("run_dir")
if not run_dir:
    st.info("Select a run on the Filter page.")
    st.stop()

results_file = os.path.join(run_dir, "results", "optimized_offers.csv")
if not os.path.exists(results_file):
    st.warning("optimized_offers.csv not found.")
    st.stop()

offers = pd.read_csv(results_file)
prop, rev = load_predictions(run_dir)

st.header("Evaluation")

# Build matrices for Evaluator
products = [c.replace("probability_", "") for c in prop.columns]
prop_mat = prop[[f"probability_{p}" for p in products]].values
rev_mat = rev[[f"expected_revenue_{p}" for p in products]].values
selection = np.zeros_like(prop_mat)

for _, row in offers.iterrows():
    client = row["Client"]
    product = row["Product"]
    if client in prop.index and product in products:
        i = prop.index.get_loc(client)
        j = products.index(product)
        selection[i, j] = 1

metrics = [TotalRevenueMetric(), RevenuePerContactMetric(), AcceptanceRateMetric()]
evaluator = Evaluator(metrics=metrics)
results = evaluator.evaluate(selection, prop_mat, rev_mat)

st.write(pd.DataFrame(results, index=[0]))
