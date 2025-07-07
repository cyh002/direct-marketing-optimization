"""Evaluation metrics for optimized offers."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import streamlit as st
from src.streamlit_utils import load_predictions
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
if prop is None or rev is None:
    st.warning("Prediction files not found.")
    st.stop()

st.header("Evaluation")

# Build matrices for Evaluator
# Fix: Only consider columns that start with "probability_" to extract product names
probability_columns = [c for c in prop.columns if c.startswith("probability_")]
products = [c.replace("probability_", "") for c in probability_columns]
prop_mat = prop[probability_columns].values
rev_mat = rev[[f"expected_revenue_{p}" for p in products]].values
selection = np.zeros_like(prop_mat)

for _, row in offers.iterrows():
    client = row["Client"]
    product = row["product"]
    if client in prop["Client"].values and product in products:
        i = prop[prop["Client"] == client].index[0]
        j = products.index(product)
        selection[i, j] = 1

metrics = [TotalRevenueMetric(), RevenuePerContactMetric(), AcceptanceRateMetric()]
evaluator = Evaluator(metrics=metrics)
results = evaluator.evaluate(selection, prop_mat, rev_mat)

st.write(pd.DataFrame(results, index=[0]))