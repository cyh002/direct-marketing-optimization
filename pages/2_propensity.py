"""Propensity model visualizations."""
from __future__ import annotations

import os
import streamlit as st
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from src.streamlit_utils import (
    load_predictions,
    load_test_data,
    list_products,
    get_feature_importance,
    classification_metrics,
    load_metadata,
)


run_dir = st.session_state.get("run_dir")
if not run_dir:
    st.info("Select a run on the Filter page.")
    st.stop()

st.header("Propensity Models")
prop, _ = load_predictions(run_dir)
if prop is None:
    st.warning("Propensity predictions not found.")
    st.stop()
test_df = load_test_data(run_dir)
model_root = os.path.join(run_dir, "models", "propensity")
products = list_products(model_root)

for product in products:
    st.subheader(product)
    col_name = f"probability_{product}"
    if test_df is not None and f"Sale_{product}" in test_df:
        # Merge dataframes on Client column instead of reindexing
        merged_df = pd.merge(
            prop[['Client', col_name]],
            test_df[['Client', f"Sale_{product}"]],
            on='Client',
            how='inner'
        )
        y_true = merged_df[f"Sale_{product}"]
        y_prob = merged_df[col_name]
        metrics = classification_metrics(y_true, y_prob)
        st.write(pd.DataFrame(metrics, index=[0]))
        cm = ConfusionMatrixDisplay.from_predictions(
            y_true, (y_prob > 0.5).astype(int)
        )
        st.pyplot(cm.figure_)
    model_dir = os.path.join(model_root, product)
    if os.path.exists(model_dir):
        meta = load_metadata(model_dir)
        if meta:
            st.markdown(f"**Model:** {meta.get('model_name', 'Unknown')}")
            st.write(
                {
                    "train_score": meta.get("train_score"),
                    "test_score": meta.get("test_score"),
                }
            )
        joblibs = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]
        if joblibs:
            fi = get_feature_importance(os.path.join(model_dir, joblibs[0]))
            if fi is not None:
                st.bar_chart(fi.sort_values(ascending=False))
