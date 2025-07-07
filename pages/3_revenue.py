"""Revenue model visualizations."""
from __future__ import annotations

import os
import streamlit as st
import pandas as pd
from src.streamlit_utils import (
    load_predictions,
    load_test_data,
    list_products,
    get_feature_importance,
    regression_metrics,
    load_metadata,
)

run_dir = st.session_state.get("run_dir")
if not run_dir:
    st.info("Select a run on the Filter page.")
    st.stop()

st.header("Revenue Models")
_, rev = load_predictions(run_dir)
if rev is None:
    st.warning("Revenue predictions not found.")
    st.stop()
test_df = load_test_data(run_dir)
model_root = os.path.join(run_dir, "models", "revenue")
products = list_products(model_root)

for product in products:
    st.subheader(product)
    col_name = f"expected_revenue_{product}"
    if test_df is not None and f"Revenue_{product}" in test_df:
        # Merge dataframes on Client column
        merged_df = pd.merge(
            rev[['Client', col_name]],
            test_df[['Client', f"Revenue_{product}"]],
            on='Client',
            how='inner'
        )

        y_true = merged_df[f"Revenue_{product}"]
        y_pred = merged_df[col_name]
        metrics = regression_metrics(y_true, y_pred)
        st.write(pd.DataFrame(metrics, index=[0]))
    model_dir = os.path.join(model_root, product)
    if os.path.exists(model_dir):
        joblibs = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]
        if joblibs:
            fi = get_feature_importance(os.path.join(model_dir, joblibs[0]))
            if fi is not None:
                st.bar_chart(fi)
        meta = load_metadata(model_dir)
        if meta:
            st.json(meta)
