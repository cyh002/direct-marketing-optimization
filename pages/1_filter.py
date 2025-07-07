"""Run selection page."""
from __future__ import annotations

import streamlit as st
import os
import pandas as pd
from src.streamlit_utils import list_run_directories, list_products, load_metadata

st.header("Select Output Folder")
run_dirs = list_run_directories()
if not run_dirs:
    st.warning("No run directories found in 'outputs'.")
else:
    idx = 0
    if "run_dir" in st.session_state and st.session_state["run_dir"] in run_dirs:
        idx = run_dirs.index(st.session_state["run_dir"])
    choice = st.selectbox("Available runs", run_dirs, index=idx)
    st.session_state["run_dir"] = choice
    st.write(f"Current selection: {choice}")

    prop_root = os.path.join(choice, "models", "propensity")
    rev_root = os.path.join(choice, "models", "revenue")
    model_rows = []
    for model_type, root_dir in [("propensity", prop_root), ("revenue", rev_root)]:
        products = list_products(root_dir)
        for product in products:
            meta = load_metadata(os.path.join(root_dir, product))
            model_rows.append({
                "Model type": model_type,
                "Product": product,
                "Model": meta.get("model_name", "N/A"),
            })

    if model_rows:
        st.subheader("Models Used")
        st.table(pd.DataFrame(model_rows))
