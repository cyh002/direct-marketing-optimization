"""Display optimized client list."""
from __future__ import annotations

import os
import pandas as pd
import streamlit as st

run_dir = st.session_state.get("run_dir")
if not run_dir:
    st.info("Select a run on the Filter page.")
    st.stop()

results_file = os.path.join(run_dir, "results", "optimized_offers.csv")
if not os.path.exists(results_file):
    st.warning("optimized_offers.csv not found.")
    st.stop()

offers = pd.read_csv(results_file)

st.header("Optimized Client List")
max_rows = st.number_input("Rows to display", min_value=1, max_value=len(offers), value=min(20, len(offers)))
st.dataframe(offers.head(int(max_rows)))
