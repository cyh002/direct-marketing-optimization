"""Run selection page."""
from __future__ import annotations

import streamlit as st
from src.streamlit_utils import list_run_directories

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
