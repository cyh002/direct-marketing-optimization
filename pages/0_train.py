"""Train pipeline configuration page."""
from __future__ import annotations

import os
import subprocess
import yaml
import streamlit as st


def load_base_config() -> dict:
    """Load base Hydra configuration."""
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "conf", "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


st.header("Train")
base_cfg = load_base_config()
train_cfg = base_cfg.get("training", {})
opt_cfg = base_cfg.get("optimization", {})

with st.form("train_form"):
    k_folds = st.number_input(
        "Cross-validation folds",
        min_value=2,
        value=int(train_cfg.get("k_folds", 5)),
    )
    sample_fraction = st.slider(
        "Sample fraction",
        min_value=0.01,
        max_value=1.0,
        value=float(train_cfg.get("sample_fraction", 1.0)),
        step=0.01,
    )
    train_enabled = st.checkbox(
        "Enable training",
        value=bool(train_cfg.get("train_enabled", True)),
    )
    contact_limit = st.number_input(
        "Contact limit",
        min_value=1,
        value=int(opt_cfg.get("contact_limit", 100)),
    )
    submitted = st.form_submit_button("Run")

if submitted:
    overrides = [
        f"training.k_folds={int(k_folds)}",
        f"training.sample_fraction={float(sample_fraction)}",
        f"training.train_enabled={'true' if train_enabled else 'false'}",
        f"optimization.contact_limit={int(contact_limit)}",
    ]
    cmd = ["uv", "run", "main.py"] + overrides
    st.write("Running: " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    st.text(result.stdout)
    st.text(result.stderr)
    if result.returncode == 0:
        st.success("Training completed")
    else:
        st.error("Training failed")
