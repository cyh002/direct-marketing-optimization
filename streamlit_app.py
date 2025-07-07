"""Streamlit dashboard to visualize model run outputs."""
from __future__ import annotations

import glob
import json
import os
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def list_run_directories(base_dir: str = "outputs") -> List[str]:
    """Return available Hydra run directories.

    Args:
        base_dir: Base outputs directory.

    Returns:
        Sorted list of run directory paths.
    """
    pattern = os.path.join(base_dir, "*", "*")
    return sorted([p for p in glob.glob(pattern) if os.path.isdir(p)])


def load_predictions(run_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load propensity and revenue predictions for a run."""
    prop_path = os.path.join(run_dir, "inference", "propensity_predictions.csv")
    rev_path = os.path.join(run_dir, "inference", "revenue_predictions.csv")
    prop = pd.read_csv(prop_path, index_col="Client")
    rev = pd.read_csv(rev_path, index_col="Client")
    return prop, rev


def load_test_data(run_dir: str) -> pd.DataFrame | None:
    """Load test split if available."""
    test_path = os.path.join(run_dir, "preprocessed", "test.csv")
    if os.path.exists(test_path):
        return pd.read_csv(test_path)
    return None


def load_metadata(model_dir: str) -> dict:
    """Load model metadata from directory."""
    meta_files = [f for f in os.listdir(model_dir) if f.endswith("_metadata.json")]
    if not meta_files:
        return {}
    with open(os.path.join(model_dir, meta_files[0]), "r", encoding="utf-8") as f:
        return json.load(f)


def get_feature_importance(pipeline_path: str) -> pd.Series | None:
    """Extract feature importances from a saved pipeline."""
    pipe = joblib.load(pipeline_path)
    preproc = pipe.named_steps.get("preprocessor")
    model = pipe.named_steps.get("model")

    if hasattr(model, "coef_"):
        importances = np.ravel(model.coef_)
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return None

    if preproc and hasattr(preproc, "get_feature_names_out"):
        names = preproc.get_feature_names_out()
    else:
        names = getattr(pipe, "feature_names_in_", None)
    if names is None or len(names) != len(importances):
        names = [f"f{i}" for i in range(len(importances))]
    return pd.Series(importances, index=names).sort_values(ascending=False)


def plot_confusion_matrix(y_true: pd.Series, y_prob: pd.Series) -> None:
    """Display confusion matrix using a 0.5 threshold."""
    cm = confusion_matrix(y_true, y_prob > 0.5)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    st.pyplot(disp.figure_)


def show_confusion_matrices(prop: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Render confusion matrices for each product."""
    st.header("Confusion Matrices")
    products = [c.replace("probability_", "") for c in prop.columns]
    for product in products:
        col = f"probability_{product}"
        if f"Sale_{product}" not in test_df:
            continue
        st.subheader(product)
        plot_confusion_matrix(test_df[f"Sale_{product}"], prop[col].reindex(test_df.index))


def show_feature_importances(run_dir: str) -> None:
    """Render bar charts of feature importances."""
    st.header("Feature Importances")
    for model_type in ["propensity", "revenue"]:
        st.subheader(model_type.capitalize())
        type_dir = os.path.join(run_dir, "models", model_type)
        if not os.path.exists(type_dir):
            continue
        for product in os.listdir(type_dir):
            model_dir = os.path.join(type_dir, product)
            joblib_files = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]
            if not joblib_files:
                continue
            series = get_feature_importance(os.path.join(model_dir, joblib_files[0]))
            if series is None:
                continue
            st.markdown(f"**{product}**")
            st.bar_chart(series)
            metadata = load_metadata(model_dir)
            if metadata:
                st.write(metadata)


def main() -> None:
    """Streamlit app entry point."""
    st.title("Direct Marketing Run Dashboard")
    run_dirs = list_run_directories()
    if not run_dirs:
        st.warning("No run directories found in 'outputs'.")
        return
    run_dir = st.sidebar.selectbox("Select run", run_dirs)
    prop, _ = load_predictions(run_dir)
    test_df = load_test_data(run_dir)
    if test_df is not None:
        show_confusion_matrices(prop, test_df)
    else:
        st.info("Test data not available for confusion matrices.")
    show_feature_importances(run_dir)
    results_file = os.path.join(run_dir, "results", "optimized_offers.csv")
    if os.path.exists(results_file):
        offers = pd.read_csv(results_file)
        st.header("Optimized Offers")
        st.dataframe(offers.head())


if __name__ == "__main__":
    main()
