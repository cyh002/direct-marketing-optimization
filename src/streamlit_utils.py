"""Utility functions for Streamlit dashboard."""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


BASE_OUTPUT_DIR = "outputs"


def list_run_directories(base_dir: str = BASE_OUTPUT_DIR) -> List[str]:
    """Return available Hydra run directories."""
    pattern = os.path.join(base_dir, "*", "*")
    return sorted(p for p in glob.glob(pattern) if os.path.isdir(p))


def load_predictions(run_dir: str) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load propensity and revenue predictions for a run.

    Missing files are ignored and ``None`` is returned instead of raising an
    exception.
    """
    prop_path = os.path.join(run_dir, "inference", "propensity_predictions.csv")
    rev_path = os.path.join(run_dir, "inference", "revenue_predictions.csv")

    prop = pd.read_csv(prop_path) if os.path.exists(prop_path) else None
    rev = pd.read_csv(rev_path) if os.path.exists(rev_path) else None
    return prop, rev


def load_test_data(run_dir: str) -> pd.DataFrame | None:
    """Load test dataset if present."""
    test_path = os.path.join(run_dir, "preprocessed", "test.csv")
    if os.path.exists(test_path):
        return pd.read_csv(test_path)
    return None


def load_metadata(model_dir: str) -> Dict:
    """Load model metadata from a directory."""
    meta_files = [f for f in os.listdir(model_dir) if f.endswith("_metadata.json")]
    if not meta_files:
        return {}
    with open(os.path.join(model_dir, meta_files[0]), "r", encoding="utf-8") as f:
        return json.load(f)


def get_feature_importance(pipeline_path: str) -> pd.Series | None:
    """Return sorted feature importances from a saved pipeline."""
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


def list_products(model_type_dir: str) -> List[str]:
    """Return available products for a model type."""
    if not os.path.exists(model_type_dir):
        return []
    return [p for p in os.listdir(model_type_dir) if os.path.isdir(os.path.join(model_type_dir, p))]


def classification_metrics(y_true: pd.Series, y_prob: pd.Series) -> Dict[str, float]:
    """Compute basic classification metrics."""
    y_pred = (y_prob > 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Compute regression evaluation metrics."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def load_offers(run_dir: str) -> pd.DataFrame | None:
    """Return optimized offers for a run if present."""
    path = os.path.join(run_dir, "results", "optimized_offers.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def summarize_run(run_dir: str) -> Dict:
    """Compute key metrics for a single run."""
    offers = load_offers(run_dir)
    total_rev = offers["expected_revenue"].sum() if offers is not None else float("nan")
    mean_prob = offers["probability"].mean() if offers is not None else float("nan")

    train_scores = []
    test_scores = []
    for model_type in ["propensity", "revenue"]:
        root = os.path.join(run_dir, "models", model_type)
        for prod in list_products(root):
            meta = load_metadata(os.path.join(root, prod))
            if meta:
                train_scores.append(meta.get("train_score"))
                test_scores.append(meta.get("test_score"))

    best_train = max(train_scores) if train_scores else float("nan")
    best_test = max(test_scores) if test_scores else float("nan")

    return {
        "run": os.path.basename(run_dir),
        "path": run_dir,
        "total_revenue": total_rev,
        "mean_probability": mean_prob,
        "best_train_score": best_train,
        "best_test_score": best_test,
    }


def summarize_runs(base_dir: str = BASE_OUTPUT_DIR) -> pd.DataFrame:
    """Aggregate summary statistics for all runs."""
    runs = list_run_directories(base_dir)
    rows = [summarize_run(r) for r in runs]
    return pd.DataFrame(rows)
