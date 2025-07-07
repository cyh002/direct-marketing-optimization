"""Revenue model visualizations."""

from __future__ import annotations

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

st.header("💰 Revenue Models")
st.markdown(
    """
This page shows the performance of models that predict the expected revenue from each product.
Examine model metrics, feature importance, and predictions to understand what drives customer spending.
"""
)

_, rev = load_predictions(run_dir)
if rev is None:
    st.warning("⚠️ Revenue predictions not found.")
    st.stop()
test_df = load_test_data(run_dir)
model_root = os.path.join(run_dir, "models", "revenue")
products = list_products(model_root)

for product in products:
    st.subheader(f"📊 {product} Revenue Analysis")
    col_name = f"expected_revenue_{product}"

    # Display metrics and scatter plot
    if test_df is not None and f"Revenue_{product}" in test_df:
        # Merge dataframes on Client column
        merged_df = pd.merge(
            rev[["Client", col_name]],
            test_df[["Client", f"Revenue_{product}"]],
            on="Client",
            how="inner",
        )

        y_true = merged_df[f"Revenue_{product}"]
        y_pred = merged_df[col_name]
        metrics = regression_metrics(y_true, y_pred)

        # Show metrics as individual cards in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAE", f"{metrics.get('mae', 0):.2f}")
        with col2:
            st.metric("MSE", f"{metrics.get('mse', 0):.2f}")
        with col3:
            st.metric("R²", f"{metrics.get('r2', 0):.4f}")

        # Create scatter plot of actual vs predicted
        fig = px.scatter(
            x=y_true,
            y=y_pred,
            labels={"x": "Actual Revenue", "y": "Predicted Revenue"},
            title=f"Actual vs Predicted Revenue - {product}",
        )

        # Add perfect prediction line
        max_val = max(y_true.max(), y_pred.max())
        min_val = min(y_true.min(), y_pred.min())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Perfect Prediction",
                line=dict(color="red", dash="dash"),
            )
        )

        fig.update_layout(
            width=800,
            height=600,
            xaxis=dict(title="Actual Revenue"),
            yaxis=dict(title="Predicted Revenue"),
        )

        st.plotly_chart(fig)

        # Add residual plot
        residuals = y_pred - y_true

        fig = px.scatter(
            x=y_pred,
            y=residuals,
            labels={"x": "Predicted Revenue", "y": "Residuals"},
            title=f"Residual Plot - {product}",
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red")

        fig.update_layout(
            width=800,
            height=400,
            xaxis=dict(title="Predicted Revenue"),
            yaxis=dict(title="Residuals (Predicted - Actual)"),
        )

        st.plotly_chart(fig)

        # Add residual distribution
        fig = px.histogram(
            residuals,
            title="Distribution of Residuals",
            labels={"value": "Residual Value"},
            nbins=30,
        )

        fig.update_layout(
            width=800,
            height=400,
            xaxis=dict(title="Residuals"),
            yaxis=dict(title="Count"),
            showlegend=False,
        )

        st.plotly_chart(fig)

    # Model metadata and feature importance
    model_dir = os.path.join(model_root, product)
    if os.path.exists(model_dir):
        meta = load_metadata(model_dir)
        if meta:
            st.subheader("Model Details")
            model_col1, model_col2, model_col3 = st.columns(3)
            with model_col1:
                st.metric("Model Type", meta.get("model_name", "Unknown"))

            # For revenue models, handle negative MSE scores by negating them for display
            train_score = meta.get("train_score", "N/A")
            test_score = meta.get("test_score", "N/A")

            if isinstance(train_score, (int, float)) and train_score < 0:
                train_score = -train_score
            if isinstance(test_score, (int, float)) and test_score < 0:
                test_score = -test_score

            with model_col2:
                st.metric(
                    "Train Score",
                    (
                        f"{train_score:.4f}"
                        if isinstance(train_score, (int, float))
                        else "N/A"
                    ),
                )
            with model_col3:
                st.metric(
                    "Test Score",
                    (
                        f"{test_score:.4f}"
                        if isinstance(test_score, (int, float))
                        else "N/A"
                    ),
                )

        # Feature importance visualization
        joblibs = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]
        if joblibs:
            fi = get_feature_importance(os.path.join(model_dir, joblibs[0]))
            if fi is not None:
                st.subheader("Feature Importance")

                # Take top 15 features for better visualization
                top_features = fi.sort_values(ascending=False).head(15)

                fig = px.bar(
                    x=top_features.values,
                    y=top_features.index,
                    orientation="h",
                    labels={"x": "Importance", "y": "Feature"},
                    title=f"Top 15 Features - {product}",
                    color=top_features.values,
                    color_continuous_scale="Viridis",
                )

                fig.update_layout(
                    yaxis={"categoryorder": "total ascending"}, height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Show All Features"):
                    st.dataframe(
                        pd.DataFrame(
                            {"Feature": fi.index, "Importance": fi.values}
                        ).sort_values("Importance", ascending=False)
                    )

    st.markdown("---")
