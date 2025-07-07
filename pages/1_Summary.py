"""Overview of all runs."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
from src.streamlit_utils import summarize_runs

st.header("📊 Run Summary Dashboard")
st.markdown(
    """
This dashboard provides an overview of all model runs, comparing key performance metrics across experiments.
Use this page to identify the best-performing models for direct marketing optimization.
"""
)

# Get summary data
summary_df = summarize_runs()
if summary_df.empty:
    st.info(
        "No run directories found in 'outputs'. Please run the optimization workflow first."
    )
    st.stop()

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["📈 Key Metrics", "🏆 Top Performers", "📋 Detailed Data"])

with tab1:
    st.subheader("Performance Overview")

    # Create two metrics columns
    col1, col2 = st.columns(2)

    with col1:
        # Revenue metrics visualization
        fig_revenue = px.bar(
            summary_df,
            x="run",
            y="total_revenue",
            title="Expected Revenue by Run",
            labels={"total_revenue": "Expected Revenue", "run": "Run ID"},
            color="total_revenue",
            color_continuous_scale="Viridis",
        )
        fig_revenue.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_revenue, use_container_width=True)

    with col2:
        # Probability metrics visualization
        fig_prob = px.bar(
            summary_df,
            x="run",
            y="mean_probability",
            title="Mean Purchase Probability by Run",
            labels={"mean_probability": "Mean Probability", "run": "Run ID"},
            color="mean_probability",
            color_continuous_scale="Viridis",
        )
        fig_prob.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_prob, use_container_width=True)

    # Model scores visualization
    st.subheader("Model Scores Comparison")
    score_df = pd.melt(
        summary_df,
        id_vars=["run"],
        value_vars=["best_train_score", "best_test_score"],
        var_name="score_type",
        value_name="score",
    )
    score_df["score_type"] = score_df["score_type"].map(
        {"best_train_score": "Train Score", "best_test_score": "Test Score"}
    )

    fig_scores = px.bar(
        score_df,
        x="run",
        y="score",
        color="score_type",
        barmode="group",
        title="Train vs. Test Scores by Run",
        labels={"score": "Score Value", "run": "Run ID", "score_type": "Score Type"},
    )
    fig_scores.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_scores, use_container_width=True)

with tab2:
    st.subheader("Top Performing Models")

    # Create metrics for best performers
    col1, col2 = st.columns(2)

    with col1:
        best_rev = summary_df.loc[summary_df["total_revenue"].idxmax()]
        st.metric(
            label="🏆 Highest Expected Revenue",
            value=f"€{best_rev['total_revenue']:,.2f}",
            delta=f"Run: {best_rev['run']}",
        )

        best_train = summary_df.loc[summary_df["best_train_score"].idxmax()]
        st.metric(
            label="🎯 Best Train Score",
            value=f"{best_train['best_train_score']:.4f}",
            delta=f"Run: {best_train['run']}",
        )

    with col2:
        best_prob = summary_df.loc[summary_df["mean_probability"].idxmax()]
        st.metric(
            label="✅ Best Average Probability",
            value=f"{best_prob['mean_probability']:.4f}",
            delta=f"Run: {best_prob['run']}",
        )

        best_test = summary_df.loc[summary_df["best_test_score"].idxmax()]
        st.metric(
            label="📊 Best Test Score",
            value=f"{best_test['best_test_score']:.4f}",
            delta=f"Run: {best_test['run']}",
        )

    # Add chart showing performance of best model across metrics
    st.subheader("Best Overall Model Analysis")
    # Get run with highest average rank across all metrics
    summary_df["avg_rank"] = (
        summary_df[
            ["total_revenue", "mean_probability", "best_train_score", "best_test_score"]
        ]
        .rank(ascending=False)
        .mean(axis=1)
    )
    best_overall = summary_df.loc[summary_df["avg_rank"].idxmin()]

    st.info(f"Best overall model based on average rank: **{best_overall['run']}**")

    # Create radar chart data for the best model
    best_model_metrics = {
        "Metric": ["Revenue", "Probability", "Train Score", "Test Score"],
        "Value": [
            best_overall["total_revenue"] / summary_df["total_revenue"].max(),
            best_overall["mean_probability"] / summary_df["mean_probability"].max(),
            best_overall["best_train_score"] / summary_df["best_train_score"].max(),
            best_overall["best_test_score"] / summary_df["best_test_score"].max(),
        ],
    }
    best_metrics_df = pd.DataFrame(best_model_metrics)

    fig_radar = px.line_polar(
        best_metrics_df,
        r="Value",
        theta="Metric",
        line_close=True,
        range_r=[0, 1],
        title=f"Performance Profile: {best_overall['run']}",
    )
    fig_radar.update_traces(fill="toself")
    st.plotly_chart(fig_radar, use_container_width=True)

with tab3:
    st.subheader("Complete Data")
    st.dataframe(summary_df, use_container_width=True)

    st.download_button(
        label="Download Summary Data",
        data=summary_df.to_csv().encode("utf-8"),
        file_name="model_runs_summary.csv",
        mime="text/csv",
    )

st.markdown("---")
st.markdown(
    """
**Note:** The dashboard shows performance metrics across all model runs. 
To explore a specific run in detail, use the Filter page to select a run and navigate to other analysis pages.
"""
)
