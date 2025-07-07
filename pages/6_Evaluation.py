"""Evaluation metrics for optimized offers."""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.streamlit_utils import load_predictions

st.header("📏 Optimization Evaluation")
st.markdown(
    """
This page evaluates the performance of the optimized marketing campaign. 
It shows key metrics like total revenue, revenue per contact, acceptance rate, and ROI.
Compare product performance and understand the expected impact of your marketing efforts.
"""
)

run_dir = st.session_state.get("run_dir")
if not run_dir:
    st.info("🔍 Please select a run on the Filter page first.")
    st.stop()


# Add caching to expensive operations
@st.cache_data
def load_and_process_data(run_dir):
    results_file = os.path.join(run_dir, "results", "optimized_offers.csv")
    eval_metrics_file = os.path.join(run_dir, "results", "evaluation_metrics.csv")

    if not os.path.exists(results_file) or not os.path.exists(eval_metrics_file):
        return None, None, None, None, None, None

    offers = pd.read_csv(results_file)
    results_df = pd.read_csv(eval_metrics_file)
    prop, rev = load_predictions(run_dir)

    # Preprocess data once
    probability_columns = [c for c in prop.columns if c.startswith("probability_")]
    products = [c.replace("probability_", "") for c in probability_columns]

    # Create detailed results in a vectorized way
    detailed_results = offers.merge(
        prop[["Client"] + probability_columns], on="Client", how="left"
    ).merge(
        rev[["Client"] + [f"expected_revenue_{p}" for p in products]],
        on="Client",
        how="left",
    )

    # Use vectorized operations instead of row-by-row
    for product in products:
        mask = detailed_results["product"] == product
        detailed_results.loc[mask, "purchase_probability"] = detailed_results.loc[
            mask, f"probability_{product}"
        ]
        detailed_results.loc[mask, "expected_revenue"] = detailed_results.loc[
            mask, f"expected_revenue_{product}"
        ]

    return offers, prop, rev, detailed_results, products, results_df


offers, prop, rev, detailed_results, products, results_df = load_and_process_data(
    run_dir
)

if offers is None:
    st.error(
        f"Could not find `optimized_offers.csv` or `evaluation_metrics.csv` in `{run_dir}`."
    )
    st.stop()

results = results_df.iloc[0].to_dict()

# Display key metrics in a more attractive way
st.subheader("🔑 Key Performance Metrics")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Total Revenue", f"€{results.get('total_revenue', 0):,.2f}", delta="Projected"
    )
with col2:
    st.metric(
        "Revenue per Contact",
        f"€{results.get('revenue_per_contact', 0):,.2f}",
        delta="Per Customer",
    )
with col3:
    st.metric(
        "Acceptance Rate",
        f"{results.get('acceptance_rate', 0)*100:.2f}%",
        delta="Expected",
    )
with col4:
    st.metric("ROI", f"{results.get('roi', 0):.2f}x", delta="Return on Investment")

# Campaign overview
st.subheader("📊 Campaign Overview")

# Count offers by product
product_counts = offers["product"].value_counts().reset_index()
product_counts.columns = ["Product", "Count"]

# Calculate expected revenue by product
offers_with_prob = offers.merge(prop, on="Client", how="left").merge(
    rev, on="Client", how="left"
)

product_revenue = []
for product in products:
    mask = offers["product"] == product
    count = mask.sum()
    if count > 0:
        prob = offers_with_prob.loc[mask, f"probability_{product}"].mean()
        rev_per_contact = offers_with_prob.loc[
            mask, f"expected_revenue_{product}"
        ].mean()
        total_rev = count * prob * rev_per_contact
        product_revenue.append(
            {
                "Product": product,
                "Count": count,
                "Probability": prob,
                "Revenue per Sale": rev_per_contact,
                "Expected Revenue": total_rev,
            }
        )

product_revenue_df = pd.DataFrame(product_revenue)

# Create two columns for charts
col1, col2 = st.columns(2)

with col1:
    # Product distribution chart
    fig = px.pie(
        product_counts,
        values="Count",
        names="Product",
        title="Distribution of Offers by Product",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Expected revenue by product
    if not product_revenue_df.empty:
        fig = px.bar(
            product_revenue_df,
            x="Product",
            y="Expected Revenue",
            color="Product",
            title="Expected Revenue by Product",
            labels={"Expected Revenue": "Expected Revenue (€)"},
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_layout(xaxis_title="Product", yaxis_title="Expected Revenue (€)")
        st.plotly_chart(fig, use_container_width=True)

# Display detailed metrics for products
if not product_revenue_df.empty:
    st.subheader("📈 Product Performance Metrics")

    # Format the dataframe for better display
    display_df = product_revenue_df.copy()
    display_df["Probability"] = display_df["Probability"].apply(
        lambda x: f"{x*100:.2f}%"
    )
    display_df["Revenue per Sale"] = display_df["Revenue per Sale"].apply(
        lambda x: f"€{x:.2f}"
    )
    display_df["Expected Revenue"] = display_df["Expected Revenue"].apply(
        lambda x: f"€{x:.2f}"
    )

    st.dataframe(display_df, use_container_width=True)

# Customer segments analysis
st.subheader("👥 Customer Segments Analysis")

# Merge offers with customer data if available
try:
    test_path = os.path.join(run_dir, "preprocessed", "test.csv")
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        if "Client" in test_df.columns:
            merged_data = offers.merge(test_df, on="Client", how="left")

            # Look for demographic columns
            demo_cols = [
                c for c in merged_data.columns if c in ["Age", "Sex", "Sex_F", "Sex_M"]
            ]

            if demo_cols:
                st.write("Demographic breakdown of targeted customers:")

                tabs = st.tabs(
                    ["Age Distribution", "Gender Distribution", "Product by Age"]
                )

                with tabs[0]:
                    if "Age" in demo_cols:
                        fig = px.histogram(
                            merged_data,
                            x="Age",
                            nbins=20,
                            title="Age Distribution of Targeted Customers",
                            labels={"x": "Age", "y": "Count"},
                            color_discrete_sequence=["#3366CC"],
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with tabs[1]:
                    gender_col = None
                    if "Sex" in demo_cols:
                        gender_col = "Sex"
                    elif "Sex_F" in demo_cols and "Sex_M" in demo_cols:
                        # Create gender column from one-hot encoded columns
                        merged_data["Gender"] = merged_data.apply(
                            lambda row: (
                                "Female" if row.get("Sex_F", 0) == 1 else "Male"
                            ),
                            axis=1,
                        )
                        gender_col = "Gender"

                    if gender_col:
                        gender_counts = (
                            merged_data[gender_col].value_counts().reset_index()
                        )
                        gender_counts.columns = ["Gender", "Count"]

                        fig = px.pie(
                            gender_counts,
                            values="Count",
                            names="Gender",
                            title="Gender Distribution of Targeted Customers",
                            hole=0.4,
                            color_discrete_sequence=["#FF6692", "#3366CC"],
                        )
                        fig.update_traces(
                            textposition="inside", textinfo="percent+label"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with tabs[2]:
                    if "Age" in demo_cols:
                        fig = px.box(
                            merged_data,
                            x="product",
                            y="Age",
                            color="product",
                            title="Age Distribution by Product",
                            labels={"product": "Product", "Age": "Age"},
                            color_discrete_sequence=px.colors.qualitative.Bold,
                        )
                        st.plotly_chart(fig, use_container_width=True)
except Exception:
    st.info("Customer demographic analysis not available or error occurred.")


# Download options
st.subheader("📥 Download Results")


@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


# Prepare the detailed results for download
if detailed_results is not None:
    col1, col2 = st.columns(2)

    with col1:
        csv_results = convert_df_to_csv(results_df)
        st.download_button(
            label="Download Summary Metrics",
            data=csv_results,
            file_name="campaign_metrics_summary.csv",
            mime="text/csv",
        )

    with col2:
        csv_detailed = convert_df_to_csv(detailed_results)
        st.download_button(
            label="Download Detailed Results",
            data=csv_detailed,
            file_name="campaign_detailed_results.csv",
            mime="text/csv",
        )

st.markdown("---")
st.markdown(
    """
**Note:** The evaluation metrics provide estimates based on the models' predictions. 
Actual campaign performance may vary depending on market conditions and customer behavior.
"""
)
