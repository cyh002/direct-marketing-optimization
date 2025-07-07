"""Display optimized client list."""

from __future__ import annotations

import os
import pandas as pd
import streamlit as st
import plotly.express as px
from src.streamlit_utils import load_predictions

st.header("👥 Optimized Client List")
st.markdown(
    """
This page displays the final list of clients selected by the optimization algorithm for the marketing campaign.
These are the most promising prospects for each product based on predicted purchase probability and expected revenue.
"""
)

run_dir = st.session_state.get("run_dir")
if not run_dir:
    st.info("🔍 Please select a run on the Filter page first.")
    st.stop()

results_file = os.path.join(run_dir, "results", "optimized_offers.csv")
if not os.path.exists(results_file):
    st.warning("⚠️ optimized_offers.csv not found.")
    st.stop()

offers = pd.read_csv(results_file)

# Load prediction data to enrich the client list
prop, rev = load_predictions(run_dir)
if prop is not None and rev is not None:
    # Get product names from probability columns
    probability_columns = [c for c in prop.columns if c.startswith("probability_")]
    products = [c.replace("probability_", "") for c in probability_columns]

    # Merge with prediction data
    merged_offers = offers.merge(prop, on="Client", how="left").merge(
        rev, on="Client", how="left"
    )

    # Add expected revenue and probability columns
    for idx, row in merged_offers.iterrows():
        product = row["product"]
        if product in products:
            merged_offers.loc[idx, "purchase_probability"] = row.get(
                f"probability_{product}", None
            )
            merged_offers.loc[idx, "expected_revenue"] = row.get(
                f"expected_revenue_{product}", None
            )
else:
    merged_offers = offers

# Summary statistics
st.subheader("📊 Campaign Summary")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Clients", f"{len(merged_offers):,}")
with col2:
    num_products = merged_offers["product"].nunique()
    st.metric("Products Offered", num_products)
with col3:
    if "purchase_probability" in merged_offers.columns:
        avg_prob = merged_offers["purchase_probability"].mean()
        st.metric("Avg. Purchase Probability", f"{avg_prob:.1%}")

# Product distribution visualization
st.subheader("🍰 Product Distribution")

product_counts = merged_offers["product"].value_counts().reset_index()
product_counts.columns = ["Product", "Count"]

fig = px.pie(
    product_counts,
    values="Count",
    names="Product",
    title="Distribution of Offers by Product",
    color_discrete_sequence=px.colors.qualitative.Bold,
)
fig.update_traces(textposition="inside", textinfo="percent+label")
st.plotly_chart(fig, use_container_width=True)

# Client List with filters
st.subheader("🔍 Client Selection")

# Sidebar filters
st.sidebar.subheader("Filter Options")

# Product filter
if "product" in merged_offers.columns:
    product_list = ["All"] + sorted(merged_offers["product"].unique().tolist())
    product_filter = st.sidebar.selectbox("Filter by Product", options=product_list)
else:
    product_filter = "All"

# Probability threshold (if available)
prob_threshold = 0.0
if "purchase_probability" in merged_offers.columns:
    prob_threshold = st.sidebar.slider(
        "Min Purchase Probability",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        format="%.2f",
    )

# Revenue threshold (if available)
rev_threshold = 0.0
if "expected_revenue" in merged_offers.columns:
    rev_threshold = st.sidebar.slider(
        "Min Expected Revenue",
        min_value=0.0,
        max_value=(
            float(merged_offers["expected_revenue"].max())
            if not merged_offers["expected_revenue"].empty
            else 100.0
        ),
        value=0.0,
        step=10.0,
        format="%.2f",
    )

# Text search
search_term = st.sidebar.text_input("Search by Client ID")

# Apply filters
filtered_offers = merged_offers.copy()

if product_filter != "All":
    filtered_offers = filtered_offers[filtered_offers["product"] == product_filter]

if prob_threshold > 0 and "purchase_probability" in filtered_offers.columns:
    filtered_offers = filtered_offers[
        filtered_offers["purchase_probability"] >= prob_threshold
    ]

if rev_threshold > 0 and "expected_revenue" in filtered_offers.columns:
    filtered_offers = filtered_offers[
        filtered_offers["expected_revenue"] >= rev_threshold
    ]

if search_term:
    filtered_offers = filtered_offers[
        filtered_offers["Client"].astype(str).str.contains(search_term)
    ]

# Sorting options
sort_options = ["Client", "product"]
if "purchase_probability" in filtered_offers.columns:
    sort_options.append("purchase_probability")
if "expected_revenue" in filtered_offers.columns:
    sort_options.append("expected_revenue")

sort_col = st.selectbox("Sort by", options=sort_options)
sort_order = st.radio(
    "Sort order", options=["Ascending", "Descending"], horizontal=True
)
ascending = sort_order == "Ascending"

filtered_offers = filtered_offers.sort_values(by=sort_col, ascending=ascending)

# Display client list with pagination
st.subheader("📋 Client List")
st.write(
    f"Showing {len(filtered_offers)} clients (filtered from {len(merged_offers)} total)"
)

# Add formatting
if "purchase_probability" in filtered_offers.columns:
    filtered_offers["purchase_probability"] = filtered_offers[
        "purchase_probability"
    ].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
if "expected_revenue" in filtered_offers.columns:
    filtered_offers["expected_revenue"] = filtered_offers["expected_revenue"].apply(
        lambda x: f"€{x:.2f}" if pd.notnull(x) else "N/A"
    )

# Display controls
col1, col2 = st.columns(2)
with col1:
    max_rows = st.number_input(
        "Rows to display",
        min_value=1,
        max_value=len(filtered_offers),
        value=min(20, len(filtered_offers)),
    )
with col2:
    page_size = max_rows
    total_pages = (
        (len(filtered_offers) - 1) // page_size + 1 if len(filtered_offers) > 0 else 1
    )
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)

start_idx = (page - 1) * page_size
end_idx = start_idx + page_size
display_offers = filtered_offers.iloc[start_idx:end_idx].copy()

st.dataframe(display_offers, use_container_width=True)

st.markdown(f"Showing page {page} of {total_pages}")

# Download options
st.subheader("📥 Download Options")


@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


col1, col2 = st.columns(2)

with col1:
    csv_filtered = convert_df_to_csv(filtered_offers)
    st.download_button(
        label="Download Filtered List",
        data=csv_filtered,
        file_name="filtered_client_list.csv",
        mime="text/csv",
    )

with col2:
    csv_all = convert_df_to_csv(merged_offers)
    st.download_button(
        label="Download Full List",
        data=csv_all,
        file_name="complete_client_list.csv",
        mime="text/csv",
    )

st.markdown("---")
st.markdown(
    """
**Note:** This client list represents the optimal marketing targets based on the models' predictions.
To maximize campaign performance, focus on clients with higher purchase probability and expected revenue.
"""
)
