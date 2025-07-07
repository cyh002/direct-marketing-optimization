"""Run selection page."""
from __future__ import annotations

import streamlit as st
import os
import pandas as pd
import plotly.express as px
from src.streamlit_utils import list_run_directories, list_products, load_metadata

st.header("🔍 Run Selection")

st.markdown("""
This page lets you select a specific model run to analyze in detail. 
Once selected, you can navigate to other pages to explore propensity models, 
revenue predictions, and optimization results for this specific run.
""")

# Create columns for better layout
col1, col2 = st.columns([2, 1])

run_dirs = list_run_directories()
if not run_dirs:
    st.warning("⚠️ No run directories found in 'outputs'. Please run the optimization workflow first.")
else:
    with col1:
        st.subheader("Available Runs")
        idx = 0
        if "run_dir" in st.session_state and st.session_state["run_dir"] in run_dirs:
            idx = run_dirs.index(st.session_state["run_dir"])
        
        choice = st.selectbox(
            "Select a run to analyze:",
            run_dirs,
            index=idx,
            format_func=lambda x: f"Run: {os.path.basename(x)}"
        )
        st.session_state["run_dir"] = choice
        
        # Add a refresh button
        if st.button("🔄 Refresh Run List"):
            st.experimental_rerun()
    
    with col2:
        st.subheader("Selected Run")
        st.info(f"**Current selection:** {os.path.basename(choice)}")
        
        # Add timestamp if available
        try:
            timestamp = os.path.getmtime(choice)
            st.caption(f"Last modified: {pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d %H:%M:%S')}")
        except:
            pass
    
    # Create tabs for different model types
    tab1, tab2 = st.tabs(["🎯 Propensity Models", "💰 Revenue Models"])
    
    prop_root = os.path.join(choice, "models", "propensity")
    rev_root = os.path.join(choice, "models", "revenue")
    
    # Process model information
    propensity_models = []
    revenue_models = []
    
    # Get propensity models
    products = list_products(prop_root)
    for product in products:
        meta = load_metadata(os.path.join(prop_root, product))
        propensity_models.append({
            "Product": product,
            "Model": meta.get("model_name", "N/A"),
            "Train Score": meta.get("train_score", "N/A"),
            "Test Score": meta.get("test_score", "N/A")
        })
    
    # Get revenue models
    products = list_products(rev_root)
    for product in products:
        meta = load_metadata(os.path.join(rev_root, product))
        # For revenue models, handle negative MSE scores by negating them
        train_score = meta.get("train_score", "N/A")
        test_score = meta.get("test_score", "N/A")
        
        # Convert negative MSE scores to positive for display
        if isinstance(train_score, (int, float)) and train_score < 0:
            train_score = -train_score
        if isinstance(test_score, (int, float)) and test_score < 0:
            test_score = -test_score
            
        revenue_models.append({
            "Product": product,
            "Model": meta.get("model_name", "N/A"),
            "Train Score": train_score,
            "Test Score": test_score
        })
    
    # Display propensity models
    with tab1:
        if propensity_models:
            # Convert to DataFrame
            prop_df = pd.DataFrame(propensity_models)
            
            # Convert score columns to numeric, coercing errors to NaN
            for col in ['Train Score', 'Test Score']:
                prop_df[col] = pd.to_numeric(prop_df[col], errors='coerce')
            
            # Display the DataFrame
            st.dataframe(prop_df, use_container_width=True)
            
            # Add a visualization if there are multiple products
            if len(propensity_models) > 1:
                fig = px.bar(
                    prop_df,
                    x="Product", 
                    y=["Train Score", "Test Score"],
                    barmode="group",
                    title="Propensity Model Performance by Product",
                    color_discrete_sequence=["#36B37E", "#00B8D9"]
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No propensity models found for this run")
    
    # Display revenue models
    with tab2:
        if revenue_models:
            # Convert to DataFrame
            rev_df = pd.DataFrame(revenue_models)
            
            # Convert score columns to numeric, coercing errors to NaN
            for col in ['Train Score', 'Test Score']:
                rev_df[col] = pd.to_numeric(rev_df[col], errors='coerce')
            
            # Display the DataFrame
            st.dataframe(rev_df, use_container_width=True)
            
            # Add a visualization if there are multiple products
            if len(revenue_models) > 1:
                fig = px.bar(
                    rev_df,
                    x="Product", 
                    y=["Train Score", "Test Score"],
                    barmode="group",
                    title="Revenue Model Performance by Product",
                    color_discrete_sequence=["#6554C0", "#0052CC"]
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No revenue models found for this run")

    st.markdown("---")
    st.markdown("""
    **Next Steps:**
    - Navigate to the **Propensity** page to examine purchase likelihood models
    - Visit the **Revenue** page to explore expected revenue predictions
    - Check the **Evaluation** page to see optimization performance metrics
    - View the **Client List** to see the final marketing targets
    """)