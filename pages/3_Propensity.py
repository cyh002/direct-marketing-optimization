"""Propensity model visualizations."""
from __future__ import annotations

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from src.streamlit_utils import (
    load_predictions,
    load_test_data,
    list_products,
    get_feature_importance,
    classification_metrics,
    load_metadata,
)


run_dir = st.session_state.get("run_dir")
if not run_dir:
    st.info("Select a run on the Filter page.")
    st.stop()

st.header("🎯 Propensity Models")
st.markdown("""
This page shows the performance of models that predict the likelihood of customers purchasing each product.
Examine model metrics, feature importance, and predictions to understand what drives purchase decisions.
""")

prop, _ = load_predictions(run_dir)
if prop is None:
    st.warning("⚠️ Propensity predictions not found.")
    st.stop()
test_df = load_test_data(run_dir)
model_root = os.path.join(run_dir, "models", "propensity")
products = list_products(model_root)

for product in products:
    st.subheader(f"📊 {product} Product Analysis")
    col_name = f"probability_{product}"
    
    # Display metrics in columns
    if test_df is not None and f"Sale_{product}" in test_df:
        # Merge dataframes on Client column instead of reindexing
        merged_df = pd.merge(
            prop[['Client', col_name]],
            test_df[['Client', f"Sale_{product}"]],
            on='Client',
            how='inner'
        )
        y_true = merged_df[f"Sale_{product}"]
        y_prob = merged_df[col_name]
        metrics = classification_metrics(y_true, y_prob)
        
        # Show metrics as individual cards in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        with col2:
            st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        with col3:
            st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        with col4:
            st.metric("AUC", f"{metrics.get('auc', 0):.4f}")
        
        # Create two columns for visualizations
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Create enhanced confusion matrix
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            y_pred = (y_prob > 0.5).astype(int)
            cm = ConfusionMatrixDisplay.from_predictions(
                y_true, y_pred, 
                cmap='Blues',
                ax=ax,
                colorbar=False,
                display_labels=["No Purchase", "Purchase"]
            )
            plt.title(f"Confusion Matrix - {product}")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            st.pyplot(fig)
        
        with viz_col2:
            # Add ROC curve
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'AUC = {roc_auc:.4f}',
                line=dict(color='royalblue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color='gray', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=f'ROC Curve - {product}',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                yaxis=dict(scaleanchor="x", scaleratio=1),
                xaxis=dict(constrain='domain'),
                width=500,
                height=400,
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.5)'),
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig)
            
        # Add probability distribution chart
        st.subheader("Purchase Probability Distribution")
        hist_data = [
            y_prob[y_true == 0], 
            y_prob[y_true == 1]
        ]
        group_labels = ['Non-Purchasers', 'Purchasers']
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=hist_data[0],
            name=group_labels[0],
            opacity=0.75,
            marker_color='red',
            nbinsx=20
        ))
        fig.add_trace(go.Histogram(
            x=hist_data[1],
            name=group_labels[1],
            opacity=0.75,
            marker_color='green',
            nbinsx=20
        ))
        
        fig.update_layout(
            title_text='Distribution of Purchase Probabilities',
            xaxis_title_text='Probability',
            yaxis_title_text='Count',
            bargap=0.2,
            bargroupgap=0.1,
            barmode='overlay'
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
                st.metric("Model Type", meta.get('model_name', 'Unknown'))
            with model_col2:
                st.metric("Train Score", f"{meta.get('train_score', 'N/A'):.4f}" 
                          if isinstance(meta.get('train_score'), (int, float)) else "N/A")
            with model_col3:
                st.metric("Test Score", f"{meta.get('test_score', 'N/A'):.4f}" 
                          if isinstance(meta.get('test_score'), (int, float)) else "N/A")
                
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
                    orientation='h',
                    labels={'x': 'Importance', 'y': 'Feature'},
                    title=f'Top 15 Features - {product}',
                    color=top_features.values,
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("Show All Features"):
                    st.dataframe(pd.DataFrame({
                        'Feature': fi.index,
                        'Importance': fi.values
                    }).sort_values('Importance', ascending=False))
                    
    st.markdown("---")