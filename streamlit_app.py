"""Entry point for the Streamlit dashboard."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Marketing Dashboard", layout="wide")

st.title("🚀 Direct Marketing Optimization")

st.markdown("""
## 📊 Project Overview

This dashboard presents the results of a direct marketing optimization project that helps banks target the right customers with the right products to maximize revenue. Using machine learning, we predict both:

1. **🎯 Propensity models**: Which customers are most likely to purchase each product
2. **💰 Revenue models**: How much revenue each customer is expected to generate

The dashboard then optimizes marketing campaigns by selecting the best customer-product pairs within contact limits.

### 📦 Products Analyzed
- 💼 Mutual Funds (MF)
- 💳 Credit Cards (CC)
- 💵 Consumer Loans (CL)

### ✨ Features

- **📈 Summary**: Compare metrics across different model runs
- **🔍 Filter**: Select specific model runs to analyze
- **🎲 Propensity**: View purchase likelihood predictions and feature importance
- **💸 Revenue**: Explore expected revenue predictions and key drivers
- **📏 Evaluation**: Analyze optimization performance metrics
- **👥 Client List**: View the final optimized marketing target list

### 🔗 Resources

- **🐙 GitHub**: [Direct Marketing Optimization](https://github.com/cyh002/direct-marketing-optimization)
- **📡 MLflow Server**: [http://localhost:5000](http://localhost:5000)
- **📚 Documentation**: See the project README for detailed instructions

### 🛠️ Tech Stack

This project uses Python 3.12 with:
- 🌐 Streamlit for visualization
- 🤖 Scikit-learn for machine learning models
- ⚙️ Hydra for configuration management
- 🔍 Optuna for hyperparameter tuning
- 📊 MLflow for experiment tracking
- 🧮 CVXPY for optimization
- 📦 Docker for containerization
""")

st.sidebar.success("Select a page above")

if st.button("🚀 Start MLflow Server"):
    st.markdown("Starting MLflow server on port 5000...")
    import os
    os.system("mlflow server --host 0.0.0.0 --port 5000 &")
    st.success("✅ MLflow server started! Visit [http://localhost:5000](http://localhost:5000) to view experiments.")
