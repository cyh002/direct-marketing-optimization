"""Overview of all runs."""
from __future__ import annotations

import streamlit as st
from src.streamlit_utils import summarize_runs

st.header("Run Summary")
summary_df = summarize_runs()
if summary_df.empty:
    st.info("No run directories found in 'outputs'.")
    st.stop()

st.dataframe(summary_df)

best_rev = summary_df.loc[summary_df["total_revenue"].idxmax()]
st.subheader("Best Expected Revenue")
st.write(f"{best_rev['run']} - {best_rev['total_revenue']:.2f}")

best_prob = summary_df.loc[summary_df["mean_probability"].idxmax()]
st.subheader("Best Average Probability")
st.write(f"{best_prob['run']} - {best_prob['mean_probability']:.4f}")

best_train = summary_df.loc[summary_df["best_train_score"].idxmax()]
st.subheader("Best Train Score")
st.write(f"{best_train['run']} - {best_train['best_train_score']:.4f}")

best_test = summary_df.loc[summary_df["best_test_score"].idxmax()]
st.subheader("Best Test Score")
st.write(f"{best_test['run']} - {best_test['best_test_score']:.4f}")

st.bar_chart(summary_df.set_index("run")[["total_revenue", "mean_probability"]])
