import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Methodology", layout="wide")

# GLOBAL TICKER STATE
if "ticker" not in st.session_state:
    st.session_state["ticker"] = "A"

ticker = st.session_state["ticker"]

st.sidebar.markdown("---")
st.sidebar.markdown(f"### Current ticker: `{ticker}`")


# FILE LOADER (ROBUST PATH HANDLING)
def find_file(filename):
    paths = [
        Path(filename),
        Path("data") / filename,
        Path(__file__).parent.parent / filename,
        Path(__file__).parent.parent / "data" / filename,
    ]
    for p in paths:
        if p.exists():
            return p
    return None


@st.cache_data
def load_csv_safe(filename):
    path = find_file(filename)
    if path is None:
        return None
    return pd.read_csv(path)


# PAGE TITLE
st.title("Methodology")

st.markdown(
"""
This app evaluates healthcare companies using three perspectives:
- Valuation (model-based classification)
- Peer comparison (relative positioning)
- Risk analysis (CAPM-based metrics)
"""
)

# MODEL PREDICTION
st.markdown("## 1) What the Valuation Model Predicts")

st.markdown("""
- **Overvalued:** The company's profile appears expensive relative to learned patterns  
- **Fairly valued:** The company aligns with model expectations  
- **Undervalued:** The company appears attractive relative to peers and fundamentals  
""")

# DATA SOURCES
st.markdown("## 2) Data Sources")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**WRDS / Compustat**")
    st.write("Used offline for financial fundamentals and model training.")

with col2:
    st.markdown("**yfinance**")
    st.write("Used for market-based data where applicable.")

with col3:
    st.markdown("**Fama-French Data Library**")
    st.write("Used for factor-based risk modeling inputs.")

# FEATURES
st.markdown("## 3) Feature Groups / Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Profitability & Efficiency**")
    st.write("- ROA\n- Operating Margin\n- Liquidity indicators")

with col2:
    st.markdown("**Growth & Change Features**")
    st.write("- Revenue growth\n- Year-over-year changes\n- Trend direction")

with col3:
    st.markdown("**Relative Valuation & Peer Context**")
    st.write("- Price-to-sales\n- Price-to-book\n- Peer-relative metrics")

# FINAL MODEL
st.markdown("## 4) Final Model Choice")

st.write(
"""
The final model is **Logistic Regression**, selected for:
- strong performance
- interpretability
- clean deployment

It provides a clear mapping between financial features and valuation classification.
"""
)

# MODEL EVALUATION (FIXED)
st.markdown("## 5) Model Evaluation")

pred_df = load_csv_safe("valuation_test_predictions.csv")

if pred_df is not None:
    try:
        accuracy = (pred_df["actual_label"] == pred_df["predicted_label"]).mean()
        st.metric("Test Accuracy", f"{accuracy:.2%}")
    except:
        st.warning("Prediction file loaded but expected columns not found.")
else:
    st.warning("valuation_test_predictions.csv not found.")

# CONFUSION MATRIX (FIXED)
st.markdown("### Confusion Matrix")

if pred_df is not None:
    try:
        cm = pd.crosstab(pred_df["actual_label"], pred_df["predicted_label"])

        cm_df = cm.reset_index().melt(id_vars="actual_label")

        chart = alt.Chart(cm_df).mark_rect().encode(
            x="predicted_label:O",
            y="actual_label:O",
            color="value:Q"
        )

        st.altair_chart(chart, use_container_width=True)

    except:
        st.warning("Could not generate confusion matrix.")
else:
    st.info("Confusion matrix unavailable.")

# MODEL COMPARISON (FIXED)
st.markdown("## 6) Alternative Models Tested")

comp_df = load_csv_safe("valuation_model_comparison_results.csv")

if comp_df is not None:
    st.dataframe(comp_df, use_container_width=True)
else:
    st.warning("valuation_model_comparison_results.csv not found.")

# MODEL EXPLANATION
st.markdown("## 7) Model Explanation / Coefficients")

coef_df = load_csv_safe("valuation_final_model_coefficients.csv")

if coef_df is not None:
    selected_class = st.selectbox(
        "Choose a valuation class",
        coef_df["class"].unique()
    )

    subset = coef_df[coef_df["class"] == selected_class]

    chart = alt.Chart(subset).mark_bar().encode(
        x="coefficient:Q",
        y=alt.Y("feature:N", sort="-x")
    )

    st.altair_chart(chart, use_container_width=True)

else:
    st.warning("Coefficient file not found.")
