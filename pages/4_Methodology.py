# pages/4_Methodology.py

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

# LABEL MAP
CLASS_LABELS = {
    0: "Overvalued",
    1: "Fairly valued",
    2: "Undervalued",
    "0": "Overvalued",
    "1": "Fairly valued",
    "2": "Undervalued",
}

# FILE HELPERS
def find_file(filename):
    paths = [
        Path(filename),
        Path("data") / filename,
        Path(__file__).parent.parent / filename,
        Path(__file__).parent.parent / "data" / filename,
        Path(__file__).parent.parent / "models" / filename,
    ]
    for p in paths:
        if p.exists():
            return p
    return None


@st.cache_data
def load_csv(filename):
    path = find_file(filename)
    if path is None:
        return None
    return pd.read_csv(path)


# LOAD FILES
comparison_df = load_csv("valuation_model_comparison_results.csv")
pred_df = load_csv("valuation_test_predictions.csv")
coef_df = load_csv("valuation_final_model_coefficients.csv")

# HELPERS
def normalize_label(value):
    if pd.isna(value):
        return value
    return CLASS_LABELS.get(value, CLASS_LABELS.get(str(value), value))


def prettify_feature_name(feature):
    return (
        feature.replace("_", " ")
        .replace("roa", "ROA")
        .title()
        .replace("Roa", "ROA")
    )


def build_confusion_matrix(df):
    if df is None:
        return None

    required_cols = {"actual_valuation_label", "predicted_valuation_label"}
    if not required_cols.issubset(df.columns):
        return None

    actual = df["actual_valuation_label"].map(normalize_label)
    predicted = df["predicted_valuation_label"].map(normalize_label)

    ordered_labels = ["Overvalued", "Fairly valued", "Undervalued"]

    cm = pd.crosstab(
        pd.Categorical(actual, categories=ordered_labels),
        pd.Categorical(predicted, categories=ordered_labels),
        dropna=False
    )

    cm.index.name = "Actual"
    cm.columns.name = "Predicted"

    return cm


def compute_accuracy(df):
    if df is None:
        return None

    required_cols = {"actual_valuation_label", "predicted_valuation_label"}
    if not required_cols.issubset(df.columns):
        return None

    return (df["actual_valuation_label"] == df["predicted_valuation_label"]).mean()


def compute_per_class_metrics(df):
    if df is None:
        return None

    required_cols = {"actual_valuation_label", "predicted_valuation_label"}
    if not required_cols.issubset(df.columns):
        return None

    actual = df["actual_valuation_label"]
    predicted = df["predicted_valuation_label"]

    classes = sorted(set(actual.dropna().unique()).union(set(predicted.dropna().unique())))

    rows = []
    for cls in classes:
        tp = ((actual == cls) & (predicted == cls)).sum()
        fp = ((actual != cls) & (predicted == cls)).sum()
        fn = ((actual == cls) & (predicted != cls)).sum()
        support = (actual == cls).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        rows.append({
            "Class": normalize_label(cls),
            "Support": int(support),
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        })

    return pd.DataFrame(rows)


def coefficient_class_options(df):
    if df is None:
        return []

    options = []
    if "coefficient_class_0" in df.columns:
        options.append(("Overvalued", "coefficient_class_0"))
    if "coefficient_class_1" in df.columns:
        options.append(("Fairly valued", "coefficient_class_1"))
    if "coefficient_class_2" in df.columns:
        options.append(("Undervalued", "coefficient_class_2"))

    return options


# PAGE
st.title("Methodology")
st.caption(
    "This page explains the project objective, data sources, modeling approach, evaluation results, and workflow behind ValueLens."
)

top1, top2, top3, top4 = st.columns(4)
top1.metric("Final Model", "Logistic Regression")
top2.metric("Task Type", "3-Class Classification")
top3.metric("Sector Focus", "Healthcare")
top4.metric("Primary App Goal", "Valuation + Peers + Risk")

st.divider()

# PROJECT OBJECTIVE
st.markdown("## 1) Project Objective")
st.write(
    "ValueLens is a financial analytics app designed to evaluate healthcare companies from three complementary angles: "
    "**valuation**, **peer comparison**, and **market risk**. "
    "The core modeling task is a three-class classification problem that estimates whether a company appears "
    "**Overvalued**, **Fairly valued**, or **Undervalued** based on financial and valuation features."
)

# MODEL OUTPUT
st.markdown("## 2) What the Valuation Model Predicts")
st.markdown(
    """
- **Overvalued:** the company’s feature profile looks expensive relative to the model’s learned patterns  
- **Fairly valued:** the company’s profile appears broadly aligned with the model’s learned patterns  
- **Undervalued:** the company’s profile looks relatively attractive compared with the model’s learned patterns  
"""
)
st.caption(
    "This is a classification model, not a direct intrinsic-value calculator. It is designed to identify valuation status based on patterns in financial data."
)

# DATA SOURCES
st.markdown("## 3) Data Sources")

d1, d2, d3 = st.columns(3)

with d1:
    st.markdown("### WRDS / Compustat")
    st.write(
        "Used offline for historical firm fundamentals and accounting data. "
        "These data supported dataset construction, feature engineering, and model training."
    )

with d2:
    st.markdown("### yfinance")
    st.write(
        "Used for public/free market data components. "
        "This supports deployment-safe market-based analytics and optional live/public updates."
    )

with d3:
    st.markdown("### Kenneth R. French Data Library")
    st.write(
        "Used for factor-based risk modeling inputs that support CAPM-style market risk analysis."
    )

st.caption(
    "The deployed app does not rely on WRDS at runtime. WRDS was used offline to build and train the project artifacts; the app runs from saved CSV/PKL outputs and public/free sources where applicable."
)

# FEATURES
st.markdown("## 4) Feature Groups / Inputs")

f1, f2, f3 = st.columns(3)

with f1:
    st.markdown("### Profitability & Efficiency")
    st.markdown(
        """
- ROA  
- Operating Margin  
- Liquidity / working-capital style indicators  
"""
    )

with f2:
    st.markdown("### Growth & Change Features")
    st.markdown(
        """
- Revenue Growth  
- Year-over-year changes in key ratios  
- Directional changes in operating performance  
"""
    )

with f3:
    st.markdown("### Relative Valuation & Peer Context")
    st.markdown(
        """
- Price to Sales  
- Price to Book  
- Peer-relative versions of selected metrics  
- Relative leverage / profitability comparisons  
"""
    )

st.caption(
    "These feature groups were designed to capture not only a company’s own financial profile, but also how that profile compares with peers."
)

# FINAL MODEL CHOICE
st.markdown("## 5) Final Model Choice")

if comparison_df is not None and "model" in comparison_df.columns:
    logistic_row = comparison_df[comparison_df["model"].astype(str).str.lower() == "logistic regression"]

    if not logistic_row.empty:
        logistic_row = logistic_row.iloc[0]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Selected Model", str(logistic_row["model"]))
        if "train_accuracy" in logistic_row.index:
            m2.metric("Train Accuracy", f"{logistic_row['train_accuracy']:.1%}")
        if "test_accuracy" in logistic_row.index:
            m3.metric("Test Accuracy", f"{logistic_row['test_accuracy']:.1%}")
        if "test_f1" in logistic_row.index:
            m4.metric("Test Macro F1", f"{logistic_row['test_f1']:.3f}")

st.write(
    "The final model used in the app is **Logistic Regression**. It was selected because it offered a strong balance of "
    "**performance**, **interpretability**, and **clean deployment**. "
    "Compared with more complex alternatives, logistic regression makes it easier to explain why a company was classified into a given valuation category."
)

# MODEL EVALUATION
st.markdown("## 6) Model Evaluation")

accuracy = compute_accuracy(pred_df)
per_class_df = compute_per_class_metrics(pred_df)

if accuracy is not None:
    eval1, eval2 = st.columns(2)
    eval1.metric("Test Accuracy", f"{accuracy:.1%}")
    if comparison_df is not None and "model" in comparison_df.columns and "test_f1" in comparison_df.columns:
        logistic_row = comparison_df[comparison_df["model"].astype(str).str.lower() == "logistic regression"]
        if not logistic_row.empty:
            eval2.metric("Test Macro F1", f"{logistic_row.iloc[0]['test_f1']:.3f}")
else:
    st.warning(
        "The evaluation file was found, but the required columns "
        "`actual_valuation_label` and `predicted_valuation_label` were not available in the expected format."
    )

if per_class_df is not None:
    st.markdown("### Per-Class Summary")
    st.dataframe(
        per_class_df.style.format({
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

# CONFUSION MATRIX
st.markdown("### Confusion Matrix")

cm = build_confusion_matrix(pred_df)

if cm is not None:
    cm_long = (
        cm.reset_index()
        .melt(id_vars="Actual", var_name="Predicted", value_name="Count")
    )

    heatmap = (
        alt.Chart(cm_long)
        .mark_rect()
        .encode(
            x=alt.X("Predicted:N", title="Predicted Label"),
            y=alt.Y("Actual:N", title="Actual Label"),
            color=alt.Color("Count:Q", title="Count"),
            tooltip=["Actual", "Predicted", "Count"],
        )
        .properties(height=320)
    )

    text = (
        alt.Chart(cm_long)
        .mark_text(fontSize=13)
        .encode(
            x="Predicted:N",
            y="Actual:N",
            text="Count:Q",
            color=alt.condition(
                alt.datum.Count > cm_long["Count"].mean(),
                alt.value("white"),
                alt.value("black"),
            ),
        )
    )

    st.altair_chart(heatmap + text, use_container_width=True)
else:
    st.warning(
        "The confusion matrix could not be created from the current prediction file. "
        "The page expects `actual_valuation_label` and `predicted_valuation_label`."
    )

# ALTERNATIVE MODELS
st.markdown("## 7) Alternative Models Tested")

if comparison_df is not None:
    expected_cols = ["model", "train_accuracy", "test_accuracy", "train_f1", "test_f1"]
    available_cols = [col for col in expected_cols if col in comparison_df.columns]

    comparison_display = comparison_df[available_cols].copy()

    st.dataframe(
        comparison_display.style.format({
            "train_accuracy": "{:.1%}",
            "test_accuracy": "{:.1%}",
            "train_f1": "{:.3f}",
            "test_f1": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Several classification models were evaluated before selecting the final model. "
        "This comparison shows that the chosen model was intentional rather than arbitrary."
    )
else:
    st.warning("`valuation_model_comparison_results.csv` could not be found.")

# MODEL EXPLANATION
st.markdown("## 8) Model Explanation / Coefficient-Based Insight")

if coef_df is not None and "feature" in coef_df.columns:
    class_options = coefficient_class_options(coef_df)

    if class_options:
        class_label_lookup = {label: col for label, col in class_options}
        selected_label = st.selectbox(
            "Choose a valuation class to inspect",
            options=[label for label, _ in class_options]
        )
        selected_col = class_label_lookup[selected_label]

        coef_display = coef_df[["feature", selected_col]].copy()
        coef_display["abs_coef"] = coef_display[selected_col].abs()
        coef_display = coef_display.sort_values("abs_coef", ascending=False).head(12)
        coef_display["feature_label"] = coef_display["feature"].map(prettify_feature_name)

        coef_chart = (
            alt.Chart(coef_display)
            .mark_bar()
            .encode(
                x=alt.X(f"{selected_col}:Q", title="Coefficient"),
                y=alt.Y("feature_label:N", sort="-x", title=None),
                tooltip=[
                    alt.Tooltip("feature_label:N", title="Feature"),
                    alt.Tooltip(f"{selected_col}:Q", title="Coefficient", format=".4f")
                ],
            )
            .properties(height=360)
        )

        st.altair_chart(coef_chart, use_container_width=True)
        st.caption(
            "These coefficients show which features most strongly influence the logistic regression model for the selected class. "
            "Larger absolute values indicate stronger directional influence."
        )
    else:
        st.warning(
            "The coefficient file was found, but it does not contain the expected columns: "
            "`coefficient_class_0`, `coefficient_class_1`, and `coefficient_class_2`."
        )
else:
    st.warning("`valuation_final_model_coefficients.csv` could not be found.")

# WORKFLOW
st.markdown("## 9) Workflow Summary")

with st.expander("Show pipeline summary"):
    st.markdown(
        """
1. **Collect historical financial and market data** using course-approved sources  
2. **Construct cleaned modeling datasets** and engineer ratios, change features, and peer-relative features  
3. **Train and compare multiple models** offline using the prepared training data  
4. **Export deployment-safe artifacts** such as CSV files and saved model objects  
5. **Deploy the app** using saved files plus public/free sources where appropriate  
"""
    )

# LIMITATIONS
st.markdown("## 10) Limitations")

with st.expander("Show limitations and caveats"):
    st.markdown(
        """
- The valuation model is a **classification tool**, not a direct intrinsic value calculator  
- Model outputs depend on the selected features, training period, and class definitions  
- Peer-based comparisons are only as strong as the chosen comparison group  
- CAPM-style risk metrics simplify reality and do not capture all sources of uncertainty  
- Public/live add-ons should be treated as **supplemental snapshots**, not replacements for the main offline analysis  
"""
    )

st.divider()
st.caption(
    "This page now reads directly from the real project files: "
    "`valuation_test_predictions.csv`, `valuation_model_comparison_results.csv`, and `valuation_final_model_coefficients.csv`."
)
