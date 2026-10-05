import streamlit as st
import pandas as pd
import plotly.express as px

from utils.model_evaluation import (
    load_model_comparison,
    rank_models,
    prepare_metrics_for_chart,
    get_confusion_matrix_df,
    get_gb_classification_report,
    get_gb_feature_importance
)

st.set_page_config(
    page_title="Model Evaluation",
    page_icon="🏆",
    layout="wide"
)

st.title("🏆 Predictive Model Evaluation")
st.caption(
    "Evaluate the classification models used to predict "
    "Observed Learning Progress Track."
)

@st.cache_data
def get_results():
    return load_model_comparison()

results = get_results()

# ============================================================
# MODEL COMPARISON
# ============================================================
st.header("1. 🏆 Model Comparison")

if results is not None and not results.empty:
    ranked = rank_models(results)
    display_df = ranked.copy()

    for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        if col in display_df.columns:
            display_df[col] = (
                display_df[col] * 100
            ).round(2).astype(str) + "%"

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    chart_df = prepare_metrics_for_chart(results)

    fig = px.bar(
        chart_df,
        x="Model",
        y=["Accuracy", "Precision", "Recall", "F1 Score"],
        barmode="group",
        title="Accuracy, Precision, Recall and F1 Score"
    )
    fig.update_layout(
        yaxis_title="Score (%)",
        xaxis_title="Model",
        yaxis_range=[0, 100]
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("model_comparison.csv could not be found.")

# ============================================================
# GRADIENT BOOSTING DETAILED RESULTS
# ============================================================
st.header("2. 🚀 Gradient Boosting Detailed Evaluation")

gb_report = get_gb_classification_report()

if gb_report is not None and not gb_report.empty:
    display_report = gb_report.copy()

    for col in ["Precision", "Recall", "F1 Score"]:
        if col in display_report.columns:
            display_report[col] = (
                display_report[col] * 100
            ).round(2).astype(str) + "%"

    st.dataframe(
        display_report,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("""
### Interpretation

**Fast Progress** was predicted most effectively.

**Moderate Progress** achieved a lower level of performance.

**Slow Progress** was the most difficult class to distinguish.

The minority-class difficulty is important when interpreting the final
predictive performance.
""")

# ============================================================
# CONFUSION MATRIX
# ============================================================
st.header("3. 🔲 Gradient Boosting Confusion Matrix")

cm_df = get_confusion_matrix_df()

if cm_df is not None and not cm_df.empty:
    fig = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        title="Gradient Boosting Confusion Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
### How to read the matrix

The diagonal values represent correct predictions.

The off-diagonal values represent misclassifications.

The matrix helps show which learning-progress groups are easier or
more difficult for the model to distinguish.
""")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
st.header("4. 🧠 Important Predictive Features")

importance_df = get_gb_feature_importance()

if importance_df is not None and not importance_df.empty:
    top_features = importance_df.head(10).sort_values("Importance")

    fig = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 10 Gradient Boosting Feature Importances"
    )
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
### Key finding

Feature importance indicates which input variables contributed most to
the Gradient Boosting model's decisions.

In the current project results, **Daily Study Time** is the most important
feature, followed by **Weekly Practice Frequency**, **English Learning
Motivation**, **Listening Comprehension Confidence**, and
**English Usage Frequency**.

Feature importance shows contribution to model decisions; it does not
prove that a feature directly causes learning progress.
""")

# ============================================================
# FINAL MODEL SELECTION
# ============================================================
st.header("5. 🎯 Final Model Selection")

st.success(
    "Gradient Boosting was selected as the final predictive model because "
    "it achieved the highest reported overall performance."
)

final_df = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting"
    ],
    "Accuracy": [
        "50.50%",
        "56.50%",
        "57.00%"
    ],
    "F1 Score": [
        "50.70%",
        "56.78%",
        "57.21%"
    ],
    "Decision": [
        "Baseline",
        "Strong alternative",
        "Selected"
    ]
})

st.dataframe(
    final_df,
    use_container_width=True,
    hide_index=True
)

st.markdown("""
### Evaluation conclusion

The three classification approaches were compared using Accuracy,
Precision, Recall, and F1 Score. Gradient Boosting produced the strongest
reported result and was therefore selected for the predictive component.

This page focuses only on predictive model evaluation. Descriptive
statistics, learner profiling, K-Means clustering, and hidden-pattern
analysis are handled by the Descriptive Analytics page.
""")
