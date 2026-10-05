import streamlit as st

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Overview - English Learning Data Mining Portal",
    page_icon="📌",
    layout="wide"
)

# =========================================================
# HEADER
# =========================================================
st.title("📌 Project Overview & Methodology")

st.markdown("""
This **English Learning Data Mining Portal** analyzes **2,000 English learner
records** to identify learning patterns, predict learner progress, and provide
data-driven recommendations.

The system combines **descriptive analytics, supervised machine learning,
and FP-Growth association rule mining**. Three classification models are
evaluated—**Decision Tree, Random Forest, and Gradient Boosting**—while
FP-Growth is used separately to discover relationships among learning habits
and learner characteristics.
""")

st.divider()

# =========================================================
# PROJECT METRICS
# =========================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Dataset Size", "2,000 Records")

with col2:
    st.metric("Predictor Variables", "18")

with col3:
    st.metric("Classification Models", "3")

with col4:
    st.metric("Pattern Mining", "FP-Growth")

st.divider()

# =========================================================
# PROBLEM AND SOLUTION
# =========================================================
st.subheader("🎯 Problem Addressed")

st.markdown("""
Traditional English-learning surveys can collect large amounts of learner
information, but manually interpreting these responses is difficult.

The system addresses three main problems:

- **Learner progress is difficult to predict** from multiple demographic,
  behavioral, and confidence-related variables.
- **Imbalanced progress classes** can cause a machine-learning model to favor
  the majority classes.
- **Relationships between learning habits** may not be obvious from individual
  variables alone.

To address these problems, the proposed system combines classification,
class-balancing, model comparison, and association-rule mining.
""")

st.divider()

# =========================================================
# METHODOLOGY
# =========================================================
st.subheader("🧠 Methodology")

tab1, tab2, tab3, tab4 = st.tabs([
    "🌲 Classification",
    "⚖️ Class Balancing",
    "⛏️ FP-Growth",
    "📊 Evaluation"
])

with tab1:
    st.markdown("### Supervised Machine Learning")

    st.markdown("""
    Three classification algorithms are used to predict the
    **Observed Learning Progress Track**:

    **1. Decision Tree**
    - Provides an interpretable baseline model.
    - Produces understandable decision paths.
    - Useful for explaining how learner characteristics influence predictions.

    **2. Random Forest**
    - Combines multiple decision trees.
    - Reduces dependence on a single tree.
    - Provides more stable predictions and feature importance.

    **3. Gradient Boosting**
    - Builds trees sequentially to correct previous prediction errors.
    - Captures more complex relationships between learner characteristics.
    - Provides the strongest performance among the evaluated models.
    """)

with tab2:
    st.markdown("### Handling Class Imbalance")

    st.markdown("""
    The original target distribution was imbalanced:

    - **Moderate Progress:** 867 records — 43.35%
    - **Fast Progress:** 859 records — 42.95%
    - **Slow Progress:** 274 records — 13.70%

    The **Slow Progress** class was substantially smaller than the other
    classes.

    To reduce this imbalance, **SMOTE (Synthetic Minority Over-sampling
    Technique)** was applied **only to the training data**.

    Before SMOTE:

    - Fast Progress: 687
    - Moderate Progress: 694
    - Slow Progress: 219

    After SMOTE:

    - Fast Progress: 694
    - Moderate Progress: 694
    - Slow Progress: 694

    This produced **2,082 balanced training samples** while keeping the
    original test set unchanged.
    """)

with tab3:
    st.markdown("### FP-Growth Association Rule Mining")

    st.markdown("""
    FP-Growth is used as a separate **unsupervised data-mining technique**.

    It discovers frequently occurring combinations of learner attributes
    and identifies association rules using:

    - **Support** — how frequently a pattern occurs.
    - **Confidence** — how reliably the consequent follows the antecedent.
    - **Lift** — how much stronger the relationship is compared with random
      co-occurrence.

    FP-Growth is particularly suitable because it discovers frequent patterns
    without generating the large candidate sets used by Apriori.

    The resulting rules can be used to identify common combinations of
    learning habits and support recommendation generation.
    """)

with tab4:
    st.markdown("### Model Evaluation")

    st.markdown("""
    The classification models are evaluated using:

    - **Accuracy**
    - **Weighted Precision**
    - **Weighted Recall**
    - **Weighted F1 Score**
    - **Classification Report**
    - **Confusion Matrix**
    - **Feature Importance**

    The dataset is divided into **80% training data and 20% testing data**
    using stratified sampling with a fixed random state.

    Model performance is compared rather than relying on a single algorithm.
    """)

st.divider()

# =========================================================
# PROBLEMS AND IMPROVEMENTS
# =========================================================
st.subheader("🔧 Problems Identified & Improvements")

problems = [
    (
        "1. Imbalanced Target Classes",
        "The Slow Progress class contained only 274 of the 2,000 records.",
        "SMOTE was applied only to the training set to balance the three target classes."
    ),
    (
        "2. Categorical Variables",
        "Several learner attributes were categorical rather than numerical.",
        "Categorical variables were converted into numerical representations using LabelEncoder, with the encoders saved for consistent prediction-time processing."
    ),
    (
        "3. Missing or Invalid Numerical Values",
        "Numerical fields could contain missing or non-numeric values.",
        "Numerical values were converted using numeric coercion and missing values were replaced using training-data medians."
    ),
    (
        "4. Single-Model Limitation",
        "Using only a Decision Tree would make it difficult to determine whether another algorithm could perform better.",
        "Decision Tree, Random Forest, and Gradient Boosting were trained and compared using the same evaluation framework."
    ),
    (
        "5. Model Over-Reliance on Majority Classes",
        "Without class balancing, predictions may favor the larger Fast and Moderate Progress classes.",
        "SMOTE was used during training to give the minority Slow Progress class greater representation."
    ),
    (
        "6. Model Evaluation Limitation",
        "Accuracy alone does not show how well each individual progress class is predicted.",
        "Precision, recall, F1 score, classification reports, and confusion matrices were added for more comprehensive evaluation."
    ),
    (
        "7. Prediction Consistency",
        "A deployed model must receive features in exactly the same format and order used during training.",
        "Feature lists, categorical encoders, and the target encoder were saved as separate model artifacts and reused during prediction."
    ),
    (
        "8. Limited Pattern Discovery",
        "Classification alone focuses on prediction and may not reveal combinations of learner behaviors.",
        "FP-Growth was added to discover frequent learning patterns and association rules."
    ),
]

for title, problem, improvement in problems:
    with st.expander(title):
        st.markdown(f"**Problem:** {problem}")
        st.markdown(f"**Improvement:** {improvement}")

st.divider()

# =========================================================
# MODEL PERFORMANCE
# =========================================================
st.subheader("📈 Final Model Performance")

st.markdown("""
The final evaluation shows that **Gradient Boosting achieved the best overall
classification performance** among the three evaluated models.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Decision Tree Accuracy",
        "50.50%"
    )

with col2:
    st.metric(
        "Random Forest Accuracy",
        "56.50%",
        "+6.00 pp vs DT"
    )

with col3:
    st.metric(
        "Gradient Boosting Accuracy",
        "57.00%",
        "+0.50 pp vs RF"
    )

st.markdown("""
### Model Comparison

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
|---|---:|---:|---:|---:|
| Decision Tree | 50.50% | 51.15% | 50.50% | 50.70% |
| Random Forest | 56.50% | 57.26% | 56.50% | 56.78% |
| **Gradient Boosting** | **57.00%** | **57.60%** | **57.00%** | **57.21%** |

Gradient Boosting is selected as the **best-performing classification model**
because it achieved the highest value across all four reported metrics.
""")

st.divider()

# =========================================================
# GRADIENT BOOSTING FINDINGS
# =========================================================
st.subheader("🔍 Key Findings from Gradient Boosting")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Classification Performance

    **Fast Progress**
    - Precision: 66%
    - Recall: 66%
    - F1 Score: 66%

    **Moderate Progress**
    - Precision: 55%
    - Recall: 51%
    - F1 Score: 53%

    **Slow Progress**
    - Precision: 38%
    - Recall: 47%
    - F1 Score: 42%
    """)

with col2:
    st.markdown("""
    ### Important Predictive Features

    The strongest supplied Gradient Boosting feature importances were:

    1. **Daily Study Time** — 33.24%
    2. **Weekly Practice Frequency** — 17.75%
    3. **English Learning Motivation** — 11.06%
    4. **Listening Confidence** — 7.63%
    5. **English Usage Frequency** — 4.30%

    These results indicate that study behavior and learner engagement are
    important predictors in the trained model.

    **Note:** Feature importance indicates model contribution, not causation.
    """)

st.divider()

# =========================================================
# IMPROVEMENT RESULT
# =========================================================
st.subheader("🚀 Improvement from Baseline")

st.markdown("""
A simple baseline of approximately **30% accuracy** was used as a reference
for the three-class classification problem.

The final Gradient Boosting model achieved **57.00% accuracy**.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Baseline", "30.00%")

with col2:
    st.metric("Gradient Boosting", "57.00%")

with col3:
    st.metric("Absolute Improvement", "+27.00 pp")

st.markdown("""
The relative improvement from the approximately 30% baseline is:

**(57% − 30%) / 30% × 100 = 90%**

Therefore, the final Gradient Boosting accuracy is approximately **90% higher
than the baseline accuracy in relative terms**.

This improvement should be interpreted as the result of the **overall
data-mining workflow**, including preprocessing, feature encoding, stratified
splitting, class balancing, and model selection/tuning—not as an effect of
SMOTE alone.
""")

st.divider()

# =========================================================
# WORKFLOW
# =========================================================
st.subheader("🗺️ Complete System Workflow")

workflow = [
    ("1️⃣", "Data Collection", "Collect English-learning survey responses."),
    ("2️⃣", "Data Cleaning", "Remove invalid records and handle missing values."),
    ("3️⃣", "Exploratory Analysis", "Analyze distributions, variables, and target classes."),
    ("4️⃣", "Feature Preparation", "Encode categorical variables and prepare numerical features."),
    ("5️⃣", "Train/Test Split", "Use an 80/20 stratified split."),
    ("6️⃣", "SMOTE", "Balance the training classes without modifying the test set."),
    ("7️⃣", "Model Training", "Train Decision Tree, Random Forest, and Gradient Boosting."),
    ("8️⃣", "Model Evaluation", "Compare accuracy, precision, recall, F1, and confusion matrices."),
    ("9️⃣", "FP-Growth", "Discover frequent learner patterns and association rules."),
    ("🔟", "Prediction", "Predict the learning progress track for a new learner profile."),
    ("1️⃣1️⃣", "Recommendations", "Use model results and association patterns to support learning recommendations.")
]

for icon, title, description in workflow:
    col1, col2, col3 = st.columns([0.7, 2.2, 7.1])

    with col1:
        st.markdown(f"### {icon}")

    with col2:
        st.markdown(f"**{title}**")

    with col3:
        st.markdown(description)

st.divider()

# =========================================================
# LIMITATIONS
# =========================================================
st.subheader("⚠️ Current Limitations")

st.markdown("""
The system has several limitations that should be acknowledged during
evaluation:

- The dataset contains **2,000 survey records**, so the results may not
  generalize to every English learner population.
- The **Slow Progress** class remains difficult for the final model to
  classify accurately, despite SMOTE.
- The classification accuracy of **57%** shows that learner progress is
  influenced by multiple factors that are not perfectly captured by the
  available variables.
- The encoded categorical representation is practical for the current
  implementation, but more advanced approaches such as **SMOTENC** could be
  investigated in future work.
- Feature importance identifies variables used strongly by the model but
  should not be interpreted as proof of causal relationships.
- Association rules describe co-occurrence and do not by themselves prove
  cause-and-effect relationships.
""")

st.divider()

# =========================================================
# FINAL SUMMARY
# =========================================================
st.subheader("✅ Overall Summary")

st.markdown("""
The improved system moves beyond a single Decision Tree model by combining
**three supervised classification algorithms with FP-Growth association rule
mining**.

The methodology was strengthened by addressing class imbalance, standardizing
feature preparation, using a fixed train/test evaluation strategy, comparing
multiple models, and adding detailed evaluation metrics.

Among the classification models, **Gradient Boosting achieved the best
performance with 57.00% accuracy, 57.60% weighted precision, 57.00% weighted
recall, and 57.21% weighted F1 score**.

FP-Growth complements the predictive models by identifying relationships among
learner characteristics and learning behaviors. Together, the techniques
provide both **predictive insight** and **pattern-based insight** for English
learning analysis.
""")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#64748b; font-size:0.85rem;">
        Use the sidebar to explore Dataset Analytics, Descriptive Analytics,
        Predictive Engine, Model Evaluation, and FP-Growth Mining Logs.
    </div>
    """,
    unsafe_allow_html=True
)