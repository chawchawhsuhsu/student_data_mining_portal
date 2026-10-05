import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils.data_loader import load_data

st.set_page_config(
    page_title="Descriptive Analytics",
    page_icon="📊",
    layout="wide"
)

TARGET = "Observed Learning Progress Track"

SKILLS = [
    "Vocabulary Mastery Confidence",
    "Grammar & Sentence Structure Confidence",
    "Listening Comprehension Confidence",
    "Speaking Fluency & Oral Expression Confidence",
    "Reading Confidence",
    "Writing Confidence"
]

CLUSTER_FEATURES = [
    "Daily Study Time (Minutes)",
    "Years Learning English",
    *SKILLS
]


@st.cache_data
def get_data():
    return load_data()


df = get_data()

if df is None or df.empty:
    st.error("Dataset could not be loaded.")
    st.stop()

st.title("📊 Descriptive Analytics & Learner Profiling")
st.caption(
    "Explore learner characteristics, discover hidden patterns, "
    "and identify groups of similar learners."
)

# =========================================================
# 1. DATASET OVERVIEW
# =========================================================
st.header("1. 📌 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Learners", f"{len(df):,}")
c2.metric("Variables", df.shape[1])
c3.metric("Missing Cells", int(df.isna().sum().sum()))
c4.metric(
    "Progress Tracks",
    df[TARGET].nunique() if TARGET in df.columns else "N/A"
)

st.write(
    "Descriptive analysis summarizes the characteristics of the learners "
    "and explores relationships and groups without predicting an individual learner."
)

# =========================================================
# 2. LEARNER CHARACTERISTICS
# =========================================================
st.header("2. 📈 Learner Characteristics")

tab1, tab2, tab3, tab4 = st.tabs([
    "Skill Confidence",
    "Progress Tracks",
    "Learning Approaches",
    "Study Habits"
])

with tab1:
    available = [c for c in SKILLS if c in df.columns]

    if available:
        avg = (
            df[available]
            .apply(pd.to_numeric, errors="coerce")
            .mean()
            .sort_values()
            .reset_index()
        )
        avg.columns = ["Skill", "Average Confidence"]

        fig = px.bar(
            avg,
            x="Average Confidence",
            y="Skill",
            orientation="h",
            title="Average Skill Confidence"
        )
        fig.update_xaxes(range=[0, 5])
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    if TARGET in df.columns:
        counts = df[TARGET].value_counts().reset_index()
        counts.columns = ["Progress Track", "Learners"]

        fig = px.bar(
            counts,
            x="Progress Track",
            y="Learners",
            text="Learners",
            title="Observed Learning Progress Tracks"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    col = "Primary Learning Approach"

    if col in df.columns:
        counts = df[col].value_counts().reset_index()
        counts.columns = ["Learning Approach", "Learners"]

        fig = px.pie(
            counts,
            names="Learning Approach",
            values="Learners",
            hole=0.4,
            title="Preferred Learning Approaches"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    study_col = "Daily Study Time (Minutes)"

    if study_col in df.columns:
        study = pd.to_numeric(df[study_col], errors="coerce").dropna()

        fig = px.histogram(
            study,
            x=study,
            nbins=20,
            title="Daily Study Time Distribution"
        )
        fig.update_xaxes(title="Daily Study Time (Minutes)")
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 3. HIDDEN LEARNING PATTERNS
# =========================================================
st.header("3. 🔎 Hidden Learning Patterns")
st.caption(
    "These patterns show combinations of characteristics found in the dataset. "
    "They describe co-occurrence and do not prove cause and effect."
)

numeric = df.copy()

for col in SKILLS + ["Daily Study Time (Minutes)", "Years Learning English"]:
    if col in numeric.columns:
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")


def add_pattern(patterns, name, condition, explanation):
    condition = condition.fillna(False)

    if condition.any():
        count = int(condition.sum())
        percentage = count / len(df) * 100
        patterns.append(
            (
                name,
                f"{count:,} learners ({percentage:.1f}%) {explanation}"
            )
        )


patterns = []

if all(c in numeric.columns for c in [
    "Daily Study Time (Minutes)",
    "Speaking Fluency & Oral Expression Confidence"
]):
    add_pattern(
        patterns,
        "Exposure / Communication Pattern",
        (numeric["Daily Study Time (Minutes)"] <= 30)
        & (numeric["Speaking Fluency & Oral Expression Confidence"] >= 4),
        "report 30 minutes or less of daily study time while reporting high speaking confidence."
    )

    add_pattern(
        patterns,
        "High Study / Low Speaking",
        (numeric["Daily Study Time (Minutes)"] >= 60)
        & (numeric["Speaking Fluency & Oral Expression Confidence"] <= 2),
        "report at least 60 minutes of daily study but low speaking confidence."
    )

if all(c in numeric.columns for c in [
    "Listening Comprehension Confidence",
    "Speaking Fluency & Oral Expression Confidence"
]):
    add_pattern(
        patterns,
        "Strong Communication Pair",
        (numeric["Listening Comprehension Confidence"] >= 4)
        & (numeric["Speaking Fluency & Oral Expression Confidence"] >= 4),
        "report high listening and speaking confidence."
    )

if all(c in numeric.columns for c in [
    "Grammar & Sentence Structure Confidence",
    "Reading Confidence",
    "Speaking Fluency & Oral Expression Confidence"
]):
    add_pattern(
        patterns,
        "Knowledge–Communication Gap",
        (numeric["Grammar & Sentence Structure Confidence"] >= 4)
        & (numeric["Reading Confidence"] >= 4)
        & (numeric["Speaking Fluency & Oral Expression Confidence"] <= 3),
        "show strong grammar and reading confidence but lower speaking confidence."
    )

if all(c in numeric.columns for c in [
    "Vocabulary Mastery Confidence",
    "Writing Confidence"
]):
    add_pattern(
        patterns,
        "Language Production Strength",
        (numeric["Vocabulary Mastery Confidence"] >= 4)
        & (numeric["Writing Confidence"] >= 4),
        "report high vocabulary and writing confidence."
    )

if patterns:
    for name, explanation in patterns:
        with st.container(border=True):
            st.subheader(f"🔹 {name}")
            st.write(explanation)
else:
    st.info("No strong predefined descriptive pattern was detected.")

# =========================================================
# 4. K-MEANS CLUSTERING
# =========================================================
st.header("4. 👥 Learner Clustering")

st.write(
    "K-Means groups learners with similar numerical learning characteristics. "
    "The clusters are used to describe learner segments, not to predict progress."
)

available_features = [c for c in CLUSTER_FEATURES if c in df.columns]

if len(available_features) >= 2:
    cluster_data = df[available_features].copy()

    for col in available_features:
        cluster_data[col] = pd.to_numeric(cluster_data[col], errors="coerce")

    clean_cluster = cluster_data.dropna()

    if len(clean_cluster) >= 3:
        max_k = min(6, len(clean_cluster) - 1)

        k = st.slider(
            "Number of learner clusters (K)",
            min_value=2,
            max_value=max_k,
            value=min(3, max_k)
        )

        scaler = StandardScaler()
        X = scaler.fit_transform(clean_cluster)

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        labels = model.fit_predict(X)

        clustered = clean_cluster.copy()
        clustered["Cluster"] = labels + 1

        # Cluster size
        sizes = (
            clustered["Cluster"]
            .value_counts()
            .sort_index()
            .reset_index()
        )
        sizes.columns = ["Cluster", "Learners"]
        sizes["Percentage"] = sizes["Learners"] / len(clustered) * 100

        fig = px.bar(
            sizes,
            x="Cluster",
            y="Learners",
            text="Learners",
            title="Learners in Each Cluster"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cluster profile
        profile = (
            clustered
            .groupby("Cluster")[available_features]
            .mean()
            .round(2)
        )

        st.subheader("Cluster Profiles")
        st.write(
            "The averages show the typical characteristics of learners in each group."
        )
        st.dataframe(profile, use_container_width=True)

        # Progress composition
        if TARGET in df.columns:
            target_values = df.loc[clean_cluster.index, TARGET]

            cluster_target = pd.DataFrame({
                "Cluster": labels + 1,
                TARGET: target_values.values
            })

            composition = (
                pd.crosstab(
                    cluster_target["Cluster"],
                    cluster_target[TARGET],
                    normalize="index"
                )
                .mul(100)
                .round(1)
            )

            st.subheader("Progress Track Composition by Cluster")
            st.dataframe(composition, use_container_width=True)

            composition_plot = (
                composition
                .reset_index()
                .melt(
                    id_vars="Cluster",
                    var_name="Progress Track",
                    value_name="Percentage"
                )
            )

            fig = px.bar(
                composition_plot,
                x="Cluster",
                y="Percentage",
                color="Progress Track",
                barmode="group",
                title="Observed Progress Track Composition Within Clusters"
            )
            fig.update_yaxes(title="Percentage (%)")
            st.plotly_chart(fig, use_container_width=True)

        # Two-dimensional visualization
        if len(available_features) >= 2:
            plot_df = pd.DataFrame({
                "Feature 1": X[:, 0],
                "Feature 2": X[:, 1],
                "Cluster": (labels + 1).astype(str)
            })

            fig = px.scatter(
                plot_df,
                x="Feature 1",
                y="Feature 2",
                color="Cluster",
                title=(
                    f"Cluster Visualization: "
                    f"{available_features[0]} vs {available_features[1]}"
                )
            )
            fig.update_xaxes(title=available_features[0] + " (standardized)")
            fig.update_yaxes(title=available_features[1] + " (standardized)")
            st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Cluster numbers are labels only. Cluster 1 is not automatically "
            "better or worse than Cluster 2. Interpret clusters using their profiles."
        )
    else:
        st.warning("Not enough complete numeric records are available for clustering.")
else:
    st.warning("The required clustering features are not available in the dataset.")

# =========================================================
# 5. SKILL RELATIONSHIPS
# =========================================================
st.header("5. 🔗 Skill Relationships")

available_skills = [c for c in SKILLS if c in df.columns]

if len(available_skills) >= 2:
    skill_numeric = (
        df[available_skills]
        .apply(pd.to_numeric, errors="coerce")
    )

    corr = skill_numeric.corr().round(2)

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Between Skill Confidence Measures"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Correlation describes how two variables move together. "
        "It does not establish a causal relationship."
    )

# =========================================================
# 6. INDIVIDUAL LEARNER PROFILE
# =========================================================
st.header("6. 🧑‍🎓 Individual Learner Profile")

st.write(
    "Enter a learner profile to describe its strengths and compare it "
    "with the dataset. This section does not use a machine-learning classifier."
)

with st.form("learner_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 10, 100, 22)

        language_level = st.selectbox(
            "Current Language Level",
            sorted(df["Current Language Level"].dropna().astype(str).unique())
        )

        role = st.selectbox(
            "Primary Role",
            sorted(df["Primary Role"].dropna().astype(str).unique())
        )

        approach = st.selectbox(
            "Primary Learning Approach",
            sorted(df["Primary Learning Approach"].dropna().astype(str).unique())
        )

    with col2:
        study_time = st.slider(
            "Daily Study Time (Minutes)",
            0, 180, 30, 5
        )

        exposure = st.selectbox(
            "Daily English Exposure",
            sorted(df["Daily English Exposure"].dropna().astype(str).unique())
        )

        years_learning = st.number_input(
            "Years Learning English",
            0, 50, 3
        )

        practice_frequency = st.selectbox(
            "Weekly Practice Frequency",
            sorted(df["Weekly Practice Frequency"].dropna().astype(str).unique())
        )

    with col3:
        motivation = st.selectbox(
            "English Learning Motivation",
            sorted(df["English Learning Motivation"].dropna().astype(str).unique())
        )

        usage_frequency = st.selectbox(
            "English Usage Frequency",
            sorted(df["English Usage Frequency"].dropna().astype(str).unique())
        )

        learning_goal = st.selectbox(
            "Learning Goal",
            sorted(df["Learning Goal"].dropna().astype(str).unique())
        )

        practice_activity = st.selectbox(
            "Preferred Practice Activity",
            sorted(df["Preferred Practice Activity"].dropna().astype(str).unique())
        )

    st.subheader("🎯 Skill Confidence")

    s1, s2, s3 = st.columns(3)

    with s1:
        vocabulary = st.slider("Vocabulary", 1, 5, 3)
        grammar = st.slider("Grammar", 1, 5, 3)

    with s2:
        listening = st.slider("Listening", 1, 5, 3)
        speaking = st.slider("Speaking", 1, 5, 3)

    with s3:
        reading = st.slider("Reading", 1, 5, 3)
        writing = st.slider("Writing", 1, 5, 3)

    submitted = st.form_submit_button(
        "🔍 Analyze My Profile",
        use_container_width=True
    )

if submitted:
    skill_values = {
        "Vocabulary": vocabulary,
        "Grammar": grammar,
        "Listening": listening,
        "Speaking": speaking,
        "Reading": reading,
        "Writing": writing
    }

    strongest = max(skill_values, key=skill_values.get)
    weakest = min(skill_values, key=skill_values.get)
    average_skill = float(np.mean(list(skill_values.values())))

    if average_skill >= 4:
        learner_type = "High-Confidence Learner"
    elif average_skill >= 3:
        learner_type = "Developing Learner"
    else:
        learner_type = "Foundation-Building Learner"

    if speaking >= 4 and listening >= 4:
        learner_type = "Communication-Oriented Learner"
    elif reading >= 4 and grammar >= 4 and speaking <= 3:
        learner_type = "Knowledge-Strong but Communication-Developing Learner"
    elif study_time >= 60 and speaking <= 2:
        learner_type = "High-Study / Low-Speaking Learner"

    st.subheader("🧠 Learner Profile")
    st.success(f"### {learner_type}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Confidence", f"{average_skill:.1f}/5")
    c2.metric("Strongest Skill", strongest)
    c3.metric("Development Area", weakest)
    c4.metric("Study Time", f"{study_time} min")

    radar = go.Figure()
    radar.add_trace(
        go.Scatterpolar(
            r=list(skill_values.values()),
            theta=list(skill_values.keys()),
            fill="toself",
            name="Your Profile"
        )
    )

    radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[1, 5]
            )
        ),
        height=450
    )

    st.plotly_chart(radar, use_container_width=True)

    comparison = []

    for skill, value in skill_values.items():
        original_col = next(
            (c for c in SKILLS if skill.lower() in c.lower()),
            None
        )

        if original_col in df.columns:
            values = pd.to_numeric(
                df[original_col],
                errors="coerce"
            ).dropna()

            if len(values):
                comparison.append({
                    "Skill": skill,
                    "Your Rating": value,
                    "Dataset Percentile": (values <= value).mean() * 100,
                    "Dataset Average": values.mean()
                })

    if comparison:
        comp = pd.DataFrame(comparison)

        st.subheader("📊 Your Profile vs Dataset")

        st.dataframe(
            comp.style.format({
                "Dataset Percentile": "{:.1f}%",
                "Dataset Average": "{:.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            comp,
            x="Skill",
            y=["Your Rating", "Dataset Average"],
            barmode="group",
            title="Your Skill Confidence vs Dataset Average"
        )

        fig.update_yaxes(range=[0, 5])
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TECHNICAL DETAILS
# =========================================================
with st.expander("🔬 Technical Details"):
    st.markdown("""
### Descriptive Analysis

This page uses descriptive statistics, distributions, rule-based
pattern detection, correlation analysis, learner profiling, and
K-Means clustering.

### K-Means Clustering

K-Means groups learners according to similarity in selected numerical
learning characteristics. StandardScaler is applied before clustering
so variables with different scales contribute more fairly.

The clusters are exploratory and descriptive. They do not predict the
learning-progress target.

### Hidden Patterns

The hidden-pattern section identifies predefined combinations of
characteristics that occur together in the dataset. These are descriptive
findings and do not establish causal relationships.

### Separation from Predictive Analysis

Decision Tree, Random Forest, Gradient Boosting, model comparison,
accuracy, precision, recall, F1 score, confusion matrix, feature
importance, and final model selection belong to the Predictive /
Model Evaluation component.
""")
