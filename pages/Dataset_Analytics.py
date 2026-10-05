import os

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Dataset Analytics - English Learning Data Mining Portal",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dataset Analytics")

st.caption(
    "Explore dataset structure, learner distributions, learning "
    "behaviour, hidden patterns, correlations, and the train-test split."
)


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# =========================================================
# DATASET
# =========================================================

DATASET_FILE = os.path.join(
    PROJECT_ROOT,
    "english_learning_dataset_2000.xlsx"
)

if not os.path.exists(DATASET_FILE):

    DATASET_FILE = os.path.join(
        PROJECT_ROOT,
        "data",
        "english_learning_dataset_2000.xlsx"
    )


@st.cache_data
def load_dataset(path):

    if not os.path.exists(path):
        return None

    return pd.read_excel(path)


df = load_dataset(DATASET_FILE)


if df is None:

    st.error(
        "Dataset file `english_learning_dataset_2000.xlsx` "
        "was not found in the project folder."
    )

    st.stop()


# =========================================================
# COLUMN DEFINITIONS
# =========================================================

TARGET = "Observed Learning Progress Track"

STUDY = "Daily Study Time (Minutes)"

EXPOSURE = "Daily English Exposure"

SPEAKING = "Speaking Fluency & Oral Expression Confidence"


SKILLS = [
    "Vocabulary Mastery Confidence",
    "Grammar & Sentence Structure Confidence",
    "Listening Comprehension Confidence",
    "Speaking Fluency & Oral Expression Confidence",
    "Reading Confidence",
    "Writing Confidence"
]


available_skills = [
    c for c in SKILLS
    if c in df.columns
]


# =========================================================
# HELPER
# =========================================================

def numeric_column(data, column):

    return pd.to_numeric(
        data[column],
        errors="coerce"
    )


def readable_skill_name(name):

    return (
        name
        .replace(" Confidence", "")
        .replace(" & Oral Expression", "")
        .replace(" & Sentence Structure", "")
    )


# =========================================================
# 1. DATASET OVERVIEW
# =========================================================

st.divider()

st.subheader("1. Dataset Overview")


duplicates = int(
    df.duplicated().sum()
)

missing_cells = int(
    df.isna().sum().sum()
)


c1, c2, c3, c4, c5 = st.columns(5)


c1.metric(
    "Total Learners",
    f"{len(df):,}"
)

c2.metric(
    "Variables",
    df.shape[1]
)

c3.metric(
    "Missing Cells",
    f"{missing_cells:,}"
)

c4.metric(
    "Duplicate Rows",
    f"{duplicates:,}"
)

if TARGET in df.columns:

    c5.metric(
        "Progress Classes",
        df[TARGET].nunique()
    )

else:

    c5.metric(
        "Progress Classes",
        "N/A"
    )


# =========================================================
# DATA QUALITY
# =========================================================

with st.expander("🔎 Data Quality & Structure"):

    quality_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [
            str(dtype)
            for dtype in df.dtypes
        ],
        "Missing Values": [
            int(df[c].isna().sum())
            for c in df.columns
        ],
        "Unique Values": [
            int(df[c].nunique())
            for c in df.columns
        ]
    })

    st.dataframe(
        quality_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 2. LEARNER DISTRIBUTIONS
# =========================================================

st.subheader("2. Learner Distributions")


tab1, tab2, tab3 = st.tabs([
    "Demographics",
    "Learning Characteristics",
    "Goals & Activities"
])


# =========================================================
# DEMOGRAPHICS
# =========================================================

with tab1:

    col1, col2 = st.columns(2)


    with col1:

        if "Age" in df.columns:

            age_data = numeric_column(
                df,
                "Age"
            ).dropna()

            fig = px.histogram(
                age_data,
                x="Age",
                nbins=20,
                title="Age Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    with col2:

        if "Current Language Level" in df.columns:

            counts = (
                df["Current Language Level"]
                .value_counts()
                .reset_index()
            )

            counts.columns = [
                "Language Level",
                "Learners"
            ]

            fig = px.pie(
                counts,
                names="Language Level",
                values="Learners",
                hole=0.4,
                title="Current Language Level Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    if "Primary Role" in df.columns:

        counts = (
            df["Primary Role"]
            .value_counts()
            .reset_index()
        )

        counts.columns = [
            "Primary Role",
            "Learners"
        ]

        fig = px.bar(
            counts,
            x="Primary Role",
            y="Learners",
            title="Learner Roles"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# LEARNING CHARACTERISTICS
# =========================================================

with tab2:

    col1, col2 = st.columns(2)


    with col1:

        if STUDY in df.columns:

            fig = px.histogram(
                df,
                x=STUDY,
                nbins=20,
                title="Daily Study Time Distribution"
            )

            fig.update_xaxes(
                title="Minutes per Day"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    with col2:

        if "Years Learning English" in df.columns:

            fig = px.histogram(
                df,
                x="Years Learning English",
                nbins=20,
                title="Years Learning English"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    col1, col2 = st.columns(2)


    with col1:

        if EXPOSURE in df.columns:

            counts = (
                df[EXPOSURE]
                .value_counts()
                .reset_index()
            )

            counts.columns = [
                "English Exposure",
                "Learners"
            ]

            fig = px.bar(
                counts,
                x="English Exposure",
                y="Learners",
                title="Daily English Exposure"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    with col2:

        if "Weekly Practice Frequency" in df.columns:

            counts = (
                df["Weekly Practice Frequency"]
                .value_counts()
                .reset_index()
            )

            counts.columns = [
                "Practice Frequency",
                "Learners"
            ]

            fig = px.bar(
                counts,
                x="Practice Frequency",
                y="Learners",
                title="Weekly Practice Frequency"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =========================================================
# GOALS & ACTIVITIES
# =========================================================

with tab3:

    col1, col2 = st.columns(2)


    with col1:

        if "Learning Goal" in df.columns:

            counts = (
                df["Learning Goal"]
                .value_counts()
                .reset_index()
            )

            counts.columns = [
                "Learning Goal",
                "Learners"
            ]

            fig = px.bar(
                counts,
                x="Learning Goal",
                y="Learners",
                title="Learning Goals"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    with col2:

        if "Preferred Practice Activity" in df.columns:

            counts = (
                df["Preferred Practice Activity"]
                .value_counts()
                .reset_index()
            )

            counts.columns = [
                "Practice Activity",
                "Learners"
            ]

            fig = px.pie(
                counts,
                names="Practice Activity",
                values="Learners",
                hole=0.4,
                title="Preferred Practice Activities"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =========================================================
# 3. SKILL CONFIDENCE
# =========================================================

st.divider()

st.subheader("3. Skill Confidence Distribution")


if available_skills:

    numeric_skills = (
        df[available_skills]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


    # -----------------------------------------------------
    # AVERAGE
    # -----------------------------------------------------

    avg_scores = (
        numeric_skills
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    avg_scores.columns = [
        "Skill",
        "Average Rating"
    ]

    avg_scores["Skill"] = (
        avg_scores["Skill"]
        .apply(readable_skill_name)
    )


    fig = px.bar(
        avg_scores,
        x="Skill",
        y="Average Rating",
        title="Average Confidence by Skill"
    )

    fig.update_yaxes(
        range=[1, 5]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # -----------------------------------------------------
    # DISTRIBUTION
    # -----------------------------------------------------

    skill_choice = st.selectbox(
        "Select a skill to inspect its distribution:",
        available_skills
    )

    selected_values = numeric_skills[
        skill_choice
    ].dropna()


    fig = px.histogram(
        selected_values,
        x=skill_choice,
        nbins=5,
        title=f"{readable_skill_name(skill_choice)} Rating Distribution"
    )

    fig.update_xaxes(
        dtick=1
    )

    fig.update_yaxes(
        title="Number of Learners"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# 4. LEARNING BEHAVIOUR RELATIONSHIPS
# =========================================================

st.divider()

st.subheader("4. Learning Behaviour Relationships")


if EXPOSURE in df.columns and SPEAKING in df.columns:

    temp = df[
        [EXPOSURE, SPEAKING]
    ].copy()

    temp[SPEAKING] = pd.to_numeric(
        temp[SPEAKING],
        errors="coerce"
    )


    exposure_avg = (
        temp
        .dropna()
        .groupby(EXPOSURE)[SPEAKING]
        .mean()
        .reset_index()
    )


    fig = px.bar(
        exposure_avg,
        x=EXPOSURE,
        y=SPEAKING,
        title="Average Speaking Confidence by English Exposure"
    )

    fig.update_yaxes(
        range=[1, 5]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.info(
        "This visualization describes an association in the dataset. "
        "It does not establish that English exposure causes higher "
        "speaking confidence."
    )


# =========================================================
# 5. CORRELATION
# =========================================================

st.divider()

st.subheader("5. Skill Correlation Matrix")


if len(available_skills) >= 2:

    numeric_skills = (
        df[available_skills]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )

    corr = numeric_skills.corr()


    labels = [
        readable_skill_name(c)
        for c in available_skills
    ]

    corr.columns = labels
    corr.index = labels


    fig = px.imshow(
        corr,
        text_auto=".2f",
        title="Relationships Between Skill Confidence Areas",
        zmin=-1,
        zmax=1,
        aspect="auto"
    )

    fig.update_layout(
        height=550
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    with st.expander("📖 Correlation Interpretation"):

        st.markdown("""
        **Correlation ranges from -1 to +1.**

        - **+0.70 to +1.00** → Strong positive relationship
        - **+0.40 to +0.69** → Moderate positive relationship
        - **-0.39 to +0.39** → Weak relationship
        - **-0.40 to -0.69** → Moderate negative relationship
        - **-0.70 to -1.00** → Strong negative relationship

        Correlation describes how variables move together.
        It does not establish causation.
        """)


# =========================================================
# 6. HIDDEN PATTERN DISCOVERY
# =========================================================

st.divider()

st.subheader("6. Hidden Learning Pattern Discovery")

st.caption(
    "Patterns are identified using interpretable combinations of "
    "learner characteristics. These are descriptive rules, not "
    "machine-learning predictions."
)


pattern_results = []


# ---------------------------------------------------------
# PATTERN 1
# ---------------------------------------------------------

if STUDY in df.columns and SPEAKING in df.columns:

    study_values = numeric_column(
        df,
        STUDY
    )

    speaking_values = numeric_column(
        df,
        SPEAKING
    )


    mask = (
        (study_values >= 60) &
        (speaking_values <= 2)
    )

    count = int(mask.sum())

    pattern_results.append({
        "Pattern": "High Study / Low Speaking",
        "Learners": count,
        "Percentage": count / len(df) * 100,
        "Definition": "Study ≥60 min/day and speaking confidence ≤2"
    })


# ---------------------------------------------------------
# PATTERN 2
# ---------------------------------------------------------

if (
    STUDY in df.columns
    and SPEAKING in df.columns
):

    study_values = numeric_column(
        df,
        STUDY
    )

    speaking_values = numeric_column(
        df,
        SPEAKING
    )

    mask = (
        (study_values <= 30) &
        (speaking_values >= 4)
    )

    count = int(mask.sum())

    pattern_results.append({
        "Pattern": "Exposure Advantage",
        "Learners": count,
        "Percentage": count / len(df) * 100,
        "Definition": "Study ≤30 min/day and speaking confidence ≥4"
    })


# ---------------------------------------------------------
# PATTERN 3
# ---------------------------------------------------------

if (
    "Listening Comprehension Confidence" in df.columns
    and SPEAKING in df.columns
):

    listening_values = numeric_column(
        df,
        "Listening Comprehension Confidence"
    )

    speaking_values = numeric_column(
        df,
        SPEAKING
    )

    mask = (
        (listening_values >= 4) &
        (speaking_values >= 4)
    )

    count = int(mask.sum())

    pattern_results.append({
        "Pattern": "Strong Communication Pair",
        "Learners": count,
        "Percentage": count / len(df) * 100,
        "Definition": "Listening ≥4 and speaking ≥4"
    })


# ---------------------------------------------------------
# PATTERN 4
# ---------------------------------------------------------

if (
    "Grammar & Sentence Structure Confidence" in df.columns
    and "Reading Confidence" in df.columns
    and SPEAKING in df.columns
):

    grammar_values = numeric_column(
        df,
        "Grammar & Sentence Structure Confidence"
    )

    reading_values = numeric_column(
        df,
        "Reading Confidence"
    )

    speaking_values = numeric_column(
        df,
        SPEAKING
    )

    mask = (
        (grammar_values >= 4) &
        (reading_values >= 4) &
        (speaking_values <= 3)
    )

    count = int(mask.sum())

    pattern_results.append({
        "Pattern": "Knowledge–Communication Gap",
        "Learners": count,
        "Percentage": count / len(df) * 100,
        "Definition": "Grammar ≥4, reading ≥4 and speaking ≤3"
    })


# ---------------------------------------------------------
# PATTERN 5
# ---------------------------------------------------------

if (
    "Vocabulary Mastery Confidence" in df.columns
    and "Writing Confidence" in df.columns
):

    vocabulary_values = numeric_column(
        df,
        "Vocabulary Mastery Confidence"
    )

    writing_values = numeric_column(
        df,
        "Writing Confidence"
    )

    mask = (
        (vocabulary_values >= 4) &
        (writing_values >= 4)
    )

    count = int(mask.sum())

    pattern_results.append({
        "Pattern": "Language Production Strength",
        "Learners": count,
        "Percentage": count / len(df) * 100,
        "Definition": "Vocabulary ≥4 and writing ≥4"
    })


# ---------------------------------------------------------
# DISPLAY PATTERNS
# ---------------------------------------------------------

if pattern_results:

    pattern_df = pd.DataFrame(
        pattern_results
    )

    display_df = pattern_df[
        [
            "Pattern",
            "Learners",
            "Percentage",
            "Definition"
        ]
    ].copy()

    display_df["Percentage"] = (
        display_df["Percentage"]
        .round(1)
    )

    st.dataframe(
        display_df.style.format({
            "Percentage": "{:.1f}%"
        }),
        use_container_width=True,
        hide_index=True
    )


    fig = px.bar(
        pattern_df,
        x="Pattern",
        y="Learners",
        title="Detected Hidden Learning Patterns"
    )

    fig.update_xaxes(
        tickangle=-25
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.info(
        "These patterns describe groups of learners with similar "
        "characteristics. They should not be interpreted as causal "
        "relationships or formal statistical diagnoses."
    )


# =========================================================
# 7. TRAIN / TEST SPLIT
# =========================================================

st.divider()

st.subheader("7. Train–Test Split")


st.write(
    "The dataset is divided into training and testing subsets using "
    "an 80:20 split. Stratification is used so that the distribution "
    "of learning-progress classes is preserved as closely as possible "
    "between the two subsets."
)


if TARGET in df.columns:

    # -----------------------------------------------------
    # PREPARE TARGET
    # -----------------------------------------------------

    split_data = df.dropna(
        subset=[TARGET]
    ).copy()

    X = split_data.drop(
        columns=[TARGET]
    )

    y = split_data[TARGET].astype(str)


    # -----------------------------------------------------
    # SAME SPLIT SETTINGS AS TRAINING
    # -----------------------------------------------------

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )


        train_size = len(X_train)
        test_size = len(X_test)


        # -------------------------------------------------
        # METRICS
        # -------------------------------------------------

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Training Set",
            f"{train_size:,} learners"
        )

        c2.metric(
            "Testing Set",
            f"{test_size:,} learners"
        )

        c3.metric(
            "Split Ratio",
            "80% / 20%"
        )


        # -------------------------------------------------
        # VISUALIZE SPLIT
        # -------------------------------------------------

        split_df = pd.DataFrame({
            "Dataset": [
                "Training Set",
                "Testing Set"
            ],
            "Learners": [
                train_size,
                test_size
            ]
        })


        fig = px.pie(
            split_df,
            names="Dataset",
            values="Learners",
            hole=0.45,
            title="Training vs Testing Dataset"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # -------------------------------------------------
        # CLASS DISTRIBUTION
        # -------------------------------------------------

        train_distribution = (
            y_train
            .value_counts(normalize=True)
            .mul(100)
            .reset_index()
        )

        train_distribution.columns = [
            "Progress Track",
            "Percentage"
        ]

        train_distribution["Dataset"] = "Training"


        test_distribution = (
            y_test
            .value_counts(normalize=True)
            .mul(100)
            .reset_index()
        )

        test_distribution.columns = [
            "Progress Track",
            "Percentage"
        ]

        test_distribution["Dataset"] = "Testing"


        distribution_df = pd.concat(
            [
                train_distribution,
                test_distribution
            ],
            ignore_index=True
        )


        fig = px.bar(
            distribution_df,
            x="Progress Track",
            y="Percentage",
            color="Dataset",
            barmode="group",
            title="Progress-Track Distribution: Training vs Testing"
        )

        fig.update_yaxes(
            title="Percentage of Learners",
            ticksuffix="%"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        with st.expander("📖 Why use Train–Test Split?"):

            st.markdown("""
            ### Training Set

            The training set is used by the machine-learning algorithms
            to learn relationships between learner characteristics and
            the observed learning-progress track.

            ### Testing Set

            The testing set is kept separate from model training and is
            used to estimate how well the trained model performs on
            previously unseen data.

            ### Why 80:20?

            An 80:20 split provides a relatively large portion of the
            dataset for learning while retaining an independent subset
            for evaluation.

            ### Why Stratification?

            Stratification helps maintain a similar distribution of the
            target classes in both the training and testing subsets.

            ### Important

            SMOTE is **not applied to the testing set**.

            In the project's training methodology, SMOTE is applied
            only to the training data after the train-test split.
            This prevents synthetic training information from entering
            the independent test set.
            """)


    except ValueError as e:

        st.warning(
            "The train-test split could not be generated: "
            + str(e)
        )


else:

    st.warning(
        f"Target column `{TARGET}` was not found."
    )


# =========================================================
# 8. PCA LEARNER PATTERN MAP
# =========================================================

st.divider()

st.subheader("8. PCA Learner Pattern Map")


if len(available_skills) >= 2:

    pca_data = (
        df[available_skills]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


    # -----------------------------------------------------
    # MEDIAN IMPUTATION
    # -----------------------------------------------------

    pca_data = pca_data.fillna(
        pca_data.median()
    )


    # -----------------------------------------------------
    # STANDARDIZATION
    # -----------------------------------------------------

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        pca_data
    )


    # -----------------------------------------------------
    # PCA
    # -----------------------------------------------------

    pca = PCA(
        n_components=2,
        random_state=42
    )

    components = pca.fit_transform(
        scaled_data
    )


    pca_df = pd.DataFrame(
        components,
        columns=[
            "PC1",
            "PC2"
        ]
    )


    if TARGET in df.columns:

        pca_df["Progress Track"] = (
            df[TARGET]
            .astype(str)
        )

    else:

        pca_df["Progress Track"] = "Unknown"


    if STUDY in df.columns:

        pca_df["Study Time"] = numeric_column(
            df,
            STUDY
        )


    if SPEAKING in df.columns:

        pca_df["Speaking Confidence"] = numeric_column(
            df,
            SPEAKING
        )


    # -----------------------------------------------------
    # HIGH STUDY / LOW SPEAKING PATTERN
    # -----------------------------------------------------

    pca_df["Study Pattern"] = "Typical Pattern"


    if (
        STUDY in pca_df.columns
        and SPEAKING in pca_df.columns
    ):

        mask = (
            (pca_df["Study Time"] >= 60) &
            (pca_df["Speaking Confidence"] <= 2)
        )

        pca_df.loc[
            mask,
            "Study Pattern"
        ] = "High Study / Low Speaking"


    # -----------------------------------------------------
    # SCATTER
    # -----------------------------------------------------

    hover_columns = [
        c for c in [
            "Study Time",
            "Speaking Confidence"
        ]
        if c in pca_df.columns
    ]


    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Progress Track",
        symbol="Study Pattern",
        hover_data=hover_columns,
        title="PCA Learner Pattern Map",
        opacity=0.75
    )


    fig.update_layout(
        height=550
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    explained = (
        pca.explained_variance_ratio_
    )


    st.caption(
        f"PC1 explains {explained[0] * 100:.1f}% of variance; "
        f"PC2 explains {explained[1] * 100:.1f}%."
    )


    with st.expander("📖 How to interpret PCA"):

        st.markdown("""
        **Principal Component Analysis (PCA)** reduces several
        correlated skill-confidence variables into fewer dimensions
        so that learner profiles can be visualized.

        Learners positioned close together have more similar
        confidence profiles, while learners farther apart have
        more different profiles.

        PCA is used here for visualization and exploratory analysis.
        It is not the classification algorithm used by the predictive
        models.
        """)


# =========================================================
# 9. RAW DATA
# =========================================================

st.divider()

with st.expander("📄 View Raw Dataset Sample"):

    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "English Learning Data Mining Portal • Dataset Analytics"
)