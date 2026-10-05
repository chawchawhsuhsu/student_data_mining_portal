import streamlit as st


# =========================================================
# SKILL COLUMNS
# =========================================================

SKILL_COLUMNS = [
    "Vocabulary Mastery Confidence",
    "Grammar & Sentence Structure Confidence",
    "Listening Comprehension Confidence",
    "Speaking Fluency & Oral Expression Confidence",
    "Reading Confidence",
    "Writing Confidence"
]


# =========================================================
# CATEGORY HELPER
# =========================================================

def get_categories(
    df,
    column
):

    if column not in df.columns:
        return []

    return sorted(
        df[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


# =========================================================
# CREATE LEARNER PROFILE
# =========================================================

def create_learner_profile(df):

    st.subheader(
        "👤 Build Your Learner Profile"
    )

    st.write(
        "Enter your current English-learning characteristics. "
        "The information will be processed by the trained "
        "classification models."
    )

    profile = {}

    # =====================================================
    # BACKGROUND
    # =====================================================

    st.markdown(
        "### 👤 Background"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        profile["Age"] = st.number_input(
            "Age",
            min_value=16,
            max_value=80,
            value=22
        )

    with col2:

        categories = get_categories(
            df,
            "Current Language Level"
        )

        if categories:

            profile[
                "Current Language Level"
            ] = st.selectbox(
                "Current Language Level",
                categories
            )

    with col3:

        categories = get_categories(
            df,
            "Primary Role"
        )

        if categories:

            profile[
                "Primary Role"
            ] = st.selectbox(
                "Primary Role",
                categories
            )

    # =====================================================
    # LEARNING HABITS
    # =====================================================

    st.markdown(
        "### ⏱️ Learning Habits"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        profile[
            "Daily Study Time (Minutes)"
        ] = st.slider(
            "Daily Study Time (Minutes)",
            min_value=0,
            max_value=180,
            value=30,
            step=5
        )

    with col2:

        categories = get_categories(
            df,
            "Weekly Practice Frequency"
        )

        if categories:

            profile[
                "Weekly Practice Frequency"
            ] = st.selectbox(
                "Weekly Practice Frequency",
                categories
            )

    with col3:

        profile[
            "Years Learning English"
        ] = st.slider(
            "Years Learning English",
            min_value=0,
            max_value=30,
            value=3
        )

    # =====================================================
    # LEARNING APPROACH
    # =====================================================

    categories = get_categories(
        df,
        "Primary Learning Approach"
    )

    if categories:

        profile[
            "Primary Learning Approach"
        ] = st.selectbox(
            "Primary Learning Approach",
            categories
        )

    # =====================================================
    # EXPOSURE & USAGE
    # =====================================================

    st.markdown(
        "### 🌐 English Exposure & Usage"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        categories = get_categories(
            df,
            "Daily English Exposure"
        )

        if categories:

            profile[
                "Daily English Exposure"
            ] = st.selectbox(
                "Daily English Exposure",
                categories
            )

    with col2:

        categories = get_categories(
            df,
            "English Usage Frequency"
        )

        if categories:

            profile[
                "English Usage Frequency"
            ] = st.selectbox(
                "English Usage Frequency",
                categories
            )

    with col3:

        categories = get_categories(
            df,
            "English Learning Motivation"
        )

        if categories:

            profile[
                "English Learning Motivation"
            ] = st.selectbox(
                "English Learning Motivation",
                categories
            )

    # =====================================================
    # GOALS & ACTIVITIES
    # =====================================================

    st.markdown(
        "### 🎯 Goals & Practice"
    )

    col1, col2 = st.columns(2)

    with col1:

        categories = get_categories(
            df,
            "Learning Goal"
        )

        if categories:

            profile[
                "Learning Goal"
            ] = st.selectbox(
                "Learning Goal",
                categories
            )

    with col2:

        categories = get_categories(
            df,
            "Preferred Practice Activity"
        )

        if categories:

            profile[
                "Preferred Practice Activity"
            ] = st.selectbox(
                "Preferred Practice Activity",
                categories
            )

    # =====================================================
    # SKILL CONFIDENCE
    # =====================================================

    st.markdown(
        "### 🧠 Current Skill Confidence"
    )

    skill_labels = {
        "Vocabulary Mastery Confidence":
            "📚 Vocabulary Mastery",

        "Grammar & Sentence Structure Confidence":
            "✍️ Grammar & Sentence Structure",

        "Listening Comprehension Confidence":
            "🎧 Listening Comprehension",

        "Speaking Fluency & Oral Expression Confidence":
            "🗣️ Speaking Fluency",

        "Reading Confidence":
            "📖 Reading",

        "Writing Confidence":
            "📝 Writing"
    }

    skill_cols = st.columns(3)

    for i, skill in enumerate(
        SKILL_COLUMNS
    ):

        with skill_cols[i % 3]:

            profile[skill] = st.slider(
                skill_labels[skill],
                min_value=1,
                max_value=5,
                value=3,
                key=f"profile_{skill}"
            )

    return profile