import os

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_data

from utils.learner_profile import (
    create_learner_profile
)

from utils.model_engine import (
    load_trained_models,
    load_model_results,
    predict_all_models,
    get_majority_prediction,
    get_feature_importance
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Predictive Engine - English Learning Data Mining Portal",
    page_icon="🔮",
    layout="wide"
)


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    BASE_DIR
)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)


# =========================================================
# LOAD DATA
# =========================================================

df = load_data()

if df is None or df.empty:

    st.error(
        "The English learning dataset could not be loaded."
    )

    st.stop()


# =========================================================
# LOAD TRAINED MODELS
# =========================================================

@st.cache_resource
def get_models():

    return load_trained_models(
        MODEL_DIR
    )


@st.cache_data
def get_results():

    return load_model_results(
        PROJECT_DIR
    )


models = get_models()

model_results = get_results()


# =========================================================
# HEADER
# =========================================================

st.title(
    "🔮 Predictive Engine"
)

st.caption(
    "Use the trained machine-learning models to "
    "estimate the learner's expected learning "
    "progress track."
)

st.divider()


# =========================================================
# MODEL STATUS
# =========================================================

if not models:

    st.error(
        "❌ No trained models were found."
    )

    st.write(
        "The application expected these files:"
    )

    st.code(
        """
models/
├── decision_tree_model.pkl
├── random_forest_model.pkl
└── gradient_boosting_model.pkl
        """
    )

    st.write(
        f"Current model directory:"
    )

    st.code(
        MODEL_DIR
    )

    st.stop()


st.success(
    f"✅ Loaded {len(models)} trained model(s): "
    f"{', '.join(models.keys())}"
)


# =========================================================
# LEARNER PROFILE
# =========================================================

st.divider()

st.header(
    "1. 👤 Build Your Learner Profile"
)

profile = create_learner_profile(
    df
)


# =========================================================
# ANALYZE
# =========================================================

analyze = st.button(
    "🔮 Analyze My Learning Profile",
    type="primary",
    use_container_width=True
)


if analyze:

    # =====================================================
    # PREDICTION
    # =====================================================

    st.divider()

    st.header(
        "2. 🤖 Machine Learning Prediction"
    )

    predictions, probability_tables = (
        predict_all_models(
            models,
            profile
        )
    )

    valid_predictions = {
        name: prediction
        for name, prediction
        in predictions.items()
        if not str(prediction).startswith(
            "Error"
        )
    }

    if not valid_predictions:

        st.error(
            "The trained models could not process "
            "this learner profile."
        )

        for name, prediction in predictions.items():

            st.write(
                f"**{name}:** {prediction}"
            )

        st.stop()


    # =====================================================
    # FINAL PREDICTION
    # =====================================================

    (
        final_prediction,
        vote_count,
        total_models
    ) = get_majority_prediction(
        valid_predictions
    )


    st.success(
        f"### 🎯 Predicted Learning Progress Track\n\n"
        f"**{final_prediction}**"
    )


    # =====================================================
    # MODEL AGREEMENT
    # =====================================================

    if total_models == 3:

        if vote_count == 3:

            st.success(
                "✅ All three trained models agree "
                "on this prediction."
            )

        elif vote_count == 2:

            st.info(
                "ℹ️ Two of the three trained models "
                "agree on this prediction."
            )

        else:

            st.warning(
                "⚠️ The three models produced "
                "different predictions."
            )

    else:

        st.info(
            f"The prediction is based on "
            f"{total_models} available trained model(s)."
        )


    # =====================================================
    # MODEL PREDICTIONS
    # =====================================================

    st.subheader(
        "🔍 Individual Model Predictions"
    )

    prediction_df = pd.DataFrame({
        "Model": list(
            predictions.keys()
        ),
        "Prediction": list(
            predictions.values()
        )
    })

    st.dataframe(
        prediction_df,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # PROBABILITIES
    # =====================================================

    st.subheader(
        "📊 Prediction Probabilities"
    )

    available_probability_models = [
        name
        for name, table
        in probability_tables.items()
        if not table.empty
    ]

    if available_probability_models:

        tabs = st.tabs(
            available_probability_models
        )

        for tab, model_name in zip(
            tabs,
            available_probability_models
        ):

            with tab:

                probability_df = (
                    probability_tables[
                        model_name
                    ].copy()
                )

                display_df = (
                    probability_df.copy()
                )

                display_df[
                    "Probability (%)"
                ] = (
                    display_df[
                        "Probability"
                    ] * 100
                ).round(2)

                display_df = display_df[
                    [
                        "Track",
                        "Probability (%)"
                    ]
                ]

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

                fig = px.bar(
                    display_df,
                    x="Track",
                    y="Probability (%)",
                    range_y=[0, 100],
                    title=(
                        f"{model_name} "
                        "Prediction Probability"
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


    # =====================================================
    # MODEL PERFORMANCE
    # =====================================================

    st.divider()

    st.header(
        "3. 📈 Trained Model Performance"
    )

    if model_results.empty:

        st.warning(
            "model_comparison.csv was not found."
        )

        st.code(
            os.path.join(
                MODEL_DIR,
                "model_comparison.csv"
            )
        )

    else:

        performance_display = (
            model_results.copy()
        )

        percentage_columns = [
            "Accuracy",
            "Precision",
            "Recall",
            "Weighted F1",
            "Macro F1",
            "F1"
        ]

        for column in percentage_columns:

            if column in performance_display.columns:

                performance_display[column] = (
                    performance_display[column]
                    * 100
                ).round(2)

        st.dataframe(
            performance_display,
            use_container_width=True,
            hide_index=True
        )

        # -------------------------------------------------
        # FIND BEST MODEL
        # -------------------------------------------------

        f1_column = None

        for candidate in [
            "Weighted F1",
            "Macro F1",
            "F1"
        ]:

            if candidate in model_results.columns:

                f1_column = candidate
                break

        if f1_column:

            best_row = model_results.loc[
                model_results[
                    f1_column
                ].idxmax()
            ]

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "🏆 Best Model",
                    best_row["Model"]
                )

            with c2:

                st.metric(
                    f"Best {f1_column}",
                    f"{best_row[f1_column]:.2%}"
                )

            if "Accuracy" in best_row:

                with c3:

                    st.metric(
                        "Best Accuracy",
                        f"{best_row['Accuracy']:.2%}"
                    )


        # -------------------------------------------------
        # PERFORMANCE CHART
        # -------------------------------------------------

        chart_columns = [
            column
            for column in [
                "Accuracy",
                "Precision",
                "Recall",
                "Weighted F1"
            ]
            if column in model_results.columns
        ]

        if chart_columns:

            chart_df = model_results.melt(
                id_vars="Model",
                value_vars=chart_columns,
                var_name="Metric",
                value_name="Score"
            )

            fig = px.bar(
                chart_df,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                range_y=[0, 1],
                title="Trained Model Comparison"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.divider()

    st.header(
        "4. 🔎 Important Model Features"
    )

    selected_model = st.selectbox(
        "Select trained model",
        list(models.keys())
    )

    importance_df = get_feature_importance(
        models[selected_model]
    )

    if importance_df.empty:

        st.info(
            "Feature importance is not available "
            "for this saved model."
        )

    else:

        top_n = min(
            15,
            len(importance_df)
        )

        top_features = (
            importance_df
            .head(top_n)
            .sort_values(
                "Importance",
                ascending=True
            )
        )

        fig = px.bar(
            top_features,
            x="Importance",
            y="Feature",
            orientation="h",
            title=(
                f"Top {top_n} Features — "
                f"{selected_model}"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.caption(
            "Feature importance describes the contribution "
            "of features to the trained model's decisions. "
            "It does not prove causation."
        )


    # =====================================================
    # SKILL PROFILE
    # =====================================================

    st.divider()

    st.header(
        "5. 📊 Your Skill Profile"
    )

    skill_names = [
        "Vocabulary",
        "Grammar",
        "Listening",
        "Speaking",
        "Reading",
        "Writing"
    ]

    skill_scores = [
        profile[
            "Vocabulary Mastery Confidence"
        ],

        profile[
            "Grammar & Sentence Structure Confidence"
        ],

        profile[
            "Listening Comprehension Confidence"
        ],

        profile[
            "Speaking Fluency & Oral Expression Confidence"
        ],

        profile[
            "Reading Confidence"
        ],

        profile[
            "Writing Confidence"
        ]
    ]

    skill_df = pd.DataFrame({
        "Skill": skill_names,
        "Confidence": skill_scores
    })

    strongest = skill_df[
        "Confidence"
    ].idxmax()

    weakest = skill_df[
        "Confidence"
    ].idxmin()

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "💪 Strongest Area",
            skill_df.loc[
                strongest,
                "Skill"
            ],
            f"{skill_df.loc[strongest, 'Confidence']}/5"
        )

    with c2:

        st.metric(
            "🎯 Priority Development Area",
            skill_df.loc[
                weakest,
                "Skill"
            ],
            f"{skill_df.loc[weakest, 'Confidence']}/5"
        )


    # -----------------------------------------------------
    # BAR CHART
    # -----------------------------------------------------

    fig = px.bar(
        skill_df,
        x="Skill",
        y="Confidence",
        range_y=[0, 5],
        title="Current English Skill Confidence"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # =====================================================
    # RADAR + DATASET COMPARISON
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        radar = go.Figure()

        radar.add_trace(
            go.Scatterpolar(
                r=skill_scores + [
                    skill_scores[0]
                ],
                theta=skill_names + [
                    skill_names[0]
                ],
                fill="toself",
                name="Your Profile"
            )
        )

        radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5]
                )
            ),
            title="Learner Skill Profile",
            height=420
        )

        st.plotly_chart(
            radar,
            use_container_width=True
        )

    with col2:

        mapping = {
            "Vocabulary":
                "Vocabulary Mastery Confidence",

            "Grammar":
                "Grammar & Sentence Structure Confidence",

            "Listening":
                "Listening Comprehension Confidence",

            "Speaking":
                "Speaking Fluency & Oral Expression Confidence",

            "Reading":
                "Reading Confidence",

            "Writing":
                "Writing Confidence"
        }

        dataset_scores = []

        for skill in skill_names:

            column = mapping[skill]

            if column in df.columns:

                value = pd.to_numeric(
                    df[column],
                    errors="coerce"
                ).mean()

                dataset_scores.append(
                    round(value, 2)
                )

            else:

                dataset_scores.append(0)

        comparison_df = pd.DataFrame({
            "Skill": (
                skill_names +
                skill_names
            ),

            "Score": (
                skill_scores +
                dataset_scores
            ),

            "Profile": (
                ["Your Profile"] * 6 +
                ["Dataset Average"] * 6
            )
        })

        fig = px.bar(
            comparison_df,
            x="Skill",
            y="Score",
            color="Profile",
            barmode="group",
            range_y=[0, 5],
            title="Your Profile vs Dataset Average"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # =====================================================
    # LEARNER TYPE
    # =====================================================

    st.divider()

    st.header(
        "6. 🧠 Learner Characteristics"
    )

    average_skill = (
        sum(skill_scores) /
        len(skill_scores)
    )

    speaking = profile[
        "Speaking Fluency & Oral Expression Confidence"
    ]

    listening = profile[
        "Listening Comprehension Confidence"
    ]

    grammar = profile[
        "Grammar & Sentence Structure Confidence"
    ]

    reading = profile[
        "Reading Confidence"
    ]

    study_time = profile[
        "Daily Study Time (Minutes)"
    ]

    if average_skill >= 4:

        learner_type = (
            "High-Confidence Learner"
        )

    elif (
        study_time >= 60
        and speaking <= 2
    ):

        learner_type = (
            "High-Study / "
            "Communication-Developing Learner"
        )

    elif (
        speaking >= 4
        and listening >= 4
    ):

        learner_type = (
            "Communication-Oriented Learner"
        )

    elif (
        grammar >= 4
        and reading >= 4
        and speaking <= 3
    ):

        learner_type = (
            "Knowledge-Strong / "
            "Communication-Developing Learner"
        )

    elif average_skill <= 2.5:

        learner_type = (
            "Foundation-Building Learner"
        )

    else:

        learner_type = (
            "Developing Balanced Learner"
        )

    st.success(
        f"### 🎯 Identified Learner Type: "
        f"{learner_type}"
    )

    st.caption(
        "This is a rule-based descriptive profile. "
        "It is separate from the machine-learning prediction."
    )


    # =====================================================
    # HIDDEN PATTERNS
    # =====================================================

    st.subheader(
        "🔍 Patterns Detected"
    )

    patterns = []

    exposure = profile.get(
        "Daily English Exposure",
        ""
    )

    usage = profile.get(
        "English Usage Frequency",
        ""
    )

    if (
        str(exposure).lower()
        in ["high_exposure", "high exposure"]
        and str(usage).lower()
        in ["often", "daily"]
    ):

        patterns.append(
            (
                "🌐 Strong English Exposure",
                "Your reported exposure and usage "
                "provide frequent opportunities "
                "to encounter English."
            )
        )

    if (
        study_time >= 60
        and speaking <= 2
    ):

        patterns.append(
            (
                "🗣️ Study–Speaking Gap",
                "You report substantial study time "
                "while speaking confidence is "
                "comparatively lower."
            )
        )

    if (
        listening >= 4
        and speaking >= 4
    ):

        patterns.append(
            (
                "🎧🗣️ Strong Communication Pair",
                "Listening and speaking are both "
                "among your stronger reported skills."
            )
        )

    if (
        grammar >= 4
        and reading >= 4
        and speaking <= 3
    ):

        patterns.append(
            (
                "📖 Knowledge–Communication Gap",
                "Reported language knowledge is "
                "stronger than spoken communication confidence."
            )
        )

    vocabulary = profile[
        "Vocabulary Mastery Confidence"
    ]

    writing = profile[
        "Writing Confidence"
    ]

    if (
        vocabulary >= 4
        and writing >= 4
    ):

        patterns.append(
            (
                "📚✍️ Language Production Strength",
                "Vocabulary and writing are both "
                "reported as strong areas."
            )
        )

    if (
        study_time <= 30
        and speaking >= 4
    ):

        patterns.append(
            (
                "⚡ Efficient Communication Profile",
                "You report strong speaking confidence "
                "despite a shorter daily study period."
            )
        )

    practice = str(
        profile.get(
            "Weekly Practice Frequency",
            ""
        )
    ).lower()

    if practice in [
        "every day",
        "everyday",
        "daily",
        "7 days"
    ]:

        patterns.append(
            (
                "🔄 High Practice Consistency",
                "Frequent practice provides regular "
                "opportunities for reinforcement."
            )
        )

    if patterns:

        for title, description in patterns:

            with st.expander(
                title,
                expanded=True
            ):

                st.write(
                    description
                )

    else:

        st.info(
            "No strong predefined behavioral "
            "pattern was detected."
        )

    st.caption(
        "These are rule-based descriptive patterns. "
        "They do not establish causal relationships."
    )


    # =====================================================
    # PERSONALIZED STRATEGY
    # =====================================================

    st.divider()

    st.header(
        "7. 💡 Personalized Improvement Strategy"
    )

    recommendations = []

    if speaking <= 2:

        recommendations.append(
            (
                "🗣️ Speaking",
                "High",
                "Make speaking a primary development "
                "area. Begin with short speaking sessions "
                "using familiar topics."
            )
        )

    elif speaking == 3:

        recommendations.append(
            (
                "🗣️ Speaking",
                "Medium",
                "Increase active speaking practice through "
                "short presentations and conversations."
            )
        )

    else:

        recommendations.append(
            (
                "🗣️ Speaking",
                "Maintain",
                "Maintain your speaking strength through "
                "spontaneous discussions and unfamiliar topics."
            )
        )

    if vocabulary <= 2:

        recommendations.append(
            (
                "📚 Vocabulary",
                "High",
                "Build vocabulary through context and "
                "reuse new words in your own sentences."
            )
        )

    elif vocabulary == 3:

        recommendations.append(
            (
                "📚 Vocabulary",
                "Medium",
                "Expand active vocabulary using collocations "
                "and common expressions."
            )
        )

    if grammar <= 2:

        recommendations.append(
            (
                "✍️ Grammar",
                "High",
                "Practice practical grammar through short "
                "writing and speaking activities."
            )
        )

    elif grammar == 3:

        recommendations.append(
            (
                "✍️ Grammar",
                "Medium",
                "Practice grammar through writing and speaking."
            )
        )

    if listening <= 2:

        recommendations.append(
            (
                "🎧 Listening",
                "High",
                "Use short English audio clips with repeated "
                "listening and identify key expressions."
            )
        )

    elif listening == 3:

        recommendations.append(
            (
                "🎧 Listening",
                "Medium",
                "Mix educational content with natural "
                "English conversations."
            )
        )

    if reading <= 2:

        recommendations.append(
            (
                "📖 Reading",
                "High",
                "Begin with short texts appropriate to "
                "your current level."
            )
        )

    if writing <= 2:

        recommendations.append(
            (
                "📝 Writing",
                "High",
                "Write short paragraphs regularly and "
                "review grammar and sentence structure."
            )
        )

    if study_time < 30:

        recommendations.append(
            (
                "⏱️ Study Routine",
                "High",
                "Build consistency first with short "
                "regular study sessions."
            )
        )

    elif study_time < 60:

        recommendations.append(
            (
                "⏱️ Study Routine",
                "Medium",
                "Use structured sessions combining "
                "input and active practice."
            )
        )

    else:

        recommendations.append(
            (
                "⏱️ Study Routine",
                "Maintain",
                "Focus on learning quality and active "
                "practice rather than simply increasing hours."
            )
        )

    if str(exposure).lower() in [
        "low_exposure",
        "low exposure"
    ]:

        recommendations.append(
            (
                "🌐 English Exposure",
                "High",
                "Increase everyday English exposure "
                "through articles, videos, podcasts, "
                "and conversations."
            )
        )

    goal = profile.get(
        "Learning Goal",
        ""
    )

    goal_advice = {

        "Basic Communication":
            "Prioritize everyday expressions, listening, "
            "and conversational practice.",

        "Academic Success":
            "Prioritize academic vocabulary, reading, "
            "structured writing, and note-taking.",

        "Career Development":
            "Practice professional vocabulary, presentations, "
            "interviews, and workplace communication.",

        "Fluency":
            "Increase spontaneous speaking and natural "
            "English listening.",

        "Exam Preparation":
            "Combine timed practice, grammar review, "
            "reading strategies, and practice tests.",

        "Personal Development":
            "Use topics you genuinely enjoy to create "
            "a sustainable learning routine."
    }

    if goal:

        recommendations.append(
            (
                "🎯 Goal Strategy",
                "Personalized",
                goal_advice.get(
                    goal,
                    "Align your practice with your "
                    "main learning goal."
                )
            )
        )

    high_priority = [
        item
        for item in recommendations
        if item[1] == "High"
    ]

    if high_priority:

        st.subheader(
            "🚨 Priority Actions"
        )

        for category, priority, advice in high_priority:

            st.warning(
                f"**{category}**\n\n"
                f"{advice}"
            )

    st.subheader(
        "📋 Complete Strategy"
    )

    for category, priority, advice in recommendations:

        if priority != "High":

            with st.expander(
                f"{category} — {priority}"
            ):

                st.write(
                    advice
                )


    # =====================================================
    # WEEKLY PLAN
    # =====================================================

    st.divider()

    st.header(
        "8. 📅 Suggested Weekly Focus"
    )

    priority_skill = skill_names[
        weakest
    ]

    weekly_plan = {

        "Vocabulary": [
            "Learn useful vocabulary in context.",
            "Create sentences using the new vocabulary.",
            "Review previously learned words."
        ],

        "Grammar": [
            "Review one grammar pattern.",
            "Write example sentences.",
            "Use the pattern during speaking practice."
        ],

        "Listening": [
            "Listen to a short English clip.",
            "Replay it and identify key expressions.",
            "Summarize what you heard."
        ],

        "Speaking": [
            "Speak about a familiar topic.",
            "Prepare a short response.",
            "Practice spontaneous conversation."
        ],

        "Reading": [
            "Read a short English article.",
            "Identify the main idea.",
            "Collect useful vocabulary."
        ],

        "Writing": [
            "Write a short paragraph.",
            "Review grammar and structure.",
            "Rewrite after correcting mistakes."
        ]
    }

    for i, activity in enumerate(
        weekly_plan[priority_skill],
        1
    ):

        st.markdown(
            f"**Day {i}:** {activity}"
        )

    st.info(
        f"Your weekly focus prioritizes "
        f"**{priority_skill}** because it is "
        f"your lowest-confidence reported skill."
    )


    # =====================================================
    # TECHNICAL EXPLANATION
    # =====================================================

    st.divider()

    with st.expander(
        "🔬 Technical Interpretation"
    ):

        st.markdown(
            """
            ### Machine Learning Methodology

            The Predictive Engine uses three trained
            supervised classification models:

            **1. Decision Tree**

            Used as an interpretable baseline classifier.

            **2. Random Forest**

            Used as an ensemble alternative that combines
            multiple decision trees.

            **3. Gradient Boosting**

            Used to evaluate whether sequential boosting
            can improve classification performance.

            The models were trained externally and saved as
            model artifacts. The Streamlit application loads
            these trained models rather than retraining them.

            The learner's profile is passed directly into the
            same trained model pipeline.

            The final displayed prediction uses majority
            voting among the available trained classifiers.

            The learner characteristics and hidden patterns
            are rule-based interpretations and are separate
            from the machine-learning prediction.
            """
        )


    # =====================================================
    # LIMITATIONS
    # =====================================================

    with st.expander(
        "⚠️ Model Limitations"
    ):

        st.write(
            "The predicted learning track is a model-based "
            "classification and should not be interpreted as "
            "a guaranteed future outcome."
        )

        st.write(
            "Some learner characteristics are self-reported, "
            "so confidence ratings may not exactly represent "
            "objective language proficiency."
        )

        st.write(
            "Model performance depends on the quality, size, "
            "class distribution, and representativeness of "
            "the training dataset."
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64748b;
        font-size:0.85rem;
    ">
        English Learning Data Mining Portal ·
        Predictive & Prescriptive Analysis
    </div>
    """,
    unsafe_allow_html=True
)