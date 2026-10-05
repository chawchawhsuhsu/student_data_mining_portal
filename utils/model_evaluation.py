import os
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)


# ============================================================
# PROJECT PATHS
# ============================================================

def get_project_root():
    """Find the project root containing the models folder."""

    current = os.path.dirname(os.path.abspath(__file__))

    while True:
        if os.path.exists(os.path.join(current, "models")):
            return current

        parent = os.path.dirname(current)

        if parent == current:
            return os.path.dirname(current)

        current = parent


# ============================================================
# MODEL COMPARISON
# ============================================================

def load_model_comparison(project_root=None):

    if project_root is None:
        project_root = get_project_root()

    possible_paths = [
        os.path.join(project_root, "models", "model_comparison.csv"),
        os.path.join(project_root, "model_comparison.csv"),
    ]

    for path in possible_paths:

        if os.path.exists(path):

            df = pd.read_csv(path)

            return df

    return pd.DataFrame()


# ============================================================
# GB ARTIFACTS
# ============================================================

def load_gb_artifacts(project_root=None):

    if project_root is None:
        project_root = get_project_root()

    model_dir = os.path.join(
        project_root,
        "models"
    )

    artifacts = {}

    files = {
        "model": "gradient_boosting_model.pkl",
        "features": "model_features.pkl",
        "encoders": "feature_encoders.pkl",
        "target_encoder": "target_encoder.pkl"
    }

    for key, filename in files.items():

        path = os.path.join(
            model_dir,
            filename
        )

        if os.path.exists(path):

            try:
                artifacts[key] = joblib.load(path)

            except Exception as e:

                artifacts[key] = None
                print(
                    f"Could not load {filename}: {e}"
                )

        else:

            artifacts[key] = None

    return artifacts


# ============================================================
# BASELINE IMPROVEMENT
# ============================================================

def calculate_baseline_improvement(
    baseline_accuracy,
    final_accuracy
):

    percentage_point_gain = (
        final_accuracy - baseline_accuracy
    )

    relative_gain = (
        percentage_point_gain /
        baseline_accuracy
    ) * 100

    return {
        "percentage_point_gain":
            percentage_point_gain,

        "relative_gain":
            relative_gain
    }


# ============================================================
# MODEL RANKING
# ============================================================

def rank_models(results):

    if results.empty:
        return results

    ranked = results.copy()

    ranked = ranked.sort_values(
        "Accuracy",
        ascending=False
    ).reset_index(drop=True)

    ranked.insert(
        0,
        "Rank",
        range(1, len(ranked) + 1)
    )

    return ranked


# ============================================================
# METRIC CONVERSION
# ============================================================

def prepare_metrics_for_chart(results):

    if results.empty:
        return pd.DataFrame()

    chart_df = results.copy()

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    for col in metric_columns:

        if col in chart_df.columns:

            chart_df[col] = (
                chart_df[col] * 100
            )

    return chart_df


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

def get_target_distribution():

    return pd.DataFrame({
        "Class": [
            "Moderate Progress",
            "Fast Progress",
            "Slow Progress"
        ],
        "Count": [
            867,
            859,
            274
        ]
    })


# ============================================================
# SMOTE DISTRIBUTION
# ============================================================

def get_smote_distribution():

    return pd.DataFrame({

        "Class": [
            "Fast Progress",
            "Moderate Progress",
            "Slow Progress"
        ],

        "Before SMOTE": [
            687,
            694,
            219
        ],

        "After SMOTE": [
            694,
            694,
            694
        ]
    })


# ============================================================
# CONFUSION MATRIX
# ============================================================

def get_gb_confusion_matrix():

    return np.array([
        [113, 46, 13],
        [55, 89, 29],
        [2, 27, 26]
    ])


def get_confusion_matrix_df():

    matrix = get_gb_confusion_matrix()

    labels = [
        "Fast Progress",
        "Moderate Progress",
        "Slow Progress"
    ]

    return pd.DataFrame(
        matrix,
        index=[
            f"Actual: {x}"
            for x in labels
        ],
        columns=[
            f"Predicted: {x}"
            for x in labels
        ]
    )


# ============================================================
# GB CLASSIFICATION REPORT
# ============================================================

def get_gb_classification_report():

    return pd.DataFrame({

        "Class": [
            "Fast Progress",
            "Moderate Progress",
            "Slow Progress"
        ],

        "Precision": [
            0.66,
            0.55,
            0.38
        ],

        "Recall": [
            0.66,
            0.51,
            0.47
        ],

        "F1 Score": [
            0.66,
            0.53,
            0.42
        ],

        "Support": [
            172,
            173,
            55
        ]
    })


# ============================================================
# GB FEATURE IMPORTANCE
# ============================================================

def get_gb_feature_importance():

    return pd.DataFrame({

        "Feature": [
            "Daily Study Time (Minutes)",
            "Weekly Practice Frequency",
            "English Learning Motivation",
            "Listening Comprehension Confidence",
            "English Usage Frequency",
            "Speaking Fluency & Oral Expression Confidence",
            "Daily English Exposure",
            "Age",
            "Grammar & Sentence Structure Confidence",
            "Years Learning English",
            "Primary Learning Approach",
            "Current Language Level",
            "Vocabulary Mastery Confidence",
            "Reading Confidence",
            "Learning Goal"
        ],

        "Importance": [
            0.332420,
            0.177494,
            0.110552,
            0.076256,
            0.043039,
            0.035875,
            0.031365,
            0.029808,
            0.024264,
            0.020384,
            0.019924,
            0.019002,
            0.017720,
            0.017532,
            0.017128
        ]
    })