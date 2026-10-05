from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILES = {
    "Decision Tree": "decision_tree_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "Gradient Boosting": "gradient_boosting_model.pkl",
}

FEATURES_FILE = "model_features.pkl"
ENCODERS_FILE = "feature_encoders.pkl"
TARGET_ENCODER_FILE = "target_encoder.pkl"
RESULTS_FILE = "model_comparison.csv"


# ============================================================
# PROJECT PATH
# ============================================================

def get_project_root():

    current = Path(__file__).resolve()

    for parent in current.parents:

        if (parent / "models").is_dir():
            return parent

    return current.parent.parent


PROJECT_ROOT = get_project_root()

MODEL_DIR = PROJECT_ROOT / "models"


# ============================================================
# FIND FILE
# ============================================================

def find_file(filename):

    locations = [
        MODEL_DIR / filename,
        PROJECT_ROOT / filename,
        PROJECT_ROOT / "model" / filename,
    ]

    for path in locations:

        if path.exists():
            return path

    return None


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

@st.cache_resource
def load_trained_models(model_dir=None):

    models = {}

    for model_name, filename in MODEL_FILES.items():

        # ----------------------------------------------------
        # First check supplied model directory
        # ----------------------------------------------------

        if model_dir is not None:

            path = Path(model_dir) / filename

            if not path.exists():
                path = find_file(filename)

        else:

            path = find_file(filename)

        if path is None:
            continue

        try:

            models[model_name] = joblib.load(path)

        except Exception as e:

            st.warning(
                f"Could not load {model_name}: {e}"
            )

    return models


# ============================================================
# LOAD MODEL RESULTS
# ============================================================

@st.cache_data
def load_model_results(project_dir=None):

    possible_paths = []

    if project_dir is not None:

        project_dir = Path(project_dir)

        possible_paths.extend([
            project_dir / "models" / RESULTS_FILE,
            project_dir / RESULTS_FILE,
            project_dir / "model" / RESULTS_FILE,
        ])

    possible_paths.extend([
        MODEL_DIR / RESULTS_FILE,
        PROJECT_ROOT / RESULTS_FILE,
        PROJECT_ROOT / "model" / RESULTS_FILE,
    ])

    checked = set()

    for path in possible_paths:

        path = Path(path)

        if str(path) in checked:
            continue

        checked.add(str(path))

        if path.exists():

            try:
                return pd.read_csv(path)

            except Exception as e:

                st.warning(
                    f"Could not read model comparison file: {e}"
                )

                return pd.DataFrame()

    return pd.DataFrame()


# ============================================================
# LOAD FEATURE INFORMATION
# ============================================================

@st.cache_resource
def load_feature_information():

    features_path = find_file(
        FEATURES_FILE
    )

    encoders_path = find_file(
        ENCODERS_FILE
    )

    target_encoder_path = find_file(
        TARGET_ENCODER_FILE
    )

    features = None
    encoders = {}
    target_encoder = None

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    if features_path is not None:

        try:
            features = joblib.load(
                features_path
            )

        except Exception as e:

            st.warning(
                f"Could not load model features: {e}"
            )


    # --------------------------------------------------------
    # Feature encoders
    # --------------------------------------------------------

    if encoders_path is not None:

        try:
            encoders = joblib.load(
                encoders_path
            )

        except Exception as e:

            st.warning(
                f"Could not load feature encoders: {e}"
            )


    # --------------------------------------------------------
    # Target encoder
    # --------------------------------------------------------

    if target_encoder_path is not None:

        try:

            target_encoder = joblib.load(
                target_encoder_path
            )

        except Exception as e:

            st.warning(
                f"Could not load target encoder: {e}"
            )

    return (
        features,
        encoders,
        target_encoder
    )


# ============================================================
# ENCODE LEARNER PROFILE
# ============================================================

def encode_profile(profile):

    (
        feature_names,
        encoders,
        _
    ) = load_feature_information()

    if feature_names is None:

        raise ValueError(
            "model_features.pkl could not be loaded."
        )

    encoded_profile = {}

    # --------------------------------------------------------
    # Process every feature in the SAME order used during
    # model training
    # --------------------------------------------------------

    for feature in feature_names:

        if feature not in profile:

            raise ValueError(
                f"Missing profile feature: {feature}"
            )

        value = profile[feature]

        # ----------------------------------------------------
        # Categorical feature
        # ----------------------------------------------------

        if feature in encoders:

            encoder = encoders[feature]

            value = str(value)

            # ------------------------------------------------
            # Check whether category existed during training
            # ------------------------------------------------

            if value not in encoder.classes_:

                available = ", ".join(
                    map(
                        str,
                        encoder.classes_
                    )
                )

                raise ValueError(
                    f"Unknown value '{value}' "
                    f"for '{feature}'. "
                    f"Expected one of: {available}"
                )

            encoded_profile[feature] = (
                encoder.transform([value])[0]
            )

        # ----------------------------------------------------
        # Numeric feature
        # ----------------------------------------------------

        else:

            try:

                encoded_profile[feature] = (
                    float(value)
                )

            except (ValueError, TypeError):

                raise ValueError(
                    f"Feature '{feature}' must be numeric. "
                    f"Received: {value}"
                )

    return pd.DataFrame(
        [encoded_profile],
        columns=feature_names
    )


# ============================================================
# PREDICT ONE MODEL
# ============================================================

def predict_model(
    model,
    profile
):

    try:

        # ----------------------------------------------------
        # Convert Streamlit profile into the EXACT numeric
        # representation used during training
        # ----------------------------------------------------

        encoded_profile = encode_profile(
            profile
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction_encoded = model.predict(
            encoded_profile
        )[0]

        # ----------------------------------------------------
        # Convert numeric target back to original label
        # ----------------------------------------------------

        (
            _,
            _,
            target_encoder
        ) = load_feature_information()

        if target_encoder is not None:

            prediction = (
                target_encoder
                .inverse_transform(
                    [prediction_encoded]
                )[0]
            )

        else:

            prediction = str(
                prediction_encoded
            )

        # ----------------------------------------------------
        # Probabilities
        # ----------------------------------------------------

        probability_df = pd.DataFrame()

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = (
                model.predict_proba(
                    encoded_profile
                )[0]
            )

            model_classes = model.classes_

            if target_encoder is not None:

                classes = (
                    target_encoder
                    .inverse_transform(
                        model_classes
                    )
                )

            else:

                classes = model_classes

            probability_df = pd.DataFrame({
                "Track": classes,
                "Probability": probabilities
            }).sort_values(
                "Probability",
                ascending=False
            )

        return (
            prediction,
            probability_df
        )

    except Exception as e:

        return (
            f"Error: {e}",
            pd.DataFrame()
        )


# ============================================================
# PREDICT ALL MODELS
# ============================================================

def predict_all_models(
    models,
    profile
):

    predictions = {}

    probability_tables = {}

    for model_name, model in models.items():

        prediction, probability_df = (
            predict_model(
                model,
                profile
            )
        )

        predictions[model_name] = prediction

        probability_tables[
            model_name
        ] = probability_df

    return (
        predictions,
        probability_tables
    )


# ============================================================
# MAJORITY VOTE
# ============================================================

def get_majority_prediction(
    predictions
):

    if not predictions:

        return (
            "No prediction available",
            0,
            0
        )

    values = list(
        predictions.values()
    )

    vote_counts = (
        pd.Series(values)
        .value_counts()
    )

    final_prediction = (
        vote_counts.index[0]
    )

    vote_count = int(
        vote_counts.iloc[0]
    )

    total_models = len(values)

    return (
        final_prediction,
        vote_count,
        total_models
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(model):

    if not hasattr(
        model,
        "feature_importances_"
    ):

        return pd.DataFrame(
            columns=[
                "Feature",
                "Importance"
            ]
        )

    importance = (
        model.feature_importances_
    )

    (
        feature_names,
        _,
        _
    ) = load_feature_information()

    if feature_names is None:

        feature_names = [
            f"Feature {i + 1}"
            for i in range(
                len(importance)
            )
        ]

    if len(feature_names) != len(
        importance
    ):

        feature_names = [
            f"Feature {i + 1}"
            for i in range(
                len(importance)
            )
        ]

    return (
        pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        })
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )


# ============================================================
# MODEL FEATURES
# ============================================================

def get_model_features():

    (
        feature_names,
        _,
        _
    ) = load_feature_information()

    return feature_names


# ============================================================
# DEBUG INFORMATION
# ============================================================

def get_artifact_status():

    files = [
        "decision_tree_model.pkl",
        "random_forest_model.pkl",
        "gradient_boosting_model.pkl",
        "model_features.pkl",
        "feature_encoders.pkl",
        "target_encoder.pkl",
        "model_comparison.csv",
    ]

    rows = []

    for filename in files:

        path = find_file(filename)

        rows.append({
            "File": filename,
            "Found": path is not None,
            "Path": (
                str(path)
                if path
                else "Not found"
            )
        })

    return pd.DataFrame(rows)