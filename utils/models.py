import streamlit as st

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier
)

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold
)

from sklearn.tree import DecisionTreeClassifier


def create_baseline(preprocessor):
    """Create the previous Decision Tree."""

    return ImbPipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=10,
                    random_state=42
                )
            )
        ]
    )


def create_model_pipelines(preprocessor):

    dt_pipeline = ImbPipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "smote",
                SMOTE(
                    random_state=42,
                    k_neighbors=5
                )
            ),
            (
                "classifier",
                DecisionTreeClassifier(
                    random_state=42
                )
            )
        ]
    )

    rf_pipeline = ImbPipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "smote",
                SMOTE(
                    random_state=42,
                    k_neighbors=5
                )
            ),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    gb_pipeline = ImbPipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "smote",
                SMOTE(
                    random_state=42,
                    k_neighbors=5
                )
            ),
            (
                "classifier",
                GradientBoostingClassifier(
                    random_state=42
                )
            )
        ]
    )

    return {
        "Decision Tree": dt_pipeline,
        "Random Forest": rf_pipeline,
        "Gradient Boosting": gb_pipeline
    }


def get_parameter_grids():

    return {

        "Decision Tree": {

            "classifier__max_depth": [
                3, 5, 7, 9, 12
            ],

            "classifier__min_samples_split": [
                2, 5, 10, 20
            ],

            "classifier__min_samples_leaf": [
                1, 2, 5, 10
            ],

            "classifier__criterion": [
                "gini",
                "entropy"
            ]
        },

        "Random Forest": {

            "classifier__n_estimators": [
                100, 200
            ],

            "classifier__max_depth": [
                5, 8, 12, None
            ],

            "classifier__min_samples_split": [
                2, 5, 10
            ],

            "classifier__min_samples_leaf": [
                1, 2, 4
            ],

            "classifier__max_features": [
                "sqrt",
                "log2"
            ]
        },

        "Gradient Boosting": {

            "classifier__n_estimators": [
                50, 100, 150
            ],

            "classifier__learning_rate": [
                0.03, 0.05, 0.10
            ],

            "classifier__max_depth": [
                2, 3, 4
            ],

            "classifier__min_samples_split": [
                2, 5, 10
            ],

            "classifier__min_samples_leaf": [
                1, 2, 5
            ]
        }
    }


@st.cache_resource
def optimize_models(
    X_train,
    y_train,
    preprocessor
):
    """Run GridSearchCV for all models."""

    pipelines = create_model_pipelines(
        preprocessor
    )

    parameter_grids = get_parameter_grids()

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    results = {}

    for name, pipeline in pipelines.items():

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grids[name],
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            verbose=0
        )

        grid.fit(
            X_train,
            y_train
        )

        results[name] = grid

    return results