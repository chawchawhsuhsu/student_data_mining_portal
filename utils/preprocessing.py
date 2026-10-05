import pandas as pd

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def remove_identifier_columns(df):
    """Remove timestamp and ID-like columns."""

    columns_to_remove = []

    for column in df.columns:
        name = column.lower()

        if (
            "timestamp" in name
            or name == "id"
            or name.endswith("_id")
        ):
            columns_to_remove.append(column)

    return df.drop(
        columns=columns_to_remove,
        errors="ignore"
    )


def prepare_features(df, target_col):
    """Separate X/y and identify feature types."""

    model_df = remove_identifier_columns(df)

    X = model_df.drop(columns=[target_col])
    y = model_df[target_col].astype(str)

    numeric_features = X.select_dtypes(
        include=[
            "int64",
            "float64",
            "int32",
            "float32"
        ]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()

    return (
        X,
        y,
        numeric_features,
        categorical_features
    )


def create_preprocessor(
    numeric_features,
    categorical_features
):
    """Create reusable preprocessing pipeline."""

    numeric_transformer = SimpleImputer(
        strategy="median"
    )

    categorical_transformer = ImbPipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                numeric_features
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_features
            )
        ],
        remainder="drop"
    )