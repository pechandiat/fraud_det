"""Pipeline builder for the fraud detection model."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_pipeline(
    columns_to_onehot: list,
    columns_to_impute: list,
    numerical_rest: list,
    random_state: int = 42,
) -> Pipeline:
    """
    Build a machine learning pipeline.

    Args:
        columns_to_onehot (list): List of categorical columns to one-hot encode.
        columns_to_impute (list): List of numerical columns to impute.
        numerical_rest (list): List of numerical columns to keep unchanged.
        random_state (int): Random state for reproducibility.

    Returns:
        Pipeline: A scikit-learn pipeline.
    """
    # Define the column transformer
    transform = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                columns_to_onehot,
            ),
            ("numerical_imputed", SimpleImputer(strategy="median"), columns_to_impute),
            ("numerical_rest", "passthrough", numerical_rest),
        ],
        remainder="passthrough",  # Keep other columns unchanged
    )

    # Create the pipeline
    pipeline = Pipeline(
        steps=[
            ("transform", transform),
            (
                "model",
                RandomForestClassifier(
                    random_state=random_state, n_estimators=100, class_weight="balanced"
                ),
            ),
        ]
    )

    return pipeline
