"""Pipeline builder for the fraud detection model."""

from sklearn.compose import ColumnTransformer  # type: ignore[import]
from sklearn.impute import SimpleImputer  # type: ignore[import]
from sklearn.pipeline import Pipeline  # type: ignore[import]
from sklearn.preprocessing import OneHotEncoder  # type: ignore[import]


def build_pipeline(
    columns_to_onehot: list,
    columns_to_impute: list,
    numerical_rest: list,
    model,
) -> Pipeline:
    """
    Build a machine learning pipeline.

    Args:
        columns_to_onehot (list): List of categorical columns to one-hot encode.
        columns_to_impute (list): List of numerical columns to impute.
        numerical_rest (list): List of numerical columns to keep unchanged.
        model: model to build
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
    pipeline = Pipeline(steps=[("transform", transform), ("model", model)])

    return pipeline
