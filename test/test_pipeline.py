"""Tests for src/pipeline.py"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from src.pipeline import build_pipeline


def test_pipeline_fits_and_predicts(sample_df: pd.DataFrame):
    """Test that pipeline fits the model and produces predictions."""
    x = sample_df.drop(columns=["Fraude", "K"])
    x["Monto"] = x["Monto"].astype(float)
    x["Q"] = x["Q"].astype(float)
    x["R"] = x["R"].astype(float)
    y = sample_df["Fraude"]

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    pipe = build_pipeline(
        columns_to_onehot=["J"],
        columns_to_impute=["C"],
        numerical_rest=["A", "B", "D", "Monto", "Q", "R"],
        model=model,
    )

    pipe.fit(x, y)
    predictions = pipe.predict(x)

    assert len(predictions) == len(x)
    assert set(predictions).issubset({0, 1})


def test_imputer_fills_null_with_median(sample_df: pd.DataFrame):
    """Test that imputer fills null values with the median."""
    x = sample_df.drop(columns=["Fraude", "K"])
    x["Monto"] = x["Monto"].astype(float)
    x["Q"] = x["Q"].astype(float)
    x["R"] = x["R"].astype(float)

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    pipe = build_pipeline(
        columns_to_onehot=["J"],
        columns_to_impute=["C"],
        numerical_rest=["A", "B", "D", "Monto", "Q", "R"],
        model=model,
    )
    pipe.fit(x, sample_df["Fraude"])

    imputer = pipe.named_steps["transform"].named_transformers_["numerical_imputed"]
    expected_median = x["C"].median()

    assert imputer.statistics_[0] == expected_median
