"""FastAPI application serving the fraud detection model from the MLflow registry."""

import os

import mlflow
import pandas as pd  # type: ignore
from fastapi import FastAPI
from src.config import (
    COLUMNS_TO_FLAG,
    COLUMNS_TO_NUMBER,
    MODEL_ALIAS,
    MODEL_NAME,
    MODEL_VERSION,
)
from src.data import create_null_indicator, to_numeric
from src.schemas import PredictionResponse, Transaction

app = FastAPI(title="Fraud Detection API")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")


@app.get("/health")
def health():
    """Check API health and model status.

    Returns:
        Dictionary with status, model name, alias, and model version.
    """
    return {"status": "ok", "model": MODEL_NAME, "alias": MODEL_ALIAS, "model_version":MODEL_VERSION}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    """Predict fraud probability for a transaction.

    Args:
        transaction: Transaction object containing transaction details.

    Returns:
        PredictionResponse with is_fraud flag and fraud_probability.
    """
    df = pd.DataFrame([transaction.model_dump()])

    df = to_numeric(df, columns_to_number=COLUMNS_TO_NUMBER)
    df = create_null_indicator(df, columns_to_flag=COLUMNS_TO_FLAG)

    proba = model.predict_proba(df)[0, 1]
    return PredictionResponse(
        is_fraud=bool(proba >= 0.5),
        fraud_probability=float(proba),
    )

