# export_model.py
"""One-off script: downloads the champion model from MLflow and saves it
locally, so the Gradio demo can be deployed without depending on a live
MLflow server."""

import mlflow
from src.config import MODEL_NAME

mlflow.set_tracking_uri("http://localhost:5000")

model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
mlflow.sklearn.save_model(
    model,
    "gradio_demo/model",
    skops_trusted_types=[
        "numpy.dtype",
        "xgboost.core.Booster",
        "xgboost.sklearn.XGBClassifier",
        "collections.OrderedDict",
        "lightgbm.basic.Booster",
        "lightgbm.sklearn.LGBMClassifier",
    ],
)

print("Model exported to gradio_demo/model/")
