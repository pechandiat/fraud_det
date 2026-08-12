
"""fraud_det.main

Main entrypoint for training and evaluating the fraud detection model.

This module loads data, applies preprocessing, builds a pipeline, runs
stratified cross-validation, and logs results and the trained model to MLflow.
"""
import argparse
import json

import mlflow
import mlflow.sklearn
from sklearn.model_selection import StratifiedKFold, cross_validate  # type: ignore
from src.data import clean_features, create_null_indicator, load_data, to_numeric
from src.models import build_model
from src.pipeline import build_pipeline

DATA_PATH = "data/Fraud_Dataset.csv"
COLUMNS_TO_DROP = ["K"]
COLUMNS_TO_NUMBER = ["Q", "R", "Monto"]
COLUMNS_TO_FLAG = ["C"]
COLUMNS_TO_ONEHOT = ["J"]
COLUMNS_TO_IMPUTE = ["C"]
TARGET = "Fraude"

def parse_args():
    """Parse command-line arguments for the fraud detection model.

    Returns:
        argparse.Namespace: Parsed arguments containing model type, run name,
                          hyperparameters, and data path.
    """
    parser = argparse.ArgumentParser(description="Train and tack fraud detection model")
    parser.add_argument("--model", type=str, default="random_forest",
                        choices=["random_forest","lightgbm","xgboost"])
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--params", type=str, default='{}',
                        help='Hyperparameters as JSON, example: \'{"n_estimators":200}\'')
    parser.add_argument("--data-path",type=str, default=DATA_PATH)
    return parser.parse_args()

def main():
    """Train and evaluate a Random Forest fraud detection model using cross-validation.

    Loads the fraud dataset, preprocesses features (numeric conversion, cleaning,
    null indicators), builds a pipeline with one-hot encoding and imputation,
    and performs stratified k-fold cross-validation with multiple metrics.
    Logs all results to MLflow.
    """

    args = parse_args()
    param_overrides = json.loads(args.params)
    run_name = args.run_name or f"{args.model}_baseline"

    df = load_data(args.data_path)

    df = to_numeric(df, columns_to_number=COLUMNS_TO_NUMBER)

    df = clean_features(df, columns_to_drop=COLUMNS_TO_DROP)

    df = create_null_indicator(df, columns_to_flag=COLUMNS_TO_FLAG)

    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    numerical_rest = [
        c for c in x.columns if c not in COLUMNS_TO_ONEHOT + COLUMNS_TO_IMPUTE
    ]

    model, final_params = build_model(args.model, **param_overrides)
    pipe = build_pipeline(
        model=model,
        columns_to_onehot=COLUMNS_TO_ONEHOT,
        columns_to_impute=COLUMNS_TO_IMPUTE,
        numerical_rest=numerical_rest
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    mlflow.set_experiment("fraud_detection")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model", args.model)
        for k, v in final_params.items():
            mlflow.log_param(k, v)

        cv_results = cross_validate(pipe, x, y, cv=cv, scoring=scoring)

        for metric in scoring:
            scores = cv_results[f"test_{metric}"]
            mlflow.log_metric(f"{metric}_mean", scores.mean())
            mlflow.log_metric(f"{metric}_std", scores.std())

        pipe.fit(x, y)
        mlflow.sklearn.log_model(
            pipe,
            "model",
            skops_trusted_types=[
                "numpy.dtype",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
                "collections.OrderedDict",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMClassifier",
            ],
        )


if __name__ == "__main__":
    main()
