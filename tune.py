"""
tune.py
Hyperparameter tuning script for fraud detection using Optuna and LightGBM.

Usage: python tune.py

This script loads data, builds a preprocessing and modeling pipeline,
performs cross-validated hyperparameter search with Optuna, logs runs
to MLflow, and saves the best model.
"""

import mlflow
import mlflow.sklearn
import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from src.config import (
    COLUMNS_TO_DROP,
    COLUMNS_TO_FLAG,
    COLUMNS_TO_IMPUTE,
    COLUMNS_TO_NUMBER,
    COLUMNS_TO_ONEHOT,
    DATA_PATH,
    TARGET,
)
from src.data import clean_features, create_null_indicator, load_data, to_numeric
from src.pipeline import build_pipeline

N_TRIALS = 40

df = load_data(DATA_PATH)
df = to_numeric(df, columns_to_number=COLUMNS_TO_NUMBER)
df = clean_features(df, columns_to_drop=COLUMNS_TO_DROP)
df = create_null_indicator(df, columns_to_flag=COLUMNS_TO_FLAG)
x = df.drop(columns=[TARGET])
y = df[TARGET]
numerical_rest = [c for c in x.columns if c not in COLUMNS_TO_ONEHOT + COLUMNS_TO_IMPUTE]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}

mlflow.set_experiment("fraud-detection")


def objective(trial, parent_run_id):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial: Optuna trial object for suggesting hyperparameters.
        parent_run_id: MLflow parent run ID for nested run tracking.

    Returns:
        float: Mean PR-AUC score from cross-validation.
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "class_weight": "balanced",
        "random_state": 42,
    }

    model = LGBMClassifier(**params, verbose=-1)
    pipe = build_pipeline(columns_to_onehot=COLUMNS_TO_ONEHOT,
                          columns_to_impute=COLUMNS_TO_IMPUTE,
                          numerical_rest=numerical_rest,
                          model=model)

    with mlflow.start_run(
        run_name=f"trial_{trial.number}", nested=True, parent_run_id=parent_run_id
    ):
        mlflow.log_params(params)
        cv_results = cross_validate(pipe, x, y, cv=cv, scoring=scoring)
        pr_auc_mean = cv_results["test_pr_auc"].mean()

        for metric in scoring:
            scores = cv_results[f"test_{metric}"]
            mlflow.log_metric(f"{metric}_mean", scores.mean())
            mlflow.log_metric(f"{metric}_std", scores.std())

    return pr_auc_mean


def main():
    """
    Main function that orchestrates hyperparameter tuning and model training.

    Performs Optuna hyperparameter optimization on LightGBM with MLflow tracking,
    logs the best hyperparameters and metrics, trains the best model, and saves it.
    """
    with mlflow.start_run(run_name="lgbm_tuning") as parent_run:
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, parent_run.info.run_id), n_trials=N_TRIALS
        )

        mlflow.log_param("n_trials", N_TRIALS)
        mlflow.log_metric("best_pr_auc", study.best_value)
        for k, v in study.best_params.items():
            mlflow.log_param(f"best_{k}", v)

        print("Best hyperparameters found:")
        print(study.best_params)
        print(f"Best PR-AUC: {study.best_value:.4f}")

        best_model = LGBMClassifier(
            **study.best_params, class_weight="balanced", random_state=42, verbose=-1
        )
        best_pipe = build_pipeline(columns_to_onehot=COLUMNS_TO_ONEHOT,
                                   columns_to_impute=COLUMNS_TO_IMPUTE,
                                   numerical_rest=numerical_rest,
                                   model=best_model)
        best_pipe.fit(x, y)
        mlflow.sklearn.log_model(
            best_pipe,
            "model",
            skops_trusted_types=[
                "collections.OrderedDict",
                "numpy.dtype",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMClassifier",
            ],
        )


if __name__ == "__main__":
    main()
