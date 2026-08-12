"""
models.py
---------
Model registry and builder utilities for fraud detection project.

Provides a mapping of model names to their classifier classes and default
parameters, plus a helper to instantiate models with optional overrides.

Do not edit DEFAULT_PARAMS in production without validating behavior.
"""

from lightgbm import LGBMClassifier  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from xgboost import XGBClassifier  # type: ignore

MODEL_REGISTRY = {
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
    "lightgbm": LGBMClassifier
}

DEFAULT_PARAMS = {
    "random_forest":{"n_estimators": 100,"class_weight": "balanced","random_state": 42},
    "xgboost": {"n_estimators": 100, "scale_pos_weight": 2.7,
                "random_state": 42, "eval_metric": "logloss"},
    "lightgbm": {"n_estimators": 100, "class_weight": "balanced", "random_state": 42}
}

def build_model(model_name:str, **overrides):
    """Build a classifier by name with default parameters, allowing overrides.

    Args:
        model_name (str): Key of the model to build. Must be one of the keys in
            MODEL_REGISTRY (e.g. "random_forest", "xgboost", "lightgbm").
        **overrides: Arbitrary keyword arguments to override the default
            parameters defined in DEFAULT_PARAMS for the selected model.

    Returns:
        tuple: (estimator, params) where `estimator` is an instantiated scikit-
            compatible classifier and `params` is the final parameter dict used
            to construct it.

    Raises:
        ValueError: If `model_name` is not present in MODEL_REGISTRY.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not recognized. Options: {list(MODEL_REGISTRY)} ")
    params = {**DEFAULT_PARAMS[model_name], **overrides}
    return MODEL_REGISTRY[model_name](**params), params
