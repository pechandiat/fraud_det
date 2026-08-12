"""Register a trained MLflow model in the model registry.

This utility registers a model artifact from a specific MLflow run under a
configured model name so it can be versioned and managed via the MLflow model
registry.
"""

import argparse

import mlflow


def register(run_id: str, model_name: str):
    """Register an MLflow model from a specific run.

    Args:
        run_id: Identifier of the MLflow run that contains the trained model.
        model_name: Name to assign to the registered model in the MLflow model registry.
    """
    result = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_name)
    print(f"Registered model: {result.name}, version {result.version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Registers model from mlflow run"
    )
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="fraud-detection")
    args = parser.parse_args()
    register(args.run_id, args.model_name)
