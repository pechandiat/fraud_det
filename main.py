import mlflow
import mlflow.sklearn
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from src.data import clean_features, create_null_indicator, load_data, to_numeric
from src.pipeline import build_pipeline

DATA_PATH = "data/Fraud_Dataset.csv"
COLUMNS_TO_DROP = ['K']
COLUMNS_TO_NUMBER = ['Q','R','Monto']
COLUMNS_TO_FLAG = ['C']
COLUMNS_TO_ONEHOT = ['J']
COLUMNS_TO_IMPUTE = ['C']
TARGET = 'Fraude'

mlflow.set_experiment("fraud_detection")

def main():

    df = load_data(path=DATA_PATH)

    df = to_numeric(df, columns_to_number=COLUMNS_TO_NUMBER)

    df = clean_features(df, columns_to_drop=COLUMNS_TO_DROP)

    df = create_null_indicator(df, columns_to_flag=COLUMNS_TO_FLAG)

    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    numerical_rest = [c for c in x.columns if c not in COLUMNS_TO_ONEHOT + COLUMNS_TO_IMPUTE]

    pipe = build_pipeline(columns_to_onehot=COLUMNS_TO_ONEHOT, columns_to_impute=COLUMNS_TO_IMPUTE, numerical_rest=numerical_rest)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    with mlflow.start_run(run_name = "rf_baseline_balanced"):
        mlflow.log_param("model","RandomForestClassification")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("n_estimators", 100)

        cv_results = cross_validate(pipe, x, y, cv=cv, scoring=scoring)

        for metric in scoring:
            scores = cv_results[f"test_{metric}"]
            mlflow.log_metric(f"{metric}_mean", scores.mean())
            mlflow.log_metric(f"{metric}_std", scores.std())

        pipe.fit(x, y)
        mlflow.sklearn.log_model(
            pipe, "model",
            skops_trusted_types=["numpy.dtype"]
            )

if __name__ == "__main__":
    main()
