# Fraud Detection — End-to-End ML/MLOps Project

Complete Machine Learning lifecycle project: from data exploration to a model served via API, containerized, with experiment tracking and versioning through MLflow.

## Table of Contents

- [Fraud Detection — End-to-End ML/MLOps Project](#fraud-detection--end-to-end-mlmlops-project)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [The Path to the Final Dataset](#the-path-to-the-final-dataset)
    - [First Attempt: Synthetic 9-CSV Dataset (Discarded)](#first-attempt-synthetic-9-csv-dataset-discarded)
    - [Final Dataset: `Fraud_Dataset.csv`](#final-dataset-fraud_datasetcsv)
  - [Data Exploration and Cleaning](#data-exploration-and-cleaning)
  - [Modeling](#modeling)
    - [Pipeline](#pipeline)
    - [Model Comparison](#model-comparison)
    - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
  - [Model Registry](#model-registry)
  - [Model Serving: FastAPI](#model-serving-fastapi)
  - [Tests](#tests)
  - [Deployment with Docker](#deployment-with-docker)
    - [Deployment Issues Resolved](#deployment-issues-resolved)
  - [Project Structure](#project-structure)
  - [How to run](#how-to-run)
    - [Prerequisites](#prerequisites)
    - [First-time setup](#first-time-setup)
    - [Option A — Docker Compose (full stack)](#option-a--docker-compose-full-stack)
    - [Option B — Local development (recommended for day-to-day work)](#option-b--local-development-recommended-for-day-to-day-work)
    - [Standalone Gradio demo](#standalone-gradio-demo)
    - [Gradio demo (HuggingFace)](#gradio-demo-huggingface)
    - [Tests](#tests-1)
  - [CI/CD](#cicd)
  - [Common gotchas](#common-gotchas)
  - [Key Learnings](#key-learnings)

## Project Overview

A binary classification model for detecting fraudulent transactions, built following a workflow representative of a real-world MLOps project:

1. Prototyping and validating the idea in a Jupyter Notebook
2. Migrating to reproducible `.py` modules
3. Systematic model comparison (Random Forest, XGBoost, LightGBM)
4. Hyperparameter tuning with Bayesian optimization (Optuna)
5. Tracking all experiments with MLflow
6. Versioning the winning model in the MLflow Model Registry
7. Serving the model through a REST API (FastAPI)
8. Reproducible deployment with Docker Compose (MLflow server + API, in separate containers communicating over a network)

**Final model result (tuned LightGBM):** F1 ≈ 0.62, PR-AUC ≈ 0.70, ROC-AUC ≈ 0.82 (validated with 5-fold stratified cross-validation).

## The Path to the Final Dataset

This project went through two datasets before arriving at the current one — and that decision, along with its diagnosis, is itself part of the project's value.

### First Attempt: Synthetic 9-CSV Dataset (Discarded)

The first dataset combined 9 CSV files (customer profiles, fraud indicators, suspicious activity, merchant information, transaction amounts, and transaction metadata). After building the complete pipeline and training a baseline model, the diagnosis revealed that **the dataset had no real learnable signal**:

- The predicted probabilities were practically identical across the fraud/non-fraud classes
- `feature_importances_` were evenly distributed, with no dominant variable
- ROC-AUC fell below 0.5 (worse than random) after setting `class_weight="balanced"`

Conclusion: the fraud label in that synthetic dataset had no real causal relationship with the other columns — a common pattern in synthetic datasets generated without sufficient care to simulate fraud.

### Final Dataset: `Fraud_Dataset.csv`

The dataset from a previous fraud project was reused — anonymized columns (`A`, `B`, `C`, ...) plus `Monto` and the `Fraude` target. Unlike the previous dataset, this one showed real and significant correlations with the target in the initial correlation matrix.

- **16,880 rows**, with a target containing **~27% fraud** (much more balanced than the original dataset, which was around 4.5%)
- `Monto`, `Q`, and `R` required conversion from text to numeric
- Column `K`: **76% null values**, with no business context to confidently impute → discarded
- Column `C`: 19% nulls, left-skewed distribution → imputed with the median, and an indicator column `C_is_null` was added to preserve the signal that the value was missing
- Column `J` (country code, 19 unique values): encoded with `OneHotEncoder`

## Data Exploration and Cleaning

The complete prototyping work (EDA, correlation matrix, target distribution, null-value diagnosis, feature engineering) lives in `notebooks/prototyping.ipynb`. The validated cleaning logic was migrated to `src/data.py`:

- `load_data()` — loads the CSV
- `to_numeric()` — converts text columns to numeric (`Monto`, `Q`, `R`)
- `clean_features()` — removes discarded columns (`K`)
- `create_null_indicator()` — generates `*_is_null` columns before imputation

## Modeling

### Pipeline

Defined in `src/pipeline.py` with `sklearn.Pipeline` + `ColumnTransformer`:

- Median imputation for `C`
- One-hot encoding for `J`
- Passthrough for the remaining numeric columns

### Model Comparison

Three models were compared with `class_weight="balanced"` (or its equivalent), validated with `StratifiedKFold` (5 folds), using **PR-AUC** as the primary decision metric (more appropriate than ROC-AUC or accuracy given the class imbalance):

| Model | F1 | PR-AUC | Precision | CV Duration |
|---|---|---|---|---|
| Random Forest (balanced) | 0.602 | 0.672 | 0.602 | 9.7s |
| XGBoost (baseline) | 0.613 | 0.692 | 0.572 | 3.8min |
| XGBoost (tuned) | 0.617 | 0.693 | 0.555 | 1.2min |
| **LightGBM (baseline)** | **0.622** | **0.701** | 0.546 | **2.4s** |

**LightGBM won on F1 and PR-AUC while also being dramatically faster** — it was selected as the base model for tuning.

### Hyperparameter Tuning

Bayesian search with **Optuna** (40 trials), directly optimizing PR-AUC, with each trial logged as a nested run in MLflow (`tune.py`). Best result: **PR-AUC = 0.7027** — an improvement of only +0.0017 over the untuned baseline, within the model's own statistical noise margin (std ±0.013).

> **Honest note:** Hyperparameter tuning did not significantly improve the model. This indicates that LightGBM's defaults were already close to the optimum achievable with this feature set, and that the actual improvement ceiling (~0.70 PR-AUC) is limited by the information available in the data, not by the model. Since the columns are anonymous (`A`, `B`, `C`...) and have no interpretable business meaning, additional feature engineering was not attempted — doing so would have meant guessing combinations without a conceptual basis, with a real risk of introducing noise or overfitting to the specific dataset.

Best hyperparameters found:
```text
n_estimators=443, num_leaves=59, max_depth=10, learning_rate=0.033,
min_child_samples=17, subsample=0.71, colsample_bytree=0.84
```

## Experiment Tracking with MLflow

All training runs (baselines for the 3 models and the 40 tuning trials) are registered in MLflow: parameters, CV metrics (mean and standard deviation per metric), and the serialized model as an artifact.

- `main.py` — trains an individual model, configurable through the CLI:
  ```bash
  uv run python main.py --model lightgbm --run-name lgbm_baseline
  uv run python main.py --model xgboost --params '{"n_estimators": 300}'
  ```
- `tune.py` — runs the Optuna search over LightGBM (winner of model comparison), logging each trial as a nested run under a parent run

## Model Registry

The winning model (tuned LightGBM) was registered in the MLflow Model Registry under the name `fraud-detection-lgbm`, with the `champion` alias pointing to the active version. Using an alias (instead of a fixed version number) makes it possible to promote a new model without having to modify or redeploy the code that consumes it.

```bash
uv run python register_model.py --run-id <RUN_ID> --model-name fraud-detection-lgbm
```

## Model Serving: FastAPI

`app.py` exposes the model as a minimal REST API:

- `GET /health` — health check
- `POST /predict` — receives transaction data (validated with Pydantic in `src/schemas.py`, including ranges observed during training as non-restrictive documentation, and a closed list of valid country codes) and returns `is_fraud` + `fraud_probability`

The model is loaded at startup directly from the Model Registry via alias:

```python
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
```

Interactive documentation is available at `/docs` (Swagger UI).

## Tests

```bash
uv add --dev pytest httpx2
uv run pytest test/ -v
```

Cubre: preprocesamiento de datos, el pipeline de sklearn, el factory de modelos, la validación de schemas con Pydantic, y los endpoints de la API (con el modelo de MLflow mockeado, sin depender de que el servidor esté corriendo).

## Deployment with Docker

Two independent services, orchestrated with `docker-compose.yml`:

- **`mlflow-server`** — tracking server (SQLite + artifact store with HTTP proxy), exposed on `:5000`
- **`api`** — FastAPI API, exposed on `:8000`, connected to `mlflow-server` through the internal Docker Compose network (`MLFLOW_TRACKING_URI=http://mlflow-server:5000`)

Both containers use `uv` for dependency management, with Docker layers optimized for caching (dependencies are installed before copying the application code).

### Deployment Issues Resolved

Documented here because they are representative of real infrastructure bugs, not just issues specific to this project:

| Issue | Cause | Solution |
|---|---|---|
| `PermissionError: /app` when logging the model | `--artifacts-destination` pointed to a local filesystem path inside the container; the client attempted to write there directly | Remove the flag and use MLflow's default artifact proxy |
| `Invalid Host header — possible DNS rebinding attack` | MLflow security middleware (≥3.5) rejected the `mlflow-server` hostname | `--allowed-hosts "localhost:*,mlflow-server:*"` (port wildcard required) |
| `OSError: libgomp.so.1: cannot open shared object file` | LightGBM depends on OpenMP (a native C library), which was missing from the minimal base image | `apt-get install -y libgomp1` in the API `Dockerfile` |

## Project Structure

```
fraud_det/
├── app.py                  # FastAPI application
├── main.py                 # Training CLI (parametrized by model)
├── tune.py                 # Optuna hyperparameter tuning + MLflow logging
├── register_model.py       # Programmatic Model Registry registration
├── src/
│   ├── config.py           # Shared constants (columns, paths, model name)
│   ├── data.py              # Data loading and cleaning utilities
│   ├── pipeline.py          # sklearn ColumnTransformer + Pipeline builder
│   ├── models.py            # Model factory (registry + defaults)
│   └── schemas.py           # Pydantic request/response contracts for the API
├── test/
│   ├── conftest.py          # Shared pytest fixtures
│   ├── test_data.py
│   ├── test_pipeline.py
│   ├── test_models.py
│   ├── test_schemas.py
│   └── test_api.py
├── notebooks/
│   └── prototyping.ipynb   # Original prototyping: EDA, dataset diagnosis, baseline
├── data/
│   └── Fraud_Dataset.csv
├── Dockerfile               # API image
├── Dockerfile.mlflow        # MLflow tracking server image
├── docker-compose.yml       # Orchestrates both services
├── pyproject.toml / uv.lock # Dependencies (managed with uv)
├── .gitignore                # Excludes mlruns/, mlflow.db, __pycache__/, .venv/
└── README.md
```
## How to run

### Prerequisites

- [`uv`](https://docs.astral.sh/uv/) installed
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed,
  **open and running** — WSL2 users specifically need the WSL integration
  enabled (Docker Desktop → Settings → Resources → WSL Integration → enable
  for your distro)

### First-time setup

```bash
uv sync
```

### Option A — Docker Compose (full stack)

```bash
docker compose up -d
```

- MLflow UI: http://localhost:5000
- API (Swagger): http://localhost:8000/docs

To stop (keeps all data — experiments, registered models):
```bash
docker compose down
```
To stop **and wipe all MLflow data** (start fully clean):
```bash
docker compose down -v
```

### Option B — Local development (recommended for day-to-day work)

Keep MLflow as the single source of truth running in Docker, and iterate on
code locally with fast reload — much quicker than rebuilding the API image
on every change.

```bash
# 1. Start only the MLflow server (leave it running in the background)
docker compose up -d mlflow-server

# 2. Point your local scripts to it — required in EVERY new terminal
#    session, it does not persist across sessions
export MLFLOW_TRACKING_URI=http://localhost:5000

# 3. Verify it's actually set before training/tuning anything
echo $MLFLOW_TRACKING_URI
```

> ⚠️ If `MLFLOW_TRACKING_URI` is empty, `main.py`/`tune.py` will silently
> fall back to a local `./mlruns` folder instead of the Dockerized server —
> the script will run without errors, but the run won't show up in the
> MLflow UI at `localhost:5000`. This is the most common source of "where
> did my experiment go?" confusion when resuming work after a break.

```bash
# Train a single model, logged to MLflow
uv run python main.py --model lightgbm --run-name lgbm_baseline

# Hyperparameter tuning with Optuna
uv run python tune.py

# Register the best run's model
uv run python register_model.py --run-id <RUN_ID> --model-name fraud-detection-lgbm
# then set the "champion" alias via the MLflow UI, or:
python -c "
import mlflow
client = mlflow.MlflowClient()
client.set_registered_model_alias(name='fraud-detection-lgbm', alias='champion', version='<VERSION>')
"

# Run the API with hot-reload
uv run uvicorn app:app --reload
```

### Standalone Gradio demo

Self-contained — bundles its own exported copy of the model, so it doesn't
depend on the MLflow server or the API being up.

```bash
# One-time export (requires mlflow-server running + MLFLOW_TRACKING_URI set,
# see Option B above)
uv run python export_model.py

uv run python gradio_demo/app.py
```

### Gradio demo (HuggingFace)

A standalone, publicly-hosted demo, separate from the production API:

**Live at:** https://huggingface.co/spaces/pechandiat/fraud-detection

Design choice: the demo bundles its own exported copy of the model
(`gradio_demo/model/`, produced by `export_model.py`) instead of calling the
live API or MLflow server. This keeps it fully self-contained and
deployable to a free public host without exposing any local infrastructure
— the standard approach for ML demos on Hugging Face Spaces.

The Space runs on the free ZeroGPU tier (free-tier accounts can no longer
select CPU-only hardware directly), so `app.py`'s prediction function is
decorated with `@spaces.GPU` purely to satisfy that platform requirement —
the model itself still runs on CPU internally.

To update the demo with a newer model version:
```bash
# 1. Promote the new model to the "champion" alias in MLflow (see Model Registry above)
# 2. Re-export it
export MLFLOW_TRACKING_URI=http://localhost:5000
uv run python export_model.py
# 3. Commit gradio_demo/model/ and push — CD handles the rest
git add gradio_demo/model/
git commit -m "Update demo model"
git push origin master
```


### Tests

```bash
uv run pytest test/ -v
```

Covers data preprocessing, the sklearn pipeline, the model factory, API
request/response schemas, and the API endpoints themselves (with the MLflow
model mocked out — no live server required to run the test suite).

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`), triggered on every push/PR to
`master`:

| Job | What it does |
|---|---|
| `test` | Installs dependencies with `uv` and runs the full `pytest` suite |
| `docker-build` | Builds both Docker images (API and MLflow server), gated on `test` passing |
| `deploy-demo` | On pushes to `master` only (not PRs): uploads `gradio_demo/` to the Hugging Face Space via the `hf` CLI, gated on `test` passing |

`docker-build` and `deploy-demo` both run in parallel once `test` succeeds —
neither depends on the other.

Required repo secret: `HF_TOKEN` (a Hugging Face access token with write
permissions, used by `deploy-demo`).

## Common gotchas

| Symptom                                                                      | Cause                                                                        | Fix                                                                                         |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `docker: command not found` in WSL                                           | Docker Desktop isn't open, or WSL integration is disabled                    | Open Docker Desktop; check Settings → Resources → WSL Integration                           |
| Experiment doesn't show up in MLflow UI after running `main.py`/`tune.py`    | `MLFLOW_TRACKING_URI` not set in the current terminal session                | `export MLFLOW_TRACKING_URI=http://localhost:5000` (needed in every new terminal)           |
| `PermissionError` when logging a model to the Dockerized MLflow server       | Server misconfigured with `--artifacts-destination` pointing to a local path | Let the server use its default proxied artifact serving (no `--artifacts-destination` flag) |
| `Invalid Host header` from the MLflow server                                 | Security middleware rejecting the caller's hostname                          | `--allowed-hosts "localhost:*,mlflow-server:*"`                                             |
| `OSError: libgomp.so.1: cannot open shared object file` in the API container | LightGBM needs OpenMP, missing from the slim base image                      | `apt-get install -y libgomp1` in the API `Dockerfile`                                       |

## Key Learnings

- **Diagnosing a lack of signal in a dataset is just as valuable as training a good model** — it prevents investing effort in tuning/feature engineering on a foundation that has nothing to learn.
- **PR-AUC, not accuracy or ROC-AUC alone, is the right criterion for choosing between models with imbalanced classes.**
- **Hyperparameter tuning has diminishing returns** — it is useful for squeezing out the margin provided by the model, not for compensating for a lack of information in the data.
- **Recognize when an adjustment (e.g. `class_weight`) improves actual model discrimination vs. when it only shifts the precision/recall trade-off** (ROC-AUC/PR-AUC barely changing ⇒ the model did not learn more, it only moved the decision threshold).
- **Preprocessing code must be shared literally between training and inference** (`src/data.py` is used by both `main.py` and `app.py`) to avoid *training-serving skew*.

