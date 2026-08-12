# src/config.py
"""Shared configuration constants for the fraud detection project."""

DATA_PATH = "data/Fraud_Dataset.csv"
TARGET = "Fraude"

COLUMNS_TO_DROP = ["K"]
COLUMNS_TO_NUMBER = ["Q", "R", "Monto"]
COLUMNS_TO_FLAG = ["C"]
COLUMNS_TO_ONEHOT = ["J"]
COLUMNS_TO_IMPUTE = ["C"]

MODEL_NAME = "fraud-detection-lgbm"
