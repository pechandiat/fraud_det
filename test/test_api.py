"""Tests for app.py """

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Fixture that provides a TestClient with mocked ML model for testing."""
    with patch("mlflow.sklearn.load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        mock_load.return_value = mock_model

        import app

        yield TestClient(app.app)


def test_health_endpoint(client):
    """Test that the health endpoint returns status 200 and ok status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_valid_response(client):
    """Test that the predict endpoint returns valid fraud prediction response."""
    payload = {
        "A": 15.0,
        "B": 5.0,
        "C": 120.5,
        "D": 30.0,
        "E": 10.0,
        "F": 0.0,
        "G": 1.0,
        "H": 5.0,
        "I": 8.0,
        "J": "CO",
        "L": 2.0,
        "M": 5.0,
        "N": 3.0,
        "O": 1.0,
        "P": 10.0,
        "S": 25.0,
        "Monto": 150.0,
        "Q": 100.0,
        "R": 90.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["is_fraud"] is True
    assert body["fraud_probability"] == 0.8


def test_predict_rejects_invalid_country(client):
    """Test that the predict endpoint rejects invalid country codes with 422 status."""
    payload = {
        "A": 15.0,
        "B": 5.0,
        "C": 120.5,
        "D": 30.0,
        "E": 10.0,
        "F": 0.0,
        "G": 1.0,
        "H": 5.0,
        "I": 8.0,
        "J": "ZZ",
        "L": 2.0,
        "M": 5.0,
        "N": 3.0,
        "O": 1.0,
        "P": 10.0,
        "S": 25.0,
        "Monto": 150.0,
        "Q": 100.0,
        "R": 90.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
