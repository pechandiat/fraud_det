"""Tests for src/schemas.py"""

import pytest
from pydantic import ValidationError
from src.schemas import Transaction

VALID_PAYLOAD = {
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


def test_transaction_accepts_valid_payload():
    """Test that Transaction accepts a valid payload."""
    t = Transaction(**VALID_PAYLOAD)
    assert t.J == "CO"


def test_transaction_allows_null_c():
    """Test that Transaction allows C field to be None."""
    payload = {**VALID_PAYLOAD, "C": None}
    t = Transaction(**payload)
    assert t.C is None


def test_transaction_rejects_unknown_country_code():
    """Test that Transaction rejects unknown country codes."""
    payload = {**VALID_PAYLOAD, "J": "ZZ"}
    with pytest.raises(ValidationError):
        Transaction(**payload)


def test_transaction_rejects_missing_required_field():
    """Test that Transaction rejects payloads with missing required fields."""
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "A"}
    with pytest.raises(ValidationError):
        Transaction(**payload)
